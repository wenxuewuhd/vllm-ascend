# SPDX-License-Identifier: Apache-2.0
"""The slot provider: "chunk k needs experts [e...]" -> a ``ChunkSlot``.

Between ``stream_prefill.py`` (which experts travel together) and
``stream_chain.py`` (compute) sits the I/O: host gather into a *pinned* staging
buffer (``weight_source.WeightSource``), H2D (four copies, one per weight
group), then the production post-load hook, whose outputs are the slot's four
tensors. The cast is never re-derived here: ``npu_format_cast``'s arguments, the
transposes and the scale reshape live only in
``AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading``, which
this module calls on a chunk-sized module, and the device buffer *shapes* come
from the same method (``get_weight`` / ``get_dynamic_quant_param``).

The **ND device buffers** (the H2D destination) are allocated ONCE and reused by
every chunk of every layer; plain ND, so ``Tensor.copy_`` works -- the
restriction that refuses ordinary writes is on *internal-format* destinations.
The **NZ output of the cast is per chunk**, freed when the caller drops the
previous slot: ``npu_format_cast`` returns a new tensor, no API casts into an
existing buffer, and reusing one NZ buffer would mean cast-into-a-temp plus
``torch_npu.copy_memory_(dst.data, src.data)`` -- the only write channel this
build accepts for internal format, since ``Tensor.copy_``, offset views and
``index_select(out=)`` are refused and ``copy_memory_`` ignores the view range
and writes the whole storage. The temp is allocated either way: no HBM saved,
one D2D copy added, and the stable address it would buy only matters for graph
replay, which streaming excludes (``stream_prefill.assert_eager_only``). The
**host** side is the ping-pong: two pinned buffers, so chunk k+1's gather
overlaps chunk k's H2D; copies are submitted non-blocking and a pinned buffer is
not refilled until the event of the H2D that read it has fired.

Two traps only the card can see: the post-hook tensors are NOT reported as the
format they were cast to (the hook is *given* 29 and ``get_npu_format`` returns
**50**, ``FRACTAL_NZ_C0_16``, so a criterion that pins 29 goes red on correct
code); and a **short last chunk** hands the hook ``buffer[:n]`` -- contiguous, at
storage offset 0, but still a view, and this build refuses offset views on
internal-format destinations (the cast reads ND and writes a fresh tensor, so it
is expected to be fine, but only the card can say so).

``tests/ut/kt_offload/test_stream_io.py`` pins the boundary. Stale bytes -- chunk
k served chunk k-1's weights -- stays plausible on-card, so it gets two
independent detectors here: a per-buffer stamp and an optional byte probe.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.kt_offload.stream_chain import SLOT_WEIGHT_ATTRS, ChunkSlot

#: Of the four slot parameters, the two that go through the NZ cast and the two
#: that do not. ``SLOT_WEIGHT_ATTRS`` holds all four, which is the right set to
#: hand ``format_nd`` but the wrong set to raw-copy into: a scale is a plain ND
#: view of the buffer passed in, so writing it needs an ordinary ``copy_``.
NZ_WEIGHT_ATTRS = ("w13_weight", "w2_weight")
ND_SCALE_ATTRS = ("w13_weight_scale", "w2_weight_scale")
from vllm_ascend.kt_offload.weight_source import (
    DEFAULT_COPY_THREADS,
    GROUPS,
    ChunkLayout,
    GatherResult,
    WeightSource,
    ZeroCoverageError,
)

__all__ = [
    "PARAM_FOR_GROUP",
    "DEFAULT_HOST_BUFFERS",
    "StreamIOError",
    "DeviceSeam",
    "SlotCoverage",
    "StreamSlotProvider",
    "ServedNZChunk",
    "DEFAULT_SLOT_COPY_THREADS",
    "chunk_param_shapes",
    "layout_signature",
    "default_alloc_pinned",
    "default_alloc_device",
    "default_h2d",
    "default_wait",
    "default_post_load",
]

#: ``weight_source`` group name -> the production parameter it fills. Group names
#: are the checkpoint side (w13 = w1 then w3 per expert, a plain byte concat);
#: parameter names are the ones ``process_weights_after_loading`` rewrites
#: (``SLOT_WEIGHT_ATTRS``). Explicit rather than derived, so a rename on either
#: side fails a check instead of silently filling the wrong buffer.
PARAM_FOR_GROUP: dict[str, str] = {
    "w13_weight": "w13_weight",
    "w13_scale": "w13_weight_scale",
    "w2_weight": "w2_weight",
    "w2_scale": "w2_weight_scale",
}

#: Host-side pinned buffers. Two = one being filled while one is being copied
#: out; the device side stays at one slot.
DEFAULT_HOST_BUFFERS = 2

#: mmap -> pinned staging threads, from ``weight_source.DEFAULT_COPY_THREADS``.
#: Measured on this host: it is the peak of the read leg and it balances the H2D
#: floor. The reference implementation's default is ``KT_MXFP4_COPY_THREADS=32``,
#: tuned on a 40-core single-NUMA box; here 32 is *slower* and more is worse
#: still. Do not "restore" 32 without re-running that bench on this machine.
DEFAULT_SLOT_COPY_THREADS = DEFAULT_COPY_THREADS


class StreamIOError(RuntimeError):
    """The slot handed back would not be the chunk's weights (stale, unformatted, mis-sized)."""


# ---------------------------------------------------------------------------
# The four device-touching steps, behind one injectable seam
# ---------------------------------------------------------------------------


def default_alloc_pinned(nbytes: int) -> torch.Tensor:
    """Host staging: page-locked uint8.

    A pageable source is several times slower on this box, and only a pinned
    source can be copied ``non_blocking``.
    """
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def default_alloc_device(shape: Sequence[int], dtype: torch.dtype, device: Any) -> torch.Tensor:
    """The reused ND slot. Allocated once per provider; see the module docstring."""
    return torch.empty(tuple(shape), dtype=dtype, device=device)


def default_h2d(dst: torch.Tensor, src: torch.Tensor, *, what: str) -> Any:
    """Submit one group's copy and hand back something to wait on.

    ``non_blocking=True`` returns long before the bytes have moved, which is what
    lets the copy overlap the next gather; the returned event is what keeps
    refilling the pinned source safe. Off-card there is no stream, so the handle
    is ``None``.
    """
    dst.copy_(src, non_blocking=True)
    if dst.device.type == "npu":
        import torch_npu  # noqa: F401  registers torch.npu; imported lazily

        event = torch.npu.Event()
        event.record()
        return event
    return None


def default_wait(handle: Any) -> None:
    if handle is not None:
        handle.synchronize()


def default_post_load(quant_method: Any, module: torch.nn.Module) -> None:
    """Run the PRODUCTION post-load hook on the chunk module.

    ``AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading`` owns
    the cast arguments (``npu_format_cast(..., 29,
    customize_dtype=float8_e4m3fn, input_dtype=float4_e2m1fn_x2)``), the weight
    transpose and the scale reshape/transpose.

    kt PATCHES that method on the layer's quant_method
    (``layer.attach_after_init``), wrapping it in ``zero_unused_slots`` and
    ``after_weights_loaded``. Those two belong to the layer's own weights and
    must NEVER run on a chunk or on the folded swap's staging:
    ``after_weights_loaded`` rebinds the live routing tables of the layer in
    flight to a throwaway module and reloads its entire CPU half (the wrapper
    registry refuses the second wrapper only after that damage). So this takes
    the hook kt wrapped, not kt's wrapper; unpatched methods (every other
    backend) carry no stamp and are used as they are.
    """
    hook = quant_method.process_weights_after_loading
    hook = getattr(hook, "_kt_offload_original", hook)
    hook(module)


@dataclass(frozen=True)
class DeviceSeam:
    """The four device-touching steps, injectable so the boundary can be modelled.

    Defaults are the real thing; the unit tests replace all four with recorders,
    since a host cannot pin memory, copy to a device or hold internal format.
    """

    alloc_pinned: Callable[[int], torch.Tensor] = default_alloc_pinned
    alloc_device: Callable[..., torch.Tensor] = default_alloc_device
    h2d: Callable[..., Any] = default_h2d
    wait: Callable[[Any], None] = default_wait
    post_load: Callable[[Any, torch.nn.Module], None] = default_post_load


# ---------------------------------------------------------------------------
# Shapes: asked of the production method, never written out here
# ---------------------------------------------------------------------------


def moe_scheme(quant_method: Any, *required: str) -> Any:
    """The ``AscendMoEScheme`` behind a layer's ``quant_method``, checked.

    Production hands the layer's ``quant_method``, which is
    ``AscendFusedMoEMethod`` -- a WRAPPER; what the streaming chain needs
    (``get_weight``, ``quant_type``, ``act_quant_type``) is on the scheme it
    holds (``method_adapters.py``: ``self.quant_method = scheme``; e.g.
    ``AscendW4A8MXFPDynamicFusedMoEMethod`` in ``w4a8_mxfp4.py``). The post-load
    hook is the other way round -- on the WRAPPER, and it is the wrapper kt
    patches (``default_post_load`` unwraps that patch) -- so **the two objects
    must not be collapsed into one**.

    ``required`` names the attributes the caller is about to use, checked here
    at once. Unwrapping is one explicit level, not a ``getattr`` fallback chain:
    a fallback would also accept an object carrying an inner ``quant_method``
    for an unrelated reason, and a wrong shape or quant type fails much later,
    as a buffer the GEMM quietly rejects.
    """
    if not required:
        raise ValueError("moe_scheme needs the attribute names the caller is about to use; "
                         "without them it cannot tell a scheme from a wrapper")
    if all(hasattr(quant_method, a) for a in required):
        return quant_method
    inner = getattr(quant_method, "quant_method", None)
    if inner is not None and all(hasattr(inner, a) for a in required):
        return inner
    missing_outer = [a for a in required if not hasattr(quant_method, a)]
    missing_inner = ("n/a" if inner is None
                     else [a for a in required if not hasattr(inner, a)])
    raise StreamIOError(
        f"neither {type(quant_method).__name__} (missing {missing_outer}) nor its inner quant_method "
        f"({type(inner).__name__ if inner is not None else 'absent'}, missing {missing_inner}) carries "
        f"{list(required)}; the streaming chain reads these off the scheme the model loader used, and "
        "there is no second source for them")


def shape_declaring_method(quant_method: Any) -> Any:
    """The object that declares a layer's parameter shapes: the scheme inside the
    layer's ``quant_method``, not the wrapper it is. See ``moe_scheme``.
    """
    return moe_scheme(quant_method, "get_weight", "get_dynamic_quant_param")


def chunk_param_shapes(quant_method: Any, chunk_experts: int, intermediate_size: int, hidden_size: int,
                       params_dtype: torch.dtype = torch.bfloat16) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``name -> (shape, dtype)`` for a chunk-sized layer, from the quant method itself.

    ``get_weight`` / ``get_dynamic_quant_param`` are what the model loader asks
    for the real layer; asking them for ``chunk_experts`` rows is the only
    difference. The empty tensors they return are never kept.
    """
    if chunk_experts <= 0:
        raise ValueError(f"chunk_experts must be > 0, got {chunk_experts}")
    decl = shape_declaring_method(quant_method)
    shapes = dict(decl.get_weight(chunk_experts, intermediate_size, hidden_size, params_dtype))
    shapes.update(decl.get_dynamic_quant_param(chunk_experts, intermediate_size, hidden_size, params_dtype))
    missing = [n for n in SLOT_WEIGHT_ATTRS if n not in shapes]
    if missing:
        raise StreamIOError(f"{type(decl).__name__} does not declare {missing}; the streaming slot "
                            f"is built from the parameters the post-load hook rewrites ({list(SLOT_WEIGHT_ATTRS)})")
    out: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for name in SLOT_WEIGHT_ATTRS:
        t = shapes[name]
        if t.dtype != torch.uint8:
            # The DS method declares the E8M0 scales uint8 so the load is a byte
            # move: into a typed E8M0 buffer the copy becomes a value conversion
            # that flattens every exponent, with no error -- the same trap
            # weight_source refuses to walk into.
            raise StreamIOError(f"{name} is declared {t.dtype}, not uint8: the streaming path moves BYTES, "
                                "and a non-uint8 destination turns the copy into a value conversion")
        if t.dim() != 3 or int(t.shape[0]) != chunk_experts:
            raise StreamIOError(f"{name} shape {tuple(t.shape)} is not [chunk_experts={chunk_experts}, n, k]")
        out[name] = (tuple(int(s) for s in t.shape), t.dtype)
    return out


def _check_group_map() -> None:
    """The two name spaces must still line up; a rename on either side fails here."""
    group_names = {g for g, _ in GROUPS}
    if set(PARAM_FOR_GROUP) != group_names:
        raise StreamIOError(f"weight_source groups {sorted(group_names)} do not match the staging map "
                            f"{sorted(PARAM_FOR_GROUP)}")
    if set(PARAM_FOR_GROUP.values()) != set(SLOT_WEIGHT_ATTRS):
        raise StreamIOError(f"staging map fills {sorted(PARAM_FOR_GROUP.values())} but the chunk slot is "
                            f"{sorted(SLOT_WEIGHT_ATTRS)}")


# ---------------------------------------------------------------------------
# The chunk-sized module the production hook runs on
# ---------------------------------------------------------------------------


class _ChunkModule(torch.nn.Module):
    """Carries exactly the four weights ``process_weights_after_loading`` rewrites.

    The parameters *alias* the provider's persistent ND buffers rather than being
    them: the hook assigns ``layer.w13_weight.data = <cast output>``, so were the
    attribute the buffer itself, the reused ND slot would be repointed at the
    per-chunk NZ tensor and every later chunk would cast whatever the last cast
    left. A ``Parameter`` shares the storage but is its own tensor object.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        super().__init__()
        for name in SLOT_WEIGHT_ATTRS:
            self.register_parameter(name, torch.nn.Parameter(tensors[name], requires_grad=False))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@dataclass
class SlotCoverage:
    """What the provider actually moved. Zero is a failure, never a pass.

    A path that gathered and served nothing produces the same output as
    "streaming is off" (the reference implementation falls back to hybrid
    silently in several places); this accounting is what tells them apart.
    """

    chunks_served: int = 0
    gathers: int = 0
    bytes_gathered: int = 0
    bytes_to_device: int = 0
    h2d_calls: int = 0
    pinned_buffers: int = 0
    pinned_bytes: int = 0
    expert_slots_served: int = 0          # counts multiplicity across layers/chunks
    experts_produced: set[tuple[int, int]] = field(default_factory=set)   # (layer, expert)
    layers_served: set[int] = field(default_factory=set)

    def summary(self) -> str:
        return (f"stream_io coverage: chunks={self.chunks_served} experts={len(self.experts_produced)} "
                f"layers={len(self.layers_served)} gathered={self.bytes_gathered} B "
                f"to_device={self.bytes_to_device} B in {self.h2d_calls} copies "
                f"pinned={self.pinned_buffers}x{self.pinned_bytes} B")

    def violations(self, expected: Iterable[tuple[int, int]] | None = None,
                   *, bytes_per_expert: int | None = None) -> list[str]:
        bad: list[str] = []
        if self.chunks_served == 0:
            bad.append("no chunk was served: zero coverage is a failure, not a pass")
        if self.bytes_gathered == 0:
            bad.append("no byte was gathered from the checkpoint")
        if self.bytes_to_device == 0:
            bad.append("no byte was copied to the device: the slot would hold whatever was there before")
        if not self.experts_produced:
            bad.append("no expert was produced")
        if bytes_per_expert is not None:
            want = self.expert_slots_served * bytes_per_expert
            if self.bytes_to_device != want:
                bad.append(f"{self.bytes_to_device} B reached the device but {self.expert_slots_served} expert "
                           f"slot(s) x {bytes_per_expert} B = {want} B were served")
        if expected is not None:
            want_set = {(int(a), int(b)) for a, b in expected}
            if self.experts_produced != want_set:
                missing = sorted(want_set - self.experts_produced)
                extra = sorted(self.experts_produced - want_set)
                bad.append(f"produced {len(self.experts_produced)} of the expected {len(want_set)} "
                           f"(layer, expert) pairs (missing {missing[:8]}, extra {extra[:8]})")
        return bad

    def check(self, expected: Iterable[tuple[int, int]] | Sequence[int] | None = None, *, layer: int | None = None,
              bytes_per_expert: int | None = None, strict: bool = True) -> list[str]:
        """Return the violations; with ``strict`` (the default) raise on any.

        ``expected`` is mandatory under ``strict`` and must come from OUTSIDE
        this object -- an expectation derived from what it checks cannot fail.
        Pass e.g. ``stream_prefill.resident_chunk_ids(placement)``.
        """
        if strict and expected is None:
            raise ValueError(
                "SlotCoverage.check(strict=True) needs the expected experts, and they must not come from "
                "the provider: an expectation read off the object under test cannot fail.")
        bad = self.violations(_normalise_expected(expected, layer), bytes_per_expert=bytes_per_expert)
        if bad and strict:
            raise ZeroCoverageError(f"{self.summary()} -- " + "; ".join(bad))
        return bad


def _normalise_expected(expected: Any, layer: int | None) -> list[tuple[int, int]] | None:
    if expected is None:
        return None
    items = list(expected)
    if not items:
        raise ValueError("expected experts is empty: 'nothing was supposed to stream' is not a criterion")
    if all(isinstance(e, (tuple, list)) and len(e) == 2 for e in items):
        return [(int(a), int(b)) for a, b in items]
    if layer is None:
        raise ValueError("expected experts given as plain ids needs layer= so the pairs are unambiguous")
    return [(int(layer), int(e)) for e in items]


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


@dataclass
class _HostBuffer:
    """One pinned staging buffer and what it currently holds."""

    buffer: torch.Tensor
    key: tuple | None = None            # (layer, chunk_index, experts) it holds
    pending: tuple | None = None        # key a background gather is filling in
    future: Future | None = None
    h2d_handle: Any = None              # event of the last H2D that READ this buffer
    gather: GatherResult | None = None


@dataclass(frozen=True)
class ServedNZChunk:
    """What a chunk looks like in NZ device mode: NZ weights by address, ND scales as tensors.

    ``nz_weights`` are whole-storage NZ tensors holding ``num_slots`` slots back
    to back; slot ``i`` of parameter ``p`` starts at
    ``i * nz_bytes_per_slot[p]`` of its storage. That per-slot offset is the one
    property the raw channel rests on, and it is the same property
    ``_upload_nz`` already writes through -- the NZ permutation is per expert, so
    an expert's bytes are a contiguous, equally sized block. ``stream`` is the
    ACL stream the H2D went to: a raw read issued on another stream is ordered
    against nothing.
    """

    nz_weights: Mapping[str, torch.Tensor]
    nz_bytes_per_slot: Mapping[str, int]
    nd_scales: Mapping[str, torch.Tensor]
    num_slots: int
    stream: int


class StreamSlotProvider:
    """Turns ``(chunk_index, expert_ids)`` into a ``ChunkSlot`` on the device.

    Use it as ``run_streamed_layer(slot_for=provider.slot_for, ...)`` after
    ``set_layer(layer_idx)``: one layer at a time, because that is the shape of
    the callback and because deferring any part of a layer across layers is
    forbidden. The device and host buffers outlive the layer and are reused by
    every layer.

    **Construct it before the KV-cache profile run**: every byte of the slot is
    allocated in ``__init__``, and a slot taken afterwards comes out of memory
    vLLM has already promised to the KV cache.

    ``probe_bytes`` (default 0 = off) reads a few bytes back from each device
    buffer after the copy and compares them with the staging buffer. It costs a
    device->host sync per chunk, so it is a diagnostic, not the steady state --
    but it is the only detector that can see a chunk served with the *previous*
    chunk's bytes. Head AND tail are probed: bytes two chunks happen to share
    have no discriminating power.
    """

    def __init__(self, source: WeightSource | str, quant_method: Any, *, hidden_size: int, intermediate_size: int,
                 chunk_experts: int, device: Any = "npu", params_dtype: torch.dtype = torch.bfloat16,
                 seam: DeviceSeam | None = None, host_buffers: int = DEFAULT_HOST_BUFFERS,
                 copy_threads: int = DEFAULT_SLOT_COPY_THREADS, layer: int | None = None,
                 probe_bytes: int = 0, strict: bool = True):
        _check_group_map()
        if host_buffers < 1:
            raise ValueError(f"host_buffers must be >= 1, got {host_buffers}")
        # 🔴 copy_threads applies ONLY when this builds the source itself. Given
        # a source object, the object's own thread count is what runs, and
        # accepting a DIFFERENT number here is how the bank arm ended up on 4
        # threads while this default said 8 (see wiring.setup_streaming).
        # Refused only when it CONTRADICTS the source: passing the same number
        # states a fact, and leaving the default is not a claim about a source
        # this provider did not build.
        if not isinstance(source, str):
            own = getattr(source, "copy_threads", None)
            if own is not None and copy_threads != DEFAULT_SLOT_COPY_THREADS and copy_threads != own:
                raise ValueError(
                    f"StreamSlotProvider: copy_threads={copy_threads} contradicts the source "
                    f"object's own {own}, and the SOURCE's number is what gathers -- this argument "
                    "only takes effect when the provider builds the source from a path. Set it on "
                    "the source.")
        self.source = (WeightSource(source, copy_threads=copy_threads) if isinstance(source, str) else source)
        self.owns_source = isinstance(source, str)
        self.quant_method = quant_method
        self.chunk_experts = int(chunk_experts)
        self.device = device
        self.seam = seam or DeviceSeam()
        self.strict = strict
        self.probe_bytes = int(probe_bytes)
        self.coverage = SlotCoverage()
        self._layer = layer
        self._lock = threading.Lock()

        self.shapes = chunk_param_shapes(quant_method, self.chunk_experts, intermediate_size, hidden_size,
                                         params_dtype)
        #: bytes one expert occupies across the four parameters -- the number the
        #: H2D accounting is checked against.
        self.bytes_per_expert = sum(_prod(shape[1:]) for shape, _ in self.shapes.values())
        self.staging_bytes = self.bytes_per_expert * self.chunk_experts

        # One device slot, allocated once and reused by every chunk.
        self._device: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in self.shapes.items():
            buf = self.seam.alloc_device(shape, dtype, device)
            if tuple(buf.shape) != shape or buf.dtype != dtype:
                raise StreamIOError(f"device buffer {name} came back {buf.dtype}{tuple(buf.shape)}, "
                                    f"asked for {dtype}{shape}")
            self._device[name] = buf
        self._device_key: tuple | None = None
        # Stage 2 (default off): the staging arrives in device NZ layout, so the
        # cast is skipped and the NZ-tagged destination is built once.
        self._nz_device = False
        self._nz_slot: ChunkSlot | None = None
        self._nz_bytes: dict[str, int] = {}
        self._nz_h2d_stream: int | None = None

        # Host-side ping-pong, pinned.
        self._host: list[_HostBuffer] = []
        for _ in range(host_buffers):
            buf = self.seam.alloc_pinned(self.staging_bytes)
            if buf.dtype != torch.uint8 or buf.device.type != "cpu" or not buf.is_contiguous():
                raise StreamIOError(f"staging buffer must be contiguous CPU uint8, got {buf.dtype} "
                                    f"on {buf.device}")
            if buf.numel() < self.staging_bytes:
                raise StreamIOError(f"staging buffer holds {buf.numel()} B, a chunk needs {self.staging_bytes} B")
            self._host.append(_HostBuffer(buf))
            self.coverage.pinned_buffers += 1
            self.coverage.pinned_bytes += int(buf.numel())
        self._prefetch_pool: ThreadPoolExecutor | None = None
        #: wall clock per stage of slot_for, and how many chunks it covers
        self._stage_s: dict[str, float] = {"acquire": 0.0, "upload": 0.0, "format": 0.0}
        self._stage_n = 0
        #: Per report window: bytes the H2D actually moved, and slots filled
        #: from HBM instead. The GiB figure in stage_report is derived from the
        #: CONFIGURED chunk size, so without these a run that skipped a quarter
        #: of every chunk would report the same GiB as one that skipped nothing
        #: -- i.e. the one line that is supposed to show the mechanism would not.
        self._stage_h2d_bytes = 0
        self._stage_hbm_slots = 0
        self._stage_full_chunks = 0
        self._stage_mixed_chunks = 0
        self._stage_d2d_calls = 0
        self._stage_d2d_bytes = 0
        self._record_bytes_checked = False
        self._last_host: _HostBuffer | None = None
        #: Set per streamed forward by ``set_resident``: the module whose
        #: parameters already hold some of this layer's experts in HBM, and
        #: logical id -> its slot there (<0 = not resident). None = stream
        #: everything, which is what shipped through stage 3.
        self._res_module: Any | None = None
        self._res_slot: tuple[int, ...] | None = None
        self._res_per: dict[str, int] = {}
        self.coverage_from_hbm = 0

    # -- layer binding ----------------------------------------------------
    @property
    def layer(self) -> int:
        if self._layer is None:
            raise StreamIOError("no layer bound: call set_layer(idx) before asking for a chunk "
                                "(the slot_for callback carries the chunk index, not the layer)")
        return self._layer

    def set_layer(self, layer: int) -> StreamSlotProvider:
        """Bind the provider to a layer. Drops any prefetch aimed at the old one."""
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError(f"layer must be a non-negative int, got {layer!r}")
        self._drain_prefetch()
        self._layer = layer
        return self

    # -- the resident half: 26% of the streamed bytes, bit-identically -------
    def set_resident(self, module: Any, logical_to_slot: Sequence[int] | None) -> None:
        """Bind the experts that are ALREADY in HBM for the layer about to stream.

        With this set, a chunk's resident members are not gathered and not
        H2D'd: their slots in the chunk buffer are filled by a
        device-to-device copy out of ``module``'s parameters, which hold the
        same bytes. Everything downstream sees a COMPLETE chunk, so the chunk
        partition, the slot tables and the fp32 accumulation order are
        untouched and the layer's output is **bit-identical** to streaming all
        384 -- which is the whole reason for doing it at this level rather than
        computing the resident half separately (measure/PLAN_G_stream_only_offloaded.md).

        🔴 ONE SNAPSHOT per forward. Under ``dynamic_resident`` the fold moves
        experts between HBM and the CPU at the end of a layer, so a membership
        decision taken twice inside one forward could leave a slot nobody fills.
        The caller passes the same snapshot it gives the fold, and passes None
        to stream everything.
        """
        if logical_to_slot is None:
            self._res_module, self._res_slot, self._res_per = None, None, {}
            return
        if not self._nz_device:
            raise StreamIOError(
                "set_resident needs NZ device mode (the bank): the ND path copies whole group "
                "blocks out of the staging, so a slot it did not stream would be overwritten with "
                "stale bytes rather than kept")
        if module is None:
            raise StreamIOError(
                "set_resident needs the module whose parameters hold the resident experts: with no "
                "source for those slots, the chunks containing them would be computed from "
                "whatever the buffer held last")
        l2s = tuple(int(x) for x in logical_to_slot)
        per: dict[str, int] = {}
        for name in SLOT_WEIGHT_ATTRS:
            t = getattr(module, name, None)
            if t is None:
                raise StreamIOError(
                    f"the resident module has no {name}; all four parameters are needed, because a "
                    "slot filled for the weights but not the scales is wrong in a way no shape "
                    "check would catch")
            rows = int(t.shape[0])
            if rows <= 0:
                raise StreamIOError(f"the resident module's {name} has no rows")
            nbytes = t.untyped_storage().nbytes()
            if nbytes % rows:
                raise StreamIOError(
                    f"{name}: resident storage {nbytes} B is not a multiple of its {rows} slots, so "
                    "a per-slot offset into it is undefined")
            per[name] = nbytes // rows
            want = (self._nz_bytes.get(name) if name in NZ_WEIGHT_ATTRS
                    else self._slot_bytes_nd(name))
            if per[name] != want:
                raise StreamIOError(
                    f"{name}: the resident parameter holds {per[name]} B per slot, the chunk buffer "
                    f"{want} B. Both come out of the same process_weights_after_loading, so this is "
                    "a layout divergence and every copy after it would be misaligned")
            bad = [e for e, sl in enumerate(l2s) if sl >= rows]
            if bad:
                raise StreamIOError(
                    f"logical_to_slot sends experts {bad[:4]} to slots past the resident "
                    f"parameter's {rows}")
        self._res_module, self._res_slot, self._res_per = module, l2s, per

    def _slot_bytes_nd(self, name: str) -> int:
        """Bytes one slot occupies in the chunk's ND SCALE buffer.

        The ND scale buffers are the ones kept alive in NZ device mode (the
        slot's scales are transposed views of them), so this is the destination
        a resident scale row is copied into.
        """
        t = self._device.get(name)
        if t is None:
            raise StreamIOError(
                f"{name}: no ND buffer on this provider, so a resident scale has nowhere to land")
        nbytes = t.untyped_storage().nbytes()
        if nbytes % self.chunk_experts:
            raise StreamIOError(
                f"{name}: ND storage {nbytes} B is not a multiple of {self.chunk_experts} slots")
        return nbytes // self.chunk_experts

    def _split_chunk(self, ids: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """(positions to stream, positions already in HBM) for this chunk."""
        if self._res_slot is None:
            return tuple(range(len(ids))), ()
        bad = [e for e in ids if e >= len(self._res_slot)]
        if bad:
            raise StreamIOError(
                f"expert ids {bad[:4]} are outside the {len(self._res_slot)}-entry residency map")
        res = tuple(i for i, e in enumerate(ids) if self._res_slot[e] >= 0)
        return tuple(i for i in range(len(ids)) if i not in set(res)), res

    def _fill_resident(self, ids: tuple[int, ...], positions: Sequence[int],
                       num_experts: int) -> int:
        """D2D the resident members' slots of the chunk buffer, weights and scales.

        Returns the ACL stream the copies went to. A FULLY resident chunk runs
        no H2D at all, so this is the only write its bytes get -- and
        ``served_nz`` (the fold) refuses to hand out a chunk without a stream to
        order its own raw read against. Reporting it here is what makes a
        chunk's provenance irrelevant to every consumer.
        """
        import acl  # device-only dependency, imported at the call site

        from vllm_ascend.kt_offload.fold_swap import merge_runs  # noqa: PLC0415 - import cycle

        module, l2s = self._res_module, self._res_slot
        if module is None or l2s is None:          # _split_chunk guarantees it
            raise StreamIOError("_fill_resident without a binding")
        pairs = [(p, l2s[ids[p]]) for p in positions]
        runs = merge_runs(pairs)
        stream = torch.npu.current_stream().npu_stream
        for name in SLOT_WEIGHT_ATTRS:
            src = getattr(module, name)
            per = self._res_per[name]
            # 🔴 Raw D2D through the STORAGE for all four, scales included.
            # `Tensor.copy_` is not an option on either side: the weights are in
            # the internal NZ format, and the SCALES are a reshape+transpose
            # VIEW of their bytes (w4a8_mxfp4.process_weights_after_loading does
            # `.reshape(g, n, k // 2, 2).transpose(-3, -2)`), so copying the view
            # would transpose the data -- the same trap stream_layer's ND scale
            # copy documents. Measured as `aclnnInplaceCopy ... 161002` on the
            # first on-card attempt. The storages hold the same per-slot bytes
            # in the same order, which is what set_resident asserts.
            if name in ND_SCALE_ATTRS:
                base = self._device[name].untyped_storage().data_ptr()
                cap = self._slot_bytes_nd(name) * num_experts
            else:
                base = getattr(self._nz_slot, name).untyped_storage().data_ptr()
                cap = self._nz_bytes[name] * num_experts
            src_base = src.untyped_storage().data_ptr()
            for dst_row, src_slot, nrows in runs:
                nbytes = per * nrows
                off = per * dst_row
                if off < 0 or off + nbytes > cap:
                    raise StreamIOError(
                        f"{name}: resident fill would write [{off}, {off + nbytes}) of a {cap} B "
                        f"chunk buffer")
                # ACL_MEMCPY_DEVICE_TO_DEVICE = 3. destMax is a promise, there is
                # no srcMax, so both ends are bounded here: the destination above,
                # and the source by set_resident's per-slot arithmetic plus the
                # slot range it checked against the parameter's own rows.
                ret = acl.rt.memcpy_async(base + off, nbytes, src_base + per * src_slot, nbytes,
                                          3, stream)
                if ret != 0:
                    raise StreamIOError(f"{name}: resident D2D memcpy_async returned {ret}")
                self._stage_d2d_calls += 1
                self._stage_d2d_bytes += nbytes
        self.coverage_from_hbm += len(positions)
        return int(stream)

    # -- the callback -----------------------------------------------------
    def _key(self, chunk_index: int, ids: tuple[int, ...],
             sub: tuple[int, ...] | None = None) -> tuple:
        """What identifies the bytes in the buffers: layer, chunk, ids, SUBSET.

        The subset belongs in the key because a buffer gathered for a different
        subset holds different bytes in the slots that were skipped. Leaving it
        out would let a prefetch for the whole chunk satisfy a request for part
        of it, or the reverse -- right shape, plausible bytes, wrong answer.
        """
        return (self.layer, int(chunk_index), ids, sub)

    def slot_for(self, chunk_index: int, expert_ids: Sequence[int],
                 only_slots: Sequence[int] | None = None) -> ChunkSlot:
        """Gather -> H2D -> production post-load hook -> ``ChunkSlot``.

        ``only_slots`` streams just those POSITIONS of the chunk. The rest of
        the slot keeps whatever it held, so the caller must fill them another
        way -- in production from the resident parameter already in HBM, which
        saves 26% of the streamed bytes while leaving the chunk partition, and
        therefore the fp32 accumulation order, untouched (see
        measure/PLAN_G_stream_only_offloaded.md).
        """
        ids = tuple(int(e) for e in expert_ids)
        if not ids:
            raise ZeroCoverageError(f"chunk {chunk_index}: no expert to stream")
        if len(ids) > self.chunk_experts:
            raise StreamIOError(f"chunk {chunk_index} has {len(ids)} experts but the slot holds "
                                f"{self.chunk_experts}; the device buffer is allocated once and not grown")
        if len(set(ids)) != len(ids):
            raise ValueError(f"chunk {chunk_index}: duplicate expert ids {ids}")
        stream_pos, res_pos = self._split_chunk(ids)
        if res_pos and only_slots is not None:
            raise StreamIOError(
                f"chunk {chunk_index}: only_slots was given AND {len(res_pos)} of its experts are "
                "bound as resident. Two different answers to 'which slots does the host fill' is "
                "one too many")
        # The key first: every branch below needs it, including the one that
        # never touches the host. `sub` is the subset the HOST is responsible
        # for -- () means "none of it", which is a different claim from None
        # ("all of it") and has to be in the key as such.
        if res_pos:
            sub = self._check_subset(chunk_index, ids, stream_pos) if stream_pos else ()
        else:
            sub = self._check_subset(chunk_index, ids, only_slots)
        key = self._key(chunk_index, ids, sub)
        # 🔴 A chunk can be ENTIRELY resident, and under the shipped config most
        # of them are: `resident_strategy: prefix` with 100 of 384 makes chunks
        # 0, 1 and 2 (experts 0-95) fully resident and chunk 3 mixed. So this is
        # the common case, not an edge: skip the host side completely and fill
        # all 32 slots from HBM. The device buffers are allocated once at
        # construction, so nothing here depends on a gather having run.
        if res_pos and not stream_pos:
            _t0 = time.perf_counter()
            self._device_key = None
            # 🔴 The D2D is this chunk's ONLY write, so the stream it went to is
            # what the fold's raw read has to be ordered against. Without this
            # `served_nz` raises "no H2D has run yet" on the very first chunk of
            # the very first layer -- which is exactly how the fifth on-card
            # attempt failed.
            self._nz_h2d_stream = self._fill_resident(ids, res_pos, len(ids))
            self._stage_hbm_slots += len(res_pos)
            self._stage_full_chunks += 1
            self._device_key = key
            slot = self._format(len(ids))
            self._stage_s["format"] += time.perf_counter() - _t0
            self._stage_n += 1
            self.coverage.chunks_served += 1
            self.coverage.expert_slots_served += len(ids)
            self.coverage.layers_served.add(key[0])
            self.coverage.experts_produced.update((key[0], e) for e in ids)
            return slot
        # Per-stage wall clock. These three are the whole of slot_for, and
        # ``_acquire`` includes waiting for a prefetch -- which is what tells
        # overlap apart from serialisation.
        _t0 = time.perf_counter()
        host = self._acquire(key)
        _t1 = time.perf_counter()
        views = host.gather.layout.views(host.buffer)
        self._check_record_bytes(host.gather.layout, len(ids))
        if self._nz_device:
            self._upload_nz(host, views, len(ids), key)
        else:
            self._upload(host, views, len(ids), key)
        _t2 = time.perf_counter()
        if res_pos:
            # After the H2D, on the same stream: the GEMM reads this buffer, and
            # both copies are ordered against it by the stream itself.
            self._fill_resident(ids, res_pos, len(ids))
            self._stage_hbm_slots += len(res_pos)
            self._stage_mixed_chunks += 1
        if self.probe_bytes:
            self._probe(views, len(ids), key)
        slot = self._format(len(ids))
        _t3 = time.perf_counter()
        self._stage_s["acquire"] += _t1 - _t0
        self._stage_s["upload"] += _t2 - _t1
        self._stage_s["format"] += _t3 - _t2
        self._stage_n += 1
        # Every 120 chunks (10 layers at the default 12 chunks/layer): enough
        # lines to see the split drift between layers, few enough not to be the
        # cost itself.
        if self._stage_n >= 120:
            logger.info("kt_offload stream I/O: %s", self.stage_report())

        self.coverage.chunks_served += 1
        self.coverage.expert_slots_served += len(ids)
        self.coverage.layers_served.add(key[0])
        self.coverage.experts_produced.update((key[0], e) for e in ids)
        return slot

    def stage_report(self, *, reset: bool = True) -> str:
        """Where the streamed bytes' wall clock went, and the implied bandwidth.

        Compare that bandwidth against the two legs' known rates to see which is
        off. ``acquire`` covers waiting for the prefetch, so an acquire far above
        the gather's own cost means the prefetch is not running ahead.
        """
        n = self._stage_n
        if not n:
            return "no chunk served"
        gib = (self.staging_bytes * n) / (1 << 30)
        moved_gib = self._stage_h2d_bytes / float(1 << 30)
        total = sum(self._stage_s.values())
        parts = []
        for name in ("acquire", "upload", "format"):
            t = self._stage_s[name]
            parts.append(f"{name} {t:.2f}s ({100 * t / total:.0f}%, {gib / t:.1f} GiB/s)" if t > 0
                         else f"{name} 0")
        out = (f"{n} chunk(s), {gib:.1f} GiB staged, {total:.2f}s total "
               f"({1000 * total / n:.1f} ms/chunk): " + ", ".join(parts))
        if self._stage_hbm_slots:
            # 🔴 The mechanism, in the server's own words: how much the host
            # actually moved against what a whole-chunk run would have, and how
            # many slots came from HBM instead.
            out += (f"; HOST MOVED {moved_gib:.1f} of {gib:.1f} GiB "
                    f"({100.0 * moved_gib / gib:.1f}%), {self._stage_hbm_slots} slot(s) filled "
                    f"from HBM instead in {self._stage_d2d_calls} D2D call(s) "
                    f"({self._stage_d2d_bytes / float(1 << 30):.2f} GiB, "
                    f"{self._stage_full_chunks} whole / {self._stage_mixed_chunks} mixed chunks)")
        if reset:
            self._stage_h2d_bytes = 0
            self._stage_hbm_slots = 0
            self._stage_full_chunks = 0
            self._stage_mixed_chunks = 0
            self._stage_d2d_calls = 0
            self._stage_d2d_bytes = 0
            self._stage_s = {"acquire": 0.0, "upload": 0.0, "format": 0.0}
            self._stage_n = 0
        return out

    # -- 1. host gather ---------------------------------------------------
    def prefetch(self, chunk_index: int, expert_ids: Sequence[int],
                 only_slots: Sequence[int] | None = None) -> bool:
        """Start filling a pinned buffer for a chunk in the background.

        The ping-pong: host side only, no device buffer doubled. Returns whether
        a gather was started; already-held or already-pending chunks start
        nothing.
        """
        ids = tuple(int(e) for e in expert_ids)
        if not ids:
            raise ZeroCoverageError(f"chunk {chunk_index}: nothing to prefetch")
        # 🔴 THE KEY MUST BE THE ONE `slot_for` WILL COMPUTE, so the subset is
        # derived here exactly as it is there (lines above in `slot_for`) rather
        # than taken from the caller. Under `set_resident` the host is
        # responsible only for the STREAMED positions, so a prefetch that asked
        # for the whole chunk would build a DIFFERENT key: `slot_for` would then
        # ignore the prefetched buffer and gather again, and the gather it did
        # would be the full chunk -- undoing stream_resident_from_hbm's 26% and
        # paying for a wasted pass on top. The caller must not have to know
        # about residency; this is the only place that can get it right.
        stream_pos, res_pos = self._split_chunk(ids)
        if res_pos and only_slots is not None:
            raise StreamIOError(
                f"chunk {chunk_index}: only_slots was given AND {len(res_pos)} of its experts are "
                "bound as resident; prefetch cannot answer 'which slots does the host fill' twice "
                "either")
        if res_pos:
            sub = self._check_subset(chunk_index, ids, stream_pos) if stream_pos else ()
        else:
            sub = self._check_subset(chunk_index, ids, only_slots)
        if sub == ():
            # Fully resident: the host fills nothing, so there is nothing to
            # start in the background. `slot_for` will serve it from HBM.
            return False
        key = self._key(chunk_index, ids, sub)
        with self._lock:
            if any(h.key == key or h.pending == key for h in self._host):
                return False
            # Never refill the buffer the current chunk is still copying out of:
            # _fill would block on its H2D event and the prefetch would serialise
            # with the very copy it is supposed to overlap.
            target = None
            for h in self._host:
                if h.future is None and h is not self._last_host:
                    target = h
                    break
            if target is None:
                return False
            target.pending = key
            if self._prefetch_pool is None:
                self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kt_slot_prefetch")
            target.future = self._prefetch_pool.submit(self._fill, target, key)
            return True

    def _check_subset(self, chunk_index: int, ids: tuple[int, ...],
                      only_slots: Sequence[int] | None) -> tuple[int, ...] | None:
        """Normalise and refuse an impossible subset. None = the whole chunk."""
        if only_slots is None:
            return None
        sub = tuple(sorted({int(x) for x in only_slots}))
        if not sub:
            raise ZeroCoverageError(
                f"chunk {chunk_index}: only_slots is empty, so this would H2D whatever the staging "
                "held last")
        if sub[0] < 0 or sub[-1] >= len(ids):
            raise StreamIOError(
                f"chunk {chunk_index}: only_slots {sub} outside the {len(ids)}-slot chunk")
        if len(sub) == len(ids):
            return None                      # a "subset" of everything is the whole chunk
        if not getattr(self.source, "supports_subset_gather", False):
            raise StreamIOError(
                f"{type(self.source).__name__} cannot gather a subset of a chunk, so asking for one "
                "would stream the whole chunk and then H2D only part of it")
        return sub

    def _served_subset(self) -> tuple[int, ...] | None:
        """The subset of the chunk currently in the device buffers."""
        return self._device_key[3] if self._device_key is not None else None

    def _fill(self, host: _HostBuffer, key: tuple) -> None:
        # The pinned buffer must not be overwritten while the copy that reads it
        # is still in flight; non_blocking H2D is exactly the case where "the
        # call returned" does not mean "the bytes were read".
        if host.h2d_handle is not None:
            self.seam.wait(host.h2d_handle)
            host.h2d_handle = None
        host.key = None
        host.gather = None
        result = (self.source.gather(key[0], key[2], host.buffer)
                  if key[3] is None else
                  self.source.gather(key[0], key[2], host.buffer, only_slots=key[3]))
        host.gather = result
        host.key = key
        with self._lock:
            self.coverage.gathers += 1
            self.coverage.bytes_gathered += result.coverage.produced_bytes

    def _acquire(self, key: tuple) -> _HostBuffer:
        host = self._acquire_buffer(key)
        self._last_host = host
        return host

    def _acquire_buffer(self, key: tuple) -> _HostBuffer:
        with self._lock:
            pending = next((h for h in self._host if h.pending == key), None)
        if pending is not None:
            # .result() re-raises the worker's exception instead of handing back a
            # half-filled buffer.
            pending.future.result()
            with self._lock:
                pending.future, pending.pending = None, None
            if pending.key != key or pending.gather is None:
                raise StreamIOError(f"prefetch for {key} finished without filling its buffer")
            return pending
        with self._lock:
            held = next((h for h in self._host if h.key == key and h.future is None), None)
        if held is not None:
            return held
        with self._lock:
            target = next((h for h in self._host if h.future is None), None)
        if target is None:
            target = self._host[key[1] % len(self._host)]
            target.future.result()
            with self._lock:
                target.future, target.pending = None, None
        self._fill(target, key)
        return target

    def _drain_prefetch(self) -> None:
        for h in self._host:
            if h.future is not None:
                try:
                    h.future.result()
                finally:
                    h.future, h.pending = None, None

    def _check_record_bytes(self, layout: ChunkLayout, num_experts: int) -> None:
        """The checkpoint's per-expert record must be what the parameter holds.

        Checked once, being a property of the checkpoint and the model dims. A
        mismatch means the provider was built for different dimensions than the
        weights on disk, which would otherwise surface as a size error deep
        inside a copy, or not at all.
        """
        if self._record_bytes_checked:
            return
        for group, param in PARAM_FOR_GROUP.items():
            if group not in layout.groups:
                raise StreamIOError(f"the gather layout has no group {group!r}")
            record = layout.groups[group][1]
            want = _prod(self.shapes[param][0][1:])
            if record != want:
                raise StreamIOError(
                    f"checkpoint gives {record} B per expert for {group}, but {param} holds {want} B per "
                    f"expert ({self.shapes[param][0]}): the provider's hidden/intermediate sizes do not "
                    "match the checkpoint")
        if layout.nbytes != self.bytes_per_expert * num_experts:
            raise StreamIOError(f"chunk layout is {layout.nbytes} B, parameters need "
                                f"{self.bytes_per_expert * num_experts} B")
        self._record_bytes_checked = True

    # -- 2. H2D -----------------------------------------------------------
    def _device_view(self, name: str, num_experts: int) -> torch.Tensor:
        buf = self._device[name]
        return buf if num_experts == int(buf.shape[0]) else buf[:num_experts]

    # -- what layout are the chunks in? -----------------------------------
    @property
    def nz_device(self) -> bool:
        """True once ``enable_nz_device`` has run: chunks are FRACTAL_NZ, no ND landing buffers.

        Part of the provider's contract rather than a private flag because a
        consumer that copies bytes OUT of a chunk (``fold_swap``) has to agree
        with it about the layout; guessing ND here is what staged 12,000 experts
        of never-written bytes with no error.
        """
        return self._nz_device

    @property
    def nz_bytes_per_slot(self) -> Mapping[str, int]:
        """Bytes one slot occupies in each NZ weight buffer; empty in ND mode."""
        return dict(self._nz_bytes)

    # -- stage 2: the staging already holds device-layout NZ bytes ---------
    def enable_nz_device(self) -> None:
        """Switch the device side to "the bytes arriving are already NZ".

        With the bank as the source the staging holds FRACTAL_NZ bytes in device
        order, so the ND landing buffer and the per-chunk cast are both dead
        weight: the cast would re-permute bytes that are already permuted. This
        builds the NZ-tagged destination ONCE, through the production
        ``format_nd`` -- the tag has to come from the real cast because the
        operator dispatches on it (a plain ND tensor holding the same NZ bytes is
        refused with 161002, measured as V4).

        What survives, and why each half is handled differently:
          * the two WEIGHTS become fresh NZ allocations. torch cannot write into
            an internal-format tensor at all (``copy_`` refuses internal formats
            with ERR00007, an offset view trips ``storage_offset() == 0``), so
            they are filled by a raw ACL host-to-device memcpy into their
            storage -- the channel V3-D4 measured as landing exactly where it
            should and leaving neighbours byte-identical.
          * the two SCALES stay plain ND views of the buffers passed in, because
            scales never go through the cast. They are filled with an ordinary
            ``copy_``, and writing the ND buffer updates the slot's view.
        """
        if self._nz_device:
            return
        self._nz_device = True
        self._nz_slot = self.format_nd({n: self._device[n] for n in SLOT_WEIGHT_ATTRS},
                                       self.chunk_experts)
        for name in NZ_WEIGHT_ATTRS:
            t = getattr(self._nz_slot, name)
            st = t.untyped_storage()
            if st.nbytes() % self.chunk_experts:
                raise StreamIOError(
                    f"{name}: NZ storage {st.nbytes()} B is not a multiple of "
                    f"{self.chunk_experts} slots, so a per-slot offset is undefined")
            self._nz_bytes[name] = st.nbytes() // self.chunk_experts
        # Drop the ND WEIGHT buffers: the cast handed back fresh NZ allocations
        # and nothing writes the ND ones again, so holding them is pure waste
        # (0.527 GiB of HBM for the 32-expert chunk). The ND SCALE buffers MUST
        # stay -- the slot's scales are views of them, and the per-chunk scale
        # copy writes through those views.
        freed = 0
        for name in NZ_WEIGHT_ATTRS:
            t = self._device.pop(name, None)
            if t is not None:
                freed += t.numel() * t.element_size()
                del t
        if freed:
            import gc
            gc.collect()
            torch.npu.empty_cache()
        logger.info("kt_offload stream I/O: NZ device mode -- staging arrives in device layout, "
                    "no per-chunk cast; per-slot bytes %s; freed %d B (%.3f GiB) of ND weight "
                    "landing buffers", dict(self._nz_bytes), freed, freed / float(1 << 30))

    def _upload_nz(self, host: "_HostBuffer", views: Mapping[str, torch.Tensor],
                   num_experts: int, key: tuple) -> None:
        """H2D straight into the NZ-tagged buffers. No cast follows this."""
        import acl  # device-only dependency, imported at the call site

        self._device_key = None
        stream = torch.npu.current_stream().npu_stream
        # Which ACL stream these raw copies went to. A caller that does its own
        # raw copy OUT of the NZ slot (fold_swap) is ordered against this H2D
        # only if it issues on the same stream: both are submitted directly,
        # so the stream itself orders them and no event is involved. Issued on
        # another stream they are ordered against nothing, so the fold compares
        # this against its own current stream and refuses a mismatch.
        self._nz_h2d_stream = stream
        # Which positions of the chunk the gather filled, as runs: a whole chunk
        # is the single run (0, num_experts) and every call below is then
        # byte-for-byte what it was before subsets existed.
        runs = self._filled_runs(host, num_experts)
        for group in (g for g, _ in GROUPS):
            param = PARAM_FOR_GROUP[group]
            src = views[group].reshape(-1)
            per_slot = int(src.numel()) // num_experts
            if param in ND_SCALE_ATTRS:
                dst = self._device_view(param, num_experts).reshape(-1)
                if dst.numel() != src.numel():
                    raise StreamIOError(f"{param}: device view {dst.numel()} B, staging {src.numel()} B")
                for first, count in runs:
                    lo, hi = first * per_slot, (first + count) * per_slot
                    dst[lo:hi].copy_(src[lo:hi], non_blocking=True)
            else:
                want = self._nz_bytes[param] * num_experts
                if int(src.numel()) != want:
                    raise StreamIOError(
                        f"{param}: staging group {group} has {src.numel()} B, the NZ buffer needs "
                        f"{want} B for {num_experts} slots")
                base = getattr(self._nz_slot, param).untyped_storage().data_ptr()
                per = self._nz_bytes[param]
                for first, count in runs:
                    nbytes, off = per * count, per * first
                    # ACL_MEMCPY_HOST_TO_DEVICE = 1. destMax is what we promise
                    # to write and there is NO srcMax, so both ends are bounded
                    # here: the runs come from slot indices range-checked in
                    # _check_subset, and off+nbytes <= want follows.
                    ret = acl.rt.memcpy_async(base + off, nbytes, src.data_ptr() + off, nbytes, 1, stream)
                    if ret != 0:
                        raise StreamIOError(f"{param}: H2D memcpy_async returned {ret}")
            moved = sum(c for _, c in runs) * per_slot
            self.coverage.h2d_calls += len(runs)
            self.coverage.bytes_to_device += moved
            self._stage_h2d_bytes += moved
        # The raw copies are on the current stream; an event on it is what makes
        # refilling the pinned source safe, exactly as the ND path's handle does.
        ev = torch.npu.Event()
        ev.record()
        host.h2d_handle = ev
        self._device_key = key

    def _filled_runs(self, host: "_HostBuffer", num_experts: int) -> list[tuple[int, int]]:
        """``[(first_slot, count)]`` for the positions this gather filled.

        A whole-chunk gather returns exactly ``[(0, num_experts)]``. The run
        merge is the fold's (``fold_swap.merge_runs``) rather than a second
        implementation of the same arithmetic; imported at the call site because
        fold_swap imports this module.
        """
        produced_slots = getattr(host.gather.layout, "produced_slots", None)
        if produced_slots is None:
            return [(0, num_experts)]
        pos = produced_slots()
        if len(pos) >= num_experts:
            return [(0, num_experts)]
        from vllm_ascend.kt_offload.fold_swap import merge_runs  # noqa: PLC0415 - import cycle

        return [(first, n) for first, _src, n in merge_runs([(p, p) for p in pos])]

    def _upload(self, host: _HostBuffer, views: Mapping[str, torch.Tensor], num_experts: int, key: tuple) -> None:
        produced = getattr(host.gather.layout, "produced", None)
        if produced is not None and produced is not host.gather.layout.experts:
            raise StreamIOError(
                "a subset gather reached the ND upload path, which copies whole group blocks and "
                "would overwrite the slots it did not stream with stale staging. Subsets are NZ "
                "device mode only (the bank).")
        self._device_key = None   # stale until every group has been copied
        handle = None
        # In staging order (``weight_source.GROUPS``), so the pinned buffer is
        # read front to back rather than jumped around.
        for group in (g for g, _ in GROUPS):
            param = PARAM_FOR_GROUP[group]
            dst = self._device_view(param, num_experts).reshape(-1)
            src = views[group].reshape(-1)
            if dst.numel() != src.numel():
                raise StreamIOError(f"{param}: device view holds {dst.numel()} B, staging group {group} "
                                    f"has {src.numel()} B")
            handle = self.seam.h2d(dst, src, what=param)
            self.coverage.h2d_calls += 1
            self.coverage.bytes_to_device += int(src.numel())
            self._stage_h2d_bytes += int(src.numel())
        # The pinned source stays off-limits until this copy has drained.
        host.h2d_handle = handle
        self._device_key = key

    def _probe(self, views: Mapping[str, torch.Tensor], num_experts: int, key: tuple) -> None:
        if key[3] is not None:
            # A subset gather left part of the staging holding the PREVIOUS
            # chunk's bytes, and those slots of the device buffer were filled
            # from HBM instead -- so a head/tail comparison against the staging
            # would be a guaranteed false alarm, exactly what _probe_nz's own
            # docstring warns about for the ND buffers.
            return
        if self._nz_device:
            self._probe_nz(views, num_experts, key)
            return
        for group, param in PARAM_FOR_GROUP.items():
            dev = self._device_view(param, num_experts).reshape(-1)
            src = views[group].reshape(-1)
            n = min(self.probe_bytes, src.numel())
            if n == 0:
                continue
            for where, window in (("head", slice(0, n)), ("tail", slice(src.numel() - n, src.numel()))):
                got = dev[window].to("cpu")
                if not torch.equal(got, src[window]):
                    raise StreamIOError(
                        f"chunk {key[1]} layer {key[0]}: the device buffer for {param} does not hold the "
                        f"bytes that were staged for it ({where} {n} B differ). The copy was issued but the "
                        "slot would have been computed with some other chunk's weights.")

    # -- 3. format (the production hook) ----------------------------------
    def _probe_nz(self, views: Mapping[str, torch.Tensor], num_experts: int, key: tuple) -> None:
        """Read back head and tail of each written range, in NZ mode.

        The weights cannot be read with ``.to("cpu")``: torch refuses internal
        formats (ERR00007), and ``.cpu()`` on an NZ tensor would undo the very
        permutation under test. So the weights go through a raw ACL
        device-to-host copy out of their storage, and the scales -- plain ND
        views -- are compared the ordinary way.

        Without this branch the probe would compare the ND landing buffer, which
        NZ mode never writes (and now does not even allocate), against NZ
        staging: a guaranteed false alarm rather than a check.
        """
        import acl  # device-only dependency, imported at the call site

        for group, param in PARAM_FOR_GROUP.items():
            src = views[group].reshape(-1)
            n = min(self.probe_bytes, src.numel())
            if n == 0:
                continue
            if param in ND_SCALE_ATTRS:
                dev = self._device_view(param, num_experts).reshape(-1)
                windows = (("head", slice(0, n)), ("tail", slice(src.numel() - n, src.numel())))
                for where, w in windows:
                    if not torch.equal(dev[w].to("cpu"), src[w]):
                        raise StreamIOError(
                            f"chunk {key[1]} layer {key[0]}: NZ-mode scale {param} {where} does not "
                            "match the staging it was copied from")
                continue
            base = getattr(self._nz_slot, param).untyped_storage().data_ptr()
            total = self._nz_bytes[param] * num_experts
            for where, off in (("head", 0), ("tail", total - n)):
                got = torch.empty(n, dtype=torch.uint8)
                # ACL_MEMCPY_DEVICE_TO_HOST = 2
                ret = acl.rt.memcpy(got.data_ptr(), n, base + off, n, 2)
                if ret != 0:
                    raise StreamIOError(f"{param}: probe D2H returned {ret}")
                want = src[off:off + n] if where == "head" else src[src.numel() - n:]
                if not torch.equal(got, want.cpu() if want.device.type != "cpu" else want):
                    raise StreamIOError(
                        f"chunk {key[1]} layer {key[0]}: NZ-mode {param} {where} does not hold the "
                        "bytes the staging had -- the raw H2D did not land where it was asked to")

    def _format(self, num_experts: int) -> ChunkSlot:
        if self._nz_device:
            if self._device_key is None:
                raise StreamIOError("the NZ device buffers were not filled for this chunk")
            return self._nz_slot
        if self._device_key is None:
            raise StreamIOError("the device buffers were not filled for this chunk: refusing to format and "
                                "hand back whatever the previous chunk left there")
        return self.format_nd({name: self._device_view(name, num_experts) for name in SLOT_WEIGHT_ATTRS},
                              num_experts)

    def format_nd(self, tensors: Mapping[str, torch.Tensor], num_experts: int) -> ChunkSlot:
        """Run the production post-load hook on ND device tensors and hand back the slot.

        The one place the cast happens, for a chunk AND for the folded swap's
        whole-layer staging (``fold_swap.py``). The ND tensors are not modified;
        the hook rebinds the module's parameters -- the weights to a fresh
        ``npu_format_cast`` output, the scales to a 4-D transposed VIEW of the ND
        bytes, so the caller must keep the ND scale storage alive and untouched
        while it uses the slot.
        """
        module = _ChunkModule(tensors)
        before = {name: layout_signature(getattr(module, name).data) for name in SLOT_WEIGHT_ATTRS}
        self.seam.post_load(self.quant_method, module)
        after = {name: getattr(module, name).data for name in SLOT_WEIGHT_ATTRS}
        unchanged = [n for n in SLOT_WEIGHT_ATTRS if layout_signature(after[n]) == before[n]]
        if unchanged:
            # All four are rewritten by the production hook: the weights get a
            # new ``npu_format_cast`` allocation (address) then a transpose
            # (stride), the scales a reshape to 4-D and a transpose (shape and
            # stride). Same address, shape AND stride means untouched, i.e. raw
            # ND bytes reaching the GEMM -- at best aclnnGroupedMatmulV4
            # refusing the call (161002), at worst numbers. Identity (``is``) is
            # NOT the test: ``param.data`` hands back a fresh tensor object on
            # every access, so it would pass for a hook that did nothing.
            raise StreamIOError(f"the post-load hook left {unchanged} exactly as they were: the chunk would "
                                "be handed to the GEMM in the checkpoint's ND layout, not the layout "
                                "process_weights_after_loading produces")
        slot = ChunkSlot(**{name: after[name] for name in SLOT_WEIGHT_ATTRS})
        bad = slot.violations(expect_slots=num_experts)
        if bad:
            raise StreamIOError(f"the formatted chunk is not a usable slot: {'; '.join(bad)}")
        return slot

    def served_nz(self, chunk_index: int, expert_ids: Sequence[int]) -> "ServedNZChunk":
        """The chunk just served, as it exists in NZ device mode. The fold's source.

        Counterpart of :meth:`served_nd` for the bank path, where the ND landing
        buffers do not exist (they are freed by ``enable_nz_device``). Two halves,
        because the two halves are genuinely different objects:

          * the two WEIGHTS are the NZ-tagged destination the raw H2D wrote.
            torch cannot read a row out of them (``index_select`` on an internal
            format is refused, and ``.cpu()`` would undo the permutation), so
            they are handed over as ``nz_weights`` together with the bytes ONE
            slot occupies and the ACL ``stream`` the H2D used -- everything a
            caller needs to copy a slot out by address, and nothing it needs to
            do so on the wrong stream.
          * the two SCALES never went through the cast: they are the same plain
            ND device views ``served_nd`` would have handed back, usable with
            ordinary torch ops.

        The staleness check is ``served_nd``'s, for the same reason: another
        chunk, or this index on another layer, is right shape, plausible bytes,
        wrong expert.
        """
        if not self._nz_device:
            raise StreamIOError(
                "served_nz() is only meaningful in NZ device mode; without the bank the chunk's bytes "
                "are in the ND landing buffers, which served_nd() hands back")
        ids = tuple(int(e) for e in expert_ids)
        key = self._key(chunk_index, ids, self._served_subset())
        if self._device_key != key:
            raise StreamIOError(f"the device buffers hold {self._device_key}, not {key}: the chunk asked for "
                                "was not the last one served, and its bytes are gone")
        if self._nz_h2d_stream is None:
            raise StreamIOError("no H2D has run yet, so there is no stream to order a raw read against")
        return ServedNZChunk(
            nz_weights={n: getattr(self._nz_slot, n) for n in NZ_WEIGHT_ATTRS},
            nz_bytes_per_slot=dict(self._nz_bytes),
            nd_scales={n: self._device_view(n, len(ids)) for n in ND_SCALE_ATTRS},
            num_slots=self.chunk_experts,
            stream=self._nz_h2d_stream,
        )

    def served_nd(self, chunk_index: int, expert_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """The ND device bytes of the chunk just served, in the chunk's slot order.

        For the folded swap (``fold_swap.py``), which copies the chosen experts
        into its whole-layer staging between chunks. They are the H2D
        destination, still in the checkpoint's ND layout (the cast writes a fresh
        tensor and leaves them alone), valid only until the next ``slot_for``
        overwrites them -- hence the key check: another chunk, or this index on
        another layer, is right shape, plausible bytes, wrong expert.
        """
        if self._nz_device:
            raise StreamIOError(
                "served_nd() has no meaning in NZ device mode: the staging goes straight into the "
                "NZ-tagged destination, so the ND landing buffers are never written (and are freed "
                "at start-up). A caller that read them would get unwritten bytes and no error. The "
                "config refuses bank_dir together with dynamic_resident for exactly this reason.")
        ids = tuple(int(e) for e in expert_ids)
        key = self._key(chunk_index, ids, self._served_subset())
        if self._device_key != key:
            raise StreamIOError(f"the device buffers hold {self._device_key}, not {key}: the chunk asked for "
                                "was not the last one served, and its bytes are gone")
        return {name: self._device_view(name, len(ids)) for name in SLOT_WEIGHT_ATTRS}

    # -- coverage / teardown ----------------------------------------------
    def check_coverage(self, expected: Iterable[Any] | None = None, *, layer: int | None = None,
                       strict: bool | None = None) -> list[str]:
        """Coverage verdict; ``expected`` must come from outside (see ``SlotCoverage.check``)."""
        return self.coverage.check(expected, layer=layer, bytes_per_expert=self.bytes_per_expert,
                                   strict=self.strict if strict is None else strict)

    def close(self) -> None:
        self._drain_prefetch()
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=True)
            self._prefetch_pool = None
        for h in self._host:
            if h.h2d_handle is not None:
                self.seam.wait(h.h2d_handle)
                h.h2d_handle = None
        if self.owns_source:
            self.source.close()


def layout_signature(t: torch.Tensor) -> tuple:
    """Address + shape + stride: what a cast/transpose changes and a no-op does not.

    ``stride()`` is wrapped because internal-format tensors regularly refuse
    ordinary metadata calls on this build; a refusal is itself a difference from
    the ND input, so ``None`` keeps the comparison meaningful instead of raising.
    """
    try:
        stride = tuple(t.stride())
    except Exception:  # noqa: BLE001 - see the docstring
        stride = None
    return (t.data_ptr(), tuple(t.shape), stride)


def _prod(dims: Sequence[int]) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n
