# SPDX-License-Identifier: Apache-2.0
"""Start-up self-check: is CANN's FRACTAL_NZ byte geometry the one we assume?

The shared-NZ route keeps ONE copy of the expert weights on the host, already in
the NPU's internal layout, and lets the CPU MoE kernel (kt-kernel) and the NPU
read *the same bytes*. That only works because we know where
``npu_format_cast(..., 29, ...)`` puts each byte -- and we know it because we
reverse-engineered it: **CANN documents none of this**. Today the CPU and device
sides are decoupled, so a geometry change would be caught by the GEMM; once they
share bytes, a CANN minor version that moves to another C0 or another fractal
order makes the CPU kernel compute *wrong numbers*, silently. No exception, no
NaN, just answers that are a bit off.

So: at start-up, cast a two-expert tensor of known content through the production
path, permute the same ND bytes on the host with the formula this version
assumes, and compare byte for byte. Match -> go. Mismatch -> refuse to start.
A few tens of milliseconds, and the difference between "wrong numbers in
production" and "the process did not come up".

**Who pays**: whoever depends on the geometry, and nobody else. The check is OFF
unless a switch that actually shares NZ bytes with the CPU kernel is on (today
``bank_dir`` and ``kt_reads_bank``). A
default-on check would hand every deployment -- including the ones that never
share a byte -- a start-up dependency on this cast hook: a CANN release that
changed the geometry would stop engines that did not need the guarantee. On the
other hand a deployment that DOES share bytes must not be able to turn it off and
keep the route: the switch binding, not a separate flag, is what makes that true.

The geometry (derived with a bit-plane probe, then replayed with random content
on four shapes, 0 mismatching bytes)::

    one expert's weight is logically [N, Kb] uint8    (Kb = K_fp4 / 2 BYTES)
    dst = e*(N*Kb) + (kb//16)*(N*16) + n*16 + (kb%16)

i.e. the physical buffer is a C-order ``buf[E][Kb/16][N][16]`` -- **C0 = 16
bytes**, ``gidx = kb//16`` selecting the k-block. The equivalent description "288
basic blocks of 16x16 per stripe, row-major inside a block" is the SAME geometry,
not a second one, because ``(n//16)*256 + (n%16)*16 == n*16``.

Production shapes (MXFP4, hidden 5120, moe_intermediate 2304, group_size 32)::

    w13_weight  ND [E, 4608, 2560]   N=4608  Kb=2560  (160 gidx)
    w2_weight   ND [E, 5120, 1152]   N=5120  Kb=1152  ( 72 gidx)

``group_size = 32`` fp4 is exactly 16 bytes, exactly one C0, exactly one k-group,
exactly one shared scale byte. The **scales stay ND** -- the production hook only
*views* them (reshape + transpose) -- so they are out of scope here.

Two traps, both paid for on card:

* ``Kb % 16 == 0`` is a *precondition* of the formula, not a detail. A shape that
  is not a multiple of C0 gets padded storage and the closed form stops holding,
  which is why :func:`permute_nd_to_nz` asserts it rather than producing
  plausible nonsense.
* an already-contiguous offset slice ``nd[i:i+c].contiguous()`` is a **no-op**
  (same storage, non-zero offset) and the cast then dies with ``setStorage out of
  bounds``. Anything handed to the cast must be a ``.clone()``. This module
  allocates its own case tensors, so it never hits it; whoever extends it will.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.logger import logger

#: C0, in bytes, for the FRACTAL_NZ layout the fp4 weight cast produces.
NZ_C0 = 16

#: ``format`` argument the production hook passes to ``npu_format_cast``. 29 is
#: what it ASKS for; ``get_npu_format`` on the result reports **50**
#: (FRACTAL_NZ_C0_16), so a check that pins 29 on the output goes red on correct
#: code. See ``stream_io`` module docstring.
NZ_CAST_FORMAT = 29

#: ACL_MEMCPY_DEVICE_TO_HOST.
_ACL_D2H = 2

#: What the self-check casts: the model's REAL ``hidden`` and
#: ``moe_intermediate``, with only the expert count cut to 2. So ``N`` and ``Kb``
#: are production's exactly -- w13 ``[2, 4608, 2560]``, w2 ``[2, 5120, 1152]``,
#: 160 and 72 gidx -- and the check sees the same byte geometry the CPU kernel
#: will read, at the same scale.
#:
#: Shrinking ``Kb`` instead would be cheaper and would miss the interesting
#: failure: a geometry that changes only past some tile count, or starts padding
#: at a given size, is invisible at Kb=160 and fatal at Kb=2560. Only ``E`` is
#: safe to cut, the expert axis being a plain concatenation (``e*(N*Kb)``,
#: verified for E > 1), and E=2 still exercises it.
#:
#: Footprint: 22.5 MiB (w13) + 11.25 MiB (w2) of ND and the same again of NZ,
#: ~67.5 MiB at the peak, freed before the function returns. The whole LAYER
#: would be 1.75 GiB -- that is E=384, which the self-check does not need.
SELFCHECK_EXPERTS = 2
SELFCHECK_HIDDEN = 5120
SELFCHECK_INTERMEDIATE = 2304
SELFCHECK_GROUP_SIZE = 32


class NZGeometryMismatch(RuntimeError):
    """CANN's NZ byte geometry is not the one this version assumes. Do not start."""


def nz_storage_bytes(num_experts: int, N: int, Kb: int) -> int:
    """Expected NZ storage size: the layout is dense, no padding."""
    return int(num_experts) * int(N) * int(Kb)


def permute_nd_to_nz(nd_bytes: np.ndarray, num_experts: int, N: int, Kb: int, c0: int = NZ_C0) -> np.ndarray:
    """Host-side ND -> NZ byte permutation, by reshape/transpose only.

    ``nd_bytes`` is the ND buffer as a flat (or ``[E, N, Kb]``) uint8 array; the
    result is the flat NZ buffer. No python loop: the whole permutation is one
    view change plus one copy, which is what keeps the self-check inside its
    millisecond budget.

    How it is a pure reshape: the destination is C-order ``buf[E][Kb/c0][N][c0]``
    and ``buf[e, g, n, r] == nd[e, n, g*c0 + r]``. So view the source as
    ``[E, N, Kb/c0, c0]`` -- splitting the k axis into (block, offset-in-block),
    which is free, both axes being contiguous -- and swap the N axis with the
    block axis. ``ascontiguousarray`` then materialises exactly the NZ order.
    """
    if Kb % c0:
        raise ValueError(f"Kb={Kb} is not a multiple of C0={c0}: refusing to permute (see nz_offset)")
    flat = np.ascontiguousarray(nd_bytes).reshape(-1)
    if flat.dtype != np.uint8:
        raise ValueError(f"ND bytes must be uint8, got {flat.dtype}")
    want = nz_storage_bytes(num_experts, N, Kb)
    if flat.size != want:
        raise ValueError(f"ND buffer has {flat.size} bytes, expected E*N*Kb = {want}")
    src = flat.reshape(num_experts, N, Kb // c0, c0)
    return np.ascontiguousarray(src.transpose(0, 2, 1, 3)).reshape(-1)


# ---------------------------------------------------------------------------
# Deterministic content
# ---------------------------------------------------------------------------


#: Deterministic synthetic bytes for the self-check's ND side. splitmix64 of the
#: byte index, so the pattern is reproducible across runs and machines and has no
#: structure that could accidentally agree with a wrong permutation.
#:
#: 🔴 This used to be a SEPARABLE hash -- two mix tables XORed, low and high
#: halves of the offset -- so that any subset of offsets could be evaluated
#: without materialising the rest. That property existed for one caller: the
#: chunked comparison, which read a gidx slice at a time. That arm was deleted on
#: 2026-11-21 as an unused benchmark arm that had lost its own measurement, and
#: with it the only use of `at()`. What is left needs whole buffers only, so the
#: separable form is gone too: ~45 lines, two arrays, a pinned salt and a
#: docstring about random access that nothing wanted.
def _content_bytes(total: int) -> np.ndarray:
    """``total`` bytes, byte i = splitmix64(i + 1) & 0xFF."""
    h = (np.arange(total, dtype=np.uint64) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    return (h & np.uint64(0xFF)).astype(np.uint8)


def device_bytes(t: Any) -> np.ndarray:
    """The raw bytes of a device tensor's whole storage, host side.

    ``.cpu()`` is not an option: on an internal-format tensor it converts back to
    ND on the way out, which is precisely the permutation under test. So the
    storage is copied with ``acl.rt.memcpy`` off ``data_ptr()`` -- the only read
    channel that sees the bytes as the card holds them.

    Deliberately NOT ``acl.rt.host_register_v2`` + ``host_get_device_pointer``:
    that API has taken two cards out in this container. A plain D2H memcpy into
    ordinary (unpinned) host memory is all this needs.

    The destination is a numpy array the memcpy writes into directly. Going
    through a ``ctypes`` buffer and ``np.frombuffer(bytes(buf))`` instead costs a
    second 23 MB allocation and a full copy of it, which on a host whose page
    cache is under pressure is the most expensive thing in the self-check.
    """
    out = np.empty(storage_bytes(t), dtype=np.uint8)
    torch.npu.synchronize()
    _d2h_into(storage_ptr(t), out, 0)
    return out


def storage_bytes(t: Any) -> int:
    """Size of the tensor's whole storage, in bytes."""
    return int(t.untyped_storage().nbytes())


def storage_ptr(t: Any) -> int:
    """Device address of the start of the tensor's storage."""
    storage = t.untyped_storage()
    try:
        return int(storage.data_ptr())
    except Exception:  # noqa: BLE001 - internal format refuses metadata
        return int(t.data_ptr())


def _d2h_into(ptr: int, out: np.ndarray, byte_offset: int) -> None:
    """Copy ``out.nbytes`` from device ``ptr + byte_offset`` into ``out``.

    No synchronisation of its own: the caller syncs ONCE before a run of these.
    Two ``torch.npu.synchronize()`` calls per slice would cost more than the
    copies -- the self-check makes 464 of them -- and ``acl.rt.memcpy`` is itself
    blocking, so the bytes are there when it returns.
    """
    import acl  # noqa: PLC0415 - lazy: the CANN python bindings, like torch_npu

    nbytes = int(out.nbytes)
    ret = acl.rt.memcpy(out.ctypes.data, nbytes, ptr + byte_offset, nbytes, _ACL_D2H)
    if ret != 0:
        raise RuntimeError(f"acl.rt.memcpy D2H of {nbytes} B at +{byte_offset} returned {ret}")


# ---------------------------------------------------------------------------
# The production path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One weight the check casts: the parameter it stands for, and its ND geometry."""

    name: str
    param: str
    num_experts: int
    N: int
    Kb: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.num_experts, self.N, self.Kb)


def selfcheck_cases(
    num_experts: int = SELFCHECK_EXPERTS, hidden: int = SELFCHECK_HIDDEN, intermediate: int = SELFCHECK_INTERMEDIATE
) -> list[Case]:
    """w13 and w2, derived the way production derives them.

    ``w13`` is ``[E, 2*intermediate, hidden/2]`` and ``w2`` is
    ``[E, hidden, intermediate/2]`` (the /2 is fp4 packing: two values per byte).
    Both Kb must be multiples of C0 -- the precondition of the whole formula --
    so the dimensions are checked here rather than producing a self-check that
    passes on a shape the production layout never has.

    The defaults are the real ``hidden``/``intermediate`` with E=2. The arguments
    exist so a unit test can run a small shape off-card-sized (a UT cannot spend
    67 MiB of HBM per case), NOT so the start-up check can be made cheaper by
    shrinking ``Kb``: see the note on :data:`SELFCHECK_HIDDEN`.
    """
    cases = [
        Case("w13", "w13_weight", num_experts, 2 * intermediate, hidden // 2),
        Case("w2", "w2_weight", num_experts, hidden, intermediate // 2),
    ]
    for c in cases:
        if c.Kb % NZ_C0:
            raise ValueError(
                f"nz self-check case {c.name}: Kb={c.Kb} is not a multiple of C0={NZ_C0}. The "
                f"shapes must keep that property (hidden % {2 * NZ_C0} == 0 and "
                f"intermediate % {2 * NZ_C0} == 0); got hidden={hidden}, intermediate={intermediate}."
            )
        if c.N % NZ_C0:
            raise ValueError(f"nz self-check case {c.name}: N={c.N} is not a multiple of {NZ_C0}.")
    return cases


def production_post_load(module: Any, quant_method: Any = None) -> None:
    """Run the REAL ``process_weights_after_loading`` on ``module``.

    The point of the self-check is that it exercises the path production takes,
    so the cast arguments are not written out here: they live in
    ``AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading``
    (``quantization/methods/w4a8/w4a8_mxfp4.py``) and this calls that function.

    ``quant_method`` is the layer's own method object when the caller has one;
    kt patches it (``layer.attach_after_init``), and the wrapper must NOT run
    here -- ``after_weights_loaded`` would rebind a live layer's routing tables
    to this throwaway module -- so the stamped original is unwrapped, exactly as
    ``stream_io.default_post_load`` does. Without a ``quant_method`` the method
    is imported and called unbound: it reads nothing off ``self``.
    """
    if quant_method is not None:
        hook = quant_method.process_weights_after_loading
        hook = getattr(hook, "_kt_offload_original", hook)
        hook(module)
        return
    from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDSDynamicFusedMoEMethod

    AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading(None, module)


def _npu_format(t: Any) -> int | None:
    """``get_npu_format``, reported not enforced.

    The hook asks for 29 and the result reports 50; pinning either would be a
    criterion that goes red on correct code. It is in the failure message because
    it is the first thing to look at when the bytes stop matching.
    """
    try:
        import torch_npu  # noqa: PLC0415 - lazy, as above

        return int(torch_npu.get_npu_format(t))
    except Exception:  # noqa: BLE001 - internal format refuses metadata
        return None


class _SelfCheckModule:
    """The four attributes ``process_weights_after_loading`` rewrites.

    A plain object, not an ``nn.Module``: the hook only does ``x.data = ...`` on
    four attributes and reads ``.shape``, and a bare object keeps the self-check
    importable without touching the layer machinery. ``_Slot`` gives ``.data``
    assignment and forwards ``.shape`` the way a ``Parameter`` would.
    """

    class _Slot:
        def __init__(self, t: Any) -> None:
            self.data = t

        @property
        def shape(self) -> Any:
            return self.data.shape

    def __init__(self, **tensors: Any) -> None:
        for name, t in tensors.items():
            setattr(self, name, _SelfCheckModule._Slot(t))


# ---------------------------------------------------------------------------
# Chunked upload and comparison
# ---------------------------------------------------------------------------


#: Target size of one host staging chunk for the ND upload, bytes. The slice the
#: comparison works on is ``N * C0`` (72 KiB for w13, 80 KiB for w2), so this is
#: the same order: the peak host allocation of the whole self-check is a handful
#: of these, not the 22 MiB of a whole weight.
def _upload_nd(case: Case, content: np.ndarray, device: Any) -> Any:
    """The case's ND bytes on the device.

    Build the whole ``[E, N, Kb]`` array on the host and copy it in one H2D --
    a 22 MiB host allocation per case. A chunked variant that filled the device
    buffer from 128 KiB staging chunks was deleted on 2026-11-21: nothing ever
    asked for it.
    """
    total = nz_storage_bytes(case.num_experts, case.N, case.Kb)
    return torch.from_numpy(content[:total].reshape(case.shape)).to(device)


@dataclass
class _Mismatch:
    """Where the NZ bytes first stopped agreeing, and how many disagree in all."""

    first: int
    expected: int
    actual: int
    total: int


def _compare_nz(
    out: Any, case: Case, content: np.ndarray, permute: Callable[..., np.ndarray] | None
) -> _Mismatch | None:
    """Compare every NZ byte of ``out`` against the formula.

    Two implementations, same verdict and same reported numbers (checked by
    diffing the failure messages of four wrong formulas, byte for byte):

    Permute the whole ND buffer on the host, read the whole NZ buffer back in
    one D2H, one ``np.array_equal``. A second implementation that walked a gidx
    slice at a time was deleted on 2026-11-21: nothing ever passed
    ``chunked=True``, and its own docstring recorded it as 5.5x slower than this
    on a healthy host.
    """
    total = nz_storage_bytes(case.num_experts, case.N, case.Kb)
    nd = content[:total].reshape(case.shape)
    want = (permute or permute_nd_to_nz)(nd, case.num_experts, case.N, case.Kb)
    got = device_bytes(out)
    if np.array_equal(got, want):
        return None
    diff = np.flatnonzero(got != want)
    first = int(diff[0])
    return _Mismatch(first, int(want[first]), int(got[first]), int(diff.size))


def verify_nz_geometry(
    device: Any = None,
    *,
    quant_method: Any = None,
    cases: Sequence[Case] | None = None,
    post_load: Callable[..., None] | None = None,
    permute: Callable[..., np.ndarray] | None = None,
    group_size: int = SELFCHECK_GROUP_SIZE,
) -> dict[str, Any]:
    """Cast a known two-expert tensor through production and compare NZ bytes.

    Every one of the 35 389 440 bytes is compared, either way.

    The whole ND buffer is permuted on the host and compared in one D2H. A
    chunked variant and its timing write-up were deleted on 2026-11-21: it was
    a benchmark arm that lost (5.5x slower on a healthy host, by its own
    measurement) and nothing ever selected it.
    every reported number are identical, which is checked rather than assumed by
    diffing the failure messages of four wrong formulas byte for byte across both
    paths.

    Returns a report dict (per-case shapes, byte counts, elapsed ms) on success.
    Raises :class:`NZGeometryMismatch` on the first differing byte -- the caller
    must let that propagate: a start-up that continues past it computes wrong
    numbers on the CPU half for the life of the process.

    ``permute`` is the test injection point: ``None`` (production) evaluates the
    formula per slice, and a whole-buffer function such as
    :func:`permute_nd_to_nz` or a deliberately wrong one is used as the
    expectation instead. ``post_load`` is the same idea for the hook.
    """
    cases = list(cases) if cases is not None else selfcheck_cases()
    runner = post_load if post_load is not None else production_post_load
    by_param = {c.param: c for c in cases}
    missing = [p for p in ("w13_weight", "w2_weight") if p not in by_param]
    if missing:
        raise ValueError(
            f"the NZ self-check must cover both production shapes; no case for {missing}. "
            "A check that only exercises w13 cannot see a geometry change that depends on Kb."
        )
    if device is None:
        device = torch.device("npu", torch.npu.current_device())

    t0 = time.perf_counter()
    content = {c.name: _content_bytes(nz_storage_bytes(c.num_experts, c.N, c.Kb)) for c in cases}
    # Fresh allocations, so nothing here is an offset slice of someone else's
    # storage: an already contiguous ``nd[i:i+c].contiguous()`` is a no-op and
    # the cast then fails with ``setStorage out of bounds`` (use ``.clone()`` if
    # you ever slice).
    tensors: dict[str, Any] = {}
    for c in cases:
        tensors[c.param] = _upload_nd(c, content[c.name], device)
    # The hook also rewrites the two scales (reshape + transpose of the SAME
    # storage, no cast); they must be present and their k must be even, or the
    # production function raises before reaching the weights.
    for param in ("w13_weight", "w2_weight"):
        c = by_param[param]
        # k_scale = K_fp4 / group_size = (2*Kb) / group_size, and the hook needs
        # it even (it reshapes k -> (k//2, 2)).
        k_scale = (2 * c.Kb) // group_size
        if (2 * c.Kb) % group_size or k_scale % 2:
            raise ValueError(
                f"nz self-check case {c.name}: K_fp4={2 * c.Kb} gives {k_scale} scale columns for "
                f"group_size={group_size}; the production hook needs a whole, even number of them."
            )
        tensors[f"{param}_scale"] = torch.full((c.num_experts, c.N, k_scale), 127, dtype=torch.uint8, device=device)

    module = _SelfCheckModule(**tensors)
    runner(module, quant_method)
    torch.npu.synchronize()

    report: dict[str, Any] = {"cases": [], "group_size": group_size}
    for c in cases:
        out = getattr(module, c.param).data
        want_bytes = nz_storage_bytes(c.num_experts, c.N, c.Kb)
        got_bytes = storage_bytes(out)
        fmt = _npu_format(out)
        if got_bytes != want_bytes:
            raise NZGeometryMismatch(
                f"NZ self-check {c.name}: the cast of ND {c.shape} produced {got_bytes} B of storage, "
                f"expected E*N*Kb = {want_bytes} B (npu_format={fmt}). The layout is no longer the dense, "
                f"unpadded buffer this version assumes -- it looks padded or re-tiled.\n"
                "CANN 的 NZ 几何与本版本假设不符，拒绝启动。 "
                "(CANN's NZ geometry does not match this build's assumption; refusing to start.)"
            )
        bad = _compare_nz(out, c, content[c.name], permute)
        if bad is not None:
            e, rest = divmod(bad.first, c.N * c.Kb)
            gidx, rest2 = divmod(rest, c.N * NZ_C0)
            n, r = divmod(rest2, NZ_C0)
            raise NZGeometryMismatch(
                f"NZ self-check {c.name} FAILED: ND shape {c.shape} (E={c.num_experts}, N={c.N}, "
                f"Kb={c.Kb}, C0={NZ_C0}, {c.Kb // NZ_C0} gidx), npu_format={fmt}, "
                f"{want_bytes} B compared, {bad.total} differ.\n"
                f"  first mismatch at NZ byte offset {bad.first} "
                f"(= expert {e}, gidx {gidx}, n {n}, byte-in-C0 {r} under this version's formula)\n"
                f"  expected 0x{bad.expected:02x} ({bad.expected}), "
                f"actual 0x{bad.actual:02x} ({bad.actual})\n"
                f"  assumed geometry: dst = e*(N*Kb) + (kb//{NZ_C0})*(N*{NZ_C0}) + n*{NZ_C0} + (kb%{NZ_C0})\n"
                "CANN 的 NZ 几何与本版本假设不符，拒绝启动。 "
                "(CANN's NZ geometry does not match this build's assumption; refusing to start. "
                "The CPU MoE kernel reads these same bytes, so continuing would not raise -- it would "
                "compute wrong numbers.)"
            )
        report["cases"].append(
            {
                "name": c.name,
                "param": c.param,
                "nd_shape": c.shape,
                "bytes": want_bytes,
                "gidx": c.Kb // NZ_C0,
                "npu_format": fmt,
                "slices": c.num_experts * (c.Kb // NZ_C0),
                "slice_bytes": c.N * NZ_C0,
            }
        )
        del out
        content.pop(c.name, None)

    # The ND inputs, the NZ outputs and the scales all go now: this runs on the
    # start-up path, before the KV-cache profile measures free HBM, so leaving
    # ~67 MiB held would shrink the cache by it.
    tensors.clear()
    del module
    torch.npu.empty_cache()

    report["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
    report["c0"] = NZ_C0
    report["total_bytes"] = sum(x["bytes"] for x in report["cases"])
    return report


_VERIFIED: dict[str, Any] | None = None


def verify_nz_geometry_once(device: Any = None, **kw: Any) -> dict[str, Any]:
    """:func:`verify_nz_geometry`, at most once per process.

    The geometry is a property of the CANN build, not of a layer, so running it
    on each of 58 MoE layers buys nothing and costs 58x. A failure is NOT cached:
    it propagates every time, so a caller that swallows the first one still cannot
    get a quiet start-up.
    """
    global _VERIFIED
    if _VERIFIED is None:
        _VERIFIED = verify_nz_geometry(device, **kw)
        # Log INSIDE the `is None` branch: once per process, on the run that
        # actually did the work. A failure never reaches here (it propagates),
        # so this line appearing means the geometry was checked AND matched.
        log_pass(_VERIFIED)
    return _VERIFIED


#: Config switches that make the CPU kernel and the NPU read the SAME NZ bytes,
#: and therefore depend on this geometry being right. Any one of them on forces
#: the self-check on.
#:
#: 🔴 This tuple held only ``stream_layer_buffer`` until 2026-11-21, while the
#: route that actually has kt read these bytes in place is ``kt_reads_bank``.
#: So in the shipped configuration -- ``bank_dir`` + ``kt_reads_bank``, no
#: ``stream_layer_buffer`` -- ``check_enabled`` returned False and this guard
#: NEVER RAN: the logs of the 2026-11-21 measurement say
#: ``nz_geometry: self-check off`` forty times and carry no PASS line. The guard
#: was written for exactly that route's silent failure mode and was the one thing
#: not protecting it. Both switches below are bare-NZ routes:
#:   * ``bank_dir``      -- streaming writes host-permuted bytes into an
#:                          NZ-tagged buffer and ``npu_format_cast`` no longer runs;
#:   * ``kt_reads_bank`` -- the CPU kernel reads those same bytes in place.
NZ_DEPENDENT_SWITCHES = ("bank_dir", "kt_reads_bank")


def log_pass(report: Mapping[str, Any]) -> None:
    """One INFO on the PASS path, so "no log line" stops being ambiguous.

    Without it a correct run and a skipped guard look identical, and this
    guard's whole purpose is to turn a silent CANN geometry change into a
    start-up failure. The observed format is REPORTED, never asserted: the
    hook asks for 29 and ``get_npu_format`` on the result reports 50
    (FRACTAL_NZ_C0_16), so pinning either number would go red on correct
    behaviour -- see :data:`NZ_CAST_FORMAT`.
    """
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME") or "unknown"
    cases = list(report.get("cases") or ())
    shapes = (
        "; ".join(
            f"{c.get('name')}({c.get('param')}) nd={tuple(c.get('nd_shape') or ())} "
            f"gidx={c.get('gidx')} fmt={c.get('npu_format')}"
            for c in cases
        )
        or "none"
    )
    logger.info(
        "kt_offload nz_geometry: PASS -- FRACTAL_NZ byte geometry matches the assumed "
        "dst = e*(N*Kb) + (kb//16)*(N*16) + n*16 + (kb%%16). %d case(s) compared "
        "byte-for-byte over %d B, C0=%s, %.1f ms; %s; hook asks format %s; cann=%s",
        len(cases),
        int(report.get("total_bytes") or 0),
        report.get("c0"),
        float(report.get("elapsed_ms") or 0.0),
        shapes,
        NZ_CAST_FORMAT,
        os.path.basename(str(cann)),
    )


def check_enabled(cfg: Any) -> bool:
    """Should the self-check run? Resolved in ONE place, like ``switches.resolve``.

    OFF by default. Forced ON by any switch in :data:`NZ_DEPENDENT_SWITCHES`,
    and ``verify_nz_geometry=true`` turns it on by itself (for a CANN-upgrade
    smoke test on a deployment that does not use those routes yet).

    Note what is deliberately NOT possible: setting ``verify_nz_geometry=false``
    while a dependent switch is on does not disable the check. A route whose
    correctness rests on this geometry must not be able to switch off its own
    guard -- to get rid of the check, turn the route off.
    """
    if any(bool(getattr(cfg, name, False)) for name in NZ_DEPENDENT_SWITCHES):
        return True
    return bool(getattr(cfg, "verify_nz_geometry", False))
