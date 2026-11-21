# SPDX-License-Identifier: Apache-2.0
"""Stage 1: accumulate the streamed chunks into ONE whole-layer NZ buffer.

WHY
  Today every chunk's ``npu_format_cast`` output is a NEW tensor that is dropped
  when the next chunk replaces it. That is what forces the layer to be split at
  all: ND + NZ of a whole layer is 13.45 GiB against an 11.41 GiB budget, so the
  layer travels in 12 pieces of 32 experts. It also means the GEMM's weight
  address changes every chunk (no graph capture) and the folded swap needs its
  own 1.75 GiB whole-layer ND staging to collect the chosen rows.

  Measured on card (2026-11-11), three facts make a whole-layer NZ buffer legal:
    * a per-expert cast is byte-identical to that expert's slice of a whole-cast,
      and the byte count is conserved exactly (no NZ padding at these shapes);
    * ``acl.rt.memcpy_async`` into ``bank_base + e*PER`` hits exactly slot e and
      leaves its neighbours byte-identical;
    * the GEMM accepts a NZ-tagged buffer whose bytes we wrote ourselves, and
      the result is bitwise equal to the cast tensor's.

WHAT THIS MODULE DOES NOT DO
  It does not change the arithmetic. ``H2'`` on card: the whole-layer form with a
  384-long group list in which only this chunk's 32 groups carry tokens is
  **bitwise equal** (``maxabsdiff = 0``) to today's per-chunk form. The 352 empty
  groups cost +3.6% of the chunk's GEMM call (0.2938 vs 0.2836 ms), i.e. about
  0.13 ms per layer against 228 ms of host read -- and they are hidden behind it.

TWO THINGS THE PLATFORM FORCED
  1. **The layer buffer must be passed WHOLE.** A NZ tensor's format descriptor
     belongs to the *storage*, so ``layer_nz[32:64]`` still presents groupNum=384
     to the operator, while a second check in the same call reads 32 from the
     view's shape -- the two never agree, so a slice is unusable as a weight
     (561103 / 161002). Hence ``layer_tables()`` below: an IDENTITY slot table
     plus a keep mask, over all ``num_experts`` slots.
  2. **One ``torch.npu.synchronize()`` before each raw write.** With
     ``TASK_QUEUE_ENABLE != 0`` torch_npu submits ops through a host-side queue
     consumed by another thread, so a ``memcpy_async`` submitted directly from
     the Python thread can land AHEAD of the cast that produced its source
     (measured: 12/100 wrong, always the last slot). The sync is nearly free
     here: the device has 1-2 ms of cast against 19 ms of host gather per chunk,
     so it is already waiting. See ``mxfp4_fused_op.py`` for the same finding.

Origin: dsv41 stream-prefill, stage 1.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.kt_offload.stream_chain import (
    SLOT_WEIGHT_ATTRS,
    ChainCoverage,
    ChunkSlot,
    StreamChainResult,
    run_chunk,
)
from vllm_ascend.kt_offload.stream_prefill import (
    ChunkPlan,
    ChunkTables,
    assert_eager_only,
    check_chunk_plan,
)

#: Bytes read back from the head and tail of each written slot range to prove the
#: write landed. 0 disables the probe. Small on purpose: this catches a failed or
#: partial copy, which is the failure the raw path adds; it is not a checksum.
DEFAULT_PROBE_BYTES = 256

#: Only these two are internal-format (NZ) and need the raw byte path. The two
#: scale tensors stay ND row-major -- production only *views* them -- so an
#: ordinary ``Tensor.copy_`` works and is stream-ordered by torch itself.
NZ_ATTRS = ("w13_weight", "w2_weight")
ND_ATTRS = ("w13_weight_scale", "w2_weight_scale")


class StreamLayerError(RuntimeError):
    """The layer buffer is not in the state the chain is about to rely on."""


# ---------------------------------------------------------------------------
# The layer-wide routing tables
# ---------------------------------------------------------------------------


def layer_tables(
    plan: ChunkPlan,
    chunk_index: int,
    *,
    device: torch.device | str = "cpu",
    id_dtype: torch.dtype = torch.int32,
    weight_dtype: torch.dtype = torch.float32,
) -> ChunkTables:
    """``ChunkTables`` for chunk ``k`` over a whole-layer buffer.

    Two differences from ``stream_prefill.chunk_tables``:

    * the chunk's OWN experts sit at their own buffer position (slot id ==
      logical expert id), because the buffer holds every expert of the layer.
      Every OTHER route is sent to one single already-written slot, exactly as
      ``stream_prefill.chunk_tables`` sends them to dummy slot 0;
    * ``expert_ids`` is every expert, not the chunk's 32. ``run_chunk`` indexes
      it by *slot* to name the experts a route reached, so with an identity table
      it must be the identity list or the names come out wrong. The caller
      restores the chunk's own 32 on the returned ``ChunkRun`` for accounting --
      see ``run_streamed_layer_buffered``.

    The keep mask still selects only this chunk's experts, so a route to an
    expert of another chunk is zeroed; the slot table then puts that zeroed row
    on the dummy.

    🔴 The dummy must be a slot THIS forward has already written, and it must be
    one of this chunk's own: the whole-layer buffer still holds the PREVIOUS
    layer's bytes in every slot chunk ``k`` has not reached yet, and unloaded
    weights can be NaN (NaN * 0 is NaN -- ``placement.py`` and
    ``stream_prefill.ChunkTables`` argue the same point). ``ids[0]`` is written
    by ``provider.slot_for(k, ...)`` before ``run_chunk`` is called, so it is
    the natural choice. Slot 0 would NOT be safe here for k > 0 only by luck of
    chunk 0 being written first -- using this chunk's own first expert makes the
    guarantee local instead of depending on write order.

    🔴 Why this is not the identity table it used to be: an identity slot table
    leaves every one of the layer's experts as a NON-EMPTY group, because
    ``split_routing`` only zeroes the WEIGHT of an out-of-chunk route, it does
    not stop that row being dispatched to that expert's group. ``dispatch_chunk``
    then calls ``npu_moe_init_routing`` with ``expert_num = num_slots``, so all
    384 groups receive rows and the GEMM reads all 384 experts' weights --
    12x the weight traffic of one chunk. Measured on card: 7.20 ms per chunk
    against 4.11 ms for the per-chunk form, i.e. +75%, which is +37 ms per layer
    and accounted for the whole of stage 1's end-to-end TTFT regression.
    Declared-but-empty groups are nearly free (< 0.15 us each); what costs is a
    group that RECEIVES A ROW, at roughly 0.52 us per MB of that expert's
    weights. With this table the chunk again touches only its own 32 experts,
    measured at 4.18 ms -- the same as the per-chunk form.
    """
    if not 0 <= chunk_index < plan.num_chunks:
        raise IndexError(f"chunk {chunk_index} out of range (plan has {plan.num_chunks})")
    n = plan.num_experts
    ids = plan.chunks[chunk_index]
    if not ids:
        raise StreamLayerError(f"chunk {chunk_index} is empty; there is no slot to send dummy rows to")
    # Out-of-chunk routes go to this chunk's first expert -- a slot this forward
    # has already written. NOT the identity: see the docstring.
    slot = torch.full((n,), int(ids[0]), dtype=torch.int64)
    keep = torch.zeros(n, dtype=torch.float32)
    for e in ids:
        slot[e] = int(e)
        keep[e] = 1.0
    return ChunkTables(
        chunk_index=chunk_index,
        expert_ids=tuple(range(n)),
        safe_slot_table=slot.to(device=device, dtype=id_dtype),
        keep_table=keep.to(device=device, dtype=weight_dtype),
    )


# ---------------------------------------------------------------------------
# The buffer
# ---------------------------------------------------------------------------


@dataclass
class LayerWriteLedger:
    """Which (layer, chunk) pairs were written into the buffer since ``begin``.

    Host-side bookkeeping, not a checksum: it catches a chunk whose write never
    ran, which is the failure mode that would otherwise feed the GEMM the
    PREVIOUS layer's bytes for those slots -- with tokens in those groups, so
    they are read. Plausible numbers, wrong answer.
    """

    layer: int | None = None
    written: set[int] = field(default_factory=set)
    bytes_written: int = 0

    def begin(self, layer: int) -> None:
        self.layer, self.written, self.bytes_written = layer, set(), 0

    def note(self, chunk_index: int, nbytes: int) -> None:
        if chunk_index in self.written:
            raise StreamLayerError(
                f"chunk {chunk_index} written twice into the layer buffer: the second write would "
                "silently replace the first and the coverage check cannot see it"
            )
        self.written.add(chunk_index)
        self.bytes_written += nbytes

    def violations(self, plan: ChunkPlan) -> list[str]:
        missing = sorted(set(range(plan.num_chunks)) - self.written)
        if missing:
            return [
                f"chunks {missing[:8]} were never written into the layer buffer: their slots "
                f"still hold whatever was there before (the previous layer, or the poison)"
            ]
        return []


class LayerSlotProvider:
    """Wraps a ``StreamSlotProvider``: same I/O, plus accumulation into one buffer.

    The inner provider keeps doing exactly what it does today (host gather into
    pinned, H2D into the reused ND buffers, the production post-load hook). This
    class adds the step after it: copy the chunk's bytes into slot range
    ``[k*chunk, k*chunk+len)`` of a whole-layer buffer, and hand the chain the
    WHOLE buffer instead of the chunk.

    Everything not overridden is delegated, so ``set_layer``, ``served_nd``,
    ``layer``, ``device``, ``check_coverage`` and the fold's hook keep working on
    the inner provider -- the folded swap still reads the chunk's ND bytes, which
    is what it wants.
    """

    def __init__(
        self,
        inner: Any,
        *,
        num_experts: int,
        chunk_experts: int,
        nd_shapes: Mapping[str, Sequence[int]],
        probe_bytes: int = DEFAULT_PROBE_BYTES,
        sync_before_write: bool = True,
    ):
        """``nd_shapes``: per-expert ND shape of each of the four params.

        Taken from the caller rather than inferred from a chunk's slot, because
        a slot's weights are already cast AND transposed -- recovering the
        pre-transpose shape from a transposed view is exactly the kind of guess
        this module should not make. ``stream_io.chunk_param_shapes(quant_method,
        1, intermediate_size, hidden_size, ...)`` is where production has them.
        """
        self.inner = inner
        self.num_experts = int(num_experts)
        self.chunk_experts = int(chunk_experts)
        self.nd_shapes = {k: tuple(v) for k, v in nd_shapes.items()}
        missing = [n for n in SLOT_WEIGHT_ATTRS if n not in self.nd_shapes]
        if missing:
            raise StreamLayerError(f"nd_shapes is missing {missing}: all four params need one")
        self.probe_bytes = int(probe_bytes)
        self.sync_before_write = bool(sync_before_write)
        self.sync_scope = "device"
        self.ledger = LayerWriteLedger()
        self._slot: ChunkSlot | None = None
        self._nd_scale: dict[str, torch.Tensor] = {}
        self._base: dict[str, int] = {}
        self._per_expert: dict[str, int] = {}

    # -- delegation -------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Only reached for names this class does not define; keeps the wrapper
        # transparent for set_layer / served_nd / layer / device / close.
        return getattr(self.inner, name)

    # -- allocation -------------------------------------------------------
    def build(self, device: Any, dtypes: Mapping[str, torch.dtype]) -> ChunkSlot:
        """Allocate the whole-layer buffer THROUGH THE PRODUCTION TRANSFORM.

        The four buffers are created as ND and handed to
        ``StreamSlotProvider.format_nd``, i.e. the same
        ``process_weights_after_loading`` every chunk goes through. That is the
        only way to be sure the layer buffer has the layout and the NZ format
        TAG the operator expects -- the tag matters because the operator
        dispatches on it (a plain ND tensor holding the same NZ bytes is refused
        with 161002), and the layout must not be re-derived here.

        What survives the call:
          * the two weights become FRESH NZ allocations (the cast's output), so
            the ND weight buffers are dropped right after;
          * the two scales stay VIEWS of the ND buffers we pass in, so those are
            kept alive on this object -- and they are what the per-chunk scale
            copy writes into, which is why updating them updates the slot.
        """
        if self._slot is not None:
            return self._slot
        nd = {
            name: torch.zeros((self.num_experts, *self.nd_shapes[name]), dtype=dtypes[name], device=device)
            for name in SLOT_WEIGHT_ATTRS
        }
        slot = self.inner.format_nd(nd, self.num_experts)
        self._slot = slot
        for name in ND_ATTRS:
            self._nd_scale[name] = nd[name]  # the slot's scale is a view of this
        for name in SLOT_WEIGHT_ATTRS:
            t = getattr(slot, name)
            st = t.untyped_storage()
            if st.nbytes() % self.num_experts:
                raise StreamLayerError(
                    f"{name}: storage {st.nbytes()} B is not a multiple of {self.num_experts} "
                    "slots, so a per-slot byte offset is not defined"
                )
            # The VIEW must cover the WHOLE storage. Measured on card: the
            # operator reads the group count from the NZ format descriptor,
            # which belongs to the *storage*, while a second check in the same
            # call reads it from the view's shape -- so a tensor whose view is
            # smaller than its storage can never satisfy both (561103 one way,
            # 161002 the other, with contradictory messages). Today the buffer is
            # cast from a [num_experts, n, k] zeros tensor so the two agree; this
            # refuses the day someone allocates the bank with headroom and hands
            # the GEMM a sub-view of it.
            view_bytes = t.numel() * t.element_size()
            if view_bytes != st.nbytes() or t.storage_offset() != 0:
                raise StreamLayerError(
                    f"{name}: the view covers {view_bytes} B of a {st.nbytes()} B storage at offset "
                    f"{t.storage_offset()}. The layer buffer must own its whole storage: the operator "
                    "takes the group count from the storage's NZ descriptor and from the view's shape "
                    "in two different checks, and a partial view can never satisfy both."
                )
            self._base[name] = st.data_ptr()
            self._per_expert[name] = st.nbytes() // self.num_experts
            logger.info(
                "kt_offload stream_layer: %s %s storage=%d B (%d B/slot) fmt-tagged",
                name,
                tuple(t.shape),
                st.nbytes(),
                self._per_expert[name],
            )
        for name in NZ_ATTRS:
            del nd[name]  # the NZ output owns its own bytes
        return slot

    @property
    def layer_slot(self) -> ChunkSlot:
        if self._slot is None:
            raise StreamLayerError("the layer buffer has not been built: call build(device, dtypes) once")
        return self._slot

    # -- the write --------------------------------------------------------
    def _write_nz(
        self, name: str, first_slot: int, src: torch.Tensor, count: int, src_first: int = 0, src_per: int | None = None
    ) -> int:
        """Copy ``count`` slots into the layer buffer at ``first_slot``.

        ``src_first``/``src_per`` let the source start at a slot other than 0
        and carry its own per-slot size; ``src_per`` is asserted equal to the
        destination's, because both buffers come out of the same
        ``process_weights_after_loading`` and a mismatch means the record
        layouts have diverged, not that a copy can be rescaled. The whole-chunk
        call passes neither and is unchanged.
        """
        import acl  # device-only dependency, imported at the call site

        per = self._per_expert[name]
        if src_per is not None and int(src_per) != per:
            raise StreamLayerError(
                f"{name}: source holds {src_per} B per slot, the layer buffer {per} B. Both come "
                "from the same process_weights_after_loading, so this is a layout divergence, not "
                "a copy that can be rescaled"
            )
        nbytes = per * count
        # aclrtMemcpy/Async take destMax but have NO srcMax (acl_rt.h:2262/2309),
        # and destMax is the amount we PROMISE to write rather than the buffer's
        # remaining capacity -- so neither an over-read of `src` nor an
        # out-of-range slot is visible to the runtime. Measured, not assumed:
        # the same shape of over-read returns 0, leaves the card healthy and puts
        # nothing in dmesg, i.e. it surfaces as wrong numbers, never as an error.
        # `nbytes` is derived from the DESTINATION's per-slot size while `src`
        # contributes only a data_ptr(), so both bounds must be asserted here.
        if first_slot < 0 or first_slot + count > self.num_experts:
            raise StreamLayerError(
                f"{name}: slots [{first_slot}, {first_slot + count}) fall outside the "
                f"{self.num_experts}-slot layer buffer"
            )
        src_off = per * int(src_first)
        src_bytes = src.untyped_storage().nbytes() - src_off
        if src_off < 0 or src_bytes < nbytes:
            raise StreamLayerError(
                f"{name}: reading {nbytes} B from a source storage of only {src_bytes} B "
                f"(first_slot={first_slot} count={count} per={per}) -- aclrtMemcpy has no "
                f"srcMax, so this would be a silent out-of-bounds read, not an error"
            )
        if src_bytes != nbytes and src_first == 0 and src_per is None:
            # Not fatal: a source with headroom is legal. But for a WHOLE-chunk
            # write it means our model of the chunk layout is off somewhere, so
            # it must not pass in silence. A run out of a mixed chunk, or a slot
            # out of the resident parameter, has headroom by construction.
            logger.warning(
                "kt_offload stream_layer: %s source storage is %d B for a %d B write (first_slot=%d count=%d)",
                name,
                src_bytes,
                nbytes,
                first_slot,
                count,
            )
        dst = self._base[name] + first_slot * per
        stream = torch.npu.current_stream().npu_stream
        ret = acl.rt.memcpy_async(dst, nbytes, src.untyped_storage().data_ptr() + src_off, nbytes, 3, stream)
        if ret != 0:
            raise StreamLayerError(f"{name}: memcpy_async into slot {first_slot} returned {ret}")
        return nbytes

    def slot_for(self, chunk_index: int, expert_ids: Sequence[int]) -> ChunkSlot:
        ids = tuple(int(e) for e in expert_ids)
        # 🔴 The chunk arrives COMPLETE even when some of its experts were not
        # streamed: the inner provider fills those slots from the resident
        # parameter itself (stream_io.set_resident), so this copy stays the
        # whole-chunk copy it always was. Splitting it here as well would skip
        # the slots twice -- and the second skip has nothing to fill it.
        chunk = self.inner.slot_for(chunk_index, ids)
        if self._slot is None:
            raise StreamLayerError(
                "slot_for before build(): the layer buffer must be allocated once, "
                "at wiring time, not lazily on the first chunk of a forward"
            )
        count = len(ids)
        first = chunk_index * self.chunk_experts
        if first + count > self.num_experts:
            raise StreamLayerError(
                f"chunk {chunk_index} of {count} experts starts at slot {first}, past the layer's "
                f"{self.num_experts} slots: the chunk tiling and the buffer disagree"
            )
        if self.sync_before_write:
            # See the module docstring: the cast is a torch op and may still be
            # sitting in torch_npu's host-side queue when we submit the raw copy.
            # Device scope waits for EVERY stream on the card; stream scope waits
            # only for the one the cast went to. Both must flush the host queue
            # for the write to be ordered -- if "stream" does not, the symptom is
            # wrong numbers, not an error, so it is gated and defaults to device.
            if self.sync_scope == "stream":
                torch.npu.current_stream().synchronize()
            else:
                torch.npu.synchronize()
        moved = 0
        for name in NZ_ATTRS:
            moved += self._write_nz(name, first, getattr(chunk, name), count)
        nd = self.inner.served_nd(chunk_index, ids)
        for name in ND_ATTRS:
            # ND destination: an ordinary copy_ is valid and torch orders it.
            # The source must be the ND bytes, NOT the slot's 4-D transposed
            # view of them -- copying that would transpose the data.
            self._nd_scale[name][first : first + count].copy_(nd[name])
        self.ledger.note(chunk_index, moved)
        if self.probe_bytes:
            self._probe(chunk, first, count)
        return self._slot

    def _probe(self, chunk: ChunkSlot, first: int, count: int) -> None:
        """Head/tail bytes of the written range must equal the chunk's own.

        Small on purpose: this catches a failed or partial copy, which is the
        failure the raw path adds. It is not a checksum.
        """
        import ctypes

        import acl

        torch.npu.synchronize()
        for name in NZ_ATTRS:
            per = self._per_expert[name]
            n = min(self.probe_bytes, per)
            src_base = getattr(chunk, name).untyped_storage().data_ptr()
            for where, off in (("head", 0), ("tail", per * count - n)):
                got, want = (ctypes.c_char * n)(), (ctypes.c_char * n)()
                if (
                    acl.rt.memcpy(ctypes.addressof(got), n, self._base[name] + first * per + off, n, 2) != 0
                    or acl.rt.memcpy(ctypes.addressof(want), n, src_base + off, n, 2) != 0
                ):
                    raise StreamLayerError(f"{name}: probe read-back failed")
                if bytes(got) != bytes(want):
                    raise StreamLayerError(
                        f"{name}: the layer buffer does not hold the bytes written for slots "
                        f"[{first},{first + count}) ({where} {n} B differ). The copy was issued but "
                        "the GEMM would run on some other chunk's weights."
                    )


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_streamed_layer_buffered(
    *,
    quant_method,
    provider: LayerSlotProvider,
    plan: ChunkPlan,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_config,
    top_k: int,
    capture_sizes,
    min_tokens: int,
    slot_for: Any = None,
    resident_layer: torch.nn.Module | None = None,
    accum_dtype: torch.dtype = torch.float32,
    after_chunk: Callable[[int, Sequence[int]], None] | None = None,
    expected_expert_ids: Sequence[int] | None = None,
    strict: bool = True,
) -> StreamChainResult:
    """``run_streamed_layer`` over a whole-layer buffer. Same arithmetic.

    Signature mirrors ``stream_chain.run_streamed_layer`` so the call site picks
    a function and changes nothing else; ``slot_for`` is accepted and ignored
    (this runner goes through ``provider.slot_for``, which is the same inner
    provider plus the accumulation step).

    The only differences from ``run_streamed_layer``:
      * ``layer_tables`` instead of ``chunk_tables`` (identity slots + keep mask);
      * the provider hands back the WHOLE buffer every chunk, because a slice of
        a NZ tensor is not a legal weight -- see the module docstring;
      * the per-chunk ``ChunkRun`` gets its chunk's own expert ids put back, so
        ``ChainCoverage`` keeps judging per-chunk coverage exactly as today;
      * the write ledger refuses a layer in which some chunk never landed.
    """
    if strict:
        if expected_expert_ids is None:
            raise ValueError(
                "run_streamed_layer_buffered(strict=True) needs expected_expert_ids: a "
                "coverage check whose expectation is the plan under test cannot fail."
            )
        bad = check_chunk_plan(plan, expected_expert_ids)
        if bad:
            raise ValueError("stream_layer: the chunk plan does not cover what was expected: " + "; ".join(bad))
    # Same refusal as the per-chunk chain: this path is eager-only, and a token
    # count that is also a capture size means a graph would be captured over it.
    assert_eager_only(int(hidden_states.shape[:-1].numel()), capture_sizes, min_tokens, where="stream_layer")
    provider.ledger.begin(int(provider.layer))
    coverage = ChainCoverage()
    runs: list[Any] = []
    total: torch.Tensor | None = None
    for k in range(plan.num_chunks):
        tables = layer_tables(plan, k, device=topk_ids.device, id_dtype=topk_ids.dtype, weight_dtype=topk_weights.dtype)
        slot = provider.slot_for(k, plan.chunks[k])
        run = run_chunk(
            quant_method=quant_method,
            slot=slot,
            tables=tables,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            moe_config=moe_config,
            top_k=top_k,
            resident_layer=resident_layer,
        )
        # ``tables.expert_ids`` had to be the identity for run_chunk's slot->expert
        # naming; the accounting wants the chunk's own set back.
        run = dataclasses.replace(run, expert_ids=tuple(plan.chunks[k]))
        runs.append(run)
        coverage.add(run)
        contribution = run.output.to(accum_dtype)
        total = contribution if total is None else total + contribution
        if after_chunk is not None:
            after_chunk(k, plan.chunks[k])
    bad = provider.ledger.violations(plan)
    if bad:
        raise StreamLayerError("; ".join(bad))
    coverage.check(plan, strict=strict)
    if total is None:
        raise ValueError(f"{coverage.summary()} -- no chunk produced an output tensor")
    return StreamChainResult(output=total.to(hidden_states.dtype), coverage=coverage, runs=tuple(runs))


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def install_layer_buffer(
    inner: Any, *, num_experts: int, probe_bytes: int = DEFAULT_PROBE_BYTES, sync_scope: str = "device"
) -> LayerSlotProvider:
    """Wrap the ONE streaming provider and allocate the ONE layer buffer.

    Shapes and dtypes come off the inner provider's own ``shapes``
    (``stream_io.chunk_param_shapes``, i.e. the quant method's ``get_weight`` /
    ``get_dynamic_quant_param``), so this never re-derives a layout. Called once
    per MODEL, like the provider itself: one buffer per layer would be 40 x 6.75
    GiB, and ``set_layer`` rebinds the single one on every streamed forward.
    """
    shapes = getattr(inner, "shapes", None)
    if not shapes:
        raise StreamLayerError(
            "the streaming provider has no `shapes`: the layer buffer is built from the quant "
            "method's declared ND shapes (stream_io.chunk_param_shapes), not from a chunk's "
            "post-cast slot, whose weights are already transposed"
        )
    nd_shapes = {n: tuple(v[0][1:]) for n, v in shapes.items()}
    dtypes = {n: v[1] for n, v in shapes.items()}
    wrapper = LayerSlotProvider(
        inner,
        num_experts=int(num_experts),
        chunk_experts=int(inner.chunk_experts),
        nd_shapes=nd_shapes,
        probe_bytes=int(probe_bytes),
    )
    if sync_scope not in ("device", "stream"):
        raise StreamLayerError(f"sync_scope must be 'device' or 'stream', got {sync_scope!r}")
    wrapper.sync_scope = sync_scope
    wrapper.build(inner.device, dtypes)
    logger.info(
        "kt_offload stream_layer: layer buffer installed for %d experts (%d per chunk, probe %d B)",
        num_experts,
        inner.chunk_experts,
        probe_bytes,
    )
    logger.info(
        "kt_offload stream_layer: pre-write flush scope = %s (device = all streams, stream = only the cast's stream)",
        sync_scope,
    )
    return wrapper
