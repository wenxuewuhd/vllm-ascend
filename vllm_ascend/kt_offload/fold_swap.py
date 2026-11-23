# SPDX-License-Identifier: Apache-2.0
"""The folded swap-in: make the hot set resident on the streaming pass itself.

``resident_driver.py`` pulls the new resident weights off the host a SECOND
time. Here the streaming prefill already carries every expert past the device
chunk by chunk, so the chosen experts are copied out of each chunk as it goes by
and nothing is pulled twice.

Every forward of a prefill adds its ``bincount(topk_ids)`` into the layer's
running heat; only the LAST streaming forward turns that heat into a new
resident set and commits, so a K-chunk prompt stages and commits ONCE, on the
whole prompt's heat. Which forward that is cannot be known here -- see
``prefill_window`` and :meth:`ResidentFold._decides_now`.

* ``topk_ids`` is the MoE layer's INPUT: the layer's whole heat is known before
  any expert streams. Policy and refusals stay the driver's
  (``resident_driver.plan_from_heat``).
* bs=1 only: ``dynamic_resident`` requires ``max_num_seqs == 1`` (``config.py``)
  -- not a safety limit, but "accumulate per request, move once at the end" has
  no commit point when several requests prefill at once, and their heats average
  each other out.
* Staging is ND and whole-layer: ``[num_slots, ...]`` per parameter in the
  checkpoint's ND layout, allocated ONCE through the provider's device seam and
  reused by every layer. Row ``i`` holds ``new_ids[i]`` -- ascending logical id,
  the slot order ``LayerPlacement`` gives. Its bytes come from the provider's ND
  H2D buffers (``StreamSlotProvider.served_nd``), which still hold the chunk
  when ``after_chunk`` runs.
* One cast, one write: the filled staging goes through
  ``StreamSlotProvider.format_nd`` (the production
  ``process_weights_after_loading`` every chunk goes through) and on to the
  committer, whose weight writer here is :func:`fold_weight_write`.
* Every per-expert write into an NZ tensor is refused on card (``copy_`` refuses
  internal formats, an offset view trips ``storage_offset() == 0``,
  NZ->ND->write->NZ fails), and the one channel that works,
  ``torch_npu.copy_memory_``, ignores the view and writes the WHOLE storage;
  resident bytes cannot be read back to ND either. So the staging must be filled
  completely by this pass: a chosen expert that came by in no chunk has no
  source, and the fold raises ``FoldIncompleteError`` instead of writing zeros
  or the previous layer's rows into its slot.

``resident_swap.ResidentSwapCommitter`` is reused as is (stage order, ownership
ledger, roll-forward, post-commit checks). Its ``writer_busy`` guard is fed
:meth:`ResidentFold.layer_in_flight`, True from ``begin`` until the layer's last
chunk has been folded: it refuses a commit while THIS layer is still folding,
nothing more. A forward IS in flight when the fold commits, so it may never be
read as "no forward is running"; a second thread or stream, or two steps' Python
overlapping under async scheduling, is refused by ``OwnershipLedger``'s writer
identity (one (thread, stream) for its lifetime, one commit at a time).

The commit therefore happens at the end of THIS layer's streamed MoE, inside the
forward, not at the step boundary: the streamed forward never reads a layer's
resident weights, tables or kt mask (``maybe_stream_prefill`` skips
route/apply/merge, every expert comes from the slot) and no later layer reads
this layer's objects; it is eager only, the write enqueued after the layer's
GEMMs with no graph being captured (``begin`` refuses unless the capture
predicate says a positive ``False``); and parking until the step boundary would
hold one cast staging per layer, which does not fit the device memory budget.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.kt_offload import prefill_window
from vllm_ascend.kt_offload.placement import LayerPlacement
from vllm_ascend.kt_offload.resident_driver import (
    SLOT_WEIGHT_ATTRS,
    LayerPlan,
    TopKHeatPolicy,
    capture_state,
    pinned_resident_ids,
    plan_from_heat,
)
from vllm_ascend.kt_offload.resident_swap import (
    LayerSwapTargets,
    ResidentSwapCommitter,
    SwapReport,
    ZeroCoverageError,
)
from vllm_ascend.kt_offload.stream_io import ND_SCALE_ATTRS, NZ_WEIGHT_ATTRS

__all__ = [
    "FoldError",
    "FoldIncompleteError",
    "FoldCoverage",
    "FoldedLayer",
    "FoldReport",
    "ResidentFold",
    "fold_weight_write",
    "heat_from_topk",
]


class FoldError(RuntimeError):
    """The fold refuses: a precondition, a stale chunk, or a staging that moved under it."""


class FoldIncompleteError(FoldError):
    """A chosen expert never came by in any chunk: its staging row has no source."""


# ---------------------------------------------------------------------------
# The one write channel into a resident parameter
# ---------------------------------------------------------------------------


def _assert_owns_storage(t: torch.Tensor, what: str) -> None:
    nbytes = t.numel() * t.element_size()
    if t.storage_offset() != 0 or t.untyped_storage().nbytes() != nbytes:
        raise ValueError(
            f"{what} must own its whole storage (offset={t.storage_offset()}, view bytes={nbytes}, "
            f"storage bytes={t.untyped_storage().nbytes()}): copy_memory_ ignores the view and copies "
            "storages end to end"
        )


def _base_ptr(t: torch.Tensor) -> int:
    """The STORAGE address, not the view's. The identity of a staging buffer.

    ``data_ptr()`` would do for the ND staging (offset 0) but is one of the
    metadata calls an internal-format tensor can refuse on this build, and the
    NZ staging has to be identified by the same rule as the ND one or the
    "did the staging move" guard means two different things in the two modes.
    """
    return t.untyped_storage().data_ptr()


def fold_weight_write(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Whole-tensor write into a resident parameter: ``torch_npu.copy_memory_``, nothing else.

    The only channel measured to accept an internal-format destination. It
    copies the source STORAGE into the destination STORAGE whatever views it is
    given (a 1-expert view rewrote all 8 experts of the storage), so both ends
    must be whole storages of the same size -- otherwise the neighbours of a
    view are overwritten, or the copy runs short. No host branch: the resident
    weights of a streamed layer live on the device.

    ``test_fold_swap.py`` asserts this function is the committer's writer and
    scans this module's source for any OTHER write: an in-place tensor method
    (the whole ``name_`` family, so ``index_add_`` and friends need no blacklist
    entry), an ``out=`` keyword, a ``.data =`` rebind, a subscript assignment
    outside the handful of plain Python containers this module keeps, a dynamic
    ``getattr`` / ``setattr``, and ``copy_memory_`` called anywhere but here on
    anything but whole parameters. A source scan cannot see a write made from
    another module or by a C extension; two run-time layers cover that: the
    platform refuses every per-expert / in-place write into an internal-format
    (NZ) tensor, and the committer re-reads ``data_ptr()`` of every target after
    every write, which catches a rebind whatever performed it.
    """
    _assert_owns_storage(dst, "fold_weight_write: destination")
    _assert_owns_storage(src, "fold_weight_write: source")
    if dst.untyped_storage().nbytes() != src.untyped_storage().nbytes():
        raise ValueError(
            f"fold_weight_write: source storage is {src.untyped_storage().nbytes()} B, destination "
            f"{dst.untyped_storage().nbytes()} B; a whole-storage copy between them is not a write of this "
            "parameter"
        )
    import torch_npu  # device-only dependency, imported at the call site

    torch_npu.copy_memory_(dst.data, src.data)


def _row_bytes(staging: torch.Tensor) -> int:
    """Bytes of one expert's row of a ``[num_slots, ...]`` staging buffer."""
    return staging.numel() // int(staging.shape[0]) * staging.element_size()


def _stage_rows(staging: torch.Tensor, rows: torch.Tensor, src: torch.Tensor, cols: torch.Tensor) -> int:
    """``staging[rows] = src[cols]`` on the ND staging. The only write into the staging.

    ND on both sides, so a per-expert write is legal here (the refusals are
    about internal-format destinations). Never used on a resident parameter.

    Returns the bytes this copy moved: the rows ``index_copy_`` was actually
    handed times the size of one staging row -- deliberately not derived from
    the caller's ``filled`` / ``staged`` bookkeeping (see
    ``_Active.bytes_staged``).
    """
    staging.index_copy_(0, rows, src.index_select(0, cols))
    return int(rows.numel()) * _row_bytes(staging)


#: ``aclrtMemcpyKind`` (acl_rt.h): HOST_TO_HOST 0, HOST_TO_DEVICE 1,
#: DEVICE_TO_HOST 2, DEVICE_TO_DEVICE 3. The H2D leg of the bank path uses 1
#: from ``stream_io._upload_nz``; this module uses 3 and nothing else.
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def merge_runs(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """``(dst_row, src_slot)`` pairs -> ``(dst_row, src_slot, nrows)`` runs. Pure.

    A run is a stretch where BOTH sides advance by one, so one memcpy of
    ``nrows`` slots moves it: the per-expert blocks are equally sized and
    contiguous, so consecutive rows on both sides are one contiguous block on
    both sides. Pairs are sorted by destination first, which is what makes a run
    findable at all -- a chunk's experts arrive in slot order, and their rows are
    ascending but with gaps.

    It is an optimisation of the CALL COUNT only: the bytes moved are the same
    either way, and ``fold_stage_write_nz`` bounds-checks each run it is given.
    Returned runs never overlap on the destination, because a repeated
    destination row is refused here -- two sources for one row is the bug
    ``after_chunk``'s ``filled`` check is about, caught again at the copy.
    """
    # The open run is a local, and finished runs are only ever appended: the
    # write-channel scan refuses subscript assignment on anything it cannot
    # prove is a plain container, and a run table is not worth an exemption.
    out: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = None
    seen: set[int] = set()
    for dst, src in sorted(pairs):
        if dst in seen:
            raise FoldError(
                f"destination row {dst} appears twice in one chunk's copy plan: two sources "
                "would be written into one slot"
            )
        seen.add(dst)
        if cur is not None and dst == cur[0] + cur[2] and src == cur[1] + cur[2]:
            cur = (cur[0], cur[1], cur[2] + 1)
            continue
        if cur is not None:
            out.append(cur)
        cur = (dst, src, 1)
    if cur is not None:
        out.append(cur)
    return out


def fold_stage_write_nz(
    dst: torch.Tensor, src: torch.Tensor, *, bytes_per_slot: int, runs: Sequence[tuple[int, int, int]], stream: int
) -> int:
    """The ONE write into an NZ staging buffer: a raw ACL device-to-device memcpy.

    Why a raw memcpy at all. In NZ device mode (the bank path) the chunk's bytes
    arrive straight into an internal-format tensor and the ND landing buffers do
    not exist, so there is nothing for ``_stage_rows``' ``index_copy_`` to read:
    torch cannot read a row out of an internal-format tensor, cannot write one
    into it, and ``.cpu()`` on one would undo the permutation. What does work is
    what ``stream_io._upload_nz`` already does in the other direction -- address
    arithmetic over the storage -- because the NZ permutation is PER EXPERT, so
    slot ``i`` is a contiguous block at ``i * bytes_per_slot``. This copies whole
    slots between two such blocks and never looks inside one, so it is correct
    for ND and NZ alike without knowing which it has.

    Why it is sound HERE and would not be in ND mode. ``TASK_QUEUE_ENABLE=1``
    puts torch_npu's ops in a host-side queue consumed by another thread, while
    this call goes to the stream immediately -- so it can be submitted BEFORE a
    torch op that precedes it in program order (measured: 12/100 wrong slots).
    That makes a raw copy safe only when nothing torch enqueued earlier writes
    its source or reads its destination:

      * source -- in NZ mode the chunk's weight bytes are written by
        ``_upload_nz``'s raw H2D, also issued directly, on the stream the caller
        passes here, so the stream itself orders them. In ND mode the landing
        buffers are written by a torch ``copy_``, which this could overtake.
      * destination -- the NZ staging is written only through this function, and
        read by the committer's ``copy_memory_``, which is enqueued AFTER: the
        safe direction. Across layers it is not (the next layer's copy could
        overtake the previous commit's read), which is what ``ResidentFold``'s
        barrier is for.

    ``destMax`` is what we promise to write; ``aclrtMemcpyAsync`` has NO
    ``srcMax`` (acl_rt.h:2309), so the source end is bounds-checked here or not
    at all. Returns the bytes the copies were given to move.
    """
    import acl  # device-only dependency, imported at the call site

    if bytes_per_slot <= 0:
        raise FoldError(f"bytes_per_slot={bytes_per_slot}: a slot cannot be empty")
    _assert_owns_storage(dst, "fold_stage_write_nz: destination")
    _assert_owns_storage(src, "fold_stage_write_nz: source")
    dst_st, src_st = dst.untyped_storage(), src.untyped_storage()
    dst_n, src_n = dst_st.nbytes(), src_st.nbytes()
    for what, n in (("destination", dst_n), ("source", src_n)):
        if n % bytes_per_slot:
            raise FoldError(
                f"fold_stage_write_nz: the {what} is {n} B, not a whole number of "
                f"{bytes_per_slot} B slots; a per-slot offset into it is undefined"
            )
    dst_base, src_base = dst_st.data_ptr(), src_st.data_ptr()
    counts: list[int] = []
    for dst_row, src_row, nrows in runs:
        if nrows <= 0 or dst_row < 0 or src_row < 0:
            raise FoldError(
                f"fold_stage_write_nz: run (dst={dst_row}, src={src_row}, n={nrows}) is not a "
                "forward run of at least one slot"
            )
        count = nrows * bytes_per_slot
        d_off, s_off = dst_row * bytes_per_slot, src_row * bytes_per_slot
        if d_off + count > dst_n:
            raise FoldError(
                f"fold_stage_write_nz: rows {dst_row}..{dst_row + nrows - 1} end at "
                f"{d_off + count} B of a {dst_n} B destination"
            )
        if s_off + count > src_n:
            raise FoldError(
                f"fold_stage_write_nz: source rows {src_row}..{src_row + nrows - 1} end at "
                f"{s_off + count} B of a {src_n} B source -- and aclrtMemcpyAsync has no srcMax, "
                "so nothing downstream would catch it"
            )
        ret = acl.rt.memcpy_async(dst_base + d_off, count, src_base + s_off, count, ACL_MEMCPY_DEVICE_TO_DEVICE, stream)
        if ret != 0:
            raise FoldError(
                f"fold_stage_write_nz: D2D memcpy_async returned {ret} for {count} B "
                f"(dst row {dst_row}, src row {src_row})"
            )
        # Appended only after the call returned 0: a copy that was refused is
        # not bytes moved, and finish() reconciles against this number.
        counts.append(count)
    return sum(counts)


# ---------------------------------------------------------------------------
# Heat
# ---------------------------------------------------------------------------


def heat_from_topk(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Per-expert route counts of this forward, as a float64 host row.

    One device->host transfer of ``num_experts`` ints per layer. Every route
    counts, whatever its weight: the router chose the expert, which is what
    "hot" means. An id outside ``[0, num_experts)`` is an error: ``bincount``
    would grow the row, and the extra entries would be read as heat of experts
    that do not exist.
    """
    ids = topk_ids.reshape(-1).long()
    if ids.numel() == 0:
        return torch.zeros(num_experts, dtype=torch.float64)
    # The range check reads the COUNTS, not a min/max probe of the ids:
    # ``bincount`` returns ``max(max_id + 1, minlength)`` entries, so a row
    # longer than the expert count IS an out-of-range id, and a negative one
    # makes ``bincount`` itself raise. Same guarantee, and ``numel()`` is
    # metadata -- ``int(ids.max())`` would be a device->host sync in a hot path.
    counts = torch.bincount(ids, minlength=num_experts)
    if counts.numel() != num_experts:
        raise FoldError(f"topk_ids reach expert {counts.numel() - 1}, outside the layer's [0, {num_experts})")
    return counts.to("cpu", torch.float64)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldedLayer:
    """A layer whose staging was filled and cast; ready to commit."""

    layer: int
    plan: LayerPlan
    weights: dict[str, torch.Tensor]
    staged: tuple[int, ...]  # chosen experts staged, in the order they came by
    chunks_seen: int
    #: Bytes the staging copies moved for this layer, reconciled in ``finish``
    #: against ``len(chosen) x bytes per expert``. See ``_Active.bytes_staged``.
    bytes_staged: int
    # ``weights`` is NOT a snapshot: the production post-load hook rebinds the
    # two weights to a fresh ``npu_format_cast`` output but the two SCALES to a
    # 4-D transposed VIEW of the ND bytes it was given (``stream_io.format_nd``)
    # -- and those bytes are the staging, which every layer reuses. A
    # ``FoldedLayer`` held while another layer stages would commit THAT layer's
    # scale bytes: right size, right dtype, every guard green (the resident
    # read-back cannot compare NZ), wrong answer. ``commit`` refuses if anything
    # has staged since this stamp.
    staging_generation: int


@dataclass
class FoldReport:
    step: int
    layer: int
    plan: LayerPlan
    swap: SwapReport
    staged: int
    bytes_staged: int = 0

    @property
    def weights_verified(self) -> bool | None:
        return self.swap.weights_verified


@dataclass
class FoldCoverage:
    """Cumulative accounting; a run that folded nothing is a failure, not a pass."""

    decisions_seen: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    layers_folded: int = 0
    layers_committed: int = 0
    layers_recovered: int = 0
    layers_unverified: int = 0
    experts_staged: int = 0
    #: Bytes the staging copies actually moved (see ``_Active.bytes_staged``:
    #: a denominator check, NOT a statement about the content of those bytes).
    bytes_staged: int = 0
    pairs_moved: set[tuple[int, int]] = field(default_factory=set)
    staged_pairs: set[tuple[int, int]] = field(default_factory=set)
    #: Requests whose whole prefill went by without one commit. NOT part of the
    #: ``seen == folded + skipped`` denominator: it is counted between forwards,
    #: by ``ResidentFold.note_prefill_end``, and the case it exists for is the
    #: one where no decision was taken at all.
    prefills_without_commit: int = 0

    def summary(self) -> str:
        return (
            f"fold coverage: decisions {self.decisions_seen} (skipped {dict(sorted(self.skipped.items()))}), "
            f"layers folded {self.layers_folded} committed {self.layers_committed} "
            f"(recovered {self.layers_recovered}, unverified {self.layers_unverified}), "
            f"experts staged {self.experts_staged} ({self.bytes_staged} B copied), "
            f"pairs moved {len(self.pairs_moved)}, "
            f"prefills that committed nothing {self.prefills_without_commit}"
        )

    def violations(self, expected_staged: Iterable[tuple[int, int]] | None = None) -> list[str]:
        bad = []
        taken = self.layers_folded + sum(self.skipped.values())
        if taken != self.decisions_seen:
            bad.append(
                f"denominator: seen {self.decisions_seen} != folded {self.layers_folded} + skipped "
                f"{sum(self.skipped.values())}"
            )
        if self.layers_committed == 0:
            bad.append("no layer was committed: zero coverage is a failure, not a pass")
        if self.experts_staged == 0:
            bad.append("no expert was staged")
        if self.experts_staged and self.bytes_staged == 0:
            bad.append(f"{self.experts_staged} expert stagings copied 0 bytes: the accounting and the copies disagree")
        if not self.pairs_moved:
            bad.append("no (layer, expert) changed owner")
        if self.prefills_without_commit:
            bad.append(
                f"{self.prefills_without_commit} request(s) prefilled end to end without one commit: "
                "their heat was accumulated and thrown away"
            )
        if expected_staged is not None:
            want = {(int(a), int(b)) for a, b in expected_staged}
            if not want:
                raise ValueError(
                    "expected staged pairs is empty: 'nothing was supposed to be staged' is not a criterion"
                )
            if self.staged_pairs != want:
                bad.append(
                    f"staged {len(self.staged_pairs)} (layer, expert) pairs, expected {len(want)} "
                    f"(missing {sorted(want - self.staged_pairs)[:8]}, "
                    f"extra {sorted(self.staged_pairs - want)[:8]})"
                )
        return bad

    def check(self, expected_staged: Iterable[tuple[int, int]] | None = None, *, strict: bool = True) -> list[str]:
        """Verdict. Under ``strict`` the expectation is mandatory and must come from outside.

        An expectation read off the fold's own staged set cannot fail; pass the
        chosen set the scenario was BUILT to produce.
        """
        if strict and expected_staged is None:
            raise ValueError(
                "FoldCoverage.check(strict=True) needs the expected (layer, expert) pairs, and they "
                "must not come from the fold"
            )
        bad = self.violations(expected_staged)
        if bad and strict:
            raise ZeroCoverageError(f"{self.summary()} -- " + "; ".join(bad))
        return bad


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------


@dataclass
class _Active:
    layer: int
    plan: LayerPlan
    row_of: dict[int, int]
    filled: list[bool]
    staging_addrs: dict[str, int]
    staged: list[int] = field(default_factory=list)
    chunks_seen: int = 0
    # The sum over the copies that ACTUALLY RAN of (rows ``index_copy_`` was
    # handed) x (bytes of one staging row), returned by ``_stage_rows`` itself
    # rather than recomputed from the hook's intent; ``finish`` reconciles it
    # against ``len(chosen) x bytes per expert`` over the four parameters. So it
    # fails when a chosen row was never copied, when a copy ran short, when a row
    # was copied twice, and -- this is why it exists -- when ``filled`` /
    # ``staged`` (set UNCONDITIONALLY after ``_stage_rows`` returns) have come
    # apart from the copies. It counts rows, not content: wrong bytes with the
    # right shapes, or a row clobbered IN PLACE by a later chunk (the address
    # does not change, so ``_staging_moved`` sees nothing; the generation stamp
    # counts only the fold's OWN writes), reconcile to exactly this number. A
    # DENOMINATOR self-consistency check, not a replacement for the bit-for-bit
    # criterion against an independently built resident set.
    bytes_staged: int = 0


class ResidentFold:
    """Fold the swap-in into the streaming pass. One layer at a time, eager only.

    ``placements`` / ``targets`` are what ``ResidentSwapCommitter`` takes; the
    fold builds the committer itself so that its writer is
    :func:`fold_weight_write`, its ``writer_busy`` guard is
    :meth:`layer_in_flight` (about THIS layer, not about forwards) and its
    roll-forward re-reads the staging this fold cast -- nothing to re-pull, which
    is the point of folding.

    ``provider`` is the streaming ``StreamSlotProvider`` (duck-typed): its
    ``shapes`` / ``seam`` / ``device`` allocate the staging, ``served_nd`` copies
    a chunk out, ``format_nd`` casts. It must be the SAME object that served the
    chunks -- ``after_chunk`` copies out of its H2D buffers -- which is why
    ``layer.install_resident_fold`` installs one provider into both.

    ``live_objects(layer)`` returns ``{"weights", "slot_tables", "keep_tables",
    "kt_mask"}`` as the LAYER holds them at that moment and is handed straight to
    the committer, which refuses any commit whose targets are not those objects
    (``resident_swap.live_target_violations``). ``None`` leaves that check off,
    right only for a fixture with no forward to detach from; the production
    wiring always passes one.

    ``window_source()`` answers "whose prefill is this forward, and is it the
    last chunk of it" -- ``prefill_window.require`` in production, which raises
    if the runner never published one rather than letting the fold decide on
    every chunk. ``None`` from it is a step the fold declines (not an error).

    ``stream_min_tokens`` is the layer's ``stream_prefill_min_tokens``, handed
    down by ``layer.install_resident_fold`` from the same config the streaming
    gate reads; it is what makes "the deciding chunk" and "the last chunk"
    different things (see ``_decides_now``).

    Call order per streamed layer: ``begin(layer, topk_ids)``; if it returned a
    plan, pass ``after_chunk`` to ``run_streamed_layer``; then ``finish()`` and
    ``commit(folded)``; ``abort()`` if the streamed layer raised in between. On
    every chunk but the deciding one ``begin`` returns ``None`` and the other
    three are not reached.
    """

    def __init__(
        self,
        placements: Mapping[int, LayerPlacement],
        targets: Mapping[int, LayerSwapTargets],
        *,
        provider: Any,
        policy: Callable[..., Sequence[int]] | None = None,
        step_source: Callable[[], int] = lambda: 0,
        capture_predicate: Callable[[], bool | None] = capture_state,
        stream_id: Callable[[], object] = lambda: None,
        stream_min_tokens: int,
        strict: bool = True,
        on_write: Callable[[str, str], None] | None = None,
        live_objects: Callable[[int], Mapping[str, Any]] | None = None,
        window_source: Callable[[], prefill_window.PrefillWindow | None] = prefill_window.require,
    ):
        self.provider = provider
        self.policy = policy or TopKHeatPolicy()
        self.step_source = step_source
        self.capture_predicate = capture_predicate
        self.window_source = window_source
        # ONE number, read where the gate that uses it lives (the layers'
        # kt_offload_config) and handed down by ``layer.install_resident_fold``:
        # the fold's commit point is decided by the same threshold that decides
        # whether a forward streams at all, so a second reading of the config
        # could put the commit on a forward that takes the hybrid path.
        self.stream_min_tokens = int(stream_min_tokens)
        if self.stream_min_tokens <= 0:
            raise FoldError(
                f"stream_min_tokens={stream_min_tokens}: it is the layer's "
                "stream_prefill_min_tokens, which config.py refuses at or below 0"
            )
        self.strict = bool(strict)
        self.coverage = FoldCoverage()
        # The running heat of the request being prefilled: one row per layer,
        # ``[num_experts]`` float64 on the host. ``_heat_key`` / ``_heat_next``
        # say which request it belongs to and where its next chunk must start;
        # ``_heat_seq`` makes the "same window?" test run once per step rather
        # than once per layer (all 40 layers see one window).
        self._heat: dict[int, torch.Tensor] = {}
        self._heat_key: str | None = None
        self._heat_next: int | None = None
        self._heat_seq: int | None = None
        # "Did anything come of the prefill that just ended?", per request
        # rather than cumulative: request A committing 40 layers must not answer
        # for request B, which the fold may never have been asked about at all.
        # See ``note_prefill_end``.
        self._request_key: str | None = None
        self._request_decisions = 0
        self._request_commits = 0
        self._decided_at: dict[int, int] = {}
        # Bumped by every write into the staging; a FoldedLayer records the
        # value at ``finish`` so ``commit`` can tell whether its scale views
        # (which alias the staging) still hold its own bytes.
        self._staging_generation = 0
        self._warned: set[str] = set()
        self._active: _Active | None = None
        self._committing: FoldedLayer | None = None
        self._lock = threading.Lock()
        self.committer = ResidentSwapCommitter(
            placements,
            targets,
            strict=strict,
            weight_writer=fold_weight_write,
            weight_refetch=self._refetch,
            stream_id=stream_id,
            writer_busy=self.layer_in_flight,
            on_write=on_write,
            live_objects=live_objects,
        )

        layers = sorted(placements)
        if not layers:
            raise FoldError("no layer to fold")
        for layer in layers:
            p, tg = self.committer.placement(layer), self.committer.targets(layer)
            if p.num_resident != p.num_slots:
                raise FoldError(
                    f"layer {layer}: {p.num_resident} resident experts in {p.num_slots} slots. The commit writes "
                    f"the whole [{p.num_slots}, ...] storage and the staging holds only the resident rows, so "
                    "the zeroed tail slots would be overwritten; not implemented (same limit as the driver)."
                )
            if set(tg.weights) != set(SLOT_WEIGHT_ATTRS):
                raise FoldError(
                    f"layer {layer}: commit targets are {sorted(tg.weights)}, the staging carries "
                    f"{list(SLOT_WEIGHT_ATTRS)}"
                )
        num_slots = {self.committer.placement(layer).num_slots for layer in layers}
        if len(num_slots) != 1:
            raise FoldError(f"layers disagree on num_slots {sorted(num_slots)}; it is one number per model")
        self.num_slots = num_slots.pop()
        self.layers = tuple(layers)

        # ``nz_device`` is in the list although it is a bool: a provider without
        # it cannot be asked which layout its chunks are in, and guessing ND
        # would stage unwritten bytes with every guard green (the bug the
        # bank+fold config refusal was standing in for).
        for need in ("shapes", "seam", "device", "served_nd", "served_nz", "format_nd", "nz_device"):
            if getattr(provider, need, None) is None:
                raise FoldError(
                    f"the slot provider has no {need!r}; the fold copies chunks out of it and casts through it"
                )
        # The staging: allocated once, through the provider's device seam, with
        # the provider's per-expert shapes (which the production method declared).
        self._staging: dict[str, torch.Tensor] = {}
        for name in SLOT_WEIGHT_ATTRS:
            shape, dtype = provider.shapes[name]
            want = (self.num_slots, *tuple(shape[1:]))
            buf = provider.seam.alloc_device(want, dtype, provider.device)
            if tuple(buf.shape) != want or buf.dtype != dtype:
                raise FoldError(f"staging {name} came back {buf.dtype}{tuple(buf.shape)}, asked for {dtype}{want}")
            self._staging[name] = buf
        # The addresses the scale views of every cast slot are taken against.
        self._staging_addrs0 = {n: _base_ptr(t) for n, t in self._staging.items()}
        ptrs = list(self._staging_addrs0.values())
        if len(set(ptrs)) != len(ptrs):
            raise FoldError("staging buffers share storage")
        # Bytes one expert occupies across all four staging buffers: the unit
        # ``finish`` reconciles the copies against.
        self._expert_bytes = sum(_row_bytes(t) for t in self._staging.values())
        # -- NZ (bank) mode ---------------------------------------------------
        # The chunks arrive in FRACTAL_NZ and the ND landing buffers are gone,
        # so the weight half of the staging has to be NZ too and is filled by
        # address (``fold_stage_write_nz``). Decided once, from the provider, and
        # never re-read: a provider that switched layout mid-run would be staging
        # one layout into a buffer tagged as the other.
        self._nz = bool(provider.nz_device)
        self._nz_bytes: dict[str, int] = {}
        self._nz_slot: Any | None = None
        self._needs_barrier = False
        if self._nz:
            self._enable_nz_staging()

    def _enable_nz_staging(self) -> None:
        """Turn the weight half of the staging NZ, ONCE, and keep the cast slot forever.

        The cast is run on the freshly allocated ND staging for one reason: to
        obtain an NZ-tagged allocation of the right size through the production
        hook, because the operator dispatches on that tag (a plain tensor holding
        the same bytes is refused with 161002). Its CONTENT is irrelevant and the
        ND staging is deliberately not zeroed first -- every row of the staging
        is rewritten on every folded layer (``num_resident == num_slots`` is
        enforced above and ``finish`` refuses a row that was not filled), so no
        byte of this allocation ever reaches a resident parameter unwritten.

        The slot is kept for the life of the fold instead of being re-cast per
        layer, which is what makes the fold free of casts in this mode:

          * its WEIGHTS are the NZ allocations, which are now the staging;
          * its SCALES are 4-D views of the ND scale buffers (``format_nd``),
            which stay and are still written in place by ``_stage_rows`` -- so
            the same slot object describes every layer, and ``commit``'s
            ``staging_generation`` guard is what keeps a stale one from being
            committed.
        """
        slot = self.provider.format_nd(self._staging, self.num_slots)
        tensors = slot.tensors()
        pairs: list[tuple[str, int]] = []
        for name in NZ_WEIGHT_ATTRS:
            nbytes = tensors[name].untyped_storage().nbytes()
            if nbytes % self.num_slots:
                raise FoldError(
                    f"staging {name}: the cast produced {nbytes} B for {self.num_slots} slots, "
                    "not a whole number of slots; a per-slot offset into it is undefined"
                )
            per_slot = nbytes // self.num_slots
            nd_row = _row_bytes(self._staging[name])
            if per_slot != nd_row:
                raise FoldError(
                    f"staging {name}: one slot is {per_slot} B in NZ and {nd_row} B in ND. The fold's "
                    "byte accounting, the chunk's per-slot offsets and the resident parameter's size are all "
                    "one number; NZ was measured to be a pure permutation with no padding on this model, and "
                    "this says it is not."
                )
            pairs.append((name, per_slot))
        nz_bytes = dict(pairs)
        # The chunk side's per-slot size must be the same number, or a slot-sized
        # block means one thing at the source and another at the destination.
        chunk_bytes = self.provider.nz_bytes_per_slot
        for name in NZ_WEIGHT_ATTRS:
            if int(chunk_bytes.get(name, -1)) != nz_bytes[name]:
                raise FoldError(
                    f"{name}: a slot is {chunk_bytes.get(name)} B in the provider's chunk buffer "
                    f"and {nz_bytes[name]} B in the fold's staging"
                )
        freed = sum(self._staging[n].numel() * self._staging[n].element_size() for n in NZ_WEIGHT_ATTRS)
        for name in NZ_WEIGHT_ATTRS:
            self._staging[name] = tensors[name]
        # The ND weight buffers were allocated only to be cast; nothing writes
        # them again. (The ND SCALE buffers MUST stay: the slot's scale views
        # alias them and every layer writes through those views.)
        import gc

        gc.collect()
        if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
            torch.npu.empty_cache()
        self._nz_slot = slot
        self._nz_bytes = nz_bytes
        self._staging_addrs0 = {n: _base_ptr(t) for n, t in self._staging.items()}
        if self._expert_bytes != sum(nz_bytes.values()) + sum(_row_bytes(self._staging[n]) for n in ND_SCALE_ATTRS):
            raise FoldError(
                "the per-expert byte total changed when the staging became NZ; the reconciliation "
                "in finish() would be against the wrong unit"
            )
        logger.info(
            "kt_offload fold: NZ staging -- weights filled by raw D2D at %s B/slot, scales still ND; "
            "no per-layer cast; freed %d B (%.3f GiB) of ND weight staging; "
            "barrier=device synchronize",
            dict(nz_bytes),
            freed,
            freed / float(1 << 30),
        )

    # -- state ------------------------------------------------------------
    def layer_in_flight(self) -> bool:
        """True while a layer is being folded: the committer (``writer_busy``) refuses to write then.

        NOT "a forward is in flight": a forward always is when the fold commits,
        and this is False by then because ``finish`` cleared ``_active``. It only
        stops a commit issued from inside the layer that is still filling the
        staging it would read. Concurrency is the ledger's job, not this one's.
        """
        return self._active is not None

    @property
    def staging(self) -> dict[str, torch.Tensor]:
        return dict(self._staging)

    def _skip(self, why: str) -> None:
        self.coverage.skipped[why] = self.coverage.skipped.get(why, 0) + 1
        return None

    def _say_once(self, why: str, message: str) -> None:
        """Warn once per reason, for a whole process. The loud half of ``_skip_loudly``.

        Separate from ``_skip`` because not everything loud is a skipped
        DECISION: ``note_prefill_end`` fires between forwards, and filing it as
        a skip would put a number in the numerator of
        ``decisions_seen == folded + sum(skipped)`` that no decision produced.
        """
        if why not in self._warned:
            self._warned.add(why)
            logger.warning("kt_offload fold: %s", message)

    def _skip_loudly(self, why: str, message: str) -> None:
        """A skip that means the fold CANNOT work, not that it had nothing to do.

        Logged once per reason: a silent no-op and a working fold that chose not
        to move anything produce the same counters otherwise, and only one of
        them is a bug.
        """
        self._say_once(why, message)
        return self._skip(why)

    # -- a whole prefill went by ------------------------------------------
    def note_prefill_end(self, key: str) -> None:
        """A request stopped prefilling. If nothing was committed during it, be loud.

        The one check that can see a prefill the fold was never asked about --
        everything else is counted inside ``begin``, and the failure this exists
        for is that ``begin`` is not reached: ``layer.maybe_stream_prefill``
        returns before the fold on any forward shorter than
        ``stream_prefill_min_tokens``. ``_decides_now`` moves the commit off that
        chunk; this says so out loud if some OTHER member of the family (a
        preemption, a scheduler that splits the tail, a future gate) puts a
        prefill back in the same hole.

        Wired by ``layer.install_resident_fold`` via
        ``prefill_window.on_prefill_end``; fires once per prefill, from
        ``observe``, on the step after the last one the request was prefilling.
        """
        key = str(key)
        mine = key == self._request_key
        if mine and self._request_commits:
            # Without it an in-server run cannot tell "the resident set moved"
            # from "the fold was never reached", the one distinction every
            # dynamic_resident measurement rests on.
            logger.info(
                "kt_offload fold: request %s committed %d layer(s) over its prefill (%s)",
                key,
                self._request_commits,
                self.coverage.summary(),
            )
            return
        self.coverage.prefills_without_commit += 1
        decisions = self._request_decisions if mine else 0
        if decisions == 0:
            self._say_once(
                "prefill_never_reached_the_fold",
                f"request {key}: its whole prefill went by without the folded swap-in being asked once. "
                f"Every forward of it was declined before the fold -- a forward of fewer than "
                f"{self.stream_min_tokens} tokens (stream_prefill_min_tokens) takes the hybrid path and "
                "never reaches it. The request's heat was never counted and its resident set never moved; "
                "nothing is computed incorrectly, but dynamic_resident did NOT run for this request.",
            )
        else:
            self._say_once(
                "prefill_ended_without_commit",
                f"request {key}: {decisions} fold decision(s) over its prefill and not one commit, so its "
                f"resident set never moved ({self.coverage.summary()}). The heat was accumulated and "
                "thrown away.",
            )

    # -- heat, accumulated across the request's prefill chunks --------------
    def accumulated_heat(self, layer: int) -> torch.Tensor | None:
        """This layer's running heat for the request being prefilled (a copy), or ``None``."""
        row = self._heat.get(int(layer))
        return None if row is None else row.clone()

    @property
    def heat_request(self) -> str | None:
        """The request the running heat belongs to."""
        return self._heat_key

    def _note_window(self, window: prefill_window.PrefillWindow) -> None:
        """Once per step: does this window continue the running heat, or start it over?

        The reset must happen ONCE per step, not once per layer: every layer of
        one forward is handed the same window, so a per-layer "first chunk? then
        clear" would wipe layer 0's row while layer 1 was still being decided.

        A window continues the accumulation only when it is the same request AND
        its tokens begin exactly where the previous chunk ended. Anything else --
        a different request, a prompt restarted after preemption, a first chunk
        that begins above 0 because the prefix cache covered the front, a step
        this process never saw -- starts a new one; continuing across any of
        those would let one request's routing decide another's resident set.
        """
        if window.seq == self._heat_seq:
            return
        if window.key != self._heat_key or window.computed != self._heat_next:
            self._heat.clear()
            self._heat_key = window.key
        self._heat_next = window.computed + window.scheduled
        self._heat_seq = window.seq

    def _decides_now(self, window: prefill_window.PrefillWindow) -> bool:
        """The LAST forward of this prefill that streams decides; the others only count.

        Not ``window.last``, and the difference is the common case: a forward
        only reaches this fold if ``layer.maybe_stream_prefill`` streamed it,
        which needs ``num_tokens >= cfg.stream_prefill_min_tokens``, and the last
        chunk of a prompt holds ``prompt_len mod max_num_batched_tokens`` tokens
        -- the chunk most likely to be under any threshold. Deciding on
        ``window.last`` alone therefore lets a whole request accumulate heat and
        commit nothing, with every counter self-consistent and no log line.

        So: this forward commits when no LATER forward of the same prefill can
        stream. ``remaining`` is what the prefill has left after this one and no
        later forward can carry more, so ``remaining < stream_min_tokens`` means
        every one of them takes the hybrid path however the scheduler cuts them
        up. The whole rule lives here so a test can take it away
        (``test_mut_deciding_on_every_chunk_commits_K_times``,
        ``test_mut_committing_only_on_the_window_marked_last_never_commits``).

        Boundaries: ``remaining == 0`` is ``window.last``, the ordinary case. At
        ``remaining == stream_min_tokens`` exactly the gate is ``<``, so the
        following forward DOES stream and the decision is left to it (splitting
        those tokens further would leave every piece short and commit nothing,
        which ``note_prefill_end`` shouts about; it cannot happen at the
        ``max_num_seqs == 1`` ``dynamic_resident`` requires). A prompt that fits
        in one forward commits on it; one SHORTER than the threshold never
        streams at all, which is ``note_prefill_end``'s case. Nothing here reads
        ``computed`` or ``first``, only what is left, so a prefix-cache hit
        behaves like a shorter prompt.

        Do NOT "fix" this family by lowering ``stream_prefill_min_tokens``:
        ``wiring._refuse_streamable_capture_size`` refuses at start-up any
        threshold at or below the largest ``cudagraph_capture_sizes`` entry (a
        captured size that would also stream dies mid-serve in
        ``stream_prefill.assert_eager_only``). The threshold has a floor, so the
        commit point has to move instead.
        """
        if window.last:
            return True
        return 0 < window.remaining < self.stream_min_tokens

    def _accumulate(self, layer: int, counts: torch.Tensor) -> torch.Tensor:
        row = self._heat.get(layer)
        if row is None:
            row = torch.zeros_like(counts)
            self._heat[layer] = row
        row.add_(counts)
        return row

    # -- 1. decide, before chunk 0 -----------------------------------------
    def begin(self, layer: int, topk_ids: torch.Tensor) -> LayerPlan | None:
        """Add this forward's routing to the request's running heat; decide on its last chunk.

        ``None`` = no fold on this layer in this forward (reason in
        ``coverage.skipped``); the caller then streams exactly as without the
        fold. That is the answer for every chunk but the deciding one, so a
        K-chunk prompt stages and commits ONCE, not K times.
        """
        with self._lock:
            if self._active is not None:
                raise FoldError(
                    f"layer {self._active.layer} is still being folded: finish() or abort() it "
                    "before beginning another layer"
                )
            # Before the counters: a missing runner hook is a wiring failure,
            # not a decision this fold declined to take.
            window = self.window_source()
            self.coverage.decisions_seen += 1
            try:
                plan = self._decide(layer, topk_ids, window)
                if plan is not None and self._nz and self._needs_barrier:
                    # Once per layer that will be staged, not once per chunk: the
                    # staging writes of THIS layer are what has to come after the
                    # PREVIOUS layer's commit read it, and nothing between those
                    # two points touches the staging.
                    self._nz_barrier_wait()
                    self._needs_barrier = False
                return plan
            except BaseException as exc:
                # Every exit AFTER the counter must land in the denominator, or
                # ``decisions_seen == folded + sum(skipped)`` is false for the
                # rest of the process. Named after the exception, so a failure is
                # not filed under a reason that means something else.
                self._skip(f"raised_{type(exc).__name__}")
                raise

    def _decide(
        self, layer: int, topk_ids: torch.Tensor, window: prefill_window.PrefillWindow | None
    ) -> LayerPlan | None:
        """``begin`` with the lock held and the decision counted. Every exit is accounted."""
        cap = self.capture_predicate()
        if cap is True:
            return self._skip("captured")
        if cap is not False:
            # Only a positive "not captured" opens the gate (the driver's rule).
            return self._skip("capture_state_unknown")
        if layer not in self.layers:
            raise FoldError(f"layer {layer} is not one of the folded layers {list(self.layers)}")
        if window is None:
            return self._skip_loudly(
                "no_prefill_window",
                f"layer {layer}: this step is not one request's prefill as far as the scheduler output "
                "goes (a decode step, or several requests prefilling at once). The folded swap-in "
                "accumulates a request's heat and commits on its last prefill chunk, and it has no "
                "commit point here, so it neither counts this forward nor moves anything.",
            )
        if window.key != self._request_key:
            # Per-request, so the previous request's commits cannot answer for
            # this one (``note_prefill_end``). Counted here rather than in
            # ``_note_window``, which runs once per STEP and not at all for a
            # request whose every forward is skipped before it.
            self._request_key, self._request_decisions, self._request_commits = window.key, 0, 0
        self._request_decisions += 1
        p = self.committer.placement(layer)
        pinned = pinned_resident_ids(self.committer.targets(layer), p)
        if len(pinned) >= p.num_resident:
            # Every resident expert is pinned, i.e. kt holds CPU weights for
            # none of them (subset load): the policy has zero free slots and can
            # only propose the current set. Without this branch the fold reports
            # "no_change", which reads like "the heat says nothing should move".
            return self._skip_loudly(
                "all_resident_pinned",
                f"layer {layer}: all {p.num_resident} resident experts are pinned (kt reports CPU weights "
                "for none of them), so the folded swap cannot move anything. The full-set load ("
                "KT_FULLSET_LOAD) has to be on before dynamic_resident can do anything at all.",
            )
        if self._decided_at.get(layer) == window.seq:
            raise FoldError(
                f"layer {layer} has already been decided under prefill window {window} -- the runner "
                "published no new one between the two forwards. Accumulating this forward's routing "
                "again would count its tokens twice, and a window whose 'last' is stale would commit a "
                "second time on a chunk that is not the last. prefill_window.observe() must run once "
                "per execute_model."
            )
        self._note_window(window)
        self._decided_at[layer] = window.seq
        heat = self._accumulate(layer, heat_from_topk(topk_ids, p.num_experts))
        if not self._decides_now(window):
            # Accumulated, and that is all: no plan, so ``fold_streamed_layer``
            # streams with no ``after_chunk`` hook -- nothing is staged, cast or
            # committed, and the pass is bit for bit the one with the switch off.
            return self._skip("not_last_chunk")
        # ``clone()``: the policy is handed the accumulation's value, never
        # the row it would go on adding into.
        plan = plan_from_heat(layer, p, heat.clone(), self.policy, pinned)
        if plan is None:
            return self._skip("no_counts")
        if not (plan.promote or plan.demote):
            return self._skip("no_change")
        self._active = _Active(
            layer=layer,
            plan=plan,
            row_of={e: i for i, e in enumerate(plan.new_ids)},
            filled=[False] * len(plan.new_ids),
            staging_addrs={n: _base_ptr(t) for n, t in self._staging.items()},
        )
        return plan

    # -- 2. after every chunk ---------------------------------------------
    def after_chunk(self, chunk_index: int, expert_ids: Sequence[int]) -> None:
        """Copy the chosen experts of this chunk into their new rows of the staging."""
        a = self._active
        if a is None:
            raise FoldError("after_chunk without begin(): the hook is wired to a layer the fold did not open")
        self._check_staging(a)
        if int(self.provider.layer) != a.layer:
            raise FoldError(
                f"the provider is bound to layer {self.provider.layer}, the fold is on layer "
                f"{a.layer}: the chunk's bytes belong to another layer"
            )
        ids = tuple(int(e) for e in expert_ids)
        # Both accessors make the same staleness check (this layer, this chunk
        # index, these ids) and both refuse the mode they are not for, so asking
        # the wrong one cannot quietly hand back the other layout's buffers.
        served = self.provider.served_nz(chunk_index, ids) if self._nz else self.provider.served_nd(chunk_index, ids)
        rows, cols = [], []
        for pos, e in enumerate(ids):
            row = a.row_of.get(e)
            if row is None:
                continue
            if a.filled[row]:
                raise FoldError(
                    f"layer {a.layer}: expert {e} came by in two chunks; its staging row would be "
                    "written twice from two different sources"
                )
            rows.append(row)
            cols.append(pos)
        a.chunks_seen += 1
        if not rows:
            return
        dev = self.provider.device
        r = torch.tensor(rows, dtype=torch.int64, device=dev)
        c = torch.tensor(cols, dtype=torch.int64, device=dev)
        self._staging_generation += 1
        # The copies report how much they moved; the two lines below do not.
        if self._nz:
            a.bytes_staged += self._stage_chunk_nz(served, rows, cols, r, c)
        else:
            a.bytes_staged += sum(_stage_rows(self._staging[name], r, served[name], c) for name in SLOT_WEIGHT_ATTRS)
        for row, pos in zip(rows, cols):
            a.filled[row] = True
            a.staged.append(ids[pos])

    def _stage_chunk_nz(
        self, served: Any, rows: Sequence[int], cols: Sequence[int], r: torch.Tensor, c: torch.Tensor
    ) -> int:
        """One chunk into the NZ staging: scales by ``index_copy_``, weights by address.

        The split is not a style choice. The scales never went through the cast,
        so both ends are ND and the ordinary torch write is both legal and
        ordered by the host queue like everything else. The weights are
        internal-format on both ends, where torch has no per-row write at all,
        and the raw channel is sound here for the reasons
        :func:`fold_stage_write_nz` sets out -- one of which is that the copy is
        issued on the SAME stream as the H2D that produced its source, so this
        refuses any other.
        """
        now = int(torch.npu.current_stream().npu_stream)
        if now != int(served.stream):
            raise FoldError(
                f"the chunk's H2D went to ACL stream {served.stream} and this forward is on {now}: a raw "
                "device-to-device copy issued on a different stream is ordered against the H2D by nothing, "
                "so it could read the previous chunk's bytes"
            )
        if int(served.num_slots) <= 0:
            raise FoldError("the served chunk declares no slots")
        for pos in cols:
            if not 0 <= int(pos) < int(served.num_slots):
                raise FoldError(f"slot {pos} is outside the chunk's {served.num_slots} slots")
        runs = merge_runs(list(zip((int(x) for x in rows), (int(x) for x in cols))))
        scales = [_stage_rows(self._staging[name], r, served.nd_scales[name], c) for name in ND_SCALE_ATTRS]
        weights = [
            fold_stage_write_nz(
                self._staging[name],
                served.nz_weights[name],
                bytes_per_slot=self._nz_bytes[name],
                runs=runs,
                stream=int(served.stream),
            )
            for name in NZ_WEIGHT_ATTRS
        ]
        return sum(scales) + sum(weights)

    def _nz_barrier_wait(self) -> None:
        """Order the next layer's raw staging writes after the last commit's read of it.

        The committer reads the staging through ``torch_npu.copy_memory_``, which
        goes into the host-side task queue; the next layer's staging writes are
        raw ACL copies, which go to the stream at once and can therefore be
        submitted BEFORE it. Without a barrier the commit of layer L can read
        rows layer L+1 has already overwritten -- the same overtaking that was
        measured as 12/100 wrong slots on the H2D path, and it would be silent:
        every byte count, every address and every coverage check still agrees.

        A device synchronize, because it is the one barrier that unambiguously
        flushes the queue as well as the stream. A three-way knob once offered
        ``"stream"`` (cheaper, never measured) and ``"none"`` (whose declared
        purpose was to drive the failure on card as a positive control, i.e. to
        make the product compute wrong answers); both were deleted on 2026-11-21
        as nothing set them.
        """
        torch.npu.synchronize()

    def _staging_moved(self, expected: Mapping[str, int]) -> list[str]:
        """Staging buffers whose address is no longer the one recorded. One place, two callers.

        Both the per-chunk guard and the pre-commit guard ask this, so a test
        that takes "has the staging moved" away takes it away everywhere.
        """
        return [n for n, t in self._staging.items() if _base_ptr(t) != expected[n]]

    def _check_staging(self, a: _Active) -> None:
        moved = self._staging_moved(a.staging_addrs)
        if moved:
            raise FoldError(
                f"layer {a.layer}: staging {moved} was reallocated mid-layer; the rows staged before are gone"
            )

    # -- 3. after the last chunk ------------------------------------------
    def finish(self) -> FoldedLayer:
        """Check every chosen row was filled, cast the staging once, close the layer."""
        a = self._active
        if a is None:
            raise FoldError("finish() without begin()")
        try:
            self._check_staging(a)
            missing = [e for e, row in a.row_of.items() if not a.filled[row]]
            if missing:
                raise FoldIncompleteError(
                    f"layer {a.layer}: chosen experts {sorted(missing)[:8]} ({len(missing)} of "
                    f"{len(a.row_of)}) never came by in any of the {a.chunks_seen} chunk(s). Their bytes exist "
                    "nowhere on the device (the resident NZ bytes cannot be read back to ND), so "
                    "the whole-tensor write would carry stale rows into their slots."
                )
            want = len(a.row_of) * self._expert_bytes
            if a.bytes_staged != want:
                raise FoldError(
                    f"layer {a.layer}: the staging copies moved {a.bytes_staged} B, but {len(a.row_of)} chosen "
                    f"experts x {self._expert_bytes} B = {want} B were accounted as staged. The rows this "
                    "layer believes it filled and the copies that actually ran have come apart ("
                    "J2). 🔴 The reverse does NOT hold: equal byte counts say nothing about the CONTENT of "
                    "the staging -- only the bit-for-bit criterion does."
                )
            if self._nz:
                if len(a.plan.new_ids) != self.num_slots:
                    raise FoldError(
                        f"layer {a.layer}: the plan has {len(a.plan.new_ids)} rows and the NZ staging holds "
                        f"{self.num_slots} slots. The prebuilt slot describes the WHOLE staging, so a short "
                        "plan would commit the tail slots as well -- whatever the previous layer left there."
                    )
                # The staging IS the cast output (built once, in
                # ``_enable_nz_staging``): its weights are the NZ allocations the
                # chunks were copied into and its scales are the views
                # ``format_nd`` took of the ND scale buffers, which this layer
                # has just written through. Casting again is impossible as well
                # as pointless -- ``npu_format_cast`` refuses a tensor that is
                # already in an internal format.
                slot = self._nz_slot
            else:
                slot = self.provider.format_nd(self._staging, len(a.plan.new_ids))
        except FoldIncompleteError:
            # The label means what the exception means: a chosen expert never
            # came by. Keeps the denominator whole (seen == folded + skipped).
            self._skip("incomplete")
            raise
        except BaseException as exc:
            # Named after the exception -- a cast that raised, a staging that
            # moved and a byte count that did not reconcile must not borrow
            # "incomplete"'s meaning -- and still counted, so the denominator
            # survives.
            self._skip(f"raised_{type(exc).__name__}")
            raise
        finally:
            self._active = None
        self.coverage.layers_folded += 1
        self.coverage.experts_staged += len(a.staged)
        self.coverage.bytes_staged += a.bytes_staged
        self.coverage.staged_pairs |= {(a.layer, e) for e in a.staged}
        return FoldedLayer(
            layer=a.layer,
            plan=a.plan,
            weights=slot.tensors(),
            staged=tuple(a.staged),
            chunks_seen=a.chunks_seen,
            bytes_staged=a.bytes_staged,
            staging_generation=self._staging_generation,
        )

    def abort(self) -> None:
        """Drop the layer in flight (the streamed layer raised). Nothing was written to it."""
        if self._active is not None:
            self._skip("aborted")
            self._active = None

    # -- 4. commit ----------------------------------------------------------
    def commit(self, folded: FoldedLayer, step: int | None = None) -> FoldReport:
        """Hand the cast staging to the committer. Its guards, its order, its roll-forward."""
        step = int(self.step_source() if step is None else step)
        if folded.staging_generation != self._staging_generation:
            raise FoldError(
                f"layer {folded.layer}: the staging has been written "
                f"{self._staging_generation - folded.staging_generation} time(s) since this layer was cast, "
                "and the scales of a cast slot are VIEWS of the staging "
                "(stream_io.format_nd) -- committing now would write another layer's scale bytes with every "
                "guard green. Commit a folded layer before staging the next one."
            )
        moved = self._staging_moved(self._staging_addrs0)
        if moved:
            raise FoldError(
                f"layer {folded.layer}: staging {moved} was reallocated before the commit; the "
                "cast slot's scale views point into the old storage"
            )
        self._committing = folded
        try:
            rep = self.committer.commit(step, folded.layer, folded.plan.new_ids, folded.weights)
        finally:
            self._committing = None
            # Set whatever the commit did: a write that raised half way through
            # has still enqueued reads of the staging, and those are what the
            # next layer's raw copies must not overtake.
            if self._nz:
                self._needs_barrier = True
        cov = self.coverage
        if rep.ok:
            if set(rep.promoted) != set(folded.plan.promote) or set(rep.demoted) != set(folded.plan.demote):
                raise FoldError(
                    f"layer {folded.layer}: the plan promoted {folded.plan.promote[:8]} / demoted "
                    f"{folded.plan.demote[:8]}, the commit reports {rep.promoted[:8]} / {rep.demoted[:8]}"
                )
            cov.layers_committed += 1
            self._request_commits += 1
            cov.layers_recovered += int(rep.recovered)
            cov.layers_unverified += int(rep.weights_verified is not True)
            cov.pairs_moved |= {(folded.layer, e) for e in rep.promoted + rep.demoted}
            if rep.recovered:
                logger.warning(
                    "kt_offload fold: layer %d reached the new set only by rolling forward: %s", folded.layer, rep.error
                )
        return FoldReport(
            step=step,
            layer=folded.layer,
            plan=folded.plan,
            swap=rep,
            staged=len(folded.staged),
            bytes_staged=folded.bytes_staged,
        )

    def _refetch(self, layer: int, new_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """The committer's roll-forward source: the cast staging being committed, re-offered.

        Nothing is pulled again: it is the same cast output the failed write
        read, still alive (``commit`` holds it), handed back only for the layer
        and set it was cast for.
        """
        f = self._committing
        if f is None or f.layer != int(layer) or tuple(f.plan.new_ids) != tuple(int(e) for e in new_ids):
            raise FoldError(
                f"roll-forward asked for layer {layer} {tuple(new_ids)[:8]}, but the fold is "
                f"committing {None if f is None else (f.layer, f.plan.new_ids[:8])}"
            )
        return dict(f.weights)

    # -- convenience for the wiring ---------------------------------------
    def fold_streamed_layer(
        self,
        layer: int,
        topk_ids: torch.Tensor,
        run: Callable[[Callable[[int, tuple[int, ...]], None] | None], Any],
    ) -> tuple[Any, FoldReport | None]:
        """``begin`` -> ``run(after_chunk)`` -> ``finish`` -> ``commit``, with ``abort`` on failure.

        ``run(hook)`` must run the streamed layer and call ``hook`` after every
        chunk; it is called with ``None`` when this layer is not folded, so the
        caller streams exactly as without the fold.
        """
        plan = self.begin(layer, topk_ids)
        if plan is None:
            return run(None), None
        try:
            out = run(self.after_chunk)
            folded = self.finish()
        except BaseException:
            self.abort()
            raise
        return out, self.commit(folded)
