# SPDX-License-Identifier: Apache-2.0
"""Stage 2: serve the prefill staging out of the /dev/shm NZ bank.

Drop-in for :class:`weight_source.WeightSource` at the one seam the streaming
path uses -- ``source.gather(layer, experts, staging)`` (``stream_io.py:607``).

What changes, and what deliberately does not
--------------------------------------------
Today the staging is filled with **ND** bytes read out of the checkpoint, and
the device casts them to FRACTAL_NZ after the H2D. The bank already holds
**NZ** bytes, so this fills the staging with the device-consumable layout and
the cast disappears. That is the whole of stage 2's prefill saving: the 192
per-chunk selecting/concatenating/reordering memcpys against the checkpoint
become plain runs out of one file, and the device-side cast goes away.

Consequences the caller must honour (NOT handled here):
  * the device chunk buffer must already carry the NZ format tag and be filled
    by a raw copy -- V4 measured that a NZ-tagged buffer whose bytes we wrote
    ourselves is accepted by the GEMM and bitwise equal, but an ND-tagged
    buffer holding the same bytes is REFUSED (161002);
  * ``npu_format_cast`` must no longer run on the chunk.

Why the segment table is validated once and never in the loop
-------------------------------------------------------------
A full pass is ~9.83M segments (640 per expert x 384 x 40). Per-copy bounds
checks there would cost more than the layout saves, and ``aclrtMemcpy`` has no
``srcMax`` to fall back on. So :func:`bank_build.validate_segments` runs once
per (chunk shape) and the copy loop carries no checks at all.
"""

from __future__ import annotations

import contextlib
import mmap
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from vllm.logger import logger

from vllm_ascend.kt_offload.bank_build import (
    MIN_LOADER_THREADS,
    W13_SECTIONS,
    BankLayout,
    check_loader_placement,
    chunk_segments,
    layer_path,
    load_manifest,
    one_per_l3_cpus,
    validate_segments,
)


class BankSourceError(RuntimeError):
    pass


#: staging group name (``weight_source.GROUPS``) -> bank device-tensor name.
#: The group names are the contract ``stream_io`` reads the staging through;
#: keeping the mapping explicit means a rename on either side fails a check
#: instead of quietly filling the wrong block.
GROUP_TO_TENSOR = {
    "w13_weight": "w13",
    "w2_weight": "w2",
    "w13_scale": "w13_scale",
    "w2_scale": "w2_scale",
}


class BankChunkLayout:
    """``weight_source.ChunkLayout``-compatible description of a bank chunk.

    ``stream_io`` only uses ``.experts``, ``.groups`` and ``.nbytes``, plus
    ``.views(staging)``. Implementing that surface keeps the whole streaming
    path unchanged, which is the point: the only thing that differs is where
    the bytes came from and that they are already in device NZ order.

    Note the per-expert record bytes are IDENTICAL to the ND ones -- NZ is a
    pure byte permutation with no padding (V1 measured the storage ratio as
    1.000000) -- so ``stream_io._check_record_bytes`` passes untouched.
    """

    __slots__ = ("layer", "experts", "groups", "nbytes", "produced")

    def __init__(
        self,
        layer: int,
        experts: tuple[int, ...],
        groups: dict[str, tuple[int, int]],
        nbytes: int,
        produced: tuple[int, ...] | None = None,
    ):
        self.layer = layer
        self.experts = experts
        self.groups = groups
        self.nbytes = nbytes
        #: 🔴 The slots a SUBSET gather filled, or the whole chunk. ``experts``
        #: stays the whole chunk on purpose: it fixes the staging geometry (the
        #: group-major bases and the view sizes below), and deriving that from a
        #: subset would mis-place every group after the first. Only the coverage
        #: and the caller's device-side fill care which slots were produced.
        self.produced = experts if produced is None else produced

    def views(self, staging: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = staging.view(-1)
        c = len(self.experts)
        return {g: flat[off : off + c * rec].view(c, rec) for g, (off, rec) in self.groups.items()}

    def produced_slots(self) -> tuple[int, ...]:
        """Positions within the chunk that this gather filled, ascending."""
        if self.produced is self.experts:
            return tuple(range(len(self.experts)))
        pos = {e: i for i, e in enumerate(self.experts)}
        return tuple(sorted(pos[e] for e in self.produced))


class BankCoverage:
    """``weight_source.Coverage``-compatible counts for one bank gather.

    Must be a real object, not ``None``: ``stream_io`` accumulates
    ``result.coverage.produced_bytes`` on every chunk (``stream_io.py:627``).
    Handing back ``None`` survived the equivalence check and the warmup and only
    died on the first battery request -- the shape of bug a short smoke test
    cannot see.
    """

    __slots__ = ("requested_tensors", "produced_tensors", "produced_bytes", "triples")

    def __init__(self, requested_tensors: int, produced_tensors: int, produced_bytes: int, triples: set):
        self.requested_tensors = requested_tensors
        self.produced_tensors = produced_tensors
        self.produced_bytes = produced_bytes
        self.triples = triples

    @property
    def produced_triples(self) -> int:
        return len(self.triples)


class BankGatherResult:
    """What ``gather`` hands back; mirrors ``weight_source.GatherResult``."""

    __slots__ = ("layout", "coverage")

    def __init__(self, layout: BankChunkLayout, coverage: BankCoverage):
        self.layout = layout
        self.coverage = coverage


class _Strided:
    """One bulk copy: ``nruns`` runs of ``run`` bytes, source contiguous,
    destination strided by ``dst_stride``.

    This is the whole reason the gather is fast. The segment table has 643
    entries per expert, 640 of them 18,432 B w13 part-runs, and driving those
    one at a time from python is the failure mode the project already measured:
    fine-grained python-driven fan-out collapses to 5-6 GiB/s and gets SLOWER
    with more threads (GIL round-trips). Measured here first-hand: 4.5 GiB/s,
    6.6x slower than the checkpoint gather it was supposed to beat.

    But the 160 gidx runs of one (expert, lane) are CONTIGUOUS in the bank and
    EQUALLY SPACED in the staging, so numpy expresses all 160 as one strided
    assignment -- 4 calls per expert instead of 643, a 160x cut in interpreter
    work, with the copying itself done by memcpy inside numpy.
    """

    __slots__ = ("src_offset", "dst_offset", "nruns", "run", "dst_stride")

    def __init__(self, src_offset: int, dst_offset: int, nruns: int, run: int, dst_stride: int):
        self.src_offset = src_offset
        self.dst_offset = dst_offset
        self.nruns = nruns
        self.run = run
        self.dst_stride = dst_stride


def _copy_strided(dst: Any, src: Any, plans: Sequence[_Strided], cpu: int | None = None) -> None:
    """Run a slice of the strided plan. No bounds checks: the table was validated."""
    if cpu is not None:
        # Placement is a performance property, not correctness.
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, {cpu})
    for p in plans:
        if p.nruns == 1:
            dst[p.dst_offset : p.dst_offset + p.run] = src[p.src_offset : p.src_offset + p.run]
            continue
        d = dst[p.dst_offset : p.dst_offset + p.nruns * p.dst_stride]
        d.reshape(p.nruns, p.dst_stride)[:, : p.run] = src[p.src_offset : p.src_offset + p.nruns * p.run].reshape(
            p.nruns, p.run
        )


class BankSource:
    """Reads expert weights out of the NZ bank instead of the checkpoint."""

    #: This source can fill a subset of a chunk's slots (``gather(only_slots=)``),
    #: which ``stream_io`` checks for before offering one -- a source without it
    #: would stream the whole chunk and then H2D only part of it.
    supports_subset_gather = True

    def __init__(
        self,
        bank_dir: str,
        *,
        layout: BankLayout | None = None,
        num_experts: int | None = None,
        copy_threads: int = MIN_LOADER_THREADS,
        pin_cpus: bool = True,
    ):
        self.bank_dir = bank_dir
        man = load_manifest(bank_dir)
        self.manifest = man
        self.layout = (
            layout or BankLayout(**{k: v for k, v in man["layout"].items() if k in BankLayout.__dataclass_fields__})
        ).check()
        if num_experts is not None and int(num_experts) != self.layout.num_experts:
            raise BankSourceError(
                f"the bank was built for {self.layout.num_experts} experts, the model has "
                f"{num_experts}; a bank is not reusable across expert counts"
            )
        self.num_experts = self.layout.num_experts
        self.built_layers = {int(k) for k in man["layers"]}
        self._maps: dict[int, tuple[mmap.mmap, int]] = {}
        self._arrays: dict[int, Any] = {}
        self._lock = threading.Lock()
        # (num_experts_in_chunk) -> (segments, dst_base, nbytes). Built on first
        # use and reused: the table depends only on the chunk SHAPE when the ids
        # are contiguous, so a 12-chunk layer builds it once.
        self._tables: dict[tuple[int, ...], tuple[list, dict[str, int], int]] = {}
        # The frozen loader contract, asserted rather than documented: fewer than
        # four threads costs 12-14% of prefill staging, and threads packed into one
        # CCD drop the host leg from 66 to 26 GiB/s. Neither shows up in a
        # correctness test. ``one_per_l3_cpus`` is the measured-best placement.
        # Override, declared in envs.py. Two uses, and the second is why it
        # ships: the thread count is fixed when this object is built, so an A/B
        # over it needs one server start per arm; AND production asks for 16
        # threads one per L3 domain, which one_per_l3_cpus REFUSES on a host
        # with fewer than 16 domains, so lowering this is the only way to start
        # at all there. Note it also weakens the placement contract below:
        # min_threads becomes min(this, MIN_LOADER_THREADS).
        from vllm_ascend import envs  # noqa: PLC0415 - avoids a module-level cycle

        env = (envs.VLLM_ASCEND_DSV41_KT_LOADER_THREADS or "").strip()
        if env:
            try:
                want = int(env)
            except ValueError:
                raise BankSourceError(
                    f"VLLM_ASCEND_DSV41_KT_LOADER_THREADS={env!r} is not an integer; a loader "
                    "thread count that cannot be read must not silently mean the default"
                ) from None
            if want < 1:
                raise BankSourceError(f"VLLM_ASCEND_DSV41_KT_LOADER_THREADS={want} must be >= 1")
            logger.info(
                "kt_offload bank_source: loader threads %d -> %d (VLLM_ASCEND_DSV41_KT_LOADER_THREADS)",
                int(copy_threads),
                want,
            )
            copy_threads = want
        self.copy_threads = max(1, int(copy_threads))
        self._cpus: list[int] = []
        if self.copy_threads > 1:
            self._cpus = one_per_l3_cpus(self.copy_threads)
            check_loader_placement(self._cpus, min_threads=min(self.copy_threads, MIN_LOADER_THREADS))
        self._pin_cpus = bool(pin_cpus) and bool(self._cpus)
        self._pool: ThreadPoolExecutor | None = None
        logger.info(
            "kt_offload bank_source: %s, %d layer(s), %d experts, %d B/expert",
            bank_dir,
            len(self.built_layers),
            self.num_experts,
            self.layout.bytes_per_expert,
        )
        if self._cpus:
            logger.info(
                "kt_offload bank_source: %d loader thread(s) on cpus %s (one per L3/CCD domain)",
                self.copy_threads,
                self._cpus,
            )

    # -- staging geometry -------------------------------------------------
    def dst_base(self, count: int) -> tuple[dict[str, int], int]:
        """Group-major bases, in ``weight_source.GROUPS`` order.

        The pipeline downstream reads the staging as four contiguous
        ``[count, record]`` blocks in this order, so the bases must follow it
        exactly or the H2D hands the GEMM the wrong block.
        """
        base: dict[str, int] = {}
        off = 0
        for name in ("w13", "w2", "w13_scale", "w2_scale"):
            base[name] = off
            off += count * self.layout.device_record(name)
        return base, off

    def _result(
        self, layer: int, experts: tuple[int, ...], produced: tuple[int, ...], plans: Sequence[Any], nbytes: int
    ) -> BankGatherResult:
        """One gather's layout + counts.

        🔴 Two different expert tuples, and they are not interchangeable.
        ``experts`` is the WHOLE chunk and fixes the staging LAYOUT: the
        group-major bases are ``count * device_record``, so computing them from
        a subset would mis-place every group after the first and the H2D would
        hand the GEMM the wrong block. ``produced`` is what this call actually
        wrote, and it is what the coverage reports -- ``triples`` mirrors the
        checkpoint source's bookkeeping so a downstream coverage check sees the
        same shape and cannot mistake a subset gather for a whole one.
        """
        triples = {(layer, e, proj) for e in produced for proj in ("w1", "w2", "w3")}
        cov = BankCoverage(
            requested_tensors=len(plans), produced_tensors=len(plans), produced_bytes=nbytes, triples=triples
        )
        return BankGatherResult(self.chunk_layout(layer, experts, produced), cov)

    def chunk_layout(
        self, layer: int, experts: tuple[int, ...], produced: tuple[int, ...] | None = None
    ) -> BankChunkLayout:
        """The staging layout for the WHOLE chunk; ``produced`` only names the
        slots a subset gather filled (it does not change a single offset)."""
        base, nbytes = self.dst_base(len(experts))
        groups = {g: (base[t], self.layout.device_record(t)) for g, t in GROUP_TO_TENSOR.items()}
        return BankChunkLayout(layer, experts, groups, nbytes, produced)

    def _table(self, experts: tuple[int, ...], only: frozenset[int] | None = None):
        key = experts if only is None else (experts, tuple(sorted(only)))
        got = self._tables.get(key)
        if got is not None:
            return got
        base, nbytes = self.dst_base(len(experts))
        segs = chunk_segments(self.layout, experts, base)
        # Validated ONCE, here, from the per-segment table -- so the strided plan
        # below is only an execution strategy for a table already proven correct.
        rep = validate_segments(segs, self.layout.layer_bytes, nbytes)
        plans = self._strided_plan(experts, base, only)
        moved = sum(p.nruns * p.run for p in plans)
        # With a subset the staging is still sized for the whole chunk; what the
        # plan must move is the subset's share of it, exactly.
        want = (
            nbytes
            if only is None
            else sum(self.layout.device_record(t) for t in ("w13", "w2", "w13_scale", "w2_scale")) * len(only)
        )
        if moved != want:
            which = "the staging is" if only is None else f"the {len(only)}-slot subset needs"
            raise BankSourceError(
                f"strided plan moves {moved} B but {which} {want} B; the plan and the validated segment table disagree"
            )
        if only is None:
            logger.info(
                "kt_offload bank_source: %d-expert chunk -- %d segments collapsed to %d strided copies, %s",
                len(experts),
                len(segs),
                len(plans),
                rep,
            )
        got = (plans, base, nbytes)
        self._tables[key] = got
        return got

    # -- layer files ------------------------------------------------------
    def _strided_plan(
        self, experts: tuple[int, ...], base: dict[str, int], only: frozenset[int] | None = None
    ) -> list[_Strided]:
        """The segment table expressed as bulk strided copies, part-major.

        ``only`` restricts the plan to those SLOT indices (not expert ids). The
        slots left out are not written at all -- whatever the staging holds for
        them stays, which is why the caller has to be the one that knows those
        slots are filled from somewhere else (the resident parameter, in HBM:
        see PLAN_G_stream_only_offloaded.md). A gather that quietly skipped a
        slot nobody else fills would feed the GEMM the previous chunk's bytes
        with tokens routed to it.
        """
        L = self.layout
        run = L.w13_part_run_bytes
        slab = L.w13_slab_bytes
        ng = L.w13_gidx_count
        out: list[_Strided] = []
        for slot, e in enumerate(experts):
            if only is not None and slot not in only:
                continue
            for lane, name in enumerate(W13_SECTIONS):
                out.append(
                    _Strided(
                        L.expert_offset(name, e),
                        base["w13"] + slot * L.device_record("w13") + lane * run,
                        ng,
                        run,
                        slab,
                    )
                )
            for name in ("w2", "w13_scale", "w2_scale"):
                rec = L.section_record(name)
                out.append(_Strided(L.expert_offset(name, e), base[name] + slot * rec, 1, rec, rec))
        return out

    def _layer_map(self, layer: int) -> tuple[mmap.mmap, int]:
        got = self._maps.get(layer)
        if got is not None:
            return got
        with self._lock:
            got = self._maps.get(layer)
            if got is not None:
                return got
            if layer not in self.built_layers:
                raise BankSourceError(
                    f"layer {layer} is not in the bank ({sorted(self.built_layers)[:4]}...); "
                    "a partially built bank must not be served -- the missing layers would "
                    "read as whatever the staging held last"
                )
            path = layer_path(self.bank_dir, layer)
            size = os.path.getsize(path)
            if size != self.layout.layer_bytes:
                raise BankSourceError(f"layer {layer} file is {size} B, the layout says {self.layout.layer_bytes}")
            fd = os.open(path, os.O_RDONLY)
            try:
                # r--s: read-only SHARED. kt throws on PRIVATE+WRITABLE, and a
                # private map would also fault in a second copy of every page.
                mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
            finally:
                os.close(fd)
            got = (mm, size)
            self._maps[layer] = got
            return got

    def _layer_array(self, layer: int):
        """Zero-copy uint8 numpy view of the layer file.

        Cached because ``np.frombuffer`` holds a buffer export on the mmap: the
        array must be dropped before ``mmap.close()`` or close raises
        ``BufferError`` -- and on an exception path that BufferError would mask
        the real one. :meth:`close` drops these first.
        """
        got = self._arrays.get(layer)
        if got is None:
            mm, _ = self._layer_map(layer)
            got = np.frombuffer(mm, dtype=np.uint8)
            self._arrays[layer] = got
        return got

    # -- the seam ---------------------------------------------------------
    def gather(
        self, layer: int, experts: Sequence[int], staging: torch.Tensor, only_slots: Sequence[int] | None = None
    ) -> BankGatherResult:
        """Fill ``staging`` with ``experts`` of ``layer`` in device NZ layout.

        ``only_slots`` fills just those positions of the chunk and leaves the
        rest of the staging untouched. The caller is then responsible for those
        slots -- in production they are the experts already resident in HBM,
        whose layer-buffer slots are filled by a device-to-device copy instead
        (26% of the streamed bytes, bit-identically: see
        measure/PLAN_G_stream_only_offloaded.md). The returned coverage counts
        the slots this call actually produced, not the chunk, so a downstream
        coverage check cannot mistake a subset gather for a whole one.
        """
        experts = tuple(int(e) for e in experts)
        if not experts:
            raise BankSourceError(f"layer {layer}: empty expert chunk")
        if len(set(experts)) != len(experts):
            raise BankSourceError(f"layer {layer}: duplicate expert ids {experts}")
        if staging.dtype != torch.uint8 or staging.device.type != "cpu" or not staging.is_contiguous():
            raise TypeError(f"staging must be a contiguous CPU uint8 tensor, got {staging.dtype} on {staging.device}")
        only = None
        if only_slots is not None:
            only = frozenset(int(s) for s in only_slots)
            bad = sorted(s for s in only if not 0 <= s < len(experts))
            if bad:
                raise BankSourceError(
                    f"only_slots {bad} fall outside the {len(experts)}-slot chunk; a slot index is "
                    "a position in this chunk, not an expert id"
                )
            if not only:
                raise BankSourceError(
                    "only_slots is empty: a gather that fills nothing is a zero-coverage call, and "
                    "the caller would then H2D whatever the staging held last"
                )
        plans, _, nbytes = self._table(experts, only)
        if staging.numel() < nbytes:
            raise BankSourceError(f"staging holds {staging.numel()} B, chunk needs {nbytes} B")
        src = self._layer_array(layer)
        view = staging.view(-1).numpy()
        produced = tuple(experts) if only is None else tuple(e for i, e in enumerate(experts) if i in only)
        moved = sum(p.nruns * p.run for p in plans)
        if self.copy_threads == 1:
            _copy_strided(view, src, plans)
            return self._result(layer, tuple(experts), produced, plans, moved)
        # Split by SEGMENT INDEX, not by byte range: the table is part-major, so
        # contiguous index ranges are contiguous source runs, which is the access
        # pattern the layout was measured with (four sequential streams beat one
        # interleaved stream by 1.4-1.6%).
        n, t = len(plans), self.copy_threads
        bounds = [(i * n // t, (i + 1) * n // t) for i in range(t)]
        futs = [
            self._executor().submit(_copy_strided, view, src, plans[a:b], self._cpus[i] if self._pin_cpus else None)
            for i, (a, b) in enumerate(bounds)
            if a < b
        ]
        for f in futs:
            f.result()
        return self._result(layer, tuple(experts), produced, plans, moved)

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.copy_threads, thread_name_prefix="kt_bank")
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        with self._lock:
            # Drop the numpy views BEFORE closing: each holds a buffer export.
            self._arrays.clear()
            for mm, _ in self._maps.values():
                mm.close()
            self._maps.clear()


__all__ = ["BankSource", "BankSourceError"]
