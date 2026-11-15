# SPDX-License-Identifier: Apache-2.0
"""NUMA placement of the checkpoint pages the hybrid zero-copy path shares.

Two consumers read the *same physical pages* -- the page cache of the model's
safetensors shards: **kt-kernel** for ``w13`` only
(``KTOffloadConfig.zerocopy_w13``; its CPU expert weights point at the mapped
checkpoint instead of a memcpy into this node's BufferB, while ``w2`` keeps the
copy path), and **streaming prefill** (``weight_source.py``) for ``w13``,
``w2`` *and* every ``.scale``. One set of pages can have only one owner, so
placement is decided here by file offset and kt's own pinning/migration is
switched off (``KT_ZEROCOPY_MLOCK=0``). Nothing here touches the NPU.

Two legs, because neither syscall covers the other's half:

* a page **already in the page cache** -> ``move_pages(MPOL_MF_MOVE_ALL)``.
  ``mbind``/``set_mempolicy`` cannot touch it: they bind what the kernel
  allocates *from then on*, and a cached page is reused where it is.
* a page **not in the page cache** -> ``move_pages`` cannot touch it: nothing
  to migrate, and it would be allocated on whatever node first reads it. So we
  read it ourselves under ``set_mempolicy(MPOL_BIND, {target})``. On a cold
  checkpoint this is by far the larger leg, and it **reads from disk**: its
  throughput is reported, and it is the start-up cost.

``move_pages`` needs ``MPOL_MF_MOVE_ALL`` (4); plain ``MPOL_MF_MOVE`` (2)
leaves a large ``-EBUSY`` tail on file pages mapped by more than one process --
our case, since kt maps them too. A page must also be **in this process's page
table** before it can be named: a fresh ``mmap`` that has not been dereferenced
gives ``-ENOENT`` for every page, so we touch one byte per page first -- in the
migrate leg only pages ``mincore`` reports as resident, so it never turns into
a disk read behind your back; in the fault-in leg non-resident pages on
purpose, charged to ``fault_seconds``. A failing ``move_pages`` is *graceful*
(the page stays put, locality gets worse) where ``mbind`` is a hard constraint
that can force reclaim or OOM on a full node.

The per-page ``status[]`` check is not optional: the syscall's own return value
has reported ``policy_failures=0`` on a run where not one page had moved. Every
count below comes from ``status[i]``, and ``needed_move`` (pages *not already*
on their target before the call) lets a caller assert the mutation was applied
-- without it an implementation that does nothing also "succeeds".
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# The vllm logger, not a module-level ``vllm_ascend.*`` one: in the worker
# process only ``configure_ascend_file_logging()`` runs, which never sets the
# ``vllm_ascend`` logger's level, so an INFO under that name is dropped at the
# level gate and never reaches serve.log.
from vllm.logger import logger

PAGE_SIZE = 4096
GIB = 1 << 30

# mm/mempolicy.h
MPOL_MF_MOVE = 2
MPOL_MF_MOVE_ALL = 4

# sys/mman.h
_PROT_READ = 0x1
_MAP_SHARED = 0x01

#: ``madvise(MADV_POPULATE_READ)``, Linux 5.14+. Populates the page-table
#: entries for a range and **blocks** until its pages are in the page cache and
#: up to date -- one syscall for a whole run, and ctypes releases the GIL for
#: it. Reading one byte per page from Python instead does the same work while
#: holding the GIL across every major fault, so N threads serialise into one
#: I/O at a time.
MADV_POPULATE_READ = 22

# ``move_pages`` has no libc wrapper in glibc's public headers we can rely on,
# so it goes through ``syscall(2)`` and the number is per-architecture. An
# unknown machine raises rather than guessing: a wrong number is a *different*
# syscall.
_SYS_MOVE_PAGES = {"x86_64": 279, "aarch64": 239}
_SYS_SET_MEMPOLICY = {"x86_64": 238, "aarch64": 237}
_SYS_GET_MEMPOLICY = {"x86_64": 239, "aarch64": 236}
_SYS_READAHEAD = {"x86_64": 187, "aarch64": 213}

# linux/mempolicy.h. The *allocation* policies: what places a page that does
# not exist yet -- the other half of this module.
MPOL_DEFAULT = 0
MPOL_PREFERRED = 1
MPOL_BIND = 2

#: ``set_mempolicy`` nodemask is an array of unsigned longs; two nodes fit one
#: word. A node id past this needs a wider mask and is refused rather than
#: silently truncated into the wrong node.
_NODEMASK_BITS = 64

NODE_SYSFS = "/sys/devices/system/node"


class PlacementError(RuntimeError):
    """The placement could not be planned or applied as asked."""


class ZeroPlacementError(PlacementError):
    """A plan or a migration covered nothing.

    Never a silent pass: "no pages planned" and "no pages resident" both look
    like a successful no-op in any statistic that is only a ratio.
    """


class MutationNotAppliedError(PlacementError):
    """``needed_move == 0``: every page was already on its target node.

    The run may be healthy, but it is **not** evidence that this module placed
    anything, and it cannot serve as an arm of an A/B.
    """


# ---------------------------------------------------------------------------
# 1. the inputs to the derivation: what each node can actually give
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeMemory:
    """One node's memory, in bytes, as ``/sys/.../nodeN/meminfo`` reported it."""

    node: int
    mem_free: int
    file_pages: int
    shmem: int

    @property
    def reclaimable_file_cache(self) -> int:
        """File cache we could take back: ``FilePages - Shmem``.

        Shmem counts in FilePages but there is no swap here (``SwapTotal=0``),
        so it cannot be reclaimed -- and on node0 most of it is engram's, which
        no plan of ours may move. Leaving it in would overstate node0's capacity
        by more than a factor of two.
        """
        return max(0, self.file_pages - self.shmem)

    def available(self, reserve_bytes: int) -> int:
        return self.mem_free + self.reclaimable_file_cache - reserve_bytes

    def __str__(self) -> str:
        return (
            f"node{self.node}: MemFree={self.mem_free / GIB:.1f} GiB "
            f"reclaimable_file_cache={self.reclaimable_file_cache / GIB:.1f} GiB "
            f"(FilePages={self.file_pages / GIB:.1f} - Shmem={self.shmem / GIB:.1f})"
        )


def read_node_memory(nodes: Sequence[int], *, sysfs: str = NODE_SYSFS) -> dict[int, NodeMemory]:
    """Read ``MemFree`` / ``FilePages`` / ``Shmem`` per node, in bytes.

    Separate from :func:`derive_fraction` so the derivation stays a pure
    function a unit test can feed readings to.
    """
    want = {"MemFree", "FilePages", "Shmem"}
    out: dict[int, NodeMemory] = {}
    for n in nodes:
        path = os.path.join(sysfs, f"node{n}", "meminfo")
        fields: dict[str, int] = {}
        with open(path) as f:
            for line in f:
                # The same file also carries unitless counts ("Node 0
                # HugePages_Total: 0"), so the unit is checked per line.
                parts = line.split()
                if len(parts) < 4 or parts[0] != "Node":
                    continue
                key = parts[2].rstrip(":")
                if parts[-1] != "kB":
                    if key in want:
                        raise PlacementError(f"{path}: {key} is not in kB: {line.strip()!r}")
                    continue
                fields[key] = int(parts[-2]) * 1024
        missing = want - fields.keys()
        if missing:
            raise PlacementError(f"{path}: missing {sorted(missing)}")
        out[n] = NodeMemory(n, fields["MemFree"], fields["FilePages"], fields["Shmem"])
    return out


def derive_fraction(
    reserve_gib: float,
    tp: int,
    *,
    need_bytes_per_part: int,
    memory: Mapping[int, NodeMemory] | None = None,
    nodes: Sequence[int] | None = None,
) -> list[float]:
    """``fraction[p] = min(1, what node p can give / what part p needs)``.

    ``available_i = MemFree_i + reclaimable_file_cache_i - reserve``. ``memory``
    is injectable so this stays a pure function under test; ``None`` reads
    ``/sys`` now. Pass the *same* derived list to both arms of an A/B: what a
    node can give drifts by tens of GiB within one night on this host, so two
    arms that each derive their own ratio are not comparing the switch.
    """
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    if reserve_gib < 0:
        raise ValueError(f"reserve_gib must be >= 0, got {reserve_gib}")
    if need_bytes_per_part <= 0:
        raise ValueError(
            f"need_bytes_per_part must be > 0, got {need_bytes_per_part}. Without it there is "
            "no way to say whether a node can hold its part, and the only answer that does not "
            "need it -- a normalised split -- is the bug this parameter exists to prevent.")
    node_ids = list(nodes) if nodes is not None else list(range(tp))
    if len(node_ids) != tp:
        raise ValueError(f"nodes lists {len(node_ids)} node(s) but tp is {tp}")
    mem = read_node_memory(node_ids) if memory is None else {n: memory[n] for n in node_ids}

    reserve = int(reserve_gib * GIB)
    # The readings the ratio is derived FROM: a bare ratio in a log is
    # unfalsifiable a day later.
    for n in node_ids:
        logger.info("kt numa_placement: %s reserve=%.1f GiB avail=%.1f GiB",
                    mem[n], reserve_gib, mem[n].available(reserve) / GIB)

    avail = [max(0, mem[n].available(reserve)) for n in node_ids]
    if sum(avail) <= 0:
        # No "reasonable" fallback: a uniform split would place bytes on a
        # node with no room, under a perfectly ordinary-looking 50/50 in the
        # log.
        raise PlacementError(
            f"no node has memory left after reserve={reserve_gib} GiB: "
            + ", ".join(f"node{n} avail={mem[n].available(reserve) / GIB:.1f} GiB" for n in node_ids)
        )
    # Each entry is independent: part p keeps all of its own bytes local unless
    # node p cannot hold them. Normalising by sum(avail) instead -- which this
    # used to do -- makes the entries sum to 1, and since fraction[p] is p's
    # OWN local share, that pins the mean local share at 1/tp whatever the
    # memory readings are.
    return [min(1.0, a / need_bytes_per_part) for a in avail]


def validate_fraction(fraction: Sequence[float], tp: int) -> tuple[float, ...]:
    """Length ``tp``, every entry in [0, 1]. Entries are INDEPENDENT.

    Deliberately NO "sums to 1" rule: ``fraction[p]`` is *the share of part p's
    own bytes that stays on part p's node*, not p's share of a global
    distribution, so perfect locality is ``[1.0] * tp``, which sums to ``tp``.
    Under a sum-to-1 rule the mean local share is ``1/tp`` **whatever values are
    chosen** -- with tp=2 half of w13 is always read across the socket.
    """
    f = tuple(float(x) for x in fraction)
    if len(f) != tp:
        raise ValueError(f"fraction lists {len(f)} value(s) but tp is {tp}")
    if any(x < 0.0 or x > 1.0 for x in f):
        raise ValueError(f"fraction={list(f)} has a value outside [0, 1]")
    return f


def mirror_fraction(fraction: Sequence[float]) -> list[float]:
    """The positive control: the same split, reversed.

    Pages are somewhere whatever we do, so "placement succeeded" proves nothing.
    The mirrored fraction must produce a *visibly different* measured
    distribution; if it does not, the placement did not happen.
    """
    return list(reversed([float(x) for x in fraction]))


# ---------------------------------------------------------------------------
# 2. the plan: which byte range of which file goes to which node
# ---------------------------------------------------------------------------

#: Tensors kt reads zero-copy *and* streaming reads: split across nodes by TP
#: part. Each is ``[inter, hidden]`` row-major and TP slices ``inter``, i.e.
#: whole **rows**, so part p owns the contiguous byte interval
#: ``[p*nbytes/tp, (p+1)*nbytes/tp)``.
SPLIT_TENSORS: tuple[tuple[str, str], ...] = (("w1", "weight"), ("w3", "weight"))

#: Tensors only streaming prefill reads: they go to the streaming node whole.
#:
#: ``w2`` is here NOT because it is small enough. ``w2`` is ``[hidden, inter]``
#: and TP slices ``inter``, which in a row-major tensor is a **column** slice:
#: every part owns a strip of every row, so its bytes are not contiguous and a
#: 4 KiB page carries bytes both parts want. Per-page placement can therefore
#: make ``w2`` at most ``1/tp`` local, which is why ``w2`` keeps kt's copy path
#: (memcpy into this node's BufferB) and why the only consumer of its *file
#: pages* is streaming prefill. Do not "unify" this with the branch above by
#: slicing w2 the same way: the slices would be wrong, and they would look
#: right.
STREAMING_ONLY_TENSORS: tuple[tuple[str, str], ...] = (
    ("w2", "weight"),
    ("w1", "scale"),
    ("w3", "scale"),
    ("w2", "scale"),
)


@dataclass(frozen=True)
class PlacementSegment:
    """One contiguous byte interval of one file, and the node it should be on."""

    path: str
    start: int
    nbytes: int
    node: int
    layer: int
    expert: int
    proj: str
    kind: str
    part: int | None  # TP part for split tensors; None for whole-tensor ones
    reason: str  # "w13-local" | "w13-overflow" | "streaming-only"

    @property
    def end(self) -> int:
        return self.start + self.nbytes


@dataclass(frozen=True)
class PlacementPlan:
    segments: tuple[PlacementSegment, ...]
    fraction: tuple[float, ...]
    fraction_source: str  # "derived" | "configured"
    part_nodes: tuple[int, ...]
    streaming_node: int
    #: file -> ascending, non-overlapping ``(first_page, last_page_exclusive, node)``
    page_runs: dict[str, tuple[tuple[int, int, int], ...]]

    @property
    def planned_pages(self) -> int:
        return sum(hi - lo for runs in self.page_runs.values() for lo, hi, _ in runs)

    def planned_pages_by_node(self) -> dict[int, int]:
        out: Counter[int] = Counter()
        for runs in self.page_runs.values():
            for lo, hi, node in runs:
                out[node] += hi - lo
        return dict(out)

    @property
    def split_tensor_names(self) -> tuple[str, ...]:
        """The tensors ``fraction`` actually divides, read off the segments.

        🔴 Which tensors those are is a property of the SOURCE, not a constant:
        the checkpoint splits w13 only (its w2 is a column slice of every row,
        so a 4 KiB page carries both parts' bytes), while the NZ bank splits w2
        too (TP divides down's k and in NZ the k-block index is outermost, so a
        part is a contiguous, page-exclusive record). Reporting a hard-coded
        "w13" made ``explain()`` claim w2 went whole to the streaming node while
        the very number beside it -- 0.9412 = (w13 + w2)/total -- proved it had
        been split.
        """
        return tuple(sorted({f"{s.proj}.{s.kind}" for s in self.segments if s.part is not None}))

    @property
    def whole_tensor_names(self) -> tuple[str, ...]:
        """The tensors that go whole to ``streaming_node``; see above."""
        return tuple(sorted({f"{s.proj}.{s.kind}" for s in self.segments if s.part is None}))

    @property
    def split_share(self) -> float:
        """Fraction of the planned bytes that ``fraction`` actually governs.

        ``fraction`` divides the tensors ``split_tensor_names`` lists -- w13 for
        the checkpoint, w13 **and w2** for the bank; whatever is left goes whole
        to ``streaming_node``. A node's share of *all* planned pages is
        therefore ``fraction[node] * split_share``, NOT ``fraction[node]``.
        """
        split = sum(s.nbytes for s in self.segments if s.part is not None)
        total = sum(s.nbytes for s in self.segments)
        return split / total if total else 0.0

    def explain(self) -> str:
        """The four percentages this module reports, each with its denominator.

        They are over four different populations: ``fraction[p]`` is a share
        **of the split tensors**, the plan's per-node percentages are over **all
        planned pages** (``fraction[p] * split_share``). Comparing them directly
        reads as a placement failure that does not exist.
        """
        by, n = self.planned_pages_by_node(), self.planned_pages
        rows = [
            f"  fraction          = {[round(f, 4) for f in self.fraction]}  "
            f"({self.fraction_source}) -- share OF W13 ONLY, per TP part",
            f"  split_share       = {self.split_share:.4f} -- the share of the planned bytes "
            f"`fraction` divides ({', '.join(self.split_tensor_names) or 'nothing'}); the rest "
            f"({1 - self.split_share:.4f}) is "
            f"{', '.join(self.whole_tensor_names) or 'nothing'} and goes whole to "
            f"node{self.streaming_node}",
            "  plan, of ALL planned pages = "
            + str({k: f"{100 * v / max(1, n):.2f}%" for k, v in sorted(by.items())}),
        ]
        for i, node in enumerate(self.part_nodes):
            if node != self.streaming_node:
                rows.append(
                    f"    check: fraction[{i}] x split_share = {self.fraction[i]:.4f} x "
                    f"{self.split_share:.4f} = {100 * self.fraction[i] * self.split_share:.2f}% "
                    f"= node{node}'s all-pages share above")
        rows.append(
            "  at runtime: landed/asked_for are OF THE FAULTED-IN SUBSET, and "
            "FINAL is OF ALL RESIDENT PLANNED PAGES -- neither is comparable to `fraction`")
        return "plan denominators:\n" + "\n".join(rows)

    def planned_bytes_by_node(self) -> dict[int, int]:
        out: Counter[int] = Counter()
        for s in self.segments:
            out[s.node] += s.nbytes
        return dict(out)


def _split_ranges(start: int, nbytes: int, tp: int) -> list[tuple[int, int]]:
    if nbytes % tp:
        raise PlacementError(
            f"tensor of {nbytes} B does not divide into {tp} TP parts; the byte interval "
            f"a part owns would not be exact and every page below it would be misplaced"
        )
    step = nbytes // tp
    return [(start + p * step, step) for p in range(tp)]


def _place_part(
    part: int,
    start: int,
    length: int,
    fraction: Sequence[float],
    part_nodes: Sequence[int],
) -> list[tuple[int, int, int, str]]:
    """``(start, nbytes, node, reason)`` for one TP part's byte interval.

    ``fraction[part]`` of the interval stays on ``part_nodes[part]`` as a
    **contiguous prefix**; the rest overflows to the other nodes in proportion
    to their own fraction, again as contiguous intervals in node order.
    Contiguous, not page-striped, on purpose: any per-page-period assignment
    resonates with any per-page-period sampling, so a strided audit can read a
    true 50/50 as 100% on one node. Prefixes have no period to hit.
    """
    tp = len(fraction)
    local = int(round(fraction[part] * length))
    local = max(0, min(length, local))
    out: list[tuple[int, int, int, str]] = []
    if local:
        out.append((start, local, part_nodes[part], "w13-local"))
    rest = length - local
    if not rest:
        return out
    others = [p for p in range(tp) if p != part]
    weights = [fraction[p] for p in others]
    wsum = sum(weights)
    if wsum <= 0:
        # fraction[part] == 1 but rounding left a tail, or a degenerate split:
        # spread what is left evenly rather than dropping it.
        weights = [1.0] * len(others)
        wsum = float(len(others))
    off = start + local
    for i, p in enumerate(others):
        n = rest if i == len(others) - 1 else int(round(rest * weights[i] / wsum))
        n = max(0, min(n, start + length - off))
        if n:
            out.append((off, n, part_nodes[p], "w13-overflow"))
            off += n
        rest -= n
    return out


def _page_runs(segments: Iterable[PlacementSegment]) -> dict[str, tuple[tuple[int, int, int], ...]]:
    """Byte segments -> exclusive page ownership per file.

    Every boundary is rounded to the **nearest** page boundary, so a page goes
    to whichever segment holds most of it: ownership stays exclusive (no page is
    migrated twice in a race) and pages mostly belonging to somebody else's
    tensor are declined.
    """
    per_file: dict[str, list[PlacementSegment]] = {}
    for s in segments:
        per_file.setdefault(s.path, []).append(s)
    out: dict[str, tuple[tuple[int, int, int], ...]] = {}
    for path, segs in per_file.items():
        segs.sort(key=lambda s: s.start)
        prev_end = -1
        runs: list[list[int]] = []
        for s in segs:
            if s.start < prev_end:
                raise PlacementError(
                    f"{path}: segments overlap at byte {s.start} ({s.layer}/{s.expert}/{s.proj}."
                    f"{s.kind}); a page would have two owners"
                )
            prev_end = s.end
            lo = (s.start + PAGE_SIZE // 2) // PAGE_SIZE
            hi = (s.end + PAGE_SIZE // 2) // PAGE_SIZE
            if hi <= lo:
                continue  # smaller than a page after rounding: owns nothing
            if runs and runs[-1][2] == s.node and runs[-1][1] == lo:
                runs[-1][1] = hi
            else:
                runs.append([lo, hi, s.node])
        if runs:
            out[path] = tuple((lo, hi, n) for lo, hi, n in runs)
    return out


def plan_placement(
    source,
    layers: Sequence[int],
    experts: Sequence[int],
    *,
    tp: int,
    part_nodes: Sequence[int] | None = None,
    streaming_node: int = 0,
    fraction: Sequence[float] | None = None,
    fraction_source: str | None = None,
    reserve_gib: float = 16.0,
) -> PlacementPlan:
    """Where every byte of ``layers`` x ``experts`` should live.

    ``source`` is anything with ``locate(layer, expert, proj, kind) -> ByteSpan``
    (``weight_source.WeightSource``). ``fraction=None`` derives the split from
    the nodes' memory now; anything else is taken as pinned by the operator and
    only validated.
    """
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    layers, experts = tuple(layers), tuple(experts)
    nodes = tuple(range(tp)) if part_nodes is None else tuple(int(n) for n in part_nodes)
    if len(nodes) != tp:
        raise ValueError(f"part_nodes lists {len(nodes)} node(s) but tp is {tp}")
    if fraction is None:
        # What one part must hold locally: the W13 bytes this plan covers,
        # split tp ways. Deriving without it is what pinned the split at 1/tp.
        w13_bytes = sum(source.locate(layer, expert, proj, kind).nbytes
                        for layer in layers for expert in experts
                        for proj, kind in tuple(getattr(source, "split_tensors", SPLIT_TENSORS)))
        frac = tuple(derive_fraction(reserve_gib, tp, nodes=nodes,
                                     need_bytes_per_part=w13_bytes // tp))
        source_kind = "derived"
    else:
        frac = validate_fraction(fraction, tp)
        # A caller that derived the split itself says so: a derived ratio and
        # one pinned by the operator are different claims, and only one of them
        # is reproducible.
        source_kind = fraction_source or "configured"
    # A *different* claim from the report logged after the run: deriving a
    # ratio is not landing it.
    logger.info(
        "kt numa_placement: fraction=%s (%s, OF W13 ONLY) part_nodes=%s streaming_node=%d",
        [round(x, 4) for x in frac], source_kind, list(nodes), streaming_node,
    )

    # A source may own the question "which tensors split per TP part, and where
    # is part p's byte range", because the answer is a property of its LAYOUT,
    # not of this module. The checkpoint's answer is the default below; the NZ
    # bank's is different in two ways, and both matter:
    #   * its w13 parts are separate SECTIONS, so part p is not
    #     `[p*nbytes/tp, ...)` of one span -- it is a different file offset;
    #   * its w2 CAN be split, because TP divides down's k and in NZ the k-block
    #     index is the outermost axis, so part p is a contiguous page-aligned
    #     prefix. In the checkpoint w2 is a column slice of every row and cannot
    #     be split at all, which is what the STREAMING_ONLY_TENSORS note below
    #     is about. Do not read the bank's behaviour as contradicting it.
    split_tensors = tuple(getattr(source, "split_tensors", SPLIT_TENSORS))
    stream_only = tuple(t for t in (SPLIT_TENSORS + STREAMING_ONLY_TENSORS)
                        if t not in split_tensors)
    split_ranges = getattr(source, "split_ranges", None)

    segs: list[PlacementSegment] = []
    for layer in layers:
        for expert in experts:
            for proj, kind in split_tensors:
                if split_ranges is not None:
                    parts = split_ranges(layer, expert, proj, kind, tp)
                else:
                    span = source.locate(layer, expert, proj, kind)
                    parts = [(span.path, st, nb)
                             for st, nb in _split_ranges(span.start, span.nbytes, tp)]
                if len(parts) != tp:
                    raise ValueError(
                        f"split_ranges({layer}, {expert}, {proj}, {kind}) returned {len(parts)} "
                        f"range(s), expected one per TP part ({tp})")
                for part, (path, s, n) in enumerate(parts):
                    for st, nb, node, reason in _place_part(part, s, n, frac, nodes):
                        segs.append(PlacementSegment(path, st, nb, node, layer, expert,
                                                     proj, kind, part, reason))
            for proj, kind in stream_only:
                span = source.locate(layer, expert, proj, kind)
                segs.append(PlacementSegment(span.path, span.start, span.nbytes, streaming_node,
                                             layer, expert, proj, kind, None, "streaming-only"))
    runs = _page_runs(segs)
    plan = PlacementPlan(tuple(segs), frac, source_kind, nodes, streaming_node, runs)
    # Every percentage this module prints, with the population it is over.
    logger.info("kt numa_placement: %s", plan.explain())
    if plan.planned_pages == 0:
        raise ZeroPlacementError(
            f"planned 0 pages from {len(segs)} segment(s) over layers={list(layers)} "
            f"experts={len(experts)}; a placement that covers nothing must not pass"
        )
    return plan


# ---------------------------------------------------------------------------
# 3. execution: migrate the resident pages
# ---------------------------------------------------------------------------


@dataclass
class MigrationReport:
    planned_pages: int = 0
    #: planned pages already in the page cache at the census. This is the share
    #: move_pages can do anything about.
    resident_pages: int = 0
    #: pages the migrate leg actually examined = census-resident PLUS whatever
    #: the fault-in leg brought in. The denominator for the percentages below;
    #: ``resident_pages`` there turns an all-cold file's 0 pages into a nonsense
    #: percentage.
    examined_pages: int = 0
    #: pages that were NOT already on their target node before the call. The
    #: number an arm asserts on: without it a no-op also "succeeds".
    needed_move: int = 0
    #: pages whose post-migration ``status[i]`` IS the target node. A page that
    #: was already there counts here too -- "landed", not "was carried".
    moved: int = 0
    #: pages whose ``status[i]`` is anything else: a negative errno, or another
    #: node. A small steady tail of ``-ENOENT`` is the mincore -> move_pages
    #: race (resident when we looked, reclaimed before the syscall named it),
    #: not a placement failure.
    failed_pages: int = 0
    #: pages re-offered to move_pages after a first -EBUSY (see apply_placement)
    retried_pages: int = 0
    seconds: float = 0.0
    before_by_node: dict[int, int] = field(default_factory=dict)
    after_by_node: dict[int, int] = field(default_factory=dict)
    #: errno name -> page count, from ``status[i] < 0``
    errno_pages: dict[str, int] = field(default_factory=dict)
    #: where ALL resident planned pages ended up, not just the examined ones.
    #: ``after_by_node`` covers only what the migrate leg looked at, which under
    #: ``migrate_scope="suspect"`` would read as if placement had shrunk.
    #: Assembled for free from the two legs, this is the number to compare
    #: across scopes; ``final_on_target`` / ``final_off_target`` score the same
    #: population against each page's own planned node. It is the state **at the
    #: moment each page was placed**, not a later re-read: placement is not
    #: permanent -- these are shared file pages and the kernel keeps its own
    #: counsel. ``audit_placement`` re-derives it from scratch and answers
    #: "where are they now".
    final_by_node: dict[int, int] = field(default_factory=dict)
    final_on_target: int = 0
    final_off_target: int = 0

    # -- the fault-in leg: pages that were NOT in the page cache -------------
    # move_pages can only place a page that exists. On a cold checkpoint most
    # planned pages are not resident; those are placed by the *allocation*
    # policy -- set_mempolicy(MPOL_BIND) around the read that creates them.

    #: planned pages that were not resident at the census
    fault_planned_pages: int = 0
    #: of those, the ones this run actually brought in (mincore says so
    #: afterwards). The ``needed_move`` of this leg: 0 means no evidence.
    faulted_in: int = 0
    #: still not resident after the read. Never counted as success.
    fault_failed_pages: int = 0
    #: what the fault-in leg was ASKED to do: the planned node of the pages it
    #: was going to read, over exactly the pages in ``fault_by_node``. Without
    #: it ``landed`` has no honest comparator -- ``landed`` is over the
    #: **non-resident** pages only while the plan's percentages are over **all**
    #: planned pages, and the two populations differ by whatever was already
    #: cached, which is not spread evenly over the nodes. Comparing them reads
    #: as a placement failure that may be pure denominator.
    fault_target_by_node: dict[int, int] = field(default_factory=dict)
    #: where the newly read pages landed, measured BEFORE any migration -- this
    #: is what the bind-node0 / bind-node1 control compares. If the migrate leg
    #: runs afterwards it will correct mistakes, and then both arms look right.
    fault_by_node: dict[int, int] = field(default_factory=dict)
    #: faulted pages whose node is not the one they were bound to. The cause is
    #: read-ahead spilling past a run: the pages read beyond a run's end are
    #: allocated under the policy still in force for that run. It grows with the
    #: read-ahead size but plateaus at the device's ``read_ahead_kb``. It is
    #: also why the migrate leg runs AFTER the fault leg rather than instead of
    #: it: move_pages corrects exactly these.
    fault_misplaced: int = 0
    #: wall clock of the fault-in leg alone. This is disk read time and it is
    #: what decides start-up cost.
    fault_seconds: float = 0.0

    @property
    def fault_bytes(self) -> int:
        return self.faulted_in * PAGE_SIZE

    @property
    def fault_gib_per_s(self) -> float:
        return (self.fault_bytes / GIB / self.fault_seconds) if self.fault_seconds > 0 else 0.0

    def merge(self, other: MigrationReport) -> None:
        for name in ("planned_pages", "resident_pages", "examined_pages", "needed_move", "moved",
                     "failed_pages", "retried_pages", "fault_planned_pages",
                     "faulted_in", "fault_failed_pages", "fault_misplaced",
                     "final_on_target", "final_off_target"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for dst, src in ((self.before_by_node, other.before_by_node),
                         (self.after_by_node, other.after_by_node),
                         (self.fault_by_node, other.fault_by_node),
                         (self.fault_target_by_node, other.fault_target_by_node),
                         (self.final_by_node, other.final_by_node),
                         (self.errno_pages, other.errno_pages)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v

    def require_mutation_applied(self) -> MigrationReport:
        """Raise unless this run actually had something to change.

        An arm whose pages were already on their targets placed nothing, so it
        cannot be the treated arm of an A/B however good its numbers look.
        """
        if self.needed_move == 0:
            raise MutationNotAppliedError(
                f"needed_move=0 of {self.examined_pages} examined page(s): every page was already "
                f"on its target node, so this run is not evidence that the migrate leg placed "
                f"anything"
            )
        return self

    def require_applied(self) -> MigrationReport:
        """The assertion a driver should use: every leg that HAD work did it.

        Each leg alone can be a legitimate no-op (an all-cold file gives the
        migrate leg nothing to move, an all-hot one gives the fault-in leg
        nothing to read), so asserting both unconditionally would fail a good
        run. What must never pass is a run where *neither* leg applied anything,
        or where a leg that had work applied none of it.
        """
        # Gate on ``resident_pages`` (resident at the census), not
        # ``examined_pages``: the migrate leg is only answerable for pages that
        # were already cached. Pages the fault-in leg brought in are examined
        # too and come back ``needed_move=0`` precisely *because* the bind
        # worked, so gating on examined_pages turns a successful fault-in into
        # "the mutation was not applied".
        if self.resident_pages:
            self.require_mutation_applied()
        if self.fault_planned_pages and (self.faulted_in or self.fault_failed_pages):
            self.require_fault_applied()
        if self.needed_move == 0 and self.faulted_in == 0:
            raise MutationNotAppliedError(
                f"neither leg applied anything (needed_move=0, faulted_in=0) over "
                f"{self.planned_pages} planned page(s); this is not an arm"
            )
        return self

    def require_fault_applied(self) -> MigrationReport:
        """The fault-in leg's version of the same assertion.

        ``faulted_in == 0`` means every planned page was already cached, so
        ``set_mempolicy`` placed nothing and this arm cannot be compared
        against a differently-bound one.
        """
        if self.faulted_in == 0:
            raise MutationNotAppliedError(
                f"faulted_in=0 of {self.fault_planned_pages} non-resident planned page(s): the "
                f"allocation policy placed nothing, so this run is not evidence of fault-in "
                f"placement"
            )
        return self

    def fault_str(self) -> str:
        n = self.faulted_in or 1
        t = sum(self.fault_target_by_node.values()) or 1
        landed = {k: f"{100 * v / n:.2f}%" for k, v in sorted(self.fault_by_node.items())}
        # The comparator for `landed`, over the SAME pages: a gap between
        # these two is the bind actually missing, while a gap against the plan's
        # global percentages may be nothing but the denominator.
        want = {k: f"{100 * v / t:.2f}%" for k, v in sorted(self.fault_target_by_node.items())}
        return (
            f"fault-in: planned={self.fault_planned_pages} faulted_in={self.faulted_in} "
            f"failed={self.fault_failed_pages} misplaced={self.fault_misplaced} "
            f"landed_of_faulted={landed} vs asked_for_of_faulted={want} "
            f"read {self.fault_bytes / GIB:.2f} GiB in {self.fault_seconds:.2f}s "
            f"= {self.fault_gib_per_s:.2f} GiB/s"
        )

    def __str__(self) -> str:
        res = self.examined_pages or 1
        fin = sum(self.final_by_node.values())
        final = {k: f"{100 * v / max(1, fin):.2f}%" for k, v in sorted(self.final_by_node.items())}
        ok = 100.0 * self.final_on_target / max(1, self.final_on_target + self.final_off_target)
        return (
            f"planned={self.planned_pages} resident_at_census={self.resident_pages} "
            f"examined={self.examined_pages} "
            f"needed_move={self.needed_move} ({100 * self.needed_move / res:.2f}%) "
            f"moved={self.moved} failed={self.failed_pages} retried={self.retried_pages} "
            f"before_of_examined={ {k: f'{100 * v / res:.2f}%' for k, v in sorted(self.before_by_node.items())} } "
            f"after_of_examined={ {k: f'{100 * v / res:.2f}%' for k, v in sorted(self.after_by_node.items())} } "
            f"errno={dict(sorted(self.errno_pages.items()))} "
            f"FINAL_of_all_resident={final} on_target={ok:.2f}% over {fin} page(s) "
            f"in {self.seconds:.2f}s | "
            + self.fault_str()
        )


class _Libc:
    """``mmap``/``mincore``/``move_pages`` through ctypes, set up once."""

    _inst: _Libc | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        mach = platform.machine()
        if mach not in _SYS_MOVE_PAGES:
            raise PlacementError(
                f"no move_pages syscall number known for machine {mach!r}; refusing to guess "
                f"(a wrong number is a different syscall)"
            )
        self.sys_move_pages = _SYS_MOVE_PAGES[mach]
        self.sys_set_mempolicy = _SYS_SET_MEMPOLICY[mach]
        self.sys_get_mempolicy = _SYS_GET_MEMPOLICY[mach]
        self.sys_readahead = _SYS_READAHEAD[mach]
        self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_long]
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_ubyte)]
        self.libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    @classmethod
    def get(cls) -> _Libc:
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def map_readonly(self, path: str) -> tuple[int, int, int]:
        """``PROT_READ | MAP_SHARED`` over the whole file -> ``(addr, size, fd)``.

        Not ``MAP_PRIVATE``/``mmap.ACCESS_COPY``: a private writable mapping
        that is later mlocked copy-on-writes **every** page into an anonymous
        duplicate. The shared read-only mapping is also what makes these the
        same physical pages kt and streaming prefill see.

        The fd is handed back open, not closed: ``readahead(2)`` in the
        fault-in leg needs it, and reopening per batch would race a checkpoint
        being replaced under us.
        """
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            addr = self.libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        except BaseException:
            os.close(fd)
            raise
        if not addr or addr == ctypes.c_void_p(-1).value:
            os.close(fd)
            raise PlacementError(f"mmap({path}) failed: {os.strerror(ctypes.get_errno())}")
        return addr, size, fd

    def unmap(self, addr: int, size: int, fd: int | None = None) -> None:
        self.libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
        if fd is not None:
            os.close(fd)

    # -- allocation policy: the only thing that can place a page that does not
    # -- exist yet. move_pages cannot: it needs a page to already be there.

    def set_mempolicy(self, mode: int, node: int | None) -> None:
        """``set_mempolicy(2)`` for **this thread** (a task attribute, not a
        process one -- see ``_bound_policy`` for why that matters)."""
        if node is None:
            rc = self.libc.syscall(self.sys_set_mempolicy, MPOL_DEFAULT,
                                   ctypes.c_void_p(None), ctypes.c_ulong(0))
        else:
            if not 0 <= node < _NODEMASK_BITS:
                raise PlacementError(
                    f"node {node} does not fit a {_NODEMASK_BITS}-bit nodemask; widen the mask "
                    f"rather than letting the bit wrap onto a different node"
                )
            nmask = (ctypes.c_ulong * 1)(1 << node)
            rc = self.libc.syscall(self.sys_set_mempolicy, mode, nmask,
                                   ctypes.c_ulong(_NODEMASK_BITS))
        if rc != 0:
            raise PlacementError(
                f"set_mempolicy(mode={mode}, node={node}) failed: "
                f"{os.strerror(ctypes.get_errno())}"
            )

    def get_mempolicy(self) -> tuple[int, int]:
        """``(mode, nodemask)`` of this thread. Used to *prove* the reset ran."""
        mode = ctypes.c_int(0)
        nmask = (ctypes.c_ulong * 1)(0)
        rc = self.libc.syscall(self.sys_get_mempolicy, ctypes.byref(mode), nmask,
                               ctypes.c_ulong(_NODEMASK_BITS), ctypes.c_void_p(None),
                               ctypes.c_ulong(0))
        if rc != 0:
            raise PlacementError(f"get_mempolicy failed: {os.strerror(ctypes.get_errno())}")
        return mode.value, nmask[0]

    def populate(self, addr: int, first_page: int, npages: int) -> bool:
        """``MADV_POPULATE_READ`` over a page run: page tables in, I/O waited on.

        Returns ``False`` (and does nothing) on a kernel without it, so the
        caller falls back to the per-page touch instead of skipping the
        population and getting ``-ENOENT`` from every ``move_pages``.
        """
        rc = self.libc.madvise(ctypes.c_void_p(addr + first_page * PAGE_SIZE),
                               ctypes.c_size_t(npages * PAGE_SIZE), MADV_POPULATE_READ)
        if rc == 0:
            return True
        err = ctypes.get_errno()
        if err == errno.EINVAL:  # pre-5.14 kernel: no such advice
            return False
        raise PlacementError(
            f"madvise(MADV_POPULATE_READ, {npages} pages) failed: {os.strerror(err)}")

    def readahead(self, fd: int, offset: int, count: int) -> None:
        """``readahead(2)``. The page cache pages it allocates come from the
        *calling thread's* mempolicy, which is what makes the bind work."""
        rc = self.libc.syscall(self.sys_readahead, fd, ctypes.c_long(offset),
                               ctypes.c_size_t(count))
        if rc != 0:
            raise PlacementError(
                f"readahead(fd={fd}, off={offset}, n={count}) failed: "
                f"{os.strerror(ctypes.get_errno())}"
            )

    def resident(self, addr: int, first_page: int, npages: int) -> list[int]:
        """Page indices (absolute, from ``first_page``) that are in the page cache."""
        vec = (ctypes.c_ubyte * npages)()
        rc = self.libc.mincore(ctypes.c_void_p(addr + first_page * PAGE_SIZE),
                               ctypes.c_size_t(npages * PAGE_SIZE), vec)
        if rc != 0:
            raise PlacementError(f"mincore failed: {os.strerror(ctypes.get_errno())}")
        return [first_page + i for i in range(npages) if vec[i] & 1]

    def move(self, addr: int, pages: Sequence[int], node: int | None,
             flags: int) -> list[int]:
        """``move_pages`` over absolute page indices; returns ``status[]``.

        ``node=None`` is the query form (``nodes`` NULL), which reports where
        each page is without moving anything.
        """
        n = len(pages)
        arr = (ctypes.c_void_p * n)(*[addr + p * PAGE_SIZE for p in pages])
        status = (ctypes.c_int * n)()
        nodes = None if node is None else (ctypes.c_int * n)(*([node] * n))
        rc = self.libc.syscall(self.sys_move_pages, 0, ctypes.c_ulong(n), arr,
                               ctypes.cast(nodes, ctypes.c_void_p) if nodes is not None
                               else ctypes.c_void_p(None),
                               status, flags)
        if rc < 0:
            # Only the whole-call failure. The per-page verdict is status[]:
            # rc==0 has been seen alongside "not one page moved".
            raise PlacementError(
                f"move_pages(n={n}, node={node}, flags={flags}) failed: "
                f"{os.strerror(ctypes.get_errno())}"
            )
        return list(status)


def _errno_name(status: int) -> str:
    """``status[i] < 0`` -> a name a human can act on (``-16`` -> ``EBUSY(...)``)."""
    try:
        name = errno.errorcode.get(-status, f"errno{-status}")
        return f"{name}({os.strerror(-status)})"
    except (ValueError, OverflowError):  # pragma: no cover - defensive
        return f"errno{-status}"


def _touch(buf, pages: Iterable[int]) -> None:
    """Fault the page-table entries in, one byte per page. Cannot be skipped.

    Called with **resident** pages by the migrate leg (an untouched mapping
    gives ``move_pages`` ``-ENOENT`` for every page) and with **non-resident**
    pages by the fault-in leg, where the major fault is the point.
    """
    for p in pages:
        _ = buf[p * PAGE_SIZE]


@dataclass
class _Batch:
    """One unit of work: a page interval of one file and the node it belongs on."""

    path: str
    addr: int
    size: int
    fd: int
    buf: object
    lo: int
    hi: int
    node: int
    resident: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    #: what the migrate leg will examine. Set by the census and narrowed by the
    #: fault-in leg -- see MIGRATE_SCOPES.
    migrate: list[int] = field(default_factory=list)


#: What the migrate leg examines.
#:
#: ``"suspect"`` (default): after the fault-in leg, the only pages still in
#: question are the ones it reports **misplaced** (it measured where each page
#: it read landed) and the ones **already resident at the census**, which it
#: never touched. The rest was read under ``MPOL_BIND`` and verified on target
#: one step earlier; re-querying it is a large share of start-up time spent
#: re-asking a question we wrote down.
#: ``"all"`` is the old behaviour, kept as the arm the narrowed one is compared
#: against.
#: ``"resident"`` is the **negative control** -- it drops the misplaced set on
#: purpose, so they are never corrected and the final distribution must visibly
#: degrade. Without it, "faster" and "did not do the work" look the same.
MIGRATE_SCOPES = ("suspect", "all", "resident")


def _runs_of(pages: Sequence[int]) -> list[tuple[int, int]]:
    """Ascending page indices -> ``(first, count)`` contiguous runs, for readahead."""
    out: list[tuple[int, int]] = []
    for p in pages:
        if out and out[-1][0] + out[-1][1] == p:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((p, 1))
    return out


class _bound_policy:
    """``set_mempolicy(mode, {node})`` for the calling thread, reset on exit.

    The reset is in a ``finally`` and it is not optional. A memory policy is a
    **task** attribute: leave ``MPOL_BIND`` set on a pool thread and every later
    allocation on it -- ours, torch's, anything -- is pinned to that node until
    the thread dies, which on a node with 4 GiB free is an OOM with no obvious
    cause and no trace back to here. A policy is inherited only by threads
    *created* while it is live, so an existing pool picks up nothing but a leak
    is not confined to the caller: ``apply_placement`` builds the migrate pool
    after the fault-in leg has run.

    ``__exit__`` therefore does not call the reset and hope: it retries once,
    asks ``get_mempolicy`` what the thread actually is, logs CRITICAL if it is
    still bound, and never replaces the body's exception with its own.
    """

    def __init__(self, libc: _Libc, mode: int, node: int) -> None:
        self.libc, self.mode, self.node = libc, mode, node

    def __enter__(self) -> _bound_policy:
        self.libc.set_mempolicy(self.mode, self.node)
        return self

    def _reset(self) -> PlacementError | None:
        try:
            self.libc.set_mempolicy(MPOL_DEFAULT, None)
        except PlacementError as exc:
            return exc
        return None

    def __exit__(self, *exc) -> None:
        # The reset has to survive its own failure. A bare
        # ``set_mempolicy(MPOL_DEFAULT)`` fails two ways at once if the syscall
        # returns non-zero: the thread stays MPOL_BIND for the rest of its life
        # -- and so does every thread it *creates* afterwards, since a memory
        # policy is inherited at clone time -- and the PlacementError it raises
        # REPLACES whatever the body was failing with. A hard-bound thread whose
        # node is full gets a failed allocation, not a fallback;
        # ``std::aligned_alloc`` returning nullptr then passes an
        # ``assert(ptr % 64 == 0)`` (0 % 64 == 0) and the next write is a
        # SIGSEGV with nothing in the log.
        err = self._reset()
        if err is not None:
            err = self._reset()  # one retry: a transient failure is cheap to survive
        if err is None:
            return
        # "we called reset" and "this thread is unbound" are different claims,
        # and only get_mempolicy separates them.
        try:
            mode, mask = self.libc.get_mempolicy()
        except PlacementError:
            mode, mask = -1, -1
        if mode == MPOL_DEFAULT:
            logger.warning(
                "kt numa_placement: set_mempolicy(MPOL_DEFAULT) reported a failure on thread "
                "%s (%s) but get_mempolicy says the policy is already default; continuing",
                threading.current_thread().name, err)
            return
        logger.critical(
            "kt numa_placement: could NOT reset the memory policy of thread %s (%s); "
            "get_mempolicy=(mode=%d, mask=%#x). That thread is left bound to node%d: every "
            "later allocation on it, and on every thread it creates from now on, is pinned "
            "to that node and will fail rather than fall back when the node is full.",
            threading.current_thread().name, err, mode, mask, self.node)
        # Never mask the body's exception with ours: the log above is the
        # report and the original error wins.
        if exc[0] is None:
            raise err


def _populate(libc: _Libc, addr: int, buf, pages: Sequence[int]) -> None:
    """Get ``pages`` into this process's page table, by run, without the GIL.

    ``move_pages`` only knows a page that is in the page table, so this cannot
    be skipped -- but not one page at a time from Python: contiguous pages
    become one ``madvise`` each, and the per-page read is kept only for a kernel
    too old to have the advice.
    """
    for first, count in _runs_of(pages):
        if not libc.populate(addr, first, count):
            _touch(buf, range(first, first + count))


def apply_placement(
    plan: PlacementPlan,
    *,
    threads: int = 8,
    batch_pages: int = 65536,
    flags: int = MPOL_MF_MOVE_ALL,
    retries: int = 1,
    fault_in: bool = True,
    fault_policy: int = MPOL_BIND,
    readahead_pages: int = 512,
    migrate_scope: str = "suspect",
    dry_run: bool = False,
) -> MigrationReport:
    """Put every planned page on its target node, by whichever of the two
    mechanisms applies to it.

    A cached page is migrated by ``move_pages(MPOL_MF_MOVE_ALL)``; one that does
    not exist yet is read here under ``set_mempolicy(MPOL_BIND, {target})``.
    That second leg **reads from disk**: ``fault_seconds`` /
    ``fault_gib_per_s`` are the cost and set start-up time, and ``dry_run``
    skips it, reporting ``fault_planned_pages`` so the size is known first.

    ``MPOL_BIND`` is a hard constraint: a full target node means reclaim (or
    failure), not a spill to the other node. ``MPOL_PREFERRED`` degrades to
    worse locality instead and is the right choice when the target is tight.
    Neither is guessed for you.
    """
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    if batch_pages < 1:
        raise ValueError(f"batch_pages must be >= 1, got {batch_pages}")
    if retries < 0:
        raise ValueError(f"retries must be >= 0, got {retries}")
    if readahead_pages < 1:
        raise ValueError(f"readahead_pages must be >= 1, got {readahead_pages}")
    if flags not in (MPOL_MF_MOVE, MPOL_MF_MOVE_ALL):
        raise ValueError(f"flags must be MPOL_MF_MOVE or MPOL_MF_MOVE_ALL, got {flags}")
    if migrate_scope not in MIGRATE_SCOPES:
        raise ValueError(f"migrate_scope must be one of {MIGRATE_SCOPES}, got {migrate_scope!r}")
    if fault_policy not in (MPOL_BIND, MPOL_PREFERRED):
        raise ValueError(
            f"fault_policy must be MPOL_BIND ({MPOL_BIND}) or MPOL_PREFERRED "
            f"({MPOL_PREFERRED}), got {fault_policy}"
        )
    if flags == MPOL_MF_MOVE:
        # Without MOVE_ALL, file pages kt also maps come back -EBUSY in bulk.
        # Allowed, but never silently.
        logger.warning("kt numa_placement: MPOL_MF_MOVE without MOVE_ALL leaves a large -EBUSY "
                       "tail on pages other processes also map")

    libc = _Libc.get()
    total = MigrationReport()
    lock = threading.Lock()

    def run(fn, items):
        if threads == 1 or len(items) <= 1:
            return [fn(i) for i in items]
        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="kt_numa") as pool:
            return list(pool.map(fn, items))

    batches: list[_Batch] = []
    opened: list[tuple[int, int, int]] = []
    try:
        for path, runs in plan.page_runs.items():
            addr, size, fd = libc.map_readonly(path)
            opened.append((addr, size, fd))
            buf = (ctypes.c_char * size).from_address(addr)
            for lo, hi, node in runs:
                for b in range(lo, hi, batch_pages):
                    batches.append(_Batch(path, addr, size, fd, buf, b,
                                          min(b + batch_pages, hi), node))

        # -- census: who is in the page cache and who is not ------------------
        def census(b: _Batch) -> None:
            b.resident = libc.resident(b.addr, b.lo, b.hi - b.lo)
            have = set(b.resident)
            b.missing = [p for p in range(b.lo, b.hi) if p not in have]
            # The fault-in leg will not look at pages already in the page
            # cache, so they are in every scope.
            b.migrate = list(b.resident)

        run(census, batches)
        for b in batches:
            total.planned_pages += b.hi - b.lo
            total.resident_pages += len(b.resident)
            total.fault_planned_pages += len(b.missing)
            # At the census, not in the fault leg: a dry run must answer "what
            # was the fault-in leg going to be asked for" without reading.
            if b.missing:
                total.fault_target_by_node[b.node] = (
                    total.fault_target_by_node.get(b.node, 0) + len(b.missing))

        # The size of the read about to be done, per node, against what that
        # node has free right now: MPOL_BIND on a node that cannot take it
        # means reclaim, not a fallback.
        want: Counter[int] = Counter()
        for b in batches:
            want[b.node] += len(b.missing)
        if want:
            try:
                mem = read_node_memory(sorted(want))
                for n, pages in sorted(want.items()):
                    logger.info("kt numa_placement: fault-in will read %.2f GiB onto node%d "
                                "(MemFree %.1f GiB)", pages * PAGE_SIZE / GIB, n,
                                mem[n].mem_free / GIB)
                    if fault_policy == MPOL_BIND and pages * PAGE_SIZE > mem[n].mem_free:
                        logger.warning(
                            "kt numa_placement: MPOL_BIND wants %.2f GiB on node%d which has "
                            "%.1f GiB free -- the kernel will reclaim to make room; "
                            "MPOL_PREFERRED degrades to worse locality instead",
                            pages * PAGE_SIZE / GIB, n, mem[n].mem_free / GIB)
            except PlacementError as exc:  # sysfs unreadable: say so, do not die
                logger.warning("kt numa_placement: could not read node meminfo (%s)", exc)

        # -- leg 2: pages that do not exist yet -> allocation policy ----------
        if fault_in and not dry_run:
            t_fault = time.perf_counter()

            def fault(b: _Batch) -> MigrationReport:
                rep = MigrationReport()
                if not b.missing:
                    return rep
                with _bound_policy(libc, fault_policy, b.node):
                    for first, count in _runs_of(b.missing):
                        # readahead() queues the I/O for the whole run first,
                        # so the device sees a deep queue; the pages it
                        # allocates come from THIS thread's policy, which is
                        # the placement.
                        for off in range(0, count, readahead_pages):
                            n = min(readahead_pages, count - off)
                            libc.readahead(b.fd, (first + off) * PAGE_SIZE, n * PAGE_SIZE)
                        # Then wait for the run in one syscall: a per-page
                        # touch holds the GIL across every major fault, which
                        # turns 8 threads into one outstanding read.
                        if not libc.populate(b.addr, first, count):
                            _touch(b.buf, range(first, first + count))
                # Do not take "we read it" for success. Ask mincore again.
                now = set(libc.resident(b.addr, b.lo, b.hi - b.lo))
                landed = [p for p in b.missing if p in now]
                rep.faulted_in = len(landed)
                rep.fault_failed_pages = len(b.missing) - len(landed)
                if landed:
                    status = libc.move(b.addr, landed, None, 0)
                    misplaced = []
                    for p, st in zip(landed, status):
                        if st >= 0:
                            rep.fault_by_node[st] = rep.fault_by_node.get(st, 0) + 1
                        if st != b.node:
                            rep.fault_misplaced += 1
                            misplaced.append(p)
                    # The narrowing: a page the bind placed on its target was
                    # just measured there, so handing it to the migrate leg
                    # re-asks a question we wrote the answer to.
                    if migrate_scope == "all":
                        b.migrate = sorted(set(b.migrate) | set(landed))
                    elif migrate_scope == "suspect":
                        b.migrate = sorted(set(b.migrate) | set(misplaced))
                    # "resident": leave b.migrate as the census set
                    # Every landed page the scope did NOT hand to migrate keeps
                    # the node just measured, so the final tally stays complete
                    # however much the migrate leg was narrowed.
                    handed = set(b.migrate)
                    for p, st in zip(landed, status):
                        if p not in handed and st >= 0:
                            rep.final_by_node[st] = rep.final_by_node.get(st, 0) + 1
                            if st == b.node:
                                rep.final_on_target += 1
                            else:
                                rep.final_off_target += 1
                return rep

            for r in run(fault, batches):
                with lock:
                    total.merge(r)
            total.fault_seconds = time.perf_counter() - t_fault
            logger.info("kt numa_placement: %s", total.fault_str())

        # -- leg 1: pages already in the page cache -> move_pages -------------
        t_move = time.perf_counter()

        def migrate(b: _Batch) -> MigrationReport:
            rep = MigrationReport()
            pages = b.migrate
            if not pages:
                return rep
            rep.examined_pages = len(pages)
            # Resident pages only, so this waits on nothing and reads nothing;
            # it just puts them in the page table for move_pages.
            _populate(libc, b.addr, b.buf, pages)
            before = libc.move(b.addr, pages, None, 0)
            for st in before:
                if st >= 0:
                    rep.before_by_node[st] = rep.before_by_node.get(st, 0) + 1
                if st != b.node:
                    rep.needed_move += 1
            if dry_run:
                rep.after_by_node = dict(rep.before_by_node)
                rep.final_by_node = dict(rep.before_by_node)
                return rep
            status = libc.move(b.addr, pages, b.node, flags)
            # -EBUSY is the one status worth a second try: it means the page
            # was pinned, dirty or under writeback at that instant, not that it
            # cannot move.
            for _ in range(retries):
                busy = [p for p, st in zip(pages, status) if st == -errno.EBUSY]
                if not busy:
                    break
                rep.retried_pages += len(busy)
                again = dict(zip(busy, libc.move(b.addr, busy, b.node, flags)))
                status = [again.get(p, st) for p, st in zip(pages, status)]
            # The per-page verdict: the syscall's own return value has
            # reported success over a run in which nothing moved.
            for st in status:
                if st == b.node:
                    rep.moved += 1
                    rep.after_by_node[st] = rep.after_by_node.get(st, 0) + 1
                    rep.final_by_node[st] = rep.final_by_node.get(st, 0) + 1
                    rep.final_on_target += 1
                else:
                    rep.failed_pages += 1
                    if st >= 0:
                        rep.after_by_node[st] = rep.after_by_node.get(st, 0) + 1
                        rep.final_by_node[st] = rep.final_by_node.get(st, 0) + 1
                        rep.final_off_target += 1
                        key = f"landed on node{st}"
                    else:
                        key = _errno_name(st)
                    rep.errno_pages[key] = rep.errno_pages.get(key, 0) + 1
            return rep

        for r in run(migrate, batches):
            with lock:
                # ``resident_pages`` was counted at the census; the per-batch
                # reports here must not add it a second time.
                total.merge(r)
        total.seconds = total.fault_seconds + (time.perf_counter() - t_move)
    finally:
        for addr, size, fd in opened:
            libc.unmap(addr, size, fd)

    covered = total.resident_pages + (total.fault_planned_pages if dry_run else total.faulted_in)
    if covered == 0:
        raise ZeroPlacementError(
            f"none of the {total.planned_pages} planned page(s) were placed: "
            f"resident={total.resident_pages} faulted_in={total.faulted_in}; a run that "
            f"touched nothing must not report success"
        )
    # What actually landed, counted per page, plus the failures -- a claim
    # distinct from the fraction logged before the run.
    logger.info("kt numa_placement: %s", total)
    scored = total.final_on_target + total.final_off_target
    if scored and total.final_on_target < 0.99 * scored:
        # The per-leg counters can all read "nothing to do, nothing went
        # wrong" on a run that is well off target overall.
        logger.warning(
            "kt numa_placement: only %.2f%% of the %d placed page(s) are on their planned node "
            "(%d off). The per-leg counters can all look clean while this is true; re-check with "
            "audit_placement() and compare against the fault-in leg's landed-vs-asked_for.",
            100.0 * total.final_on_target / scored, scored, total.final_off_target)
    return total



@dataclass
class PlacementAudit:
    """Where the planned pages actually are, asked from scratch.

    ``MigrationReport.after_by_node`` covers only the pages the migrate leg
    **examined**, and ``suspect`` examines far fewer than ``all``, so comparing
    two such reports says the narrowed arm placed fewer pages when it placed the
    same ones. "The final distribution must not degrade" has to be measured over
    *all* planned pages by something that did not do the placing.
    """

    resident_pages: int = 0
    on_target: int = 0
    off_target: int = 0
    by_node: dict[int, int] = field(default_factory=dict)
    target_by_node: dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def on_target_pct(self) -> float:
        return 100.0 * self.on_target / self.resident_pages if self.resident_pages else 0.0

    def merge(self, other: PlacementAudit) -> None:
        self.resident_pages += other.resident_pages
        self.on_target += other.on_target
        self.off_target += other.off_target
        for dst, src in ((self.by_node, other.by_node),
                         (self.target_by_node, other.target_by_node)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v

    def __str__(self) -> str:
        n = self.resident_pages or 1
        return (
            f"audit: resident={self.resident_pages} on_target={self.on_target} "
            f"({self.on_target_pct:.2f}%) off_target={self.off_target} "
            f"actual_of_resident={ {k: f'{100 * v / n:.2f}%' for k, v in sorted(self.by_node.items())} } "
            f"planned_of_resident={ {k: f'{100 * v / n:.2f}%' for k, v in sorted(self.target_by_node.items())} } "
            f"in {self.seconds:.2f}s"
        )


def audit_placement(plan: PlacementPlan, *, threads: int = 8,
                    batch_pages: int = 65536) -> PlacementAudit:
    """Read back where every resident planned page is. Moves nothing.

    Deliberately shares no state with ``apply_placement``: it re-mincores and
    re-queries, so an arm that narrowed what it examined cannot narrow what it
    is judged on.
    """
    libc = _Libc.get()
    total = PlacementAudit()
    lock = threading.Lock()
    t0 = time.perf_counter()
    for path, runs in plan.page_runs.items():
        addr, size, fd = libc.map_readonly(path)
        try:
            buf = (ctypes.c_char * size).from_address(addr)
            batches = [(b, min(b + batch_pages, hi), node)
                       for lo, hi, node in runs for b in range(lo, hi, batch_pages)]

            def one(b, addr=addr, buf=buf) -> PlacementAudit:
                lo, hi, node = b
                a = PlacementAudit()
                pages = libc.resident(addr, lo, hi - lo)
                if not pages:
                    return a
                a.resident_pages = len(pages)
                a.target_by_node[node] = len(pages)
                _populate(libc, addr, buf, pages)
                for st in libc.move(addr, pages, None, 0):
                    if st >= 0:
                        a.by_node[st] = a.by_node.get(st, 0) + 1
                    if st == node:
                        a.on_target += 1
                    else:
                        a.off_target += 1
                return a

            if threads == 1 or len(batches) <= 1:
                reports = [one(b) for b in batches]
            else:
                with ThreadPoolExecutor(max_workers=threads,
                                        thread_name_prefix="kt_audit") as pool:
                    reports = list(pool.map(one, batches))
            with lock:
                for r in reports:
                    total.merge(r)
        finally:
            libc.unmap(addr, size, fd)
    total.seconds = time.perf_counter() - t0
    logger.info("kt numa_placement: %s", total)
    return total


def place_weights(
    source,
    layers: Sequence[int],
    experts: Sequence[int],
    *,
    tp: int,
    part_nodes: Sequence[int] | None = None,
    streaming_node: int = 0,
    fraction: Sequence[float] | None = None,
    fraction_source: str | None = None,
    reserve_gib: float = 16.0,
    threads: int = 8,
    batch_pages: int = 65536,
    retries: int = 1,
    fault_in: bool = True,
    fault_policy: int = MPOL_BIND,
    readahead_pages: int = 512,
    migrate_scope: str = "suspect",
    dry_run: bool = False,
    require_mutation: bool = False,
) -> tuple[PlacementPlan, MigrationReport]:
    """Plan + migrate in one call. The entry point wiring should use."""
    plan = plan_placement(source, layers, experts, tp=tp, part_nodes=part_nodes,
                          streaming_node=streaming_node, fraction=fraction,
                          fraction_source=fraction_source, reserve_gib=reserve_gib)
    report = apply_placement(plan, threads=threads, batch_pages=batch_pages,
                             retries=retries, fault_in=fault_in,
                             fault_policy=fault_policy, readahead_pages=readahead_pages,
                             migrate_scope=migrate_scope, dry_run=dry_run)
    if require_mutation:
        report.require_applied()
    return plan, report


# ---------------------------------------------------------------------------
# CLI -- one layer at a time, on the real checkpoint
# ---------------------------------------------------------------------------


def _parse_ints(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt_dir")
    ap.add_argument("--layers", default="0")
    ap.add_argument("--experts", default="0-383")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--part-nodes", default=None, help="default: 0,1,... per TP part")
    ap.add_argument("--streaming-node", type=int, default=0)
    ap.add_argument("--fraction", default=None, help="pinned split, e.g. 0.41,0.59")
    ap.add_argument("--mirror", action="store_true", help="positive control: reverse the fraction")
    ap.add_argument("--reserve-gib", type=float, default=16.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-pages", type=int, default=65536)
    ap.add_argument("--retries", type=int, default=1, help="extra passes over -EBUSY pages")
    ap.add_argument("--no-fault-in", action="store_true",
                    help="skip the second leg: leave non-resident pages to chance")
    ap.add_argument("--fault-policy", choices=("bind", "preferred"), default="bind",
                    help="bind = exact but hard (reclaims a full node); preferred = graceful")
    ap.add_argument("--readahead-pages", type=int, default=512)
    ap.add_argument("--migrate-scope", choices=MIGRATE_SCOPES, default="suspect",
                    help="suspect = only what fault-in flagged plus what it never saw; "
                         "all = re-examine everything (the old, slow arm); "
                         "resident = negative control, never corrects the misplaced")
    ap.add_argument("--explain", action="store_true",
                    help="print every percentage with the population it is over, and stop")
    ap.add_argument("--audit", action="store_true",
                    help="after placing, re-read where EVERY planned page is (the only "
                         "cross-scope-comparable number)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from vllm_ascend.kt_offload.weight_source import WeightSource

    frac = [float(x) for x in args.fraction.split(",")] if args.fraction else None
    nodes = _parse_ints(args.part_nodes) if args.part_nodes else None
    src = WeightSource(args.ckpt_dir)
    if args.mirror:
        if frac is None:
            # Same need as plan_placement's own derivation, computed the same way.
            w13 = sum(src.locate(layer, expert, proj, kind).nbytes
                      for layer in _parse_ints(args.layers)
                      for expert in _parse_ints(args.experts)
                      for proj, kind in SPLIT_TENSORS)
            frac = derive_fraction(args.reserve_gib, args.tp, nodes=nodes,
                                   need_bytes_per_part=w13 // args.tp)
        frac = mirror_fraction(frac)
    if args.explain:
        plan = plan_placement(src, _parse_ints(args.layers), _parse_ints(args.experts),
                              tp=args.tp, part_nodes=nodes,
                              streaming_node=args.streaming_node, fraction=frac,
                              reserve_gib=args.reserve_gib)
        print(plan.explain())
        return 0
    plan, report = place_weights(
        src, _parse_ints(args.layers), _parse_ints(args.experts), tp=args.tp,
        part_nodes=nodes, streaming_node=args.streaming_node, fraction=frac,
        reserve_gib=args.reserve_gib, threads=args.threads,
        batch_pages=args.batch_pages, retries=args.retries,
        fault_in=not args.no_fault_in,
        fault_policy=MPOL_BIND if args.fault_policy == "bind" else MPOL_PREFERRED,
        readahead_pages=args.readahead_pages, migrate_scope=args.migrate_scope,
        dry_run=args.dry_run,
    )
    print(f"plan: {plan.planned_pages} pages, by node {plan.planned_pages_by_node()}, "
          f"fraction={[round(x, 4) for x in plan.fraction]} ({plan.fraction_source})")
    print(f"apply: {report}")
    print(f"       {report.fault_str()}")
    if args.audit:
        print(f"audit: {audit_placement(plan, threads=args.threads)}")
    # needed_move / faulted_in are assertions, not statistics: say so in the
    # exit code, so a driver script cannot mistake a no-op for an arm.
    if not args.dry_run:
        try:
            report.require_applied()
        except MutationNotAppliedError as exc:
            print(f"NOT A VALID ARM: {exc}")
            return 2
    return 0 if report.failed_pages == 0 and report.fault_failed_pages == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
