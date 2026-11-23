# SPDX-License-Identifier: Apache-2.0
"""``numa_placement``: the plan, the derivation, and the assertion that guards them.

The plan and ``derive_fraction`` are judged on injected meminfo readings and on
byte intervals computed from this file's own constants. The ``needed_move > 0``
and ``faulted_in > 0`` assertions are tested from the red side: without those
arms an implementation that does nothing passes everything else here.
"""

from __future__ import annotations

import os
import threading

import pytest

from vllm_ascend.kt_offload import numa_placement as np_
from vllm_ascend.kt_offload.numa_placement import (
    MigrationReport,
    MutationNotAppliedError,
    NodeMemory,
    PlacementError,
    PlacementPlan,
    ZeroPlacementError,
    derive_fraction,
    mirror_fraction,
    plan_placement,
    validate_fraction,
)

GIB = 1 << 30

# One fake shard, with the real per-expert byte counts of the checkpoint, so the
# TP division and the page rounding see realistic numbers.
W13_BYTES = 11796480 // 2  # w1 and w3 are separate tensors; w13 is their sum
W2_BYTES = 5898240
SCALE_BYTES = 1105920 // 3


class FakeSpan:
    def __init__(self, path, start, nbytes):
        self.path, self.start, self.nbytes = path, start, nbytes


class FakeSource:
    """Just enough of ``WeightSource``: ``locate`` -> a span with a byte interval.

    Experts are laid out back to back in one file, in the checkpoint's order
    (w1, w3, w2 weights then the scales), starting at a non-page-aligned offset
    because a real safetensors data section does too.
    """

    ORDER = (
        ("w1", "weight", W13_BYTES),
        ("w3", "weight", W13_BYTES),
        ("w2", "weight", W2_BYTES),
        ("w1", "scale", SCALE_BYTES),
        ("w3", "scale", SCALE_BYTES),
        ("w2", "scale", SCALE_BYTES),
    )
    PER_EXPERT = sum(n for _, _, n in ORDER)

    def __init__(self, path="/fake/shard-0.safetensors", data_start=1234, num_layers=2, num_experts=4):
        self.path, self.data_start = path, data_start
        self.num_layers, self.num_experts = num_layers, num_experts

    def locate(self, layer, expert, proj, kind="weight"):
        if not (0 <= layer < self.num_layers and 0 <= expert < self.num_experts):
            raise KeyError((layer, expert))
        base = self.data_start + (layer * self.num_experts + expert) * self.PER_EXPERT
        for p, k, n in self.ORDER:
            if (p, k) == (proj, kind):
                return FakeSpan(self.path, base, n)
            base += n
        raise KeyError((proj, kind))


def _mem(free_gib, file_gib, shmem_gib, node=0):
    return NodeMemory(node, int(free_gib * GIB), int(file_gib * GIB), int(shmem_gib * GIB))


def test_w13_is_split_into_contiguous_per_part_intervals():
    """TP slices ``inter``, which is w1/w3's ROW dimension in a row-major
    tensor, so part p owns exactly ``[p*nbytes/tp, (p+1)*nbytes/tp)``."""
    src = FakeSource(num_layers=1, num_experts=1)
    plan = plan_placement(src, [0], [0], tp=2, fraction=[0.5, 0.5], streaming_node=0)
    span = src.locate(0, 0, "w1", "weight")
    w1 = sorted((s for s in plan.segments if (s.proj, s.kind) == ("w1", "weight")), key=lambda s: s.start)
    # Two parts x (local prefix + overflow) with fraction 0.5 each.
    assert [s.part for s in w1] == [0, 0, 1, 1]
    # Contiguous and gapless over exactly the tensor.
    assert w1[0].start == span.start
    assert w1[-1].end == span.start + span.nbytes
    for a, b in zip(w1, w1[1:]):
        assert a.end == b.start
    # Part 0's own interval is the first half, part 1's the second.
    half = span.nbytes // 2
    assert [s.start for s in w1 if s.part == 0][0] == span.start
    assert [s.start for s in w1 if s.part == 1][0] == span.start + half


def test_fraction_is_a_contiguous_prefix_on_the_parts_own_node():
    """0.25 means part 0 keeps its first quarter on node 0 and spills the rest
    -- a prefix, never a per-page stripe, which any per-page sampling would
    resonate with."""
    src = FakeSource(num_layers=1, num_experts=1)
    plan = plan_placement(src, [0], [0], tp=2, fraction=[0.25, 0.75], streaming_node=0)
    span = src.locate(0, 0, "w3", "weight")
    part0 = sorted(
        (s for s in plan.segments if (s.proj, s.kind) == ("w3", "weight") and s.part == 0), key=lambda s: s.start
    )
    half = span.nbytes // 2
    assert part0[0].node == 0 and part0[0].reason == "w13-local"
    assert part0[0].start == span.start
    assert part0[0].nbytes == round(0.25 * half)
    assert part0[1].node == 1 and part0[1].reason == "w13-overflow"
    assert part0[1].start == part0[0].end
    assert part0[0].nbytes + part0[1].nbytes == half


def test_w2_and_every_scale_go_to_the_streaming_node_whole():
    """w2 is ``[hidden, inter]`` and TP slices COLUMNS, so its per-part bytes
    are not contiguous and no per-page placement can beat 1/tp local. It keeps
    kt's copy path; only streaming prefill reads its file pages -- so it is not
    split here, by design, not by oversight."""
    src = FakeSource(num_layers=1, num_experts=2)
    plan = plan_placement(src, [0], [0, 1], tp=2, fraction=[0.5, 0.5], streaming_node=1)
    whole = [s for s in plan.segments if (s.proj, s.kind) != ("w1", "weight") and (s.proj, s.kind) != ("w3", "weight")]
    assert {(s.proj, s.kind) for s in whole} == {("w2", "weight"), ("w1", "scale"), ("w3", "scale"), ("w2", "scale")}
    for s in whole:
        assert s.node == 1, s
        assert s.part is None
        assert s.reason == "streaming-only"
        span = src.locate(s.layer, s.expert, s.proj, s.kind)
        assert (s.start, s.nbytes) == (span.start, span.nbytes)


def test_page_ownership_is_exclusive_and_ascending():
    """A page may have only one owner: two owners means two threads migrating
    the same page in opposite directions and a count that adds up to more than
    100%."""
    src = FakeSource()
    plan = plan_placement(src, [0, 1], [0, 1, 2, 3], tp=2, fraction=[0.3, 0.7])
    runs = plan.page_runs[src.path]
    assert runs
    prev_hi = -1
    for lo, hi, node in runs:
        assert lo >= prev_hi, runs
        assert hi > lo
        prev_hi = hi
    assert plan.planned_pages == sum(hi - lo for lo, hi, _ in runs)
    assert sum(plan.planned_pages_by_node().values()) == plan.planned_pages


def test_split_refuses_a_tensor_that_does_not_divide_by_tp():
    src = FakeSource(num_layers=1, num_experts=1)
    with pytest.raises(PlacementError, match="does not divide"):
        plan_placement(src, [0], [0], tp=7, fraction=[1 / 7] * 7)


def test_zero_coverage_plan_raises():
    """An empty request must not be a quiet success: "nothing planned" and
    "everything planned correctly" look identical in every ratio."""
    src = FakeSource()
    with pytest.raises(ZeroPlacementError):
        plan_placement(src, [], [], tp=2, fraction=[0.5, 0.5])


def test_mirror_fraction_is_a_different_split():
    """The positive control's only job: it must not equal the arm it controls."""
    f = [0.41, 0.59]
    assert mirror_fraction(f) == [0.59, 0.41]
    src = FakeSource(num_layers=1, num_experts=2)
    a = plan_placement(src, [0], [0, 1], tp=2, fraction=f).planned_bytes_by_node()
    b = plan_placement(src, [0], [0, 1], tp=2, fraction=mirror_fraction(f)).planned_bytes_by_node()
    assert a != b, "the mirrored arm plans the same bytes on the same nodes; it controls nothing"


# 契约：fraction[p] = min(1, node p 能给的 / part p 需要的)。各项独立，不归一
# 化 —— 要求和为 1 会把平均局部性钉死在 1/tp，与内存读数无关。

NEED = int(100 * GIB)  # 每个 part 要放 100 GiB


def test_derive_fraction_is_full_locality_when_both_nodes_have_room():
    """默认是全局部性，不是按可用量摊派。"""
    mem = {0: _mem(200.0, 0.0, 0.0, 0), 1: _mem(400.0, 0.0, 0.0, 1)}
    f = derive_fraction(0.0, 2, memory=mem, need_bytes_per_part=NEED)
    assert f == [1.0, 1.0]
    assert sum(f) == 2.0, "各项独立 ⇒ 完美局部性的和是 tp，不是 1"


def test_derive_fraction_is_not_normalised():
    """同一台机器上 node1 的余量是 node0 的两倍，也不该把 node0 压到 1/3：
    它自己的 node 放得下，就不该跨 socket 读。"""
    mem = {0: _mem(150.0, 0.0, 0.0, 0), 1: _mem(300.0, 0.0, 0.0, 1)}
    f = derive_fraction(0.0, 2, memory=mem, need_bytes_per_part=NEED)
    assert f == [1.0, 1.0]
    assert f[0] != pytest.approx(150.0 / 450.0)


def test_derive_fraction_cuts_back_only_the_node_that_is_short():
    """node0 只放得下一半，那就只压 node0；node1 不受牵连。"""
    mem = {0: _mem(50.0, 0.0, 0.0, 0), 1: _mem(400.0, 0.0, 0.0, 1)}
    f = derive_fraction(0.0, 2, memory=mem, need_bytes_per_part=NEED)
    assert f[0] == pytest.approx(0.5, abs=1e-6)
    assert f[1] == 1.0


def test_derive_fraction_reads_free_plus_reclaimable_file_cache():
    # node0: free 52.8 + (FilePages 222.8 - Shmem 159.9) = 115.7 GiB
    # node1: free 247.2 + (FilePages 73.1 - Shmem 46.2)  = 274.1 GiB
    mem = {0: _mem(52.8, 222.8, 159.9, 0), 1: _mem(247.2, 73.1, 46.2, 1)}
    f = derive_fraction(0.0, 2, memory=mem, need_bytes_per_part=NEED)
    assert f == [1.0, 1.0]  # 两边都放得下 100 GiB
    # 把需求抬到 200 GiB，node0 的 115.7 就不够了，而 node1 的 274.1 够
    f2 = derive_fraction(0.0, 2, memory=mem, need_bytes_per_part=int(200 * GIB))
    assert f2[0] == pytest.approx(115.7 / 200.0, abs=1e-3)
    assert f2[1] == 1.0


def test_derive_fraction_shmem_is_not_counted_as_reclaimable():
    """Shmem 在 FilePages 里但回收不掉（engram 表就是这么占着 node0 的）。"""
    with_shmem = {0: _mem(10.0, 200.0, 190.0, 0), 1: _mem(10.0, 200.0, 190.0, 1)}
    f = derive_fraction(0.0, 2, memory=with_shmem, need_bytes_per_part=int(50 * GIB))
    # 可用 = 10 + (200 - 190) = 20 GiB，不是 10 + 200
    assert f[0] == pytest.approx(20.0 / 50.0, abs=1e-3)


def test_derive_fraction_subtracts_the_reserve_from_each_node():
    mem = {0: _mem(120.0, 0.0, 0.0, 0), 1: _mem(120.0, 0.0, 0.0, 1)}
    f0 = derive_fraction(0.0, 2, memory=mem, need_bytes_per_part=NEED)
    f16 = derive_fraction(16.0, 2, memory=mem, need_bytes_per_part=NEED)
    assert f0 == [1.0, 1.0]  # 120 > 100
    assert f16 == [1.0, 1.0]  # 120-16=104，仍 > 100
    f40 = derive_fraction(40.0, 2, memory=mem, need_bytes_per_part=NEED)
    assert f40[0] == pytest.approx(80.0 / 100.0, abs=1e-3)  # 120-40=80 < 100


def test_derive_fraction_clamps_a_node_that_is_past_its_reserve():
    """可用量为负时算 0，不是负比例。"""
    mem = {0: _mem(1.0, 0.0, 0.0, 0), 1: _mem(400.0, 0.0, 0.0, 1)}
    f = derive_fraction(16.0, 2, memory=mem, need_bytes_per_part=NEED)
    assert f[0] == 0.0
    assert f[1] == 1.0


def test_derive_fraction_raises_when_the_reserve_exceeds_everything():
    mem = {0: _mem(1.0, 0.0, 0.0, 0), 1: _mem(1.0, 0.0, 0.0, 1)}
    with pytest.raises(PlacementError, match="no node has memory left"):
        derive_fraction(16.0, 2, memory=mem, need_bytes_per_part=NEED)


def test_derive_fraction_rejects_bad_arguments():
    mem = {0: _mem(100.0, 0.0, 0.0, 0), 1: _mem(100.0, 0.0, 0.0, 1)}
    with pytest.raises(ValueError, match="tp must be"):
        derive_fraction(0.0, 0, memory=mem, need_bytes_per_part=NEED)
    with pytest.raises(ValueError, match="reserve_gib must be"):
        derive_fraction(-1.0, 2, memory=mem, need_bytes_per_part=NEED)
    with pytest.raises(ValueError, match="nodes lists"):
        derive_fraction(0.0, 2, memory=mem, need_bytes_per_part=NEED, nodes=[0])


def test_derive_fraction_refuses_without_the_need():
    """没有「需要多少」就无法判断某个节点够不够。"""
    mem = {0: _mem(100.0, 0.0, 0.0, 0), 1: _mem(100.0, 0.0, 0.0, 1)}
    with pytest.raises(ValueError, match="need_bytes_per_part"):
        derive_fraction(0.0, 2, memory=mem, need_bytes_per_part=0)


def test_read_node_memory_parses_sysfs(tmp_path):
    for n, free in ((0, 18142884), (1, 234314980)):
        d = tmp_path / f"node{n}"
        d.mkdir()
        (d / "meminfo").write_text(
            f"Node {n} MemTotal:       395824700 kB\n"
            f"Node {n} MemFree:        {free} kB\n"
            f"Node {n} FilePages:      243787308 kB\n"
            f"Node {n} Shmem:          167679644 kB\n"
        )
    mem = np_.read_node_memory([0, 1], sysfs=str(tmp_path))
    assert mem[0].mem_free == 18142884 * 1024
    assert mem[0].reclaimable_file_cache == (243787308 - 167679644) * 1024
    assert mem[1].mem_free == 234314980 * 1024


def test_read_node_memory_raises_on_a_missing_field(tmp_path):
    d = tmp_path / "node0"
    d.mkdir()
    (d / "meminfo").write_text("Node 0 MemFree:        1 kB\n")
    with pytest.raises(PlacementError, match="missing"):
        np_.read_node_memory([0], sysfs=str(tmp_path))


@pytest.mark.parametrize(
    "frac, tp, match",
    [
        ([0.5, 0.3, 0.2], 2, "lists 3"),  # wrong length
        ([0.5], 2, "lists 1"),
        # 没有"和为 1"这条规则：各项独立，完美局部性是 [1.0]*tp
        ([1.5, 0.5], 2, r"outside \[0, 1\]"),  # 越界仍要拒
        ([-0.1, 1.0], 2, r"outside \[0, 1\]"),
        ([-0.1, 1.1], 2, "outside"),  # negative entry
        ([1.5, -0.5], 2, "outside"),
    ],
)
def test_validate_fraction_rejects(frac, tp, match):
    """A bad split must raise, never fall back to 50/50: two A/B arms that
    placed pages differently is exactly what pinning the fraction prevents."""
    with pytest.raises(ValueError, match=match):
        validate_fraction(frac, tp)


def test_plan_placement_validates_a_configured_fraction():
    src = FakeSource()
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        plan_placement(src, [0], [0], tp=2, fraction=[0.5, 1.6])


def test_plan_placement_accepts_full_locality():
    """``[1.0, 1.0]`` sums to 2 and is the intended default: every part keeps
    all of its own w13 local."""
    src = FakeSource()
    plan = plan_placement(src, [0], [0], tp=2, fraction=[1.0, 1.0])
    assert plan.fraction == (1.0, 1.0)
    # every w13 segment must be "local" -- nothing overflows
    reasons = {seg.reason for seg in plan.segments if seg.reason.startswith("w13")}
    assert reasons == {"w13-local"}, reasons


def test_plan_placement_records_where_the_fraction_came_from():
    """The log has to say *derived* or *configured*; "41/59" alone is not
    reproducible a day later."""
    src = FakeSource()
    plan = plan_placement(src, [0], [0], tp=2, fraction=[0.4, 0.6])
    assert plan.fraction_source == "configured"
    assert plan.fraction == (0.4, 0.6)


def test_needed_move_zero_is_rejected():
    """The whole point of ``needed_move``. This test exists so the guard itself
    is known to be able to go red: build the report an implementation that did
    nothing would produce, and require that it is refused."""
    rep = MigrationReport(
        planned_pages=1000,
        resident_pages=1000,
        needed_move=0,
        moved=1000,
        failed_pages=0,
        before_by_node={0: 1000},
        after_by_node={0: 1000},
    )
    assert rep.failed_pages == 0 and rep.moved == rep.resident_pages  # "looks perfect"
    with pytest.raises(MutationNotAppliedError, match="needed_move=0"):
        rep.require_mutation_applied()


def test_needed_move_positive_passes():
    rep = MigrationReport(planned_pages=1000, resident_pages=1000, needed_move=500, moved=1000, failed_pages=0)
    assert rep.require_mutation_applied() is rep


def test_report_merge_adds_every_counter():
    """Per-batch reports are merged under a lock; a counter dropped here would
    under-report failures, which is the direction that hides a bad run."""
    a = MigrationReport(
        planned_pages=10,
        resident_pages=8,
        needed_move=4,
        moved=3,
        failed_pages=5,
        before_by_node={0: 8},
        after_by_node={1: 3},
        errno_pages={"EBUSY(Device or resource busy)": 5},
    )
    b = MigrationReport(
        planned_pages=20,
        resident_pages=16,
        needed_move=6,
        moved=16,
        failed_pages=0,
        before_by_node={1: 16},
        after_by_node={1: 16},
    )
    a.merge(b)
    assert (a.planned_pages, a.resident_pages, a.needed_move, a.moved, a.failed_pages) == (30, 24, 10, 19, 5)
    assert a.before_by_node == {0: 8, 1: 16}
    assert a.after_by_node == {1: 19}
    assert a.errno_pages == {"EBUSY(Device or resource busy)": 5}


def test_apply_placement_rejects_bad_arguments():
    src = FakeSource()
    plan = plan_placement(src, [0], [0], tp=2, fraction=[0.5, 0.5])
    with pytest.raises(ValueError, match="threads"):
        np_.apply_placement(plan, threads=0)
    with pytest.raises(ValueError, match="batch_pages"):
        np_.apply_placement(plan, batch_pages=0)
    with pytest.raises(ValueError, match="MPOL_MF_MOVE"):
        np_.apply_placement(plan, flags=0)


def test_apply_placement_raises_when_nothing_is_resident(monkeypatch, tmp_path):
    """Zero resident pages is a failure, not a pass: it is what a wrong path, a
    cold cache or a mapping nobody touched all look like."""
    shard = tmp_path / "shard.safetensors"
    shard.write_bytes(b"\0" * (4 * 1024 * 1024))
    src = FakeSource(path=str(shard), data_start=0, num_layers=1, num_experts=1)
    src.ORDER = (
        ("w1", "weight", 1 << 20),
        ("w3", "weight", 1 << 20),
        ("w2", "weight", 1 << 20),
        ("w1", "scale", 4096),
        ("w3", "scale", 4096),
        ("w2", "scale", 4096),
    )
    src.PER_EXPERT = sum(n for _, _, n in src.ORDER)
    plan = plan_placement(src, [0], [0], tp=2, fraction=[0.5, 0.5])
    monkeypatch.setattr(np_._Libc, "resident", lambda self, addr, lo, n: [])
    with pytest.raises(ZeroPlacementError, match="resident"):
        np_.apply_placement(plan, threads=1)


def test_move_pages_syscall_number_is_never_guessed(monkeypatch):
    """A wrong syscall number is a *different* syscall, so an unknown machine
    must refuse rather than pick something plausible."""
    monkeypatch.setattr(np_.platform, "machine", lambda: "s390x")
    monkeypatch.setattr(np_._Libc, "_inst", None)
    with pytest.raises(PlacementError, match="refusing to guess"):
        np_._Libc()


def _nodes_available() -> bool:
    try:
        return len(np_.read_node_memory([0, 1])) == 2
    except Exception:
        return False


needs_two_nodes = pytest.mark.skipif(not _nodes_available(), reason="needs two NUMA nodes")


def test_runs_of_collapses_contiguous_pages():
    """readahead() takes an offset and a length, so the missing pages have to be
    handed to it as runs. A wrong grouping reads bytes we did not ask for."""
    assert np_._runs_of([]) == []
    assert np_._runs_of([5]) == [(5, 1)]
    assert np_._runs_of([1, 2, 3, 7, 8, 20]) == [(1, 3), (7, 2), (20, 1)]


@needs_two_nodes
def test_bound_policy_sets_and_resets_this_thread():
    libc = np_._Libc.get()
    assert libc.get_mempolicy() == (np_.MPOL_DEFAULT, 0)
    with np_._bound_policy(libc, np_.MPOL_BIND, 1):
        assert libc.get_mempolicy() == (np_.MPOL_BIND, 1 << 1)
    assert libc.get_mempolicy() == (np_.MPOL_DEFAULT, 0)


@needs_two_nodes
def test_bound_policy_resets_even_when_the_body_raises():
    """The reset is in a finally because a memory policy is a *task* attribute:
    leaking MPOL_BIND onto a pool thread pins every later allocation on that
    thread -- ours, torch's, anything -- to one node."""
    libc = np_._Libc.get()
    with pytest.raises(RuntimeError, match="boom"), np_._bound_policy(libc, np_.MPOL_BIND, 0):
        raise RuntimeError("boom")
    assert libc.get_mempolicy() == (np_.MPOL_DEFAULT, 0)


@needs_two_nodes
def test_a_pool_thread_does_not_inherit_the_main_threads_policy():
    """A thread only inherits a memory policy if it was *created* after the
    policy was set, so an existing pool keeps MPOL_DEFAULT -- which is why every
    worker sets its own policy inside the fault-in leg."""
    from concurrent.futures import ThreadPoolExecutor

    libc = np_._Libc.get()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(lambda: None).result()  # force the thread to exist first
        with np_._bound_policy(libc, np_.MPOL_BIND, 1):
            assert libc.get_mempolicy() == (np_.MPOL_BIND, 1 << 1)
            assert pool.submit(libc.get_mempolicy).result() == (np_.MPOL_DEFAULT, 0)
    finally:
        pool.shutdown()


class _ResetFails:
    """A ``_Libc`` whose *reset* arm fails; the bind arm is the real syscall,
    because the question is what the calling thread is left as and a fake cannot
    answer that. ``fail_times`` bounds how long the reset stays broken, so the
    retry can be told apart from the give-up."""

    def __init__(self, real, fail_times=1 << 30):
        self.real, self.left, self.resets = real, fail_times, 0

    def set_mempolicy(self, mode, node):
        if mode == np_.MPOL_DEFAULT and node is None:
            self.resets += 1
            if self.left > 0:
                self.left -= 1
                raise PlacementError("set_mempolicy(mode=0, node=None) failed: injected")
        return self.real.set_mempolicy(mode, node)

    def __getattr__(self, name):  # get_mempolicy and the rest stay real
        return getattr(self.real, name)


class _RecordingLogger:
    def __init__(self):
        self.lines = []

    def _rec(self, level):
        def log(fmt, *args):
            self.lines.append((level, fmt % args if args else fmt))

        return log

    def __getattr__(self, name):
        return self._rec(name)


@needs_two_nodes
def test_a_failing_reset_never_replaces_the_bodys_exception(monkeypatch):
    """The finally must not throw over the thing it was protecting: a raising
    reset would replace the fault-in leg's own error, so the one traceback an
    operator gets would name the cleanup and not the cause."""
    real = np_._Libc.get()
    rec = _RecordingLogger()
    monkeypatch.setattr(np_, "logger", rec)
    try:
        with (
            pytest.raises(RuntimeError, match="the real failure"),
            np_._bound_policy(_ResetFails(real), np_.MPOL_BIND, 1),
        ):
            raise RuntimeError("the real failure")
        crit = [m for lvl, m in rec.lines if lvl == "critical"]
        assert crit and "left bound to node1" in crit[0], rec.lines
        # and it told the truth: the thread really is still bound
        assert real.get_mempolicy() == (np_.MPOL_BIND, 1 << 1)
    finally:
        real.set_mempolicy(np_.MPOL_DEFAULT, None)
    assert real.get_mempolicy() == (np_.MPOL_DEFAULT, 0)


@needs_two_nodes
def test_a_failing_reset_is_loud_when_the_body_succeeded(monkeypatch):
    """The other half: a clean body must not return normally while the thread it
    ran on is left hard-bound. Silence here is the failure mode -- the process
    carries on and dies somewhere else, hours later, with no trace back."""
    real = np_._Libc.get()
    monkeypatch.setattr(np_, "logger", _RecordingLogger())
    try:
        with pytest.raises(PlacementError, match="injected"), np_._bound_policy(_ResetFails(real), np_.MPOL_BIND, 1):
            pass
    finally:
        real.set_mempolicy(np_.MPOL_DEFAULT, None)


@needs_two_nodes
def test_the_reset_is_retried_once_before_it_gives_up(monkeypatch):
    """A transient failure must not cost a hard-bound thread: retry, then ask
    get_mempolicy rather than trusting the second call either."""
    real = np_._Libc.get()
    monkeypatch.setattr(np_, "logger", _RecordingLogger())
    flaky = _ResetFails(real, fail_times=1)
    with np_._bound_policy(flaky, np_.MPOL_BIND, 1):
        assert real.get_mempolicy() == (np_.MPOL_BIND, 1 << 1)
    assert flaky.resets == 2
    assert real.get_mempolicy() == (np_.MPOL_DEFAULT, 0)


@needs_two_nodes
def test_a_pool_thread_is_clean_for_the_next_task_after_one_raises():
    """Thread-pool reuse is how a leak escapes the batch that caused it.
    ``max_workers=1`` so the reuse is guaranteed and asserted (same tid), not
    hoped for."""
    from concurrent.futures import ThreadPoolExecutor

    libc = np_._Libc.get()
    pool = ThreadPoolExecutor(max_workers=1)
    try:

        def failing_task():
            with np_._bound_policy(libc, np_.MPOL_BIND, 1):
                raise RuntimeError("batch died mid fault-in")

        def next_task():
            return threading.get_ident(), libc.get_mempolicy()

        first_tid = pool.submit(threading.get_ident).result()
        with pytest.raises(RuntimeError, match="batch died"):
            pool.submit(failing_task).result()
        tid, policy = pool.submit(next_task).result()
        assert tid == first_tid, "the pool did not reuse the thread; this test proves nothing"
        assert policy == (np_.MPOL_DEFAULT, 0), (
            f"the next task on the same pool thread started under {policy}; "
            f"MPOL_BIND leaked out of the task that raised"
        )
    finally:
        pool.shutdown()


@needs_two_nodes
def test_a_failing_fault_leg_leaves_the_calling_thread_unbound(tmp_path, monkeypatch):
    """End to end, on the thread that matters: the fault-in leg runs **inline on
    the caller** whenever ``threads == 1``, and in production that caller is the
    engine's own thread, so a leak there is permanent for the process and is
    inherited by every thread created afterwards."""
    src, evict = _cold_shard(tmp_path)
    evict()
    plan = plan_placement(src, [0], [0, 1], tp=2, fraction=[1.0, 0.0], streaming_node=1)
    libc = np_._Libc.get()

    def boom(fd, offset, count):
        raise PlacementError("readahead failed: injected mid fault-in")

    monkeypatch.setattr(libc, "readahead", boom)
    try:
        with pytest.raises(PlacementError, match="injected mid fault-in"):
            np_.apply_placement(plan, threads=1, readahead_pages=64)
        assert libc.get_mempolicy() == (np_.MPOL_DEFAULT, 0), (
            "the fault-in leg raised and left the CALLING thread bound; in production "
            "that is the engine thread, for the life of the process"
        )
        # the amplifier: a thread created after the leak inherits it at clone time
        seen = []
        t = threading.Thread(target=lambda: seen.append(libc.get_mempolicy()))
        t.start()
        t.join()
        assert seen == [(np_.MPOL_DEFAULT, 0)], f"a thread created after the fault-in leg inherited {seen}"
    finally:
        libc.set_mempolicy(np_.MPOL_DEFAULT, None)


def test_set_mempolicy_refuses_a_node_past_the_nodemask():
    """A node id wider than the mask would wrap onto a *different* node -- the
    same class of silent-wrong as guessing a syscall number."""
    libc = np_._Libc.get()
    with pytest.raises(PlacementError, match="nodemask"):
        libc.set_mempolicy(np_.MPOL_BIND, np_._NODEMASK_BITS)


def test_apply_placement_validates_the_fault_in_arguments():
    src = FakeSource()
    plan = plan_placement(src, [0], [0], tp=2, fraction=[0.5, 0.5])
    with pytest.raises(ValueError, match="readahead_pages"):
        np_.apply_placement(plan, readahead_pages=0)
    with pytest.raises(ValueError, match="fault_policy"):
        np_.apply_placement(plan, fault_policy=np_.MPOL_DEFAULT)


def test_require_fault_applied_is_red_when_nothing_was_faulted_in():
    """The fault leg's version of the needed_move assertion, from the red side:
    a report that read nothing must not pass as an arm, however clean it looks."""
    rep = MigrationReport(planned_pages=1000, fault_planned_pages=1000, faulted_in=0, fault_failed_pages=0)
    with pytest.raises(MutationNotAppliedError, match="faulted_in=0"):
        rep.require_fault_applied()


def test_require_applied_passes_when_the_fault_leg_did_all_the_work():
    """An all-cold file gives the migrate leg nothing to move, so needed_move=0
    is the *expected* result of a successful bind -- not a failure. The gate is
    examined_pages, not resident_pages."""
    rep = MigrationReport(
        planned_pages=1000,
        resident_pages=0,
        examined_pages=1000,
        needed_move=0,
        moved=1000,
        fault_planned_pages=1000,
        faulted_in=1000,
        fault_by_node={0: 1000},
    )
    assert rep.require_applied() is rep


def test_require_applied_is_red_when_neither_leg_applied_anything():
    rep = MigrationReport(
        planned_pages=1000, resident_pages=0, examined_pages=0, needed_move=0, fault_planned_pages=0, faulted_in=0
    )
    with pytest.raises(MutationNotAppliedError, match="neither leg"):
        rep.require_applied()


def test_require_applied_still_checks_the_migrate_leg_when_pages_were_resident():
    rep = MigrationReport(planned_pages=1000, resident_pages=1000, examined_pages=1000, needed_move=0, moved=1000)
    with pytest.raises(MutationNotAppliedError, match="needed_move=0"):
        rep.require_applied()


def test_percentages_use_examined_not_resident():
    """With everything brought in by the fault leg ``resident_pages`` is 0, so a
    resident denominator prints 'before={0: 5508000.00%}'."""
    rep = MigrationReport(
        planned_pages=100,
        resident_pages=0,
        examined_pages=100,
        moved=100,
        before_by_node={0: 100},
        after_by_node={0: 100},
    )
    assert "5508000" not in str(rep)
    assert "before_of_examined={0: '100.00%'}" in str(rep)


def test_fault_throughput_is_reported():
    """The read rate is not optional: at this size it is the start-up time."""
    rep = MigrationReport(faulted_in=262144, fault_seconds=0.5)
    assert rep.fault_bytes == 262144 * np_.PAGE_SIZE
    assert rep.fault_gib_per_s == pytest.approx(2.0)
    assert "GiB/s" in rep.fault_str()
    # No division by zero when the leg did not run.
    assert MigrationReport().fault_gib_per_s == 0.0


def _can_migrate(tmp_path) -> bool:
    """move_pages(MPOL_MF_MOVE_ALL) needs CAP_SYS_NICE; skip rather than fail."""
    f = tmp_path / "probe"
    f.write_bytes(b"\0" * (64 * 1024))
    libc = np_._Libc.get()
    addr, size, fd = libc.map_readonly(str(f))
    try:
        pages = libc.resident(addr, 0, size // np_.PAGE_SIZE)
        if not pages:
            return False
        np_._touch((__import__("ctypes").c_char * size).from_address(addr), pages)
        libc.move(addr, pages, 0, np_.MPOL_MF_MOVE_ALL)
        return True
    except PlacementError:
        return False
    finally:
        libc.unmap(addr, size, fd)


def _cold_shard(tmp_path, experts=2):
    """A file whose pages are NOT in the page cache.

    Evicted with ``posix_fadvise(DONTNEED)`` on this one fd -- never
    ``drop_caches``, which would throw away the checkpoint every other tenant on
    this host is reading.
    """
    w13, w2, sc = 1 << 20, 1 << 20, 4096
    order = (
        ("w1", "weight", w13),
        ("w3", "weight", w13),
        ("w2", "weight", w2),
        ("w1", "scale", sc),
        ("w3", "scale", sc),
        ("w2", "scale", sc),
    )
    per = sum(n for _, _, n in order)
    path = tmp_path / "cold.safetensors"
    with open(path, "wb") as f:
        f.write(os.urandom(1234 + experts * per))

    src = FakeSource(path=str(path), data_start=1234, num_layers=1, num_experts=experts)
    src.ORDER, src.PER_EXPERT = order, per

    def evict():
        os.sync()
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    return src, evict


@needs_two_nodes
def test_fault_in_places_pages_that_move_pages_cannot_touch(tmp_path):
    """Two arms, same bytes and same code, only the target node differs, and the
    measured landing site must follow it. With one arm, "the pages are resident
    and on node X afterwards" proves only that reading a file puts pages
    somewhere."""
    if not _can_migrate(tmp_path):
        pytest.skip("move_pages(MPOL_MF_MOVE_ALL) not permitted here")
    src, evict = _cold_shard(tmp_path)
    landed = {}
    for node in (0, 1):
        evict()
        _plan, rep = np_.place_weights(
            src,
            [0],
            [0, 1],
            tp=2,
            streaming_node=node,
            fraction=[1.0, 0.0] if node == 0 else [0.0, 1.0],
            threads=2,
            readahead_pages=64,
        )
        # the mutation was applied: these pages did not exist before this call
        assert rep.fault_planned_pages > 0
        assert rep.faulted_in > 0
        rep.require_applied()
        # and "we read it" was not taken for success -- mincore was asked again
        assert rep.fault_failed_pages == 0, rep.fault_str()
        total = sum(rep.fault_by_node.values())
        landed[node] = 100.0 * rep.fault_by_node.get(node, 0) / total
    assert landed[0] > 90 and landed[1] > 90, (
        f"the bind did not decide where the pages landed: {landed}; without that "
        f"the fault-in leg is just reading the file"
    )


@needs_two_nodes
def test_dry_run_reports_the_read_without_doing_it(tmp_path):
    """The size of the read has to be knowable before it is paid: a dry run must
    report it and fault nothing in."""
    src, evict = _cold_shard(tmp_path)
    evict()
    plan = plan_placement(src, [0], [0, 1], tp=2, fraction=[0.5, 0.5], streaming_node=0)
    rep = np_.apply_placement(plan, threads=1, dry_run=True)
    assert rep.fault_planned_pages > 0
    assert rep.faulted_in == 0 and rep.fault_seconds == 0.0
    # still cold afterwards
    rep2 = np_.apply_placement(plan, threads=1, dry_run=True)
    assert rep2.fault_planned_pages == rep.fault_planned_pages


@needs_two_nodes
def test_fault_in_disabled_over_a_cold_file_is_a_failure_not_a_pass(tmp_path):
    """Placing nothing must never read as success -- this is the shape the whole
    module exists to refuse."""
    src, evict = _cold_shard(tmp_path)
    evict()
    plan = plan_placement(src, [0], [0, 1], tp=2, fraction=[0.5, 0.5], streaming_node=0)
    with pytest.raises(ZeroPlacementError, match="touched nothing"):
        np_.apply_placement(plan, threads=1, fault_in=False)


def test_a_caller_that_derived_the_fraction_says_so(tmp_path):
    """`derived` and `configured` are different claims: one is reproducible and
    one is the machine's mood at that minute. A caller (``setup_placement``)
    that derived the split and then pins it for one call must not have the plan
    report it as operator-pinned."""
    src = FakeSource()
    plan = plan_placement(src, [0], [0], tp=2, fraction=[0.4, 0.6], fraction_source="derived")
    assert plan.fraction_source == "derived"
    assert plan_placement(src, [0], [0], tp=2, fraction=[0.4, 0.6]).fraction_source == "configured"


def test_populate_issues_one_madvise_per_contiguous_run():
    """Per run, not per page: that ratio is the whole point."""
    calls = []

    class FakeLibc:
        def populate(self, addr, first, count):
            calls.append((first, count))
            return True

    np_._populate(FakeLibc(), 0, None, [1, 2, 3, 7, 8, 20])
    assert calls == [(1, 3), (7, 2), (20, 1)]


def test_populate_falls_back_to_the_per_page_touch_on_an_old_kernel():
    """MADV_POPULATE_READ is 5.14+. Without it the pages must still reach the
    page table -- skipping that silently gives -ENOENT from every move_pages."""
    touched = []

    class FakeLibc:
        def populate(self, addr, first, count):
            return False

    class FakeBuf:
        def __getitem__(self, i):
            touched.append(i)
            return b"\0"

    np_._populate(FakeLibc(), 0, FakeBuf(), [4, 5])
    assert touched == [4 * np_.PAGE_SIZE, 5 * np_.PAGE_SIZE]


def test_the_fault_leg_does_not_touch_page_by_page(monkeypatch, tmp_path):
    """Reinstating the per-page touch in the fault-in leg must go red here
    instead of quietly costing throughput."""
    if not _nodes_available():
        pytest.skip("needs two NUMA nodes")
    src, evict = _cold_shard(tmp_path)
    evict()
    plan = plan_placement(src, [0], [0, 1], tp=2, fraction=[0.5, 0.5], streaming_node=0)

    def no_touch(buf, pages):
        raise AssertionError(
            "the fault-in leg touched pages from Python; that holds the GIL across every "
            "major fault and serialises the threads (0.56 vs 2.85 GiB/s measured)"
        )

    monkeypatch.setattr(np_, "_touch", no_touch)
    rep = np_.apply_placement(plan, threads=2, readahead_pages=64)
    assert rep.faulted_in > 0


class _FakeLibc:
    """Enough of _Libc to observe WHICH pages each leg touches. Deterministic on
    purpose: which pages get examined is a decision, and a decision can be
    asserted exactly."""

    def __init__(self, size, resident, node_of, misplace_to=None):
        self.size = size
        self.resident_pages = set(resident)
        self.node_of = dict(node_of)  # page -> node it is currently on
        self.misplace_to = misplace_to or {}  # page -> node it will land on
        self.examined = []  # pages handed to a real move
        self.queried = []

    def map_readonly(self, path):
        return 0, self.size, 3

    def unmap(self, addr, size, fd=None):
        pass

    def resident(self, addr, first_page, npages):
        return [p for p in range(first_page, first_page + npages) if p in self.resident_pages]

    def populate(self, addr, first, count):
        return True

    def readahead(self, fd, offset, count):
        pass

    def set_mempolicy(self, mode, node):
        self._policy = node

    def move(self, addr, pages, node, flags):
        if node is None:
            self.queried.extend(pages)
            return [self.node_of.get(p, 0) for p in pages]
        self.examined.extend(pages)
        for p in pages:
            self.node_of[p] = node
        return [node] * len(pages)

    def fault_in(self, pages):
        """What the fault-in leg's bind does in this fake: the pages appear,
        on the bound node unless misplace_to says otherwise."""
        for p in pages:
            self.resident_pages.add(p)
            self.node_of[p] = self.misplace_to.get(p, self._policy)


def _fake_plan_single_file(npages, node):
    """A plan of one file, one run, all on `node` -- so the scope maths is the
    only thing under test."""
    return PlacementPlan(
        segments=(),
        fraction=(1.0,),
        fraction_source="configured",
        part_nodes=(node,),
        streaming_node=node,
        page_runs={"/fake": ((0, npages, node),)},
    )


@pytest.fixture
def fake_libc(monkeypatch):
    holder = {}

    def install(libc):
        holder["libc"] = libc
        monkeypatch.setattr(np_._Libc, "get", classmethod(lambda cls: libc))
        # the fault-in leg's "read" is readahead+populate in the real thing;
        # here populate is where the pages appear.
        real_populate = libc.populate

        def populate(addr, first, count):
            libc.fault_in([p for p in range(first, first + count) if p not in libc.resident_pages])
            return real_populate(addr, first, count)

        libc.populate = populate
        return libc

    return install


def test_suspect_scope_examines_only_the_flagged_and_the_unseen(fake_libc):
    """The narrowing, stated exactly: census-resident ∪ fault-in's misplaced."""
    n = 100
    # pages 0-9 already cached (on the wrong node); 10-99 not cached yet.
    # Of the ones read in, 50 and 51 will land on node 1 despite the bind.
    libc = fake_libc(
        _FakeLibc(n * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1})
    )
    plan = _fake_plan_single_file(n, node=0)
    rep = np_.apply_placement(plan, threads=1, migrate_scope="suspect")
    assert rep.fault_misplaced == 2
    assert sorted(libc.examined) == list(range(10)) + [50, 51]


def test_all_scope_re_examines_every_page(fake_libc):
    """The old arm, kept so criterion 1 has something to compare against."""
    n = 100
    libc = fake_libc(
        _FakeLibc(n * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1})
    )
    plan = _fake_plan_single_file(n, node=0)
    np_.apply_placement(plan, threads=1, migrate_scope="all")
    assert sorted(libc.examined) == list(range(n))


def test_resident_scope_is_the_negative_control(fake_libc):
    """It drops the misplaced set on purpose: pages 50/51 stay on node 1. If
    this arm ever matched 'suspect', the correction would be doing nothing and
    'faster' would be indistinguishable from 'skipped the work'."""
    n = 100
    libc = fake_libc(
        _FakeLibc(n * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1})
    )
    plan = _fake_plan_single_file(n, node=0)
    np_.apply_placement(plan, threads=1, migrate_scope="resident")
    assert sorted(libc.examined) == list(range(10))
    assert libc.node_of[50] == 1 and libc.node_of[51] == 1  # never corrected


def test_migrate_scope_is_validated():
    src = FakeSource()
    plan = plan_placement(src, [0], [0], tp=2, fraction=[0.5, 0.5])
    with pytest.raises(ValueError, match="migrate_scope"):
        np_.apply_placement(plan, migrate_scope="everything")


def test_audit_is_measured_over_every_planned_page_not_the_examined_ones(fake_libc):
    """``after_by_node`` covers only what the migrate leg examined, and
    'suspect' examines a fraction of what 'all' does, so comparing the two
    reports would read a narrower arm as a worse one."""
    n = 100
    fake_libc(
        _FakeLibc(n * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1})
    )
    plan = _fake_plan_single_file(n, node=0)
    rep = np_.apply_placement(plan, threads=1, migrate_scope="suspect")
    assert rep.examined_pages == 12  # the report's view: 12 pages
    audit = np_.audit_placement(plan, threads=1)
    assert audit.resident_pages == n  # the audit's view: all 100
    assert audit.on_target == n and audit.off_target == 0
    assert audit.on_target_pct == 100.0


def test_audit_counts_pages_left_on_the_wrong_node(fake_libc):
    fake_libc(
        _FakeLibc(
            100 * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1}
        )
    )
    plan = _fake_plan_single_file(100, node=0)
    np_.apply_placement(plan, threads=1, migrate_scope="resident")
    audit = np_.audit_placement(plan, threads=1)
    assert audit.off_target == 2  # 50 and 51, never corrected
    assert audit.on_target_pct == 98.0
    assert audit.by_node == {0: 98, 1: 2}
    assert audit.target_by_node == {0: 100}


def test_final_by_node_covers_every_resident_page_whatever_the_scope(fake_libc):
    """``after_by_node`` shrinks with the scope; ``final_by_node`` must not. All
    three scopes see the same 100 resident pages, and the two that correct the
    misplaced ones must report the same final distribution."""
    finals = {}
    for scope in ("all", "suspect", "resident"):
        fake_libc(
            _FakeLibc(
                100 * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1}
            )
        )
        plan = _fake_plan_single_file(100, node=0)
        rep = np_.apply_placement(plan, threads=1, migrate_scope=scope)
        assert sum(rep.final_by_node.values()) == 100, (scope, rep.final_by_node)
        finals[scope] = dict(rep.final_by_node)
    assert finals["all"] == finals["suspect"] == {0: 100}
    # the negative control leaves 50/51 behind, and the tally shows it
    assert finals["resident"] == {0: 98, 1: 2}


def test_final_by_node_agrees_with_an_independent_audit_when_nothing_drifts(fake_libc):
    """With no drift, the cheap tally and the from-scratch audit are the same
    numbers. Only the fake can prove this: on a real machine the tally is the
    state when each page was placed and the audit the state when it ran, so
    demanding equality there would be testing the kernel's mood."""
    for scope in ("all", "suspect", "resident"):
        fake_libc(
            _FakeLibc(
                100 * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1}
            )
        )
        plan = _fake_plan_single_file(100, node=0)
        rep = np_.apply_placement(plan, threads=1, migrate_scope=scope)
        audit = np_.audit_placement(plan, threads=1)
        assert rep.final_by_node == audit.by_node, scope


def test_landed_is_reported_against_the_pages_it_was_actually_asked_for(fake_libc):
    """``landed`` is a distribution over the NON-RESIDENT pages only, while the
    plan's percentages are over ALL planned pages, so comparing the two reads as
    a placement failure that may be pure denominator. ``asked_for`` is the
    comparator over the same population."""
    # 60 pages planned for node 0, 40 for node 1; the node-0 ones are already
    # cached, so the fault-in leg is only ever asked for node-1 pages -- and 10
    # of those 40 land on node 0 anyway, so `landed` and `asked_for` are
    # DIFFERENT distributions. A fixture where they coincide cannot tell the
    # right denominator from the wrong one.
    libc = _FakeLibc(
        100 * np_.PAGE_SIZE,
        resident=range(60),
        node_of={p: 0 for p in range(60)},
        misplace_to={p: 0 for p in range(60, 70)},
    )
    fake_libc(libc)
    plan = PlacementPlan(
        segments=(),
        fraction=(0.6, 0.4),
        fraction_source="configured",
        part_nodes=(0, 1),
        streaming_node=1,
        page_runs={"/fake": ((0, 60, 0), (60, 100, 1))},
    )
    assert plan.planned_pages_by_node() == {0: 60, 1: 40}  # the global plan: 60% node0
    rep = np_.apply_placement(plan, threads=1, migrate_scope="suspect")
    # ...but what fault-in was asked for is 100% node1, and that is what its
    # landed distribution has to be read against.
    # asked for: all 40 on node1. landed: 10 of them on node0.
    assert rep.fault_target_by_node == {1: 40}
    assert rep.fault_by_node == {0: 10, 1: 30}
    assert "asked_for_of_faulted={1: '100.00%'}" in rep.fault_str()
    assert "landed_of_faulted={0: '25.00%', 1: '75.00%'}" in rep.fault_str()
    # and the global plan (60% node0) is none of those numbers -- which is the
    # comparison that read as a 7 pp placement failure on card.
    assert plan.planned_pages_by_node() == {0: 60, 1: 40}


def test_final_on_target_scores_every_placed_page_against_its_own_plan(fake_libc):
    """One number for "did it work", instead of two distributions to difference.
    The negative control leaves 2 of 100 behind and must say 98%."""
    for scope, want_ok in (("all", 100), ("suspect", 100), ("resident", 98)):
        fake_libc(
            _FakeLibc(
                100 * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1}
            )
        )
        plan = _fake_plan_single_file(100, node=0)
        rep = np_.apply_placement(plan, threads=1, migrate_scope=scope)
        scored = rep.final_on_target + rep.final_off_target
        assert scored == 100, (scope, scored)
        assert rep.final_on_target == want_ok, (scope, rep.final_on_target)


def test_a_placement_that_mostly_missed_warns(fake_libc, monkeypatch):
    """Loud, not silent: no per-leg counter reports a run whose legs all
    succeeded but which left most pages off target. Captured off the module's
    own logger rather than ``caplog``, which never sees it because this package
    logs through ``vllm.logger`` and that does not propagate to root."""
    said = []
    monkeypatch.setattr(np_.logger, "warning", lambda msg, *a, **k: said.append(msg % a if a else msg))
    fake_libc(
        _FakeLibc(
            100 * np_.PAGE_SIZE,
            resident=range(10),
            node_of={p: 1 for p in range(10)},
            misplace_to={p: 1 for p in range(20, 60)},
        )
    )
    plan = _fake_plan_single_file(100, node=0)
    rep = np_.apply_placement(plan, threads=1, migrate_scope="resident")
    assert rep.final_off_target == 40
    assert any("on their planned node" in m for m in said), said


def test_a_placement_that_landed_does_not_warn(fake_libc, monkeypatch):
    """The other half: the warning has to stay quiet when it should, or it is
    noise and gets ignored the one time it matters."""
    said = []
    monkeypatch.setattr(np_.logger, "warning", lambda msg, *a, **k: said.append(msg % a if a else msg))
    fake_libc(
        _FakeLibc(
            100 * np_.PAGE_SIZE, resident=range(10), node_of={p: 1 for p in range(10)}, misplace_to={50: 1, 51: 1}
        )
    )
    plan = _fake_plan_single_file(100, node=0)
    np_.apply_placement(plan, threads=1, migrate_scope="suspect")
    assert not [m for m in said if "on their planned node" in m], said


def test_split_share_is_w13s_share_of_the_planned_bytes():
    """w2 and every scale go whole to the streaming node; `fraction` never
    touches them, so it governs only w13's share of the planned bytes."""
    src = FakeSource(num_layers=1, num_experts=4)
    plan = plan_placement(src, [0], range(4), tp=2, fraction=[0.5, 0.5], streaming_node=1)
    w13 = 2 * W13_BYTES  # w1.weight + w3.weight
    total = w13 + W2_BYTES + 3 * SCALE_BYTES  # + w2.weight + the three scales
    assert plan.split_share == pytest.approx(w13 / total, abs=1e-6)
    assert plan.split_share == pytest.approx(0.6275, abs=1e-3)


class FakeBankSource(FakeSource):
    """A source that splits w2 as well, which the NZ bank does."""

    split_tensors = (("w1", "weight"), ("w3", "weight"), ("w2", "weight"))


def test_explain_names_the_tensors_it_actually_split():
    """🔴 The report must not hard-code "w13".

    Measured 2026-11-15 on the `now` arm: the server printed `split_share =
    0.9412` -- which is (w13 + w2)/total, i.e. w2 HAD been split per part -- and
    on the same line claimed "the rest is w2+scale and goes whole to node1".
    The number and the sentence contradicted each other, and only the number was
    right. A reader who trusted the sentence would have gone hunting for a w2
    locality bug that does not exist.
    """
    src = FakeBankSource(num_layers=1, num_experts=4)
    plan = plan_placement(src, [0], range(4), tp=2, fraction=[1.0, 1.0], streaming_node=1)
    assert "w2.weight" in plan.split_tensor_names
    assert plan.whole_tensor_names == ("w1.scale", "w2.scale", "w3.scale")
    total = 2 * W13_BYTES + W2_BYTES + 3 * SCALE_BYTES
    assert plan.split_share == pytest.approx((2 * W13_BYTES + W2_BYTES) / total, abs=1e-6)
    rest = plan.explain().split("the rest (")[1]
    assert "w2.weight" not in rest.split("goes whole")[0]
    assert "w2.scale" in rest


def test_explain_still_says_w13_only_for_the_checkpoint():
    """The counterexample: the checkpoint cannot split w2, and must still say so."""
    plan = plan_placement(
        FakeSource(num_layers=1, num_experts=4), [0], range(4), tp=2, fraction=[0.5, 0.5], streaming_node=1
    )
    assert plan.split_tensor_names == ("w1.weight", "w3.weight")
    assert "w2.weight" in plan.whole_tensor_names
    assert "w2.weight" in plan.explain().split("the rest (")[1]


@pytest.mark.parametrize("f0, expect_all_pages", [(0.2229, 13.99), (0.3552, 22.29)])
def test_fraction_times_split_share_is_the_all_pages_share(f0, expect_all_pages):
    """``fraction[0]`` is a share OF W13 and ``fraction[0] * split_share`` is the
    all-pages share: both correct, over different populations. The two cases are
    each other's counterexample -- 0.2229 prints as "22.29%", which is exactly
    how a w13 ratio gets read as an all-pages plan."""
    src = FakeSource(num_layers=1, num_experts=8)
    plan = plan_placement(src, [0], range(8), tp=2, part_nodes=[0, 1], streaming_node=1, fraction=[f0, 1 - f0])
    share = 100.0 * plan.planned_pages_by_node()[0] / plan.planned_pages
    assert share == pytest.approx(expect_all_pages, abs=0.05)
    assert share == pytest.approx(100 * f0 * plan.split_share, abs=0.05)
    # and the two are NOT equal -- so nobody "fixes" one into the other
    assert abs(share - 100 * f0) > 5.0


def test_explain_names_the_population_of_every_percentage():
    """The deliverable: one reading that puts them side by side. If these
    labels are dropped, the next person repeats the same misreading."""
    src = FakeSource(num_layers=1, num_experts=4)
    plan = plan_placement(src, [0], range(4), tp=2, part_nodes=[0, 1], streaming_node=1, fraction=[0.2229, 0.7771])
    text = plan.explain()
    assert "OF W13 ONLY" in text
    assert "of ALL planned pages" in text
    assert "FAULTED-IN SUBSET" in text
    assert "ALL RESIDENT PLANNED PAGES" in text
    # the arithmetic is shown, not just asserted in a docstring
    assert "0.2229 x 0.6275 = 13.99%" in text
    assert "{0: '13.99%', 1: '86.01%'}" in text


def test_runtime_labels_carry_their_denominator():
    """Same discipline in the per-run log line."""
    rep = MigrationReport(
        planned_pages=100,
        resident_pages=10,
        examined_pages=10,
        faulted_in=90,
        fault_planned_pages=90,
        fault_by_node={0: 90},
        fault_target_by_node={0: 90},
        before_by_node={0: 10},
        after_by_node={0: 10},
        final_by_node={0: 100},
        final_on_target=100,
    )
    assert "landed_of_faulted=" in rep.fault_str()
    assert "asked_for_of_faulted=" in rep.fault_str()
    text = str(rep)
    assert "FINAL_of_all_resident=" in text
    assert "before_of_examined=" in text and "after_of_examined=" in text


def test_audit_labels_carry_their_denominator():
    a = np_.PlacementAudit(
        resident_pages=100, on_target=98, off_target=2, by_node={0: 98, 1: 2}, target_by_node={0: 100}
    )
    assert "actual_of_resident=" in str(a) and "planned_of_resident=" in str(a)
