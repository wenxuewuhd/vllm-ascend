"""Unit tests for the offline NZ weight-bank builder.

No NPU: the ND -> NZ step is a pure byte permutation, so every assertion runs on the
host. What these tests guard is the *writer* and the *segment table* -- layout
arithmetic, section placement, the pass-through of the scales, and the equivalence
that justifies the whole split layout (four per-part sections reassembled through the
segment table must equal the device's whole-expert permutation, byte for byte). That
the permutation itself matches CANN's real cast is a separate check
(``nz_geometry.verify_nz_geometry``, which needs a card).
"""

import json
import os

import numpy as np
import pytest
import torch

from vllm_ascend.kt_offload import bank_build, nz_geometry

# A layout small enough to build in a test while satisfying every precondition
# ``check()`` enforces: both Kb and both N multiples of C0, group_size // 2 == C0, and
# a w13 section record that is a whole number of 4 KiB pages (32 * 128 = 4096).
TINY = bank_build.BankLayout(num_experts=3, hidden=256, intermediate=64, group_size=32, tp_count=2)

#: What the real deployment moves per expert, spelled out so a layout change that
#: silently alters it fails here rather than in a 269 GiB build.
REAL_BYTES_PER_EXPERT = 18_800_640
REAL_LAYER_BYTES = 384 * REAL_BYTES_PER_EXPERT


class FakeSource:
    """Stands in for ``WeightSource``: deterministic ND bytes at the right sizes."""

    def __init__(self, layout: bank_build.BankLayout):
        self.layout = layout

    def _nbytes(self, proj: str, kind: str) -> int:
        lay = self.layout
        if kind == "weight":
            return lay.w2_n * lay.w2_kb if proj == "w2" else lay.intermediate * lay.w13_kb
        if proj == "w2":
            return lay.section_record("w2_scale")
        return lay.section_record("w13_scale") // 2

    def locate(self, layer: int, expert: int, proj: str, kind: str = "weight"):
        return (layer, expert, proj, kind)

    def read(self, span):
        layer, expert, proj, kind = span
        n = self._nbytes(proj, kind)
        # Content depends on every coordinate, so a section written at the wrong offset
        # cannot accidentally compare equal to the right one.
        seed = (layer * 1_000 + expert) * 17 + (hash((proj, kind)) % 251)
        idx = np.arange(n, dtype=np.uint64)
        vals = ((idx + 1) * np.uint64(0x9E3779B97F4A7C15) >> np.uint64(24)) ^ np.uint64(seed)
        return torch.from_numpy((vals & np.uint64(0xFF)).astype(np.uint8))


# --------------------------------------------------------------------------- layout


def test_real_layout_sizes_match_the_deployment():
    lay = bank_build.BankLayout().check()
    assert lay.w13_part_n == 1152
    assert lay.w13_kb == 2560
    assert lay.w13_rows_total == 4608
    assert lay.w2_n == 5120
    assert lay.w2_kb == 1152
    assert lay.w13_slab_bytes == 73_728
    assert lay.w13_part_run_bytes == 18_432
    assert lay.w13_gidx_count == 160
    for name in bank_build.W13_SECTIONS:
        assert lay.section_record(name) == 2_949_120
    assert lay.section_record("w2") == 5_898_240
    assert lay.section_record("w13_scale") == 737_280
    assert lay.section_record("w2_scale") == 368_640
    assert lay.device_record("w13") == 11_796_480
    assert lay.bytes_per_expert == REAL_BYTES_PER_EXPERT
    assert lay.layer_bytes == REAL_LAYER_BYTES


def test_sections_tile_the_layer_exactly():
    """Seven sections, no gaps, no overlap, last one ends at the file size."""
    lay = TINY.check()
    assert len(bank_build.SECTIONS) == 7
    spans = []
    for name in bank_build.SECTIONS:
        base, rec = lay.section_offset(name), lay.section_record(name)
        spans.append((base, base + lay.num_experts * rec))
    spans.sort()
    assert spans[0][0] == 0
    assert spans[-1][1] == lay.layer_bytes
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start, "sections must be adjacent with no padding"


def test_expert_offsets_are_disjoint_and_ordered():
    lay = TINY.check()
    for name in bank_build.SECTIONS:
        rec = lay.section_record(name)
        offs = [lay.expert_offset(name, e) for e in range(lay.num_experts)]
        assert offs == sorted(offs)
        for a, b in zip(offs, offs[1:]):
            assert b - a == rec
    with pytest.raises(IndexError):
        lay.expert_offset(bank_build.W13_SECTIONS[0], lay.num_experts)


def test_each_w13_section_is_a_whole_number_of_pages():
    """The premise of per-part page placement: parts must never share a page."""
    lay = bank_build.BankLayout().check()
    page = os.sysconf("SC_PAGE_SIZE")
    for name in bank_build.W13_SECTIONS:
        assert lay.section_record(name) % page == 0
    # w2 is stored whole-expert, but TP splits its k, and in NZ gidx is outermost, so
    # part 0 is the contiguous prefix -- which must also be page-aligned.
    assert lay.section_record("w2") % page == 0
    assert (lay.section_record("w2") // lay.tp_count) % page == 0


def test_part_of_section_follows_the_device_row_order():
    lay = TINY.check()
    assert [lay.part_of_section(n) for n in bank_build.W13_SECTIONS] == [0, 1, 0, 1]
    with pytest.raises(KeyError):
        lay.part_of_section("w2")


def test_check_rejects_layouts_the_guarantees_do_not_hold_for():
    with pytest.raises(bank_build.BankBuildError, match="multiple of C0"):
        bank_build.BankLayout(hidden=5120 + 2).check()
    # group_size // 2 must be exactly one C0, otherwise a 16-byte NZ block stops being
    # one k-group sharing one scale byte -- the coincidence the scheme rests on.
    with pytest.raises(bank_build.BankBuildError, match="no longer be one k-group"):
        bank_build.BankLayout(group_size=64).check()
    with pytest.raises(bank_build.BankBuildError, match="not divisible by tp_count"):
        bank_build.BankLayout(intermediate=2305, tp_count=2).check()
    # A section record that is not a whole number of pages would make the two TP parts
    # share pages, which is exactly the thing the split layout exists to avoid.
    with pytest.raises(bank_build.BankBuildError, match="could not be placed per node"):
        bank_build.BankLayout(num_experts=3, hidden=64, intermediate=32, tp_count=2).check()


# ------------------------------------------------------------------------- contents


def test_each_section_is_its_own_tp_half_permuted():
    lay = TINY.check()
    src = FakeSource(lay)
    sections = bank_build.expert_sections(src, 3, 1, lay)
    per = lay.intermediate // lay.tp_count

    for name, proj, half in bank_build.W13_PARTS:
        nd = src.read(src.locate(3, 1, proj, "weight")).numpy().reshape(lay.intermediate, lay.w13_kb)
        rows = nd[half * per : (half + 1) * per].reshape(-1)
        want = nz_geometry.permute_nd_to_nz(rows, 1, lay.w13_part_n, lay.w13_kb)
        assert np.array_equal(sections[name], want), name
        # A permutation moves bytes; it must not lose or invent any.
        assert sorted(sections[name].tolist()) == sorted(rows.tolist())
        # ...and it must not be the identity, which would mean 'permute' became a copy.
        assert not np.array_equal(sections[name], rows)

    w2 = src.read(src.locate(3, 1, "w2", "weight")).numpy()
    assert np.array_equal(sections["w2"], nz_geometry.permute_nd_to_nz(w2, 1, lay.w2_n, lay.w2_kb))


def test_scales_reach_the_bank_verbatim():
    """The production hook only re-views the scales, so their bytes must not change."""
    lay = TINY.check()
    src = FakeSource(lay)
    sections = bank_build.expert_sections(src, 3, 1, lay)
    s1 = src.read(src.locate(3, 1, "w1", "scale")).numpy()
    s3 = src.read(src.locate(3, 1, "w3", "scale")).numpy()
    assert np.array_equal(sections["w13_scale"], np.concatenate([s1, s3]))
    assert np.array_equal(sections["w2_scale"], src.read(src.locate(3, 1, "w2", "scale")).numpy())


# ------------------------------------------------- the equivalence that justifies A


def test_segments_reassemble_the_device_whole_expert_w13():
    """Four per-part sections + the segment table == the whole-expert NZ permutation.

    This is the load-bearing test of the split layout. Storing w13 per TP part is only
    admissible if the staging copies put it back together into exactly the bytes the
    device would have seen from a whole-expert bank. Anything wrong in the segment
    table -- a lane offset, a slab stride, the part-major loop order -- shows up here.
    """
    lay = TINY.check()
    src = FakeSource(lay)
    sections = bank_build.expert_sections(src, 5, 2, lay)

    # What a whole-expert bank would hold: permute (w1 stacked on w3) at N = 2*inter.
    w1 = src.read(src.locate(5, 2, "w1", "weight")).numpy().reshape(-1)
    w3 = src.read(src.locate(5, 2, "w3", "weight")).numpy().reshape(-1)
    want = nz_geometry.permute_nd_to_nz(np.concatenate([w1, w3]), 1, lay.w13_rows_total, lay.w13_kb)
    assert want.nbytes == lay.device_record("w13")

    # Run the segment table for slot 0 of a staging buffer holding only w13.
    staging = np.zeros(lay.device_record("w13"), dtype=np.uint8)
    base = {"w13": 0, "w2": 0, "w13_scale": 0, "w2_scale": 0}
    moved = 0
    for seg in bank_build.expert_segments(lay, 2, 0, base):
        if seg.section not in bank_build.W13_SECTIONS:
            continue
        srcbuf = sections[seg.section]
        rel = seg.src_offset - lay.expert_offset(seg.section, 2)
        staging[seg.dst_offset : seg.dst_offset + seg.nbytes] = srcbuf[rel : rel + seg.nbytes]
        moved += seg.nbytes

    assert moved == want.nbytes, "the segments must cover the device tensor exactly once"
    assert np.array_equal(staging, want)


def test_segment_table_shape_and_order():
    lay = bank_build.BankLayout().check()
    base = {t: 0 for t in bank_build.DEVICE_TENSORS}
    segs = list(bank_build.expert_segments(lay, 7, 0, base))

    # 160 slabs x 4 sections for w13, plus one run each for w2 and the two scales.
    assert len(segs) == 4 * lay.w13_gidx_count + 3 == 643
    stats = bank_build.segment_stats(segs)
    assert stats["bytes"] == lay.bytes_per_expert
    assert stats["min_bytes"] == lay.w13_part_run_bytes == 18_432

    # part-major: each section's slabs run consecutively, sources strictly ascending.
    w13 = [s for s in segs if s.section in bank_build.W13_SECTIONS]
    order = [s.section for s in w13]
    assert order == [n for n in bank_build.W13_SECTIONS for _ in range(lay.w13_gidx_count)]
    for name in bank_build.W13_SECTIONS:
        offs = [s.src_offset for s in w13 if s.section == name]
        assert offs == sorted(offs)
        for a, b in zip(offs, offs[1:]):
            assert b - a == lay.w13_part_run_bytes, "within a section the source must be sequential"


def test_chunk_segments_index_the_staging_buffer_by_slot_not_expert_id():
    lay = bank_build.BankLayout().check()
    base = {t: 0 for t in bank_build.DEVICE_TENSORS}
    experts = (301, 7, 42)  # deliberately non-contiguous and unsorted
    segs = bank_build.chunk_segments(lay, experts, base)

    assert len(segs) == len(experts) * 643
    # Destinations must tile the three slots of each device tensor, with no overlap and
    # nothing past the end of the chunk's staging buffer.
    for tensor in bank_build.DEVICE_TENSORS:
        names = bank_build.W13_SECTIONS if tensor == "w13" else (tensor,)
        rec = lay.device_record(tensor)
        touched = np.zeros(len(experts) * rec, dtype=bool)
        for seg in segs:
            if seg.section in names:
                lo, hi = seg.dst_offset, seg.dst_offset + seg.nbytes
                assert hi <= touched.size
                assert not touched[lo:hi].any(), f"{tensor}: destination written twice"
                touched[lo:hi] = True
        assert touched.all(), f"{tensor}: destination not fully covered"

    # Sources must come from the given expert ids, not from slots.
    srcs = {s.src_offset for s in segs if s.section == "w2"}
    assert srcs == {lay.expert_offset("w2", e) for e in experts}


def test_chunk_segments_rejects_an_incomplete_destination_map():
    lay = TINY.check()
    with pytest.raises(bank_build.BankBuildError, match="missing device tensors"):
        bank_build.chunk_segments(lay, [0], {"w13": 0})


# --------------------------------------------------------------------- build/verify


def test_build_layer_then_verify_round_trips(tmp_path):
    lay = TINY.check()
    src = FakeSource(lay)
    out = str(tmp_path / "layer000.nzbank")
    written = bank_build.build_layer(src, 0, out, lay, drop_cache=False)
    assert written == lay.layer_bytes
    assert os.path.getsize(out) == lay.layer_bytes
    assert not os.path.exists(out + ".partial"), "the partial file must be renamed, not left behind"

    report = bank_build.verify_layer(src, 0, str(tmp_path), lay, experts=range(lay.num_experts))
    assert report["bytes_compared"] == lay.layer_bytes


def test_open_layer_is_read_only_shared(tmp_path):
    """kt throws on a PRIVATE+WRITABLE mapping, so the loader must hand it ``r--s``."""
    lay = TINY.check()
    bank_build.build_layer(FakeSource(lay), 0, str(tmp_path / "layer000.nzbank"), lay, drop_cache=False)
    mm, size = bank_build.open_layer(str(tmp_path), 0)
    try:
        assert size == lay.layer_bytes
        with pytest.raises((TypeError, ValueError)):
            mm[0] = 0
    finally:
        mm.close()


def test_verify_layer_catches_a_single_flipped_byte(tmp_path):
    lay = TINY.check()
    src = FakeSource(lay)
    path = tmp_path / "layer000.nzbank"
    bank_build.build_layer(src, 0, str(path), lay, drop_cache=False)

    target = lay.expert_offset("w2", 1) + 7
    with open(path, "r+b") as fh:
        fh.seek(target)
        original = fh.read(1)
        fh.seek(target)
        fh.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(bank_build.BankBuildError, match=f"bank offset {target}"):
        bank_build.verify_layer(src, 0, str(tmp_path), lay, experts=[1])


def test_verify_layer_catches_two_experts_written_on_top_of_each_other(tmp_path):
    """The failure mode the section arithmetic can actually produce."""
    lay = TINY.check()
    src = FakeSource(lay)
    path = tmp_path / "layer000.nzbank"
    bank_build.build_layer(src, 0, str(path), lay, drop_cache=False)

    sec = bank_build.W13_SECTIONS[0]
    rec = lay.section_record(sec)
    with open(path, "r+b") as fh:
        fh.seek(lay.expert_offset(sec, 0))
        payload = fh.read(rec)
        fh.seek(lay.expert_offset(sec, 1))
        fh.write(payload)
    with pytest.raises(bank_build.BankBuildError, match=sec):
        bank_build.verify_layer(src, 0, str(tmp_path), lay, experts=[1])


def test_verify_layer_rejects_a_truncated_file(tmp_path):
    lay = TINY.check()
    src = FakeSource(lay)
    path = tmp_path / "layer000.nzbank"
    bank_build.build_layer(src, 0, str(path), lay, drop_cache=False)
    with open(path, "r+b") as fh:
        fh.truncate(lay.layer_bytes - 1)
    with pytest.raises(bank_build.BankBuildError, match="layout says"):
        bank_build.verify_layer(src, 0, str(tmp_path), lay)


def test_build_writes_a_manifest_that_pins_the_layout_and_the_contract(tmp_path):
    lay = TINY.check()
    man = bank_build.build(FakeSource(lay), str(tmp_path), layers=[0, 2], layout=lay, drop_cache=False)
    assert man["format"] == bank_build.BANK_FORMAT
    assert set(man["layers"]) == {"0", "2"}
    assert man["derived"]["bytes_per_expert"] == lay.bytes_per_expert
    assert set(man["derived"]["sections"]) == set(bank_build.SECTIONS)
    # The loader contract is part of the frozen format, not just documentation.
    assert man["loader_contract"]["min_threads"] == bank_build.MIN_LOADER_THREADS
    assert bank_build.load_manifest(str(tmp_path)) == man

    # A bank written by a different format revision must be refused, not read.
    with open(bank_build.manifest_path(str(tmp_path)), encoding="utf-8") as fh:
        stale = json.load(fh)
    stale["format"] = "kt-nz-bank/1"
    with open(bank_build.manifest_path(str(tmp_path)), "w", encoding="utf-8") as fh:
        json.dump(stale, fh)
    with pytest.raises(bank_build.BankBuildError, match="bank format"):
        bank_build.load_manifest(str(tmp_path))


def test_build_refuses_when_the_filesystem_has_no_room(tmp_path, monkeypatch):
    lay = bank_build.BankLayout().check()  # 6.724 GiB per layer

    class TinyFs:
        f_bavail = 1
        f_frsize = 4096

    monkeypatch.setattr(bank_build.os, "statvfs", lambda _p: TinyFs())
    with pytest.raises(bank_build.BankBuildError, match="available, need"):
        bank_build.build(FakeSource(lay), str(tmp_path), layers=[0], layout=lay)


# ------------------------------------------------------------------ loader contract


def test_loader_placement_accepts_one_thread_per_l3_domain(monkeypatch):
    monkeypatch.setattr(bank_build, "l3_domain_of", lambda cpu: (cpu // 8) * 8)
    got = bank_build.check_loader_placement([0, 8, 16, 24])
    assert got["threads"] == 4
    assert got["l3_domains"] == [0, 8, 16, 24]


def test_loader_placement_rejects_too_few_threads(monkeypatch):
    monkeypatch.setattr(bank_build, "l3_domain_of", lambda cpu: (cpu // 8) * 8)
    with pytest.raises(bank_build.LoaderPlacementError, match="at least 4 threads"):
        bank_build.check_loader_placement([0, 8])


def test_loader_placement_rejects_threads_packed_into_one_ccd(monkeypatch):
    """The failure that costs 2.5x on the host leg and is invisible to a correctness test."""
    monkeypatch.setattr(bank_build, "l3_domain_of", lambda cpu: (cpu // 8) * 8)
    with pytest.raises(bank_build.LoaderPlacementError, match="L3/CCD domain"):
        bank_build.check_loader_placement([0, 1, 2, 3])


def test_loader_placement_rejects_duplicate_cpus(monkeypatch):
    monkeypatch.setattr(bank_build, "l3_domain_of", lambda cpu: (cpu // 8) * 8)
    with pytest.raises(bank_build.LoaderPlacementError, match="duplicates"):
        bank_build.check_loader_placement([0, 8, 8, 16])


def test_one_per_l3_picks_distinct_domains(monkeypatch):
    monkeypatch.setattr(bank_build, "l3_domain_of", lambda cpu: (cpu // 8) * 8)
    cpus = bank_build.one_per_l3_cpus(4, candidates=range(64))
    assert cpus == [0, 8, 16, 24]
    assert bank_build.check_loader_placement(cpus)["l3_domains"] == [0, 8, 16, 24]
    with pytest.raises(bank_build.LoaderPlacementError, match="distinct L3/CCD domains"):
        bank_build.one_per_l3_cpus(4, candidates=range(8))


def test_l3_domain_of_reads_the_real_topology():
    """Not monkeypatched: the real sysfs must be parseable, or the guard is useless."""
    doms = {bank_build.l3_domain_of(c) for c in range(min(32, os.cpu_count() or 1))}
    assert len(doms) >= 1
    assert all(isinstance(d, int) for d in doms)


# ------------------------------------------------- decode prefetch placement (kt busy)
#
# A different contract from the prefill loader's, because it applies in a different
# phase. Prefill staging runs with the CPU MoE idle, so spreading across L3 domains is
# what matters there. The decode prefetch runs with kt at full tilt, where the L3
# spread stopped mattering (one/two/four CCDs within 3.4 points, inside the noise) and
# two other effects dominated: SMT siblings of kt's cores -59%, sharing a physical core
# with a kt worker -57%, against ~-23% for idle cores on any CCD.


def _fake_topology(monkeypatch, *, cores_per_ccd=8, logical=128):
    """8-core CCDs and SMT pairs cpu N / cpu N+logical, like the real host."""
    monkeypatch.setattr(bank_build, "l3_domain_of", lambda cpu: ((cpu % logical) // cores_per_ccd) * cores_per_ccd)
    monkeypatch.setattr(
        bank_build, "smt_siblings_of", lambda cpu: frozenset({cpu % logical, cpu % logical + logical})
    )


def test_prefetch_placement_accepts_idle_cores_on_one_ccd(monkeypatch):
    """Distinct CCDs are deliberately NOT required here -- they measured inside noise."""
    _fake_topology(monkeypatch)
    got = bank_build.check_prefetch_placement([48, 49, 50, 51], kt_cpus=range(0, 48))
    assert got["threads"] == 4
    assert got["l3_domains"] == [48]
    assert got["physical_cores"] == [48, 49, 50, 51]


def test_prefetch_placement_rejects_smt_siblings_of_kt_workers(monkeypatch):
    """The single worst placement measured: -59% on kt."""
    _fake_topology(monkeypatch)
    with pytest.raises(bank_build.LoaderPlacementError, match="SMT sibling of kt worker"):
        bank_build.check_prefetch_placement([128, 129, 130, 131], kt_cpus=[0, 1, 2, 3])


def test_prefetch_placement_rejects_sharing_cores_with_kt(monkeypatch):
    _fake_topology(monkeypatch)
    with pytest.raises(bank_build.LoaderPlacementError, match="also kt worker cpus"):
        bank_build.check_prefetch_placement([0, 8, 16, 24], kt_cpus=[0, 8, 16, 24])


def test_prefetch_placement_rejects_siblings_inside_the_pool(monkeypatch):
    _fake_topology(monkeypatch)
    with pytest.raises(bank_build.LoaderPlacementError, match="are SMT siblings"):
        bank_build.check_prefetch_placement([48, 176, 49, 50])


def test_prefetch_placement_caps_the_thread_count(monkeypatch):
    """8 threads took 9 more points off kt than 4, and 4 is already 3.7x what is needed."""
    _fake_topology(monkeypatch)
    with pytest.raises(bank_build.LoaderPlacementError, match="more than 4"):
        bank_build.check_prefetch_placement([48, 49, 50, 51, 52], kt_cpus=[0])
    with pytest.raises(bank_build.LoaderPlacementError, match="duplicates"):
        bank_build.check_prefetch_placement([48, 48, 49, 50])
    with pytest.raises(bank_build.LoaderPlacementError, match="pool is empty"):
        bank_build.check_prefetch_placement([])


def test_prefetch_placement_without_kt_cpus_still_checks_the_pool(monkeypatch):
    _fake_topology(monkeypatch)
    assert bank_build.check_prefetch_placement([6, 14, 22, 30])["threads"] == 4


def test_smt_siblings_reads_the_real_topology():
    """Not monkeypatched: a guard built on unparseable sysfs is no guard."""
    sibs = bank_build.smt_siblings_of(0)
    assert 0 in sibs
    assert all(isinstance(c, int) for c in sibs)
