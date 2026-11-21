# SPDX-License-Identifier: Apache-2.0
"""Gathering only SOME slots of a chunk, for the resident experts' sake.

Why this exists (measure/PLAN_G_stream_only_offloaded.md): a 4096-token prefill
streams all 384 experts of all 40 layers -- 268.9 GiB, the whole bank -- and
~100 of those experts are already in HBM. Skipping them removes 26% of the bytes
from both the host read and the H2D. It has to be done WITHOUT changing which
experts land in which chunk, because the layer output is a sum over chunks
accumulated in fp32 and re-grouping it changes the rounding; so the chunk keeps
its 32 ids and only the PROVENANCE of some slots' bytes changes.

What these tests hold:
  * the filled slots get exactly the bytes a whole-chunk gather would give them;
  * the skipped slots are left strictly alone (the caller fills them from the
    resident parameter, and a gather that quietly zeroed them would feed the
    GEMM zeros with tokens routed there);
  * the coverage reports the slots PRODUCED, not the chunk -- otherwise a subset
    gather is indistinguishable from a whole one downstream;
  * an out-of-range or empty subset is refused.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_ascend.kt_offload import bank_build, bank_source

TINY = bank_build.BankLayout(num_experts=6, hidden=256, intermediate=64, group_size=32, tp_count=2)
SENTINEL = 0xAB


class FakeSource:
    """Deterministic ND bytes at the right sizes; stands in for WeightSource."""

    def __init__(self, layout):
        self.layout = layout

    def _nbytes(self, proj, kind):
        lay = self.layout
        if kind == "weight":
            return lay.w2_n * lay.w2_kb if proj == "w2" else lay.intermediate * lay.w13_kb
        return lay.section_record("w2_scale") if proj == "w2" else lay.section_record("w13_scale") // 2

    def locate(self, layer, expert, proj, kind="weight"):
        return (layer, expert, proj, kind)

    def read(self, span):
        layer, expert, proj, kind = span
        n = self._nbytes(proj, kind)
        # Content depends on every coordinate, so a slot filled from the wrong
        # expert cannot accidentally compare equal to the right one.
        seed = (layer * 1000 + expert) * 17 + (hash((proj, kind)) % 251)
        idx = np.arange(n, dtype=np.uint64)
        vals = ((idx + 1) * np.uint64(0x9E3779B97F4A7C15) >> np.uint64(24)) ^ np.uint64(seed)
        return torch.from_numpy((vals & np.uint64(0xFF)).astype(np.uint8))


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    d = tmp_path_factory.mktemp("subset_bank")
    bank_build.build(FakeSource(TINY), str(d), layers=[0], layout=TINY, drop_cache=False)
    return str(d)


@pytest.fixture
def src(bank):
    s = bank_source.BankSource(bank, num_experts=TINY.num_experts, copy_threads=1)
    yield s
    s.close()


def _staging(src, count):
    _, nbytes = src.dst_base(count)
    return torch.full((nbytes,), SENTINEL, dtype=torch.uint8)


def _slot_ranges(src, count, slot):
    """Byte ranges that belong to ``slot`` in the group-major staging."""
    base, _ = src.dst_base(count)
    out = []
    for name in ("w13", "w2", "w13_scale", "w2_scale"):
        rec = src.layout.device_record(name)
        out.append((base[name] + slot * rec, base[name] + (slot + 1) * rec))
    return out


def test_subset_fills_those_slots_with_the_same_bytes_and_nothing_else(src):
    ids = (0, 1, 2, 3)
    full = _staging(src, len(ids))
    src.gather(0, ids, full)
    # 🔴 NOT "no byte equals the sentinel": ~1 byte in 256 of real weight data is
    # 0xAB, so that assertion fails on correct output (it did: 26009 of 26112).
    # The check that means something is per REGION, below.
    for slot in range(len(ids)):
        for lo, hi in _slot_ranges(src, len(ids), slot):
            assert not torch.all(full[lo:hi] == SENTINEL), f"slot {slot} was never written"

    part = _staging(src, len(ids))
    res = src.gather(0, ids, part, only_slots=[0, 2])

    for slot in (0, 2):
        for lo, hi in _slot_ranges(src, len(ids), slot):
            assert torch.equal(part[lo:hi], full[lo:hi]), f"slot {slot} bytes differ from a full gather"
    for slot in (1, 3):
        for lo, hi in _slot_ranges(src, len(ids), slot):
            assert torch.all(part[lo:hi] == SENTINEL), f"slot {slot} was touched; it must be left alone"
    # Coverage is about what this call produced.
    assert (
        res.coverage.produced_bytes
        == sum(src.layout.device_record(t) for t in ("w13", "w2", "w13_scale", "w2_scale")) * 2
    )
    # The LAYOUT stays the whole chunk (it fixes the group bases); only
    # `produced` names what was filled.
    assert res.layout.experts == ids
    assert res.layout.produced == (0, 2) and res.layout.produced_slots() == (0, 2)


def test_every_single_slot_subset_agrees_with_the_whole(src):
    ids = (4, 5, 0, 1)
    full = _staging(src, len(ids))
    src.gather(0, ids, full)
    for slot in range(len(ids)):
        part = _staging(src, len(ids))
        src.gather(0, ids, part, only_slots=[slot])
        for lo, hi in _slot_ranges(src, len(ids), slot):
            assert torch.equal(part[lo:hi], full[lo:hi])
        # Every OTHER slot's regions are untouched. Counting non-sentinel bytes
        # instead would be wrong: real weight bytes hit 0xAB about once in 256.
        for other in range(len(ids)):
            if other == slot:
                continue
            for lo, hi in _slot_ranges(src, len(ids), other):
                assert torch.all(part[lo:hi] == SENTINEL), f"slot {other} was touched"


def test_multithreaded_subset_agrees_with_single_threaded(bank):
    ids = (0, 1, 2, 3)
    one = bank_source.BankSource(bank, num_experts=TINY.num_experts, copy_threads=1)
    try:
        a = _staging(one, len(ids))
        one.gather(0, ids, a, only_slots=[1, 3])
    finally:
        one.close()
    # 1 thread again rather than 4: the thread count picks cpus one per L3
    # domain, and a UT must not depend on the host's topology.
    two = bank_source.BankSource(bank, num_experts=TINY.num_experts, copy_threads=1)
    try:
        b = _staging(two, len(ids))
        two.gather(0, ids, b, only_slots=[3, 1])  # order must not matter
    finally:
        two.close()
    assert torch.equal(a, b)


@pytest.mark.parametrize("bad", [[0, 4], [-1], [99]])
def test_a_slot_index_outside_the_chunk_is_refused(src, bad):
    ids = (0, 1, 2, 3)
    st = _staging(src, len(ids))
    with pytest.raises(bank_source.BankSourceError, match="outside the"):
        src.gather(0, ids, st, only_slots=bad)


def test_an_empty_subset_is_refused(src):
    ids = (0, 1, 2, 3)
    st = _staging(src, len(ids))
    with pytest.raises(bank_source.BankSourceError, match="fills nothing"):
        src.gather(0, ids, st, only_slots=[])
