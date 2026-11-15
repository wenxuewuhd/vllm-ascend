# SPDX-License-Identifier: Apache-2.0
"""Stage 1 (whole-layer NZ buffer): the parts that hold without a device.

The device-side claims -- that a per-expert cast equals its slice of a
whole-cast, that a byte write at ``base + e*PER`` hits exactly slot e, and that
the whole-layer form is bitwise equal to the per-chunk form -- were measured on
card and live in the harnesses, not here. What is testable off-card is the part
that decides whether the GEMM is handed the right routing and whether a missed
write can pass unnoticed.
"""
import pytest
import torch

from vllm_ascend.kt_offload.stream_layer import (
    LayerWriteLedger,
    StreamLayerError,
    layer_tables,
)
from vllm_ascend.kt_offload.stream_prefill import build_chunk_plan


def _plan(n=8, c=2):
    return build_chunk_plan(n, c)


def test_layer_tables_only_this_chunk_s_experts_can_receive_rows():
    """The property that costs money if it is wrong.

    This replaces an earlier test that asserted the slot table was the IDENTITY.
    That test encoded a bug: split_routing only zeroes the WEIGHT of an
    out-of-chunk route, it does not stop the row being dispatched to that
    expert's group, and dispatch_chunk passes expert_num = num_slots. So an
    identity table made all 384 groups non-empty and the GEMM read all 384
    experts' weights -- measured at +75% per chunk, +37 ms per layer, which was
    the whole of stage 1's TTFT regression. Empty groups are nearly free; a
    group that RECEIVES A ROW costs ~0.52 us per MB of that expert's weights.
    """
    plan = _plan(8, 2)
    for k, ids in enumerate(plan.chunks):
        t = layer_tables(plan, k)
        reached = set(t.safe_slot_table.tolist())
        assert reached <= set(ids), (
            f"chunk {k} can reach slots {sorted(reached - set(ids))} outside its own "
            f"{sorted(ids)}; every one of those is a non-empty group whose weights the GEMM reads")


def test_layer_tables_maps_this_chunk_s_experts_to_their_own_slot():
    # The buffer holds every expert, so for the chunk's OWN experts a logical id
    # IS its slot. Anything else would send a route to the wrong expert with a
    # plausible weight.
    plan = _plan(8, 2)
    for k, ids in enumerate(plan.chunks):
        t = layer_tables(plan, k)
        for e in ids:
            assert t.safe_slot_table[e].item() == e, (k, e)


def test_layer_tables_keep_mask_selects_only_this_chunk():
    plan = _plan(8, 2)
    for k, ids in enumerate(plan.chunks):
        t = layer_tables(plan, k)
        kept = [i for i, v in enumerate(t.keep_table.tolist()) if v != 0.0]
        assert kept == sorted(ids), (k, kept, ids)


def test_layer_tables_expert_ids_is_the_identity_list():
    # run_chunk names the experts a route reached by indexing expert_ids BY SLOT.
    # The chunk's own experts still sit at their own slot, so the list must stay
    # the identity or the coverage report blames the wrong experts.
    plan = _plan()
    t = layer_tables(plan, 0)
    assert t.expert_ids == tuple(range(plan.num_experts))


def test_layer_tables_dummy_is_a_slot_this_chunk_has_already_written():
    # A dropped route lands on the dummy with weight 0, and the dummy must hold
    # real weights (NaN * 0 is NaN). The whole-layer buffer still holds the
    # PREVIOUS layer's bytes in every slot this forward has not reached, so the
    # dummy has to be one of this chunk's own experts -- provider.slot_for(k)
    # writes those before run_chunk is called. Slot 0 would only be safe by the
    # luck of chunk 0 being written first.
    plan = _plan(8, 2)
    for k, ids in enumerate(plan.chunks):
        t = layer_tables(plan, k)
        outside = [e for e in range(plan.num_experts) if e not in ids]
        for e in outside:
            assert t.safe_slot_table[e].item() in ids, (k, e)
            assert t.keep_table[e].item() == 0.0, (k, e)


def test_layer_tables_rejects_an_out_of_range_chunk():
    plan = _plan()
    with pytest.raises(IndexError):
        layer_tables(plan, plan.num_chunks)


def test_layer_tables_honours_a_non_ascending_expert_order():
    # 1c: the chunk tiling may follow the checkpoint's physical order, which is
    # the ASCII order of the tensor names, not the numeric one. 12 experts, not
    # 8: below 10 the ASCII and numeric orders are the same list, so the premise
    # of this test was vacuous.
    order = [int(x) for x in sorted(str(i) for i in range(12))]
    assert order != sorted(order), "the order under test must actually be non-ascending"
    plan = build_chunk_plan(12, 3, expert_ids=order)
    t = layer_tables(plan, 0)
    ids = plan.chunks[0]
    # Not the identity: an expert this chunk did not bring has no slot of its own
    # to be routed to, so it is sent to a slot this forward HAS written (the
    # chunk's first), and the keep table zeroes its contribution.
    want = [e if e in ids else ids[0] for e in range(12)]
    assert t.safe_slot_table.tolist() == want
    kept = [i for i, v in enumerate(t.keep_table.tolist()) if v != 0.0]
    assert kept == sorted(ids)


def test_ledger_refuses_a_second_write_of_the_same_chunk():
    led = LayerWriteLedger()
    led.begin(3)
    led.note(0, 100)
    with pytest.raises(StreamLayerError, match="written twice"):
        led.note(0, 100)


def test_ledger_reports_a_chunk_that_was_never_written():
    plan = _plan(8, 2)          # 4 chunks
    led = LayerWriteLedger()
    led.begin(3)
    for k in (0, 1, 3):
        led.note(k, 10)
    bad = led.violations(plan)
    assert len(bad) == 1 and "[2]" in bad[0]


def test_ledger_is_clean_when_every_chunk_was_written():
    plan = _plan(8, 2)
    led = LayerWriteLedger()
    led.begin(0)
    for k in range(plan.num_chunks):
        led.note(k, 10)
    assert led.violations(plan) == []
    assert led.bytes_written == 10 * plan.num_chunks


def test_config_refuses_the_switch_without_stream_prefill():
    from vllm_ascend.kt_offload.config import KTOffloadConfig
    with pytest.raises(ValueError, match="stream_layer_buffer needs stream_prefill"):
        KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=8,
                        stream_layer_buffer=True)
