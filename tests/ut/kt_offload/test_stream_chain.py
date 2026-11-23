# SPDX-License-Identifier: Apache-2.0
"""The W4A8 MXFP4 chunk chain, judged where a CPU can judge it: the arguments.

There is no MXFP4 grouped matmul on a host, so nothing here says the chain
computes the right values -- that is an on-card question. What it does pin is the
half whose failures are argument-shaped and which on-card returns a plausible
tensor anyway: ``group_list_type`` paired with the wrong encoding, weights read
off the resident layer instead of the chunk's slot, a scale paired with the wrong
projection. A fake ``torch_npu`` / ``torch.ops._C_ascend`` records every kernel
call, and the criteria are the ARGUMENTS and the ORDER.

The fake is a faithful *bookkeeping* model, not a numeric one: init routing
really permutes and counts per slot, un-permute really applies the router
weights, and the two GEMMs return ones, so a chunk's output is the sum of the
router weights that reached it and the chunk decomposition is checkable without
any kernel. Each criterion carries a mutation arm.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch_npu
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.kt_offload import stream_chain
from vllm_ascend.kt_offload.stream_chain import (
    SLOT_WEIGHT_ATTRS,
    ChainCoverage,
    ChunkRun,
    ChunkSlot,
    assert_group_list_pairing,
    build_chunk_experts_input,
    build_chunk_layer,
    dispatch_chunk,
    group_list_type_for,
    run_chunk,
    run_streamed_layer,
)
from vllm_ascend.kt_offload.stream_prefill import ChunkTables, build_chunk_plan, chunk_tables
from vllm_ascend.ops.fused_moe.moe_utils import enable_fusion_gmmswigluquant
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDSDynamicFusedMoEMethod

# Small but shaped like V4.1: hidden/intermediate are multiples of the MXFP group
# size (32), so the scale layouts production reshapes are legal.
HIDDEN, INTER, GROUP = 64, 64, 32
N_EXPERTS, TOPK, TOKENS = 16, 4, 8
SWIGLU_LIMIT = 10.0  # what the model serves with; 0.0 would mean "no clamp"
RESIDENT_SLOTS = 16  # what a slot mis-sized "like the resident layer" would be


@pytest.fixture(scope="module", autouse=True)
def ascend_runtime():
    """The config the production builders read: ``build_mlp_compute_input`` asks
    ``enable_fusion_gmmswigluquant()`` for the fusion flag, and leaving it unset
    would test a path production never takes. Torn down (and the lru_cache
    cleared) so the rest of the suite sees the process it saw before."""
    clear_ascend_config()
    cfg = VllmConfig()
    ctx = set_current_vllm_config(cfg)
    ctx.__enter__()
    init_ascend_config(cfg)
    enable_fusion_gmmswigluquant.cache_clear()
    yield cfg
    enable_fusion_gmmswigluquant.cache_clear()
    ctx.__exit__(None, None, None)
    clear_ascend_config()


def quant_method():
    """The real DS W4A8 MXFP method; __init__ only reads group_size/dynamic_eplb."""
    qm = AscendW4A8MXFPDSDynamicFusedMoEMethod.__new__(AscendW4A8MXFPDSDynamicFusedMoEMethod)
    qm.group_size = GROUP
    qm.dynamic_eplb = False
    return qm


def moe_config():
    return SimpleNamespace(
        experts_per_token=TOPK,
        num_experts=N_EXPERTS,
        num_logical_experts=N_EXPERTS,
        num_local_experts=RESIDENT_SLOTS,
        activation=MoEActivation.SILU,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        swiglu_limit=SWIGLU_LIMIT,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )


def make_slot(num_slots: int, tag: float = 1.0) -> ChunkSlot:
    """Slot buffers with the shapes the production method itself declares."""
    qm = quant_method()
    shapes = qm.get_weight(num_slots, INTER, HIDDEN, torch.bfloat16)
    shapes.update(qm.get_dynamic_quant_param(num_slots, INTER, HIDDEN, torch.bfloat16))
    made = {name: torch.full(tuple(t.shape), int(tag) % 251 + 1, dtype=t.dtype) for name, t in shapes.items()}
    return ChunkSlot(**{name: made[name] for name in SLOT_WEIGHT_ATTRS})


def resident_layer(num_slots: int = RESIDENT_SLOTS) -> torch.nn.Module:
    """A stand-in for the layer that holds the resident experts."""
    mod = torch.nn.Module()
    slot = make_slot(num_slots, tag=7.0)
    for name, t in slot.tensors().items():
        object.__setattr__(mod, name, t)
    mod.local_num_experts = num_slots
    return mod


def routing(tokens: int = TOKENS, seed: int = 0, num_experts: int = N_EXPERTS):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(num_experts, generator=g)[:TOPK] for _ in range(tokens)])
    w = torch.rand(tokens, TOPK, generator=g, dtype=torch.float32) + 0.1
    return ids.to(torch.int32), w


def hidden(tokens: int = TOKENS, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 100)
    return torch.randn(tokens, HIDDEN, generator=g).to(torch.bfloat16)


# The NPU boundary, modelled.


@dataclass
class Call:
    name: str
    args: tuple
    kwargs: dict

    def arg(self, i: int) -> Any:
        return self.args[i]


class NpuTape:
    """Records every kernel call, keeping BOTH the object and a snapshot.

    The object is what identity checks need ("slot buffer or resident
    parameter?"). The snapshot is what value checks need: production's gmm1
    calls ``dispose_tensor`` on its input, emptying the very tensor the tape
    holds.
    """

    def __init__(self):
        self.calls: list[Call] = []

    def record(self, name, args, kwargs) -> Call:
        def snap(v):
            if isinstance(v, torch.Tensor):
                return v.detach().clone()
            if isinstance(v, (list, tuple)):
                return type(v)(snap(x) for x in v)
            return v

        call = Call(name=name, args=tuple(args), kwargs=dict(kwargs))
        call.snapshot = {k: snap(v) for k, v in kwargs.items()}  # type: ignore[attr-defined]
        call.arg_snapshot = tuple(snap(a) for a in args)  # type: ignore[attr-defined]
        self.calls.append(call)
        return call

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.calls]

    def one(self, name: str, occurrence: int = 0) -> Call:
        hits = [c for c in self.calls if c.name == name]
        assert len(hits) > occurrence, f"{name} was called {len(hits)} times, wanted #{occurrence}"
        return hits[occurrence]


def _fake_init_routing_v2(tape):
    def fake(hidden_states, topk_ids, **kw):
        tape.record("npu_moe_init_routing_v2", (hidden_states, topk_ids), kw)
        tokens, top_k = topk_ids.shape
        flat = topk_ids.reshape(-1).long()
        order = torch.argsort(flat, stable=True)
        permuted = hidden_states.reshape(tokens, -1)[torch.div(order, top_k, rounding_mode="floor")]
        lo, hi = kw["active_expert_range"]
        counts = torch.bincount(flat, minlength=max(hi, int(flat.max()) + 1))[lo:hi].to(torch.int32)
        scale = None
        if kw["quant_mode"] == 3:
            # The permuted activations come back quantized with a per-token MX
            # scale. E8M0 has no CPU dtype, so the scale is a fp32 stand-in --
            # only its SHAPE travels through the production reshape.
            permuted = permuted.to(torch.float8_e4m3fn)
            scale = torch.ones(permuted.shape[0], HIDDEN // GROUP, dtype=torch.float32)
        return permuted, order.to(torch.int32), counts, scale

    return fake


def _fake_grouped_matmul(tape):
    def fake(**kw):
        tape.record("npu_grouped_matmul", (), kw)
        rows = kw["x"][0].shape[0]
        weight = kw["weight"][0]
        # gmm1 produces gate+up (2I), gmm2 produces hidden (H). Told apart by the
        # weight that was handed in, which is the thing under test anyway.
        width = 2 * INTER if weight.shape[1] == 2 * INTER else HIDDEN
        return [torch.ones(rows, width, dtype=torch.bfloat16)]

    return fake


def _fake_swiglu_group_quant(tape):
    def fake(hidden_states, **kw):
        tape.record("npu_swiglu_group_quant", (hidden_states,), kw)
        rows = hidden_states.shape[0]
        out = hidden_states[:, :INTER].to(torch.float8_e4m3fn)
        return out, torch.ones(rows, INTER // GROUP, dtype=torch.float32), None

    return fake


def _fake_unpermute(tape):
    def fake(permuted_tokens, sorted_indices, probs, **kw):
        tape.record(
            "npu_moe_token_unpermute",
            (),
            {"permuted_tokens": permuted_tokens, "sorted_indices": sorted_indices, "probs": probs, **kw},
        )
        tokens, top_k = probs.shape
        idx = sorted_indices.long()
        rows = torch.div(idx, top_k, rounding_mode="floor")
        weight = probs.reshape(-1)[idx].unsqueeze(1).float()
        out = torch.zeros(tokens, permuted_tokens.shape[1], dtype=torch.float32)
        out.index_add_(0, rows, permuted_tokens.float() * weight)
        return out.to(torch.bfloat16)

    return fake


@pytest.fixture
def tape(monkeypatch):
    t = NpuTape()
    monkeypatch.setattr(torch_npu, "npu_moe_init_routing_v2", _fake_init_routing_v2(t), raising=False)
    monkeypatch.setattr(torch_npu, "npu_grouped_matmul", _fake_grouped_matmul(t), raising=False)
    monkeypatch.setattr(torch_npu, "npu_moe_token_unpermute", _fake_unpermute(t), raising=False)
    monkeypatch.setattr(
        torch.ops, "_C_ascend", SimpleNamespace(npu_swiglu_group_quant=_fake_swiglu_group_quant(t)), raising=False
    )
    return t


EXPECTED_ORDER = [
    "npu_moe_init_routing_v2",
    "npu_grouped_matmul",  # gmm1 (fused with the activation quant below)
    "npu_swiglu_group_quant",
    "npu_grouped_matmul",  # gmm2
    "npu_moe_token_unpermute",
]


# The oracle: what each kernel must have been given, derived independently.


def expected_counts(chunk_ids: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Per-slot route counts, derived from the rewritten routing, not from the
    chain. Out-of-chunk routes parked on slot 0 count too: they are really
    permuted and computed, and the combine is what multiplies them away."""
    return torch.bincount(chunk_ids.reshape(-1).long(), minlength=num_slots).to(torch.int64)


def check_chain_args(
    tape: NpuTape,
    *,
    slot: ChunkSlot,
    num_slots: int,
    tokens: int,
    top_k: int,
    chunk_ids: torch.Tensor,
    chunk_weights: torch.Tensor,
    swiglu_limit: float = SWIGLU_LIMIT,
) -> list[str]:
    """Violations of the argument contract; ``[]`` means it holds."""
    bad: list[str] = []
    if tape.names != EXPECTED_ORDER:
        return [f"call order {tape.names} != {EXPECTED_ORDER}"]

    counts = expected_counts(chunk_ids, num_slots)

    ir = tape.one("npu_moe_init_routing_v2")
    k = ir.kwargs
    if k["expert_num"] != num_slots:
        bad.append(
            f"init routing expert_num {k['expert_num']} != the chunk's {num_slots} "
            "(taken from the resident slot count?)"
        )
    if list(k["active_expert_range"]) != [0, num_slots]:
        bad.append(f"init routing active_expert_range {k['active_expert_range']} != [0, {num_slots}]")
    if k["expert_tokens_num_type"] != 1:
        bad.append(f"init routing expert_tokens_num_type {k['expert_tokens_num_type']} != 1 (counts)")
    if k["expert_tokens_num_flag"] is not True:
        bad.append("init routing expert_tokens_num_flag is not True: no group list would come back")
    if k["quant_mode"] != 3:
        bad.append(f"init routing quant_mode {k['quant_mode']} != 3 (MXFP, no incoming dynamic scale)")
    if k["active_num"] != tokens * top_k:
        bad.append(f"init routing active_num {k['active_num']} != tokens*top_k {tokens * top_k}")
    if k["scale"] is not None:
        bad.append("init routing was given a prepare-stage scale; the chunk path has no prepare stage")
    if not torch.equal(ir.arg_snapshot[1], chunk_ids):
        bad.append("init routing did not get the chunk's rewritten topk_ids")

    g1 = tape.one("npu_grouped_matmul", 0)
    if g1.kwargs["weight"][0] is not slot.w13_weight:
        bad.append("gmm1 weight is not the chunk slot's w13_weight")
    if g1.kwargs["antiquant_scale"][0] is not slot.w13_weight_scale:
        bad.append("gmm1 antiquant_scale is not the chunk slot's w13_weight_scale")
    if g1.kwargs.get("scale") is not None:
        bad.append("gmm1 got a dequant `scale`; W4A8 MXFP passes the block scale as antiquant_scale")
    if not torch.equal(g1.snapshot["group_list"], counts.cumsum(dim=0)):
        bad.append(
            f"gmm1 group_list {g1.snapshot['group_list'].tolist()} is not the cumulative form of "
            f"the per-expert counts {counts.tolist()} (production converts with cumsum_group_list)"
        )
    if g1.kwargs.get("group_list_type", 0) != 0:
        bad.append(f"gmm1 group_list_type {g1.kwargs.get('group_list_type')} != 0; its group_list is cumulative")
    if g1.kwargs["x_dtype"] is not torch.float8_e4m3fn:
        bad.append(f"gmm1 x_dtype {g1.kwargs['x_dtype']} != float8_e4m3fn")
    if g1.kwargs["weight_dtype"] != torch_npu.float4_e2m1fn_x2:
        bad.append(f"gmm1 weight_dtype {g1.kwargs['weight_dtype']} != float4_e2m1fn_x2")
    if g1.kwargs["per_token_scale_dtype"] != torch_npu.float8_e8m0fnu:
        bad.append("gmm1 per_token_scale_dtype != float8_e8m0fnu")
    if g1.kwargs["split_item"] != 2 or g1.kwargs["group_type"] != 0:
        bad.append(f"gmm1 split_item/group_type {g1.kwargs['split_item']}/{g1.kwargs['group_type']} != 2/0")
    if g1.kwargs["output_dtype"] is not torch.bfloat16:
        bad.append(f"gmm1 output_dtype {g1.kwargs['output_dtype']} != bfloat16")
    pts = g1.snapshot["per_token_scale"][0]
    if pts.shape != (int(counts.sum()), HIDDEN // GROUP // 2, 2):
        bad.append(
            f"gmm1 per_token_scale shape {tuple(pts.shape)} is not the normalized MX layout "
            f"({int(counts.sum())}, {HIDDEN // GROUP // 2}, 2)"
        )

    sq = tape.one("npu_swiglu_group_quant")
    if sq.kwargs["quant_mode"] != 2:
        bad.append(f"swiglu quant_mode {sq.kwargs['quant_mode']} != 2")
    if sq.kwargs["dst_type"] is not torch.float8_e4m3fn:
        bad.append(f"swiglu dst_type {sq.kwargs['dst_type']} != float8_e4m3fn")
    if float(sq.kwargs["clamp_value"]) != float(swiglu_limit):
        bad.append(
            f"swiglu clamp_value {sq.kwargs['clamp_value']} != the layer's swiglu_limit {swiglu_limit} "
            "(moe_config not threaded through?)"
        )

    g2 = tape.one("npu_grouped_matmul", 1)
    if g2.kwargs["weight"][0] is not slot.w2_weight:
        bad.append("gmm2 weight is not the chunk slot's w2_weight")
    if g2.kwargs["antiquant_scale"][0] is not slot.w2_weight_scale:
        bad.append("gmm2 antiquant_scale is not the chunk slot's w2_weight_scale")
    if g2.kwargs["group_list_type"] != 1:
        bad.append(
            f"gmm2 group_list_type {g2.kwargs['group_list_type']} != 1 while its group_list is counts: "
            "the operator accepts this pairing and returns wrong numbers"
        )
    if not torch.equal(g2.snapshot["group_list"], counts):
        bad.append(f"gmm2 group_list {g2.snapshot['group_list'].tolist()} != the per-expert counts {counts.tolist()}")
    if g2.kwargs["x_dtype"] is not torch.float8_e4m3fn or g2.kwargs["weight_dtype"] != torch_npu.float4_e2m1fn_x2:
        bad.append("gmm2 x_dtype/weight_dtype are not the W4A8 MXFP pair")

    up = tape.one("npu_moe_token_unpermute")
    if not torch.equal(up.snapshot["probs"], chunk_weights.float()):
        bad.append("combine did not get the chunk's router weights (out-of-chunk routes not zeroed?)")
    if up.snapshot["sorted_indices"].numel() != tokens * top_k:
        bad.append(f"combine got {up.snapshot['sorted_indices'].numel()} indices, expected {tokens * top_k}")
    return bad


def run_one_chunk(
    tape,
    *,
    plan_chunk: int = 0,
    chunk_experts: int = 4,
    tables: ChunkTables | None = None,
    slot: ChunkSlot | None = None,
    seed: int = 0,
    **kw,
):
    """One chunk through the production chain; returns (run, slot, tables, ids, weights)."""
    plan = build_chunk_plan(N_EXPERTS, chunk_experts)
    tables = tables if tables is not None else chunk_tables(plan, plan_chunk)
    slot = slot if slot is not None else make_slot(len(tables.expert_ids), tag=3.0)
    ids, w = routing(seed=seed)
    x = hidden(seed=seed)
    run = run_chunk(
        quant_method=quant_method(),
        slot=slot,
        tables=tables,
        hidden_states=x,
        topk_ids=ids,
        topk_weights=w,
        moe_config=moe_config(),
        top_k=TOPK,
        **kw,
    )
    return run, slot, tables, ids, w


def chunk_routing(tables: ChunkTables, ids: torch.Tensor, w: torch.Tensor):
    from vllm_ascend.kt_offload.stream_prefill import split_routing_for_chunk

    r = split_routing_for_chunk(ids, w, tables)
    return r.topk_ids, r.topk_weights


def test_chain_calls_the_production_kernels_in_order(tape):
    run_one_chunk(tape)
    assert tape.names == EXPECTED_ORDER


def test_chain_arguments_are_pinned(tape):
    """The positive control for every mutation arm below: unmutated must be clean."""
    run, slot, tables, ids, w = run_one_chunk(tape)
    cid, cw = chunk_routing(tables, ids, w)
    assert (
        check_chain_args(
            tape,
            slot=slot,
            num_slots=len(tables.expert_ids),
            tokens=TOKENS,
            top_k=TOPK,
            chunk_ids=cid,
            chunk_weights=cw,
        )
        == []
    )
    assert run.group_list_type == 1
    assert run.tokens == TOKENS and run.live_routes > 0


@pytest.mark.parametrize("chunk_experts", [1, 2, 4, 16])
def test_group_count_comes_from_the_chunk_not_the_resident_slot_count(tape, chunk_experts):
    """``expert_num`` / ``active_expert_range`` decide how many groups the GEMM
    sees, so borrowing the resident ``num_local_experts`` would pad the group
    list with experts the slot does not hold."""
    _, slot, tables, ids, w = run_one_chunk(tape, chunk_experts=chunk_experts)
    k = tape.one("npu_moe_init_routing_v2").kwargs
    assert k["expert_num"] == len(tables.expert_ids) <= RESIDENT_SLOTS
    assert list(k["active_expert_range"]) == [0, len(tables.expert_ids)]
    if chunk_experts < RESIDENT_SLOTS:
        assert k["expert_num"] != RESIDENT_SLOTS


def test_gemms_read_the_chunk_slot_not_the_resident_layer(tape):
    resident = resident_layer()
    run_one_chunk(tape, resident_layer=resident)
    for call, name in (
        (tape.one("npu_grouped_matmul", 0), "w13_weight"),
        (tape.one("npu_grouped_matmul", 1), "w2_weight"),
    ):
        got = call.kwargs["weight"][0]
        assert got is not getattr(resident, name)
        assert got.data_ptr() != getattr(resident, name).data_ptr()


def test_group_list_encoding_is_derived_once():
    assert group_list_type_for(1) == 1
    assert group_list_type_for(0) == 0
    with pytest.raises(ValueError):
        group_list_type_for(2)


def test_group_list_pairing_guard_is_not_dead(tape):
    """Positive control: a mismatched pair must be refused before the GEMMs run."""
    plan = build_chunk_plan(N_EXPERTS, 4)
    tables = chunk_tables(plan, 0)
    slot = make_slot(len(tables.expert_ids))
    ids, w = routing()
    layer = build_chunk_layer(slot, expert_ids=tables.expert_ids)
    fei = build_chunk_experts_input(
        quant_method=quant_method(),
        layer=layer,
        hidden_states=hidden(),
        topk_weights=w,
        topk_ids=ids,
        activation=MoEActivation.SILU,
    )
    d = dispatch_chunk(fei, num_slots=slot.num_slots, top_k=TOPK)
    assert_group_list_pairing(d)  # the honest one passes
    with pytest.raises(ValueError, match="group_list_type"):
        assert_group_list_pairing(replace(d, group_list_type=0))


def test_shim_layer_redirects_every_weight_the_production_method_reads(tape):
    """Drift guard, read from the production source rather than a list kept next
    to it: a method that grows ``layer.w13_bias`` would otherwise take it from
    whatever object the shim happens to be."""
    src = inspect.getsource(AscendW4A8MXFPDSDynamicFusedMoEMethod.__mro__[1])
    tree = ast.parse(src)
    read: set[str] = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in (
            "apply_gmm1_act_quant",
            "apply_gmm1",
            "apply_act_quant",
            "apply_gmm2",
        ):
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "layer":
                    read.add(node.attr)
    assert read, "found no layer.<attr> reads at all -- the scan is looking at the wrong object"
    assert read <= set(SLOT_WEIGHT_ATTRS), (
        f"the MLP stage reads {sorted(read - set(SLOT_WEIGHT_ATTRS))} off the layer, which the chunk "
        "shim does not redirect: those would come from the resident layer"
    )


def test_slot_must_hold_exactly_the_chunk_experts():
    slot = make_slot(RESIDENT_SLOTS)
    with pytest.raises(ValueError, match="chunk"):
        build_chunk_layer(slot, expert_ids=(0, 1, 2, 3))


def test_slot_that_is_the_resident_storage_is_refused():
    resident = resident_layer()
    borrowed = ChunkSlot(**{name: getattr(resident, name) for name in SLOT_WEIGHT_ATTRS})
    with pytest.raises(ValueError, match="resident"):
        build_chunk_layer(borrowed, expert_ids=tuple(range(RESIDENT_SLOTS)), resident_layer=resident)


def test_moe_config_is_required(tape):
    """Without it the MLP stage silently runs with swiglu_limit=0, i.e. no clamp."""
    ids, w = routing()
    with pytest.raises(ValueError, match="moe_config"):
        run_chunk(
            quant_method=quant_method(),
            slot=make_slot(4),
            tables=chunk_tables(build_chunk_plan(N_EXPERTS, 4), 0),
            hidden_states=hidden(),
            topk_ids=ids,
            topk_weights=w,
            moe_config=None,
            top_k=TOPK,
        )


def test_expert_map_is_refused(tape):
    """Two remaps would be applied to the same ids; the chunk table is already one."""
    plan = build_chunk_plan(N_EXPERTS, 4)
    tables = chunk_tables(plan, 0)
    slot = make_slot(len(tables.expert_ids))
    layer = build_chunk_layer(slot, expert_ids=tables.expert_ids)
    ids, w = routing()
    fei = build_chunk_experts_input(
        quant_method=quant_method(),
        layer=layer,
        hidden_states=hidden(),
        topk_weights=w,
        topk_ids=ids,
        activation=MoEActivation.SILU,
    )
    fei = replace(fei, routing=replace(fei.routing, expert_map=torch.arange(N_EXPERTS)))
    with pytest.raises(ValueError, match="expert_map"):
        dispatch_chunk(fei, num_slots=slot.num_slots, top_k=TOPK)


def stream_layer(tape, *, chunk_experts: int = 4, seed: int = 0, capture_sizes=(1, 2, 4), **kw):
    plan = build_chunk_plan(N_EXPERTS, chunk_experts)
    slots = {}

    def slot_for(index, expert_ids):
        slots[index] = make_slot(len(expert_ids), tag=index + 1)
        return slots[index]

    ids, w = routing(seed=seed)
    res = run_streamed_layer(
        quant_method=quant_method(),
        plan=plan,
        slot_for=slot_for,
        expected_expert_ids=range(N_EXPERTS),
        hidden_states=hidden(seed=seed),
        topk_ids=ids,
        topk_weights=w,
        moe_config=moe_config(),
        top_k=TOPK,
        capture_sizes=list(capture_sizes),
        min_tokens=TOKENS,
        **kw,
    )
    return res, plan, slots, ids, w


def test_every_chunk_runs_the_whole_chain(tape):
    res, plan, _, _, _ = stream_layer(tape)
    assert tape.names == EXPECTED_ORDER * plan.num_chunks
    assert res.coverage.chunks_run == plan.num_chunks


def test_chunk_sum_reproduces_every_route(tape):
    """With the GEMMs returning ones, each chunk contributes its own router
    weights, so the sum over chunks must equal the row sum of the ORIGINAL
    routing: a dropped, doubled or rescaled route shows up as a wrong total."""
    res, plan, _, ids, w = stream_layer(tape)
    want = w.sum(dim=1)
    got = res.output.float().sum(dim=1) / HIDDEN
    # The only tolerance in this file, and it is a property of the MODEL, not of
    # the chain: combine and the chunk sum round through bf16 (relative spacing
    # 2**-8) once per chunk.
    tol = 2 * plan.num_chunks * 2**-8
    assert torch.allclose(got, want, rtol=tol, atol=0.0), f"{got.tolist()} != {want.tolist()}"


def test_coverage_is_reported(tape):
    res, plan, _, _, _ = stream_layer(tape)
    cov = res.coverage
    assert cov.chunks_run == plan.num_chunks
    assert cov.experts_planned == set(range(N_EXPERTS))
    assert cov.experts_hit and cov.experts_hit <= cov.experts_planned
    assert cov.tokens == TOKENS
    assert cov.live_routes == TOKENS * TOPK
    assert cov.violations(plan) == []
    text = cov.summary()
    assert f"chunks={plan.num_chunks}" in text and f"tokens={TOKENS}" in text and "experts=" in text


def test_eager_only_guard_runs_before_any_kernel(tape):
    with pytest.raises(ValueError, match="cudagraph_capture_sizes"):
        stream_layer(tape, capture_sizes=(1, 2, 4, TOKENS))  # this forward IS a captured size
    assert tape.calls == [], "a kernel ran before the eager-only guard refused the forward"


def test_streaming_below_the_threshold_is_refused(tape):
    plan = build_chunk_plan(N_EXPERTS, 4)
    ids, w = routing()
    with pytest.raises(ValueError, match="threshold"):
        run_streamed_layer(
            quant_method=quant_method(),
            plan=plan,
            expected_expert_ids=range(N_EXPERTS),
            slot_for=lambda i, e: make_slot(len(e)),
            hidden_states=hidden(),
            topk_ids=ids,
            topk_weights=w,
            moe_config=moe_config(),
            top_k=TOPK,
            capture_sizes=None,
            min_tokens=512,
        )
    assert tape.calls == []


def test_zero_coverage_raises():
    with pytest.raises(ValueError, match="zero coverage"):
        ChainCoverage().check()


def test_coverage_without_a_live_route_raises(tape):
    """Every route zeroed = the chain ran and computed nothing. Not a pass."""
    plan = build_chunk_plan(N_EXPERTS, N_EXPERTS)
    ids, w = routing()
    with pytest.raises(ValueError, match="no live route"):
        run_streamed_layer(
            quant_method=quant_method(),
            plan=plan,
            expected_expert_ids=range(N_EXPERTS),
            slot_for=lambda i, e: make_slot(len(e)),
            hidden_states=hidden(),
            topk_ids=ids,
            topk_weights=torch.zeros_like(w),
            moe_config=moe_config(),
            top_k=TOPK,
            capture_sizes=None,
            min_tokens=TOKENS,
        )


def test_strict_false_reports_instead_of_raising():
    cov = ChainCoverage()
    bad = cov.check(strict=False)
    assert bad and any("zero coverage" in b for b in bad)
    assert "chunks=0" in cov.summary()


def test_coverage_names_a_chunk_no_route_reached(tape):
    """A chunk nobody routed to is legal but must be visible, never silent."""
    cov = ChainCoverage()
    cov.add(
        ChunkRun(
            chunk_index=0,
            expert_ids=(0, 1),
            output=torch.zeros(1),
            group_list=torch.zeros(2),
            group_list_type=1,
            tokens=TOKENS,
            live_routes=3,
            experts_hit=(0,),
        )
    )
    cov.add(
        ChunkRun(
            chunk_index=1,
            expert_ids=(2, 3),
            output=torch.zeros(1),
            group_list=torch.zeros(2),
            group_list_type=1,
            tokens=TOKENS,
            live_routes=0,
            experts_hit=(),
        )
    )
    assert cov.chunks_without_routes == [1]
    assert "cold_experts" in cov.summary() and "empty_chunks" in cov.summary()
    assert cov.violations() == []  # legal: reported, not failed


# Mutation arms. Each must make the argument oracle red.


def test_mut_group_list_type_counts_declared_cumulative_is_red(tape, monkeypatch):
    """1 -> 0 at the single place the pairing is decided. The internal guard stays
    quiet (both ends moved together), which is why the criterion has to be the
    arguments and not the guard."""
    monkeypatch.setattr(stream_chain, "group_list_type_for", lambda _t: 0)
    run, slot, tables, ids, w = run_one_chunk(tape)
    cid, cw = chunk_routing(tables, ids, w)
    bad = check_chain_args(
        tape, slot=slot, num_slots=len(tables.expert_ids), tokens=TOKENS, top_k=TOPK, chunk_ids=cid, chunk_weights=cw
    )
    assert bad and any("group_list" in b for b in bad), bad


def test_mut_weights_taken_from_the_resident_layer_is_red(tape):
    """Same shapes, wrong bytes: only identity separates them."""
    resident = resident_layer(num_slots=4)
    borrowed = ChunkSlot(**{name: getattr(resident, name) for name in SLOT_WEIGHT_ATTRS})
    chunk_slot = make_slot(4, tag=3.0)
    run, _, tables, ids, w = run_one_chunk(tape, slot=borrowed)  # no resident_layer guard given
    cid, cw = chunk_routing(tables, ids, w)
    bad = check_chain_args(
        tape,
        slot=chunk_slot,
        num_slots=len(tables.expert_ids),
        tokens=TOKENS,
        top_k=TOPK,
        chunk_ids=cid,
        chunk_weights=cw,
    )
    assert bad and any("slot's w13_weight" in b or "slot's w2_weight" in b for b in bad), bad


def test_mut_keep_table_not_applied_is_red(tape):
    """Out-of-chunk routes left alive: they land on the dummy slot with their weight."""
    plan = build_chunk_plan(N_EXPERTS, 4)
    honest = chunk_tables(plan, 0)
    leaky = ChunkTables(
        chunk_index=honest.chunk_index,
        expert_ids=honest.expert_ids,
        safe_slot_table=honest.safe_slot_table,
        keep_table=torch.ones_like(honest.keep_table),
    )
    run, slot, _, ids, w = run_one_chunk(tape, tables=leaky)
    cid, cw = chunk_routing(honest, ids, w)  # judged against the honest split
    bad = check_chain_args(
        tape, slot=slot, num_slots=len(honest.expert_ids), tokens=TOKENS, top_k=TOPK, chunk_ids=cid, chunk_weights=cw
    )
    assert bad and any("router weights" in b for b in bad), bad


def test_mut_scales_paired_with_the_wrong_projection_is_red(tape):
    """w13's block scale handed to gmm2 and w2's to gmm1: plausible shapes, wrong numbers."""
    good = make_slot(4, tag=3.0)
    swapped = ChunkSlot(
        w13_weight=good.w13_weight,
        w13_weight_scale=good.w2_weight_scale,
        w2_weight=good.w2_weight,
        w2_weight_scale=good.w13_weight_scale,
    )
    run, _, tables, ids, w = run_one_chunk(tape, slot=swapped)
    cid, cw = chunk_routing(tables, ids, w)
    bad = check_chain_args(
        tape, slot=good, num_slots=len(tables.expert_ids), tokens=TOKENS, top_k=TOPK, chunk_ids=cid, chunk_weights=cw
    )
    assert bad and any("antiquant_scale" in b for b in bad), bad


def test_mut_slot_sized_by_the_resident_count_is_red(tape):
    """A slot sized by the resident count: 16 experts for a 4-expert chunk."""
    with pytest.raises(ValueError, match="chunk"):
        run_one_chunk(tape, slot=make_slot(RESIDENT_SLOTS))
    assert tape.calls == []


def test_mut_a_dropped_chunk_is_red(tape):
    """Coverage is judged against the plan, so a chunk that never ran shows."""
    res, plan, _, _, _ = stream_layer(tape)
    cov = res.coverage
    cov.chunks_run -= 1
    cov.experts_planned -= set(plan.chunks[-1])
    assert cov.violations(plan) != []
    with pytest.raises(ValueError):
        cov.check(plan)


# -- the chunk lookahead, driven through the loop --------------------------
#
# 🔴 Until 2026-11-21 NO test passed `prefetch_for` to `run_streamed_layer` at
# all: the production wiring of the single largest TTFT change in this work was
# untested. The tests below drive the loop itself, so the call order, the
# last-chunk guard and the "an optimisation may not fail the layer" contract are
# all covered by something that fails when they break.


def test_lookahead_is_asked_for_the_next_chunk_before_the_current_one_runs(tape):
    """One call per chunk boundary, each BEFORE the chunk it overlaps with."""
    order = []

    def prefetch_for(index, ids):
        order.append(("prefetch", index, tuple(ids)))
        return True

    plan = build_chunk_plan(N_EXPERTS, 4)
    slots = {}

    def slot_for(index, expert_ids):
        order.append(("slot_for", index, tuple(expert_ids)))
        slots[index] = make_slot(len(expert_ids), tag=index + 1)
        return slots[index]

    ids, w = routing(seed=0)
    run_streamed_layer(
        quant_method=quant_method(),
        plan=plan,
        slot_for=slot_for,
        prefetch_for=prefetch_for,
        expected_expert_ids=range(N_EXPERTS),
        hidden_states=hidden(seed=0),
        topk_ids=ids,
        topk_weights=w,
        moe_config=moe_config(),
        top_k=TOPK,
        capture_sizes=[1, 2, 4],
        min_tokens=TOKENS,
    )
    prefetched = [i for kind, i, _ in order if kind == "prefetch"]
    assert prefetched == list(range(1, plan.num_chunks)), (
        f"expected one lookahead for chunks 1..{plan.num_chunks - 1}, got {prefetched}"
    )
    # Each prefetch must come before the slot_for of the SAME chunk, or the
    # buffer it filled is not the one that chunk consumes.
    for k in range(1, plan.num_chunks):
        assert order.index(("prefetch", k, tuple(plan.chunks[k]))) < order.index(
            ("slot_for", k, tuple(plan.chunks[k]))
        ), f"chunk {k} was gathered before its lookahead started"


def test_lookahead_is_asked_with_the_ids_the_chunk_will_be_served_with(tape):
    """A different id tuple would key a different buffer: every prefetch a miss."""
    seen = {}

    def prefetch_for(index, ids):
        seen[index] = tuple(ids)
        return True

    plan = build_chunk_plan(N_EXPERTS, 4)

    def slot_for(index, expert_ids):
        if index in seen:
            assert seen[index] == tuple(expert_ids), (
                f"chunk {index}: lookahead asked for {seen[index]}, slot_for was given {tuple(expert_ids)}"
            )
        return make_slot(len(expert_ids), tag=index + 1)

    ids, w = routing(seed=0)
    run_streamed_layer(
        quant_method=quant_method(),
        plan=plan,
        slot_for=slot_for,
        prefetch_for=prefetch_for,
        expected_expert_ids=range(N_EXPERTS),
        hidden_states=hidden(seed=0),
        topk_ids=ids,
        topk_weights=w,
        moe_config=moe_config(),
        top_k=TOPK,
        capture_sizes=[1, 2, 4],
        min_tokens=TOKENS,
    )
    assert len(seen) == plan.num_chunks - 1


def test_no_lookahead_past_the_last_chunk(tape):
    """A single-chunk plan must not prefetch, and plan.chunks[k+1] must be in range."""
    calls = []
    plan = build_chunk_plan(N_EXPERTS, N_EXPERTS)  # one chunk
    assert plan.num_chunks == 1
    ids, w = routing(seed=0)
    run_streamed_layer(
        quant_method=quant_method(),
        plan=plan,
        slot_for=lambda i, e: make_slot(len(e), tag=i + 1),
        prefetch_for=lambda i, e: calls.append(i),
        expected_expert_ids=range(N_EXPERTS),
        hidden_states=hidden(seed=0),
        topk_ids=ids,
        topk_weights=w,
        moe_config=moe_config(),
        top_k=TOPK,
        capture_sizes=[1, 2, 4],
        min_tokens=TOKENS,
    )
    assert calls == [], f"prefetched past the end of a {plan.num_chunks}-chunk plan: {calls}"


def test_a_raising_lookahead_does_not_fail_the_layer(tape):
    """The loop's own contract: the lookahead is an optimisation, not a step.

    Before this was guarded, a prefetch that raised aborted a layer whose compute
    would have succeeded -- and `slot_for` raises for the same reasons a few lines
    later, so nothing is lost by continuing.
    """

    def prefetch_for(index, ids):
        raise RuntimeError("bank went away")

    plan = build_chunk_plan(N_EXPERTS, 4)
    ids, w = routing(seed=0)
    res = run_streamed_layer(
        quant_method=quant_method(),
        plan=plan,
        slot_for=lambda i, e: make_slot(len(e), tag=i + 1),
        prefetch_for=prefetch_for,
        expected_expert_ids=range(N_EXPERTS),
        hidden_states=hidden(seed=0),
        topk_ids=ids,
        topk_weights=w,
        moe_config=moe_config(),
        top_k=TOPK,
        capture_sizes=[1, 2, 4],
        min_tokens=TOKENS,
    )
    assert res is not None
    assert tape.names == EXPECTED_ORDER * plan.num_chunks, "a failed lookahead changed what the layer computed"
