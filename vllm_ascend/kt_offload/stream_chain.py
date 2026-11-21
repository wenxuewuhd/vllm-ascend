# SPDX-License-Identifier: Apache-2.0
"""The W4A8 MXFP4 compute chain for ONE streamed chunk of experts.

``stream_prefill.py`` decides *which* experts travel together and rewrites the
routing for a chunk. This module is the other half: given a chunk's four weight
buffers already on the device, run ``dispatch -> gmm1+act+quant -> gmm2 ->
combine`` over them and return that chunk's contribution.

No kernel argument list is written out here: every weight the MLP stage touches
is read off ``mlp_compute_input.layer``, so this path builds a **shim layer whose
four weight attributes are the chunk's slot buffers** and calls the production
objects unchanged -- ``AscendW4A8MXFPDynamicFusedMoEMethod.apply_gmm1_act_quant``
/ ``apply_gmm2`` through ``apply_moe_mlp`` (which picks the fused-activation
branch in production too), ``build_fused_experts_input`` /
``build_mlp_compute_input`` (the payloads
``AscendW4A8MXFPDynamicFusedMoEMethod.apply`` builds), and
``DeviceOperator.npu_moe_init_routing`` / ``npu_moe_token_unpermute`` (the
device indirection ``TokenDispatcherWithAllGather`` calls).

The dispatcher is bypassed because ``AllGatherCommImpl`` is a process-level
singleton whose ``num_local_experts`` comes from the *last* layer that called
``setup_moe_comm_method`` -- ``num_slots``, the resident count -- while a chunk
has ``len(expert_ids)`` slots, and ``expert_num`` / ``active_expert_range``
decide how many groups the GEMM sees; routing a chunk through it would silently
pad or truncate the group list. Dispatch and combine are therefore done here,
mirroring ``TokenDispatcherWithAllGather.token_dispatch``
(``expert_tokens_num_type=1``, ``active_expert_range=[0, num_slots]``,
``quant_mode=3`` for MXFP with no incoming dynamic scale), while the MLP stage
stays the production one.

``group_list_type`` is one decision, not two. v2 init routing with
``expert_tokens_num_type=1`` emits per-expert COUNTS, which the GEMM must be told
with ``group_list_type=1``; ``expert_tokens_num_type=0`` (and v1 routing) emits
CUMULATIVE prefix sums, i.e. ``group_list_type=0``. The operator accepts either
pairing and **silently returns wrong numbers** for the wrong one -- no shape or
dtype error. So the value is derived from the routing call that produced the
group list, by :func:`group_list_type_for`, and checked by
:func:`assert_group_list_pairing` before the GEMMs run. "Just pass 1 everywhere"
is wrong because the production method is asymmetric: ``apply_gmm1_act_quant``
converts the list with ``cumsum_group_list(..., 0)`` and lets
``npu_grouped_matmul`` default to 0, while ``apply_gmm2`` passes it through.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.kt_offload.stream_prefill import (
    ChunkPlan,
    ChunkTables,
    assert_eager_only,
    check_chunk_plan,
    chunk_tables,
    split_routing_for_chunk,
)
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput, build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import build_mlp_compute_input
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoEAllGatherCombineMetadata, MoETokenDispatchOutput
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.ops.fused_moe.token_dispatcher import EXPERT_TOKEN_NUMS_TYPE_COUNT, EXPERT_TOKEN_NUMS_TYPE_CUMSUM
from vllm_ascend.quantization.quant_type import QuantType

__all__ = [
    "SLOT_WEIGHT_ATTRS",
    "ChunkSlot",
    "ChunkShimLayer",
    "ChunkDispatch",
    "ChunkRun",
    "ChainCoverage",
    "StreamChainResult",
    "group_list_type_for",
    "assert_group_list_pairing",
    "build_chunk_layer",
    "dispatch_chunk",
    "combine_chunk",
    "run_chunk",
    "run_streamed_layer",
]

#: The four attributes the W4A8 MXFP4 MLP stage reads off ``mlp_compute_input.layer``.
#: The shim layer redirects exactly these and nothing else; the UT re-derives this set
#: from the production source, so an upstream method that starts reading a fifth weight
#: fails there instead of silently using resident bytes.
SLOT_WEIGHT_ATTRS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


# ---------------------------------------------------------------------------
# The chunk's weights, and the layer object that points at them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSlot:
    """The four device buffers holding one chunk's experts, in slot order.

    Slot ``i`` holds ``expert_ids[i]`` of the chunk -- the order
    ``ChunkTables.safe_slot_table`` maps into. Plain tensors, not ``Parameter``s:
    the streaming buffer is allocated once and refilled per chunk, while the
    resident weights stay where they are.
    """

    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor

    @property
    def num_slots(self) -> int:
        return int(self.w13_weight.shape[0])

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in SLOT_WEIGHT_ATTRS}

    def violations(self, expect_slots: int | None = None) -> list[str]:
        """Structural problems with the slot buffers; ``[]`` means none.

        Deliberately shape-light: on-card these are post-``npu_format_cast``
        buffers whose inner dims are transposed and 4-bit packed, so only the
        expert dimension is meaningful in every layout -- and it is the one that
        matters, a buffer with the *resident* slot count instead of the chunk's
        being the dispatcher-singleton mistake this module exists to avoid.
        """
        bad: list[str] = []
        for name, t in self.tensors().items():
            if not isinstance(t, torch.Tensor):
                bad.append(f"{name} is {type(t).__name__}, not a tensor")
                continue
            # dim() is deliberately NOT pinned to 3: the post-load MXFP scales
            # are 4-D -- ``w4a8_mxfp4.process_weights_after_loading`` runs them
            # through ``reshape(g, n, k // 2, 2).transpose(-3, -2)``. Only dim 0
            # means the same thing in every layout; anything stricter rejects
            # the production layout.
            if t.dim() < 2:
                bad.append(f"{name} has {t.dim()} dims {tuple(t.shape)}, expected [experts, ...]")
        if bad:
            return bad
        rows = {name: int(t.shape[0]) for name, t in self.tensors().items()}
        if len(set(rows.values())) != 1:
            bad.append(f"slot buffers disagree on the expert dimension: {rows}")
        elif expect_slots is not None and self.num_slots != expect_slots:
            bad.append(
                f"slot buffers hold {self.num_slots} experts but the chunk has {expect_slots}: "
                "the chunk's group count (expert_num / active_expert_range) is taken from the "
                "chunk, not from the resident slot count"
            )
        devices = {name: str(t.device) for name, t in self.tensors().items()}
        if len(set(devices.values())) != 1:
            bad.append(f"slot buffers live on different devices: {devices}")
        return bad


class ChunkShimLayer(torch.nn.Module):
    """A layer object whose weight attributes are the chunk's slot buffers.

    It exists so the production quant method can be called *unchanged*: it reads
    ``layer.w13_weight`` & co, and here those names resolve to the streamed
    chunk. ``build_chunk_layer`` is the only place the redirection happens, so
    "which weights did the GEMM get" has exactly one answer to check.
    """

    def __init__(self, slot: ChunkSlot, expert_ids: Sequence[int]):
        super().__init__()
        for name, t in slot.tensors().items():
            # Plain attribute assignment on purpose: a Parameter/buffer would copy
            # or re-wrap, and the identity of the slot tensor is what the UT pins.
            object.__setattr__(self, name, t)
        self.slot = slot
        self.expert_ids = tuple(int(e) for e in expert_ids)
        self.local_num_experts = slot.num_slots

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return f"slots={self.slot.num_slots}, expert_ids={self.expert_ids}"


def build_chunk_layer(
    slot: ChunkSlot, *, expert_ids: Sequence[int], resident_layer: torch.nn.Module | None = None
) -> ChunkShimLayer:
    """Point the four weight names at ``slot`` after checking it really is one.

    ``resident_layer`` is optional but recommended: given it, this refuses a
    "slot" that is in fact the resident parameter (same storage). Streaming the
    resident weights is not a crash -- it quietly computes the resident layer's
    answer and reports full coverage.
    """
    ids = tuple(int(e) for e in expert_ids)
    bad = slot.violations(expect_slots=len(ids))
    if bad:
        raise ValueError(f"chunk slot is not usable: {'; '.join(bad)}")
    if resident_layer is not None:
        shared = []
        for name, t in slot.tensors().items():
            r = getattr(resident_layer, name, None)
            if isinstance(r, torch.Tensor) and (r is t or r.data_ptr() == t.data_ptr()):
                shared.append(name)
        if shared:
            raise ValueError(
                f"slot buffers {shared} are the resident layer's own storage. The streaming chain "
                "would then compute the resident layer's answer and still report coverage; the slot "
                "must be a separate buffer holding this chunk's experts."
            )
    return ChunkShimLayer(slot, ids)


# ---------------------------------------------------------------------------
# group_list encoding: one decision, derived once
# ---------------------------------------------------------------------------


def group_list_type_for(expert_tokens_num_type: int) -> int:
    """The ``group_list_type`` that goes with an init-routing ``expert_tokens_num_type``.

    ``EXPERT_TOKEN_NUMS_TYPE_COUNT`` (1) means v2 routing emits per-expert
    COUNTS, read by the GEMM as ``group_list_type=1``;
    ``EXPERT_TOKEN_NUMS_TYPE_CUMSUM`` (0, and v1 routing) emits CUMULATIVE
    prefix sums, i.e. 0. The numbers coincide, the meanings do not, and
    mispairing them is a silent wrong answer.
    """
    if expert_tokens_num_type == EXPERT_TOKEN_NUMS_TYPE_COUNT:
        return 1
    if expert_tokens_num_type == EXPERT_TOKEN_NUMS_TYPE_CUMSUM:
        return 0
    raise ValueError(
        f"unknown expert_tokens_num_type {expert_tokens_num_type}; "
        f"expected {EXPERT_TOKEN_NUMS_TYPE_COUNT} (counts) or "
        f"{EXPERT_TOKEN_NUMS_TYPE_CUMSUM} (cumulative)"
    )


def assert_group_list_pairing(dispatch: ChunkDispatch) -> None:
    """Refuse a dispatch whose group list and type flag disagree."""
    expected = group_list_type_for(dispatch.expert_tokens_num_type)
    if dispatch.group_list_type != expected:
        raise ValueError(
            f"group_list_type {dispatch.group_list_type} does not match the routing that produced the "
            f"group list (expert_tokens_num_type={dispatch.expert_tokens_num_type} => "
            f"{expected}). The grouped matmul accepts either pairing and silently returns wrong "
            "numbers for the wrong one."
        )


# ---------------------------------------------------------------------------
# Dispatch / combine, bypassing the process-level dispatcher singleton
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkDispatch:
    """What init routing produced for one chunk, plus the decisions behind it."""

    hidden_states: torch.Tensor
    expanded_row_idx: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    expert_tokens_num_type: int
    dynamic_scale: torch.Tensor | None
    topk_weights: torch.Tensor
    restore_shape: torch.Size
    num_slots: int
    quant_mode: int
    active_num: int

    def token_dispatch_output(self) -> MoETokenDispatchOutput[MoEAllGatherCombineMetadata]:
        """The production payload, so the MLP stage is built the production way."""
        return MoETokenDispatchOutput(
            hidden_states=self.hidden_states,
            group_list=self.group_list,
            group_list_type=self.group_list_type,
            dynamic_scale=self.dynamic_scale,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=self.topk_weights,
                expanded_row_idx=self.expanded_row_idx,
                restore_shape=self.restore_shape,
            ),
        )


def dispatch_chunk(
    fused_experts_input: MoEFusedExpertsInput,
    *,
    num_slots: int,
    top_k: int,
    expert_tokens_num_type: int = EXPERT_TOKEN_NUMS_TYPE_COUNT,
) -> ChunkDispatch:
    """Permute + quantize the tokens into this chunk's expert order.

    Mirrors ``TokenDispatcherWithAllGather.token_dispatch`` for the shape this
    path is in: no expert map (single card, no EP), no LoRA, no prepare-stage
    scale, expert range the *chunk's* ``[0, num_slots)``. Not mirrored: where
    ``num_local_experts`` comes from -- the whole point (module docstring).
    """
    quant = fused_experts_input.quant
    hidden_states = fused_experts_input.hidden_states
    topk_ids = fused_experts_input.topk_ids
    topk_weights = fused_experts_input.topk_weights
    if num_slots <= 0:
        raise ValueError(f"a chunk with {num_slots} slots has nothing to compute")
    if fused_experts_input.routing.expert_map is not None:
        raise ValueError(
            "stream_chain does not take an expert_map: the chunk's own slot order is "
            "already the mapping, and a second remap would be applied twice"
        )

    # A prepare stage would have handed us a scale; the chunk path has no prepare
    # stage, so init routing does the first quantization (MXFP => quant_mode 3).
    dynamic_scale = fused_experts_input.routing.pertoken_scale
    with_quant = quant.dispatch_with_quant and quant.quant_type != QuantType.W8A8FP
    if with_quant and dynamic_scale is None:
        quant_mode = 9 if quant.is_w4a4_mxfp else (3 if quant.is_mxfp else 1)
    else:
        quant_mode = -1
    act_quant_type = quant.mxfp.act_quant_type if quant.mxfp is not None else None

    num_tokens = hidden_states.shape[:-1].numel()
    active_num = num_tokens * top_k
    # Imported here, not at module scope: vllm_ascend.device.device_op pulls in
    # vllm_ascend.ops -> moe_mlp -> device_op again. At module scope that loop
    # only survives when something else has already finished initialising ops --
    # which the unit tests do via conftest and a plain script does not
    # ("partially initialized module").
    from vllm_ascend.device.device_op import DeviceOperator

    sorted_hidden_states, expanded_row_idx, expert_tokens, out_scale = DeviceOperator.npu_moe_init_routing(
        hidden_states,
        topk_ids,
        scale=dynamic_scale,
        active_num=active_num,
        expert_num=num_slots,
        expert_tokens_num_type=expert_tokens_num_type,
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_slots],
        quant_mode=quant_mode,
        act_quant_type=act_quant_type,
    )
    return ChunkDispatch(
        hidden_states=sorted_hidden_states,
        expanded_row_idx=expanded_row_idx,
        group_list=expert_tokens.to(torch.int64),
        group_list_type=group_list_type_for(expert_tokens_num_type),
        expert_tokens_num_type=expert_tokens_num_type,
        dynamic_scale=out_scale if with_quant else None,
        topk_weights=topk_weights,
        restore_shape=hidden_states.shape,
        num_slots=num_slots,
        quant_mode=quant_mode,
        active_num=active_num,
    )


def combine_chunk(mlp_output: torch.Tensor, dispatch: ChunkDispatch) -> torch.Tensor:
    """Un-permute and apply the router weights, as ``token_combine`` does.

    The router weights are the chunk's: routes outside it carry 0, so their
    (dummy-slot) rows contribute nothing. Kept in fp32 as production does --
    rounding to the activation dtype loses precision that accumulates across
    layers.
    """
    from vllm_ascend.device.device_op import DeviceOperator

    out = DeviceOperator.npu_moe_token_unpermute(
        permuted_tokens=mlp_output,
        sorted_indices=dispatch.expanded_row_idx,
        probs=dispatch.topk_weights.float(),
    )
    if len(dispatch.restore_shape) == 3:
        out = out.view(dispatch.restore_shape)
    return out


# ---------------------------------------------------------------------------
# One chunk, end to end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkRun:
    """One chunk's contribution plus what it is evidence of."""

    chunk_index: int
    expert_ids: tuple[int, ...]
    output: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    tokens: int
    live_routes: int
    experts_hit: tuple[int, ...]

    @property
    def cold_experts(self) -> tuple[int, ...]:
        hit = set(self.experts_hit)
        return tuple(e for e in self.expert_ids if e not in hit)


def build_chunk_experts_input(
    *,
    quant_method,
    layer: ChunkShimLayer,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation,
) -> MoEFusedExpertsInput:
    """The payload ``AscendW4A8MXFPDynamicFusedMoEMethod.apply`` builds.

    The MXFP dtypes come from the method itself (``quant_type``,
    ``act_quant_type``), so a method that changes its activation dtype changes
    this too.
    """
    return build_fused_experts_input(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        layer=layer,
        quant_type=quant_method.quant_type,
        dynamic_eplb=getattr(quant_method, "dynamic_eplb", False),
        expert_map=None,
        global_redundant_expert_num=0,
        mc2_mask=None,
        apply_router_weight_on_input=False,
        pertoken_scale=None,
        activation=activation,
        mxfp_act_quant_type=quant_method.act_quant_type,
        mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
        mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_use_bf16=(hidden_states.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
    )


# -- diagnostic chunk timing (default off) ----------------------------------
# Both runners -- run_streamed_layer (switch off) and run_streamed_layer_buffered
# (switch on) -- call run_chunk, so timing it here measures BOTH arms with the
# same code. Events are recorded per chunk but READ once per forward: reading an
# event forces a synchronise, and a per-chunk synchronise is one of the very
# things under suspicion, so measuring it that way would create what it measures.
_TIME_CHUNKS = False
_CHUNK_EVENTS: list = []
_CHUNK_MS: list = []
#: Sample 1 chunk in this many. Each live event is a runtime resource, and an
#: unbounded backlog risks exhausting the event pool mid-campaign; a sampled
#: distribution answers "is the ON arm's GEMM slower" just as well.
_TIME_EVERY = 4
_CHUNK_SEEN = 0
#: Drain once this many events are pending. At 1-in-4 sampling this is ~960
#: chunks of wall time, so the oldest events are long complete and the
#: synchronise inside drain is a no-op rather than a stall we would be measuring.
_DRAIN_AT = 240


def set_chunk_timing(on: bool) -> None:
    global _TIME_CHUNKS
    _TIME_CHUNKS = bool(on)


def drain_chunk_timing() -> list:
    """Resolve pending events and return the per-chunk milliseconds."""
    global _CHUNK_EVENTS, _CHUNK_MS
    if _CHUNK_EVENTS:
        torch.npu.synchronize()
        for a, b in _CHUNK_EVENTS:
            with contextlib.suppress(Exception):
                _CHUNK_MS.append(a.elapsed_time(b))
        _CHUNK_EVENTS = []
    out, _CHUNK_MS = _CHUNK_MS, []
    return out


def run_chunk(**kw) -> ChunkRun:
    """Timing shim around :func:`_run_chunk_impl`; identical when timing is off."""
    if not _TIME_CHUNKS:
        return _run_chunk_impl(**kw)
    global _CHUNK_SEEN
    _CHUNK_SEEN += 1
    if _CHUNK_SEEN % _TIME_EVERY:
        return _run_chunk_impl(**kw)
    a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    a.record()
    out = _run_chunk_impl(**kw)
    b.record()
    _CHUNK_EVENTS.append((a, b))
    # Drain only once the backlog is ~2 forwards old (480 chunks each). By then
    # the oldest events are long complete, so the synchronise inside drain is a
    # no-op rather than a stall we would then be measuring.
    if len(_CHUNK_EVENTS) >= _DRAIN_AT:
        ms = drain_chunk_timing()
        if ms:
            ms_sorted = sorted(ms)
            logger.info(
                "kt_offload chunk timing: n=%d  p50 %.3f ms  p90 %.3f  min %.3f  max %.3f  sum %.1f ms",
                len(ms_sorted),
                ms_sorted[len(ms_sorted) // 2],
                ms_sorted[int(len(ms_sorted) * 0.9)],
                ms_sorted[0],
                ms_sorted[-1],
                sum(ms_sorted),
            )
    return out


def _run_chunk_impl(
    *,
    quant_method,
    slot: ChunkSlot,
    tables: ChunkTables,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_config,
    top_k: int,
    resident_layer: torch.nn.Module | None = None,
    expert_tokens_num_type: int = EXPERT_TOKEN_NUMS_TYPE_COUNT,
) -> ChunkRun:
    """Compute one chunk's contribution to the layer output.

    ``moe_config`` is required: without it ``build_mlp_compute_input`` falls back
    to ``swiglu_limit=0``, silently turning the clamp off (V4.1 serves with
    10.0). ``top_k`` is the router's, not the chunk's -- it only sizes
    ``active_num``.
    """
    if moe_config is None:
        raise ValueError(
            "moe_config is required: without it the MLP stage would run with "
            "swiglu_limit=0 (no clamp) and the activation default, silently"
        )
    routing = split_routing_for_chunk(topk_ids, topk_weights, tables)
    layer = build_chunk_layer(slot, expert_ids=tables.expert_ids, resident_layer=resident_layer)
    fused_experts_input = build_chunk_experts_input(
        quant_method=quant_method,
        layer=layer,
        hidden_states=hidden_states,
        topk_weights=routing.topk_weights,
        topk_ids=routing.topk_ids,
        activation=moe_config.activation,
    )
    dispatch = dispatch_chunk(
        fused_experts_input, num_slots=slot.num_slots, top_k=top_k, expert_tokens_num_type=expert_tokens_num_type
    )
    assert_group_list_pairing(dispatch)

    mlp_compute_input = build_mlp_compute_input(
        fused_experts_input=fused_experts_input,
        token_dispatch_output=dispatch.token_dispatch_output(),
        moe_config=moe_config,
    )
    mlp_output, _before_gmm2_evt = apply_moe_mlp(mlp_compute_input, quant_method)
    output = combine_chunk(mlp_output, dispatch)

    # Coverage accounting, at the cost of ONE device->host transfer per chunk: a
    # per-slot count (chunk_experts ints) giving both the live-route total and
    # the experts actually reached. Deliberately not a per-route ``.tolist()``
    # (num_tokens*top_k ids across the seam every chunk), and deliberately not
    # skipped: without it, "streaming ran and computed nothing" is
    # indistinguishable from success.
    live = (routing.topk_weights != 0).reshape(-1)
    hits = torch.bincount(routing.topk_ids.long().reshape(-1)[live], minlength=slot.num_slots).cpu().tolist()
    hit = [tables.expert_ids[i] for i, n in enumerate(hits[: len(tables.expert_ids)]) if n]
    return ChunkRun(
        chunk_index=tables.chunk_index,
        expert_ids=tables.expert_ids,
        output=output,
        group_list=dispatch.group_list,
        group_list_type=dispatch.group_list_type,
        tokens=int(hidden_states.shape[:-1].numel()),
        live_routes=int(sum(hits)),
        experts_hit=tuple(hit),
    )


# ---------------------------------------------------------------------------
# Coverage: what actually streamed
# ---------------------------------------------------------------------------


@dataclass
class ChainCoverage:
    """How much of the layer really went through the streaming chain.

    Zero coverage is a failure, not a pass: a path that ran no chunk, or chunks
    no route reached, produces the same timing and output as "streaming is off".
    """

    chunks_run: int = 0
    tokens: int = 0
    live_routes: int = 0
    experts_planned: set[int] = field(default_factory=set)
    experts_hit: set[int] = field(default_factory=set)
    chunks_without_routes: list[int] = field(default_factory=list)

    def add(self, run: ChunkRun) -> None:
        self.chunks_run += 1
        self.tokens = max(self.tokens, run.tokens)
        self.live_routes += run.live_routes
        self.experts_planned.update(run.expert_ids)
        self.experts_hit.update(run.experts_hit)
        if run.live_routes == 0:
            self.chunks_without_routes.append(run.chunk_index)

    def summary(self) -> str:
        cold = sorted(self.experts_planned - self.experts_hit)
        return (
            f"stream_chain coverage: chunks={self.chunks_run} "
            f"experts={len(self.experts_hit)}/{len(self.experts_planned)} hit "
            f"tokens={self.tokens} live_routes={self.live_routes}"
            + (f" cold_experts={cold[:8]}{'...' if len(cold) > 8 else ''}" if cold else "")
            + (f" empty_chunks={self.chunks_without_routes[:8]}" if self.chunks_without_routes else "")
        )

    def violations(self, plan: ChunkPlan | None = None) -> list[str]:
        bad: list[str] = []
        if self.chunks_run == 0:
            bad.append("no chunk ran: zero coverage is a failure, not a pass")
        if self.tokens == 0:
            bad.append("no token reached the streaming chain")
        if self.live_routes == 0:
            bad.append(
                "no live route reached any chunk: every route was zeroed, so the chain "
                "computed nothing that can contribute to the output"
            )
        if not self.experts_hit:
            bad.append("no expert received a route")
        if plan is not None:
            want = set(plan.covered)
            if self.experts_planned != want:
                missing = sorted(want - self.experts_planned)
                extra = sorted(self.experts_planned - want)
                bad.append(
                    f"chunks run cover {len(self.experts_planned)} of the plan's {len(want)} "
                    f"experts (missing {missing[:8]}, extra {extra[:8]})"
                )
            if self.chunks_run != plan.num_chunks:
                bad.append(f"{self.chunks_run} chunk(s) ran but the plan has {plan.num_chunks}")
        if not self.experts_hit <= self.experts_planned:
            bad.append(f"experts hit but never planned: {sorted(self.experts_hit - self.experts_planned)[:8]}")
        return bad

    def check(self, plan: ChunkPlan | None = None, *, strict: bool = True) -> list[str]:
        """Return the violations; with ``strict`` (the default) raise on any."""
        bad = self.violations(plan)
        if bad and strict:
            raise ValueError(f"{self.summary()} -- " + "; ".join(bad))
        return bad


@dataclass(frozen=True)
class StreamChainResult:
    output: torch.Tensor
    coverage: ChainCoverage
    runs: tuple[ChunkRun, ...]


SCHEME_ATTRS = ("quant_type", "act_quant_type", "get_weight")


def run_streamed_layer(
    *,
    quant_method,
    plan: ChunkPlan,
    slot_for: Callable[[int, tuple[int, ...]], ChunkSlot],
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_config,
    top_k: int,
    capture_sizes: Sequence[int] | None,
    min_tokens: int,
    resident_layer: torch.nn.Module | None = None,
    accum_dtype: torch.dtype = torch.float32,
    expert_tokens_num_type: int = EXPERT_TOKEN_NUMS_TYPE_COUNT,
    expected_expert_ids: Sequence[int] | None = None,
    strict: bool = True,
    after_chunk: Callable[[int, tuple[int, ...]], None] | None = None,
    prefetch_for: Callable[[int, tuple[int, ...]], bool] | None = None,
) -> StreamChainResult:
    """Run every chunk of ``plan`` and sum the contributions.

    ``slot_for(chunk_index, expert_ids)`` hands back the buffers for that chunk;
    the weight I/O (host gather, H2D, format) lives behind it on purpose. The sum

    ``prefetch_for(chunk_index, expert_ids)``, when given, is asked to START the
    NEXT chunk's host gather before this chunk is computed. It must be
    non-blocking and may decline (returning False) when no buffer is free or the
    chunk needs no host bytes; the loop ignores the answer, because
    ``slot_for`` is correct either way -- it finds the buffer ready or fills it
    itself. Leaving it None is the pre-2026-11-18 behaviour: strictly serial,
    with the host idle during compute and the NPU idle during the gather.
    is accumulated in ``accum_dtype`` (fp32 by default) and cast back at the end.

    ``after_chunk(chunk_index, expert_ids)`` runs once per chunk, after that
    chunk's contribution is computed and before the next ``slot_for`` -- i.e.
    while the slot provider still holds this chunk's bytes. It is the folded
    swap's hook (``fold_swap.ResidentFold.after_chunk``), which reads the chunk
    and does not change what the chain computes.
    """
    # This chain reads the MoE *scheme* (quant_type, act_quant_type, and the
    # apply_* entry points), which production keeps INSIDE the layer's
    # quant_method (AscendFusedMoEMethod is a wrapper; the caller resolves it via
    # stream_io.moe_scheme). Refusing a wrong object here is loud: the wrapper
    # would otherwise die deep in the chain as an AttributeError with no hint
    # that the fix is one unwrap at the boundary.
    _missing = [a for a in SCHEME_ATTRS if not hasattr(quant_method, a)]
    if _missing:
        raise TypeError(
            f"run_streamed_layer needs the MoE scheme, not the wrapper: "
            f"{type(quant_method).__name__} is missing {_missing}. Resolve it at the call site with "
            f"stream_io.moe_scheme(quant_method, *stream_chain.SCHEME_ATTRS)."
        )

    # The coverage expectation must come from OUTSIDE the plan. Judging the plan
    # against itself passes a plan that is missing a chunk -- it is
    # self-consistent, just incomplete.
    if strict:
        if expected_expert_ids is None:
            raise ValueError(
                "run_streamed_layer(strict=True) needs expected_expert_ids: a coverage check whose "
                "expectation is the plan under test cannot fail. Pass the experts this layer is "
                "supposed to stream (e.g. stream_prefill.resident_chunk_ids(placement))."
            )
        bad = check_chunk_plan(plan, expected_expert_ids)
        if bad:
            raise ValueError("stream_chain: the chunk plan does not cover what was expected: " + "; ".join(bad))
    assert_eager_only(int(hidden_states.shape[:-1].numel()), capture_sizes, min_tokens, where="stream_chain")
    coverage = ChainCoverage()
    runs: list[ChunkRun] = []
    total: torch.Tensor | None = None
    for k in range(plan.num_chunks):
        tables = chunk_tables(plan, k, device=topk_ids.device, id_dtype=topk_ids.dtype, weight_dtype=topk_weights.dtype)
        slot = slot_for(k, tables.expert_ids)
        # 🔴 START THE NEXT CHUNK'S HOST GATHER BEFORE COMPUTING THIS ONE.
        # Without this the loop is strictly serial -- slot_for(k) blocks on the
        # gather, run_chunk(k) then has the NPU to itself while the host threads
        # idle, and `StreamSlotProvider.prefetch` (which exists and is tested)
        # had NO production caller at all. Measured 2026-11-18 on the shipped
        # path: staging is 4.02 s of a 15.47 s TTFT per bank pass, 88-91% of it
        # the host gather, and it lands on TTFT at a coefficient of 1.12, i.e.
        # none of it is hidden. Per chunk the gather is 8.4 ms against 23.9 ms
        # of compute, so at depth 1 the whole gather fits inside the compute.
        # Two pinned buffers already exist (DEFAULT_HOST_BUFFERS = 2, logged as
        # "1.121 GiB of pinned staging"), so this costs no memory -- the
        # ping-pong was built for exactly this and never switched on.
        if prefetch_for is not None and k + 1 < plan.num_chunks:
            # plan.chunks[k+1] is the ids directly; building k+1's tables here
            # would allocate device tensors a chunk early for nothing.
            prefetch_for(k + 1, plan.chunks[k + 1])
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
            expert_tokens_num_type=expert_tokens_num_type,
        )
        runs.append(run)
        coverage.add(run)
        contribution = run.output.to(accum_dtype)
        total = contribution if total is None else total + contribution
        if after_chunk is not None:
            after_chunk(k, tables.expert_ids)
    coverage.check(plan, strict=strict)
    if total is None:
        # Only reachable with strict=False; the caller asked for the violations
        # instead of an exception, and must not get a plausible-looking zero.
        raise ValueError(f"{coverage.summary()} -- no chunk produced an output tensor")
    return StreamChainResult(output=total.to(hidden_states.dtype), coverage=coverage, runs=tuple(runs))
