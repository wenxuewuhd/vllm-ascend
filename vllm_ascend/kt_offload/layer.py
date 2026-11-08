# SPDX-License-Identifier: Apache-2.0
"""Per-layer kt offload runtime and the hooks AscendRoutedExperts calls.

Hook contract (routed_experts.py); every call site is guarded by
``self.kt_offload is not None``, so with the feature off nothing here runs.

  __init__       ``create_layer_offload(...)`` before ``RoutedExperts.__init__``
                 (shrinks ``moe_config.num_local_experts`` to ``num_slots`` so the
                 quant method allocates only the resident rows), then
                 ``attach_after_init(layer)``.
  weight_loader  ``_map_global_expert_id_to_local_expert_id`` -> ``slot_of``;
                 CPU experts map to -1, which the upstream loader treats as
                 "not local" and skips (vllm routed_experts.py weight_loader).
  post-load      quant_method.process_weights_after_loading is wrapped: unused
                 slots are zeroed, then the original runs, then the routing
                 tables go to the device and the kt wrapper loads its experts.
  forward_impl   ``route(...)`` right after expert selection (logical ids),
                 ``merge(...)`` right after finalize.

Two execution paths, chosen per forward by whether the stream is being captured:

  eager    kt ``submit_forward`` / ``sync_forward`` (host blocks, device syncs).
  captured D2H into kt's pinned buffers, a ``_launch_host_func`` node that runs
           ``run_pinned_forward_sync`` at every replay, H2D back. Nothing on
           this path may depend on a runtime condition: whatever the capture
           run skips is absent from every replay.
"""

from __future__ import annotations

import inspect
import os
import re
import weakref
from time import CLOCK_REALTIME, clock_gettime_ns, perf_counter_ns
from typing import Any

import torch

# The ``vllm`` logger, not ``logging.getLogger(__name__)`` or
# ``init_logger(__name__)``: in the worker process the ``vllm_ascend`` logger
# has a file handler but never a level, so a ``vllm_ascend.*`` name loses INFO
# at the level gate and its WARNING/ERROR reach only the ascend log file.
from vllm.logger import logger

from vllm_ascend.kt_offload import (
    async_wait,
    device_signal,
    report_affinity,
    seam_timing,
    shared_window,
    side_stream,
    switches,
)
from vllm_ascend.kt_offload.placement import (
    LayerPlacement,
    NpuRouting,
    OffloadPlan,
    assert_routing_dtypes,
    host_ops_per_layer,
    packed_layout,
    plan_from_config,
    split_routing,
)
from vllm_ascend.kt_offload.resident_driver import SLOT_WEIGHT_ATTRS
from vllm_ascend.kt_offload.resident_swap import (
    LayerSwapTargets,
    kt_cpu_weight_ids_of,
    kt_mask_of,
    live_target_violations,
)

# One plan per process: placement is a whole-model decision made once.
_PLAN: OffloadPlan | None = None
_PLAN_KEY: tuple | None = None

# Captured graphs hold host-func nodes whose payloads are read at every
# replay; they must live for the whole process.
_CAPTURED_PAYLOADS: list = []
# layer_idx -> number of times its host-func node has run.
HOST_FUNC_CALLS: dict[int, int] = {}
# kt's pinned/device buffers for the capture sizes are process-wide
# (KExpertsCPUBuffer is class-level); they are registered and allocated once.
_CAPTURE_BUFFERS_READY: tuple | None = None
# One side stream for the whole process, created outside any capture.
_SIDE_STREAM = None
# A captured graph keeps record/wait nodes pointing at these event objects, so
# every pair stays alive for the life of the process.
_SIDE_EVENTS: list = []
# A6 guard: id() of every store the side stream touches -> the layer that owns
# it. Two layers on one store would be kt's ``layer_idx % KT_BUFFER_DEPTH``
# pinned-buffer sharing coming back. See side_stream.buffer_ownership_ok.
_SIDE_BUFFER_OWNER: dict[int, int] = {}
# P1a: the one completion thread of this process (async_wait.Completer), and
# the ExternalEvent objects the captured graph waits on. Like _SIDE_EVENTS,
# the events must outlive the capture: a graph node points at each one.
_COMPLETER = None
_EXT_EVENTS: list = []
# P1b: the process-wide parts of the device-signal path -- the polling thread,
# the pinned words the graph's 8-byte copies land in, the device step counter,
# and which layers are on this path (the lowest of them owns the one increment
# per replay). All created at weight-load time; none may be made in a capture.
_POLLER = None
_SIGNALS = None
_STEP_COUNTER = None
_SIGNAL_LAYERS: set[int] = set()


def _layer_index(layer_name: str) -> int:
    """Checkpoint layer number of ``...layers.<N>.ffn.experts``."""
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", layer_name + ".")
    if m is None:
        raise ValueError(f"kt_offload: cannot find 'layers.<N>.' in MoE layer name {layer_name!r}")
    return int(m.group(1))


def get_plan(cfg, num_layers: int, num_experts: int, top_k: int) -> OffloadPlan:
    global _PLAN, _PLAN_KEY
    key = (id(cfg), num_layers, num_experts, top_k)
    if _PLAN is None or key != _PLAN_KEY:
        _PLAN = plan_from_config(cfg, num_layers, num_experts, top_k)
        _PLAN_KEY = key
        rep = {k: v for k, v in _PLAN.report.items() if k != "npu_hit_rate"}
        logger.info(
            "kt_offload plan: %d layers x %d experts, %d NPU slots/layer; %s",
            num_layers,
            num_experts,
            _PLAN.num_slots,
            rep,
        )
    return _PLAN


def bind_routed_experts_args(init_fn, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Name the arguments AscendRoutedExperts forwards to RoutedExperts.__init__."""
    bound = inspect.signature(init_fn).bind_partial(None, *args, **kwargs)
    return dict(bound.arguments)


#: The kt build reads these at **load time**, inside the compiled extension.
#: ``KT_ZEROCOPY_MLOCK=0`` is deliberate, not an inherited default: placement has
#: exactly one owner -- ``numa_placement`` migrates and faults in these pages by
#: file offset before kt reads them, and kt's pin/migrate would run over it.
KT_ZEROCOPY_ENV: dict[str, str] = {
    "KT_ZEROCOPY_WEIGHTS": "1",
    "KT_ZEROCOPY_SCOPE": "w13",
    "KT_ZEROCOPY_MLOCK": "0",
}

#: ``KT_ZEROCOPY_WEIGHTS`` is off for "" and "0" (kt parses it that way on both
#: sides); every other value is on.
_ZC_OFF = ("", "0")


def _zerocopy_env_conflict(name: str, want: str | None) -> str | None:
    """``None`` if the environment agrees with the config, else why it does not."""
    have = os.environ.get(name)
    if have is None:
        return None
    if name == "KT_ZEROCOPY_WEIGHTS":
        have_on, want_on = have not in _ZC_OFF, want is not None and want not in _ZC_OFF
        if have_on != want_on:
            return f"{name}={have!r} in the environment says {'on' if have_on else 'off'}"
        return None
    if want is None:
        # Zero-copy off in the config but a scope/mlock exported anyway:
        # harmless in itself, the fingerprint of a half-applied manual setup.
        return f"{name}={have!r} is exported while kt_offload_config.zerocopy_w13 is false"
    if have != want:
        return f"{name}={have!r} in the environment, {want!r} from the config"
    return None


def apply_zerocopy_env(cfg) -> dict[str, str]:
    """Turn ``zerocopy_w13`` into the environment kt actually reads.

    Returns what was set (empty when the switch is off). A disagreement between
    an exported ``KT_ZEROCOPY_*`` and the config **raises** rather than silently
    overriding or silently obeying. With the switch off this sets **nothing**.
    """
    want = KT_ZEROCOPY_ENV if getattr(cfg, "zerocopy_w13", False) else {}
    conflicts = [
        msg for name in KT_ZEROCOPY_ENV
        if (msg := _zerocopy_env_conflict(name, want.get(name))) is not None
    ]
    if conflicts:
        raise RuntimeError(
            "kt_offload: the KT_ZEROCOPY_* environment disagrees with "
            f"kt_offload_config.zerocopy_w13={getattr(cfg, 'zerocopy_w13', False)}: "
            + "; ".join(conflicts)
            + ". Refusing to guess which one the run should be attributed to -- unset the "
            "variables and let the config drive them, or fix the config."
        )
    for name, value in want.items():
        os.environ[name] = value
    return dict(want)


def _rss_anon_bytes() -> int | None:
    """``RssAnon`` of this process, or ``None`` if /proc did not say."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


#: MXFP4 group size along ``hidden`` (checkpoint ``quant_config.group_size``).
#: Only feeds the log line below, which warns and never refuses.
_MXFP4_GROUP_SIZE = 32


def _kt_staging_bytes(hidden_size: int, intermediate_size: int,
                      group_size: int = _MXFP4_GROUP_SIZE) -> tuple[int, int]:
    """Host BYTES kt stages per CPU expert per layer: ``(w13, w2 + scales)``.

    Per fp4-moe.hpp: each proj is ``inter_tp * hidden / 2`` B (FP4 nibbles),
    each scale is ``(hidden / group_size) * inter_tp * 2`` B (bf16, not the
    NPU side's 1-byte E8M0). ``do_numa_job`` runs once per TP part, so summing
    the parts cancels the ``/tp`` -- do not divide here.

    Excludes the AMX BufferB buffers, so both values are floors -- and the copy
    path overshoots its floor by more, because zero-copy skips gate/up BufferB
    too (fp4-moe.hpp's ScaleOnly constructor allocates only the scale).
    """
    inter = int(intermediate_size)
    one_proj = inter * int(hidden_size) // 2
    one_scale = (int(hidden_size) // int(group_size)) * inter * 2
    return 2 * one_proj, one_proj + 3 * one_scale


def _report_zerocopy_effect(layer_idx: int, enabled: bool, d_anon: int | None,
                            w13_bytes: int, other_bytes: int) -> None:
    """Say, per layer, whether kt's CPU half actually skipped copying w13.

    A **proxy**: kt exposes no interface to ask, so this reads the one
    consequence that cannot be faked -- zero-copy w13 points kt's weights at the
    checkpoint's page cache, so anonymous memory grows by ``w2 + scale`` rather
    than by all three.

    Warns rather than raises: ``d_anon`` is whole-process, so no threshold here
    is sharp enough to stop a server on. The hard refusals live where they can
    be sharp -- ``_refuse_without_zerocopy_support`` (reads the .so) and
    ``setup_placement`` (raises on ``planned == 0``).
    """
    if d_anon is None:
        return
    expect_copy = w13_bytes + other_bytes
    expect_zc = other_bytes
    logger.info(
        "kt_offload L%d: zerocopy_w13=%s, RssAnon +%.3f GiB (copy path would be ~%.3f, "
        "zero-copy ~%.3f)", layer_idx, enabled, d_anon / float(1 << 30),
        expect_copy / float(1 << 30), expect_zc / float(1 << 30))
    if enabled and expect_copy > 0 and d_anon >= 0.8 * expect_copy:
        logger.warning(
            "kt_offload L%d: zerocopy_w13 is ON but anonymous memory grew %.3f GiB, ~what the "
            "COPY path costs (%.3f GiB); w13 looks like it was copied anyway. Check that the "
            "kt build reads KT_ZEROCOPY_SCOPE and that nothing reset the environment after "
            "import.", layer_idx, d_anon / float(1 << 30), expect_copy / float(1 << 30))


def offload_enabled(ascend_config) -> bool:
    """True only for an explicit ``enabled: true`` (a mock or missing section is off)."""
    cfg = getattr(ascend_config, "kt_offload_config", None)
    return cfg is not None and getattr(cfg, "enabled", False) is True


def create_layer_offload(ascend_config, init_fn, args: tuple, kwargs: dict) -> KTLayerOffload | None:
    """Build the layer's offload state, or None when the feature is off.

    Called before ``RoutedExperts.__init__``; mutates ``moe_config`` so the
    weights are allocated for ``num_slots`` experts only. With the feature off
    it returns before touching anything.
    """
    if not offload_enabled(ascend_config):
        return None
    from vllm.config import get_current_vllm_config

    cfg = ascend_config.kt_offload_config

    named = bind_routed_experts_args(init_fn, args, kwargs)
    layer_name: str = named["layer_name"]
    moe_config = named["moe_config"]
    vllm_config = get_current_vllm_config()
    hf = vllm_config.model_config.hf_text_config
    num_layers = int(hf.num_hidden_layers)
    num_experts = int(moe_config.num_logical_experts)
    top_k = int(moe_config.experts_per_token)

    layer_idx = _layer_index(layer_name)
    if layer_idx >= num_layers:
        from vllm_ascend.kt_offload.config import uses_hbm_dspark_draft

        if uses_hbm_dspark_draft(vllm_config):
            draft_hf = vllm_config.speculative_config.draft_model_config.hf_config
            num_draft_layers = (
                getattr(draft_hf, "n_mtp_layers", None) or getattr(draft_hf, "num_nextn_predict_layers", None) or 3
            )
            if layer_idx < num_layers + num_draft_layers:
                logger.info("kt_offload: %s keeps all %d draft experts in HBM", layer_name, num_experts)
                return None
        raise ValueError(
            f"kt_offload: MoE layer {layer_name!r} is past the {num_layers} main layers (MTP/draft); "
            "offload of draft experts is not implemented."
        )
    if moe_config.num_experts != num_experts:
        raise ValueError(
            "kt_offload: redundant/shared-fused experts are not supported "
            f"(num_experts={moe_config.num_experts}, logical={num_experts})."
        )
    if moe_config.ep_size != 1 or moe_config.tp_size != 1:
        raise ValueError(f"kt_offload: needs ep_size=tp_size=1, got {moe_config.ep_size}/{moe_config.tp_size}")
    if getattr(ascend_config, "mix_placement", False):
        raise ValueError("kt_offload: mix_placement appends shared experts to the routed ids; not supported.")
    if ascend_config.eplb_config.dynamic_eplb:
        raise ValueError("kt_offload: dynamic EPLB is not supported.")

    plan = get_plan(cfg, num_layers, num_experts, top_k)
    placement = plan.layer(layer_idx)
    moe_config.num_local_experts = plan.num_slots
    return KTLayerOffload(cfg, placement, layer_idx, vllm_config)


class KTLayerOffload:
    """CPU half of one MoE layer: kt wrapper + NPU routing tables."""

    def __init__(self, cfg, placement: LayerPlacement, layer_idx: int, vllm_config):
        self.cfg = cfg
        # Only the diagnostic sampler needs this; a test double need not carry it.
        self.num_layers = int(getattr(vllm_config.model_config.hf_text_config, "num_hidden_layers", 0)) or (
            placement.layer_idx + 1
        )
        self._ids_i64: dict[int, torch.Tensor] = {}
        self.placement = placement
        self.layer_idx = layer_idx
        hf = vllm_config.model_config.hf_text_config
        self.hidden_size = int(hf.hidden_size)
        self.intermediate_size = int(hf.moe_intermediate_size)
        self.top_k = int(hf.num_experts_per_tok)
        self.weight_path = cfg.weight_path or vllm_config.model_config.model
        self.chunked_prefill_size = cfg.chunked_prefill_size or int(vllm_config.scheduler_config.max_num_batched_tokens)
        self.swiglu_limit = 0.0
        self.wrapper = None
        self._slot_tables: dict[torch.dtype, torch.Tensor] = {}
        self._keep_tables: dict[torch.dtype, torch.Tensor] = {}
        self._pending: torch.Tensor | None = None
        self._pending_captured = False
        # Whether the captured submit filled OUR buffers (forked or packed) or
        # kt's own; merge() reads it to pick where the H2D comes from.
        self._pending_own_out = False
        self._pending_join: tuple | None = None
        # A6 across steps: set by _claim_side_buffers, cleared ONLY by
        # _join_side_stream. route()'s blanket _pending_join reset must not
        # clear it -- see _claim_side_buffers.
        self._side_inflight = False
        # Merge accounting (shared_window): exactly ONE merge call per route,
        # whoever makes it -- with VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW on the
        # caller is fused_moe._forward_impl instead of
        # routed_experts.forward_impl. Both halves of that split fail silently.
        # Counted here, checked at the next route -- see _assert_merge_balance.
        self._merge_owed = False
        self._merges_since_route = 0
        # Host-func count of this layer when its capture recorded the node.
        self._hostfunc_at_capture: int | None = None
        # Every optimisation switch comes from switches.resolve -- cfg, env and
        # the automatic defaults in one place (validate=False: config validation
        # already refused what it had to). Eager is applied below, once
        # graph_enabled is known.
        self.switches = switches.resolve(cfg, validate=False)
        self.overlap = self.switches.side_stream.on
        self.packed = bool(getattr(cfg, "packed_transfer", False))
        # P1a (async_wait.ENV_NAME; unset = on when packed or the side stream
        # is). The captured node submits and returns; the device waits on this
        # layer's ExternalEvent instead of on the callback's return. Captured
        # path only: eager already blocks the host.
        self.async_wait = self.switches.async_wait.on
        # P1b (device_signal.ENV_NAME; explicit "1" only). No host-func node at
        # all: an 8-byte copy of a per-step device counter behind the input copy
        # is the start signal, and a polling thread runs kt and records the same
        # ExternalEvent P1a uses. An automatic P1a steps aside for it; both
        # explicit is refused in config.
        self.device_signal = self.switches.device_signal.on
        # shared_window.ENV_NAME; unset = follows the side stream. When on,
        # forward_impl hands back an un-merged routed_out and
        # fused_moe._forward_impl calls merge() after the shared-expert MLP, so
        # that MLP lands in the window where the main stream only waits for kt.
        # Nothing in THIS class changes: merge() does the same work later in
        # program order, so the side stream being off still runs correctly.
        self.shared_in_window = self.switches.shared_in_window.on
        self._packed: dict[int, dict] = {}
        # bs -> (slot index in the pinned signal array, the 1-element view the
        # graph's counter copy writes). P1b only.
        self._signal_words: dict[int, tuple] = {}
        # Set in route() while capturing: did THIS layer emit the step
        # counter's increment? Exactly the lowest signal layer must.
        self._ops_prefix: list = []
        # device_signal.STEPS for this layer when its node was emitted, so the
        # merge can prove the polling thread did not run during the capture.
        self._steps_at_capture: int | None = None
        # bs -> this layer's ExternalEvent. One per (layer, batch size): the
        # graph's wait/reset nodes point at the object, so a second layer
        # sharing one would release the wrong layer's H2D.
        self._ext_events: dict[int, Any] = {}
        # The op sequence of the captured submit in flight, for the P1a
        # criterion (async_wait.ext_event_ops_ok). None outside one.
        self._submit_ops: list | None = None
        # Graph mode: capture sizes are known from the config before any
        # capture; kt needs them registered and their buffers allocated first.
        mc, cc = vllm_config.model_config, getattr(vllm_config, "compilation_config", None)
        mode = getattr(cc, "cudagraph_mode", None)
        self.graph_enabled = (
            not getattr(mc, "enforce_eager", True) and mode is not None and getattr(mode, "name", str(mode)) != "NONE"
        )
        sizes = getattr(cc, "cudagraph_capture_sizes", None) or []
        self.capture_sizes = sorted(set(sizes)) if self.graph_enabled else []
        if not self.graph_enabled:
            # In eager mode an AUTOMATIC P1a has no graph to act on and an
            # AUTOMATIC shared-window move has no window to fill; only an
            # explicit env "1" keeps either (P1a then warns and drops itself in
            # _enable_async_wait). The side stream keeps its value: eager never
            # forks whatever it says (route()), and the tests rely on the
            # attribute reflecting the config.
            if self.switches.async_wait.source != switches.ENV:
                self.async_wait = False
            if self.switches.shared_in_window.source != switches.ENV:
                self.shared_in_window = False
        # -- streaming prefill ---------------------------------------------
        # The provider that hands a chunk its device-side weights is the S5
        # weight I/O, which does not exist yet; None means maybe_stream_prefill
        # refuses rather than silently taking the hybrid path. Injected (rather
        # than built here) so the unit tests drive the fork without an NPU.
        self._stream_slot_provider = None
        # The provider OBJECT behind that callback (install_resident_fold): the
        # fold reads chunks back out of it, so it must be the very object that
        # served them, and it has to be told which layer is streaming
        # (``set_layer``). Injecting only the callback leaves this None.
        self._stream_provider = None
        # Set by the last streamed forward; a report with zero coverage is a
        # failure, so this is read by the tests and by the on-card runner.
        self.stream_coverage = None
        # The folded swap-in (fold_swap.ResidentFold), injected like the slot
        # provider; consulted only when cfg.dynamic_resident is on, and then
        # required (fail closed).
        self._resident_fold = None
        # Step number for the swap ledger, installed with the fold: one step per
        # model forward, counted on the first folded layer (see
        # ``_ForwardSteps``).
        self._steps: _ForwardSteps | None = None
        # Report of the last fold on this layer (None = not folded).
        self.fold_report = None
        # The module this offload belongs to, remembered by
        # ``after_weights_loaded``: ``live_swap_objects`` re-reads the weight
        # parameters off it before every commit (they are what the GEMM and a
        # capture read, and the post-load hook rebinds ``param.data``).
        self._module = None
        # Device-side resident mask, allocated by ``build_swap_targets`` only.
        self._device_mask: torch.Tensor | None = None
        # Kept for one question: has the KV cache already been sized? The fold's
        # staging must be allocated BEFORE the profile run, and
        # ``num_gpu_blocks`` is the observable that says the profile has run.
        self._cache_config = getattr(vllm_config, "cache_config", None)

    @property
    def num_slots(self) -> int:
        return self.placement.num_slots

    # -- construction-time ------------------------------------------------
    def attach_after_init(self, layer) -> None:
        """Finish wiring once RoutedExperts.__init__ has run."""
        layer.local_num_experts = self.num_slots
        if getattr(layer, "log2phy", None) is not None:
            raise ValueError("kt_offload: a log2phy map is active; offload owns the placement.")
        # kt's clamp is off at 0.0; it must be handed the model's value
        # explicitly (DSV4.1: 10.0), or the CPU experts silently skip the clamp
        # the NPU experts apply (npu_swiglu_group_quant clamp_value).
        limit = getattr(layer.moe_config, "swiglu_limit", None)
        if limit is None:
            raise ValueError(
                "kt_offload: moe_config.swiglu_limit is unset; refusing to run the CPU experts "
                "without the clamp the NPU experts apply."
            )
        self.swiglu_limit = float(limit)

        quant_method = layer.quant_method
        original = quant_method.process_weights_after_loading

        def process_weights_after_loading(mod, _orig=original, _self=self):
            _self.zero_unused_slots(mod)
            _orig(mod)
            _self.after_weights_loaded(mod)

        # Stamped so the streaming path can find its way back to the unpatched
        # hook (``stream_io.default_post_load`` reads both attributes): the
        # chunk/staging formatter wants ONLY the cast, and running this wrapper
        # on a throwaway module makes kt build a SECOND wrapper for this layer
        # and reload its whole CPU half, after rebinding the live tables.
        process_weights_after_loading._kt_offload_patched = True
        process_weights_after_loading._kt_offload_original = original
        # Instance attribute: the model loader calls
        # module.quant_method.process_weights_after_loading(module).
        quant_method.process_weights_after_loading = process_weights_after_loading

    def zero_unused_slots(self, layer) -> None:
        """Unused tail slots were never loaded; make their bytes defined."""
        unused = self.placement.unused_slots
        if not unused:
            return
        for name, param in layer.named_parameters(recurse=False):
            if param.dim() >= 1 and param.shape[0] == self.num_slots:
                param.data[unused[0] :].zero_()

    def after_weights_loaded(self, layer) -> None:
        """Pre-bake device tables and load the kt experts (before any capture)."""
        _maybe_start_host_func_log(self.num_layers)
        device = self._device_of(layer)
        p = self.placement
        self._module = layer
        self._slot_tables = {dt: p.safe_slot_table(dt).to(device) for dt in (torch.int32, torch.int64)}
        self._keep_tables = {dt: p.keep_table(dt).to(device) for dt in (torch.float32, torch.bfloat16)}
        # The two lines above bind FRESH tensors on every call, so anything built
        # from the previous ones (the swap targets) is detached from here on --
        # which is why the committer re-reads ``live_swap_objects`` before every
        # commit. install_resident_fold collects the layers here: the first
        # moment a layer can hand over commit targets.
        _register_offload_layer(self)
        if not p.cpu_ids:
            return  # every expert resident: nothing for kt to do
        self.wrapper = self._build_wrapper()
        # With shared_in_window the kt merge is made by fused_moe._forward_impl
        # after the shared-expert MLP. 🔴 Criterion for changing it: bit-identity
        # with the switch off, never TPOT -- a dropped merge is silent and faster.
        # The switch and how it resolved are reported once by config.py.
        # Seam timing, if the env var asks for it. The CPUInfer probe splits kt's
        # submit from its sync without touching kt's C++; where WE make those two
        # calls (packed and forked) no probe is needed, and asking for one anyway
        # overwrites our stamps and lengthens the very host-func node the side
        # stream exists to hide.
        own_node = self.packed or self.overlap
        seam_timing.maybe_enable(self.num_layers, None if own_node else self.wrapper, split=own_node)
        if self.overlap and self.graph_enabled:
            # Created here, never inside a capture, and subscribed for host-func
            # reports before any capture can reach it. Graph mode only: eager
            # never forks (route()).
            side = _side_stream(device)
            _subscribe_stream(side)
            # 🔴 Criterion for changing this: bit-identity with the serial arm,
            # never TPOT -- every known failure of this path is faster and wrong.
            if not self._packed_supported():
                raise RuntimeError(
                    "kt_offload: overlap_side_stream runs the CPU experts on this layer's OWN pinned "
                    "buffers, so it needs kt's moe.forward_task and cpu_infer.submit/sync; this kt build "
                    f"({type(self.wrapper).__name__}) does not expose them. Turn the side stream off "
                    f"({side_stream.ENV_NAME}=0 or overlap_side_stream=false; it is on by default)."
                )
        if self.packed:
            if not self._packed_supported():
                raise RuntimeError(
                    "kt_offload: packed_transfer needs kt's moe.forward_task and cpu_infer.submit/sync; "
                    f"this kt build ({type(self.wrapper).__name__}) does not expose them."
                )
        if self.packed or self.overlap:
            # Graph mode: every capture size up front (nothing may allocate
            # inside a capture). Eager allocates lazily, on first use.
            for bs in self.capture_sizes:
                self._packed_buffers(bs, device)
        if self.async_wait:
            self._enable_async_wait(device)
        if self.device_signal:
            self._enable_device_signal(device)
        if self.graph_enabled:
            self._prepare_capture_buffers(device)

    def _enable_async_wait(self, device) -> None:
        """P1a: this layer's ExternalEvents, and the one completion thread.

        Both at weight-load time, because both must exist before any capture: an
        ``ExternalEvent`` created inside a capture is not the object the graph's
        node points at, and a stream created there is not one the graph may use.
        ``Completer.start`` is synchronous, so a thread that cannot take an ACL
        context is a start-up error rather than a hang at the first replay.
        """
        global _COMPLETER
        if not self.graph_enabled:
            # Eager already blocks the host inside route(): there is no
            # callback to return early from, and no graph to hold the H2D back.
            logger.warning(
                "kt_offload: %s=1 has no effect in eager mode (the async wait is a captured-path change); "
                "layer %d runs the ordinary submit+sync.",
                async_wait.ENV_NAME,
                self.layer_idx,
            )
            self.async_wait = False
            return
        if not self._packed_supported():
            raise RuntimeError(
                f"kt_offload: {async_wait.ENV_NAME}=1 needs kt's moe.forward_task and "
                f"cpu_infer.submit/sync to split the submit from the sync; this kt build "
                f"({type(self.wrapper).__name__}) does not expose them."
            )
        if not (self.packed or self.overlap):
            raise RuntimeError(
                f"kt_offload: {async_wait.ENV_NAME}=1 needs this layer's own pinned buffers "
                "(packed_transfer or overlap_side_stream); config._validate_preconditions should have "
                "refused this already."
            )
        for bs in self.capture_sizes:
            self._ext_events[bs] = _new_external_event(device)
            _EXT_EVENTS.append(self._ext_events[bs])
        if _COMPLETER is None:
            _COMPLETER = _make_completer(device)
            _COMPLETER.start()
        # The host-func node only submits and returns; the device waits on an
        # ExternalEvent that the completer records after syncing.
        # 🔴 Criterion for changing this: bit-identity with the switch off, never
        # TPOT -- releasing early is faster, wrong, and silent.

    def _enable_device_signal(self, device) -> None:
        """P1b: this layer's ExternalEvents and signal words, and the poller.

        All at weight-load time, because all of it is forbidden inside a capture:
        allocating the pinned words, creating the device counter, creating an
        ``ExternalEvent`` the graph's nodes will point at, and starting a thread
        that needs an ACL context.

        The self-test at the end drives the whole signal path once, eagerly, and
        refuses to start if any link of it does not work. Each would otherwise
        fail as an unexplained hang at the first decode step.
        """
        global _POLLER, _SIGNALS, _STEP_COUNTER
        if not self.graph_enabled:
            # Eager blocks the host inside route(); there is no graph to hold
            # the H2D back and no node to remove.
            logger.warning(
                "kt_offload: %s=1 has no effect in eager mode (removing the host-func node is a "
                "captured-path change); layer %d runs the ordinary submit+sync.",
                device_signal.ENV_NAME,
                self.layer_idx,
            )
            self.device_signal = False
            return
        if not self._packed_supported():
            raise RuntimeError(
                f"kt_offload: {device_signal.ENV_NAME}=1 needs kt's moe.forward_task and "
                f"cpu_infer.submit/sync -- with no callback in the graph, the polling thread is what calls "
                f"them; this kt build ({type(self.wrapper).__name__}) does not expose them."
            )
        if not (self.packed or self.overlap):
            raise RuntimeError(
                f"kt_offload: {device_signal.ENV_NAME}=1 needs this layer's own pinned buffers "
                "(packed_transfer or overlap_side_stream); config._validate_preconditions should have "
                "refused this already."
            )
        device_signal.load_native()  # a start-up error here, not a hang at the first replay
        if _SIGNALS is None:
            # Capacity for every layer of the model at every capture size, so a
            # layer that loads later never has to grow (and never has to move)
            # an array the polling thread is already reading.
            _SIGNALS = _SignalWords(max(self.num_layers, self.layer_idx + 1) * max(len(self.capture_sizes), 1))
            _STEP_COUNTER = _StepCounter(device)
            _selftest_signal_path(_SIGNALS, _STEP_COUNTER)
        for bs in self.capture_sizes:
            # One event per (layer, bs) like P1a: the graph's wait/reset nodes
            # point at the object, so two layers sharing one would release the
            # wrong layer's H2D.
            event = self._ext_events.get(bs)
            if event is None:
                event = _new_external_event(device)
                self._ext_events[bs] = event
                _EXT_EVENTS.append(event)
            slot, word = _SIGNALS.take()
            self._signal_words[bs] = (slot, word)
        _SIGNAL_LAYERS.add(self.layer_idx)
        if _POLLER is None:
            _POLLER = _make_poller(device)
            _POLLER.start()
        for bs in self.capture_sizes:
            slot, _word = self._signal_words[bs]
            _POLLER.register(device_signal.Entry(self.layer_idx, bs, slot, self._poll_run, self._ext_events[bs]))
        logger.warning(
            "kt_offload: DEVICE SIGNAL ENABLED for layer %d (%s=1). 图里不再有 host-func 节点：输入 D2H 之后"
            "再挂一次 8 字节计数器 D2H 作为启动信号，常驻轮询线程看到本步步号就跑 kt 并 record，设备等的是"
            " ExternalEvent。实验性，默认关；判据是与关着的那一臂逐位相同，不是 TPOT —— 放行早 = 更快且算错，"
            "且不报错 (every known failure of this path is faster and wrong). %d capture size(s), slots %s.",
            self.layer_idx,
            device_signal.ENV_NAME,
            len(self._signal_words),
            sorted(s for s, _ in self._signal_words.values()),
        )

    def _poll_run(self, bs: int) -> None:
        """This layer's CPU experts, called on the polling thread.

        What ``_kt_packed_forward`` does inside the callback, minus the callback:
        the graph's counter copy has landed, so the input copy in front of it on
        the same stream has too and the pinned buffer holds this step's routing.
        Exceptions are NOT caught here -- ``device_signal.Poller._handle`` logs
        them and records the event anyway, so a failure degrades to stale output
        with a message rather than the device waiting for ever.

        The ``forward_task`` call below is deliberately a literal copy of the one
        in ``_kt_packed_forward`` (whose docstring lists the four properties of
        that C++ task this depends on). The two must not drift;
        ``test_device_signal.py::test_the_poller_builds_the_same_kt_task``
        compares the arguments both of them pass.
        """
        wrapper = self.wrapper
        buf = self._packed[bs]
        _assert_no_pending(wrapper, self.layer_idx, self.layer_idx - 1, "before this step's submit")
        task = wrapper.moe.forward_task(
            buf["bsz"].data_ptr(),
            buf["pin_ids"].shape[-1],
            buf["pin_ids"].data_ptr(),
            buf["pin_weights"].data_ptr(),
            buf["pin_hidden"].data_ptr(),
            buf["pin_out"].data_ptr(),
            False,  # incremental: no deferred experts
        )
        wrapper.cpu_infer.submit(task)
        wrapper.cpu_infer.sync(0)
        _assert_no_pending(wrapper, self.layer_idx, self.layer_idx, "after this step drained")
    # -- the folded swap's commit targets -----------------------------------
    def live_swap_objects(self) -> dict[str, Any]:
        """The objects this layer really routes and computes through, read NOW.

        Handed to ``ResidentSwapCommitter`` as ``live_objects``, which compares
        them with the commit targets (``live_target_violations``) before every
        single commit. Nothing is captured in a closure: the case guarded against
        is ``after_weights_loaded`` running a second time, rebinding
        ``_slot_tables`` / ``_keep_tables`` and rebuilding the kt wrapper, which
        a build-time snapshot could not see.

        ``device_mask`` is deliberately absent: nothing on the device reads a
        resident mask in this build, so there is no live object to compare
        against.
        """
        if self._module is None:
            raise RuntimeError(
                f"kt_offload layer {self.layer_idx}: no module bound; live_swap_objects() is only meaningful "
                "after process_weights_after_loading (which is where the tables and the kt wrapper appear)")
        if self.wrapper is None:
            raise RuntimeError(
                f"kt_offload layer {self.layer_idx}: no kt wrapper, so there is no kt mask to write. A layer "
                "with every expert resident has no CPU half and cannot take part in a swap.")
        return {
            # The parameters themselves: the GEMM and a captured graph read
            # these objects, not a copy of them.
            "weights": {name: getattr(self._module, name) for name in SLOT_WEIGHT_ATTRS},
            # ``self._slot_tables`` / ``self._keep_tables`` THEMSELVES --
            # route() indexes exactly these dicts. Rebuilding them from the
            # placement here would produce equal values in different storage and
            # make this whole check a tautology.
            "slot_tables": self._slot_tables,
            "keep_tables": self._keep_tables,
            # kt copies the mask it was GIVEN into its own pinned tensor and
            # hands C++ that one's data_ptr, so the mask this layer passed to
            # the constructor is not the one kt reads (resident_swap.kt_mask_of).
            "kt_mask": kt_mask_of(self.wrapper),
        }

    def build_swap_targets(self, layer) -> LayerSwapTargets:
        """The ONLY legal way to build this layer's commit targets.

        Every object in the result is the live one: the layer's parameters,
        ``_slot_tables`` / ``_keep_tables``, kt's own pinned mask, and the
        demotable set from what kt actually loaded. Building any of them afresh
        from the placement gives equal values in different storage, and then a
        swap moves the weights while the routing keeps pointing at the old slots:
        no error, plausible magnitudes, wrong answer. The result is checked
        against ``live_swap_objects()`` before it is returned.

        Call it after ``after_weights_loaded`` (the tables are baked and the kt
        wrapper exists only from there) and before any graph capture (a capture
        records the addresses these targets are pinned to).
        """
        if layer is not self._module:
            raise RuntimeError(
                f"kt_offload layer {self.layer_idx}: build_swap_targets() was given a different module than "
                "the one process_weights_after_loading ran on; the targets would name another layer's tensors")
        if not self._slot_tables or not self._keep_tables:
            raise RuntimeError(
                f"kt_offload layer {self.layer_idx}: routing tables are not baked yet; build_swap_targets() "
                "must run after process_weights_after_loading")
        if _capturing():
            raise RuntimeError(
                f"kt_offload layer {self.layer_idx}: build_swap_targets() inside a graph capture. The targets "
                "and the staging they are written from must exist before anything is captured.")
        live = self.live_swap_objects()
        p = self.placement
        device = self._device_of(layer)
        if self._device_mask is None:
            # A real device tensor, allocated once and written by every commit.
            # Nothing on the device READS it today (routing goes through the two
            # tables, kt reads its own pinned copy); it is kept current because
            # the ownership ledger's NPU claim is read from it, and so a future
            # device-side consumer does not find a mask that went stale at the
            # first swap.
            self._device_mask = p.resident_mask().to(device)
        targets = LayerSwapTargets(
            layer_idx=self.layer_idx,
            weights=dict(live["weights"]),
            slot_tables=dict(live["slot_tables"]),
            keep_tables=dict(live["keep_tables"]),
            device_mask=self._device_mask,
            kt_mask=live["kt_mask"],
            kt_cpu_weight_ids=kt_cpu_weight_ids_of(self.wrapper),
        )
        detached = live_target_violations(targets, **self.live_swap_objects())
        if detached:
            raise RuntimeError(
                f"kt_offload layer {self.layer_idx}: the targets this builder produced are not the objects the "
                "forward reads -- " + "; ".join(detached))
        return targets

    # -- packed transfer (cfg.packed_transfer) ------------------------------

    def _packed_supported(self) -> bool:
        moe = getattr(self.wrapper, "moe", None)
        infer = getattr(self.wrapper, "cpu_infer", None)
        return all(
            (
                moe is not None and hasattr(moe, "forward_task"),
                infer is not None and hasattr(infer, "submit") and hasattr(infer, "sync"),
            )
        )

    def _packed_buffers(self, bs: int, device) -> dict:
        """One combined input buffer per batch size, and our own output pair.

        Own buffers on purpose: kt hands layer L the pinned slot ``L % 2``, so
        two layers share one buffer and the ordering between one layer's output
        H2D and the other's CPU write is something the caller has to maintain.
        Per-layer buffers remove that coupling, and one combined input region
        turns kt's three input copies into one (placement.host_ops_per_layer).
        """
        buf = self._packed.get(bs)
        if buf is not None:
            return buf
        if bs > self.chunked_prefill_size:
            # kt's own wrappers check this (experts_base._check_qlen_fits_cpp_buffers);
            # we call the C++ task directly, so the check is ours. The C++ output
            # pointer advances qlen * hidden_size with no bound of its own, so
            # past chunked_prefill_size it writes outside the allocation.
            raise ValueError(
                f"kt_offload packed_transfer: {bs} tokens exceeds chunked_prefill_size "
                f"{self.chunked_prefill_size}; the CPU MoE would write past its output buffer."
            )
        lay = packed_layout(bs, self.hidden_size, self.top_k)
        dev = torch.empty(lay["nbytes"], dtype=torch.uint8, device=device)
        pin = _pinned_empty((lay["nbytes"],), torch.uint8)

        def view(store, region, dtype, shape):
            off, nbytes = lay[region]
            return store[off : off + nbytes].view(dtype).view(*shape)

        buf = {
            "layout": lay,
            "dev": dev,
            "pin": pin,
            "dev_hidden": view(dev, "hidden", torch.bfloat16, (bs, self.hidden_size)),
            "dev_ids": view(dev, "ids", torch.int64, (bs, self.top_k)),
            "dev_weights": view(dev, "weights", torch.float32, (bs, self.top_k)),
            "pin_hidden": view(pin, "hidden", torch.bfloat16, (bs, self.hidden_size)),
            "pin_ids": view(pin, "ids", torch.int64, (bs, self.top_k)),
            "pin_weights": view(pin, "weights", torch.float32, (bs, self.top_k)),
            "pin_out": _pinned_empty((bs, self.hidden_size), torch.bfloat16),
            "dev_out": torch.empty((bs, self.hidden_size), dtype=torch.bfloat16, device=device),
            "bsz": _pinned_empty((1,), torch.int32).fill_(bs),
        }
        for name, dtype, shape in (
            ("pin_hidden", torch.bfloat16, (bs, self.hidden_size)),
            ("pin_ids", torch.int64, (bs, self.top_k)),
            ("pin_weights", torch.float32, (bs, self.top_k)),
            ("pin_out", torch.bfloat16, (bs, self.hidden_size)),
        ):
            t = buf[name]
            # The C++ task takes raw pointers: a non-contiguous view or the
            # wrong dtype is read as if it were the layout it expects.
            if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
                raise AssertionError(
                    f"kt_offload packed buffer {name}: {t.dtype}{tuple(t.shape)} "
                    f"contiguous={t.is_contiguous()}, want {dtype}{shape} contiguous"
                )
        self._packed[bs] = buf
        return buf

    def _packed_submit(self, hs: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> dict:
        """Fill the combined buffer on the device, then one host-device copy."""
        buf = self._packed.get(hs.shape[0])
        if buf is None:
            if _capturing():
                raise RuntimeError(
                    f"kt_offload: capturing {hs.shape[0]} tokens with packed_transfer but no buffer was "
                    f"prepared for that size (prepared: {sorted(self._packed)}); allocating inside a capture "
                    "is not possible."
                )
            buf = self._packed_buffers(hs.shape[0], hs.device)
        _device_fill(buf["dev_hidden"], hs)  # device-side fills: ordinary device work
        _device_fill(buf["dev_ids"], topk_ids)
        _device_fill(buf["dev_weights"], topk_weights)
        _host_copy(buf["pin"], buf["dev"])  # the one host-device copy
        return buf

    def _packed_payload(self, bs: int) -> tuple:
        buf = self._packed[bs]
        return (self.wrapper, buf, self.layer_idx)

    def _launch_cpu_node(self, stream, bs: int) -> None:
        """The captured CPU-expert node(s) for this layer, on ``stream``.

        One place for both own-buffer captured paths (packed on the compute
        stream, forked on the side stream), so the op sequence the P1a criterion
        reads is recorded exactly where the nodes are emitted. The caller has
        already enqueued the D2H of the combined input buffer on this same
        stream; ``merge`` enqueues the H2D, appends to ``_submit_ops`` and runs
        the criterion there, once the whole shape exists.

        With ``async_wait`` off this is one host-func node that submits, waits
        and returns. With ``device_signal`` on there is no host-func node at all:
        four ordinary device ops, and the only host involvement is a thread
        watching the word the third of them writes.
        """
        self._submit_ops = [*self._ops_prefix, ("d2h", bs)]
        self._ops_prefix = []
        if self.device_signal:
            self._launch_signal_node(stream, bs)
            return
        if not self.async_wait:
            payload = self._packed_payload(bs)
            _CAPTURED_PAYLOADS.append(payload)
            _launch_host_func(stream, _kt_packed_forward, payload)
            self._submit_ops.append(("hostfunc", self.layer_idx))
            return
        event = self._ext_events.get(bs)
        if event is None:
            raise RuntimeError(
                f"kt_offload async wait: layer {self.layer_idx} has no ExternalEvent for {bs} tokens "
                f"(prepared: {sorted(self._ext_events)}); creating one inside a capture would not be the "
                "object the graph's node points at."
            )
        payload = (*self._packed_payload(bs), event)
        _CAPTURED_PAYLOADS.append(payload)
        _launch_host_func(stream, _kt_packed_submit_only, payload)
        self._submit_ops.append(("hostfunc", self.layer_idx))
        # The device waits here for the completion thread. The reset behind it
        # is what torch_npu says an ExternalEvent needs before it is reused
        # (streams.py). Both are graph NODES; neither is a host-side record
        # (that one belongs to the completion thread, at replay time).
        _ext_wait(event, stream)
        self._submit_ops.append(("ext_wait", event))
        _ext_reset(event, stream)
        self._submit_ops.append(("ext_reset", event))

    def _launch_signal_node(self, stream, bs: int) -> None:
        """P1b: the 8-byte start signal, then the same wait/reset pair.

        A **second copy on the same stream**, never a tail field appended to the
        input copy: same-stream tasks run in order, so "the counter landed"
        implies "the inputs landed", whereas the write order *inside* one DMA is
        not specified. That implication is the whole safety argument for letting
        another thread start kt without being told by the device.

        Everything the polling thread will read must be written before this copy
        -- today the combined pinned input buffer, written by the copy
        immediately in front of this one, on this stream.
        """
        event = self._ext_events.get(bs)
        entry = self._signal_words.get(bs)
        if event is None or entry is None:
            raise RuntimeError(
                f"kt_offload device signal: layer {self.layer_idx} has no ExternalEvent/signal word for {bs} "
                f"tokens (prepared: {sorted(self._signal_words)}); creating either inside a capture would "
                "not be the object the graph's nodes point at."
            )
        if _STEP_COUNTER is None:
            raise RuntimeError("kt_offload device signal: the step counter was never created")
        _slot, word = entry
        _host_copy(word, _STEP_COUNTER.value)
        self._submit_ops.append(("signal", (self.layer_idx, bs)))
        if _capturing():
            self._steps_at_capture = device_signal.STEPS.get(self.layer_idx, 0)
        # Same pair as P1a, and for the same reasons: the wait is the only
        # thing holding the H2D back, and the reset is what torch_npu says an
        # ExternalEvent needs before it is reused.
        _ext_wait(event, stream)
        self._submit_ops.append(("ext_wait", event))
        _ext_reset(event, stream)
        self._submit_ops.append(("ext_reset", event))

    def host_ops(self) -> dict[str, int]:
        """Host-device operations this layer costs per step (for the accounting)."""
        return host_ops_per_layer(self.packed)

    def _prepare_capture_buffers(self, device: torch.device) -> None:
        """Register every capture size with kt, and give THIS layer its buffers.

        Two scopes, and conflating them is a bug. kt's KExpertsCPUBuffer is
        class-level, so registering the capture sizes and allocating its pinned
        buffers happens once per process; ``self._ids_i64`` is per layer instance
        and must be rebuilt on every call, or only the first layer built ever
        gets id buffers and every other layer's first captured forward raises
        ``no preallocated int64 id buffer``.

        Only registered sizes are kept by KExpertsCPUBuffer, and nothing may
        allocate pinned host or device memory inside a capture.
        """
        global _CAPTURE_BUFFERS_READY
        key = (tuple(self.capture_sizes), self.hidden_size, self.top_k, str(device))
        if _CAPTURE_BUFFERS_READY is not None and key != _CAPTURE_BUFFERS_READY:
            raise RuntimeError(f"kt_offload: capture buffers already prepared for {_CAPTURE_BUFFERS_READY}, now {key}")
        too_big = [bs for bs in self.capture_sizes if bs > self.chunked_prefill_size]
        if too_big:
            raise ValueError(
                f"kt_offload: capture sizes {too_big} exceed chunked_prefill_size {self.chunked_prefill_size}"
            )
        kt_buffer = _kt_buffer_class()
        if _CAPTURE_BUFFERS_READY is None:  # process-wide half, once
            type(self.wrapper).set_capture_batch_sizes(list(self.capture_sizes))
            for bs in self.capture_sizes:
                kt_buffer.get_buffer(
                    torch.empty((bs, self.hidden_size), dtype=torch.bfloat16, device=device), self.top_k
                )
            _CAPTURE_BUFFERS_READY = key
            logger.info(
                "kt_offload: kt capture buffers ready for %d sizes (max %d)",
                len(self.capture_sizes),
                max(self.capture_sizes, default=0),
            )
        # Per layer instance, every time. kt casts the ids to int64 before its
        # D2H; under capture that cast allocates on whichever stream we are
        # enqueueing to, and on the side stream that is not the capture stream's
        # pool. Hand kt ids that are already int64, from a buffer allocated here
        # before any capture, and kt's .to(torch.long) is the identity and
        # allocates nothing. One buffer per layer, never shared (A6).
        self._ids_i64 = {
            bs: torch.empty((bs, self.top_k), dtype=torch.int64, device=device) for bs in self.capture_sizes
        }
        missing = [bs for bs in self.capture_sizes if bs not in kt_buffer.capture_buffers]
        if missing:
            raise RuntimeError(f"kt_offload: kt did not keep capture buffers for sizes {missing}")

    @staticmethod
    def _device_of(layer) -> torch.device:
        for param in layer.parameters(recurse=False):
            return param.device
        raise RuntimeError("kt_offload: MoE layer has no parameters")

    def _build_wrapper(self):
        if self.cfg.force_sync_submit:
            os.environ["KT_FORCE_SYNC_SUBMIT"] = "1"
        # Before the import, like KT_FORCE_SYNC_SUBMIT above: kt reads these
        # inside the extension at load time, and a later assignment is ignored
        # without a word.
        zerocopy_env = apply_zerocopy_env(self.cfg)
        anon_before = _rss_anon_bytes()
        if self.graph_enabled:
            # We subscribe the capture stream ourselves (_subscribe_stream);
            # kt must not start its own poller or subscribe it a second time.
            os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"
        from kt_kernel import KTMoEWrapper  # host-only dependency, imported when enabled

        _install_capture_probe(KTMoEWrapper)

        wrapper = KTMoEWrapper(
            layer_idx=self.layer_idx,
            num_experts=self.placement.num_experts,
            num_experts_per_tok=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
            gpu_experts_mask=self.placement.resident_mask(),
            cpuinfer_threads=self.cfg.cpuinfer_threads,
            threadpool_count=self.cfg.threadpool_count,
            weight_path=self.weight_path,
            chunked_prefill_size=self.chunked_prefill_size,
            method=self.cfg.method,
            numa_nodes=self.cfg.numa_nodes,
            swiglu_limit=self.swiglu_limit,
        )
        # kt indexes experts by logical id; with EPLB off physical == logical.
        # device="cpu" is explicit: this runs while the model loads, where the
        # default device is the NPU, and kt's argument is a *host* map
        # (physical_to_logical_map_cpu).
        wrapper.load_weights(torch.arange(self.placement.num_experts, dtype=torch.int64, device="cpu"))
        # The only kt-side evidence reachable from here; see
        # _report_zerocopy_effect for what it is and what it is not.
        anon_after = _rss_anon_bytes()
        d_anon = None if anon_before is None or anon_after is None else anon_after - anon_before
        n_cpu = max(0, int(self.placement.num_experts) - len(self.placement.resident_ids))
        per_w13, per_other = _kt_staging_bytes(self.hidden_size, self.intermediate_size)
        _report_zerocopy_effect(self.layer_idx, bool(zerocopy_env),
                                d_anon, n_cpu * per_w13, n_cpu * per_other)
        _refuse_cross_layer(wrapper, self.layer_idx)  # A1: deferral cannot be produced
        # Resolve the report-thread pinning here, on the main thread and after
        # kt's pool exists: 'auto' reads kt's real threads to check its model,
        # and a refusal must refuse the start-up rather than warn inside a
        # graph node. Off unless VLLM_ASCEND_DSV41_KT_REPORT_CPU is set.
        report_affinity.resolve_once(self.cfg)
        _register_wrapper(wrapper, self.layer_idx)  # A5/A9: one wrapper per index, one CPUInfer
        return wrapper

    # -- forward ------------------------------------------------------------
    def maybe_stream_prefill(
        self,
        layer,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        in_profile_run: bool,
    ):
        """Compute this layer's MoE from streamed weights, or return ``None``.

        ``None`` means this forward takes the hybrid path and the caller carries
        on exactly as before. A tensor means the whole MoE is done and the caller
        must skip ``route``/``apply``/``merge`` altogether: running ``route`` as
        well submits the CPU half for experts the stream already computed, so
        every one of them lands in the output twice, with no error anywhere.

        Every reason to decline is decided BEFORE any work; anything that goes
        wrong after that raises instead of quietly falling back to hybrid.
        """
        if not getattr(self.cfg, "stream_prefill", False):
            return None
        if self._stream_slot_provider is None:
            # The weight I/O (S5) is not wired yet. Fail closed rather than
            # leave a switch that is on and silently does nothing.
            raise RuntimeError(
                "kt_offload: stream_prefill is on but no slot provider is installed "
                "(the streaming weight I/O, plan stage S5, is not implemented). "
                "Refusing rather than falling back to hybrid, which would make every "
                "'streaming on' reading a hybrid reading.")
        if in_profile_run:
            # The profile run routes at random, so streaming it would measure a
            # shape production never sees -- and the NPU peak it is there to
            # measure is unchanged by which half computes the experts.
            return None
        num_tokens = int(hidden_states.shape[:-1].numel())
        if num_tokens < self.cfg.stream_prefill_min_tokens:
            return None
        if _capturing():
            # Not an error: decode steps are captured and are exactly the
            # forwards that must NOT stream. assert_eager_only() covers the
            # other direction (a captured size that is big enough to stream).
            return None

        from vllm_ascend.kt_offload import stream_chain, stream_io, stream_prefill

        stream_prefill.assert_eager_only(
            num_tokens, self.capture_sizes, self.cfg.stream_prefill_min_tokens,
            where=f"kt_offload layer {self.layer_idx}")
        # EVERY expert, not just the offloaded ones: the fork skips
        # route/apply/merge entirely, so whatever this plan does not cover
        # contributes NOTHING to the layer's output -- no error, plausible
        # numbers, wrong answer. Streaming only the offloaded set and letting the
        # normal NPU path compute the resident ones is cheaper, but it is a
        # different wiring.
        expected_ids = list(range(self.placement.num_experts))
        plan = stream_prefill.build_chunk_plan(
            self.placement.num_experts,
            self.cfg.stream_prefill_chunk_experts,
            expert_ids=expected_ids,
        )
        if self._stream_provider is not None:
            # The provider serves one layer at a time and the chunk callback
            # carries only the chunk index, so it must be told which layer is
            # streaming; the fold reads its chunks back out of that same object
            # and refuses if the two disagree.
            self._stream_provider.set_layer(self.layer_idx)
        run_kwargs = dict(
            # The scheme, not the wrapper: production's ``layer.quant_method``
            # is ``AscendFusedMoEMethod``, which HOLDS the scheme. The chain
            # reads quant_type / act_quant_type / apply_* off the scheme, while
            # the post-load hook belongs to the wrapper.
            quant_method=stream_io.moe_scheme(layer.quant_method, *stream_chain.SCHEME_ATTRS),
            plan=plan,
            slot_for=self._stream_slot_provider,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            moe_config=layer.moe_config,
            top_k=self.top_k,
            capture_sizes=self.capture_sizes,
            min_tokens=self.cfg.stream_prefill_min_tokens,
            resident_layer=layer,
            expected_expert_ids=expected_ids,
            strict=True,
        )
        fold = self._fold_for_this_forward()
        if fold is None:
            result = stream_chain.run_streamed_layer(**run_kwargs)
        else:
            # The chosen experts are copied out of each chunk as it streams past
            # and committed once the layer's last chunk is done. A layer the fold
            # declines (no change, capture state unknown, ...) streams unhooked.
            result, report = fold.fold_streamed_layer(
                self.layer_idx, topk_ids,
                lambda hook: stream_chain.run_streamed_layer(
                    **run_kwargs, **({} if hook is None else {"after_chunk": hook})))
            self.fold_report = report
            if report is not None:
                # The committer owns the placement now; route() reads the
                # tables it rewrote in place, anything else reads this.
                self.placement = fold.committer.placement(self.layer_idx)
        self.stream_coverage = result.coverage
        return result.output

    def _fold_for_this_forward(self):
        """The folded swap, if the switch is on. On without a fold installed raises."""
        if not getattr(self.cfg, "dynamic_resident", False):
            return None
        if self._resident_fold is None:
            raise RuntimeError(
                "kt_offload: dynamic_resident is on but no resident fold is installed on this layer "
                "(layer.install_resident_fold was never called, or it did not cover this layer). Refusing "
                "rather than streaming without it, which would make every 'dynamic resident on' reading a "
                "static-resident reading.")
        if self._steps is not None:
            # One step per model forward, counted on the first folded layer:
            # every layer of this forward commits under the same number (the
            # ledger allows equal, not older) and the next forward is newer.
            self._steps.advance(self.layer_idx)
        return self._resident_fold

    def route(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        in_profile_run: bool,
    ) -> NpuRouting:
        """Submit the CPU half with the original routing; return the NPU half.

        ``topk_ids`` must be logical ids (log2phy is refused at init), and
        this must run after the profile-run randomisation so both halves see
        the same ids. The result is a named pair, so a call site cannot bind
        the two tensors the wrong way round.
        """
        assert_routing_dtypes(topk_ids, topk_weights, "kt_offload route")
        for name, dtype, baked in (
            ("topk_ids", topk_ids.dtype, self._slot_tables),
            ("topk_weights", topk_weights.dtype, self._keep_tables),
        ):
            if baked and dtype not in baked:
                raise TypeError(
                    f"kt_offload route: no {name} table baked for {dtype}; the placement tables were built "
                    f"for {sorted(str(d) for d in baked)} in process_weights_after_loading."
                )
        # Before anything is overwritten: the previous route's merge must have
        # happened exactly once. Placed after the dtype checks above so a route
        # that refused its arguments never arms the counter.
        self._assert_merge_balance(hidden_states)
        self._merge_owed = True
        self._merges_since_route = 0
        self._pending = None
        self._pending_captured = False
        self._pending_own_out = False
        # Cleared on every route, not only in the captured branch: a forward
        # that captured and then runs eager would carry a stale cross-stream
        # dependency. ``_side_inflight`` deliberately survives this reset and is
        # cleared only by _join_side_stream, so a layer whose side-stream work
        # was never joined still reaches _claim_side_buffers.
        self._pending_join = None
        if self.wrapper is not None and hidden_states.shape[0] > 0:
            if hidden_states.dtype != torch.bfloat16:
                raise TypeError(f"kt_offload: kt takes bf16 activations, got {hidden_states.dtype}")
            # A3: what kt would READ as incremental for this submit. The knob
            # (A1) says deferral cannot be produced; this says none was seen.
            _assert_no_pending(self.wrapper, self.layer_idx, self.layer_idx - 1, "before this layer's submit")
            if _capturing():
                # No runtime condition here (not even the profile flag): a
                # branch the capture run skips is missing from every replay.
                hs = hidden_states.contiguous()
                # P1b: the one increment per replay, on the compute stream,
                # before anything this step copies. Lowest signal layer only, so
                # every layer's word carries the SAME number -- which is what
                # makes "this word is not this step's" detectable at all.
                # Ordered in front of every layer's counter copy by
                # construction: on the compute stream those copies come later,
                # on the side stream they sit behind a fork event this layer
                # records after its own fills.
                self._ops_prefix = []
                if self.device_signal and self.layer_idx == min(_SIGNAL_LAYERS):
                    _STEP_COUNTER.bump()
                    self._ops_prefix = [("bump", self.layer_idx)]
                # overlap first, packed second: with both on the side stream
                # uses the same packed per-layer buffers and only the stream
                # differs. Testing packed first would make "side stream on"
                # silently mean "packed serial".
                if self.overlap:
                    self._submit_captured_forked(hs, topk_ids, topk_weights)
                    self._pending_own_out = True
                elif self.packed:
                    self._packed_submit(hs, topk_ids, topk_weights)
                    stream = torch.npu.current_stream(hs.device)
                    _subscribe_stream(stream)
                    self._launch_cpu_node(stream, hs.shape[0])
                    self._pending_own_out = True
                else:
                    self._submit_captured(hs, topk_ids, topk_weights)
                self._pending, self._pending_captured = hs, True
            elif not (in_profile_run and self.cfg.skip_cpu_in_profile_run):
                # Eager is never forked, whatever overlap_side_stream says (the
                # reference implementation does the same; see "Why there is no
                # eager fork" below for why a fork buys nothing here).
                hs = hidden_states.contiguous()
                if self.packed:
                    self._packed_submit(hs, topk_ids, topk_weights)
                    # Wait for our own copy only: kt's submit_forward calls
                    # torch.npu.synchronize(device), which waits for every stream
                    # on the device, twice per layer (again in sync_forward). Our
                    # per-layer buffers remove the slot-reuse reason for that.
                    _wait_for_copy(hs.device)
                    _kt_packed_forward(self._packed_payload(hs.shape[0]))
                else:
                    self.wrapper.submit_forward(hs, topk_ids, topk_weights, _stream_handle(hs.device))
                self._pending = hs
        return split_routing(
            topk_ids,
            topk_weights,
            self._slot_tables[topk_ids.dtype],
            self._keep_tables[topk_weights.dtype],
        )

    def _assert_merge_balance(self, hidden_states: torch.Tensor) -> None:
        """Exactly one merge() call per route, whoever makes it.

        The failure is silent. ``merge()`` is the only place the CPU half of the
        experts is added, and with ``VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW`` on
        it is called from a different file (fused_moe._forward_impl) than the one
        that routes (routed_experts.forward_impl). Forgetting it yields the
        resident half alone -- right shape, plausible values, nothing raised;
        calling it twice is wrong even though the second call happens to find
        nothing pending today.

        Checked at the NEXT route rather than at the second merge, because
        "merge with nothing pending is a no-op" is an existing contract (the
        profile-run and all-resident paths rely on it). One layer's step of delay
        is enough, and in graph mode route/merge run once, at capture.

        Carve-out: with the side stream on, a capture that submitted and was
        never joined is refused more specifically by ``_claim_side_buffers``
        below, which the submit always reaches in that state.
        """
        if not self._merge_owed:
            return
        merges = self._merges_since_route
        report = shared_window.merge_balance_ok(merges, layer_idx=self.layer_idx)
        if report["ok"]:
            return
        if (
            merges == 0
            and self._side_inflight
            and _capturing()
            and self.wrapper is not None
            and hidden_states.shape[0] > 0
        ):
            return
        raise AssertionError(
            f"kt_offload: layer {self.layer_idx} reached route() with {merges} merge call(s) since its "
            f"previous route, expected exactly 1 ({report['why']}). With "
            f"{shared_window.ENV_NAME}=1 the merge is made by fused_moe._forward_impl after the "
            "shared-expert MLP; with it off by routed_experts.forward_impl. Exactly one of them must run."
        )

    def merge(self, routed_out: torch.Tensor) -> torch.Tensor:
        """Add the CPU half at the routed-output point (already scaled weights)."""
        # Counted for every call, including the no-op ones (profile run,
        # all-resident layer, nothing pending): what has to be exactly one is the
        # number of CALLS. See _assert_merge_balance.
        self._merges_since_route += 1
        hs, self._pending = self._pending, None
        if hs is None:
            return routed_out
        if routed_out.shape != hs.shape:
            raise AssertionError(f"kt_offload: routed_out {tuple(routed_out.shape)} vs kt input {tuple(hs.shape)}")
        if self._pending_captured:
            self._pending_captured = False
            # The side stream's host-func node must have returned before the
            # H2D reads the pinned output: join first, at every replay.
            self._join_side_stream(hs.device)
            # H2D from the pinned output the host-func node filled; stream
            # order puts it after that node at every replay. ``_pending_own_out``
            # (not ``self.packed``) says whose buffer that is: the forked path
            # uses our per-layer buffers whether or not packed_transfer is on.
            own_out, self._pending_own_out = self._pending_own_out, False
            cpu_out = self._packed_output(hs) if own_out else self.wrapper.copy_forward_output_to_device(hs)
        elif self.packed:
            cpu_out = self._packed_output(hs)
        else:
            cpu_out = self.wrapper.sync_forward(hs, _stream_handle(hs.device))
        # A4: after this layer drained, its own flag must be clear. A stale
        # True survives an exception on kt's pre-submit checks and then makes
        # the NEXT drain wait for one task fewer -- an early return, not a
        # raise, so nothing else would notice.
        _assert_no_pending(self.wrapper, self.layer_idx, self.layer_idx, "after this layer drained")
        # One rounding to the activation dtype instead of two.
        return (routed_out.float() + cpu_out.float()).to(routed_out.dtype)

    def _packed_output(self, hs: torch.Tensor) -> torch.Tensor:
        """H2D of our own pinned output; ordered after the host-func node."""
        buf = self._packed[hs.shape[0]]
        _host_copy(buf["dev_out"], buf["pin_out"])
        if self.device_signal and self._submit_ops is not None:
            # Same place, same reason as the P1a criterion below: the whole
            # recorded shape exists only now. What it adds is the check that P1b
            # actually happened (**no host-func node**) and the order of the two
            # copies, which is what makes the start signal safe at all.
            ops = [*self._submit_ops, ("h2d", hs.shape[0])]
            self._submit_ops = None
            report = device_signal.signal_ops_ok(ops)
            first = min(_SIGNAL_LAYERS) if _SIGNAL_LAYERS else self.layer_idx
            bumped = any(k == "bump" for k, _ in ops)
            if bumped != (self.layer_idx == first):
                raise RuntimeError(
                    f"kt_offload device signal: layer {self.layer_idx} "
                    f"{'emitted' if bumped else 'did not emit'} the step counter's increment, but the "
                    f"layer that must emit it (and only it) is {first}. Two increments per replay make "
                    "every word run ahead of its expected step; none makes every replay wait for ever."
                )
            if not report["ok"]:
                raise RuntimeError(
                    f"kt_offload device signal: layer {self.layer_idx} recorded the wrong node sequence "
                    f"({report['detail']}): {report['why']}."
                )
            if _capturing() and self._steps_at_capture is not None:
                grew = device_signal.STEPS.get(self.layer_idx, 0) - self._steps_at_capture
                self._steps_at_capture = None
                if grew:
                    raise RuntimeError(
                        f"kt_offload device signal: the polling thread ran layer {self.layer_idx}'s CPU "
                        f"experts {grew} time(s) DURING its capture. The counter copy was executed instead "
                        "of recorded, so every replay would reuse that one CPU output -- faster and wrong."
                    )
        if self.async_wait and self._submit_ops is not None:
            # The whole recorded shape exists only now, so the criterion runs
            # here. It is the only in-process evidence that the RESET went into
            # the graph next to the wait: without it every replay after the first
            # reads the previous step's CPU output, with nothing raised.
            ops = [*self._submit_ops, ("h2d", hs.shape[0])]
            self._submit_ops = None
            report = async_wait.ext_event_ops_ok(ops)
            if not report["ok"]:
                raise RuntimeError(
                    f"kt_offload async wait: layer {self.layer_idx} recorded the wrong node sequence "
                    f"({report['detail']}): {report['why']}."
                )
        return buf["dev_out"]

    def _submit_captured(self, hs: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
        """Record D2H + host-func for this layer into the graph being captured."""
        if not self.graph_enabled:
            raise RuntimeError("kt_offload: a capture reached a layer built for eager mode (enforce_eager on?)")
        bs = hs.shape[0]
        if bs not in _kt_buffer_class().capture_buffers:
            raise RuntimeError(
                f"kt_offload: capturing {bs} tokens but kt has no pre-allocated buffers for that size "
                f"(prepared: {self.capture_sizes}). Allocating inside the capture is not possible."
            )
        stream = torch.npu.current_stream(hs.device)
        _subscribe_stream(stream)
        self.wrapper.copy_inputs_to_cpu_buffers(hs, topk_ids, topk_weights)
        # kt reads only the shape of this tensor at replay; a meta tensor keeps
        # no graph-pool memory alive.
        shape_ref = torch.empty(tuple(hs.shape), dtype=hs.dtype, device="meta")
        payload = (self.wrapper, shape_ref, int(stream.npu_stream), self.layer_idx)
        _CAPTURED_PAYLOADS.append(payload)
        _launch_host_func(stream, _kt_host_forward, payload)

    # -- side-stream overlap (cfg.overlap_side_stream) ----------------------

    # Why there is no eager fork: replacing kt's device-wide
    # torch.npu.synchronize() with side.synchronize() buys nothing (the side
    # stream's first op waits on an event recorded on the compute stream *after*
    # the previous layer's H2D, so it transitively waits for that stream anyway)
    # and puts kt's pinned-slot reuse (slot = layer_idx % 2) and its temp-buffer
    # reallocation on a bs change outside the protection of that device sync.

    def _side_buffers(self, bs: int, device) -> dict:
        """This layer's OWN device + pinned buffers for the side-stream path.

        The same per-layer buffers ``packed_transfer`` uses. Two independent
        reasons:

        * **A6.** kt hands layer L the pinned slot ``L % 2``, so L and L+2 share
          one buffer; the serial path is ordered by being on one stream, the
          forked one has to arrange it.
        * **Lifetime.** kt's ``copy_inputs_to_cpu_buffers`` reads
          ``topk_weights`` *from the side stream*, and the caller drops its last
          reference the moment ``route()`` returns (routed_experts.py); inside a
          capture the allocator may hand that address to a later allocation whose
          writes are ordered against the compute stream only. Nothing the side
          stream reads may come from the graph's pool -- the compute stream
          copies every input into these buffers *before* the fork event.
        """
        buf = self._packed.get(bs)
        if buf is not None:
            return buf
        if _capturing():
            raise RuntimeError(
                f"kt_offload side stream: capturing {bs} tokens but no side buffer was prepared for that "
                f"size (prepared: {sorted(self._packed)}); allocating inside a capture is not possible."
            )
        return self._packed_buffers(bs, device)

    def _submit_captured_forked(self, hs: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
        """Fill our own buffers on the compute stream, then fork the D2H + host-func.

        ``fork.record(compute)`` must come **after** the three ``_device_fill``
        calls, so the side stream's wait on that event orders every one of them
        before anything the side stream does. Record the fork first and the side
        stream reads the *previous step's* routing ids whenever it gets there
        first -- a read racing a compute-stream write with no event between them,
        bit-wrong and silent.
        """
        if not self.graph_enabled:
            raise RuntimeError("kt_offload: a capture reached a layer built for eager mode (enforce_eager on?)")
        device = hs.device
        bs = hs.shape[0]
        buf = self._side_buffers(bs, device)
        compute, side = torch.npu.current_stream(device), _side_stream(device)
        self._claim_side_buffers(buf)
        # Positive control for the silent non-replay failure: if this layer's
        # callback runs between here and its merge while we are still capturing,
        # it was EXECUTED instead of recorded and every replay would reuse that
        # one CPU output. merge() checks the delta (side_stream.capture_hostfunc_ok).
        self._hostfunc_at_capture = HOST_FUNC_CALLS.get(self.layer_idx, 0)
        # Device-side fills, on the compute stream, from tensors whose storage
        # is only guaranteed to be ours while this call runs. The ids cast to
        # int64 happens here too, in a buffer that predates the capture.
        _device_fill(buf["dev_hidden"], hs)
        _device_fill(buf["dev_ids"], topk_ids)
        _device_fill(buf["dev_weights"], topk_weights)
        fork, join = _new_event(device), _new_event(device)
        _SIDE_EVENTS.append((fork, join))  # the graph's nodes reference these
        fork.record(compute)  # AFTER the fills: that is what orders them
        side.wait_event(fork)
        with _stream_ctx(side, device):
            # The capture must follow the event wait onto the side stream;
            # otherwise everything below runs once, at capture time, and every
            # replay reuses that one CPU output -- faster and wrong, which no
            # timing check can see.
            if not _capturing():
                raise RuntimeError(
                    "kt_offload: the graph capture did not propagate to the side stream, so the CPU-expert "
                    "D2H and host-func node would not be recorded (replays would reuse the capture run's "
                    "output). Run with overlap_side_stream=false, or with a torch_npu whose capture follows "
                    "an event wait onto another stream."
                )
            _host_copy(buf["pin"], buf["dev"])  # the one D2H, our own buffer
            self._launch_cpu_node(side, bs)
        self._pending_join = (join, side, compute)

    def _claim_side_buffers(self, buf: dict) -> None:
        """A6, across steps: our own buffers, and the previous submit joined.

        * **Sharing.** Every store the side stream touches must belong to this
          layer alone. kt's pinned pair is handed out as ``layer_idx % 2``, so
          going back to those would put L and L+2 on one store, and this registry
          would say so.
        * **Across steps.** ``route()`` clears ``_pending_join`` on the way in,
          so ``_side_inflight`` is cleared **only** by ``_join_side_stream``: a
          second submit before that join, in this step or any later one, lands
          here. The compute stream waits for this layer's join inside this
          layer's merge, so the next step's fills sit behind it.
        """
        if self._side_inflight:
            raise AssertionError(
                f"kt_offload side stream: layer {self.layer_idx} is submitting again while its previous "
                "side-stream work has not been joined. Its own buffers would be overwritten while the "
                "host-func node still has to read them, and nothing orders the two."
            )
        for name in ("dev", "pin", "pin_out", "dev_out"):
            store = buf.get(name)
            if store is None:
                raise AssertionError(f"kt_offload side stream: layer {self.layer_idx} has no {name!r} buffer")
            owner = _SIDE_BUFFER_OWNER.setdefault(id(store), self.layer_idx)
            if owner != self.layer_idx:
                raise AssertionError(
                    f"kt_offload side stream: layers {owner} and {self.layer_idx} share the {name!r} buffer. "
                    f"Per-layer buffers are what replaced kt's pinned slot sharing "
                    f"(layer_idx % {side_stream.KT_BUFFER_DEPTH}); with them shared again nothing orders "
                    "one layer's write against the other's read."
                )
        self._side_inflight = True

    def _release_side_buffers(self) -> None:
        self._side_inflight = False

    def _check_capture_hostfunc(self) -> None:
        """No callback of this layer may have RUN during its capture."""
        before = self._hostfunc_at_capture
        self._hostfunc_at_capture = None
        if before is None or not _capturing():
            return
        report = side_stream.capture_hostfunc_ok(
            {self.layer_idx: before}, {self.layer_idx: HOST_FUNC_CALLS.get(self.layer_idx, 0)}
        )
        if not report["ok"]:
            raise RuntimeError(
                f"kt_offload side stream: layer {self.layer_idx}'s host func ran during the capture "
                f"({report['detail']}), so it was executed, not recorded: {report['why']}."
            )

    def _ids_for_kt(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """int64 ids in a buffer that predates the capture (see _prepare_capture_buffers).

        The copy is enqueued on the caller's stream (the compute stream), not
        inside the side-stream context, so nothing allocates or writes there.

        The forked path does not call this: it fills its own int64 buffer before
        the fork, because a fill enqueued after ``fork.record(compute)`` is
        ordered against nothing. This stays as the guard for any captured path
        that hands kt its own ids; ``_submit_captured`` does not, so kt's
        ``.to(torch.long)`` there allocates on the capture stream, which is
        allowed.
        """
        if topk_ids.dtype == torch.int64:
            return topk_ids
        buf = self._ids_i64.get(topk_ids.shape[0])
        if buf is None:
            raise RuntimeError(
                f"kt_offload: no preallocated int64 id buffer for {topk_ids.shape[0]} tokens "
                f"(prepared: {sorted(self._ids_i64)}); a cast inside the capture would allocate."
            )
        buf.copy_(topk_ids)
        return buf

    def _join_side_stream(self, device) -> None:
        """Make the compute stream wait for this layer's side-stream work."""
        pending, self._pending_join = self._pending_join, None
        self._check_capture_hostfunc()
        self._release_side_buffers()
        if pending is None:
            if self.overlap:
                raise AssertionError(
                    "kt_offload: merge() found no side-stream join to record although overlap is on; "
                    "the H2D would read the pinned output before the host func filled it."
                )
            return
        join, side, compute = pending
        join.record(side)
        compute.wait_event(join)


# There is deliberately no start-up probe for the captured fork/join path: a
# probe built from a scalar copy and a trivial host function is weaker than its
# subject and passes where the subject fails. The switch is refused in config
# instead. A probe may come back only if it reproduces both known failures
# (ACL 107025 and the silent no-replay).


def _kt_packed_forward(payload) -> None:
    """Run this layer's CPU experts straight on our combined pinned buffer.

    kt's own entry points read its three pinned allocations; here the same C++
    task is handed our three pointers into one buffer, so the step paid one
    host-device copy instead of three. Exceptions must not escape a graph node.

    Four properties of that C++ task this relies on:
      * the arguments are positional and ``qlen`` is a *pointer* to an int32
        batch size, not the number (ext_bindings.cpp ForwardBindings::Args);
      * every buffer must stay alive and unchanged until the task has run --
        ours live in ``self._packed`` for the process, the D2H before this node
        and the H2D after it are on the same stream, and each layer has its own
        buffer, so kt's ``layer_idx % 2`` slot sharing does not apply to us;
      * ``incremental=False`` overwrites the output (True accumulates into it);
      * kt's queue is one serial FIFO worker and re-raises a task's exception
        only at the next sync, so the submit and the sync stay in the same node
        here; a layer that submits without syncing hides its errors for ever.
    """
    wrapper, buf, layer_idx = payload
    report_affinity.apply_in_callback()  # no-op unless the switch is on; once per process
    # Seam timing, same segments as _kt_host_forward. Here the submit and the
    # sync are ours, so the split needs no CPUInfer probe: ``kt_pre`` is the
    # forward_task build, ``kt_tail`` is empty by construction.
    rec = seam_timing.RECORDER
    ts = seam_timing.TS
    if rec is not None:
        # CLOCK_REALTIME first: it is what the profile's HOSTFUNC_CALLBACK.ts
        # is joined against (A = this - that), so it is the earliest stamp.
        ts[seam_timing.T_ENTER_RT] = clock_gettime_ns(CLOCK_REALTIME)
        ts[seam_timing.T_SUBMIT_IN] = 0
        ts[seam_timing.T_ENTER] = perf_counter_ns()
    try:
        # A3/A4 per replay: in graph mode route()/merge() run once, at capture,
        # so this node is the only place a per-step check exists. Raising here
        # is caught below and logged -- the graph has nowhere to return to.
        _assert_no_pending(wrapper, layer_idx, layer_idx - 1, "before this node's submit")
        if rec is not None:
            ts[seam_timing.T_PRE] = perf_counter_ns()
        task = wrapper.moe.forward_task(
            buf["bsz"].data_ptr(),
            buf["pin_ids"].shape[-1],
            buf["pin_ids"].data_ptr(),
            buf["pin_weights"].data_ptr(),
            buf["pin_hidden"].data_ptr(),
            buf["pin_out"].data_ptr(),
            False,  # incremental: no deferred experts
        )
        if rec is None:
            wrapper.cpu_infer.submit(task)
            wrapper.cpu_infer.sync(0)
        else:
            ts[seam_timing.T_SUBMIT_IN] = perf_counter_ns()
            wrapper.cpu_infer.submit(task)
            ts[seam_timing.T_SUBMIT_OUT] = ts[seam_timing.T_SYNC_IN] = perf_counter_ns()
            wrapper.cpu_infer.sync(0)
            ts[seam_timing.T_SYNC_OUT] = ts[seam_timing.T_KT_OUT] = perf_counter_ns()
        _assert_no_pending(wrapper, layer_idx, layer_idx, "after this node drained")
        HOST_FUNC_CALLS[layer_idx] = HOST_FUNC_CALLS.get(layer_idx, 0) + 1
        if rec is not None:
            rec.record(layer_idx, ts, perf_counter_ns())
    except BaseException as exc:  # noqa: BLE001
        logger.error(
            "kt_offload packed host-func for layer %d raised: %r. The CPU-expert output of this step is "
            "NOT this step's -- the device reads whatever the pinned buffer held.",
            layer_idx,
            exc,
        )


def _kt_packed_submit_only(payload) -> None:
    """P1a graph node: build kt's task, submit it, return. Do NOT wait.

    The D2H of the combined input buffer is the op immediately before this node
    on the same stream. Against ``_kt_packed_forward`` only the waiting moves:
    the task is handed to the completion thread, which syncs and records the
    ExternalEvent the next two graph nodes wait on and reset.

    Every exit path must hand the event over, or the device waits on an event
    nobody records and the replay never finishes. So a failure before the submit
    still enqueues ``wrapper=None`` ("nothing to wait for, let it through"),
    which degrades to exactly the failure the serial node already has -- the H2D
    reads the previous step's CPU output.
    """
    wrapper, buf, layer_idx, event = payload
    report_affinity.apply_in_callback()  # no-op unless the switch is on; once per process
    rec = seam_timing.RECORDER
    ts = seam_timing.TS
    if rec is not None:
        ts[seam_timing.T_ENTER_RT] = clock_gettime_ns(CLOCK_REALTIME)
        ts[seam_timing.T_SUBMIT_IN] = 0
        ts[seam_timing.T_ENTER] = perf_counter_ns()
    handed_over = False
    try:
        _assert_no_pending(wrapper, layer_idx, layer_idx - 1, "before this node's submit")
        if rec is not None:
            ts[seam_timing.T_PRE] = perf_counter_ns()
        task = wrapper.moe.forward_task(
            buf["bsz"].data_ptr(),
            buf["pin_ids"].shape[-1],
            buf["pin_ids"].data_ptr(),
            buf["pin_weights"].data_ptr(),
            buf["pin_hidden"].data_ptr(),
            buf["pin_out"].data_ptr(),
            False,  # incremental: no deferred experts
        )
        if rec is not None:
            ts[seam_timing.T_SUBMIT_IN] = perf_counter_ns()
        wrapper.cpu_infer.submit(task)
        _COMPLETER.submit(wrapper, layer_idx, event)
        handed_over = True
        HOST_FUNC_CALLS[layer_idx] = HOST_FUNC_CALLS.get(layer_idx, 0) + 1
        if rec is not None:
            # kt's compute (segment D) and the drain (segment E) happen on the
            # completion thread now, so the sync stamps collapse onto the submit
            # and the D/E columns read ~0 BY CONSTRUCTION, not because the CPU
            # got faster.
            ts[seam_timing.T_SUBMIT_OUT] = ts[seam_timing.T_SYNC_IN] = perf_counter_ns()
            ts[seam_timing.T_SYNC_OUT] = ts[seam_timing.T_KT_OUT] = ts[seam_timing.T_SYNC_IN]
            rec.record(layer_idx, ts, perf_counter_ns())
    except BaseException as exc:  # noqa: BLE001
        logger.error(
            "kt_offload async-wait host-func for layer %d raised: %r. The CPU-expert output of this step "
            "is NOT this step's -- the device reads whatever the pinned buffer held.",
            layer_idx,
            exc,
        )
    finally:
        if not handed_over:
            # Nothing was submitted (or the hand-off itself failed): release the
            # device rather than leaving the graph waiting for ever.
            try:
                _COMPLETER.submit(None, layer_idx, event)
            except BaseException as exc:  # noqa: BLE001
                logger.critical(
                    "kt_offload async wait: layer %d could not hand its ExternalEvent to the completion "
                    "thread (%r); this replay will not finish.",
                    layer_idx,
                    exc,
                )


def _kt_host_forward(payload) -> None:
    """Graph-node callback: run this layer's CPU experts on the pinned buffers.

    Exceptions must not escape (nothing on the ACL side receives them); the
    device would then read the previous step's CPU output. Log loudly.
    """
    wrapper, shape_ref, stream_handle, layer_idx = payload
    report_affinity.apply_in_callback()  # no-op unless the switch is on; once per process
    # Seam timing (VLLM_ASCEND_DSV41_KT_SEAM_TIMING): off -> RECORDER is None
    # and this costs one global read and one branch per segment boundary. On,
    # it reads clocks and stores integers; it changes no argument, no order
    # and no device operation. See seam_timing.py for what each segment is.
    rec = seam_timing.RECORDER
    ts = seam_timing.TS
    if rec is not None:
        # CLOCK_REALTIME first: it is what the profile's HOSTFUNC_CALLBACK.ts
        # is joined against (A = this - that), so it is the earliest stamp.
        ts[seam_timing.T_ENTER_RT] = clock_gettime_ns(CLOCK_REALTIME)
        ts[seam_timing.T_SUBMIT_IN] = 0
        ts[seam_timing.T_ENTER] = perf_counter_ns()
    try:
        _assert_no_pending(wrapper, layer_idx, layer_idx - 1, "before this node's submit")
        if rec is not None:
            ts[seam_timing.T_PRE] = perf_counter_ns()
        wrapper.run_pinned_forward_sync(shape_ref, stream_handle)
        if rec is not None:
            ts[seam_timing.T_KT_OUT] = perf_counter_ns()
        _assert_no_pending(wrapper, layer_idx, layer_idx, "after this node drained")
        HOST_FUNC_CALLS[layer_idx] = HOST_FUNC_CALLS.get(layer_idx, 0) + 1
        if rec is not None:
            # Taken before the store, so the bookkeeping this module costs is
            # not counted inside the callback it measures.
            rec.record(layer_idx, ts, perf_counter_ns())
    except BaseException as exc:  # noqa: BLE001
        logger.error(
            "kt_offload host-func for layer %d raised inside a graph node: %r. The CPU-expert output "
            "of this step is NOT this step's -- the device reads whatever the pinned buffer held.",
            layer_idx,
            exc,
        )


# -- platform indirections (monkeypatched by the CPU unit tests) -------------


def _capturing() -> bool:
    """Is the current stream being captured?

    Same predicate as the engram host gather
    (models/deepseek_v4_1/engram_embedding.py _stream_is_capturing). Copied,
    not imported: importing that package applies the V4.1 platform patches.
    ``is True`` so a test double returning a Mock does not read as capturing.
    """
    predicate = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    if predicate is None:
        return False
    return predicate() is True


_REPORT_SUBSCRIBED: set[int] = set()


def _subscribe_stream(stream) -> None:
    """Subscribe the capture stream for host-func reports; swallow only 107011.

    torch_npu >= 2.10 already subscribes a capture stream inside NPUGraph, and
    engram may have subscribed it too; a repeat subscribe raises 107011, which
    means "already subscribed". Anything else propagates.
    """
    import torch_npu

    key = int(stream.npu_stream)
    if key in _REPORT_SUBSCRIBED:
        return
    try:
        torch_npu.npu._subscribe_report(stream)
    except RuntimeError as exc:
        if "107011" not in str(exc):
            raise
    _REPORT_SUBSCRIBED.add(key)




# -- the no-cross-layer contract, read off kt's own bookkeeping --------------
#
# kt keeps one process-level dict, ``BaseMoEWrapper._layer_has_pending_deferred``,
# keyed by bare layer_idx (experts_base.py:411). Three facts follow, and the
# checks below read the dict through kt's public accessor rather than the
# switch that is supposed to keep it empty:
#   * a submit reads the PREVIOUS layer's flag and turns it into
#     ``incremental=True`` (:667/:744/:819) -- a silent accumulation across a
#     layer boundary, which this deployment forbids;
#   * a layer's own flag sets its drain allowance to 1 (:771/:942), i.e. the
#     sync waits for one task FEWER. That failure returns early instead of
#     raising, and the output read afterwards is half written;
#   * the flag is written False only after a successful immediate submit
#     (:697/:755/:844), while the checks before it (`_check_qlen_fits_cpp_buffers`
#     at :593/:650/:728, `_wait_device` at :817) can raise first and leave a
#     stale True behind.
# A1 (max_deferred_experts_per_token == 0) says deferral cannot be *produced*;
# these say nothing was *read*. Both are kept: they fail in different places.
_PENDING_API_WARNED = False
# weak refs: a wrapper that has been dropped no longer writes kt's flags, and
# a test (or a rebuilt model) may legitimately construct a fresh one for the
# same index. What must never happen is two LIVE wrappers on one index.
_WRAPPER_REGISTRY: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()


def _pending_reader(wrapper):
    """kt's public accessor, or None on a build that predates it."""
    global _PENDING_API_WARNED
    reader = getattr(wrapper, "has_pending_deferred", None)
    if reader is None and not _PENDING_API_WARNED:
        _PENDING_API_WARNED = True
        logger.warning(
            "kt_offload: this kt build has no has_pending_deferred(); the no-cross-layer contract is "
            "checked only by the configuration knob (A1), not by what kt actually read. Coverage of "
            "the A3/A4 checks on this build is zero."
        )
    return reader


def _assert_no_pending(wrapper, layer_idx: int, which: int, when: str) -> None:
    """Refuse to continue if kt's flag for layer ``which`` is set."""
    reader = _pending_reader(wrapper)
    if reader is None:
        return
    if not reader(which):
        return
    own = which == layer_idx
    raise RuntimeError(
        f"kt_offload: layer {layer_idx}: kt reports a pending deferred task for layer {which} {when}. "
        + (
            "Its own flag makes this layer's drain allowance 1 (experts_base.py:771/:942), so the sync "
            "returns one task early and the output read after it is half written."
            if own
            else "The next submit would read it as incremental=True (experts_base.py:667/:744/:819) and "
            "accumulate the previous layer's CPU experts into this layer's output."
        )
        + " Crossing a layer boundary is forbidden for this deployment; a stale True is also what an "
        "exception on kt's pre-submit checks leaves behind."
    )


def _reset_wrapper_registry() -> None:
    """Forget every registered wrapper. For tests that stand in for a fresh
    process; a service never calls this, and calling it would throw away the
    only record that two wrappers share an index."""
    _WRAPPER_REGISTRY.clear()


def _register_wrapper(wrapper, layer_idx: int) -> None:
    """A5/A9: the bookkeeping is process-level, so our wrappers must not share
    a layer index, and they must share one CPUInfer.

    kt's dict is keyed by the bare layer index: a second wrapper on that index
    would write flags this one then reads as "the previous layer left
    something". Two CPUInfer instances would mean two queues, so "the queue is
    serial" -- what every ordering argument here rests on -- would quietly stop
    being true.
    """
    previous = _WRAPPER_REGISTRY.get(layer_idx)
    if previous is not None and previous is not wrapper:
        raise RuntimeError(
            f"kt_offload: a second live kt wrapper was built for layer_idx {layer_idx}. kt's deferred "
            "bookkeeping is a process-level dict keyed by that index (experts_base.py:411), so the two "
            "would read each other's flags."
        )
    _WRAPPER_REGISTRY[layer_idx] = wrapper
    # Unwrap the seam-timing probe: every wrapper gets its own pass-through
    # over the *same* CPUInfer, so comparing the probes would read "two
    # queues" where there is one. The invariant is about the queue.
    infer = seam_timing.unwrap_infer(getattr(wrapper, "cpu_infer", None))
    for other_idx, other in _WRAPPER_REGISTRY.items():
        other_infer = seam_timing.unwrap_infer(getattr(other, "cpu_infer", None))
        if other is wrapper or other_infer is None or infer is None:
            continue
        if other_infer is not infer:
            raise RuntimeError(
                f"kt_offload: layer {layer_idx} and layer {other_idx} have different CPUInfer objects "
                f"({infer!r} vs {other_infer!r}). Two queues means the serial-FIFO ordering every "
                "cross-layer argument here rests on no longer holds."
            )

# ---------------------------------------------------------------------------
# The folded swap-in: the one assembly point
# ---------------------------------------------------------------------------

#: layer index -> that layer's live offload state, filled by
#: ``after_weights_loaded``. Weak on purpose: production holds every one of them
#: through its MoE module, and a unit test's stand-in does not survive into the
#: next test. The one-per-index invariant is enforced on the kt wrapper
#: (``_register_wrapper``); here the newest registration simply wins.
_OFFLOAD_LAYERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _register_offload_layer(rt: KTLayerOffload) -> None:
    _OFFLOAD_LAYERS[int(rt.layer_idx)] = rt


def registered_offload_layers() -> dict[int, KTLayerOffload]:
    """Every MoE layer whose weights have been loaded, by layer index."""
    return dict(sorted(_OFFLOAD_LAYERS.items()))


def _reset_offload_registry() -> None:
    """Forget every registered layer. For tests standing in for a fresh process."""
    _OFFLOAD_LAYERS.clear()


class _ForwardSteps:
    """Monotone step number for the swap ledger: one per model FORWARD.

    ``OwnershipLedger.claim`` refuses a commit stamped older than the last one,
    and the fold commits inside the forward, at each layer's boundary -- so every
    layer of one forward must commit under the SAME number (equal is allowed,
    going backwards is not) and the next forward must be strictly newer. A
    per-layer counter would satisfy neither.

    vLLM's runner publishes no step number here, so the count is taken on the
    first folded layer, which every forward reaches once and before every other
    folded layer.
    """

    def __init__(self, first_layer: int):
        self.first_layer = int(first_layer)
        self._n = 0

    def advance(self, layer_idx: int) -> None:
        if int(layer_idx) == self.first_layer:
            self._n += 1

    def value(self) -> int:
        return self._n


def offload_layers_to_install(layers=None, *, what: str = "fold") -> dict[int, KTLayerOffload]:
    """The layers an installer works on: the caller's list, or the registry.

    Empty is a refusal, never an empty install: the registry is filled by
    ``after_weights_loaded``, so "nothing registered" means the installer ran
    before the weights were loaded.
    """
    if layers is None:
        offloads = registered_offload_layers()
    else:
        offloads = {int(rt.layer_idx): rt for rt in layers}
    if not offloads:
        raise RuntimeError(
            f"kt_offload: no MoE layer has registered its offload state, so there is nothing to {what}. "
            "The installers run after the model's weights are loaded "
            "(process_weights_after_loading), not before.")
    return offloads


def refuse_after_kv_profile(offloads: dict[int, KTLayerOffload], what: str) -> None:
    """Everything the streaming path allocates must precede the KV profile.

    ``what`` is allocated here (the fold's ND staging, the provider's slot), and
    memory taken after the profile is memory vLLM has already promised to the KV
    cache. A layer whose ``cache_config`` already reports ``num_gpu_blocks`` is
    a layer whose profile has been run.
    """
    for idx, rt in offloads.items():
        blocks = getattr(rt._cache_config, "num_gpu_blocks", None)
        if blocks:
            raise RuntimeError(
                f"kt_offload: layer {idx}: the KV cache has already been sized ({blocks} blocks), so the "
                f"profile run is over. {what} would come out of memory vLLM has already promised to the "
                "KV cache. Install it before the profile run.")


def install_stream_provider(provider, *, layers=None) -> dict[int, KTLayerOffload]:
    """Install the ONE streaming slot provider on every MoE layer (no swap-in).

    ``cfg.stream_prefill`` without ``cfg.dynamic_resident``: the layers need the
    callback and the provider object, and nothing else. One provider per MODEL,
    told which layer is streaming through ``set_layer`` -- one per layer would be
    40 slots of HBM, and ``maybe_stream_prefill`` binds the layer on every
    forward anyway.

    ``install_resident_fold`` ends with this same installation, so the two paths
    cannot drift apart in what "wired" means.
    """
    offloads = offload_layers_to_install(layers, what="stream")
    for need in ("slot_for", "set_layer"):
        if getattr(provider, need, None) is None:
            raise RuntimeError(
                f"kt_offload: the slot provider has no {need!r}; the same object serves the streamed chunks "
                "and is read back by the fold, so it must be the streaming StreamSlotProvider")
    for rt in offloads.values():
        rt._stream_provider = provider
        rt._stream_slot_provider = provider.slot_for
    return offloads


def install_resident_fold(provider, *, layers=None, step_source=None, **fold_kw):
    """Build the ONE folded swap for the whole model and install it in every layer.

    ``ResidentFold`` is per MODEL and ``KTLayerOffload`` is per LAYER: the fold
    owns a single ND staging of ``[num_slots, ...]`` that every layer reuses,
    because one staging per layer would not fit in HBM. So it is built once, from
    every layer's targets, and the same object is handed to all of them.

    ``provider`` is the streaming ``StreamSlotProvider``, and the SAME object is
    installed as each layer's slot callback: the fold copies the chosen experts
    back out of the buffers that provider just served, so a second provider would
    hand it another layer's bytes with every check green.

    **Call this before the KV-cache profile run** -- the staging and the
    provider's slot are allocated here, and memory taken after the profile is
    memory vLLM has already promised to the KV cache; this refuses when
    ``cache_config`` already reports ``num_gpu_blocks``. Call it after every MoE
    layer's ``process_weights_after_loading`` (which fills the registry below)
    and before any graph capture.
    """
    from vllm_ascend.kt_offload import prefill_window
    from vllm_ascend.kt_offload.fold_swap import ResidentFold  # only the dynamic path needs it

    offloads = offload_layers_to_install(layers, what="fold")
    for need in ("slot_for", "set_layer"):
        if getattr(provider, need, None) is None:
            raise RuntimeError(
                f"kt_offload: the slot provider has no {need!r}; the same object serves the streamed chunks "
                "and is read back by the fold, so it must be the streaming StreamSlotProvider")
    refuse_after_kv_profile(offloads, "The fold's ND staging is allocated here and")
    placements = {idx: rt.placement for idx, rt in offloads.items()}
    targets = {idx: rt.build_swap_targets(rt._module) for idx, rt in offloads.items()}
    steps = _ForwardSteps(min(offloads))
    # The streaming threshold travels from the layers to the fold HERE and is
    # read nowhere else. ``maybe_stream_prefill`` declines a forward of fewer
    # than this many tokens, so the fold's "last chunk" and the prompt's last
    # chunk are not the same forward whenever the tail is short -- most long
    # prompts (fold_swap._decides_now). Taken off the same ``rt.cfg`` objects
    # whose gate it is: a second reading of the config (here or in
    # prefill_window) would be a second truth the commit would depend on.
    thresholds = {int(rt.cfg.stream_prefill_min_tokens) for rt in offloads.values()}
    if len(thresholds) != 1:
        raise RuntimeError(
            f"kt_offload: the MoE layers disagree on stream_prefill_min_tokens ({sorted(thresholds)}); one "
            "fold serves every layer and its commit point is decided by that number")
    fold = ResidentFold(
        placements, targets, provider=provider,
        stream_min_tokens=thresholds.pop(),
        step_source=steps.value if step_source is None else step_source,
        # Re-read before every commit, from the layer, never from a snapshot:
        # this is what makes "the commit writes what the forward reads" a
        # standing property rather than something that was true once.
        live_objects=lambda idx: offloads[idx].live_swap_objects(),
        **fold_kw)
    if fold.committer.live_objects is None:
        raise RuntimeError("kt_offload: the fold was built without the live-object check; refusing to install it")
    # The fold commits once per REQUEST, on the last chunk of its prefill, and
    # only the scheduler output knows which chunk that is. Armed here, so with
    # dynamic_resident off the runner's one line costs a bool test
    # (prefill_window.observe).
    prefill_window.arm()
    # The window module is the only thing that sees every step, so only it can
    # notice a prefill that ended having committed nothing -- including one whose
    # every forward was too short to stream and never reached the fold at all.
    prefill_window.on_prefill_end(fold.note_prefill_end)
    for rt in offloads.values():
        rt._resident_fold = fold
        rt._steps = steps if step_source is None else None
    # The same installation the stream-only path performs: one provider, on
    # every layer, reached through the same two attributes.
    install_stream_provider(provider, layers=list(offloads.values()))
    logger.info(
        "kt_offload: folded swap-in installed on layers %s (one staging of %d slots, shared; step source %s)",
        sorted(offloads), fold.num_slots, "forward counter" if step_source is None else "caller-supplied")
    return fold


def _refuse_cross_layer(wrapper, layer_idx: int) -> None:
    """Hard constraint: nothing may cross a layer boundary.

    Deferring part of a token's CPU experts to a later layer or step is
    forbidden, so kt's deferral must be off. It defaults to 0 and we never ask
    for it, but this reads it off the object that would do the deferring, at the
    moment it exists, and refuses rather than trusting a default.

    The other half of the same constraint, ``incremental=False`` on every
    forward_task call, is asserted where that call is made (and in
    tests/ut/kt_offload/test_packed_transfer.py).
    """
    seen = {}
    for attr in ("max_deferred_experts_per_token", "num_deferred_experts_per_token", "deferred_experts"):
        value = getattr(wrapper, attr, None)
        if value is None:
            continue
        seen[attr] = value
        try:
            nonzero = int(value) != 0
        except (TypeError, ValueError):
            nonzero = bool(value)
        if nonzero:
            raise RuntimeError(
                f"kt_offload: layer {layer_idx}: kt reports {attr}={value!r}, which defers part of a token's "
                "CPU experts past this layer. Crossing a layer boundary is forbidden for this deployment "
                "(design doc, hard constraint): the CPU contribution must be merged in the same layer that "
                "routed it."
            )
    if not seen:
        logger.warning(
            "kt_offload: layer %d: this kt build exposes no deferral attribute to check; the no-cross-layer "
            "constraint is unverified on this build (looked for max_deferred_experts_per_token and friends)",
            layer_idx,
        )

def _launch_host_func(stream, fn, payload) -> None:
    import torch_npu

    torch_npu.npu._launch_host_func(stream, fn, payload)


def _side_stream(device):
    """The one side stream of this process, created outside any capture."""
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = _new_stream(device)
    return _SIDE_STREAM


def _new_stream(device):
    return torch.npu.Stream(device=device)


def _new_event(device):
    del device
    return torch.npu.Event()


def _new_external_event(device):
    """An event the device can wait on BEFORE anyone records it.

    ``torch.npu.ExternalEvent`` is ``graph_external=True``
    (torch_npu/npu/streams.py:211-228): under ACL it is the
    ``aclrtValueWait``/``aclrtValueWrite`` pair (acl_rt.h:3802/3814), it is
    captured as a node rather than executed, and unlike a plain Event its
    ``wait`` may precede its ``record``. Same object as the one the attention
    path already captures (attention_v1.py:1036-1038, mla_v1.py:1744-1747) and
    releases from the host during replay (mla_v1.py:968).
    """
    del device
    return torch.npu.ExternalEvent()


def _ext_wait(event, stream) -> None:
    """Record ``wait`` into the graph: the device stops here until it is signalled."""
    event.wait(stream)


def _ext_reset(event, stream) -> None:
    """Record ``reset`` into the graph, right behind the wait.

    Without this node the event stays signalled after the first replay and
    every later replay's wait is a no-op, so the H2D copies the pinned output
    before this step's CPU task has written it. Faster, wrong, silent. The
    order (wait, then reset) is the one attention_v1.py:1036-1038 uses.
    """
    event.reset(stream)


def _make_completer(device):
    """The process-wide completion thread, with its own ACL context and stream."""
    index = device.index if getattr(device, "index", None) is not None else torch.npu.current_device()

    def set_device() -> None:
        # The record comes from a thread that is neither the main thread nor
        # ACL's report thread, while a replay is in flight. Without a context
        # of its own every ACL call from here fails.
        torch.npu.set_device(index)

    return async_wait.Completer(set_device, lambda: _new_stream(torch.device("npu", index)))


class _SignalWords:
    """The pinned words P1b's 8-byte copies land in, one array for the process.

    One allocation, handed out in 128-byte slots (``device_signal.STRIDE_WORDS``):
    the polling thread scans every slot while the device DMAs into one of them,
    so two layers on one cache line would mean each write invalidating the line
    the spin is reading for the other. One array rather than one tensor per
    layer because the spin loop takes a base pointer and a stride -- and
    because it must never be reallocated while that thread is reading it.
    """

    def __init__(self, capacity: int):
        # +1: the self-test gets a slot of its own. Driving a real layer's word
        # would leave it holding the step the poller is waiting for, and the
        # poller would run that layer's kt before a single replay happened.
        self.capacity = int(capacity) + 1
        self.words = _pinned_empty((self.capacity * device_signal.STRIDE_WORDS,), torch.int64)
        self.words.zero_()
        self.n = 0

    def take(self) -> tuple[int, torch.Tensor]:
        if self.n >= self.capacity:
            raise RuntimeError(
                f"kt_offload device signal: out of signal slots ({self.capacity}); the array is sized for "
                "every layer at every capture size and must not be reallocated while the polling thread "
                "reads it."
            )
        slot, self.n = self.n, self.n + 1
        off = slot * device_signal.STRIDE_WORDS
        return slot, self.words[off : off + 1]


class _StepCounter:
    """One device scalar, incremented once per replay by the lowest signal layer."""

    def __init__(self, device):
        self.value = torch.zeros((1,), dtype=torch.int64, device=device)
        # Pre-allocated addend: ``add_(1)`` would turn a Python int into a
        # tensor, and nothing may allocate inside a capture.
        self.one = torch.ones((1,), dtype=torch.int64, device=device)

    def bump(self) -> None:
        _device_add(self.value, self.one)


def _selftest_signal_path(signals: _SignalWords, counter: _StepCounter) -> None:
    """Drive the whole signal path once, eagerly, before any capture.

    Cheap, and it turns three silent hangs into start-up errors: an int64 add
    or an 8-byte D2H the platform will not do, a DMA write the host cannot see
    without a cache flush (the pinned word would just never change), and a
    compiled helper that loaded but does not agree with what torch reads.
    """
    slot, word = signals.take()
    counter.bump()
    _host_copy(word, counter.value)
    torch.npu.synchronize()
    got = int(word.item())
    if got != 1:
        raise RuntimeError(
            f"kt_offload device signal: the self-test counter copy left {got} in the pinned word, want 1. "
            "The device increment or the 8-byte D2H did not do what the whole mechanism assumes; refusing "
            "to start rather than waiting for a signal that will not come."
        )
    waiter = device_signal.Waiter(signals.words.data_ptr(), signals.capacity)
    native = waiter.read(slot)
    if native != 1:
        raise RuntimeError(
            f"kt_offload device signal: torch reads {got} in the self-test word and the polling helper "
            f"reads {native}. The thread that matters is the one that reads {native}."
        )
    waiter.set_expected(slot, 1)
    if waiter.wait(50_000_000, 0, 200_000) != slot:
        raise RuntimeError("kt_offload device signal: the polling helper did not see an already-set word")
    waiter.set_expected(slot, 2)
    if waiter.wait(20_000_000, 0, 200_000) != -1:
        raise RuntimeError(
            "kt_offload device signal: the polling helper matched a word that has not reached its step. "
            "A waiter that returns early would start kt before the inputs are there."
        )
    # Back to 0: the first replay must write step 1, which is what every
    # layer's poller entry starts out expecting.
    counter.value.zero_()
    torch.npu.synchronize()
    logger.info(
        "kt_offload device signal: self-test passed (device add -> 8-byte D2H -> pinned word -> spin loop), "
        "%d slots of %d bytes",
        signals.capacity,
        device_signal.STRIDE_WORDS * 8,
    )


def _make_poller(device):
    """The process-wide polling thread, with its own ACL context and stream.

    Not bound to a core, and there is no knob to bind it: pinning it measured
    slower rather than faster, and this thread is idle whenever a layer is not
    being signalled and goes cold after a second of nothing.
    """
    index = device.index if getattr(device, "index", None) is not None else torch.npu.current_device()

    def set_device() -> None:
        # Same requirement as the P1a completion thread: the record is an ACL
        # call from a thread that is neither the main thread nor ACL's report
        # thread, so it needs a context of its own.
        torch.npu.set_device(index)

    def make_waiter():
        return device_signal.Waiter(_SIGNALS.words.data_ptr(), _SIGNALS.capacity)

    return device_signal.Poller(set_device, lambda: _new_stream(torch.device("npu", index)), make_waiter)


def _stream_ctx(stream, device):
    del device
    return torch.npu.stream(stream)


def _handle_of(stream) -> int:
    return int(stream.npu_stream)


def _kt_buffer_class():
    from kt_kernel.experts_base import KExpertsCPUBuffer

    return KExpertsCPUBuffer


def _install_capture_probe(kt_wrapper_cls) -> None:
    """Point kt's capture check at our predicate instead of sglang's flag.

    Our captured path never calls kt's _wait_device, so this is a second line
    of defence; a kt build without the hook only gets a warning.
    """
    setter = getattr(kt_wrapper_cls, "set_capture_probe", None)
    if setter is None:
        logger.warning(
            "kt_offload: this kt-kernel has no set_capture_probe (needs ktransformers 292be81+); "
            "kt's own device sync falls back to torch.npu.is_current_stream_capturing() alone."
        )
        return
    setter(_capturing)


def _maybe_start_host_func_log(num_layers: int) -> None:
    """Start the per-layer sampler if the env var asks for it.

    Which counter it samples has to follow the path in use. With P1b on there is
    no host-func node, so ``HOST_FUNC_CALLS`` stays at 0 for ever and
    ``host_func_report`` says nothing about a run where nothing grew: the sampler
    would look *clean* on the arm it was meant to police. It samples the polling
    thread's counter instead, and the log line says which one it read.
    """
    from vllm_ascend import envs
    from vllm_ascend.kt_offload import hostfunc_log

    raw = envs.VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG
    if not raw:
        return
    try:
        interval = float(raw)
    except ValueError:
        logger.warning("kt_offload: VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=%r is not a number of seconds", raw)
        return
    if device_signal.enabled_by_env():
        hostfunc_log.start(interval, device_signal.STEPS, num_layers, what="device-signal")
    else:
        hostfunc_log.start(interval, HOST_FUNC_CALLS, num_layers)


def _pinned_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Pinned host memory: the only kind an async host-device copy may use."""
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _device_fill(dst: torch.Tensor, src: torch.Tensor) -> None:
    """A device-side copy into the combined buffer (not a host-device operation)."""
    dst.copy_(src)


def _device_add(dst: torch.Tensor, src: torch.Tensor) -> None:
    """The step counter's increment: one device-side add, once per replay.

    Its own indirection for the same reason ``_device_fill`` has one -- the CPU
    model has to see it as an op ON A STREAM, because where it sits relative to
    the counter's copy to the host is part of the protocol.
    """
    dst.add_(src)


def _host_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """One host-device copy: the operation whose count we are reducing."""
    dst.copy_(src, non_blocking=True)


def _wait_for_copy(device) -> None:
    """Host waits for the work already enqueued on this stream, nothing wider."""
    event = _new_event(device)
    event.record(torch.npu.current_stream(device))
    event.synchronize()


def _stream_handle(device: torch.device) -> int:
    stream = torch.npu.current_stream(device)
    return int(stream.npu_stream)
