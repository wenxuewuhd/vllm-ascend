# SPDX-License-Identifier: Apache-2.0
"""The one place production builds the streaming path.

The only caller that constructs a ``StreamSlotProvider`` and calls
``install_resident_fold``; the runner's side is a single line in
``vllm_ascend/worker/worker.py::NPUWorker.load_model``. Two hard boundaries put
it there, in this order:

1. **after the weights are loaded.** ``KTLayerOffload.after_weights_loaded``
   bakes the routing tables, builds the kt wrapper and registers the layer
   (``layer.py`` ``_register_offload_layer``), and runs from the model loader's
   ``process_weights_after_loading``. Before that there is no layer to collect,
   no table to point a commit at and no ``quant_method`` whose post-load hook
   can cast a chunk.
2. **before the KV-cache profile run.** The provider's device slot and the
   fold's ND staging are allocated here; memory taken after the profile is
   memory vLLM has already promised to the KV cache. The observable is
   ``cache_config.num_gpu_blocks``, and ``layer.refuse_after_kv_profile``
   refuses on it rather than trusting a comment.

``NPUWorker.load_model`` sits between them: vLLM's uniproc executor calls
``load_model()`` on the worker, and only later does the engine core call
``determine_available_memory()`` -> ``model_runner.profile_run()`` and then
``initialize_from_config()`` -> ``initialize_kv_cache()``, which is what sets
``num_gpu_blocks``. Putting the call in the *worker* rather than in a model
runner also makes it independent of ``VllmConfig.use_v2_model_runner``: both
``worker/model_runner_v1.py`` and ``worker/v2/model_runner.py`` are reached
through the same ``self.model_runner.load_model()``.

One provider per MODEL, not per layer: the slot is 0.56 GiB at
``chunk_experts=32`` so 40 do not fit, and the fold reads its chunks back out of
the very buffers the stream served, so a second provider would hand it another
layer's bytes with every check green (``fold_swap.ResidentFold``). The provider
is bound to the layer in flight with ``set_layer``, which
``maybe_stream_prefill`` does on every streamed forward.

With ``stream_prefill`` off this returns ``None`` before it has read the layer
registry, allocated a byte or touched a layer attribute. With it on and the
wiring impossible, it raises: a switch that is accepted and silently does nothing
is the exact failure the whole path is built to refuse (``config.py``
``_validate_preconditions``, ``layer.maybe_stream_prefill``).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from typing import Any

from vllm.logger import logger

from vllm_ascend.kt_offload.layer import (
    KTLayerOffload,
    install_resident_fold,
    install_stream_provider,
    offload_enabled,
    offload_layers_to_install,
    refuse_after_kv_profile,
)
from vllm_ascend.kt_offload.numa_placement import (
    derive_fraction,
    place_weights,
    read_node_memory,
)
from vllm_ascend.kt_offload.stream_io import StreamSlotProvider
from vllm_ascend.kt_offload.weight_source import WeightSource


#: Diagnostic allocations kept alive for the process lifetime (debug_hold_hbm_bytes).
_DEBUG_HBM_HOLDS: list = []


@dataclass(frozen=True)
class StreamingWiring:
    """What was installed. ``fold`` is ``None`` unless ``dynamic_resident``."""

    provider: StreamSlotProvider
    fold: Any | None
    layers: tuple[int, ...]


def _agreed(offloads: dict[int, KTLayerOffload], what: str, get) -> Any:
    """One value for the whole model, or a refusal naming the layers that differ.

    The provider is per model, so a per-layer hidden size or checkpoint has no
    representation here and must not be averaged, picked-first or ignored.
    """
    seen = {idx: get(rt) for idx, rt in offloads.items()}
    distinct = set(seen.values())
    if len(distinct) != 1:
        raise RuntimeError(
            f"kt_offload: the MoE layers disagree on {what}: {seen}. One streaming provider serves every "
            "layer, so there is no way to honour a per-layer value.")
    return distinct.pop()


def _refuse_streamable_capture_size(offloads: dict[int, KTLayerOffload], min_tokens: int) -> None:
    """A captured size at or above the streaming threshold, refused at start-up.

    ``stream_prefill.assert_eager_only`` raises when a forward's token count is
    one of ``cudagraph_capture_sizes``, i.e. the first big prefill landing on
    such a size dies in the middle of serving. It is knowable at start-up from
    the same two numbers, so it is refused at start-up.

    This is also a FLOOR on ``stream_prefill_min_tokens``: "lower the threshold
    until every last chunk of a prompt streams" would walk into the captured
    sizes and be refused here. The commit point moves instead
    (``fold_swap.ResidentFold._decides_now``).
    """
    for idx, rt in offloads.items():
        clash = sorted(s for s in (rt.capture_sizes or []) if int(s) >= int(min_tokens))
        if clash:
            raise RuntimeError(
                f"kt_offload: layer {idx}: cudagraph_capture_sizes {clash} are at or above "
                f"stream_prefill_min_tokens={min_tokens}, so an eager forward of that many tokens would try "
                "to stream a size that is also captured (stream_prefill.assert_eager_only refuses it mid-"
                "serve). Raise stream_prefill_min_tokens above the largest captured size, or drop that size "
                "from the capture list.")


def setup_streaming(vllm_config, *, ascend_config=None, layers=None) -> StreamingWiring | None:
    """Build the streaming provider (and the folded swap) and install them.

    Returns ``None`` when the feature is off, having allocated nothing and
    touched no layer; otherwise what was installed, after checking that every
    layer really holds it.
    """
    if ascend_config is None:
        from vllm_ascend.ascend_config import get_ascend_config

        try:
            ascend_config = get_ascend_config()
        except RuntimeError:
            # No Ascend config in this process means no kt_offload_config, so
            # the switches cannot be on and there is nothing to wire -- the same
            # answer ``offload_enabled`` gives one line down, not a fallback for
            # a wiring that failed (nothing below this point has run yet).
            return None
    if not offload_enabled(ascend_config):
        return None
    cfg = ascend_config.kt_offload_config
    stream = bool(getattr(cfg, "stream_prefill", False))
    dynamic = bool(getattr(cfg, "dynamic_resident", False))
    if not stream and not dynamic:
        # The only quiet exit. Nothing above this line reads the layer registry
        # or allocates.
        return None
    if dynamic and not stream:
        # config._validate_preconditions refuses this combination at start-up;
        # if it is ever reached anyway, the fold has no streaming pass to fold
        # into and there is no path that makes an expert resident.
        raise RuntimeError(
            "kt_offload: dynamic_resident without stream_prefill; the swap-in is folded into the streaming "
            "pass and there is nothing to fold into.")

    offloads = offload_layers_to_install(layers, what="stream")
    # Both boundaries, before a single byte is allocated. The CALLER's config is
    # asked first: it is the one the profile run is about to size, and it answers
    # even for a layer built without a cache_config.
    blocks = getattr(getattr(vllm_config, "cache_config", None), "num_gpu_blocks", None)
    if blocks:
        raise RuntimeError(
            f"kt_offload: the KV cache has already been sized ({blocks} blocks) in the config handed to "
            "setup_streaming, so the profile run is over. The streaming slot and the fold's ND staging are "
            "allocated here and would come out of memory vLLM has already promised to the KV cache "
            "; wire the streaming path before the profile run, from NPUWorker.load_model.")
    refuse_after_kv_profile(offloads, "The streaming slot and the fold's ND staging are allocated here and")
    _refuse_streamable_capture_size(offloads, int(cfg.stream_prefill_min_tokens))

    # Every parameter of the provider comes off the layers themselves -- the same
    # objects the forward runs on -- not re-derived from vllm_config here.
    # KTLayerOffload.__init__ already read them from
    # ``model_config.hf_text_config``; reading them again would be a second
    # source of truth for the shapes the cast depends on.
    weight_path = _agreed(offloads, "the checkpoint kt reads (cfg.weight_path or model_config.model)",
                          lambda rt: rt.weight_path)
    hidden_size = _agreed(offloads, "hidden_size", lambda rt: int(rt.hidden_size))
    intermediate_size = _agreed(offloads, "moe_intermediate_size", lambda rt: int(rt.intermediate_size))
    num_experts = _agreed(offloads, "the routed expert count", lambda rt: int(rt.placement.num_experts))
    device = _agreed(offloads, "the device their weights live on",
                     lambda rt: KTLayerOffload._device_of(rt._module))
    # The cast is the layer's own post-load hook. ``stream_io.default_post_load``
    # unwraps kt's patch (``_kt_offload_original``): handing it the patched
    # wrapper would rebind the live tables of whichever layer is in flight and
    # reload that layer's entire CPU half.
    _agreed(offloads, "the quant method class of their MoE weights",
            lambda rt: type(rt._module.quant_method).__name__)
    first = offloads[min(offloads)]
    quant_method = first._module.quant_method

    # A chunk size that does not divide the expert count leaves a SHORT final
    # chunk, and the cast of a short chunk has never been run on card
    # (``stream_io`` flags it as card-unknown). The default 32 divides 384, so
    # this only bites someone who tunes the knob, and refusing at start-up is how
    # they find out.
    if num_experts % int(cfg.stream_prefill_chunk_experts) != 0:
        raise ValueError(
            f"kt_offload_config.stream_prefill_chunk_experts={cfg.stream_prefill_chunk_experts} does not "
            f"divide the layer's {num_experts} experts: the last chunk of every layer would be short "
            f"({num_experts % int(cfg.stream_prefill_chunk_experts)} of the slot's "
            f"{cfg.stream_prefill_chunk_experts}), and the post-load cast of a slot that is only partly "
            "filled has no on-card reading. Pick a divisor of the expert count.")

    # Stage 2: the bank replaces the checkpoint as the source of streamed
    # chunks. Chosen HERE because the provider's device side must agree with it:
    # the bank's bytes are already FRACTAL_NZ, so the cast must not run.
    bank_dir = str(getattr(cfg, "bank_dir", "") or "")
    if bank_dir:
        from vllm_ascend.kt_offload.bank_source import BankSource
        # 🔴 EXPLICIT thread count. BankSource's own default is
        # MIN_LOADER_THREADS (4) while the checkpoint arm below gets
        # weight_source.DEFAULT_COPY_THREADS (8), so leaving it out quietly
        # halved the streaming read leg's threads when the bank replaced the
        # checkpoint -- and that leg is ~94% of the staging loop's wall clock.
        # 16 is what measure/bench_gather_numa.py measured (12 paired rounds,
        # one whole layer per pass): 4 thr = 50.7 GiB/s, 8 = 65.9, 16 = 69.6
        # with this index split and 76.9 with a node-matched one. The 8 was
        # measured on the CHECKPOINT's scattered-segment gather, which is a
        # different access pattern from the bank's 224 bulk strided runs.
        # VLLM_ASCEND_DSV41_KT_LOADER_THREADS overrides this per server start.
        source = BankSource(bank_dir, num_experts=num_experts, copy_threads=16)
    else:
        source = WeightSource(weight_path, num_experts=num_experts)
    provider = StreamSlotProvider(
        source,
        quant_method,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        chunk_experts=int(cfg.stream_prefill_chunk_experts),
        device=device,
    )
    # We built the source, so the provider's close() owns it too (it only
    # infers ownership when it was handed a path).
    provider.owns_source = True
    if bank_dir:
        # Must come before anything reads a slot: it builds the NZ-tagged
        # destination once, through the production format_nd, and from then on
        # the per-chunk cast is skipped.
        provider.enable_nz_device()

    # Diagnostics (both default off), set up before anything is installed so
    # that an OFF arm can be made to carry the same allocation as an ON arm.
    from vllm_ascend.kt_offload import stream_chain as _sc
    _sc.set_chunk_timing(bool(getattr(cfg, "debug_time_chunks", False)))
    _dummy_hbm = None
    _hold = int(getattr(cfg, "debug_hold_hbm_bytes", 0) or 0)
    if _hold > 0:
        # Reproduce "one large block held for the process lifetime" WITHOUT any
        # stage-1 code path. Removing the fold's 1.75 GiB did not move the
        # regression, so headroom is refuted -- but the 6.75 GiB layer buffer was
        # present in every ON arm of both campaigns, and fragmentation is not
        # headroom. This separates "holding it" from "using it".
        _dummy_hbm = torch.empty(_hold, dtype=torch.uint8, device=device)
        logger.info("kt_offload: DIAGNOSTIC holding %d B (%.3f GiB) of HBM for the process lifetime",
                    _hold, _hold / float(1 << 30))

    # Stage 1 (default off): accumulate every chunk's cast output into ONE
    # whole-layer NZ buffer instead of dropping it. The wrap MUST happen here
    # rather than inside install_stream_provider: from here on, exactly one
    # object is "the provider", and _assert_installed, the fold and
    # StreamingWiring all have to be handed that same object. Wrapping inside
    # the installer left the caller holding the unwrapped provider, so the
    # layers held the wrapper, the fold was built against the inner object, and
    # _assert_installed's identity check refused the start-up with "layer 0 did
    # not take the streaming provider". LayerSlotProvider is a transparent
    # proxy (__getattr__ -> inner) for set_layer / served_nd / layer / device /
    # close, so everything downstream keeps working unchanged.
    if bool(getattr(cfg, "stream_layer_buffer", False)):
        from vllm_ascend.kt_offload import stream_layer
        provider = stream_layer.install_layer_buffer(
            provider,
            sync_scope=str(getattr(cfg, "stream_layer_sync_scope", "device")),
            num_experts=num_experts,
            probe_bytes=int(getattr(cfg, "stream_layer_probe_bytes",
                                    stream_layer.DEFAULT_PROBE_BYTES)),
        )

    fold = None
    try:
        if dynamic:
            # Installs the provider on every layer as well; the fold and the
            # stream must be served by the SAME object.
            fold = install_resident_fold(provider, layers=list(offloads.values()))
        else:
            install_stream_provider(provider, layers=list(offloads.values()))
    except BaseException:
        provider.close()
        raise

    _assert_installed(offloads, provider, fold, dynamic=dynamic)
    logger.info(
        "kt_offload: streaming prefill wired on layers %s -- one provider (%d experts/chunk, %d B/expert, "
        "%.3f GiB of pinned staging) reading %s; folded swap-in %s",
        sorted(offloads), provider.chunk_experts, provider.bytes_per_expert,
        provider.coverage.pinned_bytes / float(1 << 30), weight_path,
        "installed" if fold is not None else "off (dynamic_resident=false)")
    if _dummy_hbm is not None:
        # StreamingWiring is frozen, and a local would be freed on return -- the
        # diagnostic would then silently measure nothing. Hold a module-level
        # reference for the process lifetime, which is exactly what is being
        # reproduced.
        _DEBUG_HBM_HOLDS.append(_dummy_hbm)
    # 🔴 Last step, and only with kt_reads_bank: by now kt has loaded (it read
    # the checkpoint for the SCALES, which it converts and cannot alias, and
    # that faulted in all ~269 GiB of expert bytes) and the streaming provider
    # is bound to the bank. So nothing reads the checkpoint's expert tensors
    # again, and holding their page cache is pure duplication -- which is the
    # whole point of stage 3. Measured 2026-11-15: without this the resident
    # checkpoint page cache stayed at 267.3 GiB even though placement had
    # already moved to the bank.
    #
    # Safe only in this configuration; the function says why.
    if getattr(cfg, "kt_reads_bank", False):
        from vllm_ascend.kt_offload.bank_kt_source import drop_checkpoint_expert_cache

        drop_checkpoint_expert_cache(weight_path)
    return StreamingWiring(provider=provider, fold=fold, layers=tuple(sorted(offloads)))



def setup_placement(vllm_config, *, ascend_config=None, layers=None, experts=None):
    """Put the checkpoint's expert pages on the right NUMA node, once, for the
    whole model. Returns ``(plan, report)``, or ``None`` when the switch is off.

    Must run **before** ``model_runner.load_model()``: kt (zero-copy w13) reads
    these pages inside ``KTMoEWrapper.load_weights``, which runs from
    ``process_weights_after_loading``, i.e. *inside* that call. ``setup_streaming``
    is deliberately after it (it needs the layers to exist), which is why this is
    a separate entry point. Going first is also half the work: a page placed
    before kt reads it is placed once, while a page kt read first was already
    allocated in the wrong place and has to be migrated.

    ``planned == 0`` raises: the config asked for zero-copy w13, and a plan that
    covers no bytes means the switch did not reach anything.
    """
    if ascend_config is None:
        from vllm_ascend.ascend_config import get_ascend_config

        try:
            ascend_config = get_ascend_config()
        except RuntimeError:
            return None
    if not offload_enabled(ascend_config):
        return None
    cfg = ascend_config.kt_offload_config
    if not getattr(cfg, "zerocopy_w13", False):
        # The one quiet exit, and it allocates nothing and reads no file.
        return None

    # The same probe the config validator runs -- repeated because this function
    # is reachable on its own.
    cfg._refuse_without_zerocopy_support()

    tp = int(cfg.threadpool_count)
    part_nodes = list(cfg.numa_nodes) if cfg.numa_nodes else list(range(tp))
    if cfg.zerocopy_node_fraction is not None:
        fraction = list(cfg.zerocopy_node_fraction)
        source_kind = "configured"
    else:
        fraction = derive_fraction(float(cfg.zerocopy_reserve_gib), tp, nodes=part_nodes)
        source_kind = "derived"

    streaming_node = cfg.zerocopy_streaming_node
    if streaming_node is None:
        # w2 and the scales have one consumer (streaming prefill) and are not
        # split, so they go wherever there is room -- measured, not assumed.
        mem = read_node_memory(part_nodes)
        streaming_node = max(part_nodes, key=lambda n: mem[n].available(
            int(float(cfg.zerocopy_reserve_gib) * (1 << 30))))
        streaming_source = "derived"
    else:
        streaming_node = int(streaming_node)
        streaming_source = "configured"

    weight_path = cfg.weight_path or vllm_config.model_config.model
    # 🔴 With kt_reads_bank the pages that must be placed are the BANK's, not the
    # checkpoint's: kt reads the bank, streaming prefill reads the bank, and
    # nothing reads the checkpoint's expert tensors during serving. Placing the
    # checkpoint anyway is what kept its 268.95 GiB faulted in and migrated --
    # measured 2026-11-15 as `planned=70502400` with a total working set that
    # did not move at all (550.0 vs 549.5 GiB).
    kt_bank = None
    if getattr(cfg, "kt_reads_bank", False):
        from vllm_ascend.kt_offload.bank_kt_source import BankPlacementSource, KtBankSource

        hidden = int(vllm_config.model_config.hf_text_config.hidden_size)
        inter = int(vllm_config.model_config.hf_text_config.moe_intermediate_size)
        kt_bank = KtBankSource(cfg.bank_dir, tp_count=tp, hidden=hidden, intermediate=inter)
        source = BankPlacementSource(kt_bank)
    else:
        # WeightSource's own default expert count: KTOffloadConfig has no such
        # field (num_resident_experts is a different number).
        source = WeightSource(weight_path)
    layer_ids = list(layers) if layers is not None else list(
        range(int(vllm_config.model_config.hf_text_config.num_hidden_layers)))
    n_experts = kt_bank.num_experts if kt_bank is not None else source.num_experts
    expert_ids = list(experts) if experts is not None else list(range(n_experts))

    logger.info(
        "kt_offload: placing %d layer(s) x %d experts of %s -- fraction=%s (%s), "
        "part_nodes=%s, streaming_node=%d (%s)",
        len(layer_ids), len(expert_ids),
        (f"the NZ BANK {cfg.bank_dir} (w13 per part + w2 per part)" if kt_bank is not None
         else f"the checkpoint {weight_path} (w13 per part, w2 streaming-only)"),
        [round(f, 4) for f in fraction],
        source_kind, part_nodes, streaming_node, streaming_source)
    try:
        plan, report = place_weights(
            source, layer_ids, expert_ids, tp=tp, part_nodes=part_nodes,
            streaming_node=streaming_node, fraction=fraction,
            fraction_source=source_kind, reserve_gib=float(cfg.zerocopy_reserve_gib),
        )
    finally:
        # The bank source's own mmaps are not opened by placement (it only
        # reports offsets), and the one kt holds pointers into is the registry's
        # in layer.py -- closing this one must not touch that.
        if kt_bank is not None:
            kt_bank.close()
        else:
            source.close()

    if plan.planned_pages == 0:
        raise RuntimeError(
            f"kt_offload: zerocopy_w13 is on but the placement planned 0 pages over "
            f"{len(layer_ids)} layer(s) and {len(expert_ids)} expert(s) of {weight_path}. "
            "The switch reached nothing, so the run is the copy path with a zero-copy label "
            "on it.")
    # The start-up self-report: every number an operator needs to tell "it
    # worked" from "it ran". fault_seconds is the disk read, i.e. the start-up
    # cost.
    logger.info("kt_offload: placement done -- %s", report)
    if report.needed_move == 0 and report.faulted_in == 0:
        logger.warning(
            "kt_offload: the placement changed nothing (needed_move=0, faulted_in=0). Every "
            "page was already where it should be, which is possible on a second start-up -- "
            "but this run is NOT evidence that placement is working, and it cannot be an arm "
            "of an A/B.")
    return plan, report


def _assert_installed(offloads: dict[int, KTLayerOffload], provider, fold, *, dynamic: bool) -> None:
    """Read the wiring back off the layers. A partial install must not survive.

    ``maybe_stream_prefill`` and ``_fold_for_this_forward`` refuse a layer that
    is missing its provider or its fold, so a half-installed model would fail on
    the first long prefill instead of at start-up, in whichever layer the request
    happens to reach first.
    """
    for idx, rt in offloads.items():
        if rt._stream_provider is not provider:
            raise RuntimeError(
                f"kt_offload: layer {idx} did not take the streaming provider; the fold reads its chunks out "
                "of the object that served them, so every layer must hold the same one")
        if rt._stream_slot_provider != provider.slot_for:
            raise RuntimeError(
                f"kt_offload: layer {idx}'s slot callback is not this provider's slot_for")
        if dynamic and fold is not None and fold.stream_min_tokens != int(rt.cfg.stream_prefill_min_tokens):
            # Read back off the layer that applies the gate, not off the number
            # we passed down. The fold commits on the last forward that will
            # STREAM, and which one that is depends on this threshold: if the two
            # came apart, the commit would land on a forward that takes the
            # hybrid path and nothing would move, with every counter
            # self-consistent.
            raise RuntimeError(
                f"kt_offload: layer {idx} streams at >= {rt.cfg.stream_prefill_min_tokens} tokens but the "
                f"folded swap-in was built for {fold.stream_min_tokens}; they are the same number or the "
                "fold commits on a forward that never streams")
        if dynamic and rt._resident_fold is not fold:
            raise RuntimeError(
                f"kt_offload: layer {idx} did not take the folded swap-in; dynamic_resident is on and "
                "_fold_for_this_forward would refuse this layer on the first streamed prefill")
        if not dynamic and rt._resident_fold is not None:
            raise RuntimeError(
                f"kt_offload: layer {idx} carries a folded swap-in although dynamic_resident is off")


__all__ = ["StreamingWiring", "setup_streaming", "setup_placement"]
