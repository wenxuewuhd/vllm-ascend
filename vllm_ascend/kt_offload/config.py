# SPDX-License-Identifier: Apache-2.0
"""``additional_config["kt_offload_config"]``.

Every field is a start-up setting. The resident expert set is fixed for the life
of the process: kt-kernel reads ``gpu_experts_mask`` through a raw pointer from
``load_weights`` on, so mutating it afterwards segfaults.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from vllm_ascend.config_utils import config

ResidentStrategy = Literal["prefix", "frequency", "explicit"]


@config
class KTOffloadConfig:
    """Single-card MoE offload to kt-kernel. Off by default.

    Example::

        "kt_offload_config": {
            "enabled": true,
            "resident_strategy": "frequency",
            "num_resident_experts": 100,
            "frequency_table_path": "/path/gsm8k_topk_freq.pt",
            "cpuinfer_threads": 96,
            "threadpool_count": 2,
            "numa_nodes": [0, 1]
        }

    Resident set, one of:
      * ``prefix``    -- experts ``0..N-1`` on every layer.
      * ``frequency`` -- per layer, the ``N`` experts with the highest count in
        ``frequency_table_path`` (``[num_layers, num_experts]``, global expert
        ids; ties broken by lower id).
      * ``explicit``  -- ``resident_experts_path``: a JSON file listing the
        resident logical ids of every layer. Layers may differ in set and in
        size.
    """

    enabled: bool = False

    # -- where the CPU experts come from --------------------------------
    # Checkpoint directory kt-kernel reads the CPU experts from. None = the
    # served model's own directory (the native MXFP4 safetensors).
    weight_path: str | None = None
    method: str = "MXFP4"

    # -- which experts stay on the NPU ---------------------------------
    resident_strategy: ResidentStrategy = "prefix"
    # Resident experts per layer for prefix/frequency. Ignored by explicit.
    num_resident_experts: int = 0
    frequency_table_path: str | None = None
    resident_experts_path: str | None = None

    # -- kt-kernel CPU runtime -------------------------------------------
    # Required when enabled (no silent default: it decides the host cores used).
    cpuinfer_threads: int = 0
    threadpool_count: int = 2
    numa_nodes: list[int] | None = None
    # Largest token count one forward hands to kt. 0 = the scheduler's
    # max_num_batched_tokens. kt raises if a forward exceeds it.
    chunked_prefill_size: int = 0

    # -- behaviour ---------------------------------------------------------
    # The profile run routes every token to random experts (force load balance).
    # The CPU half is skipped: it does not change the NPU memory peak the profile
    # measures, and would push ~3/4 of the routes through the host for nothing.
    skip_cpu_in_profile_run: bool = True
    # Run kt's synchronous submit/sync path rather than ACL stream callbacks.
    # Sets KT_FORCE_SYNC_SUBMIT=1 for the process.
    force_sync_submit: bool = True
    # The side stream (CPU-expert D2H + host-func node captured onto a second
    # stream, joined back in merge). ON by default -- deliberately, unlike the
    # other switches here. Captured path only: eager never forks. Off: this
    # false, or VLLM_ASCEND_DSV41_KT_SIDE_STREAM=0 (which wins). Read the
    # effective value -- and the P1a / shared-expert-move defaults, which follow
    # it -- from kt_offload/switches.py, never from this field.
    overlap_side_stream: bool = True
    # Fold kt's three host-device input copies into one, using our own pinned
    # buffers instead of kt's slot-reuse pair. Off by default; the criterion is
    # bit-identical output when flipped.
    packed_transfer: bool = False

    # -- streaming prefill / dynamic hot experts ---------------------------
    # Long prefill: stream a layer's expert weights up from host DDR and do the
    # whole MoE on the NPU, instead of paying the CPU path per token.
    stream_prefill: bool = False
    # Experts moved per chunk: trades HBM for passes (a larger chunk needs a
    # proportionally larger staging slot; a whole layer at once does not fit in
    # free HBM). Chunking inside a layer is allowed; deferring part of one is not.
    stream_prefill_chunk_experts: int = 32
    # Below this many tokens in one forward, the layer takes the hybrid path.
    # The CPU path's cost grows with the token count while a streamed layer's
    # does not, so streaming only pays from a few thousand tokens up. 512 sits
    # deliberately BELOW that crossover so the mechanism can be exercised; a
    # production setting belongs above it.
    stream_prefill_min_tokens: int = 512
    # Re-pick which experts stay resident from the heat of the prefill that just
    # streamed past. Needs the CPU side to hold weights for EVERY expert it might
    # swap in -- a miss is a wrong answer, not an error. The swap is FOLDED into
    # the streaming pass (fold_swap.py), so it needs stream_prefill. The heat is
    # accumulated over a request's WHOLE prefill (vLLM cuts a long prompt into
    # several forwards) and the set moves once, on the last chunk, which needs
    # max_num_seqs == 1 (see _refuse_batched_dynamic_resident).
    dynamic_resident: bool = False

    #: Point kt's w13 BufferB straight at the checkpoint's page cache instead of
    #: memcpy-ing a private copy, and migrate those pages to the node of the TP
    #: part that reads them. Saves 2/3 of the weight bytes. w2 deliberately keeps
    #: the copy path: TP slices it along a dimension that is *columns* in the
    #: source, so a 4 KiB page spans rows both parts read and no per-page
    #: placement can make it local -- zero-copying it too regressed kt compute
    #: badly, while w13 alone is free.
    zerocopy_w13: bool = False

    #: Headroom to leave on each node when deriving where w13's pages go, GiB.
    #: The operator's JUDGEMENT; the split itself is derived from it.
    zerocopy_reserve_gib: float = 16.0

    #: For each TP part, the share of ITS OWN w13 bytes that stays on its own
    #: node; the rest overflows to the other node(s). Entries are INDEPENDENT
    #: and ``[1.0, 1.0]`` (perfect locality) is the intended default, NOT a
    #: distribution that sums to 1 -- requiring that pins the mean local share
    #: at 1/tp whatever the values, i.e. half of w13 always read across the
    #: socket. ``None`` derives it at load time from each node's measured free +
    #: reclaimable memory, because what a node can give drifts over hours.
    #:
    #: An A/B run MUST pin this: two arms that each derive their own ratio differ
    #: by more than the switch under test the moment the machine drifts.
    zerocopy_node_fraction: list[float] | None = None

    #: Which node holds ``w2`` and every ``.scale``. Read only by streaming
    #: prefill (kt keeps the copy path for w2), so unlike w13 they have a single
    #: consumer and are not split at all. ``None`` picks the node with the most
    #: free-plus-reclaimable memory at load time; an A/B run MUST pin it, for the
    #: same reason as ``zerocopy_node_fraction``.
    zerocopy_streaming_node: int | None = None

    @model_validator(mode="after")
    def _validate_values(self):
        if not self.enabled:
            return self
        if self.cpuinfer_threads <= 0:
            raise ValueError("kt_offload_config.cpuinfer_threads must be set (> 0) when enabled.")
        if self.threadpool_count <= 0:
            raise ValueError("kt_offload_config.threadpool_count must be > 0.")
        if self.numa_nodes is not None and len(self.numa_nodes) != self.threadpool_count:
            raise ValueError(
                f"kt_offload_config.numa_nodes lists {len(self.numa_nodes)} node(s) "
                f"but threadpool_count is {self.threadpool_count}."
            )
        if self.chunked_prefill_size < 0:
            raise ValueError("kt_offload_config.chunked_prefill_size must be >= 0.")
        if self.resident_strategy in ("prefix", "frequency") and self.num_resident_experts <= 0:
            raise ValueError(
                f"kt_offload_config.resident_strategy={self.resident_strategy!r} needs num_resident_experts > 0."
            )
        if self.resident_strategy == "frequency" and not self.frequency_table_path:
            raise ValueError("kt_offload_config.resident_strategy='frequency' needs frequency_table_path.")
        if self.resident_strategy == "explicit" and not self.resident_experts_path:
            raise ValueError("kt_offload_config.resident_strategy='explicit' needs resident_experts_path.")
        if self.stream_prefill_chunk_experts <= 0:
            raise ValueError("kt_offload_config.stream_prefill_chunk_experts must be > 0 "
                             "(0 would stream nothing and still cost a pass).")
        if self.stream_prefill_min_tokens <= 0:
            raise ValueError("kt_offload_config.stream_prefill_min_tokens must be > 0.")
        if self.zerocopy_reserve_gib < 0:
            raise ValueError("kt_offload_config.zerocopy_reserve_gib must be >= 0.")
        if self.zerocopy_streaming_node is not None and self.zerocopy_streaming_node < 0:
            raise ValueError(
                f"kt_offload_config.zerocopy_streaming_node={self.zerocopy_streaming_node} "
                "must be a node id >= 0 (None derives it from the nodes' free memory).")
        if self.zerocopy_node_fraction is not None:
            # An illegal split must not fall back to something reasonable: a
            # silent 50/50 would make an A/B compare two arms that placed their
            # pages differently -- the exact failure the field exists to stop.
            f = self.zerocopy_node_fraction
            if len(f) != self.threadpool_count:
                raise ValueError(
                    f"kt_offload_config.zerocopy_node_fraction lists {len(f)} fraction(s) but "
                    f"threadpool_count is {self.threadpool_count}; it is one per TP part.")
            if any(v < 0.0 or v > 1.0 for v in f):
                raise ValueError(
                    f"kt_offload_config.zerocopy_node_fraction={f} has a value outside [0, 1].")
        return self

    def _refuse_without_zerocopy_support(self) -> None:
        """``zerocopy_w13`` is SILENT when the kt build does not support it.

        ``KT_ZEROCOPY_*`` are read inside kt's compiled extension, so exporting
        them against a build that predates them is ignored without a word: the
        run quietly takes the copy path, correct but with 2/3 more anonymous
        memory than the pre-flight account granted. There is no exported symbol
        to ask, so the probe looks for the switch names in the extension this
        process actually imported -- deliberately not ``os.environ``, since the
        operator who exported the variable is what cannot be trusted here.
        """
        if not self.zerocopy_w13:
            return
        try:
            import kt_kernel
        except Exception as exc:                             # noqa: BLE001
            raise ValueError(
                "kt_offload_config.zerocopy_w13: kt_kernel is not importable "
                f"({type(exc).__name__}: {exc}), so whether this build supports zero-copy "
                "weights cannot be established.") from exc
        import glob
        import os
        pkg = os.path.dirname(getattr(kt_kernel, "__file__", "") or "")
        sos = glob.glob(os.path.join(pkg, "kt_kernel_ext*.so")) if pkg else []
        if not sos:
            raise ValueError(
                "kt_offload_config.zerocopy_w13: no kt_kernel_ext*.so next to "
                f"{getattr(kt_kernel, '__file__', '?')}, so the build cannot be probed. "
                "Refusing rather than assuming the switch works.")
        try:
            with open(sos[0], "rb") as f:
                blob = f.read()
        except OSError as exc:
            raise ValueError(
                f"kt_offload_config.zerocopy_w13: cannot read {sos[0]} ({exc}) to establish "
                "whether it supports zero-copy weights.") from exc
        missing = [n for n in (b"KT_ZEROCOPY_WEIGHTS", b"KT_ZEROCOPY_SCOPE") if n not in blob]
        if missing:
            raise ValueError(
                "kt_offload_config.zerocopy_w13: this kt build has no "
                f"{b', '.join(missing).decode()} ({sos[0]}). This switch needs the kt-kernel "
                "patch that ships with this feature; apply it and rebuild kt, or set "
                "zerocopy_w13=false.\n"
                "Refusing rather than falling back, because the fallback is silent: kt would "
                "take the copy path, hold a full private copy of w13, and the run would look "
                "identical except for ~2/3 more anonymous memory than the pre-flight account "
                "granted -- which surfaces later as an OOM with nothing pointing back here.")

    def _refuse_without_full_set_load(self) -> None:
        """``dynamic_resident`` needs kt to hold CPU weights for EVERY expert.

        Exporting ``KT_FULLSET_LOAD`` against a kt build that does not know the
        variable is a SILENT no-op: kt keeps loading the CPU subset, every
        resident expert stays pinned, the fold commits nothing, and the run looks
        exactly like "the policy had nothing to move". So the precondition is not
        "the operator exported it" but "kt, as imported in THIS process, reports
        the full set".
        """
        if not self.dynamic_resident:
            return
        try:
            from kt_kernel.utils import amx
        except Exception as exc:                             # noqa: BLE001
            raise ValueError(
                "kt_offload_config.dynamic_resident: kt_kernel.utils.amx is not importable "
                f"({type(exc).__name__}: {exc}), so whether kt loads every expert's CPU weights "
                "cannot be established. The swap-in has nowhere to demote a resident expert to "
                "unless it does.") from exc
        probe = getattr(amx, "_fullset_load_requested", None)
        if probe is None:
            raise ValueError(
                "kt_offload_config.dynamic_resident: this kt build has no KT_FULLSET_LOAD "
                f"(no _fullset_load_requested in {getattr(amx, '__file__', '?')}). This switch needs "
                "the kt-kernel patch that ships with this feature; apply it and rebuild kt, or set "
                "dynamic_resident=false.\n"
                "Refusing rather than falling back, because the fallback is silent: kt would keep the "
                "CPU subset, every resident expert would stay pinned, the folded swap-in would commit "
                "nothing -- indistinguishable from 'the heat said no change', so a run taken that way "
                "reports 'dynamic hot experts give no gain' and is wrong.")
        if not probe():
            raise ValueError(
                "kt_offload_config.dynamic_resident needs KT_FULLSET_LOAD=1: with the CPU expert "
                "subset (amx.py:_cpu_expert_subset_enabled) a resident expert has no CPU BufferB, so "
                "it cannot be demoted, the policy has zero free slots and the fold is a silent no-op "
                ".")

    def _refuse_batched_dynamic_resident(self, vllm_config: Any) -> None:
        """``dynamic_resident`` runs at ``max_num_seqs == 1``, and nowhere else.

        A **timing** limit, not a safety one: a larger batch computes nothing
        wrongly, but the commit POINT goes away. The heat is accumulated across a
        request's prefill chunks and moved once, on the last one; with several
        requests prefilling at once there is no "last one" -- their chunks
        interleave and their heats average each other out, so the "hot" set would
        be the batch's flat average. A decayed global tally moved every N steps
        would work for a batch; it is a different design.

        ``scheduler_config.max_num_seqs`` is the field; the model runner reads
        the same one as ``self.max_num_reqs``.
        """
        if not self.dynamic_resident:
            return
        sched = getattr(vllm_config, "scheduler_config", None)
        max_num_seqs = getattr(sched, "max_num_seqs", None)
        if max_num_seqs is None:
            raise ValueError(
                "kt_offload_config.dynamic_resident needs scheduler_config.max_num_seqs to check that this "
                "is a one-sequence-at-a-time server, and the config handed to the validator has no "
                "scheduler_config. Refusing rather than assuming the batch size.")
        if int(max_num_seqs) != 1:
            raise ValueError(
                f"kt_offload_config.dynamic_resident requires max_num_seqs=1; this server is configured for "
                f"{int(max_num_seqs)}. The folded swap-in accumulates one request's routing over its whole "
                "prefill and moves the resident set once, on the last chunk of that prefill. With more than "
                "one sequence there is no such point -- several prefills interleave, and their heats average "
                "each other out, so the set that would be made resident is the batch's flat average rather "
                "than any request's hot set. Nothing is computed incorrectly at a larger batch; the feature "
                "simply has no defined moment to act and nothing to act on. Serve with --max-num-seqs 1, or "
                "turn dynamic_resident off (stream_prefill on its own has no such limit).")

    def _validate_preconditions(self, vllm_config: Any) -> None:
        """Parallel/runtime settings the M1 implementation depends on."""
        if not self.enabled:
            return
        # The side stream, P1a and the shared-expert move are on by default; P1b
        # is off. What each switch actually is -- cfg, env and the automatic
        # defaults together -- comes from ONE function, so this check, the layer
        # runtime and the log below cannot disagree about what "unset" means. It
        # raises on what was asked for explicitly and cannot be had (P1a/P1b "1"
        # without packed_transfer or the side stream; P1a and P1b both "1").
        # packed_transfer + overlap_side_stream is accepted: ``route()`` tests
        # ``overlap`` first and the forked path uses the packed per-layer buffers.
        from vllm.logger import logger

        from vllm_ascend.kt_offload import device_signal, shared_window, switches

        sw = switches.resolve(self, validate=True)
        eager = bool(getattr(getattr(vllm_config, "model_config", None), "enforce_eager", False))
        logger.info(
            "kt_offload switches: %s%s",
            sw.summary(),
            " -- enforce_eager: the side stream / async wait / device signal are captured-path "
            "changes and an automatic shared-window move is not made, so eager runs its original path"
            if eager
            else "",
        )
        # P1b (VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL): no host-func node at all --
        # an 8-byte copy of a per-step device counter is the start signal and a
        # polling thread does the rest. Its precondition and its conflict with
        # an explicit P1a are refused in switches.resolve; this one is its own.
        if sw.device_signal.on:
            from vllm_ascend import envs

            if envs.VLLM_ASCEND_DSV41_KT_SEAM_TIMING:
                raise ValueError(
                    f"{device_signal.ENV_NAME}=1 removes the host-func callback, and the seam instrument "
                    "(VLLM_ASCEND_DSV41_KT_SEAM_TIMING) measures segments *inside* that callback: with P1b "
                    "on it would report on a callback that never runs and its A/F columns would be joined "
                    "against HOSTFUNC_CALLBACK rows that do not exist. Refusing rather than printing a "
                    "report about nothing: an instrument that reports on nothing reads exactly like "
                    "one that found nothing."
                )
        # The shared-expert MLP moves in front of the kt merge. An explicit "1"
        # without the side stream is not refused -- the order is still correct
        # there -- but say so. (Unset follows the side stream, so it cannot land
        # here.)
        if sw.shared_in_window.on and not sw.side_stream.on:
            logger.warning(
                "%s=1 without the side stream (%s): the merge still moves behind the "
                "shared-expert MLP and the result is unchanged, but there is no idle window to hide it in "
                "(the main stream waits for the CPU experts inside merge() either way), so expect no "
                "TPOT difference.",
                shared_window.ENV_NAME,
                sw.side_stream,
            )
        if self.dynamic_resident and not self.stream_prefill:
            raise ValueError(
                "kt_offload_config.dynamic_resident needs stream_prefill: the swap-in is folded into the "
                "streaming prefill pass and there is no other path that makes an expert "
                "resident.")
        self._refuse_batched_dynamic_resident(vllm_config)
        self._refuse_without_full_set_load()
        self._refuse_without_zerocopy_support()
        if self.stream_prefill:
            # The two ways a streamed run could go silently wrong are refused
            # where they can be checked, not here. "On but nothing ever
            # streamed": layer.py's maybe_stream_prefill raises instead of
            # falling back to the hybrid path when the slot provider is missing,
            # and wiring._refuse_streamable_capture_size refuses a threshold at
            # or below the largest capture size (such a size would stream and be
            # captured, and die mid-serve in assert_eager_only). "The swap-in
            # moved nothing": _refuse_without_full_set_load above.
            if self.stream_prefill_min_tokens <= 0:
                raise ValueError(
                    "kt_offload_config.stream_prefill needs stream_prefill_min_tokens > 0; the "
                    "threshold is what keeps a streamed forward out of a captured size.")
        pc = vllm_config.parallel_config
        sizes = {
            "tensor_parallel_size": pc.tensor_parallel_size,
            "data_parallel_size": pc.data_parallel_size,
            "pipeline_parallel_size": pc.pipeline_parallel_size,
        }
        wrong = {k: v for k, v in sizes.items() if v != 1}
        if wrong:
            raise ValueError(
                f"kt_offload_config needs a single card (TP=DP=PP=1); got {wrong}. "
                "The NPU half is placed as one EP rank that owns every resident expert."
            )
        if pc.enable_eplb:
            raise ValueError("kt_offload_config is incompatible with EPLB (it owns the expert placement).")
        if vllm_config.speculative_config is not None and not uses_hbm_dspark_draft(vllm_config):
            raise ValueError(
                "kt_offload_config supports only DeepSeek-V4.1 DSpark with HBM draft experts; "
                "offload of MTP/draft experts is not implemented."
            )


def uses_hbm_dspark_draft(vllm_config: Any) -> bool:
    """Only the registered V4.1 DSpark draft keeps all experts on the NPU."""
    spec = getattr(vllm_config, "speculative_config", None)
    draft = getattr(spec, "draft_model_config", None)
    return (
        getattr(spec, "method", None) == "dspark"
        and getattr(draft, "_architecture", None) == "DSparkDeepseekV41DraftModel"
    )
