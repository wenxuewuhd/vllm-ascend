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
    # Stage 1: keep every chunk's cast bytes in ONE whole-layer NZ buffer instead
    # of dropping them, and hand the GEMM that buffer with a group list in which
    # only the current chunk's groups carry tokens. Measured on card: bitwise
    # equal to the per-chunk form (maxabsdiff 0), +3.6% on the chunk's GEMM call
    # (the 352 empty groups), which is ~0.13 ms per layer against 228 ms of host
    # read. What it buys: the folded swap's 1.75 GiB whole-layer ND staging
    # becomes unnecessary, the weight address stops moving, and the resident
    # experts can be taken from HBM instead of the host. Needs stream_prefill.
    # DEFAULT OFF: the default path is unchanged until this has run end to end.
    # NOTE on HBM while both this and dynamic_resident are on: the fold still
    # keeps its own 1.75 GiB whole-layer ND staging, so the card holds the layer
    # buffer (~7.2 GiB) AND that staging AND the chunk buffers (~1.1 GiB) -- about
    # 10 GiB of an 11.41 GiB budget. Making the fold read the layer buffer instead
    # is the step that reclaims it, and it is NOT part of this switch.
    stream_layer_buffer: bool = False
    # Streaming prefill moves all 384 experts of all 40 layers on EVERY request
    # -- 268.9 GiB, the whole bank -- including the ~num_resident_experts that
    # are already in HBM. With this on, a chunk's resident members are neither
    # gathered nor H2D'd: their layer-buffer slots are filled by a
    # device-to-device copy out of the resident parameter, which holds the same
    # bytes. 26% of the streamed bytes at num_resident_experts=100, on the leg
    # the server's staging report puts at ~94% of the staging loop.
    #
    # 🔴 Done this way round PRECISELY so the output stays bit-identical: the
    # fill goes into the provider's own chunk buffer, so every consumer still
    # sees a COMPLETE chunk and the partition, the slot tables and the fp32
    # accumulation order are untouched -- only the provenance of some slots'
    # bytes changes. Streaming the offloaded set alone and computing the
    # resident half separately would be less code and WOULD change the rounding
    # (see measure/PLAN_G_stream_only_offloaded.md).
    #
    # 🔴 DEFAULT OFF, and it must stay off: this switch carries PRECONDITIONS
    # (bank_dir, stream_prefill), so defaulting it on makes every config that
    # does not meet them fail validation -- 20 of this repo's own config tests
    # did, immediately, when it was tried. A switch with preconditions is opted
    # into by the deployment config, not by the dataclass.
    #
    # The measured recommendation is to TURN IT ON wherever the preconditions
    # hold (measure/s3_g.json does):
    #   correctness  7/7 prompts token-identical to stage 3, both rounds
    #   mechanism    "HOST MOVED 49.7 of 67.2 GiB (74.0%), 1000 slot(s) filled
    #                from HBM instead" -- 198.9 GiB per request instead of 268.9
    #   TTFT         steady state 13.79 -> 12.19 s at p=4096 (-11.6%), nine
    #                requests per arm with NO overlap between the two ranges
    #                (off 13.51-14.44, on 11.77-12.53)
    #   TPOT         unchanged, as it must be: nothing in decode streams
    # 🔴 Known and NOT explained: the first two requests of a server's life cost
    # ~4 s more than without it. The staging counters refute both obvious
    # causes -- the reports before and after the jump have the SAME fill shape
    # (whole=0, mixed=120, ~3000 D2D calls, 17.5 GiB) and differ 4x in
    # ms/chunk. It coincides with the fold re-learning the hot set. Set this to
    # False to get the stage-3 path back.
    stream_resident_from_hbm: bool = False
    # DIAGNOSTIC, default off. Hold this many bytes of HBM for the process's
    # lifetime, allocated at wiring time like the stage-1 layer buffer is.
    # Purpose: the stage-1 TTFT regression survived removing the fold's 1.75 GiB
    # (headroom went 693 -> 3614 MB, regression did not move), so "not enough
    # headroom" is refuted -- but "holding one large long-lived block" is NOT,
    # because the 6.75 GiB layer buffer was present in every ON arm of both
    # campaigns. Setting this to 7219445760 on an OFF arm reproduces the
    # allocation WITHOUT any stage-1 code path, which separates the two.
    debug_hold_hbm_bytes: int = 0
    # DIAGNOSTIC, default off. Time each run_chunk (the GEMMs) with NPU events.
    # The events are read once per forward, not per chunk, so the measurement
    # does not add the per-chunk synchronise it is meant to measure around.
    debug_time_chunks: bool = False
    # Scope of the flush before each raw NZ write. "device" is today's
    # behaviour: torch.npu.synchronize(), which waits for ALL streams on the
    # card. "stream" uses current_stream().synchronize() (npuStreamSynchronize),
    # which waits only for the stream the cast was submitted on.
    #
    # Why this is the right axis: the hazard is HOST-SIDE submission order --
    # with TASK_QUEUE_ENABLE != 0 torch_npu queues ops on a host thread, so a
    # raw memcpy_async issued from Python can overtake the cast that produced
    # its source (12/100 wrong, always the last slot). An event swap does NOT
    # fix that: Event.record() goes through the SAME host queue, and a raw
    # aclrtStreamWaitEvent would overtake it exactly as the memcpy does. A
    # stream-scoped flush keeps the guarantee and is strictly narrower, which
    # matters because production carries kt's side stream, engram's host-func
    # and the swap commits on other streams.
    #
    # 🔴 "stream" MUST be validated for CORRECTNESS, not just speed: if it does
    # not also flush the host queue, the failure is silent wrong numbers, which
    # is exactly what the device-wide sync was introduced to prevent.
    stream_layer_sync_scope: str = "device"
    # Stage 2 (default off): serve the prefill staging out of a prebuilt NZ bank
    # in /dev/shm instead of gathering from the checkpoint. Set to the bank
    # directory, e.g. "/dev/shm/dsv41_nz_bank"; empty keeps the checkpoint path.
    #
    # What it changes: the bank already holds device-layout FRACTAL_NZ bytes, so
    # the 192 scattered per-chunk segments that select + concatenate + reorder
    # become contiguous runs, AND the per-chunk device cast disappears (the
    # staging arrives already permuted). Measured host-read leg, paired and
    # alternating: 220.6 -> 104.2 ms per layer, -52.4% against a 1.2% null floor.
    #
    # 🔴 On its own this RAISES host memory: the bank and the checkpoint page
    # cache coexist. The -93.6 GiB only arrives once kt reads the same bank
    # (stage 3), which lets the checkpoint mapping go.
    bank_dir: str = ""
    # How the fold orders a commit's read of its NZ staging against the NEXT
    # layer's writes to it. Only the bank path has the problem: in NZ mode the
    # staging is filled by raw ACL copies, which go to the stream immediately,
    # while the commit reads it through torch_npu, which goes through the
    # host-side task queue -- so the next layer's writes can be submitted first
    # and the commit reads bytes that have already been replaced. Silently:
    # every byte count, address and coverage check still agrees.
    #
    # "device" (default) is the one barrier that unambiguously flushes queue and
    # stream both; "stream" is the cheaper candidate, not the default until it
    # is measured; "none" removes it, to drive the failure on card as a positive
    # control. One wait per folded layer, on the one forward per request that
    # commits.
    fold_nz_barrier: str = "device"
    # Stage 3: let kt consume the SAME NZ bank the NPU streams from, in place,
    # instead of reading the checkpoint. That is what removes the duplicate --
    # ~269 GiB on this model, because the checkpoint's expert pages have to stay
    # resident today purely for kt's zero-copy w13.
    #
    # Requires bank_dir and zerocopy_w13. MXFP4 only: no other kt backend knows
    # the FRACTAL_NZ geometry, and handing one these pointers would have it read
    # the bytes as row-major and compute wrong numbers silently (kt refuses).
    #
    # 🔴 It costs TPOT. The design measured the kernel-level cost at
    # -5.1..-5.6% and planned to repay it with stage 4 (decode prefetch); the
    # routing trace taken 2026-11-15 refutes stage 4 as designed (a demand LRU
    # loses to a static frequency set at every cap >= 64, and the miss rate is
    # 4x the transfer budget), so the cost is PERMANENT. The trade is one copy
    # instead of two for roughly +5% TPOT.
    kt_reads_bank: bool = False
    # Start-up self-check of CANN's FRACTAL_NZ byte geometry (kt_offload/nz_geometry.py):
    # cast a two-expert tensor at the REAL w13/w2 shapes and compare the NZ bytes against
    # the formula this build assumes. Once per process, ~60 ms.
    #
    # OFF by default, and FORCED ON by stream_layer_buffer -- any route that has kt read
    # the same NZ bytes the NPU reads depends on that geometry, which CANN does not
    # document and we reverse-engineered. A CANN release that changed C0 or the fractal
    # order would not raise there: it would compute wrong numbers, quietly. Setting this
    # false does NOT disable the check while such a route is on -- a route must not be
    # able to switch off its own guard.
    #
    # Read the effective value from nz_geometry.check_enabled(cfg), never from this field.
    verify_nz_geometry: bool = False
    # Bytes read back from the head and tail of each written slot range to prove
    # the raw write landed; 0 disables. Only meaningful with stream_layer_buffer.
    stream_layer_probe_bytes: int = 256
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
    # Which expert tensors kt consumes IN PLACE from the checkpoint mapping.
    #   "w13" (default) gate/up only. w13 is a contiguous row block per TP part,
    #          so its pages place 100% node-local. w2/down is COPIED into kt's
    #          own node-local memory: 84.4 GiB on this model.
    #   "all"  w2/down in place too, which frees that 84.4 GiB -- at a locality
    #          cost that is a property of the CHECKPOINT's layout, not of the
    #          code: w2 is [hidden, full_inter/2] row-major and TP splits
    #          `inter`, so each part owns a column slice of every row. With
    #          full_inter/2 = 1024 B a row is a quarter of a page, so one page
    #          carries BOTH parts' bytes and no per-page placement can beat
    #          1/tp_count local. zerocopy_weights.hpp records -28.3% for it.
    #
    # 🔴 The penalty is the CHECKPOINT's, not w2's. A bank that stores w2 split
    # per TP part makes each part's bytes contiguous and the penalty disappears
    # -- which is what stage 3 is actually for. Until then this knob is the
    # cheap way to MEASURE the 84.4 GiB instead of assuming it.
    zerocopy_scope: str = "w13"

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
            raise ValueError(
                "kt_offload_config.stream_prefill_chunk_experts must be > 0 "
                "(0 would stream nothing and still cost a pass)."
            )
        if self.stream_prefill_min_tokens <= 0:
            raise ValueError("kt_offload_config.stream_prefill_min_tokens must be > 0.")
        # bank_dir + dynamic_resident WAS refused here: the fold copied its chosen
        # experts out of the ND H2D landing buffers (stream_io.served_nd), which
        # the bank path never writes, so it staged UNWRITTEN bytes into the
        # resident set with no error -- just wrong experts from the next decode
        # step on. The fold now reads NZ instead (fold_swap._stage_chunk_nz,
        # fed by stream_io.served_nz, which refuses the mode it is not for), so
        # the pair is allowed. What enforces it is no longer a config check: the
        # fold asks the provider for its layout (``nz_device``) and both
        # accessors refuse the other mode, which a config check could not do for
        # a provider that is handed round.
        if self.kt_reads_bank and not self.bank_dir:
            raise ValueError("kt_offload_config.kt_reads_bank needs bank_dir: it points kt at the bank.")
        if self.kt_reads_bank and not self.zerocopy_w13:
            raise ValueError(
                "kt_offload_config.kt_reads_bank needs zerocopy_w13: the bank is consumed IN PLACE, "
                "and the copy path would memcpy NZ bytes into a row-major BufferB."
            )
        if self.kt_reads_bank and self.zerocopy_scope != "all":
            # 🔴 Silent wrong answer, not a slowdown. Under KT_ZEROCOPY_SCOPE=w13
            # kt STAGES w2 itself, and fp4-moe.hpp does that with an ND
            # column-slice walk over `config.down_projs[0][expert]` -- index 0,
            # not the TP part, and ND strides. The bank's w2 is FRACTAL_NZ and
            # its pointers are per part, so that walk would read the wrong bytes
            # and report nothing. (Found 2026-11-15 while looking for an arm
            # that isolates w2's contribution to the TPOT regression; the
            # combination had never been run.)
            raise ValueError(
                "kt_offload_config.kt_reads_bank needs zerocopy_scope='all': with 'w13' kt stages "
                "w2 itself through an ND column walk, which cannot read the bank's NZ w2 -- it "
                "would compute wrong numbers rather than fail."
            )
        if self.zerocopy_scope not in ("w13", "all"):
            raise ValueError(
                "kt_offload_config.zerocopy_scope must be 'w13' or 'all' "
                "(the two scopes kt's KT_ZEROCOPY_SCOPE parser accepts)."
            )
        if self.zerocopy_scope != "w13" and not self.zerocopy_w13:
            raise ValueError(
                "kt_offload_config.zerocopy_scope only means anything with "
                "zerocopy_w13=true; with zero-copy off kt copies everything."
            )
        if self.fold_nz_barrier not in ("device", "stream", "none"):
            raise ValueError("kt_offload_config.fold_nz_barrier must be 'device', 'stream' or 'none'.")
        if self.fold_nz_barrier == "none" and not self.bank_dir:
            raise ValueError(
                "kt_offload_config.fold_nz_barrier='none' only means anything with bank_dir; "
                "it removes the ordering between a commit's read of the fold staging and the "
                "next layer's raw writes to it, and exists to drive that failure on card."
            )
        if self.bank_dir and not self.stream_prefill:
            raise ValueError(
                "kt_offload_config.bank_dir needs stream_prefill: the bank only "
                "replaces the source of the streamed chunks."
            )
        if self.stream_layer_sync_scope not in ("device", "stream"):
            raise ValueError("kt_offload_config.stream_layer_sync_scope must be 'device' or 'stream'.")
        if self.debug_hold_hbm_bytes < 0:
            raise ValueError("kt_offload_config.debug_hold_hbm_bytes must be >= 0.")
        if self.stream_layer_buffer and not self.stream_prefill:
            raise ValueError(
                "kt_offload_config.stream_layer_buffer needs stream_prefill: it only "
                "changes where the streamed chunks accumulate."
            )
        if self.stream_resident_from_hbm:
            # Each precondition is a different silent-wrong-answer if missing.
            # NOTE it does NOT need stream_layer_buffer: the fill happens in the
            # provider's own chunk buffer, so both the per-chunk form (what
            # production runs) and the whole-layer form see a complete chunk.
            if not self.stream_prefill:
                raise ValueError(
                    "kt_offload_config.stream_resident_from_hbm needs stream_prefill: it only "
                    "changes where a streamed chunk's bytes come from."
                )
            if not self.bank_dir:
                raise ValueError(
                    "kt_offload_config.stream_resident_from_hbm needs bank_dir: skipping slots "
                    "requires a source that can gather a subset of a chunk, and only the NZ bank "
                    "can (the checkpoint gather writes whole group blocks)."
                )
            if self.num_resident_experts <= 0:
                raise ValueError(
                    "kt_offload_config.stream_resident_from_hbm with num_resident_experts=0 would "
                    "skip nothing; it is on by mistake."
                )
        if self.stream_layer_probe_bytes < 0:
            raise ValueError("kt_offload_config.stream_layer_probe_bytes must be >= 0.")
        if self.zerocopy_reserve_gib < 0:
            raise ValueError("kt_offload_config.zerocopy_reserve_gib must be >= 0.")
        if self.zerocopy_streaming_node is not None and self.zerocopy_streaming_node < 0:
            raise ValueError(
                f"kt_offload_config.zerocopy_streaming_node={self.zerocopy_streaming_node} "
                "must be a node id >= 0 (None derives it from the nodes' free memory)."
            )
        if self.zerocopy_node_fraction is not None:
            # An illegal split must not fall back to something reasonable: a
            # silent 50/50 would make an A/B compare two arms that placed their
            # pages differently -- the exact failure the field exists to stop.
            f = self.zerocopy_node_fraction
            if len(f) != self.threadpool_count:
                raise ValueError(
                    f"kt_offload_config.zerocopy_node_fraction lists {len(f)} fraction(s) but "
                    f"threadpool_count is {self.threadpool_count}; it is one per TP part."
                )
            if any(v < 0.0 or v > 1.0 for v in f):
                raise ValueError(f"kt_offload_config.zerocopy_node_fraction={f} has a value outside [0, 1].")
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
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "kt_offload_config.zerocopy_w13: kt_kernel is not importable "
                f"({type(exc).__name__}: {exc}), so whether this build supports zero-copy "
                "weights cannot be established."
            ) from exc
        import glob
        import os

        pkg = os.path.dirname(getattr(kt_kernel, "__file__", "") or "")
        sos = glob.glob(os.path.join(pkg, "kt_kernel_ext*.so")) if pkg else []
        if not sos:
            raise ValueError(
                "kt_offload_config.zerocopy_w13: no kt_kernel_ext*.so next to "
                f"{getattr(kt_kernel, '__file__', '?')}, so the build cannot be probed. "
                "Refusing rather than assuming the switch works."
            )
        try:
            with open(sos[0], "rb") as f:
                blob = f.read()
        except OSError as exc:
            raise ValueError(
                f"kt_offload_config.zerocopy_w13: cannot read {sos[0]} ({exc}) to establish "
                "whether it supports zero-copy weights."
            ) from exc
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
                "granted -- which surfaces later as an OOM with nothing pointing back here."
            )

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
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "kt_offload_config.dynamic_resident: kt_kernel.utils.amx is not importable "
                f"({type(exc).__name__}: {exc}), so whether kt loads every expert's CPU weights "
                "cannot be established. The swap-in has nowhere to demote a resident expert to "
                "unless it does."
            ) from exc
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
                "reports 'dynamic hot experts give no gain' and is wrong."
            )
        if not probe():
            raise ValueError(
                "kt_offload_config.dynamic_resident needs KT_FULLSET_LOAD=1: with the CPU expert "
                "subset (amx.py:_cpu_expert_subset_enabled) a resident expert has no CPU BufferB, so "
                "it cannot be demoted, the policy has zero free slots and the fold is a silent no-op "
                "."
            )

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
                "scheduler_config. Refusing rather than assuming the batch size."
            )
        if int(max_num_seqs) != 1:
            raise ValueError(
                f"kt_offload_config.dynamic_resident requires max_num_seqs=1; this server is configured for "
                f"{int(max_num_seqs)}. The folded swap-in accumulates one request's routing over its whole "
                "prefill and moves the resident set once, on the last chunk of that prefill. With more than "
                "one sequence there is no such point -- several prefills interleave, and their heats average "
                "each other out, so the set that would be made resident is the batch's flat average rather "
                "than any request's hot set. Nothing is computed incorrectly at a larger batch; the feature "
                "simply has no defined moment to act and nothing to act on. Serve with --max-num-seqs 1, or "
                "turn dynamic_resident off (stream_prefill on its own has no such limit)."
            )

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
                "resident."
            )
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
                    "threshold is what keeps a streamed forward out of a captured size."
                )
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
