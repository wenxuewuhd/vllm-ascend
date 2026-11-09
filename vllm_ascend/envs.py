#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # Build only the vllm_ascend_C extension and keep vllm_ascend/_cann_ops_custom
    # untouched (COMPILE_CUSTOM_KERNELS=0 builds nothing at all).
    "SKIP_ACLNN_BUILD": lambda: bool(int(os.getenv("SKIP_ACLNN_BUILD", "0"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    # NOTE (R44): this is the ONLY key in this file that appears anywhere in the
    # gating of the operators listed in the block below, and even here it is a
    # co-requirement, not an enabler: ascend_config.py:151-155 only ASSERTS that
    # it is set once additional_config.eplb_config.dynamic_eplb is already true.
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # ----------------------------------------------------------------------
    # Operators that are BUILT INTO csrc BUT NOT INSTALLED in the A5 operator
    # package -- i.e. one flipped switch away from failing. Inventory, evidence
    # and the per-operator verdicts:
    #     build_artifacts/official_code_review/o_csrc_vs_installed_ops.md
    #
    # WHEN THEY FAIL: at the operator's FIRST CALL, not at import and not at
    # startup. Every torch schema in csrc/torch_binding.cpp is registered
    # unconditionally, so the extension imports and the service comes up
    # exactly as it does on a machine where these operators exist. "The server
    # started" is not evidence that any of them is installed.
    #
    # NONE OF THEM IS AN ENVIRONMENT VARIABLE, so none of them can be given a
    # comment next to its own entry here -- scanned by read point, not by
    # scanning this file (they are additional_config fields, a vLLM config, or
    # not user-facing at all). Where each one's switch actually lives:
    #   FusedSparseAttentionOverlap   additional_config.sparse_kv_offload_config
    #                                 .use_fused_overlap  (ascend_config.py:1327,
    #                                 default False; parent .enabled :1322)
    #   StoreKVBlock                  additional_config.enable_sparse_li_c8
    #                                 (ascend_config.py:450, default False) AND a
    #                                 PD kv_producer node (derived at :705)
    #   DispatchFFNCombine(+BF16/W4A8) additional_config.enable_fused_mc2 == 1
    #                                 (ascend_config.py:418, default 0). The BF16
    #                                 and W4A8 variants have no Python switch at
    #                                 all: csrc/mc2/dispatch_ffn_combine/
    #                                 dispatch_ffn_combine_torch_adpt.h:38-73
    #                                 picks one by weight dtype.
    #   CopyAndExpandEagleInputs      vLLM's speculative_config.parallel_drafting
    #                                 (vllm/config/speculative.py:166, default
    #                                 False) via needs_extra_input_slots; nothing
    #                                 in vllm-ascend gates it.
    #   TransposeKvCacheByBlock       additional_config
    #                                 .enable_transpose_kv_cache_by_block
    #                                 (ascend_config.py:427, DEFAULT TRUE) AND
    #                                 enable_custom_op() AND a Mooncake P2P KV
    #                                 transfer deployment. On A5 it is held off
    #                                 only by enable_custom_op() -- see below.
    #                                 R55: both fused call sites now refuse
    #                                 through custom_op_guards
    #                                 .require_transpose_kv_cache_by_block, so
    #                                 flipping the switch produces an
    #                                 attributed refusal instead of an
    #                                 EXEC_NPU_CMD failure.
    #   GroupedMatmulSwigluQuantWeightNzTensorList
    #                                 eplb_config.dynamic_eplb
    #                                 (ascend_config.py:116, default False) AND
    #                                 ascend_fusion_config.fusion_ops_gmmswigluquant
    #                                 (:102, default True) AND a W8A8-dynamic
    #                                 checkpoint AND enable_custom_op().
    #   AddRmsNormBias                no user switch: enable_custom_op() alone
    #                                 (utils.py:419) for ops/layernorm.py:73,99;
    #                                 the fusion-pass patterns additionally need
    #                                 additional_config.ascend_compilation_config
    #                                 .fuse_norm_quant (ascend_config.py:70,
    #                                 default True).
    #                                 R55: the two layernorm.py call sites now
    #                                 refuse through custom_op_guards
    #                                 .require_add_rms_norm_bias. The THREE
    #                                 norm_quant_fusion_pass.py pattern
    #                                 declarations are NOT guarded -- see that
    #                                 module's docstring for why, and the R55
    #                                 suite scans for them on every run.
    #
    # enable_custom_op() is FALSE ON A5 (hardware_profile.py's A5 capability set
    # has no RUNTIME_CUSTOM_OPS), which is what holds the last three off today --
    # a hardware probe, not a decision anybody made about these operators. The
    # defaults above are therefore misleading read on their own: two of them say
    # True.
    #
    # R55 addressed exactly those two (TransposeKvCacheByBlock, AddRmsNormBias):
    # their call paths now RAISE with an attribution instead of being rescued by
    # enable_custom_op(). The criterion there is whether a kernel is installed,
    # NOT whether enable_custom_op() is true -- the flag is the accident, not the
    # decision. The other five in this list are still held off by their own
    # switches only.
    # ----------------------------------------------------------------------
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
    # DeepSeek-V4.1 mHC pre: "1" (default) uses npu_mhc_pre + npu_mhc_sinkhorn
    # instead of upstream's torch reference; "0" selects the torch reference.
    # On by default since the 2026-09-19 4-card A/B (together with the inverse
    # RoPE operator: TPOT -25..-38%, 12/12 cells separated).
    "VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR": lambda: os.getenv("VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR", "1"),
    # DeepSeek-V4.1: "1" carries each layer's configured compress ratio (0
    # included) to the metadata builder as its own spec field. Off: the
    # builder's compressor_ratio also sets tokens_per_state, so redirecting it
    # changes the page size too; enabling this requires splitting the two.
    "VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO": lambda: os.getenv("VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO", "0"),
    # DeepSeek-V4.1 sliding-window cache block size, in tokens. "0" (default)
    # keeps whatever the attention layer passed -- upstream hard-codes 32 at
    # vllm/models/deepseek_v4_1/attention.py:456. A positive value replaces it
    # and must appear verbatim in the SWA backend's supported kernel block
    # sizes. The SWA page is 32 tokens * 512 * 2 B = 32 KiB, which the planner
    # pads up to the 128 KiB full-MLA page (4.00x), and each of the 40
    # single-layer SWA groups then charges a whole 529 KiB pool block per
    # admitted block. 128 makes the page exact and cuts the admitted blocks
    # per request by 4. Ascend's own V4.0 path already runs SWA at
    # DSV4_BLOCK_SIZES[128][0][1] = 128 on the same backend.
    "VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE": lambda: int(os.getenv("VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE") or 0),
    # DeepSeek-V4.1 Engram storage: "ram" (default, legacy staged files)
    # or "ssd" (ordinary file-backed mmap; requires a verified manifest).
    # Non-sensitive; this selects backing storage, not the gather/compute path.
    "VLLM_ASCEND_DSV41_ENGRAM_BACKEND": lambda: os.getenv("VLLM_ASCEND_DSV41_ENGRAM_BACKEND", "ram"),
    # Absolute table directory. Empty (default) uses /dev/shm/engram_stage
    # for "ram"; "ssd" requires an explicit directory. Non-sensitive.
    "VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR": lambda: os.getenv("VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR", ""),
    # DeepSeek-V4.1 engram: "1" runs the n-gram hash, the id D2H, the host
    # gather and the H2D in the model runner before every forward, outside any
    # captured graph; the graph then only dequantises a persistent staging
    # buffer. "0" keeps the host gather inside the forward, recorded as a
    # host-function node under graph capture.
    "VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH": lambda: os.getenv("VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH", "0"),
    # DeepSeek-V4.1: "1" (default) undoes RoPE before the output projection with
    # one inplace_partial_rotary_mul over the rope slice (sin negated) instead
    # of the torch ops; "0" selects the torch ops. A5 honours partial_slice
    # bit-exactly (card A4). On by default since the 2026-09-19 4-card A/B
    # (together with the mHC pre operator).
    "VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR": lambda: os.getenv("VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR", "1"),
    # Forward RoPE (q and kv) through the same inplace_partial_rotary_mul the
    # inverse already uses. "1" (default) = operator, "0" = the torch
    # decomposition. ⚠ The two are not bit-identical: the operator rotates in
    # the tensor dtype, the torch form in float32 with one rounding at the
    # end. Kept switchable so an accuracy question can be answered by flipping
    # it rather than by reverting. Measured saving: ~1.5 ms of a 28.8 ms
    # bs=1 decode step (适配方案 §6.9.15).
    # mHC 的两处 torch 分解换融合件：RMSNorm 走 npu_rms_norm（模型里 q/kv norm
    # 已在用），stream collapse 走一次 bmm。"1"（默认）= 融合，"0" = 原 torch 式。
    # ⚠ 与 RoPE 那条不同，这两处【不保证逐位相同】：归约顺序不同，差几个 ULP。
    # 合计约 2.1 ms / 28.8 ms 一步（适配方案 §6.9.15）。
    "VLLM_ASCEND_DSV41_MHC_FUSED_OPS": lambda: os.getenv("VLLM_ASCEND_DSV41_MHC_FUSED_OPS", "1"),
    "VLLM_ASCEND_DSV41_FORWARD_ROPE_OPERATOR": lambda: os.getenv("VLLM_ASCEND_DSV41_FORWARD_ROPE_OPERATOR", "1"),
    # 融合 [T] -> [T, 2] 的 slot mapping：8 个算子 + 1 次 copy_ 压成 1 次下发，
    # 每步 42 组。设 0 退回 torch 分解（两条路逐位相同），用于二分回归。
    "VLLM_ASCEND_DSV41_FUSED_SLOT_MAPPING": lambda: os.getenv("VLLM_ASCEND_DSV41_FUSED_SLOT_MAPPING", "1"),
    # indexer 的 q RoPE / k_norm / k 尾部旋转换成已在树里跑着的算子（第 2 轮序 4）。
    # 🔴 k 侧的 rms_norm 是【非逐位】的：手写版中途那次 bf16 round trip 是喂给下游
    #    absmax 的（indexer_k_store.py:82-86），而 MXFP4 的块 scale 是
    #    exp2(ceil(log2(...)))，1 ulp 差跨过 2 的幂边界就整块跳一个 binade。
    #    设 0 退回手写版，用于把精度问题变成一次翻转而不是一次回退。
    "VLLM_ASCEND_DSV41_INDEXER_FUSED_ROPE": lambda: os.getenv("VLLM_ASCEND_DSV41_INDEXER_FUSED_ROPE", "1"),
    "VLLM_ASCEND_DSV41_INDEXER_FUSED_NORM": lambda: os.getenv("VLLM_ASCEND_DSV41_INDEXER_FUSED_NORM", "1"),
    # 视觉 MoE router 走融合算子（调两次 + where 实现按行选表），替掉
    # 702 个 torch 小算子 / 1.292 ms 每步。设 0 退回 torch 分解。
    # 两条路已在卡上对拍：topk_ids 逐位相同、权重差在 fp32 eps 量级。
    "VLLM_ASCEND_DSV41_FUSED_VISION_ROUTER": lambda: os.getenv("VLLM_ASCEND_DSV41_FUSED_VISION_ROUTER", "1"),
    # DeepSeek-V4.1 indexer K write: quantise and store through CANN's
    # indexer_quant_cache instead of indexer_compress_epilog_v2.
    # k_norm/RoPE/slot gating are unchanged either way.
    #
    # 🔴 Default "1" since the MXFP4 index K cache (batch 1). This is no
    # longer a performance choice: the cache row is 68 bytes (64 packed E2M1
    # values + 4 E8M0 scales) and indexer_quant_cache with
    # quant_mode="mxfp4" is the only writer that produces it.
    # indexer_compress_epilog_v2 writes the fp8 row (128 values + one fp32
    # scale = 132 bytes) and refuses this cache rather than filling it
    # (indexer_k_store._refuse_epilog_on_an_mxfp4_row).
    #
    # ⚠ So "0" is not a rollback on its own -- it has to move together with
    # the cache spec in caches.py, in one change. Half of it leaves two page
    # sizes in one KV cache group, and MLA specs cannot be page-padded
    # (vllm/v1/core/kv_cache_utils.py:1392-1397 raises).
    #
    # The earlier default was "0" pending a GSM8K/TPOT comparison on the card
    # (CARD_TEST_TABLE 戊); that comparison was of fp8-vs-fp8 quantisers and
    # no longer decides this flag.
    "VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE": lambda: os.getenv("VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE", "1"),
    # DeepSeek-V4.1 graph mode without VLLM_USE_BREAKABLE_CUDAGRAPH=1: "1"
    # allows it, anything else refuses to start. Default "0". Not a
    # performance switch -- it is the deliberate opt-out for the preflight in
    # platform.py:_require_validated_dsv41_graph_mode, which exists because
    # forgetting VLLM_USE_BREAKABLE_CUDAGRAPH=1 is otherwise silent: the run
    # takes the piecewise torch.compile + plain ACLGraph path that no DSV4.1
    # accuracy or soak number covers. Set it only to run that path on purpose.
    "VLLM_ASCEND_DSV41_ALLOW_UNVALIDATED_GRAPH_MODE": lambda: os.getenv(
        "VLLM_ASCEND_DSV41_ALLOW_UNVALIDATED_GRAPH_MODE", "0"),
    # DeepSeek-V4.1 MoE offload (M0c): output directory for the per-layer
    # routed-expert frequency counter (vllm_ascend/kt_offload/expert_counter.py).
    # Empty (default) = off. Diagnostic only. Does NOT require --enforce-eager
    # (T3.8/O4): whether a forward may be counted is decided per call from the
    # capture predicate, not from the config -- a captured or undecidable call
    # counts nothing, and replays, which run no Python, are reported as
    # unobservable rather than guessed. Read once at import. Not sensitive.
    "VLLM_ASCEND_DSV41_EXPERT_COUNT": lambda: os.getenv("VLLM_ASCEND_DSV41_EXPERT_COUNT", ""),
    # DeepSeek-V4.1 MoE offload: how many counted forwards between the expert
    # counter's periodic dumps ("200" = the default it always had; the exit dump
    # happens regardless). Only read when VLLM_ASCEND_DSV41_EXPERT_COUNT is set.
    #
    # Why it is settable: the counts are cumulative and one process writes one
    # file, so the ONLY way to attribute counts to a single request is to diff
    # two dumps taken at that request's boundaries -- and a dump only ever
    # happens inside a counted forward. At "1" every forward dumps, so the last
    # dump before a request ends IS its last forward, and consecutive
    # boundary snapshots slice per request exactly (bar the one-forward skew
    # noted in expert_counter.dump). Larger values leak up to that many forwards
    # of the previous request into the next slice.
    #
    # Cost at "1": one D2H of 2 x [40, 384] int64 (245 KiB) plus a torch.save
    # per forward. Diagnostic only, and the value is recorded in every dump
    # ("dump_every") so a reading can never be wrong about which granularity
    # produced it. Not a number below 1; anything unparsable keeps 200 and logs
    # a WARNING (this is a statistics hook inside the serving forward -- it must
    # not raise). Read once at import. Not sensitive.
    "VLLM_ASCEND_DSV41_EXPERT_COUNT_EVERY": lambda: os.getenv("VLLM_ASCEND_DSV41_EXPERT_COUNT_EVERY", "200"),
    # DeepSeek-V4.1 MoE offload: sampling interval in seconds for the per-layer
    # host-func counters (vllm_ascend/kt_offload/hostfunc_log.py). Only a
    # host-func node inside the graph increments them, so a layer whose count
    # does not grow while decoding has no CPU-expert work in the graph. "" or
    # "0" (default) = off. Diagnostic; read once at import. Not sensitive.
    "VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG", ""),
    # DeepSeek-V4.1 MoE offload: sampling interval in seconds for the host-func
    # seam timing (vllm_ascend/kt_offload/seam_timing.py), which splits the
    # 107.5 us/layer that NOTIFY_WAIT pays on top of kt's own compute into
    # pre / submit / sync / post. "" or "0" (default) = off. When on it only
    # reads clocks: no device operation and no output byte changes. Diagnostic;
    # read once per layer at weight-load time. Not sensitive.
    "VLLM_ASCEND_DSV41_KT_SEAM_TIMING": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_SEAM_TIMING", ""),
    # DeepSeek-V4.1 MoE offload: the side stream (the CPU-expert D2H +
    # host-func node captured onto a second stream, joined back in merge).
    # ON BY DEFAULT since MOE_OFFLOAD_PLAN §4.33, after the whole-network
    # acceptances §4.26 (bit-identical, GSM8K 50/50, 1319 = 1275/1319), §4.32
    # and §4.33 (merged version: decode 1170/1170 bit-identical, GSM8K 50/50,
    # 1319 = 1283/1319). Unset/"" (default) = follow
    # kt_offload_config.overlap_side_stream (default true); "0" = off whatever
    # the config says (the kill switch); "1" = on if the config allows it (the
    # pre-§4.33 explicit permit, now redundant). Any other value refuses to
    # start. Captured path only: eager never forks. Resolved together with the
    # three switches below in vllm_ascend/kt_offload/switches.py, the only
    # reader; the start-up log line "kt_offload switches" prints every
    # effective value and its source. 🔴 Every known way it fails is faster
    # and wrong: a change to it is judged by bit-identity, never TPOT.
    # Not sensitive.
    "VLLM_ASCEND_DSV41_KT_SIDE_STREAM": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_SIDE_STREAM", ""),
    # DeepSeek-V4.1 MoE offload, P1a: stop the device waiting for the host-func
    # callback to RETURN. The captured node then only builds kt's task and
    # submits it; the graph goes on to an ExternalEvent wait + reset, and a
    # completion thread calls cpu_infer.sync(0) and records that event
    # (vllm_ascend/kt_offload/async_wait.py). Saves the ACL notify-back leg of
    # the seam (PLAN §4.22 F1/F2: A+F 78 us/layer, event waits 0.03-0.14 us).
    # ON BY DEFAULT since MOE_OFFLOAD_PLAN §4.33 (accepted in §4.28: 0.53
    # ms/token, bit-identical; §4.32/§4.33 as part of the merged version).
    # Three states: unset/"" (default) = automatic -- on when the layer owns
    # its pinned buffers (packed_transfer or the side stream), off otherwise
    # and off when VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL=1, never an error;
    # "1" = on, and refuses to start without packed_transfer or the side
    # stream, or together with an explicit DEVICE_SIGNAL=1; "0" = off (the
    # callback keeps doing the sync). Any other value refuses to start.
    # Captured-path change only: in eager an automatic value is off.
    # Resolved in vllm_ascend/kt_offload/switches.py.
    # 🔴 Every way it can fail is FASTER and wrong (the H2D reads the previous
    # step's CPU output and nothing raises), so the criterion for any change
    # is bit-identity against the arm with it off, never TPOT. Not sensitive.
    "VLLM_ASCEND_DSV41_KT_ASYNC_WAIT": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_ASYNC_WAIT", ""),
    # DeepSeek-V4.1 MoE offload, P1b: take the host-func node out of the graph
    # altogether. The layer becomes D2H(inputs) -> D2H(8-byte step counter) ->
    # ExternalEvent wait -> reset -> H2D, and a resident polling thread watches
    # the counter word, runs kt and records the event
    # (vllm_ascend/kt_offload/device_signal.py). What it saves on top of P1a is
    # the dispatch leg AND the 35.75 us of device idle that sits in front of
    # every HOSTFUNC_CALLBACK and in front of no other node type (PLAN §4.22
    # F1). OFF BY DEFAULT, also after §4.33 (§4.30: the mechanism did not
    # hold on card). "1" = on; unset/"" or "0" = the graph keeps its host-func
    # node; any other value refuses to start. Needs the layer's OWN pinned
    # buffers (packed_transfer or the side stream) and graph mode. With it on,
    # an automatic (unset) VLLM_ASCEND_DSV41_KT_ASYNC_WAIT turns itself off;
    # both set to "1" explicitly refuses to start -- they start the same work
    # two ways. Resolved in vllm_ascend/kt_offload/switches.py. Builds a small
    # .so at start-up (the spin must not hold the GIL) and refuses to start if
    # it cannot.
    # 🔴 Every way it can fail is FASTER and wrong, so the criterion is
    # bit-identity against the arm with it off, never TPOT. Read at config
    # validation and at weight-load time. Not sensitive.
    "VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL", ""),
    # Where the polling helper of the switch above is compiled to. Empty
    # (default) = ~/.cache, then /tmp: the build has to land on a mount that is
    # not noexec, and this project points TMPDIR at /dev/shm, which often is.
    # Not sensitive.
    "VLLM_ASCEND_DSV41_KT_POLL_CACHE": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_POLL_CACHE", ""),
    # DeepSeek-V4.1 MoE offload: move the shared-expert MLP in front of the kt
    # merge, so it runs inside the window where the main stream is only waiting
    # for the CPU experts. The shared expert reads the MoE layer's input hidden
    # states and depends on neither routed_out nor the CPU half; it sits behind
    # the join purely because of call order (merge() is called inside
    # routed_experts.forward_impl, the shared MLP only after it returns). With
    # "1" the routed path hands back an UN-merged routed_out and
    # fused_moe._forward_impl merges after ascend_shared_experts.forward(),
    # still before it returns, so nothing reads routed_out in between.
    # ON BY DEFAULT since MOE_OFFLOAD_PLAN §4.33 (the D arm of the second
    # merged-version acceptance: decode 1170/1170 bit-identical against C,
    # GSM8K 50/50, 1319 = 1283/1319). Three states: unset/"" (default) =
    # follow the side stream (on when it is on; off in eager); "1" = on (a
    # warning without the side stream: correct, but no idle window to fill);
    # "0" = the merge stays where it is. Any other value refuses to start.
    # Resolved in vllm_ascend/kt_offload/switches.py.
    # 🔴 A dropped merge is SILENT (the CPU half of the experts vanishes and
    # the tensor still looks fine), so the criterion is bit-identity against
    # the arm with it off, never TPOT; the layer counts its merge calls per
    # route (vllm_ascend/kt_offload/shared_window.py). Not sensitive.
    "VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW", ""),
    # DeepSeek-V4.1 MoE offload: pin the ONE ACL report thread (the thread
    # torch_npu creates on the first _subscribe_report and that delivers every
    # _launch_host_func callback) to a PU kt's spinning workers do not own.
    # "" (default) = never call sched_setaffinity, byte-for-byte the current
    # behaviour. A number = that PU. "auto" = model kt's own binding rule
    # (vllm_ascend/kt_offload/report_affinity.py) and take the last fully idle
    # core, preferring the device's NUMA node; auto refuses start-up when it
    # cannot prove which PUs kt holds. Resolved once at weight-load time,
    # applied inside the first host-func callback (the only code of ours that
    # runs on that thread), and read back with sched_getaffinity. Not sensitive.
    "VLLM_ASCEND_DSV41_KT_REPORT_CPU": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_REPORT_CPU", ""),
    # DeepSeek-V4.1 MoE offload: additionally put that report thread on
    # SCHED_FIFO at this priority (1-99). "" or "0" (default) = off. Requires
    # CAP_SYS_NICE; without it the call fails and the service continues on
    # SCHED_OTHER with a warning, and the log/artifact records whether it took
    # effect (the policy is read back). Only read when
    # VLLM_ASCEND_DSV41_KT_REPORT_CPU is set. Not sensitive.
    "VLLM_ASCEND_DSV41_KT_REPORT_FIFO": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_REPORT_FIFO", ""),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
