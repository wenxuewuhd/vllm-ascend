import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import torch
import torch.distributed as dist
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.aclnn_availability import require_aclnn_op
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_attn_kv_plan import (
    dsv4_indexer_qk_dtype,
    dsv4_indexer_scale_dtype,
    get_dsa_attn_kv_plan,
    is_a5_bf16_kv_enabled,
    qli_v2_quant_mode,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_pcp,
    get_or_register_attention_buffer,
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.distributed.parallel_state import get_otp_group
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata
from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerMetadata, IndexerOverlapPlan
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa, get_full_cos_and_sin_dsa
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import get_potential_max_tokens, is_deepseek_v4_family, npu_stream_switch, oproj_tp_enable
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore


# The SAS and QLI metadata operators use a fixed 1024-element int32 layout.
DSA_METADATA_BUFFER_SIZE = 1024

# --------------------------------------------------------------------------
# npu_rms_norm_dynamic_quant: a name that another operator answers to.
#
# ``RmsNormDynamicQuant`` is declared in vllm-ascend's own csrc, but its
# ``AddConfig`` lists only ascend910 variants, so no A5 operator package
# contains it. ``GetOpApiFuncAddr`` (csrc/aclnn_torch_adapter/op_api_common.h)
# searches the vendor packages first and then falls through to CANN's own
# ``libopapi.so`` -- which EXPORTS A SYMBOL OF THE SAME NAME. dlsym succeeds,
# the TORCH_CHECK inside EXEC_NPU_CMD passes, and the call proceeds against a
# different operator:
#
#   CANN 9.2 aclnnop/aclnn_rms_norm_dynamic_quant.h declares
#       x, gamma, smoothScalesOptional, betaOptional, epsilon, dstType,
#       yOut, scaleOut                                              -> 8
#   csrc/torch_binding.cpp's npu_rms_norm_dynamic_quant_npu passes
#       x, gamma, smooth_scale, smooth_scale2, beta, epsilon, output_mask,
#       dst_type, y_out, y2_out, scale_out, scale2_out              -> 12
#
# Nothing in the stack compares those. An arity mismatch is not reported as an
# error at all; it is an ABI violation whose cost is unrelated to its nature.
#
# The branch below is unreachable today only because DSV4.1 is MXFP8/MXFP4 and
# ``_is_w8a8_dynamic`` is false. "Currently unreachable" is not "fixed": the
# first person to run a W8A8 DSA on A5 walks straight into it. Hence a guard
# that RAISES at the call site rather than a comment.
#
# 12 is a literal copied out of another repository file, so it is re-derived by
# the R50 test suite from ``csrc/torch_binding.cpp`` at the revision under
# test; if somebody edits that EXEC_NPU_CMD the test goes red rather than this
# constant going quietly stale.
_RMS_NORM_DYNAMIC_QUANT_ARG_COUNT = 12


def _require_rms_norm_dynamic_quant():
    """Refuse the W8A8 q_norm path unless OUR operator is the one installed.

    What this can judge:
      * the torch schema is registered (it always is -- registration is
        unconditional, which is why it cannot be the criterion);
      * whether the aclnn entry points resolve at all;
      * WHICH library they resolve out of -- the discriminator that matters
        here, because a name-existence check reads identically whether the
        operator is installed or whether CANN's same-named one answered;
      * how many operator arguments the INSTALLED declaration takes, when a
        declaration can be found, against the 12 this build passes.

    What this cannot judge, and says so in the message instead of pretending:
      * the argument ORDER and MEANING of whatever did resolve -- a shared
        object carries neither;
      * the arity of an operator supplied by a vendor package, because vendor
        packages ship no aclnnop header to read;
      * ``vllm_ascend/attention/context_parallel/dsa_cp.py`` has the same call
        and is NOT guarded from here. It is owned by another change in flight;
      * whether CANN's implementation is semantically ours when the arity
        happens to match. It reports the provenance mismatch regardless.
    """
    return require_aclnn_op(
        torch,
        torch_npu,
        torch_name="npu_rms_norm_dynamic_quant",
        aclnn_api="aclnnRmsNormDynamicQuant",
        op_type="RmsNormDynamicQuant",
        feature="W8A8-dynamic DSA q_norm (npu_rms_norm_dynamic_quant)",
        expected_arg_count=_RMS_NORM_DYNAMIC_QUANT_ARG_COUNT,
        require_vendor_package=True,
    )

CompressorMetadataOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

_DSV4_DSA_OVERLAP_STREAM = None
CompressorForwardOutput = tuple[torch.Tensor, torch.Tensor]
CompressorOverlapOutput = tuple[CompressorForwardOutput, torch.npu.Event]

_COMPRESSOR_METADATA_CACHE_KEY = "dsv4_compressor_metadata_cache"


def reset_compressor_metadata_cache() -> None:
    """Drop the cached compressor metadata before a composite forward enters its next substep."""
    try:
        forward_context = get_forward_context()
    except AssertionError:
        # No active forward context (e.g. draft-model unit tests): no cache to reset.
        return
    forward_context.additional_kwargs.pop(_COMPRESSOR_METADATA_CACHE_KEY, None)


def get_or_compute_compressor_metadata(
    metadata: Any,
    compress_ratio: int,
    vllm_config: VllmConfig,
) -> CompressorMetadataOutput:
    """Compute the DSV4 compressor metadata once per cache group in the current substep.

    DSV4 runs one :class:`AscendCompressor` forward per attention layer, but every
    layer in a cache group consumes the same compressor metadata. The outputs are
    therefore memoized on the current forward context so the device-side
    ``compressor_metadata`` operator runs only once per group.
    """
    cache_group_key = getattr(metadata, "cache_group_key", "")
    if not cache_group_key:
        raise ValueError(
            "DSV4 compressor metadata requires a cache group key; the AscendDSAReqMetadata builder did not set one"
        )
    forward_context = get_forward_context()
    cache: dict[str, CompressorMetadataOutput] = forward_context.additional_kwargs.setdefault(
        _COMPRESSOR_METADATA_CACHE_KEY,
        {},
    )
    cached_metadata = cache.get(cache_group_key)
    if cached_metadata is not None:
        return cached_metadata

    assert metadata.full_compress_cos is not None
    assert metadata.full_compress_sin is not None
    assert metadata.num_compressed_tokens is not None
    assert metadata.start_pos is not None
    assert metadata.num_actual_reqs is not None
    full_compress_cos = metadata.full_compress_cos.view(
        metadata.full_compress_cos.shape[0],
        metadata.full_compress_cos.shape[-1],
    )
    full_compress_sin = metadata.full_compress_sin.view(
        metadata.full_compress_sin.shape[0],
        metadata.full_compress_sin.shape[-1],
    )
    computed_metadata = torch.ops._C_ascend.compressor_metadata(
        full_compress_cos,
        full_compress_sin,
        metadata.query_start_loc,
        metadata.start_pos,
        metadata.block_table,
        metadata.storage_block_size,
        get_dsa_attn_kv_plan(vllm_config).get_dsa_compressor_slot_mapping_format(),
        compress_ratio,
        metadata.num_compressed_tokens,
        metadata.num_actual_reqs,
    )
    cache[cache_group_key] = computed_metadata
    return computed_metadata


def build_compressor_metadata_out(
    metadata: Any,
    compress_ratio: int,
    outputs: CompressorMetadataOutput,
    vllm_config: VllmConfig,
) -> None:
    assert metadata.full_compress_cos is not None
    assert metadata.full_compress_sin is not None
    assert metadata.start_pos is not None
    assert metadata.num_actual_reqs is not None
    full_compress_cos = metadata.full_compress_cos.view(
        metadata.full_compress_cos.shape[0],
        metadata.full_compress_cos.shape[-1],
    )
    full_compress_sin = metadata.full_compress_sin.view(
        metadata.full_compress_sin.shape[0],
        metadata.full_compress_sin.shape[-1],
    )
    torch.ops._C_ascend.compressor_metadata_out(
        full_compress_cos,
        full_compress_sin,
        metadata.query_start_loc,
        metadata.start_pos,
        metadata.block_table,
        metadata.storage_block_size,
        get_dsa_attn_kv_plan(vllm_config).get_dsa_compressor_slot_mapping_format(),
        compress_ratio,
        metadata.num_actual_reqs,
        *outputs,
    )


def dsv4_dsa_overlap_stream() -> torch.npu.Stream:
    global _DSV4_DSA_OVERLAP_STREAM
    if _DSV4_DSA_OVERLAP_STREAM is None:
        _DSV4_DSA_OVERLAP_STREAM = torch_npu.npu.Stream()
    return _DSV4_DSA_OVERLAP_STREAM


def _is_w8a8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8DynamicLinearMethod``."""
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None or isinstance(quant_method, AscendUnquantizedLinearMethod):
        return False
    inner_method = getattr(quant_method, "quant_method", None)
    return isinstance(inner_method, AscendW8A8DynamicLinearMethod)


def _has_weight_scale(linear) -> bool:
    return getattr(linear, "weight_scale", None) is not None


def _dsa_layout_kv(vllm_config: VllmConfig) -> str:
    return get_dsa_attn_kv_plan(vllm_config).layout_kv


def _dsa_swa_only_cmp_ratio(compress_ratio: int, vllm_config: VllmConfig) -> int:
    """Map a layer's configured ratio onto the operator's ``cmp_ratio``.

    The configured ratio is 0 on layers with no compressed stream, while
    sparse_flash_mla rejects 0 ("Cmp_ratio must be greater than 0"). So the
    A5 BF16 plan reports 1 for ratio <= 1 and simply supplies no compressed
    stream where there is none. Ratio 2 is passed through: CANN 9.2 accepts
    it together with the sliding window (see sparse_attn_args).
    """
    if is_a5_bf16_kv_enabled(vllm_config) and compress_ratio <= 1:
        return 1
    return max(compress_ratio, 1)


def is_deepseek_v41_text(hf_text_config) -> bool:
    """V4.1 only. V4.0's descriptor values are the defaults and stay put."""
    return str(getattr(hf_text_config, "model_type", "")).startswith(
        "deepseek_v41")


def is_deepseek_v41_model(vllm_config) -> bool:
    """Is the model this impl was built for a V4.1 one?

    ``is_deepseek_v41_text`` above asks a single config object for its
    ``model_type``. That is the right question where it is used -- a
    descriptor built from the target's text config -- and the wrong one here,
    because the one way a V4.1 checkpoint reaches this class is the
    speculative-decoding path, and upstream's dspark branch *overwrites* the
    draft instance's ``model_type`` with ``"deepseek_v4"``
    (``vllm/config/speculative.py``, the ``method == "dspark"`` branch). The
    field that would be read is exactly the field that was rewritten.

    So three signals are tried, none of which that rewrite touches:

      * ``kv_source_layer_ids``: the structural one, and the same field
        ``attends_compressed_stream`` below already uses to recognise the
        V4.1 topology "whatever it calls itself". V4.1 names its four
        kv-source layers; V4.0 has no such key.
      * the ``model_type`` of the config and of the text config, for a config
        that has not been through the rewrite (``"deepseek_v41"`` /
        ``"deepseek_v41_text"``).
      * ``type(hf_config).model_type``: the *class* attribute
        (``DeepseekV41Config.model_type == "deepseek_v41"``), which an
        assignment to the instance leaves alone.

    A V4.0 config answers no to all three, so every V4.0 caller keeps the
    behaviour it has today.
    """
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "kv_source_layer_ids", None) is not None:
        return True
    hf_text_config = getattr(model_config, "hf_text_config", None)
    return (is_deepseek_v41_text(hf_config)
            or is_deepseek_v41_text(hf_text_config)
            or is_deepseek_v41_text(type(hf_config))
            or is_deepseek_v41_text(type(hf_text_config)))


def applies_dsa_q_rms(vllm_config) -> bool:
    """Does this checkpoint want the second, weightless per-head q RMSNorm?

    V4.0 does: after ``wq_b`` its q goes through a weightless RMSNorm before
    RoPE, which is what ``DeviceOperator.apply_dsa_q_rms`` performs.

    V4.1 does not. Its reference implementation goes ``wq_b`` straight to
    ``apply_rotary_emb`` (inference/model.py, ``Attention.forward``);
    upstream's fused prolog is called with ``apply_q_norm=False`` at every
    V4.1 dtype branch; the official v27 Ascend V4.1 DSpark attention sets
    ``impl.apply_q_norm = False`` with the comment "V4.1 applies Q LoRA
    RMSNorm only"; and this tree's own V4.1 attention path skips it
    (``vllm_ascend/models/deepseek_v4_1/qkv_prolog.py``). Applying it anyway
    normalises q a second time -- no shape changes and nothing raises, so it
    would be a silent accuracy defect.
    """
    return not is_deepseek_v41_model(vllm_config)


def attends_compressed_stream(compress_ratio: int, vllm_config) -> bool:
    """Does this layer's attention read a compressed stream?

    Asked at three places that must agree -- whether to pass the compressed
    cumulative lengths, whether to supply the stream itself, and which mask
    the original stream gets.

    It is deliberately *not* the same question as "does this layer own a
    compressed cache". V4.1 has four source layers and thirty-eight that attend
    to one, so a single predicate would be wrong for every consumer.

    V4.0 and V4.1 disagree on the threshold. V4.0 pools by 4 or 128 and treats
    ratio 1 as "sliding window only"; V4.1 gives layers 20..39 ratio 1 meaning
    "compressed, pooling disabled" -- a stream that is read. The two are told
    apart by whether the checkpoint declares ``kv_source_layer_ids``, which is
    the field that actually answers the question rather than a version label:
    a config that names its source layers has the V4.1 topology whatever it
    calls itself.
    """
    # hf_config: the checkpoint's JSON nests these fields under text_config,
    # but DeepseekV41Config exposes them at the top level, and
    # get_hf_text_config returns that same object.
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    declares_sources = getattr(hf_config, "kv_source_layer_ids", None) is not None
    if declares_sources:
        return compress_ratio >= 1
    # V4.0: unchanged from the `compress_ratio <= 1` this replaced.
    return compress_ratio > 1


#: Union over every plan's set, for the few call sites that run before a plan
#: is resolved. Deliberately permissive: refusing there would reject ratios a
#: plan accepts, and the per-plan check still runs later.
ANY_PLAN_COMPRESS_RATIOS = (0, 1, 2, 4, 128)


def validate_compress_ratio(compress_ratio: int, where: str, plan=None) -> int:
    """Reject ratios the dispatched operator cannot honour.

    The supported set belongs to the plan, not to this module: the three plans
    in dsa_attn_kv_plan call three different operators, and DeepSeek-V4.1's
    ratio 2 is accepted by one of them and not the others. Hard-coding one set
    here made this guard plan-blind -- it refused ratio 2 even after switching
    to a plan built for it.

    ``plan`` may be omitted only where no plan is reachable yet; the union of
    every plan's set is used then, which rejects nothing a plan would accept.
    """
    if plan is not None:
        supported = tuple(plan.supported_compress_ratios)
        scope = f"the {type(plan).__name__} in use"
    else:
        supported = ANY_PLAN_COMPRESS_RATIOS
        scope = "any plan (no plan resolved at this point)"

    if compress_ratio not in supported:
        raise ValueError(
            f"{where}: compress_ratio={compress_ratio} is not supported by "
            f"{scope}, which accepts {list(supported)}. DeepSeek-V4.1 needs "
            "ratio 2 on layers 2..19 and 1 on 20..39, and the plans differ in "
            "which they accept -- this is about which operator is dispatched, "
            "not about what the hardware can do.")
    return compress_ratio


class AscendDSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA"

    @staticmethod
    def get_builder_cls():
        from vllm_ascend.utils import enable_dsa_cp

        use_dsa_cp = enable_dsa_cp()
        use_pcp = enable_pcp()
        if use_dsa_cp and use_pcp:
            raise ValueError("Legacy DSACP and PCP cannot be enabled at the same time.")
        if use_dsa_cp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder

            return AscendDSACPMetadataBuilder
        if use_pcp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPMetadataBuilder

            return AscendDSAPCPMetadataBuilder
        return AscendDSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_scale_shape(num_blocks: int, block_size: int, scale_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, scale_size

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase[Any]]:
        from vllm_ascend.utils import enable_dsa_cp

        use_dsa_cp = enable_dsa_cp()
        use_pcp = enable_pcp()
        if use_dsa_cp and use_pcp:
            raise ValueError("Legacy DSACP and PCP cannot be enabled at the same time.")
        if use_dsa_cp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl

            return AscendDSACPImpl
        if use_pcp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPImpl

            return AscendDSAPCPImpl
        return AscendDSAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8, 16, 32, 64, 128]


class AscendDSAC2Backend(AscendDSABackend):
    """DeepSeek-V4.1's ratio-2 compressed indexer/KV cache.

    The listed sizes are not a free choice. ``AscendMLAAttentionSpec`` recovers the
    physical page from the logical one by dividing the ratio back out
    (``core/kv_cache_interface.py``: ``storage_block_size = block_size //
    tokens_per_state``, and ``real_page_size_bytes`` is computed from that), so a
    spec's ``block_size`` is always already multiplied by the ratio. At ratio 2
    Ascend's physical 32/64/128-token pages therefore present as 64/128/256 raw
    scheduler tokens, and those three numbers are exactly what
    ``select_common_block_size`` is handed.

    The derivation only runs when the caller leaves ``storage_block_size``
    unset. A caller that passes it explicitly escapes the convention, and this
    list would then be wrong for that caller -- so pass ``block_size`` already
    multiplied and let the spec derive the rest, as every DSV4 site does.

    Do NOT reuse ``AscendDSAC4Backend`` for ratio 2. Its list is [128, 256, 512],
    which covers ratio 2 at cache block_size 64 and 128 and fails only at 32 --
    i.e. it passes on the default block size and breaks on the small one. That is
    verified as the red case in ``tests/ut/attention/test_dsa_c2_backend.py``.

    V4.1 uses ratios 0, 1 and 2 only (``vllm/models/deepseek_v4_1/attention.py``:
    ``compress_ratio not in (0, 1, 2)`` raises). Ratio 1 needs [32, 64, 128],
    which ``AscendDSABackend`` and ``AscendDSASWABackend`` already list, so no
    ratio-1 class is needed. Ratio 0 means sliding window and must be mapped to 1
    before it reaches the spec, or the page size divides to zero.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C2"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Align with upstream's logical block-size contract: Ascend's physical
        # 32/64/128-token C2 pages represent 64/128/256 raw scheduler tokens.
        return [64, 128, 256]


class AscendDSAC4Backend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C4"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Align with upstream's logical block-size contract: Ascend's physical
        # 32/64/128-token C4 pages represent 128/256/512 raw scheduler tokens.
        return [128, 256, 512]


class AscendDSAC128Backend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C128"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Align with upstream's logical block-size contract: Ascend's physical
        # 32/64/128-token C128 pages represent 4096/8192/16384 raw scheduler tokens.
        return [4096, 8192, 16384]


class AscendDSASWABackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [32, 64, 128]


class AscendDSAC4StateBackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C4_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8]


class AscendDSAC128StateBackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C128_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        if get_current_hardware_profile().supports(HardwareCapability.DSA_C128_STATE_SMALL_BLOCK_SIZES):
            return [4, 8, 16]
        return [8, 16, 32]


@dataclass
class AscendDSAReqMetadata:
    """Unified metadata for all requests in one attention invocation."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor | None
    storage_block_size: int
    query_start_loc: torch.Tensor

    num_compressed_tokens: int | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    full_compress_sin: torch.Tensor = None
    full_compress_cos: torch.Tensor = None
    start_pos: torch.Tensor | None = None
    num_actual_reqs: int | None = None
    # Every layer of a cache group receives the same builder output, so this
    # key lets the compressor metadata be computed once per group per substep.
    cache_group_key: str = ""
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    qli_cu_seqlens_q: torch.Tensor = None
    qli_seqused_k: torch.Tensor = None
    qli_cmp_residual_k: torch.Tensor = None
    compressor_metadata: CompressorMetadataOutput | None = None
    compressor_metadata_group_id: int | None = None
    attn_mask: torch.Tensor | None = None
    cu_cmp_seqlen_list: torch.Tensor = None
    ori_win_left: int | None = None
    ori_win_right: int | None = None
    dspark_swa_indices: torch.Tensor | None = None
    vision_swa_indices: torch.Tensor | None = None
    #: The per-token slots for this group, published whatever the ratio.
    #: slot_mapping above is None for compressing groups by design; this is
    #: not, because V4.1's indexer K writer needs it at every ratio. Placed
    #: last because a defaulted field cannot precede the undefaulted ones.
    token_slot_mapping: torch.Tensor | None = None
    #: Valid entries per query row in ``ori_sparse_indices``, int32.
    #:
    #: Not optional in practice whenever indices are supplied. The operator
    #: reads ``oriSparseBlockCount = hasOriTopkLength ? Min(length, width) :
    #: width`` -- so omitting it declares the *whole row* valid, and a row
    #: padded with -1 then sends the kernel to gather at those slots.
    #:
    #: Carried from whoever built the table (the per-row count of its
    #: ``visible`` mask), never recounted at the call site.
    ori_topk_length: torch.Tensor | None = None
    #: Compressed-stream length per request, ``seq_lens // cmp_ratio``, int32.
    #:
    #: Not ``cu_cmp_seqlen_list``. That one is cumulative and is what V4.0
    #: sends as the operator's ``cu_seqlens_cmp_kv`` under layout TND; this is
    #: one value per request and is what ``seqused_cmp_kv`` means under a
    #: paged layout. They differ in length by exactly one, which is the only
    #: cheap way to tell them apart -- their dtype, device and plausibility
    #: are identical.
    seqused_cmp_kv: torch.Tensor | None = None


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # The dimension of the attention heads
    head_dim: int | None = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    req_metadata: AscendDSAReqMetadata | None = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: torch.Tensor | None = None

    # Longest sequence in the batch, from common_attn_metadata. Already a
    # Python int there -- computed by the runner from the CPU copy of seq_lens
    # -- so carrying it costs nothing and saves a device read.
    max_seq_len: int = 0

    # This group's compression ratio. Carried so block_size below can be right
    # for every group rather than only for the ones where the two page sizes
    # coincide. 1 for SWA and for ratio-1 layers; 2 for V4.1's ratio-2 layers.
    tokens_per_state: int = 1

    # Names upstream's DeepSeek-V4.1 indexer reads straight off this object
    # (vllm/models/deepseek_v4_1/attention.py). Ascend keeps the
    # per-request payload one level down, in req_metadata, so without these the
    # V4.1 layer fails with an AttributeError before reaching any operator.
    # Added rather than rearranged: nothing in this tree branches on their
    # presence, and both derive from values already built.

    @property
    def slot_mapping(self) -> torch.Tensor | None:
        """Per-token slots, for upstream's V4.1 readers.

        The field of the same name on AscendDSAReqMetadata is deliberately None
        when the group compresses: that stream is scattered by its own operator
        and a per-token mapping there would be misleading. V4.1's indexer K
        writer needs one at every ratio -- it writes the indexer cache at group
        boundaries, and _produce_k reads it through this name -- so the null is
        kept as a field and answered here, from the mapping the builder
        publishes unconditionally.

        Scoped by being a property: V4.0 reads the field directly and is
        unaffected.
        """
        if self.req_metadata is None:
            return None
        if self.req_metadata.slot_mapping is not None:
            return self.req_metadata.slot_mapping
        return self.req_metadata.token_slot_mapping

    @property
    def num_prefill_tokens(self) -> int:
        return self.num_actual_tokens - self.num_decode_tokens

    @property
    def query_start_loc(self) -> torch.Tensor | None:
        return None if self.req_metadata is None else self.req_metadata.query_start_loc

    @property
    def block_table(self) -> torch.Tensor | None:
        return None if self.req_metadata is None else self.req_metadata.block_table

    @property
    def block_size(self) -> int | None:
        """The logical block size: raw scheduler tokens per block.

        Not simply ``storage_block_size``. Ascend holds the storage page fixed
        and multiplies the spec's ``block_size`` by the compression ratio, so
        the two coincide only at ratio 1. A property returning the storage page
        would be right for SWA -- where every reader of this happens to be --
        and silently wrong by a factor of two for V4.1's ratio-2 groups, while
        looking equally authoritative to both.

        Reconstructed from the ratio instead, so there is no group for which it
        is the wrong number. Upstream asserts ``swa_kv_cache.shape[1:] ==
        (block_size, head_dim)`` against it, which is the check that would have
        caught the storage-page version on SWA and passed it everywhere else.
        """
        if self.req_metadata is None:
            return None
        return self.req_metadata.storage_block_size * max(self.tokens_per_state, 1)


DSAMetadataDict: TypeAlias = dict[str, AscendDSAMetadata]


@dataclass(frozen=True)
class AscendDSALayerMetadata:
    attention: AscendDSAMetadata | None
    swa: AscendDSAMetadata
    compressor: AscendCompressorMetadata | None = None
    indexer: AscendIndexerMetadata | None = None


def _require_req_metadata(metadata: AscendDSAMetadata) -> AscendDSAReqMetadata:
    assert metadata.req_metadata is not None
    return metadata.req_metadata


def get_dspark_sparse_sas_window(vllm_config: Any) -> tuple[int, int]:
    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    block_size = vllm_config.speculative_config.num_speculative_tokens
    return window_size + block_size - 1, 0


def _aligned_dspark_index_width(window_size: int, block_size: int, alignment: int = 128) -> int:
    min_width = int(window_size) + int(block_size)
    return ((min_width + alignment - 1) // alignment) * alignment


def build_dspark_swa_indices(
    block_table: torch.Tensor,
    num_speculative_tokens: int,
    window_size: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_decode_tokens: int | None = None,
    index_width: int | None = None,
    indices_output: torch.Tensor | None = None,
    buffer: torch.Tensor | None = None,
    lengths_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DSpark non-causal visible slot ids for a paged SWA cache.

    Each token in a draft block sees the trailing context window plus the
    whole current draft block. Invalid/padded rows get lens=0 and -1 slots.

    When ``buffer`` is given, the per-token slots are copied into its leading
    rows and the returned tensor is a slice view of ``buffer``. This keeps the
    address stable across async ACL-graph replays, where the DSA operator
    captures ``ori_sparse_indices``'s data pointer at capture time.
    """
    if index_width is None:
        index_width = _aligned_dspark_index_width(window_size, num_speculative_tokens)
    min_width = int(window_size) + int(num_speculative_tokens)
    if index_width < min_width:
        raise ValueError(
            "DSpark SWA index_width must cover window_size + block_size: "
            f"index_width={index_width}, required={min_width}"
        )
    if query_start_loc is None or seq_lens is None:
        raise ValueError("DSpark SWA query_start_loc and seq_lens must both be provided")

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    prefix_lens = seq_lens - query_lens
    start_pos = (prefix_lens - int(window_size)).clamp(min=0)
    visible_lens = seq_lens - start_pos

    # Per-request visible-position grid [req_count, index_width]. Columns
    # j >= visible_len are out of range and masked to -1 below.
    cols = torch.arange(index_width, device=start_pos.device)
    col_mask = cols[None, :] < visible_lens[:, None]
    pos = start_pos[:, None] + cols[None, :]
    block_nums = pos // block_size
    # Clamp to valid block-table columns so gather never goes OOB on the
    # out-of-range columns (their results are discarded by col_mask anyway).
    safe_nums = block_nums.clamp(min=0, max=int(block_table.shape[1]) - 1)
    block_offsets = pos % block_size
    block_ids = torch.gather(block_table, 1, safe_nums)
    slot_ids = (block_ids * block_size + block_offsets).to(torch.int32)
    slot_ids = slot_ids.where(col_mask, torch.full_like(slot_ids, -1))

    per_token_slots = torch.repeat_interleave(slot_ids, query_lens, dim=0, output_size=num_decode_tokens).unsqueeze(1)
    per_token_lens = torch.repeat_interleave(visible_lens, query_lens, dim=0, output_size=num_decode_tokens)

    if indices_output is not None:
        if indices_output.shape != per_token_slots.shape:
            raise ValueError(
                "DSpark SWA indices output shape does not match active metadata: "
                f"output={tuple(indices_output.shape)}, active={tuple(per_token_slots.shape)}"
            )
        indices_output.copy_(per_token_slots)
        per_token_slots = indices_output

    if buffer is not None:
        # Copy the freshly built indices into the caller-provided buffer and hand
        # back a zero-copy view of it: ACL graph capture freezes tensor addresses,
        # so the DSA operator must read from the stable buffer at replay instead of
        # a freshly allocated tensor.
        num_rows = per_token_slots.shape[0]
        assert num_rows <= buffer.shape[0], (
            f"dspark_swa_indices needs {num_rows} rows but `buffer` only has {buffer.shape[0]}"
        )
        buffer[:num_rows].copy_(per_token_slots)
        per_token_slots = buffer[:num_rows]

    if lengths_output is not None:
        if lengths_output.shape != per_token_lens.shape:
            raise ValueError(
                "DSpark SWA lengths output shape does not match active metadata: "
                f"output={tuple(lengths_output.shape)}, active={tuple(per_token_lens.shape)}"
            )
        lengths_output.copy_(per_token_lens)
        per_token_lens = lengths_output
    return per_token_slots, per_token_lens


def mm_ranges_have_spans(mm_ranges) -> bool:
    """Whether these multimodal document ranges contain an actual image span.

    ``mm_req_doc_ranges`` is ``{req_index: [(start, end), ...]}`` and the
    runner builds it for **every** request whenever the checkpoint is a
    mm-prefix LM, appending nothing when a request carries no image
    (``model_runner_v1.py``). A text-only request on V4.1-Flash
    therefore arrives as ``{0: []}`` -- a truthy dict of empty lists.

    Testing the dict rather than its contents would put every text-only
    prefill through the bidirectional-vision index path. V4.0 never meets
    this because a text-only V4.0 checkpoint has ``vision_n_layers = 0``,
    which zeroes ``max_image_tokens``; V4.1-Flash is multimodal
    (``is_mm_prefix_lm``) while the request is not.

    The vision path is the wrong configuration for such a request: an
    explicit index list puts the operator in ori-only sparse mode (mask 0,
    window bounds disabled) with an index width of ``sliding_window +
    vision_max_n_token``, where plain sliding window passes mask 4 and no
    list at all.
    """
    if not mm_ranges:
        return False
    values = getattr(mm_ranges, "values", None)
    if values is None:
        # Not a mapping. Fall back to the container's own truthiness rather
        # than guessing at a shape.
        return True
    has_spans = any(bool(ranges) for ranges in values())
    if not has_spans:
        from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

        # Announced from where the branch is taken.
        announce_once(
            "mm_ranges_require_spans",
            "a request whose multimodal ranges hold no span takes no vision "
            "index path.")
    return has_spans


def build_vision_bidirectional_swa_indices(
    block_table: torch.Tensor,
    window_size: int,
    max_image_tokens: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    mm_prefix_ranges: dict[int, list[tuple[int, int]]],
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paged original-KV indices with bidirectional image spans.

    Ranges are inclusive absolute token positions. Tokens outside an image
    keep the normal causal sliding window. A token inside an image sees the
    union of that causal window and its complete image span. The fixed output
    width is ``window_size + max_image_tokens`` so it is graph- and
    operator-workspace friendly.
    """
    if max_image_tokens <= 0:
        raise ValueError("max_image_tokens must be positive for vision SWA")

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(
            query_lens.shape[0],
            device=query_start_loc.device,
            dtype=torch.long,
        ),
        query_lens,
        output_size=num_tokens,
    )
    token_offsets = torch.arange(num_tokens, device=query_start_loc.device) - query_start_loc[req_ids]
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets
    start_positions = (positions - int(window_size) + 1).clamp_min(0)
    end_positions = positions.clone()

    for req_idx, ranges in mm_prefix_ranges.items():
        if req_idx >= query_lens.shape[0]:
            continue
        for span_start, span_end in ranges:
            if span_end < span_start:
                raise ValueError(f"Invalid image span [{span_start}, {span_end}]")
            if span_end - span_start + 1 > max_image_tokens:
                raise ValueError(
                    f"Image span exceeds vision_max_n_token: span=[{span_start}, {span_end}], max={max_image_tokens}"
                )
            in_span = (req_ids == req_idx) & (positions >= span_start) & (positions <= span_end)
            start_positions = torch.where(
                in_span,
                torch.minimum(
                    start_positions,
                    start_positions.new_full((), span_start),
                ),
                start_positions,
            )
            end_positions = torch.where(
                in_span,
                torch.maximum(
                    end_positions,
                    end_positions.new_full((), span_end),
                ),
                end_positions,
            )

    visible_lens = end_positions - start_positions + 1
    index_width = int(window_size) + int(max_image_tokens)
    columns = torch.arange(index_width, device=block_table.device)
    visible = columns.unsqueeze(0) < visible_lens.unsqueeze(1)
    visible_positions = start_positions.unsqueeze(1) + columns.unsqueeze(0)
    block_numbers = visible_positions // int(block_size)
    safe_block_numbers = block_numbers.clamp(
        min=0,
        max=block_table.shape[1] - 1,
    )
    request_block_tables = block_table[req_ids]
    block_ids = torch.gather(
        request_block_tables,
        1,
        safe_block_numbers,
    )
    slot_ids = (block_ids * int(block_size) + visible_positions % int(block_size)).to(torch.int32)
    slot_ids = slot_ids.where(visible, torch.full_like(slot_ids, -1))
    return slot_ids.unsqueeze(1), visible_lens.to(torch.int32)


def _add_ori_topk_length(attn_kwargs: dict, req_metadata) -> None:
    """Send the valid-entry count alongside an explicit index list.

    Without it the operator declares the whole row valid and gathers at the
    -1 padding. The builders here already produce the count for both index
    kinds, so this is carrying a number that exists rather than deriving one.

    Shared with V4.0, whose tables come from the same builders with the same
    -1 padding. Left absent when the metadata carries none.
    """
    length = getattr(req_metadata, "ori_topk_length", None)
    if length is not None:
        attn_kwargs["ori_topk_length"] = length


class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    _request_capacity_factor: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.metadata_cls = metadata_cls if metadata_cls is not None else AscendDSAMetadata
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.logical_block_size = kv_cache_spec.block_size
        self.storage_block_size = kv_cache_spec.storage_block_size
        scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        self.spec_slot_mapping = None
        self.dspark_swa_indices_buffer: torch.Tensor | None = None
        self.dspark_swa_topk_length_buffer: torch.Tensor | None = None
        if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION) and not is_a5_bf16_kv_enabled(
            vllm_config
        ):
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens,)  # type: ignore
        else:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)  # type: ignore
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.spec_sas_metadata = [
                torch.zeros(
                    DSA_METADATA_BUFFER_SIZE,
                    dtype=torch.int32,
                    device=self.device,
                )
                for _ in range(spec_token_num)
            ]
            # Shared static buffer for dspark_swa_indices, so its address
            # stays stable across async ACL-graph replays.
            _dspark_index_width = _aligned_dspark_index_width(
                self.model_config.hf_config.sliding_window, spec_token_num
            )
            max_dspark_rows = max(
                scheduler_config.max_num_batched_tokens,
                scheduler_config.max_num_seqs * (self.speculative_config.num_speculative_tokens + 1),
            )
            self.dspark_swa_indices_buffer = torch.zeros(
                (max_dspark_rows, 1, _dspark_index_width),
                dtype=torch.int32,
                device=self.device,
            )
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.common_ratio_to_sas_metadata: dict | None = None
        self.seq_lens: torch.Tensor = None

        # The layer's configured stream ratio, when a spec states one. It is
        # consulted first because it is the only field here that can hold 0:
        # both fallbacks below are page divisors, which clamp 0 to 1 and make
        # a pure-sliding-window layer look like one with a compressed stream.
        # The fallback chain is V4.0's, where the page divisor and the stream
        # ratio coincide.
        #
        # ``compressor_ratio`` also feeds ``tokens_per_state``, which
        # ``AscendDSAMetadata.block_size`` multiplies the storage page by, so
        # changing what it reads changes the page size too. Reviving
        # VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO means splitting it into two names.
        #
        # vLLM #51718 renamed ``compress_ratio`` to ``tokens_per_state``.
        stated = getattr(kv_cache_spec, "stream_compress_ratio", None)
        if stated is not None:
            self.compressor_ratio = stated
        else:
            self.compressor_ratio = getattr(
                kv_cache_spec,
                "compress_ratio",
                getattr(kv_cache_spec, "tokens_per_state", 0),
            )
        if not layer_names:
            raise ValueError("DSV4 compressor metadata builder requires at least one layer name")
        # vLLM assigns the builder result to every layer in an attention group.
        self.cache_group_key = layer_names[0]
        self.hadamard = None
        self._init_hadamard(layer_names)
        max_num_reqs = scheduler_config.max_num_seqs * self._request_capacity_factor
        self.start_pos_prefill: torch.Tensor = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        self.sas_metadata_buffer: torch.Tensor = torch.zeros(
            DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=self.device
        )
        self.qli_metadata_buffer: torch.Tensor = torch.zeros(
            DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=self.device
        )
        # QLI v2 PA_BBND reads the compressed K length plus the residual from
        # the original length. Persistent buffers keep their addresses stable
        # during graph replay. Full-decode graphs pad the request count beyond
        # max_num_seqs (cudagraph capture sizes plus the FIA dummy request), so
        # size the per-request buffers for the graph-mode maximum.
        max_qli_reqs = max_num_reqs
        compilation_config = self.vllm_config.compilation_config
        if compilation_config.cudagraph_mode != CUDAGraphMode.NONE and compilation_config.cudagraph_capture_sizes:
            max_qli_reqs = max(max_qli_reqs, compilation_config.max_cudagraph_capture_size)
        # +1 holds the FIA dummy request inserted by mixed-batch padding.
        self.qli_seqused_k: torch.Tensor = torch.zeros(max_qli_reqs + 1, dtype=torch.int32, device=self.device)
        self.qli_cmp_residual_k: torch.Tensor = torch.zeros(max_qli_reqs + 1, dtype=torch.int32, device=self.device)
        # V4.1's seqused_cmp_kv, persistent for the same reason and sized by
        # the same formula as the qli_* buffers.
        self.seqused_cmp_kv_buffer: torch.Tensor = torch.zeros(max_qli_reqs + 1, dtype=torch.int32, device=self.device)
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self._zero_i32 = torch.tensor([0], device=self.device, dtype=torch.int32)
        # Note(qcs): we use two dimension slot_mapping for kvcache with shape
        # [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
        self.compressor_metadata_buffers: CompressorMetadataOutput | None = None

    def _init_hadamard(self, layer_names: list[str]) -> None:
        hf_config = self.model_config.hf_config
        if not is_deepseek_v4_family(hf_config):
            return

        indexer_head_dim = hf_config.index_head_dim
        try:
            from scipy.linalg import hadamard  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("Please install scipy") from e
        log_dim = math.ceil(math.log2(indexer_head_dim))
        dim_padded = 2**log_dim
        self.hadamard = get_or_register_attention_buffer(
            self.vllm_config,
            layer_names,
            "_dsa_hadamard",
            lambda: torch.tensor(hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device).to(
                torch.bfloat16
            ),
        )

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens <= self.decode_threshold:
                decodes.append(i)
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill], decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        return modified_batch

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    def _num_compressor_metadata_rows(self, num_reqs: int) -> int:
        assert self.num_actual_tokens is not None
        num_tokens = self.num_actual_tokens
        return min(num_tokens, num_tokens // self.compressor_ratio + num_reqs)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendDSAMetadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_reqs = kwargs.get("num_actual_reqs")
        self.common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata")
        assert self.common_ratio_to_sas_metadata is not None
        self.set_num_actual_tokens(common_attn_metadata)
        num_input_tokens = common_attn_metadata.num_input_tokens

        if self.common_ratio_to_sas_metadata.get("num_decodes") is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.decode_threshold,
                )
            )
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            assert self.num_decodes + self.num_prefills == num_reqs
            assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens
            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            self.common_ratio_to_sas_metadata["seq_lens_cpu"] = seq_lens_cpu
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            cos, sin = get_cos_and_sin_dsa(
                input_positions,
                use_cache=self.num_prefills == 0,
            )
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                self.common_ratio_to_sas_metadata["num_decodes"],
                self.common_ratio_to_sas_metadata["num_prefills"],
                self.common_ratio_to_sas_metadata["num_decode_tokens"],
                self.common_ratio_to_sas_metadata["num_prefill_tokens"],
            )
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            seq_lens_cpu = self.common_ratio_to_sas_metadata["seq_lens_cpu"]
            cos = self.common_ratio_to_sas_metadata["cos"]
            sin = self.common_ratio_to_sas_metadata["sin"]

        # CommonAttentionMetadata uses logical raw-token slots. They directly
        # describe only uncompressed SWA/state caches; C4/C128 physical slots
        # are generated later from the logical block table by compressor_metadata.
        if self.compressor_ratio <= 1:
            slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
            # out= 写进常驻 buffer：省掉这里原本那次 copy_，也是 FULL 图要求的
            # —— 图里烘死的是这块 buffer 的地址，必须原地重填（见 48d3202a6）。
            get_dsa_attn_kv_plan(self.vllm_config).format_dsa_slot_mapping(
                slot_mapping,
                self.storage_block_size,
                out=self.slot_mapping[:num_input_tokens],
            )

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        req_metadata = self.build_req_metadata(
            common_attn_metadata=common_attn_metadata,
            seq_lens_cpu=seq_lens_cpu,
            num_actual_reqs=num_actual_reqs,
            cos=cos,
            sin=sin,
            full_graph_mode=kwargs.get("full_graph_mode", False),
        )

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            hadamard=self.hadamard,
            max_seq_len=common_attn_metadata.max_seq_len,
            tokens_per_state=max(self.compressor_ratio, 1),
        )

    def _build_sas_metadata(
        self,
        metadata_cache: dict,
        layer_name: str,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seqlen_q: int | torch.Tensor,
        max_seqlen_kv: int | torch.Tensor,
        cu_seqlens_ori_kv: torch.Tensor | None,
        cu_seqlens_cmp_kv: torch.Tensor | None,
        ori_mask_mode: int | None = None,
        ori_win_left: int | None = None,
        ori_win_right: int | None = None,
        has_cmp_kv: bool | None = None,
        cmp_topk: int | None = None,
        seqused_cmp_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the descriptor the sparse-attention operator reads its tiling
        from.

        The four mask/window parameters default to the values V4.0 hard-coded
        here, so every existing call site produces a byte-identical descriptor.
        They are parameters because the descriptor and the operator call must
        agree item by item: the kernel derives its tiling from this blob and
        then receives the real arguments, and a disagreement is not rejected --
        it surfaced as an aicore exception with tilingKey 0 and every core
        failing at once.

        V4.1 passes the values its call site derives: a layer whose original
        stream is selected by an explicit index list is in the operator's
        "ori-only sparse mode" (mask mode 0, window bounds -1); every other
        layer is sliding-window mode.
        """
        # Keyed on the compression ratio alone, which assumes cache groups
        # with the same ratio can share one descriptor. The descriptor encodes
        # a per-core split of the query tokens (query_start_loc, max_seqlen_q,
        # head count), which is identical across groups within one build. The
        # dict is rebuilt every _build_attention_metadata.
        sas_metadata = metadata_cache.get(layer_name)
        if sas_metadata is None:
            tp_size = get_tensor_model_parallel_world_size()
            n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
            index_topk = self.model_config.hf_config.index_topk
            validate_compress_ratio(self.compressor_ratio, layer_name,
                                    get_dsa_attn_kv_plan(self.vllm_config))
            cmp_ratio = (
                _dsa_swa_only_cmp_ratio(self.compressor_ratio, self.vllm_config)
                if self.compressor_ratio <= 1 or self.compressor_ratio == 2
                else 4
                if self.compressor_ratio == 4
                else 128
            )
            kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
            metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
            metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
            sas_metadata = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(seq_lens),
                cmp_topk=((index_topk if self.compressor_ratio == 4 else 0)
                          if cmp_topk is None else cmp_topk),
                cmp_ratio=cmp_ratio,
                # 4: sliding window. None means "what V4.0 passes", so the
                # descriptor is unchanged for every caller that does not care.
                ori_mask_mode=4 if ori_mask_mode is None else ori_mask_mode,
                cmp_mask_mode=3,  # 3:causal
                ori_win_left=(self.model_config.hf_config.sliding_window - 1
                              if ori_win_left is None else ori_win_left),
                ori_win_right=0 if ori_win_right is None else ori_win_right,
                layout_q="TND",
                layout_kv=_dsa_layout_kv(self.vllm_config),
                has_ori_kv=True,
                has_cmp_kv=(self.compressor_ratio > 1
                            if has_cmp_kv is None else has_cmp_kv),
                # Omitted entirely when None, so V4.0's descriptor keeps the
                # argument list it has today. The operator demands it under
                # PA_BBND once a compressed stream is declared and the mask
                # mode is not 0 -- V4.0's FP8 plan uses PA_ND and never meets
                # that condition.
                **({} if seqused_cmp_kv is None
                   else {"seqused_cmp_kv": seqused_cmp_kv}),
            )
            metadata_cache[layer_name] = sas_metadata

        self.sas_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = sas_metadata
        return self.sas_metadata_buffer

    def _build_qli_metadata(
        self,
        metadata_cache: dict,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
    ) -> torch.Tensor:
        qli_metadata = metadata_cache.get("qli")
        if qli_metadata is None:
            # QLI v2 PA_BBND reads the compressed K length plus the residual
            # from the original length. Write both into persistent builder
            # buffers so their addresses remain stable during graph replay.
            seq_lens_i32 = seq_lens
            if seq_lens_i32.dtype != torch.int32:
                seq_lens_i32 = seq_lens_i32.to(torch.int32)
            num_reqs = seq_lens_i32.shape[0]
            qli_seqused_k = self.qli_seqused_k[:num_reqs]
            qli_cmp_residual_k = self.qli_cmp_residual_k[:num_reqs]
            # DeepSeek-V4 compresses by 4 throughout, but V4.1 varies the ratio per layer (2 on layers
            # 2..19). Everything below has to agree on one value: the compressed length, the residual
            # the operator checks against cmp_ratio, max_seqlen_k, and cmp_ratio itself. A ratio of 0
            # or 1 means no compression, for which 1 is the identity.
            cmp_ratio = max(self.compressor_ratio, 1)
            torch.div(seq_lens_i32, cmp_ratio, rounding_mode="floor", out=qli_seqused_k)
            torch.remainder(
                seq_lens_i32,
                cmp_ratio,
                out=qli_cmp_residual_k,
            )
            qli_metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
                num_heads_q=self.model_config.hf_config.index_n_heads,
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                topk=self.model_config.hf_config.index_topk,
                # Must match the quant_mode select_topk passes; see
                # qli_v2_quant_mode. No tensors here, so derive both dtypes.
                quant_mode=qli_v2_quant_mode(dsv4_indexer_qk_dtype(), dsv4_indexer_scale_dtype()),
                cu_seqlens_q=query_start_loc,
                seqused_k=qli_seqused_k,
                cmp_residual_k=qli_cmp_residual_k,
                batch_size=len(seq_lens),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv // cmp_ratio,
                layout_q="TND",
                layout_k="PA_BBND",
                mask_mode=3,
                cmp_ratio=cmp_ratio,
                device=str(self.seqused_q.device),
            )
            metadata_cache["qli"] = qli_metadata

        self.qli_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = qli_metadata
        return self.qli_metadata_buffer

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        if self.compressor_ratio > 1 and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            output_shape = (max_tokens, 1, 1, self.model_config.hf_config.qk_rope_head_dim)
            self.compressor_metadata_buffers = (
                torch.empty(output_shape, dtype=torch.float32, device=self.device),
                torch.empty(output_shape, dtype=torch.float32, device=self.device),
                self.slot_mapping,
            )

    def enable_dspark_device_metadata(self, max_num_tokens: int) -> None:
        self.enable_device_metadata()
        assert self.speculative_config is not None
        index_width = _aligned_dspark_index_width(
            self.model_config.hf_config.sliding_window,
            self.speculative_config.num_speculative_tokens,
        )
        self.dspark_swa_indices_buffer = torch.empty(
            (max_num_tokens, 1, index_width),
            dtype=torch.int32,
            device=self.device,
        )
        self.dspark_swa_topk_length_buffer = torch.empty(
            max_num_tokens, dtype=torch.int32, device=self.device
        )

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def build_req_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        num_actual_reqs: int | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        full_graph_mode: bool = False,
    ) -> AscendDSAReqMetadata:
        assert self.common_ratio_to_sas_metadata is not None
        metadata_cache = self.common_ratio_to_sas_metadata
        assert self.num_actual_tokens is not None
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
        max_seqlen_kv = torch.max(seq_lens_cpu[:num_reqs]).item()
        has_prefill = self.num_prefills > 0

        self.start_pos_prefill.fill_(0)
        self.start_pos_prefill[:num_reqs] = seq_lens - seq_lens_q
        if num_actual_reqs is None:
            num_actual_reqs = num_reqs
        else:
            num_actual_reqs = min(num_actual_reqs, num_reqs)
            if num_actual_reqs < num_reqs:
                self.start_pos_prefill[num_actual_reqs:num_reqs].fill_(0)
                self.block_table[num_actual_reqs:num_reqs, ...].fill_(0)
        layer_name = f"c{self.compressor_ratio}"
        cu_seqlens_ori_kv = None
        cu_seqlens_cmp_kv = None
        dspark_swa_indices = None
        vision_swa_indices = None
        # Beside the tensors they describe, so a branch that sets one and not
        # the other is a syntax-level mistake rather than a silent None.
        dspark_topk_length = None
        vision_topk_length = None
        ori_win_left, ori_win_right = self.model_config.hf_config.sliding_window - 1, 0
        if not has_prefill and not common_attn_metadata.causal:
            # DSpark non-causal parallel drafting: every draft query attends to
            # the trailing context window plus the whole current draft block.
            # Not gated on the SAS metadata cache: the indices depend on the
            # current step's block table / sequence lengths, so they must be
            # rebuilt whenever a DSpark draft step runs.
            assert self.speculative_config is not None
            dspark_swa_indices, dspark_topk_length = build_dspark_swa_indices(
                self.block_table[: self.num_decodes],
                self.speculative_config.num_speculative_tokens,
                self.model_config.hf_config.sliding_window,
                self.storage_block_size,
                query_start_loc[: self.num_decodes + 1],
                self.seq_lens[: self.num_decodes],
                self.num_decode_tokens,
                buffer=self.dspark_swa_indices_buffer,
            )
            ori_win_left, ori_win_right = get_dspark_sparse_sas_window(self.vllm_config)
        # Text-only requests and lightweight metadata fixtures do not carry
        # multimodal document ranges. Treat those as having no vision spans.
        mm_ranges = getattr(common_attn_metadata, "mm_req_doc_ranges", None)
        max_image_tokens = (
            getattr(
                self.model_config.hf_config,
                "vision_max_n_token",
                0,
            )
            if getattr(
                self.model_config.hf_config,
                "vision_n_layers",
                0,
            )
            > 0
            else 0
        )
        if has_prefill and max_image_tokens > 0 and mm_ranges_have_spans(mm_ranges):
            actual_reqs = num_reqs if num_actual_reqs is None else num_actual_reqs
            vision_swa_indices, vision_topk_length = \
                build_vision_bidirectional_swa_indices(
                block_table=self.block_table[:actual_reqs],
                window_size=self.model_config.hf_config.sliding_window,
                max_image_tokens=max_image_tokens,
                block_size=self.storage_block_size,
                query_start_loc=query_start_loc[: actual_reqs + 1],
                seq_lens=seq_lens[:actual_reqs],
                mm_prefix_ranges=mm_ranges,
                num_tokens=self.num_actual_tokens,
            )
        if not has_prefill and self.common_ratio_to_sas_metadata.get(layer_name) is None:
            cu_seqlens_ori_kv = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                self.common_ratio_to_sas_metadata,
                "cu_seqlens_ori_kv",
                seq_lens,
                num_reqs,
                self._zero_i32,
                self.cu_seqlens_ori_kv,
            )
            cu_seqlens_cmp_kv = DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        elif has_prefill:
            cu_seqlens_ori_kv = query_start_loc

        # The descriptor and the operator call must agree item by item: the
        # kernel takes its tiling from the descriptor and then receives the
        # real arguments, and a disagreement is not rejected -- it appeared as
        # an aicore exception with tilingKey 0 and every core failing at once.
        #
        # An empty dict leaves V4.0's hard-coded values in place. V4.1 computes
        # them with the same functions its call site uses, so the two cannot
        # drift.
        #
        # Asked by identity (is_deepseek_v41_model), not by reading
        # ``hf_text_config.model_type``: that field is exactly what upstream's
        # dspark branch overwrites with "deepseek_v4" on the draft instance
        # (vllm/config/speculative.py). This was the builder's ONLY version
        # discriminator, so a rewritten config would have silently taken the
        # V4.0 descriptor branch for a V4.1 model -- an empty sas_mask_kwargs
        # means V4.0's hard-coded mask mode and has_cmp_kv=False, which is the
        # descriptor/call disagreement this comment warns about.
        sas_mask_kwargs: dict = {}
        seqused_cmp_kv = None
        if is_deepseek_v41_model(self.vllm_config):
            from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import (
                is_plain_sliding_window,
                ori_mask_mode_for,
                select_ori_sparse_indices,
            )

            attends_cmp = attends_compressed_stream(self.compressor_ratio,
                                                    self.vllm_config)
            ori_indices = select_ori_sparse_indices(
                attends_cmp, vision_swa_indices, dspark_swa_indices)
            plain_window = is_plain_sliding_window(ori_indices)
            sas_mask_kwargs = {
                "ori_mask_mode": ori_mask_mode_for(ori_indices),
                "ori_win_left": ori_win_left if plain_window else -1,
                "ori_win_right": ori_win_right if plain_window else -1,
                "has_cmp_kv": attends_cmp,
            }
            if attends_cmp:
                # cmp_topk and the indices the call passes are one decision:
                # a descriptor promising a top-k that the call does not supply
                # is the same disagreement class as the mask mode, and the
                # operator requires the compressed lengths alongside them.
                ratio = max(self.compressor_ratio, 1)
                lens = seq_lens if seq_lens.dtype == torch.int32 else \
                    seq_lens.to(torch.int32)
                sas_mask_kwargs["cmp_topk"] = (
                    self.model_config.hf_config.index_topk)
                # Computed once and published on the req metadata below, so
                # the descriptor and the call cannot hold different numbers.
                # Written into a persistent buffer and published as a view
                # starting at dim 0: FULL graphs bake in the tensor address,
                # so a fresh tensor per build would leave replays reading the
                # capture-time lengths.
                cmp_len = torch.div(lens, ratio, rounding_mode="floor")
                n = int(cmp_len.shape[0])
                buf = self.seqused_cmp_kv_buffer
                if n > buf.shape[0]:
                    # A fresh-tensor fallback would silently reintroduce the
                    # stale-length defect under FULL graphs.
                    raise RuntimeError(
                        f"DSV4.1 seqused_cmp_kv buffer too small: need {n}, "
                        f"capacity {int(buf.shape[0])}. It must cover the "
                        "request count after FULL decode graph padding (same "
                        "sizing as the qli_* buffers).")
                buf[:n].copy_(cmp_len.to(torch.int32))
                seqused_cmp_kv = buf[:n]
                sas_mask_kwargs["seqused_cmp_kv"] = seqused_cmp_kv

        if self._device_metadata_enabled:

            def build_sas_metadata() -> None:
                self._build_sas_metadata(
                    metadata_cache=metadata_cache,
                    layer_name=layer_name,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                    **sas_mask_kwargs,
                )

            def build_qli_metadata() -> None:
                self._build_qli_metadata(
                    metadata_cache=metadata_cache,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                )

            if self.compressor_ratio == 4:
                self._device_metadata_tasks = (
                    DeviceMetadataTask(DeviceMetadataStage.INDEXER, build_qli_metadata, id(self.qli_metadata_buffer)),
                    DeviceMetadataTask(DeviceMetadataStage.ATTENTION, build_sas_metadata, id(self.sas_metadata_buffer)),
                )
            else:
                self._device_metadata_tasks = (
                    DeviceMetadataTask(DeviceMetadataStage.ATTENTION, build_sas_metadata, id(self.sas_metadata_buffer)),
                )
            sas_metadata = self.sas_metadata_buffer
            qli_metadata = self.qli_metadata_buffer if self.compressor_ratio == 4 else None
        else:
            self._device_metadata_tasks = ()
            sas_metadata = self._build_sas_metadata(
                metadata_cache=metadata_cache,
                layer_name=layer_name,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                **sas_mask_kwargs,
            )
            # Same gate as the device-metadata branch above, which registers
            # the INDEXER task only at ratio 4. The two branches publish the
            # same field and it has one reader: V4.0's indexer
            # (models/deepseek_v4/indexer.py:222,239). That reader also takes
            # qli_cu_seqlens_q / qli_seqused_k / qli_cmp_residual_k, which are
            # published at ratio 4 only (below, after the branches join), so
            # ratio 4 is already the condition under which the descriptor is
            # read -- it was simply missing here.
            #
            # V4.1 never reads the field. It builds its own descriptor in
            # models/deepseek_v4_1/sparse_indexer.py:_qli_metadata, and its
            # ratios are 0, 1 and 2 only (vllm/models/deepseek_v4_1/
            # attention.py raises on anything else), so without this gate every
            # V4.1 build on this branch launched
            # npu_quant_lightning_indexer_v2_metadata on the device for a
            # descriptor that was then dropped.
            qli_metadata = (
                self._build_qli_metadata(
                    metadata_cache=metadata_cache,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                )
                if self.compressor_ratio == 4
                else None
            )

        full_compress_cos, full_compress_sin = None, None
        if self.compressor_ratio > 1:
            if full_graph_mode:
                num_tokens = common_attn_metadata.num_input_tokens
                num_compressed_tokens = min(num_tokens, num_tokens // self.compressor_ratio + num_reqs)
                num_actual_reqs = num_reqs
            else:
                num_compressed_tokens = self._num_compressor_metadata_rows(num_reqs)
            full_compress_cos, full_compress_sin = get_full_cos_and_sin_dsa(layer_name)
            slot_mapping = None
        else:
            num_compressed_tokens = self.num_actual_tokens
            slot_mapping = self.slot_mapping[: self.num_actual_tokens]

        # Published whatever the ratio. The variable above is None for
        # compressing groups on purpose -- their compressed stream is scattered
        # by its own operator -- but the per-token mapping is computed either
        # way, and V4.1's indexer K writer needs it at every ratio.
        token_slot_mapping = self.slot_mapping[: self.num_actual_tokens]

        qli_cu_seqlens_q = None
        qli_seqused_k = None
        qli_cmp_residual_k = None
        if self.compressor_ratio == 4:
            # QLI v2 PA_BBND reads the compressed K length plus the residual
            # from the original length; the persistent builder buffers are
            # refreshed in _build_qli_metadata.
            qli_cu_seqlens_q = query_start_loc
            qli_seqused_k = self.qli_seqused_k[:num_reqs]
            qli_cmp_residual_k = self.qli_cmp_residual_k[:num_reqs]

        req_metadata = AscendDSAReqMetadata(
            block_table=self.block_table[:num_reqs, ...],
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            token_slot_mapping=token_slot_mapping,
            storage_block_size=self.storage_block_size,
            query_start_loc=query_start_loc,
            num_compressed_tokens=num_compressed_tokens,
            sin=sin,
            cos=cos,
            full_compress_sin=full_compress_sin,
            full_compress_cos=full_compress_cos,
            start_pos=self.start_pos_prefill[:num_reqs],
            num_actual_reqs=num_actual_reqs,
            cache_group_key=self.cache_group_key,
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            qli_cu_seqlens_q=qli_cu_seqlens_q,
            qli_seqused_k=qli_seqused_k,
            qli_cmp_residual_k=qli_cmp_residual_k,
            attn_mask=None,
            cu_cmp_seqlen_list=cu_seqlens_cmp_kv,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa_indices=dspark_swa_indices,
            vision_swa_indices=vision_swa_indices,
            ori_topk_length=(vision_topk_length if vision_swa_indices
                             is not None else dspark_topk_length),
            seqused_cmp_kv=seqused_cmp_kv,
        )
        if self._device_metadata_enabled and self.compressor_metadata_buffers is not None:
            assert num_compressed_tokens is not None
            buffers = self.compressor_metadata_buffers
            outputs = (
                buffers[0][:num_compressed_tokens],
                buffers[1][:num_compressed_tokens],
                buffers[2][:num_compressed_tokens],
            )
            group_id = id(buffers[0])
            req_metadata.compressor_metadata = outputs
            req_metadata.compressor_metadata_group_id = group_id
            self._device_metadata_tasks = (
                DeviceMetadataTask(
                    DeviceMetadataStage.COMPRESSOR,
                    lambda: build_compressor_metadata_out(
                        req_metadata, self.compressor_ratio, outputs, self.vllm_config
                    ),
                    group_id,
                ),
                *self._device_metadata_tasks,
            )
        return req_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
            )
        )
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]

        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        if self.num_prefills:
            cos, sin = get_cos_and_sin_dsa(input_positions)
        else:
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True, draft_index=draft_index)
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        assert self.spec_slot_mapping is not None
        self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = get_dsa_attn_kv_plan(
            self.vllm_config
        ).format_dsa_slot_mapping(slot_mapping, self.storage_block_size)
        req_metadata = self.build_req_metadata_for_drafting(
            draft_index=draft_index,
            common_attn_metadata=common_attn_metadata,
            cos=cos,
            sin=sin,
        )

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            hadamard=None,
            max_seq_len=common_attn_metadata.max_seq_len,
            tokens_per_state=max(self.compressor_ratio, 1),
        )

    def build_req_metadata_for_drafting(
        self,
        draft_index: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        cos,
        sin,
    ) -> AscendDSAReqMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seqlen_kv = torch.max(seq_lens_cpu[:num_reqs]).item()
        has_prefill = self.num_prefills > 0

        dspark_swa_indices = None
        dspark_topk_length = None
        build_dspark_swa = None
        ori_win_left = self.model_config.hf_config.sliding_window - 1
        ori_win_right = 0
        if not common_attn_metadata.causal:
            assert self.speculative_config is not None
            dspark_swa_args = (
                self.block_table[:num_reqs],
                self.speculative_config.num_speculative_tokens,
                self.model_config.hf_config.sliding_window,
                self.storage_block_size,
                query_start_loc,
                seq_lens,
                self.num_actual_tokens,
            )
            if self._device_metadata_enabled and not has_prefill:
                if self.dspark_swa_indices_buffer is None or self.dspark_swa_topk_length_buffer is None:
                    raise RuntimeError(
                        "DSpark device metadata buffers must be initialized before building draft attention metadata"
                    )
                if self.num_actual_tokens > self.dspark_swa_indices_buffer.shape[0]:
                    raise ValueError(
                        "DSpark SWA metadata rows exceed the persistent buffer capacity: "
                        f"active={self.num_actual_tokens}, capacity={self.dspark_swa_indices_buffer.shape[0]}"
                    )
                dspark_swa_indices = self.dspark_swa_indices_buffer[: self.num_actual_tokens]
                dspark_topk_length = self.dspark_swa_topk_length_buffer[: self.num_actual_tokens]
                build_dspark_swa = lambda: build_dspark_swa_indices(
                    *dspark_swa_args,
                    indices_output=dspark_swa_indices,
                    lengths_output=dspark_topk_length,
                )
            else:
                dspark_swa_indices, dspark_topk_length = build_dspark_swa_indices(
                    *dspark_swa_args)
                dspark_swa_indices = dspark_swa_indices[: self.num_actual_tokens]
            ori_win_left, ori_win_right = get_dspark_sparse_sas_window(self.vllm_config)

        cu_seqlens_ori_kv = (
            query_start_loc
            if has_prefill
            else DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                None,
                "draft_cu_seqlens_ori_kv",
                seq_lens,
                num_reqs,
                self._zero_i32,
                self.cu_seqlens_ori_kv,
            )
        )
        cu_seqlens_cmp_kv = (
            None if has_prefill else DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        )
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        draft_mask_kwargs = dict(
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
        )
        if is_deepseek_v41_model(self.vllm_config):
            from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import ori_mask_mode_for

            draft_mask_kwargs["ori_mask_mode"] = ori_mask_mode_for(dspark_swa_indices)
            draft_mask_kwargs["cmp_mask_mode"] = 0
            if dspark_swa_indices is not None:
                draft_mask_kwargs.update(ori_win_left=-1, ori_win_right=-1)

        def build_attention_metadata() -> torch.Tensor:
            if build_dspark_swa is not None:
                build_dspark_swa()
            result = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=num_reqs,
                cmp_ratio=1,
                **draft_mask_kwargs,
                layout_q="TND",
                layout_kv=_dsa_layout_kv(self.vllm_config),
                has_ori_kv=True,
                has_cmp_kv=False,
            )
            if not has_prefill:
                assert self.spec_sas_metadata is not None
                self.spec_sas_metadata[draft_index - 1][:DSA_METADATA_BUFFER_SIZE].copy_(
                    result[:DSA_METADATA_BUFFER_SIZE]
                )
                return self.spec_sas_metadata[draft_index - 1]
            return result

        if build_dspark_swa is not None:
            sas_metadata = self.spec_sas_metadata[draft_index - 1]

            def run_attention_metadata() -> None:
                build_attention_metadata()

            self._device_metadata_tasks = (
                DeviceMetadataTask(DeviceMetadataStage.ATTENTION, run_attention_metadata, id(sas_metadata)),
            )
        else:
            self._device_metadata_tasks = ()
            sas_metadata = build_attention_metadata()

        assert self.spec_slot_mapping is not None
        slot_mapping = self.spec_slot_mapping[draft_index - 1][: self.num_actual_tokens]
        return AscendDSAReqMetadata(
            block_table=self.block_table[:num_reqs, ...],
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            # Never None on this path, so the two are the same tensor.
            token_slot_mapping=slot_mapping,
            storage_block_size=self.storage_block_size,
            query_start_loc=query_start_loc,
            num_compressed_tokens=self.num_actual_tokens,
            sin=sin,
            cos=cos,
            start_pos=self.seq_lens[:num_reqs] - seq_lens_q,
            num_actual_reqs=num_reqs,
            cache_group_key=self.cache_group_key,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            attn_mask=None,
            cu_cmp_seqlen_list=cu_seqlens_cmp_kv,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa_indices=dspark_swa_indices,
            ori_topk_length=dspark_topk_length,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        assert attn_metadata is not None
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendDSAImpl(AttentionImplBase[Any]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.vllm_config = kwargs["vllm_config"]
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5
        self.support_fp8_attention = get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)

        # MLA Args
        self.wq_a = kwargs["wq_a"]
        self.wq_b = kwargs["wq_b"]
        self.wkv = kwargs["wkv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs["q_norm_without_weight"]
        # V4.0 applies the second, weightless per-head q RMSNorm; V4.1 must
        # not (see applies_dsa_q_rms). Kept as a settable attribute, the name
        # the official v27 uses, so a model class that knows better can still
        # override it -- but it is *derived from the config* rather than
        # defaulted to True, because the only way a V4.1 checkpoint gets here
        # is through a model class that did not set it.
        self.apply_q_norm = applies_dsa_q_rms(self.vllm_config)
        self.kv_norm = kwargs["kv_norm"]

        # CV wrapper: split wq_a/wkv/wq_b into quantize(Vector) + matmul(Cube)
        self.cv_wq_a = CVLinearWrapper(self.wq_a)
        self.cv_wkv = CVLinearWrapper(self.wkv)
        self.cv_wq_b = CVLinearWrapper(self.wq_b)

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")
        self.swa_cache_layer = kwargs.get("swa_cache_layer")
        assert self.swa_cache_layer is not None

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        ascend_config = get_ascend_config()
        self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap
        if self.multistream_dsv4_dsa_overlap and is_a5_bf16_kv_enabled(self.vllm_config):
            self.multistream_dsv4_dsa_overlap = False

    def _get_layer_metadata(
        self,
        attn_layer_name: str,
        attn_metadata: DSAMetadataDict,
    ) -> AscendDSALayerMetadata:
        assert self.swa_cache_layer is not None
        swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
        attention_metadata = None
        compressor_metadata = None
        indexer_metadata = None

        if self.compress_ratio > 1:
            assert self.compressor is not None
            attention_metadata = attn_metadata[attn_layer_name]
            compressor_metadata = AscendCompressorMetadata(
                cache=attention_metadata,
                state=attn_metadata[self.compressor.state_cache.prefix],
            )
            if self.compress_ratio == 4:
                assert self.indexer is not None
                assert self.indexer.compressor is not None
                indexer_cache_metadata = attn_metadata[self.indexer.k_cache.prefix]
                indexer_metadata = AscendIndexerMetadata(
                    compressor=AscendCompressorMetadata(
                        cache=indexer_cache_metadata,
                        state=attn_metadata[self.indexer.compressor.state_cache.prefix],
                    ),
                )

        return AscendDSALayerMetadata(
            attention=attention_metadata,
            swa=swa_metadata,
            compressor=compressor_metadata,
            indexer=indexer_metadata,
        )

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        # dsa does not need to update graph params
        pass

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # Attention impls are not walked by vllm's process_weights_after_loading
        # dispatcher (only LinearMethodBase subclasses are). OTP buffers are
        # allocated lazily on the first _forward_o_proj call, which always runs
        # before ACL graph capture (profiling run triggers it).
        pass

    def _forward_o_proj(self, o_proj_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        num_tokens = o_proj_input.shape[0]
        group_hidden_dim = o_proj_input.shape[1] * o_proj_input.shape[2] // self.n_local_groups
        o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, group_hidden_dim)
        # A5 (Ascend950) uses an FP8-quantized o_proj path (dynamic MX quant
        # + quantized batch matmul). Preserve it as-is: it predates and is
        # orthogonal to the OTP path below, so it must win first.
        use_a5_quant_o_proj = self.support_fp8_attention and _has_weight_scale(self.wo_a)
        if use_a5_quant_o_proj:
            o = o_proj_input
            o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
            o = torch_npu.npu_transpose_quant_batchmatmul(
                o,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=(0, 0, 32),
                x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
            o = o.reshape(num_tokens, -1)
            output[...] = self.wo_b(o)
        elif oproj_tp_enable():
            oproj_group = get_otp_group()
            oproj_tp_size = oproj_group.world_size
            if self.n_local_groups % oproj_tp_size != 0:
                raise ValueError(
                    "n_local_groups must be divisible by "
                    f"oproj_tensor_parallel_size, got {self.n_local_groups} "
                    f"and {oproj_tp_size}."
                )

            groups_per_rank = self.n_local_groups // oproj_tp_size

            o_proj_input = o_proj_input.view(num_tokens, oproj_tp_size, groups_per_rank, group_hidden_dim)
            # Pad to a static exchange size so the all_to_all / reduce_scatter
            # shapes are identical across all ACL graph buckets — variable
            # shapes desync the HCCL communicator during graph replay.
            # potential_max_tokens is computed once in the model runner __init__,
            # so reading it here is a cheap global lookup.
            exchange_num_tokens = get_potential_max_tokens()
            if exchange_num_tokens < num_tokens:
                raise ValueError(
                    "oproj static exchange capacity must cover local tokens, "
                    f"got {exchange_num_tokens} and {num_tokens}."
                )
            # Lazily allocate static send/recv buffers on first call. The
            # profiling run hits this path before ACL graph capture, so the
            # buffers exist and keep a stable device address across all later
            # capture/replay cycles (graph replay requires the same address
            # that was recorded at capture; new_zeros per call would desync
            # the HCCL operator).
            if not hasattr(self, "_oproj_send_buf"):
                buf_shape = (oproj_tp_size, exchange_num_tokens, groups_per_rank, group_hidden_dim)
                self._oproj_send_buf = torch.zeros(buf_shape, dtype=o_proj_input.dtype, device=o_proj_input.device)
                self._oproj_recv_buf = torch.empty_like(self._oproj_send_buf)
            send = self._oproj_send_buf
            recv = self._oproj_recv_buf
            # In-place fill into the address-stable buffer: zero the padding
            # tail, then copy the real tokens.
            send.zero_()
            send[:, :num_tokens].copy_(o_proj_input.transpose(1, 0))
            dist.all_to_all_single(recv.view(-1), send.view(-1), group=oproj_group.device_group)
            o_proj_input = recv.view(oproj_tp_size * exchange_num_tokens, groups_per_rank, group_hidden_dim)
            o_proj_input = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1,
            )
            o_proj_input = o_proj_input.reshape(oproj_tp_size * exchange_num_tokens, -1)
            o_proj_output = self.wo_b(o_proj_input)
            # reduce_scatter via a raw dist collective into an address-stable
            # static buffer. oproj_group.reduce_scatter is a list-based wrapper
            # that allocates per call, which desyncs the HCCL operator recorded
            # at capture during ACL graph replay — the same reason all_to_all
            # and the embedding TP path use raw dist + static buffers.
            if not hasattr(self, "_oproj_rs_out_buf"):
                self._oproj_rs_out_buf = torch.empty(
                    (exchange_num_tokens, o_proj_output.shape[-1]),
                    dtype=o_proj_output.dtype,
                    device=o_proj_output.device,
                )
            dist.reduce_scatter_tensor(self._oproj_rs_out_buf, o_proj_output, group=oproj_group.device_group)
            output[...] = self._oproj_rs_out_buf[:num_tokens]
        else:
            # A5 BF16 wo_a is reshaped to [groups, hidden, rank] at load time,
            # matching the A3 layout expected by npu_transpose_batchmatmul.
            o_proj_input = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1,
            )
            o_proj_input = o_proj_input.reshape(num_tokens, -1)
            output[...] = self.wo_b(o_proj_input)
        return output

    def _prepare_caches_before_attention(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataDict,
    ) -> bool:
        """Prepare cache updates and report whether local writes can be skipped."""
        return False

    def _get_o_proj_input_shape(
        self,
        attn_metadata: DSAMetadataDict | None,
    ) -> tuple[int, int, int]:
        return (
            _EXTRA_CTX.num_tokens,
            self.n_local_heads,
            self.head_dim,
        )

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataDict,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        output_padded = output
        o_proj_input_shape = self._get_o_proj_input_shape(attn_metadata)
        if attn_metadata is None:
            # Profiling run: run o_proj on zero input so HCCL collectives are
            # captured by the ACL graph.  Non-OTP just zeros the output.
            if oproj_tp_enable():
                o_proj_input = hidden_states.new_zeros(o_proj_input_shape)
                self._forward_o_proj(o_proj_input, output)
            else:
                output.fill_(0)
            return output
        layer_metadata = self._get_layer_metadata(layer_name, attn_metadata)
        common_attn_metadata = layer_metadata.attention
        if common_attn_metadata is None:
            common_attn_metadata = layer_metadata.swa
        actual_tokens = common_attn_metadata.num_actual_tokens

        o_proj_input = hidden_states.new_zeros(o_proj_input_shape)
        assert kv_cache is not None, "kv_cache tensor tuple must be provided."
        wait_for_kv_layer_from_connector(layer_name)
        cache_is_prepared = self._prepare_caches_before_attention(
            layer_name,
            hidden_states,
            kv_cache,
            attn_metadata,
        )
        if actual_tokens == 0:
            output.zero_()
            notify_kv_cache_written(layer_name)
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
            return output

        req_metadata = _require_req_metadata(common_attn_metadata)
        o_proj_input[:actual_tokens] = self._forward_attention(
            layer_name,
            hidden_states[:actual_tokens],
            kv_cache,
            layer_metadata,
            cache_is_prepared,
        )
        cos = req_metadata.cos[layer_name]
        sin = req_metadata.sin[layer_name]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input[:actual_tokens].unsqueeze(1),
            cos[:actual_tokens],
            -sin[:actual_tokens],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # o
        self._forward_o_proj(o_proj_input, output)
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output_padded

    def _mla_prolog_single_stream(
        self,
        hidden_states,
        cos,
        sin,
        swa_kv_cache,
        slot_mapping,
        write_swa_cache=True,
    ):
        """Run the MLA prolog on the current stream."""
        share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
        if share_hs_quant:
            hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            q_a = torch_npu.npu_quant_matmul(
                hs_int8,
                self.wq_a.weight,
                self.wq_a.weight_scale,
                pertoken_scale=hs_pertoken_scale,
                bias=self.wq_a.bias,
                output_dtype=hidden_states.dtype,
            )
        else:
            q_a = self.wq_a(hidden_states)

        # q
        if _is_w8a8_dynamic(self.wq_b):
            # Guarded, not called directly: the guard hands back the operator it
            # just vetted, so the check and the call cannot drift apart.
            rms_norm_dynamic_quant = _require_rms_norm_dynamic_quant()
            qr, qr_pertoken_scale = rms_norm_dynamic_quant(q_a, self.q_norm.weight, epsilon=self.eps)
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            qr = self.q_norm(q_a)
            q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
            qr_pertoken_scale = None
        if self.apply_q_norm:
            q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # win kv & tok_dis
        if write_swa_cache:
            if share_hs_quant:
                kv = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wkv.weight,
                    self.wkv.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wkv.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # swa exec kv
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                swa_kv_cache,
                kv,
                slot_mapping,
            )

        return q, qr, qr_pertoken_scale

    def _mla_prolog_multistream(
        self,
        hidden_states,
        cos,
        sin,
        swa_kv_cache,
        slot_mapping,
        is_prefill=False,
        tail_overlap_fn: Callable[[], CompressorForwardOutput] | None = None,
    ):
        """3-block multi-stream: 3-stage CV parallel + serial tail

        Block partition (V: Vector, C: Cube, AIV: AI Vector):
          Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
          Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
          Part3: q_b_matmul[C]             ||  kv_norm[V] + rope[V] + scatter[AIV]
          Tail:  q_rms[V] + rope[V]  ||  optional compressor metadata + compressor

        Each stream's data is self-contained; no cross-stream sync is needed between blocks.
        Only the tail wait_stream ensures scatter is complete.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()

        is_w8a8 = _is_w8a8_dynamic(self.wq_b)

        # Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
        # When wq_a and wkv have the same quant_method and the same
        # communication status, their quantize() outputs are equivalent.
        # Share the result instead of calling quantize() twice on the same input.
        # - W8A8 no-comm: saves one npu_dynamic_quant (full-tensor read + absmax).
        # - MXFP8 no-comm: saves one npu_dynamic_mx_quant (full-tensor read +
        #   per-group scale).
        # - W4A8 no-comm: saves one no-op pass-through (kernel launch + ref).
        # - TP comm: both return (hidden_states, None); shareable when custom_op
        #   types match (same communication path).
        share_quant = (
            type(self.cv_wq_a._quant_method) is type(self.cv_wkv._quant_method)
            and self.cv_wq_a._has_communication == self.cv_wkv._has_communication
        )
        e_kv_quant_done = None
        if share_quant:
            q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)
            kv_quant, kv_pertoken_scale = q_quant, q_pertoken_scale
        else:
            q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)
            e_q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_q_quant_done)
                kv_quant, kv_pertoken_scale = self.cv_wkv.quantize(hidden_states)
                e_kv_quant_done = torch.npu.current_stream().record_event()

        wq_a_result = self.cv_wq_a.matmul(q_quant, q_pertoken_scale)

        # Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
        e_part2_start = main_stream.record_event()
        if e_kv_quant_done is not None:
            main_stream.wait_event(e_kv_quant_done)

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part2_start)
            kv = self.cv_wkv.matmul(kv_quant, kv_pertoken_scale)
            e_kv_matmul_done = torch.npu.current_stream().record_event()

        if is_prefill:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = self.cv_wq_b.quantize(qr)
            qr_pertoken_scale = None
        elif is_w8a8:
            # Same guard as the single-stream prolog above; see
            # _require_rms_norm_dynamic_quant for what it can and cannot judge.
            rms_norm_dynamic_quant = _require_rms_norm_dynamic_quant()
            qr, qr_pertoken_scale = rms_norm_dynamic_quant(
                wq_a_result, self.q_norm.weight, epsilon=self.eps
            )
            q_b_quant, q_b_scale = qr, qr_pertoken_scale
        else:
            # MXFP8: the split-out quantize() (Vector) overlaps with kv_matmul
            # (Cube) in Part2, and the pair is returned for the Indexer to
            # reuse. Non-splittable schemes (W4A8, bf16) keep the pass-through
            # (scale stays None).
            q_b_quant, q_b_scale = self.cv_wq_b.quantize(self.q_norm(wq_a_result))
            qr, qr_pertoken_scale = q_b_quant, q_b_scale

        # Part3: q_b_matmul[C]  ||  kv_norm[V] + rope[V] + scatter[AIV]
        e_part3_start = main_stream.record_event()
        # kv_matmul and q_b_matmul are both Cube ops. Ensure kv_matmul (launched on
        # aux_stream) completes before q_b_matmul starts so they do not contend for
        # the Cube units. kv_norm (Vector) follows kv_matmul on aux_stream and is
        # unaffected as it overlaps with q_b_matmul.
        main_stream.wait_event(e_kv_matmul_done)

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part3_start)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)

        if is_prefill:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))
        elif is_w8a8:
            q = torch_npu.npu_quant_matmul(
                q_b_quant,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_b_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))

        # Join the Q and SWA-KV branches, then reuse the auxiliary stream for
        # independent tail work while q_rms[V] + rope[V] run on the main stream.
        main_stream.wait_stream(aux_stream)

        tail_overlap_output: CompressorOverlapOutput | None = None
        if tail_overlap_fn is not None:
            e_tail_start = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_tail_start)
                overlap_result = tail_overlap_fn()
                e_tail_overlap_done = torch.npu.current_stream().record_event()
            tail_overlap_output = overlap_result, e_tail_overlap_done

        if self.apply_q_norm:
            q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        return q, qr, qr_pertoken_scale, tail_overlap_output

    def _maybe_update_compressed_caches_and_select_topk(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata: AscendDSALayerMetadata,
        qr_pertoken_scale: torch.Tensor | None,
        compress_kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
        compressor_overlap_output: CompressorOverlapOutput | None = None,
        write_cache: bool = True,
    ) -> torch.Tensor | None:
        """Update compressed caches and return Indexer top-k indices."""
        compressor = self.compressor
        assert compressor is not None
        assert layer_metadata.compressor is not None

        if self.compress_ratio == 4:
            assert self.indexer is not None
            assert layer_metadata.indexer is not None

            def compute_attention_compressed_kv() -> tuple[torch.Tensor, torch.Tensor]:
                if compressor_overlap_output is not None:
                    overlap_result, compressor_done = compressor_overlap_output
                    torch.npu.current_stream().wait_event(compressor_done)
                    return overlap_result
                return compressor(
                    hidden_states=hidden_states,
                    state_cache=state_cache,
                    metadata=layer_metadata.compressor,
                )

            def scatter_attention_compressed_kv(
                compressed_kv: torch.Tensor,
                compress_slot_mapping: torch.Tensor,
            ) -> None:
                if compressed_kv.shape[0] > 0:
                    get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                        compress_kv_cache,
                        compressed_kv,
                        compress_slot_mapping,
                    )

            overlap_plan = IndexerOverlapPlan(
                compute_attention_compressed_kv=compute_attention_compressed_kv,
                scatter_attention_compressed_kv=scatter_attention_compressed_kv,
                aux_stream=dsv4_dsa_overlap_stream() if self.multistream_dsv4_dsa_overlap else None,
            )
            return self.indexer(
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                metadata=layer_metadata.indexer,
                overlap_plan=overlap_plan,
                layer_name=layer_name,
                qr_pertoken_scale=qr_pertoken_scale,
                write_cache=write_cache,
            )

        if not write_cache:
            return None
        if compressor_overlap_output is not None:
            (compressed_kv, compress_slot_mapping), compressor_done = compressor_overlap_output
            torch.npu.current_stream().wait_event(compressor_done)
        else:
            compressed_kv, compress_slot_mapping = compressor(
                hidden_states=hidden_states,
                state_cache=state_cache,
                metadata=layer_metadata.compressor,
            )
        if compressed_kv.shape[0] > 0:
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                compress_kv_cache,
                compressed_kv,
                compress_slot_mapping,
            )
        return None

    def _forward_attention(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata: AscendDSALayerMetadata,
        cache_is_prepared: bool = False,
    ) -> torch.Tensor:
        # DSA PCP sets cache_is_prepared after global cache updates and forces
        # single-stream attention because there is no local KV update to overlap.
        if cache_is_prepared and self.multistream_dsv4_dsa_overlap:
            raise RuntimeError("Prepared DSA caches require single-stream attention.")

        (
            compress_kv_cache,
            swa_kv_cache,
            state_cache,
            _,
            _,
            _,
        ) = DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)

        common_attn_metadata = layer_metadata.attention
        if common_attn_metadata is None:
            common_attn_metadata = layer_metadata.swa
        swa_metadata = layer_metadata.swa

        common_metadata = _require_req_metadata(common_attn_metadata)
        swa_req_metadata = _require_req_metadata(swa_metadata)
        has_prefill = common_attn_metadata.num_prefills > 0
        num_tokens = hidden_states.shape[0]
        cos = common_metadata.cos[layer_name][:num_tokens]
        sin = common_metadata.sin[layer_name][:num_tokens]
        actual_seq_lengths_query = common_metadata.query_start_loc
        actual_seq_lengths_key = common_metadata.seq_lens
        ori_win_left = self.window_size - 1 if swa_req_metadata.ori_win_left is None else swa_req_metadata.ori_win_left
        ori_win_right = 0 if swa_req_metadata.ori_win_right is None else swa_req_metadata.ori_win_right

        compressor_tail_fn = None
        if self.multistream_dsv4_dsa_overlap and self.compress_ratio > 1:
            compressor = self.compressor
            tail_compressor_metadata = layer_metadata.compressor
            assert compressor is not None
            assert tail_compressor_metadata is not None

            def compressor_tail_fn() -> CompressorForwardOutput:
                return compressor(
                    hidden_states=hidden_states,
                    state_cache=state_cache,
                    metadata=tail_compressor_metadata,
                )

        if self.multistream_dsv4_dsa_overlap:
            q, qr, qr_pertoken_scale, compressor_overlap_output = self._mla_prolog_multistream(
                hidden_states,
                cos,
                sin,
                swa_kv_cache,
                swa_req_metadata.slot_mapping,
                is_prefill=has_prefill,
                tail_overlap_fn=compressor_tail_fn,
            )
        else:
            compressor_overlap_output = None
            q, qr, qr_pertoken_scale = self._mla_prolog_single_stream(
                hidden_states,
                cos,
                sin,
                swa_kv_cache,
                swa_req_metadata.slot_mapping,
                write_swa_cache=not cache_is_prepared,
            )

        compress_topk_idxs = None
        compressor_metadata = None
        if self.compress_ratio > 1:
            compressor_metadata = layer_metadata.compressor
            assert compressor_metadata is not None
            compress_topk_idxs = self._maybe_update_compressed_caches_and_select_topk(
                layer_name=layer_name,
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                layer_metadata=layer_metadata,
                qr_pertoken_scale=qr_pertoken_scale,
                compress_kv_cache=compress_kv_cache,
                state_cache=state_cache,
                compressor_overlap_output=compressor_overlap_output,
                write_cache=not cache_is_prepared,
            )

        notify_kv_cache_written(layer_name)
        wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(common_metadata.sas_metadata))
        record_attention_compute_start()
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        attn_op = kv_plan.get_dsa_sparse_attn_op()
        attn_kwargs = kv_plan.get_dsa_sparse_attn_base_kwargs()
        if has_prefill:
            kv_plan.add_dsa_sparse_attn_extra_kwargs(attn_kwargs, cu_seqlens_ori_kv=actual_seq_lengths_query)
        attends_compressed = attends_compressed_stream(self.compress_ratio, self.vllm_config)
        if attends_compressed:
            kv_plan.add_dsa_sparse_attn_extra_kwargs(attn_kwargs, cu_seqlens_cmp_kv=common_metadata.cu_cmp_seqlen_list)

        attn_kwargs.update(
            ori_kv=swa_kv_cache,
            ori_block_table=swa_req_metadata.block_table,
            cu_seqlens_q=actual_seq_lengths_query,
            seqused_kv=actual_seq_lengths_key,
            sinks=self.attn_sink,
            metadata=common_metadata.sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=_dsa_swa_only_cmp_ratio(self.compress_ratio, self.vllm_config),
            # The operator refuses sliding-window mode alongside a compressed
            # stream ("SWA mode requires cmp_ratio=1"), and a non-windowed mode
            # requires both bounds at -1.
            ori_mask_mode=0 if attends_compressed else 4,
            ori_win_left=-1 if attends_compressed else ori_win_left,
            ori_win_right=-1 if attends_compressed else ori_win_right,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
        )

        # Vision prefill uses explicit original-KV indices so tokens inside an
        # image span can see the complete span bidirectionally. Compressed KV
        # selection remains active and is supplied independently below.
        if swa_req_metadata.vision_swa_indices is not None:
            attn_kwargs["ori_sparse_indices"] = swa_req_metadata.vision_swa_indices
            _add_ori_topk_length(attn_kwargs, swa_req_metadata)

        # DSpark indices apply only to layers without a compressed stream
        # (the same precedence as sparse_attn_args.select_ori_sparse_indices).
        if not attends_compressed:
            if swa_req_metadata.dspark_swa_indices is not None:
                attn_kwargs["ori_sparse_indices"] = swa_req_metadata.dspark_swa_indices
                _add_ori_topk_length(attn_kwargs, swa_req_metadata)
        else:
            assert compressor_metadata is not None
            attn_kwargs.update(
                cmp_kv=compress_kv_cache,
                cmp_block_table=common_metadata.block_table,
                cmp_mask_mode=3,
            )
            if self.compress_ratio == 4:
                assert compress_topk_idxs is not None
                attn_kwargs["cmp_sparse_indices"] = compress_topk_idxs

        return attn_op(q, **attn_kwargs)[0]
