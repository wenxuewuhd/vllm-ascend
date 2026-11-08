#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

import functools
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list, maybe_normalize_mxfp_scale_layout
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg
from vllm_ascend.utils import FP8_METHOD, dispose_tensor

from ..base import (
    AscendLinearScheme,
    AscendMoEScheme,
    QuantType,
    WeightSwitchGatherSpec,
)
from ..registry import register_scheme

# An mx block scale is a raw e8m0 exponent BYTE. Two different contracts can carry that
# byte from the checkpoint into a layer parameter, and each one is a PAIR of halves --
# what the model's load_weights hands to the weight loader, and what dtype the parameter
# declares:
#
#   byte contract   loader passes ``loaded_weight.view(torch.uint8)`` into a uint8 param,
#                   so ``copy_`` moves bytes. The MoE scales use this
#                   (vllm/models/deepseek_v4_1/nvidia/model.py, the ``".experts." in name``
#                   branch is the only place the uint8 view is taken).
#   value contract  loader passes the float8_e8m0fnu tensor into a float32 param, so
#                   ``copy_`` converts the VALUE. Every e8m0 code is an exactly
#                   representable float32 (or NaN), so the byte survives in the fp32
#                   exponent field and comes back out with ``>> 23 & 0xFF``. The DS linear
#                   scales below use this, because their names contain no ``.experts.``.
#
# Crossing the halves is silent: both scale dtypes are one byte wide, every shape still
# matches, and nothing downstream complains -- it just turns a byte move into arithmetic
# (or the reverse) and destroys every scale. So the decode below is exhaustive over
# dtypes and RAISES on anything it was not written for, instead of sending the rest
# through the float32 bit extraction, and `assert_e8m0_scale_load_preserves_bytes` proves
# a (loader half, declared dtype) pair by running the copy and comparing BYTES.
E8M0_CODE_COUNT = 256
FP32_EXPONENT_SHIFT = 23
FP32_EXPONENT_MASK = 0xFF
FP32_MANTISSA_MASK = 0x7FFFFF


def _scale_dtype_provenance(dtype: torch.dtype, declared_dtype: torch.dtype | None, prefix: str | None) -> str:
    """Say WHERE this dtype came from, so the reader knows whose problem it is.

    "Unsupported dtype" sends the reader nowhere: the same message fits a bad checkpoint
    and a changed declaration in this file, and those are opposite fixes.
    """
    where = f" on {prefix}" if prefix else ""
    if declared_dtype is None:
        return (
            f"The caller passed no declared dtype, so the origin of the {dtype} tensor{where} cannot "
            "be attributed. Pass declared_dtype=self.scale_param_dtype to find out."
        )
    if dtype == declared_dtype:
        return (
            f"The parameter{where} has exactly the dtype this class declares: "
            f"scale_param_dtype={declared_dtype}, set in "
            f"{AscendW8A8MXFP8DSDynamicLinearMethod.__name__} and allocated by get_pergroup_param in "
            "this file. The checkpoint is not involved -- change the declaration here, and move "
            "loader_gives_scale_byte_view with it."
        )
    return (
        f"The parameter{where} was allocated as scale_param_dtype={declared_dtype}, and something "
        f"REPLACED its .data with a {dtype} tensor after allocation. This class did not do that. The "
        "steps that can: the checkpoint tensor handed to the weight loader by "
        "vllm/models/deepseek_v4_1/nvidia/model.py load_weights (which substitutes "
        "loaded_weight.view(torch.uint8) only for names containing '.experts.', so these names should "
        "arrive as float8_e8m0fnu), and restore_weights_for_rl_loading, which leaves the already "
        f"processed uint8 scales in place. Check the checkpoint's .scale dtype{where} first."
    )


def recover_e8m0_scale_bytes(
    scale: torch.Tensor,
    *,
    declared_dtype: torch.dtype | None = None,
    param_name: str = "weight_scale",
    prefix: str | None = None,
) -> torch.Tensor:
    """Return the raw e8m0 exponent byte of every element of a block-scale tensor.

    Every dtype that carries the byte has its OWN branch. There is deliberately no
    fall-through: the branch that used to catch everything else decoded a uint8 byte view
    as float32 bits, at exactly the right shape.
    """
    dtype = scale.dtype
    if dtype == torch.float8_e8m0fnu:
        # Byte contract: the byte itself, reinterpreted.
        return scale.view(torch.uint8)
    if dtype == torch.uint8:
        # Byte contract: already the byte, because the loader handed over a uint8 view.
        return scale
    if dtype == torch.int8:
        # Byte contract: also byte-exact, two's complement wraps back under the view.
        return scale.view(torch.uint8)
    if dtype == torch.float32:
        # Value contract: the byte is in the exponent field of an exactly representable
        # power of two.
        bits = scale.view(torch.int32)
        exponent = bits >> FP32_EXPONENT_SHIFT & FP32_EXPONENT_MASK
        # Tripwire, not a proof: a value-copied e8m0 scale is a power of two, so its
        # mantissa is zero except for the two codes fp32 cannot hold as a normal number
        # (code 0 -> the subnormal 2**-127, code 255 -> NaN). A byte that was copied as a
        # NUMBER instead -- the failure this whole comment is about -- lands off a power
        # of two for 247 of the 256 codes, so any real scale tile trips this.
        off_power_of_two = (bits & FP32_MANTISSA_MASK != 0) & (exponent != 0) & (exponent != FP32_EXPONENT_MASK)
        # This is a load-time validity check. Reduce on CPU because the
        # image's CANN ReduceAny kernel times out on 40,960 bool elements.
        off_power_of_two_cpu = off_power_of_two.cpu()
        if bool(off_power_of_two_cpu.any()):
            raise ValueError(
                f"{param_name}: float32 block scales must be value-copied e8m0 codes, i.e. powers of "
                f"two, but {int(off_power_of_two_cpu.sum())} of {off_power_of_two.numel()} elements are not. "
                "The checkpoint bytes were most likely copied as numbers rather than converted from "
                "float8_e8m0fnu, which destroys every exponent. "
                + _scale_dtype_provenance(dtype, declared_dtype, prefix)
            )
        return exponent.to(torch.uint8)
    raise ValueError(
        f"{param_name} has dtype {dtype}, which carries no e8m0 exponent byte this code knows how to "
        "read: float8_e8m0fnu/uint8/int8 carry it as a raw byte and float32 carries it in the exponent "
        "field. Decoding any other dtype as float32 bits would produce wrong scales at exactly the "
        "right shape, with nothing downstream to complain. "
        + _scale_dtype_provenance(dtype, declared_dtype, prefix)
    )


@functools.cache
def assert_e8m0_scale_load_preserves_bytes(param_dtype: torch.dtype, loader_gives_byte_view: bool) -> None:
    """Fail unless this (loader half, declared dtype) pair round-trips every e8m0 byte.

    Run before any weight is loaded. This is behavioural on purpose: a ``dtype == uint8``
    allow-list passes every destructive dtype nobody thought to list, and rejects the
    byte-safe ones (int8) that are not on it.
    """
    # 🔴 device="cpu" 是必须的，不是风格。这是纯 host 侧的 256 字节契约往返，与设备无关；
    # 但它在 load_model 里被调用，那时 torch 的默认设备已被 DeviceContext 接管切到 NPU
    # （torch/utils/_device.py 的 __torch_function__），于是这句会去调 aclnnArange，
    # 而它**不支持 uint8**：
    #   AclNN_Parameter_Error(EZ1001): Tensor out not implemented for DT_UINT8,
    #   should be in [DT_FLOAT16,DT_FLOAT,DT_DOUBLE,DT_INT64,DT_INT32,DT_BFLOAT16]
    # ⇒ 整个模型构造在起服务阶段就挂（2026-09-20 的回归实测）。
    # CPU 套件看不见这一类：那里的默认设备是 CPU，而**默认设备不在任何签名里**。
    codes = torch.arange(E8M0_CODE_COUNT, dtype=torch.uint8, device="cpu")
    checkpoint = codes.view(torch.float8_e8m0fnu)
    loaded = checkpoint.view(torch.uint8) if loader_gives_byte_view else checkpoint
    param = torch.empty(codes.shape, dtype=param_dtype, device="cpu")
    try:
        param.copy_(loaded)
        recovered = recover_e8m0_scale_bytes(param)
        preserved = recovered.dtype == torch.uint8 and torch.equal(recovered, codes)
    except Exception as exc:
        raise ValueError(
            f"Loading an e8m0 block scale as {param_dtype} "
            f"(loader_gives_byte_view={loader_gives_byte_view}) failed: {exc}"
        ) from exc
    if not preserved:
        differing = int((recovered.view(torch.uint8) != codes).sum()) if recovered.numel() == codes.numel() else -1
        raise ValueError(
            f"Declaring an e8m0 block scale as {param_dtype} with "
            f"loader_gives_byte_view={loader_gives_byte_view} does not preserve the exponent "
            f"bytes: {differing} of {E8M0_CODE_COUNT} codes come back different. Both halves of "
            "the contract have to move together -- the loader's view and this dtype."
        )


@register_scheme("W8A8_MXFP8", "linear")
class AscendW8A8MXFP8DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_MXFP8 (Microscaling FP8) quantization.

    This scheme uses microscaling FP8 quantization with per-group scales.
    The activation is dynamically quantized to FP8 (E4M3FN format) with
    microscaling, and weights are stored in FP8 format with per-group scales.
    """

    model_dtype = None
    weight_switch_gather_specs = (
        WeightSwitchGatherSpec("weight"),
        WeightSwitchGatherSpec("weight_scale"),
    )
    weight_switch_output_gather_specs = (
        WeightSwitchGatherSpec("weight", gather_dim=1),
        WeightSwitchGatherSpec("weight_scale", gather_dim=1),
    )
    supports_weight_switch = True

    def __init__(self):
        vllm_config = get_current_vllm_config()
        quant_description = getattr(vllm_config.quant_config, "quant_description", None) or {}
        self.group_size = quant_description.get("group_size", 32)
        self.dynamic_mx_quant_scale_alg = get_dynamic_mx_quant_scale_alg(vllm_config)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, cdiv(input_size, self.group_size), dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            quantized_x, pertoken_scale = x
            original_shape = quantized_x.shape
            output_dtype = torch.bfloat16
        else:
            # reshape x for Qwen VL models
            original_shape = x.shape
            if x.dim() > 2:
                x = x.view(-1, x.shape[-1])
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                x,
                dst_type=torch.float8_e4m3fn,
                scale_alg=self.dynamic_mx_quant_scale_alg,
            )
            output_dtype = x.dtype

        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=[1, 1, self.group_size],
        )
        # reshape output for Qwen VL models
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP8 inference.

        This method transforms weights for NPU MXFP8 computation:
        - weight: (output_size, input_size) -> (input_size, output_size)
        - weight_scale: (n_dim, k_dim) -> (k_dim//2, n_dim, 2)

        For RL training scenarios where weights need to be reloaded multiple times,
        this method stores original shapes and can be called multiple times safely.
        Use restore_weights_for_rl_loading() before weight reload, then call this
        method again after loading.

        Address stability for ACL graph:
        The transformed buffer is what the ACL graph captures and replays. It is
        allocated once and cached on the layer; subsequent calls copy the
        (re)loaded original-shape data in place into the cached buffer so its
        data_ptr never changes across RL weight reloads.
        """

        # Check if already transformed to avoid double transformation
        if getattr(layer, "_mxfp8_transformed", False):
            return

        # Store original shapes for RL weight reloading
        # Only store on first call (when shapes are in original format)
        if not hasattr(layer, "_mxfp8_original_shapes"):
            layer._mxfp8_original_shapes = {
                "weight": tuple(layer.weight.data.shape),
                "weight_scale": tuple(layer.weight_scale.data.shape),
            }

        n_dim, k_dim = layer.weight_scale.data.shape
        # Shape should be padded if it cannot be divided by 2
        if layer.weight_scale.data.shape[-1] % 2 != 0:
            reshaped_scale = F.pad(layer.weight_scale.data, (0, 1), mode="constant", value=0)
            reshaped_scale = reshaped_scale.reshape(n_dim, k_dim // 2 + 1, 2)
        else:
            reshaped_scale = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        target_scale = reshaped_scale.transpose(0, 1)

        if not hasattr(layer, "_mxfp8_weight_buf"):
            # First call: allocate the persistent transformed buffers.
            layer._mxfp8_weight_buf = layer.weight.data.transpose(0, 1).contiguous()
            layer._mxfp8_scale_buf = target_scale.contiguous()
        else:
            # Subsequent calls (RL reload path): copy in place to keep data_ptr stable.
            layer._mxfp8_weight_buf.copy_(layer.weight.data.transpose(0, 1).contiguous())
            layer._mxfp8_scale_buf.copy_(target_scale.contiguous())

        layer.weight.data = layer._mxfp8_weight_buf
        layer.weight_scale.data = layer._mxfp8_scale_buf

        # Mark as transformed
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore weights to original shapes for RL weight reloading.

        This method must be called BEFORE model.load_weights() in RL training
        loops to restore the tensors to their original shapes that the weight
        loader expects.

        After weight loading, call process_weights_after_loading() again to
        re-apply the MXFP8 transformations.

        Shape transformations reversed:
        - weight: (input_size, output_size) -> (output_size, input_size)
        - weight_scale: (k_dim//2, n_dim, 2) -> (n_dim, k_dim)
        """

        if not getattr(layer, "_mxfp8_transformed", False):
            # Not transformed, nothing to restore
            return

        if not hasattr(layer, "_mxfp8_original_shapes"):
            err_msg = (
                "[vllm-ascend/W8A8_MXFP8] Cannot restore weights: original "
                "shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        orig_shapes = layer._mxfp8_original_shapes
        orig_scale_shape = orig_shapes["weight_scale"]

        # Restore weight: (input_size, output_size) -> (output_size, input_size)
        target_weight = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight.data.copy_(target_weight)

        # Restore weight_scale: (k_dim//2, n_dim, 2) -> (n_dim, k_dim)
        # Current shape: (k_dim//2, n_dim, 2)
        # Target shape: (n_dim, k_dim)
        target_scale = layer.weight_scale.data.transpose(0, 1).reshape(orig_scale_shape).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).reshape(orig_scale_shape)
        layer.weight_scale.data.copy_(target_scale)

        # Mark as not transformed (ready for weight loading)
        layer._mxfp8_transformed = False


@register_scheme("W8A8_MXFP8", "moe")
class AscendW8A8MXFP8DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W8A8_MXFP8."""

    model_dtype = None
    quant_type: QuantType = QuantType.W8A8MXFP
    act_quant_type: torch.dtype = torch.float8_e4m3fn
    fused_activations = frozenset({"silu"})
    supports_eplb = True

    def __init__(self):
        vllm_config = get_current_vllm_config()
        quant_description = getattr(vllm_config.quant_config, "quant_description", None) or {}
        self.group_size = quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else ascend_config.eplb_config.dynamic_eplb

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.float8_e4m3fn
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.float8_e4m3fn
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
        )
        return param_dict

    def apply(
        self,
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if topk_weights is None or topk_ids is None:
            raise RuntimeError("topk_weights and topk_ids must be set before fused MoE execution.")

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
                mxfp_act_quant_type=torch.float8_e4m3fn,
                mxfp_weight_quant_type=torch.float8_e4m3fn,
                mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
            ),
            quant_method=self,
        )

    @staticmethod
    def get_eplb_weight_views(layer: torch.nn.Module) -> list[torch.Tensor]:
        return [
            layer.w13_weight.transpose(1, 2),
            layer.w2_weight.transpose(1, 2),
            layer.w13_weight_scale.transpose(1, 2),
            layer.w2_weight_scale.transpose(1, 2),
        ]

    def process_weights_after_loading(self, layer):
        """Process weights after loading for MXFP8 inference.

        This method transforms weights for NPU MXFP8 computation:
        - w13_weight: (g_num, n_size, k_size) -> (g_num, k_size, n_size)
        - w2_weight: (g_num, n_size, k_size) -> (g_num, k_size, n_size)
        - w13_weight_scale: (g_num, n_size, k_size) -> (g_num, k_size//2, n_size, 2)
        - w2_weight_scale: (g_num, n_size, k_size) -> (g_num, k_size//2, n_size, 2)

        For RL training scenarios where weights need to be reloaded multiple times,
        this method stores original shapes and can be called multiple times safely.
        Use restore_weights_for_rl_loading() before weight reload, then call this
        method again after loading.
        """

        # Check if already transformed to avoid double transformation
        if getattr(layer, "_mxfp8_transformed", False):
            return

        # Store original shapes for RL weight reloading
        # Only store on first call (when shapes are in original format)
        if not hasattr(layer, "_mxfp8_original_shapes"):
            layer._mxfp8_original_shapes = {
                "w13_weight": tuple(layer.w13_weight.data.shape),
                "w13_weight_scale": tuple(layer.w13_weight_scale.data.shape),
                "w2_weight": tuple(layer.w2_weight.data.shape),
                "w2_weight_scale": tuple(layer.w2_weight_scale.data.shape),
            }

        g_num, n_size, k_size = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        g_num, n_size, k_size = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2)

        # Mark as transformed
        layer._mxfp8_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Restore weights to original shapes for RL weight reloading.

        This method must be called BEFORE model.load_weights() in RL training
        loops to restore the tensors to their original shapes that the weight
        loader expects.

        After weight loading, call process_weights_after_loading() again to
        re-apply the MXFP8 transformations.

        Shape transformations reversed:
        - w13_weight: (g_num, k_size, n_size) -> (g_num, n_size, k_size)
        - w2_weight: (g_num, k_size, n_size) -> (g_num, n_size, k_size)
        - w13_weight_scale: (g_num, k_size//2, n_size, 2) -> (g_num, n_size, k_size)
        - w2_weight_scale: (g_num, k_size//2, n_size, 2) -> (g_num, n_size, k_size)
        """

        if not getattr(layer, "_mxfp8_transformed", False):
            # Not transformed, nothing to restore
            return

        if not hasattr(layer, "_mxfp8_original_shapes"):
            err_msg = (
                "[vllm-ascend/W8A8_MXFP8] Cannot restore weights: original "
                "shapes not recorded. "
                "This should not happen if process_weights_after_loading was called first."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        orig_shapes = layer._mxfp8_original_shapes

        def _restore(weight_key: str, scale_key: str):
            """Helper to restore a single MoE weight and its scale using safe memory copies."""
            # --- 1. Restore Weight ---
            weight_tensor = getattr(layer, weight_key)
            target_weight = weight_tensor.data.transpose(1, 2).contiguous()
            weight_tensor.data = weight_tensor.data.transpose(1, 2)
            weight_tensor.data.copy_(target_weight)

            # --- 2. Restore Weight Scale ---
            scale_tensor = getattr(layer, scale_key)
            orig_scale_shape = orig_shapes[scale_key]

            target_scale = scale_tensor.data.transpose(1, 2).reshape(orig_scale_shape).contiguous()
            scale_tensor.data = scale_tensor.data.transpose(1, 2).view(orig_scale_shape)
            scale_tensor.data.copy_(target_scale)

        _restore("w13_weight", "w13_weight_scale")
        _restore("w2_weight", "w2_weight_scale")

        # Mark as not transformed (ready for weight loading)
        layer._mxfp8_transformed = False

    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        assert layer is not None

        hidden_states, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=[layer.w13_weight],
            group_list=cumsum_group_list(mlp_compute_input.group_list, mlp_compute_input.group_list_type, 0),
            weight_scale=[layer.w13_weight_scale],
            x_scale=pertoken_scale,
            dequant_mode=2,
            quant_mode=2,
            dequant_dtype=torch.float32,
            quant_dtype=torch.float8_e4m3fn,
            x_dtype=None,
            weight_dtype=None,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        assert layer is not None
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale],
            per_token_scale=[pertoken_scale],
            bias=None,
            split_item=2,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            output_dtype=torch.bfloat16,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=self.act_quant_type, scale_alg=get_dynamic_mx_quant_scale_alg()
        )
        return hidden_states, maybe_normalize_mxfp_scale_layout(dynamic_scale)

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        layer = mlp_compute_input.layer
        assert layer is not None
        input_dtype = mlp_compute_input.hidden_states.dtype
        use_bf16 = input_dtype in [torch.bfloat16, torch.float8_e4m3fn]
        output_dtype = (
            input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else (torch.bfloat16 if use_bf16 else torch.float16)
        )
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale],
            bias=None,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            output_dtype=output_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=None,
            weight_dtype=None,
        )[0]


@register_scheme(FP8_METHOD, "ds_linear")
class AscendW8A8MXFP8DSDynamicLinearMethod(AscendW8A8MXFP8DynamicLinearMethod):
    """Linear method for DS W8A8 mxfp quantization with square block scales.

    The checkpoint stores one e8m0 scale per ``weight_block_size`` tile -- 128 x 128 for DeepSeek-V4,
    32 x 32 for V4.1 -- and ``process_weights_after_loading`` broadcasts those out to the
    ``group_size``-along-K layout (32 x 1) that the GEMM consumes.
    """

    model_dtype = None
    weight_switch_gather_specs = (
        WeightSwitchGatherSpec("weight"),
        WeightSwitchGatherSpec("weight_scale"),
    )
    weight_switch_output_gather_specs = (
        WeightSwitchGatherSpec("weight"),
        WeightSwitchGatherSpec("weight_scale"),
    )
    supports_weight_switch = True
    # The two halves of the scale contract, kept next to each other so neither can move
    # alone. False: the DS linear scale names hold no ".experts.", so the model's
    # load_weights hands the float8_e8m0fnu tensor straight to the weight loader.
    scale_param_dtype = torch.float32
    loader_gives_scale_byte_view = False

    def __init__(self, weight_block_size):
        super().__init__()
        # Only square tiles are handled, and the broadcast below needs whole groups per tile.
        if weight_block_size[0] != weight_block_size[1]:
            raise NotImplementedError(f"Non-square FP8 weight blocks are unsupported: {weight_block_size}.")
        if weight_block_size[0] % self.group_size != 0:
            raise NotImplementedError(
                f"FP8 weight block size {weight_block_size[0]} is not a multiple of the mx group size "
                f"{self.group_size}, so block scales cannot be broadcast to the GEMM's layout."
            )
        self.block_size = weight_block_size[0]
        vllm_config = get_current_vllm_config()
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        hf_config = vllm_config.model_config.hf_config
        self.n_groups = hf_config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.o_lora_rank = hf_config.o_lora_rank

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        # Declared float32: the loader value-copies the e8m0 checkpoint scale into it, and
        # process_weights_after_loading recovers the exponent byte from the fp32 bits. This relies
        # on that value copy being exact; declaring it e8m0/uint8 here would change that contract.
        # The guard proves that round trip on all 256 codes before anything is loaded, so a change
        # to either half of the contract fails here instead of silently rescaling the model.
        assert_e8m0_scale_load_preserves_bytes(self.scale_param_dtype, self.loader_gives_scale_byte_view)
        params_dict["weight_scale"] = torch.empty(
            output_size // self.block_size, input_size // self.block_size, dtype=self.scale_param_dtype
        )
        params_dict["_packed_dim"] = 0
        params_dict["_packed_factor"] = self.block_size
        return params_dict

    def process_weights_after_loading(self, layer):
        # Exhaustive over the dtypes that carry an e8m0 byte; anything else raises instead of
        # falling through to the float32 bit extraction, which used to take every other dtype
        # -- a uint8 view included -- and decode it as fp32 bits.
        layer.weight_scale.data = recover_e8m0_scale_bytes(
            layer.weight_scale.data,
            declared_dtype=self.scale_param_dtype,
            param_name="weight_scale",
            prefix=getattr(layer, "prefix", None),
        )
        # Broadcast the [out/block, in/block] tile scales to [out, in/group_size], the layout
        # npu_quant_matmul consumes: (4, 128) for a 128 block, (1, 32) for a 32 block.
        # Expand the byte scales on CPU at load time; this image's large
        # RepeatInterleave NPU kernel also times out for uint8 scale tiles.
        scale_device = layer.weight_scale.device
        layer.weight_scale.data = (
            layer.weight_scale.data.cpu()
            .repeat_interleave(self.block_size // self.group_size, dim=1)
            .repeat_interleave(self.block_size, dim=0)
            .to(scale_device)
        )
        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

        if layer.prefix.endswith("wo_a"):
            layer.weight.data = (
                layer.weight.data.T.reshape(self.n_local_groups, self.o_lora_rank, -1).transpose(1, 2).contiguous()
            )
            layer.weight_scale.data = (
                layer.weight_scale.data.transpose(0, 1)
                .reshape(self.n_local_groups, self.o_lora_rank, -1, 2)
                .transpose(1, 2)
                .contiguous()
            )
