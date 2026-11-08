# SPDX-License-Identifier: Apache-2.0
"""Reuse the upstream V4.1 drafter with Ascend mHC and context-KV operations."""

import torch
from vllm.models.deepseek_v4_1.nvidia import dspark as upstream

from vllm_ascend.models.deepseek_v4_1.mhc import _collapse_with, mhc_post_ascend
from vllm_ascend.models.deepseek_v4_1.qkv_prolog import _rope_tail_impl, scatter_rows
from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice


def hc_collapse_ascend(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    # Upstream takes (x, pre_mix); the shared target helper reverses the order.
    if x.ndim != 3 or pre_mix.ndim != 2:
        raise ValueError("expected (tokens, hc, hidden) and (tokens, hc)")
    if pre_mix.shape != x.shape[:2]:
        raise ValueError("x and pre_mix first two axes must agree")
    return _collapse_with(pre_mix, x)


def insert_context_kv_ascend(
    attn: torch.nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    cache_layer = attn.swa_cache_layer
    cache = cache_layer.kv_cache
    if cache.dtype is not torch.bfloat16:
        raise NotImplementedError("DSpark requires a bf16 SWA cache; use --kv-cache-dtype bfloat16")
    declared = getattr(cache_layer, "block_size", None)
    if declared is not None and declared != cache.shape[1]:
        raise ValueError(f"SWA cache has block size {cache.shape[1]}, but the layer declares {declared}")
    if cache.shape[2] != attn.head_dim or kv.shape[0] != positions.shape[0]:
        raise ValueError("context KV dimensions must match the cache head and position count")

    cos, sin = cos_sin_for_slice(attn.rotary_emb.cos_sin_cache, positions, attn.rope_head_dim)
    # Reuse the target's rotation and paged write without allocating a dummy Q.
    rotated = _rope_tail_impl(kv)(kv, cos, sin, attn.rope_head_dim)
    scatter_rows(cache, rotated, slot_mapping, cache.shape[1])


class AscendDSparkDeepseekV41ForCausalLM(upstream.DSparkDeepseekV4ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        # The package has already applied target adaptations. Bind the three
        # drafter-only hardware calls before constructing the upstream model.
        upstream.mhc_post_tilelang = mhc_post_ascend
        upstream.hc_collapse_triton = hc_collapse_ascend
        upstream._insert_context_kv = insert_context_kv_ascend
        super().__init__(vllm_config=vllm_config, prefix=prefix)
