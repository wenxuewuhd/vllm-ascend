# SPDX-License-Identifier: Apache-2.0
"""The three hardware-reaching entry points of the V4.1 DSpark drafter.

Upstream's drafter (``vllm/models/deepseek_v4_1/nvidia/dspark.py``) touches
hardware in exactly three module-level names, each read from that module's
globals at call time, so each can be rebound rather than forked:

===========================  ============================================
upstream name                why it cannot run here
===========================  ============================================
``mhc_post_tilelang``        tilelang is not installed
``hc_collapse_triton``       Triton; the target's collapse is already an
                             Ascend operator, so this reuses it
``_insert_context_kv``       ``torch.ops._C.fused_deepseek_v4_*``, a CUDA
                             C++ extension that does not exist here
===========================  ============================================

``mhc_post_tilelang`` already has a replacement in :mod:`.mhc` (the target's
decoder layer calls it too). The other two live here.

This module deliberately imports nothing from :mod:`.platform_patches` or
:mod:`.dspark`: ``platform_patches.rebind_dspark_functions`` imports *this*,
and a reverse edge would make that a cycle.
"""

from __future__ import annotations

import torch

from vllm_ascend.models.deepseek_v4_1.mhc import _collapse_with
from vllm_ascend.models.deepseek_v4_1.qkv_prolog import qnorm_rope_kv_insert


def hc_collapse_ascend(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """``hc_collapse_triton``'s contract, on the target model's collapse.

    Upstream collapses the ``hc_mult`` residual streams into one with the
    previous sublayer's pre-mix coefficients -- the same weighted sum over the
    stream axis that :func:`.mhc._collapse_with` already does for the target's
    delayed mHC path, and therefore the same operator and the same
    ``VLLM_ASCEND_DSV41_MHC_FUSED_OPS`` switch. Reusing it is the point: a
    second implementation of this sum could disagree with the target's.

    ⚠ Argument order is **reversed** between the two. Upstream takes
    ``(x, pre_mix)``, ``_collapse_with`` takes ``(pre_mix, residual)``. Both
    arguments are floating tensors whose leading dimension is the token count,
    so swapping them would not raise -- it would broadcast
    ``[T, hc]`` against ``[T, hc, D]`` and return silently wrong values. Hence
    the shape checks below, which upstream's kernel also asserts.
    """
    if x.ndim != 3:
        raise ValueError(
            f"x has shape {tuple(x.shape)}; expected (tokens, hc, hidden). "
            "This is the argument upstream passes first and "
            "_collapse_with takes second, so a shape that is not 3-D is "
            "most likely the two swapped."
        )
    if pre_mix.ndim != 2:
        raise ValueError(f"pre_mix has shape {tuple(pre_mix.shape)}; expected (tokens, hc).")
    if pre_mix.shape != x.shape[:2]:
        raise ValueError(
            f"pre_mix is {tuple(pre_mix.shape)} against x "
            f"{tuple(x.shape)}; the first two axes must agree. Unequal "
            "token counts broadcast rather than raise."
        )
    out = _collapse_with(pre_mix, x)
    if out.shape != (x.shape[0], x.shape[2]):
        raise RuntimeError(
            f"collapse returned {tuple(out.shape)} for x {tuple(x.shape)}; expected {(x.shape[0], x.shape[2])}."
        )
    return out


def insert_context_kv_ascend(
    attn: torch.nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """``_insert_context_kv``'s contract, on the Ascend prolog.

    Upstream fuses RoPE + quant + paged insert into one CUDA op per cache
    dtype and passes a dummy query it then discards, because context tokens
    have no query. :func:`.qkv_prolog.qnorm_rope_kv_insert` is the decomposed
    bf16 branch the Ascend attention already uses for the target
    (``AscendDeepseekV41Attention._fused_qnorm_rope_kv_insert``), so the
    drafter writes its sliding-window context KV through the same code the
    target writes its own KV through.

    The dummy query is still built: the shared helper rotates ``q`` and
    ``kv`` together off one cos/sin pair, and its returned ``q`` is dropped
    here exactly as upstream drops the one its kernel wrote.
    """
    swa_cache = attn.swa_cache_layer.kv_cache
    if swa_cache.dtype is not torch.bfloat16:
        raise NotImplementedError(
            f"the Ascend drafter covers the bf16 SWA cache; this one is "
            f"{swa_cache.dtype}. Upstream's other two branches are not this "
            "function with a different dtype: the uint8 (fp8_ds_mla) branch "
            "writes a different paged layout, and the per-tensor fp8 branch "
            "needs attn._flashinfer_fp8_{kv_scale,q_scale_inv}, which the "
            "Ascend attention does not define. Serve with "
            "--kv-cache-dtype bfloat16, which is what the single-card recipe "
            "already passes."
        )

    # Block size from the cache itself, cross-checked against the layer. The
    # two index each other: a disagreement writes rows at the wrong offset
    # inside the right block instead of failing.
    block_size = swa_cache.shape[1]
    declared = getattr(attn.swa_cache_layer, "block_size", None)
    if declared is not None and declared != block_size:
        raise ValueError(
            f"{attn.swa_cache_layer.prefix}: the cache is "
            f"{tuple(swa_cache.shape)} so block_size is {block_size}, but the "
            f"layer declares {declared}."
        )
    if swa_cache.shape[2] != attn.head_dim:
        raise ValueError(f"the SWA cache's row is {swa_cache.shape[2]} wide against a head dim of {attn.head_dim}.")
    if kv.shape[0] != positions.shape[0]:
        raise ValueError(
            f"kv has {kv.shape[0]} rows and positions {positions.shape[0]}; "
            "they are indexed together, and the shorter one would silently "
            "rotate the wrong tokens."
        )

    dummy_q = torch.zeros(
        (kv.shape[0], attn.n_local_heads, attn.head_dim),
        dtype=kv.dtype,
        device=kv.device,
    )
    qnorm_rope_kv_insert(
        q=dummy_q,
        kv=kv,
        positions=positions,
        cos_sin_cache=attn.rotary_emb.cos_sin_cache,
        swa_kv_cache=swa_cache,
        slot_mapping=slot_mapping,
        eps=attn.eps,
        rope_head_dim=attn.rope_head_dim,
    )
