# SPDX-License-Identifier: Apache-2.0
"""Import-time platform adaptations for DeepSeek-V4.1 on Ascend.

Everything here rebinds a name upstream already exposes for platform
differences, rather than forking the code around it. Applied at import, before
any forward pass, because one of them guards a compiler crash that no
``try``/``except`` can catch.
"""

from __future__ import annotations

from typing import Any

#: Upstream's switch for reading a ratio-2 request's two pooled rows as one
#: tile. Its comment already records that ROCm's Triton cannot legalize
#: ``tt.join`` on pointers; Ascend's cannot either.
JOIN_FLAG = "_JOIN_ROW_PTRS"


def _compress_module() -> Any:
    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as mod

    return mod


def disable_join_row_ptrs() -> bool:
    """Take the separate-loads path in the compressor kernel. Idempotent.

    On the joined path Ascend's compiler does not raise -- it aborts, in
    ``OffsetAnalysis.cpp`` while parsing a pointer type, taking the process and
    a core file with it. So this cannot be a runtime fallback: the flag is read
    as a ``tl.constexpr`` when the kernel is first compiled, and by then there
    is nothing left to recover into.

    The two branches are arithmetically the same -- ``tl.join`` plus a two-line
    softmax against two loads plus an explicit online softmax -- so this changes
    scheduling, not semantics. Upstream reaches the joined path here only
    because its guard tests for ROCm specifically rather than for the
    limitation.

    Returns True if the flag was changed.
    """
    mod = _compress_module()
    if not hasattr(mod, JOIN_FLAG):
        raise AttributeError(
            f"upstream no longer defines {JOIN_FLAG!r} in "
            "fused_compress_quant_cache. The joined-pointer path may have been "
            "removed, or renamed -- but if it still exists and nothing selects "
            "away from it, compiling a ratio-2 compressor here aborts the "
            "process. Re-read the module before assuming this patch is "
            "obsolete."
        )
    if getattr(mod, JOIN_FLAG) is False:
        return False
    setattr(mod, JOIN_FLAG, False)
    return True


def join_row_ptrs_disabled() -> bool:
    """Whether the compressor will take the separate-loads path."""
    return getattr(_compress_module(), JOIN_FLAG) is False


def assert_ready_for_ratio_two(where: str) -> None:
    """Refuse to reach a ratio-2 compressor compile with the joined path live.

    Without this the failure surfaces inside the compiler: it aborts the
    process in OffsetAnalysis.cpp and leaves a core file, at whatever moment
    the first ratio-2 kernel happens to be compiled -- which is well after
    everything looks configured. Checking here moves it somewhere readable.
    """
    if not join_row_ptrs_disabled():
        raise RuntimeError(
            f"{where}: compressor ratio 2 is about to be compiled while "
            f"{JOIN_FLAG} is still true. Ascend's compiler does not raise on "
            "that path -- it aborts the process. apply_all() has not run, or "
            "something re-imported the module afterwards."
        )


#: Classes upstream constructs by bare name, and the module whose globals the
#: construction site reads. All three are built inline inside an ``__init__``,
#: so subclassing the owner cannot replace them -- and for the caches,
#: subclassing the cache itself is worse than useless: ``sparse_swa`` raises on
#: a duplicate prefix, so a second instance would abort rather than shadow the
#: first.
CLASS_REBINDS = (
    ("vllm.models.deepseek_v4_1.attention", "DeepseekV4SWACache", "AscendDeepseekV41SWACache"),
    ("vllm.models.deepseek_v4_1.compressor", "CompressorStateCache", "AscendDeepseekV41CompressorStateCache"),
    ("vllm.models.deepseek_v4_1.common.engram", "ParallelEngramEmbedding", "AscendEngramEmbedding"),
    ("vllm.models.deepseek_v4_1.attention", "DeepseekV4IndexerCache", "AscendDeepseekV41IndexerCache"),
)


def rebind_upstream_classes() -> list[str]:
    """Point the V4.1 construction sites at the Ascend subclasses.

    Returns the names rebound. Refuses rather than skipping when a name is
    gone: these sites are reached during KV cache setup, and a patch that
    silently became a no-op would surface much later as a spec the Ascend
    runner cannot reshape, or a backend with no get_kv_cache_shape.
    """
    import importlib

    from vllm_ascend.models.deepseek_v4_1 import caches

    rebound = []
    for module_name, upstream_name, ascend_name in CLASS_REBINDS:
        module = importlib.import_module(module_name)
        # Checked before reading: a bare getattr raises Python's own terse
        # AttributeError and this explanation is never reached.
        if not hasattr(module, upstream_name):
            raise AttributeError(
                f"{module_name} no longer defines {upstream_name!r}. The V4.1 "
                "class it names may have been renamed or moved. Until this is "
                "repointed the site builds upstream's class, which on Ascend "
                "means a kv-cache spec the runner cannot reshape, a backend "
                "with no get_kv_cache_shape, or -- for the engram table -- an "
                "attempt to allocate the whole table."
            )
        replacement = getattr(caches, ascend_name)
        if getattr(module, upstream_name) is replacement:
            continue
        setattr(module, upstream_name, replacement)
        rebound.append(f"{module_name}.{upstream_name}")
    return rebound


#: Hyper-connection entry points the V4.1 decoder layer calls directly in its
#: forward. They are plain module-level names in upstream's model module, bound
#: there by a ``from ... import`` at import time, so rebinding the attribute on
#: the already-imported module reaches every call site without forking the
#: layer. tilelang is not installed here and these are the only two names that
#: need it.
MHC_REBINDS = (
    ("mhc_pre_delayed_tilelang", "mhc_pre_delayed_ascend"),
    ("mhc_post_tilelang", "mhc_post_ascend"),
)


def rebind_mhc_functions() -> list[str]:
    """Point the decoder layer's mHC calls at the tilelang-free versions.

    Returns the names rebound. Refuses rather than skipping when one is gone:
    the failure it prevents is an ImportError inside the first decoder layer's
    forward, which is late enough that the model has already loaded and every
    earlier stage has reported success.
    """
    from vllm_ascend.models.deepseek_v4_1 import mhc

    upstream = _upstream_model_module()
    rebound = []
    for upstream_name, ascend_name in MHC_REBINDS:
        if not hasattr(upstream, upstream_name):
            raise AttributeError(
                f"upstream's V4.1 model module no longer defines "
                f"{upstream_name!r}. Until this is repointed the decoder layer "
                "calls tilelang, which is not installed here, and it fails in "
                "the first forward rather than at load."
            )
        replacement = getattr(mhc, ascend_name)
        if getattr(upstream, upstream_name) is replacement:
            continue
        setattr(upstream, upstream_name, replacement)
        rebound.append(upstream_name)
    return rebound


#: The V4.1 DSpark drafter's hardware entry points, as module-level names in
#: upstream's ``nvidia/dspark.py``. Same mechanism as MHC_REBINDS above: the
#: drafter reads all three from its own module globals at call time, so
#: rebinding the attributes reaches every call site without forking the 558
#: lines around them. ``mhc_post_tilelang`` is shared with the target and its
#: replacement lives in ``mhc``; the other two live in ``dspark_ops``.
DSPARK_REBINDS = (
    ("mhc_post_tilelang", "mhc_post_ascend", "mhc"),
    ("hc_collapse_triton", "hc_collapse_ascend", "dspark_ops"),
    ("_insert_context_kv", "insert_context_kv_ascend", "dspark_ops"),
)


def _upstream_dspark_module() -> Any:
    import vllm.models.deepseek_v4_1.nvidia.dspark as upstream

    return upstream


def _dspark_replacement(ascend_name: str, owner: str) -> Any:
    from vllm_ascend.models.deepseek_v4_1 import dspark_ops, mhc

    return getattr({"mhc": mhc, "dspark_ops": dspark_ops}[owner], ascend_name)


def rebind_dspark_functions() -> list[str]:
    """Point the V4.1 drafter's three hardware calls at the Ascend versions.

    Returns the names rebound. Applied for every V4.1 load, not only when
    speculative decoding is configured: the cost is one module import that
    defines classes and allocates nothing, and making it conditional would put
    the rebind after the point where the drafter can already be constructed.

    Refuses rather than skipping when a name is gone. The failure it prevents
    is in the drafter's *first forward* -- a tilelang import that is not
    installed, or ``torch.ops._C.fused_deepseek_v4_*``, which does not exist
    here -- which is after 7.39 GiB of draft weights have loaded and every
    earlier stage has reported success.
    """
    upstream = _upstream_dspark_module()
    rebound = []
    for upstream_name, ascend_name, owner in DSPARK_REBINDS:
        if not hasattr(upstream, upstream_name):
            raise AttributeError(
                f"upstream's V4.1 dspark module no longer defines "
                f"{upstream_name!r}. Until this is repointed the drafter calls "
                "CUDA's version of it, which on Ascend means an ImportError or "
                "a missing torch.ops symbol in the first draft forward."
            )
        replacement = _dspark_replacement(ascend_name, owner)
        if getattr(upstream, upstream_name) is replacement:
            continue
        setattr(upstream, upstream_name, replacement)
        rebound.append(upstream_name)
    return rebound


def dspark_rebinds_in_force() -> list[str]:
    """Names that are **not** pointing at the Ascend version, as messages.

    Empty means all three are in force. Read from the live module rather than
    from what ``apply_all`` returned, because an upstream module re-imported
    afterwards restores its own names while that list still reports success.
    """
    upstream = _upstream_dspark_module()
    wrong = []
    for upstream_name, ascend_name, owner in DSPARK_REBINDS:
        bound = getattr(upstream, upstream_name, None)
        if bound is _dspark_replacement(ascend_name, owner):
            continue
        wrong.append(
            f"{upstream.__name__}.{upstream_name} is {getattr(bound, '__name__', bound)!r}, expected {ascend_name}"
        )
    return wrong


def _apply_platform_patches_first() -> None:
    """Ensure vllm_ascend's global platform patches are in place."""
    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)


def _upstream_model_module():
    import vllm.models.deepseek_v4_1.nvidia.model as upstream

    return upstream


def apply_all() -> list[str]:
    """Apply every adaptation and report which. Safe to call more than once.

    Returns the names applied, and refuses to return quietly having done
    nothing: a patch set that silently becomes a no-op looks exactly like one
    that worked, and these two guard a compiler abort and a CUDA-only class
    selection respectively.
    """
    from vllm_ascend.models.deepseek_v4_1.attn_selector import apply_attn_selector_patch

    # Must run before any rebind below imports upstream's model module, which
    # from-imports FusedMoEFactory; importing it first would freeze the CUDA
    # factory there and the Ascend MoE router would never be installed.
    _apply_platform_patches_first()

    applied = []
    disable_join_row_ptrs()
    if join_row_ptrs_disabled():
        applied.append(JOIN_FLAG)
    apply_attn_selector_patch()
    applied.append("_select_dsv4_attn_cls")
    applied.extend(rebind_upstream_classes())
    applied.extend(rebind_mhc_functions())
    applied.extend(rebind_dspark_functions())

    # Ordering-immune, unlike the platform patch above: its one call site is
    # inside the module that owns the name, so the rebind takes effect whenever
    # it happens. See moe_router for why FusedMoEFactory is not like that.
    from vllm_ascend.models.deepseek_v4_1.moe_router import apply_moe_router_patch

    if apply_moe_router_patch():
        applied.append("create_fused_moe_router")

    # Same shape as the MoE router hook: the call site reads its own module's
    # globals, so this lands whenever it is applied.
    from vllm_ascend.models.deepseek_v4_1.indexer_q import apply_indexer_q_patch

    if apply_indexer_q_patch():
        applied.append("fused_indexer_q_rope_quant")

    # Same shape again. This one guards the other half of the indexer: the K
    # producer upstream ships only as a Triton kernel, so there is no platform
    # branch to select away from, and a per-token paged store is the pointer
    # shape that aborts triton-ascend rather than raising.
    from vllm_ascend.models.deepseek_v4_1.indexer_k_store import apply_indexer_k_store_patch

    if apply_indexer_k_store_patch():
        applied.append("indexer_k_norm_rope_store")

    # The compressor's latent publish: with a bf16 KV cache the slot mapping
    # arrives as (block, offset) pairs, which upstream reads as 2T flat slots.
    from vllm_ascend.models.deepseek_v4_1.compressor_insert import apply_compressor_insert_patch

    if apply_compressor_insert_patch():
        applied.append("rope_quant_insert")

    # The selection buffer's second writer: the short-context path fills it
    # with a Triton kernel and skips the row accounting done in forward_oot.
    from vllm_ascend.models.deepseek_v4_1.sparse_indexer import apply_short_context_accounting_patch

    if apply_short_context_accounting_patch():
        applied.append("_fill_short_context_topk_indices")

    if not applied:
        raise RuntimeError(
            "apply_all() applied nothing. Either upstream changed shape and "
            "every patch became a no-op, or this is being called against a "
            "module set it does not recognise."
        )
    return applied
