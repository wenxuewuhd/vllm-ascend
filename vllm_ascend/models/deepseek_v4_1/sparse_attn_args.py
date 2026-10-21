# SPDX-License-Identifier: Apache-2.0
"""Assemble and check the arguments to ``sparse_flash_mla``.

Every rule enforced here is one the operator states about itself (CANN
9.2.0), re-stated as a host-side check so a violation surfaces as one
readable error on CPU instead of an operator rejection -- or, for the rules
the operator does not check, an aicore exception -- on the card.

This is deliberately a pure function over already-built metadata. What
supplies that metadata is a separate question -- V4.0 builds it inside its
attention backend impl, which V4.1 does not use -- and keeping the two apart
means the rules do not have to be settled twice.

Mask modes. "Ori-only sparse mode requires both mask modes to be 0": an
explicit index list selects the original stream, so the window bounds have
nothing left to say and the call is mode 0. This was modelled from a
measurement rather than a reading -- the operator refused (4, 0) for a call
that passed ori_sparse_indices. Without an index list the original stream is
masked by the sliding window, mode 4 with bounds 127/0, on every V4.1 layer.

The operator's text says "SWA mode requires cmp_ratio=1", which would make
the window and a compressed stream mutually exclusive on ratio-2 layers.
CANN 9.2 on A5 accepts mode 4 with window bounds at cmp_ratio 2, and that is
the reference semantics (window plus ratio-2 causal compressed entries), so
the window is sent at cmp_ratio 1 and 2. Other ratios keep the documented
rule.

A wrong mask mode is not always rejected -- it can yield a finite, correctly
shaped, wrong answer -- which is why the mode is derived in one place
(``ori_mask_mode_for``) and used by both the descriptor and the call.
"""

from __future__ import annotations

from typing import Any

import torch

#: Accepted KV layouts. PA_ND -- which the FP8 plan uses -- belongs to a
#: different operator and is rejected here.
LAYOUTS_KV = ("TND", "BSND", "PA_BBND")

#: q's head dimension. The operator has no rope_head_dim parameter at all, so
#: the 64 rope dims are part of this 512 rather than declared separately.
HEAD_DIM = 512

#: Sliding-window mask mode, for a layer that attends its own stream by window
#: bounds rather than by an explicit index list.
SLIDING_WINDOW_MASK_MODE = 4
#: The mode required when the original stream is selected by explicit
#: indices -- the operator calls that "ori-only sparse mode" and refuses
#: anything else: "Parameters ori_mask_mode and cmp_mask_mode ... have
#: incorrect values 4, 0. Ori-only sparse mode requires both mask modes to
#: be 0."
NO_MASK_MODE = 0
#: Causal, for the compressed stream.
CMP_MASK_MODE = 3

#: Ratios at which the operator accepts window bounds. The documented rule is
#: "SWA mode requires cmp_ratio=1"; CANN 9.2 also accepts 2.
WINDOW_CMP_RATIOS = (1, 2)

#: The operator's mode rules, read out of its host library
#: (``op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_transformer.so``)
#: via ``strings``:
#:
#:   "ori_sparse_indices is only supported in SWA mode."
#:   "SWA mode requires cmp_mask_mode=0"
#:   "SWA mode requires cmp_ratio=1"
#:   "Sparse ori_kv mode requires ori_mask_mode=cmp_mask_mode=0"
#:   "Ori_topk_length is required for SWA ori sparse"
#:   "Cmp_topk_length is only supported in CSA mode"
#:   "When cmp_sparse_indices is not nullptr(CSA), cmp_sparse_indices cannot
#:    be empty tensor"
#:   "cmp_residual_k is required when mask_mode is 3 with cmp_ratio not equal
#:    to 1!"
#:
#: Supplying ``ori_sparse_indices`` *selects* SWA, and SWA's rules then apply
#: to everything else in the call; supplying ``cmp_sparse_indices`` selects
#: CSA. How the mode is chosen when several optional tensors are present is
#: not stated beyond these two implications.


def select_ori_sparse_indices(attends_compressed: bool, vision_swa_indices,
                              dspark_swa_indices):
    """The explicit original-KV index list this call will use, or ``None``.

    Shared by the descriptor builder and the call site so both reach the same
    answer. DSpark's indices override vision's, and only when no compressed
    stream is attended.

    This selector does not change index space. SparseFlashMla requires logical
    token positions within each request and performs the page lookup through
    ``ori_block_table`` itself. Its DSpark and vision producers must therefore
    use ``logical_indices=True``. Passing physical slot ids here applies the
    page mapping twice and silently reads unrelated KV, even though the tensor
    shape, dtype and valid lengths are correct.
    """
    if not attends_compressed and dspark_swa_indices is not None:
        return dspark_swa_indices
    return vision_swa_indices


def ori_sparse_selection(attends_compressed: bool, req) -> tuple:
    """``(indices, topk_length)`` together, because they are one decision.

    The operator reads ``oriSparseBlockCount = hasOriTopkLength ?
    Min(length, width) : width``, so indices without a length declare every
    slot in the row valid -- the ``-1`` padding included -- and the kernel
    gathers at those. The length is carried from whoever built the table and
    never recounted here.
    """
    indices = select_ori_sparse_indices(
        attends_compressed,
        getattr(req, "vision_swa_indices", None),
        getattr(req, "dspark_swa_indices", None))
    return indices, getattr(req, "ori_topk_length", None)


def is_plain_sliding_window(ori_sparse_indices) -> bool:
    """Whether this call is sliding-window mode (mask mode 4 with bounds).

    An explicit index list masks the original stream itself (mode 0).
    Without one, every V4.1 layer takes the window, whether or not it also
    attends a compressed stream: ratio-0/1 layers are sent cmp_ratio 1, and
    ratio-2 layers cmp_ratio 2, which CANN 9.2 accepts in window mode.
    """
    return ori_sparse_indices is None


def ori_mask_mode_for(ori_sparse_indices) -> int:
    """What masks this layer's original stream.

    ================  ===================  ====
    indices present   the list masks it    0
    no indices        the window masks it  4
    ================  ===================  ====

    Mode 0 is NO_MASK in the operator's own words, so it is only correct when
    the index list does the masking.
    """
    if is_plain_sliding_window(ori_sparse_indices):
        return SLIDING_WINDOW_MASK_MODE
    return NO_MASK_MODE


def _device_scalar(reduce, when_capturing: int) -> int:
    """Read a device scalar back to the host; ``when_capturing`` under capture.

    ``int(t)`` synchronises the stream, which is illegal during graph capture.
    Capture only happens during warmup on dummy data, and replayed requests
    are not captured, so the check still runs on every real batch. The caller
    passes a value that lets its own check pass.

    🔴 ``reduce`` is a **callable**, not a tensor, and that is the whole point
    of this signature. Passing ``table.max()`` evaluates the reduction at the
    call site -- including **during capture**, where the result is thrown away
    but the kernel has already been launched, and therefore **recorded into
    the graph and re-run on every replay for the life of the process**. The
    check was free in eager and pure waste in FULL graph mode, which is the
    mode production runs in. Measured 2026-09-22 (§6.9.15): these reductions
    were 512 kernels and **0.98 ms of a 28.8 ms decode step, 3.4%**, computing
    values nobody reads.
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import _stream_is_capturing

    if _stream_is_capturing():
        return when_capturing
    return int(reduce())


def _device_predicate(reduce) -> bool:
    """A device boolean as a host boolean; treated as true under capture.

    Same reasoning as ``_device_scalar``, including that ``reduce`` is a
    callable so that capture launches nothing.
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import _stream_is_capturing

    if _stream_is_capturing():
        return True
    return bool(reduce())


def _check(condition: bool, operator_message: str, detail: str = "") -> None:
    if condition:
        return
    raise ValueError(
        f"sparse_flash_mla would reject this call: {operator_message}"
        + (f"\n  {detail}" if detail else ""))


def _first_decrease(t) -> str:
    """Where ``cu_seqlens_q`` decreases, for the error message.

    Only called on the raising path, which is never under capture (the
    predicate passes there), so the host read-back is legal.
    """
    try:
        v = t.detach().to("cpu").tolist()
    except Exception as exc:  # the ValueError below must still be raised
        return f"; values unreadable ({type(exc).__name__}: {exc})"
    bad = [i for i in range(1, len(v)) if v[i] < v[i - 1]]
    if not bad:
        return f"; no decrease on the host copy (length {len(v)})"
    i = bad[0]
    return (f"; first decrease at index {i}: {v[i - 1]} -> {v[i]} "
            f"({len(bad)} decreases, length {len(v)})")


def _check_block_table_addresses_this_cache(kwargs: dict[str, Any]) -> None:
    """Refuse a block table or lengths that do not fit the cache they index.

    The table comes from the attention metadata and the cache from the layer,
    and the operator does not range-check them; every core reads through the
    same table, so one wrong table faults all of them. Bounds are read from
    the tensor passed here, never from a configured num_blocks: a configured
    value is true of every cache group, which is the confusion being checked
    for.

    ``.max()`` synchronises, once per layer per step.
    """
    # seqused_kv, not seqused_ori_kv: this runs before the sparse_flash_mla
    # adapter renames it.
    for table_key, cache_key, used_key in (
            ("ori_block_table", "ori_kv", "seqused_kv"),
            ("cmp_block_table", "cmp_kv", "seqused_cmp_kv")):
        table = kwargs.get(table_key)
        cache = kwargs.get(cache_key)
        if table is None or cache is None:
            continue
        num_blocks, block_size = int(cache.shape[0]), int(cache.shape[1])
        if table.numel() == 0:
            raise ValueError(
                f"{table_key} is empty, so this check did not run. An empty "
                "table is not a batch needing no blocks; it addresses nothing.")
        # -1 under capture always passes `highest >= num_blocks`.
        highest = _device_scalar(lambda table=table: table.max(), -1)
        if highest >= num_blocks:
            raise ValueError(
                f"{table_key}'s highest block id is {highest} and {cache_key} "
                f"holds {num_blocks} blocks. These come from different "
                "objects: the table from the attention metadata, the cache "
                "from the layer. An id at or past the end most likely means "
                "the table belongs to another cache group -- V4.1 has several, "
                "with different block counts.")
        used = kwargs.get(used_key)
        if used is None or used.numel() == 0:
            raise ValueError(
                f"{used_key} is missing or empty, so the capacity check did "
                f"not run against {cache_key}.")
        capacity = num_blocks * block_size
        # -1 under capture always passes `longest > capacity`.
        longest = _device_scalar(lambda used=used: used.max(), -1)
        if longest > capacity:
            raise ValueError(
                f"{used_key}'s longest sequence is {longest} tokens and "
                f"{cache_key} holds {capacity} ({num_blocks} blocks of "
                f"{block_size}). Lengths and cache are sized by different code "
                "paths; a length past the end reads another allocation.")


#: The descriptor blob the ``*_metadata`` operator produces, in int32 words.
#: Same number as ``dsa_v1.DSA_METADATA_BUFFER_SIZE``, restated rather than
#: imported so the check cannot pass by construction.
DESCRIPTOR_WORDS = 1024


def _check_descriptor_is_populated(kwargs: dict[str, Any],
                                   q: torch.Tensor) -> None:
    """The coarsest possible check on ``metadata``.

    The descriptor is a 1024-word blob carrying the per-core split of query
    tokens. Nothing here can tell whether its *contents* match this call: a
    descriptor built for another configuration is well-formed, and a blob
    that belongs to someone else passes every check in this function. "Not
    all zeros" catches only the case where the metadata operator never ran.

    Zero query tokens legitimately produce an all-zero split, so that case is
    reported as *not checked* rather than passed silently.
    """
    metadata = kwargs.get("metadata")
    _check(metadata is not None,
           "metadata must be provided",
           "the operator takes its tiling from the descriptor; without one "
           "the tiling key comes out 0 and every core faults at the same pc")

    numel = getattr(metadata, "numel", None)
    if numel is None:
        # A pointer or an opaque handle: nothing to inspect.
        _log_descriptor_not_checked("metadata is not a tensor")
        return

    _check(numel() == DESCRIPTOR_WORDS,
           f"metadata must be {DESCRIPTOR_WORDS} int32 words",
           f"got {numel()}; a short blob is read past its end by the kernel")

    num_tokens = int(q.shape[0]) if q.ndim >= 1 else 0
    if num_tokens == 0:
        _log_descriptor_not_checked(
            "the batch has no query tokens, so an all-zero split is what an "
            "empty batch legitimately produces")
        return

    _check(_device_predicate(lambda: metadata.any()),
           "metadata must be a populated descriptor",
           f"all {DESCRIPTOR_WORDS} words are zero while this call has "
           f"{num_tokens} query tokens, so the metadata operator never wrote "
           "into this buffer. Note the converse does not hold: a non-zero "
           "blob may still have been built for another configuration.")


_LOGGED_DESCRIPTOR_SKIPS: set[str] = set()


def _log_descriptor_not_checked(reason: str) -> None:
    """Say, once per reason, that the descriptor check was skipped -- a
    skipped check and a passed one otherwise look the same."""
    if reason in _LOGGED_DESCRIPTOR_SKIPS:
        return
    _LOGGED_DESCRIPTOR_SKIPS.add(reason)
    from vllm.logger import logger

    logger.info("DSV4.1 sparse_flash_mla: descriptor contents NOT checked -- %s",
                reason)


def validate_sparse_flash_mla_args(kwargs: dict[str, Any], q: torch.Tensor,
                                   num_requests: int) -> dict[str, Any]:
    """Check one assembled call. Returns the kwargs so it can wrap a build."""
    layout_kv = kwargs.get("layout_kv")
    layout_q = kwargs.get("layout_q")
    cmp_ratio = kwargs.get("cmp_ratio")
    has_cmp_kv = kwargs.get("cmp_kv") is not None

    _check(layout_kv in LAYOUTS_KV,
           f"layout_kv must be in [{', '.join(LAYOUTS_KV)}]",
           f"got {layout_kv!r}; PA_ND belongs to a different operator")

    _check(q.shape[-1] == HEAD_DIM,
           "head_dim must be 512, and q's head dimension must also be 512",
           f"q's last dimension is {q.shape[-1]}; note the operator has no "
           "rope_head_dim parameter, so the rope tail is part of the 512")

    _check(kwargs.get("sinks") is not None, "sinks must be provided",
           "attn_sink is [64] fp32 and cannot be skipped")

    if layout_q == "TND":
        cu_seqlens_q = kwargs.get("cu_seqlens_q")
        _check(cu_seqlens_q is not None,
               "layout_q=TND requires cu_seqlens_q, cumulative, length T+1")
        _check(cu_seqlens_q.ndim == 1 and cu_seqlens_q.numel() == num_requests + 1,
               "layout_q=TND requires cu_seqlens_q, cumulative, length T+1",
               f"got shape {tuple(cu_seqlens_q.shape)} for {num_requests} requests")
        # The detail does a host read-back, so it is built only when raising.
        if not _device_predicate(
                lambda: torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1])):
            _check(False,
                   "layout_q=TND requires cu_seqlens_q, cumulative, length T+1",
                   "the values are not non-decreasing, so they are lengths "
                   "rather than a cumulative sum"
                   + _first_decrease(cu_seqlens_q))

    if layout_kv != "TND":
        for name in ("cu_seqlens_ori_kv", "cu_seqlens_cmp_kv"):
            _check(kwargs.get(name) is None,
                   "cu_seqlens_*_kv are accepted only when layout_kv is TND",
                   f"{name} was supplied with layout_kv={layout_kv!r}; under a "
                   "paged layout the KV lengths travel in seqused_ori_kv")

    _check(cmp_ratio is not None and cmp_ratio > 0,
           "cmp_ratio must be greater than 0",
           f"got {cmp_ratio!r}; a layer with no compressed stream still has to "
           "report a positive ratio, so 0 is never a legal value here even "
           "though 0 is a legal configured compress_ratio")

    windowed = (kwargs.get("ori_win_left"), kwargs.get("ori_win_right")) != (-1, -1)
    if windowed:
        # The rule is about the ratio, not about the presence of a compressed
        # stream: ratio-1 layers carrying a compressed stream take the window.
        _check(cmp_ratio in WINDOW_CMP_RATIOS,
               "SWA mode requires cmp_ratio = 1 (CANN 9.2 also accepts 2)",
               f"window bounds {kwargs.get('ori_win_left')!r}/"
               f"{kwargs.get('ori_win_right')!r} are set but "
               f"cmp_ratio={cmp_ratio}; non-sliding-window modes require "
               "ori_win_left = ori_win_right = -1")

    # No "has_cmp_kv requires seqused_cmp_kv" or "requires cmp_residual_kv"
    # rule here: these kwargs are pre-adapter, and the sparse_flash_mla adapter
    # derives both from seqused_ori_kv and cmp_ratio when they are absent.
    # Only the shape of a residual that *was* supplied is the caller's own.
    if has_cmp_kv:
        residual = kwargs.get("cmp_residual_kv")
        if residual is not None:
            # Despite the name it is not KV data: the operator rejects anything
            # but a 1-D tensor ("incorrect shape dim 3. It should be 1").
            _check(residual.ndim == 1,
                   "cmp_residual_kv incorrect shape dim. It should be 1",
                   f"got {residual.ndim} dims, shape "
                   f"{tuple(residual.shape)}; despite the name this is one "
                   "value per request, not KV data")
            _check(residual.numel() == num_requests,
                   "cmp_residual_kv incorrect shape dim. It should be 1",
                   f"got {residual.numel()} values for {num_requests} requests")

    seqused_cmp_kv = kwargs.get("seqused_cmp_kv")
    if seqused_cmp_kv is not None:
        # One value per request. A cumulative array has num_requests + 1, and
        # that count is the only cheap thing that separates the two: the
        # operator would read a cumulative array as lengths without complaint.
        _check(seqused_cmp_kv.numel() == num_requests,
               "seqused_cmp_kv must hold one length per request",
               f"got {seqused_cmp_kv.numel()} values for {num_requests} "
               f"requests"
               + (" -- that is num_requests + 1, the length of a *cumulative* "
                  "sequence-length array. cu_cmp_seqlen_list is that array; "
                  "this parameter wants seq_lens // cmp_ratio."
                  if seqused_cmp_kv.numel() == num_requests + 1 else ""))
        # And the rank, which the count alone does not pin: a [1, 1] tensor
        # has one element too.
        _check(seqused_cmp_kv.ndim == 1,
               "seqused_cmp_kv must be a 1-D per-request array",
               f"got {seqused_cmp_kv.ndim} dims, shape "
               f"{tuple(seqused_cmp_kv.shape)}; the value comes from the "
               "compressed group's req metadata (seqused_cmp_kv, computed as "
               "seq_lens // cmp_ratio), not from cu_cmp_seqlen_list")

    expected_rank = LAYOUT_KV_RANK.get(layout_kv)
    if expected_rank is not None:
        for name in ("ori_kv", "cmp_kv"):
            cache = kwargs.get(name)
            if cache is None:
                continue
            _check(cache.ndim == expected_rank,
                   f"{name} must have {expected_rank} axes under "
                   f"layout_kv={layout_kv}",
                   f"got {cache.ndim} {tuple(cache.shape)}; the unit KV-head "
                   "axis is the one usually missing -- pass the cache through "
                   "kv_with_layout_rank rather than adding it by hand")

    _check_descriptor_is_populated(kwargs, q)
    _check_block_table_addresses_this_cache(kwargs)
    _check_modes_are_not_both_requested(kwargs)

    return kwargs


#: How many axes each KV layout describes. The operator reads ``ori_kv``'s
#: stride vector against this and rejects a length mismatch by name
#: ("Ori_kv strideVec size[3] not match layout_kv expect len[4]"), so the rank
#: is part of the layout, not a property of the cache. Keys must be names
#: LAYOUTS_KV accepts.
LAYOUT_KV_RANK = {
    "PA_BBND": 4,   # blocks, block, heads, dim
    "BSND": 4,      # batch, seq, heads, dim
    "TND": 3,       # tokens, heads, dim
}


def _check_modes_are_not_both_requested(kwargs: dict[str, Any]) -> None:
    """Refuse a call that asks the operator for two modes at once.

    ``ori_sparse_indices`` *selects* SWA ("ori_sparse_indices is only
    supported in SWA mode"), and SWA then requires cmp_mask_mode 0. Sending
    it alongside a compressed stream at mode 3 asks for SWA and for
    compressed-causal simultaneously, and the operator rejects the pair::

        EZ0027: Parameters ori_mask_mode and cmp_mask_mode of SparseFlashMla
                have incorrect values 0, 3.
                Reason: Sparse ori_kv mode requires ori_mask_mode=
                cmp_mask_mode=0.

    Raising rather than dropping the indices: dropping them would produce a
    call the operator accepts while the original stream is no longer
    selected the way the caller asked.
    """
    if kwargs.get("ori_sparse_indices") is None:
        return
    cmp_mask_mode = kwargs.get("cmp_mask_mode", NO_MASK_MODE)
    if cmp_mask_mode == NO_MASK_MODE:
        return
    _check(
        False,
        "Sparse ori_kv mode requires ori_mask_mode=cmp_mask_mode=0 (EZ0027); "
        "and separately, ori_sparse_indices is only supported in SWA mode, "
        "which requires cmp_ratio=1",
        f"this call supplies ori_sparse_indices "
        f"{tuple(kwargs['ori_sparse_indices'].shape)} together with "
        f"cmp_mask_mode={cmp_mask_mode} and cmp_ratio="
        f"{kwargs.get('cmp_ratio')}. Those select two different modes. "
        "Text-only requests build no vision indices (mm_ranges_have_spans), "
        "so look at the producer of ori_sparse_indices "
        "(select_ori_sparse_indices).")


def kv_with_layout_rank(cache, layout_kv: str):
    """Give the operator a KV cache at the rank ``layout_kv`` describes.

    Every KV cache argument of this call goes through here. The head axis is
    dropped upstream -- upstream's own V4.1 consumers read the cache as three
    dimensions -- while the operator wants four under a paged layout.

    Dispatched on the layout rather than unconditionally restored: ``TND`` is
    three axes, and the operator only checks the stride vector's *length*, so
    an unconditional ``unsqueeze`` would be accepted with the axes meaning
    something else.
    """
    expected = LAYOUT_KV_RANK.get(layout_kv)
    if expected is None:
        raise ValueError(
            f"no KV rank recorded for layout_kv={layout_kv!r}; known layouts "
            f"are {sorted(LAYOUT_KV_RANK)}. Passing the cache through "
            "unchanged would let the operator check the stride length against "
            "a layout nothing here understands.")
    if cache is None or cache.ndim == expected:
        return cache
    if cache.ndim == expected - 1 and expected == 4:
        # The axis bind_value dropped: num_kv_heads, at index -2.
        return cache.unsqueeze(-2)
    raise ValueError(
        f"ori_kv is {cache.ndim}-dimensional ({tuple(cache.shape)}) and "
        f"layout_kv={layout_kv!r} describes {expected} axes. Only the single "
        "dropped head axis is restored here; any other difference means the "
        "cache is not the layout this plan selected.")


def build_sparse_flash_mla_kwargs(
    *,
    q: torch.Tensor,
    num_requests: int,
    swa_kv_cache: torch.Tensor,
    swa_block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    metadata: Any,
    softmax_scale: float,
    cmp_ratio: int,
    layout_kv: str,
    attends_compressed: bool,
    ori_win_left: int,
    ori_win_right: int,
    cmp_kv: torch.Tensor | None = None,
    cmp_block_table: torch.Tensor | None = None,
    seqused_cmp_kv: torch.Tensor | None = None,
    cmp_residual_kv: torch.Tensor | None = None,
    # Logical token positions within each request, before ori_block_table lookup.
    ori_sparse_indices: torch.Tensor | None = None,
    #: Valid entries per row of ori_sparse_indices. Required whenever indices
    #: are supplied; see ori_sparse_selection for why the default is unsafe.
    ori_topk_length: torch.Tensor | None = None,
    cmp_sparse_indices: torch.Tensor | None = None,
    #: Valid entries per row of cmp_sparse_indices. Required whenever the
    #: compressed indices are supplied, for the reason ori_topk_length is.
    cmp_topk_length: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Assemble one call, mirroring ``DSAImpl._forward_attention``.

    ``attends_compressed`` is passed in rather than inferred from
    ``cmp_ratio``: V4.0 reads ratio <= 1 as "no compressed stream", while V4.1
    gives layers 20..39 ratio 1 to mean "compressed, pooling disabled".

    The window bounds are sent only in sliding-window mode (no explicit
    index list) and forced to -1 otherwise, because "non-sliding-window modes
    require ori_win_left = ori_win_right = -1".
    """
    plain_window = is_plain_sliding_window(ori_sparse_indices)
    kwargs: dict[str, Any] = dict(
        ori_kv=swa_kv_cache,
        ori_block_table=swa_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode_for(ori_sparse_indices),
        ori_win_left=ori_win_left if plain_window else -1,
        ori_win_right=ori_win_right if plain_window else -1,
        layout_q="TND",
        layout_kv=layout_kv,
    )

    if ori_sparse_indices is not None:
        if ori_topk_length is None:
            raise ValueError(
                "ori_sparse_indices was supplied without ori_topk_length. "
                "The operator then takes the whole row as valid -- "
                f"{tuple(ori_sparse_indices.shape)} here -- and gathers at "
                "the -1 padding, which faults every core at one address. "
                "Omitting the length is not a safe default; it is the most "
                "dangerous one.")
        kwargs["ori_sparse_indices"] = ori_sparse_indices
        kwargs["ori_topk_length"] = ori_topk_length

    if attends_compressed:
        if cmp_kv is None:
            raise ValueError(
                "attends_compressed is set but no cmp_kv was supplied. The "
                "layer would then be configured for a compressed stream "
                "while attending only its window, which the operator has no "
                "reason to reject.")
        kwargs.update(
            cmp_kv=cmp_kv,
            cmp_block_table=cmp_block_table,
            cmp_mask_mode=CMP_MASK_MODE,
        )
        if seqused_cmp_kv is not None:
            kwargs["seqused_cmp_kv"] = seqused_cmp_kv
        if cmp_residual_kv is not None:
            kwargs["cmp_residual_kv"] = cmp_residual_kv
        if cmp_sparse_indices is not None:
            # [T, KV_N, K2] under layout_q=TND. Checked here because the
            # operator's own refusal is truncated and does not say what it
            # wanted.
            _check(cmp_sparse_indices.ndim == 3,
                   "cmp_sparse_indices must be [T, KV_N, K2] under layout_q=TND",
                   f"got {cmp_sparse_indices.ndim} dims "
                   f"{tuple(cmp_sparse_indices.shape)}; the KV-head axis is "
                   "the one usually missing, since the buffer upstream keeps "
                   "is [tokens, index_topk]")
            _check(cmp_sparse_indices.shape[0] == q.shape[0],
                   "cmp_sparse_indices must have one row per query token",
                   f"got {cmp_sparse_indices.shape[0]} rows for "
                   f"{q.shape[0]} query tokens")
            # The same pairing the ori side enforces: the selection buffer is
            # -1 padded, and omitting the length declares the padding valid.
            if cmp_topk_length is None:
                raise ValueError(
                    "cmp_sparse_indices was supplied without cmp_topk_length. "
                    "The operator then takes the whole row as valid -- "
                    f"{tuple(cmp_sparse_indices.shape)} here -- and gathers "
                    "at the -1 padding the indexer wrote. Omitting the length "
                    "is not a safe default; it is the most dangerous one.")
            kwargs["cmp_sparse_indices"] = cmp_sparse_indices
            kwargs["cmp_topk_length"] = cmp_topk_length

    return validate_sparse_flash_mla_args(kwargs, q, num_requests)
