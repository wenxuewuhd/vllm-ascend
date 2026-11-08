# SPDX-License-Identifier: Apache-2.0
"""Widen a half-width cos/sin row to the width the rotary operator takes.

``inplace_partial_rotary_mul`` wants ``cos`` as wide as the slice it rotates
(``cos.size(-1) == end - start``), while upstream's ``cos_sin_cache[pos]`` holds
``rope_head_dim/2`` cosines followed by the same number of sines -- one of each
per interleaved pair. Getting from one to the other is a widening, and there
are two widenings that produce the right shape:

    A  [c0, c0, c1, c1, ...]      each value to its pair's two channels
    B  [c0, c1, ..., c0, c1, ...] the half repeated

Both are ``rope_head_dim`` wide, so nothing downstream can tell them apart. B
rotates by the wrong angle on every other pair, which is a regular,
plausible-looking error and not a detectable one.

A is correct, and not because it is the more natural reading: it is what the
Ascend path that already drives this operator does.
``ops/rope_dsv4.py:218`` builds its table as
``freqs.cos().repeat_interleave(2, dim=-1)``, and that table is what V4.0 feeds
to the same operator today. The choice is copied from a working consumer rather
than derived from the docstring of the thing being called.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict

import torch


def widen_interleaved(half: torch.Tensor) -> torch.Tensor:
    """``[..., n]`` of per-pair values to ``[..., 2n]`` of per-channel values.

    Channel ``i`` belongs to pair ``i // 2``, which is what makes this
    ``repeat_interleave`` and not ``repeat``.
    """
    # Keep the pair order using copy operations; CANN RepeatInterleave times out
    # on this image for the 4096 x 32 FP32 prefill table.
    return torch.stack((half, half), dim=-1).flatten(-2)


def cos_sin_for_slice(cos_sin_cache: torch.Tensor, positions: torch.Tensor,
                      rope_head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The operator's ``(cos, sin)`` for a batch of positions.

    ``cos_sin_cache`` is upstream's: ``[max_pos, rope_head_dim]``, the cosines
    in the first half and the sines in the second. Returned already widened, so
    the caller never sees the half-width form and cannot widen it twice.
    """
    if cos_sin_cache.shape[-1] != rope_head_dim:
        raise ValueError(
            f"cos_sin_cache is {cos_sin_cache.shape[-1]} wide and the rope "
            f"head dim is {rope_head_dim}. The cache holds half a rotation's "
            "cosines followed by its sines, so these must be equal; a mismatch "
            "means one of them is the *pair* count rather than the channel "
            "count, and the halves would be split in the wrong place.")
    hit = _cached_slice(cos_sin_cache, positions, rope_head_dim)
    if hit is not None:
        return hit[0], hit[1]
    half = rope_head_dim // 2
    row = cos_sin_cache[positions].to(torch.float32)
    cos = widen_interleaved(row[..., :half])
    sin = widen_interleaved(row[..., half:])
    _remember_slice(cos_sin_cache, positions, rope_head_dim, cos, sin)
    return cos, sin


def neg_sin_for_slice(cos_sin_cache: torch.Tensor, positions: torch.Tensor,
                      rope_head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, -sin)`` -- what the **inverse** rotation takes.

    Negating here rather than at the call site is what lets the negation be
    cached with the pair: the inverse path used to compute ``-sin`` fresh on
    every layer, 40 extra ``Neg`` kernels a step for a value that is the same
    every time.
    """
    cos, sin = cos_sin_for_slice(cos_sin_cache, positions, rope_head_dim)
    entry = _lookup(cos_sin_cache, positions, rope_head_dim)
    # ``entry.cos is cos`` and not just "an entry exists": it proves the entry
    # is the one that produced *this* ``cos`` (and so this ``sin``), rather
    # than one that happened to be looked up under the same key afterwards.
    if entry is not None and entry.cos is cos:
        if entry.neg_sin is None:
            entry.neg_sin = -sin
        return cos, entry.neg_sin
    return cos, -sin


#: How many distinct ``(table, positions)`` pairs one forward pass may keep.
#:
#: 🔴 Why this used to be ONE entry, and why it can now be more. With one
#: caller (the main KV RoPE: forward in ``qkv_prolog``, inverse in ``rope``)
#: every lookup in a pass wanted the same pair, so one entry was enough. The
#: indexer's q RoPE (its own ``index_q_cos_sin_cache``) and k RoPE (on a
#: ``compressed_pos`` rebuilt every layer, ``indexer_k_store.py:339``) are
#: moving onto this function too; with one entry they would evict the main KV
#: pair every layer and give back the 0.86 ms it saved. Two things made more
#: entries unsafe before and are what this version adds:
#:
#:   1. the key carried ``id(forward_context)``. An id is an address, and once
#:      the pass's context is freed the next pass's context can be allocated
#:      at the same address -- same id, same persistent ``positions`` address,
#:      last step's angles. The whole table is now dropped when the pass
#:      changes, with the pass held by weak reference (``is``, not ``id``).
#:   2. the key held ``data_ptr()`` but not the tensor. A temporary such as
#:      ``compressed_pos`` is freed after its layer, and the caching allocator
#:      hands the same address to the next layer's temporary -- which at the
#:      ratio-2 -> ratio-1 boundary (layers 19/20 of V4.1-Flash's
#:      ``compress_ratios``) has the same shape and *different values*. With
#:      one entry, the main KV call between the two indexer calls evicted it
#:      first, so this never surfaced: **the eviction was the guard**. Each
#:      entry now holds its input tensors, so an address that is in the table
#:      cannot be recycled while it is there.
#:
#: The number. One pass of V4.1-Flash asks for at most five pairs that are
#: reused: the main KV pair on two tables (``rope_params_for_layer`` gives
#: ratio-0 layers a different table from the rest, and upstream's ``get_rope``
#: shares one instance per parameter set), the indexer q pair, and the
#: indexer k pair for each of the two indexer ratios once ``compressed_pos``
#: is built once per ratio instead of per layer. Eight leaves room for keys
#: that are never reused (a per-layer ``compressed_pos`` inserts one per
#: layer) to churn through without pushing out a live one. Worst-case memory
#: is 8 x (cos + sin + -sin) = 8 x 3 x T x 64 x 4 bytes, ~100 MB at a
#: 16k-token prefill and nothing at decode sizes, and it is released when the
#: next pass starts.
_CAPACITY = 8


class _Entry:
    """One remembered pair.

    ``cos_sin_cache`` and ``positions`` are kept **only** to hold their memory:
    while this entry exists, nothing else can be allocated at the addresses its
    key names, so a key match cannot be a recycled address. They are never
    read.
    """

    __slots__ = ("cos_sin_cache", "positions", "cos", "sin", "neg_sin")

    def __init__(self, cos_sin_cache, positions, cos, sin):
        self.cos_sin_cache = cos_sin_cache
        self.positions = positions
        self.cos = cos
        self.sin = sin
        self.neg_sin: torch.Tensor | None = None


#: key -> _Entry, least recently used first. LRU rather than FIFO because the
#: pairs worth keeping are the ones touched every layer: a fresh key evicts
#: the pair that has gone longest unused, which is a dead temporary or the
#: ratio-0 table after layer 1, never the pair the next layer is about to ask
#: for. Not ``functools.lru_cache``: it would key on the tensors' hash, which
#: is object identity -- a new view of the same buffer every layer would
#: miss, and a recycled object would hit.
_CACHE: OrderedDict[tuple, _Entry] = OrderedDict()

#: Weak reference to the forward pass ``_CACHE`` belongs to.
_CACHE_PASS: list = [None]


def _forward_token() -> object | None:
    """The object identifying the forward pass now running, or None.

    🔴 This -- not the tensor addresses -- is what makes the cache safe. The
    runner reuses one persistent ``positions`` buffer, so its ``data_ptr()``
    is the **same address with different contents** on the next step; keying
    on addresses alone would serve one step's angles to the next. Outside a
    forward pass (unit tests, profile runs) there is no token and nothing is
    cached, which is the conservative direction.
    """
    try:
        from vllm.forward_context import get_forward_context

        return get_forward_context()
    except Exception:  # noqa: BLE001 -- no context is a valid answer
        return None


def _table_for_this_pass() -> OrderedDict[tuple, _Entry] | None:
    """``_CACHE``, emptied first if it was filled by another pass. None: don't cache.

    Compared with ``is`` through a weak reference, not by ``id``: a dead
    pass's weakref returns None and a live one returns that object, so neither
    can compare equal to a new pass that reuses its address. Weak, so the
    cache does not keep a finished pass's attention metadata alive.
    """
    token = _forward_token()
    if token is None:
        return None
    ref = _CACHE_PASS[0]
    if ref is None or ref() is not token:
        _CACHE.clear()
        try:
            _CACHE_PASS[0] = weakref.ref(token)
        except TypeError:
            # Not weak-referenceable: no way to tell this pass from the next
            # one without keeping it alive, so do not cache at all.
            _CACHE_PASS[0] = None
            return None
    return _CACHE


def _slice_key(cos_sin_cache: torch.Tensor, positions: torch.Tensor,
               rope_head_dim: int) -> tuple:
    """Addresses, not identities: callers pass a fresh *view* of the same
    persistent buffer every layer (``positions[:num_tokens]``), which is a new
    tensor object each time and must still hit. Shape, strides and dtype make
    two views of one address that read different elements different keys."""
    return (cos_sin_cache.device, cos_sin_cache.data_ptr(),
            tuple(cos_sin_cache.shape), cos_sin_cache.dtype,
            positions.device, positions.data_ptr(), tuple(positions.shape),
            positions.stride(), positions.dtype, rope_head_dim)


def _lookup(cos_sin_cache, positions, rope_head_dim) -> _Entry | None:
    """The entry computed earlier **in this same forward pass**, if any.

    The main KV path asks for one ``(cache, positions)`` pair per layer and
    rotates with it twice: the forward RoPE in ``qkv_prolog`` and the inverse
    in ``rope`` each call this once per layer, on the same table and the same
    positions, 80 calls a step. (Two tables across the 40 layers, see
    ``_CAPACITY``; the indexer adds its own pairs.) Measured 2026-09-22 (§6.9.15): the repeated widening
    was 312 kernels and **0.86 ms of a 28.8 ms decode step**, recomputing an
    identical answer.

    ⚠ Contract: within one pass, a hit trusts that nothing wrote the inputs in
    place since the entry was made -- the address is live (the entry holds
    it), but its bytes are not re-read. True of every caller: positions are
    filled before the pass and tables at load time.

    ⚠ Under FULL graph capture this Python runs once and the reused tensor
    becomes a graph-internal value, which is the same saving at replay.
    """
    table = _table_for_this_pass()
    if table is None:
        return None
    key = _slice_key(cos_sin_cache, positions, rope_head_dim)
    entry = table.get(key)
    if entry is not None:
        table.move_to_end(key)
    return entry


def _cached_slice(cos_sin_cache, positions, rope_head_dim):
    entry = _lookup(cos_sin_cache, positions, rope_head_dim)
    if entry is None:
        return None
    return entry.cos, entry.sin


def _remember_slice(cos_sin_cache, positions, rope_head_dim, cos, sin) -> None:
    table = _table_for_this_pass()
    if table is None:
        return
    key = _slice_key(cos_sin_cache, positions, rope_head_dim)
    table[key] = _Entry(cos_sin_cache, positions, cos, sin)
    table.move_to_end(key)
    while len(table) > _CAPACITY:
        table.popitem(last=False)
