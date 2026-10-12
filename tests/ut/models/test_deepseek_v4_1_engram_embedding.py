# SPDX-License-Identifier: Apache-2.0
"""The engram table must not be allocated, and that must be enforced.

The first real launch died in upstream's ``ParallelEngramEmbedding.__init__``
with ``rtsMallocHost ... out of memory``. It was trying the pinned-host path --
the one that took an NPU down -- and what stopped it was the host being too
small, not any decision of ours. On a larger machine it would have succeeded.

So the tests that matter here are not "does it fail on this host". They are
"does it fail for the right reason, including when the allocation would have
worked".
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.engram_embedding import (
    AscendStagedEngramEmbedding, tables_not_allocated)


def test_allocations_are_diverted_to_meta():
    with tables_not_allocated():
        t = torch.empty(1024, 256, dtype=torch.float8_e4m3fn)
    assert t.device.type == "meta"
    assert t.shape == (1024, 256) and t.dtype == torch.float8_e4m3fn


def test_pin_memory_is_dropped_not_honoured():
    """The specific argument that reaches aclrtMallocHostWithCfg."""
    with tables_not_allocated():
        t = torch.empty(8, 8, dtype=torch.uint8, device="cpu", pin_memory=True)
    assert t.device.type == "meta"
    assert not t.is_pinned()


def test_torch_empty_is_restored_afterwards():
    before = torch.empty
    with tables_not_allocated():
        pass
    assert torch.empty is before
    assert torch.empty(4).device.type != "meta"


def test_it_is_restored_even_if_construction_raises():
    before = torch.empty
    with pytest.raises(ValueError):
        with tables_not_allocated():
            raise ValueError("constructor blew up")
    assert torch.empty is before


# -- the guard, which must not depend on running out of memory ---------------


class _Fake(AscendStagedEngramEmbedding):
    """Bypasses upstream's __init__; only the guard is under test."""

    def __init__(self, weight, scale):
        self.weight = weight
        self.weight_scale_inv = scale


def test_meta_parameters_are_accepted():
    f = _Fake(torch.empty(1024, 256, dtype=torch.float8_e4m3fn, device="meta"),
              torch.empty(1024, 8, dtype=torch.uint8, device="meta"))
    f._assert_nothing_was_allocated()


def test_the_guard_fires_on_an_allocation_that_SUCCEEDED():
    """The whole point.

    Last time the banned path was stopped by OOM. A guard that only triggers
    when the allocation fails is not a guard -- it is the machine's memory
    limit wearing one. This allocation is tiny and succeeds, and the guard must
    still refuse.
    """
    f = _Fake(torch.empty(16, 8, dtype=torch.float8_e4m3fn),   # real, succeeds
              torch.empty(16, 1, dtype=torch.uint8, device="meta"))
    with pytest.raises(RuntimeError) as exc:
        f._assert_nothing_was_allocated()
    msg = str(exc.value)
    assert "weight" in msg
    # and it has to say that OOM is not the guard, or the next person reads
    # "it failed on this host" as "it is prevented"
    assert "running out of memory here is not the guard" in msg


def test_the_scale_parameter_is_checked_too():
    f = _Fake(torch.empty(16, 8, dtype=torch.float8_e4m3fn, device="meta"),
              torch.empty(16, 1, dtype=torch.uint8))          # real
    with pytest.raises(RuntimeError, match="weight_scale_inv"):
        f._assert_nothing_was_allocated()


def test_storage_is_a_dead_end():
    """_storage() is upstream's UVA / host-registration entry point."""
    f = _Fake(torch.empty(1, 1, device="meta"), torch.empty(1, 1, device="meta"))
    with pytest.raises(RuntimeError) as exc:
        f._storage()
    msg = str(exc.value)
    assert "must not be reached" in msg
    assert "read-only mapping" in msg


def test_unbound_tables_say_how_to_bind_them():
    f = _Fake(torch.empty(1, 1, device="meta"), torch.empty(1, 1, device="meta"))
    f._layer_id = 14
    with pytest.raises(RuntimeError, match="bind_staged_tables"):
        f.staged_tables()


def test_the_mixin_precedes_upstream_so_its_init_wins():
    from vllm_ascend.models.deepseek_v4_1.caches import AscendEngramEmbedding

    mro = [c.__name__ for c in AscendEngramEmbedding.__mro__[:3]]
    assert mro == ["AscendEngramEmbedding", "AscendStagedEngramEmbedding",
                   "ParallelEngramEmbedding"], (
        "upstream's __init__ must not win -- it allocates the shard before "
        f"anything else gets a say; got {mro}")
    assert "__init__" in vars(AscendStagedEngramEmbedding)


# -- the lookup path, up to the kernel seam ----------------------------------

from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout

LAYOUT = EngramLayout(n_hash_cols=24, dim=256, quant_block=32,
                      layer_ids=(1,), num_embeddings=(512,))


class _Lookup(AscendStagedEngramEmbedding):
    """Enough of upstream's surface to exercise lookup(); no NPU involved."""

    def __init__(self, tables, part=24, total=24, start=0):
        import numpy as np
        self.weight = torch.empty(1, 1, device="meta")
        self.weight_scale_inv = torch.empty(1, 1, device="meta")
        self.part_n_hash_cols, self.n_hash_cols = part, total
        self.head_start = start
        self._tables = tables
        self.dequantised_rows = None
        self._np = np

    def staged_tables(self):
        return self._tables

    def _dequantize_staged(self, rows, out):     # the seam, stubbed
        self.dequantised_rows = rows


def _tables(rows=512):
    import numpy as np
    w = np.zeros((rows, 256), dtype=np.uint8)
    s = np.zeros((rows, 8), dtype=np.uint8)
    w[:, 0] = np.arange(rows) % 256
    s[:, 0] = np.arange(rows) % 251
    return w, s


def test_lookup_stages_the_rows_each_token_asked_for():
    import numpy as np

    tables = _tables()
    layer = _Lookup(tables)
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")

    rng = np.random.default_rng(11)
    ids = rng.integers(0, 512, size=(5, 24)).astype(np.int32)
    layer.lookup(torch.from_numpy(ids), out=None)

    assert layer.dequantised_rows == 5 * 24
    for t in range(5):
        for c in range(24):
            r = LAYOUT.row_index(t, c, LAYOUT.n_hash_cols)
            assert layer._dev_weight[r, 0].item() == tables[0][ids[t, c], 0]
            assert layer._dev_scale[r, 0].item() == tables[1][ids[t, c], 0]


def test_flattening_indices_matches_upstreams_row_numbering():
    """Upstream numbers rows token * LOCAL_HEADS + head and reads
    ``indices + token * TOTAL_HEADS + head``. With the columns undivided that
    is row-major order, which is what lookup() relies on -- so this pins the
    two numbering schemes against each other rather than trusting they agree.
    """
    import numpy as np

    ids = np.arange(5 * 24, dtype=np.int32).reshape(5, 24)
    flat = ids.reshape(-1)
    for t in range(5):
        for c in range(24):
            assert flat[LAYOUT.row_index(t, c, LAYOUT.n_hash_cols)] == ids[t, c]


def test_an_empty_batch_does_nothing():
    layer = _Lookup(_tables())
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
    layer.lookup(torch.zeros(0, 24, dtype=torch.int32), out=None)
    assert layer.dequantised_rows is None


def test_more_tokens_than_the_buffers_were_sized_for_is_refused():
    layer = _Lookup(_tables())
    layer.allocate_staging(LAYOUT, max_tokens=4, device="cpu")
    with pytest.raises(ValueError, match="staging buffers were sized for"):
        layer.lookup(torch.zeros(5, 24, dtype=torch.int32), out=None)


def test_lookup_without_staging_says_what_to_call():
    layer = _Lookup(_tables())
    with pytest.raises(RuntimeError, match="allocate_staging"):
        layer.lookup(torch.zeros(2, 24, dtype=torch.int32), out=None)


def test_lookup_gathers_only_this_ranks_columns():
    """TP4's last rank: given all 24 columns it must stage exactly columns
    18..23, token-major with stride 6. The ids are GLOBAL table rows and the
    whole table is mapped on every rank, so no rebase is involved."""
    import numpy as np

    tables = _tables()
    layer = _Lookup(tables, part=6, total=24, start=18)
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")

    rng = np.random.default_rng(23)
    ids = rng.integers(0, 512, size=(5, 24)).astype(np.int32)
    layer.lookup(torch.from_numpy(ids), out=None)

    assert layer.dequantised_rows == 5 * 6
    for t in range(5):
        for c in range(6):
            r = LAYOUT.row_index(t, c, 6)
            want = ids[t, 18 + c]
            assert layer._dev_weight[r, 0].item() == tables[0][want, 0]
            assert layer._dev_scale[r, 0].item() == tables[1][want, 0]


def test_the_ranks_together_stage_exactly_what_tp1_stages():
    """The differential property that matters: reassembling the four TP4 ranks
    must reproduce the TP=1 gather exactly. A wrong head_start or a wrong
    stride yields valid row numbers for the wrong columns, which nothing
    downstream would notice -- this is what catches it."""
    import numpy as np

    tables = _tables()
    rng = np.random.default_rng(29)
    ids = rng.integers(0, 512, size=(5, 24)).astype(np.int32)

    ref = _Lookup(tables)
    ref.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
    ref.lookup(torch.from_numpy(ids), out=None)
    reference = {}
    for t in range(5):
        for c in range(24):
            reference[(t, c)] = ref._dev_weight[
                LAYOUT.row_index(t, c, 24), 0].item()

    seen = {}
    for rank in range(4):
        shard = _Lookup(tables, part=6, total=24, start=rank * 6)
        shard.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
        shard.lookup(torch.from_numpy(ids), out=None)
        assert shard.dequantised_rows == 5 * 6
        for t in range(5):
            for c in range(6):
                seen[(t, rank * 6 + c)] = shard._dev_weight[
                    LAYOUT.row_index(t, c, 6), 0].item()

    assert seen == reference


@pytest.mark.parametrize("part,total,start", [
    (6, 24, 20),     # window runs off the end of the model's columns
    (12, 8, 0),      # total disagrees with the layout
])
def test_lookup_refuses_a_malformed_column_shard(part, total, start):
    layer = _Lookup(_tables(), part=part, total=total, start=start)
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
    with pytest.raises(ValueError):
        layer.lookup(torch.zeros(2, 24, dtype=torch.int32), out=None)


def test_the_kernel_takes_the_agreed_interface():
    import inspect

    from vllm_ascend.ops.triton.engram_dequant_staging import launch

    params = inspect.signature(launch).parameters
    assert list(params)[:3] == ["staging_weight", "staging_scale", "out"]
    assert "dim" in params and "quant_block" in params


# -- the log line that became untrue -----------------------------------------


def test_the_stale_offload_line_is_dropped():
    """Upstream logs it from inside `if cpu_offload:` regardless of what the
    allocation did, so with the allocation diverted it reports an intention."""
    import logging

    from vllm_ascend.models.deepseek_v4_1.engram_embedding import (
        _STALE_OFFLOAD_LOG, _upstream_offload_line_suppressed)

    logger = logging.getLogger("vllm.models.deepseek_v4_1.common.engram")
    seen = []
    sink = logging.Handler()
    sink.emit = lambda record: seen.append(record.getMessage())
    logger.addHandler(sink)
    logger.setLevel(logging.INFO)
    try:
        with _upstream_offload_line_suppressed():
            logger.info("Engram table %s: 384006168 rows x 256, 94.42 GiB per rank",
                        _STALE_OFFLOAD_LOG)
            logger.info("Built engram token map (129280 -> 99092 ids)")
        logger.info("after the context, %s", _STALE_OFFLOAD_LOG)
    finally:
        logger.removeHandler(sink)

    # the untrue line, identified by its number rather than by position
    assert not any("94.42 GiB per rank" in m for m in seen), (
        f"the untrue line reached the log: {seen}")
    # the filter is surgical: other lines from the same logger survive
    assert any("Built engram token map" in m for m in seen)
    # and it is removed with the context, not left on the logger forever
    assert any(m.startswith("after the context") for m in seen), (
        "the filter must come off again, or it would hide a line that becomes "
        "true again if the design changes back")


def test_the_number_it_printed_was_specific_enough_to_believe():
    """Why suppressing it beats leaving it.

    94.42 GiB per rank over four ranks and two layers is 755.4 GiB, against a
    755 GiB host. A reader would conclude host memory had just been exhausted.
    The measured figure is VmPin = 0.
    """
    per_rank_per_layer = 94.42
    assert round(per_rank_per_layer * 4 * 2, 1) == 755.4


def test_the_replacement_line_says_what_actually_happened():
    import inspect

    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        AscendStagedEngramEmbedding

    source = inspect.getsource(AscendStagedEngramEmbedding.__init__)
    assert "0 bytes pinned per rank" in source
    assert "does not happen" in source


def test_staging_is_allocated_at_construction_not_on_first_lookup():
    """The ordering constraint, stated where it can go red.

    Graph replay reads the device addresses recorded at capture. A buffer
    allocated on the first lookup is a different address than the one the graph
    replays against, and that mismatch reports nothing -- it reads whatever is
    at the captured address. Construction is the only point guaranteed to
    precede capture, so this pins that the call is there.
    """
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import engram_embedding as mod

    init = inspect.getsource(mod.AscendStagedEngramEmbedding.__init__)
    assert "_allocate_staging_at_construction" in init, (
        "staging must be sized during __init__; anywhere later is after "
        "graph capture")
    body = inspect.getsource(mod.AscendStagedEngramEmbedding
                             ._allocate_staging_at_construction)
    assert "max_num_batched_tokens" in body
    assert "max_model_len" not in body.split('"""')[2], (
        "the bound is tokens per forward, not tokens per sequence")


def test_an_oversized_batch_is_refused_by_name():
    """What makes taking the tight bound safe.

    max_num_batched_tokens and max_model_len differ by a factor of two here.
    Choosing the smaller one is defensible only because exceeding it raises
    with both numbers in the message rather than writing past the buffer.
    """
    from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout

    f = _Lookup(_tables())
    f._layout = EngramLayout(n_hash_cols=24, dim=256, quant_block=32,
                             layer_ids=(1, 14), num_embeddings=(512, 512))
    f._max_tokens = 16
    with pytest.raises(ValueError, match="exceeds the 16"):
        f.lookup(torch.zeros((32, f.part_n_hash_cols), dtype=torch.int64),
                 torch.zeros((32, 8)))


def _layout_two_layers(rows=(512, 640)):
    from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout

    return EngramLayout(n_hash_cols=24, dim=256, quant_block=32,
                        layer_ids=(1, 14), num_embeddings=rows)


class _Identifiable(AscendStagedEngramEmbedding):
    def __init__(self, num_embeddings):
        self.num_embeddings = num_embeddings


def test_the_layer_identity_comes_from_the_row_count():
    """Upstream's constructor carries neither a prefix nor a layer id.

    The two engram layers have different vocabularies, so the row count names
    the layer. Pinned because the alternative -- guessing, or defaulting to the
    first -- binds the other layer's table and then maps, gathers and returns
    embeddings from it with no error anywhere.
    """
    layout = _layout_two_layers()
    assert _Identifiable(512)._own_layer_id(layout) == 1
    assert _Identifiable(640)._own_layer_id(layout) == 14


def test_an_ambiguous_row_count_refuses():
    """Two layers with equal vocabularies would make this a coin flip."""
    layout = _layout_two_layers(rows=(512, 512))
    with pytest.raises(RuntimeError, match="cannot tell which engram layer"):
        _Identifiable(512)._own_layer_id(layout)


def test_an_unmatched_row_count_refuses_and_shows_the_table():
    with pytest.raises(RuntimeError) as exc:
        _Identifiable(999)._own_layer_id(_layout_two_layers())
    msg = str(exc.value)
    assert "999 rows" in msg and "1: 512" in msg and "14: 640" in msg


def test_both_construction_time_hooks_are_called():
    """Every buffer and binding that must precede capture is wired into
    construction."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import engram_embedding as mod

    body = inspect.getsource(mod.AscendStagedEngramEmbedding
                             ._allocate_staging_at_construction)
    assert "allocate_staging(" in body
    assert "_allocate_pinned_for_capture(" in body
    assert "bind_staged_tables(" in body


def test_the_callers_per_token_buffer_becomes_gathered_rows():
    """Upstream hands [num_tokens, n_hash_cols, dim]; the kernel wants rows.

    Both are right about their own axis. EngramLayout.row_index is
    token * n_hash_cols + col, which is exactly the row-major order of the
    first two axes, so one buffer answers both descriptions.
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _flatten_out_rows

    out = torch.zeros(8, 24, 256)
    flat = _flatten_out_rows(out, 8 * 24)
    assert flat.shape == (192, 256)
    # The same storage, so the kernel's writes are the caller's reads.
    flat[5, 0] = 3.5
    assert out[0, 5, 0] == 3.5


def test_an_already_flat_buffer_is_passed_through():
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _flatten_out_rows

    out = torch.zeros(192, 256)
    assert _flatten_out_rows(out, 192) is out


def test_a_non_contiguous_buffer_is_refused_rather_than_copied():
    """reshape would return a copy, the kernel would fill the copy, and the
    caller would read back its previous contents with nothing raised."""
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _flatten_out_rows

    out = torch.zeros(8, 24, 512)[:, :, ::2]        # non-contiguous view
    assert not out.is_contiguous()
    with pytest.raises(ValueError, match="not contiguous"):
        _flatten_out_rows(out, 8 * 24)


def test_a_hash_column_count_mismatch_says_which_two_numbers():
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _flatten_out_rows

    with pytest.raises(ValueError, match="disagree about how many hash columns"):
        _flatten_out_rows(torch.zeros(8, 12, 256), 8 * 24)


def test_a_flat_buffer_of_the_wrong_length_is_refused():
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _flatten_out_rows

    with pytest.raises(ValueError, match="already in"):
        _flatten_out_rows(torch.zeros(8, 256), 192)


# ------------------------------------------------ the wall, named in its frame

def test_capture_without_pinned_buffers_is_refused():
    """The capture route needs the pinned buffers allocated before capture."""
    import unittest.mock as mock

    from vllm_ascend.models.deepseek_v4_1 import engram_embedding

    layer = _Lookup(_tables())
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
    with mock.patch.object(engram_embedding, "_stream_is_capturing",
                           return_value=True):
        with pytest.raises(RuntimeError, match="pinned capture buffers"):
            layer.lookup(torch.zeros(2, 24, dtype=torch.int32), out=None)


def test_an_unavailable_predicate_does_not_refuse():
    """The third state is permissive on purpose. A predicate that cannot
    be asked must not stop a server that works today; the cost of a wrong
    "no" is the CANN error we already get, the cost of a wrong "yes" is a
    server that will not start."""
    import unittest.mock as mock

    from vllm_ascend.models.deepseek_v4_1 import engram_embedding

    class _NoPredicate:
        pass

    with mock.patch.object(engram_embedding.torch, "npu", _NoPredicate()), \
            mock.patch.object(engram_embedding.torch, "cuda", _NoPredicate()):
        assert engram_embedding._stream_is_capturing() is False


def test_a_raising_predicate_does_not_refuse():
    import unittest.mock as mock

    from vllm_ascend.models.deepseek_v4_1 import engram_embedding

    class _Raises:
        @staticmethod
        def is_current_stream_capturing():
            raise RuntimeError("no graph subsystem here")

    with mock.patch.object(engram_embedding.torch, "npu", _Raises()):
        assert engram_embedding._stream_is_capturing() is False


def test_the_predicate_exists_on_this_platform():
    """The guard is worth nothing if nothing can ever answer it. Evaluated
    here rather than written down, because which module carries it has moved
    before."""
    import torch

    assert any(
        hasattr(getattr(torch, name, None), "is_current_stream_capturing")
        for name in ("npu", "cuda"))


def test_lookup_asks_before_it_copies():
    """The capture check has to come before the synchronising D2H, which
    capture rejects."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        AscendStagedEngramEmbedding

    body = inspect.getsource(AscendStagedEngramEmbedding.lookup)
    guard_at = body.index("_stream_is_capturing()")
    copy_at = body.index('.to("cpu"')
    assert guard_at < copy_at, (
        "the capture check sits after the synchronising copy, so the copy "
        "raises first and the message that explains it never runs")


# ------------------------------ the host staging buffer is reused every forward

def test_the_host_staging_buffer_is_one_buffer_reused(monkeypatch):
    """The property that makes the copy's timing matter.

    ``_host_weight`` is allocated once at construction and rewritten by every
    forward. If the H2D issued from it can still be in flight when the next
    forward's gather overwrites those bytes, the device sees a mix of two
    requests -- and deterministically so, because the request sequence fixes
    the overwrite sequence.
    """
    layer = _Lookup(_tables())
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
    first = layer._host_weight
    layer.lookup(torch.zeros(2, 24, dtype=torch.int32), out=None)
    assert layer._host_weight is first, (
        "a fresh buffer per forward would remove the hazard entirely; this "
        "test exists to say that is not what happens")
