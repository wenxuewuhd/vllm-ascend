# SPDX-License-Identifier: Apache-2.0
"""The staging convention, and the two things that can silently break it.

The CPU writes rows in an order the device recomputes from its program id. If
those disagree, both sides still handle well-formed bytes -- they are merely
paired with the wrong token -- so nothing raises. That is the failure this
module is defined once to prevent, and these are its guards.
"""

import json
import os
from types import SimpleNamespace

import pytest

from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout
from vllm_ascend.models.deepseek_v4_1.engram_staging import (DEFAULT_STAGE_DIR,
                                                             staged_file_name)

CHECKPOINT = "/workspace/models/DeepSeek-V4.1-Flash"


def _real_config():
    with open(os.path.join(CHECKPOINT, "config.json")) as handle:
        return SimpleNamespace(**json.load(handle)["text_config"])


@pytest.fixture
def layout():
    return EngramLayout(n_hash_cols=24, dim=256, quant_block=32,
                        layer_ids=(1, 14),
                        num_embeddings=(384006168, 384016682))


def test_derived_shapes(layout):
    assert layout.n_scales == 8
    assert layout.row_bytes == 264


def test_row_index_is_token_major(layout):
    """Token-major, not head-major. Both orderings are dense and total, which
    is exactly why swapping them produces no error."""
    assert layout.row_index(0, 0, 24) == 0
    assert layout.row_index(0, 23, 24) == 23
    assert layout.row_index(1, 0, 24) == 24
    assert layout.row_index(5, 7, 24) == 127
    # every (token, col) pair maps to a distinct row, with no gaps
    seen = {layout.row_index(t, c, 24) for t in range(4) for c in range(24)}
    assert seen == set(range(96))


def test_row_index_is_dense_over_the_local_columns(layout):
    """The staging buffer holds only this rank's columns, so the stride is the
    LOCAL count. Passing the model-wide 24 here under TP=4 is precisely the
    bug the old refuse-all-sharding guard existed to prevent: it would leave
    three quarters of the buffer unwritten and read ids for the wrong rows."""
    local = 6                      # 24 hash columns over 4 ranks
    assert layout.row_index(0, 0, local) == 0
    assert layout.row_index(0, 5, local) == 5
    assert layout.row_index(1, 0, local) == 6
    seen = {layout.row_index(t, c, local)
            for t in range(4) for c in range(local)}
    assert seen == set(range(4 * local))


def test_staging_sizes_match_the_measured_figures(layout):
    # 24 rows x 264 B per token per layer; both engram layers per step.
    assert layout.staging_bytes(1) * 2 == 12_672
    assert layout.staging_bytes(256) * 2 == 3_244_032
    assert layout.staging_bytes(8192) * 2 == 103_809_024
    assert layout.staging_shapes(256) == ((6144, 256), (6144, 8))


def test_staging_sizes_follow_the_local_column_count(layout):
    """Under TP the buffer is sized for this rank's columns only; sizing it
    with the model-wide count would over-allocate by the TP size and disagree
    with the row count the fill path derives from part_n_hash_cols."""
    assert layout.staging_shapes(256, 6) == ((1536, 256), (1536, 8))
    assert layout.staging_bytes(256, 6) * 4 == layout.staging_bytes(256)
    # the default stays the undivided count, i.e. the TP=1 behaviour
    assert layout.staging_shapes(256, None) == layout.staging_shapes(256)


# -- guard 1: the TP=1 degeneracy this numbering relies on --------------------


@pytest.mark.parametrize("local,total,start", [
    (24, 24, 0),     # TP=1: the undivided case
    (12, 24, 0),     # sharded across two ranks ...
    (12, 24, 12),    # ... and this is the second one
    (6, 24, 18),     # TP=4, last rank -- the shape that used to be refused
])
def test_well_formed_column_shards_are_accepted(layout, local, total, start):
    """Sharding is supported now: the ids are global row numbers and every
    rank maps the whole table, so a rank only gathers its own columns."""
    layout.assert_column_shard_is_well_formed(local, total, start)


@pytest.mark.parametrize("local,total,start,match", [
    (24, 24, 24, "of only 24"),       # window runs off the end
    (6, 24, 20, "of only 24"),        # ... and so does an undivisible one
    (8, 8, 0, "disagrees with"),      # a different column count entirely
    (0, 24, 0, "degenerate"),         # owns nothing
    (6, 24, -1, "degenerate"),        # negative start
])
def test_malformed_column_shard_is_refused(layout, local, total, start, match):
    """A bad window would gather valid row numbers for the WRONG columns,
    which no downstream check notices, so it has to fail here."""
    with pytest.raises(ValueError, match=match):
        layout.assert_column_shard_is_well_formed(local, total, start)


# -- guard 2: config shapes against the files themselves ----------------------


@pytest.mark.skipif(not os.path.isfile(os.path.join(CHECKPOINT, "config.json")),
                    reason="checkpoint config not available")
def test_from_config_on_the_real_checkpoint():
    got = EngramLayout.from_config(_real_config())
    assert got.n_hash_cols == 24, "(max_ngram_size - 1) * n_heads = 3 * 8"
    assert got.dim == 256
    assert got.n_scales == 8
    assert got.layer_ids == (1, 14)


@pytest.mark.skipif(
    not os.path.isdir(DEFAULT_STAGE_DIR)
    or not os.path.isfile(os.path.join(CHECKPOINT, "config.json")),
    reason="engram tables are not staged on this machine")
def test_the_real_tables_agree_with_the_real_config():
    """The same fact by two routes: config shapes vs actual file sizes."""
    EngramLayout.from_config(_real_config()).validate_against_files(
        DEFAULT_STAGE_DIR, staged_file_name)


@pytest.mark.skipif(not os.path.isdir(DEFAULT_STAGE_DIR),
                    reason="engram tables are not staged on this machine")
def test_the_attention_head_dim_is_caught(layout):
    """head_dim is 256 for engram and 512 for attention in this model.

    Reading the wrong one makes every gathered row straddle two real rows,
    with no error anywhere -- so the check has to be against the bytes, and
    the message has to name the confusion.
    """
    wrong = EngramLayout(n_hash_cols=24, dim=512, quant_block=32,
                         layer_ids=(1,), num_embeddings=(384006168,))
    with pytest.raises(ValueError) as exc:
        wrong.validate_against_files(DEFAULT_STAGE_DIR, staged_file_name)
    msg = str(exc.value)
    assert "512" in msg and "256" in msg
    assert "attention" in msg


def test_a_missing_table_is_reported_not_skipped(tmp_path, layout):
    with pytest.raises(ValueError, match="missing at"):
        layout.validate_against_files(str(tmp_path), staged_file_name)


# -- the CPU gather, and the ordering hazard it introduces --------------------


def _synthetic_tables(layout, rows=512):
    """Tables whose contents encode their own row number, so a mispaired row
    is identifiable rather than merely different."""
    import numpy as np
    w = np.zeros((rows, layout.dim), dtype=np.uint8)
    s = np.zeros((rows, layout.n_scales), dtype=np.uint8)
    w[:, 0] = np.arange(rows) % 256
    w[:, 1] = (np.arange(rows) // 256) % 256
    s[:, 0] = np.arange(rows) % 251
    return w, s


def _empty_staging(layout, num_tokens):
    import numpy as np
    (wshape, sshape) = layout.staging_shapes(num_tokens)
    return (np.zeros(wshape, dtype=np.uint8), np.zeros(sshape, dtype=np.uint8))


def test_each_staging_row_holds_the_table_row_that_token_asked_for(layout):
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    num_tokens, rows = 7, 512
    wt, st = _synthetic_tables(layout, rows)
    rng = np.random.default_rng(3)
    # hash_ids[token][col] -> table row
    hash_ids = rng.integers(0, rows, size=(num_tokens, layout.n_hash_cols))

    row_ids = np.empty(num_tokens * layout.n_hash_cols, dtype=np.int64)
    for t in range(num_tokens):
        for c in range(layout.n_hash_cols):
            row_ids[layout.row_index(t, c, layout.n_hash_cols)] = hash_ids[t, c]

    ow, os_ = _empty_staging(layout, num_tokens)
    gather_into_staging(wt, st, row_ids, ow, os_)

    for t in range(num_tokens):
        for c in range(layout.n_hash_cols):
            r = layout.row_index(t, c, layout.n_hash_cols)
            want = hash_ids[t, c]
            assert np.array_equal(ow[r], wt[want]), (t, c)
            assert np.array_equal(os_[r], st[want]), (t, c)


def test_a_head_major_ordering_is_wrong_without_being_invalid(layout):
    """The failure this layout module exists to prevent.

    Head-major and token-major are both dense, total, in-range orderings. Swap
    them and every staging row still holds a real table row -- just the wrong
    one for that position. Nothing raises; the only symptom is accuracy.
    """
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    num_tokens, rows = 7, 512
    wt, st = _synthetic_tables(layout, rows)
    rng = np.random.default_rng(3)
    hash_ids = rng.integers(0, rows, size=(num_tokens, layout.n_hash_cols))

    def build(head_major):
        ids = np.empty(num_tokens * layout.n_hash_cols, dtype=np.int64)
        for t in range(num_tokens):
            for c in range(layout.n_hash_cols):
                r = ((c * num_tokens + t) if head_major
                     else layout.row_index(t, c, layout.n_hash_cols))
                ids[r] = hash_ids[t, c]
        ow, os_ = _empty_staging(layout, num_tokens)
        gather_into_staging(wt, st, ids, ow, os_)
        return ow

    correct, swapped = build(False), build(True)
    assert not np.array_equal(correct, swapped), "the orderings must differ"
    # ... and every row of the wrong one is still a genuine table row
    for r in range(swapped.shape[0]):
        assert np.any(np.all(wt == swapped[r], axis=1)), (
            "a mispaired row is still well-formed data, which is exactly why "
            "nothing downstream can notice")


def test_an_off_by_one_row_shift_is_wrong_without_being_invalid(layout):
    """The other silent mis-pairing: a shift, not a transposition.

    ``test_a_head_major_ordering_is_wrong_without_being_invalid`` covers a
    transposition. This covers a shift, and neither covers the other: a
    transposition leaves row 0 alone and moves distant rows far, a shift moves
    every row by exactly one. A guard that catches one can be blind to the
    other, so both need a case that fails.

    A shift is the likelier of the two to be written by hand -- it is what
    ``token * n_hash_cols + col - 1``, an off-by-one in a loop bound, or a
    buffer written from index 1 all produce -- and it is the harder to see,
    because 24 of every 25 rows still land inside the same token's block.

    The point of the test is the second half: nothing refuses it. Every row is
    still a genuine table row, the buffer is still exactly the right length,
    and ``gather_into_staging`` accepts it without complaint. The only symptom
    is accuracy, which is why this has to be a test and cannot be a guard.
    """
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    num_tokens, rows = 7, 512
    wt, st = _synthetic_tables(layout, rows)
    rng = np.random.default_rng(11)
    hash_ids = rng.integers(0, rows, size=(num_tokens, layout.n_hash_cols))
    n_staging = num_tokens * layout.n_hash_cols

    def build(shift):
        ids = np.empty(n_staging, dtype=np.int64)
        for t in range(num_tokens):
            for c in range(layout.n_hash_cols):
                ids[(layout.row_index(t, c, layout.n_hash_cols) + shift)
                    % n_staging] = hash_ids[t, c]
        ow, os_ = _empty_staging(layout, num_tokens)
        gather_into_staging(wt, st, ids, ow, os_)
        return ow

    correct = build(0)

    # The criterion must be able to say "same" as well as "different", or its
    # disagreement below proves nothing about the shift.
    assert np.array_equal(correct, build(0)), (
        "the same ordering must reproduce itself -- otherwise the comparison "
        "is picking up something other than the row assignment")

    shifted = build(-1)
    assert not np.array_equal(correct, shifted), (
        "an off-by-one row shift must change the staging buffer")

    # ... and here is why nothing downstream can notice.
    assert shifted.shape == correct.shape, "the shift does not change the length"
    for r in range(shifted.shape[0]):
        assert np.any(np.all(wt == shifted[r], axis=1)), (
            "every row of the shifted buffer is still a real table row: the "
            "bytes are well-formed, only the pairing with a token is wrong")


def test_threading_does_not_change_the_result(layout):
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    rows = 4096
    num_tokens = 2048                      # 49152 rows, over the threshold
    wt, st = _synthetic_tables(layout, rows)
    row_ids = np.random.default_rng(0).integers(
        0, rows, size=num_tokens * layout.n_hash_cols, dtype=np.int64)

    a_w, a_s = _empty_staging(layout, num_tokens)
    b_w, b_s = _empty_staging(layout, num_tokens)
    gather_into_staging(wt, st, row_ids, a_w, a_s, threads=1)
    gather_into_staging(wt, st, row_ids, b_w, b_s, threads=16)
    assert np.array_equal(a_w, b_w) and np.array_equal(a_s, b_s)


def test_a_staging_buffer_of_the_wrong_length_is_refused(layout):
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    wt, st = _synthetic_tables(layout, 256)
    ow, os_ = _empty_staging(layout, 4)
    row_ids = np.zeros(layout.n_hash_cols * 5, dtype=np.int64)   # 5 tokens
    with pytest.raises(ValueError, match="index each other positionally"):
        gather_into_staging(wt, st, row_ids, ow, os_)
