# SPDX-License-Identifier: Apache-2.0
"""The descriptor and the call must reach the mask from one function.

They used to reach it from two expressions over the same facts, and the two
agreed -- which is the worst state for a second source to be in, because
agreement is also the acceptance criterion. A precedence change would have
moved one and not the other, and nothing would have said so.

Also here: the descriptor check, whose value is mostly in what it admits it
cannot see.
"""

import inspect

import pytest
import torch

from vllm_ascend.attention import sparse_flash_mla as adapter
from vllm_ascend.models.deepseek_v4_1 import sparse_attn_args as args

# ---------------------------------------------------------------- one source

def test_the_descriptor_builder_calls_the_call_sites_predicate():
    """Derived from the builder's source, so a reintroduced second expression
    fails here rather than at an aicore exception three weeks later."""
    from vllm_ascend.attention import dsa_v1

    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder.build_req_metadata)
    assert "is_plain_sliding_window(" in source, (
        "the descriptor no longer asks the call site's predicate; it is "
        "computing the mask from its own expression again")
    assert "select_ori_sparse_indices(" in source, (
        "the descriptor is choosing between vision and dspark indices with "
        "its own precedence rule again")


def test_no_hand_rolled_has_indices_remains_in_the_builder():
    """The specific expression that was the second source."""
    from vllm_ascend.attention import dsa_v1

    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder.build_req_metadata)
    assert "has_indices" not in source


@pytest.mark.parametrize("attends_compressed", [False, True])
@pytest.mark.parametrize("vision", [None, "V"])
@pytest.mark.parametrize("dspark", [None, "D"])
def test_the_precedence_is_the_call_sites(attends_compressed, vision, dspark):
    """All eight combinations, against the rule stated at the call site.

    Written out rather than delegated to the function under test: a test that
    recomputes the rule cannot catch the rule changing.
    """
    got = args.select_ori_sparse_indices(attends_compressed, vision, dspark)
    if not attends_compressed and dspark is not None:
        assert got == "D"
    else:
        assert got == vision


def test_a_layer_that_attends_compressed_keeps_visions_indices():
    """The one case the precedence exists for, named."""
    assert args.select_ori_sparse_indices(True, "V", "D") == "V"


# ------------------------------------------------------------ the descriptor

def _kwargs(metadata):
    return {"metadata": metadata}


def test_a_missing_descriptor_is_refused():
    with pytest.raises(ValueError, match="metadata must be provided"):
        args._check_descriptor_is_populated(_kwargs(None), torch.zeros(4, 512))


def test_a_short_descriptor_is_refused():
    with pytest.raises(ValueError, match="1024 int32 words"):
        args._check_descriptor_is_populated(_kwargs(torch.zeros(64,
                                                               dtype=torch.int32)),
                                            torch.zeros(4, 512))


def test_an_all_zero_descriptor_with_tokens_is_refused():
    blob = torch.zeros(args.DESCRIPTOR_WORDS, dtype=torch.int32)
    with pytest.raises(ValueError, match="never wrote into this buffer"):
        args._check_descriptor_is_populated(_kwargs(blob), torch.zeros(4, 512))


def test_a_populated_descriptor_passes():
    blob = torch.zeros(args.DESCRIPTOR_WORDS, dtype=torch.int32)
    blob[3] = 28
    args._check_descriptor_is_populated(_kwargs(blob), torch.zeros(4, 512))


def test_an_empty_batch_is_reported_as_not_checked_not_as_passed(caplog):
    """Third state. An empty batch legitimately produces an all-zero split,
    so the check cannot fire -- and silence would be identical to a pass."""
    import logging

    from vllm.logger import logger as vllm_logger

    args._LOGGED_DESCRIPTOR_SKIPS.clear()
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    vllm_logger.addHandler(handler)
    try:
        blob = torch.zeros(args.DESCRIPTOR_WORDS, dtype=torch.int32)
        args._check_descriptor_is_populated(_kwargs(blob), torch.zeros(0, 512))
    finally:
        vllm_logger.removeHandler(handler)

    assert any("NOT checked" in m for m in records), (
        "an unchecked descriptor produced no output, which reads exactly like "
        f"a passed one; got {records}")


def test_the_check_says_what_it_cannot_see():
    """The failure this port actually hit was a well-formed descriptor built
    for another configuration -- 83 non-zero words. If the docstring stops
    saying so, the check starts being read as stronger than it is."""
    promise = " ".join(args._check_descriptor_is_populated.__doc__.split())
    assert "belongs to someone else passes every check" in promise


# ------------------------------------------------- the compressed-length gate

def test_ratio_one_with_a_compressed_stream_now_gets_its_lengths():
    """Layers 20..39: ratio 1 means "compressed, pooling disabled" in V4.1."""
    kwargs = {"cmp_ratio": 1, "seqused_ori_kv": torch.tensor([8, 16]),
              "cmp_kv": torch.zeros(1)}
    adapter._add_compressed_kv_lengths(kwargs)
    assert "seqused_cmp_kv" in kwargs
    torch.testing.assert_close(kwargs["seqused_cmp_kv"], torch.tensor([8, 16]))
    torch.testing.assert_close(kwargs["cmp_residual_kv"], torch.tensor([0, 0]))


def test_ratio_one_without_a_compressed_stream_still_gets_nothing():
    """Layers 0/1/40..42 clamp to 1 too; the ratio cannot separate them."""
    kwargs = {"cmp_ratio": 1, "seqused_ori_kv": torch.tensor([8, 16])}
    adapter._add_compressed_kv_lengths(kwargs)
    assert "seqused_cmp_kv" not in kwargs


def test_the_descriptor_vocabulary_is_read_too():
    kwargs = {"cmp_ratio": 1, "seqused_ori_kv": torch.tensor([8]),
              "has_cmp_kv": True}
    adapter._add_compressed_kv_lengths(kwargs)
    assert "seqused_cmp_kv" in kwargs


def test_has_cmp_kv_false_overrides_a_ratio_that_would_say_yes():
    """A field that states the fact beats a ratio that merely implies one."""
    kwargs = {"cmp_ratio": 2, "seqused_ori_kv": torch.tensor([8]),
              "has_cmp_kv": False}
    adapter._add_compressed_kv_lengths(kwargs)
    assert "seqused_cmp_kv" not in kwargs


def test_v40s_reading_is_unchanged_when_neither_field_is_present():
    """No has_cmp_kv, no cmp_kv: the caller is V4.0's and keeps ratio > 1."""
    for ratio, expected in ((0, False), (1, False), (2, True), (4, True)):
        kwargs = {"cmp_ratio": ratio, "seqused_ori_kv": torch.tensor([8])}
        adapter._add_compressed_kv_lengths(kwargs)
        assert ("seqused_cmp_kv" in kwargs) is expected, ratio


def test_an_explicit_length_is_never_overwritten():
    supplied = torch.tensor([3])
    kwargs = {"cmp_ratio": 2, "seqused_ori_kv": torch.tensor([8]),
              "cmp_kv": torch.zeros(1), "seqused_cmp_kv": supplied}
    adapter._add_compressed_kv_lengths(kwargs)
    assert kwargs["seqused_cmp_kv"] is supplied
