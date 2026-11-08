# SPDX-License-Identifier: Apache-2.0
"""SparseFlashMla applies the block table to request-relative ori indices."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the npu device

from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan
from vllm_ascend.attention.dsa_v1 import build_dspark_swa_indices
from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import build_sparse_flash_mla_kwargs


@torch.inference_mode()
def test_nonidentity_pages_match_dense_reference_and_reject_physical_indices():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend device")
    torch.npu.set_device(0)
    plan = get_dsa_attn_kv_plan(SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="bfloat16")))
    if not plan.uses_sparse_flash_mla:
        pytest.skip("requires the A5 BF16 SparseFlashMla backend")
    page, seq_len, query_len, heads, dim = 128, 150, 5, 64, 512
    generator = torch.Generator().manual_seed(321)
    q_cpu = (torch.randn(query_len, heads, dim, generator=generator) * 0.25).to(torch.bfloat16)
    keys = (torch.randn(seq_len, dim, generator=generator) * 0.2 + 1).to(torch.bfloat16)
    sink_cpu = torch.linspace(-0.5, 0.5, heads)
    q, sinks = q_cpu.npu(), sink_cpu.npu()
    cu_q = torch.tensor([0, query_len], dtype=torch.int32).npu()
    seq = torch.tensor([seq_len], dtype=torch.int32).npu()
    # The trailing 128 context tokens and all five queries span two logical pages.
    visible = keys[seq_len - query_len - page :].float()
    logits = q_cpu.float() @ visible.T / math.sqrt(dim)
    probs = torch.softmax(torch.cat((logits, sink_cpu.view(1, heads, 1).expand(query_len, -1, -1)), dim=-1), dim=-1)
    expected = probs[..., :-1] @ visible
    outputs = []
    for physical_pages in [(5, 3), (2, 6)]:
        cache = torch.full((7, page, 1, dim), -4, dtype=torch.bfloat16)
        cache[physical_pages[0], :, 0] = keys[:page]
        cache[physical_pages[1], : seq_len - page, 0] = keys[page:]
        cache = cache.npu()
        # Extra zero columns keep the deliberately wrong physical-index control
        # in allocated table/cache bounds; it must fail numerically, not crash.
        table = torch.tensor([[*physical_pages, 0, 0, 0, 0, 0, 0]], dtype=torch.int32).npu()
        metadata = plan.get_dsa_sparse_attn_metadata_op()(
            **plan.get_dsa_sparse_attn_metadata_kwargs(q.device),
            num_heads_q=heads,
            num_heads_kv=1,
            head_dim=dim,
            cu_seqlens_q=cu_q,
            cu_seqlens_ori_kv=torch.tensor([0, seq_len], dtype=torch.int32).npu(),
            cu_seqlens_cmp_kv=None,
            seqused_q=torch.tensor([query_len], dtype=torch.int32).npu(),
            seqused_kv=seq,
            max_seqlen_q=query_len,
            max_seqlen_kv=seq_len,
            batch_size=1,
            cmp_ratio=1,
            ori_mask_mode=0,
            cmp_mask_mode=0,
            ori_win_left=-1,
            ori_win_right=-1,
            layout_q="TND",
            layout_kv=plan.layout_kv,
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        for logical_indices in [True, False]:
            indices, lengths = build_dspark_swa_indices(
                table, query_len, page, page, cu_q, seq, query_len, logical_indices=logical_indices
            )
            kwargs = build_sparse_flash_mla_kwargs(
                q=q,
                num_requests=1,
                swa_kv_cache=cache,
                swa_block_table=table,
                cu_seqlens_q=cu_q,
                seqused_kv=seq,
                sinks=sinks,
                metadata=metadata,
                softmax_scale=1 / math.sqrt(dim),
                cmp_ratio=1,
                layout_kv=plan.layout_kv,
                attends_compressed=False,
                ori_win_left=-1,
                ori_win_right=-1,
                ori_sparse_indices=indices,
                ori_topk_length=lengths,
            )
            actual = plan.get_dsa_sparse_attn_op()(q, **kwargs)[0].float().cpu()
            if logical_indices:
                torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.01)
                outputs.append(actual)
            else:
                assert (actual - expected).abs().max() > 0.1
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.015, rtol=0.01)
