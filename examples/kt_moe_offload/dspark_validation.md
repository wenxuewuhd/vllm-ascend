# Single-card DSpark validation

Use the [README setup and launch command](README.md#experimental-single-card-dspark).
The tested stack is vLLM `a9686bc6e6809a49cb3e3c592a337d0721c3619f`,
CANN 9.2.0-beta.2, single-card text generation, bf16 KV, target graphs, and eager
draft execution. The PR targets `dsv41-moe-offload-pr` at
`59ce1e18a4f93c20a872e4450bd14e891c556cb7`.

## Current adapter

The adapter inherits upstream forward and checkpoint loading, binds three
Ascend hardware entry points during drafter construction, and reuses the
target's mHC, RoPE, and paged KV-write helpers.

The 2026-11-02 simplification passed 59 isolated CPU tests using real PyTorch;
six model-index-gated tests skipped when the checkpoint index was absent.
The checks stub vLLM/hardware imports and cover registration, hardware binding,
context-KV insertion, logical/physical indices, persistent buffers, proposer
input preparation, and target/draft expert-placement boundaries. Full vLLM
integration and a fresh native NPU validation of this revision remain pending.

## Historical validation and limits

Before simplification, runtime `8147c5f4e131939878ff30086f69d5d885142431`
passed 280 targeted unit tests, one native A5 attention regression, and five
generation smoke cases. The native regression compares shuffled KV pages with
a dense reference and confirms that the former physical-index implementation
fails. The smoke run accepted 142/600 proposed tokens (23.67%).
These results apply to that earlier runtime only.

Later asynchronous K2/K5 LongBench samples lacked an OFF control and produced
different full output sequences across K. A later controlled experiment did not
complete its final OFF bracket. Neither establishes stable DSpark acceleration
or a complete model-quality score; non-DSpark README timings do not apply here.
Multi-card and multimodal end-to-end generation remain unverified.

Acceptance is `A/T`, with accepted draft tokens A and proposed tokens T.
The counter-based length including a target token is `1 + A/R`, where R counts
verification rounds. EOS/output caps can truncate the final delivered round.

## Optional CPU kernel

[ktransformers-AK PR #1](https://github.com/wenxuewuhd/ktransformers-AK/pull/1)
contains the CPU expert weight-reuse implementation, build instructions, and
its separate validation history. Rebuild that branch and select it through
`KT_SITE`; updating vllm-ascend alone does not install it. The existing kernel
is sufficient for functional DSpark execution. Historical kernel measurements
predate the submitted kernel port and do not validate a fresh build of it or
establish a stable whole-model gain.
