# DSV4.1-Flash single-card MoE offload — what is measured, and how to turn it on

One copy of the expert weights lives on the host in the NPU's FRACTAL_NZ
layout (`/dev/shm/dsv41_nz_bank`, 268.95 GiB, 40 layer files). Streaming
prefill reads it, and kt-kernel's CPU MoE reads the **same bytes in place** —
that is what removes the duplicate copy. The decode half is unchanged.

Everything below was measured on card, on this host, and every claim has a line
the server itself prints. Numbers from a single round, or without a null pair,
are marked as such.

## Gains

| | before | after | evidence |
|---|---|---|---|
| **Host working set** | 558.8 GiB | **288.8 GiB** (−270.0) | `dropped the checkpoint's clean page cache over 48 shard(s): 267.3 -> 0.0 GiB`, `numa_placement: planned=70502400 resident_at_census=70502400` |
| **HBM** | — | **−2.175 GiB**, plus −12.3 GiB of KV budget | sampler peak |
| **TTFT @p=4096** | 15.50 s | **12.19 s (−21%)** | two independent steps, below |
| **TPOT** | 23.5–26.2 ms | **+16% (27.5–30.8 ms)** 🔴 | `measure/chain_tpot.sh`, 3 rounds vs the NO-BANK baseline — the cost of the scheme, localised below |
| **Correctness** | — | **7/7 prompts token-identical**, at every step | `measure/eqdiff.py` against the stage-3 capture |

The TTFT figure is two changes:

* **loader threads 4 → 16** (`wiring.py`): the streaming read leg is ~94% of the
  staging loop, and it was running on `MIN_LOADER_THREADS` where the checkpoint
  path it replaced used `DEFAULT_COPY_THREADS`. Read leg **47 → 69 GiB/s**;
  TTFT **−9.7% at p=1024 and −11.9% at p=4096, 3 of 3 rounds**. The server says
  which it used: `bank_source: 16 loader thread(s) on cpus [0, 8, ..., 120]`.
* **`stream_resident_from_hbm`** (this file's subject): a prefill streamed all
  384 experts of all 40 layers — 268.9 GiB per request — including the ~100
  already in HBM. Their slots of the chunk buffer are now filled by a
  device-to-device copy from the resident parameter instead. **198.9 GiB per
  request instead of 268.9**; steady-state TTFT **13.79 → 12.19 s (−11.6%)**,
  nine requests per arm with **no overlap** between the two ranges (off
  13.51–14.44, on 11.77–12.53). The server says
  `HOST MOVED 49.7 of 67.2 GiB (74.0%), 1000 slot(s) filled from HBM instead`.

🔴 It is done at the provider level, filling the chunk buffer, **so that the
chunk partition, the slot tables and the fp32 accumulation order are untouched
and the output is bit-identical**. Streaming only the offloaded set and
computing the resident half separately would be less code and would change the
rounding; then the 7-prompt capture could only be asked for token identity, not
bit identity, and the gate would be weaker than the change it guards.

## Turning it on

```json
{"kt_offload_config": {
  "enabled": true, "resident_strategy": "prefix", "num_resident_experts": 100,
  "cpuinfer_threads": 96, "threadpool_count": 2, "numa_nodes": [0, 1],
  "packed_transfer": false, "stream_prefill": true, "dynamic_resident": true,
  "zerocopy_w13": true, "zerocopy_node_fraction": [1.0, 1.0],
  "zerocopy_streaming_node": 1, "debug_time_chunks": true,
  "bank_dir": "/dev/shm/dsv41_nz_bank",
  "zerocopy_scope": "all",
  "kt_reads_bank": true,
  "stream_resident_from_hbm": true
}}
```

| switch | what it does | needs |
|---|---|---|
| `bank_dir` | the NPU streams from the NZ bank instead of the checkpoint | a built bank (`bank_build.py`) |
| `zerocopy_w13` + `zerocopy_scope: "all"` | kt consumes w13 **and** w2 in place | `KT_ZEROCOPY_WEIGHTS=1` in the server's env |
| `kt_reads_bank` | kt's pointers go to the **bank**, so the expert bytes exist once | `bank_dir` + `zerocopy_w13` |
| `stream_resident_from_hbm` | a chunk's resident members come from HBM, not the host | `bank_dir` + `stream_prefill` |

🔴 **`stream_resident_from_hbm` is default OFF in the dataclass and must stay
that way**: it carries preconditions, so defaulting it on makes every config
without them fail validation — 20 of this repo's own config tests did,
immediately, when that was tried. A switch with preconditions is opted into by
the deployment config.

Env knobs, all optional: `VLLM_ASCEND_DSV41_KT_LOADER_THREADS` overrides the
gather's thread count per server start (the count is fixed when the source is
built, so an A/B over it needs one start per arm);
`VLLM_ASCEND_DSV41_KT_KNOB_FILE` lets a measurement flip kt's kernel knobs
inside a running server.

## The kernel side is NOT in this repository

kt-kernel is a separate checkout on a detached HEAD with uncommitted work, and
it must not be committed to. The changes that go with this feature are kept as
patches next to the measurements:

* `backups/pfd_20261115/kt_unstaged.diff` and `kt_staged.diff`
* build tag in the live `.so`: `KT_PREFETCH_NZ16_SHORT4_NTA_ND_MIN40_OVER_G4_STRIDED`

What they contain: the MXFP4 kernel reads FRACTAL_NZ by a **stride change**
(`group_stride() == n`, 16 contiguous bytes per (row, k-group) — NZ's C0 is 16 B
and one MXFP4 k-group is also exactly 16 B), per-TP-part pointer indexing, and
the prefetch policy. The prefetch ADDRESS used to advance 16 B per group while
the load advanced N·16 B, i.e. it prefetched the wrong data for every bank arm
ever measured; fixing it plus retuning the policy to **(long 16, short 4, NTA)**
is worth **−5.9% of kt's decode GEMV on a quiet machine and −9.2% on a busy
one**, 0 of 40 paired rounds slower, p=1.8e-12.

## Not done, and known

* **Accuracy has never been run.** GSM8K reference 97.27. 🔴 Its prompts are
  ~100 tokens, below the 512-token streaming threshold, so at the production
  threshold it exercises neither the bank nor the fold;
  `measure/gsm8k_bank_stream.json` lowers it to 64.
* 🔴 **TPOT costs +16% against a no-bank baseline, and it is the NZ traversal
  in kt's decode GEMV.** Measured 2026-11-15 and localised in three steps:

  1. **Which half of the stack** (`measure/chain_tpot.sh`, 3 rounds, arms
     alternating, first three requests of each server discarded). A bank that
     only feeds the NPU's streaming prefill is **free** — `s2/base` =
     0.844 / 0.946 / 0.972, never slower in three rounds. kt *also* reading it
     costs **+1.2 / +22.6 / +23.3%**, and the `now` arm's lowest round median
     (27.52 ms) is above the `s2` arm's highest (27.21).
  2. **Which part of that step** (`measure/kt_layout_bench.py`: the real bank,
     the production kernel, two MOE instances in ONE process, arms
     alternating, outputs gated bit-identical). The same weights read as ND
     instead of NZ are **36% faster** (24/24 rounds, p=1.2e-07; 38.7% in a
     single-node repeat against a ±5.03% null floor).
  3. **Which mechanism.** Not placement: the server reports
     `on_target=100.00%` over all 70,502,400 pages, and `split_share=0.9412`
     = (w13+w2)/total, i.e. w2 *is* split per part and both halves are local.
     Not cache-line waste: the kernel already blocks rows by four and
     `accumulate_group` reads rows ni..ni+3 at the same group, which under NZ
     is one whole 64 B line. **Not the TLB**: a byte-identical NZ copy on
     2 MiB pages (`ShmemPmdMapped` 100%) measured **1.0009**, 12/24 rounds
     slower. What is left is the traversal itself — one 64 B line per
     18,432 B stride, against ND's sequential stream — which no page size can
     fix and a wider row block in the kernel can.

  🔴 The earlier "+2.8…5.0%, cause: the NZ walk costs the kernel ~5%" was wrong
  in both halves, and the reason it survived is worth keeping: every control
  that said "TPOT is unchanged" was **bank-vs-bank**, so the bank's cost
  cancelled on both sides. An internal control cannot see a cost its two arms
  share.
* **`stream_resident_from_hbm` costs ~4 s once**, over the first two requests of
  a server's life, and that is **not explained**. The staging counters refute
  both obvious causes: the reports before and after the jump have the same fill
  shape (whole=0, mixed=120, ~3000 D2D calls, 17.5 GiB) and differ 4× in
  ms/chunk.
* Not implemented: the node-matched gather partition (offline **76.9 vs 69.6
  GiB/s** at the same 16 threads), reading the scales from the bank (~16 GiB
  resident plus a 269 GiB load-time transient), and the w13 N-reorder.
