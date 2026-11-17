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
| **TPOT** | — | 🔴 **NOT MEASURABLE on this host** | four instruments, each with a null pair: floors ±12.8…14.3% against an effect of ~4%. Every TPOT number this project ever quoted is retracted. The cost's MECHANISM and its KERNEL-level size are known — see below |
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
* 🔴 **TPOT: the mechanism is settled, the end-to-end size is not measurable
  here.** Three things are established, and the fourth is a measurement limit.

  **1. What the cost is.** Measured at the kernel — the real bank, the
  production kernel, two MOE instances in ONE process, arms alternating, every
  arm's output gated **bit-identical** before any timing
  (`measure/kt_layout_bench.py`, `measure/kt_pfd_bench.py`, null floor
  **±1.72%**):

  | | ratio | rounds | p |
  |---|---|---|---|
  | the same weights read as **ND** instead of NZ (before the two fixes) | **36–47% faster** | 24/24 | 1.2e-07 |
  | the NZ copy on **2 MiB pages** (so: not the TLB) | 1.0009 | 12/24 | — |
  | **row block 4 → 8** (`KT_NZ_ROW_BLOCK`, shipped) | rb4 is **+13.51%** | 24/24 | 1.2e-07 |
  | **a rebuilt m=1 inner loop** (`KT_NZ_G_OUTER`, shipped, default ON) | **−24.86%** | 0/20 | 1.9e-06 |
  | **→ ND vs NZ after both** | **ND only 8.64% faster** | 0/20 | 1.9e-06 |

  Where the cost is NOT: placement (`on_target=100.00%` over all 70,502,400
  pages), cache-line waste (a block's four rows at one group are one whole
  line), the TLB (2 MiB pages measured 1.0009).

  🔴 **What it IS has not been established, and the first answer was wrong.**
  The gap was attributed to the loop order — "the kernel walks NZ in ND's
  order, rows outer and groups inner". It does not: `accumulate_group(g)` has
  always had g outermost, so the old and new inner loops issue the **same
  addresses in the same order**. The data refutes the story as well: if a
  longer contiguous run per group were the win, `go8` (128 B) would be far
  worse than `go256` (4 KiB), and they measure 0.7514 against 0.7548. The row
  block R barely matters in the new loop.

  **The 24.9% was then split by measurement** (2026-11-17, same bench, 20
  paired rounds each), and all three of the first candidates are dead:

  | candidate | arm | result |
  |---|---|---|
  | one prefetch per 64 B line vs per row | rb8 ± dedup | 0.8785 vs 0.8810 — **nothing** |
  | the force-unroll pragma | go8 vs go8_unroll | 0.7832 vs 0.7785 — **nothing** |
  | the scale handling | — | the new loop does **more** scalar work here (R byte-loads per group against a vectorised per-row expansion), so it is a cost it pays, not the gain |

  What the split IS:
  * **~14% survives with NO prefetch anywhere.** At `--distances 0` on both
    loops the rebuilt one is still **13.79% faster** (0/20, p=1.9e-06). This
    part is **unexplained**.
  * **~8% is prefetch COVERAGE**, a mechanism nobody had noticed: the shipped
    loop calls `prefetch_group(g)` once per **four** groups (`for (; g + 3 <
    group_count; g += 4)`), so groups g+1..g+3 were never prefetched at all.
    The rebuilt loop prefetches every group. Confirmed from the other side —
    turning prefetching off costs **3.9%** in the old loop and **11.6%** in the
    new one.

  A distance re-sweep on the rebuilt loop leaves the shipped constants best:
  d16 is the optimum (d0 +11.6%, d8 +2.1%, d24 +2.0%, d32 +4.0%, d48 +4.8%, all
  17–19/20 rounds), and short 2/8/16 are all inconclusive against short 4.

  **2. What was done about it.** Two kernel changes, both NZ-only, both
  zero-memory, both leaving each row's accumulation order untouched so the
  output **cannot** change:
  * `KT_NZ_ROW_BLOCK` (default 8): accumulate 8 rows per block instead of 4.
  * `KT_NZ_G_OUTER` (default on): a rebuilt inner loop — see above for what it
    actually changes and for what is still unexplained.

  Together they take ND's advantage from 36–47% to **8.64%**, and ND still
  carries w2's 1/tp locality handicap, so the true remainder is smaller.
  Verified on card **four independent times, 7/7 prompts token-identical**
  (`rb8_eq`, `rb_ab`, `rb_alt`, `go_eq`), plus a per-arm bit-identity gate
  inside both benches that runs before any timing.

  🔴 Two reproducible non-monotonicities remain **unexplained** and are kept
  reachable by the knobs so the next person can reproduce them in one run:
  `nz_row_block=16` is worse than both 8 and 32 (+19%, +15%, +25% across three
  runs), and under the new loop `go32` (0.8665) is worse than go8, go64, go128
  and go256 (0.75–0.79).

  **3. 🔴 What it is worth end to end is UNKNOWN, and that is a property of this
  host, not an oversight.** Four instruments, each with a null pair (the same
  configuration measured twice, in the same position structure as the
  comparison):

  | instrument | design | TPOT floor | TTFT floor |
  |---|---|---|---|
  | `chain_tpot.sh` | a server per arm, 128 gen-tokens | ±13.06% | — |
  | `chain_rb_ab.sh` | ONE server, knob per block, 128 tokens | ±12.17% | ±5.23% |
  | `chain_rb_ab.sh` | ONE server, knob per block, 1024 tokens | ±14.33% | ±3.97% |
  | `chain_rb_alt.sh` | ONE server, knob per **request** (~40 s apart), 1024 tokens | ±12.79% (robust ±5.85%) | **±2.85%** |

  Against a ±5.85% robust floor the signal is **−1.30%, 4/10 rounds, p=0.75** —
  INCONCLUSIVE. TTFT, the negative control, behaves exactly as it must (+0.64%,
  inside its own ±2.29%), and its floor is **4.5× tighter in the same run**, so
  TPOT is the noisy quantity here, not the harness. `measure/chain_tpot_floor.sh`
  pinned the one thing that does help — two **back-to-back** runs of one config
  at 1024 tokens agree to ±1.29% — which means what is left is wall-clock drift
  between a pair's halves on a box whose load swings 24 → 200.

  **So the trade, stated honestly: −270 GiB of host memory and −21% TTFT, for a
  decode cost that started at 36–47% of kt's decode GEMV and is now 8.64%, and
  whose end-to-end size has never been resolved on this host.** Anyone quoting a
  TPOT percentage for this work is quoting noise — including the arithmetic
  bound that used to be here.

* **`stream_resident_from_hbm` costs ~4 s once**, over the first two requests of
  a server's life, and that is **not explained**. The staging counters refute
  both obvious causes: the reports before and after the jump have the same fill
  shape (whole=0, mixed=120, ~3000 D2D calls, 17.5 GiB) and differ 4× in
  ms/chunk.
* Not implemented: the node-matched gather partition (offline **76.9 vs 69.6
  GiB/s** at the same 16 threads), reading the scales from the bank (~16 GiB
  resident plus a 269 GiB load-time transient), and the w13 N-reorder.
