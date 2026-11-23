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
| **TTFT @p=4096** | 23.62 s | **8.88 s** (2.66× faster) | 2026-11-21, paired on one host, each arm in the host-memory state it would really have; `chunk lookahead ON`, `layout=NZ walk=GROUP-outer` |
| **TPOT** | 24.12 ms | **24.08 ms** — unchanged | 2026-11-21, same paired run at gen 512. 0.2% is far inside the ±12…14% two-server floor. 🔴 Earlier TPOT numbers remain retracted; this is the first one with a fair base arm |
| **Correctness** | — | **7/7 prompts token-identical**, at every step | `measure/eqdiff.py` against the stage-3 capture |

The TTFT figure is three changes, largest first:

* **the chunk lookahead** (`stream_chain.py`, `stream_io.py`): the streaming
  chain now starts the next chunk's host gather while the current chunk
  computes. `provider.prefetch()` existed, was tested, and had **no production
  caller**. The acquire leg falls **3.57 s → 0.10–0.19 s at identical bytes
  moved**; `VLLM_ASCEND_DSV41_KT_PREFETCH=0` restores the serial path in the
  same binary. The server says `chunk lookahead ON (depth 1, provider.prefetch)`.

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

* ✅ **Accuracy RUN 2026-11-17, all 1319 GSM8K questions: 97.04 against the
  published 97.27.** That is **+0.23 points = 3 questions = 0.49 standard
  errors** (1 SE = 0.47 pts at n=1319), i.e. no detectable difference; resolving
  a real 0.23-point gap would need n ≈ 21,700. The stronger statement for the
  same code remains the bit-identity: **7/7 prompts token-identical on card in
  four independent captures**.

  🔴 **What this arm covers, and how it was chosen.** The kernel work of
  2026-11-15..17 is entirely in the **m=1 decode** path; prefill runs the
  untouched mat-mat kernel. So accuracy does not need streaming prefill — it
  needs `kt_reads_bank` and decode, which runs at a batch. The arm
  (`measure/gsm8k_tput.json`) therefore keeps `kt_reads_bank` +
  `zerocopy_scope=all` and raises `stream_prefill_min_tokens` to 100000 so
  nothing streams, which lets it run at `max_num_seqs=4` and finish 1319
  questions in 83 minutes. The server's own line confirms the arm did what it
  was meant to: `[kt prefetch] ...RB8_GOUTERON...: gate/up layout=NZ`, and
  `bank-as-kt-source` appears in the log. `requests that committed all 40
  layers: 0` is the DESIGNED state here (no streaming, no fold), not a failure.
  🔴 The first arm tried, `gsm8k_bank_stream.json`, sets only `bank_dir` — the
  NPU streams from the bank while **kt keeps reading the checkpoint in ND** — so
  it would have scored a clean 97 having executed none of the changed code.
  **Streaming prefill and the folded hot set are still unvalidated by any
  dataset score**; they need their own single-request arms.

  Costs, from the same run and worth keeping: `max_num_seqs=8` **OOM'd** the
  card during the init profiling batch (1.10 GiB short), so the batch is 4 and
  `max_model_len` 5120 (ais_bench's gsm8k config needs ≥ 5120). And generation
  throughput at batch 4 is **34–50 tokens/s in total** across the 4 sequences,
  i.e. ~100 ms per token per sequence against ~30 ms at batch 1 — **batching
  barely raises total throughput**, which is the signature of a per-token expert
  read that does not amortise: each token routes to its own top-k, so four
  tokens touch up to 32 different experts. (The flatness is measured; the
  attribution to the CPU MoE's weight traffic is inference — it is the only
  component with that property, but no per-stage timing was taken.)
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


---

## 🔴🔴 2026-11-18: streaming prefill should not run, and the bank is still worth it

> ### ⚠️ AMENDED 2026-11-21 — the 2.03x below predates the chunk lookahead
>
> This section's recommendation rests on `streamg` costing 2.03x `nostream` at
> prompt 4096 (15.47 s vs 7.62 s). **That gap was measured before the chunk
> lookahead existed.** With the lookahead on, the same streaming configuration
> measures **8.88 s** -- so the gap to `nostream`'s 7.62 s is about **1.17x**,
> not 2.03x, and the headline figure of this file is now a *streaming* number.
>
> **What this does and does not change.** `nostream` is probably still the
> faster path at this prompt length; the margin has gone from decisive to
> narrow, which is a different recommendation. The comparison has not been
> re-run since the lookahead landed, so **the 2.03x is obsolete and the
> re-measurement is owed** -- `measure/chain_cross.sh` with the lookahead on,
> and with the bank's page-cache effect controlled for (see the retraction at
> the end of this file).
>
> Read everything below as the state of knowledge on 2026-11-18.


`measure/chain_cross.sh`, three requests per cell, no overlapping ranges:

| prompt | `nostream` | `stream` | `streamg` (+G′) | stream/nostream | streamg/nostream |
|---|---|---|---|---|---|
| 1024 | **2.28 s** | 24.42 | 14.44 | **10.72×** | 6.34× |
| 2048 | **3.99 s** | 23.87 | 14.77 | 5.98× | 3.70× |
| 4096 | **7.62 s** | 24.71 | 15.47 | 3.24× | 2.03× |
| 8192 | **15.35 s** | 49.88 | 29.30 | **3.25×** | 1.91× |

**Why there is no crossover.** Streaming moves the whole bank once per PREFILL
WINDOW (~4096 tokens), not once per request: flat at 24.4/23.9/24.7 from 1024 to
4096, then doubling to 49.88 at 8192. So both sides are linear in prompt length
and the ratio converges to ~3.25× (~1.9× with G′) instead of approaching 1. The
expectation of a crossover came from treating streaming's cost as constant per
request, which it is not.

**The recommendation, and what it keeps.** Raise
`stream_prefill_min_tokens` above the longest prompt the deployment sees.
`bank_dir` requires `stream_prefill`, so streaming stays *enabled* and simply
never fires — which means:

* the bank stays, `kt_reads_bank` stays, and the **−270 GiB host working set
  stays**;
* TPOT is unaffected (29.9 vs 31.1 ms, inside this host's ±13% floor);
* accuracy is already measured on exactly this configuration: **GSM8K 97.04**
  (`measure/gsm8k_tput.json` is `mech_nostream.json`);
* `max_num_seqs` is no longer pinned to 1 by the fold, so the throughput profile
  becomes available.

🔴 **The bank's value is that kt READS it, not that it is STREAMED.** And the
two TTFT improvements this file used to headline — loader threads 4→16, and
`stream_resident_from_hbm` — are streaming-vs-streaming. They are real (G′ is
worth −37…−41% of streaming's TTFT here, more than the −20.5% first measured),
but they improve a path that should not run. An internal control cannot see a
cost its two arms share; this is the third time that has bitten this work, after
the retracted TPOT numbers.

**Not overclaimed:** measured to 8192 only (`MAXLEN=10240` is the HBM ceiling
with 100 resident experts), one server per arm (irrelevant at 2–10× against a
±3–5% TTFT floor), and the ~4096 window is inferred from the doubling at 8192
rather than read from a counter.

### What the mechanism counters say, for the record

Deterministic, no noise floor (`measure/chain_mech.sh`, prompt 4096):

| arm | chunks | GiB staged | HOST MOVED | D2D calls | TTFT | TPOT |
|---|---|---|---|---|---|---|
| `nostream` | 0 | 0.0 | — | — | 7.14 s | 29.9 ms |
| `stream` | 2400 | 1344.0 | — | — | 19.63 s | 31.1 ms |
| `streamg` | 2400 | 1344.0 | 994.0 of 1344.0 (74.0%) | 3,200 | 15.60 s | 31.0 ms |
| `fold` | 2400 | 1344.0 | 994.0 of 1344.0 (74.0%) | **48,720** | 17.19 s | 26.0 ms |
| `stream4` (batch 4) | 2400 | 1344.0 | — | — | 83.01 s | 82.9 ms |

* **268.8 GiB per request** streamed whatever the prompt length (120 chunks ×
  67.2 GiB × 4 reports), `acquire` 88–91% of the staging loop.
* **G′ moves 26% fewer bytes** and that is exactly what the counter says.
* **The fold moves the SAME bytes in 15× the D2D calls** (48,720 vs 3,200),
  because `merge_runs` needs adjacent logical ids to be both resident and the
  hot set is scattered. 🔴 Choosing slots differently cannot fix that — the
  destination row advances only for adjacent ids.
* **The fold's hot set converges**: `decisions 80 (skipped {'no_change': 40})`,
  `experts staged 4000 (70.0 GiB copied)` — the whole resident set replaced on
  the folding request, no change on the next.
* **Streaming does run at a batch, and it is catastrophic**: four requests ×
  269 GiB ≈ 1 TiB, TTFT 83 s.


---

## ✅ 2026-11-19: the chunk lookahead was never switched on, and switching it on is the largest TTFT change in this work

`StreamSlotProvider.prefetch()` existed, was tested, and had **no production
caller**: `grep -rn '\.prefetch(' vllm_ascend/` found only tests.
`stream_chain.run_streamed_layer`'s loop was strictly
`slot_for(k) → run_chunk(k) → slot_for(k+1)`, so `_acquire_buffer` always fell
through to an inline synchronous `_fill` — the host gather sat on the critical
path with the NPU idle, and the NPU computed with the host threads idle.

`run_streamed_layer` now takes `prefetch_for` and starts chunk k+1's gather
before computing chunk k. Depth 1, which is exactly what
`DEFAULT_HOST_BUFFERS = 2` was built for, so **it costs no memory** (the pinned
staging was already `1.121 GiB` either way).

**The deterministic half, which no load can distort** — same bytes, same
gathers, same `HOST MOVED 49.7 of 67.2 GiB (74.0%)`:

| | acquire leg | share of the staging loop |
|---|---|---|
| serial (before) | **3.57 s** per bank pass | 88–91% |
| lookahead (after) | **0.10–0.19 s** | 40–82% of a much smaller total |

The 357–650 GiB/s the timer then reports is not the host going faster; it is the
blocking time disappearing — `slot_for` arrives to find the buffer already
filled, so the timer only catches the residual wait. That is the signature of
the overlap.

🔴 **One trap closed on the way in.** Under `set_resident`, `slot_for` builds the
buffer key from the STREAMED positions only; a `prefetch` that asked for the
whole chunk would build a DIFFERENT key, so every prefetch would miss, the
gather it paid for would be the full chunk, and
`stream_resident_from_hbm`'s 26% saving would be silently undone — while the
logs still looked right. `prefetch` therefore derives the subset itself, by the
same lines `slot_for` uses, and declines a fully resident chunk. Three tests
pin it (`test_prefetch_derives_the_streamed_subset_not_the_whole_chunk` asserts
the key carries `(0, 2)` and is **not** the whole-chunk key).

`VLLM_ASCEND_DSV41_KT_PREFETCH=0` restores the serial loop in the same binary,
and the server logs which path is live (`kt_offload stream: chunk lookahead
ON/OFF`).

### base vs everything, one chain, two arms — FIRST READ, host NOT quiet

`measure/chain_ab.sh`, prompt 4096, 10 requests per cell, loadavg 38–54.
`base` = `s2_off.json` (no bank, kt reads the checkpoint in ND, streaming at
the production threshold, fold on) with `KT_NZ_ROW_BLOCK=4`, `KT_NZ_G_OUTER=0`,
`VLLM_ASCEND_DSV41_KT_PREFETCH=0` — every change of this work off, in the same
binary. `now` = the full stack at its shipped defaults. Both arms self-identify
in the log (`layout=ND walk=row-outer` / `layout=NZ walk=GROUP-outer`).

| gen | arm | TTFT p50 | TTFT min | TPOT p50 | TPOT min |
|---|---|---|---|---|---|
| 128 | base | 39.40 s | 36.61 | 32.66 ms | 29.92 |
| 128 | **now** | **10.26 s** | **9.88** | **29.94 ms** | **27.24** |
| 128 | *README expectation* | *22.20 s* | — | *26.01 ms* | — |
| 512 | base | 40.13 s | 38.46 | 34.38 ms | 28.95 |
| 512 | **now** | **9.34 s** | **8.86** | **28.89 ms** | **26.16** |

TTFT **0.26× / 0.23×**; TPOT 0.917× (inside the ±12–14% cross-start floor) and
0.840× (outside it).

🔴 **Why this is a first read and not the number.** The host was at loadavg
38–54, and it hurt the two arms very differently: `base`'s TTFT came out
**1.8× worse than the documented expectation** (39.40 against 22.20) with its
`acquire` leg at **20.6 GiB/s** against the usual 58–78, while `now` measured
9.34 s against 9.32 s on a quieter run — i.e. **`now` is barely load-sensitive
and `base` is nothing but**. That is itself the mechanism (the gather is off the
critical path), but it means the RATIO is inflated. A quiet-window re-run of the
same chain is queued behind `measure/when_quiet.sh`.

🔴 **And one direction is opposite to the kernel measurement, unexplained.** The
kernel bench says ND is 8.64% faster than NZ, so `base`'s TPOT should be
slightly better. It is worse, by 8–16%. The leading candidate is that `base`'s
kt reads w13 in place from the **checkpoint mmap**, whose pages are evictable
page cache (it needs 267 GiB resident and the host had 312–368 GiB available),
while the bank lives in **tmpfs and cannot be evicted** — so under memory
pressure `base` may be paying major faults per decode step. That would make
"cannot be reclaimed" a real advantage of the bank, separate from speed. It is
**not established**: it needs fault counters, and the quiet re-run will say
whether the gap survives without memory pressure.

## The headline: TTFT 2.66x, TPOT unchanged (2026-11-21)

Paired on one host, one card, prompt 4096, gen 512, 10 requests per arm,
`max_num_seqs=1`, each arm in the host-memory state it would really have:

| | base (no bank, checkpoint source) | **now (full stack)** | |
|---|---|---|---|
| TTFT p50 | 23.62 s | **8.88 s** | **0.376x, 2.66x faster** |
| TPOT p50 | 24.12 ms | **24.08 ms** | **0.998x -- unchanged** |

```text
base ttft [22.79 22.98 23.02 23.53 23.61 23.63 23.77 23.77 24.05 24.10]
now  ttft [ 8.53  8.58  8.61  8.76  8.81  8.95  8.99  9.03  9.16  9.20]
base tpot [23.50 23.81 23.93 24.00 24.12 24.12 24.12 24.14 24.20 24.29]
now  tpot [23.81 23.87 23.92 23.92 24.01 24.14 24.18 24.26 24.30 28.35]
```

TPOT's two samples overlap almost completely; 0.2% is far inside the
+-12..14% two-server-start floor, so it reads as unchanged. `base` here
reproduces `README.md:314`'s documented 22.20 s baseline, which is what makes
the pair trustworthy.

Arm identity, from the deterministic counters (no noise floor):

```text
base:  chunk lookahead OFF   layout=ND  walk=row-outer    acquire 2.36s (94%, 28.4 GiB/s)
now:   chunk lookahead ON    layout=NZ  walk=GROUP-outer  acquire 0.08s (31%, 838.8 GiB/s)
       HOST MOVED 49.7 of 67.2 GiB (74.0%)
```

Reproduction from nothing: `vllm_ascend/kt_offload/REPRODUCING.md`.

### 🔴 Retraction: every earlier now/base ratio was invalid

0.260x, 0.233x, 0.145x, 0.144x, 0.136x, 0.133x and the "-26.9%/-30.5% TPOT"
beside them are **withdrawn**. The `base` arm had been measured while the
**bank was still resident**. `base` reads the 476 GiB checkpoint through its
page cache; the bank is 269 GiB of tmpfs, which **cannot be reclaimed**, so the
checkpoint could not stay cached and base's gather went partly to disk:

| base | bank resident | bank deleted |
|---|---|---|
| acquire | 4.1 / 6.2 / 20.6 GiB/s | 27.0 / 26.4 / 28.4 GiB/s |
| TTFT p50 | 60.79 / 68.08 / 76.65 s | **23.62 s** |

A fair pair therefore needs **two runs** -- `ARMS=base` with the bank deleted,
then rebuild it and `ARMS=now` -- because the arms' memory needs are
incompatible. Host load was not the main variable: the fair base run sat at
loadavg 14.5-27.7.

### The NZ layout's 8.64% does not reach end-to-end TPOT

The kernel bench measures ND beating NZ by 8.64%, so `base` should show the
slightly better TPOT. It does not: 24.12 vs 24.08 ms, inside the floor. The
end-to-end runs that appeared to show `now` 8-39% *better* at TPOT were the
same page-cache artifact. **The layout cost is real in the kernel and invisible
in the product**, because the CPU MoE is not the decode bottleneck.

## What a four-agent review changed (2026-11-21)

Four adversarial reviews ran over this branch and the kernel half: the streaming
prefetch path, the bank path, a minimality audit, and the AMX kernel. Every
finding below was re-verified here before being acted on, and two of the reviews'
claims were rejected.

### Two defects outranked everything else, and both were in this work

**The lookahead crashed any server with `stream_layer_buffer=true`.**
`layer.py` put `prefetch_for` into `run_kwargs` unconditionally and then chose
`run_streamed_layer_buffered`, which does not accept it. Verified by
introspection rather than by reading:

```text
only run_streamed_layer accepts: ['expert_tokens_num_type', 'prefetch_for']
only buffered accepts:          ['provider']
```

`TypeError` on the first streamed prefill, inside the MoE forward, while the
start-up log said `chunk lookahead ON`. Stage 1 is **deleted** rather than
patched: it contributed nothing to the 8.88 s figure (that run has the switch
unset and logs zero calls into the buffered runner), it had never run end to
end, and turning it on cost HBM instead of saving it -- the fold keeps its own
1.75 GiB staging, so the card would hold ~10 GiB of an 11.41 GiB budget.

**The NZ geometry guard never ran in the shipped configuration.**
`NZ_DEPENDENT_SWITCHES` held only `("stream_layer_buffer",)`, while the route
that has the CPU kernel read these bytes in place is `kt_reads_bank`. The logs of
the measurement above prove it:

```text
nz_geometry: self-check off   x40
PASS lines:                     0
```

895 lines written for one route's silent failure mode, not protecting that
route. The tuple is now `("bank_dir", "kt_reads_bank")`.

### 🔴 The 8.64% kernel figure was measured against a control this work changed

The ND walk had `constexpr int prefetch_groups = 40` with a
`g + pfd < group_count` guard, so on the down projection (36 groups) it never
prefetched and on gate/up (72) it prefetched at 40. That had become
`min(40, group_count/4)` -- 9 and 18 -- while two comments a few lines away still
read "the ND arm is what the control measurements use. An unmeasured change to
the control is not a free change." Restored to the literal 40. **The 8.64% is
now NZ against the ND arm that shipped before this work**, which is what it was
always claimed to be.

### Verified after all of it

`7/7 streaming prompts token-identical => PASS`, against the same stage-3
reference capture as before, with the kernel rebuilt: ND restored, the
`g_unroll` / `pf_dedup` / `legacy_stride` knobs deleted, `g_dispatch` down from
7 instantiations to 2, and the n-outer chain down to R=4. The build tag moved
with the code -- `ND_MIN40` to `ND40` -- because a tag that misdescribes the
binary is worse than no tag.

### Rejected

* "`_tables` costs ~33 s of table building per request." The 69.4 ms per new
  subset key and the unbounded growth are real, but 33 s cannot hide inside an
  8.88 s TTFT, and the fold commits its hot set **once** per server, so the key
  set stabilises. Kept as the leading explanation for something previously
  unexplained: the first request of a server costs 21.33 s against 8.88 s
  steady state.
* "Delete `VLLM_ASCEND_DSV41_KT_LOADER_THREADS` as a pure A/B lever." A second
  review found that `wiring.py` passes `copy_threads=16` and `one_per_l3_cpus`
  **raises** on a host with fewer than 16 L3 domains, so lowering this is the
  only way to start the server there. It ships; its documentation, which named a
  default production never uses, is fixed.

### 🔴 Breaking: five config fields were removed

`stream_layer_buffer`, `stream_layer_sync_scope`, `stream_layer_probe_bytes`,
`debug_time_chunks`, `debug_hold_hbm_bytes`. `KTOffloadConfig` rejects unknown
keys, so **a deployment whose JSON still sets any of them will not start**:

```text
pydantic_core.ValidationError: 1 validation error for VllmConfig
kt_offload_config.debug_time_chunks
  Unexpected keyword argument [input_value=True]
```

That is not hypothetical -- it is how the first regression run failed, because
29 of this project's own configs set `debug_time_chunks` and `REPRODUCING.md`
printed it in a recipe described as "the full stack, verbatim".
