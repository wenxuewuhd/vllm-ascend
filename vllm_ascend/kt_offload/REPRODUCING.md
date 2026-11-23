# Reproducing TTFT 8.88 s / TPOT 24.08 ms from nothing

Measured 2026-11-21 00:50–01:40 UTC, card 3, prompt 4096, gen 512, 10 requests,
`max_num_seqs=1`. The numbers this produces, and their spread over those ten
requests:

```text
TTFT  p50 8.88 s   min 8.53   max 9.20     (spread 0.67 s)
TPOT  p50 24.08 ms min 23.81  max 24.30    (spread 0.49 ms)
```

For reference, `README.md:314` documents the project's expectation at the same
prompt (gen 128, quiet host): **TTFT p50 22.20 s, TPOT p50 26.01 ms**, and
TPOT's own min there is 25.85 ms — above everything this configuration produced.

This figure has been reproduced **five times across loadavg 10 to 54**
(8.4–9.3 s TTFT, 23.8–26.9 ms TPOT). It is insensitive to host load, which is
the point of step 6 below.

---

## 0. What must already be true

| | |
|---|---|
| checkpoint | `/workspace/models/DeepSeek-V4.1-Flash` (476 GiB) |
| host memory | 755 GiB total. The bank takes 269 GiB of `/dev/shm` and tmpfs **cannot be reclaimed** |
| card | one Ascend950PR with ~131 GiB HBM, free (`npu-smi info -t proc-mem -i <card>` says `No process in device`, or only a few-hundred-MB foreign context) |
| trees | `dsv41/vllm-ascend` at branch `dsv41-bank-ttft-20261115`; `dsv41/ktransformers` detached at `105359b` with the working-tree patch from `backups/gouter_20261117/` |

🔴 **`dsv41/ktransformers` must never be committed to.** Its changes live only
in the working tree and in `backups/gouter_20261117/` (`fp4-moe.hpp`,
`ext_bindings.cpp`, `kt_unstaged.diff`, `HEAD.txt`).

---

## 1. Build the kernel

```bash
bash measure/kt_rebuild.sh        # ~95 s; refuses to run while a server is up
```

Verify — the script prints it, and the number must be 1:

```text
KT_PREFETCH_NZ16_SHORT4_NTA_RB8_GOUTERON_ND40_G4_STRIDED occurrences in the .so: 1
=== KT_REBUILD_DONE
```

That tag encodes the four kernel changes that matter: prefetch distance 16 long
/ 4 short, hint NTA, **row block 8**, and the **rebuilt group-outer m=1 inner
loop**. A build without `RB8_GOUTERON` in the tag is not this configuration.

## 2. Build the bank

```bash
python3 measure/build_bank.py --phase full      # ~5.6 min, 269 GiB
```

Verify:

```text
[full] built 40 layer(s) in ~336s
du: 269G  /dev/shm/dsv41_nz_bank
verify_layer(0|20|39): bytes_compared 94003200     <- byte-for-byte against the checkpoint
layers in manifest: 40
```

🔴 **Redo this after every reboot.** `/dev/shm` does not survive one, and this
host rebooted three times in four days (2026-11-16, 11-18, 11-19). Every chain
under `measure/run_retry.sh` now preflights it and refuses in one second rather
than failing six minutes into a server start.

## 3. The configuration

`measure/mech_fold.json` — the full stack, verbatim:

```json
{"kt_offload_config": {
  "enabled": true, "resident_strategy": "prefix", "num_resident_experts": 100,
  "cpuinfer_threads": 96, "threadpool_count": 2, "numa_nodes": [0, 1],
  "packed_transfer": false, "stream_prefill": true, "dynamic_resident": true,
  "zerocopy_w13": true, "zerocopy_node_fraction": [1.0, 1.0],
  "zerocopy_streaming_node": 1,
  "bank_dir": "/dev/shm/dsv41_nz_bank", "zerocopy_scope": "all",
  "kt_reads_bank": true, "stream_resident_from_hbm": true,
  "stream_prefill_min_tokens": 64
}}
```

What each of the four non-default switches does:

* `bank_dir` — streaming prefill reads the bank instead of the checkpoint.
* `zerocopy_scope: "all"` + `kt_reads_bank` — kt's CPU MoE consumes w13 **and**
  w2 in place from the bank, in FRACTAL_NZ. This is what the kernel work acts
  on; without it `nz_layout` is false and none of it executes.
* `stream_resident_from_hbm` (G′) — the chunk buffer's already-resident slots
  are filled by device-to-device copy instead of being gathered from the host.
  Worth 26% of the host bytes: `HOST MOVED 49.7 of 67.2 GiB (74.0%)`.
* `stream_prefill_min_tokens: 64` — low so that every request of this size
  really streams. **Production ships 512**; at the production threshold a
  4096-token prompt still streams, but a GSM8K-length one does not.

## 4. Start the server

```bash
CARD_OVERRIDE=3 MAXSEQS=1 MAXLEN=10240 KT_FULLSET_LOAD=1 GPUUTIL=0.80 \
  KT_CONFIG="$(cat measure/mech_fold.json)" bash measure/launch.sh myrun
```

`MAXSEQS=1` is required: `dynamic_resident` refuses any other value, because the
hot set is committed once at the end of a prefill and several interleaved
prefills have no such moment. `MAXLEN=10240` fits a 4096 prompt plus 512 output
with room; `MAXLEN=16384` OOMs this card at 100 resident experts.

Ready in **~420–480 s**. Then verify the five lines below; **if any is wrong the
numbers mean nothing**:

```text
kt_offload stream: chunk lookahead ON (depth 1, provider.prefetch)
[kt prefetch] KT_PREFETCH_...RB8_GOUTERON...: gate/up layout=NZ ... walk=GROUP-outer
kt_offload bank-as-kt-source: 384 experts/layer, 18800640 B each
kt numa_placement: ... on_target=100.00% over 70502400 page(s)
kt_offload stream I/O: 120 chunk(s), 67.2 GiB staged, 0.2s total   <- after the first request
```

* `lookahead ON` — the chunk prefetch. OFF means the host gather is back on the
  critical path and TTFT will be ~15 s instead of ~9 s.
* `layout=NZ` + `walk=GROUP-outer` — kt is reading the bank with the rebuilt
  loop. `layout=ND` means `kt_reads_bank` did not take and the kernel work is
  not running.
* `on_target=100.00%` — the bank's pages are on their planned NUMA node.

## 5. Measure

```bash
python3 measure/bench_raw.py --port 8100 --label myrun \
  --out measure/out/myrun.json --mode pure --concurrency 1 \
  --requests 10 --prompt-tokens 4096 --gen-tokens 512
```

Discard nothing and report the median, min and max of all ten: `ttft_s_all` comes
back **already sorted**, so the start-up transient cannot be identified from it
and a `sorted(...)[2:]` "discard" drops the two FASTEST samples instead (it
biased an earlier median upward by 4%).

One request per server start does pay a one-off cost: the folded hot set commits
once, ~35 GiB, visible as a single `committed 40 layer(s)` line and one TTFT
sample around 21 s. With ten requests it sits in the max, not the median.

## 6. What this does NOT need, and what it does

**Does not need a quiet host.** This configuration was 8.4–9.3 s at loadavg 10,
28, 31, 34–46 and 38–54. The chunk lookahead takes the host gather off the
critical path, and the bank lives in tmpfs where it cannot be evicted, so
neither host contention nor page-cache pressure reaches it.

🔴 **Does need the bank gone if you are also measuring the no-bank baseline in
the same session.** The bank's 269 GiB of tmpfs is unreclaimable and starves the
checkpoint's page cache, which the no-bank arm's gather depends on — its
`acquire` leg was measured at 4.1, 6.2 and 20.6 GiB/s against a normal 58–78.
Every `now/base` ratio taken with the bank resident is therefore invalid. Order
the arms as: **remove the bank → measure base → rebuild the bank → measure now.**

🔴 **The staging report's `acquire` leg no longer measures host throughput.**
With the lookahead on it reports 0.1–0.2 s at "357–650 GiB/s", which is not the
host going faster — it is the blocking time disappearing, because `slot_for`
arrives to find the buffer already filled. The real gather still moves ~199 GiB
per request (84 reports × 67.2 GiB over 22 requests ≈ 257 GiB staged, 74% of it
from the host); it now runs on the prefetch thread, which has no timer. To
measure the host leg again, instrument that thread.

## 7. Correctness, if you want to re-check it

```bash
bash measure/chain_rb8_eq.sh        # 7 prompts x 128 tokens, token-identical
ARM=tput CARD=3 MAXSEQS=4 NSAMP=1319 MAXLEN=5120 \
  bash measure/run_retry.sh chain_gsm8k.sh gsm8k_tput 2    # GSM8K, ~83 min
```

Expected: `7/7 streaming prompts token-identical => PASS` (seen four times), and
GSM8K **97.04** against the published 97.27 — 3 questions, 0.49 standard errors
at n=1319.
