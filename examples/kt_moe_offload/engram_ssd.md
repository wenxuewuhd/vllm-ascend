# DeepSeek-V4.1 Engram on SSD

The optional `ssd` backend maps the four Engram weight/scale tables read-only.
CPU gather, pinned staging, H2D copies and dequantization remain unchanged.
The default is still `ram` with `/dev/shm/engram_stage`.

## Prepare and serve

After configuring the dependencies in the [deployment guide](README.md), run
from the repository root on Linux. Replace RAM table staging with:

```bash
export MODEL=/path/to/DeepSeek-V4.1-Flash
export SSD_STAGE=/path/on/ssd/engram
python3 examples/kt_moe_offload/prepare_engram_ssd.py \
  --checkpoint "$MODEL" --output "$SSD_STAGE"

# Optional full re-verification; reads all source and destination table bytes.
python3 examples/kt_moe_offload/prepare_engram_ssd.py \
  --checkpoint "$MODEL" --output "$SSD_STAGE" --verify

export VLLM_ASCEND_DSV41_ENGRAM_BACKEND=ssd
export VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR="$SSD_STAGE"
bash examples/kt_moe_offload/serve_lowlatency.sh
```

Preparation needs an empty disk-backed directory and approximately 188.83 GiB
for the tested Flash checkpoint, on a filesystem supporting hard links (e.g. XFS).
It streams with an 8 MiB buffer, verifies SHA256
by independent readback and publishes read-only files without replacing existing
paths. The manifest is published last; inspect interrupted output before retrying.

Use an absolute path visible inside the serving container, with a read-only
mount. SSD mode rejects tmpfs/ramfs; identify the physical backing device
separately. Startup validates the checkpoint index/headers and the exact opened
table descriptors against the manifest without scanning table payloads. Copying
or replacing files invalidates their identities; prepare a new verified directory
rather than editing the manifest. External writes/truncation are unsupported.

Mappings are fixed before graph capture: restart to switch backends. Unset both
variables for the legacy RAM default, or explicitly select `ram` and an absolute
RAM stage directory. These variables do not enable DSpark or change graph settings.

## Cache-aware measurement

This is ordinary `mmap`, with Linux page cache participating. There is no
`O_DIRECT`, full-table pinning or hard cache budget. Preparation/full verification
warms the cache; even a zero-residency startup can gain cached pages through
verification requests and throughput calibration.

Keep checkpoint, workload, graphs and CPU/NUMA placement fixed. Record table page
residency before startup, before the throughput client and afterward, along with
NPU process ownership and host pressure throughout calibration and measurement.
Report client TPOT and native decode throughput separately; retain SSE text-chunk
length warnings and cross-check actual generated-token counters.
