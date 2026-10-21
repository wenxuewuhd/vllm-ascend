# DeepSeek-V4.1-Flash on a single Ascend card, MoE experts offloaded to CPU

The model's routed experts do not fit in one card's HBM. This path keeps a
prefix of the experts resident on the NPU and runs the rest on the host CPU
through [kt-kernel](https://github.com/kvcache-ai/ktransformers), so one card
serves a model that otherwise needs four.

Follow the steps in order; each one depends on the previous. At a glance:

```
1  container image        check yours against the versions listed
2  source trees           vllm (fork, pinned) + this repo
3  build artifacts        vllm_ascend_C*.so, custom vendor, _version.py
4  ds41 operator wheels    --target a directory of their own
5  engram tables          189 GiB copied into /dev/shm  (3-5 min)
6  kt-kernel              build + install into its own directory
7  run                    serve_lowlatency.sh | serve_throughput.sh
8-10  check, measure, stop
```

Steps 3 to 6 are one-time per machine. Step 5 is redone after every host reboot
or container rebuild, because /dev/shm is RAM.

## 0. What you need

| | |
|---|---|
| Hardware | 1 × Ascend A5 (Ascend950PR), 128 GB HBM; host RAM as below, across 2 NUMA nodes |
| Model | DeepSeek-V4.1-Flash, MXFP4 weights (~475 GiB on disk) |
| Disk | The weights are read through the page cache, so leave room for them to stay cached |

Host RAM, for the default configuration in the launch script (validated on a
755 GiB machine with 377/378 GiB per node):

| | |
|---|---|
| Engram tables in `/dev/shm` (tmpfs, i.e. RAM) | **189 GiB** |
| Offloaded expert weights | **~370 GiB** (≈135 on node0, ≈235 on node1) |
| Floor | **~560 GiB** |
| Wanted on top | whatever is left, as page cache for the checkpoint |

🔴 The per-node split matters, not just the total. kt binds each worker thread
to one NUMA node with `MPOL_BIND | STRICT`, so a node that fills up gets an
allocation failure rather than spilling to the other node — with plenty of free
memory elsewhere on the machine.

## 1. Container image

Validated inside `quay.io/ascend/vllm-ascend:nightly-main-a5`.

🔴 That is a **nightly tag and it moves.** Pulling it today gives you a
different image than the one these numbers were taken on, so check what you get
against the versions below rather than trusting the tag:

```bash
cat /etc/os-release | head -2                # Ubuntu 24.04
python3 -V                                   # 3.12.13, system python, no venv
cat /usr/local/Ascend/driver/version.info    # Version=25.7.rc1.6, ascendhal 7.35.23
python3 -c "import torch, torch_npu, triton; print(torch.__version__, torch_npu.__version__, triton.__version__)"
                                             # 2.10.0+cpu  2.10.0.post4.dev20260715  3.2.0
npu-smi info -t usages -i 0 | grep 'HBM Usage'   # the card is idle
```

Those are the versions this path was validated on, read off the validated
machine. `torch_npu 2.10.0.post4.dev20260715` is a dev snapshot.

🔴 **Do not let any `pip` command touch `torch` or `torch_npu`.** The validated
`torch_npu` build is a dev snapshot that is no longer downloadable; a dependency
resolver that "upgrades" it leaves the environment unrecoverable. Every install
below therefore uses `--no-deps --no-index`.

### CANN 9.2

The image above ships **CANN 9.1**, and this path needs **9.2.0-beta.2** at
runtime. Install it alongside 9.1 rather than over it, and point `CANN` at its
root (the directory holding `set_env.sh`); it is about 12 GB.

```bash
export CANN=/path/to/cann-9.2.0-beta.2
ls "$CANN/set_env.sh"          # must exist; the launch scripts source it
```

The exact version is **`cann-9.2.0-beta.2`**, published by Ascend; download it
the way you normally obtain CANN. This path was validated against a copy of an
already-installed 9.2.0-beta.2 tree rather than a fresh install, so what is
asserted here is the version and that it resolves at runtime — not a particular
installation procedure.

🔴 **9.1 has to be stripped off the paths, not merely overridden.** 9.2's own
`set_env.sh` removes only its own entries, so a 9.1 left on `PATH` /
`LD_LIBRARY_PATH` / `PYTHONPATH` keeps winning — and the container's
`~/.bashrc` and `/etc/profile` both source 9.1's. `_common.sh` strips 9.1 by
path component before sourcing 9.2. The failure when it is not stripped reads
as "this operator does not exist" or `EZ1009`, which points at the operator
rather than at the toolkit.

Check that 9.2 is what resolves:

```bash
source "$CANN/set_env.sh"
echo "$ASCEND_TOOLKIT_HOME"     # must be under $CANN
command -v bishengir-compile atc  # both must resolve under $CANN
```

## 2. Source trees

```bash
# vllm 0.27.1 plus DeepSeek-V4.1 fixes that are not upstream yet
git clone https://github.com/wenxuewuhd/vllm.git vllm
git -C vllm checkout a9686bc6e6

git clone https://github.com/wenxuewuhd/vllm-ascend.git vllm-ascend
git -C vllm-ascend checkout dsv41-moe-offload
```

🔴 **Not upstream `vllm-project/vllm`.** The pinned commit carries five
DeepSeek-V4.1 bug fixes on top of 0.27.1 — the engram hash prologue, the engram
lookback sizing, `query_start_loc` ordering, the (block, offset) slot pairs, and
one breakable-cudagraph replay. Upstream at the same version starts and then
fails or answers wrongly inside the engram path.

Both trees are used from `PYTHONPATH`, not installed. If the image already has
an editable `vllm`/`vllm-ascend`, these shadow it.

## 3. vllm-ascend build artifacts

`vllm_ascend/` needs three gitignored products that a fresh clone does not have:

```
vllm_ascend/vllm_ascend_C*.so
vllm_ascend/_cann_ops_custom/vendors/custom_transformer/
vllm_ascend/_version.py
```

**`vllm_ascend_C*.so`** — build in-place under CANN 9.2:

```bash
COMPILE_CUSTOM_KERNELS=1 SKIP_ACLNN_BUILD=1 python3 setup.py build_ext --inplace
```

**`_cann_ops_custom/vendors/custom_transformer/`** — the custom operator
package. Two ways:

```bash
# A. copy a real directory from the image's own vllm-ascend, if it has one
cp -a /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors       vllm-ascend/vllm_ascend/_cann_ops_custom/

# B. build it (~26 min, needs catlass >= v1.6.2), IN A SEPARATE TREE
csrc/build_aclnn.sh <root> ascend950
```

🔴 **Build it in a separate tree.** `csrc/build_aclnn.sh` starts with
`rm -rf $ROOT_DIR/vllm_ascend/_cann_ops_custom`, and if `ROOT_DIR` points at the
tree you are serving from it deletes the package you are using. Build elsewhere,
then copy the result in.

🔴 **It must be a real directory, not a symlink**, if you set
`ASCEND_CUSTOM_OPP_PATH` by hand — a symlink there gives `EZ1009`. (The symlink
*inside* the package, `op_tiling/liboptiling.so`, is normal; leave it alone.)

A package built with 9.2.0-beta.2 has 786 files. One built with 9.1 has 759 and
is a different build, not an older copy of the same one.

**`_version.py`** — copy it from a tree that has one, or let `setup.py`
generate it.

Without `vllm_ascend_C*.so` the server fails at import. Without the custom
vendor it gets much further and then fails **inside an operator** with `EZ1001`,
which does not name the missing vendor -- so check for it before blaming the
model.

## 4. ds41 operator package — required

Three wheels, obtained from your Ascend vendor alongside the model -- they are
not on PyPI and are not part of CANN. The versions this was validated with:

```
cannbotdsl-0.5.0-cp312-cp312-manylinux_2_28_x86_64.whl
cann_ops_transformer-1.0.0-py3-none-any.whl
cannbot_arena_net_ops-0.1.0-cp312-cp312-linux_x86_64.whl
```

```bash
WHEELS=/path/to/the/three/wheels
DS41_SITE=$PWD/ds41_site        # 🔴 on disk, not /dev/shm -- tmpfs is gone
                                #    after a host reboot
python3 -m pip install --no-deps --no-index --target "$DS41_SITE" \
  "$WHEELS"/cannbotdsl-*.whl \
  "$WHEELS"/cann_ops_transformer-*.whl \
  "$WHEELS"/cannbot_arena_net_ops-*.whl
```

`--no-deps --no-index` keeps pip from reaching out and, more importantly, from
"upgrading" `torch` or `torch_npu`: the validated `torch_npu` is a dev snapshot
that is no longer downloadable, and a resolver that replaces it leaves the
environment unrecoverable.

🔴 **`--target` is required, not tidiness.** CANN ships its own package named
`cann_ops_transformer`, and `set_env.sh` puts it at the head of `PYTHONPATH`. A
default `pip install` lands behind it: `import` then succeeds, the ds41
operators are simply **absent**, and nothing is reported. The launch script
verifies the resolution order and refuses to start if CANN's copy wins.

> The `.run` self-extracting archive shipped next to those wheels installs a
> single operator (`compressor_v2`) into `opp/vendors/` and **overwrites**
> several OpTypes this path uses. This path does not call it. Do not install it.

All three wheels are needed: `cann_ops_transformer` is the torch binding layer,
`cannbot_arena_net_ops` (whose top-level package is the bare name `ops`) holds
the precompiled kernels, and `cannbotdsl` does the ahead-of-time lookup and
loading at **run** time, not only at build time.

**Check the install, because a bad one is silent.** Without these operators the
candidate-selection layers fall back and `max_model_len` stays capped at 16384 —
which the throughput profile below exceeds — and the server prints one warning
and **starts anyway**. Two conditions, both required:

```python
import importlib.util, torch
print(importlib.util.find_spec("cann_ops_transformer").origin)   # must be under $DS41_SITE
print(len(torch._C._jit_get_schemas_for_operator("cann_ops_transformer::ds41")))  # must be 8
```

⚠️ Do **not** check by counting `dir(cann_ops_transformer.ops.ds41)`. There is no
`ops/ds41/` directory on disk; the module is built at import time and exposes
only 7 public names, the eighth living on `torch.ops` alone. Counting gives a
false red.

⚠️ `import cann_ops_transformer` rebuilds `~/.cache/torch_extensions`, which is
where the server loads from. If you are experimenting, point
`TORCH_EXTENSIONS_DIR` somewhere of your own first. Do not set it by hand when
starting the server — `_common.sh` gives each tree its own directory.

## 5. Engram tables — required

Two of the model's embedding tables (layers 1 and 14, **188.8 GiB together**) do
not go to the device. The runtime mmaps them read-only from `/dev/shm/engram_stage/`
(the path is fixed in `vllm_ascend/models/deepseek_v4_1/engram_staging.py`), and
**refuses to start if they are absent or the wrong size** — the loader compares
against the byte counts in the checkpoint header.

Four files must be present:

```
layers.1.engram.embed.weight.bin    98,305,579,008 B
layers.1.engram.embed.scale.bin      3,072,049,344 B
layers.14.engram.embed.weight.bin   98,308,270,592 B
layers.14.engram.embed.scale.bin     3,072,133,456 B
```

Each is the identically named tensor copied **verbatim** out of the checkpoint
(`layers.{1,14}.engram.embed.{weight,scale}`, which the index maps to the last
two shards). Copy the bytes, do not go through a tensor library — the loader
compares byte counts, and a dtype conversion changes them:

```python
import json, mmap, os, struct
MODEL, STAGE = "/path/to/DeepSeek-V4.1-Flash", "/dev/shm/engram_stage"
os.makedirs(STAGE, exist_ok=True)
wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
for key in ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale",
            "layers.14.engram.embed.weight", "layers.14.engram.embed.scale"):
    path = f"{MODEL}/{wm[key]}"
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
        beg, end = hdr[key]["data_offsets"]
        f.seek(8 + hlen + beg)
        with open(f"{STAGE}/{key}.bin", "wb") as out:
            left = end - beg
            while left:                       # 1 GiB at a time; these are ~92 GiB each
                chunk = f.read(min(left, 1 << 30))
                out.write(chunk); left -= len(chunk)
    print(key, os.path.getsize(f"{STAGE}/{key}.bin"))
```

Staging takes about 3-5 minutes and **permanently occupies 189 GiB of host RAM**
while the service runs. Check the sizes against the table above before starting
the server — the loader will refuse, but it refuses after loading the model.

🔴 `/dev/shm` is tmpfs: the tables are gone after a host reboot or a container
rebuild, and must be staged again. A directory that exists but holds truncated
files is the trap to watch for — check the byte counts, not just the names:

```bash
ls -l /dev/shm/engram_stage/
du -sb /dev/shm/engram_stage     # 202,758,032,400 B when complete
```

> If staging fails because huge pages cannot be obtained, defragment
> (`echo 1 > /proc/sys/vm/compact_memory`) and retry; the tables themselves are
> not the problem.

## 6. kt-kernel

```bash
git clone https://github.com/wenxuewuhd/ktransformers-AK.git kt-kernel
git -C kt-kernel checkout dsv41-singlecard-offload
cd kt-kernel/kt-kernel
```

That branch is `v0.7.0.post1` plus the changes this path needs. kt-kernel is a
separate project, so the work is carried there rather than vendored here. What
it adds:

| Provides | Read by |
|---|---|
| `KT_ZEROCOPY_WEIGHTS`, `KT_ZEROCOPY_SCOPE` | `operators/zerocopy_weights.hpp`, `operators/amx/fp4-moe.hpp` |
| `KT_FULLSET_LOAD` | `python/utils/amx.py` |

🔴 Take the whole branch, not the two features alone. The rest of it (expert
subset allocation, CCD-striped worker binding, the graph-capture probe, the
CPU-only drain) is default-off or behaviour-preserving, but the configuration
these instructions were validated on had all of it.

⚠️ The binary the numbers in this directory were measured on was built from a
newer upstream base than that branch. The differences are unrelated to the
offload path — a GPTQ-Int4 packed backend and one `__AVX512BF16__` guard — and
the zero-copy and full-set-load sources match line for line, but the branch is
not byte-equivalent to what was measured.

Build prerequisites: a C++20 compiler with AVX-512 (AMX preferred), OpenMP,
`pkg-config` and `libhwloc-dev`. Then:

```bash
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --no-index --target "$KT_SITE" dist/kt_kernel-*.whl
```

That branch adds what the two optional switches need:

| Switch | Needs | If missing |
|---|---|---|
| `zerocopy_w13` | `KT_ZEROCOPY_WEIGHTS`, `KT_ZEROCOPY_SCOPE` | server refuses to start |
| `dynamic_resident` | `KT_FULLSET_LOAD` | server refuses to start |

🔴 The refusal is deliberate. Both switches are plain environment variables to
kt, so a stock kt **ignores them without a word**: `zerocopy_w13` would
silently hold a full private copy of the weights (~2/3 more anonymous memory
than the budget assumed, surfacing later as an OOM with nothing pointing back),
and `dynamic_resident` would commit nothing while reading exactly like "the
heat said no change". The config probes the kt that this process actually
imported rather than trusting that the variables were exported.

Set both to `false` if you do not want that branch. The base offload works
without them.

## 7. Run

Two profiles, because the switches that buy time-to-first-token on long prompts
do nothing for short ones. Pick by what the workload looks like:

| | `serve_lowlatency.sh` | `serve_throughput.sh` |
|---|---|---|
| `--max-num-seqs` | 1 | 8 |
| streaming prefill + dynamic resident experts | **on** | off |
| SWA block override | **128** ⚠️ | **128** |
| `--max-model-len` | 8192 | 32768 |
| use it for | long prompts, one at a time | many short requests |

```bash
export MODEL=/path/to/DeepSeek-V4.1-Flash
export CANN=/usr/local/Ascend/cann-9.2.0-beta.2
export VLLM_TREE=$PWD/vllm ASCEND_TREE=$PWD/vllm-ascend
export DS41_SITE=$PWD/ds41_site KT_SITE=$PWD/kt_site
export CARD=0 PORT=8000

bash examples/kt_moe_offload/serve_lowlatency.sh    # or serve_throughput.sh
```

Each script's header lists the measurements taken **on that exact
configuration** — they are not interchangeable. The 3.60x time-to-first-token
belongs to the low-latency one; the GSM8K and GPQA scores belong to the
throughput one.

Both capture graphs. `--enforce-eager` costs 3.7x on decode here (22.99 ms per
output token against 85.24 ms, a single variable changed ten minutes apart), and
every number quoted in this directory — latency and accuracy alike — was taken
with graphs captured, so an eager run is not comparable to any of them.

The SWA block override is on in both. It was measured separately, with
streaming prefill and dynamic resident experts on and only the override
differing: the same 30 prompts gave **bit-identical token ids, 30/30**. Its
capacity effect depends on `max_model_len` — 3.85x at 8192, 3.58x at 32768 —
so neither figure carries to a third setting.

**Start-up is slow and that is expected.** Model load is a few minutes; with
`zerocopy_w13` the weights are then faulted in and migrated to their NUMA node,
which adds about 3-4 minutes (measured: ~400 s to ready). A run that looks hung at "loading weights" is
usually still doing that.

### Experimental single-card DSpark

DeepSeek-V4.1-Flash includes three DSpark draft layers in the target checkpoint's
`mtp.*` tensors. Each draft layer keeps all 128 routed experts in HBM; its
`kt_offload` is `None`. The target's 40 layers continue to use the configured
100 resident expert slots and CPU offload. The draft tensors total about
7.39 GiB, including about 6.72 GiB of expert weights.

After completing the setup above, select an idle physical card and use the
low-latency profile with these additional arguments:

```bash
export CARD=4  # choose an idle physical device on your host
npu-smi info -t proc-mem -i "$CARD"

export EXTRA_ARGS='--language-model-only --no-async-scheduling --compilation-config {"cudagraph_capture_sizes":[6],"max_cudagraph_capture_size":6} --speculative-config {"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}'
bash examples/kt_moe_offload/serve_lowlatency.sh
```

This configuration uses one request at a time, a maximum model length of 8192,
bf16 KV, eager draft execution, and a six-token target verification graph.
Async scheduling and multi-card DSpark have not been validated for this path.

Check for three startup messages containing
`DSpark draft expert placement: local=128 logical=128 kt_offload=None` and
inspect actual request counters after generation:

```bash
curl --noproxy '*' -s "http://127.0.0.1:$PORT/metrics" \
  | sed -n '/^vllm:spec_decode_.*_total/p'
```

The 2026-10-21 Ascend950 run passed five generation smoke checks: arithmetic,
Chinese text, streaming output, a 2101-token prefill, and 192-token generation.
However, **254 drafts proposed 1270 tokens and only 2 were accepted (0.1575%)**;
per-position accepted counts were `[2, 0, 0, 0, 0]`. This establishes execution
and counter activity, but does not establish useful acceleration or accuracy
equivalence. The very low acceptance rate remains unresolved. The non-DSpark
latency and accuracy results elsewhere in this document do not apply to it.

On a shared host, release only the Engram staging files and mounts owned by
your run. The smoke check reused existing Engram tables through a read-only
mount; after stopping its service and container, the device and port were free
and the original shared files were unchanged.

## 8. Check it is actually serving

`Application startup complete` is printed before the model can answer. Ask it:

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dsv41",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 64, "temperature": 0
}' | python3 -m json.tool
```

## 9. Measure it

The server is started once and measured from another window; nothing here
restarts it.

```bash
PORT=8000 CONCURRENCY=1 REQUESTS=5 PROMPT_TOKENS=4096 GEN_TOKENS=128 \
  bash examples/kt_moe_offload/bench_latency.sh
```

On the low-latency profile, one card, 4096 in / 128 out at concurrency 1:

```
TTFT  mean 21.40 s    TPOT  mean 22.57 ms   (5 requests, TPOT spread 0.22 ms)
```

Time-to-first-token barely moves with prompt length — 19.8 s at 1024 tokens,
21.4 s at 4096, 21.9 s at 8184 — because streaming prefill passes the expert
weights once per prefill and that fixed cost dominates. Do not extrapolate it
from a token count.

The prompt is English prose, cycled and trimmed to the exact token count you
ask for; `PROMPT_FILE=/path/to/text` substitutes your own. For this model that
choice is not cosmetic. Routed experts are picked per token, so a prompt built
from one repeated filler token reaches a far narrower set of experts than real
text does — and a narrow set is a cheap one here, because more of the work stays
on the experts resident in HBM instead of falling through to the CPU. Filler
prompts therefore flatter TPOT. vLLM's own `random` dataset errs the other way,
walking consecutive vocabulary ids. Prose sits between them; your own traffic
is better still.

🔴 A single reading is an order of magnitude, not a measurement: the machine
this was developed on drifts 2.5x across time windows at the same commit. Run
both arms of any comparison interleaved, in the same window.

## 10. Stop

```bash
pkill -f 'vllm serve .* --port 8000'
```

Then wait for the card to drain before starting anything else — host memory is
returned in seconds but HBM takes considerably longer, and a restart during that
window fails with an out-of-memory error that reads as "this configuration does
not fit".

```bash
until [ "$(npu-smi info -t usages -i 0 | awk -F: '/HBM Usage Rate/{gsub(/ /,"",$2);print $2}')" -le 6 ]; do sleep 5; done
```

### Releasing the engram tables

Stopping the server does **not** release them. They live in `/dev/shm`, which is
RAM, and they stay there holding **188.8 GiB** until something removes them —
which is deliberate, so that restarting the server does not pay the staging cost
again. Check and remove:

```bash
du -sb /dev/shm/engram_stage          # 202,758,032,400 B when complete
rm -rf /dev/shm/engram_stage          # frees 188.8 GiB immediately
```

🔴 **Remove them only when you are finished with the model on this host.** The
next start refuses to run without them, so removing costs a re-stage (3-5
minutes of copying, and the checkpoint has to be readable again).

🔴 **Re-staging does not reproduce the same memory layout.** tmpfs pages are
allocated on whichever NUMA node the writing thread happened to run on, so the
split across nodes is whatever first-touch gave you that time, and it is not
under your control unless you bind the staging yourself. That matters here
because kt binds its worker threads to one node with `MPOL_BIND | STRICT`: if a
re-stage leaves one node fuller than before, the expert weights can fail to
allocate on that node while the machine as a whole has plenty free. If you hit
that after a re-stage, it is the layout, not the budget.

⚠️ We did not re-stage under an explicit binding, so there is no measured
recipe here for making the split deterministic — only the warning that it
varies.

⚠️ `free -g` will not show the space coming back as "free" in the way you might
expect; tmpfs pages are released immediately on `rm`, but page cache for the
checkpoint will expand to use what was freed. Read `MemAvailable` in
`/proc/meminfo` rather than the free column.

## Failure modes worth knowing

| Symptom | Cause |
|---|---|
| Worker segfaults after a few hundred requests; the Python traceback makes no sense | jemalloc preloaded. The script unsets `LD_PRELOAD`; check nothing re-adds it. |
| Operator fails with `EZ1001` | `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/` missing (step 3). |
| `import` works but ds41 operators are missing | CANN's `cann_ops_transformer` is ahead of `DS41_SITE` (step 4). |
| Refuses to start naming `KT_ZEROCOPY_*` or `KT_FULLSET_LOAD` | kt-kernel is a stock build (step 6), or `KT_SITE` points at an older one. |
| Allocation fails during weight load although the machine has free memory | kt's worker threads are hard-bound to one NUMA node; that node is full. Lower `num_resident_experts` or rebalance the offload across nodes. |

## Tuning

`num_resident_experts` is the one knob with a real trade-off: more resident
experts means less CPU work per token, but they compete with the KV cache for
HBM. Changing it changes **which** experts run in the NPU's quantised path and
which run on the CPU in bf16, so it moves the numerics — re-check accuracy after
changing it, do not assume it is free.
