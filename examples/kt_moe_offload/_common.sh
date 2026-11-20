# Shared environment for the launch scripts. Sourced, not executed.
# Every line here prevents a specific failure; the comment says which. None is
# a tuning knob -- the two scripts differ only in the serve arguments below.

: "${MODEL:?path to the DeepSeek-V4.1-Flash weights}"
: "${CANN:?path to the CANN 9.2 toolkit root (the dir holding set_env.sh)}"
: "${VLLM_TREE:?vllm checkout}"
: "${ASCEND_TREE:?vllm-ascend checkout}"
: "${DS41_SITE:?ds41 operator package, installed with pip --target}"
: "${KT_SITE:?kt-kernel package, installed with pip --target}"

CARD=${CARD:-0}
PORT=${PORT:-8000}
GPUUTIL=${GPUUTIL:-0.9}
# Experts kept in HBM; the rest run on the CPU. Raising it needs the memory
# account redone -- overshooting fails during weight load, not at start-up.
RESIDENT=${RESIDENT:-100}

# jemalloc is preloaded by some Ascend images. With it, a worker takes SIGSEGV
# after roughly 365-430 requests and faulthandler prints a misleading Python
# stack. Short benchmarks stay under that count and look clean.
unset LD_PRELOAD

# CANN 9.2's set_env.sh only removes its own entries, so a 9.1 toolkit left on
# PATH/LD_LIBRARY_PATH keeps winning. Strip it by path component first.
_strip() { echo "${1:-}" | tr ':' '\n' | grep -vE '/Ascend/(cann-9\.1\.0|ascend-toolkit|nnal)' | paste -sd: -; }
export PATH=$(_strip "${PATH:-}")
export LD_LIBRARY_PATH=$(_strip "${LD_LIBRARY_PATH:-}")
export PYTHONPATH=$(_strip "${PYTHONPATH:-}")
set +u; source "$CANN/set_env.sh"; set -u   # set_env.sh reads ZSH_VERSION undefined

# 🔴 DS41_SITE first: CANN ships its own cann_ops_transformer and set_env.sh puts
# it at the head of PYTHONPATH. Behind it, `import` still succeeds and the ds41
# operators are simply absent, with nothing reported.
# KT_SITE last so it cannot shadow vllm or vllm_ascend.
export PYTHONPATH="$DS41_SITE:$VLLM_TREE:$ASCEND_TREE${PYTHONPATH:+:$PYTHONPATH}:$KT_SITE"

python3 - <<'PY' || exit 1
import os, sys
want = os.environ["DS41_SITE"]
try:
    import cann_ops_transformer as m
except Exception as exc:
    sys.exit(f"cann_ops_transformer is not importable: {type(exc).__name__}: {exc}")
got = os.path.dirname(os.path.dirname(getattr(m, "__file__", "") or ""))
if os.path.realpath(got) != os.path.realpath(want):
    sys.exit(f"cann_ops_transformer resolves to {got}, not DS41_SITE {want}. CANN's "
             "own copy is shadowing the ds41 one; the operators would be silently missing.")
print(f"  ok: cann_ops_transformer -> {got}")
PY

# The ds41 operators are built on first use by torch's JIT extension loader,
# which caches by extension name in one directory shared by every tree on the
# host. Its rebuild check hashes the compile flags, and those carry
# ASCEND_CUSTOM_OPP_PATH -- i.e. this tree's path. So a launch from any other
# tree rewrites the same cache entries with its own flags, and the next launch
# here rebuilds all of them: measured 10 of 11 operators recompiled on launches
# minutes apart from an unchanged tree, about a minute of start-up each time.
# Giving each tree its own directory makes the build happen once.
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions_kt/$(echo "$ASCEND_TREE" | md5sum | cut -c1-12)}
mkdir -p "$TORCH_EXTENSIONS_DIR"

export ASCEND_RT_VISIBLE_DEVICES="$CARD"
export TMPDIR="${TMPDIR:-/dev/shm}"

# 🔴 Required by dynamic_resident, and the config REFUSES to start without it.
# Without it kt loads CPU weights only for the offloaded subset, so a resident
# expert has no CPU buffer to be demoted into: the policy has zero free slots
# and the swap-in becomes a silent no-op -- a run that reads exactly like
# "the heat said no change". Both profiles set it because every measurement
# quoted in these scripts was taken with it set. AVX2 requires the native
# full-set-load patch; serve_avx2.sh keeps streaming/dynamic but uses copies.
export KT_FULLSET_LOAD=1

# 🔴 Graph mode, not eager. Decode is ~3.5x slower with --enforce-eager
# (85-98 ms per output token measured here against 25-27 ms captured), and every
# latency and accuracy number quoted in these scripts was taken with graphs
# captured. Capturing without this variable set raises at start-up.
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}

# Common serve arguments. The two scripts add their own on top.
COMMON_ARGS=(
  --served-model-name dsv41
  --tensor-parallel-size 1
  --data-parallel-size 1
  --gpu-memory-utilization "$GPUUTIL"
  --kv-cache-dtype bfloat16
  --block-size 128
  --no-enable-prefix-caching
  --trust-remote-code
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}'
)
