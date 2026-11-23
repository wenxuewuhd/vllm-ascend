#!/usr/bin/env bash
# AVX2 MXFP4 copy profile. Pass actual CPU threads and NUMA nodes for this cpuset.
# Usage: MODEL=... CANN=... ... bash serve_avx2.sh 24 '[0]'
set -euo pipefail
[ "$#" = 2 ] || { echo "Usage: bash serve_avx2.sh CPU_THREADS '[NUMA_NODE,...]'" >&2; exit 1; }

KT_CONFIG=$(python3 - "$1" "$2" "${RESIDENT:-100}" <<'PY'
import json
import os
from pathlib import Path
import sys

threads, nodes, resident = int(sys.argv[1]), json.loads(sys.argv[2]), int(sys.argv[3])
allowed = os.sched_getaffinity(0)
if not isinstance(nodes, list) or not nodes or any(type(n) is not int or n < 0 for n in nodes):
    sys.exit('NUMA nodes must be a nonempty JSON list of nonnegative integers')
if len(set(nodes)) != len(nodes) or not len(nodes) <= threads <= len(allowed):
    sys.exit(f'Need unique NUMA nodes and len(nodes) <= CPU_THREADS <= cpuset CPUs ({len(allowed)})')
for node in nodes:
    cpulist = Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().strip()
    cpus = set()
    for part in cpulist.split(','):
        ends = [int(v) for v in part.split('-')]
        cpus.update(range(ends[0], ends[-1] + 1))
    if not cpus & allowed:
        sys.exit(f'NUMA node {node} has no CPUs in this cpuset')
print(json.dumps({'kt_offload_config': {
    'enabled': True, 'resident_strategy': 'prefix', 'num_resident_experts': resident,
    'cpuinfer_threads': threads, 'threadpool_count': len(nodes), 'numa_nodes': nodes,
    'packed_transfer': False, 'stream_prefill': True, 'stream_prefill_min_tokens': 512,
    'dynamic_resident': True, 'zerocopy_w13': False,
}}))
PY
)
export KT_CONFIG
export KT_MXFP4_BACKEND=avx2
export KT_MXFP4_COMPACT_SCALES=0
echo "  profile=avx2-copy threads=$1 numa_nodes=$2 stream_prefill=true min_tokens=512 dynamic_resident=true zerocopy_w13=false scales=fp32"
exec bash "$(dirname "${BASH_SOURCE[0]}")/serve_lowlatency.sh"
