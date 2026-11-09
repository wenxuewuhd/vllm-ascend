#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare or explicitly verify SSD Engram tables with bounded memory."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true", help="Read/hash complete existing output; never overwrite")
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be positive")
    source = Path(__file__).resolve().parents[2] / "vllm_ascend" / "models" / "deepseek_v4_1" / "engram_storage.py"
    spec = importlib.util.spec_from_file_location("engram_storage", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    action = module.verify_tables if args.verify else module.prepare_tables
    manifest = action(
        args.checkpoint,
        args.output,
        chunk_bytes=args.chunk_mib << 20,
        progress=lambda event: print(json.dumps(event), flush=True),
    )
    print(
        json.dumps(
            {
                "status": "verified" if args.verify else "prepared",
                "manifest": str(Path(args.output).resolve() / "manifest.json"),
                "total_bytes": sum(t["nbytes"] for t in manifest["tables"].values()),
                "warning": "Full hashing warms the page cache; establish cold state "
                "separately before SSD cold measurements.",
            },
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
