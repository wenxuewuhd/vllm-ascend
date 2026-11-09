# SPDX-License-Identifier: Apache-2.0
"""Preconditions on the staged engram tables.

The engram tables do not fit in device memory, so they are staged as raw files
in tmpfs or SSD files and memory-mapped read-only. A short file reads as zeros past its end,
and zeros are valid engram payload, so nothing downstream can detect it.

The expected size therefore comes from the checkpoint (the safetensors
``data_offsets``), not from anything the staging step recorded, and is
compared with integer equality. A missing directory or file (``/dev/shm`` is
emptied by a reboot) takes the same refusal path.
"""

from __future__ import annotations

import json
import os
import struct

#: Where the staging step writes the raw tables. One file per tensor, named
#: for the tensor with "/" replaced -- "." is kept, so the name still reads as
#: the checkpoint's.
DEFAULT_STAGE_DIR = "/dev/shm/engram_stage"

#: safetensors: 8-byte little-endian header length, then that many bytes of
#: JSON, then the data section that ``data_offsets`` indexes into.
_HEADER_LEN_BYTES = 8


def staging_config(stage_dir: str | None = None) -> tuple[str, str]:
    """Resolve the backend and directory at call time, before table binding.

    Function defaults must not capture an imported path: launchers select the
    same directory for verification and mapping through the central env module.
    Existing explicit RAM paths remain supported.
    """
    from vllm_ascend import envs

    backend = envs.VLLM_ASCEND_DSV41_ENGRAM_BACKEND
    if backend not in ("ram", "ssd"):
        raise ValueError(f"Engram backend must be 'ram' or 'ssd', got {backend!r}")
    directory = os.fspath(stage_dir) if stage_dir is not None else envs.VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR
    if not directory:
        if backend == "ssd":
            raise ValueError("SSD Engram requires an explicit absolute stage directory")
        directory = DEFAULT_STAGE_DIR
    if not os.path.isabs(directory):
        raise ValueError(f"Engram stage directory must be absolute: {directory!r}")
    return backend, directory


def staged_file_name(tensor_name: str) -> str:
    return tensor_name.replace("/", "_") + ".bin"


def expected_nbytes(checkpoint_dir: str, tensor_name: str) -> int:
    """Length of one tensor according to the checkpoint's own header."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path, "r") as handle:
        weight_map = json.load(handle)["weight_map"]
    if tensor_name not in weight_map:
        raise KeyError(
            f"{tensor_name!r} is not in {index_path}. The staged table does "
            "not correspond to a tensor in this checkpoint.")

    shard_path = os.path.join(checkpoint_dir, weight_map[tensor_name])
    with open(shard_path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(_HEADER_LEN_BYTES))[0]
        header = json.loads(handle.read(header_len))
    start, end = header[tensor_name]["data_offsets"]
    return end - start


def verify_staged_tables(tensor_names: list[str], checkpoint_dir: str,
                         stage_dir: str | None = None) -> dict[str, int]:
    """Refuse to start unless every staged table is exactly the right length.

    Returns the verified sizes.
    """
    if not tensor_names:
        raise ValueError(
            "verify_staged_tables() was given no tensor names. Checking "
            "nothing and checking everything both return without complaint, "
            "so this refuses instead.")

    backend, stage_dir = staging_config(stage_dir)
    manifest = None
    if backend == "ssd":
        from vllm_ascend.models.deepseek_v4_1.engram_storage import (
            load_manifest,
            validate_filesystem,
        )

        validate_filesystem(stage_dir, backend)
        manifest = load_manifest(stage_dir, checkpoint_dir)

    if not os.path.isdir(stage_dir):
        raise RuntimeError(
            f"engram staging directory {stage_dir!r} does not exist. It is "
            "tmpfs, so a reboot empties it and the tables have to be staged "
            "again; this is not a sign that they are damaged.")

    verified: dict[str, int] = {}
    problems: list[str] = []
    for name in tensor_names:
        expected = expected_nbytes(checkpoint_dir, name)
        path = os.path.join(stage_dir, staged_file_name(name))
        if not os.path.isfile(path):
            problems.append(f"{name}: expected {expected:,} bytes, file is missing ({path})")
            continue
        actual = os.path.getsize(path)
        if actual != expected:
            short_by = expected - actual
            problems.append(
                f"{name}: expected {expected:,} bytes, found {actual:,} "
                f"({short_by:+,} bytes)")
            continue
        if manifest is not None:
            from vllm_ascend.models.deepseek_v4_1.engram_storage import validate_table

            spec = manifest["tables"][name]
            validate_table(manifest, name, os.stat(path, follow_symlinks=False), *spec["shape"])
        verified[name] = actual

    if problems:
        raise RuntimeError(
            "staged engram tables do not match the checkpoint:\n  "
            + "\n  ".join(problems)
            + f"\n\n{len(verified)} of {len(tensor_names)} tables verified. "
            "Sizes come from the safetensors data_offsets, so this is not a "
            "tolerance that can be loosened -- a table that is short by any "
            "number of bytes reads zeros past the end, and zeros are valid "
            "engram payload. If the staging directory is empty or the files "
            "are missing, /dev/shm was most likely cleared by a reboot and "
            "the tables need staging again.")

    return verified


# -- content, not just length -------------------------------------------------
#
# A stray write changes a table's content but not its length, so the size
# check cannot see it. Two independent folds per file, so a change that
# preserves one is still caught by the other.

_FOLD_CHUNK = 1 << 30
_U64_MASK = (1 << 64) - 1


def content_fingerprint(path: str) -> tuple[int, int]:
    """``(sum, xor)`` over the file's 64-bit words, both mod 2**64.

    Mapped read-only so the check cannot modify the table it checks.
    """
    import mmap

    import numpy as np

    size = os.path.getsize(path)
    handle = os.open(path, os.O_RDONLY)
    total, xor, done = 0, 0, 0
    try:
        while done < size:
            length = min(_FOLD_CHUNK, size - done)
            length -= length % 8          # whole words only
            if not length:
                break
            mapping = mmap.mmap(handle, length, flags=mmap.MAP_SHARED,
                                prot=mmap.PROT_READ, offset=done)
            words = np.frombuffer(mapping, dtype=np.uint64)
            total = (total + int(words.sum(dtype=np.uint64))) & _U64_MASK
            xor ^= int(np.bitwise_xor.reduce(words))
            del words                     # numpy holds a view; close() refuses otherwise
            mapping.close()
            done += length
    finally:
        os.close(handle)
    return total, xor


def fingerprint_staged_tables(tensor_names: list[str],
                              stage_dir: str | None = None
                              ) -> dict[str, tuple[int, int]]:
    """Fingerprint every staged table, for recording after staging."""
    _, stage_dir = staging_config(stage_dir)
    return {
        name: content_fingerprint(os.path.join(stage_dir, staged_file_name(name)))
        for name in tensor_names
    }


def verify_staged_content(expected: dict[str, tuple[int, int]],
                          stage_dir: str | None = None) -> None:
    """Refuse to start if any table's content has changed since it was staged.

    ``expected`` is what ``fingerprint_staged_tables`` returned when the tables
    were written. Sizes are checked separately by ``verify_staged_tables``;
    this is the arm that catches a write, which leaves the size untouched.
    """
    if not expected:
        raise ValueError(
            "verify_staged_content() was given no fingerprints. Checking "
            "nothing returns just as quietly as checking everything.")

    _, stage_dir = staging_config(stage_dir)
    problems = []
    for name, (want_sum, want_xor) in expected.items():
        path = os.path.join(stage_dir, staged_file_name(name))
        if not os.path.isfile(path):
            problems.append(f"{name}: file is missing ({path})")
            continue
        got_sum, got_xor = content_fingerprint(path)
        if (got_sum, got_xor) != (want_sum, want_xor):
            problems.append(
                f"{name}: fingerprint changed\n      expected sum={want_sum:016x} "
                f"xor={want_xor:016x}\n      found    sum={got_sum:016x} "
                f"xor={got_xor:016x}")

    if problems:
        raise RuntimeError(
            "staged engram tables have changed since they were written:\n  "
            + "\n  ".join(problems)
            + "\n\nRe-stage rather than continuing.")
