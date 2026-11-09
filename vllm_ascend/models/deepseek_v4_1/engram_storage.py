# SPDX-License-Identifier: Apache-2.0
"""Metadata and bounded-memory preparation for Engram SSD tables.

This module deliberately uses only the Python standard library. Startup reads
metadata and validates the already opened table descriptor; full payload hashing
belongs to preparation or an explicit offline verification.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "vllm-ascend-engram-ssd"
VERSION = 1
INDEX_FILE = "model.safetensors.index.json"
MANIFEST_FILE = "manifest.json"
TENSOR_NAMES = tuple(f"layers.{layer}.engram.embed.{kind}" for layer in (1, 14) for kind in ("weight", "scale"))
TABLE_TYPES = {"weight": ("F8_E4M3", 256), "scale": ("F8_E8M0", 8)}
STAT_FIELDS = ("ino", "dev", "size", "mtime_ns", "ctime_ns")
DEFAULT_CHUNK_BYTES = 8 << 20
MAX_METADATA_BYTES = 32 << 20
TMPFS_MAGIC = 0x01021994
RAMFS_MAGIC = 0x858458F6


def staged_file_name(tensor_name: str) -> str:
    return tensor_name.replace("/", "_") + ".bin"


def _stat_identity(value) -> dict:
    return {key: getattr(value, f"st_{key}") for key in STAT_FIELDS}


@contextmanager
def _open_readonly(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{path}: not a regular file")
        # Metadata reads must not prefetch bytes beyond a safetensors header.
        with os.fdopen(fd, "rb", buffering=0, closefd=False) as handle:
            yield handle
    finally:
        os.close(fd)


def _read_exact(handle, nbytes):
    chunks = []
    while nbytes:
        chunk = handle.read(nbytes)
        if not chunk:
            raise ValueError("short read: incomplete metadata")
        chunks.append(chunk)
        nbytes -= len(chunk)
    return b"".join(chunks)


def _read_json(path):
    with _open_readonly(path) as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > MAX_METADATA_BYTES:
            raise ValueError(f"invalid or oversized metadata file: {path}")
        raw = _read_exact(handle, before.st_size)
        if _stat_identity(before) != _stat_identity(os.fstat(handle.fileno())):
            raise ValueError(f"metadata changed while reading: {path}")
        return json.loads(raw), raw, _stat_identity(before)


def _integer(value):
    return type(value) is int and value >= 0


def _check_stat(value):
    if (
        not isinstance(value, dict)
        or set(value) != set(STAT_FIELDS)
        or not all(_integer(value[key]) for key in STAT_FIELDS)
    ):
        raise ValueError("invalid file stat identity in Engram manifest")


def _table_shape(name, entry, origin="manifest"):
    dtype, width = TABLE_TYPES[name.rsplit(".", 1)[-1]]
    shape = entry.get("shape") if isinstance(entry, dict) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(_integer(dim) and dim > 0 for dim in shape)
        or shape[1] != width
        or entry.get("dtype") != dtype
    ):
        raise ValueError(f"{name}: {origin} shape or dtype mismatch")
    return shape


def _validate_entry(name, entry):
    shape = _table_shape(name, entry)
    if entry.get("file") != staged_file_name(name):
        raise ValueError(f"{name}: invalid table filename")
    if entry.get("nbytes") != shape[0] * shape[1]:
        raise ValueError(f"{name}: invalid table nbytes")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name}: invalid SHA256")
    _check_stat(entry.get("stat"))
    if entry["stat"]["size"] != entry["nbytes"]:
        raise ValueError(f"{name}: stat size differs from nbytes")
    source = entry.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("shard"), str)
        or Path(source["shard"]).name != source["shard"]
        or not _integer(source.get("data_offset"))
        or not isinstance(source.get("header_sha256"), str)
        or len(source["header_sha256"]) != 64
    ):
        raise ValueError(f"{name}: invalid source metadata")
    _check_stat(source.get("stat"))


def _checkpoint_specs(checkpoint_dir):
    directory = Path(checkpoint_dir)
    index, raw, index_stat = _read_json(directory / INDEX_FILE)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no weight_map")
    specs = {}
    shards = {}
    for name in TENSOR_NAMES:
        shard = weight_map.get(name)
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"{name}: missing or unsafe checkpoint shard")
        if shard not in shards:
            with _open_readonly(directory / shard) as handle:
                before = os.fstat(handle.fileno())
                prefix = _read_exact(handle, 8)
                length = struct.unpack("<Q", prefix)[0]
                if length > MAX_METADATA_BYTES or length > before.st_size - 8:
                    raise ValueError(f"{shard}: invalid safetensors header size")
                header_raw = _read_exact(handle, length)
                header = json.loads(header_raw)
                if _stat_identity(before) != _stat_identity(os.fstat(handle.fileno())):
                    raise ValueError(f"{shard}: checkpoint changed while reading")
                shards[shard] = (header, 8 + length, before, hashlib.sha256(header_raw).hexdigest())
        header, base, source_stat, header_sha = shards[shard]
        tensor = header.get(name)
        if not isinstance(tensor, dict):
            raise ValueError(f"{name}: missing safetensors tensor")
        dtype, width = TABLE_TYPES[name.rsplit(".", 1)[-1]]
        shape = _table_shape(name, tensor, "checkpoint")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_integer(x) for x in offsets)
            or offsets[1] - offsets[0] != shape[0] * width
            or base + offsets[1] > source_stat.st_size
        ):
            raise ValueError(f"{name}: invalid or truncated checkpoint range")
        specs[name] = {
            "file": staged_file_name(name),
            "nbytes": shape[0] * width,
            "shape": shape,
            "dtype": dtype,
            "source": {
                "shard": shard,
                "data_offset": base + offsets[0],
                "header_sha256": header_sha,
                "stat": _stat_identity(source_stat),
            },
        }
    for layer in (1, 14):
        if (
            specs[f"layers.{layer}.engram.embed.weight"]["shape"][0]
            != specs[f"layers.{layer}.engram.embed.scale"]["shape"][0]
        ):
            raise ValueError(f"layer {layer}: weight and scale row counts differ")
    checkpoint = {"index_file": INDEX_FILE, "index_sha256": hashlib.sha256(raw).hexdigest(), "index_stat": index_stat}
    return checkpoint, specs


def load_manifest(stage_dir, checkpoint_dir=None) -> dict:
    """Read and validate metadata, without reading any table payload.

    Supplying checkpoint_dir additionally binds the manifest to the current
    checkpoint index and safetensors headers. Call validate_table with fstat of
    the exact descriptor being mapped before exposing the mapping to readers.
    """
    manifest, _, _ = _read_json(Path(stage_dir) / MANIFEST_FILE)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or type(manifest.get("version")) is not int
        or manifest["version"] != VERSION
    ):
        raise ValueError("unsupported Engram SSD manifest schema/version")
    tables = manifest.get("tables")
    if not isinstance(tables, dict) or set(tables) != set(TENSOR_NAMES):
        raise ValueError("manifest must contain exactly the four Engram tables")
    for name, entry in tables.items():
        _validate_entry(name, entry)
    checkpoint = manifest.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("index_file") != INDEX_FILE
        or not isinstance(checkpoint.get("index_sha256"), str)
        or len(checkpoint["index_sha256"]) != 64
    ):
        raise ValueError("invalid checkpoint identity in Engram manifest")
    _check_stat(checkpoint.get("index_stat"))
    for layer in (1, 14):
        if (
            tables[f"layers.{layer}.engram.embed.weight"]["shape"][0]
            != tables[f"layers.{layer}.engram.embed.scale"]["shape"][0]
        ):
            raise ValueError(f"layer {layer}: manifest row counts differ")
    if checkpoint_dir is not None:
        current_checkpoint, specs = _checkpoint_specs(checkpoint_dir)
        if checkpoint != current_checkpoint:
            raise ValueError("checkpoint index identity differs from manifest")
        for name, spec in specs.items():
            if any(tables[name].get(key) != value for key, value in spec.items()):
                raise ValueError(f"{name}: checkpoint header identity changed")
    return manifest


def validate_table(manifest, tensor_name, stat_result, rows, width) -> None:
    """Validate an opened table's identity, layout and readonly permission."""
    if tensor_name not in TENSOR_NAMES:
        raise ValueError(f"unknown Engram table: {tensor_name}")
    entry = manifest.get("tables", {}).get(tensor_name)
    _validate_entry(tensor_name, entry)
    if type(rows) is not int or type(width) is not int or entry["shape"] != [rows, width]:
        raise ValueError(f"{tensor_name}: runtime shape differs from manifest")
    if not stat.S_ISREG(stat_result.st_mode):
        raise ValueError(f"{tensor_name}: table is not a regular file")
    if stat_result.st_mode & 0o222:
        raise ValueError(f"{tensor_name}: SSD table must have readonly permissions")
    if entry["stat"] != _stat_identity(stat_result):
        raise ValueError(f"{tensor_name}: table file identity changed")


def validate_filesystem(stage_dir, backend) -> None:
    """Reject RAM filesystems for SSD; RAM preserves legacy disk compatibility.

    A disk-backed filesystem does not itself prove physical NVMe placement;
    deployment records must additionally identify the underlying block device.
    """
    if backend not in ("ram", "ssd"):
        raise ValueError(f"unknown Engram storage backend: {backend}")
    if not Path(stage_dir).is_dir():
        raise ValueError(f"Engram directory does not exist: {stage_dir}")
    if backend == "ram":
        return
    if not sys.platform.startswith("linux"):
        raise RuntimeError("SSD filesystem validation requires Linux statfs")
    libc = ctypes.CDLL(None, use_errno=True)
    buffer = ctypes.create_string_buffer(512)
    statfs = libc.statfs
    statfs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    statfs.restype = ctypes.c_int
    if statfs(os.fsencode(stage_dir), buffer) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(stage_dir))
    magic = ctypes.c_ulong.from_buffer(buffer).value & 0xFFFFFFFF
    if magic in (TMPFS_MAGIC, RAMFS_MAGIC):
        raise ValueError(f"SSD Engram directory is on a RAM filesystem: {stage_dir}")


def _hash_exact(handle, nbytes, chunk_bytes=DEFAULT_CHUNK_BYTES):
    digest = hashlib.sha256()
    remaining = nbytes
    while remaining:
        chunk = handle.read(min(chunk_bytes, remaining))
        if not chunk:
            raise ValueError(f"short read: missing {remaining} of {nbytes} bytes")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _copy_exact(source, destination, nbytes, chunk_bytes):
    digest = hashlib.sha256()
    remaining = nbytes
    while remaining:
        chunk = source.read(min(chunk_bytes, remaining))
        if not chunk:
            raise ValueError(f"short read: missing {remaining} of {nbytes} bytes")
        # Buffered file writes guarantee all bytes or raise. Supporting ordinary
        # raw file-like objects as well makes partial writes explicit.
        view = memoryview(chunk)
        while view:
            written = destination.write(view)
            if written is None or written <= 0:
                raise OSError("short write while staging Engram table")
            view = view[written:]
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(source, destination):
    # Both paths share a directory. link publishes a verified file atomically
    # without replacing an existing name; a crash leaves an incomplete set.
    os.link(source, destination, follow_symlinks=False)
    source.unlink()


def _write_json_atomic(path, payload):
    partial = path.with_name(path.name + ".partial")
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fchmod(handle.fileno(), 0o444)
        os.fsync(handle.fileno())
    _publish(partial, path)
    _fsync_directory(path.parent)


def _assert_checkpoint_stable(checkpoint_dir, checkpoint, specs):
    current_checkpoint, current_specs = _checkpoint_specs(checkpoint_dir)
    if current_checkpoint != checkpoint or current_specs != specs:
        raise ValueError("checkpoint changed during preparation/verification")


def _report_progress(progress, event, name, nbytes, started):
    if progress is not None:
        progress(
            {
                "event": event,
                "tensor": name,
                "nbytes": nbytes,
                "table_elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )


def prepare_tables(checkpoint_dir, output_dir, chunk_bytes=DEFAULT_CHUNK_BYTES, progress=None) -> dict:
    """Stream all four tables, verify independent readback, then publish.

    Failure leaves any .partial files plus preparation.failed.json for manual
    inspection. An incomplete or complete output is never silently overwritten.
    """
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_filesystem(output, "ssd")
    lock = output / ".prepare.lock"
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(lock_fd, f"pid={os.getpid()} started={time.time()}\n".encode())
        os.fsync(lock_fd)
        existing = [path.name for path in output.iterdir() if path != lock]
        if existing:
            raise FileExistsError(
                f"output is not empty ({', '.join(existing)}); use --verify "
                "for complete artifacts, or inspect/remove incomplete output"
            )
        checkpoint, specs = _checkpoint_specs(checkpoint_dir)
        entries = {}
        for name, spec in specs.items():
            started = time.monotonic()
            _report_progress(progress, "copy_started", name, spec["nbytes"], started)
            destination = output / spec["file"]
            partial = output / (spec["file"] + ".partial")
            source_path = Path(checkpoint_dir) / spec["source"]["shard"]
            with _open_readonly(source_path) as source:
                if _stat_identity(os.fstat(source.fileno())) != spec["source"]["stat"]:
                    raise ValueError(f"{name}: checkpoint changed before copy")
                source.seek(spec["source"]["data_offset"])
                dest_fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(dest_fd, "wb") as dest:
                    digest = _copy_exact(source, dest, spec["nbytes"], chunk_bytes)
                    dest.flush()
                    os.fsync(dest.fileno())
                if _stat_identity(os.fstat(source.fileno())) != spec["source"]["stat"]:
                    raise ValueError(f"{name}: checkpoint changed during copy")
            _report_progress(progress, "copy_complete", name, spec["nbytes"], started)
            with _open_readonly(partial) as handle:
                readback_stat = _stat_identity(os.fstat(handle.fileno()))
                if _hash_exact(handle, spec["nbytes"], chunk_bytes) != digest or handle.read(1):
                    raise ValueError(f"{name}: independent readback SHA256 differs")
                if os.fstat(handle.fileno()).st_size != spec["nbytes"]:
                    raise ValueError(f"{name}: destination size differs")
                if _stat_identity(os.fstat(handle.fileno())) != readback_stat:
                    raise ValueError(f"{name}: destination changed during readback")
                _report_progress(progress, "readback_complete", name, spec["nbytes"], started)
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
            _publish(partial, destination)
            _fsync_directory(output)
            entries[name] = dict(spec, sha256=digest, stat=_stat_identity(destination.stat()))
            _report_progress(progress, "published", name, spec["nbytes"], started)
        _assert_checkpoint_stable(checkpoint_dir, checkpoint, specs)
        manifest = {
            "schema": SCHEMA,
            "version": VERSION,
            "created_unix_ns": time.time_ns(),
            "checkpoint": checkpoint,
            "tables": entries,
        }
        for name, entry in entries.items():
            validate_table(manifest, name, (output / entry["file"]).stat(), *entry["shape"])
        _write_json_atomic(output / MANIFEST_FILE, manifest)
        return manifest
    except Exception as error:
        # Never modify an existing complete output on a duplicate invocation.
        if not (output / MANIFEST_FILE).exists():
            failure = output / "preparation.failed.json"
            if not failure.exists():
                _write_json_atomic(
                    failure,
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "failed_unix_ns": time.time_ns(),
                        "status": "incomplete",
                    },
                )
        raise
    finally:
        os.close(lock_fd)
        lock.unlink()
        _fsync_directory(output)


def verify_tables(checkpoint_dir, output_dir, chunk_bytes=DEFAULT_CHUNK_BYTES, progress=None) -> dict:
    """Explicit full-table SHA256 verification against source checkpoint."""
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    output = Path(output_dir)
    validate_filesystem(output, "ssd")
    manifest = load_manifest(output, checkpoint_dir)
    checkpoint, specs = _checkpoint_specs(checkpoint_dir)
    for name, entry in manifest["tables"].items():
        started = time.monotonic()
        _report_progress(progress, "verify_started", name, entry["nbytes"], started)
        with _open_readonly(output / entry["file"]) as table:
            before = os.fstat(table.fileno())
            validate_table(manifest, name, before, *entry["shape"])
            digest = _hash_exact(table, entry["nbytes"], chunk_bytes)
            if digest != entry["sha256"]:
                raise ValueError(f"{name}: staged SHA256 differs from manifest")
            if _stat_identity(before) != _stat_identity(os.fstat(table.fileno())):
                raise ValueError(f"{name}: staged file changed during verification")
        with _open_readonly(Path(checkpoint_dir) / entry["source"]["shard"]) as source:
            if _stat_identity(os.fstat(source.fileno())) != entry["source"]["stat"]:
                raise ValueError(f"{name}: checkpoint file changed")
            source.seek(entry["source"]["data_offset"])
            digest = _hash_exact(source, entry["nbytes"], chunk_bytes)
            if digest != entry["sha256"]:
                raise ValueError(f"{name}: checkpoint SHA256 differs from manifest")
            if _stat_identity(os.fstat(source.fileno())) != entry["source"]["stat"]:
                raise ValueError(f"{name}: checkpoint changed during verification")
        _report_progress(progress, "verified", name, entry["nbytes"], started)
    _assert_checkpoint_stable(checkpoint_dir, checkpoint, specs)
    return manifest
