# SPDX-License-Identifier: Apache-2.0
"""Standalone fixture tests: no torch, vLLM, or NPU import is required."""

import hashlib
import importlib.util
import io
import json
import mmap
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "engram_storage_under_test", _REPO / "vllm_ascend/models/deepseek_v4_1/engram_storage.py"
)
storage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(storage)


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    weight_map = {}
    expected = {}
    for layer, rows in ((1, 17), (14, 19)):
        shard = f"layer-{layer}.safetensors"
        header = {"unrelated": {"shape": [5], "dtype": "U8", "data_offsets": [0, 5]}}
        payload = bytearray(b"hello")
        for kind, dtype, width in (("weight", "F8_E4M3", 256), ("scale", "F8_E8M0", 8)):
            name = f"layers.{layer}.engram.embed.{kind}"
            data = bytes((index * 37 + layer) % 256 for index in range(rows * width))
            offset = len(payload)
            payload.extend(data)
            header[name] = {"shape": [rows, width], "dtype": dtype, "data_offsets": [offset, len(payload)]}
            weight_map[name] = shard
            expected[name] = data
        raw = json.dumps(header).encode()
        (directory / shard).write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    (directory / storage.INDEX_FILE).write_text(json.dumps({"weight_map": weight_map}))
    return directory, expected


def prepare(checkpoint, output, **kwargs):
    return storage.prepare_tables(checkpoint[0], output, chunk_bytes=317, **kwargs)


def write_manifest(output, manifest):
    path = output / storage.MANIFEST_FILE
    path.chmod(0o600)
    path.write_text(json.dumps(manifest))
    path.chmod(0o444)


def change_header(checkpoint_dir, mutate):
    shard = checkpoint_dir / "layer-1.safetensors"
    original = shard.read_bytes()
    length = struct.unpack("<Q", original[:8])[0]
    header = json.loads(original[8 : 8 + length])
    mutate(header)
    raw = json.dumps(header).encode()
    shard.write_bytes(struct.pack("<Q", len(raw)) + raw + original[8 + length :])


def test_prepare_exact_bytes_sha_and_mmap_tail(checkpoint, tmp_path):
    output = tmp_path / "ssd"
    manifest = prepare(checkpoint, output)
    assert set(manifest["tables"]) == set(checkpoint[1])
    assert not list(output.glob("*.partial"))
    assert not (output / ".prepare.lock").exists()
    assert not (output / "preparation.failed.json").exists()
    for name, expected in checkpoint[1].items():
        entry = manifest["tables"][name]
        path = output / entry["file"]
        assert path.read_bytes() == expected
        assert entry["sha256"] == hashlib.sha256(expected).hexdigest()
        assert entry["nbytes"] == len(expected)
        assert path.stat().st_mode & 0o222 == 0
        assert entry["nbytes"] % mmap.PAGESIZE != 0
        with path.open("rb") as handle:
            storage.validate_table(manifest, name, os.fstat(handle.fileno()), *entry["shape"])
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as table:
                assert table[-entry["shape"][1] :] == expected[-entry["shape"][1] :]
                with pytest.raises(TypeError):
                    table[0] = 1
    assert storage.load_manifest(output, checkpoint[0]) == manifest
    assert storage.verify_tables(checkpoint[0], output, 73) == manifest


class SmallReads(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))


class SmallWrites(io.BytesIO):
    def write(self, data):
        return super().write(data[:2])


def test_partial_reads_and_writes_are_completed():
    destination = SmallWrites()
    digest = storage._copy_exact(SmallReads(b"0123456789"), destination, 10, 7)
    assert destination.getvalue() == b"0123456789"
    assert digest == hashlib.sha256(b"0123456789").hexdigest()
    assert storage._hash_exact(SmallReads(b"0123456789"), 10, 7) == digest
    assert storage._read_exact(SmallReads(b"0123456789"), 10) == b"0123456789"


@pytest.mark.parametrize("operation", ["copy", "hash", "metadata"])
def test_short_source_never_silently_zero_fills(operation):
    with pytest.raises(ValueError, match="short read"):
        if operation == "copy":
            storage._copy_exact(io.BytesIO(b"abc"), io.BytesIO(), 4, 3)
        elif operation == "hash":
            storage._hash_exact(io.BytesIO(b"abc"), 4, 3)
        else:
            storage._read_exact(io.BytesIO(b"abc"), 4)


def test_zero_write_is_a_failure():
    class NoWrite(io.BytesIO):
        def write(self, data):
            return 0

    with pytest.raises(OSError, match="short write"):
        storage._copy_exact(io.BytesIO(b"abc"), NoWrite(), 3, 2)


@pytest.mark.parametrize(
    "change",
    [
        lambda header: header["layers.1.engram.embed.weight"].update(dtype="U8"),
        lambda header: header["layers.1.engram.embed.weight"].update(shape=[17, 255]),
        lambda header: header["layers.1.engram.embed.scale"].update(shape=[16, 8]),
        lambda header: header["layers.1.engram.embed.scale"].update(data_offsets=[4357, 99999]),
    ],
)
def test_bad_checkpoint_layout_refused(checkpoint, tmp_path, change):
    change_header(checkpoint[0], change)
    output = tmp_path / "ssd"
    with pytest.raises(ValueError):
        prepare(checkpoint, output)
    assert not (output / storage.MANIFEST_FILE).exists()
    assert json.loads((output / "preparation.failed.json").read_text())["status"] == "incomplete"


def test_truncated_checkpoint_refused(checkpoint, tmp_path):
    shard = checkpoint[0] / "layer-14.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    output = tmp_path / "ssd"
    with pytest.raises(ValueError, match="truncated"):
        prepare(checkpoint, output)
    assert not (output / storage.MANIFEST_FILE).exists()


def test_independent_readback_detects_copy_corruption(checkpoint, tmp_path, monkeypatch):
    real_copy = storage._copy_exact

    def corrupt(source, dest, nbytes, chunk_bytes):
        digest = real_copy(source, dest, nbytes, chunk_bytes)
        dest.seek(0)
        dest.write(b"corrupt")
        return digest

    monkeypatch.setattr(storage, "_copy_exact", corrupt)
    output = tmp_path / "ssd"
    with pytest.raises(ValueError, match="readback SHA256"):
        prepare(checkpoint, output)
    assert len(list(output.glob("*.bin.partial"))) == 1
    assert not list(output.glob("*.bin"))
    assert not (output / storage.MANIFEST_FILE).exists()


def test_checkpoint_changes_during_copy_never_publish_manifest(checkpoint, tmp_path, monkeypatch):
    real_copy = storage._copy_exact

    def change_index(source, dest, nbytes, chunk_bytes):
        result = real_copy(source, dest, nbytes, chunk_bytes)
        index = checkpoint[0] / storage.INDEX_FILE
        index.write_text(index.read_text() + " ")
        return result

    monkeypatch.setattr(storage, "_copy_exact", change_index)
    output = tmp_path / "ssd"
    with pytest.raises(ValueError, match="checkpoint changed"):
        prepare(checkpoint, output)
    assert not (output / storage.MANIFEST_FILE).exists()
    assert (output / "preparation.failed.json").exists()


def test_atomic_publication_only_after_verified_payload(checkpoint, tmp_path, monkeypatch):
    published = []
    real_publish = storage._publish
    output = tmp_path / "ssd"

    def observe(source, dest):
        assert source.name.endswith(".partial")
        assert not dest.exists()
        assert not (output / storage.MANIFEST_FILE).exists()
        assert source.stat().st_mode & 0o222 == 0
        if dest.name.endswith(".bin"):
            name = dest.name[:-4]
            assert source.read_bytes() == checkpoint[1][name]
        else:
            assert dest.name == storage.MANIFEST_FILE
            assert len(list(output.glob("*.bin"))) == 4
        real_publish(source, dest)
        published.append(dest.name)

    monkeypatch.setattr(storage, "_publish", observe)
    prepare(checkpoint, output)
    assert published[-1] == storage.MANIFEST_FILE
    assert len(published) == 5


def test_atomic_publish_refuses_existing_destination(tmp_path):
    partial = tmp_path / "data.partial"
    final = tmp_path / "data"
    partial.write_bytes(b"new")
    final.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        storage._publish(partial, final)
    assert final.read_bytes() == b"old"
    assert partial.read_bytes() == b"new"


def test_existing_complete_output_refused_and_explicit_verify_allowed(checkpoint, tmp_path):
    output = tmp_path / "ssd"
    manifest = prepare(checkpoint, output)
    snapshots = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="use --verify"):
        prepare(checkpoint, output)
    assert snapshots == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
    assert storage.verify_tables(checkpoint[0], output) == manifest


def test_incomplete_output_and_concurrent_lock_refused(checkpoint, tmp_path):
    output = tmp_path / "ssd"
    output.mkdir()
    partial = output / "unfinished.partial"
    partial.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError, match="output is not empty"):
        prepare(checkpoint, output)
    assert partial.read_bytes() == b"do not overwrite"
    assert not (output / storage.MANIFEST_FILE).exists()
    lock = output / ".prepare.lock"
    lock.write_text("other writer")
    with pytest.raises(FileExistsError):
        prepare(checkpoint, output)
    assert lock.read_text() == "other writer"


def test_load_manifest_does_not_read_payload(checkpoint, tmp_path, monkeypatch):
    output = tmp_path / "ssd"
    manifest = prepare(checkpoint, output)

    def no_payload(*args, **kwargs):
        pytest.fail("startup scanned table payload")

    monkeypatch.setattr(storage, "_hash_exact", no_payload)
    for path in output.glob("*.bin"):
        path.unlink()
    assert storage.load_manifest(output, checkpoint[0]) == manifest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(schema="unknown"),
        lambda data: data.update(version=99),
        lambda data: data["tables"].pop("layers.1.engram.embed.weight"),
        lambda data: data["tables"]["layers.1.engram.embed.weight"].update(dtype="F8_E8M0"),
        lambda data: data["tables"]["layers.1.engram.embed.weight"].update(shape=[17, 255]),
        lambda data: data["tables"]["layers.1.engram.embed.weight"].update(file="../outside.bin"),
        lambda data: data["tables"]["layers.1.engram.embed.weight"].update(sha256="q" * 64),
        lambda data: data["tables"]["layers.1.engram.embed.weight"]["stat"].pop("ino"),
    ],
)
def test_invalid_manifest_refused(checkpoint, tmp_path, mutate):
    output = tmp_path / "ssd"
    manifest = prepare(checkpoint, output)
    mutate(manifest)
    write_manifest(output, manifest)
    with pytest.raises(ValueError):
        storage.load_manifest(output)


def test_missing_invalid_and_symlink_manifest_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.load_manifest(tmp_path)
    path = tmp_path / storage.MANIFEST_FILE
    path.write_text("{incomplete")
    with pytest.raises(ValueError):
        storage.load_manifest(tmp_path)
    path.unlink()
    source = tmp_path / "other.json"
    source.write_text("{}")
    path.symlink_to(source)
    with pytest.raises(OSError):
        storage.load_manifest(tmp_path)


def test_runtime_shape_and_replaced_file_identity_refused(checkpoint, tmp_path):
    output = tmp_path / "ssd"
    manifest = prepare(checkpoint, output)
    name = storage.TENSOR_NAMES[0]
    entry = manifest["tables"][name]
    path = output / entry["file"]
    with pytest.raises(ValueError, match="runtime shape"):
        storage.validate_table(manifest, name, path.stat(), 1, 256)
    replacement = output / "replacement"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o444)
    replacement.replace(path)
    with pytest.raises(ValueError, match="identity changed"):
        storage.validate_table(manifest, name, path.stat(), *entry["shape"])
    with pytest.raises(ValueError):
        storage.verify_tables(checkpoint[0], output)


def test_truncated_or_writable_table_refused(checkpoint, tmp_path):
    output = tmp_path / "ssd"
    manifest = prepare(checkpoint, output)
    name = storage.TENSOR_NAMES[0]
    entry = manifest["tables"][name]
    path = output / entry["file"]
    path.chmod(0o600)
    with pytest.raises(ValueError, match="readonly"):
        storage.validate_table(manifest, name, path.stat(), *entry["shape"])
    with path.open("r+b") as handle:
        handle.truncate(entry["nbytes"] - 1)
    path.chmod(0o444)
    with pytest.raises(ValueError, match="identity changed"):
        storage.validate_table(manifest, name, path.stat(), *entry["shape"])


def test_full_verify_detects_corruption_even_with_updated_stat(checkpoint, tmp_path):
    output = tmp_path / "ssd"
    manifest = prepare(checkpoint, output)
    name = storage.TENSOR_NAMES[-1]
    entry = manifest["tables"][name]
    path = output / entry["file"]
    path.chmod(0o600)
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        handle.write(b"x")
    path.chmod(0o444)
    # Simulate a tampered stat record so the independent hash check is exercised.
    entry["stat"] = storage._stat_identity(path.stat())
    write_manifest(output, manifest)
    with pytest.raises(ValueError, match="SHA256 differs"):
        storage.verify_tables(checkpoint[0], output)


def test_checkpoint_index_identity_is_bound(checkpoint, tmp_path):
    output = tmp_path / "ssd"
    prepare(checkpoint, output)
    path = checkpoint[0] / storage.INDEX_FILE
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checkpoint index identity"):
        storage.load_manifest(output, checkpoint[0])


def test_filesystem_backend_and_tmpfs_rejection(tmp_path):
    storage.validate_filesystem(tmp_path, "ram")
    storage.validate_filesystem(tmp_path, "ssd")
    with pytest.raises(ValueError, match="unknown"):
        storage.validate_filesystem(tmp_path, "typo")
    with pytest.raises(ValueError, match="does not exist"):
        storage.validate_filesystem(tmp_path / "absent", "ram")
    if Path("/dev/shm").is_dir():
        storage.validate_filesystem("/dev/shm", "ram")
        with pytest.raises(ValueError, match="RAM filesystem"):
            storage.validate_filesystem("/dev/shm", "ssd")


def test_cli_prepare_and_verify(checkpoint, tmp_path):
    script = _REPO / "examples/kt_moe_offload/prepare_engram_ssd.py"
    output = tmp_path / "ssd"
    command = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(checkpoint[0]),
        "--output",
        str(output),
        "--chunk-mib",
        "1",
    ]
    first = subprocess.run(command, text=True, capture_output=True, check=True)
    assert json.loads(first.stdout.splitlines()[-1])["status"] == "prepared"
    result = subprocess.run(command + ["--verify"], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "verified"
    refused = subprocess.run(command, text=True, capture_output=True)
    assert refused.returncode != 0
    assert "use --verify" in refused.stderr


def test_nonregular_checkpoint_source_refused(checkpoint, tmp_path):
    shard = checkpoint[0] / "layer-1.safetensors"
    shard.unlink()
    shard.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        prepare(checkpoint, tmp_path / "ssd")


def test_partial_published_set_is_never_complete(checkpoint, tmp_path, monkeypatch):
    real_publish = storage._publish
    output = tmp_path / "ssd"

    def interrupt(source, dest):
        if dest.name == storage.staged_file_name(storage.TENSOR_NAMES[2]):
            raise OSError("simulated disk publication failure")
        real_publish(source, dest)

    monkeypatch.setattr(storage, "_publish", interrupt)
    with pytest.raises(OSError, match="simulated disk"):
        prepare(checkpoint, output)
    assert len(list(output.glob("*.bin"))) == 2
    assert len(list(output.glob("*.bin.partial"))) == 1
    assert not (output / storage.MANIFEST_FILE).exists()
    assert (output / "preparation.failed.json").exists()
    with pytest.raises(FileNotFoundError):
        storage.load_manifest(output)


def test_progress_reports_independent_readback_and_publish(checkpoint, tmp_path):
    events = []
    storage.prepare_tables(checkpoint[0], tmp_path / "ssd", 317, events.append)
    assert [event["event"] for event in events] == [
        stage
        for _ in storage.TENSOR_NAMES
        for stage in ("copy_started", "copy_complete", "readback_complete", "published")
    ]
    assert all(event["nbytes"] > 0 for event in events)
    assert all(event["table_elapsed_seconds"] >= 0 for event in events)
