# SPDX-License-Identifier: Apache-2.0
"""SSD metadata must describe the same bytes the embedding maps.

These tests exercise the real preparation, checkpoint metadata verification,
binding and row gather together. The tables are small ordinary files, and no
NPU allocation, graph capture or service is required.
"""

import json
import mmap
import os
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.models.deepseek_v4_1 import engram_storage as storage
from vllm_ascend.models.deepseek_v4_1.engram_embedding import AscendStagedEngramEmbedding, _read_only_table
from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout, gather_into_staging
from vllm_ascend.models.deepseek_v4_1.engram_staging import DEFAULT_STAGE_DIR, staging_config, verify_staged_tables

BACKEND = "VLLM_ASCEND_DSV41_ENGRAM_BACKEND"
STAGE_DIR = "VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR"
LAYOUT = EngramLayout(n_hash_cols=24, dim=256, quant_block=32, layer_ids=(1, 14), num_embeddings=(13, 17))


def _write_checkpoint(directory, *, wrong_dtype=False):
    directory.mkdir()
    weight_map = {}
    payloads = {}
    for layer, rows in zip(LAYOUT.layer_ids, LAYOUT.num_embeddings):
        shard = f"model-layer-{layer}.safetensors"
        header, chunks, offset = {}, [], 0
        for kind, width, dtype in (("weight", 256, "F8_E4M3"), ("scale", 8, "F8_E8M0")):
            name = f"layers.{layer}.engram.embed.{kind}"
            payload = ((np.arange(rows * width, dtype=np.uint32) + layer * 19) % 251).astype(np.uint8).tobytes()
            payloads[name] = payload
            header[name] = {
                "dtype": "U8" if wrong_dtype else dtype,
                "shape": [rows, width],
                "data_offsets": [offset, offset + len(payload)],
            }
            weight_map[name] = shard
            offset += len(payload)
            chunks.append(payload)
        raw = json.dumps(header).encode()
        (directory / shard).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))
    (directory / storage.INDEX_FILE).write_text(json.dumps({"weight_map": weight_map}))
    return payloads


@pytest.fixture(autouse=True)
def _storage_environment(monkeypatch):
    monkeypatch.setenv(BACKEND, "ram")
    monkeypatch.delenv(STAGE_DIR, raising=False)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    payloads = _write_checkpoint(checkpoint)
    stage = tmp_path / "ssd"
    manifest = storage.prepare_tables(checkpoint, stage, chunk_bytes=127)
    monkeypatch.setenv(BACKEND, "ssd")
    monkeypatch.setenv(STAGE_DIR, str(stage))
    return SimpleNamespace(checkpoint=checkpoint, stage=stage, manifest=manifest, payloads=payloads)


def _embedding():
    embedding = object.__new__(AscendStagedEngramEmbedding)
    embedding._staged_tables = None
    embedding._layer_id = None
    return embedding


def _rewrite_manifest(stage, manifest):
    path = stage / storage.MANIFEST_FILE
    path.chmod(0o600)
    path.write_text(json.dumps(manifest))
    path.chmod(0o444)


def test_staging_config_resolves_environment_after_import(monkeypatch, tmp_path):
    assert staging_config() == ("ram", DEFAULT_STAGE_DIR)
    first, second = tmp_path / "first", tmp_path / "second"
    monkeypatch.setenv(STAGE_DIR, str(first))
    assert staging_config() == ("ram", str(first))
    monkeypatch.setenv(STAGE_DIR, str(second))
    assert staging_config() == ("ram", str(second))
    assert staging_config(first) == ("ram", str(first))


@pytest.mark.parametrize(
    "backend,directory,match",
    [
        ("unknown", "", "backend"),
        ("ssd", "", "explicit absolute"),
        ("ssd", "relative/path", "absolute"),
        ("ram", "relative/path", "absolute"),
    ],
)
def test_invalid_storage_config_refuses(monkeypatch, backend, directory, match):
    monkeypatch.setenv(BACKEND, backend)
    monkeypatch.setenv(STAGE_DIR, directory)
    with pytest.raises(ValueError, match=match):
        staging_config()


def test_ssd_verify_and_embedding_bind_resolve_the_same_directory(prepared):
    names = list(storage.TENSOR_NAMES)
    checked = verify_staged_tables(names, str(prepared.checkpoint))
    assert checked == {name: len(prepared.payloads[name]) for name in names}
    for layer in LAYOUT.layer_ids:
        embedding = _embedding()
        embedding.bind_staged_tables(LAYOUT, layer, checkpoint_dir=str(prepared.checkpoint))
        assert embedding._layer_id == layer
        assert embedding._engram_stage_dir == str(prepared.stage)
        assert embedding._engram_storage_backend == "ssd"
        for kind, table in zip(("weight", "scale"), embedding.staged_tables()):
            assert table.tobytes() == prepared.payloads[f"layers.{layer}.engram.embed.{kind}"]


def test_ssd_binding_requires_current_checkpoint_identity(prepared):
    with pytest.raises(ValueError, match="checkpoint"):
        _embedding().bind_staged_tables(LAYOUT, 1)


def test_changed_checkpoint_metadata_refuses_binding(prepared):
    path = prepared.checkpoint / storage.INDEX_FILE
    index = json.loads(path.read_text())
    index["metadata"] = {"different_checkpoint": True}
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="checkpoint.*identity"):
        _embedding().bind_staged_tables(LAYOUT, 1, checkpoint_dir=str(prepared.checkpoint))


@pytest.mark.parametrize("field,value", [("dtype", "U8"), ("shape", [13, 128])])
def test_manifest_dtype_and_shape_refuse_binding(prepared, field, value):
    prepared.manifest["tables"]["layers.1.engram.embed.weight"][field] = value
    _rewrite_manifest(prepared.stage, prepared.manifest)
    with pytest.raises(ValueError, match="shape or dtype"):
        _embedding().bind_staged_tables(LAYOUT, 1, checkpoint_dir=str(prepared.checkpoint))


def test_checkpoint_dtype_is_checked_before_mapping(prepared, tmp_path):
    wrong = tmp_path / "wrong-checkpoint"
    _write_checkpoint(wrong, wrong_dtype=True)
    with pytest.raises(ValueError, match="checkpoint shape or dtype mismatch"):
        _embedding().bind_staged_tables(LAYOUT, 1, checkpoint_dir=str(wrong))


def test_runtime_layout_cannot_bind_another_layers_rows(prepared):
    swapped = replace(LAYOUT, num_embeddings=LAYOUT.num_embeddings[::-1])
    with pytest.raises(ValueError, match="does not match the config"):
        _embedding().bind_staged_tables(swapped, 1, checkpoint_dir=str(prepared.checkpoint))


def test_ssd_mapping_is_readonly_and_demand_mapped(prepared, monkeypatch):
    calls = []
    real_mmap = mmap.mmap

    def record_mmap(*args, **kwargs):
        calls.append(kwargs)
        return real_mmap(*args, **kwargs)

    monkeypatch.setattr(mmap, "mmap", record_mmap)
    embedding = _embedding()
    embedding.bind_staged_tables(LAYOUT, 1, checkpoint_dir=str(prepared.checkpoint))
    assert len(calls) == 2
    assert all(call["flags"] == mmap.MAP_SHARED and call["prot"] == mmap.PROT_READ for call in calls)
    for table in embedding.staged_tables():
        assert not table.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            table[0, 0] = 1


def test_binding_does_not_run_full_payload_hashing(prepared, monkeypatch):
    def forbid_payload_scan(*args, **kwargs):
        pytest.fail("startup must not hash or copy the complete table")

    monkeypatch.setattr(storage, "_hash_exact", forbid_payload_scan)
    monkeypatch.setattr(storage, "_copy_exact", forbid_payload_scan)
    _embedding().bind_staged_tables(LAYOUT, 1, checkpoint_dir=str(prepared.checkpoint))


def test_same_size_content_write_invalidates_manifest_identity(prepared):
    path = prepared.stage / storage.staged_file_name("layers.1.engram.embed.weight")
    path.chmod(0o600)
    with path.open("r+b") as handle:
        original = handle.read(1)
        handle.seek(0)
        handle.write(bytes([original[0] ^ 1]))
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    with pytest.raises(ValueError, match="table file identity changed"):
        _embedding().bind_staged_tables(LAYOUT, 1, checkpoint_dir=str(prepared.checkpoint))


def test_bound_mapping_survives_env_changes_and_cannot_be_rebound(prepared, monkeypatch, tmp_path):
    embedding = _embedding()
    embedding.bind_staged_tables(LAYOUT, 1, checkpoint_dir=str(prepared.checkpoint))
    before = embedding.staged_tables()
    pointers = tuple(table.__array_interface__["data"][0] for table in before)
    monkeypatch.setenv(BACKEND, "ram")
    monkeypatch.setenv(STAGE_DIR, str(tmp_path / "missing"))
    after = embedding.staged_tables()
    assert all(left is right for left, right in zip(before, after))
    assert tuple(table.__array_interface__["data"][0] for table in after) == pointers
    assert embedding._engram_stage_dir == str(prepared.stage)
    assert embedding._engram_storage_backend == "ssd"
    with pytest.raises(RuntimeError, match="already bound"):
        embedding.bind_staged_tables(LAYOUT, 14)


def test_ram_and_ssd_gather_first_last_and_duplicate_rows_are_identical(prepared, monkeypatch, tmp_path):
    ssd = _embedding()
    ssd.bind_staged_tables(LAYOUT, 14, checkpoint_dir=str(prepared.checkpoint))
    ram_dir = tmp_path / "ram-reference"
    ram_dir.mkdir()
    for name, payload in prepared.payloads.items():
        (ram_dir / storage.staged_file_name(name)).write_bytes(payload)
    monkeypatch.setenv(BACKEND, "ram")
    ram = _embedding()
    ram.bind_staged_tables(LAYOUT, 14, stage_dir=ram_dir)
    row_ids = np.array([0, 16, 5, 5, 0, 16], dtype=np.int64)
    actual = []
    for embedding in (ram, ssd):
        weight = np.empty((len(row_ids), 256), dtype=np.uint8)
        scale = np.empty((len(row_ids), 8), dtype=np.uint8)
        gather_into_staging(*embedding.staged_tables(), row_ids, weight, scale)
        actual.append((weight, scale))
    assert np.array_equal(actual[0][0], actual[1][0])
    assert np.array_equal(actual[0][1], actual[1][1])
    assert np.array_equal(actual[1][0][2], actual[1][0][3])


def test_direct_mapping_refuses_symlink_and_wrong_length(prepared, tmp_path):
    path = prepared.stage / storage.staged_file_name("layers.1.engram.embed.weight")
    link = tmp_path / "table-link.bin"
    link.symlink_to(path)
    with pytest.raises(OSError):
        _read_only_table(str(link), 13, 256)
    with pytest.raises(ValueError, match="does not match the config"):
        _read_only_table(str(path), 14, 256)
