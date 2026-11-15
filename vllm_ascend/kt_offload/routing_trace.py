# SPDX-License-Identifier: Apache-2.0
"""Per-step routed-expert TRACE -- the one number stage 4 has always been gated on.

``expert_counter.py`` accumulates FREQUENCIES, which answer "which experts are
hot overall". Stage 4's gain depends on something frequencies cannot express:
**temporal locality** -- per step, how many of the routed experts are already
resident, i.e. what a C-slot LRU would hit. So this appends the per-call expert
SET instead of adding into a histogram.

Kept separate from ``expert_counter`` on purpose: that module's value is its
denominators (``forwards.seen == counted + sum(excluded)``), and adding a second
output with different inclusion rules into it would put those at risk.

🔴 Two properties of the trace that decide how it may be read:

* **It must be taken with graph capture OFF.** Under capture the per-step Python
  never runs (``expert_counter`` excludes such calls for the same reason), so
  there is nothing to record. That is sound rather than a compromise: routed ids
  are a function of the hidden states, so for one prompt under greedy decoding
  an eager run and a captured run route **identically**. The trace is therefore
  regime-independent even though the run that produces it is not.
* **A trace run is not a timing run.** Reading ``topk_ids`` means a device->host
  copy per layer per step, which serialises the step. Never quote TTFT or TPOT
  from a traced run.

Off unless ``VLLM_ASCEND_DSV41_ROUTING_TRACE=<file>`` is set; read once at
import, so with it unset nothing here runs and the call site costs one bool test.

Format (little-endian, append-only, one record per counted call)::

    magic   "KTRT"  u32 version=1  u32 n_layers  u32 n_experts   (header, once)
    record: u32 seq · u16 layer · u16 rows · u16 topk · u16 n_ids · n_ids x u16

``seq`` increments on every layer-0 call, so it is the forward (step) index;
records of one step share it. ids are the LOGICAL expert ids of that call, in
router order, duplicates kept (a row may route to the same expert twice is not
possible, but two rows may share one, and the multiplicity is information).
"""

from __future__ import annotations

import os
import re
import struct
import threading

import torch
from vllm.logger import logger

#: Same rules as expert_counter: main layers only, no dspark/mtp.
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
MAGIC = b"KTRT"
VERSION = 1
HEADER = struct.Struct("<4sIII")
REC = struct.Struct("<IHHHH")
#: Records buffered before a write. 40 layers/step, so this is ~50 steps.
FLUSH_EVERY = 2048


def _path() -> str:
    return os.environ.get("VLLM_ASCEND_DSV41_ROUTING_TRACE", "").strip()


ENABLED = bool(_path())

_LOCK = threading.Lock()
_BUF: list[bytes] = []
_SEQ = 0
_FH = None
_N_LAYERS = 40
_N_EXPERTS = 384
_CALLS = 0
_SKIPPED: dict[str, int] = {}


def _skip(why: str) -> None:
    _SKIPPED[why] = _SKIPPED.get(why, 0) + 1


def _open():
    global _FH
    if _FH is None:
        p = _path()
        _FH = open(p, "wb", buffering=0)
        _FH.write(HEADER.pack(MAGIC, VERSION, _N_LAYERS, _N_EXPERTS))
        logger.info("kt_offload routing trace: writing %s (graph capture must be OFF for this to "
                    "see anything; a traced run is NOT a timing run)", p)
    return _FH


def record(layer, topk_ids: torch.Tensor, force_load_balance: bool) -> None:
    """Append one router output. Call only when ENABLED (the call site checks)."""
    global _SEQ, _CALLS
    if force_load_balance:
        _skip("force_load_balance (profile/dummy run: random ids)")
        return
    name = getattr(layer, "layer_name", "") or ""
    m = _LAYER_RE.search(name)
    if m is None or "dspark" in name or "mtp" in name:
        _skip("not a main layer")
        return
    layer_idx = int(m.group(1))
    if layer_idx >= _N_LAYERS:
        _skip(f"layer index >= {_N_LAYERS}")
        return
    try:
        ids = topk_ids.detach().reshape(-1).to("cpu", torch.int64).tolist()
    except Exception as exc:  # noqa: BLE001 - a trace must never take the server down
        _skip(f"copy failed: {type(exc).__name__}")
        return
    rows = int(topk_ids.shape[0]) if topk_ids.dim() else 1
    topk = len(ids) // rows if rows else len(ids)
    with _LOCK:
        if layer_idx == 0:
            _SEQ += 1
        _CALLS += 1
        body = b"".join(struct.pack("<H", i & 0xFFFF) for i in ids)
        _BUF.append(REC.pack(_SEQ, layer_idx, rows, topk, len(ids)) + body)
        if len(_BUF) >= FLUSH_EVERY:
            _flush_locked()


def _flush_locked() -> None:
    if not _BUF:
        return
    fh = _open()
    fh.write(b"".join(_BUF))
    _BUF.clear()


def close() -> None:
    global _FH
    with _LOCK:
        _flush_locked()
        if _FH is not None:
            _FH.close()
            _FH = None
        logger.info("kt_offload routing trace: %d calls over %d forwards, skipped %s",
                    _CALLS, _SEQ, dict(sorted(_SKIPPED.items())))


if ENABLED:
    import atexit

    atexit.register(close)
