# SPDX-License-Identifier: Apache-2.0
"""The bank as kt's weight source: per-expert, per-TP-part NZ addresses.

Stage 3 is "both consumers read ONE copy". The NPU already reads the bank; this
is what kt needs to read the same bytes in place, so the checkpoint's expert
pages stop having to stay resident (~269 GiB of duplication).

Why this lives on the vllm-ascend side rather than inside kt: the geometry is
ours (``bank_build`` froze it, ``nz_geometry`` reverse-engineered it), it is the
part that is easy to get silently wrong, and here it can be unit-tested against
the real manifest without a card, a server or kt.

What kt gets, per expert and per TP part, is three base addresses and the NZ
dims that go with them::

    gate   b = w13_gate_p<i> + e*2,949,120   n = inter/tp   k = hidden
    up     b = w13_up_p<i>   + e*2,949,120   n = inter/tp   k = hidden
    down   b = w2            + e*5,898,240 + i*2,949,120
                                             n = hidden     k = inter/tp

and ``group_stride() == n`` in the kernel then IS the dense stride
(``n*16`` B: 18,432 for a w13 part, 81,920 for w2), which is what makes a
strided view unnecessary.

🔴 Three properties this module asserts rather than assumes, because each one
silently produces wrong numbers or lost locality if it does not hold:

* **w2 needs no split.** TP divides down's ``k``, and in NZ the k-block index is
  the OUTERMOST axis, so part 0 is the contiguous *prefix* of the expert's w2 --
  byte-identical to a split. Asserted as: part bytes = record/tp exactly, and
  the part offset is a whole number of pages.
* **page exclusivity per part.** Every section offset and per-expert record must
  be a multiple of the page size, or two parts share a page and no per-node
  placement can be better than 1/tp_count local.
* **no huge pages.** ``MADV_NOHUGEPAGE``, matching ``bank_build``: a 2 MiB page
  spans 28.4 w13 slabs so it necessarily holds both parts' bytes and locality
  collapses to 50% (measured -28.3% against -10.3%), while a task touches only
  160 4 KiB pages -- far inside the 4096-entry L2 TLB. Huge pages here cost and
  buy nothing.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import mmap
import os
from dataclasses import dataclass

from vllm.logger import logger

from vllm_ascend.kt_offload import nz_geometry

#: ``w13`` is stored per TP part, ``w2`` whole (see the module docstring).
W13_SECTIONS = ("w13_gate_p0", "w13_gate_p1", "w13_up_p0", "w13_up_p1")
MADV_NOHUGEPAGE = 15
MADV_WILLNEED = 3


class BankKtSourceError(RuntimeError):
    """The bank's geometry is not the one kt can consume in place."""


@dataclass(frozen=True)
class PartWeights:
    """One expert's three base addresses for one TP part, with their NZ dims."""

    gate_ptr: int
    up_ptr: int
    down_ptr: int
    #: w13: n = inter/tp, k = hidden.  w2: n = hidden, k = inter/tp.
    w13_n: int
    w13_k: int
    w2_n: int
    w2_k: int

    def group_stride_bytes(self) -> tuple[int, int]:
        """(w13, w2) dense strides, i.e. what ``group_stride()*16`` must equal."""
        return self.w13_n * nz_geometry.NZ_C0, self.w2_n * nz_geometry.NZ_C0


#: posix_fadvise(2)
POSIX_FADV_DONTNEED = 4


def drop_checkpoint_expert_cache(weight_path: str) -> tuple[int, int]:
    """Drop the checkpoint shards' clean page cache. Returns (before, after) bytes.

    With ``kt_reads_bank`` nothing reads the checkpoint's EXPERT tensors during
    serving any more -- kt reads the bank, streaming prefill reads the bank, and
    the non-expert weights are already in HBM. But kt's loader still materialises
    per-expert tensors at load time to get the SCALES (it converts those, it
    cannot alias them), which faults all ~269 GiB of expert bytes into the page
    cache. They are then cold and reclaimable, and `mincore` still counts them as
    resident, so the duplication looks like it is still there when functionally
    it is not.

    This makes that explicit rather than waiting for memory pressure to do it.
    The pages are CLEAN (the mapping is read-only), so dropping them cannot lose
    data; the cost of being wrong is a re-read from disk, not corruption.

    🔴 Only safe because of what reads what. Do not call this on a run where the
    streaming path still uses ``WeightSource`` (i.e. without ``bank_dir``): the
    NPU would then re-read the checkpoint on every prefill, from disk.
    """
    import glob as _glob

    shards = sorted(_glob.glob(os.path.join(weight_path, "*.safetensors")))
    if not shards:
        logger.warning("kt_offload: no .safetensors under %s; nothing to drop", weight_path)
        return (0, 0)
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int]

    def _resident(paths) -> int:
        tot = 0
        for path in paths:
            with contextlib.suppress(OSError):
                tot += _resident_bytes(path)
        return tot

    before = _resident(shards)
    for path in shards:
        fd = os.open(path, os.O_RDONLY)
        try:
            libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    after = _resident(shards)
    logger.info(
        "kt_offload: dropped the checkpoint's clean page cache over %d shard(s): "
        "%.1f -> %.1f GiB resident (%.1f GiB released). kt and streaming prefill both "
        "read the bank now, so nothing re-reads these during serving.",
        len(shards),
        before / float(1 << 30),
        after / float(1 << 30),
        (before - after) / float(1 << 30),
    )
    return (before, after)


def _resident_bytes(path: str) -> int:
    """Resident page-cache bytes of a file, via mincore over a read-only map."""
    sz = os.path.getsize(path)
    if sz == 0:
        return 0
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    page = os.sysconf("SC_PAGE_SIZE")
    fd = os.open(path, os.O_RDONLY)
    resident = 0
    try:
        step = 8 << 30
        off = 0
        while off < sz:
            n = min(step, sz - off)
            addr = libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, off)
            if addr is None or ctypes.c_long(addr).value == -1:
                return resident
            npages = (n + page - 1) // page
            vec = ctypes.create_string_buffer(npages)
            if libc.mincore(ctypes.c_void_p(addr), n, vec) == 0:
                resident += sum(1 for b in vec.raw[:npages] if b & 1) * page
            libc.munmap(ctypes.c_void_p(addr), n)
            off += n
    finally:
        os.close(fd)
    return resident


@dataclass(frozen=True)
class BankSpan:
    """``ByteSpan``'s shape, for the fields ``numa_placement`` reads."""

    key: str
    path: str
    start: int
    nbytes: int
    dtype: str = "uint8"
    shape: tuple = ()


class BankPlacementSource:
    """The bank as ``numa_placement``'s source, so the BANK's pages get placed.

    Stage 3 moved kt's reads to the bank but placement was still pointed at the
    checkpoint, so its 268.95 GiB kept being faulted in and migrated -- measured
    2026-11-15: `planned=70502400 resident_at_census=70121128` and a total
    working set that did not move (550.0 vs 549.5 GiB). kt no longer NEEDED
    those pages; we were still putting them there.

    Two differences from the checkpoint source, both properties of the layout:

    * **w13 parts are separate sections**, so part p is not
      ``[p*nbytes/tp, ...)`` of one span -- hence ``split_ranges``.
    * **w2 splits too.** TP divides down's k and in NZ the k-block index is the
      outermost axis, so part p is a contiguous, page-aligned prefix. In the
      checkpoint w2 is a column slice of every row and cannot be split at all
      (see ``numa_placement.STREAMING_ONLY_TENSORS``); the bank is not a
      counter-example to that note, it is a different layout.

    Only the scales stay whole on the streaming node: kt converts them and never
    aliases them, so their file pages have one consumer.
    """

    #: w2 joins the split here -- see the class docstring.
    split_tensors = (("w1", "weight"), ("w3", "weight"), ("w2", "weight"))

    #: (proj, kind) -> (section, half) where `half` picks a sub-range of the
    #: per-expert record: None = all of it, 0/1 = the first/second half.
    #:
    #: 🔴 `w13_scale` holds BOTH w1's and w3's scales per expert -- w1's rows
    #: then w3's (bank_build.py:315). Mapping both (proj, kind) pairs to the
    #: whole section made every scale page claimed twice, and
    #: `_page_runs` refused it: "segments overlap ... a page would have two
    #: owners". That guard is why this was found before a card run.
    _SECTION = {
        ("w1", "weight"): ("w13_gate_p{part}", None),
        ("w3", "weight"): ("w13_up_p{part}", None),
        ("w2", "weight"): ("w2", None),
        ("w1", "scale"): ("w13_scale", 0),
        ("w3", "scale"): ("w13_scale", 1),
        ("w2", "scale"): ("w2_scale", None),
    }

    def _span(self, layer: int, expert: int, proj: str, kind: str) -> tuple[str, int, int]:
        ent = self._SECTION.get((proj, kind))
        if ent is None:
            raise BankKtSourceError(f"no bank section for ({proj}, {kind})")
        name, half = ent
        if "{part}" in name:
            name = name.format(part=0)
        off, rec = self.src.sections[name]
        base = off + expert * rec
        if half is None:
            return self._path(layer), base, rec
        if rec % 2:
            raise BankKtSourceError(f"{name}: record {rec} B does not halve")
        return self._path(layer), base + half * (rec // 2), rec // 2

    def __init__(self, src: KtBankSource):
        self.src = src

    def _path(self, layer: int) -> str:
        return self.src._layer_path(layer)

    def locate(self, layer: int, expert: int, proj: str, kind: str) -> BankSpan:
        path, start, nbytes = self._span(layer, expert, proj, kind)
        return BankSpan(key=f"L{layer}.e{expert}.{proj}.{kind}", path=path, start=start, nbytes=nbytes)

    def split_ranges(self, layer: int, expert: int, proj: str, kind: str, tp: int) -> list[tuple[str, int, int]]:
        """``[(path, start, nbytes)]`` per TP part."""
        if tp != self.src.tp_count:
            raise BankKtSourceError(
                f"placement asked for tp={tp} but the bank source was built for "
                f"{self.src.tp_count}; the section layout depends on it"
            )
        path = self._path(layer)
        tmpl, half = self._SECTION[(proj, kind)]
        if half is not None:
            raise BankKtSourceError(f"({proj}, {kind}) is a half-section; it is not split per part")
        if "{part}" in tmpl:
            # w13: one section per part, each already the part's own bytes.
            out = []
            for part in range(tp):
                off, rec = self.src.sections[tmpl.format(part=part)]
                out.append((path, off + expert * rec, rec))
            return out
        # w2: one section, k whole; part p is the p-th contiguous slice.
        off, rec = self.src.sections[tmpl]
        per = rec // tp
        if rec % tp or per % self.src.page:
            raise BankKtSourceError(
                f"{tmpl}: record {rec} B does not split into {tp} whole-page parts, so the "
                "parts would share pages and placement could not beat 1/tp local"
            )
        base = off + expert * rec
        return [(path, base + part * per, per) for part in range(tp)]


class KtBankSource:
    """Maps the bank read-only and hands out per-expert, per-part addresses."""

    def __init__(self, bank_dir: str, *, tp_count: int, hidden: int, intermediate: int):
        self.bank_dir = str(bank_dir)
        self.tp_count = int(tp_count)
        path = os.path.join(self.bank_dir, "manifest.json")
        if not os.path.exists(path):
            raise BankKtSourceError(f"no manifest at {path}: {bank_dir} is not a bank")
        with open(path) as fh:
            self.manifest = json.load(fh)
        d = self.manifest["derived"]
        self.sections = {k: tuple(v) for k, v in d["sections"].items()}
        self.bytes_per_expert = int(d["bytes_per_expert"])
        self.layer_bytes = int(d["layer_bytes"])
        self.num_experts = int(self.manifest.get("num_experts", 0)) or self._experts_from(d)
        self.page = os.sysconf("SC_PAGE_SIZE")
        self.hidden, self.intermediate = int(hidden), int(intermediate)
        self._maps: dict[int, mmap.mmap] = {}
        self._bases: dict[int, int] = {}
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._check_geometry()

    def _experts_from(self, d) -> int:
        return self.layer_bytes // self.bytes_per_expert

    # -- the three assertions ---------------------------------------------
    def _check_geometry(self) -> None:
        bad: list[str] = []
        inter_tp = self.intermediate // self.tp_count
        want_w13 = inter_tp * (self.hidden // 2)
        want_w2 = self.hidden * (self.intermediate // 2)
        for name in W13_SECTIONS:
            if name not in self.sections:
                bad.append(f"missing section {name}")
                continue
            off, rec = self.sections[name]
            if rec != want_w13:
                bad.append(f"{name}: record {rec} B, expected inter/tp * hidden/2 = {want_w13}")
            if off % self.page or rec % self.page:
                bad.append(
                    f"{name}: offset {off} / record {rec} is not page aligned ({self.page} B), "
                    "so two TP parts would share pages and no per-node placement can beat "
                    "1/tp_count local"
                )
        if "w2" not in self.sections:
            bad.append("missing section w2")
        else:
            off, rec = self.sections["w2"]
            if rec != want_w2:
                bad.append(f"w2: record {rec} B, expected hidden * inter/2 = {want_w2}")
            if rec % self.tp_count:
                bad.append(
                    f"w2: record {rec} B does not divide by tp_count {self.tp_count}, so part 0 "
                    "is not the contiguous prefix the kernel addresses"
                )
            part = rec // self.tp_count
            if off % self.page or part % self.page:
                bad.append(f"w2: offset {off} / per-part {part} is not page aligned ({self.page} B)")
        if bad:
            raise BankKtSourceError(f"{self.bank_dir} cannot be consumed in place by kt:\n  " + "\n  ".join(bad))
        logger.info(
            "kt_offload bank-as-kt-source: %d experts/layer, %d B each; w13 part record %d B "
            "(%d pages), w2 part %d B (%d pages); all page-exclusive per TP part",
            self.num_experts,
            self.bytes_per_expert,
            want_w13,
            want_w13 // self.page,
            want_w2 // self.tp_count,
            (want_w2 // self.tp_count) // self.page,
        )

    # -- mapping ----------------------------------------------------------
    def _layer_path(self, layer: int) -> str:
        return os.path.join(self.bank_dir, f"layer{int(layer):03d}.nzbank")

    def base(self, layer: int) -> int:
        """Map the layer read-only shared and return its base address."""
        if layer in self._bases:
            return self._bases[layer]
        p = self._layer_path(layer)
        sz = os.path.getsize(p)
        if sz != self.layer_bytes:
            raise BankKtSourceError(f"{p} is {sz} B, the manifest says {self.layer_bytes}")
        fd = os.open(p, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, sz, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        addr = self._addr_of(mm)
        # NOHUGEPAGE first, then WILLNEED: asking for the pages after a huge-page
        # policy was already applied would be too late (see the docstring).
        self._libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(sz), MADV_NOHUGEPAGE)
        self._libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(sz), MADV_WILLNEED)
        self._maps[layer] = mm
        self._bases[layer] = addr
        return addr

    @staticmethod
    def _addr_of(mm: mmap.mmap) -> int:
        """Base address of a PROT_READ mmap. ``c_char.from_buffer`` needs a
        writable buffer, so go through the read-only buffer protocol instead."""
        obj = ctypes.py_object(mm)
        addr = ctypes.c_void_p()
        size = ctypes.c_ssize_t()
        if ctypes.pythonapi.PyObject_AsReadBuffer(obj, ctypes.byref(addr), ctypes.byref(size)) != 0:
            raise BankKtSourceError("could not take the address of the bank mapping")
        return int(addr.value)

    # -- what kt consumes -------------------------------------------------
    def weights(self, layer: int, expert: int, part: int) -> PartWeights:
        if not 0 <= part < self.tp_count:
            raise BankKtSourceError(f"part {part} outside [0, {self.tp_count})")
        if not 0 <= expert < self.num_experts:
            raise BankKtSourceError(f"expert {expert} outside [0, {self.num_experts})")
        b = self.base(layer)
        g_off, g_rec = self.sections[f"w13_gate_p{part}"]
        u_off, u_rec = self.sections[f"w13_up_p{part}"]
        d_off, d_rec = self.sections["w2"]
        d_part = d_rec // self.tp_count
        inter_tp = self.intermediate // self.tp_count
        return PartWeights(
            gate_ptr=b + g_off + expert * g_rec,
            up_ptr=b + u_off + expert * u_rec,
            # w2 is stored whole; part i is the contiguous prefix/suffix of it.
            down_ptr=b + d_off + expert * d_rec + part * d_part,
            w13_n=inter_tp,
            w13_k=self.hidden,
            w2_n=self.hidden,
            w2_k=inter_tp,
        )

    def kt_pointer_arrays(self, layer: int) -> dict[str, list[list[int]]]:
        """``{"gate"|"up"|"down": [[addr per expert] per TP part]}`` for kt.

        Exactly the shape ``KTMoEWrapper(nz_bank_ptrs=...)`` takes, which the C++
        indexes as ``gate_projs[part][expert]``. Built here rather than in kt
        because the geometry is ours and is asserted against the manifest in
        ``_check_geometry``.
        """
        out: dict[str, list[list[int]]] = {"gate": [], "up": [], "down": []}
        for part in range(self.tp_count):
            g, u, d = [], [], []
            for e in range(self.num_experts):
                w = self.weights(layer, e, part)
                g.append(w.gate_ptr)
                u.append(w.up_ptr)
                d.append(w.down_ptr)
            out["gate"].append(g)
            out["up"].append(u)
            out["down"].append(d)
        return out

    def close(self) -> None:
        self._bases.clear()
        for mm in self._maps.values():
            mm.close()
        self._maps.clear()
