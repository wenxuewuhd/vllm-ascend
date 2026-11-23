"""Build the host-side NZ weight bank, offline and on the host only.

The bank is the single copy of the routed-expert weights that both consumers read:
kt-kernel reads its own TP part in place on the CPU, and the NPU takes the same bytes
through a pinned staging buffer. Nothing in this module touches an NPU -- the
ND -> FRACTAL_NZ step is a pure byte permutation, reverse-engineered and validated
against the production cast in :mod:`vllm_ascend.kt_offload.nz_geometry`.

Why offline rather than at start-up
-----------------------------------
kt's zero-copy path aliases the mapping directly -- ``set_external_weights`` points
``BufferB::b`` at the caller's pages -- so the bytes kt reads must ALREADY be NZ in the
mapping it is handed. Casting on the NPU and pulling 269 GiB back with D2H would cost
start-up time and still leave no mapping to hand kt. Permuting on the host sidesteps
both, and measures at ~1.5 GiB/s single-threaded (one 6.7 GiB layer in 4.5 s).

The frozen layout, and why
--------------------------
``w13`` is stored **split per TP part** (four sections, N = 1152 each), ``w2`` and the
two scales **whole-expert**. One file per layer, group-major: every expert's bytes for
a section are contiguous within that section.

Two measurements decided this, both paired and same-process:

* Splitting w13 per part keeps each part's view contiguous, which is worth **+10.3%**
  of kt's weight-streaming throughput in the production decode regime (402 vs 360
  GB/s, dual-socket, per-layer barrier). 84% of that is the strided access itself, not
  page placement -- remote pages measured <=2%, at the noise floor.
* The price is that the device's whole-expert w13 slab must then be assembled from
  four sources, 640 copies of 18,432 B per expert. In the real staging pipeline that
  costs **nothing**: 4 loader threads one-per-CCD reach 66.5 GiB/s on the host leg,
  the H2D leg is the bottleneck at ~52.8, and the composite is pinned there for both
  layouts (52.79 vs 52.80, A-vs-B -0.07% against a 0.05% noise floor). At 8-16 threads
  the split layout is 3.8-5.6% *faster*.

``w2`` needs no split: TP divides down's ``k``, and in NZ ``gidx`` is the outermost
axis, so part 0 is the contiguous *prefix* of each expert's w2 -- byte-identical to a
split file, 720 whole pages, no page sharing. The scales are never zero-copied by kt
(the checkpoint ships ue8m0 and kt's BufferB stores FP32 or compacted E8M0), so they
stay whole and reach the device as one run per expert.

Two conditions come with the layout and are enforced here, not just documented:

* ``MADV_NOHUGEPAGE`` on every mapping. A 2 MiB page spans 28.4 slabs, so every one
  would hold both parts' bytes and locality collapses to ~50%: measured **-28.3%** vs
  -10.3%, with zero upside (one task touches 160 pages, far inside a 4096-entry L2
  TLB) -- and on this host node 1 cannot allocate them anyway.
* The loader must run **>= 4 threads, one per L3/CCD domain** (see
  :func:`check_loader_placement`). At 1-2 threads the 18 KB segments cost 12-14% of
  prefill staging, and threads packed into one CCD drop the host leg from 66 to 26
  GiB/s.

Per layer, with the deployment's shapes::

    w13_gate_p0  384 x 2,949,120 B   NZ, N = inter/tp = 1152, Kb = hidden/2 = 2560
    w13_gate_p1  384 x 2,949,120 B
    w13_up_p0    384 x 2,949,120 B
    w13_up_p1    384 x 2,949,120 B
    w2           384 x 5,898,240 B   NZ, N = hidden   = 5120, Kb = inter/2 = 1152
    w13_scale    384 x   737,280 B   ND, verbatim from the checkpoint
    w2_scale     384 x   368,640 B   ND, verbatim from the checkpoint
                      ------------
                 7,219,445,760 B = 6.724 GiB per layer, 268.95 GiB for 40 layers

The scales are copied byte for byte: the production hook only does
``reshape(g, n, k // 2, 2).transpose(-3, -2)`` on them, which is a view -- it changes
shape and strides and leaves the bytes as the checkpoint stores them. Only the weights
are permuted.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import mmap
import os
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import regex as re

from vllm_ascend.kt_offload import nz_geometry

#: ``(section, checkpoint projection, which half of that projection's rows)``.
#: The device's w13 rows are ``gate | up`` and TP splits each in half, so the device
#: order is gate_p0, gate_p1, up_p0, up_p1 -- which is exactly ``w1`` then ``w3``.
W13_PARTS = (
    ("w13_gate_p0", "w1", 0),
    ("w13_gate_p1", "w1", 1),
    ("w13_up_p0", "w3", 0),
    ("w13_up_p1", "w3", 1),
)

#: Sections the device assembles its w13 tensor from, in device row order.
W13_SECTIONS = tuple(name for name, _, _ in W13_PARTS)

#: Sections in the order they are laid out inside a layer file.
SECTIONS = W13_SECTIONS + ("w2", "w13_scale", "w2_scale")

#: Device tensors a staging buffer holds, as named by ``dst_base``.
DEVICE_TENSORS = ("w13", "w2", "w13_scale", "w2_scale")

#: Format tag written into the manifest. Bump it whenever the byte layout changes, so
#: a stale bank is refused rather than read as if it were the current one.
BANK_FORMAT = "kt-nz-bank/2-split-w13"

#: Default place for the bank. tmpfs, so pages can never be evicted or swapped
#: (SwapTotal is 0 on this host) while staying ordinary NUMA-accounted memory that
#: mbind/move_pages can place -- which kt's page placement needs and pinned memory
#: cannot give (it is a /dev/davinci_manager device mapping: mbind returns EINVAL).
DEFAULT_BANK_DIR = "/dev/shm/dsv41_nz_bank"

#: Fewer prefill-loader threads than this, or threads sharing an L3 domain, and the
#: 18 KB segment list stops being free. Measured with kt idle, which is the regime
#: streaming prefill actually runs in -- the CPU MoE is not used there.
MIN_LOADER_THREADS = 4

#: Upper bound on the decode prefetch pool. Measured with kt at full tilt: 8 threads
#: took 9 more points off kt than 4 did, and 4 already deliver 17.3 GiB/s against the
#: 4.7 GiB/s a 4-expert-per-step prefetch needs.

_NUM_LAYERS_FALLBACK = 40
_L3_SHARED_CPU_LIST = "/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"
_THREAD_SIBLINGS = "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"


class BankBuildError(RuntimeError):
    """The bank could not be built, or what was built does not match the plan."""


class LoaderPlacementError(RuntimeError):
    """The loader's threads sit where the measured layout guarantees do not hold."""


@dataclasses.dataclass(frozen=True)
class BankLayout:
    """Byte geometry of one layer file. Every size is derived, none hand-written."""

    num_experts: int = 384
    hidden: int = 5120
    intermediate: int = 2304
    group_size: int = 32
    #: TP parts the w13 row axis is split across. 1 stores w13 whole-expert, which is
    #: the layout the freeze rejected; it stays expressible so the comparison can be
    #: re-run without touching the writer.
    tp_count: int = 2

    @property
    def w13_part_n(self) -> int:
        """Rows in one w13 section: one projection's share for one TP part."""
        return self.intermediate // self.tp_count

    @property
    def w13_kb(self) -> int:
        return self.hidden // 2

    @property
    def w2_n(self) -> int:
        return self.hidden

    @property
    def w2_kb(self) -> int:
        return self.intermediate // 2

    @property
    def w13_rows_total(self) -> int:
        """Rows in the device's w13 tensor: gate and up stacked."""
        return 2 * self.intermediate

    @property
    def w13_slab_bytes(self) -> int:
        """Bytes one gidx slab occupies in the *device's* w13 tensor."""
        return self.w13_rows_total * nz_geometry.NZ_C0

    @property
    def w13_part_run_bytes(self) -> int:
        """Bytes one section contributes to one device slab."""
        return self.w13_part_n * nz_geometry.NZ_C0

    @property
    def w13_gidx_count(self) -> int:
        return self.w13_kb // nz_geometry.NZ_C0

    def section_record(self, section: str) -> int:
        """Bytes one expert occupies in ``section``."""
        if section in W13_SECTIONS:
            return self.w13_part_n * self.w13_kb
        if section == "w2":
            return self.w2_n * self.w2_kb
        if section == "w13_scale":
            return self.w13_rows_total * (self.hidden // self.group_size)
        if section == "w2_scale":
            return self.w2_n * (self.intermediate // self.group_size)
        raise KeyError(section)

    def device_record(self, tensor: str) -> int:
        """Bytes one expert occupies in a *device* tensor."""
        if tensor == "w13":
            return self.w13_gidx_count * self.w13_slab_bytes
        return self.section_record(tensor)

    def section_offset(self, section: str) -> int:
        off = 0
        for name in SECTIONS:
            if name == section:
                return off
            off += self.num_experts * self.section_record(name)
        raise KeyError(section)

    def expert_offset(self, section: str, expert: int) -> int:
        if not 0 <= expert < self.num_experts:
            raise IndexError(f"expert {expert} out of range for {self.num_experts}")
        return self.section_offset(section) + expert * self.section_record(section)

    @property
    def bytes_per_expert(self) -> int:
        return sum(self.section_record(s) for s in SECTIONS)

    @property
    def layer_bytes(self) -> int:
        return self.num_experts * self.bytes_per_expert

    def part_of_section(self, section: str) -> int:
        """Which TP part owns ``section`` -- what page placement has to follow."""
        for name, _proj, half in W13_PARTS:
            if name == section:
                return half
        raise KeyError(f"{section} is not split per TP part")

    def check(self) -> BankLayout:
        """Refuse a shape the NZ formula does not hold for, loudly and early."""
        c0 = nz_geometry.NZ_C0
        if self.intermediate % self.tp_count:
            raise BankBuildError(f"intermediate={self.intermediate} is not divisible by tp_count={self.tp_count}")
        for name, n, kb in (
            ("w13 section", self.w13_part_n, self.w13_kb),
            ("w2", self.w2_n, self.w2_kb),
        ):
            if kb % c0:
                raise BankBuildError(
                    f"{name}: Kb={kb} is not a multiple of C0={c0}; the NZ geometry assumes "
                    "whole C0 blocks along the reduction axis"
                )
            if n % c0:
                raise BankBuildError(f"{name}: N={n} is not a multiple of {c0}")
        if self.group_size // 2 != c0:
            # group_size fp4 = group_size/2 bytes must be exactly one C0. That is the
            # coincidence the whole scheme rests on: one 16-byte block is one k-group
            # and shares exactly one scale byte.
            raise BankBuildError(
                f"group_size={self.group_size} fp4 = {self.group_size // 2} B != C0={c0}; "
                "a 16-byte NZ block would no longer be one k-group"
            )
        if self.hidden % self.group_size or self.intermediate % self.group_size:
            raise BankBuildError("hidden and intermediate must both be multiples of group_size")
        # Page placement per TP part is only possible if each part's record is a whole
        # number of pages; otherwise the parts share pages and the +10.3% erodes.
        page = os.sysconf("SC_PAGE_SIZE")
        rec = self.section_record(W13_SECTIONS[0])
        if rec % page:
            raise BankBuildError(
                f"a w13 section record is {rec} B, not a multiple of the {page} B page size: the two TP "
                "parts would share pages and could not be placed per node"
            )
        return self


def manifest_path(bank_dir: str) -> str:
    return os.path.join(bank_dir, "manifest.json")


def layer_path(bank_dir: str, layer: int) -> str:
    return os.path.join(bank_dir, f"layer{layer:03d}.nzbank")


# --------------------------------------------------------------------------- build


def _nd_rows(src: Any, layer: int, expert: int, proj: str, half: int, layout: BankLayout) -> np.ndarray:
    """One TP part's half of a checkpoint weight tensor's rows, as flat ND uint8."""
    buf = src.read(src.locate(layer, expert, proj, "weight")).numpy().reshape(-1)
    rows, cols = layout.intermediate, layout.w13_kb
    want = rows * cols
    if buf.nbytes != want:
        raise BankBuildError(
            f"layer {layer} expert {expert} {proj}.weight: {buf.nbytes} B from the checkpoint, "
            f"expected {want} B for [{rows}, {cols}]"
        )
    per = rows // layout.tp_count
    return buf.reshape(rows, cols)[half * per : (half + 1) * per].reshape(-1)


def _nd_flat(src: Any, layer: int, expert: int, proj: str, kind: str, want: int) -> np.ndarray:
    buf = src.read(src.locate(layer, expert, proj, kind)).numpy().reshape(-1)
    if buf.nbytes != want:
        raise BankBuildError(
            f"layer {layer} expert {expert} {proj}.{kind}: {buf.nbytes} B from the checkpoint, expected {want} B"
        )
    return buf


def expert_sections(src: Any, layer: int, expert: int, layout: BankLayout) -> dict[str, np.ndarray]:
    """The seven byte blocks this expert contributes, already in bank form."""
    out: dict[str, np.ndarray] = {}
    for name, proj, half in W13_PARTS:
        nd = _nd_rows(src, layer, expert, proj, half, layout)
        out[name] = nz_geometry.permute_nd_to_nz(nd, 1, layout.w13_part_n, layout.w13_kb)
    w2_nd = _nd_flat(src, layer, expert, "w2", "weight", layout.w2_n * layout.w2_kb)
    out["w2"] = nz_geometry.permute_nd_to_nz(w2_nd, 1, layout.w2_n, layout.w2_kb)
    # Scales pass through untouched, w1's rows then w3's -- which is the device's
    # gate_p0 | gate_p1 | up_p0 | up_p1 row order.
    half_scale = layout.section_record("w13_scale") // 2
    out["w13_scale"] = np.concatenate(
        [_nd_flat(src, layer, expert, proj, "scale", half_scale) for proj in ("w1", "w3")]
    )
    out["w2_scale"] = _nd_flat(src, layer, expert, "w2", "scale", layout.section_record("w2_scale"))
    for name, arr in out.items():
        rec = layout.section_record(name)
        if arr.nbytes != rec:
            raise BankBuildError(f"layer {layer} expert {expert} {name}: produced {arr.nbytes} B, layout says {rec}")
    return out


def _write_experts_into(mm: mmap.mmap, src: Any, layer: int, layout: BankLayout) -> None:
    """Fill a mapped layer file. Views live and die inside this frame -- see build_layer."""
    dst = np.frombuffer(mm, dtype=np.uint8)
    try:
        for expert in range(layout.num_experts):
            for name, arr in expert_sections(src, layer, expert, layout).items():
                off = layout.expert_offset(name, expert)
                dst[off : off + arr.nbytes] = arr
    finally:
        # Drop the export before unwinding: a traceback keeps this frame alive, so a
        # surviving name here would make the caller's mm.close() raise BufferError and
        # mask whatever actually went wrong.
        del dst


def build_layer(src: Any, layer: int, out_path: str, layout: BankLayout, *, drop_cache: bool = True) -> int:
    """Write one layer's bank file. Returns the bytes written.

    ``drop_cache`` advises the kernel to forget the checkpoint pages this layer
    consumed. Without it the build walks 269 GiB of page cache into a machine that also
    has to hold the bank, and those pages are dead weight the moment the layer is
    written -- in steady state neither consumer reads the checkpoint.
    """
    layout.check()
    total = layout.layer_bytes
    tmp_path = out_path + ".partial"
    with open(tmp_path, "w+b") as fh:
        fh.truncate(total)
        mm = mmap.mmap(fh.fileno(), total)
        try:
            _advise_no_hugepage(mm)
            _write_experts_into(mm, src, layer, layout)
            mm.flush()
        finally:
            mm.close()
        if drop_cache:
            _fadvise_dontneed(fh.fileno(), 0, total)
    os.replace(tmp_path, out_path)
    return total


def _advise_no_hugepage(mm: mmap.mmap) -> None:
    """Pin the mapping to base pages; 2 MiB pages cost 18 points here and buy nothing."""
    with contextlib.suppress(OSError, AttributeError, ValueError):
        mm.madvise(mmap.MADV_NOHUGEPAGE)


def _fadvise_dontneed(fd: int, offset: int, length: int) -> None:
    """Best-effort ``POSIX_FADV_DONTNEED``; a kernel that refuses is not an error."""
    with contextlib.suppress(OSError, AttributeError):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)


def build(
    src: Any,
    bank_dir: str = DEFAULT_BANK_DIR,
    *,
    layers: Sequence[int] | None = None,
    layout: BankLayout | None = None,
    drop_cache: bool = True,
    progress: Any = None,
) -> dict[str, Any]:
    """Build (or extend) a bank and write its manifest. Returns the manifest.

    Passing a subset of ``layers`` is the supported way to validate on a few layers
    first -- a full bank is 268.95 GiB and /dev/shm may not have room for it yet.
    """
    layout = (layout or BankLayout()).check()
    if layers is None:
        layers = range(int(getattr(src, "num_layers", 0)) or _NUM_LAYERS_FALLBACK)
    layers = tuple(int(x) for x in layers)

    os.makedirs(bank_dir, exist_ok=True)
    _require_room(bank_dir, layout.layer_bytes * len(layers))

    built: dict[str, int] = {}
    for layer in layers:
        n = build_layer(src, layer, layer_path(bank_dir, layer), layout, drop_cache=drop_cache)
        built[str(layer)] = n
        if progress is not None:
            progress(layer, n)

    man = {
        "format": BANK_FORMAT,
        "layout": dataclasses.asdict(layout),
        "derived": {
            "bytes_per_expert": layout.bytes_per_expert,
            "layer_bytes": layout.layer_bytes,
            "sections": {s: [layout.section_offset(s), layout.section_record(s)] for s in SECTIONS},
            "w13_slab_bytes": layout.w13_slab_bytes,
            "w13_part_run_bytes": layout.w13_part_run_bytes,
        },
        "geometry": "dst = e*(N*Kb) + (kb//16)*(N*16) + n*16 + (kb%16)",
        "loader_contract": {
            "min_threads": MIN_LOADER_THREADS,
            "placement": "one thread per L3/CCD domain",
            "page_size": "base pages only (MADV_NOHUGEPAGE)",
            "copy_order": "part-major: all of one section's slabs before the next section",
        },
        "layers": built,
    }
    with open(manifest_path(bank_dir), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    return man


def _require_room(bank_dir: str, need: int) -> None:
    st = os.statvfs(bank_dir)
    free = st.f_bavail * st.f_frsize
    if free < need:
        raise BankBuildError(
            f"{bank_dir}: {free / 2**30:.1f} GiB available, need {need / 2**30:.1f} GiB. On this host "
            "/dev/shm is shared with other projects -- check what is using it before making room, and "
            "build a subset of layers with layers=... meanwhile."
        )


def load_manifest(bank_dir: str) -> dict[str, Any]:
    with open(manifest_path(bank_dir), encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("format") != BANK_FORMAT:
        raise BankBuildError(f"{bank_dir}: bank format {man.get('format')!r}, this build writes {BANK_FORMAT!r}")
    return man


def open_layer(bank_dir: str, layer: int) -> tuple[mmap.mmap, int]:
    """Map one layer file read-only and shared -- the ``r--s`` form kt requires.

    kt refuses a PRIVATE+WRITABLE mapping on purpose: ``mlock`` on ``rw-p`` breaks COW
    and materialises an anonymous copy of every page, which gives two copies instead of
    zero. ``MAP_SHARED`` + ``PROT_READ`` both lets kt alias the pages and lets them be
    placed per NUMA node.
    """
    fd = os.open(layer_path(bank_dir, layer), os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    _advise_no_hugepage(mm)
    return mm, size


# ------------------------------------------------------------------- verification


def _compare_experts(
    mm: mmap.mmap, src: Any, layer: int, layout: BankLayout, experts: tuple[int, ...]
) -> dict[str, Any]:
    checked = 0
    for expert in experts:
        for name, arr in expert_sections(src, layer, expert, layout).items():
            off = layout.expert_offset(name, expert)
            # mm[a:b] copies, so nothing here exports a pointer into the mapping -- a
            # view would survive on the traceback of a mismatch and make the caller's
            # mm.close() raise BufferError instead of this error.
            got = np.frombuffer(mm[off : off + arr.nbytes], dtype=np.uint8)
            if not np.array_equal(got, arr):
                bad = int(np.flatnonzero(got != arr)[0])
                raise BankBuildError(
                    f"layer {layer} expert {expert} {name}: mismatch at byte {bad} of {arr.nbytes} "
                    f"(bank offset {off + bad}); expected 0x{int(arr[bad]):02x}, found 0x{int(got[bad]):02x}"
                )
            checked += arr.nbytes
    return {"layer": layer, "experts": experts, "bytes_compared": checked}


def verify_layer(
    src: Any,
    layer: int,
    bank_dir: str,
    layout: BankLayout,
    *,
    experts: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Re-derive a sample of experts and compare the bank byte for byte.

    This checks the *writer*, not the geometry -- the geometry is checked against the
    real cast by :func:`nz_geometry.verify_nz_geometry`. What goes wrong here is an
    offset: a section written at the wrong place, or two experts written on top of each
    other. Comparing whole experts catches both.
    """
    if experts is None:
        n = layout.num_experts
        experts = sorted({0, 1, n // 2, n - 2, n - 1})
    mm, size = open_layer(bank_dir, layer)
    try:
        if size != layout.layer_bytes:
            raise BankBuildError(f"layer {layer}: file is {size} B, layout says {layout.layer_bytes}")
        return _compare_experts(mm, src, layer, layout, tuple(experts))
    finally:
        mm.close()


# ----------------------------------------------------------------- staging segments


@dataclasses.dataclass(frozen=True)
class Segment:
    """One copy from the bank into the device-layout staging buffer."""

    section: str
    src_offset: int
    dst_offset: int
    nbytes: int


def expert_segments(layout: BankLayout, expert: int, slot: int, dst_base: dict[str, int]) -> Iterator[Segment]:
    """The copies that put one expert into ``slot`` of a device-layout staging buffer.

    ``dst_base`` maps a device tensor name (:data:`DEVICE_TENSORS`) to that tensor's
    base offset in the staging buffer; ``slot`` is the expert's index *within the
    chunk*, which is what the staging buffer is indexed by.

    Emitted in **part-major** order -- all of one section's slabs before moving to the
    next section -- which measured 1.4-1.6% faster than interleaving the four sections
    within each slab, above the noise floor at both thread counts tested. Four
    sequential streams beat one four-way-interleaved stream.
    """
    run = layout.w13_part_run_bytes
    slab = layout.w13_slab_bytes
    for lane, name in enumerate(W13_SECTIONS):
        src0 = layout.expert_offset(name, expert)
        dst0 = dst_base["w13"] + slot * layout.device_record("w13") + lane * run
        for gidx in range(layout.w13_gidx_count):
            yield Segment(name, src0 + gidx * run, dst0 + gidx * slab, run)
    for name in ("w2", "w13_scale", "w2_scale"):
        rec = layout.section_record(name)
        yield Segment(name, layout.expert_offset(name, expert), dst_base[name] + slot * rec, rec)


def chunk_segments(layout: BankLayout, experts: Sequence[int], dst_base: dict[str, int]) -> list[Segment]:
    """Segments for one streaming chunk. ``experts`` need not be contiguous ids."""
    missing = [t for t in DEVICE_TENSORS if t not in dst_base]
    if missing:
        raise BankBuildError(f"dst_base is missing device tensors: {missing}")
    out: list[Segment] = []
    for slot, expert in enumerate(experts):
        out.extend(expert_segments(layout, expert, slot, dst_base))
    return out


def validate_segments(segments: Sequence[Segment], src_bytes: int, dst_bytes: int) -> dict[str, Any]:
    """Check a segment table ONCE, when it is built. Never in the copy loop.

    A full prefill pass moves 640 segments x 384 experts x 40 layers = ~9.83M
    segments. A per-copy assertion there would cost more than the layout saves,
    so the safety has to come from an O(n) pass over the TABLE, which is built
    once and reused for every chunk of every layer.

    What it proves, which is exactly the criterion the 32 bank unit tests apply
    to the writer:
      * every source read is inside the layer file;
      * every destination write is inside the staging buffer;
      * the destination is covered **exactly once** -- no overlap (which would
        silently win by whichever copy landed last) and no gap (which would
        leave the previous chunk's bytes in place, also silently).

    Raises :class:`BankBuildError` on the first problem, naming it. Returns the
    coverage summary so a caller can log what it validated.
    """
    if not segments:
        raise BankBuildError(
            "empty segment table: a chunk that copies nothing would "
            "leave the staging buffer holding the previous chunk's bytes"
        )
    # Destination coverage by interval merge: O(n log n), no dst-sized bitmap
    # (the staging buffer is ~600 MB and this runs on the start-up path).
    spans = sorted((s.dst_offset, s.dst_offset + s.nbytes) for s in segments)
    covered = 0
    prev_end = None
    for lo, hi in spans:
        if lo < 0 or hi > dst_bytes:
            raise BankBuildError(f"segment writes [{lo}, {hi}) outside the {dst_bytes} B staging buffer")
        if prev_end is not None and lo < prev_end:
            raise BankBuildError(
                f"segments overlap in the destination at offset {lo} (previous ended {prev_end}); "
                "two copies would write the same staging bytes and the later one would win silently"
            )
        covered += hi - lo
        prev_end = hi
    for seg in segments:
        if seg.src_offset < 0 or seg.src_offset + seg.nbytes > src_bytes:
            raise BankBuildError(
                f"segment {seg.section} reads [{seg.src_offset}, {seg.src_offset + seg.nbytes}) "
                f"outside the {src_bytes} B layer file"
            )
    if covered != dst_bytes:
        raise BankBuildError(
            f"segments cover {covered} of {dst_bytes} staging bytes ({dst_bytes - covered} B left "
            "uncovered); the gap would still hold the previous chunk's weights"
        )
    return {
        "segments": len(segments),
        "dst_bytes_covered": covered,
        "contiguous_from_zero": spans[0][0] == 0,
        **segment_stats(segments),
    }


def segment_stats(segments: Sequence[Segment]) -> dict[str, Any]:
    """Shape of a segment list: count, total, and the smallest run.

    The 18,432 B w13 runs are far below the ~256 KB size at which a small entry
    batched next to a large one can stall a transfer -- harmless because these are
    *host* memcpys, not device transfers; the device still sees whole contiguous
    tensors out of the staging buffer.
    """
    sizes = [s.nbytes for s in segments]
    return {
        "count": len(segments),
        "bytes": sum(sizes),
        "min_bytes": min(sizes) if sizes else 0,
        "max_bytes": max(sizes) if sizes else 0,
    }


# ------------------------------------------------------------------ loader contract


def _read_cpu_list(path: str, cpu: int, what: str) -> list[int]:
    """Parse a sysfs cpu list (``0-7,128-135``) into every cpu it names."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read().strip()
    except OSError as exc:
        raise LoaderPlacementError(f"cannot read the {what} of cpu {cpu}: {exc}") from exc
    out: list[int] = []
    for piece in raw.split(","):
        if not piece:
            continue
        lo, _, hi = piece.partition("-")
        if not re.fullmatch(r"\d+", lo) or (hi and not re.fullmatch(r"\d+", hi)):
            raise LoaderPlacementError(f"cpu {cpu}: cannot parse {what} {raw!r}")
        out.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    if not out:
        raise LoaderPlacementError(f"cpu {cpu}: empty {what}")
    return out


def l3_domain_of(cpu: int) -> int:
    """Id of the L3/CCD domain ``cpu`` belongs to: the lowest cpu sharing its L3."""
    return min(_read_cpu_list(_L3_SHARED_CPU_LIST.format(cpu=cpu), cpu, "L3 domain"))


def check_loader_placement(cpus: Sequence[int], *, min_threads: int = MIN_LOADER_THREADS) -> dict[str, Any]:
    """Refuse a loader placement the layout's guarantees were not measured at.

    This is the **streaming prefill** loader, which runs while the CPU MoE is idle --
    prefill routes everything through the NPU. On that otherwise-free machine the split
    layout is free *only* in the configuration it was measured in, and there are two
    ways to lose it, both measured: fewer than four threads costs 12-14% of prefill
    staging (the host leg drops below the H2D leg and stops being hidden), and threads
    packed into a single CCD drop the host leg from 66 to 26 GiB/s. Neither failure is
    visible in a correctness test, so it is asserted at start-up instead.

    The decode prefetch pool runs in the opposite regime -- kt at full tilt -- where
    the L3 spread stops mattering and two other rules take over. Use
    a pool of its own would need a check of its own.
    """
    cpus = tuple(int(c) for c in cpus)
    if len(set(cpus)) != len(cpus):
        raise LoaderPlacementError(f"loader cpu list has duplicates: {cpus}")
    if len(cpus) < min_threads:
        raise LoaderPlacementError(
            f"the bank loader needs at least {min_threads} threads, got {len(cpus)}: below that the 18 KB "
            "segment list stops being hidden behind the H2D leg and costs 12-14% of prefill staging"
        )
    domains = sorted({l3_domain_of(c) for c in cpus})
    if len(domains) < min_threads:
        raise LoaderPlacementError(
            f"the bank loader's {len(cpus)} threads span only {len(domains)} L3/CCD domain(s) ({domains}); "
            f"it needs at least {min_threads}. Threads packed into one CCD measured 26 GiB/s on the host "
            "leg against 66 GiB/s spread one per CCD"
        )
    return {"cpus": cpus, "l3_domains": domains, "threads": len(cpus)}


def one_per_l3_cpus(count: int, *, candidates: Sequence[int] | None = None) -> list[int]:
    """Pick ``count`` cpus in distinct L3/CCD domains -- the measured best placement."""
    if candidates is None:
        candidates = range(os.cpu_count() or 0)
    picked: list[int] = []
    seen: set[int] = set()
    for cpu in candidates:
        dom = l3_domain_of(cpu)
        if dom in seen:
            continue
        seen.add(dom)
        picked.append(cpu)
        if len(picked) == count:
            return picked
    raise LoaderPlacementError(
        f"only {len(picked)} distinct L3/CCD domains found among the candidate cpus, need {count}"
    )
