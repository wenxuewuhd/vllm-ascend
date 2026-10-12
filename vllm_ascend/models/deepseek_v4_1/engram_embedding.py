# SPDX-License-Identifier: Apache-2.0
"""Engram embedding backed by staged, read-only host tables.

Upstream's ``ParallelEngramEmbedding.__init__`` allocates the whole table
shard, either in pinned host memory (``cpu_offload=True``) or in HBM. At TP=1
that is about 189 GiB per rank, which fits in neither, and the pinned-host
registration path must not be used on this platform.

Instead:

1. The table Parameters are created on the ``meta`` device (right shape and
   dtype, no storage); the data is read from a staged file in ``/dev/shm``.
2. Each forward gathers the needed rows on the host, copies them to HBM in one
   H2D, and a Triton kernel dequantises them. Under graph capture the host
   gather is recorded as a host-function node so it replays with the graph.

   With ``VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH=1`` the gather and the H2D
   run in the model runner before every forward instead
   (``stage_out_of_graph``, driven by ``engram_out_of_graph.prepare``), and
   the forward only dequantises the persistent staging buffer: no
   host-function node, no synchronisation inside the forward.
"""

from __future__ import annotations

import contextlib

import torch

from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout
from vllm_ascend.models.deepseek_v4_1.engram_staging import (DEFAULT_STAGE_DIR,
                                                             staged_file_name)

#: Upstream logs this from inside ``if cpu_offload:`` regardless of what the
#: allocation did; with the allocation diverted the line is untrue.
_STALE_OFFLOAD_LOG = "offloaded to pinned host memory"


class _DropStaleOffloadLine:
    """Filter out upstream's offload INFO line, whose size figure no longer
    describes any real allocation."""

    def filter(self, record) -> bool:
        return _STALE_OFFLOAD_LOG not in record.getMessage()


@contextlib.contextmanager
def _upstream_offload_line_suppressed():
    import logging

    logger = logging.getLogger(
        "vllm.models.deepseek_v4_1.common.engram")
    drop = _DropStaleOffloadLine()
    logger.addFilter(drop)
    try:
        yield
    finally:
        logger.removeFilter(drop)


@contextlib.contextmanager
def tables_not_allocated():
    """Force ``torch.empty`` onto ``meta`` for the duration.

    Scoped to upstream's ``__init__``, whose only ``torch.empty`` calls are the
    table and its scales. Preferred over copying the constructor, which would
    drift silently.
    """
    real_empty = torch.empty

    def empty_on_meta(*sizes, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "meta"
        return real_empty(*sizes, **kwargs)

    torch.empty = empty_on_meta
    try:
        yield
    finally:
        torch.empty = real_empty


def _read_only_table(path: str, rows: int, width: int):
    """A read-only memory map of one staged table."""
    import mmap
    import os

    import numpy as np

    size = os.path.getsize(path)
    if size != rows * width:
        raise ValueError(
            f"{path}: expected {rows:,} x {width} = {rows * width:,} bytes, "
            f"found {size:,}; the staged file does not match the config.")
    handle = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(handle, size, flags=mmap.MAP_SHARED,
                            prot=mmap.PROT_READ)
    finally:
        os.close(handle)
    return np.frombuffer(mapping, dtype=np.uint8).reshape(rows, width), mapping


class AscendStagedEngramEmbedding:
    """Mixin over upstream's ``ParallelEngramEmbedding``.

    Mixed in ahead of it so these win the MRO::

        class AscendEngramEmbedding(AscendStagedEngramEmbedding,
                                    ParallelEngramEmbedding):
            pass
    """

    #: Set once construction has bound the staged tables.
    _staged_tables: tuple | None = None

    def __init__(self, *args, **kwargs) -> None:
        with tables_not_allocated(), _upstream_offload_line_suppressed():
            super().__init__(*args, **kwargs)

        self._assert_nothing_was_allocated()
        self._layer_id = None          # bound by bind_staged_tables()

        # Replaces the suppressed upstream line.
        from vllm.logger import logger
        logger.info(
            "Engram table not allocated: %d rows x %d stays in the staged "
            "read-only mapping, 0 bytes resident and 0 bytes pinned per rank. "
            "Upstream's \"%s\" line is suppressed here because the allocation "
            "it describes does not happen.",
            getattr(self, "part_num_embeddings", -1),
            getattr(self, "dim", -1), _STALE_OFFLOAD_LOG)

        self._allocate_staging_at_construction()

    def _allocate_staging_at_construction(self) -> None:
        """Size the per-step buffers at construction, not on first lookup.

        Construction is the only point guaranteed to precede graph capture,
        and replay reads the addresses recorded at capture. The bound is
        tokens per forward (``max_num_batched_tokens``); ``lookup`` refuses a
        larger batch.
        """
        from vllm.config import get_current_vllm_config
        from vllm.platforms import current_platform

        vllm_config = get_current_vllm_config()
        if vllm_config is None:
            raise RuntimeError(
                "engram staging cannot be sized: no vllm config is current "
                "during construction. Deferring the allocation to the first "
                "lookup would place it after graph capture.")
        layout = EngramLayout.from_config(vllm_config.model_config.hf_text_config)
        self.allocate_staging(
            layout, vllm_config.scheduler_config.max_num_batched_tokens,
            current_platform.device_type)
        self._allocate_pinned_for_capture()
        self.bind_staged_tables(layout, self._own_layer_id(layout))

    def _own_layer_id(self, layout) -> int:
        """Which engram layer this instance is, from its own row count.

        Upstream's constructor carries neither a prefix nor a layer id, but
        the engram layers have different vocabulary sizes. An ambiguous match
        refuses: binding the wrong layer's table would not error anywhere.
        """
        rows = getattr(self, "num_embeddings", None)
        matches = [lid for lid, n in zip(layout.layer_ids, layout.num_embeddings)
                   if n == rows]
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(
            f"cannot tell which engram layer this is: {rows} rows matches "
            f"{matches or 'none'} of {dict(zip(layout.layer_ids, layout.num_embeddings))}. "
            "The identity comes from the row count because upstream's "
            "constructor carries neither a prefix nor a layer id.")

    # -- guards --------------------------------------------------------------

    def _assert_nothing_was_allocated(self) -> None:
        """Check the interception rather than trust it: a successful
        allocation would not raise by itself."""
        for name in ("weight", "weight_scale_inv"):
            param = getattr(self, name, None)
            if param is None:
                continue
            if param.device.type != "meta":
                raise RuntimeError(
                    f"engram {name} was allocated on {param.device} with "
                    f"{param.numel() * param.element_size() / 1024**3:.1f} GiB "
                    "of real storage. It must not be: the table is read from "
                    "the staged mapping, and the pinned-host path must not be "
                    "used. Note that running out of memory here is not the "
                    "guard -- a larger host would have let it through.")

    def _storage(self):
        raise RuntimeError(
            "engram _storage() is the UVA / host-registration entry point and "
            "must not be reached. The engram tables are not device-addressable "
            "on this platform: rows are gathered on the host from a read-only "
            "mapping and copied to the device.")

    # -- the staged tables ---------------------------------------------------

    def bind_staged_tables(self, layout: EngramLayout, layer_id: int,
                           stage_dir: str = DEFAULT_STAGE_DIR) -> None:
        """Map this layer's staged weight and scale tables, read-only."""
        rows = dict(zip(layout.layer_ids, layout.num_embeddings))[layer_id]
        weight, wmap = _read_only_table(
            f"{stage_dir}/{staged_file_name(f'layers.{layer_id}.engram.embed.weight')}",
            rows, layout.dim)
        scale, smap = _read_only_table(
            f"{stage_dir}/{staged_file_name(f'layers.{layer_id}.engram.embed.scale')}",
            rows, layout.n_scales)
        self._staged_tables = (weight, scale, wmap, smap)
        self._layer_id = layer_id

    def staged_tables(self):
        if self._staged_tables is None:
            raise RuntimeError(
                f"engram tables for layer {self._layer_id} are not bound; call "
                "bind_staged_tables() after construction.")
        return self._staged_tables[0], self._staged_tables[1]

    # -- the lookup path -----------------------------------------------------

    def allocate_staging(self, layout: EngramLayout, max_tokens: int,
                         device) -> None:
        """Pre-allocate the per-step buffers once, at the maximum batch, so
        their device addresses stay fixed for graph replay."""
        import numpy as np

        # This rank's columns only. Sizing with the model-wide n_hash_cols
        # would over-allocate by the TP size and, worse, disagree with the
        # row count the fill path computes from part_n_hash_cols.
        (wshape, sshape) = layout.staging_shapes(max_tokens,
                                                 self.part_n_hash_cols)
        self._layout = layout
        self._max_tokens = max_tokens
        self._host_weight = np.empty(wshape, dtype=np.uint8)
        self._host_scale = np.empty(sshape, dtype=np.uint8)
        self._dev_weight = torch.empty(wshape, dtype=torch.uint8, device=device)
        self._dev_scale = torch.empty(sshape, dtype=torch.uint8, device=device)
        self._pin_ready = False

    def _allocate_pinned_for_capture(self) -> None:
        """Pinned host buffers for the graph-capture route.

        Allocated before capture: allocations and synchronous copies are
        rejected while a stream is capturing, and async copies are only
        capturable from pinned memory. Ids get one buffer per dtype so no
        conversion runs inside the capture.
        """
        wshape = tuple(self._dev_weight.shape)
        sshape = tuple(self._dev_scale.shape)
        rows = wshape[0]
        # device="cpu" is explicit: the worker's default device is the NPU.
        self._pin_ids64 = torch.empty(rows, dtype=torch.int64,
                                      device="cpu", pin_memory=True)
        self._pin_ids32 = torch.empty(rows, dtype=torch.int32,
                                      device="cpu", pin_memory=True)
        self._pin_weight = torch.empty(wshape, dtype=torch.uint8,
                                       device="cpu", pin_memory=True)
        self._pin_scale = torch.empty(sshape, dtype=torch.uint8,
                                      device="cpu", pin_memory=True)
        for _name, _t in (("ids64", self._pin_ids64), ("ids32", self._pin_ids32),
                          ("weight", self._pin_weight), ("scale", self._pin_scale)):
            if _t.device.type != "cpu" or not _t.is_pinned():
                raise RuntimeError(
                    f"engram capture buffer {_name} is on {_t.device} "
                    f"pinned={_t.is_pinned()}; the capture route needs "
                    "dense pinned CPU memory or its async copies are not "
                    "capturable.")
        self._pin_ids64_np = self._pin_ids64.numpy()
        self._pin_ids32_np = self._pin_ids32.numpy()
        self._pin_weight_np = self._pin_weight.numpy()
        self._pin_scale_np = self._pin_scale.numpy()
        self._pin_ready = True

    def _submit_under_capture(self, indices: torch.Tensor, out: torch.Tensor,
                              rows: int, weight_table, scale_table) -> None:
        """Record the host gather as a graph node.

        The id D2H and the staged-row H2D become async copies through the
        pinned buffers, and the numpy gather becomes a ``_launch_host_func``
        node that runs on the host at replay. ``rows`` is baked into the
        graph, which is valid because it is a pure function of the token
        count (the gather does no de-duplication).
        """
        import torch_npu

        if out_of_graph_enabled():
            raise RuntimeError(
                "engram reached the host-function capture route with "
                "VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH=1. Under that flag the "
                "rows are staged by the model runner before the forward and "
                "the graph must contain no host-function node.")
        flat = indices.reshape(-1)
        if flat.numel() != rows:
            raise AssertionError(
                f"{flat.numel()} ids for {rows} staging rows -- the row "
                "numbering the gather recomputes would not match. "
                "(EngramLayout.row_index is the only definition of it; ids "
                "must already be narrowed to this rank's columns by "
                "own_columns().)")
        if flat.dtype == torch.int64:
            pin_ids, ids_np = self._pin_ids64, self._pin_ids64_np
        elif flat.dtype == torch.int32:
            pin_ids, ids_np = self._pin_ids32, self._pin_ids32_np
        else:
            raise TypeError(
                f"hash ids are {flat.dtype}; the pinned id buffers are int32 "
                "and int64. Converting here would allocate inside the "
                "capture.")

        stream = torch.npu.current_stream(indices.device)
        _ensure_subscribe_report(stream)

        pin_ids[:rows].copy_(flat, non_blocking=True)

        # The node reads the payload at replay, so it must outlive the graph.
        payload = (weight_table, scale_table, ids_np,
                   self._pin_weight_np, self._pin_scale_np, rows)
        _CAPTURED_PAYLOADS.append(payload)
        torch_npu.npu._launch_host_func(stream, _engram_host_gather, payload)

        self._dev_weight[:rows].copy_(self._pin_weight[:rows],
                                      non_blocking=True)
        self._dev_scale[:rows].copy_(self._pin_scale[:rows],
                                     non_blocking=True)

        self._dequantize_staged(rows, out)

    def own_columns(self, ids: torch.Tensor) -> torch.Tensor:
        """Narrow ``[tokens, n_hash_cols]`` to this rank's hash columns.

        Callers upstream compute all of the model's hash columns (the hash
        kernel writes every column for every token), while the staging buffer
        holds only this rank's. The ids are GLOBAL table row numbers and every
        rank maps the whole table, so narrowing is all that is needed -- no
        rebase by vocab_start, which is what upstream's sharded-buffer
        backends do instead (vllm engram.py:602).
        """
        cols = self.part_n_hash_cols
        self._layout.assert_column_shard_is_well_formed(
            cols, self.n_hash_cols, self.head_start)
        if ids.shape[-1] == cols:
            return ids                      # already narrowed (or TP=1)
        if ids.shape[-1] != self.n_hash_cols:
            raise ValueError(
                f"engram ids have {ids.shape[-1]} columns, expected either "
                f"this rank's {cols} or the model's {self.n_hash_cols}.")
        return ids[..., self.head_start:self.head_start + cols]

    def lookup(self, indices: torch.Tensor, out: torch.Tensor,
               background: bool = False) -> None:
        """Gather on the host, copy once, let the kernel read HBM.

        ``indices`` is ``[num_tokens, n_hash_cols]`` of table rows. Flattened
        row-major it is already in staging order (``EngramLayout.row_index``)
        while the hash columns are undivided, which is checked by
        ``own_columns`` narrowing them to this rank's.
        """
        from vllm_ascend.models.deepseek_v4_1.engram_layout import \
            gather_into_staging

        num_tokens = indices.shape[0]
        rows = num_tokens * self.part_n_hash_cols
        if not rows:
            return

        layout = getattr(self, "_layout", None)
        if layout is None:
            raise RuntimeError(
                "engram staging buffers are not allocated; call "
                "allocate_staging() after construction.")
        indices = self.own_columns(indices)
        if num_tokens > self._max_tokens:
            raise ValueError(
                f"{num_tokens} tokens exceeds the {self._max_tokens} the "
                "staging buffers were sized for. They are allocated once so "
                "their device addresses stay stable for graph replay, so this "
                "cannot grow on demand.")

        if out_of_graph_enabled():
            # The runner staged this step's rows before the forward
            # (stage_out_of_graph); only the dequantisation is part of the
            # forward, so graph replay reads nothing but persistent buffers.
            # Host-side check: runs in eager and at capture, not at replay.
            staged = self._oog_staged_rows
            self._oog_staged_rows = None
            if staged != rows:
                raise RuntimeError(
                    f"engram lookup for {rows} staging rows, but the runner "
                    f"staged {staged} for this forward. With "
                    "VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH=1 the model runner "
                    "must call prepare_engram_out_of_graph() before every "
                    "forward; without it the dequantisation would read the "
                    "previous step's rows.")
            self._dequantize_staged(rows, out)
            return

        weight_table, scale_table = self.staged_tables()

        if _stream_is_capturing():
            if not getattr(self, "_pin_ready", False):
                raise RuntimeError(
                    f"engram lookup for {num_tokens} tokens reached graph "
                    "capture without the pinned capture buffers; call "
                    "_allocate_pinned_for_capture() before capture.")
            self._submit_under_capture(indices, out, rows,
                                       weight_table, scale_table)
            return

        # Synchronising: the hash ids must reach the host to be gathered.
        row_ids = indices.reshape(-1).to("cpu", torch.int64).numpy()

        host_weight = self._host_weight[:rows]
        host_scale = self._host_scale[:rows]
        gather_into_staging(weight_table, scale_table, row_ids,
                            host_weight, host_scale)

        self._dev_weight[:rows].copy_(
            torch.from_numpy(host_weight), non_blocking=True)
        self._dev_scale[:rows].copy_(
            torch.from_numpy(host_scale), non_blocking=True)

        self._dequantize_staged(rows, out)

    #: Rows staged by stage_out_of_graph() for the forward about to run;
    #: consumed (reset to None) by lookup().
    _oog_staged_rows: int | None = None

    def stage_out_of_graph(self, host_ids: torch.Tensor,
                           num_actual_tokens: int, num_tokens: int) -> None:
        """Stage one step's rows before the forward, outside any graph.

        ``host_ids`` is ``[num_actual_tokens, n_hash_cols]`` on the host: the
        D2H copy of the ids the device hash computed this step (the caller,
        ``engram_out_of_graph.prepare``, is the only one that produces it).
        Rows ``[num_actual_tokens * cols, num_tokens * cols)`` -- the padding
        of a padded batch -- are zeroed in HBM, so the kernel writes +0.0 for
        them: an all-zero ue8m0 byte is the fp32 bit pattern 0.
        """
        from vllm_ascend.models.deepseek_v4_1.engram_layout import \
            gather_into_staging

        if _stream_is_capturing():
            raise RuntimeError(
                "engram stage_out_of_graph() ran inside a graph capture; it "
                "synchronises and runs host code, so it must run before the "
                "captured forward.")
        cols = self.part_n_hash_cols
        host_ids = self.own_columns(host_ids)
        if not 0 <= num_actual_tokens <= num_tokens <= self._max_tokens:
            raise ValueError(
                f"engram staging for {num_actual_tokens} real of {num_tokens} "
                f"tokens against buffers sized for {self._max_tokens}.")
        if not getattr(self, "_pin_ready", False):
            raise RuntimeError(
                "engram pinned staging buffers are not allocated; they are "
                "allocated at construction.")
        if (host_ids.device.type != "cpu"
                or tuple(host_ids.shape) != (num_actual_tokens, cols)):
            raise ValueError(
                f"engram host ids are {tuple(host_ids.shape)} on "
                f"{host_ids.device}; expected ({num_actual_tokens}, {cols}) "
                "on the host.")
        rows = num_tokens * cols
        real = num_actual_tokens * cols
        self._oog_staged_rows = None
        if real:
            weight_table, scale_table = self.staged_tables()
            row_ids = host_ids.reshape(-1).numpy().astype("int64")
            gather_into_staging(weight_table, scale_table, row_ids,
                                self._pin_weight_np[:real],
                                self._pin_scale_np[:real])
            # Pinned -> async. The next step's id D2H is synchronous on the
            # same stream, so these copies finish before the host rewrites
            # the pinned buffers.
            self._dev_weight[:real].copy_(self._pin_weight[:real],
                                          non_blocking=True)
            self._dev_scale[:real].copy_(self._pin_scale[:real],
                                         non_blocking=True)
        if rows > real:
            self._dev_weight[real:rows].zero_()
            self._dev_scale[real:rows].zero_()
        self._oog_staged_rows = rows

    def _dequantize_staged(self, rows: int, out: torch.Tensor) -> None:
        """Dequantise the staged rows in HBM into ``out``.

        Reads ``self._dev_weight[:rows]`` and ``self._dev_scale[:rows]``,
        both contiguous, so no hash ids or scale gather are needed.
        """
        from vllm_ascend.ops.triton.engram_dequant_staging import launch

        flat = _flatten_out_rows(out, rows)
        launch(self._dev_weight[:rows], self._dev_scale[:rows], flat,
               dim=self._layout.dim, quant_block=self._layout.quant_block)


# Captured graphs hold nodes that reference these payloads and read them at
# replay; they must live for the whole process.
_CAPTURED_PAYLOADS: list = []
_REPORT_SUBSCRIBED: set = set()


def _ensure_subscribe_report(stream) -> None:
    """Subscribe the stream for host-func callbacks, which are otherwise not
    delivered, without an error.

    torch_npu >= 2.10 already subscribes the capture stream inside NPUGraph
    and a second subscribe raises 107011; only that code is swallowed.
    """
    import torch_npu

    key = int(stream.npu_stream)
    if key in _REPORT_SUBSCRIBED:
        return
    try:
        torch_npu.npu._subscribe_report(stream)
    except RuntimeError as exc:
        if "107011" not in str(exc):
            raise
    _REPORT_SUBSCRIBED.add(key)


def _engram_host_gather(payload) -> None:
    """Graph-node callback: gather this step's rows into the pinned buffers.

    Exceptions must not escape: nothing on the ACL side receives them, and the
    device would silently read the previous step's rows. Log instead.
    """
    import numpy as np

    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    try:
        weight_table, scale_table, ids_np, host_weight, host_scale, rows = \
            payload
        row_ids = ids_np[:rows]
        if row_ids.dtype != np.int64:
            row_ids = row_ids.astype(np.int64)
        gather_into_staging(weight_table, scale_table, row_ids,
                            host_weight[:rows], host_scale[:rows])
    except BaseException as exc:                             # noqa: BLE001
        from vllm.logger import logger
        logger.error(
            "engram host gather callback raised inside a graph node: %r. "
            "The staged rows for this step are NOT this step's rows -- the "
            "device will read whatever the buffer held before.", exc)


def out_of_graph_enabled() -> bool:
    """``VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH=1``: the runner stages the rows
    before the forward and the forward only dequantises them."""
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH == "1"


def _stream_is_capturing() -> bool:
    """Whether a graph capture is in progress, or ``False`` if unaskable.

    Permissive on purpose: an unavailable predicate must not block a path
    that works outside capture.
    """
    for module in (getattr(torch, "npu", None), getattr(torch, "cuda", None)):
        predicate = getattr(module, "is_current_stream_capturing", None)
        if predicate is None:
            continue
        try:
            # ``is True``: a test double returning a Mock must not read as
            # capturing.
            return predicate() is True
        except Exception:
            return False
    return False


def _flatten_out_rows(out: torch.Tensor, rows: int) -> torch.Tensor:
    """View the caller's ``[num_tokens, n_hash_cols, dim]`` buffer as the
    ``[rows, dim]`` the kernel writes.

    ``EngramLayout.row_index`` is ``token * n_hash_cols + col``, the row-major
    order of the first two axes, so the same memory serves both. ``view``
    rather than ``reshape``: a reshape copy would receive the kernel's writes
    and the caller would read stale data.
    """
    if out.ndim == 2:
        if out.shape[0] != rows:
            raise ValueError(
                f"the output buffer has {out.shape[0]} rows and the gather "
                f"produced {rows}. A two-dimensional buffer is already in "
                "gathered-row order, so these have to match.")
        return out
    if out.ndim == 3:
        if out.shape[0] * out.shape[1] != rows:
            raise ValueError(
                f"the output buffer is {tuple(out.shape)}, whose first two "
                f"axes give {out.shape[0] * out.shape[1]} rows, but the gather "
                f"produced {rows}. The buffer is [num_tokens, n_hash_cols, "
                "dim]; a mismatch means the caller and the gather disagree "
                "about how many hash columns a token has.")
        if not out.is_contiguous():
            raise ValueError(
                f"the output buffer {tuple(out.shape)} is not contiguous, so "
                "it cannot be viewed as gathered rows.")
        return out.view(rows, out.shape[-1])
    raise ValueError(
        f"the output buffer has {out.ndim} dimensions; this takes "
        "[rows, dim] or [num_tokens, n_hash_cols, dim].")
