# SPDX-License-Identifier: Apache-2.0
"""G′: a chunk's RESIDENT members come from HBM, not from the host.

Streaming prefill moves all 384 experts of all 40 layers on every request --
268.9 GiB, the whole bank -- and ~100 of them are already in HBM. This path
fills those slots of the chunk buffer with a device-to-device copy out of the
resident parameter instead, which removes 26% of the bytes from both the host
read and the H2D.

🔴 It is done at this level, inside the provider, so that everything downstream
still sees a COMPLETE chunk and the output stays **bit-identical**: the chunk
partition, the slot tables and the fp32 accumulation order are untouched, and
only the provenance of some slots' bytes changes. That claim is what the on-card
7-prompt capture tests. What is testable off-card, and tested here, is the
bookkeeping that would silently break it:

  * which positions of a chunk are streamed and which are not;
  * that an impossible or ambiguous binding is refused rather than ignored;
  * that the runs the two fills use cover every position exactly once (a
    position covered zero times is the PREVIOUS chunk's bytes under live
    routes -- plausible numbers, wrong answer, no error).
"""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.kt_offload import stream_io
from vllm_ascend.kt_offload.stream_chain import SLOT_WEIGHT_ATTRS
from vllm_ascend.kt_offload.stream_io import StreamIOError, StreamSlotProvider

NUM_EXPERTS = 8
PER_NZ = 64            # bytes per slot in the (fake) NZ weight buffers


class _Resident(torch.nn.Module):
    """A module with the four parameters, sized for ``slots`` experts.

    The NZ weights are flat byte buffers of PER_NZ per slot, which is what the
    provider's per-slot arithmetic works on; the scales are ordinary ND rows.
    """

    def __init__(self, slots: int = 4, per: int = PER_NZ):
        super().__init__()
        for n in stream_io.NZ_WEIGHT_ATTRS:
            setattr(self, n, torch.zeros((slots, per), dtype=torch.uint8))
        for n in stream_io.ND_SCALE_ATTRS:
            setattr(self, n, torch.zeros((slots, PER_ND), dtype=torch.uint8))


CHUNK_SLOTS = 4
PER_ND = 16            # bytes per slot in the (fake) ND scale buffers


def _prov(nz: bool = True):
    """A provider with only the fields the bookkeeping under test reads.

    Built without __init__ on purpose: the real one allocates device buffers
    through the seam, and none of that is what these tests are about. The ND
    scale buffers are here because a resident scale row is copied into THEM
    (through the storage -- the parameter's own scale tensor is a transposed
    view, see _fill_resident).
    """
    p = object.__new__(StreamSlotProvider)
    p._nz_device = nz
    p._nz_bytes = {n: PER_NZ for n in stream_io.NZ_WEIGHT_ATTRS}
    p.chunk_experts = CHUNK_SLOTS
    p._device = {n: torch.zeros((CHUNK_SLOTS, PER_ND), dtype=torch.uint8)
                 for n in stream_io.ND_SCALE_ATTRS}
    p._res_module = None
    p._res_slot = None
    p._res_per = {}
    p.coverage_from_hbm = 0
    return p


def test_off_by_default_everything_streams():
    p = _prov()
    assert p._split_chunk((0, 1, 2, 3)) == ((0, 1, 2, 3), ())


def test_split_names_the_resident_positions_only():
    p = _prov()
    l2s = [-1, 0, -1, 1, -1, -1, -1, -1]          # experts 1 and 3 are in HBM
    p.set_resident(_Resident(2), l2s)
    assert p._split_chunk((0, 1, 2, 3)) == ((0, 2), (1, 3))
    assert p._split_chunk((4, 5, 6, 7)) == ((0, 1, 2, 3), ())


def test_set_resident_needs_nz_device_mode():
    p = _prov(nz=False)
    with pytest.raises(StreamIOError, match="needs NZ device mode"):
        p.set_resident(_Resident(2), [-1] * NUM_EXPERTS)


def test_set_resident_refuses_a_missing_module_or_parameter():
    p = _prov()
    with pytest.raises(StreamIOError, match="needs the module"):
        p.set_resident(None, [-1] * NUM_EXPERTS)
    incomplete = _Resident(2)
    delattr(incomplete, "w2_weight_scale")
    with pytest.raises(StreamIOError, match="has no w2_weight_scale"):
        p.set_resident(incomplete, [-1] * NUM_EXPERTS)


def test_set_resident_refuses_a_per_slot_size_that_does_not_match_the_chunk():
    # Both buffers come out of the same process_weights_after_loading, so a
    # mismatch is a layout divergence, not something to rescale.
    p = _prov()
    with pytest.raises(StreamIOError, match="per slot"):
        p.set_resident(_Resident(2, per=PER_NZ + 16), [-1] * NUM_EXPERTS)


def test_set_resident_refuses_a_slot_past_the_resident_parameter():
    p = _prov()
    with pytest.raises(StreamIOError, match="past the resident parameter"):
        p.set_resident(_Resident(2), [0, 1, 2, -1, -1, -1, -1, -1])


def test_none_turns_it_back_off():
    p = _prov()
    p.set_resident(_Resident(2), [-1, 0, -1, 1, -1, -1, -1, -1])
    p.set_resident(None, None)
    assert p._split_chunk((0, 1, 2, 3)) == ((0, 1, 2, 3), ())
    assert p._res_module is None and p._res_per == {}


def test_an_id_outside_the_residency_map_is_refused():
    p = _prov()
    p.set_resident(_Resident(2), [-1, 0, -1, 1, -1, -1, -1, -1])
    with pytest.raises(StreamIOError, match="outside the 8-entry residency map"):
        p._split_chunk((7, 8))


def test_the_runs_the_two_fills_use_cover_each_position_once():
    """The arithmetic the streamed and resident fills share.

    Not a tautology: `merge_runs` is the fold's and it refuses a repeated
    destination, which is what makes "covered exactly once" checkable.
    """
    from vllm_ascend.kt_offload.fold_swap import merge_runs

    l2s = [-1, 3, -1, -1, 0, -1, 1, 2]
    p = _prov()
    p.set_resident(_Resident(4), l2s)
    ids = (4, 5, 6, 7)
    stream_pos, res_pos = p._split_chunk(ids)
    assert stream_pos == (1,) and res_pos == (0, 2, 3)
    runs = merge_runs([(q, q) for q in stream_pos]) + merge_runs([(q, l2s[ids[q]]) for q in res_pos])
    covered: list[int] = []
    for first, _src, n in runs:
        covered.extend(range(first, first + n))
    assert sorted(covered) == list(range(len(ids)))
    # Two copies for three resident slots: chunk positions 2,3 map to resident
    # slots 1,2, which advance together.
    assert merge_runs([(q, l2s[ids[q]]) for q in res_pos]) == [(0, 0, 1), (2, 1, 2)]


# --------------------------------------------------------------------------- the binding


class _Prov:
    def __init__(self, can: bool = True):
        self.seen: list[tuple] = []
        if can:
            self.set_resident = lambda module, l2s: self.seen.append((module, l2s))


class _Placement:
    def __init__(self, l2s):
        self._l2s = list(l2s)
        self.reads = 0

    def logical_to_slot(self):
        self.reads += 1
        return torch.tensor(self._l2s, dtype=torch.int64)


def _fake_self(prov, *, want: bool, l2s):
    import types

    from vllm_ascend.kt_offload.layer import KTLayerOffload

    ns = types.SimpleNamespace(
        _stream_provider=prov,
        cfg=types.SimpleNamespace(stream_resident_from_hbm=want),
        placement=_Placement(l2s),
        layer_idx=7,
        _resident_half_logged=False,
    )
    ns.bind = lambda layer: KTLayerOffload._bind_resident_half(ns, layer)
    return ns


def test_binding_is_a_no_op_when_the_switch_is_off():
    prov = _Prov()
    ns = _fake_self(prov, want=False, l2s=[-1, 0, 1])
    ns.bind(object())
    # Explicitly turned OFF on the provider rather than left alone: the provider
    # outlives the forward, and a stale binding would skip slots on a layer that
    # did not ask for it.
    assert prov.seen == [(None, None)]
    assert ns.placement.reads == 0


def test_binding_passes_the_snapshot_once():
    prov = _Prov()
    ns = _fake_self(prov, want=True, l2s=[-1, 0, -1, 1])
    layer = object()
    ns.bind(layer)
    assert prov.seen == [(layer, [-1, 0, -1, 1])]
    assert ns.placement.reads == 1, "the membership snapshot must be taken exactly once per forward"


def test_binding_refuses_a_provider_that_cannot_fill_a_skipped_slot():
    prov = _Prov(can=False)
    ns = _fake_self(prov, want=True, l2s=[-1, 0])
    with pytest.raises(RuntimeError, match="nothing can fill the slots the host would skip"):
        ns.bind(object())


def test_a_fully_resident_chunk_is_the_common_case_not_an_error():
    """🔴 Under the shipped config MOST chunks are fully resident.

    `resident_strategy: prefix` with num_resident_experts=100 of 384 and
    chunk_experts=32 makes chunks 0, 1 and 2 (experts 0..95) entirely resident
    and chunk 3 mixed. The first version of this path raised on such a chunk
    ("the host gather would have nothing to do"), which would have failed the
    very first streamed forward of the very first on-card arm. The split must
    simply report that nothing is streamed.
    """
    p = _prov()
    l2s = [i if i < 4 else -1 for i in range(NUM_EXPERTS)]     # 4 of 8 resident
    p.set_resident(_Resident(4), l2s)
    assert p._split_chunk((0, 1, 2, 3)) == ((), (0, 1, 2, 3))
    assert p._split_chunk((4, 5, 6, 7)) == ((0, 1, 2, 3), ())
    assert p._split_chunk((2, 3, 4, 5)) == ((2, 3), (0, 1))


def test_the_shipped_geometry_makes_three_chunks_fully_resident():
    """The arithmetic that made the above a blocking bug rather than an edge."""
    n, chunk, res = 384, 32, 100
    fully = [k for k in range(n // chunk)
             if all(e < res for e in range(k * chunk, (k + 1) * chunk))]
    mixed = [k for k in range(n // chunk)
             if any(e < res for e in range(k * chunk, (k + 1) * chunk)) and k not in fully]
    assert fully == [0, 1, 2] and mixed == [3]


# ------------------------------------------------------- slot_for's control flow
#
# 🔴 These exist because BOTH bugs that reached the card were in slot_for's
# ordering, not in the split: first a refusal of the fully-resident chunk (which
# is the COMMON case), then an `UnboundLocalError` because the fully-resident
# branch ran before `key` was assigned. Neither was reachable from a test that
# only called _split_chunk, so neither was caught. These drive slot_for itself
# with the device-touching parts stubbed.


def _drivable(nz: bool = True):
    """A provider whose slot_for can run: the device work is recorded, not done."""
    p = _prov(nz)
    p.chunk_experts = CHUNK_SLOTS
    p.coverage = stream_io.SlotCoverage()
    p._stage_s = {"acquire": 0.0, "upload": 0.0, "format": 0.0}
    p._stage_n = 0
    p._stage_h2d_bytes = 0
    p._stage_hbm_slots = 0
    p._stage_full_chunks = 0
    p._stage_mixed_chunks = 0
    p._stage_d2d_calls = 0
    p._stage_d2d_bytes = 0
    p._device_key = None
    p._layer = 3
    # _check_subset refuses a source that cannot gather part of a chunk; that
    # refusal is part of the contract, so the stub has to declare it can.
    import types

    p.source = types.SimpleNamespace(supports_subset_gather=True)
    p.filled: list[tuple] = []
    p.formatted: list[int] = []
    p._nz_h2d_stream = None
    # Returns the stream it wrote on, as the real one does: a fully resident
    # chunk has no H2D, so this is the only write the fold can be ordered
    # against and slot_for must publish it.
    FAKE_STREAM = 4242

    def _fill(ids, pos, n):
        p.filled.append((tuple(ids), tuple(pos), n))
        return FAKE_STREAM

    p._fill_resident = _fill
    p.fake_stream = FAKE_STREAM
    p._format = lambda n: (p.formatted.append(n) or "SLOT")
    return p


def test_slot_for_on_a_fully_resident_chunk_touches_no_host_buffer():
    p = _drivable()
    p.set_resident(_Resident(CHUNK_SLOTS), [0, 1, 2, 3])
    ids = (0, 1, 2, 3)
    slot = p.slot_for(0, ids)
    assert slot == "SLOT"
    assert p.filled == [(ids, (0, 1, 2, 3), 4)]
    # The key records that the host is responsible for NOTHING -- which is a
    # different claim from None ("the whole chunk"), and served_nz/served_nd
    # compare against it.
    assert p._device_key == (3, 0, ids, ())
    assert p._served_subset() == ()
    assert p._stage_hbm_slots == 4 and p._stage_n == 1
    assert p._stage_full_chunks == 1 and p._stage_mixed_chunks == 0
    # 🔴 served_nz refuses a chunk with no stream to order against, and a fully
    # resident chunk never runs an H2D -- so the D2D's stream must be published.
    assert p._nz_h2d_stream == p.fake_stream
    assert p.coverage.chunks_served == 1
    assert p.coverage.experts_produced == {(3, e) for e in ids}


def test_slot_for_refuses_a_subset_and_a_residency_at_once():
    p = _drivable()
    p.set_resident(_Resident(2), [-1, 0, -1, 1])
    with pytest.raises(StreamIOError, match="one too many"):
        p.slot_for(0, (0, 1, 2, 3), only_slots=[0])


def test_slot_for_on_a_MIXED_chunk_streams_only_the_rest_and_fills_the_residents():
    """Chunk 3 of every layer is mixed under the shipped config (28 + 4).

    The host must be asked for exactly the streamed positions -- that subset is
    what the gather fills and what the H2D then moves -- and the resident
    positions must be filled afterwards, on the same stream.
    """
    p = _drivable()
    p.set_resident(_Resident(2), [-1, 0, -1, 1])          # positions 1 and 3 resident
    ids = (0, 1, 2, 3)
    asked: list = []

    class _Layout:
        experts = ids
        produced = (0, 2)

        def produced_slots(self):
            return (0, 2)

        def views(self, buf):
            return {}

    class _Host:
        buffer = torch.zeros(8, dtype=torch.uint8)

        class gather:
            layout = _Layout()

    def _acquire(key):
        asked.append(key)
        return _Host()

    p._acquire = _acquire
    p._check_record_bytes = lambda layout, n: None
    # The real _upload_nz is what publishes the key once every group has been
    # copied; the stub keeps that part so the staleness check downstream
    # (served_nz / served_nd) is exercised too.
    def _upload_nz(host, views, n, key):
        p._device_key = key

    p._upload_nz = _upload_nz
    p.probe_bytes = 0
    p._nz_device = True

    slot = p.slot_for(3, ids)
    assert slot == "SLOT"
    # The host was asked for the streamed positions ONLY, and that is in the key.
    assert asked == [(3, 3, ids, (0, 2))]
    assert p.filled == [(ids, (1, 3), 4)]
    assert p._device_key == (3, 3, ids, (0, 2))
    assert p._stage_mixed_chunks == 1 and p._stage_full_chunks == 0
