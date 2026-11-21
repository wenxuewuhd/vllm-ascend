# SPDX-License-Identifier: Apache-2.0
"""Guards for the ONE consumer that reads bank bytes in place.

`bank_kt_source` turns manifest numbers into RAW POINTERS that go to kt's C++ as
``nz_bank_ptrs`` and are dereferenced with no length. Nothing else checks them:
the NPU side goes through ``validate_segments``, this side goes through
``_check_geometry`` and nothing more. The module had no test file at all until
2026-11-21, which is why both defects below shipped.
"""

import json
import os

import pytest

from vllm_ascend.kt_offload import bank_build
from vllm_ascend.kt_offload.bank_kt_source import BankKtSourceError, KtBankSource


def _real_sections():
    """The shipped layout, read from BankLayout rather than hard-coded."""
    lay = bank_build.BankLayout()
    names = list(bank_build.W13_SECTIONS) + ["w2", "w13_scale", "w2_scale"]
    return lay, {n: (lay.section_offset(n), lay.section_record(n)) for n in names}


def _src(sections, *, layer_bytes=None, num_experts=None):
    """A source with the geometry fields set and nothing mapped."""
    lay, _ = _real_sections()
    s = object.__new__(KtBankSource)
    s.bank_dir = "/nonexistent/bank"
    s.tp_count = lay.tp_count
    s.hidden = lay.hidden
    s.intermediate = lay.intermediate
    s.num_experts = lay.num_experts if num_experts is None else num_experts
    s.bytes_per_expert = lay.bytes_per_expert
    s.layer_bytes = lay.layer_bytes if layer_bytes is None else layer_bytes
    s.page = os.sysconf("SC_PAGE_SIZE")
    s.sections = dict(sections)
    return s


def test_the_shipped_layout_is_accepted():
    """The baseline: every check below must pass on the real geometry, or the
    negative tests prove nothing."""
    _lay, sec = _real_sections()
    _src(sec)._check_geometry()


def test_a_section_that_leaves_the_mapping_is_refused():
    """The bound. The shipped sections tile the layer exactly, so moving the LAST
    one puts its final expert's record past the end of the mapping -- and that
    address is handed to C++ and dereferenced with no length."""
    _lay, sec = _real_sections()
    off, rec = sec["w2_scale"]
    sec["w2_scale"] = (off + 4096, rec)
    with pytest.raises(BankKtSourceError, match="does not fit the layer"):
        _src(sec)._check_geometry()


def test_the_w2_shift_that_used_to_slip_through_is_refused():
    """🔴 The defect, as reproduced: shifting w2 by 256 KiB -- still page aligned,
    record unchanged -- passed every check there was. Here it is caught by the
    overlap rule, because w13_scale begins exactly where w2 ends: the shifted w2
    would be read over the scales, and the scales out of w2's last records."""
    _lay, sec = _real_sections()
    off, rec = sec["w2"]
    sec["w2"] = (off + 256 * 1024, rec)
    with pytest.raises(BankKtSourceError, match="overlap"):
        _src(sec)._check_geometry()


def test_a_missing_section_is_refused():
    _lay, sec = _real_sections()
    del sec["w2"]
    with pytest.raises(BankKtSourceError, match="missing section w2"):
        _src(sec)._check_geometry()


def test_an_unaligned_offset_is_refused():
    """Page alignment is not cosmetic: two TP parts sharing a page cap locality
    at 1/tp_count local, measured as -28.3%."""
    _lay, sec = _real_sections()
    off, rec = sec["w2"]
    sec["w2"] = (off + 64, rec)
    with pytest.raises(BankKtSourceError, match="not page aligned"):
        _src(sec)._check_geometry()


def test_a_stale_bank_format_is_refused(tmp_path):
    """🔴 The second defect. This consumer read manifest.json directly, so a bank
    written by a different format revision was accepted in place. Any revision
    that keeps the record sizes -- a different C0, a different within-section
    order, a different fractal order -- passes _check_geometry, and kt then reads
    correctly-sized, wrongly-ordered bytes with nothing to catch it."""
    man = {
        "format": "kt-nz-bank/1-whole-w13-TOTALLY-DIFFERENT-BYTE-ORDER",
        "num_experts": 384,
        "layers": {},
        "derived": {"bytes_per_expert": 1, "layer_bytes": 1, "sections": {}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(BankKtSourceError, match="bank format"):
        KtBankSource(str(tmp_path), tp_count=2, hidden=5120, intermediate=2304)


def test_a_missing_manifest_is_refused(tmp_path):
    with pytest.raises(BankKtSourceError, match="is not a bank"):
        KtBankSource(str(tmp_path), tp_count=2, hidden=5120, intermediate=2304)
