"""``KTOffloadConfig`` refusals that have to fire before a card is booked.

Every check here is one whose absence would produce a reading nobody can
interpret rather than an error.
"""

import pytest


# dynamic_resident needs kt to actually hold every expert's CPU weights; a kt
# build without KT_FULLSET_LOAD ignores the export, silently, and reads "no gain"
def _dyn_cfg(**kw):
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    base = dict(
        enabled=True,
        cpuinfer_threads=8,
        num_resident_experts=100,
        overlap_side_stream=False,
        stream_prefill=True,
        dynamic_resident=True,
    )
    base.update(kw)
    return KTOffloadConfig(**base)


class _FakeAmx:
    def __init__(self, answer=None, has_probe=True):
        self.__file__ = "/fake/kt_kernel/utils/amx.py"
        if has_probe:
            self._fullset_load_requested = lambda: answer


def _with_amx(monkeypatch, module):
    """Make ``from kt_kernel.utils import amx`` inside the method yield *module*."""
    import sys
    import types

    pkg = types.ModuleType("kt_kernel")
    utils = types.ModuleType("kt_kernel.utils")
    utils.amx = module
    pkg.utils = utils
    monkeypatch.setitem(sys.modules, "kt_kernel", pkg)
    monkeypatch.setitem(sys.modules, "kt_kernel.utils", utils)
    if module is not None:
        monkeypatch.setitem(sys.modules, "kt_kernel.utils.amx", module)


def test_full_set_load_off_is_refused(monkeypatch):
    _with_amx(monkeypatch, _FakeAmx(answer=False))
    with pytest.raises(ValueError, match="KT_FULLSET_LOAD=1"):
        _dyn_cfg()._refuse_without_full_set_load()


def test_full_set_load_on_is_accepted(monkeypatch):
    _with_amx(monkeypatch, _FakeAmx(answer=True))
    _dyn_cfg()._refuse_without_full_set_load()  # must not raise


def test_a_kt_build_without_the_switch_is_refused(monkeypatch):
    """A kt build with no KT_FULLSET_LOAD at all ignores the export silently, so
    the run reads as "no change" instead of failing: refuse at start-up instead.
    """
    _with_amx(monkeypatch, _FakeAmx(has_probe=False))
    with pytest.raises(ValueError, match="kt-kernel patch"):
        _dyn_cfg()._refuse_without_full_set_load()


def test_an_unimportable_kt_is_refused(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def boom(name, *a, **kw):
        if name.startswith("kt_kernel"):
            raise ImportError("no kt here")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", boom)
    with pytest.raises(ValueError, match="not importable"):
        _dyn_cfg()._refuse_without_full_set_load()


def test_the_check_stands_down_when_dynamic_resident_is_off(monkeypatch):
    """Zero coverage of the probe is correct here, and must not raise."""
    _with_amx(monkeypatch, _FakeAmx(answer=False))
    _dyn_cfg(dynamic_resident=False)._refuse_without_full_set_load()


# zerocopy_w13 needs a kt build that HAS the switch: KT_ZEROCOPY_* are read
# inside kt's compiled extension, so exporting them against an older build is
# ignored without a word and the run silently takes the copy path
def _zc_cfg(**kw):
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    base = dict(
        enabled=True, cpuinfer_threads=8, num_resident_experts=100, overlap_side_stream=False, zerocopy_w13=True
    )
    base.update(kw)
    return KTOffloadConfig(**base)


def _with_kt_ext(monkeypatch, tmp_path, blob: bytes | None):
    """Make ``import kt_kernel`` yield a package whose ext .so holds *blob*.

    ``blob is None`` means "package directory with no extension in it", which
    is the case where the build cannot be probed at all.
    """
    import sys
    import types

    pkg_dir = tmp_path / "kt_kernel"
    pkg_dir.mkdir()
    if blob is not None:
        (pkg_dir / "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so").write_bytes(blob)
    pkg = types.ModuleType("kt_kernel")
    pkg.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "kt_kernel", pkg)


# Padding around the names so a match cannot come from the test's own framing.
_NOISE = b"\x7fELF" + bytes(4096)


def test_a_build_carrying_both_switch_names_is_accepted(monkeypatch, tmp_path):
    _with_kt_ext(
        monkeypatch, tmp_path, _NOISE + b"KT_ZEROCOPY_WEIGHTS\x00" + _NOISE + b"KT_ZEROCOPY_SCOPE\x00" + _NOISE
    )
    _zc_cfg()._refuse_without_zerocopy_support()  # must not raise


@pytest.mark.parametrize(
    "blob, match",
    [
        # a build that predates zero-copy must not be used silently
        (_NOISE + b"KT_FULLSET_LOAD\x00" + _NOISE, "kt-kernel patch"),
        # KT_ZEROCOPY_WEIGHTS shipped before KT_ZEROCOPY_SCOPE did; such a build
        # takes the FULL zero-copy path, which is not what the switch asks for
        (_NOISE + b"KT_ZEROCOPY_WEIGHTS\x00" + _NOISE, "KT_ZEROCOPY_SCOPE"),
        # no extension to read: refuse rather than assume
        (None, "cannot be probed"),
    ],
    ids=["no_switch", "only_one_name", "unprobeable"],
)
def test_a_build_that_cannot_support_zerocopy_is_refused(monkeypatch, tmp_path, blob, match):
    _with_kt_ext(monkeypatch, tmp_path, blob)
    with pytest.raises(ValueError, match=match):
        _zc_cfg()._refuse_without_zerocopy_support()


def test_no_kt_at_all_is_refused(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "kt_kernel", None)  # import yields None -> attribute fails
    with pytest.raises(ValueError):
        _zc_cfg()._refuse_without_zerocopy_support()


def test_the_check_is_skipped_when_the_switch_is_off(monkeypatch, tmp_path):
    """Off must cost nothing -- otherwise every unrelated run needs a new kt."""
    _with_kt_ext(monkeypatch, tmp_path, _NOISE)  # a build WITHOUT the names
    _zc_cfg(zerocopy_w13=False)._refuse_without_zerocopy_support()  # must not raise


def test_the_probe_reads_the_build_not_the_environment(monkeypatch, tmp_path):
    """The export is precisely what is silently ignored, so it must not satisfy
    the check.
    """
    monkeypatch.setenv("KT_ZEROCOPY_WEIGHTS", "1")
    monkeypatch.setenv("KT_ZEROCOPY_SCOPE", "w13")
    _with_kt_ext(monkeypatch, tmp_path, _NOISE + b"KT_FULLSET_LOAD\x00" + _NOISE)
    with pytest.raises(ValueError, match="kt-kernel patch"):
        _zc_cfg()._refuse_without_zerocopy_support()


# zerocopy_node_fraction: an illegal split must RAISE, never fall back. The
# field exists so an A/B pins the same split on both arms; a silent fallback to
# 50/50 would let two arms place their pages differently
def _frac_cfg(**kw):
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    base = dict(
        enabled=True, cpuinfer_threads=8, num_resident_experts=100, overlap_side_stream=False, threadpool_count=2
    )
    base.update(kw)
    return KTOffloadConfig(**base)


def test_a_pinned_split_is_accepted():
    cfg = _frac_cfg(zerocopy_node_fraction=[0.41, 0.59])
    assert cfg.zerocopy_node_fraction == [0.41, 0.59]


def test_none_is_accepted_and_means_derive_at_load_time():
    assert _frac_cfg().zerocopy_node_fraction is None


def test_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one per TP part"):
        _frac_cfg(zerocopy_node_fraction=[1.0])


def test_full_locality_is_accepted(caplog):
    """``fraction[p]`` is part p's local share, so ``[1.0, 1.0]`` sums to 2 and is
    the intended default: every TP part keeps all of its own w13 on its own node.
    """
    cfg = _frac_cfg(zerocopy_node_fraction=[1.0, 1.0])
    assert cfg.zerocopy_node_fraction == [1.0, 1.0]


def test_an_uneven_but_independent_split_is_accepted():
    """Cutting ONE part back (its node cannot hold its share) must not force
    the other part to give up locality it can afford."""
    assert _frac_cfg(zerocopy_node_fraction=[0.6, 1.0]).zerocopy_node_fraction == [0.6, 1.0]


def test_mean_local_share_is_not_capped(monkeypatch):
    """A ``sum(f) == 1`` rule would pin the mean local share at 1/tp; a
    configuration whose mean is above that must stay legal.
    """
    f = _frac_cfg(zerocopy_node_fraction=[1.0, 1.0]).zerocopy_node_fraction
    assert sum(f) / len(f) == 1.0 > 1.0 / 2


def test_a_negative_fraction_is_refused():
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        _frac_cfg(zerocopy_node_fraction=[-0.1, 1.1])


def test_negative_reserve_is_refused():
    with pytest.raises(ValueError, match="zerocopy_reserve_gib"):
        _frac_cfg(zerocopy_reserve_gib=-1.0)


def test_stream_resident_from_hbm_needs_stream_prefill_and_the_bank():
    """Each precondition is a different silent wrong answer if it is missing.

    The switch skips streaming the experts already in HBM and fills those slots
    of the chunk buffer by D2D instead -- 26% of the streamed bytes, with the
    chunk partition and therefore the output bit-identical. Without the bank the
    source cannot gather a subset of a chunk at all, so the skipped slots would
    keep the previous chunk's bytes under live routes; without stream_prefill
    there is no chunk at all.
    """
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    base = dict(
        enabled=True,
        cpuinfer_threads=8,
        num_resident_experts=100,
        overlap_side_stream=False,
        stream_prefill=True,
        bank_dir="/dev/shm/dsv41_nz_bank",
        stream_resident_from_hbm=True,
    )
    KTOffloadConfig(**base)  # the combination that is allowed

    with pytest.raises(ValueError, match="needs stream_prefill"):
        KTOffloadConfig(**{**base, "stream_prefill": False})
    with pytest.raises(ValueError, match="needs bank_dir"):
        KTOffloadConfig(**{**base, "bank_dir": ""})
    # num_resident_experts=0 is refused before this switch is even looked at
    # (resident_strategy='prefix' needs > 0). Asserted here so the combination
    # is known to be refused by SOMETHING -- the backstop in
    # stream_resident_from_hbm's own validation is unreachable from this
    # strategy, and an unreachable check must not be mistaken for the one doing
    # the work.
    with pytest.raises(ValueError, match="needs num_resident_experts > 0"):
        KTOffloadConfig(**{**base, "num_resident_experts": 0})


def test_stream_resident_from_hbm_stays_off_by_default_because_it_has_preconditions():
    """🔴 Measured as a win (-11.6% steady-state TTFT, 7/7 token-identical) and
    STILL default off.

    It needs bank_dir and stream_prefill, so defaulting it on makes every config
    that does not have them fail validation -- 20 of the tests in this file did,
    immediately, when that was tried on 2026-11-15. A switch with preconditions
    is opted into by the deployment config (measure/s3_g.json), never by the
    dataclass. Asserted here so the next person who reads the measurement and
    reaches for the default finds the reason first.
    """
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    base = dict(enabled=True, cpuinfer_threads=8, num_resident_experts=100, overlap_side_stream=False)
    assert KTOffloadConfig(**base).stream_resident_from_hbm is False
    with pytest.raises(ValueError, match="needs bank_dir"):
        KTOffloadConfig(**base, stream_prefill=True, stream_resident_from_hbm=True)


def test_kt_reads_bank_refuses_scope_w13():
    """🔴 The combination that would compute wrong numbers rather than fail.

    With zerocopy_scope='w13' kt stages w2 itself, and fp4-moe.hpp does it with
    an ND column-slice walk over down_projs[0][expert] -- index 0 rather than
    the TP part, and ND strides. The bank's w2 is FRACTAL_NZ with per-part
    pointers, so that walk reads the wrong bytes and reports nothing. The
    combination had simply never been run until someone went looking for an arm
    that isolates w2's share of the TPOT regression.
    """
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    base = dict(
        enabled=True,
        cpuinfer_threads=8,
        num_resident_experts=100,
        overlap_side_stream=False,
        stream_prefill=True,
        bank_dir="/dev/shm/dsv41_nz_bank",
        zerocopy_w13=True,
        kt_reads_bank=True,
    )
    KTOffloadConfig(**base, zerocopy_scope="all")
    with pytest.raises(ValueError, match="needs zerocopy_scope='all'"):
        KTOffloadConfig(**base, zerocopy_scope="w13")
