# SPDX-License-Identifier: Apache-2.0
"""The runtime kernel-knob watcher used for in-process A/B measurement.

What these tests are for: the watcher exists so both arms of a kt prefetch A/B
run in ONE server process. That makes it measurement-critical in two ways, and
each is tested here rather than discovered during a campaign:

* it must APPLY what the file says and report back what kt actually took (a
  clamped or ignored value would leave the log claiming an arm that is not
  running);
* it must never raise into the server -- a malformed file has to be a logged
  no-op, not a dead worker in the middle of a card session.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from vllm_ascend.kt_offload import kt_knob_ab as kb


class _Ext:
    """Stands in for kt_kernel_ext: records calls, clamps like the C++ does.

    ``missing`` sets a setter to None, which is exactly what ``apply`` checks
    for -- a kt build predating the knob.
    """

    def __init__(self, *, clamp: int | None = None, missing: tuple[str, ...] = ()):
        self.calls: list[tuple[str, object]] = []
        self._groups = 8
        self._legacy = False
        self._clamp = clamp
        self._short = -1
        self._hint = 0
        for name in missing:
            setattr(self, name, None)

    def set_nz_prefetch_groups(self, v):
        self.calls.append(("set_nz_prefetch_groups", v))
        self._groups = min(v, self._clamp) if self._clamp is not None else v

    def set_nz_prefetch_groups_short(self, v):
        self.calls.append(("set_nz_prefetch_groups_short", v))
        self._short = v

    def set_nz_prefetch_hint(self, v):
        self.calls.append(("set_nz_prefetch_hint", v))
        self._hint = v

    def nz_prefetch_groups_short(self):
        return self._short

    def nz_prefetch_hint(self):
        return self._hint

    def set_nz_prefetch_legacy_stride(self, v):
        self.calls.append(("set_nz_prefetch_legacy_stride", v))
        self._legacy = bool(v)

    def nz_prefetch_groups(self):
        return self._groups

    def nz_prefetch_legacy_stride(self):
        return self._legacy


def _wait_for(pred, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached within %.1f s" % timeout)


def test_parse_accepts_one_line_of_every_knob():
    assert kb.parse("nz_prefetch_groups=16 nz_prefetch_groups_short=-1 nz_prefetch_hint=2 "
                    "nz_prefetch_legacy_stride=1") == {
        "nz_prefetch_groups": 16,
        "nz_prefetch_groups_short": -1,   # -1 = follow the long-row distance
        "nz_prefetch_hint": 2,            # NTA
        "nz_prefetch_legacy_stride": True,
    }


def test_parse_ignores_comments_and_blank_input():
    assert kb.parse("") == {}
    assert kb.parse("# nz_prefetch_groups=16\n") == {}
    assert kb.parse("nz_prefetch_groups=0 # the no-prefetch arm") == {"nz_prefetch_groups": 0}


@pytest.mark.parametrize("text", [
    "nz_prefetch_groups",                 # not key=value
    "prefetch=8",                         # unknown knob: a typo must not be a silent no-op
    "nz_prefetch_groups=eight",
    "nz_prefetch_legacy_stride=yes",      # only 0/1, so "on" cannot mean "off"
])
def test_parse_refuses_what_it_cannot_read(text):
    with pytest.raises(ValueError):
        kb.parse(text)


def test_apply_calls_the_setters_and_reports_what_kt_took():
    ext = _Ext()
    got = kb.apply({"nz_prefetch_groups": 24, "nz_prefetch_legacy_stride": True}, ext=ext)
    assert ("set_nz_prefetch_groups", 24) in ext.calls
    assert ("set_nz_prefetch_legacy_stride", True) in ext.calls
    assert got == {"nz_prefetch_groups": 24, "nz_prefetch_legacy_stride": True}


def test_apply_reports_the_CLAMPED_value_not_the_requested_one():
    # The C++ clamps the distance; if the watcher echoed the request instead of
    # reading back, a sweep arm could be logged as 9000 while 4096 ran.
    ext = _Ext(clamp=4096)
    got = kb.apply({"nz_prefetch_groups": 99999}, ext=ext)
    assert got["nz_prefetch_groups"] == 4096


def test_apply_refuses_a_kt_build_without_the_knob():
    ext = _Ext(missing=("set_nz_prefetch_groups",))
    with pytest.raises(RuntimeError, match="runtime knob is not in the LIVE binary"):
        kb.apply({"nz_prefetch_groups": 8}, ext=ext)


def test_watcher_is_off_unless_the_env_var_names_a_file(monkeypatch):
    monkeypatch.delenv(kb.ENV_KNOB_FILE, raising=False)
    assert kb.knob_file() == ""
    assert kb.start_if_configured() is None
    monkeypatch.setenv(kb.ENV_KNOB_FILE, "   ")
    assert kb.start_if_configured() is None


def test_watcher_applies_a_change_and_survives_a_malformed_file(tmp_path, monkeypatch):
    ext = _Ext()
    real_apply = kb.apply
    monkeypatch.setattr(kb, "apply", lambda values: real_apply(values, ext=ext))

    path = tmp_path / "knob"
    stop = threading.Event()
    t = threading.Thread(target=kb._watch, args=(str(path), 0.01, stop), daemon=True)
    t.start()
    try:
        # A file that does not exist yet is not an error: the harness writes it
        # after the server is up.
        time.sleep(0.05)
        assert t.is_alive() and not ext.calls
        path.write_text("nz_prefetch_groups=16")
        _wait_for(lambda: ("set_nz_prefetch_groups", 16) in ext.calls)
        # Garbage must be a logged no-op, and the thread must still be alive for
        # the next arm: a measurement tool may not kill what it measures.
        path.write_text("nz_prefetch_groups=bogus")
        time.sleep(0.1)
        assert t.is_alive()
        path.write_text("nz_prefetch_legacy_stride=1")
        _wait_for(lambda: ("set_nz_prefetch_legacy_stride", True) in ext.calls)
    finally:
        stop.set()
        t.join(timeout=2)
        assert not t.is_alive()


def test_env_var_is_declared_in_envs_py():
    # AGENTS.md: every env var belongs in the central dict, with documentation.
    from vllm_ascend import envs

    assert kb.ENV_KNOB_FILE in envs.env_variables
    assert envs.env_variables[kb.ENV_KNOB_FILE]() == os.environ.get(kb.ENV_KNOB_FILE, "")
