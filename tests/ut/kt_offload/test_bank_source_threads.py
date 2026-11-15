# SPDX-License-Identifier: Apache-2.0
"""How many threads gather streamed chunks out of the bank -- and how it is set.

Why this file exists: ``wiring.setup_streaming`` built ``BankSource`` without a
``copy_threads``, so when the bank replaced the checkpoint as the streaming
source the read leg silently went from ``weight_source.DEFAULT_COPY_THREADS``
(8) to ``bank_source.MIN_LOADER_THREADS`` (4) -- on the leg the server's own
staging report puts at ~94% of the staging loop's wall clock. Nothing failed,
nothing said anything.

So the count is now settable per server start (an A/B over it cannot live in one
process: the source is built once at start-up), and these tests hold the two
properties that matter for a measurement knob: it takes effect, and a value it
cannot read is an error rather than a silent default.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_ascend.kt_offload import bank_build, bank_source

#: Small enough to build in a test, and still satisfies every precondition
#: ``BankLayout.check()`` enforces (see test_bank_build.TINY).
TINY = bank_build.BankLayout(num_experts=3, hidden=256, intermediate=64, group_size=32, tp_count=2)


class FakeSource:
    """Deterministic ND bytes at the right sizes; stands in for WeightSource."""

    def __init__(self, layout):
        self.layout = layout

    def _nbytes(self, proj, kind):
        lay = self.layout
        if kind == "weight":
            return lay.w2_n * lay.w2_kb if proj == "w2" else lay.intermediate * lay.w13_kb
        return lay.section_record("w2_scale") if proj == "w2" else lay.section_record("w13_scale") // 2

    def locate(self, layer, expert, proj, kind="weight"):
        return (layer, expert, proj, kind)

    def read(self, span):
        layer, expert, proj, kind = span
        n = self._nbytes(proj, kind)
        idx = np.arange(n, dtype=np.uint64)
        vals = (idx + np.uint64(layer * 31 + expert * 7 + len(proj) + len(kind))) & np.uint64(0xFF)
        return torch.from_numpy(vals.astype(np.uint8))


@pytest.fixture
def tiny_bank(tmp_path):
    bank_build.build(FakeSource(TINY), str(tmp_path), layers=[0], layout=TINY, drop_cache=False)
    return str(tmp_path)


def test_default_is_the_min_loader_thread_count(tiny_bank, monkeypatch):
    # Documents the value that was in force through stage 3, so a change to it
    # is a visible change to this test rather than a silent change to TTFT.
    monkeypatch.delenv("VLLM_ASCEND_DSV41_KT_LOADER_THREADS", raising=False)
    src = bank_source.BankSource(tiny_bank, num_experts=TINY.num_experts)
    try:
        assert src.copy_threads == bank_build.MIN_LOADER_THREADS == 4
    finally:
        src.close()


def test_the_caller_can_still_pass_its_own(tiny_bank, monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSV41_KT_LOADER_THREADS", raising=False)
    src = bank_source.BankSource(tiny_bank, num_experts=TINY.num_experts, copy_threads=1)
    try:
        assert src.copy_threads == 1
    finally:
        src.close()


def test_env_overrides_the_caller(tiny_bank, monkeypatch):
    # 1 rather than 8 so the test does not depend on the host having 8 distinct
    # L3 domains: above 1, BankSource also asserts its cpu placement.
    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_LOADER_THREADS", "1")
    src = bank_source.BankSource(tiny_bank, num_experts=TINY.num_experts, copy_threads=4)
    try:
        assert src.copy_threads == 1
    finally:
        src.close()


@pytest.mark.parametrize("value", ["abc", "", "0", "-2", "4.5"])
def test_an_unreadable_count_is_refused_or_ignored_explicitly(tiny_bank, monkeypatch, value):
    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_LOADER_THREADS", value)
    if value.strip() == "":
        # Empty means "not set", which is the documented default -- not an error.
        src = bank_source.BankSource(tiny_bank, num_experts=TINY.num_experts, copy_threads=1)
        try:
            assert src.copy_threads == 1
        finally:
            src.close()
        return
    with pytest.raises(bank_source.BankSourceError):
        bank_source.BankSource(tiny_bank, num_experts=TINY.num_experts, copy_threads=1)


def test_env_var_is_declared_in_envs_py():
    from vllm_ascend import envs

    assert "VLLM_ASCEND_DSV41_KT_LOADER_THREADS" in envs.env_variables
