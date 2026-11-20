# SPDX-License-Identifier: Apache-2.0
"""A built AVX2 kernel must not pass AMX-only zero-copy/full-set checks."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_ascend.kt_offload.config import KTOffloadConfig


def install_kt(monkeypatch, *, fullset):
    backend = type("AVX2MXFP4_MOE", (), {})
    amx = SimpleNamespace(
        _select_mxfp4_backend=lambda: backend,
        AVX2MXFP4_MOE=backend,
        _fullset_load_requested=lambda: True,
        mxfp4_backend_capabilities=lambda: {"fullset_load": fullset},
    )
    pkg, utils = ModuleType("kt_kernel"), ModuleType("kt_kernel.utils")
    utils.amx = amx
    pkg.utils = utils
    monkeypatch.setitem(sys.modules, "kt_kernel", pkg)
    monkeypatch.setitem(sys.modules, "kt_kernel.utils", utils)
    monkeypatch.setitem(sys.modules, "kt_kernel.utils.amx", amx)
    return amx


def test_avx2_cannot_claim_zero_copy_from_strings_in_another_backend(monkeypatch):
    install_kt(monkeypatch, fullset=True)
    with pytest.raises(ValueError, match="AVX2MXFP4_MOE.*FP32 scales"):
        KTOffloadConfig(zerocopy_w13=True)._refuse_without_zerocopy_support()


@pytest.mark.parametrize("has_probe", [False, True])
def test_old_native_avx2_load_is_refused_even_when_python_knows_fullset(monkeypatch, has_probe):
    amx = install_kt(monkeypatch, fullset=False)
    if not has_probe:
        del amx.mxfp4_backend_capabilities
    with pytest.raises(ValueError, match="native full-set-load support"):
        KTOffloadConfig(dynamic_resident=True)._refuse_without_full_set_load()


def test_patched_avx2_copy_path_keeps_dynamic_residency(monkeypatch):
    install_kt(monkeypatch, fullset=True)
    cfg = KTOffloadConfig(dynamic_resident=True, stream_prefill=True, zerocopy_w13=False)
    cfg._refuse_without_full_set_load()
    cfg._refuse_without_zerocopy_support()
