# SPDX-License-Identifier: Apache-2.0
"""Regression test for the provider-identity contract.

A patch once wrapped the streaming provider INSIDE
``install_stream_provider``. The wrapper went onto every layer, but the caller
kept the unwrapped object, so:

  * ``wiring._assert_installed`` compares with ``is not`` and refused start-up
    ("layer 0 did not take the streaming provider");
  * the fold was built against the inner object while the layers held the
    wrapper, which the assertion's own message warns about ("the fold reads its
    chunks out of the object that served them");
  * ``StreamingWiring`` carried the inner object too.

So the contract is: ``install_stream_provider`` installs EXACTLY the object it
was handed, and any wrapping happens in ``setup_streaming`` before anything is
installed. The ten stream_layer unit tests could not catch this because they
drive stubs and never go through the installer.
"""

import types

from vllm_ascend.kt_offload.layer import install_stream_provider


class _FakeProvider:
    def __init__(self):
        self.chunk_experts = 32
        self.shapes = {}

    def slot_for(self, chunk_index, expert_ids):
        raise AssertionError("not called by installation")

    def set_layer(self, *a, **k):
        pass


def _fake_layer(layer_idx=0):
    cfg = types.SimpleNamespace(stream_prefill=True)
    rt = types.SimpleNamespace(
        cfg=cfg,
        placement=types.SimpleNamespace(num_experts=384),
        _stream_provider=None,
        _stream_slot_provider=None,
        # Distinct, because ``offload_layers_to_install`` keys the layers by
        # ``layer_idx``: two fakes sharing 0 collapse into ONE entry, so the
        # first one is never installed and this file used to assert that.
        layer_idx=layer_idx,
    )
    return rt


def test_installer_does_not_replace_the_provider():
    """Every layer must hold the very object passed in -- identity, not equality.

    Any wrapping belongs to ``setup_streaming``, which owns the single-provider
    contract; the installer itself must never substitute an object.
    """
    provider = _FakeProvider()
    layers = [_fake_layer(0), _fake_layer(1)]
    install_stream_provider(provider, layers=layers)
    for rt in layers:
        assert rt._stream_provider is provider, (
            "the installer substituted a different provider object; "
            "wiring._assert_installed compares identity and would refuse start-up"
        )
        assert rt._stream_slot_provider == provider.slot_for
