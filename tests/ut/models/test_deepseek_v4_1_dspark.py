# SPDX-License-Identifier: Apache-2.0
"""The V4.1 DSpark drafter: the three rebinds, the routing, and the weights.

Three separate things can make "DSpark is enabled" false while every log line
still looks healthy, and there is one group of tests here for each:

1. **The rebinds became no-ops.** Upstream renames or inlines one of the three
   hardware entry points and the drafter quietly calls CUDA's version. That
   surfaces in the first draft forward, after 7.39 GiB of draft weights have
   loaded. ``test_upstream_dspark_contract`` pins what is being replaced.
2. **The routing picked the wrong network.** V4.0's drafter is a different
   model that happens to answer to a similar name; pointing V4.1 at it is a
   wrong-weights failure. ``test_v41_routes_*`` pins which class gets built.
3. **The weights are not where the loader looks.** The draft weights ship
   inside the *target* checkpoint under ``mtp.*``, so there is no separate
   download whose absence would be obvious. ``test_checkpoint_*`` runs the
   real mapping function over the real checkpoint's real tensor names.

None of this needs an NPU, a card, or a loaded model.
"""

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

from vllm_ascend.models.deepseek_v4_1 import dspark_ops, platform_patches

# Importing the package applies the patches, which is what the production path
# relies on; done here so the assertions below describe that path.
import vllm_ascend.models.deepseek_v4_1  # noqa: F401  isort:skip

MODEL_DIR = Path(os.environ.get("DSV41_MODEL_DIR", "/workspace/models/DeepSeek-V4.1-Flash"))
needs_checkpoint = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors.index.json").exists(),
    reason=f"no checkpoint index under {MODEL_DIR} (set DSV41_MODEL_DIR to point at one)",
)


@pytest.fixture
def upstream_dspark():
    """Upstream's drafter module, with all three names restored afterwards."""
    import vllm.models.deepseek_v4_1.nvidia.dspark as module

    saved = {name: getattr(module, name, None) for name, _, _ in platform_patches.DSPARK_REBINDS}
    yield module
    for name, value in saved.items():
        if value is not None:
            setattr(module, name, value)


# --- 1. the rebinds ---------------------------------------------------------


def test_upstream_dspark_contract(upstream_dspark):
    """Pin what the three rebinds are replacing.

    Each must still be a module-level name *and* still be reached through the
    module global at the call site. An inlined call, a method, or a local
    alias would make the rebind a silent no-op -- the drafter would build,
    load and then fail in its first forward.
    """
    for name, _, _ in platform_patches.DSPARK_REBINDS:
        assert hasattr(upstream_dspark, name), f"{name} is gone from upstream"

    model_forward = inspect.getsource(upstream_dspark.DSparkDeepseekV4Model.forward)
    assert "mhc_post_tilelang(" in model_forward
    assert "hc_collapse_triton(" in model_forward

    precompute = inspect.getsource(upstream_dspark.DSparkDeepseekV4Model.precompute_and_store_context_kv)
    assert "_insert_context_kv(" in precompute


def test_package_import_puts_every_rebind_in_force():
    """The production path: import the package, get all three."""
    assert platform_patches.dspark_rebinds_in_force() == []


def test_rebind_is_idempotent(upstream_dspark):
    platform_patches.rebind_dspark_functions()
    assert platform_patches.rebind_dspark_functions() == []
    assert platform_patches.dspark_rebinds_in_force() == []


def test_rebind_raises_when_a_name_is_gone(upstream_dspark, monkeypatch):
    monkeypatch.delattr(upstream_dspark, "hc_collapse_triton")
    with pytest.raises(AttributeError, match="no longer defines"):
        platform_patches.rebind_dspark_functions()


def test_apply_all_reports_the_dspark_rebinds():
    applied = platform_patches.apply_all()
    assert isinstance(applied, list)
    # apply_all is idempotent, so a second call reports nothing newly rebound;
    # what must hold either way is that the names are in force.
    assert platform_patches.dspark_rebinds_in_force() == []


def test_drafter_refuses_to_build_without_the_rebinds(upstream_dspark, monkeypatch):
    """The assert in the constructor, which is the last line of defence."""
    from vllm_ascend.models.deepseek_v4_1.dspark import AscendDSparkDeepseekV41ForCausalLM

    monkeypatch.setattr(upstream_dspark, "hc_collapse_triton", lambda *a: None)
    with pytest.raises(RuntimeError, match="rebinds are not in force"):
        AscendDSparkDeepseekV41ForCausalLM(vllm_config=None)


# --- 2. the collapse, numerically -------------------------------------------


def test_hc_collapse_matches_the_reference_sum():
    import torch

    torch.manual_seed(0)
    tokens, hc, hidden = 7, 4, 32
    x = torch.randn(tokens, hc, hidden, dtype=torch.bfloat16)
    pre_mix = torch.randn(tokens, hc, dtype=torch.float32)

    got = dspark_ops.hc_collapse_ascend(x, pre_mix)
    want = (pre_mix.unsqueeze(-1) * x.float()).sum(dim=1).to(x.dtype)

    assert got.shape == (tokens, hidden)
    assert got.dtype == x.dtype
    torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=0)


def test_hc_collapse_refuses_swapped_arguments():
    """The trap this wrapper exists for.

    Upstream takes ``(x, pre_mix)``; the function it delegates to takes
    ``(pre_mix, residual)``. Both are float tensors whose leading axis is the
    token count, so a swap broadcasts instead of raising.
    """
    import torch

    x = torch.randn(5, 4, 16, dtype=torch.bfloat16)
    pre_mix = torch.randn(5, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="expected \\(tokens, hc, hidden\\)"):
        dspark_ops.hc_collapse_ascend(pre_mix, x)


def test_hc_collapse_refuses_mismatched_token_counts():
    import torch

    x = torch.randn(5, 4, 16, dtype=torch.bfloat16)
    pre_mix = torch.randn(6, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="first two axes must agree"):
        dspark_ops.hc_collapse_ascend(x, pre_mix)


# --- 3. the routing ---------------------------------------------------------


def _spec_config(target_model_type, draft_arch, module_name):
    """A duck-typed SpeculativeConfig, enough for the two helpers under test."""
    from vllm_ascend.patch.platform import patch_speculative_config as psc

    draft_hf = SimpleNamespace(model_type="deepseek_v4", n_routed_experts=384, dspark_n_routed_experts=128)
    draft = SimpleNamespace(
        hf_config=draft_hf,
        _architecture=draft_arch,
        registry=SimpleNamespace(models={draft_arch: SimpleNamespace(module_name=module_name, class_name="X")}),
    )
    return psc, SimpleNamespace(
        target_model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=target_model_type)),
        draft_model_config=draft,
    )


def test_v41_pair_is_detected_by_the_target_model_type():
    psc, cfg = _spec_config("deepseek_v41", "X", "m")
    assert psc._is_v41_dspark_pair(cfg) is True


def test_v40_pair_is_not_detected_as_v41():
    psc, cfg = _spec_config("deepseek_v4", "X", "m")
    assert psc._is_v41_dspark_pair(cfg) is False


def test_the_ported_module_is_allow_listed():
    from vllm_ascend.patch.platform import patch_speculative_config as psc

    assert frozenset({"vllm_ascend.models.deepseek_v4_1.dspark"}) == psc.V41_DSPARK_DRAFTER_MODULES


def test_v41_resolving_to_the_ported_drafter_is_accepted():
    psc, cfg = _spec_config("deepseek_v41", "DSparkDeepseekV41DraftModel", "vllm_ascend.models.deepseek_v4_1.dspark")
    assert psc.V41_DSPARK_ARCHITECTURE == "DSparkDeepseekV41DraftModel"
    psc._refuse_v40_drafter_for_v41_target(cfg)  # must not raise


def test_v41_resolving_to_the_v40_drafter_is_still_refused():
    """The guard that was the whole reason this port was needed."""
    psc, cfg = _spec_config("deepseek_v41", "DSparkDraftModel", "vllm_ascend.models.deepseek_v4.dspark")
    with pytest.raises(NotImplementedError) as excinfo:
        psc._refuse_v40_drafter_for_v41_target(cfg)
    message = str(excinfo.value)
    # The message must name the right architecture to move to, and must not
    # still claim nothing is ported.
    assert "DSparkDeepseekV41DraftModel" in message
    assert "is ported to vllm-ascend yet" not in message
    assert "Do not simply allow-list" in message


def test_normalize_writes_the_architecture_it_is_given():
    from dataclasses import dataclass

    from vllm_ascend.patch.platform import patch_speculative_config as psc

    @dataclass
    class ArchConfig:
        architectures: list
        model_type: str
        is_mm_prefix_lm: bool

    recorded = {}

    class HF:
        model_type = "deepseek_v41_text"
        dspark_target_layer_ids = [37, 38, 39]

        def update(self, d):
            recorded.update(d)

    draft = SimpleNamespace(
        hf_config=HF(),
        model_arch_config=ArchConfig(["Whatever"], "x", True),
        architectures=["Whatever"],
        registry=SimpleNamespace(inspect_model_cls=lambda *a: ("info", "arch")),
    )
    psc._normalize_deepseek_v4_dspark_draft(draft, "DSparkDeepseekV41DraftModel")

    assert recorded["architectures"] == ["DSparkDeepseekV41DraftModel"]
    assert draft.model_arch_config.architectures == ["DSparkDeepseekV41DraftModel"]
    # model_type stays v4: it is what keeps is_deepseek_v4_family true and
    # stops the multimodal conversion re-running.
    assert draft.model_arch_config.model_type == "deepseek_v4"


def test_registration_and_allow_list_name_the_same_module():
    """The registry entry and the allow-list must not be able to drift apart.

    ``_refuse_v40_drafter_for_v41_target`` compares the registry entry's
    ``module_name`` against ``V41_DSPARK_DRAFTER_MODULES``. If the module is
    renamed and only one of the two is updated, a correct V4.1 setup is
    refused (or, worse, a wrong one is let through). Read the same two things
    the refusal reads.
    """
    from vllm import ModelRegistry

    from vllm_ascend.models import register_model
    from vllm_ascend.patch.platform import patch_speculative_config as psc

    register_model()
    entry = ModelRegistry.models[psc.V41_DSPARK_ARCHITECTURE]
    assert entry.module_name == "vllm_ascend.models.deepseek_v4_1.dspark"
    assert entry.class_name == "AscendDSparkDeepseekV41ForCausalLM"
    assert entry.module_name in psc.V41_DSPARK_DRAFTER_MODULES


def test_the_drafter_subclasses_upstreams():
    """Not a fork: the 558 lines of drafter logic must still be upstream's."""
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM

    from vllm_ascend.models.deepseek_v4_1.dspark import AscendDSparkDeepseekV41ForCausalLM

    assert issubclass(AscendDSparkDeepseekV41ForCausalLM, DSparkDeepseekV4ForCausalLM)
    # Only the constructor is overridden; everything else is inherited.
    overridden = {
        name
        for name, value in vars(AscendDSparkDeepseekV41ForCausalLM).items()
        if callable(value) and not name.startswith("__doc")
    }
    assert overridden == {"__init__"}, overridden


# --- 4. the weights, against the real checkpoint ----------------------------

_STAGE_RE = re.compile(r"mtp\.(\d+)\.")


def _draft_tensor_names():
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    return sorted(k for k in index["weight_map"] if k.startswith("mtp."))


def _remap(name, *, confidence_head=True):
    """Call the real ``_remap_dspark_name`` with a duck-typed ``self``.

    The method reads only ``self.model.confidence_head``, so this exercises
    the production mapping code rather than a copy of its rules.
    """
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM

    fake = SimpleNamespace(model=SimpleNamespace(confidence_head=confidence_head))
    return DSparkDeepseekV4ForCausalLM._remap_dspark_name(fake, name)


@needs_checkpoint
def test_checkpoint_carries_the_draft_weights():
    """The draft weights are in the target checkpoint, not a separate one."""
    names = _draft_tensor_names()
    stages = {int(_STAGE_RE.match(n).group(1)) for n in names}
    assert stages == {0, 1, 2}, f"expected 3 mtp stages, got {sorted(stages)}"
    assert len(names) > 2000, f"only {len(names)} mtp tensors"

    # The two heads that make this DSpark rather than plain MTP.
    assert "mtp.2.markov_head.embed.weight" in names
    assert "mtp.2.markov_head.head.weight" in names
    assert "mtp.2.confidence_head.proj.weight" in names


@needs_checkpoint
def test_every_draft_tensor_maps_to_a_parameter_path():
    """No ``mtp.*`` tensor silently falls through the mapping.

    A name that maps to None is skipped by ``load_weights`` without a word,
    so an unmapped tensor is a weight that never loads.
    """
    unmapped = [n for n in _draft_tensor_names() if _remap(n) is None]
    assert unmapped == [], f"{len(unmapped)} unmapped, e.g. {unmapped[:5]}"


@needs_checkpoint
def test_markov_factors_are_renamed_to_the_parameter_names():
    """The checkpoint says ``embed``/``head``; the module registers w1/w2."""
    assert _remap("mtp.2.markov_head.embed.weight") == "model.markov_head.markov_w1.weight"
    assert _remap("mtp.2.markov_head.head.weight") == "model.markov_head.markov_w2.weight"


@needs_checkpoint
def test_confidence_head_is_dropped_when_the_head_is_disabled():
    assert _remap("mtp.2.confidence_head.proj.weight", confidence_head=None) is None


@needs_checkpoint
def test_draft_decoder_weights_land_on_the_three_draft_layers():
    """Per-stage tensors go to ``model.layers.{0,1,2}``, heads to model level."""
    seen_layers, seen_heads = set(), set()
    for name in _draft_tensor_names():
        mapped = _remap(name)
        if mapped.startswith("model.layers."):
            seen_layers.add(mapped.split(".")[2])
        else:
            seen_heads.add(mapped.removeprefix("model.").split(".")[0])
    assert seen_layers == {"0", "1", "2"}, seen_layers
    assert seen_heads <= {"main_proj", "main_norm", "norm", "markov_head", "confidence_head"}, seen_heads


@needs_checkpoint
def test_target_loader_still_drops_the_draft_weights():
    """Enabling DSpark must not make the target load ``mtp.*`` as well.

    The target's mapper drops them (``"mtp.": None``). If that stopped being
    true, the 7.39 GiB would be loaded twice -- once into the target, where
    there is no parameter for it.
    """
    from vllm.models.deepseek_v4_1.nvidia.model import _make_deepseek_v4_weights_mapper

    mapper = _make_deepseek_v4_weights_mapper("fp4")
    assert mapper.orig_to_new_substr.get("mtp.", "missing") is None


def _context_attention(cache, block_size=8):
    import torch

    return SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=block_size, prefix="draft-test"),
        n_local_heads=1,
        head_dim=512,
        rope_head_dim=4,
        eps=1e-20,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.tensor([[0.0, 0.0, 1.0, 1.0]]).repeat(8, 1)),
    )


def test_context_insert_rotates_tail_and_leaves_invalid_slots_untouched():
    import torch

    cache = torch.full((3, 8, 512), 0.25, dtype=torch.bfloat16)
    kv = torch.ones(3, 512, dtype=torch.bfloat16)
    kv[:, -4:] = torch.tensor([2, 3, 4, 5], dtype=torch.bfloat16)
    expected = cache.clone()
    rotated = kv[0].clone()
    rotated[-4:] = torch.tensor([-3, 2, -5, 4], dtype=torch.bfloat16)
    expected[1, 2] = rotated
    expected[2, 3] = rotated
    dspark_ops.insert_context_kv_ascend(
        _context_attention(cache),
        kv,
        torch.tensor([0, 1, 2]),
        torch.tensor([[1, 2], [-1, -1], [2, 3]], dtype=torch.int32),
    )
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


def test_context_insert_refuses_an_unimplemented_cache_dtype():
    import torch

    cache = torch.zeros(3, 8, 512, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="bf16 SWA cache"):
        dspark_ops.insert_context_kv_ascend(_context_attention(cache), None, None, None)


def test_context_insert_refuses_a_mismatched_page_size():
    import torch

    cache = torch.zeros(3, 8, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="layer declares"):
        dspark_ops.insert_context_kv_ascend(_context_attention(cache, block_size=16), None, None, None)
