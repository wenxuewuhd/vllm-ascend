from dataclasses import replace

from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

from vllm_ascend.utils import is_deepseek_v4_family

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override


# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _orig_hf_config_override(hf_config)
    architectures = hf_config.architectures or ()
    if hf_config.model_type == "qwen3" and "DSparkDraftModel" in architectures:
        dflash_config = hf_config.dflash_config
        hf_config.update(
            {
                "architectures": ["Qwen3DSparkModel"],
                "mask_token_id": dflash_config["mask_token_id"],
                "target_layer_ids": dflash_config["target_layer_ids"],
            }
        )
    return hf_config


def _normalize_deepseek_v4_dspark_draft(draft_model_config, architecture: str = "DSparkDraftModel") -> None:
    """Restore the DSpark draft architecture after VL config conversion.

    DeepSeek-V4-Vision uses the same checkpoint for the target and DSpark
    drafter.  vLLM first rewrites that checkpoint to ``DSparkDraftModel``, but
    rebuilding ``model_arch_config`` with multimodal detection can restore the
    top-level ``*ForConditionalGeneration`` architecture.  The drafter would
    then instantiate a second full VL target and register duplicate attention
    layer names.  Update both config representations without re-running the
    multimodal architecture conversion.
    """
    hf_config = getattr(draft_model_config, "hf_config", None)
    if (
        hf_config is None
        or not is_deepseek_v4_family(hf_config)
        or getattr(hf_config, "dspark_target_layer_ids", None) is None
    ):
        return

    # ``model_type`` stays "deepseek_v4" for V4.1 too. That is what
    # upstream already wrote here before this function took an
    # architecture, and it is what keeps ``is_deepseek_v4_family`` true
    # and stops the multimodal architecture conversion re-running. The
    # drafter reads its own hyper-parameters from ``hf_config``, whose
    # V4.1 fields (dspark_n_routed_experts, dspark_target_layer_ids,
    # num_nextn_predict_layers) this does not touch. So the one thing
    # that differs for V4.1 is which class gets built.
    hf_config.update({"architectures": [architecture]})
    draft_model_config.model_arch_config = replace(
        draft_model_config.model_arch_config,
        architectures=[architecture],
        model_type="deepseek_v4",
        is_mm_prefix_lm=False,
    )
    model_info, architecture = draft_model_config.registry.inspect_model_cls(
        draft_model_config.architectures,
        draft_model_config,
    )
    draft_model_config._model_info = model_info
    draft_model_config._architecture = architecture


#: Architecture string for the V4.1 DSpark drafter. Distinct from V4.0's
#: ``DSparkDraftModel`` because the two networks differ -- see
#: ``_refuse_v40_drafter_for_v41_target`` for what goes wrong if a V4.1
#: target reaches the V4.0 drafter.
V41_DSPARK_ARCHITECTURE = "DSparkDeepseekV41DraftModel"

#: Modules that implement a DeepSeek-V4.1 DSpark drafter on Ascend. A V4.1
#: target whose drafter resolves outside this set is refused.
V41_DSPARK_DRAFTER_MODULES: frozenset[str] = frozenset({"vllm_ascend.models.deepseek_v4_1.dspark"})


def _is_v41_model_type(model_type) -> bool:
    return isinstance(model_type, str) and model_type.startswith("deepseek_v41")


def _is_v41_dspark_pair(spec_config) -> bool:
    """Whether this target/draft pair is V4.1's.

    Upstream rewrites the draft config's *instance* ``model_type`` to
    ``"deepseek_v4"``, so V4.1 is recognised by the target's
    ``model_type`` or by the draft config's *class* attribute, which that
    rewrite leaves alone. Used by both the routing below and the refusal
    above, so the architecture that gets written and the one that gets
    checked cannot disagree.
    """
    target_hf = getattr(getattr(spec_config, "target_model_config", None), "hf_config", None)
    draft_hf = getattr(getattr(spec_config, "draft_model_config", None), "hf_config", None)
    return _is_v41_model_type(getattr(target_hf, "model_type", None)) or _is_v41_model_type(
        getattr(type(draft_hf), "model_type", None)
    )


def _refuse_v40_drafter_for_v41_target(spec_config) -> None:
    """Refuse a V4.1 target whose DSpark drafter resolved to anything unported.

    Upstream rewrites the draft config's *instance* ``model_type`` to
    ``"deepseek_v4"``, so V4.1 is recognised by the target's ``model_type``
    or by the draft config's *class* attribute, which that rewrite leaves
    alone. The decision is made on the registry entry that will actually be
    built, against an allow-list.
    """
    target_hf = getattr(getattr(spec_config, "target_model_config", None), "hf_config", None)
    draft = getattr(spec_config, "draft_model_config", None)
    if not _is_v41_dspark_pair(spec_config):
        return
    draft_hf = getattr(draft, "hf_config", None)
    arch = draft._architecture
    entry = draft.registry.models[arch]
    if entry.module_name in V41_DSPARK_DRAFTER_MODULES:
        return
    raise NotImplementedError(
        "DeepSeek-V4.1 target (model_type="
        f"{getattr(target_hf, 'model_type', None)!r}) with speculative method='dspark' "
        f"resolved the drafter to {entry.module_name}:{entry.class_name} (architecture "
        f"{arch!r}). That is the DeepSeek-V4.0 DSpark drafter: it sizes the draft MoE by "
        f"the target's n_routed_experts={getattr(draft_hf, 'n_routed_experts', '?')} "
        "(this checkpoint's drafter has dspark_n_routed_experts="
        f"{getattr(draft_hf, 'dspark_n_routed_experts', '?')}), expects mtp.*.hc_head_* "
        "and markov_w1/markov_w2, which the V4.1 checkpoint does not have, and reads "
        "config.num_hash_layers, which V4.1 configs do not define. Select "
        "DSparkDeepseekV41DraftModel. Do not simply allow-list the V4.0 drafter."
    )


def _dspark_post_init(self):
    _orig_post_init(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        _normalize_deepseek_v4_dspark_draft(
            draft_model_config, V41_DSPARK_ARCHITECTURE if _is_v41_dspark_pair(self) else "DSparkDraftModel"
        )
        _refuse_v40_drafter_for_v41_target(self)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
