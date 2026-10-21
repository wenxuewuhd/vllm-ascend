# SPDX-License-Identifier: Apache-2.0
"""Registry entry point for the DeepSeek-V4.1-Flash DSpark drafter on Ascend.

Subclasses upstream's drafter instead of forking it, the same way
:mod:`.model` does for the target. Everything in upstream's
``vllm/models/deepseek_v4_1/nvidia/dspark.py`` is platform-neutral except
three module-level names, which :mod:`.platform_patches` rebinds (see
:mod:`.dspark_ops`); in particular

* the draft decoder blocks are upstream's ``DeepseekV4DecoderLayer``, whose
  attention class this package already rebinds (:mod:`.attn_selector`), so the
  draft layers get :class:`.attention.AscendDeepseekV41Attention` for free;
* the weight loader is plain Python over the target checkpoint's ``mtp.*``
  tensors and needs no adaptation at all.

**The draft weights are not a separate download.** They ship inside the
target checkpoint under ``mtp.{0,1,2}.*`` -- 2401 tensors, 7.39 GiB of the
475.24 GiB total -- which the target's own weight mapper drops
(``"mtp.": None`` in upstream's ``_make_deepseek_v4_weights_mapper``). So
enabling DSpark reads weights that are already on disk and already being
skipped, rather than adding any.

Why this is a separate architecture string from V4.0's ``DSparkDraftModel``:
the two drafters are not interchangeable. V4.1 sizes its draft MoE by
``dspark_n_routed_experts`` (128, against the backbone's 384), has no learned
``hc_head_*`` (it collapses the hyper-connection copies with the last layer's
pre-mix instead), and ships the Markov factors as ``markov_head.embed`` /
``markov_head.head``. Pointing V4.1 at V4.0's drafter is a wrong-weights
failure, which is why ``patch_speculative_config`` refuses it by name.
"""

from __future__ import annotations

from vllm.logger import logger
from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM

from vllm_ascend.models.deepseek_v4_1.platform_patches import DSPARK_REBINDS, dspark_rebinds_in_force
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts


class AscendDSparkDeepseekV41ForCausalLM(DSparkDeepseekV4ForCausalLM):
    """DeepSeek-V4.1-Flash's DSpark drafter on Ascend.

    The constructor checks that this package's DSpark rebinds are in force.
    Without them the drafter builds and loads correctly and then fails in its
    first forward -- ``mhc_post_tilelang`` on an import that is not installed,
    or ``torch.ops._C.fused_deepseek_v4_*`` on a symbol that does not exist --
    which is after the 7.39 GiB of draft weights have loaded and every earlier
    stage has reported success.

    Only the three DSpark names are checked here. The target-side rebinds are
    asserted by :class:`.model.AscendDeepseekV41ForCausalLM`, which is always
    built in the same process: a drafter exists only alongside its target.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        _assert_dspark_patches_applied()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        checked = 0
        for layer in self.model.layers:
            for experts in layer.ffn.modules():
                if not isinstance(experts, AscendRoutedExperts):
                    continue
                expected = self.config.dspark_n_routed_experts
                if experts.kt_offload is not None or experts.moe_config.num_local_experts != expected:
                    raise RuntimeError("DSpark requires every draft expert to remain in HBM")
                checked += 1
                logger.info(
                    "DSpark draft expert placement: local=%d logical=%d kt_offload=None weight_device=%s",
                    experts.moe_config.num_local_experts,
                    experts.moe_config.num_logical_experts,
                    experts.w13_weight.device,
                )
        if checked != len(self.model.layers):
            raise RuntimeError("Could not verify every DSpark layer's expert placement")


def _assert_dspark_patches_applied() -> None:
    """Refuse to build a drafter whose hardware entry points are CUDA's.

    Reads the live bindings rather than whatever ``apply_all`` reported: an
    upstream module re-imported after the rebinds would restore its original
    names while that list still says success.
    """
    wrong = dspark_rebinds_in_force()
    if wrong:
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark rebinds are not in force:\n  "
            + "\n  ".join(wrong)
            + "\nImporting vllm_ascend.models.deepseek_v4_1 applies them "
            f"({len(DSPARK_REBINDS)} names), so this means something "
            "re-imported upstream's dspark module afterwards and restored its "
            "original names."
        )
