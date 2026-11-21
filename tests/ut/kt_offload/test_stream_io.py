# SPDX-License-Identifier: Apache-2.0
"""The slot provider, judged where a CPU can judge it: WHAT WAS ASKED FOR.

A host has no pinned memory, no H2D and no internal format, so nothing here says
the weights reach the card or that the layout is the one the GEMM accepts. What
it does pin is request-shaped: which byte spans were gathered, that staging went
through the pinned seam, that each chunk's copy really happened, that the cast is
delegated rather than re-derived, and that the slot is sized by the chunk.

The byte oracle is ``safetensors.safe_open`` reached through ``weight_source``,
not a second reader written here; the four device-touching steps are replaced by
a recorder, and each criterion carries a mutation arm that must go red.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.kt_offload import stream_io
from vllm_ascend.kt_offload import weight_source as ws
from vllm_ascend.kt_offload.stream_chain import SLOT_WEIGHT_ATTRS, ChunkSlot
from vllm_ascend.kt_offload.stream_io import (
    PARAM_FOR_GROUP,
    DeviceSeam,
    SlotCoverage,
    StreamIOError,
    StreamSlotProvider,
    chunk_param_shapes,
    default_alloc_pinned,
    default_post_load,
)
from vllm_ascend.kt_offload.weight_source import WeightSource, ZeroCoverageError, scan
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDSDynamicFusedMoEMethod

# Small but shaped like V4.1: hidden and intermediate are multiples of the MXFP
# group size, so every reshape the production post-load hook does is legal.
HIDDEN, INTER, GROUP = 64, 64, 32
SYN_LAYERS, SYN_EXPERTS = 2, 6
CHUNK = 3
RESIDENT_SLOTS = 16  # what a slot mis-sized "like the resident layer" would be

# Points at a real checkpoint; the tests that use it skip when it is absent.
# Override with KT_TEST_CKPT to run them against a checkpoint elsewhere.
REAL_CKPT = os.environ.get("KT_TEST_CKPT", "/workspace/models/DeepSeek-V4.1-Flash")
REAL_HIDDEN, REAL_INTER = 5120, 2304
REAL_LAYER, REAL_EXPERTS = 20, (7, 129)
needs_real = pytest.mark.skipif(
    not os.path.exists(os.path.join(REAL_CKPT, "model.safetensors.index.json")),
    reason=f"real checkpoint not at {REAL_CKPT}",
)


def quant_method(group_size: int = GROUP):
    """The real DS W4A8 MXFP method; ``__init__`` only reads group_size/dynamic_eplb."""
    qm = AscendW4A8MXFPDSDynamicFusedMoEMethod.__new__(AscendW4A8MXFPDSDynamicFusedMoEMethod)
    qm.group_size = group_size
    qm.dynamic_eplb = False
    return qm


# A synthetic checkpoint whose experts differ in their FIRST and LAST bytes.
SYN_SHAPES = {
    ("w1", "weight"): (INTER, HIDDEN // 2),
    ("w3", "weight"): (INTER, HIDDEN // 2),
    ("w2", "weight"): (HIDDEN, INTER // 2),
    ("w1", "scale"): (INTER, HIDDEN // GROUP),
    ("w3", "scale"): (INTER, HIDDEN // GROUP),
    ("w2", "scale"): (HIDDEN, INTER // GROUP),
}


def _syn_bytes(layer, expert, proj, kind) -> torch.Tensor:
    """Bytes that differ between experts at BOTH ends of every record.

    Not random: the stale-buffer arm needs chunk k and chunk k-1 to differ in the
    few bytes the probe reads, so a probe of any width >= 1 discriminates.
    """
    n = SYN_SHAPES[(proj, kind)][0] * SYN_SHAPES[(proj, kind)][1]
    base = (layer * 97 + expert * 13 + ws.PROJS.index(proj) * 3 + ws.KINDS.index(kind)) % 7
    if kind == "scale":
        # E8M0 band: weight_source refuses a 0x00 scale byte (it is the hole a
        # mis-located read leaves), so stay inside 117..127.
        vals = (torch.arange(n, dtype=torch.int64) + expert * 3 + base) % (ws.E8M0_HI - ws.E8M0_LO + 1) + ws.E8M0_LO
    else:
        vals = (torch.arange(n, dtype=torch.int64) * 7 + expert * 29 + base * 101) % 251 + 1
    return vals.to(torch.uint8).reshape(SYN_SHAPES[(proj, kind)])


@pytest.fixture(scope="module")
def syn_ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("stream_io_ckpt")
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    shards = {n: {} for n in names}
    wmap = {}
    for layer in range(SYN_LAYERS):
        for e in range(SYN_EXPERTS):
            for proj in ws.PROJS:
                for kind in ws.KINDS:
                    k = ws.expert_key(layer, e, proj, kind)
                    fn = names[0] if (layer == 0 or e < 2) else names[1]
                    raw = _syn_bytes(layer, e, proj, kind)
                    shards[fn][k] = raw.view(torch.float8_e8m0fnu) if kind == "scale" else raw.view(torch.int8)
                    wmap[k] = fn
    # a neighbour tensor so expert offsets are not all multiples of the records
    shards[names[0]]["layers.0.ffn.gate.weight"] = torch.zeros(3, 5, dtype=torch.bfloat16)
    wmap["layers.0.ffn.gate.weight"] = names[0]
    for fn, tensors in shards.items():
        save_file(tensors, str(d / fn))
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    return str(d)


def oracle_bytes(ckpt_dir: str, layer: int, expert: int, proj: str, kind: str) -> torch.Tensor:
    """``safe_open`` bytes for one tensor -- the reader under test is not involved."""
    from safetensors import safe_open

    key = ws.expert_key(layer, expert, proj, kind)
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        fn = json.load(f)["weight_map"][key]
    with safe_open(os.path.join(ckpt_dir, fn), framework="pt") as h:
        return h.get_tensor(key).contiguous().view(-1).view(torch.uint8).clone()


def expected_record(ckpt_dir: str, layer: int, expert: int, param: str) -> torch.Tensor:
    """What one expert's record in ``param`` must be, straight from the oracle.

    w13 = w1 then w3, per expert -- the byte concat the fused-MoE loader builds
    (shard 0 -> first half, shard 1 -> second half).
    """
    pieces = {
        "w13_weight": (("w1", "weight"), ("w3", "weight")),
        "w13_weight_scale": (("w1", "scale"), ("w3", "scale")),
        "w2_weight": (("w2", "weight"),),
        "w2_weight_scale": (("w2", "scale"),),
    }[param]
    return torch.cat([oracle_bytes(ckpt_dir, layer, expert, p, k) for p, k in pieces])


def cpu_post_load(qm, module) -> None:
    """A CPU stand-in for ``process_weights_after_loading``: only the shape
    rearrangement and the rebinding of all four parameters. The cast has no host
    equivalent, so this says nothing about the layout -- which is why separate
    tests pin that the default seam is the production hook.
    """
    module.w13_weight.data = module.w13_weight.data.transpose(1, 2)
    module.w2_weight.data = module.w2_weight.data.transpose(1, 2)
    g, n, k = module.w13_weight_scale.shape
    module.w13_weight_scale.data = module.w13_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)
    g, n, k = module.w2_weight_scale.shape
    module.w2_weight_scale.data = module.w2_weight_scale.data.reshape(g, n, k // 2, 2).transpose(-3, -2)


class Recorder:
    """Records what was ASKED for at each of the four device seams."""

    def __init__(self, *, h2d=None, post_load=None, alloc_device=None):
        self.pinned: list[int] = []
        self.device: list[tuple] = []
        self.device_tensors: dict[str, torch.Tensor] = {}
        self.h2d_calls: list[dict] = []
        self.waits: list[object] = []
        self.post_load_calls: list[dict] = []
        self._h2d = h2d
        self._post_load = post_load
        self._alloc_device = alloc_device

    # 1. pinned staging
    def alloc_pinned(self, nbytes: int) -> torch.Tensor:
        self.pinned.append(int(nbytes))
        # A host has no page-locked memory to hand out; what is checkable here is
        # that the provider asked this seam rather than allocating for itself.
        return torch.zeros(int(nbytes), dtype=torch.uint8)

    # 2. the one device slot
    def alloc_device(self, shape, dtype, device) -> torch.Tensor:
        self.device.append((tuple(shape), dtype, device))
        if self._alloc_device is not None:
            t = self._alloc_device(shape, dtype, device)
        else:
            # zeroed, so "nothing was copied" is visible rather than plausible
            t = torch.zeros(tuple(shape), dtype=dtype)
        name = SLOT_WEIGHT_ATTRS[len(self.device) - 1] if len(self.device) <= len(SLOT_WEIGHT_ATTRS) else "?"
        self.device_tensors[name] = t
        return t

    # 3. H2D
    def h2d(self, dst, src, *, what):
        self.h2d_calls.append(
            {"what": what, "dst_bytes": int(dst.numel()), "src_bytes": int(src.numel()), "dst_ptr": dst.data_ptr()}
        )
        if self._h2d is not None:
            return self._h2d(dst, src, what=what)
        dst.copy_(src)
        return ("handle", len(self.h2d_calls))

    def wait(self, handle):
        self.waits.append(handle)

    # 4. the production post-load hook
    def post_load(self, qm, module):
        self.post_load_calls.append(
            {
                "quant_method": qm,
                "module": module,
                "before": {n: getattr(module, n).data for n in SLOT_WEIGHT_ATTRS},
            }
        )
        (self._post_load or cpu_post_load)(qm, module)

    def seam(self) -> DeviceSeam:
        return DeviceSeam(
            alloc_pinned=self.alloc_pinned,
            alloc_device=self.alloc_device,
            h2d=self.h2d,
            wait=self.wait,
            post_load=self.post_load,
        )


def make_provider(ckpt_dir, rec: Recorder | None = None, *, chunk=CHUNK, layer=0, **kw):
    rec = rec or Recorder()
    prov = StreamSlotProvider(
        WeightSource(ckpt_dir, num_experts=SYN_EXPERTS),
        quant_method(),
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        chunk_experts=chunk,
        device="cpu",
        seam=rec.seam(),
        layer=layer,
        **kw,
    )
    return prov, rec


def test_copy_threads_is_refused_when_it_would_be_ignored(syn_ckpt):
    """A `copy_threads` the provider cannot honour must not be accepted silently.

    Given a SOURCE OBJECT, the object's own thread count is what gathers, so the
    provider's argument does nothing. That is how the bank arm ended up on
    `MIN_LOADER_THREADS` (4) while this default said 8 -- on the leg the server's
    staging report puts at ~94% of the staging loop. The number is now explicit
    at the one place that builds the source (`wiring.setup_streaming`), and
    passing it here instead is an error.
    """
    src = WeightSource(syn_ckpt, num_experts=SYN_EXPERTS, copy_threads=2)
    rec = Recorder()
    with pytest.raises(ValueError, match="contradicts the source"):
        StreamSlotProvider(
            src,
            quant_method(),
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            chunk_experts=CHUNK,
            device="cpu",
            seam=rec.seam(),
            copy_threads=4,
        )
    # Two things that are NOT contradictions: the default (no claim about a
    # source this provider did not build), and the source's own number restated.
    for ct in (stream_io.DEFAULT_SLOT_COPY_THREADS, 2):
        prov = StreamSlotProvider(
            src,
            quant_method(),
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            chunk_experts=CHUNK,
            device="cpu",
            seam=rec.seam(),
            copy_threads=ct,
        )
        assert prov.source is src and prov.source.copy_threads == 2


def record_mismatches(rec: Recorder, ckpt_dir: str, layer: int, experts) -> list[str]:
    """Every per-expert record on the 'device', against the safe_open oracle."""
    bad = []
    for name in SLOT_WEIGHT_ATTRS:
        buf = rec.device_tensors[name]
        for i, e in enumerate(experts):
            got = buf[i].reshape(-1)
            want = expected_record(ckpt_dir, layer, e, name)
            if got.numel() != want.numel():
                bad.append(f"{name}[{i}] (expert {e}): {got.numel()} B != {want.numel()} B")
            elif not torch.equal(got, want):
                n = int((got != want).sum())
                bad.append(f"{name}[{i}] (expert {e}): {n} byte(s) differ from the checkpoint")
    return bad


def test_module_imports_standalone():
    """Under pytest, conftest has already initialised ``vllm_ascend.ops`` before
    any test imports anything, so a module-scope import cycle inside kt_offload is
    invisible here but kills a script on its first import."""
    r = subprocess.run(
        [sys.executable, "-c", "import vllm_ascend.kt_offload.stream_io"], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, f"stream_io cannot be imported on its own:\n{r.stderr[-2000:]}"


def test_group_map_lines_up_with_both_name_spaces():
    assert set(PARAM_FOR_GROUP) == {g for g, _ in ws.GROUPS}
    assert set(PARAM_FOR_GROUP.values()) == set(SLOT_WEIGHT_ATTRS)


def test_shapes_come_from_the_production_quant_method():
    shapes = chunk_param_shapes(quant_method(), CHUNK, INTER, HIDDEN)
    qm = quant_method()
    want = qm.get_weight(CHUNK, INTER, HIDDEN, torch.bfloat16)
    want.update(qm.get_dynamic_quant_param(CHUNK, INTER, HIDDEN, torch.bfloat16))
    for name in SLOT_WEIGHT_ATTRS:
        assert shapes[name] == (tuple(want[name].shape), want[name].dtype)
    # every parameter is uint8: the streaming path moves bytes (E8M0 scales die
    # in a typed copy), and that is asserted, not assumed
    assert {d for _, d in shapes.values()} == {torch.uint8}


def test_chunk_dimensions_must_match_the_checkpoint(syn_ckpt):
    """Built for the wrong hidden size => refused, not silently short-copied."""
    rec = Recorder()
    prov = StreamSlotProvider(
        WeightSource(syn_ckpt, num_experts=SYN_EXPERTS),
        quant_method(),
        hidden_size=HIDDEN * 2,
        intermediate_size=INTER,
        chunk_experts=CHUNK,
        device="cpu",
        seam=rec.seam(),
        layer=0,
    )
    with pytest.raises(StreamIOError, match="per expert"):
        prov.slot_for(0, (0, 1, 2))


def test_chunk_holds_exactly_the_requested_experts_bytes(syn_ckpt):
    prov, rec = make_provider(syn_ckpt)
    experts = (1, 4, 5)
    slot = prov.slot_for(0, experts)
    assert isinstance(slot, ChunkSlot)
    assert record_mismatches(rec, syn_ckpt, 0, experts) == [], "staged bytes differ from safe_open"
    assert prov.coverage.experts_produced == {(0, e) for e in experts}
    assert prov.coverage.bytes_gathered == prov.bytes_per_expert * len(experts)


def test_arm_expert_gathered_from_the_wrong_offset(syn_ckpt):
    """MUTATION ARM: the locator resolves each expert to its neighbour."""
    prov, rec = make_provider(syn_ckpt)
    real_locate = prov.source.locate
    prov.source.locate = lambda layer, expert, proj, kind="weight": real_locate(
        layer, (expert + 1) % SYN_EXPERTS, proj, kind
    )
    experts = (1, 4, 5)
    prov.slot_for(0, experts)
    bad = record_mismatches(rec, syn_ckpt, 0, experts)
    assert bad, "a chunk gathered from the wrong offsets was accepted by the byte oracle"
    # it is wrong for every parameter, not just one that happened to move
    assert {b.split("[")[0] for b in bad} == set(SLOT_WEIGHT_ATTRS)


def test_gather_covers_every_layer_the_provider_is_bound_to(syn_ckpt):
    prov, rec = make_provider(syn_ckpt)
    prov.slot_for(0, (0, 1, 2))
    prov.set_layer(1)
    prov.slot_for(0, (3, 4))
    assert prov.coverage.layers_served == {0, 1}
    assert record_mismatches(rec, syn_ckpt, 1, (3, 4)) == []


def test_layer_must_be_bound_before_a_chunk_is_asked_for(syn_ckpt):
    rec = Recorder()
    prov = StreamSlotProvider(
        WeightSource(syn_ckpt, num_experts=SYN_EXPERTS),
        quant_method(),
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        chunk_experts=CHUNK,
        device="cpu",
        seam=rec.seam(),
    )
    with pytest.raises(StreamIOError, match="no layer bound"):
        prov.slot_for(0, (0, 1))


def test_empty_chunk_is_zero_coverage_not_a_cheap_one(syn_ckpt):
    prov, _ = make_provider(syn_ckpt)
    with pytest.raises(ZeroCoverageError):
        prov.slot_for(0, ())


def test_chunk_larger_than_the_slot_is_refused(syn_ckpt):
    prov, _ = make_provider(syn_ckpt, chunk=2)
    with pytest.raises(StreamIOError, match="slot holds"):
        prov.slot_for(0, (0, 1, 2))


def test_staging_goes_through_the_pinned_seam(syn_ckpt):
    prov, rec = make_provider(syn_ckpt)
    assert rec.pinned == [prov.staging_bytes] * stream_io.DEFAULT_HOST_BUFFERS
    assert prov.staging_bytes == prov.bytes_per_expert * CHUNK
    assert prov.coverage.pinned_buffers == stream_io.DEFAULT_HOST_BUFFERS


def _asks_for_pinned(src: str) -> bool:
    return "pin_memory=True" in src


def test_default_staging_allocator_asks_for_page_locked_memory():
    """A pageable source copies several times slower than a page-locked one."""
    src = inspect.getsource(default_alloc_pinned)
    assert _asks_for_pinned(src)
    # MUTATION ARM (syntax-safe): drop the flag and the same predicate must go red
    assert not _asks_for_pinned(src.replace("pin_memory=True", "")), "the pinned check has no power"
    # and the provider must not allocate staging behind the seam's back
    assert inspect.getsource(stream_io).count("pin_memory") == 1


def test_h2d_is_issued_per_chunk_with_the_right_sizes(syn_ckpt):
    prov, rec = make_provider(syn_ckpt)
    prov.slot_for(0, (0, 1, 2))
    calls = rec.h2d_calls
    assert [c["what"] for c in calls] == [PARAM_FOR_GROUP[g] for g, _ in ws.GROUPS], (
        "the copies must follow the staging layout, so the pinned buffer is read front to back"
    )
    shapes = chunk_param_shapes(quant_method(), CHUNK, INTER, HIDDEN)
    for c in calls:
        want = shapes[c["what"]][0]
        assert c["src_bytes"] == c["dst_bytes"] == want[0] * want[1] * want[2]
    assert prov.coverage.h2d_calls == len(SLOT_WEIGHT_ATTRS)
    assert prov.coverage.bytes_to_device == prov.bytes_per_expert * 3


def test_one_device_slot_reused_by_every_chunk(syn_ckpt):
    """No device-side ping-pong: same buffers, same addresses, every chunk."""
    prov, rec = make_provider(syn_ckpt)
    prov.slot_for(0, (0, 1, 2))
    prov.slot_for(1, (3, 4, 5))
    assert len(rec.device) == len(SLOT_WEIGHT_ATTRS), "device buffers were allocated more than once"
    first = {c["what"]: c["dst_ptr"] for c in rec.h2d_calls[:4]}
    second = {c["what"]: c["dst_ptr"] for c in rec.h2d_calls[4:]}
    assert first == second
    # the host side is where the double buffering lives
    assert len({id(h.buffer) for h in prov._host}) == stream_io.DEFAULT_HOST_BUFFERS


def test_short_last_chunk_uses_the_first_rows_of_the_slot(syn_ckpt):
    prov, rec = make_provider(syn_ckpt, chunk=4)
    slot = prov.slot_for(0, (2, 5))
    assert slot.num_slots == 2
    for c in rec.h2d_calls:
        shape = chunk_param_shapes(quant_method(), 4, INTER, HIDDEN)[c["what"]][0]
        assert c["src_bytes"] == 2 * shape[1] * shape[2]
    assert record_mismatches(rec, syn_ckpt, 0, (2, 5)) == []


def test_format_refuses_a_chunk_whose_copy_never_happened(syn_ckpt):
    """MUTATION ARM: a control path that reaches the cast without an upload."""
    prov, _ = make_provider(syn_ckpt)
    with pytest.raises(StreamIOError, match="not filled|were not filled"):
        prov._format(CHUNK)


def test_arm_chunk_served_with_the_previous_chunks_bytes(syn_ckpt):
    """MUTATION ARM (the one that stays plausible on-card): H2D moves nothing.

    Without ``probe_bytes`` the provider cannot see it -- the copy was issued, the
    sizes were right, the slot has the right expert count, and the numbers would
    be the previous chunk's. With it, the chunk is refused.
    """
    calls = {"n": 0}

    def stale_h2d(dst, src, *, what):
        calls["n"] += 1
        if calls["n"] > len(SLOT_WEIGHT_ATTRS):  # chunk 0 copies, chunk 1 does not
            return ("handle", calls["n"])
        dst.copy_(src)
        return ("handle", calls["n"])

    prov, rec = make_provider(syn_ckpt, Recorder(h2d=stale_h2d))
    prov.slot_for(0, (0, 1, 2))
    slot = prov.slot_for(1, (3, 4, 5))
    assert slot.num_slots == 3, "the mis-served chunk still looks structurally fine"
    assert record_mismatches(rec, syn_ckpt, 0, (3, 4, 5)), "stale bytes were byte-identical to the truth"
    assert record_mismatches(rec, syn_ckpt, 0, (0, 1, 2)) == [], "the device still holds chunk 0"

    calls["n"] = 0
    prov2, _ = make_provider(syn_ckpt, Recorder(h2d=stale_h2d), probe_bytes=8)
    prov2.slot_for(0, (0, 1, 2))
    with pytest.raises(StreamIOError, match="does not hold the bytes that were staged"):
        prov2.slot_for(1, (3, 4, 5))


def test_probe_passes_when_the_copy_really_happened(syn_ckpt):
    """Positive control for the arm above: the probe must not be always-red."""
    prov, rec = make_provider(syn_ckpt, probe_bytes=8)
    prov.slot_for(0, (0, 1, 2))
    prov.slot_for(1, (3, 4, 5))
    assert record_mismatches(rec, syn_ckpt, 0, (3, 4, 5)) == []
    assert prov.coverage.chunks_served == 2


def _code_names(src: str) -> set[str]:
    """Every identifier, attribute and keyword a piece of source USES -- blind to
    strings, comments and docstrings, which may name the cast to explain why it
    is not called."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def test_default_post_load_seam_is_the_production_hook():
    assert DeviceSeam().post_load is default_post_load
    src = inspect.getsource(default_post_load)
    # It delegates to the hook kt WRAPPED (``_kt_offload_original``), not kt's
    # wrapper: running kt's loader additions on a chunk module rebinds the live
    # layer's routing tables and reloads its CPU half.
    assert "quant_method.process_weights_after_loading" in src
    assert "_kt_offload_original" in src
    # the name it delegates to must exist on the production class (a rename
    # upstream fails here instead of at the first on-card run)
    assert callable(AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading)


def test_module_does_not_re_derive_the_cast():
    """MUTATION ARM (syntax-safe): spelling the cast out here would be a second truth."""
    import vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 as prod

    tokens = {"npu_format_cast", "customize_dtype", "float4_e2m1fn_x2", "float8_e8m0fnu"}
    prod_src = inspect.getsource(prod.AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading)
    assert {t for t in tokens if t in prod_src} >= {"npu_format_cast", "customize_dtype", "float4_e2m1fn_x2"}, (
        "these are not the cast's names"
    )

    # AST, not substring: the module docstring *discusses* the cast (it has to --
    # explaining why it delegates), and a text search cannot tell prose from code.
    assert tokens & _code_names(inspect.getsource(stream_io)) == set(), "stream_io re-derives the cast"
    # MUTATION ARM (syntax-safe): the same reader over code that DOES spell the
    # cast out must come up red, or the check is decoration
    assert tokens & _code_names(prod_src), "the AST check cannot see a cast even when there is one"


def test_post_load_hook_receives_the_chunk_module_and_the_quant_method(syn_ckpt):
    prov, rec = make_provider(syn_ckpt)
    slot = prov.slot_for(0, (0, 1, 2))
    assert len(rec.post_load_calls) == 1
    call = rec.post_load_calls[0]
    assert call["quant_method"] is prov.quant_method
    module = call["module"]
    assert sorted(dict(module.named_parameters())) == sorted(SLOT_WEIGHT_ATTRS)
    # the module's parameters must ALIAS the one device slot, not copy it
    for name in SLOT_WEIGHT_ATTRS:
        assert call["before"][name].data_ptr() == rec.device_tensors[name].data_ptr()
    # ... and the hook's rebinding must not have repointed the reused slot
    for name in SLOT_WEIGHT_ATTRS:
        assert rec.device_tensors[name].shape == chunk_param_shapes(quant_method(), CHUNK, INTER, HIDDEN)[name][0]
    # and every parameter really came back rearranged (address, shape or stride)
    for name in SLOT_WEIGHT_ATTRS:
        assert stream_io.layout_signature(getattr(slot, name)) != stream_io.layout_signature(call["before"][name])


def test_arm_post_load_hook_skipped(syn_ckpt):
    """MUTATION ARM: raw ND bytes handed to the GEMM."""
    prov, _ = make_provider(syn_ckpt, Recorder(post_load=lambda qm, module: None))
    with pytest.raises(StreamIOError, match="post-load hook left"):
        prov.slot_for(0, (0, 1, 2))


def test_arm_slot_sized_by_the_resident_count(syn_ckpt):
    """MUTATION ARM: a slot sized by the resident expert count."""

    def resident_sized(qm, module):
        cpu_post_load(qm, module)
        for name in SLOT_WEIGHT_ATTRS:
            t = getattr(module, name).data
            reps = [RESIDENT_SLOTS] + [1] * (t.dim() - 1)
            getattr(module, name).data = t[:1].repeat(*reps)

    prov, _ = make_provider(syn_ckpt, Recorder(post_load=resident_sized))
    with pytest.raises(StreamIOError, match="the chunk has 3"):
        prov.slot_for(0, (0, 1, 2))


def test_arm_device_buffers_allocated_with_the_wrong_row_count(syn_ckpt):
    """MUTATION ARM: the same mistake one step earlier, at allocation time."""

    def resident_alloc(shape, dtype, device):
        return torch.zeros((RESIDENT_SLOTS,) + tuple(shape)[1:], dtype=dtype)

    with pytest.raises(StreamIOError, match="asked for"):
        make_provider(syn_ckpt, Recorder(alloc_device=resident_alloc))


def test_zero_coverage_fails(syn_ckpt):
    """MUTATION ARM: a provider that served nothing must not report success."""
    prov, _ = make_provider(syn_ckpt)
    with pytest.raises(ZeroCoverageError, match="zero coverage"):
        prov.check_coverage([0, 1, 2], layer=0)
    assert "chunks=0" in prov.coverage.summary()


def test_coverage_expectation_must_come_from_outside(syn_ckpt):
    """A check whose expectation is the object under test cannot fail."""
    prov, _ = make_provider(syn_ckpt)
    prov.slot_for(0, (0, 1, 2))
    with pytest.raises(ValueError, match="must not come from the provider"):
        prov.check_coverage(None)
    assert prov.check_coverage([0, 1, 2], layer=0) == []


def test_coverage_notices_a_chunk_that_was_planned_but_never_served(syn_ckpt):
    prov, _ = make_provider(syn_ckpt)
    prov.slot_for(0, (0, 1, 2))
    bad = prov.coverage.check([(0, e) for e in range(SYN_EXPERTS)], strict=False)
    assert bad and "missing [(0, 3), (0, 4), (0, 5)]" in bad[0]


def test_coverage_byte_accounting_is_checked_against_the_expert_count(syn_ckpt):
    prov, _ = make_provider(syn_ckpt)
    prov.slot_for(0, (0, 1, 2))
    prov.slot_for(1, (3, 4, 5))
    assert prov.coverage.bytes_to_device == prov.coverage.bytes_gathered == prov.bytes_per_expert * 6
    assert prov.check_coverage(range(SYN_EXPERTS), layer=0) == []
    # MUTATION ARM: bytes that never reached the device, accounting untouched
    prov.coverage.bytes_to_device -= 1
    bad = prov.coverage.check(
        [(0, e) for e in range(SYN_EXPERTS)], bytes_per_expert=prov.bytes_per_expert, strict=False
    )
    assert bad and "expert slot(s)" in bad[0]


def test_empty_expected_set_is_not_a_criterion():
    with pytest.raises(ValueError, match="not a criterion"):
        SlotCoverage().check([], strict=True)


def test_prefetch_fills_the_other_host_buffer_and_is_not_re_gathered(syn_ckpt):
    prov, rec = make_provider(syn_ckpt)
    gathers = []
    real_gather = prov.source.gather

    def counting(layer, experts, staging):
        gathers.append((layer, tuple(experts)))
        return real_gather(layer, experts, staging)

    prov.source.gather = counting
    prov.slot_for(0, (0, 1, 2))
    assert prov.prefetch(1, (3, 4, 5)) is True
    assert prov.prefetch(1, (3, 4, 5)) is False, "a pending chunk must not be gathered twice"
    prov.slot_for(1, (3, 4, 5))
    assert gathers == [(0, (0, 1, 2)), (0, (3, 4, 5))]
    assert record_mismatches(rec, syn_ckpt, 0, (3, 4, 5)) == []
    prov.close()


def test_a_pinned_buffer_is_not_refilled_while_its_copy_is_in_flight(syn_ckpt):
    """The reason the H2D seam hands back a handle at all."""
    prov, rec = make_provider(syn_ckpt, host_buffers=1)
    prov.slot_for(0, (0, 1, 2))
    assert rec.waits == []
    prov.slot_for(1, (3, 4, 5))
    assert rec.waits, "the staging buffer was refilled without waiting for its H2D"


def test_a_failing_gather_is_raised_not_swallowed(syn_ckpt):
    """A silent fallback to the hybrid path would file the run as a streamed one."""
    prov, _ = make_provider(syn_ckpt)

    def boom(layer, experts, staging):
        raise OSError("short read")

    prov.source.gather = boom
    with pytest.raises(OSError, match="short read"):
        prov.slot_for(0, (0, 1, 2))
    assert prov.coverage.chunks_served == 0
    prov.close()


@needs_real
def test_real_checkpoint_chunk_matches_the_safe_open_oracle():
    """Two real (layer, expert) pairs, end to end, against safetensors.

    ``scan`` is the oracle: once it is green for these exact keys,
    ``src.read(src.locate(...))`` IS the ``safe_open`` bytes, so the staged
    records are compared against that rather than against a second reader.
    """
    src = WeightSource(REAL_CKPT)
    rep = scan(src, [REAL_LAYER], REAL_EXPERTS)
    assert rep.mismatches == [] and rep.ok, rep.mismatches[:4]
    assert rep.coverage.produced_tensors == len(REAL_EXPERTS) * 3 * 2

    rec = Recorder()
    prov = StreamSlotProvider(
        src,
        quant_method(),
        hidden_size=REAL_HIDDEN,
        intermediate_size=REAL_INTER,
        chunk_experts=len(REAL_EXPERTS),
        device="cpu",
        seam=rec.seam(),
        layer=REAL_LAYER,
    )
    assert prov.bytes_per_expert == 11_796_480 + 5_898_240 + 737_280 + 368_640 == 18_800_640
    slot = prov.slot_for(0, REAL_EXPERTS)
    assert slot.num_slots == len(REAL_EXPERTS)

    pieces = {
        "w13_weight": (("w1", "weight"), ("w3", "weight")),
        "w13_weight_scale": (("w1", "scale"), ("w3", "scale")),
        "w2_weight": (("w2", "weight"),),
        "w2_weight_scale": (("w2", "scale"),),
    }
    for name, members in pieces.items():
        buf = rec.device_tensors[name]
        for i, e in enumerate(REAL_EXPERTS):
            want = torch.cat([src.read(src.locate(REAL_LAYER, e, p, k)) for p, k in members])
            assert torch.equal(buf[i].reshape(-1), want), f"{name}[{i}] (expert {e})"
    assert prov.check_coverage(REAL_EXPERTS, layer=REAL_LAYER) == []
    prov.close()


@needs_real
def test_real_checkpoint_arm_wrong_expert(tmp_path):
    """MUTATION ARM on the real bytes: neighbouring expert, same sizes."""
    src = WeightSource(REAL_CKPT)
    a, b = REAL_EXPERTS
    got = src.read(src.locate(REAL_LAYER, a, "w1", "weight"))
    other = src.read(src.locate(REAL_LAYER, b, "w1", "weight"))
    assert got.numel() == other.numel(), "the arm must differ in content, not in size"
    assert not torch.equal(got, other), "two distinct experts hold identical bytes; the oracle has no power"
    src.close()


def test_default_seam_objects_are_the_real_ones():
    seam = DeviceSeam()
    assert seam.alloc_pinned is stream_io.default_alloc_pinned
    assert seam.alloc_device is stream_io.default_alloc_device
    assert seam.h2d is stream_io.default_h2d
    assert seam.wait is stream_io.default_wait
    assert seam.post_load is stream_io.default_post_load


def test_h2d_default_submits_without_blocking():
    """Non-blocking submit is what lets the copy overlap."""
    src = inspect.getsource(stream_io.default_h2d)
    assert "non_blocking=True" in src
    assert "non_blocking=True" not in src.replace("non_blocking=True", ""), "check has no power"
    # off-card there is no stream, so the handle is None and wait() is a no-op
    dst, host = torch.zeros(4, dtype=torch.uint8), torch.arange(4, dtype=torch.uint8)
    assert stream_io.default_h2d(dst, host, what="w2_weight") is None
    assert torch.equal(dst, host)
    stream_io.default_wait(None)


def test_seam_namespace_is_exactly_the_four_device_steps():
    assert sorted(f for f in DeviceSeam.__dataclass_fields__) == sorted(
        ["alloc_pinned", "alloc_device", "h2d", "wait", "post_load"]
    )
    # every field has a real default, so an injected seam replaces only what a
    # test means to replace and the rest stays production
    assert all(callable(f.default) for f in DeviceSeam.__dataclass_fields__.values())
