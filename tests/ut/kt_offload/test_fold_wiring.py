# SPDX-License-Identifier: Apache-2.0
"""The folded swap's commit targets are assembled from the LIVE model: every arm
judges ``live_target_violations(targets, **layer.live_swap_objects()) == []``,
and the expectations come from this file's constants (``HOT``, the per-expert
byte fills), never from the fold, the placement or the staging. Only the two
card-only pieces are modelled -- ``torch_npu.copy_memory_`` and the provider's
chunk/cast seam.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.kt_offload import fold_swap, prefill_window
from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import LayerPlacement, build_plan
from vllm_ascend.kt_offload.resident_driver import SLOT_WEIGHT_ATTRS
from vllm_ascend.kt_offload.resident_swap import (
    ALL_CPU_WEIGHTS,
    LayerSwapTargets,
    SwapError,
    live_target_violations,
)

E, SLOTS, CHUNK, TOPK = 12, 4, 3, 2
MOE_LAYERS = (1, 2)  # not 0: a layer mix-up must be able to show
NUM_LAYERS = 4
OLD = (0, 1, 2, 3)
HOT = (1, 5, 7, 10)  # keeps 1 (slot 1 -> 0), promotes 5/7/10, spans all four chunks
#: one row per expert, per parameter; small but of four different widths
PER_EXPERT = {"w13_weight": (6,), "w13_weight_scale": (2,), "w2_weight": (4,), "w2_weight_scale": (2,)}


def chunks_of(num_experts: int = E, chunk: int = CHUNK) -> list[tuple[int, ...]]:
    return [tuple(range(i, min(i + chunk, num_experts))) for i in range(0, num_experts, chunk)]


def expert_bytes(layer: int, expert: int, name: str) -> torch.Tensor:
    """The 'checkpoint' bytes of one expert's one parameter. The oracle's source."""
    n = PER_EXPERT[name][0]
    base = (7 * layer + 13 * expert + 3 * SLOT_WEIGHT_ATTRS.index(name)) % 200 + 1
    return torch.arange(base, base + n, dtype=torch.int64).remainder(251).to(torch.uint8)


def cast_rows(nd: torch.Tensor) -> torch.Tensor:
    """The device cast, modelled: a fresh tensor whose bytes are NOT the ND bytes.
    Per row, like the real per-expert cast, so the oracle can apply it one expert at a time."""
    return nd.flip(-1).contiguous()


def _copy_memory_(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``torch_npu.copy_memory_``: whole storage to whole storage, view ignored."""
    d = torch.empty(0, dtype=torch.uint8).set_(dst.untyped_storage())
    s = torch.empty(0, dtype=torch.uint8).set_(src.untyped_storage())
    if d.numel() != s.numel():
        raise RuntimeError(f"copy_memory_: {s.numel()} B into {d.numel()} B")
    d.copy_(s)


class FakeSlot:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._t = tensors

    def tensors(self) -> dict[str, torch.Tensor]:
        return dict(self._t)


class FakeProvider:
    """The streaming provider's seam, as the fold uses it.

    ``format_nd`` reproduces the production contract: the weights come back as a
    fresh cast, the SCALES are views of the ND bytes they were given (the staging).
    """

    def __init__(self):
        self.shapes = {n: ((CHUNK, *shape), torch.uint8) for n, shape in PER_EXPERT.items()}
        self.device_allocs = 0
        self.device = "cpu"
        self.seam = SimpleNamespace(alloc_device=self._alloc)
        self._layer: int | None = None
        self.served: list[tuple[int, int, tuple[int, ...]]] = []

    def _alloc(self, shape, dtype, device):
        self.device_allocs += 1
        return torch.zeros(tuple(shape), dtype=dtype)

    @property
    def layer(self) -> int:
        if self._layer is None:
            raise RuntimeError("no layer bound")
        return self._layer

    def set_layer(self, layer: int) -> FakeProvider:
        self._layer = int(layer)
        return self

    def slot_for(self, chunk_index: int, expert_ids):
        raise AssertionError("the streaming chain is not exercised here (test_fold_swap.py does that)")

    #: This fake serves ND chunks, so the fold must take the ND path. It is part
    #: of the provider contract (``fold_swap`` reads it rather than guessing),
    #: because guessing ND for a bank-backed provider staged 12,000 experts of
    #: never-written bytes with every guard green.
    nz_device = False

    def served_nd(self, chunk_index: int, expert_ids) -> dict[str, torch.Tensor]:
        ids = tuple(int(e) for e in expert_ids)
        self.served.append((self.layer, int(chunk_index), ids))
        return {n: torch.stack([expert_bytes(self.layer, e, n) for e in ids]) for n in SLOT_WEIGHT_ATTRS}

    def served_nz(self, chunk_index: int, expert_ids):
        raise AssertionError("this provider is in ND mode; the fold must not ask for NZ chunks")

    @property
    def nz_bytes_per_slot(self) -> dict[str, int]:
        return {}

    def format_nd(self, staging: dict[str, torch.Tensor], n_rows: int) -> FakeSlot:
        out = {}
        for name, t in staging.items():
            out[name] = t if name.endswith("_scale") else cast_rows(t[:n_rows])
        return FakeSlot(out)


class FakeKt:
    """kt, as ``resident_swap`` reads it: its OWN pinned mask, and a full-set load."""

    def __init__(self, num_experts: int, resident):
        self.num_experts = num_experts
        # kt copies the mask it is given into this tensor and hands C++ its
        # data_ptr; the tensor layer.py passed to the constructor is not it.
        self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
        self.gpu_experts_mask[list(resident)] = True
        self.loaded = list(range(num_experts))  # KT_FULLSET_LOAD

    def cpu_weight_expert_ids(self):
        return list(self.loaded)

    def cpu_expert_ids(self):
        return [e for e in range(self.num_experts) if not bool(self.gpu_experts_mask[e])]


def _vcfg(profiled: bool = False):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=8, moe_intermediate_size=4, num_experts_per_tok=TOPK, num_hidden_layers=NUM_LAYERS
            ),
            model="/m",
            enforce_eager=True,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"), cudagraph_capture_sizes=[]),
        cache_config=SimpleNamespace(num_gpu_blocks=1234 if profiled else None),
    )


def make_module() -> torch.nn.Module:
    mod = torch.nn.Module()
    for name, shape in PER_EXPERT.items():
        mod.register_parameter(
            name, torch.nn.Parameter(torch.zeros((SLOTS, *shape), dtype=torch.uint8), requires_grad=False)
        )
    return mod


def load_module(mod: torch.nn.Module, layer: int, placement: LayerPlacement) -> None:
    """What a server STARTED with ``placement`` holds: cast bytes at ``slot_of(e)``."""
    for name in SLOT_WEIGHT_ATTRS:
        param = getattr(mod, name)
        for e in placement.resident_ids:
            nd = expert_bytes(layer, e, name)
            param.data[placement.slot_of(e)] = nd if name.endswith("_scale") else cast_rows(nd)


class Model:
    """Two MoE layers whose weights have been loaded; nothing installed yet."""

    def __init__(self, monkeypatch, *, profiled: bool = False, resident=OLD):
        kl._reset_offload_registry()
        monkeypatch.setattr(kl, "_capturing", lambda: False)
        monkeypatch.setattr(kl.KTLayerOffload, "_build_wrapper", lambda rt: FakeKt(E, rt.placement.resident_ids))
        monkeypatch.setattr(torch_npu, "copy_memory_", _copy_memory_, raising=False)
        sets = [list(resident)] * NUM_LAYERS
        plan = build_plan(sets, E)
        cfg = KTOffloadConfig(
            enabled=True,
            cpuinfer_threads=1,
            num_resident_experts=SLOTS,
            stream_prefill=True,
            stream_prefill_min_tokens=4,
            stream_prefill_chunk_experts=CHUNK,
            dynamic_resident=True,
        )
        self.rt: dict[int, kl.KTLayerOffload] = {}
        self.mod: dict[int, torch.nn.Module] = {}
        for i in MOE_LAYERS:
            rt = kl.KTLayerOffload(cfg, plan.layer(i), i, _vcfg(profiled))
            mod = make_module()
            load_module(mod, i, plan.layer(i))
            rt.after_weights_loaded(mod)
            self.rt[i], self.mod[i] = rt, mod
        self.provider = FakeProvider()

    def install(self, **kw):
        return kl.install_resident_fold(self.provider, layers=list(self.rt.values()), **kw)


@pytest.fixture
def model(monkeypatch):
    m = Model(monkeypatch)
    yield m
    kl._reset_offload_registry()


def topk_ids(hot=HOT) -> torch.Tensor:
    """Routing whose unique top-SLOTS is ``hot``."""
    cold = [e for e in range(E) if e not in tuple(hot)]
    rows = [[tuple(hot)[i % len(hot)], cold[i % len(cold)]] for i in range(32)]
    return torch.tensor(rows, dtype=torch.int32)


_SEQ = itertools.count(1)


def publish_one_chunk(key: str | None = None, tokens: int = 32):
    """What the runner publishes for a prompt that fits in one prefill forward."""
    n = next(_SEQ)
    w = prefill_window.PrefillWindow(
        key=key or f"req{n}", first=True, last=True, computed=0, scheduled=tokens, prompt_len=tokens, seq=n
    )
    prefill_window.publish(w)
    return w


def run_layer(model: Model, fold, layer: int, *, window=None):
    """One streamed layer: the fold's own call order, with the chunks faked."""
    if window is None:
        publish_one_chunk()
    else:
        prefill_window.publish(window)
    model.provider.set_layer(layer)

    def run(hook):
        for i, ids in enumerate(chunks_of()):
            if hook is not None:
                hook(i, ids)
        return f"out{layer}"

    return fold.fold_streamed_layer(layer, topk_ids(), run)


def detached_targets(rt: kl.KTLayerOffload) -> LayerSwapTargets:
    """The obvious wrong wiring -- every table rebuilt from the placement -- as the
    positive control: if ``live_target_violations`` cannot see THIS it sees nothing."""
    p = rt.placement
    return LayerSwapTargets(
        layer_idx=p.layer_idx,
        weights={n: getattr(rt._module, n).detach().clone() for n in SLOT_WEIGHT_ATTRS},
        slot_tables={dt: p.safe_slot_table(dt) for dt in (torch.int32, torch.int64)},
        keep_tables={dt: p.keep_table(dt) for dt in (torch.float32, torch.bfloat16)},
        device_mask=p.resident_mask(),
        kt_mask=p.resident_mask(),
        kt_cpu_weight_ids=ALL_CPU_WEIGHTS,
    )


def test_the_criterion_sees_the_wrong_wiring(model):
    rt = model.rt[MOE_LAYERS[0]]
    bad = live_target_violations(detached_targets(rt), **rt.live_swap_objects())
    kinds = {b.split(":")[0].split("[")[0] for b in bad}
    assert kinds == {"weights", "slot_table", "keep_table", "kt_mask"}, bad
    assert all("leave the routing behind" in b for b in bad)


def test_the_assembled_targets_are_what_the_forward_reads(model):
    fold = model.install()
    for i, rt in model.rt.items():
        tg = fold.committer.targets(i)
        assert live_target_violations(tg, **rt.live_swap_objects()) == []
        # Said again by identity: stronger than the address compare the production
        # check can afford (``param.data`` is a new object each time).
        for dt in (torch.int32, torch.int64):
            assert tg.slot_tables[dt] is rt._slot_tables[dt]
        for dt in (torch.float32, torch.bfloat16):
            assert tg.keep_tables[dt] is rt._keep_tables[dt]
        for name in SLOT_WEIGHT_ATTRS:
            assert tg.weights[name] is getattr(model.mod[i], name)
        assert tg.kt_mask is rt.wrapper.gpu_experts_mask
        assert tg.kt_cpu_weight_ids == ALL_CPU_WEIGHTS  # what this kt says it loaded


@pytest.mark.parametrize("key", ["slot_tables", "keep_tables", "weights", "kt_mask"])
def test_mut_a_target_built_afresh_is_refused(monkeypatch, key):
    """Mutation arm: one target built anew instead of taken from the layer. Same
    values, different storage -- the commit would move the weights and leave the
    routing behind, so the assembly itself has to refuse it."""
    m = Model(monkeypatch)
    rt0 = m.rt[MOE_LAYERS[0]]
    honest = kl.KTLayerOffload.live_swap_objects

    def fresh(self):
        live = dict(honest(self))
        p = self.placement
        if key == "slot_tables":
            live[key] = {dt: p.safe_slot_table(dt) for dt in (torch.int32, torch.int64)}
        elif key == "keep_tables":
            live[key] = {dt: p.keep_table(dt) for dt in (torch.float32, torch.bfloat16)}
        elif key == "kt_mask":
            live[key] = p.resident_mask()
        else:
            live[key] = {n: t.detach().clone() for n, t in live[key].items()}
        return live

    monkeypatch.setattr(kl.KTLayerOffload, "live_swap_objects", fresh)
    with pytest.raises(RuntimeError, match="not the objects the forward reads"):
        rt0.build_swap_targets(m.mod[MOE_LAYERS[0]])
    with pytest.raises(RuntimeError, match="not the objects the forward reads"):
        m.install()
    kl._reset_offload_registry()


def test_a_fold_is_never_installed_without_the_live_check(model):
    """Built once, re-checked before every commit: a rebake detaches it silently."""
    fold = model.install()
    assert fold.committer.live_objects is not None
    for i, rt in model.rt.items():
        got, want = fold.committer.live_objects(i), rt.live_swap_objects()
        # the callback reaches the LAYER, not a copy taken when the fold was built
        assert got["kt_mask"] is want["kt_mask"] and got["slot_tables"] is want["slot_tables"]
        assert got["slot_tables"] is rt._slot_tables


def test_a_rebake_detaches_the_targets_and_the_commit_is_refused(model):
    fold, i = model.install(), MOE_LAYERS[0]
    rt, mod = model.rt[i], model.mod[i]
    # Refused for a DIFFERENT reason (wrong resident count), so nothing is written
    # either way; while the targets are attached it fails on that reason instead.
    args = (1, i, (1, 2), {n: torch.zeros_like(getattr(mod, n)) for n in SLOT_WEIGHT_ATTRS})
    with pytest.raises(SwapError) as before:
        fold.committer.commit(*args)
    assert "resident count" in str(before.value)
    assert "not what the forward reads" not in str(before.value)

    rt.after_weights_loaded(mod)  # process_weights_after_loading runs again
    with pytest.raises(SwapError, match="not what the forward reads"):
        fold.committer.commit(*args)
    assert live_target_violations(fold.committer.targets(i), **rt.live_swap_objects()) != []


def test_one_staging_is_shared_by_every_layer(model):
    """One device staging for the whole model, not one per layer."""
    fold = model.install()
    assert model.provider.device_allocs == len(SLOT_WEIGHT_ATTRS)
    assert len(fold.layers) == len(MOE_LAYERS)
    assert {id(model.rt[i]._resident_fold) for i in MOE_LAYERS} == {id(fold)}
    staged = {n: t.data_ptr() for n, t in fold.staging.items()}
    run_layer(model, fold, MOE_LAYERS[0])
    run_layer(model, fold, MOE_LAYERS[1])
    assert model.provider.device_allocs == len(SLOT_WEIGHT_ATTRS)
    assert {n: t.data_ptr() for n, t in fold.staging.items()} == staged


def test_the_step_number_is_one_per_forward_not_one_per_layer(model):
    """The ledger refuses a commit older than the last one; equal is allowed."""
    fold = model.install()
    seen = []
    for _ in range(3):
        for i in MOE_LAYERS:  # a forward walks the layers in order
            model.rt[i]._fold_for_this_forward()
            seen.append(fold.step_source())
    assert seen == [1, 1, 2, 2, 3, 3]


def test_the_provider_is_installed_as_the_slot_callback_too(model):
    """The fold reads chunks back out of the provider that served them."""
    fold = model.install()
    for i in MOE_LAYERS:
        assert model.rt[i]._stream_provider is model.provider
        assert model.rt[i]._stream_slot_provider == model.provider.slot_for
    assert fold.provider is model.provider


def test_the_install_hands_the_fold_the_layers_streaming_threshold(model):
    """One threshold, read once: ``maybe_stream_prefill`` declines a forward under
    it and the fold commits on the last forward that WILL stream. Two readings
    would schedule the commit onto a non-streaming forward, silently."""
    fold = model.install()
    want = {int(rt.cfg.stream_prefill_min_tokens) for rt in model.rt.values()}
    assert len(want) == 1, "the fixture must give every layer the same threshold"
    assert fold.stream_min_tokens == want.pop()


def test_the_install_registers_the_end_of_prefill_check(model, monkeypatch):
    """A prefill that ends having committed nothing must be loud, in PRODUCTION
    wiring: said behaviourally, so the warning has to come out of the fold that
    ``install_resident_fold`` built rather than out of a hand-made one."""
    said = []
    monkeypatch.setattr(fold_swap.logger, "warning", lambda fmt, *a: said.append(fmt % a if a else fmt))
    fold = model.install()
    assert fold.note_prefill_end in prefill_window._end_hooks

    publish_one_chunk(key="never-folded")
    assert said == [], "nothing has ended yet"
    prefill_window.publish(None)  # the step after: not prefilling any more
    assert len(said) == 1 and "never-folded" in said[0], said
    assert fold.coverage.prefills_without_commit == 1


def test_install_refuses_once_the_kv_cache_has_been_sized(monkeypatch):
    m = Model(monkeypatch, profiled=True)
    with pytest.raises(RuntimeError, match="KV cache has already been sized"):
        m.install()
    kl._reset_offload_registry()


def test_install_is_accepted_before_the_profile(model):
    """Positive control for the gate above: the same call, profile not yet run."""
    assert model.install() is not None


def test_install_without_a_registered_layer_raises(monkeypatch):
    kl._reset_offload_registry()
    with pytest.raises(RuntimeError, match="no MoE layer has registered"):
        kl.install_resident_fold(FakeProvider())


def test_build_refuses_a_foreign_module(model):
    rt = model.rt[MOE_LAYERS[0]]
    with pytest.raises(RuntimeError, match="different module"):
        rt.build_swap_targets(model.mod[MOE_LAYERS[1]])


def test_build_refuses_before_the_tables_are_baked(monkeypatch):
    monkeypatch.setattr(kl.KTLayerOffload, "_build_wrapper", lambda rt: FakeKt(E, rt.placement.resident_ids))
    plan = build_plan([list(OLD)] * NUM_LAYERS, E)
    rt = kl.KTLayerOffload(
        KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=SLOTS), plan.layer(1), 1, _vcfg()
    )
    with pytest.raises(RuntimeError, match="no module bound"):
        rt.live_swap_objects()
    with pytest.raises(RuntimeError, match="different module"):
        rt.build_swap_targets(make_module())


def test_build_refuses_inside_a_capture(model, monkeypatch):
    """A capture records the addresses; the targets must exist before it."""
    monkeypatch.setattr(kl, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside a graph capture"):
        model.rt[MOE_LAYERS[0]].build_swap_targets(model.mod[MOE_LAYERS[0]])


def test_a_layer_with_no_kt_half_cannot_be_folded(monkeypatch):
    """Every expert resident: no kt wrapper, so no mask to write. Fail closed."""
    kl._reset_offload_registry()
    monkeypatch.setattr(kl, "_capturing", lambda: False)
    plan = build_plan([list(range(E))] * NUM_LAYERS, E)
    rt = kl.KTLayerOffload(
        KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=E), plan.layer(1), 1, _vcfg()
    )
    mod = torch.nn.Module()
    for name, shape in PER_EXPERT.items():
        mod.register_parameter(
            name, torch.nn.Parameter(torch.zeros((E, *shape), dtype=torch.uint8), requires_grad=False)
        )
    rt.after_weights_loaded(mod)
    assert rt.wrapper is None
    with pytest.raises(RuntimeError, match="no kt wrapper"):
        rt.build_swap_targets(mod)
    kl._reset_offload_registry()


def expected_placement(layer: int, ids=HOT) -> LayerPlacement:
    return LayerPlacement(layer_idx=layer, num_experts=E, num_slots=SLOTS, resident_ids=tuple(sorted(ids)))


def test_the_commit_lands_on_the_objects_the_forward_reads(model):
    fold = model.install()
    i = MOE_LAYERS[0]
    rt, mod = model.rt[i], model.mod[i]
    out, rep = run_layer(model, fold, i)
    assert out == f"out{i}" and rep is not None and rep.swap.ok and not rep.swap.recovered

    want = expected_placement(i)
    # The committer owns the placement; this fixture drives the fold directly,
    # without the streaming chain that copies it back onto the layer.
    assert fold.committer.placement(i).resident_ids == want.resident_ids
    # the tables route() indexes, read off the layer, against this file's HOT
    for dt in (torch.int32, torch.int64):
        assert torch.equal(rt._slot_tables[dt], want.safe_slot_table(dt))
    for dt in (torch.float32, torch.bfloat16):
        assert torch.equal(rt._keep_tables[dt], want.keep_table(dt))
    assert torch.equal(rt.wrapper.gpu_experts_mask, want.resident_mask())
    assert torch.equal(rt._device_mask, want.resident_mask())
    # the parameters the GEMM reads, row by row, against the per-expert bytes
    for name in SLOT_WEIGHT_ATTRS:
        param = getattr(mod, name)
        for e in want.resident_ids:
            nd = expert_bytes(i, e, name)
            row = nd if name.endswith("_scale") else cast_rows(nd)
            assert torch.equal(param.data[want.slot_of(e)], row), (name, e)
    assert rep.swap.weights_verified is True
    # the other layer was not touched by this layer's commit
    other = model.rt[MOE_LAYERS[1]]
    assert fold.committer.placement(MOE_LAYERS[1]).resident_ids == OLD
    assert torch.equal(
        other._slot_tables[torch.int32],
        LayerPlacement(layer_idx=MOE_LAYERS[1], num_experts=E, num_slots=SLOTS, resident_ids=OLD).safe_slot_table(
            torch.int32
        ),
    )


def test_every_layer_lands_and_the_coverage_is_the_chosen_set(model):
    fold = model.install()
    for i in MOE_LAYERS:
        run_layer(model, fold, i)
    want_pairs = {(i, e) for i in MOE_LAYERS for e in HOT}
    assert fold.coverage.check(expected_staged=want_pairs) == []
    assert fold.coverage.layers_committed == len(MOE_LAYERS)
    for i in MOE_LAYERS:
        assert fold.committer.placement(i).resident_ids == tuple(sorted(HOT))


def test_mut_a_commit_whose_tables_were_built_afresh_is_refused(model, monkeypatch):
    """The same run with ONE table detached at commit time: the bytes would still
    be right and the routing would keep pointing at the old slots, so without the
    live check it would pass with every guard green."""
    fold = model.install()
    i = MOE_LAYERS[0]
    rt = model.rt[i]
    rt._slot_tables = {dt: rt.placement.safe_slot_table(dt) for dt in (torch.int32, torch.int64)}
    with pytest.raises(SwapError, match="not what the forward reads"):
        run_layer(model, fold, i)
    assert fold.committer.placement(i).resident_ids == OLD
