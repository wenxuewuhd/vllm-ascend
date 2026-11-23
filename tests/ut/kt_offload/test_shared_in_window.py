# SPDX-License-Identifier: Apache-2.0
"""The shared-expert MLP moves in front of the kt merge.

The criteria in ``kt_offload/shared_window.py``, the runtime accounting on
``KTLayerOffload`` (exactly one ``merge()`` per ``route()``, whoever makes it),
and the production wiring -- the last read out of the source, since those
functions need a device, a forward context and a comm method. Every checker is
also driven on mutated source, so a passing assertion is not a vacuous one.

CPU only; nothing here imports kt_kernel or touches an NPU.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload import shared_window as sw
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan

ROOT = pathlib.Path(__file__).resolve().parents[3]
ROUTED_EXPERTS = ROOT / "vllm_ascend/ops/fused_moe/routed_experts.py"
FUSED_MOE = ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py"

H = 8


def test_env_switch_reads_unset_as_default(monkeypatch):
    """Unset is "" (the default follows the side stream), never "0";
    ``enabled_by_env`` answers only "was it explicitly 1"."""
    from vllm_ascend import envs

    monkeypatch.delenv(sw.ENV_NAME, raising=False)
    assert envs.VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW == ""
    assert sw.enabled_by_env() is False
    monkeypatch.setenv(sw.ENV_NAME, "1")
    assert sw.enabled_by_env() is True
    monkeypatch.setenv(sw.ENV_NAME, "0")
    assert sw.enabled_by_env() is False
    monkeypatch.setenv(sw.ENV_NAME, "true")  # unreadable: refused, not silently off
    with pytest.raises(ValueError, match=sw.ENV_NAME):
        sw.enabled_by_env()


def test_merge_balance_criterion():
    assert sw.merge_balance_ok(1)["ok"]
    assert not sw.merge_balance_ok(0)["ok"]  # the CPU half silently dropped
    assert not sw.merge_balance_ok(2)["ok"]  # duplicated call site


def _deferred_ops():
    """The shape asked for: shared ops inside the fork..join window."""
    return [
        ("fork", None),
        ("window", "resident gmm"),
        ("shared", "dynamic_quant"),
        ("shared", "qbmm1"),
        ("shared", "swiglu"),
        ("shared", "qbmm2"),
        ("join", None),
        ("merge", None),
        ("read", "runner add(shared, routed)"),
    ]


def test_window_ops_criterion_accepts_the_deferred_order():
    rep = sw.window_ops_ok(_deferred_ops())
    assert rep["ok"], rep


def test_window_ops_criterion_accepts_the_side_stream_off_order():
    """No fork/join at all (overlap off): still exactly one merge, still first."""
    ops = [("window", "resident gmm"), ("shared", "qbmm1"), ("merge", None), ("read", "runner add")]
    assert sw.window_ops_ok(ops)["ok"]


@pytest.mark.parametrize(
    "mutate, expect",
    [
        # the un-moved order: shared after the merge buys nothing
        (lambda ops: [o for o in ops if o[0] != "shared"] + [("shared", "qbmm1")], "AFTER the merge"),
        # merge dropped entirely: the CPU half never lands
        (lambda ops: [o for o in ops if o[0] != "merge"], "0 merge(s)"),
        # merge twice
        (lambda ops: ops[:-1] + [("merge", None)] + ops[-1:], "2 merge(s)"),
        # a reader of routed_out before the merge (the runner's add moved up)
        (lambda ops: ops[:-2] + [("read", "runner add")] + ops[-2:], "before the merge"),
        # the join dropped: the H2D would read a pinned output nobody filled
        (lambda ops: [o for o in ops if o[0] != "join"], "without the join"),
        # shared moved out of the window but still before the merge: correct,
        # yet pointless -- the window is already over
        (
            lambda ops: (
                [o for o in ops if o[0] not in ("shared",)][:2] + [("join", None), ("shared", "qbmm1"), ("merge", None)]
            ),
            "after the join",
        ),
    ],
)
def test_window_ops_criterion_mutants_are_red(mutate, expect):
    rep = sw.window_ops_ok(mutate(_deferred_ops()))
    assert not rep["ok"], rep
    assert any(expect in p for p in rep["detail"]), rep["detail"]


class FakeKT:
    """kt stand-in: the CPU half is a deterministic function of the input."""

    def __init__(self):
        self.submits = []
        self.syncs = 0
        self._last = None

    def submit_forward(self, hs, ids, w, stream):
        self.submits.append((hs, ids, w))
        self._last = hs

    def sync_forward(self, hs, stream):
        self.syncs += 1
        return (hs.float() * 0.5).to(hs.dtype)

    def has_pending_deferred(self, layer_idx):
        return False


def _runtime(monkeypatch, shared_in_window=False, overlap=False):
    monkeypatch.setattr(kl, "_stream_handle", lambda device: 0)
    monkeypatch.setenv(sw.ENV_NAME, "1" if shared_in_window else "0")
    plan = build_plan([[0, 1, 2]], 16)
    cfg = KTOffloadConfig(
        enabled=True,
        cpuinfer_threads=4,
        num_resident_experts=1,
        overlap_side_stream=overlap,
    )
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=2),
            model="/m",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
    )
    rt = kl.KTLayerOffload(cfg, plan.layer(0), 0, vcfg)
    monkeypatch.setattr(rt, "_build_wrapper", lambda: FakeKT())
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3), requires_grad=False)
    rt.after_weights_loaded(layer)
    return rt


def _inputs(bs=2):
    hs = torch.arange(bs * H, dtype=torch.float32).reshape(bs, H).to(torch.bfloat16) / 7.0
    ids = torch.tensor([[0, 9], [4, 1]][:bs], dtype=torch.int32)
    w = torch.rand(bs, 2) + 0.1
    return hs, ids, w


def test_switch_follows_the_env(monkeypatch):
    assert _runtime(monkeypatch, shared_in_window=False).shared_in_window is False
    assert _runtime(monkeypatch, shared_in_window=True).shared_in_window is True


def _shared_mlp(hs: torch.Tensor) -> torch.Tensor:
    """Stand-in with the production shape: reads the layer input, nothing else."""
    return (hs.float() * 1.25 + 0.5).to(hs.dtype)


def _step(rt, hs, ids, w, *, defer: bool):
    """One layer the way _forward_impl drives it, with and without the switch."""
    rt.route(hs, w, ids, in_profile_run=False)
    routed = torch.full_like(hs, 0.125)
    ops = [("fork", None), ("window", "resident")]
    if not defer:
        routed = rt.merge(routed)
        ops += [("join", None), ("merge", None)]
    shared = _shared_mlp(hs)
    ops.append(("shared", "mlp"))
    if defer:
        routed = rt.merge(routed)
        ops += [("join", None), ("merge", None)]
    ops.append(("read", "runner add"))
    return (shared.float() + routed.float()).to(hs.dtype), ops


def test_both_orders_are_bit_identical_and_merge_exactly_once(monkeypatch):
    hs, ids, w = _inputs()
    out_off, ops_off = _step(_runtime(monkeypatch, shared_in_window=False), hs, ids, w, defer=False)
    rt_on = _runtime(monkeypatch, shared_in_window=True)
    out_on, ops_on = _step(rt_on, hs, ids, w, defer=True)
    assert torch.equal(out_off, out_on)  # bit-identical, not "close"
    assert rt_on._merges_since_route == 1
    assert sw.window_ops_ok(ops_on)["ok"]
    # the un-deferred order is correct too, it just leaves the window empty
    rep = sw.window_ops_ok(ops_off)
    assert not rep["ok"] and any("AFTER the merge" in p for p in rep["detail"])


def test_several_steps_in_a_row_stay_balanced(monkeypatch):
    rt = _runtime(monkeypatch, shared_in_window=True)
    hs, ids, w = _inputs()
    for _ in range(4):
        _step(rt, hs, ids, w, defer=True)
    assert rt.wrapper.syncs == 4


def test_mutant_forgot_the_merge_is_red(monkeypatch):
    """The deferred call site skipped: the CPU half of this layer vanishes."""
    rt = _runtime(monkeypatch, shared_in_window=True)
    hs, ids, w = _inputs()
    rt.route(hs, w, ids, in_profile_run=False)
    _shared_mlp(hs)  # ... and the merge that should follow is missing
    with pytest.raises(AssertionError, match="0 merge call"):
        rt.route(hs, w, ids, in_profile_run=False)


def test_mutant_merged_twice_is_red(monkeypatch):
    """Both call sites ran: forward_impl did not honour defer_kt_merge."""
    rt = _runtime(monkeypatch, shared_in_window=True)
    hs, ids, w = _inputs()
    rt.route(hs, w, ids, in_profile_run=False)
    routed = rt.merge(torch.full_like(hs, 0.125))
    rt.merge(routed)
    with pytest.raises(AssertionError, match="2 merge call"):
        rt.route(hs, w, ids, in_profile_run=False)


def test_a_no_op_merge_still_counts(monkeypatch):
    """Profile run: kt is skipped, but the call must still be made exactly once,
    or "the merge is a no-op here" becomes a place a missing call site hides."""
    rt = _runtime(monkeypatch, shared_in_window=True)
    hs, ids, w = _inputs()
    rt.route(hs, w, ids, in_profile_run=True)
    assert rt.wrapper.submits == []
    with pytest.raises(AssertionError, match="0 merge call"):
        rt.route(hs, w, ids, in_profile_run=True)


def test_a_route_that_refuses_its_arguments_does_not_arm_the_counter(monkeypatch):
    """A route rejected at the door owes no merge, so the next one is not blamed:
    the counter is armed after the routing-dtype checks."""
    rt = _runtime(monkeypatch, shared_in_window=True)
    hs, ids, w = _inputs()
    with pytest.raises(TypeError, match="topk_ids must be int32/int64"):
        rt.route(hs, ids, w, in_profile_run=False)  # swapped
    rt.route(hs, w, ids, in_profile_run=False)  # not blamed for the refusal
    rt.merge(torch.full_like(hs, 0.125))


def test_the_a6_carve_out_is_narrow(monkeypatch):
    """The un-joined-submit case is left to the more specific
    ``_claim_side_buffers``, but only where that guard is guaranteed to run
    (captured + overlap). Anywhere else the merge counter is the only thing that
    can speak, and it must."""
    rt = _runtime(monkeypatch, shared_in_window=True)
    hs, _, _ = _inputs()
    rt._merge_owed, rt._merges_since_route, rt._side_inflight = True, 0, True
    monkeypatch.setattr(kl, "_capturing", lambda: True)
    rt._assert_merge_balance(hs)  # silent: _claim_side_buffers will refuse
    monkeypatch.setattr(kl, "_capturing", lambda: False)
    with pytest.raises(AssertionError, match="0 merge call"):
        rt._assert_merge_balance(hs)


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _merge_call_sites(src: str) -> list[int]:
    """Lines calling ``<something>.merge(`` on a kt offload handle."""
    return [
        n.lineno
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "merge"
        and ast.unparse(n.func.value).endswith("kt_offload")
    ]


def test_there_is_exactly_one_kt_merge_call_site_in_the_tree():
    """'Exactly once per forward' is only checkable if there is one place to check."""
    hits = []
    for path in ROOT.joinpath("vllm_ascend").rglob("*.py"):
        if _merge_call_sites(path.read_text()):
            hits.append(path.relative_to(ROOT))
    assert hits == [pathlib.Path("vllm_ascend/ops/fused_moe/routed_experts.py")], hits
    assert len(_merge_call_sites(ROUTED_EXPERTS.read_text())) == 1


def test_forward_impl_takes_defer_kt_merge_keyword_only_and_defaults_off():
    fn = _func(ast.parse(ROUTED_EXPERTS.read_text()), "forward_impl")
    names = [a.arg for a in fn.args.kwonlyargs]
    assert "defer_kt_merge" in names, names
    assert fn.args.args == [fn.args.args[0]] and fn.args.args[0].arg == "self"  # everything else keyword-only
    default = fn.args.kw_defaults[names.index("defer_kt_merge")]
    assert isinstance(default, ast.Constant) and default.value is False


def _merge_is_guarded_by_defer(src: str) -> bool:
    """``self.kt_merge(...)`` inside forward_impl runs only when NOT deferring."""
    fn = _func(ast.parse(src), "forward_impl")
    for node in ast.walk(fn):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)):
            continue
        if ast.unparse(node.test.operand) != "defer_kt_merge":
            continue
        if any(
            isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "kt_merge"
            for b in node.body
            for c in ast.walk(b)
        ):
            return True
    return False


def test_forward_impl_skips_the_merge_when_deferring():
    assert _merge_is_guarded_by_defer(ROUTED_EXPERTS.read_text())


def test_forward_impl_guard_check_mutants_are_red():
    # the switch ignored: merge happens in both arms -> with the deferred call
    # site also running, the layer would see two merges
    unguarded = "def forward_impl(self, *, defer_kt_merge=False):\n    routed = self.kt_merge(routed)\n"
    assert not _merge_is_guarded_by_defer(unguarded)
    # inverted: merged only when deferring, i.e. never in the normal arm
    inverted = (
        "def forward_impl(self, *, defer_kt_merge=False):\n"
        "    if defer_kt_merge:\n"
        "        routed = self.kt_merge(routed)\n"
    )
    assert not _merge_is_guarded_by_defer(inverted)


def _defer_arg(call: ast.Call) -> str | None:
    """How a ``forward_impl`` call asks for the deferral, as source text.
    Production passes ``**routed_kwargs`` (empty with the switch off, so the call
    is byte-identical to upstream's); None means the call cannot defer."""
    for kw in call.keywords:
        if kw.arg == "defer_kt_merge":
            return f"defer_kt_merge={ast.unparse(kw.value)}"
        if kw.arg is None:
            return f"**{ast.unparse(kw.value)}"
    return None


def _order_problems(src: str) -> list[str]:
    """What must hold in ``_forward_impl``: one deferred routed call, the merge
    after the shared MLP and before the return, and the short-circuit branch
    (no shared experts) left alone."""
    tree = ast.parse(src)
    fn = _func(tree, "_forward_impl")
    problems = []
    lines: dict[str, list[int]] = {"routed_defer": [], "routed_plain": [], "shared": [], "merge": [], "return": []}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "forward_impl":
                defer = _defer_arg(node)
                lines["routed_defer" if defer is not None else "routed_plain"].append(node.lineno)
                if defer not in (None, "**routed_kwargs"):
                    problems.append(f"forward_impl called with {defer}, expected the flag itself (**routed_kwargs)")
            elif node.func.attr == "forward" and "shared_experts" in ast.unparse(node.func.value):
                lines["shared"].append(node.lineno)
            elif node.func.attr == "kt_merge":
                lines["merge"].append(node.lineno)
        elif isinstance(node, ast.Return) and node.value is not None and "routed_out" in ast.unparse(node.value):
            lines["return"].append(node.lineno)
    if len(lines["merge"]) != 1:
        problems.append(f"{len(lines['merge'])} kt_merge call(s) in _forward_impl, expected exactly 1")
    if len(lines["routed_defer"]) != 1:
        problems.append(f"{len(lines['routed_defer'])} routed call(s) passing defer_kt_merge, expected exactly 1")
    if len(lines["shared"]) != 1:
        problems.append(f"{len(lines['shared'])} shared-expert forward call(s), expected exactly 1")
    if problems:
        return problems
    if not lines["merge"][0] > lines["shared"][0]:
        problems.append("the kt merge is not after the shared-expert MLP: nothing moved into the window")
    for ret in lines["return"]:
        if ret < lines["merge"][0]:
            problems.append("_forward_impl returns routed_out before the merge: the CPU half is not in it")
    for plain in lines["routed_plain"]:
        if plain > lines["shared"][0]:
            problems.append("a routed call after the shared MLP does not pass defer_kt_merge")
    return problems


def test_production_forward_impl_defers_and_merges_after_the_shared_mlp():
    assert _order_problems(FUSED_MOE.read_text()) == []


MUTANTS = {
    # the merge moved back in front of the shared MLP: correct, no benefit
    "merge-before-shared": (
        """
def _forward_impl(self):
    if self.ascend_shared_experts is None:
        return self.routed_experts.forward_impl(hidden_states=h)
    routed_kwargs = {"defer_kt_merge": True} if self.routed_experts.kt_shared_in_window else {}
    routed_out = self.routed_experts.forward_impl(hidden_states=h, **routed_kwargs)
    routed_out = self.routed_experts.kt_merge(routed_out)
    shared_out = self.ascend_shared_experts.forward(x)
    return shared_out, routed_out
""",
        "not after the shared-expert MLP",
    ),
    # the deferred call site dropped: the CPU half is silently missing
    "no-merge": (
        """
def _forward_impl(self):
    routed_kwargs = {"defer_kt_merge": True} if self.routed_experts.kt_shared_in_window else {}
    routed_out = self.routed_experts.forward_impl(hidden_states=h, **routed_kwargs)
    shared_out = self.ascend_shared_experts.forward(x)
    return shared_out, routed_out
""",
        "0 kt_merge call(s)",
    ),
    # merged twice in the same branch
    "double-merge": (
        """
def _forward_impl(self):
    routed_kwargs = {"defer_kt_merge": True} if self.routed_experts.kt_shared_in_window else {}
    routed_out = self.routed_experts.forward_impl(hidden_states=h, **routed_kwargs)
    shared_out = self.ascend_shared_experts.forward(x)
    routed_out = self.routed_experts.kt_merge(routed_out)
    routed_out = self.routed_experts.kt_merge(routed_out)
    return shared_out, routed_out
""",
        "2 kt_merge call(s)",
    ),
    # returned before the merge: the runner adds shared to an un-merged tensor
    "return-before-merge": (
        """
def _forward_impl(self):
    routed_kwargs = {"defer_kt_merge": True} if self.routed_experts.kt_shared_in_window else {}
    routed_out = self.routed_experts.forward_impl(hidden_states=h, **routed_kwargs)
    shared_out = self.ascend_shared_experts.forward(x)
    if early:
        return shared_out, routed_out
    routed_out = self.routed_experts.kt_merge(routed_out)
    return shared_out, routed_out
""",
        "returns routed_out before the merge",
    ),
    # the flag hard-wired instead of read from the layer
    "hardwired-true": (
        """
def _forward_impl(self):
    routed_out = self.routed_experts.forward_impl(hidden_states=h, defer_kt_merge=True)
    shared_out = self.ascend_shared_experts.forward(x)
    routed_out = self.routed_experts.kt_merge(routed_out)
    return shared_out, routed_out
""",
        "expected the flag itself",
    ),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_forward_impl_order_mutants_are_red(name):
    src, expect = MUTANTS[name]
    problems = _order_problems(src)
    assert problems, f"{name} passed the check"
    assert any(expect in p for p in problems), problems


def test_the_short_circuit_branch_keeps_the_merge_inside_the_routed_path():
    """No shared experts -> nothing to hide the merge behind -> no deferral."""
    fn = _func(ast.parse(FUSED_MOE.read_text()), "_forward_impl")
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "forward_impl"
    ]
    plain = [c for c in calls if _defer_arg(c) is None]
    assert len(plain) == 1, [ast.unparse(c) for c in calls]  # the ascend_shared_experts is None branch


def test_no_other_caller_of_forward_impl_defers():
    """Every other call site in the tree (SP wrapper, other models, DBO/MTP)
    leaves the merge where it always was."""
    offenders = []
    for path in ROOT.joinpath("vllm_ascend").rglob("*.py"):
        if path == FUSED_MOE:
            continue
        for n in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "forward_impl"
                and _defer_arg(n) is not None
            ):
                offenders.append((str(path.relative_to(ROOT)), n.lineno))
    assert offenders == [], offenders


# The production ``AscendMoERunner._forward_impl`` itself, run unbound on a
# stand-in ``self`` around a real KTLayerOffload, so the order and the merge
# count are observed rather than read out of the source. The routed half is a
# stand-in whose tail and whose ``kt_merge`` / ``kt_shared_in_window`` ARE the
# production members; the mutants are applied to the production source text.

import contextlib  # noqa: E402
import inspect  # noqa: E402
import textwrap  # noqa: E402

import regex as re  # noqa: E402

from vllm_ascend.ops.fused_moe import fused_moe as fm  # noqa: E402
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: E402


class _Stream:
    def record_event(self):
        return object()


class _Routed:
    kt_shared_in_window = AscendRoutedExperts.kt_shared_in_window
    kt_merge = AscendRoutedExperts.kt_merge

    def __init__(self, rt, log, *, with_event, ignore_defer=False):
        self.kt_offload, self.log, self.with_event, self.ignore_defer = rt, log, with_event, ignore_defer
        self.events = []

    def forward_impl(self, *, hidden_states, router_logits, input_ids=None, defer_kt_merge=False):
        _, ids, w = _inputs(hidden_states.shape[0])
        self.log.append("route")
        self.kt_offload.route(hidden_states, w, ids, in_profile_run=False)
        routed_out = (hidden_states.float() * -0.75).to(hidden_states.dtype)  # the resident half
        if self.ignore_defer or not defer_kt_merge:  # the production tail
            routed_out = self.kt_merge(routed_out)
        if not self.with_event:
            return routed_out
        ev = SimpleNamespace(before_combine=None)
        self.events.append(ev)
        return routed_out, ev


class _Shared:
    def __init__(self, log):
        self.log, self.events_seen = log, []

    def forward(self, x, fused_moe_evts, input_is_gathered=False, defer_output_wait=False):
        self.log.append("shared")
        self.events_seen.append(fused_moe_evts)
        return _shared_mlp(x)


def _runner(rt, log, *, has_shared, ignore_defer=False):
    routed = _Routed(rt, log, with_event=has_shared, ignore_defer=ignore_defer)
    # The rt's merge logs itself, wherever it is called from.
    real_merge = rt.merge

    def logged_merge(x):
        log.append("merge")
        return real_merge(x)

    rt.merge = logged_merge
    ns = SimpleNamespace(
        _sequence_parallel_context=contextlib.nullcontext,
        ascend_shared_experts=_Shared(log) if has_shared else None,
        is_internal_router=False,
        routed_experts=routed,
        routed_input_transform=None,
        routed_output_transform=None,
        _can_overlap_sp_shared_with=lambda t: False,
    )
    # ``_forward_impl``'s `with` acquires this too. It was added to production
    # after this fixture was written, so every test in this file died at
    # "'SimpleNamespace' object has no attribute '_layer_moe_context'" -- ten
    # tests of the shared-expert merge ORDER silently not running.
    # Bound from the PRODUCTION class rather than replaced with a nullcontext:
    # it reads ``self._layer_moe_comm_method`` (absent here, so a plain yield),
    # and binding it means this file notices if the context starts doing
    # something instead of passing because it was stubbed out.
    ns._layer_moe_context = fm.AscendMoERunner._layer_moe_context.__get__(ns)
    return ns


def _production_forward_impl(mutate=None):
    """AscendMoERunner._forward_impl from the tree, optionally mutated as text."""
    src = textwrap.dedent(inspect.getsource(fm.AscendMoERunner._forward_impl))
    if mutate is not None:
        new = mutate(src)
        assert new != src, "the mutation did not apply -- the production source changed shape"
        src = new
    ns = dict(vars(fm))
    exec(compile(src, str(FUSED_MOE), "exec"), ns)  # noqa: S102
    return ns["_forward_impl"]


def _drive(monkeypatch, fwd, *, switch, has_shared=True, steps=3, ignore_defer=False):
    """``steps`` forwards of one layer; returns (outputs, per-step op logs, runner, problems)."""
    monkeypatch.setattr(torch.npu, "current_stream", lambda *a, **k: _Stream(), raising=False)
    rt = _runtime(monkeypatch, shared_in_window=switch)
    log: list[str] = []
    runner = _runner(rt, log, has_shared=has_shared, ignore_defer=ignore_defer)
    outs, logs, problems = [], [], []
    for k in range(steps):
        hs, _, _ = _inputs()
        hs = hs + k  # a different input per step
        start = len(log)
        try:
            res = fwd(runner, hs, torch.zeros(hs.shape[0], 4), None)
        except AssertionError as e:  # the runtime guard
            problems.append(f"step {k}: {e}")
            break
        shared, routed = res if isinstance(res, tuple) else (None, res)
        outs.append(routed if shared is None else (shared.float() + routed.float()).to(hs.dtype))
        logs.append(log[start:])
    return outs, logs, runner, problems


def _judge(monkeypatch, fwd, **kw):
    """Everything asked of the switch-on arm, as a list of problems."""
    ref, _, _, _ = _drive(monkeypatch, _production_forward_impl(), switch=False)
    outs, logs, runner, problems = _drive(monkeypatch, fwd, switch=True, **kw)
    for k, lg in enumerate(logs):
        if lg.count("merge") != 1:
            problems.append(f"step {k}: {lg.count('merge')} merge(s): {lg}")
        elif lg != ["route", "shared", "merge"]:
            problems.append(f"step {k}: order {lg}, want route -> shared -> merge")
    if len(outs) != len(ref) or not all(torch.equal(a, b) for a, b in zip(outs, ref)):
        problems.append("output not bit-identical to the switch-off arm")
    shared = runner.ascend_shared_experts
    if shared is not None and [id(e) for e in shared.events_seen] != [id(e) for e in runner.routed_experts.events]:
        problems.append("the shared expert did not get the routed half's FusedMoEEvents object")
    return problems


def test_real_forward_impl_switch_off_keeps_the_original_order(monkeypatch):
    outs, logs, _, problems = _drive(monkeypatch, _production_forward_impl(), switch=False)
    assert problems == []
    assert logs == [["route", "merge", "shared"]] * 3


def test_real_forward_impl_switch_on_merges_after_shared_exactly_once(monkeypatch):
    assert _judge(monkeypatch, _production_forward_impl()) == []


@pytest.mark.parametrize("switch", [False, True])
def test_real_forward_impl_no_shared_expert_branch_merges_inside_routed(monkeypatch, switch):
    """``ascend_shared_experts is None``: nothing to hide the merge behind, so
    the merge stays in the routed half whatever the switch says."""
    outs, logs, _, problems = _drive(monkeypatch, _production_forward_impl(), switch=switch, has_shared=False)
    assert problems == []
    assert logs == [["route", "merge"]] * 3


def test_forward_impl_called_directly_merges_inside(monkeypatch):
    """Any other caller of forward_impl (SP wrapper, other models, DBO/MTP)
    passes no defer_kt_merge and gets a merged tensor back, switch on or off."""
    rt = _runtime(monkeypatch, shared_in_window=True)
    log: list[str] = []
    runner = _runner(rt, log, has_shared=False)
    hs, _, _ = _inputs()
    for _ in range(3):
        runner.routed_experts.forward_impl(hidden_states=hs, router_logits=None)
    assert log == ["route", "merge"] * 3


def _drop(line):
    return lambda src: src.replace(line, "pass  # MUTANT", 1)


REAL_MUTANTS = {
    # the deferred call site deleted: the CPU half of every layer vanishes
    "no-merge": (_drop("routed_out = self.routed_experts.kt_merge(routed_out)"), "0 merge call"),
    # the deferral not passed to the routed half: it merges, and so does the caller
    "not-passed": (
        lambda src: src.replace("**routed_kwargs,", "", 1),
        "2 merge call",
    ),
    # the switch read as always-off: correct and pointless -- the order check says so
    "switch-ignored": (
        lambda src: src.replace(
            'defer_kt_merge = getattr(self.routed_experts, "kt_shared_in_window", False) is True',
            "defer_kt_merge = False",
            1,
        ),
        "want route -> shared -> merge",
    ),
    # the merge moved behind the return: the runner would add an un-merged tensor
    "merge-after-return": (
        lambda src: re.sub(
            r"\n(\s+)if defer_kt_merge:\n", r"\n\1return shared_out, routed_out\n\1if defer_kt_merge:\n", src, count=1
        ),
        "0 merge call",
    ),
    # the events handed to the shared expert swapped for a fresh object
    "events-replaced": (
        lambda src: re.sub(
            r"fused_moe_events,(\s+)input_is_gathered",
            r"type(fused_moe_events)(before_combine=None),\1input_is_gathered",
            src,
            count=1,
        ),
        "FusedMoEEvents object",
    ),
}


@pytest.mark.parametrize("name", sorted(REAL_MUTANTS))
def test_real_forward_impl_mutants_are_red(monkeypatch, name):
    mutate, expect = REAL_MUTANTS[name]
    problems = _judge(monkeypatch, _production_forward_impl(mutate))
    assert problems, f"mutant {name} passed"
    assert any(expect in p for p in problems), problems


def test_routed_half_ignoring_defer_is_red(monkeypatch):
    """The other half of 'not-passed': forward_impl ignores the flag it is given."""
    problems = _judge(monkeypatch, _production_forward_impl(), ignore_defer=True)
    assert any("2 merge call" in p for p in problems), problems


# The side-stream simulator of test_side_stream_overlap: a shared-expert op
# enqueued before the merge lands between the fork and the join; enqueued after,
# behind the H2D. Same values either way, and only the first is hidden.

from tests.ut.kt_offload import test_side_stream_overlap as sso  # noqa: E402

sim = sso.sim  # the fixture, by name

SHARED_COST = 10.0  # fits inside the modelled window (side 45 vs resident 30)


def _capture_with_shared(monkeypatch, sim, boxes, *, overlap, shared_before_merge):
    rt, _ = sso._runtime(monkeypatch, sim, overlap=overlap, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = sso._inputs(7)
    rt.route(hs, w, ids, in_profile_run=False)
    sim.enqueue(sim.compute, sso.Op("npu_experts", lambda: None, 30.0))
    shared = sso.Op("shared_mlp", lambda: None, SHARED_COST)
    if shared_before_merge:
        sim.enqueue(sim.compute, shared)
    rt.merge(torch.zeros(1, sso.H, dtype=torch.bfloat16))
    if not shared_before_merge:
        sim.enqueue(sim.compute, shared)
    sim.capturing = False
    assert rt._merges_since_route == 1
    return {n: list(ops) for n, ops in sim.programs.items()}


def _replays(monkeypatch, sim, *, overlap, shared_before_merge):
    sim.reset_programs()
    boxes = sso._boxes()
    progs = _capture_with_shared(monkeypatch, sim, boxes, overlap=overlap, shared_before_merge=shared_before_merge)
    outs, walls = [], []
    for v in (2.0, 5.0, 9.0):
        y, t = sso._replay(sim, progs, boxes[0], boxes[3], v)
        outs.append(y)
        walls.append(t)
    return progs, outs, max(walls)


def test_side_stream_on_shared_lands_inside_the_window(monkeypatch, sim):
    p_after, o_after, t_after = _replays(monkeypatch, sim, overlap=True, shared_before_merge=False)
    p_before, o_before, t_before = _replays(monkeypatch, sim, overlap=True, shared_before_merge=True)
    kinds = [o.kind for o in p_before["compute"]]
    assert kinds.index("shared_mlp") < kinds.index("wait") < kinds.index("h2d"), kinds  # fork..join
    kinds_after = [o.kind for o in p_after["compute"]]
    assert kinds_after.index("shared_mlp") > kinds_after.index("h2d"), kinds_after
    assert sso._bits_equal(o_after, o_before), (o_after, o_before)
    assert t_after - t_before == pytest.approx(SHARED_COST), (t_after, t_before)


def test_side_stream_off_either_order_is_the_same(monkeypatch, sim):
    """Overlap off: correct, bit-identical, and nothing to gain (no window)."""
    _, o_after, t_after = _replays(monkeypatch, sim, overlap=False, shared_before_merge=False)
    _, o_before, t_before = _replays(monkeypatch, sim, overlap=False, shared_before_merge=True)
    assert sso._bits_equal(o_after, o_before)
    assert t_after == pytest.approx(t_before)
