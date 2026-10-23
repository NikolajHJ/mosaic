"""Small engineering fixtures; these establish no native-model science result."""

from dataclasses import replace
import json

import pytest
import torch

from latentmol.boltz.active_set import ContextBank, make_branch, run_arm, score_at, select_slots, validate_branch
from latentmol.graph_policy import graph_from_decoder
from latentmol.multimodel.engine import Counters
from latentmol.optimization import StepControl
from latentmol.repair_guidance.config import RepairGuidanceConfig
from latentmol.repair_guidance.loss import FrozenRepairTarget, fixed_target_nll, target_score_at
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob, hard_graph_log_prob_terms
from test_boltz_active_set_runtime import SPEC, _config, _Decoder, _Run, _Scorer


def _target(spec, active, *, step=0):
    atoms = active.long()[None]
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros(1, spec.max_nodes, spec.max_nodes, dtype=torch.long)
    slots = active.nonzero().flatten().tolist()
    for left, right in zip(slots, slots[1:]):
        bonds[0, left, right] = bonds[0, right, left] = 1
    return FrozenRepairTarget.create(HardGraphBatch(atoms, charges, bonds), active,
                                    refresh_step=step, receipt_path="fixture-reference.json")


def test_exact_sum_counts_pairs_once_and_padding_has_no_derivative():
    spec = GraphSpec(max_nodes=4)
    active = torch.tensor([True, False, True, False])
    values = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for shape in
              ((1, 4, 10), (1, 4, 5), (1, 4, 4, 5))]
    decoded = DecoderOutput(*values)
    graph = graph_from_decoder(decoded, spec, temperature=.7, fixed_active=active)
    target = _target(spec, active)
    nll, detail = fixed_target_nll(decoded, graph, target, spec, temperature=.7)
    torch.testing.assert_close(nll, -hard_graph_log_prob(decoded, target.graph, spec,
                                                       temperature=.7, fixed_active=active)[0])
    terms = hard_graph_log_prob_terms(decoded, target.graph, spec, temperature=.7, fixed_active=active)
    for name in ("atoms", "charges", "bonds"):
        assert detail["terms"][name] == float(-terms[name][0])
    expected_edge = -((values[2][0, 0, 2] + values[2][0, 2, 0]) / (2 * .7)).log_softmax(0)[1]
    torch.testing.assert_close(-terms["bonds"][0], expected_edge)
    assert detail["decisions"] == {"atoms": 2, "charges": 2, "bonds": 1}
    normalized, _ = fixed_target_nll(decoded, graph, target, spec, temperature=.7,
                                     normalization="mean_per_decision")
    torch.testing.assert_close(normalized, nll / 5)
    ga, gc, gb = torch.autograd.grad(nll, values)
    assert torch.equal(ga[..., 0], torch.zeros_like(ga[..., 0]))
    assert not ga[:, ~active].any() and not gc[:, ~active].any()
    assert not gb[:, ~active].any() and not gb[:, :, ~active].any()
    assert not gb.diagonal(dim1=1, dim2=2).any()


def test_target_underflow_retains_finite_nonzero_gradient_and_rejects_other_shared_graph():
    spec = GraphSpec(max_nodes=1, atomic_numbers=(0, 6, 7), formal_charges=(0,))
    atoms = torch.tensor([[[0., -1000., 0.]]], requires_grad=True)
    decoded = DecoderOutput(atoms, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1, 5))
    active = torch.tensor([True])
    graph = graph_from_decoder(decoded, spec, fixed_active=active)
    target = _target(spec, active)
    nll, detail = fixed_target_nll(decoded, graph, target, spec)
    assert detail["target_probability_underflows"]["atoms"] == 1
    assert nll == 1000
    gradient, = torch.autograd.grad(nll, atoms)
    assert gradient.tolist() == [[[0., -1., 1.]]]
    wrong = graph_from_decoder(decoded, spec, temperature=2, fixed_active=active)
    # Avoid a graph whose probability has underflowed at both temperatures.
    decoded = DecoderOutput(torch.tensor([[[0., -2., 0.]]]), decoded.charges, decoded.bonds)
    right = graph_from_decoder(decoded, spec, fixed_active=active)
    wrong = graph_from_decoder(decoded, spec, temperature=2, fixed_active=active)
    with pytest.raises(AssertionError):
        fixed_target_nll(decoded, wrong, target, spec)
    fixed_target_nll(decoded, right, target, spec)


def test_target_is_private_detached_and_mask_or_mutation_fails_closed():
    active = torch.tensor([True, False, False])
    target = _target(SPEC, active)
    original = target.graph.atoms.clone()
    active[0] = False
    assert target.active[0] and torch.equal(target.graph.atoms, original)
    with pytest.raises(ValueError, match="identity/mask"):
        target.verify(active)
    target.graph.atoms[0, 0] = 2
    with pytest.raises(ValueError, match="identity/mask"):
        target.verify(target.active)


def _guided_cfg(tmp_path, *, weight=.2, interval=1, steps=2):
    base = _config(tmp_path)
    return replace(base, arms=("fixed_initial_mask",), fd_policy="local_v1",
                   step_control=StepControl(version="armijo_v1"),
                   optimizer=replace(base.optimizer, steps=steps, learning_rate=.1),
                   repair_guidance=RepairGuidanceConfig(provider={}, weight=weight, refresh_interval=interval))


@pytest.mark.parametrize("policy", ["local_v1", "branch_local_v2", "additive_branch_v3", "pointwise_native_ad_v4"])
def test_supported_derivative_policy_reaches_native_provider_setup(tmp_path, monkeypatch, policy):
    """Catch a loader/runtime policy mismatch before submitting a native pilot.

    The sentinel stops at provider construction. It establishes dispatch only;
    actual native gradient and repair checks remain mandatory in readiness.
    """
    import latentmol.repair_guidance.provider as provider
    class ProviderReached(RuntimeError):
        pass
    def reached(*args, **kwargs):
        raise ProviderReached("provider setup reached")
    monkeypatch.setattr(provider, "make_provider", reached)
    cfg = replace(_guided_cfg(tmp_path), fd_policy=policy)
    decoder = _Decoder()
    start = torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(start), temperature=1., threshold=.5)
    with pytest.raises(ProviderReached):
        run_arm("fixed_initial_mask", decoder, start, initial, ContextBank(cfg, SPEC), _Scorer(), cfg,
                "c" * 64, _Run(tmp_path), Counters())


def test_total_gradient_is_weighted_sum_and_independent_repair_fd_runs(tmp_path):
    cfg = _guided_cfg(tmp_path)
    decoder = _Decoder()
    point = torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(point), temperature=1., threshold=.5)
    target = _target(SPEC, initial.active)
    branch = make_branch(_Scorer(), ContextBank(cfg, SPEC), initial, cfg, "c" * 64)
    plain, binding_gradient = score_at(decoder, point, branch, replace(cfg, repair_guidance=None),
                                       Counters(), "fixture", gradients=True)
    repair_row, repair_gradient = target_score_at(decoder, point, initial.active, target, gradients=True)
    total, total_gradient = score_at(decoder, point, branch, cfg, Counters(), "fixture", gradients=True,
                                     repair_target=target)
    torch.testing.assert_close(total_gradient, binding_gradient + cfg.repair_guidance.weight * repair_gradient)
    assert total["loss"] == pytest.approx(plain["loss"] + cfg.repair_guidance.weight * repair_row["loss"])
    assert total["repair_guidance"]["status"] == "active"
    report = validate_branch(decoder, point, branch, cfg, Counters(), repair_target=target)
    assert report["native_seed_parity"]
    assert report["repair_only_validation"]["status"] == "validated_nonzero"
    assert report["local_fd_validation"]["passed"]


def test_refresh_is_between_steps_only_failure_clears_target_and_endpoint_keeps_last(tmp_path, monkeypatch):
    cfg = _guided_cfg(tmp_path, interval=1, steps=3)
    calls = []
    class ProviderFixture:
        def refresh(self, decoder, point, active, *, temperature, step, directory, counters):
            calls.append((step, point.detach().clone()))
            target = None if step == 1 else _target(SPEC, active, step=step)
            return target, {"attempts": [{"status": "accepted" if target else "rejected"}], "step": step}
    monkeypatch.setattr("latentmol.repair_guidance.provider.make_provider", lambda *args: ProviderFixture())
    decoder = _Decoder()
    start = torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(start), temperature=1., threshold=.5)
    run = _Run(tmp_path)
    summary = run_arm("fixed_initial_mask", decoder, start, initial, ContextBank(cfg, SPEC),
                      _Scorer(), cfg, "c" * 64, run, Counters())
    assert [item[0] for item in calls] == [0, 1, 2]
    assert summary["accepted_updates"] == 3
    assert summary["repair_guidance"]["guided_accepted_updates"] == 2
    assert summary["endpoint"]["score"]["repair_guidance"]["refresh_step"] == 2
    folder = tmp_path / "fixed_initial_mask"
    failed_refresh = json.loads((folder / "repair-refresh-objective-0001.json").read_text())
    assert failed_refresh["old_target_sha256"] and failed_refresh["new_target_sha256"] is None
    row = json.loads((folder / "metrics-0001.json").read_text())
    assert row["score"]["loss"] == row["score"]["binding_loss"]
    # Oracle count is only the declared refreshes, despite many FD/trial probes.
    assert summary["repair_guidance"]["refreshes"] == len(calls)
    assert all(json.loads((folder / f"gate-{i:04d}.json").read_text())["passed"] for i in range(3))


def test_guidance_rejects_changing_occupancy_arm_and_absent_config_is_legacy_identity(tmp_path):
    cfg = _config(tmp_path)
    assert "repair_guidance" not in cfg.record()
    assert "dbmol" not in cfg.record()
    guided = replace(cfg, repair_guidance=RepairGuidanceConfig(provider={}), fd_policy="local_v1",
                     step_control=StepControl(version="armijo_v1"))
    decoder = _Decoder()
    start = torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(start), temperature=1., threshold=.5)
    with pytest.raises(ValueError, match="fixed-mask"):
        run_arm("active_set", decoder, start, initial, ContextBank(cfg, SPEC), _Scorer(), guided,
                "c" * 64, _Run(tmp_path), Counters())
