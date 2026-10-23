"""Allocated-only integration fixtures; not evidence of native Boltz readiness."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentmol.boltz import additive_branch_fd as candidate
from latentmol.boltz.additive_fd import COMPONENTS, EPSILONS, primitive_weights
from latentmol.boltz.dbmol_objective import COMPONENT_OUTPUTS, CONTACT_HEAD64_ARITHMETIC, reduce_objective
from latentmol.multimodel.config import Objective
from latentmol.multimodel.engine import Counters, scalar_loss


def fixture(monkeypatch, *, repair=False):
    """Pinned readout topology plus ideal FP64 linear scalar primitives.

    The readout invocations test scoped hook coverage only; simple expressions
    make all directional/secant checks predictable without pretrained models.
    This is an engineering fixture, not a simulated molecular experiment.
    """
    def head():
        result = nn.Module()
        for name in ("affinity_out_mlp", "to_affinity_pred_value", "to_affinity_pred_score"):
            layers = [nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU()]
            if name != "affinity_out_mlp": layers.append(nn.Linear(4, 4))
            setattr(result, name, nn.Sequential(*layers))
        return result.eval().requires_grad_(False)
    heads = (head(), head())
    scorer = SimpleNamespace(affinity=SimpleNamespace(model=SimpleNamespace(affinity_ensemble=True,
        affinity_module1=SimpleNamespace(affinity_heads=heads[0]),
        affinity_module2=SimpleNamespace(affinity_heads=heads[1]))))
    dbmol = SimpleNamespace(profile="dbmol_fixed_pose_v1", objective_arithmetic=CONTACT_HEAD64_ARITHMETIC,
        affinity_regression_weight=1., affinity_logit_weight=1., contact_weight=2.5, confidence_weight=2.5,
        confidence_weights={"iptm": 1., "complex_iplddt": .3, "complex_ipde": .3,
                            "complex_pde": .1, "confidence_score": .3, "ptm": .1})
    config = SimpleNamespace(dbmol=dbmol, fd_policy=candidate.POLICY, recycling_steps=0,
        objective=Objective(output="dbmol_total_loss"), temperature=1.,
        diagnostics=SimpleNamespace(seed=17, fd_rtol=.05, fd_atol=1e-6, minimum_fd_passes=2),
        repair_guidance=SimpleNamespace(normalization="sum") if repair else None)
    branch = SimpleNamespace(scorer=scorer)
    def native(decoder, point, branch, config, *, target=None, weight=0.):
        for one in heads:
            hidden = one.affinity_out_mlp(point + 2.)
            one.to_affinity_pred_value(hidden)
            one.to_affinity_pred_score(hidden)
        x = point.double().sum()
        primitive_names = primitive_weights(config.dbmol, "combined")
        values = {name: (index + 1.) * (x + 1.) for index, name in enumerate(primitive_names)}
        values["dbmol_contact_loss"] = (values["dbmol_contact_pocket_loss"] + values["dbmol_contact_anchor_loss"]) / 2
        outputs = reduce_objective(values, config.dbmol)
        total = scalar_loss(outputs, config.objective)[0]
        if target is not None:
            outputs["repair_nll"] = point.sum() + 1.  # Actual native repair product remains FP32.
            total = total + weight * outputs["repair_nll"]
        return {**outputs, "combined": total}
    monkeypatch.setattr(candidate, "_native_scalars", native)
    point = torch.zeros(1, 4)
    decoder, target, weight = object(), object() if repair else None, .01 if repair else 0.
    probe = point.clone().requires_grad_(True)
    outputs = native(decoder, probe, branch, config, target=target, weight=weight)
    components = {key: torch.autograd.grad(outputs[key], probe, retain_graph=True)[0].detach()
                  for key in COMPONENT_OUTPUTS}
    gradient = torch.autograd.grad(outputs["combined"], probe)[0].detach()
    measured = {"loss": float(outputs["combined"].detach()),
                "components": {key: float(value.detach()) for key, value in outputs.items()
                               if key not in ("combined", "repair_nll")}}
    return decoder, point, branch, config, measured, gradient, components, target, weight


def run_case(values, report):
    decoder, point, branch, config, measured, gradient, components, target, weight = values
    candidate.validate_branch_derivatives(decoder, point, branch, config, Counters(), report, lambda: None,
        measured, gradient, components, target=target, weight=weight)


def reader_gate(values, report):
    """Attach the independently supplied wrapper/configuration evidence."""
    from copy import deepcopy
    from latentmol.boltz.additive_contract import profile_record
    measured = deepcopy(values[4])
    if values[7] is not None:
        measured["repair_guidance"] = {"status": "active", "weight": values[8],
            "normalized_nll": report["repair_only_validation"]["baseline_loss"]}
    report.update(passed=True, status="passed", native_seed_parity=True, measured=measured)
    return profile_record(values[3].dbmol, values[3].objective,
                          repair_weight=values[8] if values[7] is not None else None)


@pytest.mark.parametrize("repair", [False, True])
def test_saved_gate_reader_recomputes_certificates_and_rejects_stale_raw_inputs(monkeypatch, repair):
    from latentmol.repair_guidance.integration_smoke import full_gate_passed
    values, report = fixture(monkeypatch, repair=repair), {}
    run_case(values, report)
    # The production active-set wrapper owns these three outer state receipts.
    expected = reader_gate(values, report)
    assert full_gate_passed(report, require_repair=repair, expected_profile=expected)
    raw = report["component_gradients"]["dbmol_contact_loss"]["additive_inputs"]["natural"][0]
    primitive = next(iter(raw["primitives"].values()))
    primitive["analytic"] += 10.
    assert not full_gate_passed(report, require_repair=repair, expected_profile=expected)


@pytest.mark.parametrize("repair", [False, True])
def test_complete_candidate_preserves_original_secants_and_both_native_ad_proofs(monkeypatch, repair):
    values, report = fixture(monkeypatch, repair=repair), {}
    run_case(values, report)
    assert report["derivative_contract"]["passed"]
    assert report["derivative_contract"]["version"] == "additive_branch_v3"
    assert report["derivative_contract"]["branch_holding_used_in_optimization"] is False
    assert report["local_branch_baseline_parity"]["passed"]
    items = {**report["component_gradients"], "combined": report["local_fd_validation"]}
    assert set(items) == set(COMPONENTS)
    for key, item in items.items():
        assert item["passed"] and item["status"] == "validated_nonzero"
        assert set(item["original_net_fd"]) == {"natural", "held"}
        for mode in ("natural", "held"):
            assert len(item["additive_inputs"][mode]) == 4 * len(EPSILONS)
            assert len(item["original_net_fd"][mode]["finite_differences"]) == 4 * len(EPSILONS)
            assert item["additive_certificates"][mode]["passed"]
            assert all(row["recomposition"]["passed"] for row in item["additive_inputs"][mode])
        assert len(item["independent_forward_ad"]["rows"]) == 4
    if repair:
        assert report["repair_only_validation"]["passed"]
        assert report["repair_only_validation"]["repair_derivative_contract"] == "unchanged_branch_local_v2"
    for model in (values[2].scorer.affinity.model.affinity_module1.affinity_heads,
                  values[2].scorer.affinity.model.affinity_module2.affinity_heads):
        assert all(not module._forward_hooks for module in model.modules())


def test_changed_supplied_gradient_is_fatal_before_endpoint_approval(monkeypatch):
    values = list(fixture(monkeypatch))
    values[5] = values[5] + 1.
    report = {}
    with pytest.raises(ValueError, match="baseline parity failed"):
        run_case(values, report)
    assert not report["local_branch_baseline_parity"]["passed"]
    assert not report["derivative_contract"]["passed"]


def test_native_forward_tangent_failure_attempts_all_components_and_cannot_pass(monkeypatch):
    values = fixture(monkeypatch)
    original = candidate._native_scalars
    def missing_tangent(*args, **kwargs):
        outputs = original(*args, **kwargs)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(args[1])
        if tangent is not None:
            # Retain baseline primal but deliberately remove one native tangent.
            outputs["affinity_pred_value"] = torch.autograd.forward_ad.unpack_dual(outputs["affinity_pred_value"]).primal
        return outputs
    monkeypatch.setattr(candidate, "_native_scalars", missing_tangent)
    report = {}
    with pytest.raises(ValueError, match="derivative gate failed"):
        run_case(values, report)
    assert set(report["component_gradients"]) == set(COMPONENT_OUTPUTS)
    assert all(item["passed"] is False for item in report["component_gradients"].values())
    assert report["local_fd_validation"]["passed"] is False
    assert report["derivative_contract"]["passed"] is False


def test_natural_contact_subterm_failure_cannot_be_hidden_by_held_success(monkeypatch):
    values = fixture(monkeypatch)
    original = candidate._native_scalars
    def changed_natural_contact(decoder, point, branch, config, **kwargs):
        outputs = original(decoder, point, branch, config, **kwargs)
        primal, _ = torch.autograd.forward_ad.unpack_dual(point)
        relu = branch.scorer.affinity.model.affinity_module1.affinity_heads.affinity_out_mlp[1]
        # Deliberately inconsistent natural endpoint behavior leaves the exact
        # baseline/native AD unchanged but corrupts contact secants. Held
        # probes remain perfect. This tests the natural requirement against
        # actual bad endpoint rows, not a manually falsified aggregate flag.
        if not relu._forward_hooks and bool(primal.any()):
            outputs["dbmol_contact_pocket_loss"] = outputs["dbmol_contact_pocket_loss"] + point.double().sum()
            outputs["dbmol_contact_loss"] = (outputs["dbmol_contact_pocket_loss"] + outputs["dbmol_contact_anchor_loss"]) / 2
            outputs = reduce_objective(outputs, config.dbmol)
            outputs["combined"] = scalar_loss(outputs, config.objective)[0]
        return outputs
    monkeypatch.setattr(candidate, "_native_scalars", changed_natural_contact)
    report = {}
    with pytest.raises(ValueError, match="dbmol_contact_loss"):
        run_case(values, report)
    item = report["component_gradients"]["dbmol_contact_loss"]
    assert item["additive_certificates"]["held"]["passed"]
    assert not item["additive_certificates"]["natural"]["passed"]
    assert not item["passed"]
    assert report["local_fd_validation"]["passed"]  # Later components still ran.


@pytest.mark.parametrize("mutation", ["profile", "policy", "missing_component", "targetless_weight"])
def test_unsupported_contracts_fail_before_a_native_probe(monkeypatch, mutation):
    values = list(fixture(monkeypatch))
    if mutation == "profile": values[3].dbmol.objective_arithmetic = "native_v1"
    if mutation == "policy": values[3].fd_policy = "branch_local_v2"
    if mutation == "missing_component": values[6].pop(COMPONENT_OUTPUTS[0])
    if mutation == "targetless_weight": values[8] = .1
    monkeypatch.setattr(candidate, "_native_scalars", lambda *a, **k: pytest.fail("Unsupported mode executed native model"))
    with pytest.raises(ValueError): run_case(values, {})
