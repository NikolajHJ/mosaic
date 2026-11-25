"""Allocated-HPC fixtures for the pointwise certificate, without Boltz weights.

The tiny frozen scorer is an engineering fixture, not binding evidence. It uses
the production shared-graph adapter and DBMol arithmetic, so forward duals must
traverse the real decoder/graph/branch route and stopped-target NLL unchanged.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz import adapt_graph, pointwise_ad as validation
from latentmol.boltz.active_set import score_at
from latentmol.boltz.additive_contract import profile_record
from latentmol.boltz.dbmol_objective import (
    COMPONENT_OUTPUTS, CONFIDENCE_WEIGHTS, CONTACT_HEAD64_ARITHMETIC,
    DBMolObjectiveConfig, reduce_objective,
)
from latentmol.boltz.pointwise_contract import verify_gate
from latentmol.boltz.provenance import context_digest
from latentmol.multimodel.config import Objective
from latentmol.multimodel.engine import Counters
from latentmol.multimodel.study import tensor_digest
from latentmol.repair_guidance.loss import FrozenRepairTarget
from latentmol.representation import DecoderOutput
from test_boltz_adapter import prepared_fixture


class TinyDecoder(torch.nn.Module):
    """Smooth, parameter-free decoder with distinct category dependence on z."""
    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def decode(self, z):
        spec = self.spec
        def logits(shape, offset):
            count = 1
            for size in shape: count *= size
            rows = torch.arange(count * z.numel(), device=z.device, dtype=z.dtype).reshape(count, z.numel())
            weights = ((rows + offset) % 13 - 6) / 17
            return (weights @ z.flatten()).reshape(shape)
        return DecoderOutput(logits((1, spec.max_nodes, len(spec.atomic_numbers)), 1),
            logits((1, spec.max_nodes, len(spec.formal_charges)), 4),
            logits((1, spec.max_nodes, spec.max_nodes, len(spec.bond_names)), 7))


class TinyScorer(torch.nn.Module):
    """Actual mixed-precision native operation tree; no learned parameters."""
    def __init__(self, dbmol, flat=None):
        super().__init__()
        self.config = dbmol
        self.flat = flat

    def forward(self, features, typed, coordinates):
        x = (features["ref_element"] * torch.arange(128., device=typed.device) / 100).sum()
        x = x + features["ref_charge"].sum() / 7 + features["token_bonds"].sum() / 11
        x = x + (typed * torch.arange(7., device=typed.device)).sum() / 13
        names = ("affinity_pred_value", "affinity_logits_binary", "dbmol_contact_pocket_loss",
                 "dbmol_contact_anchor_loss", *CONFIDENCE_WEIGHTS)
        outputs = {name: (index + 1.) * (x.double() if "contact" in name else x)
                   for index, name in enumerate(names)}
        if self.flat:
            selected = names if self.flat == "total" else ("affinity_pred_value", "affinity_logits_binary")
            for name in selected:
                outputs[name] = outputs[name] * 0
        outputs["dbmol_contact_loss"] = (outputs["dbmol_contact_pocket_loss"] + outputs["dbmol_contact_anchor_loss"]) / 2
        return reduce_objective(outputs, self.config)


class TinyBranch(torch.nn.Module):
    def __init__(self, prepared, scorer):
        super().__init__()
        self.prepared, self.scorer = prepared, scorer
        self.context = context_digest(prepared)
        self.manifest = SimpleNamespace(context_sha256=self.context)

    def verify_context_unchanged(self):
        if context_digest(self.prepared) != self.context:
            raise ValueError("Fixture context mutated")
        return self.context

    def forward(self, graph):
        features, typed = adapt_graph(graph, self.prepared)
        return self.scorer(features, typed, self.prepared.coordinates)


def fixture(repair=False, *, flat=None):
    prepared = prepared_fixture()
    dbmol = DBMolObjectiveConfig(confidence_checkpoint={"path": "fixture", "sha256": "c" * 64},
        pocket_residues=({"asym_id": 0, "residue_index": 0},),
        objective_arithmetic=CONTACT_HEAD64_ARITHMETIC)
    config = SimpleNamespace(dbmol=dbmol, fd_policy=validation.POLICY, recycling_steps=0,
        objective=Objective(output="dbmol_total_loss"), temperature=1.,
        diagnostics=SimpleNamespace(seed=17, fd_rtol=.05, fd_atol=1e-6, minimum_fd_passes=2,
                                    parity_rtol=1e-4, parity_atol=1e-5),
        repair_guidance=SimpleNamespace(weight=.01 if repair else 0., weight_schedule=None,
                                       normalization="sum"),
        element_diagnostics=False, step_control=None)
    decoder = TinyDecoder(prepared.spec).eval().requires_grad_(False)
    scorer = TinyScorer(dbmol, flat=flat).eval().requires_grad_(False)
    branch = TinyBranch(prepared, scorer).eval().requires_grad_(False)
    point = torch.tensor([[.2, -.1, .3, -.2, .1, .15, -.05, .25]])
    target = (FrozenRepairTarget.create(prepared.seed, prepared.active,
        refresh_step=15, receipt_path="fixture-target.json") if repair else None)
    counters, components = Counters(), {}
    weight = .01 if repair else 0.
    measured, gradient = score_at(decoder, point, branch, config, counters, "fixture_measurement", gradients=True,
        component_gradients=components, repair_target=target, repair_weight=weight)
    report = {"passed": False, "status": "running", "native_seed_parity": True,
        "point_sha256": tensor_digest({"z": point}), "context_sha256": branch.context,
        "measured": measured}
    return SimpleNamespace(decoder=decoder, point=point, branch=branch, config=config,
        counters=counters, report=report, measured=measured, gradient=gradient, components=components,
        target=target, weight=weight)


def run(case):
    validation.validate_branch_derivatives(case.decoder, case.point, case.branch, case.config,
        case.counters, case.report, lambda: None, case.measured, case.gradient, case.components,
        target=case.target, weight=case.weight)
    case.report.update(passed=True, status="passed")
    return case.report


@pytest.mark.parametrize("repair", [False, True])
def test_native_pointwise_certificate_actual_shared_graph_and_stopped_target(repair):
    case = fixture(repair)
    report = run(case)
    profile = profile_record(case.config.dbmol, case.config.objective,
                            repair_weight=case.weight if repair else None)
    # Persisted receipts use sorted JSON in real workers. Direction dictionary
    # insertion order must not become an accidental authentication requirement.
    persisted = json.loads(json.dumps(report, sort_keys=True))
    assert verify_gate(persisted, require_repair=repair, expected_profile=profile)
    assert report["derivative_contract"]["full_network_fd_required"] is False
    assert report["derivative_contract"]["historical_gate_approval"] is False
    assert report["derivative_contract"]["branch_holding_used_in_optimization"] is False
    assert report["actual_pointwise_full_forwards"] <= 32
    assert report["native_grouped_baseline_parity"]["passed"]
    assert report["native_input_bridge_audit"]["passed"]
    assert len(report["natural_secant_diagnostics"]["rows"]) == 4
    assert all(row["required_for_pass"] is False for row in report["natural_secant_diagnostics"]["rows"])
    assert report["pointwise_state_unchanged"] == {"context": True, "target": True, "frozen_versions": True}
    assert not case.branch._forward_pre_hooks
    assert all(parameter.grad is None for parameter in case.decoder.parameters())
    if repair:
        assert report["repair_only_validation"]["passed"]
        assert report["derivative_contract"]["target_sha256"] == case.target.identity
        assert not any(value.requires_grad for value in (case.target.graph.atoms, case.target.graph.charges, case.target.graph.bonds))


def test_supplied_gradient_corruption_rejected_before_native_ad_approval():
    case = fixture()
    case.gradient = case.gradient + .1
    with pytest.raises(ValueError, match="parity"):
        run(case)
    assert case.report["native_grouped_baseline_parity"]["supplied_gradient_exact"]["combined"] is False
    assert case.report["derivative_contract"]["passed"] is False


@pytest.mark.parametrize("flat", ["component", "total"])
def test_flat_attached_derivatives_are_checked_and_reader_accepts_the_new_contract(flat):
    case = fixture(flat=flat)
    report = run(case)
    profile = profile_record(case.config.dbmol, case.config.objective)
    assert verify_gate(report, require_repair=False, expected_profile=profile)
    item = report["component_gradients"]["dbmol_affinity_loss"]
    assert item["gradient_norm"] == 0 and item["status"] == "validated_low_signal"
    assert item["direction_selection"]["gradient_label_is_random_probe"]
    assert all(row["reverse_projection"] == 0 and row["passed"]
               for row in item["independent_forward_ad"]["rows"])
    if flat == "total":
        assert report["local_fd_validation"]["status"] == "validated_low_signal"
    # Receipt authentication still binds every zero derivative to its own AD
    # rows and original probes; a fabricated flat-gradient claim cannot pass.
    item["independent_forward_ad"]["rows"][0]["reverse_projection"] = .1
    assert not verify_gate(report, require_repair=False, expected_profile=profile)


def test_explicit_legacy_pointwise_policy_keeps_nonzero_qualification():
    case = fixture()
    case.config.fd_policy = validation.LEGACY_POLICY
    report = run(case)
    assert report["derivative_contract"]["version"] == validation.LEGACY_POLICY
    assert verify_gate(report, require_repair=False,
                       expected_profile=profile_record(case.config.dbmol, case.config.objective))
    case = fixture(flat="component")
    case.config.fd_policy = validation.LEGACY_POLICY
    with pytest.raises(ValueError, match="Pointwise native AD gate failed"):
        run(case)


def test_missing_independent_tangent_is_not_a_finite_difference_waiver(monkeypatch):
    case = fixture()
    original = validation._native_scalars
    def missing(*args, **kwargs):
        outputs = original(*args, **kwargs)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(outputs["affinity_pred_value"])
        if tangent is not None:
            outputs["affinity_pred_value"] = primal
        return outputs
    monkeypatch.setattr(validation, "_native_scalars", missing)
    with pytest.raises(ValueError, match="Pointwise native AD gate failed"):
        run(case)
    assert set(case.report["component_gradients"]) == set(COMPONENT_OUTPUTS)
    assert all(not item["passed"] for item in case.report["component_gradients"].values())
    assert not case.branch._forward_pre_hooks


def test_unexpected_secant_execution_error_fails_closed(monkeypatch):
    case = fixture()
    original = case.counters.call
    def broken(model, purpose, action, function, device):
        if purpose == "pointwise_natural_secant_diagnostic":
            raise RuntimeError("fixture endpoint execution error")
        return original(model, purpose, action, function, device)
    monkeypatch.setattr(case.counters, "call", broken)
    with pytest.raises(RuntimeError, match="endpoint execution error"):
        run(case)
    assert case.report["derivative_contract"]["passed"] is False
    assert not case.branch._forward_pre_hooks


def test_failed_affine_bridge_is_mandatory(monkeypatch):
    case = fixture()
    monkeypatch.setattr(validation, "validate_bridge", lambda *args, **kwargs: {"passed": False})
    with pytest.raises(ValueError, match="bridge audit failed"):
        run(case)


@pytest.mark.parametrize("mutation", ["training", "parameter", "policy", "recycles", "unbound_weight"])
def test_unsupported_state_contract_rejected(mutation):
    case = fixture()
    if mutation == "training": case.decoder.train()
    elif mutation == "parameter": case.branch.scorer.register_parameter("bad", torch.nn.Parameter(torch.ones(1)))
    elif mutation == "policy": case.config.fd_policy = "local_v1"
    elif mutation == "recycles": case.config.recycling_steps = 1
    else: case.weight = .01
    with pytest.raises(ValueError):
        run(case)
    assert "derivative_contract" not in case.report


def test_target_mutation_and_stale_profile_are_rejected():
    case = fixture(True)
    case.target.graph.charges[0, 0] = 1
    with pytest.raises(ValueError, match="target identity"):
        run(case)
    case = fixture()
    report = run(case)
    profile = profile_record(case.config.dbmol, case.config.objective)
    changed = deepcopy(profile)
    changed["dbmol"]["contact_weight"] = 1.
    assert not verify_gate(report, require_repair=False, expected_profile=changed)
    assert not verify_gate(report, require_repair=False)


def test_reader_retains_finite_secant_disagreement_without_calling_it_ad_failure():
    case = fixture()
    report = run(case)
    # Receipt-only adversarial fixture: a large observed finite chord remains
    # visible but is deliberately not a pointwise derivative acceptance test.
    row = report["natural_secant_diagnostics"]["rows"][0]
    row["f_plus"]["combined"] += 100.
    row["secants"]["combined"] = (row["f_plus"]["combined"] - row["f_minus"]["combined"]) / (2 * row["epsilon"])
    assert abs(row["secants"]["combined"] - row["tangents"]["combined"]) > 1.
    assert verify_gate(report, require_repair=False,
        expected_profile=profile_record(case.config.dbmol, case.config.objective))
