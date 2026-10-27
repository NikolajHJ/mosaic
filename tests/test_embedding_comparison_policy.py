"""Final-policy arithmetic/binding fixtures; numerical execution is HPC-only."""

import json

import pytest
import torch

from latentmol.boltz.embedding_comparison import FrozenComparisonObjective, run_matched_optimization
from latentmol.boltz.embedding_comparison_policy import (
    DEFAULT_POLICY, FINAL_FLOAT64_POLICY, comparison_graph, policy_binding, policy_record)
from latentmol.boltz import embedding_comparison_validation as validation
from latentmol.graph_policy import fixed_occupancy_graph
from latentmol.multimodel.study import tensor_digest
from latentmol.representation import DecoderOutput, GraphSpec
from test_embedding_comparison_validation import comparison_fixture


def logits_fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(140036)
    spec = GraphSpec(max_nodes=3, atomic_numbers=(0, 6, 7),
                     formal_charges=(-1, 0, 1), bond_names=("NONE", "SINGLE"))
    active = torch.tensor([True, False, True])
    logits = DecoderOutput(torch.randn(1, 3, 3, generator=generator, dtype=dtype),
        torch.randn(1, 3, 3, generator=generator, dtype=dtype),
        torch.randn(1, 3, 3, 2, generator=generator, dtype=dtype))
    return spec, active, logits


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_historical_default_preserves_original_values_and_reverse_gradient(dtype):
    spec, active, decoded = logits_fixture(dtype)
    inputs = [getattr(decoded, key).requires_grad_(True) for key in ("atoms", "charges", "bonds")]
    ordinary = fixed_occupancy_graph(decoded, active, spec, 1.)
    declared = comparison_graph(decoded, active, spec)
    coefficients = [torch.arange(value.numel(), dtype=dtype).reshape_as(value) for value in inputs]
    expected = sum((getattr(ordinary, key) * coefficient).sum() for key, coefficient in
                   zip(("atoms", "charges", "bonds"), coefficients))
    actual = sum((getattr(declared, key) * coefficient).sum() for key, coefficient in
                 zip(("atoms", "charges", "bonds"), coefficients))
    left = torch.autograd.grad(expected, inputs)
    right = torch.autograd.grad(actual, inputs)
    for key, a, b in zip(("atoms", "charges", "bonds"), left, right):
        assert torch.equal(getattr(ordinary, key), getattr(declared, key))
        assert torch.equal(a, b)


def test_promoted_policy_matches_allocated_diagnostic_control_and_native_input_dtype():
    from latentmol.boltz.embedding_gradient_diagnostic import graph_policy
    spec, active, decoded = logits_fixture()
    expected = graph_policy({key: getattr(decoded, key) for key in ("atoms", "charges", "bonds")},
                            active, spec, wide=True)
    actual = comparison_graph(decoded, active, spec, policy=FINAL_FLOAT64_POLICY)
    for key in expected:
        assert torch.equal(getattr(actual, key), expected[key])
        assert getattr(actual, key).dtype == decoded.atoms.dtype


def test_promoted_policy_retains_fixed_mask_neutral_padding_symmetry_and_none_diagonal():
    spec, active, decoded = logits_fixture()
    graph = comparison_graph(decoded, active, spec, policy=FINAL_FLOAT64_POLICY)
    assert torch.equal(graph.atoms[..., 0], (~active)[None].float())
    assert torch.equal(graph.charges[:, ~active], torch.tensor([[[0., 1., 0.]]]))
    assert torch.equal(graph.bonds, graph.bonds.transpose(1, 2))
    diagonal = graph.bonds[0].diagonal(dim1=0, dim2=1)
    assert torch.equal(diagonal[0], torch.ones_like(diagonal[0]))
    assert diagonal[1:].count_nonzero() == 0
    assert graph.bonds[:, ~active, :, 1:].count_nonzero() == 0


def test_saturated_softmax_control_repairs_local_vjp_jvp_cancellation_without_changing_dtype():
    """Large fixed cotangents expose FP32 softmax cancellation deterministically."""
    spec = GraphSpec(max_nodes=1, atomic_numbers=(0, 6, 7),
                     formal_charges=(0,), bond_names=("NONE", "SINGLE"))
    active = torch.tensor([True])
    point = torch.tensor([[[0., 20., 0.]]], dtype=torch.float32)
    direction = torch.tensor([[[0., 1., -1.]]], dtype=torch.float32)
    cotangent = torch.tensor([[[0., 1e6, 0.]]], dtype=torch.float32)

    def observe(policy):
        def function(atoms):
            decoded = DecoderOutput(atoms, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1, 2))
            return comparison_graph(decoded, active, spec, policy=policy).atoms
        probe = point.clone().requires_grad_(True)
        graph = function(probe)
        gradient = torch.autograd.grad(graph, probe, grad_outputs=cotangent)[0]
        reverse = float((gradient.double() * direction.double()).sum())
        with torch.no_grad(), torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(point, direction)
            primal, tangent = torch.autograd.forward_ad.unpack_dual(function(dual))
            assert torch.equal(primal, graph) and tangent.dtype == point.dtype
            forward = float((tangent.double() * cotangent.double()).sum())
        return abs(reverse - forward), graph.detach()

    historical_error, old = observe(DEFAULT_POLICY)
    promoted_error, new = observe(FINAL_FLOAT64_POLICY)
    assert old.dtype == new.dtype == point.dtype
    assert historical_error > 1e-4
    assert promoted_error < 1e-7
    assert promoted_error < historical_error * .001


def test_policy_binding_is_explicit_stable_distinct_and_rejects_silent_fallbacks():
    assert policy_record()["policy"] == DEFAULT_POLICY
    assert policy_binding() == policy_binding(DEFAULT_POLICY)
    assert policy_binding(DEFAULT_POLICY) != policy_binding(FINAL_FLOAT64_POLICY)
    assert len(policy_binding(FINAL_FLOAT64_POLICY)) == 64
    spec, active, decoded = logits_fixture()
    with pytest.raises(ValueError, match="Unknown"):
        comparison_graph(decoded, active, spec, policy="unrecognized")
    with pytest.raises(ValueError, match="temperature"):
        comparison_graph(decoded, active, spec, policy=FINAL_FLOAT64_POLICY, temperature=2.)


def test_injected_mapping_and_decoded_hard_labels_are_unchanged_by_final_graph_precision():
    prepared, scorer, bridge, vae, z = comparison_fixture()
    old = FrozenComparisonObjective(scorer, prepared, bridge, vae)
    new = FrozenComparisonObjective(scorer, prepared, bridge, vae,
                                    comparison_graph_policy=FINAL_FLOAT64_POLICY)
    before = old.evaluate("learned", z, require_gradient=True)
    after = new.evaluate("learned", z, require_gradient=True)
    assert before["scores"] == after["scores"]
    assert torch.equal(before["gradient"], after["gradient"])
    old_audit, new_audit = old.audit(z), new.audit(z)
    for group in ("conditioned", "natural"):
        for key in ("atoms", "charges", "bonds"):
            assert torch.equal(old_audit["tensors"][group][key], new_audit["tensors"][group][key])
    assert old_audit["record"]["scores"]["hard_fixed_context"] == new_audit["record"]["scores"]["hard_fixed_context"]
    assert new_audit["record"]["comparison_graph_policy_sha256"] == policy_binding(FINAL_FLOAT64_POLICY)
    assert new.assert_unchanged(full=True)["frozen_state_unchanged"]


@pytest.mark.parametrize("scoring_path", ["relaxed", "learned"])
def test_fresh_gate_production_objective_and_audit_bind_the_same_declared_policy(tmp_path, scoring_path):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    report = validation.learned_start_gate(scorer, prepared, bridge, vae, z,
        tmp_path / (scoring_path + ".json"), scoring_path=scoring_path,
        comparison_graph_policy=FINAL_FLOAT64_POLICY)
    objective = FrozenComparisonObjective(scorer, prepared, bridge, vae,
                                          comparison_graph_policy=FINAL_FLOAT64_POLICY)
    measured = objective.evaluate(scoring_path, z, require_gradient=True)
    assert report["passed"] and report["graph_policy_binding_required"]
    assert measured["scores"] == report["baseline_scores"]
    assert tensor_digest({"gradient": measured["gradient"]}) == report["production_gradient_sha256"]
    assert report["comparison_graph_policy"] == measured["comparison_graph_policy"]
    assert report["comparison_graph_policy_sha256"] == objective.comparison_graph_policy_sha256
    assert objective.audit(z)["record"]["comparison_graph_policy_sha256"] == report["comparison_graph_policy_sha256"]


def test_policy_mutation_is_detected_before_the_next_native_forward():
    prepared, scorer, bridge, vae, z = comparison_fixture()
    objective = FrozenComparisonObjective(scorer, prepared, bridge, vae,
                                          comparison_graph_policy=FINAL_FLOAT64_POLICY)
    objective.comparison_graph_policy = DEFAULT_POLICY
    with pytest.raises(ValueError, match="policy changed"):
        objective.evaluate("relaxed", z)
    assert objective.calls["boltz_forward_attempted"] == 0


def test_gate_rejects_mismatched_production_policy_metadata(tmp_path, monkeypatch):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    original = validation.score_relaxed_path
    def mismatched(*args, **kwargs):
        result = original(*args, **kwargs)
        result["comparison_graph_policy_sha256"] = policy_binding(DEFAULT_POLICY)
        return result
    monkeypatch.setattr(validation, "score_relaxed_path", mismatched)
    path = tmp_path / "mismatched-policy.json"
    with pytest.raises(ValueError, match="declared graph arithmetic policy"):
        validation.learned_start_gate(scorer, prepared, bridge, vae, z, path,
            scoring_path="relaxed", comparison_graph_policy=FINAL_FLOAT64_POLICY)
    assert not json.loads(path.read_text())["passed"]


def test_optimizer_rejects_audit_using_different_policy_before_applying_any_update(tmp_path):
    from test_embedding_comparison import ToyObjective
    class WrongAudit(ToyObjective):
        comparison_graph_policy_record = policy_record(FINAL_FLOAT64_POLICY)
        comparison_graph_policy_sha256 = policy_binding(FINAL_FLOAT64_POLICY)
    with pytest.raises(ValueError, match="Checkpoint audit changed"):
        run_matched_optimization({"start": torch.zeros(1, 2)}, WrongAudit(), tmp_path / "run",
            steps=1, learning_rate=1., max_gradient_norm=1., trust_radius=3.)
    report = json.loads((tmp_path / "run" / "comparison.json").read_text())
    assert report["status"] == "failed_not_passed" and report["runs"][0]["applied"] == 0
