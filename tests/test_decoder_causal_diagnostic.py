"""HPC-only engineering fixtures for declared decoder counterfactuals.

These test intervention scope, baseline parity, memory/alignment guards and
restoration. They do not validate a pretrained decoder or attribute source079's
actual numerical failure. No numerical fixture has been run locally.
"""

from copy import deepcopy
import json

import pytest
import torch

from latentmol.boltz.decoder_causal_diagnostic import (
    DecoderObserver, decoder_control, evaluate_decoder, fixed_readout,
    reference_from_plan, selected_case,
)
from latentmol.models.autoencoder import GraphAutoencoder, ModelConfig
from latentmol.models.refined_decoder import symmetric_pair_features
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import GraphSpec, RelaxedGraph


def _model(seed=17):
    spec = GraphSpec(max_nodes=3, atomic_numbers=(0, 6, 7), formal_charges=(-1, 0, 1), bond_names=("NONE", "SINGLE"))
    config = ModelConfig(hidden=8, latent=4, layers=1, decoder_variant="pair_refine",
                         global_hidden=8, pair_hidden=4, refinement_rounds=2, attention_heads=2)
    # Reproduce fixture weights without altering later fixtures' RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return GraphAutoencoder(spec, config).eval().requires_grad_(False)


def test_frozen_abs_preserves_baseline_values_and_derivative_including_diagonal():
    observer = DecoderObserver()
    h = torch.tensor([[[.2, -.7], [.5, .1]]], requires_grad=True)
    observer.begin("original_fp32", capture=True)
    original = observer.absolute_features(symmetric_pair_features, h)
    observer.backward = True
    original_gradient, = torch.autograd.grad(original.sum(), h)
    observer.backward = False
    observer.finish()
    observer.begin("frozen_abs_fp32")
    frozen = observer.absolute_features(symmetric_pair_features, h)
    frozen_gradient, = torch.autograd.grad(frozen.sum(), h)
    trace = observer.finish()
    assert torch.equal(frozen, original)
    assert torch.equal(frozen_gradient, original_gradient)
    assert trace["sign_crossings"] == 0
    assert observer.baselines[0]["signs"][0, 0, 0].count_nonzero() == 0


def test_frozen_abs_declares_changed_endpoint_after_sign_crossing():
    observer = DecoderObserver()
    baseline = torch.tensor([[[1.], [2.]]], requires_grad=True)
    observer.begin("original_fp32", capture=True)
    value = observer.absolute_features(symmetric_pair_features, baseline)
    observer.backward = True
    torch.autograd.grad(value[..., 1:2].sum(), baseline)
    observer.backward = False
    observer.finish()
    endpoint = torch.tensor([[[3.], [2.]]])
    observer.begin("frozen_abs_fp32")
    frozen = observer.absolute_features(symmetric_pair_features, endpoint)
    trace = observer.finish()
    assert frozen[0, 0, 1, 1] == -1
    assert symmetric_pair_features(endpoint)[0, 0, 1, 1] == 1
    assert trace["sign_crossings"] == 2
    assert trace["crossings_with_nonzero_baseline_output_cotangent"] == 2


def test_stable_occupancy_is_unconditioned_real_mass_not_renormalized():
    observer = DecoderObserver()
    observer.begin("stable_occupancy_fp32", capture=True)
    observer.current_atoms = torch.tensor([[[20., 0., 0.]]], requires_grad=True)
    probabilities = observer.current_atoms.softmax(-1)
    subtraction = 1 - probabilities[..., 0]
    stable = probabilities[..., 1:].sum(-1)
    effective = observer.refinement(0, lambda h, e, c, occupancy: occupancy, None, None, None, subtraction)
    assert subtraction.item() == 0
    assert stable.item() > 0
    assert torch.equal(effective, stable)
    assert effective.item() < 1e-6
    assert observer.occupancy_calls[0]["subtraction_clamped_count"] == 1
    assert observer.occupancy_calls[0]["stable_clamped_count"] == 1
    observer.abort()


@pytest.mark.parametrize("seed", [17, 23, 29])
def test_decoder_copy_original_baseline_matches_and_methods_restore_on_failure(seed):
    model = _model(seed)
    point = torch.tensor([[.3, -.2, .4, .1]])
    active = torch.tensor([True, True, False])
    reference_point = point.clone().requires_grad_()
    graph = graph_from_decoder(model.decode(reference_point), model.spec, fixed_active=active)
    fields = tuple(getattr(graph, key) for key in ("atoms", "charges", "bonds"))
    base = tuple(getattr(graph, key).detach() for key in ("atoms", "charges", "bonds"))
    # Constant categorical cotangents would only differentiate sum(P)=1.
    # Exercise every graph channel with deterministic nonconstant values.
    cotangents = tuple(torch.arange(value.numel(), dtype=value.dtype).reshape_as(value) / value.numel()
                       for value in base)
    reference_gradient, = torch.autograd.grad(fields, reference_point, grad_outputs=cotangents)
    observer = DecoderObserver()
    namespace = model.decoder.forward.__func__.__globals__
    original_helper = namespace["symmetric_pair_features"]
    original_attributes = [dict(model.decoder.__dict__)] + [dict(layer.__dict__) for layer in model.decoder.rounds]
    with pytest.raises(RuntimeError, match="fixture failure"):
        with decoder_control(model, observer):
            row, gradient = evaluate_decoder(model, point, active, 1., base, cotangents,
                observer, "original_fp32", gradients=True, capture=True)
            assert row["value"] == 0
            assert len(row["trace"]["occupancy_rounds"]) == 2
            assert len(observer.baselines) == 4  # Pair init, two rounds, final bond features.
            assert gradient.isfinite().all()
            assert torch.equal(gradient, reference_gradient)
            frozen_row, frozen_gradient = evaluate_decoder(model, point, active, 1., base, cotangents,
                observer, "frozen_abs_fp32", gradients=True)
            assert frozen_row["value"] == 0
            assert frozen_row["graph_sha256"] == row["graph_sha256"]
            assert torch.equal(frozen_gradient, reference_gradient)
            raise RuntimeError("fixture failure")
    assert namespace["symmetric_pair_features"] is original_helper
    for owner, expected in zip((model.decoder, *model.decoder.rounds), original_attributes):
        assert owner.__dict__.keys() == expected.keys()
        assert owner.__dict__.get("forward") is expected.get("forward")
        assert owner.__dict__.get("_node_predictions") is expected.get("_node_predictions")
    assert not observer.active and observer.current_atoms is None


def test_abs_capture_budget_and_invocation_alignment_fail_explicitly():
    h = torch.ones(1, 2, 2)
    observer = DecoderObserver(maximum_sign_elements=1)
    observer.begin("original_fp32", capture=True)
    with pytest.raises(ValueError, match="memory budget"):
        observer.absolute_features(symmetric_pair_features, h)
    observer.abort()
    observer = DecoderObserver()
    observer.begin("original_fp32", capture=True)
    observer.absolute_features(symmetric_pair_features, h)
    observer.finish()
    observer.begin("original_fp32")
    with pytest.raises(ValueError, match="alignment"):
        observer.absolute_features(symmetric_pair_features, torch.ones(1, 3, 2))
    observer.abort()


def test_float64_copy_and_widened_readout_are_explicit_without_original_mutation():
    model = _model()
    state = deepcopy(model.state_dict())
    wide = deepcopy(model).double()
    assert all(torch.equal(wide.state_dict()[name].to(value), value) for name, value in state.items())
    tensors = (torch.tensor([1., 2.]), torch.tensor([3.]), torch.tensor([4.]))
    graph = RelaxedGraph(*tensors)
    assert fixed_readout(graph, tensors, tuple(torch.ones_like(value) for value in tensors)).dtype == torch.float64
    assert fixed_readout(graph, tensors, tuple(torch.ones_like(value) for value in tensors)).item() == 0
    assert all(torch.equal(model.state_dict()[name], value) for name, value in state.items())


def test_prespecified_saved_case_cannot_change_after_outcomes():
    case = {"id": "prior-3", "step": 50, "arm": "fixed_initial_mask"}
    assert selected_case({"cases": [{"id": "prior-0"}, case]}) is case
    with pytest.raises(ValueError, match="prespecified"):
        selected_case({"cases": [{**case, "step": 40}]})


def test_completed_source081_reference_requires_pinned_proof_and_matching_identities(tmp_path):
    from latentmol.boltz.failed_gate_attribution import _digest, _sha
    folder = tmp_path / "source-081-attention-causal-attempt001-replay" / "prior-3"
    folder.mkdir(parents=True)
    attribution, run = folder / "attribution.json", folder / "run.json"
    identities = {key: "1" * 64 for key in ("point_sha256", "context_sha256", "decoder_state_sha256", "scorer_state_sha256")}
    proof = {"status": "completed", "diagnostic_completed": True, "native_seed_parity": True,
             "frozen_state_unchanged": True, "parent_unchanged": True,
             "identities_before": identities, "identities_after": identities,
             "controls": {"native": {"status": "completed", "gradients": {
                 "dbmol_contact_loss": {"sha256": "2" * 64, "norm": 1.}}}}}
    case = {"id": "prior-3", "step": 50, "arm": "fixed_initial_mask", "parent_run": "/original",
            "parent_artifacts_sha256": {}}
    plan = {"production_release": "/release", "production_manifest_sha256": "3" * 64,
            "production_archive_sha256": "4" * 64}
    source_files = {"code.py": "5" * 64}
    record = {"status": "completed", "config": {**plan, "case": deepcopy(case)},
              "source": {"files": source_files, "sha256": _digest(source_files)},
              "source_archive_sha256": "6" * 64}
    attribution.write_text(json.dumps(proof))
    run.write_text(json.dumps(record))
    case["parent_artifacts_sha256"] = {str(attribution): _sha(attribution), str(run): _sha(run)}
    assert reference_from_plan(plan, case)["native_contact_gradient"]["sha256"] == "2" * 64
    proof["identities_after"] = {"point_sha256": "7" * 64}
    attribution.write_text(json.dumps(proof))
    case["parent_artifacts_sha256"][str(attribution)] = _sha(attribution)
    with pytest.raises(ValueError, match="incomplete or has different"):
        reference_from_plan(plan, case)
