"""Complete-cut diagnostic engineering fixtures; run only on allocated HPC."""

import json
import time

import pytest
import torch

from latentmol.boltz.embedding_gradient_diagnostic import (
    _comparison, _dot, _equal_nested, _replay, _vector_comparison, authenticated_case,
    graph_policy, independent_logits, native_channels)
from latentmol.experiments import file_sha256
from latentmol.representation import DecoderOutput
from test_embedding_validation import frozen_models


def test_complete_sibling_duality_keeps_all_paths_and_tensor_evidence():
    point = torch.tensor([[.2, -.7]], dtype=torch.float64)
    direction = torch.tensor([[.3, .4]], dtype=torch.float64)
    cotangent = {"linear": torch.tensor([[1., -2.]], dtype=torch.float64),
                 "nonlinear": torch.tensor([[3., 4.]], dtype=torch.float64)}
    function = lambda values: dict(linear=2 * values["z"], nonlinear=values["z"].square())
    row, tensors = _replay(function, {"z": point}, {"z": direction}, cotangent)
    expected = 2 * cotangent["linear"] + 2 * point * cotangent["nonlinear"]
    torch.testing.assert_close(tensors["input_vjp"]["z"], expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors["output_jvp"]["nonlinear"], 2 * point * direction, rtol=0, atol=0)
    assert row["absolute_error"] < 1e-14 and row["primal_exact_between_ad_modes"]
    assert not row["threshold_used_for_promotion"]
    assert all(not value.requires_grad for value in tensors["input_vjp"].values())


def test_dropped_sibling_cannot_silently_enter_cut_conservation():
    point = torch.ones(1, 2)
    with pytest.raises(ValueError, match="sibling"):
        _replay(lambda values: {"one": values["z"]}, {"z": point}, {"z": point},
                {"one": point, "two": point})
    with pytest.raises(ValueError, match="complete"):
        _dot({"one": point}, {"two": point})


def test_feedback_head_outputs_are_cloned_before_extracting_cut_cotangents():
    z = torch.tensor([[.5]], dtype=torch.float64, requires_grad=True)
    atom = 2 * z
    charge = atom.square()
    bond = charge * atom
    boundary = independent_logits(DecoderOutput(atom, charge, bond))
    score = 3 * boundary["atoms"].sum() + 5 * boundary["charges"].sum() + 7 * boundary["bonds"].sum()
    expected = torch.autograd.grad(score, z, retain_graph=True)[0]
    coefficients = dict(zip(boundary, torch.autograd.grad(score, tuple(boundary.values())), strict=True))
    assert [value.item() for value in coefficients.values()] == [3., 5., 7.]
    function = lambda values: dict(atoms=2 * values["z"], charges=(2 * values["z"]).square(),
                                  bonds=(2 * values["z"]).pow(3))
    _, evidence = _replay(function, {"z": z.detach()}, {"z": torch.ones_like(z)}, coefficients)
    torch.testing.assert_close(evidence["input_vjp"]["z"], expected, rtol=0, atol=0)


def test_duality_diagnostic_retains_failure_without_relabelling_pass():
    result = _comparison(.01, .02)
    assert not result["within_inherited_threshold"]
    assert result["absolute_error"] == .01
    assert not result["threshold_used_for_promotion"]
    with pytest.raises(ValueError, match="Nonfinite"):
        _comparison(float("nan"), 0.)


def test_replay_shared_deadline_is_not_ignored():
    point = torch.ones(1, 2)
    with pytest.raises(TimeoutError):
        _replay(lambda values: {"one": values["z"]}, {"z": point}, {"z": point},
                {"one": point}, deadline=time.monotonic() - 1)


def test_policy_cut_matches_original_and_wide_control_stays_explicit():
    prepared, _, _, vae = frozen_models()
    z = torch.randn(1, vae.config.latent)
    decoded = vae.decode(z)
    logits = {key: getattr(decoded, key) for key in ("atoms", "charges", "bonds")}
    original = graph_policy(logits, prepared.active, prepared.spec)
    wide = graph_policy(logits, prepared.active, prepared.spec, wide=True)
    from latentmol.graph_policy import fixed_occupancy_graph
    reference = fixed_occupancy_graph(decoded, prepared.active, prepared.spec)
    for key in original:
        assert torch.equal(original[key], getattr(reference, key))
        assert original[key].dtype == wide[key].dtype == torch.float32
        assert bool(wide[key].isfinite().all())
    assert torch.equal(original["atoms"][..., 0], (~prepared.active)[None].float())


def test_native_chemistry_cut_reconstructs_every_changed_adapter_channel():
    prepared, _, _, _ = frozen_models()
    hard = prepared.seed.to_relaxed(prepared.spec)
    graph = {key: getattr(hard, key) for key in ("atoms", "charges", "bonds")}
    from latentmol.boltz.adapter import adapt_graph
    features, bonds = adapt_graph(hard, prepared)
    cut = native_channels(graph, prepared)
    assert set(cut) == {"ref_element", "ref_charge", "token_bonds", "bond_types"}
    assert all(torch.equal(cut[key], features[key]) for key in ("ref_element", "ref_charge", "token_bonds"))
    assert torch.equal(cut["bond_types"], bonds)


def test_authenticated_starts_require_all_three_parent_records(tmp_path):
    cases = [dict(record=dict(case_id=f"case-{i:03d}"), z=torch.full((1, 2), float(i)),
                  active=torch.tensor([True, False])) for i in (1, 2, 3)]
    parent = dict(cases=cases, graph_spec=dict(version=1))
    torch.save(parent, tmp_path / "parent.pt")
    torch.save(parent, tmp_path / "starts.pt")
    config = dict(diagnostic_parent_starts=dict(path=str(tmp_path / "parent.pt"),
                  sha256=file_sha256(tmp_path / "parent.pt")))
    chosen = authenticated_case(config, tmp_path)
    assert chosen["record"]["case_id"] == "case-003"
    changed = dict(cases=[dict(row) for row in cases], graph_spec=dict(version=1))
    changed["cases"][0]["z"] = torch.zeros(1, 2)
    torch.save(changed, tmp_path / "starts.pt")
    with pytest.raises(ValueError, match="all three"):
        authenticated_case(config, tmp_path)


def test_nested_identity_comparison_checks_tensor_values_and_keys():
    point = torch.tensor([1., 2.])
    assert _equal_nested({"x": [point, "fixed"]}, {"x": [point.clone(), "fixed"]})
    assert not _equal_nested({"x": point}, {"x": point + 1})
    assert not _equal_nested({"x": point}, {"y": point})


def test_input_vjp_evidence_compares_complete_vectors_not_only_one_projection():
    actual = {"a": torch.tensor([1., 2.]), "b": torch.tensor([3.])}
    expected = {"a": torch.tensor([1., 1.]), "b": torch.tensor([3.])}
    row = _vector_comparison(actual, expected)
    assert row["complete"] and not row["bitwise_exact"]
    assert row["difference_l2"] == row["maximum_absolute_difference"] == 1.
    assert row["rows"]["b"]["bitwise_exact"]
    with pytest.raises(ValueError, match="every captured"):
        _vector_comparison({"a": actual["a"]}, expected)
