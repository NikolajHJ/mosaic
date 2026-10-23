"""HPC-only fixtures for loss signs, masks, probability support and config gates.

These are engineering tests. They cannot establish pretrained-model fidelity,
chemical validity improvement or biological pocket correctness.
"""

import pytest
import torch
import yaml

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.dbmol_objective import (
    CONFIDENCE_SIGNS, CONFIDENCE_WEIGHTS, DBMolObjectiveConfig,
    contact_loss, pocket_token_mask, reduce_objective,
)
from latentmol.boltz.active_set_rescore import _paired_results
from latentmol.optimization import NonfiniteObjective


def profile(tmp_path, **changes):
    raw = {"confidence_checkpoint": {"path": "conf.ckpt", "sha256": "a" * 64},
           "pocket_residues": [{"asym_id": 0, "residue_index": 2}]}
    raw.update(changes)
    return DBMolObjectiveConfig.from_mapping(raw, tmp_path)


def test_composite_has_paper_signs_and_gradients_without_probability_saturation(tmp_path):
    cfg = profile(tmp_path)
    raw = {k: torch.tensor(float(i + 1), dtype=torch.float64, requires_grad=True)
           for i, k in enumerate(CONFIDENCE_WEIGHTS)}
    raw.update(affinity_pred_value=torch.tensor(2., requires_grad=True),
               affinity_logits_binary=torch.tensor(100., requires_grad=True),
               dbmol_contact_loss=torch.tensor(3., requires_grad=True))
    result = reduce_objective(raw, cfg)
    expected_confidence = -1 - .3 * 2 + .3 * 3 + .1 * 4 - .3 * 5 - .1 * 6
    torch.testing.assert_close(result["dbmol_total_loss"], torch.tensor(2 - 100 + 2.5 * 3 + 2.5 * expected_confidence, dtype=torch.float64))
    result["dbmol_total_loss"].backward()
    assert raw["affinity_pred_value"].grad == 1
    assert raw["affinity_logits_binary"].grad == -1
    assert raw["dbmol_contact_loss"].grad == 2.5
    for key in CONFIDENCE_WEIGHTS:
        assert raw[key].grad == pytest.approx(2.5 * CONFIDENCE_SIGNS[key] * CONFIDENCE_WEIGHTS[key])


def test_contact_uses_best_atom_per_pocket_residue_and_global_anchor():
    probabilities = torch.tensor([[[[.2, .2, .3, .3], [.3, .3, .2, .2], [.8, .1, .05, .05], [.1, .1, .4, .4]],
                                   [[.2, .2, .3, .3], [.3, .3, .2, .2], [.2, .1, .3, .4], [.3, .3, .2, .2]],
                                   [[.2, .2, .3, .3]] * 4, [[.2, .2, .3, .3]] * 4]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    pocket = torch.tensor([True, True, False, False])
    ligand = ~pocket
    out = contact_loss(logits, pocket, ligand, epsilon=1e-8,
                       bin_centers=torch.tensor([2., 6., 10., 14.]))
    expected_pocket = (-torch.log(torch.tensor(.9 + 1e-8)) - torch.log(torch.tensor(.6 + 1e-8))) / 2
    expected_anchor = -torch.log(torch.tensor(.9 + 1e-8))
    torch.testing.assert_close(out["dbmol_contact_loss"], ((expected_pocket + expected_anchor) / 2).double())
    out["dbmol_contact_loss"].backward()
    assert logits.grad[0, 0, 2].norm() > 0 and logits.grad[0, 1, 3].norm() > 0
    assert logits.grad[0, 0, 3].norm() == 0 and logits.grad[0, 1, 2].norm() == 0
    assert logits.grad[0, 2:].norm() == 0


def test_tiny_contact_probability_retains_finite_loss_and_gradient():
    logits = torch.tensor([[[[-500., -501., 1., 0.]]]], dtype=torch.float64)
    logits = logits.expand(1, 2, 2, 4).clone().requires_grad_()
    result = contact_loss(logits, torch.tensor([True, False]), torch.tensor([False, True]),
                          bin_centers=torch.tensor([2., 6., 10., 14.]))
    result["dbmol_contact_loss"].backward()
    assert result["dbmol_contact_loss"].isfinite()
    assert logits.grad.isfinite().all()
    # Squaring these tiny float64 derivatives can underflow inside a norm.
    assert logits.grad.abs().max() > 0


def test_only_nonfinite_scalar_terms_are_recoverable_trial_rejections(tmp_path):
    raw = {k: torch.tensor(.5) for k in CONFIDENCE_WEIGHTS}
    raw.update(affinity_pred_value=torch.tensor(1.), affinity_logits_binary=torch.tensor(2.),
               dbmol_contact_loss=torch.tensor(float("inf")))
    with pytest.raises(NonfiniteObjective, match="DBMol term"):
        reduce_objective(raw, profile(tmp_path))
    raw["dbmol_contact_loss"] = torch.tensor([1., 2.])
    with pytest.raises(ValueError) as caught:
        reduce_objective(raw, profile(tmp_path))
    assert not isinstance(caught.value, NonfiniteObjective)


def test_inferred_pocket_reference_rejects_changed_anchor(tmp_path):
    from test_boltz_scoring import prepared_fixture
    from latentmol.boltz.provenance import context_digest
    prepared = prepared_fixture()
    cfg = profile(tmp_path, pocket_origin="initial_pose_heavy_atom_8A",
                  pocket_reference={"prepared_sha256": "b" * 64,
                                    "context_sha256": context_digest(prepared)})
    cfg.verify_reference(prepared, "b" * 64)
    prepared.coordinates[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="frozen anchor"):
        cfg.verify_reference(prepared, "b" * 64)


@pytest.mark.parametrize("change", ["missing", "vector", "nan"])
def test_missing_or_invalid_terms_do_not_silently_drop_a_loss(tmp_path, change):
    raw = {k: torch.tensor(.5) for k in CONFIDENCE_WEIGHTS}
    raw.update(affinity_pred_value=torch.tensor(1.), affinity_logits_binary=torch.tensor(2.),
               dbmol_contact_loss=torch.tensor(.1))
    if change == "missing": del raw["iptm"]
    elif change == "vector": raw["iptm"] = torch.ones(2)
    else: raw["iptm"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="DBMol term"):
        reduce_objective(raw, profile(tmp_path))


@pytest.mark.parametrize("change", ["unknown", "empty_pocket", "duplicate", "head_hash", "bin", "negative", "missing_conf", "pose_receipt"])
def test_profile_rejects_ambiguous_inputs(tmp_path, change):
    raw = profile(tmp_path).record()
    if change == "unknown": raw["invented"] = 1
    elif change == "empty_pocket": raw["pocket_residues"] = []
    elif change == "duplicate": raw["pocket_residues"] *= 2
    elif change == "head_hash": raw["confidence_checkpoint"]["sha256"] = "bad"
    elif change == "bin": raw["bin_policy"] = "unknown"
    elif change == "negative": raw["confidence_weight"] = -1
    elif change == "missing_conf": raw["confidence_weights"].pop("iptm")
    else: raw.update(pocket_origin="initial_pose_heavy_atom_8A", pocket_reference=None)
    with pytest.raises(ValueError):
        DBMolObjectiveConfig.from_mapping(raw, tmp_path)


def test_pocket_resolves_native_residue_identity_instead_of_moving_ligand_indices(tmp_path):
    cfg = profile(tmp_path)
    features = {"token_pad_mask": torch.ones(1, 4), "mol_type": torch.tensor([[0, 0, 3, 3]]),
                "asym_id": torch.tensor([[0, 0, 1, 1]]), "residue_index": torch.tensor([[1, 2, 0, 0]]),
                "affinity_token_mask": torch.tensor([[0, 0, 1, 1]])}
    pocket, ligand = pocket_token_mask(features, cfg)
    assert pocket.tolist() == [False, True, False, False]
    assert ligand.tolist() == [False, False, True, True]
    features["residue_index"][0, 1] = 99
    with pytest.raises(ValueError, match="missing or ambiguous"):
        pocket_token_mask(features, cfg)


def test_active_set_legacy_record_unchanged_and_composite_opt_in_is_strict(tmp_path):
    from test_boltz_active_set_runtime import _config
    legacy = _config(tmp_path)
    assert "dbmol" not in legacy.record()
    raw = legacy.record()
    raw.pop("context_manifest", None); raw.pop("context_identity", None)
    raw.update(version=1, name="dbmol-fixture", fd_policy="local_v1",
               device="cuda", decoder={"path": "decoder.pt", "sha256": "b" * 64},
               affinity_checkpoint={"path": "affinity.ckpt", "sha256": "c" * 64},
               objective={"output": "dbmol_total_loss", "direction": "minimize"},
               dbmol=profile(tmp_path).record())
    path = tmp_path / "dbmol.yaml"
    path.write_text(yaml.safe_dump(raw))
    parsed = ActiveSetConfig.load(path)
    assert parsed.record()["dbmol"]["contact_weight"] == 2.5
    del raw["dbmol"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="selected together"):
        ActiveSetConfig.load(path)


def test_fresh_affinity_does_not_masquerade_as_full_composite_loss():
    selection = {"arms": {"fixed_initial_mask": {"start": {"smiles": "CC"},
                 "endpoint": {"smiles": "CN"}, "endpoint_valid": True, "endpoint_changed": True}}}
    rows = [{"smiles": s, "seed": seed, "retention_eligible": True,
             "affinity": {"affinity_probability_binary": p, "affinity_pred_value": 1.0}}
            for s, p in (("CC", .2), ("CN", .4)) for seed in (17, 23)]
    result = _paired_results(selection, rows, {"output": "dbmol_total_loss", "direction": "minimize"})
    for row in result["fixed_initial_mask"]["paired_seeds"]:
        assert row["endpoint_minus_start_binary"] == pytest.approx(.2)
        assert row["endpoint_minus_start_signed_objective"] is None
        assert row["configured_objective_available"] is False
