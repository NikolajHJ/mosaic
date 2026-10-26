"""Allocated-HPC fixtures for paired relaxed/learned start derivative gates."""

from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz import additive_fd
from latentmol.boltz import embedding_comparison_validation as validation
from latentmol.boltz.dbmol_objective import (COMPONENT_OUTPUTS, CONFIDENCE_WEIGHTS,
                                           CONTACT_HEAD64_ARITHMETIC)
from latentmol.boltz.embedding_bridge import HiddenEmbeddingBridge
from test_embedding_validation import ScorerFixture, frozen_models


class DBMolFixture(ScorerFixture):
    """A smooth native-shaped scorer with the declared full objective tree."""

    def __init__(self):
        super().__init__()
        self.recycling_steps = 0
        self.dbmol_config = SimpleNamespace(objective_arithmetic=CONTACT_HEAD64_ARITHMETIC,
            confidence_weights=CONFIDENCE_WEIGHTS, affinity_regression_weight=1.,
            affinity_logit_weight=1., contact_weight=2.5, confidence_weight=2.5)

    def forward(self, features, bonds, coordinates):
        signal = super().forward(features, bonds, coordinates)["dbmol_total_loss"].double()
        values = dict(affinity_pred_value=signal, affinity_logits_binary=.3 * signal,
            dbmol_contact_pocket_loss=signal.square() + .2,
            dbmol_contact_anchor_loss=.2 * signal + .3)
        values.update({name: (.11 + .07 * index) * signal.square() + .01 * signal
                       for index, name in enumerate(CONFIDENCE_WEIGHTS)})
        for key in (*COMPONENT_OUTPUTS, "combined"):
            weights = additive_fd.primitive_weights(self.dbmol_config, key)
            result = additive_fd.grouped_native_expression(
                {name: values[name] for name in weights}, self.dbmol_config, key)
            values["dbmol_total_loss" if key == "combined" else key] = result
        return values


def comparison_fixture():
    prepared, _, _, vae = frozen_models()
    # Build a new bank after loading the scorer, as in the real allocation.
    from latentmol.boltz.embedding_bridge import NativeProjectionBank
    scorer = DBMolFixture()
    bank = NativeProjectionBank.from_models({"affinity": scorer.model}).eval()
    generator = torch.Generator().manual_seed(941)
    z = torch.randn(1, vae.config.latent, generator=generator)
    _, hidden = vae.decode_with_features(z)
    bridge = HiddenEmbeddingBridge(bank, atom_input_dim=hidden.atom_hidden.shape[-1],
        pair_input_dim=hidden.bond_input.shape[-1],
        charge_input_dim=hidden.charge_input.shape[-1], hidden_dim=8).eval().requires_grad_(False)
    return prepared, scorer, bridge, vae, z


@pytest.mark.parametrize("scoring_path", ["relaxed", "learned"])
def test_paired_gate_independent_forward_reverse_and_finite_chords(tmp_path, scoring_path):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    path = tmp_path / (scoring_path + ".json")
    report = validation.learned_start_gate(scorer, prepared, bridge, vae, z, path,
                                          scoring_path=scoring_path)
    assert json.loads(path.read_text()) == report
    assert report["passed"] and report["status"] == "passed_pointwise_start_only"
    assert report["scoring_path"] == scoring_path
    assert report["production_local_scalar_exact"]
    assert report["grad_no_grad_primal_exact"] and report["production_repeated_primal_exact"]
    assert all(report["native_grouped_parity"]["scalar_exact"].values())
    assert all(report["native_grouped_parity"]["reverse_exact"].values())
    assert report["native_grouped_parity"]["production_total_reverse_exact"]
    assert len(report["independent_forward_ad"]) == 4
    assert all(row["passed"] for row in report["independent_forward_ad"])
    assert len(report["finite_difference_observations"]) == 12
    assert all(not row["secant_agreement_required"] for row in report["finite_difference_observations"])
    assert report["frozen_modules_unchanged"] and report["context_unchanged"]
    assert not report["useful_gradient_validated"] and not report["molecular_improvement_validated"]
    assert report["calls"]["independent_forward_ad_attempted"] == 4
    with pytest.raises(FileExistsError):
        validation.learned_start_gate(scorer, prepared, bridge, vae, z, path,
                                      scoring_path=scoring_path)


@pytest.mark.parametrize("scoring_path", ["relaxed", "learned"])
def test_forward_ad_corruption_cannot_be_hidden_by_good_finite_chords(tmp_path, monkeypatch, scoring_path):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    original = validation._local_scores

    @contextmanager
    def corrupted(*arguments, **kwargs):
        with original(*arguments, **kwargs) as (scores, receipt):
            primal, tangent = torch.autograd.forward_ad.unpack_dual(scores["dbmol_total_loss"])
            if tangent is not None:
                scores = {**scores, "dbmol_total_loss": torch.autograd.forward_ad.make_dual(primal, tangent * 2)}
            yield scores, receipt

    monkeypatch.setattr(validation, "_local_scores", corrupted)
    path = tmp_path / "corrupted.json"
    with pytest.raises(ValueError, match=scoring_path + ".*forward/reverse AD"):
        validation.learned_start_gate(scorer, prepared, bridge, vae, z, path,
                                      scoring_path=scoring_path)
    report = json.loads(path.read_text())
    assert not report["passed"] and report["status"] == "failed_not_passed"
    assert report["stage"] == "independent_native_forward_ad"
    assert any(not row["passed"] for item in report["independent_forward_ad"] for row in item["rows"])
    assert "finite_difference_observations" not in report
    assert report["frozen_modules_unchanged"] and report["context_unchanged"]


def test_component_zero_gradient_fails_without_substitute_start(tmp_path, monkeypatch):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    original = scorer.forward

    def zero_affinity(*args, **kwargs):
        values = original(*args, **kwargs)
        # Keep native recomposition algebra exact but remove this component's
        # signal: the common start must fail rather than be outcome-replaced.
        values["affinity_pred_value"] = values["affinity_pred_value"] * 0
        values["affinity_logits_binary"] = values["affinity_logits_binary"] * 0
        for key in (COMPONENT_OUTPUTS[0], "combined"):
            weights = additive_fd.primitive_weights(scorer.dbmol_config, key)
            values["dbmol_total_loss" if key == "combined" else key] = additive_fd.grouped_native_expression(
                {name: values[name] for name in weights}, scorer.dbmol_config, key)
        return values

    monkeypatch.setattr(scorer, "forward", zero_affinity)
    path = tmp_path / "zero-component.json"
    with pytest.raises(ValueError, match="Zero/uninformative.*component"):
        validation.learned_start_gate(scorer, prepared, bridge, vae, z, path)
    report = json.loads(path.read_text())
    assert not report["passed"]
    assert report["component_gradient_norms"][COMPONENT_OUTPUTS[0]] == 0
    assert "independent_forward_ad" not in report


def test_relaxed_score_has_shared_hard_identity_and_attached_gradient():
    from latentmol.boltz.embedding_validation import score_learned_bridge
    prepared, scorer, bridge, vae, z = comparison_fixture()
    old = validation.score_relaxed_path(scorer, prepared, vae, z, require_gradient=True)
    learned = score_learned_bridge(scorer, prepared, bridge, vae, z,
                                    require_gradient=True, include_hard_score=False)
    for name in ("atoms", "charges", "bonds"):
        assert torch.equal(getattr(old["hard"], name), getattr(learned["hard"], name))
    assert old["gradient"].shape == z.shape and old["gradient_norm"] > 0
    assert not old["gradient"].requires_grad
    assert old["injection_receipt"] == {"scoring_path": "relaxed", "injection_used": False}


def test_deadline_and_unsupported_path_preserve_no_pass(tmp_path):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    path = tmp_path / "deadline.json"
    with pytest.raises(TimeoutError):
        validation.learned_start_gate(scorer, prepared, bridge, vae, z, path, deadline=0.)
    report = json.loads(path.read_text())
    assert not report["passed"] and report["error_type"] == "TimeoutError"
    with pytest.raises(ValueError, match="scoring_path"):
        validation.learned_start_gate(scorer, prepared, bridge, vae, z,
            tmp_path / "unsupported.json", scoring_path="different")
