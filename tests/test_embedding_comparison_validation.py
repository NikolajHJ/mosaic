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
        signal = self.signal_transform(signal)
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

    @staticmethod
    def signal_transform(value):
        return value


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


@pytest.mark.parametrize("derivative_policy", [validation.LEGACY_VERSION, validation.VERSION])
def test_connected_flat_component_is_correct_but_legacy_qualification_is_retained(tmp_path, monkeypatch,
                                                                               derivative_policy):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    original = scorer.forward

    def zero_affinity(*args, **kwargs):
        values = original(*args, **kwargs)
        # Multiplication by zero leaves a real attached derivative path. The
        # derivative is zero, which is different from a detached required path.
        values["affinity_pred_value"] = values["affinity_pred_value"] * 0
        values["affinity_logits_binary"] = values["affinity_logits_binary"] * 0
        for key in (COMPONENT_OUTPUTS[0], "combined"):
            weights = additive_fd.primitive_weights(scorer.dbmol_config, key)
            values["dbmol_total_loss" if key == "combined" else key] = additive_fd.grouped_native_expression(
                {name: values[name] for name in weights}, scorer.dbmol_config, key)
        return values

    monkeypatch.setattr(scorer, "forward", zero_affinity)
    path = tmp_path / "zero-component.json"
    if derivative_policy == validation.LEGACY_VERSION:
        with pytest.raises(ValueError, match="Zero/uninformative.*component"):
            validation.learned_start_gate(scorer, prepared, bridge, vae, z, path,
                                          derivative_policy=derivative_policy)
    else:
        validation.learned_start_gate(scorer, prepared, bridge, vae, z, path,
                                      derivative_policy=derivative_policy)
    report = json.loads(path.read_text())
    assert report["component_gradient_norms"][COMPONENT_OUTPUTS[0]] == 0
    if derivative_policy == validation.LEGACY_VERSION:
        assert not report["passed"] and "independent_forward_ad" not in report
    else:
        assert report["passed"]
        assert report["component_signal"][COMPONENT_OUTPUTS[0]] == "low_or_zero_signal"
        assert all(row["passed"] for item in report["independent_forward_ad"] for row in item["rows"])


@pytest.mark.parametrize("detached", [False, True])
def test_flat_component_does_not_waive_a_missing_or_detached_required_path(tmp_path, monkeypatch, detached):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    original = scorer.forward

    def missing_affinity(*args, **kwargs):
        values = original(*args, **kwargs)
        for name in ("affinity_pred_value", "affinity_logits_binary"):
            values[name] = (values[name].detach() if detached
                            else torch.zeros_like(values[name], requires_grad=True))
        for key in (COMPONENT_OUTPUTS[0], "combined"):
            weights = additive_fd.primitive_weights(scorer.dbmol_config, key)
            values["dbmol_total_loss" if key == "combined" else key] = additive_fd.grouped_native_expression(
                {name: values[name] for name in weights}, scorer.dbmol_config, key)
        return values

    monkeypatch.setattr(scorer, "forward", missing_affinity)
    path = tmp_path / "missing-component.json"
    with pytest.raises(ValueError, match="Detached required|Missing required"):
        validation.learned_start_gate(scorer, prepared, bridge, vae, z, path)
    assert not json.loads(path.read_text())["passed"]


def test_stationary_total_is_correct_without_claiming_an_informative_update(tmp_path, monkeypatch):
    prepared, scorer, bridge, vae, z = comparison_fixture()
    original = scorer.forward
    monkeypatch.setattr(scorer, "forward", lambda *args, **kwargs:
                        {key: value * 0 for key, value in original(*args, **kwargs).items()})
    report = validation.learned_start_gate(scorer, prepared, bridge, vae, z, tmp_path / "stationary.json")
    assert report["passed"] and report["gradient_norm"] == 0
    assert report["gradient_label_is_seeded_probe"]
    assert report["optimization_qualification"]["informative_total"] is False
    assert all(row["passed"] for item in report["independent_forward_ad"] for row in item["rows"])


@pytest.mark.parametrize("stationary", [False, True])
def test_tiny_wrong_forward_derivative_cannot_hide_below_the_absolute_ad_tolerance(tmp_path, stationary):
    class WrongFlatDerivative(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value * (0. if stationary else 1e-8)

        @staticmethod
        def backward(ctx, cotangent):
            return cotangent * (0. if stationary else 1e-8)

        @staticmethod
        def jvp(ctx, tangent):
            return torch.full_like(tangent, 1e-7) if stationary else -tangent * 1e-8

    prepared, scorer, bridge, vae, z = comparison_fixture()
    scorer.signal_transform = WrongFlatDerivative.apply
    path = tmp_path / "tiny-wrong-derivative.json"
    with pytest.raises(ValueError, match="forward/reverse AD"):
        validation.learned_start_gate(scorer, prepared, bridge, vae, z, path)
    report = json.loads(path.read_text())
    if not stationary:
        assert report["gradient_norm"] > validation.MIN_GRADIENT_NORM
    rows = [row for item in report["independent_forward_ad"] for row in item["rows"]]
    assert any(row["absolute_error"] <= row["allowed_error"]
               and not row["relative_ad_consistency"]["passed"] for row in rows)
    assert not report["passed"]


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
