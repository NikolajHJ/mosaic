"""Matched comparison scheduling/native boundary fixtures; allocated HPC only."""

import json
from pathlib import Path
import time

import pytest
import torch

from latentmol.boltz.embedding_bridge import HiddenEmbeddingBridge
from latentmol.boltz.embedding_comparison import FrozenComparisonObjective, run_matched_optimization
from latentmol.experiments import file_sha256
from latentmol.optimization import NonfiniteObjective, StepControl
from test_embedding_validation import frozen_models


class ToyObjective:
    """Smooth known objective isolates paired scheduling from Boltz plumbing."""

    def __init__(self, *, kind="quadratic", fail=None):
        self.kind, self.fail = kind, fail
        self.calls = dict(forward=0, backward=0, audit=0)
        self.inputs = []

    def assert_unchanged(self, *, full=False):
        return dict(context_sha256="fixed-toy-context", full_state_check=full,
                    frozen_state_unchanged=True, context_unchanged=True)

    def loss(self, z):
        if self.kind == "flat":
            return 1.
        return float((z - 1).square().sum())

    def evaluate(self, path, z, *, require_gradient=False):
        self.calls["forward"] += 1
        self.inputs.append((path, z.detach().clone(), require_gradient))
        if self.fail is not None and not require_gradient:
            self.fail(z)
        self.calls["backward"] += int(require_gradient)
        gradient = None
        if require_gradient:
            gradient = torch.zeros_like(z) if self.kind == "flat" else 2 * (z - 1)
            if self.kind == "wrong-gradient":
                gradient = -gradient
        loss = self.loss(z)
        return dict(loss=loss, loss_dtype=str(z.dtype), scores={"dbmol_total_loss": loss},
                    context_sha256="fixed-toy-context", gradient=gradient)

    def audit(self, z):
        self.calls["audit"] += 1
        loss = self.loss(z)
        label = int(z.sum() > 0)
        labels = dict(atoms=torch.tensor([[label]]), charges=torch.zeros(1, 1, dtype=torch.long),
                      bonds=torch.zeros(1, 1, 1, dtype=torch.long))
        return dict(record=dict(context_sha256="fixed-toy-context",
            classifications={"conditioned": dict(strict_teacher_valid=True),
                             "natural": dict(strict_teacher_valid=True)},
            canonical_identities={"conditioned": str(label), "natural": str(label)},
            scores={key: {"dbmol_total_loss": loss} for key in ("relaxed", "learned", "hard_fixed_context")}),
            tensors={"conditioned": labels, "natural": labels, "z": z.detach().clone()})


def run(tmp_path, objective, **kwargs):
    return run_matched_optimization({"start0": torch.zeros(1, 2, dtype=torch.float64)},
        objective, tmp_path / "run", steps=kwargs.pop("steps", 3),
        learning_rate=kwargs.pop("learning_rate", .25), max_gradient_norm=10.,
        trust_radius=3., **kwargs)


def test_paired_arms_share_exact_start_update_policy_and_completed_budget(tmp_path):
    objective = ToyObjective()
    result = run(tmp_path, objective)
    assert result["matched_update_budget_completed"]
    assert result["status"] == "completed_comparison_not_binding_validated"
    relaxed, learned = result["runs"]
    assert relaxed["start_latent_sha256"] == learned["start_latent_sha256"]
    assert relaxed["path"] == "relaxed" and learned["path"] == "learned"
    for item in result["runs"]:
        assert item["applied"] == item["improving"] == 3
        assert item["forced_fallback"] == item["no_op"] == 0
        assert len(item["checkpoints"]) == 4
        assert item["callback_calls"] == dict(attempted=6, completed=6, failed=0)
    for left, right in zip(relaxed["checkpoints"], learned["checkpoints"]):
        assert left["latent_sha256"] == right["latent_sha256"]
        assert left["scores"] == right["scores"]
    assert objective.calls == dict(forward=12, backward=6, audit=8)
    assert not result["total_loss_cross_boundary_ranking"]


def test_every_checkpoint_tensor_json_and_selected_trial_are_consistent(tmp_path):
    result = run(tmp_path, ToyObjective(), steps=2)
    for arm in result["runs"]:
        for checkpoint in arm["checkpoints"]:
            tensors = torch.load(checkpoint["tensor_path"], weights_only=True)
            assert file_sha256(Path(checkpoint["tensor_path"])) == checkpoint["tensor_sha256"]
            assert float((tensors["z"] - 1).square().sum()) == checkpoint["scores"][arm["path"]]["dbmol_total_loss"]
            assert tensors["conditioned"]["atoms"].item() == int(tensors["z"].sum() > 0)
            if checkpoint["update"]:
                assert checkpoint["selected_objective_loss"] == checkpoint["scores"][arm["path"]]["dbmol_total_loss"]
                assert checkpoint["audit_minus_selected_objective_loss"] == 0
        best = arm["best_observed"]
        assert best["update"] == 2 and best["objective_path"] == arm["path"]
        assert best["context_sha256"] == "fixed-toy-context"
        assert best["tensor_sha256"] == arm["checkpoints"][-1]["tensor_sha256"]
    saved = json.loads((tmp_path / "run" / "comparison.json").read_text())
    assert saved == result


def test_flat_gradient_consumes_requested_updates_as_honest_no_ops(tmp_path):
    result = run(tmp_path, ToyObjective(kind="flat"), steps=4)
    for arm in result["runs"]:
        assert arm["applied"] == arm["no_op"] == arm["full_step_nonincrease"] == 4
        assert arm["improving"] == arm["forced_fallback"] == 0
        assert all(item["acceptance_kind"] == "full_step_nonincrease" for item in arm["iterations"])
        assert len({item["latent_sha256"] for item in arm["checkpoints"]}) == 1
        assert arm["best_observed"]["update"] == 0


def test_uphill_best_finite_fallback_is_applied_and_not_called_improvement(tmp_path):
    result = run(tmp_path, ToyObjective(kind="wrong-gradient"), steps=2,
                 control=StepControl(version="best_trial_v2", max_trials=3))
    for arm in result["runs"]:
        assert arm["applied"] == arm["forced_fallback"] == 2
        assert arm["improving"] == arm["no_op"] == 0
        assert all(item["accepted_trial"] == 2 and item["acceptance_kind"] == "best_finite_fallback"
                   for item in arm["iterations"])
        assert arm["best_observed"]["update"] == 0
        assert arm["checkpoints"][-1]["scores"][arm["path"]]["dbmol_total_loss"] > 2


def test_declared_nonfinite_trial_remains_in_journal_and_smaller_step_can_run(tmp_path):
    def fail(z):
        if float(z.sum()) > 2:
            raise NonfiniteObjective("fixture overflow")
    result = run(tmp_path, ToyObjective(fail=fail), steps=1, learning_rate=1.)
    for arm in result["runs"]:
        assert arm["applied"] == 1
        assert arm["iterations"][0]["oracle_calls_nonfinite"] == 1
        assert arm["iterations"][0]["accepted_scale"] == .5
        journal = tmp_path / "run" / "start0" / arm["path"] / "objective-calls.jsonl"
        rows = [json.loads(row) for row in journal.read_text().splitlines()]
        assert any(row["status"] == "failed" and row["error_type"] == "NonfiniteObjective" for row in rows)
        assert all("latent" in row and "latent_sha256" in row for row in rows)


def test_all_nonfinite_trials_fail_instead_of_fabricating_completed_budget(tmp_path):
    def fail(z):
        raise NonfiniteObjective("no finite trial")
    with pytest.raises(NonfiniteObjective, match="No finite candidate"):
        run(tmp_path, ToyObjective(fail=fail), steps=2,
            control=StepControl(version="best_trial_v2", max_trials=3))
    saved = json.loads((tmp_path / "run" / "comparison.json").read_text())
    assert saved["status"] == "failed_not_passed" and not saved["matched_update_budget_completed"]
    assert len(saved["runs"]) == 1
    arm = saved["runs"][0]
    assert arm["applied"] == 0 and len(arm["checkpoints"]) == 1
    assert arm["iterations"][0]["oracle_calls_nonfinite"] == 3
    assert arm["callback_calls"] == dict(attempted=4, completed=1, failed=3)


def test_unexpected_native_error_is_persisted_then_propagates(tmp_path):
    def fail(z):
        raise RuntimeError("native bug")
    with pytest.raises(RuntimeError, match="native bug"):
        run(tmp_path, ToyObjective(fail=fail))
    saved = json.loads((tmp_path / "run" / "comparison.json").read_text())
    assert saved["error_type"] == "RuntimeError" and saved["frozen_after"]["full_state_check"]
    arm = saved["runs"][0]
    assert arm["applied"] == 0 and arm["iterations"][0]["oracle_calls_unexpected_errors"] == 1


def test_failed_post_update_audit_preserves_the_actual_applied_latent(tmp_path):
    class AuditFailure(ToyObjective):
        def audit(self, z):
            if bool(z.count_nonzero()):
                raise RuntimeError("hard rescore failed")
            return super().audit(z)
    with pytest.raises(RuntimeError, match="hard rescore failed"):
        run(tmp_path, AuditFailure())
    saved = json.loads((tmp_path / "run" / "comparison.json").read_text())
    arm = saved["runs"][0]
    assert arm["applied"] == 1 and len(arm["checkpoints"]) == 2
    checkpoint = arm["checkpoints"][-1]
    assert checkpoint["status"] == "failed_audit_not_passed"
    assert file_sha256(Path(checkpoint["latent_tensor_path"])) == checkpoint["latent_tensor_sha256"]
    assert torch.equal(torch.load(checkpoint["latent_tensor_path"], weights_only=True)["z"],
                       torch.full((1, 2), .5, dtype=torch.float64))
    assert "scores" not in checkpoint and arm["best_observed"]["update"] == 0


def test_deadline_does_not_create_a_success_or_apply_updates(tmp_path):
    with pytest.raises(TimeoutError, match="deadline"):
        run(tmp_path, ToyObjective(), deadline=time.monotonic() - 1)
    saved = json.loads((tmp_path / "run" / "comparison.json").read_text())
    assert saved["status"] == "failed_not_passed" and saved["runs"] == []


@pytest.mark.parametrize("kwargs", [dict(steps=0), dict(paths=("learned", "relaxed")),
    dict(control=StepControl(version="armijo_v1")), dict(learning_rate=float("inf"))])
def test_undeclared_or_invalid_matched_settings_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run(tmp_path, ToyObjective(), **kwargs)
    assert not (tmp_path / "run").exists()


def native_objective():
    prepared, scorer, bank, vae = frozen_models()
    z = torch.randn(1, vae.config.latent)
    _, hidden = vae.decode_with_features(z)
    bridge = HiddenEmbeddingBridge(bank, atom_input_dim=hidden.atom_hidden.shape[-1],
        pair_input_dim=hidden.bond_input.shape[-1], charge_input_dim=hidden.charge_input.shape[-1],
        hidden_dim=8).eval().requires_grad_(False)
    return FrozenComparisonObjective(scorer, prepared, bridge, vae, bank=bank), z


def test_native_fixture_paths_have_attached_finite_gradients_and_closed_hooks():
    objective, z = native_objective()
    observations = {path: objective.evaluate(path, z, require_gradient=True) for path in ("relaxed", "learned")}
    assert objective.calls["backward_attempted"] == objective.calls["backward_completed"] == 2
    assert objective.calls["boltz_forward_attempted"] == objective.calls["boltz_forward_completed"] == 2
    for result in observations.values():
        assert result["gradient"].shape == z.shape and not result["gradient"].requires_grad
        assert bool(result["gradient"].isfinite().all()) and result["gradient_norm"] > 0
        assert result["loss_dtype"] == "torch.float32"
    assert observations["learned"]["injection_receipt"]["all_channels_consumed"]
    # The fixture's ordinary graph call must remain ordinary after learned backward.
    again = objective.evaluate("relaxed", z, require_gradient=True)
    assert again["scores"] == observations["relaxed"]["scores"]
    torch.testing.assert_close(again["gradient"], observations["relaxed"]["gradient"], rtol=0, atol=0)
    assert objective.assert_unchanged(full=True)["frozen_state_unchanged"]


def test_native_audit_keeps_both_hard_policies_and_exact_forward_counts():
    objective, z = native_objective()
    audit = objective.audit(z)
    assert set(audit["tensors"]) == {"z", "conditioned", "natural"}
    assert set(audit["record"]["canonical_identities"]) == {"conditioned", "natural"}
    assert "strict_teacher_valid" in audit["record"]["classifications"]["conditioned"]
    assert set(audit["record"]["scores"]) == {"relaxed", "learned", "hard_fixed_context"}
    assert audit["record"]["feature_audit"]
    assert objective.calls["boltz_forward_attempted"] == objective.calls["boltz_forward_completed"] == 3
    assert objective.calls["forward_by_boundary"] == dict(relaxed=1, learned=1, hard=1)
    assert objective.calls["backward_attempted"] == 0


def test_native_frozen_parameter_mutation_is_detected_before_another_forward():
    objective, z = native_objective()
    with torch.no_grad():
        next(objective.bridge.parameters()).add_(1)
    with pytest.raises(ValueError, match="changed"):
        objective.evaluate("learned", z)
    assert objective.calls["boltz_forward_attempted"] == 0


def test_native_fixed_context_mutation_is_detected_before_another_forward():
    objective, z = native_objective()
    objective.prepared.coordinates.add_(1)
    with pytest.raises(ValueError, match="changed"):
        objective.evaluate("relaxed", z)
    assert objective.calls["boltz_forward_attempted"] == 0
