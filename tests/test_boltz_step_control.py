"""HPC fixtures for optional step control in the real active-set orchestrator."""

from dataclasses import replace
import json

import pytest
import torch
import yaml

from latentmol.boltz import active_set
from latentmol.config_identity import config_sha256
from latentmol.multimodel.engine import Counters
from latentmol.optimization import NonfiniteObjective, StepControl
from test_boltz_active_set_runtime import _config, _Decoder, _Scorer, _invoke, _yaml_configuration, SPEC


def test_legacy_config_record_is_unchanged_without_step_control(tmp_path):
    config = _config(tmp_path)
    expected = config.record()
    assert "step_control" not in expected
    assert replace(config, step_control=None).record() == expected
    assert config_sha256(replace(config, step_control=None).record()) == config_sha256(expected)
    opted = replace(config, step_control=StepControl(version="armijo_v1"))
    assert opted.record()["step_control"]["version"] == "armijo_v1"
    assert config_sha256(opted.record()) != config_sha256(expected)


def test_versioned_step_control_is_explicitly_loaded_and_rejects_extras(tmp_path):
    raw = _yaml_configuration()
    path = tmp_path / "config.yaml"
    raw["step_control"] = {"version": "armijo_v1", "max_trials": 8, "minimum_improvement_ulps": 2.}
    path.write_text(yaml.safe_dump(raw))
    assert active_set.ActiveSetConfig.load(path).step_control.max_trials == 8
    raw["step_control"]["unknown"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="step_control"):
        active_set.ActiveSetConfig.load(path)


def test_actual_run_arm_accepts_steps_and_keeps_context_switches_separately_audited(tmp_path):
    config = replace(_config(tmp_path), step_control=StepControl(version="armijo_v1"))
    summary = _invoke("active_set", _Decoder(), _Scorer(), config, tmp_path)
    assert summary["accepted_updates"] == summary["requested_steps"] == 2
    assert summary["completed_requested_budget"] and not summary["stalled"]
    assert summary["context_switches"] == 1 and summary["gated_contexts"] == 2
    for step in range(2):
        control = json.loads((tmp_path / "active_set" / f"step-control-{step:04d}.json").read_text())
        assert control["accepted"] and control["actual_loss_improvement"] > 0
        assert control["gradient_norm"] > 0 and control["clipping_factor"] > 0
        assert control["baseline_loss_dtype"] == "torch.float64"
        accepted = control["trials"][control["accepted_trial"]]
        assert accepted["objective_metadata"]["scope"].startswith("Trial retains original fixed mask")
    assert json.loads((tmp_path / "active_set" / "transition-0001.json").read_text())["status"] == "completed"


def test_actual_run_arm_stall_preserves_saved_point_and_reports_uncompleted_budget(tmp_path, monkeypatch):
    config = replace(_config(tmp_path), step_control=StepControl(version="armijo_v1", max_trials=4))
    original_score = active_set.score_at
    def no_trial_gain(decoder, point, branch, config, counters, purpose, **kwargs):
        if purpose == "step_control":
            # Native-input and FD gates still use the real smooth frozen fixture.
            # Only explicit optimizer trials have a deliberately stalled objective.
            return {"loss": 100., "loss_dtype": "torch.float64"}, None
        return original_score(decoder, point, branch, config, counters, purpose, **kwargs)
    monkeypatch.setattr(active_set, "score_at", no_trial_gain)
    summary = _invoke("active_set", _Decoder(), _Scorer(), config, tmp_path)
    assert summary["stalled"] and not summary["completed_requested_budget"]
    assert summary["accepted_updates"] == 0 and summary["steps"] == 0
    assert summary["attempted_updates"] == 1 and summary["endpoint"]["step"] == 0
    assert summary["endpoint"]["update_status"] == "stalled"
    folder = tmp_path / "active_set"
    assert not (folder / "state-0001.pt").exists()
    saved = torch.load(folder / "state-0000.pt", weights_only=True)
    assert torch.equal(saved["z"], torch.zeros(1, 1, dtype=torch.float64))
    assert json.loads((folder / "step-control-0000.json").read_text())["oracle_calls_attempted"] == 4


def test_future_policy_finishes_requested_budget_when_all_trials_worsen(tmp_path, monkeypatch):
    config = replace(_config(tmp_path), step_control=StepControl())
    original_score = active_set.score_at
    def worse_trials(decoder, point, branch, config, counters, purpose, **kwargs):
        if purpose == "step_control":
            return {"loss": 100. + float(point.double().norm()), "loss_dtype": "torch.float64"}, None
        return original_score(decoder, point, branch, config, counters, purpose, **kwargs)
    monkeypatch.setattr(active_set, "score_at", worse_trials)
    summary = _invoke("fixed_initial_mask", _Decoder(), _Scorer(), config, tmp_path)
    assert summary["completed_requested_budget"] and not summary["stalled"]
    assert summary["steps"] == summary["applied_updates"] == config.optimizer.steps
    assert summary["attempted_updates"] == config.optimizer.steps
    assert summary["forced_fallback_updates"] == config.optimizer.steps
    assert summary["improving_updates"] == 0 and summary["best_binding_states"]
    assert (tmp_path / "fixed_initial_mask" / f"state-{config.optimizer.steps:04d}.pt").is_file()
    for step in range(config.optimizer.steps):
        row = json.loads((tmp_path / "fixed_initial_mask" / f"metrics-{step:04d}.json").read_text())
        assert row["selection_reason"] == "best_finite_fallback" and not row["strictly_improved"]
        assert row["step_control_oracle_calls"] == 16


@pytest.mark.parametrize("kind", ["nonfinite", "wrong_shape", "wrong_type"])
def test_trial_score_rejects_only_true_nonfinite_scalar_heads(tmp_path, kind):
    class BadHeadScorer(_Scorer):
        def forward(self, features, types, coordinates):
            output = super().forward(features, types, coordinates)
            output["affinity_pred_value"] = (torch.tensor(float("nan")) if kind == "nonfinite" else
                                               torch.ones(2) if kind == "wrong_shape" else .5)
            return output
    config = replace(_config(tmp_path), step_control=StepControl(version="armijo_v1"))
    decoder, point = _Decoder(), torch.zeros(1, 1, dtype=torch.float64)
    selection = active_set.select_slots(decoder.decode(point), temperature=1., threshold=.5)
    branch = active_set.make_branch(BadHeadScorer(), active_set.ContextBank(config, SPEC), selection, config, "c" * 64)
    if kind == "nonfinite":
        with pytest.raises(NonfiniteObjective):
            active_set.score_at(decoder, point, branch, config, Counters(), "step_control",
                                gradients=False, reject_nonfinite_objective=True)
    else:
        with pytest.raises(ValueError) as failure:
            active_set.score_at(decoder, point, branch, config, Counters(), "step_control",
                                gradients=False, reject_nonfinite_objective=True)
        assert not isinstance(failure.value, NonfiniteObjective)
