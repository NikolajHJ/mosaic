"""HPC-only terminal-boundary fixtures, without pretrained model assets.

Metadata fixtures distinguish explicit stalls from truncated/failed runs. Real
``run_arm`` fixtures exercise the helper against the orchestrator's own saved
reports, and endpoint fixtures preserve the existing chemistry/hash checks.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from latentmol.boltz import active_set
from latentmol.boltz.active_set_rescore import select_endpoints
from latentmol.boltz.trajectory_terminal import read_arm_terminal, validate_arm_terminal
from latentmol.experiments import file_sha256
from latentmol.optimization import StepControl
from test_boltz_active_set_rescore import ARMS, _write, pilot
from test_boltz_active_set_runtime import _config, _Decoder, _Scorer, _invoke


def _accepted(control):
    trial = {"index": 0, "scale": 1., "oracle_called": True, "accepted": True,
             "reason": "accepted", "actual_loss_improvement": .1,
             "observed_improvement_guard": 1e-8, "required_armijo_improvement": 1e-4,
             "strictly_decreased": True, "informative_decrease": True, "armijo_satisfied": True}
    return {"policy": control.record(), "status": "accepted", "accepted": True, "stalled": False,
            "trials": [trial], "oracle_calls_attempted": 1, "oracle_calls_completed": 1,
            "oracle_calls_nonfinite": 0, "oracle_calls_unexpected_errors": 0,
            "accepted_trial": 0, "accepted_scale": 1., "actual_loss_improvement": .1}


def _stalled(control):
    return {"policy": control.record(), "status": "stalled", "accepted": False, "stalled": True,
            "stall_reason": "trial_budget_exhausted", "actual_loss_improvement": 0.,
            "actual_displacement_norm": 0., "oracle_calls_attempted": control.max_trials,
            "oracle_calls_completed": control.max_trials, "oracle_calls_nonfinite": 0,
            "oracle_calls_unexpected_errors": 0,
            "trials": [{"index": index, "scale": scale, "oracle_called": True,
                        "accepted": False, "reason": "no_strict_decrease"}
                       for index, scale in enumerate(control.record()["trial_scales"])]}


def _evidence(*, accepted=1, requested=3, stalled=True):
    control = StepControl(version="armijo_v1", max_trials=2)
    config = {"optimizer": {"steps": requested}, "step_control": {
        "version": control.version, "max_trials": control.max_trials,
        "armijo": control.armijo, "minimum_improvement_ulps": control.minimum_improvement_ulps}}
    rows = [{"step": step, **({"update_status": "accepted"} if step < accepted else
                             {"update_status": "stalled"} if stalled else {})}
            for step in range(accepted + 1)]
    summary = {"steps": accepted, "requested_steps": requested, "accepted_updates": accepted,
               "attempted_updates": accepted + int(stalled), "stalled": stalled,
               "completed_requested_budget": not stalled,
               "optimization_status": "stalled" if stalled else "completed_budget",
               "step_control": control.record(), "endpoint": deepcopy(rows[-1])}
    reports = {step: _accepted(control) for step in range(accepted)}
    if stalled:
        reports[accepted] = _stalled(control)
    return config, summary, rows, reports


def _validate(evidence):
    config, summary, rows, reports = evidence
    return validate_arm_terminal(config, summary, metric_rows=rows, step_reports=reports)


def test_legacy_requires_full_budget_and_preserves_the_config():
    config = {"optimizer": {"steps": 3}}
    before = deepcopy(config)
    receipt = validate_arm_terminal(config, {"steps": 3, "endpoint": {"step": 3}})
    assert receipt["completed_budget"] and not receipt["stalled"]
    assert receipt["endpoint_step"] == receipt["accepted_updates"] == 3
    assert config == before and "step_control" not in config
    for summary in ({"steps": 1, "endpoint": {"step": 1}},
                    {"steps": 3, "endpoint": {"step": 3}, "stalled": True}):
        with pytest.raises(ValueError, match="Legacy trajectory"):
            validate_arm_terminal(config, summary)


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_explicit_stall_is_a_terminal_point_without_budget_completion(accepted):
    receipt = _validate(_evidence(accepted=accepted))
    assert receipt["kind"] == "stalled" and receipt["stalled"]
    assert not receipt["completed_budget"]
    assert receipt["endpoint_step"] == receipt["accepted_updates"] == accepted
    assert receipt["attempted_updates"] == accepted + 1


def test_opted_in_full_budget_has_no_extra_stalled_attempt():
    receipt = _validate(_evidence(accepted=3, stalled=False))
    assert receipt["kind"] == "completed_budget" and receipt["completed_budget"]
    assert receipt["attempted_updates"] == receipt["accepted_updates"] == 3


@pytest.mark.parametrize("field,value", [("requested_steps", 4), ("requested_steps", True), ("steps", 2),
    ("accepted_updates", True), ("attempted_updates", 1), ("stalled", "true"),
    ("completed_requested_budget", True), ("optimization_status", "failed")])
def test_inconsistent_summary_never_promotes_a_partial_run(field, value):
    evidence = _evidence()
    evidence[1][field] = value
    with pytest.raises(ValueError):
        _validate(evidence)


def test_missing_intermediate_metrics_and_extra_unrecorded_attempt_are_rejected():
    config, summary, rows, reports = _evidence(accepted=2)
    with pytest.raises(ValueError, match="contiguous"):
        validate_arm_terminal(config, summary, metric_rows=[rows[0], rows[-1]], step_reports=reports)
    reports[3] = deepcopy(reports[2])
    with pytest.raises(ValueError, match="attempt sequence"):
        validate_arm_terminal(config, summary, metric_rows=rows, step_reports=reports)


def test_missing_or_incomplete_stall_report_is_not_an_explicit_terminal():
    config, summary, rows, reports = _evidence()
    with pytest.raises(ValueError, match="attempt sequence"):
        validate_arm_terminal(config, summary, metric_rows=rows, step_reports={0: reports[0]})
    reports[1]["trials"].pop()
    reports[1]["oracle_calls_attempted"] = reports[1]["oracle_calls_completed"] = 1
    with pytest.raises(ValueError, match="exhausted trial budget"):
        validate_arm_terminal(config, summary, metric_rows=rows, step_reports=reports)


def test_failed_or_unproven_accepted_attempt_is_not_a_terminal_trajectory():
    evidence = _evidence()
    evidence[3][0]["trials"][0]["actual_loss_improvement"] = 0.
    with pytest.raises(ValueError, match="sufficient-decrease"):
        _validate(evidence)
    evidence = _evidence()
    evidence[3][1]["status"] = "failed"
    with pytest.raises(ValueError, match="does not prove stalled"):
        _validate(evidence)


def test_unfinished_trial_or_inconsistent_oracle_count_cannot_explain_a_stall():
    evidence = _evidence()
    evidence[3][1]["trials"][-1]["reason"] = None
    with pytest.raises(ValueError, match="finished outcome"):
        _validate(evidence)
    evidence = _evidence()
    evidence[3][1]["oracle_calls_attempted"] = 0
    with pytest.raises(ValueError, match="oracle count"):
        _validate(evidence)


@pytest.mark.parametrize("field", ["actual_loss_improvement", "actual_displacement_norm"])
def test_boolean_zero_cannot_claim_unchanged_stalled_point(field):
    evidence = _evidence()
    evidence[3][1][field] = False
    with pytest.raises(ValueError, match="must be finite"):
        _validate(evidence)


def test_trial_outcome_and_completed_call_evidence_cannot_disagree():
    evidence = _evidence()
    evidence[3][1]["trials"][0]["reason"] = "accepted"
    with pytest.raises(ValueError, match="finished outcome"):
        _validate(evidence)
    evidence = _evidence()
    evidence[3][1]["oracle_calls_completed"] = 0
    with pytest.raises(ValueError, match="completed calls"):
        _validate(evidence)


def test_terminal_helper_reads_actual_accepted_run_arm_reports(tmp_path):
    config = replace(_config(tmp_path), step_control=StepControl())
    summary = _invoke("active_set", _Decoder(), _Scorer(), config, tmp_path)
    receipt = read_arm_terminal(tmp_path / "active_set", config.record(), summary)
    assert receipt["kind"] == "completed_budget" and receipt["accepted_updates"] == 2


def test_terminal_helper_reads_actual_unchanged_stalled_run_arm_reports(tmp_path, monkeypatch):
    config = replace(_config(tmp_path), step_control=StepControl(version="armijo_v1", max_trials=2))
    original_score = active_set.score_at
    def trial_plateau(decoder, point, branch, config, counters, purpose, **kwargs):
        if purpose == "step_control":
            return {"loss": 100., "loss_dtype": "torch.float64"}, None
        return original_score(decoder, point, branch, config, counters, purpose, **kwargs)
    monkeypatch.setattr(active_set, "score_at", trial_plateau)
    summary = _invoke("active_set", _Decoder(), _Scorer(), config, tmp_path)
    receipt = read_arm_terminal(tmp_path / "active_set", config.record(), summary)
    assert receipt["kind"] == "stalled" and receipt["endpoint_step"] == 0
    assert not (tmp_path / "active_set" / "metrics-0001.json").exists()


def _best_evidence(losses, *, selected, kind, zero=()):
    """Saved v2 metadata only: no model call, repair or tensor arithmetic."""
    control = StepControl(version="best_trial_v2", max_trials=3)
    config = {"optimizer": {"steps": 1}, "step_control": {
        "version": control.version, "max_trials": control.max_trials}}
    rows = [{"step": 0, "update_status": "accepted"}, {"step": 1}]
    trials = [{"index": index, "scale": 2. ** -index, "oracle_called": True,
        "accepted": index == selected, "reason": "accepted" if index == selected else "no_strict_decrease",
        "finite_proposal": True, "finite_candidate": True,
        "candidate_loss": loss, "actual_loss_improvement": 4. - loss,
        "strictly_decreased": loss < 4., "actual_displacement_norm": 0. if index in zero else .1,
        "zero_displacement": index in zero} for index, loss in enumerate(losses)]
    report = {"policy": control.record(), "status": "accepted", "accepted": True, "stalled": False,
        "baseline_loss": 4., "trials": trials, "oracle_calls_attempted": len(trials),
        "oracle_calls_completed": len(trials), "oracle_calls_nonfinite": 0,
        "oracle_calls_unexpected_errors": 0, "accepted_trial": selected, "accepted_scale": 2. ** -selected,
        "acceptance_kind": kind, "monotone_descent": losses[selected] < 4. and selected not in zero,
        "no_op": selected in zero, "selected_loss": losses[selected],
        "actual_loss_improvement": 4. - losses[selected],
        "actual_displacement_norm": trials[selected]["actual_displacement_norm"]}
    summary = {"steps": 1, "requested_steps": 1, "accepted_updates": 1, "attempted_updates": 1,
        "stalled": False, "completed_requested_budget": True, "optimization_status": "completed_budget",
        "step_control": control.record(), "endpoint": deepcopy(rows[-1])}
    return config, summary, rows, {0: report}


@pytest.mark.parametrize("losses,selected,kind,zero", [
    ([3.9], 0, "first_improvement", ()),
    ([4.2, 3.9], 1, "first_improvement", ()),
    ([4.2, 4.1, 4.3], 1, "best_finite_fallback", ()),
    ([4.2, 4.1, 4.1], 1, "best_finite_fallback", ()),
    ([4.], 0, "full_step_nonincrease", ()),
    ([4.], 0, "full_step_nonincrease", (0,)),
    ([3.9], 0, "full_step_nonincrease", (0,)),
])
def test_best_trial_terminal_authenticates_improvement_uphill_equal_and_noop(losses, selected, kind, zero):
    receipt = _validate(_best_evidence(losses, selected=selected, kind=kind, zero=zero))
    assert receipt["completed_budget"] and not receipt["stalled"]
    assert receipt["step_control_version"] == "best_trial_v2"
    assert "descent" in receipt["claim_limit"]


@pytest.mark.parametrize("losses,selected,kind,zero", [
    ([4., 4., 4.], 0, "best_finite_fallback", ()),
    ([4., 3.9], 1, "first_improvement", ()),
    ([3.9, 4.1, 4.2], 0, "best_finite_fallback", (0,)),
    ([4.2], 0, "full_step_nonincrease", ()),
    ([3.9], 0, "full_step_nonincrease", ()),
    ([4.2, 4.], 1, "full_step_nonincrease", ()),
])
def test_best_trial_full_step_nonincrease_never_continues_or_mislabels_a_smaller_trial(losses, selected, kind, zero):
    with pytest.raises(ValueError):
        _validate(_best_evidence(losses, selected=selected, kind=kind, zero=zero))


@pytest.mark.parametrize("defect", ["not_best", "not_earliest_tie", "short_fallback", "late_improvement",
    "hidden_earlier_improvement", "false_monotonicity", "false_gain", "false_noop", "false_selected_loss",
    "oracle_error", "wrong_index", "inconsistent_status"])
def test_best_trial_terminal_rejects_favorable_or_incomplete_selection_claims(defect):
    evidence = _best_evidence([4.2, 4.1, 4.3], selected=1, kind="best_finite_fallback")
    report = evidence[3][0]
    if defect == "not_best": report["trials"][0]["candidate_loss"] = 4.05; report["trials"][0]["actual_loss_improvement"] = 4. - 4.05
    if defect == "not_earliest_tie": report["trials"][0]["candidate_loss"] = 4.1; report["trials"][0]["actual_loss_improvement"] = 4. - 4.1
    if defect == "short_fallback": report["trials"].pop(); report["oracle_calls_attempted"] = report["oracle_calls_completed"] = 2
    if defect == "late_improvement": report["acceptance_kind"] = "first_improvement"
    if defect == "hidden_earlier_improvement":
        report["trials"][0].update(candidate_loss=3.9, actual_loss_improvement=4. - 3.9, strictly_decreased=True)
    if defect == "false_monotonicity": report["monotone_descent"] = True
    if defect == "false_gain": report["actual_loss_improvement"] = 1.
    if defect == "false_noop": report["no_op"] = True
    if defect == "false_selected_loss": report["selected_loss"] = 0.
    if defect == "oracle_error": report["oracle_calls_unexpected_errors"] = 1
    if defect == "wrong_index": report["accepted_trial"] = 20
    if defect == "inconsistent_status": report["status"] = "stalled"
    with pytest.raises(ValueError):
        _validate(evidence)


def test_best_trial_never_accepts_a_historical_partial_stall_as_full_budget():
    evidence = _evidence()
    control = StepControl(version="best_trial_v2", max_trials=2)
    evidence[0]["step_control"]["version"] = control.version
    evidence[1]["step_control"] = control.record()
    with pytest.raises(ValueError, match="full requested update budget"):
        _validate(evidence)


def _opt_stalled_pilot(pilot):
    """Modify only fixture metadata; preserve all raw candidate and gate receipts."""
    record = json.loads((pilot / "run.json").read_text())
    record["config"]["step_control"] = _evidence()[0]["step_control"]
    record["config_sha256"] = hashlib.sha256(json.dumps(record["config"], sort_keys=True).encode()).hexdigest()
    _write(pilot / "run.json", record)
    parent_summary = json.loads((pilot / "summary.json").read_text())
    for arm in ARMS:
        # The two matched arms are allowed different valid termination steps.
        accepted = 0 if arm == "active_set" else 1
        _, terminal, _, reports = _evidence(accepted=accepted, requested=2)
        folder = pilot / arm
        summary = json.loads((folder / "summary.json").read_text())
        candidates = [json.loads(line) for line in (folder / "candidates.jsonl").read_text().splitlines()]
        candidates = [row for row in candidates if row["step"] <= accepted]
        (folder / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in candidates) + "\n")
        metrics = [{"step": step, "context_sha256": "a" * 64,
                    "argmax_smiles": candidates[step]["smiles"],
                    "update_status": "accepted" if step < accepted else "stalled"}
                   for step in range(accepted + 1)]
        summary.update({**terminal, "endpoint": metrics[-1]})
        _write(folder / "summary.json", summary)
        for row in metrics:
            _write(folder / f"metrics-{row['step']:04d}.json", row)
        for index, report in reports.items():
            _write(folder / f"step-control-{index:04d}.json", report)
        parent_summary["arms"][arm] = summary
        parent_summary["candidate_file_sha256"][arm] = file_sha256(folder / "candidates.jsonl")
    _write(pilot / "summary.json", parent_summary)
    return pilot


def test_native_selector_uses_each_explicit_stall_endpoint_and_seals_evidence(pilot):
    selection = select_endpoints(_opt_stalled_pilot(pilot))
    assert selection["steps"] == 2  # Requested budget stays separate from actual terminals.
    for arm, expected_step in (("active_set", 0), ("fixed_initial_mask", 1)):
        chosen = selection["arms"][arm]
        assert chosen["termination"]["stalled"] and not chosen["termination"]["completed_budget"]
        assert chosen["endpoint"]["step"] == expected_step
        assert f"{arm}/step-control-{expected_step:04d}.json" in selection["parent_artifact_sha256"]
        assert f"{arm}/metrics-{expected_step:04d}.json" in selection["parent_artifact_sha256"]


def test_native_selector_still_rejects_failed_parent_or_missing_stall(pilot):
    _opt_stalled_pilot(pilot)
    report = pilot / "active_set" / "step-control-0000.json"
    report.unlink()
    with pytest.raises(ValueError, match="attempt sequence"):
        select_endpoints(pilot)
    record = json.loads((pilot / "run.json").read_text())
    record["status"] = "failed"
    _write(pilot / "run.json", record)
    with pytest.raises(ValueError, match="not complete"):
        select_endpoints(pilot)
