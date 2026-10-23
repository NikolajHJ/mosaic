"""HPC-only fixture checks; no scheduler calls or native model execution."""

from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
import submit_defog_single_recovery as recovery
from latentmol.experiments import write_json


def _raw():
    return {"name": "ramped-prior-0", "output_root": "/old", "optimizer": {"steps": 100, "learning_rate": 400.},
        "arms": ["fixed_initial_mask"], "start": {"kind": "standard_normal", "seed": 0},
        "initial_latent": {"path": "/saved-z", "sha256": "a" * 64}, "fd_policy": "local_v1",
        "gradient_audit_interval": 50, "dbmol": {"contact_weight": 2.5}, "step_control": {"method": "backtracking"},
        "diagnostics": {"seed": 1729, "fd_rtol": .05, "fd_atol": 1e-6},
        "repair_guidance": {"refresh_interval": 10, "attempts_per_refresh": 2, "weight_schedule": {"kind": "ramp"},
                            "provider": {"python": "/defog/python", "charge_policy": "neutral_only"}}}


def test_single_config_changes_only_name_and_destination():
    original = _raw()
    clone = recovery.clone_case(original, Path("/new/optimization"))
    assert recovery.scientific_config(clone) == recovery.scientific_config(original)
    assert original["output_root"] == "/old"
    assert clone["repair_guidance"]["provider"]["charge_policy"] == "neutral_only"
    assert clone["optimizer"]["steps"] == 100


@pytest.mark.parametrize("key,value", [("gradient_audit_interval", 100), ("fd_policy", None),
                                       ("start", {"kind": "standard_normal", "seed": 3})])
def test_not_a_shortened_or_different_seed_resume(key, value):
    raw = _raw()
    raw[key] = value
    with pytest.raises(ValueError, match="original prior-0"):
        recovery.clone_case(raw, Path("/new"))


def test_scheduler_plan_declares_eight_hours_and_managed_mps_without_submission():
    rows = recovery.workflow(Path("/work3/user/releases/source-fix"), Path("/work3/user"),
        {"path": "/work3/user/plan.json", "sha256": "a" * 64},
        {"path": "/work3/user/readiness.json", "sha256": "b" * 64}, archive_sha="c" * 64, manifest_sha="d" * 64)
    assert len(rows) == 2
    native = rows[1]
    assert native["arguments"][native["arguments"].index("-W") + 1] == "8:00"
    assert "num=1:mode=exclusive_process:mps=yes:j_exclusive=yes" in native["arguments"]
    assert native["dependency"] == "done:cpu"
    assert recovery.OPTIMIZATION_SECONDS == 21600
    assert recovery.RESCORE_SECONDS == 900
    assert "6000" not in native["job_command"]


def test_completed_attribution_or_manual_passed_boolean_is_not_readiness(tmp_path):
    path = tmp_path / "not-readiness.json"
    write_json(path, {"version": 1, "status": "passed", "diagnostic_completed": True})
    with pytest.raises(ValueError, match="actual native"):
        recovery.validate_readiness(recovery.reference(path), {})


def test_incomplete_fd_directions_are_not_ready():
    policy = {"epsilons": [.001, .003]}
    with pytest.raises(ValueError, match="complete unchanged-policy"):
        recovery._completed_fd({"passed": True, "policy": policy, "direction_results": {"gradient": {"passed": True}}}, policy)


def test_fd_readiness_requires_all_original_steps_even_when_direction_flags_pass():
    policy = {"epsilons": [.001, .003]}
    item = {"passed": True, "policy": policy,
        "direction_results": {label: {"passed": True} for label in ("gradient", "random_1", "random_2", "random_3")},
        "finite_differences": [{"direction": label, "epsilon": .001}
            for label in ("gradient", "random_1", "random_2", "random_3")]}
    with pytest.raises(ValueError, match="prescribed FD rows"):
        recovery._completed_fd(item, policy)


def test_failure_plan_cannot_belong_to_another_production_release(tmp_path):
    parent_path, failed_path = tmp_path / "parent.json", tmp_path / "failed.json"
    write_json(parent_path, {"version": 1, "cases": [{"id": f"prior-{i}"} for i in range(10)],
                            "source_archive_sha256": "a" * 64, "source_manifest_sha256": "b" * 64})
    write_json(failed_path, {"version": 1, "cases": [{"id": i} for i in recovery.FAILURE_CASES],
                            "production_archive_sha256": "c" * 64, "production_manifest_sha256": "b" * 64})
    with pytest.raises(ValueError, match="original full cohort"):
        recovery.original_plans(recovery.reference(parent_path), recovery.reference(failed_path))


@pytest.mark.parametrize("drift", ["yaml_temperature", "resolved_objective_scale"])
def test_saved_state_readiness_rejects_changed_scientific_configuration(tmp_path, drift):
    original_path, replay_path = tmp_path / "original.yaml", tmp_path / "replay.yaml"
    raw = _raw()
    raw["temperature"] = 1.
    original_path.write_text(yaml.safe_dump(raw))
    if drift == "yaml_temperature":
        raw["temperature"] = .5
    replay_path.write_text(yaml.safe_dump(raw))
    original = {"config_path": str(original_path), "config_file_sha256": recovery.reference(original_path)["sha256"],
                "objective": {"scale": 1.}, "name": "old", "output_root": "/old"}
    replayed = {**original, "config_path": str(replay_path),
                "config_file_sha256": recovery.reference(replay_path)["sha256"], "name": "new", "output_root": "/new"}
    if drift == "resolved_objective_scale":
        replayed["objective"] = {"scale": 2.}
    with pytest.raises(ValueError, match="scientific configuration"):
        recovery._matching_replay_config({"config": original}, {"config": {"replayed_config": replayed}})


def _native_summary(*, invalid_start=False, invalid_endpoint=False):
    both_valid = not invalid_start and not invalid_endpoint
    predictions = 2 * (int(not invalid_start) + int(not invalid_endpoint))
    return {"failed_predictions": 0, "identity_ineligible_predictions": 0,
        "retention_eligible_predictions": predictions, "prediction_attempts": predictions,
        "selection": {"arms": {"fixed_initial_mask": {
            "start": {"smiles": None if invalid_start else "C", "failure": "sanitization" if invalid_start else None},
            "endpoint": {"smiles": None if invalid_endpoint else "CC", "failure": "sanitization" if invalid_endpoint else None}}}},
        "comparisons": {"fixed_initial_mask": {"identity_comparable": both_valid, "paired_seeds": [
            {"seed": seed, "paired_retention_eligible": both_valid} for seed in (17, 23)]}}}


@pytest.mark.parametrize("kind", ["prediction_failed", "identity_ineligible", "incomplete_evaluation"])
def test_completed_rescore_process_is_not_always_a_usable_comparison(kind):
    summary = _native_summary()
    if kind == "prediction_failed":
        summary["failed_predictions"] = 1
    elif kind == "identity_ineligible":
        summary["identity_ineligible_predictions"] = 1
    else:
        summary["comparisons"]["fixed_initial_mask"]["paired_seeds"][0]["paired_retention_eligible"] = False
    result = recovery.fresh_pose_outcome(summary, (17, 23))
    assert result["process_completed"] and result["status"] == kind
    assert not result["computational_completion"]


@pytest.mark.parametrize("invalid_endpoint", [False, True])
def test_invalid_start_is_unscorable_without_fabricating_an_execution_failure(invalid_endpoint):
    result = recovery.fresh_pose_outcome(_native_summary(invalid_start=True, invalid_endpoint=invalid_endpoint), (17, 23))
    assert result["status"] == "completed" and result["computational_completion"]
    assert result["comparison_status"] == "unscorable" and result["eligible_pair_count"] == 0
    assert result["raw_failures"]["start"] == "sanitization"
    assert result["prediction_attempts"] == (0 if invalid_endpoint else 2)


def test_valid_endpoint_prediction_cannot_be_omitted_under_invalid_start_skip():
    summary = _native_summary(invalid_start=True)
    summary["prediction_attempts"] = 0
    summary["retention_eligible_predictions"] = 0
    assert recovery.fresh_pose_outcome(summary, (17, 23))["status"] == "incomplete_evaluation"


def _controller(monkeypatch, tmp_path, *, outcome="completed", rescore_fails=False, invalid_start=False):
    import latentmol.repair_guidance.random_study as study
    import latentmol.boltz.workflow_process as process
    import latentmol.boltz.stability_study as stability
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_raw()))
    template = tmp_path / "template.yaml"
    template.write_text("version: 1\n")
    p = {"case": {"id": "prior-0", "config": recovery.reference(config), "native_template": recovery.reference(template)},
        "name": "fixture-single", "output_root": str(tmp_path / "out"), "cache": str(tmp_path / "cache"),
        "source_manifest_sha256": "a" * 64, "source_archive_sha256": "b" * 64,
        "allocation_seconds": 27000}
    monkeypatch.setenv("LSB_JOBID", "123")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "b" * 64)
    monkeypatch.setattr(recovery, "source_manifest", lambda *args: {"sha256": "a" * 64})
    monkeypatch.setattr(recovery, "load_plan", lambda *args: p)
    monkeypatch.setattr(recovery, "validate_readiness", lambda *args: {"readiness_fixture": True})
    child = tmp_path / "child"
    child.mkdir()
    write_json(child / "run.json", {"run_id": "child"})
    native = tmp_path / "native"
    native.mkdir()
    write_json(native / "summary.json", _native_summary(invalid_start=invalid_start))
    calls = []
    def optimize(case, parent, label, identity, timeout):
        calls.append(timeout)
        if outcome == "failed":
            raise RuntimeError("fixture optimizer failure")
        return {"status": outcome, "accepted_updates": 100 if outcome == "completed" else 12,
                "repair_evidence": {"repair_validation_passed": True}, "child_run": str(child)}
    def logged(command, log, timeout_seconds):
        calls.append(timeout_seconds)
        if "latentmol.gpu_process_probe" in command:
            write_json(Path(command[-1]), {"status": "passed"})
        return {"timed_out": False, "returncode": 0, "run_paths": [str(native)]}
    def verified(*args, **kwargs):
        if rescore_fails:
            raise RuntimeError("fixture rescore failure")
        return native
    monkeypatch.setattr(study, "_optimize", optimize)
    monkeypatch.setattr(process, "run_logged", logged)
    monkeypatch.setattr(stability, "verified_child", verified)
    return calls


@pytest.mark.parametrize("outcome", ["failed", "stalled"])
def test_failed_or_incomplete_case_makes_controller_fail_and_retains_receipt(monkeypatch, tmp_path, outcome):
    calls = _controller(monkeypatch, tmp_path, outcome=outcome)
    with pytest.raises(RuntimeError):
        recovery.run(Path("/plan"), "c" * 64, {})
    root = next((tmp_path / "out/controller").iterdir())
    report = recovery.read(root / "recovery.json")
    assert not report["completed_requested_case"]
    assert report["case"]["status"] == "failed"
    assert recovery.read(root / "run.json")["status"] == "failed"
    assert 21600 in calls and 900 not in calls


def test_rescore_failure_is_not_silently_successful_optimization(monkeypatch, tmp_path):
    calls = _controller(monkeypatch, tmp_path, rescore_fails=True)
    with pytest.raises(RuntimeError, match="rescore"):
        recovery.run(Path("/plan"), "c" * 64, {})
    root = next((tmp_path / "out/controller").iterdir())
    assert recovery.read(root / "run.json")["status"] == "failed"
    assert recovery.read(root / "recovery.json")["case"]["optimization"]["completed_requested_budget"]
    assert 21600 in calls and 900 in calls


def test_complete_case_and_rescore_are_required_for_success(monkeypatch, tmp_path):
    calls = _controller(monkeypatch, tmp_path)
    root = recovery.run(Path("/plan"), "c" * 64, {})
    report = recovery.read(root / "recovery.json")
    assert report["completed_requested_case"]
    assert report["case"]["fresh_pose"]["status"] == "completed"
    assert calls == [180, 21600, 900]


def test_full_computational_recovery_can_finish_with_unscorable_comparison(monkeypatch, tmp_path):
    _controller(monkeypatch, tmp_path, invalid_start=True)
    root = recovery.run(Path("/plan"), "c" * 64, {})
    report = recovery.read(root / "recovery.json")
    assert report["computational_completion"] and report["completed_requested_case"]
    assert report["comparison_status"] == "unscorable"
    assert report["case"]["fresh_pose"]["eligible_pair_count"] == 0
    assert report["case"]["optimization"]["completed_requested_budget"]
    assert recovery.read(root / "run.json")["status"] == "completed"
