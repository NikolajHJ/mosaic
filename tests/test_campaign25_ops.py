"""Allocated transition contracts; scientific success is never inferred from a callback."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("campaign25_ops", ROOT / "scripts/campaign25_ops.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("status", ["DONE", "D\n ONE", "EXIT", "E\nXIT"])
def test_actual_scheduler_fields_accept_lsf_line_wrap_only(status):
    result, queue = module.parse_final(f"Job <29700001>, Status <{status}>, Queue <gpu\n l40s>", "29700001")
    assert result in ("DONE", "EXIT") and queue == "gpul40s"


@pytest.mark.parametrize("text", ["Job <29700002>, Status <DONE>", "Job <29700001>, Status <RUN>", "Job <29700001>, Status <PEND>", "Job not found", "Job <29700001>, Status <DONE_FAKE>"])
def test_no_aged_out_pending_wrong_job_or_fabricated_done(text):
    with pytest.raises(ValueError):
        module.parse_final(text, "29700001")


@pytest.mark.parametrize("arm,queue", [("protenix", "gpua10"), ("boltz2", "gpul40s")])
def test_actual_long_submission_vector_has_matching_gpu_shape_and_no_inter_arm_dependency(tmp_path, monkeypatch, arm, queue):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Job <29700003> is submitted" if command[0] == "bsub" else "Job <29700003>, Status <PEND>", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    result = module.submit(tmp_path / "source", tmp_path, "a" * 64, "run", arm=arm)
    command = calls[0]
    assert command[command.index("-q") + 1] == queue
    assert command[command.index("-n") + 1] == "4"
    assert command[command.index("-gpu") + 1] == "num=1:mode=exclusive_process"
    assert command[command.index("-W") + 1] == "23:00" and "-w" not in command
    assert result["job_id"] == "29700003"


def test_unknown_submission_is_preserved_without_invented_job_id(tmp_path, monkeypatch):
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="Connection closed before receipt", stderr=""))
    with pytest.raises(RuntimeError, match="unknown"):
        module.submit(tmp_path / "source", tmp_path, "a" * 64, "run", arm="boltz2")
    record = module.read(tmp_path / "run-boltz2/submission.json")
    assert "job_id" not in record
    with pytest.raises(FileExistsError):
        module.submit(tmp_path / "source", tmp_path, "a" * 64, "run", arm="boltz2")


@pytest.mark.parametrize("all_failed", [False, True])
def test_only_arms_with_actual_passed_decisions_are_submitted(tmp_path, monkeypatch, all_failed):
    for arm in module.ARMS:
        status = "passed" if not all_failed and arm == "protenix" else "failed"
        module.write(tmp_path / ("decision-" + arm + ".json"), {"arm": arm, "status": status, "projected_bytes": 1024**3})
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="You are using 1.00G of 30.00G quota", returncode=0))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 * 1024**3))
    actual = []
    def submit(*a, **k):
        actual.append(k["arm"])
        return {"job_id": "29700003", "arm": k["arm"]}
    monkeypatch.setattr(module, "submit", submit)
    module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)
    assert actual == ([] if all_failed else ["protenix"])
    assert len(module.read(tmp_path / "long-submissions.json")["jobs"]) == len(actual)


def test_full_chemistry_submission_rejected_before_creating_run_or_calling_scheduler(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Scheduler must not be called")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="held"):
        module.submit(tmp_path / "source", tmp_path, "a" * 64, "run", arm="boltz2_chemistry")
    assert not (tmp_path / "run-boltz2_chemistry").exists()


def test_independent_baseline_can_start_without_other_decisions_and_is_not_duplicated(tmp_path, monkeypatch):
    arm = "boltz2"
    module.write(tmp_path / ("decision-" + arm + ".json"), {"arm": arm, "status": "passed", "projected_bytes": 1024**3})
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="You are using 1.00G of 30.00G quota", returncode=0))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 * 1024**3))
    actual = []
    def submit(*a, **k):
        actual.append(k["arm"])
        parent = tmp_path / ("run-" + k["arm"])
        parent.mkdir(exist_ok=False)
        result = {"job_id": "29700003", "arm": k["arm"], "returncode": 0}
        module.write(parent / "submission.json", result)
        return result
    monkeypatch.setattr(module, "submit", submit)
    module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)
    module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)
    assert actual == ["boltz2"]
    record = module.read(tmp_path / "long-submissions.json")
    assert record["pending_readiness_arms"] == ["protenix", "boltz2_chemistry"]
    assert record["held_arms"][0]["automatic_long_submission"] is False


def test_passed_chemistry_stays_held_and_exposes_saved_pilot_runtime(tmp_path, monkeypatch):
    parent = tmp_path / "validate-boltz2_chemistry/payload/study"
    parent.mkdir(parents=True)
    module.write(parent / "summary.json", {"status": "passed", "runtime_gate": {"status": "passed", "mean_update_seconds": [1, 2], "conservative_projected_seconds": 31800}})
    module.write(tmp_path / "decision-boltz2_chemistry.json", {"arm": "boltz2_chemistry", "status": "passed", "projected_bytes": 1024**3})
    def forbidden(*a, **k):
        raise AssertionError("Chemistry pass must not call scheduler/quota/submit")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(module, "submit", forbidden)
    module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)
    assert module.read(tmp_path / "long-submissions.json")["jobs"] == []
    estimate = module.read(tmp_path / "chemistry-runtime-estimate.json")
    assert estimate["runtime_available"] and estimate["runtime_gate"]["mean_update_seconds"] == [1, 2]
    assert estimate["launch_policy"]["status"] == "held_for_runtime_review"


def test_unknown_previous_baseline_submission_blocks_retry(tmp_path):
    module.write(tmp_path / "long-dispatch-started-protenix.json", {})
    with pytest.raises(RuntimeError, match="no duplicate"):
        module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)
