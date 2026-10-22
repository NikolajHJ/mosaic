"""Allocated transition contracts; scientific success is never inferred from a callback."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = _repository_root(__file__)
sys.path.insert(0, str(_layout_path(ROOT / "scripts")))
spec = importlib.util.spec_from_file_location("campaign25_pocket_ops", _layout_path(ROOT / "scripts/campaign25_pocket_ops.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def launcher_pins(monkeypatch):
    monkeypatch.setenv("MOSAIC25_LAUNCHER_MANIFEST", "c" * 64)
    monkeypatch.setenv("MOSAIC25_LAUNCHER_CERT_SHA", "d" * 64)


@pytest.mark.parametrize("status", ["DONE", "D\n ONE", "EXIT", "E\nXIT"])
def test_actual_scheduler_fields_accept_lsf_line_wrap_only(status):
    result, queue = module.parse_final(f"Job <29700001>, Status <{status}>, Queue <gpu\n l40s>", "29700001")
    assert result in ("DONE", "EXIT") and queue == "gpul40s"


@pytest.mark.parametrize("text", ["Job <29700002>, Status <DONE>", "Job <29700001>, Status <RUN>", "Job <29700001>, Status <PEND>", "Job not found", "Job <29700001>, Status <DONE_FAKE>"])
def test_no_aged_out_pending_wrong_job_or_fabricated_done(text):
    with pytest.raises(ValueError):
        module.parse_final(text, "29700001")


@pytest.mark.parametrize("arm,queue", [("protenix", "gpul40s"), ("boltz2", "gpul40s"), ("boltz2_chemistry", "gpul40s")])
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


def test_unknown_arm_rejected_before_creating_run_or_calling_scheduler(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Scheduler must not be called")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="Unknown"):
        module.submit(tmp_path / "source", tmp_path, "a" * 64, "run", arm="unknown")
    assert not (tmp_path / "run-unknown").exists()


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
    assert record["held_arms"] == []
    assert record["chemistry_authorization"]["automatic_long_submission"] is True



def test_unknown_previous_baseline_submission_blocks_retry(tmp_path):
    module.write(tmp_path / "long-dispatch-started-protenix.json", {})
    with pytest.raises(RuntimeError, match="no duplicate"):
        module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)


def test_approved_chemistry_included_in_independent_dispatch_and_whole_campaign_quota(tmp_path, monkeypatch):
    for arm in module.ARMS:
        module.write(tmp_path / ("decision-" + arm + ".json"), {"arm": arm, "status": "passed", "projected_bytes": 1024**3})
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="You are using 1.00G of 30.00G quota", returncode=0))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=20 * 1024**3))
    actual = []
    def submit(*a, **k):
        actual.append(k["arm"])
        return {"job_id": "29700003", "arm": k["arm"]}
    monkeypatch.setattr(module, "submit", submit)
    module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)
    assert actual == list(module.ARMS)
    gate = module.read(tmp_path / "campaign-storage-gate-protenix-boltz2-boltz2_chemistry.json")
    assert gate["required_bytes"] == 7 * 1024**3
    assert module.read(tmp_path / "long-submissions.json")["held_arms"] == []


def test_user_approval_never_promotes_failed_chemistry(tmp_path, monkeypatch):
    module.write(tmp_path / "decision-boltz2_chemistry.json", {"arm": "boltz2_chemistry", "status": "failed"})
    def forbidden(*a, **k):
        raise AssertionError("Failed chemistry must not call scheduler/quota/submit")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(module, "submit", forbidden)
    module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)
    assert module.read(tmp_path / "long-submissions.json")["jobs"] == []


@pytest.mark.parametrize("final", ["DONE", "EXIT"])
def test_stale_run_is_recorded_then_actual_final_required(tmp_path, monkeypatch, final):
    (tmp_path / "stdout.29700001.log").write_text("Subject: Job 29700001:\nSuccessfully completed.\n")
    values = iter(["RUN", final])
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=f"Job <29700001>, Status <{next(values)}>, Queue <gpul40s>", stderr=""))
    waits = []
    monkeypatch.setattr(module.time, "sleep", waits.append)
    status, queue, proofs = module.final_job("29700001", tmp_path)
    assert status == final and queue == "gpul40s" and waits == [10]
    assert len(list(tmp_path.glob("scheduler-final-*.txt"))) == 2
    assert len(proofs) == 2


def test_never_promote_stale_run_to_done_after_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="Job <29700001>, Status <RUN>, Queue <gpul40s>", stderr=""))
    monkeypatch.setattr(module.time, "sleep", lambda *a: None)
    with pytest.raises(ValueError, match="not ended"):
        module.final_job("29700001", tmp_path)
    assert len(list(tmp_path.glob("scheduler-final-*.txt"))) == 7


def test_done_still_requires_actual_lsf_success(tmp_path, monkeypatch):
    (tmp_path / "stdout.29700001.log").write_text("Subject: Job 29700001:\nExited with exit code 1\n")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="Job <29700001>, Status <DONE>, Queue <gpul40s>", stderr=""))
    with pytest.raises(ValueError, match="LSF"):
        module.final_job("29700001", tmp_path)


def test_line_wrapped_field_name_is_parsed_without_changing_identity():
    assert module.parse_final("Job <29700001>, Sta\n tus <DONE>, Que\nue <gpu\nl40s>", "29700001") == ("DONE", "gpul40s")


@pytest.mark.parametrize("protenix_only", [False, True])
def test_retry_scope_does_not_duplicate_live_boltz_preparation(tmp_path, monkeypatch, protenix_only):
    arms = ["protenix"] if protenix_only else list(module.ARMS)
    module.write(tmp_path / "dispatch-started.json", {"arms": arms})
    calls = []
    def submit(source, root, manifest, phase, **kwargs):
        calls.append((phase, kwargs.get("arm", "none")))
        return {"job_id": str(29700000 + len(calls))}
    monkeypatch.setattr(module, "submit", submit)
    module.start_validations(tmp_path / "source", tmp_path, "a" * 64)
    expected = [("validate", "protenix"), ("after-ready", "protenix")]
    if not protenix_only:
        expected += [("prepare", "none"), ("after-prepare", "none")]
    assert calls == expected
    assert module.read(tmp_path / "initial-submissions.json")["arms"] == arms


@pytest.mark.parametrize("arms", [[], ["boltz2"], ["protenix", "protenix"], ["unknown"]])
def test_invalid_retry_scope_fails_before_any_scheduler_call(tmp_path, monkeypatch, arms):
    module.write(tmp_path / "dispatch-started.json", {"arms": arms})
    def forbidden(*args, **kwargs):
        raise AssertionError("No submission from an invalid scope")
    monkeypatch.setattr(module, "submit", forbidden)
    with pytest.raises(ValueError, match="scope"):
        module.start_validations(tmp_path / "source", tmp_path, "a" * 64)


def test_protenix_retry_dispatch_never_promotes_or_waits_for_other_arms(tmp_path, monkeypatch):
    module.write(tmp_path / "dispatch-started.json", {"arms": ["protenix"]})
    module.write(tmp_path / "decision-protenix.json", {"arm": "protenix", "status": "failed"})
    module.write(tmp_path / "decision-boltz2.json", {"arm": "boltz2", "status": "passed", "projected_bytes": 1})
    def forbidden(*args, **kwargs):
        raise AssertionError("Out-of-scope Boltz decision cannot dispatch")
    monkeypatch.setattr(module, "submit", forbidden)
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    module.dispatch_finished(tmp_path / "source", tmp_path, "a" * 64)
    result = module.read(tmp_path / "long-submissions.json")
    assert result["jobs"] == [] and result["pending_readiness_arms"] == []
    assert [d["arm"] for d in result["failed_arms"]] == ["protenix"]
