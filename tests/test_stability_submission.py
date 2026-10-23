"""HPC-only submission fixtures; fake scheduler calls never allocate real jobs."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


def helper():
    path = Path(__file__).resolve().parents[1] / "scripts/hpc/submit_stability_study.py"
    spec = importlib.util.spec_from_file_location("stability_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(module, tmp_path):
    return module.workflow(tmp_path / "release", tmp_path, tmp_path / "protocol.json", "a" * 64,
                           archive_sha="b" * 64, manifest_sha="c" * 64)


def test_plan_uses_serial_gate_dependencies_and_bounded_resources(tmp_path):
    module = helper(); rows = plan(module, tmp_path)
    assert [r["dependency"] for r in rows] == [None, "done:cpu", "done:smoke", "ended:primary"]
    assert rows[0]["arguments"][rows[0]["arguments"].index("-q") + 1] == "hpc"
    assert all(r["arguments"][r["arguments"].index("-q") + 1] == "gpul40s" for r in rows[1:])
    assert [r["arguments"][r["arguments"].index("-W") + 1] for r in rows] == ["0:30", "0:45", "23:00", "23:00"]
    assert rows[-1]["job_command"][-1] == "--resume"
    assert rows[0]["job_command"][:3] == ["env", f"LATENTMOL_SOURCE_ARCHIVE={tmp_path}/release.tar.gz",
                                        f"LATENTMOL_WORK_ROOT={tmp_path}"]
    assert "scripts/verify_stability_release.py" in rows[0]["job_command"]


def test_submission_receipts_are_idempotent_and_continuation_requires_successful_smoke(tmp_path, monkeypatch):
    module = helper(); calls = []
    def bsub(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=f"Job <{100 + len(calls)}> is submitted to queue <hpc>.\n", stderr="")
    monkeypatch.setattr(module.subprocess, "run", bsub)
    rows = plan(module, tmp_path); receipts = tmp_path / "receipts"; identity = {"protocol_sha": "a" * 64}
    jobs = module.submit(rows, receipts, identity)
    assert len(calls) == 4 and jobs["continuation"] == "104"
    assert calls[3][calls[3].index("-w") + 1] == "ended(103) && done(102)"
    assert module.submit(rows, receipts, identity) == jobs and len(calls) == 4
    with pytest.raises(ValueError): module.submit(rows, receipts, {"protocol_sha": "b" * 64})
    assert len(calls) == 4


@pytest.mark.parametrize("failure", ["timeout", "unclear_reply", "nonzero_reply"])
def test_unknown_submission_cannot_be_retried_blindly(tmp_path, monkeypatch, failure):
    module = helper(); calls = []
    def bsub(command, **kwargs):
        calls.append(command)
        if failure == "timeout": raise subprocess.TimeoutExpired(command, 60)
        return SimpleNamespace(returncode=1 if failure == "nonzero_reply" else 0,
                               stdout="unrecognized scheduler reply", stderr="fixture")
    monkeypatch.setattr(module.subprocess, "run", bsub)
    receipts = tmp_path / "receipts"; rows = plan(module, tmp_path)
    with pytest.raises(RuntimeError): module.submit(rows, receipts, {})
    with pytest.raises(FileExistsError): module.submit(rows, receipts, {})
    assert len(calls) == 1 and (receipts / "cpu-intent.json").is_file()


def test_protocol_preparation_recovers_exact_metadata_but_rejects_changed_identity(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_stability_protocol.py"
    spec = importlib.util.spec_from_file_location("stability_preparation", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    target = tmp_path / "protocol.json"
    module.save_exact(target, {"identity": "original"})
    before = target.read_bytes(); module.save_exact(target, {"identity": "original"})
    assert before == target.read_bytes()
    with pytest.raises(ValueError): module.save_exact(target, {"identity": "changed"})
    assert json.loads(target.read_text())["identity"] == "original"


def test_external_dependencies_are_conjoined_and_receipts_remain_idempotent(tmp_path, monkeypatch):
    module = helper(); calls = []
    def bsub(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=f"Job <{100 + len(calls)}> is submitted to queue <hpc>.\n", stderr="")
    monkeypatch.setattr(module.subprocess, "run", bsub)
    rows = plan(module, tmp_path)
    rows[1]["external_dependencies"] = [{"kind": "done", "job_id": "29728602"}]
    receipts, identity = tmp_path / "receipts", {"source": "fixture"}
    original = module.submit(rows, receipts, identity)
    assert calls[1][calls[1].index("-w") + 1] == "done(101) && done(29728602)"
    assert module.submit(rows, receipts, identity) == original and len(calls) == 4
    rows[1]["external_dependencies"][0]["job_id"] = "29728603"
    with pytest.raises(ValueError, match="does not match"):
        module.submit(rows, receipts, identity)
    assert len(calls) == 4


@pytest.mark.parametrize("external", [None, {}, [{"kind": "done", "job_id": "1) || ended(2"}],
    [{"kind": "done", "job_id": "１２３"}], [{"kind": "done", "job_id": "0"}],
    [{"kind": "done", "job_id": 123}], [{"kind": "done", "job_id": True}],
    [{"kind": "exit", "job_id": "123"}], [{"kind": "done", "job_id": "123", "extra": True}]])
def test_invalid_external_dependencies_fail_before_any_submission(tmp_path, monkeypatch, external):
    module = helper(); calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    rows = plan(module, tmp_path); rows[1]["external_dependencies"] = external
    receipts = tmp_path / "receipts"
    with pytest.raises(ValueError, match="External|external"):
        module.submit(rows, receipts, {})
    assert calls == [] and not receipts.exists()


@pytest.mark.parametrize("dependency", ["done) || ended:cpu", "done:missing", "done:cpu:extra", True])
def test_invalid_internal_dependencies_fail_before_any_submission(tmp_path, monkeypatch, dependency):
    module = helper(); calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    rows = plan(module, tmp_path); rows[1]["dependency"] = dependency
    with pytest.raises(ValueError, match="internal"):
        module.submit(rows, tmp_path / "receipts", {})
    assert calls == []
