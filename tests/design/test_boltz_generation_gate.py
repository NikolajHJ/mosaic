"""The conditional BRD4 gate uses pilot outcomes once and preserves job receipts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PATH = (Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae"
        / "restart-operations/arm_boltz_generations.py")
spec = importlib.util.spec_from_file_location("brd4_gate", PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_only_six_selected_jobs_once_and_no_submission_without_decision(tmp_path, monkeypatch):
    commands = {}
    for arm in gate.ARMS:
        (tmp_path / arm).mkdir()
        commands[arm] = ["bsub", "-J", arm, "-q", "testing", "runner", "--output", f"/runs/{arm}"]
    manifest = dict(control=str(tmp_path), commands=commands)
    monkeypatch.setattr(gate, "validated_results", lambda manifest: ["outcomes"])
    selector = SimpleNamespace(select_policies=lambda rows: dict(status="inconclusive", selected_policies=[]))
    monkeypatch.setitem(sys.modules, "boltz_policy_selection", selector)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, flock=lambda fd, mode: None))
    calls = []

    def bsub(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=f"Job <{1000 + len(calls)}> is submitted to queue <testing>.",
                               stderr="")

    monkeypatch.setattr(gate.subprocess, "run", bsub)
    assert gate.dispatch(manifest) == dict(status="inconclusive", jobs={})
    assert not calls
    # A fresh artifact root represents the alternative observed outcome.
    (tmp_path / "gate-decision.json").unlink()
    selector.select_policies = lambda rows: dict(status="selected", selected_policies=["fixed150", "extend"])
    first = gate.dispatch(manifest)
    assert first == gate.dispatch(manifest)
    assert set(first["jobs"]) == {f"{k}-{s}" for k in ("fixed150", "extend") for s in gate.PANELS}
    assert len(calls) == 6


def test_lost_submission_reply_reconciles_exact_scheduler_command(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, flock=lambda fd, mode: None))
    argv = ["bsub", "-J", "job-name", "-q", "test", "/usr/bin/python3", "/control/launch.py", "--sha", "abcdef"]
    gate.publish(tmp_path / "submit-intent.json", argv)
    response = SimpleNamespace(returncode=0, stdout="718 job-name test /usr/bin/python3 /control/launch.py --sha stale")
    calls = []

    def status(query, **kwargs):
        calls.append(query[0])
        return response

    monkeypatch.setattr(gate.subprocess, "run", status)
    with pytest.raises(RuntimeError, match="refused to repeat"):
        gate.submit_once(tmp_path, argv)
    response.stdout = response.stdout.replace("stale", "abcdef")
    row = gate.submit_once(tmp_path, argv)
    assert row["job_id"] == 718 and row["reconciled"]
    assert calls == ["bjobs", "bjobs"]
    assert json.loads((tmp_path / "submission.json").read_text()) == row
