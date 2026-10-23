"""Allocated metadata fixtures for serial, nonpromoting diagnostic sequencing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
import chemistry_collapse_suite as suite
import chemistry_short_campaign as short_controller
from test_chemistry_short_campaign import fixture as short_fixture


def _fixture(monkeypatch, tmp_path):
    base, events, queued = short_fixture(monkeypatch, tmp_path)
    control = dict(version="adamw-displacement-scale-v1", factor=.25)
    base.plan["optimizer_step_control"] = dict(control)
    base.test["optimizer_step_control"] = dict(control)
    base.test_path.write_text(json.dumps(base.test))
    base.test_sha = hashlib.sha256(base.test_path.read_bytes()).hexdigest()
    base.plan["self_test_plan"]["sha256"] = base.test_sha
    base.plan["collapse_diagnostic"] = dict(version="epoch36-collapse-finite-step-v1", updates=100,
        instrument_updates=[39, 96], source_rows=dict(path="/work3/fixture.jsonl", sha256="f" * 64))
    base.policy = dict(base.plan["guarded_restart"])
    monkeypatch.setattr(suite.ShortCampaign, "__init__", lambda self, path, sha: self.__dict__.update(base.__dict__))
    campaign = suite.CollapseSuite(base.path, base.sha)
    monkeypatch.setattr(suite, "quota_receipt", lambda minimum: events.append("quota") or {"minimum_gib": minimum})
    monkeypatch.setattr(suite, "authenticated", lambda path, sha: events.append("reference"))
    campaign.receipt_job = lambda stage: next(str(2001 + i) for i, row in enumerate(queued) if row[0] == stage)
    monkeypatch.setattr(suite.time, "sleep", lambda _: None)
    return campaign, events, queued


def _terminal(monkeypatch, state="DONE", *, stderr="", returncode=0):
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=state, stderr=stderr, returncode=returncode)
    monkeypatch.setattr(suite.subprocess, "run", query)
    return calls


def _mechanism(campaign, *, status="completed_disposable_not_training", completed=True, restore=True):
    root = campaign.root / "collapse_diagnostic"
    run = root / "fixture"
    run.mkdir(parents=True)
    report = dict(version="epoch36-collapse-finite-step-v1", status=status, diagnostic_completed=completed,
        restoration=dict(passed=restore), plan_sha256=campaign.sha, archive_sha256=campaign.plan["archive_sha256"],
        manifest_sha256=campaign.plan["manifest_sha256"], job_id="2001", run_dir=str(run),
        training_updates_promoted=0, promotable=False, epoch_complete=False,
        policy=deepcopy(campaign.plan["collapse_diagnostic"]),
        updates=[dict(relative_update=index, historical_receipt_matches=True) for index in range(1, 101)],
        spikes=[dict(relative_update=39), dict(relative_update=96)])
    (campaign.root / "collapse_diagnostic.json").write_text(json.dumps(report))
    (run / "diagnostic.json").write_text(json.dumps(report))
    (run / "run.json").write_text(json.dumps(dict(status="completed" if completed else "failed",
        environment=dict(job_id="2001"), source_archive_sha256=campaign.plan["archive_sha256"],
        source=dict(sha256=campaign.plan["manifest_sha256"]))))
    return report


def test_cpu_then_two_hour_mechanism_precedes_new_separate24hour_proof(monkeypatch, tmp_path):
    c, events, queued = _fixture(monkeypatch, tmp_path)
    first = c.after_cpu()
    assert [row[0] for row in queued] == ["collapse-mechanism", "after-mechanism"]
    assert queued[0][2] == dict(gpu=True, limit="2:00")
    assert queued[1][2] == dict(dependency="ended(2001)")
    command = queued[0][1][-1]
    assert "latentmol.chemistry_collapse_diagnostic" in command and c.sha in command
    assert "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1" in command
    assert "tests/test_chemistry_collapse_diagnostic.py" in command
    assert "tests/test_chemistry_optimizer_step.py" in command
    assert command.index("pytest") < command.index(" && ") < command.index("latentmol.chemistry_collapse_diagnostic")
    assert first["mechanism_seconds"] == 6600 and first["disposable"]
    assert "benchmark" not in [row[0] for row in queued]
    assert c.after_cpu() == first and len(queued) == 2
    _terminal(monkeypatch); _mechanism(c)
    second = c.after_mechanism()
    assert [row[0] for row in queued] == ["collapse-mechanism", "after-mechanism", "benchmark", "after-benchmark"]
    assert queued[2][2] == dict(gpu=True, limit="24:00")
    assert queued[3][2] == dict(dependency="ended(2003)")
    assert second["mechanism"]["passed"] is True and second["validation_updates"] == 2500
    assert "latentmol.chemistry_guarded_validation" in queued[2][1][-1]
    assert "test_chemistry_optimizer_step.py" in queued[2][1][-1]
    assert c.after_mechanism() == second and len(queued) == 4


@pytest.mark.parametrize("state", ["DONE", "EXIT"])
def test_terminal_failed_baseline_does_not_block_independent_disposable_weak_proof(monkeypatch, tmp_path, state):
    c, _, queued = _fixture(monkeypatch, tmp_path)
    queued.append(("benchmark", ["fixture"], {}))
    (c.root / "stability_gate.json").write_text('{"status":"failed"}')
    _terminal(monkeypatch, state)
    # Any accidental attempt to use the baseline-success completion method is
    # a fixture failure. Only terminal sequencing is authorized at this point.
    c.complete = lambda *args: (_ for _ in ()).throw(AssertionError("Baseline success was consulted"))
    result = c.advance("benchmark")
    assert [row[0] for row in queued] == ["benchmark", "self-test", "after-self-test"]
    assert queued[1][2] == dict(gpu=True, limit="24:00")
    assert queued[2][2] == dict(dependency="ended(2002)")
    command = queued[1][1][-1]
    assert str(c.test_path) in command and c.test_sha in command
    assert result["baseline_success_required"] is False and result["weak_training"] == "forbidden"
    assert result["baseline_terminal"]["state"] == state
    assert c.advance("benchmark") == result and len(queued) == 3


def test_failed_disposable_mechanism_can_run_own_weak_gates_but_never_baseline_proof(monkeypatch, tmp_path):
    c, _, queued = _fixture(monkeypatch, tmp_path)
    c.after_cpu(); _terminal(monkeypatch, "EXIT"); _mechanism(c, status="failed", completed=False)
    result = c.after_mechanism()
    assert result["status"] == "mechanism_failed_weak_test_only"
    assert result["mechanism"]["passed"] is False
    assert [row[0] for row in queued] == ["collapse-mechanism", "after-mechanism", "self-test", "after-self-test"]
    assert result["weak"]["weak_training"] == "forbidden"


@pytest.mark.parametrize("stage", ["mechanism", "baseline"])
@pytest.mark.parametrize("state,stderr,returncode", [("RUN", "", 0), ("PEND", "", 0),
    ("", "Job <2001> is not found", 1), ("DONE", "Untrusted status", 0), ("EXIT", "", 1)])
def test_nonterminal_or_uncertain_predecessors_block_all_following_gpu_work(monkeypatch, tmp_path, stage, state, stderr, returncode):
    c, _, queued = _fixture(monkeypatch, tmp_path)
    _terminal(monkeypatch, state, stderr=stderr, returncode=returncode)
    if stage == "mechanism":
        c.after_cpu(); _mechanism(c)
        action = c.after_mechanism
        count = 2
    else:
        queued.append(("benchmark", ["fixture"], {}))
        action = lambda: c.advance("benchmark")
        count = 1
    with pytest.raises(ValueError, match="confirmed terminal"):
        action()
    assert len(queued) == count


@pytest.mark.parametrize("change", [lambda r: r.update(plan_sha256="9" * 64),
    lambda r: r.update(job_id="999"), lambda r: r.update(archive_sha256="9" * 64),
    lambda r: r.update(promotable=True), lambda r: r.update(epoch_complete=True),
    lambda r: r.update(training_updates_promoted=1)])
def test_wrong_mechanism_identity_never_submits_any_followup(monkeypatch, tmp_path, change):
    c, _, queued = _fixture(monkeypatch, tmp_path)
    c.after_cpu(); _terminal(monkeypatch)
    report = _mechanism(c); change(report)
    (c.root / "collapse_diagnostic.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity"):
        c.after_mechanism()
    assert len(queued) == 2


def test_after_weak_restores_original_baseline_promotion_gate_not_disposable_state(monkeypatch, tmp_path):
    c, _, queued = _fixture(monkeypatch, tmp_path)
    queued.append(("benchmark", ["fixture"], {}))
    _terminal(monkeypatch, "EXIT")
    c.advance("benchmark")
    called = []
    def reject(self, stage):
        called.append(stage)
        assert self is c and self.plan["short_continuation"]["purpose"] == "baseline_training"
        raise ValueError("Baseline native/stability proof failed")
    monkeypatch.setattr(suite.GuardedRestart, "advance", reject)
    monkeypatch.setattr(short_controller.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="Baseline native/stability"):
        c.after_self_test()
    assert called == ["benchmark"]
    assert not any(row[0].startswith("segment") for row in queued)


def test_all_new_contracts_are_reused_and_constructor_requires_matched_quarter_policy(monkeypatch, tmp_path):
    c, _, _ = _fixture(monkeypatch, tmp_path)
    command = c.cpu_contract_command("python")
    assert all(path in command for path in suite.FIXTURES)
    assert len(command) == len(set(command))
    c.test["optimizer_step_control"]["factor"] = .5
    with pytest.raises(ValueError, match="same explicit quarter"):
        suite.CollapseSuite(c.path, c.sha)


def test_saved_decision_or_lost_response_cannot_duplicate_submission(monkeypatch, tmp_path):
    c, _, queued = _fixture(monkeypatch, tmp_path)
    c.after_cpu()
    (c.root / "after-cpu-decision.json").unlink()
    with pytest.raises(FileExistsError):
        c.after_cpu()
    assert len(queued) == 2
