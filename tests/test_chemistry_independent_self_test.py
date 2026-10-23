"""Allocated metadata fixtures: independent weak proof never queues training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
import chemistry_independent_self_test as controller
from test_chemistry_short_continuation import baseline_plan


def _fixture(monkeypatch, tmp_path, *, purpose="self_guidance_test", iteration=False, quarter=False):
    work = tmp_path / "work"
    release, root = work / "releases" / "fixture", work / "runs" / "independent-self"
    release.mkdir(parents=True)
    root.mkdir(parents=True)
    plan = baseline_plan(root, purpose)
    if iteration:
        plan["self_guidance_iteration"] = dict(version="epoch36-valid-self-iteration-v1", weight=.1)
        plan["protocol"]["valid_prior_self_guidance"]["weight"] = .1
    if quarter or iteration:
        plan["optimizer_step_control"] = dict(version="adamw-displacement-scale-v1", factor=.25)
    plan.update(work_root=str(work), release=str(release), worker_startup={"version": "chemistry-worker-startup-v1"},
        loss_v1_reference=dict(module=str(release / "old-loss.py"), sha256="e" * 64))
    path = work / "plan.json"
    path.write_text(json.dumps(plan))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    def init(self, path, sha):
        self.path, self.sha, self.plan = Path(path), sha, deepcopy(plan)
        self.work, self.release, self.root = work, release, root
        self.identity = dict(plan_sha256=sha, archive_sha256=plan["archive_sha256"],
                             manifest_sha256=plan["manifest_sha256"])
        self.policy = dict(plan["guarded_restart"])
    monkeypatch.setattr(controller.GuardedRestart, "__init__", init)
    campaign = controller.IndependentSelfTest(path, sha)
    events, queued = [], []
    campaign.require_cpu = lambda: events.append("cpu")
    campaign.require_predecessor_stopped = lambda: events.append("predecessor")
    campaign.worker = lambda command, label: command
    def queue(stage, command, **options):
        queued.append((stage, command, options))
        return "2001"
    campaign.queue = queue
    monkeypatch.setattr(controller, "quota_receipt", lambda minimum: events.append("quota") or {"minimum_gib": minimum})
    monkeypatch.setattr(controller, "authenticated", lambda path, sha: events.append("reference"))
    return campaign, events, queued


def test_constructor_rejects_baseline_training_ownership(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="disposable weak-self"):
        _fixture(monkeypatch, tmp_path, purpose="baseline_training")


def test_failed_baseline_is_not_a_dependency_and_gpu_has_no_training_callback(monkeypatch, tmp_path):
    campaign, events, queued = _fixture(monkeypatch, tmp_path)
    failed_baseline = campaign.work / "runs" / "failed-baseline"
    failed_baseline.mkdir()
    (failed_baseline / "stability_gate.json").write_text('{"status":"failed"}')
    result = campaign.after_cpu()
    assert events == ["cpu", "predecessor", "quota", "reference"]
    assert len(queued) == 1 and queued[0][0] == "independent-self-test"
    assert queued[0][2] == {"gpu": True, "limit": "24:00"}
    command = queued[0][1][-1]
    assert "latentmol.chemistry_guarded_validation" in command
    assert str(campaign.path) in command and campaign.sha in command
    assert "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1" in command
    assert "tests/test_chemistry_valid_self_training.py" in command
    assert " train " not in command and " advance " not in command
    assert result["validation_updates"] == 2500 and result["training"] == "forbidden"
    assert result["baseline_success_required"] is False
    assert result["training_callback"] == "none" and result["promotable"] is False
    assert campaign.after_cpu() == result and len(queued) == 1


@pytest.mark.parametrize("stage", ["benchmark", "segment-001", "final", "independent-self-test"])
def test_no_successful_weak_proof_can_advance_or_train(monkeypatch, tmp_path, stage):
    campaign, _, queued = _fixture(monkeypatch, tmp_path)
    (campaign.root / "stability_gate.json").write_text('{"status":"passed"}')
    with pytest.raises(ValueError, match="cannot promote or train"):
        campaign.advance(stage)
    assert not queued


@pytest.mark.parametrize("gate", ["cpu", "predecessor", "quota", "reference"])
def test_own_cpu_source_quota_or_predecessor_failure_blocks_gpu_and_is_retained(monkeypatch, tmp_path, gate):
    campaign, _, queued = _fixture(monkeypatch, tmp_path)
    def fail(*args):
        raise ValueError("Injected " + gate)
    if gate == "cpu":
        campaign.require_cpu = fail
    elif gate == "predecessor":
        campaign.require_predecessor_stopped = fail
    elif gate == "quota":
        monkeypatch.setattr(controller, "quota_receipt", fail)
    else:
        monkeypatch.setattr(controller, "authenticated", fail)
    with pytest.raises(ValueError, match="Injected"):
        campaign.after_cpu()
    assert not queued
    saved = json.loads((campaign.root / "after-cpu-decision.json").read_text())
    assert saved["status"] == "stopped" and saved["training"] == "forbidden"
    assert campaign.after_cpu() == saved


def test_active_phase_or_uncertain_existing_intent_prevents_duplicate_job(monkeypatch, tmp_path):
    campaign, _, queued = _fixture(monkeypatch, tmp_path)
    (campaign.root / "active-phase.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="phase lock"):
        campaign.after_cpu()
    assert not queued
    (campaign.root / "after-cpu-decision.json").unlink()
    (campaign.root / "active-phase.json").unlink()
    with pytest.raises(FileExistsError):
        campaign.after_cpu()
    assert not queued


def test_wrong_plan_decision_is_never_reused(monkeypatch, tmp_path):
    campaign, _, queued = _fixture(monkeypatch, tmp_path)
    (campaign.root / "after-cpu-decision.json").write_text('{"identity":{"other":"plan"}}')
    with pytest.raises(ValueError, match="different sealed"):
        campaign.after_cpu()
    assert not queued


def test_cpu_commands_reuse_all_original_and_weak_fixtures_without_duplicate_paths(monkeypatch, tmp_path):
    campaign, _, _ = _fixture(monkeypatch, tmp_path)
    command = campaign.cpu_contract_command("python")
    assert all(path in command for path in controller.FIXTURES)
    assert len(command) == len(set(command))
    for mode in ("cpu", "after-cpu"):
        argv = campaign.controller(mode)
        assert "scripts/hpc/chemistry_independent_self_test.py" in argv
        assert argv[-1] == mode
    with pytest.raises(ValueError):
        campaign.controller("advance", after_stage="benchmark")


@pytest.mark.parametrize("iteration", [False, True])
def test_quarter_step_has_mandatory_optimizer_cpu_cuda_and_iteration_dispatch_is_explicit(monkeypatch, tmp_path, iteration):
    campaign, _, queued = _fixture(monkeypatch, tmp_path, quarter=True, iteration=iteration)
    cpu = campaign.cpu_contract_command("python")
    assert "tests/test_chemistry_optimizer_step.py" in cpu
    assert len(cpu) == len(set(cpu))
    result = campaign.after_cpu()
    command = queued[0][1][-1]
    assert "tests/test_chemistry_optimizer_step.py" in command
    assert "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1" in command
    if iteration:
        assert all(path in cpu and path in command for path in controller.ITERATION_FIXTURES)
        assert "latentmol.chemistry_self_strength_validation" in command
        assert "latentmol.chemistry_guarded_validation" not in command
    else:
        assert "latentmol.chemistry_guarded_validation" in command
        assert "latentmol.chemistry_self_strength_validation" not in command
    assert result["training_callback"] == "none" and result["training"] == "forbidden"
    assert len(queued) == 1
