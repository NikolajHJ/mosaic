"""Metadata lifecycle tests; no scheduler, Torch, JAX or native model calls."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def campaign(tmp_path):
    spec = importlib.util.spec_from_file_location("latent_protenix_campaign_fixture", Path(__file__).with_name("campaign.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = module.Campaign.__new__(module.Campaign)
    c.control, c.root, c.release = tmp_path / "control", tmp_path / "results", tmp_path / "release"
    c.control.mkdir()
    c.receipts, c.path, c.sha = c.control / "receipts", c.control / "plan.json", "a" * 64
    c.plan = dict(source_job="111", source_result="/epoch45/result.json", source_study_sha256="b" * 64,
                  target_updates=50000, pilot_updates=200, archive_sha256="c" * 64,
                  torch_python="/torch/python", mosaic_python="/jax/python", mosaic_config="/mosaic.json",
                  pilot_config="/pilot.json", high_weight_config="/high.json", full_config="/full.json")
    return module, c


def stage_result(c, stage, *, start=0, completed=200):
    return dict(status="completed" if stage == "pilot" else "segment_completed", start_step=start,
                completed_step=completed, updates_this_segment=completed - start,
                elapsed_seconds=200., checkpoint_sha256="d" * 64,
                state=str(c.root / stage / "state.pt"), technical_checks={"finite": True, "frozen": True},
                weight_probes={"1.0": {"finite": True}, "10.0": {"finite": True}})


def test_chunk_budget_includes_margin_and_reserve(campaign):
    module, _ = campaign
    assert module.chunk_budget(200., 200, 50000) == 50000
    assert module.chunk_budget(2000., 200, 50000) == 6276
    assert module.chunk_budget(200., 200, 17) == 17


@pytest.mark.parametrize("elapsed,updates,remaining", [(0., 200, 50), (float("nan"), 200, 50), (1., 0, 50), (1., 200, 0)])
def test_bad_runtime_evidence_never_launches(campaign, elapsed, updates, remaining):
    with pytest.raises(ValueError):
        campaign[0].chunk_budget(elapsed, updates, remaining)


def test_queue_waits_for_epoch45_and_callback_uses_ended(campaign):
    _, c = campaign
    calls = []
    c.submit = lambda stage, args, **kwargs: calls.append((stage, args, kwargs)) or str(200 + len(calls))
    result = c.queue()
    assert result["pilot"] == "201"
    assert calls[0][2] == {"gpu": True, "dependency": "done(111)"}
    assert calls[1][2] == {"dependency": "ended(201)"}


def test_actual_gpu_submission_exclusive_and_idempotent(campaign, monkeypatch):
    module, c = campaign
    calls = []
    def invoke(args, **kwargs):
        calls.append(args)
        return dict(returncode=0, stdout="" if args[0] == "bjobs" else "Job <222> is submitted to queue <gpuv100>.\n", stderr="")
    monkeypatch.setattr(module, "command", invoke)
    assert c.submit("pilot", ["true"], gpu=True) == "222"
    args = next(a for a in calls if a[0] == "bsub")
    assert args[args.index("-gpu") + 1] == "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G"
    assert args[args.index("-W") + 1] == "2:00"
    assert args[args.index("-n") + 1] == "8"
    assert args[args.index("-env") + 1] == "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
    assert c.submit("pilot", ["true"], gpu=True) == "222"
    assert sum(a[0] == "bsub" for a in calls) == 1


def test_callback_submission_strips_previous_gpu_routes(campaign, monkeypatch):
    module, c = campaign
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES1", "3")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/previous-allocation")
    environments = []
    def invoke(args, **kwargs):
        if args[0] == "bsub":
            environments.append(kwargs["env"])
        return dict(returncode=0, stdout="" if args[0] == "bjobs" else "Job <222> is submitted to queue <hpc>.\n", stderr="")
    monkeypatch.setattr(module, "command", invoke)
    c.submit("after-pilot", ["true"], dependency="ended(111)")
    assert len(environments) == 1
    assert "CUDA_VISIBLE_DEVICES" not in environments[0]
    assert "CUDA_VISIBLE_DEVICES1" not in environments[0]
    assert "CUDA_MPS_PIPE_DIRECTORY" not in environments[0]


def test_lost_submission_cannot_be_blindly_retried(campaign, monkeypatch):
    module, c = campaign
    calls = []
    monkeypatch.setattr(module, "command", lambda args, **kwargs: calls.append(args) or dict(returncode=0, stdout="ambiguous" if args[0] == "bsub" else "", stderr=""))
    with pytest.raises(RuntimeError, match="Unknown submission"):
        c.submit("pilot", ["true"], gpu=True)
    with pytest.raises(RuntimeError, match="reconcile"):
        c.submit("pilot", ["true"], gpu=True)
    assert sum(a[0] == "bsub" for a in calls) == 1


def test_passed_pilots_launch_fresh_main_not_pilot_state(campaign):
    _, c = campaign
    c.validated_result = lambda stage, **kwargs: stage_result(c, stage, completed=200 if stage == "pilot" else 20)
    calls = []
    c.queue_main = lambda index, count, resume=None: calls.append((index, count, resume)) or {"stage": "main-0001"}
    c.after_pilot()
    assert calls == [(1, 50000, None)]


def test_weight10_failure_does_not_launch_main(campaign):
    _, c = campaign
    def result(stage, **kwargs):
        if stage == "pilot-weight10":
            raise ValueError("high-weight finite check failed")
        return stage_result(c, stage)
    c.validated_result = result
    c.queue_main = lambda *a, **kw: pytest.fail("No main launch")
    with pytest.raises(ValueError, match="high-weight"):
        c.after_pilot()


def test_source_requires_completed_original45_and_hash(campaign, monkeypatch):
    module, c = campaign
    monkeypatch.setattr(module, "require_done", lambda job: {})
    source = dict(status="completed_epoch", epoch=45, study_sha256="b" * 64,
                  frozen_anchors_unchanged=True, terminal_checks={"repair": True}, checkpoint="/45.pt", checkpoint_sha256="d" * 64)
    c.scratch = lambda path, operation="json": source if operation == "json" else "d" * 64
    assert c.source()["epoch"] == 45
    source["study_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="original checks"):
        c.source()


def test_worker_runs_two_independent_pilots_and_preserves_ready(campaign, monkeypatch):
    module, c = campaign
    monkeypatch.setenv("LSB_JOBID", "999")
    monkeypatch.setenv("DBMOL_LAUNCH_READY", "/ready.json")
    monkeypatch.setenv("DBMOL_LAUNCH_NONCE", "nonce")
    c.source = lambda: dict(path="/45.pt", sha256="d" * 64)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(returncode=0))
    c.run_stage("pilot", 200)
    assert len(calls) == 2
    normal, high = calls[0][0], calls[1][0]
    assert normal[normal.index("--max-updates") + 1] == "200"
    assert high[high.index("--max-updates") + 1] == "20"
    assert high[high.index("--config") + 1] == "/high.json"
    assert high[high.index("--output") + 1] == str(c.root / "pilot-weight10")
    assert "--resume" not in normal and "--resume" not in high
    assert "DBMOL_LAUNCH_READY" not in calls[1][1]["env"]
    assert "DBMOL_LAUNCH_NONCE" not in calls[1][1]["env"]
    for _, kwargs in calls:
        assert kwargs["env"]["LATENTMOL_SOURCE_SHA256"] == "c" * 64
        assert kwargs["env"]["LATENTMOL_CAMPAIGN_SHA256"] == c.sha


def test_main_resumes_only_after_success_and_stops_at50k(campaign):
    _, c = campaign
    c.validated_result = lambda stage: stage_result(c, stage, start=30000, completed=50000)
    c.queue_main = lambda *a, **kwargs: pytest.fail("No phase beyond50k")
    result = c.advance("main-0002")
    assert result["status"] == "completed_50000"


def test_failed_native_allocation_cannot_adopt_saved_result(campaign, monkeypatch):
    module, c = campaign
    module.save(c.receipts / "pilot.json", {"job_id": "222"})
    monkeypatch.setattr(module, "require_done", lambda job: (_ for _ in ()).throw(ValueError("EXIT")))
    c.scratch = lambda *a: pytest.fail("Do not read/promote after EXIT")
    with pytest.raises(ValueError, match="EXIT"):
        c.validated_result("pilot")


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_actual_wrapper_python_prelude_authenticates_and_applies_route(tmp_path, wrong_hash):
    # Execute the actual wrapper's stdlib prelude; intercept only POSIX exec
    # so this metadata test is also safe under the Windows Python launcher.
    repo = Path(__file__).resolve().parents[3]
    script = (repo / "scripts/hpc/protenix_latent_run.sh").read_text()
    prelude = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    helper = tmp_path / "routing.py"
    helper.write_text("def resolve_visibility(environment):\n    return {'selected_cuda_visible_devices': environment['CUDA_VISIBLE_DEVICES0']}\n")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(cuda_ownership_helper=str(helper),
        cuda_ownership_sha256="0" * 64 if wrong_hash else hashlib.sha256(helper.read_bytes()).hexdigest())))
    worker = tmp_path / "child.py"
    worker.write_text("import os,json\nprint(json.dumps({'visible':os.environ['CUDA_VISIBLE_DEVICES']}))\n")
    env = dict(os.environ, LSB_JOBID="123", CUDA_VISIBLE_DEVICES0="2")
    harness = "import os,json\nos.execvpe=lambda file,args,env: print(json.dumps({'visible':env['CUDA_VISIBLE_DEVICES']}))\n" + prelude
    result = subprocess.run([sys.executable, "-c", harness, sys.executable, str(worker), "--plan", str(plan),
                             "--plan-sha256", hashlib.sha256(plan.read_bytes()).hexdigest()],
                            env=env, capture_output=True, text=True, timeout=10)
    if wrong_hash:
        assert result.returncode != 0 and "helper changed" in result.stderr
        assert not result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"visible": "2"}
        assert "cuda_visibility_resolved" in result.stderr
