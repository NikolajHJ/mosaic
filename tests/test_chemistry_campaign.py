"""Metadata-only fail-closed chemistry controller fixtures, allocated HPC only.

No live scheduler, SSH, GPU, checkpoint loading or model arithmetic is invoked.
Every subprocess response is synthetic and every evidence file is temporary.
These fixtures certify controller behavior, never an actual scientific result.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/hpc"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("_chemistry_campaign_fixture", SCRIPTS / "chemistry_campaign.py")
campaign_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign_module)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _campaign(tmp_path):
    """Bypass only external release discovery; exercise real controller methods."""
    campaign = campaign_module.Campaign.__new__(campaign_module.Campaign)
    campaign.work = tmp_path / "work"
    campaign.release = campaign.work / "releases" / "source104"
    campaign.root = campaign.work / "runs" / "chemistry"
    campaign.receipts = campaign.root / "receipts"
    campaign.path = campaign.work / "submissions" / "plan.json"
    campaign.sha = "a" * 64
    campaign.release.mkdir(parents=True)
    campaign.root.mkdir(parents=True)
    (campaign.work / "logs").mkdir()
    campaign.plan = {"archive_sha256": "b" * 64, "manifest_sha256": "c" * 64,
                     "parent_checkpoint_sha256": "d" * 64, "parent_epoch": 31,
                     "parent_updates": 93, "training_molecules": 513,
                     "dataset_sha256": "e" * 64, "split_sha256": "f" * 64,
                     "protocol": {"maximum_segments": 20, "total_epochs": 100,
                                  "prior_ratio": 3, "real_batch_size": 256,
                                  "segment_wall_seconds": 22 * 3600, "benchmark_wall_seconds": 6300},
                     "work_root": str(campaign.work), "release": str(campaign.release),
                     "campaign": str(campaign.root)}
    campaign.identity = {"plan_sha256": campaign.sha, "archive_sha256": campaign.plan["archive_sha256"],
                         "manifest_sha256": campaign.plan["manifest_sha256"],
                         "parent_sha256": campaign.plan["parent_checkpoint_sha256"]}
    return campaign


def _cpu_gate(campaign, **changes):
    value = {"status": "passed", "returncode": 0, "job_id": "11", "identity": deepcopy(campaign.identity)}
    value.update(changes)
    _json(campaign.root / "cpu_gate.json", value)
    return value


def _optimized(campaign, monkeypatch=None):
    campaign.plan["execution"] = deepcopy(campaign_module.OPTIMIZED_EXECUTION)
    if monkeypatch is not None:
        monkeypatch.setenv("LSB_JOBID", "11")
        monkeypatch.setenv("LSB_DJOB_NUMPROC", "16")
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.setenv("MKL_NUM_THREADS", "4")
    return campaign


def test_optimized_resources_reserve_workers_without_changing_main_runtime(tmp_path):
    campaign = _optimized(_campaign(tmp_path))
    for stage, gpu in (("cpu", False), ("benchmark", True), ("segment-001", True)):
        command = campaign.arguments(stage, gpu=gpu)
        assert command[command.index("-n") + 1] == "16"
        assert "rusage[mem=4GB]" in command
        assert "rusage[mem=16GB]" not in command
    callback = campaign.arguments("after-benchmark")
    assert callback[callback.index("-n") + 1] == "2"
    worker = campaign.worker(["python", "-m", "fixture"], "fixture")
    wrapper = worker.index("scripts/hpc/boltz_run.sh")
    assert worker[wrapper + 1:wrapper + 5] == ["env", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4"]
    assert worker.index("scripts/verify_stability_release.py") > wrapper + 4


def test_optimized_cpu_gate_requires_same_allocation_profile_and_preserves_source_binding(tmp_path, monkeypatch):
    campaign = _optimized(_campaign(tmp_path), monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "scripts/benchmark_atom_preserving_search.py" in command:
            _json(campaign.root / "repair_profile.json", {"status": "passed", "job_id": "11", "identity": campaign.identity})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign_module.subprocess, "run", run)
    campaign.cpu()
    assert len(calls) == 2
    assert all(name in calls[0] for name in campaign_module.OPTIMIZED_FIXTURES)
    assert calls[1][-2:] == ["--output", str(campaign.root / "repair_profile.json")]
    monkeypatch.setattr(campaign, "complete", lambda stage: "11")
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "11")
    campaign.require_cpu()
    profile = campaign_module.read(campaign.root / "repair_profile.json")
    profile["identity"]["parent_sha256"] = "0" * 64
    _json(campaign.root / "repair_profile.json", profile)
    with pytest.raises(ValueError, match="Identity mismatch"):
        campaign.require_cpu()


@pytest.mark.parametrize("profile_status,profile_job,profile_code", [("failed", "11", 0), ("passed", "12", 0), ("passed", "11", 1)])
def test_optimized_cpu_profile_failure_cannot_approve_fixtures(tmp_path, monkeypatch, profile_status, profile_job, profile_code):
    campaign = _optimized(_campaign(tmp_path), monkeypatch)
    def run(command, **kwargs):
        if "scripts/benchmark_atom_preserving_search.py" in command:
            _json(campaign.root / "repair_profile.json", {"status": profile_status, "job_id": profile_job, "identity": campaign.identity})
            return SimpleNamespace(returncode=profile_code)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign_module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="fixtures failed"):
        campaign.cpu()
    assert campaign_module.read(campaign.root / "cpu_gate.json")["status"] == "failed"


def test_optimized_completion_allows_more_run_lag_but_requires_actual_done(tmp_path, monkeypatch):
    campaign = _optimized(_campaign(tmp_path))
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    statuses, calls, sleeps = iter(["RUN"] * 8 + ["DONE"]), [], []
    monkeypatch.setattr(campaign_module.time, "sleep", sleeps.append)
    def status(arguments, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout=next(statuses), stderr="")
    monkeypatch.setattr(campaign_module.subprocess, "run", status)
    assert campaign.complete("benchmark") == "123"
    assert len(calls) == 9 and len(sleeps) == 8
    assert (campaign_module.OPTIMIZED_COMPLETION_QUERIES * campaign_module.COMPLETION_QUERY_TIMEOUT
            + (campaign_module.OPTIMIZED_COMPLETION_QUERIES - 1) * campaign_module.COMPLETION_SETTLE_SECONDS) <= 120


def test_optimized_failed_persisted_benchmark_never_waits_for_run_lag(tmp_path, monkeypatch):
    campaign = _optimized(_campaign(tmp_path))
    _cpu_gate(campaign)
    _benchmark_gate(campaign, status="failed", gates={"complete_epoch_fits": False})
    calls = []
    monkeypatch.setattr(campaign, "require_cpu", lambda: None)
    monkeypatch.setattr(campaign, "complete", lambda stage: calls.append(stage))
    with pytest.raises(ValueError, match="scientific promotion gates"):
        campaign.advance("benchmark")
    assert calls == []


@pytest.mark.parametrize("change", ["missing_gate", "missing_proof", "not_passed", "wrong_status",
    "not_restored", "empty_checks", "false_check", "untyped_check", "wrong_execution"])
def test_optimized_promotion_requires_actual_reference_parity_proof_before_scheduler_wait(tmp_path, monkeypatch, change):
    campaign = _optimized(_campaign(tmp_path))
    proof = {"passed": True, "status": "passed", "restoration": {"passed": True},
             "checks": {"exact_update": True, "exact_rng": True}}
    gates = {"finite_gradients": True, "exact_reference_execution": True}
    execution = deepcopy(campaign.plan["execution"])
    if change == "missing_gate":
        gates.pop("exact_reference_execution")
    elif change == "missing_proof":
        proof = {}
    elif change == "not_passed":
        proof["passed"] = False
    elif change == "wrong_status":
        proof["status"] = "running"
    elif change == "not_restored":
        proof["restoration"]["passed"] = False
    elif change == "empty_checks":
        proof["checks"] = {}
    elif change == "false_check":
        proof["checks"]["exact_rng"] = False
    elif change == "untyped_check":
        proof["checks"]["exact_update"] = 1
    elif change == "wrong_execution":
        execution["workers"] = 4
    _benchmark_gate(campaign, gates=gates, native_execution_parity=proof, execution=execution)
    complete, queued = [], []
    monkeypatch.setattr(campaign, "require_cpu", lambda: None)
    monkeypatch.setattr(campaign, "complete", lambda stage: complete.append(stage))
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queued.append((args, kwargs)))
    with pytest.raises(ValueError, match="full-update reference parity"):
        campaign.advance("benchmark")
    assert complete == queued == []


def test_optimized_passing_reference_proof_still_requires_done_and_queues_only_one_segment(tmp_path, monkeypatch):
    campaign = _optimized(_campaign(tmp_path))
    proof = {"passed": True, "status": "passed", "restoration": {"passed": True},
             "checks": {"exact_update": True, "exact_rng": True}}
    _benchmark_gate(campaign, gates={"finite_gradients": True, "exact_reference_execution": True},
                    native_execution_parity=proof, execution=deepcopy(campaign.plan["execution"]))
    _predecessors(monkeypatch, campaign)
    monkeypatch.setattr(campaign, "require_cpu", lambda: None)
    monkeypatch.setattr(campaign_module, "quota_receipt", lambda minimum: {"minimum_gib": minimum})
    queued = []
    monkeypatch.setattr(campaign, "queue", lambda stage, command, **kwargs:
                        queued.append((stage, command, kwargs)) or str(100 + len(queued)))
    campaign.advance("benchmark")
    # Fresh submissions communicate through their durable decision artifact;
    # only reentry returns an existing decision. Keep that public API unchanged.
    decision = campaign_module.read(campaign.root / "advance-after-benchmark.json")
    assert decision["status"] == "training_submitted_not_passed"
    assert [item[0] for item in queued] == ["segment-001", "after-segment-001"]


def _benchmark_gate(campaign, **changes):
    composition = campaign.root / "training_composition.json"
    _json(composition, {"split": "train", "element_counts": [10, 2], "charge_counts": [0, 12, 0]})
    value = {"status": "passed", "plan_sha256": campaign.sha,
             "parent_sha256": campaign.plan["parent_checkpoint_sha256"],
             "composition_sha256": campaign_module.file_sha256(composition),
             "run_dir": str(campaign.root / "benchmark" / "native"),
             "gates": {"finite_gradients": True, "coverage": True}}
    value.update(changes)
    _json(campaign.root / "benchmark_gate.json", value)
    _json(Path(value["run_dir"]) / "gate.json", value)
    _json(Path(value["run_dir"]) / "run.json",
          {"status": "completed", "environment": {"job_id": "22"},
           "source_archive_sha256": campaign.plan["archive_sha256"],
           "source": {"sha256": campaign.plan["manifest_sha256"]}})
    return value


def _predecessors(monkeypatch, campaign):
    monkeypatch.setattr(campaign, "complete", lambda stage: "11" if stage == "cpu" else "22")
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "11" if stage == "cpu" else "22")


@pytest.mark.parametrize("seconds,expected_limit", [(22 * 3600, "23:00"), (71 * 3600, "72:00")])
@pytest.mark.parametrize("after_stage", ["benchmark", "segment-001"])
def test_serial_segment_allocation_uses_immutable_internal_duration(tmp_path, monkeypatch, seconds, expected_limit, after_stage):
    """Exercise actual promotion wiring, including the resumed serial branch."""
    campaign = _campaign(tmp_path)
    campaign.plan["protocol"]["segment_wall_seconds"] = seconds
    _cpu_gate(campaign)
    _benchmark_gate(campaign)
    _predecessors(monkeypatch, campaign)
    if after_stage != "benchmark":
        _segment(campaign)
    original = deepcopy(campaign.plan)
    monkeypatch.setattr(campaign_module, "quota_receipt", lambda minimum: {"minimum_gib": minimum})
    queued = []
    monkeypatch.setattr(campaign, "queue", lambda stage, command, **kwargs:
                        queued.append((stage, command, kwargs)) or str(100 + len(queued)))
    campaign.advance(after_stage)
    number = "001" if after_stage == "benchmark" else "002"
    assert [item[0] for item in queued] == ["segment-" + number, "after-segment-" + number]
    assert queued[0][2]["gpu"] is True
    assert queued[0][2]["limit"] == expected_limit
    assert queued[1][2]["dependency"] == "ended(101)"
    assert campaign.plan == original


@pytest.mark.parametrize("invalid", [True, 0, -1, 255600.0, 255601, float("inf"), float("nan"), "255600"])
def test_invalid_segment_duration_never_queues_next_training_allocation(tmp_path, monkeypatch, invalid):
    campaign = _campaign(tmp_path)
    campaign.plan["protocol"]["segment_wall_seconds"] = invalid
    _cpu_gate(campaign)
    _benchmark_gate(campaign)
    _predecessors(monkeypatch, campaign)
    monkeypatch.setattr(campaign_module, "quota_receipt", lambda minimum: {"minimum_gib": minimum})
    queued = []
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queued.append((args, kwargs)))
    with pytest.raises(ValueError, match="segment_wall_seconds"):
        campaign.advance("benchmark")
    assert queued == []


@pytest.mark.parametrize("json_string_path", [False, True])
def test_actual_authentication_accepts_path_and_json_string_without_weakening_hash(tmp_path, json_string_path):
    """Exercise the reused real byte verifier, not a mocked authentication stub.

    Scientific plans serialize paths as JSON strings. The historical verifier
    expects Path objects, so the controller must normalize at its boundary while
    retaining exact SHA validation. No scheduler or scientific module is called.
    """
    artifact = tmp_path / "immutable-source.tar.gz"
    artifact.write_bytes(b"sealed fixture source bytes")
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    argument = str(artifact) if json_string_path else artifact
    assert campaign_module.authenticated(argument, expected) is None
    with pytest.raises(ValueError, match="Identity mismatch"):
        campaign_module.authenticated(argument, "0" * 64)
    with pytest.raises(ValueError, match="Identity mismatch"):
        campaign_module.authenticated(argument, "malformed-digest")
    artifact.write_bytes(b"changed fixture source bytes")
    with pytest.raises(ValueError, match="Identity mismatch"):
        campaign_module.authenticated(argument, expected)


def _segment(campaign, *, through=33):
    """Complete contiguous synthetic epoch metadata; checkpoint bytes are opaque."""
    run = campaign.root / "segments" / "001" / "native"
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint.pt"
    checkpoint.write_bytes(b"opaque fixture checkpoint bytes; never torch-loaded")
    checkpoint_sha = campaign_module.file_sha256(checkpoint)
    previous = campaign.plan["parent_epoch"]
    count = campaign.plan["training_molecules"]
    updates_per_epoch = (count + 255) // 256
    rows = [{"epoch": epoch, "updates": campaign.plan["parent_updates"] + (epoch - previous) * updates_per_epoch,
             "real_points": count, "fresh_prior_points": campaign.plan["protocol"]["prior_ratio"] * count,
             "health": {"finite": True, "composition": True}, "epoch_seconds": 1.,
             "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha}
            for epoch in range(previous + 1, through + 1)]
    record = {"status": "completed", "returncode": 0, "environment": {"job_id": "22"},
              "source_archive_sha256": campaign.plan["archive_sha256"],
              "source": {"sha256": campaign.plan["manifest_sha256"]},
              "config": {"_plan_sha256": campaign.sha}, "starting_epoch": previous,
              "completed_epoch": through, "updates": rows[-1]["updates"],
              "resume_checkpoint_sha256": campaign.plan["parent_checkpoint_sha256"],
              "benchmark_gate_sha256": campaign_module.file_sha256(campaign.root / "benchmark_gate.json")}
    completion = {"status": "passed", "plan_sha256": campaign.sha, "segment": 1,
                  "epoch_complete": True, "completed_epoch": through, "updates": rows[-1]["updates"],
                  "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                  "stop_reason": "planned_horizon" if through == 100 else "wall_budget_epoch_boundary",
                  "run_dir": str(run)}
    _json(run / "run.json", record)
    _json(run / "latest_epoch.json", rows[-1])
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    _json(campaign.root / "segment-001-completion.json", completion)
    return run, record, rows, completion


def test_gpu_and_cpu_resource_arguments_use_proven_hardware_and_bounded_dependencies(tmp_path):
    campaign = _campaign(tmp_path)
    gpu = campaign.arguments("benchmark", gpu=True, limit="2:00", dependency="done(11)")
    cpu = campaign.arguments("cpu", gpu=False, limit="0:30")
    assert gpu[gpu.index("-q") + 1] == "gpua100"
    assert gpu[gpu.index("-n") + 1] == "4"
    assert "rusage[mem=16GB]" in gpu and "span[hosts=1]" in gpu
    assert "select[gpu80gb]" in gpu
    assert "exclusive_process" in gpu[gpu.index("-gpu") + 1]
    assert "gmem=80GB" not in gpu[gpu.index("-gpu") + 1]
    assert gpu[gpu.index("-w") + 1] == "done(11)"
    assert gpu[gpu.index("-W") + 1] == "2:00"
    assert cpu[cpu.index("-q") + 1] == "hpc" and "-gpu" not in cpu
    command = campaign.worker(["python", "-m", "fixture"], "fixture")
    assert "--archive-sha256" in command and "--manifest-sha256" in command
    assert campaign.plan["archive_sha256"] in command
    assert campaign.plan["manifest_sha256"] in command


@pytest.mark.parametrize("workers", [12, 20, 28])
def test_new_benchmark_and_training_resource_receipts_keep_declared_32_slots(tmp_path, monkeypatch, workers):
    campaign = _campaign(tmp_path)
    campaign.plan["execution"] = {"backend": "optimized", "workers": workers, "torch_threads": 4,
                                   "gpu_cpus": 32, "overlap": True}
    for stage in ("benchmark", "segment-001"):
        args = campaign.arguments(stage, gpu=True)
        assert args[args.index("-n") + 1] == "32"
        assert "rusage[mem=4GB]" in args
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "32")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    assert campaign_module.execution_settings(campaign.plan, allocated=True)["workers"] == workers
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "16")
    with pytest.raises(ValueError, match="allocated CPU slots"):
        campaign_module.execution_settings(campaign.plan, allocated=True)


def test_ratio2_serial_epoch_bookkeeping_requires_exact_two_prior_points_per_real(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    campaign.plan["protocol"].update(prior_ratio=2, composition_prior_weight=1.5)
    _cpu_gate(campaign)
    _benchmark_gate(campaign)
    _predecessors(monkeypatch, campaign)
    _segment(campaign)
    monkeypatch.setattr(campaign_module, "quota_receipt", lambda minimum: {"minimum_gib": minimum})
    queued = []
    monkeypatch.setattr(campaign, "queue", lambda stage, command, **kwargs:
                        queued.append((stage, command, kwargs)) or str(100 + len(queued)))
    campaign.advance("segment-001")
    assert [item[0] for item in queued] == ["segment-002", "after-segment-002"]


def test_queue_receipt_is_idempotent_and_retains_exact_identity(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = []
    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        assert arguments[0] == "bsub"
        return SimpleNamespace(returncode=0, stdout="Job <123> is submitted to queue <hpc>.\n", stderr="")
    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)
    assert campaign.queue("cpu", ["python", "-m", "fixture"]) == "123"
    assert campaign.queue("cpu", ["python", "-m", "fixture"]) == "123"
    assert len(calls) == 1
    assert campaign.receipt_job("cpu") == "123"
    receipt = campaign_module.read(campaign.receipts / "cpu-receipt.json")
    assert receipt["status"] == "submitted_not_passed"
    assert receipt["intent"]["identity"] == campaign.identity


def test_unknown_submission_outcome_cannot_be_retried(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = []
    def timeout(arguments, **kwargs):
        calls.append(arguments)
        raise subprocess.TimeoutExpired(arguments, 60)
    monkeypatch.setattr(campaign_module.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="unknown"):
        campaign.queue("cpu", ["python", "-m", "fixture"])
    with pytest.raises(FileExistsError):
        campaign.queue("cpu", ["python", "-m", "fixture"])
    assert len(calls) == 1
    assert campaign_module.read(campaign.receipts / "cpu-unknown.json")["status"] == "unknown_not_retryable"
    assert not (campaign.receipts / "cpu-receipt.json").exists()


@pytest.mark.parametrize("corruption", ["identity", "job_id", "intent"])
def test_submission_receipt_binding_mismatch_rejected(tmp_path, corruption):
    campaign = _campaign(tmp_path)
    intent = {"identity": deepcopy(campaign.identity), "command": ["bsub", "fixture"]}
    receipt = {"intent": deepcopy(intent), "job_id": "123", "status": "submitted_not_passed"}
    if corruption == "identity":
        intent["identity"]["parent_sha256"] = "0" * 64
        receipt["intent"] = deepcopy(intent)
    elif corruption == "job_id":
        receipt["job_id"] = "123; injected"
    else:
        receipt["intent"]["command"] = ["bsub", "changed"]
    _json(campaign.receipts / "cpu-intent.json", intent)
    _json(campaign.receipts / "cpu-receipt.json", receipt)
    with pytest.raises(ValueError, match="identity differs"):
        campaign.receipt_job("cpu")


@pytest.mark.parametrize("status", ["RUN", "PEND", "EXIT", "UNKWN"])
def test_pending_or_failed_predecessor_never_counts_as_complete(tmp_path, monkeypatch, status):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    sleeps = []
    monkeypatch.setattr(campaign_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=status, stderr=""))
    with pytest.raises(ValueError, match="did not finish successfully"):
        campaign.complete("benchmark")
    assert sleeps == ([campaign_module.COMPLETION_SETTLE_SECONDS] * (campaign_module.COMPLETION_QUERIES - 1)
                      if status == "RUN" else [])


@pytest.mark.parametrize("running_queries", [0, 1, 3])
def test_scheduler_run_settles_to_explicit_done_within_bounded_queries(tmp_path, monkeypatch, running_queries):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    statuses, calls, sleeps = iter(["RUN"] * running_queries + ["DONE"]), [], []
    def status(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout=next(statuses), stderr="")
    monkeypatch.setattr(campaign_module.subprocess, "run", status)
    monkeypatch.setattr(campaign_module.time, "sleep", sleeps.append)
    assert campaign.complete("benchmark") == "123"
    assert len(calls) == running_queries + 1
    assert all(arguments == ["bjobs", "-a", "-noheader", "-o", "stat", "123"] for arguments, _ in calls)
    assert all(kwargs["timeout"] == campaign_module.COMPLETION_QUERY_TIMEOUT for _, kwargs in calls)
    assert sleeps == [campaign_module.COMPLETION_SETTLE_SECONDS] * running_queries
    assert (campaign_module.COMPLETION_QUERIES * campaign_module.COMPLETION_QUERY_TIMEOUT
            + (campaign_module.COMPLETION_QUERIES - 1) * campaign_module.COMPLETION_SETTLE_SECONDS) < 30


def test_exhausted_running_predecessor_never_uses_completed_artifact_fallback(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    calls, sleeps = [], []
    monkeypatch.setattr(campaign_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or SimpleNamespace(returncode=0, stdout="RUN", stderr=""))
    _json(campaign.root / "allocations" / "one" / "run.json",
          {"status": "completed", "returncode": 0, "environment": {"job_id": "123"}})
    with pytest.raises(ValueError, match="did not finish successfully: RUN"):
        campaign.complete("benchmark")
    assert len(calls) == campaign_module.COMPLETION_QUERIES
    assert sleeps == [campaign_module.COMPLETION_SETTLE_SECONDS] * (campaign_module.COMPLETION_QUERIES - 1)


@pytest.mark.parametrize("status,returncode", [("EXIT", 0), ("PEND", 0), ("UNKWN", 0),
                                              ("DONE", 255), ("RUN", 255), ("", 255)])
def test_scheduler_retry_is_only_for_successfully_observed_run(tmp_path, monkeypatch, status, returncode):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    calls, sleeps = [], []
    monkeypatch.setattr(campaign_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or SimpleNamespace(returncode=returncode, stdout=status,
                                                                      stderr="scheduler query failed"))
    _json(campaign.root / "allocations" / "one" / "run.json",
          {"status": "completed", "returncode": 0, "environment": {"job_id": "123"}})
    with pytest.raises(ValueError, match="did not finish successfully"):
        campaign.complete("benchmark")
    assert len(calls) == 1 and sleeps == []


def test_scheduler_query_timeout_stops_without_retry_or_artifact_approval(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    sleeps, calls = [], []
    monkeypatch.setattr(campaign_module.time, "sleep", sleeps.append)
    def timeout(arguments, **kwargs):
        calls.append(arguments)
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])
    monkeypatch.setattr(campaign_module.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.complete("benchmark")
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("records", [[], [{"status": "failed", "returncode": 1}],
                                    [{"status": "completed", "returncode": 0},
                                     {"status": "completed", "returncode": 0}]])
def test_aged_out_predecessor_requires_unique_successful_persisted_allocation(tmp_path, monkeypatch, records):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=255, stdout="", stderr="Job <123> is not found"))
    for index, record in enumerate(records):
        _json(campaign.root / "allocations" / str(index) / "run.json",
              {**record, "environment": {"job_id": "123"}})
    with pytest.raises(ValueError, match="unique completed allocation"):
        campaign.complete("benchmark")


def test_aged_out_predecessor_accepts_one_successful_allocation_only(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=255, stdout="", stderr="Job <123> is not found"))
    _json(campaign.root / "allocations" / "one" / "run.json",
          {"status": "completed", "returncode": 0, "environment": {"job_id": "123"}})
    assert campaign.complete("benchmark") == "123"


@pytest.mark.parametrize("changes", [{"status": "failed"}, {"returncode": 1}, {"job_id": "other"},
                                     {"identity": {"plan_sha256": "wrong"}}])
def test_failed_or_wrong_cpu_gate_stops_promotion_before_any_queue(tmp_path, monkeypatch, changes):
    campaign = _campaign(tmp_path)
    _predecessors(monkeypatch, campaign)
    _cpu_gate(campaign, **changes)
    _benchmark_gate(campaign)
    queue = []
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queue.append((args, kwargs)))
    with pytest.raises(ValueError, match="CPU gate"):
        campaign.advance("benchmark")
    assert not queue


@pytest.mark.parametrize("changes", [{"status": "failed"}, {"plan_sha256": "wrong"},
                                     {"parent_sha256": "wrong"}, {"gates": {}},
                                     {"gates": {"finite_gradients": False}}, {"gates": {"finite_gradients": 1}}])
def test_failed_or_wrong_benchmark_gate_never_queues_training(tmp_path, monkeypatch, changes):
    campaign = _campaign(tmp_path)
    _predecessors(monkeypatch, campaign)
    _cpu_gate(campaign)
    _benchmark_gate(campaign, **changes)
    queue = []
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queue.append((args, kwargs)))
    with pytest.raises(ValueError, match="Benchmark"):
        campaign.advance("benchmark")
    assert not queue


def test_changed_training_composition_blocks_promotion(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _predecessors(monkeypatch, campaign)
    _cpu_gate(campaign)
    _benchmark_gate(campaign)
    _json(campaign.root / "training_composition.json", {"split": "val", "element_counts": [1, 1]})
    queue = []
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queue.append((args, kwargs)))
    with pytest.raises(ValueError, match="Identity mismatch"):
        campaign.advance("benchmark")
    assert not queue


def test_quota_uses_official_finite_user_row_and_preserves_raw_evidence(monkeypatch):
    stdout = "s234818|306121||0 Byte|unlimited||0|unlimited\ns234818|306121||87.98 GiB|100.00 GiB||234849|20000000\n"
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=stdout, stderr="fixture"))
    result = campaign_module.quota_receipt(10)
    assert result["remaining_gib"] == ["12.02"]
    assert result["stdout"] == stdout and result["stderr"] == "fixture"
    assert result["minimum_gib"] == 10


@pytest.mark.parametrize("stdout", ["", "s234818|306121||0 Byte|unlimited||0|unlimited\n",
                                    "other|306121||0 GiB|100 GiB||0|100\n",
                                    "s234818|306121||95 GiB|100 GiB||0|100\n",
                                    "s234818|306121||NaN GiB|100 GiB||0|100\n",
                                    "s234818|306121||-1 GiB|100 GiB||0|100\n"])
def test_missing_unknown_or_insufficient_quota_fails_closed(monkeypatch, stdout):
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=stdout, stderr=""))
    with pytest.raises(ValueError):
        campaign_module.quota_receipt(10)


def test_initial_suite_only_queues_fixture_benchmark_and_promotion_callback(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    queued = []
    monkeypatch.setattr(campaign_module, "quota_receipt", lambda minimum: {"minimum_gib": minimum})
    def fake_queue(stage, command, **kwargs):
        queued.append((stage, command, kwargs))
        return str(11 * len(queued))
    monkeypatch.setattr(campaign, "queue", fake_queue)
    result = campaign.queue_initial()
    assert [row[0] for row in queued] == ["cpu", "benchmark", "after-benchmark"]
    assert queued[1][2]["dependency"] == "done(11)"
    assert queued[2][2]["dependency"] == "ended(22)"
    assert result["status"] == "submitted_not_passed_no_training_yet"
    assert not any(stage.startswith("segment") for stage, _, _ in queued)
    command = queued[1][1]
    assert command[-3:-1] == ["bash", "-c"]
    precheck, benchmark = command[-1].split(" && ")
    assert shlex.split(precheck) == ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1", "python", "-m",
                                     "pytest", "-q", "tests/test_chemistry_resume_cuda.py"]
    assert shlex.split(benchmark) == ["python", "-m", "latentmol.chemistry_continuation", "benchmark",
                                      "--plan", str(campaign.path), "--plan-sha256", campaign.sha]
    assert "tests/test_chemistry_resume_cuda.py" in campaign_module.FIXTURES


def test_run_to_done_settling_does_not_waive_failed_scientific_benchmark(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "11" if stage == "cpu" else "22")
    _cpu_gate(campaign)
    _benchmark_gate(campaign, status="failed")
    statuses = iter(["DONE", "RUN", "DONE"])
    sleeps, queued = [], []
    monkeypatch.setattr(campaign_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=next(statuses), stderr=""))
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queued.append((args, kwargs)))
    with pytest.raises(ValueError, match="Benchmark did not pass"):
        campaign.advance("benchmark")
    assert sleeps == [campaign_module.COMPLETION_SETTLE_SECONDS]
    assert queued == []
    assert campaign_module.read(campaign.root / "advance-after-benchmark.json")["status"] == "stopped"


@pytest.mark.parametrize("content", ['{"epoch":31,"epoch":32}', '{"metric":NaN}',
                                    '{"metric":Infinity}', '{"metric":1e999}'])
def test_ambiguous_or_nonfinite_controller_metadata_is_rejected(tmp_path, content):
    path = tmp_path / "invalid-metadata.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        campaign_module.read(path)


def test_unknown_scheduler_failure_is_not_treated_as_aged_out_completion(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(campaign, "receipt_job", lambda stage: "123")
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=255, stdout="", stderr="scheduler connection failed"))
    _json(campaign.root / "allocations" / "one" / "run.json",
          {"status": "completed", "returncode": 0, "environment": {"job_id": "123"}})
    with pytest.raises(ValueError, match="did not finish successfully"):
        campaign.complete("benchmark")


def test_cpu_fixture_execution_records_failure_and_requires_allocation(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = []
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.setattr(campaign_module.subprocess, "run", lambda *args, **kwargs:
                        calls.append((args, kwargs)) or SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="allocated LSF"):
        campaign.cpu()
    assert not calls
    monkeypatch.setenv("LSB_JOBID", "77")
    with pytest.raises(RuntimeError, match="fixtures failed"):
        campaign.cpu()
    receipt = campaign_module.read(campaign.root / "cpu_gate.json")
    assert receipt["status"] == "failed" and receipt["returncode"] == 1
    assert receipt["job_id"] == "77" and receipt["identity"] == campaign.identity
    assert set(campaign_module.FIXTURES).issubset(receipt["command"])


def test_passed_benchmark_promotes_one_segment_and_reentered_callback_never_duplicates(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _predecessors(monkeypatch, campaign)
    _cpu_gate(campaign); _benchmark_gate(campaign)
    monkeypatch.setattr(campaign_module, "quota_receipt", lambda minimum: {"minimum_gib": minimum})
    queue = []
    def fake_queue(stage, command, **kwargs):
        queue.append((stage, command, kwargs))
        return str(100 + len(queue))
    monkeypatch.setattr(campaign, "queue", fake_queue)
    campaign.advance("benchmark")
    assert [row[0] for row in queue] == ["segment-001", "after-segment-001"]
    assert queue[1][2]["dependency"] == "ended(101)"
    assert "--benchmark-gate-sha256" in queue[0][1]
    result = campaign.advance("benchmark")
    assert len(queue) == 2
    assert result["status"] == "training_submitted_not_passed"
    assert result["completed_epoch"] == campaign.plan["parent_epoch"]
    assert (campaign.root / "advance-after-benchmark-intent.json").exists()


def test_unknown_callback_intent_blocks_retry_and_preserves_existing_evidence(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    intent = campaign.root / "advance-after-benchmark-intent.json"
    _json(intent, {"identity": campaign.identity, "status": "unknown"})
    before = intent.read_bytes()
    queue = []
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queue.append(args))
    with pytest.raises(FileExistsError):
        campaign.advance("benchmark")
    assert not queue and intent.read_bytes() == before


def test_existing_decision_for_another_identity_is_never_reused(tmp_path):
    campaign = _campaign(tmp_path)
    path = campaign.root / "advance-after-benchmark.json"
    _json(path, {"identity": {"plan_sha256": "other"}, "status": "training_submitted_not_passed"})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="another phase"):
        campaign.advance("benchmark")
    assert path.read_bytes() == before


def test_active_or_failed_phase_lock_blocks_another_allocation(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _predecessors(monkeypatch, campaign)
    _cpu_gate(campaign); _benchmark_gate(campaign)
    _json(campaign.root / "active-phase.json", {"status": "failed_incomplete_epoch"})
    queue = []
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queue.append(args))
    with pytest.raises(ValueError, match="active/failed"):
        campaign.advance("benchmark")
    assert not queue


def test_promotion_quota_failure_stops_before_training_submission(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _predecessors(monkeypatch, campaign)
    _cpu_gate(campaign); _benchmark_gate(campaign)
    def no_quota(minimum):
        raise ValueError("Insufficient work3 quota fixture")
    monkeypatch.setattr(campaign_module, "quota_receipt", no_quota)
    queue = []
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queue.append(args))
    with pytest.raises(ValueError, match="Insufficient"):
        campaign.advance("benchmark")
    assert not queue
    result = campaign_module.read(campaign.root / "advance-after-benchmark.json")
    assert result["status"] == "stopped" and "quota" in result["error"]


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reordered", "sample_count", "ratio",
                                        "updates", "health", "starting_epoch", "resume", "benchmark",
                                        "completion_updates", "source", "job", "checkpoint_bytes",
                                        "latest", "segment_number", "stop_reason"])
def test_incomplete_or_wrong_segment_history_never_promotes(tmp_path, monkeypatch, corruption):
    campaign = _campaign(tmp_path)
    _predecessors(monkeypatch, campaign)
    _cpu_gate(campaign); _benchmark_gate(campaign)
    run, record, rows, completion = _segment(campaign)
    if corruption == "missing":
        rows.pop(0)
    elif corruption == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif corruption == "reordered":
        rows.reverse()
    elif corruption == "sample_count":
        rows[0]["real_points"] += 1
    elif corruption == "ratio":
        rows[0]["fresh_prior_points"] -= 1
    elif corruption == "updates":
        rows[0]["updates"] += 1
    elif corruption == "health":
        rows[0]["health"]["finite"] = 1  # Truthy metadata must not count as passed.
    elif corruption == "starting_epoch":
        record["starting_epoch"] -= 1
    elif corruption == "resume":
        record["resume_checkpoint_sha256"] = "0" * 64
    elif corruption == "benchmark":
        record["benchmark_gate_sha256"] = "0" * 64
    elif corruption == "completion_updates":
        completion["updates"] += 1
    elif corruption == "source":
        record["source"]["sha256"] = "0" * 64
    elif corruption == "job":
        record["environment"]["job_id"] = "23"
    elif corruption == "checkpoint_bytes":
        Path(completion["checkpoint"]).write_bytes(b"changed checkpoint bytes")
    elif corruption == "latest":
        stale = deepcopy(rows[-1]); stale["real_points"] = 1; stale["fresh_prior_points"] = 3
        _json(run / "latest_epoch.json", stale)
    elif corruption == "segment_number":
        completion["segment"] = 2
    else:
        completion["stop_reason"] = "crashed"
    _json(run / "run.json", record)
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    _json(campaign.root / "segment-001-completion.json", completion)
    queue = []
    monkeypatch.setattr(campaign, "queue", lambda *args, **kwargs: queue.append(args))
    with pytest.raises(ValueError):
        campaign.advance("segment-001")
    assert not queue
    assert campaign_module.read(campaign.root / "advance-after-segment-001.json")["status"] == "stopped"


@pytest.mark.parametrize("through,next_stage", [(33, "segment-002"), (100, "final")])
def test_authenticated_contiguous_history_resumes_only_next_serial_segment_or_final(tmp_path, monkeypatch,
                                                                                  through, next_stage):
    campaign = _campaign(tmp_path)
    _predecessors(monkeypatch, campaign)
    _cpu_gate(campaign); _benchmark_gate(campaign)
    run, _, _, completion = _segment(campaign, through=through)
    monkeypatch.setattr(campaign_module, "quota_receipt", lambda minimum: {"minimum_gib": minimum})
    queued = []
    def fake_queue(stage, command, **kwargs):
        queued.append((stage, command, kwargs))
        return str(100 + len(queued))
    monkeypatch.setattr(campaign, "queue", fake_queue)
    campaign.advance("segment-001")
    assert queued[0][0] == next_stage
    command = queued[0][1]
    assert command[command.index("--resume") + 1] == completion["checkpoint"]
    assert command[command.index("--resume-sha256") + 1] == completion["checkpoint_sha256"]
    assert queued[0][2]["gpu"]
    if through < 100:
        assert [entry[0] for entry in queued] == ["segment-002", "after-segment-002"]
        assert queued[1][2]["dependency"] == "ended(101)"
    else:
        assert len(queued) == 1
