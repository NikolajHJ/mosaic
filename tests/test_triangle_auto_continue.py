"""Allocated, metadata-only fixtures for fail-closed triangle continuation.

No checkpoint loading, model arithmetic, SSH or live scheduler calls occur.
Synthetic records exercise bounded decisions; archived running source098 JSON
tests schema compatibility only, never claims an allocation has completed.
Run these checks on DTU HPC, not locally.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts/hpc"
health_spec = importlib.util.spec_from_file_location("triangle_continuation_health", SCRIPTS / "triangle_continuation_health.py")
health = importlib.util.module_from_spec(health_spec)
sys.modules[health_spec.name] = health
health_spec.loader.exec_module(health)
controller_spec = importlib.util.spec_from_file_location("_triangle_auto_continue_fixture", SCRIPTS / "triangle_auto_continue.py")
controller = importlib.util.module_from_spec(controller_spec)
controller_spec.loader.exec_module(controller)


def cohort():
    return {"atom_ce": .1, "bond_ce": .2, "charge_ce": .05, "reconstruction": .35,
            "validity": .8, "exact_reconstruction": .7, "chemistry_samples": 16}


def campaign(previous=0, through=2):
    """Small complete telemetry records with the trainer's actual field names."""
    config = {"batch_size": 256, "mode": "vae", "model": {"refinement_rounds": 4}}
    protocol = {"requested_counts": {"train": 513, "val": 16, "test": 16},
        "expected_parameters": {"encoder": 5, "decoder": 7, "total": 12},
        "monitor_every": 10, "deep_monitor_epochs": [25, 50, 75, 100]}
    identity = {"source_sha256": "a" * 64, "source_archive_sha256": "b" * 64,
                "dataset_sha256": "c" * 64, "split_sha256": "d" * 64}
    runtime = {"torch": "fixture", "device_type": "cuda", "matmul_tf32": False}
    resume = None if previous == 0 else {"checkpoint": "/fixture/previous/last.pt", "sha256": "e" * 64,
                                      "completed_epoch": previous, "parent_run_id": "previous-run"}
    rows = []
    for epoch in range(previous + 1, through + 1):
        row = {"epoch": epoch, "at": "fixture-timestamp", "epoch_updates": 3, "updates": 3 * epoch,
            "seconds": 10., "seconds_including_extended_diagnostics": 12., "training_seconds": 8.,
            "training_molecules_per_second": 64., "training_updates_per_second": .375,
            "learning_rate": .001, "evaluation_seconds": 1., "checkpoint_seconds": 1.,
            "beta": .0001, "gradient_norm_max": 1., "gradient_norm_mean_per_update": .5,
            "cumulative_training_seconds": 8. * epoch, "cumulative_evaluation_seconds": float(epoch),
            "clipped_updates": 0, "gradient_clipping_rate": 0., "training_latent_policy": "posterior_sample",
            "training_kl_used_in_objective": True,
            "train": {**cohort(), "kl": .2, "total": .4}, "val": cohort(),
            "train_probe": cohort(), "val_probe": cohort(),
            "extended_diagnostics": {"pre_clip_gradient_diagnostics_observations": 1,
                "peak_allocated_vram_bytes_training_and_standard_validation": 1024,
                "peak_reserved_vram_bytes_training_and_standard_validation": 2048,
                "peak_host_resident_bytes_process_lifetime": 4096,
                "pre_clip_group_gradient_means": {"encoder": .1, "decoder": .2, "triangle": .05},
                "sparse_activation_absmax": {f"round{i}/{name}": 2. for i in range(4) for name in ("atoms", "pairs")},
                "posterior_mu_absmax": 1., "posterior_logvar_min": -3., "posterior_logvar_max": .5}}
        if epoch % protocol["monitor_every"] == 0 or epoch in protocol["deep_monitor_epochs"]:
            panel = {"split": "val", "reconstruction": {"validity": .8}}
            if epoch in protocol["deep_monitor_epochs"]:
                panel.update(prior={}, sampled_reconstruction={}, neighborhood={}, interpolation={})
            row["extended_diagnostics"].update(panel=panel, panel_seconds=1.)
        rows.append(row)
    run = {"run_id": "fixture-run", "status": "completed", "completed_epoch": through,
        "planned_epochs": 100, "environment": {"job_id": "101"},
        "source": {"sha256": identity["source_sha256"]}, "source_archive_sha256": identity["source_archive_sha256"],
        "config": deepcopy(config), "runtime_identity": deepcopy(runtime),
        "dataset_manifest": {"dataset_sha256": identity["dataset_sha256"], "split_sha256": identity["split_sha256"]},
        "training_molecules": 513, "parameter_gate": deepcopy(protocol["expected_parameters"]),
        "large_corpus_protocol": deepcopy(protocol), "joint_encoder_decoder_training": True,
        "extra_random_prior_training_points": 0, "chemistry_training_weight": 0.,
        "planned_horizon_complete": through == 100,
        "stop_reason": "planned_horizon" if through == 100 else "wall_budget_epoch_boundary",
        "resume": deepcopy(resume), "updates": 3 * through}
    return {"run": run, "rows": rows, "latest": {key: value for key, value in deepcopy(rows[-1]).items() if key != "at"},
            "previous_epoch": previous, "expected_job": "101", "identity": identity,
            "config": config, "protocol": protocol, "runtime": runtime, "expected_resume": resume}


def sync_latest(data):
    data["latest"] = {key: deepcopy(value) for key, value in data["rows"][-1].items() if key != "at"}


@pytest.mark.parametrize("previous,through,action", [(0, 2, "segment"), (11, 12, "segment"), (99, 100, "final")])
def test_complete_epoch_boundary_chooses_continuation_or_final(previous, through, action):
    result = health.validate_health(**campaign(previous, through))
    assert result["status"] == "passed"
    assert result["from_epoch"] == previous and result["through_epoch"] == through
    assert result["next_action"] == action
    assert result["updates"] == 3 * through


def test_zero_gradients_and_declining_quality_are_allowed_without_new_early_stopping():
    data = campaign()
    last = data["rows"][-1]
    last["gradient_norm_max"] = last["gradient_norm_mean_per_update"] = 0.
    last["extended_diagnostics"]["pre_clip_group_gradient_means"] = dict.fromkeys(("encoder", "decoder", "triangle"), 0.)
    for name in ("val", "train_probe", "val_probe"):
        last[name]["validity"] = last[name]["exact_reconstruction"] = 0.
        last[name]["reconstruction"] = 20.
    sync_latest(data)
    result = health.validate_health(**data)
    assert result["status"] == "passed" and result["next_action"] == "segment"
    assert result["last_validation_validity"] == 0.


@pytest.mark.parametrize("field,value", [("status", "running"), ("planned_epochs", 99),
    ("completed_epoch", 101), ("joint_encoder_decoder_training", False),
    ("extra_random_prior_training_points", 1), ("chemistry_training_weight", .1),
    ("training_molecules", 514), ("stop_reason", "oom"), ("planned_horizon_complete", True)])
def test_run_or_training_scope_mismatch_stops_continuation(field, value):
    data = campaign()
    data["run"][field] = value
    with pytest.raises(ValueError):
        health.validate_health(**data)


@pytest.mark.parametrize("field", ["source", "source_archive", "job", "dataset", "split", "parameters", "config", "runtime", "protocol"])
def test_identity_mismatches_stop_continuation(field):
    data = campaign()
    run = data["run"]
    if field == "source":
        run["source"]["sha256"] = "f" * 64
    elif field == "source_archive":
        run["source_archive_sha256"] = "f" * 64
    elif field == "job":
        run["environment"]["job_id"] = "102"
    elif field in ("dataset", "split"):
        run["dataset_manifest"][field + "_sha256"] = "f" * 64
    elif field == "parameters":
        run["parameter_gate"]["decoder"] += 1
    elif field == "config":
        run["config"]["batch_size"] = 512
    elif field == "runtime":
        run["runtime_identity"]["torch"] = "other"
    else:
        run["large_corpus_protocol"]["monitor_every"] = 25
    with pytest.raises(ValueError):
        health.validate_health(**data)


@pytest.mark.parametrize("corruption", ["missing", "reordered", "duplicate", "latest", "epoch_updates", "updates", "run_updates"])
def test_epoch_log_corruption_is_rejected(corruption):
    data = campaign()
    if corruption == "missing":
        data["rows"].pop(0)
    elif corruption == "reordered":
        data["rows"].reverse()
    elif corruption == "duplicate":
        data["rows"][1] = deepcopy(data["rows"][0])
    elif corruption == "latest":
        data["latest"]["train"]["reconstruction"] += .01
    elif corruption == "run_updates":
        data["run"]["updates"] += 1
    else:
        data["rows"][-1][corruption] += 1
        sync_latest(data)
    with pytest.raises(ValueError):
        health.validate_health(**data)


@pytest.mark.parametrize("corruption", ["missing_round", "missing_gradient", "missing_memory", "zero_memory", "no_observations", "logvar_bounds", "clip_rate", "latent_policy", "kl_policy", "missing_panel", "test_panel", "missing_deep"])
def test_missing_or_inconsistent_required_diagnostics_stop_continuation(corruption):
    data = campaign(24, 25) if corruption in ("missing_panel", "test_panel", "missing_deep") else campaign()
    row = data["rows"][-1]
    extended = row["extended_diagnostics"]
    if corruption == "missing_round":
        del extended["sparse_activation_absmax"]["round2/pairs"]
    elif corruption == "missing_gradient":
        del extended["pre_clip_group_gradient_means"]["encoder"]
    elif corruption == "missing_memory":
        del extended["peak_allocated_vram_bytes_training_and_standard_validation"]
    elif corruption == "zero_memory":
        extended["peak_reserved_vram_bytes_training_and_standard_validation"] = 0
    elif corruption == "no_observations":
        extended["pre_clip_gradient_diagnostics_observations"] = 0
    elif corruption == "logvar_bounds":
        extended["posterior_logvar_min"] = 2.
    elif corruption == "clip_rate":
        row["clipped_updates"] = 1
    elif corruption == "latent_policy":
        row["training_latent_policy"] = "posterior_mean"
    elif corruption == "kl_policy":
        row["training_kl_used_in_objective"] = False
    elif corruption == "missing_panel":
        del extended["panel"]
    elif corruption == "test_panel":
        extended["panel"]["split"] = "test"
    else:
        del extended["panel"]["interpolation"]
    sync_latest(data)
    with pytest.raises(ValueError):
        health.validate_health(**data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_nested_optional_or_required_telemetry_is_rejected(value):
    data = campaign()
    data["rows"][0]["extended_diagnostics"]["optional_nested"] = {"values": [None, value]}
    with pytest.raises(ValueError, match="Nonfinite"):
        health.validate_health(**data)


@pytest.mark.parametrize("text", ['{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}', '{"a":1e999}', '{"a":1,"a":2}', '{"a":'])
def test_strict_json_rejects_nonfinite_duplicate_and_malformed_records(text):
    with pytest.raises(ValueError):
        health.strict_json(text)


def test_strict_json_allows_null_conditional_summaries():
    assert health.strict_json('{"summary":null,"values":[0,1.2]}') == {"summary": None, "values": [0, 1.2]}


@pytest.mark.parametrize("corruption", ["checkpoint", "sha256", "parent_run_id", "completed_epoch", "missing"])
def test_continuation_requires_exact_resume_identity(corruption):
    data = campaign(11, 12)
    if corruption == "missing":
        data["run"]["resume"] = None
    elif corruption == "completed_epoch":
        data["run"]["resume"][corruption] = 10
    else:
        data["run"]["resume"][corruption] = "changed"
    with pytest.raises(ValueError, match="checkpoint/epoch"):
        health.validate_health(**data)


def test_first_allocation_cannot_resume_a_checkpoint():
    data = campaign()
    data["run"]["resume"] = {"checkpoint": "/other/last.pt"}
    with pytest.raises(ValueError, match="unexpectedly resumed"):
        health.validate_health(**data)


def test_archived_source098_running_metadata_schema_at_synthetic_completed_boundary():
    """Only status/stop fields are synthesized; original numerical JSON is kept.

    This is not evidence that the archived running allocation completed, and
    does not certify checkpoint/Adam/RNG compatibility. Those gates remain
    the responsibility of allocated production continuation and the controller.
    """
    fixture = Path(__file__).with_name("fixtures") / "triangle_continuation_source098"
    provenance = health.strict_json((fixture / "provenance.json").read_text())
    for name, expected in provenance["files_sha256"].items():
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == expected
    run = health.strict_json((fixture / "run.json").read_text())
    rows = [health.strict_json(line) for line in (fixture / "metrics.jsonl").read_text().splitlines()]
    latest = health.strict_json((fixture / "latest_epoch.json").read_text())
    assert run["status"] == "running" and run["completed_epoch"] == 11
    run.update(status="completed", stop_reason="wall_budget_epoch_boundary", planned_horizon_complete=False)
    identity = {"source_sha256": run["source"]["sha256"], "source_archive_sha256": run["source_archive_sha256"],
        "dataset_sha256": run["dataset_manifest"]["dataset_sha256"], "split_sha256": run["dataset_manifest"]["split_sha256"]}
    result = health.validate_health(run, rows, latest, previous_epoch=0, expected_job=run["environment"]["job_id"],
        identity=identity, config=run["config"], protocol=run["large_corpus_protocol"], runtime=run["runtime_identity"])
    assert result["through_epoch"] == 11 and result["next_action"] == "segment"


def bare_controller(tmp_path, through=2):
    """Bypass I/O authentication only for isolated decision/command fixtures."""
    instance = controller.Controller.__new__(controller.Controller)
    instance.root, instance.bundle = tmp_path / "control", tmp_path / "bundle"
    instance.release, instance.campaign = tmp_path / "release", tmp_path / "campaign"
    instance.release.mkdir()
    instance.policy = {"max_segments": 20}
    instance.policy_path, instance.policy_sha = tmp_path / "policy.json", "p" * 64
    instance.identity = {"source": "fixture-scientific-identity"}
    instance.operator_identity = {"policy_sha256": instance.policy_sha, "scientific_identity": instance.identity}
    instance.original = SimpleNamespace(PYTHON=Path("/fixture/python"), LAUNCHER=Path("/fixture/launcher.py"),
        ARCHIVE_SHA="a" * 64, MANIFEST_SHA="b" * 64, BINDINGS_SHA="c" * 64, WORK=tmp_path / "work")
    instance.cpu_gate = lambda: None
    instance.completed_health = lambda _stage: {"health": {"through_epoch": through, "next_action": "final" if through == 100 else "segment"},
        "quota": {"status": "passed"}, "checkpoint": "/fixture/run/last.pt", "checkpoint_sha256": "d" * 64,
        "run_id": "run-before", "evidence_sha256": {"fixture": "e" * 64}}
    return instance


def forbid_subprocess(*_args, **_kwargs):
    raise AssertionError("Failed/duplicate health decision must not submit or run a subprocess")


@pytest.mark.parametrize("through,kind,next_id", [(2, "segment", "002"), (100, "final", "auto100")])
def test_advance_constructs_single_next_submission_and_persists_lineage_before_callback(tmp_path, monkeypatch, through, kind, next_id):
    instance = bare_controller(tmp_path, through)
    commands, callbacks = [], []
    def execute(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "fixture receipt", "")
    monkeypatch.setattr(controller.subprocess, "run", execute)
    instance.submitted = lambda *_args: ("202", {})
    def callback(stage, job):
        # This catches the race where a callback could start before its parent
        # lineage decision was durable. A separate callback receipt follows.
        decision = controller.read(instance.root / "decisions/segment-001/decision.json")
        assert decision["status"] == "continued" and decision["next_stage"] == stage
        assert decision["next_job_id"] == job
        callbacks.append((stage, job))
        return "203"
    instance.queue_callback = callback
    instance.advance("segment-001")
    assert len(commands) == 1
    command, options = commands[0]
    assert command == ["/fixture/python", "/fixture/launcher.py", "--stage", kind,
        "--archive-sha256", "a" * 64, "--manifest-sha256", "b" * 64, "--control-sha256", "c" * 64,
        "--run-id", next_id, "--after-stage", "segment-001"] + (["--segment-id", next_id] if kind == "segment" else [])
    assert options["cwd"] == instance.release and options["timeout"] == 180
    output = instance.root / "decisions/segment-001"
    decision = controller.read(output / "decision.json")
    assert decision["status"] == ("continued" if kind == "segment" else "final_evaluation_submitted")
    if kind == "segment":
        assert callbacks == [("segment-002", "202")]
        assert controller.read(output / "callback.json")["next_callback_job_id"] == "203"
    else:
        assert callbacks == [] and not (output / "callback.json").exists()


def test_failed_health_records_failure_without_submission(tmp_path, monkeypatch):
    instance = bare_controller(tmp_path)
    def unhealthy(_stage):
        raise ValueError("incomplete epoch telemetry")
    instance.completed_health = unhealthy
    monkeypatch.setattr(controller.subprocess, "run", forbid_subprocess)
    instance.queue_callback = forbid_subprocess
    with pytest.raises(ValueError, match="incomplete epoch"):
        instance.advance("segment-001")
    failure = controller.read(instance.root / "decisions/segment-001/failure.json")
    assert failure["status"] == "stopped_inspect_before_retry"
    assert not (instance.root / "decisions/segment-001/submission-command.json").exists()


def test_duplicate_intent_cannot_retry_an_unknown_scheduler_outcome(tmp_path, monkeypatch):
    instance = bare_controller(tmp_path)
    path = instance.root / "decisions/segment-001/intent.json"
    controller.fresh(path, {"stage": "segment-001", "unknown_outcome": True})
    instance.completed_health = forbid_subprocess
    monkeypatch.setattr(controller.subprocess, "run", forbid_subprocess)
    with pytest.raises(FileExistsError):
        instance.advance("segment-001")
    assert controller.read(path)["unknown_outcome"] is True


def test_scheduler_nonzero_receipt_stops_without_callback_or_blind_retry(tmp_path, monkeypatch):
    instance = bare_controller(tmp_path)
    calls = []
    def failed(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "queue rejection")
    monkeypatch.setattr(controller.subprocess, "run", failed)
    instance.queue_callback = forbid_subprocess
    with pytest.raises(subprocess.CalledProcessError):
        instance.advance("segment-001")
    assert len(calls) == 1
    output = instance.root / "decisions/segment-001"
    assert controller.read(output / "submission-reply.json")["returncode"] == 1
    assert controller.read(output / "failure.json")["status"] == "stopped_inspect_before_retry"
    assert not (output / "decision.json").exists()


def test_callback_uses_ended_dependency_so_failed_predecessor_is_observed(tmp_path):
    instance = bare_controller(tmp_path)
    plans = []
    instance.submitted = lambda *_args: ("101", {})
    def submit(plan, receipts, identity):
        plans.append((plan, receipts, identity))
        return {"after-segment-001": "102"}
    instance.submit = submit
    assert instance.queue_callback("segment-001", "101") == "102"
    rows, receipts, identity = plans[0]
    assert len(rows) == 1 and rows[0]["dependency"] is None
    args = rows[0]["arguments"]
    assert args[args.index("-w") + 1] == "ended(101)"
    assert args[args.index("-q") + 1] == "hpc"
    assert rows[0]["job_command"] == ["/fixture/python", str(instance.bundle / "scripts/hpc/triangle_auto_continue.py"),
        "--policy", str(instance.policy_path), "--policy-sha256", instance.policy_sha,
        "--mode", "advance", "--after-stage", "segment-001"]
    assert receipts == instance.root / "receipts" and identity == instance.operator_identity
    with pytest.raises(ValueError, match="Wrong predecessor"):
        instance.queue_callback("segment-001", "999")
    assert len(plans) == 1


@pytest.mark.parametrize("stage", [None, "segment-000", "segment-021", "segment-1", "final-auto100", "segment-001;bad"])
def test_stage_name_is_bounded_and_cannot_inject_scheduler_text(stage):
    with pytest.raises(ValueError):
        controller.stage_number(stage)


def test_quota_headroom_is_bounded_and_includes_final_reserve():
    assert controller.required_quota_gib(0, 1) == Decimal("13")
    assert controller.required_quota_gib(100, 20) == Decimal("2")
    assert controller.required_quota_gib(99, 19) == Decimal("2.3125")
