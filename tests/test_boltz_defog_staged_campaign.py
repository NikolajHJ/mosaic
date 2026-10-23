"""Allocated metadata fixtures for serial <=24h long-comparison scheduling.

All jobs, scheduler replies and native-reader outputs here are fabricated. No
model, optimizer, live scheduler or checkpoint is executed by these fixtures.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from latentmol.config_identity import CONFIG_HASH_VERSION, config_sha256


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/hpc"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("_staged_campaign_fixture", SCRIPTS / "boltz_defog_staged_campaign.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _wrapper(campaign, job, command, *, cpu=False, **changes):
    directory = campaign.root / ("readiness-allocations" if cpu else "stage-allocations") / job
    config = {"command": command}
    value = {"status": "completed", "returncode": 0, "environment": {"job_id": job}, "config": config,
        "config_sha256": config_sha256(config), "config_hash_version": CONFIG_HASH_VERSION,
        "source_archive_sha256": campaign.manifest["source_archive_sha256"],
        "source": {"sha256": campaign.manifest["source_manifest_sha256"]}, **changes}
    _json(directory / "run.json", value)
    return directory / "run.json"


def _campaign(tmp_path):
    # Production __init__ authenticates the real /work3 namespace/archive.
    # These fixtures replace discovery only; all completion/queue readers run.
    campaign = module.StagedCampaign.__new__(module.StagedCampaign)
    campaign.work = tmp_path / "work"
    campaign.root = campaign.work / "submissions" / "round"
    campaign.release = campaign.work / "releases" / "release"
    campaign.release.mkdir(parents=True)
    campaign.root.mkdir(parents=True)
    campaign.path = campaign.root / "campaign.json"
    campaign.sha = "a" * 64
    campaign.receipts = campaign.root / "stage-submissions"
    output = campaign.work / "runs" / "comparison"
    plan_path = campaign.root / "readiness/prepared/plan.json"
    campaign.plan = {"output_root": str(output), "cases": []}
    for name in ("boltz_only", "boltz_defog"):
        path = plan_path.parent / (name + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        campaign.plan["cases"].append({"id": name, "config": module.ref(path)})
    _json(plan_path, campaign.plan)
    bootstrap_path = campaign.root / "bootstrap.json"
    campaign.bootstrap = {"archive_sha256": "b" * 64, "manifest_sha256": "c" * 64,
        "release": str(campaign.release), "control": str(campaign.root), "campaign": str(output)}
    _json(bootstrap_path, campaign.bootstrap)
    operator = campaign.root / "readiness_operator.py"
    operator.write_text("# fabricated allocated operator\n", encoding="utf-8")
    campaign.manifest = {"name": "round", "source_archive_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64, "plan": module.ref(plan_path),
        "bootstrap": module.ref(bootstrap_path), "readiness_operator": module.ref(operator)}
    readiness = {"status": "passed", "job_id": "101", "source": campaign.bootstrap,
        "bootstrap_sha256": campaign.manifest["bootstrap"]["sha256"],
        "operator_sha256": campaign.manifest["readiness_operator"]["sha256"],
        "plan": campaign.manifest["plan"], "tests": list((*module.TESTS, *module.EXTRA_TESTS)),
        "prepared_configurations": {row["id"]: row["config"]["sha256"] for row in campaign.plan["cases"]},
        "stages": {}}
    for name in ("preflight", "prepare", "validate"):
        log = campaign.root / "readiness" / (name + ".log")
        log.write_text(name, encoding="utf-8")
        readiness["stages"][name] = {"returncode": 0, "timed_out": False, "failure": None,
            "interrupted": False, "log": str(log), "log_sha256": module.file_sha256(log)}
    readiness_path = campaign.root / "readiness/completion.json"
    _json(readiness_path, readiness)
    campaign.manifest["cpu_readiness"] = module.ref(readiness_path)
    campaign.identity = {"campaign_sha256": campaign.sha, "plan_sha256": campaign.manifest["plan"]["sha256"],
        "archive_sha256": "b" * 64, "manifest_sha256": "c" * 64}
    command = ["python", str(operator), "--bootstrap", str(bootstrap_path),
        "--bootstrap-sha256", campaign.manifest["bootstrap"]["sha256"],
        "--operator-sha256", campaign.manifest["readiness_operator"]["sha256"]]
    _wrapper(campaign, "101", command, cpu=True)
    return campaign


def _scheduler(monkeypatch, *, status="DONE"):
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        if command[0] == "bjobs":
            return SimpleNamespace(returncode=0, stdout=status + "\n", stderr="")
        if command[0] == "bsub":
            return SimpleNamespace(returncode=0, stdout=f"Job <{200 + len([v for v in calls if v[0] == 'bsub'])}> is submitted to queue <fake>.\n", stderr="")
        stage = command[command.index("--stage") + 1]
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"status": "passed", "stage": stage,
            "case_ids": ["boltz_only"] if stage == "control" else ["boltz_only", "boltz_defog"]}))
    monkeypatch.setattr(module.subprocess, "run", invoke)
    monkeypatch.setattr(module, "quota_receipt", lambda: {"minimum_gib": 4, "remaining_gib": ["20"]})
    return calls


def _stage_receipt(campaign, stage, job, previous=None):
    path = Path(campaign.plan["output_root"]) / "staged" / stage / "receipt.json"
    report = path.parent / "work/report.json"
    _json(report, {"status": "accepted_terminal"})
    receipt = {"version": module.STAGE_VERSION, "stage": stage, "status": "passed", "job_id": job,
        "source": {"source_archive_sha256": "b" * 64, "source_manifest_sha256": "c" * 64},
        "plan": campaign.manifest["plan"], "budget": module.STAGE_BUDGETS[stage], "previous": previous,
        "report": module.ref(report)}
    _json(path, receipt)
    return path


@pytest.mark.parametrize("stage,hours", [("control", 18), ("guided", 24), ("evaluation", 6)])
def test_stage_resources_are_individual_24h_bounded(tmp_path, stage, hours):
    campaign = _campaign(tmp_path)
    arguments = campaign.arguments(stage)
    assert arguments[arguments.index("-W") + 1] == f"{hours}:00"
    assert arguments[arguments.index("-n") + 1] == "4"
    assert "rusage[mem=16GB]" in arguments
    assert arguments[arguments.index("-gpu") + 1] == "num=1:mode=exclusive_process:mps=yes:j_exclusive=yes"
    callback = campaign.arguments(stage, callback=True)
    assert callback[callback.index("-W") + 1] == "0:30"
    assert callback[callback.index("-q") + 1] == "hpc" and "-gpu" not in callback
    assert sum(row["allocation_seconds"] for row in module.STAGE_BUDGETS.values()) == 48 * 3600
    assert sum(row["internal_wall_seconds"] for row in module.STAGE_BUDGETS.values()) == 46 * 3600


def test_actual_readiness_and_durable_queue_are_idempotent(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = _scheduler(monkeypatch)
    first = campaign.queue_control()
    second = campaign.queue_control()
    assert first["jobs"] == second["jobs"]
    submitted = [command for command in calls if command[0] == "bsub"]
    assert len(submitted) == 2
    assert "ended(201)" in submitted[1]
    assert "--stage control" in submitted[0][-1]
    assert "advance --after-stage control" in submitted[1][-1]


@pytest.mark.parametrize("mutation", ["status", "missing_test", "process", "log", "config", "source", "operator"])
def test_readiness_failure_never_queues_native(tmp_path, monkeypatch, mutation):
    campaign = _campaign(tmp_path)
    calls = _scheduler(monkeypatch)
    path = Path(campaign.manifest["cpu_readiness"]["path"])
    record = module.read(path)
    if mutation == "status": record["status"] = "failed"
    elif mutation == "missing_test": record["tests"].remove(module.EXTRA_TESTS[0])
    elif mutation == "process": record["stages"]["prepare"]["returncode"] = 1
    elif mutation == "log": (campaign.root / "readiness/validate.log").write_text("changed")
    elif mutation == "config": record["prepared_configurations"]["boltz_only"] = "d" * 64
    elif mutation == "source": record["source"] = {"different": True}
    elif mutation == "operator": record["operator_sha256"] = "d" * 64
    _json(path, record)
    campaign.manifest["cpu_readiness"] = module.ref(path)
    with pytest.raises(ValueError): campaign.queue_control()
    assert not any(command[0] == "bsub" for command in calls)


@pytest.mark.parametrize("status", ["EXIT", "PEND", "UNKNOWN"])
def test_failed_scheduler_cannot_be_overridden_by_completed_files(tmp_path, monkeypatch, status):
    campaign = _campaign(tmp_path)
    calls = _scheduler(monkeypatch, status=status)
    with pytest.raises(ValueError, match="did not finish"): campaign.queue_control()
    assert not any(command[0] == "bsub" for command in calls)


def test_unique_wrapper_is_required_even_when_scheduler_is_done(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = _scheduler(monkeypatch)
    value = module.read(campaign.root / "readiness-allocations/101/run.json")
    _json(campaign.root / "readiness-allocations/duplicate/run.json", value)
    with pytest.raises(ValueError, match="unique"): campaign.queue_control()
    assert not any(command[0] == "bsub" for command in calls)


def test_unknown_submission_retains_exclusive_intent(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _scheduler(monkeypatch)
    def unknown(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 60)
    monkeypatch.setattr(module.subprocess, "run", unknown)
    monkeypatch.setattr(campaign, "require_cpu", lambda: {})
    with pytest.raises(RuntimeError, match="unknown"): campaign.queue_control()
    assert (campaign.receipts / "control-intent.json").is_file()
    with pytest.raises(FileExistsError): campaign.queue_control()


def test_advance_requires_done_wrapper_native_validation_then_submits_guided(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = _scheduler(monkeypatch)
    queued = campaign.queue_control()["jobs"]
    _wrapper(campaign, queued["control"], campaign._stage_command("control"))
    _stage_receipt(campaign, "control", queued["control"])
    monkeypatch.setenv("LSB_JOBID", queued["after-control"])
    result = campaign.advance("control")
    assert result["status"] == "successor_submitted_not_passed"
    validations = [command for command in calls if "validate-receipt" in command]
    assert len(validations) == 1 and validations[0][validations[0].index("--stage") + 1] == "control"
    submitted = [command for command in calls if command[0] == "bsub"]
    assert len(submitted) == 4 and "--stage guided" in submitted[2][-1]
    assert "--previous-receipt" in submitted[2][-1]
    campaign.advance("control")
    assert len([command for command in calls if command[0] == "bsub"]) == 4


@pytest.mark.parametrize("mutation", ["job", "source", "budget", "status", "previous"])
def test_stage_receipt_mismatch_blocks_successor(tmp_path, monkeypatch, mutation):
    campaign = _campaign(tmp_path)
    calls = _scheduler(monkeypatch)
    queued = campaign.queue_control()["jobs"]
    _wrapper(campaign, queued["control"], campaign._stage_command("control"))
    path = _stage_receipt(campaign, "control", queued["control"])
    receipt = module.read(path)
    if mutation == "job": receipt["job_id"] = "999"
    elif mutation == "source": receipt["source"]["source_archive_sha256"] = "d" * 64
    elif mutation == "budget": receipt["budget"] = {"allocation_seconds": 90000}
    elif mutation == "status": receipt["status"] = "failed"
    elif mutation == "previous": receipt["previous"] = {"path": "unexpected", "sha256": "d" * 64}
    _json(path, receipt)
    monkeypatch.setenv("LSB_JOBID", queued["after-control"])
    with pytest.raises(ValueError): campaign.advance("control")
    assert len([command for command in calls if command[0] == "bsub"]) == 2


def test_native_reader_failure_is_preserved_and_never_promoted(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = _scheduler(monkeypatch)
    queued = campaign.queue_control()["jobs"]
    _wrapper(campaign, queued["control"], campaign._stage_command("control"))
    _stage_receipt(campaign, "control", queued["control"])
    monkeypatch.setenv("LSB_JOBID", queued["after-control"])
    real = module.subprocess.run
    def rejected(command, **kwargs):
        if "validate-receipt" in command:
            return SimpleNamespace(returncode=1, stdout="{}", stderr="failed original guard")
        return real(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", rejected)
    with pytest.raises(ValueError, match="native stage"): campaign.advance("control")
    assert module.read(campaign.root / "stage-validation/control/result.json")["status"] == "failed"
    assert len([command for command in calls if command[0] == "bsub"]) == 2


def test_advance_rejects_unallocated_or_wrong_callback(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    _scheduler(monkeypatch)
    campaign.queue_control()
    monkeypatch.setenv("LSB_JOBID", "999")
    with pytest.raises(ValueError, match="actual declared allocated callback"): campaign.advance("control")


@pytest.mark.parametrize("defect", ["stage", "membership", "allocation"])
def test_saved_validation_for_another_stage_or_allocation_cannot_be_reused(tmp_path, monkeypatch, defect):
    campaign = _campaign(tmp_path)
    _scheduler(monkeypatch)
    queued = campaign.queue_control()["jobs"]
    _wrapper(campaign, queued["control"], campaign._stage_command("control"))
    _stage_receipt(campaign, "control", queued["control"])
    monkeypatch.setenv("LSB_JOBID", queued["after-control"])
    campaign.advance("control")
    saved = campaign.root / "stage-validation/control/result.json"
    value = module.read(saved)
    if defect == "stage": value["validation"]["stage"] = "guided"
    elif defect == "membership": value["validation"]["case_ids"] = ["boltz_only", "boltz_defog"]
    else: value["callback_job_id"] = "999"
    _json(saved, value)
    with pytest.raises(ValueError, match="Existing stage validation"):
        campaign.advance("control")


def test_final_evaluation_validation_does_not_submit_a_fourth_stage(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = _scheduler(monkeypatch)
    queued = campaign.queue_control()["jobs"]
    previous = None
    for stage in module.STAGES:
        job = campaign.receipt_job(stage)
        _wrapper(campaign, job, campaign._stage_command(stage, previous))
        path = _stage_receipt(campaign, stage, job, previous)
        monkeypatch.setenv("LSB_JOBID", campaign.receipt_job("after-" + stage))
        result = campaign.advance(stage)
        previous = module.ref(path)
    assert result["status"] == "comparison_terminal_validated"
    assert result["final_report"] == module.read(path)["report"]
    assert len([command for command in calls if command[0] == "bsub"]) == 6


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}',
    '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e400}', '{"x":-1e400}'])
def test_campaign_json_rejects_duplicate_or_nonfinite_metadata(text):
    with pytest.raises(ValueError):
        module.strict_json(text)


def test_campaign_json_preserves_valid_scalar_types_and_nested_values():
    value = module.strict_json('{"values":[1,1.0,true,null,"molecule"],"nested":{"a":2e-5}}')
    assert value == {"values": [1, 1.0, True, None, "molecule"], "nested": {"a": 2e-5}}
    assert type(value["values"][0]) is int and type(value["values"][1]) is float
    assert type(value["values"][2]) is bool


def test_controller_has_no_dependency_on_chemistry_campaign_modules():
    import ast
    tree = ast.parse(Path(module.__file__).read_text())
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "triangle_continuation_health" not in modules and "chemistry_campaign" not in modules
    assert "latentmol.config_identity" in modules
