"""Allocated metadata fixtures; synthetic receipts are not scientific evidence."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
import failure_guidance_campaign as campaign
from latentmol.experiments import file_sha256, write_json


REAL_SUITE = campaign.RepairSuite


def predecessor_fixture(tmp_path, monkeypatch):
    work = tmp_path / "work"
    old_root, new_root = (work / "runs" / name for name in ("old", "new"))
    old_release, new_release = (work / "releases" / name for name in ("old", "new"))
    for path in (old_root, new_root, old_release, new_release, work / "logs"):
        path.mkdir(parents=True)
    starting = dict(epoch=36, updates=101, path=str(work / "epoch36.pt"), sha256="f" * 64)
    reference = dict(version="failed-repair-execution-reference-v1", starting_checkpoint=starting,
                     release=str(work / "releases/oracle"), module_sha256="d" * 64)
    policy = dict(version=campaign.VERSION, variants=list(campaign.IDS), source_plan="source121",
                  source_plan_sha256="c" * 64, source_job="905", source_callback_job="904")
    common = dict(protocol=dict(prior_ratio=2), execution=dict(backend="optimized", workers=12,
        torch_threads=4, gpu_cpus=16, overlap=True), config_sha256="1" * 64,
        dataset_sha256="2" * 64, parent_checkpoint_sha256="3" * 64, seed=17,
        execution_reference=deepcopy(reference))
    completed = []
    original = SimpleNamespace(root=old_root, release=old_release, receipts=old_root / "receipts",
        sha="a" * 64, policy=deepcopy(policy), plan=dict(deepcopy(common), release=str(old_release),
            campaign=str(old_root), archive_sha256="4" * 64, manifest_sha256="5" * 64),
        receipt_job=lambda stage: {"native": "901", "after-native": "903"}[stage],
        complete=lambda stage: completed.append(stage))
    suite = object.__new__(REAL_SUITE)
    suite.root, suite.work, suite.release, suite.sha = new_root, work, new_release, "b" * 64
    suite.policy = policy
    suite.plan = dict(deepcopy(common), release=str(new_release), campaign=str(new_root),
                      archive_sha256="6" * 64, manifest_sha256="7" * 64)
    suite.identity = dict(plan_sha256=suite.sha, archive_sha256="6" * 64,
                          manifest_sha256="7" * 64, parent_sha256="3" * 64)
    suite.receipt_job = lambda stage: "902"
    origin_path = work / "submissions/old.json"
    monkeypatch.setattr(campaign, "RepairSuite", lambda path, sha: original
        if str(path) == str(origin_path) and sha == original.sha else pytest.fail("Changed predecessor suite"))
    def checked_plan(path, sha):
        campaign.authenticated(path, sha)
        return campaign.read(path)
    monkeypatch.setattr(campaign, "validate_plan", checked_plan)
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **kwargs:
        SimpleNamespace(returncode=0, stdout="DONE\n", stderr=""))
    write_json(old_root / "starting-checkpoint.json", starting)
    rows = []
    for name in campaign.IDS:
        path = old_root / "plans" / (name + ".json")
        path.parent.mkdir(exist_ok=True)
        write_json(path, dict(campaign=str(old_root / "variants" / name), failure_variant_id=name,
            failure_suite_sha256=original.sha, starting_checkpoint=starting,
            archive_sha256=original.plan["archive_sha256"], manifest_sha256=original.plan["manifest_sha256"]))
        rows.append(dict(id=name, path=str(path), sha256=file_sha256(path)))
    write_json(old_root / "bound-plans.json", dict(suite_sha256=original.sha, variants=rows,
        binding_sha256=file_sha256(old_root / "starting-checkpoint.json")))
    selection = dict(suite_sha256=original.sha, job_id="901", winner=None,
        reason="no_passing_new_objective", bound_plans_sha256=file_sha256(old_root / "bound-plans.json"))
    write_json(old_root / "selection.json", selection)
    write_json(old_root / "promotion.json", dict(status="stopped_no_passing_new_objective", selection=selection))
    log = work / "logs/after-native.903.out"
    log.write_text("Subject: Job 903: <after-native> in cluster <dcc> Done\n\nSuccessfully completed.\n")
    original.receipts.mkdir()
    write_json(original.receipts / "after-native-receipt.json", dict(intent=dict(command=[
        "bsub", "-oo", str(work / "logs/after-native.%J.out")]), job_id="903"))
    artifacts = [old_root / name for name in
                 ("starting-checkpoint.json", "bound-plans.json", "selection.json", "promotion.json")]
    artifacts.append(log)
    suite.plan["execution_predecessor"] = dict(version="failed-repair-execution-predecessor-v1",
        suite=str(origin_path), suite_sha256=original.sha, release=str(old_release),
        archive_sha256=original.plan["archive_sha256"], manifest_sha256=original.plan["manifest_sha256"],
        starting_checkpoint=deepcopy(starting), artifacts={str(p): file_sha256(p) for p in artifacts})
    return SimpleNamespace(suite=suite, original=original, starting=starting, log=log, completed=completed)


def replace_artifact(case, path, value):
    write_json(path, value)
    case.suite.plan["execution_predecessor"]["artifacts"][str(path)] = file_sha256(path)


def test_optional_predecessor_keeps_historical_suites_unchanged():
    suite = object.__new__(REAL_SUITE)
    suite.plan = {}
    assert suite.execution_predecessor() is None


@pytest.mark.parametrize("aged_out", [False, True])
def test_same_source_stopped_campaign_and_terminal_callback_are_authenticated(tmp_path, monkeypatch, aged_out):
    case = predecessor_fixture(tmp_path, monkeypatch)
    if aged_out:
        monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs:
            SimpleNamespace(returncode=0, stdout="", stderr="Job <903> is not found\n"))
    assert case.suite.execution_predecessor() == case.suite.plan["execution_predecessor"]
    assert case.completed == ["native"]


@pytest.mark.parametrize("field", ["protocol", "execution", "config_sha256", "dataset_sha256",
                                   "parent_checkpoint_sha256", "seed", "execution_reference"])
def test_changed_objective_resources_source_or_fixed_oracle_cannot_reuse_predecessor(tmp_path, monkeypatch, field):
    case = predecessor_fixture(tmp_path, monkeypatch)
    case.suite.plan[field] = {"changed": True} if field in ("protocol", "execution", "execution_reference") else "changed"
    with pytest.raises(ValueError, match="changed"):
        case.suite.execution_predecessor()


@pytest.mark.parametrize("defect", ["missing", "extra", "changed_bytes", "wrong_callback_job", "no_done"])
def test_predecessor_requires_the_exact_complete_hashed_stop_evidence(tmp_path, monkeypatch, defect):
    case = predecessor_fixture(tmp_path, monkeypatch)
    artifacts = case.suite.plan["execution_predecessor"]["artifacts"]
    if defect == "missing": artifacts.pop(str(case.log))
    elif defect == "extra": artifacts[str(case.suite.work / "other.json")] = "0" * 64
    elif defect == "changed_bytes": case.log.write_text("changed")
    else:
        case.log.write_text(case.log.read_text().replace("Job 903:", "Job 999:")
                            if defect == "wrong_callback_job" else "Subject: Job 903: Failed\n")
        artifacts[str(case.log)] = file_sha256(case.log)
    with pytest.raises((ValueError, OSError)):
        case.suite.execution_predecessor()


@pytest.mark.parametrize("status", ["RUN", "PEND", "EXIT", "permission", "wrong_missing"])
def test_live_or_unknown_callback_never_uses_historical_done_proof(tmp_path, monkeypatch, status):
    case = predecessor_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0,
        stdout=status if status in ("RUN", "PEND", "EXIT") else "",
        stderr="Permission denied" if status == "permission" else "Job <999> is not found" if status == "wrong_missing" else ""))
    with pytest.raises(ValueError, match="terminal evidence"):
        case.suite.execution_predecessor()


@pytest.mark.parametrize("field,value", [("winner", "bond6"), ("job_id", "999"), ("reason", "unknown")])
def test_selected_or_unbound_outcome_is_not_a_stopped_predecessor(tmp_path, monkeypatch, field, value):
    case = predecessor_fixture(tmp_path, monkeypatch)
    selection = campaign.read(case.original.root / "selection.json")
    selection[field] = value
    replace_artifact(case, case.original.root / "selection.json", selection)
    replace_artifact(case, case.original.root / "promotion.json",
                     dict(status="stopped_no_passing_new_objective", selection=selection))
    with pytest.raises(ValueError, match="preserved stopped"):
        case.suite.execution_predecessor()


@pytest.mark.parametrize("relative", ["active-phase.json", "advance-after-benchmark-intent.json",
    "advance-after-benchmark.json", "receipts/segment-001-intent.json", "receipts/segment-001-receipt.json",
    "receipts/after-segment-001-intent.json", "receipts/final-intent.json", "segments/001/checkpoint.pt"])
@pytest.mark.parametrize("root", ["suite", "bond6"])
def test_any_pending_or_completed_training_successor_blocks_the_new_campaign(tmp_path, monkeypatch, relative, root):
    case = predecessor_fixture(tmp_path, monkeypatch)
    path = case.original.root / relative if root == "suite" else case.original.root / "variants/bond6" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture metadata, not a checkpoint")
    with pytest.raises(ValueError, match="successor"):
        case.suite.execution_predecessor()


def test_current_bound_start_must_match_the_immediate_predecessor(tmp_path, monkeypatch):
    case = predecessor_fixture(tmp_path, monkeypatch)
    write_json(case.suite.root / "starting-checkpoint.json", dict(case.starting, epoch=37))
    with pytest.raises(ValueError, match="current starting checkpoint"):
        case.suite.execution_predecessor()


def test_changed_parent_is_rejected_before_a_new_binding_or_queue(tmp_path, monkeypatch):
    case = predecessor_fixture(tmp_path, monkeypatch)
    case.suite.require_cpu = lambda: None
    case.suite.retire_original_callback = lambda: None
    case.suite.source = lambda: None
    monkeypatch.setattr(campaign, "healthy_handoff", lambda *args: dict(case.starting, epoch=37))
    with pytest.raises(ValueError, match="authenticated starting checkpoint"):
        case.suite.bind_parent()
    assert not (case.suite.root / "starting-checkpoint.json").exists()
    assert not (case.suite.root / "plans").exists()


def loss_gate_fixture(case):
    suite = case.suite
    path = suite.root / "loss-validation-pytest.xml"
    path.write_text('<testsuites><testsuite tests="22" failures="0" errors="0" skipped="0" /></testsuites>')
    reference = suite.plan["execution_reference"]
    gate = dict(status="passed", returncode=0, suite_sha256=suite.sha, identity=suite.identity,
        job_id="902", command=[str(suite.work / "env/boltz311/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--junitxml", str(path), "tests/test_failure_loss_validation_reuse.py"],
        environment=dict(LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES="1", LATENTMOL_FAILURE_REFERENCE_MODULE=str(
            Path(reference["release"]) / "src/latentmol/repair_failure_guidance.py")),
        reference_module_sha256=reference["module_sha256"], junit_sha256=file_sha256(path))
    write_json(suite.root / "loss-validation-gate.json", gate)
    return path, gate


def test_separate_cuda_loss_gate_binds_current_source_job_oracle_and_xml(tmp_path, monkeypatch):
    case = predecessor_fixture(tmp_path, monkeypatch)
    loss_gate_fixture(case)
    case.suite.require_loss_validation()


@pytest.mark.parametrize("field", ["status", "returncode", "suite_sha256", "identity", "job_id", "command",
                                   "environment", "reference_module_sha256", "junit_sha256"])
def test_cuda_loss_gate_cannot_be_replayed_or_weakened(tmp_path, monkeypatch, field):
    case = predecessor_fixture(tmp_path, monkeypatch)
    _, gate = loss_gate_fixture(case)
    gate[field] = "changed"
    write_json(case.suite.root / "loss-validation-gate.json", gate)
    with pytest.raises((ValueError, OSError)):
        case.suite.require_loss_validation()


@pytest.mark.parametrize("attribute,value", [("tests", "0"), ("failures", "1"), ("errors", "1"), ("skipped", "1")])
def test_cuda_loss_gate_needs_executed_unskipped_parity_cases(tmp_path, monkeypatch, attribute, value):
    case = predecessor_fixture(tmp_path, monkeypatch)
    path, gate = loss_gate_fixture(case)
    attributes = dict(tests="22", failures="0", errors="0", skipped="0")
    attributes[attribute] = value
    path.write_text('<testsuites><testsuite ' + ' '.join(k + '="' + v + '"' for k, v in attributes.items()) + ' /></testsuites>')
    gate["junit_sha256"] = file_sha256(path)
    write_json(case.suite.root / "loss-validation-gate.json", gate)
    with pytest.raises(ValueError, match="without failures or skips"):
        case.suite.require_loss_validation()


def test_new_metadata_and_both_loss_modules_are_in_the_allocated_cpu_gate():
    assert {"tests/test_failure_execution_predecessor.py", "tests/test_repair_loss_validation_reuse.py",
            "tests/test_failure_loss_validation_reuse.py"} <= set(campaign.SPEED_FIXTURES)
