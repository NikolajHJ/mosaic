"""Allocated metadata fixtures; synthetic records are not scientific evidence."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
import failure_guidance_campaign as campaign
from latentmol.experiments import file_sha256, write_json


REAL_SUITE = campaign.RepairSuite


def retry_fixture(tmp_path, monkeypatch):
    work = tmp_path / "work"
    old_root, new_root, source_root = (work / "runs" / name for name in ("old", "new", "source"))
    for path in (old_root, new_root, source_root, work / "logs"):
        path.mkdir(parents=True)
    policy = dict(version=campaign.VERSION, variants=list(campaign.IDS), source_plan="original-plan",
                  source_plan_sha256="c" * 64, source_job="905", source_callback_job="904")
    protocol = dict(prior_ratio=2, failure_guidance=dict(loss=dict(local_bond_weight=.1)))
    execution = dict(backend="optimized", workers=12, torch_threads=4, gpu_cpus=16, overlap=True)
    completed = []
    jobs = {"native": "901", "retirement-readiness": "902", "after-native": "903"}
    original = SimpleNamespace(root=old_root, receipts=old_root / "receipts", sha="a" * 64,
        plan=dict(protocol=deepcopy(protocol), execution=deepcopy(execution),
                  archive_sha256="d" * 64, manifest_sha256="e" * 64), policy=deepcopy(policy),
        receipt_job=lambda stage: jobs[stage], complete=lambda stage: completed.append(stage))
    source = SimpleNamespace(root=source_root, receipt_job=lambda stage: "904")
    suite = SimpleNamespace(root=new_root, work=work, sha="b" * 64, policy=policy,
        plan=dict(protocol=protocol, execution=execution), source=lambda: source)
    origin_path = work / "submissions/original-suite.json"

    def constructor(path, sha):
        assert str(path) == str(origin_path) and sha == original.sha
        return original
    monkeypatch.setattr(campaign, "RepairSuite", constructor)

    def query(command, **kwargs):
        assert command[:5] == ["bjobs", "-a", "-noheader", "-o", "stat"]
        assert kwargs == dict(capture_output=True, text=True, timeout=15)
        assert command[-1] in ("903", "904")
        return SimpleNamespace(returncode=0, stdout="", stderr=f"Job <{command[-1]}> is not found\n")
    monkeypatch.setattr(campaign.subprocess, "run", query)
    starting = dict(epoch=36, updates=101, path=str(source_root / "epoch_0036.pt"), sha256="f" * 64)
    write_json(old_root / "starting-checkpoint.json", starting)
    bound = dict(suite_sha256=original.sha, binding_sha256=file_sha256(old_root / "starting-checkpoint.json"),
        variants=[dict(id=name, path=str(old_root / "plans" / (name + ".json")), sha256=str(i) * 64)
                  for i, name in enumerate(campaign.IDS, 1)])
    write_json(old_root / "bound-plans.json", bound)
    selection = dict(suite_sha256=original.sha, job_id="901", winner=None,
        reason="no_passing_new_objective", bound_plans_sha256=file_sha256(old_root / "bound-plans.json"))
    write_json(old_root / "selection.json", selection)
    write_json(old_root / "promotion.json", dict(status="stopped_no_passing_new_objective", selection=selection))
    for row in bound["variants"][1:]:
        (old_root / "variants" / row["id"]).mkdir(parents=True)
        write_json(old_root / "variants" / row["id"] / "failure_readiness.json", dict(
            status="failed", job_id="901", plan_sha256=row["sha256"],
            archive_sha256=original.plan["archive_sha256"], manifest_sha256=original.plan["manifest_sha256"],
            error="RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation"))
    confirmation = work / "submissions/source123-retirement-confirmation-001/confirmation.json"
    confirmation.parent.mkdir(parents=True)
    write_json(confirmation, dict(suite_sha256=original.sha, callback_job="904", job_id="902",
        status="retired_confirmed", observed="observed_EXIT", returncode=0, stdout="EXIT\n", stderr=""))
    callback_log = work / "logs/original-after-native.903.out"
    callback_log.write_text("Subject: Job 903: <source104-chemistry-after-native> in cluster <dcc> Done\n"
                            "\nSuccessfully completed.\n")
    original.receipts.mkdir()
    write_json(original.receipts / "after-native-receipt.json", dict(intent=dict(command=[
        "bsub", "-oo", str(work / "logs/original-after-native.%J.out")]), job_id="903"))
    artifacts = {old_root / name for name in ("starting-checkpoint.json", "bound-plans.json", "selection.json",
        "promotion.json", "variants/fallback/failure_readiness.json", "variants/bond6/failure_readiness.json")}
    artifacts.update((confirmation, callback_log))
    suite.plan["retry_origin"] = dict(version="deterministic-repair-events-retry-v1", suite=str(origin_path),
        suite_sha256=original.sha, artifacts={str(path): file_sha256(path) for path in artifacts})
    return SimpleNamespace(suite=suite, original=original, source=source, starting=starting,
                           confirmation=confirmation, callback_log=callback_log, completed=completed)


def replace_artifact(case, path, value):
    write_json(path, value)
    case.suite.plan["retry_origin"]["artifacts"][str(path)] = file_sha256(path)


@pytest.mark.parametrize("expected", ["DONE", "EXIT"])
@pytest.mark.parametrize("returncode", [0, 255])
def test_exact_expiry_requires_separately_confirmed_terminal_evidence(expected, returncode):
    reply = SimpleNamespace(returncode=returncode, stdout="", stderr="Job <123> is not found\n")
    with pytest.raises(ValueError, match="terminal evidence"):
        campaign.require_terminal_metadata(reply, "123", expected=expected)
    assert campaign.require_terminal_metadata(reply, "123", expected=expected, confirmed_missing=True) == (
        "persisted_" + expected + "_now_aged_out")


@pytest.mark.parametrize("status", ["DONE", "EXIT"])
def test_successful_terminal_query_is_observed_directly(status):
    reply = SimpleNamespace(returncode=0, stdout=status + "\n", stderr="")
    assert campaign.require_terminal_metadata(reply, "123", expected=status) == "observed_" + status


@pytest.mark.parametrize("returncode,stdout,stderr", [
    (0, "RUN", ""), (0, "PEND", ""), (0, "EXIT", ""), (1, "DONE", ""),
    (0, "", ""), (0, "", "Permission denied"), (0, "", "Job <999> is not found"),
    (0, "RUN", "Job <123> is not found"), (0, "Job <123> is not found", ""),
    (0, "", "Job <123> is not found\nPermission denied"), (0, "DONE", "Permission denied"),
])
def test_live_wrong_or_ambiguous_scheduler_output_never_uses_persisted_proof(returncode, stdout, stderr):
    reply = SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    with pytest.raises(ValueError, match="terminal evidence"):
        campaign.require_terminal_metadata(reply, "123", expected="DONE", confirmed_missing=True)


def test_retry_authenticates_same_parent_and_both_persisted_terminal_proofs(tmp_path, monkeypatch):
    case = retry_fixture(tmp_path, monkeypatch)
    proof = campaign.authenticate_retry_origin(case.suite)
    assert proof["status"] == "authenticated_retry_origin"
    assert proof["starting_checkpoint"] == case.starting
    assert proof["original_suite_sha256"] == case.original.sha
    assert case.completed == ["native"]
    assert proof["observations"]["source_callback"]["observed"] == "persisted_EXIT_now_aged_out"
    assert proof["observations"]["old_native_callback"]["observed"] == "persisted_DONE_now_aged_out"


@pytest.mark.parametrize("change", ["loss", "execution", "source_policy", "same_campaign", "recursive_origin"])
def test_changed_objective_resources_source_or_reused_campaign_cannot_retry(tmp_path, monkeypatch, change):
    case = retry_fixture(tmp_path, monkeypatch)
    if change == "loss": case.suite.plan["protocol"]["failure_guidance"]["loss"]["local_bond_weight"] = .2
    elif change == "execution": case.suite.plan["execution"]["workers"] = 20
    elif change == "source_policy": case.suite.policy["source_job"] = "999"
    elif change == "same_campaign": case.original.root = case.suite.root
    else: case.original.plan["retry_origin"] = {}
    with pytest.raises(ValueError, match="changed"):
        campaign.authenticate_retry_origin(case.suite)


@pytest.mark.parametrize("change", ["missing", "unexpected", "changed_bytes", "deleted"])
def test_all_origin_evidence_remains_complete_and_hash_bound(tmp_path, monkeypatch, change):
    case = retry_fixture(tmp_path, monkeypatch)
    artifacts = case.suite.plan["retry_origin"]["artifacts"]
    path = case.confirmation
    if change == "missing": artifacts.pop(str(path))
    elif change == "unexpected": artifacts[str(case.suite.work / "extra.json")] = "0" * 64
    elif change == "changed_bytes": path.write_text("{}")
    else: path.unlink()
    with pytest.raises((ValueError, OSError)):
        campaign.authenticate_retry_origin(case.suite)


@pytest.mark.parametrize("change", ["suite", "binding", "selection", "order"])
def test_original_shared_parent_and_plan_evidence_must_agree(tmp_path, monkeypatch, change):
    case = retry_fixture(tmp_path, monkeypatch)
    path = case.original.root / "bound-plans.json"
    bound = campaign.read(path)
    if change == "suite": bound["suite_sha256"] = "0" * 64
    elif change == "binding": bound["binding_sha256"] = "0" * 64
    elif change == "order": bound["variants"].reverse()
    else:
        selection_path = case.original.root / "selection.json"
        selection = campaign.read(selection_path); selection["bound_plans_sha256"] = "0" * 64
        replace_artifact(case, selection_path, selection)
        replace_artifact(case, case.original.root / "promotion.json",
                         dict(status="stopped_no_passing_new_objective", selection=selection))
    if change != "selection": replace_artifact(case, path, bound)
    with pytest.raises(ValueError, match="shared parent binding"):
        campaign.authenticate_retry_origin(case.suite)


@pytest.mark.parametrize("field,value", [("status", "passed"), ("job_id", "999"), ("plan_sha256", "0" * 64),
    ("archive_sha256", "0" * 64), ("manifest_sha256", "0" * 64), ("error", "native gradient failed")])
def test_another_failure_is_not_the_authorized_cuda_execution_retry(tmp_path, monkeypatch, field, value):
    case = retry_fixture(tmp_path, monkeypatch)
    path = case.original.root / "variants/bond6/failure_readiness.json"
    record = campaign.read(path); record[field] = value
    replace_artifact(case, path, record)
    with pytest.raises(ValueError, match="different native failure"):
        campaign.authenticate_retry_origin(case.suite)


@pytest.mark.parametrize("field,value", [("job_id", "999"), ("callback_job", "999"),
    ("observed", "previously_confirmed_EXIT_now_aged_out"), ("stdout", "RUN"), ("stderr", "Permission denied")])
def test_callback_cancellation_requires_the_actual_allocated_exit_confirmation(tmp_path, monkeypatch, field, value):
    case = retry_fixture(tmp_path, monkeypatch)
    record = campaign.read(case.confirmation); record[field] = value
    replace_artifact(case, case.confirmation, record)
    with pytest.raises(ValueError, match="cancellation confirmation"):
        campaign.authenticate_retry_origin(case.suite)


@pytest.mark.parametrize("role", ["source_callback", "old_native_callback"])
def test_current_pending_or_running_callback_blocks_retry(tmp_path, monkeypatch, role):
    case = retry_fixture(tmp_path, monkeypatch)
    job = "904" if role == "source_callback" else "903"
    def query(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="RUN" if command[-1] == job else "",
                               stderr="" if command[-1] == job else f"Job <{command[-1]}> is not found")
    monkeypatch.setattr(campaign.subprocess, "run", query)
    with pytest.raises(ValueError, match="terminal evidence"):
        campaign.authenticate_retry_origin(case.suite)


@pytest.mark.parametrize("change", ["wrong_job", "exited", "no_success"])
def test_missing_callback_record_still_needs_exact_persisted_lsf_done(tmp_path, monkeypatch, change):
    case = retry_fixture(tmp_path, monkeypatch)
    value = case.callback_log.read_text()
    if change == "wrong_job": value = value.replace("Job 903:", "Job 999:")
    elif change == "exited": value += "Exited with exit code 1.\n"
    else: value = value.replace("Successfully completed.", "Incomplete")
    case.callback_log.write_text(value)
    case.suite.plan["retry_origin"]["artifacts"][str(case.callback_log)] = file_sha256(case.callback_log)
    with pytest.raises(ValueError, match="LSF DONE"):
        campaign.authenticate_retry_origin(case.suite)


@pytest.mark.parametrize("name", ["advance-after-segment-001-intent.json", "receipts/segment-002-intent.json"])
def test_an_original_source_successor_blocks_retry_even_when_pending(tmp_path, monkeypatch, name):
    case = retry_fixture(tmp_path, monkeypatch)
    path = case.source.root / name; path.parent.mkdir(exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="already acted"):
        campaign.authenticate_retry_origin(case.suite)


def test_retry_after_cpu_binds_directly_without_a_dependency_on_expired_source():
    calls = []
    suite = SimpleNamespace(plan={"retry_origin": {}}, require_cpu=lambda: calls.append("cpu"),
                            bind_parent=lambda: calls.append("bind"))
    REAL_SUITE.after_cpu(suite)
    assert calls == ["cpu", "bind"]


def test_retry_rejects_changed_parent_before_any_binding_write_or_queue(tmp_path, monkeypatch):
    case = retry_fixture(tmp_path, monkeypatch)
    suite = case.suite
    suite.require_cpu = lambda: None
    suite.retire_original_callback = lambda: None
    changed = dict(case.starting, epoch=37)
    monkeypatch.setattr(campaign, "healthy_handoff", lambda *args: changed)
    monkeypatch.setattr(campaign, "authenticate_retry_origin", lambda *args: dict(starting_checkpoint=case.starting))
    with pytest.raises(ValueError, match="starting checkpoint"):
        REAL_SUITE.bind_parent(suite)
    assert not (suite.root / "starting-checkpoint.json").exists()
    assert not (suite.root / "plans").exists()
