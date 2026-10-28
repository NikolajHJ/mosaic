"""Allocated stdlib-controller fixtures; no real scheduler/model operations.

These tests are part of the sealed HPC CPU gate. All scheduler/quota processes
are fakes and fixture files stay under pytest's temporary directory.
"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tarfile

import pytest


def load_module():
    location = Path(__file__).resolve().parents[1] / "scripts/hpc/embedding_long_campaign.py"
    spec = importlib.util.spec_from_file_location("embedding_long_campaign_fixture", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    mod = load_module()
    work = tmp_path / "work"
    monkeypatch.setattr(mod, "ROOT", work)
    release = work / "releases" / mod.NAMESPACE.replace("source142", "source-142", 1)
    config = release / mod.CONFIG
    settings = dict(version="guarded-embedding-long-v1", case_ids=list(mod.CASES),
        updates_per_case=10000, observe_every=10, segment_internal_seconds=82800,
        segment_wall="24:00", pilot_safety_factor=1.5, segment_reserve_seconds=1800,
        max_segments_per_case=1000)
    save(config, {"long_campaign": settings})
    manifest = release / "embedding-release-manifest.json"
    save(manifest, {"members": {mod.CONFIG: mod.digest(config)}})
    archive = Path(str(release) + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        for path in (config, manifest):
            stream.add(path, arcname=path.relative_to(release).as_posix(), recursive=False)
    campaign = mod.Campaign(release, work / "runs" / mod.NAMESPACE,
        mod.digest(archive), mod.digest(manifest), mod.digest(config))
    campaign.prepare()
    frozen = {"vae": "1" * 64, "native": "2" * 64, "starts": "3" * 64}

    def asset(node, name, content=b"fixture asset"):
        path = campaign.root / node / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": name, "sha256": mod.digest(path)}

    def result(node, **extra):
        value = dict(campaign.identity, status="passed", stage=mod.stage_for(node),
            frozen_identities=dict(frozen), artifacts={})
        value.update(extra)
        save(campaign.root / node / "result.json", value)
        return value

    junit = asset("cpu", "cpu-tests.xml", b'<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"><testcase name="one"/><testcase name="two"/></testsuite></testsuites>')
    cpu = result("cpu", artifacts={"junit": junit}, tests_passed=2)
    checkpoint, calibration = asset("fit", "bridge.pt"), asset("fit", "calibration.json", b'{}')
    model = {"checkpoint_sha256": checkpoint["sha256"], "calibration_sha256": calibration["sha256"]}
    fit = result("fit", artifacts={"checkpoint": checkpoint, "calibration": calibration},
        fit_completed=True, calibration_passed=True, model_identity=model)
    profile = dict(seconds_per_update=20.0, startup_seconds=120.0, bytes_per_update=100.0, fixed_bytes=1024.0)
    cases, artifacts = [], {}
    for case in mod.CASES:
        artifacts[case + "_state"] = asset("pilot", case + ".pt", case.encode())
        cases.append(dict(case=case, completed_updates=20, gates_passed=True,
            strict_valid_best=True, accepted_nonzero_steps=1, profile=dict(profile)))
    pilot = result("pilot", artifacts=artifacts, model_identity=model, cases=cases)

    def completed(node, request=None, job="1234"):
        intent = dict(identity=campaign.identity, release=str(campaign.release), campaign=str(campaign.root),
            node=node, label=campaign.root.name + "-" + node, request=request or {})
        save(campaign.control / (node + "-intent.json"), intent)
        save(campaign.control / (node + "-receipt.json"), dict(status="submitted_not_passed", job_id=job,
            intent_sha256=mod.digest(campaign.control / (node + "-intent.json"))))
        save(campaign.control / (node + "-execution.json"), dict(identity=campaign.identity, node=node,
            job_id=job, returncode=0, status="completed", result_sha256=mod.digest(campaign.root / node / "result.json")))
        return intent

    def scheduler(command, **kwargs):
        if command[0] == "bjobs":
            job = command[-1]
            candidates = [path for path in campaign.control.glob("*-receipt.json") if "callback" not in path.name]
            selected = next(path for path in candidates if json.loads(path.read_text())["job_id"] == job)
            node = selected.name.removesuffix("-receipt.json")
            return SimpleNamespace(returncode=0, stdout=job + " DONE " + campaign.root.name + "-" + node + "\n", stderr="")
        raise AssertionError("Unexpected real operation: " + repr(command))

    monkeypatch.setattr(mod.subprocess, "run", scheduler)
    monkeypatch.setattr(mod, "quota", lambda: {"minimum_free_gib": 20.0, "stdout": "fresh fixture"})
    return SimpleNamespace(mod=mod, campaign=campaign, result=result, asset=asset, completed=completed,
        frozen=frozen, model=model, cpu=cpu, fit=fit, pilot=pilot, profile=profile, settings=settings,
        scheduler=scheduler)


def test_no_model_imports_in_controller():
    import ast
    path = Path(__file__).resolve().parents[1] / "scripts/hpc/embedding_long_campaign.py"
    tree = ast.parse(path.read_text())
    names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and name.split(".")[0] in {"torch", "numpy", "latentmol", "boltz"} for name in names)


@pytest.mark.parametrize("node,stage", [("cpu", "cpu"), ("fit", "fit"), ("pilot", "pilot"),
    ("case-001-segment-0001", "segment"), ("case-003-segment-1000", "segment"), ("followup", "followup")])
def test_closed_node_names(scenario, node, stage):
    assert scenario.mod.stage_for(node) == stage


@pytest.mark.parametrize("node", ["gpu", "../cpu", "case-004-segment-0001", "case-001-segment-1", "case-001-segment-0001/extra"])
def test_unknown_node_names_rejected(scenario, node):
    with pytest.raises(ValueError, match="Unknown"):
        scenario.mod.stage_for(node)


def test_authentic_cpu_fit_pilot_contracts_pass(scenario):
    for node in ("cpu", "fit", "pilot"):
        assert scenario.campaign.result(node)["status"] == "passed"


@pytest.mark.parametrize("key", ["source_sha256", "manifest_sha256", "plan_sha256"])
def test_result_identity_tamper_stops_handoff(scenario, key):
    value = copy.deepcopy(scenario.cpu)
    value[key] = "f" * 64
    save(scenario.campaign.root / "cpu/result.json", value)
    with pytest.raises(ValueError, match="source/manifest/plan"):
        scenario.campaign.result("cpu")


@pytest.mark.parametrize("xml", [b'<testsuite tests="0"/>',
    b'<testsuite><testcase><failure/></testcase></testsuite>',
    b'<testsuite><testcase><error/></testcase></testsuite>',
    b'<testsuite><testcase><skipped/></testcase></testsuite>',
    b'<testsuite errors="1"><testcase/></testsuite>'])
def test_junit_no_empty_fail_error_or_skip_waiver(scenario, xml):
    value = copy.deepcopy(scenario.cpu)
    value["artifacts"]["junit"] = scenario.asset("cpu", "rejected.xml", xml)
    save(scenario.campaign.root / "cpu/result.json", value)
    with pytest.raises(ValueError, match="JUnit"):
        scenario.campaign.result("cpu")


def test_junit_count_must_equal_structured_receipt(scenario):
    value = dict(scenario.cpu, tests_passed=3)
    save(scenario.campaign.root / "cpu/result.json", value)
    with pytest.raises(ValueError, match="count"):
        scenario.campaign.result("cpu")


@pytest.mark.parametrize("kind", ["bytes", "escape", "missing"])
def test_artifact_tamper_or_escape_rejected(scenario, kind):
    value = copy.deepcopy(scenario.fit)
    if kind == "bytes":
        (scenario.campaign.root / "fit/bridge.pt").write_bytes(b"changed")
    elif kind == "escape":
        value["artifacts"]["checkpoint"]["path"] = "../cpu/cpu-tests.xml"
    else:
        del value["artifacts"]["checkpoint"]
    save(scenario.campaign.root / "fit/result.json", value)
    with pytest.raises(ValueError, match="Artifact|artifact"):
        scenario.campaign.result("fit")


@pytest.mark.parametrize("field", ["fit_completed", "calibration_passed"])
def test_incomplete_fit_calibration_stops_pilot(scenario, field):
    value = dict(scenario.fit, **{field: False})
    save(scenario.campaign.root / "fit/result.json", value)
    with pytest.raises(ValueError, match="incomplete"):
        scenario.campaign.result("fit")


@pytest.mark.parametrize("field,value", [("gates_passed", False), ("strict_valid_best", False),
    ("accepted_nonzero_steps", 0), ("completed_updates", 0), ("completed_updates", 10000)])
def test_pilot_requires_every_fixed_case_and_real_progress(scenario, field, value):
    result = copy.deepcopy(scenario.pilot)
    result["cases"][2][field] = value
    save(scenario.campaign.root / "pilot/result.json", result)
    with pytest.raises(ValueError):
        scenario.campaign.result("pilot")


@pytest.mark.parametrize("kind", ["drop", "reorder", "duplicate", "model", "frozen"])
def test_pilot_identity_selection_and_order_cannot_drift(scenario, kind):
    value = copy.deepcopy(scenario.pilot)
    if kind == "drop":
        value["cases"].pop()
    elif kind == "reorder":
        value["cases"].reverse()
    elif kind == "duplicate":
        value["cases"][2] = value["cases"][0]
    elif kind == "model":
        value["model_identity"]["checkpoint_sha256"] = "f" * 64
    else:
        value["frozen_identities"]["vae"] = "f" * 64
    save(scenario.campaign.root / "pilot/result.json", value)
    with pytest.raises(ValueError):
        scenario.campaign.result("pilot")


@pytest.mark.parametrize("key,value", [("seconds_per_update", 0), ("seconds_per_update", -1),
    ("seconds_per_update", float("inf")), ("startup_seconds", float("nan")),
    ("bytes_per_update", -1), ("fixed_bytes", None), ("fixed_bytes", True)])
def test_profiles_reject_unmeasured_or_nonfinite_values(scenario, key, value):
    profile = dict(scenario.profile, **{key: value})
    with pytest.raises(ValueError, match="finite"):
        scenario.campaign.profile(profile)


def test_first_segment_counts_pilot_and_uses_conservative_budget(scenario):
    node, request = scenario.campaign.next_segment("pilot")
    assert node == "case-001-segment-0001"
    assert request["start_update"] == 20 and request["endpoint"] == 10000
    assert request["max_updates"] == int((82800 - 120 - 1800) / (20 * 1.5))
    assert request["observation_every"] == 10
    assert request["resume_state_sha256"] == scenario.pilot["artifacts"]["case-001_state"]["sha256"]


def add_segment(scenario, completed=50):
    campaign, mod = scenario.campaign, scenario.mod
    node, request = campaign.next_segment("pilot")
    state = scenario.asset(node, "state.pt", b"continued state")
    scenario.result(node, status="completed_case" if completed == 10000 else "completed_segment",
        artifacts={"state": state}, case="case-001", start_update=request["start_update"],
        completed_updates=completed, resume_state_sha256=request["resume_state_sha256"],
        model_identity=scenario.model, guards_passed=True, guard_stall=False, observation_every=10)
    scenario.completed(node, request)
    return node, request


def test_round_robin_moves_to_second_case_after_authenticated_first_segment(scenario):
    node, _ = add_segment(scenario)
    next_node, request = scenario.campaign.next_segment(node)
    assert next_node == "case-002-segment-0001" and request["start_update"] == 20
    assert scenario.campaign.progress()["case-001"]["completed_updates"] == 50


@pytest.mark.parametrize("field,value", [("start_update", 0), ("completed_updates", 20),
    ("completed_updates", 10001), ("case", "case-002"), ("resume_state_sha256", "a" * 64),
    ("guards_passed", False), ("guard_stall", True), ("observation_every", 100),
    ("status", "scientific_stall_target_incomplete")])
def test_segment_tamper_stall_and_budget_reset_stop_successors(scenario, field, value):
    node, _ = add_segment(scenario)
    result = json.loads((scenario.campaign.root / node / "result.json").read_text())
    result[field] = value
    save(scenario.campaign.root / node / "result.json", result)
    with pytest.raises(ValueError):
        scenario.campaign.result(node)


@pytest.mark.parametrize("state", ["RUN", "PEND", "EXIT", "DONE wrong-name"])
def test_current_scheduler_must_confirm_exact_job_name_done(scenario, monkeypatch, state):
    scenario.completed("cpu")
    monkeypatch.setattr(scenario.mod.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout="1234 " + state + " " + scenario.campaign.root.name + "-cpu", stderr=""))
    with pytest.raises(ValueError, match="exact-name DONE"):
        scenario.campaign.require_done("cpu")


def test_aged_scheduler_requires_exact_saved_execution_evidence(scenario, monkeypatch):
    scenario.completed("cpu")
    monkeypatch.setattr(scenario.mod.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=255, stdout="", stderr="Job <1234> is not found"))
    scenario.campaign.require_done("cpu")
    execution = scenario.campaign.control / "cpu-execution.json"
    value = json.loads(execution.read_text())
    value["job_id"] = "9999"
    save(execution, value)
    with pytest.raises(ValueError, match="allocated execution"):
        scenario.campaign.require_done("cpu")


def test_changed_result_after_execution_is_not_a_saved_pass(scenario):
    scenario.completed("cpu")
    value = dict(scenario.cpu, altered=True)
    save(scenario.campaign.root / "cpu/result.json", value)
    with pytest.raises(ValueError, match="allocated execution"):
        scenario.campaign.require_done("cpu")


@pytest.mark.parametrize("existing", ["intent", "receipt", "callback-intent", "callback-receipt"])
def test_prior_submission_attempt_stops_before_quota_or_scheduler(scenario, monkeypatch, existing):
    path = scenario.campaign.control / ("cpu-" + existing + ".json")
    path.write_text("{}")
    monkeypatch.setattr(scenario.mod, "quota", lambda: pytest.fail("must not query quota for duplicate"))
    with pytest.raises(FileExistsError, match="already attempted"):
        scenario.campaign.dispatch("cpu")


def test_confirmed_worker_gets_separate_one_slot_ended_callback(scenario, monkeypatch):
    calls = []
    def bsub(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout="Job <" + str(111 + len(calls)) + "> is submitted", stderr="")
    monkeypatch.setattr(scenario.mod.subprocess, "run", bsub)
    receipt = scenario.campaign.dispatch("cpu")
    assert receipt["worker"]["job_id"] == "112"
    assert calls[0][0][calls[0][0].index("-n") + 1] == "4"
    assert calls[1][0][calls[1][0].index("-n") + 1] == "1"
    assert "ended(112)" in calls[1][0] and "0:30" in calls[1][0]
    assert "advance --release" in calls[1][1]["input"] and "--after cpu" in calls[1][1]["input"]


def test_ambiguous_scheduler_timeout_never_retries(scenario, monkeypatch):
    calls = []
    def timeout(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired("bsub", 60)
    monkeypatch.setattr(scenario.mod.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="unknown"):
        scenario.campaign.dispatch("cpu")
    receipt = json.loads((scenario.campaign.control / "cpu-receipt.json").read_text())
    assert receipt["status"] == "ambiguous_submission_no_retry"
    with pytest.raises(FileExistsError):
        scenario.campaign.dispatch("cpu")
    assert len(calls) == 1


def test_projected_storage_guard_prevents_gpu_submission(scenario, monkeypatch):
    node, request = scenario.campaign.next_segment("pilot")
    request["projected_bytes"] = 16 * 1024 ** 3
    monkeypatch.setattr(scenario.mod.subprocess, "run", lambda *args, **kwargs: pytest.fail("no bsub expected"))
    with pytest.raises(RuntimeError, match="storage"):
        scenario.campaign.dispatch(node, request)


def test_advance_failure_does_not_write_transition_or_queue_gpu(scenario):
    scenario.completed("cpu")
    value = dict(scenario.cpu, status="failed")
    save(scenario.campaign.root / "cpu/result.json", value)
    with pytest.raises(ValueError):
        scenario.campaign.advance("cpu")
    assert not (scenario.campaign.control / "advance-after-cpu.json").exists()
    assert not (scenario.campaign.control / "fit-intent.json").exists()


def test_actual_worker_cli_binds_provenance_and_resource_budget(scenario):
    command = scenario.campaign.worker_command("cpu", {})
    assert command[1:3] == ["-m", "latentmol.boltz.embedding_long_experiment"]
    for flag, value in (("--stage", "cpu"), ("--config", str(scenario.campaign.release / scenario.mod.CONFIG)),
        ("--source-sha256", scenario.campaign.archive_sha256),
        ("--plan-sha256", scenario.campaign.plan_sha256),
        ("--manifest-sha256", scenario.campaign.identity["manifest_sha256"]), ("--seconds", "2400")):
        assert command[command.index(flag) + 1] == value


def test_segment_cli_uses_exact_saved_resume_without_budget_reset(scenario):
    node, request = scenario.campaign.next_segment("pilot")
    command = scenario.campaign.worker_command(node, {"request": request})
    assert command[command.index("--case") + 1] == "case-001"
    assert command[command.index("--resume-state") + 1] == request["resume"]
    assert command[command.index("--resume-sha256") + 1] == request["resume_state_sha256"]
    assert command[command.index("--segment-updates") + 1] == str(request["max_updates"])
    assert command[command.index("--seconds") + 1] == "82800"


def test_allocated_stage_rejects_missing_lsf_ownership(scenario, monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(RuntimeError, match="LSF"):
        scenario.campaign.run_stage("cpu")


def test_bounded_metadata_rejects_nonfinite_json_and_oversize(scenario, tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        scenario.mod.read(path)
    path.write_text("{}" * 10)
    with pytest.raises(ValueError, match="bounded"):
        scenario.mod.read(path, limit=4)


def test_artifacts_cannot_live_inside_immutable_release(scenario):
    value = copy.deepcopy(scenario.fit)
    value["artifacts"]["checkpoint"]["path"] = str(scenario.campaign.release / scenario.mod.CONFIG)
    save(scenario.campaign.root / "fit/result.json", value)
    with pytest.raises(ValueError, match="outside"):
        scenario.campaign.result("fit")


def test_final_followup_cannot_start_before_all_endpoints(scenario):
    with pytest.raises(ValueError, match="before all"):
        scenario.campaign.worker_command("followup", {})


def test_cpu_kernel_environment_is_set_before_worker_import_only(scenario, monkeypatch):
    monkeypatch.delenv("MKL_CBWR", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    environment = scenario.mod.worker_environment("cpu")
    assert environment["MKL_CBWR"] == "COMPATIBLE"
    assert environment["OMP_NUM_THREADS"] == environment["MKL_NUM_THREADS"] == "4"
    gpu = scenario.mod.worker_environment("pilot")
    assert "MKL_CBWR" not in gpu and gpu["OMP_NUM_THREADS"] == "7"


def test_generated_cpu_argv_reaches_real_experiment_main_with_exact_provenance(scenario, monkeypatch):
    # Exercise the actual parser/dispatch/provenance boundary without any model
    # setup, dataset loading, scheduler access or CUDA operation.
    from latentmol.boltz import embedding_long_experiment as experiment
    output = scenario.campaign.root / "cpu-cli"
    command = scenario.campaign.worker_command("cpu", {})
    command[command.index("--output") + 1] = str(output)
    calls = []
    monkeypatch.setattr(experiment, "validate_config", lambda config: {})
    monkeypatch.setattr(experiment, "frozen_identities", lambda config: scenario.frozen)
    monkeypatch.setattr(experiment, "cpu", lambda config, target, deadline:
        calls.append((target, deadline)) or dict(status="passed", artifacts={}, tests_passed=0))
    monkeypatch.setenv("LSB_JOBID", "142001")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", scenario.campaign.archive_sha256)
    monkeypatch.setattr(experiment.sys, "argv", ["embedding_long_experiment", *command[3:]])
    experiment.main()
    result = json.loads((output / "result.json").read_text())
    assert len(calls) == 1 and calls[0][0] == output
    assert result["status"] == "passed" and result["stage"] == "cpu"
    assert result["source_sha256"] == scenario.campaign.archive_sha256
    assert result["plan_sha256"] == scenario.campaign.plan_sha256
    assert result["manifest_sha256"] == scenario.campaign.identity["manifest_sha256"]
    assert result["frozen_identities"] == scenario.frozen


def test_followup_resource_budget_remains_full_day(scenario, monkeypatch):
    calls = []
    def bsub(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout="Job <" + str(9000 + len(calls)) + "> submitted", stderr="")
    monkeypatch.setattr(scenario.mod.subprocess, "run", bsub)
    scenario.campaign.dispatch("followup")
    assert calls[0][calls[0].index("-W") + 1] == "24:00"
    # Endpoints are validated before execution/selection, not replaced by a
    # shortened native followup allocation after the last optimization segment.


def test_predecessor_metadata_is_pinned_into_submission(scenario, monkeypatch):
    monkeypatch.setattr(scenario.mod.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout="Job <9999> submitted", stderr=""))
    scenario.campaign.dispatch("fit")
    intent = scenario.campaign.node_intent("fit")
    assert intent["dependencies"] == {"cpu": scenario.mod.digest(scenario.campaign.root / "cpu/result.json")}


def test_new_immutable_attempt_gets_separate_exact_release_run_and_controls(scenario):
    old = scenario.campaign
    new_release = old.release.with_name(old.release.name.replace("attempt001", "attempt002"))
    old.release.rename(new_release)
    Path(str(old.release) + ".tar.gz").rename(Path(str(new_release) + ".tar.gz"))
    new = scenario.mod.Campaign(new_release, old.root.with_name(old.root.name.replace("attempt001", "attempt002")),
        old.archive_sha256, old.identity["manifest_sha256"], old.plan_sha256)
    assert new.control.name.endswith("attempt002")
    assert new.control != old.control and new.root != old.root
    assert new.release.name == new.root.name.replace("source142", "source-142", 1)


def test_zero_attempt_namespace_is_rejected_before_any_external_action(scenario):
    old = scenario.campaign
    with pytest.raises(ValueError, match="namespace"):
        scenario.mod.Campaign(old.release, old.root.with_name(old.root.name.replace("attempt001", "attempt000")),
            old.archive_sha256, old.identity["manifest_sha256"], old.plan_sha256)
