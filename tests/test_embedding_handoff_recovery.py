"""Allocated control-only recovery fixtures; no live scheduler/model work."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tarfile

import pytest


def module():
    location = Path(__file__).resolve().parents[1] / "scripts/hpc/embedding_handoff_recovery.py"
    spec = importlib.util.spec_from_file_location("_handoff_recovery_fixture", location)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def response(record=None, **values):
    value = dict(returncode=0, stderr="", stdout=json.dumps(dict(COMMAND="bjobs", JOBS=1,
        RECORDS=[record or dict(JOBID="29839755", STAT="DONE", JOB_NAME="exact-fit-label")])))
    value.update(values)
    return value


def terminal(job="29839755", label="exact-fit-label", release=Path("/work3/fixture-release")):
    return (f"JOB={job} HOST=n-62-18-12 SOURCE={release}\n"
        f"Subject: Job {job}: <{label}> in cluster <dcc> Done\n"
        "Started at Wed Oct 28 16:23:47 2026\n"
        "Terminated at Wed Oct 28 16:39:52 2026\n"
        "Results reported at Wed Oct 28 16:39:52 2026\nSuccessfully completed.\n")


def test_actual_dtu_zero_returncode_aged_out_json_contract():
    actual = response(dict(JOBID="29839755", ERROR="Job <29839755> is not found"))
    assert module().parse_scheduler(actual, "29839755", "exact-fit-label") == "aged_out"


def test_structured_live_done_requires_exact_full_job_and_name():
    assert module().parse_scheduler(response(), "29839755", "exact-fit-label") == "DONE"


@pytest.mark.parametrize("status", ["RUN", "PEND", "EXIT", "UNKWN", "POST_DONE", ""])
def test_non_done_live_states_do_not_pass(status):
    value = response(dict(JOBID="29839755", STAT=status, JOB_NAME="exact-fit-label"))
    with pytest.raises(ValueError, match="DONE"):
        module().parse_scheduler(value, "29839755", "exact-fit-label")


@pytest.mark.parametrize("kind", ["wrong_job", "wrong_label", "truncated_label", "unknown_error", "wrong_error_job",
    "empty_records", "two_records", "wrong_command", "wrong_count", "empty_stdout", "network", "return255", "mixed_error_state"])
def test_unknown_or_mismatched_scheduler_output_is_never_missing(kind):
    record = dict(JOBID="29839755", STAT="DONE", JOB_NAME="exact-fit-label")
    value = response(record)
    parsed = json.loads(value["stdout"])
    if kind == "wrong_job": parsed["RECORDS"][0]["JOBID"] = "99999999"
    elif kind == "wrong_label": parsed["RECORDS"][0]["JOB_NAME"] = "another-job"
    elif kind == "truncated_label": parsed["RECORDS"][0]["JOB_NAME"] = "exact-fit*"
    elif kind == "unknown_error": parsed["RECORDS"] = [dict(JOBID="29839755", ERROR="Scheduler unavailable")]
    elif kind == "wrong_error_job": parsed["RECORDS"] = [dict(JOBID="29839755", ERROR="Job <9> is not found")]
    elif kind == "empty_records": parsed["RECORDS"] = []
    elif kind == "two_records": parsed["RECORDS"] *= 2
    elif kind == "wrong_command": parsed["COMMAND"] = "bhist"
    elif kind == "wrong_count": parsed["JOBS"] = 0
    elif kind == "empty_stdout": value["stdout"] = ""
    elif kind == "network": value["stderr"] = "Connection reset"
    elif kind == "return255": value["returncode"] = 255
    else: parsed["RECORDS"][0]["ERROR"] = "Job <29839755> is not found"
    if kind != "empty_stdout": value["stdout"] = json.dumps(parsed)
    with pytest.raises(ValueError):
        module().parse_scheduler(value, "29839755", "exact-fit-label")


def test_legacy_empty_formatted_response_is_not_reinterpreted_as_json_success():
    value = dict(returncode=0, stdout="", stderr="Job <29839755> is not found\n")
    with pytest.raises(ValueError):
        module().parse_scheduler(value, "29839755", "exact-fit-label")


@pytest.mark.parametrize("mutation", ["wrong_job", "wrong_name", "wrong_source", "exit_subject", "missing_success", "no_finish", "duplicate_subject"])
def test_outer_lsf_success_proof_is_independent_and_exact(mutation):
    text = terminal()
    if mutation == "wrong_job": text = text.replace("29839755", "29839756")
    elif mutation == "wrong_name": text = text.replace("exact-fit-label", "another-label")
    elif mutation == "wrong_source": text = text.replace("/work3/fixture-release", "/other/release")
    elif mutation == "exit_subject": text = text.replace(" Done\n", " Exited\n")
    elif mutation == "missing_success": text = text.replace("Successfully completed.", "Exited with exit code 1.")
    elif mutation == "no_finish": text = text.replace("Terminated at ", "Not yet terminated ")
    else: text += "Subject: Job 29839755: <exact-fit-label> in cluster <dcc> Done\n"
    with pytest.raises(ValueError):
        module().terminal_passed(text, "29839755", "exact-fit-label", Path("/work3/fixture-release"))


def test_exact_successful_outer_lsf_summary_passes():
    module().terminal_passed(terminal(), "29839755", "exact-fit-label", Path("/work3/fixture-release"))


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    mod = module()
    repository = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    monkeypatch.setattr(mod, "ROOT", work)
    parent_name = "source142-epoch36-guarded-embedding-long-attempt001"
    release = work / "releases" / parent_name.replace("source142", "source-142", 1)
    campaign = work / "runs" / parent_name
    control = work / "submissions" / parent_name

    def pack(release, members):
        manifest = release / "embedding-release-manifest.json"
        save(manifest, {"version": 1, "members": {name: mod.digest(release / name) for name in members}})
        archive = Path(str(release) + ".tar.gz")
        with tarfile.open(archive, "w:gz") as stream:
            for name in [*members, "embedding-release-manifest.json"]:
                stream.add(release / name, arcname=name, recursive=False)
        return dict(source_sha256=mod.digest(archive), manifest_sha256=mod.digest(manifest))

    settings = dict(version="guarded-embedding-long-v1", case_ids=["case-001", "case-002", "case-003"],
        updates_per_case=10000, observe_every=10, segment_internal_seconds=82800, segment_wall="24:00",
        pilot_safety_factor=1.5, segment_reserve_seconds=1800, max_segments_per_case=1000)
    save(release / mod.PARENT_CONFIG, {"long_campaign": settings})
    worker_relative = "src/latentmol/boltz/embedding_long_experiment.py"
    for name in (mod.PARENT_SCRIPT, "scripts/hpc/embedding_operator.py", worker_relative):
        target = release / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, target)
    identity = pack(release, [mod.PARENT_CONFIG, mod.PARENT_SCRIPT, "scripts/hpc/embedding_operator.py", worker_relative])
    identity["plan_sha256"] = mod.digest(release / mod.PARENT_CONFIG)
    frozen = {"vae": "1" * 64, "context": "2" * 64}

    def asset(path, content=b"fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return dict(path=str(path), sha256=mod.digest(path))

    junit = asset(campaign / "cpu/cpu-tests.xml", b'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="cpu"/></testsuite></testsuites>')
    checkpoint = asset(campaign / "fit/fit/bridge.pt")
    calibration = asset(campaign / "fit/fit/calibration.json", b'{}')
    cpu = dict(identity, stage="cpu", status="passed", frozen_identities=frozen, tests_passed=1, artifacts={"junit": junit})
    save(campaign / "cpu/result.json", cpu)
    fit = dict(identity, stage="fit", status="passed", frozen_identities=frozen,
        fit_completed=True, calibration_passed=True, artifacts=dict(checkpoint=checkpoint, calibration=calibration),
        model_identity=dict(checkpoint_sha256=checkpoint["sha256"], calibration_sha256=calibration["sha256"]))
    save(campaign / "fit/result.json", fit)
    for node, job in (("cpu", "29839742"), ("fit", "29839755")):
        intent = dict(identity=identity, node=node, release=str(release), campaign=str(campaign),
            label=parent_name + "-" + node, request={})
        save(control / (node + "-intent.json"), intent)
        save(control / (node + "-receipt.json"), dict(status="submitted_not_passed", job_id=job,
            intent_sha256=mod.digest(control / (node + "-intent.json"))))
        save(control / (node + "-execution.json"), dict(identity=identity, node=node, job_id=job,
            status="completed", returncode=0, result_sha256=mod.digest(campaign / node / "result.json")))
    log = work / "logs" / (parent_name + "-fit.29839755.out")
    terminal_asset = asset(log, terminal(label=parent_name + "-fit", release=release).encode())
    assets = {name: dict(path=str(path), sha256=mod.digest(path)) for name, path in {
        "cpu_result": campaign / "cpu/result.json", "cpu_execution": control / "cpu-execution.json",
        "fit_result": campaign / "fit/result.json", "fit_execution": control / "fit-execution.json"}.items()}
    assets.update(checkpoint=checkpoint, calibration=calibration, fit_terminal=terminal_asset)
    config = dict(version=mod.VERSION, scheduler_policy=mod.POLICY,
        parent=dict(identity, release=str(release), campaign=str(campaign), fit_job_id="29839755",
            controller_sha256=mod.digest(release / mod.PARENT_SCRIPT), artifacts=assets),
        readiness=dict(cpu_tests=["tests/test_embedding_handoff_recovery.py"]))
    repair_name = "source143-embedding-handoff-recovery-attempt001"
    repair = work / "releases" / repair_name.replace("source143", "source-143", 1)
    save(repair / mod.CONFIG, config)
    (repair / mod.SCRIPT).parent.mkdir(parents=True)
    shutil.copyfile(repository / mod.SCRIPT, repair / mod.SCRIPT)
    repair_identity = pack(repair, [mod.CONFIG, mod.SCRIPT])
    repair_identity["plan_sha256"] = mod.digest(repair / mod.CONFIG)
    proof_dir = work / "runs" / repair_name / "cpu"
    proof_junit = asset(proof_dir / "cpu-tests.xml", b'<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="repair"/></testsuite>')
    proof = dict(repair_identity, version=mod.READINESS_VERSION, status="passed", original_identity=identity,
        original_controller_sha256=config["parent"]["controller_sha256"], job_id="143001",
        junit=dict(proof_junit, tests=1, failures=0, errors=0, skipped=0))
    cli = dict(status="passed", argv=mod.make_parent_campaign(config).worker_command("pilot", {}),
        actual_worker_sha256=mod.digest(release / worker_relative), reached="native_smoke",
        stop="before_cuda_or_model_setup", parent_writes=0, job_submissions=0,
        cuda_initializations=0, native_model_loads=0, updates=0, cpu_module_imports=["numpy", "torch"],
        cuda_visible_devices="", cuda_initialized_before=False, cuda_initialized_after=False)
    save(proof_dir / "pilot-cli.json", cli)
    proof.update(actual_pilot_cli=cli, actual_pilot_cli_asset=dict(path=str(proof_dir / "pilot-cli.json"),
        sha256=mod.digest(proof_dir / "pilot-cli.json")))
    save(proof_dir / "readiness.json", proof)
    args = SimpleNamespace(release=release, campaign=campaign, archive_sha256=identity["source_sha256"],
        manifest_sha256=identity["manifest_sha256"], plan_sha256=identity["plan_sha256"], repair_release=repair,
        repair_archive_sha256=repair_identity["source_sha256"], repair_manifest_sha256=repair_identity["manifest_sha256"],
        repair_plan_sha256=repair_identity["plan_sha256"], readiness=proof_dir / "readiness.json",
        readiness_sha256=mod.digest(proof_dir / "readiness.json"))
    monkeypatch.setattr(mod, "__file__", str(repair / mod.SCRIPT))
    return SimpleNamespace(mod=mod, config=config, args=args, campaign=campaign, control=control,
        release=release, repair=repair, work=work, identity=identity, proof=proof, log=log)


def test_readonly_builder_generates_original_pilot_cli_and_preserves_parent_controls(sealed):
    value = sealed.mod.make_parent_campaign(sealed.config)
    command = value.worker_command("pilot", {})
    assert command[command.index("--source-sha256") + 1] == sealed.identity["source_sha256"]
    assert command[command.index("--config") + 1] == str(sealed.release / sealed.mod.PARENT_CONFIG)
    assert command[command.index("--bridge-sha256") + 1] == sealed.config["parent"]["artifacts"]["checkpoint"]["sha256"]
    assert value.control == sealed.control


@pytest.mark.parametrize("operation", ["dispatch", "advance", "run_stage", "arguments", "followup_command"])
def test_parent_only_builder_cannot_mutate_or_write_final_state_manifest(sealed, operation):
    value = sealed.mod.make_parent_campaign(sealed.config)
    with pytest.raises(RuntimeError):
        if operation == "dispatch": value.dispatch("pilot")
        elif operation == "advance": value.advance("fit")
        elif operation == "run_stage": value.run_stage("pilot")
        elif operation == "arguments": value.arguments("advance", after="fit")
        else: value.worker_command("followup", {})
    assert not (sealed.control / "final-states.json").exists()


def test_ready_repair_routes_all_future_invocations_to_absolute_new_helper(sealed):
    value = sealed.mod.build_campaign(sealed.args)
    for action, values in (("run-stage", {"node": "pilot"}), ("advance", {"after": "pilot"})):
        command = value.arguments(action, **values)
        assert command[1] == str(sealed.repair / sealed.mod.SCRIPT)
        assert command[command.index("--release") + 1] == str(sealed.release)
        assert command[command.index("--archive-sha256") + 1] == sealed.identity["source_sha256"]
        assert command[command.index("--readiness-sha256") + 1] == sealed.args.readiness_sha256
    assert value.control == sealed.control


@pytest.mark.parametrize("stage", ["cpu", "fit"])
def test_repair_cannot_refit_or_rerun_original_cpu(sealed, stage):
    value = sealed.mod.build_campaign(sealed.args)
    with pytest.raises(ValueError, match="forbids"):
        value.dispatch(stage)
    with pytest.raises(ValueError, match="forbids"):
        value.run_stage(stage)


def test_exact_observed_aged_out_fit_requires_both_terminal_and_execution_proofs(sealed, monkeypatch):
    value = sealed.mod.build_campaign(sealed.args)
    monkeypatch.setattr(sealed.mod.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(**response(dict(JOBID="29839755", ERROR="Job <29839755> is not found"))))
    value.require_done("fit")
    evidence = list(value.repair.control.glob("scheduler/fit-*.json"))
    assert len(evidence) == 2
    assert any(read.get("stdout") for read in [json.loads(path.read_text()) for path in evidence])
    assert not (sealed.control / "advance-after-fit.json").exists()


def test_scheduler_error_is_preserved_before_failed_handoff(sealed, monkeypatch):
    value = sealed.mod.build_campaign(sealed.args)
    monkeypatch.setattr(sealed.mod.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(**response(stderr="unexpected unavailable scheduler")))
    with pytest.raises(ValueError):
        value.require_done("fit")
    evidence = list(value.repair.control.glob("scheduler/*.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text())["stderr"] == "unexpected unavailable scheduler"


@pytest.mark.parametrize("asset_name", ["fit_result", "fit_execution", "checkpoint", "calibration", "fit_terminal"])
def test_changed_saved_parent_or_terminal_bytes_reject_recovery(sealed, asset_name):
    path = Path(sealed.config["parent"]["artifacts"][asset_name]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        sealed.mod.build_campaign(sealed.args)


@pytest.mark.parametrize("field", ["status", "source_sha256", "manifest_sha256", "plan_sha256", "original_controller_sha256", "job_id"])
def test_readiness_proof_cannot_be_rebound_to_unpassed_or_other_release(sealed, field):
    proof = copy.deepcopy(sealed.proof)
    proof[field] = "failed" if field == "status" else "not-the-pinned-value"
    save(sealed.args.readiness, proof)
    sealed.args.readiness_sha256 = sealed.mod.digest(sealed.args.readiness)
    with pytest.raises(ValueError, match="readiness"):
        sealed.mod.build_campaign(sealed.args)


@pytest.mark.parametrize("kind", ["missing", "argv", "worker", "updates", "inline", "cuda_initializations",
    "native_model_loads", "cuda_visible_devices", "cuda_initialized_before", "cuda_initialized_after",
    "missing_torch", "forbidden_module", "duplicate_module"])
def test_actual_pilot_dispatch_proof_is_mandatory_exact_and_hash_bound(sealed, kind):
    proof = copy.deepcopy(sealed.proof)
    cli = copy.deepcopy(proof["actual_pilot_cli"])
    if kind == "missing":
        del proof["actual_pilot_cli_asset"]
    elif kind == "inline":
        proof["actual_pilot_cli"]["stop"] = "after_model_setup"
    else:
        if kind == "argv": cli["argv"][cli["argv"].index("--source-sha256") + 1] = "f" * 64
        elif kind == "worker": cli["actual_worker_sha256"] = "f" * 64
        elif kind in ("cuda_initializations", "native_model_loads", "updates"): cli[kind] = 1
        elif kind == "cuda_visible_devices": cli[kind] = "0"
        elif kind in ("cuda_initialized_before", "cuda_initialized_after"): cli[kind] = True
        elif kind == "missing_torch": cli["cpu_module_imports"] = ["numpy"]
        elif kind == "forbidden_module": cli["cpu_module_imports"] = ["boltz", "numpy", "torch"]
        else: cli["cpu_module_imports"] = ["numpy", "torch", "torch"]
        cli_path = Path(proof["actual_pilot_cli_asset"]["path"])
        save(cli_path, cli)
        proof["actual_pilot_cli"] = cli
        proof["actual_pilot_cli_asset"]["sha256"] = sealed.mod.digest(cli_path)
    save(sealed.args.readiness, proof)
    sealed.args.readiness_sha256 = sealed.mod.digest(sealed.args.readiness)
    with pytest.raises(ValueError):
        sealed.mod.build_campaign(sealed.args)


def test_legacy_no_import_readiness_proof_is_not_promoted(sealed):
    proof = copy.deepcopy(sealed.proof)
    proof["version"] = "embedding-handoff-readiness-v1"
    save(sealed.args.readiness, proof)
    sealed.args.readiness_sha256 = sealed.mod.digest(sealed.args.readiness)
    with pytest.raises(ValueError, match="readiness"):
        sealed.mod.build_campaign(sealed.args)


def test_original_helper_dependency_is_not_taken_from_repair_import_path(sealed, monkeypatch):
    import sys
    fake = SimpleNamespace(authenticate=lambda *args: pytest.fail("untrusted repair helper imported"))
    monkeypatch.setitem(sys.modules, "embedding_operator", fake)
    value = sealed.mod.make_parent_campaign(sealed.config)
    assert value.worker_command("pilot", {})
    assert sys.modules["embedding_operator"] is fake


def test_dispatch_uses_original_scientific_environment_and_new_callback_helper(sealed, monkeypatch):
    value = sealed.mod.build_campaign(sealed.args)
    calls = []
    def run(command, **kwargs):
        if command == ["getquota_work3.sh"]:
            return SimpleNamespace(returncode=0, stdout="s234818|306121|| 40.00 GiB| 100.00 GiB|| 1| 20\n", stderr="")
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="Job <" + str(9000 + len(calls)) + "> submitted", stderr="")
    monkeypatch.setattr(sealed.mod.subprocess, "run", run)
    value.dispatch("pilot")
    assert len(calls) == 2
    for _, call in calls:
        assert "LATENTMOL_SOURCE_ARCHIVE=" + str(sealed.release) + ".tar.gz" in call["input"]
        assert str(sealed.repair / sealed.mod.SCRIPT) in call["input"]
        assert "cd " + str(sealed.release) in call["input"]
    assert "ended(9001)" in calls[1][0]
    assert not (value.repair.control / "pilot-intent.json").exists()
    assert (sealed.control / "pilot-intent.json").exists()


def test_ambiguous_existing_original_pilot_intent_blocks_recovery(sealed, monkeypatch):
    value = sealed.mod.build_campaign(sealed.args)
    save(sealed.control / "pilot-intent.json", dict(status="ambiguous_existing_attempt"))
    monkeypatch.setattr(sealed.mod.subprocess, "run", lambda *args, **kwargs: pytest.fail("no duplicate scheduler operation"))
    with pytest.raises(FileExistsError):
        value.dispatch("pilot")


def test_no_initial_submission_action_is_exposed():
    with pytest.raises(SystemExit):
        module().main(["submit-initial"])
