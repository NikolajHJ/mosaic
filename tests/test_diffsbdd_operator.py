"""Immutable allocated-stage contracts, executed only on allocated HPC.

These fixtures import no numerical packages and perform no real scheduling,
environment installation, model work or remote operations. Every subprocess
boundary is replaced; files belong only to pytest's private temporary tree.
"""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def operator_module():
    location = Path(__file__).resolve().parents[1] / "scripts/hpc/diffsbdd_operator.py"
    spec = importlib.util.spec_from_file_location("diffsbdd_operator_under_test", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sealed_fixture(operator, tmp_path, monkeypatch):
    work = tmp_path / "independent-work"
    name = "diffsbdd-vae-alternation-v1-attempt001"
    source = work / "releases" / name / "source"
    source.mkdir(parents=True)
    config = source / "configs/protocol.json"
    config.parent.mkdir()
    config.write_text('{"version":"diffsbdd-protocol-fixture-v1"}\n', encoding="utf-8")
    package = source / "src/latentmol/diffsbdd/experiment.py"
    package.parent.mkdir(parents=True)
    package.write_text('EXPERIMENT = "authenticated fixture; never executed"\n', encoding="utf-8")
    manifest = source / "source-manifest.json"
    manifest.write_text(json.dumps(dict(version=operator.SOURCE_VERSION,
        files={path.relative_to(source).as_posix(): operator.digest(path)
               for path in (config, package)})), encoding="utf-8")
    runtime = work / "env/diffsbdd-v1-attempt001/bin/python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("fixture interpreter; subprocess boundary is replaced", encoding="utf-8")
    monkeypatch.setattr(operator, "WORK_ROOT", work)
    monkeypatch.setattr(operator, "RUNTIME_PYTHON", runtime)
    monkeypatch.setenv("LSB_JOBID", "12345678")
    monkeypatch.setenv("LSB_QUEUE", "hpc")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "4")
    return dict(source_root=source, source_manifest=manifest,
                source_manifest_sha256=operator.digest(manifest), config=config,
                config_sha256=operator.digest(config), run_root=work / "runs" / name,
                python=runtime, internal_seconds=120)


def authenticate(operator, arguments):
    return operator.authenticate_source(**{key: arguments[key] for key in
        ("source_root", "source_manifest", "source_manifest_sha256", "config", "config_sha256")})


def stage_output(operator, root, stage, *, passed=True, checks=None):
    root.mkdir(parents=True)
    artifact = root / "evidence.json"
    artifact.write_text('{"independent-fixture-evidence":true}\n', encoding="utf-8")
    manifest = root / "output-manifest.json"
    manifest.write_text(json.dumps(dict(version=operator.OUTPUT_VERSION,
        files={"evidence.json": operator.digest(artifact)})), encoding="utf-8")
    result = dict(version=operator.STAGE_VERSION, stage=stage, passed=passed,
        checks={"fixture_evidence": True} if checks is None else checks,
        output_manifest=dict(path="output-manifest.json", sha256=operator.digest(manifest)))
    (root / "stage.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def successful_launch(operator, calls):
    def launch(command, source_root, environment, log_path, timeout):
        calls.append(dict(command=command, source_root=source_root, environment=environment,
                          log_path=log_path, timeout=timeout))
        Path(log_path).write_bytes(b"fixture stage process output\n")
        stage = command[command.index("--stage") + 1]
        output = Path(command[command.index("--output") + 1])
        stage_output(operator, output, stage)
        return 0

    return launch


def test_complete_source_config_and_manifest_bindings(tmp_path, monkeypatch):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    identity = authenticate(operator, arguments)
    assert identity["source_manifest_sha256"] == arguments["source_manifest_sha256"]
    assert identity["config_sha256"] == arguments["config_sha256"]
    assert identity["source_members"] == 2
    package = arguments["source_root"] / "src/latentmol/diffsbdd/experiment.py"
    package.write_text("CHANGED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source member SHA256"):
        authenticate(operator, arguments)


@pytest.mark.parametrize("kind", ("manifest", "config"))
def test_wrong_expected_source_or_config_stops_authentication(tmp_path, monkeypatch, kind):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    arguments["source_manifest_sha256" if kind == "manifest" else "config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest SHA256" if kind == "manifest" else "Config SHA256"):
        authenticate(operator, arguments)


@pytest.mark.parametrize("relative", ("src/unsealed.py", ".pytest_cache/hidden.py", "src/__pycache__/hidden.py"))
def test_any_unsealed_python_file_is_rejected_including_hidden_cache_sources(tmp_path, monkeypatch, relative):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    extra = arguments["source_root"] / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("UNSEALED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        authenticate(operator, arguments)


def test_bytecode_and_pytest_metadata_are_the_only_ignored_runtime_caches(tmp_path, monkeypatch):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    for relative in ("src/__pycache__/fixture.cpython-test.pyc", ".pytest_cache/v/cache/nodeids"):
        path = arguments["source_root"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime cache")
    authenticate(operator, arguments)


@pytest.mark.parametrize("relative", ("../outside.py", "src/../outside.py", "/absolute.py", "src\\windows.py"))
def test_manifest_paths_cannot_escape_or_alias_source_root(tmp_path, monkeypatch, relative):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    manifest = json.loads(arguments["source_manifest"].read_text())
    manifest["files"][relative] = "0" * 64
    arguments["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    arguments["source_manifest_sha256"] = operator.digest(arguments["source_manifest"])
    with pytest.raises(ValueError, match="Unsafe manifest member"):
        authenticate(operator, arguments)


def test_exclusive_receipt_write_preserves_first_evidence(tmp_path):
    operator = operator_module()
    receipt = tmp_path / "receipt.json"
    operator.write_new(receipt, {"attempt": "original"})
    before = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        operator.write_new(receipt, {"attempt": "replacement"})
    assert receipt.read_bytes() == before


@pytest.mark.parametrize("stage,queue,accepted", (
    ("cpu", "hpc", True), ("native", "gpua100", True), ("fit", "gpua10", True),
    ("cpu", "gpua100", False), ("scorer", "hpc", False), ("native", "", False)))
def test_queue_class_matches_each_stage(stage, queue, accepted):
    operator = operator_module()
    env = {"LSB_JOBID": "12345678", "LSB_QUEUE": queue}
    if accepted:
        assert operator.allocation(stage, env)["LSB_JOBID"] == "12345678"
    else:
        with pytest.raises(RuntimeError, match="allocation|queue class"):
            operator.allocation(stage, env)
    with pytest.raises(RuntimeError, match="allocation"):
        operator.allocation(stage, {"LSB_QUEUE": queue})


@pytest.mark.parametrize("passed,checks", ((1, {"a": True}), ("true", {"a": True}),
    (False, {"a": True}), (True, {"a": 1}), (True, {}), (True, {"a": False})))
def test_pass_requires_exact_booleans_and_all_checks(tmp_path, passed, checks):
    operator = operator_module()
    stage_output(operator, tmp_path / "cpu", "cpu", passed=passed, checks=checks)
    with pytest.raises(ValueError, match="all-checks pass"):
        operator.validate_stage_output(tmp_path / "cpu", "cpu")


@pytest.mark.parametrize("change", ("artifact", "extra", "manifest"))
def test_passed_output_requires_unchanged_complete_inventory(tmp_path, change):
    operator = operator_module()
    root = tmp_path / "cpu"
    stage_output(operator, root, "cpu")
    operator.validate_stage_output(root, "cpu")
    if change == "artifact":
        (root / "evidence.json").write_text("{}\n", encoding="utf-8")
    elif change == "extra":
        (root / "unreported.json").write_text("{}\n", encoding="utf-8")
    else:
        (root / "output-manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch|file set differs"):
        operator.validate_stage_output(root, "cpu")


def test_output_symlinks_cannot_point_to_shared_or_external_artifacts(tmp_path):
    operator = operator_module()
    root = tmp_path / "cpu"
    stage_output(operator, root, "cpu")
    (root / "evidence.json").rename(root / "actual-evidence.json")
    (root / "evidence.json").symlink_to(root / "actual-evidence.json")
    with pytest.raises(ValueError, match="symlinked artifacts"):
        operator.validate_stage_output(root, "cpu")


def test_run_launches_one_process_binds_results_and_refuses_in_place_retry(tmp_path, monkeypatch):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(operator, "launch", successful_launch(operator, calls))
    result = operator.run_stage(stage="cpu", **arguments)
    assert result["passed"] is True and result["numerical_processes_launched"] == 1
    assert len(calls) == 1 and 0 < calls[0]["timeout"] <= arguments["internal_seconds"]
    assert calls[0]["command"][1:3] == ["-m", "latentmol.diffsbdd.experiment"]
    assert calls[0]["environment"]["PYTHONPATH"] == str(arguments["source_root"] / "src")
    assert calls[0]["environment"]["DIFFSBDD_CONFIG_SHA256"] == arguments["config_sha256"]
    receipt = arguments["run_root"] / "operator/cpu/receipt.json"
    before = receipt.read_bytes()
    with pytest.raises(FileExistsError, match="already attempted"):
        operator.run_stage(stage="cpu", **arguments)
    assert len(calls) == 1 and receipt.read_bytes() == before


def test_missing_allocation_writes_failed_receipt_without_starting_process(tmp_path, monkeypatch):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    monkeypatch.delenv("LSB_JOBID")
    calls = []
    monkeypatch.setattr(operator, "launch", successful_launch(operator, calls))
    with pytest.raises(RuntimeError, match="allocation"):
        operator.run_stage(stage="cpu", **arguments)
    receipt = json.loads((arguments["run_root"] / "operator/cpu/receipt.json").read_text())
    assert not calls and receipt["passed"] is False
    assert receipt["numerical_processes_launched"] == 0


@pytest.mark.parametrize("failure", ("nonzero", "timeout", "failed_gate"))
def test_process_or_gate_failure_keeps_outputs_and_failed_receipt(tmp_path, monkeypatch, failure):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)

    def launch(command, source_root, environment, log_path, timeout):
        Path(log_path).write_bytes(b"preserved failure output\n")
        output = Path(command[command.index("--output") + 1])
        if failure == "timeout":
            output.mkdir()
            (output / "partial.json").write_text("{}\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(command, timeout)
        stage_output(operator, output, "cpu", passed=failure != "failed_gate")
        return 9 if failure == "nonzero" else 0

    monkeypatch.setattr(operator, "launch", launch)
    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        operator.run_stage(stage="cpu", **arguments)
    control = arguments["run_root"] / "operator/cpu"
    receipt = json.loads((control / "receipt.json").read_text())
    assert receipt["passed"] is False and receipt["numerical_processes_launched"] == 1
    assert receipt["status"] == ("timed_out_not_passed" if failure == "timeout" else "failed_not_passed")
    assert (control / "stage.log").read_bytes() == b"preserved failure output\n"
    assert (arguments["run_root"] / "cpu").is_dir()


def test_complete_predecessor_chain_is_bound_and_rechecked(tmp_path, monkeypatch):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(operator, "launch", successful_launch(operator, calls))
    operator.run_stage(stage="cpu", **arguments)
    monkeypatch.setenv("LSB_QUEUE", "gpua100")
    native = operator.run_stage(stage="native", **arguments)
    assert native["predecessors"][0]["stage"] == "cpu"
    assert native["predecessors"][0]["operator_receipt_sha256"] == operator.digest(
        arguments["run_root"] / "operator/cpu/receipt.json")
    (arguments["run_root"] / "cpu/evidence.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        operator.run_stage(stage="scorer", **arguments)
    assert len(calls) == 2
    receipt = json.loads((arguments["run_root"] / "operator/scorer/receipt.json").read_text())
    assert receipt["passed"] is False and receipt["numerical_processes_launched"] == 0


def test_skipping_a_required_predecessor_never_launches(tmp_path, monkeypatch):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    monkeypatch.setenv("LSB_QUEUE", "gpua100")
    calls = []
    monkeypatch.setattr(operator, "launch", successful_launch(operator, calls))
    with pytest.raises(FileNotFoundError):
        operator.run_stage(stage="fit", **arguments)
    assert not calls


@pytest.mark.parametrize("changed", ("old_campaign", "old_release", "old_runtime"))
def test_existing_campaigns_releases_and_runtimes_are_never_admitted(tmp_path, monkeypatch, changed):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(operator, "launch", successful_launch(operator, calls))
    if changed == "old_campaign":
        arguments["run_root"] = arguments["run_root"].with_name("source138-existing-vae-continuation")
    elif changed == "old_release":
        other = arguments["source_root"].parent.with_name("unrelated-old-release") / "source"
        other.mkdir(parents=True)
        arguments["source_root"] = other
    else:
        other = arguments["python"].parents[2] / "boltz311/bin/python"
        other.parent.mkdir(parents=True)
        other.write_text("old runtime", encoding="utf-8")
        arguments["python"] = other
    with pytest.raises(ValueError, match="namespace|independent experiment runtime"):
        operator.run_stage(stage="cpu", **arguments)
    assert not calls
    if changed == "old_campaign":
        assert not arguments["run_root"].exists()


@pytest.mark.parametrize("seconds", (0, -1, float("nan"), float("inf"), 2401))
def test_internal_time_caps_fail_before_any_attempt_mutation(tmp_path, monkeypatch, seconds):
    operator = operator_module()
    arguments = sealed_fixture(operator, tmp_path, monkeypatch)
    arguments["internal_seconds"] = seconds
    with pytest.raises(ValueError, match="wall time"):
        operator.run_stage(stage="cpu", **arguments)
    assert not arguments["run_root"].exists()


def test_json_duplicate_fields_do_not_silently_override_pass_decisions(tmp_path):
    operator = operator_module()
    path = tmp_path / "ambiguous.json"
    path.write_text('{"passed":false,"passed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        operator.read_json(path)


def test_timeout_terminates_the_only_child_process_group(tmp_path, monkeypatch):
    operator = operator_module()
    calls = []

    class Process:
        pid = 123456789
        waits = 0

        def poll(self):
            return None

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(["fixture"], timeout)
            return -15

        def terminate(self):
            calls.append("terminate")

    process = Process()
    monkeypatch.setattr(operator.subprocess, "Popen", lambda *args, **kwargs: process)
    if operator.os.name == "posix":
        monkeypatch.setattr(operator.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        operator.launch(["never-executed"], tmp_path, {}, tmp_path / "stage.log", 1)
    assert len(calls) == 1 and process.waits == 2
