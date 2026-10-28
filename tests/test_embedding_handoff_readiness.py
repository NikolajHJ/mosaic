"""Allocated stdlib receipt/readiness fixtures; no real scheduler/model calls."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/hpc/embedding_handoff_readiness.py"
    spec = importlib.util.spec_from_file_location("handoff_readiness_fixture", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def save(mod, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return mod.asset(path)


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    mod = module()
    release, campaign = tmp_path / "releases/parent", tmp_path / "runs/parent"
    starts = campaign / "cpu/starts.pt"
    starts.parent.mkdir(parents=True)
    starts.write_bytes(b"authenticated original three z/mask tensors")
    original_config = dict(bridge=dict(path="old-pilot.pt", sha256="1" * 64),
        precision_correction=dict(parent_starts=mod.asset(starts)))
    save(mod, release / mod.PARENT_CONFIG, original_config)
    controller = release / mod.CONTROLLER
    controller.parent.mkdir(parents=True)
    controller.write_text("# authentic original controller fixture\n", encoding="utf-8")
    worker = release / "src/latentmol/boltz/embedding_long_experiment.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# original worker bytes\n", encoding="utf-8")
    members = {p.relative_to(release).as_posix(): mod.digest(p) for p in release.rglob("*") if p.is_file()}
    save(mod, release / "embedding-release-manifest.json", dict(members=members))
    archive = Path(str(release) + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        for p in release.rglob("*"):
            if p.is_file():
                stream.add(p, arcname=p.relative_to(release).as_posix(), recursive=False)
    identity = dict(source_sha256=mod.digest(archive),
        manifest_sha256=mod.digest(release / "embedding-release-manifest.json"),
        plan_sha256=mod.digest(release / mod.PARENT_CONFIG))
    junit = campaign / "cpu/cpu-tests.xml"
    junit.write_text('<testsuite failures="0" errors="0" skipped="0">' +
        ''.join('<testcase name="c%d"/>' % i for i in range(450)) + '</testsuite>', encoding="utf-8")
    frozen = dict(vae="2" * 64, bridge=original_config["bridge"]["sha256"], starts=mod.digest(starts))
    cpu = dict(**identity, status="passed", stage="cpu", job_id="29839742",
        existing_vae_continuation="untouched", frozen_identities=frozen, tests_passed=450,
        cpu_execution=dict(profile=mod.PROFILE, environment=dict(mod.ENVIRONMENT)),
        starts_readiness=dict(all_three_parent_starts_exact=True, frozen_vae_unchanged=True),
        artifacts=dict(junit=mod.asset(junit), starts=mod.asset(starts)))
    cpu_asset = save(mod, campaign / "cpu/result.json", cpu)
    cpuexec = dict(identity=identity, node="cpu", status="completed", returncode=0, error=None,
                   job_id=cpu["job_id"], result_sha256=cpu_asset["sha256"])
    cpuexec_asset = save(mod, tmp_path / "controls/cpu-execution.json", cpuexec)
    checkpoint = campaign / "fit/fit/best-bridge.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact previously trained original142 bridge tensor bytes")
    monkeypatch.setattr(mod, "CHECKPOINT_SHA", mod.digest(checkpoint))
    bindings = {k: identity[k] for k in ("source_sha256", "plan_sha256")}
    training = dict(bindings=bindings, warm_start=original_config["bridge"], desired_score_labels=False)
    train_asset = save(mod, campaign / "fit/fit/training-plan.json", training)
    data_asset = save(mod, campaign / "fit/paired-data/manifest.json",
        dict(long_data=dict(benchmark_starts=original_config["precision_correction"]["parent_starts"])))
    calibration = dict(version="epoch36-bridge-dev-q99-calibration-v1", bindings=bindings,
        bridge_sha256=mod.CHECKPOINT_SHA, split="dev", quantile=.99, quantile_method="nearest_rank",
        cap_floor=1e-8, audit_used_for_calibration=False, desired_score_labels=False,
        warm_start=original_config["bridge"], training_plan_sha256=train_asset["sha256"],
        dataset_manifest_sha256=data_asset["sha256"], vae_checkpoint=dict(epoch=36, sha256="2" * 64),
        vae_state_sha256="3" * 64, bank_state_sha256="4" * 64,
        bank_metadata=dict(projections=[dict(key="projection")]), metric_names=["projection/mse"],
        dev_errors={"projection/mse": [.1, .2, .3]}, caps={"projection/mse": .3}, dev_examples=3)
    cal_asset = save(mod, campaign / "fit/fit/calibration.json", calibration)
    monkeypatch.setattr(mod, "CALIBRATION_SHA", cal_asset["sha256"])
    inner = dict(status="completed_long_bridge_fit_not_native_validated", bindings=bindings,
        completed_steps=5000, requested_steps=5000, dev_improved=True,
        frozen_vae_and_bank_unchanged=True, warm_start_unchanged=True, audit_consumed=True,
        checkpoint_sha256=mod.CHECKPOINT_SHA, calibration_sha256=mod.CALIBRATION_SHA,
        best_dev_loss=.01, initial_dev_loss=.05, training_plan_sha256=train_asset["sha256"],
        dataset_manifest_sha256=data_asset["sha256"], vae_checkpoint=calibration["vae_checkpoint"],
        vae_state_sha256=calibration["vae_state_sha256"], bank_state_sha256=calibration["bank_state_sha256"])
    fit = dict(**identity, status="passed", stage="fit", job_id="29839755",
        existing_vae_continuation="untouched", frozen_identities=frozen,
        model_identity=dict(checkpoint_sha256=mod.CHECKPOINT_SHA, calibration_sha256=mod.CALIBRATION_SHA),
        fit_completed=True, calibration_passed=True, fit=inner,
        artifacts=dict(checkpoint=mod.asset(checkpoint), calibration=cal_asset,
                       training_plan=train_asset, paired_data_manifest=data_asset))
    fit_asset = save(mod, campaign / "fit/result.json", fit)
    fitexec = dict(identity=identity, node="fit", status="completed", returncode=0, error=None,
                   job_id=fit["job_id"], result_sha256=fit_asset["sha256"])
    fitexec_asset = save(mod, tmp_path / "controls/fit-execution.json", fitexec)
    terminal = tmp_path / "logs/fit.out"
    terminal.parent.mkdir()
    terminal.write_text(f'JOB={fit["job_id"]} HOST=fixture SOURCE={release}\n' + json.dumps(fitexec) +
        f'\nSubject: Job {fit["job_id"]}: <{campaign.name}-fit> in cluster <dcc> Done\nSuccessfully completed.\n',
        encoding="utf-8")
    config = dict(version=mod.CONFIG_VERSION, parent=dict(release=str(release), campaign=str(campaign),
        **identity, controller_sha256=mod.digest(controller), artifacts=dict(cpu_result=cpu_asset,
        cpu_execution=cpuexec_asset, fit_result=fit_asset, fit_execution=fitexec_asset,
        checkpoint=mod.asset(checkpoint), calibration=cal_asset, fit_terminal=mod.asset(terminal))),
        readiness=dict(cpu_tests=["tests/test_embedding_handoff_readiness.py"]))

    def rebound_cpu(value):
        config["parent"]["artifacts"]["cpu_result"] = save(mod, campaign / "cpu/result.json", value)
        cpuexec["result_sha256"] = config["parent"]["artifacts"]["cpu_result"]["sha256"]
        config["parent"]["artifacts"]["cpu_execution"] = save(mod, tmp_path / "controls/cpu-execution.json", cpuexec)

    return SimpleNamespace(mod=mod, config=config, release=release, campaign=campaign,
        cpu=cpu, fit=fit, calibration=calibration, checkpoint=checkpoint, rebound_cpu=rebound_cpu)


def test_adopts_original450_frozen_fit_and_dev_caps_without_model_loading(scenario):
    result = scenario.mod.validate_parent(scenario.config)
    assert result["status"] == "passed" and result["original_cpu_junit"]["tests"] == 450
    assert result["model_identity"]["checkpoint_sha256"] == scenario.mod.CHECKPOINT_SHA
    assert result["new_calibration"] is False and result["refit"] is False


@pytest.mark.parametrize("change", ["member", "extra_source"])
def test_complete_original_release_authentication_rejects_changed_or_extra_source(scenario, change):
    path = scenario.release / (scenario.mod.CONTROLLER if change == "member" else "extra.py")
    path.write_text("# changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="controller|Release"):
        scenario.mod.validate_parent(scenario.config)


@pytest.mark.parametrize("change", ["count", "profile", "starts", "status"])
def test_rebinding_cpu_receipts_does_not_bypass_original_readiness_contract(scenario, change):
    cpu = copy.deepcopy(scenario.cpu)
    if change == "count": cpu["tests_passed"] = 449
    elif change == "profile": cpu["cpu_execution"]["profile"] = "other"
    elif change == "starts": cpu["starts_readiness"]["all_three_parent_starts_exact"] = False
    else: cpu["status"] = "failed"
    scenario.rebound_cpu(cpu)
    with pytest.raises(ValueError):
        scenario.mod.validate_parent(scenario.config)


def test_changed_checkpoint_bytes_cannot_be_adopted(scenario):
    scenario.checkpoint.write_bytes(b"substituted model")
    with pytest.raises(ValueError, match="authentication"):
        scenario.mod.validate_parent(scenario.config)


def test_terminal_log_must_embed_the_exact_passed_execution_receipt(scenario):
    item = scenario.config["parent"]["artifacts"]["fit_terminal"]
    path = Path(item["path"])
    path.write_text(path.read_text().replace('"returncode": 0', '"returncode": 1'), encoding="utf-8")
    item["sha256"] = scenario.mod.digest(path)
    with pytest.raises(ValueError, match="exact passed allocated execution"):
        scenario.mod.validate_parent(scenario.config)


@pytest.mark.parametrize("xml", ['<testsuite/>', '<testsuite><testcase><skipped/></testcase></testsuite>',
    '<testsuite errors="1"><testcase/></testsuite>'])
def test_junit_cannot_promote_empty_skipped_or_failed_tests(tmp_path, xml):
    mod = module(); path = tmp_path / "junit.xml"; path.write_text(xml, encoding="utf-8")
    with pytest.raises(ValueError): mod.junit(path)


def test_probe_uses_actual_generated_argv_and_exact_saved_model_flags(scenario, monkeypatch):
    mod, config = scenario.mod, scenario.config
    command = [mod.sys.executable, "-m", "latentmol.boltz.embedding_long_experiment", "--stage", "pilot",
        "--source-sha256", config["parent"]["source_sha256"], "--plan-sha256", config["parent"]["plan_sha256"],
        "--bridge-sha256", mod.CHECKPOINT_SHA, "--calibration-sha256", mod.CALIBRATION_SHA]
    monkeypatch.setattr(mod, "load_module", lambda *args: SimpleNamespace(
        make_parent_campaign=lambda config: SimpleNamespace(worker_command=lambda node, intent: command)))

    def process(argv, **kwargs):
        spec = json.loads(argv[-1])
        assert spec["command"] == command and kwargs["cwd"] == scenario.release
        assert kwargs["env"]["LATENTMOL_SOURCE_SHA256"] == config["parent"]["source_sha256"]
        assert kwargs["env"]["PYTHONPATH"] == str(scenario.release / "src")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        result = dict(status="passed", argv=command, updates=0,
            actual_worker_sha256=spec["module_sha256"], reached="native_smoke", stop="before_cuda_or_model_setup",
            parent_writes=0, job_submissions=0, cuda_initializations=0, native_model_loads=0,
            cpu_module_imports=["numpy", "torch"], cuda_visible_devices="",
            cuda_initialized_before=False, cuda_initialized_after=False)
        return SimpleNamespace(returncode=0, stdout="HANDOFF_CLI_PROBE_JSON=" + json.dumps(result), stderr="")

    monkeypatch.setattr(mod.subprocess, "run", process)
    assert mod.pilot_cli_probe(config)["updates"] == 0
    command[command.index("--bridge-sha256") + 1] = "f" * 64
    with pytest.raises(ValueError, match="changed inherited"):
        mod.pilot_cli_probe(config)


def test_actual_original_eager_cpu_package_reaches_pilot_dispatch_without_cuda():
    """Use real package __init__/adapter/contracts and the original main parser.

    This regression intentionally needs the authenticated HPC parent assets.
    It never substitutes a package, parser, VAE, or native scorer. The readiness
    subprocess hides CUDA and stops before the first native/model boundary.
    """
    assert os.environ.get("LSB_JOBID"), "This regression requires the allocated HPC readiness job"
    mod = module()
    config = mod.read(Path(__file__).resolve().parents[1] / "configs/embedding_handoff_recovery.json")
    receipt = mod.pilot_cli_probe(config)
    assert "torch" in receipt["cpu_module_imports"]
    assert set(receipt["cpu_module_imports"]) <= {"torch", "numpy"}
    assert receipt["cuda_visible_devices"] == ""
    assert receipt["cuda_initialized_before"] is receipt["cuda_initialized_after"] is False
    assert receipt["cuda_initializations"] == receipt["native_model_loads"] == 0
    assert receipt["parent_writes"] == receipt["job_submissions"] == receipt["updates"] == 0


def test_readiness_code_has_no_numerical_imports():
    import ast
    path = Path(__file__).resolve().parents[1] / "scripts/hpc/embedding_handoff_readiness.py"
    tree = ast.parse(path.read_text())
    names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(n and n.split(".")[0] in {"torch", "numpy", "latentmol", "boltz", "rdkit"} for n in names)
