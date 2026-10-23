"""Fail-closed stage/provenance contracts; execute only on allocated DTU HPC."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest
import torch
from torch import nn

from latentmol.boltz import embedding_experiment as experiment
from latentmol.boltz.embedding_training import EPOCH36_SHA256
import latentmol.evaluate as evaluate


def test_loader_reads_training_mode_from_training_configuration(monkeypatch):
    """ModelConfig describes architecture and has never owned the AE/VAE mode."""
    model = nn.Linear(2, 2)
    checkpoint = {"epoch": 36, "model_config": {"hidden": 2}, "train_config": {"mode": "vae"}}
    asset = {"path": "authenticated-original36.pt", "sha256": EPOCH36_SHA256}
    authenticated = []
    monkeypatch.setattr(experiment, "verified", lambda value: authenticated.append(value) or Path(value["path"]))
    monkeypatch.setattr(evaluate, "load_model", lambda path, device: (model, checkpoint))
    loaded, returned = experiment.load_vae({"vae": asset}, "cpu")
    assert authenticated == [asset]
    assert returned is checkpoint and loaded is model
    assert not loaded.training and all(not parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.embedding_bridge_checkpoint_identity == {"epoch": 36, "sha256": EPOCH36_SHA256}


@pytest.mark.parametrize("epoch,mode", [(35, "vae"), (36, "ae")])
def test_loader_rejects_other_epoch_or_training_mode(monkeypatch, epoch, mode):
    model = nn.Linear(2, 2)
    checkpoint = {"epoch": epoch, "model_config": {"hidden": 2}, "train_config": {"mode": mode}}
    monkeypatch.setattr(experiment, "verified", lambda value: Path(value["path"]))
    monkeypatch.setattr(evaluate, "load_model", lambda path, device: (model, checkpoint))
    with pytest.raises(ValueError, match="epoch-36 VAE"):
        experiment.load_vae({"vae": {"path": "mock.pt", "sha256": EPOCH36_SHA256}}, "cpu")
    assert not hasattr(model, "embedding_bridge_checkpoint_identity")


@pytest.mark.parametrize("returncode,tests,failures,errors,skipped", [
    (1, 3, 1, 0, 0), (0, 3, 0, 0, 1), (0, 0, 0, 0, 0)])
def test_cpu_contract_failure_stops_before_checkpoint_loading(
        tmp_path, monkeypatch, returncode, tests, failures, errors, skipped):
    loaded = []
    monkeypatch.setattr(experiment, "load_vae", lambda *args: loaded.append(args))

    def fake_run(command, **kwargs):
        (tmp_path / "cpu-tests.xml").write_text(
            f'<testsuites><testsuite tests="{tests}" failures="{failures}" errors="{errors}" skipped="{skipped}"/></testsuites>')
        return SimpleNamespace(returncode=returncode, stdout="mock allocated pytest result\n")

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="did not all pass"):
        experiment.cpu_stage({"cpu_tests": ["mock_fixture.py"]}, tmp_path, time.monotonic() + 10)
    assert not loaded
    saved = json.loads((tmp_path / "cpu-gate.json").read_text())
    assert saved["junit"] == {"tests": tests, "failures": failures, "errors": errors, "skipped": skipped}
    assert saved["status"] == "failed_not_passed"


def test_entrypoint_requires_allocation_before_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.setattr(sys, "argv", ["embedding_experiment", "cpu", "--plan", "missing.json",
                                     "--output", str(tmp_path / "not-created")])
    with pytest.raises(RuntimeError, match="LSF allocation"):
        experiment.main()
    assert not (tmp_path / "not-created").exists()


def test_entrypoint_preserves_failed_stage_with_exact_source_and_plan_identity(tmp_path, monkeypatch):
    from latentmol.experiments import file_sha256
    source_sha = "c" * 64
    config = {"version": experiment.VERSION, "internal_seconds": {"cpu": 10},
              "vae": {"path": "original36.pt", "sha256": EPOCH36_SHA256}}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(config))
    output = tmp_path / "cpu"
    monkeypatch.setenv("LSB_JOBID", "mock-allocated-job")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(sys, "argv", ["embedding_experiment", "cpu", "--plan", str(plan), "--output", str(output)])
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)

    def fail_stage(config, output, deadline):
        raise RuntimeError("injected stage failure")

    monkeypatch.setattr(experiment, "cpu_stage", fail_stage)
    with pytest.raises(RuntimeError, match="injected stage failure"):
        experiment.main()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed_not_passed"
    assert result["source_sha256"] == source_sha
    assert result["plan_sha256"] == file_sha256(plan)
    assert result["job_id"] == "mock-allocated-job"
    assert result["existing_vae_continuation"] == "untouched"
    assert result["error"] == "injected stage failure"
    assert result["elapsed_seconds"] >= 0
