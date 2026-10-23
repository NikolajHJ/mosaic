"""HPC-only orchestration regressions; no full external model/data required."""

from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import random
import shlex
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import latentmol.large_vae as study
import latentmol.train as training
from latentmol.large_vae_evaluation import evaluate_panel
from latentmol.losses import LossConfig
from latentmol.models import GraphAutoencoder, ModelConfig
from test_training import TinyDataset, assert_nested_equal


class RNGConsumingObserver:
    compact_indices = True

    def attach(self, *args):
        pass

    def before_epoch(self, *args):
        pass

    def before_batch(self, *args):
        pass

    def after_backward(self, *args):
        pass

    def after_epoch(self, *args):
        random.random()
        np.random.random()
        torch.rand(10)
        # Even a non-shuffled DataLoader consumes a base seed on iteration.
        list(torch.utils.data.DataLoader([1, 2], batch_size=1))
        return {"canary": "RNG-neutralized by training boundary"}


def test_observer_rng_and_compact_cohorts_preserve_exact_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "GraphDataset", lambda path, split: TinyDataset(split))
    config = training.TrainConfig(mode="vae", epochs=2, batch_size=2, validation_limit=4,
                                   chemistry_limit=2, train_probe_limit=4, device="cpu", checkpoint_every=1,
                                   model=ModelConfig(hidden=8, latent=4, layers=1))
    whole = training.train(config, tmp_path, tmp_path / "whole", observer=RNGConsumingObserver())
    first = training.train(config, tmp_path, tmp_path / "split", stop_after_epoch=1, observer=RNGConsumingObserver())
    resumed = training.train(config, tmp_path, tmp_path / "split", resume_from=first / "last.pt", observer=RNGConsumingObserver())
    a = torch.load(whole / "last.pt", weights_only=True)
    b = torch.load(resumed / "last.pt", weights_only=True)
    for key in ("model_state", "optimizer_state"):
        assert_nested_equal(a[key], b[key])
    assert_nested_equal(a["continuation"]["rng"], b["continuation"]["rng"])
    assert a["continuation"]["indices"]["train"]["count"] == 8


def test_wall_guard_fails_before_starting_an_epoch_that_cannot_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "GraphDataset", lambda path, split: TinyDataset(split))
    config = training.TrainConfig(mode="vae", epochs=2, batch_size=2, device="cpu",
                                   model=ModelConfig(hidden=8, latent=4, layers=1))
    with pytest.raises(RuntimeError, match="cannot safely fit"):
        training.train(config, tmp_path, tmp_path / "runs", wall_budget_seconds=.001, initial_epoch_seconds=10)
    records = list((tmp_path / "runs").glob("*/run.json"))
    assert len(records) == 1 and json.loads(records[0].read_text())["status"] == "failed"
    assert not list((tmp_path / "runs").glob("*/last.pt"))


def test_segment_persists_progress_using_bound_selected_config_identity(tmp_path, monkeypatch):
    config = training.TrainConfig()
    # Authentication/training are mocked, but the real progress writer hashes
    # the protocol and checkpoint. Supply actual files, just as the allocated
    # dispatcher does; passing tmp_path here would hash a directory.
    config_path = tmp_path / "base.yaml"
    selected_path = tmp_path / "selected-training.yaml"
    protocol_path = tmp_path / "protocol.json"
    gate_path = tmp_path / "passed.json"
    config_path.write_text("mode: vae\nbatch_size: 256\n", encoding="utf-8")
    selected_path.write_text("mode: vae\nbatch_size: 512\n", encoding="utf-8")
    protocol_path.write_text(json.dumps({"segment_wall_budget_seconds": 100}), encoding="utf-8")
    gate_path.write_text("{}\n", encoding="utf-8")
    identity = {"selected_config_sha256": study.file_sha256(selected_path),
                "protocol_sha256": study.file_sha256(protocol_path)}
    monkeypatch.setattr(study, "read_protocol", lambda p: json.loads(p.read_text()))
    def fake_authenticate(base, protocol, data, gate):
        assert base == config_path and protocol == protocol_path and gate == gate_path
        return {"conservative_epoch_seconds": 1}, object(), config, identity
    monkeypatch.setattr(study, "authenticate_gate", fake_authenticate)
    def fake_train(config, data, run_root, **kwargs):
        run = run_root / "known-run"
        run.mkdir()
        (run / "last.pt").write_bytes(b"checkpoint")
        (run / "run.json").write_text(json.dumps({"completed_epoch": 1, "stop_reason": "wall_budget_epoch_boundary"}))
        return run
    monkeypatch.setattr(study, "train", fake_train)
    study.segment(config_path, protocol_path, tmp_path, tmp_path / "campaign", segment_id="001", gate_path=gate_path)
    progress = json.loads((tmp_path / "campaign/progress.json").read_text())
    assert progress["selected_config_sha256"] == study.file_sha256(selected_path)
    assert progress["selected_config_sha256"] != study.file_sha256(config_path)
    assert progress["protocol_sha256"] == study.file_sha256(protocol_path)
    assert progress["checkpoint_sha256"] == study.file_sha256(Path(progress["checkpoint"]))
    assert progress["identity"] == identity and progress["epoch"] == 1
    assert not (tmp_path / "campaign/active-segment.json").exists()


def test_checkpoint_config_json_preserves_tuple_list_resume_identity():
    original = asdict(training.TrainConfig(checkpoint_epochs=(25, 50), lr_milestones=((61, .0003),)))
    roundtrip = json.loads(json.dumps(original))
    assert original != roundtrip
    assert json.dumps(original, sort_keys=True) == json.dumps(roundtrip, sort_keys=True)


def test_generation_all_attempt_denominators_and_monitor_rng_neutrality(tmp_path):
    dataset = TinyDataset("val")
    dataset.split = "val"
    dataset.smiles = [row["smiles"] for row in dataset.rows]
    model = GraphAutoencoder(dataset.spec, ModelConfig(hidden=8, latent=4, layers=1))
    protocol = study.read_protocol(Path("configs/large_triangle_protocol.json"))
    protocol.update(prior_count=5, evaluation_batch_size=2, categorical_samples_per_latent=2,
                    local_geometry_count=4, interpolation_pairs=2)
    # TinyDataset deliberately consumes global RNG; isolate that canary here to
    # test evaluator loaders, which real GraphDataset exposes deterministically.
    class DeterministicDataset(torch.utils.data.Dataset):
        split, spec, smiles = dataset.split, dataset.spec, dataset.smiles
        def __len__(self):
            return len(dataset)
        def __getitem__(self, index):
            row = dataset.rows[index]
            return {name: torch.tensor(row[key], dtype=torch.long) for name, key in
                    (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))}
    before = torch.get_rng_state().clone()
    report = evaluate_panel(model, DeterministicDataset(), protocol, tmp_path / "panel", limit=4, deep=True,
                            loss=LossConfig(size_weight=0))
    assert torch.equal(before, torch.get_rng_state())
    assert report["prior"]["argmax"]["attempts"] == 5
    assert report["prior"]["categorical"]["attempts"] == 10
    with gzip.open(tmp_path / "panel/prior.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 15 and all("graph" in row and "valid" in row for row in rows)


@pytest.mark.parametrize("field,value", [("mode", "ae"), ("device", "cpu"), ("num_workers", 1),
                                        ("learning_rate", float("nan")), ("clip_grad", float("inf"))])
def test_reject_unsupported_large_training_modes(field, value):
    config = training.TrainConfig.from_yaml(Path("configs/vae_large_triangle.yaml"))
    with pytest.raises(ValueError):
        study.validate_config(replace(config, **{field: value}))


@pytest.mark.parametrize("scheduler_state", ["DONE", "", "RUN"])
def test_durable_completed_stage_requires_artifacts_even_after_scheduler_history_expires(tmp_path, monkeypatch, scheduler_state):
    monkeypatch.syspath_prepend(str(Path.cwd() / "scripts/hpc"))
    import submit_large_triangle as submission
    from latentmol.experiments import file_sha256
    campaign = tmp_path / "campaign"
    receipts = campaign / "receipts"
    completions = campaign / "stage-completions"
    receipts.mkdir(parents=True)
    completions.mkdir()
    checkpoint = campaign / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable checkpoint")
    release = tmp_path / "release"
    release.mkdir()
    identity = {"release": str(release), "campaign": str(campaign), "bindings_sha256": "1" * 64,
                "manifest_sha256": "2" * 64, "archive_sha256": "3" * 64}
    plan = submission.workflow("segment", release, tmp_path, campaign, tmp_path / "bindings.json",
                               "1" * 64, "3" * 64, "2" * 64, run_id="001", segment_id="001")[0]
    intent = {"identity": identity, "cwd": str(release),
              "command": plan["arguments"] + [shlex.join(plan["job_command"])]}
    (receipts / "segment-001-intent.json").write_text(json.dumps(intent))
    (receipts / "segment-001-receipt.json").write_text(json.dumps({"intent": intent, "job_id": "17",
                                                                  "status": "submitted_not_passed"}))
    completion = {"status": "passed", "stage_returncode": 0, "job_id": "17", "stage": "segment-001",
                  "bindings_sha256": "1" * 64, "source_sha256": "2" * 64, "source_archive_sha256": "3" * 64,
                  "artifacts": {str(checkpoint): file_sha256(checkpoint)}}
    (completions / "segment-001.json").write_text(json.dumps(completion))
    monkeypatch.setattr(submission.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=scheduler_state, stderr=""))
    if scheduler_state == "RUN":
        with pytest.raises(ValueError, match="not DONE"):
            submission.require_completed_stage(receipts, "segment-001", identity)
    else:
        assert submission.require_completed_stage(receipts, "segment-001", identity) is None
        checkpoint.write_bytes(b"changed checkpoint")
        with pytest.raises(ValueError, match="Completion artifact SHA256 mismatch"):
            submission.require_completed_stage(receipts, "segment-001", identity)


@pytest.mark.parametrize("scheduler_state", ["PEND", "RUN", "DONE", "", "EXIT"])
def test_queued_benchmark_needs_authenticated_prepare_and_successful_completion(tmp_path, monkeypatch, scheduler_state):
    monkeypatch.syspath_prepend(str(Path.cwd() / "scripts/hpc"))
    import submit_large_triangle as submission
    calls = []
    monkeypatch.setattr(submission, "require_submitted_stage", lambda *args: ("29", "a" * 64))
    monkeypatch.setattr(submission, "require_completed_stage", lambda *args: calls.append("completed"))
    monkeypatch.setattr(submission.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=scheduler_state, stderr=""))
    if scheduler_state == "EXIT":
        with pytest.raises(ValueError, match="not active or successfully complete"):
            submission.require_queued_prepare(tmp_path, {})
        assert not calls
    else:
        job, receipt_sha, dependencies = submission.require_queued_prepare(tmp_path, {})
        assert (job, receipt_sha) == ("29", "a" * 64)
        assert dependencies == (("29",) if scheduler_state in ("PEND", "RUN") else ())
        assert calls == ([] if dependencies else ["completed"])


def test_benchmark_workflow_pins_prepare_job_and_receipt_to_runtime_guard(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd() / "scripts/hpc"))
    import submit_large_triangle as submission
    rows = submission.workflow("benchmark", tmp_path / "release", tmp_path, tmp_path / "campaign",
        tmp_path / "bindings.json", "1" * 64, "2" * 64, "3" * 64,
        run_id="attempt001", dependency_ids=("29",), prepare_job_id="29", prepare_receipt_sha256="4" * 64)
    assert rows[0]["arguments"][rows[0]["arguments"].index("-w") + 1] == "done(29)"
    command = rows[0]["job_command"]
    assert command[command.index("--prepare-job-id") + 1] == "29"
    assert command[command.index("--prepare-receipt-sha256") + 1] == "4" * 64
    with pytest.raises(ValueError, match="Only benchmark"):
        submission.workflow("segment", tmp_path / "release", tmp_path, tmp_path / "campaign",
            tmp_path / "bindings.json", "1" * 64, "2" * 64, "3" * 64,
            run_id="attempt001", prepare_job_id="29", prepare_receipt_sha256="4" * 64)


def test_runtime_benchmark_checks_bound_manifest_and_prepare_receipt_before_model_work(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd() / "scripts/hpc"))
    import large_triangle_stage as stages
    campaign = tmp_path / "campaign"
    (campaign / "receipts").mkdir(parents=True)
    (campaign / "stage-completions").mkdir()
    data = tmp_path / "data"
    data.mkdir()
    manifest = data / "manifest.json"
    manifest.write_text("{}")
    receipt = campaign / "receipts/prepare-receipt.json"
    receipt.write_text("{}")
    completion = campaign / "stage-completions/prepare.json"
    completion.write_text(json.dumps({"artifacts": {}}))
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "3" * 64)
    monkeypatch.setattr(stages, "source_manifest", lambda _: {"sha256": "2" * 64})
    monkeypatch.setattr(stages, "require_stage_completion", lambda *args: "29")
    args = SimpleNamespace(bindings_sha256="1" * 64, prepare_job_id="29",
                           prepare_receipt_sha256=stages.file_sha256(receipt))
    value = {"campaign": str(campaign), "data": str(data)}
    with pytest.raises(ValueError, match="did not certify the bound data manifest"):
        stages.require_benchmark_preparation(args, value, tmp_path)
    completion.write_text(json.dumps({"artifacts": {str(manifest): stages.file_sha256(manifest)}}))
    assert stages.require_benchmark_preparation(args, value, tmp_path) == "29"
    receipt.write_text("changed receipt")
    with pytest.raises(ValueError, match="receipt changed"):
        stages.require_benchmark_preparation(args, value, tmp_path)
