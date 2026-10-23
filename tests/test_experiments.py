"""Provenance and checkpoint tests for HPC execution."""

import json

import pytest
import torch

from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.sampling import subset_indices


def test_failed_experiment_keeps_error_and_config(tmp_path):
    with pytest.raises(RuntimeError):
        with ExperimentRun(tmp_path, "failure-test", {"seed": 17}, tmp_path) as run:
            run.log({"step": 1, "loss": 2.0})
            raise RuntimeError("intentional test exception")
    record = json.loads((run.path / "run.json").read_text())
    assert record["status"] == "failed"
    assert record["config"]["seed"] == 17
    assert len(record["config_sha256"]) == 64
    assert "intentional test exception" in (run.path / "traceback.txt").read_text()


def test_sampling_is_reproducible_and_not_canonical_prefix():
    first = subset_indices(1000, 100, 17)
    assert first == subset_indices(1000, 100, 17)
    assert first != list(range(100))
    assert first != subset_indices(1000, 100, 29)
    assert len(set(first)) == 100


def test_checkpoint_roundtrip_and_schema_rejection(tmp_path):
    spec, config = GraphSpec(max_nodes=4), ModelConfig(hidden=16, latent=8, layers=1)
    model = GraphAutoencoder(spec, config).eval()
    from dataclasses import asdict
    path = tmp_path / "model.pt"
    torch.save({"format_version": 1, "graph_spec": spec.to_dict(), "model_config": asdict(config),
                "model_state": model.state_dict()}, path)
    restored, _ = load_model(path, "cpu", spec)
    z = torch.randn(2, 8)
    assert torch.equal(model.decode(z).atoms, restored.decode(z).atoms)
    with pytest.raises(ValueError, match="schemas differ"):
        load_model(path, "cpu", GraphSpec(max_nodes=5))
