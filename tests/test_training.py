"""Numerical continuation/diagnostic regressions; execute only on DTU HPC."""

from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.models import ModelConfig
from latentmol.representation import GraphSpec
import latentmol.train as training


class TinyDataset(torch.utils.data.Dataset):
    """Small actual graphs; consume all CPU RNG streams to test restoration."""

    def __init__(self, split):
        self.spec = GraphSpec(max_nodes=4)
        smiles = ["C", "CC", "CO", "CCO", "CN", "CCC", "CNC", "CCN"]
        self.rows = [encode_smiles(value, self.spec) for value in smiles[:8 if split == "train" else 4]]
        self.manifest = {"dataset_sha256": "tiny-dataset", "split_sha256": "tiny-split"}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Actual GraphDataset is deterministic; these canary draws make omitted
        # Python/NumPy/Torch RNG restoration observable without changing labels.
        random.random()
        np.random.random()
        torch.rand(())
        row = self.rows[index]
        return {name: torch.from_numpy(row[key].astype(np.int64)) for name, key in
                (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))}


@pytest.fixture
def tiny_config(monkeypatch):
    monkeypatch.setattr(training, "GraphDataset", lambda path, split: TinyDataset(split))
    return training.TrainConfig(mode="vae", epochs=3, batch_size=2, learning_rate=0.003,
                                validation_limit=4, chemistry_limit=3, train_probe_limit=4,
                                device="cpu", checkpoint_every=1, full_chemistry_every=2,
                                lr_milestones=((2, 0.0003),),
                                model=ModelConfig(hidden=8, latent=4, layers=1))


def assert_nested_equal(left, right):
    """Compare optimizer/RNG state exactly, including tensors and Python tuples."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("mode", ["ae", "vae"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires an HPC GPU allocation"))])
def test_epoch_resume_matches_uninterrupted_weights_optimizer_rng_and_diagnostics(tmp_path, tiny_config, workers, mode, device):
    config = replace(tiny_config, num_workers=workers, mode=mode, device=device)
    whole = training.train(config, tmp_path, tmp_path / "whole")
    first = training.train(config, tmp_path, tmp_path / "split", stop_after_epoch=1)
    resumed = training.train(config, tmp_path, tmp_path / "split", resume_from=first / "last.pt")
    assert torch.are_deterministic_algorithms_enabled()
    final_a = torch.load(whole / "last.pt", map_location="cpu", weights_only=True)
    final_b = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    for name in ("model_state", "optimizer_state"):
        assert_nested_equal(final_a[name], final_b[name])
    assert_nested_equal(final_a["continuation"]["rng"], final_b["continuation"]["rng"])
    assert final_a["continuation"]["updates"] == final_b["continuation"]["updates"] == 12
    assert final_b["continuation"]["lr_at_checkpoint"] == 0.0003
    assert final_a["epoch"] == final_b["epoch"] == 3
    row_a = json.loads((whole / "latest_epoch.json").read_text())
    row_b = json.loads((resumed / "latest_epoch.json").read_text())
    for key in ("train", "val", "train_probe", "val_probe", "gradient_clipping_rate", "gradient_norm_max"):
        assert_nested_equal(row_a[key], row_b[key])
    # A full-chemistry epoch expands only main validation, never the selection cohort.
    events = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    epoch_two = next(row for row in events if row["epoch"] == 2)
    assert epoch_two["val"]["chemistry_samples"] == 4
    assert epoch_two["train_probe"]["chemistry_samples"] == epoch_two["val_probe"]["chemistry_samples"] == 3
    assert len(final_b["continuation"]["indices"]["train_probe"]) == len(final_b["continuation"]["indices"]["val_probe"])
    assert (resumed / "epoch_0003.pt").exists()
    # New checkpoint additions must not break the existing inference-only loader.
    from latentmol.evaluate import load_model
    evaluated, _ = load_model(resumed / "last.pt", "cpu", expected_spec=TinyDataset("train").spec)
    assert_nested_equal(evaluated.state_dict(), final_b["model_state"])


def test_resume_retains_ancestral_best_when_no_resumed_epoch_improves(tmp_path, tiny_config, monkeypatch):
    def constant_metrics(model, loader, loss_config, device, chemistry_limit, training_mode):
        return {"reconstruction": 1., "validity": 0.5, "exact_reconstruction": 0.25,
                "chemistry_samples": 3}, []
    monkeypatch.setattr(training, "reconstruction_metrics", constant_metrics)
    first = training.train(tiny_config, tmp_path, tmp_path / "runs", stop_after_epoch=1)
    resumed = training.train(tiny_config, tmp_path, tmp_path / "runs", resume_from=first / "last.pt", stop_after_epoch=2)
    for alias in ("best.pt", "best_validity.pt"):
        before = torch.load(first / alias, weights_only=True)
        after = torch.load(resumed / alias, weights_only=True)
        assert after["epoch"] == before["epoch"] == 1
        assert_nested_equal(after["model_state"], before["model_state"])
    last = torch.load(resumed / "last.pt", weights_only=True)
    assert last["epoch"] == 2
    assert all(entry["epoch"] == 1 for entry in last["continuation"]["selections"].values())


def test_resume_rejects_changed_algorithm_data_config_and_legacy_state(tmp_path, tiny_config):
    first = training.train(tiny_config, tmp_path, tmp_path / "runs", stop_after_epoch=1)
    checkpoint = torch.load(first / "last.pt", weights_only=True)
    state = checkpoint["continuation"]
    data = TinyDataset("train")

    def validate(value=checkpoint, config=tiny_config, algorithm=state["algorithm"], dataset=data):
        return training.validate_continuation(value, config, dataset, algorithm, state["runtime"], state["indices"])

    validate(config=replace(tiny_config, epochs=4))  # Planned horizon extension only.
    with pytest.raises(ValueError, match="training config"):
        validate(config=replace(tiny_config, learning_rate=0.1))
    with pytest.raises(ValueError, match="algorithm"):
        validate(algorithm={"sha256": "changed"})
    changed = TinyDataset("train")
    changed.manifest = {**changed.manifest, "split_sha256": "other"}
    with pytest.raises(ValueError, match="dataset or split"):
        validate(dataset=changed)
    with pytest.raises(ValueError, match="remains evaluable"):
        validate(value={**checkpoint, "continuation": None})
    with pytest.raises(ValueError, match="runtime"):
        training.validate_continuation(checkpoint, tiny_config, data, state["algorithm"],
                                       {**state["runtime"], "torch": "different"}, state["indices"])


def test_rng_state_roundtrip_including_cached_gaussians_is_weights_only_safe(tmp_path):
    training.seed_everything(17)
    generators = {"train": torch.Generator().manual_seed(9)}
    random.gauss(0, 1)
    np.random.normal()
    state = training.capture_rng_state(generators, include_cuda=False)
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    expected = (random.gauss(0, 1), np.random.normal(), torch.rand(3), torch.randperm(9, generator=generators["train"]))
    training.seed_everything(89)
    training.restore_rng_state(torch.load(path, weights_only=True), generators)
    observed = (random.gauss(0, 1), np.random.normal(), torch.rand(3), torch.randperm(9, generator=generators["train"]))
    assert_nested_equal(expected, observed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an HPC GPU allocation")
def test_cuda_rng_state_is_restored_from_weights_only_checkpoint(tmp_path):
    training.seed_everything(31)
    generators = {"train": torch.Generator().manual_seed(41)}
    state = training.capture_rng_state(generators, include_cuda=True)
    path = tmp_path / "cuda_rng.pt"
    torch.save(state, path)
    expected = torch.rand(11, device="cuda")
    torch.rand(27, device="cuda")
    training.restore_rng_state(torch.load(path, weights_only=True), generators)
    assert torch.equal(expected, torch.rand(11, device="cuda"))


def test_absolute_epoch_learning_rate_policy_and_legacy_defaults():
    legacy = training.TrainConfig()
    assert legacy.epochs == 60 and legacy.lr_milestones == ()
    config = replace(legacy, epochs=120, lr_milestones=((61, 0.0003), (81, 0.0001), (101, 0.00003)))
    assert [training.learning_rate_at_epoch(config, epoch) for epoch in (60, 61, 80, 81, 100, 101, 120)] == [
        0.001, 0.0003, 0.0003, 0.0001, 0.0001, 0.00003, 0.00003]
    with pytest.raises(ValueError, match="milestones"):
        replace(config, lr_milestones=((81, 0.001), (61, 0.0001)))
