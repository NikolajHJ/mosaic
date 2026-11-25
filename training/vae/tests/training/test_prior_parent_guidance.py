"""Regression checks for same-latent occupancy supervision and held-out reporting."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.datasets.bulk_view import BalancedSparseCorpus, build_balanced_view
from latentmol.experiments import file_sha256
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.train import TrainConfig, capture_rng_state, state_digest
from latentmol.training import combined_continuation as continuation
from latentmol.training import prior_repair
from latentmol.training.occupancy_monitor import OccupancyMonitor
from test_bulk_view import _sources


def test_only_valid_reference_graphs_supervise_element_charge_and_bonds():
    spec = GraphSpec(max_nodes=4)
    data = encode_smiles("CO", spec)
    hard = HardGraphBatch(*(torch.as_tensor(data[key]).unsqueeze(0).repeat((2,) + (1,)*data[key].ndim)
                            for key in ("atom_types", "charges", "bond_types")))
    # The second draw has two disconnected atoms. The frozen reference still
    # supplies its occupancy on that draw, but no conditional chemical labels.
    hard.bonds[1].zero_()
    reference_output = DecoderOutput(*(5. * F.one_hot(x.long(), n).float() for x, n in zip(
        (hard.atoms, hard.charges, hard.bonds),
        (len(spec.atomic_numbers), len(spec.formal_charges), len(spec.bond_names)), strict=True)))

    class Reference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec

        def decode(self, z):
            assert len(z) == 2
            return reference_output

    p, labels, valid = prior_repair.parent_prior_targets(Reference().eval(), torch.zeros(2, 4))
    assert valid.tolist() == [True, False] and not p.requires_grad
    torch.testing.assert_close(p, 1 - reference_output.atoms.softmax(-1)[..., 0])
    live = DecoderOutput(*(torch.zeros_like(x, requires_grad=True)
                           for x in (reference_output.atoms, reference_output.charges, reference_output.bonds)))
    loss = prior_repair.conditional_parent_loss(live, labels, valid)
    loss.backward()
    for logits in (live.atoms, live.charges, live.bonds):
        assert logits.grad.isfinite().all()
        assert logits.grad[0].abs().sum() > 0
        assert logits.grad[1].abs().sum() == 0
    assert live.atoms.grad[..., 0].abs().sum() == 0
    assert live.atoms.grad[:, 2:].abs().sum() == 0
    live.atoms.grad = None
    F.binary_cross_entropy_with_logits(prior_repair.occupancy_logits(live.atoms), p).backward()
    assert live.atoms.grad[1].abs().sum() > 0


def test_combined_cli_records_prior_and_pubchem_panels_without_perturbing_rng(tmp_path, monkeypatch):
    canonical, packed, options = _sources(tmp_path)
    view = tmp_path / "view"
    manifest = build_balanced_view(canonical, packed, view, **options)
    corpus = BalancedSparseCorpus(view)
    model = GraphAutoencoder(corpus.spec, ModelConfig(hidden=8, latent=4, layers=1, encoder_pooling="canonical"))
    tc = TrainConfig(mode="vae", device="cpu", batch_size=2, model=model.config)
    parent = tmp_path / "parent.pt"
    opt = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    torch.save(dict(format_version=1, graph_spec=corpus.spec.to_dict(), model_config=asdict(model.config),
                    train_config=asdict(tc), loss_config=asdict(tc.loss), model_state=model.state_dict(),
                    optimizer_state=opt.state_dict(), epoch=46, continuation=dict(epoch=46, row=0, updates=8,
                        epoch_complete=True, rng=capture_rng_state({"fresh_prior": torch.Generator().manual_seed(17)}, False))), parent)
    reference = {"path": str(parent), "sha256": file_sha256(parent)}
    config = json.loads((Path(__file__).resolve().parents[2] / "experiments/vae/epoch46-pubchem60m-prior15.json").read_bytes())
    config.pop("cuda_requirement")
    config.update(data=str(view), data_sha256=manifest["dataset_sha256"], checkpoint=reference,
                  batch_sizes=[2], microbatch_sizes=[1], checkpoint_every_updates=1)
    config["protocol"]["microbatch_size"] = 1
    config["execution"].update(torch_threads=1, workers=0)
    config["classification_execution"] = None
    config["prior_only"].update(real_per_prior=2, guidance=dict(
        version="frozen-parent-occupancy-chemistry-v1", reference=reference, occupancy_weight=1., conditional_weight=1.))
    panel = dict(seed=5191, prior_points=4, validation_points=1, interval_updates=2, early_updates=[1], batch_size=2)
    # An independent probe verifies that evaluating in training mode neither
    # advances the training RNG nor changes parameters or the caller's mode.
    rng = torch.get_rng_state().clone()
    frozen = state_digest(model.state_dict())
    monitor = OccupancyMonitor(dict(panel, monitor_motifs=True, generated_roundtrip_points=3),
                               corpus.split("val"), model, tmp_path, 8)
    monitor.record(model, dict(updates=8), force=True)
    monitor.record(model, dict(updates=8), force=True)
    assert torch.equal(torch.get_rng_state(), rng) and model.training
    assert state_digest(model.state_dict()) == frozen
    assert len(monitor.path.read_text().splitlines()) == 1
    rt = json.loads(monitor.path.read_text())["generated_roundtrip"]
    assert rt["leg_start_update"] == 8 and rt["fixed_at_leg_start"] == rt["current_prior_parents"]
    scores = rt["fixed_at_leg_start"]
    assert 0 <= scores["mean_exact"] <= scores["mean_valid"] <= scores["eligible_parents"] <= scores["prior_attempts"] == 3
    monitor.record(model, dict(updates=9), force=True)
    first, repeated = (json.loads(row) for row in monitor.path.read_text().splitlines())
    assert first["prior_categorical_motifs"] == repeated["prior_categorical_motifs"]
    assert first["prior_argmax_motifs"]["attempts"] == first["prior_categorical_motifs"]["attempts"] == 4
    assert first["heldout_input_motifs"]["attempts"] == 1 and torch.equal(torch.get_rng_state(), rng)
    config["quality_panel"] = panel
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["combined_continuation", "--config", str(cfg), "--output", str(output),
                                      "--device", "cpu", "--batch-size", "2", "--max-updates", "2"])
    continuation.main()
    metrics = [json.loads(s) for s in (output / "quality-panel.jsonl").read_text().splitlines()]
    assert [r["updates"] for r in metrics] == [8, 9, 10]
    for row in metrics:
        assert row["validation_rows"] == [1] and row["prior_points"] == 4
        assert 0 <= row["reconstruction_exact"] <= row["reconstruction_valid"] <= 1
        assert row["prior_unique_valid"] <= row["prior_valid"] <= 4
        assert sum(row["prior_size_histogram"].values()) == row["prior_valid"] + sum(row["prior_failures"].values()) == 4
    steps = [json.loads(s) for s in (output / "training.jsonl").read_text().splitlines()]
    for step in steps:
        assert step["fresh_prior_points"] == 1 and step["pre_clip_gradient_norm"] > 0
        assert step["prior_guidance"] == prior_repair.GUIDANCE_VERSION and step["prior_occupancy_bce"] > 0
        assert step["total"] == pytest.approx(step["reconstruction"] + step["beta"] * step["kl"]
            + .375 * step["prior_repair"] + step["prior_occupancy_bce"] + step["prior_conditional_ce"])
    assert file_sha256(parent) == reference["sha256"]
