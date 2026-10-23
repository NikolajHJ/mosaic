"""HPC engineering fixtures; synthetic success is not scientific evidence."""

from dataclasses import asdict
import json

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.decoder_finetune import (FineTuneConfig, assess_advantage, fit_case, make_generators,
    make_panel, read_protocol, rng_receipt, tensor_sha)
from latentmol.decoder_regularization import binary_log_probabilities, freeze_encoder, reference_anchors
from latentmol.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.losses import LossConfig
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.train import state_digest


def test_anchor_zero_at_identity_and_finite_confident_tails():
    spec = GraphSpec(max_nodes=3)
    generator = torch.Generator().manual_seed(27)
    output = DecoderOutput(torch.randn(2, 3, 10, generator=generator),
        torch.randn(2, 3, 5, generator=generator), torch.randn(2, 3, 3, 5, generator=generator))
    same = reference_anchors(output, output, spec)
    assert all(abs(float(value)) < 1e-7 for value in same.values())
    logits = torch.tensor([[1000., -1000., -900.]], requires_grad=True)
    logp = binary_log_probabilities(logits, [1, 2])
    assert bool(torch.isfinite(logp).all())
    logp[:, 1].sum().backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum()) > 0


def test_anchor_blocks_padding_mask_escape_and_reference_gradients():
    spec = GraphSpec(max_nodes=2)
    reference = DecoderOutput(torch.zeros(1, 2, 10, requires_grad=True),
        torch.zeros(1, 2, 5, requires_grad=True), torch.zeros(1, 2, 2, 5, requires_grad=True))
    atoms = torch.full((1, 2, 10), -20.)
    atoms[..., 0] = 20
    bonds = torch.zeros(1, 2, 2, 5)
    bonds[..., -1] = 20
    current = DecoderOutput(atoms.requires_grad_(), torch.zeros(1, 2, 5), bonds.requires_grad_())
    anchors = reference_anchors(current, reference, spec)
    assert float(anchors["occupancy_anchor"]) > 10
    assert float(anchors["aromatic_anchor"]) > 1
    sum(anchors.values()).backward()
    assert reference.atoms.grad is None and reference.bonds.grad is None
    assert float(current.atoms.grad.abs().sum()) > 0 and float(current.bonds.grad.abs().sum()) > 0


def test_dedicated_streams_and_strict_anchored_config():
    left, right = make_generators(17, "cpu"), make_generators(17, "cpu")
    torch.randn(100, generator=left["connectivity"])
    for name in ("prior", "posterior", "loader"):
        torch.testing.assert_close(left[name].get_state(), right[name].get_state())
    assert tensor_sha(make_panel(8, 4, 17)) == tensor_sha(make_panel(8, 4, 17))
    with pytest.raises(ValueError):
        FineTuneConfig(occupancy_anchor_weight=0)
    with pytest.raises(ValueError):
        FineTuneConfig(chemistry={"size_weight": 1})
    with pytest.raises(ValueError):
        FineTuneConfig(connectivity={"weight": 0})


class TinyDataset:
    def __init__(self, spec):
        self.spec = spec
        rows = [encode_smiles(smi, spec) for smi in ("C", "CC", "CO", "CN")]
        self.smiles = [row["smiles"] for row in rows]
        self.rows = [{key: torch.tensor(row[source], dtype=torch.long) for key, source in
                      (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))} for row in rows]
        self.manifest = {"dataset_sha256": "a" * 64, "split_sha256": "b" * 64}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_matched_real_cpu_continuations_freeze_encoder_and_restore_eval_compatible_checkpoint(tmp_path):
    spec = GraphSpec(max_nodes=3)
    config = ModelConfig(hidden=8, latent=4, layers=1)
    model = GraphAutoencoder(spec, config)
    initial = tmp_path / "initial.pt"
    torch.save({"format_version": 1, "graph_spec": spec.to_dict(), "model_config": asdict(config),
        "model_state": model.state_dict(), "loss_config": asdict(LossConfig()),
        "dataset_sha256": "a" * 64, "train_config": {"mode": "vae", "beta": .001},
        "epoch": 89}, initial)
    data = TinyDataset(spec)
    fine = FineTuneConfig(epochs=1, batch_size=2, prior_batch_size=2, chemistry_reconstruction_batch=2)
    results = [fit_case(initial, data, data, tmp_path / "runs", fine, seed=17, coefficient=alpha,
        train_limit=4, val_limit=4, validation_panel=make_panel(2, 4, 81000), device="cpu",
        source_root=tmp_path, budget_seconds=120) for alpha in (0., .1, 1.)]
    assert results[0]["terminal_rng"] == results[1]["terminal_rng"] == results[2]["terminal_rng"]
    assert results[0]["encoder_sha256"] == results[1]["encoder_sha256"] == state_digest(model.encoder.state_dict())
    for result in results:
        final, checkpoint = load_model(__import__("pathlib").Path(result["checkpoint"]), "cpu", spec)
        assert state_digest(final.encoder.state_dict()) == state_digest(model.encoder.state_dict())
        assert state_digest(final.decoder.state_dict()) != state_digest(model.decoder.state_dict())
        assert checkpoint["decoder_finetune"]["exact_historical_resume"] is False
        assert "continuation" not in checkpoint
        assert checkpoint["decoder_finetune"]["rng_states"]


def test_advantage_requires_reconstruction_and_both_raw_generation_modes():
    reference = {"validity": .99, "exact_reconstruction": .98}
    positive = {"decision": {"validity_positive_ci_both_modes": True, "unique_valid_per_attempt_retained_both_modes": True}}
    assert assess_advantage(reference, reference, reference, positive, positive)["advantage_established_within_prespecified_panel"]
    degraded = {"validity": .99, "exact_reconstruction": .95}
    assert not assess_advantage(reference, reference, degraded, positive, positive)["advantage_established_within_prespecified_panel"]
