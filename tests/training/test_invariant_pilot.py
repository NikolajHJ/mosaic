"""Bounded CPU fixture for the new branch, its anchor and paired evaluation."""
from dataclasses import asdict
import json

import pytest
import torch

from latentmol.datasets.data import prepare_dataset
from latentmol.experiments import file_sha256
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.train import state_digest
from latentmol.training.invariant_pilot import initialize_invariant, latent_summary, reference_decoder_kl, run_pilot
from latentmol.training.losses import LossConfig


def test_warm_start_keeps_decoder_and_identity_anchor_has_zero_gradient():
    torch.manual_seed(8)
    reference = GraphAutoencoder(GraphSpec(max_nodes=6), ModelConfig(hidden=8, latent=4, layers=1))
    reference.requires_grad_(False); reference.eval()
    before = state_digest(reference.state_dict())
    model, receipt = initialize_invariant(reference)
    assert receipt["removed"] == ["encoder.position"]
    assert receipt["initialized"] == ["encoder.pool.1.weight"]
    assert model.config.encoder_pooling == "invariant"
    assert state_digest(model.decoder.state_dict()) == state_digest(reference.decoder.state_dict())
    loss = reference_decoder_kl(model, reference, torch.randn(3, 4))
    assert float(loss) == pytest.approx(0., abs=1e-7)
    loss.backward()
    assert max(float(p.grad.abs().max()) for p in model.decoder.parameters() if p.grad is not None) < 1e-6
    assert state_digest(reference.state_dict()) == before


@pytest.mark.parametrize("protocol", [1, 2, 3])
def test_full_small_pilot_preserves_parent_and_compares_order_robustness(tmp_path, protocol):
    spec = GraphSpec(max_nodes=8)
    source = tmp_path/"molecules.smi"
    source.write_text("\n".join(["C", "CC", "CCC", "CCCC", "CCO", "CCN", "CCS", "CCF", "CO", "CN", "CCl", "CBr"]))
    data = tmp_path/"data"
    prepare_dataset(source, data, spec=spec, train_count=6, val_count=4, seed=13)
    manifest = json.loads((data/"manifest.json").read_text())
    parent = tmp_path/"epoch45.pt"
    reference = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=4,
        decoder_variant="triangle_refine", global_hidden=8, encoder_residual_layers=2,
        pair_hidden=4, refinement_rounds=4, attention_heads=2, triangle_hidden=2))
    torch.save(dict(format_version=1, epoch=45, graph_spec=spec.to_dict(), model_config=asdict(reference.config),
        model_state=reference.state_dict(), dataset_sha256=manifest["dataset_sha256"],
        loss_config=asdict(LossConfig()), train_config=dict(mode="vae")), parent)
    evaluation = dict(prior_count=4, prior_seeds=[17, 18], categorical_seed=19, temperature=1.,
        reconstruction_count=4, row_seed=20, exclude_reconstruction_panels=[], anchor_count=2,
        directions=1, direction_seed=21, scales=[.05], batch_size=2,
        atom_order=dict(count=2, permutations=2, seed=22))
    config = dict(version="invariant-matched-vae-pilot-v1", epoch=45, checkpoint=str(parent),
        checkpoint_sha256=file_sha256(parent), data=str(data), dataset_sha256=manifest["dataset_sha256"],
        data_manifest_sha256=file_sha256(data/"manifest.json"), seed=23, train_seed=24,
        train_count=6, batch_size=2, warmup_updates=2, joint_updates=2,
        warmup_learning_rate=1e-4, encoder_learning_rate=1e-4, decoder_learning_rate=1e-5,
        weight_decay=1e-5, clip_grad=5., beta=.001, warmup_mu_weight=1., warmup_logvar_weight=.1,
        prior_anchor_count=2, prior_anchor_weight=.25, checkpoint_every=2, health_every=2,
        health_reconstruction_count=2, health_prior_count=4, health_seed=25,
        matching=dict(iterations=1, degree_weight=.1, isomorphism_states=64, assignment_backend="numpy"),
        evaluation=evaluation)
    if protocol == 2:
        config.update(version="invariant-matched-vae-pilot-v2", warmup_objective="teacher_latent",
            teacher_calibration_count=4, joint_mu_weight=2., joint_logvar_weight=.1,
            invariant_encoder=dict(aggregation="sum", degree_features=True, sum_scale=3.))
    if protocol == 3:
        config.update(version="invariant-matched-vae-pilot-v3", warmup_objective="reconstruction_and_latent",
            teacher_calibration_count=4, joint_mu_weight=2., joint_logvar_weight=.1,
            invariant_encoder=dict(pooling="canonical"))
    output = tmp_path/"pilot"
    result = run_pilot(config, output, device="cpu")
    assert result["status"] == "completed" and result["frozen_parent_unchanged"]
    assert not result["interpolation_loss_used"] and not result["chemistry_search_used"]
    assert file_sha256(parent) == config["checkpoint_sha256"]
    assert result["candidates"]["invariant-pilot"]["atom_order"]["posterior_within_tolerance_fraction"] == 1.
    assert (output/"comparison/summary.md").is_file()
    warm = torch.load(result["checkpoints"]["warmup"]["path"], map_location="cpu", weights_only=True)
    assert all(torch.equal(value, warm["model_state"]["decoder."+key]) for key, value in reference.decoder.state_dict().items())
    if protocol == 2:
        assert warm["model_config"]["encoder_aggregation"] == "sum"
        calibration = json.loads((output/"teacher-calibration.json").read_text())
        assert calibration["split"] == "train" and len(calibration["indices"]) == 4
        records = [json.loads(line) for line in (output/"training.jsonl").read_text().splitlines()]
        updates = [row for row in records if "loss" in row]
        warmup = [row for row in updates if row["phase"] == "warmup"]
        assert all(row["reconstruction"] is None and row["teacher_only_warmup"] for row in warmup)
        assert all(not row["matching"]["used"] for row in warmup)
        assert all(row["coordinate_mse"] > 0 for row in updates if row["phase"] == "joint")
        assert "effective_rank" in json.loads((output/"initial-health/summary.json").read_text())["latent"]
    if protocol == 3:
        start = json.loads((output/"canonical-start.json").read_text())
        assert start["full_graph_labels_identical"] and start["max_mu_abs_error"] == 0
        assert result["encoder_input_mode"] == "canonical"
        assert result["initialization"]["encoder_initially_identical"]
        assert json.loads((output/"initial-health/summary.json").read_text())["latent"]["teacher_mse"] == 0
    with pytest.raises(FileExistsError):
        run_pilot(config, output, device="cpu")


def test_latent_health_flags_biased_constant_codes():
    torch.manual_seed(7)
    teacher = torch.randn(16, 8)
    constant = torch.full_like(teacher, 3.)
    stats = latent_summary(constant, teacher, scale=.2)
    assert stats["effective_rank"] == 0 and stats["variance_ratio"] == 0
    assert stats["normalized_teacher_mse"] > 30
    assert latent_summary(teacher, teacher)["teacher_mse"] == 0
