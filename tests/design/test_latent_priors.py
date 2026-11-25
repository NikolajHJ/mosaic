"""Numerical Stage 1 checks, executed only in an HPC allocation."""

import pytest
import torch

from latentmol.latent_optimization import optimize_latents
from latentmol.latent_priors import LatentReference
from latentmol.latent_study import _attempt_metrics, _cohorts


def _reference():
    generator = torch.Generator().manual_seed(11)
    codes = torch.randn(64, 6, generator=generator)
    codes[:, 4] *= 0.2
    return LatentReference.fit(codes, reference_count=32, components=3, seed=12)


def test_reference_provenance_and_chunking():
    reference = _reference()
    artifact = reference.artifact({"checkpoint_sha256": "original", "split": "train"})
    restored = LatentReference.from_artifact(artifact, {"checkpoint_sha256": "original", "split": "train"})
    with pytest.raises(ValueError, match="provenance"):
        LatentReference.from_artifact(artifact, {"checkpoint_sha256": "different", "split": "train"})
    sample = torch.tensor([[0.1, 0.3, -0.4, 0.6, 0.08, -0.2]])
    whole = restored.log_density(sample, 1.0, chunk_size=32)
    chunked = restored.log_density(sample, 1.0, chunk_size=5)
    torch.testing.assert_close(whole, chunked, atol=2e-5, rtol=2e-5)
    assert 0 < reference.variance_retained <= 1


def test_reference_gradient_matches_central_difference():
    reference = _reference()
    point = torch.tensor([[0.1, 0.3, -0.4, 0.6, 0.08, -0.2]], requires_grad=True)
    reference.log_density(point, 1.1, chunk_size=7).sum().backward()
    analytic = float(point.grad[0, 2])
    epsilon = 1e-3
    plus, minus = point.detach().clone(), point.detach().clone()
    plus[0, 2] += epsilon
    minus[0, 2] -= epsilon
    finite = float((reference.log_density(plus, 1.1, chunk_size=7)
                    - reference.log_density(minus, 1.1, chunk_size=7)) / (2 * epsilon))
    assert abs(analytic - finite) < 0.01


def test_residual_barrier_has_finite_gradient_at_training_center():
    reference = _reference()
    point = reference.center.detach().clone().unsqueeze(0).requires_grad_(True)
    reference.barrier(point, bandwidth=1.0, threshold=-1e9).sum().backward()
    assert point.grad is not None and torch.isfinite(point.grad).all()


def test_latent_updater_and_cohort_denominators():
    starts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchor = torch.zeros_like(starts)
    endpoint, summary = optimize_latents(starts, anchor, lambda x: x.square().sum(-1),
                                         steps=8, learning_rate=0.1, max_move=0.5)
    assert torch.all((endpoint - starts).norm(dim=-1) <= 0.50001)
    assert summary["final_loss"] < summary["first_loss"]
    cohorts = _cohorts(1000, 23)
    assert len(set(cohorts["calibration"] + cohorts["pilot"] + cohorts["promotion_reserved"])) == 712
    rows = [
        {"validation_index": 0, "input_heavy_atoms": 2, "baseline": {"smiles": "CC"},
         "endpoint": {"smiles": "CO", "heavy_atoms": 2}},
        {"validation_index": 1, "input_heavy_atoms": 1, "baseline": {"smiles": None},
         "endpoint": {"smiles": "C", "heavy_atoms": 1}},
        {"validation_index": 2, "input_heavy_atoms": 2, "baseline": {"smiles": "CN"},
         "endpoint": {"smiles": "CN", "heavy_atoms": 2}},
        {"validation_index": 3, "input_heavy_atoms": 1, "baseline": {"smiles": "N"},
         "endpoint": {"smiles": None, "heavy_atoms": 1}},
    ]
    metrics = _attempt_metrics(rows)
    assert metrics["attempts"] == 4
    assert metrics["valid_to_different_valid"] == 1
    assert metrics["valid_changed_size_compliant"] == 1
    assert metrics["invalid_to_valid_rescues"] == 1
    assert metrics["unchanged_valid"] == 1
