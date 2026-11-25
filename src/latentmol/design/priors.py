"""Frozen training-code geometry for comparable latent displacements and priors.

The empirical prior models encoded *training means*. It is not a molecule
likelihood, a chemistry validator, or the VAE's standard-normal prior.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LatentReference(nn.Module):
    """Per-model scaling, PCA KDE, and a guard on discarded PCA directions.

    All buffers are fitted once on training molecules and frozen during a study.
    A scalar bandwidth is expressed in whitened principal-component coordinates.
    """

    def __init__(self, center: Tensor, scale: Tensor, basis: Tensor,
                 eigenvalues: Tensor, references: Tensor, residual_limit: Tensor,
                 variance_retained: Tensor):
        super().__init__()
        for name, value in (("center", center), ("scale", scale),
                            ("basis", basis), ("eigenvalues", eigenvalues),
                            ("references", references), ("residual_limit", residual_limit),
                            ("variance_retained", variance_retained)):
            self.register_buffer(name, value.detach().float().clone())

    @classmethod
    def fit(cls, codes: Tensor, *, reference_count: int = 20000,
            components: int = 32, seed: int = 17) -> "LatentReference":
        """Fit only on train-split posterior means, with deterministic references."""
        if codes.ndim != 2 or len(codes) < 2 or not torch.isfinite(codes).all():
            raise ValueError("Expected at least two finite training latent vectors")
        if not 1 <= components < codes.shape[1] or reference_count < 2:
            raise ValueError("Invalid PCA dimension or KDE reference count")
        codes = codes.detach().to(device="cpu", dtype=torch.float64)
        center = codes.mean(0)
        raw_scale = codes.std(0, unbiased=False)
        floor = max(float(raw_scale.median()) * 0.1, 1e-3)
        scale = raw_scale.clamp_min(floor)
        normalized = (codes - center) / scale
        covariance = normalized.T @ normalized / len(codes)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)[:components]
        values = eigenvalues[order].clamp_min(1e-4)
        basis = eigenvectors[:, order]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(len(codes), generator=generator)[:reference_count]
        references = (normalized[indices] @ basis) / values.sqrt()
        residual = torch.linalg.vector_norm(normalized - (normalized @ basis) @ basis.T, dim=-1)
        residual_limit = torch.quantile(residual, 0.95)
        variance_retained = eigenvalues[order].sum() / eigenvalues.clamp_min(0).sum()
        return cls(center, scale, basis, values, references, residual_limit,
                   variance_retained)

    def normalize(self, codes: Tensor) -> Tensor:
        return (codes - self.center) / self.scale

    def denormalize(self, normalized: Tensor) -> Tensor:
        return self.center + normalized * self.scale

    def features(self, codes: Tensor) -> Tensor:
        return (self.normalize(codes) @ self.basis) / self.eigenvalues.sqrt()

    def residual(self, codes: Tensor) -> Tensor:
        normalized = self.normalize(codes)
        return torch.linalg.vector_norm(normalized - (normalized @ self.basis) @ self.basis.T, dim=-1)

    def log_density(self, codes: Tensor, bandwidth: float, *, chunk_size: int = 2048) -> Tensor:
        """Differentiable Gaussian KDE; chunks avoid a [batch, pool, dimension] tensor."""
        if bandwidth <= 0 or chunk_size < 1:
            raise ValueError("Bandwidth and chunk size must be positive")
        features = self.features(codes)
        reference_norms = self.references.square().sum(-1)
        terms = []
        for chunk in self.references.split(chunk_size):
            norms = reference_norms[len(terms) * chunk_size:len(terms) * chunk_size + len(chunk)]
            distances = (features.square().sum(-1, keepdim=True) + norms[None, :]
                         - 2.0 * features @ chunk.T).clamp_min(0)
            terms.append(torch.logsumexp(-distances / (2.0 * bandwidth**2), dim=-1))
        log_kernel = torch.logsumexp(torch.stack(terms, dim=-1), dim=-1)
        dimension = features.shape[-1]
        return (log_kernel - math.log(len(self.references))
                - dimension * math.log(bandwidth) - dimension * math.log(2 * math.pi) / 2)

    def barrier(self, codes: Tensor, *, bandwidth: float, threshold: float) -> Tensor:
        """Penalize low training-code density and excursions into discarded directions."""
        density_deficit = (threshold - self.log_density(codes, bandwidth)).clamp_min(0)
        residual_excess = (self.residual(codes) - self.residual_limit).clamp_min(0)
        return density_deficit.square() + residual_excess.square()

    def artifact(self, metadata: dict) -> dict:
        return {"format_version": 1, "metadata": metadata,
                "center": self.center.cpu(), "scale": self.scale.cpu(),
                "basis": self.basis.cpu(), "eigenvalues": self.eigenvalues.cpu(),
                "references": self.references.cpu(), "residual_limit": self.residual_limit.cpu(),
                "variance_retained": self.variance_retained.cpu()}

    @classmethod
    def from_artifact(cls, value: dict, expected_metadata: dict) -> "LatentReference":
        if value.get("format_version") != 1 or value.get("metadata") != expected_metadata:
            raise ValueError("Latent reference provenance mismatch")
        return cls(*(value[key] for key in
                     ("center", "scale", "basis", "eigenvalues", "references",
                      "residual_limit", "variance_retained")))
