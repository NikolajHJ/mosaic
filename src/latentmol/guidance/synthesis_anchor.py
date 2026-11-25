"""Frozen smooth latent-library attraction; no new model or predictor training."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


def _finite_matrix(value, name):
    value = torch.as_tensor(value)
    if (value.ndim != 2 or not value.shape[0] or not value.shape[1]
            or not bool(value.isfinite().all())):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return value


def _positive(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def _default_scale(latents):
    return max(float(latents.std()), .1) if latents.numel() > 1 else .1


@torch.no_grad()
def calibrated_temperature(latents, reference_points, *, scale=1., target=4.,
                           iterations=64, minimum=1e-6, maximum=1e4):
    """Choose a fixed softminimum temperature using only saved latent vectors.

    The mean exp(entropy) over the reference panel is matched to ``target``.
    No decoder, synthesis critic or optimization trajectory is evaluated. The
    bounded endpoint is returned if tied distances cannot reach the target;
    callers record the achieved effective count instead of claiming a match.
    """
    bank = _finite_matrix(latents, "Latent library").detach().cpu().double()
    points = _finite_matrix(reference_points, "Calibration points").detach().cpu().double()
    if (points.shape[-1] != bank.shape[-1] or not _positive(scale)
            or not _positive(target) or not 1 <= target <= len(bank)
            or type(iterations) is not int or not 1 <= iterations <= 128
            or not _positive(minimum) or not _positive(maximum) or minimum >= maximum):
        raise ValueError("Incompatible latent-temperature calibration inputs")
    distance = ((points[:, None] - bank[None]) / scale).square().mean(-1)
    if not bool(distance.isfinite().all()):
        raise ValueError("Nonfinite latent-temperature calibration distances")

    def effective(temperature):
        log_weights = (-distance / temperature).log_softmax(-1)
        return (-(log_weights.exp() * log_weights).sum(-1)).exp().mean().item()

    if target <= effective(minimum):
        return float(minimum)
    if target >= effective(maximum):
        return float(maximum)
    low, high = math.log(minimum), math.log(maximum)
    for _ in range(iterations):
        middle = (low + high) * .5
        if effective(math.exp(middle)) < target:
            low = middle
        else:
            high = middle
    return math.exp((low + high) * .5)


def _slot_indices(slots):
    if isinstance(slots, torch.Tensor):
        if slots.ndim != 1:
            raise ValueError("Latent-library active slots must be a one-dimensional mask or index list")
        slots = (slots.nonzero().flatten().tolist() if slots.dtype == torch.bool else slots.tolist())
    if (not isinstance(slots, (list, tuple)) or not slots
            or any(type(index) is not int or index < 0 for index in slots)
            or len(set(slots)) != len(slots)):
        raise ValueError("Latent-library active slots must be distinct nonnegative indices")
    return sorted(slots)


class LatentLibraryCritic(nn.Module):
    def __init__(self, latents, *, temperature=1., scale=None, mode="softmin"):
        super().__init__()
        latents = _finite_matrix(latents, "Latent library").float()
        if not _positive(temperature):
            raise ValueError("Latent-library temperature must be finite and positive")
        if mode not in {"softmin", "centroid"}:
            raise ValueError("Latent-library mode must be softmin or centroid")
        self.register_buffer("latents", latents.detach().clone())
        self.mode = mode
        self.temperature = float(temperature)
        self.scale = float(scale) if scale is not None else _default_scale(latents)
        if not _positive(self.scale):
            raise ValueError("Latent-library scale must be finite and positive")
        self.requires_grad_(False)
        self.adapter_metadata = {"scope": "Smooth latent proximity, not synthesis likelihood",
                                 "library_size": len(latents), "temperature": self.temperature,
                                 "distance_scale": self.scale, "mode": self.mode}

    def _point(self, point):
        try:
            point = _finite_matrix(point, "Latent point")
        except ValueError as exc:
            raise ValueError("Latent point and attraction library are incompatible") from exc
        if not point.is_floating_point() or point.shape[-1] != self.latents.shape[-1]:
            raise ValueError("Latent point and attraction library are incompatible")
        return point

    def forward_latent(self, point):
        point = self._point(point)
        bank = self.latents.to(point)
        if self.mode == "centroid":
            difficulty = ((point - bank.mean(0)) / self.scale).square().mean(-1)
        else:
            distance = ((point[:, None] - bank[None]) / self.scale).square().mean(-1)
            difficulty = -self.temperature * torch.logsumexp(-distance / self.temperature, -1)
            # Duplicating every bank entry leaves the objective unchanged.
            difficulty = difficulty + self.temperature * torch.log(point.new_tensor(len(bank)))
        if not bool(difficulty.isfinite().all()):
            raise ValueError("Nonfinite latent-library attraction objective")
        return {"difficulty": difficulty, "raw_score": -difficulty}

    @torch.no_grad()
    def diagnostics(self, point):
        """Detached, per-point geometry diagnostics for locality and shrinkage."""
        point = self._point(point).detach()
        distance = ((point[:, None] - self.latents.to(point)[None]) / self.scale).square().mean(-1)
        if not bool(distance.isfinite().all()):
            raise ValueError("Nonfinite latent-library diagnostics")
        log_weights = (-distance / self.temperature).log_softmax(-1)
        if self.mode == "centroid":
            log_weights = torch.full_like(log_weights, -math.log(len(self.latents)))
        weights = log_weights.exp()
        entropy = -(weights * log_weights).sum(-1)
        return dict(nearest_distance=distance.min(-1).values,
                    nearest_rms=distance.min(-1).values.sqrt() * self.scale,
                    entropy=entropy, effective_anchors=entropy.exp(),
                    max_anchor_weight=weights.max(-1).values,
                    latent_norm=point.norm(dim=-1))

    @classmethod
    def from_asset(cls, asset, *, checkpoint_sha256, active_slots=None, reference_points=None):
        path = Path(asset["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError("Latent-library asset changed")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("Latent library belongs to another decoder")
        if active_slots is not None:
            if ("active_slots" not in data
                    or _slot_indices(data["active_slots"]) != _slot_indices(active_slots)):
                raise ValueError("Latent-library active slots differ from the search")
        target = asset.get("target_effective_anchors")
        if target is not None and ("temperature" in asset or asset.get("mode", "softmin") != "softmin"):
            raise ValueError("Choose either explicit temperature or effective-anchor calibration in softmin mode")
        temperature = asset.get("temperature", 1.)
        if target is not None:
            if reference_points is None:
                raise ValueError("Effective-anchor calibration requires saved reference points")
            bank = _finite_matrix(data["latents"], "Latent library").float()
            scale = data.get("scale") if data.get("scale") is not None else _default_scale(bank)
            temperature = calibrated_temperature(bank, reference_points, scale=scale, target=target)
        model = cls(data["latents"], temperature=temperature, scale=data.get("scale"),
                    mode=asset.get("mode", "softmin"))
        model.adapter_metadata.update(source_policy=data.get("selection"), path=str(path),
                                      active_slots=data.get("active_slots"))
        if target is not None:
            achieved = float(model.diagnostics(torch.as_tensor(reference_points).float())["effective_anchors"].mean())
            model.adapter_metadata["calibration"] = dict(
                policy="bounded_saved_founder_mean_effective_anchors", target=target,
                achieved=achieved, matched=math.isclose(achieved, target, rel_tol=1e-4, abs_tol=1e-4))
        return model
