"""Opt-in local latent coverage using the existing molecular repair objective.

This changes the distribution of chemistry training points, not the chemistry
oracle. A quarter of the prior budget eventually comes from small neighborhoods
of real molecules. The other three quarters remain independent N(0,I) draws.
No new local-valence loss or reconstruction pseudo-label is introduced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from .representation import HardGraphBatch


@dataclass(frozen=True)
class NeighborhoodConfig:
    version: str = "local-latent-chemistry-v1"
    fraction: float = .25
    noise_scale: float = .05
    ramp_updates: int = 100

    def __post_init__(self):
        if (self.version != "local-latent-chemistry-v1"
                or type(self.fraction) is not float or not 0 < self.fraction <= .5
                or type(self.noise_scale) is not float or not math.isfinite(self.noise_scale)
                or not 0 < self.noise_scale <= .2
                or type(self.ramp_updates) is not int or self.ramp_updates < 1):
            raise ValueError("Invalid bounded local latent sampling policy")


class LocalLatentPrior:
    """Detached sampling around current real-data encoder means.

    Noise is 0.05 times each posterior standard deviation, with each sampled
    direction normalized to unit RMS. Thus the standardized displacement RMS
    is exactly 0.05, rather than occasionally making large random excursions.
    The encoder supplies sampling coordinates under no_grad; only the usual
    real reconstruction/KL branch trains it directly. The existing chemistry
    objective trains the live decoder on these local coordinates.
    """

    def __init__(self, config: NeighborhoodConfig, generator, *, start_update):
        if type(start_update) is not int or start_update < 0:
            raise ValueError("Local coverage needs a fixed starting update")
        if str(generator.device) != "cpu":
            raise ValueError("Local sampling uses its own checkpointed CPU generator")
        self.config, self.generator, self.start_update = config, generator, start_update
        self._record = None

    @torch.no_grad()
    def sample(self, model, hard, standard, *, update, microbatch_size):
        if self._record is not None and self._seen != len(self._kinds):
            raise ValueError("Previous local sampling target receipt is incomplete")
        if (type(update) is not int or update < self.start_update or microbatch_size < 1
                or standard.ndim != 2 or standard.shape[1] != model.config.latent
                or not bool(torch.isfinite(standard).all())):
            raise ValueError("Invalid local coverage input")
        total = len(standard)
        progress = min((update - self.start_update + 1) / self.config.ramp_updates, 1.)
        count = int(total * self.config.fraction * progress)
        self._kinds = ["standard"] * (total - count) + ["local"] * count
        self._seen = 0
        self._coverage = {k: dict(attempts=0, strict_valid=0, accepted_targets=0,
                                  reference_targets=0, chemical_failures=0,
                                  representation_failures=0) for k in ("standard", "local")}
        self._record = dict(policy=asdict(self.config), update=update,
                            standard_points=total-count, local_points=count,
                            ramp_fraction=progress, source_batch_rows=[],
                            mean_displacement_l2=0., max_displacement_l2=0.,
                            standardized_displacement_rms=self.config.noise_scale if count else 0.)
        result = standard.detach().clone()
        if count:
            # Sampling with replacement supports final short real batches.
            rows = torch.randint(len(hard.atoms), (count,), generator=self.generator)
            direction = torch.randn(count, model.config.latent, generator=self.generator)
            direction /= direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
            self._record["source_batch_rows"] = rows.tolist()
            norms = []
            for begin in range(0, count, microbatch_size):
                end = min(begin + microbatch_size, count)
                ids = rows[begin:end].to(hard.atoms.device)
                selected = HardGraphBatch(*(getattr(hard, k)[ids] for k in ("atoms", "charges", "bonds")))
                if not bool((selected.atoms != 0).any(-1).all()):
                    raise ValueError("Local coverage requires nonempty real source molecules")
                mu, logvar = model.encode(selected.to_relaxed(model.spec))
                std = (.5 * logvar).exp()
                if not bool(torch.isfinite(mu).all() & torch.isfinite(std).all()):
                    raise FloatingPointError("Nonfinite local posterior coordinates")
                delta = self.config.noise_scale * std * direction[begin:end].to(mu)
                result[total-count+begin:total-count+end] = mu + delta
                norms.extend(delta.norm(dim=-1).cpu().tolist())
            self._record.update(mean_displacement_l2=sum(norms)/len(norms),
                                max_displacement_l2=max(norms))
        return result

    def observe_targets(self, receipt, *, start, stop, reference_eligible=()):
        if start != self._seen or stop > len(self._kinds) or stop <= start:
            raise ValueError("Local coverage receipts are missing, repeated or out of order")
        attempts = receipt.get("attempts", [])
        if len(attempts) != stop-start:
            raise ValueError("Local coverage requires every raw teacher attempt")
        for index, (kind, attempt) in enumerate(zip(self._kinds[start:stop], attempts)):
            out = self._coverage[kind]
            classification = attempt.get("chemical_classification", {})
            out["attempts"] += 1
            out["strict_valid"] += int(classification.get("strict_teacher_valid") is True)
            out["accepted_targets"] += int(attempt.get("status") == "accepted")
            out["reference_targets"] += int(index in reference_eligible)
            out["chemical_failures"] += int(classification.get("failure_kind") == "chemical")
            out["representation_failures"] += int(classification.get("failure_kind") == "representation")
        self._seen = stop

    def summary(self):
        if self._record is None or self._seen != len(self._kinds):
            raise ValueError("Local coverage has incomplete teacher observations")
        return dict(self._record, teacher_coverage=self._coverage,
                    normalization="same all-attempt prior mean over the declared mixture",
                    objective="unchanged atom-preserving repair plus frozen same-z reference/occupancy and population composition",
                    sampling_gradient="detached encoder coordinates; live decoder retains its gradient")
