"""Measure compact VAE occupancy on the same held-out standard-normal latent vectors."""

from collections import Counter
import hashlib
import math
from statistics import mean, median

import torch

from latentmol.chemistry import inspect_graph
from latentmol.chemistry_early_health import preserve_observation_state
from latentmol.evaluation.checkpoint_comparison import _checked_decode
from latentmol.sampling import sample_hard_graph


def _summarize(counts, outcomes, capacity):
    full = counts.count(capacity)
    empty = counts.count(0)
    valid = [result.smiles for result in outcomes if result.smiles is not None]
    failures = Counter(result.failure_reason for result in outcomes if result.smiles is None)
    return dict(prior_points=len(counts), prior_at_capacity=full, prior_at_capacity_fraction=full / len(counts),
                prior_empty=empty, prior_empty_fraction=empty / len(counts),
                prior_hard_count_mean=mean(counts), prior_hard_count_median=median(counts),
                prior_size_histogram={str(k): v for k, v in sorted(Counter(counts).items())},
                prior_valid=len(valid), prior_valid_fraction=len(valid) / len(counts),
                prior_unique_valid=len(set(valid)), prior_failures=dict(failures))


class CompactPriorPanel:
    """Argmax and temperature-one expected counts, with raw RDKit validity."""

    def __init__(self, latent_dim, *, seed, count, batch_size, stress_radius_multiplier=None):
        if any(type(n) is not int or n < 1 for n in (latent_dim, count, batch_size)):
            raise ValueError("Require positive integer prior panel dimensions")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Require a nonnegative signed 64-bit prior panel seed")
        if stress_radius_multiplier is not None and (type(stress_radius_multiplier) not in (int, float)
                or not math.isfinite(stress_radius_multiplier) or stress_radius_multiplier <= 1):
            raise ValueError("Require a finite stress_radius_multiplier greater than one")
        self.codes = torch.randn(count, latent_dim, dtype=torch.float32, device="cpu",
                                 generator=torch.Generator(device="cpu").manual_seed(seed))
        self.batch_size = batch_size
        self.stress_codes = None
        self.metadata = dict(seed=seed, count=count, batch_size=batch_size, latent_dim=latent_dim,
                             distribution="standard_normal", temperature=1.0,
                             categorical_seed=(seed + 1) % 2**63,
                             codes_sha256=hashlib.sha256(self.codes.numpy().tobytes()).hexdigest())
        if stress_radius_multiplier is not None:
            self.stress_codes = self.codes * stress_radius_multiplier
            self.metadata.update(stress_radius_multiplier=stress_radius_multiplier,
                                 stress_codes_sha256=hashlib.sha256(self.stress_codes.numpy().tobytes()).hexdigest())

    def observe(self, model):
        """Count every attempted graph, including empty or unsanitizable outputs."""
        if self.codes.shape[1] != model.config.latent:
            raise ValueError("Fixed prior panel and model differ in latent dimension")
        result = self._observe_codes(model, self.codes)
        if self.stress_codes is not None:
            result["stress_panel"] = dict(radius_multiplier=self.metadata["stress_radius_multiplier"],
                                          **self._observe_codes(model, self.stress_codes))
        return result

    def _observe_codes(self, model, panel_codes):
        counts = {"argmax": [], "categorical": []}
        outcomes = {"argmax": [], "categorical": []}
        soft_counts = []
        generator = torch.Generator(device="cpu").manual_seed(self.metadata["categorical_seed"])
        with preserve_observation_state(model, {}):
            for codes in panel_codes.split(self.batch_size):
                decoded = _checked_decode(model, codes)
                # The full atom-vocabulary argmax used by generation differs from
                # thresholding the probability of any real element at one half.
                soft_counts.extend((1 - decoded.atoms.softmax(-1)[..., 0]).sum(-1).tolist())
                graphs = dict(argmax=decoded.harden(model.spec),
                              categorical=sample_hard_graph(decoded, model.spec, generator=generator))
                for method, graph in graphs.items():
                    counts[method].extend((graph.atoms != 0).sum(-1).tolist())
                    outcomes[method].extend(inspect_graph(a, c, b, model.spec)
                                            for a, c, b in zip(graph.atoms, graph.charges, graph.bonds, strict=True))
        summaries = {key: _summarize(counts[key], outcomes[key], model.spec.max_nodes) for key in counts}
        return {**summaries["argmax"], "prior_soft_count_mean": mean(soft_counts),
                "categorical": summaries["categorical"]}
