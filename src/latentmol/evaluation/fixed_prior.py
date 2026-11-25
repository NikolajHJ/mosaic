"""Small paired prior-validity observations during training."""

from collections import Counter

import torch

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.chemistry_early_health import preserve_observation_state
from latentmol.evaluation.checkpoint_comparison import _checked_decode, paired_difference
from latentmol.sampling import sample_hard_graph


class FixedPriorProbe:
    """Reuse CPU latents and categorical RNG draws across arms and updates."""

    def __init__(self, latent_dim, *, count, latent_seed, categorical_seed, batch_size=64):
        for name, value in (("latent_dim", latent_dim), ("count", count), ("batch_size", batch_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"Require positive integer {name}")
        for name, value in (("latent_seed", latent_seed), ("categorical_seed", categorical_seed)):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f"Require a nonnegative integer {name} within signed 64-bit range")
        self._settings = dict(latent_dim=latent_dim, count=count, latent_seed=latent_seed,
                              categorical_seed=categorical_seed, batch_size=batch_size,
                              temperature=1., validity_rule="raw_chemical_valid_before_schema_roundtrip")
        self._codes = torch.randn(count, latent_dim, dtype=torch.float32,
                                  generator=torch.Generator().manual_seed(latent_seed))

    def observe(self, model):
        """Return all-attempt chemistry summaries and paired Boolean outcomes."""
        if model.config.latent != self._settings["latent_dim"]:
            raise ValueError("Fixed prior panel latent dimension differs from the model")
        generator = torch.Generator().manual_seed(self._settings["categorical_seed"])
        outcomes = {"argmax": [], "categorical": []}
        with preserve_observation_state(model, {}):
            for codes in self._codes.split(self._settings["batch_size"]):
                decoded = _checked_decode(model, codes)
                graphs = dict(argmax=decoded.harden(model.spec), categorical=sample_hard_graph(
                    decoded, model.spec, generator=generator))
                for method, graph in graphs.items():
                    for atoms, charges, bonds in zip(graph.atoms, graph.charges, graph.bonds, strict=True):
                        result = classify_repair_graph(atoms, charges, bonds, model.spec)
                        if result["failure_kind"] == "input":
                            raise ValueError(f"Incompatible decoded graph in fixed prior probe: {result['reason']}")
                        outcomes[method].append(result)

        result = {"panel": dict(self._settings, graph_spec=model.spec.to_dict())}
        for method, rows in outcomes.items():
            valid = [row["raw_chemical_valid"] is True for row in rows]
            strict = [row["strict_teacher_valid"] is True for row in rows]
            result[method] = dict(attempts=len(rows), valid_count=sum(valid), validity=sum(valid) / len(rows),
                strict_valid_count=sum(strict), strict_validity=sum(strict) / len(rows),
                valid_by_sample=valid, strict_valid_by_sample=strict,
                failure_kinds=dict(Counter(row["failure_kind"] for row in rows)),
                failure_reasons=dict(Counter(f"{row['failure_kind']}:{row['reason']}" for row in rows
                                             if row["failure_kind"] != "none")))
        return result


def compare_fixed_prior(before, after):
    """Compare the same latent and draw positions at two observations."""
    if before["panel"] != after["panel"]:
        raise ValueError("Fixed prior comparisons require identical panel definitions")
    return {method: paired_difference(before[method]["valid_by_sample"], after[method]["valid_by_sample"])
            for method in ("argmax", "categorical")}
