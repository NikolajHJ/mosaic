"""Compare raw paired samples from the two matched ring-size training arms."""

import math

import numpy as np

from latentmol.guidance.ring_size import ring_size_score
from latentmol.guidance.ring_size_reference import COUNTS_BY_HAC


def _rates(scores):
    rows = [(n if v else 0, cy, tri, large) for n, v, cy, tri, large in
            zip(*(scores[k] for k in ("occupied_atoms", "valid", "any_ring", "triangle", "ring8plus")), strict=True)]
    _, _, stats = ring_size_score(np.asarray(rows, dtype=int))
    return {key: stats[key] for key in ("ring_reference_supported_valid", "ring3_observed_rate",
            "ring8plus_observed_rate", "ring3_reference_rate", "ring8plus_reference_rate",
            "ring3_excess", "ring8plus_excess")}


def ring_decision(control, candidate, prior_count, recon_count):
    """Choose using paired topology gains and all-draw validity and diversity safeguards."""
    for scores in (control, candidate):
        for key in ("valid", "aromatic", "amide", "any_ring", "triangle", "ring8plus", "occupied_atoms", "scaffold", "exact"):
            if len(scores[key]) != (recon_count if key == "exact" else prior_count):
                raise ValueError(f"Incomplete ring-comparison panel for {key}")
    rates = [_rates(x) for x in (control, candidate)]
    supported = all(r["ring_reference_supported_valid"] > 0 for r in rates)
    directions = [key for key, excess in (("triangle", "ring3_excess"), ("ring8plus", "ring8plus_excess"))
                  if supported and rates[0][excess] > 0]
    # Compare topology on the same latents at the same hard atom count. A switch
    # to an invalid or smaller graph cannot contribute a paired topology gain.
    changes = []
    for i, atoms in enumerate(control["occupied_atoms"]):
        if (control["valid"][i] and candidate["valid"][i] and control["any_ring"][i] and candidate["any_ring"][i]
                and atoms == candidate["occupied_atoms"][i]
                and 0 < atoms < len(COUNTS_BY_HAC) and COUNTS_BY_HAC[atoms][0]):
            changes.append(sum(int(control[key][i]) - int(candidate[key][i]) for key in directions))
    net = sum(changes)
    se = (math.sqrt(len(changes) * sum((d - net / len(changes))**2 for d in changes) / (len(changes) - 1))
          if len(changes) >= 2 else None)
    counts = {}
    for key in ("valid", "aromatic", "amide", "any_ring", "exact"):
        left, right = control[key], candidate[key]
        counts[key] = dict(n=len(left), control=sum(left), candidate=sum(right),
                           gains=sum(b and not a for a, b in zip(left, right)),
                           losses=sum(a and not b for a, b in zip(left, right)))
    scaffolds = [len(set(x["scaffold"]) - {"", None}) for x in (control, candidate)]
    counts.update(scaffolds=dict(control=scaffolds[0], candidate=scaffolds[1]),
                  ring_rates=dict(control=rates[0], candidate=rates[1]),
                  topology=dict(common_valid_cyclic_same_size=len(changes), metrics=directions, net=net, paired_se=se))
    checks = dict(
        each_size_matched_excess_nonincreasing=supported and all(rates[1][k] <= rates[0][k]
            for k in ("ring3_excess", "ring8plus_excess")),
        paired_topology_improves=se is not None and net >= 4 and net > se,
        scaffold_yield_preserved=scaffolds[1] >= scaffolds[0] - .01 * prior_count)
    for key, tolerance in (("valid", .01 * prior_count), ("any_ring", .01 * prior_count),
                           ("aromatic", .0025 * prior_count), ("amide", .005 * prior_count), ("exact", 2)):
        checks[key + "_preserved"] = counts[key]["candidate"] >= counts[key]["control"] - tolerance
    return ("candidate" if all(checks.values()) else "control"), counts, checks
