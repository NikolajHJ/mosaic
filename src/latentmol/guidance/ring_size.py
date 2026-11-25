"""Batch ring-size calibration on valid Gaussian-prior decodes.

The target is the real probability of at least one ring of the given size,
matched to the heavy-atom counts of the valid draws. Gradients redistribute
credit *within* the observed valid cyclic graphs at fixed heavy-atom count.
"""

import math

import numpy as np

from latentmol.guidance.ring_size_reference import COUNTS_BY_HAC, REFERENCE_ID, TRAINING_VIEW_SHA256


def validate_ring_policy(policy, dataset_sha256=None):
    if (not isinstance(policy, dict) or set(policy) != {"weight", "reference"}
            or policy["reference"] != REFERENCE_ID):
        raise ValueError("Unknown ring-size reference or policy fields")
    weight = policy["weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("Ring-size weight must be finite and nonnegative")
    if dataset_sha256 is not None and dataset_sha256 != TRAINING_VIEW_SHA256:
        raise ValueError("Ring-size reference was measured on a different training view")


def ring_size_score(rows, reference=COUNTS_BY_HAC):
    """Return per-draw score coefficients, an observed penalty and diagnostics.

    Rows contain (HAC, any-ring, any triangle, any ring >=8). Invalid draws
    have HAC zero. Rates include acyclic valid molecules; unobserved reference
    sizes are omitted. Each component is half the squared positive gap.

    For the surrogate, freeze each empirical gap and the mass of valid, cyclic
    draws in each size stratum. A hit minus the mean of the other cyclic rows
    at that size then estimates the centered conditional score. Averaging the
    two components gives the coefficients of grad log q(sample). This plug-in
    gradient targets composition under fixed stratum masses. It is not the
    exact derivative of the random finite-batch penalty. Acyclic and invalid
    outcomes receive zero direct credit; shared model parameters can still
    alter their frequencies, so track both separately.
    """
    sizes = rows[:, 0]
    cyclic = rows[:, 1].astype(bool)
    hits = rows[:, 2:]
    ref = np.asarray(reference)
    in_range = (sizes > 0) & (sizes < len(ref))
    supported = in_range.copy()
    supported[in_range] = ref[sizes[in_range], 0] > 0
    n = int(supported.sum())
    weights = np.zeros(len(rows), dtype=np.float64)
    stats = dict(ring_reference_supported_valid=n, ring_reference_unsupported_valid=int((sizes > 0).sum()) - n,
                 ring_any_hits=int(cyclic.sum()), ring_matched_cyclic_rows=0)
    counts = hits[supported].sum(axis=0)
    observed = np.zeros(2)
    expected = np.zeros(2)
    if n:
        observed = counts / n
        target = ref[sizes[supported]]
        expected = (target[:, 1:] / target[:, :1]).mean(axis=0)
    excess = np.maximum(observed - expected, 0.)
    eligible = supported & cyclic
    for size in np.unique(sizes[eligible]):
        mask = eligible & (sizes == size)
        group_n = int(mask.sum())
        if group_n < 2:
            continue
        values = hits[mask]
        leave_one_out = (values.sum(axis=0) - values) / (group_n - 1)
        weights[mask] = ((values - leave_one_out) * excess).mean(axis=1) / n
        stats["ring_matched_cyclic_rows"] += group_n
    for i, name in enumerate(("ring3", "ring8plus")):
        stats[f"{name}_hits"] = int(counts[i])
        stats[f"{name}_observed_rate"] = float(observed[i]) if n else None
        stats[f"{name}_reference_rate"] = float(expected[i]) if n else None
        stats[f"{name}_excess"] = float(excess[i]) if n else None
    stats["ring_size_informative_rows"] = int(np.count_nonzero(weights))
    penalty = float(.5 * np.mean(excess**2))
    return weights, penalty, stats
