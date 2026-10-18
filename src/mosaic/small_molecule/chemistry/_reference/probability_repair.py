"""Detached reference extraction; see source_manifest.json. Do not edit manually."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import math
import numpy as np
from .quality import graph_identity


@dataclass(frozen=True)
class ProbabilityRepairConfig:
    """Explicit shared budget; coefficients are declared, never fitted."""

    max_edits: int = 3
    checks_per_depth: int = 256
    beam_width: int = 4
    probability_weight: float = 1.0
    probability_floor: float = 1e-12

    def __post_init__(self):
        for name in ('max_edits', 'checks_per_depth', 'beam_width'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Search budgets must be positive integers')
        if (not math.isfinite(self.probability_weight) or self.probability_weight < 0
                or not math.isfinite(self.probability_floor)
                or not 0 < self.probability_floor < 1):
            raise ValueError('Require nonnegative probability weight and floor in (0,1)')

    def policy(self):
        return {'version': 'probability-repair-v1', **asdict(self),
                'occupancy': 'fixed', 'alignment': 'shared_slots',
                'cost': 'edit_count + probability_weight * sum(max(0,log(p_old/p_new)))',
                'probability_semantics': 'conditional categorical marginals at one fixed latent',
                'priority': 'problem locality then probability regret; beam by problem burden then cost',
                'minimum_is_proven': False, 'transform': '1/(1+best_observed_weighted_cost)'}


def _array(value, *, integer=False):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    if integer and a.dtype.kind not in 'iu':
        raise ValueError('Graph labels must be integers')
    return np.array(a, dtype=np.int64 if integer else np.float64, copy=True)


def validate_probabilities(probabilities, spec):
    """Accept actual shared-graph categorical probabilities, never raw logits."""
    p = tuple(_array(x) for x in probabilities)
    n = spec.max_nodes
    shapes = ((n, len(spec.atomic_numbers)), (n, len(spec.formal_charges)),
              (n, n, len(spec.bond_names)))
    if len(p) != 3 or any(a.shape != shape for a, shape in zip(p, shapes)):
        raise ValueError('Probability shapes must match the shared graph schema')
    if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()
           or not np.allclose(a.sum(-1), 1, rtol=1e-5, atol=1e-6) for a in p):
        raise ValueError('Require finite normalized categorical probabilities')
    if not np.allclose(p[2], p[2].transpose(1, 0, 2), rtol=1e-5, atol=1e-6):
        raise ValueError('Unordered bonds require symmetric probabilities')
    return p


def probability_identity(probabilities, spec):
    """Exact probability identity prevents cache reuse across different latents."""
    p = validate_probabilities(probabilities, spec)
    h = hashlib.sha256(spec.fingerprint().encode())
    for a in p:
        h.update(np.asarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def measure_probability_edits(raw, repaired, probabilities, spec, config=None):
    """Explain every witness edit with both its old and replacement probability.

    Bonds count once using i<j. Probabilities are NOT renormalized among edited
    alternatives. NO_ATOM and padding remain fixed. Clipping is disclosed per
    edit; zero replacement support has zero replacement mass (an unclipped
    ratio is infinite for positive old support, undefined when both are zero).
    """
    config = config or ProbabilityRepairConfig()
    left, right = tuple(map(lambda a: _array(a, integer=True), raw)), tuple(map(lambda a: _array(a, integer=True), repaired))
    p = validate_probabilities(probabilities, spec)
    # Reuse the raw graph validator instead of silently canonicalizing bad labels.
    for graph in (left, right):
        graph_identity(*graph, spec)
        if any(x.shape != shape for x, shape in zip(graph, ((spec.max_nodes,), (spec.max_nodes,), (spec.max_nodes, spec.max_nodes)))):
            raise ValueError('Require fully padded shared graph labels')
        if not np.array_equal(graph[2], graph[2].T) or np.any(np.diag(graph[2])):
            raise ValueError('Require symmetric bonds without self loops')
    active = left[0] != 0
    if not np.array_equal(active, right[0] != 0):
        raise ValueError('Probability repair requires unchanged atom occupancy')
    if (np.any(left[1][~active] != right[1][~active])
            or np.any(left[2][~(active[:, None] & active[None, :])] != right[2][~(active[:, None] & active[None, :])])):
        raise ValueError('Repair cannot edit padding charge/bonds')
    changes = []
    for field, name in enumerate(('atom', 'charge', 'bond')):
        for position in np.argwhere(left[field] != right[field]):
            slot = tuple(int(x) for x in position)
            if field == 2 and slot[0] >= slot[1]:
                continue
            old, new = int(left[field][slot]), int(right[field][slot])
            old_p, new_p = float(p[field][slot + (old,)]), float(p[field][slot + (new,)])
            regret = max(0.0, math.log(max(old_p, config.probability_floor)) - math.log(max(new_p, config.probability_floor)))
            changes.append({'field': name, 'slots': list(slot), 'old_label': old, 'new_label': new,
                            'old_probability': old_p, 'new_probability': new_p,
                            'probability_regret_nats': regret,
                            'weighted_cost': 1 + config.probability_weight * regret,
                            'probability_clipped': old_p < config.probability_floor or new_p < config.probability_floor,
                            'replacement_has_zero_support': new_p == 0})
    regret = sum(x['probability_regret_nats'] for x in changes)
    zero = any(x['replacement_has_zero_support'] for x in changes)
    surprisal = sum(-math.log(max(x['new_probability'], config.probability_floor)) for x in changes)
    true_log_mass = None if zero else sum(math.log(x['new_probability']) for x in changes)
    return {'edit_count': len(changes), 'edit_counts': {name: sum(x['field'] == name for x in changes) for name in ('atom', 'charge', 'bond')},
            'probability_regret_nats': regret, 'weighted_cost': len(changes) + config.probability_weight * regret,
            'replacement_surprisal_nats_clipped': surprisal, 'replacement_log_mass': true_log_mass,
            'replacement_mass': 0.0 if zero else math.exp(true_log_mass),
            'zero_support_changes': sum(x['replacement_has_zero_support'] for x in changes),
            'clipped_changes': sum(x['probability_clipped'] for x in changes),
            'changes': changes, 'occupancy_edits': 0,
            'mass_interpretation': 'Factorized mass of changed labels only; not full-molecule or validity probability'}
