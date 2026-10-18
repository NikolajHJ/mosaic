"""Probability-aware distance to observed valid repairs, outside autograd.

This bounded search keeps atom occupancy and slot identity fixed. One atom,
charge, or unordered bond-label change costs one edit plus its nonnegative
decoder log-probability regret. It measures a verified *witness*, not a proven
globally closest molecule or the probability of chemical validity. The raw
graph remains the only molecule passed to external scoring branches.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .quality import diagnose_graph, graph_identity
from .repair_search import round_robin_candidates, single_label_edits


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


def search_probability_repair(raw, probabilities, spec, config=None):
    """Search past the first valid graph; retain the best verified witnesses.

    The unit-edit and weighted optima may be different molecules. Both are
    explicitly best observed upper bounds. Capped checks and a pruned beam
    cannot establish a global minimum, even when a one-edit witness exists.
    """
    config = config or ProbabilityRepairConfig()
    original = tuple(_array(a, integer=True) for a in raw)
    p = validate_probabilities(probabilities, spec)
    diagnostic = diagnose_graph(*original, spec)
    result = {'policy': config.policy(), 'raw_graph_id': diagnostic.get('graph_id'),
              'probability_sha256': probability_identity(p, spec), 'raw_valid': diagnostic['valid'],
              'checks': 0, 'checks_by_depth': [], 'valid_witnesses_observed': 0,
              'best_observed_edit_count': None, 'best_observed_weighted_cost': None,
              'weighted_proximity': None, 'minimum_edit_witness': None, 'weighted_witness': None,
              'distance_is_exact': False, 'distance_lower_bound': 0 if diagnostic['valid'] else 1}
    if (not diagnostic.get('schema_supported') or not diagnostic.get('heavy_atoms')
            or not diagnostic.get('categorical_support_labels_ok')):
        return {**result, 'status': 'unsupported', 'reason': 'Requires nonempty canonical hard-label support'}
    if diagnostic['valid']:
        return {**result, 'status': 'already_valid', 'best_observed_edit_count': 0,
                'best_observed_weighted_cost': 0.0, 'weighted_proximity': 1.0, 'distance_is_exact': True}
    # Validate the no-op pair before exploring; includes symmetric/no-padding edits.
    measure_probability_edits(original, original, p, spec, config)
    seen = {tuple(a.tobytes() for a in original)}
    beam = [(original, diagnostic)]
    best_unit = best_weighted = None
    sequence = 0
    for depth in range(1, config.max_edits + 1):
        frontier, checked, capped = [], 0, False
        def rank_operation(op):
            priority, field, i, j, new = op
            slot = (i, j) if field == 2 else (i,)
            old = int(original[field][slot])
            regret = max(0.0, math.log(max(float(p[field][slot + (old,)]), config.probability_floor))
                         - math.log(max(float(p[field][slot + (new,)]), config.probability_floor)))
            return (priority, regret, field, i, j, new)
        streams = [single_label_edits(graph, original, spec, diag, operation_rank=rank_operation)
                   for graph, diag in beam]
        for candidate in round_robin_candidates(streams):
            key = tuple(a.tobytes() for a in candidate)
            if key in seen:
                continue
            if checked >= config.checks_per_depth:
                capped = True
                break
            seen.add(key)
            candidate_diag = diagnose_graph(*candidate, spec)
            cost = measure_probability_edits(original, candidate, p, spec, config)
            checked += 1
            result['checks'] += 1
            if cost['edit_count'] != depth:
                raise RuntimeError('Edit generation changed a categorical label twice')
            if candidate_diag['valid']:
                result['valid_witnesses_observed'] += 1
                witness = {**cost, 'graph_id': candidate_diag['graph_id'], 'smiles': candidate_diag['smiles'],
                           'labels': {name: a.tolist() for name, a in zip(('atoms', 'charges', 'bonds'), candidate)}}
                if best_unit is None or (cost['edit_count'], cost['weighted_cost']) < (best_unit['edit_count'], best_unit['weighted_cost']):
                    best_unit = witness
                if best_weighted is None or (cost['weighted_cost'], cost['edit_count']) < (best_weighted['weighted_cost'], best_weighted['edit_count']):
                    best_weighted = witness
                # Positive additional costs cannot improve a valid witness.
                continue
            sequence += 1
            burden = candidate_diag.get('problem_burden')
            burden = 1000000 if burden is None else burden
            rank = (burden + max(0, candidate_diag.get('component_count', 1) - 1), cost['weighted_cost'], sequence)
            frontier.append((rank, candidate, candidate_diag))
            frontier.sort(key=lambda x: x[0])
            del frontier[config.beam_width:]
        result['checks_by_depth'].append({'depth': depth, 'checks': checked, 'capped': capped})
        if depth == 1 and not capped and best_unit is None:
            result['distance_lower_bound'] = 2
        beam = [(item[1], item[2]) for item in frontier]
        if not beam:
            break
    if best_weighted is None:
        return {**result, 'status': 'censored', 'reason': 'No verified witness within the declared budget'}
    return {**result, 'status': 'witness_found', 'best_observed_edit_count': best_unit['edit_count'],
            'best_observed_weighted_cost': best_weighted['weighted_cost'],
            'weighted_proximity': 1 / (1 + best_weighted['weighted_cost']),
            'minimum_edit_witness': best_unit, 'weighted_witness': best_weighted}


class ProbabilityRepairEvaluator:
    """Bounded reusable evaluator keyed by graph, exact probabilities and policy."""

    def __init__(self, spec, config=None, *, cache_capacity=128):
        if type(cache_capacity) is not int or cache_capacity < 1:
            raise ValueError('Cache capacity must be positive')
        self.spec, self.config = spec, config or ProbabilityRepairConfig()
        self.cache_capacity, self.cache = cache_capacity, OrderedDict()
        self.checks = self.cache_hits = 0

    def diagnose(self, atoms, charges, bonds, *, probabilities):
        key = (graph_identity(atoms, charges, bonds, self.spec), probability_identity(probabilities, self.spec),
               json.dumps(self.config.policy(), sort_keys=True))
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return json.loads(json.dumps(self.cache[key]))
        result = search_probability_repair((atoms, charges, bonds), probabilities, self.spec, self.config)
        self.checks += result['checks']
        self.cache[key] = result
        if len(self.cache) > self.cache_capacity:
            self.cache.popitem(last=False)
        return json.loads(json.dumps(result))
