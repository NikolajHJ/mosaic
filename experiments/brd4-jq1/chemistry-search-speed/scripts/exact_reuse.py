"""Experiment-only exact native-fact reuse and probability-cost validation reuse.

This adapter never caches a repair target across changing probabilities. Native
facts are keyed by complete slot-ordered integer arrays and the schema. The cost
adapter keys its immutable validation context by all raw labels, every float64
probability byte, schema and cost configuration. Candidate label, structural,
occupancy and padding checks still execute on every cost call.

The context changes process-global module references temporarily. It is intended
for one benchmark at a time, not concurrent preparations or production adoption.
"""

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import math

import numpy as np


_ACTIVE = False


def _array_key(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError("Unsupported cache-key dtype")
    return array.dtype.str, array.shape, array.tobytes(order="C")


def _graph_key(graph, spec):
    return spec.fingerprint(), tuple(_array_key(value) for value in graph)


class _LRU:
    def __init__(self, maximum, stats, prefix):
        self.maximum = maximum
        self.stats = stats
        self.prefix = prefix
        self.values = OrderedDict()

    def lookup(self, key):
        if key not in self.values:
            return False, None
        value = self.values.pop(key)
        self.values[key] = value
        return True, value

    def put(self, key, value):
        self.values[key] = value
        if len(self.values) > self.maximum:
            self.values.popitem(last=False)
            self.stats[self.prefix + "_evictions"] += 1
        self.stats[self.prefix + "_entries"] = len(self.values)


def _cost_function(reference, quality, cache, stats):
    original = reference.measure_probability_edits

    def measure(raw, repaired, probabilities, spec, config=None):
        stats["cost_calls"] += 1
        config = config or reference.ProbabilityRepairConfig()
        # Preserve historical behavior for duck-typed/custom configuration
        # objects instead of broadening this optimization's applicability.
        if type(config) is not reference.ProbabilityRepairConfig:
            stats["cost_passthrough"] += 1
            return original(raw, repaired, probabilities, spec, config)
        left = tuple(reference._array(value, integer=True) for value in raw)
        right = tuple(reference._array(value, integer=True) for value in repaired)
        unvalidated = tuple(reference._array(value) for value in probabilities)
        key = (
            _graph_key(left, spec),
            tuple(_array_key(value) for value in unvalidated),
            config.max_edits,
            config.checks_per_depth,
            config.beam_width,
            config.probability_weight,
            config.probability_floor,
        )
        found, context = cache.lookup(key)
        if found:
            stats["cost_invariant_hits"] += 1
            p, edit_terms = context
        else:
            # Same validation and order as the reference: probabilities before
            # graph identity, then exact padded-shape/symmetry constraints.
            p = reference.validate_probabilities(unvalidated, spec)
            reference.graph_identity(*left, spec)
            _validate_graph(left, spec, quality)
            edit_terms = {}
            cache.put(key, (p, edit_terms))
            stats["cost_validation_count"] += 1

        # The reference hashes each candidate only to validate labels. Preserve
        # all those validations while omitting the discarded JSON/hash result.
        _validate_graph(right, spec, quality)
        active = left[0] != 0
        if not np.array_equal(active, right[0] != 0):
            raise ValueError("Probability repair requires unchanged atom occupancy")
        if (
            np.any(left[1][~active] != right[1][~active])
            or np.any(
                left[2][~(active[:, None] & active[None, :])]
                != right[2][~(active[:, None] & active[None, :])]
            )
        ):
            raise ValueError("Repair cannot edit padding charge/bonds")

        # np.argwhere ordering, Python math.log, summation order and the final
        # len+weight*sum formula intentionally match the reference exactly.
        changes = []
        for field, name in enumerate(("atom", "charge", "bond")):
            for position in np.argwhere(left[field] != right[field]):
                slot = tuple(int(value) for value in position)
                if field == 2 and slot[0] >= slot[1]:
                    continue
                old, new = int(left[field][slot]), int(right[field][slot])
                term_key = field, slot, old, new
                term = edit_terms.get(term_key)
                if term is None:
                    old_p = float(p[field][slot + (old,)])
                    new_p = float(p[field][slot + (new,)])
                    regret = max(
                        0.0,
                        math.log(max(old_p, config.probability_floor))
                        - math.log(max(new_p, config.probability_floor)),
                    )
                    term = {
                        "field": name,
                        "slots": list(slot),
                        "old_label": old,
                        "new_label": new,
                        "old_probability": old_p,
                        "new_probability": new_p,
                        "probability_regret_nats": regret,
                        "weighted_cost": 1 + config.probability_weight * regret,
                        "probability_clipped": old_p < config.probability_floor
                        or new_p < config.probability_floor,
                        "replacement_has_zero_support": new_p == 0,
                    }
                    edit_terms[term_key] = term
                    stats["cost_edit_term_misses"] += 1
                else:
                    stats["cost_edit_term_hits"] += 1
                # Never expose mutable cached dictionaries through a witness.
                changes.append(deepcopy(term))
        regret = sum(value["probability_regret_nats"] for value in changes)
        zero = any(value["replacement_has_zero_support"] for value in changes)
        surprisal = sum(
            -math.log(max(value["new_probability"], config.probability_floor))
            for value in changes
        )
        true_log_mass = None if zero else sum(
            math.log(value["new_probability"]) for value in changes
        )
        return {
            "edit_count": len(changes),
            "edit_counts": {
                name: sum(value["field"] == name for value in changes)
                for name in ("atom", "charge", "bond")
            },
            "probability_regret_nats": regret,
            "weighted_cost": len(changes) + config.probability_weight * regret,
            "replacement_surprisal_nats_clipped": surprisal,
            "replacement_log_mass": true_log_mass,
            "replacement_mass": 0.0 if zero else math.exp(true_log_mass),
            "zero_support_changes": sum(
                value["replacement_has_zero_support"] for value in changes
            ),
            "clipped_changes": sum(value["probability_clipped"] for value in changes),
            "changes": changes,
            "occupancy_edits": 0,
            "mass_interpretation": "Factorized mass of changed labels only; not full-molecule or validity probability",
        }

    return measure


def _validate_graph(graph, spec, quality):
    # graph_identity calls _labels for validation, then hashes copies. Hashing
    # contributes no constraint; _labels retains categorical/padding semantics.
    quality._labels(*graph, spec)
    expected = (
        (spec.max_nodes,),
        (spec.max_nodes,),
        (spec.max_nodes, spec.max_nodes),
    )
    if any(value.shape != shape for value, shape in zip(graph, expected)):
        raise ValueError("Require fully padded shared graph labels")
    if not np.array_equal(graph[2], graph[2].T) or np.any(np.diag(graph[2])):
        raise ValueError("Require symmetric bonds without self loops")


@contextmanager
def install_reuse(mode="combined", max_entries=8192):
    """Temporarily install exact adapters and yield mutable accounting counters.

    ``cache`` reuses native hard-graph facts only. ``cost`` reuses exact input
    validation and per-edit costs only. ``combined`` enables both. Each cache
    has an independent finite entry limit; eviction only recomputes facts.
    Exceptions are never cached. Original module references always restore.
    """
    global _ACTIVE
    if mode not in ("cache", "cost", "combined"):
        raise ValueError("Unknown exact-reuse mode")
    if type(max_entries) is not int or not 1 <= max_entries <= 65536:
        raise ValueError("Require a bounded positive integer cache size")
    if _ACTIVE:
        raise RuntimeError("Exact reuse does not support nested installations")

    from mosaic.small_molecule.chemistry import preparation
    from mosaic.small_molecule.chemistry._reference import (
        atom_preserving_chemistry as base,
        chemistry,
        probability_repair as probability,
        quality,
        repair_progress as progress,
    )

    stats = {
        key: 0
        for key in (
            "cache_hits", "cache_misses", "cache_evictions", "cache_entries",
            "diagnose_hits", "diagnose_misses", "inspect_hits", "inspect_misses",
            "encode_hits", "encode_misses", "cache_passthrough",
            "cost_calls", "cost_validation_count", "cost_invariant_hits",
            "cost_evictions", "cost_entries", "cost_passthrough",
            "cost_edit_term_hits", "cost_edit_term_misses",
        )
    }
    native_cache = _LRU(max_entries, stats, "cache")
    cost_cache = _LRU(max_entries, stats, "cost")
    originals = []

    def patch(module, name, value):
        originals.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def memo(function, name):
        def wrapped(*args, **kwargs):
            if kwargs:
                # The production sites use positional arguments. Preserve the
                # entire original call/error contract for other callers.
                stats["cache_passthrough"] += 1
                return function(*args, **kwargs)
            try:
                if name == "encode":
                    smiles, spec = args
                    if not isinstance(smiles, str):
                        raise TypeError("Uncacheable SMILES argument")
                    key = name, spec.fingerprint(), smiles
                else:
                    atoms, charges, bonds, spec = args
                    key = name, _graph_key((atoms, charges, bonds), spec)
            except (ValueError, TypeError, IndexError, AttributeError):
                stats["cache_passthrough"] += 1
                return function(*args)
            found, value = native_cache.lookup(key)
            if found:
                stats["cache_hits"] += 1
                stats[name + "_hits"] += 1
                return deepcopy(value)
            stats["cache_misses"] += 1
            stats[name + "_misses"] += 1
            value = function(*args)
            native_cache.put(key, deepcopy(value))
            return deepcopy(value)

        return wrapped

    _ACTIVE = True
    try:
        if mode in ("cache", "combined"):
            inspected = memo(chemistry.inspect_graph, "inspect")
            diagnosed = memo(quality.diagnose_graph, "diagnose")
            encoded = memo(chemistry.encode_smiles, "encode")
            for module in (chemistry, quality, progress, preparation):
                patch(module, "inspect_graph", inspected)
            for module in (quality, base, progress):
                patch(module, "diagnose_graph", diagnosed)
            for module in (chemistry, base, progress, preparation):
                patch(module, "encode_smiles", encoded)
        if mode in ("cost", "combined"):
            measured = _cost_function(probability, quality, cost_cache, stats)
            for module in (probability, base, progress):
                patch(module, "measure_probability_edits", measured)
        yield stats
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)
        _ACTIVE = False
