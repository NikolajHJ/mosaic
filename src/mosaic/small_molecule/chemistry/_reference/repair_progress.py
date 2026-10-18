"""Experimental detached search escalation and explicitly invalid partial targets.

This module is NOT wired into the sealed source121 campaign. Native chemistry
checks run only in target preparation, never in a differentiable forward pass.
An observed valid witness is distinct from a graph with verified partial
diagnostic progress. Neither is a proven nearest-valid repair.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from .atom_preserving_chemistry import _bond_edits
from .chemistry import encode_smiles, inspect_graph
from .probability_repair import ProbabilityRepairConfig, measure_probability_edits, validate_probabilities
from .quality import diagnose_graph
from .repair_search import round_robin_candidates


@dataclass(frozen=True)
class ProgressSearchConfig:
    """Additional per-failure budget, separate from the original three edits.

    Atom changes are opt-in and receive an extra edit penalty. Charges and
    occupancy remain fixed. Partial targets never contain atom substitutions;
    a substitution may guide training only after a complete valid witness.
    The extra penalty reduces, but cannot eliminate, high-valence shortcuts.
    """
    version: str = "repair-progress-v1"
    max_edits: int = 6
    checks_per_depth: int = 32
    beam_width: int = 4
    probability_weight: float = 1.0
    allow_atom_edits: bool = False
    atom_edit_penalty: float = 3.0
    max_atom_edits: int = 1

    def __post_init__(self):
        if self.version != "repair-progress-v1" or type(self.allow_atom_edits) is not bool:
            raise ValueError("Require an explicit experimental progress policy")
        for name, maximum in (("max_edits", 12), ("checks_per_depth", 4096), ("beam_width", 128), ("max_atom_edits", 2)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Invalid bounded progress search setting: " + name)
        for name in ("probability_weight", "atom_edit_penalty"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid progress search coefficient: " + name)
        if self.atom_edit_penalty < 1:
            raise ValueError("Atom edits must have an additional positive cost")

    def record(self):
        return {**asdict(self), "occupancy": "fixed", "charges": "fixed",
                "partial_atom_edits": False, "minimum_is_proven": False,
                "probability_floor_in_detached_cost_only": 1e-12,
                "cost": "edit_count + probability_weight*sum(max(0,log(p_old/p_new))) + atom_edit_penalty*atom_edits",
                "extra_budget_scope": "repeat/escalate only failed original rows; previous work is not free",
                "validity": "connected native sanitization and canonical schema roundtrip"}


def diagnostic_vector(diagnostic):
    """Native observed problem types and fragments, not a calibrated distance."""
    if not diagnostic.get("schema_supported") or not diagnostic.get("categorical_support_labels_ok"):
        return None
    if not diagnostic.get("heavy_atoms") or diagnostic.get("problem_burden") is None:
        return None
    result = Counter()
    for problem in diagnostic.get("problems", []):
        result[str(problem["type"])] += max(1, len(set(problem.get("slots", []))))
    result["extra_components"] = max(0, diagnostic["component_count"] - 1)
    if not diagnostic.get("valid") and not any(result.values()):
        # A schema-roundtrip failure cannot be treated as diagnostic success.
        result["unresolved_native_failure"] = 1
    return dict(result)


def strictly_improves(before, after):
    """Require Pareto improvement: no new/worse observed failure category."""
    if before is None or after is None:
        return False
    keys = set(before) | set(after)
    return (all(after.get(k, 0) <= before.get(k, 0) for k in keys)
            and any(after.get(k, 0) < before.get(k, 0) for k in keys))


def _labels(graph):
    return tuple(tuple(int(x) for x in field) if index < 2 else
                 tuple(tuple(int(x) for x in row) for row in field)
                 for index, field in enumerate(graph))


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ProgressWitness:
    """Immutable, slot-aligned labels with an honest complete/partial tag."""
    kind: str
    original: tuple
    labels: tuple
    cost: float
    before: tuple
    after: tuple
    identity: str

    @classmethod
    def create(cls, kind, original, labels, cost, before, after):
        if kind not in ("valid", "partial"):
            raise ValueError("Unknown witness kind")
        old, new = _labels(original), _labels(labels)
        values = dict(kind=kind, original=old, labels=new, cost=float(cost),
                      before=tuple(sorted(before.items())), after=tuple(sorted(after.items())))
        return cls(**values, identity=_signature(values))

    def verify(self, spec):
        values = {key: getattr(self, key) for key in ("kind", "original", "labels", "cost", "before", "after")}
        if self.identity != _signature(values) or self.kind not in ("valid", "partial"):
            raise ValueError("Progress witness identity changed")
        if not math.isfinite(self.cost) or self.cost < 1:
            raise ValueError("Progress witness cost must be finite and positive")
        for vector in (self.before, self.after):
            if len(dict(vector)) != len(vector) or any(type(k) is not str or type(v) is not int or v < 0 for k, v in vector):
                raise ValueError("Progress diagnostic vector changed")
        for graph in (self.original, self.labels):
            a, q, b = graph
            if len(a) != spec.max_nodes or len(q) != len(a) or len(b) != len(a) or any(len(row) != len(a) for row in b):
                raise ValueError("Progress witness schema shape differs")
            for field, upper in ((a, len(spec.atomic_numbers)), (q, len(spec.formal_charges)),
                                 (tuple(x for row in b for x in row), len(spec.bond_names))):
                if any(type(x) is not int or not 0 <= x < upper for x in field):
                    raise ValueError("Progress witness categorical label differs")
            if any(b[i][i] or any(b[i][j] != b[j][i] for j in range(len(a))) for i in range(len(a))):
                raise ValueError("Progress witness bond symmetry differs")
            if any(b[i][j] for i in range(len(a)) for j in range(len(a)) if not a[i] or not a[j]):
                raise ValueError("Progress witness has padding bonds")
        if self.original[1] != self.labels[1] or any(bool(a) != bool(b) for a, b in zip(self.original[0], self.labels[0])):
            raise ValueError("Progress witness changed charge/occupancy")
        if self.kind == "partial" and (self.original[0] != self.labels[0]
                or not strictly_improves(dict(self.before), dict(self.after))):
            raise ValueError("Partial witness lacks atom-preserving native progress")
        if self.original == self.labels:
            raise ValueError("An unchanged invalid graph is not a repair target")


def _atom_edits(graph, original, spec, config):
    changed = np.count_nonzero(graph[0] != original[0])
    if changed >= config.max_atom_edits:
        return
    for slot in np.flatnonzero(original[0]):
        if graph[0][slot] != original[0][slot]:
            continue
        for label in range(1, len(spec.atomic_numbers)):
            if label != original[0][slot]:
                candidate = tuple(field.copy() for field in graph)
                candidate[0][slot] = label
                yield candidate


def search_progress_repair(raw, probabilities, spec, config=None):
    """Search a declared extra budget; keep verified partial progress if needed.

    This is a research implementation, not the optimized production teacher.
    Benchmark coverage AND wall time before using it in a training campaign.
    Full authoritative native checks are repeated before retaining a witness.
    """
    config = ProgressSearchConfig() if config is None else config
    if not isinstance(config, ProgressSearchConfig):
        raise ValueError("Require an explicit bounded progress search configuration")
    original = tuple(np.asarray(field).copy() for field in raw)
    p = validate_probabilities(probabilities, spec)
    cost_config = ProbabilityRepairConfig(max_edits=config.max_edits,
        checks_per_depth=config.checks_per_depth, beam_width=config.beam_width,
        probability_weight=config.probability_weight)
    measure_probability_edits(original, original, p, spec, cost_config)

    def diagnose(graph):
        value = diagnose_graph(*graph, spec)
        inspected = inspect_graph(*graph, spec)
        valid = inspected.smiles is not None
        if valid:
            try:
                valid = encode_smiles(inspected.smiles, spec)["smiles"] == inspected.smiles
            except ValueError:
                valid = False
        value = {**value, "valid": valid}
        return value, diagnostic_vector(value)

    initial, before = diagnose(original)
    result = dict(policy=config.record(), status="no_progress", checks=0,
                  before=before, witness=None, valid_witnesses=0, partial_witnesses=0,
                  source_probability_sha256=_signature([v.tolist() for v in p]))
    if initial["valid"]:
        return {**result, "status": "already_valid"}
    if before is None:
        return {**result, "status": "unsupported"}
    seen = {_labels(original)}
    beam = [(original, initial)]
    best_valid = best_partial = None
    for depth in range(1, config.max_edits + 1):
        streams = []
        for graph, diagnostic in beam:
            bonds = _bond_edits(graph, original, spec, diagnostic, p, cost_config)
            streams.append(round_robin_candidates([bonds, _atom_edits(graph, original, spec, config)])
                           if config.allow_atom_edits else bonds)
        frontier = []
        checked = 0
        for candidate in round_robin_candidates(streams):
            key = _labels(candidate)
            if key in seen:
                continue
            if checked >= config.checks_per_depth:
                break
            seen.add(key)
            checked += 1
            result["checks"] += 1
            diagnostic, after = diagnose(candidate)
            measured = measure_probability_edits(original, candidate, p, spec, cost_config)
            if measured["edit_count"] != depth or measured["edit_counts"]["charge"]:
                raise RuntimeError("Progress search violated the depth/charge invariant")
            cost = measured["weighted_cost"] + config.atom_edit_penalty * measured["edit_counts"]["atom"]
            if diagnostic["valid"]:
                result["valid_witnesses"] += 1
                if best_valid is None or cost < best_valid.cost:
                    best_valid = ProgressWitness.create("valid", original, candidate, cost, before, after)
                continue
            if (np.array_equal(original[0], candidate[0]) and strictly_improves(before, after)):
                result["partial_witnesses"] += 1
                key = (sum(after.values()), cost)
                if best_partial is None or key < (sum(dict(best_partial.after).values()), best_partial.cost):
                    best_partial = ProgressWitness.create("partial", original, candidate, cost, before, after)
            if after is not None:
                frontier.append(((sum(after.values()), cost, checked), candidate, diagnostic))
                frontier.sort(key=lambda value: value[0])
                del frontier[config.beam_width:]
        beam = [(item[1], item[2]) for item in frontier]
        if not beam:
            break
    witness = best_valid or best_partial
    if witness is not None:
        witness.verify(spec)
        result.update(status=witness.kind, witness=witness)
    return result
