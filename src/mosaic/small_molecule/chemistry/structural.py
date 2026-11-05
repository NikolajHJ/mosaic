"""Opt-in constructive, atom-preserving repair proposals outside autodiff.

Build a degree-cap-compatible connected backbone, then remove/downgrade excess
bonds without removing that backbone. Probability regret ranks edits. Every
accepted target independently passes the existing native/schema roundtrip.
This bounded heuristic neither proves nearest repair nor changes the loss.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from ._reference.chemistry import encode_smiles, inspect_graph
from ._reference.chemistry_regularization import ALLOWED_VALENCES
from ._reference.probability_repair import (
    measure_probability_edits, probability_identity, validate_probabilities,
)
from ._reference.quality import graph_identity


@dataclass(frozen=True)
class StructuralRepairConfig:
    version: str = "connected-backbone-repair-v1"
    max_candidates: int = 6
    max_bond_operations: int = 1024

    def __post_init__(self):
        if self.version != "connected-backbone-repair-v1":
            raise ValueError("Unknown structural repair version")
        for name, limit in (("max_candidates", 6), ("max_bond_operations", 4096)):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= limit:
                raise ValueError("Invalid bounded structural " + name)

    def record(self):
        return {**asdict(self), "atom_identity": "fixed_exact_slot_labels",
                "formal_charge": "fixed", "occupancy": "fixed",
                "validity": "connected_native_sanitization_and_schema_roundtrip",
                "operation_budget_scope": "per_constructed_candidate",
                "minimum_is_proven": False, "fallback": "unchanged_original_beam_then_progress"}


def structural_config(value):
    return value if isinstance(value, StructuralRepairConfig) else StructuralRepairConfig(**value)


def _backbone(slots, caps, raw_bonds, costs, single, prefer_existing):
    """Greedy tree on the complete slot graph; never seal a saturated component.

    Keeping a spare degree in each unfinished component avoids joining two
    monovalent atoms into a dead-end fragment while other atoms remain outside.
    All chemical validity claims still come from native validation, not caps.
    """
    components = {i: {i} for i in slots}
    owner = {i: i for i in slots}
    degree = {i: 0 for i in slots}
    tree = []
    while len(components) > 1:
        proposals = []
        residual = {key: sum(caps[i] - degree[i] for i in group)
                    for key, group in components.items()}
        for offset, i in enumerate(slots):
            for j in slots[offset + 1:]:
                a, b = owner[i], owner[j]
                if a == b or degree[i] >= caps[i] or degree[j] >= caps[j]:
                    continue
                merged = len(components[a]) + len(components[b])
                if merged < len(slots) and residual[a] + residual[b] - 2 < 1:
                    continue
                key = (float(costs[i, j, single]), i, j)
                if prefer_existing:
                    key = (int(raw_bonds[i, j] == 0), *key)
                proposals.append((key, i, j, a, b))
        if not proposals:
            return None
        _, i, j, a, b = min(proposals)
        merged = components.pop(b)
        components[a].update(merged)
        for vertex in merged:
            owner[vertex] = a
        degree[i] += 1
        degree[j] += 1
        tree.append((i, j))
    return tree


def _trim(original, tree, caps, orders, costs, single, none, budget, strip_aromatic):
    bonds = original.copy()
    operations = 0
    if strip_aromatic:
        aromatic = next((i for i, order in enumerate(orders) if order == 3), None)
        if aromatic is not None:
            for i, j in zip(*np.triu_indices(len(bonds), 1)):
                if bonds[i, j] == aromatic:
                    bonds[i, j] = bonds[j, i] = single
                    operations += 1
    for i, j in tree:
        if bonds[i, j] == none:
            bonds[i, j] = bonds[j, i] = single
            operations += 1
    protected = set(tree)
    valence = orders[bonds].sum(axis=1)
    while np.any(valence > 2 * caps):
        if operations >= budget:
            return None
        overflow = np.maximum(valence - 2 * caps, 0)
        proposals = []
        for i, j in zip(*np.triu_indices(len(bonds), 1)):
            old = int(bonds[i, j])
            if not overflow[i] and not overflow[j]:
                continue
            for new, order in enumerate(orders):
                if order >= orders[old] or ((i, j) in protected and new == none):
                    continue
                relief = min(overflow[i], orders[old] - order) + min(overflow[j], orders[old] - order)
                if relief <= 0:
                    continue
                delta = float(costs[i, j, new] - costs[i, j, old])
                proposals.append(((delta / relief, delta, i, j, new), i, j, new))
        if not proposals:
            return None
        _, i, j, new = min(proposals)
        difference = orders[int(bonds[i, j])] - orders[new]
        bonds[i, j] = bonds[j, i] = new
        valence[i] -= difference
        valence[j] -= difference
        operations += 1
    return (bonds, operations) if operations <= budget else None


def search_structural_repair(raw, probabilities, spec, search, config=None, *, accept=None):
    """Return a fully verified witness or an honest miss for the old fallback."""
    config = StructuralRepairConfig() if config is None else structural_config(config)
    original = tuple(np.array(v, copy=True) for v in raw)
    p = validate_probabilities(probabilities, spec)
    measure_probability_edits(original, original, p, spec, search)
    result = {"policy": config.record(), "raw_graph_id": graph_identity(*original, spec),
              "probability_sha256": probability_identity(p, spec), "raw_valid": False,
              "weighted_witness": None, "best_observed_weighted_cost": None,
              "checks": 0, "candidates": [], "distance_is_exact": False,
              "beam_search_performed": False}

    def validate(graph):
        result["checks"] += 1
        if accept is not None and not accept(graph, spec):
            return None
        inspected = inspect_graph(*graph, spec)
        if inspected.smiles is None:
            return None
        try:
            encoded = encode_smiles(inspected.smiles, spec)
        except ValueError:
            return None
        return inspected.smiles if encoded["smiles"] == inspected.smiles else None

    if validate(original) is not None:
        return {**result, "status": "already_valid", "raw_valid": True,
                "best_observed_weighted_cost": 0.0}
    slots = np.flatnonzero(original[0]).tolist()
    caps = np.zeros(spec.max_nodes, dtype=np.int64)
    for i in slots:
        allowed = ALLOWED_VALENCES.get(spec.atomic_numbers[int(original[0][i])], {}).get(
            spec.formal_charges[int(original[1][i])])
        if not allowed:
            return {**result, "status": "unsupported", "reason": "No declared valence cap for fixed labels"}
        caps[i] = max(allowed)
    if not slots or (len(slots) > 1 and
                     (any(caps[i] < 1 for i in slots) or sum(caps[i] for i in slots) < 2 * (len(slots) - 1))):
        return {**result, "status": "censored", "reason": "Fixed labels lack connected-backbone degree capacity"}
    none, single = spec.bond_names.index("NONE"), spec.bond_names.index("SINGLE")
    if none != 0:
        raise ValueError("Structural repair requires the existing NONE-zero schema")
    orders = np.asarray([{"NONE": 0, "SINGLE": 2, "DOUBLE": 4, "TRIPLE": 6, "AROMATIC": 3}[name]
                         for name in spec.bond_names])
    costs = np.zeros_like(p[2])
    for i, j in zip(*np.triu_indices(spec.max_nodes, 1)):
        old = int(original[2][i, j])
        for new in range(len(spec.bond_names)):
            regret = max(0., math.log(max(float(p[2][i, j, old]), search.probability_floor)) -
                         math.log(max(float(p[2][i, j, new]), search.probability_floor)))
            costs[i, j, new] = costs[j, i, new] = 0. if old == new else 1. + search.probability_weight * regret
    seen, best, constructed = set(), None, 0
    for prefer_existing in (False, True):
        tree = _backbone(slots, caps, original[2], costs, single, prefer_existing)
        if tree is None:
            continue
        for mode in ("trim", "trim_without_aromatic", "single_backbone"):
            if constructed >= config.max_candidates:
                break
            constructed += 1
            if mode == "single_backbone":
                bonds = np.zeros_like(original[2])
                for i, j in tree:
                    bonds[i, j] = bonds[j, i] = single
                operations = int(np.count_nonzero(np.triu(bonds != original[2], 1)))
                proposed = (bonds, operations) if operations <= config.max_bond_operations else None
            else:
                proposed = _trim(original[2], tree, caps, orders, costs, single, none,
                                 config.max_bond_operations, mode == "trim_without_aromatic")
            if proposed is None:
                result["candidates"].append({"kind": mode, "status": "operation_budget_or_capacity_miss"})
                continue
            bonds, operations = proposed
            identity = bonds.tobytes()
            if identity in seen:
                continue
            seen.add(identity)
            target = (original[0].copy(), original[1].copy(), bonds)
            smiles = validate(target)
            cost = measure_probability_edits(original, target, p, spec, search)
            result["candidates"].append({"kind": mode, "prefer_existing_edges": prefer_existing,
                "status": "valid" if smiles is not None else "native_or_roundtrip_rejected",
                "bond_operations": operations, "bond_edits": cost["edit_count"],
                "weighted_cost": cost["weighted_cost"]})
            if smiles is not None and (best is None or
                    (cost["weighted_cost"], cost["edit_count"]) < (best["weighted_cost"], best["edit_count"])):
                best = {**cost, "graph_id": graph_identity(*target, spec), "smiles": smiles,
                        "labels": {name: value.tolist() for name, value in zip(("atoms", "charges", "bonds"), target)}}
    if best is None:
        return {**result, "status": "censored", "reason": "No verified structural candidate within budget"}
    return {**result, "status": "witness_found", "weighted_witness": best,
            "best_observed_weighted_cost": best["weighted_cost"]}
