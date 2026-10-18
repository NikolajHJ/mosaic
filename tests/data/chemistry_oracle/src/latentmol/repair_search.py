"""Bounded hard-graph edit search for evaluation, not a learned chemistry prior.

The verifier is supplied by the caller. Only occupied atom types, occupied
charges and unordered bonds change; occupancy and slot identity remain fixed.
Each changed categorical label costs one edit. A witness establishes an upper
bound on repair distance in this search space, not necessarily the closest
molecule. No witness leaves distance/proximity unknown, with explicit bounds.
"""

from dataclasses import asdict, dataclass
from collections import deque
from typing import Callable

import numpy as np

from .representation import GraphSpec


@dataclass(frozen=True)
class RepairSearchConfig:
    """Same deterministic budget for each unique graph in every comparison arm."""

    max_edits: int = 3
    checks_per_depth: int = 256
    beam_width: int = 4

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in asdict(self).values()):
            raise ValueError("Repair search budgets must be positive integers")

    def policy(self):
        return {"algorithm": "bounded-label-beam-v1", **asdict(self),
                "occupancy": "fixed", "edit_cost": "one_per_changed_category",
                "priority": "problem_slots_then_fragment_links_then_remaining_labels",
                "transform": "1/(1+best_observed_edit_cost)"}


def _arrays(graph):
    return tuple(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x,
                            dtype=np.int64).copy() for x in graph)


def _key(graph):
    return tuple(x.tobytes() for x in graph)


def round_robin_candidates(streams):
    """Allocate each layer's budget across retained parents, not just the first."""
    pending = deque(iter(stream) for stream in streams)
    while pending:
        stream = pending.popleft()
        try:
            yield next(stream)
            pending.append(stream)
        except StopIteration:
            pass


def single_label_edits(graph, original, spec, diagnostic, *, operation_rank=None):
    """Deterministic single-label neighbors; never change a label twice on a path.

    Consequently depth equals exact slot-label Hamming cost to the raw graph.
    Bond direction is never a separate edit. No atoms may disappear as a shortcut.
    """
    atoms, charges, bonds = graph
    slots = np.flatnonzero(atoms).tolist()
    implicated = {int(i) for p in diagnostic.get("problems", []) for i in p["slots"]}
    operations = []
    # Stable priority puts problem-local bonds before substitutions, then links
    # between disconnected components, before all remaining schema labels.
    component = {}
    for start in slots:
        if start in component:
            continue
        label, stack = start, [start]
        component[start] = label
        while stack:
            i = stack.pop()
            for j in slots:
                if j not in component and bonds[i, j]:
                    component[j] = label
                    stack.append(j)
    for offset, i in enumerate(slots):
        for j in slots[offset + 1:]:
            if bonds[i, j] != original[2][i, j]:
                continue
            priority = 0 if i in implicated or j in implicated else 2 if component[i] != component[j] else 3
            for value in range(len(spec.bond_names)):
                if value != bonds[i, j]:
                    operations.append((priority, 2, i, j, value))
        for field, categories in ((0, range(1, len(spec.atomic_numbers))),
                                  (1, range(len(spec.formal_charges)))):
            if graph[field][i] != original[field][i]:
                continue
            for value in categories:
                if value != graph[field][i]:
                    operations.append((1 if i in implicated else 4, field, i, -1, value))
    # Optional ordering changes discovery only, never the allowed edit space or
    # the validity oracle. Rank operations before allocating full graph copies.
    ordered = sorted(operations, key=operation_rank) if operation_rank else sorted(operations)
    for _, field, i, j, value in ordered:
        candidate = tuple(x.copy() for x in graph)
        if field == 2:
            candidate[2][i, j] = candidate[2][j, i] = value
        else:
            candidate[field][i] = value
        yield candidate


def search_repair(graph, spec: GraphSpec, diagnostic: dict, verify: Callable,
                  config: RepairSearchConfig | None = None) -> dict:
    """Find a verified repair within a declared deterministic beam/check budget.

    Capped layers do not establish that shorter repairs do not exist. A complete
    one-edit layer permits the stronger distance lower bound two. Actual checks
    and theoretical one-edit neighbors are reported separately. The raw input
    is never replaced by the witness in any scorer or downstream model input.
    """
    config = config or RepairSearchConfig()
    fresh = verify(*graph, spec)
    if fresh.get("graph_id") != diagnostic.get("graph_id") or fresh["valid"] != diagnostic["valid"]:
        raise ValueError("Repair diagnostic belongs to a different graph or validity state")
    result = {"policy": config.policy(), "raw_graph_id": diagnostic.get("graph_id"),
              "checks": 0, "checks_by_depth": [], "one_edit_exhausted": False,
              "best_observed_edit_cost": None, "proximity": None,
              "proximity_bounds": [0.0, 1.0], "distance_lower_bound": None,
              "distance_is_exact_in_fixed_occupancy_space": False, "witness": None}
    if (not diagnostic.get("schema_supported") or not diagnostic.get("heavy_atoms")
            or not diagnostic.get("categorical_support_labels_ok")):
        return {**result, "status": "unsupported", "reason": "Requires nonempty canonical hard-label support"}
    if diagnostic["valid"]:
        return {**result, "status": "already_valid", "best_observed_edit_cost": 0,
                "distance_lower_bound": 0, "proximity": 1.0, "proximity_bounds": [1.0, 1.0],
                "distance_is_exact_in_fixed_occupancy_space": True}
    original = _arrays(graph)
    occupied = int(np.count_nonzero(original[0]))
    result["one_edit_neighbors"] = (occupied * (len(spec.atomic_numbers) - 2 + len(spec.formal_charges) - 1)
                                    + occupied * (occupied - 1) // 2 * (len(spec.bond_names) - 1))
    beam = [(original, diagnostic)]
    seen = {_key(original)}
    lower = 1
    sequence = 0
    for depth in range(1, config.max_edits + 1):
        frontier, checked, capped = [], 0, False
        streams = [single_label_edits(current, original, spec, current_diagnostic) for current, current_diagnostic in beam]
        for candidate in round_robin_candidates(streams):
            key = _key(candidate)
            if key in seen:
                continue
            if checked >= config.checks_per_depth:
                capped = True
                break
            seen.add(key)
            candidate_diagnostic = verify(*candidate, spec)
            checked += 1
            result["checks"] += 1
            if candidate_diagnostic["valid"]:
                result["checks_by_depth"].append(checked)
                counts = {"atom": int(np.count_nonzero(candidate[0] != original[0])),
                          "charge": int(np.count_nonzero(candidate[1] != original[1])),
                          "bond": int(np.count_nonzero(np.triu(candidate[2] != original[2], 1))),
                          "occupancy": 0}
                cost = sum(counts.values())
                if cost != depth:
                    raise RuntimeError("Edit generator violated direct-cost invariant")
                return {**result, "status": "repaired", "best_observed_edit_cost": cost,
                        "edit_counts": counts, "witness_graph_id": candidate_diagnostic["graph_id"],
                        "distance_lower_bound": lower, "proximity": 1 / (1 + cost),
                        "proximity_bounds": [1 / (1 + cost), 1 / (1 + lower)],
                        "distance_is_exact_in_fixed_occupancy_space": lower == cost,
                        "witness": {"atoms": candidate[0].tolist(), "charges": candidate[1].tolist(),
                                    "bonds": candidate[2].tolist(), "smiles": candidate_diagnostic["smiles"]}}
            sequence += 1
            burden = candidate_diagnostic.get("problem_burden")
            # Heuristic ordering only; it never serves as the repair cost.
            rank = ((burden if burden is not None else 1000000)
                    + max(0, candidate_diagnostic.get("component_count", 1) - 1))
            frontier.append((rank, sequence, candidate, candidate_diagnostic))
            frontier.sort(key=lambda item: (item[0], item[1]))
            del frontier[config.beam_width:]
        result["checks_by_depth"].append(checked)
        if depth == 1 and not capped:
            result["one_edit_exhausted"] = True
            lower = 2
        beam = [(item[2], item[3]) for item in frontier]
        if not beam:
            break
    return {**result, "status": "censored", "distance_lower_bound": lower,
            "proximity_bounds": [0.0, 1 / (1 + lower)],
            "reason": "No verified repair found under the declared search budget"}
