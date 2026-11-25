"""Invariant-reusing execution of the unchanged bounded bond-repair search.

This is an implementation backend, not a new teacher or validity objective.
The original ordered search remains in ``atom_preserving_chemistry`` as the
reference. No candidate, budget, beam state, tie or valid witness is skipped.
Only invariant validation/logarithms and unnecessary rejected-candidate receipt
construction are reused. Final witnesses still use the public full validator.
All numerical checks and benchmarks belong on allocated DTU HPC.
"""
from __future__ import annotations

import math

import numpy as np

from .atom_preserving_diagnostics import BondOnlyDiagnostic
from .chemistry import encode_smiles
from .probability_repair import measure_probability_edits, probability_identity, validate_probabilities
from .quality import graph_identity
from .repair_search import round_robin_candidates


class BondSearchContext:
    """Immutable per-point probabilities, active pairs and scalar regret table.

    The table uses exactly the reference's scalar ``math.log`` operations.
    Every candidate's total is summed in row-major unordered-pair order, not
    path order, preserving floating point ranking and tie behavior.
    """

    def __init__(self, original, probabilities, spec, config):
        self.original, self.probabilities, self.spec, self.config = original, probabilities, spec, config
        self.slots = np.flatnonzero(original[0]).tolist()
        self.pairs = [(i, j) for offset, i in enumerate(self.slots) for j in self.slots[offset + 1:]]
        self.regrets = {}
        for i, j in self.pairs:
            old = int(original[2][i, j])
            old_log = math.log(max(float(probabilities[2][i, j, old]), config.probability_floor))
            for new in range(len(spec.bond_names)):
                if new != old:
                    self.regrets[i, j, new] = max(0.0, old_log - math.log(
                        max(float(probabilities[2][i, j, new]), config.probability_floor)))

    def cost(self, bonds):
        positions = np.argwhere(np.triu(self.original[2] != bonds, 1))
        regret = sum(self.regrets[int(i), int(j), int(bonds[i, j])] for i, j in positions)
        count = len(positions)
        return count, count + self.config.probability_weight * regret

    def neighbors(self, graph, diagnostic):
        """Recompute state-dependent priorities; reuse only immutable regret."""
        implicated = {int(i) for problem in diagnostic.get("problems", []) for i in problem["slots"]}
        component = {}
        for start in self.slots:
            if start in component:
                continue
            stack, component[start] = [start], start
            while stack:
                i = stack.pop()
                for j in self.slots:
                    if j not in component and graph[2][i, j]:
                        component[j] = start
                        stack.append(j)
        operations = []
        for i, j in self.pairs:
            old = int(self.original[2][i, j])
            if graph[2][i, j] != old:
                continue
            priority = 0 if i in implicated or j in implicated else 2 if component[i] != component[j] else 3
            for new in range(len(self.spec.bond_names)):
                if new != old:
                    operations.append((priority, self.regrets[i, j, new], i, j, new))
        for _, _, i, j, new in sorted(operations):
            # Atoms and charges are read-only invariants. Only bonds are copied.
            bonds = graph[2].copy()
            bonds[i, j] = bonds[j, i] = new
            yield graph[0], graph[1], bonds


def search_atom_preserving_repair_fast(raw, probabilities, spec, config):
    """Return the same complete scientific search receipt as the reference.

    Public input validation and independent final target validation remain.
    Noncanonical/unsupported input takes the original path, not a new verdict.
    The shared fixed-label diagnostic omits unused supplementary quality work;
    its RDKit decisions must pass paired reference equivalence before promotion.
    """
    from .atom_preserving_chemistry import repair_policy, search_atom_preserving_repair

    original = tuple(np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value).copy()
                     for value in raw)
    p = validate_probabilities(probabilities, spec)
    try:
        engine = BondOnlyDiagnostic(original, spec)
    except (ValueError, RuntimeError, TypeError, KeyError, IndexError, OverflowError):
        return search_atom_preserving_repair(original, p, spec, config)
    diagnostic = engine.diagnose(original[2], include_graph_id=True)
    raw_rdkit_valid = diagnostic["valid"]
    raw_roundtrip_error = None
    if raw_rdkit_valid:
        try:
            if encode_smiles(diagnostic["smiles"], spec)["smiles"] != diagnostic["smiles"]:
                raise ValueError("Canonical schema roundtrip identity differs")
        except ValueError as error:
            raw_roundtrip_error = f"{type(error).__name__}: {error}"
            diagnostic = {**diagnostic, "valid": False,
                          "problem_burden": max(1, diagnostic.get("problem_burden") or 0)}
    result = {"policy": repair_policy(config), "raw_graph_id": diagnostic.get("graph_id"),
              "probability_sha256": probability_identity(p, spec), "raw_valid": diagnostic["valid"],
              "raw_rdkit_valid": raw_rdkit_valid, "raw_roundtrip_error": raw_roundtrip_error,
              "checks": 0, "checks_by_depth": [], "valid_witnesses_observed": 0,
              "roundtrip_rejected_candidates": 0,
              "weighted_witness": None, "best_observed_weighted_cost": None,
              "distance_is_exact": False}
    if (not diagnostic.get("schema_supported") or not diagnostic.get("heavy_atoms")
            or not diagnostic.get("categorical_support_labels_ok")):
        return {**result, "status": "unsupported", "reason": "Requires nonempty canonical hard-label support"}
    # Validate the no-op once, including padding and unordered-bond invariants.
    measure_probability_edits(original, original, p, spec, config)
    if diagnostic["valid"]:
        return {**result, "status": "already_valid", "best_observed_weighted_cost": 0.0}
    context = BondSearchContext(original, p, spec, config)
    seen, beam, best, sequence = {original[2].tobytes()}, [(original, diagnostic)], None, 0
    for depth in range(1, config.max_edits + 1):
        frontier, checked, capped = [], 0, False
        streams = [context.neighbors(graph, diag) for graph, diag in beam]
        for candidate in round_robin_candidates(streams):
            identity = candidate[2].tobytes()
            if identity in seen:
                continue
            if checked >= config.checks_per_depth:
                capped = True
                break
            seen.add(identity)
            candidate_diag = engine.diagnose(candidate[2])
            count, weighted = context.cost(candidate[2])
            checked += 1
            result["checks"] += 1
            if count != depth:
                raise RuntimeError("Bond-only generator violated the direct-cost invariant")
            if candidate_diag["valid"]:
                try:
                    if encode_smiles(candidate_diag["smiles"], spec)["smiles"] != candidate_diag["smiles"]:
                        raise ValueError("Canonical schema roundtrip identity differs")
                except ValueError:
                    result["roundtrip_rejected_candidates"] += 1
                    candidate_diag = {**candidate_diag, "valid": False,
                                      "problem_burden": max(1, candidate_diag.get("problem_burden") or 0)}
            if candidate_diag["valid"]:
                result["valid_witnesses_observed"] += 1
                # Preserve the public detailed edit receipt and its exact cost.
                cost = measure_probability_edits(original, candidate, p, spec, config)
                if cost["edit_count"] != count or cost["weighted_cost"] != weighted:
                    raise RuntimeError("Cached bond cost differs from the public exact edit metric")
                witness = {**cost, "graph_id": graph_identity(*candidate, spec), "smiles": candidate_diag["smiles"],
                           "labels": {name: value.tolist() for name, value in zip(("atoms", "charges", "bonds"), candidate)}}
                if best is None or (cost["weighted_cost"], cost["edit_count"]) < (best["weighted_cost"], best["edit_count"]):
                    best = witness
                continue
            sequence += 1
            burden = candidate_diag.get("problem_burden")
            rank = ((1000000 if burden is None else burden)
                    + max(0, candidate_diag.get("component_count", 1) - 1), weighted, sequence)
            frontier.append((rank, candidate, candidate_diag))
            frontier.sort(key=lambda item: item[0])
            del frontier[config.beam_width:]
        result["checks_by_depth"].append({"depth": depth, "checks": checked, "capped": capped})
        beam = [(item[1], item[2]) for item in frontier]
        if not beam:
            break
    if best is None:
        return {**result, "status": "censored", "reason": "No valid bond-only witness within the declared budget"}
    return {**result, "status": "witness_found", "weighted_witness": best,
            "best_observed_weighted_cost": best["weighted_cost"]}
