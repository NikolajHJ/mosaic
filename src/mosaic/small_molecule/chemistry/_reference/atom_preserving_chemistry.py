"""Detached reference extraction; see source_manifest.json. Do not edit manually."""
from __future__ import annotations
import math
import numpy as np
from .chemistry import encode_smiles
from .probability_repair import ProbabilityRepairConfig, validate_probabilities, probability_identity, measure_probability_edits
from .quality import diagnose_graph
from .repair_search import round_robin_candidates
from .representation import GraphSpec


VERSION = "atom-preserving-bond-repair-v1"


def repair_policy(config: ProbabilityRepairConfig) -> dict:
    """Declare a restricted search without changing historical repair defaults."""
    return {**config.policy(), "version": VERSION, "allowed_edits": ["bond"],
            "atom_identity": "fixed_exact_slot_labels", "formal_charge": "fixed",
            "occupancy": "fixed", "alignment": "shared_slots",
            "validity": "nonempty_connected_RDKit_sanitization_and_schema_roundtrip",
            "already_valid": "skip_without_self_target", "sampling": "argmax_no_rng"}


def _key(graph):
    return tuple(value.tobytes() for value in graph)


def _bond_edits(graph, original, spec, diagnostic, probabilities, config):
    """Generate only unordered bond neighbors; no atom/charge proposal exists."""
    slots = np.flatnonzero(graph[0]).tolist()
    implicated = {int(i) for problem in diagnostic.get("problems", []) for i in problem["slots"]}
    component = {}
    for start in slots:
        if start in component:
            continue
        stack, component[start] = [start], start
        while stack:
            i = stack.pop()
            for j in slots:
                if j not in component and graph[2][i, j]:
                    component[j] = start
                    stack.append(j)
    operations = []
    for offset, i in enumerate(slots):
        for j in slots[offset + 1:]:
            if graph[2][i, j] != original[2][i, j]:
                continue  # Never change the same label twice on one path.
            priority = 0 if i in implicated or j in implicated else 2 if component[i] != component[j] else 3
            old = int(original[2][i, j])
            for new in range(len(spec.bond_names)):
                if new == old:
                    continue
                regret = max(0.0, math.log(max(float(probabilities[2][i, j, old]), config.probability_floor))
                             - math.log(max(float(probabilities[2][i, j, new]), config.probability_floor)))
                operations.append((priority, regret, i, j, new))
    for _, _, i, j, new in sorted(operations):
        candidate = tuple(value.copy() for value in graph)
        candidate[2][i, j] = candidate[2][j, i] = new
        yield candidate


def search_atom_preserving_repair(raw, probabilities, spec: GraphSpec,
                                  config: ProbabilityRepairConfig) -> dict:
    """Observe best valid weighted witness under a bounded bond-only search.

    The shared edit-cost implementation supplies probability regret. Budget
    exhaustion and beam pruning remain explicit; failure is not assigned an
    invented finite distance and never authorizes an atom/charge substitution.
    """
    original = tuple(np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value).copy()
                     for value in raw)
    p = validate_probabilities(probabilities, spec)
    diagnostic = diagnose_graph(*original, spec)
    raw_rdkit_valid = diagnostic["valid"]
    raw_roundtrip_error = None
    if raw_rdkit_valid:
        try:
            if encode_smiles(diagnostic["smiles"], spec)["smiles"] != diagnostic["smiles"]:
                raise ValueError("Canonical schema roundtrip identity differs")
        except ValueError as error:
            # Sanitization alone does not certify schema roundtrip (notably
            # some aromatic label assignments). Continue the bounded search
            # rather than silently treating this candidate as already valid.
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
    # This validates labels, symmetry, diagonal and padding before any search.
    measure_probability_edits(original, original, p, spec, config)
    if diagnostic["valid"]:
        return {**result, "status": "already_valid", "best_observed_weighted_cost": 0.0}
    seen, beam, best, sequence = {_key(original)}, [(original, diagnostic)], None, 0
    for depth in range(1, config.max_edits + 1):
        frontier, checked, capped = [], 0, False
        streams = [_bond_edits(graph, original, spec, diag, p, config) for graph, diag in beam]
        for candidate in round_robin_candidates(streams):
            identity = _key(candidate)
            if identity in seen:
                continue
            if checked >= config.checks_per_depth:
                capped = True
                break
            seen.add(identity)
            candidate_diag = diagnose_graph(*candidate, spec)
            cost = measure_probability_edits(original, candidate, p, spec, config)
            checked += 1
            result["checks"] += 1
            if cost["edit_count"] != depth or cost["edit_counts"]["atom"] or cost["edit_counts"]["charge"]:
                raise RuntimeError("Bond-only generator violated the direct-cost invariant")
            if candidate_diag["valid"]:
                # Schema representability is stronger than RDKit sanitization.
                try:
                    encoded = encode_smiles(candidate_diag["smiles"], spec)
                    if encoded["smiles"] != candidate_diag["smiles"]:
                        raise ValueError("Canonical schema roundtrip identity differs")
                except ValueError:
                    result["roundtrip_rejected_candidates"] += 1
                    candidate_diag = {**candidate_diag, "valid": False,
                                      "problem_burden": max(1, candidate_diag.get("problem_burden") or 0)}
            if candidate_diag["valid"]:
                result["valid_witnesses_observed"] += 1
                witness = {**cost, "graph_id": candidate_diag["graph_id"], "smiles": candidate_diag["smiles"],
                           "labels": {name: value.tolist() for name, value in
                                      zip(("atoms", "charges", "bonds"), candidate)}}
                if best is None or (cost["weighted_cost"], cost["edit_count"]) < (best["weighted_cost"], best["edit_count"]):
                    best = witness
                continue
            sequence += 1
            burden = candidate_diag.get("problem_burden")
            rank = ((1000000 if burden is None else burden)
                    + max(0, candidate_diag.get("component_count", 1) - 1), cost["weighted_cost"], sequence)
            frontier.append((rank, candidate, candidate_diag))
            frontier.sort(key=lambda value: value[0])
            del frontier[config.beam_width:]
        result["checks_by_depth"].append({"depth": depth, "checks": checked, "capped": capped})
        beam = [(item[1], item[2]) for item in frontier]
        if not beam:
            break
    if best is None:
        return {**result, "status": "censored", "reason": "No valid bond-only witness within the declared budget"}
    return {**result, "status": "witness_found", "weighted_witness": best,
            "best_observed_weighted_cost": best["weighted_cost"]}
