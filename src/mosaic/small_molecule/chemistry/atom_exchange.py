"""Detached single-atom exchanges over the existing structural builder."""

import math

import numpy as np

from ._reference.chemistry import inspect_graph
from ._reference.chemistry_regularization import ALLOWED_VALENCES
from ._reference.probability_repair import measure_probability_edits, validate_probabilities
from ._reference.quality import diagnose_graph, graph_identity
from .environment_policy import BOND_ORDERS, assess_environment
from .structural import search_structural_repair


def within_common_organic_scope(graph, spec):
    assessment, = assess_environment(*([np.asarray(labels).tolist()] for labels in graph), spec)
    return all(assessment[key] for key in (
        "fully_assessed", "local_valence_ok", "within_common_organic_scope",
    ))


def search_atom_exchange(raw, probabilities, spec, search, *, atom_edit_penalty=1.0,
                         max_substitutions=4):
    """Compare scoped bond-only repair to a bounded shortlist of one-atom edits.

    A target is scored from the original graph, including all bond changes. New
    sulfur must fit the old atom's bond-order cap as well as the common-organic
    scope. Charges and occupied slots remain fixed. This is proposal search.
    """
    if (type(atom_edit_penalty) not in (int, float) or not math.isfinite(atom_edit_penalty)
            or atom_edit_penalty <= 0 or type(max_substitutions) is not int
            or not 1 <= max_substitutions <= 8):
        raise ValueError("Require a positive finite penalty and a bounded shortlist")

    p = validate_probabilities(probabilities, spec)
    original = tuple(np.asarray(labels) for labels in raw)
    baseline = search_structural_repair(original, p, spec, search, accept=within_common_organic_scope)
    result = {"baseline": baseline, "selected": baseline, "substitutions_tested": 0}
    if baseline["status"] == "already_valid":
        return result

    best_cost = math.inf
    witness = baseline["weighted_witness"]
    if witness is not None:
        best_cost = witness["weighted_cost"]
        result["selected"] = {**baseline, "weighted_witness": {**witness, "repair_cost": best_cost}}

    atoms, charges, bonds = original
    diagnostic = diagnose_graph(*original, spec)
    implicated = {slot for problem in diagnostic.get("problems", ()) for slot in problem["slots"]}
    orders = np.asarray([3 if name == "AROMATIC" else 2 * BOND_ORDERS[name]
                         for name in spec.bond_names])
    alternatives = []
    for slot in np.flatnonzero(atoms):
        old = int(atoms[slot])
        charge = spec.formal_charges[int(charges[slot])]
        old_allowed = ALLOWED_VALENCES.get(spec.atomic_numbers[old], {}).get(charge)
        for label, element in enumerate(spec.atomic_numbers[1:], start=1):
            if label == old or p[0][slot, label] == 0 or not ALLOWED_VALENCES.get(element, {}).get(charge):
                continue
            if element == 16 and not old_allowed:
                continue
            old_p, new_p = (max(float(p[0][slot, v]), search.probability_floor) for v in (old, label))
            cost = 1 + search.probability_weight * max(0.0, math.log(old_p) - math.log(new_p))
            cost += atom_edit_penalty
            if cost < best_cost:
                alternatives.append((cost, int(slot), label, max(old_allowed) if element == 16 else None))

    alternatives.sort(key=lambda item: (item[1] not in implicated, item[:3]))
    for lower_bound, slot, label, sulfur_cap in alternatives[:max_substitutions]:
        if lower_bound >= best_cost:
            continue
        swapped_atoms = atoms.copy()
        swapped_atoms[slot] = label
        swapped = (swapped_atoms, charges, bonds)

        def accept(graph, candidate_spec):
            return ((sulfur_cap is None or int(orders[graph[2][slot]].sum()) <= 2 * sulfur_cap)
                    and within_common_organic_scope(graph, candidate_spec))

        candidate = search_structural_repair(swapped, p, spec, search, accept=accept)
        result["substitutions_tested"] += 1
        if candidate["status"] == "already_valid":
            target = swapped
            smiles = inspect_graph(*target, spec).smiles
        elif candidate["weighted_witness"] is not None:
            new = candidate["weighted_witness"]
            target = tuple(np.asarray(new["labels"][key]) for key in ("atoms", "charges", "bonds"))
            smiles = new["smiles"]
        else:
            continue
        measured = measure_probability_edits(original, target, p, spec, search)
        cost = measured["weighted_cost"] + atom_edit_penalty * measured["edit_counts"]["atom"]
        if cost < best_cost:
            best_cost = cost
            result["selected"] = {
                **baseline, "status": "witness_found", "reason": None,
                "best_observed_weighted_cost": cost,
                "weighted_witness": {
                    **measured, "repair_cost": cost, "smiles": smiles,
                    "graph_id": graph_identity(*target, spec),
                    "labels": {key: labels.tolist() for key, labels in
                               zip(("atoms", "charges", "bonds"), target)},
                },
                "exchange": {"slot": slot, "old_label": int(atoms[slot]), "new_label": label,
                             "penalty": atom_edit_penalty, "introduced_sulfur_old_cap": sulfur_cap},
            }
    return result
