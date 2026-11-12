"""Detached endpoint defects and bounded, explicitly verified repair witnesses.

Nothing here changes an optimization state, selects its reported best molecule,
or evaluates a native objective. A witness gives an upper bound on the edit
distance. Failure to find one is censored evidence, not proof of irreparability.
"""

import math
from types import SimpleNamespace

import numpy as np


ELEMENTS = ("C", "N", "O", "S", "F", "Cl")
CAPS = np.array([4., 3., 2., 2., 1., 1.])
BOND_ORDER = np.array([0., 1., 2., 3., 1.5])


def _geometry(elements, bonds):
    n = len(elements)
    i, j = np.triu_indices(n, 1)
    parent = np.arange(n)

    def find(atom):
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    used = np.zeros(n)
    orders = BOND_ORDER[bonds]
    np.add.at(used, i, orders)
    np.add.at(used, j, orders)
    for row in np.flatnonzero(bonds):
        parent[find(i[row])] = find(j[row])
    labels = np.array([find(a) for a in range(n)])
    _, sizes = np.unique(labels, return_counts=True)
    excess = np.maximum(used - CAPS[elements], 0)
    return labels, sorted(sizes.tolist(), reverse=True), used, excess


def _verify(elements, bonds):
    """Use the same raw, unrepaired RDKit readout as the native campaign."""
    from rdkit import rdBase
    from mosaic.small_molecule.Stage1.diagnostics import raw_readout

    # The main readout constructs its hard graph. Supplying detached NumPy
    # rows here avoids any differentiated/native computation in this search.
    graph = SimpleNamespace(X=np.eye(6, dtype=np.float32)[elements],
                            E_upper=np.eye(5, dtype=np.float32)[bonds],
                            n_atoms=len(elements))
    with rdBase.BlockLogs():
        _, report = raw_readout(graph)
    return report


def _heuristic(elements, bonds, edits, source_edges):
    """Ranking only; this is not a certified distance lower bound."""
    _, sizes, _, excess = _geometry(elements, bonds)
    # A valence edit may temporarily fragment a graph, so penalize both
    # defects rather than rejecting all moves that add a component.
    difficulty = 2 * max(len(sizes) - 1, 0) + float(excess.sum())
    aromatic = int(np.count_nonzero(bonds == 4))
    probability_cost = sum(-math.log(max(float(source_edges[edit["row"], edit["to"]]), 1e-12))
                           for edit in edits)
    return difficulty, aromatic, probability_cost, bonds.tobytes()


def _proposals(elements, bonds, edits):
    """Single bond-label edits: deletion, order reduction or feasible bridge.

    Each pair can be changed once. Because all reduced bond categories are
    proposed directly, repeating an edit on one pair is unnecessary and would
    make the action count differ from the final label-edit distance.
    """
    n = len(elements)
    i, j = np.triu_indices(n, 1)
    labels, _, used, _ = _geometry(elements, bonds)
    touched = {edit["row"] for edit in edits}
    caps = CAPS[elements]
    for row, category in enumerate(bonds.tolist()):
        if row in touched:
            continue
        if category:
            # Aromatic-to-single is an explicit label change; whether the
            # resulting graph sanitizes is checked by RDKit, never assumed.
            options = [value for value in (0, 1, 2, 3)
                       if value != category and BOND_ORDER[value] < BOND_ORDER[category]]
        elif (labels[i[row]] != labels[j[row]] and used[i[row]] + 1 <= caps[i[row]]
              and used[j[row]] + 1 <= caps[j[row]]):
            options = [1]
        else:
            options = []
        for value in options:
            candidate = bonds.copy()
            candidate[row] = value
            edit = {"row": row, "pair": [int(i[row]), int(j[row])],
                    "from": int(category), "to": int(value),
                    "operation": "bridge" if category == 0 else "delete" if value == 0 else "reduce_order"}
            yield candidate, edits + [edit]


def assess_final(graph, max_edits=3, checks=128, beam_width=4):
    """Return raw defects and the closest *found* verified repair witness.

    Bounds apply to a bond-only beam search with fixed atom labels/occupancy.
    ``checks`` is the total RDKit-call budget including the raw endpoint.
    A sanitized connected witness keeps every atom. It is never fed back into
    the optimization or treated as one of its generated valid molecules.
    """
    for name, value, lower in (("max_edits", max_edits, 0), ("checks", checks, 1),
                               ("beam_width", beam_width, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    x, e = np.asarray(graph.X), np.asarray(graph.E_upper)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != 6:
        raise ValueError("Endpoint atoms must have shape [N>=1,6]")
    n = len(x)
    if e.shape != (n * (n - 1) // 2, 5):
        raise ValueError("Endpoint bonds must have shape [N*(N-1)//2,5]")
    for name, values in (("X", x), ("E_upper", e)):
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite floating probabilities")
        if (values < -1e-6).any() or not np.allclose(values.sum(-1), 1., atol=1e-6, rtol=0):
            raise ValueError(f"{name} must contain probability simplex rows")
    elements = x.argmax(-1).astype(np.int32)
    original = e.argmax(-1).astype(np.int32)
    report = _verify(elements, original)
    labels, sizes, used, excess = _geometry(elements, original)
    raw_valid = bool(report["rdkit_sanitized"] and len(sizes) == 1)
    result = {
        "schema": "defog-stage3-endpoint-v1", "diagnostic_only": True,
        "changes_are_used_by_optimization": False,
        "validity_policy": "raw_argmax_RDKit_sanitized_and_single_fragment",
        "raw": {
            "valid_connected": raw_valid, "rdkit_sanitized": report["rdkit_sanitized"],
            "canonical_smiles": report["canonical_smiles"],
            "sanitization_error": report["sanitization_error"],
            "component_count": len(sizes), "component_sizes": sizes,
            "atoms_outside_largest_component": n - sizes[0],
            "neutral_hard_valence": {
                "aromatic_order": 1.5, "caps_by_element": dict(zip(ELEMENTS, CAPS.tolist())),
                "used_per_atom": used.tolist(), "caps_per_atom": CAPS[elements].tolist(),
                "violating_atoms": np.flatnonzero(excess > 1e-8).tolist(),
                "total_excess": float(excess.sum()), "maximum_excess": float(excess.max(initial=0)),
            },
        },
        "minimum_bond_additions_for_connectivity": max(len(sizes) - 1, 0),
        "closest_verified_witness": None, "distance_upper_bound": None,
        "distance_unknown": True, "minimum_is_proven": False,
        "no_witness_means": "unknown/censored within this bounded bond-only proposal search",
        "search": {
            "max_edits": max_edits, "checks_limit": checks, "beam_width": beam_width,
            "checks_used": 1, "states_generated": 1, "depth_reached": 0,
            "checks_by_depth": [],
            "atom_count_preserved": n, "atom_labels_fixed": True,
            "proposal_space": "bond deletion, order reduction, neutral-capacity single-bond bridge",
            "edit_unit": "one unordered pair changes bond category; a pair is edited at most once",
            "heuristic_is_distance_bound": False, "termination": None,
        },
    }

    def witness(bonds, edits, verified):
        count = int(np.count_nonzero(bonds != original))
        if count != len(edits):
            raise RuntimeError("Repair witness edit-count mismatch")
        result.update({
            "closest_verified_witness": {
                "edit_count": count, "smiles": verified["canonical_smiles"],
                "element_indices": elements.tolist(), "bond_upper_indices": bonds.tolist(),
                "edits": edits,
                "verification": {"rdkit_sanitized": True, "component_count": 1,
                                 "atoms_retained": n, "all_atom_labels_preserved": True},
                "method": "bounded_bond_label_beam", "is_upper_bound": True,
            },
            "distance_upper_bound": count, "distance_unknown": False,
            "minimum_is_proven": count == 0,
        })

    if raw_valid:
        witness(original, [], report)
        result["search"]["termination"] = "already_valid"
        return result
    if len(sizes) - 1 > max_edits:
        result["search"]["termination"] = "connectivity_lower_bound_exceeds_edit_budget"
        return result
    if max_edits == 0:
        result["search"]["termination"] = "edit_budget_zero"
        return result
    frontier = [(original, [])]
    visited = {original.tobytes()}
    for depth in range(1, max_edits + 1):
        candidates = []
        for bonds, edits in frontier:
            for changed, next_edits in _proposals(elements, bonds, edits):
                identity = changed.tobytes()
                if identity in visited:
                    continue
                visited.add(identity)
                candidates.append((_heuristic(elements, changed, next_edits, e), changed, next_edits))
        result["search"]["states_generated"] = len(visited)
        if not candidates:
            result["search"]["termination"] = "proposal_space_exhausted"
            return result
        candidates.sort(key=lambda candidate: candidate[0])
        next_frontier = []
        result["search"]["depth_reached"] = depth
        remaining = checks - result["search"]["checks_used"]
        if remaining <= 0:
            result["search"]["termination"] = "verification_budget_exhausted"
            return result
        # Reserve checks for deeper repairs rather than consuming the entire
        # budget on hundreds of one-edit partial bridges. The retained beam
        # is still bounded, and omitted proposals remain explicitly untested.
        quota = min(remaining, max(beam_width, remaining // (max_edits - depth + 1)))
        checked_at_depth = 0
        result["search"]["checks_by_depth"].append({"depth": depth, "limit": quota, "used": 0})
        for rank, bonds, edits in candidates[:quota]:
            if result["search"]["checks_used"] >= checks:
                result["search"]["termination"] = "verification_budget_exhausted"
                return result
            verified = _verify(elements, bonds)
            result["search"]["checks_used"] += 1
            checked_at_depth += 1
            result["search"]["checks_by_depth"][-1]["used"] = checked_at_depth
            if verified["rdkit_sanitized"] and verified["fragment_count"] == 1:
                witness(bonds, edits, verified)
                result["search"]["termination"] = "verified_witness_found"
                return result
            if len(next_frontier) < beam_width:
                next_frontier.append((bonds, edits))
        frontier = next_frontier
    result["search"]["termination"] = "edit_budget_exhausted"
    return result
