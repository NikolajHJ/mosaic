"""Stage 2 valence projection as an extra detached repair proposal."""

import time

import numpy as np

from mosaic.small_molecule.chemistry._reference.probability_repair import measure_probability_edits
from mosaic.small_molecule.chemistry._reference.quality import graph_identity
from mosaic.small_molecule.chemistry.structural import search_structural_repair


POLICY = "stage2-valence-proposal-v1"


def choose_projected_repair(raw, probabilities, spec, search, structural, ordinary):
    from stage2_operators import BOND_STATES, ELEMENTS, Stage2Config, project_valence

    if spec.atomic_numbers != (0, 6, 7, 8, 16, 9, 17) or spec.formal_charges != (0,) or (
        ("NO_BOND", *spec.bond_names[1:]) != BOND_STATES or len(ELEMENTS) != 6
    ) or not np.all(raw[0] > 0):
        raise ValueError("Stage 2 proposal requires the matched fixed-occupancy neutral vocabulary")
    before = ordinary["best_observed_weighted_cost"]
    note = {"policy": POLICY, "ordinary_cost": before, "projected_cost": None, "selected": False}
    if before is not None and before <= 1:
        return ordinary, {**note, "status": "ordinary_at_edit_cost_lower_bound"}

    started = time.perf_counter()
    i, j = np.triu_indices(spec.max_nodes, 1)
    edges = probabilities[2][i, j]
    atom_labels = np.eye(len(ELEMENTS), dtype=edges.dtype)[raw[0] - 1]
    # Sharpen the proposal input to expose over-valence in the hard readout.
    # This mixture is never passed to the live scorer or used to measure cost.
    hard = np.eye(len(BOND_STATES), dtype=edges.dtype)[raw[2][i, j]]
    try:
        projected, projection = project_valence(
            atom_labels, .5 * (edges + hard), config=Stage2Config(readout_caps=True)
        )
    except ArithmeticError as error:
        return ordinary, {**note, "status": "projection_numerical_miss", "error": str(error),
                          "seconds": time.perf_counter() - started}
    note["projection"] = projection
    if not projection["active"]:
        return ordinary, {**note, "status": "inactive", "seconds": time.perf_counter() - started}
    if not np.isfinite(projected).all():
        raise FloatingPointError("Nonfinite projected repair probabilities")

    graph_bonds = raw[2].copy()
    graph_bonds[i, j] = graph_bonds[j, i] = projected.argmax(-1)
    projected_graph = raw[0], raw[1], graph_bonds
    projected_bonds = probabilities[2].copy()
    projected_bonds[i, j] = projected_bonds[j, i] = projected
    result = search_structural_repair(
        projected_graph, (*probabilities[:2], projected_bonds), spec, search, structural
    )
    note.update(status=result["status"], search_checks=result["checks"],
                seconds=time.perf_counter() - started)
    if result["status"] == "already_valid":
        from mosaic.small_molecule.chemistry._reference.chemistry import inspect_graph

        target = projected_graph
        smiles = inspect_graph(*target, spec).smiles
    elif result["status"] == "witness_found":
        found = result["weighted_witness"]
        target = tuple(np.asarray(found["labels"][k]) for k in ("atoms", "charges", "bonds"))
        smiles = found["smiles"]
    else:
        return ordinary, note

    cost = measure_probability_edits(raw, target, probabilities, spec, search)
    if cost["edit_counts"]["atom"] or cost["edit_counts"]["charge"] or not smiles:
        raise ValueError("Projected repair changed atom labels or lost its validity proof")
    value = cost["weighted_cost"]
    note.update(projected_cost=value, projected_edits=cost["edit_count"])
    if before is not None and value >= before:
        return ordinary, note

    note["selected"] = True
    witness = {**cost, "graph_id": graph_identity(*target, spec), "smiles": smiles,
               "labels": {k: v.tolist() for k, v in zip(("atoms", "charges", "bonds"), target)}}
    return {**ordinary, "status": "witness_found", "weighted_witness": witness,
            "best_observed_weighted_cost": value}, note


def smoke_projection():
    """Exercise the actual solver, RDKit validation and original-cost selection."""
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec

    spec = mosaic_spec(4)
    atoms = np.ones(4, dtype=np.int64)
    charges = np.zeros(4, dtype=np.int64)
    bonds = np.zeros((4, 4), dtype=np.int64)
    bonds[0, 1:] = bonds[1:, 0] = (3, 1, 1)
    raw = (atoms, charges, bonds)
    probabilities = (
        np.eye(len(spec.atomic_numbers))[atoms], np.ones((4, 1)),
        .9 * np.eye(len(spec.bond_names))[bonds] + .1 / len(spec.bond_names),
    )
    search = ProbabilityRepairConfig()
    ordinary = {"status": "censored", "best_observed_weighted_cost": None}
    accepted, note = choose_projected_repair(raw, probabilities, spec, search, None, ordinary)
    witness = accepted.get("weighted_witness")
    if not note["selected"] or not witness or witness["edit_count"] < 1:
        raise ValueError("Projection fixture did not produce a valid bond-only repair")
    kept, tie = choose_projected_repair(raw, probabilities, spec, search, None, accepted)
    if kept is not accepted or tie["selected"] or tie["projected_cost"] != witness["weighted_cost"]:
        raise ValueError("Projection tie changed the ordinary repair target")
    return {"status": "passed", "checks": ["validated_bond_only_witness", "original_cost_tie_retains_target"]}
