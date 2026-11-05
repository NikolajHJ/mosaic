"""Compare native outcomes and exact detached targets on committed trajectories."""

import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture_target(prepared):
    """Record only labels and identity relevant to repair comparison.

    Preparation identity includes probabilities and provenance, so its changes
    are unsuitable for counting target flips. The arm uses one fully occupied
    neutral molecule; route 1 and 2 are its complete verified targets.
    """
    row, = prepared.receipt["rows"]
    route = int(prepared.arrays["route"][0])
    require(route in range(5) and row["route"] == route, "Invalid repair route")
    record = {"preparation_identity": prepared.identity, "route": route}
    if route in (1, 2):
        atoms = prepared.arrays["target_atoms"][0].tolist()
        charges = prepared.arrays["target_charges"][0].tolist()
        matrix = prepared.arrays["target_bonds"][0].tolist()
        n = len(atoms)
        require(n > 0 and len(charges) == len(matrix) == n and all(atoms)
                and not any(charges) and all(len(line) == n for line in matrix),
                "Expected a complete fixed-occupancy neutral target")
        require(all(matrix[i][i] == 0 and
                    all(matrix[i][j] == matrix[j][i] for j in range(i + 1, n))
                    for i in range(n)), "Target bond matrix is not symmetric")
        bonds = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        record.update(atoms=atoms, bonds_sha256=hashlib.sha256(
            json.dumps(bonds, separators=(",", ":")).encode()).hexdigest())
    return record


def target_transitions(rows):
    """Count flips between consecutive update sources with active full targets."""
    counts = {
        "update_sources": len(rows),
        "adjacent_source_pairs": max(0, len(rows) - 1),
        "complete_targets": sum(row["target"]["route"] in (1, 2) for row in rows),
        "active_complete_targets": sum(row["active_complete"] for row in rows),
        "active_atom_swap_targets": sum(
            row["active_complete"] and bool(row["atom_swap_slots"]) for row in rows),
        "active_atom_swap_slots": sum(
            len(row["atom_swap_slots"]) for row in rows if row["active_complete"]),
        "consecutive_active_pairs": 0, "target_flips": 0,
        "atom_only_flips": 0, "bond_only_flips": 0, "atom_and_bond_flips": 0,
        "target_activations": 0, "target_deactivations": 0,
    }
    for before, after in zip(rows, rows[1:]):
        was_active, is_active = before["active_complete"], after["active_complete"]
        counts["target_activations"] += int(not was_active and is_active)
        counts["target_deactivations"] += int(was_active and not is_active)
        if was_active and is_active:
            counts["consecutive_active_pairs"] += 1
            old, new = before["target"], after["target"]
            atom = old["atoms"] != new["atoms"]
            bond = old["bonds_sha256"] != new["bonds_sha256"]
            counts["target_flips"] += int(atom or bond)
            counts["atom_only_flips"] += int(atom and not bond)
            counts["bond_only_flips"] += int(bond and not atom)
            counts["atom_and_bond_flips"] += int(atom and bond)
    pairs = counts["consecutive_active_pairs"]
    counts["flip_fraction"] = counts["target_flips"] / pairs if pairs else None
    return counts


def read_committed(path):
    with path.open("rb") as stream:
        records = []
        for line in stream:
            require(line.endswith(b"\n"), f"Incomplete JSONL row in {path.name}")
            record = json.loads(line)
            require(isinstance(record, dict), f"Invalid JSONL object in {path.name}")
            records.append(record)
    return records


def score(value):
    require(type(value) in (float, int) and math.isfinite(value), "Nonfinite contact score")
    return float(value)


def raw_scope(readout, spec):
    from mosaic.small_molecule.chemistry.environment_policy import assess_environment

    n = spec.max_nodes
    atoms = [v + 1 for v in readout["element_indices"]]
    upper = readout["bond_upper_indices"]
    require(len(atoms) == n and len(upper) == n * (n - 1) // 2 and spec.formal_charges == (0,),
            "Hard readout shape or neutral schema changed")
    bonds = [[0] * n for _ in range(n)]
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            bonds[i][j] = bonds[j][i] = upper[index]
            index += 1
    assessed, = assess_environment([atoms], [[0] * n], [bonds], spec)
    within_scope = all(assessed[key] for key in (
        "fully_assessed", "local_valence_ok", "within_common_organic_scope",
    ))
    return atoms, within_scope, bool(assessed["sulfur_scope_exclusions"])


def _validity(rows):
    return {
        "count": len(rows),
        "native_valid_connected": sum(row["native_valid_connected"] for row in rows),
        "scoped_valid_connected": sum(row["scoped_valid_connected"] for row in rows),
        "sulfur_scope_excluded": sum(row["sulfur_scope_excluded"] for row in rows),
        "distinct_native_valid": len({row["smiles"] for row in rows if row["native_valid_connected"]}),
        "distinct_scoped_valid": len({row["smiles"] for row in rows if row["scoped_valid_connected"]}),
    }


def _contact(rows, field):
    return {
        "first": rows[0][field],
        "last": rows[-1][field],
        "best": max(row[field] for row in rows),
        "best_native_valid": max(
            (row[field] for row in rows if row["native_valid_connected"]), default=None),
        "best_scoped_valid": max(
            (row[field] for row in rows if row["scoped_valid_connected"]), default=None),
    }


def summarize_trajectory(study, targets, spec, *, expected_updates, hard_contacts=None):
    """Join exact target labels to committed source states and their updates.

    The final state participates in validity and contact statistics. Its target
    never participates in update-source flip statistics since no update uses it.
    """
    study = Path(study)
    states = read_committed(study / "states.jsonl")
    updates = read_committed(study / "updates.jsonl")
    require(type(expected_updates) is int and expected_updates > 0
            and len(updates) == expected_updates and len(states) == expected_updates + 1,
            "Committed state and update counts differ")
    start = states[0]["global_step"]
    steps = list(range(start, start + len(states)))
    require([s["global_step"] for s in states] == steps
            and [u["global_step"] for u in updates] == steps[:-1]
            and set(targets) == set(steps), "States, targets and updates do not cover the same horizon")
    for state, update in zip(states, updates):
        require(state["preparation_identity"] == update["preparation_identity"]
                and state["repair_weight"] == update["repair_weight"],
                "An update used another preparation or weight")

    rows = []
    for state in states:
        step = state["global_step"]
        target = targets[step]
        require(target["preparation_identity"] == state["preparation_identity"]
                and target["route"] == state["chemistry_preparation"][0]["route"],
                "Captured target differs from saved state")
        raw = state["readout"]
        raw_atoms, within_scope, sulfur_excluded = raw_scope(raw, spec)
        native = raw["sanitized"] is True and raw["disconnected"] is False
        require(native == state["strict_valid_connected"]
                and (not native or isinstance(raw["canonical_smiles"], str)),
                "Native validity differs from committed state")
        complete = target["route"] in (1, 2)
        weight = score(state["repair_weight"])
        require(weight >= 0, "Unexpected negative repair weight")
        swap_slots = []
        if complete:
            swap_slots = [i for i, (old, new) in enumerate(zip(raw_atoms, target["atoms"], strict=True))
                          if old != new]
            require(len(swap_slots) <= 1, "Matched exchange exceeded one atom")
        rows.append({
            "global_step": step, "target": target, "active_complete": complete and weight > 0,
            "atom_swap_slots": swap_slots,
            "native_valid_connected": native, "scoped_valid_connected": native and within_scope,
            "sulfur_scope_excluded": sulfur_excluded, "smiles": raw["canonical_smiles"],
            "relaxed_contact_score": -score(state["contact_loss"]),
        })
    summary = {
        "global_start": steps[0], "global_end": steps[-1], "applied_updates": expected_updates,
        "validity": _validity(rows), "update_source_validity": _validity(rows[:-1]),
        "repair_target_churn": target_transitions(rows[:-1]),
        "relaxed_contact": _contact(rows, "relaxed_contact_score"),
        "rows": rows,
    }
    if hard_contacts is not None:
        require(set(hard_contacts) <= set(steps) and {steps[0], steps[-1]} <= set(hard_contacts),
                "Hard-score grid must include saved endpoints")
        hard_rows = []
        for row in rows:
            step = row["global_step"]
            if step in hard_contacts:
                hard_rows.append({
                    "global_step": step, "hard_contact_score": score(hard_contacts[step]),
                    "native_valid_connected": row["native_valid_connected"],
                    "scoped_valid_connected": row["scoped_valid_connected"],
                })
        summary["hard_contact"] = {
            **_contact(hard_rows, "hard_contact_score"), "forwards": len(hard_rows), "points": hard_rows,
        }
    return summary
