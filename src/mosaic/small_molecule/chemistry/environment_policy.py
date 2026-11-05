"""Portable, dependency-free common-organic neighbor policy.

This is a deliberately scoped graph prior, not a claim about all realizable
chemistry. The four-nonfluorine-neighbor sulfur limit intentionally excludes
rare persulfuranes. Fluorine is exempt from that count, NEVER from valence.
Keep this file byte-identical to Mosaic's chemistry/environment_policy.py.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

VERSION = "common-organic-neighbor-v1"
SULFUR_NONFLUORINE_LIMIT = 4
BOND_ORDERS = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1}
# Same charge-aware limits as the historical rule table. None is unsupported,
# not a guessed zero-capacity atom. Charge labels are actual formal charges.
VALENCES = {
    6: {-2: (2,), -1: (3,), 0: (4,), 1: (3,), 2: (2,)},
    7: {-2: (1,), -1: (2,), 0: (3,), 1: (4,), 2: (3,)},
    8: {-2: (0,), -1: (1,), 0: (2,), 1: (3,), 2: (4,)},
    9: {-2: None, -1: (0,), 0: (1,), 1: (2,), 2: (3,)},
    15: {-2: (1, 3, 5), -1: (2, 4, 6), 0: (3, 5), 1: (4,), 2: (3,)},
    16: {-2: (0, 2, 4), -1: (1, 3, 5), 0: (2, 4, 6), 1: (3, 5), 2: (4,)},
    17: {-2: None, -1: (0,), 0: (1,), 1: (2, 4, 6), 2: (3, 5)},
    35: {-2: None, -1: (0,), 0: (1,), 1: (2, 4, 6), 2: (3, 5)},
    53: {-2: None, -1: (0, 2, 4, 6), 0: (1, 3, 5), 1: (2, 4, 6), 2: (3, 5)},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_environment_policy(value):
    if (type(value) is not dict or set(value) != {"version", "weight"}
            or value.get("version") != VERSION
            or type(value.get("weight")) not in (int, float)
            or not math.isfinite(value["weight"]) or value["weight"] < 0):
        raise ValueError("Require explicit common-organic-neighbor-v1 and finite nonnegative weight")
    return deepcopy(value)


def prepare_environment_context(atoms, charges, bonds, spec):
    """Freeze original categorical labels, never a repair witness.

    Neighbor capacity is the maximum allowed valence minus its OTHER hard
    incident bond orders. A neighbor already overfull elsewhere has zero
    remaining capacity: removing this edge is still allowed. This is a local
    bond-risk construction; it does not certify the neighbor's other bonds.
    Aromatic order one is an optimistic bound, not Kekule/aromatic validation.
    """
    if set(spec.bond_names) != set(BOND_ORDERS) or spec.bond_names[0] != "NONE":
        raise ValueError("Require the five explicit bond categories")
    atoms, charges, bonds = deepcopy(atoms), deepcopy(charges), deepcopy(bonds)
    batch, n = len(atoms), spec.max_nodes
    if not batch or len(charges) != batch or len(bonds) != batch:
        raise ValueError("Require aligned nonempty batch")
    elements, formal, caps, active, orders, remaining = [], [], [], [], [], []
    assessments = []
    for row in range(batch):
        if (len(atoms[row]) != n or len(charges[row]) != n or len(bonds[row]) != n
                or any(len(line) != n for line in bonds[row])):
            raise ValueError("Neighbor context shape disagrees with schema")
        for labels, vocabulary in ((atoms[row], spec.atomic_numbers), (charges[row], spec.formal_charges)):
            if any(type(v) is not int or not 0 <= v < len(vocabulary) for v in labels):
                raise ValueError("Context contains invalid categorical labels")
        z = [spec.atomic_numbers[v] for v in atoms[row]]
        q = [spec.formal_charges[v] for v in charges[row]]
        occupied = [v != 0 for v in z]
        cap = []
        for element, charge in zip(z, q):
            allowed = VALENCES.get(element, {}).get(charge)
            cap.append(max(allowed) if allowed else -1)
        edge_orders = []
        for i in range(n):
            line = []
            for j in range(n):
                label = bonds[row][i][j]
                if (type(label) is not int or not 0 <= label < len(spec.bond_names)
                        or label != bonds[row][j][i]):
                    raise ValueError("Context bonds must be symmetric categorical labels")
                if (i == j or not occupied[i] or not occupied[j]) and label != 0:
                    raise ValueError("Context has a ghost or diagonal bond")
                line.append(BOND_ORDERS[spec.bond_names[label]])
            edge_orders.append(line)
        totals = [sum(line) for line in edge_orders]
        residual = [[(-1 if cap[j] < 0 else max(cap[j] - totals[j] + edge_orders[i][j], 0))
                     for j in range(n)] for i in range(n)]
        violations, exclusions, unassessed = [], [], []
        for i in range(n):
            if not occupied[i]:
                continue
            if cap[i] < 0:
                unassessed.append(i)
            elif totals[i] > cap[i]:
                violations.append(dict(slot=i, element=z[i], charge=q[i],
                                       bond_order_sum=totals[i], maximum=cap[i]))
            nonfluorine = sum(int(edge_orders[i][j] > 0 and z[j] != 9) for j in range(n))
            if z[i] == 16 and nonfluorine > SULFUR_NONFLUORINE_LIMIT:
                exclusions.append(dict(slot=i, nonfluorine_neighbors=nonfluorine,
                                       maximum=SULFUR_NONFLUORINE_LIMIT))
        assessments.append(dict(local_valence_ok=not violations,
            within_common_organic_scope=not exclusions, valence_violations=violations,
            sulfur_scope_exclusions=exclusions, unassessed_slots=unassessed,
            fully_assessed=not unassessed, native_chemistry_validation_performed=False))
        elements.append(z); formal.append(q); caps.append(cap); active.append(occupied)
        orders.append(edge_orders); remaining.append(residual)
    return dict(version=VERSION, schema_sha256=spec.fingerprint(), max_nodes=n,
        atoms=elements, charges=formal, caps=caps, active=active, bond_orders=orders,
        neighbor_remaining_caps=remaining, assessments=assessments,
        semantics=dict(atom_charge_occupancy="detached original hard context",
            neighbor_capacity="max(native_charge_cap - other_fixed_bond_orders, 0); NONE always allowed",
            sulfur="at most four nonfluorine heavy neighbors; rare cases deliberately scope-excluded",
            aromatic="order-one optimistic bound; native sanitization remains required",
            limitations="conditional local event, not global validity, stability or synthesis probability"))


def assess_environment(atoms, charges, bonds, spec):
    """Return local defects and scoped exclusions separately from native validity."""
    return prepare_environment_context(atoms, charges, bonds, spec)["assessments"]
