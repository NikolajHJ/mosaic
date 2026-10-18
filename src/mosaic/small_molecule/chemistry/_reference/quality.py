"""Detached reference extraction; see source_manifest.json. Do not edit manually."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from rdkit import Chem, rdBase
from .chemistry import inspect_graph
from .representation import GraphSpec


VERSION = "graded-chemistry-v2"


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "iu":
        raise ValueError("Hard graph labels must be integers")
    return result.astype(np.int64, copy=True)


def _labels(atoms, charges, bonds, spec):
    """Validate shape and active labels, matching the existing occupancy contract."""
    a, q, b = map(_array, (atoms, charges, bonds))
    if a.ndim != 1 or q.shape != a.shape or b.shape != (len(a), len(a)) or not 0 < len(a) <= spec.max_nodes:
        raise ValueError("Hard graph shape differs from schema")
    if np.any(a < 0) or np.any(a >= len(spec.atomic_numbers)):
        raise ValueError("Atom category outside schema")
    active = a != 0
    pair = active[:, None] & active[None, :]
    if (np.any(q[active] < 0) or np.any(q[active] >= len(spec.formal_charges))
            or np.any(b[pair] < 0) or np.any(b[pair] >= len(spec.bond_names))):
        raise ValueError("Active charge/bond category outside schema")
    # Padding channels have no chemical identity in our decoder's sampling law.
    q[~active] = spec.formal_charges.index(0)
    b[~pair] = 0
    return a, q, b


def graph_identity(atoms, charges, bonds, spec) -> str:
    """Hash exact slot-ordered labels, including otherwise ignored padding."""
    _labels(atoms, charges, bonds, spec)
    labels = tuple(_array(value) for value in (atoms, charges, bonds))
    payload = {"schema": spec.fingerprint(), "labels": [x.tolist() for x in labels]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _components(active, bonds):
    remaining, sizes = set(np.flatnonzero(active).tolist()), []
    while remaining:
        todo, size = [remaining.pop()], 0
        while todo:
            node = todo.pop()
            size += 1
            neighbors = {j for j in remaining if bonds[node, j] != 0 or bonds[j, node] != 0}
            remaining -= neighbors
            todo.extend(neighbors)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def diagnose_graph(atoms, charges, bonds, spec: GraphSpec) -> dict:
    """Measure a single hard attempt without correcting or rejecting chemistry.

    local_consistency = 1/(1 + problem burden). Burden counts distinct atoms
    implicated by RDKit diagnostics (or one for a global problem), plus radicals
    and explicit label/symmetry defects. It is not a graph edit distance. Unknown
    construction failures are unavailable, never implicitly perfect. Low valence
    is not penalized: hydrogens are implicit in this representation.
    """
    inspected = inspect_graph(atoms, charges, bonds, spec)
    result = {"version": VERSION, "valid": inspected.smiles is not None,
              "smiles": inspected.smiles, "failure_reason": inspected.failure_reason,
              "failure_detail": inspected.detail, "schema_supported": False,
              "local_consistency": None, "connectedness": None}
    try:
        a, q, b = _labels(atoms, charges, bonds, spec)
    except (ValueError, TypeError, IndexError) as exc:
        return {**result, "diagnostic_unavailable_reason": str(exc)}
    active = np.flatnonzero(a).tolist()
    raw_q, raw_b = _array(charges), _array(bonds)
    padding = a == 0
    padding_charges = int(np.count_nonzero(raw_q[padding] != spec.formal_charges.index(0)))
    padding_bonds = int(np.count_nonzero(raw_b[padding[:, None] | padding[None, :]]))
    sizes = _components(a != 0, b)
    symmetry = int(np.count_nonzero(np.triu(b != b.T, 1)))
    self_bonds = int(np.count_nonzero(np.diag(b)))
    result.update(schema_supported=True, graph_id=graph_identity(atoms, charges, bonds, spec),
                  heavy_atoms=len(active), component_sizes=sizes, component_count=len(sizes),
                  total_formal_charge=sum(spec.formal_charges[int(q[i])] for i in active),
                  asymmetric_pairs=symmetry, self_bonds=self_bonds,
                  nonneutral_padding_slots=padding_charges, nonzero_padding_bond_entries=padding_bonds,
                  categorical_support_labels_ok=not (padding_charges or padding_bonds or symmetry or self_bonds),
                  connectedness=(1 / len(sizes) if sizes else 0.0))
    if not active:
        return {**result, "local_consistency": 0.0, "problem_burden": 1,
                "problems": [{"type": "empty", "slots": []}],
                "valence_checked_atoms": 0, "valence_coverage": None,
                "overvalence_excess": None}
    if symmetry or self_bonds:
        return {**result, "local_consistency": 1 / (1 + symmetry + self_bonds),
                "problem_burden": symmetry + self_bonds,
                "problems": [{"type": "structural_labels", "slots": []}],
                "valence_checked_atoms": 0, "valence_coverage": 0.0,
                "overvalence_excess": None}
    bond_types = {"SINGLE": Chem.BondType.SINGLE, "DOUBLE": Chem.BondType.DOUBLE,
                  "TRIPLE": Chem.BondType.TRIPLE, "AROMATIC": Chem.BondType.AROMATIC}
    try:
        rw = Chem.RWMol()
        for i in active:
            atom = Chem.Atom(spec.atomic_numbers[int(a[i])])
            atom.SetFormalCharge(spec.formal_charges[int(q[i])])
            rw.AddAtom(atom)
        for left, i in enumerate(active):
            for right in range(left + 1, len(active)):
                name = spec.bond_names[int(b[i, active[right]])]
                if name == "NONE":
                    continue
                rw.AddBond(left, right, bond_types[name])
                if name == "AROMATIC":
                    rw.GetAtomWithIdx(left).SetIsAromatic(True)
                    rw.GetAtomWithIdx(right).SetIsAromatic(True)
        mol = rw.GetMol()
        with rdBase.BlockLogs():
            detected = Chem.DetectChemistryProblems(mol)
        problems, implicated, global_count = [], set(), 0
        for problem in detected:
            ids = ([problem.GetAtomIdx()] if hasattr(problem, "GetAtomIdx") else
                   list(problem.GetAtomIndices()) if hasattr(problem, "GetAtomIndices") else [])
            slots = [active[i] for i in ids]
            implicated.update(slots)
            global_count += int(not slots)
            problems.append({"type": problem.GetType(), "slots": slots})
        # A sanitized graph may still contain radicals, which our schema rejects.
        with rdBase.BlockLogs():
            sanitized = Chem.Mol(mol)
            flag = Chem.SanitizeMol(sanitized, catchErrors=True)
        if int(flag) == 0:
            radical_slots = [active[x.GetIdx()] for x in sanitized.GetAtoms()
                             if x.GetNumRadicalElectrons()]
            implicated.update(radical_slots)
            if radical_slots:
                problems.append({"type": "radical", "slots": radical_slots})
        elif not detected:
            # Keep unknown sanitize failures as one observable problem.
            global_count += 1
            problems.append({"type": "sanitization_other", "slots": []})
        burden = len(implicated) + global_count
        result.update(problems=problems, problem_burden=burden,
                      local_consistency=1 / (1 + burden))
    except (ValueError, RuntimeError, KeyError, OverflowError) as exc:
        result.update(diagnostic_unavailable_reason=type(exc).__name__)
    # Supplementary severity only: neutral elements use RDKit's finite allowed
    # valences. A small explicit charged table and no aromatic approximation.
    charged_caps = {(7, 1): 4, (7, -1): 2, (8, 1): 3, (8, -1): 1,
                    (6, 1): 3, (6, -1): 3}
    orders = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
    checks = []
    for i in active:
        names = [spec.bond_names[int(b[i, j])] for j in active]
        if any(name not in orders for name in names):
            continue
        element, charge = spec.atomic_numbers[int(a[i])], spec.formal_charges[int(q[i])]
        try:
            allowed = list(Chem.GetPeriodicTable().GetValenceList(element)) if charge == 0 else []
        except (ValueError, RuntimeError, OverflowError):
            continue
        cap = max(allowed) if allowed and min(allowed) >= 0 else charged_caps.get((element, charge))
        if cap is None:
            continue
        valence = sum(orders[name] for name in names)
        checks.append({"slot": i, "bond_order_sum": valence, "cap": cap,
                       "excess": max(0, valence - cap)})
    result.update(valence_checks=checks, valence_checked_atoms=len(checks),
                  valence_coverage=len(checks) / len(active),
                  overvalence_excess=sum(x["excess"] for x in checks) if checks else None)
    return result
