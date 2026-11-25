"""Build compact offline layouts for saved hard structures and raw argmax graphs."""

import math

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor


def _coords(molecule):
    conformer_id = rdDepictor.Compute2DCoords(molecule, canonOrient=True, forceRDKit=True)
    positions = molecule.GetConformer(conformer_id).GetPositions()
    coords = [(float(x), float(y)) for x, y, _ in positions]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in coords):
        raise ValueError("Nonfinite 2D coordinates")
    return coords


def _normalized(coords, labels):
    if not coords:
        return []
    xs, ys = zip(*coords)
    left, right, bottom, top = min(xs), max(xs), min(ys), max(ys)
    mid_x, mid_y = (left + right) / 2, (bottom + top) / 2
    extent = max(right - left, top - bottom)
    scale = 0.88 / extent if extent else 0.0
    return [
        [round(0.5 + (x - mid_x) * scale, 4), round(0.5 - (y - mid_y) * scale, 4), label, charge]
        for (x, y), (label, charge) in zip(coords, labels)
    ]


def _molecule_structure(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        return None
    labels = [(atom.GetSymbol(), atom.GetFormalCharge()) for atom in molecule.GetAtoms()]
    coords = _coords(molecule)
    bonds = [
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType())]
        for bond in molecule.GetBonds()
    ]
    return {"atoms": _normalized(coords, labels), "bonds": bonds}


def _raw_structure(graph, periodic_table):
    labels = [
        (periodic_table.GetElementSymbol(number) if 1 <= number <= 118 else "?", charge)
        for number, charge in graph["atoms"]
    ]
    bonds = graph["bonds"]
    try:
        layout_graph = Chem.RWMol()
        for number, _ in graph["atoms"]:
            layout_graph.AddAtom(Chem.Atom(number if 1 <= number <= 118 else 0))
        for left, right, _ in bonds:
            layout_graph.AddBond(left, right, Chem.BondType.SINGLE)
        # The temporary graph determines coordinates only. Keep the recorded
        # bond orders and formal charges for display, including invalid ones.
        coords = _coords(layout_graph)
    except (RuntimeError, ValueError):
        count = len(labels)
        coords = [(math.cos(math.tau * i / count), math.sin(math.tau * i / count)) for i in range(count)]
    return {"atoms": _normalized(coords, labels), "bonds": bonds, "raw": True}


def build_structures(snapshot: dict) -> dict[str, object]:
    """Return normalized, y-down 2D graphs indexed by SMILES or raw graph key."""
    smiles_set = set()
    raw_keys = set()
    for job in snapshot.get("jobs", []):
        for archive in job.get("archives") or []:
            for elites in archive.get("elites", {}).values():
                smiles_set.update(elite.get("smiles") for elite in elites)
        for branch in job.get("branches") or []:
            smiles_set.add(branch.get("parent_smiles"))
            for row in branch.get("points") or []:
                if row[4]:
                    smiles_set.add(row[5])
                elif len(row) > 7 and row[7]:
                    raw_keys.add(row[7])
            for candidates in (branch.get("candidates") or {}).values():
                smiles_set.update(row.get("smiles") for row in candidates)

    structures = {}
    periodic_table = Chem.GetPeriodicTable()
    with rdBase.BlockLogs():
        for smiles in sorted(s for s in smiles_set if isinstance(s, str) and s):
            try:
                structure = _molecule_structure(smiles)
            except (RuntimeError, ValueError):
                continue
            if structure is not None:
                structures[smiles] = structure
        raw_graphs = snapshot.get("raw_graphs") or {}
        for key in sorted(raw_keys & raw_graphs.keys()):
            structures[key] = _raw_structure(raw_graphs[key], periodic_table)
    return structures
