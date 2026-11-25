"""Measure one-query DeFoG responses to known single-bond corruptions."""

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from mosaic_defog import DeFoGTeacher
from mosaic_defog.graph import ELEMENTS, Graph


BOND_TYPES = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)


def graph_readout(atoms, bonds, pairs):
    mol = Chem.RWMol()
    for element in atoms:
        mol.AddAtom(Chem.Atom(ELEMENTS[element]))
    for (i, j), label in zip(pairs, bonds):
        if label:
            mol.AddBond(int(i), int(j), BOND_TYPES[label])
            if label == 4:
                mol.GetAtomWithIdx(int(i)).SetIsAromatic(True)
                mol.GetAtomWithIdx(int(j)).SetIsAromatic(True)
    components = len(Chem.GetMolFrags(mol))
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError) as error:
        return {"components": components, "sanitized": False, "valid": False,
                "error": str(error), "smiles": None}
    return {"components": components, "sanitized": True, "valid": components == 1,
            "smiles": Chem.MolToSmiles(mol)}


def changed(labels, index, label):
    result = labels.copy()
    result[index] = label
    return result


def softened(labels, categories):
    return 0.98 * np.eye(categories, dtype=np.float32)[labels] + np.float32(0.02 / categories)


def test_input(teacher, atoms, bonds, pairs, corruption=None):
    x, e = softened(atoms, 6), softened(bonds, 5)
    q = teacher.predict(Graph(x, e), time=0.8)
    steps = {}
    for fraction in (0., 0.1, 0.25, 0.5, 0.75, 1.):
        nx, ne = (x + fraction * (q.X - x)).argmax(1), (e + fraction * (q.E_upper - e)).argmax(1)
        steps[str(fraction)] = graph_readout(nx, ne, pairs) | {
            "atom_edits": int(np.count_nonzero(nx != atoms)),
            "bond_edits": int(np.count_nonzero(ne != bonds))}
    result = {"input_bonds": bonds.tolist(), "input_atoms": atoms.tolist(),
              "target_bonds": q.E_upper.tolist(), "target_atoms": q.X.tolist(), "steps": steps}
    if corruption:
        i, original_label = corruption
        edits = np.argwhere(np.arange(5)[None, :] != bonds[:, None])
        delta = q.E_upper - e
        scores = delta[edits[:, 0], edits[:, 1]] - delta[edits[:, 0], bonds[edits[:, 0]]]
        # A repair can restore the known edge or use another verified one-edge edit.
        verified = [bool(graph_readout(atoms, changed(bonds, j, label), pairs)["valid"])
                    for j, label in edits]
        best = scores[verified].max()
        original_index = np.flatnonzero((edits[:, 0] == i) & (edits[:, 1] == original_label)).item()
        if not verified[original_index]:
            raise ValueError("Known repair failed to restore a valid molecule")
        result.update(corrupted_pair=pairs[i].tolist(), original_label=int(original_label),
                      restoring_probability_gain=float(delta[i, original_label]),
                      restoring_margin_gain=float(scores[original_index]),
                      restoring_rank=1 + int((scores > scores[original_index]).sum()),
                      best_verified_rank=1 + int((scores > best).sum()),
                      verified_edit_count=sum(verified),
                      edit_count=len(edits), rank_baseline_expected=(len(edits) + 1) / (sum(verified) + 1))
    return result


def run(args):
    rows, parents = [], set()
    assets = json.loads(args.assets.read_text())
    run_prefixes = ("defog-w0p5", "repair-w1")
    with DeFoGTeacher(assets["teacher_source"], assets["teacher_checkpoint"],
                      python=assets["teacher_python"], threads=2) as teacher, rdBase.BlockLogs():
        for seed in (17, 23, 42):
            for prefix in run_prefixes:
                folder = args.campaign / "runs" / f"{prefix}__fragmented__seed-{seed:05d}"
                with np.load(folder / "top_valid.npz", allow_pickle=False) as top:
                    atom_rows = top["hard_X"].argmax(-1)
                    bond_rows = top["hard_E_upper"].argmax(-1)
                selected = 0
                for rank, (atoms, bonds) in enumerate(zip(atom_rows, bond_rows), 1):
                    pairs = np.array(np.triu_indices(len(atoms), 1)).T
                    parent = graph_readout(atoms, bonds, pairs)
                    if len(atoms) != 25 or not parent["valid"]:
                        raise ValueError("Selected archive must contain valid connected 25-atom parents")
                    if parent["smiles"] in parents:
                        continue
                    parents.add(parent["smiles"])
                    metadata = {"seed": seed, "method": prefix, "parent_rank": rank,
                                "parent_smiles": parent["smiles"]}
                    rows.append(metadata | {"case": "clean"} | test_input(teacher, atoms, bonds, pairs))
                    singles = np.flatnonzero(bonds == 1)
                    rng = np.random.default_rng(np.random.SeedSequence([seed, rank, len(parents)]))
                    for name, replacement in (("deleted_bridge", 0), ("valence_violation", 2)):
                        for i in rng.permutation(singles):
                            corrupted = changed(bonds, i, replacement)
                            readout = graph_readout(atoms, corrupted, pairs)
                            if name == "deleted_bridge":
                                eligible = readout["components"] == 2 and readout["sanitized"]
                            else:
                                eligible = (readout["components"] == 1 and not readout["sanitized"]
                                            and "valence" in readout["error"].lower())
                            if eligible:
                                rows.append(metadata | {"case": name}
                                            | test_input(teacher, atoms, corrupted, pairs, (i, bonds[i])))
                                break
                        else:
                            rows.append(metadata | {"case": name, "status": "no_eligible_edit"})
                    selected += 1
                    if selected == 2:
                        break
                if selected != 2:
                    raise ValueError(f"Too few distinct parents in {folder}")
        result = {"teacher": teacher.identity, "frozen": teacher.verify_frozen(),
                  "time": 0.8, "input_uniform_mass": 0.02, "cases": rows}
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("assets", "campaign", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    run(parser.parse_args())
