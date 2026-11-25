"""Count carbonyl families in real molecules and raw VAE draws, including verified cyclic urea precursors."""

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from rdkit import Chem, rdBase

from experiments.vae.training_data_eda.chemistry_profile import GROUP_DEFINITIONS


QUERIES = {k: Chem.MolFromSmarts(GROUP_DEFINITIONS[k][1]) for k in ("amide", "sulfonamide", "nitrile")}
# Atomic numbers include aromatic resonance representations; no tautomer enumeration.
QUERIES.update(
    urea=Chem.MolFromSmarts("[O;D1]=[#6;D3](~[#7;+0])~[#7;+0]"),
    carbamate=Chem.MolFromSmarts("[O;D1]=[#6;D3](~[#8;D2;+0])~[#7;+0]"),
    lactam=Chem.MolFromSmarts("[O;D1]=[#6;D3](@[#6])@[#7;+0]"),
)
PRECURSOR = Chem.MolFromSmarts("[O;D1;H1;+0]-[C;H1;+0](@[#7;+0])@[#7;+0]")


def urea_centers(mol):
    cyclic, other = set(), set()
    for _, c, n1, n2 in mol.GetSubstructMatches(QUERIES["urea"]):
        if all(mol.GetBondBetweenAtoms(c, n).IsInRing() for n in (n1, n2)):
            cyclic.add(c)
        else:
            other.add(c)
    return cyclic, other


def flags(mol):
    cyclic, other = urea_centers(mol)
    found = {name: mol.HasSubstructMatch(query) for name, query in QUERIES.items()}
    found.update(cyclic_urea=bool(cyclic), noncyclic_urea=bool(other))
    # Every cyclic carbonyl here has only two ring neighbors, both nitrogens.
    ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings() if cyclic.intersection(r)]
    found["cyclic_urea_5_or_6_membered"] = any(n in (5, 6) for n in ring_sizes)
    return found


def cyclic_urea_repair_sizes(mol):
    """Return perceived ring sizes at verified OH→carbonyl edit sites."""
    sizes = set()
    for o, c, *_ in mol.GetSubstructMatches(PRECURSOR):
        edit = Chem.RWMol(mol)
        edit.GetBondBetweenAtoms(c, o).SetBondType(Chem.BondType.DOUBLE)
        if Chem.SanitizeMol(edit, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            continue
        if any(atom.GetNumRadicalElectrons() for atom in edit.GetAtoms()):
            continue
        if c in urea_centers(edit)[0] and Chem.MolFromSmiles(Chem.MolToSmiles(edit)) is not None:
            sizes.update(len(ring) for ring in edit.GetRingInfo().AtomRings() if c in ring)
    return sizes


def can_complete(mol):
    return bool(cyclic_urea_repair_sizes(mol))


def count(rows, inspect_precursors, by_size=None):
    total = Counter()
    for smiles in rows:
        total["attempts"] += 1
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        total["valid"] += 1
        found = flags(mol)
        total.update(found)
        if by_size is not None:
            size = by_size.setdefault(mol.GetNumHeavyAtoms(), Counter())
            size.update(found)
            size["valid"] += 1
        if inspect_precursors and not found["cyclic_urea"]:
            sizes = cyclic_urea_repair_sizes(mol)
            total["verified_cyclic_urea_OH_precursor"] += bool(sizes)
            total["verified_5_or_6_membered_OH_precursor"] += bool(sizes & {5, 6})
    return dict(total)


def size_matched_reference(reference, generated):
    """Expected motif counts at the heavy-atom sizes of the valid generated draws."""
    groups = tuple(QUERIES) + ("cyclic_urea", "cyclic_urea_5_or_6_membered", "noncyclic_urea")
    expected = dict.fromkeys(groups, 0.0)
    matched = 0
    for atoms, population in generated.items():
        if atoms not in reference:
            continue
        rows = reference[atoms]
        matched += population["valid"]
        for name in groups:
            expected[name] += population["valid"] * rows[name] / rows["valid"]
    return {"matched_valid_draws": matched, "expected_molecules": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--raw", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(rdkit=rdBase.rdkitVersion, populations={}, size_matched_training={})
    train_sizes = {}
    with rdBase.BlockLogs():
        if args.train:
            with gzip.open(args.train, "rt", encoding="utf8") as stream:
                result["populations"]["training_sample"] = count(
                    (r["smiles"] for r in csv.DictReader(stream, delimiter="\t")), False, train_sizes)
        for path in args.raw:
            panels = {}
            targets = []
            with gzip.open(path, "rt", encoding="utf8") as stream:
                for line in stream:
                    item = json.loads(line)
                    panels.setdefault(item["panel"], []).append(item["smiles"] if item["valid"] else None)
                    if item["panel"] == "reconstruction":
                        targets.append(item["target"])
            for panel, rows in panels.items():
                key = f"{path.name}/{panel}"
                sizes = {}
                result["populations"][key] = count(rows, True, sizes)
                if train_sizes:
                    result["size_matched_training"][key] = size_matched_reference(train_sizes, sizes)
            if targets:
                result["populations"][f"{path.name}/real_targets"] = count(targets, False)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
