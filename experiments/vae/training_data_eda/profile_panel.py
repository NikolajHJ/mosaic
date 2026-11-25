"""Profile a uniform panel from the published VAE training view. See experiment.md."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors as D
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemistry_profile import molecule_profile, summarize_profiles


ELEMENTS = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
FEATURES = (
    "heavy_atoms", "molecular_weight", "logP", "TPSA", "H_bond_donors", "H_bond_acceptors",
    "rotatable_bonds", "fraction_C_sp3", "aromatic_atom_fraction", "ring_count",
    "heterocycles", "largest_ring_size", "branch_atom_fraction", "nitrogen_atoms",
    "oxygen_atoms", "sulfur_atoms", "halogen_atoms", "charged_atoms",
)
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=False)


def measure(args):
    view_row, heavy_atoms, cycle_rank, smiles, fingerprint = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Invalid or disconnected training molecule at view row {view_row}")
    row = molecule_profile(mol)
    rank = mol.GetNumBonds() - mol.GetNumAtoms() + 1
    if (row["heavy_atoms"] != heavy_atoms or rank != cycle_rank
            or (rank > 0) != (row["ring_count"] > 0)):
        raise ValueError(f"Mismatch between packed graph and RDKit at view row {view_row}")
    atoms = list(mol.GetAtoms())
    counts = Counter(a.GetSymbol() for a in atoms)
    if counts.keys() - set(ELEMENTS):
        raise ValueError(f"Element outside the graph schema at view row {view_row}")
    row.update({f"n_{el}": counts[el] for el in ELEMENTS})
    row.update(
        view_row=view_row, cycle_rank=rank, molecular_weight=Descriptors.MolWt(mol),
        logP=Crippen.MolLogP(mol), TPSA=D.CalcTPSA(mol), H_bond_donors=D.CalcNumHBD(mol),
        H_bond_acceptors=D.CalcNumHBA(mol), rotatable_bonds=D.CalcNumRotatableBonds(mol),
        fraction_C_sp3=D.CalcFractionCSP3(mol),
        aromatic_atom_fraction=sum(a.GetIsAromatic() for a in atoms) / heavy_atoms,
        branch_atom_fraction=sum(a.GetDegree() >= 3 for a in atoms) / heavy_atoms,
        nitrogen_atoms=counts["N"], oxygen_atoms=counts["O"], sulfur_atoms=counts["S"],
        halogen_atoms=sum(counts[el] for el in ("F", "Cl", "Br", "I")),
        charged_atoms=sum(a.GetFormalCharge() != 0 for a in atoms),
        net_charge=Chem.GetFormalCharge(mol), spiro_atoms=D.CalcNumSpiroAtoms(mol),
        bridgehead_atoms=D.CalcNumBridgeheadAtoms(mol),
        scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if rank else "",
    )
    if rank and not row["scaffold"]:
        raise ValueError(f"Ring-bearing row {view_row} has an empty scaffold")
    bits = np.packbits(FP.GetFingerprintAsNumPy(mol)).tobytes() if fingerprint else None
    return row, bits


def profile(panel_path, output, workers, seed, fingerprint_count):
    from dimensions import analyze_dimensions

    panel = pd.read_csv(panel_path, sep="\t", keep_default_na=False)
    if not len(panel) or any(panel[k].duplicated().any() for k in ("view_row", "parent_row", "smiles")):
        raise ValueError("Expected a nonempty panel of distinct training graphs")
    fingerprint_rows = set(np.random.default_rng(seed + 1).choice(
        len(panel), size=min(len(panel), fingerprint_count), replace=False))
    tasks = ((r.view_row, r.heavy_atoms, r.cycle_rank, r.smiles, i in fingerprint_rows)
             for i, r in enumerate(panel.itertuples(index=False)))
    output.mkdir(parents=True, exist_ok=False)
    rows, fingerprints = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row, bits in pool.map(measure, tasks, chunksize=128):
            rows.append(row)
            if bits is not None:
                fingerprints.append(np.frombuffer(bits, dtype=np.uint8))
    df = pd.DataFrame.from_records(rows)
    del rows
    df["in_fingerprint_panel"] = [i in fingerprint_rows for i in range(len(df))]
    if not np.isfinite(df.select_dtypes("number").to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite panel descriptor or count")
    df.to_csv(output / "profiles.csv.gz", index=False)
    for name, table in summarize_profiles(df).items():
        table.to_csv(output / f"{name}.csv", index=False)
    descriptors = df.loc[:, list(FEATURES)]
    descriptors.describe(percentiles=[.05, .25, .5, .75, .95]).T.to_csv(output / "descriptor_summary.csv")
    descriptors.corr().to_csv(output / "descriptor_correlations.csv")
    pd.DataFrame([dict(element=el, molecules=int(df[f"n_{el}"].gt(0).sum()),
                       pct_molecules=100 * df[f"n_{el}"].gt(0).mean(),
                       pct_atoms=100 * df[f"n_{el}"].sum() / df.heavy_atoms.sum())
                  for el in ELEMENTS]).to_csv(output / "elements.csv", index=False)
    scaf = df.loc[df.scaffold.ne(""), "scaffold"].value_counts().rename_axis("scaffold").rename("n")
    scaf.to_csv(output / "scaffolds.csv.gz")
    bits = np.unpackbits(np.stack(fingerprints), axis=1)
    # Two disjoint random half-panels give independent within-pair graph identities.
    order = np.random.default_rng(seed + 2).permutation(len(bits))
    left, right = np.array_split(order[:(len(order) // 2) * 2], 2)
    intersection = (bits[left] & bits[right]).sum(axis=1)
    union = (bits[left] | bits[right]).sum(axis=1)
    similarity = intersection / union
    if not np.isfinite(similarity).all():
        raise ValueError("Nonfinite pair similarity")
    fp_ids = df.loc[df.in_fingerprint_panel, "view_row"].to_numpy()
    pd.DataFrame(dict(view_row_a=fp_ids[left], view_row_b=fp_ids[right], tanimoto=similarity)
                 ).to_csv(output / "random_pair_similarities.csv.gz", index=False)
    meta = dict(seed=seed, n=len(df), rdkit=rdBase.rdkitVersion, features=FEATURES,
                fingerprint_rows=len(bits), fingerprint_seed=seed + 1, random_pair_seed=seed + 2,
                scaffold_distinct=len(scaf), scaffold_singletons=int(scaf.eq(1).sum()),
                scaffold_top10=int(scaf.head(10).sum()), nonempty_scaffolds=int(scaf.sum()),
                pair_tanimoto_p05_p50_p95=np.quantile(similarity, [.05, .5, .95]).tolist(),
                spiro=int(df.spiro_atoms.gt(0).sum()), bridgehead=int(df.bridgehead_atoms.gt(0).sum()),
                any_charge=int(df.charged_atoms.gt(0).sum()), net_charge=int(df.net_charge.ne(0).sum()))
    (output / "panel_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    analyze_dimensions(descriptors.to_numpy(), list(FEATURES), output, seed=seed, fingerprints=bits)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fingerprints", type=int, default=15000)
    args = parser.parse_args()
    if args.workers < 1 or args.fingerprints < 2:
        parser.error("Use at least one worker and two fingerprints")
    profile(args.panel, args.output, args.workers, args.seed, args.fingerprints)
