"""Prepare a relaxed ligand graph separately from an optimization run."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mosaic.small_molecule.state import (
    BOND_STATES,
    ELEMENTS,
    RelaxedGraph,
    graph_from_hard,
    validate_graph,
)


def make_graph(mol):
    from rdkit import Chem

    mol = Chem.RemoveHs(Chem.Mol(mol))
    atoms = list(mol.GetAtoms())
    if len(atoms) < 2 or any(atom.GetFormalCharge() != 0 for atom in atoms):
        raise ValueError("Use a neutral molecule with at least two heavy atoms")
    try:
        elements = [ELEMENTS.index(atom.GetSymbol()) for atom in atoms]
    except ValueError as error:
        raise ValueError(f"Allowed elements: {', '.join(ELEMENTS)}") from error
    pairs = [(i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))]
    lookup = {pair: index for index, pair in enumerate(pairs)}
    bonds = [0] * len(pairs)
    kinds = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
        Chem.BondType.AROMATIC: "AROMATIC",
    }
    for bond in mol.GetBonds():
        pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if bond.GetBondType() not in kinds:
            raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")
        bonds[lookup[pair]] = BOND_STATES.index(kinds[bond.GetBondType()])
    return graph_from_hard(elements, bonds), mol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles")
    source.add_argument("--sdf", type=Path)
    parser.add_argument("--atom-noise", type=float, default=0.0)
    parser.add_argument("--bond-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not np.isfinite(x) or not 0 <= x <= 1 for x in (args.atom_noise, args.bond_noise)):
        raise ValueError("Noise levels must be between zero and one")
    from rdkit import Chem

    if args.sdf:
        mol = Chem.MolFromMolFile(str(args.sdf), removeHs=True, sanitize=True)
    else:
        mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError("Input molecule could not be read")
    hard, mol = make_graph(mol)
    rng = np.random.default_rng(args.seed)
    atom_noise = rng.random(hard.X.shape, dtype=np.float32)
    bond_noise = rng.random(hard.E_upper.shape, dtype=np.float32)
    atom_noise /= atom_noise.sum(axis=-1, keepdims=True)
    bond_noise /= bond_noise.sum(axis=-1, keepdims=True)
    graph = RelaxedGraph(
        X=(1 - args.atom_noise) * np.asarray(hard.X) + args.atom_noise * atom_noise,
        E_upper=(1 - args.bond_noise) * np.asarray(hard.E_upper) + args.bond_noise * bond_noise,
    )
    validate_graph(graph)
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / "initial_graph.npz"
    np.savez_compressed(artifact, X=np.asarray(graph.X), E_upper=np.asarray(graph.E_upper))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    metadata = {
        "schema": "mosaic-ligand-initialization-v1",
        "graph_file": artifact.name,
        "graph_sha256": digest,
        "smiles": Chem.MolToSmiles(mol, canonical=False),
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "elements_in_atom_order": [atom.GetSymbol() for atom in mol.GetAtoms()],
        "heavy_atoms": mol.GetNumAtoms(),
        "noise": {"atom": args.atom_noise, "bond": args.bond_noise},
        "seed": args.seed,
        "noise_policy": "independent uniform probability rows blended with the hard graph",
    }
    (args.output / "initialization.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {mol.GetNumAtoms()} slots in {args.output}")


if __name__ == "__main__":
    main()
