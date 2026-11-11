"""Measure what Stage 1 does to the ligand without changing it.

The soft graph can contain partial atoms and bonds. We track how certain its
probabilities become and how much bond mass it has. We also choose the most
likely atom and bond in each row to see whether that discrete graph would be
a usable molecule. Failed checks are recorded; the graph is never repaired.
"""

import numpy as np

from mosaic.small_molecule.core.state import ELEMENTS, graph_diagnostics, graph_from_hard


# Aligned with ELEMENTS and the stored bond categories, respectively.
VALENCE_LIMIT = {"C": 4, "N": 3, "O": 2, "S": 2, "F": 1, "Cl": 1}
BOND_ORDER = np.array([0, 1, 2, 3, 1.5])


def valence_excess(atom_probs, bond_probs):
    """Compare each atom's bond-order sum with its allowed neutral valence."""
    valence = np.zeros(len(atom_probs))
    i, j = np.triu_indices(len(atom_probs), k=1)
    order = bond_probs @ BOND_ORDER
    np.add.at(valence, i, order)
    np.add.at(valence, j, order)
    limit = atom_probs @ np.array([VALENCE_LIMIT[element] for element in ELEMENTS])
    excess = np.maximum(valence - limit, 0)
    return int(np.count_nonzero(excess > 1e-6)), float(excess.sum())


def graph_metrics(graph):
    x = np.asarray(graph.X)
    edges = np.asarray(graph.E_upper)
    count, excess = valence_excess(x, edges)
    return graph_diagnostics(graph) | {
        "atom_saturated_fraction": float(np.mean(x.max(axis=-1) >= 0.99)),
        "bond_saturated_fraction": float(np.mean(edges.max(axis=-1) >= 0.99)),
        "expected_valence_violation_count": count,
        "expected_valence_excess": excess,
    }


def raw_readout(graph):
    """Read the most likely category per row and check the result in RDKit."""
    from rdkit import Chem

    elements = np.asarray(graph.X).argmax(axis=-1)
    bonds = np.asarray(graph.E_upper).argmax(axis=-1)
    hard_graph = graph_from_hard(elements, bonds)
    count, excess = valence_excess(np.asarray(hard_graph.X), np.asarray(hard_graph.E_upper))

    mol = Chem.RWMol()
    for index in elements:
        mol.AddAtom(Chem.Atom(ELEMENTS[index]))
    bond_types = (
        None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE, Chem.BondType.AROMATIC,
    )
    i, j = np.triu_indices(graph.n_atoms, k=1)
    for a, b, category in zip(i.tolist(), j.tolist(), bonds.tolist(), strict=True):
        if category:
            mol.AddBond(a, b, bond_types[category])
            if category == 4:
                mol.GetAtomWithIdx(a).SetIsAromatic(True)
                mol.GetAtomWithIdx(b).SetIsAromatic(True)

    molecule = mol.GetMol()
    fragments = len(Chem.GetMolFrags(molecule, sanitizeFrags=False))
    smiles, error = None, None
    try:
        Chem.SanitizeMol(molecule)
        smiles = Chem.MolToSmiles(molecule, canonical=True)
    except ValueError as exc:
        error = str(exc)

    return hard_graph, {
        "element_indices": elements.tolist(),
        "bond_upper_indices": bonds.tolist(),
        "fragment_count": fragments,
        "rdkit_sanitized": smiles is not None,
        "canonical_smiles": smiles,
        "sanitization_error": error,
        "valence_violation_count": count,
        "valence_excess": excess,
    }
