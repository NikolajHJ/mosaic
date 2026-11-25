"""Lossless conversion for the explicitly limited, hydrogen-implicit graph schema.

Stereochemistry is deliberately discarded. Isotopes, radicals, fragments, and
chemistry needing unrepresented atom features are excluded rather than repaired.
In particular, explicit hydrogen counts are not graph features: a molecule is
accepted only when decoding its graph recovers its canonical stereo-free SMILES.
RDKit validity is not evidence of stability, synthesizability, or useful binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from .representation import GraphSpec


class MoleculeRejected(ValueError):
    """A rejected source molecule, with a machine-readable rejection reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


def canonicalize_smiles(smiles: str, spec: GraphSpec) -> str:
    """Return canonical stereo-free SMILES after schema eligibility checks.

    This does not remove salts, neutralize charges, repair chemistry, or perform
    tautomer normalization. Representability is additionally checked by
    :func:`encode_smiles`, because element/charge/bond labels omit hydrogen counts.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise MoleculeRejected("empty")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise MoleculeRejected("invalid_smiles")
    if not mol.GetNumAtoms():
        raise MoleculeRejected("empty")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise MoleculeRejected("disconnected")
    if mol.GetNumAtoms() > spec.max_nodes:
        raise MoleculeRejected("too_many_atoms", str(mol.GetNumAtoms()))
    allowed_elements = set(spec.atomic_numbers) - {0}
    allowed_charges = set(spec.formal_charges)
    allowed_bonds = set(spec.bond_names) - {"NONE"}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in allowed_elements:
            raise MoleculeRejected("unsupported_element", atom.GetSymbol())
        if atom.GetFormalCharge() not in allowed_charges:
            raise MoleculeRejected("unsupported_charge", str(atom.GetFormalCharge()))
        if atom.GetIsotope():
            raise MoleculeRejected("isotope")
        if atom.GetNumRadicalElectrons():
            raise MoleculeRejected("radical")
    for bond in mol.GetBonds():
        if str(bond.GetBondType()) not in allowed_bonds:
            raise MoleculeRejected("unsupported_bond", str(bond.GetBondType()))
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def encode_smiles(smiles: str, spec: GraphSpec) -> dict[str, Any]:
    """Encode one molecule using canonical atom ordering and verify roundtrip.

    Returns ``smiles`` and int8 ``atom_types``, ``charges``, ``bond_types`` arrays.
    Atom and charge values are vocabulary indices, not atomic numbers or charges.
    Padding uses NO_ATOM, neutral charge, and NONE bonds. Molecules that cannot be
    represented losslessly after deliberate stereo stripping raise
    :class:`MoleculeRejected`; they are never silently changed to a related graph.
    """
    return encode_canonical_smiles(canonicalize_smiles(smiles, spec), spec)


def encode_canonical_smiles(canonical: str, spec: GraphSpec) -> dict[str, Any]:
    """Encode the stored slot order and require an exact graph-to-SMILES round-trip.

    Ingest already validated this spelling. Applying the raw-input
    canonicalizer again can change the aromatic spelling of rare graphs.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        raise MoleculeRejected("canonical_parse_failure")
    if not 1 <= mol.GetNumAtoms() <= spec.max_nodes:
        raise MoleculeRejected("canonical_size")
    element_index = {value: index for index, value in enumerate(spec.atomic_numbers)}
    charge_index = {value: index for index, value in enumerate(spec.formal_charges)}
    bond_index = {value: index for index, value in enumerate(spec.bond_names)}
    if any(len(values) > 128 for values in
           (spec.atomic_numbers, spec.formal_charges, spec.bond_names)):
        raise ValueError("The cache format supports at most 128 categories per feature.")
    atoms = np.full(spec.max_nodes, element_index[0], dtype=np.int8)
    charges = np.full(spec.max_nodes, charge_index[0], dtype=np.int8)
    bonds = np.full((spec.max_nodes, spec.max_nodes), bond_index["NONE"], dtype=np.int8)
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        element = element_index.get(atom.GetAtomicNum(), 0)
        charge = charge_index.get(atom.GetFormalCharge())
        if element == 0 or charge is None:
            raise MoleculeRejected("canonical_atom_labels")
        atoms[index] = element
        charges[index] = charge
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        label = bond_index.get(str(bond.GetBondType()), bond_index["NONE"])
        if label == bond_index["NONE"]:
            raise MoleculeRejected("unsupported_bond")
        bonds[left, right] = bonds[right, left] = label
    if decode_graph(atoms, charges, bonds, spec) != canonical:
        raise MoleculeRejected("representation_roundtrip")
    return {"smiles": canonical, "atom_types": atoms, "charges": charges,
            "bond_types": bonds}


def _integer_array(value: Any) -> np.ndarray:
    """Read an already-discrete array/tensor without rounding probabilities."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "iu":
        raise ValueError("Graph labels must be integer category indices.")
    return result


@dataclass(frozen=True)
class DecodeResult:
    """A raw decode and its first failure, with no chemistry repair.

    Failure categories describe the stage that rejected the graph. For failed
    RDKit sanitization, detail is the exception class rather than an inferred
    chemical diagnosis; several defects can coexist in one prediction.
    """

    smiles: str | None = None
    failure_reason: str | None = None
    detail: str | None = None


def _label_molecule(atom_values, charge_values, active_bonds, spec, active):
    """Build the same unvalidated RDKit graph for inspection and slot mapping."""
    rw = Chem.RWMol()
    for index in active:
        atom = Chem.Atom(int(spec.atomic_numbers[int(atom_values[index])]))
        atom.SetFormalCharge(int(spec.formal_charges[int(charge_values[index])]))
        rw.AddAtom(atom)
    rdkit_bonds = {"SINGLE": Chem.BondType.SINGLE, "DOUBLE": Chem.BondType.DOUBLE,
                   "TRIPLE": Chem.BondType.TRIPLE, "AROMATIC": Chem.BondType.AROMATIC}
    for left in range(len(active)):
        for right in range(left + 1, len(active)):
            name = spec.bond_names[int(active_bonds[left, right])]
            if name == "NONE":
                continue
            if name not in rdkit_bonds:
                raise MoleculeRejected("unsupported_bond")
            rw.AddBond(left, right, rdkit_bonds[name])
            if name == "AROMATIC":
                rw.GetAtomWithIdx(left).SetIsAromatic(True)
                rw.GetAtomWithIdx(right).SetIsAromatic(True)
                rw.GetBondBetweenAtoms(left, right).SetIsAromatic(True)
    return rw.GetMol()


def inspect_graph(atoms: Any, charges: Any, bonds: Any, spec: GraphSpec) -> DecodeResult:
    """Inspect a discrete graph, preserving the first observed rejection reason.

    Inputs are one unbatched integer NumPy graph or detached torch graph. NO_ATOM
    slots and their incident bonds are absent by definition, including unused
    charge labels. Active bonds must be symmetric, valid, and have no self loops.
    Aromatic atom flags follow aromatic bonds. No charge/bond/valence repair or
    largest-fragment selection is performed. Stereo and explicit H counts cannot
    be recovered because the representation does not contain those features.
    """
    try:
        atom_values = _integer_array(atoms)
        charge_values = _integer_array(charges)
        bond_values = _integer_array(bonds)
        size = len(atom_values)
        if (atom_values.shape != (size,) or charge_values.shape != (size,)
                or bond_values.shape != (size, size) or size > spec.max_nodes):
            return DecodeResult(failure_reason="shape")
        if np.any(atom_values < 0) or np.any(atom_values >= len(spec.atomic_numbers)):
            return DecodeResult(failure_reason="atom_category")
        active = [index for index, value in enumerate(atom_values)
                  if spec.atomic_numbers[int(value)] != 0]
        if not active:
            return DecodeResult(failure_reason="empty")
        active_charges = charge_values[active]
        if np.any(active_charges < 0) or np.any(active_charges >= len(spec.formal_charges)):
            return DecodeResult(failure_reason="charge_category")
        active_bonds = bond_values[np.ix_(active, active)]
        none_index = spec.bond_names.index("NONE")
        if (np.any(active_bonds < 0) or np.any(active_bonds >= len(spec.bond_names))
                or not np.array_equal(active_bonds, active_bonds.T)
                or np.any(np.diag(active_bonds) != none_index)):
            return DecodeResult(failure_reason="bond_labels_or_symmetry")
        if any(spec.bond_names[int(value)] not in ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
               for value in np.unique(active_bonds)):
            return DecodeResult(failure_reason="unsupported_bond")
        mol = _label_molecule(atom_values, charge_values, active_bonds, spec, active)
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
        except (ValueError, RuntimeError) as error:
            return DecodeResult(failure_reason="sanitization", detail=type(error).__name__)
        if len(Chem.GetMolFrags(mol)) != 1:
            return DecodeResult(failure_reason="disconnected")
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            return DecodeResult(failure_reason="radical")
        return DecodeResult(smiles=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    except (ValueError, TypeError, IndexError, KeyError, RuntimeError, OverflowError) as error:
        return DecodeResult(failure_reason="input_or_construction", detail=type(error).__name__)


def decode_graph(atoms: Any, charges: Any, bonds: Any, spec: GraphSpec) -> str | None:
    """Return canonical SMILES or None; use inspect_graph for failure diagnostics."""
    return inspect_graph(atoms, charges, bonds, spec).smiles


def canonical_slot_order(atoms: Any, charges: Any, bonds: Any, spec: GraphSpec) -> list[int]:
    """A lossless hard-graph bijection to the dataset's canonical SMILES order.

    This is discrete input preprocessing, not a relaxed-graph operation or
    chemistry repair. Verify every atom, charge and bond label after mapping;
    reject graphs needing aromatization/normalization or unrepresented features.
    Symmetric atom matches may differ, but their resulting full label tensors
    must equal the same canonical encoding. Padding moves to the final slots.
    """
    checked = inspect_graph(atoms, charges, bonds, spec)
    if checked.smiles is None:
        raise MoleculeRejected("canonical_input_"+str(checked.failure_reason))
    atom_values, charge_values, bond_values = map(_integer_array, (atoms, charges, bonds))
    if len(atom_values) != spec.max_nodes:
        raise ValueError("Canonical encoder requires its full declared slot capacity")
    encoded = encode_canonical_smiles(checked.smiles, spec)
    if (np.array_equal(atom_values, encoded["atom_types"]) and
            np.array_equal(charge_values, encoded["charges"]) and
            np.array_equal(bond_values, encoded["bond_types"])):
        return list(range(spec.max_nodes))
    active = np.flatnonzero(atom_values != 0).tolist()
    molecule = _label_molecule(atom_values, charge_values, bond_values[np.ix_(active, active)], spec, active)
    with rdBase.BlockLogs():
        Chem.SanitizeMol(molecule)
    canonical = Chem.MolFromSmiles(encoded["smiles"])
    match = molecule.GetSubstructMatch(canonical, useChirality=False)
    if len(match) != len(active) or len(set(match)) != len(active):
        raise MoleculeRejected("canonical_input_no_bijection")
    order = [active[i] for i in match]+np.flatnonzero(atom_values == 0).tolist()
    if not (np.array_equal(atom_values[order], encoded["atom_types"]) and
            np.array_equal(charge_values[order], encoded["charges"]) and
            np.array_equal(bond_values[np.ix_(order, order)], encoded["bond_types"])):
        raise MoleculeRejected("canonical_input_not_label_preserving")
    return order
