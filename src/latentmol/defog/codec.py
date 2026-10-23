"""Explicit fixed-occupancy views of our graph for public GuacaMol DeFoG.

DeFoG GuacaMol has element and bond labels but no formal-charge or NO_ATOM
channel. The default preserves the historical neutral-only contract. A separate
opt-in policy projects away input charge and lifts sampled native labels with
the pinned native N/O/S charge rule. It never changes the canonical input graph;
the lifted output is an explicitly different detached reference, not a native
charge prediction or an identity-preserving representation of the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

import torch
from torch.nn import functional as F

from latentmol.representation import GraphSpec, HardGraphBatch


GUACAMOL_ATOMIC_NUMBERS = (6, 7, 8, 9, 5, 35, 17, 53, 15, 16, 34, 14)
GUACAMOL_BONDS = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
STRICT_CHARGE_POLICY = "strict_neutral_v1"
PROJECT_CHARGE_POLICY = "project_input_native_output_v1"
CHARGE_POLICIES = (STRICT_CHARGE_POLICY, PROJECT_CHARGE_POLICY)
NATIVE_CHARGE_RULE = "pinned_per_bond_valence_plus1_NOS_v1"


@dataclass(frozen=True)
class DefogHardGraph:
    """One DeFoG graph; diagonal and masked edges contain zeros, not NONE one-hot."""

    x: torch.Tensor  # [1, n, 12]
    e: torch.Tensor  # [1, n, n, 5]
    node_mask: torch.Tensor  # [1, n]
    occupied_slots: tuple[int, ...]


def to_guacamol_hard(graph: HardGraphBatch, spec: GraphSpec, *,
                     charge_policy: str = STRICT_CHARGE_POLICY) -> DefogHardGraph:
    """Map one hard graph under an explicit input feature-fidelity policy.

    Default strict mode rejects charge loss. Optional projected mode preserves
    every element/bond label and slot while omitting charge from this view;
    callers must retain projection_receipt and the original charged candidate.
    Neither mode checks validity or mutates the canonical graph. Full output
    RDKit inspection remains separate.
    """
    if charge_policy not in CHARGE_POLICIES:
        raise ValueError("Unknown GuacaMol charge policy")
    if graph.atoms.ndim != 2 or graph.atoms.shape != (1, spec.max_nodes):
        raise ValueError("Expected one full-capacity graph")
    if graph.charges.shape != graph.atoms.shape or graph.bonds.shape != (1, spec.max_nodes, spec.max_nodes):
        raise ValueError("Graph dimensions do not match schema")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if any(value.dtype not in integer_types for value in
           (graph.atoms, graph.charges, graph.bonds)):
        raise ValueError("Hard graph labels must be integers")
    atom_labels = graph.atoms[0]
    charge_labels = graph.charges[0]
    bond_labels = graph.bonds[0]
    if (atom_labels < 0).any() or (atom_labels >= len(spec.atomic_numbers)).any():
        raise ValueError("Atom labels outside schema")
    occupied = (atom_labels != 0).nonzero(as_tuple=False).flatten()
    if occupied.numel() < 1:
        raise ValueError("Empty graph")
    pair = (atom_labels != 0)[:, None] & (atom_labels != 0)[None, :]
    if (bond_labels[~pair] != spec.bond_names.index("NONE")).any():
        raise ValueError("Bonds to NO_ATOM padding are unsupported")
    if (charge_labels[atom_labels == 0] != spec.formal_charges.index(0)).any():
        raise ValueError("Padding slots must carry neutral charge")
    if charge_policy == PROJECT_CHARGE_POLICY and (
            (charge_labels < 0).any() or (charge_labels >= len(spec.formal_charges)).any()):
        raise ValueError("Formal charge category outside canonical schema")
    if charge_policy == STRICT_CHARGE_POLICY and (charge_labels[occupied] != spec.formal_charges.index(0)).any():
        raise ValueError("Formal charges are unsupported by DeFoG GuacaMol")
    atomic_numbers = [spec.atomic_numbers[int(atom_labels[i])] for i in occupied]
    if any(number not in GUACAMOL_ATOMIC_NUMBERS for number in atomic_numbers):
        raise ValueError("Active element outside DeFoG GuacaMol vocabulary")
    active_bonds = bond_labels[occupied[:, None], occupied[None, :]]
    if (active_bonds < 0).any() or (active_bonds >= len(spec.bond_names)).any():
        raise ValueError("Bond labels outside schema")
    if not torch.equal(active_bonds, active_bonds.T) or torch.diagonal(active_bonds).any():
        raise ValueError("Bonds must be symmetric with NONE diagonal")
    bond_names = [spec.bond_names[int(value)] for value in active_bonds.flatten()]
    if any(name not in GUACAMOL_BONDS for name in bond_names):
        raise ValueError("Bond type outside DeFoG GuacaMol vocabulary")
    x_labels = torch.tensor([GUACAMOL_ATOMIC_NUMBERS.index(number) for number in atomic_numbers],
                            device=atom_labels.device)
    e_labels = torch.tensor([GUACAMOL_BONDS.index(name) for name in bond_names],
                            device=bond_labels.device).reshape(occupied.numel(), occupied.numel())
    x = F.one_hot(x_labels, len(GUACAMOL_ATOMIC_NUMBERS)).float().unsqueeze(0)
    e = F.one_hot(e_labels, len(GUACAMOL_BONDS)).float().unsqueeze(0)
    diagonal = torch.arange(occupied.numel(), device=e.device)
    e[:, diagonal, diagonal, :] = 0
    return DefogHardGraph(x=x, e=e, node_mask=torch.ones_like(x[..., 0], dtype=torch.bool),
                          occupied_slots=tuple(int(i) for i in occupied.tolist()))


def from_guacamol_labels(x: torch.Tensor, e: torch.Tensor, spec: GraphSpec,
                        occupied_slots: tuple[int, ...]) -> HardGraphBatch:
    """Restore native integer labels to their original canonical slot IDs.

    Native DeFoG has no charge head. Zero formal charge is the declared supported
    subset, not a neutralization operation: the input codec already rejects any
    charged candidate. A native B, Se or Si emission is rejected if our shared
    schema does not represent it. No SMILES conversion/reordering is involved.
    """
    n = len(occupied_slots)
    if (not n or tuple(sorted(set(occupied_slots))) != occupied_slots
            or any(type(slot) is not int or not 0 <= slot < spec.max_nodes
                   for slot in occupied_slots)):
        raise ValueError("Occupied slot IDs must be sorted, unique and within capacity")
    if x.shape != (1, n) or e.shape != (1, n, n):
        raise ValueError("Native output changed atom count or batch shape")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if x.dtype not in integer_types or e.dtype not in integer_types:
        raise ValueError("Native output labels must be integer categories")
    if ((x < 0).any() or (x >= len(GUACAMOL_ATOMIC_NUMBERS)).any()
            or (e < 0).any() or (e >= len(GUACAMOL_BONDS)).any()):
        raise ValueError("Native output category outside checkpoint vocabulary")
    if not torch.equal(e, e.transpose(1, 2)) or torch.diagonal(e[0]).any():
        raise ValueError("Native output bonds must be symmetric with NONE diagonal")
    numbers = [GUACAMOL_ATOMIC_NUMBERS[int(value)] for value in x[0]]
    names = [GUACAMOL_BONDS[int(value)] for value in e.flatten()]
    if any(number not in spec.atomic_numbers for number in numbers):
        raise ValueError("Native output element is unsupported by the shared graph schema")
    if any(name not in spec.bond_names for name in names):
        raise ValueError("Native output bond is unsupported by the shared graph schema")
    atoms = x.new_zeros((1, spec.max_nodes), dtype=torch.long)
    charges = x.new_full((1, spec.max_nodes), spec.formal_charges.index(0), dtype=torch.long)
    bonds = e.new_full((1, spec.max_nodes, spec.max_nodes),
                       spec.bond_names.index("NONE"), dtype=torch.long)
    slots = torch.tensor(occupied_slots, dtype=torch.long, device=x.device)
    atoms[0, slots] = x.new_tensor([spec.atomic_numbers.index(number) for number in numbers])
    bonds[0, slots[:, None], slots[None, :]] = e.new_tensor(
        [spec.bond_names.index(name) for name in names]).reshape(n, n)
    return HardGraphBatch(atoms, charges, bonds)


def projection_receipt(graph: HardGraphBatch, spec: GraphSpec, mapped: DefogHardGraph) -> dict:
    """Hash exact native tensors and disclose the original, unmodified charges.

    Charge-only alternatives intentionally collide in the native digest while
    their canonical candidate/request hashes remain different. No neutral
    replacement HardGraph is made, and no formal charge is injected into the
    checkpoint's own X/E-derived extra features.
    """
    digest = hashlib.sha256()
    for name, tensor in (("X", mapped.x), ("E", mapped.e), ("node_mask", mapped.node_mask)):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    charges = [spec.formal_charges[int(graph.charges[0, slot])] for slot in mapped.occupied_slots]
    return {"charge_policy": PROJECT_CHARGE_POLICY, "feature_fidelity": "lossy_formal_charge",
            "native_projection_sha256": digest.hexdigest(), "occupied_slots": list(mapped.occupied_slots),
            "canonical_input_formal_charges": charges,
            "nonzero_charge_slots": [slot for slot, charge in zip(mapped.occupied_slots, charges) if charge],
            "input_formal_charge_sum": sum(charges), "canonical_candidate_modified": False,
            "native_charge_channel": False, "native_extra_features": "unchanged_checkpoint_X_E_only",
            "scope": "Only element/bond features are observed; original formal charge is not conditioned on."}


def native_charge_molecule(x: torch.Tensor, e: torch.Tensor):
    """Auditable replica of the pinned native *output* charge builder only.

    Inputs have already passed the integer/native-vocabulary/diagonal codec.
    Match native atom order, upper-triangle bond order and per-bond property
    sanitization exactly. The rule only sets +1 on N/O/S after a normal+1
    valence exception. It neither guesses negative/H-dependent charges nor
    calls correct_mol, fragment extraction or any SMILES roundtrip. Native
    parity is a separate required HPC gate before using this optional mode.
    """
    from rdkit import Chem
    bonds = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
    normal_valence = {7: 3, 8: 2, 16: 2}
    mol = Chem.RWMol()
    for label in x[0].detach().cpu():
        mol.AddAtom(Chem.Atom(GUACAMOL_ATOMIC_NUMBERS[int(label)]))
    upper = torch.triu(e[0].detach().cpu())
    for left, right in torch.nonzero(upper).tolist():
        if left == right:
            continue
        mol.AddBond(left, right, bonds[int(upper[left, right])])
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except ValueError as error:
            # The pinned native rule parses exactly these two integers from
            # RDKit's exception. Fail closed if the pinned runtime disagrees.
            message = str(error)
            values = list(map(int, re.findall(r"\d+", message[message.find("#"):])))
            if len(values) != 2:
                raise ValueError("Native charge rule could not parse valence exception") from error
            index, valence = values
            atom = mol.GetAtomWithIdx(index)
            number = atom.GetAtomicNum()
            if number in normal_valence and valence - normal_valence[number] == 1:
                atom.SetFormalCharge(1)
    return mol


def from_guacamol_native_charges(x, e, spec, occupied_slots):
    """Lift native labels into a full canonical target plus explicit rule receipt.

    The native label validation and slot scatter remain the historical codec.
    Only the new target's charges are inferred. Full-graph chemistry inspection
    must subsequently pass; native inference alone is not a validity guarantee.
    """
    from rdkit import Chem, rdBase
    from latentmol.defog.assets import DEFOG_COMMIT
    target = from_guacamol_labels(x, e, spec, occupied_slots)
    mol = native_charge_molecule(x, e)
    if mol.GetNumAtoms() != len(occupied_slots):
        raise ValueError("Native charge conversion changed atom count")
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    if any(charge not in spec.formal_charges for charge in charges):
        raise ValueError("Native inferred charge is outside the canonical vocabulary")
    if any(charge not in (0, 1) or (charge and atom.GetAtomicNum() not in (7, 8, 16))
           for atom, charge in zip(mol.GetAtoms(), charges)):
        raise ValueError("Native charge rule emitted an unsupported charge")
    for slot, charge in zip(occupied_slots, charges):
        target.charges[0, slot] = spec.formal_charges.index(charge)
    # RDKit's full sanitization may itself rewrite formal charges/bond orders
    # (e.g. nitro cleanup). Such a rewrite would bypass the declared native
    # charge rule. Inspect a copy and retain/reject it, never import its labels.
    identity = {"passed": False, "policy": "full_sanitization_preserves_exact_native_atom_bond_and_inferred_charge_labels"}
    sanitized = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(sanitized)
    except (ValueError, RuntimeError) as error:
        identity.update(reason="sanitization_failed", error_type=type(error).__name__)
    else:
        numbers = [atom.GetAtomicNum() for atom in sanitized.GetAtoms()]
        after_charges = [atom.GetFormalCharge() for atom in sanitized.GetAtoms()]
        after_e = torch.zeros_like(e).cpu()
        for bond in sanitized.GetBonds():
            name = str(bond.GetBondType())
            if name not in GUACAMOL_BONDS:
                identity["unsupported_sanitized_bond"] = name
                break
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            after_e[0, i, j] = after_e[0, j, i] = GUACAMOL_BONDS.index(name)
        identity.update(sanitized_formal_charges=after_charges,
            atom_labels_unchanged=numbers == [GUACAMOL_ATOMIC_NUMBERS[int(v)] for v in x[0]],
            charge_labels_unchanged=after_charges == charges,
            bond_labels_unchanged="unsupported_sanitized_bond" not in identity and torch.equal(after_e, e.cpu()))
        identity["passed"] = all(identity[key] for key in
                                 ("atom_labels_unchanged", "charge_labels_unchanged", "bond_labels_unchanged"))
        identity["reason"] = None if identity["passed"] else "sanitization_changed_declared_labels"
    receipt = {"rule": NATIVE_CHARGE_RULE, "native_source_commit": DEFOG_COMMIT,
        "native_function": "src/analysis/rdkit_functions.py:build_molecule_with_partial_charges",
        "rdkit_version": rdBase.rdkitVersion, "occupied_slots": list(occupied_slots),
        "formal_charges": charges,
        "charged_canonical_slots": [slot for slot, charge in zip(occupied_slots, charges) if charge],
        "neural_charge_prediction": False, "native_atom_bond_labels_unchanged": True,
        "fragment_extraction": False, "reindexing": False, "bond_correction": False,
        "sanitization_identity": identity,
        "scope": "Deterministic output-reference charge conversion; negatives and explicit-H charge states are not inferred."}
    return target, receipt
