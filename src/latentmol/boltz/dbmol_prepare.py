"""HPC-only preprocessing of a frozen, initial-pose-derived DBMol pocket.

This is separate from differentiable scoring. It selects residues once using
native heavy-atom coordinates, then writes hashes and native residue identities.
No optimized molecule, outcome score, graph repair or learned model is used.
"""

import torch

from .provenance import context_digest


def initial_pose_pocket(prepared, *, cutoff=8.0):
    """Any native protein heavy atom within cutoff of any reference ligand atom.

    A residue remains selected in every count context, even if later coordinates
    or ligand occupancy differ. This is an engineering pocket, not proof of the
    intended biological binding site. The caller records prepared-file identity.
    """
    prepared.validate()
    features = prepared.features
    protein_tokens = (features["mol_type"][0] == 0) & features["token_pad_mask"][0].bool()
    elements = features["ref_element"][0].argmax(-1)
    atom_mask = features["atom_pad_mask"][0].bool() & (elements > 1)
    protein_atoms = (features["atom_to_token"][0].float() @ protein_tokens.float()) > 0
    protein_atoms &= atom_mask
    ligand_atoms = prepared.slot_to_atom[prepared.active]
    if not protein_atoms.any() or not ligand_atoms.numel():
        raise ValueError("Initial-pose pocket requires native protein and ligand heavy atoms")
    coords = prepared.coordinates.reshape(-1, 3)
    indices = protein_atoms.nonzero().flatten()
    nearby = torch.cdist(coords[indices], coords[ligand_atoms]).min(-1).values < cutoff
    selected_atoms = indices[nearby]
    selected_tokens = features["atom_to_token"][0, selected_atoms].sum(0) > 0
    selected_tokens &= protein_tokens
    if not selected_tokens.any():
        raise ValueError("Initial reference pose has no protein heavy atoms within the declared pocket cutoff")
    residues = sorted({(int(features["asym_id"][0, i]), int(features["residue_index"][0, i]))
                       for i in selected_tokens.nonzero().flatten()})
    return {"pocket_residues": [{"asym_id": a, "residue_index": r} for a, r in residues],
            "pocket_origin": "initial_pose_heavy_atom_8A",
            "reference_context_sha256": context_digest(prepared),
            "selection_cutoff_angstrom": cutoff,
            "protein_heavy_atoms_in_pocket": int(selected_atoms.numel()),
            "claim_limit": "Frozen initial-pose engineering pocket, not a biologically validated site"}
