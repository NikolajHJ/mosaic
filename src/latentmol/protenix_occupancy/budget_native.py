"""Native contact scores on matched ligand-core and protein-pocket pairs."""

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from latentmol.multimodel.external import file_sha256
from latentmol.protenix_occupancy.deletion_prepare import _features, native_file_atom_names


class ReferenceGeometryError(ValueError):
    """The valid chemical graph did not yield a reference conformer."""


def score_topology(native, mol, retained_atoms, protein_sequence, pocket_residues, *,
                   geometry_seed, mol_path, geometry_max_iterations=None):
    """Score one hard topology, preserving the declared heavy-atom index order.

    The caller binds the pinned native source and CCD assets before this call.
    The loss averages negative log contact probability over a fixed Cartesian
    product of retained ligand atoms and one-based protein pocket residues.
    """
    from protenix.data.inference.json_to_feature import SampleDictToFeatures

    retained_atoms, pocket_residues = tuple(retained_atoms), tuple(pocket_residues)
    size = mol.GetNumAtoms()
    if not retained_atoms or len(set(retained_atoms)) != len(retained_atoms) or any(
            not 0 <= index < size for index in retained_atoms):
        raise ValueError("Require unique retained ligand atom indices")
    if not pocket_residues or len(set(pocket_residues)) != len(pocket_residues) or any(
            not 1 <= residue <= len(protein_sequence) for residue in pocket_residues):
        raise ValueError("Require unique one-based protein pocket residues")
    if any(atom.GetAtomicNum() <= 1 for atom in mol.GetAtoms()) or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Require one connected heavy-atom molecule")

    # The VAE graph schema has no stereochemistry, and 3D MOL loading can
    # infer stereochemistry from a particular conformer.
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    elements = [atom.GetSymbol().upper() for atom in mol.GetAtoms()]
    embedded = Chem.AddHs(mol)
    embedded.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = geometry_seed
    params.numThreads = 1
    if geometry_max_iterations is not None:
        params.maxIterations = geometry_max_iterations
    try:
        embed_status = AllChem.EmbedMolecule(embedded, params)
    except RuntimeError as exc:
        if "Invariant Violation" not in str(exc) or "upper bound not greater than lower bound" not in str(exc):
            raise
        raise ReferenceGeometryError(f"ETKDG failed for topology {mol_path}: inconsistent distance bounds") from exc
    if embed_status != 0:
        raise ReferenceGeometryError(f"ETKDG failed for topology {mol_path}")
    embedded = Chem.RemoveHs(embedded)
    if embedded.GetNumAtoms() != size or not embedded.GetConformer().Is3D():
        raise ValueError("Ligand embedding lost its declared 3D heavy-atom set")
    mol_path = Path(mol_path).resolve()
    with mol_path.open("x", encoding="utf-8") as handle:
        handle.write(Chem.MolToMolBlock(embedded))
    loaded = Chem.MolFromMolFile(str(mol_path), removeHs=True)
    if loaded is None or Chem.MolToSmiles(loaded, canonical=True, isomericSmiles=False) != canonical:
        raise ValueError("Native input MOL did not preserve ligand chemistry")

    sample = {"name": mol_path.stem, "sequences": [
        {"proteinChain": {"sequence": protein_sequence, "count": 1}},
        {"ligand": {"ligand": f"FILE_{mol_path}", "count": 1}},
    ]}
    atoms = SampleDictToFeatures(sample).get_atom_array()
    features, _ = _features(atoms)
    ligand = features["is_ligand"].nonzero().flatten()
    if (len(ligand) != size or [str(atoms.element[i]).upper() for i in ligand.tolist()] != elements
            or [str(atoms.atom_name[i]) for i in ligand.tolist()] != native_file_atom_names(elements)):
        raise ValueError("Native FILE import changed ligand atom identity or order")
    ligand_tokens = features["atom_to_token_idx"][ligand]
    if ligand_tokens.unique().numel() != size:
        raise ValueError("Native FILE import needs one token per ligand atom")
    expected_bonds = torch.from_numpy(Chem.GetAdjacencyMatrix(mol)).bool()
    native_bonds = features["token_bonds"][ligand_tokens[:, None], ligand_tokens].bool()
    if not torch.equal(native_bonds, expected_bonds):
        raise ValueError("Native FILE import changed ligand connectivity")

    representatives = (features["is_protein"].bool() & features["distogram_rep_atom_mask"].bool())
    protein_atoms = representatives.nonzero().flatten().tolist()
    protein_tokens = {int(atoms.res_id[index]): int(features["atom_to_token_idx"][index])
                      for index in protein_atoms}
    if len(protein_atoms) != len(protein_tokens) or set(protein_tokens) != set(range(1, len(protein_sequence) + 1)):
        raise ValueError("Native FILE import did not preserve one token per protein position")

    logits = native.native_logits(native.prepare_features(features))
    num_tokens = len(features["token_index"])
    if logits.shape != (num_tokens, num_tokens, 64) or not logits.isfinite().all():
        raise ValueError("Require finite native 64-bin distogram logits")
    left = ligand_tokens[list(retained_atoms)].to(logits.device)
    right = torch.tensor([protein_tokens[index] for index in pocket_residues], device=logits.device)
    readout = logits[left[:, None], right].double()
    # Use only complete bins below the 8 Å cutoff. The bin spanning 8 Å
    # cannot be resolved from the distogram and contributes no contact mass.
    boundaries = torch.linspace(2.3125, 21.6875, 63)
    contact_bins = int((boundaries < 8.0).sum())
    log_contact = (readout[..., :contact_bins].logsumexp(-1) - readout.logsumexp(-1))
    loss = float(-log_contact.mean())
    return loss, dict(
        geometry_seed=geometry_seed, mol_sha256=file_sha256(mol_path), heavy_atoms=size,
        core_atoms=len(retained_atoms), pocket_residues=list(pocket_residues),
        contact_pairs=log_contact.numel(), contact_bins=contact_bins,
        included_upper_bound_angstrom=float(boundaries[contact_bins - 1]),
        mean_contact_probability=float(log_contact.exp().mean()),
    )
