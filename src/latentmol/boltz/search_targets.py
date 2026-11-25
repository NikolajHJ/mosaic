"""Detached native Boltz2 preparation from an explicit crystal pocket.

Uses the deposited ligand pose and native CCD names. No predicted/random pose,
diffusion, or ligand-to-pose rematching occurs during optimization.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from latentmol.chemistry import inspect_graph
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec, HardGraphBatch
from .contracts import AFFINITY_SHA256, PreparedComplex
from .dbmol_objective import DBMolObjectiveConfig, pocket_token_mask

CONFIDENCE_SHA256 = "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def seed_in_ccd_order(molecule, spec, *, allow_reference_roundtrip_failure=False):
    """Encode labels while preserving native CCD atom names and ordering."""
    from rdkit import Chem
    molecule = Chem.RemoveHs(Chem.Mol(molecule))
    Chem.RemoveStereochemistry(molecule)
    count = molecule.GetNumAtoms()
    if not 1 <= count <= spec.max_nodes or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("Require one connected CCD ligand within the graph capacity")
    atoms = torch.zeros((1, spec.max_nodes), dtype=torch.long)
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros((1, spec.max_nodes, spec.max_nodes), dtype=torch.long)
    names = []
    for index, atom in enumerate(molecule.GetAtoms()):
        if not atom.HasProp("name"):
            raise ValueError("Native CCD ligand lacks atom-name correspondence")
        names.append(atom.GetProp("name"))
        atoms[0, index] = spec.atomic_numbers.index(atom.GetAtomicNum())
        charges[0, index] = spec.formal_charges.index(atom.GetFormalCharge())
    if len(set(names)) != count:
        raise ValueError("Native CCD ligand atom names are duplicated")
    for bond in molecule.GetBonds():
        category = spec.bond_names.index(str(bond.GetBondType()))
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[0, i, j] = bonds[0, j, i] = category
    seed = HardGraphBatch(atoms, charges, bonds)
    checked = inspect_graph(atoms[0], charges[0], bonds[0], spec)
    expected = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    if checked.smiles != expected and (checked.smiles is not None or not allow_reference_roundtrip_failure):
        raise ValueError(f"CCD reference is not representable in the search schema: {checked.failure_reason}")
    return molecule, seed, names, expected


def selected_label_chains(mmcif, target):
    """Resolve author chain/residue locators into mmCIF label subchains."""
    import gemmi
    pocket = target["pocket"]
    if file_sha256(mmcif) != pocket["mmcif_sha256"]:
        raise ValueError("Target mmCIF bytes differ from the recorded input")
    block = gemmi.cif.read_file(str(mmcif)).sole_block()
    scheme = block.find("_pdbx_poly_seq_scheme.", ["asym_id", "pdb_strand_id"])
    proteins = {str(row[0]) for row in scheme if row[1] == pocket["protein_chain"]}
    locator = pocket["reference_ligand_locator"]
    table = block.find("_atom_site.", ["label_asym_id", "auth_asym_id", "auth_seq_id", "auth_comp_id"])
    ligands = {str(row[0]) for row in table if row[1] == locator["chain"]
               and row[2] == str(locator["residue"]) and row[3] == locator["component"]}
    if len(proteins) != 1 or len(ligands) != 1 or proteins == ligands:
        raise ValueError("Exact protein/reference-ligand label subchains are missing or ambiguous")
    return next(iter(proteins)), next(iter(ligands))


def verify_native_pocket(tokens, target, protein_asym):
    """Require the user's biological residues in the native cropped view."""
    records = []
    for row in target["pocket"]["residues"]:
        index = row["sequence_position"] - 1
        selected = tokens[(tokens["asym_id"] == protein_asym) & (tokens["res_idx"] == index)
                          & (tokens["mol_type"] == 0)]
        if len(selected) != 1 or str(selected[0]["res_name"]) != row["residue_name"]:
            raise ValueError(f"User pocket residue missing/ambiguous after native crop: {row['residue_name']}{row['author_residue']}")
        if not bool(selected[0]["resolved_mask"]):
            raise ValueError("A selected pocket residue lacks a resolved crystal frame")
        records.append(dict(asym_id=int(protein_asym), residue_index=int(index)))
    return records


def verify_pose_distances(actual_ligand, source_ligand, actual_pocket, source_pocket):
    """Audit relative chain placement with direct FP64 differences.

    ``cdist``'s FP32 matrix-multiply path loses diagonal precision for larger
    coordinate sets. Direct differences also test relative protein placement,
    which a ligand-only intramolecular comparison would miss.
    """
    def distances(left, right):
        return torch.linalg.vector_norm(left.double()[:, None] - right.double()[None, :], dim=-1)
    intra_actual = distances(actual_ligand, actual_ligand)
    intra_source = distances(source_ligand, source_ligand)
    cross_actual = distances(actual_ligand, actual_pocket)
    cross_source = distances(source_ligand, source_pocket)
    torch.testing.assert_close(intra_actual, intra_source, atol=5e-4, rtol=1e-5)
    torch.testing.assert_close(cross_actual, cross_source, atol=5e-4, rtol=1e-5)
    return dict(ligand_pair_max_distance_error=float((intra_actual-intra_source).abs().max()),
                ligand_pocket_max_distance_error=float((cross_actual-cross_source).abs().max()),
                arithmetic="Direct coordinate differences in float64; no coordinate changes")


def prepare_target(target, mmcif, cache, output, *, spec=None,
                   affinity_checkpoint=None, confidence_checkpoint=None,
                   max_protein_tokens=200, max_tokens=256, max_atoms=2048):
    """Create a native context and explicit DBMol profile in a new directory.

    The default crop follows native affinity inference. All supplied pocket
    residues must survive; failure never silently changes the objective.
    """
    if not (os.environ.get("LSB_JOBID", "").isdigit() or os.environ.get("PBS_JOBID")):
        raise RuntimeError("Boltz target preparation requires an allocated job")
    from rdkit.Chem import Descriptors
    from boltz.data import const
    from boltz.data.crop.affinity import AffinityCropper
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.module.inferencev2 import collate
    from boltz.data.mol import load_canonicals, load_molecules
    from boltz.data.parse.mmcif import parse_mmcif
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import AffinityInfo, ChainInfo, Input, Record
    from .prepare import installed_boltz_provenance

    spec = spec or GraphSpec()
    mmcif, cache, output = Path(mmcif), Path(cache), Path(output)
    affinity_checkpoint = Path(affinity_checkpoint or cache / "boltz2_aff.ckpt")
    confidence_checkpoint = Path(confidence_checkpoint or cache / "boltz2_conf.ckpt")
    if file_sha256(affinity_checkpoint) != AFFINITY_SHA256 or file_sha256(confidence_checkpoint) != CONFIDENCE_SHA256:
        raise ValueError("Boltz2 affinity/confidence checkpoint identity differs")
    provenance = installed_boltz_provenance()
    protein_label, ligand_label = selected_label_chains(mmcif, target)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    molecules = load_canonicals(cache / "mols")
    parsed = parse_mmcif(str(mmcif), mols=molecules, moldir=cache / "mols",
                         use_assembly=False, compute_interfaces=False)
    names = parsed.data.chains["name"].tolist()
    selected_names = {protein_label, ligand_label}
    if any(names.count(name) != 1 for name in selected_names):
        raise ValueError("Native parser did not preserve the selected label subchains")
    structure = replace(parsed.data, mask=np.array([name in selected_names for name in names], dtype=bool)).remove_invalid_chains()
    protein_id = int(np.flatnonzero(structure.chains["name"] == protein_label)[0])
    ligand_id = int(np.flatnonzero(structure.chains["name"] == ligand_label)[0])
    ligand_chain = structure.chains[ligand_id]
    if (int(structure.chains[protein_id]["mol_type"]) != const.chain_type_ids["PROTEIN"]
            or int(ligand_chain["mol_type"]) != const.chain_type_ids["NONPOLYMER"]
            or int(ligand_chain["res_num"]) != 1):
        raise ValueError("Require exactly one protein polymer and one noncovalent reference ligand residue")
    for bond in structure.bonds:
        if {int(bond["chain_1"]), int(bond["chain_2"])} == {protein_id, ligand_id}:
            raise ValueError("Reference ligand has a covalent protein link; specify a covalent design protocol")
    component = target["pocket"]["reference_ligand_locator"]["component"]
    molecules.update(load_molecules(cache / "mols", {component} - set(molecules)))
    molecule, seed, atom_names, canonical = seed_in_ccd_order(molecules[component], spec,
        allow_reference_roundtrip_failure=True)
    roundtrip = inspect_graph(seed.atoms[0], seed.charges[0], seed.bonds[0], spec)
    reference_roundtrip = dict(matched=roundtrip.smiles == canonical,
        ccd_smiles=canonical, decoded_smiles=roundtrip.smiles,
        failure_reason=roundtrip.failure_reason, detail=roundtrip.detail,
        scope="Reference pose/feature fixture only; generated output validity is never relaxed")
    count = molecule.GetNumAtoms()
    if count != target["initialized_atoms"] or count != target["pocket"]["reference_ligand_heavy_atoms"]:
        raise ValueError("Requested occupancy and CCD/reference ligand counts differ")
    affinity = AffinityInfo(chain_id=ligand_id, mw=float(Descriptors.MolWt(molecule)))
    chains = [ChainInfo(chain_id=i, chain_name=str(row["name"]), mol_type=int(row["mol_type"]),
                        cluster_id=-1, msa_id=-1, num_residues=int(row["res_num"]))
              for i, row in enumerate(structure.chains)]
    record = Record(id=target["id"], structure=parsed.info, chains=chains, interfaces=[], affinity=affinity)
    data = Input(structure=structure, msa={}, record=record)
    full_tokens = Boltz2Tokenizer().tokenize(data)
    verify_native_pocket(full_tokens.tokens, target, protein_id)
    tokenized = AffinityCropper(max_tokens_protein=max_protein_tokens).crop(full_tokens,
        max_tokens=max_tokens, max_atoms=max_atoms)
    pocket_residues = verify_native_pocket(tokenized.tokens, target, protein_id)
    molecules.update(load_molecules(cache / "mols", set(tokenized.tokens["res_name"]) - set(molecules)))
    torch.manual_seed(42)
    features = Boltz2Featurizer().process(tokenized, molecules=molecules,
        random=np.random.default_rng(42), training=False, max_atoms=None, max_tokens=None,
        max_seqs=1, pad_to_max_seqs=False, single_sequence_prop=1.0,
        compute_frames=True, compute_constraint_features=True, override_method="other", compute_affinity=True)
    batch = {key: value for key, value in collate([features]).items() if isinstance(value, torch.Tensor)}
    ligand_tokens = tokenized.tokens[tokenized.tokens["affinity_mask"]]
    if len(ligand_tokens) != count or not np.all(ligand_tokens["atom_num"] == 1) or not np.all(ligand_tokens["resolved_mask"]):
        raise ValueError("Native crop lacks the complete resolved reference ligand atom set")
    slots = {name: i for i, name in enumerate(atom_names)}
    slot_atom = torch.full((spec.max_nodes,), -1, dtype=torch.long)
    slot_token = torch.full_like(slot_atom, -1)
    mapping, atom_row = [], 0
    source_coordinates = torch.zeros((count, 3), dtype=torch.float32)
    pocket_native_atoms, source_pocket_coordinates = [], []
    pocket_indices = {row["residue_index"] for row in pocket_residues}
    for token_row, token in enumerate(tokenized.tokens):
        if token["affinity_mask"]:
            native = tokenized.structure.atoms[int(token["atom_idx"])]
            name = str(native["name"])
            if name not in slots or slot_atom[slots[name]] != -1:
                raise ValueError("Native ligand atom names do not match the unique CCD atom map")
            slot = slots[name]
            slot_atom[slot], slot_token[slot] = atom_row, token_row
            source_coordinates[slot] = torch.from_numpy(np.array(native["coords"], dtype=np.float32))
            mapping.append(dict(slot=slot, atom_name=name, native_atom=atom_row,
                                native_token=token_row, source_atom=int(token["atom_idx"])))
        elif int(token["asym_id"]) == protein_id and int(token["res_idx"]) in pocket_indices:
            center_offset = int(token["center_idx"]) - int(token["atom_idx"])
            pocket_native_atoms.append(atom_row + center_offset)
            point = tokenized.structure.atoms[int(token["center_idx"])]["coords"]
            source_pocket_coordinates.append(np.array(point, dtype=np.float32))
        atom_row += int(token["atom_num"])
    actual_coordinates = batch["coords"][0, 0, slot_atom[:count]]
    pose_audit = verify_pose_distances(actual_coordinates, source_coordinates,
        batch["coords"][0, 0, pocket_native_atoms], torch.from_numpy(np.array(source_pocket_coordinates)))
    provenance.update(target_id=target["id"], target=target,
        seed_smiles=canonical, slot_map=sorted(mapping, key=lambda row: row["slot"]),
        mmcif_sha256=file_sha256(mmcif), reference_pose_sha256=file_sha256(mmcif),
        reference_pose_source="Deposited chain A/reference ligand crystal coordinates; no diffusion prediction",
        pose_distance_audit=pose_audit,
        pose_policy="fixed crystallographic pose, translated by native featurizer; detached",
        occupancy_policy="fixed complete CCD ligand map; packed slots in CCD heavy-atom order",
        graph_stereochemistry="not represented; native reference stereochemical metadata remains fixed",
        reference_roundtrip=reference_roundtrip,
        warnings=(["Reference CCD graph cannot round-trip through the topology codec (e.g. explicit aromatic N-H); "
                   "native CCD features are preserved without Kekulization. This reference is not a generated molecule."]
                  if not reference_roundtrip["matched"] else []),
        protein_modifications="native deposited CCD residue identities retained; no canonical substitution",
        msa_policy="native query-only dummy MSA; no external MSA or templates", fixed_msa_row_indices=[0], preparation_seed=42,
        target_sequence_sha256=hashlib.sha256(target["pocket"]["protein_sequence"].encode()).hexdigest(),
        crop=dict(max_protein_tokens=max_protein_tokens, max_tokens=max_tokens, max_atoms=max_atoms,
                  full_tokens=len(full_tokens.tokens), retained_tokens=len(tokenized.tokens),
                  retained_protein_tokens=int((tokenized.tokens["mol_type"] == 0).sum()),
                  all_requested_pocket_residues_retained=True),
        fixed_features="crystal pose, CCD ref_pos, atom names, chirality, crop, query MSA, masks and auxiliary native fields")
    prepared = PreparedComplex(batch, batch["coords"].detach().clone(), slot_atom, slot_token, seed, spec, provenance)
    prepared.validate()
    provenance["native_seed_feature_parity"] = True
    prepared_path = output / "prepared.pt"
    prepared.save(prepared_path)
    dbmol = DBMolObjectiveConfig.from_mapping(dict(
        confidence_checkpoint=dict(path=str(confidence_checkpoint.resolve()), sha256=CONFIDENCE_SHA256),
        pocket_origin="explicit_native_residues", pocket_residues=pocket_residues), output)
    pocket_token_mask(prepared.features, dbmol)
    _write(output / "dbmol.json", dbmol.record())
    config = dict(prepared=str(prepared_path.resolve()), prepared_sha256=file_sha256(prepared_path),
        checkpoint=str(affinity_checkpoint.resolve()), checkpoint_sha256=AFFINITY_SHA256,
        recycling_steps=0, allow_lossy_charge=True, dbmol=dbmol.record())
    _write(output / "scorer-config.json", config)
    receipt = dict(status="prepared_only", prepared=dict(path=config["prepared"], sha256=config["prepared_sha256"]),
                   scorer_config=str(output / "scorer-config.json"), dbmol=dbmol.record(),
                   graph_spec=spec.to_dict(), provenance=provenance,
                   scope="Detached CPU native feature preparation only; no native model/gradient test performed")
    _write(output / "preparation.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mmcif", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--max-protein-tokens", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-atoms", type=int, default=2048)
    args = parser.parse_args()
    target = json.loads(args.config.read_text())["arms"][args.target]["target"]
    spec = None
    if args.vae_checkpoint:
        checkpoint = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=True)
        spec = GraphSpec.from_dict(checkpoint["graph_spec"])
    receipt = prepare_target(target, args.mmcif, args.cache, args.output, spec=spec,
        max_protein_tokens=args.max_protein_tokens, max_tokens=args.max_tokens, max_atoms=args.max_atoms)
    print(json.dumps(dict(status=receipt["status"], prepared=receipt["prepared"])))


if __name__ == "__main__":
    main()
