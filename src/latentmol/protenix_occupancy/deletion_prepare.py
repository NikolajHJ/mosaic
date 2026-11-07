"""Allocated-CPU preparation of compact and ordinary native deletion endpoints.

The compact endpoint removes one real atom/token while preserving every retained
reference coordinate. The two ordinary endpoints independently enter the pinned
FILE_ importer from valid, separately embedded MOL files. Their generated native
atom names and reference conformers are allowed to differ; explicit atom maps
keep the experimental readout and its denominator identical. No model runs here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import traceback

from latentmol.protenix_occupancy.prepare import (
    NATIVE_COMMIT, _bind_native, _require_allocation, _sha, _write_json,
)

PREPARATION_VERSION = "protenix-native-deletion-case-preparation-v1"
AXIS_AUDIT_VERSION = "protenix-tiny-raw-feature-axis-audit-v1"

# Closed schema of get_all_input_features + the exact query/template dummy
# convention used in the authenticated parent. Never infer axes from sizes:
# token and atom counts can coincide in small fixtures.
FEATURE_AXES = {
    "token_index": ("T",), "residue_index": ("T",),
    "asym_id": ("T",), "entity_id": ("T",), "sym_id": ("T",),
    "restype": ("T", 32), "token_bonds": ("T", "T"),
    "ref_pos": ("A", 3), "ref_mask": ("A",),
    "ref_element": ("A", 128), "ref_charge": ("A",),
    "ref_atom_name_chars": ("A", 4, 64), "ref_space_uid": ("A",),
    "has_frame": ("T",), "frame_atom_index": ("T", 3),
    "atom_to_token_idx": ("A",), "atom_to_tokatom_idx": ("A",),
    "is_protein": ("A",), "is_ligand": ("A",),
    "is_dna": ("A",), "is_rna": ("A",),
    "resolution": (1,), "mol_id": ("A",),
    "mol_atom_index": ("A",), "entity_mol_id": ("A",),
    "pae_rep_atom_mask": ("A",), "plddt_m_rep_atom_mask": ("A",),
    "distogram_rep_atom_mask": ("A",), "modified_res_mask": ("A",),
    "bond_mask": ("A", "A"),
    "msa": (1, "T"), "has_deletion": (1, "T"),
    "deletion_value": (1, "T"), "profile": ("T", 32),
    "deletion_mean": ("T",),
    "prot_pair_num_alignments": (), "prot_unpair_num_alignments": (),
    "rna_pair_num_alignments": (), "rna_unpair_num_alignments": (),
    "template_restype": (4, "T"),
    "template_all_atom_mask": (4, "T", 37),
    "template_all_atom_positions": (4, "T", 37, 3),
}


def native_file_atom_names(elements: list[str]) -> list[str]:
    """Pinned FILE importer names each uppercase element by file-order count."""
    counts = {}
    names = []
    for element in elements:
        if not isinstance(element, str) or not element:
            raise ValueError("FILE atom elements must be nonempty strings")
        symbol = element.upper()
        counts[symbol] = counts.get(symbol, 0) + 1
        names.append(f"{symbol}{counts[symbol]}")
    return names


def validate_hydrogen_replacement(full: dict, deleted: dict, optional: str,
                                  neighbor: str) -> None:
    """Require exactly halogen removal and one replacement H at its neighbor."""
    if (optional not in full or neighbor not in full or neighbor == optional
            or set(deleted) != set(full) - {optional}):
        raise ValueError("Hydrogen accounting does not cover the exact retained atom set")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0
           for counts in (full, deleted) for count in counts.values()):
        raise ValueError("Hydrogen counts must be nonnegative integers")
    if full[optional] != 0 or deleted[neighbor] != full[neighbor] + 1:
        raise ValueError("Terminal halogen deletion did not replace its bond by one hydrogen")
    if any(count != full[name] for name, count in deleted.items() if name != neighbor):
        raise ValueError("Deletion altered hydrogen counts outside the optional neighbor")


def audit_feature_schema(features: dict) -> dict:
    """Require the complete unbatched raw schema, legal indices and no caches."""
    import torch

    expected = set(FEATURE_AXES) | {"constraint_feature"}
    if set(features) != expected:
        raise ValueError(f"Unsupported native feature fields: missing={sorted(expected-set(features))}, "
                         f"unexpected={sorted(set(features)-expected)}")
    if features["constraint_feature"] != {}:
        raise ValueError("Only the declared empty native constraint input is supported")
    n_atom = int(features["ref_pos"].shape[0])
    n_token = int(features["token_index"].shape[0])
    if n_atom < 1 or n_token < 1:
        raise ValueError("Native deletion endpoints must retain atoms and tokens")
    sizes = {"A": n_atom, "T": n_token}
    fields = {}
    for key, axes in FEATURE_AXES.items():
        value = features[key]
        shape = tuple(sizes.get(axis, axis) for axis in axes)
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.layout != torch.strided or tuple(value.shape) != shape
                or value.requires_grad):
            raise ValueError(f"Invalid raw native feature shape/device/graph: {key}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite raw native feature: {key}")
        fields[key] = {"axes": list(axes), "shape": list(shape), "dtype": str(value.dtype)}
    if not torch.equal(features["token_index"], torch.arange(n_token)):
        raise ValueError("Native token indices must be contiguous and zero-based")
    mapping = features["atom_to_token_idx"]
    if mapping.dtype != torch.long or bool(((mapping < 0) | (mapping >= n_token)).any()):
        raise ValueError("Atom-to-token mapping is outside the compact token array")
    if mapping.unique().numel() != n_token:
        raise ValueError("Each native token needs at least one atom")
    frames = features["frame_atom_index"]
    if frames.dtype != torch.long or bool(((frames < -1) | (frames >= n_atom)).any()):
        raise ValueError("Token frame references an absent atom")
    if bool(((frames < 0).any(-1) & features["has_frame"].bool()).any()):
        raise ValueError("Valid native frame contains an absent atom")
    reps = features["distogram_rep_atom_mask"].bool()
    if int(reps.sum()) != n_token or mapping[reps].unique().numel() != n_token:
        raise ValueError("Distogram representatives do not map one-to-one to tokens")
    return {"version": AXIS_AUDIT_VERSION, "atom_count": n_atom,
            "token_count": n_token, "fields": fields, "unknown_fields": []}


def _select_axes(features: dict, atom_keep, token_keep) -> dict:
    """Apply the explicit schema; values containing indices are remapped later."""
    result = {}
    for key, axes in FEATURE_AXES.items():
        value = features[key]
        for axis_index, axis in enumerate(axes):
            if axis == "A":
                value = value.index_select(axis_index, atom_keep)
            elif axis == "T":
                value = value.index_select(axis_index, token_keep)
        result[key] = value.clone()
    result["constraint_feature"] = {}
    return result


def remap_pairs(pair_indices, old_to_new):
    """Preserve readout row order exactly; an absent readout atom is an error."""
    import torch

    if (pair_indices.dtype != torch.long or pair_indices.ndim != 2
            or pair_indices.shape[1] != 2 or old_to_new.dtype != torch.long
            or old_to_new.ndim != 1):
        raise ValueError("Readout remapping requires integer pairs and one index map")
    if bool(((pair_indices < 0) | (pair_indices >= len(old_to_new))).any()):
        raise ValueError("Original readout index is outside its declared token array")
    mapped = old_to_new[pair_indices]
    if bool((mapped < 0).any()):
        raise ValueError("Fixed readout contains a deleted or unmapped token")
    if mapped.unique(dim=0).shape[0] != mapped.shape[0]:
        raise ValueError("Remapping collapsed distinct fixed readout pairs")
    return mapped


def _features(atoms):
    from protenix.data.core.featurizer import Featurizer
    from protenix.data.tokenizer import AtomArrayTokenizer
    from protenix.data.utils import data_type_transform, make_dummy_feature

    tokens = AtomArrayTokenizer(atoms).get_token_array()
    features = Featurizer(tokens, atoms, ref_pos_augment=False,
        lig_atom_rename=False, include_discont_poly_poly_bonds=True).get_all_input_features()
    features["constraint_feature"] = {}
    features = data_type_transform(make_dummy_feature(features, dummy_feats=["msa", "template"]))
    audit_feature_schema(features)
    return features, tokens


def _compact(parent: dict, full_atoms) -> tuple[dict, object, dict]:
    import numpy as np
    import torch
    from protenix.data.core.featurizer import Featurizer
    from latentmol.multimodel.external import tensor_tree_sha256

    original = parent["features"]
    audit = audit_feature_schema(original)
    optional_atom, optional_token = parent["optional_atom"], parent["optional_token"]
    atom_keep = torch.tensor([i for i in range(audit["atom_count"]) if i != optional_atom])
    token_keep = torch.tensor([i for i in range(audit["token_count"]) if i != optional_token])
    if int((original["atom_to_token_idx"] == optional_token).sum()) != 1:
        raise ValueError("Optional token must consist of exactly the preselected atom")
    atom_map = torch.full((audit["atom_count"],), -1, dtype=torch.long)
    token_map = torch.full((audit["token_count"],), -1, dtype=torch.long)
    atom_map[atom_keep] = torch.arange(len(atom_keep))
    token_map[token_keep] = torch.arange(len(token_keep))
    atoms = full_atoms[atom_keep.numpy()]
    # mol_atom_index is a within-molecule index, not an original atom identity.
    local_indices = np.empty(len(atoms), dtype=np.int32)
    for mol_id in np.unique(atoms.mol_id):
        selected = np.flatnonzero(atoms.mol_id == mol_id)
        local_indices[selected] = np.arange(len(selected))
    atoms.set_annotation("mol_atom_index", local_indices)
    regenerated, tokens = _features(atoms)
    expected = _select_axes(original, atom_keep, token_keep)
    expected["token_index"] = torch.arange(len(token_keep))
    expected["atom_to_token_idx"] = token_map[expected["atom_to_token_idx"]]
    expected["mol_atom_index"] = torch.from_numpy(local_indices.astype(np.int64))
    # Re-featurization recenters a smaller reference space. Deliberately undo
    # that translation, then derive legal native frames on retained coordinates.
    regenerated["ref_pos"] = expected["ref_pos"].clone()
    with_frames = Featurizer.get_token_frame(tokens, atoms,
        regenerated["ref_pos"], regenerated["ref_mask"])
    regenerated["has_frame"] = torch.tensor(with_frames.get_annotation("has_frame"), dtype=torch.long)
    regenerated["frame_atom_index"] = torch.tensor(with_frames.get_annotation("frame_atom_index"), dtype=torch.long)
    expected["has_frame"] = regenerated["has_frame"]
    expected["frame_atom_index"] = regenerated["frame_atom_index"]
    mismatches = [key for key in FEATURE_AXES if not torch.equal(regenerated[key], expected[key])]
    if mismatches:
        raise ValueError(f"Compact native regeneration changed undeclared retained features: {mismatches}")
    compact_audit = audit_feature_schema(regenerated)
    pairs = remap_pairs(parent["pair_indices"], token_map)
    metadata = {"kind": "compact_native_deletion_retained_reference_geometry",
        "original_to_endpoint_atom": atom_map.tolist(), "endpoint_to_original_atom": atom_keep.tolist(),
        "original_to_endpoint_token": token_map.tolist(), "endpoint_to_original_token": token_keep.tolist(),
        "feature_audit": compact_audit, "all_undeclared_retained_features_exact": True,
        "exceptions": ["contiguous token and molecule-local indices", "native frames recomputed after deletion"],
        "retained_ref_pos_exact": torch.equal(regenerated["ref_pos"], original["ref_pos"][atom_keep]),
        "retained_ref_pos_sha256": tensor_tree_sha256(regenerated["ref_pos"]),
        "feature_sha256": tensor_tree_sha256(regenerated),
        "pair_sha256": tensor_tree_sha256({"pairs": pairs, "bins": parent["target_bins"]})}
    return regenerated, pairs, metadata


def _molecule(case: dict, parent: dict, *, deleted: bool):
    from rdkit import Chem
    from protenix.data.core import ccd

    source = ccd.get_component_rdkit_mol(case["ligand_ccd_id"])
    if source is None:
        raise ValueError("Preselected ligand is absent from the pinned CCD")
    mol = Chem.Mol(source)
    inverse = {int(index): str(name) for name, index in source.atom_map.items()}
    for atom in mol.GetAtoms():
        atom.SetProp("study_atom_name", inverse[atom.GetIdx()])
    mol = Chem.RemoveHs(mol)
    Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    full_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    if full_smiles != parent["metadata"]["full_smiles"]:
        raise ValueError("Authentic full endpoint chemistry changed")
    if deleted:
        indices = [a.GetIdx() for a in mol.GetAtoms()
                   if a.GetProp("study_atom_name") == case["optional_atom_name"]]
        if len(indices) != 1:
            raise ValueError("Optional atom does not have one unique CCD identity")
        optional = mol.GetAtomWithIdx(indices[0])
        if optional.GetDegree() != 1 or optional.GetSymbol() != case["optional_element"]:
            raise ValueError("Deletion is not the predeclared terminal halogen")
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(indices[0])
        mol = rw.GetMol()
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(False)
        Chem.SanitizeMol(mol)
    expected_smiles = parent["metadata"]["deleted_smiles" if deleted else "full_smiles"]
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    if (canonical != expected_smiles or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms())):
        raise ValueError("Ordinary endpoint is not the declared connected neutral chemistry")
    return mol


def _fresh(parent: dict, case: dict, output_root: Path, seed: int, *, deleted: bool):
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from latentmol.multimodel.external import tensor_tree_sha256

    label = "deleted" if deleted else "full"
    mol = _molecule(case, parent, deleted=deleted)
    names = [a.GetProp("study_atom_name") for a in mol.GetAtoms()]
    elements = [a.GetSymbol().upper() for a in mol.GetAtoms()]
    hydrogen_counts = {a.GetProp("study_atom_name"): int(a.GetTotalNumHs()) for a in mol.GetAtoms()}
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    # Both endpoints use the same ordinary importer and independently embedded
    # seed. These are new reference geometries, not a fixed-pose affinity test.
    embedded = Chem.AddHs(mol)
    embedded.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    if AllChem.EmbedMolecule(embedded, params) != 0:
        raise ValueError(f"Declared independent ETKDG embedding failed for {label}")
    embedded = Chem.RemoveHs(embedded)
    if not embedded.GetConformer().Is3D() or embedded.GetNumAtoms() != len(names):
        raise ValueError("Ordinary MOL endpoint lost its declared 3D heavy-atom set")
    mol_path = output_root / f"{case['case_id']}.{label}.mol"
    if mol_path.exists():
        raise FileExistsError(mol_path)
    mol_path.write_text(Chem.MolToMolBlock(embedded), encoding="utf-8")
    loaded = Chem.MolFromMolFile(str(mol_path), removeHs=True)
    if loaded is None or Chem.MolToSmiles(loaded, canonical=True, isomericSmiles=False) != canonical:
        raise ValueError("Written ordinary MOL did not preserve endpoint chemistry")
    sample = {"name": f"{case['case_id']}-{label}", "sequences": [
        {"proteinChain": {"sequence": parent["metadata"]["protein_sequence"], "count": 1}},
        {"ligand": {"ligand": "FILE_" + str(mol_path.resolve()), "count": 1}}]}
    atoms = SampleDictToFeatures(sample).get_atom_array()
    features, _ = _features(atoms)
    ligand = torch.nonzero(features["is_ligand"], as_tuple=False).flatten().tolist()
    if len(ligand) != len(names) or [str(atoms.element[i]).upper() for i in ligand] != elements:
        raise ValueError("FILE importer changed the declared heavy-atom order or elements")
    if [str(atoms.atom_name[i]) for i in ligand] != native_file_atom_names(elements):
        raise ValueError("FILE importer changed the pinned file-order atom naming convention")
    original_n = len(parent["features"]["ref_pos"])
    original_t = len(parent["features"]["token_index"])
    atom_map = torch.full((original_n,), -1, dtype=torch.long)
    token_map = torch.full((original_t,), -1, dtype=torch.long)
    native_names = {name: index for name, index in zip(names, ligand)}
    original_names = {row["deposited_atom_name"]: row for row in parent["metadata"]["native_atom_correspondence"]}
    for name, endpoint_index in native_names.items():
        original = original_names[name]
        atom_map[original["atom_index"]] = endpoint_index
        token_map[original["token_index"]] = features["atom_to_token_idx"][endpoint_index]
    # Standard protein atoms precede the ligand in the sealed parent and both
    # ordinary imports; require all their retained chemical features exactly.
    source_protein = torch.nonzero(parent["features"]["is_protein"], as_tuple=False).flatten()
    target_protein = torch.nonzero(features["is_protein"], as_tuple=False).flatten()
    if len(source_protein) != len(target_protein):
        raise ValueError("Ordinary endpoint changed the fixed native protein atom set")
    for key in ("ref_element", "ref_charge", "ref_atom_name_chars", "ref_mask", "ref_pos", "ref_space_uid"):
        if not torch.equal(parent["features"][key][source_protein], features[key][target_protein]):
            raise ValueError(f"Ordinary endpoint changed fixed protein correspondence: {key}")
    atom_map[source_protein] = target_protein
    for old_atom, new_atom in zip(source_protein.tolist(), target_protein.tolist()):
        old_token = int(parent["features"]["atom_to_token_idx"][old_atom])
        new_token = int(features["atom_to_token_idx"][new_atom])
        if int(token_map[old_token]) not in (-1, new_token):
            raise ValueError("Protein token mapping is not one-to-one")
        token_map[old_token] = new_token
    expected_missing = [parent["optional_atom"]] if deleted else []
    if torch.nonzero(atom_map < 0, as_tuple=False).flatten().tolist() != expected_missing:
        raise ValueError("Ordinary endpoint lost an undeclared atom identity")
    endpoint_atom_map = torch.full((len(features["ref_pos"]),), -1, dtype=torch.long)
    present = torch.nonzero(atom_map >= 0, as_tuple=False).flatten()
    endpoint_atom_map[atom_map[present]] = present
    if bool((endpoint_atom_map < 0).any()) or atom_map[present].unique().numel() != len(present):
        raise ValueError("Ordinary atom identity map is not a bijection")
    endpoint_token_map = torch.full((len(features["token_index"]),), -1, dtype=torch.long)
    present_t = torch.nonzero(token_map >= 0, as_tuple=False).flatten()
    endpoint_token_map[token_map[present_t]] = present_t
    if bool((endpoint_token_map < 0).any()) or token_map[present_t].unique().numel() != len(present_t):
        raise ValueError("Ordinary token identity map is not a bijection")
    expected_bonds = parent["features"]["token_bonds"][endpoint_token_map][:, endpoint_token_map]
    if not torch.equal(features["token_bonds"], expected_bonds):
        raise ValueError("Ordinary FILE importer changed mapped native bond connectivity")
    pairs = remap_pairs(parent["pair_indices"], token_map)
    metadata = {"kind": "fresh_ordinary_FILE_native_" + label,
        "canonical_smiles": canonical, "chemical_validity": "RDKit sanitize/neutral/connected/roundtrip passed",
        "mol_path": str(mol_path.resolve()), "mol_sha256": _sha(mol_path),
        "mol_heavy_atom_order_ccd_names": names, "total_hydrogen_counts": hydrogen_counts,
        "native_generated_atom_names_exact": True,
        "native_atom_name_correspondence": [{"original_ccd_name": name, "native_name": str(atoms.atom_name[index]),
            "native_atom_index": index, "native_token_index": int(features["atom_to_token_idx"][index])}
            for name, index in zip(names, ligand)],
        "original_to_endpoint_atom": atom_map.tolist(), "endpoint_to_original_atom": endpoint_atom_map.tolist(),
        "original_to_endpoint_token": token_map.tolist(), "endpoint_to_original_token": endpoint_token_map.tolist(),
        "feature_audit": audit_feature_schema(features), "feature_sha256": tensor_tree_sha256(features),
        "pair_sha256": tensor_tree_sha256({"pairs": pairs, "bins": parent["target_bins"]}),
        "geometry_convention": "Independent ETKDGv3 3D MOL; same seed and FILE importer for full/deleted; native per-space centering; no augmentation",
        "hydrogen_convention": "Delete declared terminal halogen, recalculate implicit H, embed with explicit H then remove H before FILE import",
        "embedding_seed": seed, "embedding_threads": 1,
        "reference_geometry_sha256": tensor_tree_sha256(features["ref_pos"])}
    return features, pairs, metadata


def prepare_deletion_case(parent: dict, case: dict, provenance: dict,
                          output_root: Path, seed: int) -> dict:
    """Augment one authenticated first-stage case without mutating its parent."""
    _require_allocation()
    import numpy as np
    import torch
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from latentmol.multimodel.external import tensor_tree_sha256

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if parent.get("schema_version") != 1 or parent.get("case_id") != case["case_id"]:
        raise ValueError("Unsupported or mismatched original case bundle")
    if parent["metadata"]["case_definition"] != case:
        raise ValueError("Predeclared case definition changed after parent preparation")
    original_hash = tensor_tree_sha256(parent["features"])
    if original_hash != parent["metadata"]["native_feature_sha256"]:
        raise ValueError("Parent native feature hash is not authentic")
    readout_hash = tensor_tree_sha256({"pairs": parent["pair_indices"], "bins": parent["target_bins"]})
    if readout_hash != parent["metadata"]["target_pair_sha256"]:
        raise ValueError("Parent fixed experimental readout is not authentic")
    sample = {"name": case["case_id"], "sequences": [
        {"proteinChain": {"sequence": parent["metadata"]["protein_sequence"], "count": 1}},
        {"ligand": {"ligand": "CCD_" + case["ligand_ccd_id"], "count": 1}}]}
    full_atoms = SampleDictToFeatures(sample).get_atom_array()
    regenerated, _ = _features(full_atoms)
    if tensor_tree_sha256(regenerated) != original_hash:
        raise ValueError("Independent original native reconstruction differs from authenticated parent")
    compact, compact_pairs, compact_meta = _compact(parent, full_atoms)
    fresh_full, full_pairs, full_meta = _fresh(parent, case, output_root, seed, deleted=False)
    fresh_deleted, deleted_pairs, deleted_meta = _fresh(parent, case, output_root, seed, deleted=True)
    validate_hydrogen_replacement(full_meta["total_hydrogen_counts"],
        deleted_meta["total_hydrogen_counts"], case["optional_atom_name"],
        parent["metadata"]["optional_neighbor_name"])
    if tensor_tree_sha256(parent["features"]) != original_hash:
        raise ValueError("Deletion preparation mutated the frozen full parent features")
    if any(len(p) != len(parent["pair_indices"]) for p in (compact_pairs, full_pairs, deleted_pairs)):
        raise ValueError("Deletion preparation changed the fixed readout denominator")
    metadata = dict(parent["metadata"], deletion_preparation_version=PREPARATION_VERSION,
        deletion_preparation_provenance=provenance,
        parent_features_unchanged=True, parent_readout_sha256=readout_hash,
        deletion_original_reconstruction_exact=True,
        compact_reference_status="prepared_not_native_validated",
        fresh_deleted_reference_status="prepared_not_native_validated",
        fresh_full_reference_status="prepared_not_native_validated",
        compact_deletion=compact_meta, fresh_full=full_meta, fresh_deleted=deleted_meta,
        model_scoring_performed=False)
    return dict(parent, metadata=metadata,
        compact_features=compact, compact_pair_indices=compact_pairs,
        fresh_full_features=fresh_full, fresh_full_pair_indices=full_pairs,
        fresh_deleted_features=fresh_deleted, fresh_pair_indices=deleted_pairs,
        compact_atom_map=torch.tensor(compact_meta["original_to_endpoint_atom"]),
        compact_token_map=torch.tensor(compact_meta["original_to_endpoint_token"]),
        fresh_full_atom_map=torch.tensor(full_meta["original_to_endpoint_atom"]),
        fresh_full_token_map=torch.tensor(full_meta["original_to_endpoint_token"]),
        fresh_deleted_atom_map=torch.tensor(deleted_meta["original_to_endpoint_atom"]),
        fresh_deleted_token_map=torch.tensor(deleted_meta["original_to_endpoint_token"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-source-root", type=Path, required=True)
    parser.add_argument("--native-source-sha256", required=True)
    parser.add_argument("--ccd-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    job_id = _require_allocation()
    args.output_root.mkdir(parents=True, exist_ok=False)
    report = {"preparation_version": PREPARATION_VERSION, "job_id": job_id,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cases_json_sha256": _sha(args.cases_json), "parent_manifest_sha256": _sha(args.parent_manifest),
        "preparation_source_sha256": _sha(Path(__file__)), "model_scoring_performed": False,
        "cases": [], "status": "running", "all_three_inputs_supported": False}
    try:
        if report["parent_manifest_sha256"] != args.parent_manifest_sha256:
            raise ValueError("Parent preparation manifest checksum mismatch")
        parent_manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
        plan = json.loads(args.cases_json.read_text(encoding="utf-8"))
        if (plan.get("schema_version") != 1 or plan.get("native_commit") != NATIVE_COMMIT
                or parent_manifest.get("status") != "prepared_not_native_validated"
                or parent_manifest.get("all_three_inputs_supported") is not True
                or parent_manifest.get("cases_json_sha256") != report["cases_json_sha256"]):
            raise ValueError("Parent manifest or sealed case plan is unsupported")
        cases = plan["cases"]
        if [c["role"] for c in cases] != ["development", "confirmation_1", "confirmation_2"]:
            raise ValueError("Require the original score-blind three-case order")
        if [r["case_id"] for r in parent_manifest["cases"]] != [c["case_id"] for c in cases]:
            raise ValueError("Parent preparation changed the score-blind case order")
        provenance = _bind_native(args.native_source_root.resolve(), args.native_source_sha256,
                                  args.ccd_root.resolve())
        import torch

        for case, row in zip(cases, parent_manifest["cases"]):
            path = Path(row["bundle_path"])
            if _sha(path) != row["bundle_sha256"]:
                raise ValueError("Parent case bundle checksum mismatch")
            parent = torch.load(path, map_location="cpu", weights_only=True)
            if parent["metadata"] != row["metadata"]:
                raise ValueError("Parent case metadata differs from its authenticated manifest")
            if _sha(args.input_root / case["structure_file"]) != case["structure_sha256"]:
                raise ValueError("Original experimental structure snapshot checksum mismatch")
            bundle = prepare_deletion_case(parent, case, dict(provenance,
                parent_bundle_path=str(path.resolve()), parent_bundle_sha256=row["bundle_sha256"],
                parent_manifest_sha256=args.parent_manifest_sha256), args.output_root, args.seed)
            output = args.output_root / f"{case['case_id']}.pt"
            torch.save(bundle, output)
            summary = {"case_id": case["case_id"], "role": case["role"],
                "bundle_path": str(output.resolve()), "bundle_sha256": _sha(output),
                "metadata": bundle["metadata"]}
            _write_json(args.output_root / f"{case['case_id']}.json", summary)
            report["cases"].append(summary)
        report["status"] = "deletion_inputs_prepared_not_native_validated"
        report["all_three_inputs_supported"] = True
        return_code = 0
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        return_code = 1
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(args.output_root / "preparation.json", report)
    print(json.dumps({"status": report["status"], "case_count": len(report["cases"]),
        "report": str((args.output_root / "preparation.json").resolve())}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
