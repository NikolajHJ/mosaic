"""Allocated-CPU native hard-input preparation for the occupancy study.

Nothing here runs a model or selects a case using its score.  Full native
features and the fixed experimental readout are sealed before a GPU run.
Compacted and independently prepared deletions remain explicitly unavailable
in this first C/Z parity-and-gradient phase.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

PREPARATION_VERSION = "protenix-occupancy-native-case-preparation-v1"
NATIVE_COMMIT = "510faf9f5cc8fbbbb8d9055934d3fa7e6877c76d"
CCD_SHA256 = {
    "components.cif": "7240b17369ccfbbcc86e2d02dc8c9db59f46c32e0420f58889c6c121c60bfef0",
    "components.cif.rdkit_mol.pkl": "2d6caced2d26c62015115a1d0a50f4106755a300e0c2a2d2b5c101c9038dcbcd",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _require_allocation() -> str:
    job_id = os.environ.get("LSB_JOBID", "")
    if not job_id.isdigit():
        raise RuntimeError("Numerical preparation requires an allocated DTU LSF job")
    return job_id


def _bind_native(source_root: Path, expected_source_hash: str, ccd_root: Path) -> dict:
    """Bind paths before importing CCD; authenticate the pickle before unpickling."""
    sys.path.insert(0, str(source_root))
    from latentmol.multimodel.external import native_source_sha256

    actual = native_source_sha256(source_root, "protenix")
    if actual != expected_source_hash:
        raise ValueError(f"Native Protenix source hash mismatch: {actual}")
    assets = {}
    for name, expected in CCD_SHA256.items():
        path = ccd_root / name
        actual_hash = _sha(path)
        if actual_hash != expected:
            raise ValueError(f"Pinned CCD asset hash mismatch: {path}: {actual_hash}")
        assets[name] = {"path": str(path.resolve()), "sha256": actual_hash}
    if "protenix.data.core.ccd" in sys.modules:
        raise RuntimeError("CCD imported before study asset binding; start a fresh process")
    from configs.configs_data import data_configs

    data_configs["ccd_components_file"] = str(ccd_root / "components.cif")
    data_configs["ccd_components_rdkit_mol_file"] = str(ccd_root / "components.cif.rdkit_mol.pkl")
    from protenix.data.inference import json_to_feature

    imported = Path(json_to_feature.__file__).resolve()
    if not imported.is_relative_to(source_root):
        raise ValueError("Official featurizer imported outside the authenticated source")
    return {"native_commit": NATIVE_COMMIT, "native_source_sha256": actual,
            "ccd_assets": assets}


def _column(category, name: str) -> list[str]:
    return [str(value) for value in category[name].as_array(str)]


def _read_crystal(case: dict, path: Path) -> tuple[str, dict, dict, dict]:
    """Read real coordinates; resolve alternatives by occupancy then alt ID."""
    import numpy as np
    from biotite.structure.io.pdbx import CIFFile

    if _sha(path) != case["structure_sha256"]:
        raise ValueError(f"Crystal snapshot checksum mismatch: {path}")
    block = CIFFile.read(path).block
    poly = block["entity_poly"]
    ids = _column(poly, "entity_id")
    sequences = _column(poly, "pdbx_seq_one_letter_code_can")
    sequence = "".join(sequences[ids.index(case["protein_entity_id"])].split())
    if not sequence or any(letter not in "ACDEFGHIKLMNPQRSTVWY" for letter in sequence):
        raise ValueError("Only a declared standard protein sequence is supported")
    asym = block["struct_asym"]
    asym_entities = dict(zip(_column(asym, "id"), _column(asym, "entity_id")))
    if asym_entities.get(case["protein_label_asym_id"]) != case["protein_entity_id"]:
        raise ValueError("Protein chain/entity selection disagrees with the deposition")

    site = block["atom_site"]
    names = ("label_asym_id", "label_seq_id", "label_atom_id", "label_comp_id",
             "label_alt_id", "pdbx_PDB_model_num", "Cartn_x", "Cartn_y", "Cartn_z",
             "occupancy", "type_symbol")
    columns = {name: _column(site, name) for name in names}
    records = [{name: columns[name][i] for name in names}
               for i in range(len(columns["label_atom_id"]))]
    models = sorted({row["pdbx_PDB_model_num"] for row in records})
    if models != ["1"]:
        raise ValueError(f"Pilot requires one experimental model, found {models}")
    protein, ligand, choices = {}, {}, {}
    for row in records:
        chain = row["label_asym_id"]
        if chain not in (case["protein_label_asym_id"], case["ligand_label_asym_id"]):
            continue
        if row["type_symbol"].upper() in ("H", "D"):
            continue
        is_protein = chain == case["protein_label_asym_id"]
        if not is_protein and row["label_comp_id"] != case["ligand_ccd_id"]:
            raise ValueError("Selected ligand chain contains an unexpected component")
        key = (int(row["label_seq_id"]), row["label_atom_id"]) if is_protein else row["label_atom_id"]
        prefix = "protein" if is_protein else "ligand"
        # The rank is independent of model scores; ties use the lexical alt ID.
        rank = (-float(row["occupancy"]), row["label_alt_id"])
        compound_key = (prefix, key)
        if compound_key not in choices or rank < choices[compound_key][0]:
            xyz = np.asarray([float(row[f"Cartn_{axis}"]) for axis in "xyz"], dtype=np.float32)
            if not np.isfinite(xyz).all():
                raise ValueError("Experimental coordinates are not finite")
            (protein if is_protein else ligand)[key] = xyz
            choices[compound_key] = (rank, row["label_alt_id"], float(row["occupancy"]))
    if not protein or not ligand:
        raise ValueError("Selected native protein or ligand instance has no coordinates")
    if "struct_conn" in block:
        conn = block["struct_conn"]
        types = _column(conn, "conn_type_id")
        left = _column(conn, "ptnr1_label_asym_id")
        right = _column(conn, "ptnr2_label_asym_id")
        for kind, lchain, rchain in zip(types, left, right):
            if kind.startswith("covale") and case["ligand_label_asym_id"] in (lchain, rchain):
                raise ValueError("Pilot excludes covalently attached crystal ligands")
    choice_log = {str(key): {"alt_id": value[1], "occupancy": value[2]}
                  for key, value in choices.items() if value[1] not in (".", "?")}
    return sequence, protein, ligand, {"alternative_selection": choice_log,
        "resolved_protein_atom_count": len(protein), "resolved_ligand_heavy_atoms": len(ligand)}


def _chemistry(case: dict) -> dict:
    """Validate both endpoints without any model calls or learned selection."""
    from rdkit import Chem
    from protenix.data.core import ccd

    original = ccd.get_component_rdkit_mol(case["ligand_ccd_id"])
    if original is None:
        raise ValueError("Selected component is absent from the authenticated CCD")
    mol = Chem.Mol(original)
    inverse = {int(index): name for name, index in original.atom_map.items()}
    for atom in mol.GetAtoms():
        atom.SetProp("study_atom_name", inverse[atom.GetIdx()])
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    Chem.RemoveStereochemistry(mol)
    # The pilot uses a declared stereo-free convention on naturally achiral cases.
    if Chem.FindMolChiralCenters(mol, includeUnassigned=True):
        raise ValueError("Retained stereochemical ambiguity is excluded")
    if len(Chem.GetMolFrags(mol)) != 1 or any(atom.GetFormalCharge() != 0 or atom.GetIsotope() != 0 for atom in mol.GetAtoms()):
        raise ValueError("Both endpoints require connected neutral nonisotopic chemistry")
    atom_names = [atom.GetProp("study_atom_name") for atom in mol.GetAtoms()]
    optional_index = atom_names.index(case["optional_atom_name"])
    optional = mol.GetAtomWithIdx(optional_index)
    if optional.GetSymbol() != case["optional_element"] or optional.GetDegree() != 1:
        raise ValueError("Preselected optional atom is not the declared terminal element")
    if optional.GetFormalCharge() or optional.GetIsAromatic():
        raise ValueError("Charged or aromatic optional atoms are outside this pilot")
    neighbor = optional.GetNeighbors()[0]
    if mol.GetBondBetweenAtoms(optional_index, neighbor.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("Terminal deletion requires a native single bond")
    rw = Chem.RWMol(mol)
    rw.RemoveAtom(optional_index)
    deleted = rw.GetMol()
    # Recalculate implicit hydrogens after replacing the terminal halogen by H.
    for atom in deleted.GetAtoms():
        atom.SetNoImplicit(False)
    Chem.SanitizeMol(deleted)
    if not deleted.GetNumAtoms() or len(Chem.GetMolFrags(deleted)) != 1 or any(atom.GetFormalCharge() for atom in deleted.GetAtoms()):
        raise ValueError("Preselected deletion does not make a valid neutral connected endpoint")
    full_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    deleted_smiles = Chem.MolToSmiles(deleted, canonical=True, isomericSmiles=False)
    bonds = [[min(atom_names[b.GetBeginAtomIdx()], atom_names[b.GetEndAtomIdx()]),
              max(atom_names[b.GetBeginAtomIdx()], atom_names[b.GetEndAtomIdx()]),
              str(b.GetBondType())] for b in mol.GetBonds()]
    graph_description = {"atoms": [[name, atom.GetSymbol(), atom.GetFormalCharge()]
                                  for name, atom in zip(atom_names, mol.GetAtoms())],
                         "bonds": sorted(bonds)}
    deleted_names = [atom.GetProp("study_atom_name") for atom in deleted.GetAtoms()]
    deleted_graph = {"atoms": [[name, atom.GetSymbol(), atom.GetFormalCharge()]
                               for name, atom in zip(deleted_names, deleted.GetAtoms())],
        "bonds": sorted([[min(deleted_names[b.GetBeginAtomIdx()], deleted_names[b.GetEndAtomIdx()]),
                          max(deleted_names[b.GetBeginAtomIdx()], deleted_names[b.GetEndAtomIdx()]),
                          str(b.GetBondType())] for b in deleted.GetBonds()]),
        "implicit_hydrogens": {name: int(atom.GetNumImplicitHs())
                               for name, atom in zip(deleted_names, deleted.GetAtoms())}}
    graph_description["implicit_hydrogens"] = {name: int(atom.GetNumImplicitHs())
        for name, atom in zip(atom_names, mol.GetAtoms())}
    graph_hash = hashlib.sha256(json.dumps(graph_description, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    deleted_hash = hashlib.sha256(json.dumps(deleted_graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"full_smiles": full_smiles, "deleted_smiles": deleted_smiles,
            "ligand_heavy_atoms": int(mol.GetNumAtoms()), "ligand_graph_sha256": graph_hash,
            "deleted_graph_sha256": deleted_hash,
            "full_smiles_sha256": hashlib.sha256(full_smiles.encode()).hexdigest(),
            "deleted_smiles_sha256": hashlib.sha256(deleted_smiles.encode()).hexdigest(),
            "full_graph": graph_description, "deleted_graph": deleted_graph,
            "ccd_heavy_atom_names": atom_names,
            "optional_neighbor_name": neighbor.GetProp("study_atom_name"),
            "deletion_convention": "Delete the preselected terminal halogen and recalculate implicit H; stereo-free canonical SMILES"}


def prepare_case(case: dict, plan: dict, input_root: Path, provenance: dict, seed: int) -> dict:
    _require_allocation()
    import numpy as np
    import torch
    from protenix.data.core.featurizer import Featurizer
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.tokenizer import AtomArrayTokenizer
    from protenix.data.utils import data_type_transform, make_dummy_feature
    from latentmol.multimodel.external import tensor_tree_sha256

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    crystal_path = input_root / case["structure_file"]
    sequence, protein_xyz, ligand_xyz, crystal_metadata = _read_crystal(case, crystal_path)
    if len(sequence) > plan["max_protein_tokens"]:
        raise ValueError("Declared protein exceeds the prospective token limit")
    chemistry = _chemistry(case)
    n_ligand = chemistry["ligand_heavy_atoms"]
    if not plan["min_ligand_heavy_atoms"] <= n_ligand <= plan["max_ligand_heavy_atoms"] or n_ligand != case["expected_ligand_heavy_atoms"]:
        raise ValueError("Ligand heavy-atom count disagrees with the sealed panel")
    sample = {"name": case["case_id"], "sequences": [
        {"proteinChain": {"sequence": sequence, "count": 1}},
        {"ligand": {"ligand": "CCD_" + case["ligand_ccd_id"], "count": 1}}]}
    atoms = SampleDictToFeatures(sample).get_atom_array()
    tokens = AtomArrayTokenizer(atoms).get_token_array()
    # Native CCD reference coordinates are retained. Only their random rigid
    # augmentation is disabled, as declared before preparation/model scoring.
    features = Featurizer(tokens, atoms, ref_pos_augment=False,
                          lig_atom_rename=False, include_discont_poly_poly_bonds=True).get_all_input_features()
    features["constraint_feature"] = {}
    features = make_dummy_feature(features, dummy_feats=["msa", "template"])
    features = data_type_transform(features)
    ligand_indices = torch.nonzero(features["is_ligand"], as_tuple=False).flatten().tolist()
    if len(ligand_indices) != n_ligand:
        raise ValueError("Native importer changed the declared ligand heavy-atom set")
    native_names = [str(atoms.atom_name[index]) for index in ligand_indices]
    if set(native_names) != set(chemistry["ccd_heavy_atom_names"]) or len(set(native_names)) != len(native_names):
        raise ValueError("Native/CCD heavy-atom names do not map one-to-one")
    if set(ligand_xyz) != set(native_names):
        raise ValueError("The selected experimental ligand lacks the exact native heavy-atom set")
    optional_atom = ligand_indices[native_names.index(case["optional_atom_name"])]
    optional_token = int(features["atom_to_token_idx"][optional_atom])
    ligand_tokens = features["atom_to_token_idx"][ligand_indices]
    if ligand_tokens.unique().numel() != n_ligand:
        raise ValueError("Pilot requires native one-token-per-ligand-heavy-atom input")
    if int(features["ref_element"][optional_atom].sum()) != 1:
        raise ValueError("Optional atom is not a hard native element input")
    expected_atomic = {"I": 53, "Br": 35}[case["optional_element"]] - 1
    if int(features["ref_element"][optional_atom].argmax()) != expected_atomic:
        raise ValueError("Native element encoding does not equal atomic number minus one")
    # Construct targets from deposited distogram representative atom coordinates,
    # never from native ideal/reference conformers or predictions.
    representative_indices = torch.nonzero(features["distogram_rep_atom_mask"], as_tuple=False).flatten().tolist()
    if len(representative_indices) != len(tokens):
        raise ValueError("Native distogram representative order is not one-to-one with tokens")
    rep_by_token = {}
    for index in representative_indices:
        token = int(features["atom_to_token_idx"][index])
        if token in rep_by_token:
            raise ValueError("Native distogram token has multiple representative atoms")
        rep_by_token[token] = index
    retained_ligand = [int(token) for token in ligand_tokens if int(token) != optional_token]
    protein_reps, excluded_protein = [], []
    for token, index in sorted(rep_by_token.items()):
        if not bool(features["is_protein"][index]):
            continue
        key = (int(atoms.res_id[index]), str(atoms.atom_name[index]))
        if key in protein_xyz:
            protein_reps.append((token, protein_xyz[key]))
        else:
            excluded_protein.append({"native_token": token, "label_seq_id": key[0], "atom_name": key[1]})
    if not protein_reps or len(retained_ligand) != n_ligand - 1:
        raise ValueError("Readout needs resolved protein representatives and retained ligand atoms")
    pairs, distances = [], []
    for ligand_token in retained_ligand:
        index = rep_by_token[ligand_token]
        xyz = ligand_xyz[str(atoms.atom_name[index])]
        for protein_token, protein_coord in protein_reps:
            pairs.append(sorted((ligand_token, protein_token)))
            distances.append(float(np.linalg.norm(xyz - protein_coord)))
    pair_indices = torch.tensor(pairs, dtype=torch.long)
    true_distances = torch.tensor(distances, dtype=torch.float32)
    boundaries = torch.linspace(2.3125, 21.6875, 63)
    target_bins = (true_distances[:, None] > boundaries[None, :]).sum(-1).long()
    if pair_indices.unique(dim=0).shape[0] != pair_indices.shape[0] or bool((pair_indices == optional_token).any()):
        raise ValueError("Fixed readout contains duplicate pairs or optional-atom terms")
    metadata = {"case_id": case["case_id"], "role": case["role"], "case_definition": case,
        "preparation_version": PREPARATION_VERSION, "seed": seed, **provenance,
        **chemistry, **crystal_metadata, "protein_sequence": sequence,
        "protein_tokens": len(sequence), "protein_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "experimental_structure_sha256": _sha(crystal_path),
        "native_feature_sha256": tensor_tree_sha256(features),
        "target_pair_sha256": tensor_tree_sha256({"pairs": pair_indices, "bins": target_bins}),
        "target_pair_count": len(pairs), "excluded_unresolved_protein_representatives": excluded_protein,
        "native_atom_correspondence": [{"atom_index": index,
            "token_index": int(features["atom_to_token_idx"][index]), "deposited_atom_name": str(atoms.atom_name[index])}
            for index in ligand_indices],
        "optional_atom_last": optional_atom == len(atoms) - 1,
        "reference_geometry": "Native CCD conformers, native per-ref-space centering retained, ref_pos_augment=False; experimental coordinates supply target bins only",
        "msa": "Official make_dummy_feature query-only convention; no sequence search",
        "templates": "Official make_dummy_feature all-masked template tensors; no template input",
        "compact_reference_status": "not_run_first_C_Z_phase",
        "fresh_deleted_reference_status": "not_run_first_C_Z_phase",
        "model_scoring_performed": False}
    return {"schema_version": 1, "case_id": case["case_id"], "features": features,
        "optional_atom": optional_atom, "optional_token": optional_token,
        "pair_indices": pair_indices, "target_bins": target_bins,
        "experimental_pair_distances": true_distances, "metadata": metadata,
        "compact_features": None, "fresh_deleted_features": None,
        "compact_pair_indices": None, "fresh_pair_indices": None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "cases_json_sha256": _sha(args.cases_json), "preparation_source_sha256": _sha(Path(__file__)),
        "model_scoring_performed": False, "cases": [], "status": "running"}
    try:
        plan = json.loads(args.cases_json.read_text(encoding="utf-8"))
        if plan.get("schema_version") != 1 or plan.get("native_commit") != NATIVE_COMMIT:
            raise ValueError("Unaudited case schema or native commit")
        if [case["role"] for case in plan["cases"]] != ["development", "confirmation_1", "confirmation_2"]:
            raise ValueError("Require sealed development/two-confirmation case order")
        provenance = _bind_native(args.native_source_root.resolve(), args.native_source_sha256,
                                  args.ccd_root.resolve())
        import torch

        # All three supported hard inputs are prepared before any model score.
        # Persist every completed input even if a later case fails support.
        for case in plan["cases"]:
            bundle = prepare_case(case, plan, args.input_root, provenance, args.seed)
            bundle_path = args.output_root / (case["case_id"] + ".pt")
            torch.save(bundle, bundle_path)
            summary = {"case_id": case["case_id"], "role": case["role"],
                "bundle_path": str(bundle_path.resolve()), "bundle_sha256": _sha(bundle_path),
                "metadata": bundle["metadata"]}
            _write_json(args.output_root / (case["case_id"] + ".json"), summary)
            report["cases"].append(summary)
        report["status"] = "prepared_not_native_validated"
        report["all_three_inputs_supported"] = True
        return_code = 0
    except Exception:
        report["status"] = "failed"
        report["all_three_inputs_supported"] = False
        report["error"] = traceback.format_exc()
        return_code = 1
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(args.output_root / "preparation.json", report)
    print(json.dumps({"status": report["status"], "case_count": len(report["cases"]),
                      "report": str((args.output_root / "preparation.json").resolve())}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
