"""Contract tests for the train-only Stage 3 support diagnostic; run on HPC."""

import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from latentmol.boltz.training_support import motif_flags, top_neighbors, run_support_audit
from latentmol.data import _split_digest, _write_npz
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec


def test_motifs_distinguish_oxygen_ring_and_aniline():
    ring = motif_flags(Chem.MolFromSmiles("NC(Cc1ccc2c(c1)O2)C(=O)O"))
    aniline = motif_flags(Chem.MolFromSmiles("Nc1ccc(CC(N)C(=O)O)cc1"))
    assert ring["oxygen_three_ring"]
    assert ring["oxygen_three_ring_two_aromatic_atoms"]
    assert not aniline["oxygen_three_ring"]
    assert aniline["aniline_substructure"]


def test_top_neighbors_keep_exact_match_and_stable_order():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles = ["CC", "CCC", "CC"]
    references = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in smiles]
    top = top_neighbors(references[0], references, smiles, limit=2)
    assert top == [{"smiles": "CC", "tanimoto": 1.0},
                   {"smiles": "CC", "tanimoto": 1.0}]


def test_full_audit_excludes_validation_and_canonicalizes_exact_query(tmp_path):
    data_dir = tmp_path / "dataset"
    audit_dir = tmp_path / "audit"
    data_dir.mkdir()
    audit_dir.mkdir()
    spec = GraphSpec()
    smiles = ["CC", "CCC", "O"]
    split = np.array([0, 1, 2], dtype=np.int8)
    atoms = np.zeros((3, spec.max_nodes), dtype=np.int8)
    atoms[0, :2] = 1
    atoms[1, :3] = 1
    atoms[2, 0] = 3
    arrays = {"smiles": np.asarray(smiles), "split": split, "atom_types": atoms}
    _write_npz(data_dir / "dataset.npz", arrays)
    (data_dir / "manifest.json").write_text(json.dumps({
        "dataset_sha256": file_sha256(data_dir / "dataset.npz"),
        "split_sha256": _split_digest(smiles, split),
        "spec": spec.to_dict(), "spec_fingerprint": spec.fingerprint(),
        "counts": {"train": 1}}))
    (audit_dir / "run.json").write_text(json.dumps({
        "name": "selected-molecule-chemical-audit", "status": "completed",
        "run_id": "fixture-audit", "source_archive_sha256": "fixture-source"}))
    (audit_dir / "molecules.json").write_text(json.dumps([
        {"input_smiles": "C-C", "native_standardized_smiles": "CC", "selected_by": {}},
        {"input_smiles": "CCC", "native_standardized_smiles": "CCC", "selected_by": {}}]))
    result = run_support_audit(data_dir, audit_dir, tmp_path / "runs")
    rows = json.loads((result / "candidate_training_support.json").read_text())
    assert rows[0]["comparisons"]["submitted"]["canonical_query_smiles"] == "CC"
    assert rows[0]["comparisons"]["submitted"]["exact_training_identity"]
    assert not rows[1]["comparisons"]["submitted"]["exact_training_identity"]
    assert json.loads((result / "summary.json").read_text())["train_reference_molecules"] == 1
