"""Train-only structural support audit for the frozen Stage 2 candidate pool.

Fingerprint proximity and motif frequency describe the training distribution;
they are neither molecular likelihoods nor chemical or synthetic quality scores.
"""

from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from latentmol.data import _split_digest
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.representation import GraphSpec


_ANILINE = Chem.MolFromSmarts("[NX3;H2][c]")


def motif_flags(molecule: Chem.Mol) -> dict[str, bool]:
    """Return two explicit ring features and an aniline substructure flag."""
    oxygen_rings = []
    for ring in molecule.GetRingInfo().AtomRings():
        if len(ring) == 3 and any(molecule.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in ring):
            oxygen_rings.append(ring)
    def two_aromatic_ring_atoms(ring) -> bool:
        nonoxygen = [i for i in ring if molecule.GetAtomWithIdx(i).GetAtomicNum() != 8]
        return len(nonoxygen) == 2 and all(
            molecule.GetAtomWithIdx(i).GetIsAromatic() for i in nonoxygen)

    return {"oxygen_three_ring": bool(oxygen_rings),
            "oxygen_three_ring_two_aromatic_atoms": any(
                two_aromatic_ring_atoms(ring) for ring in oxygen_rings),
            "aniline_substructure": molecule.HasSubstructMatch(_ANILINE)}


def top_neighbors(fingerprint, references: list, smiles: list[str], limit: int = 3) -> list[dict]:
    """Rank fingerprints without materializing a dense query-by-reference matrix."""
    similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, references)
    best = heapq.nlargest(limit, enumerate(similarities), key=lambda pair: (pair[1], -pair[0]))
    return [{"smiles": smiles[index], "tanimoto": float(value)} for index, value in best]


def run_support_audit(dataset_dir: Path, chemical_audit_dir: Path, run_root: Path) -> Path:
    manifest_path = dataset_dir / "manifest.json"
    dataset_path = dataset_dir / "dataset.npz"
    manifest = json.loads(manifest_path.read_text())
    if file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Training dataset hash differs from its manifest")
    spec = GraphSpec.from_dict(manifest["spec"])
    if spec.fingerprint() != manifest["spec_fingerprint"]:
        raise ValueError("Training graph schema differs from its manifest")
    parent = json.loads((chemical_audit_dir / "run.json").read_text())
    if parent.get("name") != "selected-molecule-chemical-audit" or parent.get("status") != "completed":
        raise ValueError("Chemical-audit parent is not completed")
    candidate_path = chemical_audit_dir / "molecules.json"
    candidate_rows = json.loads(candidate_path.read_text())
    if not candidate_rows:
        raise ValueError("Chemical-audit parent has no candidate rows")
    config = {"dataset_path": str(dataset_path), "dataset_sha256": manifest["dataset_sha256"],
              "split_sha256": manifest["split_sha256"],
              "graph_spec_fingerprint": manifest["spec_fingerprint"],
              "chemical_audit_run": parent["run_id"],
              "chemical_audit_source_sha256": parent["source_archive_sha256"],
              "candidate_rows_sha256": file_sha256(candidate_path),
              "rdkit_version": rdBase.rdkitVersion,
              "fingerprint": "Morgan radius 2, 2048 bits, no chirality",
              "reference": "unstandardized canonical SMILES in the 215000-molecule training split only"}
    with ExperimentRun(run_root, "stage3-training-support-preflight", config, Path.cwd()) as run:
        with np.load(dataset_path, allow_pickle=False) as data:
            split = data["split"]
            all_smiles = data["smiles"].tolist()
            if _split_digest(all_smiles, split) != manifest["split_sha256"]:
                raise ValueError("Training split assignments differ from the manifest")
            train_smiles = data["smiles"][split == 0].tolist()
            train_atoms = data["atom_types"][split == 0]
            atom_counts = np.count_nonzero(train_atoms, axis=1)
        if len(train_smiles) != manifest["counts"]["train"]:
            raise ValueError("Training reference count differs from the manifest")
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                               includeChirality=False)
        fingerprints = []
        motif_counts: Counter[str] = Counter()
        size13_counts: Counter[str] = Counter()
        size13_fingerprints = []
        size13_smiles = []
        for smiles, size in zip(train_smiles, atom_counts):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Previously accepted training SMILES cannot be parsed: {smiles}")
            fingerprint = generator.GetFingerprint(molecule)
            fingerprints.append(fingerprint)
            flags = motif_flags(molecule)
            for key, present in flags.items():
                motif_counts[key] += int(present)
                if size == 13:
                    size13_counts[key] += int(present)
            if size == 13:
                size13_fingerprints.append(fingerprint)
                size13_smiles.append(smiles)
        reference_set = set(train_smiles)
        rows = []
        for candidate in candidate_rows:
            comparisons = {}
            for role, smiles in (("submitted", candidate["input_smiles"]),
                                  ("native_standardized", candidate.get("native_standardized_smiles"))):
                if smiles is None:
                    comparisons[role] = None
                    continue
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    comparisons[role] = {"smiles": smiles, "parse_failure": True}
                    continue
                fingerprint = generator.GetFingerprint(molecule)
                canonical = Chem.MolToSmiles(molecule, isomericSmiles=False)
                comparisons[role] = {"smiles": smiles,
                                     "canonical_query_smiles": canonical,
                                     "parse_failure": False,
                                     "exact_training_identity": canonical in reference_set,
                                     "motifs": motif_flags(molecule),
                                     "nearest_train": top_neighbors(fingerprint, fingerprints,
                                                                    train_smiles),
                                     "nearest_train_13_heavy_atoms": top_neighbors(
                                         fingerprint, size13_fingerprints, size13_smiles)}
            rows.append({"input_smiles": candidate["input_smiles"],
                         "selected_by": candidate.get("selected_by"),
                         "comparisons": comparisons})
        write_json(run.path / "candidate_training_support.json", rows)
        summary = {"train_reference_molecules": len(train_smiles),
                   "train_13_heavy_atoms": len(size13_smiles),
                   "train_motif_counts": dict(motif_counts),
                   "train_13_heavy_atom_motif_counts": dict(size13_counts),
                   "candidate_submitted_identical_to_train": sum(
                       bool(row["comparisons"]["submitted"].get("exact_training_identity"))
                       for row in rows),
                   "candidate_native_identical_to_train": sum(
                       bool((row["comparisons"]["native_standardized"] or {}).get(
                           "exact_training_identity")) for row in rows),
                   "candidate_rows": len(rows),
                   "limitation": "Observed selected candidates are a diagnostic challenge set; fingerprint proximity and motif frequency do not validate a molecular prior or its gradient."}
        write_json(run.path / "summary.json", summary)
        run.log({"stage": "training_support", **summary})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--chemical-audit-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(run_support_audit(args.dataset_dir, args.chemical_audit_dir, args.run_root))


if __name__ == "__main__":
    main()
