import csv
import gzip
import json

import pytest

from latentmol.datasets.chembl_audit import run_audit
from latentmol.datasets.data import GraphDataset, prepare_dataset
from latentmol.experiments import file_sha256


def test_audit_deduplicates_every_split_and_reports_novel_scaffolds(tmp_path):
    moses_source = tmp_path / "moses.smi"
    moses_source.write_text("CCO\nc1ccccc1\nc1ccncc1\n", encoding="utf-8")
    moses = tmp_path / "moses"
    prepare_dataset(moses_source, moses, train_count=1, val_count=1)
    benzene_in_train = "c1ccccc1" in GraphDataset(moses, "train").smiles
    chembl = tmp_path / "chemreps.txt.gz"
    with gzip.open(chembl, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("chembl_id", "canonical_smiles", "standard_inchi", "standard_inchi_key"))
        molecules = ("CCO", "OCC", "c1ccccc1", "c1ccncc1", "Cc1ccccc1", "c1ccoc1", "CCC", "C.C", "[Na+]")
        writer.writerows((f"CHEMBL{i}", molecule, "", "") for i, molecule in enumerate(molecules))
    report_dir = tmp_path / "audit"
    report = run_audit(chembl, moses, report_dir, workers=2, source_sha256=file_sha256(chembl))
    assert report["counts"] == {
        "source_rows": 9, "rejected_rows": 2, "accepted_rows": 7,
        "duplicate_rows": 1, "unique_molecules": 6,
        "overlap_moses": {"train": 1, "val": 1, "test": 1}, "new_molecules": 3,
    }
    assert report["source"]["complete"]
    assert report["rejections"] == {"disconnected": 1, "unsupported_element": 1}
    assert report["scaffolds"]["new_vs_moses_all"] == 1
    assert report["scaffolds"]["new_molecule_membership"] == {
        "acyclic": 1, "seen_train": int(benzene_in_train),
        "seen_holdouts_only": int(not benzene_in_train), "unseen_moses": 1,
    }
    eligible = report_dir / report["eligible"]["path"]
    assert file_sha256(eligible) == report["eligible"]["sha256"]
    with gzip.open(eligible, "rt", encoding="utf-8", newline="") as stream:
        records = {row["smiles"]: row for row in csv.DictReader(stream)}
    assert set(records) == {"Cc1ccccc1", "c1ccoc1", "CCC"}
    assert records["Cc1ccccc1"]["scaffold_in_moses_any"] == "1"
    assert records["c1ccoc1"]["scaffold_in_moses_any"] == "0"
    assert records["CCC"]["scaffold"] == ""
    assert json.loads((report_dir / "summary.json").read_text())["eligible"]["count"] == 3
    with pytest.raises(FileExistsError):
        run_audit(chembl, moses, report_dir)
