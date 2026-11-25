import csv
import gzip

import pytest

from latentmol.datasets import bulk_finalize
from latentmol.datasets.bulk_finalize import VERSION, finalize_corpus
from latentmol.datasets.bulk_ingest import ingest
from latentmol.datasets.chembl_audit import run_audit
from latentmol.datasets.combined import CombinedCorpus, VERSION as COMBINED_VERSION, prepare_combined
from latentmol.datasets.data import large_manifest_digest, prepare_dataset
from latentmol.experiments import file_sha256


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [tuple(line.rstrip("\n").split("\t")) for line in stream]


def test_bulk_corpus_deduplicates_globally_and_preserves_holdouts(tmp_path, monkeypatch):
    moses_input = tmp_path / "moses.smi"
    moses_input.write_text("CCO\nCCN\nCCC\n", encoding="utf-8")
    moses = tmp_path / "moses"
    prepare_dataset(moses_input, moses, train_count=1, val_count=1)
    chembl_input = tmp_path / "chemreps.txt.gz"
    with gzip.open(chembl_input, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("chembl_id", "canonical_smiles"))
        writer.writerows((f"CHEMBL{i}", smiles) for i, smiles in enumerate(
            ("CCF", "C1CCCCC1", "CC1CCCCC1", "c1ccccc1", "Cc1ccccc1", "C1CCCC1", "CC1CCCC1")))
    audit = tmp_path / "audit"
    run_audit(chembl_input, moses, audit)
    baseline_path = tmp_path / "baseline"
    baseline = prepare_combined(
        dict(version=COMBINED_VERSION, moses_cache=str(moses),
             moses_manifest_sha256=file_sha256(moses / "manifest.json"), audit_summary=str(audit / "summary.json"),
             audit_summary_sha256=file_sha256(audit / "summary.json"),
             chembl_holdout_counts={"val": 2, "test": 2}, split_seed=453704),
        baseline_path,
    )
    corpus = CombinedCorpus(baseline_path)
    pubchem_rows = [
        corpus.split("val", source="moses").smiles[0], corpus.split("test", source="chembl").smiles[0],
        "FC1CCCCC1", "Fc1ccccc1", "FC1CCCC1", "CCCl", "CCBr", "CCCl",
    ]
    sources = []
    for source, shards in (("pubchem", [pubchem_rows, ["CCCl", "CCF"]]),
                           ("external", [["CCBr", "CCl"]])):
        root = tmp_path / source
        root.mkdir()
        for index, rows in enumerate(shards):
            raw = tmp_path / f"{source}-{index}.smi"
            raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
            ingest(raw, root / f"shard-{index}", source=source, input_format="smiles")
        sources.append(dict(name=source, shard_root=str(root)))
    config = dict(version=VERSION, baseline=dict(path=str(baseline_path), dataset_sha256=baseline["dataset_sha256"]),
                  sources=sources, minimum_train_count=50_000_000, hash_buckets=4)
    output = tmp_path / "out"
    manifest = finalize_corpus(config, output, scratch_root=tmp_path)
    assert manifest["counts"] == {"train": 8, "val": 3, "test": 3, "total": 14}
    assert manifest["target_reached"] is False
    assert manifest["source_counts"]["pubchem"] == dict(
        normalized=10, excluded_exact_holdout=2, excluded_reserved_scaffold=2, duplicates=3, unique_train=3)
    assert manifest["source_counts"]["external"]["duplicates"] == 1
    assert manifest["dataset_sha256"] == large_manifest_digest(manifest)
    for relative, metadata in manifest["files"].items():
        assert metadata["sha256"] == file_sha256(output / relative)
    for split in ("val", "test"):
        assert [smiles for smiles, _ in _rows(output / f"{split}.tsv.gz")] == list(corpus.split(split).smiles)
    train_rows = [row for path in (output / "train").glob("*.tsv.gz") for row in _rows(path)]
    train = dict(train_rows)
    assert len(train) == len(train_rows) == 8
    assert (train["CCF"], train["CCBr"], train["CCl"]) == ("chembl37", "pubchem", "external")
    with pytest.raises(ValueError, match="publication cap"):
        finalize_corpus(config, tmp_path / "over-cap", scratch_root=tmp_path, max_output_bytes=1)
    assert not (tmp_path / "over-cap").exists()

    def interrupted_copy(source, destination, metadata):
        destination.write_bytes(b"partial")
        raise OSError("test publication interruption")

    monkeypatch.setattr(bulk_finalize, "_copy_checked", interrupted_copy)
    with pytest.raises(OSError, match="publication interruption"):
        finalize_corpus(config, tmp_path / "interrupted", scratch_root=tmp_path)
    assert not (tmp_path / "interrupted").exists()
