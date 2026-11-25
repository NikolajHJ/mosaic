import csv
import gzip

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.datasets.chembl_audit import run_audit
from latentmol.datasets.combined import CombinedCorpus, VERSION, prepare_combined
from latentmol.datasets.data import GraphDataset, prepare_dataset
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import file_sha256


@pytest.mark.parametrize("workers", [1, 2])
def test_combined_compact_cache_retains_moses_and_reserves_whole_unseen_scaffolds(tmp_path, workers):
    source = tmp_path / "moses.smi"
    source.write_text("CCO\nCCN\nCCC\n", encoding="utf-8")
    moses = tmp_path / "moses"
    prepare_dataset(source, moses, train_count=1, val_count=1)
    moses_hash = file_sha256(moses / "manifest.json")
    chembl = tmp_path / "chemreps.txt.gz"
    molecules = ("CCO", "CCF", "C1CCCCC1", "CC1CCCCC1", "c1ccccc1", "Cc1ccccc1",
                 "C1CCCC1", "CC1CCCC1")
    with gzip.open(chembl, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("chembl_id", "canonical_smiles"))
        writer.writerows((f"CHEMBL{i}", smiles) for i, smiles in enumerate(molecules))
    audit = tmp_path / "audit"
    run_audit(chembl, moses, audit)
    config = dict(version=VERSION, moses_cache=str(moses), moses_manifest_sha256=moses_hash,
                  audit_summary=str(audit / "summary.json"), audit_summary_sha256=file_sha256(audit / "summary.json"),
                  chembl_holdout_counts={"val": 2, "test": 2}, split_seed=453704)
    output = tmp_path / "combined"
    report = prepare_combined(config, output, workers=workers)
    corpus = CombinedCorpus(output, expected_sha256=report["dataset_sha256"])
    assert report["counts"] == {"train": 4, "val": 3, "test": 3, "total": 10}
    assert report["sources"]["chembl"]["holdout_scaffolds"] == {"val": 1, "test": 1}

    role = {}
    for split in ("train", "val", "test"):
        moses_rows = list(GraphDataset(moses, split).smiles)
        chembl_rows = corpus.split(split, source="chembl")
        both = corpus.split(split)
        assert list(corpus.split(split, source="moses").smiles) == moses_rows
        assert list(both.smiles) == moses_rows + list(chembl_rows.smiles)
        assert verify_canonical_training_cache(both, list(range(len(both)))) == len(both)
        for i, smiles in enumerate(chembl_rows.smiles):
            assert smiles not in role
            role[smiles] = split
            graph = encode_smiles(smiles, corpus.spec)
            sample = chembl_rows[i]
            for key, expected in (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types")):
                torch.testing.assert_close(sample[key], torch.tensor(graph[expected], dtype=torch.long))
    assert role["CCF"] == "train"
    for a, b in (("C1CCCCC1", "CC1CCCCC1"), ("c1ccccc1", "Cc1ccccc1"), ("C1CCCC1", "CC1CCCC1")):
        assert role[a] == role[b]
    assert file_sha256(moses / "manifest.json") == moses_hash
    with pytest.raises(FileExistsError):
        prepare_combined(config, output)

    with (output / "edges.bin").open("ab") as stream:
        stream.write(b"\0")
    with pytest.raises(ValueError, match="Combined cache data hash differs"):
        CombinedCorpus(output)
