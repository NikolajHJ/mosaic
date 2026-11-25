import pickle

import numpy as np
import pytest

from latentmol.datasets.bulk_cache import SparseGraphDataset
from latentmol.datasets.bulk_cache_build import build_sparse_corpus
from latentmol.datasets.bulk_view import BalancedSparseCorpus, build_balanced_view
from latentmol.datasets.ordering import verify_canonical_training_cache

from test_bulk_cache_build import _membership


def _sources(tmp_path, **limits):
    canonical, packed = tmp_path / "canonical", tmp_path / "packed"
    digest = _membership(canonical)
    parent = build_sparse_corpus(canonical, packed, manifest_sha256=digest, reserve_bytes=0, seed=71, **limits)
    return canonical, packed, dict(corpus_manifest_sha256=digest, expected_cache_sha256=parent["dataset_sha256"],
                                  train_count=6, add_per_holdout=1)


def test_balanced_view_preserves_base_holdouts_and_reads_exact_parent_graphs(tmp_path):
    canonical, packed, options = _sources(tmp_path)
    output = tmp_path / "balanced"
    report = build_balanced_view(canonical, packed, output, **options)
    assert report == build_balanced_view(canonical, packed, tmp_path / "repeat", **options)
    assert report["counts"] == {"train": 6, "val": 2, "test": 2, "total": 10}
    assert report["active_atom_counts"]["train"]["2"] == 1
    assert report["active_atom_counts"]["train"]["3"] == 5
    assert sum(meta["bytes"] for meta in report["files"].values()) == 32
    corpus = BalancedSparseCorpus(output, expected_sha256=report["dataset_sha256"])
    train, val, test = (corpus.split(role) for role in ("train", "val", "test"))
    assert val.smiles[0] == "C" and test.smiles[0] == "CC"
    assert len(set(train.smiles) | set(val.smiles) | set(test.smiles)) == 10
    assert val.smiles[1] != "CCC" and test.smiles[1] != "CCC"
    parent_train = SparseGraphDataset(packed, "train")
    by_smiles = {smiles: row for row, smiles in enumerate(parent_train.smiles)}
    for dataset, indexes in ((train, range(len(train))), (val, [1]), (test, [1])):
        for index in indexes:
            expected = parent_train[by_smiles[dataset.smiles[index]]]
            for name, value in expected.items():
                np.testing.assert_array_equal(dataset[index][name], value)
        assert verify_canonical_training_cache(dataset, range(len(dataset))) == len(dataset)
        reopened = pickle.loads(pickle.dumps(dataset))
        assert reopened.smiles[:] == dataset.smiles[:]


def test_view_refuses_partial_parent_storage_shortfall_and_corrupt_indexes(tmp_path):
    short = tmp_path / "short"
    short.mkdir()
    canonical, packed, options = _sources(short, max_output_bytes=33_000)
    with pytest.raises(ValueError, match="complete, authenticated source corpus"):
        build_balanced_view(canonical, packed, short / "view", **options)

    canonical, packed, options = _sources(tmp_path)
    output = tmp_path / "balanced"
    with pytest.raises(OSError, match="Insufficient storage"):
        build_balanced_view(canonical, packed, output, max_output_bytes=32, **options)
    assert not output.exists()
    build_balanced_view(canonical, packed, output, **options)
    path = output / "train.u32"
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="index hash differs"):
        BalancedSparseCorpus(output)
