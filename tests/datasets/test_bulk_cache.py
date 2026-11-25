import itertools
import pickle

import numpy as np
import pytest

from latentmol.chemistry import decode_graph, encode_smiles
from latentmol.datasets.bulk_cache import (
    MANIFEST_RESERVE_BYTES, SparseGraphDataset, validate_sparse_cache, write_sparse_cache,
)
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.representation import GraphSpec


SPEC = GraphSpec(max_nodes=12)
ROWS = [("val", "c1ccccc1", "moses"), ("test", "C[N+](C)(C)C", "chembl37"),
        ("train", "C", "pubchem"), ("train", "CCO", "moses"), ("train", "C#N", "pubchem")]


def _write(output, rows=ROWS, **kwargs):
    options = dict(spec=SPEC, source_dataset_sha256="a" * 64, max_bytes=1024**2,
                   expected_holdout_counts={"val": 1, "test": 1})
    options.update(kwargs)
    return write_sparse_cache(iter(rows), output, **options)


def test_packed_cache_round_trips_and_parallel_bytes_match(tmp_path):
    cache = tmp_path / "serial"
    manifest = _write(cache)
    parallel = _write(tmp_path / "parallel", workers=2)
    assert manifest == parallel
    assert manifest["counts"] == {"train": 3, "val": 1, "test": 1, "total": 5}
    assert manifest["source_counts"]["train"] == {"pubchem": 2, "moses": 1}
    assert manifest["budget"]["file_bytes"] == sum(item["bytes"] for item in manifest["files"].values())
    receipt = validate_sparse_cache(cache, expected_sha256=manifest["dataset_sha256"])
    for split in ("train", "val", "test"):
        dataset = SparseGraphDataset(cache, split, validated_cache=receipt)
        expected = [smiles for role, smiles, _ in ROWS if role == split]
        assert dataset.smiles[:] == expected
        assert dataset.smiles[-1] == expected[-1]
        for row, smiles in enumerate(expected):
            graph = encode_smiles(smiles, SPEC)
            for key, graph_key in (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types")):
                np.testing.assert_array_equal(dataset[row][key].numpy(), graph[graph_key])
        assert verify_canonical_training_cache(dataset, range(len(dataset))) == len(expected)
        reopened = pickle.loads(pickle.dumps(dataset))
        assert reopened.smiles[:] == expected


def test_storage_and_time_stop_preserve_holdouts_and_exact_prefix(tmp_path):
    cap = MANIFEST_RESERVE_BYTES + 24 + sum(
        11 + 2 * nodes + 3 * edges + len(smiles)
        for (_, smiles, _), nodes, edges in zip(ROWS, (6, 5, 1), (6, 4, 0))
    )
    calls = []

    def check_storage(staged):
        calls.append(staged)
        return cap

    output = tmp_path / "budget"
    manifest = _write(output, check_storage=check_storage)
    assert manifest["counts"] == {"train": 1, "val": 1, "test": 1, "total": 3}
    assert manifest["budget"]["effective_bytes"] == cap
    assert manifest["budget"]["stopped_for_budget"]
    assert not manifest["budget"]["stopped_for_time"]
    assert calls[0] == 0 and calls[-1] == cap - MANIFEST_RESERVE_BYTES
    assert sum(item.stat().st_size for item in output.iterdir()) <= cap
    assert SparseGraphDataset(output, "train").smiles[:] == ["C"]

    clock = itertools.count()
    timed = _write(tmp_path / "deadline", should_stop=lambda: next(clock) > 0)
    assert timed["counts"] == manifest["counts"]
    assert timed["budget"]["stopped_for_time"]
    assert not timed["budget"]["stopped_for_budget"]


def test_preserves_pubchem_identity_when_raw_recanonicalization_is_not_idempotent(tmp_path):
    # The saved corpus spelling passed ingest. The raw SMILES encoder's
    # additional canonicalization changed it and stopped DTU job 29903998.
    smiles = "CNC(=O)c1ccccc1N1C2=NC1=C(Cl)C(Cl)=N2"
    cache = tmp_path / "pubchem"
    _write(cache, rows=[*ROWS[:2], ("train", smiles, "pubchem")], spec=GraphSpec())
    data = SparseGraphDataset(cache, "train")
    assert data.smiles[0] == smiles
    graph = data[0]
    assert decode_graph(graph["atoms"], graph["charges"], graph["bonds"], data.spec) == smiles
    assert verify_canonical_training_cache(data, [0]) == 1


def test_noncanonical_and_corrupt_inputs_fail_closed(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="not canonical"):
        _write(output, rows=[*ROWS, ("train", "OCC", "pubchem")])
    assert not output.exists()
    with pytest.raises(ValueError, match="must precede"):
        _write(output, rows=reversed(ROWS))
    assert not output.exists()
    with pytest.raises(OSError, match="cannot fit all"):
        _write(output, max_bytes=MANIFEST_RESERVE_BYTES + 40)
    assert not output.exists()
    _write(output)
    with pytest.raises(FileExistsError):
        _write(output)
    with (output / "train.graphs.bin").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="differs from its manifest"):
        validate_sparse_cache(output)
