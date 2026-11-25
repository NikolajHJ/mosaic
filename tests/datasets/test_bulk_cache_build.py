import gzip
from types import SimpleNamespace

import pytest
from rdkit import rdBase

from latentmol.datasets import bulk_cache_build
from latentmol.datasets.bulk_cache import MANIFEST_RESERVE_BYTES, SparseGraphDataset
from latentmol.datasets.bulk_cache_build import CanonicalMembership, GIB, LiveCacheBudget, build_sparse_corpus
from latentmol.datasets.bulk_finalize import VERSION
from latentmol.datasets.data import large_manifest_digest
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import GraphSpec


def test_quota_uses_pool_account_and_accounts_for_own_staged_bytes(tmp_path, monkeypatch):
    official = """
old storagepool 4):
       s234818|306121||      0 Byte|   unlimited||        0|unlimited
new storagepool 6):
       s234818|306121||   91.00 GiB|  100.00 GiB||   607872| 20000000
"""
    assert bulk_cache_build.parse_work3_headroom(official, "s234818") == int(8.98 * GIB)
    with pytest.raises(ValueError, match="requested account"):
        bulk_cache_build.parse_work3_headroom(official, "another")

    monkeypatch.setattr(bulk_cache_build, "disk_usage", lambda path: SimpleNamespace(free=100 * GIB))
    quotas = iter([9 * GIB, 17 * GIB // 2, 5 * GIB])
    budget = LiveCacheBudget(tmp_path / "unmade", reserve_bytes=4 * GIB, quota=lambda: next(quotas))
    assert budget.check(0) == 5 * GIB
    assert budget.check(GIB // 2) == 5 * GIB
    assert budget.check(GIB) == 2 * GIB
    assert budget.report["checks"] == 3


def _membership(root):
    root.mkdir()
    (root / "train").mkdir()
    records = {
        "val.tsv.gz": ["C\tmoses"],
        "test.tsv.gz": ["CC\tmoses"],
        "train/part-0000.tsv.gz": ["CCC\tmoses", "CCN\tpubchem", "CCO\tpubchem", "C=O\tpubchem"],
        "train/part-0001.tsv.gz": ["CCF\tpubchem", "CCCl\tpubchem", "CCBr\tpubchem", "CNC\tpubchem"],
    }
    files = {}
    for name, lines in records.items():
        path = root / name
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    spec = GraphSpec()
    manifest = dict(version=VERSION, spec=spec.to_dict(), spec_fingerprint=spec.fingerprint(),
                    rdkit_version=rdBase.rdkitVersion, counts=dict(train=8, val=1, test=1, total=10),
                    files=files, columns=["canonical_smiles", "source"], source_priority=["moses", "pubchem"],
                    source_counts={"moses": {"unique_train": 1}, "pubchem": {"unique_train": 7}})
    manifest["dataset_sha256"] = large_manifest_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return file_sha256(root / "manifest.json")


def test_build_authenticates_membership_and_spends_budget_on_reproducible_prefix(tmp_path):
    canonical = tmp_path / "canonical"
    digest = _membership(canonical)
    membership = CanonicalMembership(canonical, manifest_sha256=digest)
    selection = list(membership.rows(71))
    assert selection[:2] == [("val", "C", "moses"), ("test", "CC", "moses")]
    assert selection == list(membership.rows(71))
    assert selection != list(membership.rows(72))

    # Fixed metadata + split sentinels, exact two holdout and two training records.
    atoms = {"C": 1, "CC": 2, "C=O": 2}
    records = selection[:4]
    size = MANIFEST_RESERVE_BYTES + 24
    for _, smiles, _ in records:
        count = atoms.get(smiles, 3)
        size += 11 + 2 * count + 3 * (count - 1) + len(smiles)
    output = tmp_path / "packed"
    report = build_sparse_corpus(canonical, output, manifest_sha256=digest,
                                 max_output_bytes=size, reserve_bytes=0, seed=71)
    assert report["budget"]["stopped_for_budget"]
    assert report["counts"] == dict(train=2, val=1, test=1, total=4)
    assert list(SparseGraphDataset(output, "train").smiles) == [row[1] for row in selection[2:4]]
    assert list(SparseGraphDataset(output, "val").smiles) == ["C"]
    assert list(SparseGraphDataset(output, "test").smiles) == ["CC"]
    assert report["provenance"]["selection"]["available_counts"] == membership.counts
    with pytest.raises(ValueError, match="manifest hash"):
        CanonicalMembership(canonical, manifest_sha256="0" * 64)
    shard = canonical / "train" / "part-0000.tsv.gz"
    contents = shard.read_bytes()
    shard.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
    with pytest.raises(ValueError, match="artifact hash"):
        list(CanonicalMembership(canonical, manifest_sha256=digest).rows(71))
