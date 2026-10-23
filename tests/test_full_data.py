"""Full-corpus splitting retains earlier holdouts and exact requested counts."""

import numpy as np
import pytest

from latentmol.data import GraphDataset, assign_splits, prepare_dataset
from latentmol.data_audit import audit_cache
from latentmol.sampling import subset_indices


def test_expanded_cache_preserves_prior_roles_and_exact_counts(tmp_path):
    source = tmp_path / "source.smi"
    source.write_text("\n".join("C" * size for size in range(1, 21)))
    old_path, new_path = tmp_path / "pilot", tmp_path / "full"
    old = prepare_dataset(source, old_path, limit=10, seed=17)
    full = prepare_dataset(source, new_path, train_count=15, val_count=3,
                           reserve_cache=old_path, seed=17)
    assert [full["counts"][key] for key in ("train", "val", "test")] == [15, 3, 2]
    assert full["provenance"]["reserved_cache"]["dataset_sha256"] == old["dataset_sha256"]
    roles = []
    for split in ("train", "val", "test"):
        previous, current = GraphDataset(old_path, split), GraphDataset(new_path, split)
        assert set(previous.smiles) <= set(current.smiles)
        roles.append(set(current.smiles))
        audit = full["audit"]["by_split"][split]
        assert sum(audit["heavy_atom_histogram"]) == len(current)
    assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
    assert len(set.union(*roles)) == 20
    audit = audit_cache(new_path, old_path, {"train": 15, "val": 3, "test": 2})
    assert audit["passed"]
    assert audit["pairwise_intersections"] == {"train:val": 0, "train:test": 0, "val:test": 0}
    assert audit["old_split_rows_new_split_columns"] == {
        "train": {"train": 8, "val": 0, "test": 0},
        "val": {"train": 0, "val": 1, "test": 0},
        "test": {"train": 0, "val": 0, "test": 1}}
    with pytest.raises(ValueError, match="Unexpected split counts"):
        audit_cache(new_path, old_path, {"train": 16, "val": 2, "test": 2})


@pytest.mark.parametrize("train,val,reserved", [
    (9, 2, {}), (None, 1, {}), (1, None, {}), (0, 1, {}),
    (7, 1, {"absent": 0}), (7, 1, {"0": 5}),
    (7, 1, {"0": 1, "1": 1}),
])
def test_invalid_explicit_or_reserved_split_is_rejected(train, val, reserved):
    with pytest.raises(ValueError):
        assign_splits([str(i) for i in range(10)], np.random.default_rng(17), train, val, reserved)


def test_complete_evaluation_can_have_unbiased_reproducible_prefixes():
    assert subset_indices(100, None, 17) == list(range(100))
    order = subset_indices(100, 100, 17, shuffle_full=True)
    assert order == subset_indices(100, None, 17, shuffle_full=True)
    assert sorted(order) == list(range(100))
    assert order[:10] != list(range(10))
