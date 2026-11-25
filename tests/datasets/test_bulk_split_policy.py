from itertools import product

import pytest

from latentmol.datasets.bulk_split_policy import capped_waterfill, select_pubchem_holdouts, source_aligned_candidates


def test_capped_waterfill_matches_brute_force_integer_optimum():
    capacities = {1: 1, 2: 4, 3: 2, 4: 5, 5: 0}
    choices = list(product(*(range(count + 1) for count in capacities.values())))
    for target in range(sum(capacities.values()) + 2):
        quotas = capped_waterfill(capacities, target)
        feasible = [counts for counts in choices if sum(counts) == min(target, sum(capacities.values()))]
        assert tuple(quotas.values()) in feasible
        assert sum(q*q for q in quotas.values()) == min(sum(q*q for q in counts) for counts in feasible)
    ties = {size: 3 for size in range(1, 8)}
    assert capped_waterfill(ties, 3, seed=99) == capped_waterfill(dict(reversed(ties.items())), 3, seed=99)


def test_holdouts_only_take_aligned_pubchem_rows_with_training_slack():
    identities = ["N", "CC", "CO", "CN", "NN", "NO", "CF", "CCO"]
    source = ["pubchem", "moses", "pubchem", "pubchem", "pubchem", "pubchem", "pubchem", "pubchem"]
    atoms = [1, 2, 2, 2, 2, 2, 2, 3]
    rows = [("val", "CCCl", "moses"), ("test", "CCBr", "chembl37"),
            *[("train", text, name) for text, name in zip(identities, source)]]
    candidates = source_aligned_candidates(rows, identities, atoms)
    selection = select_pubchem_holdouts(candidates, capacities={1: 1, 2: 6, 3: 1},
                                       train_quotas={1: 1, 2: 2, 3: 1}, count_per_split=2)
    assert selection == {"val": [2, 4], "test": [3, 5]}
    with pytest.raises(ValueError, match="identity at row 1"):
        list(source_aligned_candidates(rows, ["N", "wrong"], atoms[:2]))
    with pytest.raises(ValueError, match="ended before"):
        list(source_aligned_candidates(rows[:3], identities, atoms))
    with pytest.raises(ValueError, match="outside the planned"):
        select_pubchem_holdouts((), capacities={1: 2}, train_quotas={1: 1}, count_per_split=1)
    with pytest.raises(ValueError, match="Only 1 PubChem"):
        select_pubchem_holdouts([(0, 1, "pubchem")], capacities={1: 2}, train_quotas={1: 0}, count_per_split=1)
