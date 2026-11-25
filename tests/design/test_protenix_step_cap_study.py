from types import SimpleNamespace

import pytest

from latentmol.design.protenix_restart_study import (
    matched_parent_shortlists,
    paired_budget_readout,
)


def test_step_cap_shortlists_match_within_parents_and_reserve_across_parents():
    ranked = {
        (0, "low", 0): ["cached", "shared", "low-extra"],
        (0, "low", 1): ["low-first"],
        (0, "high", 0): ["shared"],
        (0, "high", 1): ["high-first"],
        (1, "low", 0): ["shared", "high-first", "low-last"],
        (1, "low", 1): ["low-first", "low-next"],
        (1, "high", 0): ["shared", "low-first", "high-last"],
        (1, "high", 1): ["high-first", "high-next"],
    }
    branches = []
    reserves = {}
    for (parent, arm, repeat), smiles in ranked.items():
        name = f"{parent}-{arm}-{repeat}"
        branches.append(SimpleNamespace(name=name, parent_index=parent, arm=arm))
        reserves[name] = {"mixed": [SimpleNamespace(smiles=value) for value in smiles]}
    arm_caches = {"low": {"cached": {"loss": 2.4}}, "high": {}}

    selected = matched_parent_shortlists(branches, reserves, arm_caches, limit=3)
    identities = {name: [row.smiles for row in lists["mixed"]] for name, lists in selected.items()}
    assert identities == {
        "0-low-0": ["shared"],
        "0-low-1": ["low-first"],
        "0-high-0": ["shared"],
        "0-high-1": ["high-first"],
        "1-low-0": ["high-first", "low-last"],
        "1-low-1": ["low-next"],
        "1-high-0": ["low-first", "high-last"],
        "1-high-1": ["high-next"],
    }
    assert arm_caches == {"low": {"cached": {"loss": 2.4}}, "high": {}}

    readouts = [
        dict(parent_index=branch.parent_index, arm=branch.arm, summary={"updates": 500},
             candidates={"mixed": [{"smiles": smiles, "loss": None if smiles == "high-first" else 2.1}
                                   for smiles in identities[branch.name]]})
        for branch in branches
    ]
    result = paired_budget_readout(readouts, updates_per_parent=1000, attempts_per_parent=3)
    assert result["fresh_attempt_limit_per_parent"] == 3
    assert result["by_parent"] == {
        0: {
            "low": {"updates": 1000, "fresh_attempts": 2, "fresh_scored": 2},
            "high": {"updates": 1000, "fresh_attempts": 2, "fresh_scored": 1},
        },
        1: {
            "low": {"updates": 1000, "fresh_attempts": 3, "fresh_scored": 2},
            "high": {"updates": 1000, "fresh_attempts": 3, "fresh_scored": 3},
        },
    }
    assert result["by_arm"] == {
        "low": {"updates": 2000, "fresh_attempts": 5, "fresh_scored": 4},
        "high": {"updates": 2000, "fresh_attempts": 5, "fresh_scored": 4},
    }


@pytest.mark.parametrize(
    "updates, attempts",
    [
        ([500, 499, 500, 501], [2, 2, 2, 2]),
        ([500, 500, 500, 500], [2, 1, 1, 2]),
        ([500, 500, 500, 500], [3, 3, 3, 3]),
    ],
    ids=["updates-cancel-globally", "attempts-cancel-globally", "matched-but-over-limit"],
)
def test_step_cap_readout_rejects_invalid_realized_budget(updates, attempts):
    pairs = [(parent, arm) for parent in range(2) for arm in ("low", "high")]
    readouts = [
        dict(parent_index=parent, arm=arm, summary={"updates": update_count},
             candidates={"mixed": [{"loss": 2.1}] * attempt_count})
        for (parent, arm), update_count, attempt_count in zip(pairs, updates, attempts, strict=True)
    ]

    with pytest.raises(ValueError, match="inconsistent realized budgets"):
        paired_budget_readout(readouts, updates_per_parent=500, attempts_per_parent=2)
