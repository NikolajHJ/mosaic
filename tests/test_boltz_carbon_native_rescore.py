"""Native selection must retain the randomized hard control and fixed cohort."""

import pytest

from latentmol.boltz.carbon_native_rescore import select_carbon_anchor


def _score(value):
    return {"fixed_context_discrete_affinity_probability_binary": value}


def _attempt(arm, start, end, score):
    return {"arm": arm, "start_hard": {"smiles": start},
            "hard": {"smiles": end}, "hard_projection_score": _score(score)}


def test_selection_includes_changed_random_control():
    chain = "CCCCCCCCCCCCC"
    baseline = {"hard": {"smiles": chain}, "hard_projection_score": _score(0.08)}
    attempts = [_attempt("target", chain, "CCCCCCCCCCC(C)C", 0.09),
                _attempt("random_matched_displacement", chain, "CCCCCCCCCC(C)CC", 0.08)]
    selection = select_carbon_anchor(baseline, attempts)
    assert len(selection["unique_smiles"]) == 3
    assert selection["random_control_smiles"] in selection["unique_smiles"]


def test_selection_rejects_incomplete_cohort():
    chain = "CCCCCCCCCCCCC"
    baseline = {"hard": {"smiles": chain}, "hard_projection_score": _score(0.08)}
    with pytest.raises(ValueError, match="one target and one random"):
        select_carbon_anchor(baseline, [_attempt("target", chain, "CCCCCCCCCCC(C)C", 0.09)])
