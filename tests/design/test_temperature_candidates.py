"""Validate the time-based candidate cap and matched fresh-score reporting."""

import pytest

from latentmol.design.temperature_candidates import select_valid_candidates
from latentmol.design.temperature_fresh import check_edit_limits, score_candidates, summarize_arms


def observations(smiles):
    return [dict(step=step, smiles=value, valid=value is not None)
            for step, value in enumerate(smiles)]


def test_preserves_early_and_final_valid_edit_with_deterministic_time_coverage():
    panel = select_valid_candidates(observations([
        "C", None, "CC", "CCO", "CCN", "CCC", "CCCl", None, "CCF", "CCN"
    ]), edit_limit=4)
    assert [(row["step"], row["role"]) for row in panel["candidates"]] == [
        (0, "initial"), (2, "first_distinct_valid"), (3, "valid_intermediate_1"),
        (6, "valid_intermediate_2"), (9, "endpoint"),
    ]
    assert panel["distinct_valid_molecules"] == 7
    assert panel["selected_valid_edits"] == 4
    assert panel["valid_edit_coverage"] == pytest.approx(4 / 6)


def test_invalid_start_and_repeated_final_identity_do_not_add_scoring_rows():
    panel = select_valid_candidates(observations([None, "CC", None, "CCO", "CC"]), edit_limit=4)
    assert [(row["step"], row["smiles"]) for row in panel["candidates"]] == [(1, "CC"), (3, "CCO")]
    assert panel["distinct_valid_edits"] == 2
    assert panel["valid_edit_coverage"] == 1
    assert select_valid_candidates(observations([None, None]), edit_limit=4)["candidates"] == []


def test_fresh_arms_share_identical_molecule_score_and_keep_baseline_scope():
    panels = {
        "A": select_valid_candidates(observations(["CC", "CCO", "CCC"]), edit_limit=2),
        "B": select_valid_candidates(observations([None, "CCO"]), edit_limit=2),
    }
    candidates = [dict(row, start="same", arm=arm) for arm, panel in panels.items()
                  for row in panel["candidates"]]
    check_edit_limits(candidates, 2)
    with pytest.raises(ValueError, match="budget"):
        check_edit_limits(candidates, 1)
    with pytest.raises(ValueError, match="budget"):
        check_edit_limits([*candidates, candidates[-1]], 2)

    def score(molecule, index):
        return dict(status="scored", loss=0.4 - 0.1 * index)

    rows, unique = score_candidates(candidates, score)
    assert unique == 3
    first, second = summarize_arms(rows)
    assert first["baseline_status"] == "scored"
    assert first["scored_edits"] == first["improving_edits"] == 2
    assert first["best_delta_from_valid_start"] == pytest.approx(-0.2)
    assert second["baseline_status"] == "no_valid_initial"
    assert second["improving_edits"] is None
    assert second["best_delta_from_valid_start"] is None
