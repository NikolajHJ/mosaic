"""Geometry failures do not change which matched starts enter the main result."""

from pytest import approx

from latentmol.protenix_occupancy.raw_occupancy_summary import summarize


def _start(draw, guided, reversed_loss):
    def scores(values):
        return {seed: {"loss": loss} for seed, loss in zip(("294", "1201"), values)}

    return dict(draw=draw, initial={"atom_count": 25}, initial_fresh_by_seed=scores((2.0, 2.0)),
                arms={"gradient": {
                    "final": {"atom_count": 26},
                    "updates": [{"status": "accepted", "baseline": {"smiles": "CCO"},
                                 "trials": [{"atom_delta": 1, "smiles": "CCCO"}]}],
                    "fresh_by_seed": scores(guided),
                }, "reversed_gradient": {
                    "final": {"atom_count": 25}, "updates": [],
                    "fresh_by_seed": scores(reversed_loss),
                }})


def test_paired_summary_counts_only_complete_geometry_pairs():
    starts = [_start(0, (1.0, 2.0), (2.0, 2.0)),
              _start(1, (3.0, 3.0), (2.0, 2.0)),
              _start(2, (None, 1.0), (2.0, 2.0))]
    result = summarize(starts, draws=100)
    assert result == summarize(starts, draws=100)
    assert result["paired_starts"] == 2
    assert result["mean_paired_difference"] == approx(.25)
    assert result["guided_wins"] == result["reversed_wins"] == 1
    assert result["paired_difference_bootstrap_95"] == [-.5, 1.0]
    assert result["per_seed"]["1201"]["paired_starts"] == 3
    assert result["by_start"][2]["mean_paired_difference"] is None
    assert result["arms"]["gradient"]["starts_with_any_count_change"] == 3
    assert "mean_fresh_loss" not in result["arms"]["gradient"]
    assert "molecule_changing_updates" not in result["by_start"][0]["arms"]["gradient"]


def test_random_control_reports_complete_fresh_changes_and_chemistry_edits():
    starts = [_start(0, (1.0, 2.0), (2.0, 2.0)),
              _start(1, (3.0, 3.0), (2.0, 2.0)),
              _start(2, (None, 1.0), (2.0, 2.0))]
    for start in starts:
        start["arms"]["random_search"] = start["arms"].pop("reversed_gradient")
    starts[2]["initial_fresh_by_seed"]["294"]["loss"] = None
    first = starts[0]["arms"]["random_search"]
    first["final"]["atom_count"] = 24
    first["updates"] = [
        {"status": "accepted", "baseline": {"smiles": "CCO"},
         "trials": [{"atom_delta": -1, "smiles": "CO"}]},
        {"status": "accepted", "baseline": {"smiles": "CO"},
         "trials": [{"atom_delta": 0, "smiles": "CO"}]},
    ]
    starts[1]["arms"]["random_search"]["updates"] = [
        {"status": "accepted", "baseline": {"smiles": "CCO"},
         "trials": [{"atom_delta": 0, "smiles": "CCN"}]},
    ]

    result = summarize(starts, comparator="random_search", draws=100)

    assert result["paired_starts"] == 2
    assert result["mean_paired_difference"] == approx(0.25)
    assert result["guided_wins"] == result["random_search_wins"] == 1
    assert "reversed_wins" not in result
    assert result["per_seed"]["1201"]["random_search_wins"] == 1
    assert result["by_start"][0]["fresh_by_seed"]["1201"]["random_search"] == 2.0
    guided = result["arms"]["gradient"]
    assert guided["mean_fresh_loss"] == approx(2.25)
    assert guided["mean_fresh_change_from_initial"] == approx(0.25)
    assert guided["fresh_complete_starts"] == 2
    assert guided["atom_adding_updates"] == guided["molecule_changing_updates"] == 3
    random_arm = result["arms"]["random_search"]
    assert random_arm["fresh_complete_starts"] == 3
    assert random_arm["fresh_change_complete_starts"] == 2
    assert random_arm["mean_fresh_loss"] == approx(2.0)
    assert random_arm["mean_fresh_change_from_initial"] == approx(0.0)
    assert random_arm["starts_with_removal"] == random_arm["atom_removing_updates"] == 1
    assert random_arm["accepted_updates"] == 3
    assert random_arm["molecule_changing_updates"] == 2
