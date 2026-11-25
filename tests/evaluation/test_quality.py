"""Graded chemistry/repair accounting regressions. Run on DTU HPC only."""

import math

import numpy as np
import pytest

from latentmol.chemistry import encode_smiles
from latentmol.quality import (GradedEvaluator, QualityScoreConfig, combine_quality, diagnose_graph,
                               graph_edit_cost, measure_repairs, summarize_quality)
from latentmol.repair_search import RepairSearchConfig, search_repair
from latentmol.representation import GraphSpec


def graph(smiles, spec):
    encoded = encode_smiles(smiles, spec)
    return tuple(encoded[key].copy() for key in ("atom_types", "charges", "bond_types"))


def budget(count=2):
    return {"projector": "fixture-only", "checkpoint_sha256": "a" * 64,
            "attempts": count, "transitions_per_attempt": 8, "seed": 17,
            "policy": {"temperature": 1.0, "initialization": "raw_graph"}}


def invalid_co(spec):
    raw = graph("CO", spec)
    raw[2][0, 1] = raw[2][1, 0] = spec.bond_names.index("TRIPLE")
    return raw


def legacy(**overrides):
    return QualityScoreConfig.from_dict({"validity": .4, "local_consistency": .4,
                                        "connectedness": .2, **overrides})


@pytest.mark.parametrize("smiles", ["C", "CCO", "C[N+](C)(C)C", "c1ccccc1", "CS(=O)(=O)C"])
def test_valid_implicit_h_charge_and_aromatic_graphs(smiles):
    spec = GraphSpec(max_nodes=10)
    record = diagnose_graph(*graph(smiles, spec), spec)
    assert record["valid"] and record["local_consistency"] == 1
    assert record["connectedness"] == 1
    assert summarize_quality([record], legacy())["mixture"]["score"] == 100
    if smiles == "c1ccccc1":
        assert record["valence_checked_atoms"] == 0
        assert record["overvalence_excess"] is None


def test_invalid_graphs_have_severity_and_simultaneous_fragment_diagnostics():
    spec = GraphSpec(max_nodes=8)
    near = invalid_co(spec)
    near_record = diagnose_graph(*near, spec)
    broken = graph("CCCCCC", spec)
    broken[2][:6, :6] = 1
    np.fill_diagonal(broken[2], 0)  # Each of six carbons now has five neighbors.
    bad_record = diagnose_graph(*broken, spec)
    assert not near_record["valid"] and not bad_record["valid"]
    assert 0 < bad_record["local_consistency"] < near_record["local_consistency"] < 1
    assert bad_record["overvalence_excess"] == 6
    assert summarize_quality([near_record], legacy())["mixture"]["score"] > summarize_quality([bad_record], legacy())["mixture"]["score"]
    near[0][2] = spec.atomic_numbers.index(6)  # Disconnected carbon plus invalid O.
    fragmented = diagnose_graph(*near, spec)
    assert fragmented["component_count"] == 2 and fragmented["problem_burden"] > 0
    assert fragmented["connectedness"] == 0.5


def test_empty_and_unsupported_are_not_perfect_or_dropped():
    spec = GraphSpec(max_nodes=3)
    empty = (np.zeros(3, dtype=int), np.full(3, 2), np.zeros((3, 3), dtype=int))
    record = diagnose_graph(*empty, spec)
    assert summarize_quality([record], legacy())["mixture"]["score"] == 0
    malformed = diagnose_graph(empty[0].astype(float), empty[1], empty[2], spec)
    summary = summarize_quality([record, malformed])
    assert summary["attempts"] == 2 and summary["schema_unsupported"] == 1
    assert summary["mixture"]["components"]["validity"] == 0
    assert summary["mixture"]["score"] is None
    assert summarize_quality([])["mixture"]["score"] is None


def test_rdkit_unsupported_element_does_not_abort_diagnostics():
    spec = GraphSpec(max_nodes=1, atomic_numbers=(0, 200))
    record = diagnose_graph(np.array([1]), np.array([2]), np.array([[0]]), spec)
    assert not record["valid"] and record["local_consistency"] is None


def test_padding_law_violation_does_not_change_active_chemistry_or_collapse_identity():
    spec = GraphSpec(max_nodes=4)
    original = graph("C", spec)
    clean = diagnose_graph(*original, spec)
    original[1][2] = 0
    original[2][0, 2] = original[2][2, 0] = 3
    record = diagnose_graph(*original, spec)
    assert record["valid"] and record["local_consistency"] == 1
    assert not record["categorical_support_labels_ok"]
    assert record["nonneutral_padding_slots"] == 1
    assert record["nonzero_padding_bond_entries"] == 2
    assert record["graph_id"] != clean["graph_id"]


def test_score_keeps_missing_positive_component_undefined_without_renormalizing():
    components = {"validity": .5, "local_consistency": .75, "connectedness": 1}
    base = combine_quality(components, legacy())
    assert base["score"] == pytest.approx(70)
    custom = combine_quality(components, legacy(repair_quality=.2))
    assert custom["score"] is None and custom["missing_components"] == ["repair_quality"]
    assert custom["contributions"]["validity"] == pytest.approx(100 * .4 * .5 / 1.2)
    assert custom["profile_sha256"] != base["profile_sha256"]
    with pytest.raises(ValueError):
        combine_quality({"validity": math.nan})
    with pytest.raises(ValueError):
        QualityScoreConfig(validity=-1)


def test_edit_cost_counts_undirected_bond_once_and_honors_inverse_mapping():
    spec = GraphSpec(max_nodes=4)
    raw = graph("CCO", spec)
    permutation = np.array([2, 0, 1, 3])
    reordered = (raw[0][permutation], raw[1][permutation], raw[2][np.ix_(permutation, permutation)])
    assert graph_edit_cost(raw, reordered, spec, corrected_to_raw=permutation)["total"] == 0
    edited = tuple(x.copy() for x in raw)
    edited[2][0, 1] = edited[2][1, 0] = 2
    assert graph_edit_cost(raw, edited, spec)["counts"] == {"occupancy": 0, "atom": 0, "charge": 0, "bond": 1}
    with pytest.raises(ValueError, match="permutation"):
        graph_edit_cost(raw, reordered, spec, corrected_to_raw=[0, 0, 2, 3])


def test_repair_budget_censoring_and_already_valid_distance():
    spec = GraphSpec(max_nodes=4)
    raw, valid = invalid_co(spec), graph("CO", spec)
    report = measure_repairs(raw, [None, valid], spec, budget=budget())
    assert report["status"] == "repaired" and report["best_observed_edit_cost"] == 1
    assert report["repair_quality"] == pytest.approx(math.exp(-1 / 3))
    assert report["attempts"][0]["status"] == "failed"
    failed = measure_repairs(raw, [None, raw], spec, budget=budget())
    assert failed["status"] == "censored" and failed["best_observed_edit_cost"] is None
    assert failed["repair_quality"] == 0
    already = measure_repairs(valid, [graph("CC", spec), None], spec, budget=budget())
    assert already["status"] == "already_valid" and already["best_observed_edit_cost"] == 0
    assert already["attempts"][0]["edit_cost"]["total"] == 1
    with pytest.raises(ValueError, match="budget"):
        measure_repairs(raw, [valid], spec, budget=budget())


def test_repair_summary_rejects_mismatched_identity_budget_and_fixed_occupancy():
    spec = GraphSpec(max_nodes=4)
    raw, valid = invalid_co(spec), graph("CO", spec)
    report = measure_repairs(raw, [valid], spec, budget=budget(1))
    record = diagnose_graph(*raw, spec)
    summary = summarize_quality([record], legacy(repair_quality=.2), repairs=[report])
    assert summary["mixture"]["score"] is not None
    assert summary["component_observations"]["repair_quality"] == 1
    with pytest.raises(ValueError, match="different raw"):
        summarize_quality([diagnose_graph(*valid, spec)], repairs=[report])
    changed = measure_repairs(raw, [valid], spec, budget={**budget(1), "transitions_per_attempt": 9})
    with pytest.raises(ValueError, match="budgets"):
        summarize_quality([record, record], repairs=[report, changed])
    fixed = measure_repairs(raw, [graph("C", spec)], spec, budget=budget(1), fixed_occupancy=True)
    assert fixed["status"] == "censored"
    assert fixed["attempts"][0]["status"] == "occupancy_unsupported"


def test_broken_projector_output_is_retained_without_losing_later_success():
    spec = GraphSpec(max_nodes=4)
    raw, good = invalid_co(spec), graph("CO", spec)
    bad = tuple(x.copy() for x in good)
    bad[2][0, 0] = 1
    report = measure_repairs(raw, [bad, good], spec, budget=budget())
    assert report["status"] == "repaired"
    assert report["attempts"][0]["status"] == "structural_unsupported"
    assert report["attempts"][1]["attempt_index"] == 1


def test_v2_equal_thirds_uses_proximity_not_binary_validity():
    values = {"validity": 0., "repair_proximity": .5, "local_consistency": .5, "connectedness": 1.}
    result = combine_quality(values)
    assert result["score"] == pytest.approx(100 * 2 / 3)
    assert result["contributions"]["validity"] == 0
    assert result["contributions"]["repair_proximity"] == result["contributions"]["local_consistency"]
    assert combine_quality({**values, "validity": 1.})["score"] == result["score"]
    assert result["version"] == "graded-chemistry-v2"
    old = combine_quality(values, legacy())
    assert old["version"] == "graded-chemistry-v1"
    assert old["profile_sha256"] == "7c1b043a67fd56bc2de44f175ca965e759e2c785575b3bff6598b565f32b32fa"


def test_actual_one_and_two_edit_repairs_are_verified_and_leave_raw_unchanged():
    spec = GraphSpec(max_nodes=4)
    evaluator = GradedEvaluator(spec)
    raw = invalid_co(spec)
    original = tuple(x.copy() for x in raw)
    one = evaluator.diagnose(*raw)
    assert not one["valid"] and one["repair_proximity"] == .5
    report = one["repair_search"]
    assert report["best_observed_edit_cost"] == 1
    assert report["distance_is_exact_in_fixed_occupancy_space"]
    witness = tuple(np.asarray(report["witness"][key]) for key in ("atoms", "charges", "bonds"))
    assert diagnose_graph(*witness, spec)["valid"]
    assert graph_edit_cost(raw, witness, spec)["total"] == 1
    for before, after in zip(original, raw):
        np.testing.assert_array_equal(before, after)
    # Two distinct impossible oxygen valences need at least two label changes.
    two_raw = graph("OCCO", spec)
    two_raw[2][0, 1] = two_raw[2][1, 0] = 3
    two_raw[2][2, 3] = two_raw[2][3, 2] = 3
    two = evaluator.diagnose(*two_raw)["repair_search"]
    assert two["best_observed_edit_cost"] == 2
    assert two["one_edit_exhausted"] and two["distance_is_exact_in_fixed_occupancy_space"]
    assert two["proximity"] == pytest.approx(1 / 3)
    assert sum(two["edit_counts"].values()) == 2


def test_budget_exhaustion_is_unknown_with_bounds_and_never_silently_zero():
    spec = GraphSpec(max_nodes=4)
    evaluator = GradedEvaluator(spec, search_config=RepairSearchConfig(max_edits=1, checks_per_depth=1))
    record = evaluator.diagnose(*invalid_co(spec))
    search = record["repair_search"]
    assert search["checks"] == 1 and search["status"] == "censored"
    assert search["best_observed_edit_cost"] is None and record["repair_proximity"] is None
    assert search["proximity_bounds"] == [0., .5]
    assert not search["one_edit_exhausted"]
    summary = summarize_quality([record])
    assert summary["mixture"]["score"] is None
    assert summary["mixture"]["missing_components"] == ["repair_proximity"]
    assert summary["mixture"]["score_interval"][0] < summary["mixture"]["score_interval"][1]


def test_memoization_preserves_attempts_but_avoids_duplicate_search():
    spec = GraphSpec(max_nodes=4)
    evaluator = GradedEvaluator(spec)
    raw = invalid_co(spec)
    first = evaluator.diagnose(*raw)
    checks = evaluator.physical_candidate_checks
    second = evaluator.diagnose(*raw)
    assert first == second and evaluator.cache_hits == 1
    assert evaluator.physical_candidate_checks == checks > 0
    assert summarize_quality([first, second])["attempts"] == 2


def test_empty_is_ineligible_for_fixed_occupancy_and_stale_diagnostics_rejected():
    spec = GraphSpec(max_nodes=4)
    valid = graph("CO", spec)
    stale = diagnose_graph(*valid, spec)
    with pytest.raises(ValueError, match="different graph"):
        search_repair(invalid_co(spec), spec, stale, diagnose_graph)
    empty = (np.zeros(4, dtype=int), np.full(4, 2), np.zeros((4, 4), dtype=int))
    record = GradedEvaluator(spec).diagnose(*empty)
    assert record["repair_search"]["status"] == "unsupported"
    assert summarize_quality([record])["mixture"]["score"] is None


def test_legacy_profile_does_not_invoke_search():
    spec = GraphSpec(max_nodes=4)
    evaluator = GradedEvaluator(spec, legacy())
    record = evaluator.diagnose(*invalid_co(spec))
    assert "repair_search" not in record and evaluator.physical_candidate_checks == 0
    assert summarize_quality([record], legacy())["mixture"]["score"] is not None
