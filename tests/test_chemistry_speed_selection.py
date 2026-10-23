"""Allocated-HPC policy fixtures for choosing a fully gated execution variant.

These synthetic metadata records test selection boundaries, not chemistry,
measured speed or permission to train. No scheduler or model is invoked here.
Do not run locally under the project's numerical-test restriction.
"""

from copy import deepcopy

import pytest

from latentmol.chemistry_speed_selection import (
    REQUIRED_GATES,
    choose_candidate,
    summarize_candidate,
)


EXPECTED_GATES = {
    "finite", "size_retention", "nonempty_retention", "sulfur_no_inflation",
    "composition_no_deterioration", "charge_no_deterioration",
    "reconstruction_retention", "valid_reconstruction_retention", "exact_ratio",
    "exact_native_resume", "prior_target_coverage", "actual_repair_gradient",
    "decoder_gradient", "encoder_gradient", "memory", "complete_epoch_fits",
    "no_search_errors", "exact_reference_execution",
}
PARITY_CHECKS = {
    "full_logical_batch", "gradients_exact", "metrics_exact", "model_exact",
    "model_modes_exact", "optimizer_exact", "ordered_teacher_receipt_chain_exact",
    "reference_exact", "reference_gradients_exact", "reference_modes_exact",
    "reference_unchanged", "repair_target_observed", "rng_exact",
}
TRAINING_MOLECULES = 1_688_539


def candidate(prior_ratio=3, name="r3_reference", updates=3):
    return {"id": name, "prior_ratio": prior_ratio, "benchmark_updates": updates}


def native_gate(prior_ratio=3, walls=(3., 4., 5.), projected=20_000.):
    return {
        "status": "passed",
        "plan_sha256": "a" * 64,
        "gates": {name: True for name in EXPECTED_GATES},
        "native_execution_parity": {
            "status": "passed", "passed": True,
            "restoration": {"passed": True},
            "checks": {name: True for name in PARITY_CHECKS},
        },
        "updates": [{"real_points": 256, "fresh_prior_points": 256 * prior_ratio,
                     "update_seconds": wall} for wall in walls],
        "projected_epoch_seconds": projected,
    }


def observation(name, ratio, seconds, *, eligible=True):
    return {"id": name, "prior_ratio": ratio, "eligible": eligible,
            "selection_epoch_seconds": seconds}


def assert_rejected(variant, gate, **kwargs):
    """An intentional schema error or ineligible record both forbid promotion."""
    try:
        row = summarize_candidate(variant, gate, 0,
            training_molecules=TRAINING_MOLECULES, **kwargs)
    except ValueError:
        return
    assert row["eligible"] is False
    assert row.get("rejection")


def test_all_eighteen_original_gates_are_still_mandatory():
    assert REQUIRED_GATES == EXPECTED_GATES
    assert len(REQUIRED_GATES) == 18


def test_summary_ranks_maximum_update_not_lucky_mean_and_preserves_inputs():
    variant, gate = candidate(), native_gate()
    before = deepcopy((variant, gate))
    row = summarize_candidate(variant, gate, 0, training_molecules=TRAINING_MOLECULES)
    assert row["eligible"] is True
    assert row["selection_epoch_seconds"] == pytest.approx(
        5. * TRAINING_MOLECULES / 256 + 600., rel=1e-13)
    assert row["selection_epoch_seconds"] > 4. * TRAINING_MOLECULES / 256 + 600.
    assert row["timed_update_seconds"] == [3., 4., 5.]
    assert (variant, gate) == before


def test_existing_more_conservative_native_projection_is_never_lowered():
    gate = native_gate(projected=80_000.)
    row = summarize_candidate(candidate(), gate, 0, training_molecules=TRAINING_MOLECULES)
    assert row["eligible"] is True
    assert row["selection_epoch_seconds"] == 80_000.


@pytest.mark.parametrize("name", sorted(EXPECTED_GATES))
def test_a_failed_native_gate_cannot_be_overridden_by_speed(name):
    gate = native_gate(walls=(.1, .1, .1), projected=1.)
    gate["gates"][name] = False
    row = summarize_candidate(candidate(), gate, 0, training_molecules=TRAINING_MOLECULES)
    assert row["eligible"] is False
    assert choose_candidate([row], fallback_evaluated=True)["status"] == "stopped"


@pytest.mark.parametrize("change", ["missing", "extra", "empty", "nonboolean"])
def test_incomplete_or_mutated_gate_schema_is_not_scientific_approval(change):
    gate = native_gate()
    if change == "missing":
        gate["gates"].pop("memory")
    elif change == "extra":
        gate["gates"]["new_unreviewed_gate"] = True
    elif change == "empty":
        gate["gates"] = {}
    else:
        gate["gates"]["memory"] = 1
    assert_rejected(candidate(), gate)


@pytest.mark.parametrize("name", sorted(PARITY_CHECKS))
@pytest.mark.parametrize("change", ["false", "missing"])
def test_every_native_reference_parity_check_is_required(name, change):
    gate = native_gate()
    checks = gate["native_execution_parity"]["checks"]
    if change == "missing":
        checks.pop(name)
    else:
        checks[name] = False
    assert_rejected(candidate(), gate)


@pytest.mark.parametrize("change", ["extra", "empty", "list", "missing_proof",
    "not_passed", "false_passed", "false_restoration", "missing_restoration"])
def test_native_proof_shape_and_restoration_are_mandatory(change):
    gate = native_gate()
    proof = gate["native_execution_parity"]
    if change == "extra":
        proof["checks"]["unreviewed_extra"] = True
    elif change == "empty":
        proof["checks"] = {}
    elif change == "list":
        proof["checks"] = [True]
    elif change == "missing_proof":
        gate.pop("native_execution_parity")
    elif change == "not_passed":
        proof["status"] = "failed"
    elif change == "false_passed":
        proof["passed"] = False
    elif change == "false_restoration":
        proof["restoration"]["passed"] = False
    else:
        proof.pop("restoration")
    assert_rejected(candidate(), gate)


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_all_declared_timed_updates_are_required(count):
    assert_rejected(candidate(), native_gate(walls=(2.,) * count))


@pytest.mark.parametrize("value", [None, True, 0, -1, "3", float("nan"), float("inf")])
def test_nonfinite_nonpositive_or_coerced_timing_is_ineligible(value):
    gate = native_gate()
    gate["updates"][1]["update_seconds"] = value
    assert_rejected(candidate(), gate)


@pytest.mark.parametrize("value", [None, True, 0, -1, "30000", float("nan"), float("inf")])
def test_native_projection_must_be_positive_and_finite(value):
    assert_rejected(candidate(), native_gate(projected=value))


@pytest.mark.parametrize("field,value", [("real_points", 255), ("real_points", True),
    ("real_points", 256.), ("fresh_prior_points", 512),
    ("fresh_prior_points", 768.), ("fresh_prior_points", "768")])
def test_wrong_logical_batch_or_ratio_is_rejected(field, value):
    gate = native_gate()
    gate["updates"][0][field] = value
    assert_rejected(candidate(), gate)


def test_ratio_two_is_checked_against_its_own_declared_population():
    row = summarize_candidate(candidate(2, "r2_reference"), native_gate(2), 0,
                              training_molecules=TRAINING_MOLECULES)
    assert row["eligible"] is True
    assert row["prior_ratio"] == 2
    assert_rejected(candidate(2, "r2_wrong_count"), native_gate(3))


@pytest.mark.parametrize("returncode", [1, -9, None, False, 0., "0"])
def test_only_successful_integer_worker_exit_can_be_eligible(returncode):
    try:
        row = summarize_candidate(candidate(), native_gate(), returncode,
                                  training_molecules=TRAINING_MOLECULES)
    except ValueError:
        return
    assert row["eligible"] is False


@pytest.mark.parametrize("field,value", [("training_molecules", 0), ("training_molecules", True),
    ("training_molecules", 1.5), ("real_batch_size", 0), ("real_batch_size", False),
    ("real_batch_size", 256.)])
def test_invalid_population_arguments_fail_before_selection(field, value):
    arguments = {"training_molecules": TRAINING_MOLECULES, "real_batch_size": 256}
    arguments[field] = value
    with pytest.raises(ValueError):
        summarize_candidate(candidate(), native_gate(), 0, **arguments)


@pytest.mark.parametrize("field,value", [("id", ""), ("id", None), ("id", 1),
    ("prior_ratio", True), ("prior_ratio", 3.), ("prior_ratio", 4),
    ("benchmark_updates", 0), ("benchmark_updates", True), ("benchmark_updates", 3.)])
def test_invalid_candidate_declaration_fails_closed(field, value):
    variant = candidate()
    variant[field] = value
    with pytest.raises(ValueError):
        summarize_candidate(variant, native_gate(), 0, training_molecules=TRAINING_MOLECULES)


def test_qualifying_ratio_three_is_preferred_even_when_ratio_two_is_faster():
    rows = [observation("r2_fast", 2, 10_000.), observation("r3_b", 3, 32_000.),
            observation("r3_a", 3, 30_000.)]
    before = deepcopy(rows)
    chosen = choose_candidate(rows, fallback_evaluated=True)
    assert chosen["winner"] == "r3_a"
    assert chosen["prior_ratio"] == 3
    assert chosen["status"] == "selected_not_training_started"
    assert rows == before


def test_fallback_requires_an_explicit_completed_family_observation():
    rows = [observation("r3_slow", 3, 40_000.), observation("r2_ready", 2, 20_000.)]
    result = choose_candidate(rows, fallback_evaluated=False)
    assert result["status"] == "fallback_required"
    assert result["winner"] is None


def test_threshold_is_strict_and_fastest_healthy_fallback_can_exceed_ten_hours():
    rows = [observation("r3_boundary", 3, 36_000.), observation("r2_slower", 2, 44_000.),
            observation("r2_fastest", 2, 42_000.), observation("r2_failed", 2, 1., eligible=False)]
    chosen = choose_candidate(rows, fallback_evaluated=True)
    assert chosen["winner"] == "r2_fastest"
    assert chosen["below_preference_threshold"] is False
    assert chosen["preference_threshold_seconds"] == 36_000.


def test_no_healthy_ratio_two_means_stop_not_choose_slow_ratio_three():
    rows = [observation("r3_slow", 3, 40_000.), observation("r2_failed", 2, None, eligible=False)]
    result = choose_candidate(rows, fallback_evaluated=True)
    assert result["status"] == "stopped"
    assert result["winner"] is None


@pytest.mark.parametrize("rows,fallback,status", [([], False, "fallback_required"),
    ([], True, "stopped"), ([observation("r3_bad", 3, None, eligible=False)], False, "fallback_required")])
def test_empty_or_rejected_observations_are_never_winners(rows, fallback, status):
    result = choose_candidate(rows, fallback_evaluated=fallback)
    assert result["status"] == status
    assert result["winner"] is None


@pytest.mark.parametrize("ratio,seconds", [(3, 30_000.), (2, 40_000.)])
def test_ties_use_canonical_id_independent_of_observation_order(ratio, seconds):
    rows = [observation("candidate_z", ratio, seconds), observation("candidate_a", ratio, seconds)]
    assert choose_candidate(rows, fallback_evaluated=True)["winner"] == "candidate_a"
    assert choose_candidate(list(reversed(rows)), fallback_evaluated=True)["winner"] == "candidate_a"


def test_duplicate_observations_cannot_double_count_a_variant():
    with pytest.raises(ValueError):
        choose_candidate([observation("same", 3, 10_000.), observation("same", 3, None, eligible=False)])


@pytest.mark.parametrize("value", [0, -1, True, "36000", None, float("nan"), float("inf")])
def test_preference_threshold_rejects_bad_values(value):
    with pytest.raises(ValueError):
        choose_candidate([], preferred_epoch_seconds=value)


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_fallback_flag_is_a_boolean_protocol_fact(value):
    with pytest.raises(ValueError):
        choose_candidate([], fallback_evaluated=value)


@pytest.mark.parametrize("field,value", [("id", ""), ("id", None), ("id", []),
    ("prior_ratio", 1), ("prior_ratio", True), ("prior_ratio", "3"),
    ("eligible", "true"), ("selection_epoch_seconds", None),
    ("selection_epoch_seconds", True), ("selection_epoch_seconds", 0),
    ("selection_epoch_seconds", float("nan")), ("selection_epoch_seconds", float("inf"))])
def test_malformed_eligible_observation_never_becomes_a_winner(field, value):
    row = observation("candidate", 3, 10_000.)
    row[field] = value
    with pytest.raises(ValueError):
        choose_candidate([row], fallback_evaluated=True)
