"""Allocated CPU controls; toy passes do not establish native validation."""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/protenix_precision_admission.py"
SPEC = importlib.util.spec_from_file_location("protenix_precision_admission_fixture", PATH)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


@pytest.fixture(autouse=True)
def allocated():
    assert os.environ.get("LSB_JOBID"), "Fixtures require allocated DTU HPC"


def graph(boundary=False):
    if boundary:
        return {"X": np.tile(np.asarray([1, 0, 0], np.float32), (4, 1)),
                "E_upper": np.tile(np.asarray([1, 0], np.float32), (4, 1))}
    return {"X": np.tile(np.asarray([0.5, 0.25, 0.25], np.float32), (4, 1)),
            "E_upper": np.full((4, 2), 0.5, dtype=np.float32)}


def originals():
    return {"X": [{"row": 3, "source": 0, "target": 1}], "E_upper": []}


def precision():
    return {"full_model_float64": True, "parameters_float64": True,
            "native_features_float64": True, "native_operations_float64": True,
            "final_reduction_only": False}


def evaluators(*, production_wrong=1.0, reference_wrong=1.0, corrupt_field=None,
               offset=0.0, curvature=0.0, base=None):
    base = graph() if base is None else base
    weights = {"X": np.tile(np.asarray([1, 3, -2], np.float64), (4, 1)),
               "E_upper": np.tile(np.asarray([2, -1], np.float64), (4, 1))}

    def primal(values, dtype):
        result = offset
        for field, coefficient in weights.items():
            value = values[field].astype(np.float64)
            displacement = value - base[field].astype(np.float64)
            result += np.sum(value * coefficient, dtype=np.float64)
            result += curvature * np.sum(displacement ** 3 * coefficient, dtype=np.float64)
        return dtype(result)

    def ad(values, dtype, wrong):
        gradients = {field: (coefficient + 3 * curvature * coefficient
                            * (values[field].astype(np.float64) - base[field].astype(np.float64)) ** 2)
                     for field, coefficient in weights.items()}
        gradients = {field: (value * (wrong if corrupt_field is None or field == corrupt_field else 1)).astype(dtype)
                     for field, value in gradients.items()}
        return primal(values, dtype), gradients

    return (lambda values: primal(values, np.float32),
            lambda values: ad(values, np.float32, production_wrong),
            lambda values: primal(values, np.float64),
            lambda values: ad(values, np.float64, reference_wrong))


def run(values=None, *, production_wrong=1.0, reference_wrong=1.0,
        corrupt_field=None, offset=0.0, curvature=0.0, **kwargs):
    values = graph() if values is None else values
    callbacks = evaluators(production_wrong=production_wrong, reference_wrong=reference_wrong,
                           corrupt_field=corrupt_field, offset=offset, curvature=curvature, base=values)
    return policy.evaluate(values, *callbacks, lambda: {"model": "fixed", "context": "fixed", "saved": "fixed"},
                           original_directions=originals(), reference_precision=precision(), **kwargs)


def test_linear_separate_precision_pass_preserves_original_failure():
    result = run()
    assert result["can_admit_new_child"] and result["status"] == "passed"
    assert result["original_failed_outcome_preserved"] and not result["can_certify_original_study"]
    assert result["optimization_updates"] == 0 and result["baseline_fp32_unchanged"]
    assert all(value["meaningful_strong_rows"] == 2 for value in result["field_outcomes"].values())
    assert all(row["selected_pair"] == [0.0001, 0.00003] for row in result["directions"])
    assert all(len(row["selected_measurements"]) == 2 for row in result["directions"])
    assert all(row["passed"] for row in result["directions"])


@pytest.mark.parametrize("wrong", [0.0, -1.0, 1.5])
def test_wrong_production_gradients_block(wrong):
    result = run(production_wrong=wrong)
    assert not result["can_admit_new_child"] and result["failures"]


@pytest.mark.parametrize("wrong", [0.0, -1.0, 1.5])
def test_wrong_reference_gradients_block(wrong):
    result = run(reference_wrong=wrong)
    assert not result["can_admit_new_child"] and result["failures"]


@pytest.mark.parametrize("field", ["X", "E_upper"])
@pytest.mark.parametrize("target", ["production", "reference"])
def test_single_field_corruption_cannot_hide(field, target):
    kwargs = {target + "_wrong": 1.5, "corrupt_field": field}
    result = run(**kwargs)
    assert not result["can_admit_new_child"]


def test_selection_sealed_before_any_perturbed_callback_and_no_graph_mutation():
    values = graph()
    before = {field: value.copy() for field, value in values.items()}
    calls = []
    callbacks = evaluators()

    def guarded(callback, dtype):
        def call(payload):
            assert all(value.dtype == dtype for value in payload.values())
            is_baseline = all(np.array_equal(payload[field], values[field].astype(dtype)) for field in values)
            assert is_baseline or calls, "Perturbed callback preceded immutable selection"
            return callback(payload)
        return call

    result = policy.evaluate(values, guarded(callbacks[0], np.float32), guarded(callbacks[1], np.float32),
        guarded(callbacks[2], np.float64), guarded(callbacks[3], np.float64), lambda: {"frozen": True},
        original_directions=originals(), reference_precision=precision(), seal_callback=lambda row: calls.append(row))
    assert result["can_admit_new_child"] and len(calls) == 1
    assert not calls[0]["selection_uses_FD"] and not calls[0]["after_result_fallback"]
    assert result["plan"] == calls[0]
    assert all(np.array_equal(value, before[field]) for field, value in values.items())
    assert result["baseline_reference"]["X"]["dtype"] == "float64"


def test_original_scales_are_retained_as_noncertifying_observations():
    result = run()
    for direction in result["directions"]:
        observations = direction["original_coarse_observations"]
        assert [row["epsilon"] for row in observations] == [0.01, 0.003, 0.001]
        assert all(not row["certifying"] and not row["replaces_original_gate"] for row in observations)


def test_production_is_only_read_at_saved_baseline_before_reference_selection():
    values = graph()
    callbacks = evaluators()
    counts = {"primal": 0, "ad": 0}

    def production(callback, name):
        def call(payload):
            counts[name] += 1
            assert all(np.array_equal(payload[field], value) for field, value in values.items())
            return callback(payload)
        return call

    evidence = {"path": "/immutable/original-admission.json", "bytes": 42, "sha256": "a" * 64}
    result = policy.evaluate(values, production(callbacks[0], "primal"), production(callbacks[1], "ad"),
        callbacks[2], callbacks[3], lambda: {"frozen": True}, original_directions=originals(),
        reference_precision=precision(), original_observations=evidence)
    assert result["can_admit_new_child"] and counts == {"primal": 3, "ad": 3}
    for direction in result["directions"]:
        for row in direction["original_coarse_observations"]:
            assert not row["evaluated"] and not row["production_model_reloaded"]
            assert row["outcome"] == "preserved_external_original_FP32_evidence"
            assert row["external_evidence"] == evidence


def test_strong_raw_fields_do_not_depend_on_weak_geometric_first_rows():
    values = graph()
    callbacks = evaluators()
    # First row is mathematically absent from this objective, while the other
    # rows are informative. This is a coherent primal/gradient zero-row control.
    coefficients = {"X": np.tile(np.asarray([1, 3, -2], np.float64), (4, 1)),
                    "E_upper": np.tile(np.asarray([2, -1], np.float64), (4, 1))}
    for value in coefficients.values():
        value[0] = 0

    def primal(payload, dtype):
        return dtype(sum(np.sum(payload[field] * value, dtype=np.float64) for field, value in coefficients.items()))

    def ad(payload, dtype):
        return primal(payload, dtype), {field: value.astype(dtype) for field, value in coefficients.items()}

    result = policy.evaluate(values, lambda payload: primal(payload, np.float32),
        lambda payload: ad(payload, np.float32), lambda payload: primal(payload, np.float64),
        lambda payload: ad(payload, np.float64), lambda: {"frozen": True},
        original_directions=originals(), reference_precision=precision())
    assert result["can_admit_new_child"]
    assert all(row["row"] != 0 for row in result["directions"])


def test_boundary_uses_fixed_feasible_inward_points_without_clipping():
    result = run(graph(boundary=True))
    assert result["can_admit_new_child"]
    assert all(measurement["probe"]["scheme"] == "one_sided_inward"
               for row in result["directions"] for measurement in row["selected_measurements"])
    assert result["baseline_fp32_unchanged"]


def test_nonlocal_coarse_curvature_does_not_choose_a_favorable_pair_after_FD():
    # Quartic terms create coarse curvature but leave the fixed finest pair local.
    values = graph()
    weights = {"X": np.tile(np.asarray([1, 3, -2], np.float64), (4, 1)),
               "E_upper": np.tile(np.asarray([2, -1], np.float64), (4, 1))}

    def primal(payload, dtype):
        return dtype(sum(np.sum(payload[field] * weight, dtype=np.float64)
                         + 1e7 * np.sum((payload[field].astype(np.float64) - values[field]) ** 4, dtype=np.float64)
                         for field, weight in weights.items()))

    def ad(payload, dtype):
        return primal(payload, dtype), {field: (weight + 4e7
            * (payload[field].astype(np.float64) - values[field]) ** 3).astype(dtype)
            for field, weight in weights.items()}

    result = policy.evaluate(values, lambda payload: primal(payload, np.float32), lambda payload: ad(payload, np.float32),
        lambda payload: primal(payload, np.float64), lambda payload: ad(payload, np.float64),
        lambda: {"frozen": True}, original_directions=originals(), reference_precision=precision())
    assert result["can_admit_new_child"]
    assert all(row["selected_pair"] == [0.0001, 0.00003] for row in result["directions"])


def test_selected_pair_curvature_blocks_even_if_another_scale_might_agree():
    result = run(curvature=1e12)
    assert not result["can_admit_new_child"]
    assert all(row["selected_pair"] == [0.0001, 0.00003] for row in result["directions"])
    assert any(not row["reference_endpoint_locality_passed"]
               for direction in result["directions"] for row in direction["selected_measurements"])


def test_unresolved_FP64_signal_is_blocked_without_fake_zero_gradient_pass():
    # Exact binary offset keeps production/reference scalar parity intact while
    # making every proposed delta unresolvable even in FP64.
    result = run(offset=float(2 ** 100))
    assert not result["can_admit_new_child"]
    assert any("Insufficient meaningful" in value for value in result["failures"])


def test_reference_final_reduction_cast_is_explicitly_rejected():
    metadata = precision()
    metadata["final_reduction_only"] = True
    callbacks = evaluators()
    result = policy.evaluate(graph(), *callbacks, lambda: {"frozen": True},
                              original_directions=originals(), reference_precision=metadata)
    assert not result["can_admit_new_child"]
    assert any("full-model FP64" in value for value in result["failures"])


def test_cast_reference_scalar_without_float64_raw_gradient_is_rejected():
    callbacks = evaluators()

    def fake(payload):
        scalar, gradients = callbacks[3](payload)
        return scalar, {field: value.astype(np.float32) for field, value in gradients.items()}

    result = policy.evaluate(graph(), callbacks[0], callbacks[1], callbacks[2], fake,
                              lambda: {"frozen": True}, original_directions=originals(), reference_precision=precision())
    assert not result["can_admit_new_child"]


def test_perturbed_reference_primal_and_AD_scalar_parity_is_mandatory():
    values = graph()
    callbacks = evaluators()

    def changed(payload):
        scalar, gradient = callbacks[3](payload)
        baseline = all(np.array_equal(payload[field], value.astype(np.float64))
                       for field, value in values.items())
        return scalar if baseline else scalar + np.float64(1e-5), gradient

    result = policy.evaluate(values, callbacks[0], callbacks[1], callbacks[2], changed,
        lambda: {"frozen": True}, original_directions=originals(), reference_precision=precision())
    assert not result["can_admit_new_child"]
    assert any("Perturbed reference independent primal/AD scalar parity" in value for value in result["failures"])


def test_saved_endpoint_must_have_authentic_original_X_contrast():
    result = policy.evaluate(graph(), *evaluators(), lambda: {"frozen": True},
                              original_directions={"X": [], "E_upper": []}, reference_precision=precision())
    assert not result["can_admit_new_child"]
    assert any("authentic original X" in value for value in result["failures"])


def test_future_endpoint_scope_does_not_invent_old_contrasts():
    result = policy.evaluate(graph(), *evaluators(), lambda: {"frozen": True},
        original_directions={"X": [], "E_upper": []}, scope="not_preexisting_at_new_endpoint",
        reference_precision=precision())
    assert result["can_admit_new_child"]
    assert all("authentic_original" not in row["roles"] for row in result["directions"])


def test_evaluator_graph_mutation_is_rejected():
    callbacks = evaluators()

    def mutate(payload):
        payload["X"][0, 0] = np.float32(0.7)
        return callbacks[0](payload)

    result = policy.evaluate(graph(), mutate, callbacks[1], callbacks[2], callbacks[3],
                              lambda: {"frozen": True}, original_directions=originals(), reference_precision=precision())
    assert not result["can_admit_new_child"]
    assert any("mutated" in value for value in result["failures"])


def test_frozen_identity_change_is_rejected():
    calls = []

    def frozen():
        calls.append(1)
        return {"model": len(calls)}

    result = policy.evaluate(graph(), *evaluators(), frozen,
                              original_directions=originals(), reference_precision=precision())
    assert not result["can_admit_new_child"] and not result["frozen_unchanged"]


def test_low_residual_below_uncertainty_is_unresolved_never_a_contradiction():
    result = policy._agreement(1e-9, 2e-9, 1e-7)
    assert not result["passed"] and result["outcome"] == "unresolved_precision"


def test_uncertainty_must_fit_inside_original_relative_tolerance():
    assert policy._agreement(1.0, 1.0, 0.01)["passed"]
    assert not policy._agreement(1.0, 1.0, 0.2)["passed"]
    assert policy._agreement(1.0, -1.0, 0.01)["outcome"] == "resolved_disagreement"


def test_non_FP32_saved_graph_is_rejected_before_callbacks():
    values = {field: value.astype(np.float64) for field, value in graph().items()}
    with pytest.raises(ValueError, match="FP32"):
        policy.evaluate(values, *evaluators(), lambda: {"frozen": True},
                          original_directions=originals(), reference_precision=precision())


def test_modified_sealed_plan_cannot_control_acceptance():
    def tamper(row):
        row["directions"][0]["selected_pair"] = [0.003, 0.001]

    result = run(seal_callback=tamper)
    assert result["can_admit_new_child"]
    assert all(row["selected_pair"] == [0.0001, 0.00003] for row in result["directions"])
