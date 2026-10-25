"""Allocated CPU controls for prospective admission; no pretrained claims."""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/admission.py"
SPEC = importlib.util.spec_from_file_location("prospective_admission_fixture", PATH)
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)


@pytest.fixture(autouse=True)
def allocated_cpu(monkeypatch):
    assert os.environ.get("LSB_JOBID"), "Admission fixtures require allocated DTU HPC"
    monkeypatch.setenv("LSB_JOBID", "12345")


def graph(boundary=False):
    if boundary:
        x = np.tile(np.asarray([1, 0, 0], np.float32), (4, 1))
        e = np.tile(np.asarray([1, 0], np.float32), (4, 1))
    else:
        x = np.full((4, 3), np.float32(1 / 3))
        e = np.full((4, 2), np.float32(0.5))
    return {"X": x, "E_upper": e}


def originals(empty_e=False):
    return {"X": [{"row": 0, "source": 0, "target": 1}],
            "E_upper": [] if empty_e else [{"row": 0, "source": 0, "target": 1}]}


def weights(values):
    return {"X": np.tile(np.asarray([1, 3, -2], np.float64), (values["X"].shape[0], 1)),
            "E_upper": np.tile(np.asarray([2, -1], np.float64), (values["E_upper"].shape[0], 1))}


def evaluators(values, wrong=1.0, offset=0.0):
    coefficients = weights(values)

    def primal(payload):
        return np.float32(offset + sum(np.sum(payload[field].astype(np.float64) * coefficient)
                                        for field, coefficient in coefficients.items()))

    def autodiff(payload):
        return primal(payload), {field: (value * wrong).astype(np.float32) for field, value in coefficients.items()}

    return primal, autodiff, lambda: {"model": "frozen-model", "context": "frozen-context", "key": 17}


def run(values=None, *, directions=None, scope="saved_failed_endpoint", wrong=1.0, offset=0.0):
    values = graph() if values is None else values
    planned = admission.plan(values, originals() if directions is None else directions,
                             original_direction_scope=scope)
    return admission.evaluate(planned, *evaluators(values, wrong=wrong, offset=offset))


def test_linear_raw_jacobian_admits_new_child_without_certifying_original():
    report = run()
    assert report["status"] == "passed" and report["can_admit_new_child"]
    assert report["original_run_failed_preserved"] and not report["original_gate_replaced"]
    assert not report["can_certify_original_study"] and not report["toy_pass_is_native_validation"]
    assert set(report["field_outcomes"]) == {"X", "E_upper"}
    assert all(value["meaningful_passed_directions"] >= 1 for value in report["field_outcomes"].values())


def test_all_points_and_fixed_ladders_are_sealed_before_any_callback():
    values = graph()
    before = {field: value.copy() for field, value in values.items()}
    planned = admission.plan(values, originals())
    assert planned == admission.plan(values, originals())
    assert planned["policy"]["original_epsilons"] == [0.01, 0.003, 0.001]
    assert planned["policy"]["local_epsilons"] == [0.001, 0.0003, 0.0001, 0.00003]
    assert all(np.array_equal(values[field], before[field]) for field in values)
    for direction in planned["directions"]:
        assert [value["epsilon"] for value in direction["probes"]] == [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]


def test_original_unreached_E_is_explicit_empty_not_invented():
    planned = admission.plan(graph(), originals(empty_e=True))
    e = [value for value in planned["directions"] if value["field"] == "E_upper"]
    assert len(e) == 4 and all("authentic_original" not in value["roles"] for value in e)
    report = admission.evaluate(planned, *evaluators(graph()))
    assert report["can_admit_new_child"]


def test_future_terminal_has_no_fabricated_historical_contrast():
    report = run(directions={"X": [], "E_upper": []}, scope="not_preexisting_at_new_endpoint")
    assert report["can_admit_new_child"]
    assert all("authentic_original" not in direction["roles"] for direction in report["plan"]["directions"])


def test_unreached_original_component_retains_distinct_scope_and_requires_geometry():
    report = run(directions={"X": [], "E_upper": []}, scope="original_probe_not_reached")
    assert report["can_admit_new_child"] and report["original_run_failed_preserved"]
    assert report["plan"]["original_direction_scope"] == "original_probe_not_reached"
    assert not report["can_certify_original_study"]
    assert all("authentic_original" not in direction["roles"] for direction in report["plan"]["directions"])
    assert all(value["meaningful_passed_geometry_directions"] >= 1
               for value in report["field_outcomes"].values())


@pytest.mark.parametrize("scope,directions", [("saved_failed_endpoint", {"X": [], "E_upper": []}),
                                               ("not_preexisting_at_new_endpoint", originals()),
                                               ("original_probe_not_reached", originals()),
                                               ("unknown", originals())])
def test_historical_scope_cannot_be_implicitly_changed(scope, directions):
    with pytest.raises(ValueError):
        admission.plan(graph(), directions, original_direction_scope=scope)


def test_geometry_pool_has_four_distinct_rows_and_includes_maximum_margin():
    values = graph()
    values["X"][2] = np.asarray([0.45, 0.45, 0.1], np.float32)
    planned = admission.plan(values, originals())
    for field in values:
        geometric = [value for value in planned["directions"] if value["field"] == field
                     and any(role.startswith("geometry_") for role in value["roles"])]
        assert len(geometric) == 4 and len({value["row"] for value in geometric}) == 4
    maximum = next(value for value in planned["directions"] if value["field"] == "X"
                   and "geometry_max_margin" in value["roles"])
    assert maximum["row"] == 2 and {maximum["source"], maximum["target"]} == {0, 1}


def test_baseline_precision_selects_and_seals_pair_before_perturbed_forwards():
    values = graph()
    planned = admission.plan(values, originals())
    primal, autodiff, frozen = evaluators(values)
    sealed = []

    def guarded(payload):
        changed = any(not np.array_equal(payload[field], values[field]) for field in values)
        assert not changed or sealed, "Perturbed forward preceded immutable eligibility seal"
        return primal(payload)

    report = admission.evaluate(planned, guarded, autodiff, frozen, seal_callback=lambda record: sealed.append(record))
    assert report["can_admit_new_child"] and len(sealed) == 1
    assert report["eligibility_seal"] == sealed[0]["eligibility_sha256"]
    phases = [value["phase"] for value in report["evaluation_sequence"]]
    assert phases[:7] == ["baseline_primal"] * 3 + ["baseline_autodiff"] * 3 + ["eligibility_sealed"]
    assert report["eligibility"]["pair_fallback_after_FD"] is False
    assert all(value["selected_local_pair"] == [0.0001, 0.00003] for value in report["eligibility"]["directions"])


@pytest.mark.parametrize("wrong", [0.0, -1.0, 1.5])
def test_zero_wrong_sign_and_wrong_magnitude_AD_never_admit(wrong):
    report = run(wrong=wrong)
    assert report["status"] == "failed" and not report["can_admit_new_child"] and report["failures"]


def test_wrong_geometry_gradient_blocks_even_when_authentic_contrasts_pass():
    values = graph()
    planned = admission.plan(values, originals())
    primal, autodiff, frozen = evaluators(values)

    def wrong_one_row(payload):
        scalar, gradients = autodiff(payload)
        gradients["E_upper"][1] *= 2
        return scalar, gradients

    report = admission.evaluate(planned, primal, wrong_one_row, frozen)
    assert not report["can_admit_new_child"]
    bad = [value for value in report["directions"] if value["field"] == "E_upper" and value["row"] == 1]
    assert bad and not bad[0]["admission_evidence"]


def test_zero_signal_geometry_extras_are_noncertifying_not_fake_passes():
    values = graph()
    coefficients = weights(values)
    for derivative in coefficients.values():
        derivative[1:] = 0

    def primal(payload):
        return np.float32(sum(np.sum(payload[field] * value) for field, value in coefficients.items()))

    def autodiff(payload):
        return primal(payload), {field: value.astype(np.float32) for field, value in coefficients.items()}

    report = admission.evaluate(admission.plan(values, originals()), primal, autodiff, lambda: {"frozen": True})
    assert report["can_admit_new_child"]
    for field, outcome in report["field_outcomes"].items():
        assert outcome["meaningful_passed_geometry_directions"] == 1
        extras = [value for value in report["directions"] if value["field"] == field and value["row"] > 0]
        assert len(extras) == 3 and all(not value["admission_evidence"] for value in extras)
        assert all(value["outcome"] == "noncertifying_precision_or_zero_tangent" for value in extras)


def test_original_contrast_alone_cannot_replace_a_meaningful_geometry_proof():
    values = {field: np.vstack([value, value[:1]]) for field, value in graph().items()}
    coefficients = weights(values)
    for derivative in coefficients.values():
        derivative[:4] = 0

    def primal(payload):
        return np.float32(sum(np.sum(payload[field] * value) for field, value in coefficients.items()))

    def autodiff(payload):
        return primal(payload), {field: value.astype(np.float32) for field, value in coefficients.items()}

    directions = {field: [{"row": 4, "source": 0, "target": 1}] for field in values}
    report = admission.evaluate(admission.plan(values, directions), primal, autodiff, lambda: {"frozen": True})
    assert not report["can_admit_new_child"]
    assert all(value["meaningful_passed_directions"] == 1 and value["meaningful_passed_geometry_directions"] == 0
               for value in report["field_outcomes"].values())


def test_FP32_constant_roundoff_cannot_be_called_a_zero_gradient_pass():
    report = run(offset=1e8)
    assert not report["can_admit_new_child"]
    assert all(not value["passed"] for value in report["field_outcomes"].values())
    assert all(value["selected_local_pair"] is None for value in report["eligibility"]["directions"])


def test_baseline_low_signal_selects_coarser_pair_before_FD_without_searching_outcomes():
    values = graph()
    primal, autodiff, frozen = evaluators(values)
    coefficients = {field: value * 0.01 for field, value in weights(values).items()}

    def low_signal(payload):
        return np.float32(4 + sum(np.sum(payload[field] * coefficient) for field, coefficient in coefficients.items()))

    def low_ad(payload):
        return low_signal(payload), {field: value.astype(np.float32) for field, value in coefficients.items()}

    report = admission.evaluate(admission.plan(values, originals()), low_signal, low_ad, frozen)
    assert report["eligibility"]["baseline_resolution_floor"] > 0
    assert report["can_admit_new_child"]
    assert all(value["selected_local_pair"] == [0.001, 0.0003] for value in report["eligibility"]["directions"])
    assert all(value["selected_local_pair"] != [0.0001, 0.00003] for value in report["eligibility"]["directions"])


def test_no_alternative_pair_after_preselected_finest_FD_fails():
    values = graph()
    original_primal, autodiff, frozen = evaluators(values)
    baseline = float(original_primal(values))

    def distorted_fine(payload):
        value = float(original_primal(payload))
        maximum = max(float(np.max(np.abs(payload[field] - values[field]))) for field in values)
        return np.float32(baseline + 3 * (value - baseline) if 0 < maximum < 0.00015 else value)

    report = admission.evaluate(admission.plan(values, originals()), distorted_fine, autodiff, frozen)
    assert not report["can_admit_new_child"]
    assert all(value["selected_local_pair"] == [0.0001, 0.00003] for value in report["directions"])
    assert any(any(probe["epsilon"] == 0.0003 and probe["agrees"] for probe in value["probes"])
               for value in report["directions"])


def test_central_curvature_at_coarse_scale_is_reported_without_relabeling_original():
    values = graph()
    linear, autodiff, frozen = evaluators(values)

    def nonlinear(payload):
        shift = float(payload["X"][0, 1]) - float(values["X"][0, 1])
        return np.float32(float(linear(payload)) + 1e6 * shift**3)

    report = admission.evaluate(admission.plan(values, originals()), nonlinear, autodiff, frozen)
    assert report["can_admit_new_child"]
    direction = next(value for value in report["directions"] if value["field"] == "X" and value["row"] == 0)
    assert direction["coarser_local_resolved_disagreements"] == [0.001]
    assert any(value["original_scale"] and not value["agrees"] for value in direction["probes"])
    assert not direction["coarse_observations_replace_original_gate"] and report["original_run_failed_preserved"]


def test_boundary_uses_predeclared_inward_points_without_clipping_original():
    values = graph(boundary=True)
    before = {field: value.copy() for field, value in values.items()}
    planned = admission.plan(values, originals())
    assert all(probe["scheme"] == "one_sided_inward" for direction in planned["directions"] for probe in direction["probes"])
    report = admission.evaluate(planned, *evaluators(values))
    assert report["can_admit_new_child"] and report["original_run_failed_preserved"]
    assert all(np.array_equal(values[field], before[field]) for field in values)
    assert all(not probe["legacy_central_interior_feasible"] for direction in report["directions"] for probe in direction["probes"])


def test_original_boundary_contrast_without_a_feasible_inward_point_blocks():
    values = graph(boundary=True)
    directions = originals()
    directions["X"] = [{"row": 0, "source": 1, "target": 2}]
    report = run(values, directions=directions)
    assert not report["can_admit_new_child"]
    original = next(value for value in report["directions"] if value["field"] == "X"
                    and "authentic_original" in value["roles"])
    assert original["selected_local_pair"] is None and not original["admission_evidence"]


def test_actual_FP32_displacement_JVP_is_used_instead_of_nominal_epsilon():
    values = graph()
    values["X"][:] = np.asarray([0.9995, 0.0005, 0], np.float32)
    coefficients = weights(values)
    coefficients["X"][:] = [10000, 10001, -2]
    base = sum(np.sum(values[field].astype(np.float64) * value) for field, value in coefficients.items())

    def primal(payload):
        return np.float64(0.01 + sum(np.sum(payload[field].astype(np.float64) * value)
                                     for field, value in coefficients.items()) - base)

    def autodiff(payload):
        return primal(payload), {field: value.copy() for field, value in coefficients.items()}

    planned = admission.plan(values, originals())
    report = admission.evaluate(planned, primal, autodiff, lambda: {"frozen": True})
    assert report["can_admit_new_child"]
    direction = next(value for value in planned["directions"] if value["field"] == "X" and value["row"] == 0)
    observed = next(value for value in report["directions"] if value["field"] == "X" and value["row"] == 0)
    probe = next(value for value in direction["probes"] if value["epsilon"] == 0.00003)
    actual = admission._decode(probe["actual_difference"], "float64")
    expected = float(np.sum(coefficients["X"] * actual, dtype=np.float64))
    measured = next(value for value in observed["probes"] if value["epsilon"] == 0.00003)
    assert measured["actual_FP32_displacement_JVP"] == expected
    assert expected != 2 * probe["epsilon"] * float(coefficients["X"][0, 1] - coefficients["X"][0, 0])


def test_perturbed_repeat_noise_blocks_the_sealed_pair_without_falling_back():
    values = graph()
    primal, autodiff, frozen = evaluators(values)
    calls = {}

    def noisy(payload):
        changed = any(not np.array_equal(payload[field], values[field]) for field in values)
        if not changed:
            return primal(payload)
        key = tuple(value.tobytes() for value in payload.values())
        index = calls.get(key, 0)
        calls[key] = index + 1
        return np.float32(primal(payload) + (-0.001, 0, 0.001)[index % 3])

    report = admission.evaluate(admission.plan(values, originals()), noisy, autodiff, frozen)
    assert not report["can_admit_new_child"]
    assert any(probe["evaluated"] and not probe["comparison_resolved"]
               for value in report["directions"] for probe in value["probes"] if probe["local_scale"])


@pytest.mark.parametrize("which", ["primal", "autodiff"])
def test_callback_cannot_mutate_original_or_probe_payload(which):
    values = graph()
    before = {field: value.copy() for field, value in values.items()}
    primal, autodiff, frozen = evaluators(values)

    def mutating(payload):
        payload["X"][0, 0] = 0
        return primal(payload) if which == "primal" else autodiff(payload)

    report = admission.evaluate(admission.plan(values, originals()), mutating if which == "primal" else primal,
                                mutating if which == "autodiff" else autodiff, frozen)
    assert not report["can_admit_new_child"] and any("mutated" in value for value in report["failures"])
    assert all(np.array_equal(values[field], before[field]) for field in values)


def test_mutating_external_eligibility_copy_cannot_change_internal_pair():
    values = graph()

    def alter_copy(record):
        record["directions"][0]["selected_local_pair"] = [0.01, 0.003]

    report = admission.evaluate(admission.plan(values, originals()), *evaluators(values), seal_callback=alter_copy)
    assert report["can_admit_new_child"]
    assert report["directions"][0]["selected_local_pair"] == [0.0001, 0.00003]


@pytest.mark.parametrize("mutation", ["epsilon", "policy", "point", "scope"])
def test_tampered_even_resealed_plan_cannot_change_fixed_policy(mutation):
    values = graph()
    planned = admission.plan(values, originals())
    if mutation == "epsilon":
        planned["directions"][0]["probes"][0]["epsilon"] = 0.0000001
    elif mutation == "policy":
        planned["policy"]["relative_tolerance"] = 1.0
    elif mutation == "point":
        planned["directions"][0]["probes"][0]["right"]["sha256"] = "0" * 64
    else:
        planned["original_direction_scope"] = "not_preexisting_at_new_endpoint"
    planned["plan_sha256"] = admission._seal({key: value for key, value in planned.items() if key != "plan_sha256"})
    with pytest.raises(ValueError):
        admission.evaluate(planned, *evaluators(values))


@pytest.mark.parametrize("bad", ["float64", "negative", "sum", "nonfinite", "few_rows", "missing_field"])
def test_invalid_original_simplex_is_never_repaired(bad):
    values = graph()
    if bad == "float64":
        values["X"] = values["X"].astype(np.float64)
    elif bad == "negative":
        values["X"][0] = [-0.1, 0.5, 0.6]
    elif bad == "sum":
        values["X"][0] = [0.2, 0.2, 0.2]
    elif bad == "nonfinite":
        values["X"][0, 0] = np.nan
    elif bad == "few_rows":
        values["X"] = values["X"][:3]
    else:
        values.pop("E_upper")
    with pytest.raises(ValueError):
        admission.plan(values, originals())


def test_independent_primal_AD_scalar_parity_remains_mandatory():
    values = graph()
    primal, autodiff, frozen = evaluators(values)

    def wrong_scalar(payload):
        scalar, gradients = autodiff(payload)
        return np.float32(scalar + 0.1), gradients

    report = admission.evaluate(admission.plan(values, originals()), primal, wrong_scalar, frozen)
    assert not report["can_admit_new_child"] and not report["baseline_checks"]["scalar_parity_passed"]


def test_AD_repeat_instability_blocks_before_perturbed_forwards():
    values = graph()
    primal, autodiff, frozen = evaluators(values)
    calls = [0]

    def unstable(payload):
        scalar, gradients = autodiff(payload)
        calls[0] += 1
        return scalar, {field: value * (1 + 0.1 * calls[0]) for field, value in gradients.items()}

    report = admission.evaluate(admission.plan(values, originals()), primal, unstable, frozen)
    assert not report["can_admit_new_child"]
    assert all(value["phase"] != "perturbed_primal" for value in report["evaluation_sequence"])


def test_frozen_model_context_change_blocks_even_with_correct_derivatives():
    values = graph()
    primal, autodiff, _ = evaluators(values)
    calls = [0]

    def identity():
        calls[0] += 1
        return {"model": calls[0]}

    report = admission.evaluate(admission.plan(values, originals()), primal, autodiff, identity)
    assert not report["can_admit_new_child"] and not report["frozen_unchanged"]


def test_baseline_nonfinite_scalar_is_a_failure_not_a_diagnostic_pass():
    values = graph()
    _, autodiff, frozen = evaluators(values)
    report = admission.evaluate(admission.plan(values, originals()), lambda payload: np.float32(np.nan), autodiff, frozen)
    assert not report["can_admit_new_child"] and report["status"] == "failed"


def test_all_numerical_entrypoints_require_allocation(monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(RuntimeError):
        admission.plan(graph(), originals())
