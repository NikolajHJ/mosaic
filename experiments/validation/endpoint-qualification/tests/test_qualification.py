"""Meaningful numerical toy controls; run only on allocated DTU HPC CPU."""
import base64
import copy
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[1] / "scripts/qualification.py"
SPEC = importlib.util.spec_from_file_location("endpoint_qualification", MODULE)
q = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = q
SPEC.loader.exec_module(q)


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get("LSB_JOBID"), "Numerical fixtures require a real allocated HPC job"


def graph(a=.4):
    return {"X": np.array([[a, 1 - a]], dtype=np.float32)}


def direction():
    return {"X": [{"row": 0, "source": 0, "target": 1}]}


def identity():
    return {"model": "frozen-model", "context": "fixed-context", "key": "same-key"}


def smooth(values):
    x = values["X"].astype(np.float64)
    return np.float64(x[0, 0] ** 2 + 2 * x[0, 1] ** 2)


def smooth_ad(values):
    x = values["X"]
    return smooth(values), {"X": np.array([[2 * x[0, 0], 4 * x[0, 1]]], dtype=np.float32)}


def test_smooth_independent_derivative_passes_without_certifying_original():
    plan = q.predeclare(graph(), direction())
    result = q.evaluate_plan(plan, smooth, smooth_ad, identity)
    assert result["outcome"] == "passed"
    assert result["directions"][0]["original_adjacent_agreement"]
    assert result["original_outcome"] == "failed_preserved"
    assert result["can_certify_or_restart_original_study"] is False
    assert result["frozen_unchanged"]


def test_true_wrong_gradient_is_not_hidden_by_supplementary_scales():
    def wrong(values):
        return smooth(values), {"X": -smooth_ad(values)[1]["X"]}
    result = q.evaluate_plan(q.predeclare(graph(), direction()), smooth, wrong, identity)
    assert result["outcome"] == "numerical_disagreement"
    assert not result["directions"][0]["supplementary_adjacent_agreement"]


def test_near_boundary_geometry_is_inconclusive_even_if_fine_scales_agree():
    values = graph(.00005)
    plan = q.predeclare(values, direction())
    result = q.evaluate_plan(plan, smooth, smooth_ad, identity)
    row = result["directions"][0]
    assert result["outcome"] == "boundary_inconclusive"
    assert row["supplementary_adjacent_agreement"]
    assert all(not p["evaluated"] for p in row["probes"] if p["kind"] == "original")


def test_zero_boundary_does_not_invent_a_symmetric_probe():
    plan = q.predeclare(graph(0), direction())
    result = q.evaluate_plan(plan, smooth, smooth_ad, identity)
    assert result["outcome"] == "boundary_inconclusive"
    assert all(not p["evaluated"] for p in result["directions"][0]["probes"])


def test_fp32_scalar_cancellation_is_unresolved_not_passed():
    def quantized(values):
        return np.float32(np.float32(1e8) + values["X"][0, 1])
    def ad(values):
        return quantized(values), {"X": np.array([[0, 1]], dtype=np.float32)}
    result = q.evaluate_plan(q.predeclare(graph(), direction()), quantized, ad, identity)
    assert result["outcome"] == "unresolved_precision"
    assert all(not p["resolved"] for p in result["directions"][0]["probes"])


def test_curvature_scale_dependence_preserves_original_disagreement():
    def curved(values):
        z = np.float64(values["X"][0, 1]) - .5
        return np.float64(z + 1e6 * z ** 3)
    def ad(values):
        z = np.float64(values["X"][0, 1]) - .5
        return curved(values), {"X": np.array([[0, 1 + 3e6 * z ** 2]], dtype=np.float32)}
    result = q.evaluate_plan(q.predeclare(graph(.5), direction()), curved, ad, identity)
    row = result["directions"][0]
    assert result["outcome"] == "numerical_disagreement"
    assert row["scale_dependence"] and row["supplementary_adjacent_agreement"]


def test_nonsmooth_point_records_opposed_one_sided_slopes():
    def cusp(values):
        return np.float64(abs(np.float64(values["X"][0, 1]) - .5))
    def ad(values):
        return cusp(values), {"X": np.array([[0, 1]], dtype=np.float32)}
    result = q.evaluate_plan(q.predeclare(graph(.5), direction()), cusp, ad, identity)
    first = result["directions"][0]["probes"][0]
    assert first["plus_one_sided"] > .9 and first["minus_one_sided"] < -.9
    assert result["outcome"] != "passed"


def test_geometry_plan_is_identical_before_different_evaluators():
    plan = q.predeclare(graph(), direction())
    saved = copy.deepcopy(plan)
    q.evaluate_plan(plan, smooth, smooth_ad, identity)
    assert plan == saved
    assert [p["epsilon"] for p in plan["directions"][0]["probes"]] == [
        .01, .003, .001, .0003, .00015, .000075]


def test_plan_records_exact_arrays_and_actual_fp32_displacements():
    values = graph()
    plan = q.predeclare(values, direction())
    for probe in plan["directions"][0]["probes"]:
        for point in probe["points"]:
            raw = q._decode(point["point"])
            delta = q._decode(point["delta"])
            assert np.array_equal(raw - values["X"], delta)
            assert len(base64.b64decode(point["point"]["base64"])) == raw.nbytes


def test_mutated_plan_is_rejected_before_any_native_call():
    plan = q.predeclare(graph(), direction())
    plan["directions"][0]["probes"][0]["epsilon"] = .000001
    def forbidden(*args):
        raise AssertionError("must not evaluate")
    with pytest.raises(ValueError, match="identity differs"):
        q.evaluate_plan(plan, forbidden, forbidden, forbidden)


def test_frozen_identity_change_prevents_a_diagnostic_pass():
    calls = []
    def changing():
        calls.append(1)
        return {"model": str(len(calls))}
    result = q.evaluate_plan(q.predeclare(graph(), direction()), smooth, smooth_ad, changing)
    assert result["outcome"] == "numerical_disagreement"
    assert result["frozen_unchanged"] is False


def test_independent_ad_primal_scalar_parity_is_required():
    def shifted(values):
        scalar, gradient = smooth_ad(values)
        return scalar + .1, gradient
    result = q.evaluate_plan(q.predeclare(graph(), direction()), smooth, shifted, identity)
    assert result["outcome"] == "numerical_disagreement"
    assert any("parity" in text for text in result["failures"])


def test_optional_logit_control_is_separate_and_cannot_override_failure():
    result = q.evaluate_plan(q.predeclare(graph(.00005), direction()), smooth,
                             smooth_ad, identity, logit_control=lambda: {"status": "passed"})
    assert result["outcome"] == "boundary_inconclusive"
    assert result["logit_control"]["raw_baseline_equivalence_assumed"] is False
    assert result["logit_control"]["can_replace_raw_outcome"] is False


@pytest.mark.parametrize("change", [
    {"original_epsilons": (.001, .0001)}, {"geometry_fractions": (.1, .01, .001)},
    {"repeats": 1}, {"maximum_supplementary_epsilon": .00001},
])
def test_unversioned_policy_changes_are_rejected(change):
    with pytest.raises(ValueError):
        q.predeclare(graph(), direction(), policy=q.Policy(**change))


def test_callbacks_cannot_modify_the_callers_saved_input_array():
    values = graph()
    original = values["X"].copy()
    plan = q.predeclare(values, direction())
    q.evaluate_plan(plan, smooth, smooth_ad, identity)
    assert np.array_equal(values["X"], original)


def test_diagnostic_cannot_relabel_original_failed_outcome():
    with pytest.raises(ValueError, match="promote or relabel"):
        q.evaluate_plan(q.predeclare(graph(), direction()), smooth, smooth_ad,
                        identity, original_outcome="passed")
