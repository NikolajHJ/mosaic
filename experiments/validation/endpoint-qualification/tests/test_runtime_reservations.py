"""Portable JSON and runtime-reservation unit controls; no native evaluation.

Only the standard-library driver import and resource-planning arithmetic execute.
These controls need no scheduler environment, models, scientific arrays, GPU or
LSB_JOBID. They do not establish native readiness or authorize an experiment.
Allocated native/startup/provenance integration fixtures remain guarded in the
separate test_precision_continuation_driver module.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/precision_continuation_driver.py"
SPEC = importlib.util.spec_from_file_location("precision_runtime_reservations_fixture", PATH)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def pro_runtime(projected):
    return {"status": "passed", "mean_update_seconds": [(projected - 1800) / 15000],
            "conservative_projected_seconds": projected}


@pytest.mark.parametrize("projection", [0, False, float("inf"), float("nan"), 82800, 90000])
def test_reference_runtime_is_reserved_without_reducing_original_guard(tmp_path, projection):
    path = tmp_path / "summary.json"
    # Nonfinite test documents are deliberately malformed input metadata.
    runtime = pro_runtime(projection) if type(projection) in (int, float) else {"status": "passed", "conservative_projected_seconds": projection}
    path.write_text(json.dumps({"runtime_gate": runtime, "elapsed_seconds": 600}), encoding="utf8")
    with pytest.raises(ValueError): d.runtime_budget(path, "protenix")


def test_runtime_budget_charges_measured_complete_endpoint_cohorts_without_reserving_timeout_ceilings(tmp_path):
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": pro_runtime(10001), "elapsed_seconds": 600})
    value = d.runtime_budget(path, "protenix")
    assert value["original_projected_seconds"] == 10001
    assert value["bounded_segment_projected_seconds"] == pytest.approx(27603)
    assert value["required_total_seconds"] == pytest.approx(28203)
    assert value["walltime"] == "08:15" and value["walltime_seconds"] == 29700
    assert value["limit_seconds"] == 29100 and value["original_runtime_guard_changed"] is False
    assert value["fixed_charged_seconds"] == 3000
    assert value["reference_timeout_ceiling_seconds_each"] == 5400
    assert value["reference_timeout_ceilings_reserved_as_elapsed"] is False
    assert value["requested_updates"] == value["maximum_safe_updates"] == 30000
    assert value["observed_validation_elapsed_seconds"] == 600 and value["automatic_shortening_or_fallback"] is False


def test_dbmol_runtime_keeps_all_measured_fixed_overhead(tmp_path):
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": {"status": "passed", "mean_update_seconds": 2,
        "conservative_projected_seconds": 10600, "segment_updates": 2500,
        "observed_startup_entry_check_seconds": 1000, "observed_native_check_seconds": 200,
        "terminal_check_safety_factor": 1.5, "update_safety_factor": 1.5, "reserve_seconds": 1800}, "elapsed_seconds": 600})
    value = d.runtime_budget(path, "dbmol")
    assert value["bounded_segment_projected_seconds"] == 10600
    assert value["required_total_seconds"] == 11200
    assert value["walltime"] == "03:30" and value["limit_seconds"] == 12000


def test_dbmol_runtime_uses_larger_complete_validation_bound_without_weakening_original_projection(tmp_path):
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": {"status": "passed", "mean_update_seconds": 2,
        "conservative_projected_seconds": 10600, "segment_updates": 2500,
        "observed_startup_entry_check_seconds": 1000, "observed_native_check_seconds": 200,
        "terminal_check_safety_factor": 1.5, "update_safety_factor": 1.5, "reserve_seconds": 1800},
        "elapsed_seconds": 5000})
    value = d.runtime_budget(path, "dbmol")
    assert value["original_projected_seconds"] == 10600
    assert value["fixed_original_seconds"] == 3100 and value["fixed_charged_seconds"] == 11800
    assert value["bounded_segment_projected_seconds"] == 19300 and value["required_total_seconds"] == 19900
    assert value["walltime"] == "05:45" and value["limit_seconds"] == 20100


def test_full_protenix_remaining_budget_failure_reports_largest_measured_fit_without_executing_fallback(tmp_path):
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": pro_runtime(50000), "elapsed_seconds": 600})
    with pytest.raises(d.RuntimeBudgetExceeded, match="maximum_safe_updates=16307") as error:
        d.runtime_budget(path, "protenix")
    value = error.value.budget
    assert value["status"] == "failed" and value["requested_updates"] == 30000
    assert value["maximum_safe_updates"] == 16307 and value["automatic_shortening_or_fallback"] is False
    assert value["conservative_seconds_per_update"] * value["maximum_safe_updates"] + value["fixed_charged_seconds"] + value["orchestration_seconds"] <= 82200
    assert value["conservative_seconds_per_update"] * (value["maximum_safe_updates"] + 1) + value["fixed_charged_seconds"] + value["orchestration_seconds"] > 82200
    assert value["walltime_seconds"] > 82800 and "not a worst-case guarantee" in value["measurement_scope"]


@pytest.mark.parametrize("elapsed,error", [(None, "duration missing"), (0, "duration missing"),
                                            (-1, "duration missing"), (True, "duration missing"),
                                            (float("inf"), "Nonfinite JSON constant: Infinity")])
def test_missing_or_nonfinite_complete_validation_duration_cannot_size_a_remaining_run(tmp_path, elapsed, error):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"runtime_gate": pro_runtime(10001), "elapsed_seconds": elapsed}), encoding="utf8")
    with pytest.raises(ValueError, match=error):
        d.runtime_budget(path, "protenix")


def test_dbmol_segment_still_rejects_two_reference_checks_beyond_maximum_walltime(tmp_path):
    path = tmp_path / "summary.json"
    # The inherited 23-hour check passes; the complete two-reference budget does
    # not. Shorter production intervals cannot scale away this measured cost.
    put(path, {"runtime_gate": {"status": "passed", "mean_update_seconds": 2,
        "conservative_projected_seconds": 82000, "segment_updates": 2500,
        "observed_startup_entry_check_seconds": 72400, "observed_native_check_seconds": 200,
        "terminal_check_safety_factor": 1.5, "update_safety_factor": 1.5, "reserve_seconds": 1800}, "elapsed_seconds": 600})
    with pytest.raises(d.RuntimeBudgetExceeded) as error: d.runtime_budget(path, "dbmol")
    assert error.value.budget["requested_updates"] == 2500 and error.value.budget["maximum_safe_updates"] == 2366
    assert error.value.budget["automatic_shortening_or_fallback"] is False


@pytest.mark.parametrize("change", ["status", "missing_means", "different_formula", "nonfinite_mean", "unknown_arm"])
def test_segment_walltime_never_comes_from_unpassed_or_incomplete_measurements(tmp_path, change):
    runtime = pro_runtime(50000)
    if change == "status": runtime["status"] = "failed"
    elif change == "missing_means": del runtime["mean_update_seconds"]
    elif change == "different_formula": runtime["mean_update_seconds"] = [1]
    elif change == "nonfinite_mean": runtime["mean_update_seconds"] = [float("inf")]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"runtime_gate": runtime, "elapsed_seconds": 600}), encoding="utf8")
    with pytest.raises(ValueError): d.runtime_budget(path, "missing" if change == "unknown_arm" else "protenix")


@pytest.mark.parametrize("field", ["segment_updates", "terminal_check_safety_factor", "update_safety_factor", "reserve_seconds",
                                   "observed_startup_entry_check_seconds", "observed_native_check_seconds", "conservative_projected_seconds"])
def test_dbmol_runtime_rejects_incomplete_or_rescaled_fixed_terms(tmp_path, field):
    runtime = {"status": "passed", "mean_update_seconds": 2, "conservative_projected_seconds": 10600,
               "segment_updates": 2500, "observed_startup_entry_check_seconds": 1000,
               "observed_native_check_seconds": 200, "terminal_check_safety_factor": 1.5,
               "update_safety_factor": 1.5, "reserve_seconds": 1800}
    runtime[field] = 0
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": runtime, "elapsed_seconds": 600})
    with pytest.raises(ValueError): d.runtime_budget(path, "dbmol")


