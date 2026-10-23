"""Tiny receipt fixtures: native guidance must not be confused with fallback."""

import json

from latentmol.experiments import write_json
from latentmol.repair_guidance.study import guidance_evidence


def test_evidence_reader_uses_actual_provider_receipts_and_separates_rejections(tmp_path):
    folder = tmp_path / "fixed_initial_mask"
    folder.mkdir()
    repair = folder / "repair-refresh-00000"
    repair.mkdir()
    write_json(repair / "refresh.json", {"attempts": [
        {"status": "accepted", "response": {"status": "accepted", "cost": {"native_forward_calls": 447}}},
        {"status": "rejected", "response": {"reason": "input_representation_unsupported", "cost": {"native_forward_calls": 0}}},
        {"status": "rejected", "response": {"reason": "output_chemical_invalid", "cost": {"native_forward_calls": 447}}}]})
    write_json(folder / "gate-0000.json", {"repair_only_validation": {"status": "validated_nonzero", "passed": True}})
    write_json(folder / "metrics-0000.json", {"update_status": "accepted", "score": {"repair_guidance": {
        "gradient_diagnostics": {"weighted_repair_norm": .2}}}})
    summary = {"arms": {"fixed_initial_mask": {"repair_guidance": {
        "refreshes": 1, "accepted_targets": 1, "guided_accepted_updates": 1}}}}
    result = guidance_evidence(tmp_path, summary, weight=.01)
    assert result["repair_validation_passed"]
    assert result["native_attempts"] == 2 and result["accepted_native_targets"] == 1
    assert result["unsupported_candidates"] == 1 and result["chemical_rejections"] == 1
    assert result["target_only_fd_passes"] == result["target_only_derivative_passes"] == 1
    assert result["target_only_pointwise_ad_passes"] == 0
    assert not guidance_evidence(tmp_path, summary, weight=0)["repair_validation_passed"]


def test_successful_all_fallback_has_no_guidance_validation(tmp_path):
    folder = tmp_path / "fixed_initial_mask"
    folder.mkdir()
    summary = {"arms": {"fixed_initial_mask": {"completed_requested_budget": True,
        "repair_guidance": {"refreshes": 2, "accepted_targets": 0, "guided_accepted_updates": 0}}}}
    assert not guidance_evidence(tmp_path, summary, weight=.01)["repair_validation_passed"]


def test_pointwise_target_checks_are_not_reported_as_numerical_fd(tmp_path):
    folder = tmp_path / "fixed_initial_mask"
    folder.mkdir()
    (folder / "repair-refresh-00000").mkdir()
    write_json(folder / "repair-refresh-00000/refresh.json", {"attempts": [
        {"status": "accepted", "response": {"status": "accepted", "cost": {"native_forward_calls": 447}}}]})
    for index, passed in enumerate((True, False)):
        write_json(folder / f"gate-{index:04d}.json", {
            "derivative_contract": {"version": "pointwise_native_ad_v4"},
            "repair_only_validation": {"status": "validated_nonzero", "passed": passed}})
    write_json(folder / "metrics-0000.json", {"update_status": "accepted", "score": {"repair_guidance": {
        "gradient_diagnostics": {"weighted_repair_norm": .2}}}})
    summary = {"arms": {"fixed_initial_mask": {"repair_guidance": {
        "refreshes": 1, "accepted_targets": 1, "guided_accepted_updates": 1}}}}
    result = guidance_evidence(tmp_path, summary, weight=.01)
    assert result["repair_validation_passed"] is True
    assert result["target_only_fd_passes"] == 0
    assert result["target_only_pointwise_ad_passes"] == result["target_only_derivative_passes"] == 1
    # Unknown future contracts must not silently inherit the legacy FD label.
    write_json(folder / "gate-0000.json", {"derivative_contract": {"version": "unknown"},
        "repair_only_validation": {"status": "validated_nonzero", "passed": True}})
    result = guidance_evidence(tmp_path, summary, weight=.01)
    assert result["repair_validation_passed"] is False
    assert result["target_only_fd_passes"] == result["target_only_pointwise_ad_passes"] == result["target_only_derivative_passes"] == 0
