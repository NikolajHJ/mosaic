"""HPC-only receipt regressions; synthetic gates do not establish native readiness."""

from copy import deepcopy

import pytest

from latentmol.boltz.additive_contract import profile_record, verify_forward_ad, verify_item
from latentmol.repair_guidance.integration_smoke import full_gate_passed
from test_additive_branch_fd import fixture, reader_gate, run_case


@pytest.fixture(scope="module", params=[False, True], ids=["control", "guided"])
def authenticated_receipt(request):
    # Reuse two allocated tensor fixtures, then mutate only deep-copied JSON
    # metadata in all regressions. Tests never alter a real experiment record.
    with pytest.MonkeyPatch.context() as patch:
        values, report = fixture(patch, repair=request.param), {}
        run_case(values, report)
        expected = reader_gate(values, report)
    return report, expected, request.param


def test_complete_receipt_needs_an_independently_bound_configuration(authenticated_receipt):
    gate, expected, repair = authenticated_receipt
    assert full_gate_passed(gate, require_repair=repair, expected_profile=expected)
    assert not full_gate_passed(gate, require_repair=repair)
    other = deepcopy(expected)
    other["dbmol"]["affinity_logit_weight"] *= 2
    assert not full_gate_passed(gate, require_repair=repair, expected_profile=other)


@pytest.mark.parametrize("mutation", ["baseline", "jvp", "proposal", "old_fd", "old_direction_pass",
    "held_proof", "latent_shape", "measured_loss", "measured_component", "native_parity",
    "grouped_parity", "profile", "arithmetic", "missing_mode", "natural_certificate", "missing_relu"])
def test_stale_or_transplanted_evidence_cannot_approve_a_full_gate(authenticated_receipt, mutation):
    original, expected, repair = authenticated_receipt
    gate = deepcopy(original)
    item = gate["component_gradients"]["dbmol_contact_loss"]
    if mutation == "baseline": item["baseline_loss"] += 1.
    if mutation == "jvp": item["independent_forward_ad"]["rows"][0]["tangent"] += 1.
    if mutation == "proposal": item["direction_selection"]["proposals"][0]["accepted"] = "true"
    if mutation == "old_fd": item["original_net_fd"]["held"]["finite_differences"][0]["f_plus"] += 1.
    if mutation == "old_direction_pass": item["original_net_fd"]["held"]["direction_results"]["gradient"]["passed"] = False
    if mutation == "held_proof":
        item["additive_inputs"]["held"][0]["recomposition"]["rows"]["reverse"]["native"]["sha256"] = "f" * 64
    if mutation == "latent_shape": gate["derivative_contract"]["latent_shape"] = [1, 256]
    if mutation == "measured_loss": gate["measured"]["loss"] += 1.
    if mutation == "measured_component": gate["measured"]["components"]["dbmol_contact_loss"] += 1.
    if mutation == "native_parity": gate["local_branch_baseline_parity"]["supplied_native_gradient_exact"]["combined"] = False
    if mutation == "grouped_parity": gate["local_branch_baseline_parity"]["native_grouped_scalar_exact"]["combined"] = False
    if mutation == "profile": gate["derivative_contract"]["profile"]["dbmol"]["profile"] = "unknown"
    if mutation == "arithmetic": gate["derivative_contract"]["profile"]["dbmol"]["objective_arithmetic"] = "native_v1"
    if mutation == "missing_mode": item["additive_inputs"].pop("natural")
    if mutation == "natural_certificate": item["additive_certificates"]["natural"]["rows"][0]["passed"] = False
    if mutation == "missing_relu": gate["local_branch_baseline_parity"]["captured_readout_relus"].popitem()
    assert not full_gate_passed(gate, require_repair=repair, expected_profile=expected)


@pytest.mark.parametrize("bad", [None, [], 1, "passed", {"derivative_contract": []},
    {"derivative_contract": {"version": "additive_branch_v3", "profile": []}},
    {"component_gradients": []}, {"independent_forward_ad": {"rows": [None]}}])
def test_malformed_receipts_return_false_instead_of_throwing(bad):
    assert full_gate_passed(bad) is False
    assert verify_forward_ad(bad) is False
    assert verify_item(bad, "held", {}, "combined") is False


def test_repair_pass_requires_actual_unchanged_scalar_grid(authenticated_receipt):
    original, expected, repair = authenticated_receipt
    if not repair:
        return
    for mutation in ("row", "policy", "baseline", "weight", "missing_grid"):
        gate = deepcopy(original)
        item = gate["repair_only_validation"]
        if mutation == "row": item["local_branch_fd"]["finite_differences"][0]["passed"] = False
        if mutation == "policy": item["original_repair_policy"]["version"] = "local_v1"
        if mutation == "baseline": item["baseline_loss"] += 1.
        if mutation == "weight": gate["measured"]["repair_guidance"]["weight"] *= 2
        if mutation == "missing_grid": item["local_branch_fd"]["finite_differences"].pop()
        assert not full_gate_passed(gate, require_repair=True, expected_profile=expected)


@pytest.mark.parametrize("field,value", [("profile", "unknown"), ("objective_arithmetic", "native_v1")])
def test_profile_record_rejects_an_unsupported_model_arithmetic(authenticated_receipt, field, value):
    _, expected, _ = authenticated_receipt
    dbmol = deepcopy(expected["dbmol"])
    dbmol[field] = value
    with pytest.raises(ValueError):
        profile_record(dbmol, expected["objective"], repair_weight=expected["repair_weight"])
