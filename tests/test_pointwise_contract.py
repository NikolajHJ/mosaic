"""Allocated receipt fixtures; fabricated metadata never certifies a native model."""

from copy import deepcopy
import hashlib
import math
import struct

import pytest

from latentmol.boltz.pointwise_contract import (BRIDGE_POLICY, COMPONENTS, NATIVE_CHANNELS,
    POLICY, profile_record, verify_bridge, verify_gate)
from latentmol.boltz.additive_fd import DIRECTIONS, PARITY_VERSION


def digest(name):
    return hashlib.sha256(name.encode()).hexdigest()


def tensor(dtype, shape, values, *, summary=False):
    raw = struct.pack("=" + ("f" if dtype == "torch.float32" else "d") * len(values), *values)
    result = {"shape": shape, "dtype": dtype, "sha256": hashlib.sha256(raw).hexdigest()}
    if summary:
        result.update(finite=True, max_abs=max(abs(value) for value in values))
    elif len(values) == 1:
        result["value"] = values[0]
    return result


def proof(baseline, tangent, gradient, *, repair=False):
    dtype = "torch.float32" if repair else "torch.float64"
    values = {"scalar": tensor(dtype, [], [baseline]), "jvp": tensor(dtype, [], [tangent]),
              "reverse": tensor("torch.float32", [1, 4], [gradient, 0., 0., 0.])}
    return {"version": PARITY_VERSION, "passed": True,
        "rows": {key: {"exact_equal": True, "native": value, "recomposed": deepcopy(value)}
                 for key, value in values.items()}}


def item(baseline, gradient, *, repair=False):
    projections = (gradient, gradient / 2, gradient * .75, gradient * .875)
    bindings = {name: digest("native-direction-" + name) for name in DIRECTIONS}
    threshold = .5 * gradient / math.sqrt(4)
    proposals = [{"proposal_index": index, "generator_seed": 1729,
        "direction_sha256": bindings[label], "analytic_projection_float64": projections[index + 1],
        "minimum_absolute_projection": threshold, "accepted": True, "selected_label": label,
        "rejection_reason": None} for index, label in enumerate(DIRECTIONS[1:])]
    return {"validation_contract": POLICY, "passed": True, "status": "validated_nonzero",
        "gradient_norm": gradient, "baseline_loss": baseline, "attached": True, "fixed_pose_and_mask": True,
        "direction_bindings": bindings,
        "direction_selection": {"selection_completed_before_endpoint_scores": True, "pool_exhausted": False,
            "selected_labels": list(DIRECTIONS), "latent_numel": 4, "gradient_norm": gradient,
            "minimum_absolute_random_projection": threshold, "proposals": proposals,
            "legacy_direction_sha256": bindings["random_1"]},
        "independent_forward_ad": {"rtol": .001, "atol": 1e-6, "passed": True,
            "rows": [{"direction": label, "direction_sha256": bindings[label], "primal": baseline, "tangent": projection,
                "reverse_projection": projection, "absolute_error": 0.,
                "tolerance": 1e-6 + .001 * abs(projection), "primal_exact": True, "passed": True}
                for label, projection in zip(DIRECTIONS, projections)]},
        "recomposition_proofs": {label: proof(baseline, projection, gradient, repair=repair)
                                  for label, projection in zip(DIRECTIONS, projections)}}


def comparison_group(channels, *, native=False):
    rtol, atol = (1e-4, 1e-5) if native else (1e-8, 1e-10)
    return {"passed": True, "rtol": rtol, "atol": atol,
        "rows": {name: {"actual": tensor("torch.float32" if native else "torch.float64", [1, 2], [1., 0.], summary=True),
                        "expected": tensor("torch.float64", [1, 2], [1., 0.], summary=True),
                        "max_abs_error": 0., "tolerance_max": atol + rtol,
                        "maximum_normalized_error": 0., "passed": True} for name in channels}}


def bridge():
    result = {"version": BRIDGE_POLICY, "passed": True, "graph_sha256": digest("shared-graph"),
        "context_sha256": digest("fixed-context"), "auxiliary_unchanged": True, "context_unchanged": True,
        "affine_extension_not_molecule_samples": True, "scorer_calls": 0,
        "native_mapping_parity": comparison_group(NATIVE_CHANNELS, native=True),
        "float64_value_parity": comparison_group(NATIVE_CHANNELS), "directions": []}
    for name in ("affine_1", "affine_2"):
        row = {"name": name, "direction_sha256": digest(name), "passed": True,
            "jvp": comparison_group(NATIVE_CHANNELS), "vjp": comparison_group(("atoms", "charges", "bonds")),
            "finite_differences": []}
        for epsilon in (1e-4, .001):
            fd = {**comparison_group(NATIVE_CHANNELS), "epsilon": epsilon}
            for channel, field in fd["rows"].items():
                field.update(analytic_sha256=row["jvp"]["rows"][channel]["actual"]["sha256"],
                    endpoint_plus=tensor("torch.float64", [1, 2], [1 + epsilon, 0.], summary=True),
                    endpoint_minus=tensor("torch.float64", [1, 2], [1 - epsilon, 0.], summary=True))
            row["finite_differences"].append(fd)
        result["directions"].append(row)
    return result


def fixture(*, repair=False):
    dbmol = {"profile": "dbmol_fixed_pose_v1", "objective_arithmetic": "contact_head_and_assembly_float64_v1",
        "affinity_regression_weight": 1., "affinity_logit_weight": 1., "contact_weight": 2.5, "confidence_weight": 2.5,
        "confidence_weights": {"iptm": 1., "complex_iplddt": .3, "complex_ipde": .3,
                               "complex_pde": .1, "confidence_score": .3, "ptm": .1}}
    expected = profile_record(dbmol, {"output": "dbmol_total_loss", "direction": "minimize", "scale": 1., "offset": 0.},
                              repair_weight=.25 if repair else None)
    names = set(COMPONENTS) | {"combined"}
    grouped = names | ({"repair_only"} if repair else set())
    components = {name: item(1., 1.) for name in COMPONENTS}
    total = 6.5 if repair else 6.
    contract = {"version": POLICY, "passed": True, "profile": expected, "latent_shape": [1, 4],
        "scope": "pointwise_native_AD_consistency", "point_sha256": digest("point"),
        "context_sha256": digest("fixed-context"), "graph_sha256": digest("shared-graph"),
        "target_sha256": digest("held-target") if repair else None,
        "full_network_fd_required": False, "historical_gate_approval": False, "branch_holding_used_in_optimization": False,
        "native_subgradient_conventions_retained": True}
    gate = {"passed": True, "status": "passed", "native_seed_parity": True,
        "point_sha256": contract["point_sha256"], "context_sha256": contract["context_sha256"],
        "derivative_contract": contract, "component_gradients": components,
        "actual_pointwise_full_forwards": 14, "independent_forward_ad_direction_count": 4,
        "pointwise_state_unchanged": {"frozen_versions": True, "context": True, "target": True},
        "local_fd_validation": item(total, total),
        "measured": {"loss": total, "components": {name: 1. for name in COMPONENTS}},
        "native_input_bridge_audit": bridge(),
        "native_grouped_baseline_parity": {"passed": True, "scalar_exact": True,
            "supplied_gradient_exact": {key: True for key in names},
            "native_grouped_gradient_exact": {key: True for key in grouped},
            "native_grouped_scalar_exact": {key: True for key in grouped},
            "repeated_native_scalar_exact": True, "repeated_shared_graph_exact": True, "target_unchanged": True}}
    if repair:
        gate["repair_only_validation"] = item(2., 2., repair=True)
        gate["measured"]["repair_guidance"] = {"status": "active", "weight": .25,
            "normalized_nll": 2., "target_sha256": contract["target_sha256"]}
    bindings = {"combined_gradient": gate["local_fd_validation"]["direction_bindings"]["gradient"],
                "contact_random_2": components["dbmol_contact_loss"]["direction_bindings"]["random_2"]}
    diagnostics = {"required_for_pass": False, "epsilons": [1e-4, .001], "directions": bindings, "rows": []}
    for name in bindings:
        for epsilon in (1e-4, .001):
            tangents = {**{key: (1. if name == "combined_gradient" else .75) for key in COMPONENTS},
                        "combined": total if name == "combined_gradient" else .75 * total}
            # Deliberately huge finite-step discrepancies must remain diagnostic.
            plus = {key: 100. for key in tangents}
            minus = {key: -100. for key in tangents}
            diagnostics["rows"].append({"direction": name, "direction_sha256": bindings[name], "epsilon": epsilon,
                "required_for_pass": False, "status": "completed", "f_plus": plus, "f_minus": minus,
                "tangents": tangents, "secants": {key: (plus[key] - minus[key]) / (2 * epsilon) for key in plus},
                "plus_graph_sha256": digest("plus"), "minus_graph_sha256": digest("minus")})
    gate["natural_secant_diagnostics"] = diagnostics
    return gate, expected


@pytest.mark.parametrize("repair", [False, True])
def test_explicit_pointwise_policy_needs_native_proofs_and_external_profile_not_full_network_fd(repair):
    from latentmol.repair_guidance.integration_smoke import full_gate_passed
    gate, expected = fixture(repair=repair)
    assert verify_gate(gate, require_repair=repair, expected_profile=expected)
    assert full_gate_passed(gate, require_repair=repair, expected_profile=expected)
    assert not verify_gate(gate, require_repair=repair)
    other = deepcopy(expected); other["dbmol"]["contact_weight"] = 3.
    assert not verify_gate(gate, require_repair=repair, expected_profile=other)


@pytest.mark.parametrize("mutation", ["ad", "ad_primal", "direction", "gradient_shape", "recomposition",
    "point", "context", "graph", "native_parity", "grouped_parity", "bridge", "fd_coverage", "fd_hash", "fd_arithmetic",
    "state", "count", "budget", "attached", "subgradient"])
def test_stale_local_derivative_identity_or_bridge_cannot_pass(mutation):
    gate, expected = fixture()
    item = gate["component_gradients"]["dbmol_contact_loss"]
    if mutation == "ad": item["independent_forward_ad"]["rows"][0]["tangent"] += .1
    if mutation == "ad_primal": item["independent_forward_ad"]["rows"][0]["primal"] += .1
    if mutation == "direction": item["direction_bindings"]["random_2"] = digest("other")
    if mutation == "gradient_shape": item["recomposition_proofs"]["gradient"]["rows"]["reverse"]["native"]["shape"] = [4]
    if mutation == "recomposition": item["recomposition_proofs"]["gradient"]["rows"]["jvp"]["native"]["value"] += 1.
    if mutation == "point": gate["point_sha256"] = digest("wrong")
    if mutation == "context": gate["context_sha256"] = digest("wrong")
    if mutation == "graph": gate["native_input_bridge_audit"]["graph_sha256"] = digest("wrong")
    if mutation == "native_parity": gate["native_grouped_baseline_parity"]["supplied_gradient_exact"]["combined"] = False
    if mutation == "grouped_parity": gate["native_grouped_baseline_parity"]["native_grouped_scalar_exact"]["combined"] = False
    if mutation == "bridge": gate["native_input_bridge_audit"]["auxiliary_unchanged"] = False
    if mutation == "fd_coverage": gate["natural_secant_diagnostics"]["rows"].pop()
    if mutation == "fd_hash": gate["natural_secant_diagnostics"]["rows"][0]["direction_sha256"] = digest("wrong")
    if mutation == "fd_arithmetic": gate["natural_secant_diagnostics"]["rows"][0]["secants"]["combined"] += 1.
    if mutation == "state": gate["pointwise_state_unchanged"]["target"] = False
    if mutation == "count": gate["independent_forward_ad_direction_count"] = 8
    if mutation == "budget": gate["actual_pointwise_full_forwards"] = 33
    if mutation == "attached": item["attached"] = False
    if mutation == "subgradient": gate["derivative_contract"]["native_subgradient_conventions_retained"] = False
    assert not verify_gate(gate, require_repair=False, expected_profile=expected)


@pytest.mark.parametrize("mutation", ["missing_channel", "finite", "normalized_error", "tolerance", "dtype",
    "vjp_channel", "reused_direction", "fd_scale", "fd_jvp", "fd_endpoint", "scorer_call"])
def test_bridge_rejects_partial_or_stale_affine_evidence(mutation):
    audit = bridge()
    if mutation == "missing_channel": audit["native_mapping_parity"]["rows"].pop("ref_charge")
    if mutation == "finite": audit["float64_value_parity"]["rows"]["bond_types"]["actual"]["finite"] = False
    if mutation == "normalized_error": audit["directions"][0]["jvp"]["rows"]["ref_element"]["maximum_normalized_error"] = 2.
    if mutation == "tolerance": audit["native_mapping_parity"]["rows"]["ref_charge"]["tolerance_max"] *= 2
    if mutation == "dtype": audit["native_mapping_parity"]["rows"]["ref_charge"]["actual"]["dtype"] = "torch.float64"
    if mutation == "vjp_channel": audit["directions"][0]["vjp"]["rows"]["unrecognized"] = audit["directions"][0]["vjp"]["rows"].pop("atoms")
    if mutation == "reused_direction": audit["directions"][1]["direction_sha256"] = audit["directions"][0]["direction_sha256"]
    if mutation == "fd_scale": audit["directions"][0]["finite_differences"][0]["epsilon"] = .1
    if mutation == "fd_jvp": audit["directions"][0]["finite_differences"][0]["rows"]["ref_charge"]["analytic_sha256"] = digest("wrong")
    if mutation == "fd_endpoint": audit["directions"][0]["finite_differences"][0]["rows"]["ref_charge"]["endpoint_plus"]["finite"] = False
    if mutation == "scorer_call": audit["scorer_calls"] = 1
    assert not verify_bridge(audit, graph_sha256=digest("shared-graph"), context_sha256=digest("fixed-context"))


def test_repair_target_identity_and_native_fp32_nll_are_mandatory():
    for mutation in ("target", "weight", "dtype", "missing"):
        gate, expected = fixture(repair=True)
        if mutation == "target": gate["measured"]["repair_guidance"]["target_sha256"] = digest("wrong")
        if mutation == "weight": gate["measured"]["repair_guidance"]["weight"] = .5
        if mutation == "dtype": gate["repair_only_validation"]["recomposition_proofs"]["gradient"]["rows"]["scalar"]["native"]["dtype"] = "torch.float64"
        if mutation == "missing": gate.pop("repair_only_validation")
        assert not verify_gate(gate, require_repair=True, expected_profile=expected)


def test_finite_step_disagreement_is_non_gating_but_unfinished_measurements_fail_closed():
    gate, expected = fixture()
    assert verify_gate(gate, require_repair=False, expected_profile=expected)
    old = gate["natural_secant_diagnostics"]["rows"][0]
    gate["natural_secant_diagnostics"]["rows"][0] = {key: old[key] for key in
        ("direction", "direction_sha256", "epsilon", "required_for_pass")}
    gate["natural_secant_diagnostics"]["rows"][0].update(status="failed",
        error="Coarse endpoint is nonfinite", error_type="NonfiniteObjective")
    assert not verify_gate(gate, require_repair=False, expected_profile=expected)
    gate["local_fd_validation"]["independent_forward_ad"]["rows"][0]["passed"] = False
    assert not verify_gate(gate, require_repair=False, expected_profile=expected)


@pytest.mark.parametrize("bad", [None, [], "passed", 1, {"derivative_contract": None}])
def test_malformed_receipts_reject_cleanly(bad):
    assert verify_gate(bad, expected_profile={}) is False
    assert verify_bridge(bad, graph_sha256=digest("g"), context_sha256=digest("c")) is False
