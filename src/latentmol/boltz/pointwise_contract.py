"""Pure receipt checks for selected native first-order agreement.

This contract is an explicit preselected alternative to full-network finite
secant approval, never a fallback after a failed FD test. It certifies the
declared native local AD/canonical-input checks and keeps finite-step diagnostics
separate. It does not prove every Jacobian direction, global differentiability,
chemical validity or binding. Source/context/weight identities and bounded
native Armijo descent remain mandatory at the outer workflow boundary.
"""

from collections.abc import Mapping
import hashlib
import math
import struct

from .additive_contract import _directions, _finite, _hash, profile_record, verify_forward_ad
from .additive_fd import DIRECTIONS, PARITY_VERSION
from .gradient_precision import relative_ad_consistency

LEGACY_POLICY = "pointwise_native_ad_v4"
POLICY = "pointwise_native_ad_v5"
BRIDGE_POLICY = "fixed_graph_native_affine_bridge_v1"
COMPONENTS = ("dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss")
NATIVE_CHANNELS = ("ref_element", "ref_charge", "token_bonds", "bond_types")


def _shape(value, *, scalar=False):
    return (type(value) is list and (scalar or bool(value))
            and all(type(size) is int and size > 0 for size in value)
            and (not scalar or math.prod(value) == 1))


def _witness(value, *, dtype=None, shape=None):
    if (not isinstance(value, Mapping) or not _hash(value.get("sha256"))
            or value.get("dtype") not in ("torch.float32", "torch.float64")
            or not _shape(value.get("shape"), scalar=True) and not _shape(value.get("shape"))
            or dtype is not None and value["dtype"] != dtype
            or shape is not None and value["shape"] != shape):
        return False
    return True


def _exact_proof(proof, *, baseline, tangent, latent_shape, scalar_dtype):
    if (not isinstance(proof, Mapping) or proof.get("version") != PARITY_VERSION
            or proof.get("passed") is not True or set(proof.get("rows", {})) != {"scalar", "reverse", "jvp"}):
        return False
    for key, row in proof["rows"].items():
        left, right = row.get("native"), row.get("recomposed")
        dtype = "torch.float32" if key == "reverse" else scalar_dtype
        if (row.get("exact_equal") is not True or left != right
                or not _witness(left, dtype=dtype, shape=latent_shape if key == "reverse" else None)):
            return False
        if key != "reverse":
            if not _shape(left["shape"], scalar=True) or not _finite(left.get("value")):
                return False
            expected = baseline if key == "scalar" else tangent
            if left["value"] != expected:
                return False
            raw = struct.pack("=d" if dtype == "torch.float64" else "=f", left["value"])
            if hashlib.sha256(raw).hexdigest() != left["sha256"]:
                return False
    return True


def _comparison_group(group, channels, *, rtol, atol, actual_dtype, expected_dtype="torch.float64"):
    """Validate authenticated tensor error summaries without inventing tensors.

    Full per-element comparisons execute in the sealed allocated producer.
    The reader recomputes their declared global bound and checks normalized
    errors, finite witnesses, channel coverage and actual tensor identities.
    Hashes alone cannot reconstruct a dense native tensor or its residuals.
    """
    if (not isinstance(group, Mapping) or group.get("passed") is not True
            or group.get("rtol") != rtol or group.get("atol") != atol
            or set(group.get("rows", {})) != set(channels)):
        return False
    for row in group["rows"].values():
        actual, expected = row.get("actual"), row.get("expected")
        if (not _witness(actual, dtype=actual_dtype) or not _witness(expected, dtype=expected_dtype)
                or actual["shape"] != expected["shape"]
                or any(value.get("finite") is not True or not _finite(value.get("max_abs"))
                       or value["max_abs"] < 0 for value in (actual, expected))):
            return False
        fields = ("max_abs_error", "tolerance_max", "maximum_normalized_error")
        if any(not _finite(row.get(field)) or row[field] < 0 for field in fields):
            return False
        bound = atol + rtol * expected["max_abs"]
        if (row.get("passed") is not True or row["tolerance_max"] != bound
                or row["max_abs_error"] > bound or row["maximum_normalized_error"] > 1
                or (row["max_abs_error"] == 0) != (row["maximum_normalized_error"] == 0)):
            return False
    return True


def verify_bridge(audit, *, graph_sha256, context_sha256):
    """Require native mapping plus independent FP64 affine AD/FD coverage."""
    try:
        if (audit.get("version") != BRIDGE_POLICY or audit.get("passed") is not True
                or audit.get("graph_sha256") != graph_sha256 or audit.get("context_sha256") != context_sha256
                or audit.get("auxiliary_unchanged") is not True or audit.get("context_unchanged") is not True
                or audit.get("affine_extension_not_molecule_samples") is not True
                or type(audit.get("scorer_calls")) is not int or audit["scorer_calls"] != 0
                or not _comparison_group(audit["native_mapping_parity"], NATIVE_CHANNELS,
                    rtol=1e-4, atol=1e-5, actual_dtype="torch.float32")
                or not _comparison_group(audit["float64_value_parity"], NATIVE_CHANNELS,
                    rtol=1e-8, atol=1e-10, actual_dtype="torch.float64")):
            return False
        rows = audit["directions"]
        if type(rows) is not list or [row.get("name") for row in rows] != ["affine_1", "affine_2"]:
            return False
        if any(not _hash(row.get("direction_sha256")) for row in rows) or rows[0]["direction_sha256"] == rows[1]["direction_sha256"]:
            return False
        base = audit["float64_value_parity"]["rows"]
        native = audit["native_mapping_parity"]["rows"]
        if any(native[name]["expected"] != base[name]["expected"]
               or native[name]["actual"]["shape"] != base[name]["actual"]["shape"] for name in NATIVE_CHANNELS):
            return False
        vjp_shapes = None
        for row in rows:
            if (row.get("passed") is not True
                    or not _comparison_group(row["jvp"], NATIVE_CHANNELS, rtol=1e-8, atol=1e-10, actual_dtype="torch.float64")
                    or not _comparison_group(row["vjp"], ("atoms", "charges", "bonds"),
                        rtol=1e-8, atol=1e-10, actual_dtype="torch.float64")):
                return False
            if any(row["jvp"]["rows"][name]["actual"]["shape"] != base[name]["actual"]["shape"] for name in NATIVE_CHANNELS):
                return False
            shapes = {name: value["actual"]["shape"] for name, value in row["vjp"]["rows"].items()}
            if vjp_shapes is not None and vjp_shapes != shapes:
                return False
            vjp_shapes = shapes
            secants = row["finite_differences"]
            if type(secants) is not list or [item.get("epsilon") for item in secants] != [1e-4, .001]:
                return False
            for item in secants:
                if not _comparison_group(item, NATIVE_CHANNELS, rtol=1e-8, atol=1e-10, actual_dtype="torch.float64"):
                    return False
                for channel, comparison in item["rows"].items():
                    jvp = row["jvp"]["rows"][channel]["actual"]
                    if (comparison.get("analytic_sha256") != jvp["sha256"]
                            or comparison["expected"] != jvp or comparison["actual"]["shape"] != jvp["shape"]
                            or any(not _witness(comparison.get(key), dtype="torch.float64", shape=jvp["shape"])
                                   or comparison[key].get("finite") is not True
                                   or not _finite(comparison[key].get("max_abs")) or comparison[key]["max_abs"] < 0
                                   for key in ("endpoint_plus", "endpoint_minus"))):
                        return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, struct.error):
        return False


def _low_signal_directions(item):
    """Authenticate predeclared probes without inventing a nonzero derivative."""
    selection = item["direction_selection"]
    if (selection.get("selection_rule") != "unconditioned_seeded_low_signal_v1"
            or selection.get("selection_completed_before_endpoint_scores") is not True
            or selection.get("gradient_label_is_random_probe") is not True
            or selection.get("selected_labels") != list(DIRECTIONS)
            or selection.get("pool_exhausted") is not False
            or selection.get("gradient_norm") != item["gradient_norm"]
            or type(selection.get("latent_numel")) is not int or selection["latent_numel"] < 1):
        raise ValueError("Malformed predeclared low-signal directions")
    proposals = selection["proposals"]
    if type(proposals) is not list or len(proposals) != 4:
        raise ValueError("Incomplete low-signal directions")
    ad = {row["direction"]: row for row in item["independent_forward_ad"]["rows"]}
    bindings, seed = {}, proposals[0]["generator_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Invalid low-signal direction seed")
    for index, (label, row) in enumerate(zip(DIRECTIONS, proposals)):
        if (type(row.get("proposal_index")) is not int or row["proposal_index"] != index
                or type(row.get("generator_seed")) is not int or row["generator_seed"] != seed
                or row.get("selected_label") != label or not _hash(row.get("direction_sha256"))
                or not _finite(row.get("analytic_projection_float64"))
                or row["analytic_projection_float64"] != ad[label]["reverse_projection"]):
            raise ValueError("Changed low-signal direction or projection")
        bindings[label] = row["direction_sha256"]
    return bindings


def _verify_item(item, *, baseline, latent_shape, repair=False, policy=POLICY):
    norm = item.get("gradient_norm")
    low_signal = _finite(norm) and 0 <= norm <= 1e-12
    status = "validated_low_signal" if policy == POLICY and low_signal else "validated_nonzero"
    if (item.get("validation_contract") != policy or item.get("passed") is not True
            or item.get("status") != status or item.get("baseline_loss") != baseline
            or item.get("attached") is not True or item.get("fixed_pose_and_mask") is not True
            or not _finite(norm) or norm < 0 or (policy == LEGACY_POLICY and low_signal)
            or not verify_forward_ad(item)):
        return False
    random_bindings = _low_signal_directions(item) if low_signal else _directions(item)[0]
    bindings = item["direction_bindings"]
    if (set(bindings) != set(DIRECTIONS) or (not low_signal and len(set(bindings.values())) != 4)
            or any(not _hash(value) for value in bindings.values())
            or any(bindings[name] != value for name, value in random_bindings.items())
            or item["direction_selection"]["latent_numel"] != math.prod(latent_shape)
            or set(item["recomposition_proofs"]) != set(DIRECTIONS)):
        return False
    reverse = None
    for row in item["independent_forward_ad"]["rows"]:
        if row.get("direction_sha256") != bindings[row["direction"]]:
            return False
        if policy == POLICY:
            expected = relative_ad_consistency(row["tangent"], row["reverse_projection"], norm)
            if row.get("relative_ad_consistency") != expected or not expected["passed"]:
                return False
        proof = item["recomposition_proofs"][row["direction"]]
        if not _exact_proof(proof, baseline=baseline, tangent=row["tangent"], latent_shape=latent_shape,
                            scalar_dtype="torch.float32" if repair else "torch.float64"):
            return False
        current = proof["rows"]["reverse"]
        if reverse is not None and current != reverse:
            return False
        reverse = current
    return True


def _diagnostics(diagnostics, items):
    """Retain the prescribed finite-step observations without requiring agreement."""
    bindings = {"combined_gradient": items["combined"]["direction_bindings"]["gradient"],
        "contact_random_2": items["dbmol_contact_loss"]["direction_bindings"]["random_2"]}
    if (diagnostics.get("required_for_pass") is not False or diagnostics.get("epsilons") != [1e-4, .001]
            or diagnostics.get("directions") != bindings):
        return False
    rows = diagnostics["rows"]
    expected = {(label, epsilon) for label in bindings for epsilon in (1e-4, .001)}
    if type(rows) is not list or len(rows) != 4 or {(row["direction"], row["epsilon"]) for row in rows} != expected:
        return False
    for row in rows:
        if (row.get("required_for_pass") is not False
                or row.get("direction_sha256") != bindings[row["direction"]]):
            return False
        if (row.get("status") != "completed" or any(not _hash(row.get(key)) for key in
                ("plus_graph_sha256", "minus_graph_sha256"))):
            return False
        names = set(row["tangents"])
        if (not {"combined", *COMPONENTS} <= names
                or any(set(row[key]) != names for key in ("f_plus", "f_minus", "secants"))
                or any(not _finite(value) for key in ("f_plus", "f_minus", "tangents", "secants") for value in row[key].values())
                or any(row["secants"][name] != (row["f_plus"][name] - row["f_minus"][name]) / (2 * row["epsilon"])
                       for name in names)):
            return False
        name, label = (("combined", "gradient") if row["direction"] == "combined_gradient"
                       else ("dbmol_contact_loss", "random_2"))
        tangent = next(item["tangent"] for item in items[name]["independent_forward_ad"]["rows"] if item["direction"] == label)
        if row["tangents"][name] != tangent:
            return False
    return True


def verify_gate(gate, *, require_repair=True, expected_profile=None):
    """Reject stale/malformed evidence; caller binds the actual sealed profile."""
    try:
        if type(require_repair) is not bool or expected_profile is None:
            return False
        contract = gate["derivative_contract"]
        policy = contract.get("version")
        profile = contract["profile"]
        if (gate.get("passed") is not True or gate.get("status") != "passed"
                or gate.get("native_seed_parity") is not True or policy not in (LEGACY_POLICY, POLICY)
                or contract.get("passed") is not True or profile != expected_profile
                or profile_record(profile["dbmol"], profile["objective"], repair_weight=profile["repair_weight"]) != profile
                or (profile["repair_weight"] is not None) != require_repair
                or contract.get("scope") != "pointwise_native_AD_consistency"
                or contract.get("native_subgradient_conventions_retained") is not True
                or contract.get("full_network_fd_required") is not False
                or contract.get("historical_gate_approval") is not False
                or contract.get("branch_holding_used_in_optimization") is not False):
            return False
        latent = contract["latent_shape"]
        if not _shape(latent):
            return False
        for key in ("point_sha256", "context_sha256", "graph_sha256"):
            if not _hash(contract.get(key)):
                return False
        if any(contract[key] != gate.get(key) for key in ("point_sha256", "context_sha256")):
            return False
        measured = gate["measured"]
        components = gate["component_gradients"]
        if set(components) != set(COMPONENTS):
            return False
        items = {**components, "combined": gate["local_fd_validation"]}
        baselines = {**{key: measured["components"][key] for key in COMPONENTS}, "combined": measured["loss"]}
        parity = gate["native_grouped_baseline_parity"]
        names = set(COMPONENTS) | {"combined"}
        grouped_names = names | ({"repair_only"} if require_repair else set())
        if (parity.get("passed") is not True or any(parity.get(key) is not True for key in
                ("scalar_exact", "repeated_native_scalar_exact", "repeated_shared_graph_exact", "target_unchanged"))
                or any(set(parity.get(key, {})) != required or any(value is not True for value in parity[key].values())
                    for key, required in (("supplied_gradient_exact", names),
                        ("native_grouped_gradient_exact", grouped_names), ("native_grouped_scalar_exact", grouped_names)))):
            return False
        if require_repair:
            repair = measured["repair_guidance"]
            if (not _hash(contract.get("target_sha256")) or repair.get("target_sha256") != contract["target_sha256"]
                    or repair.get("status") != "active" or repair.get("weight") != profile["repair_weight"]
                    or ("effective_repair_weight" in gate and gate["effective_repair_weight"] != profile["repair_weight"])):
                return False
            items["repair_only"], baselines["repair_only"] = gate["repair_only_validation"], repair["normalized_nll"]
        elif contract.get("target_sha256") is not None or gate.get("repair_only_validation"):
            return False
        elif measured.get("repair_guidance"):
            repair = measured["repair_guidance"]
            if (repair.get("status") != "disabled_weight_zero" or repair.get("weight") != 0
                    or repair.get("used_target", False) is not False
                    or repair.get("target_available", False) is not False):
                return False
        if any(not _verify_item(item, baseline=baselines[key], latent_shape=latent, repair=key == "repair_only",
                                policy=policy)
               for key, item in items.items()):
            return False
        unique_directions = {value for item in items.values() for value in item["direction_bindings"].values()}
        count = gate.get("independent_forward_ad_direction_count")
        forwards = gate.get("actual_pointwise_full_forwards")
        state = gate.get("pointwise_state_unchanged", {})
        if (type(count) is not int or count != len(unique_directions)
                or not (1 if policy == POLICY else 4) <= count <= 20
                or type(forwards) is not int or not 2 + count <= forwards <= 32
                or set(state) != {"frozen_versions", "context", "target"}
                or any(value is not True for value in state.values())):
            return False
        return (_diagnostics(gate["natural_secant_diagnostics"], items)
            and verify_bridge(gate["native_input_bridge_audit"], graph_sha256=contract["graph_sha256"],
                              context_sha256=contract["context_sha256"]))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, struct.error):
        return False
