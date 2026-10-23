"""Pure metadata bindings for the additive native-derivative certificate.

The actual model/AD and paired endpoints are produced inside allocated workers.
Readers must bind those receipts to the declared loss coefficients, own AD
rows, original direction/grid and both endpoint namespaces, not bare flags.
"""
import math
import re

from .additive_fd import DIRECTIONS, EPSILONS, _ulp, certify, objective_offset, primitive_weights


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def profile_record(dbmol, objective, *, repair_weight=None):
    """Record the configured arithmetic coefficients, never fit them to probes."""
    def get(value, key):
        return value[key] if isinstance(value, dict) else getattr(value, key)
    result = {"dbmol": {key: get(dbmol, key) for key in
        ("profile", "objective_arithmetic", "affinity_regression_weight", "affinity_logit_weight", "contact_weight", "confidence_weight")},
        "objective": {key:get(objective,key) for key in ("output","direction","scale","offset")},
        "repair_weight": repair_weight}
    result["dbmol"]["confidence_weights"] = dict(get(dbmol,"confidence_weights"))
    if (result["dbmol"]["profile"] != "dbmol_fixed_pose_v1"
            or result["dbmol"]["objective_arithmetic"] != "contact_head_and_assembly_float64_v1"):
        raise ValueError("Require the declared native DBMol contact-head arithmetic profile")
    # Validates all supported keys/signs, finite coefficients and repair scope.
    primitive_weights(result["dbmol"], "combined", objective=result["objective"], repair_weight=repair_weight)
    return result


def verify_forward_ad(item):
    """Pure, type-safe equivalent of the existing tight native AD reader."""
    try:
        evidence, baseline = item["independent_forward_ad"], item["baseline_loss"]
        rows = evidence["rows"]
        if (not _finite(baseline) or evidence.get("passed") is not True
                or evidence.get("rtol") != 1e-3 or evidence.get("atol") != 1e-6
                or type(rows) is not list or len(rows) != 4
                or [row.get("direction") for row in rows] != list(DIRECTIONS)):
            return False
        for row in rows:
            if any(not _finite(row.get(key)) for key in
                   ("primal", "tangent", "reverse_projection", "absolute_error", "tolerance")):
                return False
            error = abs(row["tangent"] - row["reverse_projection"])
            tolerance = 1e-6 + 1e-3 * max(abs(row["tangent"]), abs(row["reverse_projection"]))
            if (row.get("passed") is not True or row.get("primal_exact") is not True
                    or row["primal"] != baseline or row["absolute_error"] != error
                    or row["tolerance"] != tolerance or error > tolerance):
                return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False


def _directions(item):
    """Authenticate the analytic-only original proposal selection metadata."""
    selection = item["direction_selection"]
    if (selection.get("selection_completed_before_endpoint_scores") is not True
            or selection.get("pool_exhausted") is not False
            or selection.get("selected_labels") != list(DIRECTIONS)
            or type(selection.get("latent_numel")) is not int or selection["latent_numel"] < 1
            or not _finite(selection.get("gradient_norm")) or selection["gradient_norm"] <= 1e-12
            or not _finite(item.get("gradient_norm"))
            or selection["gradient_norm"] != item["gradient_norm"]):
        raise ValueError("Malformed original direction selection")
    threshold = .5 * selection["gradient_norm"] / math.sqrt(selection["latent_numel"])
    proposals = selection["proposals"]
    if (selection.get("minimum_absolute_random_projection") != threshold
            or type(proposals) is not list or not 3 <= len(proposals) <= 64):
        raise ValueError("Changed original bounded projection policy")
    ad = {row["direction"]: row for row in item["independent_forward_ad"]["rows"]}
    bindings, seed = {}, proposals[0]["generator_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Invalid original direction seed")
    for index, row in enumerate(proposals):
        if (type(row.get("proposal_index")) is not int or row["proposal_index"] != index
                or type(row.get("generator_seed")) is not int or row["generator_seed"] != seed
                or not _hash(row.get("direction_sha256"))
                or not _finite(row.get("analytic_projection_float64"))
                or row.get("minimum_absolute_projection") != threshold):
            raise ValueError("Malformed bounded direction proposal")
        accepted = abs(row["analytic_projection_float64"]) >= threshold
        label = "random_" + str(len(bindings) + 1) if accepted else None
        if (row.get("accepted") is not accepted or row.get("selected_label") != label
                or (accepted and (len(bindings) >= 3
                    or row["analytic_projection_float64"] != ad[label]["reverse_projection"]))):
            raise ValueError("Direction proposal differs from the original analytic-only selection")
        if accepted:
            bindings[label] = row["direction_sha256"]
        if len(bindings) == 3 and index != len(proposals) - 1:
            raise ValueError("Unexpected proposals after completed direction selection")
    if (list(bindings) != list(DIRECTIONS[1:])
            or selection.get("legacy_direction_sha256") != proposals[0]["direction_sha256"]):
        raise ValueError("Incomplete original random directions")
    return bindings, seed


def original_rows(item, mode, *, dtype="torch.float64"):
    """Recheck retained net FD arithmetic and its original neighboring rule.

    Net failures are permitted for additive components and remain failures.
    Repair approval still requires this unchanged scalar test to pass. The
    realized chord is retained metadata, not a new derivative correction.
    """
    if not verify_forward_ad(item):
        raise ValueError("Native forward/reverse evidence failed")
    _, seed = _directions(item)
    name = {"natural": "natural_secants", "held": "local_branch_fd"}[mode]
    old = item[name]
    if "original_net_fd" in item and old != item["original_net_fd"][mode]:
        raise ValueError("Retained original net FD aliases differ")
    policy = old["policy"]
    if (policy.get("version") != "local_v1" or type(policy.get("seed")) is not int or policy["seed"] != seed
            or policy.get("fd_rtol") != .05 or policy.get("fd_atol") != 1e-6
            or policy.get("epsilons") != list(EPSILONS)
            or policy.get("minimum_neighboring_passes") != 2
            or policy.get("random_directions") != 3 or policy.get("maximum_gaussian_proposals") != 64
            or policy.get("projection_fraction") != .5
            or old["direction_selection"] != item["direction_selection"]):
        raise ValueError("Original scalar FD policy changed")
    rows = old["finite_differences"]
    expected = {(direction, epsilon) for direction in DIRECTIONS for epsilon in EPSILONS}
    by_key = {(row["direction"], row["epsilon"]): row for row in rows}
    if type(rows) is not list or len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("Original scalar FD grid is incomplete or repeated")
    ad = {row["direction"]: row for row in item["independent_forward_ad"]["rows"]}
    results = {}
    for direction in DIRECTIONS:
        runs, current = [], []
        nominal = None
        for epsilon in EPSILONS:
            row = by_key[(direction, epsilon)]
            fields = ("analytic", "central_difference", "absolute_error", "tolerance", "f_plus", "f_minus",
                      "baseline_loss", "output_ulp", "analytic_dot_float64", "realized_plus_distance", "realized_minus_distance")
            if any(not _finite(row.get(field)) for field in fields):
                raise ValueError("Nonfinite original scalar FD row")
            if (row["baseline_loss"] != item["baseline_loss"]
                    or row["analytic_dot_float64"] != ad[direction]["reverse_projection"]
                    or row.get("loss_dtype_plus") != dtype or row.get("loss_dtype_minus") != dtype
                    or row["realized_plus_distance"] < 0 or row["realized_minus_distance"] < 0):
                raise ValueError("Original scalar FD row differs from native derivative/baseline")
            if nominal is not None and row["analytic"] != nominal:
                raise ValueError("Original analytic projection changed across scales")
            nominal = row["analytic"]
            numerical = (row["f_plus"] - row["f_minus"]) / (2 * epsilon)
            error = abs(nominal - numerical)
            tolerance = 1e-6 + .05 * max(abs(nominal), abs(numerical))
            ulp = max(_ulp(row["f_plus"], dtype), _ulp(row["f_minus"], dtype))
            informative = (abs(nominal) > 1e-6 and abs(2 * epsilon * nominal) > ulp
                and row["f_plus"] != row["f_minus"]
                and (row["realized_plus_distance"] > 0 or row["realized_minus_distance"] > 0))
            passed = error <= tolerance and informative
            if (row["central_difference"] != numerical or row["absolute_error"] != error
                    or row["tolerance"] != tolerance or row["output_ulp"] != ulp
                    or row.get("passed_tolerance") is not (error <= tolerance)
                    or row.get("informative") is not informative or row.get("passed") is not passed):
                raise ValueError("Stale original scalar FD arithmetic or pass flag")
            if passed: current.append(epsilon)
            else:
                if current: runs.append(current)
                current = []
        if current: runs.append(current)
        results[direction] = {"passed": any(len(run) >= 2 for run in runs),
            "passing_neighbor_runs": runs, "minimum_neighboring_passes": 2}
    passed = all(row["passed"] for row in results.values())
    if (old["direction_results"] != results or old.get("passed") is not passed
            or old.get("status") != ("passed" if passed else "failed")):
        raise ValueError("Stale original scalar FD aggregate")
    return by_key


def verify_repair(item):
    """Keep the original v2 repair scalar/JVP rule under the v3 outer contract."""
    try:
        if (item.get("validation_contract") != "additive_branch_v3" or item.get("passed") is not True
                or item.get("status") != "validated_nonzero"
                or item.get("repair_derivative_contract") != "unchanged_branch_local_v2"
                or item.get("original_repair_policy", {}).get("version") != "branch_local_v2"):
            return False
        original_rows(item, "natural", dtype="torch.float32")
        original_rows(item, "held", dtype="torch.float32")
        return item["local_branch_fd"]["passed"] is True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def verify_item(item, mode, profile, component):
    """Recompute full evidence and reject coherent stale/transplanted receipts."""
    try:
        if component not in ("dbmol_affinity_loss","dbmol_contact_loss","dbmol_confidence_loss","combined"):
            return False
        if mode not in ("natural", "held") or not verify_forward_ad(item):
            return False
        if any(set(item[name]) != {"natural", "held"} for name in
               ("additive_certificates", "additive_inputs", "original_net_fd")):
            return False
        if profile_record(profile["dbmol"], profile["objective"], repair_weight=profile["repair_weight"]) != profile:
            return False
        certificate = item["additive_certificates"][mode]
        inputs = item["additive_inputs"][mode]
        if (item.get("validation_contract") != "additive_branch_v3" or certificate.get("passed") is not True
                or certificate.get("rtol") != .05 or certificate.get("atol") != 1e-6
                or certificate.get("minimum_common_neighboring_passes") != 2):
            return False
        kwargs = ({"objective":profile["objective"], "repair_weight":profile["repair_weight"]}
                  if component == "combined" else {})
        weights = primitive_weights(profile["dbmol"], component, **kwargs)
        expected_offset = objective_offset(profile["objective"]) if component == "combined" else 0.
        if weights != certificate["weights"] or certificate["offset"] != expected_offset:
            return False
        bindings, _ = _directions(item)
        for label in DIRECTIONS[1:]:
            if certificate["direction_bindings"].get(label) != bindings[label]:
                return False
        ad = item["independent_forward_ad"]["rows"]
        if [r.get("direction") for r in ad] != list(DIRECTIONS):
            return False
        ad = {r["direction"]:r for r in ad}
        expected = {(direction,epsilon) for direction in DIRECTIONS for epsilon in EPSILONS}
        # Both namespaces must preserve the full original grid and describe the
        # SAME baseline/JVP/recomposition proof. Their actual secants may differ.
        namespaces = {}
        for label in ("natural","held"):
            raw = item["additive_inputs"][label]
            rows = {(r["direction"],r["epsilon"]):r for r in raw}
            old_rows = original_rows(item, label)
            if len(raw)!=len(expected) or set(rows)!=expected or set(old_rows)!=expected:
                return False
            for key,row in rows.items():
                aggregate, native, old_row = row["aggregate"],ad[key[0]],old_rows[key]
                if (aggregate.get("baseline_loss") != item.get("baseline_loss")
                        or aggregate.get("baseline_loss") != native.get("primal")
                        or aggregate.get("analytic") != native.get("tangent")
                        or aggregate.get("reverse_projection") != native.get("reverse_projection")
                        or any(aggregate.get(field)!=old_row.get(field) for field in ("f_plus","f_minus","baseline_loss"))
                        or type(old_row.get("passed")) is not bool):
                    return False
            namespaces[label] = rows
        for key, left in namespaces["natural"].items():
            right = namespaces["held"][key]
            if (left["recomposition"] != right["recomposition"]
                    or {k:v["analytic"] for k,v in left["primitives"].items()}
                       != {k:v["analytic"] for k,v in right["primitives"].items()}):
                return False
        # Both namespaces must be independently well formed, even when only a
        # held-branch pass is required for this component. Natural failures are
        # retained verbatim, never silently rebuilt as a passing certificate.
        rebuilt_certificates = {}
        for namespace in ("natural", "held"):
            stored = item["additive_certificates"][namespace]
            rebuilt = certify(weights, item["additive_inputs"][namespace],
                directions=certificate["direction_bindings"], rtol=.05, atol=1e-6, minimum_passes=2,
                offset=expected_offset)
            if rebuilt != stored:
                return False
            rebuilt_certificates[namespace] = rebuilt
        return rebuilt_certificates[mode]["passed"] is True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False
