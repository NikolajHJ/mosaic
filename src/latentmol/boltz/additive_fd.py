"""Cancellation-aware checks of a declared additive native objective.

This is a derivative certificate, not another scoring objective. A small net
derivative can subtract much larger primitive signals. Consequently, a relative
finite-difference tolerance on the net alone need not describe the numerical
accuracy of those signals. Every primitive here must independently meet the
ORIGINAL tolerance at the SAME two neighboring scales and selected direction.
There is no approval by a generous aggregate bound, a different direction for
each primitive, or an outcome-selected replacement endpoint.

Use native forward-AD tangents for primitive ``analytic`` values. Their weighted
sum follows FP64 scalar assembly; separately summed FP32 reverse gradients need
not be bitwise additive. The aggregate must additionally agree with reverse AD
at the existing tight JVP tolerance. Exact original/grouped-expression scalar,
reverse-gradient and native-JVP parity is a separate mandatory live-tensor proof.
Frozen weights, shared graph and optimizer mathematics do not change. Historical
local_v1/branch_local_v2 failures retain their own outcomes. Numerical use and
fixtures belong on allocated HPC, never local implementation checks.
"""

from collections.abc import Mapping
import hashlib
import json
import math
import struct


POLICY = "additive_native_fd_v1"
PARITY_VERSION = "exact_grouped_additive_recomposition_v1"
ANALYTIC_KIND = "native_forward_ad"
EPSILONS = (1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1)
DIRECTIONS = ("gradient", "random_1", "random_2", "random_3")
JVP_RTOL, JVP_ATOL = 1e-3, 1e-6
REPAIR_PRIMITIVE = "repair_nll"
CONFIDENCE_SIGNS = {"iptm": -1., "complex_iplddt": -1., "complex_ipde": 1.,
                    "complex_pde": 1., "confidence_score": -1., "ptm": -1.}
COMPONENTS = ("dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss", "combined")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Require a finite real number: " + name)
    return float(value)


def _get(config, name):
    return config[name] if isinstance(config, Mapping) else getattr(config, name)


def _objective(objective):
    if objective is None:
        return 1., 0., 1.
    if _get(objective, "output") != "dbmol_total_loss":
        raise ValueError("Additive combined audit requires dbmol_total_loss")
    direction = _get(objective, "direction")
    if direction not in ("minimize", "maximize"):
        raise ValueError("Unsupported objective direction")
    scale, offset = (_number(_get(objective, key), key) for key in ("scale", "offset"))
    if scale <= 0:
        raise ValueError("Require a positive objective scale")
    return (-1. if direction == "maximize" else 1.), offset, scale


def primitive_weights(dbmol, component, *, objective=None, repair_weight=None):
    """Declare coefficients without learned normalization or changing signs.

    ``repair_weight=None`` means no repaired target. Zero coefficients are
    explicitly unsupported by this first certificate, rather than silently
    dropping a configured branch or approving a zero-signal primitive.
    """
    if component not in COMPONENTS:
        raise ValueError("Unknown additive objective component")
    confidence = _get(dbmol, "confidence_weights")
    if not isinstance(confidence, Mapping) or set(confidence) != set(CONFIDENCE_SIGNS):
        raise ValueError("Declare exactly the six native confidence weights")
    affinity = {"affinity_pred_value": _number(_get(dbmol, "affinity_regression_weight"), "affinity weight"),
                "affinity_logits_binary": -_number(_get(dbmol, "affinity_logit_weight"), "logit weight")}
    contact = {"dbmol_contact_pocket_loss": .5, "dbmol_contact_anchor_loss": .5}
    conf = {key: sign * _number(confidence[key], key) for key, sign in CONFIDENCE_SIGNS.items()}
    if component == COMPONENTS[0]:
        result = affinity
    elif component == COMPONENTS[1]:
        result = contact
    elif component == COMPONENTS[2]:
        result = conf
    else:
        sign, _, scale = _objective(objective)
        contact_weight = _number(_get(dbmol, "contact_weight"), "contact weight")
        confidence_weight = _number(_get(dbmol, "confidence_weight"), "confidence weight")
        result = {**affinity, **{key: contact_weight * value for key, value in contact.items()},
                  **{key: confidence_weight * value for key, value in conf.items()}}
        result = {key: sign * value / scale for key, value in result.items()}
        if repair_weight is not None:
            weight = _number(repair_weight, "repair weight")
            if weight <= 0:
                raise ValueError("A repaired target needs a positive fixed weight")
            result[REPAIR_PRIMITIVE] = weight
    if repair_weight is not None and component != "combined":
        raise ValueError("Repair weighting belongs only to the combined objective")
    if any(not math.isfinite(value) or value == 0 for value in result.values()):
        raise ValueError("Zero or nonfinite primitive coefficients are unsupported, not passes")
    return result


def objective_offset(objective=None):
    """Constant intercept of scalar_loss; repair is added after this scaling."""
    sign, offset, scale = _objective(objective)
    return -sign * offset / scale


def grouped_native_expression(values, dbmol, component, *, objective=None, repair_weight=None):
    """Rebuild the original operation tree from actual shared native tensors.

    FP64 casts and grouping mirror contact-head DBMol reduce_objective and
    scalar_loss, including the two contact halves BEFORE their overall weight.
    The caller must compare the resulting scalar and its real reverse/forward
    derivatives with the original graph; this function does not manufacture
    those proofs. Values may be tensors or plain numbers in allocated fixtures.
    """
    weights = primitive_weights(dbmol, component, objective=objective, repair_weight=repair_weight)
    if set(values) != set(weights):
        raise ValueError("Primitive membership differs from the declared additive expression")
    def wide(key):
        value = values[key]
        return value.double() if hasattr(value, "double") else _number(value, key)
    affinity = lambda: (_get(dbmol, "affinity_regression_weight") * wide("affinity_pred_value")
                        - _get(dbmol, "affinity_logit_weight") * wide("affinity_logits_binary"))
    contact = lambda: (wide("dbmol_contact_pocket_loss") + wide("dbmol_contact_anchor_loss")) / 2
    confidence = lambda: sum(_get(dbmol, "confidence_weights")[key] * sign * wide(key)
                             for key, sign in CONFIDENCE_SIGNS.items())
    if component == COMPONENTS[0]: return affinity()
    if component == COMPONENTS[1]: return contact()
    if component == COMPONENTS[2]: return confidence()
    total = sum((affinity(), _get(dbmol, "contact_weight") * contact(),
                 _get(dbmol, "confidence_weight") * confidence()))
    sign, offset, scale = _objective(objective)
    result = sign * (total - offset) / scale
    return result if repair_weight is None else result + repair_weight * values[REPAIR_PRIMITIVE]


def exact_recomposition_proof(native, recomposed):
    """Compare live scalar/reverse/JVP tensors exactly and retain byte witnesses.

    Both mappings need exactly ``scalar``, ``reverse`` and ``jvp``. Compare
    native and grouped derivatives from the SAME graph/point/direction, not
    independently decoded molecules or a stopped target from another step.
    Torch is imported only for this explicitly allocated numerical operation.
    """
    import torch
    required = {"scalar", "reverse", "jvp"}
    if set(native) != required or set(recomposed) != required:
        raise ValueError("Recomposition requires scalar, reverse gradient and native JVP")
    proof = {"version": PARITY_VERSION, "rows": {}, "passed": False}
    for key in ("scalar", "reverse", "jvp"):
        left, right = native[key], recomposed[key]
        if any(not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64)
               or value.numel() == 0 or not bool(value.isfinite().all()) for value in (left, right)):
            raise ValueError("Missing/nonfinite recomposition tensor: " + key)
        if key != "reverse" and (left.numel() != 1 or right.numel() != 1):
            raise ValueError("Recomposition scalar/JVP must be scalar")
        exact = left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        def witness(value):
            raw = value.detach().cpu().contiguous()
            result = {"shape": list(raw.shape), "dtype": str(raw.dtype),
                      "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest()}
            if raw.numel() == 1: result["value"] = float(raw)
            return result
        proof["rows"][key] = {"exact_equal": exact, "native": witness(left), "recomposed": witness(right)}
    proof["passed"] = all(row["exact_equal"] for row in proof["rows"].values())
    return proof


def _valid_proof(proof):
    if (not isinstance(proof, Mapping) or proof.get("version") != PARITY_VERSION
            or proof.get("passed") is not True or not isinstance(proof.get("rows"), Mapping)
            or set(proof["rows"]) != {"scalar", "reverse", "jvp"}):
        return False
    for key, row in proof["rows"].items():
        if not isinstance(row, Mapping) or set(row) != {"exact_equal", "native", "recomposed"}:
            return False
        left, right = row.get("native", {}), row.get("recomposed", {})
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        digest = left.get("sha256")
        shape = left.get("shape")
        if (not isinstance(shape, list) or any(type(size) is not int or size < 1 for size in shape)
                or (key == "reverse" and not shape)):
            return False
        count = math.prod(shape)
        required_fields = {"shape", "dtype", "sha256", *(["value"] if count == 1 else [])}
        if (row.get("exact_equal") is not True or left != right
                or set(left) != required_fields
                or left.get("dtype") != ("torch.float32" if key == "reverse" else "torch.float64")
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            return False
        if key in ("scalar", "jvp") and count != 1:
            return False
        if count == 1:
            if type(left.get("value")) not in (int, float) or not math.isfinite(left["value"]):
                return False
            try:
                raw = struct.pack("=f" if key == "reverse" else "=d", left["value"])
            except (OverflowError, struct.error):
                return False
            # Scalar metadata is sufficient to recheck its byte witness. A
            # stale digest cannot authenticate a subsequently changed value.
            if hashlib.sha256(raw).hexdigest() != digest:
                return False
    return True


def _ulp(value, dtype):
    if dtype == "torch.float64": return math.ulp(value)
    if dtype != "torch.float32":
        raise ValueError("Declare actual native primitive FP32/FP64 endpoint dtype")
    absolute = abs(value)
    bits, = struct.unpack("I", struct.pack("f", absolute))
    rounded, = struct.unpack("f", struct.pack("I", bits))
    if rounded != absolute:
        raise ValueError("FP32 endpoint metadata is not an actual FP32 value")
    upper, = struct.unpack("f", struct.pack("I", bits + 1))
    lower = -2.**-149 if bits == 0 else struct.unpack("f", struct.pack("I", bits - 1))[0]
    spacing = max(upper - rounded, rounded - lower)
    if not math.isfinite(spacing): raise ValueError("Unbounded FP32 endpoint spacing")
    return spacing


def _gamma(operations):
    unit = 2.**-53
    return operations * unit / (1 - operations * unit)


def assess_row(weights, row, *, rtol, atol, offset=0.):
    """Check one common endpoint pair; aggregate bounds cannot excuse a primitive.

    ``primitives`` and ``aggregate`` contain actual ``analytic`` forward-AD
    tangents and ``f_plus/f_minus`` native values. Primitive rows additionally
    declare actual loss_dtype_plus/minus. Aggregate carries reverse_projection.
    A mandatory exact live recomposition proof is attached to this direction.
    Any remaining weighted assembly discrepancy is bounded ONLY by FP64
    expression rounding and the known native FP32 ``lambda * repair_nll``
    scalar product, not by observed mismatch, a fitted coefficient or a
    model-wide FP32 error allowance. This mirrors score_at's existing arithmetic.
    """
    if not isinstance(weights, Mapping) or not weights or any(_number(v, k) == 0 for k, v in weights.items()):
        raise ValueError("Declare nonzero finite additive coefficients")
    rtol, atol, offset = (_number(value, name) for value, name in ((rtol, "rtol"), (atol, "atol"), (offset, "offset")))
    if rtol < 0 or atol < 0: raise ValueError("Negative tolerances")
    epsilon = _number(row.get("epsilon"), "epsilon")
    if epsilon not in EPSILONS or row.get("analytic_kind") != ANALYTIC_KIND:
        raise ValueError("Require the original grid and native forward-AD primitives")
    if set(row.get("primitives", {})) != set(weights):
        raise ValueError("Unknown or missing additive primitive")
    primitive_rows, projected, secants = {}, {}, {}
    for key, coefficient in weights.items():
        source = row["primitives"][key]
        analytic, plus, minus = (_number(source.get(field), key + ":" + field)
                                 for field in ("analytic", "f_plus", "f_minus"))
        numerical = (plus - minus) / (2 * epsilon)
        if not math.isfinite(numerical):
            raise ValueError("Nonfinite primitive finite difference: " + key)
        error = abs(analytic - numerical)
        tolerance = atol + rtol * max(abs(analytic), abs(numerical))
        ulp = max(_ulp(plus, source.get("loss_dtype_plus")), _ulp(minus, source.get("loss_dtype_minus")))
        informative = abs(analytic) > atol and plus != minus and abs(2 * epsilon * analytic) > ulp
        primitive_rows[key] = {"analytic": analytic, "central_difference": numerical,
            "absolute_error": error, "tolerance": tolerance, "informative": informative,
            "output_ulp": ulp, "passed": informative and error <= tolerance}
        projected[key], secants[key] = coefficient * analytic, coefficient * numerical
    aggregate = row.get("aggregate", {})
    if (aggregate.get("loss_dtype_plus") != "torch.float64"
            or aggregate.get("loss_dtype_minus") != "torch.float64"):
        raise ValueError("Additive certificate requires actual native FP64 aggregate assembly")
    analytic, reverse, plus, minus = (_number(aggregate.get(field), "aggregate:" + field)
                                    for field in ("analytic", "reverse_projection", "f_plus", "f_minus"))
    baseline = _number(aggregate.get("baseline_loss"), "aggregate:baseline_loss")
    numerical = (plus - minus) / (2 * epsilon)
    if not math.isfinite(numerical): raise ValueError("Nonfinite aggregate finite difference")
    flat_analytic, flat_fd = math.fsum(projected.values()), math.fsum(secants.values())
    flat_plus = math.fsum(weights[key] * row["primitives"][key]["f_plus"] for key in weights) + offset
    flat_minus = math.fsum(weights[key] * row["primitives"][key]["f_minus"] for key in weights) + offset
    # 64 covers the at-most-eleven-primitive pinned native scalar expression,
    # its products, grouping, sign/scale and optional fixed-target addition.
    if len(weights) > 11: raise ValueError("Unsupported additive expression size")
    rounding = _gamma(64)
    plus_bound = rounding * (math.fsum(abs(weights[k] * row["primitives"][k]["f_plus"]) for k in weights) + abs(offset))
    minus_bound = rounding * (math.fsum(abs(weights[k] * row["primitives"][k]["f_minus"]) for k in weights) + abs(offset))
    analytic_bound = rounding * math.fsum(abs(value) for value in projected.values())
    # The sole FP32 assembly product is outside reduce_objective: score_at
    # multiplies its raw FP32 repair NLL by a Python lambda before adding it to
    # the FP64 binding objective. Two FP32 roundings cover scalar conversion
    # and multiplication; no upstream model error is granted by this bound.
    repair_rounding = 2 * 2.**-24 / (1 - 2 * 2.**-24)
    repair_bounds = {"plus": 0., "minus": 0., "analytic": 0.}
    if REPAIR_PRIMITIVE in weights:
        repair = row["primitives"][REPAIR_PRIMITIVE]
        if (repair.get("loss_dtype_plus") != "torch.float32"
                or repair.get("loss_dtype_minus") != "torch.float32"):
            raise ValueError("Native repair scalar-product policy requires raw FP32 NLL")
        products = [weights[REPAIR_PRIMITIVE] * repair[key] for key in ("analytic", "f_plus", "f_minus")]
        minimum, maximum = 2.**-126, (2. - 2.**-23) * 2.**127
        if (not minimum <= abs(weights[REPAIR_PRIMITIVE]) <= maximum
                or any(value != 0 and not minimum <= abs(value) <= maximum for value in products)):
            raise ValueError("Subnormal/overflow native repair scalar products are unsupported")
        repair_bounds = {"plus": repair_rounding * abs(weights[REPAIR_PRIMITIVE] * repair["f_plus"]),
                         "minus": repair_rounding * abs(weights[REPAIR_PRIMITIVE] * repair["f_minus"]),
                         "analytic": repair_rounding * abs(projected[REPAIR_PRIMITIVE])}
        plus_bound += repair_bounds["plus"]
        minus_bound += repair_bounds["minus"]
        analytic_bound += repair_bounds["analytic"]
    assembly = {"analytic_residual": abs(analytic - flat_analytic), "analytic_bound": analytic_bound,
        "plus_residual": abs(plus - flat_plus), "plus_bound": plus_bound,
        "minus_residual": abs(minus - flat_minus), "minus_bound": minus_bound,
        "dtype": "torch.float64", "maximum_operations": 64,
        "native_fp32_repair_scalar_product_bounds": repair_bounds}
    assembly["passed"] = (assembly["analytic_residual"] <= analytic_bound
        and assembly["plus_residual"] <= plus_bound and assembly["minus_residual"] <= minus_bound)
    assembly_fd_bound = (plus_bound + minus_bound) / (2 * epsilon) + analytic_bound
    measured_bound = math.fsum(abs(weights[k]) * item["absolute_error"] for k, item in primitive_rows.items())
    tolerance_bound = math.fsum(abs(weights[k]) * item["tolerance"] for k, item in primitive_rows.items())
    jvp_error = abs(analytic - reverse)
    jvp_tolerance = JVP_ATOL + JVP_RTOL * max(abs(analytic), abs(reverse))
    gross = math.fsum(abs(value) for value in projected.values())
    net = abs(flat_analytic)
    proof = row.get("recomposition")
    recomposition = (_valid_proof(proof)
        and proof["rows"]["scalar"]["native"]["value"] == baseline
        and proof["rows"]["jvp"]["native"]["value"] == analytic)
    error = abs(analytic - numerical)
    aggregate_informative = (abs(analytic) > atol and plus != minus
                             and abs(2 * epsilon * analytic) > max(math.ulp(plus), math.ulp(minus)))
    passed = (all(item["passed"] for item in primitive_rows.values()) and assembly["passed"]
        and recomposition and aggregate_informative and jvp_error <= jvp_tolerance
        and error <= measured_bound + assembly_fd_bound)
    return {"epsilon": epsilon, "direction": row.get("direction"), "passed": passed,
        "primitives": primitive_rows, "aggregate": {"analytic": analytic, "reverse_projection": reverse,
            "central_difference": numerical, "absolute_error": error,
            "informative": aggregate_informative,
            "original_net_tolerance": atol + rtol * max(abs(analytic), abs(numerical)),
            "original_net_passed": error <= atol + rtol * max(abs(analytic), abs(numerical)),
            "weighted_measured_error_bound": measured_bound,
            "weighted_original_tolerance_bound": tolerance_bound, "assembly_fd_bound": assembly_fd_bound,
            "gross_projection": gross, "net_projection": net,
            "cancellation_condition": gross / net if net else None,
            "zero_net_projection": net == 0},
        "assembly": assembly, "exact_grouped_recomposition": recomposition,
        "independent_forward_reverse": {"passed": jvp_error <= jvp_tolerance,
            "absolute_error": jvp_error, "tolerance": jvp_tolerance,
            "rtol": JVP_RTOL, "atol": JVP_ATOL}}


def certify(weights, rows, *, directions, rtol, atol, minimum_passes=2, offset=0.):
    """Require common neighboring successes for every predeclared direction.

    Direction labels/hashes must be selected before scoring; rows retain every
    scale. The caller owns source/context/point/target bindings. This function
    never chooses a replacement seed, drops a primitive or retests a failure.
    """
    if (tuple(directions) != DIRECTIONS or any(not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value) for value in directions.values())
            or len(set(directions.values())) != len(directions)
            or type(minimum_passes) is not int or not 2 <= minimum_passes <= len(EPSILONS)):
        raise ValueError("Declare the four original distinct directions and at least two neighboring scales")
    by_key = {}
    for row in rows:
        key = row.get("direction"), row.get("epsilon")
        if (key in by_key or key[0] not in directions or key[1] not in EPSILONS
                or row.get("direction_sha256") != directions[key[0]]):
            raise ValueError("Duplicate/unknown scale, direction or mismatched predeclared direction identity")
        by_key[key] = row
    if set(by_key) != {(direction, epsilon) for direction in directions for epsilon in EPSILONS}:
        raise ValueError("A complete unchanged direction/epsilon grid is mandatory")
    baseline_identity = None
    for direction in directions:
        direction_identity = None
        for epsilon in EPSILONS:
            row = by_key[(direction, epsilon)]
            aggregate, primitives, proof = row.get("aggregate", {}), row.get("primitives", {}), row.get("recomposition")
            if (not isinstance(aggregate, Mapping) or not isinstance(primitives, Mapping)
                    or set(primitives) != set(weights)
                    or any(not isinstance(primitives[name], Mapping) for name in weights)
                    or not _valid_proof(proof)):
                raise ValueError("Malformed primitive membership or exact recomposition proof")
            # One frozen point and one native directional derivative underlie
            # every epsilon. Mixing individually favorable tangents, baselines
            # or scalar/reverse/JVP witnesses is not one derivative certificate.
            signature = {"baseline_loss": _number(aggregate.get("baseline_loss"), "baseline_loss"),
                "analytic": _number(aggregate.get("analytic"), "aggregate analytic"),
                "reverse_projection": _number(aggregate.get("reverse_projection"), "reverse projection"),
                "primitive_analytics": {name: _number(primitives[name].get("analytic"), name + " analytic")
                                        for name in weights}, "recomposition": proof}
            identity = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if direction_identity is not None and identity != direction_identity:
                raise ValueError("Baseline, primitive analytics or recomposition proof changed across epsilon grid")
            direction_identity = identity
            # JVP and projection depend on direction; the baseline scalar and
            # full reverse vector must still identify the same point globally.
            common = json.dumps({"baseline_loss": signature["baseline_loss"],
                "scalar": proof["rows"]["scalar"], "reverse": proof["rows"]["reverse"]},
                sort_keys=True, separators=(",", ":"), allow_nan=False)
            if baseline_identity is not None and common != baseline_identity:
                raise ValueError("Baseline scalar or reverse witness changed across declared directions")
            baseline_identity = common
    observed = [assess_row(weights, by_key[(direction, epsilon)], rtol=rtol, atol=atol, offset=offset)
                for direction in directions for epsilon in EPSILONS]
    results = {}
    for direction in directions:
        runs, current = [], []
        for item in (value for value in observed if value["direction"] == direction):
            if item["passed"]: current.append(item["epsilon"])
            else:
                if current: runs.append(current)
                current = []
        if current: runs.append(current)
        results[direction] = {"passed": any(len(run) >= minimum_passes for run in runs),
                              "common_passing_neighbor_runs": runs}
    return {"version": POLICY, "passed": all(item["passed"] for item in results.values()),
        "weights": dict(weights), "offset": offset, "analytic_kind": ANALYTIC_KIND,
        "rtol": rtol, "atol": atol, "minimum_common_neighboring_passes": minimum_passes,
        "epsilons": list(EPSILONS), "direction_bindings": dict(directions),
        "rows": observed, "direction_results": results,
        "approval_requires_every_primitive_at_shared_scales": True,
        "historical_failures_reclassified": False}
