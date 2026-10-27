"""Prospective endpoint admission; original scientific failures stay failed.

All candidate points are fixed from the saved FP32 simplex before any evaluator
call. Baseline repetitions select a precision-eligible pair from a fixed local
ladder before any perturbed forward. Observed finite differences never select a
pair, alter an epsilon, change a graph, or qualify an original failed study.
Native identity/null/parity/pilot/runtime/storage authentication belongs to the
separate allocated caller. Passing toy evaluators is not native validation.
"""

from copy import deepcopy
import base64
import hashlib
import json
import os


VERSION = "prospective-endpoint-admission-v3"
ORIGINAL_EPSILONS = (0.01, 0.003, 0.001)
LOCAL_EPSILONS = (0.003, 0.001, 0.0003, 0.0001, 0.00003)
REPEATS = 3
RELATIVE_TOLERANCE = 0.15
PARITY_ATOL = 2e-5
GRADIENT_REPEAT_ATOL = 2e-5
MINIMUM_TANGENT = 1e-7
ULP_MULTIPLIER = 8.0
SPREAD_MULTIPLIER = 4.0
SIGNAL_TO_FLOOR = 2.0 / RELATIVE_TOLERANCE
GEOMETRY_ROWS = 4
SCOPES = ("saved_failed_endpoint", "original_probe_not_reached", "not_preexisting_at_new_endpoint")


def _numpy():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Endpoint numerical admission requires allocated DTU HPC")
    import numpy as np
    return np


def _require(value, message):
    if not value:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seal(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _arrays(values):
    np = _numpy()
    _require(isinstance(values, dict) and set(values) == {"X", "E_upper"}, "Both named raw graph fields are required")
    result = {}
    for field, value in sorted(values.items()):
        array = np.asarray(value)
        _require(array.dtype == np.float32 and array.ndim == 2 and array.shape[0] >= GEOMETRY_ROWS
                 and array.shape[1] >= 2 and np.isfinite(array).all()
                 and (array >= 0).all() and (array <= 1).all()
                 and np.allclose(array.sum(-1), 1, atol=2e-6, rtol=0),
                 "Require unchanged finite FP32 categorical simplex rows")
        result[field] = array.copy()
    return result


def _record(array):
    raw = array.tobytes(order="C")
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(raw).hexdigest(), "base64": base64.b64encode(raw).decode("ascii")}


def _decode(record, dtype="float32"):
    np = _numpy()
    _require(record["dtype"] == dtype, "Plan array dtype changed")
    raw = base64.b64decode(record["base64"], validate=True)
    _require(hashlib.sha256(raw).hexdigest() == record["sha256"], "Plan array byte identity changed")
    result = np.frombuffer(raw, dtype=dtype).reshape(record["shape"]).copy()
    _require(np.isfinite(result).all(), "Nonfinite plan array")
    return result


def _snapshot(values):
    return {field: _record(value) for field, value in sorted(values.items())}


def _move(values, row, source, target, epsilon, sign):
    np = _numpy()
    point = values.copy()
    amount = np.float32(sign * epsilon)
    point[row, source] = np.float32(point[row, source] - amount)
    point[row, target] = np.float32(point[row, target] + amount)
    feasible = bool(np.isfinite(point).all() and (point >= 0).all() and (point <= 1).all()
                    and np.allclose(point.sum(-1), 1, atol=2e-6, rtol=0))
    changed = bool(point[row, source] != values[row, source] and point[row, target] != values[row, target])
    return point, feasible, changed


def _probe(values, row, source, target, epsilon):
    np = _numpy()
    minus, minus_safe, minus_changed = _move(values, row, source, target, epsilon, -1)
    plus, plus_safe, plus_changed = _move(values, row, source, target, epsilon, 1)
    central = bool(minus_safe and plus_safe and minus[row, target] > 0
                   and plus[row, source] > 0 and minus_changed and plus_changed)
    if central:
        scheme, left, right, direction = "central", minus, plus, 1
    else:
        positive_margin = min(float(values[row, source]), 1 - float(values[row, target]))
        negative_margin = min(float(values[row, target]), 1 - float(values[row, source]))
        direction = 1 if positive_margin >= negative_margin else -1
        point, feasible, changed = (plus, plus_safe, plus_changed) if direction == 1 else (minus, minus_safe, minus_changed)
        if feasible and changed:
            scheme, left, right = "one_sided_inward", values.copy(), point
        else:
            scheme, left, right = "unavailable", None, None
    result = {"epsilon": epsilon, "original_scale": epsilon in ORIGINAL_EPSILONS,
              "local_scale": epsilon in LOCAL_EPSILONS, "scheme": scheme,
              "inward_sign": direction, "legacy_central_interior_feasible": central,
              "original_gate_can_be_replaced": False,
              "both_nominal_points": {"minus": {"point": _record(minus), "feasible": minus_safe,
                                                  "both_entries_changed": minus_changed},
                                      "plus": {"point": _record(plus), "feasible": plus_safe,
                                                 "both_entries_changed": plus_changed}}}
    if scheme == "unavailable":
        result.update(left=None, right=None, actual_difference=None, baseline_displacements=[])
        return result
    difference = right.astype(np.float64) - left.astype(np.float64)
    displacements = ([left.astype(np.float64) - values.astype(np.float64),
                      right.astype(np.float64) - values.astype(np.float64)] if scheme == "central"
                     else [right.astype(np.float64) - values.astype(np.float64)])
    result.update(left=_record(left), right=_record(right), actual_difference=_record(difference),
                  effective_step=float((abs(difference[row, source]) + abs(difference[row, target])) / 2),
                  baseline_displacements=[_record(value) for value in displacements])
    return result


def plan(values, original_directions, *, original_direction_scope="saved_failed_endpoint"):
    """Predeclare all points before any primal/AD call, without changing inputs.

    Original directions contain only authenticated row/source/target contrasts.
    An original field whose old checker was never reached has an empty list.
    An entire original component whose probe was not reached uses the explicit
    original-probe-not-reached scope with empty lists. New future terminal points
    use the distinct not-preexisting scope, without inventing historical probes.
    """
    arrays = _arrays(values)
    _require(original_direction_scope in SCOPES and isinstance(original_directions, dict)
             and set(original_directions) == set(arrays), "Explicit original-direction scope/field lists required")
    if original_direction_scope == "saved_failed_endpoint":
        _require(any(original_directions.values()), "Saved failed endpoint needs an authentic known contrast")
    else:
        _require(not any(original_directions.values()), "Unreached or new endpoint cannot claim preexisting original contrasts")
    result = {"version": VERSION, "original_direction_scope": original_direction_scope,
              "original_failed_outcome_must_remain_unchanged": True, "baseline": _snapshot(arrays),
              "policy": {"original_epsilons": list(ORIGINAL_EPSILONS), "local_epsilons": list(LOCAL_EPSILONS),
                         "repeats": REPEATS, "relative_tolerance": RELATIVE_TOLERANCE,
                         "scalar_parity_atol": PARITY_ATOL, "gradient_repeat_atol": GRADIENT_REPEAT_ATOL,
                         "minimum_tangent": MINIMUM_TANGENT, "ulp_multiplier": ULP_MULTIPLIER,
                         "spread_multiplier": SPREAD_MULTIPLIER, "geometry_rows": GEOMETRY_ROWS,
                         "signal_to_floor": SIGNAL_TO_FLOOR,
                         "uncertainty_inside_relative_tolerance": True},
              "directions": [], "selection": "geometry_points_before_any_evaluator; precision_pair_before_perturbed_forwards"}
    epsilons = list(ORIGINAL_EPSILONS) + [value for value in LOCAL_EPSILONS if value not in ORIGINAL_EPSILONS]
    for field, values in arrays.items():
        requested = original_directions[field]
        _require(isinstance(requested, list) and len(requested) <= 4, "At most four authentic original contrasts per field")
        selected = {}
        for direction in requested:
            _require(isinstance(direction, dict) and set(direction) == {"row", "source", "target"}, "Malformed original contrast")
            row, source, target = (direction[key] for key in ("row", "source", "target"))
            _require(all(type(value) is int for value in (row, source, target))
                     and 0 <= row < values.shape[0] and 0 <= source < values.shape[1]
                     and 0 <= target < values.shape[1] and source != target, "Original contrast outside graph")
            key = (row, min(source, target), max(source, target))
            _require(key not in selected, "Duplicate original contrast")
            selected[key] = {**direction, "roles": ["authentic_original"]}
        candidates = []
        for row, probabilities in enumerate(values):
            ranked = []
            for source in range(values.shape[1]):
                for target in range(source + 1, values.shape[1]):
                    margin = min(float(probabilities[source]), float(probabilities[target]),
                                 1 - float(probabilities[source]), 1 - float(probabilities[target]))
                    inward = max(min(float(probabilities[source]), 1 - float(probabilities[target])),
                                 min(float(probabilities[target]), 1 - float(probabilities[source])))
                    ranked.append((margin, inward, float(probabilities[source] + probabilities[target]),
                                   -source, -target, source, target))
            best = max(ranked)
            candidates.append((best[0], best[1], best[2], -row, best[-2], best[-1], row))
        chosen = sorted(candidates, reverse=True)[:GEOMETRY_ROWS]
        for rank, item in enumerate(chosen):
            row, source, target = item[-1], item[-3], item[-2]
            key = (row, min(source, target), max(source, target))
            if key not in selected:
                selected[key] = {"row": row, "source": source, "target": target, "roles": []}
            selected[key]["roles"].append("geometry_max_margin" if rank == 0 else "geometry_distinct_row")
            selected[key]["geometry_rank"] = rank
        for direction in selected.values():
            result["directions"].append({"field": field, **direction,
                "probes": [_probe(values, direction["row"], direction["source"], direction["target"], epsilon)
                           for epsilon in epsilons]})
    result["plan_sha256"] = _seal(result)
    return result


def _scalar(value):
    np = _numpy()
    components = None
    if isinstance(value, dict):
        _require(set(value) == {'value', 'native_components'}, 'Malformed component-resolved scalar')
        components = value['native_components']; value = value['value']
        _require(isinstance(components, dict) and set(components) == {'affinity', 'contact', 'confidence'},
                 'All three declared native DBMol components required')
    array = np.asarray(value)
    _require(array.shape == () and array.dtype in (np.dtype("float32"), np.dtype("float64"))
             and np.isfinite(array), "Evaluator must return one finite FP32/FP64 scalar")
    fp32 = np.float32(array)
    _require(np.isfinite(fp32), "Native scalar is not FP32 representable")
    # A Python/FP64 wrapper cannot manufacture precision absent from native FP32.
    record = {"value": float(array), "dtype": str(array.dtype),
              "resolution_ulp": max(float(abs(np.spacing(array))), float(abs(np.spacing(fp32))))}
    if components is not None:
        parts = {name: _scalar(components[name]) for name in ('affinity', 'contact', 'confidence')}
        weights = {'affinity': 1., 'contact': 2.5, 'confidence': 2.5}
        recomposed = sum(weights[k] * v['value'] for k, v in parts.items())
        _require(abs(recomposed-float(array)) <= PARITY_ATOL, 'Independent total/component scalar parity failed')
        propagated = sum(weights[k] * v['resolution_ulp'] for k, v in parts.items())
        record.update(resolution_ulp=max(record['resolution_ulp'], propagated), native_components=parts,
                      component_weights=weights, recomposed_value=recomposed,
                      resolution_scope='Weighted native component resolution; full-FP64 trunk is not claimed')
    return record


def _invoke(callback, arrays):
    payload = {field: value.copy() for field, value in arrays.items()}
    before = _snapshot(payload)
    result = callback(payload)
    _require(_snapshot(payload) == before, "Evaluator mutated a diagnostic graph payload")
    return result


def _identity(callback):
    value = callback()
    _require(isinstance(value, dict) and value, "Explicit frozen identity dictionary required")
    return json.loads(_json(value))


def _eligibility(saved_plan, gradients, gradient_repeats, floor, baseline_evidence_sha256):
    np = _numpy()
    rows = []
    for direction in saved_plan["directions"]:
        field, row, source, target = (direction[key] for key in ("field", "row", "source", "target"))
        tangents = [float(value[field][row, target] - value[field][row, source]) for value in gradient_repeats]
        tangent = float(gradients[field][row, target] - gradients[field][row, source])
        stable = max(tangents) - min(tangents) <= RELATIVE_TOLERANCE * max(abs(tangent), MINIMUM_TANGENT)
        probes = []
        for probe in direction["probes"]:
            expected = None
            predicted_signals = []
            if probe["actual_difference"] is not None:
                difference = _decode(probe["actual_difference"], "float64")
                expected = float(np.sum(gradients[field] * difference, dtype=np.float64))
                predicted_signals = [abs(float(np.sum(gradients[field] * _decode(value, "float64"), dtype=np.float64)))
                                     for value in probe["baseline_displacements"]]
            eligible = bool(probe["local_scale"] and expected is not None and stable
                            and abs(tangent) > MINIMUM_TANGENT and predicted_signals
                            and all(signal > SIGNAL_TO_FLOOR * floor for signal in predicted_signals))
            probes.append({"epsilon": probe["epsilon"], "predicted_scalar_difference": expected,
                           "predicted_endpoint_signals": predicted_signals, "precision_eligible": eligible})
        by_epsilon = {probe["epsilon"]: probe for probe in probes}
        pairs = [[first, second] for first, second in zip(LOCAL_EPSILONS, LOCAL_EPSILONS[1:])
                 if by_epsilon[first]["precision_eligible"] and by_epsilon[second]["precision_eligible"]]
        selected = pairs[-1] if pairs else None
        rows.append({"field": field, "row": row, "source": source, "target": target,
                     "roles": direction["roles"], "nominal_tangent": tangent,
                     "tangent_repeats": tangents, "tangent_repeat_stable": stable,
                     "probes": probes, "selected_local_pair": selected})
    result = {"version": VERSION, "plan_sha256": saved_plan["plan_sha256"],
              "baseline_evidence_sha256": baseline_evidence_sha256,
              "selection_rule": "finest_adjacent_fixed_local_pair_with_both_signals_above_predeclared_precision_budget",
              "baseline_resolution_floor": floor, "local_epsilons": list(LOCAL_EPSILONS), "directions": rows,
              "selection_uses_perturbed_values": False, "pair_fallback_after_FD": False}
    result["eligibility_sha256"] = _seal(result)
    return result


def agreement(observed, predicted, uncertainty):
    """Charge uncertainty inside tolerance; unresolved is never agreement."""
    _require(all(type(value) in (int, float) for value in (observed, predicted, uncertainty))
             and uncertainty >= 0, "Finite scalar interval required")
    import math
    _require(all(math.isfinite(value) for value in (observed, predicted, uncertainty)), "Nonfinite interval")
    residual = abs(observed - predicted)
    low = max(max(0.0, abs(observed) - uncertainty), abs(predicted))
    high = max(abs(observed) + uncertainty, abs(predicted))
    same_sign = abs(observed) > uncertainty and observed * predicted > 0
    passed = bool(same_sign and residual + uncertainty <= RELATIVE_TOLERANCE * low)
    wrong = bool(max(0.0, residual - uncertainty) > RELATIVE_TOLERANCE * high)
    return {"agrees": passed, "comparison_resolved": bool(passed or wrong),
            "relative_error": residual / max(abs(observed), abs(predicted)) if max(abs(observed), abs(predicted)) else None,
            "absolute_residual": residual, "uncertainty_charged_inside_tolerance": uncertainty,
            "outcome": "agreement" if passed else "resolved_disagreement" if wrong else "unresolved_precision"}


def evaluate(saved_plan, primal, autodiff, frozen_identity, *, seal_callback=None):
    """Evaluate fixed points; return separate new-child admission evidence.

    ``primal(dict)`` returns an independent native scalar; ``autodiff(dict)``
    returns scalar plus named full raw gradients. ``frozen_identity()`` returns
    JSON pins. Optional ``seal_callback(record)`` durably writes the precision
    selection before any perturbed forward. It never authorizes an optimizer.
    """
    np = _numpy()
    _require(saved_plan.get("version") == VERSION and saved_plan.get("plan_sha256") == _seal(
        {key: value for key, value in saved_plan.items() if key != "plan_sha256"}), "Immutable admission plan changed")
    arrays = _arrays({field: _decode(record) for field, record in saved_plan["baseline"].items()})
    original = {field: [] for field in arrays}
    for direction in saved_plan["directions"]:
        if "authentic_original" in direction["roles"]:
            original[direction["field"]].append({key: direction[key] for key in ("row", "source", "target")})
    canonical = plan(arrays, original, original_direction_scope=saved_plan["original_direction_scope"])
    _require(canonical == saved_plan, "Admission plan is not the fixed versioned geometry policy")
    report = {"version": VERSION, "status": "failed", "can_admit_new_child": False,
              "original_run_failed_preserved": True, "original_gate_replaced": False,
              "can_certify_original_study": False, "toy_pass_is_native_validation": False,
              "plan": deepcopy(saved_plan), "baseline_checks": {}, "directions": [],
              "field_outcomes": {}, "failures": [], "evaluation_sequence": [],
              "interpretation": "Prospective child admission only; original coarse failures remain verbatim evidence"}
    frozen_before = None
    try:
        frozen_before = _identity(frozen_identity)
        report["frozen_before"] = frozen_before
        primals, ad_scalars, repeats = [], [], []
        for number in range(REPEATS):
            primals.append(_scalar(_invoke(primal, arrays)))
            report["evaluation_sequence"].append({"phase": "baseline_primal", "repeat": number})
        for number in range(REPEATS):
            value = _invoke(autodiff, arrays)
            _require(isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict)
                     and set(value[1]) == set(arrays), "AD must return scalar and both raw gradient fields")
            ad_scalars.append(_scalar(value[0]))
            gradients = {}
            for field, baseline in arrays.items():
                derivative = np.asarray(value[1][field])
                _require(derivative.shape == baseline.shape and derivative.dtype.kind == "f"
                         and np.isfinite(derivative).all(), "Invalid finite raw AD gradient: " + field)
                gradients[field] = derivative.astype(np.float64).copy()
            repeats.append(gradients)
            report["evaluation_sequence"].append({"phase": "baseline_autodiff", "repeat": number})
        samples = primals + ad_scalars
        primal_mean = sum(value["value"] for value in primals) / REPEATS
        spread = max(value["value"] for value in samples) - min(value["value"] for value in samples)
        ulp = max(value["resolution_ulp"] for value in samples)
        floor = ULP_MULTIPLIER * ulp + SPREAD_MULTIPLIER * spread
        scalar_parity = all(abs(value["value"] - primal_mean) <= PARITY_ATOL for value in ad_scalars)
        gradient_repeat = {field: float(max(np.max(np.abs(value[field] - repeats[0][field])) for value in repeats))
                           for field in arrays}
        report["baseline_checks"] = {"primal_repeats": primals, "autodiff_scalar_repeats": ad_scalars,
            "primal_mean": primal_mean, "all_repeat_spread": spread, "resolution_ulp": ulp,
            "resolution_floor": floor, "scalar_parity_passed": scalar_parity,
            "gradient_max_repeat_differences": gradient_repeat,
            "gradient_repeat_passed": all(value <= GRADIENT_REPEAT_ATOL for value in gradient_repeat.values()),
            "gradient_repeats": [_snapshot(value) for value in repeats]}
        _require(scalar_parity, "Independent primal/AD scalar parity failed")
        _require(report["baseline_checks"]["gradient_repeat_passed"], "Raw AD repeats are unstable")
        gradients = {field: np.mean([value[field] for value in repeats], axis=0) for field in arrays}
        report["baseline_checks"]["raw_gradient_norms"] = {
            field: float(np.linalg.norm(value)) for field, value in gradients.items()}
        eligibility = _eligibility(saved_plan, gradients, repeats, floor, _seal(report["baseline_checks"]))
        report["eligibility"] = deepcopy(eligibility)
        report["eligibility_seal"] = eligibility["eligibility_sha256"]
        report["evaluation_sequence"].append({"phase": "eligibility_sealed", "sha256": report["eligibility_seal"]})
        if seal_callback is not None:
            seal_callback(deepcopy(eligibility))
        for number, (direction, selected) in enumerate(zip(saved_plan["directions"], eligibility["directions"])):
            _require(eligibility["eligibility_sha256"] == _seal({key: value for key, value in eligibility.items()
                                                               if key != "eligibility_sha256"}), "Sealed pair selection changed")
            field = direction["field"]
            observed = {key: deepcopy(direction[key]) for key in ("field", "row", "source", "target", "roles")}
            observed.update(selected_local_pair=selected["selected_local_pair"], probes=[])
            for probe, predicted in zip(direction["probes"], selected["probes"]):
                measurement = {"epsilon": probe["epsilon"], "scheme": probe["scheme"],
                    "original_scale": probe["original_scale"], "local_scale": probe["local_scale"],
                    "legacy_central_interior_feasible": probe["legacy_central_interior_feasible"], **deepcopy(predicted)}
                if probe["scheme"] == "unavailable":
                    measurement.update(evaluated=False, comparison_resolved=False, agrees=False,
                                       outcome="boundary_unavailable", reason="No feasible represented fixed point; no clipping or projection")
                else:
                    endpoints = []
                    for endpoint in ("left", "right"):
                        if endpoint == "left" and probe["scheme"] == "one_sided_inward":
                            endpoints.append(deepcopy(primals))
                            continue
                        values = {key: value.copy() for key, value in arrays.items()}
                        values[field] = _decode(probe[endpoint])
                        repetitions = [_scalar(_invoke(primal, values)) for _ in range(REPEATS)]
                        endpoints.append(repetitions)
                        report["evaluation_sequence"].append({"phase": "perturbed_primal", "direction": number,
                                                              "epsilon": probe["epsilon"], "endpoint": endpoint, "repeats": REPEATS})
                    left, right = [sum(value["value"] for value in values) / REPEATS for values in endpoints]
                    actual = right - left
                    expected = predicted["predicted_scalar_difference"]
                    endpoint_spread = max(spread, *[max(value["value"] for value in values) - min(value["value"] for value in values)
                                                  for values in endpoints])
                    endpoint_ulp = max(ulp, *[value["resolution_ulp"] for values in endpoints for value in values])
                    resolution = ULP_MULTIPLIER * endpoint_ulp + SPREAD_MULTIPLIER * endpoint_spread
                    interval = agreement(actual, expected, resolution)
                    measurement.update(evaluated=True, **interval,
                        endpoint_primal_repeats=endpoints, observed_scalar_difference=actual,
                        actual_FP32_displacement_JVP=expected, effective_step=probe["effective_step"],
                        observed_derivative=actual / probe["effective_step"],
                        represented_derivative=expected / probe["effective_step"],
                        resolution_floor=resolution, endpoint_repeat_spread=endpoint_spread)
                observed["probes"].append(measurement)
            pair = selected["selected_local_pair"]
            by_epsilon = {probe["epsilon"]: probe for probe in observed["probes"]}
            pair_pass = bool(pair and all(by_epsilon[value]["agrees"] for value in pair))
            finer_disagreements = [probe["epsilon"] for probe in observed["probes"]
                                   if pair and probe["local_scale"] and probe["epsilon"] < min(pair)
                                   and probe["comparison_resolved"] and not probe["agrees"]]
            unselected_disagreements = [probe["epsilon"] for probe in observed["probes"]
                                       if not pair and probe["local_scale"]
                                       and probe["comparison_resolved"] and not probe["agrees"]]
            coarser_disagreements = [probe["epsilon"] for probe in observed["probes"]
                                    if pair and probe["local_scale"] and probe["epsilon"] > max(pair)
                                    and probe["comparison_resolved"] and not probe["agrees"]]
            # Every baseline-eligible direction must pass; no alternative pair is
            # selected after the observed FD values. Low-signal extras are not passes.
            observed.update(pair_passed=pair_pass, finer_resolved_disagreements=finer_disagreements,
                unselected_resolved_disagreements=unselected_disagreements,
                coarser_local_resolved_disagreements=coarser_disagreements,
                admission_evidence=bool(pair_pass and not finer_disagreements and not unselected_disagreements),
                outcome="passed_local_admission" if pair_pass and not finer_disagreements
                        else "failed_selected_pair" if pair else "noncertifying_precision_or_zero_tangent",
                coarse_observations_replace_original_gate=False)
            report["directions"].append(observed)
            if pair and (not pair_pass or finer_disagreements):
                report["failures"].append("Preselected local pair/finer resolved comparison failed: " + str(number))
            if unselected_disagreements:
                report["failures"].append("Resolved local contradiction in noncertifying direction: " + str(number))
            if "authentic_original" in direction["roles"] and not observed["admission_evidence"]:
                report["failures"].append("Authentic original contrast lacks informative passed local proof: " + str(number))
        for field in arrays:
            directions = [value for value in report["directions"] if value["field"] == field]
            meaningful = [value for value in directions if value["admission_evidence"]]
            meaningful_geometry = [value for value in meaningful
                                   if any(role.startswith("geometry_") for role in value["roles"])]
            passed = bool(meaningful_geometry)
            report["field_outcomes"][field] = {"passed": passed, "meaningful_passed_directions": len(meaningful),
                "meaningful_passed_geometry_directions": len(meaningful_geometry),
                "geometry_directions": sum(any(role.startswith("geometry_") for role in value["roles"]) for value in directions),
                "noncertifying_directions": sum(not value["admission_evidence"] for value in directions)}
            if not passed:
                report["failures"].append("No reliable informative geometry local pair for raw field: " + field)
    except Exception as error:
        report["failures"].append(type(error).__name__ + ": " + str(error))
    finally:
        try:
            report["frozen_after"] = _identity(frozen_identity)
            report["frozen_unchanged"] = frozen_before is not None and report["frozen_after"] == frozen_before
        except Exception as error:
            report["frozen_unchanged"] = False
            report["failures"].append("Frozen after-check failed: " + repr(error))
        if not report["frozen_unchanged"]:
            report["failures"].append("Frozen identity changed or could not be verified")
    report["can_admit_new_child"] = bool(not report["failures"] and report["field_outcomes"]
                                         and all(value["passed"] for value in report["field_outcomes"].values()))
    report["status"] = "passed" if report["can_admit_new_child"] else "failed"
    report["receipt_sha256"] = _seal(report)
    return report
