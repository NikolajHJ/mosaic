"""Separate FP64-reference validation of an unchanged FP32 Protenix endpoint.

This is a prospective check, never a recertification of a failed saved study.
Every native operation of the independent reference must run in FP64; casting
an FP32 result is insufficient. The allocated caller authenticates that fact.
The scalar floor below measures FP64 resolution/repeat spread, not a proven
bound on an arbitrary native network's arithmetic error. Independent primal
finite differences remain mandatory. No optimizer, graph repair, projection,
scale search after finite differences, or submission exists in this module.
"""

import base64
from copy import deepcopy
import hashlib
import json
import os


VERSION = "protenix-separate-precision-admission-v1"
EPSILONS = (0.003, 0.001, 0.0003, 0.0001, 0.00003)
ORIGINAL_EPSILONS = (0.01, 0.003, 0.001)
REPEATS = 3
STRONG_ROWS = 2
RELATIVE_TOLERANCE = 0.15
LOCALITY_TOLERANCE = 0.15
MINIMUM_TANGENT = 1e-7
REFERENCE_ULPS = 64.0
REFERENCE_SPREAD = 8.0
SIGNAL_TO_FLOOR = 32.0
PRODUCTION_PARITY_ATOL = 2e-5
REFERENCE_PARITY_ATOL = 1e-10
SCOPES = ("saved_failed_endpoint", "not_preexisting_at_new_endpoint")
_FIELDS = ("X", "E_upper")
_PRECISION = {"full_model_float64": True, "parameters_float64": True,
              "native_features_float64": True, "native_operations_float64": True,
              "final_reduction_only": False}


def _numpy():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Precision admission numerical work requires allocated DTU HPC")
    import numpy as np
    return np


def _require(value, message):
    if not value:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seal(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _record(value):
    raw = value.tobytes(order="C")
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(raw).decode("ascii")}


def _snapshot(values):
    return {name: _record(value) for name, value in sorted(values.items())}


def _arrays(values):
    np = _numpy()
    _require(isinstance(values, dict) and set(values) == set(_FIELDS),
             "Both named raw graph fields are required")
    result = {}
    for name in _FIELDS:
        value = np.asarray(values[name])
        _require(value.dtype == np.float32 and value.ndim == 2
                 and value.shape[0] >= STRONG_ROWS and value.shape[1] >= 2
                 and np.isfinite(value).all() and (value >= 0).all()
                 and (value <= 1).all()
                 and np.allclose(value.sum(-1), 1, rtol=0, atol=2e-6),
                 "Require exact finite FP32 saved categorical rows: " + name)
        result[name] = value.copy()
    return result


def _identity(callback):
    value = callback()
    _require(isinstance(value, dict) and value, "Explicit frozen identity dictionary required")
    return json.loads(_json(value))


def _invoke(callback, values):
    payload = {name: value.copy() for name, value in values.items()}
    before = _snapshot(payload)
    result = callback(payload)
    _require(_snapshot(payload) == before, "Evaluator mutated a diagnostic graph")
    return result


def _scalar(value, dtype):
    np = _numpy()
    array = np.asarray(value)
    _require(array.shape == () and array.dtype == np.dtype(dtype) and np.isfinite(array),
             "Native scalar must be finite " + dtype)
    return {"value": float(array), "dtype": dtype,
            "resolution_ulp": float(abs(np.spacing(array)))}


def _ad(callback, values, dtype):
    np = _numpy()
    result = _invoke(callback, values)
    _require(isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict)
             and set(result[1]) == set(_FIELDS), "AD must return scalar and both full raw gradients")
    scalar, gradients = _scalar(result[0], dtype), {}
    for name, baseline in values.items():
        value = np.asarray(result[1][name])
        _require(value.shape == baseline.shape and value.dtype == np.dtype(dtype)
                 and np.isfinite(value).all(), "Invalid native raw gradient: " + name)
        gradients[name] = value.copy()
    return scalar, gradients


def _baseline(primal, autodiff, values, dtype):
    np = _numpy()
    primals = [_scalar(_invoke(primal, values), dtype) for _ in range(REPEATS)]
    derivatives = [_ad(autodiff, values, dtype) for _ in range(REPEATS)]
    gradients = {name: np.mean([row[1][name].astype(np.float64) for row in derivatives], axis=0)
                 for name in _FIELDS}
    repeats = {name: float(max(np.max(np.abs(row[1][name].astype(np.float64)
                                              - derivatives[0][1][name].astype(np.float64)))
                                for row in derivatives)) for name in _FIELDS}
    scalar_rows = primals + [row[0] for row in derivatives]
    spread = max(row["value"] for row in scalar_rows) - min(row["value"] for row in scalar_rows)
    parity = REFERENCE_PARITY_ATOL if dtype == "float64" else PRODUCTION_PARITY_ATOL
    mean = float(np.mean([row["value"] for row in primals], dtype=np.float64))
    report = {"primals": primals, "autodiff_scalars": [row[0] for row in derivatives],
              "primal_mean": mean, "all_repeat_spread": spread,
              "resolution_ulp": max(row["resolution_ulp"] for row in scalar_rows),
              "gradient_repeat_differences": repeats,
              "gradient_norms": {name: float(np.linalg.norm(value)) for name, value in gradients.items()},
              "gradient_repeats": [_snapshot(row[1]) for row in derivatives],
              "scalar_parity_passed": all(abs(row[0]["value"] - mean) <= parity for row in derivatives),
              "gradient_repeats_passed": all(value <= parity for value in repeats.values())}
    _require(report["scalar_parity_passed"], dtype + " independent primal/AD scalar parity failed")
    _require(report["gradient_repeats_passed"], dtype + " repeated raw gradients are unstable")
    return report, gradients


def _point(values, row, source, target, epsilon, sign):
    np = _numpy()
    result = values.copy()
    amount = result.dtype.type(sign * epsilon)
    result[row, source] = result.dtype.type(result[row, source] - amount)
    result[row, target] = result.dtype.type(result[row, target] + amount)
    feasible = bool(np.isfinite(result).all() and (result >= 0).all() and (result <= 1).all()
                    and np.allclose(result.sum(-1), values.sum(-1), rtol=0,
                                    atol=2e-6 if values.dtype == np.float32 else 1e-14))
    changed = bool(result[row, source] != values[row, source]
                   and result[row, target] != values[row, target])
    return result, feasible and changed


def _probe(values, row, source, target, epsilon):
    np = _numpy()
    minus, minus_ok = _point(values, row, source, target, epsilon, -1)
    plus, plus_ok = _point(values, row, source, target, epsilon, 1)
    central = bool(minus_ok and plus_ok and minus[row, target] > 0 and plus[row, source] > 0)
    if central:
        left, right, scheme = minus, plus, "central"
    else:
        positive = min(float(values[row, source]), 1 - float(values[row, target]))
        negative = min(float(values[row, target]), 1 - float(values[row, source]))
        point, ok = (plus, plus_ok) if positive >= negative else (minus, minus_ok)
        left, right, scheme = (values.copy(), point, "one_sided_inward") if ok else (None, None, "unavailable")
    delta = right.astype(np.float64) - left.astype(np.float64) if left is not None else None
    displacements = ([] if left is None else [left.astype(np.float64) - values.astype(np.float64),
                                              right.astype(np.float64) - values.astype(np.float64)]
                     if central else [delta])
    return {"epsilon": epsilon, "scheme": scheme, "left": left, "right": right,
            "delta": delta, "displacements": displacements}


def _prediction(gradient, probe):
    np = _numpy()
    return None if probe["delta"] is None else float(np.sum(gradient * probe["delta"], dtype=np.float64))


def _encoded_probe(probe):
    return {"epsilon": probe["epsilon"], "scheme": probe["scheme"],
            "left": None if probe["left"] is None else _record(probe["left"]),
            "right": None if probe["right"] is None else _record(probe["right"]),
            "actual_displacement": None if probe["delta"] is None else _record(probe["delta"])}


def _selected_probes(values, gradient, row, source, target, floor):
    np = _numpy()
    tangent = float(gradient[row, target] - gradient[row, source])
    probes = [_probe(values, row, source, target, epsilon) for epsilon in EPSILONS]
    eligible = []
    for probe in probes:
        signals = [abs(float(np.sum(gradient * displacement, dtype=np.float64)))
                   for displacement in probe["displacements"]]
        probe["predicted_delta"] = _prediction(gradient, probe)
        probe["predicted_endpoint_signals"] = signals
        probe["eligible"] = bool(abs(tangent) > MINIMUM_TANGENT and signals
                                 and all(signal > SIGNAL_TO_FLOOR * floor for signal in signals))
        eligible.append(probe["eligible"])
    pairs = [list(EPSILONS[index:index + 2]) for index in range(len(EPSILONS) - 1)
             if eligible[index] and eligible[index + 1]]
    return probes, pairs[-1] if pairs else None


def _plan(values, gradients, production_gradients, original_directions, scope, floor):
    np = _numpy()
    _require(scope in SCOPES and isinstance(original_directions, dict)
             and set(original_directions) == set(_FIELDS), "Explicit historical direction scope required")
    if scope == "saved_failed_endpoint":
        _require(bool(original_directions["X"]), "Saved endpoint requires the authentic original X contrast")
    else:
        _require(not any(original_directions.values()), "A new endpoint cannot invent historical contrasts")
    rows = []
    for field in _FIELDS:
        selected = {}
        for direction in original_directions[field]:
            _require(isinstance(direction, dict) and set(direction) == {"row", "source", "target"},
                     "Malformed authentic contrast")
            row, source, target = (direction[key] for key in ("row", "source", "target"))
            _require(all(type(value) is int for value in (row, source, target))
                     and 0 <= row < values[field].shape[0]
                     and 0 <= source < values[field].shape[1] and 0 <= target < values[field].shape[1]
                     and source != target, "Authentic contrast outside saved graph")
            key = row, min(source, target), max(source, target)
            _require(key not in selected, "Duplicate authentic contrast")
            selected[key] = {**direction, "roles": ["authentic_original"]}
        candidates = []
        for row in range(values[field].shape[0]):
            choices = []
            for source in range(values[field].shape[1]):
                for target in range(source + 1, values[field].shape[1]):
                    probes, pair = _selected_probes(values[field], gradients[field], row, source, target, floor)
                    if pair is not None:
                        strongest = max(abs(probe["predicted_delta"]) for probe in probes if probe["eligible"])
                        margin = min(float(values[field][row, source]), float(values[field][row, target]))
                        choices.append((strongest, margin, -source, -target, source, target))
            if choices:
                best = max(choices)
                candidates.append((best[0], best[1], -row, best[-2], best[-1], row))
        _require(len(candidates) >= STRONG_ROWS, "Insufficient meaningful distinct-row raw contrasts: " + field)
        for rank, choice in enumerate(sorted(candidates, reverse=True)[:STRONG_ROWS]):
            row, source, target = choice[-1], choice[-3], choice[-2]
            key = row, min(source, target), max(source, target)
            if key not in selected:
                selected[key] = {"row": row, "source": source, "target": target, "roles": []}
            selected[key]["roles"].append("strong_distinct_row_" + str(rank))
        for direction in selected.values():
            row, source, target = (direction[key] for key in ("row", "source", "target"))
            probes, pair = _selected_probes(values[field], gradients[field], row, source, target, floor)
            rows.append({"field": field, **direction, "selected_pair": pair,
                         "reference_tangent": float(gradients[field][row, target] - gradients[field][row, source]),
                         "production_tangent": float(production_gradients[field][row, target]
                                                       - production_gradients[field][row, source]),
                         "probes": [{**_encoded_probe(probe), "eligible": probe["eligible"],
                                     "predicted_delta": probe["predicted_delta"],
                                     "predicted_endpoint_signals": probe["predicted_endpoint_signals"]}
                                    for probe in probes]})
    result = {"version": VERSION, "scope": scope, "baseline": _snapshot(values),
              "reference_gradient": _snapshot(gradients), "production_gradient": _snapshot(production_gradients),
              "reference_floor": floor, "directions": rows,
              "selection": "baseline_reference_AD_and_feasible_geometry_only_before_perturbed_callbacks",
              "selection_uses_FD": False, "after_result_fallback": False}
    result["plan_sha256"] = _seal(result)
    return result


def _agreement(observed, predicted, uncertainty):
    residual = abs(observed - predicted)
    denominator_low = max(max(0.0, abs(observed) - uncertainty), abs(predicted))
    denominator_high = max(abs(observed) + uncertainty, abs(predicted))
    same_sign = bool(abs(observed) > uncertainty and observed * predicted > 0)
    passed = bool(same_sign and residual + uncertainty <= RELATIVE_TOLERANCE * denominator_low)
    resolved_wrong = bool(max(0.0, residual - uncertainty) > RELATIVE_TOLERANCE * denominator_high)
    return {"observed_delta": observed, "predicted_delta": predicted, "uncertainty": uncertainty,
            "absolute_residual": residual, "relative_error": residual / max(abs(observed), abs(predicted))
            if max(abs(observed), abs(predicted)) else None, "same_resolved_sign": same_sign,
            "passed": passed, "outcome": "agreement" if passed else "resolved_disagreement"
            if resolved_wrong else "unresolved_precision"}


def evaluate(values, production_primal, production_autodiff, reference_primal, reference_autodiff,
             frozen_identity, *, original_directions, scope="saved_failed_endpoint", reference_precision,
             original_observations=None, seal_callback=None):
    """Validate without updating the exact saved FP32 graph.

    Production callbacks receive copied FP32 raw fields, reference callbacks
    receive FP64 raw fields. The baseline promotion is exactly reversible.
    Reference perturbations are FP64 diagnostic points, never saved molecules.
    ``reference_precision`` claims require independent native caller proof;
    dtype checks alone cannot authenticate an entire model's execution.
    """
    np = _numpy()
    arrays = _arrays(values)
    baseline32 = _snapshot(arrays)
    promoted = {name: value.astype(np.float64) for name, value in arrays.items()}
    report = {"version": VERSION, "status": "failed", "can_admit_new_child": False,
              "can_certify_original_study": False, "original_failed_outcome_preserved": True,
              "optimization_updates": 0, "scope": scope, "baseline_fp32": baseline32,
              "baseline_reference": _snapshot(promoted), "reference_precision": deepcopy(reference_precision),
              "original_observations": deepcopy(original_observations),
              "directions": [], "field_outcomes": {}, "failures": [], "evaluation_sequence": [],
              "floor_scope": "FP64_scalar_resolution_and_repeat_spread_not_a_proven_native_forward_error_bound",
              "policy": {"epsilons": list(EPSILONS), "original_epsilons": list(ORIGINAL_EPSILONS),
                         "relative_tolerance": RELATIVE_TOLERANCE, "locality_tolerance": LOCALITY_TOLERANCE,
                         "strong_distinct_rows_per_field": STRONG_ROWS, "reference_ulps": REFERENCE_ULPS,
                         "reference_spread_multiplier": REFERENCE_SPREAD, "signal_to_floor": SIGNAL_TO_FLOOR,
                         "repeats": REPEATS, "minimum_tangent": MINIMUM_TANGENT,
                         "reference_is_separate_execution_not_production_dtype_change": True}}
    before = None
    try:
        _require(isinstance(reference_precision, dict) and all(reference_precision.get(key) is value
                                                             for key, value in _PRECISION.items()),
                 "Authenticated full-model FP64 reference required; final-reduction casts are insufficient")
        _require(all(np.array_equal(promoted[name].astype(np.float32), value)
                     for name, value in arrays.items()), "Reference baseline is not exact lossless promotion")
        before = _identity(frozen_identity)
        report["frozen_before"] = before
        production, production_gradients = _baseline(production_primal, production_autodiff, arrays, "float32")
        reference, reference_gradients = _baseline(reference_primal, reference_autodiff, promoted, "float64")
        report["baselines"] = {"production": production, "reference": reference}
        _require(abs(production["primal_mean"] - reference["primal_mean"]) <= PRODUCTION_PARITY_ATOL,
                 "Production/reference baseline scalar parity failed")
        _require(all(value > MINIMUM_TANGENT for value in production["gradient_norms"].values()),
                 "Production raw field has no meaningful gradient")
        floor = REFERENCE_ULPS * reference["resolution_ulp"] + REFERENCE_SPREAD * reference["all_repeat_spread"]
        planned = _plan(promoted, reference_gradients, production_gradients, original_directions, scope, floor)
        report["plan"] = deepcopy(planned)
        report["evaluation_sequence"].append({"phase": "baseline_selection_sealed", "sha256": planned["plan_sha256"]})
        if seal_callback is not None:
            seal_callback(deepcopy(planned))
        reference_cache = {}

        for number, direction in enumerate(planned["directions"]):
            _require(planned["plan_sha256"] == _seal({key: value for key, value in planned.items()
                                                     if key != "plan_sha256"}), "Predeclared selection changed")
            field, row, source, target = (direction[key] for key in ("field", "row", "source", "target"))
            result = {key: deepcopy(direction[key]) for key in ("field", "row", "source", "target", "roles", "selected_pair")}
            result.update(selected_measurements=[], original_coarse_observations=[])
            pair = direction["selected_pair"]
            if pair is None:
                report["failures"].append("Mandatory contrast has no meaningful fixed adjacent pair: " + str(number))
            for epsilon in pair or []:
                probe = _probe(promoted[field], row, source, target, epsilon)
                _require(probe["delta"] is not None, "Preselected reference point became unavailable")
                endpoint_rows, endpoint_gradients, endpoint_ad_scalars = [], [], []
                for endpoint in ("left", "right"):
                    point = {name: value.copy() for name, value in promoted.items()}
                    point[field] = probe[endpoint].copy()
                    # One independent scalar/AD pair per new point. The native
                    # caller may share one full value-and-grad execution across
                    # these two callbacks; readback is not a repeat-spread test.
                    point_key = _seal(_snapshot(point))
                    if point_key not in reference_cache:
                        reference_cache[point_key] = [_scalar(_invoke(reference_primal, point), "float64")]
                    endpoint_rows.append(reference_cache[point_key])
                    ad_scalar, gradient = _ad(reference_autodiff, point, "float64")
                    _require(abs(ad_scalar["value"] - reference_cache[point_key][0]["value"])
                             <= REFERENCE_PARITY_ATOL,
                             "Perturbed reference independent primal/AD scalar parity failed")
                    endpoint_ad_scalars.append(ad_scalar)
                    endpoint_gradients.append(gradient)
                observed = float(np.mean([value["value"] for value in endpoint_rows[1]], dtype=np.float64)
                                 - np.mean([value["value"] for value in endpoint_rows[0]], dtype=np.float64))
                endpoint_ulp = max(floor / REFERENCE_ULPS,
                                   *[value["resolution_ulp"] for rows in endpoint_rows for value in rows])
                uncertainty = max(floor, REFERENCE_ULPS * endpoint_ulp)
                predicted = _prediction(reference_gradients[field], probe)
                production_predicted = _prediction(production_gradients[field], probe)
                fd = _agreement(observed, predicted, uncertainty)
                gradient_parity = _agreement(production_predicted, predicted, 0.0)
                endpoint_predictions = [_prediction(gradient[field], probe) for gradient in endpoint_gradients]
                locality = all(abs(value - predicted) <= LOCALITY_TOLERANCE * max(abs(value), abs(predicted))
                               and value * predicted > 0 for value in endpoint_predictions)
                passed = bool(fd["passed"] and gradient_parity["passed"] and locality)
                result["selected_measurements"].append({"epsilon": epsilon, "probe": _encoded_probe(probe),
                    "independent_reference_primal_values": endpoint_rows, "reference_fd": fd,
                    "reference_endpoint_AD_scalars": endpoint_ad_scalars,
                    "point_readback_is_independent_repeat_test": False,
                    "production_vs_reference_AD": gradient_parity,
                    "reference_endpoint_displacement_predictions": endpoint_predictions,
                    "reference_endpoint_locality_passed": locality, "passed": passed})
                report["evaluation_sequence"].append({"phase": "fixed_reference_FD", "direction": number, "epsilon": epsilon})
            # Original native FP32 observations are immutable external evidence.
            # Never reload the production model after the separate FP64 worker
            # begins, or invent current coarse scalar observations from casts.
            for epsilon in ORIGINAL_EPSILONS:
                probe = _probe(arrays[field], row, source, target, epsilon)
                observation = {"epsilon": epsilon, "probe": _encoded_probe(probe),
                               "certifying": False, "replaces_original_gate": False,
                               "evaluated": False, "production_model_reloaded": False,
                               "external_evidence": deepcopy(original_observations),
                               "outcome": "preserved_external_original_FP32_evidence"
                               if original_observations else "external_original_evidence_not_provided",
                               "production_AD_displacement_prediction": _prediction(production_gradients[field], probe)}
                result["original_coarse_observations"].append(observation)
            result["passed"] = bool(pair and len(result["selected_measurements"]) == 2
                                     and all(value["passed"] for value in result["selected_measurements"]))
            report["directions"].append(result)
            if not result["passed"]:
                report["failures"].append("Mandatory reference FD/production parity/locality failed: " + str(number))
        for field in _FIELDS:
            rows = [row for row in report["directions"] if row["field"] == field]
            strong = [row for row in rows if any(role.startswith("strong_distinct_row_") for role in row["roles"])]
            report["field_outcomes"][field] = {"passed": bool(len(strong) == STRONG_ROWS
                                                               and all(row["passed"] for row in rows)),
                "meaningful_strong_rows": len(strong), "all_mandatory_contrasts_passed": all(row["passed"] for row in rows)}
    except Exception as error:
        report["failures"].append(type(error).__name__ + ": " + str(error))
    finally:
        report["baseline_fp32_unchanged"] = _snapshot(arrays) == baseline32
        try:
            report["frozen_after"] = _identity(frozen_identity)
            report["frozen_unchanged"] = before is not None and before == report["frozen_after"]
        except Exception as error:
            report["frozen_unchanged"] = False
            report["failures"].append("Frozen after-check failed: " + repr(error))
        if not report["frozen_unchanged"] or not report["baseline_fp32_unchanged"]:
            report["failures"].append("Saved baseline or frozen model/context identity changed")
    report["can_admit_new_child"] = bool(not report["failures"] and len(report["field_outcomes"]) == 2
                                         and all(value["passed"] for value in report["field_outcomes"].values()))
    report["status"] = "passed" if report["can_admit_new_child"] else "failed"
    report["receipt_sha256"] = _seal(report)
    return report
