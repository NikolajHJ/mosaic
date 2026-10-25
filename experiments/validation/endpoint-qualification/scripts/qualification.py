"""Loss-independent FP32 endpoint plans; separate diagnostic evidence only.

Native evaluators, source/input authentication, original null/component tests and
allocated resource/storage limits belong to the caller. This module never loads
models, changes a graph, updates an optimizer or certifies/restarts a saved run.
NumPy is imported lazily, only after an allocation guard.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import os


VERSION = "endpoint-qualification-diagnostic-v1"
OUTCOMES = ("passed", "numerical_disagreement", "boundary_inconclusive",
            "unresolved_precision")


@dataclass(frozen=True)
class Policy:
    original_epsilons: tuple = (0.01, 0.003, 0.001)
    geometry_fractions: tuple = (0.25, 0.125, 0.0625)
    maximum_epsilon: float = 0.01
    maximum_supplementary_epsilon: float = 0.0003
    repeats: int = 3
    relative_tolerance: float = 0.15
    scalar_parity_atol: float = 2e-5
    gradient_repeat_atol: float = 2e-5
    minimum_tangent: float = 1e-7
    scalar_ulp_multiplier: float = 2.0
    repeat_spread_multiplier: float = 4.0


def _numpy():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Endpoint numerical diagnostics require allocated DTU HPC")
    import numpy as np
    return np


def _policy(policy):
    np = _numpy()
    if not isinstance(policy, Policy):
        raise TypeError("Require an explicit diagnostic Policy")
    if policy.original_epsilons != (0.01, 0.003, 0.001):
        raise ValueError("Original diagnostic scales cannot be replaced")
    if (policy.geometry_fractions != (0.25, 0.125, 0.0625)
            or policy.maximum_epsilon != 0.01
            or policy.maximum_supplementary_epsilon != 0.0003 or policy.repeats != 3):
        raise ValueError("Require the versioned loss-independent geometry/repeat policy")
    for key in ("relative_tolerance", "scalar_parity_atol", "gradient_repeat_atol",
                "minimum_tangent", "scalar_ulp_multiplier", "repeat_spread_multiplier"):
        value = getattr(policy, key)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid finite diagnostic threshold: " + key)
    if not 0 < policy.relative_tolerance < 1:
        raise ValueError("Relative tolerance must be below one")
    if asdict(policy) != asdict(Policy()):
        raise ValueError("Diagnostic thresholds are versioned; do not tune them to a result")


def _arrays(values):
    np = _numpy()
    if not isinstance(values, dict) or not values:
        raise ValueError("Named categorical FP32 arrays are required")
    result = {}
    for name in sorted(values):
        if not isinstance(name, str) or not name:
            raise ValueError("Array names must be nonempty strings")
        array = np.asarray(values[name])
        if (array.dtype != np.float32 or array.ndim != 2 or array.shape[1] < 2
                or not array.shape[0] or not np.isfinite(array).all()
                or (array < 0).any() or (array > 1).any()
                or not np.allclose(array.sum(-1), 1, atol=2e-6, rtol=0)):
            raise ValueError("Require finite original FP32 categorical simplex rows")
        result[name] = array.copy()
        result[name].flags.writeable = False
    return result


def _record(array):
    raw = array.tobytes(order="C")
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(raw).decode("ascii")}


def _snapshot(arrays):
    return {key: _record(value) for key, value in sorted(arrays.items())}


def _copies(arrays):
    return {key: value.copy() for key, value in arrays.items()}


def _seal(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def predeclare(values, directions=None, *, policy=Policy()):
    """Build every contrast/scale/actual FP32 point before any evaluator call.

    ``directions`` maps each field to a nonempty list of row/source/target dicts,
    normally including the authenticated original failing contrast. Without an
    explicit list, geometry alone selects the pair with largest symmetric margin
    (ties: original row/category order); no gradient or observed loss is used.
    """
    np = _numpy()
    _policy(policy)
    arrays = _arrays(values)
    if directions is not None and set(directions) != set(arrays):
        raise ValueError("Every field must have an explicit contrast plan")
    result = {"version": VERSION, "policy": asdict(policy),
              "selection": "explicit_original_or_geometry_only_before_evaluator_calls",
              "baseline": _snapshot(arrays), "directions": [],
              "original_failed_outcome_must_remain_unchanged": True}
    for field, values in arrays.items():
        selected = None if directions is None else directions[field]
        if selected is None:
            candidates = [(min(float(row[a]), float(row[b])), -r, -a, -b, r, a, b)
                          for r, row in enumerate(values)
                          for a in range(values.shape[1]) for b in range(a + 1, values.shape[1])]
            best = max(candidates)
            selected = [{"row": best[-3], "source": best[-2], "target": best[-1]}]
        if not isinstance(selected, (list, tuple)) or not 1 <= len(selected) <= 4:
            raise ValueError("Require one to four predeclared contrasts per field")
        seen = set()
        for direction in selected:
            if not isinstance(direction, dict) or set(direction) != {"row", "source", "target"}:
                raise ValueError("Contrast must contain only row/source/target")
            r, a, b = (direction[k] for k in ("row", "source", "target"))
            if (any(type(v) is not int for v in (r, a, b)) or not 0 <= r < values.shape[0]
                    or not 0 <= a < values.shape[1] or not 0 <= b < values.shape[1]
                    or a == b or (r, a, b) in seen):
                raise ValueError("Invalid or duplicate predeclared contrast")
            seen.add((r, a, b))
            margin = min(float(values[r, a]), float(values[r, b]),
                         1 - float(values[r, a]), 1 - float(values[r, b]))
            scales = [(float(e), "original") for e in policy.original_epsilons]
            # The complete bounded ladder is fixed now, before any scalar/AD call.
            geometry_base = min(policy.maximum_supplementary_epsilon,
                                margin * policy.geometry_fractions[0])
            for fraction in policy.geometry_fractions:
                epsilon = geometry_base * fraction / policy.geometry_fractions[0]
                if epsilon > 0 and not any(e == epsilon for e, _ in scales):
                    scales.append((epsilon, "geometry_supplementary"))
            row = {"field": field, **direction, "symmetric_margin": margin, "probes": []}
            for epsilon, kind in scales:
                points = []
                for sign in (-1, 1):
                    point = values.copy()
                    point[r, a] = np.float32(point[r, a] - np.float32(sign * epsilon))
                    point[r, b] = np.float32(point[r, b] + np.float32(sign * epsilon))
                    delta = point - values
                    changed = bool(point[r, a] != values[r, a] and point[r, b] != values[r, b])
                    feasible = bool(np.isfinite(point).all() and (point >= 0).all()
                                    and (point <= 1).all()
                                    and np.allclose(point.sum(-1), 1, atol=2e-6, rtol=0))
                    points.append({"sign": sign, "point": _record(point), "delta": _record(delta),
                                   "simplex_feasible": feasible, "both_entries_changed": changed,
                                   "affected_entries_positive": bool(point[r, a] > 0 and point[r, b] > 0),
                                   "maximum_row_sum_error": float(np.abs(point.sum(-1) - 1).max())})
                row["probes"].append({"epsilon": epsilon, "kind": kind,
                                      "strict_interior": epsilon < margin,
                                      "points": points})
            result["directions"].append(row)
    result["plan_sha256"] = _seal(result)
    return result


def _decode(record):
    np = _numpy()
    if record["dtype"] != "float32":
        raise ValueError("Plan array dtype changed")
    raw = base64.b64decode(record["base64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Plan array byte identity changed")
    array = np.frombuffer(raw, dtype=np.float32).reshape(record["shape"]).copy()
    return array


def _scalar(value):
    np = _numpy()
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(array):
        raise ValueError("Evaluator must return one finite FP32/FP64 scalar")
    numeric = float(array)
    ulp = float(abs(np.spacing(array)))
    return {"value": numeric, "dtype": str(array.dtype), "ulp": ulp}


def _adjacent(probes):
    return any(a["agrees"] and b["agrees"] for a, b in zip(probes, probes[1:]))


def evaluate_plan(plan, primal, autodiff, frozen_identity, *, original_outcome="failed_preserved",
                  logit_control=None):
    """Evaluate immutable probes; callbacks must use independent frozen forwards.

    ``primal(dict_of_arrays)`` returns a scalar from a fresh backward-disabled
    forward. ``autodiff(dict_of_arrays)`` returns (scalar, named raw gradients).
    ``frozen_identity()`` returns an authenticated JSON identity. Optional
    ``logit_control()`` is separate coordinate evidence and never changes outcome.
    No callback is allowed to perform optimizer updates or mutate saved results.
    """
    np = _numpy()
    if plan.get("version") != VERSION or plan.get("plan_sha256") != _seal({
            key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise ValueError("Immutable predeclared plan identity differs")
    policy = Policy(**{key: tuple(value) if key in ("original_epsilons", "geometry_fractions") else value
                       for key, value in plan["policy"].items()})
    _policy(policy)
    if original_outcome != "failed_preserved":
        raise ValueError("This diagnostic cannot promote or relabel an original failed study")
    arrays = _arrays({key: _decode(value) for key, value in plan["baseline"].items()})
    receipt = {"version": VERSION, "status": "diagnostics_completed_original_failure_preserved",
               "scope": "separate_saved_endpoint_diagnostic_only", "plan": plan,
               "original_outcome": original_outcome, "can_certify_or_restart_original_study": False,
               "original_scientific_gate_replaced": False, "primal_repeats": [],
               "autodiff_repeats": [], "directions": [], "failures": []}
    before = frozen_identity()
    # Ensure identities are JSON serializable and cannot change through an alias.
    receipt["frozen_before"] = json.loads(json.dumps(before, sort_keys=True, allow_nan=False))
    gradients = []
    try:
        for _ in range(policy.repeats):
            receipt["primal_repeats"].append(_scalar(primal(_copies(arrays))))
            scalar, gradient = autodiff(_copies(arrays))
            if set(gradient) != set(arrays):
                raise ValueError("AD gradient fields differ from the predeclared graph")
            checked = {}
            for key, original in arrays.items():
                value = np.asarray(gradient[key])
                if value.dtype != np.float32 or value.shape != original.shape or not np.isfinite(value).all():
                    raise ValueError("Malformed/nonfinite original raw AD gradient")
                checked[key] = value.copy()
            gradients.append(checked)
            receipt["autodiff_repeats"].append({"scalar": _scalar(scalar), "gradients": _snapshot(checked)})
        baseline = sum(v["value"] for v in receipt["primal_repeats"]) / policy.repeats
        spread = max(v["value"] for v in receipt["primal_repeats"]) - min(v["value"] for v in receipt["primal_repeats"])
        parity = max(abs(v["scalar"]["value"] - baseline) for v in receipt["autodiff_repeats"])
        gradient_spread = max(float(np.abs(value[key] - gradients[0][key]).max())
                              for value in gradients for key in arrays)
        receipt["baseline_measurement"] = {"primal_mean": baseline, "primal_repeat_spread": spread,
                                            "maximum_ad_primal_scalar_error": parity,
                                            "maximum_gradient_repeat_difference": gradient_spread}
        if parity > policy.scalar_parity_atol:
            receipt["failures"].append("Independent primal/AD scalar parity failed")
        if gradient_spread > policy.gradient_repeat_atol:
            receipt["failures"].append("AD repeats are not stable within the declared tolerance")
        for direction in plan["directions"]:
            field, r, a, b = (direction[k] for k in ("field", "row", "source", "target"))
            gradient = gradients[0][field]
            nominal = float(gradient[r, b]) - float(gradient[r, a])
            tangent_repeats = [float(g[field][r, b]) - float(g[field][r, a]) for g in gradients]
            tangent_spread = max(tangent_repeats) - min(tangent_repeats)
            measured = {key: direction[key] for key in ("field", "row", "source", "target", "symmetric_margin")}
            measured.update(analytic_nominal=nominal, analytic_nominal_repeats=tangent_repeats,
                            analytic_nominal_repeat_spread=tangent_spread, probes=[])
            for probe in direction["probes"]:
                item = {"epsilon": probe["epsilon"], "kind": probe["kind"], "agrees": False,
                        "resolved": False, "evaluated": False, "geometry": probe["points"]}
                if not probe["strict_interior"] or not all(p["simplex_feasible"] and p["affected_entries_positive"] for p in probe["points"]):
                    item["reason"] = "boundary_inconclusive"
                elif not all(p["both_entries_changed"] for p in probe["points"]):
                    item["reason"] = "unresolved_precision"
                else:
                    repeats, point_arrays = [], []
                    for point in probe["points"]:
                        values = _decode(point["point"])
                        point_arrays.append(values)
                        payload = _copies(arrays)
                        payload[field] = values
                        repeats.append([_scalar(primal(_copies(payload))) for _ in range(policy.repeats)])
                    epsilon = probe["epsilon"]
                    minus, plus = [sum(v["value"] for v in values) / policy.repeats for values in repeats]
                    fd = (plus - minus) / (2 * epsilon)
                    represented = float(np.sum(gradient.astype(np.float64) *
                                        (point_arrays[1].astype(np.float64) - point_arrays[0].astype(np.float64)))) / (2 * epsilon)
                    repeat_spread = max(spread, *[max(v["value"] for v in values) - min(v["value"] for v in values) for values in repeats])
                    ulp = max(v["ulp"] for values in repeats for v in values)
                    signal = abs(plus - minus)
                    floor = max(policy.scalar_ulp_multiplier * ulp, policy.repeat_spread_multiplier * repeat_spread)
                    resolved = bool(signal > floor and abs(represented) > policy.minimum_tangent
                                    and tangent_spread <= policy.relative_tolerance * max(abs(nominal), policy.minimum_tangent))
                    relative = abs(fd - represented) / max(abs(fd), abs(represented)) if max(abs(fd), abs(represented)) else None
                    agrees = bool(resolved and fd * represented > 0 and relative <= policy.relative_tolerance)
                    item.update(evaluated=True, resolved=resolved, agrees=agrees,
                                primal_endpoint_repeats=repeats, finite_difference=fd,
                                analytic_actual_fp32_displacement=represented,
                                relative_error=relative, endpoint_difference_signal=signal,
                                measurement_resolution_floor=floor, repeat_spread=repeat_spread,
                                plus_one_sided=(plus - baseline) / epsilon,
                                minus_one_sided=(baseline - minus) / epsilon,
                                central_second_difference=(plus - 2 * baseline + minus) / epsilon**2,
                                reason="agreement" if agrees else ("numerical_disagreement" if resolved else "unresolved_precision"))
                measured["probes"].append(item)
            original = [v for v in measured["probes"] if v["kind"] == "original"]
            supplementary = [v for v in measured["probes"] if v["kind"] == "geometry_supplementary"]
            boundary = any(v.get("reason") == "boundary_inconclusive" for v in original)
            if boundary:
                outcome = "boundary_inconclusive"
            elif _adjacent(original):
                outcome = "passed"
            elif any(v["resolved"] and not v["agrees"] for v in original):
                outcome = "numerical_disagreement"
            else:
                outcome = "unresolved_precision"
            measured.update(outcome=outcome, original_adjacent_agreement=_adjacent(original),
                            supplementary_adjacent_agreement=_adjacent(supplementary),
                            scale_dependence=bool(any(v["resolved"] and not v["agrees"] for v in original)
                                                  and any(v["agrees"] for v in supplementary)),
                            interpretation="Supplementary agreement cannot replace original failure; no unique mechanism inferred")
            receipt["directions"].append(measured)
        if logit_control is not None:
            value = logit_control()
            receipt["logit_control"] = {"scope": "separate_saved_logit_coordinate_control",
                                          "raw_baseline_equivalence_assumed": False,
                                          "can_replace_raw_outcome": False,
                                          "evidence": json.loads(json.dumps(value, allow_nan=False))}
    finally:
        receipt["frozen_after"] = json.loads(json.dumps(frozen_identity(), sort_keys=True, allow_nan=False))
        receipt["frozen_unchanged"] = receipt["frozen_before"] == receipt["frozen_after"]
        if not receipt["frozen_unchanged"]:
            receipt["failures"].append("Frozen identity changed during diagnostic evaluation")
    if receipt["failures"]:
        receipt["outcome"] = "numerical_disagreement"
    else:
        states = {v["outcome"] for v in receipt["directions"]}
        receipt["outcome"] = next((v for v in ("numerical_disagreement", "boundary_inconclusive",
                                                "unresolved_precision") if v in states), "passed")
    receipt["receipt_sha256"] = _seal(receipt)
    return receipt
