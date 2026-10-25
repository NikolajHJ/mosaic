"""Allocated, diagnostic-only re-evaluation of an immutable failed admission.

No new point, epsilon, acceptance rule, optimizer update or restart is defined.
The caller authenticates native source/context/model/snapshot identity and owns
the allocation/runtime guard. This module preserves the supplied fixed plan,
deduplicates its exact graph bytes and records scalar/gradient/contact evidence.
"""
from copy import deepcopy
import base64
import hashlib
import json
import os
from pathlib import Path


VERSION = "protenix-fixed-point-forensics-v1"
MAXIMUM_OUTPUT_BYTES = 256 * 1024**2
MAXIMUM_UNIQUE_GRAPHS = 109
SCALES = (0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003)


def _numpy():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Protenix numerical forensics require allocated DTU HPC")
    import numpy as np
    return np


def _require(value, message):
    if not value:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seal(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _array_record(value):
    raw = value.tobytes(order="C")
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(raw).decode("ascii")}


def _decode(record):
    np = _numpy()
    _require(isinstance(record, dict) and record.get("dtype") == "float32",
             "Fixed native input must retain exact FP32 dtype")
    raw = base64.b64decode(record["base64"], validate=True)
    _require(hashlib.sha256(raw).hexdigest() == record["sha256"], "Fixed point byte hash changed")
    array = np.frombuffer(raw, dtype=np.float32).reshape(record["shape"]).copy()
    _require(array.ndim == 2 and array.shape[1] >= 2 and np.isfinite(array).all()
             and (array >= 0).all() and (array <= 1).all()
             and np.allclose(array.sum(-1), 1, atol=2e-6, rtol=0),
             "A fixed evaluated point is not a finite categorical simplex")
    _require(_array_record(array) == record, "Fixed point noncanonical dtype/shape/bytes")
    return array


def _snapshot(values):
    return {name: _array_record(value) for name, value in sorted(values.items())}


def _graph_id(values):
    return _seal({name: {key: value[key] for key in ("dtype", "shape", "sha256")}
                  for name, value in _snapshot(values).items()})


def _identity(callback):
    result = callback()
    _require(isinstance(result, dict) and result, "Explicit frozen identity is required")
    return json.loads(_json(result))


def _contacts(value, rows, shape=None, dtype="float32"):
    np = _numpy()
    result = np.asarray(value)
    _require(result.dtype == np.dtype(dtype) and result.ndim == 2
             and result.shape[0] == rows and result.shape[1] >= 2
             and (shape is None or result.shape == shape) and np.isfinite(result).all(),
             "Contact matrix shape, precision or finiteness changed")
    return result.copy()


def _contact_summary(contacts, baseline_winners=None):
    """Host FP64 reductions of already supplied contacts are diagnostic only."""
    np = _numpy()
    winners = np.argmax(contacts, axis=1)
    ordered = np.sort(contacts.astype(np.float64), axis=1)
    result = {"winners": winners.tolist(), "top_two_margins": (ordered[:, -1] - ordered[:, -2]).tolist(),
              "negative_mean_best_contact_float64": float(-np.mean(ordered[:, -1], dtype=np.float64)),
              "scope": "Reduction/branch diagnostic only; no objective replacement",
              "can_certify_or_restart_original_study": False}
    if baseline_winners is not None:
        fixed = np.asarray(baseline_winners, dtype=np.int64)
        result["changed_winner_rows_from_baseline"] = np.flatnonzero(winners != fixed).tolist()
        result["negative_mean_fixed_baseline_winner_float64"] = float(
            -np.mean(contacts[np.arange(contacts.shape[0]), fixed].astype(np.float64), dtype=np.float64))
    return result


def _evaluate(evaluator, values):
    np = _numpy()
    payload = {name: value.copy() for name, value in values.items()}
    before = _snapshot(payload)
    raw = evaluator(payload)
    _require(_snapshot(payload) == before, "Native evaluator mutated a fixed input payload")
    _require(isinstance(raw, dict) and {"primal", "ad_value", "gradients", "primal_contacts", "ad_contacts"} <= set(raw),
             "Missing independent scalar/gradient/contact evidence")
    result = {}
    for name in ("primal", "ad_value"):
        scalar = np.asarray(raw[name])
        _require(scalar.shape == () and scalar.dtype.kind == "f" and np.isfinite(scalar),
                 "Native scalar is not finite floating point")
        result[name] = float(scalar)
    _require(isinstance(raw["gradients"], dict) and set(raw["gradients"]) == set(values),
             "Both raw gradient fields are required")
    result["gradients"] = {}
    for name, baseline in values.items():
        gradient = np.asarray(raw["gradients"][name])
        _require(gradient.dtype == np.float32 and gradient.shape == baseline.shape and np.isfinite(gradient).all(),
                 "Raw native gradient shape, dtype or finiteness changed")
        result["gradients"][name] = gradient.copy()
    result["primal_contacts"] = _contacts(raw["primal_contacts"], values["X"].shape[0])
    result["ad_contacts"] = _contacts(raw["ad_contacts"], values["X"].shape[0], result["primal_contacts"].shape)
    for name in ("primal_contacts64", "ad_contacts64"):
        if raw.get(name) is not None:
            result[name] = _contacts(raw[name], values["X"].shape[0], result["primal_contacts"].shape, "float64")
    return result


def _budget(output, maximum):
    total = 0
    for path in output.rglob("*"):
        _require(not path.is_symlink(), "Diagnostic output symlink is forbidden")
        if path.is_file():
            total += path.stat().st_size
    _require(total <= maximum, "Diagnostic persistent output cap exceeded; partial evidence preserved")
    return total


def _write(output, record):
    temporary = output / "report.json.next"
    with temporary.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, output / "report.json")


def _save_evaluation(output, point_id, repeat, values, result, baseline_winners):
    np = _numpy()
    path = output / (point_id + "-repeat-" + str(repeat) + ".npz")
    arrays = {**values, "primal": np.asarray(result["primal"], np.float64),
              "ad_value": np.asarray(result["ad_value"], np.float64),
              **{"gradient_" + name: value for name, value in result["gradients"].items()},
              **{name: value for name, value in result.items() if "contacts" in name}}
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    proof = {"path": str(path), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    summary = {"repeat": repeat, "primal": result["primal"], "ad_value": result["ad_value"],
               "primal_ad_scalar_difference": result["primal"] - result["ad_value"],
               "proof": proof, "gradients": {name: _array_record(value) for name, value in result["gradients"].items()},
               "contacts": {}, "can_certify_or_restart_original_study": False}
    for name, value in result.items():
        if "contacts" in name:
            key = "ad_contacts" if name.startswith("ad_") else "primal_contacts"
            summary["contacts"][name] = {"array": {k: v for k, v in _array_record(value).items() if k != "base64"},
                                          **_contact_summary(value, baseline_winners[key])}
    return summary


def _dot(gradients, displacement):
    np = _numpy()
    return float(sum(np.sum(gradients[name].astype(np.float64) * delta, dtype=np.float64)
                     for name, delta in displacement.items()))


def run_forensics(admission_record, evaluator, frozen_identity, output, *,
                  maximum_output_bytes=MAXIMUM_OUTPUT_BYTES,
                  maximum_unique_graphs=MAXIMUM_UNIQUE_GRAPHS):
    """Re-evaluate only the failed admission's exact baseline and fixed points.

    Evaluator(dict X/E_upper FP32) returns independent ``primal``/``ad_value``,
    both raw FP32 ``gradients`` and FP32 ``primal_contacts``/``ad_contacts``.
    Optional ``primal_contacts64``/``ad_contacts64`` are reduction-only controls
    made from the same native FP32 logits. Their different arithmetic never
    changes the production scalar, graph or optimizer. The output path must be
    fresh; runtime limits and original source authentication belong to caller.
    """
    np = _numpy()
    original = deepcopy(admission_record)
    original_seal = _seal(original)
    _require(original.get("version") == "prospective-endpoint-admission-v1"
             and original.get("status") == "failed" and original.get("can_admit_new_child") is False
             and original.get("original_run_failed_preserved") is True
             and original.get("can_certify_original_study") is False,
             "Forensics requires the preserved failed prospective admission")
    _require(original.get("receipt_sha256") == _seal({k: v for k, v in original.items() if k != "receipt_sha256"}),
             "Failed admission receipt hash changed")
    plan = original["plan"]
    _require(plan.get("plan_sha256") == _seal({k: v for k, v in plan.items() if k != "plan_sha256"})
             and plan.get("original_failed_outcome_must_remain_unchanged") is True,
             "Original fixed plan hash/status changed")
    _require(type(maximum_output_bytes) is int and 0 < maximum_output_bytes <= MAXIMUM_OUTPUT_BYTES
             and type(maximum_unique_graphs) is int and 1 <= maximum_unique_graphs <= MAXIMUM_UNIQUE_GRAPHS,
             "Forensics output/graph limits cannot be enlarged")
    _require(len(_json(plan).encode()) < maximum_output_bytes, "Output cap cannot hold the immutable fixed plan")
    baseline = {name: _decode(row) for name, row in plan["baseline"].items()}
    _require(set(baseline) == {"X", "E_upper"} and len(plan["directions"]) == 9,
             "The authenticated Protenix plan must retain all nine directions")
    requests, graphs = [], {}
    baseline_id = _graph_id(baseline)
    graphs[baseline_id] = baseline
    for number, direction in enumerate(plan["directions"]):
        _require(tuple(point["epsilon"] for point in direction["probes"]) == SCALES,
                 "Every original fixed scale must remain in declared order")
        field = direction["field"]
        _require(field in baseline, "Fixed direction field changed")
        row = {key: deepcopy(direction[key]) for key in ("field", "row", "source", "target", "roles")}
        row.update(direction_index=number, probes=[])
        for probe in direction["probes"]:
            item = {"epsilon": probe["epsilon"], "scheme": probe["scheme"], "left": None, "right": None,
                    "original_fixed_probe": deepcopy(probe), "can_certify_or_restart_original_study": False}
            if probe["scheme"] != "unavailable":
                _require(probe["scheme"] in ("central", "one_sided_inward"), "Undeclared fixed scheme")
                for side in ("left", "right"):
                    values = {name: value.copy() for name, value in baseline.items()}
                    values[field] = _decode(probe[side])
                    _require(values[field].shape == baseline[field].shape, "Fixed graph shape changed")
                    identity = _graph_id(values)
                    graphs.setdefault(identity, values)
                    item[side] = identity
            else:
                _require(probe.get("left") is None and probe.get("right") is None,
                         "Unavailable original points cannot be invented")
            row["probes"].append(item)
        requests.append(row)
    _require(len(graphs) <= maximum_unique_graphs, "Fixed graph cohort exceeds declared cap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"version": VERSION, "status": "diagnostics_running", "scope": "separate_fixed_point_diagnostic_only",
              "original_admission_sha256": original_seal, "original_admission_receipt_sha256": original["receipt_sha256"],
              "original_failed_admission_status": "failed", "original_run_failed_preserved": True,
              "can_certify_or_restart_original_study": False, "can_admit_new_child": False,
              "original_optimization_settings_changed": False, "original_scientific_gate_replaced": False,
              "original_plan": deepcopy(plan), "unique_graph_count": len(graphs), "baseline_graph_id": baseline_id,
              "points": [], "directions": requests, "failures": [],
              "maximum_persistent_output_bytes": maximum_output_bytes,
              "interpretation": "Diagnostic observations, without any gradient acceptance rule or continuation authorization"}
    _write(output, report)
    results, before, error = {}, None, None
    try:
        before = _identity(frozen_identity)
        report["frozen_before"] = before
        baseline_results, baseline_winners = [], None
        report["baseline_repeats"] = {"primal": [], "ad_value": []}
        for number, (identity, values) in enumerate(graphs.items()):
            evaluations = []
            entry = {"graph_id": identity, "input": _snapshot(values), "evaluations": [],
                     "can_certify_or_restart_original_study": False}
            report["points"].append(entry)
            for repeat in range(3 if identity == baseline_id else 1):
                result = _evaluate(evaluator, values)
                evaluations.append(result)
                if identity == baseline_id:
                    baseline_results.append(result)
                    if baseline_winners is None:
                        baseline_winners = {name: np.argmax(result[name], axis=1).tolist()
                                            for name in ("primal_contacts", "ad_contacts")}
                    for name in ("primal", "ad_value"):
                        report["baseline_repeats"][name].append(result[name])
                entry["evaluations"].append(_save_evaluation(output, "point-" + str(number).zfill(4), repeat,
                                                             values, result, baseline_winners))
                _write(output, report)
                _budget(output, maximum_output_bytes)
            results[identity] = evaluations
        mean_baseline_gradient = {name: np.mean([r["gradients"][name].astype(np.float64) for r in baseline_results], axis=0)
                                  for name in baseline}
        for direction in report["directions"]:
            for probe in direction["probes"]:
                if probe["left"] is None:
                    probe["analysis"] = {"evaluated": False, "reason": "Original fixed point unavailable"}
                    continue
                left_values, right_values = graphs[probe["left"]], graphs[probe["right"]]
                delta = {name: right_values[name].astype(np.float64) - left_values[name].astype(np.float64) for name in baseline}
                left, right = results[probe["left"]], results[probe["right"]]
                means = lambda rows, key: float(np.mean([r[key] for r in rows], dtype=np.float64))
                endpoint_gradients = [{name: np.mean([r["gradients"][name].astype(np.float64) for r in rows], axis=0)
                                       for name in baseline} for rows in (left, right)]
                trapezoid = {name: .5 * (endpoint_gradients[0][name] + endpoint_gradients[1][name]) for name in baseline}
                predicted = _dot(mean_baseline_gradient, delta)
                primal_delta = means(right, "primal") - means(left, "primal")
                ad_delta = means(right, "ad_value") - means(left, "ad_value")
                transport = _dot(trapezoid, delta)
                analysis = {"evaluated": True, "baseline_AD_dot_actual_displacement": predicted,
                            "independent_primal_difference": primal_delta, "AD_returned_scalar_difference": ad_delta,
                            "independent_primal_minus_baseline_AD_prediction": primal_delta - predicted,
                            "AD_scalar_minus_baseline_AD_prediction": ad_delta - predicted,
                            "endpoint_gradient_trapezoid_dot_actual_displacement": transport,
                            "independent_primal_minus_endpoint_gradient_trapezoid": primal_delta - transport,
                            "curvature_control_scope": "Endpoint trapezoid diagnostic only; not an independent oracle or acceptance rule",
                            "can_certify_or_restart_original_study": False}
                for name in ("primal_contacts", "ad_contacts", "primal_contacts64", "ad_contacts64"):
                    if all(name in r for r in left + right):
                        key = "ad_contacts" if name.startswith("ad_") else "primal_contacts"
                        reductions = []
                        for rows in (left, right):
                            summaries = [_contact_summary(r[name], baseline_winners[key]) for r in rows]
                            reductions.append({"best": float(np.mean([s["negative_mean_best_contact_float64"] for s in summaries])),
                                               "fixed": float(np.mean([s["negative_mean_fixed_baseline_winner_float64"] for s in summaries])),
                                               "changed_winner_rows": sorted(set(i for s in summaries for i in s["changed_winner_rows_from_baseline"]))})
                        analysis[name + "_reduction_control"] = {"best_difference": reductions[1]["best"] - reductions[0]["best"],
                            "fixed_baseline_winner_difference": reductions[1]["fixed"] - reductions[0]["fixed"],
                            "left_changed_winner_rows": reductions[0]["changed_winner_rows"],
                            "right_changed_winner_rows": reductions[1]["changed_winner_rows"],
                            "scope": "Reduction and fixed-branch control only; production objective unchanged"}
                probe["analysis"] = analysis
        _require(_seal(admission_record) == original_seal, "Original admission payload changed during diagnostics")
        report["status"] = "diagnostics_completed_original_failure_preserved"
    except Exception as caught:
        error = caught
        report["status"] = "diagnostics_failed_original_failure_preserved"
        report["failures"].append(type(caught).__name__ + ": " + str(caught))
    finally:
        try:
            report["frozen_after"] = _identity(frozen_identity)
            report["frozen_unchanged"] = before is not None and report["frozen_after"] == before
            _require(report["frozen_unchanged"], "Frozen identity changed or could not be verified")
        except Exception as caught:
            report["status"] = "diagnostics_failed_original_failure_preserved"
            report["frozen_unchanged"] = False
            report["failures"].append(type(caught).__name__ + ": " + str(caught))
            error = error or caught
        report["original_payload_unchanged"] = _seal(admission_record) == original_seal
        _write(output, report)
        try:
            report["persistent_output_bytes_observed"] = _budget(output, maximum_output_bytes)
            _write(output, report)
            _budget(output, maximum_output_bytes)
        except Exception as caught:
            error = error or caught
    if error is not None:
        raise error
    return report
