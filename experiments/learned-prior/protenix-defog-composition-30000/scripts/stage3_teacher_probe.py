"""Bounded teacher repeatability/locality probes, separate from optimization.

The default budget is 32 prediction requests: baseline, repeat and six convex
perturbations at each of four teacher times. No Protenix model is loaded.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class ProbeIntegrityError(RuntimeError):
    """A frozen/graph-integrity failure, retaining all collected observations."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_rows(x, e):
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != 6:
        raise ValueError("Probe atoms must use shape [N>=1,6]")
    if e.shape != (len(x) * (len(x) - 1) // 2, 5):
        raise ValueError("Probe bonds must use shape [N*(N-1)//2,5]")
    for name, rows in (("X", x), ("E_upper", e)):
        if rows.dtype.kind != "f" or not np.isfinite(rows).all():
            raise ValueError(f"{name} must hold finite floating probabilities")
        if (rows < 0).any() or (rows > 1).any() or not np.allclose(rows.sum(-1), 1, atol=1e-6, rtol=0):
            raise ValueError(f"{name} must hold probability simplex rows")


def _difference(x, e, reference_x, reference_e):
    atom_tv = np.abs(x.astype(np.float64) - reference_x).sum(-1) / 2
    bond_tv = np.abs(e.astype(np.float64) - reference_e).sum(-1) / 2
    all_tv = np.concatenate((atom_tv, bond_tv))
    return {
        "atom_mean_tv": float(atom_tv.mean()), "atom_max_tv": float(atom_tv.max(initial=0)),
        "bond_mean_tv": float(bond_tv.mean()) if len(bond_tv) else 0.,
        "bond_max_tv": float(bond_tv.max(initial=0)),
        "mean_tv": float(all_tv.mean()), "max_tv": float(all_tv.max(initial=0)),
        "atom_argmax_flips": int(np.count_nonzero(x.argmax(-1) != reference_x.argmax(-1))),
        "bond_argmax_flips": int(np.count_nonzero(e.argmax(-1) != reference_e.argmax(-1))),
    }


def _query_design(times, noise_scales, perturbation_seeds):
    times = tuple(float(t) for t in times)
    scales = tuple(float(eps) for eps in noise_scales)
    seeds = tuple(perturbation_seeds)
    if not times or len(set(times)) != len(times) or any(not math.isfinite(t) or not 0 <= t <= 1 for t in times):
        raise ValueError("Teacher times must be distinct finite values in [0,1]")
    if len(set(scales)) != len(scales) or any(not math.isfinite(eps) or not 0 <= eps <= 1 for eps in scales):
        raise ValueError("Noise scales must be distinct finite convex weights in [0,1]")
    if len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Perturbation seeds must be distinct nonnegative integers")
    return times, scales, seeds


def probe_graph(teacher, graph, times=(.2, .5, .8, .95),
                noise_scales=(1e-4, 1e-3, .01), perturbation_seeds=(1, 2)):
    """Measure prediction stability, keeping failed queries as observations.

    Noise is `(1-eps)*p + eps*q`, where every q row is Dirichlet(1).
    One direction per seed is shared across times and noise scales. Exact
    arithmetic bounds each input-row TV by eps; floating roundoff is reported.
    No failed query is retried. If a worker closes, remaining planned queries
    are marked skipped. Frozen verification failures are fatal and retained in
    ProbeIntegrityError.result rather than mislabeled compatibility results.
    """
    times, noise_scales, perturbation_seeds = _query_design(times, noise_scales, perturbation_seeds)
    x, e = np.array(graph.X, copy=True), np.array(graph.E_upper, copy=True)
    _validate_rows(x, e)
    directions = {}
    for seed in perturbation_seeds:
        rng = np.random.default_rng(seed)
        directions[seed] = (rng.dirichlet(np.ones(6), size=len(x)),
                            rng.dirichlet(np.ones(5), size=len(e)))
    before_queries = getattr(teacher, "queries", None)
    result = {
        "schema": "defog-stage3-teacher-probe-v1", "status": "running",
        "scope": "teacher repeatability/locality only; no native optimization or efficacy claim",
        "teacher_identity": _jsonable(getattr(teacher, "identity", {})),
        "graph": {"n_atoms": len(x), "n_pairs": len(e), "elements": ["C", "N", "O", "S", "F", "Cl"],
                  "bonds": ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]},
        "design": {"times": list(times), "noise_scales": list(noise_scales),
                   "perturbation_seeds": list(perturbation_seeds),
                   "noise": "convex Dirichlet(1); same direction across times/scales",
                   "planned_queries": len(times) * (2 + len(noise_scales) * len(perturbation_seeds))},
        "queries": [], "attempted_predict_calls": 0, "successful_predictions": 0,
        "failed_predictions": 0, "skipped_queries": 0, "frozen": {"status": "not_checked"},
    }

    def fail_integrity(message, row=None):
        result["status"] = "failed"
        result["failure_kind"] = "integrity"
        result["error"] = message
        if row is not None:
            row.update(status="integrity_failed", error=message)
            result["queries"].append(row)
        raise ProbeIntegrityError(message, result)

    def query(kind, teacher_time, qx, qe, baseline, *, eps=0., seed=None):
        row = {"query_index": len(result["queries"]), "kind": kind, "time": teacher_time,
               "noise_scale": eps, "perturbation_seed": seed,
               "input_delta": _difference(qx, qe, x, e), "output_delta": None,
               "prediction_displacement": None, "vocabulary_metadata": None}
        tolerance = 8 * max(np.finfo(qx.dtype).eps, np.finfo(qe.dtype).eps)
        row["input_tv_bound"] = {"bound": eps, "floating_tolerance": float(tolerance),
                                 "satisfied": bool(row["input_delta"]["max_tv"] <= eps + tolerance)}
        if not row["input_tv_bound"]["satisfied"]:
            fail_integrity("Convex perturbation exceeded its declared input-row TV bound", row)
        if getattr(teacher, "_closed", False):
            row.update(status="skipped", error="Teacher worker is closed; no resend/restart attempted")
            result["queries"].append(row)
            result["skipped_queries"] += 1
            return None
        incoming_x, incoming_e = qx.copy(), qe.copy()
        result["attempted_predict_calls"] += 1
        try:
            prediction = teacher.predict(SimpleNamespace(X=qx, E_upper=qe, n_atoms=len(qx)), time=teacher_time)
            if not np.array_equal(qx, incoming_x) or not np.array_equal(qe, incoming_e):
                fail_integrity("Teacher mutated a probe input graph", row)
            px, pe = np.array(prediction.X, copy=True), np.array(prediction.E_upper, copy=True)
            if px.shape != x.shape or pe.shape != e.shape:
                fail_integrity("Teacher changed probe atom/bond slots or vocabulary dimensions", row)
            try:
                _validate_rows(px, pe)
            except (ValueError, TypeError) as error:
                fail_integrity(f"Teacher returned invalid prediction probabilities: {error}", row)
            row.update(status="passed", vocabulary_metadata=_jsonable(getattr(prediction, "metadata", {})),
                       prediction_displacement=_difference(px, pe, qx, qe))
            if baseline is not None:
                row["output_delta"] = _difference(px, pe, baseline[0], baseline[1])
                row["prediction_exactly_repeated"] = bool(np.array_equal(px, baseline[0]) and np.array_equal(pe, baseline[1]))
            result["successful_predictions"] += 1
            result["queries"].append(row)
            return px, pe
        except ProbeIntegrityError:
            raise
        except Exception as error:
            if not np.array_equal(qx, incoming_x) or not np.array_equal(qe, incoming_e):
                fail_integrity("Teacher mutated a probe input while failing", row)
            # These are explicit transport/identity assertions in the portable
            # teacher, distinct from an unsupported native feature execution.
            if str(error).startswith(("Frozen model state or query/forward count changed",
                                      "Teacher graph, time, forward-count or target mismatch",
                                      "Teacher changed graph slots or categories",
                                      "Teacher response ID/status mismatch")):
                fail_integrity(f"Teacher integrity failure during prediction: {error}", row)
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            result["failed_predictions"] += 1
            result["queries"].append(row)
            return None

    for teacher_time in times:
        baseline = query("baseline", teacher_time, x.copy(), e.copy(), None)
        query("repeat", teacher_time, x.copy(), e.copy(), baseline)
        for eps in noise_scales:
            for seed in perturbation_seeds:
                dx, de = directions[seed]
                qx = ((1 - eps) * x + eps * dx).astype(x.dtype)
                qe = ((1 - eps) * e + eps * de).astype(e.dtype)
                _validate_rows(qx, qe)
                query("perturbation", teacher_time, qx, qe, baseline, eps=eps, seed=seed)
    after_queries = getattr(teacher, "queries", None)
    result["teacher_reported_query_delta"] = (after_queries - before_queries
                                               if before_queries is not None and after_queries is not None else None)
    if getattr(teacher, "_closed", False):
        result["frozen"] = {"status": "unavailable", "reason": "worker_closed_after_prediction_error",
                            "passed": False, "forward_count_reconciled": False}
    elif hasattr(teacher, "verify_frozen"):
        try:
            verification = teacher.verify_frozen()
            if isinstance(verification, dict):
                if verification.get("passed") is False:
                    raise RuntimeError("Teacher explicitly failed frozen verification")
                for name in ("queries", "forwards"):
                    if after_queries is not None and name in verification and verification[name] != after_queries:
                        raise RuntimeError(f"Teacher {name} count does not match its successful-query count")
            result["frozen"] = {"status": "passed", "record": _jsonable(verification),
                                "scope": "teacher's own frozen-state and query/forward-count check"}
        except Exception as error:
            result["frozen"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            fail_integrity("Teacher frozen-state or query/forward-count verification failed")
    else:
        result["frozen"] = {"status": "unavailable", "reason": "teacher_has_no_verify_frozen_method"}
    result["status"] = ("complete" if not result["failed_predictions"] and not result["skipped_queries"]
                        and result["frozen"]["status"] == "passed" else "partial")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--initialization", choices=("sparse", "dense", "fragmented"), default="sparse")
    parser.add_argument("--feature-policy", choices=("hard_aux_soft_core", "native_soft"), default="hard_aux_soft_core")
    parser.add_argument("--output", type=Path, required=True, help="Fresh result directory")
    args = parser.parse_args(argv)
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Run real teacher probes in an allocated DTU CPU/GPU job")
    # Teacher-only imports and initialization are below the allocation gate.
    from mosaic_defog import DeFoGTeacher
    from stage3_math import make_initial_graph

    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "seed": args.seed, "initialization": args.initialization,
              "feature_policy": args.feature_policy,
              "started_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["LSB_JOBID"]}
    teacher, code = None, 0
    try:
        graph = make_initial_graph(args.seed, args.initialization)
        teacher = DeFoGTeacher(assets["teacher_source"], assets["teacher_checkpoint"],
                              python=assets["teacher_python"], feature_policy=args.feature_policy,
                              stderr=args.output / "teacher.stderr")
        result.update(probe_graph(teacher, graph))
    except ProbeIntegrityError as error:
        result.update(error.result)
        code = 1
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}")
        code = 1
    finally:
        if teacher is not None:
            teacher.close()
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output / "probe.json"), "status": result["status"],
                      "successful_predictions": result.get("successful_predictions", 0)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
