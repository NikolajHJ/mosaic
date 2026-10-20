"""Allocated NumPy analysis of authenticated saved native diagnostic records.

No Torch/JAX, model execution, optimization or readiness approval occurs here.
Fine scales remain observations; the historical three-scale gate is preserved.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


VARIANTS = ("source02_checkpoint", "diagnostic_direct_blocks")
MODES = ("independent_primal", "autodiff", "gradient_enabled_forward_only")
ORIGINAL = (0.01, 0.003, 0.001)
EXTRA = (0.0003, 0.0001)
DIRECTION = {"field": "E_upper", "row": 2, "source": 4, "target": 0}
PINS = {
    "source_manifest_sha256": "5eb6e17ea8d19e52c99a589243228710994dcf137f59a6b9a6659a4fe719a7f4",
    "context_sha256": "254d5b7f11605fb41db6f5abf7015f9cc856a19dfe1d4b506ccaea9007200770",
    "checkpoint_sha256": "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e",
    "model_digest_before": "395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16",
    "context_digest_before": "495002ffa2a52cc85b58ef34424622763dfc4ca556139a98297e8789ae287aa5",
}
INITIALIZATION_SHA = "3513c7b792994de8683243315288dde33a387cd54b0b5a730c227c10b6907b07"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def agrees(analytic, fd):
    return bool(
        math.isfinite(analytic)
        and math.isfinite(fd)
        and abs(analytic) > 1e-7
        and analytic * fd > 0
        and abs(fd - analytic) <= 0.15 * max(abs(fd), abs(analytic))
    )


def analyze(diagnostic, replay, report_sha):
    import numpy as np

    diagnostic, replay = diagnostic.resolve(), replay.resolve()
    path = diagnostic / "report.json"
    require(sha(path) == report_sha, "Diagnostic report hash differs")
    record, metadata = read(path), read(replay / "replay.json")
    require(
        record["status"] == "recorded"
        and record["readiness_certificate"] is False
        and record["historical_gate_status"] == "failed",
        "Not a complete evidence-only diagnostic",
    )
    require(
        all(record[k] == v for k, v in PINS.items()),
        "Native source/asset/context identity differs",
    )
    require(
        record["frozen_identity_status"] == "passed"
        and record["model_digest_after"] == PINS["model_digest_before"]
        and record["context_digest_after"] == PINS["context_digest_before"],
        "Frozen identities failed",
    )
    require(
        record["direction"] == metadata["direction"] == DIRECTION
        and metadata["source_manifest_sha256"] == PINS["source_manifest_sha256"]
        and metadata["initialization_sha256"] == INITIALIZATION_SHA
        and metadata["readiness_certificate"] is False,
        "Replay provenance differs",
    )
    replay_sha = sha(replay / "replay.npz")
    require(
        record["replay_sha256"] == metadata["replay_sha256"] == replay_sha,
        "Replay NPZ changed",
    )
    require(record["repeats"] in (2, 3), "Unexpected repeat budget")
    repeats = range(record["repeats"])
    points = [("baseline", 0.0, 0)] + [
        (f"bond-{e:g}-{'minus' if s < 0 else 'plus'}", e, s)
        for e in ORIGINAL + EXTRA
        for s in (-1, 1)
    ]
    require(
        [(p["name"], p["epsilon"], p["sign"]) for p in metadata["points"]] == points,
        "Recorded diagnostic point grid differs",
    )
    with np.load(replay / "replay.npz", allow_pickle=False) as saved:
        xs, es = saved["point_X"], saved["point_E_upper"]
    require(
        xs.shape == (11, 6, 6)
        and es.shape == (11, 15, 5)
        and xs.dtype == es.dtype == np.float32
        and np.isfinite(xs).all()
        and np.isfinite(es).all(),
        "Malformed replay arrays",
    )
    graph_pins, simplex = {}, []
    for i, (name, epsilon, sign) in enumerate(points):
        graph_pins[name] = hashlib.sha256(xs[i].tobytes() + es[i].tobytes()).hexdigest()
        require(
            graph_pins[name] == metadata["points"][i]["graph_sha256"],
            "Replay graph identity differs",
        )
        simplex.append(
            dict(
                point=name,
                epsilon=epsilon,
                sign=sign,
                minimum_atom_mass=float(xs[i].min()),
                minimum_bond_mass=float(es[i].min()),
                maximum_atom_mass=float(xs[i].max()),
                maximum_bond_mass=float(es[i].max()),
                atom_row_sums=xs[i].sum(-1).tolist(),
                bond_row_sums=es[i].sum(-1).tolist(),
                maximum_row_sum_error=float(
                    max(abs(xs[i].sum(-1) - 1).max(), abs(es[i].sum(-1) - 1).max())
                ),
            )
        )
    rows, arrays, head_error = {}, {}, 0.0
    for row in record["evaluations"]:
        key = row["variant"], row["point"], row["repeat"], row["mode"]
        require(
            key not in rows and row["graph_sha256"] == graph_pins.get(row["point"]),
            "Duplicate/changed diagnostic graph",
        )
        raw = (diagnostic / row["raw_artifact"]).resolve(strict=True)
        require(
            raw.is_relative_to(diagnostic) and sha(raw) == row["raw_sha256"],
            "Raw diagnostic NPZ path/hash differs",
        )
        expected = {"values": (3,)}
        if row["mode"] == "autodiff":
            expected.update(gradient_X=(6, 6), gradient_E=(15, 5))
        with np.load(raw, allow_pickle=False) as saved:
            require(
                set(saved.files) == set(expected), "Unexpected raw diagnostic arrays"
            )
            data = {k: saved[k].copy() for k in saved.files}
        require(
            all(
                v.shape == expected[k]
                and v.dtype == np.float32
                and np.isfinite(v).all()
                for k, v in data.items()
            ),
            "Malformed/nonfinite diagnostic values or gradients",
        )
        require(
            data["values"].tolist() == row["values"], "JSON/raw native outputs differ"
        )
        heads = row["per_head"]
        require(
            len(heads) in (1, 2)
            and all(
                math.isfinite(h[k])
                for h in heads
                for k in ("affinity_pred_value", "affinity_logits_binary")
            ),
            "Malformed native heads",
        )
        logits = [h["affinity_logits_binary"] for h in heads]
        probabilities = [
            1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v))
            for v in logits
        ]
        reconciled = [
            float(
                np.asarray([h["affinity_pred_value"] for h in heads], np.float32).mean()
            ),
            float(np.asarray(probabilities, np.float32).mean()),
            float(np.asarray(logits, np.float32).mean()),
        ]
        head_error = max(
            head_error, max(abs(a - b) for a, b in zip(reconciled, row["values"]))
        )
        require(
            head_error <= 2e-6, "Saved per-head native ensemble cannot be reconciled"
        )
        if row["mode"] == "autodiff":
            require(
                float(data["gradient_E"][2, 0] - data["gradient_E"][2, 4])
                == row["analytic_directional"],
                "Raw gradient/recorded directional derivative differ",
            )
        require(
            all(
                isinstance(row[k], str) and len(row[k]) == 64
                for k in ("rng_before", "rng_after")
            ),
            "Malformed RNG hashes",
        )
        rows[key], arrays[key] = row, data
    expected_keys = {
        (v, p[0], r, m)
        for v in VARIANTS
        for p in points
        for r in repeats
        for m in MODES
    }
    require(
        set(rows) == expected_keys,
        "Missing/unexpected forward-mode/checkpoint/repeat observations",
    )
    differences, gradient_differences, gradient_hashes, selectors = [], [], [], []
    for point, _, _ in points:
        baseline_key = VARIANTS[0], point, 0, MODES[0]
        baseline = arrays[baseline_key]["values"]
        ad_key = VARIANTS[0], point, 0, "autodiff"
        for key in [k for k in rows if k[1] == point]:
            delta = float(np.max(np.abs(arrays[key]["values"] - baseline)))
            if (
                arrays[key]["values"].tobytes() != baseline.tobytes()
                or rows[key]["per_head"] != rows[baseline_key]["per_head"]
            ):
                differences.append(
                    dict(key=list(key), maximum_absolute_value_difference=delta)
                )
            if key[3] == "autodiff":
                changes = {
                    field: float(
                        np.max(np.abs(arrays[key][field] - arrays[ad_key][field]))
                    )
                    for field in ("gradient_X", "gradient_E")
                }
                hashes = {
                    field: hashlib.sha256(arrays[key][field].tobytes()).hexdigest()
                    for field in ("gradient_X", "gradient_E")
                }
                gradient_hashes.append(dict(key=list(key), **hashes))
                if any(
                    arrays[key][field].tobytes() != arrays[ad_key][field].tobytes()
                    for field in ("gradient_X", "gradient_E")
                ):
                    gradient_differences.append(
                        dict(key=list(key), maximum_absolute_differences=changes)
                    )
                if point == "baseline":
                    g = arrays[key]["gradient_E"]
                    row = int(np.argmax(g.max(-1) - g.min(-1)))
                    selectors.append(
                        dict(
                            variant=key[0],
                            repeat=key[2],
                            row=row,
                            source=int(g[row].argmin()),
                            target=int(g[row].argmax()),
                        )
                    )
    replays, slopes = [], []
    for variant in VARIANTS:
        for repeat in repeats:
            central = rows[variant, "baseline", repeat, "autodiff"][
                "analytic_directional"
            ]
            base = rows[variant, "baseline", repeat, "independent_primal"]["values"][0]
            probes = []
            for epsilon in ORIGINAL + EXTRA:
                minus, plus = [
                    rows[
                        variant, f"bond-{epsilon:g}-{s}", repeat, "independent_primal"
                    ]["values"][0]
                    for s in ("minus", "plus")
                ]
                fd = (plus - minus) / (2 * epsilon)
                entry = dict(
                    epsilon=epsilon,
                    analytic=central,
                    finite_difference=fd,
                    agrees=agrees(central, fd),
                    primal_endpoints=[minus, plus],
                )
                if epsilon in ORIGINAL:
                    probes.append(entry)
                slopes.append(
                    dict(
                        variant=variant,
                        repeat=repeat,
                        **entry,
                        original_gate=epsilon in ORIGINAL,
                        one_sided_minus=(base - minus) / epsilon,
                        one_sided_plus=(plus - base) / epsilon,
                        endpoint_autodiff=[
                            rows[variant, f"bond-{epsilon:g}-{s}", repeat, "autodiff"][
                                "analytic_directional"
                            ]
                            for s in ("minus", "plus")
                        ],
                    )
                )
            outcome = (
                "passed"
                if any(a["agrees"] and b["agrees"] for a, b in zip(probes, probes[1:]))
                else "failed"
            )
            replays.append(
                dict(
                    variant=variant,
                    repeat=repeat,
                    status=outcome,
                    probes=probes,
                    relative_tolerance=0.15,
                    required_adjacent_scales=2,
                    readiness_certificate=False,
                )
            )
    require(
        replays == record["original_gate_replays"],
        "Unchanged original FD-gate replay does not match saved record",
    )
    rngs = {row[k] for row in rows.values() for k in ("rng_before", "rng_after")}
    rng_fixed = len(rngs) == 1
    selectors_match = all(
        (s["row"], s["source"], s["target"]) == (2, 4, 0) for s in selectors
    )
    discrepancy = bool(
        differences or gradient_differences or not rng_fixed or not selectors_match
    )
    interpretation = (
        "Observable execution/repeat/checkpoint/RNG difference; its cause is not established."
        if discrepancy
        else "All saved values, per-head outputs and complete AD gradients agree across modes, repeats and checkpoint controls. "
        "The scale-dependent response is consistent with nonlinear or FP32 numerical behavior, with no observed wrapper discrepancy; its root cause is not established."
    )
    return dict(
        status="recorded",
        readiness_certificate=False,
        historical_gate_status="failed",
        scope="allocated saved-record analysis only; no model execution, optimization or native approval",
        diagnostic_job_id=record["job_id"],
        analysis_job_id=os.environ["LSB_JOBID"],
        report_sha256=report_sha,
        replay_sha256=replay_sha,
        pins=PINS,
        forwards=len(rows),
        backwards=sum(k[3] == "autodiff" for k in rows),
        exact_value_and_head_agreement=not differences,
        exact_full_gradient_agreement=not gradient_differences,
        value_or_head_differences=differences,
        gradient_differences=gradient_differences,
        full_gradient_hashes=gradient_hashes,
        rng_fixed=rng_fixed,
        rng_hashes=sorted(rngs),
        maximum_contrast_selectors=selectors,
        maximum_contrast_matches_original=selectors_match,
        maximum_head_reconciliation_error=head_error,
        original_gate_replays=replays,
        directional_response=slopes,
        probability_domain=simplex,
        classification="observable_execution_difference"
        if discrepancy
        else "consistent_replay_with_scale_dependent_response",
        interpretation=interpretation
        + " Extra scales never replace the original three-scale gate, and random remains uncertified.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    require(
        bool(os.environ.get("LSB_JOBID")), "Analysis requires allocated DTU HPC CPU"
    )
    require(not args.output.exists(), "Analysis output already exists")
    result = {"status": "failed", "readiness_certificate": False}
    try:
        result = analyze(args.diagnostic, args.replay, args.report_sha256)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "forwards",
                    "backwards",
                    "classification",
                    "interpretation",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
