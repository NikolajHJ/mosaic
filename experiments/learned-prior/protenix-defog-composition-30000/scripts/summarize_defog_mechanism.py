"""Compare completed teacher controls with the archived live baselines."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def measure(folder):
    summary = read(folder / "summary.json")
    status = read(folder / "status.json")
    if not (summary["complete_planned_trajectory"] and status["frozen_checks_passed"]
            and summary["states_observed"] == 501 and status["updates_completed"] == 500):
        raise ValueError(f"Incomplete trajectory or frozen checks in {folder}")
    return {k: summary[k] for k in (
        "best_valid_hard_score", "best_valid_update", "valid_state_count", "unique_valid_molecules",
        "final_valid", "final_components", "teacher_queries", "elapsed_seconds",
    )}


def compare(archive, campaign):
    rows = []
    for config in read(campaign / "plan.json")["runs"]:
        folder = campaign / "runs" / config["run_id"]
        baseline = archive / "runs" / config["paired_with_run_id"]
        if not (folder / "status.json").is_file():
            rows.append({"run_id": config["run_id"], "status": "waiting_for_run"})
            continue
        status = read(folder / "status.json")
        if status["status"] != "complete" or not status.get("frozen_checks_passed"):
            rows.append({"run_id": config["run_id"], "status": status["status"],
                         "updates": status["updates_completed"], "error": status.get("error")})
            continue
        old, new = [read(p / "config.json") for p in (baseline, folder)]
        for k in ("seed", "model_key", "scorer_identity", "learning_rate", "native_weight",
                  "initialization_parameters", "initialization"):
            if old[k] != new[k]:
                raise ValueError(f"Compared runs differ in {k}: {folder.name}")
        for k in ("projection", "cooling", "interior_uniform_mass"):
            if old["condition"][k] != new["condition"][k]:
                raise ValueError(f"Compared runs differ in {k}: {folder.name}")
        with np.load(baseline / "initial.npz", allow_pickle=False) as a:
            with np.load(folder / "initial.npz", allow_pickle=False) as b:
                if any(not np.array_equal(a[k], b[k]) for k in ("X", "E_upper", "model_key")):
                    raise ValueError(f"Initial graph or key differs: {folder.name}")
        record = {"run_id": config["run_id"], "seed": config["seed"], "status": "complete",
                  "live_reference": measure(baseline), "control": measure(folder),
                  "gpu_reference": old["hardware"]["devices"], "gpu_control": new["hardware"]["devices"]}
        shuffled = config["condition"]["teacher"].get("shuffle_within_argmax", False)
        if record["control"]["teacher_queries"] != (500 if shuffled else 1):
            raise ValueError(f"Unexpected teacher query count: {folder.name}")
        if shuffled:
            metrics = {head: [] for head in ("X", "E_upper")}
            with gzip.open(folder / "gradients.jsonl.gz", "rt") as s:
                for line in s:
                    update = json.loads(line)
                    for head in metrics:
                        metrics[head].append(update["shuffled_teacher"][head])
            record["shuffle"] = {
                head: {"degenerate_steps": sum(r["flat_shuffled_nonflat_live"] for r in data),
                       "mean_permuted_rows": float(np.mean([r["permuted_rows"] for r in data])),
                       "max_tangent_displacement_error": max(
                           abs(r["shuffled_tangent_displacement"] * r["scale"]
                               - r["teacher_tangent_displacement"]) for r in data)}
                for head, data in metrics.items()
            }
        rows.append(record)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "campaign", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    rows = compare(args.archive, args.campaign)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)
