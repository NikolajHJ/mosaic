"""Compare completed matched DeFoG and structural-repair runs without rescoring."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from zipfile import ZipFile


SEEDS = (17, 23, 42)
ARMS = ("defog-w0p5", "repair-w1")
FIELDS = (
    "run_id", "seed", "condition_id", "states_observed", "updates_observed",
    "initial_valid", "any_valid", "any_new_valid", "valid_state_count",
    "unique_valid_molecules", "last100_valid_state_count", "best_valid_hard_score",
    "best_valid_smiles", "best_valid_update", "best_new_valid_hard_score",
    "best_new_valid_smiles", "best_new_valid_update", "final_valid", "final_components",
    "final_smiles", "final_hard_score",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def initial_identity(path):
    """Hash individual NPY payloads, excluding ZIP timestamps and compression settings."""
    with ZipFile(path) as archive:
        return {name: hashlib.sha256(archive.read(name + ".npy")).hexdigest()
                for name in ("X", "E_upper", "source_X", "source_E_upper", "model_key")}


def inspect_run(root, planned):
    folder = root / "runs" / planned["run_id"]
    if not (folder / "status.json").exists():
        return {"status": "awaiting output", "run_id": planned["run_id"]}, None
    status = read(folder / "status.json")
    result = {"run_id": planned["run_id"], "status": status["status"],
              "updates_completed": status.get("updates_completed")}
    if status.get("error"):
        result["error"] = status["error"]
    if status["status"] != "complete" or not status.get("frozen_checks_passed"):
        if status["status"] == "complete":
            result["status"] = "awaiting frozen checks"
        return result, None
    cfg = read(folder / "config.json")
    summary = read(folder / "summary.json")
    same_identity = all(cfg[k] == planned[k] == summary[k]
                        for k in ("run_id", "condition_id", "campaign_id", "initialization", "seed"))
    complete = (
        same_identity and summary["status"] == "complete" and not summary["mock"]
        and summary["scientific_usable"] and summary["complete_planned_trajectory"]
        and summary["states_observed"] == 501 and summary["updates_observed"] == 500
        and summary["final_observed"] and status["updates_completed"] == 500
        and cfg["updates"] == summary["planned_updates"] == 500
        and cfg["n_atoms"] == 25 and cfg["learning_rate"] == 60 and cfg["native_weight"] == 1
        and cfg["initialization"] == "fragmented" and cfg["condition"] == planned["condition"])
    if not complete:
        raise ValueError(f"Inconsistent native endpoint or configuration for {planned['run_id']}")
    if summary["any_valid"] and not math.isfinite(summary["best_valid_hard_score"]):
        raise ValueError(f"Missing finite best valid score for {planned['run_id']}")
    condition = cfg["condition"]
    if condition["projection"]["mode"] != "connected" or condition["interior_uniform_mass"] != 1e-4:
        raise ValueError(f"Unexpected projection or mixing in {planned['run_id']}")
    hardware = cfg["hardware"]
    scorer = cfg["scorer_identity"]
    identity = {
        "initial_arrays": initial_identity(folder / "initial.npz"),
        "model_key": cfg["model_key"], "model_key_impl": cfg["model_key_impl"],
        "model_digest": scorer["model_digest"], "context_digest": scorer["context_digest"],
        "hardware": hardware, "bundle": cfg["writer_bundle_id"],
        "projection": condition["projection"], "cooling": condition["cooling"],
        "source_sha256": cfg["source_sha256"],
    }
    if not hardware.get("job_id") or not hardware.get("host") or not hardware.get("devices"):
        raise ValueError(f"Missing allocation identity for {planned['run_id']}")
    result.update({k: summary[k] for k in FIELDS})
    result["teacher_queries"] = summary.get("teacher_queries")
    result["frozen_checks_passed"] = True
    return result, identity


def compare(root):
    plan = read(root / "plan.json")
    specs = {(r["seed"], r["condition_id"]): r for r in plan["runs"]}
    if plan.get("mock") or set(specs) != {(s, a) for s in SEEDS for a in ARMS} or len(plan["runs"]) != 6:
        raise ValueError("Expected exactly three paired native runs per method")
    pairs = []
    for seed in SEEDS:
        pair = {"seed": seed, "arms": {}, "matched": False}
        identities = {}
        for arm in ARMS:
            result, identity = inspect_run(root, specs[seed, arm])
            pair["arms"][arm] = result
            identities[arm] = identity
        if all(identities.values()):
            left, right = (identities[a] for a in ARMS)
            diff = [k for k in left if left[k] != right[k]]
            if diff:
                raise ValueError(f"Seed {seed} differs across arms in {', '.join(diff)}")
            pair["matched"] = True
            pair["match_evidence"] = left
            scores = [pair["arms"][a]["best_valid_hard_score"] for a in ARMS]
            pair["best_valid_delta_defog_minus_repair"] = (
                scores[0] - scores[1] if all(s is not None for s in scores) else None)
        pairs.append(pair)

    matched = [p for p in pairs if p["matched"]]
    aggregates = {}
    for arm in ARMS:
        rows = [p["arms"][arm] for p in matched]
        scored = [r["best_valid_hard_score"] for r in rows if r["best_valid_hard_score"] is not None]
        aggregates[arm] = {
            "matched_runs": len(rows), "any_new_valid_runs": sum(r["any_new_valid"] for r in rows),
            "final_valid_runs": sum(r["final_valid"] for r in rows),
            "valid_states": sum(r["valid_state_count"] for r in rows),
            "observed_states": sum(r["states_observed"] for r in rows),
            "scored_runs": len(scored), "mean_best_valid_hard_score": statistics.fmean(scored) if scored else None,
        }
    deltas = [p["best_valid_delta_defog_minus_repair"] for p in matched
              if p["best_valid_delta_defog_minus_repair"] is not None]
    return {"campaign_id": plan["campaign_id"], "pairs": pairs, "aggregate": aggregates,
            "matched_pairs": len(matched), "scored_pairs": len(deltas),
            "defog_score_wins": sum(d > 0 for d in deltas), "repair_score_wins": sum(d < 0 for d in deltas),
            "score_ties": sum(d == 0 for d in deltas),
            "mean_paired_score_delta": statistics.fmean(deltas) if deltas else None}


def report(result):
    lines = ["# DeFoG and structural repair with connected projection", "",
             f"{result['matched_pairs']}/3 matched pairs completed 500 updates. Both methods use "
             "the same fragmented start, frozen BRD4 Protenix context and key, and GPU within each seed.",
             "", "| Seed | Method | Status | New valid | Valid states | Distinct valid | Best valid contact (update) | Final valid | Final components |",
             "|---:|---|---|---|---:|---:|---:|---|---:|"]
    for pair in result["pairs"]:
        for arm, row in pair["arms"].items():
            label = "DeFoG" if arm == "defog-w0p5" else "Structural repair"
            if row["status"] != "complete":
                progress = row.get("updates_completed")
                state = row["status"] + (f" ({progress}/500)" if progress is not None else "")
                lines.append(f"| {pair['seed']} | {label} | {state} | — | — | — | — | — | — |")
                continue
            score = row["best_valid_hard_score"]
            best = f"{score:.4f} ({row['best_valid_update']})" if score is not None else "—"
            lines.append(f"| {pair['seed']} | {label} | complete | {'yes' if row['any_new_valid'] else 'no'} | "
                         f"{row['valid_state_count']}/{row['states_observed']} | {row['unique_valid_molecules']} | "
                         f"{best} | {'yes' if row['final_valid'] else 'no'} | {row['final_components']} |")
    lines.extend(["", "Valid means the raw argmax sanitized and formed one connected component. "
                  "Hard contact is a model score, with higher values preferred. State counts include step zero "
                  "and repeated visits to the same molecule."])
    if result["matched_pairs"]:
        lines.extend(["", "| Completed paired runs | New valid | Final valid | Valid states | Mean best when valid |",
                      "|---|---:|---:|---:|---:|"])
        for arm, a in result["aggregate"].items():
            n = a["matched_runs"]
            score = a["mean_best_valid_hard_score"]
            score_text = f"{score:.4f} ({a['scored_runs']}/{n} scored)" if score is not None else "—"
            lines.append(f"| {arm} | {a['any_new_valid_runs']}/{n} | {a['final_valid_runs']}/{n} | "
                         f"{a['valid_states']}/{a['observed_states']} | {score_text} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = compare(args.campaign)
    text = report(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
