"""Collect the paired native BRD4 repair-weight experiment from its staged GPU plan."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def load_json(path):
    return json.loads(path.read_bytes())


def read_arm(result, seed, weight):
    if result["seed"] != seed or result["repair_weight"] != weight:
        raise ValueError("Arm seed or repair weight differs from the plan")
    fields = ("starting_smiles", "starting_hard_contact_loss", "selected", "updates",
              "fraction_valid_steps", "distinct_valid", "distinct_new_valid", "mean_sulfur_drift",
              "mean_final_leaf_endpoint_sulfur_drift", "endpoint_sulfur_drifts", "founder_only")
    row = {k: result[k] for k in fields}
    if not row["starting_smiles"] or not math.isfinite(row["starting_hard_contact_loss"]):
        raise ValueError("Missing founder identity or finite hard score")
    if row["updates"] < 1 or not 0 <= row["fraction_valid_steps"] <= 1:
        raise ValueError("Invalid update count or valid fraction")
    for selected in (row["selected"], row["founder_only"]["best"]):
        if selected is None:
            continue
        delta = selected["selected_hard_contact_loss"] - row["starting_hard_contact_loss"]
        if (not selected["branch"] or selected["step"] < 0 or
                not selected["smiles"] or selected["smiles"] == row["starting_smiles"] or
                not delta < 0 or not math.isclose(selected["score_change"], delta, abs_tol=1e-8)):
            raise ValueError("Selected molecule does not improve on its recorded start")
    if (row["distinct_valid"] < row["distinct_new_valid"] or row["distinct_new_valid"] < 0 or
            not math.isfinite(row["mean_sulfur_drift"]) or not math.isfinite(row["mean_final_leaf_endpoint_sulfur_drift"])):
        raise ValueError("Invalid distinct molecule count or sulfur drift")
    json.dumps(row, allow_nan=False)
    return row


def read_task(name, task, weights, plan_dir):
    folder = plan_dir / task["output"]
    seed, policy = task["seed"], task["policy"]
    entry = dict(task=name, policy=policy, seed=seed, output=str(folder), errors=[], arms={})
    founder_file = folder / "founder.json"
    if founder_file.exists():
        try:
            founder = load_json(founder_file)
            digest = founder["start_sha256"]
            if (founder["seed"] != seed or not founder["smiles"] or len(digest) != 64 or
                    any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Invalid founder identity")
            entry["founder"] = {"smiles": founder["smiles"], "start_sha256": digest}
        except (OSError, KeyError, TypeError, ValueError) as exc:
            entry["errors"].append(f"founder.json: {exc}")
    final = folder / "result.json"
    receipt = None
    if final.exists():
        try:
            data = load_json(final)
            if data["status"] != "completed" or data["seed"] != seed or data["policy"] != policy:
                raise ValueError("Completed result identity differs from plan")
            receipt = {float(k): v for k, v in data["arms"].items()}
            if set(receipt) != set(weights):
                raise ValueError("Expected arm(s) differ from completed task")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            entry["errors"].append(f"{final.name}: {exc}")
            receipt = None
    for weight in weights:
        arm_path = folder / f"repair-{weight:g}"
        arm_file = arm_path / "result.json"
        row = dict(status="awaiting_output")
        try:
            if receipt is not None:
                row = dict(status="completed", **read_arm(receipt[weight], seed, weight))
            elif arm_file.exists():
                row = dict(status="provisional", **read_arm(load_json(arm_file), seed, weight))
            elif arm_path.is_dir():
                progress = sorted(arm_path.glob("generation-*/progress.json"))
                row = dict(status="partial", latest_completed_generation=load_json(progress[-1]) if progress else None)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            row = dict(status="unreadable")
            entry["errors"].append(f"weight {weight:g}: {exc}")
        entry["arms"][str(weight)] = row
    entry["status"] = ("receipt_error" if entry["errors"] else "completed" if receipt is not None
                       else "partial" if folder.is_dir() else "awaiting_output")
    return entry


def pair_seed(policy, seed, jobs, weights):
    arms = {}
    errors = []
    founders = []
    for job in jobs:
        if set(arms).intersection(job["arms"]):
            raise ValueError(f"Duplicate repair weight for {policy}, seed {seed}")
        arms.update({w: dict(row, task=job["task"]) for w, row in job["arms"].items()})
        errors.extend(f"{job['task']}: {e}" for e in job["errors"])
        if "founder" in job:
            founders.append(job["founder"])
    if set(arms) != {str(w) for w in weights}:
        raise ValueError(f"Incomplete planned weight panel for {policy}, seed {seed}")
    arms = {str(w): arms[str(w)] for w in weights}
    observed = [r for r in arms.values() if "starting_smiles" in r]
    smiles = {r["starting_smiles"] for r in observed} | {f["smiles"] for f in founders}
    hashes = {f["start_sha256"] for f in founders}
    losses = [r["starting_hard_contact_loss"] for r in observed]
    # Reduce-scatter / matmul ordering may vary slightly between FP32 GPU allocations.
    # Accept a small absolute difference in this probability-valued contact score.
    spread = max(losses) - min(losses) if losses else None
    mismatch = len(smiles) > 1 or len(hashes) > 1 or (spread is not None and spread > 1.e-5)
    if mismatch:
        errors.append("Arms have different founder probabilities, molecular identities or hard contact scores")
        for row in observed:
            row["status"] = "unpaired"
    complete = all(row["status"] == "completed" for row in arms.values())
    status = ("receipt_error" if errors else "completed" if complete else
              "awaiting_output" if all(j["status"] == "awaiting_output" for j in jobs) else "partial")
    return dict(policy=policy, seed=seed, jobs=[j["task"] for j in jobs], errors=errors, arms=arms,
                status=status, paired_complete=complete and not errors, founder_hashes=sorted(hashes),
                founder_hard_loss_spread=spread, founder_hard_loss_tolerance=1.e-5)


def collect(plan_path):
    plan = load_json(plan_path)
    if "repair_weights" in plan:
        raw_weights = plan["repair_weights"]
    else:
        cfg = (plan_path.parent / "repair-search.json").read_bytes()
        if hashlib.sha256(cfg).hexdigest() != plan["config_sha256"]:
            raise ValueError("Staged repair config differs from the GPU plan")
        raw_weights = json.loads(cfg)["repair_weights"]
    weights = sorted(map(float, raw_weights))
    if not weights or len(set(weights)) != len(weights) or not all(math.isfinite(w) for w in weights):
        raise ValueError("Missing, nonfinite, or duplicate repair weights")
    tasks = []
    by_seed = {}
    for name, task in sorted(plan["tasks"].items()):
        task_weights = list(map(float, task.get("repair_weights", weights)))
        if not task_weights or len(set(task_weights)) != len(task_weights) or set(task_weights) - set(weights):
            raise ValueError(f"Invalid repair weights assigned to {name}")
        entry = read_task(name, task, task_weights, plan_path.parent)
        tasks.append(entry)
        by_seed.setdefault((entry["policy"], entry["seed"]), []).append(entry)
    if not tasks:
        raise ValueError("Empty repair plan")
    seeds = [pair_seed(policy, seed, jobs, weights) for (policy, seed), jobs in sorted(by_seed.items())]
    summary, winners = [], []
    for policy in sorted({t["policy"] for t in seeds}):
        group = [t for t in seeds if t["policy"] == policy]
        for w in weights:
            rows = [(t["seed"], t["arms"][str(w)]) for t in group]
            completed = [seed for seed, r in rows if r["status"] == "completed"]
            improved = [seed for seed, r in rows if r["status"] == "completed" and r["selected"] is not None]
            provisional = [seed for seed, r in rows if r["status"] == "provisional"]
            summary.append(dict(policy=policy, repair_weight=w, planned_seeds=len(group),
                                completed_seeds=completed, improved_seeds=improved, provisional_seeds=provisional,
                                paired_completed_seeds=[t["seed"] for t in group if t["paired_complete"]],
                                unresolved_seeds=[seed for seed, r in rows if r["status"] != "completed"]))
        for t in group:
            finished = {w: r for w, r in t["arms"].items() if r["status"] == "completed"}
            hits = [(float(w), r["selected"]) for w, r in finished.items() if r["selected"] is not None]
            best = min((h["selected_hard_contact_loss"] for _, h in hits), default=None)
            winners.append(dict(policy=policy, seed=t["seed"], completed_weights=list(map(float, finished)),
                                planned_weights=weights, paired_complete=t["paired_complete"], best_hard_contact_loss=best,
                                winners=[dict(repair_weight=w, **hit) for w, hit in hits
                                         if hit["selected_hard_contact_loss"] == best]))
    return dict(generated_utc=datetime.now(timezone.utc).isoformat(), plan=str(plan_path),
                score="predicted pocket contact", lower_is_better=True, weights=weights,
                tasks=tasks, seeds=seeds, summary=summary, best_per_start=winners)


def markdown(report):
    def number(value):
        return f"{value:.5f}" if value is not None else "—"

    lines = ["# BRD4 repair-weight comparison", "",
             "Lower hard Protenix predicted pocket contact loss is better. A hit is a distinct valid molecule "
             "with a lower hard score than its own seed’s founder. Vina contributes to parent choice.", "",
             "| Parent policy | Repair weight | Seeds improved / completed | Seeds planned | Complete paired seeds | Provisional arms |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in report["summary"]:
        lines.append(f"| {r['policy']} | {r['repair_weight']:g} | {len(r['improved_seeds'])} / "
                     f"{len(r['completed_seeds'])} | {r['planned_seeds']} | {len(r['paired_completed_seeds'])} | "
                     f"{len(r['provisional_seeds'])} |")
    lines += ["", "Only completed task receipts enter the denominator. Provisional arms have an arm result "
              "and await the task receipt. Missing outputs give no information about scheduler state.", ""]
    winners = {(r["policy"], r["seed"]): r for r in report["best_per_start"]}
    for task in report["seeds"]:
        winner = winners[task["policy"], task["seed"]]
        lines += [f"## {task['policy']} · seed {task['seed']}", "", f"Paired comparison is **{task['status']}**.", ""]
        for err in task["errors"]:
            lines += [f"Receipt issue — {err}", ""]
        rows = task["arms"]
        observed = [r for r in rows.values() if "starting_smiles" in r]
        if observed:
            start = observed[0]
            lines += [f"Founder `{start['starting_smiles']}` · hard loss {number(start['starting_hard_contact_loss'])}.", ""]
        lines += ["| Weight | Status | Updates | Valid steps | Distinct valid (new) | Founder-only best loss | "
                  "Search best loss | Change | Mean S drift | Mean leaf-end S drift |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for w, row in rows.items():
            status = row["status"]
            if "selected" not in row:
                progress = row.get("latest_completed_generation")
                if progress:
                    status += f", through generation {progress['generation']} ({progress['updates']} updates)"
                lines.append(f"| {float(w):g} | {status} | — | — | — | — | — | — | — | — |")
                continue
            selected = row["selected"] or {}
            founder = row["founder_only"]["best"] or {}
            lines.append(f"| {float(w):g} | {status} | {row['updates']} | {row['fraction_valid_steps']:.1%} | "
                         f"{row['distinct_valid']} ({row['distinct_new_valid']}) | "
                         f"{number(founder.get('selected_hard_contact_loss'))} | "
                         f"{number(selected.get('selected_hard_contact_loss'))} | {number(selected.get('score_change'))} | "
                         f"{row['mean_sulfur_drift']:.2f} | {row['mean_final_leaf_endpoint_sulfur_drift']:.2f} |")
        lines.append("")
        for w, row in rows.items():
            if "selected" in row:
                hit = row["selected"]
                desc = (f"`{hit['smiles']}` · {hit['branch']}, step {hit['step']}."
                        if hit else "No qualifying molecule.")
                lines.append(f"- Weight {float(w):g} · {desc}")
        if winner["winners"]:
            labels = ", ".join(f"{h['repair_weight']:g}" for h in winner["winners"])
            lines += ["", f"Lowest hard loss among completed weights **{number(winner['best_hard_contact_loss'])}** "
                      f"(repair weight {labels})."]
        lines.append("")
    lines += ["Valid-step fractions and mean sulfur drift use the states that generated optimizer updates. "
              "Sulfur drift is the signed change in hard sulfur atom count from the seed’s founder. "
              "Leaf-end drift averages the endpoints of all final-generation branches. "
              "The JSON records each leaf and the founder-only comparison.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.plan.resolve())
    encoded = json.dumps(report, indent=2, allow_nan=False)
    rendered = markdown(report)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(encoded + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
