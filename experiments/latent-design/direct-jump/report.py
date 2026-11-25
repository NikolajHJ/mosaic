"""Paired held-out comparison of direct graph policies, stratified by training cohort."""

import argparse
import json
import math
from pathlib import Path
import random
from statistics import fmean


MODELS = {
    "protenix": "fresh FILE negative mean log pocket-contact probability",
    "boltz2": "fresh native Boltz-2 cofold affinity_pred_value",
}
ARMS = ("learned", "random", "restart")
COHORTS = (0, 1)
STARTS = 32
THRESHOLD = 0.005
RESAMPLES = 10000
BOOTSTRAP_SEED = 20261119


def checked_loss(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError(f"Expected finite loss or null, got {value!r}")
    return value


def check_row(row):
    if type(row["seed"]) is not int or not isinstance(row["founder"], str) or not row["founder"]:
        raise ValueError("Missing seed or founder")
    if set(row["arms"]) != set(ARMS):
        raise ValueError("Wrong set of held-out arms")
    base = checked_loss(row["baseline_loss"])
    for arm, result in row["arms"].items():
        best = checked_loss(result["best_loss"])
        n, scored = result["candidates"], result["scored"]
        if result["calls"] != 8 or type(n) is not int or type(scored) is not int or not 0 <= scored <= n <= 4:
            raise ValueError(f"Wrong relaxed or fresh scorer budget for {arm} on {row['seed']}")
        if (best is None) != (base is None and scored == 0):
            raise ValueError(f"Missing or unexpected best score for {arm} on {row['seed']}")
        if base is not None and (best > base or scored == 0 and best != base):
            raise ValueError(f"Inconsistent founder/best score for {arm} on {row['seed']}")


def read_cohorts(root, model):
    panels, identities, seen_seeds, seen_founders = [], {}, set(), set()
    for cohort in COHORTS:
        path = root / f"{model}-{cohort}" / "eval" / "summary.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj["rows"]
        if obj["model"] != model or str(obj["cohort"]) != str(cohort) or len(rows) != STARTS:
            raise ValueError(f"Wrong model, cohort or held-out start count in {path}")
        for row in rows:
            check_row(row)
            seed, founder = row["seed"], row["founder"]
            if seed in seen_seeds or founder in seen_founders:
                raise ValueError(f"Duplicate held-out seed or founder in {path}")
            seen_seeds.add(seed)
            seen_founders.add(founder)
            identities[cohort, seed] = founder
        panels.append(sorted(rows, key=lambda r: r["seed"]))
    return panels, identities


def interval(values):
    values = sorted(values)
    result = []
    for p in (0.025, 0.975):
        pos = (len(values) - 1) * p
        a, b = math.floor(pos), math.ceil(pos)
        result.append(values[a] + (pos - a) * (values[b] - values[a]))
    return result


def paired_outcomes(differences):
    return {"wins": sum(d > 0 for d in differences), "losses": sum(d < 0 for d in differences),
            "ties": sum(d == 0 for d in differences)}


def summarize(panels):
    rows = [r for cohort in panels for r in cohort]
    groups = []
    offset = 0
    for cohort in panels:
        groups.append(range(offset, offset + len(cohort)))
        offset += len(cohort)
    base_count = sum(row["baseline_loss"] is not None for row in rows)
    arm_data, contrast_data, bootstrap_columns = {}, {}, []
    gains, hits = {}, {}

    def estimate(key, sample, record):
        record[key] = {"estimate": fmean(sample)}
        bootstrap_columns.append((sample, record[key]))

    for arm in ARMS:
        observed = [r["arms"][arm] for r in rows]
        gains[arm] = [r["baseline_loss"] - o["best_loss"] if r["baseline_loss"] is not None else 0.
                      for r, o in zip(rows, observed)]
        hits[arm] = [int(g > THRESHOLD) for g in gains[arm]]
        best = [o["best_loss"] for o in observed if o["best_loss"] is not None]
        record = {
            "verified_improvements": sum(hits[arm]),
            "starts_with_scorable_founder": base_count,
            "starts_with_scorable_candidate": sum(o["scored"] > 0 for o in observed),
            "starts_with_any_score_including_founder": len(best),
            "fresh_candidates": sum(o["candidates"] for o in observed),
            "fresh_candidates_scored": sum(o["scored"] for o in observed),
            "mean_best_loss_among_starts_with_any_score": fmean(best) if best else None,
            "mean_gain_among_scorable_founders": sum(gains[arm]) / base_count if base_count else None,
        }
        estimate("mean_best_gain", gains[arm], record)
        estimate("verified_improvement_fraction", hits[arm], record)
        arm_data[arm] = record
    for control in ("random", "restart"):
        ds = [a - b for a, b in zip(gains["learned"], gains[control])]
        dh = [a - b for a, b in zip(hits["learned"], hits[control])]
        record = {"paired_best_gain_outcomes": paired_outcomes(ds),
                  "paired_verified_improvement_outcomes": paired_outcomes(dh)}
        estimate("mean_best_gain_difference", ds, record)
        estimate("verified_improvement_fraction_difference", dh, record)
        contrast_data[f"learned_minus_{control}"] = record

    rng = random.Random(BOOTSTRAP_SEED)
    draws = [[] for _ in bootstrap_columns]
    for _ in range(RESAMPLES):
        # The same sampled starts enter every arm, sampled within fitted cohort.
        indices = [i for group in groups for i in rng.choices(group, k=len(group))]
        for (values, _), replicates in zip(bootstrap_columns, draws):
            replicates.append(sum(values[i] for i in indices) / len(indices))
    for (_, record), replicates in zip(bootstrap_columns, draws):
        record["ci95"] = interval(replicates)
    return {"starts": len(rows), "scorable_founders": base_count,
            "arms": arm_data, "paired_contrasts": contrast_data}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.models) != len(set(args.models)):
        raise ValueError("Duplicate model selection")
    identity_reference = None
    models = {}
    for model in args.models:
        panels, identities = read_cohorts(args.root, model)
        if identity_reference is not None and identity_reference != identities:
            raise ValueError("Held-out founders, cohort assignments or seeds differ between scorers")
        identity_reference = identities
        models[model] = {"objective": MODELS[model], **summarize(panels),
                         "cohorts": [dict(cohort=k, starts=len(p), scorable_founders=sum(
                             r["baseline_loss"] is not None for r in p))
                                     for k, p in zip(COHORTS, panels)]}
    report = {
        "models": models,
        "cross_scorer_heldout_identities_verified": len(models) == len(MODELS),
        "relaxed_calls_per_arm": 8, "distinct_edit_budget_per_arm": 4,
        "verified_improvement_rule": "fresh founder loss minus best fresh arm loss > 0.005",
        "gain_policy": "Founder retained when scorable. Unscorable founder contributes zero verified gain to each arm.",
        "interval": {"method": "paired percentile bootstrap of held-out starts within each fitted cohort",
                     "replicates": RESAMPLES, "seed": BOOTSTRAP_SEED, "confidence": 0.95},
    }
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
