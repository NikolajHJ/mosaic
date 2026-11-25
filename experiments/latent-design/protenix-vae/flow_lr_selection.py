"""Select the latent SGD learning rate using only disjoint tuning cohorts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def load_config(path, seen=()):
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic study config inheritance")
    config = json.loads(path.read_text(encoding="utf-8"))
    if base := config.get("base_config"):
        return load_config(path.parent / base, (*seen, path)) | config
    return config


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rate_arms(config):
    rates = config["learning_rates"]
    if not rates or any(type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0
                        for rate in rates):
        raise ValueError("Require finite positive learning rates")
    arms = {f"lr_{rate:g}": rate for rate in rates}
    if len(arms) != len(rates):
        raise ValueError("Learning rates must be distinct")
    return arms


def select_rate(roots, config):
    rates = rate_arms(config)
    scope = config_digest(config)
    expected = set(config["tuning_seeds"])
    observed = set()
    identities = {}
    all_smiles = set()
    summaries = {arm: [] for arm in rates}
    for root in roots:
        panel = json.loads((root / "tuning.json").read_text(encoding="utf-8"))
        fresh = json.loads((root / "fresh" / "summary.json").read_text(encoding="utf-8"))
        seed = panel["seed"]
        if (seed not in expected - observed or panel["config_digest"] != scope
                or panel["checkpoint_sha256"] != config["checkpoint_sha256"]
                or panel["job_id"] != fresh["job_id"] or fresh["status"] != "completed"):
            raise ValueError(f"Foreign, duplicate, or incomplete tuning cohort: {root}")
        starts = panel["starts"]
        if (len(starts) != config["tuning_starts_per_cohort"] or set(starts) & identities.keys()
                or set(starts.values()) & all_smiles or len(set(starts.values())) != len(starts)):
            raise ValueError("Tuning cohorts must have distinct seeds and starting molecules")
        identities.update(starts)
        all_smiles.update(starts.values())
        observed.add(seed)
        groups = {}
        for row in fresh["candidates"]:
            groups.setdefault((row["start"], row["arm"]), []).append(row)
        if set(groups) != {(start, arm) for start in starts for arm in rates}:
            raise ValueError("The same tuning starts must be scored for all rates")
        for start, smiles in starts.items():
            for arm in rates:
                rows = groups[start, arm]
                initial = [row for row in rows if row["role"] == "initial"]
                if len(initial) != 1 or initial[0]["smiles"] != smiles:
                    raise ValueError("Tuning arms must share their starting molecule")
                best_gain = max((max(0., -row["loss_delta_from_valid_start"])
                                 for row in rows if row["role"] != "initial"
                                 and row["loss_delta_from_valid_start"] is not None), default=0.)
                trajectory = panel["trajectories"][start][arm]
                if trajectory["scorer_calls"] != config["refine_updates"] + 1:
                    raise ValueError("Tuning arm had a different scorer budget")
                summaries[arm].append(dict(start=start, gain=best_gain,
                    scored_baseline=initial[0]["loss"] is not None,
                    scored_edits=sum(row["status"] == "scored" for row in rows if row["role"] != "initial"),
                    bounded_steps=trajectory["bounded_steps"], scorer_calls=trajectory["scorer_calls"]))
    if observed != expected:
        raise ValueError("The complete prespecified tuning panel is required before selection")
    table = [dict(rate=rate, mean_gain=mean(row["gain"] for row in summaries[arm]),
                  improved_starts=sum(row["gain"] > config["min_score_gain"] for row in summaries[arm]),
                  usable_baselines=sum(row["scored_baseline"] for row in summaries[arm]),
                  scored_edits=sum(row["scored_edits"] for row in summaries[arm]),
                  bounded_steps=sum(row["bounded_steps"] for row in summaries[arm]),
                  scorer_calls=sum(row["scorer_calls"] for row in summaries[arm]),
                  starts=summaries[arm]) for arm, rate in rates.items()]
    best = max(table, key=lambda row: (row["mean_gain"], row["improved_starts"], -row["rate"]))
    return dict(status="selected", config_digest=scope, tuning_seeds=sorted(observed),
                learning_rate=best["rate"], tuning_starts=identities, table=table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_rate(args.tuning, load_config(args.config))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(learning_rate=result["learning_rate"], tuning_starts=len(result["tuning_starts"]))))


if __name__ == "__main__":
    main()
