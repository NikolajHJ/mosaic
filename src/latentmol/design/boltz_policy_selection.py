"""Select two BRD4 generation policies from the completed paired allocation pilots."""
import argparse
import hashlib
import json
import math
from pathlib import Path


# These are the original allocation jobs. The independent seed panels also
# used different GPU families, so elapsed time and loss are compared within seed.
SOURCE = "1df1a7488b4a9a822fc960420a3163babd1f5df55710929ffe6425f72aba9ba1"
POLICIES = ("fixed150", "fixed400", "extend")
COUNTS = {"fixed150": 16, "fixed400": 6, "extend": 12}
JOBS = {
    811211: {"fixed150": 30099849, "fixed400": 30099848, "extend": 30099850},
    813211: {"fixed150": 30099855, "fixed400": 30099854, "extend": 30099856},
    817211: {"fixed150": 30099861, "fixed400": 30099860, "extend": 30099863},
}


def _validate(row, expected_jobs):
    seed, key = row["case"]["seed"], row["case_name"]
    if key not in POLICIES or seed not in JOBS:
        raise ValueError(f"Unexpected allocation pilot {key}, seed {seed}")
    policy = "promote" if key == "extend" else key
    if (str(row["job_id"]) != str(expected_jobs[seed][key]) or row["source_sha256"] != SOURCE
            or row["status"] != "completed" or row["full_objective"] != "fixed_pose_dbmol"
            or row["case"] != dict(seed=seed, start="archive", count=COUNTS[key],
                                   policy=policy, fresh_budget=48, updates=2400)
            or row["updates"] != 2400 or row["fresh_budget"] != 48 or row["policy"] != policy
            or row["execution"]["autocast"] != "none" or row["execution"]["tf32"] is not False):
        raise ValueError(f"Wrong job, inputs, completion status or budget for {key}, seed {seed}")
    hashes = row["start_hashes"]
    if len(hashes) != COUNTS[key] or len(set(hashes)) != len(hashes):
        raise ValueError("Incorrect number of distinct child starts")
    setup, total = row["setup_seconds"], row["total_seconds"]
    if not (math.isfinite(total) and 0 <= setup < total):
        raise ValueError("Invalid pilot runtime")
    prev_step = prev_time = 0
    prev_count = prev_attempts = 0
    for observation in row["milestones"]:
        step, elapsed = observation["updates"], observation["elapsed_seconds"]
        count, attempts = observation["distinct_scored"], observation["fresh_attempts"]
        if (not (prev_step < step <= 2400 and prev_time <= elapsed <= total - setup)
                or not (prev_count <= count <= attempts <= 48 and attempts >= prev_attempts)):
            raise ValueError("Invalid cumulative pilot milestone")
        for field, eligible in (("best_loss", count > 0), ("tenth_loss", count >= 10)):
            value = observation[field]
            if (value is not None) != eligible or (eligible and not math.isfinite(value)):
                raise ValueError(f"Missing or invalid milestone {field}")
        prev_step, prev_time, prev_count, prev_attempts = step, elapsed, count, attempts
    if (prev_step != row["updates"] or prev_count != row["distinct_scored"]
            or prev_attempts != row["fresh_attempts"]):
        raise ValueError("Missing final pilot milestone")
    return seed, key


def _at_common_time(row, horizon):
    snapshots = [m for m in row["milestones"] if row["setup_seconds"] + m["elapsed_seconds"] <= horizon]
    last = snapshots[-1] if snapshots else dict(updates=0, fresh_attempts=0, distinct_scored=0,
                                               best_loss=None, tenth_loss=None)
    first_ten = next((row["setup_seconds"] + m["elapsed_seconds"] for m in snapshots
                      if m["distinct_scored"] >= 10), None)
    return {k: last[k] for k in ("updates", "fresh_attempts", "distinct_scored", "best_loss", "tenth_loss")} | {
        "seconds_to_ten": first_ten, "job_id": str(row["job_id"])}


def _quality(snapshot):
    count = snapshot["distinct_scored"]
    if count >= 10:
        return (-10, snapshot["tenth_loss"], snapshot["seconds_to_ten"], snapshot["best_loss"])
    return (-count, snapshot["best_loss"] if count else 0, 0, 0)


def select_policies(results, *, fixed150_sxm2_job=None):
    """Return the choice and evidence; the dispatcher verifies terminal launch receipts."""
    expected_jobs = {seed: dict(arms) for seed, arms in JOBS.items()}
    if fixed150_sxm2_job is not None:
        if (type(fixed150_sxm2_job) is not int or fixed150_sxm2_job <= 0
                or any(fixed150_sxm2_job in arms.values() for arms in JOBS.values())):
            raise ValueError("Expected a new fixed150 SXM2 submission receipt")
        expected_jobs[811211]["fixed150"] = fixed150_sxm2_job
    by_seed = {seed: {} for seed in JOBS}
    for row in results:
        seed, key = _validate(row, expected_jobs)
        if fixed150_sxm2_job and seed == 811211 and row["model_device"] != "Tesla V100-SXM2-32GB":
            raise ValueError("Seed 811211 timing replay requires the original V100 SXM2 32 GB model")
        if key in by_seed[seed]:
            raise ValueError(f"Duplicate result for {key}, seed {seed}")
        by_seed[seed][key] = row
    scores = dict.fromkeys(POLICIES, 0.0)
    panels = []
    informative = False
    for seed, rows in by_seed.items():
        if set(rows) != set(POLICIES):
            raise ValueError(f"Missing completed allocation policies for seed {seed}: {sorted(set(POLICIES)-set(rows))}")
        if len({row["model_device"] for row in rows.values()}) != 1:
            raise ValueError(f"Matched GPU model differs within seed {seed}")
        longest = rows["fixed150"]["start_hashes"]
        for row in rows.values():
            if row["start_hashes"] != longest[:len(row["start_hashes"])]:
                raise ValueError(f"Starting children differ within seed {seed}")
        horizon = min(row["total_seconds"] for row in rows.values())
        snapshots = {key: _at_common_time(row, horizon) for key, row in rows.items()}
        qualities = {key: _quality(value) for key, value in snapshots.items()}
        informative |= len(set(qualities.values())) > 1
        ranks = {}
        for key, value in qualities.items():
            # Tied policies split the ranks they occupy, independently per GPU/seed.
            rank = (1 + sum(v < value for v in qualities.values())
                    + 0.5 * (sum(v == value for v in qualities.values()) - 1))
            ranks[key] = rank
            scores[key] += rank
        panels.append(dict(seed=seed, gpu=rows["fixed150"]["model_device"], common_seconds=horizon,
                           milestones=snapshots, ranks=ranks))
    # Fixed-150 has the simpler short baseline in a tie between fixed arms.
    fixed = min(("fixed150", "fixed400"), key=scores.get)
    candidates = [key for key in ("extend", "fixed150", "fixed400") if key != fixed]
    challenger = min(candidates, key=scores.get)
    return dict(status="selected" if informative else "inconclusive",
                selected_policies=[fixed, challenger] if informative else [],
                reason="paired_rank_sum" if informative else "no_distinguishing_outcome_at_common_time",
                rank_sums=scores, panels=panels, source_sha256=SOURCE,
                rule="Within each seed use the shortest completed job time, including setup. Prefer ten distinct "
                     "molecules, lower tenth loss, shorter observed time to ten, then lower best loss. Below ten "
                     "prefer more distinct molecules, then lower best loss. Sum equal-weight within-seed midranks. "
                     "Choose the strongest fixed control and highest remaining challenger. Fixed-control ties go "
                     "to fixed150 and challenger ties go to extend. Hold when all three outcomes tie in every seed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="The nine original allocation result.json files")
    args = parser.parse_args()
    payloads = [p.read_bytes() for p in args.results]
    decision = select_policies([json.loads(data) for data in payloads])
    decision["result_sha256"] = {str(p): hashlib.sha256(data).hexdigest() for p, data in zip(args.results, payloads)}
    print(json.dumps(decision, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
