"""Combine disjoint saved parent groups. This command only reads experiment outputs."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from latentmol.design.same_molecule_comparison import ARM_NAMES, matched_fresh_comparison


def read(path):
    return json.loads(path.read_bytes())


def combine(panel_path, directories):
    panel = read(panel_path)
    panel_digest = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    parents = [row["name"] for row in panel["parents"]]
    study, fresh, candidates = {}, {}, []
    seen, sources = set(), []
    for folder in directories:
        run = read(folder / "result.json")
        score = read(folder / "fresh" / "summary.json")
        chosen = read(folder / "fresh-candidates.json")["candidates"]
        evidence = matched_fresh_comparison(run, score, chosen)
        first, stop = evidence["parent_slice"]
        names = parents[first:stop]
        if (not 0 <= first < stop <= len(parents) or list(run["starts"]) != names
                or evidence["panel_sha256"] != panel_digest
                or evidence["checkpoint_sha256"] != panel["checkpoint_sha256"]
                or seen.intersection(names)):
            raise ValueError("Overlapping, reordered or foreign parent groups")
        if score.get("job_id") != run.get("job_id") or score["status"] != "completed" or run["status"] != "completed":
            raise ValueError("Trajectory and fresh evaluation must be completed in the same allocation")
        contract = ("checkpoint_sha256", "native_config_sha256", "same_molecule_panel", "fixed_context",
                    "step_search", "optimization_scorer_budget_per_arm", "fresh_edit_limit_per_arm", "updates_per_arm")
        if not study:
            study = dict(run, starts={})
            fresh = dict(score, candidates=[], unique_candidates=0)
        if any(study[k] != run[k] for k in contract) or any(
                fresh[k] != score[k] for k in ("geometry_seed", "objective", "max_edit_candidates_per_arm")):
            raise ValueError("Shard scoring protocols differ")
        seen.update(names)
        sources.append(dict(job=run["job_id"], parents=names, study_config_sha256=run["study_config_sha256"]))
        study["starts"].update(run["starts"])
        fresh["candidates"].extend(score["candidates"])
        fresh["unique_candidates"] += score["unique_candidates"]
        candidates.extend(chosen)
    # The per-shard checks above verify the physical slices. The pooled view can
    # include any subset of completed parents, including a later-finishing shard.
    study["parent_slice"] = [0, len(seen)]
    result = matched_fresh_comparison(study, fresh, candidates)
    del result["parent_slice"]
    result.update(status="completed" if len(seen) == len(parents) else "partial",
                  parents_measured=len(seen), parents_planned=len(parents),
                  pending_parents=[name for name in parents if name not in seen], allocations=sources)
    pairwise = []
    for left, right in itertools.combinations(ARM_NAMES, 2):
        differences = [p["arms"][left]["confirmed_improving_edits"] - p["arms"][right]["confirmed_improving_edits"]
                       for p in result["per_parent"] if p["baseline_loss"] is not None]
        pairwise.append(dict(left=left, right=right, more=sum(d > 0 for d in differences),
                             tied=differences.count(0), fewer=sum(d < 0 for d in differences)))
    result["paired_confirmed_hit_counts"] = pairwise
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--panel", required=True, type=Path)
    cli.add_argument("--run", required=True, action="append", type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    result = combine(args.panel, args.run)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "parents_measured", "totals")}, indent=2))
