"""Summarize the archived Boltz/DeFoG target changes against the raw input graphs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


def summarize(report):
    raw = report.read_bytes()
    data = raw.split(b'<script id="report-data" type="application/json">', 1)[1].split(b'</script>', 1)[0]
    frames = json.loads(data)["runs"][0]["trajectories"][0]["frames"]
    if [f["step"] for f in frames] != list(range(501)):
        raise ValueError("Expected all 501 ordered states")
    snapshots = frames[:500:5]
    targets = [f["repair_target_sha256"] for f in snapshots]
    if not all(targets) or any(f["repair_target_sha256"] != targets[min(f["step"] // 5, 99)] for f in frames):
        raise ValueError("Target missing or changed outside expected refresh cadence")
    comparisons = []
    for previous, current in zip(snapshots, snapshots[1:]):
        graphs = []
        for frame in (previous, current):
            p = frame["probabilities"]
            if p["active_slots"] != snapshots[0]["probabilities"]["active_slots"]:
                raise ValueError("Occupied slots changed")
            occupied = [p["row_slots"].index(slot) for slot in p["active_slots"]]
            rows = [p[name][i] for name in ("atoms", "charges") for i in occupied]
            rows.extend(p["bonds"][i][j] for ix, i in enumerate(occupied) for j in occupied[ix+1:])
            graphs.append(rows)
        differences = [sum(abs(a - b) for a, b in zip(left, right)) / 2
                       for left, right in zip(*graphs)]
        def winner(row):
            return max(range(len(row)), key=row.__getitem__)
        comparisons.append({
            "step": current["step"],
            "target_changed": current["repair_target_sha256"] != previous["repair_target_sha256"],
            "input_argmax_edits": sum(winner(a) != winner(b) for a, b in zip(*graphs)),
            "input_mean_row_tv": sum(differences) / len(differences),
        })
    return {"report_sha256": sha256(raw).hexdigest(), "refreshes": len(snapshots),
            "distinct_ordered_targets": len(set(targets)),
            "changed_adjacent_targets": sum(r["target_changed"] for r in comparisons),
            "unchanged_adjacent_input_argmax": sum(r["input_argmax_edits"] == 0 for r in comparisons),
            "median_input_mean_row_tv": median(r["input_mean_row_tv"] for r in comparisons),
            "pairs": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.report)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print({k: v for k, v in result.items() if k != "pairs"})
