"""Read completed wave archives at common update and fresh-score ceilings."""
import argparse
import json
from pathlib import Path


def read_waves(root):
    waves = []
    scored = 0
    run = root / "run.json"
    previous = json.loads(run.read_text(encoding="utf-8")).get("previous_waves", []) if run.is_file() else []
    rows = list(previous)
    for path in sorted(root.glob("cycle-*/archive.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            # An active worker may be in the middle of publishing this archive.
            break
    for row in rows:
        if row["cycle"] != len(waves) + 1:
            raise ValueError(f"Expected consecutive waves in {root}")
        scored += row["fresh_scored"]
        waves.append({key: row[key] for key in (
            "cycle", "cumulative_updates", "cumulative_fresh_attempts", "best_fresh_loss",
            "top_three_mean_fresh_loss", "distinct_high_scorers",
        )} | {"cumulative_fresh_scored": scored})
    return waves


def compare(roots):
    series = {str(root): read_waves(root) for root in roots}
    available = {name: waves for name, waves in series.items() if waves}
    shared = {}
    if len(available) >= 2:
        thresholds = set()
        for root in roots:
            if str(root) in available:
                search = json.loads((root / "run.json").read_text(encoding="utf-8"))["search"]
                thresholds.add((search["high_score_ceiling"], search["diversity_max_similarity"]))
        if len(thresholds) != 1:
            raise ValueError("Distinct low-loss counts use different score or similarity thresholds")
        for axis in ("cumulative_updates", "cumulative_fresh_attempts", "cumulative_fresh_scored"):
            ceiling = min(waves[-1][axis] for waves in available.values())
            points = {name: next((row for row in reversed(waves) if row[axis] <= ceiling), None)
                      for name, waves in available.items()}
            shared[axis] = dict(ceiling=ceiling, exact=all(
                row is not None and row[axis] == ceiling for row in points.values()), runs=points)
    return dict(latest={name: waves[-1] for name, waves in available.items()},
                no_completed_wave=[name for name, waves in series.items() if not waves], shared=shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.runs), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
