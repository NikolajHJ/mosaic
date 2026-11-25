"""Read saved Stage 3 observations into a compact offline report input. No model calls."""
import argparse
import gzip
import json
from pathlib import Path

TRACE = "update hard_score soft_score valid component_count hard_sulfur teacher_weight teacher_time temperature hard_valence_excess".split()
CANDIDATE = "smiles hard_score source_update visits molecular_weight qed logp tpsa is_initial_identity".split()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract(campaign, attempt):
    plan = read(campaign / "plan.json")
    runs = []
    for spec in plan["runs"]:
        folder = campaign / "runs" / spec["run_id"]
        row = read(folder / "summary.json")
        status = read(folder / "status.json")
        config = read(folder / "config.json")
        row.update(attempt=attempt, error=status.get("error"), hardware=config["hardware"],
                   frozen=status.get("frozen_checks_passed"))
        row["molecules"] = [[m.get(k) for k in CANDIDATE] for m in read(folder / "top_valid.json")["molecules"]]
        row["trace"] = []
        every = 1 if spec["condition_id"] in ("control", "weight-0p5") or attempt == "Recovery" else 25
        with gzip.open(folder / "states.jsonl.gz", "rt") as stream:
            count = 0
            for line in stream:
                s = json.loads(line)
                n = s["update"]
                if n % every == 0 or n in (row["best_valid_update"], row["updates_observed"]):
                    row["trace"].append([round(s[k], 6) if isinstance(s[k], float) else s[k] for k in TRACE])
                count += 1
        assert count == row["states_observed"] and row["trace"][-1][0] == row["updates_observed"]
        runs.append(row)
    return plan["conditions"], runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    conditions, original = extract(args.parent / "lf-retry1/campaign", "Original")
    _, recovery = extract(args.parent / "projection-kkt-retry-20261117a/campaign", "Recovery")
    assert len(original) == 1395 and len(conditions) == 155 and len(recovery) == 6
    data = dict(conditions=conditions, runs=original+recovery, trace_fields=TRACE,
                molecule_fields=CANDIDATE, source_root=str(args.parent))
    with gzip.open(args.output, "xt", encoding="utf-8", compresslevel=5) as stream:
        json.dump(data, stream, separators=(",", ":"), allow_nan=False)
    print(args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
