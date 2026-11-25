"""Clone the archived projected DeFoG arms for two matched teacher controls."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from stage3_campaign import utc, write_json


def make_plan(archive, output):
    original = json.loads(archive.read_text())
    live = [r for r in original["runs"] if r["condition_id"] == "defog-w0p5"]
    if {r["seed"] for r in live} != {17, 23, 42} or len(live) != 3:
        raise ValueError("Expected the three archived live-teacher references")
    campaign_id = output.parent.name
    configs, bundles = [], []
    for r in live:
        prior = r["condition"]["teacher"]
        if (r["initialization"], r["condition"]["projection"]["mode"], prior["weight"]["value"],
                prior["time"]["value"], prior["refresh"]) != ("fragmented", "connected", 0.5, 0.8, "current"):
            raise ValueError("Live reference has unexpected settings")
        new_runs = []
        for suffix, change in (("frozen-initial", {"refresh": "initial"}),
                               ("shuffled", {"shuffle_within_argmax": True})):
            config = deepcopy(r)
            condition_id = f"defog-w0p5-{suffix}"
            run_id = f"{condition_id}__fragmented__seed-{r['seed']:05d}"
            config.update(run_id=run_id, condition_id=condition_id, campaign_id=campaign_id,
                          paired_with_run_id=r["run_id"], family="teacher-mechanism")
            config["condition"].update(id=condition_id, family="teacher-mechanism",
                                       question="Does the learned target need current graph and row information?")
            config["condition"]["teacher"].update(change)
            configs.append(config)
            new_runs.append(config)
        bundles.append({"bundle_id": f"mechanism-seed-{r['seed']:05d}", "seed": r["seed"],
                        "family": "teacher-mechanism", "wave": 1, "initialization": "fragmented",
                        "run_ids": [c["run_id"] for c in new_runs],
                        "condition_ids": [c["condition_id"] for c in new_runs],
                        "reference_run_ids": [r["run_id"]]})
    plan = dict(original, campaign_id=campaign_id, created_utc=utc(), wave="teacher-mechanism",
                runs=configs, bundles=bundles, baseline_campaign=original["campaign_id"],
                conditions=[c["condition"] for c in configs[:2]])
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    (output / "bundles").mkdir()
    write_json(output / "plan.json", plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_plan(args.archive, args.output)
