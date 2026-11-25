"""Read saved control/candidate panels after the short control allocation (stdlib only)."""

import argparse
import json
from pathlib import Path


def optional_json(file):
    try:
        return json.loads(file.read_bytes())
    except FileNotFoundError:
        return None


def read_arm(directory):
    fork = optional_json(directory / "fork.json")
    if fork is None:
        return None, {}, {"state": "awaiting fork receipt"}
    progress = optional_json(directory / "progress.json") or {}
    latest = optional_json(directory / "latest.json") or {}
    summary = optional_json(directory / "summary.json") or {}
    start = fork["resume"]["updates"]
    panels = {}
    file = directory / "quality-panel.jsonl"
    if file.is_file():
        with file.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):  # Writer is still publishing this line.
                    break
                row = json.loads(line)
                u = row["updates"]
                if (u < start or u in panels
                        or row["generated_roundtrip"]["leg_start_update"] != start):
                    raise ValueError(f"Invalid panel cursor in {file}")
                panels[u] = row
    if panels and min(panels) != start:
        raise ValueError(f"Start panel missing in {file}")
    if summary and (summary["latest"] != latest or summary["updates"] != latest["updates"]):
        raise ValueError(f"Inconsistent terminal checkpoint receipt in {directory}")
    observed = [u for u in (progress.get("updates"), latest.get("updates"), max(panels, default=None),
                            summary.get("updates")) if u is not None]
    return fork, panels, dict(state=summary.get("stop_reason", "awaiting terminal summary"),
                              update=max(observed, default=start), checkpoint=latest.get("updates"),
                              last_panel=max(panels, default=None))


def panel_key(row):
    fixed = row["generated_roundtrip"]["fixed_at_leg_start"]
    return (row["seed"], row["dataset_sha256"], row["validation_rows"], row["validation_points"],
            row["prior_points"], fixed["prior_attempts"], fixed["eligible_parents"])


def fractions(row):
    fixed = row["generated_roundtrip"]["fixed_at_leg_start"]
    return [[fixed["mean_exact"], fixed["eligible_parents"]],
            [row["reconstruction_exact"], row["validation_points"]], [row["prior_valid"], row["prior_points"]]]


def build(control, candidate):
    c, left, cs = read_arm(control)
    v, right, vs = read_arm(candidate)
    if c and v and (c["resume"] != v["resume"] or c["parent"] != v["parent"]
                    or v["source_config_sha256"] != c["config_sha256"]
                    or Path(v["production_control"]).resolve() != control.resolve()):
        raise ValueError("Cannot pair arms with different parent, resume receipt or control recipe")
    start = (c or v)["resume"]["updates"] if c or v else None
    reference = left.get(start) or right.get(start)
    if reference and any(panel_key(r) != panel_key(reference) for arm in (left, right) for r in arm.values()):
        raise ValueError("Panel seed, held-out rows, dataset or fixed denominators differ")
    matched = [dict(update=u, control=fractions(left[u]), candidate=fractions(right[u]))
               for u in sorted(left.keys() & right.keys())]
    return dict(fork_update=start, arms={"control": cs, "candidate": vs},
                metrics=["generated_exact", "heldout_exact", "prior_valid"], matched=matched,
                latest_panels={k: fractions(p[max(p)]) if p else None
                               for k, p in (("control", left), ("candidate", right))})


def markdown(result):
    lines = ["# Eight-hour control comparison", ""]
    for arm, s in result["arms"].items():
        latest = result["latest_panels"][arm]
        description = f"{arm.capitalize()} ({s['state']})."
        if s.get("update") is not None:
            description += f" Latest recorded update {s['update']:,}."
        if latest is not None:
            description += (f" Panel {s['last_panel']:,} gave {latest[0][0]}/{latest[0][1]} exact generated parents, "
                            f"{latest[1][0]}/{latest[1][1]} exact held-out reconstructions and "
                            f"{latest[2][0]}/{latest[2][1]} valid random argmax molecules.")
        lines.extend((description, ""))
    if result["matched"]:
        lines.extend(("Each cell shows control followed by candidate on the same update and fixed cohort.", "",
                      "| Update | Generated exact | Held-out exact | Valid random argmax |",
                      "|---:|---:|---:|---:|"))
        for row in result["matched"]:
            pairs = [f"{a}/{n} vs {b}/{m}" for (a,n), (b,m) in zip(row["control"], row["candidate"])]
            lines.append("| " + " | ".join([f"{row['update']:,}"] + pairs) + " |")
    if len(result["matched"]) < 2:
        lines.extend(("", "Waiting for a shared post-fork evaluation."))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    control, candidate, output = (p.resolve() for p in (args.control, args.candidate, args.output))
    if (control == candidate or any(output == p or p in output.parents or output in p.parents
                                    for p in (control, candidate))):
        raise ValueError("Use separate training and report directories")
    result = build(control, candidate)
    output.mkdir(parents=True, exist_ok=False)
    (output / "interim.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    report = markdown(result)
    (output / "interim.md").write_text(report, encoding="utf8")
    print(report)


if __name__ == "__main__":
    main()
