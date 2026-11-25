"""Hold both first-leg workers until the last epoch-47 callback publishes their common parent."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

from vae_segment_chain import checkpoint, read, submit, write_once


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def job(control, name, args):
    # An intent without a receipt is ambiguous. Never submit twice after a transport failure.
    receipt = control / (name + ".receipt.json")
    if receipt.exists():
        record = read(receipt)
        if record["argv"][-len(args):] != args:
            raise ValueError("Existing receipt has different job arguments")
        return int(record["id"])
    return submit(control, name, args)


def watch(plan, control, predecessor, segment):
    parent_control = Path(plan["parent_plan"]).parent
    receipt = parent_control / (predecessor["job_prefix"] + "-after-sparse-%02d.receipt.json" % segment)
    parent_callback = int(read(receipt)["id"])
    dependency = "done(%d)" % parent_callback
    if segment == 1:
        for retired in plan.get("release_after_retirement", ()):
            dependency += " && ended(%d)" % int(retired)
    args = ["-q", "hpc", "-n", "1", "-R", "rusage[mem=1024]", "-W", "00:15",
            "-w", dependency, "/usr/bin/python3", str(Path(__file__).resolve()),
            "advance", str(control / "plan.json"), "--segment", str(segment)]
    return job(control, plan["prefix"] + "-parent-%02d" % segment, args)


def advance(plan, control, predecessor, segment):
    recipe = predecessor["sparse"]
    directory = Path(predecessor["run_root"]) / ("sparse-%02d" % segment)
    summary = read(directory / "summary.json")
    checkpoint(summary, directory, recipe)
    if summary["status"] == "paused":
        if segment >= recipe["max_segments"]:
            raise RuntimeError("Parent used its last segment without finishing epoch 47")
        return {"next_parent_gate": watch(plan, control, predecessor, segment + 1)}
    latest = summary["latest"]
    if (latest != read(directory / "latest.json") or latest["epoch"] != 47
            or latest["updates"] != plan["parent_updates"] or digest(Path(latest["checkpoint"])) != latest["sha256"]):
        raise RuntimeError("Completed parent receipt or checkpoint identity differs")
    # A requeued arm can share a held sibling's original checkpoint receipt.
    path = Path(plan.get("parent_receipt", control / "parent.json"))
    if not path.exists():
        write_once(path, latest)
    if read(path) != latest:
        raise ValueError("Pair was previously assigned a different checkpoint")
    ids = []
    for arm in plan["arms"]:
        record = read(control / (arm["name"] + ".receipt.json"))
        ident = str(int(record["id"]))
        state = subprocess.check_output(["bjobs", "-a", "-noheader", "-o", "stat", ident], text=True).strip()
        if state == "PSUSP":
            subprocess.run(["bresume", ident], check=True)
        elif state not in {"PEND", "RUN", "DONE"}:
            raise RuntimeError("Unexpected state of paired worker %s: %s" % (ident, state))
        ids.append(ident)
    return {"released": ids, "parent": latest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "advance"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--segment", type=int, default=1)
    args = parser.parse_args()
    control = args.plan.resolve(strict=True).parent
    plan = read(args.plan)
    previous = Path(plan["parent_plan"])
    if digest(previous) != plan["parent_plan_sha256"]:
        raise ValueError("Predecessor plan identity changed")
    predecessor = read(previous)
    if predecessor["sparse"]["config_sha256"] != plan["parent_config_sha256"]:
        raise ValueError("Predecessor recipe identity changed")
    if args.action == "submit":
        ids = [job(control, arm["name"], ["-H", *arm["args"]]) for arm in plan["arms"]]
        print({"workers": ids, "gate": watch(plan, control, predecessor, args.segment)}, flush=True)
    else:
        print(advance(plan, control, predecessor, args.segment), flush=True)


if __name__ == "__main__":
    main()
