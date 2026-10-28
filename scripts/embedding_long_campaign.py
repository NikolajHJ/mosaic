"""Local stdlib-only seal, upload and dispatch for source142 long embedding run."""
import argparse
import json
from pathlib import Path
import re
import shlex
import time

from embedding_bridge_campaign import ROOT, WORK, SSH, call, save, seal, stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "stage", "submit", "collect"))
    parser.add_argument("--attempt", default="attempt001")
    args = parser.parse_args()
    if not re.fullmatch(r"attempt[0-9]{3}", args.attempt) or args.attempt == "attempt000":
        raise ValueError("Require a new immutable attempt001..attempt999")
    folder = ROOT / "outputs/hpc-submissions" / ("source142-epoch36-guarded-embedding-long-" + args.attempt)
    if args.action == "seal":
        seal(folder, args.attempt, number=142, campaign_name="epoch36-guarded-embedding-long",
             config_relative="configs/epoch36_embedding_long.json",
             operator_relative="scripts/hpc/embedding_long_campaign.py")
        return
    identity = json.loads((folder / "seal.json").read_text())
    if args.action == "stage":
        stage(folder, identity)
        return
    argv = [WORK + "/env/boltz311/bin/python", identity["release"] + "/" + identity["operator_relative"],
            "submit-initial" if args.action == "submit" else "collect",
            "--release", identity["release"], "--archive-sha256", identity["archive_sha256"],
            "--manifest-sha256", identity["manifest_sha256"], "--plan-sha256", identity["plan_sha256"],
            "--campaign", identity["campaign"]]
    stamp = str(time.time_ns())
    reply = call(SSH + ["bash -lc " + shlex.quote(shlex.join(argv))],
                 folder / (args.action + "-" + stamp + ".json"), timeout=180)
    value = json.loads(reply["stdout"])
    if args.action == "collect":
        target = ROOT / "outputs/hpc-results" / identity["label"]
        target.mkdir(parents=True, exist_ok=True)
        save(target / ("status-" + stamp + ".json"), value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
