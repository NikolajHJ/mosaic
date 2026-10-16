"""Conservatively gate this campaign's small/medium artifacts on DTU home.

Large model caches stay on work3. Home quota reports may be four hours old;
round used GiB upward, capacity downward and retain the caller's reserve.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


def parse_home(text):
    match = re.search(r"using\s+([\d.]+)G\s+of\s+([\d.]+)G\s+quota", text, re.I)
    if not match:
        raise ValueError("Unrecognized finite DTU home quota")
    used, hard = map(float, match.groups())
    if not all(math.isfinite(x) for x in (used, hard)) or not 0 <= used < hard:
        raise ValueError("Invalid home quota sizes")
    return {
        "used_bytes_upper": math.ceil(used) * 1024**3,
        "hard_bytes_lower": math.floor(hard) * 1024**3,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 4:
        raise ValueError("At least 4 GiB reserve is required")
    if not args.run_dir.resolve().is_relative_to(Path.home() / "mosaic-runs"):
        raise ValueError(
            "Only this user's dedicated home Mosaic artifacts are supported"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "not_run",
        "run_dir": str(args.run_dir.resolve()),
        "minimum_free_gib": args.minimum_free_gib,
        "limitation": "home quota can lag four hours; GiB rounded conservatively; no file-count quota is reported; this is not a reservation",
    }
    try:
        command = subprocess.run(
            ["getquota_zhome.sh"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        (args.run_dir / "quota_zhome.txt").write_text(command.stdout)
        quota = parse_home(command.stdout)
        available = min(
            quota["hard_bytes_lower"] - quota["used_bytes_upper"],
            shutil.disk_usage(args.run_dir).free,
        )
        receipt.update(quota=quota, available_bytes=available)
        receipt["status"] = (
            "ready"
            if available >= args.minimum_free_gib * 1024**3
            else "tests_not_run_insufficient_space"
        )
    except Exception as error:
        receipt.update(status="tests_not_run_quota_unverified", error=str(error))
    (args.run_dir / "quota_preflight.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))
    if receipt["status"] != "ready":
        raise SystemExit(75)


if __name__ == "__main__":
    main()
