"""Read DTU quota once; decline numerical work when headroom is insufficient.

Run on allocated DTU HPC. No cleanup, downloads or retries occur.
--storage-pool is required and must match the run/cache location's actual pool;
there is no implicit pool default. Historical placement is not a claim about
today's storage availability or another account's permissions.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess


def parse_pool(text, pool):
    block = re.search(
        rf"storagepool\s+{pool}\)\s*:(.*?)(?=\n[^\n]*storagepool|\Z)", text, re.S
    )
    if not block:
        raise ValueError("Requested storage pool not present in DTU quota output")
    for line in block.group(1).splitlines():
        parts = line.split("||")
        if len(parts) != 3 or not re.search(r"\|\s*\d+\s*$", parts[0]):
            continue
        used_text, hard_text = parts[1].split("|")
        file_used, file_hard = parts[2].split("|")
        if "unlimited" in hard_text or "unlimited" in file_hard:
            raise ValueError(
                "Cannot establish finite quota headroom from unlimited pool output"
            )

        def size(value):
            match = re.fullmatch(r"\s*([\d.]+)\s*(Byte|KiB|MiB|GiB|TiB)\s*", value)
            if not match:
                raise ValueError(f"Unrecognized quota size: {value!r}")
            return float(match.group(1)) * (
                1024 ** ("Byte", "KiB", "MiB", "GiB", "TiB").index(match.group(2))
            )

        return {
            "used_bytes": size(used_text),
            "hard_bytes": size(hard_text),
            "used_files": int(file_used),
            "hard_files": int(file_hard),
        }
    raise ValueError("No finite user row found in selected pool")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--storage-pool", type=int, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=4.0)
    parser.add_argument("--minimum-free-files", type=int, default=10000)
    args = parser.parse_args()
    if args.minimum_free_gib <= 0 or args.minimum_free_files <= 0:
        raise ValueError("Headroom reservations must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "not_run",
        "pool": args.storage_pool,
        "run_dir": str(args.run_dir.resolve()),
        "minimum_free_gib": args.minimum_free_gib,
        "minimum_free_files": args.minimum_free_files,
        "limitation": "DTU quota output can lag actual usage; this is conservative preflight, not a reservation",
    }
    try:
        command = subprocess.run(
            ["getquota_work3.sh"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        (args.run_dir / "quota_work3.txt").write_text(command.stdout, encoding="utf-8")
        pool = parse_pool(command.stdout, args.storage_pool)
        receipt["quota"] = pool
        receipt["filesystem_free_bytes"] = shutil.disk_usage(args.run_dir).free
        headroom = min(
            pool["hard_bytes"] - pool["used_bytes"], receipt["filesystem_free_bytes"]
        )
        receipt["available_bytes"] = headroom
        receipt["available_files"] = pool["hard_files"] - pool["used_files"]
        enough = (
            headroom >= args.minimum_free_gib * 1024**3
            and receipt["available_files"] >= args.minimum_free_files
        )
        receipt["status"] = "ready" if enough else "tests_not_run_insufficient_space"
    except Exception as error:
        enough = False
        receipt["status"] = "tests_not_run_quota_unverified"
        receipt["error"] = f"{type(error).__name__}: {error}"
    (args.run_dir / "quota_preflight.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt), flush=True)
    return 0 if enough else 75


if __name__ == "__main__":
    raise SystemExit(main())
