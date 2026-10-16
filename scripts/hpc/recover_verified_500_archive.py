"""Remove one redundant archive; retain its verified local copy and raw HPC data.

Run for the authorized 1000-update campaign only after the operator rechecks
the local archive hash and its 4619-file download receipt. No other path or
recursive cleanup is supported. All raw HPC studies/source remain intact.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

ARCHIVE = Path(
    "/work3/s234818/mosaic/runs/20261016-gradient-evidence-carbon-chain-500-01.tar.gz"
)
SHA256 = "dc1e1d948c99d487f5c06897964037cd2130a983f003ad4fc1643d996bad14ba"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-local-sha256", required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or args.verified_local_sha256 != SHA256:
        raise ValueError("Allocated job and verified local-copy identity required")
    if (
        ARCHIVE.is_symlink()
        or ARCHIVE.resolve() != ARCHIVE
        or ARCHIVE.stat().st_size != 2020603152
    ):
        raise ValueError("Unexpected archive path/type/size")
    root = ARCHIVE.parent
    for name in (
        "20261016-carbon-chain-500-01",
        "20261016-carbon-chain-500-mix50-01",
        "20261016-carbon-chain-500-mix50-02",
        "20261016-carbon-chain-500-report-01",
        "20261016-carbon-chain-500-report-02",
    ):
        if not (root / name).is_dir():
            raise ValueError("Preserved raw evidence missing: " + name)
    for name, expected in (
        (
            "20261016-carbon-chain-500-01",
            "d232b7f54a725c12e0683d2fac0da7b9f31d20bb1309ffca1159c803142bb89f",
        ),
        (
            "20261016-carbon-chain-500-mix50-02",
            "466eebe4b77e6031d8972ddf77331ea18167cda1677ae62656ffc91860c10de6",
        ),
    ):
        if (
            hashlib.sha256(
                (root / name / "payload/study/summary.json").read_bytes()
            ).hexdigest()
            != expected
        ):
            raise ValueError("Raw study identity changed")
    with ARCHIVE.open("rb") as reader:
        if hashlib.file_digest(reader, "sha256").hexdigest() != SHA256:
            raise ValueError("Remote archive differs from verified local copy")
    receipt = {
        "job_id": os.environ["LSB_JOBID"],
        "path": str(ARCHIVE),
        "sha256": SHA256,
        "bytes": ARCHIVE.stat().st_size,
        "status": "verified_before_removal",
        "scope": "one redundant archive; verified local archive plus all raw HPC results/source preserved",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    ARCHIVE.unlink()
    receipt["status"] = "redundant_archive_removed"
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
