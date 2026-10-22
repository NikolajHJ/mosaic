"""Allocated scheduling-only readiness; preserve the passed optimization engine."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import json
import os
from pathlib import Path

from mosaic.small_molecule.io import file_sha256, write_json
from mosaic.small_molecule.gradient_resume import checked_path


PLAN_SHA = "4837faf4a35e49927c8fe828b65dfb91ee871da9028ade9db27e6bfb9db1ade1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Requires allocated DTU CPU")
    release = _repository_root(__file__)
    baseline = Path(
        "/work3/s234818/mosaic/releases/20261017-brd4-best-primal-04/source/source_manifest.json"
    )
    if (
        file_sha256(baseline)
        != "ec278a14e0847939f667dc2a1965f735da627684b3e7f38e0e59e624473e9873"
    ):
        raise ValueError("Passed source 04 manifest changed")
    old = json.loads(baseline.read_text())["files"]
    current = json.loads((release / "source_manifest.json").read_text())["files"]
    if {k: v for k, v in old.items() if k.startswith("src/")} != {
        k: v for k, v in current.items() if k.startswith("src/")
    }:
        raise ValueError("Numerical library changed while splitting scheduling")
    plan_path = Path(
        "/zhome/95/e/206121/mosaic-runs/20261017-brd4-best-primal-readiness-03/payload/plan/plan.json"
    )
    if file_sha256(plan_path) != PLAN_SHA:
        raise ValueError("Passed continuation plan changed")
    plan = json.loads(plan_path.read_text())
    if plan["status"] != "ready" or plan["readiness_job_id"] != "29745466":
        raise ValueError("Unexpected passed parent plan")
    count = 0
    for study in plan["studies"].values():
        parent = Path(study["parent_directory"])
        if file_sha256(parent / "summary.json") != study["parent_summary_sha256"]:
            raise ValueError("Parent summary changed")
        for entry in study["cases"]:
            if entry is not None:
                for name, digest in entry["files"].items():
                    if file_sha256(checked_path(entry["directory"], name)) != digest:
                        raise ValueError("Pinned parent case changed")
                    count += 1
    for directory, digest in [
        (plan["pilot_directory"], plan["pilot_summary_sha256"]),
        (plan["canonical_directory"], plan["canonical_summary_sha256"]),
    ]:
        if file_sha256(Path(directory) / "summary.json") != digest:
            raise ValueError("Reference summary changed")
    result = {
        "status": "ready",
        "job_id": os.environ["LSB_JOBID"],
        "plan_sha256": PLAN_SHA,
        "numerical_library_unchanged_from_source04": True,
        "verified_parent_case_files": count,
        "scope": "case scheduling only; all old model/loss/update/context identities unchanged",
    }
    write_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
