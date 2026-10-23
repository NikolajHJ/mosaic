"""Bounded sequential diagnostic replays of explicitly selected failed gates.

The controller holds no model/GPU allocation outside its serial child process.
Each child authenticates its original immutable production code and failed saved
state. Numerical failures are retained per case; no optimizer runs, tolerance
changes, automatic gate approvals or parent-file modifications are permitted.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from .workflow_process import run_logged


def load_protocol(path: Path) -> dict:
    """Require an explicit finite case list and per-child time budget."""
    raw = json.loads(path.read_text())
    fields = {"version", "name", "output_root", "timeout_seconds", "cases"}
    if not isinstance(raw, dict) or set(raw) != fields or type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("Require exactly the version-1 replay protocol fields")
    if not isinstance(raw["name"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", raw["name"]):
        raise ValueError("Invalid diagnostic study name")
    if type(raw["timeout_seconds"]) is not int or not 60 <= raw["timeout_seconds"] <= 3600:
        raise ValueError("Each replay needs a declared 60..3600 second timeout")
    if not isinstance(raw["cases"], list) or not 1 <= len(raw["cases"]) <= 16:
        raise ValueError("Declare between one and sixteen serial replay cases")
    names = set()
    for case in raw["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "parent_run", "production_release", "arm", "step"}:
            raise ValueError("Each replay needs exactly id, parent, production release, arm and step")
        if not isinstance(case["id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", case["id"]) or case["id"] in names:
            raise ValueError("Replay case IDs must be distinct path-safe identifiers")
        names.add(case["id"])
        if case["arm"] not in ("active_set", "fixed_initial_mask") or type(case["step"]) is not int or not 0 <= case["step"] <= 9999:
            raise ValueError("Invalid saved arm or step")
        for field in ("parent_run", "production_release"):
            if not isinstance(case[field], str) or not case[field] or not Path(case[field]).is_absolute():
                raise ValueError("Parent runs and original production releases require absolute paths")
    if not isinstance(raw["output_root"], str) or not Path(raw["output_root"]).is_absolute():
        raise ValueError("Declare an absolute unique-run output root")
    return raw


def run(protocol_path: Path) -> Path:
    """Completion means every requested diagnostic was attempted, not passed."""
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require an HPC allocation and immutable source archive identity")
    protocol = load_protocol(protocol_path)
    record = {**protocol, "protocol_path": str(protocol_path.resolve()),
              "protocol_file_sha256": file_sha256(protocol_path),
              "maximum_child_seconds": len(protocol["cases"]) * protocol["timeout_seconds"],
              "claim_limit": "Replay only; failed original gates remain failed; no molecular optimization."}
    launcher = Path.cwd() / "scripts/diagnose_saved_boltz_gate.py"
    if not launcher.is_file():
        raise ValueError("Run from the new immutable diagnostic release root")
    with ExperimentRun(Path(protocol["output_root"]), protocol["name"], record, Path.cwd()) as experiment:
        results = []
        for case in protocol["cases"]:
            command = [sys.executable, str(launcher), "--parent-run", case["parent_run"],
                       "--production-release", case["production_release"], "--arm", case["arm"],
                       "--step", str(case["step"]), "--run-root", str(experiment.path / case["id"])]
            result = run_logged(command, experiment.path / f"{case['id']}.log",
                                timeout_seconds=protocol["timeout_seconds"])
            completed = False
            if not result["timed_out"] and result["returncode"] == 0 and len(result["run_paths"]) == 1:
                child = Path(result["run_paths"][0])
                completed = (json.loads((child / "run.json").read_text())["status"] == "completed"
                             and json.loads((child / "diagnostic.json").read_text())["status"] == "completed")
            results.append({"case": case, "diagnostic_completed": completed, "execution": result})
            write_json(experiment.path / "cases.json", results)
        experiment.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in results),
                            diagnostic_cases_requested=len(results), original_gates_not_reclassified=True)
        return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.protocol), flush=True)


if __name__ == "__main__":
    main()
