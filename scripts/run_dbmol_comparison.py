"""Run a prescribed DBMol smoke comparison serially in one DTU allocation.

New/unvalidated until HPC receipts establish otherwise. Each child uses one
canonical graph and its own bounded immutable run; a failure remains in the
comparison table. No native pose generation, repair or candidate screening is
hidden here. This is an engineering gate, not a full optimization experiment.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.stability_study import verified_child
from latentmol.boltz.workflow_process import run_logged
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from latentmol.multimodel.study import verified_file


def run(path, expected_sha, output, selected, timeout):
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require immutable DTU allocation")
    path = verified_file(path, expected_sha)
    comparison = json.loads(path.read_text())
    candidates = {r["id"]: r for r in comparison["cases"]}
    if (not selected or len(set(selected)) != len(selected)
            or any(s not in candidates for s in selected)):
        raise ValueError("Select distinct prescribed cases; no replacement or outcome screening")
    if type(timeout) is not int or not 60 <= timeout <= 21600:
        raise ValueError("Smoke child timeout must be 60..21600 seconds")
    config = {"comparison": str(path), "comparison_sha256": expected_sha,
              "selected_cases": selected, "child_timeout_seconds": timeout,
              "scope": "Native/component derivative and short terminal gates only"}
    with ExperimentRun(output, "dbmol-native-comparison", config, Path.cwd()) as experiment:
        identity = {"source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
                    "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"]}
        rows = [{"id": s, "status": "not_attempted"} for s in selected]
        report = {"cases": rows, "native_smoke_passed": False,
                  "full_trajectory_validation": False, "scope": config["scope"]}
        report_path = experiment.path / "comparison.json"
        write_json(report_path, report)
        for index, row in enumerate(rows):
            declared = candidates[row["id"]]
            row["status"] = "running"
            write_json(report_path, report)
            try:
                cfg_path = verified_file(declared["path"], declared["sha256"])
                cfg = ActiveSetConfig.load(cfg_path)
                if cfg.optimizer.steps != declared["steps"] or not 1 <= cfg.optimizer.steps <= 3:
                    raise ValueError("This smoke controller supports at most three declared updates")
                result = run_logged([sys.executable, "-m", "latentmol.boltz.active_set",
                                     "--config", cfg_path],
                                    experiment.path / f"child-{index:02d}-{row['id']}.log",
                                    timeout_seconds=timeout)
                row["execution"] = result
                child = verified_child(result, cfg.output_root, identity,
                                       config_file_sha256=declared["sha256"])
                summary = json.loads((child / "summary.json").read_text())
                if summary.get("frozen_state_unchanged") is not True:
                    raise ValueError("Child does not certify frozen weights/context")
                arms = summary["arms"]
                if set(arms) != set(cfg.arms) or any(
                        a.get("completed_requested_budget") is not True
                        or a.get("accepted_updates") != cfg.optimizer.steps for a in arms.values()):
                    raise ValueError("Short trajectory did not complete its declared accepted updates")
                row.update(status="passed", child_run=str(child),
                           summary_sha256=file_sha256(child / "summary.json"),
                           arms=arms, oracle_cost=summary["oracle_cost"])
            except Exception as exc:
                row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            write_json(report_path, report)
            experiment.log({"stage": "native_case", "id": row["id"], "status": row["status"]})
        report["native_smoke_passed"] = all(r["status"] == "passed" for r in rows)
        write_json(report_path, report)
        experiment.annotate(native_smoke_passed=report["native_smoke_passed"],
                            full_trajectory_validation=False)
        if not report["native_smoke_passed"]:
            raise RuntimeError("One or more DBMol smoke cases failed; retained in comparison.json")
    return experiment.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--child-timeout-seconds", type=int, default=5400)
    a = p.parse_args()
    print(run(a.comparison, a.sha256, a.output, a.cases, a.child_timeout_seconds))


if __name__ == "__main__":
    main()
