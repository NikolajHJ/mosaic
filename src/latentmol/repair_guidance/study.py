"""Matched, serial engineering comparisons for frozen hard-repair guidance.

All numerical/native execution stays inside a declared DTU allocation. This
controller does not rank repaired targets by binding, replace failed cases,
train models, or certify native affinity. A completed Boltz-only fallback is
explicitly insufficient to validate the DeFoG guidance branch.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.stability_study import verified_child
from latentmol.boltz.workflow_process import run_logged
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from latentmol.multimodel.study import verified_file
from .config import RepairGuidanceConfig


def require_hpc():
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Repair experiments require an immutable DTU allocation")


def prepare(base_path, base_sha, guidance_path, guidance_sha, output, *, weights=(0., .01), steps=6):
    """Extend one authenticated fixed-mask Boltz config, changing only lambda.

    Reuse exact saved z and the already frozen pocket/pose. No pocket is derived
    again, no seed is selected on outcomes, and failed draws are not replaced.
    Guidance YAML is a RepairGuidanceConfig mapping, not an extra model graph.
    """
    require_hpc()
    base_path = verified_file(base_path, base_sha)
    guidance_path = verified_file(guidance_path, guidance_sha)
    base = ActiveSetConfig.load(base_path)
    if base.arms != ("fixed_initial_mask",) or base.fd_policy != "local_v1" or base.step_control is None:
        raise ValueError("Base must be a fixed-mask bounded Boltz configuration")
    if base.initial_latent is None or base.repair_guidance is not None:
        raise ValueError("Base needs a saved exact z and no existing repair branch")
    guidance = RepairGuidanceConfig.from_mapping(yaml.safe_load(guidance_path.read_text()), guidance_path.parent)
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("Require one to 10000 declared updates")
    weights = tuple(weights)
    if not weights or len(set(weights)) != len(weights) or 0. not in weights:
        raise ValueError("Require distinct prescribed weights including a zero-weight control")
    configs = []
    # Resolve and validate every setting before creating a partial preparation.
    for index, weight in enumerate(weights):
        adjusted = RepairGuidanceConfig.from_mapping({**guidance.record(), "weight": weight}, output)
        raw = base.record()
        raw["version"] = 1
        raw.pop("context_identity", None)  # Header is reauthenticated by parser.
        if raw["context_manifest"] is None:
            raw.pop("context_manifest")
        else:
            raw.pop("contexts")
        raw["name"] = f"repair-weight-{index:02d}"
        raw["output_root"] = str((output / "runs" / raw["name"]).resolve())
        raw["optimizer"]["steps"] = steps
        raw["repair_guidance"] = adjusted.record()
        configs.append(raw)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for raw in configs:
        path = output / (raw["name"] + ".yaml")
        with path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle, sort_keys=False)
        parsed = ActiveSetConfig.load(path)
        rows.append({"id": raw["name"], "weight": parsed.repair_guidance.weight,
                     "path": str(path.resolve()), "sha256": file_sha256(path), "steps": steps})
    comparison = {"version": 1, "cases": rows, "base": {"path": str(base_path), "sha256": base_sha},
                  "guidance": {"path": str(guidance_path), "sha256": guidance_sha},
                  "source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
                  "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"],
                  "claim_limit": "Prepared matched configs; no native or numerical result",
                  "matching": "same saved z, decoder, Boltz loss/pocket/pose, fixed slots, step rule and quota; lambda alone changes"}
    write_json(output / "comparison.json", comparison)
    return output / "comparison.json"


def guidance_evidence(child, summary, *, weight):
    """Separate native repair, accepted CE updates and derivative-check scope.

    V4 certifies pointwise native AD rather than target-only numerical finite
    differences. Retain the legacy FD counter without relabeling AD receipts.
    These counts describe saved outcomes; the full gate reader authenticates
    their scientific contents separately.
    """
    arms = summary["arms"]
    if set(arms) != {"fixed_initial_mask"}:
        raise ValueError("Unexpected guidance arm outputs")
    arm = arms["fixed_initial_mask"]
    repair = arm.get("repair_guidance", {})
    result = {"weight": weight, "accepted_targets": repair.get("accepted_targets", 0),
              "guided_accepted_updates": repair.get("guided_accepted_updates", 0),
              "target_only_fd_passes": 0, "target_only_pointwise_ad_passes": 0,
              "target_only_derivative_passes": 0, "nonzero_guided_updates": 0,
              "native_attempts": 0, "accepted_native_targets": 0,
              "unsupported_candidates": 0, "chemical_rejections": 0,
              "refreshes": repair.get("refreshes", 0), "repair_validation_passed": False}
    folder = child / "fixed_initial_mask"
    for path in sorted(folder.glob("gate-*.json")):
        gate = json.loads(path.read_text())
        target = gate.get("repair_only_validation", {})
        contract = gate.get("derivative_contract", {}).get("version")
        statuses = ("validated_nonzero", "validated_low_signal") if contract == "pointwise_native_ad_v5" else ("validated_nonzero",)
        passed = target.get("status") in statuses and target.get("passed") is True
        if passed and contract in ("pointwise_native_ad_v4", "pointwise_native_ad_v5"):
            result["target_only_pointwise_ad_passes"] += 1
            result["target_only_derivative_passes"] += 1
        elif passed and contract in (None, "branch_local_v2", "additive_branch_v3"):
            result["target_only_fd_passes"] += 1
            result["target_only_derivative_passes"] += 1
    for path in sorted(folder.glob("metrics-*.json")):
        row = json.loads(path.read_text())
        term = row["score"].get("repair_guidance", {})
        norm = term.get("gradient_diagnostics", {}).get("weighted_repair_norm", 0.)
        result["nonzero_guided_updates"] += int(row.get("update_status") == "accepted" and norm > 1e-12)
    for path in sorted(folder.glob("repair-refresh-*/refresh.json")):
        receipt = json.loads(path.read_text())
        for attempt in receipt.get("attempts", []):
            worker = attempt.get("response", {})
            cost = worker.get("cost", {})
            result["native_attempts"] += int(cost.get("native_forward_calls", 0) > 0)
            result["accepted_native_targets"] += int(attempt.get("status") == "accepted"
                and worker.get("status") == "accepted" and cost.get("native_forward_calls", 0) > 0)
            result["unsupported_candidates"] += int(worker.get("reason") == "input_representation_unsupported")
            result["chemical_rejections"] += int(worker.get("reason") == "output_chemical_invalid")
    # A zero-weight control has no repair calls by design. It validates the
    # baseline, never the additional provider/guidance path.
    if weight > 0:
        result["repair_validation_passed"] = all(result[key] > 0 for key in
            ("accepted_native_targets", "guided_accepted_updates", "target_only_derivative_passes", "nonzero_guided_updates"))
    return result


def run(comparison_path, expected_sha, output, *, timeout=10800):
    """Execute all prescribed cases serially; retain partial/failed evidence."""
    require_hpc()
    comparison_path = verified_file(comparison_path, expected_sha)
    comparison = json.loads(comparison_path.read_text())
    cases = comparison["cases"]
    if (comparison.get("version") != 1 or not cases
            or len({c["id"] for c in cases}) != len(cases)):
        raise ValueError("Invalid prescribed repair comparison")
    if type(timeout) is not int or not 60 <= timeout <= 43200:
        raise ValueError("Child timeout must be 60..43200 seconds")
    config = {"comparison": str(comparison_path), "comparison_sha256": expected_sha,
              "cases": [c["id"] for c in cases], "timeout_seconds": timeout,
              "scope": "Frozen-mask alternating target surrogate; native usefulness unvalidated"}
    with ExperimentRun(output, "repair-native-comparison", config, Path.cwd()) as experiment:
        identity = {"source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
                    "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"]}
        rows = [{"id": c["id"], "status": "not_attempted"} for c in cases]
        report = {"cases": rows, "engineering_completed": False, "native_guidance_passed": False,
                  "fresh_binding_validation": "required_not_run"}
        path = experiment.path / "comparison.json"
        write_json(path, report)
        for index, (declared, row) in enumerate(zip(cases, rows)):
            row["status"] = "running"
            write_json(path, report)
            try:
                cfg_path = verified_file(declared["path"], declared["sha256"])
                cfg = ActiveSetConfig.load(cfg_path)
                if (cfg.repair_guidance is None or cfg.optimizer.steps != declared["steps"]
                        or cfg.repair_guidance.weight != declared["weight"]):
                    raise ValueError("Declared repair case differs from authenticated config")
                execution = run_logged([sys.executable, "-m", "latentmol.boltz.active_set", "--config", cfg_path],
                    experiment.path / f"child-{index:02d}.log", timeout_seconds=timeout)
                row["execution"] = execution
                child = verified_child(execution, cfg.output_root, identity, config_file_sha256=declared["sha256"])
                summary = json.loads((child / "summary.json").read_text())
                if summary.get("frozen_state_unchanged") is not True:
                    raise ValueError("No certified frozen weights/context")
                evidence = guidance_evidence(child, summary, weight=cfg.repair_guidance.weight)
                arm = summary["arms"]["fixed_initial_mask"]
                row.update(status="completed" if arm.get("completed_requested_budget") else "stalled",
                    child_run=str(child), summary_sha256=file_sha256(child / "summary.json"),
                    repair_evidence=evidence, oracle_cost=summary["oracle_cost"])
            except Exception as exc:
                row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            write_json(path, report)
        report["engineering_completed"] = all(r["status"] == "completed" for r in rows)
        positive = [r for c, r in zip(cases, rows) if c["weight"] > 0]
        report["native_guidance_passed"] = (report["engineering_completed"] and bool(positive)
            and all(r.get("repair_evidence", {}).get("repair_validation_passed") for r in positive))
        write_json(path, report)
        experiment.annotate(engineering_completed=report["engineering_completed"],
                            native_guidance_passed=report["native_guidance_passed"])
        if not report["native_guidance_passed"]:
            raise RuntimeError("Comparison lacks actual accepted native guidance evidence; inspect retained cases")
    return experiment.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    a = commands.add_parser("prepare")
    a.add_argument("--base-config", type=Path, required=True)
    a.add_argument("--base-sha256", required=True)
    a.add_argument("--guidance", type=Path, required=True)
    a.add_argument("--guidance-sha256", required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--steps", type=int, default=6)
    a.add_argument("--weights", type=float, nargs="+", default=[0., .01])
    b = commands.add_parser("run")
    b.add_argument("--comparison", type=Path, required=True)
    b.add_argument("--sha256", required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--child-timeout-seconds", type=int, default=10800)
    args = p.parse_args()
    if args.command == "prepare":
        print(prepare(args.base_config, args.base_sha256, args.guidance, args.guidance_sha256,
                      args.output, weights=args.weights, steps=args.steps))
    else:
        print(run(args.comparison, args.sha256, args.output, timeout=args.child_timeout_seconds))


if __name__ == "__main__":
    main()
