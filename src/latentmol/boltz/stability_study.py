"""Serial, matched stability experiments using frozen Boltz and saved initial z.

UNVALIDATED until the HPC evidence in the ledger establishes otherwise. Each
trajectory has independent failure capture. A terminal stall remains a stall;
native predictions and repair witnesses never replace its canonical molecule.
Allocation continuation skips sealed results and retains interrupted attempts.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import yaml

from latentmol.config_identity import verify_config_identity
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from .workflow_process import require_child, run_logged

POLICIES = ("legacy", "small_fixed", "backtracking")
ARMS = ("fixed_initial_mask", "active_set")
SOURCE062 = "3c2fd0f49632e67a55c8a070a04a079408f94e67d3f40bee6fcdc1a326076eb5"


def read(path):
    return json.loads(Path(path).read_text())


def asset(value):
    """Require an authenticated absolute file reference, never guessed assets."""
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
        raise ValueError("Asset requires absolute path and SHA256")
    return value


def verify_asset(value):
    asset(value)
    path = Path(value["path"])
    if file_sha256(path) != value["sha256"]:
        raise ValueError(f"Asset changed: {path}")
    return path


def verified_child(result, expected_root, identity, *, config_file_sha256=None, parent_run_id=None,
                   confidence_sha256=None):
    """Authenticate returned child provenance before crediting any observations."""
    path = require_child(result, experiment=True)
    if not path.resolve().is_relative_to(Path(expected_root).resolve()):
        raise ValueError("Child output escaped its declared attempt root")
    record = read(path / "run.json")
    verify_config_identity(record)
    if (record.get("source", {}).get("sha256") != identity["source_manifest_sha256"]
            or record.get("source_archive_sha256") != identity["source_archive_sha256"]):
        raise ValueError("Child source differs from this immutable study")
    if config_file_sha256 is not None and record["config"].get("config_file_sha256") != config_file_sha256:
        raise ValueError("Child configuration differs from the prescribed policy")
    if parent_run_id is not None and record["config"].get("parent_run_id") != parent_run_id:
        raise ValueError("Native evaluation belongs to a different trajectory")
    if confidence_sha256 is not None and record["config"].get("confidence_checkpoint_sha256") != confidence_sha256:
        raise ValueError("Native evaluation changes the declared confidence checkpoint")
    return path


def load_protocol(path):
    """Predeclare cohorts, update policies, occupancy factors and compute bounds."""
    p = read(path)
    fields = {"version", "name", "output_root", "cache", "confidence_checkpoint",
              "step_comparison", "starts", "steps", "gradient_audit_interval",
              "optimization_timeout_seconds", "repair_timeout_seconds",
              "rescore_timeout_seconds", "allocation_seconds", "repair_search"}
    if not isinstance(p, dict) or set(p) != fields or type(p["version"]) is not int or p["version"] != 1:
        raise ValueError("Require the exact version-1 stability protocol")
    if not isinstance(p["name"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", p["name"]):
        raise ValueError("Invalid study name")
    for key in ("output_root", "cache"):
        if not isinstance(p[key], str) or not Path(p[key]).is_absolute():
            raise ValueError(f"{key} requires absolute path")
    bounds = {"steps": (1, 100), "gradient_audit_interval": (1, 100),
              "optimization_timeout_seconds": (60, 7200), "repair_timeout_seconds": (60, 900),
              "rescore_timeout_seconds": (60, 7200), "allocation_seconds": (600, 82800)}
    for key, (low, high) in bounds.items():
        if type(p[key]) is not int or not low <= p[key] <= high:
            raise ValueError(f"Invalid {key} budget")
    asset(p["confidence_checkpoint"])
    gate = p["step_comparison"]
    if not isinstance(gate, dict) or set(gate) != {"run", "cases", "reports"}:
        raise ValueError("Declare completed comparison evidence")
    asset(gate["run"]); asset(gate["cases"])
    if not isinstance(gate["reports"], list) or len(gate["reports"]) != 12:
        raise ValueError("Require all twelve saved-state comparison reports")
    for report in gate["reports"]:
        asset(report)
    if not isinstance(p["starts"], list) or not 1 <= len(p["starts"]) <= 16:
        raise ValueError("Declare a bounded starting cohort")
    names = set()
    for start in p["starts"]:
        if not isinstance(start, dict) or set(start) != {"id", "parent_record", "initial_latent", "native_template", "policies", "arms"}:
            raise ValueError("Malformed starting cohort member")
        if (not isinstance(start["id"], str)
                or not re.fullmatch(r"[a-zA-Z0-9_-]+", start["id"]) or start["id"] in names):
            raise ValueError("Starting IDs must be distinct and path safe")
        names.add(start["id"])
        for key in ("parent_record", "initial_latent", "native_template"):
            asset(start[key])
        for key, supported in (("policies", POLICIES), ("arms", ARMS)):
            values = start[key]
            if (not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values)
                    or len(set(values)) != len(values)
                    or any(value not in supported for value in values)):
                raise ValueError(f"Invalid {key} cohort factor")
    from latentmol.probability_repair import ProbabilityRepairConfig
    ProbabilityRepairConfig(**p["repair_search"])
    return p


def verify_comparison(gate):
    """Do not interpret a successful controller exit as twelve successful audits."""
    record = read(verify_asset(gate["run"]))
    verify_config_identity(record)
    rows = read(verify_asset(gate["cases"]))
    if (record.get("status") != "completed" or record.get("source_archive_sha256") != SOURCE062
            or len(rows) != 12 or len({r["case"]["id"] for r in rows}) != 12
            or not all(r.get("comparison_completed") is True for r in rows)):
        raise ValueError("Saved-state step comparisons are not completely verified")
    expected_paths = {str(Path(r["execution"]["run_paths"][0]) / "comparison.json") for r in rows}
    if {r["path"] for r in gate["reports"]} != expected_paths:
        raise ValueError("Comparison report membership changed")
    for a in gate["reports"]:
        d = read(verify_asset(a))
        if (d.get("status") != "completed" or d.get("comparison_completed") is not True
                or d.get("frozen_state_unchanged") is not True
                or d.get("parent_and_audit_unchanged") is not True):
            raise ValueError("A prerequisite comparison lacks unchanged scientific state")


def expand_cases(protocol):
    """No seed screening or replacement: every prescribed factor remains a row."""
    return [{"id": f"{s['id']}-{policy}-{arm}", "start": s["id"], "policy": policy,
             "arm": arm, "status": "not_attempted", "attempts": []}
            for s in protocol["starts"] for policy in s["policies"] for arm in s["arms"]]


def build_config(protocol, start, policy, arm, output_root, *, smoke=False):
    """Copy verified scientific inputs, modifying only declared campaign factors."""
    if policy not in POLICIES or arm not in ARMS:
        raise ValueError("Unsupported stability factor")
    parent = read(verify_asset(start["parent_record"]))
    verify_config_identity(parent, allow_legacy_active_set_contexts=True)
    verify_asset(start["initial_latent"]); verify_asset(start["native_template"])
    original = parent["config"]
    fields = ("mode", "decoder", "affinity_checkpoint", "anchor_count", "device", "temperature",
              "threshold", "recycling_steps", "objective", "allow_lossy_charge", "start", "diagnostics")
    raw = {key: deepcopy(original[key]) for key in fields if key in original}
    raw.setdefault("start", {"kind": "encoded_seed", "seed": None})
    if original.get("context_manifest"):
        raw["context_manifest"] = deepcopy(original["context_manifest"])
    else:
        raw["contexts"] = deepcopy(original["contexts"])
    raw.update(version=1, name=f"stability-{start['id']}-{policy}-{arm}", output_root=str(output_root),
               initial_latent=deepcopy(start["initial_latent"]), arms=[arm], element_diagnostics=True,
               fd_policy="local_v1", gradient_audit_interval=1 if smoke else protocol["gradient_audit_interval"],
               optimizer={"steps": 3 if smoke else protocol["steps"],
                          "learning_rate": 50.0 if policy == "small_fixed" else 400.0,
                          "max_gradient_norm": 0.000125, "trust_radius": 3.0})
    raw["diagnostics"]["hard_samples"] = 1 if smoke else 8
    if policy == "backtracking":
        raw["step_control"] = {"version": "armijo_v1", "max_trials": 16,
                               "armijo": 0.0001, "minimum_improvement_ulps": 2.0}
    return raw


def trajectory_report(pilot, arm):
    """Analyze recorded states inside HPC; separate loss drift from context jumps."""
    from .active_set_rescore import select_endpoints
    selected = select_endpoints(pilot)
    terminal = selected["arms"][arm]["termination"]
    folder = Path(pilot) / arm
    metrics = [read(p) for p in sorted(folder.glob("metrics-*.json"))]
    if [r["step"] for r in metrics] != list(range(terminal["endpoint_step"] + 1)):
        raise ValueError("Trajectory metric states do not match the terminal boundary")
    losses = [r["score"]["loss"] for r in metrics]
    if any(not math.isfinite(float(v)) for v in losses):
        raise ValueError("Trajectory contains nonfinite objective states")
    def fixed_input(row):
        # Same-count slot swaps reuse the count template but change conditioning.
        return row["context_sha256"], tuple(row["selection"]["active_slots"])
    same_context = [(a, b) for a, b in zip(metrics, metrics[1:]) if fixed_input(a) == fixed_input(b)]
    increases = [{"from_step": a["step"], "to_step": b["step"],
                  "loss_increase": b["score"]["loss"] - a["score"]["loss"]}
                 for a, b in same_context if b["score"]["loss"] > a["score"]["loss"]]
    controls = [read(p) for p in sorted(folder.glob("step-control-*.json"))]
    transitions = [read(p) for p in sorted(folder.glob("transition-*.json"))]
    late = metrics[-min(10, len(metrics)):]
    late_same_context = len({fixed_input(r) for r in late}) == 1
    return {"termination": terminal, "states": len(metrics),
            "initial_signed_loss": losses[0], "terminal_signed_loss": losses[-1],
            "net_proxy_improvement": losses[0] - losses[-1],
            "net_proxy_change_may_include_context_jumps": any(r.get("old_context_sha256") for r in transitions),
            "within_context_score_increases": increases,
            "fixed_input_score_jumps": [{"from_step": a["step"], "to_step": b["step"],
                "signed_loss_jump": b["score"]["loss"] - a["score"]["loss"]}
                for a, b in zip(metrics, metrics[1:]) if fixed_input(a) != fixed_input(b)],
            "maximum_latent_displacement": max(r["latent_displacement"] for r in metrics),
            "counts": [r["selection"]["count"] for r in metrics],
            "natural_counts": [r["natural_selection"]["count"] for r in metrics],
            "argmax_valid_states": sum(r["argmax_smiles"] is not None for r in metrics),
            "terminal_smiles": metrics[-1]["argmax_smiles"],
            "late_same_context": late_same_context,
            "late_loss_range": max(r["score"]["loss"] for r in late) - min(r["score"]["loss"] for r in late) if late_same_context else None,
            "step_scales": [r.get("accepted_scale") for r in controls if r.get("accepted")],
            "trace": [{"step": r["step"], "signed_loss": r["score"]["loss"],
                       "gradient_norm": r["score"]["gradient_norm"],
                       "count": r["selection"]["count"], "active_slots": r["selection"]["active_slots"],
                       "context_sha256": r["context_sha256"],
                       "oracle_cost_at_state": r.get("oracle_cost_at_state"),
                       "legacy_update": r.get("legacy_update")} for r in metrics],
            "trial_count": sum(len(r["trials"]) for r in controls),
            "rejected_trials": sum(not t["accepted"] for r in controls for t in r["trials"]),
            "transitions": transitions, "oracle_cost": read(Path(pilot) / "oracle_cost.json"),
            "claim_limit": "Descriptive frozen-pose stability; stalled is not a proven optimum or completed 100-update run."}


def seal_case(row, attempt):
    """Seal final attempt artifacts; a timeout/partial trajectory is still retained."""
    row["sealed_artifacts"] = {str(p.relative_to(attempt)): file_sha256(p)
                               for p in sorted(attempt.rglob('*')) if p.is_file()}


def publish_terminal(row, attempt, root):
    """Independent terminal receipt survives interruption before table publication."""
    seal_case(row, attempt)
    folder = root / "terminal"
    folder.mkdir(exist_ok=True)
    path = folder / f"{row['id']}.json"
    if path.exists():
        raise ValueError("A terminal case receipt already exists")
    write_json(path, row)


def verify_seal(row, root):
    if read(root / "terminal" / f"{row['id']}.json") != row:
        raise ValueError("Case table differs from its independent terminal receipt")
    attempt = Path(row["attempt_dir"])
    if not attempt.resolve().is_relative_to((root / "cases" / row["id"]).resolve()):
        raise ValueError("Sealed attempt escaped case root")
    actual = {str(p.relative_to(attempt)) for p in attempt.rglob('*') if p.is_file()}
    if actual != set(row["sealed_artifacts"]):
        raise ValueError("Sealed attempt file set changed")
    for relative, expected in row["sealed_artifacts"].items():
        if file_sha256(attempt / relative) != expected:
            raise ValueError("Sealed attempt artifact changed")


def study_summary(rows):
    """Execution coverage is distinct from stable updates and native improvements."""
    return {"planned": len(rows), "not_attempted": sum(r["status"] == "not_attempted" for r in rows),
            "terminal_verified": sum("trajectory" in r for r in rows),
            "completed_budget": sum(r.get("trajectory", {}).get("termination", {}).get("completed_budget", False) for r in rows),
            "stalled": sum(r.get("trajectory", {}).get("termination", {}).get("stalled", False) for r in rows),
            "optimization_failed": sum(r["status"] == "optimization_failed" for r in rows),
            "repair_failed": sum(r.get("repair", {}).get("returncode") not in (None, 0)
                                 or r.get("repair", {}).get("timed_out", False)
                                 or "repair_error" in r for r in rows),
            "rescore_failed": sum(r["status"] == "rescore_failed" for r in rows),
            "evaluated": sum(r["status"] == "evaluated" for r in rows),
            "claim_limit": "Requested updates are not equal oracle budgets; no experimental binding or global root-cause claim."}


def run(protocol_path, expected_sha256, *, resume=False):
    """Use one GPU at a time; continuation happens only at case boundaries."""
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require DTU HPC allocation and immutable release")
    protocol_path = Path(protocol_path)
    verify_asset({"path": str(protocol_path.resolve()), "sha256": expected_sha256})
    p = load_protocol(protocol_path)
    verify_comparison(p["step_comparison"])
    verify_asset(p["confidence_checkpoint"])
    root = Path(p["output_root"])
    identity = {"protocol_sha256": expected_sha256,
                "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"],
                "source_manifest_sha256": source_manifest(Path.cwd())["sha256"]}
    if identity["source_manifest_sha256"] != read(Path.cwd() / "release_manifest.json")["sha256"]:
        raise ValueError("Source differs from immutable manifest")
    expected_rows = expand_cases(p)
    if resume:
        if read(root / "identity.json") != identity:
            raise ValueError("Continuation source or protocol changed")
        rows = read(root / "cases.json")
        if [(r["id"], r["start"], r["policy"], r["arm"]) for r in rows] != [
                (r["id"], r["start"], r["policy"], r["arm"]) for r in expected_rows]:
            raise ValueError("Continuation case list changed")
    else:
        root.mkdir(parents=True, exist_ok=False)
        rows = expected_rows
        write_json(root / "identity.json", identity)
    def persist():
        write_json(root / "cases.json", rows)
        write_json(root / "summary.json", study_summary(rows))
    persist()
    starts = {s["id"]: s for s in p["starts"]}
    begin = time.monotonic()
    reserve = sum(p[k] for k in ("optimization_timeout_seconds", "repair_timeout_seconds", "rescore_timeout_seconds")) + 180
    with ExperimentRun(root / "allocations", p["name"], {**p, **identity, "resume": resume}, Path.cwd()) as allocation:
        for row in rows:
            receipt_path = root / "terminal" / f"{row['id']}.json"
            if receipt_path.exists() and not row.get("sealed_artifacts"):
                receipt = read(receipt_path)
                if any(receipt[k] != row[k] for k in ("id", "start", "policy", "arm")):
                    raise ValueError("Interrupted table differs from terminal case identity")
                row.clear(); row.update(receipt)
            if row.get("sealed_artifacts"):
                verify_seal(row, root)
                continue
            if time.monotonic() - begin + reserve >= p["allocation_seconds"]:
                allocation.annotate(stopped_at_case_boundary=True)
                break
            if row["status"] != "not_attempted":
                row["attempts"].append({k: deepcopy(v) for k, v in row.items() if k != "attempts"})
            history = row["attempts"]
            row_keys = {k: row[k] for k in ("id", "start", "policy", "arm")}
            row.clear(); row.update(**row_keys, attempts=history, status="optimizing")
            case_root = root / "cases" / row["id"]
            index = len(history) + 1
            while (case_root / f"attempt-{index:03d}").exists():
                history.append({"status": "orphan_interrupted", "attempt_dir": str(case_root / f"attempt-{index:03d}")})
                index += 1
            attempt = case_root / f"attempt-{index:03d}"
            attempt.mkdir(parents=True)
            row["attempt_dir"] = str(attempt)
            start = starts[row["start"]]
            config_path = attempt / "optimization.yaml"
            config_path.write_text(yaml.safe_dump(build_config(p, start, row["policy"], row["arm"], attempt / "optimization"), sort_keys=False))
            row["config_sha256"] = file_sha256(config_path)
            persist()
            optimization = run_logged([sys.executable, "-m", "latentmol.boltz.active_set", "--config", config_path],
                                      attempt / "optimization.log", timeout_seconds=p["optimization_timeout_seconds"])
            row["optimization"] = optimization
            try:
                pilot = verified_child(optimization, attempt / "optimization", identity,
                                       config_file_sha256=row["config_sha256"])
                trajectory = trajectory_report(pilot, row["arm"])
                # Encoded-chain audits do not carry z hashes; start.pt is common
                # authenticated input, and all runs record their actual z hash.
                import torch
                from latentmol.multimodel.study import tensor_digest
                actual = torch.load(pilot / "start.pt", map_location="cpu", weights_only=True)["z"]
                original = torch.load(verify_asset(start["initial_latent"]), map_location="cpu", weights_only=True)["z"]
                if tensor_digest({"z": actual}) != tensor_digest({"z": original}):
                    raise ValueError("Matched policy changed its prescribed initial z")
                row["initial_z_sha256"] = tensor_digest({"z": actual})
                row["trajectory"] = trajectory
            except Exception as exc:
                row.update(status="optimization_failed", error=str(exc), error_type=type(exc).__name__)
                publish_terminal(row, attempt, root); persist()
                continue
            write_json(attempt / "trajectory_report.json", row["trajectory"])
            row["status"] = "evaluating"
            persist()
            endpoint = row["trajectory"]["termination"]["endpoint_step"]
            repair_protocol = {"version": 1, "name": "stability-repair", "output_root": str(attempt / "repair"),
                "parents": [{"id": row["id"], "pilot": str(pilot), "run_sha256": file_sha256(pilot / "run.json"),
                             "steps": sorted({0, 25, 50, 75, p["steps"], endpoint})}], "search": p["repair_search"]}
            repair_path = attempt / "repair-protocol.json"
            write_json(repair_path, repair_protocol)
            row["repair"] = run_logged([sys.executable, "-m", "latentmol.probability_repair_audit", "--protocol", repair_path,
                                        "--sha256", file_sha256(repair_path)], attempt / "repair.log", timeout_seconds=p["repair_timeout_seconds"])
            try:
                repair_run = verified_child(row["repair"], attempt / "repair", identity)
                if read(repair_run / "run.json")["config"].get("protocol_sha256") != file_sha256(repair_path):
                    raise ValueError("Repair audit belongs to a different checkpoint protocol")
                row["repair_summary"] = read(repair_run / "summary.json")
            except Exception as exc:
                row["repair_error"] = str(exc)
            # Failed/censored repairs never change eligibility of the raw endpoint.
            native = run_logged([sys.executable, "-m", "latentmol.boltz.active_set_rescore", "--pilot", pilot,
                "--template", verify_asset(start["native_template"]), "--cache", p["cache"], "--run-root", attempt / "native"],
                attempt / "native.log", timeout_seconds=p["rescore_timeout_seconds"])
            row["native"] = native
            try:
                native_path = verified_child(native, attempt / "native", identity,
                                             parent_run_id=read(pilot / "run.json")["run_id"],
                                             confidence_sha256=p["confidence_checkpoint"]["sha256"])
                row["native_summary"] = read(native_path / "summary.json")
                row["status"] = "evaluated"
            except Exception as exc:
                row.update(status="rescore_failed", native_error=str(exc))
            publish_terminal(row, attempt, root); persist()
        # A continuation may recover only terminal receipts and run no children.
        # Publish that recovered table even when no new case calls persist().
        persist()
        allocation.annotate(**study_summary(rows), all_cases_processed=all(r.get("sealed_artifacts") for r in rows))
    return root


def smoke(protocol_path, expected_sha256, output_root):
    """Two actual native runs gate the campaign; a stall is allowed but must verify."""
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require HPC allocation")
    verify_asset({"path": str(Path(protocol_path).resolve()), "sha256": expected_sha256})
    p = load_protocol(protocol_path); verify_comparison(p["step_comparison"])
    starts = {s["id"]: s for s in p["starts"]}
    identity = {"source_manifest_sha256": read(Path.cwd() / "release_manifest.json")["sha256"],
                "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"]}
    if source_manifest(Path.cwd())["sha256"] != identity["source_manifest_sha256"]:
        raise ValueError("Source differs from immutable manifest")
    with ExperimentRun(Path(output_root), "stability-native-smoke", {"protocol_sha256": expected_sha256, "updates": 3}, Path.cwd()) as experiment:
        for start_id, arm in (("carbon-c13", "active_set"), ("prior-4", "fixed_initial_mask")):
            config_path = experiment.path / f"{start_id}.yaml"
            config_path.write_text(yaml.safe_dump(build_config(p, starts[start_id], "backtracking", arm,
                experiment.path / start_id, smoke=True), sort_keys=False))
            result = run_logged([sys.executable, "-m", "latentmol.boltz.active_set", "--config", config_path],
                                experiment.path / f"{start_id}.log", timeout_seconds=900)
            pilot = verified_child(result, experiment.path / start_id, identity,
                                   config_file_sha256=file_sha256(config_path))
            report = trajectory_report(pilot, arm)
            full_budget_policy = report["termination"].get("step_control_version") == "best_trial_v2"
            if (report["termination"]["accepted_updates"] < 1
                    or report["within_context_score_increases"] and not full_budget_policy
                    or full_budget_policy and not report["termination"]["completed_budget"]):
                raise ValueError("Native smoke has no accepted update or unexplained within-context score increase")
            write_json(experiment.path / f"{start_id}-report.json", report)
        experiment.annotate(native_smoke_passed=True, full_trajectory_validation=False)
    return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-root", type=Path)
    args = parser.parse_args()
    print(smoke(args.protocol, args.sha256, args.smoke_root) if args.smoke_root
          else run(args.protocol, args.sha256, resume=args.resume), flush=True)


if __name__ == "__main__":
    main()
