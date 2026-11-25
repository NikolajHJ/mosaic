"""Native-only recovery of a completed matched embedding comparison.

No optimization or model-weight loading occurs in this coordinator. Saved
parent gates, selection, target and checkpoint bytes are authenticated first.
Each native CLI runs in a fresh sequential child while this process remains
CUDA-free. This is essential under LSF ``exclusive_process``: allocator cache
release cannot remove the context of a still-running optimization process.

The first original request is also the native GPU smoke. It is counted once;
its failure stops the other requests, while its successful result enters the
unchanged paired report. Earlier failed attempts remain in the parent tree.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import sys
import time
import traceback

from latentmol.cuda_process import require_cuda_free_coordinator


VERSION = "embedding-native-recovery-v1"
TOP_ARTIFACTS = ("result.json", "case-summary.json", "start-gates.json",
                 "fresh-native/selection.json", "fresh-native/target.json")
CASE_ARTIFACTS = ("native-oracle.json", "relaxed-start-gate.json",
                  "learned-start-gate.json", "production-binding.json",
                  "optimization/comparison.json")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _asset(asset):
    if (not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
            or not isinstance(asset["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None):
        raise ValueError("Recovery assets require exact path and SHA256")
    path = Path(asset["path"])
    if not path.is_absolute() or _sha(path) != asset["sha256"]:
        raise ValueError("Recovery asset changed: " + str(path))
    return path


def load_plan(path, expected_sha):
    path = Path(path)
    if _sha(path) != expected_sha:
        raise ValueError("Recovery plan bytes differ")
    plan = json.loads(path.read_text())
    if (set(plan) != {"version", "parent", "internal_seconds"}
            or plan["version"] != VERSION or type(plan["internal_seconds"]) is not int
            or not 1 <= plan["internal_seconds"] <= 79200):
        raise ValueError("Unknown recovery plan or allocation budget")
    parent = plan["parent"]
    if not isinstance(parent, dict) or set(parent) != {"root", "plan", "artifacts"}:
        raise ValueError("Recovery parent declaration differs")
    return plan


def authenticate_parent(plan):
    """Reconstruct the same five roles per case without reading model tensors."""
    require_cuda_free_coordinator()
    from .embedding_comparison_experiment import native_members
    from .embedding_comparison_rescore import freeze_selection, load_target
    parent = plan["parent"]
    root = Path(parent["root"])
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("Recovery requires the exact existing parent GPU root")
    config_path = _asset(parent["plan"])
    config = json.loads(config_path.read_text())
    artifacts = parent["artifacts"]
    required = set(TOP_ARTIFACTS) | {
        f"case-{i:03d}/{name}" for i in range(1, 4) for name in CASE_ARTIFACTS}
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ValueError("Recovery must authenticate every original case and gate")
    observed = {}
    for name, expected in artifacts.items():
        relative = PurePosixPath(name)
        path = root / name
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root):
            raise ValueError("Parent artifact escaped its exact root")
        _asset({"path": str(path), "sha256": expected})
        observed[name] = json.loads(path.read_text())
    result = observed["result.json"]
    summary = observed["case-summary.json"]["cases"]
    gates = observed["start-gates.json"]["cases"]
    if (result.get("status") != "completed_matched_pilot"
            or result.get("frozen_vae_unchanged") is not True
            or result.get("plan_sha256") != parent["plan"]["sha256"]
            or result.get("cases") != summary or result.get("start_gates") != gates
            or len(summary) != 3 or len(gates) != 3):
        raise ValueError("Parent comparison lacks completed frozen optimization evidence")
    members = []
    for index, case in enumerate(summary, 1):
        case_id = f"case-{index:03d}"
        if case.get("case_id") != case_id or gates[index - 1].get("case_id") != case_id:
            raise ValueError("Parent case order changed")
        if observed[f"{case_id}/native-oracle.json"].get("status") != "passed_native_oracle_only":
            raise ValueError("Parent native oracle did not pass")
        for arm in ("relaxed", "learned"):
            gate = observed[f"{case_id}/{arm}-start-gate.json"]
            if gate.get("passed") is not True or gate.get("status") != "passed_pointwise_start_only":
                raise ValueError("Parent start derivative gate did not pass")
            if observed[f"{case_id}/production-binding.json"].get(arm, {}).get("passed") is not True:
                raise ValueError("Parent production binding did not pass")
        comparison = observed[f"{case_id}/optimization/comparison.json"]
        expected_record = root / f"{case_id}/optimization/comparison.json"
        if (Path(case["full_record"]).resolve() != expected_record
                or comparison.get("status") != "completed_comparison_not_binding_validated"
                or [(run.get("path"), run.get("status"), run.get("applied")) for run in comparison["runs"]]
                   != [(arm, "completed", 50) for arm in ("relaxed", "learned")]):
            raise ValueError("Parent trajectories do not match the prescribed completed budget")
        members.extend(native_members(case_id, comparison))
    selection = freeze_selection(members,
        pose_seeds=tuple(config["fresh_native"]["pose_seeds"]),
        native_settings=config["fresh_native"]["native_settings"])
    if selection != observed["fresh-native/selection.json"]:
        raise ValueError("Recovery changed the frozen candidate/request selection")
    if len(selection["requests"]) != 6 or result.get("native_followup", {}).get("successful_native_requests") != 0:
        raise ValueError("This recovery is scoped to all six previously failed native requests")
    checkpoint_hashes = {}
    for member in selection["members"]:
        path = Path(member["checkpoint_path"])
        if not path.resolve().is_relative_to(root):
            raise ValueError("Selected latent checkpoint escaped the parent")
        _asset({"path": str(path), "sha256": member["checkpoint_sha256"]})
        checkpoint_hashes[str(path)] = member["checkpoint_sha256"]
    target = load_target(_asset(config["native_template"]),
        config["fresh_native"]["target_sequence_sha256"], config["fresh_native"]["msa_sha256"],
        template_sha256=config["native_template"]["sha256"])
    if target != observed["fresh-native/target.json"]:
        raise ValueError("Recovery changed the original native target")
    # Native weights are pinned even though this CPU coordinator never loads them.
    for asset in (config["affinity"], config["dbmol"]["confidence_checkpoint"]):
        _asset(asset)
    fresh = config["fresh_native"]
    required_seconds = len(selection["requests"]) * (
        fresh["native_timeout_seconds"] + fresh["audit_timeout_seconds"]) + 120
    if plan["internal_seconds"] < required_seconds:
        raise ValueError("Recovery allocation cannot fit the unchanged native/audit budgets")
    require_cuda_free_coordinator()
    return config, summary, selection, target, checkpoint_hashes


def cuda_child_smoke(output, deadline):
    """A tiny isolated CUDA child must exit before ordinary native prediction."""
    from .workflow_process import run_logged, require_child
    require_cuda_free_coordinator()
    if deadline - time.monotonic() <= 150:
        raise RuntimeError("CUDA child smoke lacks bounded runtime headroom")
    target = Path(output) / "cuda-child-probe.json"
    program = (
        "import json,torch; from pathlib import Path; "
        "x=torch.ones(1,device='cuda'); torch.cuda.synchronize(); "
        f"Path({str(target)!r}).write_text(json.dumps(dict(status='passed',value=float(x.item()),"
        "cuda_initialized=torch.cuda.is_initialized(),device=torch.cuda.get_device_name(0))))"
    )
    process = run_logged([sys.executable, "-c", program], Path(output) / "cuda-child-probe.log",
                         timeout_seconds=120)
    require_child(process)
    value = json.loads(target.read_text())
    if value.get("status") != "passed" or value.get("value") != 1. or value.get("cuda_initialized") is not True:
        raise RuntimeError("Allocated CUDA child did not establish usable GPU execution")
    coordinator = require_cuda_free_coordinator()
    _write(Path(output) / "cuda-ownership-gate.json", {
        "status": "passed", "child": value, "child_process": process,
        "coordinator_after_child": coordinator,
        "scope": "Child allocated CUDA and exited; this coordinator never owned the exclusive context."})


def assert_parent_unchanged(plan, config, checkpoints):
    """A recovery report may not silently incorporate edited parent evidence."""
    root = Path(plan["parent"]["root"])
    _asset(plan["parent"]["plan"])
    for name, expected in plan["parent"]["artifacts"].items():
        _asset({"path": str(root / name), "sha256": expected})
    for name, expected in checkpoints.items():
        _asset({"path": name, "sha256": expected})
    for asset in (config["affinity"], config["dbmol"]["confidence_checkpoint"], config["native_template"]):
        _asset(asset)
    require_cuda_free_coordinator()


def run(plan_path, plan_sha256, output, *, mode="run"):
    if (not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256")
            or mode not in ("validate", "run")):
        raise RuntimeError("Native recovery requires a sealed LSF allocation and declared mode")
    ownership = require_cuda_free_coordinator()
    plan = load_plan(plan_path, plan_sha256)
    output = Path(output)
    if not output.is_absolute() or output.exists() or output.resolve().is_relative_to(Path(plan["parent"]["root"])):
        raise ValueError("Recovery output must be a fresh directory outside the completed parent")
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + plan["internal_seconds"]
    provenance = dict(version=VERSION, plan_sha256=plan_sha256,
        source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"], job_id=os.environ["LSB_JOBID"],
        host=socket.gethostname(), parent=plan["parent"], coordinator=ownership,
        optimization_rerun=False, existing_vae_continuation="untouched")
    _write(output / "started.json", {**provenance, "status": "running_not_passed"})
    try:
        config, cases, selection, target, checkpoints = authenticate_parent(plan)
        _write(output / "parent-authentication.json", dict(status="passed_saved_parent_authentication",
            parent_artifacts=plan["parent"]["artifacts"], selected_checkpoint_hashes=checkpoints,
            selection_sha256=selection["selection_sha256"], native_requests=len(selection["requests"])))
        if mode == "validate":
            result = dict(status="passed_authentication_not_prediction", native_requests=len(selection["requests"]))
        else:
            cuda_child_smoke(output, deadline)
            from .embedding_comparison_rescore import run_followup
            report = run_followup(selection, target, Path(config["boltz_cache"]), output / "fresh-native",
                deadline=deadline, native_timeout_seconds=config["fresh_native"]["native_timeout_seconds"],
                audit_timeout_seconds=config["fresh_native"]["audit_timeout_seconds"],
                require_first_native_success=True)
            from .embedding_comparison_report import generate_report
            html = generate_report(output, cases, report)
            passed = report["all_requests_terminal"] and report["successful_native_requests"] == len(selection["requests"])
            result = dict(status="passed_native_recovery" if passed else "native_recovery_incomplete_not_passed",
                native_followup=report, html_report=str(html), coordinator=require_cuda_free_coordinator())
        assert_parent_unchanged(plan, config, checkpoints)
        _write(output / "result.json", {**provenance, **result,
            "elapsed_seconds": time.monotonic() - started, "finished_at": datetime.now(timezone.utc).isoformat()})
        return result
    except BaseException as error:
        _write(output / "result.json", {**provenance, "status": "failed_not_passed",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "run"), default="run")
    args = parser.parse_args()
    result = run(args.plan, args.plan_sha256, args.output, mode=args.mode)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    if result["status"] not in ("passed_authentication_not_prediction", "passed_native_recovery"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
