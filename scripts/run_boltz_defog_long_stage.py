"""Serial allocation boundaries for the unchanged 500-update matched pair.

Control, guided optimization and checkpoint evaluation use distinct allocations.
An arm is never resumed or restarted: both begin from the original saved z.
Only an authenticated completed budget or verified Armijo stall permits the
next allocation. Native checks and final evaluation membership are unchanged.
All execution and numerical checks belong on allocated DTU HPC.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import run_boltz_defog_long_comparison as comparison


VERSION = "long-comparison-allocation-stage-v1"
STAGES = ("control", "guided", "evaluation")
STAGE_BUDGETS = {
    "control": {"allocation_seconds": 64800, "internal_wall_seconds": 61200},
    "guided": {"allocation_seconds": 86400, "internal_wall_seconds": 84600},
    "evaluation": {"allocation_seconds": 21600, "internal_wall_seconds": 19800},
}
_CASE = {"control": "boltz_only", "guided": "boltz_defog"}
_FOLLOWUP_FIELDS = {"followup_execution", "followups", "followup_completed"}


def _equal(left, right):
    from latentmol.config_identity import canonical_config_json
    return canonical_config_json(left) == canonical_config_json(right)


def stage_root(plan, stage):
    if stage not in STAGES:
        raise ValueError("Unknown allocation stage")
    return Path(plan["output_root"]) / "staged" / stage


def _budget(stage):
    """The split changes allocation boundaries, not the original total budget."""
    if (any(type(value) is not int for limits in STAGE_BUDGETS.values() for value in limits.values())
            or any(not 0 < limits["internal_wall_seconds"] < limits["allocation_seconds"] <= 86400
                   for limits in STAGE_BUDGETS.values())
            or sum(v["allocation_seconds"] for v in STAGE_BUDGETS.values()) != comparison.BUDGETS["allocation_seconds"]
            or sum(v["internal_wall_seconds"] for v in STAGE_BUDGETS.values()) != comparison.BUDGETS["internal_wall_seconds"]):
        raise ValueError("Staged allocations changed the declared total budget")
    return deepcopy(STAGE_BUDGETS[stage])


def _ref(path):
    return comparison._ref(Path(path))


def _read(path):
    return comparison._read(path)


def _write(path, value):
    comparison._write(path, value)


def _verified(ref):
    from latentmol.boltz.stability_study import verify_asset
    return verify_asset(ref)


def _tree_paths(root):
    """Seal every saved child byte, including target/start tensors and reports."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Stage output root is missing or a symlink")
    result = []
    resolved = root.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(resolved):
            raise ValueError("Stage output contains a symlink or escaped artifact")
        if path.is_file():
            result.append(path)
    if not result or len(result) > 100000:
        raise ValueError("Empty or unbounded stage artifact inventory")
    return result


def _seal(roots, *, extra_files=()):
    paths = {}
    for root in roots:
        for path in _tree_paths(root):
            paths[str(path.resolve())] = path
    for path in extra_files:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Stage attempt claim is missing or a symlink")
        paths[str(path.resolve())] = path
    return [_ref(paths[key]) for key in sorted(paths)]


def _stage_outputs(stage, parent, rows, plan, identity):
    """Include evaluation subruns written outside the lightweight stage parent."""
    roots = [parent]
    if stage in _CASE:
        roots.extend(Path(row["child_run"]) for row in rows if row["id"] == _CASE[stage])
        return roots
    from latentmol.config_identity import verify_config_identity
    for row in rows:
        followups = row["followups"]
        child_record = _read(Path(row["child_run"]) / "run.json")
        for label, run in (("repair-audit", followups["repair_audit"]["run"]),
                           ("fresh-pose", followups["fresh_pose_run"])):
            folder = Path(run)
            expected_root = Path(plan["output_root"]) / row["id"] / label
            if not folder.resolve().is_relative_to(expected_root.resolve()):
                raise ValueError("Checkpoint subrun escaped its declared allocation namespace")
            record = _read(folder / "run.json")
            verify_config_identity(record)
            if (record.get("status") != "completed"
                    or record.get("source", {}).get("sha256") != identity["source_manifest_sha256"]
                    or record.get("source_archive_sha256") != identity["source_archive_sha256"]):
                raise ValueError("Checkpoint subrun changes its source or lacks completed provenance")
            if label == "repair-audit":
                if _verified(followups["repair_audit"]["summary"]).resolve() != (folder / "summary.json").resolve():
                    raise ValueError("Checkpoint repair summary belongs to another subrun")
                expected_parent = [{"id": row["id"], "pilot": row["child_run"],
                    "run_sha256": _ref(Path(row["child_run"]) / "run.json")["sha256"],
                    "steps": list(comparison.CHECKPOINTS)}]
                if (not _equal(record.get("config", {}).get("parents"), expected_parent)
                        or not _equal(record["config"].get("search"), plan["repair_search"])
                        or record["config"].get("output_root") != str(expected_root)):
                    raise ValueError("Checkpoint repair audit changes its optimized parent/search/membership")
            elif (record.get("config", {}).get("parent_run_id") != child_record["run_id"]
                    or record["config"].get("case_id") != row["id"]
                    or record["config"].get("plan_sha256") != followups["plan_sha256"]
                    or record["config"].get("parent_run_sha256") != _ref(Path(row["child_run"]) / "run.json")["sha256"]):
                raise ValueError("Fresh-pose results belong to a different optimized arm")
            roots.append(folder)
    return roots


def _scientific_inputs(configs):
    """Rehash declared native files; no model or tensor is loaded here."""
    assets, sources = {}, {}

    def visit(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                path = _verified(value)
                assets[str(path.resolve())] = {"path": str(path.resolve()), "sha256": value["sha256"]}
            elif set(value) == {"path", "commit"}:
                source = Path(value["path"]).resolve(strict=True)
                commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                    text=True, timeout=30).strip()
                dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"],
                    text=True, timeout=30).strip()
                if commit != value["commit"] or dirty:
                    raise ValueError("Native source checkout differs from its frozen declaration")
                sources[str(source)] = {"path": str(source), "commit": commit, "clean": True}
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for cfg in configs.values():
        visit(cfg.record())
    if not assets:
        raise ValueError("Missing frozen scientific input bindings")
    return {"assets": [assets[key] for key in sorted(assets)],
            "native_sources": [sources[key] for key in sorted(sources)]}


def _revalidate_arm(case, row, cfg, identity, parent):
    """Reuse the original full native guard on authenticated saved arm evidence.

    This reader does not reload z. Its saved tensor and the initial-z equality
    proof are sealed after the allocation's actual ``verify_start`` comparison.
    The next stage rehashes both that tensor and the original declared input.
    """
    from latentmol.boltz.stability_study import verified_child
    from latentmol.boltz.trajectory_terminal import read_arm_terminal
    from latentmol.repair_guidance.matched_pilot import validate_prior_run

    if row.get("id") != case["id"] or row.get("status") not in ("passed", "stalled"):
        raise ValueError("Previous arm lacks an accepted native terminal")
    child = verified_child(row["execution"], cfg.output_root, identity,
        config_file_sha256=case["config"]["sha256"])
    if str(child) != row.get("child_run"):
        raise ValueError("Saved arm differs from its actual native process result")
    record = _read(child / "run.json")
    validate_prior_run(cfg.record(), record, base_config=case["config"],
        prior_run_path=str(child / "run.json"), child_run=str(child))
    summary_path, trajectory_path = _verified(row["summary"]), _verified(row["trajectory"])
    if (summary_path.resolve() != (child / "summary.json").resolve()
            or trajectory_path.resolve() != (Path(parent) / (case["id"] + "-trajectory.json")).resolve()):
        raise ValueError("Arm summary/trajectory does not belong to its declared stage")
    summary, trajectory = _read(summary_path), _read(trajectory_path)
    arms = summary.get("arms")
    if (summary.get("frozen_state_unchanged") is not True
            or not isinstance(arms, dict) or set(arms) != {"fixed_initial_mask"}):
        raise ValueError("Arm lacks its unchanged frozen-state/fixed-occupancy proof")
    folder = child / "fixed_initial_mask"
    if not _equal(_read(folder / "summary.json"), arms["fixed_initial_mask"]):
        raise ValueError("Inner native arm summary differs from the completed child summary")
    terminal = read_arm_terminal(folder, cfg.record(), arms["fixed_initial_mask"])
    expected_status = "stalled" if terminal["stalled"] else "passed"
    explicit = {"optimization_status": terminal["kind"], "requested_steps": terminal["requested_updates"],
        "accepted_updates": terminal["accepted_updates"], "completed_requested_budget": terminal["completed_budget"]}
    costs = [value.get("oracle_cost") for value in (row, summary, trajectory)]
    if (row["status"] != expected_status or type(row.get("accepted_updates")) is not int
            or row["accepted_updates"] != terminal["accepted_updates"]
            or not _equal({key: trajectory.get(key) for key in explicit}, explicit)
            or any(not isinstance(value, dict) or not value for value in costs)
            or any(not _equal(value, costs[0]) for value in costs[1:])
            or not isinstance(row.get("initial_z_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["initial_z_sha256"])):
        raise ValueError("Controller/trajectory/native terminal identities differ")
    budget = _read(child / "execution-budget.json")
    if not _equal(summary.get("execution_budget"), budget) or not _equal(row.get("execution_budget"), budget):
        raise ValueError("Native execution-budget receipts differ")
    gates = _read(folder / "gates.json")
    reports = {step: _read(folder / f"step-control-{step:04d}.json")
               for step in range(terminal["attempted_updates"])}
    view = {**row, "status": "stalled" if terminal["stalled"] else "completed"}
    outcome = comparison.validate_case(view, trajectory, gates, cfg, reports,
        budget_receipt=budget, terminal=terminal)
    if outcome.get("passed") is not True or not _equal(row.get("outcome"), outcome):
        raise ValueError("Original native arm guards do not pass unchanged")
    return child


def _matched(rows):
    return (len(rows) == 2 and all(row.get("initial_z_sha256") for row in rows)
            and len({row["initial_z_sha256"] for row in rows}) == 1)


def _state_report(stage, rows, *, evaluation_complete=False):
    matched = _matched(rows)
    healthy = matched and all(row.get("outcome", {}).get("healthy_terminal_passed") is True for row in rows)
    full = healthy and all(row.get("outcome", {}).get("full_500_passed") is True for row in rows)
    return {"stage": stage, "status": "accepted_terminal" if stage != "evaluation" else
                "completed" if full and evaluation_complete else "stalled_observed" if healthy and evaluation_complete else "incomplete_evaluation",
        "cases": deepcopy(rows), "matched_initial_z": matched, "engineering_passed": full,
        "full_500_passed": full, "healthy_terminal_passed": healthy,
        "evaluation_complete": evaluation_complete, "checkpoints": list(comparison.CHECKPOINTS),
        "allocation_split": deepcopy(STAGE_BUDGETS),
        "claim_limit": "Allocation boundary only; no optimizer resume, initialization replacement, native-gate waiver or convergence claim. One matched starting point; scientific advantage is separate."}


def _previous_stage(stage):
    return None if stage == "control" else STAGES[STAGES.index(stage) - 1]


def _receipt_context(stage, ref, plan_ref, plan, configs, identity):
    """Verify the bounded predecessor chain, every byte, and every native gate."""
    path = _verified(ref)
    if path.resolve() != (stage_root(plan, stage) / "receipt.json").resolve():
        raise ValueError("Receipt path differs from its fixed allocation stage")
    receipt = _read(path)
    fields = {"version", "stage", "status", "plan", "source", "budget", "previous", "experiment", "report",
              "scientific_inputs", "artifacts", "job_id", "elapsed_seconds"}
    if (not isinstance(receipt, dict) or set(receipt) != fields or receipt.get("version") != VERSION
            or receipt.get("stage") != stage or receipt.get("status") != "passed"
            or not _equal(receipt.get("plan"), plan_ref) or not _equal(receipt.get("source"), identity)
            or not _equal(receipt.get("budget"), _budget(stage))
            or type(receipt.get("elapsed_seconds")) not in (int, float)
            or not 0 <= receipt["elapsed_seconds"] < _budget(stage)["internal_wall_seconds"]
            or not isinstance(receipt.get("job_id"), str) or not re.fullmatch(r"[1-9][0-9]*", receipt["job_id"])):
        raise ValueError("Stage receipt lacks the exact successful source/plan/budget binding")
    previous = _previous_stage(stage)
    if previous is None:
        if receipt["previous"] is not None:
            raise ValueError("Control cannot inherit a different allocation")
        earlier = {"rows": [], "parents": {}, "job_ids": set()}
    else:
        earlier = _receipt_context(previous, receipt["previous"], plan_ref, plan, configs, identity)
        if receipt["job_id"] in earlier["job_ids"]:
            raise ValueError("Stages must use distinct serial allocations")
    experiment_path = _verified(receipt["experiment"])
    parent = experiment_path.parent
    if (experiment_path.name != "run.json" or not parent.resolve().is_relative_to((stage_root(plan, stage) / "work").resolve())
            or experiment_path.is_symlink()):
        raise ValueError("Stage experiment escaped its allocated output namespace")
    from latentmol.config_identity import verify_config_identity
    record = _read(experiment_path)
    verify_config_identity(record)
    expected_config = {"version": VERSION, "stage": stage, "plan": plan_ref, "previous": receipt["previous"],
                       "budget": _budget(stage), **identity}
    if (record.get("status") != "completed" or not _equal(record.get("config"), expected_config)
            or record.get("source", {}).get("sha256") != identity["source_manifest_sha256"]
            or record.get("source_archive_sha256") != identity["source_archive_sha256"]
            or record.get("environment", {}).get("job_id") != receipt["job_id"]):
        raise ValueError("Stage ExperimentRun did not complete with the declared allocation identity")
    report_path = _verified(receipt["report"])
    if report_path.resolve() != (parent / "stage-report.json").resolve():
        raise ValueError("Stage result belongs to a different experiment")
    result = _read(report_path)
    rows = result.get("cases", [])
    expected_ids = list(comparison.CASE_IDS[:1] if stage == "control" else comparison.CASE_IDS)
    if not isinstance(rows, list) or [row.get("id") for row in rows] != expected_ids:
        raise ValueError("Stage changed arm membership or order")
    inherited = {row["id"]: row for row in earlier["rows"]}
    parents = dict(earlier["parents"])
    if stage in _CASE:
        parents[_CASE[stage]] = parent
    for row in rows:
        case_id = row["id"]
        if case_id in inherited:
            actual, expected = row, inherited[case_id]
            if stage == "evaluation":
                actual = {key: value for key, value in row.items() if key not in _FOLLOWUP_FIELDS}
                expected = {key: value for key, value in expected.items() if key not in _FOLLOWUP_FIELDS}
            if not _equal(actual, expected):
                raise ValueError("A later stage rewrote an earlier accepted optimization result")
        case = next(case for case in plan["cases"] if case["id"] == case_id)
        child = _revalidate_arm(case, row, configs[case_id], identity, parents[case_id])
        if stage == "evaluation":
            if (row.get("followup_completed") is not True or not comparison.followup_completed(
                    row.get("followup_execution", {}), row.get("followups", {}), case_id=case_id, child=child,
                    plan_sha256=plan_ref["sha256"], parent_sha256=_ref(child / "run.json")["sha256"])):
                raise ValueError("Checkpoint evaluation lacks its original complete membership/process proof")
    expected_report = _state_report(stage, rows, evaluation_complete=stage == "evaluation")
    if any(not _equal(result.get(key), value) for key, value in expected_report.items()):
        raise ValueError("Stage result overstates completion or changes its declared scope")
    if stage != "control" and not _matched(rows):
        raise ValueError("Matched arms do not retain the same original initial z")
    if not _equal(receipt["scientific_inputs"], _scientific_inputs(configs)):
        raise ValueError("Frozen model/context/start inputs changed between allocations")
    if not _equal(receipt["artifacts"], _seal(_stage_outputs(stage, parent, rows, plan, identity),
            extra_files=[stage_root(plan, stage) / "attempt.json"])):
        raise ValueError("Saved stage/arm artifact membership or bytes changed")
    return {"receipt": receipt, "rows": rows, "parents": parents, "job_ids": earlier["job_ids"] | {receipt["job_id"]}}


def validate_receipt(stage, ref, plan_path, expected_sha):
    plan, configs, identity = comparison.load_plan(plan_path, expected_sha)
    return _receipt_context(stage, ref, _ref(plan_path), plan, configs, identity)


def _require_remaining(deadline, seconds, label):
    if deadline - time.monotonic() < seconds:
        raise TimeoutError("Cannot reserve complete allocated work: " + label)


def _optimize_arm(stage, plan, configs, identity, parent, deadline, *, row):
    from latentmol.repair_guidance.random_study import _optimize
    from latentmol.repair_guidance.matched_pilot import verify_start
    from latentmol.boltz.trajectory_terminal import read_arm_terminal
    from latentmol.boltz.workflow_process import run_logged

    case_id = _CASE[stage]
    cfg = configs[case_id]
    timeout = plan["optimization_timeout_seconds"][case_id]
    _require_remaining(deadline, timeout + 600, case_id)
    probe = parent / "gpu-process.json"
    execution = run_logged([sys.executable, "-m", "latentmol.gpu_process_probe", "--worker-python",
        configs["boltz_defog"].repair_guidance.provider["python"], "--output", str(probe)], parent / "gpu-process.log", timeout_seconds=180)
    _write(parent / "gpu-process-execution.json", execution)
    if (execution.get("returncode") != 0 or execution.get("timed_out") is not False
            or execution.get("failure") is not None or execution.get("interrupted") is not False
            or _read(probe).get("status") != "passed"):
        raise RuntimeError("Native Boltz/DeFoG process coexistence check failed")
    _require_remaining(deadline, timeout + 300, case_id)
    case = next(case for case in plan["cases"] if case["id"] == case_id)
    row.update(_optimize(case, parent, case_id, identity, timeout))
    child = Path(row["child_run"])
    row["initial_z_sha256"] = verify_start(child, cfg.initial_latent)
    summary, trajectory = _read(row["summary"]["path"]), _read(row["trajectory"]["path"])
    terminal = read_arm_terminal(child / "fixed_initial_mask", cfg.record(), summary["arms"]["fixed_initial_mask"])
    reports = {step: _read(child / "fixed_initial_mask" / f"step-control-{step:04d}.json")
               for step in range(terminal["attempted_updates"])}
    row["execution_budget"] = _read(child / "execution-budget.json")
    row["outcome"] = comparison.validate_case(row, trajectory, _read(child / "fixed_initial_mask/gates.json"), cfg, reports,
        budget_receipt=row["execution_budget"], terminal=terminal)
    if row["outcome"].get("passed") is not True:
        raise RuntimeError("Optimization failed its unchanged native guards")
    row["status"] = "stalled" if terminal["stalled"] else "passed"
    _revalidate_arm(case, row, cfg, identity, parent)
    return row


def _preserve_partial(stage, row, configs, identity, parent, plan):
    """Capture an aborted arm without certifying its endpoint or replacing it."""
    execution_path = parent / (_CASE[stage] + "-execution.json")
    if not execution_path.is_file():
        return
    row["execution"] = _read(execution_path)
    if row.get("child_run") is None:
        cfg = configs[_CASE[stage]]
        case = next(case for case in plan["cases"] if case["id"] == _CASE[stage])
        try:
            row["child_run"] = str(comparison._discover_partial(row["execution"], cfg, identity, case["config"]["sha256"]))
        except Exception as error:
            row["partial_discovery_error"] = str(error)


def _evaluate(rows, plan, plan_path, expected_sha, parent, deadline):
    from latentmol.boltz.workflow_process import run_logged
    timeout = plan["followup_timeout_seconds"]
    _require_remaining(deadline, len(rows) * timeout + 600, "both checkpoint evaluations")
    for row in rows:
        case_id, child = row["id"], Path(row["child_run"])
        _require_remaining(deadline, timeout + 300, case_id + " checkpoints")
        output = parent / (case_id + "-followups.json")
        command = [sys.executable, str(Path(__file__).with_name("evaluate_boltz_defog_long_checkpoints.py")),
            "--plan", str(plan_path), "--sha256", expected_sha, "--case", case_id, "--child", str(child), "--output", str(output)]
        execution = run_logged(command, parent / (case_id + "-followups.log"), timeout_seconds=timeout)
        row["followup_execution"] = execution
        row["followups"] = _read(output) if output.is_file() else {"status": "failed", "error": "Checkpoint worker produced no final report"}
        row["followup_completed"] = comparison.followup_completed(execution, row["followups"], case_id=case_id,
            child=child, plan_sha256=expected_sha, parent_sha256=_ref(child / "run.json")["sha256"])
        _write(parent / "stage-report.json", _state_report("evaluation", rows))
        if not row["followup_completed"]:
            raise RuntimeError("Checkpoint evaluation failed; no next evaluation or replacement")


def run_stage(stage, plan_path, expected_sha, *, previous_ref=None):
    """Execute one uniquely claimed allocation; success is sealed after exit."""
    if stage not in STAGES:
        raise ValueError("Unknown allocation stage")
    from latentmol.experiments import ExperimentRun
    from latentmol.repair_guidance.study import require_hpc
    require_hpc()
    job_id = os.environ.get("LSB_JOBID", "")
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        raise ValueError("Require an allocated DTU job identity")
    started = time.monotonic()
    budget = _budget(stage)
    deadline = started + budget["internal_wall_seconds"]
    plan, configs, identity = comparison.load_plan(plan_path, expected_sha)
    plan_ref = _ref(plan_path)
    predecessor = _previous_stage(stage)
    if predecessor is None:
        if previous_ref is not None:
            raise ValueError("Control must start without inherited optimization")
        earlier = {"rows": [], "parents": {}, "job_ids": set()}
    else:
        if previous_ref is None:
            raise ValueError("Missing authenticated completed predecessor allocation")
        earlier = _receipt_context(predecessor, previous_ref, plan_ref, plan, configs, identity)
        if job_id in earlier["job_ids"]:
            raise ValueError("The successor requires a distinct serial allocation")
    inputs = _scientific_inputs(configs)
    root = stage_root(plan, stage)
    root.mkdir(parents=True, exist_ok=False)
    claim = {"version": VERSION, "stage": stage, "job_id": job_id, "plan": plan_ref,
             "previous": previous_ref, "source": identity, "claimed_at": datetime.now(timezone.utc).isoformat()}
    with (root / "attempt.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    record_config = {"version": VERSION, "stage": stage, "plan": plan_ref, "previous": previous_ref,
                     "budget": budget, **identity}
    rows = deepcopy(earlier["rows"])
    experiment = None
    try:
        with ExperimentRun(root / "work", "long-500-stage-" + stage, record_config, Path.cwd()) as experiment:
            result = _state_report(stage, rows)
            result["status"] = "running"
            _write(experiment.path / "stage-report.json", result)
            try:
                if stage in _CASE:
                    current = {"id": _CASE[stage], "status": "running"}
                    rows.append(current)
                    _optimize_arm(stage, plan, configs, identity, experiment.path, deadline, row=current)
                    if stage == "guided" and not _matched(rows):
                        raise ValueError("Independent arms changed the prescribed original saved z")
                else:
                    _evaluate(rows, plan, plan_path, expected_sha, experiment.path, deadline)
                if not _equal(inputs, _scientific_inputs(configs)):
                    raise ValueError("Frozen scientific inputs changed during the allocation")
                result = _state_report(stage, rows, evaluation_complete=stage == "evaluation")
                if stage == "evaluation" and not result["healthy_terminal_passed"]:
                    raise RuntimeError("Final comparison lacks its matched healthy-terminal proof")
            except BaseException as error:
                if stage in _CASE and rows and rows[-1]["id"] == _CASE[stage]:
                    try:
                        _preserve_partial(stage, rows[-1], configs, identity, experiment.path, plan)
                    except Exception as capture_error:
                        rows[-1]["partial_capture_error"] = str(capture_error)
                    rows[-1].update(status="failed", error=str(error), error_type=type(error).__name__)
                result = _state_report(stage, rows)
                result.update(status="failed", error=str(error), error_type=type(error).__name__)
                raise
            finally:
                _write(experiment.path / "stage-report.json", result)
                children = [(row["id"], Path(row["child_run"])) for row in rows if row.get("child_run")]
                result["interactive_reports"] = comparison._reports(children, experiment.path / "interactive-reports",
                    require_matched=bool(result["engineering_passed"]))
                _write(experiment.path / "stage-report.json", result)
                experiment.annotate(stage_status=result["status"], full_500_passed=result["full_500_passed"],
                    healthy_terminal_passed=result["healthy_terminal_passed"], evaluation_complete=result["evaluation_complete"])
        artifacts = _seal(_stage_outputs(stage, experiment.path, rows, plan, identity), extra_files=[root / "attempt.json"])
        elapsed = time.monotonic() - started
        if elapsed >= budget["internal_wall_seconds"]:
            raise TimeoutError("Stage exceeded its declared internal allocation budget")
        receipt = {"version": VERSION, "stage": stage, "status": "passed", "plan": plan_ref, "source": identity,
            "budget": budget, "previous": previous_ref, "experiment": _ref(experiment.path / "run.json"),
            "report": _ref(experiment.path / "stage-report.json"), "scientific_inputs": inputs,
            "artifacts": artifacts, "job_id": job_id, "elapsed_seconds": elapsed}
        _write(root / "receipt.json", receipt)
        # A byte seal alone never approves an arm. Reuse the same reader the
        # successor callback will invoke, before returning a successful stage.
        _receipt_context(stage, _ref(root / "receipt.json"), plan_ref, plan, configs, identity)
        elapsed = time.monotonic() - started
        if elapsed >= budget["internal_wall_seconds"]:
            raise TimeoutError("Stage proof validation exceeded its internal allocation budget")
        # Include final byte sealing and saved-proof validation in the recorded
        # wall budget; this field does not change any sealed scientific file.
        receipt["elapsed_seconds"] = elapsed
        _write(root / "receipt.json", receipt)
    except BaseException as error:
        _write(root / "failure.json", {**claim, "status": "failed", "error": str(error),
            "error_type": type(error).__name__, "elapsed_seconds": time.monotonic() - started,
            "experiment": str(experiment.path) if experiment is not None else None})
        if (root / "receipt.json").exists():
            failed = _read(root / "receipt.json")
            failed.update(status="failed")
            _write(root / "receipt.json", failed)
        raise
    return root / "receipt.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate-receipt"):
        p = commands.add_parser(name)
        p.add_argument("--stage", choices=STAGES, required=True)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--sha256", required=True)
        if name == "run":
            p.add_argument("--previous-receipt", type=Path)
            p.add_argument("--previous-sha256")
        else:
            p.add_argument("--receipt", type=Path, required=True)
            p.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        if (args.previous_receipt is None) != (args.previous_sha256 is None):
            parser.error("A predecessor requires both receipt path and exact SHA256")
        previous = None if args.previous_receipt is None else {"path": str(args.previous_receipt.resolve()), "sha256": args.previous_sha256}
        print(run_stage(args.stage, args.plan, args.sha256, previous_ref=previous), flush=True)
    else:
        context = validate_receipt(args.stage, {"path": str(args.receipt.resolve()), "sha256": args.receipt_sha256}, args.plan, args.sha256)
        print(json.dumps({"status": "passed", "stage": args.stage, "case_ids": [row["id"] for row in context["rows"]],
            "scope": "Authenticated saved native proof; scheduler completion must be checked separately"}), flush=True)


if __name__ == "__main__":
    main()
