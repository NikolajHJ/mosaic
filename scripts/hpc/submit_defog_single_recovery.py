"""Prepare one prior-0 runtime recovery; submission requires real new validation.

The scientific configuration is copied from the authenticated source079 case;
only its name and output directory change. This is a fresh 100-update run from
the original saved latent, never a fabricated resume. All FD/refresh settings,
neutral-only provider eligibility and fallback behavior remain unchanged.
Computational completion requires 100 accepted updates and complete fresh-pose
execution for every eligible raw molecule, with no prediction/identity failures.
Comparison eligibility is separate: original prior-0 starts with a sanitization
failure, so requiring valid start/end pairs would guarantee failure even after a
successful trajectory. Invalid raw states remain unscorable and are never
repaired or replaced. Computational completion makes no binding/validity claim.

Preparation/plan printing cannot launch jobs. --submit and allocated execution
require a hash-pinned readiness bundle containing actual new-release native-gate
and saved-failure gate records, not a manual passed flag or completed diagnostic.
Source081 attention attribution alone is explicitly insufficient readiness.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time

import yaml

from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from latentmol.config_identity import verify_config_identity
from submit_stability_study import authenticated, read, submit
from submit_ramped_random import gpu_request


OPTIMIZATION_SECONDS = 21600
RESCORE_SECONDS = 900
ALLOCATION_SECONDS = 27000  # Internal bound leaves 30 min in the 8 h LSF job.
FAILURE_CASES = ("prior-0", "prior-3", "prior-5")
COMPONENTS = ("dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss")


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(Path(path))}


def asset(ref):
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute()
            or not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
        raise ValueError("Require an absolute path and explicit SHA256")
    path = Path(ref["path"])
    authenticated(path, ref["sha256"])
    return path


def scientific_config(raw):
    """Only destination/name are non-scientific; every other field is retained."""
    return {key: value for key, value in raw.items() if key not in ("name", "output_root")}


def clone_case(raw, output):
    if (raw.get("optimizer", {}).get("steps") != 100 or raw.get("arms") != ["fixed_initial_mask"]
            or raw.get("start", {}).get("kind") != "standard_normal" or raw["start"].get("seed") != 0
            or not raw.get("initial_latent") or raw.get("fd_policy") != "local_v1"
            or raw.get("gradient_audit_interval") != 50 or raw.get("dbmol") is None
            or raw.get("step_control") is None or raw.get("repair_guidance", {}).get("refresh_interval") != 10
            or raw["repair_guidance"].get("attempts_per_refresh") != 2
            or raw["repair_guidance"].get("weight_schedule") is None):
        raise ValueError("Require the original prior-0 fixed-mask 100-step ramped configuration")
    result = deepcopy(raw)
    result.update(name="single-recovery-prior-0", output_root=str(output))
    if scientific_config(result) != scientific_config(raw):
        raise ValueError("Recovery changed scientific configuration")
    return result


def original_plans(parent_ref, failures_ref):
    """Bind the failure cohort to the exact original production protocol."""
    parent, failures = read(asset(parent_ref)), read(asset(failures_ref))
    if (parent.get("version") != 1 or failures.get("version") != 1
            or [c["id"] for c in parent.get("cases", [])] != [f"prior-{i}" for i in range(10)]
            or [c["id"] for c in failures.get("cases", [])] != list(FAILURE_CASES)
            or failures.get("production_archive_sha256") != parent.get("source_archive_sha256")
            or failures.get("production_manifest_sha256") != parent.get("source_manifest_sha256")):
        raise ValueError("Authenticate original full cohort and three saved failed states")
    cases = {case["id"]: case for case in parent["cases"]}
    for failed in failures["cases"]:
        config = cases[failed["id"]]["config"]
        if (failed["arm"] != "fixed_initial_mask"
                or failed["parent_artifacts_sha256"].get(config["path"]) != config["sha256"]):
            raise ValueError("Failed-state plan refers to another production case")
    return parent, failures


def prepare(parent_ref, failures_ref, output, campaign_root, *, archive_sha, manifest_sha):
    """Metadata-only preparation; does not claim readiness or run a model."""
    parent, _ = original_plans(parent_ref, failures_ref)
    case = deepcopy(parent["cases"][0])
    original_config = case["config"]
    raw = yaml.safe_load(asset(original_config).read_text())
    output, campaign_root = Path(output).resolve(), Path(campaign_root).resolve()
    revised = clone_case(raw, campaign_root / "prior-0" / "optimization")
    for key in ("parent_record", "initial_latent", "native_template"):
        asset(case[key])
    if revised["initial_latent"] != case["initial_latent"]:
        raise ValueError("Original saved latent identity differs from the cohort")
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "prior-0.yaml"
    config_path.write_text(yaml.safe_dump(revised, sort_keys=False))
    case["config"] = reference(config_path)
    plan = {"version": 1, "name": "defog-single-recovery", "parent_protocol": parent_ref,
            "failed_state_plan": failures_ref, "original_config": original_config,
            "case": case, "output_root": str(campaign_root), "cache": parent["cache"],
            "optimization_timeout_seconds": OPTIMIZATION_SECONDS, "rescore_timeout_seconds": RESCORE_SECONDS,
            "allocation_seconds": ALLOCATION_SECONDS, "source_archive_sha256": archive_sha,
            "source_manifest_sha256": manifest_sha, "resume": False,
            "scope": "One fresh prior-0 100-update case with all original audits; unsubmitted and blocked until actual new-release validation."}
    path = output / "plan.json"
    write_json(path, plan)
    return path


def load_plan(path, sha):
    p = read(asset({"path": str(Path(path).resolve()), "sha256": sha}))
    if (p.get("version") != 1 or p.get("resume") is not False or p.get("case", {}).get("id") != "prior-0"
            or p.get("optimization_timeout_seconds") != OPTIMIZATION_SECONDS
            or p.get("rescore_timeout_seconds") != RESCORE_SECONDS or p.get("allocation_seconds") != ALLOCATION_SECONDS):
        raise ValueError("Require one fresh case and explicit 6 h/900 s budgets")
    parent, _ = original_plans(p["parent_protocol"], p["failed_state_plan"])
    original = yaml.safe_load(asset(p["original_config"]).read_text())
    revised = yaml.safe_load(asset(p["case"]["config"]).read_text())
    expected = clone_case(original, Path(p["output_root"]) / "prior-0" / "optimization")
    if revised != expected or p["case"] != {**parent["cases"][0], "config": p["case"]["config"]}:
        raise ValueError("Single-case configuration, start or native template drifted")
    if p["original_config"] != parent["cases"][0]["config"] or p["cache"] != parent["cache"]:
        raise ValueError("Original scientific case identity drifted")
    for key in ("parent_record", "initial_latent", "native_template"):
        asset(p["case"][key])
    return p


def _completed_run(ref, plan):
    path = asset(ref)
    record = read(path)
    verify_config_identity(record)
    if (path.name != "run.json" or record.get("status") != "completed"
            or record.get("source", {}).get("sha256") != plan["source_manifest_sha256"]
            or record.get("source_archive_sha256") != plan["source_archive_sha256"]
            or not record.get("environment", {}).get("job_id")):
        raise ValueError("Readiness requires a completed HPC run of this exact release")
    return path.parent, record


def _completed_fd(item, expected_policy):
    if (item.get("passed") is not True or item.get("policy") != expected_policy
            or set(item.get("direction_results", {})) != {"gradient", "random_1", "random_2", "random_3"}
            or not all(value.get("passed") is True for value in item["direction_results"].values())):
        raise ValueError("Readiness lacks a complete unchanged-policy FD result")
    grid = expected_policy["epsilons"]
    for label in item["direction_results"]:
        rows = [row for row in item.get("finite_differences", []) if row.get("direction") == label]
        if sorted(row["epsilon"] for row in rows) != sorted(grid):
            raise ValueError("Readiness omitted prescribed FD rows")


def _matching_replay_config(original_run, validation_run):
    """Require the actual resolved replay configuration, not a matching z alone.

    A saved-state validator records the ActiveSetConfig it used under
    config.replayed_config, including its authenticated YAML path/hash. Its
    separate execution metadata can describe the saved point and destination.
    Temperature, objective sign/scale, DBMol weights, adapters and every audit
    setting are compared; only path/name and source bookkeeping are excluded.
    """
    original = original_run["config"]
    replayed = validation_run.get("config", {}).get("replayed_config")
    if not isinstance(replayed, dict):
        raise ValueError("Saved-state validation lacks the resolved replayed_config")
    raw = []
    for config in (original, replayed):
        path = asset({"path": config["config_path"], "sha256": config["config_file_sha256"]})
        raw.append(yaml.safe_load(path.read_text()))
    if scientific_config(raw[0]) != scientific_config(raw[1]):
        raise ValueError("Saved-state replay changed original scientific configuration")
    bookkeeping = {"name", "output_root", "config_path", "config_file_sha256",
                   "source_sha256", "source_archive_sha256", "scope"}
    resolved = lambda config: {key: value for key, value in config.items() if key not in bookkeeping}
    if resolved(original) != resolved(replayed):
        raise ValueError("Saved-state replay changed resolved scientific configuration")


def validate_readiness(ref, plan):
    """Authenticate actual records; a completed attribution is never approval.

    Bundle schema: {version, native_gate:{run,summary,artifacts_sha256},
    failed_states:[{id,run,gate} x3]}. Each reference is {path,sha256}.
    Native artifacts pin every gate/metric/repair-refresh JSON used for evidence.
    Failed-state gates must live inside their completed validation run, retain
    original point/context/target/weight and pass every original local_v1 item.
    Their run config.replayed_config must retain the complete original resolved
    scientific configuration and its authenticated YAML path/hash.
    """
    ready = read(asset(ref))
    if (set(ready) != {"version", "native_gate", "failed_states"} or ready["version"] != 1
            or [row["id"] for row in ready["failed_states"]] != list(FAILURE_CASES)):
        raise ValueError("Require actual native and three saved-state validation artifacts")
    native = ready["native_gate"]
    root, record = _completed_run(native["run"], plan)
    summary_path = asset(native["summary"])
    if summary_path != root / "summary.json":
        raise ValueError("Native summary must belong to the completed gate run")
    summary = read(summary_path)
    arm = summary.get("arms", {}).get("fixed_initial_mask", {})
    if (summary.get("frozen_state_unchanged") is not True or arm.get("completed_requested_budget") is not True
            or arm.get("accepted_updates") != 6 or arm.get("requested_steps") != 6):
        raise ValueError("Native readiness requires six accepted updates with guidance evidence")
    parent, failure_plan = original_plans(plan["parent_protocol"], plan["failed_state_plan"])
    original_gate_config = yaml.safe_load(asset(parent["gate"]["config"]).read_text())
    native_config = yaml.safe_load(asset({"path": record["config"]["config_path"],
                                         "sha256": record["config"]["config_file_sha256"]}).read_text())
    if scientific_config(native_config) != scientific_config(original_gate_config):
        raise ValueError("Native gate changed the original scientific settings")
    files = sorted([*root.glob("fixed_initial_mask/gate-*.json"), *root.glob("fixed_initial_mask/metrics-*.json"),
                    *root.glob("fixed_initial_mask/repair-refresh-*/refresh.json")])
    if (not files or set(native["artifacts_sha256"]) != {str(path) for path in files}):
        raise ValueError("Readiness must pin every native evidence artifact")
    for path in files:
        authenticated(path, native["artifacts_sha256"][str(path)])
    for prefix in ("gate", "metrics"):
        if {path.name for path in files if path.name.startswith(prefix + "-")} != {f"{prefix}-{step:04d}.json" for step in range(7)}:
            raise ValueError("Native readiness must include all seven audited states")
    if any(read(path).get("passed") is not True for path in files if path.name.startswith("gate-")):
        raise ValueError("Native readiness includes an unsuccessful gate")
    from latentmol.repair_guidance.study import guidance_evidence
    evidence = guidance_evidence(root, summary, weight=1.)
    if not evidence["repair_validation_passed"]:
        raise ValueError("Native gate lacks actual repair-target/gradient evidence")
    for declared, result in zip(failure_plan["cases"], ready["failed_states"]):
        # Authenticate the actual saved z, target response, config and original
        # failed gate together; checking only a new passed boolean is insufficient.
        for old_path, old_sha in declared["parent_artifacts_sha256"].items():
            asset({"path": old_path, "sha256": old_sha})
        original_path = Path(declared["parent_run"]) / declared["arm"] / f"gate-{declared['step']:04d}.json"
        authenticated(original_path, declared["parent_artifacts_sha256"][str(original_path)])
        old = read(original_path)
        run_root, validation_run = _completed_run(result["run"], plan)
        old_run_path = Path(declared["parent_run"]) / "run.json"
        authenticated(old_run_path, declared["parent_artifacts_sha256"][str(old_run_path)])
        old_run = read(old_run_path)
        verify_config_identity(old_run)
        if (old_run.get("source_archive_sha256") != parent["source_archive_sha256"]
                or old_run.get("source", {}).get("sha256") != parent["source_manifest_sha256"]):
            raise ValueError("Original failed state belongs to another source release")
        _matching_replay_config(old_run, validation_run)
        if (validation_run.get("frozen_state_unchanged") is not True or any(
                validation_run.get(key) != old_run.get(key) or not old_run.get(key)
                for key in ("decoder_state_sha256", "scorer_state_sha256"))):
            raise ValueError("Saved-state readiness lacks unchanged original frozen-state identities")
        gate_path = asset(result["gate"])
        if not gate_path.is_relative_to(run_root):
            raise ValueError("Saved-state gate does not belong to its completed validation run")
        new = read(gate_path)
        if (old.get("passed") is not False or new.get("passed") is not True or new.get("status") != "passed"
                or new.get("native_seed_parity") is not True):
            raise ValueError("Completed diagnostics or failed gates are not recovery readiness")
        for key in ("point_sha256", "context_sha256", "active_slots", "effective_repair_weight"):
            if new.get(key) != old.get(key):
                raise ValueError("Saved-state validation changed point/context/mask/weight")
        if new.get("repair_only_validation", {}).get("target_sha256") != old["repair_only_validation"]["target_sha256"]:
            raise ValueError("Saved-state validation substituted a different repair target")
        expected_policy = next(item["policy"] for item in old["component_gradients"].values() if "policy" in item)
        for key in COMPONENTS:
            _completed_fd(new["component_gradients"][key], expected_policy)
        _completed_fd(new["local_fd_validation"], expected_policy)
        _completed_fd(new["repair_only_validation"], expected_policy)
        # Changed derivatives must still address the original failed random
        # directions, not merely choose a different easier Gaussian subset.
        for key, old_item in {**old.get("component_gradients", {}),
                              "combined": old.get("local_fd_validation", {})}.items():
            if old_item.get("passed") is not False:
                continue
            new_item = new["local_fd_validation"] if key == "combined" else new["component_gradients"][key]
            hashes = lambda item: {r["direction_sha256"] for r in item["direction_selection"]["proposals"] if r["accepted"]}
            if hashes(new_item) != hashes(old_item):
                raise ValueError("Saved-state readiness replaced original failed random directions")
    return {"readiness": ref, "native_gate": str(root), "failed_states_verified": list(FAILURE_CASES)}


def workflow(release, work, plan_ref, ready_ref, *, archive_sha, manifest_sha):
    """Pure scheduler plan: CPU fixtures then one exclusive managed-MPS A100."""
    label = release.name + "-single-defog"
    stages = [("cpu", "hpc", "0:30", 2, "4GB", None,
               ["python", "-m", "pytest", "-q", "tests/test_defog_single_recovery.py"]),
              ("native", "gpua100", "8:00", 4, "16GB", "done:cpu",
               ["python", "scripts/hpc/submit_defog_single_recovery.py", "run", "--plan", plan_ref["path"],
                "--sha256", plan_ref["sha256"], "--readiness", ready_ref["path"], "--readiness-sha256", ready_ref["sha256"]])]
    result = []
    for stage, queue, wall, cores, memory, dependency, command in stages:
        name = label + "-" + stage
        arguments = ["bsub", "-J", name, "-q", queue, "-n", str(cores), "-R", "rusage[mem=" + memory + "]",
                     "-R", "span[hosts=1]", "-W", wall, "-oo", str(work / "logs" / (name + ".%J.out")),
                     "-eo", str(work / "logs" / (name + ".%J.err"))]
        if stage == "native":
            arguments += ["-R", "select[gpu80gb]", "-gpu", gpu_request(managed_mps=True)]
        job = ["env", "LATENTMOL_SOURCE_ARCHIVE=" + str(release) + ".tar.gz", "LATENTMOL_WORK_ROOT=" + str(work),
               "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
               "--archive-sha256", archive_sha, "--manifest-sha256", manifest_sha, "--",
               "python", "-m", "latentmol.experiments", "--run-root", str(work / "runs"), "--name", name, "--", *command]
        result.append({"stage": stage, "cwd": str(release), "arguments": arguments,
                       "job_command": job, "dependency": dependency})
    return result


def fresh_pose_outcome(summary, pose_seeds):
    """Allow documented raw-invalid skips, but require all eligible predictions.

    The rescorer authenticates raw graph identities before writing selection.
    This reader checks its exact deduplicated molecule/seed quota, so missing
    predictions for a valid molecule cannot be disguised as an unscorable skip.
    """
    chosen = summary["selection"]["arms"]["fixed_initial_mask"]
    start, endpoint = chosen["start"]["smiles"], chosen["endpoint"]["smiles"]
    raw_invalid = start is None or endpoint is None
    selection_consistent = all(
        isinstance(chosen[part].get("failure"), str) and bool(chosen[part]["failure"])
        if chosen[part]["smiles"] is None else isinstance(chosen[part]["smiles"], str)
        and bool(chosen[part]["smiles"]) and chosen[part].get("failure") is None
        for part in ("start", "endpoint"))
    expected_predictions = len({value for value in (start, endpoint) if value}) * len(pose_seeds)
    comparison = summary["comparisons"]["fixed_initial_mask"]
    pairs = comparison["paired_seeds"]
    complete_rows = len(pairs) == len(pose_seeds) and {row["seed"] for row in pairs} == set(pose_seeds)
    eligible_pairs = sum(row.get("paired_retention_eligible") is True for row in pairs)
    matched = complete_rows and eligible_pairs == len(pose_seeds)
    exact_quota = (summary.get("prediction_attempts") == expected_predictions
                   and summary.get("retention_eligible_predictions") == expected_predictions)
    if summary.get("failed_predictions", 1) != 0:
        status = "prediction_failed"
    elif summary.get("identity_ineligible_predictions", 1) != 0:
        status = "identity_ineligible"
    elif (not selection_consistent or not complete_rows or not exact_quota
            or comparison.get("identity_comparable") != (not raw_invalid)
            or (raw_invalid and eligible_pairs != 0) or (not raw_invalid and not matched)):
        status = "incomplete_evaluation"
    else:
        status = "completed"
    comparison_status = "failed" if status != "completed" else "unscorable" if raw_invalid else "scorable"
    return {"status": status, "process_completed": True, "computational_completion": status == "completed",
            "comparison_status": comparison_status, "all_declared_pairs_eligible": matched,
            "eligible_pair_count": eligible_pairs, "declared_pair_count": len(pose_seeds),
            "raw_start_scorable": start is not None, "raw_endpoint_scorable": endpoint is not None,
            "raw_failures": {part: chosen[part].get("failure") for part in ("start", "endpoint")},
            "unscorable_reason": "invalid_raw_start_or_endpoint" if raw_invalid else None,
            "prediction_attempts": summary.get("prediction_attempts"), "expected_prediction_attempts": expected_predictions,
            "failed_predictions": summary.get("failed_predictions"),
            "identity_ineligible_predictions": summary.get("identity_ineligible_predictions"),
            "retention_eligible_predictions": summary.get("retention_eligible_predictions"),
            "paired_seeds": [{"seed": row["seed"], "eligible": row.get("paired_retention_eligible")}
                             for row in pairs],
            "claim_limit": "Fresh-pose model predictions only; invalid endpoints remain unscorable and are never replaced."}


def run(plan_path, sha, ready_ref):
    """Reuse the existing optimizer/rescorer; fail on every incomplete outcome."""
    from latentmol.repair_guidance.random_study import _optimize
    from latentmol.repair_guidance.study import require_hpc
    from latentmol.boltz.stability_study import verified_child
    from latentmol.boltz.workflow_process import run_logged
    from latentmol.boltz.native_rescore import POSE_SEEDS
    require_hpc()
    p = load_plan(plan_path, sha)
    identity = {"source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
                "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"]}
    if any(p[key] != value for key, value in identity.items()):
        raise ValueError("Single recovery belongs to another sealed source release")
    readiness = validate_readiness(ready_ref, p)
    started = time.monotonic()
    with ExperimentRun(Path(p["output_root"]) / "controller", p["name"],
                       {"plan": str(plan_path), "plan_sha256": sha, **identity, **readiness}, Path.cwd()) as experiment:
        report = {"completed_requested_case": False, "computational_completion": False,
                  "comparison_status": "not_evaluated", "case": {"id": "prior-0", "status": "not_attempted"},
                  "readiness": readiness, "resume": False}
        path = experiment.path / "recovery.json"
        write_json(path, report)
        try:
            raw = yaml.safe_load(asset(p["case"]["config"]).read_text())
            probe_path = experiment.path / "gpu-process.json"
            probe = run_logged([sys.executable, "-m", "latentmol.gpu_process_probe", "--worker-python",
                raw["repair_guidance"]["provider"]["python"], "--output", str(probe_path)],
                experiment.path / "gpu-process.log", timeout_seconds=180)
            report["gpu_process_execution"] = probe
            write_json(path, report)
            if (probe["timed_out"] or probe["returncode"] != 0 or read(probe_path).get("status") != "passed"):
                raise RuntimeError("Allocated managed-MPS concurrency probe failed")
            if p["allocation_seconds"] - (time.monotonic() - started) < OPTIMIZATION_SECONDS + RESCORE_SECONDS + 60:
                raise RuntimeError("Insufficient allocation time for the full declared case")
            report["case"].update(status="running", phase="optimization")
            write_json(path, report)
            result = _optimize(p["case"], experiment.path, "prior-0", identity, OPTIMIZATION_SECONDS)
            report["case"].update(result)
            report["case"]["optimization"] = {"status": result["status"], "accepted_updates": result["accepted_updates"],
                "completed_requested_budget": result["status"] == "completed" and result["accepted_updates"] == 100}
            write_json(path, report)
            if result["status"] != "completed" or result["accepted_updates"] != 100 or not result["repair_evidence"]["repair_validation_passed"]:
                raise RuntimeError("Requested 100-update guided trajectory is incomplete")
            child = Path(result["child_run"])
            report["case"]["phase"] = "fresh_pose"
            rescore_root = Path(p["output_root"]) / "prior-0" / "fresh-pose"
            execution = run_logged([sys.executable, "-m", "latentmol.boltz.active_set_rescore", "--pilot", str(child),
                "--template", str(asset(p["case"]["native_template"])), "--cache", p["cache"], "--run-root", str(rescore_root)],
                experiment.path / "prior-0-rescore.log", timeout_seconds=RESCORE_SECONDS)
            report["case"]["fresh_pose_execution"] = execution
            write_json(experiment.path / "prior-0-rescore-execution.json", execution)
            native = verified_child(execution, rescore_root, identity, parent_run_id=read(child / "run.json")["run_id"])
            fresh = fresh_pose_outcome(read(native / "summary.json"), POSE_SEEDS)
            report["case"]["fresh_pose"] = {**fresh, "run": str(native), "summary": reference(native / "summary.json")}
            report["comparison_status"] = fresh["comparison_status"]
            write_json(path, report)
            if fresh["status"] != "completed":
                raise RuntimeError("Fresh-pose evaluation is " + fresh["status"] + "; original endpoint retained")
            report["case"]["phase"] = "completed"
            report["completed_requested_case"] = True
            report["computational_completion"] = True
            report["seconds"] = time.monotonic() - started
            write_json(path, report)
            experiment.annotate(completed_requested_case=True, computational_completion=True,
                                comparison_status=fresh["comparison_status"])
            return experiment.path
        except Exception as error:
            report["case"].update(status="failed", error=str(error), error_type=type(error).__name__)
            report["retained_execution_receipts"] = [reference(receipt) for receipt in
                sorted(experiment.path.glob("*-execution.json"))]
            report.update(seconds=time.monotonic() - started, completed_requested_case=False, computational_completion=False)
            write_json(path, report)
            experiment.annotate(completed_requested_case=False, computational_completion=False,
                                comparison_status=report["comparison_status"])
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for key in ("parent-protocol", "failed-state-plan"):
        prep.add_argument("--" + key, type=Path, required=True)
        prep.add_argument("--" + key + "-sha256", required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--campaign-root", type=Path, required=True)
    prep.add_argument("--archive-sha256", required=True)
    prep.add_argument("--manifest-sha256", required=True)
    for name in ("submit", "run"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--plan", type=Path, required=True)
        cmd.add_argument("--sha256", required=True)
        cmd.add_argument("--readiness", type=Path)
        cmd.add_argument("--readiness-sha256")
        if name == "submit":
            cmd.add_argument("--release", type=Path, default=Path.cwd())
            cmd.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
            cmd.add_argument("--receipts", type=Path, required=True)
            cmd.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare({"path": str(args.parent_protocol.resolve()), "sha256": args.parent_protocol_sha256},
                      {"path": str(args.failed_state_plan.resolve()), "sha256": args.failed_state_plan_sha256},
                      args.output, args.campaign_root, archive_sha=args.archive_sha256, manifest_sha=args.manifest_sha256))
        return
    if bool(args.readiness) != bool(args.readiness_sha256):
        raise ValueError("Readiness path and SHA256 must be supplied together")
    ready = {"path": str(args.readiness.resolve()), "sha256": args.readiness_sha256} if args.readiness else None
    if args.command == "run":
        if ready is None:
            raise ValueError("Actual new-release native/FD readiness is required before running")
        print(run(args.plan, args.sha256, ready))
        return
    p = load_plan(args.plan, args.sha256)
    release, work = args.release.resolve(strict=True), args.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require a sealed DTU release under the declared work root")
    for path in (args.plan, args.receipts, Path(p["output_root"]), Path(p["cache"])):
        if not path.is_absolute() or not path.resolve().is_relative_to(work):
            raise ValueError("Recovery plan, receipts and output paths must stay under the declared DTU root")
    authenticated(Path(str(release) + ".tar.gz"), p["source_archive_sha256"])
    if source_manifest(release)["sha256"] != p["source_manifest_sha256"] or read(release / "release_manifest.json")["sha256"] != p["source_manifest_sha256"]:
        raise ValueError("Release changed after sealing")
    if ready is None:
        if args.submit:
            raise ValueError("Unresolved FD diagnosis blocks recovery submission; actual readiness artifacts required")
        print(json.dumps({"status": "prepared_not_submitted_blocked_by_validation", "plan": p}, indent=2))
        return
    validation = validate_readiness(ready, p)
    jobs = workflow(release, work, reference(args.plan), ready, archive_sha=p["source_archive_sha256"], manifest_sha=p["source_manifest_sha256"])
    identity = {"plan": reference(args.plan), "release": str(release), "readiness": validation, "source_archive_sha256": p["source_archive_sha256"]}
    if args.submit:
        (work / "logs").mkdir(exist_ok=True)
        print(json.dumps(submit(jobs, args.receipts, identity), indent=2))
    else:
        print(json.dumps({"status": "not_submitted", "identity": identity, "jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
