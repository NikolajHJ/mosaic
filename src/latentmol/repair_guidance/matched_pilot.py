"""Sequential Boltz-only / Boltzâ€“DeFoG pilot from one authenticated saved z.

This new protocol does not widen the earlier six-update certificate. Every
child runs fresh fail-closed native/derivative gates. Both branches use the
original frozen decoder and one canonical graph. Discrete repair and bounded
repair-search witnesses stay outside autograd; only molecular z changes.
Each new sealed run requires its own allocated checks; historical passes do
not certify a changed run. The 25-update certificate below remains unchanged.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.stability_study import verify_asset, verified_child
from latentmol.boltz.workflow_process import run_logged
from latentmol.config_identity import canonical_config_json, verify_config_identity
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from .integration_smoke import full_gate_passed
from .random_study import _optimize
from .study import require_hpc


CASE_IDS = ("boltz_only", "boltz_defog")
STEPS = 25
AUDIT_INTERVAL = 5
HARD_SAMPLES = 8
OCCUPIED_SLOTS = 22
BUDGETS = {"allocation_seconds": 28200, "optimization_timeout_seconds": 10800,
           "repair_timeout_seconds": 900, "rescore_timeout_seconds": 1800}
BRANCH_BUDGETS = {"allocation_seconds": 42600, "optimization_timeout_seconds": 18000,
                  "repair_timeout_seconds": 900, "rescore_timeout_seconds": 1800}


def case_record(base, case_id, output_root, *, derivative_contract="local_v1"):
    """Declare the complete difference from the short original-decoder config.

    A zero base weight alone would NOT disable a positive ramp. Remove the
    schedule in the control so its provider activation is exactly zero. Keep
    the disabled repair wrapper for common trajectory/evidence reporting.
    """
    if case_id not in CASE_IDS:
        raise ValueError("Unknown paired-pilot arm")
    raw = deepcopy(base.record())
    if derivative_contract in ("branch_local_v2", "additive_branch_v3", "pointwise_native_ad_v4"):
        raw["fd_policy"] = derivative_contract
        raw["dbmol"]["objective_arithmetic"] = "contact_head_and_assembly_float64_v1"
    elif derivative_contract != "local_v1":
        raise ValueError("Unsupported pilot derivative contract")
    raw.update(name="matched-pilot-" + case_id, output_root=str(output_root),
               gradient_audit_interval=AUDIT_INTERVAL)
    raw["optimizer"]["steps"] = STEPS
    raw["diagnostics"]["hard_samples"] = HARD_SAMPLES
    if case_id == "boltz_only":
        raw["repair_guidance"]["weight"] = 0.0
        raw["repair_guidance"].pop("weight_schedule", None)
    return raw


def validate_configs(base, configs, root, *, derivative_contract="local_v1"):
    """Pin all undeclared science and reject a nominal control with a live ramp."""
    if (base.optimizer.steps != 6 or base.gradient_audit_interval != 6
            or base.arms != ("fixed_initial_mask",) or base.fd_policy != "local_v1"
            or base.step_control is None or base.dbmol is None
            or base.dbmol.objective_arithmetic != "contact_and_assembly_float64_v1"
            or base.start.kind != "standard_normal" or base.start.seed != 0
            or base.initial_latent is None or base.repair_guidance is None
            or base.repair_guidance.refresh_interval != 5
            or base.repair_guidance.attempts_per_refresh != 4
            or base.repair_guidance.weight_schedule is None
            or "charge_policy" in base.repair_guidance.provider):
        raise ValueError("Require the authenticated strict-neutral source089 configuration")
    if set(configs) != set(CASE_IDS):
        raise ValueError("Require exactly one control and one guided case")
    for case_id, cfg in configs.items():
        expected = case_record(base, case_id, root / case_id / "optimization", derivative_contract=derivative_contract)
        if cfg.record() != expected:
            raise ValueError("Pilot changes undeclared science or matching: " + case_id)
    if configs["boltz_only"].repair_guidance.provider_activation_weight != 0:
        raise ValueError("Boltz-only control must never activate DeFoG")


def measured_descent_passed(report, trace, config, context, *, requested_updates=STEPS):
    """Recheck the native accepted trial at its own fixed target and coefficient.

    The pointwise AD contract does not assert finite-step linear accuracy.
    This independent receipt check therefore requires the actual Armijo trial,
    not a count of accepted updates or monotonicity across target refreshes.
    The scalar-ULP guard remains an observation rule, not a network error bound.
    """
    try:
        if config.step_control.version != "armijo_v1":
            return False
        index = report["accepted_trial"]
        trials = report["trials"]
        if (type(index) is not int or not 0 <= index < len(trials)
                or len(trials) != index + 1 or report["policy"] != config.step_control.record()
                or report.get("status") != "accepted" or report.get("accepted") is not True
                or report.get("stalled") is not False or trace.get("accepted") is not True
                or report["baseline_loss"] != trace["total_loss"]):
            return False
        trial = trials[index]
        fields = (report["baseline_loss"], trial["candidate_loss"],
                  trial["original_gradient_dot_actual_displacement"], trial["observed_improvement_guard"],
                  trial["actual_displacement_norm"], trial["actual_initial_displacement_norm"],
                  report["initial_displacement_norm"])
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in fields):
            return False
        baseline, candidate, dot, guard, displacement, radius, current_radius = fields
        improvement = baseline - candidate
        variation = report.get("baseline_variation")
        if (variation is not None and (type(variation) not in (int, float)
                or not math.isfinite(variation) or variation < 0)):
            return False
        expected_guard = config.step_control.minimum_improvement_ulps * max(
            math.ulp(baseline), math.ulp(candidate)) + (variation or 0.)
        metadata = trial["objective_metadata"]
        repair = metadata["repair_guidance"]
        expected_weight = config.repair_guidance.weight_at(trace["step"], requested_updates)
        return (trial.get("accepted") is True and trial.get("reason") == "accepted"
            and all(row.get("accepted") is False for row in trials[:index])
            and dot < 0 and guard == expected_guard and displacement > 0
            and report["baseline_loss_dtype"] == trial["candidate_loss_dtype"] == "torch.float64"
            and report["baseline_output_ulp"] == math.ulp(baseline)
            and trial["output_ulp"] == math.ulp(candidate)
            and report["baseline_variation_measured"] is (variation is not None)
            and all(trial.get(key) is True for key in ("strictly_decreased", "informative_decrease", "armijo_satisfied"))
            and 0 <= radius <= config.optimizer.trust_radius + 1e-6
            and 0 <= current_radius <= config.optimizer.trust_radius + 1e-6
            # The trust region is centered on the original z. An inward or
            # curved step can exceed either endpoint's distance to that center.
            and displacement <= current_radius + radius + 1e-6
            and improvement > 0 and improvement >= guard
            and improvement >= -config.step_control.armijo * dot
            and trial["actual_loss_improvement"] == improvement
            and report["actual_loss_improvement"] == improvement
            and metadata["loss"] == candidate and metadata["context_sha256"] == context
            and repair["weight"] == expected_weight
            and repair.get("target_sha256") == trace.get("repair_target_sha256")
            and repair.get("used_target") is trace.get("used_target"))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def measured_step_passed(report, trace, config, context, *, requested_updates=STEPS):
    """Authenticate an accepted update according to its actual versioned policy.

    Historical Armijo receipts retain their original sufficient-decrease proof.
    Best-trial receipts instead prove the complete search and finite selection;
    every scored finite candidate must use the same native context, held repair
    target and scheduled weight. A fallback pass is an execution-policy pass,
    not a monotonicity, Armijo or convergence claim.
    """
    try:
        if config.step_control.version == "armijo_v1":
            return measured_descent_passed(report, trace, config, context,
                                          requested_updates=requested_updates)
        if config.step_control.version != "best_trial_v2":
            return False
        from latentmol.boltz.trajectory_terminal import validate_step_report
        validate_step_report(report, config.step_control)
        baseline = report["baseline_loss"]
        variation = report.get("baseline_variation")
        current_radius = report["initial_displacement_norm"]
        if (trace.get("accepted") is not True or baseline != trace["total_loss"]
                or report["baseline_loss_dtype"] != "torch.float64"
                or report["baseline_output_ulp"] != math.ulp(baseline)
                or report["baseline_variation_measured"] is not (variation is not None)
                or type(current_radius) not in (int, float) or not math.isfinite(current_radius)
                or not 0 <= current_radius <= config.optimizer.trust_radius + 1e-6
                or variation is not None and (type(variation) not in (int, float)
                    or not math.isfinite(variation) or variation < 0)):
            return False
        weight = config.repair_guidance.weight_at(trace["step"], requested_updates)
        for trial in report["trials"]:
            if trial.get("reason") in ("nonfinite_proposal", "nonfinite_objective"):
                continue
            candidate = trial["candidate_loss"]
            dot = trial["original_gradient_dot_actual_displacement"]
            displacement = trial["actual_displacement_norm"]
            radius = trial["actual_initial_displacement_norm"]
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in (candidate, dot, displacement, radius)):
                return False
            metadata = trial["objective_metadata"]
            repair = metadata["repair_guidance"]
            if (trial.get("finite_proposal") is not True or trial.get("finite_candidate") is not True
                    or trial["candidate_loss_dtype"] != "torch.float64"
                    or trial["output_ulp"] != math.ulp(candidate)
                    or not 0 <= radius <= config.optimizer.trust_radius + 1e-6
                    or displacement < 0 or displacement > current_radius + radius + 1e-6
                    or metadata["loss"] != candidate or metadata["context_sha256"] != context
                    or repair["weight"] != weight
                    or repair.get("target_sha256") != trace.get("repair_target_sha256")
                    or repair.get("used_target") is not trace.get("used_target")):
                return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def measured_step_check_name(config):
    """Preserve historical check names while naming nonmonotone policy honestly."""
    return ("measured_native_step_policy" if config.step_control.version == "best_trial_v2"
            else "measured_native_descent")


def validate_case(result, trajectory, gates, *, guided, config=None, step_reports=None):
    """Require actual updates and native guidance, without requiring raw validity.

    Repair may fail during the pilot. Its fallback remains recorded by the
    optimizer, but a fallback trajectory cannot pass the guidance acceptance
    criteria. No target is silently substituted into the binding branch.
    """
    evidence = result.get("repair_evidence", {})
    trace = trajectory.get("trace", [])
    updates = [row for row in trace if row.get("accepted") is True]
    expected_gates = {f"step-{s:04d}" for s in range(0, STEPS + 1, AUDIT_INTERVAL)}
    def verified_gate(label, gate):
        # New contracts need coefficients from the authenticated configuration,
        # including the scheduled weight at THIS audit. A receipt cannot approve
        # itself by declaring a different, easier objective profile.
        gate_policy = gate.get("derivative_contract", {}).get("version")
        if gate_policy in ("additive_branch_v3", "pointwise_native_ad_v4"):
            if config is None or config.fd_policy != gate_policy:
                return False
            from latentmol.boltz.additive_contract import profile_record
            step = int(label.removeprefix("step-"))
            weight = config.repair_guidance.weight_at(step, STEPS) if guided else None
            profile = profile_record(config.dbmol, config.objective, repair_weight=weight)
            return full_gate_passed(gate, require_repair=guided, expected_profile=profile)
        if config is not None and config.fd_policy in ("additive_branch_v3", "pointwise_native_ad_v4"):
            return False
        return full_gate_passed(gate, require_repair=guided)
    checks = {
        "complete_budget": (result.get("status") == "completed"
            and result.get("accepted_updates") == STEPS
            and trajectory.get("requested_steps") == STEPS
            and trajectory.get("completed_requested_budget") is True),
        "all_state_receipts": [row.get("step") for row in trace] == list(range(STEPS + 1)),
        "fixed_occupancy": trajectory.get("fixed_counts") == [OCCUPIED_SLOTS],
        "raw_sampling_budget": trajectory.get("prescribed_candidate_count") == (STEPS + 1) * (HARD_SAMPLES + 1),
        "full_guards": (set(gates) == expected_gates
            and all(verified_gate(label, gate) for label, gate in gates.items())),
    }
    if config is not None and config.fd_policy == "pointwise_native_ad_v4":
        contexts = {gate.get("context_sha256") for gate in gates.values()}
        checks[measured_step_check_name(config)] = (isinstance(step_reports, dict)
            and set(step_reports) == set(range(STEPS)) and len(contexts) == 1 and None not in contexts
            and len(updates) == STEPS and all(measured_step_passed(
                step_reports[row["step"]], row, config, next(iter(contexts))) for row in updates))
    if guided:
        # No refresh occurs at the endpoint; it retains the step-20 target.
        refreshes = STEPS // AUDIT_INTERVAL
        checks["native_targets"] = (evidence.get("accepted_targets") == refreshes
            and evidence.get("refreshes") == refreshes
            and evidence.get("native_attempts", 0) > 0
            and evidence.get("accepted_native_targets", 0) >= refreshes)
        pointwise = config is not None and config.fd_policy == "pointwise_native_ad_v4"
        derivative_count = evidence.get("target_only_derivative_passes" if pointwise else "target_only_fd_passes")
        checks["actual_guidance"] = (evidence.get("repair_validation_passed") is True
            and evidence.get("guided_accepted_updates") == STEPS
            and evidence.get("nonzero_guided_updates") == STEPS
            and derivative_count == len(expected_gates)
            and (not pointwise or (evidence.get("target_only_pointwise_ad_passes") == len(expected_gates)
                                  and evidence.get("target_only_fd_passes") == 0)))
        checks["both_gradients_reach_z"] = len(updates) == STEPS and all(
            row.get("used_target") is True and row.get("gradient_measured") is True
            and all(isinstance(value, (float, int)) and math.isfinite(value) and value > 1e-12
                    for value in (row.get("gradient_diagnostics", {}).get("binding_norm"),
                                  row.get("gradient_diagnostics", {}).get("weighted_repair_norm")))
            for row in updates)
    else:
        checks["no_native_repair_or_guidance"] = (
            all(evidence.get(key) == 0 for key in ("native_attempts", "accepted_targets",
                "guided_accepted_updates", "nonzero_guided_updates", "target_only_fd_passes"))
            and all(row.get("used_target") is False and row.get("applied_weight") == 0 for row in trace))
    return {"passed": all(checks.values()), "checks": checks,
        "claim_limit": "Engineering trajectory only; raw validity and fresh binding predictions are separate outcomes"}


def verify_start(child, initial_latent):
    """Compare actual saved input tensors on HPC, never use a smoke endpoint."""
    import torch
    actual = torch.load(child / "start.pt", map_location="cpu", weights_only=True)["z"]
    original = torch.load(verify_asset(initial_latent), map_location="cpu", weights_only=True)["z"]
    if actual.shape != original.shape or actual.dtype != original.dtype or not torch.equal(actual, original):
        raise ValueError("Pilot changed the prescribed saved initial z")
    from latentmol.multimodel.study import tensor_digest
    return tensor_digest({"z": actual})


def validate_prior_run(base_record, prior, *, base_config, prior_run_path, child_run):
    """Tie the parsed base to the authenticated historical child after JSON I/O.

    ``ActiveSetConfig.record`` retains integer context keys and tuples, while
    ExperimentRun persists JSON string keys and arrays. Compare the existing
    canonical JSON identities, not Python container equality. This accepts
    only those serialization changes: scalar types, values and array order
    still matter, ambiguous keys fail, and no missing field defaults to None.
    Runtime-only fields in the historical receipt are verified by its full
    config hash but are outside the parsed base's declared-field projection.
    """
    verify_config_identity(prior, allow_legacy_active_set_contexts=True)
    prior_config = prior["config"]
    if (prior.get("status") != "completed"
            or str(Path(prior_run_path).parent) != child_run
            or prior_config.get("config_file_sha256") != base_config["sha256"]
            or prior_config.get("config_path") != base_config["path"]):
        raise ValueError("Base config is not the actual authenticated short-test child")
    missing = sorted(set(base_record) - set(prior_config))
    if missing:
        raise ValueError("Historical child is missing base config fields: " + ", ".join(missing))
    expected = canonical_config_json(base_record)
    projected = {key: prior_config[key] for key in base_record}
    if canonical_config_json(projected) != expected:
        changed = sorted(key for key in base_record
            if canonical_config_json(prior_config[key]) != canonical_config_json(base_record[key]))
        raise ValueError("Historical child differs from parsed base config fields: " + ", ".join(changed))


def run_followups(case_id, child, plan, parent, identity):
    """Bounded repair audit, then fresh raw-argmax predictions, strictly serial.

    Invalid raw endpoints stay invalid. Repair-search witnesses and DeFoG
    targets are never substituted into fresh binding scoring. Follow-up failure
    is separate from optimization failure and cannot certify an advantage.
    """
    result = {}
    root = Path(plan["output_root"]) / case_id
    audit = {"version": 1, "name": "matched-pilot-repair-" + case_id,
        "output_root": str(root / "repair-audit"),
        "parents": [{"id": case_id, "pilot": str(child), "run_sha256": file_sha256(child / "run.json"),
                     "steps": [0, 10, 20, STEPS]}], "search": plan["repair_search"]}
    path = parent / (case_id + "-repair-protocol.json")
    write_json(path, audit)
    commands = [
        ("repair_audit", [sys.executable, "-m", "latentmol.probability_repair_audit", "--protocol", str(path),
            "--sha256", file_sha256(path)], root / "repair-audit", plan["repair_timeout_seconds"]),
        ("fresh_pose", [sys.executable, "-m", "latentmol.boltz.active_set_rescore", "--pilot", str(child),
            "--template", str(verify_asset(plan["native_template"])), "--cache", plan["cache"],
            "--run-root", str(root / "fresh-pose")], root / "fresh-pose", plan["rescore_timeout_seconds"]),
    ]
    for label, command, run_root, timeout in commands:
        execution = None
        try:
            execution = run_logged(command, parent / (case_id + "-" + label + ".log"), timeout_seconds=timeout)
            write_json(parent / (case_id + "-" + label + "-execution.json"), execution)
            kwargs = {"parent_run_id": json.loads((child / "run.json").read_text())["run_id"]} if label == "fresh_pose" else {}
            native = verified_child(execution, run_root, identity, **kwargs)
            result[label] = {"status": "completed", "execution": execution, "run": str(native),
                "summary": {"path": str(native / "summary.json"), "sha256": file_sha256(native / "summary.json")}}
        except Exception as error:
            result[label] = {"status": "failed", "execution": execution,
                             "error": str(error), "error_type": type(error).__name__}
    return result


def load_plan(plan_path, expected_sha):
    """Authenticate actual CPU-prepared configs without loading any model."""
    require_hpc()
    plan = json.loads(verify_asset({"path": str(plan_path.resolve()), "sha256": expected_sha}).read_text())
    fields = {"version", "name", "output_root", "base_config", "cases", "prior_smoke", "prior_run",
              "native_template", "cache", "repair_search", "source_manifest_sha256", "source_archive_sha256", *BUDGETS}
    new_contract = plan.get("version") in (2, 3, 4)
    if new_contract: fields |= {"derivative_contract", "saved_state_readiness"}
    if set(plan) != fields or type(plan["version"]) is not int or plan["version"] not in (1, 2, 3, 4):
        raise ValueError("Unsupported matched-pilot plan")
    contract = plan.get("derivative_contract", "local_v1")
    if new_contract and contract != {2: "branch_local_v2", 3: "additive_branch_v3", 4: "pointwise_native_ad_v4"}[plan["version"]]:
        raise ValueError("Require the version's explicit derivative contract and saved-state readiness")
    if new_contract:
        if plan["version"] == 4:
            from latentmol.boltz.pointwise_readiness import validate_readiness_binding
        elif plan["version"] == 3:
            from latentmol.boltz.additive_readiness import validate_readiness_binding
        else:
            from latentmol.boltz.contact_precision_validation import validate_readiness_binding
        validate_readiness_binding(plan["saved_state_readiness"])
    expected_budgets = BRANCH_BUDGETS if new_contract else BUDGETS
    if any(plan[key] != value for key, value in expected_budgets.items()):
        raise ValueError("Changed declared allocation/child/follow-up budgets")
    identity = {"source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
                "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"]}
    if any(plan[key] != value for key, value in identity.items()):
        raise ValueError("Pilot belongs to another sealed release")
    root = Path(plan["output_root"])
    if not root.is_absolute() or not Path(plan["cache"]).is_absolute():
        raise ValueError("Require absolute pilot output/cache paths")
    if [row.get("id") for row in plan["cases"]] != list(CASE_IDS) or any(set(row) != {"id", "config"} for row in plan["cases"]):
        raise ValueError("Require the complete prescribed pair in control-first order")
    base = ActiveSetConfig.load(verify_asset(plan["base_config"]))
    configs = {row["id"]: ActiveSetConfig.load(verify_asset(row["config"])) for row in plan["cases"]}
    validate_configs(base, configs, root, derivative_contract=contract)
    smoke = json.loads(verify_asset(plan["prior_smoke"]).read_text())
    if smoke.get("passed") is not True or smoke.get("outcome", {}).get("passed") is not True:
        raise ValueError("Require the completed historical short integration proof")
    prior = json.loads(verify_asset(plan["prior_run"]).read_text())
    validate_prior_run(base.record(), prior, base_config=plan["base_config"],
        prior_run_path=plan["prior_run"]["path"], child_run=smoke["result"]["child_run"])
    verify_asset(plan["native_template"])
    # Validate the bounded search settings before allocating model children.
    from latentmol.probability_repair import ProbabilityRepairConfig
    ProbabilityRepairConfig(**plan["repair_search"])
    reserve = 180 + len(CASE_IDS) * (plan["optimization_timeout_seconds"]
        + plan["repair_timeout_seconds"] + plan["rescore_timeout_seconds"] + 60)
    if reserve >= plan["allocation_seconds"]:
        raise ValueError("Allocation cannot reserve the complete matched pair")
    return plan, base, configs, identity, reserve


def run(plan_path, expected_sha):
    """One GPU allocation; independent cases retain failures without retries."""
    plan, base, configs, identity, reserve = load_plan(plan_path, expected_sha)
    if plan["version"] in (2, 3, 4):
        if plan["version"] == 4:
            from latentmol.boltz.pointwise_readiness import require_readiness
        elif plan["version"] == 3:
            from latentmol.boltz.additive_readiness import require_readiness
        else:
            from latentmol.boltz.contact_precision_validation import require_readiness
        require_readiness(plan["saved_state_readiness"], identity)
    root = Path(plan["output_root"])
    started = time.monotonic()
    with ExperimentRun(root / "controller", plan["name"], {"plan": str(plan_path),
            "plan_sha256": expected_sha, **identity}, Path.cwd()) as experiment:
        report = {"status": "running", "engineering_passed": False,
                  "cases": [{"id": case_id, "status": "not_attempted"} for case_id in CASE_IDS],
                  "matching": "same saved z, original decoder, fixed22, pocket, Boltz objective, diagnostics and step budget; repair only differs",
                  "cost_matching": "Update/diagnostic budgets matched; DeFoG calls and target derivative audits are additional compute",
                  "chemical_advantage_established": False}
        output = experiment.path / "pilot.json"
        write_json(output, report)
        try:
            probe = experiment.path / "gpu-process.json"
            execution = run_logged([sys.executable, "-m", "latentmol.gpu_process_probe", "--worker-python",
                base.repair_guidance.provider["python"], "--output", str(probe)], experiment.path / "gpu-process.log", timeout_seconds=180)
            report["gpu_process_execution"] = execution
            if execution["timed_out"] or execution["returncode"] != 0 or json.loads(probe.read_text()).get("status") != "passed":
                report.update(status="failed_gpu_coexistence")
                write_json(output, report)
                raise RuntimeError("Allocated GPU process coexistence probe failed")
            # Reserve all remaining cases, rather than spending control's budget
            # and then selectively excluding the guided arm at the allocation end.
            remaining = plan["allocation_seconds"] - (time.monotonic() - started)
            if remaining < reserve - 180:
                report.update(status="insufficient_complete_pair_budget")
                write_json(output, report)
                raise RuntimeError("Insufficient remaining budget to start the matched pair")
            for declared, row in zip(plan["cases"], report["cases"]):
                case_id = declared["id"]
                row["status"] = "running"
                write_json(output, report)
                try:
                    row.update(_optimize(declared, experiment.path, case_id, identity, plan["optimization_timeout_seconds"]))
                    child = Path(row["child_run"])
                    row["initial_z_sha256"] = verify_start(child, base.initial_latent)
                    trajectory = json.loads(Path(row["trajectory"]["path"]).read_text())
                    gates = json.loads((child / "fixed_initial_mask/gates.json").read_text())
                    step_reports = ({step: json.loads((child / "fixed_initial_mask" /
                        f"step-control-{step:04d}.json").read_text()) for step in range(STEPS)}
                        if configs[case_id].fd_policy == "pointwise_native_ad_v4" else None)
                    row["outcome"] = validate_case(row, trajectory, gates, guided=case_id == "boltz_defog",
                        config=configs[case_id], step_reports=step_reports)
                    row["status"] = "passed" if row["outcome"]["passed"] else "failed_acceptance"
                    row.update(run_followups(case_id, child, plan, experiment.path, identity))
                except Exception as error:
                    row.update(status="failed", error=str(error), error_type=type(error).__name__)
                write_json(output, report)
            for reference in (plan["base_config"], plan["prior_smoke"], plan["prior_run"], plan["native_template"], base.initial_latent,
                              *(row["config"] for row in plan["cases"])):
                verify_asset(reference)
            report["engineering_passed"] = all(row["status"] == "passed" for row in report["cases"])
            report["matched_initial_z"] = (all(row.get("initial_z_sha256") for row in report["cases"])
                and len({row["initial_z_sha256"] for row in report["cases"]}) == 1)
            report["engineering_passed"] &= report["matched_initial_z"]
            report.update(status="passed" if report["engineering_passed"] else "failed", seconds=time.monotonic() - started,
                followups_completed=all(row.get(key, {}).get("status") == "completed"
                    for row in report["cases"] for key in ("repair_audit", "fresh_pose")))
            write_json(output, report)
            experiment.annotate(engineering_passed=report["engineering_passed"], full_default_validated=False,
                                followups_completed=report["followups_completed"])
            if not report["engineering_passed"]:
                raise RuntimeError("Matched pilot failed; retained pair is not a full-run certificate")
        except Exception as error:
            report.update(status="failed", engineering_passed=False, error=str(error), error_type=type(error).__name__)
            write_json(output, report)
            raise
    return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        load_plan(args.plan, args.sha256)
        print("Authenticated paired configs; no native model execution")
    else:
        print(run(args.plan, args.sha256))


if __name__ == "__main__":
    main()
