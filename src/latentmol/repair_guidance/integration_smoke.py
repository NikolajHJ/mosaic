"""Small, fail-closed native Boltz/DeFoG integration proof on DTU HPC.

Reuse the original prescribed prior-0 start, not a success-selected new start.
Only contact/scalar precision, audit cadence, name and output directory change
from the authenticated six-update gate. The unchanged full validator runs at
initial state, target refresh five and endpoint six. This is not approval of
any historical failed state or evidence of long-trajectory stability.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.stability_study import verify_asset
from latentmol.boltz.workflow_process import run_logged
from latentmol.experiments import ExperimentRun, source_manifest, write_json
from .random_study import _optimize
from .study import require_hpc


def validate_config(base, candidate):
    """Permit exactly the declared changes; retain frozen assets and guards."""
    original = base.record()
    if (base.optimizer.steps != 6 or base.gradient_audit_interval != 1
            or base.arms != ("fixed_initial_mask",) or base.fd_policy != "local_v1"
            or base.step_control is None or base.dbmol is None
            or base.start.kind != "standard_normal" or base.start.seed != 0
            or base.initial_latent is None or base.repair_guidance is None
            or base.repair_guidance.refresh_interval != 5
            or base.repair_guidance.attempts_per_refresh != 4
            or base.repair_guidance.weight_schedule is None
            or "charge_policy" in base.repair_guidance.provider):
        raise ValueError("Require the authenticated original six-update prior-0 gate")
    expected = deepcopy(original)
    expected["name"] = candidate.name
    expected["output_root"] = str(candidate.output_root)
    expected["gradient_audit_interval"] = 6
    expected["dbmol"]["objective_arithmetic"] = "contact_and_assembly_float64_v1"
    if candidate.record() != expected:
        raise ValueError("Smoke changes undeclared science, guard, start or repair settings")


def verified_forward_ad(item):
    """Reject stale aggregate pass flags or partial/mismatched directional rows."""
    from latentmol.boltz.additive_contract import verify_forward_ad
    return verify_forward_ad(item)


def verified_additive(item, mode, profile, component):
    """Recompute the actual same-endpoint certificate, rejecting stale flags.

    The first native v3 scope retains the original 5%/1e-6 tolerance and two
    neighboring scales. This is pure receipt validation; it does not run a
    model, construct a replacement gradient or approve the old net secants.
    """
    from latentmol.boltz.additive_contract import verify_item
    return verify_item(item, mode, profile, component)


def _full_gate_passed(gate, *, require_repair=True, expected_profile=None):
    """Read each explicitly versioned native derivative validation contract.

    A Boltz-only control has no repair derivative to certify. Stationary DBMol
    components retain their measured signal status. Historical FD/v4 contracts
    retain their qualification rules; v5 can certify attached low-signal local
    derivatives without claiming an informative optimization or guidance step.
    """
    components = gate.get("component_gradients", {})
    repair = gate.get("repair_only_validation", {})
    contract = gate.get("derivative_contract")
    if contract is not None and contract.get("version") in ("pointwise_native_ad_v4", "pointwise_native_ad_v5"):
        from latentmol.boltz.pointwise_contract import verify_gate
        return verify_gate(gate, require_repair=require_repair, expected_profile=expected_profile)
    if contract is not None and contract.get("version") == "additive_branch_v3":
        from latentmol.boltz.additive_contract import profile_record, verify_repair
        profile = contract.get("profile", {})
        try:
            if (profile_record(profile["dbmol"], profile["objective"], repair_weight=profile["repair_weight"]) != profile
                    or expected_profile is None or expected_profile != profile
                    or (profile["repair_weight"] is not None) != require_repair):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        rows = {**components, "combined": gate.get("local_fd_validation", {})}
        latent_shape = contract.get("latent_shape")
        if (not isinstance(latent_shape, list) or not latent_shape
                or any(type(value) is not int or value <= 0 for value in latent_shape)):
            return False
        measured = gate["measured"]
        if (gate["local_fd_validation"]["baseline_loss"] != measured["loss"]
                or any(components[key]["baseline_loss"] != measured["components"][key] for key in components)):
            return False
        parity = gate.get("local_branch_baseline_parity", {})
        component_names = {"dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss"}
        names = component_names | {"combined"}
        exact_maps = {"component_gradient_exact": component_names,
            "supplied_native_gradient_exact": names, "native_grouped_gradient_exact": names,
            "native_grouped_scalar_exact": names,
            "held_gradient_exact": names | ({"repair_only"} if require_repair else set())}
        if (parity.get("scalar_exact") is not True or parity.get("total_gradient_exact") is not True
                or any(set(parity.get(field, {})) != expected
                    or any(value is not True for value in parity[field].values())
                    for field, expected in exact_maps.items())):
            return False
        relus = parity["captured_readout_relus"]
        relu_names = {f"head.{member}.{field}.{index}" for member in (0, 1)
            for field in ("affinity_out_mlp", "to_affinity_pred_value", "to_affinity_pred_score") for index in (1, 3)}
        if set(relus) != relu_names:
            return False
        for row in relus.values():
            shape = row["shape"]
            minimum = row["minimum_absolute_preactivation"]
            zeros = row["exact_zero_units"]
            if (type(shape) is not list or not shape or any(type(size) is not int or size < 1 for size in shape)
                    or type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum < 0
                    or type(zeros) is not int or not 0 <= zeros <= math.prod(shape)
                    or (minimum == 0) != (zeros > 0)):
                return False
        if require_repair:
            actual_repair = measured["repair_guidance"]
            if (actual_repair.get("status") != "active" or actual_repair.get("weight") != profile["repair_weight"]
                    or repair.get("baseline_loss") != actual_repair.get("normalized_nll")
                    or ("effective_repair_weight" in gate and gate["effective_repair_weight"] != profile["repair_weight"])):
                return False
        elif repair:
            return False
        for item in rows.values():
            if (item.get("attached") is not True or item.get("fixed_pose_and_mask") is not True
                    or item.get("direction_selection", {}).get("latent_numel") != math.prod(latent_shape)):
                return False
            for mode_inputs in item.get("additive_inputs", {}).values():
                if any(row.get("recomposition", {}).get("rows", {}).get("reverse", {}).get("native", {}).get("shape")
                       != latent_shape for row in mode_inputs):
                    return False
        return (gate.get("passed") is True and gate.get("status") == "passed"
            and gate.get("native_seed_parity") is True and contract.get("passed") is True
            and contract.get("branch_holding_used_in_optimization") is False
            and contract.get("historical_gate_approval") is False
            and contract.get("natural_contact_primitive_certificate_required") is True
            and gate.get("local_branch_baseline_parity", {}).get("passed") is True
            and set(components) == {"dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss"}
            and all(row.get("passed") is True and row.get("status") == "validated_nonzero"
                    and verified_forward_ad(row) and verified_additive(row, "held", profile, key)
                    and set(row.get("original_net_fd", {})) == {"natural", "held"} for key, row in rows.items())
            and verified_additive(components["dbmol_contact_loss"], "natural", profile, "dbmol_contact_loss")
            and (not require_repair or verify_repair(repair)))
    if contract is not None:
        # A new branch contract needs its own complete evidence; an old passed
        # local_v1 flag or a bare version label cannot approve it.
        rows = [*components.values(), gate.get("local_fd_validation", {})]
        if require_repair: rows.append(repair)
        if (contract.get("version") != "branch_local_v2" or contract.get("passed") is not True
                or contract.get("branch_holding_used_in_optimization") is not False
                or gate.get("local_branch_baseline_parity", {}).get("passed") is not True
                or components.get("dbmol_contact_loss", {}).get("natural_secants", {}).get("passed") is not True
                or any(row.get("status") != "zero_at_current_point_uninformative" and
                       (not verified_forward_ad(row)
                        or row.get("local_branch_fd", {}).get("passed") is not True)
                       for row in rows)):
            return False
    return (gate.get("passed") is True and gate.get("status") == "passed"
        and gate.get("native_seed_parity") is True
        and gate.get("local_fd_validation", {}).get("passed") is True
        and (not require_repair or (repair.get("passed") is True
             and repair.get("status") == "validated_nonzero"))
        and set(components) == {"dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss"}
        and all(row.get("passed") is True or row.get("status") == "zero_at_current_point_uninformative"
                for row in components.values()))


def full_gate_passed(gate, *, require_repair=True, expected_profile=None):
    """Malformed or incomplete receipts reject cleanly; no reader exception approves.

    V3 additionally requires coefficients independently derived from the sealed
    caller configuration. A coherent receipt cannot authenticate its own model
    profile. Historical v2/local-v1 acceptance semantics are otherwise retained.
    """
    try:
        if type(require_repair) is not bool:
            return False
        return _full_gate_passed(gate, require_repair=require_repair, expected_profile=expected_profile)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def validate_outcome(result, trajectory, gates):
    """Fallback completion is insufficient; both branches must guide all updates.

    Pointwise/full FD gates remain the optimizer's responsibility. This reader
    additionally requires their saved success receipts and actual native repair
    evidence, with no ranking of outputs or substitution of repaired molecules.
    """
    evidence = result.get("repair_evidence", {})
    trace = trajectory.get("trace", [])
    updates = [row for row in trace if row.get("accepted") is True]
    checks = {
        "completed_six_updates": (result.get("status") == "completed"
            and result.get("accepted_updates") == 6
            and trajectory.get("completed_requested_budget") is True
            and trajectory.get("requested_steps") == 6),
        "complete_state_receipts": [row.get("step") for row in trace] == list(range(7)),
        "fixed_occupancy": trajectory.get("fixed_counts") == [22],
        "two_native_targets": (evidence.get("accepted_native_targets", 0) >= 2
            and evidence.get("accepted_targets", 0) == 2
            and evidence.get("refreshes", 0) == 2),
        "actual_guidance": (evidence.get("repair_validation_passed") is True
            and evidence.get("guided_accepted_updates") == 6
            and evidence.get("nonzero_guided_updates") == 6
            and evidence.get("target_only_fd_passes") == 3),
        "full_guards": (set(gates) == {"step-0000", "step-0005", "step-0006"}
            and all(full_gate_passed(gate) for gate in gates.values())),
        "both_gradients_reach_z": len(updates) == 6 and all(
            row.get("used_target") is True and row.get("gradient_measured") is True
            and all(isinstance(value, (float, int)) and math.isfinite(value) and value > 1e-12
                for value in (row.get("gradient_diagnostics", {}).get("binding_norm"),
                              row.get("gradient_diagnostics", {}).get("weighted_repair_norm")))
            for row in updates),
    }
    return {"passed": all(checks.values()), "checks": checks,
        "claim_limit": "Short native integration only; raw validity, binding efficacy and long stability are not established"}


def run(plan_path, expected_sha):
    """One MPS probe and one native child; retain all failures, never retry."""
    require_hpc()
    plan = json.loads(verify_asset({"path": str(plan_path.resolve()), "sha256": expected_sha}).read_text())
    fields = {"version", "name", "output_root", "parent_config", "config",
              "child_timeout_seconds", "source_manifest_sha256", "source_archive_sha256"}
    if (set(plan) != fields or type(plan["version"]) is not int or plan["version"] != 1
            or plan["child_timeout_seconds"] != 6600):
        raise ValueError("Require the declared two-hour/6600-second smoke protocol")
    identity = {"source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
                "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"]}
    if any(plan[key] != value for key, value in identity.items()):
        raise ValueError("Smoke belongs to another sealed source release")
    base = ActiveSetConfig.load(verify_asset(plan["parent_config"]))
    config = ActiveSetConfig.load(verify_asset(plan["config"]))
    validate_config(base, config)
    raw_root = Path(plan["output_root"])
    root = raw_root.resolve()
    if not raw_root.is_absolute() or config.output_root.resolve() != root / "optimization":
        raise ValueError("Require the unique declared smoke output layout")
    with ExperimentRun(root / "controller", plan["name"],
                       {"plan": str(plan_path), "plan_sha256": expected_sha, **identity}, Path.cwd()) as experiment:
        report = {"status": "running", "passed": False, "fresh_pose": "not_run_engineering_scope"}
        output = experiment.path / "smoke.json"
        write_json(output, report)
        try:
            probe = experiment.path / "gpu-process.json"
            execution = run_logged([sys.executable, "-m", "latentmol.gpu_process_probe",
                "--worker-python", config.repair_guidance.provider["python"], "--output", str(probe)],
                experiment.path / "gpu-process.log", timeout_seconds=180)
            report["gpu_process_execution"] = execution
            write_json(output, report)
            if (execution["timed_out"] or execution["returncode"] != 0
                    or json.loads(probe.read_text()).get("status") != "passed"):
                raise RuntimeError("Allocated Boltz/DeFoG GPU coexistence failed")
            result = _optimize({"config": plan["config"]}, experiment.path, "native-smoke", identity,
                               plan["child_timeout_seconds"])
            trajectory = json.loads(Path(result["trajectory"]["path"]).read_text())
            child = Path(result["child_run"])
            gates = json.loads((child / "fixed_initial_mask/gates.json").read_text())
            # Substantiate unchanged external parent/config/start bytes after
            # execution, in addition to the child's frozen model/context proof.
            for reference in (plan["parent_config"], plan["config"], config.initial_latent):
                verify_asset(reference)
            report.update(result=result, outcome=validate_outcome(result, trajectory, gates))
            report["passed"] = report["outcome"]["passed"]
            report["status"] = "passed" if report["passed"] else "failed"
            write_json(output, report)
            if not report["passed"]:
                raise RuntimeError("Native smoke lacks complete same-z binding and repair evidence")
            experiment.annotate(native_integration_smoke_passed=True, long_stability_certified=False)
        except Exception as error:
            report.update(status="failed", passed=False, error=str(error), error_type=type(error).__name__)
            write_json(output, report)
            experiment.annotate(native_integration_smoke_passed=False, long_stability_certified=False)
            raise
    return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(run(args.plan, args.sha256))


if __name__ == "__main__":
    main()
