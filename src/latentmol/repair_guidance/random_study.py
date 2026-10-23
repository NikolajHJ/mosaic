"""Serial saved-random-start study with a late increasing frozen repair prior.

Preparation authenticates historical starts without screening or drawing new z.
Execution first requires real native repair and derivative evidence, then runs
the declared cohort in one allocation. All failures and allocation exclusions
remain rows. Changing targets/weights make total loss values incomparable across
states; report binding, chemistry, schedule and target jumps separately instead.
"""

from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import sys
import time

import yaml

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.stability_study import verify_asset, verified_child
from latentmol.boltz.workflow_process import run_logged
from latentmol.config_identity import verify_config_identity
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from .config import RepairGuidanceConfig
from .study import guidance_evidence, require_hpc


def read(path):
    return json.loads(Path(path).read_text())


def _reference(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def build_case(base, guidance, start, output, *, gate=False):
    """Copy scientific inputs; only declared seed, budgets and lambda differ.

    A saved z is mandatory. The carbon-chain YAML is solely the authenticated
    protein/MSA template for fresh rescoring, never molecular initialization.
    """
    parent = read(verify_asset(start["parent_record"]))
    verify_config_identity(parent, allow_legacy_active_set_contexts=True)
    verify_asset(start["initial_latent"])
    verify_asset(start["native_template"])
    original = parent["config"]
    for key in ("mode", "decoder", "affinity_checkpoint", "context_manifest", "temperature", "threshold"):
        if original.get(key) != base.record().get(key):
            raise ValueError(f"Saved random start changes shared scientific input {key}")
    if original.get("start", {}).get("kind") != "standard_normal":
        raise ValueError("Cohort must retain unfiltered standard-normal initialization")
    seed = int(start["id"].removeprefix("prior-"))
    if original["start"].get("seed") != seed:
        raise ValueError("Historical seed differs from declared cohort ID")
    raw = base.record()
    raw.pop("context_identity", None)
    raw.pop("contexts", None)
    raw.update(version=1, name=("native-gate-" if gate else "ramped-") + start["id"],
               output_root=str(output), initial_latent=deepcopy(start["initial_latent"]),
               start=deepcopy(original["start"]), gradient_audit_interval=1 if gate else 50)
    raw["optimizer"]["steps"] = 6 if gate else 100
    raw["diagnostics"]["hard_samples"] = 1 if gate else 8
    repair = guidance.record()
    repair.update(refresh_interval=5 if gate else 10, attempts_per_refresh=4 if gate else 2,
                  seed=61000 + seed)
    raw["repair_guidance"] = repair
    return raw


def prepare(base_ref, starts_ref, guidance_ref, output, campaign_root):
    """HPC metadata preparation plus config parsing, no model selection.

    Exactly prior-0 through prior-9 from the authenticated source063 protocol
    are retained in that order, including eventual incompatible/invalid starts.
    Native gate reuses prior-0 but its six updates are a separate engineering
    trajectory, never a warm start substituted for the long study's saved z.
    """
    require_hpc()
    base = ActiveSetConfig.load(verify_asset(base_ref))
    if (base.arms != ("fixed_initial_mask",) or base.fd_policy != "local_v1"
            or base.step_control is None or base.dbmol is None or base.initial_latent is None
            or base.repair_guidance is not None or base.objective.output != "dbmol_total_loss"):
        raise ValueError("Require saved-z, bounded fixed-mask full DBMol base with no prior repair")
    starts_source = read(verify_asset(starts_ref))
    starts = [s for s in starts_source["starts"] if s["id"].startswith("prior-")]
    if [s["id"] for s in starts] != [f"prior-{i}" for i in range(10)]:
        raise ValueError("Require all ten saved random starts, unscreened and in seed order")
    guidance_path = verify_asset(guidance_ref)
    guidance = RepairGuidanceConfig.from_mapping(yaml.safe_load(guidance_path.read_text()), guidance_path.parent)
    if guidance.weight_schedule is None or guidance.provider_activation_weight <= 0:
        raise ValueError("Declare a positive increasing repair weight schedule")
    output = Path(output).resolve()
    campaign_root = Path(campaign_root).resolve()
    raw_cases = [build_case(base, guidance, s, campaign_root / s["id"] / "optimization") for s in starts]
    raw_gate = build_case(base, guidance, starts[0], campaign_root / "native-gate" / "optimization", gate=True)
    output.mkdir(parents=True, exist_ok=False)
    def save(raw, start):
        path = output / (raw["name"] + ".yaml")
        with path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle, sort_keys=False)
        cfg = ActiveSetConfig.load(path)
        return {"id": start["id"], "config": _reference(path), "steps": cfg.optimizer.steps,
                "parent_record": deepcopy(start["parent_record"]),
                "initial_latent": deepcopy(start["initial_latent"]),
                "native_template": deepcopy(start["native_template"])}
    protocol = {"version": 1, "name": "ramped-defog-random", "output_root": str(campaign_root),
        "base": base_ref, "starts_source": starts_ref, "guidance": guidance_ref,
        "gate": save(raw_gate, starts[0]), "cases": [save(c, s) for c, s in zip(raw_cases, starts)],
        "cache": "/work3/s234818/latentmol/cache/boltz2", "allocation_seconds": 79200,
        "gate_timeout_seconds": 7200, "optimization_timeout_seconds": 6000,
        "rescore_timeout_seconds": 900,
        "source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
        "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"],
        "claim_limit": "Prepared engineering protocol, not native validation or evidence of binding",
        "comparability": "Historical affinity-only runs are descriptive references, not matched controls for full DBMol plus ramp"}
    write_json(output / "protocol.json", protocol)
    return output / "protocol.json"


def load_protocol(path, expected_sha):
    """Authenticate prescribed membership and every config before execution."""
    protocol = read(verify_asset({"path": str(Path(path).resolve()), "sha256": expected_sha}))
    fields = {"version", "name", "output_root", "base", "starts_source", "guidance", "gate", "cases", "cache",
        "allocation_seconds", "gate_timeout_seconds", "optimization_timeout_seconds", "rescore_timeout_seconds",
        "source_manifest_sha256", "source_archive_sha256", "claim_limit", "comparability"}
    if type(protocol.get("version")) is not int or protocol["version"] != 1 or set(protocol) != fields:
        raise ValueError("Require exact ramped-random version-1 protocol")
    if [c["id"] for c in protocol["cases"]] != [f"prior-{i}" for i in range(10)]:
        raise ValueError("Random cohort membership/order changed")
    if protocol["gate"]["id"] != "prior-0":
        raise ValueError("Native gate must use the first exact unscreened start")
    for key in ("output_root", "cache"):
        if not isinstance(protocol[key], str) or not Path(protocol[key]).is_absolute():
            raise ValueError(f"Require absolute {key}")
    for key in ("allocation_seconds", "gate_timeout_seconds", "optimization_timeout_seconds", "rescore_timeout_seconds"):
        if type(protocol[key]) is not int or not 60 <= protocol[key] <= 79200:
            raise ValueError(f"Invalid declared {key}")
    for key in ("base", "starts_source", "guidance"):
        verify_asset(protocol[key])
    output_roots = set()
    for index, case in enumerate([protocol["gate"], *protocol["cases"]]):
        if set(case) != {"id", "config", "steps", "parent_record", "initial_latent", "native_template"}:
            raise ValueError("Undeclared case fields")
        cfg = ActiveSetConfig.load(verify_asset(case["config"]))
        for key in ("parent_record", "initial_latent", "native_template"):
            verify_asset(case[key])
        parent = read(case["parent_record"]["path"])
        verify_config_identity(parent, allow_legacy_active_set_contexts=True)
        expected_steps = 6 if index == 0 else 100
        expected_seed = 0 if index == 0 else index - 1
        if (case["steps"] != expected_steps or cfg.optimizer.steps != expected_steps
                or cfg.arms != ("fixed_initial_mask",) or cfg.dbmol is None or cfg.step_control is None
                or cfg.repair_guidance is None or cfg.repair_guidance.weight_schedule is None
                or cfg.start.kind != "standard_normal" or cfg.start.seed != expected_seed
                or cfg.initial_latent != case["initial_latent"]):
            raise ValueError("Prescribed schedule/start/budget differs from config")
        root = cfg.output_root.resolve()
        if not root.is_relative_to(Path(protocol["output_root"]).resolve()) or root in output_roots:
            raise ValueError("Optimization paths must be distinct and inside the study root")
        output_roots.add(root)
    return protocol


def trajectory_report(child, summary):
    """Descriptive per-state traces, without mislabeling objective jumps as drift.

    Raw chemistry diagnostics use the existing argmax and prescribed samples.
    Neither graph repair nor fresh native standardization silently replaces a
    raw output in these fields. Bond/atom diagnostics remain slot-aligned.
    """
    arm = summary["arms"]["fixed_initial_mask"]
    folder = Path(child) / "fixed_initial_mask"
    metrics = [read(p) for p in sorted(folder.glob("metrics-*.json"))]
    if not metrics or [r["step"] for r in metrics] != list(range(metrics[-1]["step"] + 1)):
        raise ValueError("Missing trajectory state receipts")
    def trace(row):
        score = row["score"]
        repair = score.get("repair_guidance", {})
        # New rows state the applied coefficient directly. Read-only summaries
        # of older rows derive it from their explicit active/no-target status;
        # scheduled ``effective_weight`` remains unchanged for comparability.
        scheduled_weight = repair.get("scheduled_weight", repair.get("weight"))
        used_target = repair.get("used_target", repair.get("status") == "active")
        applied_weight = repair.get("applied_weight", scheduled_weight if used_target else 0.)
        return {"step": row["step"], "accepted": row.get("update_status") == "accepted",
            "effective_weight": repair.get("weight"), "total_loss": score["loss"],
            "scheduled_weight": scheduled_weight, "applied_weight": applied_weight,
            "target_available": repair.get("target_available", used_target), "used_target": used_target,
            "guidance_active": repair.get("guidance_active", used_target and applied_weight is not None and applied_weight > 0),
            "gradient_measured": repair.get("gradient_measured",
                isinstance(repair.get("gradient_diagnostics"), dict) and "repair_norm" in repair["gradient_diagnostics"]),
            "repair_application_source": "explicit" if "applied_weight" in repair else "derived_from_legacy_status",
            "binding_loss": score.get("binding_loss"), "binding_components": score.get("components"),
            "repair_target_sha256": repair.get("target_sha256"),
            "repair_nll": repair.get("normalized_nll"), "raw_repair_nll": repair.get("raw_nll"),
            "weighted_repair_loss": repair.get("weighted_loss", 0. if not used_target else None),
            "repair_status": repair.get("status"), "gradient_diagnostics": repair.get("gradient_diagnostics"),
            "count": row["selection"]["count"], "natural_count": row["natural_selection"]["count"],
            "raw_argmax_smiles": row["argmax_smiles"], "sampled_and_argmax_valid": row["valid_candidates"],
            "candidate_count": row["candidate_count"], "latent_displacement": row["latent_displacement"],
            "elements": score.get("elements"), "oracle_cost_at_state": row.get("oracle_cost_at_state")}
    return {"optimization_status": arm["optimization_status"], "accepted_updates": arm["accepted_updates"],
        "requested_steps": arm["requested_steps"], "completed_requested_budget": arm["completed_requested_budget"],
        "raw_argmax_valid_states": sum(r["argmax_smiles"] is not None for r in metrics),
        "prescribed_candidate_count": sum(r["candidate_count"] for r in metrics),
        "valid_candidate_count": sum(r["valid_candidates"] for r in metrics),
        "start_smiles": metrics[0]["argmax_smiles"], "endpoint_smiles": metrics[-1]["argmax_smiles"],
        "fixed_counts": sorted({r["selection"]["count"] for r in metrics}),
        "natural_counts": [r["natural_selection"]["count"] for r in metrics],
        "weight_jump_receipts": [read(p) for p in sorted(folder.glob("repair-weight-*.json"))],
        "target_refresh_receipts": [read(p) for p in sorted(folder.glob("repair-refresh-*/refresh.json"))],
        "target_objective_jump_receipts": [read(p) for p in sorted(folder.glob("repair-refresh-objective-*.json"))],
        "oracle_cost": summary["oracle_cost"], "trace": [trace(r) for r in metrics],
        "claim_limit": "Changing weight/target invalidates cross-state total-loss monotonicity; fixed-pose model scores and raw graph diagnostics only"}


def _optimize(case, parent, label, identity, timeout):
    cfg_path = verify_asset(case["config"])
    cfg = ActiveSetConfig.load(cfg_path)
    execution = run_logged([sys.executable, "-m", "latentmol.boltz.active_set", "--config", cfg_path],
                           parent / (label + ".log"), timeout_seconds=timeout)
    write_json(parent / (label + "-execution.json"), execution)
    child = verified_child(execution, cfg.output_root, identity, config_file_sha256=case["config"]["sha256"])
    summary = read(child / "summary.json")
    if summary.get("frozen_state_unchanged") is not True:
        raise ValueError("Optimization lacks frozen-state certification")
    # Eligibility uses the positive scheduled endpoint, not base/start weight.
    evidence = guidance_evidence(child, summary, weight=cfg.repair_guidance.provider_activation_weight)
    report = trajectory_report(child, summary)
    write_json(parent / (label + "-trajectory.json"), report)
    return {"status": "completed" if report["completed_requested_budget"] else "stalled",
        "execution": execution, "child_run": str(child), "summary": _reference(child / "summary.json"),
        "trajectory": _reference(parent / (label + "-trajectory.json")), "repair_evidence": evidence,
        "accepted_updates": report["accepted_updates"], "oracle_cost": summary["oracle_cost"]}


def run(protocol_path, expected_sha):
    """One native gate and ten serial children; never retry/replace a failed seed.

    The parent holds no GPU models. Children terminate before native fresh-pose
    rescoring and subsequent seeds. Full declared per-case timeout is reserved
    at allocation boundaries; unattempted cases are explicit, not failures or
    completions. A future allocation can use retained protocol/assets, but is
    never queued automatically by this controller.
    """
    require_hpc()
    p = load_protocol(protocol_path, expected_sha)
    identity = {"source_manifest_sha256": source_manifest(Path.cwd())["sha256"],
                "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"]}
    if any(p[key] != value for key, value in identity.items()):
        raise ValueError("Prepared protocol belongs to another immutable release")
    cfg = {"protocol": str(protocol_path), "protocol_sha256": expected_sha, **identity,
           "scope": "Serial fixed-occupancy full DBMol plus detached DeFoG-target ramp; native gate required"}
    started = time.monotonic()
    with ExperimentRun(Path(p["output_root"]) / "controller", p["name"], cfg, Path.cwd()) as experiment:
        rows = [{"id": c["id"], "status": "not_attempted"} for c in p["cases"]]
        report = {"native_gate": {"status": "not_attempted"}, "cases": rows,
                  "native_gate_passed": False, "completed_requested_cohort": False,
                  "comparability": p["comparability"]}
        path = experiment.path / "campaign.json"
        write_json(path, report)
        try:
            report["native_gate"] = _optimize(p["gate"], experiment.path, "native-gate", identity, p["gate_timeout_seconds"])
            gate = report["native_gate"]
            report["native_gate_passed"] = (gate["status"] == "completed"
                and gate["repair_evidence"]["repair_validation_passed"])
        except Exception as exc:
            report["native_gate"].update(status="failed", error=str(exc), error_type=type(exc).__name__)
        write_json(path, report)
        if not report["native_gate_passed"]:
            for row in rows:
                row["reason"] = "blocked_by_native_guidance_gate"
            write_json(path, report)
            raise RuntimeError("Native random-start repair/FD/accepted-update gate failed; long cohort not launched")
        for index, (case, row) in enumerate(zip(p["cases"], rows)):
            remaining = p["allocation_seconds"] - (time.monotonic() - started)
            budget = p["optimization_timeout_seconds"] + p["rescore_timeout_seconds"] + 60
            if remaining < budget:
                for unattempted in rows[index:]:
                    unattempted["reason"] = "allocation_boundary_insufficient_declared_case_budget"
                break
            row["status"] = "running"
            write_json(path, report)
            try:
                row.update(_optimize(case, experiment.path, case["id"], identity, p["optimization_timeout_seconds"]))
                child = Path(row["child_run"])
                # A stalled but fully recorded trajectory has a real endpoint.
                # Chemistry validity gates determine which raw endpoints can be
                # rescored; repaired targets are never substituted here.
                try:
                    rescore_root = Path(p["output_root"]) / case["id"] / "fresh-pose"
                    execution = run_logged([sys.executable, "-m", "latentmol.boltz.active_set_rescore",
                        "--pilot", child, "--template", verify_asset(case["native_template"]), "--cache", p["cache"],
                        "--run-root", rescore_root], experiment.path / (case["id"] + "-rescore.log"),
                        timeout_seconds=p["rescore_timeout_seconds"])
                    write_json(experiment.path / (case["id"] + "-rescore-execution.json"), execution)
                    native = verified_child(execution, rescore_root, identity, parent_run_id=read(child / "run.json")["run_id"])
                    row["fresh_pose"] = {"status": "completed", "execution": execution,
                        "run": str(native), "summary": _reference(native / "summary.json")}
                except Exception as exc:
                    row["fresh_pose"] = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}
            except Exception as exc:
                row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            write_json(path, report)
        report["completed_requested_cohort"] = all(row["status"] == "completed" for row in rows)
        report["seconds"] = time.monotonic() - started
        write_json(path, report)
        experiment.annotate(native_gate_passed=True, completed_requested_cohort=report["completed_requested_cohort"],
            terminal_status_counts={status: sum(r["status"] == status for r in rows)
                                    for status in ("completed", "stalled", "failed", "not_attempted")})
    return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("prepare")
    for key in ("base", "starts", "guidance"):
        a.add_argument("--" + key, type=Path, required=True)
        a.add_argument("--" + key + "-sha256", required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--campaign-root", type=Path, required=True)
    b = commands.add_parser("run")
    b.add_argument("--protocol", type=Path, required=True)
    b.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        refs = [{"path": str(getattr(args, key).resolve()), "sha256": getattr(args, key + "_sha256")}
                for key in ("base", "starts", "guidance")]
        print(prepare(*refs, args.output, args.campaign_root))
    else:
        print(run(args.protocol, args.sha256))


if __name__ == "__main__":
    main()
