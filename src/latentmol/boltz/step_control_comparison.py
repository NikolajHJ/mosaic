"""Matched one-step comparisons at authenticated, derivative-validated states.

The old clipped update and the opt-in backtracking update share the same saved
latent, original start/radius, gradient, decoder, objective, mask and native pose.
Only update acceptance differs. This diagnostic never changes occupancy, resumes
an old experiment, or claims improvement in a freshly predicted molecular pose.
Run numerical evaluations only inside an immutable DTU HPC allocation.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys

import torch

from latentmol.config_identity import verify_config_identity
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.multimodel.engine import Counters
from latentmol.multimodel.study import module_digest, tensor_digest, verified_file
from latentmol.optimization import StepControl, NonfiniteObjective, backtracking_step, projected_proposal
from .active_set import ContextBank, Selection, make_branch, select_slots


def validation_evidence(path: Path, case: dict, archive_sha256: str) -> dict:
    """Require one passed source061-style audit of this exact original state.

    The archive identity is explicit, not inferred from a success-looking file.
    Config integrity, parent identity, frozen-state checks and each required
    direction must agree. Failed/unfinished audits are retained and withheld.
    """
    record_path = path.parent / "run.json"
    record = json.loads(record_path.read_text())
    report = json.loads(path.read_text())
    identity = verify_config_identity(record)
    if (record.get("status") != "completed" or report.get("status") != "completed"
            or record.get("source_archive_sha256") != archive_sha256):
        raise ValueError("Derivative audit is incomplete or has a different immutable source")
    cfg = record["config"]
    if (cfg.get("parent_run") != str(Path(case["parent_run"]).resolve())
            or cfg.get("parent_arm") != case["arm"] or cfg.get("parent_step") != case["step"]
            or cfg.get("production_release") != str(Path(case["production_release"]).resolve())):
        raise ValueError("Derivative audit belongs to a different saved state")
    local = report.get("local_fd", {})
    directions = local.get("direction_results", {})
    if (report.get("validation_passed") is not True or report.get("parent_unchanged") is not True
            or report.get("frozen_state_unchanged") is not True
            or report.get("repeatability_passed") is not True
            or local.get("passed") is not True or local.get("status") != "passed"
            or local.get("policy", {}).get("version") != "local_v1"
            or len(directions) != 4 or not all(row.get("passed") is True for row in directions.values())):
        raise ValueError("Derivative audit did not pass all required local validation checks")
    native_parity = report.get("native_seed_parity", {})
    if not native_parity or not all(value is True for value in native_parity.values()):
        raise ValueError("Derivative audit native parity did not pass")
    if report.get("identities_before") != report.get("identities_after"):
        raise ValueError("Derivative audit scientific identities changed")
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "run_path": str(record_path.resolve()), "run_sha256": file_sha256(record_path),
            "source_archive_sha256": archive_sha256, "config_identity": identity,
            "parent_artifact_sha256": cfg["parent_artifact_sha256"],
            "scientific_identities": report["identities_before"], "runtime": report["runtime"],
            "validated_gradient": report["repeats"][2]["gradient"]}


def locate_validation(root: Path, case: dict, archive_sha256: str) -> dict:
    """Never choose a favorable audit from multiple attempts after seeing results."""
    matches = list(root.glob(f"*/{case['id']}/*/validation.json"))
    if len(matches) != 1:
        raise ValueError(f"Require exactly one declared derivative audit for {case['id']}; found {len(matches)}")
    return validation_evidence(matches[0], case, archive_sha256)


def run_saved(parent: Path, run_root: Path, *, production_root: Path, arm: str,
              step: int, validation_path: Path, validation_archive_sha256: str):
    """Evaluate one old proposal and bounded backtracking from the same point."""
    from latentmol.evaluate import load_model
    from .model import FrozenBoltzAffinity, load_affinity_model
    from . import _saved_gate_diagnostic as diagnostic

    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require DTU allocation and immutable step-comparison source")
    parent, production_root = parent.resolve(), production_root.resolve()
    case = {"parent_run": str(parent), "production_release": str(production_root), "arm": arm, "step": step}
    evidence = validation_evidence(validation_path, case, validation_archive_sha256)
    record, old_gate, config, state, hashes = diagnostic._parent_inputs(
        parent, arm=arm, step=step, production_root=production_root)
    if hashes != evidence["parent_artifact_sha256"]:
        raise ValueError("Parent artifacts differ from the derivative-validated state")
    control = StepControl(version="armijo_v1")
    inputs = {"case": case, "original_config": config.record(), "original_gate": old_gate,
              "parent_artifact_sha256": hashes, "gradient_validation": evidence,
              "step_control": asdict(control), "comparison_steps": 1,
              "maximum_native_forwards": control.max_trials + 4,
              "maximum_backwards": 1, "occupancy": "fixed saved mask in both arms",
              "comparison_module_sha256": file_sha256(Path(__file__)),
              "claim_limit": "One-step fixed-pose proxy comparison, not full optimization or fresh-native affinity."}
    # Original ExperimentRun code predates canonical config-key normalization.
    inputs = json.loads(json.dumps(inputs, allow_nan=False))
    with ExperimentRun(run_root, "boltz-step-control-comparison", inputs, Path.cwd()) as experiment:
        report = {"status": "running", "comparison_completed": False,
                  "original_experiment_not_resumed": True, "occupancy_changed": False,
                  "gradient_validation": evidence, "step_control": asdict(control)}
        def persist():
            write_json(experiment.path / "comparison.json", report)
        counters = Counters(lambda value: write_json(experiment.path / "oracle_cost.json", value))
        persist()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            decoder, training = load_model(verified_file(config.decoder["path"], config.decoder["sha256"]), config.device)
            if training["train_config"]["mode"] != config.mode:
                raise ValueError("Saved decoder mode differs")
            decoder.eval().requires_grad_(False)
            bank = ContextBank(config, decoder.spec)
            point = state["z"].to(config.device)
            start = torch.load(parent / "start.pt", map_location="cpu", weights_only=True)["z"].to(point)
            selection = Selection(state["active"].to(config.device), state["no_atom"].to(config.device), config.threshold)
            scorer = FrozenBoltzAffinity(load_affinity_model(verified_file(
                config.affinity_checkpoint["path"], config.affinity_checkpoint["sha256"])).to(config.device),
                config.recycling_steps)
            branch = make_branch(scorer, bank, selection, config, record["source"]["sha256"])
            identities = {"decoder_state_sha256": module_digest(decoder), "scorer_state_sha256": module_digest(scorer),
                          "context_sha256": branch.verify_context_unchanged(), "point_sha256": tensor_digest({"z": point})}
            if identities != evidence["scientific_identities"]:
                raise ValueError("Scientific state differs from the passed derivative audit")
            runtime = diagnostic.runtime_description(point.device)
            if runtime.get("cuda_device_name") != evidence["runtime"].get("cuda_device_name"):
                raise ValueError("Comparison GPU family differs from the derivative audit")
            report.update(identities_before=identities, runtime=runtime,
                          activation_checkpoint_policy=scorer.activation_checkpoint_policy)
            with torch.no_grad():
                native = counters.call("boltz", "step_comparison_parity", "native_forward", lambda:
                    scorer.discrete_reference(branch.prepared.features, branch.prepared.coordinates), point.device)
                adapted = counters.call("boltz", "step_comparison_parity", "forward", lambda:
                    branch(branch.prepared.seed.to_relaxed(decoder.spec)), point.device)
            if set(native) != set(adapted):
                raise ValueError("Native and adapted affinity heads differ")
            for key in native:
                torch.testing.assert_close(adapted[key], native[key], rtol=config.diagnostics.parity_rtol,
                                           atol=config.diagnostics.parity_atol)
                if not bool(native[key].isfinite().all()) or not bool(adapted[key].isfinite().all()):
                    raise ValueError("Nonfinite native parity")
            report["native_seed_parity"] = True
            no_grad, _ = diagnostic._probe(decoder, point, branch, config, counters, grad_enabled=False)
            analytic, gradient = diagnostic._probe(decoder, point, branch, config, counters,
                                                   grad_enabled=True, backward=True)
            report.update(baseline_no_grad=no_grad, baseline_analytic=analytic)
            report["current_gradient_vs_cached_audit"] = {
                "sha256_equal": analytic["gradient"]["sha256"] == evidence["validated_gradient"]["sha256"],
                "current_norm": analytic["gradient"]["norm_float64"],
                "validated_norm": evidence["validated_gradient"]["norm_float64"],
                "same_gpu_family_is_not_exact_arithmetic_certification": True}
            if no_grad["loss"]["value"] != analytic["loss"]["value"]:
                raise ValueError("Grad/no-grad baseline mismatch; no update comparison performed")
            baseline = analytic["loss"]["value"]
            torch.save({"z": point.cpu(), "start": start.cpu(), "gradient": gradient.cpu(),
                        "active": selection.active.cpu()}, experiment.path / "comparison-start.pt")
            cached, cache_hits = {}, 0
            def score(candidate):
                nonlocal cache_hits
                digest = tensor_digest({"z": candidate})
                if digest in cached:
                    cache_hits += 1
                    result = cached[digest]
                    if isinstance(result, NonfiniteObjective):
                        raise result
                    return result
                try:
                    measured, _ = diagnostic._probe(decoder, candidate, branch, config, counters,
                                                     grad_enabled=False, reject_nonfinite_objective=True)
                except NonfiniteObjective as exc:
                    # The diagnostic observes an actual scalar tensor before
                    # declaring it nonfinite; malformed native outputs propagate.
                    cached[digest] = exc
                    raise
                result = {"loss": measured["loss"]["value"], "loss_dtype": measured["loss"]["dtype"],
                          "natural_selector_crossed": measured["natural_selector_crossed"], "measured": measured}
                cached[digest] = result
                return result
            parameters = {"learning_rate": config.optimizer.learning_rate,
                          "max_gradient_norm": config.optimizer.max_gradient_norm,
                          "trust_radius": config.optimizer.trust_radius}
            legacy, old_metadata = projected_proposal(point, start, gradient, **parameters)
            torch.save({"z": legacy.cpu(), "active": selection.active.cpu()}, experiment.path / "legacy-proposal.pt")
            try:
                old_score = score(legacy)
                report["legacy"] = {"status": "evaluated", "proposal": old_metadata, **old_score,
                                    "actual_loss_improvement": baseline - old_score["loss"]}
            except NonfiniteObjective as exc:
                report["legacy"] = {"status": "nonfinite_objective", "proposal": old_metadata, "error": str(exc)}
            report["backtracking"] = {}
            persist()
            result = backtracking_step(point, start, gradient, baseline_loss=baseline,
                baseline_loss_dtype=analytic["loss"]["dtype"], baseline_variation=0.0,
                control=control, score=score, report=report["backtracking"], persist=persist, **parameters)
            torch.save({"z": result.point.cpu(), "active": selection.active.cpu(), "accepted": result.accepted},
                       experiment.path / "backtracking-result.pt")
            after = {"decoder_state_sha256": module_digest(decoder), "scorer_state_sha256": module_digest(scorer),
                     "context_sha256": branch.verify_context_unchanged(), "point_sha256": tensor_digest({"z": point})}
            unchanged = after == identities and all(file_sha256(Path(path)) == digest for path, digest in hashes.items())
            unchanged = unchanged and file_sha256(validation_path) == evidence["sha256"]
            unchanged = unchanged and file_sha256(Path(evidence["run_path"])) == evidence["run_sha256"]
            if not unchanged:
                raise ValueError("Step comparison changed frozen state or original evidence")
            report.update(status="completed", comparison_completed=True, identities_after=after,
                          frozen_state_unchanged=True, parent_and_audit_unchanged=True,
                          objective_cache_hits=cache_hits, objective_points_scored=len(cached),
                          backtracking_accepted=result.accepted)
            persist()
            experiment.annotate(comparison_completed=True, backtracking_accepted=result.accepted,
                                original_experiment_not_resumed=True)
            return experiment.path
        except Exception as exc:
            report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            persist()
            raise


def run_study(protocol_path: Path, validation_root: Path, validation_archive_sha256: str):
    """Run each declared one-step comparison serially; retain withheld cases."""
    from .gate_replay_study import load_protocol
    from .workflow_process import run_logged
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require DTU allocation and immutable comparison source")
    if not re.fullmatch(r"[0-9a-f]{64}", validation_archive_sha256):
        raise ValueError("Declare a 64-character derivative-validation archive SHA256")
    protocol = load_protocol(protocol_path)
    control = StepControl(version="armijo_v1")
    inputs = {**protocol, "protocol_file_sha256": file_sha256(protocol_path),
              "gradient_validation_root": str(validation_root.resolve()),
              "gradient_validation_archive_sha256": validation_archive_sha256,
              "step_control": asdict(control), "maximum_native_forwards_per_case": control.max_trials + 4,
              "maximum_child_seconds": len(protocol["cases"]) * protocol["timeout_seconds"],
              "comparison_steps": 1, "no_full_optimization": True}
    with ExperimentRun(Path(protocol["output_root"]), protocol["name"], inputs, Path.cwd()) as experiment:
        rows = []
        for case in protocol["cases"]:
            try:
                evidence = locate_validation(validation_root, case, validation_archive_sha256)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                rows.append({"case": case, "status": "withheld_derivative_validation",
                             "reason": str(exc), "comparison_completed": False, "execution": None})
            else:
                result = run_logged([sys.executable, str(Path.cwd() / "scripts/compare_saved_boltz_steps.py"),
                    "--parent-run", case["parent_run"], "--production-release", case["production_release"],
                    "--arm", case["arm"], "--step", str(case["step"]), "--run-root", str(experiment.path / case["id"]),
                    "--validation-report", evidence["path"], "--validation-archive-sha256", validation_archive_sha256],
                    experiment.path / f"{case['id']}.log", timeout_seconds=protocol["timeout_seconds"])
                completed, accepted = False, False
                if not result["timed_out"] and result["returncode"] == 0 and len(result["run_paths"]) == 1:
                    child = Path(result["run_paths"][0])
                    measured = json.loads((child / "comparison.json").read_text())
                    completed = measured["status"] == "completed" and json.loads((child / "run.json").read_text())["status"] == "completed"
                    accepted = completed and measured["backtracking_accepted"]
                rows.append({"case": case, "status": "completed" if completed else "failed",
                             "comparison_completed": completed, "backtracking_accepted": accepted, "execution": result})
            write_json(experiment.path / "cases.json", rows)
        experiment.annotate(comparison_cases_requested=len(rows),
            comparison_cases_completed=sum(row["comparison_completed"] for row in rows),
            backtracking_steps_accepted=sum(row.get("backtracking_accepted", False) for row in rows),
            cases_withheld=sum(row["status"] == "withheld_derivative_validation" for row in rows), no_full_optimization=True)
        return experiment.path
