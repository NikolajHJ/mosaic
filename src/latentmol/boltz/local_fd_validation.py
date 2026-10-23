"""Bounded saved-state validation of an explicitly versioned local FD policy.

Child processes authenticate and import the parent's original scientific code.
Only the derivative validation helper comes from the new immutable release.
Results belong to fresh audit directories; no optimizer runs or original gate
records change. A completed audit can still report ``validation_passed=false``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.multimodel.engine import Counters
from latentmol.multimodel.study import module_digest, tensor_digest, verified_file
from .active_set import ContextBank, Selection, make_branch, select_slots
from .local_fd import POLICY, policy_record, validate_local_fd


def run_saved(parent: Path, run_root: Path, *, production_root: Path, arm: str, step: int):
    """Revalidate one frozen saved point, retaining original failure provenance."""
    from latentmol.evaluate import load_model
    from .model import FrozenBoltzAffinity, load_affinity_model
    from . import _saved_gate_diagnostic as diagnostic

    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require DTU allocation and immutable validation source identity")
    parent = parent.resolve()
    record, old_gate, config, state, hashes = diagnostic._parent_inputs(
        parent, arm=arm, step=step, production_root=production_root)
    declared = policy_record(seed=config.diagnostics.seed, rtol=config.diagnostics.fd_rtol,
                             atol=config.diagnostics.fd_atol,
                             minimum_passes=config.diagnostics.minimum_fd_passes)
    inputs = {"parent_run": str(parent), "parent_artifact_sha256": hashes,
              "parent_arm": arm, "parent_step": step, "production_release": str(production_root.resolve()),
              "parent_source_sha256": record["source"]["sha256"],
              "parent_source_archive_sha256": record["source_archive_sha256"],
              "original_gate": old_gate, "original_config": config.record(), "policy": declared,
              "validation_module_sha256": file_sha256(Path(__file__)),
              "maximum_endpoint_forwards": 90, "base_forward_repeats": 4, "base_backward_repeats": 2,
              "claim_limit": "New local_v1 validation only; original failed gate remains failed; no molecular optimization."}
    # Old ExperimentRun implementations hashed integer config keys differently
    # before JSON serialization. Normalize metadata, never scientific inputs.
    inputs = json.loads(json.dumps(inputs, allow_nan=False))
    with ExperimentRun(run_root, "boltz-local-fd-validation", inputs, Path.cwd()) as experiment:
        report = {"status": "running", "validation_passed": False, "original_gate": old_gate,
                  "parent_unchanged": False, "frozen_state_unchanged": False,
                  "original_gate_not_reclassified": True, "repeats": [], "local_fd": {}}
        def persist():
            write_json(experiment.path / "validation.json", report)
        counters = Counters(lambda value: write_json(experiment.path / "oracle_cost.json", value))
        persist()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            decoder, training = load_model(verified_file(config.decoder["path"], config.decoder["sha256"]), config.device)
            if training["train_config"]["mode"] != config.mode:
                raise ValueError("Replayed decoder mode differs from original")
            decoder.eval().requires_grad_(False)
            bank = ContextBank(config, decoder.spec)
            point = state["z"].to(config.device)
            selection = Selection(state["active"].to(config.device), state["no_atom"].to(config.device), config.threshold)
            with torch.no_grad():
                natural = select_slots(decoder.decode(point), temperature=config.temperature,
                                       threshold=config.threshold, require_nonempty=False)
                if (arm == "active_set" or step == 0) and not torch.equal(natural.active, selection.active):
                    raise ValueError("Saved gate natural selector differs from original saved mask")
                if arm == "fixed_initial_mask":
                    start = torch.load(parent / "start.pt", map_location="cpu", weights_only=True)
                    initial = select_slots(decoder.decode(start["z"].to(config.device)),
                                           temperature=config.temperature, threshold=config.threshold)
                    if not torch.equal(initial.active, selection.active):
                        raise ValueError("Saved fixed mask differs from audited initial selector")
            scorer = FrozenBoltzAffinity(load_affinity_model(verified_file(
                config.affinity_checkpoint["path"], config.affinity_checkpoint["sha256"])).to(config.device),
                config.recycling_steps)
            branch = make_branch(scorer, bank, selection, config, record["source"]["sha256"])
            if branch.manifest.context_sha256 != old_gate["context_sha256"]:
                raise ValueError("Replayed bound context differs from original gate")
            identities = {"decoder_state_sha256": module_digest(decoder), "scorer_state_sha256": module_digest(scorer),
                          "context_sha256": branch.verify_context_unchanged(), "point_sha256": tensor_digest({"z": point})}
            for key in ("decoder_state_sha256", "scorer_state_sha256"):
                if identities[key] != record[key]:
                    raise ValueError(f"Frozen scientific state differs: {key}")
            report.update(identities_before=identities, runtime=diagnostic.runtime_description(point.device),
                          prepared_masks=diagnostic.prepared_mask_report(branch.prepared.features),
                          activation_checkpoint_policy=scorer.activation_checkpoint_policy)
            with torch.no_grad():
                native = counters.call("boltz", "local_validation_parity", "native_forward", lambda:
                    scorer.discrete_reference(branch.prepared.features, branch.prepared.coordinates), point.device)
                adapted = counters.call("boltz", "local_validation_parity", "forward", lambda:
                    branch(branch.prepared.seed.to_relaxed(decoder.spec)), point.device)
            if set(native) != set(adapted):
                raise ValueError("Replayed native and adapted heads differ")
            report["native_seed_parity"] = {}
            for key in native:
                torch.testing.assert_close(adapted[key], native[key], rtol=config.diagnostics.parity_rtol,
                                           atol=config.diagnostics.parity_atol)
                if not bool(native[key].isfinite().all()) or not bool(adapted[key].isfinite().all()):
                    raise ValueError("Nonfinite native-input parity")
                report["native_seed_parity"][key] = True
            persist()
            gradients = []
            for grad_enabled in (False, True):
                for repeat in range(2):
                    measured, gradient = diagnostic._probe(decoder, point, branch, config, counters,
                        grad_enabled=grad_enabled, backward=grad_enabled)
                    report["repeats"].append({**measured, "repeat": repeat})
                    if gradient is not None:
                        gradients.append(gradient)
                    persist()
            report["repeatability"] = diagnostic.repeat_comparison(report["repeats"], old_gate["measured"]["loss"])
            report["gradient_repeat_max_absolute_difference"] = float((gradients[0] - gradients[1]).abs().max())
            # Exact repetitions are a conservative replay precondition. Noise is
            # retained as an unresolved audit failure, never absorbed into FD tol.
            repeatable = all(report["repeatability"][key] == 0 for key in
                             ("no_grad_range", "grad_enabled_range", "first_grad_minus_no_grad"))
            repeatable = repeatable and report["gradient_repeat_max_absolute_difference"] == 0
            report["repeatability_passed"] = repeatable
            baseline = report["repeats"][2]["loss"]["value"]
            def score(candidate):
                measured, _ = diagnostic._probe(decoder, candidate, branch, config, counters,
                                                 grad_enabled=False)
                return {"loss": measured["loss"]["value"], "loss_dtype": measured["loss"]["dtype"],
                        "natural_selector_crossed": measured["natural_selector_crossed"]}
            if repeatable:
                validate_local_fd(point, gradients[0], score, baseline=baseline,
                                  seed=config.diagnostics.seed, rtol=config.diagnostics.fd_rtol,
                                  atol=config.diagnostics.fd_atol,
                                  minimum_passes=config.diagnostics.minimum_fd_passes,
                                  report=report["local_fd"], persist=persist)
            else:
                report["local_fd"].update(status="unresolved_repeatability", passed=False)
            after = {"decoder_state_sha256": module_digest(decoder), "scorer_state_sha256": module_digest(scorer),
                     "context_sha256": branch.verify_context_unchanged(), "point_sha256": tensor_digest({"z": point})}
            report.update(identities_after=after, frozen_state_unchanged=after == identities,
                          parent_unchanged=all(file_sha256(Path(path)) == digest for path, digest in hashes.items()))
            if not report["frozen_state_unchanged"] or not report["parent_unchanged"]:
                raise ValueError("Saved-state validation mutated scientific state or original evidence")
            report.update(status="completed", validation_passed=bool(report["local_fd"]["passed"]))
            persist()
            experiment.annotate(validation_passed=report["validation_passed"], original_gate_not_reclassified=True)
            return experiment.path
        except Exception as exc:
            report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            persist()
            raise


def run_study(protocol_path: Path):
    """Attempt the explicit finite case list serially, including failed audits."""
    from .gate_replay_study import load_protocol
    from .workflow_process import run_logged
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require DTU allocation and immutable validation source identity")
    protocol = load_protocol(protocol_path)
    launcher = Path.cwd() / "scripts/validate_saved_boltz_gates.py"
    inputs = {**protocol, "policy": POLICY, "protocol_path": str(protocol_path.resolve()),
              "protocol_file_sha256": file_sha256(protocol_path), "maximum_endpoint_forwards_per_case": 90,
              "maximum_child_seconds": len(protocol["cases"]) * protocol["timeout_seconds"],
              "claim_limit": "Saved-state validation only, no optimization or historical reclassification."}
    with ExperimentRun(Path(protocol["output_root"]), protocol["name"], inputs, Path.cwd()) as experiment:
        rows = []
        for case in protocol["cases"]:
            result = run_logged([sys.executable, str(launcher), "--parent-run", case["parent_run"],
                                "--production-release", case["production_release"], "--arm", case["arm"],
                                "--step", str(case["step"]), "--run-root", str(experiment.path / case["id"])],
                                experiment.path / f"{case['id']}.log", timeout_seconds=protocol["timeout_seconds"])
            completed, passed = False, False
            if not result["timed_out"] and result["returncode"] == 0 and len(result["run_paths"]) == 1:
                child = Path(result["run_paths"][0])
                evidence = json.loads((child / "validation.json").read_text())
                completed = evidence["status"] == "completed" and json.loads((child / "run.json").read_text())["status"] == "completed"
                passed = completed and evidence["validation_passed"]
            rows.append({"case": case, "validation_completed": completed, "validation_passed": passed,
                         "execution": result})
            write_json(experiment.path / "cases.json", rows)
        experiment.annotate(validation_cases_completed=sum(row["validation_completed"] for row in rows),
                            validation_cases_passed=sum(row["validation_passed"] for row in rows),
                            validation_cases_requested=len(rows), original_gates_not_reclassified=True)
        return experiment.path
