"""Prespecified allocated smoke/pilot cohorts with durable per-opportunity data.

This driver consumes only actually qualified callbacks. It does not grant T2,
native scorer, gradient, runtime or learned-head qualification itself. Fresh-pose
native follow-up is a later required stage, explicitly pending in these receipts.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import torch

from .alternation import (AlternatingSampler, AlternationResult, require_h0_exact)
from .native import GaussianStream
from .transport import TransportPolicy


SMOKE_SEEDS = (17, 23, 29)
PILOT_SEEDS = tuple(range(31, 39))
SMOKE_ARMS = ("D", "H0", "A5", "A5-exact")
PILOT_ARMS = ("D", "H0", "S", "A1", "A5", "A5-exact")


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Nonfinite value in a scientific loop receipt")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"Unsupported receipt type: {type(value).__name__}")


def _write_json(path: Path, payload, *, exclusive=True):
    text = json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(text + "\n")


def _append_json(path: Path, payload):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_T2_bond_change_example(example: dict) -> dict:
    """A bool flag alone cannot certify a bond-changing return to DiffSBDD."""
    if not isinstance(example, dict):
        raise RuntimeError("Pilot requires an actual T2 bond-changing smoke receipt")
    transport = example.get("transport", {})
    builder = transport.get("native_builder", {})
    before = example.get("rebuilt_baseline", {})
    after = example.get("rebuilt_proposed", {})
    required = {
        "applied": example.get("applied") is True,
        "hard_bonds_changed": example.get("hard_bonds_changed") is True,
        "pose_changed": example.get("pose_changed") is True,
        "stage_T2": transport.get("stage") == "T2",
        "native_graph_recovered": builder.get("complete_ordered_graph_matches") is True,
        "UFF_converged": transport.get("uff_convergence_code") == 0,
        "UFF_iterations": transport.get("uff_max_iterations") == 50,
        "UFF_constraint": transport.get("uff_position_constraint_max_displacement") == 0.5,
        "UFF_force": transport.get("uff_position_constraint_force_constant") == 10.0,
    }
    displacement = transport.get("max_displacement")
    required["bounded_displacement"] = (isinstance(displacement, (int, float))
                                        and math.isfinite(displacement)
                                        and 0 < displacement <= 0.5)
    b, a = before.get("total"), after.get("total")
    required["rebuilt_nonincrease"] = (isinstance(a, (int, float))
                                        and isinstance(b, (int, float))
                                        and math.isfinite(a) and math.isfinite(b) and a <= b)
    if not all(required.values()):
        raise RuntimeError(f"Incomplete/failed T2 scientific transport example: {required}")
    return {"passed": True, "checks": required, "example": example}


def _call_counts(records: Sequence[dict]) -> dict:
    """Count recorded calls, while preserving unequal preparation/rebuild costs."""
    keys = ("forward_calls", "backward_calls")
    totals = dict.fromkeys(keys, 0)

    def visit(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in totals and isinstance(item, int):
                    totals[key] += item
                elif isinstance(item, (Mapping, list, tuple)):
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(records)
    return {"recorded_optimizer_calls": totals,
            "preparation_and_native_rebuild_cost_included_in_elapsed_time": True,
            "nominal_budgets_are_not_actual_compute_equality": True}


def run_cohort(stepper, pocket, counts, factory_builder: Callable,
               qualification: Mapping, output, *, stage="smoke", deadline,
               transport_stage="T1", native_atom_decoder=None) -> dict:
    """Run the fixed cohort, without replacing failed seeds or unfinished arms.

    ``factory_builder(output, mode, transport_stage)`` supplies a qualified
    GuidanceFactory with ``callbacks``, ``admit_terminal``,
    ``qualification_evidence`` and ``native_atom_decoder``. ``deadline`` is an
    absolute time.monotonic deadline; the caller must admit the sealed work from
    measured operation timing before launch. A deadline stop preserves partial
    artifacts and fails the stage, rather than silently shrinking its cohort.
    """
    if stage not in {"smoke", "pilot"} or transport_stage not in {"T1", "T2"}:
        raise ValueError("Unknown cohort/transport stage")
    if int(stepper.timesteps) != 500:
        raise RuntimeError("The sealed experimental cohorts require all 500 native steps")
    if len(counts) != 1 or int(counts[0]) != 25:
        raise RuntimeError("The pilot protocol requires one fixed 25-heavy-atom sample")
    if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("Explicit allocated monotonic deadline required")
    if stage == "pilot":
        if transport_stage != "T2":
            raise RuntimeError("Full pilot requires separately qualified T2 transport")
        require_T2_bond_change_example(qualification.get("T2_bond_change_example"))
    seeds = SMOKE_SEEDS if stage == "smoke" else PILOT_SEEDS
    arms = SMOKE_ARMS if stage == "smoke" else PILOT_ARMS
    levels = (75, 50, 25) if stage == "smoke" else tuple(range(20, 0, -1))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "cohort.json").exists():
        raise FileExistsError("Refusing to replace an existing cohort receipt")
    started = time.monotonic()
    protocol = {"schema": "diffsbdd-alternation-cohort-v1", "stage": stage,
                "transport_stage": transport_stage, "seeds": list(seeds),
                "arms": list(arms), "opportunity_arrival_levels": list(levels),
                "diffusion_steps": 500, "heavy_atom_count": 25,
                "status": "running_not_passed", "completed": [],
                "failed_seed_replacement": False,
                "fresh_pose_native_followup": "required_separate_stage_pending"}
    _write_json(root / "protocol.json", protocol)
    all_opportunities, completed = [], []

    def check_deadline(*_args):
        if time.monotonic() >= deadline:
            raise TimeoutError("Allocated internal deadline reached; cohort remains incomplete")

    try:
        for seed in seeds:
            baseline = None
            for arm in arms:
                check_deadline()
                directory = root / f"seed-{seed:03d}" / arm
                directory.mkdir(parents=True, exist_ok=False)
                mode = "exact" if arm in {"D", "A5-exact"} else "learned"
                factory = factory_builder(directory / "guidance", mode, transport_stage)
                decoder = native_atom_decoder
                if decoder is None:
                    decoder = factory.native_atom_decoder
                sampler = AlternatingSampler(stepper, factory.callbacks, decoder,
                                              TransportPolicy(transport_stage), qualification)
                _write_json(directory / "qualification-evidence-before.json", {
                    "phase": "before_arm", "complete_arm_evidence": False,
                    "evidence": factory.qualification_evidence})
                stream = GaussianStream(seed)
                arm_started = time.monotonic()
                arm_records = []

                def record_opportunity(receipt):
                    record = {"seed": seed, "arm": arm, "stage": stage,
                              "transport_stage": transport_stage,
                              "elapsed_seconds": time.monotonic() - arm_started,
                              **receipt}
                    _append_json(directory / "opportunities.jsonl", record)
                    arm_records.append(record)
                    all_opportunities.append(record)
                    check_deadline()

                try:
                    if arm == "S":
                        if baseline is None:
                            raise RuntimeError("S arm requires the exact completed native D result")
                        sequential = sampler.run_sequential(
                            baseline.final, factory.admit_terminal,
                            on_block=record_opportunity, before_block=check_deadline)
                        result = AlternationResult(sequential.final, sequential.blocks,
                                                   baseline.gaussian_receipts)
                    else:
                        arm_levels = () if arm == "D" else levels
                        max_updates = 0 if arm in {"D", "H0"} else 1 if arm == "A1" else 5
                        result = sampler.run(pocket, counts, stream, arm_levels,
                                             max_updates, record_opportunity,
                                             before_transition=check_deadline)
                    if arm == "D":
                        baseline = result
                    if arm == "H0":
                        _write_json(directory / "H0-equality.json", require_h0_exact(baseline, result))
                    tensors = {field.name: getattr(result.final, field.name).detach().cpu()
                               for field in fields(result.final)}
                    with (directory / "terminal.pt").open("xb") as target:
                        torch.save({"seed": seed, "arm": arm, "tensors": tensors,
                                    "fresh_pose_evaluation": "pending"}, target)
                    _write_json(directory / "qualification-evidence-after.json", {
                        "phase": "after_completed_arm", "complete_arm_evidence": True,
                        "evidence": factory.qualification_evidence})
                    arm_receipt = {
                        "seed": seed, "arm": arm, "status": "completed_sampler_arm",
                        "elapsed_seconds": time.monotonic() - arm_started,
                        "opportunities": len(arm_records),
                        "applied_handoffs": sum(r.get("applied", False) for r in arm_records),
                        "hard_graph_changes": sum(r.get("hard_graph_changed", False)
                                                  and r.get("applied", False) for r in arm_records),
                        "hard_bond_changes": sum(r.get("hard_bonds_changed", False)
                                                 and r.get("applied", False) for r in arm_records),
                        "terminal_sha256": _sha(directory / "terminal.pt"),
                        "qualification_evidence_sha256": _sha(
                            directory / "qualification-evidence-after.json"),
                        "gaussian_receipts": list(result.gaussian_receipts),
                        **_call_counts(arm_records),
                        "fresh_pose_native_followup": "required_separate_stage_pending"}
                    _write_json(directory / "arm.json", arm_receipt)
                    completed.append(arm_receipt)
                except BaseException as exc:
                    _write_json(directory / "failure.json", {
                        "seed": seed, "arm": arm, "status": "failed_or_incomplete",
                        "error_type": type(exc).__name__, "error": str(exc),
                        "elapsed_seconds": time.monotonic() - arm_started})
                    raise
                finally:
                    if not (directory / "qualification-evidence-after.json").exists():
                        _write_json(directory / "qualification-evidence-after.json", {
                            "phase": "after_failed_or_incomplete_arm", "complete_arm_evidence": False,
                            "evidence": factory.qualification_evidence})
                    with (directory / "gaussians.pt").open("xb") as target:
                        # S shares D's finished sample and consumes zero further
                        # draws. Reference its baseline instead of inventing noise.
                        torch.save({"seed": seed, "arm": arm,
                                    "namespace": stream.namespace,
                                    "draws": stream.raw_draws,
                                    "receipts": stream.receipts,
                                    "S_baseline_draws": "../D/gaussians.pt" if arm == "S" else None}, target)
        examples = []
        if transport_stage == "T2":
            for opportunity in all_opportunities:
                if opportunity.get("applied") and opportunity.get("hard_bonds_changed"):
                    try:
                        examples.append(require_T2_bond_change_example(opportunity))
                    except RuntimeError:
                        # A failed/missing necessary T2 receipt is preserved in
                        # opportunities; it cannot qualify the next stage.
                        continue
        summary = {**protocol, "status": "completed_cohort_not_fresh_pose_validation",
                   "completed": completed, "elapsed_seconds": time.monotonic() - started,
                   "actual_effective_state_transfers": sum(r.get("applied", False)
                                                            for r in all_opportunities),
                   "actual_hard_graph_transfers": sum(r.get("applied", False)
                                                       and r.get("hard_graph_changed", False)
                                                       for r in all_opportunities),
                   "T2_bond_change_examples": examples,
                   "mapping_or_chemical_efficacy_claim": False}
        if not summary["actual_effective_state_transfers"]:
            summary["status"] = "feasibility_failed_zero_effective_transfers"
        if transport_stage == "T2" and not examples:
            summary["status"] = "T2_feasibility_failed_no_qualified_bond_transfer"
        _write_json(root / "cohort.json", summary)
        if "failed" in summary["status"]:
            raise RuntimeError(summary["status"])
        return summary
    except BaseException as exc:
        if not (root / "cohort.json").exists():
            _write_json(root / "cohort.json", {
                **protocol, "status": "failed_or_incomplete", "completed": completed,
                "error_type": type(exc).__name__, "error": str(exc),
                "elapsed_seconds": time.monotonic() - started})
        raise
