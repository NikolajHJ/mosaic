"""HPC-only attribution of saved DBMol/repair finite-difference failures.

This is an observer, not a new validation policy. It evaluates the ORIGINAL
frozen production scorer at the ORIGINAL point, mask, pose and repaired target.
All components are measured on identical endpoints in both autograd modes.
It makes no optimization updates, invokes no DeFoG worker, and never approves
or rewrites a failed gate. A completed diagnostic can retain disagreements.

The controller starts fresh Python children so old scientific modules cannot
be shadowed by a newer editable checkout. Only this file is new executable
instrumentation; the child authenticates every old source file before imports.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


REPEATS = 3
CASE_TIMEOUT_SECONDS = 2700
CLAIM_LIMIT = ("Saved-state diagnostic only; no optimizer, new repair oracle calls, "
               "tolerance changes, production patch or historical gate approval. "
               "Output ULPs are not error bounds for an entire native network.")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _absolute(value):
    return isinstance(value, str) and bool(value) and Path(value).is_absolute()


def load_plan(path, expected_sha256):
    """Require a hash-pinned finite list and explicit immutable parent assets."""
    if not _hash(expected_sha256) or _sha(path) != expected_sha256:
        raise ValueError("Diagnostic plan SHA256 mismatch")
    plan = _read(path)
    fields = {"version", "name", "output_root", "production_release",
              "production_manifest_sha256", "production_archive_sha256", "cases"}
    if not isinstance(plan, dict) or set(plan) != fields or type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("Require exactly the version-1 attribution plan fields")
    if not isinstance(plan["name"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", plan["name"]):
        raise ValueError("Diagnostic name must be a path-safe identifier")
    for key in ("output_root", "production_release"):
        if not _absolute(plan[key]):
            raise ValueError(f"Require absolute {key}")
    for key in ("production_manifest_sha256", "production_archive_sha256"):
        if not _hash(plan[key]):
            raise ValueError(f"Invalid {key}")
    if not isinstance(plan["cases"], list) or not 1 <= len(plan["cases"]) <= 3:
        raise ValueError("Declare one to three cases; no outcome-selected replacement")
    names = set()
    for case in plan["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "parent_run", "arm", "step", "parent_artifacts_sha256"}:
            raise ValueError("Case needs exactly id/parent/arm/step/artifact hashes")
        if (not isinstance(case["id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", case["id"])
                or case["id"] in names or not _absolute(case["parent_run"])
                or case["arm"] != "fixed_initial_mask" or type(case["step"]) is not int or not 0 <= case["step"] <= 9999):
            raise ValueError("Require distinct fixed-mask saved cases and valid steps")
        names.add(case["id"])
        assets = case["parent_artifacts_sha256"]
        if (not isinstance(assets, dict) or not assets
                or any(not _absolute(p) or not _hash(h) for p, h in assets.items())):
            raise ValueError("Parent artifacts require absolute paths and SHA256s")
    return plan


def authenticate_source(root, record, plan):
    """Verify ALL original executable source, including uncommitted bytes."""
    source = record.get("source", {})
    files = source.get("files")
    if (not isinstance(files, dict) or not files or _digest(files) != source.get("sha256")
            or source.get("sha256") != plan["production_manifest_sha256"]
            or record.get("source_archive_sha256") != plan["production_archive_sha256"]):
        raise ValueError("Original source manifest/archive identity mismatch")
    root = Path(root).resolve()
    if _sha(Path(str(root) + ".tar.gz")) != plan["production_archive_sha256"]:
        raise ValueError("Original production archive bytes changed")
    for name, expected in files.items():
        path = (root / name).resolve()
        if (not _hash(expected) or root not in path.parents or not path.is_file()
                or _sha(path) != expected):
            raise ValueError(f"Original production file changed or escaped: {name}")


def authenticated_artifact(case, path):
    """No dynamically discovered parent file may bypass the pinned asset map."""
    path = Path(path).absolute()
    expected = case["parent_artifacts_sha256"].get(str(path))
    if expected is None or _sha(path) != expected:
        raise ValueError(f"Missing/mismatched pinned parent artifact: {path}")
    return path


def failed_item(gate, *, policy="local_v1"):
    """Component failures may have no top-level finite_differences at all."""
    if (gate.get("status") != "failed" or gate.get("passed") is not False
            or gate.get("native_seed_parity") is not True or "loss" not in gate.get("measured", {})):
        raise ValueError("Require a failed derivative gate after native parity")
    failures = [(name, item) for name, item in gate.get("component_gradients", {}).items()
                if item.get("passed") is False and (policy == "branch_local_v2" or item.get("status") == "failed")]
    total = gate.get("local_fd_validation", {})
    if total.get("status") == "failed" and total.get("passed") is False:
        failures.append(("combined", total))
    if len(failures) != 1:
        raise ValueError("Require one completed failed local_v1 derivative item")
    name, item = failures[0]
    if policy == "branch_local_v2":
        if item.get("validation_contract") != policy:
            raise ValueError("Failed item does not have the declared branch-local contract")
        # This is an explicit read-only view of the persisted branch secants,
        # not a rewrite of the historical failure or a change of its directions.
        item = item["local_branch_fd"]
    elif policy != "local_v1":
        raise ValueError("Unsupported historical derivative contract")
    if not item.get("finite_differences") or item.get("policy", {}).get("version") != "local_v1":
        raise ValueError("Only the declared original local_v1 grid is replayed")
    return name, item


def reconstruct_directions(point, gradient, item):
    """Replay original Gaussian proposals by index; never select after FD scores.

    Recorded random-direction digests are mandatory. The original policy did
    not save its gradient-direction digest; its reconstructed digest is retained
    and explicitly labelled unavailable for old bitwise direction authentication.
    A random mismatch is still measured, but cannot certify the original case.
    """
    import torch
    from latentmol.multimodel.study import tensor_digest
    if not math.isfinite(float(gradient.double().norm())) or float(gradient.double().norm()) <= 1e-12:
        raise ValueError("Replay requires a finite nonzero analytic derivative")
    selection = item["direction_selection"]
    proposals = selection["proposals"]
    failed = [name for name, result in item["direction_results"].items()
              if result.get("passed") is False and name != "gradient"]
    if not failed or len(failed) > 3:
        raise ValueError("Require one to three originally failed random directions")
    chosen = {}
    for name in failed:
        matches = [row for row in proposals if row.get("accepted") is True and row.get("selected_label") == name]
        if len(matches) != 1 or not _hash(matches[0].get("direction_sha256")):
            raise ValueError("Failed direction has no unique saved proposal hash")
        chosen[name] = matches[0]
    seeds = {row["generator_seed"] for row in chosen.values()}
    if len(seeds) != 1 or any(type(row["proposal_index"]) is not int or not 0 <= row["proposal_index"] < 64 for row in chosen.values()):
        raise ValueError("Malformed original bounded Gaussian direction receipts")
    generator = torch.Generator(device="cpu").manual_seed(next(iter(seeds)))
    replay = {}
    for index in range(max(row["proposal_index"] for row in chosen.values()) + 1):
        raw = torch.randn(point.shape, generator=generator).to(point)
        direction = raw / raw.norm()
        for name, row in chosen.items():
            if row["proposal_index"] == index:
                digest = tensor_digest({"direction": direction})
                replay[name] = (direction, {"saved_sha256": row["direction_sha256"], "replayed_sha256": digest,
                    "hash_matches_original": digest == row["direction_sha256"],
                    "proposal_index": index, "seed": row["generator_seed"]})
    direction = gradient.detach() / float(gradient.double().norm())
    replay = {"gradient": (direction, {"replayed_sha256": tensor_digest({"direction": direction}),
               "saved_sha256": None, "hash_matches_original": None,
               "scope": "Original gradient vector/direction was not saved; recreated from original frozen scorer"}), **replay}
    return replay


def _target_from_receipts(case, config, spec, active, point_sha256, gate, *, source_sha256,
                          expected_target_sha256=None):
    """Load an accepted historical target; never sample, repair or invoke workers."""
    import torch
    from latentmol.repair_guidance.loss import FrozenRepairTarget
    from latentmol.repair_guidance.provider import _target_from_record
    from latentmol.multimodel.study import hard_digest
    folder = Path(case["parent_run"]) / case["arm"]
    eligible = [(int(p.parent.name.split("-")[-1]), p) for p in folder.glob("repair-refresh-*/refresh.json")
                if p.parent.name.split("-")[-1].isdigit() and int(p.parent.name.split("-")[-1]) <= case["step"]]
    if not eligible:
        raise ValueError("No historical target refresh before the failed state")
    refresh_step, path = max(eligible)
    receipt = _read(authenticated_artifact(case, path))
    index = receipt.get("selected_attempt")
    if (receipt.get("status") != "accepted" or type(index) is not int
            or not isinstance(receipt.get("attempts"), list) or not 0 <= index < len(receipt["attempts"])
            or receipt.get("step") != refresh_step or receipt.get("temperature") != config.temperature
            or receipt.get("source_sha256") != source_sha256
            or receipt.get("occupied_slots") != active.nonzero().flatten().tolist()):
        raise ValueError("Historical refresh lacks an aligned accepted target")
    if refresh_step == case["step"] and receipt.get("latent_sha256") != point_sha256:
        raise ValueError("Historical refresh point differs from failed point")
    expected_provider = config.repair_guidance.record()
    expected_provider["weight"] = config.repair_guidance.provider_activation_weight
    if receipt.get("config") != expected_provider:
        raise ValueError("Historical provider config differs from original scheduled provider")
    attempt = receipt["attempts"][index]
    response_path = authenticated_artifact(case, attempt["response_path"])
    response = _read(response_path)
    if (response != attempt.get("response") or _sha(response_path) != attempt.get("response_sha256")
            or response.get("status") != "accepted" or attempt.get("status") != "accepted"
            or attempt.get("target_eligible") is not True or attempt.get("timeout") is not False
            or attempt.get("process_returncode") != 0
            or response.get("request_sha256") != attempt.get("request_sha256")
            # Native request/response candidate identity hashes canonical JSON;
            # the separately retained tensor digest is a different contract.
            or response.get("candidate_sha256") != attempt.get("candidate_json_sha256")
            or response.get("occupied_slots") != receipt["occupied_slots"]
            or response.get("native_order_preserved") is not True
            or response.get("slot_alignment") != "native_order_preserved"
            or any(response.get("gates", {}).get(key) is not True for key in
                   ("chemical_validity", "fixed_atom_count", "native_label_roundtrip", "neutral_charge_subset", "slot_alignment"))):
        raise ValueError("Historical selected worker response/eligibility mismatch")
    provenance = response.get("provenance", {})
    if (provenance.get("adapter_source_sha256") != receipt.get("source_sha256")
            or provenance.get("source_commit") != config.repair_guidance.provider["source"]["commit"]
            or provenance.get("checkpoint_sha256") != config.repair_guidance.provider["checkpoint"]["sha256"]
            or provenance.get("frozen_state_unchanged") is not True
            or not provenance.get("frozen_state_sha256_before")
            or provenance.get("frozen_state_sha256_before") != provenance.get("frozen_state_sha256_after")):
        raise ValueError("Historical target lacks matching frozen-model provenance")
    hard = _target_from_record(response["target"], spec)
    target = FrozenRepairTarget.create(hard, active, refresh_step=refresh_step, receipt_path=str(response_path))
    if (hard_digest(hard) != receipt.get("selected_target_sha256")
            or target.identity != (expected_target_sha256 if expected_target_sha256 is not None
                                   else gate.get("repair_only_validation", {}).get("target_sha256"))):
        raise ValueError("Historical repair target identity differs from failed gate")
    neutral = spec.formal_charges.index(0)
    if (not torch.equal(hard.bonds, hard.bonds.transpose(1, 2)) or bool(hard.bonds.diagonal(dim1=1, dim2=2).any())
            or not bool((hard.charges == neutral).all())):
        raise ValueError("Historical target violates the declared neutral symmetric graph contract")
    return target, {"refresh_step": refresh_step, "refresh_path": str(path), "response_path": str(response_path),
                    "target_sha256": target.identity, "new_worker_calls": 0}


def disabled_control_target(case, config, gate):
    """Authenticate an actual zero-activation control, never a failed repair.

    A positive scheduled provider without a selected target is NOT a control.
    Zero-weight historical refresh receipts are retained and checked so this
    observer cannot quietly convert a guided case into a binding-only replay.
    """
    guidance = config.repair_guidance
    if guidance is None or guidance.provider_activation_weight != 0 or guidance.weight_schedule is not None:
        raise ValueError("Target-free replay requires a disabled zero-activation control")
    application = gate.get("repair_application", {})
    measured = gate.get("measured", {}).get("repair_guidance", {})
    if (application.get("scheduled_weight") != 0 or application.get("target_available") is not False
            or application.get("used_target") is not False or application.get("applied_weight") != 0
            or measured.get("status") != "disabled_weight_zero" or measured.get("weight") != 0
            or gate.get("repair_only_validation") is not None):
        raise ValueError("Failed gate does not establish disabled repair guidance")
    folder = Path(case["parent_run"]) / case["arm"]
    checked = []
    # Disabled providers return before creating worker directories. The runner
    # persists their receipt inside the objective-refresh record instead.
    if any(folder.glob("repair-refresh-*/refresh.json")):
        raise ValueError("Disabled control unexpectedly contains a worker refresh directory")
    for path in sorted(folder.glob("repair-refresh-objective-*.json")):
        if not path.stem.split("-")[-1].isdigit():
            raise ValueError("Malformed control refresh path")
        if int(path.stem.split("-")[-1]) > case["step"]:
            continue
        objective = _read(authenticated_artifact(case, path))
        receipt = objective.get("receipt", {})
        if (receipt.get("status") != "disabled" or receipt.get("guidance_weight") != 0
                or receipt.get("attempts") != [] or receipt.get("selected_attempt") is not None
                or objective.get("new_target_sha256") is not None):
            raise ValueError("Zero-activation control contains a native repair attempt or target")
        checked.append(str(path))
    expected_steps = list(range(0, case["step"] + 1, guidance.refresh_interval))
    if [int(Path(path).stem.split("-")[-1]) for path in checked] != expected_steps:
        raise ValueError("Missing declared disabled control refresh receipts")
    return None, {"status": "disabled_zero_activation_control", "target_sha256": None,
                  "refresh_step": None, "checked_refresh_receipts": checked, "new_worker_calls": 0}


def forward_values(decoder, point, branch, config, counters, target, weight, *, grad_enabled):
    """Original native forward math with no backward for endpoint mode comparison.

    Loss assembly reuses production scalar_loss and fixed_target_nll, in its
    original order/dtype. This cannot substitute a different molecule or graph.
    Returned metadata is detached; no activations survive the probe call.
    """
    import torch
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.engine import scalar_loss
    from latentmol.multimodel.study import tensor_digest
    from latentmol.repair_guidance.loss import fixed_target_nll
    probe = point.detach().clone().requires_grad_(grad_enabled)
    with torch.set_grad_enabled(grad_enabled):
        decoded = decoder.decode(probe)
        graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature, fixed_active=branch.prepared.active)
        outputs = counters.call("boltz", "attribution_grad_forward" if grad_enabled else "attribution_no_grad_forward",
                                "forward", lambda: branch(graph), point.device)
        binding, _ = scalar_loss(outputs, config.objective)
        if target is None:
            if weight != 0 or config.repair_guidance.provider_activation_weight != 0:
                raise ValueError("Target-free probe requires disabled repair guidance")
            # Preserve production's exact control arithmetic: no artificial
            # add-zero repair operation and no fabricated repair likelihood.
            repair, detail, total = None, {"target_sha256": None}, binding
        else:
            repair, detail = fixed_target_nll(decoded, graph, target, decoder.spec,
                 temperature=config.temperature, normalization=config.repair_guidance.normalization)
            total = binding + weight * repair
        if not bool(total.isfinite().all()):
            raise ValueError("Nonfinite attribution objective")
        values = {name: float(value.detach()) for name, value in outputs.items()}
        values.update(binding_loss=float(binding.detach()), combined=float(total.detach()))
        repair_dtypes = {}
        if repair is not None:
            values["repair_nll"] = float(repair.detach())
            repair_dtypes["repair_nll"] = str(repair.dtype)
        return {"values": values, "dtypes": {**{k: str(v.dtype) for k, v in outputs.items()},
                **repair_dtypes, "binding_loss": str(binding.dtype), "combined": str(total.dtype)},
                "contact_selection": deepcopy(getattr(branch.scorer, "last_contact_selection", {})),
                "graph_sha256": tensor_digest({k: getattr(graph, k) for k in ("atoms", "charges", "bonds")}),
                "target_sha256": detail["target_sha256"], "weight": weight, "grad_enabled": grad_enabled}


def sum_gradient_identity(gradient, components, repair_gradient, config, weight):
    """Audit linear assembly, widening only the diagnostic vector comparison."""
    from latentmol.multimodel.study import tensor_digest
    scale = (-1 if config.objective.direction == "maximize" else 1) / config.objective.scale
    combined = (components["dbmol_affinity_loss"].double()
                + config.dbmol.contact_weight * components["dbmol_contact_loss"].double()
                + config.dbmol.confidence_weight * components["dbmol_confidence_loss"].double()) * scale
    combined = combined + weight * repair_gradient.double()
    difference = gradient.double() - combined
    return {"combined_gradient_sha256": tensor_digest({"gradient": gradient}),
            "sum_gradient_sha256_float64": tensor_digest({"gradient": combined}),
            "difference_norm_float64": float(difference.norm()), "max_absolute_difference": float(difference.abs().max()),
            "combined_norm": float(gradient.double().norm()), "sum_norm": float(combined.norm()),
            "identity_scope": "Original combined backward vs weighted component derivative sum; no normalization/patch or acceptance threshold"}


def scalar_assembly_identity(outputs, config, repair_row, weight, measured):
    """Audit endpoint assembly on detached ORIGINAL analytic-forward outputs.

    Capture via a passive branch hook, so this comparison spends no additional
    native forward. The original repair scalar is reconstructed in its actual
    dtype; widening its Python metadata cannot recover network precision.
    """
    import torch
    from latentmol.multimodel.engine import scalar_loss
    kinds = {str(k): k for k in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
    binding, _ = scalar_loss(outputs, config.objective)
    repair = (torch.tensor(repair_row["loss"], device=binding.device, dtype=kinds[repair_row["loss_dtype"]])
              if repair_row is not None else None)
    total = binding + weight * repair if repair is not None else binding
    return {"source": "Detached outputs captured during original score_at; no extra model forward",
            "instrumented_combined_loss": float(total), "production_combined_loss": measured["loss"],
            "exact_combined_scalar_match": float(total) == measured["loss"],
            "exact_component_scalars_match": all(float(v) == measured["components"][k] for k, v in outputs.items()),
            "exact_binding_scalar_match": float(binding) == measured["binding_loss"],
            "exact_repair_scalar_match": (repair_row["loss"] == measured["repair_guidance"]["normalized_nll"]
                if repair_row is not None else None), "repair_term_present": repair_row is not None,
            "combined_dtype": str(total.dtype), "production_dtype": measured["loss_dtype"]}


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item, *, persist=lambda: None,
               report=None):
    """Measure only declared failed directions plus one gradient control.

    Each endpoint forward yields every component, making branch attribution
    possible without independently re-running each scalar's finite differences.
    The original tolerance is reported per term but is not a new pass gate.
    """
    import torch
    from latentmol.boltz.active_set import score_at
    from latentmol.boltz.local_fd import EPSILONS, endpoint_row
    from latentmol.repair_guidance.loss import target_score_at
    report = report if report is not None else {}
    report.update(status="probing", claim_limit=CLAIM_LIMIT, baselines=[], finite_differences=[], original_gate_not_reclassified=True)
    gradients, component_sets, repairs = [], [], []
    for repeat in range(REPEATS):
        row = forward_values(decoder, point, branch, config, counters, target, weight, grad_enabled=False)
        report["baselines"].append({"repeat": repeat, **row})
        components = {}
        captured = {}
        def observe(_module, _inputs, outputs):
            captured.update({k: v.detach().clone() for k, v in outputs.items()})
        hook = branch.register_forward_hook(observe)
        try:
            measured, gradient = score_at(decoder, point, branch, config, counters, "attribution_analytic", gradients=True,
                                          component_gradients=components, repair_target=target, repair_weight=weight)
        finally:
            hook.remove()
        if target is None:
            repair_row, repair_gradient = None, torch.zeros_like(gradient)
        else:
            repair_row, repair_gradient = target_score_at(decoder, point, branch.prepared.active, target,
                 temperature=config.temperature, normalization=config.repair_guidance.normalization, gradients=True)
        # score_at is the production analytic path; endpoint instrumentation uses
        # the same scalar assembly and is checked against these actual values.
        values = {**measured["components"], "binding_loss": measured["binding_loss"], "combined": measured["loss"]}
        if repair_row is not None:
            values["repair_nll"] = repair_row["loss"]
        report["baselines"].append({"repeat": repeat, "grad_enabled": True, "values": values,
              "loss_dtype": measured["loss_dtype"], "contact_selection": measured["contact_selection"],
              "sum_gradient_identity": sum_gradient_identity(gradient, components, repair_gradient, config, weight),
              "scalar_assembly_identity": scalar_assembly_identity(captured, config, repair_row, weight, measured)})
        gradients.append(gradient)
        component_sets.append(components)
        repairs.append(repair_gradient)
        persist()
    gradient, components, repair_gradient = gradients[0], component_sets[0], repairs[0]
    report["gradient_repeat_max_absolute_difference"] = max(float((g - gradient).abs().max()) for g in gradients)
    report["component_gradient_repeat_max_absolute_difference"] = {
        k: max(float((g[k] - components[k]).abs().max()) for g in component_sets) for k in components}
    report["repair_gradient_repeat_max_absolute_difference"] = (max(float((g - repair_gradient).abs().max()) for g in repairs)
                                                                  if target is not None else None)
    baseline = report["baselines"][1]["values"]
    terms = {**components, "binding_loss": gradient - weight * repair_gradient, "combined": gradient}
    if target is not None:
        terms["repair_nll"] = repair_gradient
    relevant = gradient if item_name == "combined" else components[item_name]
    directions = reconstruct_directions(point, relevant, item)
    report["direction_replay"] = {name: meta for name, (_, meta) in directions.items()}
    report["random_directions_match_original"] = all(meta["hash_matches_original"] for name, (_, meta) in directions.items() if name != "gradient")
    report["baseline_ranges"] = {k: {
        "no_grad_range": max(r["values"][k] for r in report["baselines"] if not r["grad_enabled"])
                        - min(r["values"][k] for r in report["baselines"] if not r["grad_enabled"]),
        "grad_range": max(r["values"][k] for r in report["baselines"] if r["grad_enabled"])
                     - min(r["values"][k] for r in report["baselines"] if r["grad_enabled"]),
        "first_grad_minus_no_grad": baseline[k] - report["baselines"][0]["values"][k]} for k in terms}
    persist()
    for label, (direction, _) in directions.items():
        for epsilon in EPSILONS:
            for mode in (False, True):
                plus = forward_values(decoder, point + epsilon * direction, branch, config, counters, target, weight, grad_enabled=mode)
                minus = forward_values(decoder, point - epsilon * direction, branch, config, counters, target, weight, grad_enabled=mode)
                pairs = {}
                for key, derivative in terms.items():
                    p = {"loss": plus["values"][key], "loss_dtype": plus["dtypes"][key]}
                    m = {"loss": minus["values"][key], "loss_dtype": minus["dtypes"][key]}
                    pairs[key] = endpoint_row(point, direction, derivative, epsilon, p, m, baseline[key],
                                             rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
                keys = ("best_ligand_token_per_pocket", "anchor_tokens")
                report["finite_differences"].append({"direction": label, "epsilon": epsilon, "grad_enabled": mode,
                    "terms": pairs, "plus": plus, "minus": minus,
                    "contact_winners_changed": any(plus["contact_selection"].get(k) != minus["contact_selection"].get(k) for k in keys)})
                persist()
    report.update(status="completed", diagnostic_completed=True, old_gate_certified_by_this_diagnostic=False,
                  scored_directions=list(directions), maximum_scorer_forwards=REPEATS * 2 + len(directions) * len(EPSILONS) * 4)
    persist()
    return report


def _bootstrap(plan, case):
    """Child bootstrap runs before any latentmol import in its fresh process."""
    if any(k == "latentmol" or k.startswith("latentmol.") for k in sys.modules):
        raise ValueError("Fresh attribution child must bootstrap before latentmol imports")
    record = _read(authenticated_artifact(case, Path(case["parent_run"]) / "run.json"))
    for path in case["parent_artifacts_sha256"]:
        authenticated_artifact(case, path)
    authenticate_source(plan["production_release"], record, plan)
    sys.path.insert(0, str(Path(plan["production_release"]) / "src"))
    import latentmol.boltz.active_set as production
    if Path(production.__file__).resolve() != Path(plan["production_release"]).resolve() / "src/latentmol/boltz/active_set.py":
        raise ValueError("Scientific modules did not resolve to old immutable release")
    return record


def run_case(plan, case, output_root, *, expected_policy="local_v1", probe=None):
    """Authenticate a recoverable preliminary state, then collect bounded data."""
    record = _bootstrap(plan, case)
    import torch
    from latentmol.evaluate import load_model
    from latentmol.experiments import ExperimentRun, source_manifest, write_json
    from latentmol.multimodel.engine import Counters
    from latentmol.multimodel.study import module_digest, tensor_digest, verified_file
    from latentmol.boltz.active_set import ActiveSetConfig, ContextBank, Selection, make_branch, select_slots
    from latentmol.boltz.dbmol_model import FrozenBoltzDBMol, load_confidence_model
    from latentmol.boltz.model import load_affinity_model
    parent = Path(case["parent_run"])
    folder = parent / case["arm"]
    gate = _read(authenticated_artifact(case, folder / f"gate-{case['step']:04d}.json"))
    item_name, item = failed_item(gate, policy=expected_policy)
    if record.get("status") != "failed":
        raise ValueError("Only immutable failed parents may be diagnosed")
    config_path = authenticated_artifact(case, record["config"]["config_path"])
    if _sha(config_path) != record["config"]["config_file_sha256"]:
        raise ValueError("Parent config-file identity mismatch")
    config = ActiveSetConfig.load(config_path)
    for key, value in config.record().items():
        if json.loads(json.dumps(value)) != record["config"].get(key):
            raise ValueError(f"Original resolved config drift: {key}")
    if (expected_policy not in ("local_v1", "branch_local_v2") or config.dbmol is None
            or config.repair_guidance is None or config.fd_policy != expected_policy):
        raise ValueError("Require original DBMol plus repaired-target local_v1 case")
    state = torch.load(authenticated_artifact(case, folder / f"state-{case['step']:04d}.pt"), map_location="cpu", weights_only=True)
    start = torch.load(authenticated_artifact(case, parent / "start.pt"), map_location="cpu", weights_only=True)
    audit = _read(authenticated_artifact(case, parent / "start-audit.json"))
    point_sha = tensor_digest({"z": state["z"]})
    if (audit.get("passed") is not True or point_sha != gate["point_sha256"]
            or state["active"].dtype != torch.bool or state["active"].ndim != 1
            or state["active"].nonzero().flatten().tolist() != gate["active_slots"]):
        raise ValueError("Saved preliminary state does not match audited point/mask")
    if state.get("context_sha256") not in (None, gate["context_sha256"]):
        raise ValueError("Saved state context conflicts with authenticated failed gate")
    weight = config.repair_guidance.weight_at(case["step"], config.optimizer.steps)
    if (gate.get("effective_repair_weight", 0 if config.repair_guidance.provider_activation_weight == 0 else None) != weight
            or state.get("effective_repair_weight", weight) != weight):
        raise ValueError("Original held repair weight differs from failed state")
    if config.repair_guidance.weight_schedule is not None:
        schedule = _read(authenticated_artifact(case, folder / f"repair-weight-{case['step']:04d}.json"))
        if schedule.get("effective_weight") != weight or schedule.get("point_sha256") != point_sha:
            raise ValueError("Saved weight schedule differs from failed state")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    decoder, training = load_model(verified_file(config.decoder["path"], config.decoder["sha256"]), config.device)
    if training["train_config"]["mode"] != config.mode:
        raise ValueError("Original decoder mode mismatch")
    decoder.eval().requires_grad_(False)
    point = state["z"].to(config.device)
    if point.shape != (1, decoder.config.latent) or not bool(point.isfinite().all()):
        raise ValueError("Malformed saved latent point")
    active = state["active"].to(config.device)
    with torch.no_grad():
        initial = select_slots(decoder.decode(start["z"].to(config.device)), temperature=config.temperature, threshold=config.threshold)
    if not torch.equal(initial.active, active):
        raise ValueError("Saved fixed mask differs from original initial selector")
    if config.repair_guidance.provider_activation_weight == 0:
        target, target_receipt = disabled_control_target(case, config, gate)
    else:
        expected_target = state.get("repair_target_sha256") if expected_policy == "branch_local_v2" else None
        if expected_policy == "branch_local_v2" and expected_target is None:
            # A failed audit may leave the pre-audit snapshot, which predates
            # the ordinary scored-state target field. The independently pinned
            # objective-refresh receipt authenticates that exact point/target.
            jump = _read(authenticated_artifact(case, folder / f"repair-refresh-objective-{case['step']:04d}.json"))
            if jump.get("point_sha256") != point_sha or jump.get("context_sha256") != gate["context_sha256"]:
                raise ValueError("Pre-audit target refresh point/context differs")
            expected_target = jump.get("new_target_sha256")
        if expected_policy == "branch_local_v2" and not _hash(expected_target):
            raise ValueError("Branch-local saved state must authenticate its frozen repair target")
        target, target_receipt = _target_from_receipts(case, config, decoder.spec, active.cpu(), point_sha, gate,
            source_sha256=record["source"]["sha256"], expected_target_sha256=expected_target)
    if target is not None and target_receipt["refresh_step"] != case["step"]:
        # Current three declared failures occurred at refresh points. Restrict
        # this first diagnostic rather than invent missing refresh-z evidence.
        raise ValueError("This diagnostic requires a target refreshed at the exact failed state")
    bank = ContextBank(config, decoder.spec)
    config.dbmol.verify_reference(bank.templates[config.anchor_count], config.contexts[config.anchor_count]["sha256"])
    scorer = FrozenBoltzDBMol(load_affinity_model(verified_file(config.affinity_checkpoint["path"], config.affinity_checkpoint["sha256"])).to(config.device),
                             load_confidence_model(config.dbmol.confidence_checkpoint).to(config.device), config.dbmol, config.recycling_steps)
    branch = make_branch(scorer, bank, Selection(active, state["no_atom"].to(config.device), config.threshold), config, record["source"]["sha256"])
    identities = {"decoder_state_sha256": module_digest(decoder), "scorer_state_sha256": module_digest(scorer),
                  "context_sha256": branch.verify_context_unchanged(), "point_sha256": tensor_digest({"z": point})}
    if (identities["context_sha256"] != gate["context_sha256"] or identities["point_sha256"] != point_sha
            or any(identities[k] != record[k] for k in ("decoder_state_sha256", "scorer_state_sha256"))):
        raise ValueError("Replayed frozen scientific identities differ from original failure")
    inputs = {"case": case, "original_item": item_name, "production_release": plan["production_release"],
              "production_manifest_sha256": plan["production_manifest_sha256"], "production_archive_sha256": plan["production_archive_sha256"],
              "diagnostic_module_sha256": _sha(__file__), "repeats_per_mode": REPEATS,
              "case_timeout_seconds": CASE_TIMEOUT_SECONDS, "target": target_receipt, "effective_weight": weight,
              "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(output_root), case["id"], inputs, Path.cwd()) as experiment:
        report = {"status": "running", "diagnostic_completed": False, "identities_before": identities,
                  "preliminary_context_recovered_from_authenticated_gate": state.get("context_sha256") is None,
                  "target": target_receipt, "original_error": gate["error"], "original_item": item_name,
                  "original_gate_not_reclassified": True, "claim_limit": CLAIM_LIMIT,
                  "runtime": {"device": str(point.device), "device_name": torch.cuda.get_device_name(point.device),
                              "torch_version": torch.__version__, "tf32_allowed": False,
                              "production_job_id": record["environment"].get("job_id"),
                              "current_job_id": os.environ["LSB_JOBID"]}}
        counters = Counters(lambda value: write_json(experiment.path / "oracle_cost.json", value))
        def persist():
            write_json(experiment.path / "attribution.json", report)
        persist()
        try:
            # Repeat hard-input parity using unchanged native reference methods.
            with torch.no_grad():
                native = counters.call("boltz", "attribution_seed_parity", "native_forward", lambda:
                    scorer.discrete_reference(branch.prepared.features, branch.prepared.coordinates), point.device)
                adapted = counters.call("boltz", "attribution_seed_parity", "forward", lambda:
                    branch(branch.prepared.seed.to_relaxed(decoder.spec)), point.device)
            if set(native) != set(adapted):
                raise ValueError("Native/adapted output sets differ")
            for key in native:
                torch.testing.assert_close(native[key], adapted[key], rtol=config.diagnostics.parity_rtol, atol=config.diagnostics.parity_atol)
            report["native_seed_parity"] = True
            (probe or probe_case)(decoder, point, branch, config, counters, target, weight,
                                 item_name, item, persist=persist, report=report)
            after = {"decoder_state_sha256": module_digest(decoder), "scorer_state_sha256": module_digest(scorer),
                     "context_sha256": branch.verify_context_unchanged(), "point_sha256": tensor_digest({"z": point})}
            report.update(identities_after=after, frozen_state_unchanged=identities == after,
                          parent_unchanged=all(_sha(p) == h for p, h in case["parent_artifacts_sha256"].items()))
            if not report["frozen_state_unchanged"] or not report["parent_unchanged"]:
                raise ValueError("Attribution altered scientific state or original evidence")
            if source_manifest(Path(plan["production_release"]))["sha256"] != plan["production_manifest_sha256"]:
                raise ValueError("Original source changed during diagnostic")
            persist()
            experiment.annotate(diagnostic_completed=True, old_gate_certified=False)
            return experiment.path
        except Exception as error:
            report.update(status="failed", diagnostic_completed=False, error=str(error), error_type=type(error).__name__)
            persist()
            raise


def run_study(plan_path, expected):
    """Serial children, unique run directories, retained failures and timeouts."""
    plan = load_plan(plan_path, expected)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    maximum_calls = 0
    for case in plan["cases"]:
        path = Path(case["parent_run"]) / case["arm"] / f"gate-{case['step']:04d}.json"
        _, item = failed_item(_read(authenticated_artifact(case, path)))
        failed_random = sum(name != "gradient" and result.get("passed") is False
                            for name, result in item["direction_results"].items())
        # Six baselines, two hard-seed parity calls, both modes/signs at nine
        # scales for each failed random direction and one gradient control.
        maximum_calls += REPEATS * 2 + 2 + (1 + failed_random) * 9 * 4
    if maximum_calls > 312:
        raise ValueError("Plan exceeds the declared 312-scorer-call investigation budget")
    inputs = {"plan": plan, "plan_sha256": expected, "diagnostic_module_sha256": _sha(__file__),
              "maximum_scorer_calls": maximum_calls, "maximum_child_seconds": CASE_TIMEOUT_SECONDS * len(plan["cases"]),
              "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], inputs, Path.cwd()) as experiment:
        rows = []
        for case in plan["cases"]:
            result = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(plan_path).resolve()),
                "--sha256", expected, "--child", case["id"], "--child-output", str(experiment.path / case["id"])],
                experiment.path / f"{case['id']}.log", timeout_seconds=CASE_TIMEOUT_SECONDS)
            completed = False
            if not result["timed_out"] and result["returncode"] == 0 and len(result["run_paths"]) == 1:
                child = Path(result["run_paths"][0])
                evidence = _read(child / "attribution.json")
                completed = (_read(child / "run.json")["status"] == "completed" and evidence.get("status") == "completed"
                             and evidence.get("diagnostic_completed") is True and evidence.get("frozen_state_unchanged") is True
                             and evidence.get("parent_unchanged") is True)
            rows.append({"case_id": case["id"], "diagnostic_completed": completed, "execution": result,
                         "old_gate_certified": False})
            write_json(experiment.path / "cases.json", rows)
        experiment.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in rows),
                            diagnostic_cases_requested=len(rows), old_gates_certified=False)
        return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not _hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Attribution requires an immutable DTU HPC allocation")
    plan = load_plan(args.plan, args.sha256)
    if args.child is not None:
        matches = [case for case in plan["cases"] if case["id"] == args.child]
        if len(matches) != 1 or args.child_output is None:
            raise ValueError("Unknown diagnostic child or missing unique output root")
        print(run_case(plan, matches[0], args.child_output), flush=True)
    else:
        if args.child_output is not None:
            raise ValueError("Child output is only valid with --child")
        print(run_study(args.plan, args.sha256), flush=True)


if __name__ == "__main__":
    main()
