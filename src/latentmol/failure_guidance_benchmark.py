"""Allocated native evidence for the separately versioned repair objective.

The original eighteen production benchmark gates remain authoritative. This
module adds matched failed-row coverage, an actual decoder VJP, a frozen-target
directional check, and retention relative to the STARTING chemistry checkpoint.
Every benchmark update is disposable. No benchmark checkpoint is training.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import math
import os
from pathlib import Path
import time

import torch

from . import chemistry_continuation as continuation
from .atom_preserving_chemistry import repair_loss
from .experiments import file_sha256, write_json
from .failure_guidance_training import FailureTeacherBatch
from .repair_failure_guidance import FailureGuidanceConfig, failure_aware_repair_loss
from .repair_progress import ProgressSearchConfig, search_progress_repair
from .chemistry_runtime import epoch_fit_seconds
from .representation import DecoderOutput


VERSION = "failed-repair-native-readiness-v1"
PANEL_COUNT = 1024
PANEL_SEED = 123904017
MINIMUM_ROW_GRADIENT = 1e-10


def _fields(decoded):
    return decoded.atoms, decoded.charges, decoded.bonds


def _valid_self_gradient(decoded, base, spec, policy, *, target_weighting=None):
    """Actual decoder VJP and float64 FD for detached strict-valid self-labels."""
    if policy["weight"] <= 0:
        raise ValueError("Native self-guidance diagnostics require a positive declared pilot coefficient")
    from .chemistry_valid_self_guidance import prepare_valid_self_guidance, valid_prior_self_nll
    cpu = DecoderOutput(*(value.detach().cpu().clone() for value in _fields(decoded)))
    prepared = prepare_valid_self_guidance(cpu, base, spec)
    weighting_args = {"target_weighting": target_weighting} if target_weighting is not None else {}
    result = valid_prior_self_nll(decoded, prepared, spec, policy, **weighting_args)
    # Head gradients directly verify both selected and excluded rows. The
    # caller separately evaluates the VJP into the actual decoder parameters.
    gradients = torch.autograd.grad(result["weighted_loss"], _fields(decoded), retain_graph=True,
                                    allow_unused=True)
    norm = torch.stack([(torch.zeros_like(value) if gradient is None else gradient)
        .detach().double().flatten(1).square().sum(1)
        for value, gradient in zip(_fields(decoded), gradients)]).sum(0).sqrt()
    # Undo the all-attempt mean and explicit weak coefficient for row-level
    # path diagnostics. Correctly saturated targets need no confidence force.
    norm = norm * len(norm) / policy["weight"]
    eligible = set(prepared.eligible_rows)
    mask = torch.tensor([row in eligible for row in range(len(norm))], device=norm.device)
    from .sampling import hard_graph_log_prob_terms
    with torch.no_grad():
        target = prepared.graph.to(decoded.atoms.device)
        terms = hard_graph_log_prob_terms(decoded, target, spec)
        count = (target.atoms != 0).sum(-1).to(decoded.atoms.dtype)
        decisions = spec.max_nodes + count + count*(count-1)/2
        uncertainty = -(terms["atoms"]+terms["charges"]+terms["bonds"])/decisions
        if target_weighting is not None:
            from .chemistry_target_weighting import fixed_hard_target_ce
            uncertainty = fixed_hard_target_ce(decoded, target, spec, target_weighting)
        informative = mask & (uncertainty > 1e-8)
    record = dict(attempts=len(norm), eligible=len(eligible),
        informative=int(informative.sum()), saturated=int((mask & ~informative).sum()),
        selected_nonzero=int((norm[informative] > MINIMUM_ROW_GRADIENT).sum()),
        excluded_zero=bool((norm[~mask] == 0).all()),
        finite=bool(torch.isfinite(norm).all()), receipt=prepared.receipt)
    fd = None
    if bool(informative.any()):
        values = DecoderOutput(*(value.detach().double().requires_grad_() for value in _fields(decoded)))
        loss = valid_prior_self_nll(values, prepared, spec, policy, **weighting_args)["weighted_loss"]
        derivatives = torch.autograd.grad(loss, _fields(values), allow_unused=True)
        generator = torch.Generator(device=values.atoms.device).manual_seed(123017)
        directions = [torch.randn(value.shape, generator=generator, device=value.device,
                                  dtype=value.dtype) for value in _fields(values)]
        length = torch.stack([value.square().sum() for value in directions]).sum().sqrt()
        directions = [value / length for value in directions]
        analytic = sum((gradient * direction).sum() for gradient, direction in zip(derivatives, directions)
                       if gradient is not None)
        epsilon = 1e-3
        perturbed = [DecoderOutput(*(value.detach() + sign * epsilon * direction
            for value, direction in zip(_fields(values), directions))) for sign in (1, -1)]
        plus, minus = [valid_prior_self_nll(value, prepared, spec, policy, **weighting_args)["weighted_loss"]
                       for value in perturbed]
        numeric = (plus-minus)/(2*epsilon)
        fd = dict(passed=bool(torch.isfinite(analytic) & torch.isfinite(numeric)
            & torch.isclose(analytic, numeric, rtol=1e-4, atol=1e-8)) and abs(float(analytic)) > 1e-12,
            analytic=float(analytic), finite_difference=float(numeric), epsilon=epsilon,
            rtol=1e-4, atol=1e-8, scope="Fixed strictly valid self-targets, float64 logits")
    return result["weighted_loss"], record, fd


def _weighted_repair_check(decoded, base, spec, target_weighting):
    """A native fixed-witness derivative check, with no new teacher/search calls.

    Use a normalized true derivative as the FD direction to avoid declaring a
    nearly orthogonal random vector an informative gradient check. Only logits
    are perturbed; every accepted witness and all-attempt denominator stay fixed.
    """
    values = DecoderOutput(*(v.detach().double().requires_grad_() for v in _fields(decoded)))
    loss = repair_loss(values, base, spec, target_weighting=target_weighting)
    derivatives = torch.autograd.grad(loss, _fields(values), allow_unused=True)
    derivatives = [torch.zeros_like(v) if g is None else g for v,g in zip(_fields(values),derivatives)]
    length = torch.stack([g.square().sum() for g in derivatives]).sum().sqrt()
    if not bool(torch.isfinite(length)) or float(length) <= MINIMUM_ROW_GRADIENT:
        raise ValueError("Weighted accepted repair did not supply an informative native gradient")
    directions = [g/length for g in derivatives]
    analytic = sum((g*d).sum() for g,d in zip(derivatives,directions))
    epsilon=1e-3
    perturbed=[DecoderOutput(*(v.detach()+sign*epsilon*d for v,d in zip(_fields(values),directions)))
               for sign in (1,-1)]
    plus,minus=[repair_loss(v,base,spec,target_weighting=target_weighting) for v in perturbed]
    numeric=(plus-minus)/(2*epsilon)
    return dict(passed=bool(torch.isfinite(numeric) & torch.isclose(analytic,numeric,rtol=1e-4,atol=1e-8)),
        analytic=float(analytic),finite_difference=float(numeric),epsilon=epsilon,rtol=1e-4,atol=1e-8,
        policy=dict(target_weighting),accepted_targets=sum(t is not None for t in base.targets),
        attempted=len(base.targets),scope="Native frozen repair targets, weighted CE, float64 logits; no search during FD")


def _frozen_directional_check(decoded, prepared, spec, config):
    """Differentiate the same frozen target; do not rerun search at +/-epsilon."""
    values = DecoderOutput(*(v.detach().double().requires_grad_() for v in _fields(decoded)))
    def loss(output):
        result = failure_aware_repair_loss(output, prepared, spec, config)
        return result["total"] - result["valid_full_nll"]
    value = loss(values)
    gradients = torch.autograd.grad(value, _fields(values), allow_unused=True)
    generator = torch.Generator(device=values.atoms.device).manual_seed(123017)
    directions = [torch.randn(v.shape, generator=generator, device=v.device, dtype=v.dtype) for v in _fields(values)]
    norm = torch.stack([v.square().sum() for v in directions]).sum().sqrt()
    directions = [v / norm for v in directions]
    analytic = sum((g * direction).sum() for g, direction in zip(gradients, directions) if g is not None)
    epsilon = 1e-3
    plus = loss(DecoderOutput(*(v.detach() + epsilon * d for v, d in zip(_fields(values), directions))))
    minus = loss(DecoderOutput(*(v.detach() - epsilon * d for v, d in zip(_fields(values), directions))))
    numeric = (plus - minus) / (2 * epsilon)
    return dict(passed=bool(torch.isfinite(analytic) & torch.isfinite(numeric)
                            & torch.isclose(analytic, numeric, rtol=1e-4, atol=1e-8))
                            and abs(float(analytic)) > 1e-12,
                analytic=float(analytic), finite_difference=float(numeric), epsilon=epsilon,
                rtol=1e-4, atol=1e-8, dtype="float64", scope="native logits, fixed full preparation, extra guidance only")


def _search_diagnostics(tasks, teacher, spec):
    """Small fixed failed-row subset; atom-edited variants cannot be promoted."""
    observations = []
    for name, policy in (("bond_depth6", ProgressSearchConfig()),
                         ("bond_depth12", ProgressSearchConfig(max_edits=12)),
                         ("atom_depth6_diagnostic_only", ProgressSearchConfig(allow_atom_edits=True))):
        started = time.perf_counter()
        scheduled = [(*task, spec, policy) for task in tasks]
        pool = getattr(teacher, "_pool", None)
        results = pool.map_progress(scheduled) if pool is not None else [search_progress_repair(*task) for task in scheduled]
        counts, atom_edits, sulfur_before, sulfur_after = {}, 0, 0, 0
        sulfur = spec.atomic_numbers.index(16) if 16 in spec.atomic_numbers else None
        for result in results:
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            target = result["witness"]
            if target is not None:
                target.verify(spec)
                atom_edits += sum(a != b for a, b in zip(target.original[0], target.labels[0]))
                sulfur_before += sum(a == sulfur for a in target.original[0]) if sulfur is not None else 0
                sulfur_after += sum(a == sulfur for a in target.labels[0]) if sulfur is not None else 0
        observations.append(dict(id=name, policy=policy.record(), attempted=len(tasks), status_counts=counts,
            checks=sum(row["checks"] for row in results), elapsed_seconds=time.perf_counter() - started,
            target_atom_edits=atom_edits, retained_target_sulfur_before=sulfur_before,
            retained_target_sulfur_after=sulfur_after,
            promotion_eligible=False, scope="matched bounded target-search diagnostic, not a training arm"))
    return observations


def native_panel(plan):
    """Read authentic current weights; observe gradients without optimizer steps."""
    report = dict(version=VERSION, status="failed", plan_sha256=plan["_plan_sha256"],
                  archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"],
                  job_id=os.environ.get("LSB_JOBID"), panel_count=PANEL_COUNT, panel_seed=PANEL_SEED)
    with ExitStack() as resources:
        model, reference, optimizer, config, data, cohorts, generators, teacher, epoch, updates, _, _ = continuation.setup(plan)
        resources.callback(teacher.close)
        composition = continuation.composition_reference(data["train"])
        model.eval()
        # These are fixed evaluation draws, distinct from fresh-prior training RNG.
        prior_generator = torch.Generator().manual_seed(PANEL_SEED)
        latent = torch.randn(PANEL_COUNT, model.config.latent, generator=prior_generator)
        device = next(model.parameters()).device
        failure = plan["protocol"].get("failure_guidance")
        loss_config = None if failure is None else FailureGuidanceConfig(**failure["loss"])
        counts = dict(attempts=0, original_accepted=0, already_valid=0, original_censored=0,
                      unsupported=0, extra_valid=0, partial=0, failed_with_nonzero_gradient=0,
                      failed_without_nonzero_gradient=0, original_failed=0)
        norms, decoder_norms, fd, tasks, witness_atom_edits = [], [], None, [], 0
        snapshot_chain = "0" * 64
        from .chemistry_short_continuation import validate_valid_prior_self_guidance
        self_policy = validate_valid_prior_self_guidance(plan["protocol"].get("valid_prior_self_guidance"))
        self_rows, self_decoder_norms, self_fd = [], [], None
        from .chemistry_target_weighting import validate_chemistry_target_weighting
        weighting = validate_chemistry_target_weighting(plan["protocol"].get("chemistry_target_weighting"))
        weighting_args = {"target_weighting": weighting} if weighting is not None else {}
        weighted_repair_fd, weighted_repair_norm = None, None
        wall = time.monotonic()
        for start in range(0, PANEL_COUNT, plan["protocol"]["microbatch_size"]):
            stop = min(start + plan["protocol"]["microbatch_size"], PANEL_COUNT)
            decoded = model.decode(latent[start:stop].to(device))
            batch = teacher.build_targets(decoded, candidate_ids=list(range(start, stop)), update=updates)
            base = batch.base if isinstance(batch, FailureTeacherBatch) else batch
            for attempt in base.receipt["attempts"]:
                counts["attempts"] += 1
                key = {"accepted": "original_accepted", "skipped_already_valid": "already_valid",
                       "censored": "original_censored", "unsupported": "unsupported"}.get(attempt["status"])
                if key is None:
                    raise ValueError("Native panel encountered a repair error")
                counts[key] += 1
                snapshot_chain = continuation.digest(dict(previous=snapshot_chain,
                    raw=attempt["raw_sha256"], probabilities=attempt["probability_sha256"]))
            if weighting is not None and weighted_repair_fd is None and any(t is not None for t in base.targets):
                scalar = repair_loss(decoded, base, model.spec, **weighting_args)
                derivatives = torch.autograd.grad(scalar, tuple(model.decoder.parameters()),
                                                   retain_graph=True, allow_unused=True)
                weighted_repair_norm = math.sqrt(sum(float(g.detach().double().square().sum())
                                                     for g in derivatives if g is not None))
                weighted_repair_fd = _weighted_repair_check(decoded, base, model.spec, weighting)
            if self_policy is not None:
                self_loss, self_record, directional = _valid_self_gradient(decoded, base, model.spec, self_policy,
                                                                          **weighting_args)
                values = torch.autograd.grad(self_loss, tuple(model.decoder.parameters()), allow_unused=True)
                self_decoder_norms.append(math.sqrt(sum(float(value.detach().double().square().sum())
                                                       for value in values if value is not None)))
                self_rows.append(self_record)
                if self_fd is None and directional is not None:
                    self_fd = directional
            if failure is None:
                counts["original_failed"] = counts["original_censored"] + counts["unsupported"]
                continue
            if not isinstance(batch, FailureTeacherBatch):
                raise ValueError("Configured new objective did not reach its production teacher")
            prepared = batch.prepared
            result = failure_aware_repair_loss(decoded, prepared, model.spec, loss_config)
            additional = result["total"] - result["valid_full_nll"]
            # Preserve the legacy valid-witness NLL separately; never count a
            # partial target or local-risk gradient as a valid repair witness.
            if not torch.equal(result["valid_full_nll"], repair_loss(decoded, base, model.spec)):
                raise ValueError("Original accepted-witness NLL changed")
            parameters = tuple(model.decoder.parameters())
            gradients = torch.autograd.grad(additional, (*_fields(decoded), *parameters), allow_unused=True)
            head_gradients = [torch.zeros_like(v) if g is None else g for v, g in zip(_fields(decoded), gradients[:3])]
            per_row = torch.stack([g.detach().double().flatten(1).square().sum(1) for g in head_gradients]).sum(0).sqrt()
            per_row = per_row * (stop - start)  # Undo all-attempt denominator for comparable per-row norms.
            decoder_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients[3:] if g is not None))
            if not torch.isfinite(per_row).all() or not math.isfinite(decoder_norm):
                raise ValueError("Additional native guidance produced nonfinite gradients")
            decoder_norms.append(decoder_norm)
            for row, item in enumerate(prepared.rows):
                if item is None:
                    continue
                counts["original_failed"] += 1
                norm = float(per_row[row]); norms.append(norm)
                counts["failed_with_nonzero_gradient" if norm > MINIMUM_ROW_GRADIENT else "failed_without_nonzero_gradient"] += 1
                if item.witness is not None:
                    counts["extra_valid" if item.witness.kind == "valid" else "partial"] += 1
                    witness_atom_edits += sum(a != b for a, b in zip(item.witness.original[0], item.witness.labels[0]))
            if fd is None and bool((per_row > MINIMUM_ROW_GRADIENT).any()):
                fd = _frozen_directional_check(decoded, prepared, model.spec, loss_config)
            # Retain at most64 identical failed sources for the deeper/atom
            # search diagnostic. All such work is reported separately from
            # synchronized production-update timing.
            if failure["progress_search"] is not None and len(tasks) < 64:
                cpu = DecoderOutput(*(v.detach().cpu() for v in _fields(decoded)))
                hard = cpu.harden(model.spec)
                probabilities = (cpu.atoms.softmax(-1), cpu.charges.softmax(-1), cpu.symmetric_bond_logits.softmax(-1))
                for row, item in enumerate(prepared.rows):
                    if item is not None and len(tasks) < 64:
                        raw = tuple(v[row].numpy().copy() for v in (hard.atoms, hard.charges, hard.bonds))
                        p = tuple(v[row].numpy().copy() for v in probabilities)
                        tasks.append((raw, p))
        starting_population = continuation.population_panel(model, plan["protocol"]["benchmark_panel_count"],
            plan["protocol"]["evaluation_seed"], 32, composition)
        starting_rec = continuation.validation(model, data["val"], config,
            plan["protocol"]["validation_count"], plan["protocol"]["evaluation_seed"])
        starting_retention_population = continuation.population_panel(model, plan["protocol"]["monitor_prior_count"],
            plan["protocol"]["evaluation_seed"], 32, composition)
        diagnostics = _search_diagnostics(tasks, teacher, model.spec) if tasks else []
        report.update(status="observed", completed_epoch=epoch, updates=updates, counts=counts,
            source_snapshot_chain_sha256=snapshot_chain, native_directional_check=fd,
            failed_row_gradient_norms=norms, minimum_useful_gradient=MINIMUM_ROW_GRADIENT,
            additional_decoder_gradient_norms=decoder_norms, complete_target_atom_edits=witness_atom_edits,
            starting_population=starting_population, starting_reconstruction=starting_rec,
            starting_retention_population=starting_retention_population,
            search_diagnostics=diagnostics, elapsed_seconds=time.monotonic() - wall,
            diagnostics_scope="fixed prior evaluation points, no optimizer updates; not training or a full epoch")
        if self_policy is not None:
            report["valid_prior_self_guidance"] = dict(policy=self_policy, rows=self_rows,
                decoder_gradient_norms=self_decoder_norms, directional_check=self_fd)
        if weighting is not None:
            report["chemistry_target_weighting"] = dict(policy=weighting,
                repair_decoder_gradient_norm=weighted_repair_norm, repair_directional_check=weighted_repair_fd)
        if plan["protocol"].get("valid_prior_reference_guidance") is not None:
            from .chemistry_valid_reference_native import native_reference_check
            report["valid_prior_reference_guidance"] = native_reference_check(plan,model,reference,optimizer,
                config,data,cohorts,generators,teacher,epoch,updates,composition)
            report["diagnostics_scope"] = "Fixed prior evaluation panel plus two fully restored same-start native updates and a fixed invalid-live VJP; no training"
            report["elapsed_seconds"] = time.monotonic()-wall
    return report


def benchmark_candidate(plan):
    """All original gates, plus explicitly separate new-objective readiness."""
    root = Path(plan["campaign"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "failure_readiness.json"
    if path.exists():
        raise FileExistsError("Preserve every prior readiness outcome; create a sealed new attempt")
    report = dict(version=VERSION, status="failed", plan_sha256=plan["_plan_sha256"], job_id=os.environ.get("LSB_JOBID"),
                  archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"])
    try:
        panel = native_panel(plan)
        report["native_panel"] = panel
        write_json(path, report)
        # Production setup reconstructs the identical model/optimizer/RNG start;
        # none of the panel's VJPs or evaluation draws are promoted.
        continuation.benchmark(plan)
        import json
        native = json.loads((root / "benchmark_gate.json").read_text())
        before, after = panel["starting_population"], native["updated_population"]
        gates = continuation.retention_checks(before, after, panel["starting_reconstruction"],
            native["updated_reconstruction"], plan["protocol"])
        gates["raw_prior_validity_retention"] = after["raw_argmax_validity"] >= before["raw_argmax_validity"] - .02
        gates["all_original_native_gates"] = native["status"] == "passed" and all(native["gates"].values())
        conservative_seconds = max(native["projected_epoch_seconds"],
            max(row["update_seconds"] for row in native["updates"]) * plan["training_molecules"] / 256 + 600.)
        gates["conservative_complete_epoch_fits"] = epoch_fit_seconds(plan, conservative_seconds) < plan["protocol"]["segment_wall_seconds"]
        if plan["protocol"].get("failure_guidance") is not None:
            counts = panel["counts"]
            gates.update(failed_rows_observed=counts["original_failed"] > 0,
                failed_gradient_coverage=counts["failed_with_nonzero_gradient"] >= .9 * max(1, counts["original_failed"]),
                zero_guidance_reduced=counts["failed_with_nonzero_gradient"] > 0,
                native_extra_decoder_vjp=any(value > MINIMUM_ROW_GRADIENT for value in panel["additional_decoder_gradient_norms"]),
                frozen_target_directional_derivative=panel["native_directional_check"] is not None and panel["native_directional_check"]["passed"],
                no_atom_edit_shortcut=panel["complete_target_atom_edits"] == 0,
                production_extra_gradient=any((row.get("failure_only_decoder_gradient_norm") or 0.) > MINIMUM_ROW_GRADIENT
                                              for row in native["updates"]))
        if plan["protocol"].get("valid_prior_self_guidance") is not None:
            observed = panel["valid_prior_self_guidance"]
            gates.update(valid_self_rows_observed=sum(row["eligible"] for row in observed["rows"]) > 0,
                valid_self_gradient_coverage=all(row["finite"] and row["selected_nonzero"] == row["informative"]
                                                 and row["excluded_zero"] for row in observed["rows"]),
                valid_self_native_decoder_vjp=any(value > MINIMUM_ROW_GRADIENT
                                                  for value in observed["decoder_gradient_norms"]),
                valid_self_directional_derivative=observed["directional_check"] is not None
                                                  and observed["directional_check"]["passed"],
                valid_self_production_gradient=any((row.get("valid_prior_self_decoder_gradient_norm") or 0.)
                                                   > MINIMUM_ROW_GRADIENT for row in native["updates"]))
        if plan["protocol"].get("chemistry_target_weighting") is not None:
            weighting = plan["protocol"]["chemistry_target_weighting"]
            observed = panel["chemistry_target_weighting"]
            gates.update(weighted_repair_native_decoder_vjp=(observed["repair_decoder_gradient_norm"] or 0.) > MINIMUM_ROW_GRADIENT,
                weighted_repair_directional_derivative=observed["repair_directional_check"] is not None
                                                      and observed["repair_directional_check"]["passed"],
                weighted_target_policy_bound=observed["policy"] == weighting
                    and native.get("chemistry_target_weighting") == weighting
                    and all(row.get("chemistry_target_weighting") == weighting for row in native["updates"]))
            report["chemistry_target_weighting"] = dict(weighting)
        if plan["protocol"].get("valid_prior_reference_guidance") is not None:
            observed=panel["valid_prior_reference_guidance"]
            gates.update(observed["gates"])
            from .chemistry_valid_reference_policy import validate_valid_reference_test
            report["valid_reference_test"] = validate_valid_reference_test(plan)
            report["valid_prior_reference_guidance"] = dict(plan["protocol"]["valid_prior_reference_guidance"])
            expected=observed["reference_before"]
            if (native.get("valid_prior_reference_guidance") != plan["protocol"]["valid_prior_reference_guidance"]
                    or native.get("valid_reference_anchor_before") != expected
                    or native.get("valid_reference_anchor_after") != expected
                    or any(row.get("valid_prior_reference_guidance",{}).get("reference_identity") != dict(
                        epoch=36,checkpoint_sha256=expected["reference_sha256"],model_sha256=expected["model_sha256"])
                        for row in native["updates"])):
                raise ValueError("Actual native updates lost their frozen original36 objective/source binding")
        report.update(status="passed" if all(gates.values()) else "failed", gates=gates,
            conservative_epoch_seconds=conservative_seconds,
            native_benchmark_sha256=file_sha256(root / "benchmark_gate.json"),
            benchmark_weights_discarded=True, scientific_gate_schema="separate-readiness-v1-keeps-original-eighteen")
        write_json(path, report)
        if report["status"] != "passed":
            raise RuntimeError("Supplementary new-objective readiness failed")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = continuation.read_plan(args.plan, args.plan_sha256)
    plan["_plan_sha256"] = args.plan_sha256
    benchmark_candidate(plan)


if __name__ == "__main__":
    main()
