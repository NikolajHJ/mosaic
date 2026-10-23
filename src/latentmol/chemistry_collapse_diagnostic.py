"""Replay source131's first100 updates and measure finite-step loss at its spikes.

This diagnostic preserves the original repair objective, ramp, inherited AdamW
state and production shuffled stream. At updates39 and96 it decomposes the
weighted gradients and compares the actual full and quarter parameter proposal
using identical posterior/prior draws and stopped teacher targets. No target is
researched at a trial weight. Full-proposal weights and moments are restored
before the replay continues. All100 updates are discarded, even on failure.

This is mechanism evidence, never readiness or a training checkpoint. Numerical
entrypoints require an allocated GPU through the authenticated setup helper.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time

import torch

from . import chemistry_continuation as c
from .atom_preserving_chemistry import composition_statistics, population_js, repair_loss
from .chemistry_execution_validation import _snapshot, _restore
from .chemistry_guarded_validation import _compact_metrics
from .chemistry_recovery_diagnostic import (_loader, detailed_observation, panel_probabilities,
    output_drift, chemical_transitions, gradient_displacement, grouped_gradient_displacement)
from .experiments import ExperimentRun, file_sha256, write_json
from .losses import latent_kl
from .train import learning_rate_at_epoch, state_digest

VERSION = "epoch36-collapse-finite-step-v1"


def validate_policy(plan):
    value = plan.get("collapse_diagnostic")
    if (type(value) is not dict or set(value) != {"version", "updates", "instrument_updates", "source_rows"}
            or value.get("version") != VERSION or type(value.get("updates")) is not int
            or value["updates"] != 100 or value.get("instrument_updates") != [39, 96]
            or any(type(index) is not int for index in value["instrument_updates"])
            or type(value.get("source_rows")) is not dict
            or set(value["source_rows"]) != {"path", "sha256"}
            or type(value["source_rows"]["path"]) is not str
            or not PurePosixPath(value["source_rows"]["path"]).is_absolute()
            or type(value["source_rows"]["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}",value["source_rows"]["sha256"]) is None
            or plan.get("starting_checkpoint", {}).get("epoch") != 36
            or plan["protocol"].get("failure_guidance") is not None
            or plan["protocol"].get("valid_prior_self_guidance") is not None):
        raise ValueError("Require the exact original-objective100-update spike replay from epoch36")
    return deepcopy(value)


class ComponentGradients:
    """Detached weighted-term gradients; never alter the production .grad buffers."""
    def __init__(self, model):
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.values = {}

    def __call__(self, branch, terms):
        for term, scalar in terms.items():
            if not scalar.requires_grad:
                continue
            values = torch.autograd.grad(scalar, tuple(p for _, p in self.named),
                                         retain_graph=True, allow_unused=True)
            destination = self.values.setdefault(branch + "/" + term, {})
            for (name, _), value in zip(self.named, values):
                if value is not None:
                    value = value.detach().double()
                    if name in destination:
                        destination[name].add_(value)
                    else:
                        destination[name] = value.clone()

    def report(self, displacement, gradients):
        maximum = 0.
        for name, value in gradients.items():
            if value is None:
                continue
            pieces = [group[name] for group in self.values.values() if name in group]
            recovered = sum(pieces) if pieces else torch.zeros_like(value)
            maximum = max(maximum, float((recovered - value.double()).abs().max()))
            if not torch.allclose(recovered, value.double(), rtol=1e-4, atol=2e-7):
                raise ValueError("Weighted component gradients do not reconstruct the actual backward")
        return dict(maximum_gradient_reconstruction_error=maximum,
            terms={key: dict(all=gradient_displacement(values, displacement),
                             groups=grouped_gradient_displacement(values, displacement))
                   for key, values in self.values.items()})


@torch.no_grad()
def frozen_objective(model, reference, config, protocol, composition, context):
    """The actual scalar objective on one stopped-target/draw logical batch.

    Population JS is recomputed as a nonlinear scalar from the aggregate head
    counts; the streamed trainer's cotangent surrogate is NOT used as a finite
    objective. Teacher search, argmax selection and new RNG draws are absent.
    """
    device = next(model.parameters()).device
    count, prior_count = context["real_points"], context["prior_points"]
    terms = dict(reconstruction=0., kl=0., repair_real=0., repair_prior=0., occupancy=0.)
    totals = {"elements": torch.zeros(len(model.spec.atomic_numbers)-1, dtype=torch.float64, device=device),
              "charges": torch.zeros(len(model.spec.formal_charges), dtype=torch.float64, device=device)}
    for branch, total in (("real", count), ("prior", prior_count)):
        for start, stop in c.chunks(total, context["size"]):
            fraction = (stop-start)/total
            if branch == "real":
                target = c.hard_slice(context["hard"], start, stop)
                mu, logvar = model.encode(target.to_relaxed(model.spec))
                z = mu + context["epsilon"][start:stop] * (.5*logvar).exp()
            else:
                z = context["prior"][start:stop]
            decoded = model.decode(z)
            if branch == "real":
                terms["reconstruction"] += float(c.reconstruction_chunk(decoded, target, config.loss, context["denominator"]))
                terms["kl"] += context["beta"]*float(latent_kl(mu, logvar))*fraction
            weight = protocol["real_repair_weight"] if branch == "real" else protocol["prior_repair_weight"]
            terms["repair_"+branch] += context["coefficient"]*weight*float(
                repair_loss(decoded, context["targets"][branch, start], model.spec))*fraction
            statistics = composition_statistics(decoded, model.spec)
            for key in totals:
                totals[key] += getattr(statistics, key).double()*(1. if branch == "real" else context["prior_composition_weight"])
            if branch == "prior":
                occupancy = decoded.probabilities().occupancy
                old = reference.decode(z).probabilities().occupancy
                terms["occupancy"] += protocol["occupancy_anchor_weight"]*float((occupancy-old).square().mean())*fraction
    terms["element_js"] = protocol["element_js_weight"]*float(population_js(totals["elements"], composition["elements_distribution"]))
    terms["charge_js"] = protocol["charge_js_weight"]*float(population_js(totals["charges"], composition["charges_distribution"]))
    terms["total"] = sum(terms.values())
    if any(not math.isfinite(x) for x in terms.values()):
        raise FloatingPointError("Nonfinite frozen-target finite-step objective")
    return terms


def run_diagnostic(plan):
    policy = validate_policy(plan)
    source_path = Path(policy["source_rows"]["path"])
    if file_sha256(source_path) != policy["source_rows"]["sha256"]:
        raise ValueError("Historical100-update receipts changed")
    expected = [json.loads(line) for line in source_path.read_text().splitlines() if line.strip()]
    if len(expected) != 100:
        raise ValueError("Historical replay needs all100 ordered receipts")
    root, protocol = Path(plan["campaign"]), plan["protocol"]
    output = root / "collapse_diagnostic.json"
    if output.exists():
        raise FileExistsError("Preserve the previous diagnostic; use a new immutable attempt")
    started, initial = time.monotonic(), None
    # Two-hour diagnostic cap, distinct from the following proof's shared23h cap.
    deadline = started + 6600
    report = dict(version=VERSION, status="running", plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"],
        job_id=os.environ.get("LSB_JOBID"), policy=policy, training_updates_promoted=0,
        promotable=False, epoch_complete=False, spikes=[], panels={}, updates=[])
    with ExitStack() as resources, ExperimentRun(root/"collapse_diagnostic", "chemistry-collapse-mechanism",
            plan, Path(plan["release"])) as run:
        report["run_dir"] = str(run.path)
        try:
            proof = {}
            model, reference, optimizer, config, data, cohorts, generators, teacher, epoch, updates, _, _ = c.setup(
                plan, parent_authentication=proof)
            resources.callback(teacher.close)
            if epoch != 36 or updates != plan["starting_checkpoint"]["updates"]:
                raise ValueError("Collapse replay did not restore exact epoch36")
            initial = _snapshot(model, reference, optimizer, generators)
            report.update(parent_checkpoint_authentication=proof, starting_model_sha256=state_digest(model.state_dict()))
            composition = c.composition_reference(data["train"])
            overlap = bool((c.execution_settings(plan) or {}).get("overlap", False))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_epoch(config, epoch+1)
            baseline, baseline_rows = detailed_observation(model, data, config, protocol, composition, generators)
            report["panels"]["0"] = baseline
            loader = iter(_loader(data, cohorts, config, generators, "production", plan["seed"]))
            for relative in range(1, 101):
                if time.monotonic() > deadline:
                    raise TimeoutError("Targeted collapse replay runtime exhausted")
                hard = next(loader)
                kwargs = dict(deadline=deadline, overlap=overlap)
                if relative not in policy["instrument_updates"]:
                    metrics = c.chemistry_update(model, reference, optimizer, hard, config, protocol,
                        composition, teacher, generators, updates+relative-1, epoch+1, plan["parent_epoch"], **kwargs)
                else:
                    before = _snapshot(model, reference, optimizer, generators)
                    before_panel, before_rows = detailed_observation(model, data, config, protocol, composition, generators)
                    before_law = panel_probabilities(model, 1024, protocol["evaluation_seed"])
                    context, capture = {}, ComponentGradients(model)
                    metrics = c.chemistry_update(model, reference, optimizer, hard, config, protocol, composition,
                        teacher, generators, updates+relative-1, epoch+1, plan["parent_epoch"], do_step=False,
                        capture_context=context, component_observer=capture, **kwargs)
                    pre_loss = frozen_objective(model, reference, config, protocol, composition, context)
                    if not math.isclose(pre_loss["total"], metrics["total"], rel_tol=1e-5, abs_tol=2e-7):
                        raise ValueError("Frozen finite objective differs from native pre-update scalar")
                    gradients = {n: None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad, error_if_nonfinite=True)
                    optimizer.step()
                    full = _snapshot(model, reference, optimizer, generators)
                    displacement = {n: p.detach().double()-before["model"][n].double() for n,p in model.named_parameters()}
                    spike = dict(relative_update=relative, before_panel=before_panel, before_loss=pre_loss,
                        weighted_gradients=capture.report(displacement, gradients),
                        complete_gradient=gradient_displacement(gradients, displacement), trials=[])
                    del capture, gradients
                    for scale in (1., .25):
                        model.load_state_dict(full["model"], strict=True)
                        if scale != 1.:
                            with torch.no_grad():
                                for name, parameter in model.named_parameters():
                                    parameter.copy_((before["model"][name].double()+scale*displacement[name]).to(parameter))
                        loss = frozen_objective(model, reference, config, protocol, composition, context)
                        observation, rows = detailed_observation(model, data, config, protocol, composition, generators)
                        law = panel_probabilities(model, 1024, protocol["evaluation_seed"])
                        spike["trials"].append(dict(scale=scale, frozen_loss=loss,
                            finite_loss_change=loss["total"]-pre_loss["total"], observation=observation,
                            output_drift=output_drift(before_law, law),
                            chemical_transitions=chemical_transitions(before_rows, rows),
                            targets_resampled=False, extra_optimizer_steps=0))
                    _restore(full, model, reference, optimizer, generators)
                    if state_digest(model.state_dict()) != state_digest(full["model"]):
                        raise ValueError("Counterfactual did not restore the exact full proposal")
                    report["spikes"].append(spike)
                    del before, full, displacement, context, before_law
                row = _compact_metrics(metrics)
                old = expected[relative-1].get("metrics", expected[relative-1])
                for key in ("ordered_teacher_receipt_chain_sha256", "total", "pre_clip_gradient_norm"):
                    if key.endswith("sha256"):
                        same = old[key] == row[key]
                    else:
                        same = math.isclose(old[key], row[key], rel_tol=2e-5, abs_tol=2e-7)
                    if not same:
                        raise ValueError(f"Original production replay diverged at update{relative}: {key}")
                report["updates"].append(dict(relative_update=relative, metrics=row, historical_receipt_matches=True))
                if relative in (32,100):
                    observation, rows = detailed_observation(model, data, config, protocol, composition, generators)
                    report["panels"][str(relative)] = observation
                    report["panels"][str(relative)+"_transitions"] = chemical_transitions(baseline_rows, rows)
                write_json(output, report)
            report.update(status="completed_disposable_not_training", diagnostic_completed=True,
                scientific_readiness="Not established;100 replay updates are discarded")
        except BaseException as error:
            report.update(status="failed", diagnostic_completed=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if initial is not None:
                errors = []
                for action in (teacher.drain, teacher.clear_cache):
                    try:
                        action()
                    except BaseException as error:
                        errors.append(f"{type(error).__name__}: {error}")
                try:
                    _restore(initial, model, reference, optimizer, generators)
                    restored = _snapshot(model, reference, optimizer, generators)
                    checks = {key: c.nested_equal(value, restored[key]) for key,value in initial.items()}
                    report["restoration"] = dict(passed=all(checks.values()), checks=checks)
                    if not all(checks.values()):
                        errors.append("Full model/AdamW/RNG/gradient/mode restoration failed")
                except BaseException as error:
                    errors.append(f"{type(error).__name__}: {error}")
                if errors:
                    report.update(status="failed", diagnostic_completed=False, cleanup_errors=errors)
            report["elapsed_seconds"] = time.monotonic()-started
            write_json(output, report); write_json(run.path/"diagnostic.json", report)
            run.annotate(training_updates_promoted=0, promotable=False, epoch_complete=False, diagnostic_status=report["status"])
    if report.get("restoration", {}).get("passed") is not True:
        raise RuntimeError("Collapse diagnostic failed full restoration")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = c.read_plan(args.plan, args.plan_sha256)
    plan["_plan_sha256"] = args.plan_sha256
    run_diagnostic(plan)


if __name__ == "__main__":
    main()
