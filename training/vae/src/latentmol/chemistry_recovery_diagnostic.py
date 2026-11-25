"""Disposable epoch-36 optimizer/validity investigation, never training readiness.

The proposal uses one real production shuffled logical batch, the unchanged
loss/teacher and inherited AdamW state. Four fractions of that *same* parameter
displacement are observed at fixed prior coordinates. Two eight-update prefixes
then distinguish the old ordered benchmark stream from production shuffling.
Every arm restores parameters, moments, named gradients, modes and all RNGs.
There is no optimizer policy change, target resampling per scale, checkpoint
promotion, readiness receipt or training callback in this diagnostic.

UNTESTED until the sealed allocated CPU/CUDA fixtures and native probe run.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import math
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from . import chemistry_continuation as continuation
from .chemistry_execution_validation import _restore, _snapshot, native_execution_parity
from .chemistry_early_health import EarlyPanelCollector, validate_early_health_observation
from .chemistry_classification_validation import classification_execution_parity
from .chemistry_guarded_validation import _compact_metrics
from .data import collate_graphs
from .experiments import ExperimentRun, file_sha256, write_json
from .train import learning_rate_at_epoch, state_digest


VERSION = "epoch36-optimizer-mechanism-v1"
SCALES = (1., .5, .25, .125)


def validate_policy(plan):
    """A closed diagnostic scope cannot accidentally grow into a training run."""
    value = plan.get("recovery_diagnostic")
    if (type(value) is not dict or set(value) != {"version", "candidate_scales", "panel_count",
            "maximum_updates", "prefix_updates", "prefix_streams"}
            or value["version"] != VERSION
            or type(value["candidate_scales"]) is not list
            or any(type(x) is not float for x in value["candidate_scales"])
            or tuple(value["candidate_scales"]) != SCALES
            or type(value["panel_count"]) is not int or value["panel_count"] != 1024
            or type(value["maximum_updates"]) is not int or value["maximum_updates"] != 1
            or type(value["prefix_updates"]) is not int or value["prefix_updates"] != 8
            or value["prefix_streams"] != ["benchmark", "production"]):
        raise ValueError("Require the exact versioned one-proposal/two-eight-update diagnostic")
    if plan.get("starting_checkpoint", {}).get("epoch") != 36:
        raise ValueError("This diagnostic requires the authenticated completed epoch36")
    return deepcopy(value)


class BranchGradientCapture:
    """Observe real/prior full-loss VJPs without changing inputs or backward calls.

    Leaf post-accumulate hooks exclude the trainer's diagnostic autograd.grad
    calls, which do not accumulate .grad. Differences of cumulative gradients
    recover each streamed backward contribution. Decoder forwards identify the
    branch; no-grad teacher prepasses and frozen-reference forwards are ignored.
    The two branch totals include their actual global weights and all objective
    components. They do not isolate individual chemistry terms.
    """
    def __init__(self, model, real_points, prior_points, microbatch_size):
        self.model = model
        self.schedule = (["real"] * math.ceil(real_points / microbatch_size)
                         + ["prior"] * math.ceil(prior_points / microbatch_size))
        self.forward_count = 0
        self.branch = None
        self.previous = {}
        self.values = {key: {} for key in ("real", "prior")}
        self.handles = []

    def _forward(self, _module, _inputs, output):
        if not torch.is_grad_enabled():
            return
        if self.forward_count >= len(self.schedule):
            raise ValueError("Unexpected differentiable decoder forward in proposal capture")
        self.branch = self.schedule[self.forward_count]
        self.forward_count += 1

    def _accumulate(self, name, parameter):
        if self.branch is None or parameter.grad is None:
            raise ValueError("Gradient accumulated outside its declared decoder branch")
        current = parameter.grad.detach().double().clone()
        previous = self.previous.get(name)
        increment = current if previous is None else current - previous
        old = self.values[self.branch].get(name)
        self.values[self.branch][name] = increment if old is None else old + increment
        self.previous[name] = current

    def __enter__(self):
        self.handles.append(self.model.decoder.register_forward_hook(self._forward))
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.handles.append(parameter.register_post_accumulate_grad_hook(
                    lambda parameter, name=name: self._accumulate(name, parameter)))
        return self

    def __exit__(self, *_error):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def verify(self, full):
        if self.forward_count != len(self.schedule):
            raise ValueError("Incomplete real/prior proposal gradient schedule")
        maximum_error = 0.
        for name, gradient in full.items():
            pieces = [value[name] for value in self.values.values() if name in value]
            if gradient is None:
                if pieces:
                    raise ValueError("Captured a gradient for an unused parameter")
                continue
            if not pieces:
                raise ValueError("Missing accumulated branch gradient: " + name)
            recovered = sum(pieces)
            maximum_error = max(maximum_error, float((recovered - gradient.double()).abs().max()))
            if not torch.allclose(recovered, gradient.double(), rtol=1e-8, atol=1e-10):
                raise ValueError("Real/prior gradient decomposition differs from complete gradient")
        return dict(passed=True, maximum_absolute_error=maximum_error,
                    differentiable_decoder_forwards=self.forward_count)


def gradient_displacement(gradients, displacement):
    """First-order dot products, not a claim about finite-step objective descent."""
    dot = norm_g = norm_delta = 0.
    for name, delta in displacement.items():
        delta = delta.double()
        norm_delta += float(delta.square().sum())
        gradient = gradients.get(name)
        if gradient is not None:
            gradient = gradient.double()
            dot += float((gradient * delta).sum())
            norm_g += float(gradient.square().sum())
    if not all(math.isfinite(x) for x in (dot, norm_g, norm_delta)):
        raise FloatingPointError("Nonfinite gradient or AdamW parameter displacement")
    denominator = math.sqrt(norm_g * norm_delta)
    return dict(gradient_dot_displacement=dot, gradient_norm=math.sqrt(norm_g),
                displacement_norm=math.sqrt(norm_delta),
                cosine=dot / denominator if denominator else None,
                interpretation="Negative is first-order downhill; finite-step loss and validity are separate")


def grouped_gradient_displacement(gradients, displacement):
    """Keep decoder movement visible when encoder reconstruction/KL dominates."""
    return {group: gradient_displacement(gradients, {name: value for name, value in displacement.items()
                                                   if name.startswith(group + ".")})
            for group in ("encoder", "decoder")}


@torch.no_grad()
def apply_displacement(model, initial_state, displacement, scale, *, proposal_state=None):
    """Observe a fraction of one inherited AdamW proposal; never step moments twice."""
    if type(scale) is not float or scale not in SCALES:
        raise ValueError("Only the four declared parameter-displacement scales are supported")
    if scale == 1.:
        if proposal_state is None:
            raise ValueError("Scale one requires the exact saved AdamW proposal state")
        # FP32 subtraction/readdition can lose a small proposal under severe
        # cancellation. The full-step observation must use the actual weights.
        model.load_state_dict(proposal_state, strict=True)
        return
    model.load_state_dict(initial_state, strict=True)
    for name, parameter in model.named_parameters():
        parameter.add_(displacement[name], alpha=scale)


@torch.no_grad()
def panel_probabilities(model, count, seed):
    """Conditional categorical heads at the exact observer's fixed prior points.

    The bond head is symmetrized and each unordered active pair counted once.
    Occupancy gating is deliberately absent here: this measures decoder-output
    drift separately from the observer's actual hardening/chemical validity.
    """
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(count, model.config.latent, generator=generator)
    device = next(model.parameters()).device
    parts = {key: [] for key in ("atoms", "charges", "bonds")}
    with continuation.preserve_observation_state(model, {}):
        model.eval()
        for start, stop in continuation.chunks(count, 32):
            decoded = model.decode(z[start:stop].to(device))
            for key, value in (("atoms", decoded.atoms), ("charges", decoded.charges),
                               ("bonds", decoded.symmetric_bond_logits)):
                parts[key].append(value.softmax(-1).cpu())
    return {key: torch.cat(values) for key, values in parts.items()}


def output_drift(before, after):
    """TV and argmax flips relative to *starting* occupied slots/pairs.

    Inactive charge/bond labels are excluded using the fixed starting mask;
    atom TV still measures occupancy changes at every slot. Labels, head TV and
    chemical checks are separate observations, not validity certificates.
    """
    if set(before) != {"atoms", "charges", "bonds"} or before.keys() != after.keys():
        raise ValueError("Decoder drift requires all three categorical heads")
    if any(before[key].shape != after[key].shape for key in before):
        raise ValueError("Fixed-panel decoder shapes changed")
    active = before["atoms"].argmax(-1) != 0
    upper = torch.ones(active.shape[1], active.shape[1], dtype=torch.bool).triu(1)
    pair = active[:, :, None] & active[:, None, :] & upper
    masks = {"atoms": torch.ones_like(active), "charges": active, "bonds": pair}
    report = {}
    for key in before:
        if not bool(torch.isfinite(after[key]).all()):
            raise FloatingPointError("Nonfinite fixed-panel categorical probabilities")
        mask = masks[key]
        tv = .5 * (after[key].double() - before[key].double()).abs().sum(-1)
        flips = (before[key].argmax(-1) != after[key].argmax(-1)) & mask
        report[key] = dict(categorical_vectors=int(mask.sum()),
            mean_total_variation=float(tv[mask].mean()) if bool(mask.any()) else 0.,
            maximum_total_variation=float(tv[mask].max()) if bool(mask.any()) else 0.,
            argmax_label_flips=int(flips.sum()), graphs_with_label_flips=int(flips.flatten(1).any(-1).sum()))
    return report


def _loader(data, cohorts, config, generators, stream, seed):
    if stream == "production":
        return DataLoader(Subset(data["train"], cohorts["train"]), batch_size=config.batch_size,
            shuffle=True, generator=generators["train"], collate_fn=collate_graphs, num_workers=0)
    if stream == "benchmark":
        return DataLoader(Subset(data["train"], cohorts["train"][:8 * config.batch_size]),
            batch_size=config.batch_size, shuffle=False, collate_fn=collate_graphs,
            generator=torch.Generator().manual_seed(seed + 999), num_workers=0)
    raise ValueError("Undeclared disposable prefix stream")


def _sync():
    torch.cuda.synchronize()


class DiagnosticPanelCollector(EarlyPanelCollector):
    """Reuse native stage classification; retain labels for paired diagnostics only."""
    def __init__(self):
        super().__init__(4)
        self.labels = []

    def add(self, atoms, charges, bonds, spec, *, probabilities, candidate_id):
        super().add(atoms, charges, bonds, spec, probabilities=probabilities, candidate_id=candidate_id)
        self.labels.append({name: value.detach().cpu().clone() for name, value in
                            zip(("atoms", "charges", "bonds"), (atoms, charges, bonds))})


def detailed_observation(model, data, config, protocol, composition, generators):
    """The production observer's same loops, adding no extra chemistry or draws."""
    collector = DiagnosticPanelCollector()
    with continuation.preserve_observation_state(model, generators):
        population = continuation.population_panel(model, 1024, protocol["evaluation_seed"], 32,
                                                     composition, classification=collector)
        reconstruction = continuation.validation(model, data["val"], config, 256, protocol["evaluation_seed"])
    result = dict(population=population, reconstruction=reconstruction, classification=collector.summary())
    validate_early_health_observation(result)
    return result, collector


def benchmark_observation(model, protocol, composition, generators):
    """Explicit original 256-point raw panel, rather than assuming RNG prefixes."""
    collector = DiagnosticPanelCollector()
    with continuation.preserve_observation_state(model, generators):
        panel = continuation.population_panel(model, 256, protocol["evaluation_seed"], 32,
                                                composition, classification=collector)
    return panel, collector


def chemical_transitions(before, after):
    """Match every fixed coordinate; retain four actual graph pairs per transition."""
    if (len(before.records) != len(after.records) or len(before.labels) != len(before.records)
            or len(after.labels) != len(after.records)):
        raise ValueError("Fixed-panel classification/label correspondence was lost")
    counts, reasons, indices, examples = {}, {}, {}, {}
    for left, right, old_labels, new_labels in zip(before.records, after.records, before.labels, after.labels):
        if left["candidate_id"] != right["candidate_id"]:
            raise ValueError("Fixed-panel categorical identities changed")
        old_valid = left["classification"]["smiles_parse_valid"] is True
        new_valid = right["classification"]["smiles_parse_valid"] is True
        transition = ("valid" if old_valid else "invalid") + "_to_" + ("valid" if new_valid else "invalid")
        counts[transition] = counts.get(transition, 0) + 1
        indices.setdefault(transition, []).append(left["candidate_id"])
        kind = right["classification"]["failure_kind"]
        reason = kind + ":" + str(right["classification"]["reason"])
        bucket = reasons.setdefault(transition, {})
        bucket[reason] = bucket.get(reason, 0) + 1
        if old_valid == new_valid:
            continue
        retained = examples.setdefault(transition, [])
        if len(retained) < 4:
            changes = {}
            for key in old_labels:
                changed = old_labels[key] != new_labels[key]
                if key == "bonds":
                    changed &= torch.ones_like(changed, dtype=torch.bool).triu(1)
                changes[key] = [{"index": index.tolist(), "before": int(old_labels[key][tuple(index)]),
                                 "after": int(new_labels[key][tuple(index)])}
                                for index in changed.nonzero()]
            retained.append(dict(candidate_id=left["candidate_id"], before=left["classification"],
                after=right["classification"], categorical_label_changes=changes,
                before_labels={key: value.tolist() for key, value in old_labels.items()},
                after_labels={key: value.tolist() for key, value in new_labels.items()}))
    return dict(attempts=len(before.records), counts=counts, endpoint_reasons=reasons,
                candidate_ids=indices, bounded_changed_examples=examples,
                validity_rule="Same raw smiles_parse_valid as fixed population; not strict schema roundtrip")


def save_row_classifications(path, collector):
    """Preserve all paired native outcomes while bounding full graph examples."""
    rows = [{key: record[key] for key in ("candidate_id", "classification")} for record in collector.records]
    write_json(path, dict(version=VERSION, fixed_panel=True, rows=rows,
                         chemical_vs_representation_failures_separate=True))
    return dict(path=str(path), sha256=file_sha256(path), attempts=len(rows))


def run_diagnostic(plan):
    """Use authenticated production helpers; write no scientific promotion artifact."""
    policy = validate_policy(plan)
    root, protocol = Path(plan["campaign"]), plan["protocol"]
    root.mkdir(parents=True, exist_ok=True)
    output = root / "recovery_diagnostic.json"
    if output.exists():
        raise FileExistsError("Preserve prior diagnostic evidence; use a new immutable attempt")
    started = time.monotonic()
    deadline = started + 3000  # Reserve ten minutes inside the one-hour allocation.
    report = dict(version=VERSION, status="running", plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"],
        starting_checkpoint=plan["starting_checkpoint"], job_id=os.environ.get("LSB_JOBID"),
        training_updates_promoted=0, promotable=False, epoch_complete=False,
        policy=policy, timings={}, prefixes={}, trials=[], restoration=dict(passed=False))
    initial = model = reference = optimizer = generators = teacher = None
    with ExitStack() as resources, ExperimentRun(root / "recovery_diagnostic", "chemistry-recovery-diagnostic",
            plan, Path(plan["release"])) as run:
        report["run_dir"] = str(run.path)
        try:
            parent_proof = {}
            tick = time.monotonic()
            model, reference, optimizer, config, data, cohorts, generators, teacher, epoch, updates, _, _ = continuation.setup(
                plan, parent_authentication=parent_proof)
            resources.callback(teacher.close)
            if epoch != 36 or updates != plan["starting_checkpoint"]["updates"]:
                raise ValueError("Diagnostic did not restore the authenticated completed epoch36")
            initial = _snapshot(model, reference, optimizer, generators)
            report.update(parent_checkpoint_authentication=parent_proof, starting_model_sha256=state_digest(model.state_dict()),
                stored_learning_rates=[group["lr"] for group in optimizer.param_groups],
                production_epoch=epoch + 1, production_learning_rate=learning_rate_at_epoch(config, epoch + 1))
            report["timings"]["setup_seconds"] = time.monotonic() - tick
            tick = time.monotonic()
            composition = continuation.composition_reference(data["train"])
            write_json(run.path / "training_composition.json", composition)
            report["timings"]["composition_seconds"] = time.monotonic() - tick
            overlap = bool((continuation.execution_settings(plan) or {}).get("overlap", False))
            for group in optimizer.param_groups:
                group["lr"] = report["production_learning_rate"]
            tick = time.monotonic()
            report["native_execution_parity"] = native_execution_parity(model, reference, optimizer, data["train"],
                cohorts, config, protocol, composition, teacher, generators, epoch, updates,
                run.path / "native_execution_parity.json", overlap=overlap, phase_start_epoch=plan["parent_epoch"])
            report["timings"]["native_execution_parity_seconds"] = time.monotonic() - tick
            # Do not keep the warmed native-parity pool open while the matched
            # classifier experiment creates its own sequential 12-worker arms.
            teacher.close()
            teacher = None
            _restore(initial, model, reference, optimizer, generators)
            for group in optimizer.param_groups:
                group["lr"] = report["production_learning_rate"]
            runtime_hard = next(iter(_loader(data, cohorts, config, generators, "production", plan["seed"])))
            tick = time.monotonic()
            report["classification_execution_parity"] = classification_execution_parity(model, reference, optimizer,
                runtime_hard, config, protocol, composition, generators, epoch, updates, plan["parent_epoch"],
                run.path / "classification_execution_parity.json", overlap=overlap)
            report["timings"]["classification_execution_parity_seconds"] = time.monotonic() - tick
            # Recreate the exact configured teacher only because its pool was
            # closed for resource isolation; no target/search policy changes.
            from .failure_guidance_training import build_repair_teacher
            teacher = build_repair_teacher(model.spec, protocol, continuation.execution_settings(plan),
                classification_execution=plan.get("classification_execution"))
            resources.callback(teacher.close)
            _restore(initial, model, reference, optimizer, generators)
            tick = time.monotonic()
            baseline, baseline_rows = detailed_observation(model, data, config, protocol, composition, generators)
            reference_observation = continuation.observe_early_health(reference, data, config, protocol, composition, generators)
            baseline256, baseline256_rows = benchmark_observation(model, protocol, composition, generators)
            baseline_law = panel_probabilities(model, policy["panel_count"], protocol["evaluation_seed"])
            report["timings"]["baseline_observation_seconds"] = time.monotonic() - tick
            report["baseline"] = baseline
            report["baseline_benchmark256"] = baseline256
            report["baseline_benchmark256_row_classifications"] = save_row_classifications(
                run.path / "baseline-benchmark256-classifications.json", baseline256_rows)
            report["baseline_row_classifications"] = save_row_classifications(run.path / "baseline-classifications.json", baseline_rows)
            report["reference"] = reference_observation
            write_json(output, report)

            def restore_start():
                teacher.drain()
                teacher.clear_cache()
                _restore(initial, model, reference, optimizer, generators)
                for group in optimizer.param_groups:
                    group["lr"] = report["production_learning_rate"]

            restore_start()
            hard = next(iter(_loader(data, cohorts, config, generators, "production", plan["seed"])))
            report["production_first_batch_sha256"] = state_digest(dict(atoms=hard.atoms, charges=hard.charges, bonds=hard.bonds))
            tick = time.monotonic()
            with BranchGradientCapture(model, 256, 512, protocol["microbatch_size"]) as capture:
                metrics = continuation.chemistry_update(model, reference, optimizer, hard, config, protocol, composition,
                    teacher, generators, updates, epoch + 1, plan["parent_epoch"], do_step=False,
                    deadline=deadline, overlap=overlap)
            gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                         for name, parameter in model.named_parameters()}
            report["branch_decomposition"] = capture.verify(gradients)
            clipping_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad, error_if_nonfinite=True))
            optimizer.step()
            _sync()
            proposal_state = deepcopy(model.state_dict())
            displacement = {name: parameter.detach().double() - initial["model"][name].double()
                            for name, parameter in model.named_parameters()}
            proposal_sha = state_digest(model.state_dict())
            report["proposal"] = dict(metrics=_compact_metrics(metrics), model_sha256=proposal_sha,
                pre_clip_gradient_norm=clipping_norm, clip_threshold=config.clip_grad,
                full=gradient_displacement(gradients, displacement),
                full_groups=grouped_gradient_displacement(gradients, displacement),
                branches={key: gradient_displacement(values, displacement) for key, values in capture.values.items()},
                branch_groups={key: grouped_gradient_displacement(values, displacement)
                               for key, values in capture.values.items()},
                component_scope="Weighted real/prior logical-branch totals, not isolated chemistry-term gradients",
                finite_step_objective="Not reevaluated: these are first-order diagnostics, not a descent certificate")
            report["timings"]["instrumented_proposal_seconds"] = time.monotonic() - tick
            # Release large gradient copies before repeated classified panels.
            del gradients, capture
            for scale in SCALES:
                if time.monotonic() > deadline:
                    raise TimeoutError("Disposable optimizer mechanism budget exhausted")
                apply_displacement(model, initial["model"], displacement, scale, proposal_state=proposal_state)
                tick = time.monotonic()
                observation, observed_rows = detailed_observation(model, data, config, protocol, composition, generators)
                law = panel_probabilities(model, policy["panel_count"], protocol["evaluation_seed"])
                assessment = continuation.assess_early_health(observation, reference_observation["population"],
                    reference_observation["reconstruction"], baseline["population"], baseline["reconstruction"], protocol,
                    starting_classification=baseline["classification"])
                report["trials"].append(dict(scale=scale, observation=observation, assessment=assessment,
                    output_drift=output_drift(baseline_law, law), observation_seconds=time.monotonic() - tick,
                    chemical_transitions=chemical_transitions(baseline_rows, observed_rows),
                    row_classifications=save_row_classifications(run.path / f"scale-{scale}-classifications.json", observed_rows),
                    optimizer_steps=0, targets_resampled=False))
                write_json(output, report)
            del displacement, proposal_state
            for stream in policy["prefix_streams"]:
                restore_start()
                loader = iter(_loader(data, cohorts, config, generators, stream, plan["seed"]))
                rows = []
                for relative in range(1, 9):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Disposable prefix budget exhausted")
                    _sync()
                    tick = time.monotonic()
                    hard = next(loader)
                    metrics = continuation.chemistry_update(model, reference, optimizer, hard, config, protocol, composition,
                        teacher, generators, updates + relative - 1, epoch + 1, plan["parent_epoch"],
                        deadline=deadline, overlap=overlap)
                    _sync()
                    row = dict(relative_update=relative, update_seconds=time.monotonic() - tick,
                               metrics=_compact_metrics(metrics))
                    if relative == 1 and stream == "production":
                        row["proposal_model_exact"] = state_digest(model.state_dict()) == proposal_sha
                        row["proposal_receipt_exact"] = (metrics["ordered_teacher_receipt_chain_sha256"]
                            == report["proposal"]["metrics"]["ordered_teacher_receipt_chain_sha256"])
                        if not row["proposal_model_exact"] or not row["proposal_receipt_exact"]:
                            raise ValueError("Instrumented proposal changed the production first update")
                    rows.append(row)
                    run.log(dict(event="disposable_prefix_update", stream=stream, **row))
                tick = time.monotonic()
                observation, observed_rows = detailed_observation(model, data, config, protocol, composition, generators)
                endpoint256, endpoint256_rows = benchmark_observation(model, protocol, composition, generators)
                law = panel_probabilities(model, policy["panel_count"], protocol["evaluation_seed"])
                assessment = continuation.assess_early_health(observation, reference_observation["population"],
                    reference_observation["reconstruction"], baseline["population"], baseline["reconstruction"], protocol,
                    starting_classification=baseline["classification"])
                report["prefixes"][stream] = dict(updates=rows, observation=observation, assessment=assessment,
                    benchmark256=endpoint256, chemical_transitions=chemical_transitions(baseline_rows, observed_rows),
                    benchmark256_chemical_transitions=chemical_transitions(baseline256_rows, endpoint256_rows),
                    benchmark256_row_classifications=save_row_classifications(
                        run.path / f"{stream}-benchmark256-classifications.json", endpoint256_rows),
                    row_classifications=save_row_classifications(run.path / f"{stream}-classifications.json", observed_rows),
                    output_drift=output_drift(baseline_law, law), observation_seconds=time.monotonic() - tick,
                    final_model_sha256=state_digest(model.state_dict()), disposable_updates=8,
                    phase_timing_scope="Native prepass/search and synchronized whole update; no classification-counter estimate")
                write_json(output, report)
            report.update(status="completed_disposable_not_training", diagnostic_completed=True,
                scientific_readiness="Not established; no readiness, stability or training promotion is written")
        except BaseException as error:
            report.update(status="failed", diagnostic_completed=False, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            cleanup_errors = []
            if teacher is not None:
                for action in (teacher.drain, teacher.clear_cache):
                    try:
                        action()
                    except Exception as error:
                        cleanup_errors.append(f"{type(error).__name__}: {error}")
            if initial is not None:
                try:
                    _restore(initial, model, reference, optimizer, generators)
                    restored = _snapshot(model, reference, optimizer, generators)
                    checks = {key: continuation.nested_equal(value, restored[key]) for key, value in initial.items()}
                    report["restoration"] = dict(passed=all(checks.values()), checks=checks)
                    if not all(checks.values()):
                        cleanup_errors.append("Complete model/optimizer/gradient/mode/RNG restoration failed")
                except Exception as error:
                    cleanup_errors.append(f"{type(error).__name__}: {error}")
            report["elapsed_seconds"] = time.monotonic() - started
            if cleanup_errors:
                report.update(status="failed", diagnostic_completed=False, cleanup_errors=cleanup_errors)
            write_json(run.path / "diagnostic.json", report)
            write_json(output, report)
            run.annotate(training_updates_promoted=0, promotable=False, epoch_complete=False,
                         diagnostic_status=report["status"])
            if cleanup_errors:
                raise RuntimeError("Disposable mechanism cleanup failed: " + "; ".join(cleanup_errors))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = continuation.read_plan(args.plan, args.plan_sha256)
    plan["_plan_sha256"] = args.plan_sha256
    run_diagnostic(plan)


if __name__ == "__main__":
    main()
