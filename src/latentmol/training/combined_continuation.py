"""Benchmark or continue epoch 45 on the canonical combined corpus."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

import torch

from latentmol.chemistry_continuation import chemistry_update
from latentmol.chemistry_classification import VERSION as CLASSIFICATION_VERSION
from latentmol.chemistry_valid_reference_anchor import FrozenValidReferenceAnchor
from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.combined import CombinedCorpus
from latentmol.datasets.data import collate_graphs
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.failure_guidance_training import build_repair_teacher
from latentmol.guidance.repair_proposals import ProposalRepairTeacher, RepairProposalConfig
from latentmol.models import GraphAutoencoder
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.train import TrainConfig, capture_rng_state, learning_rate_at_epoch, restore_rng_state, state_digest
from latentmol.training.invariant_pilot import initialize_invariant
from latentmol.training.losses import LossConfig
from latentmol.training.prior_repair import GUIDANCE_VERSION, prior_repair_update


VERSION = "combined-vae-continuation-v1"


def _authenticated(reference):
    path = Path(reference["path"])
    if file_sha256(path) != reference["sha256"]:
        raise ValueError(f"Input digest differs for {path}")
    return path


def _read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != VERSION:
        raise ValueError("Unexpected combined continuation version")
    if any(type(config[key]) is not int or config[key] < 1 for key in
           ("epochs", "benchmark_updates", "benchmark_warmup", "checkpoint_every_updates")):
        raise ValueError("Require positive epoch, benchmark and checkpoint budgets")
    sizes = config["batch_sizes"]
    if not sizes or any(type(size) is not int or size < 1 for size in sizes) or len(sizes) != len(set(sizes)):
        raise ValueError("Require distinct positive benchmark batch sizes")
    sizes = _microbatch_sizes(config)
    if (not sizes or any(type(size) is not int or size < 1 for size in sizes)
            or sizes != sorted(set(sizes), reverse=True) or sizes[0] > min(config["batch_sizes"])):
        raise ValueError("Require descending, distinct microbatches no larger than the smallest logical batch")
    if "prior_only" in config:
        ratio = config["prior_only"]["real_per_prior"]
        weight = config["prior_only"]["weight"]
        parent_epoch = config.get("parent_epoch", 45)
        if (type(parent_epoch) is not int or parent_epoch < 1 or "baseline_continuation" in config["protocol"]):
            raise ValueError("Prior-only training needs a valid parent epoch and its own repair protocol")
        if (type(ratio) is not int or ratio < 1 or any(size % ratio for size in config["batch_sizes"])
                or not math.isfinite(weight) or weight <= 0):
            raise ValueError("Prior-only training requires an exact batch ratio and a positive weight")
        if "guidance" in config["prior_only"]:
            _validate_prior_guidance(config["prior_only"]["guidance"])
    return config


def _validate_prior_guidance(guidance):
    if (not isinstance(guidance, dict)
            or set(guidance) != {"version", "reference", "occupancy_weight", "conditional_weight"}
            or guidance["version"] != GUIDANCE_VERSION):
        raise ValueError("Unknown frozen-parent prior guidance configuration")
    for key in ("occupancy_weight", "conditional_weight"):
        value = guidance[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Prior guidance {key} must be finite and positive")
    reference = guidance["reference"]
    if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
            or not reference["path"] or not isinstance(reference.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
            or ("epoch" in reference and (type(reference["epoch"]) is not int or reference["epoch"] < 1))):
        raise ValueError("Prior guidance requires a pinned reference checkpoint")


def _microbatch_sizes(config):
    return config.get("microbatch_sizes", [config["protocol"]["microbatch_size"]])


def _benchmark_execution(config, *, benchmark, workers=None, cpus=None):
    """Change benchmark CPU resources without changing the training recipe.

    The base config digest still identifies the common loss/data recipe;
    benchmark reports separately record the effective execution settings.
    """
    if workers is None and cpus is None:
        return config
    if not benchmark:
        raise ValueError("Benchmark CPU overrides require benchmark mode")
    execution = dict(config["execution"])
    if workers is not None:
        execution["workers"] = workers
    if cpus is not None:
        execution["gpu_cpus"] = cpus
    if (type(execution["workers"]) is not int or execution["workers"] < 0
            or type(execution["gpu_cpus"]) is not int
            or execution["gpu_cpus"] < execution["workers"] + execution["torch_threads"]):
        raise ValueError("Benchmark allocation must cover repair workers and PyTorch threads")
    return dict(config, execution=execution)


def _check_cuda_requirement(config, *, benchmark=False):
    requirement = config.get("benchmark_cuda_requirement", config.get("cuda_requirement")) if benchmark else config.get("cuda_requirement")
    if requirement is None:
        return
    slots = os.environ.get("LSB_DJOB_NUMPROC", "")
    device = torch.cuda.get_device_properties(0)
    memory_gib = device.total_memory / 1024**3
    if (torch.cuda.device_count() != 1 or requirement["name_contains"].lower() not in device.name.lower()
            or not requirement["minimum_gib"] <= memory_gib <= requirement["maximum_gib"]
            or not slots.isdecimal() or int(slots) != config["execution"]["gpu_cpus"]):
        raise RuntimeError(f"Unexpected CUDA allocation: {device.name}, {memory_gib:.2f} GiB, CPU slots {slots!r}")


def _order(count, seed, epoch):
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch))


def _load(config, corpus, device):
    parent, payload = load_model(_authenticated(config["checkpoint"]), "cpu", corpus.spec)
    parent_epoch = config.get("parent_epoch", 45)
    if (payload.get("epoch") != parent_epoch or payload.get("train_config", {}).get("mode") != "vae"
            or not payload.get("continuation", {}).get("epoch_complete") or payload.get("diagnostic_only")):
        raise ValueError(f"Require the completed epoch-{parent_epoch} parent VAE")
    if "prior_only" in config:
        if parent_epoch > 45 and (payload["continuation"].get("epoch") != parent_epoch
                                  or payload["continuation"].get("row") != 0):
            raise ValueError("The prior-only parent must have completed its saved dataset pass")
        reference = None
        if guidance := config["prior_only"].get("guidance"):
            source = guidance["reference"]
            if all(source[k] == config["checkpoint"][k] for k in ("path", "sha256")):
                # This exact checkpoint was authenticated above. Keep the trainable
                # parent independent so deepcopy/resume cannot freeze its parameters.
                reference, reference_payload = deepcopy(parent), payload
            else:
                reference, reference_payload = load_model(_authenticated(source), "cpu", corpus.spec)
            if (reference_payload.get("train_config", {}).get("mode") != "vae"
                    or not reference_payload.get("continuation", {}).get("epoch_complete")
                    or reference_payload.get("diagnostic_only")
                    or ("epoch" in source and source["epoch"] != reference_payload["epoch"])
                    or reference.config.latent != parent.config.latent):
                raise ValueError("Prior guidance requires a complete reference VAE with the shared latent width")
            reference.eval().requires_grad_(False).to(device)
        return parent, payload, reference, None, None
    if payload["dataset_sha256"] != corpus.manifest["sources"]["moses"]["dataset_sha256"]:
        raise ValueError("Combined corpus is not based on the epoch-45 MOSES dataset")
    occupancy, _ = load_model(_authenticated(config["occupancy_reference"]), device, corpus.spec)
    occupancy.eval().requires_grad_(False)
    reference, _ = load_model(_authenticated(config["valid_reference"]), device, corpus.spec)
    anchor = FrozenValidReferenceAnchor(reference, reference_epoch=36,
        checkpoint_sha256=config["valid_reference"]["sha256"])
    composition = json.loads(_authenticated(config["composition"]).read_text(encoding="utf-8"))
    return parent, payload, occupancy, anchor, composition


def _fresh(parent, payload, anchor, batch_size, device, resume, *, atom_refinement_dropout=None):
    # Epoch 45 needs the lossless canonical-input wrapper; epoch 46 and later
    # already have it. Continuing those checkpoints must preserve their exact
    # architecture and weights instead of repeating an ordered-only conversion.
    if parent.config.encoder_pooling == "canonical":
        model = deepcopy(parent)
    else:
        model, _ = initialize_invariant(parent, pooling="canonical",
            aggregation=parent.config.encoder_aggregation, degree_features=parent.config.encoder_degree_features,
            sum_scale=parent.config.encoder_sum_scale)
    if atom_refinement_dropout is not None and model.config.atom_refinement_dropout != atom_refinement_dropout:
        updated = GraphAutoencoder(model.spec, replace(model.config, atom_refinement_dropout=atom_refinement_dropout))
        # AdamW restores moments by parameter position. Dropout must add no parameters
        # and strict weight loading must preserve the frozen parent and student outputs in eval.
        if list(dict(updated.named_parameters())) != list(dict(model.named_parameters())):
            raise ValueError("Atom-refinement dropout changed optimizer parameter order")
        updated.load_state_dict(model.state_dict(), strict=True)
        model = updated
    model = model.to(device).train()
    object.__setattr__(model, "_valid_reference_anchor", anchor)
    config = TrainConfig(**dict(payload["train_config"], batch_size=batch_size, device=device,
                               model=model.config, loss=LossConfig(**payload["loss_config"])))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    state = payload if resume is None else resume
    if resume is not None:
        model.load_state_dict(resume["model_state"], strict=True)
    # AdamW can reuse CPU step counters even when its moment tensors move to CUDA.
    # Each benchmark arm must begin with independent parent moments and counters.
    optimizer.load_state_dict(deepcopy(state["optimizer_state"]))
    saved_rng = state["continuation"]["rng"]
    generators = {name: torch.Generator() for name in saved_rng["loaders"]}
    restore_rng_state(saved_rng if device == "cuda" else dict(saved_rng, cuda=[]), generators)
    if "fresh_prior" not in generators:
        raise ValueError("Parent checkpoint is missing its fresh-prior RNG")
    return model, optimizer, config, generators


def _teacher(config, spec, anchor):
    if "prior_only" in config:
        # Classification is a diagnostic of this teacher, independent of
        # the legacy marker for the full combined chemistry objective.
        return ProposalRepairTeacher(spec, ProbabilityRepairConfig(**config["protocol"]["search"]),
            proposals=RepairProposalConfig(**config["proposal_config"]),
            backend=config["execution"]["backend"], workers=config["execution"]["workers"],
            classification_version=CLASSIFICATION_VERSION,
            classification_execution=config["classification_execution"])
    teacher = build_repair_teacher(spec, config["protocol"], config["execution"],
        classification_execution=config["classification_execution"], proposal_config=config["proposal_config"])
    teacher.valid_reference_anchor = anchor
    return teacher


def _update(config, model, occupancy, optimizer, hard, training, composition, teacher, generators,
            update, epoch, microbatch, timings=None):
    if "prior_only" in config:
        policy = config["prior_only"]
        return prior_repair_update(model, optimizer, hard, training, teacher, generators["fresh_prior"],
            real_per_prior=policy["real_per_prior"], prior_weight=policy["weight"],
            update=update, epoch=epoch, microbatch=microbatch, step_control=config["step_control"],
            timings=timings, reference=occupancy, guidance=policy.get("guidance"))
    protocol = dict(config["protocol"], prior_ratio=1, composition_prior_weight=3.,
                    real_batch_size=training.batch_size)
    result = chemistry_update(model, occupancy, optimizer, hard, training, protocol, composition, teacher,
        generators, update, epoch, 35, microbatch_size=microbatch, overlap=config["execution"]["overlap"],
        step_control=config["step_control"], from_canonical_cache=True, allow_equal_prior_ratio=True,
        wait_observer=timings.measure if timings is not None else None)
    if result["real_points"] != len(hard.atoms) or result["fresh_prior_points"] != len(hard.atoms):
        raise RuntimeError("The requested 1:1 real-to-fresh-prior ratio changed")
    if any(p.grad is None for p in (model.encoder.mu.weight, model.decoder.atom_head.weight)):
        raise RuntimeError("Required encoder or decoder path is detached")
    return result


def _batch(data, order, start, size):
    return collate_graphs([data[i] for i in order[start:start + size].tolist()])


def _check_frozen(occupancy, occupancy_digest, anchor):
    if occupancy is None:
        return
    if anchor is not None:
        anchor.assert_unchanged()
    if (any(module.training for module in occupancy.modules())
            or any(p.requires_grad or p.grad is not None for p in occupancy.parameters())
            or state_digest(occupancy.state_dict()) != occupancy_digest):
        raise RuntimeError("Frozen occupancy reference changed")


def _benchmark_arm(config, data, assets, output, device, order, size, microbatch, occupancy_digest):
    parent, payload, occupancy, anchor, composition = assets
    epoch = config.get("parent_epoch", 45) + 1
    initial_update = payload["continuation"]["updates"]
    timed = config["benchmark_updates"]
    warmup = config["benchmark_warmup"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times, measured_rows, accepted, attempts = [], 0, 0, 0
    error = None
    try:
        model, optimizer, training, generators = _fresh(parent, payload, anchor, size, device, None,
            atom_refinement_dropout=config.get("atom_refinement_dropout"))
        if "prior_only" in config:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_epoch(training, epoch)
        with _teacher(config, model.spec, anchor) as teacher:
            for step in range(warmup + timed):
                offset = (step * size) % len(data)
                start = time.monotonic()
                metrics = _update(config, model, occupancy, optimizer, _batch(data, order, offset, size),
                                  training, composition, teacher, generators, initial_update + step, epoch, microbatch)
                if device == "cuda":
                    torch.cuda.synchronize()
                if step >= warmup:
                    times.append(time.monotonic() - start)
                    measured_rows += metrics["real_points"]
                    accepted += metrics["repair_targets"]["prior"]
                    attempts += metrics["fresh_prior_points"]
                if step == 0 or (step + 1) % 8 == 0:
                    write_json(output / "progress.json", dict(stage="benchmark", batch_size=size,
                        microbatch_size=microbatch, update=step + 1, job_id=os.environ.get("LSB_JOBID")))
    except torch.OutOfMemoryError as exc:
        error = str(exc)
    _check_frozen(occupancy, occupancy_digest, anchor)
    reserved = torch.cuda.max_memory_reserved() if device == "cuda" else 0
    allocated = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    fraction = reserved / torch.cuda.get_device_properties(0).total_memory if device == "cuda" else 0.
    passed = (error is None and len(times) == timed and fraction <= config["protocol"]["vram_fraction_limit"]
              and ("prior_only" not in config or accepted > 0))
    arm = dict(batch_size=size, microbatch_size=microbatch, passed=passed, error=error, warmup_updates=warmup,
               timed_updates=len(times), real_points=measured_rows, fresh_prior_points=attempts,
               prior_repairs_accepted=accepted, exercised_prior_repair=accepted > 0,
               peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
               reserved_fraction=fraction, elapsed_seconds=sum(times))
    if measured_rows:
        arm.update(seconds_per_update=statistics.mean(times), real_points_per_second=measured_rows / sum(times),
                   projected_epoch_seconds=len(data) * sum(times) / measured_rows)
    return arm


def _benchmark(config, identity, data, assets, output, device):
    order = _order(len(data), config["shuffle_seed"], config.get("parent_epoch", 45) + 1)
    occupancy_digest = None if assets[2] is None else state_digest(assets[2].state_dict())
    arms = []
    for size in config["batch_sizes"]:
        for microbatch in _microbatch_sizes(config):
            arm = _benchmark_arm(config, data, assets, output, device, order, size, microbatch, occupancy_digest)
            arms.append(arm)
            if device == "cuda":
                torch.cuda.empty_cache()
            if arm["passed"]:
                break
    winner = max((arm for arm in arms if arm["passed"]), key=lambda arm: arm["real_points_per_second"],
                 default=None)
    report = dict(status="passed" if winner else "no_passing_arm", job_id=os.environ.get("LSB_JOBID"),
                  execution=config.get("execution"),
                  config_sha256=identity, dataset_sha256=data.manifest["dataset_sha256"],
                  device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                  device_memory_bytes=torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None,
                  training_molecules=len(data), arms=arms,
                  selected_batch_size=winner["batch_size"] if winner else None,
                  selected_microbatch_size=winner["microbatch_size"] if winner else None)
    write_json(output / "benchmark.json", report)
    if winner is None:
        raise RuntimeError("No benchmark batch size met the memory and finite-update checks")
    return report


def _checkpoint(output, model, optimizer, training, generators, identity, dataset, cursor, previous, microbatch,
                *, experiment_state=None):
    updates = cursor["updates"]
    path = output / f"checkpoint-{updates:09d}.pt"
    if path.exists():
        raise FileExistsError(path)
    complete_epoch = cursor["epoch"]
    payload = dict(format_version=1, graph_spec=model.spec.to_dict(), model_config=asdict(model.config),
        loss_config=asdict(training.loss), train_config=asdict(training), model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(), dataset_sha256=dataset, epoch=complete_epoch, run_id=output.name,
        combined_version=VERSION, config_sha256=identity, batch_size=training.batch_size, microbatch_size=microbatch,
        continuation=dict(cursor, rng=capture_rng_state(generators, training.device == "cuda"),
                          epoch_complete=cursor["row"] == 0))
    if experiment_state is not None:
        payload["experiment_state"] = experiment_state
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
    temporary.replace(path)
    receipt = dict(checkpoint=str(path), sha256=file_sha256(path), **cursor,
                   completed_epoch=complete_epoch)
    write_json(output / "latest.json", receipt)
    if previous is not None:
        previous.unlink()
    return receipt, path


def _train(config, identity, data, assets, output, device, size, resume, max_updates, deadline, microbatch=None,
           *, validation=None, student_factory=_fresh, teacher_factory=_teacher, update_fn=_update,
           experiment_state=None):
    parent, payload, occupancy, anchor, composition = assets
    if microbatch is None:
        microbatch = _microbatch_sizes(config)[0]
    model, optimizer, training, generators = student_factory(parent, payload, anchor, size, device, resume,
        atom_refinement_dropout=config.get("atom_refinement_dropout"))
    dataset = data.manifest["dataset_sha256"]
    parent_epoch = config.get("parent_epoch", 45)
    target_epoch = parent_epoch + config["epochs"]
    cursor = (dict(epoch=parent_epoch, row=0, updates=payload["continuation"]["updates"], real_points=0)
              if resume is None else {k: resume["continuation"][k] for k in
                                      ("epoch", "row", "updates", "real_points")})
    if (not parent_epoch <= cursor["epoch"] <= target_epoch or not 0 <= cursor["row"] < len(data)
            or cursor["row"] % size
            or cursor["real_points"] != (cursor["epoch"] - parent_epoch) * len(data) + cursor["row"]
            or cursor["updates"] != (payload["continuation"]["updates"]
                + (cursor["epoch"] - parent_epoch) * math.ceil(len(data) / size) + cursor["row"] // size)):
        raise ValueError("Combined checkpoint cursor is incompatible with training membership")
    occupancy_digest = None if occupancy is None else state_digest(occupancy.state_dict())
    initial_update = cursor["updates"]
    monitor = None
    if "quality_panel" in config:
        from latentmol.training.occupancy_monitor import OccupancyMonitor
        monitor = OccupancyMonitor(config["quality_panel"], validation, model, output, initial_update)
        monitor.record(model, cursor, force=True)
    started = time.monotonic()
    previous = None
    receipt = None
    last_saved = initial_update
    order = None
    epoch_order = None
    stop_reason = "epoch_target_reached"
    with (teacher_factory(config, model.spec, anchor) as teacher,
          (output / "training.jsonl").open("x", encoding="utf-8") as log):
        while cursor["epoch"] < target_epoch:
            if max_updates is not None and cursor["updates"] - initial_update >= max_updates:
                stop_reason = "update_budget_reached"
                break
            if deadline is not None and time.monotonic() >= deadline - 120:
                stop_reason = "allocation_budget_reached"
                break
            epoch = cursor["epoch"] + 1
            if epoch_order != epoch:
                order = _order(len(data), config["shuffle_seed"], epoch)
                epoch_order = epoch
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate_at_epoch(training, epoch)
            hard = _batch(data, order, cursor["row"], size)
            start = time.monotonic()
            metrics = update_fn(config, model, occupancy, optimizer, hard, training,
                              composition, teacher, generators, cursor["updates"], epoch, microbatch)
            cursor["updates"] += 1
            cursor["row"] += len(hard.atoms)
            cursor["real_points"] += len(hard.atoms)
            epoch_end = cursor["row"] == len(data)
            if epoch_end:
                cursor.update(epoch=epoch, row=0)
            event = dict(metrics, **cursor, seconds=time.monotonic() - start)
            if "atom_refinement_dropout" in config:
                event["atom_refinement_dropout"] = model.config.atom_refinement_dropout
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()
            if epoch_end or cursor["updates"] - last_saved >= config["checkpoint_every_updates"]:
                _check_frozen(occupancy, occupancy_digest, anchor)
                receipt, previous = _checkpoint(output, model, optimizer, training, generators,
                                                identity, dataset, cursor, previous, microbatch,
                                                experiment_state=experiment_state)
                last_saved = cursor["updates"]
                if epoch_end:
                    previous = None
            if cursor["updates"] == initial_update + 1 or cursor["updates"] % 10 == 0:
                write_json(output / "progress.json", dict(stage="training", **cursor,
                    elapsed_seconds=time.monotonic() - started, job_id=os.environ.get("LSB_JOBID"),
                    latest_metrics=metrics))
            if monitor is not None:
                monitor.record(model, cursor, force=epoch_end)
    if cursor["updates"] > last_saved:
        _check_frozen(occupancy, occupancy_digest, anchor)
        receipt, _ = _checkpoint(output, model, optimizer, training, generators, identity,
                                 dataset, cursor, previous, microbatch, experiment_state=experiment_state)
    if monitor is not None:
        monitor.record(model, cursor, force=True)
    report = dict(status="completed" if stop_reason == "epoch_target_reached" else "paused",
                  stop_reason=stop_reason, dataset_sha256=dataset, training_molecules=len(data),
                  elapsed_seconds=time.monotonic() - started, batch_size=size, latest=receipt,
                  microbatch_size=microbatch, segment_updates=cursor["updates"] - initial_update, **cursor)
    write_json(output / "summary.json", report)
    return report


def _benchmark_shape(config, report, identity, dataset, size, microbatch):
    chosen_size = report.get("selected_batch_size")
    default_microbatch = config["protocol"]["microbatch_size"]
    chosen_microbatch = report.get("selected_microbatch_size", default_microbatch)
    if (report.get("status") != "passed" or report.get("config_sha256") != identity
            or report.get("dataset_sha256") != dataset or size not in (None, chosen_size)
            or microbatch not in (None, chosen_microbatch)
            or not any(arm.get("passed") is True and arm.get("batch_size") == chosen_size
                       and arm.get("microbatch_size", default_microbatch) == chosen_microbatch
                       for arm in report.get("arms", []))):
        raise ValueError("Benchmark and training identities or selected batch shape differ")
    requirement = config.get("benchmark_cuda_requirement", config.get("cuda_requirement"))
    if requirement is not None and (
            requirement["name_contains"].lower() not in report.get("device", "").lower()
            or not requirement["minimum_gib"] <= (report.get("device_memory_bytes") or 0) / 1024**3
            <= requirement["maximum_gib"]):
        raise ValueError("Benchmark did not use the required CUDA device capacity")
    return chosen_size, chosen_microbatch


def _approved_dropout_recipe(config_path, config):
    """Check that the training-mode dropout is the only change to a pinned prior-guidance recipe."""
    rate = config.get("atom_refinement_dropout")
    source = config.get("dropout_handoff")
    if (type(rate) not in (float, int) or not 0 < rate < 1 or not isinstance(source, dict)
            or set(source) != {"source_config_path", "source_config_sha256"}):
        raise ValueError("Dropout handoff requires a positive probability and pinned source recipe")
    approved = json.loads(_authenticated(dict(path=config_path.parent / source["source_config_path"],
                                               sha256=source["source_config_sha256"])).read_bytes())
    if (not approved.get("recipe_handoff") or not approved.get("prior_only", {}).get("guidance")
            or {k: v for k, v in config.items() if k not in ("atom_refinement_dropout", "dropout_handoff")} != approved):
        raise ValueError("Dropout handoff changed settings outside atom-refinement dropout")
    return approved


def _load_training_resume(config_path, config, identity, resume_path, resume_sha256,
                          dataset_sha256, size, microbatch):
    """Allow a loss change only at the exact checkpoint authorized by this recipe."""
    dropout = "dropout_handoff" in config or "atom_refinement_dropout" in config
    approved = _approved_dropout_recipe(config_path, config) if dropout else config
    handoff = config.get("recipe_handoff")
    if resume_path is None:
        if handoff is not None:
            raise ValueError("A recipe handoff requires its saved training checkpoint")
        return None, None
    resume = torch.load(_authenticated(dict(path=resume_path, sha256=resume_sha256)),
                        map_location="cpu", weights_only=True)
    if (resume.get("combined_version") != VERSION or resume.get("dataset_sha256") != dataset_sha256
            or resume.get("batch_size") != size
            or resume.get("microbatch_size", config["protocol"]["microbatch_size"]) != microbatch):
        raise ValueError("Resume checkpoint has an incompatible dataset, batch shape or trainer")
    if resume.get("config_sha256") == identity:
        if dropout and resume.get("model_config", {}).get("atom_refinement_dropout") != config["atom_refinement_dropout"]:
            raise ValueError("Saved student dropout does not match this continuation")
        return resume, None
    if (handoff is None or resume_sha256 != handoff["checkpoint_sha256"]
            or resume.get("config_sha256") != handoff["source_config_sha256"]):
        raise ValueError("Resume checkpoint belongs to a different combined continuation")

    cursor = {key: resume["continuation"][key] for key in ("epoch", "row", "updates", "real_points")}
    if (any(cursor[key] != handoff[key] for key in cursor)
            or resume.get("epoch") != cursor["epoch"]
            or resume["continuation"].get("epoch_complete") != (cursor["row"] == 0)):
        raise ValueError("Recipe handoff cursor differs from its designated training boundary")
    source_config_path = _authenticated(dict(path=config_path.parent / handoff["source_config_path"],
                                             sha256=handoff["source_config_sha256"]))
    expected = json.loads(source_config_path.read_bytes())
    # Reuse the parent's original permutation, optimizer, data and batch schedule.
    # If the source has finished, the new segment can cover the following epoch.
    expected["epochs"] = cursor["epoch"] + 1 - expected["parent_epoch"]
    expected["recipe_handoff"] = handoff
    expected["prior_only"]["guidance"] = config["prior_only"]["guidance"]
    if "quality_panel" in config:
        expected["quality_panel"] = config["quality_panel"]
    if expected != approved:
        raise ValueError("Recipe handoff changed settings outside prior guidance, monitoring or endpoint")
    receipt = dict(**cursor, source_checkpoint=str(resume_path), checkpoint_sha256=resume_sha256,
                   source_config_sha256=handoff["source_config_sha256"],
                   new_config_sha256=identity, target_epoch=cursor["epoch"] + 1)
    if dropout:
        if resume["model_config"].get("atom_refinement_dropout", 0.) != 0.:
            raise ValueError("Expected a dropout-free student at the designated adoption point")
        receipt.update(approved_guidance_config_sha256=config["dropout_handoff"]["source_config_sha256"],
                       atom_refinement_dropout=config["atom_refinement_dropout"])
    return resume, receipt


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-workers", type=int, help="Override repair workers for benchmark mode only")
    parser.add_argument("--benchmark-cpus", type=int, help="Override allocated CPU slots for benchmark mode only")
    parser.add_argument("--timing-study", type=Path, help="Run a disposable, matched benchmark")
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    if args.benchmark and (args.resume or args.batch_size or args.microbatch_size or args.max_updates or args.benchmark_report):
        parser.error("Benchmark mode has its own batch and update budget")
    if args.timing_study and not args.benchmark:
        parser.error("A timing study requires benchmark mode")
    if bool(args.resume) != bool(args.resume_sha256):
        parser.error("Resume requires both path and SHA-256")
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("Update budget must be positive")
    if args.wall_seconds is not None and (not math.isfinite(args.wall_seconds) or args.wall_seconds <= 120):
        parser.error("Allocation budget must leave more than 120 seconds")
    if args.device == "cuda" and (not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available()):
        raise RuntimeError("CUDA training requires an allocated DTU job")
    config = _read_config(args.config)
    study = None
    if args.timing_study:
        studies = json.loads(args.timing_study.read_bytes())
        study = studies["prior_only" if "prior_only" in config else "combined"]
        config["execution"]["gpu_cpus"] = study["gpu_cpus"]
    config = _benchmark_execution(config, benchmark=args.benchmark,
                                  workers=args.benchmark_workers, cpus=args.benchmark_cpus)
    if study is not None and args.benchmark_cpus is not None:
        study = dict(study, gpu_cpus=config["execution"]["gpu_cpus"])
    if args.device == "cuda":
        _check_cuda_requirement(config, benchmark=args.benchmark)
    elif config.get("cuda_requirement") is not None:
        raise RuntimeError("The CUDA-specific combined configuration requires an allocated GPU")
    identity = file_sha256(args.config)
    torch.set_num_threads(config["execution"]["torch_threads"])
    if "prior_only" in config:
        corpus = BalancedSparseCorpus(config["data"], expected_sha256=config["data_sha256"])
    else:
        corpus = CombinedCorpus(config["data"])
    data = corpus.split("train")
    if "prior_only" in config and len(data) % config["prior_only"]["real_per_prior"]:
        raise ValueError("The last batch cannot preserve the declared ratio")
    subsets = (data,) if "prior_only" in config else (
        corpus.split("train", source=source) for source in ("moses", "chembl")
    )
    for subset in subsets:
        verify_canonical_training_cache(subset, range(min(16, len(subset))))
    assets = _load(config, corpus, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    if marker := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(marker), dict(status="models_and_inputs_ready", nonce=os.environ["DBMOL_LAUNCH_NONCE"],
                                     job_id=os.environ["LSB_JOBID"]))
    if args.benchmark:
        if study is None:
            _benchmark(config, identity, data, assets, args.output, args.device)
        else:
            from latentmol.training.timing_study import run_timing_study
            run_timing_study(config, identity, study, data, assets, args.output, args.device)
        return
    size, microbatch = args.batch_size, args.microbatch_size
    if args.benchmark_report:
        report = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
        size, microbatch = _benchmark_shape(config, report, identity, data.manifest["dataset_sha256"], size, microbatch)
    if microbatch is None:
        microbatch = _microbatch_sizes(config)[0]
    if size not in config["batch_sizes"] or microbatch not in _microbatch_sizes(config):
        parser.error("Choose a configured batch shape or provide its benchmark report")
    resume, handoff = _load_training_resume(args.config, config, identity, args.resume, args.resume_sha256,
                                            data.manifest["dataset_sha256"], size, microbatch)
    if handoff is not None:
        write_json(args.output / "recipe_handoff.json", handoff)
    _train(config, identity, data, assets, args.output, args.device, size, resume,
           args.max_updates, started + args.wall_seconds if args.wall_seconds is not None else None, microbatch,
           validation=corpus.split("val") if "quality_panel" in config else None)


if __name__ == "__main__":
    main()
