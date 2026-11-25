"""New objective-change handoff at a completed VAE epoch, never an exact replay.

The historical campaign remains immutable. This worker restores its weights,
AdamW moments and RNG, then explicitly adopts half displacement and a local
latent sampling mixture. Every allocation saves one complete epoch; incomplete
or failed snapshots are diagnostic only. Numerical execution requires LSF CUDA.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from . import chemistry_continuation as c
from .chemistry_early_health import preserve_observation_state
from .chemistry_optimizer_step import HALF_CONTINUATION_VERSION, validate_optimizer_step_receipt
from .chemistry_valid_reference_anchor import FrozenValidReferenceAnchor
from .data import GraphDataset, collate_graphs, validate_large_cache
from .evaluate import load_model
from .experiments import file_sha256, write_json
from .failure_guidance_training import build_repair_teacher
from .latent_neighborhood import LocalLatentPrior, NeighborhoodConfig
from .models.autoencoder import comparable_train_config
from .train import (algorithm_identity, capture_rng_state, compact_cohort_identity,
                    learning_rate_at_epoch, restore_rng_state, seed_everything,
                    state_digest, training_cohorts)


VERSION = "vae-local-neighborhood-continuation-v1"
STEP = dict(version=HALF_CONTINUATION_VERSION, factor=.125)


def read_json(path):
    return json.loads(Path(path).read_text())


def authenticated(path, sha):
    if file_sha256(Path(path)) != sha:
        raise ValueError("Changed immutable input: " + str(path))


def load_study(path, sha):
    authenticated(path, sha)
    study = read_json(path)
    adoption = study.get("structural_adoption")
    if (study.get("version") != VERSION or study.get("start_epoch") != (44 if adoption else 43)
            or study.get("end_epoch") != 45 or study.get("optimizer_step_control") != STEP):
        raise ValueError("Unsupported bounded neighborhood continuation")
    NeighborhoodConfig(**study["neighborhood"])
    authenticated(study["base_plan"], study["base_plan_sha256"])
    base = read_json(study["base_plan"])
    authenticated(study["release"] + ".tar.gz", study["archive_sha256"])
    # Check the few changed executables without requalifying unrelated models.
    for relative, digest in study["changed_sources"].items():
        authenticated(Path(study["release"]) / relative, digest)
    for asset, digest in study["inherited_metadata"].items():
        authenticated(asset, digest)
    if adoption:
        from .guidance.repair_proposals import RepairProposalConfig
        if adoption.get("version") != "epoch45-structural-proposals-v1":
            raise ValueError("Unknown structural adoption")
        proposals = RepairProposalConfig(**adoption["proposal_config"])
        if proposals.sample_draws != 0 or not proposals.connect_components or not proposals.prune_dense:
            raise ValueError("Require the selected deterministic structural proposal policy")
        authenticated(adoption["source_study"], adoption["source_study_sha256"])
        authenticated(study["source_completion"], adoption["source_completion_sha256"])
        source = read_json(adoption["source_study"])
        if (source.get("version") != VERSION or source.get("base_plan_sha256") != study["base_plan_sha256"]
                or source.get("neighborhood") != study["neighborhood"]
                or source.get("optimizer_step_control") != STEP):
            raise ValueError("Structural adoption changed the inherited non-search policy")
    return study, base


def structural_sampler_start(study, base, payload, checkpoint_sha):
    """Bind a new target-discovery policy to authentic44 without resetting RNG/ramp."""
    adoption = study["structural_adoption"]
    source = read_json(adoption["source_study"])
    phase = payload["chemistry_phase"]
    amendment = phase.get("neighborhood_continuation", {})
    if (payload.get("epoch") != 44 or checkpoint_sha != study["probe_checkpoint"]["sha256"]
            or phase.get("plan_sha256") != adoption["source_study_sha256"]
            or amendment.get("study_sha256") != adoption["source_study_sha256"]
            or amendment.get("base_plan_sha256") != study["base_plan_sha256"]
            or amendment.get("policy") != study["neighborhood"]
            or phase.get("protocol") != base["protocol"]
            or phase.get("optimizer_step_control") != STEP
            or payload.get("source_archive_sha256") != source["archive_sha256"]
            or type(amendment.get("start_update")) is not int
            or not 0 <= amendment["start_update"] < payload["continuation"]["updates"]):
        raise ValueError("Structural discovery must resume the exact completed epoch44 policy/checkpoint")
    return amendment["start_update"]


def setup(study, base, path, sha, study_sha):
    """Explicitly changed objective; old exact-resume readers remain untouched."""
    if not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Neighborhood worker requires an allocated CUDA job")
    if int(os.environ.get("LSB_DJOB_NUMPROC", "0")) != 16:
        raise RuntimeError("Neighborhood chemistry worker requires its 16 CPU slots")
    torch.set_num_threads(4)
    seed_everything(base["seed"])
    authenticated(base["config"], base["config_sha256"])
    config = c.load_parent_config(Path(base["config"]))
    cache = validate_large_cache(Path(base["data"]))
    data = {name: GraphDataset(base["data"], name, validated_cache=cache) for name in ("train", "val")}
    if (data["train"].manifest["dataset_sha256"] != base["dataset_sha256"]
            or data["train"].manifest["split_sha256"] != base["split_sha256"]):
        raise ValueError("Dataset/split differs from the running VAE campaign")
    authenticated(path, sha)
    model, payload = load_model(Path(path), "cpu", data["train"].spec)
    cont, phase = payload["continuation"], payload["chemistry_phase"]
    if (not cont.get("epoch_complete") or payload.get("diagnostic_only")
            or cont.get("scheduler_state") is not None
            or json.dumps(comparable_train_config(payload["train_config"]), sort_keys=True) != json.dumps(asdict(config), sort_keys=True)):
        raise ValueError("Require a completed compatible VAE checkpoint with inherited optimizer/RNG")
    cohorts = training_cohorts(config, len(data["train"]), len(data["val"]))
    if compact_cohort_identity(cohorts) != cont["indices"] or cont["split_sha256"] != base["split_sha256"]:
        raise ValueError("Training membership or split differs")
    amendment = phase.get("neighborhood_continuation")
    if study.get("structural_adoption"):
        sampler_start = structural_sampler_start(study, base, payload, sha)
    elif amendment is None:
        if (payload["epoch"] not in (42, study["start_epoch"])
                or phase.get("plan_sha256") != study["base_plan_sha256"]
                or phase.get("protocol") != base["protocol"]
                or phase.get("optimizer_step_control") != base["optimizer_step_control"]
                or payload["source_archive_sha256"] != base["archive_sha256"]):
            raise ValueError("The objective-change input does not belong to the original campaign")
        sampler_start = cont["updates"]
    else:
        if (amendment.get("study_sha256") != study_sha
                or phase.get("optimizer_step_control") != STEP
                or amendment.get("policy") != study["neighborhood"]
                or not study["start_epoch"] < payload["epoch"] < study["end_epoch"]):
            raise ValueError("New continuation policy/epoch differs")
        sampler_start = amendment["start_update"]
    if cont["updates"] != payload["epoch"] * math.ceil(len(data["train"])/config.batch_size):
        raise ValueError("Completed epoch/update accounting differs")
    model = model.to(config.device).train()
    authenticated(base["parent_checkpoint"], base["parent_checkpoint_sha256"])
    reference, _ = load_model(Path(base["parent_checkpoint"]), config.device, model.spec)
    reference.eval().requires_grad_(False)
    original36 = base["starting_checkpoint"]
    authenticated(original36["path"], original36["sha256"])
    anchor_model, _ = load_model(Path(original36["path"]), config.device, model.spec)
    anchor = FrozenValidReferenceAnchor(anchor_model, reference_epoch=36, checkpoint_sha256=original36["sha256"])
    del anchor_model
    object.__setattr__(model, "_valid_reference_anchor", anchor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(payload["optimizer_state"])
    if not c.nested_equal(optimizer.state_dict(), payload["optimizer_state"]):
        raise ValueError("Restored native AdamW moments differ")
    generators = {name: torch.Generator() for name in cont["rng"]["loaders"]}
    restore_rng_state(cont["rng"], generators)
    if amendment is None:
        generators["latent_neighborhood"] = torch.Generator().manual_seed(study["neighborhood_seed"])
    elif "latent_neighborhood" not in generators:
        raise ValueError("Missing neighborhood RNG on resumed amendment")
    sampler = LocalLatentPrior(NeighborhoodConfig(**study["neighborhood"]),
        generators["latent_neighborhood"], start_update=sampler_start)
    teacher = build_repair_teacher(model.spec, base["protocol"], base["execution"],
                                  classification_execution=base.get("classification_execution"),
                                  proposal_config=study.get("structural_adoption", {}).get("proposal_config"))
    teacher.valid_reference_anchor = anchor
    composition = read_json(Path(base["campaign"]) / "training_composition.json")
    return model, reference, optimizer, config, data, cohorts, generators, sampler, teacher, composition, payload


def observe(model, reference, optimizer, config, data, composition, generators, base, folder, update):
    """Reuse the original fixed Gaussian panels and original health thresholds."""
    observation = c.observe_early_health(model, data, config, base["protocol"], composition, generators)
    baseline = read_json(Path(base["campaign"]) / "early-health-baseline.json")
    readiness = read_json(Path(base["campaign"]) / "failure_readiness.json")
    native = readiness["native_panel"]
    parent_path = folder / "reference-baseline.json"
    if not parent_path.exists():
        with preserve_observation_state(model, generators):
            parent = dict(population=c.population_panel(reference, 1024, base["protocol"]["evaluation_seed"], 32, composition),
                reconstruction=c.validation(reference, data["val"], config, 256, base["protocol"]["evaluation_seed"]))
        write_json(parent_path, parent)
    parent = read_json(parent_path)
    # The unchanged parent and original36 baselines are not ratcheted to43.
    checks = c.assess_early_health(observation, parent["population"],
        parent["reconstruction"], native["starting_retention_population"],
        native["starting_reconstruction"], base["protocol"],
        starting_classification=baseline["observation"]["classification"])
    (folder / "early_health").mkdir(exist_ok=True)
    write_json(folder / "early_health" / f"update_{update:09d}.json", dict(**checks, observation=observation))
    if checks["status"] != "passed":
        raise RuntimeError("Inherited fixed-panel chemistry checks failed")
    return observation


def generation(model, generators, folder, epoch):
    from .chemistry_epoch45_generation import classify_saved_graphs
    from .large_vae_evaluation import evaluate_codes
    output = folder / "generation"
    output.mkdir()
    rows = []
    with preserve_observation_state(model, generators):
        for seed in (140041, 140042):
            codes = torch.randn(1024, model.config.latent, generator=torch.Generator().manual_seed(seed))
            raw = output / f"prior-{seed}.jsonl"
            metrics = evaluate_codes(model, codes, raw, dict(evaluation_seed=seed,
                evaluation_batch_size=32, categorical_samples_per_latent=4))
            raw = Path(str(raw) + ".gz")
            rows.append(dict(seed=seed, metrics=metrics, detailed=classify_saved_graphs(raw, model.spec),
                             raw_graphs=str(raw), sha256=file_sha256(raw)))
    write_json(output / "metrics.json", dict(epoch=epoch, seeds=rows,
        scope="Pure Gaussian evaluation, same paired seeds as40--45; all invalid graphs retained"))


def execute(args):
    study, base = load_study(args.study, args.study_sha256)
    folder = Path(study["campaign"]) / ("probe" if args.mode == "probe" else f"epoch{args.epoch:04d}")
    folder.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.wall_seconds
    context = None
    try:
        context = setup(study, base, args.checkpoint, args.checkpoint_sha256, args.study_sha256)
        model, reference, optimizer, config, data, cohorts, generators, sampler, teacher, composition, payload = context
        if args.mode == "train" and (args.epoch != payload["epoch"]+1 or not 44 <= args.epoch <= 45):
            raise ValueError("Training must start at exactly the next authorized completed epoch")
        epoch = payload["epoch"]+1
        updates = payload["continuation"]["updates"]
        reference_sha = state_digest(reference.state_dict())
        initial_model_sha = state_digest(model.state_dict())
        if initial_model_sha != state_digest(payload["model_state"]):
            raise ValueError("Initial model differs from input checkpoint")
        write_json(folder / "handoff.json", dict(status="restored_explicit_objective_change",
            checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
            epoch=payload["epoch"], updates=updates, study_sha256=args.study_sha256,
            model_sha256=initial_model_sha, inherited_adamw_moments=True,
            inherited_rng=True, new_rng_seed=study["neighborhood_seed"], policy=study["neighborhood"],
            optimizer_step_control=STEP,
            repair_discovery=study.get("structural_adoption", {}).get("proposal_config")))
        # An independently supervised home-storage launcher bounds all cold
        # imports/source/data/checkpoint reads up to this exact handoff. This
        # marker is startup evidence only, never a passed scientific probe.
        if os.environ.get("DBMOL_LAUNCH_READY"):
            write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ["LSB_JOBID"],
                study_sha256=args.study_sha256, completed_epoch=payload["epoch"], updates=updates))
        initial = observe(model, reference, optimizer, config, data, composition, generators, base, folder, updates)
        if args.mode == "train":
            gate = read_json(Path(study["campaign"]) / "probe" / "result.json")
            if gate.get("status") != "passed_bounded_probe" or gate.get("study_sha256") != args.study_sha256:
                raise ValueError("This exact study has not passed its bounded GPU probe")
            if study.get("structural_adoption") and (
                    gate.get("source_checkpoint_sha256") != args.checkpoint_sha256
                    or gate.get("source_completed_epoch") != 44
                    or gate.get("repair_discovery") != study["structural_adoption"]["proposal_config"]):
                raise ValueError("Structural probe checkpoint/policy differs from training")
            if gate["projected_epoch_seconds"] + 3600 > deadline-time.monotonic():
                raise RuntimeError("Measured full epoch plus reserve does not fit this allocation")
        loader = DataLoader(Subset(data["train"], cohorts["train"]), batch_size=config.batch_size,
                            shuffle=True, generator=generators["train"], collate_fn=collate_graphs, num_workers=0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at_epoch(config, epoch)
        completed, total_real, total_synthetic, seconds, local_points = 0, 0, 0, [], 0
        guided_or_valid, search_errors = 0, 0
        started = time.monotonic()
        last_observation = initial
        with teacher, (folder / "updates.jsonl").open("x") as stream:
            for hard in loader:
                if time.monotonic() > deadline-1800:
                    raise TimeoutError("Whole-epoch deadline reached; partial checkpoint is not promotable")
                tick = time.monotonic()
                metrics = c.chemistry_update(model, reference, optimizer, hard, config, base["protocol"],
                    composition, teacher, generators, updates, epoch, base["parent_epoch"],
                    deadline=deadline-1800, overlap=True, step_control=STEP, prior_sampler=sampler)
                validate_optimizer_step_receipt(metrics["optimizer_step_control"], STEP)
                if not any(p.grad is not None for p in model.decoder.parameters()):
                    raise RuntimeError("Required live decoder gradient path is detached")
                completed += 1; updates += 1
                total_real += metrics["real_points"]; total_synthetic += metrics["fresh_prior_points"]
                local_points += metrics["latent_neighborhood"]["local_points"]
                statuses = metrics["repair_statuses"]["prior"]
                guided_or_valid += statuses.get("accepted", 0) + statuses.get("skipped_already_valid", 0)
                search_errors += sum(v.get("rejected_or_error", 0) for v in metrics["repair_statuses"].values())
                elapsed = time.monotonic()-tick
                seconds.append(elapsed)
                if completed <= 10 or completed % 100 == 0:
                    stream.write(json.dumps(dict(epoch=epoch, updates=updates, completed_updates=completed,
                        seconds=elapsed, metrics=metrics), allow_nan=False)+"\n"); stream.flush()
                    print(json.dumps(dict(epoch=epoch, updates=updates, seconds=elapsed,
                        total=metrics["total"], local_points=metrics["latent_neighborhood"]["local_points"])), flush=True)
                if completed in (10, 32) or completed % 100 == 0:
                    last_observation = observe(model, reference, optimizer, config, data, composition, generators, base, folder, updates)
                if args.mode == "probe" and completed == study["probe_updates"]:
                    break
        if time.monotonic() > deadline-900:
            raise TimeoutError("Insufficient time for terminal observation/checkpoint")
        # Avoid replacing an already-recorded panel at this update.
        if not (folder / "early_health" / f"update_{updates:09d}.json").exists():
            last_observation = observe(model, reference, optimizer, config, data, composition, generators, base, folder, updates)
        if state_digest(reference.state_dict()) != reference_sha:
            raise RuntimeError("Frozen35 occupancy anchor changed")
        anchor_identity = model._valid_reference_anchor.assert_unchanged()
        terminal_checks = dict(no_search_errors=search_errors == 0,
            prior_guidance_or_validity=guided_or_valid / total_synthetic >= base["protocol"]["minimum_prior_target_fraction"])
        if not all(terminal_checks.values()):
            raise RuntimeError("Inherited end-of-epoch repair coverage/search checks failed")
        elapsed = time.monotonic()-started
        projected = elapsed/completed * math.ceil(len(data["train"])/config.batch_size) * 1.2
        result = dict(status="passed_bounded_probe" if args.mode == "probe" else "completed_epoch",
            study_sha256=args.study_sha256, epoch=epoch, completed_updates=completed,
            source_checkpoint_sha256=args.checkpoint_sha256, source_completed_epoch=payload["epoch"],
            absolute_updates=updates, real_points=total_real, synthetic_points=total_synthetic,
            local_points=local_points, training_seconds=elapsed, mean_update_seconds=sum(seconds)/len(seconds),
            projected_epoch_seconds=projected, initial=initial, final=last_observation,
            terminal_checks=terminal_checks, prior_guided_or_valid=guided_or_valid,
            frozen_anchors_unchanged=True, frozen_reference=anchor_identity,
            repair_discovery=study.get("structural_adoption", {}).get("proposal_config"),
            scope="Finite attached GPU update and inherited health checks; not derivative/native parity qualification")
        if args.mode == "probe":
            if completed != study["probe_updates"] or projected+3600 > 82800:
                raise RuntimeError("Probe did not complete or epoch runtime does not fit23h")
        else:
            if total_real != len(data["train"]) or total_synthetic != 2*total_real:
                raise ValueError("Incomplete real/synthetic epoch membership")
            checkpoint = folder / f"epoch_{epoch:04d}.pt"
            updated = deepcopy(payload)
            updated.update(epoch=epoch, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                run_id=folder.name, source_archive_sha256=study["archive_sha256"],
                source_sha256=algorithm_identity()["sha256"])
            updated["continuation"] = dict(payload["continuation"], updates=updates,
                epoch_complete=True, rng=capture_rng_state(generators, True),
                algorithm=algorithm_identity(), phase_epoch_wall_seconds=elapsed)
            updated["continuation"].pop("optimizer_step_evidence", None)
            updated["chemistry_phase"] = dict(payload["chemistry_phase"], plan_sha256=args.study_sha256,
                optimizer_step_control=STEP, neighborhood_continuation=dict(version=VERSION,
                    study_sha256=args.study_sha256, base_plan_sha256=study["base_plan_sha256"],
                    policy=study["neighborhood"], start_update=sampler.start_update))
            if study.get("structural_adoption"):
                updated["chemistry_phase"]["structural_proposals"] = dict(study["structural_adoption"],
                    source_checkpoint_sha256=args.checkpoint_sha256)
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(updated, temporary); temporary.replace(checkpoint)
            result.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint))
            generation(model, generators, folder, epoch)
        write_json(folder / "result.json", result)
        return result
    except BaseException as exc:
        failure = dict(status="failed_not_adopted", error=f"{type(exc).__name__}: {exc}",
                       study_sha256=args.study_sha256, epoch_complete=False)
        if context is not None:
            try:
                model, _, optimizer, _, _, _, generators, *_ = context
                torch.save(dict(diagnostic_only=True, promotable=False, epoch_complete=False,
                    model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                    rng=capture_rng_state(generators, True)), folder / "failed.diagnostic.pt")
            except Exception as save_error:
                failure["snapshot_error"] = repr(save_error)
        write_json(folder / "failure.json", failure)
        raise
    finally:
        if context is not None:
            context[8].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--study-sha256", required=True)
    parser.add_argument("--mode", choices=("probe", "train"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--wall-seconds", type=int, default=82800)
    args = parser.parse_args()
    if not 3600 <= args.wall_seconds <= 82800:
        parser.error("Use a bounded1--23h internal allocation budget")
    execute(args)


if __name__ == "__main__":
    main()
