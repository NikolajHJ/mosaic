"""Train the compact graph VAE on the balanced sparse view, one HPC allocation at a time."""

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

import torch

from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import file_sha256, write_json
from latentmol.models import GraphAutoencoder
from latentmol.models.autoencoder import comparable_train_config
from latentmol.train import TrainConfig, capture_rng_state, kl_weight, learning_rate_at_epoch, restore_rng_state, seed_everything
from latentmol.training.combined_continuation import _batch, _check_cuda_requirement, _order
from latentmol.training.compact_benchmark import _optimizer, _update, prior_draw_count
from latentmol.training.compact_prior import CompactOccupancyGuidance


VERSION = "compact-vae-training-v1"


def read_settings(path):
    settings = json.loads(path.read_bytes())
    training = TrainConfig.from_yaml((path.parent / settings["training_config"]).resolve())
    decay = settings.get("weight_decay", training.weight_decay)
    if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
        raise ValueError("AdamW weight_decay must be finite and nonnegative")
    training = replace(training, weight_decay=decay)
    if (settings["version"] != VERSION or training.mode != "vae"
            or training.model.encoder_pooling != "canonical" or training.loss.atom_order != "graph"
            or training.loss.size_weight or training.train_limit is not None):
        raise ValueError("Expected a full-split, graph-matched reconstruction and KL recipe")
    for key in ("checkpoint_every_updates", "panel_every_updates", "expected_training_rows", "expected_parameters"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("panel_early_updates", "keep_checkpoint_updates"):
        if any(type(n) is not int or n < 1 for n in settings.get(key, ())):
            raise ValueError(f"{key} must contain positive update numbers")
    if "prior_occupancy" in settings:
        occupancy = settings["prior_occupancy"]
        if (not isinstance(occupancy, dict)
                or set(occupancy) != {"real_per_prior", "seed", "count_kl_weight", "argmax_count_weight"}
                or type(occupancy["real_per_prior"]) is not int or not 1 <= occupancy["real_per_prior"] <= training.batch_size
                or type(occupancy["seed"]) is not int or not 0 <= occupancy["seed"] < 2**63):
            raise ValueError("Invalid compact prior occupancy settings")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (occupancy["count_kl_weight"], occupancy["argmax_count_weight"])):
            raise ValueError("Both occupancy correction weights must be finite and positive")
        if occupancy["seed"] in (training.seed + 1, settings["panel"]["seed"] + 1):
            raise ValueError("Occupancy training draws require a seed distinct from posterior and evaluation draws")
    return settings, training


def _restore(path, expected_sha, model, optimizer, training, generators, identity, dataset, count):
    if file_sha256(path) != expected_sha:
        raise ValueError("Resume checkpoint digest mismatch")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if (saved.get("compact_version") != VERSION or saved["config_sha256"] != identity
            or comparable_train_config(saved["train_config"]) != asdict(training) or saved["dataset_sha256"] != dataset
            or saved["graph_spec"] != model.spec.to_dict() or saved["training_molecules"] != count):
        raise ValueError("Resume checkpoint belongs to a different compact training recipe")
    state = saved["continuation"]
    cursor = {key: state[key] for key in ("epoch", "row", "updates", "real_points")}
    epoch, row = cursor["epoch"], cursor["row"]
    size = training.batch_size
    if (not 0 <= epoch < training.epochs or not 0 <= row < count or row % size
            or cursor["real_points"] != count * epoch + row
            or cursor["updates"] != math.ceil(count / size) * epoch + row // size):
        raise ValueError("Incompatible or already completed compact training cursor")
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    restore_rng_state(state["rng"], generators)
    return cursor


def _save(output, model, optimizer, training, generators, identity, data, cursor):
    path = output / f"checkpoint-{cursor['updates']:09d}.pt"
    if path.exists():
        raise FileExistsError(path)
    payload = dict(format_version=1, compact_version=VERSION, config_sha256=identity,
                   graph_spec=model.spec.to_dict(), model_config=asdict(model.config),
                   train_config=asdict(training), loss_config=asdict(training.loss), epoch=cursor["epoch"],
                   dataset_sha256=data.dataset_sha256, training_molecules=len(data), run_id=output.name,
                   model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                   continuation=dict(cursor, epoch_complete=cursor["row"] == 0, rng=capture_rng_state(
                       generators, next(model.parameters()).is_cuda)))
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
    temporary.replace(path)
    receipt = dict(checkpoint=str(path.resolve()), sha256=file_sha256(path), **cursor, completed_epoch=cursor["epoch"])
    write_json(output / "latest.json", receipt)
    return receipt


def train_segment(settings, training, data, output, device, identity, panel, *, resume=None,
                  resume_sha=None, deadline=None, max_updates=None):
    """Execute real updates. The epoch permutation, posterior noise and optimizer survive segmentation."""
    if next(output.iterdir(), None) is not None:
        raise FileExistsError(f"Compact segment output is occupied: {output}")
    # Direct library callers may share a newly created directory. Claim it atomically
    # before model setup; the CLI also owns its output via mkdir(exist_ok=False).
    with (output / "run.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "initializing", "config_sha256": identity}, stream)
    count = len(data)
    if count != settings["expected_training_rows"] or data.dataset_sha256 != settings["data_sha256"]:
        raise ValueError("Dataset identity or training split length differs from the recipe")
    if (resume is None) != (resume_sha is None):
        raise ValueError("Resume requires both a checkpoint path and its digest")
    checked = verify_canonical_training_cache(data, range(min(16, count)))
    seed_everything(training.seed)
    model = GraphAutoencoder(data.spec, training.model).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != settings["expected_parameters"]:
        raise ValueError(f"Expected {settings['expected_parameters']} parameters, found {parameters}")
    optimizer = _optimizer(model, training)
    generators = {"epsilon": torch.Generator().manual_seed(training.seed + 1)}
    occupancy = settings.get("prior_occupancy")
    guidance = None
    if occupancy is not None:
        histogram = data.manifest["active_atom_counts"]["train"]
        if sum(histogram.values()) != count:
            raise ValueError("Count reference must describe every real training row")
        guidance = CompactOccupancyGuidance(occupancy, histogram, data.spec.max_nodes, device)
        generators["prior_occupancy"] = torch.Generator().manual_seed(occupancy["seed"])
    cursor = dict(epoch=0, row=0, updates=0, real_points=0)
    if resume is not None:
        cursor = _restore(resume, resume_sha, model, optimizer, training, generators, identity, data.dataset_sha256, count)
    initial = cursor["updates"]
    size = training.batch_size
    write_json(output / "run.json", dict(version=VERSION, config_sha256=identity, dataset_sha256=data.dataset_sha256,
               parameters=parameters, training_molecules=count, starting_cursor=cursor, batch_size=size,
               canonical_rows_checked=checked, train_config=asdict(training), job_id=os.environ.get("LSB_JOBID"),
               device=str(device), gpu=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
               torch=str(torch.__version__), panel=panel.metadata, prior_occupancy=occupancy, resume_sha256=resume_sha))
    if marker := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(marker), dict(status="models_and_inputs_ready", nonce=os.environ["DBMOL_LAUNCH_NONCE"],
                                     job_id=os.environ["LSB_JOBID"]))
    latest = disposable = None
    keep = set(settings.get("keep_checkpoint_updates", ()))
    last_panel = None
    started = time.monotonic()
    stop_reason = "epoch_target_reached"
    order = None
    model.train()
    with (output / "quality-panel.jsonl").open("x", encoding="utf-8") as quality, (
            output / "training.jsonl").open("x", encoding="utf-8") as traces:
        def observe():
            nonlocal last_panel
            if last_panel != cursor["updates"]:
                quality.write(json.dumps({**cursor, **panel.observe(model)}, allow_nan=False) + "\n")
                quality.flush()
                last_panel = cursor["updates"]

        observe()
        while cursor["epoch"] < training.epochs:
            if max_updates is not None and cursor["updates"] - initial >= max_updates:
                stop_reason = "update_budget_reached"
                break
            if deadline is not None and time.monotonic() >= deadline - 120:
                stop_reason = "allocation_budget_reached"
                break
            epoch = cursor["epoch"] + 1
            if order is None:
                order = _order(count, settings["shuffle_seed"], epoch)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate_at_epoch(training, epoch)
            hard = _batch(data, order, cursor["row"], size)
            n = len(hard.atoms)
            eps = torch.randn(n, training.model.latent, generator=generators["epsilon"])
            beta = kl_weight(training, epoch, cursor["real_points"] + n)
            tick = time.monotonic()
            prior = None
            if guidance is not None:
                n_prior = prior_draw_count(cursor["real_points"], n, occupancy["real_per_prior"])
                if n_prior:
                    guidance.update = cursor["updates"]
                    prior = torch.randn(n_prior, training.model.latent, generator=generators["prior_occupancy"])
            metrics = _update(model, optimizer, training, hard, epsilon=eps, beta=beta, microbatch=size,
                              guidance=guidance if prior is not None else None, prior=prior)
            if occupancy is not None:
                metrics["total_prior_points"] = (cursor["real_points"] + n) // occupancy["real_per_prior"]
            cursor.update(updates=cursor["updates"] + 1, row=cursor["row"] + n, real_points=cursor["real_points"] + n)
            end = cursor["row"] == count
            if end:
                cursor.update(epoch=epoch, row=0)
                order = None
            update = cursor["updates"]
            if update == initial + 1 or update % 100 == 0 or end:
                trace = {**cursor, **metrics, "optimizer_step_seconds": time.monotonic() - tick}
                traces.write(json.dumps(trace, allow_nan=False) + "\n")
                traces.flush()
                write_json(output / "progress.json", trace)
            if end or update % settings["checkpoint_every_updates"] == 0 or update in keep:
                latest = _save(output, model, optimizer, training, generators, identity, data, cursor)
                if disposable is not None:
                    disposable.unlink()
                disposable = None if end or update in keep else Path(latest["checkpoint"])
            if end or update in settings["panel_early_updates"] or update % settings["panel_every_updates"] == 0:
                observe()
        if cursor["updates"] <= initial:
            raise RuntimeError("Allocation ended before the compact VAE could complete an optimizer update")
        if latest is None or latest["updates"] != cursor["updates"]:
            latest = _save(output, model, optimizer, training, generators, identity, data, cursor)
            if disposable is not None:
                disposable.unlink()
        observe()
    summary = dict(status="completed" if stop_reason == "epoch_target_reached" else "paused",
                   stop_reason=stop_reason, latest=latest, dataset_sha256=data.dataset_sha256,
                   training_molecules=count, segment_updates=cursor["updates"] - initial, batch_size=size,
                   microbatch_size=size, elapsed_seconds=time.monotonic() - started, **cursor)
    if occupancy is not None:
        summary["total_prior_points"] = cursor["real_points"] // occupancy["real_per_prior"]
    write_json(output / "summary.json", summary)
    return summary


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--budget-seconds", "--wall-seconds", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    args = parser.parse_args()
    settings, training = read_settings(args.config)
    if any(s is not None and s != training.batch_size for s in (args.batch_size, args.microbatch_size)):
        parser.error("Requested batch size differs from the native compact training recipe")
    if bool(args.resume) != bool(args.resume_sha256):
        parser.error("Resume requires both path and SHA-256")
    if args.budget_seconds is not None and (not math.isfinite(args.budget_seconds) or args.budget_seconds <= 120):
        parser.error("Allocation budget must leave at least 120 s for final measurements and saving")
    if not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Production CUDA training requires an allocated DTU GPU")
    _check_cuda_requirement(settings)
    torch.set_num_threads(settings["execution"]["torch_threads"])
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        from latentmol.evaluation.compact_training_panel import CompactTrainingPanel
        corpus = BalancedSparseCorpus(settings["data"], expected_sha256=settings["data_sha256"])
        panel = CompactTrainingPanel(corpus.split("val"), training, **settings["panel"])
        train_segment(settings, training, corpus.split("train"), args.output, "cuda", file_sha256(args.config), panel,
                      resume=args.resume, resume_sha=args.resume_sha256,
                      deadline=None if args.budget_seconds is None else started + args.budget_seconds)
    except Exception as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    main()
