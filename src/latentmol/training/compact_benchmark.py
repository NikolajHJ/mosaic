"""Time real optimizer steps for the compact graph VAE on the sparse training view.

Both sizes replay the same source rows and Gaussian draws from one freshly
warmed model and AdamW state. Shape-specific warmups are disposable. Only
CUDA OOM triggers a fresh accumulated-batch attempt; no trial commits weights.
"""

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from latentmol.chemistry_continuation import chunks, hard_slice, reconstruction_chunk, reconstruction_denominators
from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.data import collate_graphs
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import file_sha256, write_json
from latentmol.models import GraphAutoencoder
from latentmol.train import TrainConfig, kl_weight, seed_everything, state_digest
from latentmol.training.combined_continuation import _check_cuda_requirement
from latentmol.training.losses import latent_kl


def read_settings(path):
    settings = json.loads(path.read_text(encoding="utf-8"))
    training_path = (path.parent / settings["training_config"]).resolve()
    training = TrainConfig.from_yaml(training_path)
    sizes = settings["batch_sizes"]
    if (settings["version"] != "compact-vae-benchmark-v1" or training.mode != "vae"
            or training.model.encoder_pooling != "canonical" or training.loss.atom_order != "graph"):
        raise ValueError("Require the compact graph-matched canonical VAE training configuration")
    minimum = settings["minimum_microbatch"]
    if (type(minimum) is not int or minimum < 1 or not sizes or len(sizes) != len(set(sizes))
            or any(type(n) is not int or n < minimum or n % minimum for n in sizes)):
        raise ValueError("Invalid logical batch or microbatch sizes")
    for key in ("shared_warmup_rows", "shape_warmup_rows", "timed_rows"):
        if type(settings[key]) is not int or settings[key] < 1 or any(settings[key] % n for n in sizes):
            raise ValueError(f"{key} must cover full batches at every requested size")
    if type(settings["prior_ratio"]) is not int or not 1 <= settings["prior_ratio"] <= min(sizes):
        raise ValueError("Invalid real to prior sampling ratio")
    return settings, training, training_path


def _optimizer(model, training):
    return torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)


def _batch(data, rows):
    return collate_graphs([data[i] for i in rows])


def prior_draw_count(start, real_count, ratio):
    """Carry the unfilled fraction across steps, e.g. 17, 17, 17, 17, …, 18 at 256:15."""
    return (start + real_count) // ratio - start // ratio


def _update(model, optimizer, training, hard, *, epsilon, beta, microbatch, guidance=None, prior=None):
    """Full logical-batch CE/KL means, one optional prior loss, then one AdamW step."""
    count = len(hard.atoms)
    device = next(model.parameters()).device
    hard = hard.to(device)
    denominators = reconstruction_denominators(hard, training.loss)
    optimizer.zero_grad(set_to_none=True)
    rec_total = kl_total = 0.
    for first, last in chunks(count, microbatch):
        target = hard_slice(hard, first, last)
        mu, logvar = model.encoder(target, from_canonical_cache=True)
        z = mu + (.5 * logvar).exp() * epsilon[first:last].to(device)
        decoded = model.decode(z)
        rec = reconstruction_chunk(decoded, target, training.loss, denominators)
        kl = latent_kl(mu, logvar) * (last - first) / count
        objective = rec + beta * kl
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("Nonfinite real reconstruction or KL")
        objective.backward()
        rec_total += float(rec.detach())
        kl_total += float(kl.detach())
        del mu, logvar, z, decoded, objective, rec, kl
    metrics = dict(reconstruction=rec_total, kl=kl_total, beta=beta, prior_count=0)
    if guidance is not None:
        prior_loss, details = guidance.loss(model, prior.to(device))
        if not prior_loss.requires_grad or not bool(torch.isfinite(prior_loss)):
            raise RuntimeError("Nonfinite or detached prior objective")
        prior_loss.backward()
        metrics.update(prior_count=len(prior), weighted_prior_loss=float(prior_loss.detach()),
                       prior_guidance=details)
    if any(value.grad is None for value in
           (model.encoder.mu.weight, model.encoder.logvar.weight, model.decoder.atom_head.weight,
            model.decoder.charge_head.weight, model.decoder.bond_head[-1].weight)):
        raise RuntimeError("Missing gradient on a required compact VAE head")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.clip_grad, error_if_nonfinite=True)
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p.detach()).all() for p in model.parameters()]).all()):
        raise FloatingPointError("Nonfinite parameters after AdamW")
    metrics["gradient_norm"] = float(norm)
    return metrics


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _steps(model, optimizer, training, data, rows, *, batch_size, microbatch, samples_before,
           noise_seed, ratio, guidance, device):
    # Pre-generate CPU noise for the whole finite panel; slicing it gives identical
    # per-molecule noise at both batch sizes and for exact OOM retries.
    rng = torch.Generator().manual_seed(noise_seed)
    epsilon = torch.randn(len(rows), training.model.latent, generator=rng)
    prior_samples = prior_draw_count(samples_before, len(rows), ratio)
    priors = torch.randn(prior_samples, training.model.latent, generator=rng) if guidance else None
    prior_cursor = 0
    history = []
    _sync(device)
    start = time.perf_counter()
    for i in range(0, len(rows), batch_size):
        hard = _batch(data, rows[i:i + batch_size])
        n = len(hard.atoms)
        take = prior_draw_count(samples_before + i, n, ratio) if guidance else 0
        metrics = _update(model, optimizer, training, hard, epsilon=epsilon[i:i + n],
                          beta=kl_weight(training, 1, samples_before + i + n), microbatch=microbatch,
                          guidance=guidance, prior=priors[prior_cursor:prior_cursor + take] if guidance else None)
        history.append(metrics)
        prior_cursor += take
    _sync(device)
    return history, time.perf_counter() - start


def _fresh_from_warm(snapshot, training, spec, device):
    weights, moments = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    model = GraphAutoencoder(spec, training.model).to(device)
    model.load_state_dict(weights)
    optimizer = _optimizer(model, training)
    optimizer.load_state_dict(moments)
    return model, optimizer


def _count_weights(settings, mode):
    """Use the same effective weights for the actual objective and arm record."""
    config = settings["prior_guidance"] if mode == "prior_guided" else {}
    return {name: config.get(name, 0.0) for name in ("count_kl_weight", "argmax_count_weight")}


def _guide(settings, data, device, mode):
    from latentmol.training.compact_prior import CompactPriorGuidance
    params = {**settings["prior_guidance"], **_count_weights(settings, mode),
              "count_histogram": data.manifest["active_atom_counts"]["train"]}
    return CompactPriorGuidance(params, data.spec, str(device))


def _phase(snapshot, training, data, rows, *, settings, guidance, batch_size, microbatch, device,
           timed, samples_before, prior_panel=None):
    model, optimizer = _fresh_from_warm(snapshot, training, data.spec, device)
    if device.type == "cuda" and timed:
        torch.cuda.reset_peak_memory_stats(device)
    history, seconds = _steps(model, optimizer, training, data, rows, batch_size=batch_size,
                              microbatch=microbatch, samples_before=samples_before,
                              noise_seed=settings["data_seed"] + (2 if timed else 1),
                              ratio=settings["prior_ratio"], guidance=guidance, device=device)
    result = dict(seconds=seconds, samples=len(rows), updates=len(history),
                real_molecules_per_second=len(rows) / seconds,
                projected_training_hours=len(data) * seconds / len(rows) / 3600,
                prior_draws=sum(row["prior_count"] for row in history), updates_detail=history,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
    if prior_panel is not None:
        # Record timing and peak memory before running the held-out evaluation.
        try:
            result["prior_panel"] = prior_panel.observe(model)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError("Held-out evaluation exceeded GPU memory, reduce prior_panel.batch_size") from error
    return result


def run(settings, training, data, output, device, metadata):
    device = torch.device(device)
    include_control = settings.get("include_prior_count_control", False)
    if type(include_control) is not bool:
        raise ValueError("include_prior_count_control must be a Boolean")
    if include_control and not any(w > 0 for w in _count_weights(settings, "prior_guided").values()):
        raise ValueError("The optional matched control requires at least one positive candidate count-objective weight")
    modes = ("real_only", "prior_guided_control", "prior_guided") if include_control else ("real_only", "prior_guided")
    prior_panel = None
    if "prior_panel" in settings:
        from latentmol.evaluation.compact_prior_panel import CompactPriorPanel
        panel_settings = settings["prior_panel"]
        if panel_settings["seed"] in range(settings["data_seed"], settings["data_seed"] + 3):
            raise ValueError("Held-out panel seed overlaps with benchmark training noise")
        prior_panel = CompactPriorPanel(training.model.latent, **panel_settings)
    seed_everything(training.seed)
    model = GraphAutoencoder(data.spec, training.model).to(device)
    optimizer = _optimizer(model, training)
    parameters = {part: sum(p.numel() for p in getattr(model, part).parameters()) for part in ("encoder", "decoder")}
    if sum(parameters.values()) != settings["expected_parameters"]:
        raise ValueError(f"Unexpected compact VAE parameter counts: {parameters}")
    warm, shape, measured = (settings[k] for k in ("shared_warmup_rows", "shape_warmup_rows", "timed_rows"))
    order = random.Random(settings["data_seed"]).sample(range(len(data)), warm + shape + measured)
    checked = verify_canonical_training_cache(data, order)
    report = dict(status="running", dataset_sha256=data.dataset_sha256, metadata=metadata, parameters=parameters,
                  training_rows=len(data), canonical_rows_checked=checked,
                  row_panel_sha256=hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest(),
                  shared_warmup_rows=warm, shape_warmup_rows=shape, timed_rows=measured,
                  initial_model_sha256=state_digest(model.state_dict()), arms=[])
    write_json(output / "benchmark.json", report)
    if marker := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(marker), dict(status="models_and_inputs_ready", nonce=os.environ["DBMOL_LAUNCH_NONCE"],
                                      job_id=os.environ["LSB_JOBID"]))
    # Use the minimum microbatch here so the shared starting point does not depend on
    # whether a proposed native shape fits. Encoding and data loading are real.
    _steps(model, optimizer, training, data, order[:warm], batch_size=min(settings["batch_sizes"]),
           microbatch=settings["minimum_microbatch"], samples_before=0, noise_seed=settings["data_seed"],
           ratio=settings["prior_ratio"], guidance=None, device=device)
    snapshot = io.BytesIO()
    torch.save((model.state_dict(), optimizer.state_dict()), snapshot)
    snapshot = snapshot.getvalue()
    report["common_warmed_model_sha256"] = state_digest(model.state_dict())
    if prior_panel is not None:
        report["prior_panel"] = {**prior_panel.metadata, "before": prior_panel.observe(model)}
    del model, optimizer
    for size in settings["batch_sizes"]:
        for mode in modes:
            arm = dict(mode=mode, batch_size=size, attempts=[], status="running")
            if include_control:
                arm.update(_count_weights(settings, mode))
            report["arms"].append(arm)
            micro = size
            while micro >= settings["minimum_microbatch"]:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                arm.update(active_microbatch=micro, status="running")
                write_json(output / "benchmark.json", report)
                guidance = None if mode == "real_only" else _guide(settings, data, device, mode)
                common = dict(settings=settings, guidance=guidance, batch_size=size, microbatch=micro,
                              device=device, samples_before=warm)
                try:
                    _phase(snapshot, training, data, order[warm:warm + shape], timed=False, **common)
                    if guidance is not None:
                        # Keep the warmed worker processes, discard shape-warmup science.
                        guidance.reset_panel()
                    measured_phase = _phase(snapshot, training, data, order[warm + shape:], timed=True,
                                            prior_panel=prior_panel, **common)
                except torch.cuda.OutOfMemoryError as error:
                    if device.type != "cuda":
                        raise
                    arm["attempts"].append(dict(microbatch=micro, status="cuda_oom", error=str(error)))
                    arm["status"] = "cuda_oom"
                    write_json(output / "benchmark.json", report)
                else:
                    arm["attempts"].append(dict(microbatch=micro, status="measured"))
                    arm.update(status="measured", microbatch=micro, native_batch=micro == size, **measured_phase)
                    break
                finally:
                    if guidance is not None:
                        guidance.close()
                micro //= 2
            arm.pop("active_microbatch")
            write_json(output / "benchmark.json", report)
    report["status"] = "completed"
    write_json(output / "benchmark.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        settings, training, training_path = read_settings(args.config)
        _check_cuda_requirement(settings, benchmark=True)
        torch.set_num_threads(settings["execution"]["torch_threads"])
        corpus = BalancedSparseCorpus(settings["data"], expected_sha256=settings["data_sha256"])
        data = corpus.split("train")
        if len(data) != settings["expected_training_rows"]:
            raise ValueError("Unexpected training split size")
        gpu = torch.cuda.get_device_properties(0)
        metadata = dict(job_id=os.environ["LSB_JOBID"], gpu=gpu.name, gpu_memory_bytes=gpu.total_memory,
                        cpus=settings["execution"]["gpu_cpus"], torch_threads=torch.get_num_threads(),
                        torch=str(torch.__version__), dtype="float32", config_sha256=file_sha256(args.config),
                        train_config_sha256=file_sha256(training_path))
        run(settings, training, data, args.output, "cuda:0", metadata)
    except Exception as error:
        write_json(args.output / "failure.json", dict(type=type(error).__name__, message=str(error)))
        raise


if __name__ == "__main__":
    main()
