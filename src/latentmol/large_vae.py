"""Allocated benchmark and resumable real-only triangle-VAE experiment runner.

UNTESTED on HPC. Each stage needs an immutable release and authenticated data.
Benchmark weights are discarded. Training starts fresh with seed17, then resumes
model, optimizer and all RNGs only from complete epoch-boundary checkpoints.
The test split is inaccessible to monitoring and requires explicit final stage.
"""

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from latentmol.data import GraphDataset, collate_graphs
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from latentmol.large_data import validate_large_cache
from latentmol.large_vae_evaluation import evaluate_panel
from latentmol.losses import latent_kl, reconstruction_losses
from latentmol.models import GraphAutoencoder
from latentmol.models.autoencoder import comparable_train_config
from latentmol.train import (TrainConfig, algorithm_identity, compact_cohort_identity, runtime_identity,
                             seed_everything, state_digest, train, training_cohorts, validate_continuation)


def read_protocol(path):
    value = json.loads(path.read_text())
    if value["version"] != 1 or value["precision"] != "fp32":
        raise ValueError("Only version1 FP32 protocol is supported")
    if value["extra_prior_training_points"] or value["chemistry_training_weight"]:
        raise ValueError("This real-only study does not enable chemistry or extra prior training")
    if value["batch_candidates"] != [256, 512]:
        raise ValueError("This release benchmarks the declared batches256/512")
    positive_integers = ("benchmark_warmup_updates", "benchmark_measured_updates", "benchmark_repeats",
                         "monitor_every", "prior_count", "categorical_samples_per_latent", "evaluation_batch_size",
                         "local_geometry_count", "interpolation_pairs", "gradient_diagnostics_every_updates")
    if any(type(value[key]) is not int or value[key] < 1 for key in positive_integers):
        raise ValueError("Protocol intervals, counts and batch sizes must be positive integers")
    if value["evaluation_batch_size"] > 32:
        raise ValueError("Evaluation decoder batches are bounded to32")
    for key in ("segment_wall_budget_seconds", "evaluation_reserve_seconds", "benchmark_minimum_speedup_for_512"):
        if not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(f"Invalid positive finite budget: {key}")
    if not 0 < value["benchmark_memory_fraction_limit"] < 1:
        raise ValueError("Memory headroom fraction must be in(0,1)")
    if value["segment_wall_budget_seconds"] > 22 * 3600:
        raise ValueError("Keep at least one hour inside the23-hour allocation")
    if value["deep_monitor_epochs"] != [25, 50, 75, 100]:
        raise ValueError("Deep validation cadence must be25/50/75/100")
    if value["final_candidate_roles"] != ["epoch100", "validation_reconstruction", "validation_validity"]:
        raise ValueError("Final candidate roles must be fixed before test access")
    if any(type(value["requested_counts"][key]) is not int or value["requested_counts"][key] < 1
           for key in ("train", "val", "test")):
        raise ValueError("Requested split counts must be positive integers")
    if not value["local_geometry_scales"] or any(not math.isfinite(x) or x < 0 for x in value["local_geometry_scales"]):
        raise ValueError("Invalid neighborhood scales")
    if not value["interpolation_alphas"] or any(not math.isfinite(x) or not 0 <= x <= 1 for x in value["interpolation_alphas"]):
        raise ValueError("Invalid interpolation alphas")
    if any(len(value[key]) != len(set(value[key])) for key in ("local_geometry_scales", "interpolation_alphas")):
        raise ValueError("Panel coordinates must be unique")
    return value


def validate_config(config):
    """Reject unsupported precision/training/worker modes before expensive work."""
    from latentmol.models import ModelConfig
    expected = ModelConfig(hidden=96, latent=256, layers=4, decoder_variant="triangle_refine",
                           global_hidden=320, encoder_residual_layers=2, pair_hidden=48,
                           refinement_rounds=4, attention_heads=4, triangle_hidden=24)
    if (config.mode != "vae" or config.epochs != 100 or config.model != expected
            or torch.device(config.device).type != "cuda" or config.num_workers != 0
            or config.train_limit is not None or config.loss.size_weight != 0
            or config.batch_size not in (256, 512)):
        raise ValueError("Require declared100-epoch joint triangle VAE, CUDA FP32, workers0, full train and no size penalty")
    if any(not math.isfinite(x) or x < 0 for x in (config.beta, config.weight_decay, config.loss.padding_weight,
            config.loss.no_bond_weight, config.loss.charge_weight, config.loss.bond_weight)):
        raise ValueError("Training weights must be finite and nonnegative")
    if any(not math.isfinite(x) or x <= 0 for x in (config.learning_rate, config.clip_grad)):
        raise ValueError("Learning rate and clipping bound must be positive and finite")
    if type(config.kl_warmup_epochs) is not int or config.kl_warmup_epochs < 0:
        raise ValueError("KL warmup must be a nonnegative integer epoch count")
    return config


def assert_counts(model, protocol):
    actual = {"encoder": sum(p.numel() for p in model.encoder.parameters()),
              "decoder": sum(p.numel() for p in model.decoder.parameters()),
              "total": sum(p.numel() for p in model.parameters())}
    if actual != protocol["expected_parameters"]:
        raise ValueError(f"Architecture differs from declared parameter budget: {actual}")
    return actual


def assert_dataset_counts(dataset, protocol):
    if dataset.manifest.get("dataset_format_version") != 2:
        raise ValueError("Large-corpus training requires the authenticated disk-backed cache")
    observed = {key: dataset.manifest["counts"][key] for key in ("train", "val", "test")}
    if observed != protocol["requested_counts"]:
        raise ValueError(f"Prepared split counts differ from the declared study: {observed}")


def gradient_norm(parameters):
    values = [parameter.grad.detach().double().square().sum()
              for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def peak_host_memory_bytes():
    """Linux process RSS high-water mark; separate from GPU allocation or disk."""
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class TrainingObserver:
    """Sparse pre-clip group gradients plus fixed validation/prior checkpoints."""

    compact_indices = True

    def __init__(self, protocol):
        self.protocol = protocol
        self.hooks, self.activation_maxima = [], {}

    def attach(self, model, run, training, validation):
        self.validation = validation
        run.annotate(large_corpus_protocol=self.protocol, joint_encoder_decoder_training=True,
                     extra_random_prior_training_points=0, chemistry_training_weight=0,
                     numerical_precision="fp32", parameter_gate=assert_counts(model, self.protocol))
        for name, layer in list(model.decoder.rounds.named_children()):
            def observe(module, inputs, outputs, label=name):
                if self.capture:
                    for key, value in zip(("atoms", "pairs"), outputs):
                        peak = float(value.detach().abs().max())
                        if not math.isfinite(peak):
                            raise FloatingPointError(f"Nonfinite refinement activation {label}/{key}")
                        label_key = f"round{label}/{key}"
                        self.activation_maxima[label_key] = max(self.activation_maxima.get(label_key, 0), peak)
            self.hooks.append(layer.register_forward_hook(observe))
        self.capture = False

    def before_epoch(self, epoch):
        self.capture = False
        self.gradient_rows = []
        self.activation_maxima = {}
        torch.cuda.reset_peak_memory_stats()

    def before_batch(self, index):
        self.capture = index % self.protocol["gradient_diagnostics_every_updates"] == 0

    def after_backward(self, model, output, index):
        if self.capture:
            triangle = [p for name, p in model.decoder.named_parameters() if ".triangle." in name]
            self.gradient_rows.append({"encoder": gradient_norm(model.encoder.parameters()),
                                       "decoder": gradient_norm(model.decoder.parameters()),
                                       "triangle": gradient_norm(triangle),
                                       "posterior_mu_absmax": float(output.mu.detach().abs().max()),
                                       "posterior_logvar_min": float(output.logvar.detach().min()),
                                       "posterior_logvar_max": float(output.logvar.detach().max())})

    def after_epoch(self, model, epoch, run):
        self.capture = False
        result = {"pre_clip_gradient_diagnostics_observations": len(self.gradient_rows),
                  "pre_clip_group_gradient_means": {key: sum(row[key] for row in self.gradient_rows)
                                                       / len(self.gradient_rows)
                                                       for key in ("encoder", "decoder", "triangle")},
                  "sparse_activation_absmax": self.activation_maxima,
                  "posterior_mu_absmax": max(row["posterior_mu_absmax"] for row in self.gradient_rows),
                  "posterior_logvar_min": min(row["posterior_logvar_min"] for row in self.gradient_rows),
                  "posterior_logvar_max": max(row["posterior_logvar_max"] for row in self.gradient_rows),
                  "peak_allocated_vram_bytes_training_and_standard_validation": torch.cuda.max_memory_allocated(),
                  "peak_reserved_vram_bytes_training_and_standard_validation": torch.cuda.max_memory_reserved()}
        result["peak_host_resident_bytes_process_lifetime"] = peak_host_memory_bytes()
        deep = epoch in self.protocol["deep_monitor_epochs"]
        if deep or epoch % self.protocol["monitor_every"] == 0:
            started = time.monotonic()
            result["panel"] = evaluate_panel(model, self.validation, self.protocol,
                                             run.path / "monitor" / f"epoch{epoch:04d}",
                                             limit=4096, deep=deep)
            result["panel_seconds"] = time.monotonic() - started
        return result

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def benchmark_worker(config_path, protocol_path, data_path, batch_size, output):
    """One disposable process per batch/repeat; fresh weights and optimizer."""
    protocol = read_protocol(protocol_path)
    config = validate_config(replace(TrainConfig.from_yaml(config_path), batch_size=batch_size))
    seed_everything(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires allocated CUDA hardware")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 75 * 1024**3 or "A100" not in properties.name:
        raise RuntimeError("This benchmark requires the declared A10080GB allocation")
    dataset = GraphDataset(data_path, "train", validated_cache=validate_large_cache(data_path))
    assert_dataset_counts(dataset, protocol)
    updates = protocol["benchmark_warmup_updates"] + protocol["benchmark_measured_updates"]
    if len(dataset) < batch_size * updates:
        raise ValueError("Benchmark needs distinct real batches at the full requested batch size")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config.seed))[:batch_size * updates].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_graphs, num_workers=config.num_workers)
    model = GraphAutoencoder(dataset.spec, config.model).to(config.device)
    counts = assert_counts(model, protocol)
    initial_sha = state_digest(model.state_dict())
    initial_encoder = state_digest(model.encoder.state_dict())
    initial_decoder = state_digest(model.decoder.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    times, samples, clipped = [], 0, 0
    maximum_gradient = 0.0
    group_gradient_maxima = {"encoder": 0.0, "decoder": 0.0}
    try:
        iterator = iter(loader)
        for index in range(updates):
            torch.cuda.synchronize()
            started = time.monotonic()
            # Include disk/CPU collation and transfer in the throughput clock.
            batch = next(iterator).to(config.device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch.to_relaxed(model.spec), sample_latent=True)
            loss = reconstruction_losses(predicted.decoded, batch, config.loss)["reconstruction"]
            loss = loss + config.beta * latent_kl(predicted.mu, predicted.logvar)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite benchmark loss")
            loss.backward()
            for label, module in (("encoder", model.encoder), ("decoder", model.decoder)):
                group_gradient_maxima[label] = max(group_gradient_maxima[label], gradient_norm(module.parameters()))
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad, error_if_nonfinite=True))
            optimizer.step()
            torch.cuda.synchronize()
            if index + 1 == protocol["benchmark_warmup_updates"]:
                torch.cuda.reset_peak_memory_stats()
            elif index >= protocol["benchmark_warmup_updates"]:
                times.append(time.monotonic() - started)
                samples += len(batch.atoms)
                clipped += int(norm > config.clip_grad)
                maximum_gradient = max(maximum_gradient, norm)
        report = {"status": "passed", "batch_size": batch_size, "updates": len(times),
                  "seconds": times, "molecules_per_second": samples / sum(times),
                  "mean_update_seconds": sum(times) / len(times), "clipped_updates": clipped,
                  "pre_clip_gradient_max": maximum_gradient,
                  "group_gradient_maxima": group_gradient_maxima,
                  "encoder_weights_changed": state_digest(model.encoder.state_dict()) != initial_encoder,
                  "decoder_weights_changed": state_digest(model.decoder.state_dict()) != initial_decoder,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                  "peak_host_resident_bytes": peak_host_memory_bytes(),
                  "parameters": counts, "initial_model_sha256": initial_sha,
                  "runtime": runtime_identity(config.device), "training_molecules": len(dataset),
                  "dataset_sha256": dataset.manifest["dataset_sha256"],
                  "split_sha256": dataset.manifest["split_sha256"], "weights_discarded": True}
    except torch.cuda.OutOfMemoryError:
        report = {"status": "out_of_memory", "batch_size": batch_size, "weights_discarded": True}
    write_json(output, report)


def benchmark(config_path, protocol_path, data, output):
    protocol = read_protocol(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    with ExperimentRun(output / "records", "triangle-benchmark", protocol, Path.cwd()) as run:
        reports = {}
        for batch in protocol["batch_candidates"]:
            reports[batch] = []
            for repeat in range(protocol["benchmark_repeats"]):
                result_path = output / f"batch{batch}-repeat{repeat}.json"
                command = [sys.executable, "-m", "latentmol.large_vae", "benchmark-worker",
                           "--config", str(config_path), "--protocol", str(protocol_path),
                           "--data", str(data), "--batch-size", str(batch), "--output", str(result_path)]
                with (output / f"batch{batch}-repeat{repeat}.log").open("x") as handle:
                    subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
                reports[batch].append(json.loads(result_path.read_text()))
        eligible = {batch: values for batch, values in reports.items()
                    if all(value["status"] == "passed" and value["peak_reserved_bytes"] /
                           value["total_vram_bytes"] <= protocol["benchmark_memory_fraction_limit"] and
                           value["encoder_weights_changed"] and value["decoder_weights_changed"] and
                           all(math.isfinite(n) and n > 0 for n in value["group_gradient_maxima"].values())
                           for value in values)}
        if 256 not in eligible:
            raise RuntimeError("Batch256 failed engineering/memory headroom gate; do not start training")
        speed = {batch: sum(v["molecules_per_second"] for v in values) / len(values)
                 for batch, values in eligible.items()}
        selected = 512 if 512 in speed and speed[512] >= speed[256] * protocol["benchmark_minimum_speedup_for_512"] else 256
        # Original architecture comparison is exploratory; changing batch also
        # changes optimizer updates per epoch and is reported explicitly.
        config = replace(TrainConfig.from_yaml(config_path), batch_size=selected)
        validate_config(config)
        selected_path = output / "selected-training.yaml"
        selected_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        first = eligible[selected][0]
        identities = {(v["initial_model_sha256"], v["dataset_sha256"], v["split_sha256"],
                       json.dumps(v["runtime"], sort_keys=True))
                      for values in eligible.values() for v in values}
        if len(identities) != 1:
            raise ValueError("Benchmark repeats did not start with identical weights/data")
        gate = {"status": "passed", "selected_batch": selected,
                "config_sha256": file_sha256(config_path), "protocol_sha256": file_sha256(protocol_path),
                "selected_config_sha256": file_sha256(selected_path),
                "source_sha256": source_manifest(Path.cwd())["sha256"],
                "source_archive_sha256": run.record["source_archive_sha256"],
                "dataset_sha256": first["dataset_sha256"], "split_sha256": first["split_sha256"],
                "runtime": first["runtime"], "reports": reports,
                "conservative_epoch_seconds": max(v["mean_update_seconds"] for v in eligible[selected]) *
                math.ceil(first["training_molecules"] / selected) + protocol["evaluation_reserve_seconds"],
                "forecast_note": "short warmed real-batch extrapolation, not a completed epoch timing"}
        write_json(output / "passed.json", gate)
        run.annotate(gate=gate)


def authenticate_gate(config_path, protocol_path, data, gate_path):
    """Authenticate the sealed benchmark and its exact selected training config."""
    gate = json.loads(gate_path.read_text())
    selected_path = gate_path.parent / "selected-training.yaml"
    if (gate["status"] != "passed" or gate["config_sha256"] != file_sha256(config_path)
            or gate["protocol_sha256"] != file_sha256(protocol_path)
            or gate["selected_config_sha256"] != file_sha256(selected_path)
            or gate["source_sha256"] != source_manifest(Path.cwd())["sha256"]
            or gate["source_archive_sha256"] != os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Benchmark gate belongs to different code/configuration")
    cache = validate_large_cache(data)
    dataset = GraphDataset(data, "train", validated_cache=cache)
    assert_dataset_counts(dataset, read_protocol(protocol_path))
    if dataset.manifest["dataset_sha256"] != gate["dataset_sha256"]:
        raise ValueError("Benchmark dataset identity changed")
    config = validate_config(TrainConfig.from_yaml(selected_path))
    seed_everything(config.seed)
    if runtime_identity(config.device) != gate["runtime"]:
        raise ValueError("Allocated runtime/device differs from benchmark; rebenchmark explicitly")
    identity = {"gate_sha256": file_sha256(gate_path), "selected_config_sha256": file_sha256(selected_path),
                "protocol_sha256": file_sha256(protocol_path), "dataset_sha256": gate["dataset_sha256"],
                "split_sha256": gate["split_sha256"], "source_sha256": gate["source_sha256"],
                "source_archive_sha256": gate["source_archive_sha256"]}
    return gate, cache, config, identity


def segment(config_path, protocol_path, data, campaign, *, segment_id, gate_path):
    protocol = read_protocol(protocol_path)
    gate, cache, config, identity = authenticate_gate(config_path, protocol_path, data, gate_path)
    if not segment_id or not segment_id.replace("-", "").isalnum():
        raise ValueError("Need a unique simple segment ID")
    campaign.mkdir(parents=True, exist_ok=True)
    progress_path = campaign / "progress.json"
    resume = None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity:
            raise ValueError("Campaign progress belongs to another gate/config/data/source")
        if progress["epoch"] == config.epochs:
            raise ValueError("Training already completed; use the explicit final stage")
        resume = Path(progress["checkpoint"])
        if file_sha256(resume) != progress["checkpoint_sha256"]:
            raise ValueError("Continuation checkpoint was changed")
    segment_root = campaign / "segments" / segment_id
    segment_root.mkdir(parents=True, exist_ok=False)
    lock_path = campaign / "active-segment.json"
    if (campaign / "active-recovery.json").exists():
        raise ValueError("A recovery allocation is active")
    with lock_path.open("x") as handle:
        json.dump({"segment_id": segment_id, "identity": identity}, handle)
    write_json(segment_root / "intent.json", {"identity": identity, "resume_from": str(resume) if resume else None,
                                              "job_id": os.environ.get("LSB_JOBID")})
    observer = TrainingObserver(protocol)
    try:
        run_path = train(config, data, segment_root, resume_from=resume,
                         observer=observer, shared_cache=cache,
                         wall_budget_seconds=protocol["segment_wall_budget_seconds"],
                         initial_epoch_seconds=gate["conservative_epoch_seconds"])
    finally:
        observer.close()
        lock_path.unlink()
    record = json.loads((run_path / "run.json").read_text())
    checkpoint = run_path / "last.pt"
    write_json(progress_path, {"epoch": record["completed_epoch"], "checkpoint": str(checkpoint),
                               "checkpoint_sha256": file_sha256(checkpoint), "run_path": str(run_path),
                               "selected_config_sha256": identity["selected_config_sha256"],
                               "protocol_sha256": file_sha256(protocol_path),
                               "identity": identity, "stop_reason": record["stop_reason"]})


def recover(config_path, protocol_path, data, campaign, output, *, segment_id, gate_path):
    """Hold an exclusive recovery lock, including while GPU diagnostics run."""
    lock = campaign / "active-recovery.json"
    with lock.open("x") as handle:
        json.dump({"segment_id": segment_id, "output": str(output)}, handle)
    try:
        _recover(config_path, protocol_path, data, campaign, output, segment_id=segment_id, gate_path=gate_path)
    finally:
        lock.unlink()


def _recover(config_path, protocol_path, data, campaign, output, *, segment_id, gate_path):
    """Explicitly adopt one known failed segment; never pick an arbitrary newest file.

    A failed monitor does not invalidate the saved complete training epoch. The
    recovery allocation reruns its validation-only panel, then publishes a bound
    receipt. It never touches test data or restarts an unfinished optimizer step.
    """
    protocol = read_protocol(protocol_path)
    gate, cache, config, identity = authenticate_gate(config_path, protocol_path, data, gate_path)
    root = campaign / "segments" / segment_id
    if json.loads((root / "intent.json").read_text())["identity"] != identity:
        raise ValueError("Interrupted segment intent does not belong to this campaign")
    active = campaign / "active-segment.json"
    if active.exists() and json.loads(active.read_text()) != {"segment_id": segment_id, "identity": identity}:
        raise ValueError("A different training segment owns the campaign")
    runs = [path for path in root.iterdir() if path.is_dir() and (path / "last.pt").is_file()]
    if len(runs) != 1:
        raise ValueError("Need exactly one complete-checkpoint run in the declared interrupted segment")
    parent = runs[0]
    checkpoint_path = parent / "last.pt"
    validation = GraphDataset(data, "val", validated_cache=cache)
    training = GraphDataset(data, "train", validated_cache=cache)
    model, payload = load_model(checkpoint_path, config.device, validation.spec)
    validate_continuation(payload, config, training, algorithm_identity(), gate["runtime"],
                          compact_cohort_identity(training_cohorts(config, len(training), len(validation))))
    state = payload.get("continuation", {})
    if (not state.get("epoch_complete")
            or json.dumps(comparable_train_config(payload["train_config"]), sort_keys=True) != json.dumps(asdict(config), sort_keys=True)
            or payload["dataset_sha256"] != identity["dataset_sha256"]
            or state["split_sha256"] != identity["split_sha256"]
            or state["runtime"] != gate["runtime"]
            or state["algorithm"]["sha256"] != algorithm_identity()["sha256"]
            or payload["source_archive_sha256"] != identity["source_archive_sha256"]):
        raise ValueError("Interrupted checkpoint is incompatible with exact continuation")
    if not 1 <= payload["epoch"] <= config.epochs or payload["source_sha256"] != identity["source_sha256"]:
        raise ValueError("Interrupted checkpoint epoch/source identity differs")
    assert_counts(model, protocol)
    previous = campaign / "progress.json"
    if previous.exists():
        progress = json.loads(previous.read_text())
        if progress["identity"] != identity or progress["epoch"] > payload["epoch"]:
            raise ValueError("Recovery would replace a newer or differently bound completed epoch")
    with ExperimentRun(output, "triangle-recover", {"identity": identity}, Path.cwd()) as run:
        metrics = evaluate_panel(model, validation, protocol, run.path / "validation", limit=4096,
                                 deep=payload["epoch"] in protocol["deep_monitor_epochs"])
        # Copies recover missing mutable aliases from immutable, authenticated
        # selection snapshots; no checkpoint is selected on test performance.
        from latentmol.train import copy_checkpoint
        for label, entry in state["selections"].items():
            # last.pt is authoritative for its own completed epoch even if a
            # timeout interrupted the subsequent convenience snapshot copy.
            snapshot = checkpoint_path if entry["epoch"] == payload["epoch"] else Path(entry["path"])
            selected = torch.load(snapshot, map_location="cpu", weights_only=True)
            if state_digest(selected["model_state"]) != entry["model_sha256"] or selected["epoch"] != entry["epoch"]:
                raise ValueError("Interrupted selection snapshot differs")
            copy_checkpoint(snapshot, parent / ("best.pt" if label == "reconstruction" else "best_validity.pt"))
        receipt = {"status": "passed", "identity": identity, "checkpoint_sha256": file_sha256(checkpoint_path),
                   "epoch": payload["epoch"], "validation_only": True}
        write_json(run.path / "passed.json", receipt)
        run.annotate(metrics=metrics, receipt=receipt)
        write_json(campaign / "progress.json", {"epoch": payload["epoch"], "checkpoint": str(checkpoint_path),
                   "checkpoint_sha256": receipt["checkpoint_sha256"], "run_path": str(parent),
                   "identity": identity, "protocol_sha256": identity["protocol_sha256"],
                   "selected_config_sha256": identity["selected_config_sha256"], "stop_reason": "explicit_recovery",
                   "recovery_receipt": str(run.path / "passed.json")})
    lock = campaign / "active-segment.json"
    if lock.exists():
        if json.loads(lock.read_text()) != {"segment_id": segment_id, "identity": identity}:
            raise ValueError("Cannot clear a different active segment lock")
        lock.unlink()


def final(config_path, protocol_path, data, campaign, output, references, gate_path):
    """Only this explicit stage reads held-out test molecules; never selects on them."""
    protocol = read_protocol(protocol_path)
    progress = json.loads((campaign / "progress.json").read_text())
    gate, cache, config, identity = authenticate_gate(config_path, protocol_path, data, gate_path)
    if progress["identity"] != identity:
        raise ValueError("Final campaign lineage differs from the authenticated benchmark")
    if progress["epoch"] != config.epochs or progress["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("Final test requires the completed declared training horizon")
    parent = Path(progress["run_path"])
    checkpoint = Path(progress["checkpoint"])
    if file_sha256(checkpoint) != progress["checkpoint_sha256"]:
        raise ValueError("Final epoch checkpoint identity changed")
    if (campaign / "active-segment.json").exists() or (campaign / "active-recovery.json").exists():
        raise ValueError("Wait for the training/recovery allocation to complete before test access")
    parent_record = json.loads((parent / "run.json").read_text())
    if parent_record["status"] != "completed":
        receipt = json.loads(Path(progress["recovery_receipt"]).read_text())
        if receipt["status"] != "passed" or receipt["identity"] != identity or receipt["checkpoint_sha256"] != progress["checkpoint_sha256"]:
            raise ValueError("Failed segment has no completed authenticated recovery")
    final_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if final_payload["epoch"] != config.epochs or final_payload["dataset_sha256"] != identity["dataset_sha256"]:
        raise ValueError("Final checkpoint does not match declared completed training")
    output.mkdir(parents=True, exist_ok=False)
    dataset = GraphDataset(data, "test", validated_cache=cache)
    training = GraphDataset(data, "train", validated_cache=cache)
    training_smiles = set(training.smiles)
    candidates = {"epoch100": checkpoint, "validation_reconstruction": parent / "best.pt",
                  "validation_validity": parent / "best_validity.pt", **references}
    reports = {}
    with ExperimentRun(output / "records", "triangle-final", {"protocol": protocol}, Path.cwd()) as run:
        for label, path in candidates.items():
            model, payload = load_model(path, config.device, dataset.spec)
            if label in protocol["final_candidate_roles"]:
                assert_counts(model, protocol)
                if (comparable_train_config(payload["train_config"]) != comparable_train_config(final_payload["train_config"])
                        or payload["dataset_sha256"] != identity["dataset_sha256"]):
                    raise ValueError("Selected candidate belongs to different training settings/data")
                if label != "epoch100":
                    entry = final_payload["continuation"]["selections"][label.removeprefix("validation_")]
                    if payload["epoch"] != entry["epoch"] or state_digest(payload["model_state"]) != entry["model_sha256"]:
                        raise ValueError("Best alias differs from its authenticated validation selection")
            if model.config.latent != config.model.latent or payload["train_config"]["mode"] != "vae":
                raise ValueError("Comparison requires trained VAEs with the declared shared latent dimension")
            reports[label] = {"checkpoint_sha256": file_sha256(path), "checkpoint": str(path),
                              "epoch": payload["epoch"], "training_dataset_sha256": payload["dataset_sha256"],
                              "metrics": evaluate_panel(model, dataset, protocol, output / label,
                                                       limit=len(dataset), deep=True, training_smiles=training_smiles)}
            del model
            torch.cuda.empty_cache()
        write_json(output / "comparison.json", {"reports": reports, "test_dataset": dataset.manifest,
                                                "interpretation": "Historical comparison confounds architecture, data and training regime; no causal triangle-only claim"})
        run.annotate(comparison=str(output / "comparison.json"))


def reference_validation(config_path, protocol_path, data, output, references, gate_path):
    """Evaluate frozen historical references once on the fixed monitoring panel.

    This permits interim comparison at25/50/75 without opening held-out test.
    The reference checkpoint hashes and their original training data are saved;
    they do not initialize the new model or provide training targets.
    """
    protocol = read_protocol(protocol_path)
    gate, cache, config, identity = authenticate_gate(config_path, protocol_path, data, gate_path)
    validation = GraphDataset(data, "val", validated_cache=cache)
    training = GraphDataset(data, "train", validated_cache=cache)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    with ExperimentRun(output / "records", "triangle-reference-val", {"identity": identity}, Path.cwd()) as run:
        for label, path in references.items():
            model, payload = load_model(path, config.device, validation.spec)
            if model.config.latent != config.model.latent or payload["train_config"]["mode"] != "vae":
                raise ValueError("Historical reference must be a compatible trained VAE256")
            reports[label] = {"checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
                              "epoch": payload["epoch"], "training_dataset_sha256": payload["dataset_sha256"],
                              "metrics": evaluate_panel(model, validation, protocol, output / label,
                                                       limit=4096, deep=True, training_smiles=set(training.smiles))}
            del model
            torch.cuda.empty_cache()
        write_json(output / "comparison.json", {"status": "passed", "identity": identity, "reports": reports,
                                                "split": "val", "historical_comparison_not_causal_ablation": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "benchmark-worker", "segment", "recover", "reference-validation", "final"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--segment-id")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--reference", action="append", default=[], help="label=absolute_checkpoint_path")
    parser.add_argument("--reference-sha256", action="append", default=[], help="label=pinned_checkpoint_sha256")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        parser.error("Numerical stages require an allocated DTU HPC job")
    required = {"benchmark": ("output",), "benchmark-worker": ("output", "batch_size"),
                "segment": ("campaign", "gate", "segment_id"), "recover": ("campaign", "gate", "segment_id", "output"),
                "final": ("campaign", "gate", "output"), "reference-validation": ("gate", "output")}[args.stage]
    if any(getattr(args, key) is None for key in required):
        parser.error("Missing stage arguments: " + ", ".join("--" + key.replace("_", "-") for key in required))
    if args.stage == "benchmark":
        benchmark(args.config, args.protocol, args.data, args.output)
    elif args.stage == "benchmark-worker":
        benchmark_worker(args.config, args.protocol, args.data, args.batch_size, args.output)
    elif args.stage == "segment":
        segment(args.config, args.protocol, args.data, args.campaign, segment_id=args.segment_id, gate_path=args.gate)
    elif args.stage == "recover":
        recover(args.config, args.protocol, args.data, args.campaign, args.output,
                segment_id=args.segment_id, gate_path=args.gate)
    else:
        references = dict(item.split("=", 1) for item in args.reference)
        hashes = dict(item.split("=", 1) for item in args.reference_sha256)
        if set(hashes) != set(references) or "original" not in references:
            raise ValueError("Final comparison requires original and a pinned SHA256 for every reference")
        for label, path in references.items():
            if file_sha256(Path(path)) != hashes[label]:
                raise ValueError("Historical reference identity differs: " + label)
        if any(not label.replace("_", "").replace("-", "").isalnum() for label in references):
            raise ValueError("Reference labels must be simple names")
        if set(references) & {"epoch100", "validation_reconstruction", "validation_validity"}:
            raise ValueError("References cannot replace candidate roles")
        if args.stage == "reference-validation":
            reference_validation(args.config, args.protocol, args.data, args.output,
                                 {label: Path(path) for label, path in references.items()}, args.gate)
        else:
            final(args.config, args.protocol, args.data, args.campaign, args.output,
                  {label: Path(path) for label, path in references.items()}, args.gate)


if __name__ == "__main__":
    main()
