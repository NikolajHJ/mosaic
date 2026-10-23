"""Matched, decoder-only chemical-proximity continuation on DTU HPC.

This is a new optimizer experiment from a frozen VAE checkpoint, not an exact
resume and not training a learned chemistry prior. Chemistry remains an
explicit probabilistic rule surrogate. The untouched original and matched
continuations are evaluated on identical held-out latent panels. Production
model training is unvalidated until retained native HPC evidence exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from latentmol.chemistry_regularization import ChemistryRegularizationConfig, rule_record
from latentmol.connectivity_regularization import ConnectivityRegularizationConfig
from latentmol.data import GraphDataset, collate_graphs
from latentmol.decoder_regularization import chemistry_objective, freeze_encoder, reference_anchors, validate_coefficient
from latentmol.evaluate import load_model, reconstruction_metrics
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.losses import LossConfig, latent_kl, reconstruction_losses
from latentmol.sampling import subset_indices
from latentmol.train import seed_everything, state_digest


class FineTuneRun(ExperimentRun):
    """Attach the retained failed run path to the controller's error receipt."""
    def __exit__(self, exception_type, exception, tb):
        if exception is not None:
            exception.experiment_run_dir = str(self.path)
        return super().__exit__(exception_type, exception, tb)


@dataclass(frozen=True)
class FineTuneConfig:
    epochs: int = 10
    batch_size: int = 64
    # None explicitly matches each actual reconstruction batch, including tails.
    # The historical integer default and serialized field set stay unchanged.
    prior_batch_size: int | None = 16
    chemistry_reconstruction_batch: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 1e-5
    clip_grad: float = 5.0
    occupancy_anchor_weight: float = 1.0
    aromatic_anchor_weight: float = 1.0
    chemistry: dict = field(default_factory=lambda: {"isolated_weight": 0.0})
    connectivity: dict = field(default_factory=lambda: {"draws": 4, "weight": 1.0})

    def __post_init__(self):
        for name in ("epochs", "batch_size", "chemistry_reconstruction_batch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if self.prior_batch_size is not None and (type(self.prior_batch_size) is not int or self.prior_batch_size < 1):
            raise ValueError("prior_batch_size must be a positive integer or null (match actual batch)")
        if self.chemistry_reconstruction_batch > self.batch_size:
            raise ValueError("Reconstruction chemistry subset exceeds batch")
        for name in ("learning_rate", "weight_decay", "clip_grad", "occupancy_anchor_weight", "aromatic_anchor_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if self.learning_rate == 0 or self.clip_grad == 0 or self.occupancy_anchor_weight == 0 or self.aromatic_anchor_weight == 0:
            raise ValueError("Learning/clip and common anti-collapse anchors must be positive")
        local = ChemistryRegularizationConfig.from_mapping(self.chemistry)
        if local.size_weight != 0:
            raise ValueError("This protocol uses common occupancy KL anchors; optional reference-size loss is unsupported")
        connection = ConnectivityRegularizationConfig.from_mapping(self.connectivity)
        if connection.weight <= 0:
            raise ValueError("All arms must consume the same enabled connectivity draw budget")

    def prior_count(self, real_count):
        """Resolve declared count without changing historical fixed-int quotas."""
        if type(real_count) is not int or not 0 < real_count <= self.batch_size:
            raise ValueError("Actual reconstruction batch exceeds its declared quota")
        return real_count if self.prior_batch_size is None else self.prior_batch_size


def tensor_sha(tensor):
    """Hash shape, dtype and canonical CPU bytes, not a pickle serialization."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_panel(count, latent, seed):
    return torch.randn(count, latent, generator=torch.Generator(device="cpu").manual_seed(seed))


def make_generators(seed, device):
    """Independent streams prevent evaluation/MC from moving training inputs."""
    return {name: torch.Generator(device=which).manual_seed(seed + offset)
            for name, which, offset in (("loader", "cpu", 0), ("posterior", "cpu", 100000),
                 ("prior", "cpu", 200000), ("connectivity", device, 300000))}


def rng_receipt(generators):
    return {name: tensor_sha(value.get_state()) for name, value in generators.items()}


def subset_loader(data, limit, seed, batch_size, *, shuffle=False, generator=None):
    indices = subset_indices(len(data), min(limit, len(data)), seed, shuffle_full=False)
    return DataLoader(Subset(data, indices), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, collate_fn=collate_graphs, num_workers=0)


def _slice(output, count):
    from latentmol.representation import DecoderOutput
    return DecoderOutput(output.atoms[:count], output.charges[:count], output.bonds[:count])


def checkpoint_payload(model, original, *, epoch, coefficient, seed, config, optimizer, generators, encoder_sha):
    """Evaluation-compatible checkpoint; no historical exact-resume metadata."""
    return {"format_version": 1, "graph_spec": model.spec.to_dict(),
        "model_config": asdict(model.config), "model_state": model.state_dict(),
        "loss_config": original["loss_config"], "train_config": {"mode": "vae", "beta": original["train_config"].get("beta", 0)},
        "dataset_sha256": original["dataset_sha256"], "epoch": epoch,
        "run_id": "decoder-only-continuation", "beta_at_checkpoint": original.get("beta_at_checkpoint", 0),
        "decoder_finetune": {"version": 1, "epoch": epoch, "original_epoch": original["epoch"],
            "coefficient": coefficient, "seed": seed, "config": asdict(config),
            "encoder_sha256": encoder_sha, "exact_historical_resume": False,
            "optimizer_state": optimizer.state_dict(),
            "rng_states": {name: rng.get_state() for name, rng in generators.items()}}}


def fit_case(initializer, train_data, val_data, output_root, config, *, seed, coefficient,
             train_limit, val_limit, validation_panel, device, source_root, budget_seconds,
             extra_guidance=None, resume=None, profile_updates=0,
             stop_after_epoch=None, epoch_extension=None, checkpoint_keep_epochs=None,
             checkpoint_keep_recent=2, archive_validation_rows=False):
    """Fixed endpoint continuation; checkpoints cannot be selected using test data.

    Chemistry uses the first configured examples in each randomly shuffled
    reconstruction batch, and a fresh independent prior batch. Equal 1/2
    weights prevent their unequal batch sizes from changing the objective.
    All streams and their terminal hashes are retained even in the zero arm.
    """
    validate_coefficient(coefficient)
    if type(profile_updates) is not int or profile_updates < 0:
        raise ValueError("Profile quota must be a nonnegative integer")
    stop_epoch = config.epochs if stop_after_epoch is None else stop_after_epoch
    if type(stop_epoch) is not int or not 0 < stop_epoch <= config.epochs:
        raise ValueError("Execution epoch stop must be inside the scientific training horizon")
    if epoch_extension is not None and resume is None:
        raise ValueError("Epoch extension requires an authenticated resume")
    if type(checkpoint_keep_recent) is not int or checkpoint_keep_recent < 2:
        raise ValueError("Keep at least two completed checkpoints for recovery")
    if type(archive_validation_rows) is not bool:
        raise ValueError("Validation row archiving must be an explicit boolean")
    if checkpoint_keep_epochs is not None:
        checkpoint_keep_epochs = list(checkpoint_keep_epochs)
        if (any(type(epoch) is not int or epoch < 1 for epoch in checkpoint_keep_epochs)
                or len(set(checkpoint_keep_epochs)) != len(checkpoint_keep_epochs)):
            raise ValueError("Checkpoint milestones must be distinct positive integers")
    seed_everything(seed)
    model, original = load_model(initializer, device, expected_spec=train_data.spec)
    reference, _ = load_model(initializer, device, expected_spec=train_data.spec)
    reference.requires_grad_(False)
    reference.eval()
    parameters = freeze_encoder(model)
    encoder_sha = state_digest(model.encoder.state_dict())
    initial_decoder_sha = state_digest(model.decoder.state_dict())
    reference_sha = state_digest(reference.state_dict())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    generators = make_generators(seed, device)
    loader = subset_loader(train_data, train_limit, 17, config.batch_size,
                           shuffle=True, generator=generators["loader"])
    val_loader = subset_loader(val_data, val_limit, 81000, config.batch_size)
    loss_config = LossConfig(**original["loss_config"])
    chemistry = ChemistryRegularizationConfig.from_mapping(config.chemistry)
    connectivity = ConnectivityRegularizationConfig.from_mapping(config.connectivity)
    identity = {"procedure": "frozen_encoder_decoder_continuation_v1", "config": asdict(config),
        "initializer": str(initializer), "initializer_sha256": file_sha256(initializer),
        "coefficient": coefficient, "seed": seed, "train_count": len(loader.dataset),
        "validation_count": len(val_loader.dataset), "validation_panel_sha256": tensor_sha(validation_panel),
        "dataset_sha256": train_data.manifest["dataset_sha256"],
        "split_sha256": train_data.manifest["split_sha256"], "rules": rule_record(model.spec),
        "endpoint": "last_fixed_epoch_only", "frozen_encoder_sha256": encoder_sha,
        "initial_decoder_sha256": initial_decoder_sha, "frozen_reference_state_sha256": reference_sha,
        "common_anchors": "KL(reference occupancy || current), KL(reference aromatic || current); both arms",
        "chemistry_on_prior_and_posterior": "equal half weights; posterior subset of shuffled batch",
        "optimizer": "new AdamW; historical optimizer not resumed",
        "extra_guidance": None if extra_guidance is None else extra_guidance.identity()}
    resumed = None
    if resume is not None:
        if extra_guidance is not None:
            raise ValueError("Search teacher continuation needs its own state protocol")
        from latentmol.decoder_resume import restore_training
        resumed = restore_training(model, optimizer, generators, original, config, identity, resume,
                                   epoch_extension=epoch_extension)
        identity["resume"] = {**resume, **resumed}
        identity["optimizer"] = "Restored decoder AdamW from completed epoch"
    if stop_after_epoch is not None:
        identity["stop_after_epoch"] = stop_epoch
    if checkpoint_keep_epochs is not None or archive_validation_rows:
        identity["storage_policy"] = {"checkpoint_keep_epochs": checkpoint_keep_epochs,
            "checkpoint_keep_recent": checkpoint_keep_recent, "archive_validation_rows": archive_validation_rows}
    if resumed is not None and stop_epoch <= resumed["epoch"]:
        raise ValueError("Execution stop must advance beyond the restored epoch")
    started = time.monotonic()
    update_profile = []
    if profile_updates and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    safe_coefficient = f"{coefficient:g}".replace(".", "p")
    suffix = "" if extra_guidance is None else "-" + extra_guidance.name
    with FineTuneRun(output_root, f"decoder-s{seed}-a{safe_coefficient}" + suffix, identity, source_root) as run:
        write_json(run.path / "initial_rng.json", rng_receipt(generators))
        total_updates = 0 if resumed is None else resumed["cumulative_updates"]
        first_epoch = 1 if resumed is None else resumed["epoch"] + 1
        for epoch in range(first_epoch, stop_epoch + 1):
            sums, examples, steps = {}, 0, 0
            counts = {"reconstruction": 0, "posterior_chemistry": 0,
                      "fresh_prior": 0, "connectivity_graph_draws": 0}
            clipped_updates, largest_gradient = 0, 0.0
            if extra_guidance is not None:
                extra_guidance.begin_epoch(model, reference, epoch, total_updates, run.path)
            for hard in loader:
                if time.monotonic() - started > budget_seconds:
                    run.annotate(comparison_status="censored_time_budget", completed_epochs=epoch - 1)
                    raise TimeoutError("Case time budget exhausted; incomplete endpoint is not comparable")
                hard = hard.to(device)
                size = len(hard.atoms)
                prior_count = config.prior_count(size)
                posterior_count = min(size, config.chemistry_reconstruction_batch)
                profiling = len(update_profile) < profile_updates
                if profiling and str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                update_started = time.monotonic()
                with torch.no_grad():
                    mu, logvar = model.encode(hard.to_relaxed(model.spec))
                    noise = torch.randn(mu.shape, generator=generators["posterior"]).to(device)
                    posterior = mu + (0.5 * logvar).exp() * noise
                decoded = model.decode(posterior)
                prior = torch.randn(prior_count, model.config.latent,
                                    generator=generators["prior"]).to(device)
                prior_decoded = model.decode(prior)
                with torch.no_grad():
                    original_prior = reference.decode(prior)
                rec = reconstruction_losses(decoded, hard, loss_config)
                anchors = reference_anchors(prior_decoded, original_prior, model.spec)
                posterior_chemistry = chemistry_objective(_slice(decoded, config.chemistry_reconstruction_batch),
                    model.spec, chemistry, connectivity, generator=generators["connectivity"])
                prior_chemistry = chemistry_objective(prior_decoded, model.spec, chemistry,
                    connectivity, generator=generators["connectivity"])
                chemical_loss = 0.5 * (posterior_chemistry["chemistry"] + prior_chemistry["chemistry"])
                extra_loss = None
                extra_values = {}
                if extra_guidance is not None:
                    extra_loss, extra_values = extra_guidance.loss(model, reference, total_updates)
                loss = (rec["reconstruction"] + coefficient * chemical_loss
                    + config.occupancy_anchor_weight * anchors["occupancy_anchor"]
                    + config.aromatic_anchor_weight * anchors["aromatic_anchor"])
                if extra_loss is not None:
                    loss = loss + extra_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite decoder objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.clip_grad, error_if_nonfinite=True)
                optimizer.step()
                norm_value = float(norm)
                clipped_updates += int(norm_value > config.clip_grad)
                largest_gradient = max(largest_gradient, norm_value)
                if any(not bool(torch.isfinite(p).all()) for p in parameters):
                    raise FloatingPointError("Nonfinite decoder parameter after optimizer step")
                counts["reconstruction"] += size
                counts["posterior_chemistry"] += posterior_count
                counts["fresh_prior"] += prior_count
                counts["connectivity_graph_draws"] += connectivity.draws * (posterior_count + prior_count)
                if profiling:
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    observation = {"epoch": epoch, "update": total_updates,
                        "seconds": time.monotonic() - update_started,
                        "real_points": size, "posterior_chemistry_points": posterior_count,
                        "fresh_prior_points": prior_count,
                        "gradient_norm_before_clip": float(norm), "loss": float(loss.detach()),
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if str(device).startswith("cuda") else None}
                    update_profile.append(observation)
                    # Keep profiles separate: resume authenticates exactly one
                    # completed-epoch entry per row of metrics.jsonl.
                    with (run.path / "update_profile.jsonl").open("a") as handle:
                        handle.write(json.dumps(observation, allow_nan=False) + "\n")
                values = {**rec, **anchors, "total": loss, "chemistry": chemical_loss,
                          "encoder_kl_monitor_only": latent_kl(mu, logvar), "gradient_norm_before_clip": norm}
                values.update({"posterior_" + k: v for k, v in posterior_chemistry.items()})
                values.update({"prior_" + k: v for k, v in prior_chemistry.items()})
                values.update(extra_values)
                size = hard.atoms.shape[0]
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * size
                examples += size
                steps += 1
                total_updates += 1
            if state_digest(model.encoder.state_dict()) != encoder_sha or model.encoder.training:
                raise RuntimeError("Frozen encoder weights/buffers or eval mode changed")
            if state_digest(reference.state_dict()) != reference_sha or reference.training:
                raise RuntimeError("Frozen common reference changed")
            terminal = rng_receipt(generators)
            path = run.path / f"epoch-{epoch:03d}.pt"
            payload = checkpoint_payload(model, original, epoch=epoch, coefficient=coefficient,
                seed=seed, config=config, optimizer=optimizer, generators=generators, encoder_sha=encoder_sha)
            if resumed is not None:
                payload["decoder_finetune"]["resume"] = {**resume, **resumed}
            if extra_guidance is not None:
                payload["decoder_finetune"]["extra_guidance"] = extra_guidance.receipt()
            torch.save(payload, path)
            model.decoder.eval()
            validation, _ = reconstruction_metrics(model, val_loader, loss_config, device,
                chemistry_limit=len(val_loader.dataset), training_mode="vae", graded=False,
                training_smiles=set(train_data.smiles))
            from latentmol.decoder_quality_evaluation import evaluate_random_panel
            prior_metrics = evaluate_random_panel(model, validation_panel, set(train_data.smiles),
                run.path / f"validation-prior-{epoch:03d}", batch_size=config.batch_size,
                samples_per_latent=4, sample_seed=81001, repair_limit=0)
            if archive_validation_rows:
                from latentmol.decoder_checkpoint_storage import archive_panel_rows
                prior_metrics["storage"] = archive_panel_rows(run.path / f"validation-prior-{epoch:03d}")
            model.decoder.train()
            run.log({"epoch": epoch, "training": {k: v / examples for k, v in sums.items()},
                "validation": validation, "validation_prior": prior_metrics,
                "examples": examples, "updates": steps, "rng_sha256": terminal,
                "training_point_counts": counts,
                "gradient_updates_clipped": clipped_updates,
                "max_gradient_norm_before_clip": largest_gradient,
                "elapsed_seconds": time.monotonic() - started,
                "extra_guidance": None if extra_guidance is None else extra_guidance.receipt()})
            write_json(run.path / "latest_checkpoint.json", {"path": str(path), "sha256": file_sha256(path), "epoch": epoch})
            if checkpoint_keep_epochs is not None:
                from latentmol.decoder_checkpoint_storage import prune_owned_checkpoints
                prune_owned_checkpoints(run.path, epoch, checkpoint_keep_epochs, checkpoint_keep_recent)
        final_sha = state_digest(model.decoder.state_dict())
        if final_sha == initial_decoder_sha:
            raise RuntimeError("Decoder did not change")
        run.annotate(endpoint_checkpoint=str(path), endpoint_sha256=file_sha256(path),
            frozen_encoder_sha256=encoder_sha, decoder_changed=True, completed_epochs=stop_epoch,
            frozen_reference_state_sha256=reference_sha,
            requested_endpoint_epochs=config.epochs, endpoint_completed=stop_epoch == config.epochs,
            comparison_status="complete_fixed_endpoint" if stop_epoch == config.epochs else "complete_epoch_segment",
            terminal_rng=terminal)
        return {"run_dir": str(run.path), "checkpoint": str(path), "sha256": file_sha256(path),
                "extra_guidance": None if extra_guidance is None else extra_guidance.receipt(),
                "terminal_rng": terminal, "encoder_sha256": encoder_sha,
                "update_profile": update_profile, "elapsed_seconds": time.monotonic() - started,
                "completed_epochs": stop_epoch, "requested_epochs": config.epochs,
                "endpoint_completed": stop_epoch == config.epochs}


def read_protocol(path, expected_sha):
    if file_sha256(path) != expected_sha:
        raise ValueError("Protocol hash mismatch")
    value = json.loads(path.read_text())
    allowed = {"version", "initializer", "initializer_sha256", "data", "dataset_sha256", "split_sha256",
        "fine_tune", "seeds", "coefficients", "train_limit", "validation_limit", "validation_prior_count",
        "final_prior_count", "final_repair_limit", "case_budget_seconds", "controller_budget_seconds",
        "smoke_budget_seconds"}
    if set(value) != allowed or value["version"] != 1:
        raise ValueError("Protocol fields/version mismatch")
    FineTuneConfig(**value["fine_tune"])
    if value["coefficients"] != [0.0, 0.1, 1.0] or len(set(value["seeds"])) != len(value["seeds"]):
        raise ValueError("Declared study requires distinct seeds and 0/0.1/1 matched arms")
    for key in ("train_limit", "validation_limit", "validation_prior_count", "final_prior_count",
                "final_repair_limit", "case_budget_seconds", "controller_budget_seconds", "smoke_budget_seconds"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError("Invalid positive protocol integer: " + key)
    if any(type(seed) is not int or seed < 0 for seed in value["seeds"]):
        raise ValueError("Invalid seed")
    return value


def assess_advantage(original_rec, control_rec, candidate_rec, versus_original, versus_control):
    """Prespecified conservative evidence gate, not a significance guarantee.

    Require positive clustered raw-validity CIs in argmax and sampling, retain
    unique-valid yield, and allow at most one absolute percentage point loss in
    held-out reconstruction validity/exactness versus either reference. These
    exploratory tests have no multiple-comparison/familywise error guarantee.
    Proxy or successful repair alone cannot establish an advantage.
    """
    tolerance = 0.01
    retention = {key: all(candidate_rec[key] >= ref[key] - tolerance
                         for ref in (original_rec, control_rec))
                 for key in ("validity", "exact_reconstruction")}
    positive = all(comparison["decision"]["validity_positive_ci_both_modes"]
                   and comparison["decision"]["unique_valid_per_attempt_retained_both_modes"]
                   for comparison in (versus_original, versus_control))
    return {"advantage_established_within_prespecified_panel": bool(positive and all(retention.values())),
            "reconstruction_retention": retention, "reconstruction_tolerance_absolute": tolerance,
            "raw_validity_and_unique_yield_positive_vs_both_references": positive,
            "scope": "Exploratory panel evidence; two training seeds reported separately; no corrected significance, binding or synthesis claim"}


def run_study(protocol, output_root, source_root, *, device="cuda"):
    """One allocation: engineering pair, then serial matched scientific cases.

    A failed/censored arm is retained and prevents claiming a complete matched
    comparison. No quality threshold gates the experiment; an engineering
    smoke improvement would bias the protocol and is deliberately not used.
    """
    from latentmol.decoder_quality_evaluation import evaluate_random_panel, paired_compare
    initializer, data_path = Path(protocol["initializer"]), Path(protocol["data"])
    if file_sha256(initializer) != protocol["initializer_sha256"]:
        raise ValueError("Initializer hash mismatch")
    train_data, val_data, test_data = [GraphDataset(data_path, split) for split in ("train", "val", "test")]
    if train_data.manifest["dataset_sha256"] != protocol["dataset_sha256"] or train_data.manifest["split_sha256"] != protocol["split_sha256"]:
        raise ValueError("Protected dataset/split mismatch")
    if protocol["train_limit"] > len(train_data) or protocol["validation_limit"] > len(val_data):
        raise ValueError("Study requested unavailable data; do not silently shrink full comparison")
    reference, checkpoint = load_model(initializer, device, expected_spec=train_data.spec)
    if checkpoint["dataset_sha256"] != protocol["dataset_sha256"] or checkpoint["train_config"]["mode"] != "vae":
        raise ValueError("Initializer is not the protected VAE")
    config = FineTuneConfig(**protocol["fine_tune"])
    validation_panel = make_panel(protocol["validation_prior_count"], reference.config.latent, 81000)
    final_panel = make_panel(protocol["final_prior_count"], reference.config.latent, 91000)
    final_loader = subset_loader(test_data, len(test_data), 17, config.batch_size)
    output_root.mkdir(parents=True, exist_ok=False)
    torch.save({"validation": validation_panel, "final": final_panel}, output_root / "latent-panels.pt")
    write_json(output_root / "protocol.json", protocol)
    started = time.monotonic()
    common = dict(initializer=initializer, train_data=train_data, val_data=val_data, device=device, source_root=source_root)
    smoke = FineTuneConfig(**{**asdict(config), "epochs": 2})
    smoke_results = [fit_case(**common, output_root=output_root / "smoke", config=smoke,
        seed=protocol["seeds"][0], coefficient=coefficient, train_limit=min(512, len(train_data)),
        val_limit=min(64, len(val_data)), validation_panel=validation_panel[:32],
        budget_seconds=protocol["smoke_budget_seconds"]) for coefficient in (0.0, 1.0)]
    if smoke_results[0]["terminal_rng"] != smoke_results[1]["terminal_rng"] or smoke_results[0]["encoder_sha256"] != smoke_results[1]["encoder_sha256"]:
        raise RuntimeError("Smoke RNG budgets or frozen encoder differed between matched arms")
    write_json(output_root / "native_engineering_gate.json", {"status": "passed", "cases": smoke_results,
        "scope": "finite updates, unchanged encoder, changed decoder, identical RNG budgets; no efficacy gate"})
    loss_config = LossConfig(**checkpoint["loss_config"])
    original_rec, _ = reconstruction_metrics(reference, final_loader, loss_config, device,
        chemistry_limit=len(test_data), training_mode="vae", graded=False, training_smiles=set(train_data.smiles))
    original_prior = evaluate_random_panel(reference, final_panel, set(train_data.smiles), output_root / "original-prior",
        batch_size=config.batch_size, samples_per_latent=4, sample_seed=91001, repair_limit=protocol["final_repair_limit"])
    write_json(output_root / "original.json", {"checkpoint_sha256": protocol["initializer_sha256"],
        "reconstruction": original_rec, "prior": original_prior, "selection": "untouched third reference"})
    del reference
    cases = []
    for seed in protocol["seeds"]:
        control = None
        for coefficient in protocol["coefficients"]:
            label = f"seed{seed}-coefficient{coefficient:g}"
            remaining = protocol["controller_budget_seconds"] - (time.monotonic() - started)
            if remaining < protocol["case_budget_seconds"] + 1800:
                cases.append({"label": label, "status": "not_attempted_time_reservation", "remaining_seconds": remaining})
                write_json(output_root / "cases.json", cases)
                continue
            try:
                fit = fit_case(**common, output_root=output_root / "continuations", config=config,
                    seed=seed, coefficient=coefficient, train_limit=protocol["train_limit"],
                    val_limit=protocol["validation_limit"], validation_panel=validation_panel,
                    budget_seconds=protocol["case_budget_seconds"])
                model, _ = load_model(Path(fit["checkpoint"]), device, expected_spec=train_data.spec)
                rec, _ = reconstruction_metrics(model, final_loader, loss_config, device,
                    chemistry_limit=len(test_data), training_mode="vae", graded=False, training_smiles=set(train_data.smiles))
                evaluation_dir = Path(fit["run_dir"]) / "final-prior"
                prior = evaluate_random_panel(model, final_panel, set(train_data.smiles), evaluation_dir,
                    batch_size=config.batch_size, samples_per_latent=4, sample_seed=91001,
                    repair_limit=protocol["final_repair_limit"])
                item = {"label": label, "status": "completed", "seed": seed, "coefficient": coefficient,
                        "fit": fit, "reconstruction": rec, "prior": prior, "evaluation_dir": str(evaluation_dir),
                        "vs_original": paired_compare(output_root / "original-prior", evaluation_dir, bootstrap_seed=92000)}
                if coefficient == 0:
                    control = item
                elif control is not None:
                    if fit["terminal_rng"] != control["fit"]["terminal_rng"]:
                        raise RuntimeError("Matched terminal RNG streams differ")
                    item["vs_control"] = paired_compare(Path(control["evaluation_dir"]), evaluation_dir, bootstrap_seed=92000)
                    item["advantage_assessment"] = assess_advantage(original_rec, control["reconstruction"], rec,
                                                                  item["vs_original"], item["vs_control"])
                write_json(Path(fit["run_dir"]) / "final_evaluation.json", item)
                del model
            except Exception as error:
                item = {"label": label, "status": "failed_or_censored", "error": repr(error),
                        "seed": seed, "coefficient": coefficient,
                        "run_dir": getattr(error, "experiment_run_dir", None)}
                # Retain ordinary case failures; never compare incomplete checkpoints.
            cases.append(item)
            write_json(output_root / "cases.json", cases)
    write_json(output_root / "summary.json", {"cases": cases, "original_reconstruction": original_rec,
        "original_prior": original_prior, "elapsed_seconds": time.monotonic() - started,
        "scope": "decoder continuation; RDKit validity and train-only novelty do not establish synthesis/binding",
        "advantage_assessment": "Inspect matched clustered validity CIs, probability-repair costs, size/composition and reconstruction retention together",
        "complete": all(case["status"] == "completed" for case in cases)})
    if not all(case["status"] == "completed" for case in cases):
        raise RuntimeError("Study contains retained failed/censored/unattempted cases; matched comparison is incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Native experiment requires an allocated DTU HPC job")
    run_study(read_protocol(args.config, args.sha256), args.output_root, Path.cwd())


if __name__ == "__main__":
    main()
