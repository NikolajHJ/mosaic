"""Train the shared graph architecture as an AE or VAE on a prepared dataset.

The configuration is saved in every run and checkpoint. No property predictor is
trained here. AE/VAE pilots use identical data, architecture, seeds, optimizer,
and update budget; only posterior sampling and annealed Gaussian KL differ.
"""

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from latentmol.datasets.data import GraphDataset, collate_graphs
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.evaluate import reconstruction_metrics
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.losses import LossConfig, latent_kl, reconstruction_losses
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.sampling import subset_indices


@dataclass(frozen=True)
class TrainConfig:
    mode: str = "ae"
    seed: int = 17
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    clip_grad: float = 5.0
    beta: float = 0.001
    kl_warmup_epochs: int = 20
    train_limit: int | None = None
    validation_limit: int = 256
    chemistry_limit: int = 128
    num_workers: int = 0
    device: str = "cuda"
    checkpoint_every: int = 10
    checkpoint_epochs: tuple[int, ...] = (60,)
    train_probe_limit: int | None = None
    full_chemistry_every: int = 0
    lr_milestones: tuple[tuple[int, float], ...] = ()
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        if self.mode not in ("ae", "vae"):
            raise ValueError("mode must be ae or vae")
        if self.epochs < 1 or self.batch_size < 1 or self.beta < 0:
            raise ValueError("Invalid epoch, batch size, or KL coefficient")
        if self.learning_rate <= 0 or self.clip_grad <= 0 or self.validation_limit < 1:
            raise ValueError("Learning rate, gradient clipping, and validation limit must be positive")
        if self.train_limit is not None and self.train_limit < 1:
            raise ValueError("Training subset limit must be positive")
        if self.chemistry_limit < 1 or self.num_workers < 0:
            raise ValueError("Invalid chemistry sample limit or worker count")
        if self.checkpoint_every < 1 or any(epoch < 1 for epoch in self.checkpoint_epochs):
            raise ValueError("Checkpoint retention intervals/epochs must be positive")
        if self.train_probe_limit is not None and self.train_probe_limit < 1:
            raise ValueError("Training probe limit must be positive")
        if self.full_chemistry_every < 0:
            raise ValueError("Full chemistry interval cannot be negative")
        epochs = [epoch for epoch, _ in self.lr_milestones]
        if epochs != sorted(set(epochs)) or any(not isinstance(epoch, int) or epoch < 1 or not math.isfinite(rate) or rate <= 0
                                               for epoch, rate in self.lr_milestones):
            raise ValueError("LR milestones must have unique increasing positive epochs and positive rates")

    @classmethod
    def from_yaml(cls, path: Path):
        value = yaml.safe_load(path.read_text())
        value["model"] = ModelConfig(**value.get("model", {}))
        value["loss"] = LossConfig(**value.get("loss", {}))
        if "checkpoint_epochs" in value:
            value["checkpoint_epochs"] = tuple(value["checkpoint_epochs"])
        if "lr_milestones" in value:
            value["lr_milestones"] = tuple(tuple(pair) for pair in value["lr_milestones"])
        return cls(**value)


def seed_everything(seed: int):
    """Seed all RNGs; record deterministic settings without claiming cross-device identity."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def learning_rate_at_epoch(config: TrainConfig, epoch: int) -> float:
    """Apply milestones at the start of their absolute epoch, including after resume."""
    rate = config.learning_rate
    for milestone, value in config.lr_milestones:
        if epoch >= milestone:
            rate = value
    return rate


def algorithm_identity() -> dict:
    """Hash executable package code, independently of YAML, tests and documentation.

    Full source/archive identity is still recorded by ExperimentRun. Exact resume
    requires this algorithm identity unchanged, so a new runtime epoch horizon or
    documentation edit does not masquerade as a changed training implementation.
    """
    root = Path(__file__).resolve().parent
    files = {path.relative_to(root).as_posix(): file_sha256(path)
             for path in sorted(root.rglob("*.py"))}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def runtime_identity(device: str) -> dict:
    """Require matching numerical software/device settings for an exact continuation."""
    from rdkit import rdBase
    cuda = torch.device(device).type == "cuda"
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "rdkit": rdBase.rdkitVersion,
            "cuda_runtime": torch.version.cuda if cuda else None,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "device_type": torch.device(device).type,
            "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
            "cuda_device_count": torch.cuda.device_count() if cuda else 0,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if cuda else None,
            "cpu_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


def capture_rng_state(generators: dict[str, torch.Generator], include_cuda: bool) -> dict:
    """Encode every RNG as tensors/plain data compatible with weights_only loading.

    Workers are nonpersistent and receive new reproducible base seeds each epoch.
    At an exhausted epoch boundary their internal states are unnecessary; each
    loader's generator must still be restored before its next iterator is made.
    """
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": numpy_state[0],
                      "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                      "position": int(numpy_state[2]), "has_gaussian": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if include_cuda else [],
            "loaders": {name: generator.get_state() for name, generator in generators.items()}}


def restore_rng_state(state: dict, generators: dict[str, torch.Generator]) -> None:
    """Restore after model/optimizer/loader setup and before the next epoch iterator."""
    if set(state["loaders"]) != set(generators):
        raise ValueError("Checkpoint DataLoader generators differ")
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["algorithm"], value["keys"].cpu().numpy().astype(np.uint32),
                         value["position"], value["has_gaussian"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count differs")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for name, generator in generators.items():
        generator.set_state(state["loaders"][name].cpu())


def state_digest(state: dict) -> str:
    """Identify selected model weights independently of mutable checkpoint aliases."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_continuation(checkpoint, config, data, algorithm, runtime, indices):
    """Reject incompatible exact resumes; legacy format-1 evaluation remains valid.

    The only training-config change permitted here is extending the target epoch
    horizon. Optimizer, architecture, losses, subset/probe definitions, schedules
    and RNG-relevant settings must match. There is currently no LR scheduler.
    """
    state = checkpoint.get("continuation")
    if checkpoint.get("format_version") != 1 or not state or state.get("version") != 1:
        raise ValueError("Checkpoint lacks exact epoch-boundary continuation state; it remains evaluable")
    if not state.get("epoch_complete") or state.get("scheduler_state") is not None:
        raise ValueError("Require completed epoch and supported scheduler state (none)")
    if checkpoint["dataset_sha256"] != data.manifest["dataset_sha256"] or state["split_sha256"] != data.manifest["split_sha256"]:
        raise ValueError("Resume dataset or split hashes differ")
    if checkpoint["graph_spec"] != data.spec.to_dict():
        raise ValueError("Resume graph schema differs")
    if state["algorithm"]["sha256"] != algorithm["sha256"]:
        raise ValueError("Exact resume requires unchanged executable algorithm source")
    if state["runtime"] != runtime:
        raise ValueError("Exact resume requires matching numerical runtime/device settings")
    old, new = dict(checkpoint["train_config"]), asdict(config)
    old_epochs, new_epochs = old.pop("epochs"), new.pop("epochs")
    # JSON normalizes tuple/list representation without weakening value checks.
    if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True) or new_epochs < old_epochs:
        raise ValueError("Resume training config differs beyond an allowed epoch-horizon extension")
    if state["indices"] != indices:
        raise ValueError("Resume train/validation/probe indices differ")
    return state


def copy_checkpoint(source: Path, destination: Path) -> None:
    """Replace a convenience alias atomically; immutable epoch snapshots stay intact."""
    temporary = destination.with_suffix(".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def inherit_selections(checkpoint, resume_from: Path, run) -> dict:
    """Carry actual ancestral winners forward even if resumed epochs never improve.

    Every selected epoch has an immutable snapshot and a model-weight digest.
    Mutable best.pt aliases are never used to recover an older selected epoch.
    """
    result = {}
    for name, entry in checkpoint["continuation"]["selections"].items():
        entry = dict(entry)
        source = resume_from if entry["epoch"] == checkpoint["epoch"] else Path(entry["path"])
        selected = torch.load(source, map_location="cpu", weights_only=True)
        if selected["epoch"] != entry["epoch"] or state_digest(selected["model_state"]) != entry["model_sha256"]:
            raise ValueError(f"Inherited best {name} checkpoint identity differs")
        retained = run.path / f"inherited_best_{name}.pt"
        copy_checkpoint(source, retained)
        entry.update(path=str(retained), inherited_from=str(source), checkpoint_sha256=file_sha256(retained))
        copy_checkpoint(retained, run.path / ("best.pt" if name == "reconstruction" else "best_validity.pt"))
        result[name] = entry
    return result


def save_checkpoint(path, model, optimizer, config, data, epoch, run, continuation=None):
    payload = {
        "format_version": 1, "graph_spec": model.spec.to_dict(), "model_config": asdict(config.model),
        "loss_config": asdict(config.loss), "train_config": asdict(config), "epoch": epoch,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "dataset_sha256": data.manifest["dataset_sha256"], "run_id": run.run_id,
        "beta_at_checkpoint": config.beta * min(epoch / max(config.kl_warmup_epochs, 1), 1.0) if config.mode == "vae" else 0.0,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(config.device).type == "cuda" else [],
        "continuation": continuation,
        "source_sha256": run.record["source"]["sha256"],
        "source_archive_sha256": run.record["source_archive_sha256"],
        "parameter_counts": {
            "encoder": sum(p.numel() for p in model.encoder.parameters()),
            "decoder": sum(p.numel() for p in model.decoder.parameters()),
            "total": sum(p.numel() for p in model.parameters()),
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def training_cohorts(config, training_count, validation_count):
    """Reproducible ordered cohorts used by training and failed-segment recovery."""
    train_indices = subset_indices(training_count, config.train_limit, config.seed)
    val_indices = subset_indices(validation_count, config.validation_limit, config.seed, shuffle_full=True)
    probe_count = min(config.train_probe_limit or len(val_indices), len(train_indices), len(val_indices))
    positions = subset_indices(len(train_indices), probe_count, config.seed + 1, shuffle_full=True)
    return {"train": train_indices, "val": val_indices,
            "train_probe": [train_indices[index] for index in positions], "val_probe": val_indices[:probe_count]}


def compact_cohort_identity(cohorts):
    """Digest ordered row indices without embedding millions in every checkpoint."""
    result = {}
    for name, values in cohorts.items():
        digest = hashlib.sha256()
        for begin in range(0, len(values), 8192):
            digest.update(np.asarray(values[begin:begin + 8192], dtype="<i8").tobytes())
        result[name] = {"count": len(values), "ordered_int64_le_sha256": digest.hexdigest()}
    return result


def train(config: TrainConfig, data_path: Path, run_root: Path,
          resume_from: Path | None = None, stop_after_epoch: int | None = None,
          observer=None, wall_budget_seconds: float | None = None,
          initial_epoch_seconds: float | None = None, shared_cache=None):
    """Train or exactly continue at a completed epoch; each phase gets a new run.

    ``epochs`` is the planned horizon. ``stop_after_epoch`` is an operational
    boundary, not early stopping or a changed objective. Full validation is
    separate from matched fixed train/validation probes. Only validation selects
    checkpoints; training probe results never influence selection or the LR.
    """
    stop_epoch = config.epochs if stop_after_epoch is None else stop_after_epoch
    if not 1 <= stop_epoch <= config.epochs:
        raise ValueError("Stop boundary must be within the planned epoch horizon")
    with ExperimentRun(run_root, f"{config.mode}-train", asdict(config), Path.cwd()) as run:
        phase_started = time.monotonic()
        seed_everything(config.seed)
        cuda = torch.device(config.device).type == "cuda"
        if cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use a GPU batch job")
        train_data = GraphDataset(data_path, "train", **({"validated_cache": shared_cache} if shared_cache else {}))
        valid_data = GraphDataset(data_path, "val", **({"validated_cache": shared_cache} if shared_cache else {}))
        cohorts = training_cohorts(config, len(train_data), len(valid_data))
        train_indices, val_indices = cohorts["train"], cohorts["val"]
        if not train_indices or not val_indices:
            raise ValueError("Training and validation splits must be nonempty")
        forward_options = {}
        if config.model.encoder_pooling == "canonical":
            checked = {"train": verify_canonical_training_cache(train_data, train_indices),
                       "val": verify_canonical_training_cache(valid_data, val_indices)}
            run.annotate(canonical_cache_check=checked)
            forward_options["from_canonical_cache"] = True
        train_probe_indices, val_probe_indices = cohorts["train_probe"], cohorts["val_probe"]
        probe_count = len(train_probe_indices)
        indices = cohorts
        if observer is not None and getattr(observer, "compact_indices", False):
            # A 1.9-million-element list must not be copied into every run JSON
            # and epoch checkpoint. Persist its deterministic digest instead.
            indices = compact_cohort_identity(cohorts)
        subsets = {"train": Subset(train_data, train_indices), "val": Subset(valid_data, val_indices),
                   "train_probe": Subset(train_data, train_probe_indices),
                   "val_probe": Subset(valid_data, val_probe_indices)}
        generators = {name: torch.Generator().manual_seed(config.seed + offset)
                      for name, offset in (("train", 0), ("val", 101), ("train_probe", 102), ("val_probe", 103))}
        loaders = {name: DataLoader(subset, batch_size=config.batch_size, shuffle=name == "train",
                                   generator=generators[name], num_workers=config.num_workers,
                                   persistent_workers=False, collate_fn=collate_graphs)
                   for name, subset in subsets.items()}
        model = GraphAutoencoder(train_data.spec, config.model).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        algorithm, runtime = algorithm_identity(), runtime_identity(config.device)
        selections, updates, first_epoch = {}, 0, 1
        cumulative_train_seconds = cumulative_evaluation_seconds = 0.0
        run.annotate(dataset_manifest=train_data.manifest, graph_spec=train_data.spec.to_dict(),
                     parameter_count=sum(p.numel() for p in model.parameters()),
                     decoder_parameter_count=sum(p.numel() for p in model.decoder.parameters()),
                     encoder_parameter_count=sum(p.numel() for p in model.encoder.parameters()),
                     training_molecules=len(subsets["train"]),
                     seeded_initial_model_sha256=state_digest(model.state_dict()),
                     validation_molecules=len(subsets["val"]), matched_probe_molecules=probe_count,
                     checkpoint_selection={"best.pt": "minimum full validation reconstruction loss",
                                           "best_validity.pt": "maximum fixed validation-probe validity; ties exactness then full validation loss"},
                     train_indices=indices["train"], validation_indices=indices["val"],
                     train_probe_indices=indices["train_probe"], validation_probe_indices=indices["val_probe"],
                     chemistry_selection="fixed seeded random cohort prefixes; full validation chemistry optionally expanded at fixed epochs",
                     reconstruction_probe_latent="deterministic_mu",
                     algorithm_identity=algorithm, runtime_identity=runtime,
                     stop_after_epoch=stop_epoch, planned_epochs=config.epochs,
                     gpu=torch.cuda.get_device_name(config.device) if cuda else None)
        if resume_from is not None:
            resume_from = Path(resume_from).resolve()
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
            state = validate_continuation(checkpoint, config, train_data, algorithm, runtime, indices)
            first_epoch = checkpoint["epoch"] + 1
            if first_epoch > stop_epoch:
                raise ValueError("Resume checkpoint is already at or beyond the requested stop boundary")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            selections = inherit_selections(checkpoint, resume_from, run)
            updates = state["updates"]
            cumulative_train_seconds = state["cumulative_train_seconds"]
            cumulative_evaluation_seconds = state["cumulative_evaluation_seconds"]
            run.annotate(resume={"checkpoint": str(resume_from), "sha256": file_sha256(resume_from),
                                 "parent_run_id": checkpoint["run_id"], "completed_epoch": checkpoint["epoch"],
                                 "inherited_selections": json.loads(json.dumps(selections))})
            restore_rng_state(state["rng"], generators)
            del checkpoint, state
        if observer is not None:
            observer.attach(model, run, train_data, valid_data)
        epoch_seconds = initial_epoch_seconds
        completed_epoch = first_epoch - 1
        stop_reason = "requested_epoch_boundary"
        for epoch in range(first_epoch, stop_epoch + 1):
            # Only completed epochs are resumable. Start another epoch only if
            # a conservative estimate (including evaluation) fits the allocation.
            if wall_budget_seconds is not None:
                if not epoch_seconds or epoch_seconds <= 0:
                    raise ValueError("Bounded training needs a positive benchmark epoch estimate")
                if time.monotonic() - phase_started + 1.5 * epoch_seconds > wall_budget_seconds:
                    if completed_epoch < first_epoch:
                        raise RuntimeError("Allocation budget cannot safely fit one complete epoch")
                    stop_reason = "wall_budget_epoch_boundary"
                    break
            started = time.monotonic()
            if observer is not None:
                observer.before_epoch(epoch)
            model.train()
            rate = learning_rate_at_epoch(config, epoch)
            for group in optimizer.param_groups:
                group["lr"] = rate
            beta = config.beta * min(epoch / max(config.kl_warmup_epochs, 1), 1.0) if config.mode == "vae" else 0.0
            totals, count, batches, clipped = {}, 0, 0, 0
            norm_sum = norm_max = 0.0
            for batch in loaders["train"]:
                if observer is not None:
                    observer.before_batch(batches)
                batch = batch.to(config.device)
                optimizer.zero_grad(set_to_none=True)
                graph = batch if forward_options else batch.to_relaxed(model.spec)
                output = model(graph, sample_latent=config.mode == "vae", **forward_options)
                losses = reconstruction_losses(output.decoded, batch, config.loss)
                losses["kl"] = latent_kl(output.mu, output.logvar)
                total = losses["reconstruction"] + beta * losses["kl"]
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite training loss")
                total.backward()
                if observer is not None:
                    observer.after_backward(model, output, batches)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad, error_if_nonfinite=True)
                optimizer.step()
                norm_value = float(norm)
                norm_sum += norm_value
                norm_max = max(norm_max, norm_value)
                clipped += int(norm_value > config.clip_grad)
                size = batch.atoms.shape[0]
                for name, value in {**losses, "total": total, "gradient_norm": norm}.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
                count += size
                batches += 1
                updates += 1
            training_seconds = time.monotonic() - started
            cumulative_train_seconds += training_seconds
            model.eval()
            evaluation_started = time.monotonic()
            full_chemistry = bool(config.full_chemistry_every and epoch % config.full_chemistry_every == 0)
            validation_metrics, _ = reconstruction_metrics(
                model, loaders["val"], config.loss, config.device,
                len(subsets["val"]) if full_chemistry else config.chemistry_limit,
                training_mode=config.mode, **forward_options)
            probe_metrics = {}
            for name in ("train_probe", "val_probe"):
                probe_metrics[name], _ = reconstruction_metrics(model, loaders[name], config.loss, config.device,
                                                               config.chemistry_limit, training_mode=config.mode,
                                                               **forward_options)
            evaluation_seconds = time.monotonic() - evaluation_started
            cumulative_evaluation_seconds += evaluation_seconds
            reconstruction = validation_metrics["reconstruction"]
            if not math.isfinite(reconstruction):
                raise FloatingPointError("Non-finite validation reconstruction loss")
            val_probe = probe_metrics["val_probe"]
            rank = [val_probe["validity"], val_probe["exact_reconstruction"], -reconstruction]
            improved = []
            if "reconstruction" not in selections or reconstruction < selections["reconstruction"]["score"]:
                improved.append("reconstruction")
            if "validity" not in selections or rank > selections["validity"]["rank"]:
                improved.append("validity")
            snapshot = run.path / f"epoch_{epoch:04d}.pt"
            if improved:
                weight_hash = state_digest(model.state_dict())
                for name in improved:
                    selections[name] = {"epoch": epoch, "path": str(snapshot), "model_sha256": weight_hash,
                                        "score": reconstruction if name == "reconstruction" else val_probe["validity"],
                                        "rank": rank if name == "validity" else [-reconstruction],
                                        "validation_chemistry_samples": val_probe["chemistry_samples"]}
            continuation = {"version": 1, "epoch_complete": True, "updates": updates,
                            "algorithm": algorithm, "runtime": runtime, "indices": indices,
                            "split_sha256": train_data.manifest["split_sha256"],
                            "rng": capture_rng_state(generators, cuda), "scheduler_state": None,
                            "lr_policy": "absolute-epoch milestones", "lr_at_checkpoint": rate,
                            "selections": selections, "cumulative_train_seconds": cumulative_train_seconds,
                            "cumulative_evaluation_seconds": cumulative_evaluation_seconds}
            checkpoint_started = time.monotonic()
            save_checkpoint(run.path / "last.pt", model, optimizer, config, train_data, epoch, run, continuation)
            if improved or epoch % config.checkpoint_every == 0 or epoch in config.checkpoint_epochs or epoch == stop_epoch:
                if snapshot.exists():
                    raise FileExistsError(f"Immutable epoch snapshot already exists: {snapshot}")
                copy_checkpoint(run.path / "last.pt", snapshot)
            for name in improved:
                copy_checkpoint(snapshot, run.path / ("best.pt" if name == "reconstruction" else "best_validity.pt"))
            checkpoint_seconds = time.monotonic() - checkpoint_started
            row = {"epoch": epoch, "updates": updates, "beta": beta, "learning_rate": rate,
                   "training_latent_policy": "posterior_sample" if config.mode == "vae" else "encoder_mu",
                   "training_kl_used_in_objective": config.mode == "vae",
                   "seconds": time.monotonic() - started, "training_seconds": training_seconds,
                   "evaluation_seconds": evaluation_seconds, "checkpoint_seconds": checkpoint_seconds,
                   "training_molecules_per_second": count / max(training_seconds, 1e-12),
                   "training_updates_per_second": batches / max(training_seconds, 1e-12),
                   "cumulative_training_seconds": cumulative_train_seconds,
                   "cumulative_evaluation_seconds": cumulative_evaluation_seconds,
                   "gradient_norm_mean_per_update": norm_sum / batches, "gradient_norm_max": norm_max,
                   "clipped_updates": clipped, "epoch_updates": batches, "gradient_clipping_rate": clipped / batches,
                   "full_validation_chemistry": full_chemistry,
                   "train": {k: v / count for k, v in totals.items()}, "val": validation_metrics,
                   **probe_metrics}
            if observer is not None:
                # The observer uses dedicated generators and deterministic mu;
                # it must not advance any RNG captured in the checkpoint above.
                observer_rng = capture_rng_state(generators, cuda)
                try:
                    row["extended_diagnostics"] = observer.after_epoch(model, epoch, run)
                finally:
                    # Future observer extensions must also leave every training
                    # RNG unchanged, including DataLoader base-seed generation.
                    restore_rng_state(observer_rng, generators)
            epoch_seconds = max(epoch_seconds or 0.0, time.monotonic() - started)
            row["seconds_including_extended_diagnostics"] = time.monotonic() - started
            run.log(row)
            write_json(run.path / "latest_epoch.json", row)
            run.annotate(completed_epoch=epoch, selections=selections, updates=updates)
            completed_epoch = epoch
        if completed_epoch == config.epochs:
            stop_reason = "planned_horizon"
        run.annotate(best_validation_reconstruction=selections["reconstruction"]["score"],
                     best_validation_probe_validity=selections["validity"]["score"], updates=updates,
                     completed_epoch=completed_epoch, planned_horizon_complete=completed_epoch == config.epochs,
                     stop_reason=stop_reason,
                     phase_wall_seconds=time.monotonic() - phase_started)
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    train(TrainConfig.from_yaml(args.config), args.data, args.run_root,
          resume_from=args.resume_from, stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
