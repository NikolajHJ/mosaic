"""Authenticated epoch-boundary continuation of decoder-only training.

Source070 saved AdamW and four explicit training generators, not global RNG.
Supported decoders have deterministic forward passes. This restores dedicated
streams, without promising universal cross-device bitwise reproduction.
The untouched original initializer always remains the frozen anchor reference.
"""

from dataclasses import asdict
import json
from pathlib import Path

import torch

from latentmol.experiments import file_sha256
from latentmol.models.autoencoder import comparable_model_config
from latentmol.train import state_digest


def restore_training(model, optimizer, generators, original, config, identity, resume, *, epoch_extension=None):
    """Construct models/optimizer/loader first; restore before next iterator.

    A completed metric receipt authenticates data budget and RNG; a checkpoint
    written before validation finishes is insufficient. Search hooks would
    require separate teacher state and are deliberately unsupported here.
    """
    from latentmol.decoder_finetune import rng_receipt, tensor_sha
    required = {"checkpoint", "checkpoint_sha256", "parent_record", "parent_record_sha256",
                "parent_metrics", "parent_metrics_sha256", "parent_archive_sha256"}
    if set(resume) != required:
        raise ValueError("Unknown/missing resume fields")
    def authenticate(path, expected):
        path = Path(path)
        if file_sha256(path) != expected:
            raise ValueError("Resume artifact identity changed: " + str(path))
        return path
    path = authenticate(resume["checkpoint"], resume["checkpoint_sha256"])
    parent = json.loads(authenticate(resume["parent_record"], resume["parent_record_sha256"]).read_text())
    metrics_path = authenticate(resume["parent_metrics"], resume["parent_metrics_sha256"])
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    expected = {key: identity[key] for key in (
        "initializer_sha256", "seed", "coefficient", "train_count", "validation_count",
        "validation_panel_sha256", "dataset_sha256", "frozen_encoder_sha256")}
    if parent.get("source_archive_sha256") != resume["parent_archive_sha256"]:
        raise ValueError("Parent belongs to another source archive")
    if any(parent["config"].get(key) != value for key, value in expected.items()):
        raise ValueError("Parent continuation schedule/initializer identity differs")
    parent_config = parent["config"].get("config")
    target_config = asdict(config)
    # Explicit user-authorized longer training changes only the epoch horizon.
    # Keep strict same-config behavior for every existing caller by default.
    expected_parent_config = dict(target_config)
    if epoch_extension is not None:
        if (not isinstance(epoch_extension, dict) or set(epoch_extension) != {"from_epochs", "to_epochs"}
                or any(type(value) is not int for value in epoch_extension.values())
                or not 0 < epoch_extension["from_epochs"] < epoch_extension["to_epochs"]
                or epoch_extension["to_epochs"] != config.epochs):
            raise ValueError("Invalid declared epoch horizon extension")
        expected_parent_config["epochs"] = epoch_extension["from_epochs"]
    if parent_config != expected_parent_config or parent["config"].get("extra_guidance") is not None:
        raise ValueError("Parent loss/config/search-hook differs")
    if parent["config"].get("split_sha256") not in (None, identity["split_sha256"]):
        raise ValueError("Parent split changed")
    reference_sha = parent["config"].get("frozen_reference_state_sha256")
    if reference_sha is not None and reference_sha != identity["frozen_reference_state_sha256"]:
        raise ValueError("Parent frozen reference changed")
    if epoch_extension is not None and (parent["config"].get("split_sha256") != identity["split_sha256"]
            or reference_sha != identity["frozen_reference_state_sha256"]):
        raise ValueError("Epoch extension requires explicit split and frozen reference lineage")
    # Source070 lacked split SHA inside its case record. The recovery controller
    # separately verifies its authenticated base protocol and protected split.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    fine = checkpoint.get("decoder_finetune", {})
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or not 0 < epoch < config.epochs:
        raise ValueError("Need completed epoch before the final endpoint")
    if epoch_extension is not None and epoch > epoch_extension["from_epochs"]:
        raise ValueError("Checkpoint exceeds the declared parent epoch horizon")
    if (checkpoint.get("graph_spec") != model.spec.to_dict()
            or comparable_model_config(checkpoint.get("model_config")) != asdict(model.config)):
        raise ValueError("Resume graph/model schema differs")
    if checkpoint.get("dataset_sha256") != identity["dataset_sha256"] or checkpoint.get("loss_config") != original["loss_config"]:
        raise ValueError("Resume data/reconstruction loss differs")
    for key, value in {"epoch": epoch, "original_epoch": original["epoch"], "seed": identity["seed"],
                       "coefficient": identity["coefficient"], "config": expected_parent_config,
                       "encoder_sha256": identity["frozen_encoder_sha256"]}.items():
        if fine.get(key) != value:
            raise ValueError("Resume decoder identity differs: " + key)
    if [row.get("epoch") for row in metrics] != list(range(1, epoch + 1)):
        raise ValueError("Need all completed epoch receipts ending at checkpoint")
    updates = (identity["train_count"] + config.batch_size - 1) // config.batch_size
    if any(row.get("examples") != identity["train_count"] or row.get("updates") != updates for row in metrics):
        raise ValueError("Resume completed-epoch data budget differs")
    states = fine.get("rng_states", {})
    if set(states) != set(generators):
        raise ValueError("Resume training generator set differs")
    for state in states.values():
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("Invalid training generator state")
    if {name: tensor_sha(value.cpu()) for name, value in states.items()} != metrics[-1]["rng_sha256"]:
        raise ValueError("Checkpoint RNG differs from completed epoch")
    for module in model.decoder.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p:
            raise ValueError("Legacy resume lacks global RNG for decoder dropout")
        if isinstance(getattr(module, "dropout", None), (int, float)) and module.dropout:
            raise ValueError("Legacy resume lacks global RNG for stochastic decoder")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if state_digest(model.encoder.state_dict()) != identity["frozen_encoder_sha256"]:
        raise ValueError("Resume checkpoint changed frozen encoder")
    optimizer.load_state_dict(fine["optimizer_state"])
    for name, generator in generators.items():
        generator.set_state(states[name].cpu())
    return {"epoch": epoch, "cumulative_updates": epoch * updates,
            "training_rng": rng_receipt(generators), "parent_run_id": parent["run_id"],
            "epoch_extension": epoch_extension,
            "scope": "Epoch boundary; AdamW and dedicated streams restored; no global RNG/cross-device bitwise guarantee",
            "discarded_partial_epoch": None if parent.get("status") == "completed" and parent.get("completed_epochs") == epoch else epoch + 1}
