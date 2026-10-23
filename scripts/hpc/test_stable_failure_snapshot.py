"""HPC gate: compare old and stabilized decoders on the exact failed VAE batch.

The failure snapshot contains the model weights *after* the preceding update,
the sampled latent, and hard graph labels. This checks a forward/backward
numerical repair at that state; it does not establish long-run training quality.
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from latentmol.experiments import file_sha256
from latentmol.losses import reconstruction_losses
from latentmol.models import GraphAutoencoder
from latentmol.representation import GraphSpec, HardGraphBatch
from latentmol.train import TrainConfig, seed_everything


def finite_summary(tensor):
    value = tensor.detach()
    finite = torch.isfinite(value)
    good = value[finite]
    return {
        "shape": list(value.shape),
        "nonfinite": int((~finite).sum().item()),
        "finite_abs_max": float(good.abs().max().item()) if good.numel() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--historical-config", type=Path, required=True)
    parser.add_argument("--stable-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    old_config = TrainConfig.from_yaml(args.historical_config)
    new_config = TrainConfig.from_yaml(args.stable_config)
    old_dict, new_dict = asdict(old_config), asdict(new_config)
    old_dict["model"]["decoder_variant"] = "stable_deep_pair_refine"
    if old_dict != new_dict or old_config.mode != "vae":
        raise ValueError("Only the explicit stable decoder variant may differ")
    seed_everything(new_config.seed)
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    if snapshot["epoch"] != 7 or snapshot["batch_one_based"] != 2464:
        raise ValueError("Unexpected failed-batch identity")
    if snapshot["model_config"] != asdict(old_config.model):
        raise ValueError("Snapshot model configuration differs from the historical arm")
    spec = GraphSpec.from_dict(snapshot["graph_spec"])
    historical = GraphAutoencoder(spec, old_config.model).to(new_config.device)
    stable = GraphAutoencoder(spec, new_config.model).to(new_config.device)
    historical.load_state_dict(snapshot["model_state"], strict=True)
    stable.load_state_dict(snapshot["model_state"], strict=True)
    target = HardGraphBatch(snapshot["target_atoms"], snapshot["target_charges"],
                            snapshot["target_bonds"]).to(new_config.device)
    z = snapshot["z"].to(new_config.device)

    with torch.no_grad():
        old_logits = historical.decode(z)
        old_losses = reconstruction_losses(old_logits, target, old_config.loss)
        old_summary = {
            "atoms": finite_summary(old_logits.atoms),
            "charges": finite_summary(old_logits.charges),
            "bonds": finite_summary(old_logits.bonds),
            "losses": {name: finite_summary(value) for name, value in old_losses.items()},
        }
    if torch.isfinite(old_losses["reconstruction"]):
        raise AssertionError("The captured historical state did not reproduce the NaN")
    del historical, old_logits, old_losses

    z = z.detach().requires_grad_(True)
    new_logits = stable.decode(z)
    new_losses = reconstruction_losses(new_logits, target, new_config.loss)
    if not all(torch.isfinite(value).all() for value in
               (*new_losses.values(), new_logits.atoms, new_logits.charges,
                new_logits.bonds)):
        raise AssertionError("The stabilized decoder still has a non-finite forward/loss")
    new_losses["reconstruction"].backward()
    if z.grad is None or not torch.isfinite(z.grad).all() or z.grad.abs().sum() == 0:
        raise AssertionError("The stabilized decoder lacks a finite nonzero z gradient")
    bad_parameter_gradients = [name for name, parameter in stable.named_parameters()
                               if parameter.grad is not None and
                               not torch.isfinite(parameter.grad).all()]
    if bad_parameter_gradients:
        raise AssertionError(f"Non-finite parameter gradients: {bad_parameter_gradients[:5]}")

    report = {
        "snapshot": str(args.snapshot),
        "snapshot_sha256": file_sha256(args.snapshot),
        "historical_source_sha256": snapshot["source_archive_sha256"],
        "new_source_sha256": os.environ.get("LATENTMOL_SOURCE_SHA256"),
        "historical_config_sha256": file_sha256(args.historical_config),
        "stable_config_sha256": file_sha256(args.stable_config),
        "historical_model": old_summary,
        "stable_model": {
            "atoms": finite_summary(new_logits.atoms),
            "charges": finite_summary(new_logits.charges),
            "bonds": finite_summary(new_logits.bonds),
            "losses": {name: finite_summary(value) for name, value in new_losses.items()},
            "z_gradient": finite_summary(z.grad),
            "nonfinite_parameter_gradients": bad_parameter_gradients,
        },
        "strict_weight_transfer": True,
        "gate": "old_nan_stable_finite_forward_and_backward",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("STABLE_FAILURE_GATE=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
