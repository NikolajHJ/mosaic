"""Check whether a linear C13 seed has a usable AE/VAE latent initialization.

This CPU-only preflight does not invoke Boltz or alter any model weights. A real
carbon-chain Boltz pilot also needs a new native ligand pose and verified atom
map; the saved tyrosine pose cannot be reused as evidence for that molecule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.latent_priors import LatentReference
from latentmol.losses import LossConfig, reconstruction_losses
from latentmol.representation import HardGraphBatch
from latentmol.sampling import hard_graph_log_prob


CHAIN = "CCCCCCCCCCCCC"


def run_preflight(checkpoint_path: Path, reference_path: Path, data_manifest_path: Path,
                  run_root: Path,
                  *, mode: str, expected_checkpoint_sha256: str,
                  expected_reference_sha256: str) -> Path:
    if mode not in ("ae", "vae"):
        raise ValueError("Mode must be ae or vae")
    checkpoint_sha = file_sha256(checkpoint_path)
    reference_sha = file_sha256(reference_path)
    if checkpoint_sha != expected_checkpoint_sha256 or reference_sha != expected_reference_sha256:
        raise ValueError("Checkpoint/reference hashes differ from the predeclared inputs")
    model, training = load_model(checkpoint_path, "cpu")
    if training["train_config"]["mode"] != mode:
        raise ValueError("Checkpoint mode differs from requested mode")
    data_manifest = json.loads(data_manifest_path.read_text())
    if data_manifest["dataset_sha256"] != training["dataset_sha256"]:
        raise ValueError("Checkpoint dataset differs from prepared data")
    spec = model.spec
    reference_value = torch.load(reference_path, map_location="cpu", weights_only=True)
    metadata = reference_value["metadata"]
    expected = {"checkpoint_sha256": checkpoint_sha,
                "dataset_sha256": training["dataset_sha256"],
                "split_sha256": data_manifest["split_sha256"],
                "graph_spec": spec.fingerprint(), "mode": mode}
    if any(metadata.get(key) != value for key, value in expected.items()) or metadata.get("training_count") != 215000:
        raise ValueError("Training-code reference and checkpoint provenance differ")
    reference = LatentReference.from_artifact(reference_value, metadata)
    encoded = encode_smiles(CHAIN, spec)
    seed = HardGraphBatch(*(torch.from_numpy(encoded[key].astype("int64"))[None]
                            for key in ("atom_types", "charges", "bond_types")))
    config = {"mode": mode, "input_smiles": CHAIN, "canonical_input_smiles": encoded["smiles"],
              "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
              "reference": str(reference_path), "reference_sha256": reference_sha,
              "data_manifest": str(data_manifest_path),
              "data_manifest_sha256": file_sha256(data_manifest_path),
              "dataset_sha256": training["dataset_sha256"],
              "split_sha256": data_manifest["split_sha256"],
              "graph_spec": spec.fingerprint(),
              "policy": "encode C13 graph and decode posterior mean; no latent inversion, Boltz, or pose"}
    with ExperimentRun(run_root, f"{mode}-carbon-chain-seed-preflight", config, Path.cwd()) as run:
        model = model.eval().requires_grad_(False)
        with torch.no_grad():
            mean, logvar = model.encode(seed.to_relaxed(spec))
            decoded = model.decode(mean)
            hard = decoded.harden(spec)
            inspected = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
            normalized = reference.normalize(mean)
            losses = reconstruction_losses(decoded, seed,
                                           LossConfig(**training["loss_config"]))
            log_probability = hard_graph_log_prob(decoded, seed, spec)
            if not torch.isfinite(mean).all() or not torch.isfinite(normalized).all():
                raise ValueError("Encoded chain latent is nonfinite")
            summary = {"mode": mode, "input_smiles": encoded["smiles"],
                       "hard_decode_smiles": inspected.smiles,
                       "hard_decode_failure": inspected.failure_reason,
                       "hard_decode_exact": inspected.smiles == encoded["smiles"],
                       "hard_decode_valid": bool(inspected.smiles),
                       "hard_decode_heavy_atoms": int((hard.atoms[0] != 0).sum()),
                       "expected_heavy_atoms": float(decoded.probabilities().expected_size[0]),
                       "latent_dimension": int(mean.shape[-1]),
                       "normalized_latent_norm": float(normalized.norm()),
                       "latent_reference_residual": float(reference.residual(mean)[0]),
                       "latent_reference_residual_limit": float(reference.residual_limit),
                       "mean_posterior_std": float((0.5 * logvar).exp().mean()) if mode == "vae" else None,
                       "chain_log_probability_under_decoder": float(log_probability[0]),
                       "reconstruction": {key: float(value) for key, value in losses.items()},
                       "claim_limit": "A single C13 seed encode/decode screen, not evidence of Boltz binding or a usable carbon-chain native pose."}
            torch.save({"chain_graph": {"atoms": seed.atoms, "charges": seed.charges,
                                         "bonds": seed.bonds},
                        "encoder_mean": mean, "normalized_mean": normalized,
                        "decoder_hard": {"atoms": hard.atoms, "charges": hard.charges,
                                         "bonds": hard.bonds}}, run.path / "seed_state.pt")
            write_json(run.path / "summary.json", summary)
            run.log({"stage": "complete", **summary})
        return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("ae", "vae"), required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    args = parser.parse_args()
    run_preflight(args.checkpoint, args.reference, args.data_manifest, args.run_root,
                  mode=args.mode,
                  expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                  expected_reference_sha256=args.expected_reference_sha256)


if __name__ == "__main__":
    main()
