"""Actual frozen-VAE repair gradient probe, CPU only inside a DTU allocation.

Use a saved script rather than multiline python -c: LSF's job-command wrapping
can alter nested quotes before Python starts. This is not decoder training,
CUDA validation or evidence of improved independent generation.
"""

import argparse
import os
from pathlib import Path

import torch

from latentmol.decoder_finetune import make_panel
from latentmol.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.search_decoder_study import decode_panel, probe_fixed_teacher_gradient, read_protocol
from latentmol.search_repair_guidance import SearchRepairTeacher
from latentmol.train import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Actual-checkpoint numerical probe requires a DTU allocation")
    config = read_protocol(args.config, args.sha256)
    checkpoint_path = Path(config["initializer"])
    if file_sha256(checkpoint_path) != config["initializer_sha256"]:
        raise ValueError("Actual-checkpoint probe identity differs")
    seed_everything(17)
    model, checkpoint = load_model(checkpoint_path, "cpu")
    if checkpoint["dataset_sha256"] != config["dataset_sha256"] or checkpoint["train_config"]["mode"] != "vae":
        raise ValueError("Expected authenticated protected VAE")
    panel = make_panel(config["training_prior_count"], model.config.latent, 71000)
    teacher = SearchRepairTeacher(panel, model.spec, ProbabilityRepairConfig(**config["repair"]),
                                 generator=torch.Generator().manual_seed(71001))
    args.output_root.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        receipt = teacher.refresh(decode_panel(model, panel), update=0, directory=args.output_root / "search")
    probe = probe_fixed_teacher_gradient(model, teacher, panel, args.output_root / "probe")
    summary = {"device": "cpu", "initializer_sha256": config["initializer_sha256"],
        "source_archive_sha256": os.environ.get("LATENTMOL_SOURCE_SHA256"),
        "config_sha256": args.sha256,
        "search": {key: value for key, value in receipt.items() if key != "attempts"},
        "probe": probe,
        "scope": "Actual frozen FP32 VAE, all prescribed training-prior points; CPU only; no training or CUDA/generation efficacy claim"}
    write_json(args.output_root / "summary.json", summary)
    import json
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
