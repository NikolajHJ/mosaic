"""Compare Boltz confidence and ipSAE latent gradients at one fresh initial state.

This is a bounded, separate native GPU probe. The random latent, occupancy and
temperature follow the supplied design config. It neither starts optimization
nor touches existing runs, model weights, contexts or checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from ipsae_gradient_probe import compare_gradients
from latentmol.design.latent_boltz import BoltzLatentScorer
from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity, choose_active, schedules
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import RelaxedGraph


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Decoder checkpoint hash differs")
    if not math.isfinite(args.ipsae_weight) or args.ipsae_weight <= 0:
        raise ValueError("Probe requires a finite positive ipSAE weight")
    config = LatentDesignConfig.from_mapping(json.loads(args.config.read_text(encoding="utf-8")))
    native = json.loads(args.native_config.read_text(encoding="utf-8"))
    native["ipsae_weight"] = args.ipsae_weight
    decoder, _ = load_model(args.checkpoint, "cpu")
    decoder.eval().requires_grad_(False)
    if any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder has existing gradients")
    before = _model_identity(decoder)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    point = torch.randn((1, decoder.config.latent), generator=generator).requires_grad_(True)
    with torch.no_grad():
        active = choose_active(decoder.decode(point), decoder.spec, config.occupied_atoms)
    temperature = schedules(config, 0)["temperature"]
    graph = graph_from_decoder(decoder.decode(point), decoder.spec,
                               temperature=temperature, fixed_active=active)
    scorer = BoltzLatentScorer(decoder.spec, active, native, device=args.device)
    try:
        shared = RelaxedGraph(*(getattr(graph, name).to(scorer.device) for name in ("atoms", "charges", "bonds")))
        # Branch returns attached scalar tensors, unlike detached scorer aux.
        values = scorer.branch(shared)
        row = compare_gradients(values, point, args.ipsae_weight,
                                scorer.scorer.dbmol_config.confidence_weight)
        identity = scorer.identity
    finally:
        scorer.close()
    if _model_identity(decoder) != before or any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder changed")
    report = dict(version="ipsae-initial-latent-gradient-v1", checkpoint_sha256=args.checkpoint_sha256,
                  native_config=str(args.native_config), config=asdict(config), step=0,
                  latent_seed=config.seed, latent=point.detach().cpu().flatten().tolist(),
                  active_slots=torch.nonzero(active).flatten().tolist(), temperature=temperature,
                  native_identity=identity, **row,
                  technical_checks=dict(finite_component_gradients=True, frozen_decoder_unchanged=True,
                                        frozen_native_close_completed=True),
                  limits="One fixed-pose soft graph. No optimization, hard-molecule improvement, native gradient qualification or binding claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(dict(output=str(args.output), confidence_gradient_norm=row["original_confidence_gradient_norm"],
                          ipsae_gradient_norm=row["ipsae_loss_gradient_norm"], cosine=row["cosine"])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ipsae-weight", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
