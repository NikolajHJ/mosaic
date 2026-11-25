"""Compare ipSAE-only and original-confidence latent gradients at saved states.

Uses the same live Boltz forward, fixed pose, graph, latent and temperature for
both losses. This reads finished or explicitly saved run outputs; it never
modifies them. The two gradient vectors are loss gradients (both minimized).
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path

import numpy as np
import torch

from latentmol.design.latent_boltz import BoltzLatentScorer
from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity, schedules
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import RelaxedGraph


def compare_gradients(values, point, ipsae_weight, confidence_weight):
    """Compare unweighted objective gradients, retaining exact per-vector values."""
    ipsae = values["ligand_ipsae"]
    # dbmol_confidence_loss includes -ipsae_weight*ipsae when enabled.
    confidence = values["dbmol_confidence_loss"] + ipsae_weight * ipsae
    ip_loss = -ipsae
    for name, value in (("confidence", confidence), ("ipsae", ip_loss)):
        if value.numel() != 1 or not value.requires_grad or not bool(value.isfinite().all()):
            raise ValueError(f"{name} must be a live finite scalar")
    conf_grad, = torch.autograd.grad(confidence, point, retain_graph=True, allow_unused=True)
    ip_grad, = torch.autograd.grad(ip_loss, point, retain_graph=False, allow_unused=True)
    if any(g is None or not bool(g.isfinite().all()) for g in (conf_grad, ip_grad)):
        raise ValueError("Missing/nonfinite latent component gradient")
    conf = conf_grad.detach().double().flatten().cpu()
    ip = ip_grad.detach().double().flatten().cpu()
    cn, pn = float(conf.norm()), float(ip.norm())
    cosine = float(torch.dot(conf, ip) / (cn * pn)) if cn and pn else None
    if cosine is not None:
        cosine = max(-1., min(1., cosine))
    return dict(
        ipsae=float(ipsae.detach()), original_confidence_loss=float(confidence.detach()),
        original_confidence_gradient_norm=cn, ipsae_loss_gradient_norm=pn,
        ipsae_to_confidence_norm_ratio=(pn / cn if cn else None),
        cosine=cosine, angle_degrees=(math.degrees(math.acos(cosine)) if cosine is not None else None),
        confidence_loss_gradient=conf.tolist(), ipsae_loss_gradient=ip.tolist(),
        confidence_outer_weight=float(confidence_weight), ipsae_inner_weight=float(ipsae_weight),
        effective_confidence_gradient_norm=confidence_weight * cn,
        effective_added_ipsae_gradient_norm=confidence_weight * ipsae_weight * pn,
        effective_added_to_confidence_norm_ratio=(ipsae_weight * pn / cn if cn else None),
        zero_gradient=(cn == 0 or pn == 0),
        scope="loss gradients at the identical decoded relaxed graph, latent and fixed pose; no diffusion or coordinate gradient")


def load_states(run_path, steps):
    metadata = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    names = {field.name for field in fields(LatentDesignConfig)}
    config = LatentDesignConfig.from_mapping({k: v for k, v in metadata["config"].items() if k in names})
    requested = set(steps)
    selected = {}
    for path in sorted((run_path / "arms" / "latent" / "raw-blocks").glob("states-*.npz")):
        with np.load(path, allow_pickle=False) as raw:
            for i, step in enumerate(raw["steps"].tolist()):
                step = int(step)
                if step in requested:
                    if step in selected:
                        raise ValueError(f"Duplicate saved step {step} in {run_path}")
                    selected[step] = np.array(raw["latent"][i], copy=True)
    if set(selected) != requested:
        raise ValueError(f"Missing steps {sorted(requested-set(selected))} in {run_path}")
    return metadata, config, selected


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Decoder checkpoint hash differs")
    steps = [int(x) for x in args.steps.split(",")]
    if not steps or len(steps) != len(set(steps)) or any(x < 0 for x in steps):
        raise ValueError("Require distinct nonnegative steps")
    native = json.loads(args.native_config.read_text(encoding="utf-8"))
    native["ipsae_weight"] = args.ipsae_weight
    if not math.isfinite(args.ipsae_weight) or args.ipsae_weight <= 0:
        raise ValueError("Probe requires a finite positive ipSAE weight")
    decoder, _ = load_model(args.checkpoint, "cpu")
    decoder.eval().requires_grad_(False)
    before = _model_identity(decoder)
    rows = []
    for run_path in args.run:
        metadata, config, states = load_states(run_path, steps)
        if metadata["checkpoint_sha256"] != args.checkpoint_sha256:
            raise ValueError("Run decoder checkpoint differs from probe")
        active = torch.zeros(decoder.spec.max_nodes, dtype=torch.bool)
        active[metadata["active_slots"]] = True
        if int(active.sum()) != config.occupied_atoms:
            raise ValueError("Run occupancy metadata differs")
        scorer = BoltzLatentScorer(decoder.spec, active, native, device=args.device)
        try:
            for step in steps:
                point = torch.as_tensor(states[step], dtype=torch.float32).reshape(1, -1).requires_grad_(True)
                if point.shape != (1, decoder.config.latent) or not bool(point.isfinite().all()):
                    raise ValueError("Invalid saved latent")
                temperature = schedules(config, step)["temperature"]
                graph = graph_from_decoder(decoder.decode(point), decoder.spec,
                                           temperature=temperature, fixed_active=active)
                shared = RelaxedGraph(*(getattr(graph, k).to(scorer.device) for k in ("atoms", "charges", "bonds")))
                # Deliberately use live tensors from the branch, not detached __call__ aux.
                values = scorer.branch(shared)
                row = compare_gradients(values, point, args.ipsae_weight,
                                        scorer.scorer.dbmol_config.confidence_weight)
                rows.append(dict(run=str(run_path), step=step, temperature=temperature, **row))
                print(json.dumps(dict(run=str(run_path), step=step, cosine=row["cosine"],
                                      confidence_norm=row["original_confidence_gradient_norm"],
                                      ipsae_norm=row["ipsae_loss_gradient_norm"])), flush=True)
        finally:
            scorer.close()
    if _model_identity(decoder) != before or any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder changed")
    report = dict(version="ipsae-latent-gradient-probe-v1", checkpoint_sha256=args.checkpoint_sha256,
                  native_config=str(args.native_config), runs=[str(path) for path in args.run],
                  steps=steps, ipsae_weight=args.ipsae_weight, rows=rows,
                  limits="Frozen fixed-pose soft graphs; no correctness, discrete molecule or binding claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--native-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", default="0,250,500")
    parser.add_argument("--ipsae-weight", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()

