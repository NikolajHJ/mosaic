"""Read-only latent-gradient diagnostic on the first eligible rings of fresh Gaussian draws."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch
from rdkit import rdBase

from experiments.vae.training_data_eda.shared_ring_gate import SharedRingGate, find_ring_sites, relaxed_ring_bonds
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "gate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026112559)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--max-draws", type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.examples <= args.max_draws:
        parser.error("Require 0 < examples <= max-draws")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model, _ = load_model(args.checkpoint, "cpu")
    model.eval().requires_grad_(False)
    gate = SharedRingGate(model.spec).eval().requires_grad_(False)
    gate.load_state_dict(torch.load(args.gate, weights_only=True, map_location="cpu"))
    codes = torch.randn(args.max_draws, model.config.latent, generator=torch.Generator().manual_seed(args.seed))
    names = ("gate_probability", "baseline_aromatic_marginal", "shared_aromatic_marginal")
    rows = []
    with rdBase.BlockLogs():
        for index, code in enumerate(codes):
            z = code[None, :].detach().requires_grad_()
            decoded = model.decode(z)
            hard = decoded.harden(model.spec)
            sites = [s for s in find_ring_sites(hard, model.spec) if s.saturated]
            if not sites:
                continue
            # The hard slot set stays fixed for autograd and finite differences.
            slots = torch.tensor(sites[0].slots)
            label = model.spec.bond_names.index("AROMATIC")

            def values(prediction):
                graph = prediction.probabilities()
                raw = graph.bonds[0, slots, slots.roll(-1), label].mean()
                shared = relaxed_ring_bonds(gate, hard, prediction, model.spec)[0, slots, slots.roll(-1), label].mean()
                return (gate(graph, sites).sigmoid()[0], raw, shared)

            scalars = values(decoded)
            row = dict(index=index, slots=slots.tolist())
            for i, (name, scalar) in enumerate(zip(names, scalars, strict=True)):
                grad, = torch.autograd.grad(scalar, z, retain_graph=i < 2)
                norm = float(grad.norm())
                if not torch.isfinite(grad).all() or not torch.isfinite(scalar):
                    raise FloatingPointError(f"Nonfinite {name} at prior draw {index}")
                measure = dict(value=float(scalar.detach()), grad_norm=norm)
                # With saturated logits a small magnitude can be below FP32 differencing resolution.
                if norm:
                    unit = grad.detach() / norm
                    with torch.no_grad():
                        for eps in (0.001, 0.005):
                            upper = values(model.decode(z.detach() + eps * unit))[i]
                            lower = values(model.decode(z.detach() - eps * unit))[i]
                            measure["central_directional_" + str(eps)] = float((upper - lower) / (2 * eps))
                row[name] = measure
            rows.append(row)
            if len(rows) >= args.examples:
                break
    if any(p.grad is not None for p in (*model.parameters(), *gate.parameters())):
        raise RuntimeError("Frozen weights accumulated gradients")
    summary = {n: dict(nonzero=sum(r[n]["grad_norm"] > 0 for r in rows),
                       min=min(r[n]["grad_norm"] for r in rows),
                       median=median(r[n]["grad_norm"] for r in rows),
                       max=max(r[n]["grad_norm"] for r in rows)) for n in names if rows}
    result = dict(checkpoint_sha256=file_sha256(args.checkpoint), gate_sha256=file_sha256(args.gate),
                  seed=args.seed, examples_requested=args.examples, draws_examined=index + 1,
                  selected=len(rows), platform="cpu-fp32", torch=torch.__version__, rows=rows, summary=summary)
    with args.output.open("x", encoding="utf8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(dict(examined=index+1, selected=len(rows), summary=summary), indent=2))


if __name__ == "__main__":
    main()
