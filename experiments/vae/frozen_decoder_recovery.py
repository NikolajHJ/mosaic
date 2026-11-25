"""Can per-molecule latent updates invert a frozen decoder better than its encoder?"""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from rdkit import rdBase
import torch

from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.training.generated_roundtrip import canonical_generated_parents
from latentmol.training.losses import LossConfig, reconstruction_losses
from latentmol.training.matching import align_reconstruction_targets
from latentmol.training.occupancy_monitor import _inspect


@torch.no_grad()
def readout(decoded, targets, names, model, config):
    hard = decoded.harden(model.spec)
    aligned = align_reconstruction_targets(decoded, targets, config)
    predicted = (hard.bonds != 0).triu(1)
    aligned_edges = (aligned.bonds != 0).triu(1)
    target_edges = (targets.bonds != 0).triu(1)
    extra = predicted & ~target_edges
    fields = torch.stack([extra, ~predicted & target_edges,
                          predicted & ~aligned_edges, ~predicted & aligned_edges]).sum((-1, -2)).T.tolist()
    rows = []
    for name, result, (ex, miss, a_ex, a_miss) in zip(names, _inspect(hard, model.spec), fields, strict=True):
        rows.append(dict(smiles=result.smiles, exact=result.smiles == name, valid=result.smiles is not None,
                         extra=ex, missing=miss, aligned_extra=a_ex, aligned_missing=a_miss))
    return rows, extra


def run(args):
    checkpoint_hash = file_sha256(args.checkpoint)
    if checkpoint_hash != args.sha256:
        raise ValueError("Checkpoint digest differs from the requested frozen snapshot")
    model, state = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False).to(args.device).eval()
    config = LossConfig(**state["loss_config"])
    if config.atom_order != "graph":
        raise ValueError("This comparison requires the checkpoint's graph-matched reconstruction loss")
    if args.output.exists():
        raise FileExistsError(args.output)
    # The production monitor draws 512 Gaussian vectors on CPU and screens the first 128.
    prior = torch.randn(512, model.config.latent, generator=torch.Generator().manual_seed(args.seed))[:128]
    checkpoints = (0, 1, 5, 10, 20, 50)
    records, rejections = [], Counter()
    for start in range(0, len(prior), args.batch):
        draws = prior[start:start + args.batch].to(args.device)
        with torch.no_grad():
            hard = model.decode(draws).harden(model.spec)
            targets, info = canonical_generated_parents(hard, model.spec, include_smiles=True)
            rejections.update(info["rejections"])
            if targets is None:
                continue
            names = info["smiles"]
            parents = [dict(prior_row=start + i, target=name, steps={})
                       for i, name in zip(info["rows"], names, strict=True)]
            original = model.decode(draws[info["rows"]])
            control, _ = readout(original, targets, names, model, config)
            if not all(row["exact"] and row["aligned_extra"] == row["aligned_missing"] == 0 for row in control):
                raise ValueError("Original latent did not reproduce its parent graph")
            mu, _ = model.encoder(targets, from_canonical_cache=True)
        z = mu.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=args.lr)
        for step in range(checkpoints[-1] + 1):
            decoded = model.decode(z)
            if not all(torch.isfinite(t).all() for t in (decoded.atoms, decoded.charges, decoded.bonds)):
                raise FloatingPointError("Nonfinite decoder output")
            if step in checkpoints:
                rows, extra = readout(decoded, targets, names, model, config)
                if step == 0:
                    baseline_extra = extra.clone()
                cleared = (baseline_extra & ~extra).sum((-1, -2)).tolist()
                new = (extra & ~baseline_extra).sum((-1, -2)).tolist()
                moves = (z.detach() - mu).norm(dim=-1).tolist()
                for parent, row, lost, gained, move in zip(parents, rows, cleared, new, moves, strict=True):
                    parent["steps"][step] = dict(row, original_extra_cleared=lost, new_extra=gained,
                                                 latent_displacement=move)
            if step == checkpoints[-1]:
                break
            # Rematch against the known graph at each step. Source z is strictly an evaluation control.
            aligned = align_reconstruction_targets(decoded, targets, config)
            loss = reconstruction_losses(decoded, aligned, replace(config, atom_order="ordered"))["reconstruction"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if z.grad is None or not torch.isfinite(loss) or not torch.isfinite(z.grad).all():
                raise FloatingPointError("Missing or nonfinite latent gradient or loss")
            optimizer.step()
            with torch.no_grad():
                offset = z - mu
                scale = (args.radius / offset.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
                z.copy_(mu + offset * scale)
        records.extend(parents)
        print(f"Completed {start + len(draws)}/128 draws, {len(records)} valid parents", flush=True)
    if not records or len(records) + sum(rejections.values()) != len(prior):
        raise ValueError("Incomplete generated-parent cohort")
    if any(p.grad is not None for p in model.parameters()) or any(
        not torch.equal(t.cpu(), state["model_state"][key]) for key, t in model.state_dict().items()
    ):
        raise RuntimeError("Frozen VAE changed or accumulated gradients")
    summary = {}
    for step in checkpoints:
        rows = [p["steps"][step] for p in records]
        totals = {k: sum(r[k] for r in rows) for k in (
            "exact", "valid", "extra", "missing", "aligned_extra", "aligned_missing",
            "original_extra_cleared", "new_extra")}
        totals["newly_exact"] = sum(r["exact"] and not p["steps"][0]["exact"] for p, r in zip(records, rows))
        totals["lost_exact"] = sum(not r["exact"] and p["steps"][0]["exact"] for p, r in zip(records, rows))
        totals["mean_displacement"] = sum(r["latent_displacement"] for r in rows) / len(rows)
        summary[step] = totals
    result = dict(update=state["continuation"]["updates"], sha256=checkpoint_hash, seed=args.seed,
                  draws=len(prior), selected=len(records), distinct_targets=len({p["target"] for p in records}),
                  rejections=dict(rejections), original_latent_exact=len(records), frozen_state_verified=True,
                  device=str(args.device), gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  batch=args.batch, learning_rate=args.lr, max_displacement=args.radius,
                  loss=vars(config), summary=summary, parents=records)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=2026111909)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--radius", type=float, default=4.)
    args = parser.parse_args()
    if args.batch < 1 or args.lr <= 0 or args.radius <= 0:
        parser.error("Expected positive batch, step size and radius")
    torch.set_num_threads(2)
    with rdBase.BlockLogs():
        run(args)
