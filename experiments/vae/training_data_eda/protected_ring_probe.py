"""Matched short optimization of the VAE's existing bond readout, with fixed prior panels."""

import argparse
from collections import Counter
import json
from pathlib import Path

from rdkit import rdBase
import torch

from experiments.vae.training_data_eda.aromatic_encodings_probe import patterns, ring_logprobs, select_sites
from experiments.vae.training_data_eda.teacher_motif_training_pilot import hard_records
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.train import state_digest


@torch.no_grad()
def panel(model, count, batch, seed):
    codes = torch.randn(count, model.config.latent, generator=torch.Generator().manual_seed(seed))
    blocks = []
    for z in codes.split(batch):
        anchor = model.decode(z)
        hard = anchor.harden(model.spec)
        rings, unrelated, _ = select_sites(hard, model.spec)
        offsite = hard.bonds.triu(1).ne(0)
        for ring in rings:
            for row, i, j in ring:
                offsite[row, min(i, j), max(i, j)] = False
        blocks.append((z, anchor, rings, unrelated, offsite))
    return blocks


@torch.no_grad()
def measure(model, blocks):
    total = Counter()
    for z, anchor, rings, unrelated, offsite in blocks:
        out = model.decode(z)
        logits = out.symmetric_bond_logits
        old = anchor.symmetric_bond_logits
        probs, before = logits.double().softmax(-1), old.double().softmax(-1)
        if rings:
            result = ring_logprobs(logits, rings, patterns(model.spec)).logsumexp(-1)
            total["target_rings"] += len(rings)
            total["ring_log_event_sum"] += result.sum().item()
        total["protected_present_edges"] += offsite.sum().item()
        p, p0 = probs[offsite], before[offsite]
        total["offsite_tv_sum"] += (p - p0).abs().sum().item() / 2
        total["offsite_bond_class_changes"] += (p.argmax(-1) != p0.argmax(-1)).sum().item()
        total["offsite_presence_changes"] += ((p.argmax(-1) != 0) != (p0.argmax(-1) != 0)).sum().item()
        if unrelated:
            row, i, j = torch.tensor(unrelated).T
            ids = [model.spec.bond_names.index(t) for t in ("AROMATIC","DOUBLE")]
            change = (probs[row, i, j][:, ids] - before[row, i, j][:, ids]).sum(-1)
            total["unrelated_single_edges"] += len(change)
            total["unrelated_abs_aromatic_or_double_change_sum"] += change.abs().sum().item()
        for rec in hard_records(out.harden(model.spec), model.spec):
            total["attempts"] += 1
            total["valid"] += int(rec["valid"])
            total["valid_aromatic"] += int(rec["aromatic"])
    def avg(n,d):
        return total[n]/total[d] if total[d] else None
    return dict(total, mean_ring_log_event=avg("ring_log_event_sum","target_rings"),
                mean_offsite_tv=avg("offsite_tv_sum","protected_present_edges"),
                mean_unrelated_abs_aromatic_or_double_change=avg("unrelated_abs_aromatic_or_double_change_sum",
                                                                 "unrelated_single_edges"))


def main():
    from experiments.vae.training_data_eda.protected_ring_objective import protected_ring_loss

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[0., 1., 8., 32.])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=2026112562)
    parser.add_argument("--fit", type=int, default=160)
    parser.add_argument("--fresh", type=int, default=160)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if min(args.steps,args.fit,args.fresh,args.batch,args.lr) <= 0 or min(args.weights) < 0:
        parser.error("Invalid budget or weights")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model, state = load_model(args.checkpoint, "cpu")
    model.eval().requires_grad_(False)
    original = state_digest(model.state_dict())
    readout = model.decoder.bond_head[-1]
    initial = {k:v.detach().clone() for k,v in readout.state_dict().items()}
    with rdBase.BlockLogs():
        panels = {name:panel(model, n, args.batch, args.seed + i)
                  for i,(name,n) in enumerate((("fit",args.fit),("fresh",args.fresh)))}
        baseline = {name:measure(model,b) for name,b in panels.items()}
        fit = [block for block in panels["fit"] if block[2]]
        if not fit:
            raise ValueError("No eligible ring present in fixed fit panel")
        result = dict(checkpoint=file_sha256(args.checkpoint), global_update=state["continuation"]["updates"],
                      objective_sha256=file_sha256(Path(__file__).with_name("protected_ring_objective.py")),
                      fit_seed=args.seed, fresh_seed=args.seed+1, lr=args.lr, steps=args.steps,
                      fit_blocks_with_sites=len(fit), batch=args.batch, baseline=baseline, arms={},
                      trainable="shared final bond linear weights and bias, including reuse in refinement rounds")
        readout.requires_grad_(True)
        for weight in args.weights:
            readout.load_state_dict(initial)
            optim = torch.optim.Adam(readout.parameters(), lr=args.lr)
            activity=[]
            for i in range(args.steps):
                z,anchor,*_ = fit[i % len(fit)]
                optim.zero_grad(set_to_none=True)
                loss,stats=protected_ring_loss(model.decode(z),anchor,model.spec,retention_weight=weight)
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite objective")
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(readout.parameters(),10.,error_if_nonfinite=True)
                optim.step()
                activity.append(dict(step=i+1,loss=loss.item(),grad_norm=float(norm),stats=stats))
            results={name:measure(model,b) for name,b in panels.items()}
            result["arms"][str(weight)] = dict(metrics=results,updates=activity)
            print(weight,{name:(r["mean_ring_log_event"],r["mean_offsite_tv"]) for name,r in results.items()},flush=True)
        readout.load_state_dict(initial)
        if state_digest(model.state_dict()) != original:
            raise RuntimeError("Probe failed to restore the frozen checkpoint identity")
    with args.output.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__ == "__main__":
    main()
