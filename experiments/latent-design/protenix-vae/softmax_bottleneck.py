"""Localize temperature-dependent gradient attenuation in a frozen molecular decoder."""

import argparse
import json
from pathlib import Path
import random
from statistics import median

from rdkit import rdBase
import torch

from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.chemistry import inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.train import state_digest


SEED = 2026112289
TEMPERATURES = (1.0, 0.4, 0.7, 1.5, 2.0)
CHECKPOINT = Path("outputs/runs/e48-s02-early-stop-600116/checkpoint-000600116.pt")
PANEL = Path("outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json")
CHECKSUM = "be683894d49976ee7570b25bda9fefac6b8c53b9d9b55e767ea570b7d147690a"


def norm(vector):
    # Keep tiny but representable FP32 gradients when taking their norm.
    return vector.detach().double().norm().item()


def cosine(a, b):
    denominator = norm(a) * norm(b)
    if not denominator:
        return None
    return (a.detach().double() * b.detach().double()).sum().item() / denominator


def vectors(decoded):
    active = decoded.atoms[0].argmax(-1).ne(0)
    pairs = (active[:, None] & active[None, :]).triu(1)
    bonds = decoded.symmetric_bond_logits[0]
    present = bonds.argmax(-1).ne(0)
    # Conditional bonds isolate the head's softmax from the separate occupancy multiplier.
    return {
        "occupied_atom": decoded.atoms[0, active],
        "present_bond": bonds[pairs & present],
        "absent_bond": bonds[pairs & ~present],
        "charge": decoded.charges[0, active],
        "padding_atom": decoded.atoms[0, ~active],
    }


def trace(values, z, generator):
    weights = torch.randn(values.shape, generator=generator).to(values.device)
    weights -= weights.mean(-1, keepdim=True)
    weights /= weights.norm()
    raw, = torch.autograd.grad((values * weights).sum(), z, retain_graph=True)
    raw_norm = norm(raw)
    gap = values.detach().topk(2).values
    data = dict(rows=len(values), raw_latent_norm=raw_norm,
                median_winner_margin=(gap[:, 0] - gap[:, 1]).median().item(), temperatures={})
    unit_temp_gradient = None
    for t in TEMPERATURES:
        probs = (values / t).softmax(-1)
        # Same contrast measured first in probability coordinates, then in unscaled logit
        # and latent coordinates. Its probability-space gradient has unit norm by construction.
        prob_to_latent, prob_to_logits = torch.autograd.grad(
            (probs * weights).sum(), (z, values), retain_graph=True)
        if not all(g.isfinite().all() for g in (raw, prob_to_latent, prob_to_logits)):
            raise FloatingPointError("Nonfinite VJP")
        latent = norm(prob_to_latent)
        logits = norm(prob_to_logits)
        if t == 1.0:
            unit_temp_gradient = prob_to_latent.detach()
        data["temperatures"][str(t)] = dict(
            logit_vjp=logits, latent_vjp=latent,
            latent_to_raw_logit_ratio=latent / raw_norm if raw_norm else None,
            decoder_directional_gain=latent / logits if logits else None,
            cosine_to_T1=cosine(prob_to_latent, unit_temp_gradient),
            fraction_pmax_ge_099=probs.detach().amax(-1).ge(.99).float().mean().item())
    return data


def concentration(values):
    """Measure how broadly the softmax alone can respond to equal per-logit perturbations."""
    results = {}
    for t in TEMPERATURES:
        p = (values.detach().double() / t).softmax(-1)
        # Use sums of off-diagonal products for the diagonal too. Subtracting p
        # from one loses small, meaningful variation in nearly deterministic rows.
        diagonal = torch.eye(p.shape[-1], dtype=torch.bool, device=p.device)
        offdiag = (p[:, :, None] * p[:, None, :]).masked_fill(diagonal, 0.)
        energy = (offdiag.sum(-1).square().sum(-1) + offdiag.square().sum((-1, -2))) / t**2
        total = energy.sum().item()
        if total:
            shares = energy / energy.sum()
            results[str(t)] = dict(rows=len(values), largest_row_share=float(shares.max()),
                                   effective_rows=float(1.0 / shares.square().sum()),
                                   jacobian_frobenius=total**.5)
        else:
            results[str(t)] = dict(rows=len(values), largest_row_share=None,
                                   effective_rows=None, jacobian_frobenius=0.)
    return results


def summaries(records):
    summary = {}
    for family in sorted({r["family"] for r in records}):
        members = [r for r in records if r["family"] == family]
        heads = {}
        for head in sorted({name for r in members for name in r["heads"]}):
            hh = [r["heads"][head] for r in members if head in r["heads"]]
            levels = {}
            for t in TEMPERATURES:
                ss = [r["temperatures"][str(t)] for r in hh]
                row = {}
                for k in ss[0]:
                    numbers = [s[k] for s in ss if s[k] is not None]
                    row[k] = median(numbers) if numbers else None
                ratios = [s["latent_vjp"] / r["temperatures"]["1.0"]["latent_vjp"]
                          for s, r in zip(ss, hh) if r["temperatures"]["1.0"]["latent_vjp"] > 0]
                row.update(median_paired_latent_ratio_to_T1=median(ratios) if ratios else None,
                           zero_latent_vjps=sum(s["latent_vjp"] == 0 for s in ss))
                levels[str(t)] = row
            heads[head] = dict(n=len(hh), median_logit_latent_vjp=median(r["raw_latent_norm"] for r in hh),
                               median_logit_margin=median(r["median_winner_margin"] for r in hh),
                               temperatures=levels)
        summary[family] = dict(points=len(members), valid=sum(r["smiles"] is not None for r in members),
                               same_as_input=sum(r["smiles"] == r["source"] for r in members
                                                 if r["source"] is not None), heads=heads)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--points", default=24, type=int)
    parser.add_argument("--concentration", action="store_true",
                        help="Measure how many categorical rows carry the analytic softmax sensitivity")
    args = parser.parse_args()
    if args.output.exists() or args.points < 1:
        raise ValueError("A fresh result path and positive panel size are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if file_sha256(CHECKPOINT) != CHECKSUM:
        raise ValueError("Unexpected checkpoint")
    torch.set_num_threads(2)
    model, ck = load_model(CHECKPOINT, args.device)
    if ck["continuation"]["updates"] != 600116:
        raise ValueError("Unexpected checkpoint cursor")
    model.requires_grad_(False)
    before = state_digest(model.state_dict())
    rows = random.Random(SEED).sample(json.loads(PANEL.read_bytes())["rows"], args.points)
    labels = encode([r[2] for r in rows], model.spec).to(args.device)
    with torch.no_grad():
        means, logvars = model.encoder(labels, from_canonical_cache=True)
        prior = torch.randn(means.shape, generator=torch.Generator().manual_seed(SEED)).to(args.device)
        noise = torch.randn(means.shape, generator=torch.Generator().manual_seed(SEED + 1)).to(args.device)
        posterior = means + (.5 * logvars).exp() * noise
    records = []
    for family, batch in (("mean", means), ("posterior", posterior), ("prior", prior)):
        for i, center in enumerate(batch):
            z = center.detach()[None].requires_grad_(not args.concentration)
            out = model.decode(z)
            hard = out.harden(model.spec)
            with rdBase.BlockLogs():
                identity = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], model.spec)
            fields = {}
            for j, (head, values) in enumerate(vectors(out).items()):
                if len(values):
                    gen = torch.Generator().manual_seed(SEED + 1000 * i + j + 100)
                    fields[head] = concentration(values) if args.concentration else trace(values, z, gen)
            records.append(dict(family=family, index=i, source=rows[i][2] if family != "prior" else None,
                                smiles=identity.smiles, failure=identity.failure_reason, heads=fields))
        print("Finished", family, len(batch), flush=True)
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen checkpoint changed")
    result = dict(seed=SEED, device=args.device, torch=str(torch.__version__), checkpoint_sha256=CHECKSUM,
                  panel_sha256=file_sha256(PANEL), selected_rows=rows,
                  temperatures=TEMPERATURES, records=records)
    if not args.concentration:
        result["summary"] = summaries(records)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, allow_nan=False)


if __name__ == "__main__":
    main()
