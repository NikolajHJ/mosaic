"""Compare two latent representations of exactly the same generated molecular graph."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

import torch
from rdkit import rdBase

from latentmol.chemistry import decode_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.sampling import hard_graph_log_prob
from latentmol.training.generated_roundtrip import canonical_generated_parents


def identities(hard, spec):
    return [decode_graph(a, c, b, spec) for a, c, b in
            zip(hard.atoms, hard.charges, hard.bonds, strict=True)]


@torch.inference_mode()
def run(args):
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    ck_sha = file_sha256(args.checkpoint)
    receipt = json.loads(args.checkpoint.with_name("latest.json").read_bytes())
    if receipt["updates"] != 600116 or ck_sha != receipt["sha256"]:
        raise ValueError("Unexpected checkpoint")
    model, _ = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False)
    generator = torch.Generator().manual_seed(args.seed)
    proposals = torch.randn(args.count, model.config.latent, generator=generator)
    matched = []
    counts = Counter()
    for start in range(0, len(proposals), 8):
        prior = proposals[start:start+8]
        candidate = model.decode(prior).harden(model.spec)
        targets, screened = canonical_generated_parents(candidate, model.spec, include_smiles=True)
        if targets is None:
            continue
        means, _ = model.encoder(targets, from_canonical_cache=True)
        decoded = model.decode(means).harden(model.spec)
        for idx, orig, mu, actual in zip(screened["rows"], screened["smiles"], means,
                                          identities(decoded, model.spec), strict=True):
            counts["canonicalizable"] += 1
            if orig != actual:
                continue
            counts["same_chemical_identity"] += 1
            if len(matched) < args.pairs:
                matched.append((start+idx, orig, prior[idx], mu))
    if not matched:
        raise RuntimeError("No eligible pairs")
    records = []
    for idx, sm, z, mu in matched:
        directions = torch.randn(args.directions, len(z), generator=generator)
        directions = torch.nn.functional.normalize(directions, dim=-1)
        row = dict(prior_index=idx, smiles=sm, latent_distance=(z-mu).norm().item())
        for arm, center in (("prior", z), ("reencoded_mean", mu)):
            decoded = model.decode(center[None])
            mode = decoded.harden(model.spec)
            assert identities(mode, model.spec) == [sm]
            logp = hard_graph_log_prob(decoded, mode, model.spec).item()
            active = mode.atoms[0] != 0
            pairs = torch.triu(mode.bonds[0] != 0, diagonal=1)
            sharp = {}
            for field, logits in (("occupied_atoms", decoded.atoms[0, active]),
                                  ("present_bonds", decoded.symmetric_bond_logits[0, pairs])):
                if len(logits):
                    sharp[field] = float((logits.softmax(-1).amax(-1) >= .99).float().mean())
            arm_result = dict(log_probability_of_modal_slots=logp, latent_norm=center.norm().item(),
                              fraction_above_99=sharp, neighbors={})
            for radius in args.radii:
                hard = model.decode(center + radius*directions).harden(model.spec)
                arm_result["neighbors"][str(radius)] = identities(hard, model.spec)
            row[arm] = arm_result
        records.append(row)
    summary = {}
    for arm in ("prior", "reencoded_mean"):
        arm_stats = dict(median_logp_mode=median(r[arm]["log_probability_of_modal_slots"] for r in records),
            median_latent_norm=median(r[arm]["latent_norm"] for r in records), neighbors={})
        for radius in args.radii:
            draws = [(sm, r["smiles"]) for r in records for sm in r[arm]["neighbors"][str(radius)]]
            arm_stats["neighbors"][str(radius)] = dict(total=len(draws), same=sum(sm==parent for sm,parent in draws),
                different_valid=sum(sm is not None and sm!=parent for sm,parent in draws),
                invalid=sum(sm is None for sm,parent in draws))
        summary[arm] = arm_stats
    payload = dict(checkpoint_sha256=ck_sha, seed=args.seed, proposals=args.count, screening=counts,
                   eligible_tested=len(records), directions=args.directions, radii=args.radii,
                   summary=summary, records=records)
    (args.output/"results.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf8")
    print(json.dumps({k:v for k,v in payload.items() if k != "records"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026112352)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    with rdBase.BlockLogs():
        run(parser.parse_args())
