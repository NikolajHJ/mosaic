"""Measure how output temperature affects graph samples with latent vectors held fixed."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from statistics import median

import torch

from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.chemistry import inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import DecoderOutput
from latentmol.sampling import hard_graph_log_prob, sample_hard_graph
from latentmol.train import state_digest


def observations(graphs, spec):
    fields = [getattr(graphs, k).cpu().numpy() for k in ("atoms", "charges", "bonds")]
    return [inspect_graph(*row, spec) for row in zip(*fields, strict=True)]


def soften_conditional_types(logits, temperature):
    """Keep the probability of category zero and soften only the other categories."""
    logp = logits.log_softmax(-1)
    real = torch.logsumexp(logp[..., 1:], -1, keepdim=True)
    conditional = (logits[..., 1:] / temperature).log_softmax(-1)
    result = torch.cat((logp[..., :1], real + conditional), -1)
    if not torch.allclose(result.softmax(-1)[..., 0], logp[..., 0].exp(), atol=1e-6, rtol=1e-6):
        raise ValueError("Category-zero probability changed")
    return result


@torch.inference_mode()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    checkpoint_sha = file_sha256(args.checkpoint)
    receipt = json.loads(args.checkpoint.with_name("latest.json").read_bytes())
    if checkpoint_sha != receipt["sha256"] or receipt["updates"] != 600116:
        raise ValueError("Unexpected checkpoint")
    model, _ = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False)
    before = state_digest(model.state_dict())
    panel = json.loads(args.panel.read_bytes())
    chosen = random.Random(args.seed).sample(panel["rows"], args.centers)
    prior = torch.randn(args.centers, model.config.latent,
                        generator=torch.Generator().manual_seed(args.seed + 1))
    results = {}
    for family in ("encoded_mean", "prior"):
        records = []
        for start in range(0, args.centers, 8):
            subset = chosen[start:start + 8]
            targets = None
            if family == "encoded_mean":
                targets = encode([r[2] for r in subset], model.spec)
                latent, _ = model.encoder(targets, from_canonical_cache=True)
                originals = [o.smiles for o in observations(targets, model.spec)]
            else:
                latent = prior[start:start + len(subset)]
                originals = [None] * len(subset)
            decoded = model.decode(latent)
            modes = decoded.harden(model.spec)
            parents = observations(modes, model.spec)
            batch = [dict(center=start + i, source_smiles=sm, mode_smiles=p.smiles,
                          mode_failure=p.failure_reason, temperature={})
                     for i, (sm, p) in enumerate(zip(originals, parents, strict=True))]
            variants = [(str(t), decoded, t) for t in args.temperatures]
            if args.conditional_types:
                # The repo uses index zero for NO_ATOM and NO_BOND. Charge is untouched.
                atom_types = soften_conditional_types(decoded.atoms, 1.5)
                bond_orders = soften_conditional_types(decoded.symmetric_bond_logits, 1.5)
                for name, atoms, bonds in (("element_only", atom_types, decoded.bonds),
                                           ("bond_order_only", decoded.atoms, bond_orders),
                                           ("element_and_bond_order", atom_types, bond_orders)):
                    variants.append((name, DecoderOutput(atoms, decoded.charges, bonds), 1.0))
            for label, distribution, temp in variants:
                # Restart the RNG for each temperature so differences use paired draws.
                gen = torch.Generator().manual_seed(args.seed + 100 + start)
                logp = hard_graph_log_prob(distribution, targets if targets is not None else modes,
                                           model.spec, temperature=temp)
                if not logp.isfinite().all():
                    raise ValueError("Nonfinite log probability for an in-support graph")
                draws = [[] for _ in batch]
                for _ in range(args.draws):
                    graphs = sample_hard_graph(distribution, model.spec, generator=gen, temperature=temp)
                    for saved, observation in zip(draws, observations(graphs, model.spec), strict=True):
                        saved.append(dict(smiles=observation.smiles, failure=observation.failure_reason))
                for i, record in enumerate(batch):
                    record["temperature"][label] = dict(target_logp=logp[i].item(), draws=draws[i])
            records.extend(batch)
        summaries = {}
        for label in records[0]["temperature"]:
            total_valid = total_same = total_exact = changed_from_valid = rescued = 0
            per_center_different = []
            failures = Counter()
            for r in records:
                sampled = r["temperature"][label]["draws"]
                valid = [d["smiles"] for d in sampled if d["smiles"] is not None]
                failures.update(d["failure"] for d in sampled if d["failure"])
                total_valid += len(valid)
                total_same += sum(s == r["mode_smiles"] for s in valid)
                total_exact += sum(s == r["source_smiles"] for s in valid)
                different = {s for s in valid if r["mode_smiles"] and s != r["mode_smiles"]}
                per_center_different.append(len(different))
                if r["mode_smiles"]:
                    changed_from_valid += sum(s != r["mode_smiles"] for s in valid)
                else:
                    rescued += len(valid)
            summaries[label] = dict(draws=len(records)*args.draws, valid=total_valid,
                identical_to_mode=total_same, exact_source=total_exact if family == "encoded_mean" else None,
                differing_draws_from_valid_mode=changed_from_valid, valid_draws_from_invalid_mode=rescued,
                centers_with_distinct_valid_alternative=sum(n > 0 for n in per_center_different),
                valid_modes=sum(r["mode_smiles"] is not None for r in records),
                exact_modes=sum(r["mode_smiles"] == r["source_smiles"] for r in records)
                    if family == "encoded_mean" else None,
                median_logp_target_slots=median(r["temperature"][label]["target_logp"] for r in records),
                failure_counts=dict(failures))
        results[family] = dict(summary=summaries, centers=records)
    if state_digest(model.state_dict()) != before:
        raise ValueError("Frozen checkpoint changed")
    payload = dict(checkpoint_sha256=checkpoint_sha, panel_sha256=file_sha256(args.panel),
                   seed=args.seed, draws=args.draws, sampled_rows=chosen, groups=results)
    (args.output / "results.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf8")
    for family, group in results.items():
        print(family, json.dumps(group["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--centers", type=int, default=96)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026112351)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.4, 0.7, 1.0, 1.5, 2.0])
    parser.add_argument("--conditional-types", action="store_true",
                        help="Also warm conditional elements and/or bond orders to 1.5 at fixed presence odds")
    run(parser.parse_args())
