"""Test a marginal over verified ring bond assignments with the ordinary bond-only decoder."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase

from latentmol.chemistry import _label_molecule
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.motif_completion import _roundtrips, _valid
from latentmol.train import state_digest


def patterns(spec):
    a, s, d = [spec.bond_names.index(name) for name in ("AROMATIC", "SINGLE", "DOUBLE")]
    return torch.tensor([[a] * 6, [s, d] * 3, [d, s] * 3])


def select_sites(hard, spec):
    """Use the first eligible isolated neutral C/N six-ring per row, verified by whole-graph sanitization."""
    rings, offsite = [], []
    counts = Counter()
    labels = patterns(spec).tolist()
    for row, (a, c, b) in enumerate(zip(hard.atoms.numpy(), hard.charges.numpy(), hard.bonds.numpy())):
        active = np.flatnonzero(a)
        if not len(active):
            continue
        adjacent = b[np.ix_(active, active)]
        mol = _label_molecule(a, c, adjacent, spec, active)
        if not _valid(mol) or not _roundtrips(mol):
            continue
        counts["valid"] += 1
        seen = mol.GetRingInfo()
        exclude = set()
        for ring in seen.AtomRings():
            if len(ring) != 6 or any(seen.NumAtomRings(i) != 1 for i in ring):
                continue
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if any(
                x.GetAtomicNum() not in (6, 7) or x.GetFormalCharge() or x.GetIsAromatic() for x in atoms
            ):
                continue
            edges = list(zip(ring, ring[1:] + ring[:1]))
            if any(
                mol.GetBondBetweenAtoms(i, j).GetBondType()
                not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE)
                for i, j in edges
            ):
                continue
            counts["attempted_ring"] += 1
            identities = []
            for form in labels:
                edited = adjacent.copy()
                for (i, j), t in zip(edges, form, strict=True):
                    edited[i, j] = edited[j, i] = t
                proposal = _label_molecule(a, c, edited, spec, active)
                if not _valid(proposal) or not _roundtrips(proposal):
                    break
                if not all(proposal.GetBondBetweenAtoms(i, j).GetIsAromatic() for i, j in edges):
                    break
                identities.append(Chem.MolToSmiles(proposal, canonical=True, isomericSmiles=False))
            if len(identities) == 3 and len(set(identities)) == 1:
                rings.append([(row, int(active[i]), int(active[j])) for i, j in edges])
                exclude = set(ring)
                break
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            pair = sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])
            if (
                not bond.IsInRing()
                and not exclude.intersection((i, j))
                and bond.GetBondType() == Chem.BondType.SINGLE
                and pair in ([6, 6], [6, 7])
            ):
                offsite.append((row, int(active[i]), int(active[j])))
    counts["selected"] = len(rings)
    counts["acyclic_unrelated_edges"] = len(offsite)
    return rings, offsite, counts


def ring_logprobs(logits, rings, labels):
    row, i, j = torch.tensor(rings).unbind(-1)
    conditional = logits[row, i, j, 1:].double().log_softmax(-1)
    # Sum log probabilities over six distinct edges, then aggregate alternatives.
    return torch.stack(
        [
            conditional.gather(-1, (t - 1)[None, :, None].expand(len(rings), -1, -1)).sum((1, 2))
            for t in labels
        ],
        dim=-1,
    )


def values(model, latents, rings, offsite):
    l = model.decode(latents).symmetric_bond_logits
    labels = patterns(model.spec)
    result = {}
    if rings:
        logp = ring_logprobs(l, rings, labels)
        result["ring_log_event"] = torch.logsumexp(logp, dim=-1)
        result["ring_explicit_log_probability"] = logp[:, 0]
        result["ring_kekule_log_probability"] = torch.logsumexp(logp[:, 1:], dim=-1)
    if offsite:
        row, i, j = torch.tensor(offsite).T
        local = l[row, i, j].double()
        aro, single, double = labels[0, 0], labels[1, 0], labels[1, 1]
        result["offsite_aromatic_minus_single"] = local[:, aro] - local[:, single]
        result["offsite_double_minus_single"] = local[:, double] - local[:, single]
        result["offsite_aromatic_or_double_probability"] = (
            local[:, 1:].softmax(-1)[:, [aro - 1, double - 1]].sum(-1)
        )
    return result


def describe(values):
    if not len(values):
        return {"n": 0}
    x = np.array(values)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.1)),
        "p90": float(np.quantile(x, 0.9)),
        "fraction_positive": float(np.mean(x > 0)),
    }


def run(args):
    if min(args.draws, args.threads, args.microbatch) < 1 or not 0 < args.delta < 0.1:
        raise ValueError("Require positive budgets and a small finite parameter step")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    model, state = load_model(args.checkpoint, "cpu")
    model.eval().requires_grad_(False)
    original = state_digest(model.state_dict())
    params = tuple(model.decoder.bond_head[-1].parameters())
    for par in params:
        par.requires_grad_(True)
    weights = [p.detach().clone() for p in params]
    methods = ("explicit_only", "three_forms")
    sums = {name: [torch.zeros_like(p) for p in params] for name in methods}
    stats = {}
    blocks = []
    baseline = []
    for panel, seed in (("fit", args.seed), ("fresh", args.seed + 1)):
        totals = Counter()
        z = torch.randn(args.draws, model.config.latent, generator=torch.Generator().manual_seed(seed))
        for zz in z.split(args.microbatch):
            with torch.no_grad(), rdBase.BlockLogs():
                hard = model.decode(zz).harden(model.spec)
                rings, offsite, counts = select_sites(hard, model.spec)
            totals.update(counts)
            blocks.append((panel, zz, rings, offsite))
            if panel == "fit" and rings:
                out = model.decode(zz)
                logp = ring_logprobs(out.symmetric_bond_logits, rings, patterns(model.spec))
                for index, name in enumerate(methods):
                    comparable = logp[:, 0] if index == 0 else torch.logsumexp(logp, dim=-1)
                    grads = torch.autograd.grad(-comparable.sum(), params, retain_graph=index == 0)
                    for acc, grad in zip(sums[name], grads, strict=True):
                        acc.add_(grad.detach())
                baseline.extend(logp.detach().tolist())
        stats[panel] = dict(totals)
    report = {
        "checkpoint": file_sha256(args.checkpoint),
        "update": state["continuation"]["updates"],
        "seed": args.seed,
        "per_panel": args.draws,
        "delta": args.delta,
        "counts": stats,
        "torch": torch.__version__,
        "rdkit": rdBase.rdkitVersion,
        "note": "Unit descent direction in the reused final linear head (all invocations); baseline selection and graph slots held fixed. No trained checkpoint or sampler change.",
    }
    if not stats["fit"].get("selected", 0):
        write_json(args.out / "summary.json", report)
        return
    resp = torch.tensor(baseline, dtype=torch.float64).softmax(-1)
    report["mean_conditional_responsibility"] = dict(
        zip(("explicit", "alternating_a", "alternating_b"), resp.mean(0).tolist())
    )
    report["fit_ring_log_event_before"] = describe(torch.logsumexp(torch.tensor(baseline), dim=-1).tolist())
    directions = {}
    norms = {}
    for name in methods:
        norm = torch.sqrt(sum(g.square().sum() for g in sums[name]))
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError("Invalid head gradient")
        directions[name] = [-g / norm for g in sums[name]]
        norms[name] = float(norm) / stats["fit"]["selected"]
    report["gradient_norm_per_ring"] = norms
    report["descent_direction_cosine"] = sum(float((x * y).sum()) for x, y in zip(*directions.values()))
    report["directional"] = {}
    try:
        with torch.no_grad():
            for name, direction in directions.items():
                signs = []
                for sign in (-1, 1):
                    for par, ref, d in zip(params, weights, direction, strict=True):
                        par.copy_(ref + sign * args.delta * d)
                    data = {panel: {} for panel in stats}
                    for panel, z, rings, offsite in blocks:
                        for key, value in values(model, z, rings, offsite).items():
                            data[panel].setdefault(key, []).extend(value.tolist())
                    signs.append(data)
                result = {}
                for panel, measures in signs[0].items():
                    result[panel] = {}
                    for key, minus in measures.items():
                        derivative = (np.array(signs[1][panel][key]) - minus) / (2 * args.delta)
                        result[panel][key] = describe(derivative)
                report["directional"][name] = result
    finally:
        with torch.no_grad():
            for par, ref in zip(params, weights, strict=True):
                par.copy_(ref)
    if state_digest(model.state_dict()) != original:
        raise RuntimeError("Original head not exactly restored")
    write_json(args.out / "summary.json", report)
    print(json.dumps({"counts": stats, "directions": report["directional"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026112547)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.002)
    run(parser.parse_args())
