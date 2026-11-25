"""Locate loss of motifs along frozen posterior-to-prior paths on a held-out panel."""

import argparse
from collections import Counter
import gzip
import json
from math import sqrt
from pathlib import Path

import torch
from rdkit import rdBase

from generation_motifs import inspect_batch, motifs
from latentmol.chemistry import encode_canonical_smiles
from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec


def encode_panel(model, source, batch_size):
    means, logvars = [], []
    for start in range(0, len(source), batch_size):
        items = []
        for smiles in source[start:start + batch_size]:
            graph = encode_canonical_smiles(smiles, model.spec)
            items.append({k: torch.as_tensor(graph[v]).long() for k, v in (
                ("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})
        mean, logvar = model.encoder(collate_graphs(items), from_canonical_cache=True)
        means.append(mean)
        logvars.append(logvar)
    return torch.cat(means), torch.cat(logvars)


def posterior_and_noise(mu, lv, rng):
    epsilon = torch.randn(mu.shape, generator=rng)
    prior = torch.randn(mu.shape, generator=rng)
    return mu + (lv * .5).exp() * epsilon, prior


def latent_conditions(mu, lv, fit_mu, fit_lv, rng):
    posterior, prior = posterior_and_noise(mu, lv, rng)
    yield "encoded_mean", mu
    # t is the fraction of nominal noise variance. The independent terms have
    # squared coefficients summing to 1. Their actual covariance can differ.
    for t in (0., .25, .50, .75, 1.):
        yield f"prior_mix_{t:.2f}", sqrt(1 - t) * posterior + sqrt(t) * prior
    yield "prior_scaled_to_posterior_norm", prior * (
        posterior.norm(dim=1) / prior.norm(dim=1))[:, None]

    shuffled = torch.stack([posterior[torch.randperm(len(mu), generator=rng), k]
                            for k in range(mu.shape[1])], dim=1)
    yield "posterior_coordinate_shuffle", shuffled
    # Fit the conditional mixture's mean and covariance without any evaluation rows.
    # Diagonal posterior variance keeps the covariance full-rank when there are
    # fewer reference molecules than latent coordinates.
    center = fit_mu.double().mean(0)
    centered = fit_mu.double() - center
    cov = centered.T @ centered / len(centered) + torch.diag(fit_lv.double().exp().mean(0))
    yield "fitted_gaussian_diagonal", (center + prior.double() * cov.diag().sqrt()).float()
    cholesky = torch.linalg.cholesky(cov)
    yield "fitted_gaussian_full", (center + prior.double() @ cholesky.T).float()


def coordinate_conditions(posterior, prior, fit_mu, fit_lv, width, seed, metadata):
    # Gaussian max-entropy bound on I(source; z_j) for a mixture of diagonal
    # posteriors. This is a ranking proxy, not a direct MI measurement.
    total_var = fit_lv.exp().mean(0) + fit_mu.var(0, unbiased=False)
    info = .5 * (total_var.log() - fit_lv.mean(0))
    rank = info.argsort()
    if not 1 <= width <= len(rank) // 2:
        raise ValueError("Invalid coordinate intervention width")
    groups = {"high": rank[-width:], "low": rank[:width],
              "random": torch.randperm(len(rank), generator=torch.Generator().manual_seed(seed))[:width]}
    metadata["fitted_coordinate_information_proxy"] = info.tolist()
    metadata["groups"] = {k: v.tolist() for k, v in groups.items()}
    metadata["mean_squared_displacement"] = {}
    high_norm = (prior[:, groups["high"]] - posterior[:, groups["high"]]).norm(dim=1)
    for label, indices in groups.items():
        residual = prior[:, indices] - posterior[:, indices]
        replaced = posterior.clone()
        replaced[:, indices] = prior[:, indices]
        inserted = prior.clone()
        inserted[:, indices] = posterior[:, indices]
        metadata["mean_squared_displacement"][label] = residual.square().sum(1).mean().item()
        yield f"swap_{label}_to_prior", replaced
        yield f"insert_{label}_into_prior", inserted
        if label == "low":
            matched = posterior.clone()
            matched[:, indices] += residual * (high_norm / residual.norm(dim=1))[:, None]
            metadata["mean_squared_displacement"]["low_distance_matched"] = (
                matched - posterior).square().sum(1).mean().item()
            yield "swap_low_to_prior_high_distance", matched


def measure(model, vectors, source, indices, mode, stream, batch_size):
    if not torch.isfinite(vectors).all():
        raise ValueError(f"Nonfinite latent in {mode}")
    counts = Counter()
    failures = Counter()
    smiles_set = set()
    paired = mode == "encoded_mean" or mode.startswith(("prior_mix_", "swap_", "insert_"))
    for start in range(0, len(source), batch_size):
        output = model.decode(vectors[start:start + batch_size])
        if not all(torch.isfinite(v).all() for v in (output.atoms, output.bonds, output.charges)):
            raise ValueError(f"Nonfinite decoder output in {mode}")
        hard = output.harden(model.spec)
        for offset, row in enumerate(inspect_batch(hard, None, model.spec)):
            i = start + offset
            sm = row["smiles"]
            hit = motifs(sm) if sm else {}
            counts["draws"] += 1
            counts["valid"] += bool(sm)
            counts["aromatic"] += hit.get("aromatic_ring", False)
            counts["amide"] += hit.get("amide", False)
            counts["any_ring"] += hit.get("ring", False)
            counts["raw_aromatic_edge"] += row["aromatic_edges"] > 0
            counts["raw_aromatic_edge_cycle"] += row["aromatic_cycle"]
            counts["raw_any_cycle"] += row["cycle"]
            if sm:
                smiles_set.add(sm)
            else:
                failures[row["failure"]] += 1
            if paired:
                counts["exact_source"] += sm == source[i]
                for feature in ("aromatic_ring", "amide"):
                    positive = motifs(source[i])[feature]
                    counts[feature + "_source_positive"] += positive
                    counts[feature + "_retained"] += positive and hit.get(feature, False)
            stream.write(json.dumps(dict(condition=mode, draw=i, source_panel_index=indices[i] if paired else None,
                                         aromatic=hit.get("aromatic_ring", False), amide=hit.get("amide", False),
                                         **row)) + "\n")
    counts["unique_valid"] = len(smiles_set)
    return dict(**counts, failures=dict(failures), mean_latent_norm=vectors.norm(dim=1).mean().item())


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    panel = json.loads(args.panel.read_bytes())
    if rdBase.rdkitVersion != panel["rdkit"]:
        raise ValueError("RDKit version differs from the panel")
    spec = GraphSpec.from_dict(panel["spec"])
    model, payload = load_model(args.checkpoint, "cpu", spec)
    source = [row[2] for row in panel["rows"]]
    if len(set(source)) != len(source) or not 2 <= args.draws <= len(source) - 2:
        raise ValueError("Need distinct evaluation rows and unused fitting rows")
    rng = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(source), generator=rng)
    chosen, fit = order[:args.draws], order[args.draws:]
    mu, lv = encode_panel(model, source, args.batch)
    out = dict(seed=args.seed, draws=args.draws, fit_rows=len(fit), checkpoint_sha256=file_sha256(args.checkpoint),
               panel_sha256=file_sha256(args.panel), update=payload["continuation"]["updates"],
               torch_version=str(torch.__version__), rdkit_version=rdBase.rdkitVersion, conditions={})
    if args.mode == "coordinate-swap":
        if not 1 <= args.swap_draws <= len(chosen):
            raise ValueError("Invalid paired subset size")
        # Draw the entire path experiment's innovations before taking the subset,
        # so these rows have exactly the same posterior and prior as the path arm.
        posterior, prior = posterior_and_noise(mu[chosen], lv[chosen], rng)
        chosen = chosen[:args.swap_draws]
        out.update(draws=len(chosen), full_panel_draws=args.draws)
        conditions = coordinate_conditions(posterior[:len(chosen)], prior[:len(chosen)],
                                           mu[fit], lv[fit], args.swap_width, args.seed + 1, out)
    else:
        conditions = latent_conditions(mu[chosen], lv[chosen], mu[fit], lv[fit], rng)
    targets = [source[i] for i in chosen]
    args.output.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.output / "hard-graphs.jsonl.gz", "xt", encoding="utf8") as stream, rdBase.BlockLogs():
        for name, vectors in conditions:
            out["conditions"][name] = measure(model, vectors, targets, chosen.tolist(), name, stream, args.batch)
            (args.output / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
            print(name, out["conditions"][name], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026112131)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--mode", choices=("paths", "coordinate-swap"), default="paths")
    parser.add_argument("--swap-draws", type=int, default=96)
    parser.add_argument("--swap-width", type=int, default=32)
    run(parser.parse_args())
