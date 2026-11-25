"""Test changes to the latent sampler against one frozen VAE using a uniform training panel."""

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
import torch

from experiments.vae.motif_pilot import chemistry
from latentmol.chemistry import encode_canonical_smiles, inspect_graph
from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.train import state_digest


def encode(model, molecules, batch, device):
    means, variances = [], []
    for start in range(0, len(molecules), batch):
        graphs = []
        for smiles in molecules[start:start + batch]:
            record = encode_canonical_smiles(smiles, model.spec)
            graphs.append({k: torch.as_tensor(record[src]).long() for k, src in
                           (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})
        mu, lv = model.encoder(collate_graphs(graphs).to(device), from_canonical_cache=True)
        means.append(mu.cpu().double().numpy())
        variances.append(lv.exp().cpu().double().numpy())
    means, variances = np.concatenate(means), np.concatenate(variances)
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or np.min(variances) <= 0:
        raise ValueError("Invalid encoded posterior")
    return means, variances


def sampling_arms(mu, var, centers, labels, source_ix, noise):
    # Pooled within-cluster covariance makes even small clusters full-rank. Adding the
    # posterior variances includes variability within each encoded source molecule.
    mean = mu.mean(axis=0)
    within = mu - centers[labels]
    cov = (mu - mean).T @ (mu - mean) / len(mu) + np.diag(var.mean(axis=0))
    pooled = within.T @ within / len(mu) + np.diag(var.mean(axis=0))
    yield "standard", noise, False
    yield "center_only", mean + noise, False
    yield "diagonal", mean + cov.diagonal() ** .5 * noise, False
    yield "full", mean + noise @ np.linalg.cholesky(cov).T, False
    yield "mixture32", centers[labels[source_ix]] + noise @ np.linalg.cholesky(pooled).T, False
    for scale in (1., 1.5, 2.):
        yield f"posterior_noise_{scale:.1f}", mu[source_ix] + scale * np.sqrt(var[source_ix]) * noise, True


def fit_clusters(mu, rng, count=32):
    """Lloyd's algorithm with k-means++ seeds. Each source votes once."""
    centers = [mu[rng.integers(len(mu))]]
    nearest = np.square(mu - centers[0]).sum(axis=1)
    while len(centers) < count:
        centers.append(mu[rng.choice(len(mu), p=nearest / nearest.sum())])
        nearest = np.minimum(nearest, np.square(mu - centers[-1]).sum(axis=1))
    centers = np.stack(centers)
    fixed = np.square(mu).sum(axis=1)[:, None]
    for _ in range(25):
        distances = fixed + np.square(centers).sum(axis=1) - 2 * (mu @ centers.T)
        labels = distances.argmin(axis=1)
        if np.any(np.bincount(labels, minlength=count) == 0):
            raise ValueError("Empty component in the fitted latent mixture")
        updated = np.stack([mu[labels == k].mean(axis=0) for k in range(count)])
        if np.array_equal(centers, updated):
            break
        centers = updated
    return centers, labels


def evaluate(model, vectors, seed_smiles, panel_smiles, batch, device, stream, mode, replicate):
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite sampled latent vectors")
    counts = Counter()
    unique, scaffolds, unseen_aromatics, aromatic_scaffolds = set(), set(), set(), set()
    for start in range(0, len(vectors), batch):
        decoded = model.decode(torch.as_tensor(vectors[start:start + batch], dtype=torch.float32, device=device))
        if not all(torch.isfinite(x).all() for x in (decoded.atoms, decoded.bonds, decoded.charges)):
            raise ValueError("Nonfinite decoder output")
        hard = decoded.harden(model.spec).to("cpu")
        graphs = zip(hard.atoms.numpy(), hard.charges.numpy(), hard.bonds.numpy(), strict=True)
        for j, (a, c, b) in enumerate(graphs, start):
            check = inspect_graph(a, c, b, model.spec)
            smiles, failure = check.smiles, check.failure_reason
            # Rare sanitized graphs serialize into a canonical string that RDKit cannot
            # read back. Keep that string and raw labels for audit, and score no motif.
            warning = {}
            if smiles and Chem.MolFromSmiles(smiles) is None:
                warning = dict(emitted_smiles=smiles, atoms=a.tolist(), charges=c.tolist(),
                               edges=[[int(i), int(k), int(b[i,k])] for i,k in zip(*np.triu(b, 1).nonzero())])
                smiles, failure = None, "emitted_smiles_failed_reparse"
                counts[failure] += 1
            result = chemistry(smiles)
            source = seed_smiles[j] if seed_smiles is not None else None
            present = smiles in panel_smiles
            for key in ("valid", "aromatic", "amide"):
                counts[key] += int(result[key])
                counts[key + "_outside_panel"] += int(result[key] and not present)
            if smiles:
                unique.add(smiles)
                if result["scaffold"]:
                    scaffolds.add(result["scaffold"])
                if result["aromatic"] and not present:
                    unseen_aromatics.add(smiles)
                    aromatic_scaffolds.add(result["scaffold"])
            counts["exact_source"] += int(source is not None and source == smiles)
            stream.write(json.dumps(dict(mode=mode, replicate=replicate, draw=j, source=source,
                smiles=smiles, failure=failure, in_panel=present,
                aromatic=result["aromatic"], amide=result["amide"], **warning)) + "\n")
    return dict(counts, attempts=len(vectors), unique_valid=len(unique), unique_scaffolds=len(scaffolds),
                unique_aromatic_outside_panel=len(unseen_aromatics),
                aromatic_scaffolds_outside_panel=len(aromatic_scaffolds))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fit", type=int, default=2048)
    p.add_argument("--draws", type=int, default=256, help="Attempts per arm per replicate")
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--seed", type=int, default=2026112153)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model, checkpoint = load_model(args.checkpoint, args.device)
    model.requires_grad_(False)
    if model.training:
        raise ValueError("Expected evaluation mode")
    before = state_digest(model.state_dict())
    with gzip.open(args.panel, "rt", encoding="utf-8", newline="") as stream:
        all_smiles = [row["smiles"] for row in csv.DictReader(stream, delimiter="\t")]
    panel_set = set(all_smiles)
    if len(panel_set) != len(all_smiles) or not 32 < args.fit <= len(all_smiles):
        raise ValueError("Expected a unique training panel with enough fit rows")
    rng = np.random.default_rng(args.seed)
    train_smiles = [all_smiles[i] for i in rng.choice(len(all_smiles), args.fit, replace=False)]
    mu, var = encode(model, train_smiles, args.batch, args.device)
    centers, labels = fit_clusters(mu, np.random.default_rng(args.seed))
    report = dict(checkpoint_sha256=file_sha256(args.checkpoint), update=checkpoint["continuation"]["updates"],
        panel_sha256=file_sha256(args.panel), panel=len(all_smiles), fit=len(mu), seed=args.seed,
        rdkit=rdBase.rdkitVersion, torch=str(torch.__version__), device=args.device,
        cluster_sizes=np.bincount(labels).tolist(), replicates=[])
    with gzip.open(args.out / "draws.jsonl.gz", "xt", encoding="utf-8") as stream, rdBase.BlockLogs():
        for rep in range(args.replicates):
            source_ix = rng.choice(len(mu), args.draws, replace=False)
            noise = rng.normal(size=(args.draws, mu.shape[1]))
            arm_results = {}
            for name, z, seeded in sampling_arms(mu, var, centers, labels, source_ix, noise):
                seeds = [train_smiles[i] for i in source_ix] if seeded else None
                scores = evaluate(model, z, seeds, panel_set, args.batch, args.device, stream, name, rep)
                arm_results[name] = scores
                print(rep, name, scores["valid"], scores["aromatic"], scores["amide"], flush=True)
            report["replicates"].append(arm_results)
    if state_digest(model.state_dict()) != before:
        raise RuntimeError("Frozen model changed")
    write_json(args.out / "summary.json", report)


if __name__ == "__main__":
    main()
