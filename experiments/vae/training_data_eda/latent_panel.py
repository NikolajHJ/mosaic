"""PCA of frozen posterior means on a reproducible subsample of the training panel."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch

from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.data import collate_graphs
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.train import state_digest


CHECKPOINT_DIGEST = "408b6904281fbe5d9f92e8b40c62e301d099b0fdad87109ef0ca5d90ce172922"
DATA_DIGEST = "64247d04ec1f94d00320bb1e5a0d9c5540419eb37830050d3100775bf4385e18"
PCTS = (50, 80, 90, 95, 99)


def eigensystem(x):
    centered = x - x.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / (len(x) - 1))
    if not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0 or eigenvalues[0] < -1e-10:
        raise ValueError("Invalid posterior-mean covariance")
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    return eigenvalues, vectors[:, ::-1], centered


def thresholds(curve):
    return {str(q): int(np.searchsorted(curve, q / 100) + 1) for q in PCTS}


def analyze(panel, view, checkpoint, output, count, seed, batch_size, threads):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    if not 1 <= batch_size <= count or threads < 1:
        raise ValueError("Invalid batch size or thread count")
    if file_sha256(checkpoint) != CHECKPOINT_DIGEST:
        raise ValueError("Checkpoint digest differs from the partial-47 checkpoint")
    torch.set_num_threads(threads)
    corpus = BalancedSparseCorpus(view, expected_sha256=DATA_DIGEST)
    dataset = corpus.split("train")
    with gzip.open(panel, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    ids = np.array([int(row["view_row"]) for row in rows])
    if len(ids) < count or np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= len(dataset):
        raise ValueError("Expected sorted, unique in-range training rows")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(ids), count, replace=False))
    selected = ids[chosen]
    n_verified = verify_canonical_training_cache(dataset, selected)
    model, payload = load_model(checkpoint, "cpu", dataset.spec)
    if (model.config.encoder_pooling != "canonical" or payload["dataset_sha256"] != DATA_DIGEST
            or payload["continuation"]["updates"] != 524177 or model.training):
        raise ValueError("Incompatible checkpoint or encoder mode")
    model.requires_grad_(False)
    weights_sha = state_digest(model.state_dict())
    del payload
    prep_seconds = time.monotonic() - started
    mus, logvars = [], []
    with torch.inference_mode():
        for offset in range(0, len(chosen), batch_size):
            positions = chosen[offset:offset+batch_size]
            graphs = []
            for ix in positions:
                view_row = int(ids[ix])
                row = rows[ix]
                graph = dataset[view_row]
                if (dataset.smiles[view_row] != row["smiles"] or
                        int(graph["atoms"].count_nonzero()) != int(row["heavy_atoms"])):
                    raise ValueError(f"Panel and cached labels differ at view row {view_row}")
                graphs.append(graph)
            mu, lv = model.encoder(collate_graphs(graphs), from_canonical_cache=True)
            mus.append(mu.numpy())
            logvars.append(lv.numpy())
    x, lv = np.concatenate(mus).astype(float), np.concatenate(logvars).astype(float)
    if (x.shape != (count, model.config.latent) or x.shape != lv.shape or
            not np.isfinite(x).all() or not np.isfinite(lv).all() or
            state_digest(model.state_dict()) != weights_sha):
        raise ValueError("Invalid posterior outputs or model mutation")
    encode_seconds = time.monotonic() - started - prep_seconds
    values, axes, centered = eigensystem(x)
    fractions = values / values.sum()
    curve = np.cumsum(fractions)
    scores = centered @ axes[:, :2]
    sizes = np.array([int(rows[i]["heavy_atoms"]) for i in chosen])
    ranks = np.array([int(rows[i]["cycle_rank"]) for i in chosen])
    for i in range(2):
        if np.corrcoef(scores[:, i], sizes)[0, 1] < 0:
            scores[:, i] *= -1
    # Estimate generalization across this same panel distribution. Fit axes on
    # one random half, then measure test variance along those fixed directions.
    fit, test = np.array_split(rng.permutation(count), 2)
    _, fitted_axes, _ = eigensystem(x[fit])
    test_scores = (x[test] - x[test].mean(axis=0)) @ fitted_axes
    test_v = test_scores.var(axis=0, ddof=1)
    test_curve = np.cumsum(test_v / test_v.sum())
    summary = {
        "n_rows": count, "n_dimensions": x.shape[1], "panel_rows": len(rows),
        "seed": seed, "batch_size": batch_size, "cpu_threads": threads,
        "method": "PCA of centered posterior means in native latent units, no coordinate scaling",
        "panel_sha256": file_sha256(panel), "checkpoint_sha256": CHECKPOINT_DIGEST,
        "dataset_sha256": DATA_DIGEST, "encoder_update": 524177, "checkpoint_epoch":
        "partial epoch 47 after 47,682,960 training rows", "model_state_sha256": weights_sha,
        "sampling": "uniform panel positions without replacement using numpy PCG64",
        "all_labels_from_training_cache": True, "canonical_inputs_verified": n_verified,
        "ordered_view_rows_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "job_id": os.environ.get("LSB_JOBID"), "first_two_variance_pct": float(100*curve[1]),
        "dimensions_for_variance_pct": thresholds(curve),
        "participation_ratio": float(1 / np.sum(fractions**2)),
        "half_split_fit_test_sizes": [len(fit), len(test)],
        "heldout_dimensions_for_variance_pct": thresholds(test_curve),
        "heavy_atom_count_pearson_r": [float(np.corrcoef(col, sizes)[0, 1]) for col in scores.T],
        "cycle_rank_pearson_r": [float(np.corrcoef(col, ranks)[0, 1]) for col in scores.T],
        "mean_posterior_variance_per_coordinate": float(np.exp(lv).mean()),
        "mean_between_molecule_variance_per_coordinate": float(values.mean()),
        "prep_seconds": round(prep_seconds, 2), "encode_and_verify_seconds": round(encode_seconds, 2),
    }
    with (output / "latent_pca_variance.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "variance_share", "cumulative_share", "heldout_cumulative_share"])
        writer.writerows((i+1, f, c, t) for i, (f, c, t) in enumerate(zip(fractions, curve, test_curve)))
    with (output / "latent_pc_scores.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["view_row", "heavy_atoms", "cycle_rank", "pc1", "pc2"])
        writer.writerows(zip(selected, sizes, ranks, scores[:, 0], scores[:, 1]))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    fig.suptitle(f"Frozen VAE posterior means · {count:,} training molecules", fontsize=15)
    draw = rng.permutation(count)
    points = ax.scatter(*scores[draw].T, c=sizes[draw], cmap="viridis", s=6, alpha=.72, linewidths=0)
    fig.colorbar(points, ax=ax, label="Heavy atoms")
    ax.set(xlabel=f"PC1 ({100*fractions[0]:.1f}%)", ylabel=f"PC2 ({100*fractions[1]:.1f}%)",
           title="Molecules projected onto the first two PCs")
    xs = np.arange(1, x.shape[1] + 1)
    bx.plot(xs, curve, color="#08705e", label="Full panel")
    bx.plot(xs, test_curve, "--", color="#bd713c", label="Held-out half")
    bx.set(xlabel="Principal components", ylabel="Cumulative variance",
           ylim=(0, 1.04), title="Variance in native latent coordinates")
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    bx.legend(frameon=False, loc="lower right")
    bx.grid(axis="y", alpha=.2)
    fig.savefig(output / "latent_pca.png", dpi=160)
    plt.close(fig)
    summary["total_seconds"] = round(time.monotonic() - started, 2)
    (output / "latent_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "view", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026112003)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    analyze(**vars(parser.parse_args()))
