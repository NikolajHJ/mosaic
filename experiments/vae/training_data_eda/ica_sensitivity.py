"""Repeat the descriptor ICA with deflation on the original two fitting draws."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from dimensions import _orient, _projection, _spectrum, _standardize, _top_correlates


def fit_deflation(pc, z, seed, fit_count, max_iter, tolerance):
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(pc), fit_count, replace=False)
    init_seed = int(rng.integers(2**31))
    fit = FastICA(algorithm="deflation", whiten="unit-variance", fun="logcosh",
                  random_state=init_seed, max_iter=max_iter, tol=tolerance)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit.fit(pc[rows])
    scores = _standardize(fit.transform(pc))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite deflation solution")
    kurtosis = np.mean(scores ** 4, axis=0) - 3
    order = np.argsort(-np.abs(kurtosis), kind="stable")
    scores = scores[:, order]
    scores, correlations = _orient(scores, z.T @ scores / len(scores))
    # sklearn's deflation solver returns the maximum iterations over axes.
    # It can exhaust the cap without emitting a ConvergenceWarning.
    info = dict(fit_rows=len(rows), init_seed=init_seed, max_axis_iterations=int(fit.n_iter_),
                converged_before_limit=bool(fit.n_iter_ < max_iter and not any(
                    issubclass(w.category, ConvergenceWarning) for w in caught)),
                warnings=[str(w.message) for w in caught],
                fit_positions_sha256=hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest())
    return scores, correlations, kurtosis[order], info


def sensitivity(directory):
    output = directory / "ica_sensitivity.json"
    image = directory / "ica_deflation.png"
    if output.exists() or image.exists():
        raise FileExistsError("An ICA sensitivity output already exists")
    original = json.loads((directory / "dimensions.json").read_bytes())
    names = original["varying_descriptors"]
    raw = pd.read_csv(directory / "profiles.csv.gz", usecols=names, float_precision="round_trip")[names]
    if len(raw) != original["n_rows"] or not np.isfinite(raw).all().all():
        raise ValueError("Invalid saved descriptor panel")
    z = _standardize(raw.to_numpy())
    values, axes, expl, _, _ = _spectrum(z.T @ z / len(z))
    k = original["ica"]["n_components"]
    pc, _ = _orient(z @ axes[:, :k], axes[:, :k] * np.sqrt(values[:k]))
    if not np.isclose(100 * expl[:k].sum(), original["ica"]["pca_subspace_variance_pct"]):
        raise ValueError("Input PCA subspace no longer matches")
    # Recover the original linear ICA scores using the descriptor-score cross
    # correlations saved at full precision. Every score lies in these same PCs.
    recorded = pd.read_csv(directory / "component_descriptor_correlations.csv").pivot(
        index="descriptor", columns="component", values="pearson_r")
    target = recorded.loc[names, [f"IC{i+1}" for i in range(k)]].to_numpy()
    basis = axes[:, :k]
    original_scores = (z @ basis / values[:k]) @ (basis.T @ target)
    if (not np.allclose(z.T @ original_scores / len(z), target, atol=1e-10) or
            not np.allclose(original_scores.std(axis=0), 1, atol=1e-10)):
        raise ValueError("Could not recover the published symmetric axes")
    seeds = np.random.SeedSequence(original["seed"]).spawn(3)
    orig_fit = original["ica"]["primary_fit"]
    fits = [fit_deflation(pc, z, seed, orig_fit["fit_rows"], 1500, orig_fit["tolerance"])
            for seed in seeds[:2]]
    primary, correlations, kurtosis, info = fits[0]
    repeat, _, _, repeated_info = fits[1]
    comparison = primary.T @ repeat / len(z)
    _, repeat_cols = linear_sum_assignment(abs(comparison), maximize=True)
    old_comparison = original_scores.T @ primary / len(z)
    _, primary_cols = linear_sum_assignment(abs(old_comparison), maximize=True)
    components = [dict(component=f"D{i+1}", excess_kurtosis=float(kurtosis[i]),
                       repeat_component=f"D{j+1}", matched_repeat_abs_r=float(abs(comparison[i,j])),
                       top_descriptor_correlations=_top_correlates(correlations[:, i], names))
                  for i, j in enumerate(repeat_cols)]
    old_matches = [dict(original=f"IC{i+1}", deflation=f"D{j+1}", r=float(old_comparison[i, j]))
                   for i, j in enumerate(primary_cols)]
    result = dict(method="Deflation FastICA, logcosh, unit variance, same leading descriptor PCs",
                  rows=len(z), seed=original["seed"], n_components=k, sklearn_version=sklearn_version,
                  tolerance=orig_fit["tolerance"], max_iter_per_axis=1500,
                  pca_subspace_variance_pct=float(expl[:k].sum() * 100),
                  ordering=original["ica"]["ordering"], main_fit=info, repeat_fit=repeated_info,
                  comparison="One-to-one maximum absolute Pearson correlation on all panel rows",
                  components=components, symmetric_to_deflation=old_matches)
    if all(fit["converged_before_limit"] for fit in (info, repeated_info)):
        fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(11, 4.6))
        for ax, first in zip(axs, (0, 2)):
            labels = []
            for i in (first, first + 1):
                descriptor = components[i]["top_descriptor_correlations"][0]["descriptor"]
                labels.append(f"D{i+1} · {descriptor.replace('_', ' ')} (positive correlation)")
            _projection(ax, primary[:, first:first+2], labels)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Deflation ICA of six descriptor PCs · {len(z):,} molecules", fontsize=15)
        fig.savefig(image, dpi=170)
        plt.close(fig)
        result["projection"] = image.name
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_directory", type=Path)
    with threadpool_limits(limits=1, user_api="blas"):
        print(json.dumps(sensitivity(parser.parse_args().analysis_directory)))
