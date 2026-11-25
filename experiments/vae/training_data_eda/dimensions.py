"""Descriptor PCA, a bounded FastICA view and optional binary fingerprint PCA."""

import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


THRESHOLDS = (50, 80, 90, 95, 99)


def _spectrum(covariance):
    eigenvalues, axes = np.linalg.eigh(covariance)
    eigenvalues, axes = eigenvalues[::-1], axes[:, ::-1]
    tolerance = np.finfo(float).eps * len(eigenvalues) * eigenvalues[0]
    if eigenvalues[-1] < -max(tolerance * 10, 1e-12):
        raise ValueError("Covariance has a materially negative eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0)
    if eigenvalues.sum() <= 0:
        raise ValueError("Input has zero total variance")
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    summary = {
        "n_dimensions": len(explained),
        "numerical_rank": int(np.sum(eigenvalues > tolerance)),
        "dimensions_for_variance_pct": {
            str(pct): int(np.searchsorted(cumulative, pct / 100) + 1)
            for pct in THRESHOLDS
        },
        "participation_ratio": float(1 / (explained @ explained)),
        "first_two_variance_pct": float(100 * explained[:2].sum()),
    }
    return eigenvalues, axes, explained, cumulative, summary


def _standardize(matrix):
    centered = matrix - matrix.mean(axis=0)
    scale = centered.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("Cannot standardize a constant component")
    return centered / scale


def _ica_view(pc_scores, rng, max_fit=12000, max_iter=500, tol=1e-4):
    fit_rows = rng.choice(len(pc_scores), min(len(pc_scores), max_fit), replace=False)
    fit = pc_scores[fit_rows]
    if np.linalg.matrix_rank(fit - fit.mean(axis=0)) != fit.shape[1]:
        raise ValueError("ICA fitting panel has insufficient rank in its PCA subspace")
    model = FastICA(whiten="unit-variance", max_iter=max_iter, tol=tol,
                    random_state=int(rng.integers(2**31)))
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1, user_api="blas"):
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(fit)
    scores = model.transform(pc_scores)
    if not np.isfinite(scores).all():
        raise ValueError("ICA yielded nonfinite component scores")
    return _standardize(scores), {
        "fit_rows": len(fit), "iterations": int(model.n_iter_), "max_iterations": max_iter,
        "tolerance": tol, "converged": not any(
            issubclass(w.category, ConvergenceWarning) for w in caught),
        "warnings": [str(w.message) for w in caught],
    }


def _top_correlates(correlations, names, count=4):
    order = np.argsort(-np.abs(correlations), kind="stable")[:count]
    return [{"descriptor": names[ix], "r": float(correlations[ix])} for ix in order]


def _orient(scores, correlations):
    major = np.argmax(np.abs(correlations), axis=0)
    signs = np.sign(correlations[major, np.arange(len(major))])
    signs[signs == 0] = 1
    return scores * signs, correlations * signs


def _fingerprint_spectrum(fingerprints, rng):
    bits = np.asarray(fingerprints)
    if bits.ndim != 2 or min(bits.shape) < 2:
        raise ValueError("Fingerprints must be an N by bits matrix")
    original_rows, width = bits.shape
    if len(bits) > 20000:
        bits = bits[rng.choice(len(bits), 20000, replace=False)]
    if not np.isin(bits, [0, 1]).all():
        raise ValueError("Fingerprint features must be finite binary indicators")
    varying = bits.min(axis=0) != bits.max(axis=0)
    if not np.any(varying):
        raise ValueError("The fingerprint panel is constant")
    centered = bits[:, varying].astype(float)
    centered -= centered.mean(axis=0)
    _, _, explained, cumulative, summary = _spectrum(centered.T @ centered / len(bits))
    summary.update({
        "n_rows": len(bits), "input_rows": original_rows, "fingerprint_bits": width,
        "constant_bits": int(width - varying.sum()),
        "scaling": "mean-centered binary bit indicators, no per-bit variance scaling",
    })
    return explained, cumulative, summary


def _projection(ax, scores, labels):
    if scores.shape[1] < 2:
        ax.hist(scores[:, 0], bins="auto", color="#137365")
        ax.set(xlabel=labels[0], ylabel="Molecules")
        return
    lower, upper = np.quantile(scores[:, :2], [.001, .999], axis=0)
    display = np.all((scores[:, :2] >= lower) & (scores[:, :2] <= upper), axis=1)
    density = ax.hexbin(scores[display, 0], scores[display, 1], gridsize=62, mincnt=1,
                        bins="log", cmap="viridis", linewidths=0)
    bar = ax.figure.colorbar(density, ax=ax, shrink=.78, pad=.02)
    bar.set_label("Graphs per hexagon")
    if not display.all():
        ax.text(.98, .98, f"{100 * display.mean():.1f}% in view", ha="right", va="top",
                fontsize=8, transform=ax.transAxes)
    ax.set(xlabel=labels[0], ylabel=labels[1])


def _overview(path, pc, ic, desc_curve, fp_curve, explained, n, fp_rows):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2), layout="constrained")
    fig.suptitle("Molecular structure in the VAE training data", fontsize=17)
    _projection(axs[0, 0], pc, [f"PC{i+1} ({100 * v:.1f}% variance)"
                                for i, v in enumerate(explained[:2])])
    axs[0, 0].set_title(f"Descriptor PCA · {n:,} graphs")
    if ic is not None:
        _projection(axs[0, 1], ic, [f"IC{i+1} (SD)" for i in range(ic.shape[1])])
        axs[0, 1].set_title("Independent components · ICA")
    else:
        axs[0, 1].axis("off")
    curves = [(axs[1, 0], desc_curve, "Descriptor variance captured")]
    if fp_curve is not None:
        curves.append((axs[1, 1], fp_curve, f"Fingerprint variance · {fp_rows:,} graphs"))
    else:
        ax = axs[1, 1]
        ax.bar(range(1, len(explained) + 1), explained, color="#2c7480")
        ax.set(title="Variance contributed by each PC", xlabel="Principal component",
               ylabel="Share of variance")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    for ax, cumulative, title in curves:
        xs = np.arange(len(cumulative) + 1)
        ax.plot(xs, np.r_[0, cumulative], color="#00715e", lw=2.5)
        for pct, offset in [(90, -13), (95, 8)]:
            k = int(np.searchsorted(cumulative, pct / 100) + 1)
            ax.scatter(k, cumulative[k-1], s=26, color="#203e51", zorder=3)
            ax.annotate(f"{pct}% · {k} PCs", (k, cumulative[k-1]),
                        xytext=(5, offset), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Number of principal components", ylabel="Cumulative variance",
               title=title, ylim=(0, 1.1), xlim=(0, max(xs) * 1.12))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(axis="y", alpha=.18)
    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=175, facecolor="white")
    plt.close(fig)


def _correlation_plot(path, names, labels, corrs):
    fig, ax = plt.subplots(figsize=(7.8, max(4.2, .33 * len(names) + 1.8)),
                           layout="constrained")
    image = ax.imshow(corrs, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    ax.set_title("Descriptor correlations with component scores", pad=12)
    bar = fig.colorbar(image, ax=ax, shrink=.72)
    bar.set_label("Pearson correlation")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def analyze_dimensions(descriptors, feature_names, output_dir, seed=42, fingerprints=None):
    """Fit PCA to standardized descriptor columns and FastICA to up to six leading PCs.

    Prevalence-weighted fingerprint PCA centers raw binary bits without scaling.
    Its panel may be a separate random sample of training rows.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    products = ("dimensions.json", "pca_variance.csv", "component_descriptor_correlations.csv",
                "dimension_overview.png", "component_correlations.png")
    for filename in products:
        if (out / filename).exists():
            raise FileExistsError(out / filename)
    raw = np.asarray(descriptors, dtype=float)
    all_names = [str(name) for name in feature_names]
    if (raw.ndim != 2 or raw.shape[1] != len(all_names) or len(raw) < 3 or
            len(set(all_names)) != len(all_names) or not np.isfinite(raw).all()):
        raise ValueError("Expected finite numeric descriptors, at least 3 rows and unique column labels")
    varying = raw.min(axis=0) != raw.max(axis=0)
    if not np.any(varying):
        raise ValueError("No varying descriptors")
    names = [name for name, keep in zip(all_names, varying) if keep]
    z = _standardize(raw[:, varying])
    values, axes, explained, cumulative, pca = _spectrum(z.T @ z / len(z))
    rank = pca["numerical_rank"]
    pc_scores = z @ axes[:, :rank]
    # X is standardized and PC scores are orthogonal, so corr(X_j, PC_k)
    # equals axis[j, k] * sqrt(eigenvalue[k]).
    pc_scores, pc_corr = _orient(pc_scores, axes[:, :rank] * np.sqrt(values[:rank]))
    pca["components"] = [
        {"component": f"PC{i+1}", "variance_pct": float(100 * explained[i]),
         "top_descriptor_correlations": _top_correlates(pc_corr[:, i], names)}
        for i in range(rank)
    ]
    summary = {
        "n_rows": len(z), "input_descriptors": all_names, "varying_descriptors": names,
        "constant_descriptors": [s for s, keep in zip(all_names, varying) if not keep],
        "seed": int(seed), "pca": pca,
    }
    k = min(6, rank)
    ic_scores = None
    corr_tables = [("PC", pc_corr)]
    seed_sequences = np.random.SeedSequence(seed).spawn(3)
    if k >= 2:
        scores = pc_scores[:, :k]
        ic_scores, primary = _ica_view(scores, np.random.default_rng(seed_sequences[0]))
        repeat_scores, repeat = _ica_view(scores, np.random.default_rng(seed_sequences[1]))
        # ICA has arbitrary permutation and sign. Order by absolute excess kurtosis,
        # then orient so that the strongest measured descriptor correlation is positive.
        excess = np.mean(ic_scores ** 4, axis=0) - 3
        order = np.argsort(-np.abs(excess), kind="stable")
        ic_scores, excess = ic_scores[:, order], excess[order]
        ic_scores, ic_corr = _orient(ic_scores, z.T @ ic_scores / len(z))
        repeat_corr = np.abs(ic_scores.T @ repeat_scores / len(z))
        matched_rows, matched_cols = linear_sum_assignment(repeat_corr, maximize=True)
        stability = np.clip(repeat_corr[matched_rows, matched_cols], 0, 1)
        corr_tables.append(("IC", ic_corr))
        summary["ica"] = {
            "method": "symmetric FastICA with tanh contrast on leading descriptor PCs",
            "n_components": k, "pca_subspace_variance_pct": float(100 * cumulative[k-1]),
            "ordering": "descending absolute excess kurtosis, strongest descriptor correlation positive",
            "repeat": "independent seeded fitting-row draws and initialization; compared on all input rows",
            "primary_fit": primary, "repeat_fit": repeat,
            "components": [
                {"component": f"IC{i+1}", "excess_kurtosis": float(excess[i]),
                 "repeat_absolute_r": float(stability[i]),
                 "top_descriptor_correlations": _top_correlates(ic_corr[:, i], names)}
                for i in range(k)
            ],
        }
    else:
        summary["ica"] = {"n_components": 0, "reason": "Only one nondegenerate descriptor direction"}

    spectra = [("standardized_descriptors", explained, cumulative)]
    fp_curve = None
    if fingerprints is not None:
        fp_explained, fp_curve, fp_summary = _fingerprint_spectrum(
            fingerprints, np.random.default_rng(seed_sequences[2]))
        summary["fingerprint_pca"] = fp_summary
        spectra.append(("binary_fingerprint", fp_explained, fp_curve))
    with (out / "pca_variance.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["representation", "component", "variance_share", "cumulative_share"])
        for representation, fractions, curve in spectra:
            writer.writerows((representation, i, v, c)
                             for i, (v, c) in enumerate(zip(fractions, curve), 1))
    with (out / "component_descriptor_correlations.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["component", "descriptor", "pearson_r"])
        for prefix, correlations in corr_tables:
            for i, column in enumerate(correlations.T, 1):
                writer.writerows((f"{prefix}{i}", name, r) for name, r in zip(names, column))
    _overview(out / "dimension_overview.png", pc_scores, ic_scores, cumulative,
              fp_curve, explained, len(z), summary.get("fingerprint_pca", {}).get("n_rows"))
    labels, matrices = [], []
    for prefix, correlations in corr_tables:
        matrix = correlations[:, :4] if prefix == "PC" else correlations
        labels.extend(f"{prefix}{i+1}" for i in range(matrix.shape[1]))
        matrices.append(matrix)
    _correlation_plot(out / "component_correlations.png", names, labels, np.column_stack(matrices))
    with (out / "dimensions.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    return summary
