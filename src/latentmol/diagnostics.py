"""Training-curve diagnostics and a declared, validation-only continuation rule.

These diagnostics can justify additional compute, but cannot prove that a model
has reached its best possible solution. No test-set quantity enters the decision.
"""

import math
from pathlib import Path
from statistics import mean


def continuation_decision(history, *, window=10, loss_threshold=.005,
                          validity_threshold=.01, exact_threshold=.005):
    """Compare adjacent equal windows on stable validation cohorts.

    Loss uses full validation reconstruction. Chemical fractions use the fixed
    validation probe, so occasional full-chemistry epochs do not change their
    denominator. This is a practical trend rule, not a significance test.
    """
    if window < 1 or min(loss_threshold, validity_threshold, exact_threshold) < 0:
        raise ValueError("Window must be positive and improvement thresholds nonnegative")
    if len(history) < 2 * window:
        return {"extend": True, "reason": "insufficient_history", "window": window}
    chosen = history[-2 * window:]
    epochs = [row["epoch"] for row in chosen]
    if epochs != list(range(epochs[0], epochs[0] + len(epochs))):
        raise ValueError("Convergence history must contain contiguous, unique epochs")
    before, after = chosen[:window], chosen[window:]
    def value(rows, section, key):
        values = [row[section][key] for row in rows]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"Missing/non-finite convergence metric: {section}.{key}")
        return mean(values)
    old_loss, new_loss = value(before, "val", "reconstruction"), value(after, "val", "reconstruction")
    old_valid, new_valid = value(before, "val_probe", "validity"), value(after, "val_probe", "validity")
    old_exact, new_exact = value(before, "val_probe", "exact_reconstruction"), value(after, "val_probe", "exact_reconstruction")
    gains = {"relative_reconstruction_improvement": (old_loss - new_loss) / max(abs(old_loss), 1e-12),
             "validity_fraction_increase": new_valid - old_valid,
             "exact_fraction_increase": new_exact - old_exact}
    thresholds = dict(zip(gains, (loss_threshold, validity_threshold, exact_threshold)))
    triggers = [key for key, gain in gains.items() if gain >= thresholds[key] and gain > 0]
    train_before = value(before, "train_probe", "reconstruction")
    train_after = value(after, "train_probe", "reconstruction")
    gap_before = value(before, "val_probe", "reconstruction") - train_before
    gap_after = value(after, "val_probe", "reconstruction") - train_after
    return {"extend": bool(triggers), "reason": "validation_still_improving" if triggers else "no_threshold_crossed",
            "window": window, "before_epochs": [before[0]["epoch"], before[-1]["epoch"]],
            "after_epochs": [after[0]["epoch"], after[-1]["epoch"]],
            "before": {"reconstruction": old_loss, "validity": old_valid, "exact": old_exact},
            "after": {"reconstruction": new_loss, "validity": new_valid, "exact": new_exact},
            "gains": gains, "thresholds": thresholds, "triggers": triggers,
            "matched_deterministic_generalization_gap_before": gap_before,
            "matched_deterministic_generalization_gap_after": gap_after,
            "overfitting_warning": train_after < train_before and gap_after > gap_before and not triggers,
            "interpretation": "Validation trend under the current schedule; not proof of convergence."}


def save_learning_curves(history, destination: Path):
    """Render the complete cumulative history on HPC with the Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    def series(axis, title, traces, logarithmic=False):
        for label, accessor in traces:
            values = [accessor(row) for row in history]
            axis.plot(epochs, [float("nan") if value is None else value for value in values], label=label, linewidth=1.6)
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=.25)
        if logarithmic:
            axis.set_yscale("log")
        axis.legend(fontsize=8)
    series(axes[0, 0], "Deterministic reconstruction loss", [
        ("train probe", lambda r: r["train_probe"]["reconstruction"]),
        ("validation probe", lambda r: r["val_probe"]["reconstruction"]),
        ("full validation", lambda r: r["val"]["reconstruction"])])
    series(axes[0, 1], "Unrepaired valid fraction (fixed cohorts)", [
        ("train", lambda r: r["train_probe"]["validity"]), ("validation", lambda r: r["val_probe"]["validity"])])
    series(axes[0, 2], "Exact reconstruction fraction (fixed cohorts)", [
        ("train", lambda r: r["train_probe"]["exact_reconstruction"]),
        ("validation", lambda r: r["val_probe"]["exact_reconstruction"])])
    series(axes[1, 0], "Validation loss components", [
        (name, lambda r, key=name: r["val"][key]) for name in ("atom_ce", "charge_ce", "bond_ce", "size_mse")], True)
    series(axes[1, 1], "Learning rate", [("LR", lambda r: r["learning_rate"])], True)
    series(axes[1, 2], "Gradient norms before clipping", [
        ("mean", lambda r: r["gradient_norm_mean_per_update"]),
        ("max", lambda r: r["gradient_norm_max"])], True)
    # Quantities with different units need independent, explicitly labelled axes;
    # otherwise an active-dimension count can visually flatten the KL trace.
    def paired_axes(axis, title, left_label, left_value, right_label, right_value):
        left_line, = axis.plot(epochs, [left_value(row) for row in history], color="C0", label=left_label)
        right = axis.twinx()
        right_line, = right.plot(epochs, [right_value(row) for row in history], color="C1", label=right_label)
        axis.set(title=title, xlabel="Epoch", ylabel=left_label)
        right.set_ylabel(right_label)
        axis.tick_params(axis="y", colors="C0")
        right.tick_params(axis="y", colors="C1")
        axis.grid(alpha=.25)
        axis.legend(handles=[left_line, right_line], fontsize=8)
        return right
    paired_axes(axes[2, 0], "Latent regularity and usage (AE KL unregularized)",
                "Posterior KL", lambda r: r["val_probe"]["latent_diagnostics"]["posterior_kl_mean"],
                "Active dimensions", lambda r: r["val_probe"]["latent_diagnostics"]["active_dimensions"])
    series(axes[2, 1], "Validation bond recovery", [
        ("precision", lambda r: r["val"]["bonded_pair_precision"]),
        ("recall", lambda r: r["val"]["bonded_pair_recall"])])
    clipped_axis = paired_axes(axes[2, 2], "Atom-count error and gradient clipping",
                "Atom-count MAE", lambda r: r["val"]["discrete_size_mae"],
                "Clipped fraction", lambda r: r["gradient_clipping_rate"])
    clipped_axis.set_ylim(0, 1)
    for axis in (axes[0, 1], axes[0, 2], axes[2, 1]):
        axis.set_ylim(0, 1)
    for axis in axes.flat:
        for boundary in (60, 80, 100):
            if epochs[-1] >= boundary:
                axis.axvline(boundary, color="gray", linestyle=":", alpha=.35)
    fig.suptitle("Graph reconstruction training diagnostics — validation controls continuation", fontsize=15)
    fig.savefig(destination / "learning_curves.png", dpi=160)
    fig.savefig(destination / "learning_curves.pdf")
    plt.close(fig)
