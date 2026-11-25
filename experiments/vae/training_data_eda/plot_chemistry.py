"""Standalone census and functional-group figures for the VAE training panel."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


TEAL = "#166f70"
RUST = "#b85535"
INK = "#203b46"
MUTED = "#4e656c"
GRID = "#dde6e6"
NAMES = {
    "amide": "Amide-type carbonyl", "carboxyl": "Carboxylic acid / carboxylate",
    "organohalogen": "Organohalogen", "quaternary_ammonium": "Quaternary ammonium",
}


def _name(key):
    return NAMES.get(key, key.replace("_", " ").capitalize())


def _axis(ax, axis):
    ax.spines[:].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, lw=.8)
    ax.tick_params(length=0, pad=6, colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _header(fig, title, subtitle):
    fig.text(.045, .95, title, weight="bold", fontsize=21, color=INK)
    fig.text(.046, .899, subtitle, fontsize=11.4, color=MUTED)


def _save(fig, path):
    try:
        with path.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=160, facecolor="white")
    finally:
        plt.close(fig)


def _size_figure(census, manifest):
    selected = {int(k): int(v) for k, v in census["size_counts"].items()}
    eligible = {int(k): int(v) for k, v in manifest["available_train_active_atom_counts"].items()}
    sizes = np.array(sorted(selected))
    n_selected = np.array([selected[n] for n in sizes])
    n_eligible = np.array([eligible[n] for n in sizes])
    ranks = np.zeros((len(sizes), 5), dtype=np.int64)
    index = {size: i for i, size in enumerate(sizes)}
    for cell in census["joint_counts"]:
        ranks[index[cell["heavy_atoms"]], min(cell["cycle_rank"], 4)] += cell["count"]
    if (not np.array_equal(ranks.sum(axis=1), n_selected)
            or n_selected.sum() != census["train_rows"] or n_eligible.sum() != census["parent_rows"]
            or np.any(n_eligible < n_selected)):
        raise ValueError("Census and parent size counts do not agree")

    fig, (ax, topo) = plt.subplots(1, 2, figsize=(13.8, 6.1))
    fig.subplots_adjust(left=.075, right=.978, bottom=.234, top=.772, wspace=.23)
    _header(fig, "Molecule size and ring topology",
            f"Exact training-view census  ·  {n_selected.sum():,} connected molecular graphs")
    ax.set_title("Which sizes entered the training view?", loc="left", pad=13)
    width = .91
    ax.bar(sizes, n_selected / 1e6, width, color=TEAL, label="Selected training graphs")
    ax.bar(sizes, (n_eligible - n_selected) / 1e6, width, bottom=n_selected / 1e6,
           color="#cbd8d8", label="Additional eligible parent graphs")
    ax.set(xlabel="Heavy atoms per molecule", ylabel="Molecules (millions)")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc="upper left", bbox_to_anchor=(-.012, -.21), ncol=2, fontsize=8.8,
              columnspacing=1, handlelength=1.4)
    topo.set_title("How does ring topology vary with size?", loc="left", pad=13)
    bases = np.zeros(len(sizes))
    for i, (label, color) in enumerate(zip(
        ["0", "1", "2", "3", "4+"], ["#ecdfcf", "#bcd9d5", "#67a9a5", "#22797d", "#21475f"],
    )):
        percentages = np.divide(100 * ranks[:, i], n_selected, out=np.zeros(len(sizes)), where=n_selected > 0)
        topo.bar(sizes, percentages, width=.996, bottom=bases, color=color, label=label)
        bases += percentages
    topo.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    topo.set(xlabel="Heavy atoms per molecule", ylabel="Share of training molecules at each size", ylim=(0, 100))
    topo.legend(title="Independent cycles per molecule", loc="upper left", bbox_to_anchor=(-.01, -.19),
                ncol=5, fontsize=9, title_fontsize=9.4, handlelength=1.2, columnspacing=1.25)
    for axis in (ax, topo):
        _axis(axis, "y")
        axis.set_xlim(sizes.min() - .7, sizes.max() + .7)
        ticks = np.arange(10, sizes.max() + 1, 10)
        axis.set_xticks(np.unique(np.r_[sizes.min(), ticks[ticks >= sizes.min()]]))
    fig.text(.046, .03,
             "Bar height includes all eligible parent graphs. The topology percentages sum to 100% within each size. "
             "For connected graphs, independent cycles = bonds − atoms + 1.", fontsize=9.3, color=MUTED)
    return fig


def _wilson(counts, total):
    """Two-sided 95% Wilson score interval in percentage points."""
    value = np.asarray(counts) / total
    z = 1.959963984540054
    denom = 1 + z**2 / total
    center = (value + z**2 / (2 * total)) / denom
    half = z * np.sqrt(value * (1 - value) / total + z**2 / (4 * total**2)) / denom
    return 100 * np.maximum(0, center - half), 100 * np.minimum(1, center + half)


def _labelled_bars(ax, labels, values, colors):
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, height=.59)
    limit = max(max(values, default=0) * 1.23, 2)
    ax.set(xlim=(0, limit), yticks=y, yticklabels=labels)
    ax.invert_yaxis()
    for pos, value in zip(y, values):
        ax.text(value + limit * .012, pos, f"{value:.1f}%", color=INK, fontsize=9, va="center")
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    _axis(ax, "x")


def _group_figure(directory):
    groups = pd.read_csv(directory / "ring_by_group.csv").sort_values("n_group", ascending=False)
    summary = pd.read_csv(directory / "ring_overview.csv").set_index("feature")
    pairs = pd.read_csv(directory / "group_pairs.csv").nlargest(5, "n_both")
    n = int(groups["n_all"].iloc[0])
    nr = int(groups["n_ring"].iloc[0])
    if not (n == int(summary.loc["any_ring", "n_all"]) and nr == int(summary.loc["any_ring", "n"])):
        raise ValueError("Inconsistent molecule denominators")
    fig = plt.figure(figsize=(14.6, 8.5))
    layout = fig.add_gridspec(2, 2, width_ratios=(1.1, 1), left=.187, right=.955,
                              bottom=.143, top=.802, hspace=.64, wspace=.69)
    ax = fig.add_subplot(layout[:, 0])
    pattern, pairplot = fig.add_subplot(layout[0, 1]), fig.add_subplot(layout[1, 1])
    _header(fig, "Functional groups in ring-bearing and acyclic molecules",
            f"Uniform random training sample  ·  {n:,} molecules  ·  {len(groups)} SMARTS-defined group families")
    ax.set_title("Group frequency within each subset", loc="left", pad=14)
    y = np.arange(len(groups))
    for counts, total, color, marker, label, dy in (
        (groups["n_with_ring"], nr, TEAL, "o", "Ring-bearing", -.12),
        (groups["n_group"] - groups["n_with_ring"], n - nr, RUST, "s", "Acyclic", .12),
    ):
        if total:
            lo, hi = _wilson(counts, total)
            ax.hlines(y + dy, lo, hi, color=color, linewidth=1.35)
            ax.scatter(100 * counts / total, y + dy, c=color, s=27, marker=marker,
                       label=f"{label} (n = {total:,})", zorder=3)
    ax.set(yticks=y, yticklabels=[_name(k) for k in groups["group"]], xlabel="Molecules in each subset")
    ax.invert_yaxis()
    ax.set_xlim(left=-.7, right=min(101, max(5, ax.get_xlim()[1])))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
    ax.legend(loc="upper left", bbox_to_anchor=(-.025, -.087), ncol=2, fontsize=9,
              handletextpad=.25, columnspacing=.95)
    _axis(ax, "x")
    keys = ["ring_and_any_listed_group", "ring_without_listed_group",
            "acyclic_with_listed_group", "acyclic_without_listed_group"]
    pattern.set_title("Ring and group combinations", loc="left", pad=12)
    _labelled_bars(pattern, ["Ring + group", "Ring only", "Group only", "Neither"],
                   summary.loc[keys, "pct_all"].tolist(), [TEAL, "#78a8a7", RUST, "#bcafa1"])
    pattern.set_xlabel("Share of the entire sample")
    pairplot.set_title("Most frequent group pairs", loc="left", pad=12)
    _labelled_bars(pairplot, [f"{_name(a)}\n+ {_name(b)}" for a, b in zip(pairs.group_a, pairs.group_b)],
                   pairs["pct_all_with_both"].tolist(), "#508c92")
    pairplot.tick_params(axis="y", labelsize=8.3)
    pairplot.set_xlabel("Share of the entire sample")
    fig.text(.046, .025,
             "Points show molecule prevalence with 95% Wilson intervals. Group presence refers to the 20 listed families. "
             "Groups can occur in a ring or on a side chain, and families can overlap.",
             fontsize=9.2, color=MUTED)
    return fig


def make_figures(tables_dir, census_path, manifest_record):
    """Create two new standalone PNGs next to the chemistry tables and return their paths."""
    directory = Path(tables_dir)
    census = json.loads(Path(census_path).read_bytes())
    manifest = json.loads(Path(manifest_record).read_bytes())["view"]["manifest"]
    paths = directory / "size_and_rings.png", directory / "groups_and_rings.png"
    if any(path.exists() for path in paths):
        raise FileExistsError("A chemistry figure already exists in the output directory")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                         "axes.titlesize": 12.6, "axes.titlecolor": INK, "legend.frameon": False}):
        _save(_size_figure(census, manifest), paths[0])
        _save(_group_figure(directory), paths[1])
    return paths
