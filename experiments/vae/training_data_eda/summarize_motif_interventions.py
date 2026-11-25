"""Summarize source copies, unseen-in-panel edits and scaffold retention; plot the latent path."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.vae.motif_pilot import chemistry
from latentmol.experiments import write_json


def summarize(draws):
    by_mode = defaultdict(list)
    with gzip.open(draws, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            by_mode[row["mode"]].append(row)
    result = {}
    for mode, rows in by_mode.items():
        counts = Counter()
        unique, aromatic, amides, scaffolds, aromatic_scaffolds = (set() for _ in range(5))
        for row in rows:
            sm = row["smiles"]
            if not sm:
                continue
            mol = chemistry(sm)
            counts["valid"] += 1
            unique.add(sm)
            if mol["scaffold"]:
                scaffolds.add(mol["scaffold"])
            for key in ("aromatic", "amide"):
                counts[key] += row[key]
            if row["source"] == sm:
                counts["source_copies"] += 1
            if not row["in_panel"]:
                counts["valid_outside_panel"] += 1
                if row["aromatic"]:
                    aromatic.add(sm)
                    aromatic_scaffolds.add(mol["scaffold"])
                if row["amide"]:
                    amides.add(sm)
                if row["source"] and row["aromatic"] and mol["scaffold"] != chemistry(row["source"])["scaffold"]:
                    counts["outside_aromatic_changed_source_scaffold"] += 1
        result[mode] = dict(attempts=len(rows), **counts, unique_valid=len(unique),
            unique_scaffolds=len(scaffolds), unique_aromatics_outside_panel=len(aromatic),
            unique_aromatic_scaffolds_outside_panel=len(aromatic_scaffolds), unique_amides_outside_panel=len(amides))
    return result


def plot(paths, target):
    fractions = [0., .25, .5, .75, 1.]
    rows = [paths["conditions"][f"prior_mix_{t:.2f}"] for t in fractions]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
    for label, key, color, marker in (("Valid molecule", "valid", "#52616b", "o"),
            ("Valid with aromatic ring", "aromatic", "#105ba4", "s"),
            ("Valid with amide", "amide", "#a24a14", "^")):
        ax.plot(fractions, [100 * r[key] / r["draws"] for r in rows],
                color=color, marker=marker, linewidth=2.2, label=label)
    ax.set(xlim=(-.04, 1.04), ylim=(0, 100), yticks=range(0, 101, 20),
           xlabel="Fraction of independent Gaussian noise variance in the latent vector",
           ylabel="Percentage of all attempted decodes",
           title=f"Changing the latent input to the same frozen decoder ({rows[0]['draws']} draws per point)")
    ax.set_xticks(fractions, ["0%", "25%", "50%", "75%", "100%"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="center right", frameon=True)
    fig.savefig(target, dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--sampler", default="sampler-roundtrip")
    args = parser.parse_args()
    root = args.experiment
    dest = root / "overview"
    dest.mkdir(exist_ok=False)
    write_json(dest / "paired-sampler-counts.json", summarize(root / args.sampler / "draws.jsonl.gz"))
    plot(json.loads((root / "paths/summary.json").read_bytes()), dest / "latent-path.png")
