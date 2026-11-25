"""Render a molecule-level motif comparison and export every denominator."""

import argparse
import csv
from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw


def prior_composition(directory, report):
    elements = report["spec"]["atomic_numbers"]
    by_checkpoint, examples = {}, []
    ordered = sorted(report["checkpoints"], key=lambda key: report["checkpoints"][key]["cursor"]["updates"])
    valid_smiles = []
    for key in ordered:
        by_checkpoint[key] = {}
        with gzip.open(directory / f"{key}.jsonl.gz", "rt") as src:
            rows = [r for line in src if (r := json.loads(line))["panel"] == "prior_argmax"]
        for population, selected in (("all", rows), ("valid", [r for r in rows if r["smiles"]])):
            comps = [Counter(elements[node[1]] for node in r["nodes"]) for r in selected]
            fluorines = [c[9] for c in comps]
            by_checkpoint[key][population] = dict(n=len(comps), carbon_and_fluorine_only=sum(set(c)=={6,9} for c in comps),
                any_nitrogen=sum(c[7]>0 for c in comps), any_oxygen=sum(c[8]>0 for c in comps),
                fluorine_median=float(np.median(fluorines)) if comps else None,
                fluorine_range=[min(fluorines),max(fluorines)] if comps else None)
        if key in (ordered[0],ordered[-1]):
            valid_smiles.append({r["index"]:r["smiles"] for r in rows if r["smiles"]})
    (directory/"composition.json").write_text(json.dumps(by_checkpoint,indent=2)+"\n",encoding="utf-8")
    if len(valid_smiles)==2:
        first, last = valid_smiles
        labels=[]
        for idx in sorted(first.keys() & last.keys())[:2]:
            examples.extend([Chem.MolFromSmiles(first[idx]), Chem.MolFromSmiles(last[idx])])
            labels.extend([f"Vector {idx} | epoch 46", f"Vector {idx} | partial epoch 47"])
        if examples:
            grid = Draw.MolsToGridImage(examples,molsPerRow=2,subImgSize=(460,290),legends=labels,returnPNG=True)
            (directory/"matched-prior-examples.png").write_bytes(grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    directory = args.summary.parent
    report = json.loads(args.summary.read_bytes())
    ckpts = sorted(report["checkpoints"].values(), key=lambda c: c["cursor"]["updates"])
    training = report["training"]
    prior_composition(directory, report)
    with (directory/"frequencies.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model_update","population","motif","positive","valid","attempted",
                         "pct_of_valid","pct_of_attempts","size_matched_training_pct"])
        for motif, n in training.items():
            if motif != "count":
                writer.writerow(["","training",motif,n,training["count"],training["count"],
                                 100*n/training["count"],100*n/training["count"],""])
        for c in ckpts:
            for pop, v in c["populations"].items():
                for motif, n in v["valid_with_motif"].items():
                    expected = v["expected_counts_given_valid_sizes"][motif]
                    writer.writerow([c["cursor"]["updates"],pop,motif,n,v["valid"],v["draws"],
                                     100*n/v["valid"] if v["valid"] else "",100*n/v["draws"],
                                     100*expected/v["valid"] if v["valid"] else ""])
    first = ckpts[0]["populations"]["prior_argmax"]
    last = ckpts[-1]["populations"]["prior_argmax"]
    rec = ckpts[-1]["populations"]["reconstruction"]
    cols = [training, rec["valid_with_motif"], first["valid_with_motif"], last["valid_with_motif"]]
    den = [training["count"], rec["valid"], first["valid"], last["valid"]]
    motifs = ["ring","aromatic_ring","heterocycle","amide","amine","organohalogen","ether","alkene","ester","alcohol","nitrile"]
    names = ["Any ring","Aromatic ring","Heterocycle","Amide-type carbonyl","Amine","Carbon-bound halogen",
             "Ether","Alkene","Ester","Alcohol","Nitrile"]
    mat = np.array([[100*c[k]/n if n else np.nan for c,n in zip(cols,den)] for k in motifs])
    fig, ax = plt.subplots(figsize=(10.2, 6.8))
    fig.subplots_adjust(left=.25, right=.97, top=.75, bottom=.11)
    cmap=LinearSegmentedColormap.from_list("occurrence",["#f3f6f5","#b4d9d4","#559e9d","#13575d"])
    ax.imshow(mat, vmin=0, vmax=100, cmap=cmap, aspect="auto")
    for i,j in np.ndindex(mat.shape):
        p=mat[i,j]
        ax.text(j,i,f"{p:.1f}%",ha="center",va="center",color="white" if p>68 else "#162f38")
    ax.set_yticks(range(len(names)),names)
    ax.set_xticks(range(len(den)),[f"Real training\nn = {den[0]:,}",f"Reconstruction\npartial 47 · n = {den[1]}",
                                  f"Random prior\nepoch 46 · n = {den[2]:,}",f"Random prior\npartial 47 · n = {den[3]}"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0,pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.04,.95,"Which chemical groups appear in valid VAE outputs?",fontsize=19,weight="bold",color="#16383d")
    fig.text(.04,.895,"Each cell is the percentage of valid molecules containing the group. A molecule can count in several rows.",
             fontsize=10.4,color="#41545c")
    fig.text(.04,.045,f"From {last['draws']:,} random draws, epoch 46 yielded {first['valid_with_motif']['aromatic_ring']} valid aromatic molecules "
             f"and partial epoch 47 yielded {last['valid_with_motif']['aromatic_ring']}. Raw argmax, before repair.",fontsize=10,color="#41545c")
    fig.savefig(directory/"motif-comparison.png",dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()
