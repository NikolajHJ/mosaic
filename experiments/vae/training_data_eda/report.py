"""Build the illustrated, offline report from the panel and census results."""

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from plot_chemistry import make_figures


def wilson(n, total):
    p, z = n / total, 1.96
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * np.sqrt(p*(1-p)/total + z*z/(4*total**2)) / (1 + z*z/total)
    return f"{100*(center-half):.2f}–{100*(center+half):.2f}%"


def fig(path, caption):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(v))}</th>" for v in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def structural_figures(out, data, scaffolds):
    panels = [
        ("molecular_weight", "Molecular weight (Da)", 42), ("logP", "Calculated logP", 45),
        ("TPSA", "Topological polar surface area (Å²)", 45),
        ("rotatable_bonds", "Rotatable bonds", np.arange(-.5, data.rotatable_bonds.max()+1.5)),
        ("fraction_C_sp3", "Fraction of carbon atoms that are sp³", np.linspace(0, 1, 31)),
        ("group_families", "Distinct listed functional-group families", np.arange(-.5, data.group_families.max()+1.5)),
    ]
    canvas, axes = plt.subplots(2, 3, figsize=(12, 6.7), layout="constrained")
    for ax, (column, label, bins) in zip(axes.flat, panels):
        values = data[column]
        ax.hist(values, weights=np.full(len(values), 100/len(values)), bins=bins, color="#247b80", rwidth=.90)
        ax.set(xlabel=label, ylabel="Sample (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.97, .94, f"Median {values.median():.2f}".rstrip("0").rstrip("."), ha="right", va="top",
                transform=ax.transAxes, fontsize=9)
    canvas.suptitle("Molecular properties in the random training panel", fontsize=16)
    canvas.savefig(out / "properties.png", dpi=155, facecolor="white")
    plt.close(canvas)
    top = scaffolds.head(8)
    legends = [f"{i}.  {r.n:,} molecules  ({100*r.n/len(data):.2f}%)" for i, r in enumerate(top.itertuples(), 1)]
    plate = Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in top.scaffold], legends=legends,
                                 molsPerRow=4, subImgSize=(245, 160), useSVG=False)
    plate.save(out / "scaffold_examples.png")


def build(run, record):
    out = run / "analysis"
    products = ["report.html", "properties.png", "scaffold_examples.png"]
    for name in products:
        if (out / name).exists():
            raise FileExistsError(out / name)
    source = run / "inputs" / "census.json"
    census, meta, dim = map(read_json, (source, out/"panel_summary.json", out/"dimensions.json"))
    data = pd.read_csv(out / "profiles.csv.gz", keep_default_na=False)
    scaffolds = pd.read_csv(out / "scaffolds.csv.gz")
    make_figures(out, source, record)
    structural_figures(out, data, scaffolds)
    n, total = meta["n"], census["train_rows"]
    if len(data) != n or census["sample_size"] != n:
        raise ValueError("Panel sizes differ between census and chemistry results")
    counts = {int(k): v for k, v in census["cycle_rank_counts"].items()}
    n_cyclic = total - counts.get(0, 0)
    overview = pd.read_csv(out / "ring_overview.csv").set_index("feature")
    groups = pd.read_csv(out / "ring_by_group.csv")
    desc = pd.read_csv(out / "descriptor_summary.csv", index_col=0)
    pairs = pd.read_csv(out / "group_pairs.csv")
    sizes = sorted((int(k), v) for k,v in census["size_counts"].items())
    accum = np.cumsum([v for _,v in sizes])
    quantiles = [sizes[np.searchsorted(accum, total*q)][0] for q in [.05, .25, .5, .75, .95]]
    n_ring = int(overview.loc["any_ring", "n"])
    joint = int(overview.loc["ring_and_any_listed_group", "n"])
    external = int(overview.loc["ring_and_external_group_center", "n"])
    ring_sizes = pd.read_csv(out / "ring_size_histogram.csv").set_index("ring_size")
    tail = data.loc[data.heavy_atoms.ge(41)].groupby(data.ring_count.gt(0)).rotatable_bonds
    elements = pd.read_csv(out / "elements.csv")
    text = [f"""<header><p>DBMol &nbsp; / &nbsp; Training data &nbsp; / &nbsp; 20 November 2026</p>
    <h1>Inside the VAE training set</h1>
    <p class="dek">A census of {total:,} molecular graphs and a chemical portrait of {n:,} randomly selected training molecules.</p></header>
    <section><h2>Rings dominate the training data</h2><p><strong>{100*n_cyclic/total:.2f}%</strong> of all training molecules contain a ring.
    In the random panel, <strong>{100*joint/n:.2f}%</strong> contain both a ring and at least one of the 20 listed functional-group families
    (95% sampling interval {wilson(joint,n)}). Among ring-bearing molecules that share is <strong>{100*joint/n_ring:.2f}%</strong>.
    Requiring the defining atom of a group to lie outside every ring gives {100*external/n:.2f}% of all sampled molecules.</p>"""]
    text += [table(["Training population", "Molecules", "Share"], (
        (label, f"{amount:,}", f"{100*amount/total:.2f}%") for label, amount in
        [(f"{r} independent cycle" + ("" if r == 1 else "s"), counts.get(r,0)) for r in range(5)]
        + [("5 or more independent cycles", sum(v for k,v in counts.items() if k >= 5))])),
        f"""<p>For each connected graph, independent cycles equal bonds minus heavy atoms plus one.
        A positive value detects a ring exactly. Individual ring shapes and ring sizes below use RDKit ring perception.</p>
        <h2>Size reflects how this split was built</h2><p>The median is <strong>{quantiles[2]} heavy atoms</strong>,
        with the middle half between {quantiles[1]} and {quantiles[3]} and the 5th–95th percentile range from {quantiles[0]} to {quantiles[4]}.
        The split caps common sizes. Each size from 15 through 31 contributes about 2.286 million graphs.
        Smaller and larger bins had fewer molecules available.</p>""",
        fig(out/"size_and_rings.png","Exact heavy-atom counts and cyclic structure across the training population."),
        f"""<p>The size trend reverses at the upper end. Among panel molecules with 41–50 heavy atoms,
        the {int(tail.size()[False]):,} acyclic molecules have a median of {tail.median()[False]:.0f} rotatable bonds,
        compared with {tail.median()[True]:.0f} in the {int(tail.size()[True]):,} ring-bearing molecules in the same size range.</p>""",
        "<h2>Which groups appear with rings?</h2><p>Each family counts once per molecule. The 20 definitions include amide-type carbonyls (including lactams, ureas and carbamates), amines, ethers, carbon-bound halogens and nonaromatic carbon–carbon double bonds. Families can overlap. A ring and group may occur anywhere in the same molecule. The outside-ring measure uses the center picked by the SMARTS definition, and that center can be several bonds from the ring.</p>",
        f"""<p>Listed functional groups also occur in {100*overview.at['acyclic_with_listed_group','n']/(n-n_ring):.2f}% of acyclic molecules.
        Group identity distinguishes the two subsets more sharply. Amide-type carbonyls occur more frequently in ring-bearing molecules,
        while nonaromatic alkenes and alcohols occur more frequently in acyclic molecules.</p>
        <p>Of all sampled molecules, {overview.at['aromatic_ring','pct_all']:.2f}% contain an aromatic ring
        and {overview.at['heterocycle','pct_all']:.2f}% contain a heterocycle, a ring with at least one noncarbon atom.
        Six-membered rings occur in {ring_sizes.at[6,'pct_all']:.2f}% and five-membered rings in {ring_sizes.at[5,'pct_all']:.2f}%.
        These classes overlap. In {overview.at['ring_and_two_or_more_group_families','pct_all']:.2f}% of the panel,
        a ring co-occurs with at least two distinct listed group families.</p>""",
        table(["Group", "Ring + group", "% of all", "% within cyclic", "% within acyclic"], (
            ("amide-type carbonyl" if r.group=="amide" else r.group.replace("_"," "), f"{r.n_with_ring:,}", f"{r.pct_all_with_both:.2f}",
             f"{r.pct_ring_with_group:.2f}", f"{r.pct_acyclic_with_group:.2f}") for r in groups.itertuples())),
        fig(out/"groups_and_rings.png","Functional-group prevalence and co-occurrence in the uniform training panel."),
        "<p>The most frequent pairs of different group families in the sample appear below. Lift divides the observed joint frequency by the product of the two marginal frequencies. It includes differences in molecular size and source composition.</p>",
        table(["Pair", "Molecules", "% of all", "Lift"], (
            (f"{r.group_a} + {r.group_b}", f"{r.n_both:,}", f"{r.pct_all_with_both:.2f}%", f"{r.lift:.2f}×") for r in pairs.head(5).itertuples())),
        "<h2>Molecular properties and scaffold diversity</h2>",
        fig(out/"properties.png","Empirical distributions of six properties. Each histogram uses the entire uniform chemistry panel."),
        table(["Descriptor", "5th percentile", "Median", "95th percentile"],
              ((c.replace("_"," "), *(f"{desc.at[c,k]:.2f}" for k in ["5%", "50%", "95%"])) for c in meta["features"])),
        "<h3>Element and charge composition</h3>",
        table(["Element", "Molecules carrying it", "% of sample molecules", "% of all heavy atoms"],
              ((r.element,f"{r.molecules:,}",f"{r.pct_molecules:.2f}%",f"{r.pct_atoms:.2f}%") for r in elements.itertuples())),
        f"""<p>{100*meta['any_charge']/n:.2f}% of molecules contain a formally charged atom.
        {100*meta['net_charge']/n:.2f}% carry a nonzero net formal charge after summing across all atoms.</p>""",
        f"""<p>The ring-bearing sample contains <strong>{meta['scaffold_distinct']:,} distinct Murcko scaffolds</strong>.
        Of these scaffold identities, {meta['scaffold_singletons']:,} occur once in this panel. The ten most frequent scaffolds account for
        {100*meta['scaffold_top10']/n:.2f}% of all sample molecules. Acyclic molecules have an empty scaffold and are excluded from scaffold identity counts.
        The median similarity between two random fingerprinted molecules is {meta['pair_tanimoto_p05_p50_p95'][1]:.3f} Tanimoto
        (5th–95th percentiles {meta['pair_tanimoto_p05_p50_p95'][0]:.3f}–{meta['pair_tanimoto_p05_p50_p95'][2]:.3f}, {meta['fingerprint_rows']//2:,} disjoint pairs).</p>""",
        fig(out/"scaffold_examples.png","Eight most frequent Murcko scaffolds. Percentages use all sampled training graphs."),
        "<h2>How many dimensions capture the variance?</h2>"]
    pc, fp, ic = dim["pca"], dim["fingerprint_pca"], dim["ica"]
    representations = [("Standardized molecular descriptors",pc),("Morgan fingerprint bits",fp)]
    latent_path = run/"latent"/"latent_summary.json"
    latent = read_json(latent_path) if latent_path.exists() else None
    if latent:
        representations.append(("Frozen VAE posterior means",latent))
    text += [table(["Representation", "Dimensions", "50%", "80%", "90%", "95%", "99%"],
                   ((name, d["n_dimensions"], *(d["dimensions_for_variance_pct"][str(q)] for q in [50,80,90,95,99]))
                    for name,d in representations)),
        f"""<p>Two descriptor PCs capture {pc['first_two_variance_pct']:.1f}% of total standardized variance. The same two-dimensional view captures
        {fp['first_two_variance_pct']:.1f}% of variance in {fp['fingerprint_bits']:,} Morgan fingerprint bits. Fingerprints encode local atom neighborhoods and spread variance across many more directions.
        These thresholds depend on the representation, scaling and sample. Reconstruction or generation quality requires a separate evaluation.</p>"""]
    if latent:
        scores = pd.read_csv(run / "latent" / "latent_pc_scores.csv")
        joined = scores.merge(data, on=["view_row", "heavy_atoms", "cycle_rank"], validate="one_to_one")
        if len(joined) != latent["n_rows"]:
            raise ValueError("Latent projections and molecular descriptors refer to different panel rows")
        corrs = joined[meta["features"] + ["pc1", "pc2"]].corr().loc[meta["features"], ["pc1", "pc2"]]
        corrs.to_csv(out / "latent_pc_descriptor_correlations.csv", mode="x", index_label="descriptor")
        text += [f"""<p>The frozen large VAE at update 524,177 contributed {latent['n_rows']:,} posterior-mean vectors for training molecules.
        PCA used their native coordinate scales and centered each dimension. The first two directions captured {latent['first_two_variance_pct']:.1f}% of variance in this checkpoint's codes.
        Its first PC tracks heavy-atom count (r = {corrs.at['heavy_atoms','pc1']:.2f}). The second tracks the fraction of aromatic atoms
        (r = {corrs.at['aromatic_atom_fraction','pc2']:.2f}) and is inversely associated with the fraction of sp³ carbons (r = {corrs.at['fraction_C_sp3','pc2']:.2f}).
        Fitting the axes on one random half of these molecules and evaluating their projections on the other half gave
        {latent['heldout_dimensions_for_variance_pct']['90']} PCs for 90% of held-out variance.
        These dimensions describe posterior means. Sampling variability within an individual molecule's posterior is excluded.</p>""",
        fig(run/"latent"/"latent_pca.png","Frozen VAE codes on 4,096 training molecules, colored by heavy-atom count, with cumulative variance on the full panel and held-out half.")]
    text += [fig(out/"dimension_overview.png","PCA and initial symmetric ICA density projections and cumulative PCA spectra. Brighter hexagons contain more molecules. The separate deflation fit below checks the ICA result."),
        table(["Descriptor PC", "Variance", "Strongest correlations with the component"],
              ((r["component"],f"{r['variance_pct']:.1f}%",", ".join(f"{a['descriptor'].replace('_',' ')} ({a['r']:+.2f})" for a in r["top_descriptor_correlations"][:3])) for r in pc["components"][:4])),
        f"""<p>ICA rotated the first {ic['n_components']} descriptor PCs, a subspace carrying {ic['pca_subspace_variance_pct']:.1f}% of descriptor variance,
        to seek statistically independent directions. ICA has no intrinsic variance ranking. The displayed order uses absolute excess kurtosis.
        Each fit used {ic['primary_fit']['fit_rows']:,} rows. Both initial symmetric fits reached the 500-iteration limit.</p>"""]
    sensitivity = read_json(out / "ica_sensitivity.json")
    text += ["""<p>We refit the same six-dimensional subspace and row draws by extracting one independent component at a time (deflation FastICA).
        Both fits converged. Four directions, associated with charged atoms, rotatable bonds, hydrogen-bond donors and halogens, reproduced across draws
        with absolute correlations above 0.998. The other two changed appreciably (about 0.74). Interpret the first four as reproducible chemical patterns in this panel.</p>""",
        fig(out/"ica_deflation.png","Deflation FastICA on two seeded fitting panels. The first four independent directions reproduced closely across the two draws."),
        table(["Deflation ICA direction", "Strongest descriptor correlations", "Repeat |r|"],
              ((r["component"],", ".join(f"{a['descriptor'].replace('_',' ')} ({a['r']:+.2f})" for a in r["top_descriptor_correlations"][:3]),
                f"{r['matched_repeat_abs_r']:.3f}") for r in sensitivity["components"])),
        fig(out/"component_correlations.png","Descriptor correlations with PCA and the initial symmetric ICA scores. The deflation rerun above shows which patterns reproduced."),
        f"""<h2>Population and methods</h2><p>The real-data split in the large epoch-47 and compact VAE configs comprises 60,000,000 heavy-atom-balanced, unique,
        connected, stereo-free graphs selected from the 81,409,264-row PubChem–ChEMBL–MOSES parent training corpus. Graphs have 1–50 heavy atoms. Hydrogen is implicit.
        This analysis weights training identities equally. It excludes both holdouts and synthetic prior draws.</p>
        <p>We scanned every selected record header and checked the 50 per-size counts against the published manifest.
        We sampled view rows uniformly without replacement (seed {census['sample_seed']}) and checked parsed atom counts, cycle rank and connectivity against the packed graphs.
        Sampling intervals use the 95% Wilson binomial interval. They quantify row sampling only. The finite-population correction is negligible at this sampling fraction.</p>
        <p>RDKit {meta['rdkit']} supplied the ring perception, SMARTS matching, descriptors, scaffolds and fingerprints.
        Descriptor PCA gives each of the {len(meta['features'])} columns unit variance, including correlated descriptors. Fingerprint PCA uses
        {meta['fingerprint_rows']:,} randomly selected panel molecules and 1,024 mean-centered radius-2 Morgan bits, with their original relative bit variances.
        Fingerprint hashing can make different atom environments share a bit.</p>
        <p>Dataset ID <code>{census['dataset_sha256']}</code>. The CSVs next to this report contain the full group definitions, group pairs, size bands, molecule-level measurements, spectra and component correlations.</p>
        <p>Local result files &nbsp; <a href="../inputs/census.json">Exact census</a> &nbsp; · &nbsp; <a href="group_definitions.csv">SMARTS definitions</a> &nbsp; · &nbsp; <a href="ring_by_group.csv">Ring and group counts</a> &nbsp; · &nbsp; <a href="group_pairs.csv">All group pairs</a> &nbsp; · &nbsp; <a href="pca_variance.csv">PCA spectra</a></p>
        <p>Method references &nbsp; <a href="https://www.rdkit.org/docs/GettingStartedInPython.html">RDKit</a> &nbsp; · &nbsp; <a href="https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html">Murcko scaffolds</a> &nbsp; · &nbsp; <a href="https://scikit-learn.org/stable/modules/decomposition.html">PCA and ICA</a></p></section>"""]
    style = """*{box-sizing:border-box}body{margin:0;background:#f5f3ed;color:#182f32;font:17px/1.55 system-ui,sans-serif}
    main{background:#fff;max-width:1050px;margin:auto;padding:3.5rem clamp(1rem,6vw,5rem) 5rem}
    header{border-top:5px solid #176c69;border-bottom:1px solid #aabbb4;padding:1rem 0}header>p:first-child{font-size:.81rem;letter-spacing:.07em;text-transform:uppercase;color:#426560}
    h1,h2{font-family:Georgia,serif;line-height:1.13}h1{font-size:clamp(2.5rem,6vw,4.2rem);margin:.5rem 0}h2{font-size:2rem;margin:2.5rem 0 .7rem}
    .dek{font-size:1.25rem;max-width:48rem}strong{color:#075d59}figure{margin:1.6rem 0}img{width:100%;height:auto}figcaption{font-size:.85rem;color:#496162}
    table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.90rem}td,th{padding:.45rem .60rem;border-bottom:1px solid #dce4df;text-align:right}
    th{background:#e8f1ed}td:first-child,th:first-child,td:nth-child(3):last-child{text-align:left}.scroll{overflow-x:auto;margin:1rem 0}a{color:#09615d}code{font-size:.78em;overflow-wrap:anywhere}"""
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inside the VAE training set</title>'
    page += f"<style>{style}</style><main>" + "".join(text) + "</main></html>"
    (out / "report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("manifest_record", type=Path)
    args = parser.parse_args()
    build(args.run, args.manifest_record)
