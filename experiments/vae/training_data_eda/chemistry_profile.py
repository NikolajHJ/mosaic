"""Molecule-level ring and functional-group measurements on sanitized VAE graphs."""

from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# The first query atom is the center used in the outside-ring measurement.
# Group families overlap (for example a phenol can coexist with an aryl halide).
GROUP_DEFINITIONS = {
    "amide": ("Amide-type carbonyl bonded to nitrogen (includes lactams, ureas and carbamates)", "[CX3](=[OX1])-[NX3,NX4+]"),
    "amine": (
        "Neutral trivalent or protonated primary/secondary/tertiary amine, excluding acyl, amidine, sulfonyl and phosphoryl nitrogens",
        "[N;X3,X4;H0,H1,H2,H3;!$([N+;H0]);!$(N-[C,S,P]=[O,N,S]);!$(N-[N,O,F,Cl,Br,I]);!$(N=*);!$(N#*);+0,+1]-[#6]",
    ),
    "alcohol": ("Alcohol on saturated carbon", "[OX2H1]-[CX4]"),
    "phenol": ("Phenolic hydroxyl", "[OX2H1]-[c]"),
    "ether": ("Ether excluding acyl oxygens", "[OD2X2;H0](-[#6;!$([#6]=[O,S,N])])-[#6;!$([#6]=[O,S,N])]"),
    "ester": ("Carboxylic ester (includes formates and lactones)", "[CX3;H1,$([C]-[#6])](=[OX1])-[OX2;H0]-[#6;!$([#6]=[O,S,N])]"),
    "carboxyl": ("Carboxylic acid or carboxylate (includes formic acid)", "[CX3;H1,$([C]-[#6])](=[OX1])-[OX2H1,OX1-]"),
    "ketone": ("Ketone", "[CX3](=[OX1])(-[#6])-[#6]"),
    "aldehyde": ("Aldehyde on carbon", "[CX3H1](=[OX1])-[#6]"),
    "nitrile": ("Carbon-substituted nitrile", "[CX2](#[NX1])-[#6]"),
    "nitro": ("Carbon-bound charge-separated nitro group", "[NX3+](-[O-])(=[OX1])-[#6]"),
    "organohalogen": ("F, Cl, Br or I bonded to carbon", "[F,Cl,Br,I]-[#6]"),
    "sulfonamide": ("Sulfonamide", "[SX4](=[OX1])(=[OX1])-[NX3]"),
    "sulfone": ("Sulfone", "[SX4](=[OX1])(=[OX1])(-[#6])-[#6]"),
    "sulfoxide": ("Sulfoxide, neutral or charge-separated", "[S;D3;$([S]=[OX1]),$([S+]-[O-])](-[#6])-[#6]"),
    "thioether": ("Thioether excluding thioesters", "[SX2;H0](-[#6;!$([#6]=[O,S])])-[#6;!$([#6]=[O,S])]"),
    "thiol": ("Carbon-bound thiol", "[SX2H1]-[#6]"),
    "alkene": ("Nonaromatic carbon-carbon double bond", "[C]=[C]"),
    "alkyne": ("Carbon-carbon triple bond", "[C]#[C]"),
    "quaternary_ammonium": ("Quaternary ammonium with four carbon neighbors", "[NX4+](-[#6])(-[#6])(-[#6])-[#6]"),
}

_QUERIES = {key: Chem.MolFromSmarts(smarts) for key, (_, smarts) in GROUP_DEFINITIONS.items()}
if any(query is None for query in _QUERIES.values()):
    raise RuntimeError("Could not compile a functional-group SMARTS definition")


def molecule_profile(mol):
    """Return a flat numeric record. A family contributes once per molecule."""
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        raise ValueError("Expected a nonempty sanitized RDKit molecule")
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    ring_atoms = {atom for ring in rings for atom in ring}
    ring_sizes = {len(ring) for ring in rings}
    row = {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "ring_count": len(rings),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heterocycles": rdMolDescriptors.CalcNumHeterocycles(mol),
        "largest_ring_size": max(ring_sizes, default=0),
        "ring_atoms": len(ring_atoms),
        "ring_size_mask": sum(1 << size for size in ring_sizes),
    }
    group_count = external_count = 0
    for name, query in _QUERIES.items():
        matches = mol.GetSubstructMatches(query, uniquify=False)
        present = bool(matches)
        external = any(match[0] not in ring_atoms for match in matches)
        row[f"fg_{name}"] = int(present)
        row[f"ext_{name}"] = int(external)
        group_count += present
        external_count += external
    row["group_families"] = group_count
    row["external_group_families"] = external_count
    return row


def _pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else np.nan


def summarize_profiles(df):
    """Return denominator-explicit aggregate DataFrames for a nonempty sample."""
    import pandas as pd

    n = len(df)
    if not n:
        raise ValueError("Expected at least one molecule")
    ring = df["ring_count"].gt(0)
    has_group = df["group_families"].gt(0)
    external = df["external_group_families"].gt(0)
    n_ring = int(ring.sum())
    flags = {
        "any_ring": ring,
        "aromatic_ring": df["aromatic_rings"].gt(0),
        "heterocycle": df["heterocycles"].gt(0),
        "two_or_more_rings": df["ring_count"].ge(2),
        "ring_size_8_or_more": df["largest_ring_size"].ge(8),
        "any_listed_group": has_group,
        "ring_and_any_listed_group": ring & has_group,
        "ring_and_external_group_center": ring & external,
        "ring_and_two_or_more_group_families": ring & df["group_families"].ge(2),
        "ring_without_listed_group": ring & ~has_group,
        "acyclic_with_listed_group": ~ring & has_group,
        "acyclic_without_listed_group": ~ring & ~has_group,
    }
    overview = [
        {"feature": name, "n": int(flag.sum()), "n_all": n, "pct_all": _pct(int(flag.sum()), n)}
        for name, flag in flags.items()
    ]
    presence = {name: df[f"fg_{name}"].astype(bool) for name in GROUP_DEFINITIONS}
    group_rows = []
    for name, present in presence.items():
        total = int(present.sum())
        both = int((ring & present).sum())
        outside = int((ring & df[f"ext_{name}"].astype(bool)).sum())
        group_rows.append({
            "group": name, "n_all": n, "n_ring": n_ring, "n_acyclic": n - n_ring,
            "n_group": total, "n_with_ring": both, "n_ring_external_center": outside,
            "pct_all_with_group": _pct(total, n), "pct_all_with_both": _pct(both, n),
            "pct_ring_with_group": _pct(both, n_ring),
            "pct_acyclic_with_group": _pct(total - both, n - n_ring),
            "pct_group_with_ring": _pct(both, total),
            "pct_ring_with_external_center": _pct(outside, n_ring),
        })
    pair_rows = []
    for a, b in combinations(presence, 2):
        both_groups = presence[a] & presence[b]
        n_a, n_b = int(presence[a].sum()), int(presence[b].sum())
        joint = int(both_groups.sum())
        pair_rows.append({
            "group_a": a, "group_b": b, "n_a": n_a, "n_b": n_b, "n_both": joint,
            "n_both_with_ring": int((ring & both_groups).sum()), "n_all": n,
            "pct_all_with_both": _pct(joint, n),
            "lift": joint * n / (n_a * n_b) if n_a and n_b else np.nan,
        })

    def histogram(series, name):
        values = range(int(series.min()), int(series.max()) + 1)
        counts = series.value_counts().reindex(values, fill_value=0).rename("n")
        result = counts.rename_axis(name).reset_index()
        result["pct_all"] = result["n"] * 100 / n
        return result

    ring_size_rows = []
    for size in range(3, int(df["largest_ring_size"].max()) + 1):
        matches = int(((df["ring_size_mask"] & (1 << size)) != 0).sum())
        ring_size_rows.append({"ring_size": size, "n_molecules": matches,
                               "pct_all": _pct(matches, n), "pct_ring": _pct(matches, n_ring)})
    size_band_rows = []
    bands = (df["heavy_atoms"] - 1) // 5
    for index, subset in df.groupby(bands, sort=True):
        nr = int(subset["ring_count"].gt(0).sum())
        ng = int((subset["ring_count"].gt(0) & subset["group_families"].gt(0)).sum())
        size_band_rows.append({
            "min_heavy_atoms": int(5 * index + 1), "max_heavy_atoms": int(5 * index + 5),
            "n": len(subset), "n_ring": nr, "n_ring_and_group": ng,
            "pct_with_ring": _pct(nr, len(subset)), "pct_with_ring_and_group": _pct(ng, len(subset)),
            "mean_ring_count": subset["ring_count"].mean(),
            "mean_group_families": subset["group_families"].mean(),
        })

    return {
        "ring_overview": pd.DataFrame(overview),
        "ring_by_group": pd.DataFrame(group_rows).sort_values("n_with_ring", ascending=False),
        "size_histogram": histogram(df["heavy_atoms"], "heavy_atoms"),
        "ring_count_histogram": histogram(df["ring_count"], "ring_count"),
        "ring_size_histogram": pd.DataFrame(ring_size_rows, columns=["ring_size", "n_molecules", "pct_all", "pct_ring"]),
        "size_bands": pd.DataFrame(size_band_rows),
        "group_pairs": pd.DataFrame(pair_rows).sort_values("n_both", ascending=False),
        "group_definitions": pd.DataFrame([
            {"group": name, "label": label, "smarts": pattern}
            for name, (label, pattern) in GROUP_DEFINITIONS.items()
        ]),
    }
