"""Blind offline comparison of recoverable reduced amides and legitimate oxidizable aminoalcohols."""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase

from experiments.vae.training_data_eda.ring_site_localization import (
    aligned_output_sites,
)
from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.chemistry import MoleculeRejected, encode_canonical_smiles
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.motif_completion import (
    _ALCOHOL_NEXT_TO_N,
    AMIDE,
    _roundtrips,
    _valid,
)
from latentmol.representation import HardGraphBatch


def canonical(molecule):
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def change(mol, c, o, order):
    edited = Chem.Mol(mol)
    edited.GetBondBetweenAtoms(c, o).SetBondType(order)
    if not _valid(edited) or not _roundtrips(edited):
        return None
    return edited


def choices(molecule):
    """The same rule applies on both classes; every candidate must allow a valid amide."""
    candidates = {}
    for o, c, n in molecule.GetSubstructMatches(_ALCOHOL_NEXT_TO_N):
        if (c, o) in candidates:
            continue
        product = change(molecule, c, o, Chem.BondType.DOUBLE)
        if product is not None and any(
            cc == c and oo == o for cc, oo, _ in product.GetSubstructMatches(AMIDE)
        ):
            candidates[(c, o)] = (
                canonical(product),
                molecule.GetAtomWithIdx(c).GetDegree(),
                molecule.GetAtomWithIdx(n).GetDegree(),
            )
    return candidates


def blinded_source(smiles, spec):
    """One source/site. Exclude extra amides from both precursor classes and validate the slot map."""
    mol = Chem.MolFromSmiles(smiles)
    clean = canonical(mol)
    amides = list(dict.fromkeys((c, o) for c, o, _ in mol.GetSubstructMatches(AMIDE)))
    if len(amides) > 1:
        return None
    if amides:
        c, o = amides[0]
        base = change(mol, c, o, Chem.BondType.SINGLE)
        if base is None or base.HasSubstructMatch(AMIDE):
            return None
    else:
        base = mol
    possible = choices(base)
    if not possible:
        return None
    if amides:
        if (c, o) not in possible or possible[(c, o)][0] != clean:
            return None
    else:
        # A chemically possible oxidation away from the observed real identity is a hard negative.
        c, o = next(iter(possible))
    text = canonical(base)
    written_order = json.loads(base.GetProp("_smilesAtomOutputOrder"))
    locations = {old: new for new, old in enumerate(written_order)}
    try:
        encode_canonical_smiles(text, spec)
    except MoleculeRejected:
        return None
    parsed = Chem.MolFromSmiles(text)
    tested = choices(parsed)
    mapped = [(locations[i], locations[j]) for i, j in possible]
    if any(
        site not in tested or tested[site][0] != possible[original][0]
        for original, site in zip(possible, mapped, strict=True)
    ):
        return None
    target = (locations[c], locations[o])
    _, c_degree, n_degree = possible[(c, o)]
    return {
        "clean": clean,
        "precursor": text,
        "positive": bool(amides),
        "sites": mapped,
        "examined_site": mapped.index(target),
        "heavy": mol.GetNumAtoms(),
        "c_degree": c_degree,
        "n_degree": n_degree,
    }


def auc(positive, negative):
    if not positive or not negative:
        return None
    ordered = np.sort(negative)
    low = np.searchsorted(ordered, positive, side="left")
    high = np.searchsorted(ordered, positive, side="right")
    return float(np.mean((low + high) * 0.5) / len(negative))


def stats(entries, seed):
    scored = [r for r in entries if r["score"] is not None]
    gp = [[r["score"] for r in scored if r["positive"] == cls] for cls in (True, False)]
    obs = auc(*gp)
    by_context = defaultdict(lambda: [[], []])
    for r in scored:
        by_context[r["heavy"] // 5, r["c_degree"], r["n_degree"]][int(not r["positive"])].append(r["score"])
    weighted = weight = 0
    for p, n in by_context.values():
        if p and n:
            pairs = len(p) * len(n)
            weighted += pairs * auc(p, n)
            weight += pairs
    reps = []
    if min(map(len, gp)) >= 5:
        gen = np.random.default_rng(seed)
        for _ in range(800):
            reps.append(auc(*[gen.choice(a, len(a), replace=True).tolist() for a in gp]))
    competing = [
        r
        for r in scored
        if r["positive"] and len(r["site_scores"]) > 1 and all(x is not None for x in r["site_scores"])
    ]
    within = []
    for r in competing:
        others = [s for i, s in enumerate(r["site_scores"]) if i != r["examined_site"]]
        within.append(auc([r["score"]], others))
    return {
        "considered": dict(Counter(str(r["positive"]) for r in entries)),
        "scored_positive": len(gp[0]),
        "scored_negative": len(gp[1]),
        "attribution": dict(Counter(r["mapping"] for r in entries)),
        "auc": obs,
        "molecule_bootstrap_interval": np.quantile(reps, [0.025, 0.975]).tolist() if reps else None,
        "matched_context_auc": weighted / weight if weight else None,
        "matched_comparisons": weight,
        "median_positive": float(np.median(gp[0])) if gp[0] else None,
        "median_negative": float(np.median(gp[1])) if gp[1] else None,
        "competing_molecules": len(competing),
        "within_molecule_mean_auc": float(np.mean(within)) if within else None,
        "unique_top_positive": sum(
            r["score"] > max(v for j, v in enumerate(r["site_scores"]) if j != r["examined_site"])
            for r in competing
        ),
    }


def score(model, rows, batch_size):
    single, double = (model.spec.bond_names.index(name) for name in ("SINGLE", "DOUBLE"))
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            block = rows[start : start + batch_size]
            graph = encode([r["precursor"] for r in block], model.spec)
            mean, _ = model.encoder(graph, from_canonical_cache=True)
            out = model.decode(mean)
            native = out.harden(model.spec)
            odds = out.symmetric_bond_logits[..., double] - out.symmetric_bond_logits[..., single]

            def single_row(graph, k):
                return HardGraphBatch(
                    *(getattr(graph, key)[k : k + 1] for key in ("atoms", "charges", "bonds"))
                )

            for k, r in enumerate(block):
                sites, reason = aligned_output_sites(
                    single_row(graph, k), single_row(native, k), r["sites"], model.spec
                )
                r["site_scores"] = [float(odds[k, s[0], s[1]]) if s is not None else None for s in sites]
                r["mapping"] = reason
                r["score"] = r["site_scores"][r["examined_site"]]


def collect(smiles, spec, positive_cap):
    selected = []
    seen = set()
    positives = 0
    with rdBase.BlockLogs():
        for sm in smiles:
            if sm in seen:
                continue
            seen.add(sm)
            if positives >= positive_cap and Chem.MolFromSmiles(sm).HasSubstructMatch(AMIDE):
                continue
            row = blinded_source(sm, spec)
            if row:
                selected.append(row)
                positives += int(row["positive"])
    # The observable precursor and site cannot imply a unique clean label when
    # both originals exist; exclude all such cases and any repeated precursor.
    duplicates = Counter(r["precursor"] for r in selected)
    return [r for r in selected if duplicates[r["precursor"]] == 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "validation", "training_packet", "out"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--supplement-size", type=int, default=12000)
    p.add_argument("--seed", type=int, default=2026112571)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    model, data = load_model(args.checkpoint, "cpu")
    model.eval()
    held = [row[2] for row in json.loads(args.validation.read_bytes())["rows"]]
    with gzip.open(args.training_packet, "rt") as f:
        train = [r["smiles"] for r in csv.DictReader(f, delimiter="\t")]
    if set(held) & set(train):
        raise ValueError("Training and validation packets overlap")
    # Avoid data rows reserved for the latest matched auxiliary training pilot.
    excluded = np.random.default_rng(2026112517).choice(len(train), 8192, replace=False)
    available = np.setdiff1d(np.arange(len(train)), excluded)
    supplement = np.random.default_rng(args.seed).choice(available, args.supplement_size, replace=False)
    summary = {
        "checkpoint": file_sha256(args.checkpoint),
        "checkpoint_update": data["continuation"]["updates"],
        "validation_sha256": file_sha256(args.validation),
        "training_packet_sha256": file_sha256(args.training_packet),
        "seed": args.seed,
        "rdkit": rdBase.rdkitVersion,
        "torch": torch.__version__,
        "panels": {},
    }
    for name, smiles, cap in (
        ("heldout", held, len(held)),
        ("training_corpus_supplement", [train[i] for i in supplement], 400),
    ):
        rows = collect(smiles, model.spec, cap)
        with rdBase.BlockLogs():
            score(model, rows, 12)
        with gzip.open(args.out / (name + ".jsonl.gz"), "xt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary["panels"][name] = dict(
            input_rows=len(smiles), positive_cap=cap, **stats(rows, args.seed + 1)
        )
        write_json(args.out / "summary.json", summary)
        print(name, summary["panels"][name], flush=True)


if __name__ == "__main__":
    main()
