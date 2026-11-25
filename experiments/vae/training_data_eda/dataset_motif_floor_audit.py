"""Replay motif hit counts and measure perceived aromaticity by ring size on real training rows."""

import argparse
import csv
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from latentmol.guidance.motif_completion import AMIDE


def corpus(panel):
    counts = Counter()
    sizes = {}
    with gzip.open(panel, "rt") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError(f"Training molecule failed to parse at row {row['view_row']}")
            counts["draws"] += 1
            counts["aromatic_hits"] += rdMolDescriptors.CalcNumAromaticRings(mol) > 0
            counts["amide_hits"] += mol.HasSubstructMatch(AMIDE)
            for ring in mol.GetRingInfo().BondRings():
                bonds = [mol.GetBondWithIdx(idx) for idx in ring]
                aromatic = all(b.GetIsAromatic() for b in bonds)
                tally = sizes.setdefault(len(ring), Counter())
                tally["rings"] += 1
                tally["aromatic"] += aromatic
                if all(a.GetAtomicNum() in (6, 7) and a.GetFormalCharge() == 0
                       for b in bonds for a in (b.GetBeginAtom(), b.GetEndAtom())):
                    tally["neutral_cn_rings"] += 1
                    tally["neutral_cn_aromatic"] += aromatic
    return dict(molecules=counts, ring_instances=sizes)


def score_abs(batch, hits, floor):
    """L1 mass of one motif's leave-one-out weights before the common two-motif average."""
    weight = 0.
    for label, multiplicity in ((1, hits), (0, batch - hits)):
        if not multiplicity:
            continue
        others = hits - label
        below = max(floor - others / batch, 0.)
        above = max(floor - (others + 1) / batch, 0.)
        slope = 0.5 * (above**2 - below**2)
        weight += multiplicity * abs(slope * (label - others / (batch - 1)))
    return weight


def replay(batches, pool, floors):
    kept = len(batches) // pool * pool
    groups = [batches[i:i+pool] for i in range(0, kept, pool)]
    result = dict(panels=len(groups), tail_panels_excluded=len(batches)-kept)
    for name, old in (("aromatic_hits", .08), ("amide_hits", .12)):
        pairs = [(sum(r["draws"] for r in group), sum(r[name] for r in group)) for group in groups]
        total, hits = map(sum, zip(*pairs))
        old_mass = sum(score_abs(n,k,old) for n,k in pairs)
        new_mass = sum(score_abs(n,k,floors[name]) for n,k in pairs)
        result[name] = dict(attempts=total, hits=hits, zero_hit_panels=sum(k == 0 for _,k in pairs),
                            mean_abs_weight_per_panel_old=old_mass / len(pairs),
                            mean_abs_weight_per_panel_new=new_mass / len(pairs),
                            ratio_at_fixed_batch_size=new_mass / old_mass if old_mass else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    observed = corpus(args.panel)
    m = observed["molecules"]
    floors = {k: m[k] / m["draws"] for k in ("aromatic_hits", "amide_hits")}
    with args.training_log.open() as stream:
        rows = [json.loads(line) for line in stream]
    replays = {}
    for name, runs in (("full_leg", rows), ("last_1000_updates", rows[-1000:])):
        batches = [r["functional_groups"] for r in runs if "functional_groups" in r]
        if any(r["draws"] != 48 for r in batches):
            raise ValueError("Expected archived 48-draw motif batches")
        if not batches:
            raise ValueError("No motif batches")
        replays[name] = dict(first_update=runs[0]["updates"], last_update=runs[-1]["updates"],
                             original=replay(batches, 1, floors), pooled=replay(batches, 5, floors))
    evidence = dict(training_log_sha256=hashlib.sha256(args.training_log.read_bytes()).hexdigest(),
                    panel_sha256=hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                    corpus=observed, floors=floors, replay=replays)
    with args.out.open("x") as stream:
        json.dump(evidence, stream, indent=2)


if __name__ == "__main__":
    main()
