"""Locate missing motif components in raw graphs from a frozen VAE, including invalid draws.

The counterfactual readouts hold the occupied atoms, charges and undirected edge set fixed.
They test whether the existing skeleton could support a motif by changing bond labels only.
They are diagnostics and are excluded from every raw generation count.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from chemistry_profile import GROUP_DEFINITIONS
from latentmol.chemistry import encode_canonical_smiles, inspect_graph
from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import GraphSpec


AMIDE = Chem.MolFromSmarts(GROUP_DEFINITIONS["amide"][1])


def labels(smiles):
    if smiles is None:
        return False, False
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcNumAromaticRings(mol) > 0, mol.HasSubstructMatch(AMIDE)


def skeleton_rings(slots, bonds):
    """Compute symmetric smallest rings ignoring chemistry; indices map back to original slots."""
    skeleton = Chem.RWMol()
    for _ in slots:
        skeleton.AddAtom(Chem.Atom(6))
    i, j = np.nonzero(np.triu(bonds[np.ix_(slots, slots)], 1))
    for left, right in zip(i, j, strict=True):
        skeleton.AddBond(int(left), int(right), Chem.BondType.SINGLE)
    return [[int(slots[i]) for i in ring] for ring in Chem.GetSymmSSSR(skeleton.GetMol())]


def profile(atoms, charges, bonds, spec, logits=None):
    slots = np.flatnonzero(atoms)
    element = np.asarray(spec.atomic_numbers)[atoms]
    charge = np.asarray(spec.formal_charges)[charges]
    single, double, aromatic = [spec.bond_names.index(name) for name in ("SINGLE", "DOUBLE", "AROMATIC")]
    outcome = inspect_graph(atoms, charges, bonds, spec)
    is_aromatic, is_amide = labels(outcome.smiles)
    rings = skeleton_rings(slots, bonds)
    flags = {"valid": bool(outcome.smiles), "strict_aromatic": is_aromatic, "strict_amide": is_amide,
             "any_ring": bool(rings), "ring5_or_6": any(len(r) in (5, 6) for r in rings),
             "ring6": any(len(r) == 6 for r in rings), "nitrogen": bool((element == 7).any()),
             "oxygen": bool((element == 8).any()), "raw_aromatic_edge": bool((bonds == aromatic).any()),
             "all_single_ring5_or_6": False, "aromatic_label_cycle": False, "neutral_CN_ring6": False,
             "ring6_relabel_produces_valid_aromatic": False,
             "carbonyl": False, "carbonyl_and_nitrogen": False, "carbon_adjacent_O_and_N": False,
             "O_double_relabel_produces_valid_amide": False}
    best_margin = None
    for ring in rings:
        left, right = ring, ring[1:] + ring[:1]
        edge_labels = bonds[left, right]
        flags["aromatic_label_cycle"] |= bool((edge_labels == aromatic).all())
        if len(ring) in (5, 6):
            flags["all_single_ring5_or_6"] |= bool((edge_labels == single).all())
        if len(ring) != 6 or not np.isin(element[ring], (6, 7)).all() or charge[ring].any():
            continue
        flags["neutral_CN_ring6"] = True
        if logits is not None:
            # Smallest mean winning-logit minus aromatic-logit gap across eligible rings.
            # This includes unoccupied bond classes in the winner, as the raw head does.
            selected = logits[left, right]
            margin = float((selected.max(axis=1) - selected[:, aromatic]).mean())
            best_margin = margin if best_margin is None else min(best_margin, margin)
        if outcome.smiles and not is_aromatic and not flags["ring6_relabel_produces_valid_aromatic"]:
            proposed = bonds.copy()
            proposed[left, right] = proposed[right, left] = aromatic
            flags["ring6_relabel_produces_valid_aromatic"] = labels(
                inspect_graph(atoms, charges, proposed, spec).smiles)[0]
    for carbon in np.flatnonzero(element == 6):
        bonded = bonds[carbon] != 0
        has_carbonyl = bool(((element == 8) & (bonds[carbon] == double)).any())
        flags["carbonyl"] |= has_carbonyl
        flags["carbonyl_and_nitrogen"] |= has_carbonyl and flags["nitrogen"]
        flags["carbon_adjacent_O_and_N"] |= bool((bonded & (element == 8)).any() and
                                                 (bonded & (element == 7)).any())
        if (outcome.smiles and not is_amide and not flags["O_double_relabel_produces_valid_amide"]
                and ((element == 7) & (bonds[carbon] == single)).any()):
            for oxygen in np.flatnonzero((element == 8) & (bonds[carbon] == single)):
                proposed = bonds.copy()
                proposed[carbon, oxygen] = proposed[oxygen, carbon] = double
                if labels(inspect_graph(atoms, charges, proposed, spec).smiles)[1]:
                    flags["O_double_relabel_produces_valid_amide"] = True
                    break
    return {"flags": flags, "smiles": outcome.smiles, "failure": outcome.failure_reason,
            "best_CN_ring6_aromatic_logit_gap": best_margin}


def describe(rows):
    def counts(subset):
        return {"attempts": len(subset), **{key: sum(int(r["flags"][key]) for r in subset)
                                           for key in rows[0]["flags"]}}
    gaps = [r["best_CN_ring6_aromatic_logit_gap"] for r in rows
            if r["best_CN_ring6_aromatic_logit_gap"] is not None and not r["flags"]["strict_aromatic"]]
    return {"all": counts(rows), "valid_only": counts([r for r in rows if r["flags"]["valid"]]),
            "first_failure": dict(Counter(r["failure"] for r in rows if r["failure"])),
            "best_CN_ring6_gap_among_nonaromatic": {"n": len(gaps), "median": float(np.median(gaps)) if gaps else None}}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=2026112047)
    p.add_argument("--count", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(2)
    panel = json.loads(args.panel.read_bytes())
    spec = GraphSpec.from_dict(panel["spec"])
    if panel["rdkit"] != rdBase.rdkitVersion or not 1 <= args.count <= len(panel["rows"]) or args.batch < 1:
        raise ValueError("Panel version or requested budget incompatible")
    model, payload = load_model(args.checkpoint, "cpu", spec)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=False)
    selection = torch.randperm(len(panel["rows"]), generator=torch.Generator().manual_seed(args.seed + 1))
    selection = selection[:args.count].tolist()
    z = torch.randn(args.count, model.config.latent, generator=torch.Generator().manual_seed(args.seed))
    records = {"target": [], "reconstruction": [], "prior": []}
    with rdBase.BlockLogs(), gzip.open(args.output/"rows.jsonl.gz", "xt", encoding="utf-8") as stream:
        for start in range(0, args.count, args.batch):
            items = []
            indices = selection[start:start + args.batch]
            for i in indices:
                data = encode_canonical_smiles(panel["rows"][i][2], spec)
                items.append({k: torch.from_numpy(data[v]).long() for k, v in
                              (("atoms","atom_types"), ("charges","charges"), ("bonds","bond_types"))})
            target = collate_graphs(items)
            mu, _ = model.encoder(target, from_canonical_cache=True)
            for name in records:
                if name == "target":
                    graph, logits = target, [None] * len(indices)
                else:
                    codes = mu if name == "reconstruction" else z[start:start + len(indices)]
                    decoded = model.decode(codes)
                    graph, logits = decoded.harden(spec), decoded.symmetric_bond_logits.numpy()
                for index, a, c, b, l in zip(indices, graph.atoms.numpy(), graph.charges.numpy(),
                                             graph.bonds.numpy(), logits, strict=True):
                    item = profile(a, c, b, spec, l)
                    slots = np.flatnonzero(a)
                    i, j = np.nonzero(np.triu(b, 1))
                    raw = {"nodes": np.column_stack((slots, a[slots], c[slots])).tolist(),
                           "edges": np.column_stack((i, j, b[i, j])).tolist()}
                    stream.write(json.dumps({"population": name, "index": len(records[name]),
                                             "paired_panel_row": index, **raw, **item}) + "\n")
                    records[name].append(item)
    result = {"cursor": {k: payload["continuation"].get(k) for k in ("epoch", "row", "updates")},
              "checkpoint_sha256": file_sha256(args.checkpoint), "panel_sha256": file_sha256(args.panel),
              "seed": args.seed, "torch": torch.__version__, "rdkit": rdBase.rdkitVersion,
              "populations": {k: describe(v) for k,v in records.items()}}
    write_json(args.output/"summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
