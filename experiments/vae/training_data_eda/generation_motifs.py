"""Compare training motifs, held-out reconstructions and unconditional draws on shared inputs."""

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase

from chemistry_profile import GROUP_DEFINITIONS, molecule_profile
from latentmol.chemistry import encode_canonical_smiles, inspect_graph
from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec
from latentmol.sampling import sample_hard_graph


def flags(profile):
    return {"ring": int(profile["ring_count"]) > 0, "aromatic_ring": int(profile["aromatic_rings"]) > 0,
            "heterocycle": int(profile["heterocycles"]) > 0, "any_group": int(profile["group_families"]) > 0,
            **{name: bool(int(profile["fg_" + name])) for name in GROUP_DEFINITIONS}}


@lru_cache(maxsize=32768)
def motifs(smiles):
    return flags(molecule_profile(Chem.MolFromSmiles(smiles)))


def training_reference(path):
    counts, by_size = Counter(), defaultdict(Counter)
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            values = {**flags(row), "count": 1}
            counts.update(values)
            by_size[int(row["heavy_atoms"])].update(values)
    return counts, by_size


def cyclic(left, right, nodes):
    """Undirected cycle in the supplied edge subset, irrespective of sanitization."""
    parent = list(range(nodes))
    def root(k):
        while k != parent[k]:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for i, j in zip(left, right):
        a, b = root(i), root(j)
        if a == b:
            return True
        parent[a] = b
    return False


def summarize(rows, by_size):
    valid = [r for r in rows if r["smiles"]]
    keys = list(next(iter(by_size.values())))
    counts = Counter({k: 0 for k in keys if k != "count"})
    expected = Counter()
    for r in valid:
        counts.update(motifs(r["smiles"]))
        match = by_size[r["atoms"]]
        if not match["count"]:
            raise ValueError("Training reference lacks this heavy-atom count")
        for k in counts:
            expected[k] += match[k] / match["count"]
    exact = sum(bool(r["smiles"]) and r["smiles"] == r["target"] for r in rows)
    summary = dict(draws=len(rows), valid=len(valid), unique_valid=len({r["smiles"] for r in valid}),
                   exact=exact if rows[0]["target"] else None, valid_with_motif=counts,
                   expected_counts_given_valid_sizes=expected,
                   failures=Counter(r["failure"] for r in rows if not r["smiles"]),
                   valid_carbon_only=sum(r["carbon_only"] for r in valid),
                   all_carbon_only=sum(r["carbon_only"] for r in rows),
                   raw_with_aromatic_edge=sum(r["aromatic_edges"] > 0 for r in rows),
                   raw_with_aromatic_cycle=sum(r["aromatic_cycle"] for r in rows),
                   raw_with_any_cycle=sum(r["cycle"] for r in rows),
                   full_slots=sum(r["at_capacity"] for r in rows),
                   avg_aromatic_edge_count=np.mean([r["aromatic_edges"] for r in rows]),
                   avg_hard_atoms=np.mean([r["atoms"] for r in rows]))
    if rows[0]["target"]:
        summary["recovered_among_positives"] = {k: {
            "target_positive": sum(motifs(r["target"])[k] for r in rows),
            "recovered": sum(motifs(r["target"])[k] and motifs(r["smiles"])[k] for r in valid)
        } for k in counts}
    return summary


def inspect_batch(hard, source, spec):
    atoms, charges, bonds = (h.cpu().numpy() for h in (hard.atoms, hard.charges, hard.bonds))
    aromatic = spec.bond_names.index("AROMATIC")
    for idx, (a, c, b) in enumerate(zip(atoms, charges, bonds)):
        result = inspect_graph(a, c, b, spec)
        i, j = np.triu(b, 1).nonzero()
        ar = b[i, j] == aromatic
        occupied = np.flatnonzero(a)
        yield dict(smiles=result.smiles, failure=result.failure_reason, target=source[idx] if source else None,
                   atoms=len(occupied), at_capacity=len(occupied) == spec.max_nodes,
                   carbon_only=bool(len(occupied) and np.all(a[occupied] == spec.atomic_numbers.index(6))),
                   aromatic_edges=int(ar.sum()), aromatic_cycle=cyclic(i[ar], j[ar], len(a)),
                   cycle=cyclic(i, j, len(a)),
                   # Keep invalid graphs inspectable, with original occupied slots and label indices.
                   nodes=[[int(k), int(a[k]), int(c[k])] for k in occupied],
                   edges=np.column_stack((i, j, b[i, j])).tolist())


@torch.inference_mode()
def evaluate(model, items, source, prior, epsilon, seed, batch):
    rows = defaultdict(list)
    gen = torch.Generator().manual_seed(seed)
    for z in prior.split(batch):
        decoded = model.decode(z)
        rows["prior_argmax"].extend(inspect_batch(decoded.harden(model.spec), None, model.spec))
        sample = sample_hard_graph(decoded, model.spec, generator=gen, temperature=1.)
        rows["prior_sample"].extend(inspect_batch(sample, None, model.spec))
    for start in range(0, len(items), batch):
        stop = min(start + batch, len(items))
        mu, logvar = model.encoder(collate_graphs(items[start:stop]), from_canonical_cache=True)
        for key, z in (("reconstruction", mu), ("posterior_sample", mu + (.5*logvar).exp() * epsilon[start:stop])):
            hard = model.decode(z).harden(model.spec)
            rows[key].extend(inspect_batch(hard, source[start:stop], model.spec))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026112047)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    validation = json.loads(args.panel.read_bytes())
    assert validation["rdkit"] == rdBase.rdkitVersion
    spec = GraphSpec.from_dict(validation["spec"])
    source = [row[2] for row in validation["rows"]]
    items = []
    for sm in source:
        g = encode_canonical_smiles(sm, spec)
        items.append({k: torch.as_tensor(g[v]).long() for k, v in
                      (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})
    reference, by_size = training_reference(args.profiles)
    report = dict(seed=args.seed, batch=args.batch, torch=torch.__version__, rdkit=rdBase.rdkitVersion,
                  panel_sha256=file_sha256(args.panel), profiles_sha256=file_sha256(args.profiles),
                  spec=spec.to_dict(), training=reference,
                  held_out=dict(Counter(k for sm in source for k, v in motifs(sm).items() if v), count=len(source)),
                  checkpoints={})
    for checkpoint in args.checkpoint:
        model, payload = load_model(checkpoint, "cpu", spec)
        model.eval()
        d = model.config.latent
        prior = torch.randn(args.prior, d, generator=torch.Generator().manual_seed(args.seed))
        epsilon = torch.randn(len(items), d, generator=torch.Generator().manual_seed(args.seed+1))
        with rdBase.BlockLogs():
            rows = evaluate(model, items, source, prior, epsilon, args.seed+2, args.batch)
        result = {}
        with gzip.open(args.output/(checkpoint.stem+".jsonl.gz"), "xt", encoding="utf-8") as stream:
            for pop, records in rows.items():
                result[pop] = summarize(records, by_size)
                for idx, row in enumerate(records):
                    stream.write(json.dumps(dict(panel=pop, index=idx, **row))+"\n")
        cursor = payload["continuation"]
        report["checkpoints"][checkpoint.stem] = dict(sha256=file_sha256(checkpoint),
            cursor={k:cursor.get(k) for k in ("epoch", "updates", "row", "epoch_complete")}, populations=result)
        (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(checkpoint.stem, {k:(r["valid"], r["valid_with_motif"]["aromatic_ring"], r["raw_with_aromatic_cycle"])
                                for k,r in result.items()}, flush=True)


if __name__ == "__main__":
    main()
