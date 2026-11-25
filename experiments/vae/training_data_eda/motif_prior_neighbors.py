"""Evaluate correlated Gaussian neighbors of known strictly valid motif-positive prior decodes on CPU."""

import argparse
import gzip
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
from rdkit import Chem, rdBase
import torch

from latentmol.chemistry import _label_molecule, inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.motif_completion import AMIDE, _roundtrips, _valid
from latentmol.representation import GraphSpec


def sites(a, c, b, spec):
    slots = np.flatnonzero(a)
    if not len(slots):
        return {"aromatic": set(), "amide": set()}
    mol = _label_molecule(a, c, b[np.ix_(slots,slots)], spec, slots)
    if not _valid(mol) or not _roundtrips(mol):
        return {"aromatic": set(), "amide": set()}
    def typed(i):
        k = int(slots[i])
        return k, int(a[k]), int(c[k])
    aromatic = set()
    for ring in mol.GetRingInfo().AtomRings():
        edges = list(zip(ring,ring[1:] + ring[:1]))
        if all(mol.GetBondBetweenAtoms(i,j).GetIsAromatic() for i,j in edges):
            aromatic.add(frozenset(tuple(sorted((typed(i), typed(j)))) for i,j in edges))
    return dict(aromatic=aromatic, amide={tuple(typed(i) for i in match) for match in mol.GetSubstructMatches(AMIDE)})


def restore(r, spec):
    atoms = np.zeros(spec.max_nodes, int)
    charges = np.full_like(atoms, spec.formal_charges.index(0))
    bonds = np.zeros((len(atoms), len(atoms)), int)
    for i, a, c in r["nodes"]:
        atoms[i], charges[i] = a, c
    for i, j, e in r["edges"]:
        bonds[i,j] = bonds[j,i] = e
    return atoms, charges, bonds


def compare(base, baseline, raw, source_sites, spec):
    a, c, b = raw
    aa, cc, bb = baseline
    atoms = int((a != aa).sum())
    charges = int(((c != cc) & (a != 0) & (aa != 0)).sum())
    bonds = int(np.triu(b != bb, 1).sum())
    graph = inspect_graph(a, c, b, spec)
    observed = sites(a,c,b,spec)
    smiles = graph.smiles if graph.smiles and Chem.MolFromSmiles(graph.smiles) is not None else None
    failure = "smiles_roundtrip_failure" if graph.smiles and not smiles else graph.failure_reason
    return dict(smiles=smiles, valid=smiles is not None, failure=failure,
                exact_smiles=smiles == base["smiles"], atom_edits=atoms, charge_edits=charges, bond_edits=bonds,
                edits=atoms+charges+bonds, occupied_delta=int((a != 0).sum() - (aa != 0).sum()),
                any_motif={g: bool(s) for g,s in observed.items()},
                same_typed_slot_site={g: bool(s & source_sites[g]) for g,s in observed.items()})


def summarize(rows, name):
    total = len(rows)
    return dict(draws=total, distinct_seed_count=len({r["index"] for r in rows}), valid=sum(r["valid"] for r in rows),
                exact_smiles=sum(r["exact_smiles"] for r in rows), exact_slots=sum(r["edits"] == 0 for r in rows),
                motif_anywhere=sum(r["any_motif"][name] for r in rows),
                same_site=sum(r["same_typed_slot_site"][name] for r in rows),
                same_site_changed=sum(r["edits"] > 0 and r["same_typed_slot_site"][name] for r in rows),
                valid_changed=sum(r["edits"] > 0 and r["valid"] for r in rows),
                distinct_changed_valid=sum(r["valid"] and not r["exact_smiles"] for r in rows),
                invalid_or_lost=sum(not r["same_typed_slot_site"][name] for r in rows),
                invalid_or_lost_within_two=sum(not r["same_typed_slot_site"][name] and r["edits"]<=2 for r in rows),
                changed_within_two=sum(0 < r["edits"] <= 2 for r in rows),
                within_two=sum(r["edits"] <= 2 for r in rows), within_four=sum(r["edits"] <= 4 for r in rows),
                median_edits=statistics.median(r["edits"] for r in rows),
                median_distance=statistics.median(r["latent_distance"] for r in rows))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--start-rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-group", type=int, default=24)
    p.add_argument("--directions", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026112231)
    args = p.parse_args()
    if min(args.per_group, args.directions) < 1:
        p.error("Panel and direction budgets must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    start = time.perf_counter()
    source = json.loads(args.start_rows.with_name("summary.json").read_bytes())
    if file_sha256(args.checkpoint) != source["checkpoint"]:
        raise ValueError("Reference raw prior results belong to another checkpoint")
    with gzip.open(args.start_rows, "rt") as stream:
        rows = [row for line in stream if (row := json.loads(line))["panel"] == "prior"]
    if [r["index"] for r in rows] != list(range(source["gaussian_evaluation"])):
        raise ValueError("Missing or reordered source prior indices")
    spec = GraphSpec()
    model, state = load_model(args.checkpoint, "cpu", spec)
    model.eval()
    original = torch.randn(len(rows), model.config.latent, generator=torch.Generator().manual_seed(source["seed"]+2))
    selected = {}
    for group in ("aromatic","amide"):
        candidates = [r["index"] for r in rows if r["valid"] and r[group]]
        selected[group] = sorted(random.Random(args.seed).sample(candidates,min(len(candidates),args.per_group)))
    chosen = sorted(set().union(*selected.values()))
    eps = torch.randn(len(chosen),args.directions,model.config.latent,generator=torch.Generator().manual_seed(args.seed+1))
    reproduced, mismatch = [], []
    raw_sources, source_sites = {}, {}
    rhos = (.995, .98, .95)
    events = []
    with rdBase.BlockLogs(), gzip.open(args.out/"neighbors.jsonl.gz","xt") as out:
        for block in torch.tensor(chosen).split(8):
            hard = model.decode(original[block]).harden(spec)
            for index, a, c, b in zip(block.tolist(),hard.atoms.numpy(),hard.charges.numpy(),hard.bonds.numpy()):
                baseline = restore(rows[index],spec)
                if not all(np.array_equal(x,y) for x,y in zip(baseline,(a,c,b))):
                    mismatch.append(index)
                    continue
                raw_sources[index] = baseline
                source_sites[index] = sites(*baseline,spec)
                if not all(bool(source_sites[index][key]) == bool(rows[index][key]) for key in selected):
                    raise ValueError("Replayed strict site extraction disagrees with the parent panel")
                reproduced.append(index)
        if any(not set(indices).intersection(reproduced) for indices in selected.values()):
            raise ValueError("No exact raw prior replay for one of the requested motifs")
        mapping = {index: pos for pos,index in enumerate(chosen)}
        seed_rows = [(index,d) for index in reproduced for d in range(args.directions)]
        for rho in rhos:
            for start_idx in range(0,len(seed_rows),8):
                group = seed_rows[start_idx:start_idx+8]
                inds = [index for index,d in group]
                noise = torch.stack([eps[mapping[index],d] for index,d in group])
                z = rho * original[inds] + math.sqrt(1-rho*rho) * noise
                dist = (z-original[inds]).norm(dim=-1).tolist()
                hard = model.decode(z).harden(spec)
                for (index,d), distance, a,c,b in zip(group,dist,hard.atoms.numpy(),hard.charges.numpy(),hard.bonds.numpy()):
                    item = compare(rows[index],raw_sources[index],(a,c,b),source_sites[index],spec)
                    event = dict(index=index,direction=d,rho=rho,latent_distance=distance,**item)
                    events.append(event)
                    out.write(json.dumps(event)+"\n")
    results = {key: {str(rho): summarize([r for r in events if r["index"] in indices and r["rho"]==rho],key)
                     for rho in rhos} for key,indices in selected.items()}
    report = dict(checkpoint_sha256=source["checkpoint"], source_sha256=file_sha256(args.start_rows),
                  epoch48_update=state["continuation"]["updates"], seed=args.seed,
                  source_population=len(rows), source_valid=sum(r["valid"] for r in rows),
                  source_motifs={k:sum(r[k] for r in rows) for k in selected}, selected=selected,
                  replayed_indices=reproduced, skipped_raw_mismatch=mismatch, directions=args.directions,
                  unique_neighbor_evaluations=len(events), results=results, seconds=time.perf_counter()-start)
    write_json(args.out/"summary.json",report)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
