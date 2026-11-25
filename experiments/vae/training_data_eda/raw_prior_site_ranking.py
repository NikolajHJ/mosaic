"""Does spontaneous motif completion on nearby raw VAE draws follow the parent's bond scores?"""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from rdkit import rdBase
import torch

from experiments.vae.training_data_eda.aromatic_encodings_probe import patterns, ring_logprobs, select_sites
from latentmol.chemistry import _label_molecule
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.motif_completion import AMIDE, _completion_targets, _roundtrips, _valid
from latentmol.sampling import sample_hard_graph


def molecule(atoms, charges, bonds, spec):
    active = np.flatnonzero(atoms)
    if not len(active):
        return None, active
    mol = _label_molecule(atoms, charges, bonds[np.ix_(active, active)], spec, active)
    return (mol if _valid(mol) and _roundtrips(mol) else None), active


def amide_paths(mol, active):
    """Return O, C, N triples in actual atom slots; require all three motif atoms."""
    return {(int(active[o]),int(active[c]),int(active[n])) for c,o,n in mol.GetSubstructMatches(AMIDE)}


def cyclic_ureas(mol, active):
    """O, carbonyl C, both neutral N slots; both C–N bonds must lie on rings."""
    from experiments.vae.training_data_eda.motif_family_audit import QUERIES
    paths = set()
    for o, c, first, second in mol.GetSubstructMatches(QUERIES["urea"]):
        if all(mol.GetBondBetweenAtoms(c, n).IsInRing() for n in (first, second)):
            ns = sorted((int(active[first]), int(active[second])))
            paths.add((int(active[o]), int(active[c]), *ns))
    return paths


def cyclic_urea_candidates(decoded, base, spec, start):
    """Use one site per raw parent, with verified OH→carbonyl edit at those same four atoms."""
    from experiments.vae.training_data_eda.motif_family_audit import PRECURSOR
    logits = decoded.symmetric_bond_logits
    single, double = map(spec.bond_names.index, ("SINGLE", "DOUBLE"))
    sites = []
    for row, (a, c, b) in enumerate(zip(base.atoms.numpy(), base.charges.numpy(), base.bonds.numpy(), strict=True)):
        parent, active = molecule(a, c, b, spec)
        if parent is None or cyclic_ureas(parent, active):
            continue
        for o, carbon, n1, n2 in parent.GetSubstructMatches(PRECURSOR):
            oxygen, center = int(active[o]), int(active[carbon])
            nitrogen = sorted((int(active[n1]), int(active[n2])))
            slots = [oxygen, center, *nitrogen]
            edited = b.copy()
            edited[oxygen, center] = edited[center, oxygen] = double
            endpoint, active = molecule(a, c, edited, spec)
            if endpoint is None or tuple(slots) not in cyclic_ureas(endpoint, active):
                continue
            score = logits[row, oxygen, center, double] - logits[row, oxygen, center, single]
            sites.append(dict(parent=start+row, kind="cyclic_urea", slots=slots, score=float(score),
                              elements=a[slots].tolist(), charges=c[slots].tolist(), neighbors=[], categorical=[]))
            break
    return sites


def sample_events(hard, spec, site):
    a, c, b = (x[0].numpy() for x in (hard.atoms, hard.charges, hard.bonds))
    mol, active = molecule(a, c, b, spec)
    slots = site["slots"]
    same = a[slots].tolist() == site["elements"] and c[slots].tolist() == site["charges"]
    complete = False
    if mol is not None and same:
        if site["kind"] == "amide":
            complete = tuple(slots) in amide_paths(mol, active)
        elif site["kind"] == "cyclic_urea":
            complete = tuple(slots) in cyclic_ureas(mol, active)
        else:
            index = {int(s):i for i,s in enumerate(active)}
            edges = [(index[s],index[t]) for s,t in zip(slots,slots[1:]+slots[:1])]
            bonds = [mol.GetBondBetweenAtoms(i,j) for i,j in edges]
            complete = all(b is not None and b.GetIsAromatic() for b in bonds)
    return dict(valid=mol is not None, same_site_atoms_charges=same, completed=complete)


def sites_for_block(decoded, base, spec, start):
    logits = decoded.symmetric_bond_logits
    rings, _, _ = select_sites(base, spec)
    sites = []
    if rings:
        scores = ring_logprobs(logits, rings, patterns(spec)).logsumexp(-1).tolist()
        for ring, score in zip(rings, scores, strict=True):
            row = ring[0][0]
            sites.append((row, "aromatic", [i for _,i,_ in ring], score))
    for row, (a,c,b) in enumerate(zip(base.atoms.numpy(), base.charges.numpy(), base.bonds.numpy(), strict=True)):
        _, edges = _completion_targets(a, c, b, spec, False, True)
        if not edges:
            continue
        k,o,label = edges[0]  # completion teacher returns carbon, oxygen, DOUBLE
        assert spec.bond_names[label] == "DOUBLE"
        # Verify the proposed edit creates a motif at this exact carbon/oxygen;
        # select one nitrogen by slot number if several bind that same carbon.
        alt = b.copy()
        alt[o,k] = alt[k,o] = label
        mol, active = molecule(a, c, alt, spec)
        matches = sorted((oi,ci,ni) for oi,ci,ni in amide_paths(mol, active) if oi == o and ci == k) if mol else []
        if matches:
            score = logits[row,o,k,label] - logits[row,o,k,spec.bond_names.index("SINGLE")]
            sites.append((row, "amide", list(matches[0]), float(score)))
    return [dict(parent=start+r, kind=kind, slots=slots, score=score,
                 elements=base.atoms[r,slots].tolist(), charges=base.charges[r,slots].tolist(),
                 neighbors=[], categorical=[]) for r,kind,slots,score in sites]


def summarize(sites, kinds=("aromatic", "amide")):
    out = {}
    for kind in kinds:
        candidates = sorted([s for s in sites if s["kind"] == kind], key=lambda x:(-x["score"],x["parent"]))
        mid = (len(candidates)+1)//2
        out[kind] = {}
        for group, rows in [("high", candidates[:mid]),("low", candidates[mid:])]:
            metrics = dict(sites=len(rows), mean_score=float(np.mean([r["score"] for r in rows])) if rows else None)
            for mode in ("neighbors","categorical"):
                cm = Counter(trials=0, valid=0, same_site_atoms_charges=0, completed=0, sites_with_completion=0)
                for row in rows:
                    cm["sites_with_completion"] += any(ev["completed"] for ev in row[mode])
                    for ev in row[mode]:
                        cm["trials"] += 1
                        cm.update(ev)
                metrics[mode] = dict(cm)
            out[kind][group] = metrics
    return out


@torch.inference_mode()
def run(args):
    if min(args.draws,args.batch,args.threads)<1 or not np.isfinite(args.radius) or args.radius<=0:
        raise ValueError("Positive finite panel and perturbation parameters required")
    torch.set_num_threads(args.threads)
    model, _ = load_model(args.checkpoint, "cpu")
    model.eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    codes = torch.randn(args.draws, model.config.latent, generator=torch.Generator().manual_seed(args.seed))
    directions = torch.randn(2, model.config.latent, generator=torch.Generator().manual_seed(args.seed+1))
    directions = torch.nn.functional.normalize(directions, dim=-1)*args.radius
    directions = torch.stack([directions[0],-directions[0],directions[1],-directions[1]])
    gen = torch.Generator().manual_seed(args.seed+2)
    sites = []
    identify = cyclic_urea_candidates if args.family == "cyclic_urea" else sites_for_block
    kinds = ("cyclic_urea",) if args.family == "cyclic_urea" else ("aromatic","amide")
    with rdBase.BlockLogs():
        for start in range(0,args.draws,args.batch):
            z = codes[start:start+args.batch]
            decoded = model.decode(z)
            sites.extend(identify(decoded,decoded.harden(model.spec),model.spec,start))
        by_parent = defaultdict(list)
        for site in sites:
            by_parent[site["parent"]].append(site)
        for start in range(0,len(by_parent),args.batch):
            ids = sorted(by_parent)[start:start+args.batch]
            z = codes[ids]
            decoded = model.decode(z)
            for shift in directions:
                nearby = model.decode(z + shift).harden(model.spec)
                sampled = sample_hard_graph(decoded,model.spec,generator=gen)
                for j,key in enumerate(ids):
                    for source,mode in ((nearby,"neighbors"),(sampled,"categorical")):
                        one = type(source)(source.atoms[j:j+1],source.charges[j:j+1],source.bonds[j:j+1])
                        for site in by_parent[key]:
                            site[mode].append(sample_events(one,model.spec,site))
    with gzip.open(args.output/"raw-sites.jsonl.gz","xt",encoding="utf8") as f:
        for s in sites:
            f.write(json.dumps(s,allow_nan=False)+"\n")
    report = dict(checkpoint_sha256=file_sha256(args.checkpoint),raw_gaussian_draws=args.draws,
                  seed=args.seed, radius=args.radius, directions=4, eligible_unique_parents=len(by_parent),
                  family=args.family, groups=summarize(sites,kinds),
                  high_low="Rank sites in each motif separately, then split equally; ties by parent ID.",
                  scoring=("Site C–O DOUBLE minus SINGLE logit." if args.family == "cyclic_urea" else
                           "Conditional ring log probability of three complete assignments; amide raw double minus single logit."),
                  perturbation="Two Gaussian unit directions shared across all sites, each with its negative.",
                  event="Entire molecule valid and RDKit motif on the original slots with matching site atom labels and charges.")
    write_json(args.output/"summary.json",report)
    print(json.dumps(report))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=2026112573)
    parser.add_argument("--draws",type=int,default=512)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--radius",type=float,default=1.)
    parser.add_argument("--threads",type=int,default=3)
    parser.add_argument("--family",choices=("aromatic_amide","cyclic_urea"),default="aromatic_amide")
    run(parser.parse_args())
