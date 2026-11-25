"""Can per-bond VAE probabilities recover hidden aromatic sites in held-out molecules?"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
import torch

from experiments.vae.training_data_eda.aromatic_encodings_probe import patterns
from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.chemistry import encode_canonical_smiles, MoleculeRejected
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import HardGraphBatch


def _skeleton(graph, spec):
    """A queryable element+adjacency graph in decoder slot order, with every edge single."""
    a, q, b = (t.detach().cpu().numpy()[0] for t in (graph.atoms, graph.charges, graph.bonds))
    active = np.flatnonzero(a)
    mol = Chem.RWMol()
    for slot in active:
        atom = Chem.Atom(spec.atomic_numbers[a[slot]])
        atom.SetFormalCharge(spec.formal_charges[q[slot]])
        mol.AddAtom(atom)
    for i,j in zip(*np.triu(b[np.ix_(active,active)] != 0, 1).nonzero()):
        mol.AddBond(int(i),int(j),Chem.BondType.SINGLE)
    mol.UpdatePropertyCache(strict=False)
    return mol.GetMol(), active, q[active]


def aligned_output_sites(input_graph, output_graph, sites, spec, *, cap=256):
    """Map an edge or ordered cycle from a single input to unambiguous output slots.

    Return (locations, reason). locations is parallel to sites; None denotes
    ambiguity. A site admits rotations/reversals of the same cycle and both
    orientations of the same edge. No true bond order or motif labels enter.
    """
    src, ss, sq = _skeleton(input_graph, spec)
    dest, ds, dq = _skeleton(output_graph, spec)
    missing = [None] * len(sites)
    if (len(ss) != len(ds) or src.GetNumBonds() != dest.GetNumBonds() or not len(ss)):
        return missing, "skeleton_mismatch"
    matches = dest.GetSubstructMatches(src, uniquify=False, useChirality=False, maxMatches=cap+1)
    if len(matches) > cap:
        return missing, "mapping_cap"
    matches = [m for m in matches if np.array_equal(sq, dq[list(m)])]
    if not matches:
        return missing, "skeleton_mismatch"
    positions = {int(slot): i for i,slot in enumerate(ss)}
    mapped = []
    for site in sites:
        choices = [tuple(int(ds[m[positions[i]]]) for i in site) for m in matches]
        edge_sets = {frozenset(tuple(sorted(pair)) for pair in zip(c, c[1:]+c[:1])) for c in choices}
        mapped.append(choices[0] if len(edge_sets)==1 else None)
    return mapped, "ambiguous_site" if any(m is None for m in mapped) else "ok"


def _eligible_rings(mol):
    info = mol.GetRingInfo()
    for ring in info.AtomRings():
        if len(ring) != 6 or any(info.NumAtomRings(i) != 1 for i in ring):
            continue
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (6,7) or mol.GetAtomWithIdx(i).GetFormalCharge() for i in ring):
            continue
        yield tuple(ring)


def _set_orders(mol, ring, orders):
    for i in ring:
        mol.GetAtomWithIdx(i).SetIsAromatic(orders[0] == Chem.BondType.AROMATIC)
    for i,j,order in zip(ring,ring[1:] + ring[:1],orders,strict=True):
        bond = mol.GetBondBetweenAtoms(i,j)
        bond.SetIsAromatic(order == Chem.BondType.AROMATIC)
        bond.SetBondType(order)


def _checked_smiles(mol):
    if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def masked_example(smiles, spec):
    """Hide *all* simple wholly aromatic/saturated six-C/N rings in identical single-bond format."""
    original = Chem.MolFromSmiles(smiles)
    molecule = Chem.Mol(original)
    truth = {}
    single, double, aromatic = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.AROMATIC
    for ring in _eligible_rings(original):
        orders = [original.GetBondBetweenAtoms(i,j).GetBondType() for i,j in zip(ring,ring[1:]+ring[:1])]
        if all(b == aromatic for b in orders) or all(b == single for b in orders):
            truth[frozenset(ring)] = orders[0] == aromatic
            _set_orders(molecule, ring, [single]*6)
    if not truth:
        return None, "no_reference_site"
    canonical = _checked_smiles(molecule)
    if canonical is None:
        return None, "invalid_corruption"
    # RDKit reports the writer's traversal, which is the parser's slot order.
    order = json.loads(molecule.GetProp("_smilesAtomOutputOrder"))
    try:
        encoded = encode_canonical_smiles(canonical, spec)
    except MoleculeRejected:
        return None, "unrepresentable_corruption"
    clean = Chem.MolFromSmiles(encoded["smiles"])
    if len(order) != clean.GetNumAtoms() or len(set(order)) != len(order):
        raise ValueError("Invalid canonical traversal map")
    for bond in molecule.GetBonds():
        i,j = (order.index(a) for a in (bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
        if clean.GetBondBetweenAtoms(i,j).GetBondType() != bond.GetBondType():
            raise ValueError("Canonical map changed a bond")
    if any(clean.GetAtomWithIdx(i).GetAtomicNum() != molecule.GetAtomWithIdx(old).GetAtomicNum()
           or clean.GetAtomWithIdx(i).GetFormalCharge() != molecule.GetAtomWithIdx(old).GetFormalCharge()
           for i,old in enumerate(order)):
        raise ValueError("Canonical map changed an element or charge")
    targets = []
    for ring in _eligible_rings(clean):
        if not all(clean.GetBondBetweenAtoms(i,j).GetBondType() == single for i,j in zip(ring,ring[1:]+ring[:1])):
            continue
        # Selection depends on masked chemistry only. All equivalent target bond patterns
        # must aromatize this specific ring and give the same whole-molecule canonical identity.
        forms = []
        for values in ([aromatic]*6, [single,double]*3, [double,single]*3):
            edited = Chem.Mol(clean)
            _set_orders(edited, ring, values)
            target = _checked_smiles(edited)
            if target is None or not all(edited.GetBondBetweenAtoms(i,j).GetIsAromatic() for i,j in zip(ring,ring[1:]+ring[:1])):
                break
            forms.append(target)
        if len(forms)==3 and len(set(forms))==1:
            key = frozenset(order[i] for i in ring)
            targets.append((ring, truth[key]))
    return (canonical,targets), "ok" if targets else "no_verified_candidates"


def _auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    gap=np.asarray(pos)[:,None]-np.asarray(neg)[None,:]
    return (np.count_nonzero(gap>0) + .5*np.count_nonzero(gap==0))/gap.size


def summarize(rows, key):
    pos=[r[key] for r in rows if r["was_aromatic"]]
    neg=[r[key] for r in rows if not r["was_aromatic"]]
    order=sorted(rows,key=lambda r:(-r[key],r["index"],r["input_slots"]))
    k=min(10,len(order))  # Set prospectively for every score, without fitting a threshold.
    groups=defaultdict(list)
    for r in rows:
        groups[r["index"]].append(r)
    within=[]
    for sites in groups.values():
        a=[r[key] for r in sites if r["was_aromatic"]]
        b=[r[key] for r in sites if not r["was_aromatic"]]
        if a and b:within.append(_auc(a,b))
    return dict(aromatic=len(pos), saturated=len(neg), auc=_auc(pos,neg),
                aromatic_fraction=len(pos)/len(rows) if rows else None,
                top_k=k, top_k_aromatics=sum(r["was_aromatic"] for r in order[:k]),
                mixed_molecules=len(within), mean_within_molecule_auc=float(np.mean(within)) if within else None,
                median_positive=float(np.median(pos)) if pos else None, median_negative=float(np.median(neg)) if neg else None)


def _row(graph, i):
    return HardGraphBatch(*(x[i:i+1] for x in (graph.atoms,graph.charges,graph.bonds)))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","panel","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model,_=load_model(args.checkpoint,"cpu")
    model.eval().requires_grad_(False)
    rows=[];examples=[];counts=Counter()
    with rdBase.BlockLogs():
        # Benzene and cyclohexane differ at the hidden labels, but supply the same model input.
        aromatic,_=masked_example("c1ccccc1",model.spec)
        saturated,_=masked_example("C1CCCCC1",model.spec)
        assert aromatic[0]==saturated[0] and aromatic[1][0][1] and not saturated[1][0][1]
        for item in json.loads(args.panel.read_bytes())["rows"]:
            idx, smiles = counts["molecules"],item[2]
            counts["molecules"]+=1
            example,reason=masked_example(smiles,model.spec)
            counts["source_"+reason]+=1
            if example and example[1]:
                examples.append((idx,smiles,*example))
        for first in range(0,len(examples),8):
            batch=examples[first:first+8]
            hard=encode([x[2] for x in batch],model.spec)
            mean,_=model.encoder(hard,from_canonical_cache=True)
            pred=model.decode(mean)
            realized=pred.harden(model.spec)
            logits=pred.symmetric_bond_logits
            for j,(idx,source,masked,targets) in enumerate(batch):
                queried=[r for r,_ in targets]
                mapped,reason=aligned_output_sites(_row(hard,j),_row(realized,j),queried,model.spec)
                counts["map_"+reason]+=1
                for (ring,label),output_slots in zip(targets,mapped,strict=True):
                    counts["verified_"+("positive" if label else "negative")]+=1
                    if output_slots is None:continue
                    edges=[(j,i,k) for i,k in zip(output_slots,output_slots[1:]+output_slots[:1])]
                    local=logits[tuple(torch.tensor(edges).T)].double().log_softmax(-1)
                    low=local[:,model.spec.bond_names.index("SINGLE")].sum().item()
                    forms=patterns(model.spec)
                    log_alts=torch.stack([local.gather(1,t[:,None]).sum() for t in forms])
                    log_event=log_alts.logsumexp(0).item()
                    result=dict(index=idx,source=source,masked=masked,input_slots=ring,output_slots=output_slots,
                                was_aromatic=bool(label), aromatic_vs_saturated_log_odds=log_event-low,
                                full_event_log_probability=log_event,
                                best_edge_aromatic_probability=local[:,model.spec.bond_names.index("AROMATIC")].max().exp().item())
                    rows.append(result)
    scores={k:summarize(rows,k) for k in ("aromatic_vs_saturated_log_odds","full_event_log_probability","best_edge_aromatic_probability")}
    report=dict(checkpoint_sha256=file_sha256(args.checkpoint), panel_sha256=file_sha256(args.panel),
                counts=dict(counts), scores=scores, same_masked_benzene_cyclohexane=True, rows=rows)
    with args.output.open("x",encoding="utf8") as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(counts=counts,scores=scores),indent=2))


if __name__=="__main__":
    main()
