"""CPU feasibility audit of sparse ring-topology targets on saved raw Gaussian graphs."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import time

import numpy as np
from rdkit import Chem, rdBase

from latentmol.chemistry import _label_molecule, inspect_graph
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import GraphSpec


AMIDE = Chem.MolFromSmarts("[CX3](=[OX1])-[NX3,NX4+]")
SINGLE = Chem.BondType.SINGLE


def ring_sets(mol):
    return [tuple(ring) for ring in Chem.GetSymmSSSR(mol)]


def ring_edges(ring):
    return {tuple(sorted(p)) for p in zip(ring, ring[1:] + ring[:1])}


def flags(rings):
    sizes = [len(r) for r in rings]
    return dict(triangle=3 in sizes, four=4 in sizes, five_or_six=5 in sizes or 6 in sizes, six=6 in sizes,
                eight_or_more=any(n >= 8 for n in sizes), twelve_or_more=any(n >= 12 for n in sizes))


def counters(rings):
    lengths = Counter(map(len, rings))
    return (lengths[3], sum(n for size, n in lengths.items() if size >= 8),
            lengths[4], lengths[7], sum(n for size, n in lengths.items() if size >= 12))


def chemistry_sets(mol):
    aromatics = {frozenset(r) for r in ring_sets(mol)
                 if all(mol.GetBondBetweenAtoms(i, j).GetIsAromatic() for i, j in ring_edges(r))}
    return aromatics, set(mol.GetSubstructMatches(AMIDE))


def assess(mol, original, protected, chemistry):
    rings = ring_sets(mol)
    counts = counters(rings)
    good = {frozenset(r) for r in rings if len(r) in (5, 6)}
    if (any(a > b for a, b in zip(counts, original)) or sum(counts[:2]) >= sum(original[:2])
            or not good or not protected.issubset(good)):
        return None
    if (Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE
            or len(Chem.GetMolFrags(mol)) != 1 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    aromatics, amides = chemistry_sets(mol)
    if not (chemistry[0].issubset(aromatics) and chemistry[1].issubset(amides)):
        return None
    return dict(**flags(rings), aromatic=bool(aromatics), amide=bool(amides),
                created_five_or_six=len(good - protected), removed_unusual=sum(original[:2]) - sum(counts[:2]))


def proposals(mol, bad, arm):
    """Deterministic single cuts, chords of long rings, or endpoint-preserving single-bond swaps."""
    cuts = sorted({edge for ring in bad for edge in ring_edges(ring)
                   if mol.GetBondBetweenAtoms(*edge).GetBondType() == SINGLE})
    if arm == "cut":
        return [(edge, None) for edge in cuts]
    by_rank = []
    for cut in cuts if arm == "rewire" else [None]:
        opened = Chem.RWMol(mol)
        if cut:
            opened.RemoveBond(*cut)
        distances = Chem.GetDistanceMatrix(opened, force=True)
        left, right = np.where(np.triu((distances == 4) | (distances == 5), 1))
        for i, j in zip(left.tolist(), right.tolist()):
            if cut and ((i, j) == cut or not (i in cut or j in cut)):
                continue
            if not cut and not any(len(r) >= 8 and i in r and j in r for r in bad):
                continue
            # Prefer closures with room for a single bond and a six-membered cycle.
            if any(mol.GetAtomWithIdx(k).GetTotalNumHs() == 0 and (not cut or k not in cut) for k in (i, j)):
                continue
            by_rank.append((distances[i, j] != 5, cut, (i, j)))
    return [(cut, added) for _, cut, added in sorted(by_rank)]


def graph(row, spec):
    a = np.zeros(spec.max_nodes, int)
    c = np.zeros_like(a)
    b = np.zeros((spec.max_nodes, spec.max_nodes), int)
    for i, atom, charge in row["nodes"]:
        a[i], c[i] = atom, charge
    for i, j, bond in row["edges"]:
        b[i, j] = b[j, i] = bond
    return a, c, b


def audit(row, spec, budget, rewire_logp=None):
    a, c, b = graph(row, spec)
    slots = np.flatnonzero(a)
    verified = inspect_graph(a, c, b, spec)
    if verified.smiles != row["smiles"]:
        raise ValueError("Raw graph does not reproduce the saved strict readout")
    raw = dict(index=row["index"], atoms=len(slots), valid=bool(verified.smiles), failure=verified.failure_reason)
    if not verified.smiles:
        return raw
    mol = _label_molecule(a, c, b[np.ix_(slots, slots)], spec, slots)
    Chem.SanitizeMol(mol)
    rings = ring_sets(mol)
    bad = [r for r in rings if len(r) == 3 or len(r) >= 8]
    protect = {frozenset(r) for r in rings if len(r) in (5, 6)}
    chemistry = chemistry_sets(mol)
    raw.update(baseline=dict(**flags(rings), aromatic=bool(chemistry[0]), amide=bool(chemistry[1])),
               eligible=bool(bad), arms={})
    if not bad:
        return raw
    for arm in ("cut", "close", "rewire") if rewire_logp is None else ("rewire",):
        start = time.perf_counter()
        candidates = proposals(mol, bad, arm)
        tested = candidates[:budget]
        if rewire_logp is not None:
            def edit_cost(move):
                cut, add = move
                return -(rewire_logp[slots[cut[0]], slots[cut[1]], 0]
                         + rewire_logp[slots[add[0]], slots[add[1]], spec.bond_names.index("SINGLE")])
            tested = sorted(tested, key=edit_cost)
        result = None
        tries = 0
        for cut, added in tested:
            tries += 1
            changed = Chem.RWMol(mol)
            if cut:
                changed.RemoveBond(*cut)
            if added:
                changed.AddBond(*added, SINGLE)
            outcome = assess(changed, counters(rings), protect, chemistry)
            if outcome:
                # Verify the emitted slot-level sparse target through the same strict production readout.
                new = b.copy()
                for edge, value in ((cut, 0), (added, spec.bond_names.index("SINGLE"))):
                    if edge:
                        i, j = slots[list(edge)]
                        new[i, j] = new[j, i] = value
                checked = inspect_graph(a, c, new, spec)
                if not checked.smiles:
                    raise AssertionError("A proposed valid edit failed the shared-graph round trip")
                result = dict(flags=outcome, smiles=checked.smiles,
                              cut=slots[list(cut)].tolist() if cut else None,
                              added=slots[list(added)].tolist() if added else None)
                break
        raw["arms"][arm] = dict(candidates=len(candidates), tried=tries, accepted=result,
                                seconds=time.perf_counter() - start)
    return raw


def summarize(rows, arm):
    counts = Counter()
    accepted = Counter()
    timers = []
    for row in rows:
        if not row["valid"]:
            continue
        current = row["baseline"]
        if row["eligible"]:
            choice = row["arms"][arm]
            timers.append(choice["seconds"])
            if edit := choice["accepted"]:
                current = edit["flags"]
                accepted["edits"] += 1
                accepted["created_five_or_six"] += int(current["created_five_or_six"] > 0)
                accepted["all_unusual_removed"] += int(not (current["triangle"] or current["eight_or_more"]))
        counts.update({k: int(current[k]) for k in row["baseline"]})
    return dict(counts=dict(counts), accepted=dict(accepted), seconds=sum(timers),
                median_ms_per_eligible=1000 * statistics.median(timers) if timers else None)


def gradient_probe(checkpoint, summary_file, samples, rows, budget, count=32):
    """Re-decode matched rows on CPU and differentiate two sparse edge labels per verified rewire."""
    import torch
    from torch.nn import functional as F
    from latentmol.evaluation.evaluate import load_model

    saved = json.loads(summary_file.read_bytes())
    if saved["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("Panel and decoder checkpoint differ")
    torch.set_num_threads(2)
    spec = GraphSpec()
    model, _ = load_model(checkpoint, "cpu", spec)
    model.eval()
    z = torch.randn(len(rows), model.config.latent, generator=torch.Generator().manual_seed(saved["seed"]))
    matched = [r for r in rows if r.get("eligible") and r["arms"]["rewire"]["accepted"]][:count]
    groups = {"first_valid": [], "probability_ranked": []}
    for row in matched:
        i = row["index"]
        x = z[i:i + 1].requires_grad_(True)
        out = model.decode(x)
        raw = out.harden(spec)
        a, c, b = graph(samples[i], spec)
        if (not np.array_equal(raw.atoms[0].numpy(), a) or not np.array_equal(raw.bonds[0].numpy(), b)
                or not np.array_equal(raw.charges[0].numpy()[a > 0], c[a > 0])):
            raise ValueError("CPU decoder failed to reproduce the archived graph for the gradient probe")
        logits = out.symmetric_bond_logits[0]
        ranked = audit(samples[i], spec, budget, logits.detach().log_softmax(-1).numpy())
        for name, candidate in (("first_valid", row), ("probability_ranked", ranked)):
            target = candidate["arms"]["rewire"]["accepted"]
            remove, add = target["cut"], target["added"]
            selected = torch.stack([logits[tuple(remove)], logits[tuple(add)]])
            labels = torch.tensor([0, spec.bond_names.index("SINGLE")])
            ce = F.cross_entropy(selected, labels)
            gz, gw = torch.autograd.grad(ce, (x, model.decoder.bond_head[-1].weight),
                                         retain_graph=name == "first_valid")
            p = selected.detach().softmax(-1)
            groups[name].append(dict(ce=float(ce.detach()), cut_target_p=float(p[0, labels[0]]),
                                     closure_target_p=float(p[1, labels[1]]), latent_norm=float(gz.norm()),
                                     bond_head_weight_norm=float(gw.norm())))
    reports = {}
    for name, metrics in groups.items():
        if not metrics or not all(np.isfinite(list(m.values())).all() for m in metrics):
            raise ValueError("Gradient probe has missing or nonfinite observations")
        reports[name] = dict(graphs=len(metrics), nonzero_latent=sum(m["latent_norm"] > 0 for m in metrics),
                             nonzero_head=sum(m["bond_head_weight_norm"] > 0 for m in metrics),
                             median={k: statistics.median(m[k] for m in metrics) for k in metrics[0]})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--size-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path, help="Optionally check sparse CE gradients on 32 reproduced priors.")
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.graphs, "rt") as stream:
        samples = [r for line in stream if (r := json.loads(line))["population"] == "prior"]
    start = time.perf_counter()
    with rdBase.BlockLogs(), gzip.open(args.out / "edits.jsonl.gz", "xt") as output:
        rows = []
        for sample in samples:
            row = audit(sample, GraphSpec(), args.budget)
            rows.append(row)
            output.write(json.dumps(row) + "\n")
    valid = [r for r in rows if r["valid"]]
    eligible = [r for r in valid if r["eligible"]]
    base = Counter(k for r in valid for k, v in r["baseline"].items() if v)
    matched = json.loads(args.size_reference.read_bytes())["graph_diagnostic_prior"]
    if matched["valid"] != len(valid) or any(base[k] != n for k, n in matched["counts"].items()):
        raise ValueError("Size reference used different graphs or ring definitions")
    arms = {arm: summarize(rows, arm) for arm in ("cut", "close", "rewire")}
    min_edits = Counter()
    for row in eligible:
        arms_row = row["arms"]
        single = arms_row["cut"]["accepted"] or arms_row["close"]["accepted"]
        min_edits[1 if single else 2 if arms_row["rewire"]["accepted"] else 0] += 1
    report = dict(graphs_sha256=file_sha256(args.graphs), reference_sha256=file_sha256(args.size_reference),
                  rdkit=rdBase.rdkitVersion, attempts=len(rows), valid=len(valid), eligible=len(eligible),
                  candidate_cap_per_arm=args.budget, baseline=dict(base), arms=arms,
                  smallest_found_edits=min_edits, elapsed_seconds=time.perf_counter()-start,
                  size_matched_training_percentage=matched["expected_real_pct_at_matched_sizes"])
    if args.checkpoint:
        with rdBase.BlockLogs():
            report["gradient_probe"] = gradient_probe(args.checkpoint, args.graphs.with_name("summary.json"),
                                                      samples, rows, args.budget)
    write_json(args.out / "summary.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
