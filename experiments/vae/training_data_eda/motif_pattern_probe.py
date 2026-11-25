"""Check fixed-panel selection, cost and bond-logit gradients for the batch pattern objective."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from rdkit import rdBase
import torch

from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.functional_groups import _motif_labels
from latentmol.guidance.motif_patterns import _pattern_objective, _pattern_sites
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.sampling import sample_hard_graph


def directional_readout(g, feats, sites, graph, spec):
    """Summarize an infinitesimal final-readout gradient step with final features held fixed."""
    _, offsite, blocked, _, rings, amides = sites
    active = g.abs().sum(-1) > 0
    grad_w = g[active].T @ feats[active]
    grad_b = g[active].sum(0)
    norm = (grad_w.square().sum() + grad_b.square().sum()).sqrt()
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("Nonfinite or detached final-readout gradient")
    results = {}
    for name, target_label, positive, mask in (
            ("aromatic", "AROMATIC", rings, offsite), ("amide", "DOUBLE", amides, blocked)):
        target = spec.bond_names.index(target_label)
        single = spec.bond_names.index("SINGLE")
        move = -(feats @ (grad_w[target] - grad_w[single]) + grad_b[target] - grad_b[single]) / norm
        where = torch.as_tensor(mask) & (graph.bonds == single)
        results[name] = dict(head_gradient_norm=float(norm), direct_global_target_bias=float(grad_b[target]),
                             offsite_single_edges=int(where.sum()), offsite_median=float(move[where].median()),
                             offsite_positive_fraction=float((move[where] > 0).float().mean()))
        edges = [t for site in positive for t in site]
        if edges:
            r, i, j = torch.tensor(edges).T
            results[name]["all_candidate_edges"] = len(edges)
            results[name]["candidate_median"] = float(move[r,i,j].median())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    spec = GraphSpec()
    model, state = load_model(args.checkpoint, "cpu", spec)
    model.eval()
    priors = torch.randn(96, model.config.latent, generator=torch.Generator().manual_seed(2026112215))
    capture = []
    def last_feature(_head, inputs):
        capture[:] = [inputs[0]]
    handle = model.decoder.bond_head[-1].register_forward_pre_hook(last_feature)
    events = []
    for batch, z in enumerate(priors.split(48)):
        features, parts = [], []
        with torch.no_grad():
            for micro in z.split(4):
                parts.append(model.decode(micro))
                features.append(capture[0])
        out = DecoderOutput(*(torch.cat([getattr(p, field) for p in parts]) for field in ("atoms","charges","bonds")))
        last = torch.cat(features)
        feat = (last + last.transpose(1,2)) / 2
        with torch.no_grad(), rdBase.BlockLogs():
            gen = torch.Generator().manual_seed(2026112216 + batch)
            sampled = sample_hard_graph(out, spec, generator=gen)
            motif_labels, valid_draws = _motif_labels(sampled, spec, failures={"smiles_roundtrip": 0})
            aro_hits, amide_hits = map(int, motif_labels.sum(0))
            needs = [max(0, math.ceil(floor * len(z)) - hits) for hits, floor in ((aro_hits,.08), (amide_hits,.12))]
            hard = out.harden(spec)
            start = time.perf_counter()
            sites = _pattern_sites([t.numpy() for t in (hard.atoms,hard.charges,hard.bonds)], spec, needs[1] > 0)
            selection_time = time.perf_counter() - start
        logits = out.symmetric_bond_logits.detach().requires_grad_(True)
        start = time.perf_counter()
        loss, stats = _pattern_objective(logits, sites, spec, *needs, .10)
        grad, = torch.autograd.grad(loss, logits)
        forward_back = time.perf_counter() - start
        p = logits.detach().softmax(-1)
        offsite = p[...,spec.bond_names.index("AROMATIC")][torch.as_tensor(sites[1])]
        blocked = p[...,spec.bond_names.index("DOUBLE")][torch.as_tensor(sites[2])]
        # Positive derivative means increasing that class everywhere increases the penalty.
        derivatives = dict(offsite_aro_global_bias=float(.5 * (offsite*(1-offsite)).sum()/len(z)),
                           blocked_double_global_bias=float(.10 * (blocked*(1-blocked)).sum()/len(z)))
        permitted = torch.as_tensor(sites[1] | sites[2]).clone()
        for rings in sites[3].values():
            for row, edges, _ in rings:
                for i, j in edges:
                    permitted[row,i,j] = True
        for site in sites[4] + sites[5]:
            for row, i, j in site:
                permitted[row,i,j] = True
        if not torch.isfinite(grad).all() or grad[~permitted].count_nonzero():
            raise AssertionError("Nonfinite gradients or a pair outside the known support contributed")
        events.append(dict(batch=batch, sampled_valid=valid_draws, sampled_aromatic=aro_hits, sampled_amide=amide_hits,
                           needs=needs, loss=float(loss.detach()), selection_seconds=selection_time,
                           cost_forward_backward_seconds=forward_back, **stats, **derivatives,
                           final_readout=directional_readout(grad, feat, sites, hard, spec)))
    handle.remove()
    report = dict(checkpoint_sha256=file_sha256(args.checkpoint), update=state["continuation"]["updates"],
                  draws=96, torch=torch.__version__, rdkit=rdBase.rdkitVersion, events=events)
    write_json(args.out / "summary.json", report)
    print(json.dumps(events))


if __name__ == "__main__":
    main()
