"""CPU aperture and last-layer transfer test using the same verified positions for CE and ranking."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.motif_contrast import _contrast_sites, _paired_loss
from latentmol.representation import DecoderOutput, GraphSpec


def quantile(values):
    return dict(zip(("p10", "median", "p90"), np.quantile(values, [.1, .5, .9]).tolist()))


def compare(out, features, hard, spec, name, edges, measures):
    r, i, j, n, a, b = torch.tensor(edges).T
    desired = spec.bond_names.index("AROMATIC" if name == "aromatic" else "DOUBLE")
    single = spec.bond_names.index("SINGLE")
    logits = out.symmetric_bond_logits
    pos, neg = logits[r, i, j], logits[n, a, b]
    gap = pos[:, desired] - pos[:, single]
    control_gap = neg[:, desired] - neg[:, single]
    measures["positive_gap"].extend(gap.detach().tolist())
    measures["negative_gap"].extend(control_gap.detach().tolist())
    measures["relative_gap"].extend((gap - control_gap).detach().tolist())
    mask = torch.zeros_like(hard.bonds, dtype=torch.bool)
    mask[r, i, j] = mask[r, j, i] = mask[n, a, b] = mask[n, b, a] = True
    observed = (hard.bonds == single).triu(1) & ~mask
    # Evaluate off-site transfer on comparable elements and on every other present
    # single bond. A sparse gradient at the logits does not imply zero transfer.
    elements = torch.tensor(spec.atomic_numbers)[hard.atoms]
    x, y = elements[:, :, None], elements[:, None, :]
    matched = ((x == 6) & ((y == 7) | (y == 6))) | ((x == 7) & (y == 6)) if name == "aromatic" else (
              ((x == 6) & (y == 8)) | ((x == 8) & (y == 6)))
    methods = {"absolute_ce": .5 * F.cross_entropy(pos[:, 1:], torch.full_like(r, desired - 1))}
    terms = {"aromatic": [], "amide": []}
    terms[name] = edges
    methods["ranking"], _ = _paired_loss(out, spec, terms, margin=1., absolute_fraction=0., amide_relative_weight=1.)
    derivatives = {}
    for method, cost in methods.items():
        g, = torch.autograd.grad(cost, out.bonds, retain_graph=True)
        derivatives[method] = g
        nonzero = g.abs().sum(-1) > 0
        if torch.any(nonzero & ~mask) or bool(g[..., 0].count_nonzero()):
            raise RuntimeError("Unexpected off-site or adjacency gradient")
        grad_w = g[nonzero].T @ features[nonzero]
        grad_b = g[nonzero].sum(0)
        measures[method + "_bias_max"].append(float(grad_b.abs().max()))
        norm = (grad_w.square().sum() + grad_b.square().sum()).sqrt()
        if not torch.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Missing or nonfinite directional head gradient")
        measures[method + "_raw_final_head_norm"].append(float(norm))
        # Unit descent step of final linear layer only (features frozen). This is the
        # exact directional derivative of the margin at every edge of this batch.
        change = -(features @ (grad_w[desired] - grad_w[single]) + grad_b[desired] - grad_b[single]) / norm
        change = (change + change.transpose(1, 2)) / 2
        for group, values in (("positive", change[r, i, j]), ("control", change[n, a, b]),
                              ("offsite", change[observed]), ("offsite_matched", change[observed & matched])):
            measures[method + "_delta_" + group].extend(values.tolist())
            measures[method + "_raw_delta_" + group].extend((values * norm).tolist())
    return derivatives


def shared_bias_probe(model, chunk, all_gradients, stats):
    """Include all three intermediate uses of the same head, holding the selected labels fixed."""
    bias = model.decoder.bond_head[-1].bias
    totals = {key: torch.zeros_like(bias) for key in all_gradients}
    for start in range(0, len(chunk), 4):
        window = slice(start, start + 4)
        active = [key for key, g in all_gradients.items() if g[window].count_nonzero()]
        if not active:
            continue
        live = model.decode(chunk[window]).bonds
        for k, key in enumerate(active):
            gradient, = torch.autograd.grad(live, bias, grad_outputs=all_gradients[key][window],
                                            retain_graph=k + 1 < len(active))
            totals[key] += gradient
    for (name, method), gradient in totals.items():
        stats[name][method + "_shared_bias_max"].append(float(gradient.abs().max()))
        wanted = model.spec.bond_names.index("AROMATIC" if name == "aromatic" else "DOUBLE")
        stats[name][method + "_shared_bias_wanted"].append(float(gradient[wanted]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026112215)
    args = parser.parse_args()
    if args.draws < 48 or args.draws % 48:
        parser.error("draws must be a positive multiple of 48")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    spec = GraphSpec()
    model, state = load_model(args.checkpoint, "cpu", spec)
    model.eval().requires_grad_(False)
    model.decoder.bond_head[-1].bias.requires_grad_(True)
    priors = torch.randn(args.draws, model.config.latent, generator=torch.Generator().manual_seed(args.seed))
    last_features = []
    def capture(_module, inputs):
        # Refinement rounds can also call this head. Keep the invocation producing
        # the final reported logits for each decoded microbatch.
        last_features[:] = [inputs[0].detach()]
    head = model.decoder.bond_head[-1]
    handle = head.register_forward_pre_hook(capture)
    totals = defaultdict(int)
    statistics = {name: defaultdict(list) for name in ("aromatic", "amide")}
    selection_seconds = 0.
    for chunk in priors.split(48):
        features = []
        pieces = []
        with torch.no_grad():
            for s in chunk.split(4):
                piece = model.decode(s)
                feature = last_features[0]
                torch.testing.assert_close(F.linear(feature, head.weight, head.bias), piece.bonds)
                features.append(feature)
                pieces.append(piece)
        out = DecoderOutput(*(torch.cat([getattr(g, k) for g in pieces]) for k in ("atoms", "charges", "bonds")))
        out.bonds.requires_grad_()
        feat = torch.cat(features)
        hard = out.harden(spec)
        start = time.perf_counter()
        paired, stats = _contrast_sites([t.numpy() for t in (hard.atoms, hard.charges, hard.bonds)], spec, True, True)
        selection_seconds += time.perf_counter() - start
        for k, v in stats.items():
            totals[k] += v
        all_gradients = {}
        for name, edges in paired.items():
            if edges:
                derivatives = compare(out, feat, hard, spec, name, edges, statistics[name])
                all_gradients.update({(name, method): g for method, g in derivatives.items()})
        shared_bias_probe(model, chunk, all_gradients, statistics)
    handle.remove()
    report = dict(checkpoint_sha256=file_sha256(args.checkpoint), update=state["continuation"]["updates"],
                  draws=args.draws, seed=args.seed, sites=dict(totals), seconds_selection=selection_seconds, probes={})
    for name, measurements in statistics.items():
        summary = {}
        for key, values in measurements.items():
            summary[key] = dict(count=len(values), mean=float(np.mean(values)), **quantile(values),
                                fraction_positive=float(np.mean(np.asarray(values) > 0)), max=float(np.max(values)))
        report["probes"][name] = summary
    write_json(args.out / "summary.json", report)
    print(json.dumps(dict(sites=report["sites"], cpu_seconds=selection_seconds)))


if __name__ == "__main__":
    main()
