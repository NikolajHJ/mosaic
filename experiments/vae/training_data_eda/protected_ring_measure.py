"""Follow the same eligible rings and surrounding bonds through all pilot endpoints."""

import numpy as np
import torch
from rdkit import rdBase

from experiments.vae.training_data_eda.aromatic_encodings_probe import patterns, select_sites


@torch.inference_mode()
def build_ring_panel(model, *, seed, draws, batch_size):
    """Pick one verified ring per eligible starting graph on an independent fixed latent panel."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    codes = torch.randn(draws, model.config.latent, generator=torch.Generator().manual_seed(seed))
    groups = []
    try:
        with rdBase.BlockLogs():
            for block in codes.split(batch_size):
                output = model.decode(block.to(device))
                hard = output.harden(model.spec).to("cpu")
                rings, _, _ = select_sites(hard, model.spec)
                if not rings:
                    continue
                chosen = [ring[0][0] for ring in rings]
                entries = []
                for ring, idx in zip(rings, chosen, strict=True):
                    perimeter = [(i,j) for _,i,j in ring]
                    ring_keys = {tuple(sorted(edge)) for edge in perimeter}
                    external = [(i,j) for i,j in torch.triu(hard.bonds[idx], 1).nonzero().tolist()
                                if (i,j) not in ring_keys]
                    pairs = torch.tensor(perimeter + external, dtype=torch.long)
                    source = output.symmetric_bond_logits[idx, pairs[:, 0], pairs[:, 1]].float().cpu()
                    entries.append((pairs, source, hard.bonds[idx, pairs[:, 0], pairs[:, 1]]))
                groups.append((block[chosen], entries))
    finally:
        model.train(was_training)
    return dict(groups=groups, draws=draws, seed=seed)


@torch.inference_mode()
def measure_ring_panel(model, panel):
    """Use fixed original slots, recording soft drift and hard changes, including bond deletions."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    encodings = patterns(model.spec).to(device) - 1
    ring_logp, tv = [], []
    changed = protected = completed = 0
    try:
        for codes, entries in panel["groups"]:
            output = model.decode(codes.to(device))
            hard = output.harden(model.spec).to("cpu")
            for i, (pairs, source, labels) in enumerate(entries):
                ps = pairs.to(device)
                logits = output.symmetric_bond_logits[i, ps[:, 0], ps[:, 1]].float()
                ring = logits[:6, 1:].log_softmax(-1)
                assignments = ring[torch.arange(6, device=device)[None, :], encodings].sum(-1)
                ring_logp.append(torch.logsumexp(assignments, 0).item())
                current = hard.bonds[i, pairs[:, 0], pairs[:, 1]]
                completed += int(((encodings.cpu() + 1) == current[:6]).all(-1).any())
                if len(source) > 6:
                    diff = logits[6:].softmax(-1) - source[6:].to(device).softmax(-1)
                    tv.extend((0.5 * diff.abs().sum(-1)).tolist())
                    changed += (current[6:] != labels[6:]).sum().item()
                    protected += len(labels) - 6
    finally:
        model.train(was_training)
    return dict(seed=panel["seed"], draws=panel["draws"], sites=len(ring_logp),
                mean_ring_log_probability=float(np.mean(ring_logp)) if ring_logp else None,
                median_ring_log_probability=float(np.median(ring_logp)) if ring_logp else None,
                complete_raw_ring_assignments=completed, offsite_original_bonds=protected,
                changed_offsite_bond_labels=changed, mean_offsite_total_variation=float(np.mean(tv)) if tv else None)
