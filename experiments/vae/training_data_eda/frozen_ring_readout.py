"""Fit the shared ring decision on real structures, then evaluate a frozen VAE on fresh codes."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from rdkit import rdBase
import torch

from experiments.vae.training_data_eda.shared_ring_eval import compare_evaluations, evaluate_endpoint
from experiments.vae.training_data_eda.shared_ring_gate import SharedRingGate, find_ring_sites, gate_training_loss
from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json


@torch.no_grad()
def masked_ring_classification(gate, validation, spec):
    """Can the head tell held-out aromatic and saturated sites apart with the perimeter hidden?"""
    positives, negatives = [], []
    for hard, _ in validation:
        sites = find_ring_sites(hard, spec)
        if sites:
            scores = gate(hard.to_relaxed(spec), sites).sigmoid().tolist()
            for site, score in zip(sites, scores, strict=True):
                (positives if site.aromatic else negatives).append(score)
    pos, neg = np.asarray(positives), np.asarray(negatives)
    auc = None
    if len(pos) and len(neg):
        gap = pos[:, None] - neg[None, :]
        auc = float((np.count_nonzero(gap > 0) + 0.5 * np.count_nonzero(gap == 0)) / gap.size)
    return dict(aromatic_sites=len(pos), saturated_sites=len(neg), auc=auc,
                classified_aromatic_true=int((pos >= .5).sum()), classified_aromatic_false=int((neg >= .5).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "train-panel", "heldout-panel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026112532)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--draws", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    model, _ = load_model(args.checkpoint, "cpu")
    model.eval().requires_grad_(False)
    with gzip.open(args.train_panel, "rt") as stream:
        pool = [r["smiles"] for r in csv.DictReader(stream, delimiter="\t")]
    held = [r[2] for r in json.loads(args.heldout_panel.read_bytes())["rows"]]
    if set(held).intersection(pool):
        raise ValueError("Training and validation inputs overlap")
    rows = np.random.default_rng(args.seed).choice(len(pool), size=64 * args.updates, replace=False)
    args.output.mkdir(parents=True, exist_ok=False)
    gate = SharedRingGate(model.spec)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.001, weight_decay=.001)
    totals = dict(rows=0, sites=0, positive=0, negative=0)
    with rdBase.BlockLogs():
        for ids in rows.reshape(-1, 64):
            optimizer.zero_grad(set_to_none=True)
            loss, counts = gate_training_loss(gate, encode([pool[i] for i in ids], model.spec), model.spec)
            for key in totals:
                totals[key] += counts[key]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            if counts["sites"]:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
    torch.save(gate.state_dict(), args.output / "gate.pt")
    validation = [(encode(held[i:i + 16], model.spec), held[i:i + 16]) for i in range(0, len(held), 16)]
    summary = dict(training=totals, seed=args.seed, checkpoint_sha256=file_sha256(args.checkpoint),
                   train_sha256=file_sha256(args.train_panel), heldout_sha256=file_sha256(args.heldout_panel),
                   updates=args.updates, draws=args.draws)
    summary["heldout_masked_ring_classification"] = masked_ring_classification(gate, validation, model.spec)
    write_json(args.output / "training.json", summary)
    for label, head in (("base", None), ("gate", gate)):
        summary[label] = evaluate_endpoint(model, head, validation, args.output / (label + ".jsonl.gz"),
                                           draws=args.draws, seed=args.seed + 1, batch_size=16)
    summary["paired"] = compare_evaluations(args.output / "base.jsonl.gz", args.output / "gate.jsonl.gz")
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
