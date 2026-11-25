"""Train bond logits on verified amide near misses and matched, incompatible C–O bonds.

Run as a module for a short CPU comparison. It only updates the existing shared
final bond head on private copies of a checkpoint. Hard generation is unchanged.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from rdkit import rdBase
import torch
from torch.nn import functional as F

from experiments.vae.training_data_eda.teacher_motif_training_pilot import hard_records
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.motif_contrast import _contrast_sites, _negatives
from latentmol.sampling import sample_hard_graph
from latentmol.train import state_digest


def amide_sites(decoded, spec):
    """Detached positive/negative choices; no sampled graph enters the gradient path."""
    hard = decoded.harden(spec).to("cpu")
    arrays = tuple(x.numpy() for x in (hard.atoms, hard.charges, hard.bonds))
    paired, stats = _contrast_sites(arrays, spec, False, True)
    blocked = []
    with rdBase.BlockLogs():
        for row, (atoms, charges, bonds) in enumerate(zip(*arrays, strict=True)):
            blocked.extend(
                (row, i, j) for _, _, i, j in _negatives(atoms, charges, bonds, spec, False, True)
            )
    return paired["amide"], blocked, stats


def _gap(logits, positions, spec):
    row, i, j = torch.as_tensor(positions, device=logits.device, dtype=torch.long).reshape(-1, 3).T
    x = logits[row, i, j]
    return x[:, spec.bond_names.index("DOUBLE")] - x[:, spec.bond_names.index("SINGLE")]


def protected_amide_loss(decoded, anchor, spec, *, sites=None, protected=True):
    """Encourage an amide-positive C–O to gain DOUBLE-vs-SINGLE odds over an incompatible C–O.

    Signed changes are measured from the frozen source on the SAME latent. The
    source already separates many matched sites but gives too little absolute
    DOUBLE probability even at positives. A relative improvement margin removes
    that pre-existing offset; an absolute CE on positives breaks ties. Protect
    every blocked C–O in the batch against upward log-odds drift, allowing its
    odds to decrease. Bond presence receives no direct logit derivative.
    """
    if decoded.bonds.shape != anchor.bonds.shape:
        raise ValueError("Student and anchor bond logits differ in shape")
    if sites is None:
        with torch.no_grad():
            sites = amide_sites(decoded, spec)
    paired, blocked, stats = sites
    logits = decoded.symmetric_bond_logits
    frozen = anchor.symmetric_bond_logits.detach()
    zero = logits.reshape(-1)[0] * 0
    ce = contrast = guard = zero
    if paired:
        pos = [x[:3] for x in paired]
        neg = [x[3:] for x in paired]
        r, i, j = torch.as_tensor(pos, device=logits.device).T
        ce = F.cross_entropy(logits[r, i, j, 1:], torch.full_like(r, spec.bond_names.index("DOUBLE") - 1))
        relative_gain = (
            _gap(logits, pos, spec)
            - _gap(frozen, pos, spec)
            - _gap(logits, neg, spec)
            + _gap(frozen, neg, spec)
        )
        contrast = F.softplus(1.0 - relative_gain).mean()
    if blocked:
        difference = _gap(logits, blocked, spec) - _gap(frozen, blocked, spec)
        guard = difference.clamp_min(0).square().mean()
    loss = 0.25 * ce + contrast + guard if protected else ce
    if not loss.detach().isfinite():
        raise FloatingPointError("Nonfinite amide objective")
    return loss, dict(
        stats,
        positive_ce=float(ce.detach()),
        paired_gain_loss=float(contrast.detach()),
        blocked_upward_drift=float(guard.detach()),
        blocked_edges=len(blocked),
    )


def _summary(values):
    if not values:
        return dict(n=0)
    arr = np.asarray(values)
    return dict(
        n=len(arr),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        fraction_increasing=float(np.mean(arr > 1e-8)),
    )


def evaluate(model, blocks, output, seed):
    """Fixed, fresh Gaussian draws, original argmax and categorical samplers."""
    odds = {"positive": [], "paired_negative": [], "all_blocked": []}
    probabilities = {k: [] for k in odds}
    raised_hard = {k: 0 for k in odds}
    records = {"argmax": [], "categorical": []}
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad(), rdBase.BlockLogs(), gzip.open(output, "xt", encoding="utf8") as f:
        for z, original, (paired, blocked, _) in blocks:
            out = model.decode(z)
            logits = out.symmetric_bond_logits
            frozen = original.symmetric_bond_logits
            for name, positions in (
                ("positive", [p[:3] for p in paired]),
                ("paired_negative", [p[3:] for p in paired]),
                ("all_blocked", blocked),
            ):
                if not positions:
                    continue
                r, i, j = torch.as_tensor(positions).T
                odds[name].extend(
                    (_gap(logits, positions, model.spec) - _gap(frozen, positions, model.spec)).tolist()
                )
                idx = model.spec.bond_names.index("DOUBLE")
                probabilities[name].extend(
                    (
                        logits[r, i, j, 1:].softmax(-1)[:, idx - 1]
                        - frozen[r, i, j, 1:].softmax(-1)[:, idx - 1]
                    ).tolist()
                )
                raised_hard[name] += int(logits[r, i, j].argmax(-1).eq(idx).sum())
            for mode, hard in (
                ("argmax", out.harden(model.spec)),
                ("categorical", sample_hard_graph(out, model.spec, generator=g)),
            ):
                for rec in hard_records(hard, model.spec):
                    row = {
                        k: rec[k] for k in ("valid", "amide", "aromatic", "smiles", "scaffold", "failure")
                    }
                    row.update(mode=mode, draw=len(records[mode]))
                    records[mode].append(row)
                    f.write(json.dumps(row) + "\n")
    panels = {
        mode: dict(
            attempts=len(rows),
            valid=sum(r["valid"] for r in rows),
            amide=sum(r["amide"] for r in rows),
            aromatic=sum(r["aromatic"] for r in rows),
            distinct_valid_amides=len({r["smiles"] for r in rows if r["amide"]}),
        )
        for mode, rows in records.items()
    }
    return records, dict(
        panels=panels,
        delta_gap={k: _summary(v) for k, v in odds.items()},
        delta_conditional_double_probability={k: _summary(v) for k, v in probabilities.items()},
        argmax_double_at_original_sites=raised_hard,
    )


def run(args):
    if min(args.updates, args.batch_size, args.draws, args.threads) < 1 or args.step <= 0:
        raise ValueError("Require positive budgets")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    model, state = load_model(args.checkpoint, "cpu")
    model.eval().requires_grad_(False)
    checksum = state_digest(model.state_dict())
    head = model.decoder.bond_head[-1].requires_grad_(True)
    params = tuple(head.parameters())
    initial = [p.detach().clone() for p in params]

    def prepare(count, seed):
        z = torch.randn(count, model.config.latent, generator=torch.Generator().manual_seed(seed))
        blocks = []
        with torch.no_grad(), rdBase.BlockLogs():
            for block in z.split(args.batch_size):
                original = model.decode(block)
                blocks.append((block, original, amide_sites(original, model.spec)))
        return blocks

    fit = prepare(args.updates * args.batch_size, args.seed)
    fresh = prepare(args.draws, args.seed + 1)
    record = {}
    summary = dict(
        checkpoint_sha256=file_sha256(args.checkpoint),
        checkpoint_update=state["continuation"]["updates"],
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        head_step_l2=args.step,
        draws=args.draws,
        fit_pairs=sum(len(s[0]) for _, _, s in fit),
        rdkit=rdBase.rdkitVersion,
        torch=torch.__version__,
        arms={},
    )
    try:
        for name in ("start", "positive_ce", "protected"):
            with torch.no_grad():
                for par, base in zip(params, initial, strict=True):
                    par.copy_(base)
            stepped = 0
            if name != "start":
                with rdBase.BlockLogs(), (args.output / f"{name}-updates.jsonl").open("x") as log:
                    for z, anchor, selected in fit:
                        live = model.decode(z)
                        value, stats = protected_amide_loss(
                            live, anchor, model.spec, sites=selected, protected=name == "protected"
                        )
                        grad = torch.autograd.grad(value, params)
                        norm = torch.sqrt(sum(g.square().sum() for g in grad))
                        if not norm.isfinite():
                            raise FloatingPointError("Nonfinite head gradient")
                        # Match the effective per-step displacement. Skip empty batches
                        # in both arms, since a zero positive budget isn't a comparison.
                        if selected[0] and norm > 0:
                            with torch.no_grad():
                                for par, g in zip(params, grad, strict=True):
                                    par.add_(g, alpha=-args.step / float(norm))
                            stepped += 1
                        log.write(
                            json.dumps(dict(stats, value=float(value.detach()), norm=float(norm))) + "\n"
                        )
            displacement = float(
                torch.sqrt(
                    sum((p.detach() - b).square().sum() for p, b in zip(params, initial, strict=True))
                )
            )
            rows, report = evaluate(model, fresh, args.output / f"{name}.jsonl.gz", args.seed + 2)
            record[name] = rows
            summary["arms"][name] = dict(report, steps=stepped, weight_l2_change=displacement)
    finally:
        with torch.no_grad():
            for par, base in zip(params, initial, strict=True):
                par.copy_(base)
    if state_digest(model.state_dict()) != checksum:
        raise RuntimeError("Original head was not restored")
    summary["paired"] = {}
    for a, b in (("start", "positive_ce"), ("start", "protected"), ("positive_ce", "protected")):
        changes = {}
        for mode in ("argmax", "categorical"):
            changes[mode] = {}
            for key in ("valid", "amide", "aromatic"):
                pairs = [(l[key], r[key]) for l, r in zip(record[a][mode], record[b][mode], strict=True)]
                changes[mode][key] = dict(
                    gained=sum(bool(r and not l) for l, r in pairs),
                    lost=sum(bool(l and not r) for l, r in pairs),
                )
        summary["paired"][f"{a}_to_{b}"] = changes
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: v["panels"] for k, v in summary["arms"].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--draws", type=int, default=512)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--step", type=float, default=0.0125)
    parser.add_argument("--seed", type=int, default=2026112563)
    run(parser.parse_args())
