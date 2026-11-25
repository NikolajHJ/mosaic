"""Bounded frozen-checkpoint investigation of reconstruction versus prior support.

Uses an already downloaded, byte-verified cohort; never trains, repairs, accesses
HPC, or writes a checkpoint. Mixture density is a finite-reference diagnostic,
not an OOD certificate or a learned chemistry likelihood. Outputs retain every
attempt, including invalid graphs. Run with --help for the explicit budget.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.evaluation.evaluate import binomial_interval, load_model
from latentmol.representation import GraphSpec, HardGraphBatch
from latentmol.sampling import sample_hard_graph
from latentmol.train import state_digest


def cohort(inputs, name):
    arrays = []
    for filename in ("atom_types.npy", "charges.npy", "bond_types.npy"):
        record = inputs["cohorts"][name]["arrays"][filename]
        raw = base64.b64decode(record["base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Changed transferred cohort: " + filename)
        arrays.append(torch.from_numpy(np.frombuffer(raw, dtype=record["dtype"])
                                       .reshape(record["shape"]).copy()).long())
    return HardGraphBatch(*arrays)


def mixture_log_density(z, mu, logvar):
    """Exact log density of this declared equally weighted diagonal mixture."""
    z, mu, logvar = z.double(), mu.double(), logvar.double()
    precision = (-logvar).exp()
    quadratic = (z.square() @ precision.T - 2 * z @ (mu * precision).T
                 + (mu.square() * precision + logvar + math.log(2 * math.pi)).sum(-1))
    return torch.logsumexp(-.5 * quadratic, -1) - math.log(len(mu))


def auc(scores, valid):
    """Pairwise ranking AUC; ties count half, missing classes remain undefined."""
    yes, no = scores[valid], scores[~valid]
    if not len(yes) or not len(no):
        return None
    return float(((yes[:, None] > no).double() + .5 * (yes[:, None] == no)).mean())


def describe(values):
    if not len(values):
        return None
    return dict(mean=float(values.mean()),
                quantiles=dict(zip(("p10", "p50", "p90"),
                                   torch.quantile(values.double(), torch.tensor([.1, .5, .9], dtype=torch.double)).tolist())))


@torch.no_grad()
def run(args):
    started = time.monotonic()
    if min(args.prior_count, args.batch_size, args.threads, args.seconds) < 1:
        raise ValueError("Counts and runtime budget must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, payload = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False).eval()
    before = state_digest(model.state_dict())
    inputs = json.loads(args.inputs.read_text())
    if GraphSpec.from_dict(inputs["manifest"]["spec"]).fingerprint() != model.spec.fingerprint():
        raise ValueError("Transferred cohort and checkpoint schemas disagree")
    if set(inputs["cohorts"]["train"]["source_indices"]) & set(inputs["cohorts"]["val"]["source_indices"]):
        raise ValueError("Reference and held-out cohorts overlap")
    train, val = cohort(inputs, "train"), cohort(inputs, "val")
    reference_mu, reference_lv = [], []
    val_mu, val_lv = [], []
    for data, means, variances in ((train, reference_mu, reference_lv), (val, val_mu, val_lv)):
        for start in range(0, len(data.atoms), args.batch_size):
            if time.monotonic() - started > args.seconds:
                raise TimeoutError("Frozen evaluation exceeded its explicit budget")
            graph = HardGraphBatch(*(x[start:start + args.batch_size] for x in (data.atoms, data.charges, data.bonds)))
            mu, lv = model.encode(graph.to_relaxed(model.spec))
            if not torch.isfinite(mu).all() or not torch.isfinite(lv).all():
                raise ValueError("Nonfinite encoder output")
            means.append(mu); variances.append(lv)
    reference_mu, reference_lv = torch.cat(reference_mu), torch.cat(reference_lv)
    mu, lv = torch.cat(val_mu), torch.cat(val_lv)
    active = reference_mu.var(0, unbiased=False) > .01
    prior = torch.randn(args.prior_count, model.config.latent,
                        generator=torch.Generator().manual_seed(args.seed))
    noise = torch.randn(mu.shape, generator=torch.Generator().manual_seed(args.seed + 1))
    direction = noise / noise.square().mean(-1, keepdim=True).sqrt()
    groups = {
        "heldout_mu": mu,
        "heldout_local_005": mu + .05 * (.5 * lv).exp() * direction,
        "heldout_posterior_sample": mu + (.5 * lv).exp() * noise,
        "prior_075": .75 * prior,
        "prior_100": prior,
        "prior_125": 1.25 * prior,
    }
    summaries = {}
    with (args.output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
        for name, codes in groups.items():
            if time.monotonic() - started > args.seconds:
                raise TimeoutError("Frozen evaluation exceeded its explicit budget")
            q = mixture_log_density(codes, reference_mu, reference_lv)
            p = -.5 * (codes.double().square() + math.log(2 * math.pi)).sum(-1)
            qa = mixture_log_density(codes[:, active], reference_mu[:, active], reference_lv[:, active])
            pa = -.5 * (codes[:, active].double().square() + math.log(2 * math.pi)).sum(-1)
            rows = []
            sample_rng = torch.Generator().manual_seed(args.seed + 2)
            for start in range(0, len(codes), args.batch_size):
                if time.monotonic() - started > args.seconds:
                    raise TimeoutError("Frozen evaluation exceeded its explicit budget")
                decoded = model.decode(codes[start:start + args.batch_size])
                if not all(torch.isfinite(t).all() for t in (decoded.atoms, decoded.charges, decoded.bonds)):
                    raise ValueError("Nonfinite decoder output")
                hard = decoded.harden(model.spec)
                logp = decoded.atoms.log_softmax(-1)
                occupied = hard.atoms != 0
                entropy = (-(logp.exp() * logp).sum(-1) * occupied).sum(-1) / occupied.sum(-1).clamp_min(1)
                draws = [sample_hard_graph(decoded, model.spec, generator=sample_rng)
                         for _ in range(4)] if name == "prior_100" else []
                for offset in range(len(hard.atoms)):
                    i = start + offset
                    labels = [x[offset] for x in (hard.atoms, hard.charges, hard.bonds)]
                    c = classify_repair_graph(*labels, model.spec)
                    row = dict(group=name, index=i, valid=c["raw_chemical_valid"] is True,
                        strict_valid=c["strict_teacher_valid"] is True, reason=c["reason"],
                        failure_kind=c["failure_kind"], smiles=c["smiles"],
                        radius=float(codes[i].norm()), log_q_minus_log_p=float(q[i] - p[i]),
                        active_log_q_minus_log_p=float(qa[i] - pa[i]),
                        active_atom_entropy=float(entropy[offset]),
                        graph={key: value.tolist() for key, value in zip(("atoms", "charges", "bonds"), labels)})
                    if name == "heldout_mu":
                        row["exact_labels"] = all(torch.equal(x, getattr(val, k)[i])
                            for x, k in zip(labels, ("atoms", "charges", "bonds")))
                    if draws:
                        row["valid_categorical_draws_of_4"] = sum(
                            classify_repair_graph(d.atoms[offset], d.charges[offset], d.bonds[offset], model.spec)
                            ["raw_chemical_valid"] is True for d in draws)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
            valid = torch.tensor([r["valid"] for r in rows])
            summary = dict(attempts=len(rows), valid=int(valid.sum()),
                validity=float(valid.double().mean()), validity_wilson95=binomial_interval(int(valid.sum()), len(rows)),
                strict_valid=sum(r["strict_valid"] for r in rows),
                unique_valid=len({r["smiles"] for r in rows if r["valid"]}),
                failures=dict(Counter(r["reason"] for r in rows if not r["valid"])))
            for field in ("radius", "log_q_minus_log_p", "active_log_q_minus_log_p", "active_atom_entropy"):
                values = torch.tensor([r[field] for r in rows], dtype=torch.double)
                summary[field] = dict(valid=describe(values[valid]), invalid=describe(values[~valid]),
                    auc_higher_predicts_valid=auc(values, valid))
            if name == "heldout_mu":
                summary["exact_labels"] = sum(r["exact_labels"] for r in rows)
            if name == "prior_100":
                invalid_rows = [r for r in rows if not r["valid"]]
                summary["invalid_argmax_rescued_by_4_draws"] = sum(r["valid_categorical_draws_of_4"] > 0 for r in invalid_rows)
                summary["categorical_valid_of_all_draws"] = sum(r["valid_categorical_draws_of_4"] for r in rows)
                order = torch.argsort(q - p).tolist()
                summary["density_ratio_quartiles_low_to_high"] = [
                    dict(attempts=len(indices), valid=sum(rows[i]["valid"] for i in indices))
                    for indices in (order[j * len(order)//4:(j+1)*len(order)//4] for j in range(4))]
            summaries[name] = summary
            print(json.dumps(dict(group=name, attempts=len(rows), valid=summary["valid"],
                                  seconds=time.monotonic() - started)), flush=True)
    if state_digest(model.state_dict()) != before:
        raise RuntimeError("Frozen model changed during evaluation")
    result = dict(epoch=payload["epoch"], checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(), seed=args.seed,
        reference_molecules=len(reference_mu), heldout_molecules=len(mu), active_dimensions=int(active.sum()),
        config=vars(args) | {"checkpoint": str(args.checkpoint), "inputs": str(args.inputs), "output": str(args.output)},
        frozen_model_unchanged=True, seconds=time.monotonic() - started, groups=summaries,
        limitations=["One checkpoint; bounded train reference and held-out cohort, not a full-test estimate",
            "Finite-reference aggregate posterior density is a diagnostic proxy, not proof of OOD",
            "Scaled-prior panels are paired interventions, not standard Gaussian draws",
            "Four categorical draws share a latent; neither independent molecules nor a free repair budget",
            "No training, repair, checkpoint edits or changes to live HPC jobs"])
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=904017)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=600)
    run(parser.parse_args())
