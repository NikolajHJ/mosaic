"""Replay saved VAE categorical draws to measure how batch size and motif floors affect score gradients."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.guidance.functional_groups import _batch_shortfall
from latentmol.representation import DecoderOutput, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob, sample_hard_graph


MOTIFS = ("aromatic", "amide")


def saved_graphs(rows, spec, device):
    atoms = np.zeros((len(rows), spec.max_nodes), dtype=np.int64)
    charges = np.full_like(atoms, spec.formal_charges.index(0))
    bonds = np.zeros((*atoms.shape, spec.max_nodes), dtype=np.int64)
    for k, row in enumerate(rows):
        for i, a, c in row["nodes"]:
            atoms[k, i], charges[k, i] = a, c
        for i, j, b in row["edges"]:
            bonds[k, i, j] = bonds[k, j, i] = b
    return HardGraphBatch(*(torch.from_numpy(x).to(device) for x in (atoms, charges, bonds)))


def score_directions(decoded, graph, spec):
    """Per-row derivative of log q(graph) along shared head-category bias directions."""
    tensors = (decoded.atoms, decoded.charges, decoded.bonds)
    biases = [t.new_zeros((len(t), t.shape[-1]), requires_grad=True) for t in tensors]
    shifted = DecoderOutput(*(t + b.reshape((len(t),) + (1,) * (t.ndim-2) + (-1,))
                              for t, b in zip(tensors, biases, strict=True)))
    grad = torch.autograd.grad(hard_graph_log_prob(shifted, graph, spec).sum(), biases)
    return torch.cat(grad, dim=-1).detach().double().cpu().numpy()


def describe(gradients):
    avg = gradients.mean(axis=0)
    spread = np.sqrt(gradients.var(axis=0, ddof=1).sum())
    return dict(batches=len(gradients), mean_vector=avg.tolist(), mean_vector_norm=float(np.linalg.norm(avg)),
                rms_deviation_around_mean=float(spread),
                exactly_zero_batches=int(np.sum(np.all(gradients == 0, axis=1))))


def compare(labels, scores, floors):
    report = {}
    for batch in (48, 240):
        by_motif = {}
        for i, name in enumerate(MOTIFS):
            local_floors = [0., 0.]
            local_floors[i] = floors[i]
            gradients = []
            for start in range(0, len(labels), batch):
                weights, _ = _batch_shortfall(labels[start:start + batch], tuple(local_floors))
                gradients.append(weights @ scores[start:start + batch])
            g = np.stack(gradients)
            by_motif[name] = describe(g)
            if batch == 48:
                by_motif[name]["average_of_five_separate_48_draw_gradients"] = describe(
                    g.reshape(-1, 5, g.shape[-1]).mean(axis=1))
        report[str(batch)] = by_motif
    return report


def label_panel(folder, budgets):
    """Describe exact batching effects on all saved chemical labels without a model forward."""
    with gzip.open(folder / "control-categorical.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    used = len(rows) - len(rows) % 240
    labels = np.array([[r[k] for k in MOTIFS] for r in rows[:used]], dtype=bool)
    out = {"draws":used, "ignored_tail":len(rows)-used, "budgets":{}}
    for recipe, floors in budgets:
        out["budgets"][recipe] = {}
        for batch in (48, 240):
            grouped = labels.reshape(-1,batch,2)
            measurements = {}
            for i, name in enumerate(MOTIFS):
                counts = grouped[...,i].sum(axis=1)
                local = [0.,0.]
                local[i] = floors[i]
                coeffs = [_batch_shortfall(block,tuple(local))[0] for block in grouped]
                measurements[name] = dict(total=int(counts.sum()), batches=len(counts),
                    zero_hit_batches=int((counts == 0).sum()), frequency_sd_pp=float(100*np.std(counts/batch,ddof=1)),
                    mean_coefficient_l1=float(np.abs(coeffs).sum(axis=1).mean()))
            out["budgets"][recipe][str(batch)] = measurements
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint", "panel", "training-frequencies", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--draws", type=int, help="Use this prefix of the original 8,192-draw panel.")
    args = parser.parse_args()
    panel = json.loads((args.panel / "summary.json").read_bytes())
    if file_sha256(args.endpoint) != panel["endpoint_files"]["control"]:
        raise ValueError("Wrong frozen endpoint for the categorical panel")
    with args.training_frequencies.open(encoding="utf-8", newline="") as f:
        freq = {r["motif"]: int(r["positive"]) / int(r["attempted"]) for r in csv.DictReader(f)
                if r["population"] == "training"}
    args.output.mkdir(parents=True, exist_ok=False)
    budgets = (("previous", (.08,.12)), ("dataset", (freq["aromatic_ring"],freq["amide"])))
    (args.output / "label-panel.json").write_text(json.dumps(label_panel(args.panel, budgets),indent=2)+"\n",
                                                 encoding="utf-8")
    model, _ = load_model(args.endpoint, args.device)
    model.eval().requires_grad_(False)
    count = panel["raw_attempts_per_mode_and_arm"]
    used = count - count % 240 if args.draws is None else args.draws
    if used < 480 or used > count or used % panel["batch"] or used % 240:
        raise ValueError("Panel requires at least two complete 240-draw pools")
    torch.set_num_threads(4)
    z = torch.randn(count, model.config.latent, generator=torch.Generator().manual_seed(panel["latent_seed"]))
    generator = (torch.Generator(device="cuda").manual_seed(panel["categorical_seed"])
                 if args.device == "cuda" else None)
    scores, labels = [], []
    argmax_mismatches = 0
    t = time.perf_counter()
    with (gzip.open(args.panel / "control-categorical.jsonl.gz", "rt", encoding="utf-8") as f,
          gzip.open(args.panel / "control-argmax.jsonl.gz", "rt", encoding="utf-8") as af):
        for first in range(0, used, panel["batch"]):
            rows = [json.loads(next(f)) for _ in range(panel["batch"])]
            if [r["draw"] for r in rows] != list(range(first, first + len(rows))):
                raise ValueError("Unexpected stored draw ordering")
            with torch.no_grad():
                decoded = model.decode(z[first:first + len(rows)].to(args.device))
                saved = saved_graphs(rows, model.spec, args.device)
                if generator is not None:
                    sampled = sample_hard_graph(decoded, model.spec, generator=generator)
                    if any(not torch.equal(getattr(sampled, k), getattr(saved, k)) for k in ("atoms", "charges", "bonds")):
                        raise ValueError(f"Frozen replay differs at draw block {first}")
                stored_argmax = saved_graphs([json.loads(next(af)) for _ in rows], model.spec, args.device)
                replay = decoded.harden(model.spec)
                same = torch.ones(len(rows),device=args.device,dtype=torch.bool)
                for key in ("atoms","charges","bonds"):
                    same &= (getattr(stored_argmax,key)==getattr(replay,key)).reshape(len(rows),-1).all(-1)
                argmax_mismatches += int((~same).sum())
            scores.append(score_directions(decoded, saved, model.spec))
            labels.extend((r["aromatic"], r["amide"]) for r in rows)
            if (first + len(rows)) % 480 == 0:
                print(first + len(rows), "/", used, round(time.perf_counter()-t,1), flush=True)
    scores, labels = np.concatenate(scores), np.asarray(labels, dtype=bool)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite graph likelihood derivatives")
    seconds = time.perf_counter() - t
    report = dict(draws=used, ignored_tail=count-used, positive=labels.sum(axis=0).tolist(),
                  parameter_order=[list(model.spec.atomic_numbers), list(model.spec.formal_charges),
                                   list(model.spec.bond_names)],
                  replay_seconds=seconds, device=torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                  torch_version=torch.__version__, reference_argmax_mismatches=argmax_mismatches,
                  endpoint_sha256=panel["endpoint_files"]["control"], latent_seed=panel["latent_seed"],
                  categorical_seed=panel["categorical_seed"], budgets={})
    for name, floors in budgets:
        report["budgets"][name] = {"floors":floors, "comparisons":compare(labels, scores, floors)}
    np.savez_compressed(args.output / "scores.npz", labels=labels, scores=scores)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for label, item in report["budgets"].items():
        for batch, motifs in item["comparisons"].items():
            print(label,batch,[(k,v["exactly_zero_batches"],v["batches"],v["mean_vector_norm"],
                               v["rms_deviation_around_mean"]) for k,v in motifs.items()])


if __name__ == "__main__":
    main()
