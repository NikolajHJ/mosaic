"""Post-hoc whole-ring veto calibrated on unused real training rows. No VAE forwards."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from experiments.vae.training_data_eda.shared_ring_eval import _has_saturated_ring
from experiments.vae.training_data_eda.shared_ring_gate import SharedRingGate, find_ring_sites
from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import GraphSpec


@torch.inference_mode()
def score_real(gate, smiles, spec):
    scores = {True: [], False: []}
    with rdBase.BlockLogs():
        for i in range(0, len(smiles), 64):
            hard = encode(smiles[i:i+64], spec)
            sites = find_ring_sites(hard, spec)
            if sites:
                probs = gate(hard.to_relaxed(spec), sites).sigmoid().tolist()
                for site, prob in zip(sites, probs, strict=True):
                    scores[site.aromatic].append(prob)
    return {k: np.array(v, dtype=np.float64) for k, v in scores.items()}


def calibrate(negative, maximum):
    if not len(negative) or not np.isfinite(negative).all() or not 0 <= maximum < 1:
        raise ValueError("Need finite scores of eligible saturated calibration rings and a valid bound")
    # Strictly greater: a tie at the cutoff never exceeds the declared FP budget.
    allowed = int(np.floor(maximum * len(negative)))
    return float(np.sort(negative)[-1 - allowed])


def replay(path, threshold, calibration_smiles):
    counts = defaultdict(lambda: defaultdict(Counter))
    aromatic_ids = defaultdict(set)
    with gzip.open(path, "rt") as stream:
        lines = list(stream)
    for line in lines:
        row = json.loads(line)
        panel = row["panel"]
        if row.get("target") in calibration_smiles:
            raise ValueError("Calibration and evaluation molecule identities overlap")
        base = row.get("base", row)
        blanket = row.get("blanket_alternative", row)
        changed = bool(row["gate_changed"])
        saturated = bool(row["gate_saturated"])
        assert base["nodes"] == blanket["nodes"] and (not changed or saturated)
        approved = changed and row["gate_probability"] > threshold
        for policy, graph in [("base", base), ("blanket", blanket), ("original", row),
                              ("veto", row if approved else base)]:
            s = counts[panel][policy]
            s["attempts"] += 1
            s["eligible"] += row["gate_eligible"]
            s["saturated_sites"] += saturated
            s["converted"] += graph["smiles"] != base["smiles"]
            s["valid"] += graph["valid"]
            if graph["aromatic"] and graph["valid"]:
                s["valid_aromatic"] += 1
                aromatic_ids[panel, policy].add(graph["smiles"])
            if "target" in row:
                exact = bool(graph["valid"] and graph["smiles"] == row["target"])
                s["exact"] += exact
                if row["target_has_saturated_ring"]:
                    s["saturated_targets"] += 1
                    s["saturated_exact"] += exact
                    s["saturated_retained"] += bool(graph["valid"] and _has_saturated_ring(graph["smiles"]))
    result = {}
    for panel, policies in counts.items():
        result[panel] = {}
        for policy, count in policies.items():
            result[panel][policy] = dict(count, distinct_valid_aromatics=len(aromatic_ids[panel, policy]),
                                         aromatic_prevalence=count["valid_aromatic"] / count["attempts"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "train-panel", "training-metadata", "gate", "draws", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fit-ids", type=Path, help="JSON containing train_indices or a list of gate fit row IDs")
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026112587)
    parser.add_argument("--max-false-positive-rate", type=float, default=.10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.samples <= 0:
        raise ValueError("Need positive calibration budget")
    torch.set_num_threads(2)
    training = json.loads(args.training_metadata.read_bytes())
    train_hash = training.get("train_sha256", training.get("training_panel_sha256"))
    if (train_hash != file_sha256(args.train_panel)
            or training["checkpoint_sha256"] != file_sha256(args.checkpoint)):
        raise ValueError("Training data or checkpoint identity differs from fit metadata")
    spec = GraphSpec.from_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["graph_spec"])
    gate = SharedRingGate(spec)
    payload = torch.load(args.gate, map_location="cpu", weights_only=True)
    if "state" in payload:
        if (not args.fit_ids or payload["graph_spec"] != spec.to_dict()
                or payload["base_model_sha256"] != training["training"]["teacher_mask"]["model_sha256"]):
            raise ValueError("Matched gate requires its corresponding endpoint metadata and explicit fit row IDs")
        payload = payload["state"]
    gate.load_state_dict(payload)
    gate.eval()
    with gzip.open(args.train_panel, "rt") as f:
        pool = [r["smiles"] for r in csv.DictReader(f, delimiter="\t")]
    if args.fit_ids:
        selected = json.loads(args.fit_ids.read_bytes())
        fit_ids = selected["train_indices"] if isinstance(selected, dict) else selected
    else:
        fit_ids = np.random.default_rng(training["seed"]).choice(
            len(pool), size=training["training"]["rows"], replace=False)
    if len(set(fit_ids)) != len(fit_ids) or any(i < 0 or i >= len(pool) for i in fit_ids):
        raise ValueError("Invalid fit row IDs")
    fit_smiles = {pool[i] for i in fit_ids}
    available = [i for i, sm in enumerate(pool) if sm not in fit_smiles]
    indices = np.random.default_rng(args.seed).choice(available, size=args.samples, replace=False)
    examples = [pool[i] for i in indices]
    assert not set(examples).intersection(fit_smiles) and len(set(examples)) == len(examples)
    scores = score_real(gate, examples, spec)
    cutoff = calibrate(scores[False], args.max_false_positive_rate)
    above = {label: int(np.sum(value > cutoff)) for label, value in scores.items()}
    if above[False] > len(scores[False]) * args.max_false_positive_rate:
        raise RuntimeError("Veto violated the calibration budget")
    info = dict(calibration_rows=len(indices), fit_rows=len(fit_ids), cal_seed=args.seed,
                positive=len(scores[True]), negative=len(scores[False]), threshold=cutoff,
                eligible_rate=(len(scores[True])+len(scores[False])) / len(indices),
                allowed_false_positive_rate=args.max_false_positive_rate, false_positive=above[False],
                true_positive=above[True], false_positive_rate=above[False] / len(scores[False]),
                true_positive_rate=above[True] / len(scores[True]) if len(scores[True]) else None,
                checkpoint_sha256=training["checkpoint_sha256"], gate_sha256=file_sha256(args.gate),
                draws_sha256=file_sha256(args.draws), train_sha256=train_hash,
                comparison=replay(args.draws, cutoff, set(examples)),
                policy="Keep saved whole-ring decision only if its original score is strictly above cutoff.")
    write_json(args.output, info)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
