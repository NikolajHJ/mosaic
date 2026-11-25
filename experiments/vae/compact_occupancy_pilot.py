"""Paired CPU ablation of prior atom-count guidance on saved, disjoint PubChem panels.

The same compact model learns from real rows before branching. One pair retains
the warmed weights. A second pair starts with a trainable NO_ATOM bias shift as
a controlled challenge. No inference-time bias or atom-count cap is applied.

Reproduce from the checkout root in PowerShell with the project Python and
PYTHONPATH=src. Use two new output directories. Both commands use the same
seed and the full train-count histogram from the released 60M view. Each
optimizer update uses 24 real molecules and six fresh prior latents for this
small CPU experiment. The production timing benchmark has a 15:1 ratio.

    $script = 'experiments/vae/compact_occupancy_pilot.py'
    $inputs = @('--train-panel', 'outputs/runs/vae-occupancy-recovery-20261119a/train-panel.json',
                '--val-panel', 'outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json',
                '--cache-receipt', 'outputs/hpc-results/pubchem-large-corpus-20261105a/cache-and-view-20261106T094305211844Z.json')
    python $script @inputs --starts both --updates 16 --output outputs/runs/compact-count-paired-16
    python $script @inputs --starts stress --updates 64 --stress-radius-multiplier 1.5 --output outputs/runs/compact-count-stress-64
    python $script @inputs --arm kl-and-argmax --updates 64 --undisturbed-updates 16 --stress-radius-multiplier 1.5 --reference outputs/runs/compact-count-paired-16/comparison.json --reference outputs/runs/compact-count-stress-64/comparison.json --output outputs/runs/compact-count-argmax
"""

import argparse
import copy
import json
import random
from pathlib import Path
from time import perf_counter

import torch
from rdkit import rdBase

from latentmol.chemistry import encode_canonical_smiles
from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.compact_prior_panel import CompactPriorPanel
from latentmol.experiments import write_json
from latentmol.models import GraphAutoencoder
from latentmol.representation import GraphSpec
from latentmol.train import kl_weight, seed_everything, state_digest
from latentmol.training.compact_benchmark import _optimizer, _update, read_settings
from latentmol.training.compact_prior import CompactPriorGuidance
from latentmol.training.occupancy_monitor import _inspect


def read_json(path):
    return json.loads(path.read_bytes())


def tensor_row(smiles, spec):
    g = encode_canonical_smiles(smiles, spec)
    if g["smiles"] != smiles:
        raise ValueError("Noncanonical SMILES in a cached panel")
    return {name: torch.as_tensor(g[field], dtype=torch.long) for name, field in
            (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))}


@torch.inference_mode()
def evaluate(model, panel, validation):
    prior = panel.observe(model)
    hist = {int(k): v for k, v in prior["prior_size_histogram"].items()}
    capacity = model.spec.max_nodes
    prior["prior_within_one_of_capacity"] = hist.get(capacity - 1, 0) + hist.get(capacity, 0)
    prior["prior_empty"] = hist.get(0, 0)
    results = []
    targets = []
    count_errors = []
    was_training = model.training
    model.eval()
    try:
        for hard, smiles in validation:
            mu, _ = model.encoder(hard, from_canonical_cache=True)
            output = model.decode(mu).harden(model.spec)
            count_errors.extend(((output.atoms != 0).sum(-1) - (hard.atoms != 0).sum(-1)).abs().tolist())
            results.extend(x.smiles for x in _inspect(output, model.spec))
            targets.extend(smiles)
    finally:
        model.train(was_training)
    return {**prior, "heldout_count_exact": count_errors.count(0),
            "heldout_count_mae": sum(count_errors) / len(count_errors),
            "heldout_n": len(results), "heldout_valid": sum(s is not None for s in results),
            "heldout_exact": sum(a == b for a, b in zip(results, targets, strict=True))}


def draw_batch(table, cfg, args, index):
    rng = random.Random(args.seed + 2 * index)
    noise = torch.Generator().manual_seed(args.seed + 2 * index + 1)
    hard = collate_graphs(rng.sample(table, args.real_batch))
    eps = torch.randn(args.real_batch, cfg.model.latent, generator=noise)
    prior = torch.randn(args.prior_batch, cfg.model.latent, generator=noise)
    return hard, eps, prior


def step(model, opt, cfg, table, args, index, guide=None):
    hard, epsilon, prior = draw_batch(table, cfg, args, index)
    return _update(model, opt, cfg, hard, epsilon=epsilon,
                   beta=kl_weight(cfg, 1, index * args.real_batch), microbatch=args.microbatch,
                   guidance=guide, prior=prior)


def verify_replay(args, references, shift, budget, warm_sha, start_sha):
    """Reuse finalized earlier controls only when the starting weights and draws match."""
    for reference in references:
        if not reference.get("complete") or reference["warm_model_sha256"] != warm_sha:
            continue
        keys = ("seed", "warmup", "real_batch", "prior_batch", "microbatch", "eval_prior", "eval_real",
                "config", "train_panel", "val_panel", "cache_receipt")
        if any(str(reference["arguments"][k]) != str(getattr(args, k)) for k in keys):
            continue
        for arm in reference["arms"]:
            if (arm["no_atom_bias_subtraction"] == shift and arm["count_kl_weight"] == 0
                    and arm["starting_sha256"] == start_sha and str(budget) in arm["observations"]):
                return
    raise ValueError("A completed, identically seeded control at the same starting weights and budget is required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/vae/compact435k-v100.json"))
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--val-panel", type=Path, required=True)
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--real-batch", type=int, default=24)
    parser.add_argument("--prior-batch", type=int, default=6)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--eval-prior", type=int, default=64)
    parser.add_argument("--eval-real", type=int, default=64)
    parser.add_argument("--capacity-shift", type=float, default=3.0)
    parser.add_argument("--starts", choices=("both", "stress", "undisturbed"), default="both")
    parser.add_argument("--stress-radius-multiplier", type=float)
    parser.add_argument("--undisturbed-updates", type=int)
    parser.add_argument("--arm", choices=("cdf-and-kl", "kl-and-argmax"), default="cdf-and-kl")
    parser.add_argument("--reference", action="append", type=Path, default=[])
    parser.add_argument("--seed", type=int, default=192711)
    args = parser.parse_args()
    if not min(args.updates, args.warmup, args.real_batch, args.microbatch, args.eval_prior, args.eval_real) > 0:
        parser.error("All update and panel budgets must be positive")
    if args.prior_batch < 2 or args.capacity_shift <= 0:
        parser.error("Use at least two fresh priors per update and a positive stress shift")
    if args.undisturbed_updates is not None and args.undisturbed_updates < 1:
        parser.error("Undisturbed updates must be positive")
    if args.arm == "kl-and-argmax" and not args.reference:
        parser.error("Pass --reference to compare the candidate with the earlier matched control")
    references = [read_json(path) for path in args.reference]
    torch.set_num_threads(2)
    train = read_json(args.train_panel)
    val = read_json(args.val_panel)
    manifest = read_json(args.cache_receipt)["view"]["manifest"]
    settings, cfg, _ = read_settings(args.config)
    if not (train["dataset_sha256"] == val["dataset_sha256"] == manifest["dataset_sha256"] == settings["data_sha256"]
            and train["spec"] == val["spec"] == manifest["spec"] and train["split"] == "train"
            and val["index_sha256"] == manifest["files"]["val-added.u32"]["sha256"]
            and manifest["provenance"]["added_holdout_source"] == "pubchem"
            and train["rdkit"] == val["rdkit"] == manifest["rdkit_version"] == rdBase.rdkitVersion):
        raise ValueError("Mismatched train, validation or published count reference")
    train_smiles = [row[2] for row in train["rows"]]
    val_smiles = [row[2] for row in val["rows"]]
    if set(train_smiles) & set(val_smiles) or args.eval_real > len(val_smiles):
        raise ValueError("Invalid or overlapping evaluation inputs")
    spec = GraphSpec.from_dict(train["spec"])
    table = [tensor_row(s, spec) for s in train_smiles]
    selected = random.Random(args.seed + 101).sample(val_smiles, args.eval_real)
    validation = [(collate_graphs([tensor_row(s, spec) for s in selected[i:i + args.microbatch]]),
                   selected[i:i + args.microbatch]) for i in range(0, len(selected), args.microbatch)]
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(cfg.seed)
    model = GraphAutoencoder(spec, cfg.model)
    opt = _optimizer(model, cfg)
    report = {"arguments": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                            "reference": [str(p) for p in args.reference]},
              "dataset_sha256": train["dataset_sha256"], "available_training_rows": len(table),
              "parameter_count": sum(p.numel() for p in model.parameters()), "arms": []}
    path = args.output / "comparison.json"
    write_json(path, report)
    t = perf_counter()
    for i in range(1, args.warmup + 1):
        step(model, opt, cfg, table, args, i)
    report["warmup_seconds"] = perf_counter() - t
    report["warm_model_sha256"] = state_digest(model.state_dict())
    warm_model = copy.deepcopy(model.state_dict())
    warm_opt = copy.deepcopy(opt.state_dict())
    write_json(path, report)
    print("Finished common warmup", flush=True)

    panel = CompactPriorPanel(cfg.model.latent, seed=args.seed + 102, count=args.eval_prior,
                              batch_size=args.microbatch)
    endpoints = (CompactPriorPanel(cfg.model.latent, seed=args.seed + 102, count=args.eval_prior,
                                  batch_size=args.microbatch, stress_radius_multiplier=args.stress_radius_multiplier)
                 if args.stress_radius_multiplier is not None else panel)
    shifts = {"both": (args.capacity_shift, 0.), "stress": (args.capacity_shift,), "undisturbed": (0.,)}
    for shift in shifts[args.starts]:
        budget = args.undisturbed_updates if not shift and args.undisturbed_updates else args.updates
        weights = ((0., 0.), (1., 0.)) if args.arm == "cdf-and-kl" else ((1., 1.),)
        for weight, argmax_weight in weights:
            model.load_state_dict(warm_model)
            opt.load_state_dict(copy.deepcopy(warm_opt))
            with torch.no_grad():
                model.decoder.atom_head.bias[0].sub_(shift)
            guide = CompactPriorGuidance({**settings["prior_guidance"], "count_kl_weight": weight,
                                         "argmax_count_weight": argmax_weight, "workers": 0,
                                         "count_histogram": manifest["active_atom_counts"]["train"]}, spec, "cpu")
            if guide.count_kl_weight != weight or getattr(guide, "argmax_count_weight", 0.) != argmax_weight:
                raise RuntimeError("Prior count-KL ablation was not applied")
            arm = {"no_atom_bias_subtraction": shift, "count_kl_weight": weight, "argmax_count_weight": argmax_weight,
                   "starting_sha256": state_digest(model.state_dict()), "updates": [], "observations": {}}
            if references:
                verify_replay(args, references, shift, budget, report["warm_model_sha256"], arm["starting_sha256"])
            report["arms"].append(arm)
            try:
                arm["observations"][0] = evaluate(model, endpoints, validation)
                write_json(path, report)
                for i in range(1, budget + 1):
                    t = perf_counter()
                    result = step(model, opt, cfg, table, args, args.warmup + i, guide)
                    arm["updates"].append({**result, "update_seconds": perf_counter() - t})
                    if i in {8, 16, budget // 2, budget}:
                        arm["observations"][i] = evaluate(model, endpoints if i == budget else panel, validation)
                        print(f"bias shift {shift}, KL {weight}, argmax {argmax_weight}, update {i}", flush=True)
                    write_json(path, report)
            finally:
                guide.close()
    report["complete"] = True
    write_json(path, report)


if __name__ == "__main__":
    main()
