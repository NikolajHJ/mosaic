"""Measure weak motif guidance on a saved VAE and time matched optimizer updates."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
import time

from rdkit import rdBase
import torch

from latentmol.chemistry import encode_canonical_smiles
from latentmol.datasets.data import collate_graphs
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.guidance.functional_groups import _motif_labels, functional_group_loss
from latentmol.representation import DecoderOutput
from latentmol.training.projected_continuation import _student, projected_update


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_neighbor_auxiliary(args, model, policy):
    """Isolate the additional site search, neighboring VAE forward and its backward pass."""
    from latentmol.guidance.motif_neighbors import prior_motif_neighbor_loss

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model.to(device).train()
    prior = torch.Generator().manual_seed(args.seed)
    measurements = []
    for i in range(args.panels + 3):
        z = torch.randn(policy["draws"], model.config.latent, generator=prior).to(device)
        with torch.no_grad():
            center = model.decode(z)
        # Production already has these center logits from its prior branch.
        # Use detached leaves here so that existing reconstruction and prior
        # gradients are outside the measurement, including the no-site case.
        decoded = DecoderOutput(*(t.detach().requires_grad_() for t in (center.atoms, center.charges, center.bonds)))
        jitter = torch.Generator(device=device).manual_seed(args.seed + i + 1)
        model.zero_grad(set_to_none=False)
        synchronize(device)
        start = time.perf_counter()
        auxiliary, info = prior_motif_neighbor_loss(model, z, decoded, model.spec, jitter)
        (policy["weight"] * policy["neighbor_weight"] * auxiliary).backward()
        synchronize(device)
        seconds = time.perf_counter() - start
        if i >= 3:
            measurements.append({"seconds": seconds, **info})
    seconds = [r["seconds"] for r in measurements]
    return dict(scope="Reused center logits; timed extra RDKit inspection, neighbor forward and backward only",
                warmups=3, panels=len(seconds), mean_scheduled_seconds=mean(seconds),
                median_scheduled_seconds=median(seconds), mean_seconds_per_training_update=mean(seconds) / args.every,
                aromatic_sites=sum(r["neighbor_aromatic_sites"] for r in measurements),
                amide_sites=sum(r["neighbor_amide_sites"] for r in measurements),
                centers=sum(r["neighbor_center_count"] for r in measurements), panels_detail=measurements)


def motif_probe(args, model, policy):
    """Use independent prior vectors, retaining inputs and a single categorical draw per vector."""
    device = torch.device(args.device)
    model = model.to(device).eval()
    rng = torch.Generator().manual_seed(args.seed)
    draw_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    counts = Counter()
    head_norms, auxiliary_seconds = [], []
    for _ in range(args.panels):
        z = torch.randn(policy["draws"], model.config.latent, generator=rng)
        with torch.no_grad():
            parts = [model.decode(chunk.to(device)) for chunk in z.split(args.microbatch)]
        fields = [torch.cat([getattr(p, name) for p in parts]).requires_grad_()
                  for name in ("atoms", "charges", "bonds")]
        decoded = DecoderOutput(*fields)
        labels, valid = _motif_labels(decoded.harden(model.spec), model.spec)
        counts.update(argmax_valid=valid, argmax_aromatic=int(labels[:, 0].sum()),
                      argmax_amide=int(labels[:, 1].sum()))
        synchronize(device)
        started = time.perf_counter()
        loss, info = functional_group_loss(decoded, model.spec, draw_rng,
            aromatic_floor=policy["aromatic_floor"], amide_floor=policy["amide_floor"])
        gradients = torch.autograd.grad(policy["weight"] * loss, fields)
        synchronize(device)
        auxiliary_seconds.append(time.perf_counter() - started)
        norms = [float(g.norm()) for g in gradients]
        if not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Nonfinite motif gradient")
        head_norms.append(norms)
        counts.update(sample_valid=info["valid"], sample_aromatic=info["aromatic_hits"],
                      sample_amide=info["amide_hits"], draws=info["draws"],
                      batches_aromatic_signal=int(info["aromatic_informative"]),
                      batches_amide_signal=int(info["amide_informative"]),
                      batches_nonzero_gradient=int(any(norms)))
    return dict(panels=args.panels, counts=counts, weighted_logits_gradient_norms=head_norms,
                auxiliary_seconds=auxiliary_seconds,
                median_auxiliary_seconds=median(auxiliary_seconds[1:] or auxiliary_seconds),
                scope="eval-mode decoder; timed region is sampling, chemistry, and loss backward to logits")


def time_updates(args, parent, payload, config, policy):
    """Restore equal states per pair. Exclude model restoration and host panel loading."""
    device = torch.device(args.device)
    reference, _ = load_model(args.reference, args.device, parent.spec)
    reference.eval().requires_grad_(False)
    source = json.loads(args.panel.read_text())["rows"]
    if args.batch_size > len(source):
        raise ValueError("Real panel is smaller than the logical batch")
    graphs = [encode_canonical_smiles(r[2], parent.spec) for r in source[:args.batch_size]]
    hard = collate_graphs([{k: torch.as_tensor(g[v]).long() for k, v in
        (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))} for g in graphs])
    config = dict(config, projection=dict(config["projection"],
                  reference_priors=min(args.microbatch, config["projection"]["reference_priors"])))
    if policy["draws"] > args.microbatch:
        raise ValueError("The timed functional-group subset must fit the local microbatch")
    elapsed = {"control": [], "motif": []}
    measurements = []
    for rep in range(args.pairs + 1):
        pair = {}
        gradients = {}
        for arm in (("control", "motif") if rep % 2 == 0 else ("motif", "control")):
            model, optimizer, training, generators = _student(
                parent, payload, None, args.batch_size, args.device, None,
                count=0, seed=config["extension_seed"], atom_order=config["atom_order"],
                atom_refinement_dropout=config["atom_refinement_dropout"])
            torch.manual_seed(args.seed + rep)
            generators["fresh_prior"].manual_seed(args.seed + rep + 1)
            arm_config = config if arm == "control" else dict(config, functional_groups=policy)
            step = config["parent_updates"] + (rep + 1) * policy["every_updates"] - 1
            synchronize(device)
            start = time.perf_counter()
            metrics = projected_update(arm_config, model, reference, optimizer, hard, training,
                                       None, None, generators, step, 48, args.microbatch)
            synchronize(device)
            dt = time.perf_counter() - start
            if rep:
                elapsed[arm].append(dt)
            pair[arm] = dict(seconds=dt, metrics=metrics)
            if args.compare_gradients and rep:
                # torch.clip_grad_norm_ scales all populated gradients by this coefficient.
                clip = min(1., training.clip_grad / (metrics["pre_clip_gradient_norm"] + 1e-6))
                gradients[arm] = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()
                                            if p.grad is not None]).cpu().double() / clip
            del model, optimizer, generators
        if gradients:
            delta = gradients["motif"] - gradients["control"]
            delta_norm = float(delta.norm())
            base_norm = float(gradients["control"].norm())
            pair["gradient"] = dict(base_norm=base_norm,
                additional_norm=delta_norm, relative_norm=delta_norm / base_norm,
                cosine_to_base=float(torch.dot(gradients["control"], delta) /
                                     (base_norm * delta_norm)) if delta_norm else None)
        if rep:
            measurements.append(pair)
        print("step pair", rep, {a: round(pair[a]["seconds"], 3) for a in elapsed}, flush=True)
    control = median(elapsed["control"])
    deltas = [a - b for a, b in zip(elapsed["motif"], elapsed["control"])]
    every = policy["every_updates"]
    return dict(batch_size=args.batch_size, microbatch=args.microbatch,
                reference_priors=config["projection"]["reference_priors"], pairs=args.pairs,
                median_control_seconds=control, median_scheduled_seconds=median(elapsed["motif"]),
                median_paired_delta_seconds=median(deltas), paired_delta_seconds=deltas,
                projected_amortized_fraction=median(deltas) / control / every, pairs_detail=measurements,
                scope="real + prior projected optimizer updates; preloaded fixed real graph panel, no data I/O or evaluation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["probe", "timing", "neighbor-overhead"], required=True)
    for key in ("checkpoint", "config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--panel", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--draws", type=int, default=48)
    p.add_argument("--microbatch", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=240)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--compare-gradients", action="store_true")
    p.add_argument("--panels", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026112191)
    p.add_argument("--weight", type=float, default=0.3)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--aromatic-floor", type=float, default=0.05)
    p.add_argument("--amide-floor", type=float, default=0.08)
    args = p.parse_args()
    torch.set_num_threads(3)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    policy = dict(weight=args.weight, every_updates=args.every, draws=args.draws, seed=2026112148,
                  aromatic_floor=args.aromatic_floor, amide_floor=args.amide_floor)
    parent, payload = load_model(args.checkpoint, "cpu")
    with rdBase.BlockLogs():
        if args.mode == "probe":
            result = motif_probe(args, parent, policy)
        elif args.mode == "neighbor-overhead":
            policy["neighbor_weight"] = 3.
            result = time_neighbor_auxiliary(args, parent, policy)
        else:
            if args.reference is None or args.panel is None:
                p.error("timing needs --reference and --panel")
            result = time_updates(args, parent, payload, config, policy)
    report = dict(mode=args.mode, seed=args.seed, checkpoint=file_sha256(args.checkpoint),
                  update=payload["continuation"]["updates"], policy=policy, torch=torch.__version__,
                  device=torch.cuda.get_device_name() if args.device == "cuda" else args.device,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                  result=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print({k: v for k, v in result.items() if k not in
           ("pairs_detail", "panels_detail", "weighted_logits_gradient_norms", "auxiliary_seconds")}, flush=True)


if __name__ == "__main__":
    main()
