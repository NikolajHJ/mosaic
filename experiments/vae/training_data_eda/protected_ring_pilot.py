"""Compare ordinary VAE training with aromatic ring reward and frozen-start bond preservation."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

from rdkit import rdBase
import torch

from experiments.vae.training_data_eda.shared_ring_eval import compare_evaluations, evaluate_endpoint
from experiments.vae.training_data_eda.shared_ring_pilot import VAE, pilot_config, select_rows, stream_hashes
from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.train import learning_rate_at_epoch, state_digest
from latentmol.training.combined_continuation import _fresh
from latentmol.training.projected_continuation import projected_update


def aromatic_backward(model, anchor, args, offset, protect, *, amide=False):
    """Accumulate the auxiliary into already-computed base gradients before joint clipping.

    Each equal-budget arm sees the same extra Gaussian points. Use the evaluation
    mode for both branches so identical decoders produce identical anchor outputs.
    Each prior microbatch contributes in proportion to its fraction of the draws.
    """
    from experiments.vae.training_data_eda.protected_ring_objective import protected_ring_loss

    rng = torch.Generator().manual_seed(args.seed + 3_000_000 + offset)
    prior = torch.randn(args.ring_draws, model.config.latent, generator=rng)
    batches = []
    total = 0.
    was_training = model.training
    model.eval()
    try:
        for block in prior.split(args.microbatch):
            z = block.to(args.device)
            with torch.no_grad():
                target = anchor.decode(z)
            decoded = model.decode(z)
            if amide:
                from experiments.vae.training_data_eda.protected_amide_objective import protected_amide_loss
                aux, metrics = protected_amide_loss(decoded, target, model.spec)
                weight = args.amide_weight
            else:
                aux, metrics = protected_ring_loss(decoded, target, model.spec, temperature=args.temperature,
                                                   retention_weight=args.retention_weight if protect else 0.)
                weight = args.ring_weight
            weighted = aux * (weight * len(z) / len(prior))
            if not weighted.requires_grad or not weighted.isfinite():
                raise FloatingPointError("Detached or nonfinite aromatic auxiliary")
            weighted.backward()
            total += float(weighted.detach())
            batches.append(metrics)
    finally:
        model.train(was_training)
    count_key = "contrast_amide_pairs" if amide else "sites"
    return total, dict(weighted_loss=total, draws=len(prior), sites=sum(m[count_key] for m in batches),
                       microbatches=batches)


def run(args):
    from experiments.vae.training_data_eda.protected_ring_measure import build_ring_panel, measure_ring_panel

    budgets = (args.updates, args.draws, args.heldout, args.threads, args.eval_batch, args.ring_every, args.ring_draws)
    if min(budgets) < 1 or not 2 <= args.microbatch <= args.batch_size or args.batch_size % args.microbatch:
        raise ValueError("Require positive budgets and a microbatch of at least two dividing the batch")
    if not all(math.isfinite(v) and v > 0 for v in (
            args.ring_weight, args.retention_weight, args.temperature, args.amide_weight)):
        raise ValueError("Aromatic and retention weights and temperature must be positive and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation has no visible GPU")
    torch.set_num_threads(args.threads)
    source_hash = file_sha256(args.checkpoint)
    if args.checkpoint_sha256 and args.checkpoint_sha256 != source_hash:
        raise ValueError("Source checkpoint SHA-256 mismatch")
    # RNG state in the training checkpoint must remain on CPU for restoration.
    anchor, payload = load_model(args.checkpoint, "cpu")
    anchor.to(args.device).eval()
    config = pilot_config(args, payload)
    reference_hash = file_sha256(args.reference)
    if reference_hash != config["prior_only"]["guidance"]["reference"]["sha256"]:
        raise ValueError("Unexpected frozen prior teacher")
    teacher, _ = load_model(args.reference, args.device, anchor.spec)
    teacher.eval().requires_grad_(False)
    anchor_digest, teacher_digest = state_digest(anchor.state_dict()), state_digest(teacher.state_dict())
    start = payload["continuation"]["updates"]
    epoch = payload["continuation"]["epoch"] + 1
    selection, heldout, row_ids = select_rows(args, anchor.spec)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "rows.json", row_ids)
    summary = dict(checkpoint_sha256=source_hash, reference_sha256=reference_hash, start_update=start, epoch=epoch,
                   config=config, updates=args.updates, seed=args.seed, training_panel_sha256=file_sha256(args.train_panel),
                   heldout_panel_sha256=file_sha256(args.heldout_panel), heldout=args.heldout, gaussian_draws=args.draws,
                   rdkit=rdBase.rdkitVersion, torch=torch.__version__,
                   device=torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                   auxiliary=dict(weight=args.ring_weight, every_updates=args.ring_every, draws=args.ring_draws,
                                  retention_weight=args.retention_weight, temperature=args.temperature,
                                  include_amide=args.include_amide, amide_weight=args.amide_weight),
                   evaluations={}, training={}, ring_panel={})
    write_json(args.output / "plan.json", summary)
    marker = os.environ.get("DBMOL_LAUNCH_READY")
    if marker:
        write_json(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                      nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    with torch.no_grad(), rdBase.BlockLogs():
        panel = build_ring_panel(anchor, seed=args.seed+901, draws=min(1024, args.draws), batch_size=args.eval_batch)

    def assess(name, model):
        with torch.no_grad(), rdBase.BlockLogs():
            model.eval()
            summary["evaluations"][name] = evaluate_endpoint(
                model, None, heldout, args.output / f"{name}.jsonl.gz", draws=args.draws,
                seed=args.seed + 10, batch_size=args.eval_batch)
            summary["ring_panel"][name] = measure_ring_panel(model, panel)
        write_json(args.output / "progress.json", summary)

    assess("start", anchor)
    arms = ["control", "reward", "protected"]
    if args.include_amide:
        arms.append("amide")
    for arm in arms:
        model, optimizer, training, generators = _fresh(anchor, payload, None, args.batch_size, args.device, None)
        if state_digest(model.state_dict()) != anchor_digest or training.loss.atom_order != "graph":
            raise ValueError("Source weights or reconstruction policy changed")
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at_epoch(training, epoch)
        sites = exposures = 0
        since = time.monotonic()
        with (args.output / f"{arm}-training.jsonl").open("x") as log, rdBase.BlockLogs():
            for offset, smiles in enumerate(selection):
                torch.manual_seed(args.seed + 1000 + offset)
                generators["fresh_prior"].manual_seed(args.seed + 1_000_000 + offset)
                hard = encode(smiles, model.spec)
                aux = None
                if arm != "control" and offset % args.ring_every == 0:
                    def aux():
                        return aromatic_backward(model, anchor, args, offset, arm == "protected", amide=arm == "amide")
                metrics = projected_update(config, model, teacher, optimizer, hard, training, None, None,
                                           generators, start+offset, epoch, args.microbatch, auxiliary_before_clip=aux)
                if "auxiliary" in metrics:
                    sites += metrics["auxiliary"]["sites"]
                    exposures += 1
                log.write(json.dumps(dict(update=start + offset + 1, **metrics), allow_nan=False) + "\n")
                if offset == 0 or (offset + 1) % 16 == 0:
                    log.flush()
                    print(f"{arm} {offset+1}/{args.updates}; selected sites {sites}", flush=True)
        if (state_digest(teacher.state_dict()) != teacher_digest or state_digest(anchor.state_dict()) != anchor_digest
                or any(p.grad is not None for p in anchor.parameters())):
            raise RuntimeError("Frozen weights changed or received gradients")
        summary["training"][arm] = dict(seconds=time.monotonic()-since, source_sha256=source_hash, sites=sites,
                                        exposures=exposures, model_sha256=state_digest(model.state_dict()),
                                        end_rng=stream_hashes(generators, args.device))
        fields = {k: payload[k] for k in ("format_version", "model_config", "graph_spec")}
        torch.save(dict(fields, model_state=model.state_dict(), diagnostic_only=True,
                        updates=args.updates, source_sha256=source_hash), args.output / f"{arm}.pt")
        assess(arm, model)
        del model, optimizer
        gc.collect()
    summary["paired"] = {}
    comparisons = [("control", "reward"), ("reward", "protected"), ("control", "protected")]
    if args.include_amide:
        comparisons.append(("control", "amide"))
    for a, b in comparisons:
        summary["paired"][f"{a}_to_{b}"] = compare_evaluations(args.output/f"{a}.jsonl.gz",
                                                               args.output/f"{b}.jsonl.gz")
    write_json(args.output / "summary.json", summary)
    print("Pilot complete", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "reference", "train-panel", "heldout-panel", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config", type=Path, default=VAE / "epoch48-seven-round-generated-roundtrip.json")
    parser.add_argument("--policy", type=Path, default=VAE / "epoch48-encoder-targets.json")
    for key, default in (("updates",128),("batch-size",64),("microbatch",16),("draws",4096),("heldout",256),
                         ("eval-batch",16),("seed",2026112517),("threads",4),("ring-every",2),("ring-draws",32)):
        parser.add_argument("--"+key, type=int, default=default)
    parser.add_argument("--ring-weight", type=float, default=.1)
    parser.add_argument("--include-amide", action="store_true", help="Train an additional amide-only fork.")
    parser.add_argument("--amide-weight", type=float, default=.1)
    parser.add_argument("--retention-weight", type=float, default=1.)
    parser.add_argument("--temperature", type=float, default=2.)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
