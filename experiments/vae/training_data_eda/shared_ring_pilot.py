"""Matched short VAE forks for ring-teacher masking and a separately fitted whole-ring head."""

import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from rdkit import rdBase
import torch

from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.train import learning_rate_at_epoch, state_digest
from latentmol.training.combined_continuation import _fresh
from latentmol.training.motif_activation import _digest, policies
from latentmol.training.projected_continuation import projected_update


VAE = Path(__file__).resolve().parents[1]


def pilot_config(args, checkpoint):
    """Restore the live leg's selected objectives, resizing only draws needing a smaller pilot batch."""
    config = json.loads(args.config.read_bytes())
    metadata = checkpoint["experiment_state"]
    _, candidates = policies(config)
    activation = metadata["motif_activation"]
    chosen = candidates[activation["winner"]]
    if activation["policy_sha256"] != _digest(chosen) or checkpoint["dataset_sha256"] != config["data_sha256"]:
        raise ValueError("Source checkpoint and selected motif policy or data lineage differ")
    config["functional_groups"] = dict(chosen, score_draws=min(chosen["score_draws"], args.batch_size),
                                        draws=min(chosen["draws"], args.microbatch))
    current = json.loads(args.policy.read_bytes())
    if "generated_bond_handoff" in metadata and metadata["generated_bond_handoff"]["policy"] != current:
        raise ValueError("Checkpoint bond policy differs from the requested production objective")
    roundtrip = dict(config["generated_roundtrip"], **current,
                     excess_confidence=metadata["encoder_upgrade"]["policy"]["excess_confidence"])
    roundtrip["draws"] = min(roundtrip["draws"], args.microbatch)
    roundtrip["max_parents"] = min(roundtrip["max_parents"], roundtrip["draws"])
    config["generated_roundtrip"] = roundtrip
    config["projection"]["reference_priors"] = min(config["projection"]["reference_priors"], args.microbatch)
    config["batch_size"], config["microbatch"] = args.batch_size, args.microbatch
    return config


def select_rows(args, spec):
    with gzip.open(args.train_panel, "rt") as f:
        train = [r["smiles"] for r in csv.DictReader(f, delimiter="\t")]
    held = [row[2] for row in json.loads(args.heldout_panel.read_bytes())["rows"]]
    if set(train).intersection(held):
        raise ValueError("Training and validation packets overlap")
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(len(train), args.batch_size * args.updates, replace=False)
    selection = [[train[i] for i in group] for group in ids.reshape(args.updates, args.batch_size)]
    hold_ids = np.random.default_rng(args.seed + 1).choice(len(held), args.heldout, replace=False)
    validation = []
    for pos in range(0, len(hold_ids), args.eval_batch):
        smiles = [held[i] for i in hold_ids[pos:pos + args.eval_batch]]
        validation.append((encode(smiles, spec), smiles))
    return selection, validation, dict(train_indices=ids.tolist(), heldout_indices=hold_ids.tolist())


def stream_hashes(generators, device):
    states = {"cpu": torch.get_rng_state(), "prior": generators["fresh_prior"].get_state()}
    if device == "cuda":
        states["cuda"] = torch.cuda.get_rng_state()
    return {key: hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for key, x in states.items()}


def run(args):
    from experiments.vae.training_data_eda.shared_ring_eval import compare_evaluations, evaluate_endpoint
    from experiments.vae.training_data_eda.shared_ring_gate import SharedRingGate, gate_training_loss

    if not (2 <= args.microbatch <= args.batch_size and args.batch_size % args.microbatch == 0):
        raise ValueError("Microbatch must divide the real batch and contain at least two graphs")
    if min(args.updates, args.eval_batch, args.heldout, args.draws, args.threads) < 1:
        raise ValueError("Budgets must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("A CUDA run requires a visible GPU")
    torch.set_num_threads(args.threads)
    source_hash = file_sha256(args.checkpoint)
    if args.checkpoint_sha256 and args.checkpoint_sha256 != source_hash:
        raise ValueError("Source checkpoint SHA-256 mismatch")
    parent, payload = load_model(args.checkpoint, "cpu")
    config = pilot_config(args, payload)
    start = payload["continuation"]["updates"]
    epoch = payload["continuation"]["epoch"] + 1
    reference_hash = file_sha256(args.reference)
    if reference_hash != config["prior_only"]["guidance"]["reference"]["sha256"]:
        raise ValueError("Unexpected frozen prior teacher")
    teacher, _ = load_model(args.reference, args.device, parent.spec)
    teacher.eval().requires_grad_(False)
    parent_hash, teacher_hash = state_digest(parent.state_dict()), state_digest(teacher.state_dict())
    train, validation, row_ids = select_rows(args, parent.spec)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "rows.json", row_ids)
    summary = dict(checkpoint_sha256=source_hash, start_update=start, epoch=epoch,
                   reference_sha256=reference_hash, config=config, updates=args.updates,
                   seed=args.seed, training_panel_sha256=file_sha256(args.train_panel),
                   heldout_panel_sha256=file_sha256(args.heldout_panel), heldout=args.heldout,
                   gaussian_draws=args.draws, rdkit=rdBase.rdkitVersion, torch=torch.__version__,
                   device=torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                   gate_step="Independent gate BCE after each teacher-mask step using the same real rows.",
                   evaluations={}, training={})
    write_json(args.output / "plan.json", summary)
    marker = os.environ.get("DBMOL_LAUNCH_READY")
    if marker:
        write_json(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                      nonce=os.environ["DBMOL_LAUNCH_NONCE"]))

    def assess(name, model, gate=None):
        model.eval()
        if gate is not None:
            gate.eval()
        with rdBase.BlockLogs():
            summary["evaluations"][name] = evaluate_endpoint(
                model, gate, validation, args.output / f"{name}.jsonl.gz", draws=args.draws,
                seed=args.seed + 10, batch_size=args.eval_batch)
        write_json(args.output / "progress.json", summary)

    for arm in ("control", "teacher_mask"):
        model, optimizer, training, generators = _fresh(parent, payload, None, args.batch_size, args.device, None)
        if state_digest(model.state_dict()) != parent_hash or training.loss.atom_order != "graph":
            raise ValueError("Source weights or reconstruction policy changed")
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at_epoch(training, epoch)
        if arm == "control":
            assess("start", model)
        gate = None
        if arm == "teacher_mask":
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(args.seed + 12)
                gate = SharedRingGate(model.spec).to(args.device)
            gate_opt = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr, weight_decay=0.001)
        arm_config = dict(config, functional_groups=dict(config["functional_groups"],
                                                        release_ring_teacher=(arm == "teacher_mask")))
        # Per-update seeds keep stochastic paths matched even if a model-dependent
        # number of eligible generated parents changes how many RNG draws it uses.
        model.train()
        gate_counts = dict(rows=0, sites=0, positive=0, negative=0)
        mask_pairs = 0
        since = time.monotonic()
        with (args.output / f"{arm}-training.jsonl").open("x") as log, rdBase.BlockLogs():
            for offset, smiles in enumerate(train):
                torch.manual_seed(args.seed + 1000 + offset)
                generators["fresh_prior"].manual_seed(args.seed + 1000000 + offset)
                hard = encode(smiles, model.spec)
                metrics = projected_update(arm_config, model, teacher, optimizer, hard, training,
                                           None, None, generators, start + offset, epoch, args.microbatch)
                mask_pairs += metrics.get("ring_teacher", {}).get("masked_pairs", 0)
                if gate is not None:
                    # No decoder or encoder is part of this independent classifier's graph.
                    gate_opt.zero_grad(set_to_none=True)
                    loss, counts = gate_training_loss(gate, hard, model.spec)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite shared-ring objective")
                    if counts["sites"]:
                        loss.backward()
                        if any(p.grad is None for p in gate.parameters()):
                            raise RuntimeError("Gate parameters detached from a nonempty training batch")
                        torch.nn.utils.clip_grad_norm_(gate.parameters(), 5, error_if_nonfinite=True)
                        gate_opt.step()
                    metrics["gate"] = dict(counts, loss=float(loss.detach()))
                    for key in gate_counts:
                        gate_counts[key] += int(counts[key])
                log.write(json.dumps(dict(update=start + offset + 1, **metrics), allow_nan=False) + "\n")
                if offset == 0 or (offset + 1) % 16 == 0:
                    log.flush()
                    print(f"{arm} {offset+1}/{args.updates}", flush=True)
        if state_digest(teacher.state_dict()) != teacher_hash or state_digest(parent.state_dict()) != parent_hash:
            raise RuntimeError("Frozen source weights changed")
        summary["training"][arm] = dict(seconds=time.monotonic() - since, masked_pairs=mask_pairs,
                                        end_rng=stream_hashes(generators, args.device),
                                        model_sha256=state_digest(model.state_dict()), gate=gate_counts)
        saved = {k: payload[k] for k in ("format_version", "model_config", "graph_spec")}
        torch.save(dict(saved, model_state=model.state_dict(), diagnostic_only=True,
                        source_sha256=source_hash, updates=args.updates), args.output / f"{arm}.pt")
        assess(arm, model)
        if gate is not None:
            if any(not torch.isfinite(v).all() for v in gate.parameters()):
                raise FloatingPointError("Nonfinite ring gate parameters")
            torch.save(dict(state=gate.state_dict(), graph_spec=model.spec.to_dict(),
                            base_model_sha256=summary["training"][arm]["model_sha256"],
                            learning_rate=args.gate_lr, seed=args.seed + 12), args.output / "ring_gate.pt")
            assess("teacher_mask_gate", model, gate)
        del model, optimizer
        gc.collect()
    summary["paired"] = {}
    for a, b in (("control", "teacher_mask"), ("teacher_mask", "teacher_mask_gate"),
                 ("control", "teacher_mask_gate")):
        summary["paired"][f"{a}_to_{b}"] = compare_evaluations(args.output / f"{a}.jsonl.gz",
                                                                 args.output / f"{b}.jsonl.gz")
    write_json(args.output / "summary.json", summary)
    print("Pilot complete", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "reference", "train-panel", "heldout-panel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=VAE / "epoch48-seven-round-generated-roundtrip.json")
    parser.add_argument("--policy", type=Path, default=VAE / "epoch48-encoder-targets.json")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--heldout", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026112517)
    parser.add_argument("--gate-lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
