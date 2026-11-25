"""Start a projected-training arm at epoch 47 or resume it through epoch 48."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from latentmol.chemistry_continuation import chunks, hard_slice, reconstruction_chunk, reconstruction_denominators
from latentmol.chemistry_optimizer_step import apply_optimizer_step
from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.functional_groups import functional_group_loss, prepare_pooled_motif_score
from latentmol.guidance.valence_projection import projection_loss
from latentmol.models import GraphAutoencoder
from latentmol.models.autoencoder import comparable_model_config
from latentmol.representation import DecoderOutput
from latentmol.training.combined_continuation import VERSION, _check_cuda_requirement, _fresh, _load, _train
from latentmol.training.losses import LossConfig, latent_kl
from latentmol.training.prior_repair import conditional_parent_loss, occupancy_logits, parent_prior_targets


def append_rounds(model, optimizer, count, seed):
    """Keep the inherited Parameter objects and their AdamW moments, including counters."""
    if model.config.extra_bond_rounds or len(optimizer.param_groups) != 1:
        raise ValueError("Expected an unextended parent and one AdamW group")
    if not count:
        return
    config = replace(model.config, extra_bond_rounds=count)
    with torch.random.fork_rng(devices=[]):
        # Only draw on the CPU. Leave the continuation's prior, dropout and posterior RNGs intact.
        torch.random.default_generator.manual_seed(seed)
        extra = GraphAutoencoder(model.spec, config).decoder.extra_rounds
    model.decoder.extra_rounds = extra.to(next(model.parameters()).device)
    model.config = config
    group = optimizer.param_groups[0]["params"]
    group.extend(extra.parameters())
    if [id(p) for p in group] != [id(p) for p in model.parameters()]:
        raise RuntimeError("Expanded AdamW parameter order would change on checkpoint reload")


def _student(parent, payload, anchor, batch_size, device, resume, *, count, seed, atom_order, **kwargs):
    if resume is not None and count:
        # AdamW restores moments by parameter position. Build all seven rounds
        # before strict restoration; _fresh restores the saved RNG afterwards.
        parent = GraphAutoencoder(parent.spec, replace(parent.config, extra_bond_rounds=count))
    model, optimizer, training, generators = _fresh(parent, payload, anchor, batch_size, device, resume, **kwargs)
    if resume is None:
        append_rounds(model, optimizer, count, seed)
    if atom_order != "graph":
        raise ValueError("Both paired workers require graph-matched reconstruction")
    training = replace(training, model=model.config, loss=replace(training.loss, atom_order=atom_order))
    return model, optimizer, training, generators


def _read_resume(path, identity, config, model, payload, count, train_size):
    """Authenticate the predecessor and its unfinished position in the original shuffled pass."""
    receipt = json.loads(path.read_bytes())
    source = Path(receipt["checkpoint"]).resolve()
    expected = path.resolve().parent / f"checkpoint-{receipt['updates']:09d}.pt"
    if source != expected or file_sha256(source) != receipt["sha256"]:
        raise ValueError("Resume checkpoint path or SHA-256 differs from the segment receipt")
    state = torch.load(source, map_location="cpu", weights_only=True)
    loss = replace(LossConfig(**payload["loss_config"]), atom_order=config["atom_order"])
    if (state.get("config_sha256") != identity or state.get("combined_version") != VERSION
            or state.get("diagnostic_only") or state.get("format_version") != 1
            or state.get("dataset_sha256") != config["data_sha256"]
            or state.get("graph_spec") != model.spec.to_dict()
            or comparable_model_config(state.get("model_config")) != asdict(replace(model.config, extra_bond_rounds=count))
            or state.get("loss_config") != asdict(loss)
            or state.get("batch_size") != config["batch_size"] or state.get("microbatch_size") != config["microbatch"]):
        raise ValueError("Resume checkpoint differs from the projected training arm or schema")
    cursor = state["continuation"]
    row = cursor["row"]
    if (any(cursor[key] != receipt[key] for key in ("epoch", "row", "updates", "real_points"))
            or cursor["epoch"] != config["parent_epoch"] or state["epoch"] != cursor["epoch"]
            or receipt["completed_epoch"] != cursor["epoch"] or cursor["epoch_complete"]
            or not 0 < row < train_size or row % config["batch_size"] or cursor["real_points"] != row
            or cursor["updates"] != config["parent_updates"] + row // config["batch_size"]):
        raise ValueError("Resume cursor is not an unfinished position in the projected training epoch")
    return receipt, state


def _arm_identity(config_path, parent_sha, count):
    return hashlib.sha256(f"{file_sha256(config_path)}:{parent_sha}:{count}".encode()).hexdigest()


def _source_identity(config_path, previous_path, parent_sha, count):
    """Bind the exact previous recipe at either supported seven-round objective handoff."""
    source = json.loads(previous_path.read_bytes())
    target = json.loads(config_path.read_bytes())
    if "generated_roundtrip" in target:
        target.pop("generated_roundtrip")
        if (count != 3 or source != target or "generated_roundtrip" in source
                or source.get("functional_groups", {}).get("activation_gate", {}).get("version") != 2):
            raise ValueError("The seven-round handoff may only add generated_roundtrip to the ring comparison recipe")
        return _arm_identity(previous_path, parent_sha, count)
    source_motifs = source.pop("functional_groups", None)
    target_motifs = target.pop("functional_groups", None)
    if (count != 3 or source != target or not source_motifs or not target_motifs or source_motifs == target_motifs):
        raise ValueError("Objective handoff requires a seven-round arm differing only in functional_groups")
    return _arm_identity(previous_path, parent_sha, count)


def _scheduled_motif_policy(config, update, count, microbatch):
    """Use an existing prior microbatch. `update` counts completed optimizer steps."""
    policy = config.get("functional_groups")
    if policy is None:
        return None
    for key in ("draws", "every_updates", "seed"):
        if type(policy[key]) is not int:
            raise ValueError(f"functional_groups.{key} must be an integer")
    if not 2 <= policy["draws"] <= min(count, microbatch):
        raise ValueError("Functional-group draws must fit in the first prior microbatch")
    if "score_draws" in policy and (type(policy["score_draws"]) is not int
            or not policy["draws"] <= policy["score_draws"] <= count):
        raise ValueError("Functional-group score_draws must cover local draws and fit the logical prior batch")
    if "ring_size" in policy:
        from latentmol.guidance.ring_size import validate_ring_policy

        if "score_draws" not in policy:
            raise ValueError("Ring-size supervision requires a pooled score_draws setting")
        validate_ring_policy(policy["ring_size"], config["data_sha256"])
    if policy["every_updates"] < 1 or not 0 <= policy["seed"] < 2**63:
        raise ValueError("Invalid functional-group sampling interval or seed")
    if not math.isfinite(policy["weight"]) or policy["weight"] < 0:
        raise ValueError("Functional-group weight must be finite and nonnegative")
    if not all(0 <= policy[key] <= 1 for key in ("aromatic_floor", "amide_floor")):
        raise ValueError("Functional-group yield floors must lie between zero and one")
    step_in_leg = update - config["parent_updates"] + 1
    if policy["weight"] and step_in_leg % policy["every_updates"] == 0:
        return policy
    return None


def _scheduled_roundtrip_policy(config, update, count, microbatch):
    policy = config.get("generated_roundtrip")
    if policy is None:
        return None
    optional = {"excess_confidence", "parent_source", "false_positive_bonds"}
    if not isinstance(policy, dict) or set(policy) - optional != {"draws", "max_parents", "every_updates", "weight"}:
        raise ValueError("Invalid generated_roundtrip settings")
    from latentmol.training.generated_roundtrip import validate_parent_options

    validate_parent_options(policy.get("parent_source", "train"), policy.get("false_positive_bonds"))
    if "excess_confidence" in policy:
        from latentmol.training.confidence_margins import validate_excess_confidence
        validate_excess_confidence(policy["excess_confidence"])
    if (any(type(policy[key]) is not int for key in ("draws", "max_parents", "every_updates"))
            or not 1 <= policy["max_parents"] <= policy["draws"] <= min(count, microbatch)
            or policy["every_updates"] < 1):
        raise ValueError("Round-trip parent and draw budgets must fit the first prior microbatch with positive cadence")
    if type(policy["weight"]) not in (int, float) or not math.isfinite(policy["weight"]) or policy["weight"] < 0:
        raise ValueError("Round-trip weight must be finite and nonnegative")
    step_in_leg = update - config["parent_updates"] + 1
    return policy if policy["weight"] and step_in_leg % policy["every_updates"] == 0 else None


def projected_update(config, model, reference, optimizer, hard, training, composition, teacher, generators,
                     update, epoch, microbatch, *, auxiliary_before_clip=None):
    policy = config["projection"]
    count = len(hard.atoms)
    n_reference = policy["reference_priors"]
    if not 1 <= n_reference <= min(count, microbatch):
        raise ValueError("The reference subset must fit in the first prior microbatch")
    motif_policy = _scheduled_motif_policy(config, update, count, microbatch)
    roundtrip_policy = _scheduled_roundtrip_policy(config, update, count, microbatch)
    device = next(model.parameters()).device
    hard = hard.to(device)
    epsilon = torch.randn(count, model.config.latent, device=device)
    prior = torch.randn(count, model.config.latent, generator=generators["fresh_prior"]).to(device)
    presence, labels, valid = parent_prior_targets(reference, prior[:n_reference])
    denominators = reconstruction_denominators(hard, training.loss, dtype=next(model.parameters()).dtype)
    beta = training.beta * min(epoch / max(training.kl_warmup_epochs, 1), 1.)
    optimizer.zero_grad(set_to_none=True)
    reconstruction = kl_total = repair = 0.
    for start, stop in chunks(count, microbatch):
        target = hard_slice(hard, start, stop)
        mu, logvar = model.encoder(target, from_canonical_cache=True)
        rec = reconstruction_chunk(model.decode(mu + epsilon[start:stop] * (.5 * logvar).exp()),
                                   target, training.loss, denominators)
        kl = latent_kl(mu, logvar) * ((stop - start) / count)
        _backward(rec + beta * kl)
        reconstruction += float(rec.detach())
        kl_total += float(kl.detach())
    stats = {}
    guide = config["prior_only"]["guidance"]
    weight = policy["weight"]
    motif_metrics = {}
    weighted_motif = 0.
    weighted_roundtrip = 0.
    pooled = None
    if motif_policy and "score_draws" in motif_policy:
        sampler = torch.Generator(device=device).manual_seed(motif_policy["seed"] + update)
        pooled = prepare_pooled_motif_score(model, prior, motif_policy, microbatch, sampler)
    for start, stop in chunks(count, microbatch):
        decoded = model.decode(prior[start:stop])
        loss, details = projection_loss(decoded, model.spec, policy["max_edits"])
        for key, value in details.items():
            stats[key] = stats.get(key, 0) + value
        scaled = loss * ((stop - start) / count)
        repair += float(scaled.detach())
        objective = weight * scaled
        if pooled is not None and start in pooled.samples:
            objective = objective + motif_policy["weight"] * pooled.surrogate(start, decoded, model.spec)
        if start == 0:
            subset = DecoderOutput(*(x[:n_reference] for x in (decoded.atoms, decoded.charges, decoded.bonds)))
            occupancy = F.binary_cross_entropy_with_logits(occupancy_logits(subset.atoms), presence)
            if config.get("functional_groups", {}).get("release_ring_teacher", False):
                from latentmol.guidance.ring_teacher import conditional_ring_teacher

                conditional, ring_stats = conditional_ring_teacher(subset, labels, valid, model.spec)
                motif_metrics["ring_teacher"] = ring_stats
            else:
                conditional = conditional_parent_loss(subset, labels, valid)
            objective = objective + guide["occupancy_weight"] * occupancy + guide["conditional_weight"] * conditional
            bce, ce = float(occupancy.detach()), float(conditional.detach())
            if motif_policy:
                # An update-keyed generator keeps the base arm's prior, posterior and
                # dropout draws paired. The sampled labels never enter the base losses.
                generator = torch.Generator(device=device).manual_seed(motif_policy["seed"] + update)
                n = motif_policy["draws"]
                subset = DecoderOutput(decoded.atoms[:n], decoded.charges[:n], decoded.bonds[:n])
                objective_keys = ("aromatic_floor", "amide_floor", "completion_weight", "ring_encoding",
                                  "amide_relative_weight", "completion_mode")
                objective_args = {key: motif_policy[key] for key in objective_keys if key in motif_policy}
                neighbor_weight = motif_policy.get("neighbor_weight", 0.)
                if not math.isfinite(neighbor_weight) or neighbor_weight < 0:
                    raise ValueError("Motif-neighbor weight must be finite and nonnegative")
                if neighbor_weight:
                    objective_args["count_roundtrip_failures"] = True
                if pooled is not None:
                    objective_args["pooled_score"] = pooled
                motif, observations = functional_group_loss(subset, model.spec, generator, **objective_args)
                if neighbor_weight:
                    from latentmol.guidance.motif_neighbors import prior_motif_neighbor_loss

                    jitter = torch.Generator(device=device).manual_seed(motif_policy["seed"] + update + 1_000_000)
                    neighbor, diagnostics = prior_motif_neighbor_loss(model, prior[:n], subset, model.spec, jitter)
                    motif = motif + neighbor_weight * neighbor
                    observations.update(diagnostics, neighbor_loss=float(neighbor.detach()))
                if not motif.requires_grad:
                    raise RuntimeError("Detached functional-group training objective")
                objective = objective + motif_policy["weight"] * motif
                weighted_motif = motif_policy["weight"] * float(motif.detach())
                motif_metrics["functional_groups"] = dict(observations, weighted_loss=weighted_motif)
            if roundtrip_policy:
                from latentmol.training.generated_roundtrip import generated_roundtrip_loss

                n = roundtrip_policy["draws"]
                centers = DecoderOutput(decoded.atoms[:n], decoded.charges[:n], decoded.bonds[:n])
                cycle, cycle_stats = generated_roundtrip_loss(
                    model, centers, training.loss, max_parents=roundtrip_policy["max_parents"],
                    prior_latents=prior[:n], excess_confidence=roundtrip_policy.get("excess_confidence"),
                    parent_source=roundtrip_policy.get("parent_source", "train"),
                    false_positive_bonds=roundtrip_policy.get("false_positive_bonds"))
                if cycle_stats["selected"] and not cycle.requires_grad:
                    raise RuntimeError("Detached generated round-trip training objective")
                objective = objective + roundtrip_policy["weight"] * cycle
                weighted_roundtrip = roundtrip_policy["weight"] * float(cycle.detach())
                motif_metrics["generated_roundtrip"] = dict(cycle_stats, weighted_loss=weighted_roundtrip)
        _backward(objective)
    # A pilot can add gradients to the same AdamW step. The callable backpropagates
    # one microbatch at a time and returns the weighted objective value and metrics.
    extra_value = 0.
    if auxiliary_before_clip is not None:
        extra_value, extra_metrics = auxiliary_before_clip()
        if not math.isfinite(extra_value):
            raise FloatingPointError("Nonfinite pilot auxiliary objective")
        motif_metrics["auxiliary"] = extra_metrics
    required = [model.encoder.mu.weight, model.decoder.atom_head.weight, model.decoder.bond_head[-1].weight]
    for layer in model.decoder.extra_rounds:
        required.extend((layer.atom_gate, layer.pair_gate))
    if any(p.grad is None for p in required):
        raise RuntimeError("Detached required projection-training path")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), training.clip_grad, error_if_nonfinite=True))
    apply_optimizer_step(optimizer, model.parameters(), config["step_control"])
    if not torch.stack([p.detach().isfinite().all() for p in model.parameters()]).all():
        raise FloatingPointError("Nonfinite parameters after the update")
    return dict(real_points=count, fresh_prior_points=len(prior), reference_priors=n_reference,
                parent_valid_priors=int(valid.sum()), reconstruction=reconstruction, kl=kl_total, beta=beta,
                projection_ce=repair, projection_weight=weight, prior_occupancy_bce=bce,
                prior_conditional_ce=ce, pre_clip_gradient_norm=norm, projection_counts=stats,
                total=reconstruction + beta * kl_total + weight * repair
                      + guide["occupancy_weight"] * bce + guide["conditional_weight"] * ce
                      + weighted_motif + weighted_roundtrip + extra_value,
                **motif_metrics)


def _backward(loss):
    if not loss.requires_grad or not torch.isfinite(loss):
        raise FloatingPointError("Detached or nonfinite training objective")
    loss.backward()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--resume-receipt", type=Path, help="Previous segment's latest.json. Resume to the epoch boundary.")
    parser.add_argument("--previous-config", type=Path,
                        help="One-time handoff from this exact segment recipe to the new objective.")
    parser.add_argument("--wall-seconds", type=int, help="Allocation budget from worker startup, including checkpoint reserve.")
    parser.add_argument("--extra-rounds", type=int, choices=(0, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.previous_config is not None and args.resume_receipt is None:
        parser.error("--previous-config requires --resume-receipt")
    config = json.loads(args.config.read_bytes())
    motifs = config.get("functional_groups", {})
    if motifs.get("completion_weight", 0) or motifs.get("neighbor_weight", 0) or motifs.get("ring_size"):
        config["quality_panel"] = dict(config["quality_panel"], monitor_motifs=True,
                                      monitor_ring_sizes=bool(motifs.get("ring_size")))
    if config.get("generated_roundtrip"):
        config["quality_panel"] = dict(config["quality_panel"], generated_roundtrip_points=128)
    wall_seconds = config["wall_seconds"] if args.wall_seconds is None else args.wall_seconds
    if wall_seconds <= 120:
        parser.error("--wall-seconds must exceed the 120-second checkpoint reserve")
    config["cuda_requirement"] = config["cuda_by_extra_rounds"][str(args.extra_rounds)]
    parent = json.loads(args.parent_receipt.read_bytes())
    config["checkpoint"] = {"path": parent["checkpoint"], "sha256": parent["sha256"]}
    if not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU for the production segment")
    _check_cuda_requirement(config)
    torch.set_num_threads(config["execution"]["torch_threads"])
    start = time.monotonic()
    corpus = BalancedSparseCorpus(config["data"], expected_sha256=config["data_sha256"])
    data = corpus.split("train")
    verify_canonical_training_cache(data, range(16))
    assets = _load(config, corpus, "cuda")
    model, payload, *_ = assets
    if (payload["config_sha256"] != config["parent_config_sha256"] or len(data) != 60_000_000
            or payload["dataset_sha256"] != config["data_sha256"] or model.config.refinement_rounds != 4
            or model.config.extra_bond_rounds != 0 or model.config.decoder_variant != "triangle_refine"
            or model.config.atom_refinement_dropout != config["atom_refinement_dropout"]
            or payload["continuation"]["real_points"] != len(data)
            or payload["continuation"]["updates"] != config["parent_updates"]):
        raise ValueError("Epoch-47 checkpoint does not match the paired production parent")
    identity = _arm_identity(args.config, parent["sha256"], args.extra_rounds)
    source_identity = identity if args.previous_config is None else _source_identity(
        args.config, args.previous_config, parent["sha256"], args.extra_rounds)
    resume_receipt = resume = None
    max_updates = config["leg_updates"]
    if args.resume_receipt is not None:
        resume_receipt, resume = _read_resume(args.resume_receipt, source_identity, config, model, payload,
                                              args.extra_rounds, len(data))
        max_updates = None
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "fork.json", dict(parent=parent, config_sha256=identity,
               extra_rounds=args.extra_rounds, target_updates=max_updates, resume=resume_receipt,
               resume_config_sha256=source_identity if resume_receipt is not None else None,
               target_epoch=config["parent_epoch"] + config["epochs"], wall_seconds=wall_seconds))
    if marker := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                     nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    trainer = _train
    if "activation_gate" in config.get("functional_groups", {}):
        from latentmol.training.motif_activation import run_activation

        previous = json.loads(args.previous_config.read_bytes())["functional_groups"] if args.previous_config else None
        trainer = partial(run_activation, previous_policy=previous, input_receipt=resume_receipt)
    trainer(config, identity, data, assets, args.output, "cuda", config["batch_size"], resume,
            max_updates, start + wall_seconds, config["microbatch"],
            validation=corpus.split("val"), update_fn=projected_update,
            student_factory=partial(_student, count=args.extra_rounds, seed=config["extension_seed"],
                                    atom_order=config["atom_order"]),
            teacher_factory=lambda *_: nullcontext())


if __name__ == "__main__":
    main()
