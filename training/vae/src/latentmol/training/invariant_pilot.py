"""Short, separately saved invariant-encoder/matched-reconstruction pilot.

Compatible epoch-45 weights initialize a new model. An encoder-only coordinate
warm-up precedes joint VAE fine-tuning. A frozen reference-decoder KL anchor
protects prior generation; expensive chemistry search and interpolation losses
are deliberately absent from this architecture pilot. No parent optimizer,
checkpoint, callback or continuation is changed or resumed.

The opt-in v3 protocol instead preserves all encoder tensors behind explicit
lossless hard-input canonicalization. It tests order-robust input handling and
short fine-tuning, not learned neural invariance or differentiable soft encoding.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from latentmol.datasets.data import GraphDataset, collate_graphs
from latentmol.evaluation.atom_order import permute_batch
from latentmol.evaluation.checkpoint_comparison import decode_panel, reconstruct, run_comparison
from latentmol.evaluation.checkpoint_comparison import _validated_config as validate_comparison
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.models import GraphAutoencoder
from latentmol.sampling import subset_indices
from latentmol.train import state_digest
from latentmol.training.losses import LossConfig, latent_kl
from latentmol.training.matching import MatchingConfig, matched_reconstruction_losses


def initialize_invariant(reference, *, aggregation="mean", degree_features=False, sum_scale=3., pooling="invariant"):
    """Initialize either a learned set readout or a lossless hard-input wrapper.

    The first invariant feature block is sum(h)/capacity. Multiplying the mean
    old per-slot weight by capacity preserves the corresponding average-linear
    sum approximation. Mean/count columns start at zero. This is an explicit
    initialization, not a claim that the old encoder function is preserved.
    """
    if reference.config.encoder_pooling != "ordered":
        raise ValueError("Warm start expects an ordered reference encoder")
    if pooling not in ("invariant", "canonical"):
        raise ValueError("Select neural invariant or hard canonical input mode")
    if pooling == "canonical" and (aggregation != reference.config.encoder_aggregation or
            degree_features != reference.config.encoder_degree_features or sum_scale != reference.config.encoder_sum_scale):
        raise ValueError("Canonical warm start must retain the parent message aggregation and structural features")
    config = replace(reference.config, encoder_pooling=pooling, encoder_aggregation=aggregation,
                     encoder_degree_features=degree_features, encoder_sum_scale=sum_scale)
    model = GraphAutoencoder(reference.spec, config).to(next(reference.parameters()).device)
    old, new = reference.state_dict(), model.state_dict()
    copied, initialized = [], []
    for key in new:
        if key in old and old[key].shape == new[key].shape:
            new[key] = old[key].clone(); copied.append(key)
        elif key == "encoder.pool.1.weight":
            value = torch.zeros_like(new[key])
            average = old[key].reshape(value.shape[0], reference.spec.max_nodes, config.hidden).mean(1)
            value[:, :config.hidden] = average*reference.spec.max_nodes
            new[key] = value; initialized.append(key)
        elif key == "encoder.degree_projection.weight":
            # A small explicit structural input, not an index/position feature.
            torch.nn.init.normal_(new[key], std=.02)
            initialized.append(key)
        else:
            raise ValueError("Unexpected incompatible warm-start tensor: "+key)
    model.load_state_dict(new, strict=True)
    if state_digest(model.decoder.state_dict()) != state_digest(reference.decoder.state_dict()):
        raise RuntimeError("Warm start did not preserve the decoder")
    same_encoder = state_digest(model.encoder.state_dict()) == state_digest(reference.encoder.state_dict())
    if pooling == "canonical" and not same_encoder:
        raise RuntimeError("Canonical wrapper must preserve every encoder tensor")
    return model, dict(copied_tensors=len(copied), initialized=initialized,
        removed=[key for key in old if key not in new], decoder_initially_identical=True,
        ordered_parameters=sum(p.numel() for p in reference.parameters()),
        invariant_parameters=sum(p.numel() for p in model.parameters()),
        aggregation=aggregation, degree_features=degree_features, sum_scale=sum_scale,
        encoder_initially_identical=same_encoder, input_mode=pooling,
        optimizer="New optimizer per pilot phase; parent optimizer is not resumed")


def reference_decoder_kl(model, reference, codes):
    """True categorical KLs; occupancy/pair weights are fixed by the teacher.

    A scalar pair weight emphasizes reference bonds without class-weighting a
    KL (which would no longer have a stationary point at equal distributions).
    At identical decoders this anchor has zero gradient up to floating error.
    """
    with torch.no_grad():
        target = reference.decode(codes)
        ref_log = tuple(t.log_softmax(-1) for t in
                        (target.atoms, target.charges, target.symmetric_bond_logits))
        ref_prob = tuple(t.exp() for t in ref_log)
    value = model.decode(codes)
    logp = tuple(t.log_softmax(-1) for t in (value.atoms, value.charges, value.symmetric_bond_logits))
    kl = tuple((p*(a-b)).sum(-1) for p, a, b in zip(ref_prob, ref_log, logp))
    occupancy = 1-ref_prob[0][..., 0]
    pair = occupancy[:, :, None]*occupancy[:, None, :]
    pair = pair*torch.ones_like(pair, dtype=torch.bool).triu(1)
    pair = pair*(.1+.9*(1-ref_prob[2][..., 0]))
    atom = kl[0].mean()
    charge = (kl[1]*occupancy).sum()/occupancy.sum().clamp_min(1e-12)
    bond = (kl[2]*pair).sum()/pair.sum().clamp_min(1e-12)
    return (atom+charge+bond)/3


def _save_checkpoint(path, model, *, parent, config, loss_config, phase, step, optimizer):
    if path.exists():
        raise FileExistsError("Preserve earlier pilot checkpoints")
    payload = dict(format_version=1, epoch=0, graph_spec=model.spec.to_dict(),
        model_config=asdict(model.config), model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        dataset_sha256=parent["dataset_sha256"], loss_config=asdict(loss_config),
        train_config=dict(mode="vae", beta=config["beta"], pilot_phase=phase),
        beta_at_checkpoint=config["beta"] if phase == "joint" else 0.,
        parent_checkpoint_sha256=config["checkpoint_sha256"], parent_epoch=config["epoch"],
        pilot=dict(version=config["version"], phase=phase, phase_step=step, config=config),
        optimizer_state=optimizer.state_dict(), torch_rng_state=torch.get_rng_state())
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return dict(path=str(path), sha256=file_sha256(path), phase=phase, steps=step, epoch=0)


@torch.no_grad()
def teacher_codes(reference, data, indices, batch_size):
    """Frozen canonical targets; training calibration never reads validation data."""
    device = next(reference.parameters()).device
    means, logvars = [], []
    for start in range(0, len(indices), batch_size):
        hard = collate_graphs([data[i] for i in indices[start:start+batch_size]]).to(device)
        mu, logvar = reference.encode(hard.to_relaxed(reference.spec))
        if not bool(torch.isfinite(mu).all() and torch.isfinite(logvar).all()):
            raise FloatingPointError("Nonfinite frozen teacher coordinates")
        means.append(mu.cpu()); logvars.append(logvar.cpu())
    return torch.cat(means), torch.cat(logvars)


def latent_summary(means, target=None, scale=1.):
    """Detect constant/bias codes which active-dimension counts alone can miss."""
    means = means.detach().cpu().double()
    centered = means-means.mean(0)
    variance = centered.square().mean()
    eigenvalues = torch.linalg.svdvals(centered).square()
    mass = eigenvalues.sum()
    p = eigenvalues/mass.clamp_min(1e-30)
    rank = float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum())) if mass > 1e-20 else 0.
    result = dict(mean_norm=float(means.norm(dim=-1).mean()), mean_vector_norm=float(means.mean(0).norm()),
                  between_molecule_variance=float(variance), effective_rank=rank)
    if target is not None:
        target = target.detach().cpu().double()
        mse = (means-target).square().mean()
        baseline = (target-target.mean(0)).square().mean()
        result.update(teacher_mse=float(mse), normalized_teacher_mse=float(mse/scale),
            teacher_constant_predictor_mse=float(baseline),
            variance_ratio=float(variance/baseline.clamp_min(1e-30)))
    return result


@torch.no_grad()
def check_canonical_start(model, reference, data, indices, config):
    """A small lossless-input/parent-function check, not a chemistry quality gate."""
    from latentmol.datasets.ordering import canonical_encoder_input
    device = next(model.parameters()).device
    selected = indices[:min(64, len(indices))]
    target = collate_graphs([data[i] for i in selected]).to(device)
    rng = torch.Generator().manual_seed(config["seed"]+101)
    shuffled = permute_batch(target, torch.stack([torch.randperm(model.spec.max_nodes, generator=rng)
                                                 for _ in selected]))
    expected = target.to_relaxed(model.spec)
    restored = canonical_encoder_input(shuffled.to_relaxed(model.spec), model.spec)
    if any(not torch.equal(getattr(expected, k), getattr(restored, k)) for k in ("atoms", "charges", "bonds")):
        raise RuntimeError("Canonical input mapping changed the cached graph labels")
    student_mu, student_logvar = model.encode(shuffled.to_relaxed(model.spec))
    teacher_mu, teacher_logvar = reference.encode(expected)
    torch.testing.assert_close(student_mu, teacher_mu, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(student_logvar, teacher_logvar, rtol=1e-5, atol=1e-6)
    return dict(count=len(selected), full_graph_labels_identical=True,
        max_mu_abs_error=float((student_mu-teacher_mu).abs().max()),
        max_logvar_abs_error=float((student_logvar-teacher_logvar).abs().max()),
        scope="Supported hard training graphs only; discrete preprocessing, no claim for soft or invalid graphs")


@torch.inference_mode()
def _health(model, data, indices, output, config, loss_config, *, teacher_means=None, coordinate_scale=1.):
    was_training = model.training
    model.eval(); output.mkdir()
    try:
        limit = min(config["health_reconstruction_count"], len(indices))
        metrics, _, mu, _ = reconstruct(model, data, indices[:limit], output/"reconstruction.jsonl.gz",
            batch_size=config["batch_size"], loss_config=loss_config, anchor_count=limit)
        codes = torch.randn(config["health_prior_count"], model.config.latent,
                            generator=torch.Generator().manual_seed(config["health_seed"]))
        prior, _ = decode_panel(model, codes, output/"prior.jsonl.gz", batch_size=config["batch_size"])
        value = dict(reconstruction=metrics, prior=prior,
            latent=latent_summary(mu, teacher_means, coordinate_scale),
            scope="Fixed small monitoring panel; quality changes are observations, not pilot stop conditions")
        write_json(output/"summary.json", value)
        return dict(exact=metrics["exact_reconstruction"], reconstruction_validity=metrics["validity"], latent=value["latent"],
                    random_validity=prior["argmax"]["validity"], unique_valid_yield=prior["argmax"]["unique_valid_yield"])
    finally:
        model.train(was_training)


def _validate(config):
    if config.get("version") not in ("invariant-matched-vae-pilot-v1", "invariant-matched-vae-pilot-v2", "invariant-matched-vae-pilot-v3"):
        raise ValueError("Unknown invariant pilot protocol")
    if config.get("version") in ("invariant-matched-vae-pilot-v2", "invariant-matched-vae-pilot-v3"):
        if config["version"].endswith("v2") and config.get("warmup_objective") != "teacher_latent":
            raise ValueError("Recovery protocol requires teacher-only warm-up")
        if config["version"].endswith("v3") and (config.get("warmup_objective") != "reconstruction_and_latent" or
                config.get("invariant_encoder", {}).get("pooling") != "canonical"):
            raise ValueError("Canonical pilot requires the explicit hard-input wrapper and reconstruction warm-up")
        if type(config.get("teacher_calibration_count")) is not int or not 2 <= config["teacher_calibration_count"] <= config["train_count"]:
            raise ValueError("Teacher scale requires a bounded training-only panel")
        for key in ("joint_mu_weight", "joint_logvar_weight"):
            if not 0 < config[key] < float("inf"):
                raise ValueError("Recovery joint phase must retain finite positive latent alignment")
    for key in ("train_count", "batch_size", "warmup_updates", "joint_updates", "prior_anchor_count",
                "checkpoint_every", "health_every", "health_prior_count", "health_reconstruction_count"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Positive pilot budget required: "+key)
    for key in ("encoder_learning_rate", "decoder_learning_rate", "warmup_learning_rate", "clip_grad"):
        if not 0 < config[key] < float("inf"):
            raise ValueError("Finite positive optimization setting required: "+key)
    for key in ("beta", "prior_anchor_weight", "warmup_mu_weight", "warmup_logvar_weight", "weight_decay"):
        if not 0 <= config[key] < float("inf"):
            raise ValueError("Finite nonnegative loss setting required: "+key)
    evaluation = dict(config["evaluation"], version="paired-vae-checkpoints-v1", split="val",
                      checkpoints=[dict(id="parent")])
    validate_comparison(evaluation)
    order = evaluation.get("atom_order", {})
    if (type(order.get("count")) is not int or not 1 <= order["count"] <= evaluation["reconstruction_count"] or
            type(order.get("permutations")) is not int or order["permutations"] < 1):
        raise ValueError("A bounded held-out atom-order panel is required")


def run_pilot(config, output, *, device="cuda"):
    _validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output/"config.json", config)
    try:
        parent_path, data_path = Path(config["checkpoint"]), Path(config["data"])
        if file_sha256(parent_path) != config["checkpoint_sha256"]:
            raise ValueError("Parent epoch checkpoint changed")
        if file_sha256(data_path/"manifest.json") != config["data_manifest_sha256"]:
            raise ValueError("Dataset manifest changed")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Allocated CUDA device is unavailable")
        torch.manual_seed(config["seed"])
        train = GraphDataset(data_path, "train")
        validation = GraphDataset(data_path, "val")
        reference, parent = load_model(parent_path, device, train.spec)
        if parent.get("epoch") != config["epoch"] or parent["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("Parent epoch/dataset metadata differs")
        if train.manifest["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("Training dataset identity differs")
        reference.requires_grad_(False); reference.eval()
        reference_before = state_digest(reference.state_dict())
        model, initialization = initialize_invariant(reference, **config.get("invariant_encoder", {}))
        write_json(output/"initialization.json", initialization)
        loss_config = LossConfig(**parent["loss_config"])
        matching = MatchingConfig(**config["matching"])
        if config["train_count"] > len(train):
            raise ValueError("Declared training subset exceeds available training split")
        train_indices = subset_indices(len(train), config["train_count"], config["train_seed"])
        val_indices = subset_indices(len(validation), config["health_reconstruction_count"], config["health_seed"])
        coordinate_scale, health_teacher = 1., None
        if config.get("warmup_objective") == "teacher_latent" or config["version"].endswith("v3"):
            calibration_indices = train_indices[:config["teacher_calibration_count"]]
            calibration_mu, _ = teacher_codes(reference, train, calibration_indices, config["batch_size"])
            coordinate_scale = float(calibration_mu.var(0, unbiased=False).mean().clamp_min(1e-6))
            health_teacher, _ = teacher_codes(reference, validation, val_indices, config["batch_size"])
            write_json(output/"teacher-calibration.json", dict(split="train", indices=calibration_indices,
                coordinate_mse_scale=coordinate_scale, teacher=latent_summary(calibration_mu),
                objective="Mean squared teacher-mean error / training teacher variance",
                teacher_only_warmup=config.get("warmup_objective") == "teacher_latent"))
        if model.config.encoder_pooling == "canonical":
            write_json(output/"canonical-start.json", check_canonical_start(model, reference, train, train_indices, config))
        write_json(output/"training-panel.json", dict(split="train", indices=train_indices,
            validation_monitor_indices=val_indices, final_test_split_used=False))
        loader = DataLoader(Subset(train, train_indices), batch_size=config["batch_size"], shuffle=True,
            num_workers=0, collate_fn=collate_graphs, generator=torch.Generator().manual_seed(config["train_seed"]+1))
        def batches():
            while True:
                yield from loader
        stream = iter(batches())
        permutation_rng = torch.Generator().manual_seed(config["seed"]+1)
        prior_rng = torch.Generator().manual_seed(config["seed"]+2)
        if os.environ.get("DBMOL_LAUNCH_READY"):
            write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ.get("LSB_JOBID")))
        initial = _health(model, validation, val_indices, output/"initial-health", config, loss_config,
                          teacher_means=health_teacher, coordinate_scale=coordinate_scale)
        checkpoints, phases = {}, []
        with (output/"training.jsonl").open("x", encoding="utf-8") as log:
            for phase, budget in (("warmup", config["warmup_updates"]), ("joint", config["joint_updates"])):
                model.encoder.requires_grad_(True)
                model.decoder.requires_grad_(phase == "joint")
                model.train()
                if phase == "warmup":
                    model.decoder.eval()
                groups = [dict(params=list(model.encoder.parameters()),
                    lr=config["warmup_learning_rate"] if phase == "warmup" else config["encoder_learning_rate"])]
                if phase == "joint":
                    groups.append(dict(params=list(model.decoder.parameters()), lr=config["decoder_learning_rate"]))
                optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
                phase_started = time.monotonic()
                decoder_before = state_digest(model.decoder.state_dict())
                matching_seconds, training_seconds = 0., 0.
                for step in range(1, budget+1):
                    begin = time.monotonic()
                    target = next(stream).to(device)
                    orders = torch.stack([torch.randperm(model.spec.max_nodes, generator=permutation_rng)
                                          for _ in range(len(target.atoms))])
                    shuffled = permute_batch(target, orders)
                    optimizer.zero_grad(set_to_none=True)
                    teacher_only = phase == "warmup" and config.get("warmup_objective") == "teacher_latent"
                    if teacher_only:
                        # Do not let a poorly aligned student exploit the frozen
                        # decoder's matching loss before it learns molecular codes.
                        mu, logvar = model.encode(shuffled.to_relaxed(model.spec))
                        losses, matched = {}, dict(seconds=0., used=False)
                    else:
                        result = model(shuffled.to_relaxed(model.spec), sample_latent=phase == "joint")
                        mu, logvar = result.mu, result.logvar
                        losses, matched = matched_reconstruction_losses(result.decoded, target, loss_config, matching)
                    if phase == "warmup" or config.get("joint_mu_weight", 0) > 0:
                        with torch.no_grad():
                            ref_mu, ref_logvar = reference.encode(target.to_relaxed(model.spec))
                        coordinate = (mu-ref_mu).square().mean()/coordinate_scale
                        variance = (logvar-ref_logvar).square().mean()
                    else:
                        coordinate, variance = mu.sum()*0., logvar.sum()*0.
                    if phase == "warmup":
                        kl = coordinate*0.
                        anchor = coordinate*0.
                        loss = (0. if teacher_only else losses["reconstruction"]) + config["warmup_mu_weight"]*coordinate + config["warmup_logvar_weight"]*variance
                    else:
                        kl = latent_kl(mu, logvar)
                        codes = torch.randn(config["prior_anchor_count"], model.config.latent,
                                            generator=prior_rng).to(device)
                        anchor = reference_decoder_kl(model, reference, codes)
                        loss = (losses["reconstruction"] + config["beta"]*kl + config["prior_anchor_weight"]*anchor
                            + config.get("joint_mu_weight", 0)*coordinate + config.get("joint_logvar_weight", 0)*variance)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite pilot objective")
                    loss.backward()
                    if (model.encoder.mu.weight.grad is None or model.encoder.logvar.weight.grad is None or
                            model.encoder.atom_projection.weight.grad is None):
                        raise RuntimeError("Required encoder path is detached")
                    if phase == "joint" and (model.decoder.atom_head.weight.grad is None or
                            model.decoder.charge_head.weight.grad is None or
                            not any(p.grad is not None for p in model.decoder.bond_head.parameters())):
                        raise RuntimeError("Required atom/charge/bond decoder path is detached")
                    active_parameters = [p for p in model.parameters() if p.requires_grad]
                    norm = torch.nn.utils.clip_grad_norm_(active_parameters, config["clip_grad"], error_if_nonfinite=True)
                    optimizer.step()
                    if any(not bool(torch.isfinite(p).all()) for p in active_parameters):
                        raise FloatingPointError("Nonfinite pilot parameter update")
                    seconds = time.monotonic()-begin
                    training_seconds += seconds; matching_seconds += matched["seconds"]
                    if step <= 5 or step % config.get("log_every", 20) == 0:
                        record = dict(phase=phase, step=step, loss=float(loss.detach()),
                            reconstruction=float(losses["reconstruction"].detach()) if losses else None, kl=float(kl.detach()),
                            coordinate_mse=float(coordinate.detach())*coordinate_scale, logvar_mse=float(variance.detach()),
                            normalized_coordinate_mse=float(coordinate.detach()),
                            coordinate_mse_scale=coordinate_scale, teacher_only_warmup=teacher_only,
                            prior_decoder_kl=float(anchor.detach()), gradient_norm_before_clip=float(norm),
                            gradient_clipped=bool(norm > config["clip_grad"]), seconds=seconds, matching=matched)
                        log.write(json.dumps(record, allow_nan=False)+"\n"); log.flush()
                        print(json.dumps(dict(event="training", phase=phase, step=step,
                            loss=record["loss"], seconds=seconds, matching_seconds=matched["seconds"])), flush=True)
                    if step % config["health_every"] == 0 or step == budget:
                        health = _health(model, validation, val_indices,
                            output/f"health-{phase}-{step:05d}", config, loss_config,
                            teacher_means=health_teacher, coordinate_scale=coordinate_scale)
                        if phase == "warmup":
                            model.decoder.eval()
                        log.write(json.dumps(dict(event="health", phase=phase, step=step, **health))+"\n"); log.flush()
                        write_json(output/"progress.json", dict(stage=phase, step=step, budget=budget, health=health,
                            elapsed_seconds=time.monotonic()-started, mean_training_update_seconds=training_seconds/step))
                    if step % config["checkpoint_every"] == 0 or step == budget:
                        saved = _save_checkpoint(output/f"{phase}-{step:05d}.pt", model, parent=parent, config=config,
                            loss_config=loss_config, phase=phase, step=step, optimizer=optimizer)
                        if step == budget:
                            checkpoints[phase] = saved
                if phase == "warmup" and state_digest(model.decoder.state_dict()) != decoder_before:
                    raise RuntimeError("Frozen warm-up decoder changed")
                phases.append(dict(phase=phase, updates=budget, training_update_seconds=training_seconds,
                    matching_seconds=matching_seconds, elapsed_seconds=time.monotonic()-phase_started))
        if state_digest(reference.state_dict()) != reference_before or file_sha256(parent_path) != config["checkpoint_sha256"]:
            raise RuntimeError("Frozen epoch-45 reference changed")
        write_json(output/"training-completed.json", dict(status="completed", checkpoints=checkpoints, phases=phases,
            frozen_reference_unchanged=True, initial_health=initial))
        del model, reference, optimizer, parent, train, validation, groups, active_parameters
        del result, loss, losses, anchor, coordinate, variance, target, shuffled, stream, loader
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        evaluation = dict(config["evaluation"], version="paired-vae-checkpoints-v1",
            data=config["data"], data_manifest_sha256=config["data_manifest_sha256"],
            dataset_sha256=config["dataset_sha256"], split="val", checkpoints=[
                dict(id="epoch45", epoch=config["epoch"], path=config["checkpoint"], sha256=config["checkpoint_sha256"]),
                dict(id="invariant-warmup", epoch=0, path=checkpoints["warmup"]["path"], sha256=checkpoints["warmup"]["sha256"]),
                dict(id="invariant-pilot", epoch=0, path=checkpoints["joint"]["path"], sha256=checkpoints["joint"]["sha256"])])
        write_json(output/"evaluation-config.json", evaluation)
        comparison = run_comparison(evaluation, output/"comparison", device=device)
        if file_sha256(parent_path) != config["checkpoint_sha256"]:
            raise RuntimeError("Parent checkpoint changed during evaluation")
        canonical_mode = initialization["input_mode"] == "canonical"
        result = dict(version=config["version"], status="completed", initialization=initialization,
            checkpoints=checkpoints, phases=phases, frozen_parent_unchanged=True, interpolation_loss_used=False,
            chemistry_search_used=False, parent_continuation_resumed=False,
            encoder_input_mode=initialization["input_mode"],
            evaluation=str(output/"comparison/summary.json"), elapsed_seconds=time.monotonic()-started,
            job_id=os.environ.get("LSB_JOBID"), candidates={name:dict(
                reconstruction=v["reconstruction"]["exact_reconstruction"], random_validity=v["prior"]["argmax"]["validity"],
                unique_valid_yield=v["prior"]["argmax"]["unique_valid_yield"], atom_order=v.get("atom_order"))
                for name, v in comparison["candidates"].items()},
            scope=("Hard-input canonicalization and short fine-tuning; not a learned neural invariant encoder or soft-input support"
                if canonical_mode else "Short new-architecture pilot, not a matched full-training comparison or promotion of a new decoder"))
        write_json(output/"summary.json", result)
        lines = ["# Invariant encoder pilot", "", "Epoch 45 remains unchanged. No interpolation or chemical repair loss.", "",
            "| Checkpoint | Exact reconstruction | Random validity | Reordered exact reconstruction | Posterior within tolerance | Mean latent shift |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
        if canonical_mode:
            lines[3:3] = ["Input mode: lossless discrete canonical ordering before the retained ordered encoder.",
                "Supported hard graphs only; not a fully neural invariant or relaxed-input encoder.", ""]
        for name, value in result["candidates"].items():
            order = value["atom_order"]
            lines.append(f'| {name} | {100*value["reconstruction"]:.2f}% | {100*value["random_validity"]:.2f}% | '
                f'{100*order["permuted_exact_reconstruction"]:.2f}% | '
                f'{100*order["posterior_within_tolerance_fraction"]:.2f}% | {order["mean_mu_l2_shift"]:.5g} |')
        lines += ["", "Warm-up freezes the decoder, so its prior generation is preserved by construction.",
            "The new invariant readout is only briefly trained; this pilot does not establish its final achievable reconstruction.",
            "Robustness alone can be perfect for a poor encoder: assess it alongside reconstruction and diversity.", ""]
        (output/"summary.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(output/"progress.json", dict(stage="completed", elapsed_seconds=result["elapsed_seconds"]))
        return result
    except BaseException as error:
        write_json(output/"failure.json", dict(status="failed", error=type(error).__name__+": "+str(error),
            elapsed_seconds=time.monotonic()-started, parent_continuation_resumed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_pilot(json.loads(args.config.read_text(encoding="utf-8")), args.output, device=args.device)
    print(json.dumps(dict(status=result["status"], output=str(args.output))))


if __name__ == "__main__":
    main()
