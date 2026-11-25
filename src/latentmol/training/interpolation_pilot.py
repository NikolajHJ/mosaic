"""Matched VAE fine-tuning with mapped marginals or verified graph targets."""

import argparse
from contextlib import nullcontext
import gzip
import json
import math
import os
from pathlib import Path
import time

import torch

from latentmol.atom_preserving_chemistry import repair_loss
from latentmol.chemistry import encode_smiles
from latentmol.datasets.data import GraphDataset, collate_graphs, validate_large_cache
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.evaluation.checkpoint_comparison import _checked_decode, graph_record, paired_difference, run_comparison
from latentmol.evaluation.evaluate import load_model
from latentmol.evaluation.fixed_prior import FixedPriorProbe, compare_fixed_prior
from latentmol.evaluation.interpolation import (
    _encode, _stack, evaluate_arm, interpolation_codes, load_pair_graphs,
)
from latentmol.evaluation.molecular_pairs import relabel_categories, select_pairs
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.repair_proposals import ProposalRepairTeacher, RepairProposalConfig
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import HardGraphBatch
from latentmol.sampling import subset_indices
from latentmol.train import state_digest
from latentmol.training.invariant_pilot import _save_checkpoint, initialize_invariant, reference_decoder_kl
from latentmol.training.losses import LossConfig, latent_kl, reconstruction_losses


FRACTIONS = tuple(i / 10 for i in range(1, 10))
GRAPH_VERSION = "verified-graph-interpolation-pilot-v1"
GRAPH_FRACTIONS = (.25, .5, .75)


def mapped_interpolation_loss(decoded, a, mapped_b, orders, spec, loss_config):
    """Score the same soft graph in the two canonical endpoint slot frames.

    Each point uses the lower CE of its two frame assignments. The atom map is
    fixed by the source graphs. Both candidate losses use live decoder logits.
    """
    pairs, nodes = a.atoms.shape
    occupied = a.atoms != 0
    if not torch.equal(occupied, mapped_b.atoms != 0):
        raise ValueError("Mapped interpolation requires equal occupied slots")
    fractions = decoded.atoms.new_tensor(FRACTIONS)
    if decoded.atoms.shape[:2] != (pairs * len(fractions), nodes) or orders.shape != (pairs, nodes):
        raise ValueError("Interpolation logits and atom maps do not match the pair budget")
    left, right = a.to_relaxed(spec), mapped_b.to_relaxed(spec)
    upper = torch.ones((nodes, nodes), device=occupied.device, dtype=torch.bool).triu(1)
    bond_mask = occupied[:, :, None] & occupied[:, None, :] & upper
    atom_weights = torch.where(occupied, 1., loss_config.padding_weight)
    bond_weights = torch.where((a.bonds != 0) | (mapped_b.bonds != 0), 1., loss_config.no_bond_weight)
    targets = []
    for key, row_weight in (
        ("atoms", atom_weights),
        ("charges", occupied),
        ("bonds", bond_mask * bond_weights),
    ):
        x, y = getattr(left, key), getattr(right, key)
        alpha = fractions.reshape(1, -1, *([1] * (x.ndim - 1)))
        target = x[:, None] * (1 - alpha) + y[:, None] * alpha
        # Weight the entire categorical vector so a bond/NONE transition keeps
        # the requested probabilities instead of shifting its crossover point.
        targets.append((target * row_weight[:, None, ..., None]).flatten(0, 1))

    def score(logits):
        per_head = []
        for values, target in zip(logits, targets):
            logp = values.log_softmax(-1)
            mass = target.flatten(1).sum(-1)
            per_head.append(-(target * logp).flatten(1).sum(-1) / mass.clamp_min(1e-12))
        return torch.stack(per_head).mean(0)

    logits = (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits)
    a_frame = score(logits)
    b_frame = score(relabel_categories(*logits, orders.repeat_interleave(len(fractions), dim=0)))
    if not bool(torch.isfinite(a_frame).all() and torch.isfinite(b_frame).all()):
        raise FloatingPointError("Nonfinite mapped interpolation loss")
    use_b = b_frame < a_frame
    return torch.where(use_b, b_frame, a_frame).mean(), use_b.float().mean().detach()


def _take(graph, indices):
    return HardGraphBatch(*(getattr(graph, name)[indices] for name in ("atoms", "charges", "bonds")))


def _gradient_norm(gradients):
    norms = torch.stack([gradient.detach().norm() for gradient in gradients if gradient is not None])
    norm = float(norms.norm())
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite objective gradient before interpolation cap")
    return norm


def _backward_with_interpolation_cap(base, interpolation, parameters, weight, max_ratio):
    """Limit the auxiliary contribution before the shared optimizer gradient clip."""
    if not bool(torch.isfinite(base) & torch.isfinite(interpolation)):
        raise FloatingPointError("Nonfinite base or interpolation objective")
    auxiliary = torch.autograd.grad(interpolation, parameters, allow_unused=True)
    base.backward()
    base_norm = _gradient_norm(param.grad for param in parameters)
    auxiliary_norm = _gradient_norm(auxiliary)
    weight_used = min(weight, max_ratio * base_norm / auxiliary_norm) if auxiliary_norm else weight
    for param, gradient in zip(parameters, auxiliary):
        if gradient is None:
            continue
        if param.grad is None:
            param.grad = gradient * weight_used
        else:
            param.grad.add_(gradient, alpha=weight_used)
    return dict(base_gradient_norm=base_norm, interpolation_gradient_norm=auxiliary_norm,
                interpolation_weight_used=weight_used,
                capped_interpolation_gradient_norm=weight_used * auxiliary_norm)


def _pair_panel(dataset, config, output, progress):
    pairs, selection = select_pairs(dataset.smiles, config, progress=progress)
    write_json(output, dict(selection=selection, pairs=pairs))
    if not pairs:
        raise ValueError(f"No eligible atom-mapped pairs for {output.name}")
    return pairs, load_pair_graphs(dataset, pairs, dataset.spec), selection


def _triplet_panel(dataset, config, output, progress, *, excluded_identities=()):
    from latentmol.datasets.graph_triplets import select_triplets

    triplets, selection = select_triplets(dataset, config, progress=progress, excluded_identities=excluded_identities)
    write_json(output, dict(selection=selection, triplets=triplets))
    if not triplets:
        raise ValueError(f"No verified graph intermediates for {output.name}")

    def load_graph(row, key):
        index, smiles = row[key + "_index"], row[key + "_smiles"]
        if index is None and key != "b":
            encoded = encode_smiles(smiles, dataset.spec)
            if encoded["smiles"] != smiles:
                raise ValueError("Derived triplet graph changed canonical identity")
            return HardGraphBatch(*(torch.as_tensor(encoded[name]).long()[None]
                                    for name in ("atom_types", "charges", "bond_types")))
        if not isinstance(index, int) or not 0 <= index < len(dataset) or dataset.smiles[index] != smiles:
            raise ValueError("Triplet row does not match its cached molecule identity")
        return collate_graphs([dataset[index]])

    graphs = tuple([load_graph(row, key) for row in triplets] for key in "acb")
    return triplets, graphs, selection


@torch.inference_mode()
def _evaluate_triplets(model, triplets, graphs_a, graphs_c, fractions, batch_size, folder):
    """Check actual graph identities and retain invalid or unchanged paths."""
    folder.mkdir()
    z_a, z_c = (_encode(model, graphs, batch_size) for graphs in (graphs_a, graphs_c))
    positions = [0., *fractions, 1.]
    codes = interpolation_codes(z_a, z_c, positions)
    by_fraction = {fraction: [] for fraction in positions}
    with gzip.open(folder / "graphs.jsonl.gz", "xt", encoding="utf-8") as stream:
        for begin in range(0, len(codes), batch_size):
            hard = _checked_decode(model, codes[begin:begin + batch_size]).harden(model.spec)
            for index in range(len(hard.atoms)):
                triplet_index, offset = divmod(begin + index, len(positions))
                triplet, fraction = triplets[triplet_index], positions[offset]
                row = graph_record(hard, index, model.spec, triplet=triplet_index, fraction=fraction)
                a, b, c = (triplet[key + "_smiles"] for key in "abc")
                row.update(exact_intermediate=row["strict_valid"] and row["smiles"] == b,
                           exact_endpoint=row["strict_valid"] and row["smiles"] == {0.: a, 1.: c}.get(fraction),
                           distinct_valid_edit=row["strict_valid"] and row["smiles"] not in (a, c))
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                row.pop("graph")
                by_fraction[fraction].append(row)
    middle = by_fraction[.5]
    endpoints_exact = [a["exact_endpoint"] and c["exact_endpoint"] for a, c in zip(by_fraction[0.], by_fraction[1.])]
    hits_anywhere = [any(by_fraction[t][i]["exact_intermediate"] for t in fractions) for i in range(len(triplets))]
    cohorts = {"all_observed": [], "has_derived_endpoint": []}
    for index, row in enumerate(triplets):
        cohort = "all_observed" if row["a_index"] is not None and row["c_index"] is not None else "has_derived_endpoint"
        cohorts[cohort].append(index)
    result = dict(triplets=len(triplets), both_endpoints_exact=sum(endpoints_exact),
        verified_intermediate_reached=sum(hits_anywhere),
        all_interior_valid=sum(all(by_fraction[t][i]["strict_valid"] for t in fractions) for i in range(len(triplets))),
        points=[dict(fraction=fraction, strict_valid_count=sum(row["strict_valid"] for row in rows),
                     exact_intermediate=sum(row["exact_intermediate"] for row in rows),
                     distinct_valid_edits=sum(row["distinct_valid_edit"] for row in rows))
                for fraction, rows in by_fraction.items()],
        midpoint_by_provenance={name: dict(triplets=len(indices),
            exact_intermediate=sum(middle[i]["exact_intermediate"] for i in indices),
            strict_valid_count=sum(middle[i]["strict_valid"] for i in indices))
            for name, indices in cohorts.items()},
        latent_endpoint_distance_l2=(z_c-z_a).norm(dim=-1).tolist())
    write_json(folder / "summary.json", result)
    return result, by_fraction, [row["exact_intermediate"] for row in middle]


def _train_arm(name, weight, reference, parent, train, rows, graphs, orders,
               schedule, config, loss_config, output, progress, *, chemistry_teacher=None, observe_prior=None):
    output.mkdir()
    device = next(reference.parameters()).device
    # Reset both the posterior-sampling and prior-anchor streams for each arm.
    torch.manual_seed(config["seed"])
    model, initialization = initialize_invariant(reference, pooling="canonical")
    if not initialization["encoder_initially_identical"]:
        raise RuntimeError("Both arms must start with the same epoch weights")
    model.train()
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=config["encoder_learning_rate"]),
        dict(params=model.decoder.parameters(), lr=config["decoder_learning_rate"]),
    ], weight_decay=config["weight_decay"])
    rng = torch.Generator().manual_seed(config["seed"] + 1)
    batch_size = config["batch_size"]
    chemistry_checked = False
    graph_targets = config["version"] == GRAPH_VERSION
    gradient_ratio = config.get("interpolation_max_gradient_ratio")
    parameters = tuple(param for param in model.parameters() if param.requires_grad)
    if observe_prior is not None:
        observe_prior(name, 0, model)
    with (output / "training.jsonl").open("x", encoding="utf-8") as log:
        for step, selected_pairs in enumerate(schedule, start=1):
            started = time.monotonic()
            indices = rows[(step - 1) * batch_size:step * batch_size]
            batch = collate_graphs([train[i] for i in indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch, sample_latent=True, from_canonical_cache=True)
            reconstruction = reconstruction_losses(result.decoded, batch, loss_config)["reconstruction"]
            kl = latent_kl(result.mu, result.logvar)
            codes = torch.randn(config["prior_anchor_count"], model.config.latent, generator=rng).to(device)
            prior = reference_decoder_kl(model, reference, codes)
            total = reconstruction + config["beta"] * kl + config["prior_anchor_weight"] * prior
            record = dict(arm=name, step=step, reconstruction=float(reconstruction.detach()),
                          kl=float(kl.detach()), prior_decoder_kl=float(prior.detach()))
            if weight:
                a, b, target = (_take(graph, selected_pairs) for graph in graphs)
                mu_a, _ = model.encoder(a, from_canonical_cache=True)
                mu_b, _ = model.encoder(b, from_canonical_cache=True)
                mixed = model.decode(interpolation_codes(mu_a, mu_b, (.5,) if graph_targets else FRACTIONS))
                if graph_targets:
                    interpolation = reconstruction_losses(mixed, target, loss_config)["reconstruction"]
                else:
                    interpolation, b_share = mapped_interpolation_loss(
                        mixed, a, target, orders[selected_pairs], model.spec, loss_config)
                    record["b_frame_share"] = float(b_share)
                if step == 1:
                    # Check this new term independently of reconstruction/KL.
                    probe = (model.encoder.mu.weight, model.decoder.atom_head.weight,
                             model.decoder.charge_head.weight, next(model.decoder.bond_head.parameters()))
                    gradients = torch.autograd.grad(interpolation, probe, retain_graph=True)
                    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                        raise FloatingPointError("Nonfinite interpolation-only gradient")
                    record["interpolation_path_gradient_norms"] = dict(zip(
                        ("encoder_mean", "decoder_atom", "decoder_charge", "decoder_bond"),
                        [float(gradient.norm()) for gradient in gradients]))
                record["interpolation"] = float(interpolation.detach())
                if gradient_ratio is None:
                    total = total + weight * interpolation
                if chemistry_teacher is not None:
                    targets = chemistry_teacher.build_targets(
                        mixed, candidate_ids=range(mixed.atoms.shape[0]), update=step)
                    receipt = targets.receipt
                    statuses = receipt["status_counts"]
                    if statuses.get("rejected_or_error", 0):
                        raise RuntimeError("Interpolation repair teacher rejected its own result")
                    chemistry = repair_loss(mixed, targets, model.spec)
                    total = total + config["interpolation_chemistry"]["weight"] * chemistry
                    record.update(chemistry=float(chemistry.detach()),
                                  chemistry_status_counts=statuses,
                                  chemistry_search_checks=receipt["search_checks"])
                    if receipt["accepted_targets"] and not chemistry_checked:
                        gradients = torch.autograd.grad(chemistry, (model.encoder.mu.weight,
                            next(model.decoder.bond_head.parameters())), retain_graph=True)
                        if any(not bool(g.isfinite().all()) for g in gradients):
                            raise FloatingPointError("Nonfinite interpolation chemistry gradient")
                        record["chemistry_path_gradient_norms"] = [float(g.norm()) for g in gradients]
                        chemistry_checked = True
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"Nonfinite objective in {name} update {step}")
            if weight and gradient_ratio is not None:
                cap = _backward_with_interpolation_cap(total, interpolation, parameters, weight, gradient_ratio)
                total = total.detach() + cap["interpolation_weight_used"] * interpolation.detach()
                record.update(cap)
            else:
                total.backward()
            if model.encoder.mu.weight.grad is None or model.decoder.atom_head.weight.grad is None:
                raise RuntimeError("Required encoder/decoder training path is detached")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"], error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(param).all()) for param in model.parameters()):
                raise FloatingPointError(f"Nonfinite parameters in {name} update {step}")
            record.update(total=float(total.detach()), gradient_norm=float(norm), seconds=time.monotonic() - started)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if step == 1 or step % 25 == 0 or step == config["updates"]:
                progress(dict(stage="training", **record))
            if observe_prior is not None:
                observe_prior(name, step, model)
    return _save_checkpoint(output / "final.pt", model, parent=parent, config=config,
                            loss_config=loss_config, phase="joint", step=config["updates"], optimizer=optimizer)


def run_pilot(config, output, *, device="cuda"):
    graph_targets = config.get("version") == GRAPH_VERSION
    fractions = GRAPH_FRACTIONS if graph_targets else FRACTIONS
    if (config.get("version") not in ("atom-mapped-interpolation-pilot-v1", GRAPH_VERSION)
            or config.get("fractions") != list(fractions)):
        raise ValueError("Incompatible interpolation pilot version or evaluation fractions")
    for key in ("updates", "batch_size", "pairs_per_update", "prior_anchor_count"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("Positive pilot budget required: " + key)
    for key in ("encoder_learning_rate", "decoder_learning_rate", "interpolation_weight", "clip_grad"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Positive finite pilot setting required: " + key)
    for key in ("beta", "prior_anchor_weight", "weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError("Nonnegative finite pilot setting required: " + key)
    chemistry = config.get("interpolation_chemistry")
    gradient_ratio = config.get("interpolation_max_gradient_ratio")
    if gradient_ratio is not None and (not graph_targets or chemistry is not None
            or not math.isfinite(gradient_ratio) or not 0 < gradient_ratio <= 1):
        raise ValueError("The verified graph pilot requires an interpolation gradient ratio in (0, 1]")
    early_prior = config.get("early_prior")
    if early_prior is not None:
        steps = early_prior["steps"]
        if (not steps or any(type(step) is not int or not 0 <= step <= config["updates"] for step in steps)
                or len(set(steps)) != len(steps) or 0 not in steps):
            raise ValueError("Early prior evaluation requires unique in-budget steps including zero")
    if chemistry is not None:
        if graph_targets:
            raise ValueError("Keep the verified-graph and detached repair objectives separate")
        if not math.isfinite(chemistry["weight"]) or chemistry["weight"] <= 0:
            raise ValueError("Interpolation chemistry weight must be finite and positive")
        search = ProbabilityRepairConfig(**chemistry["search"])
        proposals = RepairProposalConfig(**chemistry["proposals"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / "config.json", config)

    def progress(value):
        value = dict(value, elapsed_seconds=time.monotonic() - started)
        write_json(output / "progress.json", value)
        print(json.dumps(value, allow_nan=False), flush=True)

    try:
        parent_path, data_path = Path(config["checkpoint"]), Path(config["data"])
        manifest_path = data_path / "manifest.json"
        if file_sha256(parent_path) != config["checkpoint_sha256"] or file_sha256(manifest_path) != config["data_manifest_sha256"]:
            raise ValueError("Parent checkpoint or data manifest hash differs")
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The allocated CUDA device is unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache = validate_large_cache(data_path) if manifest["dataset_format_version"] == 2 else None
        train = GraphDataset(data_path, "train", validated_cache=cache)
        val = GraphDataset(data_path, "val", validated_cache=cache)
        reference, payload = load_model(parent_path, device, train.spec)
        if (payload["epoch"] != config["epoch"] or
                payload["dataset_sha256"] != manifest["dataset_sha256"] or
                manifest["dataset_sha256"] != config["dataset_sha256"]):
            raise ValueError("Parent epoch or dataset identity differs")
        if payload.get("train_config", {}).get("mode") != "vae":
            raise ValueError("The interpolation pilot requires a VAE parent")
        reference.eval().requires_grad_(False)
        reference_digest = state_digest(reference.state_dict())
        loss_config = LossConfig(**payload["loss_config"])
        count = config["updates"] * config["batch_size"]
        if count > len(train):
            raise ValueError("Insufficient distinct training rows for the requested budget")
        rows = subset_indices(len(train), count, config["row_seed"], shuffle_full=True)
        checked = dict(train=verify_canonical_training_cache(train, rows),
                       val=verify_canonical_training_cache(val, list(range(min(16, len(val))))))
        if os.environ.get("DBMOL_LAUNCH_READY"):
            write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ.get("LSB_JOBID")))
        panel_builder, unit = (_triplet_panel, "triplets") if graph_targets else (_pair_panel, "pairs")
        train_pairs, train_graphs, train_selection = panel_builder(
            train, config[f"train_{unit}"], output / f"train-{unit}.json",
            lambda value: progress(dict(value, split="train")))
        val_options = {}
        if graph_targets:
            excluded_identities = set(train.smiles)
            excluded_identities.update(GraphDataset(data_path, "test", validated_cache=cache).smiles)
            val_options["excluded_identities"] = excluded_identities
        val_pairs, val_graphs, val_selection = panel_builder(
            val, config[f"validation_{unit}"], output / f"validation-{unit}.json",
            lambda value: progress(dict(value, split="val")), **val_options)
        if graph_targets:
            val_identities = {row[key + "_smiles"] for row in val_pairs for key in "abc"}
            if excluded_identities & val_identities:
                raise RuntimeError("Validation triplets overlap training or reserved test identities")
            del excluded_identities, val_options
        train_graphs = tuple(_stack(graphs).to(device) for graphs in train_graphs)
        orders = None
        if not graph_targets:
            orders = []
            for pair in train_pairs:
                slots = pair["mapping"]["b_original_atom_for_aligned_slot"]
                orders.append(slots + list(range(len(slots), train.spec.max_nodes)))
            orders = torch.tensor(orders, device=device)
        pair_rng = torch.Generator().manual_seed(config["pair_batch_seed"])
        schedule = torch.randint(len(train_pairs), (config["updates"], config["pairs_per_update"]),
                                 generator=pair_rng).tolist()
        write_json(output / "training-panels.json", dict(rows=rows, pair_batches=schedule, canonical_checks=checked))
        checkpoints = {}
        arms = [("control", 0.), ("interpolation", config["interpolation_weight"])]
        if chemistry is not None:
            arms.append(("interpolation_chemistry", config["interpolation_weight"]))
        prior_observations = {}
        observe_prior = None
        if early_prior is not None:
            prior_probe = FixedPriorProbe(reference.config.latent, **{key: value for key, value in
                                         early_prior.items() if key != "steps"})

            def observe_prior(name, step, model):
                if step not in steps:
                    return
                observation = prior_probe.observe(model)
                arm_results = prior_observations.setdefault(step, {})
                arm_results[name] = observation
                paired = (compare_fixed_prior(arm_results["control"], observation)
                          if name != "control" else None)
                write_json(output / name / f"prior-step-{step:04}.json", dict(observation=observation,
                           paired_against_control=paired))
                progress(dict(stage="early_prior_validity", arm=name, step=step, paired=paired,
                              **{method: observation[method]["validity"] for method in ("argmax", "categorical")}))

        for name, weight in arms:
            teacher = (ProposalRepairTeacher(reference.spec, search, proposals=proposals,
                       workers=chemistry["workers"]) if name == "interpolation_chemistry" else nullcontext(None))
            with teacher as active_teacher:
                checkpoints[name] = _train_arm(name, weight, reference, payload, train, rows, train_graphs,
                    orders, schedule, config, loss_config, output / name, progress,
                    chemistry_teacher=active_teacher, observe_prior=observe_prior)
        if state_digest(reference.state_dict()) != reference_digest or file_sha256(parent_path) != config["checkpoint_sha256"]:
            raise RuntimeError("Frozen epoch checkpoint or reference model changed")
        write_json(output / "training-completed.json", dict(checkpoints=checkpoints, frozen_parent_verified=True))
        del reference, payload, train_graphs

        progress(dict(stage="evaluating_interpolation"))
        candidates = [dict(id="epoch45", path=str(parent_path), epoch=config["epoch"], sha256=config["checkpoint_sha256"])]
        candidates += [dict(id=name, **{key: checkpoint[key] for key in ("path", "epoch", "sha256")})
                       for name, checkpoint in checkpoints.items()]
        paths = output / "interpolation-evaluation"
        paths.mkdir()
        path_results, argmax, intermediate_exact = {}, {}, {}
        graph_a, endpoint, mapped_b = val_graphs
        panel = dict(config["path_evaluation"], fractions=config["fractions"])
        for candidate in candidates:
            model, _ = load_model(Path(candidate["path"]), device, val.spec)
            before = state_digest(model.state_dict())
            name = candidate["id"]
            if graph_targets:
                path_results[name], raw, intermediate_exact[name] = _evaluate_triplets(
                    model, val_pairs, graph_a, endpoint, fractions, panel["batch_size"], paths / name)
                argmax[name] = {fraction: [row["strict_valid"] for row in raw[fraction]] for fraction in fractions}
            else:
                z_a, z_b = (_encode(model, graphs, panel["batch_size"]) for graphs in (graph_a, endpoint))
                path_results[name], raw = evaluate_arm(model, val_pairs, graph_a, endpoint, z_a, z_b, panel,
                                                       paths / name, mapped_b=mapped_b)
                argmax[name] = {fraction: [row["valid"] for row in raw[(fraction, "argmax")]]
                                for fraction in fractions}
            if state_digest(model.state_dict()) != before:
                raise RuntimeError(f"Evaluation changed model {name}")
            del model, raw
        paired = {str(fraction): paired_difference(argmax["control"][fraction], argmax["interpolation"][fraction])
                  for fraction in fractions}
        progress(dict(stage="evaluating_reconstruction_and_prior"))
        comparison = run_comparison(dict(config["comparison"], version="paired-vae-checkpoints-v1",
            data=config["data"], data_manifest_sha256=config["data_manifest_sha256"],
            dataset_sha256=config["dataset_sha256"], split="val", checkpoints=candidates),
            output / "comparison", device=device)
        if file_sha256(parent_path) != config["checkpoint_sha256"]:
            raise RuntimeError("Parent checkpoint changed during evaluation")
        result = dict(status="completed", job_id=os.environ.get("LSB_JOBID"), checkpoints=checkpoints,
            fractions=config["fractions"], path_evaluation=path_results, paired_argmax_validity=paired,
            comparison={name: dict(exact_reconstruction=value["reconstruction"]["exact_reconstruction"],
                argmax_prior_validity=value["prior"]["argmax"]["validity"],
                categorical_prior_validity=value["prior"]["categorical"]["validity"])
                for name, value in comparison["candidates"].items()},
            parent_sha256=config["checkpoint_sha256"], frozen_parent_verified=True,
            elapsed_seconds=time.monotonic() - started)
        if graph_targets:
            result.update(train_triplet_selection=train_selection, validation_triplet_selection=val_selection,
                          paired_exact_midpoint=paired_difference(intermediate_exact["control"],
                                                                 intermediate_exact["interpolation"]))
        else:
            result.update(train_pair_selection=train_selection, validation_pair_selection=val_selection)
        if chemistry is not None:
            result["paired_chemistry_argmax_validity"] = {
                str(fraction): paired_difference(argmax["interpolation"][fraction],
                                                 argmax["interpolation_chemistry"][fraction])
                for fraction in FRACTIONS}
        if early_prior is not None:
            result["early_prior_paired_validity"] = {step: compare_fixed_prior(arms["control"], arms["interpolation"])
                                                     for step, arms in prior_observations.items()}
        write_json(output / "summary.json", result)
        progress(dict(stage="completed"))
        return result
    except BaseException as error:
        write_json(output / "failure.json", dict(status="failed", error=type(error).__name__ + ": " + str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chemistry-config", type=Path,
                        help="Add a matched arm with repair loss at the interpolation points")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("version") == GRAPH_VERSION:
        base = json.loads(args.config.with_name(config["base_config"]).read_text(encoding="utf-8"))
        config = {**base, **config}
    if args.chemistry_config:
        if "interpolation_chemistry" in config:
            raise ValueError("Interpolation chemistry was specified twice")
        config["interpolation_chemistry"] = json.loads(args.chemistry_config.read_text(encoding="utf-8"))
    run_pilot(config, args.output, device=args.device)


if __name__ == "__main__":
    main()
