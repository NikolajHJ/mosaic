"""Matched compact-decoder pilots, including new arms against saved controls."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.evaluation.bond_message_ablation import bond_message_ablation
from latentmol.evaluation.compact_training_panel import CompactTrainingPanel
from latentmol.evaluation.decoder_topology import reconstruction_topology
from latentmol.experiments import file_sha256, write_json
from latentmol.models import GraphAutoencoder
from latentmol.train import kl_weight, seed_everything, state_digest
from latentmol.training.combined_continuation import _check_cuda_requirement
from latentmol.training.compact_benchmark import _batch, _optimizer, _sync, _update, prior_draw_count
from latentmol.training.compact_prior import CompactOccupancyGuidance
from latentmol.training.compact_train import read_settings


VARIANTS = ("triangle_refine", "bond_message")
HYBRID = "triangle_bond_message"
ALL_VARIANTS = (*VARIANTS, HYBRID)
WIDE_HYBRID = "triangle_bond_message_wide"  # Study arm; checkpoints use the ordinary hybrid decoder.


def _variants(config):
    variants = tuple(config["variants"])
    if variants == (HYBRID,) and set(config.get("controls", {})) == {"path", "sha256"}:
        return variants
    if variants in (VARIANTS, ALL_VARIANTS, (WIDE_HYBRID,)) and "controls" not in config:
        return variants
    raise ValueError("Unsupported set of decoder comparison arms or controls")


def read_comparison(path):
    config = json.loads(path.read_bytes())
    if "matched_recipe" in config:
        # Inherit the independent long study's actual data order, noise streams,
        # evaluation panel and budgets without running any of its three arms.
        reference = path.parent / config["matched_recipe"]
        parent = json.loads(reference.read_bytes())
        fields = {"variants", "matched_recipe", "decoder_widths",
                  "hpc_cuda_ownership_helper", "hpc_cuda_ownership_sha256"}
        if (set(config) != fields or _variants(config) != (WIDE_HYBRID,)
                or _variants(parent) != ALL_VARIANTS or reference.parent != path.parent
                or any(config[k] != parent[k] for k in fields if k.startswith("hpc_"))):
            raise ValueError("The wider hybrid must use the same long comparison and CUDA setup")
        config = {**parent, **config, "matched_recipe_sha256": file_sha256(reference)}
    base = (path.parent / config["base_training_run"]).resolve()
    settings, training = read_settings(base)
    variants = _variants(config)
    if (config["version"] != "compact-decoder-comparison-v1" or training.model.decoder_variant != "triangle_refine"
            or not settings.get("prior_occupancy")):
        raise ValueError("Expected a matched decoder comparison against the count-corrected triangle VAE")
    if variants == (WIDE_HYBRID,):
        widths = config.get("decoder_widths", {})
        if ("matched_recipe_sha256" not in config or set(widths) != {"pair_hidden", "triangle_hidden"}
                or any(type(w) is not int or w <= getattr(training.model, k) for k, w in widths.items())):
            raise ValueError("Specify wider pair and triangle channels for the matched hybrid")
    if variants in (ALL_VARIANTS, (WIDE_HYBRID,)):
        # Keep the original experiment's loss and batch settings; give the larger
        # study an independent initialization, noise stream, and held-out panel.
        train_seed, prior_seed, panel = config["training_seed"], config["prior_seed"], config["panel"]
        if type(train_seed) is not int or not 0 <= train_seed < 2**32:
            raise ValueError("Training initialization needs a valid NumPy seed")
        if (type(prior_seed) is not int or not 0 <= prior_seed < 2**63
                or prior_seed in (train_seed + 1, panel["seed"] + 1)
                or train_seed + 1 == panel["seed"] + 1):
            raise ValueError("Training posterior, training prior and panel prior need independent seeds")
        training = replace(training, seed=train_seed)
        settings = dict(settings, panel=panel, prior_occupancy=dict(settings["prior_occupancy"], seed=prior_seed))
    return config, settings, training


def shared_parameters(model):
    # Pair updates have different meanings in the two decoder variants. Everywhere
    # else the encoder, latent-to-slot map, categorical heads and feedback are shared.
    return {k: val.detach().cpu().clone() for k, val in model.state_dict().items()
            if not k.startswith("decoder.rounds.")}


def _saved_controls(config, report):
    reference = config["controls"]
    raw = Path(reference["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Saved comparison digest differs from this experiment's controls")
    parent = json.loads(raw)
    if (parent["status"] != "completed" or set(parent["arms"]) != set(VARIANTS)
            or any(arm["status"] != "completed" for arm in parent["arms"].values())):
        raise ValueError("Expected both completed matched control arms")
    for key in ("dataset_sha256", "training_seed", "row_order_sha256", "available_training_rows", "updates_per_arm",
                "batch_size", "microbatch", "panel", "base_run"):
        if report[key] != parent[key]:
            raise ValueError(f"Saved controls differ on {key}")
    if config["observe_updates"] != parent["config"]["observe_updates"]:
        raise ValueError("Saved controls have a different evaluation schedule")
    return parent["arms"]["bond_message"]


@torch.no_grad()
def _initialize_hybrid(model, training, probes):
    """Rebuild the historical initial message-only model without optimizer updates.

    The original comparison copied the triangle model's initial tensors outside
    the rounds into the message model. Reproduce that overwrite before copying
    all common tensors, including the message rounds, to the hybrid.
    """
    seed_everything(training.seed)
    shared = shared_parameters(GraphAutoencoder(model.spec, replace(training.model, decoder_variant="triangle_refine")))
    seed_everything(training.seed)
    control = GraphAutoencoder(model.spec, replace(training.model, decoder_variant="bond_message"))
    control.load_state_dict({**control.state_dict(), **shared}, strict=True)
    transfer = control.state_dict()
    missing = set(model.state_dict()) - set(transfer)
    if not missing or any(".triangle." not in name for name in missing):
        raise ValueError("Hybrid introduced unexpected parameters outside the triangle branches")
    model.load_state_dict({**model.state_dict(), **transfer}, strict=True)
    # A zero output projection on each added triangle makes the new branch
    # an identity at initialization. Check this on actual fixed panel codes.
    control.eval()
    model.eval()
    try:
        a, b = control.decode(probes), model.decode(probes)
        for field in ("atoms", "charges", "bonds"):
            if not torch.equal(getattr(a, field), getattr(b, field)):
                raise RuntimeError(f"Initial hybrid and bond-message control differ on {field}")
    finally:
        model.train()
    return dict(initial_bond_message_state_sha256=state_digest(transfer),
                identical_initial_logit_probe_count=len(probes))


@torch.no_grad()
def _initialize_wide_hybrid(model, training, probes):
    """Reuse every shape-compatible time-zero tensor from the narrower hybrid."""
    seed_everything(training.seed)
    reference = GraphAutoencoder(model.spec, replace(training.model, decoder_variant=HYBRID))
    _initialize_hybrid(reference, training, probes)
    source, target = reference.state_dict(), model.state_dict()
    if source.keys() != target.keys():
        raise ValueError("The capacity comparison requires the same hybrid layers")
    shared = {name: value for name, value in source.items() if target[name].shape == value.shape}
    if len(shared) == len(source) or any(name.startswith("encoder.") for name in source.keys() - shared.keys()):
        raise ValueError("The capacity comparison must widen only the decoder")
    model.load_state_dict({**target, **shared}, strict=True)
    return dict(reference_initial_state_sha256=state_digest(source), copied_state_tensors=len(shared),
                resized_state_tensors=sorted(source.keys() - shared.keys()))


def _save_endpoint(path, model, training, dataset_sha256, updates):
    payload = dict(format_version=1, graph_spec=model.spec.to_dict(), model_config=asdict(model.config),
                   model_state=model.state_dict(), train_config=asdict(training), loss_config=asdict(training.loss),
                   run_id=path.parent.name, epoch=0, updates=updates, real_points=training.batch_size * updates,
                   dataset_sha256=dataset_sha256, comparison_endpoint=path.name == "final.pt")
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
    temporary.replace(path)


def train_arm(config, settings, training, data, panel, rows, model, output, device):
    occupancy = settings["prior_occupancy"]
    batch_size = training.batch_size
    ratio = occupancy["real_per_prior"]
    optimizer = _optimizer(model, training)
    guide = CompactOccupancyGuidance(occupancy, data.manifest["active_atom_counts"]["train"],
                                     data.spec.max_nodes, device)
    epsilon_rng = torch.Generator().manual_seed(training.seed + 1)
    prior_rng = torch.Generator().manual_seed(occupancy["seed"])
    epsilon_hash, prior_hash = hashlib.sha256(), hashlib.sha256()
    long_comparison = tuple(config["variants"]) in (ALL_VARIANTS, (WIDE_HYBRID,))
    def observe():
        metrics = panel.observe(model)
        if long_comparison:
            metrics["topology"] = reconstruction_topology(model, panel.batches)
        return metrics

    observations = {0: observe()}
    write_json(output / "quality.json", observations)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_seconds, total_priors, peak_bytes = 0., 0, 0
    saved_checkpoints = {}
    started = time.perf_counter()
    with (output / "training.jsonl").open("x") as trace:
        for step in range(config["updates"]):
            _sync(device)
            start = time.perf_counter()
            offset = step * batch_size
            real = _batch(data, rows[offset:offset + batch_size])
            epsilon = torch.randn(batch_size, training.model.latent, generator=epsilon_rng)
            count = prior_draw_count(offset, batch_size, ratio)
            prior = torch.randn(count, training.model.latent, generator=prior_rng)
            guide.update = step
            metrics = _update(model, optimizer, training, real, epsilon=epsilon, prior=prior,
                              beta=kl_weight(training, 1, offset + batch_size), microbatch=config["microbatch"],
                              guidance=guide)
            _sync(device)
            seconds = time.perf_counter() - start
            training_seconds += seconds
            if device.type == "cuda":
                peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
            total_priors += count
            epsilon_hash.update(epsilon.numpy().tobytes())
            prior_hash.update(prior.numpy().tobytes())
            trace.write(json.dumps(dict(update=step + 1, real_points=offset + batch_size,
                                        training_seconds=seconds, **metrics), allow_nan=False) + "\n")
            if step == 0 or (step + 1) % 16 == 0 or step + 1 in config["observe_updates"]:
                trace.flush()
            if step + 1 in config.get("save_updates", ()):
                dest = output / f"update-{step + 1:06d}.pt"
                _save_endpoint(dest, model, training, data.dataset_sha256, step + 1)
                saved_checkpoints[step + 1] = dict(checkpoint=output.name + "/" + dest.name,
                                                   checkpoint_sha256=file_sha256(dest))
            if step + 1 in config["observe_updates"]:
                observations[step + 1] = observe()
                write_json(output / "quality.json", observations)
                if device.type == "cuda":
                    # Exclude held-out observation allocations from the hot-path measurement.
                    torch.cuda.reset_peak_memory_stats(device)
    endpoint = observations[config["updates"]]
    topology = endpoint["topology"] if long_comparison else reconstruction_topology(model, panel.batches)
    message_effect = ({"bond_message_ablation": bond_message_ablation(model, panel.batches)}
                      if model.config.decoder_variant in ("bond_message", HYBRID) else {})
    dest = output / "final.pt"
    _save_endpoint(dest, model, training, data.dataset_sha256, config["updates"])
    if long_comparison:
        from latentmol.evaluation.decoder_bond_ranking import decoder_bond_ranking
        message_effect.update(bond_ranking=decoder_bond_ranking(model, panel.batches, panel.loss),
                              intermediate_checkpoints=saved_checkpoints)
    return dict(status="completed", real_points=len(rows), prior_draws=total_priors,
                posterior_noise_sha256=epsilon_hash.hexdigest(), prior_noise_sha256=prior_hash.hexdigest(),
                endpoint=endpoint, topology=topology, **message_effect,
                training_seconds=training_seconds, wall_seconds_after_baseline=time.perf_counter() - started,
                real_molecules_per_second=len(rows) / training_seconds,
                peak_training_allocated_bytes=peak_bytes if device.type == "cuda" else None,
                checkpoint=output.name + "/final.pt", checkpoint_sha256=file_sha256(dest))


def run_comparison(config, settings, training, data, panel, output, device):
    """Create a fresh experiment directory and keep the first arm if a later arm fails."""
    device = torch.device(device)
    updates = config["updates"]
    take = updates * training.batch_size
    variants = _variants(config)
    if (type(updates) is not int or not 1 <= take <= len(data) or type(config["microbatch"]) is not int
            or not 1 <= config["microbatch"] <= training.batch_size
            or set((0, updates)) - set(config["observe_updates"])
            or any(type(n) is not int or not 0 <= n <= updates for n in config["observe_updates"])):
        raise ValueError("Invalid bounded training or evaluation budget")
    save_updates = config.get("save_updates", ())
    if (any(type(n) is not int or not 0 < n < updates for n in save_updates)
            or len(save_updates) != len(set(save_updates))):
        raise ValueError("Intermediate weight checkpoints require distinct update numbers before the endpoint")
    if data.split != "train" or data.dataset_sha256 != settings["data_sha256"]:
        raise ValueError("Expected the designated training membership")
    output.mkdir(parents=True, exist_ok=False)
    rows = random.Random(config["row_seed"]).sample(range(len(data)), take)
    checked = verify_canonical_training_cache(data, rows)
    report = dict(status="running", arms={}, updates_per_arm=updates, batch_size=training.batch_size,
                  microbatch=config["microbatch"], available_training_rows=len(data),
                  dataset_sha256=data.dataset_sha256, sampled_without_replacement=True,
                  row_seed=config["row_seed"], row_order_sha256=hashlib.sha256(
                      torch.tensor(rows, dtype=torch.int64).numpy().tobytes()).hexdigest(),
                  train_canonical_rows_checked=checked, training_seed=training.seed, panel=panel.metadata,
                  config=config, base_run=settings, device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    summary = output / "comparison.json"
    saved_arm = _saved_controls(config, report) if "controls" in config else None
    write_json(summary, report)
    shared = None
    for variant in variants:
        arm = report["arms"][variant] = {"status": "initializing"}
        write_json(summary, report)
        arm_start = time.perf_counter()
        arm_output = output / variant
        try:
            arm_output.mkdir()
            seed_everything(training.seed)
            model_config = (replace(training.model, decoder_variant=HYBRID, **config["decoder_widths"])
                            if variant == WIDE_HYBRID else replace(training.model, decoder_variant=variant))
            run = replace(training, device=str(device), model=model_config)
            model = GraphAutoencoder(data.spec, run.model)
            if variant == WIDE_HYBRID:
                arm.update(_initialize_wide_hybrid(model, training, panel.prior.codes[:4]))
            elif variant == HYBRID:
                arm.update(_initialize_hybrid(model, run, panel.prior.codes[:4]))
                if variants == ALL_VARIANTS:
                    source_digest = report["arms"]["bond_message"]["initial_bond_message_state_sha256"]
                    if arm["initial_bond_message_state_sha256"] != source_digest:
                        raise ValueError("Initial hybrid message tensors differ from the actual message-only arm")
            elif shared is None:
                shared = shared_parameters(model)
            else:
                if shared_parameters(model).keys() != shared.keys():
                    raise ValueError("The architectures differ outside their refinement rounds")
                model.load_state_dict({**model.state_dict(), **shared}, strict=True)
            if variants == ALL_VARIANTS and variant == "bond_message":
                arm["initial_bond_message_state_sha256"] = state_digest(model.state_dict())
            arm.update(initial_shared_sha256=state_digest(shared_parameters(model)),
                       encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                       decoder_parameters=sum(p.numel() for p in model.decoder.parameters()), status="training")
            model.to(device)
            if variant == "triangle_refine":
                if arm["encoder_parameters"] + arm["decoder_parameters"] != settings["expected_parameters"]:
                    raise ValueError("The baseline parameter count differs from its training recipe")
            if saved_arm and arm["initial_shared_sha256"] != saved_arm["initial_shared_sha256"]:
                raise ValueError("Hybrid shared tensors differ from the recorded initialization")
            if variant == variants[0] and device.type == "cuda" and (marker := os.environ.get("DBMOL_LAUNCH_READY")):
                write_json(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                              nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
            write_json(summary, report)
            arm.update(train_arm(config, settings, run, data, panel, rows, model, arm_output, device))
            if variant in ("bond_message", HYBRID):
                baseline = saved_arm if saved_arm is not None else report["arms"]["triangle_refine"]
                for field in ("initial_shared_sha256", "prior_noise_sha256", "posterior_noise_sha256", "prior_draws"):
                    if arm[field] != baseline[field]:
                        raise RuntimeError(f"Unmatched training arms for {field}")
            del model
        except Exception as exc:
            arm.update(status="failed", failure_type=type(exc).__name__, failure_message=str(exc))
            trace = arm_output / "training.jsonl"
            if trace.exists():
                with trace.open() as handle:
                    arm["completed_updates"] = sum(1 for _ in handle)
            arm["elapsed_seconds"] = time.perf_counter() - arm_start
            if device.type == "cuda":
                arm["cuda_peak_allocated_bytes_at_failure"] = torch.cuda.max_memory_allocated(device)
            report["status"] = "failed"
            write_json(summary, report)
            raise
        arm["elapsed_seconds"] = time.perf_counter() - arm_start
        write_json(summary, report)
    report["status"] = "completed"
    write_json(summary, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config, settings, training = read_comparison(args.config)
    if not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Production comparison requires an allocated DTU GPU")
    _check_cuda_requirement(settings)
    torch.set_num_threads(settings["execution"]["torch_threads"])
    corpus = BalancedSparseCorpus(settings["data"], expected_sha256=settings["data_sha256"])
    data = corpus.split("train")
    if len(data) != settings["expected_training_rows"]:
        raise ValueError("Incorrect balanced training membership")
    panel = CompactTrainingPanel(corpus.split("val"), training, **settings["panel"])
    run_comparison(config, settings, training, data, panel, args.output, "cuda")


if __name__ == "__main__":
    main()
