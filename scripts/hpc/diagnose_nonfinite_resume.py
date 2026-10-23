"""Replay one failed VAE epoch on HPC and locate its first non-finite forward value.

This script lives outside ``latentmol`` so the checkpoint's executable package
hash remains unchanged. It observes the existing training code; it does not
alter the model, loss, optimizer, or random numbers used by the replay.
"""

import argparse
import json
import os
from pathlib import Path

import torch

import latentmol.train as trainer


def tensor_stats(value):
    """Return small, JSON-safe diagnostics without exporting molecule contents."""
    if not isinstance(value, torch.Tensor):
        return None
    detached = value.detach()
    finite = torch.isfinite(detached)
    valid = detached[finite]
    return {
        "shape": list(detached.shape),
        "nonfinite": int((~finite).sum().item()),
        "nan": int(torch.isnan(detached).sum().item()),
        "positive_inf": int(torch.isposinf(detached).sum().item()),
        "negative_inf": int(torch.isneginf(detached).sum().item()),
        "finite_min": float(valid.min().item()) if valid.numel() else None,
        "finite_max": float(valid.max().item()) if valid.numel() else None,
        "finite_abs_max": float(valid.abs().max().item()) if valid.numel() else None,
    }


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)


def first_bad_named_tensors(values):
    for name, value in values:
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            return {"name": name, "stats": tensor_stats(value)}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--failure-snapshot", type=Path)
    args = parser.parse_args()
    config = trainer.TrainConfig.from_yaml(args.config)
    checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=True)
    if config.mode != "vae" or checkpoint["epoch"] != 6:
        raise ValueError("Diagnostic is restricted to the failed epoch-7 VAE replay")

    original_forward = trainer.GraphAutoencoder.forward
    original_losses = trainer.reconstruction_losses
    original_kl = trainer.latent_kl
    original_adamw = torch.optim.AdamW
    state = {"batch": 0, "reported": False}

    def capture_optimizer(*positional, **keywords):
        result = original_adamw(*positional, **keywords)
        state["optimizer"] = result
        return result

    def capture_forward(model, graph, sample_latent=False):
        if model.training and sample_latent:
            # Release the preceding batch's autograd graph before this forward.
            for key in ("output", "losses", "target"):
                state.pop(key, None)
            state["batch"] += 1
            state["model"] = model
            state["graph"] = graph
            state["cpu_rng_before"] = torch.get_rng_state()
            state["cuda_rng_before"] = torch.cuda.get_rng_state_all()
        result = original_forward(model, graph, sample_latent=sample_latent)
        if model.training and sample_latent:
            state["output"] = result
        return result

    def capture_losses(predicted, target, loss_config):
        result = original_losses(predicted, target, loss_config)
        if state.get("model") is not None and state["model"].training:
            state["losses"] = result
            state["target"] = target
        return result

    def replay_with_hooks():
        """Replay only the failing forward, restoring post-forward RNG afterward."""
        model = state["model"]
        first_bad = []
        growth_trace = []
        handles = []

        def observe(name):
            def hook(module, inputs, output):
                if first_bad:
                    return
                output_tensors = list(tensor_leaves(output))
                bad = next((item for item in output_tensors
                            if not torch.isfinite(item).all()), None)
                if name.startswith("decoder."):
                    output_stats = [tensor_stats(item) for item in output_tensors]
                    largest = max((item["finite_abs_max"] or 0 for item in output_stats),
                                  default=0)
                    if largest >= 100 or bad is not None or name in (
                        "decoder.nodes", "decoder.pair_init", "decoder.rounds.0",
                        "decoder.rounds.1", "decoder.bond_head.0"
                    ):
                        growth_trace.append({
                            "module": name, "type": type(module).__name__,
                            "input_abs_max": [tensor_stats(item)["finite_abs_max"]
                                              for item in tensor_leaves(inputs)],
                            "output_abs_max": [item["finite_abs_max"] for item in output_stats],
                            "output_nonfinite": [item["nonfinite"] for item in output_stats],
                        })
                if bad is not None:
                    first_bad.append({
                        "module": name,
                        "type": type(module).__name__,
                        "input": [tensor_stats(item) for item in tensor_leaves(inputs)],
                        "output": tensor_stats(bad),
                    })
            return hook

        for name, module in model.named_modules():
            if name:
                handles.append(module.register_forward_hook(observe(name)))
        cpu_after = torch.get_rng_state()
        cuda_after = torch.cuda.get_rng_state_all()
        try:
            torch.set_rng_state(state["cpu_rng_before"])
            torch.cuda.set_rng_state_all(state["cuda_rng_before"])
            with torch.no_grad():
                replay = original_forward(model, state["graph"], sample_latent=True)
            return {
                "first_nonfinite_module": first_bad[0] if first_bad else None,
                "growth_trace": growth_trace,
                "replayed_z": tensor_stats(replay.z),
                "replayed_bond_logits": tensor_stats(replay.decoded.bonds),
                "exact_replay": {
                    name: bool(torch.allclose(actual, expected, rtol=0, atol=0,
                                              equal_nan=True))
                    for name, actual, expected in (
                        ("z", replay.z, state["output"].z),
                        ("atoms", replay.decoded.atoms, state["output"].decoded.atoms),
                        ("charges", replay.decoded.charges, state["output"].decoded.charges),
                        ("bonds", replay.decoded.bonds, state["output"].decoded.bonds),
                    )
                },
            }
        finally:
            for handle in handles:
                handle.remove()
            torch.set_rng_state(cpu_after)
            torch.cuda.set_rng_state_all(cuda_after)

    def report():
        output = state["output"]
        losses = state["losses"]
        model = state["model"]
        optimizer = state["optimizer"]
        beta = config.beta * min(7 / max(config.kl_warmup_epochs, 1), 1.0)
        per_molecule_kl = -0.5 * (
            1.0 + output.logvar - output.mu.square() - output.logvar.exp()
        ).sum(-1)
        payload = {
            "diagnostic_script_sha256": trainer.file_sha256(Path(__file__)),
            "checkpoint": str(args.resume_from),
            "checkpoint_sha256": trainer.file_sha256(args.resume_from),
            "checkpoint_epoch": 6,
            "checkpoint_run_id": checkpoint["run_id"],
            "algorithm_sha256": trainer.algorithm_identity()["sha256"],
            "source_archive_sha256": os.environ.get("LATENTMOL_SOURCE_SHA256"),
            "config_sha256": trainer.file_sha256(args.config),
            "epoch": 7,
            "batch_one_based": state["batch"],
            "beta": beta,
            "batch_size": int(state["target"].atoms.shape[0]),
            "batch_atom_count_min": int((state["target"].atoms != 0).sum(-1).min()),
            "batch_atom_count_max": int((state["target"].atoms != 0).sum(-1).max()),
            "mu": tensor_stats(output.mu),
            "logvar": tensor_stats(output.logvar),
            "z": tensor_stats(output.z),
            "atom_logits": tensor_stats(output.decoded.atoms),
            "charge_logits": tensor_stats(output.decoded.charges),
            "bond_logits": tensor_stats(output.decoded.bonds),
            "losses": {name: tensor_stats(value) for name, value in losses.items()},
            "per_molecule_kl": tensor_stats(per_molecule_kl),
            "first_nonfinite_parameter": first_bad_named_tensors(model.named_parameters()),
            "first_nonfinite_optimizer_state": first_bad_named_tensors(
                (f"{index}.{name}", value)
                for index, parameter in enumerate(model.parameters())
                for name, value in optimizer.state.get(parameter, {}).items()
            ),
        }
        payload["forward_replay"] = replay_with_hooks()
        if args.failure_snapshot is not None:
            # Plain tensors only: a future decoder release can strictly load
            # the weights and score the identical failing latent/target batch.
            args.failure_snapshot.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": 7,
                "batch_one_based": state["batch"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "source_archive_sha256": payload["source_archive_sha256"],
                "graph_spec": model.spec.to_dict(),
                "model_config": checkpoint["model_config"],
                "diagnostic_script_sha256": payload["diagnostic_script_sha256"],
                "model_state": {name: value.detach().cpu()
                                for name, value in model.state_dict().items()},
                "z": output.z.detach().cpu(),
                "target_atoms": state["target"].atoms.detach().cpu(),
                "target_charges": state["target"].charges.detach().cpu(),
                "target_bonds": state["target"].bonds.detach().cpu(),
            }, args.failure_snapshot)
            payload["failure_snapshot"] = str(args.failure_snapshot)
            payload["failure_snapshot_sha256"] = trainer.file_sha256(args.failure_snapshot)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print("NONFINITE_DIAGNOSTIC=" + json.dumps(payload, sort_keys=True), flush=True)
        state["reported"] = True

    def capture_kl(mu, logvar):
        result = original_kl(mu, logvar)
        if state.get("model") is not None and state["model"].training:
            state["losses"]["kl"] = result
            total = state["losses"]["reconstruction"] + (
                config.beta * min(7 / max(config.kl_warmup_epochs, 1), 1.0)
            ) * result
            if not torch.isfinite(total):
                try:
                    report()
                except Exception as error:
                    print(f"DIAGNOSTIC_CAPTURE_ERROR={error!r}", flush=True)
                    raise
        return result

    trainer.GraphAutoencoder.forward = capture_forward
    trainer.reconstruction_losses = capture_losses
    trainer.latent_kl = capture_kl
    torch.optim.AdamW = capture_optimizer
    try:
        trainer.train(config, args.data, args.run_root,
                      resume_from=args.resume_from, stop_after_epoch=7)
    except FloatingPointError as error:
        if state["reported"] and str(error) == "Non-finite training loss":
            print("DIAGNOSTIC_REPRODUCED=1", flush=True)
            return
        raise
    raise RuntimeError("Epoch 7 completed without reproducing the original non-finite loss")


if __name__ == "__main__":
    main()
