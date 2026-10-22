"""Run a configured small-molecule graph optimization on an allocated GPU."""

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.small_molecule.experiment import LossSchedule, resolve_target, file_sha256
from mosaic.small_molecule.optimizers import evaluate_graph, projected_step
from mosaic.small_molecule.state import ELEMENTS, RelaxedGraph, validate_graph


def load_initialization(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "initialization.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "mosaic-ligand-initialization-v1":
        raise ValueError("Unsupported ligand initialization artifact")
    artifact = folder / metadata["graph_file"]
    if file_sha256(artifact) != metadata["graph_sha256"]:
        raise ValueError("Initial graph checksum differs")
    with np.load(artifact, allow_pickle=False) as data:
        graph = RelaxedGraph(jnp.asarray(data["X"]), jnp.asarray(data["E_upper"]))
    validate_graph(graph)
    if graph.n_atoms != metadata["heavy_atoms"]:
        raise ValueError("Initial graph atom count disagrees with its receipt")
    return graph, metadata


def scalar_metrics(aux):
    result = {}
    for name, value in aux.items():
        array = np.asarray(value)
        if array.shape == ():
            result[name] = float(array)
    return result


def combine_graph_losses(losses, scales, weights):
    def loss(graph, *, key):
        total = jnp.asarray(0.0, dtype=graph.X.dtype)
        metrics = {}
        for name, fn in losses.items():
            value, aux = fn(graph, key=key)
            total = total + weights[name] * scales[name] * value
            metrics[f"loss_{name}"] = value
            metrics.update({f"{name}_{k}": v for k, v in aux.items()})
        metrics["weighted_total"] = total
        return total, metrics

    return loss


def protenix_losses(model, context, pocket_tokens, config):
    from mosaic.small_molecule.losses import DBMolContactLoss, NativeContactLoss

    objectives = {}
    for row in config["losses"]:
        name = row["name"]
        if name == "contact":
            term = NativeContactLoss(row.get("cutoff_angstrom", 8.0), pocket_tokens)
        elif name == "pocket_coverage":
            term = DBMolContactLoss(
                pocket_tokens, row.get("cutoff_angstrom", 8.0)
            )
        else:
            raise ValueError(f"Protenix-Tiny has no configured loss named {name!r}")

        objectives[name] = term

    def build(weights, scales):
        def loss(graph, *, key):
            output = model.forward_trunk(context=context, graph=graph, key=key)
            total = jnp.asarray(0.0, dtype=graph.X.dtype)
            metrics = {}
            for name, objective in objectives.items():
                value, aux = objective(graph, output, key=key)
                total = total + weights[name] * scales[name] * value
                metrics[f"loss_{name}"] = value
                metrics.update({f"{name}_{k}": v for k, v in aux.items()})
            metrics["weighted_total"] = total
            return total, metrics
        return loss

    return build


def run_protenix(config, target, graph, init, output, schedule):
    from mosaic.small_molecule.features import NATIVE_ELEMENT_INDICES, prepare_context
    from mosaic.small_molecule.io import save_context
    from mosaic.small_molecule.protenix import FrozenProtenixTiny

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Protenix-Tiny optimization requires an allocated GPU")
    if graph.n_atoms != len(init["elements_in_atom_order"]):
        raise ValueError("Initial graph slot order does not match its molecule receipt")
    model = FrozenProtenixTiny.load(cache_dir=config["model"]["cache_dir"])
    context, _ = prepare_context(
        init["smiles"],
        protein_chains=(target["protein_sequence"],),
        cache_dir=config["model"]["cache_dir"],
        input_name="ligand_optimization",
        identities={"protein_sequence": target["protein_sequence"]},
    )
    native_elements = np.asarray(context.base_features["ref_element"])[
        list(context.ligand_atom_indices)
    ].argmax(axis=-1)
    expected_elements = np.asarray([
        NATIVE_ELEMENT_INDICES[ELEMENTS.index(element)]
        for element in init["elements_in_atom_order"]
    ])
    if not np.array_equal(native_elements, expected_elements):
        raise ValueError("Native ligand slot atom order differs from the initialization")
    context_files = save_context(output / "context", context)
    protein = np.asarray(context.protein_token_indices)
    residue_indices = np.asarray(context.base_features["residue_index"])[protein]
    if len(protein) != len(target["protein_sequence"]):
        raise ValueError("Prepared target tokens do not match the structure chain")
    residue_to_token = dict(zip(residue_indices.tolist(), protein.tolist()))
    tokens = tuple(
        int(residue_to_token[row["sequence_position"]]) for row in target["residues"]
    )
    loss_builder = protenix_losses(model, context, tokens, config)
    identity = {"model": model.identities, "context_files": context_files}
    return loss_builder, identity, None


def run_boltz2(config, target, graph, output):
    from mosaic.small_molecule.boltz2.bridge import AffinityWorker, affinity_loss

    if len(config["losses"]) != 1 or config["losses"][0]["name"] != "affinity":
        raise ValueError("Current Boltz2 graph adapter supports the differentiable affinity loss")
    settings = config["model"]
    prepared = Path(settings["prepared_context"])
    context_file = prepared / "context.pt"
    preparation = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    if preparation.get("protein_sequence") != target["protein_sequence"]:
        raise ValueError("Boltz2 prepared context protein sequence differs from target")
    audit = preparation.get("pocket_pose_audit", {})
    expected_positions = sorted(row["sequence_position"] for row in target["residues"])
    if audit.get("status") != "passed" or sorted(audit.get("protein_sequence_positions", [])) != expected_positions:
        raise ValueError("Boltz2 context must have a passed pose audit for this exact pocket")
    if file_sha256(context_file) != settings["context_sha256"]:
        raise ValueError("Boltz2 context checksum differs")
    if preparation.get("context_sha256") != settings["context_sha256"]:
        raise ValueError("Boltz2 preparation receipt does not match the context")
    command = [
        settings["worker_python"], "-u", "-m", "mosaic.small_molecule.boltz2.worker",
        "--context", str(context_file), "--context-sha256", settings["context_sha256"],
        "--checkpoint", settings["checkpoint"],
    ]
    worker = AffinityWorker(command, stderr=output / "boltz_worker.log")
    if worker.identity.get("pocket_definition_sha256") != preparation.get("pocket_definition_sha256"):
        worker.close()
        raise ValueError("Boltz2 worker and preparation pocket identities differ")
    if graph.n_atoms != worker.identity["n_atoms"]:
        worker.close()
        raise ValueError("Initial graph slot count differs from prepared Boltz2 context")
    return {"affinity": affinity_loss(worker)}, {
        "native_model": worker.identity,
        "prepared_context": str(context_file),
        "prepared_context_sha256": settings["context_sha256"],
    }, worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Model optimization must run in an allocated DTU HPC job")
    jax.config.update("jax_default_matmul_precision", "highest")
    config_path = args.config.resolve()
    base = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    steps = config["updates"]
    if type(steps) is not int or steps < 1:
        raise ValueError("updates must be a positive integer")
    learning_rate = float(config["learning_rate"])
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    names = tuple(row["name"] for row in config["losses"])
    if not names or len(set(names)) != len(names):
        raise ValueError("Loss names must be nonempty and unique")
    schedule = LossSchedule(names, tuple(config["weight_schedule"]), steps)
    for row in config["losses"]:
        row["scale"] = float(row.get("scale", 1.0))
        if not np.isfinite(row["scale"]) or row["scale"] < 0:
            raise ValueError("Loss scales must be finite and nonnegative")
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else (base / value).resolve()

    target_cfg = config["target"]
    structure = path(target_cfg["structure"])
    target = resolve_target(structure, target_cfg["chain"], target_cfg["pocket"])
    graph, init = load_initialization(path(config["initialization"]))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "target.json").write_text(json.dumps(target, indent=2) + "\n")
    (args.output / "config.json").write_bytes(config_path.read_bytes())
    model_name = config["model"]["name"]
    worker = None
    try:
        if model_name == "protenix-tiny":
            model_config = dict(config)
            model_config["model"]["cache_dir"] = str(path(config["model"]["cache_dir"]))
            loss_builder, identity, worker = run_protenix(
                model_config, target, graph, init, args.output, schedule
            )
        elif model_name == "boltz2":
            if any(device.platform != "cpu" for device in jax.devices()):
                raise RuntimeError("Boltz2 uses the isolated GPU worker; run the JAX optimizer on CPU")
            model_config = dict(config)
            model_config["model"] = dict(config["model"])
            model_config["model"]["prepared_context"] = str(path(config["model"]["prepared_context"]))
            model_config["model"]["checkpoint"] = str(path(config["model"]["checkpoint"]))
            model_config["model"]["worker_python"] = str(path(config["model"]["worker_python"]))
            losses, identity, worker = run_boltz2(model_config, target, graph, args.output)
            loss_builder = None
        else:
            raise ValueError("model.name must be protenix-tiny or boltz2")
        available = set(names) if loss_builder is not None else set(losses)
        if set(names) != available:
            raise ValueError(f"Configured losses {names} do not match available losses {tuple(available)}")
        scales = {row["name"]: row["scale"] for row in config["losses"]}
        identity.update({"target_structure_sha256": target["structure_sha256"], "initial_graph_sha256": init["graph_sha256"]})
        (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        states_x, states_e, trace = [np.asarray(graph.X)], [np.asarray(graph.E_upper)], []
        rng = jax.random.key(config.get("seed", 17))
        for state in range(steps + 1):
            schedule_update = min(state + 1, steps)
            weights = schedule.at(schedule_update)
            loss = (
                combine_graph_losses(losses, scales, weights)
                if loss_builder is None
                else loss_builder(weights, scales)
            )
            key = jax.random.fold_in(rng, state)
            evaluation = evaluate_graph(loss, graph, key)
            trace.append({
                "state": state,
                "loss_weights_pct": {name: 100 * weight for name, weight in weights.items()},
                "total_loss": float(evaluation.value),
                "components": scalar_metrics(evaluation.aux),
            })
            if state == steps:
                break
            proposal = projected_step(
                evaluation, eta_X=learning_rate, eta_E=learning_rate
            )
            graph = proposal.next_graph
            states_x.append(np.asarray(graph.X))
            states_e.append(np.asarray(graph.E_upper))
        (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        np.savez_compressed(
            args.output / "trajectory.npz",
            X=np.asarray(states_x, dtype=np.float32),
            E_upper=np.asarray(states_e, dtype=np.float32),
        )
        manifest = {
            "status": "complete",
            "model": model_name,
            "updates_requested": steps,
            "updates_performed": len(states_x) - 1,
            "states_written": len(states_x),
            "learning_rate": learning_rate,
            "target": target,
            "initialization": init,
            "losses": config["losses"],
            "weight_schedule": config["weight_schedule"],
            "identity": identity,
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Completed {len(states_x)-1} updates; results: {args.output}", flush=True)
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }, indent=2) + "\n")
        raise
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
