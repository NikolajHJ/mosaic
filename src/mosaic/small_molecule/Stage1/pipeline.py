"""Run the Stage 1 baseline and record how the ligand changes.

Start from a random state or a saved checkpoint selected in the general YAML.
The optional frozen VAE also accepts a SMILES start and updates its latent.
Protenix-Tiny scores whole-domain contacts. Boltz2 scores affinity with a
prepared pose. Both update atom and bond probabilities.
This baseline measures chemical problems in the resulting graphs without
using them to guide the updates.
"""

from contextlib import ExitStack
import json
import os
from pathlib import Path


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def log_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + "\n")
    stream.flush()


def run(general, stage, general_path, stage_path, output):
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Run Stage 1 inside an allocated DTU HPC GPU job.")
    with ExitStack() as resources:
        optimize(general, stage, general_path, stage_path, output, resources)


def optimize(
    general, stage, general_path, stage_path, output, resources, *,
    after_update=None, prepare_loss=None, stage_details=None,
):
    """Run the same model and optimizer for Stage 1 and its Stage 2 comparisons."""
    import jax
    import numpy as np

    from mosaic.small_molecule.core.experiment import file_sha256, resolve_target
    from mosaic.small_molecule.core.optimizers import evaluate_graph, projected_step
    from mosaic.small_molecule.core.state import RelaxedGraph, initialize_graph, validate_graph

    from .diagnostics import graph_metrics, raw_readout

    def input_path(value):
        return (general_path.parent / Path(value).expanduser()).resolve()

    use_vae = general.get("use_vae", False)
    if type(use_vae) is not bool:
        raise ValueError("use_vae must be a YAML boolean.")
    if use_vae:
        from mosaic.small_molecule.VAE.optimization import LatentOptimization

        if after_update is not None:
            raise ValueError("VAE updates require operators.mode: none; a post-step graph edit has no latent state.")

    protein = general["protein"]
    target = resolve_target(input_path(protein["structure"]), protein["chain"])
    n_atoms = general["n_atoms"]
    initialization = general.get("initialization", {})
    epochs = general["epochs"]
    eta_X = stage.get("eta_X", general["learning_rate"])
    eta_E = stage.get("eta_E", general["learning_rate"])
    seed = general.get("seed", 17)
    init_key, model_key = jax.random.split(jax.random.key(seed))
    model_key_data = np.asarray(jax.random.key_data(model_key))

    output.mkdir(parents=True, exist_ok=False)
    (output / "general_config.yaml").write_bytes(general_path.read_bytes())
    (output / "stage_config.yaml").write_bytes(stage_path.read_bytes())
    write_json(output / "target.json", target)

    model_name = general["model"]
    if model_name == "boltz2":
        from mosaic.small_molecule.core.boltz2.setup import open_affinity

        loss, identity, prepared = resources.enter_context(
            open_affinity(general, general_path, target, output, n_atoms)
        )
        score_name, score_sign = "affinity_pred_value", 1
        objective = "boltz2_fixed_pose_affinity"
        model_record = {"model": identity, "prepared_input": prepared}
        rng_policy = "seeded graph initialization; affinity uses a fixed prepared pose"
    else:
        from mosaic.small_molecule.core.features import prepare_context
        from mosaic.small_molecule.core.losses import NativeContactLoss
        from mosaic.small_molecule.core.protenix import FrozenProtenixTiny, ligand_pae_gate_prerequisites

        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("JAX cannot see an allocated GPU.")
        if protein.get("use_msa", False):
            raise ValueError("Stage 1 Protenix-Tiny uses the target sequence without an MSA.")
        jax.config.update("jax_default_matmul_precision", "highest")
        cache = input_path(general["model_cache"]) if general.get("model_cache") else None
        model = FrozenProtenixTiny.load(cache_dir=cache)
        # The chain sets up the slots. Its coordinates are disabled and the
        # graph below supplies the changing atom and bond probabilities.
        context, _ = prepare_context(
            "C" * n_atoms, protein_chains=(target["protein_sequence"],),
            cache_dir=cache, input_name="stage1",
        )
        loss = model.build_loss(context=context, loss=NativeContactLoss(cutoff_angstrom=8.0))
        score_name, score_sign = "contact_score", -1
        objective = "whole_target_contact_8A"
        model_record = {
            "model": model.identities,
            "prepared_input": context.identities,
            "pae_gate": ligand_pae_gate_prerequisites(context) | {"native_pae_trial": "not_run"},
        }
        rng_policy = "same model key at every state and for its hard readout"

    parent = None
    latent = None
    if checkpoint := initialization.get("checkpoint"):
        path = input_path(checkpoint)
        parent_manifest = json.loads((path.parent / "manifest.json").read_text())
        parent_target = json.loads((path.parent / "target.json").read_text())
        if (
            parent_manifest["objective"] != objective
            or parent_manifest["model"] != model_record["model"]
            or parent_manifest["prepared_input"] != model_record["prepared_input"]
            or parent_target["structure_sha256"] != target["structure_sha256"]
            or parent_target["chain"] != target["chain"]
        ):
            raise ValueError("The starting checkpoint must use the same target, model and objective.")
        import jax.numpy as jnp

        with np.load(path, allow_pickle=False) as saved:
            if use_vae:
                latent = LatentOptimization.create(general, general_path, n_atoms, saved=saved)
                graph = latent.graph()
            else:
                if "vae_z" in saved:
                    raise ValueError("Enable use_vae to resume this latent checkpoint.")
                graph = RelaxedGraph(jnp.asarray(saved["X"]), jnp.asarray(saved["E_upper"]))
            model_key_data = saved["model_key"]
            parent = {"path": str(path), "sha256": file_sha256(path), "state": int(saved["state"])}
        validate_graph(graph)
        if graph.n_atoms != n_atoms:
            raise ValueError("The checkpoint must match n_atoms in the general YAML.")
        model_key = jax.random.wrap_key_data(jnp.asarray(model_key_data))
    elif use_vae:
        latent = LatentOptimization.create(general, general_path, n_atoms)
        graph = latent.graph()
    else:
        graph = initialize_graph(n_atoms, init_key, **initialization["random"])

    loss_for_state = prepare_loss(loss, epochs) if prepare_loss is not None else None
    manifest = {
        "status": "running",
        "stage": stage["stage"].lower(),
        **(stage_details or {}),
        **({"starting_checkpoint": parent} if parent else {}),
        "objective": objective,
        "model_name": model_name,
        **model_record,
        "seed": seed,
        "model_key": model_key_data.tolist(),
        "rng_policy": rng_policy,
        "optimizer": latent.optimizer_record if use_vae else {
            "name": "projected_gradient", "eta_X": eta_X, "eta_E": eta_E,
        },
        "updates_requested": epochs,
    }
    write_json(output / "manifest.json", manifest)

    atoms, bonds = [], []
    initial_bond_mass = 1 - np.asarray(graph.E_upper)[:, 0]
    # Track turnover among roughly as many edges as the initial expected bonds.
    n_strong_edges = max(1, round(float(initial_bond_mass.sum())))
    initial_strong = np.argsort(-initial_bond_mass, kind="stable")[:n_strong_edges]
    initial_mask = np.zeros_like(initial_bond_mass, dtype=bool)
    initial_mask[initial_strong] = True
    completed = 0
    operator_report = None
    try:
        with (output / "trace.jsonl").open("w", encoding="utf-8") as trace, (
            output / "hard_readouts.jsonl"
        ).open("w", encoding="utf-8") as readouts:
            for state in range(epochs + 1):
                current_loss = loss_for_state(graph, state) if loss_for_state else loss
                evaluation = evaluate_graph(current_loss, graph, model_key)
                # A prepared loss may add a prior. Keep the model score separate
                # from that combined objective, including between readouts.
                native_value = evaluation.aux.get("defog_native_loss", evaluation.value)
                soft_score = score_sign * float(native_value)
                atoms.append(np.asarray(graph.X))
                bonds.append(np.asarray(graph.E_upper))
                current_mass = 1 - bonds[-1][:, 0]
                current_strong = np.argsort(-current_mass, kind="stable")[:n_strong_edges]
                row = {
                    "state": state,
                    "loss": float(evaluation.value),
                    score_name: soft_score,
                    "atom_gradient_norm": float(np.linalg.norm(evaluation.gradients.X)),
                    "bond_gradient_norm": float(np.linalg.norm(evaluation.gradients.E_upper)),
                    "atom_change_from_start": float(np.linalg.norm(atoms[-1] - atoms[0])),
                    "bond_change_from_start": float(np.linalg.norm(bonds[-1] - bonds[0])),
                    "initial_strong_edges_weakened": int(np.sum(
                        current_mass[initial_mask] < initial_bond_mass[initial_mask] - 1e-6
                    )),
                    "new_edges_among_strongest": int(np.sum(~initial_mask[current_strong])),
                    **graph_metrics(graph),
                }
                if use_vae and latent.last_gradient_norm is not None:
                    row["previous_latent_gradient_norm"] = latent.last_gradient_norm
                if model_name == "boltz2":
                    row.update({name: float(value) for name, value in evaluation.aux.items()
                                if name.startswith("affinity_")})
                if operator_report is not None:
                    row["chemistry_update"] = operator_report
                if loss_for_state and "forward_hardness" in evaluation.aux:
                    row["forward_hardness"] = float(evaluation.aux["forward_hardness"])
                row.update({name: float(value) for name, value in evaluation.aux.items()
                            if name.startswith("defog_")})
                log_row(trace, row)

                if state in (0, epochs) or state % stage["readout_every"] == 0:
                    # Score raw argmax choices with the same model inputs and key.
                    # RDKit validity is a report; it never changes the graph.
                    hard_graph, report = raw_readout(graph)
                    hard_loss, hard_metrics = loss(hard_graph, key=model_key)
                    hard_score = score_sign * float(hard_loss)
                    # For a hardening comparison, measure the gap against the
                    # underlying soft graph, not the hardened model input.
                    soft_loss = (
                        loss(graph, key=model_key)[0]
                        if "forward_hardness" in evaluation.aux else native_value
                    )
                    soft_score = score_sign * float(soft_loss)
                    log_row(readouts, {
                        "state": state,
                        f"soft_{score_name}": soft_score,
                        f"hard_{score_name}": hard_score,
                        "hard_minus_soft_loss": float(hard_loss) - float(soft_loss),
                        **(
                            {"soft_minus_hard_score": soft_score - hard_score}
                            if model_name == "protenix-tiny"
                            else {f"hard_{name}": float(value) for name, value in hard_metrics.items()}
                        ),
                        **report,
                    })
                    checkpoint = output / "checkpoint.partial.npz"
                    np.savez_compressed(
                        checkpoint, X=atoms[-1], E_upper=bonds[-1], state=state,
                        model_key=model_key_data,
                        **(latent.arrays() if use_vae else {"eta_X": eta_X, "eta_E": eta_E}),
                    )
                    checkpoint.replace(output / "checkpoint.npz")

                if state < epochs:
                    if use_vae:
                        graph = latent.step(evaluation)
                    else:
                        graph = projected_step(evaluation, eta_X=eta_X, eta_E=eta_E).next_graph
                    if after_update is not None:
                        graph, operator_report = after_update(graph, state + 1)
                    completed += 1

        np.savez_compressed(output / "trajectory.npz", X=atoms, E_upper=bonds)
        write_json(output / "manifest.json", manifest | {
            "status": "complete", "updates_performed": completed, "states_written": len(atoms),
        })
    except Exception as error:
        if atoms:
            np.savez_compressed(output / "trajectory.partial.npz", X=atoms, E_upper=bonds)
        write_json(output / "manifest.json", manifest | {
            "status": "failed", "updates_performed": completed, "states_written": len(atoms),
            "error": f"{type(error).__name__}: {error}",
        })
        raise
