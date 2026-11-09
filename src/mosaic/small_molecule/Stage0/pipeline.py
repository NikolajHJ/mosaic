"""Score and update a starting ligand with the model chosen in the general YAML.

Protenix-Tiny scores contacts; Boltz2 scores affinity with a prepared pose
held fixed. Both update the same atom and bond probabilities and save every
state, including the initial graph.
"""

from contextlib import ExitStack
import json
import os
from pathlib import Path


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(general, stage, general_path, stage_path, output):
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Run Stage 0 inside an allocated DTU HPC GPU job.")
    with ExitStack() as resources:
        optimize(general, stage, general_path, stage_path, output, resources)


def optimize(general, stage, general_path, stage_path, output, resources):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mosaic.small_molecule.experiment import LossSchedule, file_sha256, resolve_target
    from mosaic.small_molecule.optimizers import evaluate_graph, projected_step
    from mosaic.small_molecule.state import RelaxedGraph, graph_from_hard, validate_graph

    def input_path(value):
        return (general_path.parent / Path(value).expanduser()).resolve()

    protein = general["protein"]
    pocket = general.get("pocket")
    selection = None
    if pocket and "reference_ligand" in pocket:
        selection = {
            "kind": "reference_ligand",
            "ligand": pocket["reference_ligand"],
            "cutoff_angstrom": pocket.get("cutoff_angstrom", 6.0),
        }
    elif pocket:
        selection = {"kind": "residues", "residues": pocket["residues"]}
    target = resolve_target(input_path(protein["structure"]), protein["chain"], selection)

    init = general.get("initialization", {})
    metadata, graph = None, None
    smiles = init.get("smiles")
    if init.get("input"):
        folder = input_path(init["input"])
        metadata = json.loads((folder / "initialization.json").read_text(encoding="utf-8"))
        graph_path = folder / metadata["graph_file"]
        if file_sha256(graph_path) != metadata["graph_sha256"]:
            raise ValueError("Starting graph differs from its preparation record.")
        with np.load(graph_path, allow_pickle=False) as arrays:
            graph = RelaxedGraph(jnp.asarray(arrays["X"]), jnp.asarray(arrays["E_upper"]))
        validate_graph(graph)
        smiles = metadata["smiles"]

    model_name = general["model"]
    default_loss = "affinity" if model_name == "boltz2" else "contact"
    specs = stage.get("losses", [{"name": default_loss}])
    weights = {spec["name"]: spec.get("weight", 1.0) for spec in specs}
    epochs = general["epochs"]
    schedule = (
        LossSchedule(tuple(weights), tuple(stage["weight_schedule"]))
        if "weight_schedule" in stage else None
    )

    output.mkdir(parents=True, exist_ok=False)
    (output / "general_config.yaml").write_bytes(general_path.read_bytes())
    (output / "stage_config.yaml").write_bytes(stage_path.read_bytes())
    write_json(output / "target.json", target)

    if model_name == "boltz2":
        from mosaic.small_molecule.boltz2.setup import open_affinity

        if list(weights) != ["affinity"]:
            raise ValueError("Boltz2 Stage 0 uses affinity. Remove other losses from the stage YAML.")
        if graph is None and smiles:
            raise ValueError("For Boltz2, omit initialization to use the prepared chain, or use initialization.input.")
        affinity, identity, prepared = resources.enter_context(
            open_affinity(general, general_path, target, output, graph.n_atoms if graph is not None else None)
        )
        if graph is None:
            # The prepared context has one slot per carbon in a linear chain.
            n = prepared["n_atoms"]
            i, j = np.triu_indices(n, k=1)
            graph = graph_from_hard(np.zeros(n, dtype=np.int32), (j == i + 1).astype(np.int32))
            smiles = prepared["seed_smiles"]

        def score(graph, key):
            value, aux = affinity(graph, key=key)
            return {"affinity": value}, aux
    else:
        from mosaic.small_molecule.features import NATIVE_ELEMENT_INDICES, prepare_context
        from mosaic.small_molecule.losses import DBMolContactLoss, NativeContactLoss
        from mosaic.small_molecule.protenix import FrozenProtenixTiny
        from mosaic.small_molecule.state import ELEMENTS

        if protein.get("use_msa", False):
            raise ValueError("The Protenix graph path does not use a protein MSA.")
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("JAX cannot see an allocated GPU.")
        jax.config.update("jax_default_matmul_precision", "highest")
        cache = input_path(general["model_cache"]) if general.get("model_cache") else None
        model = FrozenProtenixTiny.load(cache_dir=cache)
        context, hard = prepare_context(
            smiles, protein_chains=(target["protein_sequence"],), cache_dir=cache, input_name="stage0"
        )
        if graph is None:
            graph = hard
        else:
            expected = [NATIVE_ELEMENT_INDICES[ELEMENTS.index(e)] for e in metadata["elements_in_atom_order"]]
            actual = np.asarray(context.base_features["ref_element"])[list(context.ligand_atom_indices)].argmax(-1)
            if graph.n_atoms != len(expected) or not np.array_equal(actual, expected):
                raise ValueError("Starting graph and model atom slots do not match.")

        protein_tokens = np.asarray(context.protein_token_indices)
        positions = np.asarray(context.base_features["residue_index"])[protein_tokens]
        by_position = dict(zip(positions.tolist(), protein_tokens.tolist()))
        pocket_tokens = tuple(by_position[r["sequence_position"]] for r in target["residues"])
        terms = {}
        for spec in specs:
            name, cutoff = spec["name"], spec.get("cutoff_angstrom", 8.0)
            if name == "contact":
                terms[name] = NativeContactLoss(cutoff, pocket_tokens)
            elif name == "pocket_coverage":
                terms[name] = DBMolContactLoss(pocket_tokens, cutoff)
            else:
                raise ValueError("Protenix-Tiny Stage 0 supports contact and pocket_coverage.")
        identity = model.identities

        def score(graph, key):
            prediction = model.forward_trunk(context=context, graph=graph, key=key)
            values, metrics = {}, {}
            for name, term in terms.items():
                values[name], aux = term(graph, prediction, key=key)
                metrics.update({f"{name}_{k}": v for k, v in aux.items()})
            return values, metrics

    def make_loss(effective_weights):
        def loss(graph, *, key):
            values, metrics = score(graph, key)
            total = sum(effective_weights[name] * value for name, value in values.items())
            return total, metrics | {f"loss_{name}": value for name, value in values.items()}
        return loss

    trace, atoms, bonds = [], [], []
    key = jax.random.key(general.get("seed", 17))
    rate = general["learning_rate"]
    for state in range(epochs + 1):
        factors = schedule.at(min(state + 1, epochs)) if schedule else dict.fromkeys(weights, 1.0)
        effective = {name: weights[name] * factors[name] for name in weights}
        evaluation = evaluate_graph(make_loss(effective), graph, jax.random.fold_in(key, state))
        trace.append({
            "state": state,
            "total_loss": float(evaluation.value),
            "weights": effective,
            "components": {k: float(v) for k, v in evaluation.aux.items() if np.ndim(v) == 0},
        })
        atoms.append(np.asarray(graph.X))
        bonds.append(np.asarray(graph.E_upper))
        if state < epochs:
            graph = projected_step(evaluation, eta_X=rate, eta_E=rate).next_graph

    write_json(output / "trace.json", trace)
    np.savez_compressed(output / "trajectory.npz", X=atoms, E_upper=bonds)
    write_json(output / "manifest.json", {
        "status": "complete",
        "updates_performed": epochs,
        "states_written": len(trace),
        "model": identity,
        "initial_smiles": smiles,
        "initialization": metadata,
    })
