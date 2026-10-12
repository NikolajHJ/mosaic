"""Restartable Stage 0 engineering example. Numerical execution belongs on DTU HPC.

Use the already staged, pinned Tiny assets; this entry point never requests an
MSA or downloads weights. Ligand-only uses a labelled distogram diagnostic;
protein-plus-ligand uses the declared native-bin contact debugging score.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from mosaic.small_molecule.contracts import CONTRACT_VERSION
from mosaic.small_molecule.features import prepare_context
from mosaic.small_molecule.io import (
    TrajectoryConfig,
    file_sha256,
    identity_digest,
    load_context,
    load_graph,
    run_trajectory,
    save_context,
    save_graph,
    write_json,
)
from mosaic.small_molecule.losses import DistogramDiagnosticLoss, NativeContactLoss
from mosaic.small_molecule.protenix import FrozenProtenixTiny, native_distogram_bins
from mosaic.small_molecule.provenance import environment_identity, source_identity
from mosaic.small_molecule.state import ELEMENTS, initialize_graph, mix_with_uniform
from mosaic.structure_prediction import TargetChain


def read_config(path, root):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["schema"] != CONTRACT_VERSION or config["geometry_mode"] != "off":
        raise ValueError("Only the current geometry-off contract is supported")
    if config["precision"] != "float32" or config["recycling_steps"] != 1:
        raise ValueError("Stage 0 supports float32 and exactly one trunk recycle")
    if (
        not config["modes"]
        or len(set(config["modes"])) != len(config["modes"])
        or any(
            mode not in ("ligand_only", "protein_ligand") for mode in config["modes"]
        )
    ):
        raise ValueError("Select distinct ligand_only and/or protein_ligand modes")
    TrajectoryConfig(**config["trajectory"]).validate()
    fixture_path = (root / config["fixture"]).resolve()
    if not fixture_path.is_relative_to(root):
        raise ValueError("Fixture must be inside the source release")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture["contract"] != CONTRACT_VERSION:
        raise ValueError("Fixture contract mismatch")
    return config, fixture, fixture_path


def check_fixture(graph, fixture):
    symbols = [ELEMENTS[i] for i in np.asarray(graph.X).argmax(-1)]
    if graph.n_atoms != fixture["n_heavy_atoms"] or symbols != fixture["element_order"]:
        raise ValueError(
            "Prepared native atom ordering differs from the sealed fixture"
        )
    from mosaic.small_molecule.state import BOND_STATES

    expected = {
        (i, j): BOND_STATES.index(category)
        for i, j, category in fixture["bond_assignments"]
    }
    rows = [
        expected.get((i, j), 0)
        for i in range(graph.n_atoms)
        for j in range(i + 1, graph.n_atoms)
    ]
    if not np.array_equal(rows, np.asarray(graph.E_upper).argmax(-1)):
        raise ValueError("Prepared native bond order differs from the sealed fixture")


def run_mode(
    model,
    *,
    mode,
    root,
    output,
    cache_dir,
    config,
    fixture,
    fixture_path,
    source,
    environment,
    resume,
    max_updates,
    validate,
):
    mode_dir = output / mode
    inputs = mode_dir / "inputs"
    key = jax.random.key(config["seed"])
    init_key, forward_key = jax.random.split(key)
    input_receipt = mode_dir / "input_identity.json"
    requested_identity = {
        "config_sha256": identity_digest(config),
        "fixture_sha256": file_sha256(fixture_path),
        "source": source,
        "environment": environment,
        "mode": mode,
        "model": model.identities,
    }
    if resume:
        receipt = json.loads(input_receipt.read_text(encoding="utf-8"))
        if receipt["requested_identity"] != requested_identity:
            raise ValueError(
                "Source/config/fixture/model/environment changed since the prepared run"
            )
        context = load_context(inputs, receipt["context_files"])
        hard = load_graph(inputs / "hard_graph.npz", receipt["hard_graph_sha256"])
        initial = load_graph(
            inputs / "initial_graph.npz", receipt["initial_graph_sha256"]
        )
    else:
        if (
            input_receipt.exists()
            or (mode_dir / "trajectory" / "manifest.json").exists()
        ):
            raise FileExistsError("Choose a unique output directory or use --resume")
        chains = (
            ()
            if mode == "ligand_only"
            else (TargetChain(fixture["protein_sequence"], use_msa=False),)
        )
        context, hard = prepare_context(
            fixture["smiles"],
            protein_chains=chains,
            cache_dir=cache_dir,
            input_name=f"stage0_{mode}",
        )
        check_fixture(hard, fixture)
        initialization = config["initialization"]
        if initialization["mode"] == "interior_hard":
            initial = mix_with_uniform(hard, initialization["mixing"])
        elif initialization["mode"] == "carbon_heavy":
            initial = initialize_graph(
                hard.n_atoms,
                init_key,
                **{
                    name: initialization[name]
                    for name in (
                        "carbon_probability",
                        "expected_degree",
                        "atom_noise",
                        "bond_noise",
                    )
                },
            )
        else:
            raise ValueError("Unknown initialization mode")
        receipt = {
            "requested_identity": requested_identity,
            "context_files": save_context(inputs, context),
            "hard_graph_sha256": save_graph(inputs / "hard_graph.npz", hard),
            "initial_graph_sha256": save_graph(inputs / "initial_graph.npz", initial),
        }
        write_json(input_receipt, receipt)
    objective = (
        DistogramDiagnosticLoss(
            ligand_pair=tuple(config["loss"]["diagnostic_pair"]),
            bin_index=config["loss"]["diagnostic_bin"],
        )
        if mode == "ligand_only"
        else NativeContactLoss(
            cutoff_angstrom=config["loss"]["contact_cutoff_angstrom"]
        )
    )
    loss = model.build_loss(
        context=context, loss=objective, recycling_steps=config["recycling_steps"]
    )
    report = None
    if validate:
        from mosaic.small_molecule.validation import validate_pipeline

        report = validate_pipeline(
            model,
            context,
            hard,
            key=forward_key,
            output_directory=mode_dir / "validation",
            settings=config
            | {"cache_dir": str(cache_dir.resolve()), "source_root": str(root)},
        )
        if report["status"] != "passed":
            raise RuntimeError(
                f"Native acceptance checks failed or remain pending: {mode}"
            )
    result = run_trajectory(
        loss,
        initial,
        forward_key,
        directory=mode_dir / "trajectory",
        config=TrajectoryConfig(**config["trajectory"]),
        identities=receipt,
        resolved_config=config,
        resume=resume,
        max_updates=max_updates,
    )
    if report is not None:
        write_json(mode_dir / "trajectory" / "validation.json", report)
    print(
        f"{mode}: updates={result.update_step}, complete={result.complete}, best_loss={result.best_value}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "examples/configs/small_molecule_stage0.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing the five pinned native assets (Mosaic cache/protenix)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    config, fixture, fixture_path = read_config(args.config, root)
    source, environment = source_identity(root), environment_identity()
    # One resident model serves both modes and all graph-dependent forwards.
    model = FrozenProtenixTiny.load(cache_dir=args.cache_dir)
    write_json(
        args.output / "model_manifest.json",
        {
            "identities": model.identities,
            "distogram_upper_edges_angstrom": native_distogram_bins(),
            "contact_bin_policy": "upper edge strictly below cutoff; straddling class excluded",
            "output_fields": [
                "distogram_logits",
                "distogram_bins",
                "trunk_state",
                "ligand_token_indices",
                "protein_token_indices",
            ],
            "unsupported": [
                "ligand geometry",
                "confidence/PAE",
                "coordinates",
                "sampling",
                "recycling_steps != 1",
            ],
        },
    )
    for mode in config["modes"]:
        run_mode(
            model,
            mode=mode,
            root=root,
            output=args.output,
            cache_dir=args.cache_dir,
            config=config,
            fixture=fixture,
            fixture_path=fixture_path,
            source=source,
            environment=environment,
            resume=args.resume,
            max_updates=args.max_updates,
            validate=args.validate,
        )


if __name__ == "__main__":
    main()
