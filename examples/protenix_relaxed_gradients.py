"""Run an explicit frozen Tiny contact-gradient study on allocated DTU HPC."""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path


def _case_specs(config):
    """Keep uniform-start controls distinct from mixtures of a hard scaffold."""
    initialization = config.get("initialization")
    if initialization is not None:
        if "mixing_values" in config:
            raise ValueError("Choose initialization or mixing_values, not both")
        if initialization.get("kind") != "uniform":
            raise ValueError("Unsupported initialization kind")
        numerator = initialization.get("bond_existence_numerator")
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, (int, float))
            or not math.isfinite(numerator)
            or numerator <= 0
            or set(initialization) != {"kind", "bond_existence_numerator"}
        ):
            raise ValueError(
                "Uniform initialization requires a positive finite numerator"
            )
        return [
            {
                "mixing": None,
                "initialization": initialization,
                "label": f"Uniform atoms; P(bond)={numerator:g}/N",
            }
        ]
    return [{"mixing": mixing} for mixing in config["mixing_values"]]


def _initial_graph(hard, case, key):
    """Prepare exact equal rows using Mosaic's shared graph initializer."""
    from mosaic.small_molecule.state import ELEMENTS, initialize_graph, mix_with_uniform

    if "initialization" not in case:
        return mix_with_uniform(hard, case["mixing"])
    n = hard.n_atoms
    probability = case["initialization"]["bond_existence_numerator"] / n
    if not 0 < probability < 1:
        raise ValueError("Uniform bond-existence probability must be in (0, 1)")
    return initialize_graph(
        n,
        key,
        carbon_probability=1 / len(ELEMENTS),
        expected_degree=probability * (n - 1),
        atom_noise=0,
        bond_noise=0,
        dtype=hard.X.dtype,
    )


def _contact_objective(config, context, fixture):
    """Select the declared native reduction without changing graph semantics."""
    from mosaic.small_molecule.losses import DBMolContactLoss, NativeContactLoss

    name = config.get("objective", "dbmol_pocket")
    if name == "native_whole_target":
        if config.get("pocket_protein_ordinals") or fixture.get(
            "pocket_protein_ordinals"
        ):
            raise ValueError("Whole-target contact does not accept a selected pocket")
        return NativeContactLoss(cutoff_angstrom=config["contact_cutoff_angstrom"]), ()
    if name != "dbmol_pocket":
        raise ValueError("Unknown contact objective")
    ordinals = config.get(
        "pocket_protein_ordinals", fixture.get("pocket_protein_ordinals")
    )
    if not ordinals:
        raise ValueError("A declared nonempty pocket mapping is required")
    if len(set(ordinals)) != len(ordinals) or any(
        type(i) is not int or not 0 <= i < len(context.protein_token_indices)
        for i in ordinals
    ):
        raise ValueError("Invalid fixed engineering pocket ordinals")
    pocket = tuple(context.protein_token_indices[i] for i in ordinals)
    return DBMolContactLoss(
        pocket_token_indices=pocket,
        cutoff_angstrom=config["contact_cutoff_angstrom"],
        log_epsilon=config["log_epsilon"],
    ), pocket


def _validate_prepared_config(
    config,
    previous,
    *,
    allow_update_count_change=False,
    allow_mixing_values_change=False,
    allow_initialization_change=False,
):
    """Bind a reused context to either precision-only or longer-budget replay."""
    current_shared, previous_shared = deepcopy(config), deepcopy(previous)
    if allow_initialization_change:
        if allow_mixing_values_change:
            raise ValueError(
                "Initialization replacement and mixture-only policies are exclusive"
            )
        _case_specs(config)
        _case_specs(previous)
        for shared in (current_shared, previous_shared):
            shared.pop("initialization", None)
            shared.pop("mixing_values", None)
    if allow_mixing_values_change:
        if not allow_update_count_change:
            raise ValueError("Mixture replay requires explicit longer-budget policy")
        current_shared.pop("mixing_values")
        previous_shared.pop("mixing_values")
    if allow_update_count_change:
        current_steps = current_shared["checks"].pop("steps")
        previous_steps = previous_shared["checks"].pop("steps")
        if (
            type(current_steps) is not int
            or type(previous_steps) is not int
            or not 0 < previous_steps < current_steps
        ):
            raise ValueError(
                "Update-count replay requires a strictly longer integer budget"
            )
        if current_shared != previous_shared:
            raise ValueError(
                "Update-count replay changed an unapproved configuration field"
            )
    else:
        if not allow_initialization_change:
            current_shared.pop("matmul_precision", None)
            previous_shared.pop("matmul_precision", None)
        if current_shared != previous_shared:
            raise ValueError("Matched replay may change only matmul_precision")


def _hard_graph_from_fixture(fixture):
    """Recover the sealed hard labels without regenerating a native conformer."""
    from mosaic.small_molecule.state import BOND_STATES, ELEMENTS, graph_from_hard

    elements = [ELEMENTS.index(symbol) for symbol in fixture["element_order"]]
    assignments = {
        (i, j): BOND_STATES.index(category)
        for i, j, category in fixture["bond_assignments"]
    }
    bonds = [
        assignments.get((i, j), BOND_STATES.index("NO_BOND"))
        for i in range(len(elements))
        for j in range(i + 1, len(elements))
    ]
    return graph_from_hard(elements, bonds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prepared-study",
        type=Path,
        help="Reuse the context directory of an authenticated previous study",
    )
    parser.add_argument("--prepared-summary-sha256")
    parser.add_argument(
        "--fixture-override",
        type=Path,
        help="Allocated preprocessing output; content hash is retained in the study",
    )
    parser.add_argument(
        "--allow-update-count-change",
        action="store_true",
        help="Reuse a passed study's exact context for a longer otherwise identical run",
    )
    parser.add_argument(
        "--allow-mixing-values-change",
        action="store_true",
        help="Explicitly authorize different relaxation strengths for longer-budget replay",
    )
    parser.add_argument(
        "--allow-initialization-change",
        action="store_true",
        help="Explicitly permit a different graph initialization with identical frozen context",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs/protenix_relaxed_gradients.json",
    )
    args = parser.parse_args(argv)
    if (args.prepared_study is None) != (args.prepared_summary_sha256 is None):
        parser.error(
            "--prepared-study and --prepared-summary-sha256 are required together"
        )
    if args.allow_update_count_change and args.prepared_study is None:
        parser.error("--allow-update-count-change requires --prepared-study")
    if args.allow_initialization_change and args.prepared_study is None:
        parser.error("--allow-initialization-change requires --prepared-study")
    if args.allow_mixing_values_change and not args.allow_update_count_change:
        parser.error(
            "--allow-mixing-values-change requires --allow-update-count-change"
        )
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical work requires an allocated DTU LSF job")

    # Set this before importing model code or tracing any forward: changing it
    # after compilation would not define a reproducible numerical comparison.
    config = json.loads(args.config.read_text(encoding="utf-8"))
    precision = config.get("matmul_precision", "default")
    if precision not in ("default", "highest"):
        raise ValueError("matmul_precision must be 'default' or 'highest'")
    import jax

    jax.config.update("jax_default_matmul_precision", precision)
    from mosaic.small_molecule.gradient_checks import (
        array_digest,
        check_graph_gradients,
    )
    from mosaic.small_molecule.io import (
        file_sha256,
        load_context,
        save_context,
        save_graph,
        write_json,
    )
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.provenance import environment_identity, source_identity

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Requires the job's allocated GPU")
    root = Path(__file__).resolve().parents[1]
    fixture_path = args.fixture_override or root / config["fixture"]
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "fixture.json").write_bytes(fixture_path.read_bytes())
    report = {
        "status": "running",
        "job_id": os.environ["LSB_JOBID"],
        "config": config,
        "config_sha256": file_sha256(args.config),
        "fixture_sha256": file_sha256(fixture_path),
        "fixture": fixture,
        "fixture_file": "fixture.json",
        "numerical_profile": {
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "jax_enable_x64": jax.config.jax_enable_x64,
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        },
        "prepared_context_reused": args.prepared_study is not None,
        "prepared_summary_path": (
            str((args.prepared_study / "summary.json").resolve())
            if args.prepared_study is not None
            else None
        ),
        "prepared_summary_sha256": args.prepared_summary_sha256,
        "prepared_configuration_policy": (
            "initialization_and_update_count_only"
            if args.allow_initialization_change and args.allow_update_count_change
            else "initialization_only"
            if args.allow_initialization_change
            else "update_count_and_mixing_only"
            if args.allow_mixing_values_change
            else "update_count_only"
            if args.allow_update_count_change
            else "precision_only"
        )
        if args.prepared_study is not None
        else "fresh_context",
        "cases": [],
    }
    try:
        model = FrozenProtenixTiny.load(cache_dir=args.cache_dir)
        if args.prepared_study is not None:
            summary_path = args.prepared_study / "summary.json"
            summary_sha256 = file_sha256(summary_path)
            if summary_sha256 != args.prepared_summary_sha256.lower():
                raise ValueError("Previous prepared-study summary identity mismatch")
            # Only trust context file hashes after authenticating the receipt.
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                args.allow_update_count_change or args.allow_initialization_change
            ) and previous["status"] != "passed":
                raise ValueError("Prepared study must have passed its checks")
            _validate_prepared_config(
                config,
                previous["config"],
                allow_update_count_change=args.allow_update_count_change,
                allow_mixing_values_change=args.allow_mixing_values_change,
                allow_initialization_change=args.allow_initialization_change,
            )
            report["previous_requested_updates"] = previous["config"]["checks"]["steps"]
            if previous["fixture_sha256"] != report["fixture_sha256"]:
                raise ValueError("Previous prepared-study fixture identity mismatch")
            if previous["model"] != model.identities:
                raise ValueError("Previous prepared-study model identity mismatch")
            context = load_context(
                args.prepared_study / "context", previous["context_files"]
            )
            hard = _hard_graph_from_fixture(fixture)
            if context.n_atoms != hard.n_atoms:
                raise ValueError(
                    "Prepared context and sealed hard fixture differ in size"
                )
            report["prepared_summary_sha256"] = summary_sha256
        else:
            from mosaic.small_molecule.features import prepare_context
            from mosaic.structure_prediction import TargetChain

            context, hard = prepare_context(
                fixture["smiles"],
                protein_chains=(
                    TargetChain(fixture["protein_sequence"], use_msa=False),
                ),
                cache_dir=args.cache_dir,
                input_name="relaxed-gradient-contact",
            )
        from small_molecule_stage0 import check_fixture

        check_fixture(hard, fixture)
        context_files = save_context(args.output / "context", context)
        hard_graph_sha256 = save_graph(args.output / "hard_graph.npz", hard)
        objective, pocket = _contact_objective(config, context, fixture)
        loss = model.build_loss(context=context, loss=objective)
        before_model = array_digest(model)
        before_context = array_digest(context)
        if args.prepared_study is not None and (
            before_model != previous["model_digest_before"]
            or before_context != previous["context_digest_before"]
        ):
            raise ValueError("Matched study changed frozen model or context bytes")
        report.update(
            model=model.identities,
            source=source_identity(root),
            environment=environment_identity(),
            context_files=context_files,
            hard_graph_file="hard_graph.npz",
            hard_graph_sha256=hard_graph_sha256,
            pocket_token_indices=pocket,
            objective_name=config.get("objective", "dbmol_pocket"),
            protein_token_indices=context.protein_token_indices,
            ligand_token_indices=context.ligand_token_indices,
            model_digest_before=before_model,
            context_digest_before=before_context,
            key_data=jax.random.key_data(jax.random.key(config["seed"])),
        )
        write_json(args.output / "summary.json", report)
        if "reference_smiles" in fixture:
            from mosaic.small_molecule.gradient_checks import _value
            from mosaic.small_molecule.io import hard_readout

            reference = _hard_graph_from_fixture(
                {**fixture, "bond_assignments": fixture["reference_bond_assignments"]}
            )
            reference_readout = hard_readout(reference)
            if not reference_readout["sanitized"]:
                raise ValueError("Known reference graph does not sanitize")
            from rdkit import Chem

            if reference_readout["canonical_smiles"] != Chem.MolToSmiles(
                Chem.MolFromSmiles(fixture["reference_smiles"])
            ):
                raise ValueError("Reference labels differ from the known ligand")
            reference_value = _value(
                loss,
                reference,
                jax.random.key(config["seed"]),
                args.output / "reference_primal.npz",
                compact=True,
            )
            report["reference"] = {
                "name": fixture["reference_name"],
                "declared_smiles": fixture["reference_smiles"],
                "readout": reference_readout,
                "primal_loss": reference_value,
                "graph_sha256": save_graph(
                    args.output / "reference_graph.npz", reference
                ),
                "scope": "same frozen geometry-off context, occupancy, RNG and objective; no binding free energy or pose comparison; native model observes bond existence only",
            }
            write_json(args.output / "summary.json", report)
        for index, case in enumerate(_case_specs(config)):
            result = check_graph_gradients(
                loss,
                _initial_graph(hard, case, jax.random.key(config["seed"])),
                key=jax.random.key(config["seed"]),
                output_directory=args.output / f"case-{index}",
                **config["checks"],
            )
            report["cases"].append({**case, **result})
            write_json(args.output / "summary.json", report)
        report.update(
            model_digest_after=array_digest(model),
            context_digest_after=array_digest(context),
        )
        if (
            report["model_digest_after"] != before_model
            or report["context_digest_after"] != before_context
        ):
            raise AssertionError("Frozen weights or prepared context changed")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "summary.json", report)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
