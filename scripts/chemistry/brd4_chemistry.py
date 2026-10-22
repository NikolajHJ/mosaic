"""Allocated native pilots and a fresh configured BRD4 chemistry trajectory."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical work requires an allocated DTU GPU job")
    import jax
    import numpy as np

    config = json.loads(args.config.read_text())
    jax.config.update("jax_default_matmul_precision", config["matmul_precision"])
    if jax.config.jax_enable_x64 or not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Require allocated GPU and historical FP32 profile")
    from mosaic.small_molecule.chemistry import (
        GraphLogits,
        FailureGuidanceConfig,
        mosaic_spec,
        prepare_failure_guidance,
    )
    from mosaic.small_molecule.chemistry import failure_aware_repair_loss
    from mosaic.small_molecule.chemistry._reference.probability_repair import (
        ProbabilityRepairConfig,
    )
    from mosaic.small_molecule.chemistry._reference.repair_progress import (
        ProgressSearchConfig,
    )
    from mosaic.small_molecule.chemistry.trajectory import (
        evaluate_combined,
        check_combined_directions,
        run_chemical_updates,
    )
    from mosaic.small_molecule.chemistry.initialization import initialize_graph
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
    from mosaic.small_molecule.gradient_checks import (
        array_digest,
        check_graph_gradients,
    )
    from mosaic.small_molecule.io import (
        file_sha256,
        load_context,
        save_context,
        write_json,
    )
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.provenance import source_identity, environment_identity
    from mosaic.small_molecule.state import RelaxedGraph, ELEMENTS, mix_with_uniform
    from protenix_relaxed_gradients import _hard_graph_from_fixture

    scheduled = config["version"] == "brd4-contact-chemistry-ramp-v2"
    schedule = RepairWeightSchedule(**config["repair_schedule"]) if scheduled else None
    recipe = config.get("initialization", {"kind": "chain_uniform", "mixing": 0.5})
    if (
        config["version"]
        not in ("brd4-contact-chemistry-logits-v1", "brd4-contact-chemistry-ramp-v2")
        or (not scheduled and config["mixing"] != 0.5)
        or (scheduled and schedule.total_updates != config["steps"])
        or config["n_atoms"] != 6
        or config["progress_search"]["allow_atom_edits"]
    ):
        raise ValueError(
            "Configuration changed the declared six-slot, atom-preserving scope"
        )
    if file_sha256(args.reference) != args.reference_sha256:
        raise ValueError("Composition reference identity differs")
    reference = json.loads(args.reference.read_text())
    spec = mosaic_spec(config["n_atoms"])
    if (
        reference["status"] != "passed"
        or reference["split"] != "train"
        or tuple(reference["atomic_numbers"]) != spec.atomic_numbers[1:]
        or reference["data"] != config["composition_data"]
    ):
        raise ValueError(
            "Composition reference changed its supported training population"
        )
    previous_root = Path(config["prepared_study"])
    if file_sha256(previous_root / "summary.json") != config["prepared_summary_sha256"]:
        raise ValueError("Prepared context summary identity differs")
    previous = json.loads((previous_root / "summary.json").read_text())
    if (
        previous["status"] != "passed"
        or any(
            previous["config"][key] != config[key]
            for key in ("seed", "matmul_precision", "contact_cutoff_angstrom")
        )
        or previous["config"]["objective"] != "native_whole_target"
    ):
        raise ValueError("Prepared context objective/profile differs")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "job_id": os.environ["LSB_JOBID"],
        "config": config,
        "config_sha256": file_sha256(args.config),
        "composition_reference": reference,
        "composition_reference_sha256": args.reference_sha256,
        "parameterization": "categorical_logits_v1",
        "historical_comparison": "exploratory; both objective and optimization geometry changed",
    }
    model = context = None
    started = time.perf_counter()
    try:
        context = load_context(previous_root / "context", previous["context_files"])
        fixture_path = previous_root / previous["fixture_file"]
        if file_sha256(fixture_path) != previous["fixture_sha256"]:
            raise ValueError("Prepared fixture identity differs")
        fixture = json.loads(fixture_path.read_text())
        hard = _hard_graph_from_fixture(fixture)
        if hard.n_atoms != config["n_atoms"] or context.n_atoms != hard.n_atoms:
            raise ValueError("Fixed slot count differs")
        model = FrozenProtenixTiny.load(cache_dir=args.cache_dir)
        if model.identities != previous["model"]:
            raise ValueError("Native model assets/source differ")
        report.update(
            model=model.identities,
            model_digest_before=array_digest(model),
            context_digest_before=array_digest(context),
            context_files=save_context(args.output / "context", context),
            source=source_identity(_repository_root(__file__)),
            environment=environment_identity(),
        )
        if (
            report["model_digest_before"] != config["model_digest"]
            or report["context_digest_before"] != config["context_digest"]
        ):
            raise ValueError("Frozen native model/context identity differs")
        (args.output / "fixture.json").write_bytes(fixture_path.read_bytes())
        (args.output / "configuration.json").write_bytes(args.config.read_bytes())
        (args.output / "composition_reference.json").write_bytes(
            args.reference.read_bytes()
        )
        key = jax.random.key(config["seed"])
        graph = initialize_graph(hard, recipe)
        initial = GraphLogits.from_graph(graph)
        np.savez_compressed(
            args.output / "initialization.npz",
            X=np.asarray(graph.X),
            E_upper=np.asarray(graph.E_upper),
            atom_logits=np.asarray(initial.atoms),
            bond_logits=np.asarray(initial.bonds),
            key_data=np.asarray(jax.random.key_data(key)),
        )
        report["initialization"] = {
            "recipe": recipe,
            "artifact": "initialization.npz",
            "sha256": file_sha256(args.output / "initialization.npz"),
            "native_key_seed": config["seed"],
        }
        if not all(
            np.allclose(
                np.asarray(a),
                np.asarray(b),
                atol=2 * np.finfo(np.asarray(b).dtype).eps,
                rtol=0,
            )
            for a, b in (
                (graph.X, initial.graph().X),
                (graph.E_upper, initial.graph().E_upper),
            )
        ):
            raise ValueError("Logit initialization changed requested probability law")
        loss = model.build_loss(
            context=context,
            loss=NativeContactLoss(cutoff_angstrom=config["contact_cutoff_angstrom"]),
        )
        # This remains a contact-only probe: the native bond-order null is NOT
        # expected to remain null after adding chemistry.
        report["native_contact_checks"] = check_graph_gradients(
            loss,
            initial.graph(),
            key=key,
            output_directory=args.output / "native_contact_checks",
            steps=1,
            initial_step_size=config["initial_step_size"],
            max_backtracks=config["max_backtracks"],
            primal_every_step=True,
            record_readouts=True,
            artifact_policy="compact",
            acceptance_policy="best_primal",
        )
        options = {
            "loss": FailureGuidanceConfig(**config["loss"]),
            "repair_weight": schedule.initial if scheduled else config["repair_weight"],
            "element_js_weight": config["element_js_weight"],
            "parity_atol": config["parity_atol"],
            "objective_identity": {
                "model_digest": config["model_digest"],
                "context_digest": config["context_digest"],
                "seed": config["seed"],
                "contact_cutoff_angstrom": config["contact_cutoff_angstrom"],
                "contact_source_sha256": file_sha256(
                    _repository_root(__file__)
                    / "src/mosaic/small_molecule/losses.py"
                ),
                "precision": config["matmul_precision"],
                "scope": "native whole-target contact; fixed context/key; no chemistry in this identity",
            },
        }
        search, progress = (
            ProbabilityRepairConfig(**config["search"]),
            ProgressSearchConfig(**config["progress_search"]),
        )

        def prepare(logits, update):
            return prepare_failure_guidance(
                logits.decoded(),
                spec,
                update=update,
                search=search,
                progress_search=progress,
            )

        # An internal neutral F with two single neighbors is an intentionally
        # invalid assay, separate from the requested all-carbon initialization.
        # Require actual chemistry signal through the shared logit variables,
        # not just the native contact/JS gradient on an already-valid molecule.
        import equinox as eqx

        probe_dir = args.output / "coupled_chemistry_probe"
        probe_dir.mkdir()
        assay_graph = mix_with_uniform(hard, 0.5)
        x = assay_graph.X.at[1, ELEMENTS.index("C")].set(0.5 / len(ELEMENTS))
        x = x.at[1, ELEMENTS.index("F")].set(0.5 + 0.5 / len(ELEMENTS))
        probe = GraphLogits.from_graph(RelaxedGraph(x, assay_graph.E_upper))
        probe_reference = prepare(probe, 0)
        write_json(probe_dir / "preparation.json", probe_reference.record())
        np.savez_compressed(
            probe_dir / "input.npz",
            X=np.asarray(probe.graph().X),
            E_upper=np.asarray(probe.graph().E_upper),
            atom_logits=np.asarray(probe.atoms),
            bond_logits=np.asarray(probe.bonds),
            key_data=np.asarray(jax.random.key_data(key)),
        )
        probe_value = evaluate_combined(
            loss,
            probe,
            probe_reference,
            spec,
            reference["element_counts"],
            key=key,
            config=options,
        )

        def chemistry_only(value):
            return failure_aware_repair_loss(
                value.decoded(), probe_reference, spec, options["loss"]
            )[0]

        chemistry_gradient = eqx.filter_grad(chemistry_only)(probe)
        norm = float(
            jax.numpy.sqrt(
                jax.numpy.sum(chemistry_gradient.atoms**2)
                + jax.numpy.sum(chemistry_gradient.bonds**2)
            )
        )
        np.savez_compressed(
            probe_dir / "signal.npz",
            X=np.asarray(probe.graph().X),
            E_upper=np.asarray(probe.graph().E_upper),
            atom_logits=np.asarray(probe.atoms),
            bond_logits=np.asarray(probe.bonds),
            chemistry_gradient_atom_logits=np.asarray(chemistry_gradient.atoms),
            chemistry_gradient_bond_logits=np.asarray(chemistry_gradient.bonds),
            contact_primal=probe_value.native.primal,
            chemistry_loss=probe_value.terms["chemistry_loss"],
            key_data=np.asarray(jax.random.key_data(key)),
        )
        if (
            not np.isfinite(norm)
            or norm <= 1e-10
            or probe_value.terms["chemistry_loss"] <= 0
        ):
            raise AssertionError(
                "Invalid native-input assay did not supply finite nonzero chemistry guidance"
            )
        probe_checks = check_combined_directions(
            loss,
            probe_value,
            probe_reference,
            spec,
            reference["element_counts"],
            key=key,
            config=options,
            directory=probe_dir / "derivative_checks",
        )
        report["coupled_chemistry_probe"] = {
            "status": "passed",
            "chemistry_gradient_norm": norm,
            "components": probe_value.terms,
            "preparation_identity": probe_reference.identity,
            "directions": probe_checks,
            "scope": "disposable invalid-input derivative assay; not a requested trajectory update",
        }
        if scheduled:
            final_options = {**options, "repair_weight": schedule.final}
            final_probe = evaluate_combined(
                loss,
                probe,
                probe_reference,
                spec,
                reference["element_counts"],
                key=key,
                config=final_options,
                native=probe_value.native,
            )
            report["high_weight_chemistry_probe"] = {
                "status": "passed",
                "repair_weight": schedule.final,
                "components": final_probe.terms,
                "preparation_identity": probe_reference.identity,
                "directions": check_combined_directions(
                    loss,
                    final_probe,
                    probe_reference,
                    spec,
                    reference["element_counts"],
                    key=key,
                    config=final_options,
                    directory=probe_dir / "high_weight_checks",
                ),
            }

        common = dict(
            prepare=prepare,
            spec=spec,
            reference=reference["element_counts"],
            key=key,
            step_size=config["initial_step_size"],
            max_backtracks=config["max_backtracks"],
            config=options,
        )
        pilot = run_chemical_updates(
            loss,
            initial,
            directory=args.output / "pilot",
            steps=config["pilot_steps"],
            **common,
        )
        average = float(np.mean([u["elapsed_seconds"] for u in pilot["updates"]]))
        if scheduled:
            # Deliberately invalid high-weight assay, not a fictitious late-run
            # state. Measure active repair and actual line-search exposure too.
            late_pilot = run_chemical_updates(
                loss,
                probe,
                directory=args.output / "high_weight_pilot",
                steps=config["pilot_steps"],
                **{**common, "config": final_options},
            )
            late_average = float(
                np.mean([u["elapsed_seconds"] for u in late_pilot["updates"]])
            )
            average = max(average, late_average)
            report["high_weight_pilot"] = {
                "status": late_pilot["status"],
                "repair_weight": schedule.final,
                "steps": config["pilot_steps"],
                "counts": late_pilot["counts"],
                "mean_update_seconds": late_average,
                "runtime_estimate_seconds_per_update": average,
                "disposable": True,
                "limitation": "invalid-input assay; empirical runtime estimate, not a worst-case bound or prediction of late-state chemistry/convergence",
            }
        projected = 1.5 * average * config["steps"] + 1800
        report["pilot"] = {
            "status": pilot["status"],
            "steps": config["pilot_steps"],
            "mean_update_seconds": average,
            "conservative_projected_seconds": projected,
            "counts": pilot["counts"],
            "disposable": True,
        }
        if projected + (time.perf_counter() - started) >= config["allocation_seconds"]:
            raise RuntimeError("Pilot does not support complete-budget allocation fit")
        if (
            array_digest(model) != config["model_digest"]
            or array_digest(context) != config["context_digest"]
        ):
            raise ValueError("Frozen identity changed during native pilot")
        write_json(args.output / "summary.json", report)
        # Pilot updates are disposable. Reset to the exact same immutable
        # initialization/key/reference; do not silently count them as a prefix.
        result = run_chemical_updates(
            loss,
            initial,
            directory=args.output / "case-0",
            steps=config["steps"],
            schedule=schedule,
            **common,
        )
        report["cases"] = [
            {
                "label": config.get("label", "50% chain/uniform + chemistry"),
                "initialization": recipe,
                **result,
            }
        ]
        if len(result["states"]) != config["steps"] + 1 or [
            v["step"] for v in result["states"]
        ] != list(range(config["steps"] + 1)):
            raise ValueError("Incomplete requested update budget")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if model is not None and context is not None:
            report.update(
                model_digest_after=array_digest(model),
                context_digest_after=array_digest(context),
            )
            if (
                "model_digest_before" in report
                and report["model_digest_before"] != report["model_digest_after"]
            ) or (
                "context_digest_before" in report
                and report["context_digest_before"] != report["context_digest_after"]
            ):
                report.update(status="failed", frozen_identity_error=True)
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", report)
    if report["status"] != "passed":
        raise RuntimeError("Final frozen identity check failed")


if __name__ == "__main__":
    main()
