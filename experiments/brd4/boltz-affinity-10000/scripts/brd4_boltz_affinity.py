"""Allocated Boltz affinity readiness or a fresh Mosaic delayed-chemistry arm."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "run"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--context-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--assessment", type=Path)
    parser.add_argument("--assessment-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("All numerical work requires allocated DTU HPC")
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.boltz2.bridge import AffinityWorker, affinity_loss
    from mosaic.small_molecule.boltz2.checks import check_affinity_gradients
    from mosaic.small_molecule.chemistry import (
        GraphLogits,
        FailureGuidanceConfig,
        mosaic_spec,
        prepare_failure_guidance,
        failure_aware_repair_loss,
    )
    from mosaic.small_molecule.chemistry._reference.probability_repair import (
        ProbabilityRepairConfig,
    )
    from mosaic.small_molecule.chemistry._reference.repair_progress import (
        ProgressSearchConfig,
    )
    from mosaic.small_molecule.chemistry.initialization import initialize_graph
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
    from mosaic.small_molecule.chemistry.trajectory import (
        evaluate_combined,
        check_combined_directions,
        run_chemical_updates,
    )
    from mosaic.small_molecule.io import file_sha256, write_json
    from mosaic.small_molecule.state import (
        RelaxedGraph,
        graph_from_hard,
        mix_with_uniform,
    )
    from mosaic.small_molecule.provenance import source_identity, environment_identity

    if jax.config.jax_enable_x64 or any(d.platform != "cpu" for d in jax.devices()):
        raise RuntimeError(
            "JAX chemistry must use FP32 CPU; isolated native worker owns allocated GPU"
        )
    config = json.loads(args.config.read_text())
    schedule = RepairWeightSchedule(**config["repair_schedule"])
    if (
        config["version"] != "brd4-boltz-affinity-chemistry-v1"
        or config["n_atoms"] != 6
        or config["steps"] != 10000
        or schedule.record() != RepairWeightSchedule(0.001, 50.0, 5000, 10000).record()
    ):
        raise ValueError("Changed user-authorized six-slot/10000-update/schedule scope")
    if (
        file_sha256(args.reference) != args.reference_sha256
        or file_sha256(args.prepared / "context.pt") != args.context_sha256
    ):
        raise ValueError("Prepared context/composition reference identity differs")
    reference = json.loads(args.reference.read_text())
    spec = mosaic_spec(6)
    if (
        reference["status"] != "passed"
        or reference["split"] != "train"
        or tuple(reference["atomic_numbers"]) != spec.atomic_numbers[1:]
    ):
        raise ValueError("Composition reference schema/population differs")
    assessment = None
    if args.mode == "run":
        if (
            not args.assessment
            or file_sha256(args.assessment) != args.assessment_sha256
        ):
            raise ValueError("Native readiness assessment identity differs")
        assessment = json.loads(args.assessment.read_text())
        if (
            assessment["status"] != "passed"
            or assessment["context_sha256"] != args.context_sha256
            or assessment["reference_sha256"] != args.reference_sha256
            or assessment["configs"].get(args.config.name) != file_sha256(args.config)
        ):
            raise ValueError(
                "Native readiness did not certify this configuration/context/reference"
            )
        for proof in assessment["proofs"]:
            if file_sha256(Path(proof["path"])) != proof["sha256"]:
                raise ValueError("Native readiness proof bytes changed")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "job_id": os.environ["LSB_JOBID"],
        "config": config,
        "config_sha256": file_sha256(args.config),
        "mode": args.mode,
        "context_sha256": args.context_sha256,
        "reference_sha256": args.reference_sha256,
        "source": source_identity(_repository_root(__file__)),
        "environment": environment_identity(),
        "scope": "quantitative native log10(IC50/uM), lower better; fixed pose and neutral occupancy; no VAE/decoder, no contact/confidence objective",
    }
    started = time.perf_counter()
    worker = None
    try:
        command = [
            args.worker_python,
            "-u",
            "-m",
            "mosaic.small_molecule.boltz2.worker",
            "--context",
            str(args.prepared / "context.pt"),
            "--context-sha256",
            args.context_sha256,
            "--checkpoint",
            args.checkpoint,
        ]
        worker = AffinityWorker(command, stderr=args.output / "worker.log")
        report["native_identity"] = worker.identity
        if assessment and worker.identity["model_digest"] != assessment["model_digest"]:
            raise ValueError("Model tensor identity differs from passed readiness")
        loss = affinity_loss(worker)
        hard = graph_from_hard(
            [0] * 6, [int(j == i + 1) for i in range(6) for j in range(i + 1, 6)]
        )
        native = worker.request("native_reference")["values"]
        attached, _, _ = worker.evaluate(
            np.asarray(hard.X), np.asarray(hard.E_upper), gradient=False
        )
        if not np.allclose(attached, native, atol=2e-5, rtol=0):
            raise AssertionError(
                "Independent native integer hard-seed forward parity failed"
            )
        report["native_hard_parity"] = {
            "status": "passed",
            "native": native,
            "attached": attached.tolist(),
            "atol": 2e-5,
            "scope": "same six-slot seed and pose; not freshly prepared changed chemistry",
        }
        key = jax.random.key(config["seed"])
        graph = initialize_graph(hard, config["initialization"])
        initial = GraphLogits.from_graph(graph)
        np.savez_compressed(
            args.output / "initialization.npz",
            X=np.asarray(graph.X),
            E_upper=np.asarray(graph.E_upper),
            atom_logits=np.asarray(initial.atoms),
            bond_logits=np.asarray(initial.bonds),
            key_data=np.asarray(jax.random.key_data(key)),
        )
        report["initialization_sha256"] = file_sha256(
            args.output / "initialization.npz"
        )
        report["native_gradient_checks"] = check_affinity_gradients(
            loss, initial.graph(), key, args.output / "native_gradient_checks"
        )
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

        options = {
            "loss": FailureGuidanceConfig(**config["loss"]),
            "repair_weight": schedule.initial,
            "element_js_weight": config["element_js_weight"],
            "parity_atol": 2e-5,
            "native_name": "affinity",
            "objective_identity": {
                **worker.identity,
                "context_sha256": args.context_sha256,
                "optimized_output": "affinity_pred_value",
                "coefficient": 1.0,
            },
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
        if args.mode == "validate":
            probe_graph = mix_with_uniform(hard, 0.5)
            x = probe_graph.X.at[1, 0].set(0.5 / 6).at[1, 4].set(0.5 + 0.5 / 6)
            probe = GraphLogits.from_graph(RelaxedGraph(x, probe_graph.E_upper))
            prepared = prepare(probe, 0)
            chemical, derivative = eqx.filter_value_and_grad(
                lambda v: failure_aware_repair_loss(
                    v.decoded(), prepared, spec, options["loss"]
                )[0]
            )(probe)
            norm = float(
                jnp.sqrt(jnp.sum(derivative.atoms**2) + jnp.sum(derivative.bonds**2))
            )
            if float(chemical) <= 0 or not np.isfinite(norm) or norm <= 1e-10:
                raise AssertionError(
                    "Invalid assay supplies no finite nonzero chemistry derivative"
                )
            report["chemistry_assay"] = {
                "status": "passed",
                "loss": float(chemical),
                "gradient_norm": norm,
            }
            means = []
            for label, state, weight in (
                ("early", initial, schedule.initial),
                ("high", probe, schedule.final),
            ):
                opts = {**options, "repair_weight": weight}
                fixed = prepare(probe, 0)
                evaluated = evaluate_combined(
                    loss,
                    probe,
                    fixed,
                    spec,
                    reference["element_counts"],
                    key=key,
                    config=opts,
                )
                check_combined_directions(
                    loss,
                    evaluated,
                    fixed,
                    spec,
                    reference["element_counts"],
                    key=key,
                    config=opts,
                    directory=args.output / (label + "_coupled_checks"),
                )
                pilot = run_chemical_updates(
                    loss,
                    state,
                    directory=args.output / (label + "_pilot"),
                    steps=config["pilot_steps"],
                    **{**common, "config": opts},
                )
                means.append(
                    float(np.mean([row["elapsed_seconds"] for row in pilot["updates"]]))
                )
            estimate = 1.5 * max(means) * config["steps"] + 1800
            report["runtime_gate"] = {
                "mean_update_seconds": means,
                "conservative_projected_seconds": estimate,
                "allocation_seconds": config["allocation_seconds"],
                "status": "passed"
                if estimate < config["allocation_seconds"]
                else "failed",
                "limitation": "empirical early/invalid-high-weight pilots; not a worst-case bound",
            }
            if estimate >= config["allocation_seconds"]:
                raise RuntimeError("10000-update allocation-fit gate failed")
        else:
            result = run_chemical_updates(
                loss,
                initial,
                directory=args.output / "case-0",
                steps=config["steps"],
                schedule=schedule,
                **common,
            )
            if (
                len(result["states"]) != 10001
                or [s["step"] for s in result["states"]] != list(range(10001))
                or len(result["updates"]) != 10000
            ):
                raise ValueError("Incomplete user-requested update/state budget")
            report["cases"] = [
                {
                    "label": config["label"],
                    "initialization": config["initialization"],
                    **result,
                }
            ]
            final_graph = RelaxedGraph(
                jnp.asarray(
                    np.load(args.output / "case-0" / result["states"][-1]["snapshot"])[
                        "X"
                    ]
                ),
                jnp.asarray(
                    np.load(args.output / "case-0" / result["states"][-1]["snapshot"])[
                        "E_upper"
                    ]
                ),
            )
            report["terminal_gradient_checks"] = check_affinity_gradients(
                loss, final_graph, key, args.output / "terminal_gradient_checks"
            )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if worker is not None:
            try:
                report["frozen_identity_after"] = worker.request("verify_identity")
            except Exception as error:
                report.update(status="failed", identity_error=str(error))
            worker.close()
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", report)
    if report["status"] != "passed":
        raise RuntimeError("Final frozen-identity gate failed")


if __name__ == "__main__":
    main()
