"""Allocated fresh 25-slot three-arm validation/optimization, preserving native gates."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "run"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cpu-certificate", type=Path, required=True)
    parser.add_argument("--cpu-certificate-sha256", required=True)
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--ready-sha256")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--context-sha256")
    parser.add_argument("--worker-python", default="/work3/s234818/latentmol/env/boltz311/bin/python")
    parser.add_argument("--cache", type=Path, default=Path("/work3/s234818/mosaic/cache/mosaic/protenix"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pocket", type=Path)
    parser.add_argument("--pocket-sha256")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("All numerical work requires allocated DTU HPC")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from campaign25_contract import config25, sha, verify_cpu_certificate, verify_readiness, verify_frozen_response, MODEL_DIGESTS, REFERENCE_SHA

    config = config25(json.loads(args.config.read_text()))
    verify_cpu_certificate(args.cpu_certificate, args.cpu_certificate_sha256, args.manifest)
    if sha(args.reference) != REFERENCE_SHA:
        raise ValueError("Frozen empirical reference changed")
    arm = config["arm"]
    chemistry = config["chemistry_enabled"]
    pocket = None
    if config.get("pocket") is not None:
        from mosaic.small_molecule.pocket import validate_definition
        if args.pocket is None or sha(args.pocket) != args.pocket_sha256:
            raise ValueError("Exact allocated JQ1 pocket definition required")
        pocket = validate_definition(json.loads(args.pocket.read_text()), json.loads(Path(config["fixture"]).read_text()))
    elif args.pocket is not None or args.pocket_sha256 is not None:
        raise ValueError("Historical campaign cannot silently acquire a pocket")
    certified = None
    if args.mode == "run":
        if args.ready is None or args.ready_sha256 is None:
            raise ValueError("Run requires an exact passed readiness assessment")
        certified = verify_readiness(args.ready, args.ready_sha256, args.config, args.manifest)
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec, FailureGuidanceConfig, prepare_failure_guidance, failure_aware_repair_loss
    from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
    from mosaic.small_molecule.chemistry.initialization import initialize_graph
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
    from mosaic.small_molecule.chemistry.trajectory import run_chemical_updates, evaluate_combined, check_combined_directions
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    from mosaic.small_molecule.io import write_json, save_context, load_context
    from mosaic.small_molecule.gradient_checks import array_digest, check_graph_gradients
    from mosaic.small_molecule.state import graph_from_hard, RelaxedGraph
    from mosaic.small_molecule.provenance import source_identity, environment_identity

    jax.config.update("jax_default_matmul_precision", "highest")
    expected_device = "gpu" if arm == "protenix" else "cpu"
    if jax.config.jax_enable_x64 or any(d.platform != expected_device for d in jax.devices()):
        raise RuntimeError("Wrong declared FP32 JAX backend")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running", "mode": args.mode, "job_id": os.environ["LSB_JOBID"],
        "config": config, "config_sha256": sha(args.config),
        "source_manifest_sha256": args.manifest, "source": source_identity(Path(__file__).resolve().parents[1]),
        "environment": environment_identity(), "chemistry_enabled": chemistry,
        "composition_guidance": "disabled in all three arms; coefficient zero",
    }
    if pocket is not None:
        import shutil
        shutil.copyfile(args.pocket, args.output / "pocket_definition.json")
        report.update(pocket_definition_sha256=args.pocket_sha256, diffusion_gradient=False)
        if certified and certified.get("pocket_definition_sha256") != args.pocket_sha256:
            raise ValueError("Pocket differs from passed readiness")
    worker = model = context = None
    started = time.perf_counter()
    try:
        hard = graph_from_hard([0] * 25, [int(j == i + 1) for i in range(25) for j in range(i + 1, 25)])
        key = jax.random.key(17)
        if arm == "protenix":
            from mosaic.small_molecule.features import prepare_context, inject_graph, geometry_off_native_features
            from mosaic.small_molecule.protenix import FrozenProtenixTiny, _native_trunk
            from mosaic.small_molecule.losses import NativeContactLoss

            model = FrozenProtenixTiny.load(cache_dir=args.cache)
            if args.mode == "validate":
                fixture = json.loads(Path(config["fixture"]).read_text())
                identities = None if pocket is None else {"pocket_definition_sha256": args.pocket_sha256, "protein_sequence": fixture["protein_sequence"]}
                context, paired = prepare_context(fixture["smiles"], protein_chains=(fixture["protein_sequence"],), cache_dir=args.cache, input_name="brd4-25", identities=identities)
                np.testing.assert_array_equal(paired.X, hard.X)
                np.testing.assert_array_equal(paired.E_upper, hard.E_upper)
            else:
                context = load_context(args.ready.parent / "context", certified["context_files"])
            if context.n_atoms != 25:
                raise ValueError("Protenix native preprocessing changed the ligand count")
            report["context_files"] = save_context(args.output / "context", context)
            report["model_digest"] = array_digest(model)
            report["context_digest"] = array_digest(context)
            # Independent same-geometry-off native hard input; not the distinct
            # conformer/geometry-on prediction, which is outside this objective.
            features = inject_graph(context, hard)
            for field in ("ref_element", "token_bonds"):
                np.testing.assert_array_equal(features[field], context.base_features[field])
            if pocket is None:
                _, trunk = _native_trunk(model.model, geometry_off_native_features(context), key)
                native = np.asarray(model.model.distogram_head(trunk.z))
            else:
                reference_features = geometry_off_native_features(context)
                if set(reference_features) != set(features):
                    raise ValueError("Hard/reference native input fields differ")
                for name in reference_features:
                    np.testing.assert_array_equal(features[name], reference_features[name])
                native = np.asarray(model.forward_hard_geometry_off(context=context, key=key).distogram_logits)
            attached = np.asarray(model.forward_trunk(context=context, graph=hard, key=key).distogram_logits)
            np.testing.assert_allclose(attached, native, atol=2e-5, rtol=0)
            pocket_tokens = None
            if pocket is not None:
                from mosaic.small_molecule.pocket import map_protenix_pocket
                pocket_tokens = map_protenix_pocket(context, pocket)
                report.update(pocket_validation_status="passed", native_pocket_token_indices=list(pocket_tokens))
            loss = model.build_loss(context=context, loss=NativeContactLoss(8.0, pocket_tokens))
            native_name = "contact"
        else:
            from mosaic.small_molecule.boltz2.bridge import AffinityWorker, affinity_loss
            if args.prepared is None or sha(args.prepared / "context.pt") != args.context_sha256:
                raise ValueError("Fresh 25-slot Boltz context bytes differ")
            if pocket is not None:
                prepared = json.loads((args.prepared / "preparation.json").read_text())
                if prepared.get("pocket_definition_sha256") != args.pocket_sha256 or prepared.get("pocket_pose_audit", {}).get("status") != "passed":
                    raise ValueError("Boltz prepared pose lacks matching passed JQ1 pocket audit")
                report.update(pocket_validation_status="passed", pocket_pose_audit=prepared["pocket_pose_audit"])
            worker = AffinityWorker([
                args.worker_python, "-u", "-m", "mosaic.small_molecule.boltz2.worker",
                "--context", str(args.prepared / "context.pt"), "--context-sha256", args.context_sha256,
                "--checkpoint", "/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt",
            ], stderr=args.output / "worker.log")
            if worker.identity.get("n_atoms") != 25:
                raise ValueError("Native worker does not expose 25 occupied slots")
            if pocket is not None and (worker.identity.get("pocket_definition_sha256") != args.pocket_sha256 or worker.identity.get("diffusion_gradient") is not False):
                raise ValueError("Worker lost pocket or detached pose binding")
            report.update(native_identity=worker.identity, model_digest=worker.identity["model_digest"], context_digest=worker.identity["context_digest"], context_file_sha256=args.context_sha256)
            loss, native_name = affinity_loss(worker), "affinity"
            native = worker.request("native_reference")["values"]
            attached, _, _ = worker.evaluate(np.asarray(hard.X), np.asarray(hard.E_upper), gradient=False)
            np.testing.assert_allclose(attached, native, atol=2e-5, rtol=0)
        model_key = "protenix" if arm == "protenix" else "boltz2"
        if report["model_digest"] != MODEL_DIGESTS[model_key]:
            raise ValueError("Frozen pretrained weights changed")
        if certified and report["context_digest"] != certified["context_digest"]:
            raise ValueError("Context differs from passed native readiness")
        report["native_hard_parity"] = {"status": "passed", "n_atoms": 25, "atol": 2e-5}
        spec = mosaic_spec(25)
        initial = GraphLogits.from_graph(initialize_graph(hard, config["initialization"]))
        np.savez_compressed(args.output / "initialization.npz", X=np.asarray(initial.graph().X), E_upper=np.asarray(initial.graph().E_upper), atom_logits=np.asarray(initial.atoms), bond_logits=np.asarray(initial.bonds), key_data=np.asarray(jax.random.key_data(key)))
        report["initialization_sha256"] = sha(args.output / "initialization.npz")
        if certified and report["initialization_sha256"] != certified["initialization_sha256"]:
            # NPZ containers may include archive timestamps; bind exact arrays too
            # rather than silently use this as an equality acceptance substitute.
            with np.load(args.ready.parent / "initialization.npz") as previous, np.load(args.output / "initialization.npz") as current:
                for field in previous.files:
                    np.testing.assert_array_equal(previous[field], current[field])
            report["initialization_container_changed_arrays_equal"] = True

        def raw_check(graph, directory):
            if arm == "protenix":
                return check_graph_gradients(loss, graph, key=key, output_directory=directory, steps=1, initial_step_size=0.03, max_backtracks=10, artifact_policy="compact", acceptance_policy="best_primal")
            from mosaic.small_molecule.boltz2.checks import check_affinity_gradients
            return check_affinity_gradients(loss, graph, key, directory)

        report["native_gradient_checks"] = raw_check(initial.graph(), args.output / "native_gradient_checks")
        search, progress = ProbabilityRepairConfig(**config["search"]), ProgressSearchConfig(**config["progress_search"])
        def prepare(logits, update):
            if not chemistry:
                return NativeOnlyPreparation(spec.fingerprint(), update)
            return prepare_failure_guidance(logits.decoded(), spec, update=update, search=search, progress_search=progress)

        schedule = RepairWeightSchedule(**config["repair_schedule"]) if chemistry else None
        options = {
            "loss": FailureGuidanceConfig(**config["loss"]), "repair_weight": 0.001 if chemistry else 0.0,
            "element_js_weight": 0.0, "parity_atol": 2e-5, "native_name": native_name,
            "objective_mode": "native_plus_chemistry" if chemistry else "native_only",
            "objective_identity": {"model_digest": report["model_digest"], "context_digest": report["context_digest"], "native_name": native_name, "coefficient": 1.0, "n_atoms": 25, "scope": "whole-target contact, geometry-off" if arm == "protenix" else "fixed predicted pose, zero recycles, no MW correction"},
        }
        if pocket is not None:
            options["objective_identity"].update(pocket_definition_sha256=args.pocket_sha256,
                diffusion_gradient=False, scope="JQ1 pocket-restricted contact, geometry-off" if arm == "protenix" else "JQ1-constrained frozen native pose, zero recycles, no MW correction")
        reference = json.loads(args.reference.read_text())["element_counts"]
        common = dict(prepare=prepare, spec=spec, reference=reference, key=key, step_size=0.03, max_backtracks=10)
        if args.mode == "validate":
            pilots = [("early", initial, options)]
            if chemistry:
                graph = initial.graph()
                x = graph.X.at[1, 0].set(0.5 / 6).at[1, 4].set(0.5 + 0.5 / 6)
                probe = GraphLogits.from_graph(RelaxedGraph(x, graph.E_upper))
                fixed = prepare(probe, 0)
                chemical, derivative = eqx.filter_value_and_grad(lambda z: failure_aware_repair_loss(z.decoded(), fixed, spec, options["loss"])[0])(probe)
                norm = float(jnp.sqrt(jnp.sum(derivative.atoms**2) + jnp.sum(derivative.bonds**2)))
                if not float(chemical) > 0 or not np.isfinite(norm) or norm <= 1e-10:
                    raise AssertionError("Invalid assay lacks nonzero finite chemistry signal")
                report["chemistry_assay"] = {"status": "passed", "loss": float(chemical), "gradient_norm": norm}
                # Both low and high coefficient FD gates use the invalid assay.
                for label, weight in (("low", 0.001), ("high", 50.0)):
                    opts = {**options, "repair_weight": weight}
                    evaluated = evaluate_combined(loss, probe, fixed, spec, reference, key=key, config=opts)
                    check_combined_directions(loss, evaluated, fixed, spec, reference, key=key, config=opts, directory=args.output / (label + "_coupled_checks"))
                pilots.append(("high", probe, {**options, "repair_weight": 50.0}))
            else:
                report["chemistry_assay"] = {"status": "not_applicable", "version": "scorer-only-v1", "reason": "chemical guidance is absent from this arm, not a passed chemical gate"}
            means, sizes = [], []
            for label, state, opts in pilots:
                result = run_chemical_updates(loss, state, directory=args.output / (label + "_pilot"), steps=10, config=opts, **common)
                if result["status"] != "passed" or len(result["updates"]) != 10:
                    raise AssertionError("Disposable native pilot incomplete")
                means.append(float(np.mean([u["elapsed_seconds"] for u in result["updates"]])))
                for i in range(1, 11):
                    pilot_root = args.output / (label + "_pilot")
                    sizes.append(sum((pilot_root / folder / name).stat().st_size for folder, name in [("snapshots", f"step-{i:04d}.npz"), ("references", f"step-{i:04d}.json"), ("trials", f"step-{i-1:04d}.npz")]))
            projected = 1.5 * max(means) * 10000 + 1800
            report["runtime_gate"] = {"status": "passed" if projected < 82800 else "failed", "mean_update_seconds": means, "conservative_projected_seconds": projected, "allocation_seconds": 82800, "scope": "empirical pilots, not a worst-case guarantee"}
            report["storage_projection"] = {"maximum_pilot_update_bytes": max(sizes), "projected_bytes": int(2 * max(sizes) * 10000 + 1024**3), "policy": "twice largest saved pilot update plus 1 GiB for metadata/context; fresh campaign-wide quota gate before submission"}
            if projected >= 82800:
                raise RuntimeError("10,000-update allocation-fit gate failed")
        else:
            result = run_chemical_updates(loss, initial, directory=args.output / "case-0", steps=10000, config=options, schedule=schedule, **common)
            if result["status"] != "passed" or len(result["updates"]) != 10000 or [r["step"] for r in result["states"]] != list(range(10001)):
                raise ValueError("Incomplete declared 10,000-update budget")
            report["cases"] = [{"label": config["label"], "initialization": config["initialization"], **result}]
            with np.load(args.output / "case-0" / result["states"][-1]["snapshot"]) as saved:
                terminal = RelaxedGraph(jnp.asarray(saved["X"]), jnp.asarray(saved["E_upper"]))
            report["terminal_gradient_checks"] = raw_check(terminal, args.output / "terminal_gradient_checks")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if worker is not None:
                report["frozen_identity_after"] = worker.request("verify_identity")
                verify_frozen_response(report["frozen_identity_after"], report["model_digest"], report["context_digest"])
            elif model is not None and context is not None:
                report["frozen_identity_after"] = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
                if report["frozen_identity_after"] != {"model_digest": report["model_digest"], "context_digest": report["context_digest"]}:
                    raise ValueError("Protenix frozen identity changed")
        except Exception as error:
            report.update(status="failed", identity_error=str(error))
        finally:
            if worker is not None:
                worker.close()
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", report)
    if report["status"] != "passed":
        raise RuntimeError("Final identity gate failed")
    if args.mode == "validate":
        proofs = [{"path": str(p.resolve()), "sha256": sha(p)} for p in sorted(args.output.rglob("*")) if p.is_file() and p.suffix in (".json", ".npz", ".pt", ".eqx")]
        assessment = {k: report[k] for k in ("status", "config_sha256", "source_manifest_sha256", "model_digest", "context_digest", "initialization_sha256", "runtime_gate", "storage_projection")}
        assessment.update(arm=arm, n_atoms=25, proofs=proofs)
        if pocket is not None:
            assessment.update(pocket_definition_sha256=args.pocket_sha256,
                pocket_validation_status=report["pocket_validation_status"], diffusion_gradient=False)
        if "context_files" in report:
            assessment["context_files"] = report["context_files"]
        write_json(args.output / "assessment.json", assessment)


if __name__ == "__main__":
    main()
