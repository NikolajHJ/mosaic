"""Run the frozen seed-41 screen twice with matched repair policies."""

import argparse
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from atom_exchange_audit import OVERLAY, PACKAGE, SealedChemistry
from score_transfer_audit import PROFILE, binding, bound, read, require, write


VERSION = "mosaic-seed41-atom-exchange-pair-v1"
PARENT_ARM = "annealed-sparse-s41"
PARENT_STEP = 50000
PARENT_JOB = "29880095"
UPDATES = 2000
HARD_STRIDE = 100
ARMS = ("bond_only", "atom_exchange")
EXCHANGE = {"atom_edit_penalty": 1.0, "max_substitutions": 4}


def parent_metadata(plan, plan_path):
    require(plan["version"] == VERSION and plan["updates"] == UPDATES and
            plan["hard_stride"] == HARD_STRIDE and plan["atom_exchange"] == EXCHANGE,
            "Matched comparison budget or treatment changed")
    require(plan["parent_step"] == PARENT_STEP and plan["parent_job"] == PARENT_JOB,
            "Expected exact seed-41 endpoint")
    require(Path(plan["output"]) == plan_path.resolve().parent.parent / "payload",
            "Output escapes the owned run")
    parent = read(bound(plan["parent_plan"]))
    row = parent["arms"][PARENT_ARM]
    require(plan["source_root"] == parent["code_source_root"] and
            plan["routing_source"] == parent["routing_source"] and
            plan["science_source"] == row["science_source"]["path"] and
            plan["queue"] == row["queue"] and plan["gpu_profile"] == row["gpu_profile"],
            "Parent source, routing or GPU profile differs")
    summary = read(bound(plan["parent_summary"]))
    require(summary["job_id"] == PARENT_JOB and summary["applied_updates"] == PARENT_STEP and
            summary["status"] == "completed_exploratory_updates" and summary["full_budget_complete"] and
            summary["frozen_before"] == summary["frozen_after"] == row["frozen_identity"],
            "Parent did not finish with the expected frozen scorer")
    old = row["screen"]
    require(old["init_seed"] == 41 and old["updates"] == PARENT_STEP and row["rate"] == .3 and
            row["model_key"] == 17 and row["composition_weight"] == 0 and not row["margin"] and
            old["binding"] == {"mode": "always"} and
            old["temperature_schedule"]["final"] == .4 and old["chemistry"]["final"] == 30.0 and
            old["logit_noise"]["zero_at_update"] <= PARENT_STEP and
            old["structural_repair"] == {"version": "connected-backbone-repair-v1",
                                         "max_candidates": 6, "max_bond_operations": 1024},
            "Saved endpoint policy differs")
    config = deepcopy(old)
    config.update(updates=UPDATES, temperature=.4,
                  chemistry={"mode": "constant", "coefficient": 30.0})
    config.pop("temperature_schedule")
    config.pop("logit_noise")
    config_path = bound(row["config"])
    science = read(config_path)
    require(science["max_backtracks"] == 1 and science["acceptance_policy"] == "fixed_step" and
            science["progress_search"]["allow_atom_edits"] is False,
            "Original fixed step or atom-preserving fallback differs")
    records = [plan[k] for k in ("parent_plan", "parent_manifest", "parent_summary",
                                "parent_snapshot", "original_screen")]
    records += [row[k] for k in ("config", "pocket", "context_ready")]
    for record in records:
        bound(record)
    require(bound(plan["original_screen"]) == Path(plan["source_root"]) / "protenix_chemistry_screen.py",
            "Run the actual parent screen")
    return row, config, science, records


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run(plan, plan_path, row, config, science, records):
    job = os.environ.get("LSB_JOBID", "")
    require(job.isdigit() and os.environ.get("LSB_QUEUE") == plan["queue"] and
            os.environ.get("XLA_FLAGS") == PROFILE, "Original allocated runtime required")
    routing = module(Path(plan["routing_source"]) / "scripts/cuda_ownership.py", "_seed41_cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = routing.resolve_visibility(os.environ)["selected_cuda_visible_devices"]

    source = Path(plan["source_root"])
    release = plan_path.parent.resolve()
    original = Path(plan["science_source"]) / "scripts"
    adapter = Path(row["gpu_adapter_source"]) / "scripts"
    sys.path[:0] = [str(source), str(source / "src"), str(original), str(adapter)]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(source / "src"), str(original), str(adapter),
                                              os.environ.get("PYTHONPATH", "")])
    overlay_names = OVERLAY | {"preparation"}
    overlay = SealedChemistry(release, overlay_names)
    sys.meta_path.insert(0, overlay)
    try:
        from mosaic.small_molecule.chemistry.atom_exchange import within_common_organic_scope
        from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
    finally:
        sys.meta_path.remove(overlay)
    for name in overlay_names:
        require(Path(sys.modules[f"{PACKAGE}.{name}"].__file__).resolve() == release / f"{name}.py",
                "Chemistry overlay failed to load: " + name)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest
    from atom_exchange_metrics import capture_target, summarize_trajectory
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.io import hard_readout, load_context
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.state import graph_from_hard, validate_graph
    import storage_accounting

    require(not jax.config.x64_enabled and jax.default_backend() == "gpu" and len(jax.devices("gpu")) == 1,
            "Require one allocated FP32 GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    screen = module(bound(plan["original_screen"]), "_seed41_screen")
    accelerator = module(adapter / "full_model_pilot.py", "_seed41_acceleration")
    screen.validate_screen(config)
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    started, next_gate = time.monotonic(), 0.0
    frozen = model = context = None
    result = {"status": "running", "version": VERSION, "job_id": job, "plan": binding(plan_path),
              "parent_job": PARENT_JOB, "parent_step": PARENT_STEP, "source_parent_preserved": True,
              "configuration": config, "learning_rate": .3, "model_key": 17, "noise_ratio": 0.0,
              "common_organic_scope": True, "runtime_profile": PROFILE, "arms": [],
              "contact_scope": "Frozen Protenix Tiny 8A contact, relaxed trajectory and matched hard-graph grid"}

    def tick(step=None):
        nonlocal next_gate
        now = time.monotonic()
        require(now - started < plan["internal_seconds"], "Own elapsed-time limit reached")
        if now >= next_gate:
            capacity = storage_accounting.assess_output(output.parent, plan["output_cap_bytes"], plan["file_cap"])
            write(output / "capacity-latest.json", capacity)
            require(capacity["status"] == "passed" and
                    capacity["own_output"]["actual_bytes"] <= plan["output_cap_bytes"] and
                    capacity["own_output"]["actual_files"] <= plan["file_cap"], "Own output capacity exceeded")
            next_gate = now + 300

    try:
        tick()
        tests = sorted(release.glob("test_*.py"))
        require(bool(tests), "Missing allocated fixture suite")
        code = pytest.main(["-q", "-p", "no:cacheprovider", "--junitxml", str(output / "tests.xml"),
                            *map(str, tests)])
        result["test_exit_code"] = int(code)
        require(code == 0, "Allocated target-preparation fixtures failed")
        write(output / "result.json", result)
        ready = read(row["context_ready"]["path"])
        context = load_context(Path(row["context_ready"]["path"]).parent / "context", ready["context_files"])
        model = FrozenProtenixTiny.load(cache_dir=Path(plan["model_cache"]))
        frozen = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
        require(frozen == row["frozen_identity"], "Frozen model/context differs")
        result["frozen_before"] = frozen
        loss = model.build_loss(context=context, loss=NativeContactLoss(
            8., map_protenix_pocket(context, read(row["pocket"]["path"]))))
        key = jax.random.key(17)
        spec = mosaic_spec(25)
        search = ProbabilityRepairConfig(**science["search"])
        progress = ProgressSearchConfig(**science["progress_search"])
        loss_config = FailureGuidanceConfig(**science["loss"])
        pairs = np.triu_indices(25, 1)
        kind = screen.logits_class(scheduled=True)
        with np.load(bound(plan["parent_snapshot"]), allow_pickle=False) as saved:
            require(int(saved["step"]) == PARENT_STEP and
                    np.array_equal(saved["key_data"], np.asarray(jax.random.key_data(key))) and
                    float(saved["temperature"]) == float(np.float32(.4)), "Endpoint step/key/temperature differs")
            atoms, bonds = (saved[k].copy() for k in ("atom_logits", "bond_logits"))
            require(atoms.shape == (25, 6) and bonds.shape == (300, 5) and
                    atoms.dtype == bonds.dtype == np.float32 and np.isfinite(atoms).all() and
                    np.isfinite(bonds).all(), "Unexpected saved FP32 logits")
            initial = kind(jnp.asarray(atoms), jnp.asarray(bonds), .4)
            replay = initial.graph()
            for block in ("X", "E_upper"):
                observed, expected = np.asarray(getattr(replay, block)), saved[block]
                require(np.array_equal(observed.argmax(-1), expected.argmax(-1)) and
                        np.allclose(observed, expected, atol=1e-7, rtol=1e-6), "Saved graph replay differs")
            result["endpoint_replay"] = {"snapshot": plan["parent_snapshot"],
                                         "source_contact_loss": float(saved["contact_primal"]),
                                         "exact_fp32_logits": True, "same_argmax": True}

        @eqx.filter_jit
        def hard_forward(graph):
            value, aux = loss(graph, key=key)
            finite = jnp.isfinite(value)
            for leaf in jax.tree_util.tree_leaves(aux):
                if eqx.is_array(leaf):
                    finite &= jnp.all(jnp.isfinite(leaf))
            return value, finite

        for name in ARMS:
            tick()
            arm = output / name
            acceleration = arm / "acceleration"
            acceleration.mkdir(parents=True, exist_ok=False)
            study = arm / "study"
            targets = {}
            options = None if name == "bond_only" else EXCHANGE

            def prepare(logits, step):
                prepared = prepare_failure_guidance(
                    logits.decoded(), spec, update=step, search=search, progress_search=progress,
                    structural_repair=config["structural_repair"], common_organic_scope=True,
                    atom_exchange=options)
                require(step not in targets, "Target was prepared twice for one source")
                targets[step] = capture_target(prepared)
                return prepared

            arm_started = time.monotonic()
            try:
                with accelerator.combined_adapters("protenix-random50k", acceleration) as adapters:
                    summary = screen.run_loop(loss, initial, prepare, spec, loss_config, key, config,
                                              study, steps=UPDATES, tick=tick)
            finally:
                write(arm / "targets.json", targets)
            require(summary["status"] == "completed_exploratory_updates" and
                    summary["applied_updates"] == UPDATES, "Incomplete matched update budget")
            optimization_seconds = time.monotonic() - arm_started

            hard_records = []
            for step in range(0, UPDATES + 1, HARD_STRIDE):
                tick()
                path = study / "snapshots" / f"step-{step:05d}.npz"
                with np.load(path, allow_pickle=False) as point:
                    require(int(point["step"]) == step, "Hard score snapshot differs")
                    a, b = (point[k].argmax(-1) for k in ("X", "E_upper"))
                graph = graph_from_hard(a, b)
                validate_graph(graph)
                value, finite = hard_forward(graph)
                score = -float(jax.block_until_ready(value))
                require(bool(finite) and math.isfinite(score), "Nonfinite hard contact forward")
                readout = hard_readout(graph)
                native_valid = screen.valid_connected(readout)
                matrix = np.zeros((25, 25), dtype=np.int64)
                matrix[pairs] = b
                matrix += matrix.T
                scoped_valid = native_valid and within_common_organic_scope(
                    (a + 1, np.zeros(25, dtype=np.int64), matrix), spec)
                hard_records.append({"local_step": step, "source_step": PARENT_STEP + step,
                                     "contact_score": score, "native_valid": native_valid,
                                     "scoped_valid": scoped_valid, "snapshot": binding(path)})
            hard_contacts = {r["local_step"]: r["contact_score"] for r in hard_records}
            metrics = summarize_trajectory(study, targets, spec, expected_updates=UPDATES,
                                           hard_contacts=hard_contacts)
            scoped = [r for r in hard_records if r["scoped_valid"]]
            result["arms"].append({"arm": name, "applied_updates": UPDATES, "atom_exchange": options,
                                   "targets": binding(arm / "targets.json"), "report": binding(study / "report.json"),
                                   "optimization_seconds": optimization_seconds, "acceleration": adapters,
                                   "metrics": metrics, "hard_score_calls": len(hard_records),
                                   "hard_contact_grid": hard_records,
                                   "best_scoped_hard": max(scoped, key=lambda r: r["contact_score"], default=None)})
            write(output / "result.json", result)
        first, second = (arm["hard_contact_grid"][0] for arm in result["arms"])
        require(first["native_valid"] == second["native_valid"] and
                first["scoped_valid"] == second["scoped_valid"] and
                math.isclose(first["contact_score"], second["contact_score"], abs_tol=1e-7, rel_tol=1e-6),
                "Arm starts differ on matched hard replay")
        result["status"] = "completed_matched_seed41_exchange"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            if frozen is not None:
                result["frozen_after"] = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
                require(result["frozen_after"] == frozen, "Frozen model/context changed")
            for record in records:
                bound(record)
            result["inputs_unchanged"] = True
        except BaseException as error:
            result.update(status="failed", preservation_error=repr(error))
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    path = args.plan.resolve()
    plan = read(path)
    row, config, science, records = parent_metadata(plan, path)
    if args.startup_only:
        print(json.dumps({"status": "metadata_startup_passed_no_models", "arms": ARMS,
                          "parent_step": PARENT_STEP, "configuration": config,
                          "atom_exchange": EXCHANGE, "hard_calls_per_arm": UPDATES // HARD_STRIDE + 1}))
    else:
        result = run(plan, path, row, config, science, records)
        print(json.dumps({"status": result["status"],
                          "applied_updates_per_arm": [arm["applied_updates"] for arm in result["arms"]]}))


if __name__ == "__main__":
    main()
