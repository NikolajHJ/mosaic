"""Refine a saved seed-41 hard graph with strict validity and direct contact."""

import argparse
from collections import Counter
import importlib.util
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import sys
import time

from atom_exchange_audit import OVERLAY, PACKAGE, SealedChemistry
from score_transfer_audit import PROFILE, binding, bound, read, require, write


VERSION = "mosaic-seed41-scoped-hard-contact-v1"
PARENT_JOB = "29898292"
PARENT_STEP = 52000
ROUNDS = 24
BATCH = 32
CANDIDATE_CAP = 768
RANK_LIMIT = 192
ACCEPT_MARGIN = 1e-6
REFERENCE = {"saved_valid_endpoint": .18664762377738953,
             "old_seed41_out_of_scope": .20309558510780334,
             "old_bond_only_out_of_scope": .21503520011901855}


def inputs(plan, plan_path):
    require(plan["version"] == VERSION and plan["parent_job"] == PARENT_JOB and
            plan["parent_step"] == PARENT_STEP, "Use the declared seed-41 endpoint")
    require(plan["rounds"] == ROUNDS and plan["candidate_cap"] == CANDIDATE_CAP and
            plan["batch"] == BATCH and plan["accept_margin"] == ACCEPT_MARGIN,
            "Changed bounded hard-search protocol")
    root = plan_path.resolve().parent.parent
    require(Path(plan["output"]) == root / "payload" and plan["queue"] == "gpul40s",
            "Expected new owned L40S payload")
    records = [plan[k] for k in ("source_result", "source_plan", "source_manifest", "source_lsf",
                                "original_plan", "context_ready", "pocket")]
    records += [plan["initial"]["snapshot"], *plan["overlay"]]
    for pin in records:
        bound(pin)
    prior = read(plan["source_result"]["path"])
    require(prior["job_id"] == PARENT_JOB and prior["status"] == "completed_matched_seed41_exchange" and
            prior["frozen_after"] == plan["frozen"] and prior["inputs_unchanged"],
            "Original paired job is not verified complete")
    arm, = [a for a in prior["arms"] if a["arm"] == "atom_exchange"]
    require(arm["hard_contact_grid"][-1] == plan["initial"] and plan["initial"]["source_step"] == PARENT_STEP and
            plan["initial"]["scoped_valid"] and plan["initial"]["contact_score"] == REFERENCE["saved_valid_endpoint"],
            "Require exact scoped-valid atom-exchange endpoint")
    previous_plan = read(plan["source_plan"]["path"])
    require(all(plan[k] == previous_plan[k] for k in ("source_root", "routing_source", "science_source", "model_cache")),
            "Frozen scorer runtime inputs differ")
    parent_row = read(plan["original_plan"]["path"])["arms"]["annealed-sparse-s41"]
    require(all(plan[k] == parent_row[k] for k in ("context_ready", "pocket")),
            "Pocket or context differs")
    owner = Path(plan["source_manifest"]["path"]).parent
    require({bound(pin) for pin in plan["overlay"]} == {owner / (name + ".py") for name in OVERLAY},
            "Use the previous pair's exact scope modules")
    return records, owner


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def run(plan, plan_path, records, owner):
    job = os.environ.get("LSB_JOBID", "")
    require(job.isdigit() and os.environ.get("LSB_QUEUE") == plan["queue"] and
            os.environ.get("XLA_FLAGS") == PROFILE, "Expected original allocated FP32 runtime")
    routing = module(Path(plan["routing_source"]) / "scripts/cuda_ownership.py", "_hard_refine_cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = routing.resolve_visibility(os.environ)["selected_cuda_visible_devices"]
    source = Path(plan["source_root"])
    sys.path[:0] = [str(source / "src"), str(Path(plan["science_source"]) / "scripts")]
    overlay = SealedChemistry(owner)
    sys.meta_path.insert(0, overlay)
    try:
        from mosaic.small_molecule.chemistry.atom_exchange import within_common_organic_scope
    finally:
        sys.meta_path.remove(overlay)
    for name in OVERLAY:
        require(Path(sys.modules[f"{PACKAGE}.{name}"].__file__) == owner / (name + ".py"),
                "Scoped repair code differs: " + name)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry._reference.chemistry import encode_smiles, inspect_graph
    from mosaic.small_molecule.chemistry._reference.chemistry_regularization import ALLOWED_VALENCES
    from mosaic.small_molecule.chemistry.environment_policy import BOND_ORDERS
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.io import load_context
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.optimizers import evaluate_graph
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.state import graph_from_hard, validate_graph
    from scoped_hard_neighborhood import rank_neighborhood
    import storage_accounting

    require(not jax.config.x64_enabled and jax.default_backend() == "gpu" and len(jax.devices("gpu")) == 1,
            "Require one allocated FP32 GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    started, next_gate = time.monotonic(), 0.0
    frozen = model = context = None
    result = {"version": VERSION, "job_id": job, "status": "running", "plan": binding(plan_path),
              "parent_snapshot": plan["initial"]["snapshot"], "parent_job": PARENT_JOB, "parent_step": PARENT_STEP,
              "score_scope": "Frozen Protenix Tiny 8A hard contact, fixed key 17",
              "reference_scores": REFERENCE, "candidate_forwards": 0, "ranking_vjps": 0,
              "accepted": [], "rounds": [], "rejections": {}, "relaxed_updates": 0}
    rejections = Counter()

    def persist():
        result["rejections"] = dict(rejections)
        write(output / "result.json", result)

    def tick():
        nonlocal next_gate
        now = time.monotonic()
        require(now - started < plan["internal_seconds"], "Own elapsed-time limit reached")
        if now >= next_gate:
            capacity = storage_accounting.assess_output(output.parent, plan["output_cap_bytes"], plan["file_cap"])
            write(output / "capacity-latest.json", capacity)
            require(capacity["status"] == "passed" and capacity["own_output"]["actual_bytes"] <= plan["output_cap_bytes"]
                    and capacity["own_output"]["actual_files"] <= plan["file_cap"], "Own output budget exceeded")
            next_gate = now + 300

    spec = mosaic_spec(25)
    pairs = np.triu_indices(25, 1)
    charges = np.zeros(25, dtype=np.int64)
    elements = np.asarray(spec.atomic_numbers[1:])
    orders = np.asarray([3 if name == "AROMATIC" else 2 * BOND_ORDERS[name] for name in spec.bond_names])
    key = jax.random.key(17)

    def valid_candidate(a, b, previous):
        matrix = np.zeros((25, 25), dtype=np.int64)
        matrix[pairs] = b
        matrix += matrix.T
        graph = (a + 1, charges, matrix)
        if not within_common_organic_scope(graph, spec):
            return None, "common_organic_scope"
        for slot in np.flatnonzero((elements[previous] != 16) & (elements[a] == 16)):
            cap = max(ALLOWED_VALENCES[int(elements[previous[slot]])][0])
            if int(orders[matrix[slot]].sum()) > 2 * cap:
                return None, "introduced_sulfur_old_cap"
        native = inspect_graph(*graph, spec)
        if native.smiles is None:
            return None, native.failure_reason
        try:
            encoded = encode_smiles(native.smiles, spec)
        except ValueError:
            return None, "roundtrip"
        if encoded["smiles"] != native.smiles:
            return None, "roundtrip"
        return native.smiles, None

    def record(a, b, smiles, score):
        return {"atoms": a.tolist(), "bonds": b.tolist(), "canonical_smiles": smiles,
                "native_valid_connected": True, "scoped_valid_connected": True,
                "sulfur_atoms": int(np.count_nonzero(elements[a] == 16)), "contact_score": score}

    try:
        tick()
        ready = read(plan["context_ready"]["path"])
        context = load_context(Path(plan["context_ready"]["path"]).parent / "context", ready["context_files"])
        model = FrozenProtenixTiny.load(cache_dir=Path(plan["model_cache"]))
        frozen = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
        require(frozen == plan["frozen"], "Frozen model or context differs")
        result["frozen_before"] = frozen
        loss = model.build_loss(context=context, loss=NativeContactLoss(
            8., map_protenix_pocket(context, read(plan["pocket"]["path"]))))

        @eqx.filter_jit
        def hard_forward(graph):
            value, aux = loss(graph, key=key)
            finite = jnp.isfinite(value)
            for leaf in jax.tree_util.tree_leaves(aux):
                if eqx.is_array(leaf):
                    finite &= jnp.all(jnp.isfinite(leaf))
            return value, finite

        def score(graph):
            tick()
            validate_graph(graph)
            value, finite = hard_forward(graph)
            value = -float(jax.block_until_ready(value))
            require(bool(finite) and math.isfinite(value), "Nonfinite hard forward")
            return value

        with np.load(bound(plan["initial"]["snapshot"]), allow_pickle=False) as saved:
            x, e = saved["X"], saved["E_upper"]
            require(int(saved["step"]) == 2000 and np.array_equal(saved["key_data"], np.asarray(jax.random.key_data(key))) and
                    x.shape == (25, 6) and e.shape == (300, 5) and x.dtype == e.dtype == np.float32 and
                    np.isfinite(x).all() and np.isfinite(e).all(), "Expected the exact saved neutral FP32 endpoint")
            atoms, bonds = x.argmax(-1), e.argmax(-1)
        smiles, failure = valid_candidate(atoms, bonds, atoms)
        require(smiles is not None, "Saved endpoint failed same-scope replay: " + str(failure))
        current = score(graph_from_hard(atoms, bonds))
        require(math.isclose(current, plan["initial"]["contact_score"], abs_tol=2e-6, rel_tol=1e-6),
                "Saved hard-score replay differs")
        result["initial"] = record(atoms, bonds, smiles, current)
        seen = {(tuple(atoms), tuple(bonds))}
        persist()
        stop = "round_limit"
        for round_index in range(ROUNDS):
            tick()
            if result["candidate_forwards"] >= CANDIDATE_CAP:
                stop = "forward_cap"
                break
            evaluation = evaluate_graph(loss, graph_from_hard(atoms, bonds), key)
            result["ranking_vjps"] += 1
            ranked = rank_neighborhood(atoms, bonds, -np.asarray(evaluation.gradients.X),
                                       -np.asarray(evaluation.gradients.E_upper), limit_per_kind=RANK_LIMIT)
            moves = (move for group in zip_longest(*(ranked[k] for k in ("atom", "bond", "reattach")))
                     for move in group if move is not None)
            round_record = {"round": round_index + 1, "source_score": current,
                            "gradient_forward_score": -float(evaluation.value),
                            "scored": [], "accepted": False}
            best = None
            for batch in range(2):
                count = 0
                for move in moves:
                    tick()
                    a, b = move.apply(atoms, bonds)
                    identity = (tuple(a), tuple(b))
                    if identity in seen:
                        rejections["already_scored"] += 1
                        continue
                    candidate_smiles, rejected = valid_candidate(a, b, atoms)
                    if candidate_smiles is None:
                        rejections[rejected] += 1
                        continue
                    value = score(graph_from_hard(a, b))
                    seen.add(identity)
                    result["candidate_forwards"] += 1
                    count += 1
                    entry = {"kind": move.kind, "atom_edits": move.atom_edits, "bond_edits": move.bond_edits,
                             "predicted_gain": move.predicted_gain, "contact_score": value,
                             "canonical_smiles": candidate_smiles, "batch": batch + 1}
                    round_record["scored"].append(entry)
                    if value > current + ACCEPT_MARGIN and (best is None or value > best[0]):
                        best = (value, a, b, candidate_smiles, entry)
                    if count >= BATCH or result["candidate_forwards"] >= CANDIDATE_CAP:
                        break
                if best is not None or count < BATCH or result["candidate_forwards"] >= CANDIDATE_CAP:
                    break
            if best is not None:
                current, atoms, bonds, smiles, entry = best
                result["accepted"].append({**record(atoms, bonds, smiles, current),
                                           "round": round_index + 1, "move": entry})
                round_record["accepted"] = True
            result["rounds"].append(round_record)
            persist()
            if best is None:
                stop = "forward_cap" if result["candidate_forwards"] >= CANDIDATE_CAP else "no_improving_valid_move_in_ranked_batches"
                break

        final_graph = graph_from_hard(atoms, bonds)
        final_smiles, failure = valid_candidate(atoms, bonds, atoms)
        require(final_smiles == smiles and failure is None, "Final hard graph failed scope")
        replay_score = score(final_graph)
        require(math.isclose(replay_score, current, abs_tol=2e-6, rel_tol=1e-6), "Final hard-score replay differs")
        path = output / "best-hard.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, X=np.asarray(final_graph.X), E_upper=np.asarray(final_graph.E_upper),
                                element_indices=atoms, bond_upper_indices=bonds,
                                key_data=np.asarray(jax.random.key_data(key)),
                                parent_global_step=PARENT_STEP, accepted_hard_moves=len(result["accepted"]),
                                contact_primal=-replay_score)
        result.update(status="completed_scoped_hard_contact", stop_reason=stop,
                      final={**record(atoms, bonds, smiles, replay_score), "snapshot": binding(path)},
                      reference_forwards=2)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            if frozen is not None:
                result["frozen_after"] = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
                require(result["frozen_after"] == frozen, "Frozen scorer changed")
            for pin in records:
                bound(pin)
            result["inputs_unchanged"] = True
        except BaseException as error:
            result.update(status="failed", preservation_error=repr(error))
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            persist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    path = args.plan.resolve()
    plan = read(path)
    records, owner = inputs(plan, path)
    if args.startup_only:
        print(json.dumps({"status": "metadata_startup_passed_no_models", "parent_step": PARENT_STEP,
                          "candidate_cap": CANDIDATE_CAP, "rounds": ROUNDS}))
    else:
        result = run(plan, path, records, owner)
        print(json.dumps({"status": result["status"], "accepted_moves": len(result["accepted"]),
                          "hard_contact": result["final"]["contact_score"]}))


if __name__ == "__main__":
    main()
