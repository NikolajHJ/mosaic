"""Read-only native contact audit of saved soft, hard and repair graphs.

Native imports occur only inside ``run`` on an allocated GPU.  The small plan,
binding and teacher-label helpers are deliberately standard-library-only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


VERSION = "protenix-score-transfer-audit-v1"
PROFILE = "--xla_gpu_autotune_level=0 --xla_gpu_exclude_nondeterministic_ops=true"
REFERENCE_EXTRAS = {"arrays", "identity", "preparation_seconds", "search_seconds"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def bound(record):
    require(set(record) >= {"path", "bytes", "sha256"}, "Incomplete input binding")
    actual = binding(record["path"])
    require(actual == {k: record[k] for k in actual}, "Input bytes differ: " + record["path"])
    return Path(record["path"])


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_plan(plan):
    require(plan["version"] == VERSION, "Unknown audit plan")
    require(plan.get("optimizer_updates", 0) == 0, "Audit must not request updates")
    points = plan["points"]
    require(isinstance(points, list) and 1 <= len(points) <= 24, "Require 1--24 saved points")
    ids = [p["id"] for p in points]
    require(len(ids) == len(set(ids)), "Duplicate point identifiers")
    for point in points:
        require(isinstance(point["id"], str) and point["id"] and
                all(c.isalnum() or c in "-_" for c in point["id"]), "Unsafe point identifier")
        require(type(point["step"]) is int and point["step"] >= 0, "Invalid saved step")
        temperatures = point.get("temperatures", [])
        require(isinstance(temperatures, list) and len(temperatures) <= 3, "At most three temperature controls")
        require(all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in temperatures),
                "Temperatures must be finite and positive")
    require(type(plan.get("internal_seconds", 3000)) is int and
            1 <= plan.get("internal_seconds", 3000) <= 6600, "Invalid internal time budget")
    require(type(plan.get("output_cap_bytes", 67108864)) is int and
            0 < plan.get("output_cap_bytes", 67108864) <= 268435456, "Invalid audit output cap")
    return plan


def input_bindings(plan):
    records = [plan["context_ready"], plan["pocket"], *plan.get("source_files", [])]
    for point in plan["points"]:
        records.append(point["snapshot"])
        if point.get("reference") is not None:
            records.append(point["reference"])
    return records


def teacher_labels(reference, *, step, n_atoms=25):
    """Authenticate the saved prepared payload; never search for a new target.

    Routes 3/4 contain an incomplete witness or no witness and are deliberately
    not promoted to valid repair candidates.  Route 0 records the raw valid graph.
    """
    repair = reference["repair"]
    receipt = {k: v for k, v in repair.items() if k not in REFERENCE_EXTRAS}
    identity = hashlib.sha256(json.dumps({"arrays": repair["arrays"], "receipt": receipt},
                                        sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == repair["identity"], "Saved teacher identity differs")
    require(receipt["attempted_rows"] == 1 and receipt["source_update"] == step,
            "Teacher belongs to a different saved point")
    arrays = repair["arrays"]
    require(len(arrays["route"]) == 1 and len(receipt["rows"]) == 1, "Require a single teacher row")
    route = arrays["route"][0]
    require(type(route) is int and route in range(5) and receipt["rows"][0]["route"] == route,
            "Invalid teacher route")
    result = {"route": route, "identity": identity, "receipt": receipt,
              "candidate": None, "status": "no_complete_saved_witness"}
    if route not in (0, 1, 2):
        return result
    atoms, charges, bonds = (arrays[k][0] for k in ("target_atoms", "target_charges", "target_bonds"))
    require(len(atoms) == n_atoms and len(charges) == n_atoms and len(bonds) == n_atoms,
            "Teacher atom count differs")
    require(all(type(v) is int and 1 <= v <= 6 for v in atoms), "Teacher changes fixed occupied support")
    require(all(type(v) is int and v == 0 for v in charges), "Teacher changes neutral charge support")
    require(arrays["active"] == [[True] * n_atoms], "Teacher occupancy differs")
    require(all(len(row) == n_atoms for row in bonds), "Teacher bond dimensions differ")
    require(all(type(bonds[i][j]) is int and 0 <= bonds[i][j] < 5 and
                bonds[i][j] == bonds[j][i] and (i != j or bonds[i][j] == 0)
                for i in range(n_atoms) for j in range(n_atoms)), "Invalid teacher bond labels")
    result.update(status="saved_raw_already_valid" if route == 0 else "saved_complete_repair",
                  candidate={"element_indices": [a - 1 for a in atoms],
                             "bond_upper_indices": [bonds[i][j] for i in range(n_atoms)
                                                    for j in range(i + 1, n_atoms)]})
    return result


def positive_temperature_argmax(logits, temperature):
    """Portable control: positive scaling preserves ordering; no scientific arrays."""
    require(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
            "Positive finite temperature required")
    require(logits and all(math.isfinite(v) for v in logits), "Finite nonempty row required")
    return max(range(len(logits)), key=lambda i: logits[i] / temperature)


def run(plan, plan_path):
    validate_plan(plan)
    before_inputs = [binding(bound(r)) for r in input_bindings(plan)]
    require(os.environ.get("LSB_JOBID", "").isdigit(), "Native audit requires an allocated GPU")
    if plan.get("queue") is not None:
        require(os.environ.get("LSB_QUEUE") == plan["queue"], "Audit queue differs")
    require(os.environ.get("XLA_FLAGS") == PROFILE, "Original declared execution profile required")
    if plan.get("routing_source") is not None:
        routing_path = Path(plan["routing_source"]) / "scripts/cuda_ownership.py"
        spec = importlib.util.spec_from_file_location("_score_transfer_cuda", routing_path)
        routing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(routing)
        os.environ["CUDA_VISIBLE_DEVICES"] = routing.resolve_visibility(os.environ)["selected_cuda_visible_devices"]
    sys.path.insert(0, str(Path(plan["source_root"]) / "src"))
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.io import load_context, hard_readout
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.state import RelaxedGraph, graph_from_hard, validate_graph
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.chemistry._reference.quality import graph_identity

    require(not jax.config.x64_enabled and jax.default_backend() == "gpu" and len(jax.devices("gpu")) == 1,
            "Require original FP32 and one allocated GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    result = {"version": VERSION, "status": "running", "job_id": os.environ["LSB_JOBID"],
              "plan": binding(plan_path), "inputs_before": before_inputs, "optimizer_updates": 0,
              "can_admit_new_child": False, "native_forward_evaluations": 0, "points": [],
              "scope": "Frozen geometry-off Protenix Tiny contact at saved graph/context/key; no affinity, pose or binding certification",
              "production_execution_profile": PROFILE}
    model = context = None
    frozen = None
    def persist():
        write(output / "result.json", result)
        require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= plan.get("output_cap_bytes", 67108864),
                "Own audit output cap exceeded")
    def tick():
        require(time.monotonic() - begun < plan.get("internal_seconds", 3000), "Audit internal time budget exceeded")
    try:
        ready = read(plan["context_ready"]["path"])
        context = load_context(Path(plan["context_ready"]["path"]).parent / "context", ready["context_files"])
        model = FrozenProtenixTiny.load(cache_dir=Path(plan["model_cache"]))
        frozen = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
        require(frozen == plan["frozen_identity"], "Original frozen model/context differs")
        result["frozen_before"] = frozen
        loss = model.build_loss(context=context, loss=NativeContactLoss(
            8., map_protenix_pocket(context, read(plan["pocket"]["path"]))))
        key = jax.random.key(17)
        schema = mosaic_spec(25)

        @eqx.filter_jit
        def forward(graph):
            value, aux = loss(graph, key=key)
            finite = jnp.isfinite(value)
            for leaf in jax.tree_util.tree_leaves(aux):
                if eqx.is_array(leaf):
                    finite = finite & jnp.all(jnp.isfinite(leaf))
            return value, finite

        def score(graph):
            tick()
            validate_graph(graph)
            value, finite = forward(graph)
            value = float(jax.block_until_ready(value))
            require(bool(finite) and math.isfinite(value), "Nonfinite native forward computation")
            result["native_forward_evaluations"] += 1
            readout = hard_readout(graph)
            readout.pop("pairs", None)
            return {"contact_loss": value, "contact_score": -value, "readout": readout,
                    "strict_valid_connected": bool(readout["sanitized"] and not readout["disconnected"])}

        for point in plan["points"]:
            tick()
            with np.load(point["snapshot"]["path"], allow_pickle=False) as saved:
                require(saved["step"].shape == () and saved["step"].dtype.kind in "iu" and
                        int(saved["step"]) == point["step"], "Saved snapshot step differs")
                require(np.array_equal(saved["key_data"], np.asarray(jax.random.key_data(key))), "Saved model key differs")
                require(saved["X"].dtype == np.float32 and saved["E_upper"].dtype == np.float32,
                        "Saved graph must retain original FP32")
                graph = RelaxedGraph(jnp.asarray(saved["X"]), jnp.asarray(saved["E_upper"]))
                atom_logits, bond_logits = (jnp.asarray(saved[k]) for k in ("atom_logits", "bond_logits"))
                require(atom_logits.shape == (25, 6) and bond_logits.shape == (300, 5) and
                        atom_logits.dtype == jnp.float32 and bond_logits.dtype == jnp.float32,
                        "Saved logit shapes/dtypes differ")
                require(all(bool(jnp.isfinite(v).all()) for v in (atom_logits, bond_logits)), "Nonfinite saved logits")
                source_temperature = float(saved["temperature"]) if "temperature" in saved else 1.
                saved_primal = float(saved["contact_primal"]) if "contact_primal" in saved else None
                require(math.isfinite(source_temperature) and source_temperature > 0 and
                        (saved_primal is None or math.isfinite(saved_primal)), "Nonfinite saved metadata")
            soft = score(graph)
            hard = graph_from_hard(np.asarray(graph.X).argmax(-1), np.asarray(graph.E_upper).argmax(-1))
            entry = {"id": point["id"], "arm": point["arm"], "step": point["step"],
                     "snapshot": point["snapshot"], "saved_temperature": source_temperature,
                     "saved_contact_loss": saved_primal, "soft": soft, "hard": score(hard),
                     "temperature_controls": [], "repair": {"status": "no_saved_reference"}}
            for temperature in point.get("temperatures", []):
                control = RelaxedGraph(jax.nn.softmax(atom_logits / temperature, -1),
                                       jax.nn.softmax(bond_logits / temperature, -1))
                require(np.array_equal(np.asarray(control.X).argmax(-1), np.asarray(hard.X).argmax(-1)) and
                        np.array_equal(np.asarray(control.E_upper).argmax(-1), np.asarray(hard.E_upper).argmax(-1)),
                        "Positive-temperature control unexpectedly changed argmax")
                entry["temperature_controls"].append({"temperature": temperature, **score(control)})
            if point.get("reference") is not None:
                with gzip.open(point["reference"]["path"], "rt", encoding="utf-8") as stream:
                    reference = json.load(stream)
                teacher = teacher_labels(reference, step=point["step"])
                require(teacher["receipt"]["schema_sha256"] == schema.fingerprint(), "Saved teacher schema differs")
                atoms = np.asarray(graph.X).argmax(-1) + 1
                charges = np.zeros(25, dtype=np.int64)
                bonds = np.zeros((25, 25), dtype=np.int64)
                i, j = np.triu_indices(25, 1)
                bonds[i, j] = np.asarray(graph.E_upper).argmax(-1)
                bonds[j, i] = bonds[i, j]
                require(graph_identity(atoms, charges, bonds, schema) == teacher["receipt"]["rows"][0]["raw_sha256"],
                        "Saved teacher was prepared from a different raw graph")
                entry["repair"] = {"status": teacher["status"], "route": teacher["route"],
                                   "identity": teacher["identity"], "reference": point["reference"]}
                if teacher["candidate"] is not None:
                    candidate = teacher["candidate"]
                    repaired = graph_from_hard(candidate["element_indices"], candidate["bond_upper_indices"])
                    repair_score = score(repaired)
                    require(repair_score["strict_valid_connected"], "Saved complete repair failed native sanitization/connectivity")
                    entry["repair"].update(repair_score)
            result["points"].append(entry)
            persist()
        result["status"] = "completed_read_only_score_transfer_audit"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        if frozen is not None:
            after = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
            result["frozen_after"] = after
            result["frozen_unchanged"] = after == frozen
            if after != frozen:
                result.update(status="failed", frozen_error="Frozen model/context changed")
        try:
            result["inputs_after"] = [binding(bound(r)) for r in input_bindings(plan)]
            result["inputs_unchanged"] = result["inputs_after"] == before_inputs
        except BaseException as error:
            result.update(status="failed", input_error=repr(error), inputs_unchanged=False)
        result["elapsed_seconds"] = time.monotonic() - begun
        persist()
    require(result["status"] == "completed_read_only_score_transfer_audit", "Audit preservation check failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    plan = validate_plan(read(args.plan))
    if args.startup_only:
        for record in input_bindings(plan):
            bound(record)
        print(json.dumps({"status": "metadata_startup_passed_no_models", "points": len(plan["points"]), "optimizer_updates": 0}))
        return
    result = run(plan, args.plan)
    print(json.dumps({"status": result["status"], "points": len(result["points"]),
                      "native_forward_evaluations": result["native_forward_evaluations"], "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
