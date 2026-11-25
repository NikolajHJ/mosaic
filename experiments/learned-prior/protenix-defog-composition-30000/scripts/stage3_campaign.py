"""Plan, execute and collect the LR60 Stage 3 study. Native imports are lazy."""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parents[2]


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(stream, row):
    stream.write(json.dumps(row, allow_nan=False) + "\n")
    stream.flush()


@contextmanager
def _finish_bundle(record, path):
    """Record bootstrap failures and context-manager frozen-check failures too."""
    try:
        yield
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_utc"] = utc()
        write_json(path, record)


def labels_identity(graph):
    """Atom-order identity, deliberately distinct from canonical SMILES."""
    import numpy as np
    labels = [np.asarray(graph.X).argmax(-1).tolist(),
              np.asarray(graph.E_upper).argmax(-1).tolist()]
    return hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode()).hexdigest()


def source_digest():
    """One compact code identity, without inventorying runtime environments."""
    directories = (HOME / "scripts", ROOT / "src/mosaic/small_molecule/core",
                   ROOT / "src/mosaic/small_molecule/Stage2", ROOT / "models/defog/src")
    files = {p for directory in directories for p in directory.rglob("*.py")}
    files.add(ROOT / "src/mosaic/small_molecule/Stage1/diagnostics.py")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_plan(config_path, output, campaign_id=None, max_arms=4, wave="all", mock=False):
    from stage3_design import load_design, expand_conditions, expand_runs, make_bundles

    config = load_design(config_path)
    conditions = expand_conditions(config)
    runs = expand_runs(config)
    bundles = make_bundles(config, max_arms=max_arms)
    # Run configs have one canonical location in the plan. Bundle membership
    # holds IDs only, avoiding a second copy of thousands of resolved configs.
    bundles = [{k: v for k, v in b.items() if k != "runs"} for b in bundles]
    if wave == "reference":
        bundles = [b for b in bundles if b["wave"] == 0]
        ids = {rid for b in bundles for rid in b["run_ids"]}
        runs = [r for r in runs if r["run_id"] in ids]
        selected = {r["condition_id"] for r in runs}
        conditions = [c for c in conditions if c["id"] in selected]
    elif wave != "all":
        raise ValueError("wave must be reference or all")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    campaign_id = campaign_id or output.name
    for run in runs:
        run["campaign_id"] = campaign_id
    plan = {
        "schema": "defog-stage3-v1", "campaign_id": campaign_id,
        "created_utc": utc(), "config": config,
        "wave": wave, "mock": mock, "execution_mode": "mock" if mock else "native",
        "conditions": conditions, "runs": runs, "bundles": bundles,
        "primary_endpoint": "highest hard Protenix contact score among raw sanitized connected argmax molecules",
        "evaluation": "same frozen context/key; initial identity flagged; no repair in optimization or selection",
    }
    write_json(output / "plan.json", plan)
    (output / "runs").mkdir()
    (output / "bundles").mkdir()
    return plan


def make_evaluators(native):
    """Bundle-resident compiled functions, reused across matched conditions."""
    import equinox as eqx
    from stage3_math import cool_graph, teacher_distance, simple_soft_valence_loss

    def view(g, temperature, placement, gradient):
        return cool_graph(g, temperature, gradient) if placement == "view" else g

    @eqx.filter_jit
    def native_grad(g, key, temperature, placement, gradient):
        def loss(current):
            return native(view(current, temperature, placement, gradient), key=key)
        return eqx.filter_value_and_grad(loss, has_aux=True)(g)

    @eqx.filter_jit
    def native_forward(g, key):
        return native(g, key=key)

    @eqx.filter_jit
    def prior_grad(g, target, temperature, placement, gradient, kind, atom_scale, bond_scale, normalization):
        def loss(current):
            return teacher_distance(view(current, temperature, placement, gradient), target,
                                    kind=kind, atom_scale=atom_scale, bond_scale=bond_scale,
                                    normalization=normalization)
        return eqx.filter_value_and_grad(loss)(g)

    @eqx.filter_jit
    def chem_grad(g, temperature, placement, gradient):
        return eqx.filter_value_and_grad(
            lambda current: simple_soft_valence_loss(view(current, temperature, placement, gradient))
        )(g)

    return native_grad, native_forward, prior_grad, chem_grad


def _finite(*trees):
    import jax
    import numpy as np
    for tree in trees:
        for leaf in jax.tree.leaves(tree):
            if hasattr(leaf, "dtype") and not np.isfinite(np.asarray(leaf)).all():
                raise FloatingPointError("Nonfinite objective, native output or component gradient")


def _gradient_metrics(gradient, prefix):
    import numpy as np
    result = {}
    for name in ("X", "E_upper"):
        values = np.asarray(getattr(gradient, name))
        centered = values - values.mean(-1, keepdims=True)
        result[f"{prefix}_{name}_norm"] = float(np.linalg.norm(values))
        result[f"{prefix}_{name}_tangent_norm"] = float(np.linalg.norm(centered))
        result[f"{prefix}_{name}_max_abs"] = float(np.abs(values).max(initial=0))
    return result


def shuffle_teacher_targets(view, target, seed, update):
    """Permute targets among rows with the same current winning category."""
    import numpy as np
    import jax.numpy as jnp
    from mosaic.small_molecule.core.state import RelaxedGraph

    rng = np.random.default_rng(np.random.SeedSequence([seed, update, 260325]))
    output, scales, diagnostics = [], [], {}
    for name in ("X", "E_upper"):
        current, clean = (np.asarray(getattr(g, name)) for g in (view, target))
        categories = current.argmax(-1)
        permutation = np.arange(len(current))
        for category in np.unique(categories):
            rows = np.flatnonzero(categories == category)
            permutation[rows] = rng.permutation(rows)
        shuffled = clean[permutation]

        def tangent_norm(rows):
            delta = rows - current
            return float(np.linalg.norm(delta - delta.mean(-1, keepdims=True)))

        live_norm, shuffled_norm = tangent_norm(clean), tangent_norm(shuffled)
        scale = live_norm / shuffled_norm if shuffled_norm > 0 else 1.0
        diagnostics[name] = {"permuted_rows": int(np.count_nonzero(permutation != np.arange(len(current)))),
                             "teacher_tangent_displacement": live_norm,
                             "shuffled_tangent_displacement": shuffled_norm, "scale": scale,
                             "flat_shuffled_nonflat_live": shuffled_norm == 0 and live_norm > 0}
        output.append(jnp.asarray(shuffled))
        scales.append(scale)
    return RelaxedGraph(*output), scales, diagnostics


def _components(graph):
    import numpy as np
    n = len(graph.X)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    i, j = np.triu_indices(n, 1)
    for a, b in zip(i[np.asarray(graph.E_upper).argmax(-1) != 0],
                    j[np.asarray(graph.E_upper).argmax(-1) != 0]):
        parent[find(int(a))] = find(int(b))
    sizes = {}
    for a in range(n):
        r = find(a)
        sizes[r] = sizes.get(r, 0) + 1
    return sorted(sizes.values(), reverse=True)


def _hard_readout(graph):
    from rdkit import rdBase
    from mosaic.small_molecule.Stage1.diagnostics import raw_readout
    with rdBase.BlockLogs():
        return raw_readout(graph)


def _norm(gradient):
    import numpy as np
    return float(np.sqrt(sum(np.sum(np.asarray(x, dtype=np.float64) ** 2)
                             for x in (gradient.X, gradient.E_upper))))


def run_arm(config, output, native, teacher, hard_cache, *, mock=False, evaluators=None):
    """One fresh arm; weights/teacher targets never enter the native model."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.core.contracts import RawEvaluation
    from mosaic.small_molecule.core.optimizers import projected_step
    from mosaic.small_molecule.core.state import RelaxedGraph, validate_graph
    from mosaic.small_molecule.Stage1.diagnostics import graph_metrics, valence_excess
    from stage3_math import (
        apply_projection, cool_graph, cool_state_graph, make_initial_graph,
        schedule_value, teacher_distance, simple_soft_valence_loss,
    )
    from stage3_results import summarize_run

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    updates = int(config.get("updates", 500))
    condition = config["condition"]
    prior = condition["teacher"]
    cooling = condition["cooling"]
    placement = cooling["placement"]
    graph = make_initial_graph(config["seed"], config["initialization"], config.get("n_atoms", 25))
    source_graph = graph
    interior_mass = float(condition.get("interior_uniform_mass", 0))
    if not 0 <= interior_mass < 1:
        raise ValueError("Interior uniform mass must be in [0, 1)")

    def mix_interior(current):
        return jax.tree.map(lambda rows: (1 - interior_mass) * rows + interior_mass / rows.shape[-1], current)

    if interior_mass:
        graph = mix_interior(graph)
    temperature = schedule_value(cooling["schedule"], 0, updates)
    if placement == "state":
        graph = cool_state_graph(graph, 1.0, temperature)
    key = jax.random.fold_in(jax.random.key(config["seed"]), 0xD3F09)
    key_data = np.asarray(jax.random.key_data(key))
    key_hash = hashlib.sha256(key_data.tobytes()).hexdigest()
    config = config | {"model_key": key_data.tolist(), "model_key_impl": str(jax.random.key_impl(key)),
                       "model_key_policy": "fixed per seed across conditions, states, relaxed and hard inputs"}
    write_json(output / "config.json", config)
    np.savez_compressed(output / "initial.npz", X=graph.X, E_upper=graph.E_upper,
                        source_X=source_graph.X, source_E_upper=source_graph.E_upper, model_key=key_data)

    native_grad, native_forward, prior_grad, chem_grad = evaluators or make_evaluators(native)

    active = any(schedule_value(prior["weight"], u, updates) > 0 for u in range(updates))
    chemistry_weight = float(condition.get("chemistry_weight", 0))
    chemistry_kind = condition.get("chemistry_loss", "simple_soft_valence")
    repair_step = None
    if chemistry_kind == "failure_aware_repair":
        if not chemistry_weight or not interior_mass:
            raise ValueError("Repair guidance requires a positive coefficient and an interior graph")
        from stage3_repair_comparison import make_repair_evaluator
        repair_step = make_repair_evaluator()
    elif chemistry_kind != "simple_soft_valence":
        raise ValueError(f"Unknown chemistry loss {chemistry_kind}")
    stale_target, calibration = None, 1.0
    warnings = []
    started = time.monotonic()
    states, candidates, arrays_x, arrays_e, views_x, views_e = [], {}, [], [], [], []
    candidate_details = {}
    best = None
    previous_labels, previous_graph = None, None
    query_start = teacher.queries if teacher is not None else 0
    completed = 0
    status = {"status": "running", "started_utc": utc(), "updates_completed": 0,
              "mock": mock, "pid": os.getpid()}
    write_json(output / "status.json", status)
    streams = {}
    try:
        with ExitStack() as stack:
            for name in ("states", "candidates", "gradients", "operators", "teacher"):
                if config.get("compress_jsonl", False):
                    streams[name] = stack.enter_context(gzip.open(output / f"{name}.jsonl.gz", "xt",
                                                                 encoding="utf-8", compresslevel=6))
                else:
                    streams[name] = stack.enter_context((output / f"{name}.jsonl").open("x", encoding="utf-8"))
            for update in range(updates + 1):
                temp = schedule_value(cooling["schedule"], min(update, updates - 1), updates)
                time_value = schedule_value(prior["time"], min(update, updates - 1), updates)
                weight = schedule_value(prior["weight"], min(update, updates - 1), updates)
                view = cool_graph(graph, temp, cooling["gradient"]) if placement == "view" else graph
                validate_graph(graph)
                validate_graph(view)
                if update < updates:
                    (native_value, aux), gn = native_grad(graph, key, jnp.asarray(temp, jnp.float32), placement, cooling["gradient"])
                    _finite(native_value, aux, gn)
                else:
                    native_value, aux = native_forward(view, key)
                    _finite(native_value, aux)
                soft_score = -float(native_value)
                stored_soft_score = soft_score
                if placement == "view" and temp != 1:
                    stored_loss, stored_aux = native_forward(graph, key)
                    _finite(stored_loss, stored_aux)
                    stored_soft_score = -float(stored_loss)
                hard_id = labels_identity(graph)
                cache_key = (key_hash, hard_id)
                cached = hard_cache.get(cache_key)
                if cached is None:
                    hard, report = _hard_readout(graph)
                    hard_loss, hard_aux = native_forward(hard, key)
                    _finite(hard_loss, hard_aux)
                    cached = (hard, report, -float(hard_loss))
                    hard_cache[cache_key] = cached
                hard, report, hard_score = cached
                valid = bool(report["rdkit_sanitized"] and report["fragment_count"] == 1)
                labels = (np.asarray(hard.X).argmax(-1), np.asarray(hard.E_upper).argmax(-1))
                sizes = _components(hard)
                x = np.asarray(graph.X)
                e = np.asarray(graph.E_upper)
                qx, qe = np.asarray(view.X), np.asarray(view.E_upper)
                _, view_excess = valence_excess(qx, qe)
                row = {
                    "update": update, "valid": valid, "sanitized": report["rdkit_sanitized"],
                    "connected": report["fragment_count"] == 1, "smiles": report["canonical_smiles"],
                    "hard_id": hard_id, "hard_score": hard_score, "soft_score": soft_score,
                    "stored_soft_score": stored_soft_score, "soft_hard_gap": soft_score - hard_score,
                    "stored_soft_hard_gap": stored_soft_score - hard_score,
                    "component_count": len(sizes), "largest_component": sizes[0],
                    "atoms_outside_largest_component": len(x) - sizes[0],
                    "hard_valence_excess": report["valence_excess"],
                    "hard_valence_violation_count": report["valence_violation_count"],
                    "sanitization_error": report["sanitization_error"],
                    "temperature": temp, "teacher_time": time_value, "teacher_weight": weight,
                    "expected_sulfur": float(x[:, 3].sum()), "hard_sulfur": int((labels[0] == 3).sum()),
                    "atom_edits": int((labels[0] != previous_labels[0]).sum()) if previous_labels else 0,
                    "bond_edits": int((labels[1] != previous_labels[1]).sum()) if previous_labels else 0,
                    "view_expected_valence_excess": view_excess,
                    "view_atom_entropy": float(-np.sum(qx * np.log(np.maximum(qx, 1e-30))) / len(qx)),
                    "view_bond_entropy": float(-np.sum(qe * np.log(np.maximum(qe, 1e-30))) / max(len(qe), 1)),
                    "large_gap_002": soft_score - hard_score > .02,
                    "large_gap_005": soft_score - hard_score > .05,
                    "large_gap_010": soft_score - hard_score > .1,
                    "native_aux": {name: np.asarray(value).item() for name, value in aux.items()
                                   if np.asarray(value).ndim == 0},
                    **graph_metrics(graph),
                }
                row["atom_step_l2"] = float(np.linalg.norm(x - np.asarray(previous_graph.X))) if previous_graph is not None else 0.0
                row["bond_step_l2"] = float(np.linalg.norm(e - np.asarray(previous_graph.E_upper))) if previous_graph is not None else 0.0
                emit(streams["states"], row)
                states.append(row)
                arrays_x.append(x.copy()); arrays_e.append(e.copy())
                views_x.append(qx.copy()); views_e.append(qe.copy())
                if hard_id not in candidates:
                    candidate = {
                        "hard_id": hard_id, "first_seen": update, "update": update,
                        "smiles": row["smiles"], "valid": valid, "sanitized": row["sanitized"],
                        "connected": row["connected"], "hard_score": hard_score,
                        "element_indices": labels[0].tolist(), "bond_upper_indices": labels[1].tolist(),
                    }
                    candidates[hard_id] = candidate
                    emit(streams["candidates"], candidate)
                if valid and (best is None or hard_score > best["hard_score"]):
                    best = row
                    np.savez_compressed(output / "best_valid.tmp.npz", X=hard.X, E_upper=hard.E_upper,
                                        soft_X=x, soft_E_upper=e, update=update, hard_score=hard_score,
                                        model_key=key_data)
                    (output / "best_valid.tmp.npz").replace(output / "best_valid.npz")
                if update == updates:
                    break
                gp = jax.tree.map(jnp.zeros_like, graph)
                gc = jax.tree.map(jnp.zeros_like, graph)
                prior_value, chem_value = jnp.asarray(0.), jnp.asarray(0.)
                query_seconds = 0.0
                shuffle_diagnostics = None
                target = None
                if active:
                    query_start_time = time.monotonic()
                    if prior["target"] == "identity":
                        target = jax.tree.map(jax.lax.stop_gradient, view)
                    elif stale_target is not None and prior["refresh"] == "initial":
                        target = stale_target
                    else:
                        if teacher is None:
                            raise RuntimeError("Active teacher arm has no teacher")
                        teacher_input = _hard_readout(view)[0] if prior["input"] == "hard" else view
                        reply = teacher.predict(teacher_input, time=time_value)
                        target = RelaxedGraph(jnp.asarray(reply.X), jnp.asarray(reply.E_upper))
                        validate_graph(target)
                        if target.X.shape != graph.X.shape or target.E_upper.shape != graph.E_upper.shape:
                            raise ValueError("Teacher target changed graph slots or categories")
                        if prior["refresh"] == "initial":
                            stale_target = target
                        tx = np.asarray(target.X) - qx
                        te = np.asarray(target.E_upper) - qe
                        emit(streams["teacher"], {
                            "update": update, "time": time_value,
                            "atom_tv": float(np.abs(tx).sum(-1).mean() / 2),
                            "bond_tv": float(np.abs(te).sum(-1).mean() / 2),
                            "atom_argmax_changes": int((np.asarray(target.X).argmax(-1) != qx.argmax(-1)).sum()),
                            "bond_argmax_changes": int((np.asarray(target.E_upper).argmax(-1) != qe.argmax(-1)).sum()),
                            "metrics": reply.metadata,
                        })
                    query_seconds = time.monotonic() - query_start_time
                    atom_scale, bond_scale = prior["atom_scale"], prior["bond_scale"]
                    if prior.get("shuffle_within_argmax"):
                        if (temp != 1 or prior["loss"] != "mse" or prior["refresh"] != "current"
                                or prior["target"] != "teacher" or prior["calibration"] != "none"):
                            raise ValueError("Shuffled control requires an uncooled live MSE teacher")
                        target, scales, shuffle_diagnostics = shuffle_teacher_targets(
                            view, target, config["seed"], update,
                        )
                        atom_scale *= scales[0]
                        bond_scale *= scales[1]
                    prior_value, gp = prior_grad(
                        graph, target, jnp.asarray(temp, jnp.float32), placement, cooling["gradient"], prior["loss"],
                        atom_scale, bond_scale, prior["normalization"],
                    )
                    if update == 0 and prior.get("calibration") == "initial_gradient":
                        reference = prior["calibration_reference"]
                        _, gr = prior_grad(graph, target, jnp.asarray(temp, jnp.float32), placement, cooling["gradient"], reference["loss"],
                                           reference["atom_scale"], reference["bond_scale"], reference["normalization"])
                        denominator = weight * _norm(gp)
                        if denominator > 0:
                            calibration = reference["weight"] * _norm(gr) / denominator
                        else:
                            warnings.append("Flat initial calibration gradient; multiplier left at1")
                if repair_step is not None:
                    chem_value, gc = repair_step(graph, jnp.asarray(temp, jnp.float32), placement, cooling["gradient"])
                elif chemistry_weight:
                    chem_value, gc = chem_grad(graph, jnp.asarray(temp, jnp.float32), placement, cooling["gradient"])
                applied_weight = weight * calibration
                combined = jax.tree.map(lambda n, p, c: n + applied_weight * p + chemistry_weight * c, gn, gp, gc)
                value = native_value + applied_weight * prior_value + chemistry_weight * chem_value
                _finite(prior_value, chem_value, gp, gc, combined, value)
                gradient_row = {
                    "update": update, "native_loss": float(native_value), "teacher_loss": float(prior_value),
                    "chemistry_loss": float(chem_value), "combined_loss": float(value),
                    "teacher_weight": weight, "applied_teacher_weight": applied_weight,
                    "calibration_multiplier": calibration, "teacher_query_seconds": query_seconds,
                    **_gradient_metrics(gn, "native"), **_gradient_metrics(gp, "teacher"),
                    **_gradient_metrics(gc, "chemistry"), **_gradient_metrics(combined, "combined"),
                }
                if repair_step is not None:
                    gradient_row["repair"] = repair_step.history[-1]
                if shuffle_diagnostics is not None:
                    gradient_row["shuffled_teacher"] = shuffle_diagnostics
                for leaf in ("X", "E_upper"):
                    n = gradient_row[f"native_{leaf}_tangent_norm"]
                    p = applied_weight * gradient_row[f"teacher_{leaf}_tangent_norm"]
                    gradient_row[f"weighted_teacher_{leaf}_tangent_norm"] = p
                    gradient_row[f"teacher_native_{leaf}_ratio"] = p / n if n > 0 else None
                emit(streams["gradients"], gradient_row)
                if valid and hard_id not in candidate_details:
                    detail = {"update": update, "teacher_weight": weight,
                              "applied_teacher_weight": applied_weight, "temperature": temp,
                              "teacher_time": time_value}
                    for prefix, gradient in (("native", gn), ("teacher", gp),
                                             ("chemistry", gc), ("combined", combined)):
                        for leaf in ("X", "E_upper"):
                            detail[f"gradient_{prefix}_{leaf}"] = np.asarray(getattr(gradient, leaf)).copy()
                    if target is not None:
                        detail["teacher_target_X"] = np.asarray(target.X).copy()
                        detail["teacher_target_E_upper"] = np.asarray(target.E_upper).copy()
                    candidate_details[hard_id] = detail
                evaluation = RawEvaluation(graph, value, {"native_loss": native_value}, combined, key)
                result = projected_step(evaluation, eta_X=config.get("learning_rate", 60), eta_E=config.get("learning_rate", 60))
                next_graph = result.next_graph
                if placement == "state":
                    next_temp = schedule_value(cooling["schedule"], min(update + 1, updates - 1), updates)
                    next_graph = cool_state_graph(next_graph, temp, next_temp)
                next_graph, operator_row = apply_projection(next_graph, condition["projection"], update + 1)
                if interior_mass:
                    next_graph = mix_interior(next_graph)
                    operator_row["post_projection_uniform_mass"] = interior_mass
                emit(streams["operators"], {"update": update + 1,
                    "simplex_X_l2": result.projection_residuals["X"],
                    "simplex_E_upper_l2": result.projection_residuals["E_upper"], **operator_row})
                previous_graph, previous_labels = graph, labels
                graph = next_graph
                completed += 1
                if completed % 25 == 0:
                    write_json(output / "status.json", status | {"updates_completed": completed, "updated_utc": utc()})
        status.update(status="complete", updates_completed=completed, finished_utc=utc())
    except Exception as error:
        status.update(status="failed", updates_completed=completed, finished_utc=utc(),
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        status["teacher_queries"] = (teacher.queries - query_start) if teacher is not None else 0
        status["warnings"] = warnings
        write_json(output / "status.json", status)
        if arrays_x:
            np.savez_compressed(output / "trajectory.npz", X=arrays_x, E_upper=arrays_e,
                                view_X=views_x, view_E_upper=views_e, model_key=key_data)
            np.savez_compressed(output / "final.npz", X=graph.X, E_upper=graph.E_upper,
                                update=completed, model_key=key_data)
            # Detached bounded witness: it never changes optimization or the
            # best raw molecule. No witness is an unknown distance.
            try:
                from stage3_diagnostics import assess_final
                write_json(output / "final_assessment.json", assess_final(graph))
            except Exception as error:
                warnings.append(f"Endpoint repair diagnostic unavailable: {type(error).__name__}: {error}")
                write_json(output / "status.json", status)
            from stage3_candidates import write_candidate_archive
            write_candidate_archive(output, config, states, list(candidates.values()), arrays_x, arrays_e,
                                    details_by_hard_id=candidate_details,
                                    compress=config.get("compress_jsonl", False))
        summary = summarize_run(config, states, list(candidates.values()))
        summary.update(status=status["status"], mock=mock, elapsed_seconds=status["elapsed_seconds"],
                       calibration_multiplier=calibration, teacher_queries=status["teacher_queries"])
        write_json(output / "summary.json", summary)
    return summary


class MockTeacher:
    """CPU transport-free fixture, deliberately labelled in every run."""
    def __init__(self):
        self.queries = 0

    def predict(self, graph, *, time):
        import numpy as np
        from mosaic_defog.graph import Prediction
        self.queries += 1
        x, e = np.asarray(graph.X), np.asarray(graph.E_upper)
        return Prediction(.95 * x + .05 / x.shape[-1], .95 * e + .05 / e.shape[-1], time,
                          {"mock": True})

    def verify_frozen(self):
        return {"mock": True, "queries": self.queries}


def run_bundle(plan_path, bundle_id, assets_path=None, *, mock=False, mock_updates=3):
    import jax
    import numpy as np
    import scipy
    from stage3_native import open_native, mock_native

    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if bool(plan.get("mock", False)) != bool(mock):
        raise ValueError("Mock and native execution require separate explicitly labelled plans")
    if mock and (isinstance(mock_updates, bool) or not isinstance(mock_updates, int) or mock_updates < 1 or mock_updates > 500):
        raise ValueError("mock_updates must be an integer from1to500")
    bundles = plan["bundles"]
    bundle = next((b for b in bundles if b["bundle_id"] == bundle_id), None)
    if bundle is None:
        raise ValueError(f"Unknown bundle: {bundle_id}")
    output = plan_path.parent
    if not mock and (not os.environ.get("LSB_JOBID") or not assets_path):
        raise RuntimeError("Native execution requires an allocated GPU and --assets")
    if assets_path and not Path(assets_path).is_file():
        raise FileNotFoundError(assets_path)
    ids = bundle["run_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Bundle contains duplicate run IDs")
    configs = [dict(r) for r in plan["runs"] if r["run_id"] in ids]
    if len(configs) != len(ids):
        raise ValueError("Bundle has missing or duplicate run IDs")
    for config in configs:
        rid = config["run_id"]
        if not isinstance(rid, str) or Path(rid).name != rid or rid in (".", ".."):
            raise ValueError("Unsafe run ID")
        for name, expected in (("updates", 500), ("learning_rate", 60), ("n_atoms", 25), ("native_weight", 1)):
            if config.get(name) != expected:
                raise ValueError(f"Run changed this campaign's fixed{name}={expected}")
    # Only this bundle's bounded remaining output is an admission check.
    # No account-wide provenance, unrelated jobs or global reserve multiplier.
    output_budget = len(configs) * 12 * 1024 ** 2
    if shutil.disk_usage(output).free < output_budget:
        raise OSError(f"Insufficient free space for this bundle's{output_budget}byte output allowance")
    code_digest = source_digest()
    # A permanent claim protects against duplicate/restarted writers. A failed
    # job gets an explicit new campaign/run ID; it never overwrites this one.
    claim = output / "bundles" / f"{bundle_id}.claim.json"
    with claim.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "job_id": os.environ.get("LSB_JOBID"),
                   "host": platform.node(), "claimed_utc": utc()}, stream)
    record = {"bundle_id": bundle_id, "status": "running", "started_utc": utc(),
              "host": platform.node(), "job_id": os.environ.get("LSB_JOBID"),
              "devices": [], "mock": mock, "arms": [],
              "gpu_profile": os.environ.get("MOSAIC_GPU_PROFILE"),
              "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "source_sha256": code_digest,
              "runtime": {"python": platform.python_version(), "jax": jax.__version__,
                          "numpy": np.__version__, "scipy": scipy.__version__},
              "phase": "bootstrap"}
    record_path = output / "bundles" / f"{bundle_id}.json"
    write_json(record_path, record)
    with _finish_bundle(record, record_path), ExitStack() as stack:
        record["devices"] = [{"platform": d.platform, "device_kind": d.device_kind, "id": d.id}
                             for d in jax.devices()]
        if mock:
            native, identity, finish = mock_native()
            teacher = MockTeacher()
        else:
            assets = json.loads(Path(assets_path).read_text(encoding="utf-8"))
            native, identity, finish = open_native(assets, ROOT)
            from mosaic_defog import DeFoGTeacher
            teacher = stack.enter_context(DeFoGTeacher(
                assets["teacher_source"], assets["teacher_checkpoint"], python=assets["teacher_python"],
                stderr=output / "bundles" / f"{bundle_id}.teacher.stderr",
            ))
        record["native"] = identity
        record["teacher_identity"] = getattr(teacher, "identity", {"mock": mock})
        record["phase"] = "arms"
        write_json(record_path, record)
        hard_cache = {}
        evaluators = make_evaluators(native)
        try:
            for config in configs:
                # Mock data live in their own plan/campaign, and carry a reduced
                # effective budget in config.json rather than pretending500ran.
                if mock:
                    config["planned_updates"] = config["updates"]
                    config["updates"] = mock_updates
                    config["mock"] = True
                config["compress_jsonl"] = not mock
                config["hardware"] = {"host": record["host"], "devices": record["devices"],
                                      "job_id": record["job_id"], "gpu_profile": record["gpu_profile"]}
                config["scorer_identity"] = identity
                config["runtime"] = record["runtime"]
                config["source_sha256"] = code_digest
                config["writer_bundle_id"] = bundle_id
                try:
                    summary = run_arm(config, output / "runs" / config["run_id"],
                                      native, teacher, hard_cache, mock=mock, evaluators=evaluators)
                    record["arms"].append({"run_id": config["run_id"], "status": summary["status"]})
                except Exception as error:
                    folder = output / "runs" / config["run_id"]
                    saved_path = folder / "config.json"
                    saved_config = json.loads(saved_path.read_text()) if saved_path.exists() else {}
                    if saved_config.get("writer_bundle_id") != bundle_id:
                        raise RuntimeError("Run output collision; refused to change another writer's artifacts") from error
                    arm_path = folder / "status.json"
                    arm_status = json.loads(arm_path.read_text()) if arm_path.exists() else {"updates_completed": 0}
                    arm_status.update(status="failed", finished_utc=utc(), mock=mock,
                                      error=f"{type(error).__name__}: {error}")
                    write_json(arm_path, arm_status)
                    record["arms"].append({"run_id": config["run_id"], "status": "failed",
                                           "error": f"{type(error).__name__}: {error}"})
                # A shared frozen-model failure stops the bundle; chemistry
                # failures/negative quality results never stop another arm.
                folder = output / "runs" / config["run_id"]
                try:
                    record["native_frozen"] = finish()
                    record["teacher_frozen"] = teacher.verify_frozen()
                except Exception as error:
                    if folder.is_dir():
                        arm_status_path = folder / "status.json"
                        arm_status = json.loads(arm_status_path.read_text()) if arm_status_path.exists() else {}
                        arm_status.update(status="failed", scientific_usable=False,
                                          exclusion_reason="frozen_model_or_context_check_failed",
                                          error=f"{type(error).__name__}: {error}")
                        write_json(arm_status_path, arm_status)
                        summary_path = folder / "summary.json"
                        if summary_path.exists():
                            summary = json.loads(summary_path.read_text())
                            summary.update(status="failed", scientific_usable=False,
                                           exclusion_reason=arm_status["exclusion_reason"])
                            write_json(summary_path, summary)
                    record["arms"][-1]["status"] = "failed"
                    raise
                if folder.is_dir() and (folder / "status.json").exists():
                    arm_status = json.loads((folder / "status.json").read_text())
                    arm_status.update(scientific_usable=True, frozen_checks_passed=True)
                    write_json(folder / "status.json", arm_status)
                write_json(record_path, record)
            record["status"] = "complete" if all(a["status"] == "complete" for a in record["arms"]) else "failed"
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["finished_utc"] = utc()
            write_json(record_path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Create fresh deterministic plan; no model/scheduler access")
    plan.add_argument("--config", type=Path, default=HOME / "configs/stage3-lr60-500.json")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--campaign-id")
    plan.add_argument("--max-arms", type=int, default=4)
    plan.add_argument("--wave", choices=("reference", "all"), default="all")
    plan.add_argument("--mock", action="store_true", help="CPU-fixture campaign, never mixed with native results")
    run = sub.add_parser("run-bundle")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--bundle-id", required=True)
    run.add_argument("--assets", type=Path)
    run.add_argument("--mock", action="store_true")
    run.add_argument("--mock-updates", type=int, default=3)
    collect = sub.add_parser("collect", help="Atomically rebuild plotting tables from saved runs")
    collect.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = prepare_plan(args.config, args.output, args.campaign_id, args.max_arms, args.wave, args.mock)
        print(json.dumps({"plan": str(args.output / "plan.json"),
                          "conditions": len(result["conditions"]), "runs": len(result["runs"]),
                          "bundles": len(result["bundles"])}))
    elif args.command == "run-bundle":
        result = run_bundle(args.plan, args.bundle_id, args.assets, mock=args.mock,
                            mock_updates=args.mock_updates)
        print(json.dumps(result))
        return 0 if result["status"] == "complete" else 1
    else:
        from stage3_results import write_campaign_tables
        print(json.dumps(write_campaign_tables(args.output), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
