"""Bounded real-Tiny acceptance checks, intended for allocated DTU HPC.

Importing this module does not load a checkpoint or prepare a molecule. The
runner records observed gates, never turns authored tests into passed evidence,
and retains failures. Its resume worker is a fresh process on the same allocated
node; it performs no SSH, downloads, queue submissions or environment installs.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .contracts import PreparedContext, validate_context
from .io import (
    TrajectoryConfig,
    file_sha256,
    identity_digest,
    load_context,
    run_trajectory,
    save_context,
    write_json,
)
from .optimizers import NonfiniteEvaluationError, evaluate_graph, project_graph
from .provenance import environment_identity, source_identity
from .state import (
    BOND_STATES,
    ELEMENTS,
    RelaxedGraph,
    bond_existence,
    full_bonds,
    mix_with_uniform,
    validate_graph,
)


def _digest_arrays(tree) -> str:
    """Hash real leaves, including shape/precision; static metadata is separate."""
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            continue
        array = np.ascontiguousarray(leaf)
        digest.update(
            json.dumps(
                [jax.tree_util.keystr(path), str(array.dtype), array.shape]
            ).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_output(output, *, raw_path: Path | None = None) -> dict:
    # The terminal +inf bin is native static metadata, not a generated output.
    generated = {
        "distogram_logits": output.distogram_logits,
        "trunk_s": output.trunk_state.s,
        "trunk_z": output.trunk_state.z,
    }
    if raw_path is not None:
        # Persist before checking so a failed native/conformer call retains its
        # actual nonfinite output, not just a reason string.
        _save_arrays(raw_path, **generated, distogram_bins=output.distogram_bins)
    shapes = {}
    for name, value in generated.items():
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise AssertionError(f"Nonfinite native generated {name}")
        shapes[name] = list(array.shape)
    return shapes


def _save_arrays(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle, **{name: np.asarray(value) for name, value in arrays.items()}
        )


def _close(actual, expected, *, atol, rtol, label) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{label}: nonfinite comparison")
    error = (
        float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
        if actual.size
        else 0.0
    )
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: maximum absolute error {error}")
    return error


def _perturb(graph, field, row, source, destination, mass):
    leaf = getattr(graph, field)
    changed = leaf.at[row, source].add(-mass).at[row, destination].add(mass)
    result = (
        RelaxedGraph(changed, graph.E_upper)
        if field == "X"
        else RelaxedGraph(graph.X, changed)
    )
    validate_graph(result)
    return result


def _objective(context, settings):
    from .losses import DistogramDiagnosticLoss, NativeContactLoss

    options = settings.get("loss", {})
    if context.protein_token_indices:
        return NativeContactLoss(
            cutoff_angstrom=options.get("contact_cutoff_angstrom", 8.0)
        )
    return DistogramDiagnosticLoss(
        ligand_pair=tuple(options.get("diagnostic_pair", (0, 1))),
        bin_index=options.get("diagnostic_bin", 0),
    )


def _forbidden_head(*args, **kwargs):
    raise AssertionError(
        "The Stage 0 trunk path invoked sampling/confidence or reloaded its model"
    )


def _cheap_guards(stack, model):
    import protenix.backend as backend

    stack.enter_context(
        patch.object(backend, "load_model", side_effect=_forbidden_head)
    )
    for name in ("sample_structures", "confidence_metrics"):
        stack.enter_context(
            patch.object(type(model.native_model), name, side_effect=_forbidden_head)
        )


def _representation(graph, settings):
    validate_graph(graph)
    projected = project_graph(RelaxedGraph(graph.X - 0.4, graph.E_upper + 0.4))
    validate_graph(projected)
    bonds = np.asarray(full_bonds(projected))
    np.testing.assert_array_equal(bonds, bonds.swapaxes(0, 1))
    np.testing.assert_array_equal(
        np.diag(np.asarray(bond_existence(projected))), np.zeros(graph.n_atoms)
    )
    fixture = _load_fixture(settings)
    if graph.n_atoms != fixture["n_heavy_atoms"]:
        raise AssertionError("Native fixture atom count disagrees with its manifest")
    elements = [ELEMENTS[index] for index in np.asarray(graph.X).argmax(-1)]
    if elements != fixture["element_order"]:
        raise AssertionError("Native fixture atom ordering disagrees with its manifest")
    pair_lookup = {
        (i, j): index
        for index, (i, j) in enumerate(
            ((i, j) for i in range(graph.n_atoms) for j in range(i + 1, graph.n_atoms))
        )
    }
    assignments = np.zeros(len(graph.E_upper), dtype=np.int32)
    for i, j, state in fixture["bond_assignments"]:
        assignments[pair_lookup[(i, j)]] = BOND_STATES.index(state)
    np.testing.assert_array_equal(np.asarray(graph.E_upper).argmax(-1), assignments)
    return {
        "n_atoms": graph.n_atoms,
        "fixed_occupancy": True,
        "element_order": elements,
        "pair_order": "lexicographic i<j",
        "fixture_sha256": file_sha256(_fixture_path(settings)),
    }


def _fixture_path(settings) -> Path:
    path = Path(settings["fixture"])
    return path if path.is_absolute() else _source_root(settings) / path


def _load_fixture(settings):
    return json.loads(_fixture_path(settings).read_text(encoding="utf-8"))


def _source_root(settings) -> Path:
    return Path(
        settings.get("source_root", Path(__file__).resolve().parents[3])
    ).resolve()


def _retained_parity(context, graph):
    from .features import inject_graph
    from .native_hook import GEOMETRY_FEATURE

    if context.native_features is None:
        raise AssertionError("Native hard/conformer features were not retained")
    injected = inject_graph(context, graph)
    intentional = {"ref_pos", "ref_atom_name_chars", "has_frame", "frame_atom_index"}
    retained = []
    for name, native in context.native_features.items():
        if name not in intentional:
            np.testing.assert_array_equal(injected[name], native, err_msg=name)
            retained.append(name)
    atom_map = np.asarray(context.base_features["atom_to_token_idx"])
    protein_atoms = np.flatnonzero(np.isin(atom_map, context.protein_token_indices))
    for name in ("ref_pos", "ref_atom_name_chars"):
        np.testing.assert_array_equal(
            np.asarray(injected[name])[protein_atoms],
            np.asarray(context.native_features[name])[protein_atoms],
        )
    for name in ("has_frame", "frame_atom_index"):
        indices = list(context.protein_token_indices)
        np.testing.assert_array_equal(
            np.asarray(injected[name])[indices],
            np.asarray(context.native_features[name])[indices],
        )
    np.testing.assert_array_equal(
        np.asarray(injected[GEOMETRY_FEATURE])[list(context.ligand_atom_indices)], 0
    )
    validate_context(context)
    return {
        "retained_native_fields": sorted(retained),
        "intentional_ligand_ablations": sorted(intentional),
        "protein_blocks_unchanged": True,
        "occupancy_preserved": True,
    }


def _directions(evaluation, settings):
    minimum = settings["validation"]["nonzero_tangent_atol"]
    result = []
    for field, source, destination in (("X", 0, 1), ("E_upper", 0, 1)):
        gradient = np.asarray(getattr(evaluation.gradients, field))
        tangents = gradient[:, destination] - gradient[:, source]
        row = int(np.argmax(np.abs(tangents)))
        derivative = float(tangents[row])
        if not np.isfinite(derivative) or abs(derivative) <= minimum:
            raise AssertionError(f"No nonzero {field} identity/existence tangent")
        result.append(
            {
                "field": field,
                "row": row,
                "source": source,
                "destination": destination,
                "autodiff": derivative,
            }
        )
    return result


def _derivatives(
    model, context, graph, evaluation, directions, key, settings, directory
):
    tolerances = settings["validation"]
    loss = model.build_loss(
        context=context, loss=_objective(context, settings), recycling_steps=1
    )
    records = []
    for direction in directions:
        sweep = []
        for epsilon in tolerances["epsilons"]:
            plus = _perturb(
                graph,
                direction["field"],
                direction["row"],
                direction["source"],
                direction["destination"],
                epsilon,
            )
            minus = _perturb(
                graph,
                direction["field"],
                direction["row"],
                direction["source"],
                direction["destination"],
                -epsilon,
            )
            plus_value, plus_aux = loss(plus, key=key)
            minus_value, minus_aux = loss(minus, key=key)
            for label, state, value, aux in (
                ("plus", plus, plus_value, plus_aux),
                ("minus", minus, minus_value, minus_aux),
            ):
                generated = {
                    name: values
                    for name, values in aux.items()
                    if isinstance(values, (jax.Array, np.ndarray))
                }
                _save_arrays(
                    directory
                    / f"{direction['field']}_{direction['row']}_{epsilon:g}_{label}.npz",
                    X=state.X,
                    E_upper=state.E_upper,
                    value=value,
                    **generated,
                )
                if any(
                    not np.isfinite(np.asarray(values)).all()
                    for values in generated.values()
                ):
                    raise AssertionError(
                        "Nonfinite perturbed generated native quantity"
                    )
            if (
                not np.isfinite(np.asarray(plus_value)).all()
                or not np.isfinite(np.asarray(minus_value)).all()
            ):
                raise AssertionError("Nonfinite perturbed scalar loss")
            if (
                not np.isfinite(np.asarray(plus_aux["distogram_logits"])).all()
                or not np.isfinite(np.asarray(minus_aux["distogram_logits"])).all()
            ):
                raise AssertionError("Nonfinite perturbed native output")
            finite_difference = (float(plus_value) - float(minus_value)) / (2 * epsilon)
            absolute = abs(finite_difference - direction["autodiff"])
            relative = absolute / max(
                abs(finite_difference), abs(direction["autodiff"]), 1e-12
            )
            agrees = bool(
                np.isclose(
                    finite_difference,
                    direction["autodiff"],
                    atol=tolerances["derivative_atol"],
                    rtol=tolerances["derivative_rtol"],
                )
            )
            sweep.append(
                {
                    "epsilon": epsilon,
                    "finite_difference": finite_difference,
                    "absolute_error": absolute,
                    "relative_error": relative,
                    "agrees": agrees,
                }
            )
        # Require a stable range, not one selected epsilon that happens to fit.
        if sum(entry["agrees"] for entry in sweep) < 2:
            raise AssertionError(
                f"No two-epsilon derivative agreement for {direction['field']}: {sweep}"
            )
        records.append(direction | {"epsilon_sweep": sweep})
    order = _perturb(graph, "E_upper", 0, 1, 2, 0.01)
    order_output = model.forward_trunk(context=context, graph=order, key=key)
    _finite_output(order_output, raw_path=directory / "bond_order_null.npz")
    np.testing.assert_array_equal(bond_existence(order), bond_existence(graph))
    null_error = _close(
        order_output.distogram_logits,
        evaluation.aux["distogram_logits"],
        atol=tolerances["output_atol"],
        rtol=tolerances["output_rtol"],
        label="bond-order null direction",
    )
    tangent = float(
        evaluation.gradients.E_upper[0, 2] - evaluation.gradients.E_upper[0, 1]
    )
    if abs(tangent) > tolerances["nonzero_tangent_atol"]:
        raise AssertionError(
            "The existence-only objective acquired bond-order guidance"
        )
    return {
        "selection": "largest absolute C->N / NO_BOND->SINGLE raw tangent, deterministic",
        "directions": records,
        "bond_order_null_derivative": tangent,
        "bond_order_output_error": null_error,
    }


def _context_substitution(model, context, graph, directions, key, settings, directory):
    before = _digest_arrays((context.base_features, context.native_features))
    original = model.forward_trunk(context=context, graph=graph, key=key)
    changed = graph
    for direction in directions:
        leaf = np.asarray(getattr(changed, direction["field"]))
        mass = min(0.01, float(leaf[direction["row"], direction["source"]]) / 4)
        changed = _perturb(
            changed,
            direction["field"],
            direction["row"],
            direction["source"],
            direction["destination"],
            mass,
        )
    alternate = model.forward_trunk(context=context, graph=changed, key=key)
    replay = model.forward_trunk(context=context, graph=graph, key=key)
    for label, output in (("A", original), ("B", alternate), ("A_replay", replay)):
        _finite_output(output, raw_path=directory / f"{label}.npz")
    tolerances = settings["validation"]
    replay_error = _close(
        replay.distogram_logits,
        original.distogram_logits,
        atol=tolerances["output_atol"],
        rtol=tolerances["output_rtol"],
        label="A-B-A",
    )
    if np.array_equal(
        np.asarray(original.distogram_logits), np.asarray(alternate.distogram_logits)
    ):
        raise AssertionError(
            "Identity/existence perturbations did not change native output"
        )
    atoms = jnp.asarray(context.ligand_atom_indices)
    replacement = (
        context.base_features["ref_pos"]
        .at[atoms]
        .set(
            jnp.arange(context.n_atoms * 3, dtype=jnp.float32).reshape(
                context.n_atoms, 3
            )
            + 37
        )
    )
    replaced = eqx.tree_at(
        lambda c: c.base_features,
        context,
        context.base_features | {"ref_pos": replacement},
    )
    substituted = model.forward_trunk(context=replaced, graph=graph, key=key)
    _finite_output(substituted, raw_path=directory / "unused_conformer.npz")
    conformer_error = _close(
        substituted.distogram_logits,
        original.distogram_logits,
        atol=tolerances["output_atol"],
        rtol=tolerances["output_rtol"],
        label="unused ligand conformer",
    )
    if before != _digest_arrays((context.base_features, context.native_features)):
        raise AssertionError("Prepared native/base features were mutated")
    return {
        "A_B_A_error": replay_error,
        "changed_output": True,
        "unused_ligand_conformer_error": conformer_error,
        "prepared_feature_digest": before,
        "prepared_features_unchanged": True,
    }


def _protein_regression(model, key, settings, directory):
    from mosaic.losses.protenix import get_trunk_state
    from mosaic.models.protenix import Protenix
    from mosaic.structure_prediction import TargetChain
    from .native_hook import GEOMETRY_FEATURE

    sequence = _load_fixture(settings)["protein_sequence"]
    wrapper = Protenix(protenix=model.native_model, default_sample_steps=2)
    features, _ = wrapper.target_only_features(
        [TargetChain(sequence=sequence, use_msa=False)]
    )
    geometry = jnp.ones(features["ref_pos"].shape[0], dtype=jnp.float32)
    outputs = []
    for native_model, native_features in (
        (wrapper.protenix, features),
        (model.model, features),
        (model.model, features | {GEOMETRY_FEATURE: geometry}),
    ):
        _, trunk = get_trunk_state(
            model=native_model,
            features=native_features,
            initial_recycling_state=None,
            recycling_steps=1,
            key=key,
        )
        logits = native_model.distogram_head(trunk.z)
        _save_arrays(
            directory / f"protein_path_{len(outputs)}.npz",
            distogram_logits=logits,
            trunk_s=trunk.s,
            trunk_z=trunk.z,
        )
        if any(
            not np.isfinite(np.asarray(value)).all()
            for value in (logits, trunk.s, trunk.z)
        ):
            raise AssertionError("Nonfinite protein-only regression output")
        outputs.append(logits)
    tolerance = settings["validation"]
    errors = [
        _close(
            logits,
            outputs[0],
            atol=tolerance["output_atol"],
            rtol=tolerance["output_rtol"],
            label="native protein API regression",
        )
        for logits in outputs[1:]
    ]
    return {
        "protein_sequence": sequence,
        "native_wrapper_vs_hook_errors": errors,
        "no_geometry_override_delegates_native": True,
        "protein_reference_geometry_retained": True,
    }


def _save_result(path, result):
    arrays = {
        "X": result.graph.X,
        "E_upper": result.graph.E_upper,
        "next_key": jax.random.key_data(result.next_key),
    }
    if result.best_graph is not None:
        arrays.update(
            best_X=result.best_graph.X, best_E_upper=result.best_graph.E_upper
        )
    _save_arrays(path, **arrays)
    return {
        "complete": result.complete,
        "update_step": result.update_step,
        "best_value": result.best_value,
        "next_key_impl": str(jax.random.key_impl(result.next_key)),
        "arrays_sha256": file_sha256(path),
    }


def _restart(model, context, graph, key, directory, settings, source, environment):
    cache = Path(settings["cache_dir"]).resolve()
    context_directory = directory / "sealed_context"
    context_receipt = save_context(context_directory, context)
    identities = {
        "model": dict(model.identities),
        "context": dict(context.identities),
        "prepared_context": context_receipt,
        "source": source,
        "environment": environment,
        "fixture_sha256": file_sha256(_fixture_path(settings)),
    }
    config = TrajectoryConfig(**settings["trajectory"])
    if config.steps < 5 or config.steps > 10:
        raise ValueError(
            "Stage 0 restart validation requires a bounded 5..10-step trajectory"
        )
    loss = model.build_loss(
        context=context, loss=_objective(context, settings), recycling_steps=1
    )
    uninterrupted = run_trajectory(
        loss,
        graph,
        key,
        directory=directory / "uninterrupted",
        config=config,
        identities=identities,
        resolved_config=settings,
    )
    run_trajectory(
        loss,
        graph,
        key,
        directory=directory / "resumed",
        config=config,
        identities=identities,
        resolved_config=settings,
        max_updates=2,
    )
    graph_path = directory / "initial_graph.npz"
    _save_arrays(graph_path, X=graph.X, E_upper=graph.E_upper)
    request = {
        "cache_dir": str(cache),
        "context_directory": str(context_directory.resolve()),
        "context_receipt": context_receipt,
        "graph_path": str(graph_path.resolve()),
        "graph_sha256": file_sha256(graph_path),
        "source_root": str(_source_root(settings)),
        "source": source,
        "environment": environment,
        "key_data": np.asarray(jax.random.key_data(key)).tolist(),
        "key_impl": str(jax.random.key_impl(key)),
        "trajectory": config.as_dict(),
        "identities": identities,
        "settings": settings,
        "directory": str((directory / "resumed").resolve()),
    }
    request_path = directory / "resume_request.json"
    write_json(request_path, request)
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONPATH"] = (
        str(_source_root(settings) / "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    command = [
        sys.executable,
        "-m",
        "mosaic.small_molecule.validation",
        "--resume-worker",
        str(request_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=_source_root(settings),
        env=env,
        text=True,
        capture_output=True,
        timeout=settings["validation"].get("resume_timeout_seconds", 1800),
        check=False,
    )
    (directory / "resume_worker.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (directory / "resume_worker.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(
            f"Fresh-process resume failed with exit {completed.returncode}; retain worker logs"
        )
    receipt_path = directory / "resumed" / "resume_worker.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    arrays_path = directory / "resumed" / "resume_worker.npz"
    if receipt["arrays_sha256"] != file_sha256(arrays_path) or not receipt["complete"]:
        raise AssertionError(
            "Fresh-process continuation did not persist authenticated completion"
        )
    tolerance = settings["validation"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        errors = {
            name: _close(
                arrays[name],
                getattr(uninterrupted.graph, name),
                atol=tolerance["resume_atol"],
                rtol=tolerance["resume_rtol"],
                label=f"resumed {name}",
            )
            for name in ("X", "E_upper")
        }
        np.testing.assert_array_equal(
            arrays["next_key"], jax.random.key_data(uninterrupted.next_key)
        )
        if uninterrupted.best_graph is not None:
            for name in ("X", "E_upper"):
                _close(
                    arrays["best_" + name],
                    getattr(uninterrupted.best_graph, name),
                    atol=tolerance["resume_atol"],
                    rtol=tolerance["resume_rtol"],
                    label=f"resumed best {name}",
                )
    if receipt["next_key_impl"] != str(jax.random.key_impl(uninterrupted.next_key)):
        raise AssertionError("Resumed RNG implementation changed")
    if receipt["update_step"] != config.steps or not uninterrupted.complete:
        raise AssertionError("Restart comparison has not reached its sealed endpoint")
    _close(
        receipt["best_value"],
        uninterrupted.best_value,
        atol=tolerance["resume_atol"],
        rtol=tolerance["resume_rtol"],
        label="resumed best score",
    )
    traces = [
        json.loads(line)
        for line in (directory / "uninterrupted" / "trace.jsonl")
        .read_text()
        .splitlines()
    ]
    resumed_traces = [
        json.loads(line)
        for line in (directory / "resumed" / "trace.jsonl").read_text().splitlines()
    ]
    if len(traces) != config.steps + 1 or len(resumed_traces) != len(traces):
        raise AssertionError("Restart trace lacks initial/final evaluation receipts")
    for expected, actual in zip(traces, resumed_traces, strict=True):
        for field in ("evaluated_step", "evaluation", "key_data", "key_impl", "final"):
            if expected[field] != actual[field]:
                raise AssertionError(f"Restart trace changed {field}")
        _close(
            actual["loss"],
            expected["loss"],
            atol=tolerance["resume_atol"],
            rtol=tolerance["resume_rtol"],
            label="resumed evaluated score",
        )
    return {
        "steps": config.steps,
        "fresh_process": True,
        "command": command,
        "worker_returncode": completed.returncode,
        "graph_errors": errors,
        "same_prepared_context_bytes": context_receipt,
        "next_unused_rng_matches": True,
        "evaluated_state_scores_match": True,
        "source_identity": source["execution_content_sha256"],
    }


def _failure(directory, error, gate):
    receipt = {
        "gate": gate,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "retained_npz_artifacts": [
            str(path.relative_to(directory))
            for path in sorted(directory.rglob("*.npz"))
        ],
    }
    previous = directory / "failure.json"
    if previous.is_file():
        # A resumed trajectory may already have a detailed IO failure receipt.
        # Preserve its counter/checkpoint/raw-array references in this envelope.
        receipt["previous_failure_receipt"] = json.loads(
            previous.read_text(encoding="utf-8")
        )
    if isinstance(error, NonfiniteEvaluationError):
        arrays = {
            "X": error.graph.X,
            "E_upper": error.graph.E_upper,
            "key_data": jax.random.key_data(error.key),
        }
        if error.value is not None:
            arrays["value"] = error.value
        if error.gradients is not None:
            arrays.update(
                gradient_X=error.gradients.X, gradient_E_upper=error.gradients.E_upper
            )
        if error.proposal is not None:
            arrays.update(
                proposal_X=error.proposal.X, proposal_E_upper=error.proposal.E_upper
            )
        aux_paths = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(error.aux):
            if isinstance(leaf, (jax.Array, np.ndarray)):
                name = f"aux_{len(aux_paths):04d}"
                arrays[name] = leaf
                aux_paths[name] = jax.tree_util.keystr(path)
        _save_arrays(directory / "failed_evaluation.npz", **arrays)
        receipt.update(
            raw_artifact="failed_evaluation.npz",
            aux_paths=aux_paths,
            nonfinite_paths=error.nonfinite_paths,
            key_impl=str(jax.random.key_impl(error.key)),
        )
    if isinstance(error, subprocess.TimeoutExpired):
        for label, value in (("stdout", error.stdout), ("stderr", error.stderr)):
            if value:
                text = (
                    value.decode(errors="replace")
                    if isinstance(value, bytes)
                    else value
                )
                (directory / f"resume_worker.{label}.log").write_text(
                    text, encoding="utf-8"
                )
    write_json(directory / "failure.json", receipt)


def validate_pipeline(
    model,
    context: PreparedContext,
    hard_graph: RelaxedGraph,
    *,
    key: jax.Array,
    output_directory: str | Path,
    settings: dict,
) -> dict:
    """Run A-G for one real context, write evidence, and return explicit status.

    ``settings`` is the full resolved example JSON plus ``cache_dir`` and an
    optional ``source_root``. Missing assets, prerequisites or failed gates are
    failures, never skipped passes. The caller must check ``status`` and run
    both ligand-only and protein-plus-ligand modes before claiming acceptance.
    No chemistry-validity or scientific objective-improvement gate is imposed.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "mode": "protein_ligand" if context.protein_token_indices else "ligand_only",
        "gates": {letter: {"status": "pending"} for letter in "ABCDEFG"},
        "settings": settings,
        "model": dict(model.identities),
        "context": dict(context.identities),
        "key_impl": str(jax.random.key_impl(key)),
        "key_data": np.asarray(jax.random.key_data(key)).tolist(),
        "interpretation": "real native engineering acceptance; no binding or chemical-validity claim",
    }
    write_json(directory / "validation.json", report)
    active_gate = "A"
    try:
        validate_context(context)
        if settings.get("recycling_steps") != 1:
            raise ValueError("Acceptance supports exactly one final recycle")
        epsilons = settings["validation"]["epsilons"]
        if len(set(epsilons)) < 2 or any(
            not np.isfinite(epsilon) or epsilon <= 0 for epsilon in epsilons
        ):
            raise ValueError(
                "Declare at least two distinct positive finite-difference epsilons"
            )
        if not Path(settings["cache_dir"]).is_dir():
            raise ValueError(
                "Acceptance requires the existing pinned native cache for its fresh worker"
            )
        report["source"] = source_identity(_source_root(settings))
        report["environment"] = environment_identity()
        report["config_sha256"] = identity_digest(settings)
        write_json(directory / "validation.json", report)
        before_parameters = _digest_arrays((model.native_model, model.model))
        interior = mix_with_uniform(hard_graph, settings["initialization"]["mixing"])
        save_context(directory / "inputs", context)
        _save_arrays(
            directory / "inputs" / "graphs.npz",
            hard_X=hard_graph.X,
            hard_E_upper=hard_graph.E_upper,
            X=interior.X,
            E_upper=interior.E_upper,
        )

        def passed(letter, evidence):
            report["gates"][letter] = {"status": "passed", "evidence": evidence}
            write_json(directory / "validation.json", report)

        passed("A", _representation(hard_graph, settings))
        # Acceptance may follow another trajectory in this process. Clear
        # compiled routes before guards so a cached trace cannot hide a head
        # call that would otherwise fail instrumentation. Model leaves stay
        # resident and are authenticated again at G.
        jax.clear_caches()
        clear_equinox = getattr(eqx, "clear_caches", None)
        if callable(clear_equinox):
            clear_equinox()
        with ExitStack() as guards:
            _cheap_guards(guards, model)
            active_gate = "B"
            native = model.forward_native_trunk(context=context, key=key)
            shapes = _finite_output(native, raw_path=directory / "native_hard.npz")
            passed(
                "B",
                {
                    "generated_shapes": shapes,
                    "native_conformer_reference": True,
                    "raw_artifact": "native_hard.npz",
                },
            )
            active_gate = "C"
            parity = _retained_parity(context, hard_graph)
            hard_output = model.forward_trunk(
                context=context, graph=hard_graph, key=key
            )
            _finite_output(hard_output, raw_path=directory / "geometry_off_hard.npz")
            parity["native_vs_geometry_off_output_difference"] = float(
                np.max(
                    np.abs(
                        np.asarray(native.distogram_logits)
                        - np.asarray(hard_output.distogram_logits)
                    )
                )
            )
            parity["difference_interpretation"] = (
                "intentional geometry and ligand-name ablation; not full-output parity"
            )
            passed("C", parity)
            active_gate = "D"
            loss = model.build_loss(
                context=context, loss=_objective(context, settings), recycling_steps=1
            )
            evaluation = evaluate_graph(loss, interior, key)
            _finite_output(
                model.forward_trunk(context=context, graph=interior, key=key),
                raw_path=directory / "interior_forward.npz",
            )
            directions = _directions(evaluation, settings)
            _save_arrays(
                directory / "raw_derivatives.npz",
                X=interior.X,
                E_upper=interior.E_upper,
                value=evaluation.value,
                gradient_X=evaluation.gradients.X,
                gradient_E_upper=evaluation.gradients.E_upper,
                distogram_logits=evaluation.aux["distogram_logits"],
            )
            passed(
                "D",
                {
                    "raw_artifact": "raw_derivatives.npz",
                    "nonzero_tangents": directions,
                    "graph_only_differentiation": True,
                    "loss": float(evaluation.value),
                },
            )
            active_gate = "E"
            derivatives = _derivatives(
                model,
                context,
                interior,
                evaluation,
                directions,
                key,
                settings,
                directory / "perturbations",
            )
            derivatives["context_replay"] = _context_substitution(
                model,
                context,
                interior,
                directions,
                key,
                settings,
                directory / "replay",
            )
            passed("E", derivatives)
            active_gate = "F"
            passed(
                "F",
                _restart(
                    model,
                    context,
                    interior,
                    key,
                    directory / "restart",
                    settings,
                    report["source"],
                    report["environment"],
                ),
            )
            active_gate = "G"
            regression = _protein_regression(
                model, key, settings, directory / "protein_regression"
            )
            after_parameters = _digest_arrays((model.native_model, model.model))
            if after_parameters != before_parameters:
                raise AssertionError(
                    "Frozen native/model parameter leaves changed during validation"
                )
            if source_identity(_source_root(settings)) != report["source"]:
                raise AssertionError(
                    "Execution source contents changed during validation"
                )
            if environment_identity() != report["environment"]:
                raise AssertionError(
                    "Execution environment/device/driver changed during validation"
                )
            passed(
                "G",
                regression
                | {
                    "frozen_parameter_digest": after_parameters,
                    "same_resident_object": True,
                    "additional_model_loads": 0,
                    "sampling_calls": 0,
                    "confidence_calls": 0,
                    "jax_compilation_cache_cleared": True,
                    "equinox_cache_cleared": callable(clear_equinox),
                    "guard_scope": "all parent acceptance forwards after compilation cache clear",
                },
            )
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["gates"][active_gate] = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        _failure(directory, error, active_gate)
    write_json(directory / "validation.json", report)
    return report


def _resume_worker(request_path):
    """Fresh process restores sealed context bytes and authenticates bindings."""
    from .protenix import FrozenProtenixTiny

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    directory = Path(request["directory"])
    try:
        if source_identity(request["source_root"]) != request["source"]:
            raise ValueError("Fresh-process source contents changed")
        if environment_identity() != request["environment"]:
            raise ValueError("Fresh-process environment/device/driver changed")
        if file_sha256(request["graph_path"]) != request["graph_sha256"]:
            raise ValueError("Fresh-process initial graph identity changed")
        context = load_context(request["context_directory"], request["context_receipt"])
        with np.load(request["graph_path"], allow_pickle=False) as arrays:
            graph = RelaxedGraph(
                jnp.asarray(arrays["X"]), jnp.asarray(arrays["E_upper"])
            )
        validate_graph(graph)
        model = FrozenProtenixTiny.load(cache_dir=request["cache_dir"])
        if (
            model.identities != request["identities"]["model"]
            or context.identities != request["identities"]["context"]
        ):
            raise ValueError("Fresh-process model/context identity changed")
        key = jax.random.wrap_key_data(
            jnp.asarray(request["key_data"], dtype=jnp.uint32), impl=request["key_impl"]
        )
        loss = model.build_loss(
            context=context,
            loss=_objective(context, request["settings"]),
            recycling_steps=1,
        )
        with ExitStack() as guards:
            _cheap_guards(guards, model)
            result = run_trajectory(
                loss,
                graph,
                key,
                directory=directory,
                config=TrajectoryConfig(**request["trajectory"]),
                identities=request["identities"],
                resolved_config=request["settings"],
                resume=True,
            )
        receipt = _save_result(directory / "resume_worker.npz", result)
        receipt["pid"] = os.getpid()
        write_json(directory / "resume_worker.json", receipt)
    except Exception as error:
        _failure(directory, error, "fresh-process resume")
        raise


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Internal allocated-HPC Stage 0 resume worker"
    )
    parser.add_argument("--resume-worker", required=True)
    args = parser.parse_args(argv)
    _resume_worker(args.resume_worker)


if __name__ == "__main__":
    main()
