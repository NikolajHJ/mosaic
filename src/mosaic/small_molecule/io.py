"""Detached readouts, sealed inputs, and restartable graph trajectories.

Only the supplied relaxed graph enters the loss. RDKit is imported at the
readout boundary and its successes or failures never become model inputs.
Checkpoints contain the next unused key, not a reconstructed seed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .contracts import (
    CHECKPOINT_VERSION,
    PreparedContext,
    contract_manifest,
    validate_context,
)
from .optimizers import NonfiniteEvaluationError, evaluate_graph, projected_step
from .state import (
    BOND_STATES,
    ELEMENTS,
    RelaxedGraph,
    graph_diagnostics,
    validate_graph,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        return _json_value(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported artifact metadata type: {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> None:
    """Replace a receipt atomically; nonfinite failures use explicit strings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_value(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_graph(path: str | Path, graph: RelaxedGraph) -> str:
    validate_graph(graph)
    path = Path(path)
    _write_npz(path, {"X": np.asarray(graph.X), "E_upper": np.asarray(graph.E_upper)})
    return file_sha256(path)


def load_graph(path: str | Path, expected_sha256: str) -> RelaxedGraph:
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Graph artifact identity mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        graph = RelaxedGraph(
            _restore_array(arrays["X"]), _restore_array(arrays["E_upper"])
        )
    validate_graph(graph)
    return graph


def _encode_tree(value, arrays, prefix="a"):
    """Safe JSON/NPZ codec: no pickle, imports, or arbitrary class construction."""
    if isinstance(value, (jax.Array, np.ndarray)):
        name = f"{prefix}_{len(arrays):05d}"
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            arrays[name] = np.asarray(jax.random.key_data(value))
            return {"rng_array": name, "impl": str(jax.random.key_impl(value))}
        arrays[name] = np.asarray(value)
        return {"array": name}
    if isinstance(value, dict):
        return {
            "dict": {str(k): _encode_tree(v, arrays, prefix) for k, v in value.items()}
        }
    if isinstance(value, (tuple, list)):
        return {
            "tuple" if isinstance(value, tuple) else "list": [
                _encode_tree(v, arrays, prefix) for v in value
            ]
        }
    return {"scalar": _json_value(value)}


def _decode_tree(node, arrays):
    if "array" in node:
        return _restore_array(arrays[node["array"]])
    if "rng_array" in node:
        return jax.random.wrap_key_data(
            jnp.asarray(arrays[node["rng_array"]]), impl=node["impl"]
        )
    if "dict" in node:
        return {k: _decode_tree(v, arrays) for k, v in node["dict"].items()}
    if "tuple" in node or "list" in node:
        values = [_decode_tree(v, arrays) for v in node.get("tuple", node.get("list"))]
        return tuple(values) if "tuple" in node else values
    return node["scalar"]


def _restore_array(array):
    restored = jnp.asarray(array)
    if np.dtype(restored.dtype) != array.dtype:
        raise ValueError(
            "Restoring an artifact would change its precision; match JAX x64 settings"
        )
    return restored


def save_context(directory: str | Path, context: PreparedContext) -> dict[str, str]:
    """Seal the actual prepared features, including the native hard conformer.

    Native SMILES import may produce a different conformer on another process;
    resume restores these bytes rather than re-running preparation.
    """
    validate_context(context)
    directory = Path(directory)
    arrays = {}
    metadata = {
        "contract": contract_manifest(),
        "base_features": _encode_tree(context.base_features, arrays),
        "native_features": _encode_tree(context.native_features, arrays),
        "reference_geometry_mask": _encode_tree(
            context.reference_geometry_mask, arrays
        ),
        "ligand_atom_indices": context.ligand_atom_indices,
        "ligand_token_indices": context.ligand_token_indices,
        "protein_token_indices": context.protein_token_indices,
        "atom_names": context.atom_names,
        "geometry_mode": context.geometry_mode,
        "identities": context.identities,
    }
    _write_npz(directory / "prepared_context.npz", arrays)
    write_json(directory / "prepared_context.json", metadata)
    return {
        name: file_sha256(directory / name)
        for name in ("prepared_context.npz", "prepared_context.json")
    }


def load_context(directory: str | Path, expected: dict[str, str]) -> PreparedContext:
    directory = Path(directory)
    for name in ("prepared_context.npz", "prepared_context.json"):
        if expected.get(name) != file_sha256(directory / name):
            raise ValueError(f"Prepared context identity mismatch: {name}")
    metadata = json.loads(
        (directory / "prepared_context.json").read_text(encoding="utf-8")
    )
    if metadata["contract"] != contract_manifest():
        raise ValueError("Prepared context contract mismatch")
    with np.load(directory / "prepared_context.npz", allow_pickle=False) as arrays:
        context = PreparedContext(
            base_features=_decode_tree(metadata["base_features"], arrays),
            native_features=_decode_tree(metadata["native_features"], arrays),
            reference_geometry_mask=_decode_tree(
                metadata["reference_geometry_mask"], arrays
            ),
            ligand_atom_indices=tuple(metadata["ligand_atom_indices"]),
            ligand_token_indices=tuple(metadata["ligand_token_indices"]),
            protein_token_indices=tuple(metadata["protein_token_indices"]),
            atom_names=tuple(metadata["atom_names"]),
            geometry_mode=metadata["geometry_mode"],
            identities=metadata["identities"],
        )
    validate_context(context)
    return context


def hard_readout(graph: RelaxedGraph) -> dict[str, Any]:
    """Keep raw argmax assignments before attempting unmodified sanitization."""
    validate_graph(graph)
    elements = np.asarray(graph.X).argmax(axis=-1).tolist()
    bonds = np.asarray(graph.E_upper).argmax(axis=-1).tolist()
    pairs = [(i, j) for i in range(graph.n_atoms) for j in range(i + 1, graph.n_atoms)]
    result = {
        "element_indices": elements,
        "bond_upper_indices": bonds,
        "elements": [ELEMENTS[i] for i in elements],
        "pairs": pairs,
        "policy": "raw argmax; no repair, no fragment removal",
        "sanitized": False,
        "canonical_smiles": None,
        "conversion_error": None,
        "disconnected": None,
        "fragment_count": None,
    }
    try:
        from rdkit import Chem

        molecule = Chem.RWMol()
        for index in elements:
            molecule.AddAtom(Chem.Atom(ELEMENTS[index]))
        native_bonds = (
            None,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        )
        for (i, j), category in zip(pairs, bonds, strict=True):
            if category:
                molecule.AddBond(i, j, native_bonds[category])
                if BOND_STATES[category] == "AROMATIC":
                    molecule.GetAtomWithIdx(i).SetIsAromatic(True)
                    molecule.GetAtomWithIdx(j).SetIsAromatic(True)
        molecule = molecule.GetMol()
        result["fragment_count"] = len(Chem.GetMolFrags(molecule, sanitizeFrags=False))
        result["disconnected"] = result["fragment_count"] > 1
        Chem.SanitizeMol(molecule)
        result["canonical_smiles"] = Chem.MolToSmiles(molecule, canonical=True)
        result["sanitized"] = True
    except Exception as error:
        # A missing converter is also retained honestly; callers never repair.
        result["conversion_error"] = f"{type(error).__name__}: {error}"
    return result


@dataclass(frozen=True)
class TrajectoryConfig:
    steps: int = 5
    eta_X: float = 0.01
    eta_E: float = 0.01
    rng_policy: str = "fixed"
    simplex_atol: float = 1e-6
    readout_every: int = 1

    def validate(self):
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps < 0
        ):
            raise ValueError("steps must be a nonnegative integer")
        if self.rng_policy not in ("fixed", "split"):
            raise ValueError("rng_policy must be fixed or split")
        if (
            isinstance(self.readout_every, bool)
            or not isinstance(self.readout_every, int)
            or self.readout_every < 1
        ):
            raise ValueError("readout_every must be a positive integer")
        for name in ("eta_X", "eta_E", "simplex_atol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def as_dict(self):
        self.validate()
        return {
            name: getattr(self, name)
            for name in (
                "steps",
                "eta_X",
                "eta_E",
                "rng_policy",
                "simplex_atol",
                "readout_every",
            )
        }


@dataclass(frozen=True)
class TrajectoryResult:
    graph: RelaxedGraph
    best_graph: RelaxedGraph | None
    best_value: float | None
    update_step: int
    next_key: jax.Array
    complete: bool


def _save_checkpoint(
    path,
    *,
    graph,
    next_key,
    update_step,
    evaluation_count,
    best_graph,
    best_value,
    best_step,
    best_key,
    identity,
    complete,
):
    arrays = {
        "X": np.asarray(graph.X),
        "E_upper": np.asarray(graph.E_upper),
        "next_key": np.asarray(jax.random.key_data(next_key)),
    }
    metadata = {
        "schema": CHECKPOINT_VERSION,
        "identity": identity,
        "update_step": update_step,
        "evaluation_count": evaluation_count,
        "next_key_impl": str(jax.random.key_impl(next_key)),
        "best_value": best_value,
        "best_step": best_step,
        "complete": complete,
    }
    if best_graph is not None:
        arrays.update(
            best_X=np.asarray(best_graph.X),
            best_E_upper=np.asarray(best_graph.E_upper),
            best_key=np.asarray(jax.random.key_data(best_key)),
        )
        metadata["best_key_impl"] = str(jax.random.key_impl(best_key))
    arrays["metadata"] = np.asarray(
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    )
    _write_npz(path, arrays)
    write_json(
        path.parent.parent / "latest_checkpoint.json",
        {
            "path": str(path.relative_to(path.parent.parent)).replace("\\", "/"),
            "sha256": file_sha256(path),
        },
    )


def _load_checkpoint(directory, expected_identity):
    receipt = json.loads(
        (directory / "latest_checkpoint.json").read_text(encoding="utf-8")
    )
    path = (directory / receipt["path"]).resolve()
    if (
        not path.is_relative_to(directory.resolve())
        or file_sha256(path) != receipt["sha256"]
    ):
        raise ValueError("Checkpoint receipt identity/path mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        if (
            metadata["schema"] != CHECKPOINT_VERSION
            or metadata["identity"] != expected_identity
        ):
            raise ValueError(
                "Incompatible checkpoint schema/source/config/context/model/RNG identity"
            )
        graph = RelaxedGraph(
            _restore_array(arrays["X"]), _restore_array(arrays["E_upper"])
        )
        validate_graph(graph)
        next_key = jax.random.wrap_key_data(
            jnp.asarray(arrays["next_key"]), impl=metadata["next_key_impl"]
        )
        best_graph = None
        best_key = None
        if "best_X" in arrays:
            best_graph = RelaxedGraph(
                _restore_array(arrays["best_X"]), _restore_array(arrays["best_E_upper"])
            )
            validate_graph(best_graph)
            best_key = jax.random.wrap_key_data(
                jnp.asarray(arrays["best_key"]), impl=metadata["best_key_impl"]
            )
    return graph, next_key, best_graph, best_key, metadata


def _record_raw(directory, label, *, graph, value, aux, gradients, key, proposal=None):
    arrays = {
        "X": np.asarray(graph.X),
        "E_upper": np.asarray(graph.E_upper),
        "key_data": np.asarray(jax.random.key_data(key)),
    }
    if value is not None:
        arrays["value"] = np.asarray(value)
    if gradients is not None:
        arrays.update(
            gradient_X=np.asarray(gradients.X),
            gradient_E_upper=np.asarray(gradients.E_upper),
        )
    if proposal is not None:
        arrays.update(
            proposal_X=np.asarray(proposal.X),
            proposal_E_upper=np.asarray(proposal.E_upper),
        )
    tree = _encode_tree(aux, arrays, "aux")
    arrays["aux_structure"] = np.asarray(json.dumps(tree, allow_nan=False))
    arrays["key_impl"] = np.asarray(str(jax.random.key_impl(key)))
    _write_npz(directory / "evaluations" / f"{label}.npz", arrays)


def _aux_metrics(tree, path=""):
    """Keep small detached scalar metrics in the trace; arrays stay in NPZ."""
    metrics = {}
    if isinstance(tree, dict):
        for name, value in tree.items():
            metrics.update(_aux_metrics(value, f"{path}.{name}" if path else str(name)))
    elif isinstance(tree, (tuple, list)):
        for index, value in enumerate(tree):
            metrics.update(_aux_metrics(value, f"{path}[{index}]"))
    elif isinstance(tree, (jax.Array, np.ndarray)):
        if (
            not jax.dtypes.issubdtype(tree.dtype, jax.dtypes.prng_key)
            and tree.shape == ()
        ):
            metrics[path] = _json_value(tree)
    elif tree is None or isinstance(tree, (str, int, float, bool, np.generic)):
        metrics[path] = _json_value(tree)
    return metrics


def _run_trajectory(
    loss,
    graph: RelaxedGraph,
    key: jax.Array,
    *,
    directory: str | Path,
    config: TrajectoryConfig,
    identities: dict[str, Any],
    resolved_config: dict[str, Any] | None = None,
    resume: bool = False,
    max_updates: int | None = None,
    readout: Callable[[RelaxedGraph], dict] = hard_readout,
    _receipt_state: dict | None = None,
) -> TrajectoryResult:
    """Run minimization, preserving evaluated pre-states and a final evaluation.

    ``max_updates`` pauses engineering runs without changing the sealed total
    budget, so restart comparisons use identical configurations. A failed run
    must be inspected; it is never silently resumed. Checkpoint commits own the
    trace prefix; an interrupted write past that prefix is discarded on resume.
    """
    config.validate()
    validate_graph(graph, atol=config.simplex_atol)
    directory = Path(directory)
    if max_updates is not None and (
        isinstance(max_updates, bool)
        or not isinstance(max_updates, int)
        or max_updates < 0
    ):
        raise ValueError("max_updates must be a nonnegative integer")
    sealed_config = {"trajectory": config.as_dict(), "resolved": resolved_config or {}}
    identity = {
        "contract": contract_manifest(),
        "bindings": identities,
        "config_sha256": identity_digest(sealed_config),
        "precision": str(graph.X.dtype),
        "rng_impl": str(jax.random.key_impl(key)),
        "initial_key_data": np.asarray(jax.random.key_data(key)).tolist(),
        "initial_graph_sha256": hashlib.sha256(
            np.asarray(graph.X).tobytes() + np.asarray(graph.E_upper).tobytes()
        ).hexdigest(),
    }
    identity = _json_value(identity)
    step = count = 0
    next_key = key
    best_graph = best_key = best_value = best_step = None
    trace = []
    if resume:
        if (directory / "failure.json").exists():
            raise ValueError(
                "A failed trajectory needs inspection; automatic resume is disabled"
            )
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Run identity mismatch")
        _receipt_state["active"] = True
        graph, next_key, best_graph, best_key, metadata = _load_checkpoint(
            directory, identity
        )
        step, count = metadata["update_step"], metadata["evaluation_count"]
        best_value, best_step = metadata["best_value"], metadata["best_step"]
        trace_path = directory / "trace.jsonl"
        trace = (
            [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            if trace_path.exists()
            else []
        )
        if len(trace) < count:
            raise ValueError("Checkpoint references missing evaluation receipts")
        trace = trace[:count]
        if metadata["complete"]:
            write_json(
                directory / "manifest.json",
                {
                    "identity": identity,
                    "status": "complete",
                    "updates": step,
                    "evaluations": count,
                    "best_value": best_value,
                    "best_step": best_step,
                },
            )
            return TrajectoryResult(graph, best_graph, best_value, step, next_key, True)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "manifest.json").exists():
            raise FileExistsError("Use a unique run directory or explicit resume")
        _receipt_state["active"] = True
        write_json(
            directory / "manifest.json", {"identity": identity, "status": "running"}
        )
        write_json(directory / "config.json", sealed_config)
        write_json(
            directory / "validation.json",
            {
                "status": "pending",
                "gates": {
                    letter: "pending HPC numerical validation" for letter in "ABCDEFG"
                },
            },
        )
        _save_checkpoint(
            directory / "checkpoints" / "step_000000_eval_000000.npz",
            graph=graph,
            next_key=next_key,
            update_step=0,
            evaluation_count=0,
            best_graph=None,
            best_value=None,
            best_step=None,
            best_key=None,
            identity=identity,
            complete=False,
        )
    stop = (
        config.steps if max_updates is None else min(config.steps, step + max_updates)
    )
    complete = False
    while step < stop or (step == config.steps and stop == config.steps):
        _receipt_state["update_step"] = step
        used_key = next_key
        future_key = next_key
        if config.rng_policy == "split":
            future_key, used_key = jax.random.split(next_key)
        try:
            evaluation = evaluate_graph(loss, graph, used_key, atol=config.simplex_atol)
            update = (
                None
                if step == config.steps
                else projected_step(
                    evaluation,
                    eta_X=config.eta_X,
                    eta_E=config.eta_E,
                    atol=config.simplex_atol,
                )
            )
        except NonfiniteEvaluationError as error:
            _record_raw(
                directory,
                f"failed_{step:06d}",
                graph=error.graph,
                value=error.value,
                aux=error.aux,
                gradients=error.gradients,
                key=error.key,
                proposal=error.proposal,
            )
            write_json(
                directory / "failure.json",
                {
                    "update_step": step,
                    "type": type(error).__name__,
                    "message": str(error),
                    "nonfinite_paths": error.nonfinite_paths,
                    "raw_artifact": f"evaluations/failed_{step:06d}.npz",
                },
            )
            write_json(
                directory / "manifest.json", {"identity": identity, "status": "failed"}
            )
            raise
        except Exception as error:
            write_json(
                directory / "failure.json",
                {
                    "update_step": step,
                    "type": type(error).__name__,
                    "message": str(error),
                    "checkpoint": "latest_checkpoint.json",
                },
            )
            write_json(
                directory / "manifest.json", {"identity": identity, "status": "failed"}
            )
            raise
        label = f"step_{step:06d}"
        _record_raw(
            directory,
            label,
            graph=graph,
            value=evaluation.value,
            aux=evaluation.aux,
            gradients=evaluation.gradients,
            key=used_key,
        )
        readout_path = None
        if step % config.readout_every == 0 or update is None:
            # Readout exceptions are receipts, never a graph substitution.
            try:
                detached_readout = readout(graph)
            except Exception as error:
                detached_readout = {
                    "sanitized": False,
                    "conversion_error": f"{type(error).__name__}: {error}",
                    "element_indices": np.asarray(graph.X).argmax(-1).tolist(),
                    "bond_upper_indices": np.asarray(graph.E_upper).argmax(-1).tolist(),
                }
            readout_path = f"readouts/{label}.json"
            write_json(directory / readout_path, detached_readout)
        value = float(evaluation.value)
        if best_value is None or value < best_value:
            best_graph, best_value, best_step, best_key = graph, value, step, used_key
        metrics = {
            "evaluation": count,
            "evaluated_step": step,
            "loss": value,
            "aux_metrics": _aux_metrics(evaluation.aux),
            "final": update is None,
            "graph": graph_diagnostics(graph),
            "raw_gradient_norm_X": float(
                np.linalg.norm(np.asarray(evaluation.gradients.X, dtype=np.float64))
            ),
            "raw_gradient_norm_E_upper": float(
                np.linalg.norm(
                    np.asarray(evaluation.gradients.E_upper, dtype=np.float64)
                )
            ),
            "key_data": np.asarray(jax.random.key_data(used_key)).tolist(),
            "key_impl": str(jax.random.key_impl(used_key)),
            "projection_residuals": None
            if update is None
            else update.projection_residuals,
            "evaluated_arrays": f"evaluations/{label}.npz",
            "readout": readout_path,
        }
        trace.append(metrics)
        trace_path = directory / "trace.jsonl"
        with trace_path.with_suffix(".jsonl.tmp").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in trace:
                handle.write(
                    json.dumps(_json_value(row), sort_keys=True, allow_nan=False) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(trace_path.with_suffix(".jsonl.tmp"), trace_path)
        count += 1
        next_key = future_key
        if update is not None:
            graph = update.next_graph
            step += 1
        else:
            complete = True
        _save_checkpoint(
            directory / "checkpoints" / f"step_{step:06d}_eval_{count:06d}.npz",
            graph=graph,
            next_key=next_key,
            update_step=step,
            evaluation_count=count,
            best_graph=best_graph,
            best_value=best_value,
            best_step=best_step,
            best_key=best_key,
            identity=identity,
            complete=complete,
        )
        if complete:
            break
    write_json(
        directory / "manifest.json",
        {
            "identity": identity,
            "status": "complete" if complete else "paused",
            "updates": step,
            "evaluations": count,
            "best_value": best_value,
            "best_step": best_step,
        },
    )
    return TrajectoryResult(graph, best_graph, best_value, step, next_key, complete)


def run_trajectory(
    loss,
    graph: RelaxedGraph,
    key: jax.Array,
    *,
    directory: str | Path,
    config: TrajectoryConfig,
    identities: dict[str, Any],
    resolved_config: dict[str, Any] | None = None,
    resume: bool = False,
    max_updates: int | None = None,
    readout: Callable[[RelaxedGraph], dict] = hard_readout,
) -> TrajectoryResult:
    """Run with failure receipts covering both evaluation and artifact writes.

    Loss aux must be a nested dict/list/tuple of arrays and JSON scalars. Raw
    arrays remain NPZ values, including nonfinite failure quantities. Readout
    never supplies a replacement graph; every checkpoint owns a trace prefix.
    """
    receipt_state = {"active": False, "update_step": 0}
    try:
        return _run_trajectory(
            loss,
            graph,
            key,
            directory=directory,
            config=config,
            identities=identities,
            resolved_config=resolved_config,
            resume=resume,
            max_updates=max_updates,
            readout=readout,
            _receipt_state=receipt_state,
        )
    except Exception as error:
        directory = Path(directory)
        if receipt_state["active"] and not (directory / "failure.json").exists():
            # If the destination itself is unwritable, preserve the original
            # exception; a caller/HPC launcher must retain its stderr receipt.
            try:
                step = receipt_state["update_step"]
                raw = directory / "evaluations" / f"step_{step:06d}.npz"
                write_json(
                    directory / "failure.json",
                    {
                        "update_step": step,
                        "type": type(error).__name__,
                        "message": str(error),
                        "raw_artifact": str(raw.relative_to(directory))
                        if raw.exists()
                        else None,
                        "checkpoint": "latest_checkpoint.json",
                        "phase": "execution or artifact persistence",
                    },
                )
                manifest_path = directory / "manifest.json"
                manifest = (
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists()
                    else {}
                )
                manifest["status"] = "failed"
                write_json(manifest_path, manifest)
            except OSError:
                pass
        raise
