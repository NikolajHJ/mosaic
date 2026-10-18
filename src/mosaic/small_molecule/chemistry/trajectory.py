"""Model-agnostic logit updates with a fixed repair reference per line search.

The native branch always receives the current raw shared probabilities. Reusing
its observation across a target refresh is permitted only for byte-identical
graph/key inputs; the native scalar/gradient are independent of repair targets.
"""

from dataclasses import dataclass
from pathlib import Path
import time
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..io import file_sha256, hard_readout, write_json
from ..optimizers import _nonfinite_paths, evaluate_graph
from .loss import composition_statistics, failure_aware_repair_loss, population_js
from .state import GraphLogits


@dataclass(frozen=True)
class NativeObservation:
    evaluation: object
    primal: float


@dataclass(frozen=True)
class CombinedEvaluation:
    logits: GraphLogits
    native: NativeObservation
    value: float
    primal: float
    terms: dict
    gradients: GraphLogits
    preparation_id: str


def _save_trials(path, evaluations):
    """Atomically retain every finished trial, including before a later failure."""
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            atom_logits=np.stack([np.asarray(v.logits.atoms) for v in evaluations]),
            bond_logits=np.stack([np.asarray(v.logits.bonds) for v in evaluations]),
            gradient_atom_logits=np.stack(
                [np.asarray(v.gradients.atoms) for v in evaluations]
            ),
            gradient_bond_logits=np.stack(
                [np.asarray(v.gradients.bonds) for v in evaluations]
            ),
            independent_primal=np.asarray([v.primal for v in evaluations]),
            surrogate_autodiff=np.asarray([v.value for v in evaluations]),
            contact_primal=np.asarray([v.native.primal for v in evaluations]),
            component_names=np.asarray(list(evaluations[0].terms)),
            components=np.asarray(
                [[v.terms[k] for k in evaluations[0].terms] for v in evaluations]
            ),
        )
    temporary.replace(path)


def _append_record(path, value):
    """Durable saved metadata between periodic full-report rewrites."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def evaluate_combined(
    contact_loss, logits, prepared, spec, reference, *, key, config, native=None
):
    """Keep native raw derivatives and add the exact softmax pullback.

    Differentiating softmax changes optimization coordinates, not the molecule.
    Native independent-primal and AD paths remain separately evaluated. The
    chemistry forward has its own independent primal call with the same fixed
    preparation, coefficients and composition reference.
    """
    graph = logits.graph()
    if native is None:
        raw = evaluate_graph(contact_loss, graph, key)
        primal, aux = contact_loss(graph, key=key)
        if _nonfinite_paths((primal, aux)):
            raise FloatingPointError("Nonfinite independent native forward")
        native = NativeObservation(raw, float(primal))
    else:
        raw = native.evaluation
        if (
            str(raw.key.dtype) != str(key.dtype)
            or not np.array_equal(
                np.asarray(jax.random.key_data(raw.key)),
                np.asarray(jax.random.key_data(key)),
            )
            or any(
                np.asarray(a).dtype != np.asarray(b).dtype
                or not np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in ((graph.X, raw.graph.X), (graph.E_upper, raw.graph.E_upper))
            )
        ):
            raise ValueError("Native observation reuse changed graph or RNG")
    prepared.verify(spec)

    def chemical(value):
        decoded = value.decoded()
        loss, terms = failure_aware_repair_loss(decoded, prepared, spec, config["loss"])
        stats = composition_statistics(decoded)
        js = population_js(stats["elements"], reference)
        terms = {**terms, "chemistry_loss": loss, "element_js": js}
        return config["repair_weight"] * loss + config["element_js_weight"] * js, terms

    (chemical_value, terms), gradient = eqx.filter_value_and_grad(
        chemical, has_aux=True
    )(logits)
    chemical_primal, _ = chemical(logits)

    # Each row's softmax Jacobian is diag(p)-p p^T. Do not center, clip or
    # normalize the native raw gradient before retaining/pulling it back.
    def pull(p, g):
        return p * (g - jnp.sum(p * g, -1, keepdims=True))

    gradients = GraphLogits(
        pull(graph.X, raw.gradients.X) + gradient.atoms,
        pull(graph.E_upper, raw.gradients.E_upper) + gradient.bonds,
    )
    total = float(raw.value) + float(chemical_value)
    primal = native.primal + float(chemical_primal)
    if _nonfinite_paths((logits, total, primal, terms, gradients)):
        raise FloatingPointError("Nonfinite combined loss or unmodified logit gradient")
    if abs(total - primal) > config.get("parity_atol", 2e-5):
        raise AssertionError("Native/combined independent-primal parity failed")
    terms = {k: float(v) for k, v in terms.items()}
    terms.update(contact_loss=native.primal, contact_autodiff_loss=float(raw.value))
    return CombinedEvaluation(
        logits, native, total, primal, terms, gradients, prepared.identity
    )


def _snapshot(directory, step, evaluation, *, prepared, previous_fixed_after=None):
    graph = evaluation.logits.graph()
    path = directory / "snapshots" / f"step-{step:04d}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray(graph.X),
        E_upper=np.asarray(graph.E_upper),
        atom_logits=np.asarray(evaluation.logits.atoms),
        bond_logits=np.asarray(evaluation.logits.bonds),
        gradient_atom_logits=np.asarray(evaluation.gradients.atoms),
        gradient_bond_logits=np.asarray(evaluation.gradients.bonds),
        contact_gradient_X=np.asarray(evaluation.native.evaluation.gradients.X),
        contact_gradient_E_upper=np.asarray(
            evaluation.native.evaluation.gradients.E_upper
        ),
        key_data=np.asarray(jax.random.key_data(evaluation.native.evaluation.key)),
        contact_primal=np.asarray(evaluation.native.primal),
        refreshed_surrogate=np.asarray(evaluation.primal),
        autodiff_surrogate=np.asarray(evaluation.value),
    )
    return {
        "step": step,
        "snapshot": path.relative_to(directory).as_posix(),
        "snapshot_sha256": file_sha256(path),
        "readout": hard_readout(graph),
        "contact_loss": evaluation.terms["contact_loss"],
        "refreshed_surrogate": evaluation.primal,
        "autodiff_surrogate": evaluation.value,
        "components": evaluation.terms,
        "preparation_identity": evaluation.preparation_id,
        "previous_fixed_surrogate_after": previous_fixed_after,
        "chemistry_preparation": [
            {
                "raw_native_valid": row["base"]["raw_valid"],
                "original_status": row["base"]["status"],
                "route": row["route"],
                "extra_status": None
                if row["extra"] is None
                else row["extra"]["status"],
                "component_count": row["component_count"],
                "unsupported_local_atoms": row["unsupported_local_atoms"],
            }
            for row in prepared.receipt["rows"]
        ],
    }


def run_chemical_updates(
    contact_loss,
    initial,
    *,
    prepare,
    spec,
    reference,
    key,
    directory,
    steps,
    step_size,
    max_backtracks,
    config,
):
    """Attempt every requested update under the explicitly declared best-primal policy.

    This is a new logit-coordinate study, never a resume or rewrite of the
    historical projected-probability run. Every trial uses the same preparation.
    Saved changing-target totals are labelled surrogates, not one objective.
    """
    if (
        type(steps) is not int
        or steps < 1
        or type(max_backtracks) is not int
        or max_backtracks < 1
        or not np.isfinite(step_size)
        or step_size <= 0
    ):
        raise ValueError("Invalid update budget")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "references").mkdir()
    (directory / "trials").mkdir()
    report = {
        "status": "running",
        "requested_updates": steps,
        "parameterization": "categorical_logits_v1",
        "acceptance_policy": "best_primal",
        "states": [],
        "updates": [],
        "objective_scope": "contact fixed across states; repair references refreshed; only within-state surrogate deltas comparable",
        "contact_objective_identity": config.get("objective_identity"),
        "key_data": np.asarray(jax.random.key_data(key)).tolist(),
        "key_dtype": str(key.dtype),
        "counts": {
            "applied": 0,
            "strict_improving": 0,
            "forced_fallback": 0,
            "no_op": 0,
        },
    }
    evaluation = None
    try:
        reference = jnp.asarray(reference, dtype=initial.atoms.dtype)
        if (
            np.asarray(reference).shape != (len(spec.atomic_numbers) - 1,)
            or not np.isfinite(np.asarray(reference)).all()
            or np.any(np.asarray(reference) < 0)
            or float(jnp.sum(reference)) <= 0
        ):
            raise ValueError("Invalid frozen empirical composition reference")
        prepared = prepare(initial, 0)
        write_json(directory / "references/step-0000.json", prepared.record())
        evaluation = evaluate_combined(
            contact_loss, initial, prepared, spec, reference, key=key, config=config
        )
        report["states"].append(_snapshot(directory, 0, evaluation, prepared=prepared))
        _append_record(directory / "states.jsonl", report["states"][-1])
        report["best_observed_contact"] = {
            "step": 0,
            "loss": evaluation.terms["contact_loss"],
            "snapshot": report["states"][0]["snapshot"],
            "snapshot_sha256": report["states"][0]["snapshot_sha256"],
            "objective_identity": config.get("objective_identity"),
        }
        report["initial_checks_file"] = "combined_initial_checks/report.json"
        report["initial_checks"] = check_combined_directions(
            contact_loss,
            evaluation,
            prepared,
            spec,
            reference,
            key=key,
            config=config,
            directory=directory / "combined_initial_checks",
        )
        for step in range(steps):
            update_started = time.perf_counter()
            record = {
                "step": step,
                "preparation_identity": prepared.identity,
                "fixed_before": evaluation.primal,
                "attempts": [],
            }
            report["updates"].append(record)
            best, selected, saved = None, None, []
            for trial in range(max_backtracks):
                rate = step_size * 0.5**trial
                proposal = GraphLogits(
                    evaluation.logits.atoms - rate * evaluation.gradients.atoms,
                    evaluation.logits.bonds - rate * evaluation.gradients.bonds,
                )
                # Save the proposal before native execution so an exception
                # cannot erase the input responsible for a failed trial.
                np.savez_compressed(
                    directory / "current_trial.npz",
                    atom_logits=np.asarray(proposal.atoms),
                    bond_logits=np.asarray(proposal.bonds),
                    step=step,
                    trial=trial,
                )
                candidate = evaluate_combined(
                    contact_loss,
                    proposal,
                    prepared,
                    spec,
                    reference,
                    key=key,
                    config=config,
                )
                strict = (
                    candidate.primal < evaluation.primal - 1e-7
                    and candidate.value < evaluation.value - 1e-7
                )
                record["attempts"].append(
                    {
                        "trial": trial,
                        "rate": rate,
                        "autodiff_surrogate": candidate.value,
                        "independent_primal_surrogate": candidate.primal,
                        "components": candidate.terms,
                        "strict": strict,
                    }
                )
                saved.append(candidate)
                _save_trials(directory / "trials" / f"step-{step:04d}.npz", saved)
                choice = (candidate, rate, trial)
                if best is None or candidate.primal < best[0].primal:
                    best = choice
                if strict:
                    selected = choice
                    break
            fallback = selected is None
            candidate, rate, trial = best if fallback else selected
            old, new = evaluation.logits.graph(), candidate.logits.graph()
            noop = np.array_equal(
                np.asarray(old.X), np.asarray(new.X)
            ) and np.array_equal(np.asarray(old.E_upper), np.asarray(new.E_upper))
            record.update(
                selected_trial=trial,
                accepted_step_size=rate,
                fallback_used=fallback,
                fixed_after=candidate.primal,
                fixed_increased=candidate.primal > evaluation.primal,
                no_op=noop,
                contact_before=evaluation.terms["contact_loss"],
                contact_after=candidate.terms["contact_loss"],
            )
            record["best_observed_surrogate_for_this_preparation"] = min(
                evaluation.primal, candidate.primal
            )
            report["counts"]["applied"] += 1
            report["counts"]["strict_improving"] += int(not fallback)
            report["counts"]["forced_fallback"] += int(fallback)
            report["counts"]["no_op"] += int(noop)
            prepared = prepare(candidate.logits, step + 1)
            write_json(
                directory / "references" / f"step-{step + 1:04d}.json",
                prepared.record(),
            )
            evaluation = evaluate_combined(
                contact_loss,
                candidate.logits,
                prepared,
                spec,
                reference,
                key=key,
                config=config,
                native=candidate.native,
            )
            report["states"].append(
                _snapshot(
                    directory,
                    step + 1,
                    evaluation,
                    prepared=prepared,
                    previous_fixed_after=candidate.primal,
                )
            )
            if (
                evaluation.terms["contact_loss"]
                < report["best_observed_contact"]["loss"]
            ):
                snapshot = report["states"][-1]
                report["best_observed_contact"] = {
                    "step": step + 1,
                    "loss": evaluation.terms["contact_loss"],
                    "snapshot": snapshot["snapshot"],
                    "snapshot_sha256": snapshot["snapshot_sha256"],
                    "objective_identity": config.get("objective_identity"),
                }
            record["elapsed_seconds"] = time.perf_counter() - update_started
            _append_record(directory / "states.jsonl", report["states"][-1])
            _append_record(directory / "updates.jsonl", record)
            print(
                f"update {step + 1}/{steps}: contact={evaluation.terms['contact_loss']:.8g}, refreshed_surrogate={evaluation.primal:.8g}, fallback={fallback}, no_op={noop}",
                flush=True,
            )
            if (step + 1) % 25 == 0:
                write_json(directory / "report.json", report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(directory / "report.json", report)
    return report


def check_combined_directions(*args, **kwargs):
    """Lazy shared check entry point; no model/compiler work at import."""
    from .checks import check_combined_directions as check

    return check(*args, **kwargs)
