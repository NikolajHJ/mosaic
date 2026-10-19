"""Frozen-reference native derivative gates and retained per-probe evidence."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..io import write_json
from .trajectory import evaluate_combined


def check_combined_directions(
    contact_loss, evaluation, prepared, spec, reference, *, key, config, directory=None
):
    """Two adjacent finite-difference scales per live logit field, fixed target.

    Native order-null checks remain separate: chemistry deliberately DOES see
    order redistribution that is invisible to Protenix's existence-only map.
    """
    name = config.get("native_name", "contact")
    records = []
    if evaluation.repair_weight != float(config["repair_weight"]):
        raise ValueError("Derivative baseline and trial chemistry coefficient differ")
    report = {
        "status": "running",
        "directions": records,
        "preparation_identity": prepared.identity,
        name + "_objective_identity": config.get("objective_identity"),
        "schema_sha256": spec.fingerprint(),
        "repair_weight": evaluation.repair_weight,
    }
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    try:
        for field in ("atoms", "bonds"):
            gradient = np.asarray(getattr(evaluation.gradients, field))
            contrasts = gradient.max(-1) - gradient.min(-1)
            row = int(contrasts.argmax())
            source, destination = (
                int(gradient[row].argmin()),
                int(gradient[row].argmax()),
            )
            direction = (
                jnp.zeros_like(getattr(evaluation.logits, field))
                .at[row, source]
                .set(-1)
                .at[row, destination]
                .set(1)
            )
            analytic = float(jnp.sum(getattr(evaluation.gradients, field) * direction))
            probes, agreements = [], []
            records.append(
                {
                    "field": field,
                    "row": row,
                    "source": source,
                    "destination": destination,
                    "analytic": analytic,
                    "probes": probes,
                }
            )
            for epsilon in config.get("epsilons", (0.01, 0.003, 0.001)):
                values = []
                for sign in (-1, 1):
                    state = eqx.tree_at(
                        lambda v: getattr(v, field),
                        evaluation.logits,
                        getattr(evaluation.logits, field) + sign * epsilon * direction,
                    )
                    if directory is not None:
                        np.savez_compressed(
                            directory / "current_probe.npz",
                            atom_logits=np.asarray(state.atoms),
                            bond_logits=np.asarray(state.bonds),
                            key_data=np.asarray(jax.random.key_data(key)),
                        )
                    value = evaluate_combined(
                        contact_loss,
                        state,
                        prepared,
                        spec,
                        reference,
                        key=key,
                        config=config,
                    )
                    values.append(value.primal)
                    if directory is not None:
                        graph = state.graph()
                        saved = {
                            "X": graph.X,
                            "E_upper": graph.E_upper,
                            "atom_logits": state.atoms,
                            "bond_logits": state.bonds,
                            "gradient_atom_logits": value.gradients.atoms,
                            "gradient_bond_logits": value.gradients.bonds,
                            name + "_gradient_X": value.native.evaluation.gradients.X,
                            name
                            + "_gradient_E_upper": value.native.evaluation.gradients.E_upper,
                            "combined_autodiff": value.value,
                            "combined_primal": value.primal,
                            name + "_primal": value.native.primal,
                            "repair_weight": value.repair_weight,
                            "key_data": jax.random.key_data(key),
                        }
                        saved.update(
                            {
                                k: v
                                for k, v in value.native.evaluation.aux.items()
                                if eqx.is_array(v)
                                and (v.ndim == 0 or k == "distogram_logits")
                            }
                        )
                        np.savez_compressed(
                            directory / f"{field}-{epsilon:g}-{sign:+d}.npz",
                            **{k: np.asarray(v) for k, v in saved.items()},
                        )
                fd = (values[1] - values[0]) / (2 * epsilon)
                scale = max(abs(fd), abs(analytic))
                agreement = (
                    abs(analytic) > 1e-7
                    and fd * analytic > 0
                    and abs(fd - analytic) <= 0.002 + 0.15 * scale
                    and abs(fd - analytic) <= 0.15 * scale
                )
                probes.append(
                    {
                        "epsilon": epsilon,
                        "finite_difference": fd,
                        "analytic": analytic,
                        "agrees": agreement,
                        "preparation_identity": prepared.identity,
                    }
                )
                agreements.append(agreement)
                if directory is not None:
                    write_json(directory / "report.json", report)
            if not any(a and b for a, b in zip(agreements, agreements[1:])):
                raise AssertionError(
                    f"Combined frozen-reference {field} derivative check failed"
                )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if directory is not None:
            write_json(directory / "report.json", report)
    return records
