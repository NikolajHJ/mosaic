"""Detached denoising targets and optional existing Mosaic hardening logic."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.small_molecule.io import write_json
from mosaic.small_molecule.state import RelaxedGraph


class TeacherLoss(eqx.Module):
    native: object
    teacher_X: jax.Array
    teacher_E: jax.Array
    weight: float
    native_weight: float
    hardness: jax.Array

    @eqx.filter_jit
    def __call__(self, graph, *, key):
        from mosaic.optimizers import _colabdesign_transform

        forward = RelaxedGraph(
            _colabdesign_transform(graph.X, 0.0, 1.0, self.hardness),
            _colabdesign_transform(graph.E_upper, 0.0, 1.0, self.hardness),
        )
        native_value, aux = self.native(forward, key=key)
        dx = graph.X - jax.lax.stop_gradient(self.teacher_X)
        de = graph.E_upper - jax.lax.stop_gradient(self.teacher_E)
        rx = 0.5 * jnp.mean(jnp.sum(dx * dx, axis=-1))
        re = 0.5 * jnp.mean(jnp.sum(de * de, axis=-1))
        return self.native_weight * native_value + self.weight * (rx + re), {
            **aux,
            "prior_loss_X": rx,
            "prior_loss_E": re,
            "prior_weight": jnp.asarray(self.weight),
            "native_weight": jnp.asarray(self.native_weight),
            "forward_hardness": jnp.asarray(self.hardness),
            "teacher_X": self.teacher_X,
            "teacher_E_upper": self.teacher_E,
            "prior_gradient_X": self.weight * dx / len(dx),
            "prior_gradient_E_upper": self.weight * de / len(de),
        }


def moving_teacher_factory(native, prior, settings, directory):
    directory = Path(directory) / "teacher_queries"
    directory.mkdir(exist_ok=True)

    def prepare(graph, step):
        weight = settings.get("prior_weight", 0.0)
        if prior is not None:
            x, e, receipt = prior.teacher(
                np.asarray(graph.X),
                np.asarray(graph.E_upper),
                step=step + settings.get("prior_step_offset", 0),
                seed=settings.get("prior_seed", 71),
                time=settings.get("prior_time", 0.5),
                batch_size=settings.get("prior_batch_size", 1),
            )
            noisy_x, noisy_e = receipt.pop("noisy_X"), receipt.pop("noisy_E_upper")
            np.savez_compressed(
                directory / f"step_{step:06d}.npz",
                teacher_X=x,
                teacher_E_upper=e,
                noisy_X=noisy_x,
                noisy_E_upper=noisy_e,
            )
            write_json(directory / f"step_{step:06d}.json", receipt)
        else:
            x, e = np.asarray(graph.X), np.asarray(graph.E_upper)
        # A dynamic JAX scalar avoids a new compiled graph at every ramp value.
        hardness = (
            min(step / max(1, settings["steps"] - 1), 1.0)
            if settings.get("hardening")
            else 0.0
        )
        return TeacherLoss(
            native,
            jnp.asarray(x),
            jnp.asarray(e),
            weight,
            settings.get("native_weight", 1.0),
            jnp.asarray(hardness),
        )

    return prepare
