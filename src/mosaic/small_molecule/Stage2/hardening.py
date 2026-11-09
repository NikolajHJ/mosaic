"""Let the model see increasingly discrete atom and bond choices.

Hardening is a separate comparison against a relaxed run. It changes what
the model sees and how gradients flow back. The stored ligand still uses
probabilities, so both arms can be compared using the same raw readout.
"""


HARDENING_GRADIENTS = ("inherited", "passthrough", "rescaled")


def attenuation(hardness, categories):
    """Scalar estimate of how much the inherited transform shrinks a gradient.

    The exact Jacobian of Mosaic's transition at ``soft=0, temp=1`` is
    ``h*J_softmax(p) + (1-h)*I``, because the term that survives
    ``stop_gradient`` is a softmax of ``p`` rather than ``p`` itself. For rows
    that are not already near one-hot, ``J_softmax = diag(s) - s s^T`` has
    eigenvalues around ``1/K`` on the simplex tangent, so the magnitude
    reaching ``p`` is about ``(1-h) + h/K``.

    This is the scalar surrogate ``rescaled`` inverts. It restores magnitude,
    not direction: the true Jacobian is not a multiple of the identity.
    """
    return (1.0 - hardness) + hardness / categories


def hardening_factory(native, total_steps, gradient="inherited"):
    """Forward hardening as a separate experiment, protocol 5.5.

    The forward representation is ``h*onehot + (1-h)*probabilities`` through a
    straight-through estimator, with ``h`` ramping linearly from 0 to 1. The
    optimized state stays the underlying probabilities, so the operators and
    the readout are unchanged, and the one-hot graph is rebuilt from ``p`` each
    step and never stored.

    ``gradient`` selects how the backward pass crosses that step; see
    ``experiment.yaml`` for what each mode is for. ``inherited`` reuses
    Mosaic's existing transition as the protocol asks, and carries its gradient
    attenuation; ``passthrough`` is the textbook identity-Jacobian estimator;
    ``rescaled`` keeps the inherited transform and compensates its magnitude.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from mosaic.optimizers import _colabdesign_transform
    from mosaic.small_molecule.state import RelaxedGraph

    if gradient not in HARDENING_GRADIENTS:
        raise ValueError(f"hardening gradient must be one of {HARDENING_GRADIENTS}")

    def straight_through(rows, hardness):
        """Identity on the value, identity on the gradient."""
        onehot = jax.nn.one_hot(rows.argmax(-1), rows.shape[-1], dtype=rows.dtype)
        return rows + hardness * jax.lax.stop_gradient(onehot - rows)

    def scale_gradient(rows, factor):
        """Identity on the value, ``factor`` on the gradient."""
        return rows * factor + jax.lax.stop_gradient(rows * (1.0 - factor))

    def harden(rows, hardness):
        if gradient == "passthrough":
            return straight_through(rows, hardness)
        if gradient == "rescaled":
            rows = scale_gradient(rows, 1.0 / attenuation(hardness, rows.shape[-1]))
        return _colabdesign_transform(rows, 0.0, 1.0, hardness)

    class HardenedLoss(eqx.Module):
        native: object
        hardness: jax.Array

        @eqx.filter_jit
        def __call__(self, graph, *, key):
            forward = RelaxedGraph(
                harden(graph.X, self.hardness),
                harden(graph.E_upper, self.hardness),
            )
            value, aux = self.native(forward, key=key)
            return value, {
                **aux,
                "forward_hardness": jnp.asarray(self.hardness),
                "hardening_gradient": gradient,
            }

    def prepare(graph, step):
        hardness = min(step / max(1, total_steps - 1), 1.0)
        return HardenedLoss(native, jnp.asarray(hardness))

    return prepare
