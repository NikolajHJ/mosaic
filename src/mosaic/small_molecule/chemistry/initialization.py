"""Fresh shared-graph probability initializations; no native model RNG reuse."""

import jax.numpy as jnp
import numpy as np

from ..state import RelaxedGraph, mix_with_uniform


def initialize_graph(hard, recipe):
    """Draw only on allocated compute; preserve slots and unordered pair schema.

    Dirichlet(1) is uniform on each categorical simplex, not equal categorical
    probabilities. Rows and unordered pairs are independent. The separate PCG64
    seed does not consume or change the native model's fixed key.
    """
    if recipe == {"kind": "chain_uniform", "mixing": 0.5}:
        return mix_with_uniform(hard, 0.5)
    if (
        recipe.get("kind") != "dirichlet"
        or recipe.get("concentration") != 1.0
        or type(recipe.get("seed")) is not int
        or not 0 <= recipe["seed"] < 2**32
        or set(recipe) != {"kind", "concentration", "seed"}
    ):
        raise ValueError("Unsupported random initialization recipe")
    rng = np.random.Generator(np.random.PCG64(recipe["seed"]))
    x = rng.dirichlet(np.ones(hard.X.shape[-1]), size=hard.n_atoms)
    e = rng.dirichlet(np.ones(hard.E_upper.shape[-1]), size=hard.E_upper.shape[0])
    graph = RelaxedGraph(
        jnp.asarray(x, dtype=hard.X.dtype), jnp.asarray(e, dtype=hard.E_upper.dtype)
    )
    if any(np.any(np.asarray(v) <= 0) for v in (graph.X, graph.E_upper)):
        raise ValueError("Random probabilities underflowed; do not clip or redraw")
    return graph
