"""Pure probability-space gradients of Mosaic's frozen BRD4 contact loss."""

import json
import os
from pathlib import Path

import numpy as np


class ProtenixBackend:
    def __init__(self, native, key, identity, finish):
        import equinox as eqx

        self._differentiate = eqx.filter_jit(eqx.filter_value_and_grad(native, has_aux=True))
        self._key = key
        self._finish = finish
        self.identity = identity

    def gradient(self, atom, bond):
        import jax.numpy as jnp
        from mosaic.small_molecule.core.state import RelaxedGraph, validate_graph

        if atom.shape != (25, 6) or bond.shape != (300, 5):
            raise ValueError("Expected 25 atom rows and their 300 unordered bond rows")
        if atom.dtype != np.float32 or bond.dtype != np.float32:
            raise TypeError("Direct Protenix inputs must be FP32 probabilities")
        graph = RelaxedGraph(jnp.asarray(atom), jnp.asarray(bond))
        validate_graph(graph)
        (value, _), derivatives = self._differentiate(graph, key=self._key)
        loss = float(value)
        grad_atom = np.asarray(derivatives.X)
        grad_bond = np.asarray(derivatives.E_upper)
        if grad_atom.shape != atom.shape or grad_bond.shape != bond.shape:
            raise ValueError("Direct Protenix derivative shape changed")
        if not (np.isfinite(loss) and np.isfinite(grad_atom).all() and np.isfinite(grad_bond).all()):
            raise FloatingPointError("Nonfinite direct Protenix loss or derivative")
        return loss, grad_atom, grad_bond

    def close(self):
        return self._finish()


def make_backend(seed, work: Path):
    """Load the archived Stage 3 geometry-off model once per allocation."""
    import jax
    from mosaic.small_molecule.core.state import BOND_STATES, ELEMENTS
    from stage3_native import open_native

    if not os.environ.get("LSB_JOBID", "").isdigit() or len(jax.devices("gpu")) != 1:
        raise RuntimeError("Direct Protenix requires one allocated, visible GPU")
    if ELEMENTS != ("C", "N", "O", "S", "F", "Cl") or BOND_STATES != (
        "NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"
    ):
        raise ValueError("Mosaic graph vocabulary differs from the experiment")
    source = Path(os.environ["DIRECT_JUMP_MOSAIC_SOURCE"])
    assets = json.loads(Path(os.environ["DIRECT_JUMP_PROTENIX_ASSETS"]).read_text())
    native, identity, finish = open_native(assets, source)
    key = jax.random.fold_in(jax.random.key(int(seed)), 0xD3F09)
    identity = dict(identity, rng_key=np.asarray(jax.random.key_data(key)).tolist())
    return ProtenixBackend(native, key, identity, finish)
