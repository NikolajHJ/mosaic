"""Logits for unfloored categorical chemistry losses and one shared graph."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..state import ELEMENTS, RelaxedGraph, pair_indices, validate_graph
from ._reference.representation import GraphSpec


class DecoderLogits(eqx.Module):
    """Generic source-compatible batched logits; bonds are conditional.

    Occupancy gating belongs in a scorer adapter, never in this chemistry law.
    Unordered edge logits are averaged before log-softmax.
    """

    atoms: jax.Array
    charges: jax.Array
    bonds: jax.Array

    @property
    def symmetric_bond_logits(self):
        return (self.bonds + jnp.swapaxes(self.bonds, 1, 2)) * 0.5

    def math_precision(self):
        if self.atoms.dtype in (jnp.float16, jnp.bfloat16):
            return DecoderLogits(
                *(v.astype(jnp.float32) for v in (self.atoms, self.charges, self.bonds))
            )
        return self


def validate_decoded_shape(decoded, spec, *, batch=None):
    """Trace-safe structural checks; host evaluators own finite-value checks."""
    if not isinstance(decoded, DecoderLogits) or decoded.atoms.ndim != 3:
        raise ValueError("Require batched DecoderLogits")
    count = decoded.atoms.shape[0]
    n = spec.max_nodes
    shapes = (
        (count, n, len(spec.atomic_numbers)),
        (count, n, len(spec.formal_charges)),
        (count, n, n, len(spec.bond_names)),
    )
    values = (decoded.atoms, decoded.charges, decoded.bonds)
    if (
        count < 1
        or (batch is not None and batch != count)
        or any(
            v.shape != shape
            or not jnp.issubdtype(v.dtype, jnp.floating)
            or v.dtype != values[0].dtype
            for v, shape in zip(values, shapes)
        )
    ):
        raise ValueError("Logit shapes/dtypes differ from the prepared schema")


class GraphLogits(eqx.Module):
    """Fixed-occupancy neutral Mosaic optimization variables, with no model state."""

    atoms: jax.Array
    bonds: jax.Array

    @classmethod
    def from_graph(cls, graph):
        validate_graph(graph)
        if any(np.any(np.asarray(v) <= 0) for v in (graph.X, graph.E_upper)):
            raise ValueError(
                "Logit initialization requires strictly positive probabilities"
            )
        return cls(jnp.log(graph.X), jnp.log(graph.E_upper))

    def graph(self):
        return RelaxedGraph(
            jax.nn.softmax(self.atoms, -1), jax.nn.softmax(self.bonds, -1)
        )

    def decoded(self):
        """Explicit fixed support: NO_ATOM has zero FP32 mass; charge is neutral.

        The constant -1e30 represents a structurally forbidden absent category,
        not a floor on a trainable category or its NLL. All six actual element
        logits and every unordered bond logit remain unfloored and optimized.
        """
        if (
            self.atoms.dtype not in (jnp.float32, jnp.float64)
            or self.bonds.dtype != self.atoms.dtype
        ):
            raise ValueError(
                "Fixed-support Mosaic logits require common FP32/FP64 precision"
            )
        n = self.atoms.shape[0]
        absent = jnp.full((n, 1), -1e30, dtype=self.atoms.dtype)
        atoms = jnp.concatenate((absent, self.atoms), -1)[None]
        i, j = pair_indices(n)
        bonds = jnp.zeros((n, n, self.bonds.shape[-1]), dtype=self.bonds.dtype)
        bonds = bonds.at[i, j].set(self.bonds).at[j, i].set(self.bonds)
        return DecoderLogits(atoms, jnp.zeros((1, n, 1), self.atoms.dtype), bonds[None])


def mosaic_spec(n):
    """Derive ordering from Mosaic's actual schema, never from source defaults."""
    numbers = {"C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17}
    return GraphSpec(
        max_nodes=n,
        atomic_numbers=(0, *(numbers[s] for s in ELEMENTS)),
        formal_charges=(0,),
    )
