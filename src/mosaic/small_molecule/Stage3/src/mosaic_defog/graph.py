"""Small array contract; accepts Mosaic graph objects without importing Mosaic."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

ELEMENTS = ("C", "N", "O", "S", "F", "Cl")
BONDS = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


class Graph(NamedTuple):
    """Occupied atom rows X[N,K] and lexicographic unordered bonds E_upper[P,5]."""

    X: Any
    E_upper: Any

    @property
    def n_atoms(self):
        return self.X.shape[0]


def host_array(value):
    """Detach the teacher input; the student's graph stays live in its own loss."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_arrays(x, e, *, n_elements=None):
    """Host boundary validation. Never call inside a differentiated function."""
    x, e = host_array(x), host_array(e)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("X must have shape [N,K] with occupied atom slots")
    if n_elements is not None and x.shape[1] != n_elements:
        raise ValueError(
            f"Expected {n_elements} atom categories, received {x.shape[1]}"
        )
    n = x.shape[0]
    if e.shape != (n * (n - 1) // 2, len(BONDS)):
        raise ValueError(
            "E_upper must contain one five-category row per unordered pair"
        )
    for name, value in (("X", x), ("E_upper", e)):
        if value.dtype.kind != "f":
            raise TypeError(f"{name} must contain floating probabilities")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite probabilities")
        if (value < 0).any() or (value > 1).any():
            raise ValueError(f"{name} probabilities must lie in [0,1]")
        if not np.allclose(value.sum(-1), 1, atol=1e-6, rtol=0):
            raise ValueError(f"{name} rows must sum to one")
    return x, e


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Prediction:
    """A detached clean prediction; it is neither a sample nor a repaired graph."""

    X: np.ndarray
    E_upper: np.ndarray
    time: float
    metadata: dict | None = None

    def __post_init__(self):
        x, e = validate_arrays(self.X, self.E_upper)
        if not np.isfinite(self.time) or not 0 <= self.time <= 1:
            raise ValueError("Teacher time must lie in [0,1]")
        for name, value in (("X", x), ("E_upper", e)):
            value = np.array(value, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def jax_loss(self, graph):
        """Scalar MSE with live JAX graph gradients and stopped teacher targets."""
        from .losses import jax_terms

        return jax_terms(graph, self)[0]

    def torch_loss(self, graph):
        """Scalar MSE with live Torch graph gradients and detached targets."""
        from .losses import torch_terms

        return torch_terms(graph, self)[0]
