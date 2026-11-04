"""Fixed-occupancy probability graphs for the small-molecule V0 contract.

Only ``X`` and ``E_upper`` are differentiated. Element order, unordered-pair
order and the diagonal convention are shared by every adapter and artifact.
Host validation and preparation belong outside a JIT/model forward.
"""

from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

ELEMENTS = ("C", "N", "O", "S", "F", "Cl")
BOND_STATES = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


class RelaxedGraph(eqx.Module):
    """One occupied heavy-atom slot per row, with no charge or null category.

    ``X`` has shape ``[N, 6]``; ``E_upper`` has shape ``[N*(N-1)//2, 5]``.
    Construction itself is trace-safe: call :func:`validate_graph` at the host
    boundary instead of performing value-dependent checks inside this module.
    """

    X: jax.Array
    E_upper: jax.Array

    @property
    def n_atoms(self) -> int:
        return self.X.shape[0]


def _checked_n_atoms(n_atoms: int) -> int:
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, Integral):
        raise TypeError("n_atoms must be a positive integer")
    if n_atoms < 1:
        raise ValueError("a fixed-occupancy ligand needs at least one atom")
    return int(n_atoms)


def pair_indices(n_atoms: int) -> tuple[jax.Array, jax.Array]:
    """Return lexicographic ``i < j`` indices; these are static for each shape."""
    return jnp.triu_indices(_checked_n_atoms(n_atoms), k=1)


def full_bonds(graph: RelaxedGraph) -> jax.Array:
    """Reconstruct ``E[N,N,5]`` without detaching either directed edge.

    Both off-diagonal entries depend on the same stored row, so autodiff adds
    the two model contributions. Diagonal entries are fixed NO_BOND rows.
    """
    i, j = pair_indices(graph.n_atoms)
    no_bond = jnp.array([1, 0, 0, 0, 0], dtype=graph.E_upper.dtype)
    bonds = jnp.broadcast_to(no_bond, (graph.n_atoms, graph.n_atoms, 5))
    bonds = bonds.at[i, j].set(graph.E_upper)
    return bonds.at[j, i].set(graph.E_upper)


def bond_existence(graph: RelaxedGraph) -> jax.Array:
    """Return ``B[N,N] = 1-E[...,NO_BOND]`` with an exactly zero diagonal.

    V0 loses bond order here: redistribution among present-bond categories at
    fixed NO_BOND mass is intentionally invisible to the native model.
    """
    return 1 - full_bonds(graph)[..., 0]


def validate_graph(
    graph: RelaxedGraph,
    *,
    atol: float = 1e-6,
    require_simplex: bool = True,
) -> None:
    """Check shape, floating precision, finiteness and probability invariants.

    This copies leaves to the host and must not be called in a differentiated
    model path. ``require_simplex=False`` is reserved for a finite probability-
    space proposal immediately before Euclidean projection.
    """
    if not isinstance(graph, RelaxedGraph):
        raise TypeError("graph must be a RelaxedGraph")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    x = np.asarray(graph.X)
    edges = np.asarray(graph.E_upper)
    if x.ndim != 2 or x.shape[1] != len(ELEMENTS):
        raise ValueError(f"X must have shape [N, {len(ELEMENTS)}], got {x.shape}")
    n_atoms = _checked_n_atoms(x.shape[0])
    pair_count = n_atoms * (n_atoms - 1) // 2
    if edges.shape != (pair_count, len(BOND_STATES)):
        raise ValueError(
            f"E_upper must have shape [{pair_count}, {len(BOND_STATES)}], "
            f"got {edges.shape}"
        )
    for name, leaf in (("X", x), ("E_upper", edges)):
        if not np.issubdtype(leaf.dtype, np.floating):
            raise TypeError(f"{name} must contain real floating probabilities")
        if not np.isfinite(leaf).all():
            raise ValueError(f"{name} contains nonfinite values")
        if require_simplex:
            if (leaf < -atol).any() or (leaf > 1 + atol).any():
                raise ValueError(f"{name} rows must contain probabilities in [0, 1]")
            if not np.allclose(leaf.sum(axis=-1), 1, atol=atol, rtol=0):
                raise ValueError(f"{name} rows must sum to one")
    if x.dtype != edges.dtype:
        raise TypeError("X and E_upper must use the same floating precision")


def graph_from_hard(
    element_indices,
    bond_upper_indices,
    *,
    dtype=jnp.float32,
) -> RelaxedGraph:
    """Prepare a one-hot graph from detached assignments in contract order.

    No molecule conversion, reordering or bond repair occurs here. The caller
    supplies the native importer's heavy-atom order and all unordered pairs,
    including NO_BOND assignments for absent bonds.
    """
    elements = np.asarray(element_indices)
    bonds = np.asarray(bond_upper_indices)
    if elements.ndim != 1:
        raise ValueError("element_indices must be one-dimensional")
    n_atoms = _checked_n_atoms(len(elements))
    if bonds.shape != (n_atoms * (n_atoms - 1) // 2,):
        raise ValueError("bond_upper_indices must have one index per unordered pair")
    for name, values, n_categories in (
        ("element_indices", elements, len(ELEMENTS)),
        ("bond_upper_indices", bonds, len(BOND_STATES)),
    ):
        # An empty bond list is legal for a one-atom ligand; NumPy infers float
        # for [], so only enforce integer dtype when actual assignments exist.
        if values.size and not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{name} must contain integer category indices")
        if ((values < 0) | (values >= n_categories)).any():
            raise ValueError(f"{name} contains an unsupported category")
    graph = RelaxedGraph(
        X=jax.nn.one_hot(elements.astype(np.int32), len(ELEMENTS), dtype=dtype),
        E_upper=jax.nn.one_hot(bonds.astype(np.int32), len(BOND_STATES), dtype=dtype),
    )
    validate_graph(graph)
    return graph


def mix_with_uniform(graph: RelaxedGraph, mixing: float) -> RelaxedGraph:
    """Prepare an interior graph without changing slot order or occupancy."""
    validate_graph(graph)
    if not np.isfinite(mixing) or not 0 <= mixing <= 1:
        raise ValueError("mixing must be finite and between zero and one")
    result = RelaxedGraph(
        X=(1 - mixing) * graph.X + mixing / len(ELEMENTS),
        E_upper=(1 - mixing) * graph.E_upper + mixing / len(BOND_STATES),
    )
    validate_graph(result)
    return result


def initialize_graph(
    n_atoms: int,
    key: jax.Array,
    *,
    carbon_probability: float = 0.75,
    expected_degree: float = 2.0,
    atom_noise: float = 0.02,
    bond_noise: float = 0.01,
    dtype=jnp.float32,
) -> RelaxedGraph:
    """Prepare carbon-heavy, noisy probability rows, then project once.

    The baseline bond-existence probability is ``d0/(N-1)``. Remaining element
    mass and present-bond mass are uniform over their respective categories.
    Noise is independent per atom row and per *unordered* pair; it is applied
    in probability space, never through logits. Nonzero noise/projection can
    change the realized expected degree. Defaults are engineering examples,
    not Stage 1 initialization-study results; record resolved settings.
    """
    n_atoms = _checked_n_atoms(n_atoms)
    for name, value in (
        ("carbon_probability", carbon_probability),
        ("expected_degree", expected_degree),
        ("atom_noise", atom_noise),
        ("bond_noise", bond_noise),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= carbon_probability <= 1:
        raise ValueError("carbon_probability must be between zero and one")
    if not 0 <= expected_degree <= n_atoms - 1:
        raise ValueError("expected_degree must be between zero and N-1")
    if atom_noise < 0 or bond_noise < 0:
        raise ValueError("noise amplitudes must be nonnegative")
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise TypeError("initialization dtype must be real floating")
    pair_count = n_atoms * (n_atoms - 1) // 2
    p0 = expected_degree / (n_atoms - 1) if n_atoms > 1 else 0.0
    element_row = jnp.full((len(ELEMENTS),), (1 - carbon_probability) / 5, dtype=dtype)
    element_row = element_row.at[0].set(carbon_probability)
    bond_row = jnp.full((len(BOND_STATES),), p0 / 4, dtype=dtype)
    bond_row = bond_row.at[0].set(1 - p0)
    atom_key, bond_key = jax.random.split(key)
    graph = RelaxedGraph(
        X=element_row
        + atom_noise
        * jax.random.normal(atom_key, (n_atoms, len(ELEMENTS)), dtype=dtype),
        E_upper=bond_row
        + bond_noise
        * jax.random.normal(bond_key, (pair_count, len(BOND_STATES)), dtype=dtype),
    )
    # The optimizer imports state, so preparation imports its host projection
    # lazily. Neither model imports nor a model forward happen during setup.
    from mosaic.small_molecule.optimizers import project_graph

    return project_graph(graph)


def graph_diagnostics(graph: RelaxedGraph) -> dict[str, float | int]:
    """Detached scalar diagnostics of the relaxed state, without a readout."""
    validate_graph(graph)
    x = np.asarray(graph.X)
    edges = np.asarray(graph.E_upper)

    def entropy(rows):
        if not len(rows):
            return 0.0
        # xlogx(0)=0; this does not replace or alter the stored probabilities.
        positive = rows > 0
        terms = np.zeros_like(rows)
        terms[positive] = rows[positive] * np.log(rows[positive])
        return float(-terms.sum(axis=-1).mean())

    bond_mass = float((1 - edges[:, 0]).sum())
    return {
        "n_atoms": graph.n_atoms,
        "atom_entropy": entropy(x),
        "bond_entropy": entropy(edges),
        "atom_max_probability": float(x.max(axis=-1).mean()),
        "bond_max_probability": float(edges.max(axis=-1).mean()) if len(edges) else 0.0,
        "bond_mass": bond_mass,
        "expected_degree": 2 * bond_mass / graph.n_atoms,
    }
