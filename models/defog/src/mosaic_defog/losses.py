"""Live student losses. All teacher IO and validation happen before these calls."""


def _check_shapes(graph, target):
    if graph.X.shape != target.X.shape or graph.E_upper.shape != target.E_upper.shape:
        raise ValueError("Teacher and student must have identical slots and categories")


def jax_terms(graph, target):
    import jax
    import jax.numpy as jnp

    _check_shapes(graph, target)
    x = jax.lax.stop_gradient(jnp.asarray(target.X, dtype=graph.X.dtype))
    e = jax.lax.stop_gradient(jnp.asarray(target.E_upper, dtype=graph.E_upper.dtype))
    atoms = jnp.mean(jnp.sum((graph.X - x) ** 2, axis=-1))
    bonds = (
        jnp.mean(jnp.sum((graph.E_upper - e) ** 2, axis=-1))
        if graph.E_upper.shape[0]
        else jnp.asarray(0.0, dtype=graph.E_upper.dtype)
    )
    return atoms + bonds, {"defog_atom_mse": atoms, "defog_bond_mse": bonds}


def torch_terms(graph, target):
    import torch

    _check_shapes(graph, target)
    x = torch.tensor(target.X.copy(), dtype=graph.X.dtype, device=graph.X.device)
    e = torch.tensor(
        target.E_upper.copy(), dtype=graph.E_upper.dtype, device=graph.E_upper.device
    )
    atoms = ((graph.X - x) ** 2).sum(-1).mean()
    bonds = (
        ((graph.E_upper - e) ** 2).sum(-1).mean()
        if graph.E_upper.shape[0]
        else graph.E_upper.sum() * 0
    )
    return atoms + bonds, {"defog_atom_mse": atoms, "defog_bond_mse": bonds}
