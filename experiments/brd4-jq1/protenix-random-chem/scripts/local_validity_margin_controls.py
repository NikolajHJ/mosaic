"""Small allocated controls for the bond-local guidance pilot."""

import os

import jax
import jax.numpy as jnp
import numpy as np

import local_validity_margin as margin
from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
from mosaic.small_molecule.chemistry.state import GraphLogits, mosaic_spec


def run():
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise ValueError("Margin controls require an allocation")
    spec = mosaic_spec(4)
    atoms = jnp.tile(jnp.array([[9., -9., -9., -9., -9., -9.]], dtype=jnp.float32), (4, 1))
    bonds = jnp.full((6, 5), -9., dtype=jnp.float32)
    bonds = bonds.at[:3, 1].set(9.).at[3:, 0].set(9.)
    bonds = bonds.at[0].set(jnp.array([-5., .05, .04, 0., -5.]))
    source = GraphLogits(atoms, bonds)
    prepared = prepare_failure_guidance(source.decoded(), spec, update=0)
    fixed = margin.prepare_margin(source.decoded(), prepared, spec)
    assert fixed.pairs == ((0, 1),)
    assert fixed.allowed == ((False, True, True, False, False),), fixed.record()

    value, grad = jax.value_and_grad(lambda g: fixed.value(g.decoded()))(source)
    assert float(value) > 0 and np.isfinite(grad.bonds).all()
    assert (np.asarray(grad.atoms) == 0).all()
    assert (np.asarray(grad.bonds[1:]) == 0).all()
    assert float(grad.bonds[0, 1]) < 0 < float(grad.bonds[0, 3])
    assert float(jnp.max(jnp.abs(grad.bonds))) <= margin.SLOPE_CAP * margin.WEIGHT_PER_REPAIR_UNIT + 1e-8

    # Even with substantial pooled safe mass, an unsafe winner incurs a penalty.
    unsafe = GraphLogits(atoms, bonds.at[0, 3].set(.10))
    assert float(fixed.value(unsafe.decoded())) > float(value)
    refreshed = prepare_failure_guidance(unsafe.decoded(), spec, update=1)
    assert int(refreshed.arrays["route"][0]) != 0
    assert float(margin.prepare_margin(unsafe.decoded(), refreshed, spec).value(unsafe.decoded())) == 0

    # Either valid label can win with zero loss and zero gradient.
    for label in (1, 2):
        confident = GraphLogits(atoms, bonds.at[0, label].set(2.))
        loss, derivative = jax.value_and_grad(lambda g: fixed.value(g.decoded()))(confident)
        assert float(loss) == 0 and (np.asarray(derivative.bonds) == 0).all()

    # A fixed-teacher directional derivative agrees on a smooth active branch.
    f = lambda z: fixed.value(GraphLogits(atoms, bonds.at[0, 3].set(z)).decoded())
    epsilon = .001
    numerical = float((f(epsilon) - f(-epsilon)) / (2 * epsilon))
    assert np.isclose(numerical, float(jax.grad(f)(0.)), rtol=.01, atol=1e-8)
    return {"status": "passed", "checks": ["single_and_double_legal_triple_unsafe",
            "attached_bond_only_bounded_gradient", "unsafe_argmax_fixed_in_trial",
            "both_safe_winners_zero_loss_and_gradient", "active_directional_derivative"]}
