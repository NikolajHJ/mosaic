"""Allocated JAX tests for scorer-only baselines and 25-slot callback shapes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec
from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
from mosaic.small_molecule.chemistry.trajectory import evaluate_combined, run_chemical_updates
from mosaic.small_molecule.state import graph_from_hard, mix_with_uniform


def setup25():
    hard = graph_from_hard([0] * 25, [int(j == i + 1) for i in range(25) for j in range(i + 1, 25)])
    return GraphLogits.from_graph(mix_with_uniform(hard, .5)), mosaic_spec(25)


def loss(graph, *, key):
    return (graph.X * jnp.arange(6)).sum() + .2 * (graph.E_upper * jnp.arange(5)).sum(), {}


def options():
    return {"objective_mode": "native_only", "repair_weight": 0., "element_js_weight": 0., "native_name": "affinity"}


def test_native_only_25_slot_loss_does_not_call_repair_or_composition(monkeypatch):
    from mosaic.small_molecule.chemistry import trajectory
    def forbidden(*args, **kwargs):
        raise AssertionError("No chemistry or composition function may execute")
    monkeypatch.setattr(trajectory, "failure_aware_repair_loss", forbidden)
    monkeypatch.setattr(trajectory, "composition_statistics", forbidden)
    state, spec = setup25()
    evaluation = evaluate_combined(loss, state, NativeOnlyPreparation(spec.fingerprint(), 0), spec, jnp.ones(6), key=jax.random.key(17), config=options())
    expected = jax.grad(lambda v: loss(v.graph(), key=None)[0])(state)
    np.testing.assert_allclose(evaluation.gradients.atoms, expected.atoms, atol=1e-7)
    np.testing.assert_allclose(evaluation.gradients.bonds, expected.bonds, atol=1e-7)
    assert "chemistry_loss" not in evaluation.terms


@pytest.mark.parametrize("weight", ["repair_weight", "element_js_weight"])
def test_native_only_rejects_hidden_auxiliary_guidance(weight):
    state, spec = setup25()
    config = {**options(), weight: .1}
    with pytest.raises(ValueError, match="inactive"):
        evaluate_combined(loss, state, NativeOnlyPreparation(spec.fingerprint(), 0), spec, jnp.ones(6), key=jax.random.key(17), config=config)


def test_native_only_trajectory_records_all_pairs_without_fake_teacher_pass(tmp_path):
    state, spec = setup25()
    result = run_chemical_updates(loss, state, prepare=lambda z, u: NativeOnlyPreparation(spec.fingerprint(), u), spec=spec, reference=jnp.ones(6), key=jax.random.key(17), directory=tmp_path / "study", steps=2, step_size=.03, max_backtracks=10, config=options())
    assert result["status"] == "passed"
    assert len(result["states"]) == 3
    assert all(s["chemistry_preparation"] == [] for s in result["states"])
    with np.load(tmp_path / "study" / result["states"][-1]["snapshot"]) as saved:
        assert saved["X"].shape == (25, 6)
        assert saved["E_upper"].shape == (300, 5)


def test_callback_keeps_25_slot_vjp_and_independent_primal():
    from mosaic.small_molecule.boltz2.bridge import affinity_loss
    class Toy:
        def evaluate(self, x, e, *, gradient):
            assert x.shape == (25, 6) and e.shape == (300, 5)
            gx = np.broadcast_to(np.arange(6, dtype=np.float32), x.shape).copy()
            ge = np.broadcast_to(np.arange(5, dtype=np.float32), e.shape).copy()
            values = np.asarray([(x * gx).sum() + (e * ge).sum(), .5, 0], dtype=np.float32)
            return values, gx if gradient else np.zeros_like(x), ge if gradient else np.zeros_like(e)
    state, _ = setup25()
    native = affinity_loss(Toy())
    (value, _), gradient = jax.value_and_grad(native, has_aux=True)(state.graph(), key=jax.random.key(17))
    primal, _ = native(state.graph(), key=jax.random.key(17))
    assert float(value) == float(primal)
    assert gradient.X.shape == (25, 6) and gradient.E_upper.shape == (300, 5)
