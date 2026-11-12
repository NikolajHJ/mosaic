import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mosaic_defog import Graph, Guidance, Prediction, cosine, linear
from mosaic_defog.graph import validate_arrays


def graph(n=4):
    return Graph(jnp.full((n, 6), 1 / 6), jnp.full((n * (n - 1) // 2, 5), 1 / 5))


class Teacher:
    def __init__(self):
        self.calls = []

    def predict(self, current, *, time):
        self.calls.append((np.array(current.X), np.array(current.E_upper), time))
        return Prediction(
            np.eye(6, dtype=np.float32)[np.zeros(len(current.X), int)],
            np.eye(5, dtype=np.float32)[np.zeros(len(current.E_upper), int)],
            time,
        )


def native(current, *, key=None):
    score = jnp.mean(current.X[:, 1]) + jnp.mean(current.E_upper[:, 2])
    return -score, {"contact_proxy": score}


def test_live_gradients_match_independent_mse_derivative():
    current, teacher = graph(), Teacher()
    before = tuple(np.array(v) for v in current)
    guidance = Guidance(teacher, weight=0.125, time=0.8)
    objective = guidance.prepare(native, current, 3)
    (value, aux), gradients = jax.value_and_grad(objective, has_aux=True)(current)
    target = teacher.predict(current, time=0.8)
    native_grad = jax.grad(lambda g: native(g)[0])(current)
    expected = Graph(
        native_grad.X + 0.125 * 2 * (current.X - target.X) / len(current.X),
        native_grad.E_upper
        + 0.125 * 2 * (current.E_upper - target.E_upper) / len(current.E_upper),
    )
    for actual, reference in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-7)
        assert np.isfinite(actual).all() and np.linalg.norm(actual) > 0
    for actual, reference in zip(current, before, strict=True):
        np.testing.assert_array_equal(actual, reference)
    assert np.isfinite(value) and aux["defog_weight"] == 0.125


def test_query_once_per_preparation_never_inside_jit_or_backward():
    teacher, current = Teacher(), graph()
    objective = Guidance(teacher).prepare(native, current)
    evaluate = jax.jit(jax.value_and_grad(lambda g: objective(g)[0]))
    evaluate(current)
    evaluate(Graph(current.X * 0.9 + 0.1 / 6, current.E_upper))
    assert len(teacher.calls) == 1
    Guidance(teacher).prepare(native, current, 1)
    assert len(teacher.calls) == 2


def test_factory_and_schedules_use_current_graph_and_zero_based_step():
    teacher, current = Teacher(), graph()
    factory = Guidance(
        teacher, weight=linear(0, 0.1, 3), time=linear(0.2, 0.95, 3)
    ).loss_factory(native, 3)
    objective = factory(current, 2)
    _, aux = objective(current)
    assert aux["defog_weight"] == pytest.approx(0.1)
    assert teacher.calls[0][2] == pytest.approx(0.95)
    assert linear(0, 1, 3)(20) == 1
    schedule = cosine(0, 0.1, 1500)
    assert [schedule(s) for s in (0, 750, 1500)] == pytest.approx([0, 0.1, 0])


def test_one_atom_custom_target_has_finite_zero_bond_loss():
    current = graph(1)
    target = Prediction(np.asarray(current.X), np.zeros((0, 5), np.float32), 0.8)
    value, gradients = jax.value_and_grad(target.jax_loss)(current)
    assert float(value) == 0
    assert gradients.E_upper.shape == (0, 5)


@pytest.mark.parametrize(
    "weight,time", [(-1, 0.8), (float("nan"), 0.8), (0.1, 1.1), (0.1, float("inf"))]
)
def test_invalid_schedules_fail_before_query(weight, time):
    teacher = Teacher()
    with pytest.raises(ValueError):
        Guidance(teacher, weight=weight, time=time).prepare(native, graph())
    assert not teacher.calls


def test_wrong_shapes_and_metric_collisions_fail():
    current = graph()
    wrong = Prediction(np.full((3, 6), 1 / 6), np.full((3, 5), 1 / 5), 0.8)
    with pytest.raises(ValueError, match="slots"):
        wrong.jax_loss(current)
    objective = Guidance(Teacher()).prepare(
        lambda g, key=None: (g.X.sum(), {"defog_loss": 1}), current
    )
    with pytest.raises(ValueError, match="collide"):
        objective(current)


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "off_simplex", "integer"])
def test_host_input_validation(failure):
    x, e = map(np.array, graph())
    if failure == "shape":
        e = e[:-1]
    elif failure == "nonfinite":
        x[0, 0] = np.nan
    elif failure == "off_simplex":
        e[0, 0] = 0.9
    else:
        x = x.astype(int)
    with pytest.raises((ValueError, TypeError)):
        validate_arrays(x, e)


def test_import_is_framework_and_experiment_independent():
    code = "import sys,mosaic_defog; assert not any(m in sys.modules for m in ('torch','jax','mosaic')); print(mosaic_defog.__version__)"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_prediction_owns_its_targets():
    x, e = map(np.array, graph())
    target = Prediction(x, e, 0.8)
    x[:] = 0
    assert np.allclose(target.X.sum(-1), 1)
    with pytest.raises(ValueError):
        target.E_upper[0, 0] = 1


@pytest.mark.parametrize(
    "mode", ["none", "valence", "connected", "sparse_only", "sparse"]
)
def test_current_mosaic_projection_modes_remain_external(mode):
    operators = pytest.importorskip("mosaic.small_molecule.Stage2.operators")
    state = pytest.importorskip("mosaic.small_molecule.core.state")
    optimizers = pytest.importorskip("mosaic.small_molecule.core.optimizers")
    current = state.RelaxedGraph(*graph())
    teacher = Teacher()
    guidance = Guidance(teacher)
    config = operators.Stage2Config(mode=mode, readout_caps=True)
    project = operators.post_step(config)
    for step in range(3):
        objective = guidance.prepare(native, current, step)
        evaluation = optimizers.evaluate_graph(objective, current, jax.random.key(7))
        proposal = optimizers.projected_step(
            evaluation, eta_X=0.1, eta_E=0.1
        ).next_graph
        current, receipt = project(proposal, step + 1)
        assert receipt["mode"] == mode
        state.validate_graph(current)
    assert len(teacher.calls) == 3
