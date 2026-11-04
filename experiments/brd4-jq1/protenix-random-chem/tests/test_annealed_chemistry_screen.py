"""Scheduled chemistry/temperature and checkpoint controls; arrays allocated only."""
from copy import deepcopy
import json
import math

import pytest

from test_protenix_chemistry_screen import screen, configuration, point3, tiny_native


def scheduled_configuration(updates=50000, horizon=50000):
    return configuration(
        updates=updates,
        temperature=.7,
        temperature_schedule={"version": "linear-temperature-v1", "initial": .7,
                              "final": .4, "total_updates": horizon},
        chemistry={"mode": "ramp_hold", "initial": 1., "final": 30.,
                   "hold_updates": 20000, "ramp_end": 30000},
        logit_noise={**screen.ANNEALED_NOISE, "zero_at_update": 30000})


@pytest.mark.parametrize("step,expected", [(0, .7), (20000, .58), (30000, .52),
                                          (50000, .4), (70000, .4)])
def test_temperature_uses_global_horizon_and_clamps_at_endpoint(step, expected):
    assert screen.temperature_at(scheduled_configuration(), step) == pytest.approx(expected)


def test_actual_chemistry_hold_ramp_hold_boundaries_and_noise_end():
    config = scheduled_configuration()
    policy = screen.Policy(config)
    selected = {}
    for state in range(50001):
        active = policy.source(state, False, 1., 1.)
        if state in (0, 19999, 20000, 25000, 30000, 50000):
            selected[state] = active["repair_weight"]
    assert selected[0] == selected[19999] == selected[20000] == 1.
    assert selected[25000] == pytest.approx(math.sqrt(30.))
    assert selected[30000] == selected[50000] == pytest.approx(30.)
    assert screen.noise_ratio(config["logit_noise"], 0) == .4
    assert screen.noise_ratio(config["logit_noise"], 30000) == 0.
    assert screen.noise_ratio(config["logit_noise"], 70000) == 0.


@pytest.mark.parametrize("change", [{"total_updates": 0}, {"initial": float("nan")},
                                    {"final": 0.}, {"initial": True},
                                    {"version": "undeclared"}, {"extra": 1}])
def test_malformed_temperature_schedule_rejected(change):
    config = scheduled_configuration()
    config["temperature_schedule"].update(change)
    with pytest.raises(ValueError):
        screen.validate_screen(config)


@pytest.mark.parametrize("change", [{"hold_updates": 30000}, {"ramp_end": 20000},
                                    {"final": float("nan")}, {"hold_updates": True}])
def test_malformed_chemistry_ramp_rejected(change):
    config = scheduled_configuration()
    config["chemistry"].update(change)
    with pytest.raises(ValueError):
        screen.validate_screen(config)


@pytest.mark.parametrize("temperature", [.7, .4])
def test_cold_binding_pullback_matches_independent_autodiff(temperature):
    import equinox as eqx
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.state import RelaxedGraph
    from test_protenix_chemistry_screen import actual_preparation
    point = point3(screen.logits_class(scheduled=True), temperature)
    spec, prepared = actual_preparation(point)
    key = jax.random.key(17)
    units = screen.evaluate_units(tiny_native, point, prepared, None, spec,
                                  FailureGuidanceConfig(), key)
    def reference(value):
        graph = RelaxedGraph(jax.nn.softmax(value.atoms/temperature, -1),
                             jax.nn.softmax(value.bonds/temperature, -1))
        return tiny_native(graph, key=key)[0]
    expected = eqx.filter_grad(reference)(point)
    for name in ("atoms", "bonds"):
        np.testing.assert_allclose(getattr(units["binding_gradient"], name),
                                   getattr(expected, name), rtol=2e-5, atol=2e-7)


def test_temperature_is_dynamic_without_retracing_or_an_optimized_gradient():
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    cls = screen.logits_class(scheduled=True)
    points = [point3(cls, temperature) for temperature in (.7, .69, .4)]
    assert all(eqx.is_array(p.temperature) and p.temperature.dtype == np.float32 for p in points)
    traces = []
    @eqx.filter_jit
    def score(point):
        traces.append("trace")
        return jnp.sum(jax.nn.softmax(point.decoded().atoms, -1)**2)
    values = [float(score(points[0]))]
    initial_trace_count = len(traces)
    assert initial_trace_count > 0
    values.extend(float(score(p)) for p in points[1:])
    assert len(traces) == initial_trace_count
    assert values[0] != values[-1]
    derivative = eqx.filter_grad(lambda p: jnp.sum(jax.nn.softmax(p.decoded().atoms, -1)**2))(points[0])
    assert float(derivative.temperature) == 0.


def preparation(spec):
    from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    def prepare(point, state):
        return prepare_failure_guidance(point.decoded(), spec, update=state,
            search=ProbabilityRepairConfig(), progress_search=ProgressSearchConfig())
    return prepare


def small_configuration(updates=22, horizon=22):
    config = scheduled_configuration(updates=updates, horizon=horizon)
    config["chemistry"].update(hold_updates=5, ramp_end=10)
    config["logit_noise"]["zero_at_update"] = 20
    return config


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("horizon,last_cache_reused", [(4, False), (1, True)])
def test_cooling_keeps_trial_teacher_and_temperature_fixed_then_refreshes_native(
        tmp_path, monkeypatch, horizon, last_cache_reused):
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    config = small_configuration(updates=2, horizon=horizon)
    config["chemistry"] = {"mode": "constant", "coefficient": 1.}
    point = point3(screen.logits_class(scheduled=True), .7)
    spec = mosaic_spec(3)
    calls = []
    original = screen.evaluate_units
    def evaluate(*args, **kwargs):
        calls.append((args[2].receipt["source_update"], float(np.asarray(args[1].temperature)),
                      kwargs.get("native") is not None))
        return original(*args, **kwargs)
    monkeypatch.setattr(screen, "evaluate_units", evaluate)
    result = screen.run_loop(tiny_native, point, preparation(spec), spec, FailureGuidanceConfig(),
                             jax.random.key(17), config, tmp_path/"study")
    assert result["applied_updates"] == 2
    assert [c[0] for c in calls] == [0, 0, 1, 1, 2]
    assert [c[2] for c in calls] == [False, False, False, False, last_cache_reused]
    for actual, expected in zip(calls, (0, 0, 1, 1, 2)):
        assert actual[1] == pytest.approx(screen.temperature_at(config, expected), abs=1e-7)
    states, updates = rows(tmp_path/"study/states.jsonl"), rows(tmp_path/"study/updates.jsonl")
    for step, update in enumerate(updates):
        assert update["preparation_identity"] == states[step]["preparation_identity"]
        assert update["source_temperature"] == update["trial_temperature"] == states[step]["temperature"]
        assert update["next_temperature"] == states[step+1]["temperature"]
        assert update["candidate_count"] == 1 and update["rate"] == .3


def checkpoint(parent, cls):
    import jax.numpy as jnp
    import numpy as np
    state = rows(parent/"states.jsonl")[-1]
    step = state["global_step"]
    resume = {"offset": step, "state": deepcopy(state),
              "snapshot": screen.binding(parent/"snapshots"/f"step-{step:05d}.npz"),
              "reference": screen.binding(parent/"references"/f"step-{step:05d}.json.gz")}
    with np.load(resume["snapshot"]["path"], allow_pickle=False) as data:
        point = cls(jnp.asarray(data["atom_logits"]), jnp.asarray(data["bond_logits"]), state["temperature"])
    return point, resume


def test_interrupted_cooling_restores_global_schedule_teacher_and_noise_stream(tmp_path):
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    cls = screen.logits_class(scheduled=True)
    config = small_configuration()
    spec = mosaic_spec(3)
    def run(path, point, **kwargs):
        return screen.run_loop(tiny_native, point, preparation(spec), spec, FailureGuidanceConfig(),
                               jax.random.key(17), config, path, **kwargs)
    full, partial, child = [tmp_path/name for name in ("full", "partial", "child")]
    run(full, point3(cls, .7))
    run(partial, point3(cls, .7), steps=20)
    point, resume = checkpoint(partial, cls)
    run(child, point, resume=resume)
    with np.load(full/"latest-state.npz") as a, np.load(child/"latest-state.npz") as b:
        for name in ("atom_logits", "bond_logits", "X", "E_upper", "key_data", "temperature"):
            np.testing.assert_array_equal(a[name], b[name])
    expected, actual = rows(full/"states.jsonl")[20:], rows(child/"states.jsonl")
    assert [r["global_step"] for r in actual] == [20, 21, 22]
    for a, b in zip(expected, actual):
        for key in ("repair_weight", "temperature", "preparation_identity", "contact_loss", "noise_state"):
            assert a[key] == b[key]
    assert [r["noise"]["ratio"] for r in rows(child/"updates.jsonl")] == [0., 0.]


def test_explicit_tail_policy_change_applies100_immediately_and_keeps_endpoint_state(tmp_path):
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    cls = screen.logits_class(scheduled=True)
    config = small_configuration(updates=20, horizon=20)
    spec = mosaic_spec(3)
    parent = tmp_path/"parent"
    screen.run_loop(tiny_native, point3(cls, .7), preparation(spec), spec, FailureGuidanceConfig(),
                    jax.random.key(17), config, parent)
    parent_before = {name: (parent/name).read_bytes() for name in ("states.jsonl", "updates.jsonl", "latest-state.npz")}
    point, resume = checkpoint(parent, cls)
    assert resume["state"]["repair_weight"] == 30.
    resume["policy_change"] = True
    child_config = deepcopy(config)
    child_config.update(updates=22, chemistry={"mode": "constant", "coefficient": 100.})
    child = tmp_path/"tail"
    screen.run_loop(tiny_native, point, preparation(spec), spec, FailureGuidanceConfig(),
                    jax.random.key(17), child_config, child, resume=resume)
    states = rows(child/"states.jsonl")
    assert [r["repair_weight"] for r in states] == [100., 100., 100.]
    assert all(r["temperature"] == pytest.approx(.4, abs=1e-7) for r in states)
    assert states[0]["preparation_identity"] == resume["state"]["preparation_identity"]
    assert [r["noise"]["ratio"] for r in rows(child/"updates.jsonl")] == [0., 0.]
    for name, contents in parent_before.items():
        assert (parent/name).read_bytes() == contents
    with np.load(resume["snapshot"]["path"]) as a, np.load(child/"snapshots/step-00020.npz") as b:
        for name in ("atom_logits", "bond_logits", "X", "E_upper", "key_data"):
            np.testing.assert_array_equal(a[name], b[name])

@pytest.mark.parametrize("totals,selected", [
    ([0.9, 0.8, 0.7, 0.71, 0.8, 0.85, 0.9, 0.92, 0.95], 2),
    ([1.1, 1.0, 1.01], None),
    ([0.95, 0.95, 0.96], 0),
    ([1.1, 1.08, 0.9], 2),
])
def test_halvings_choose_best_strict_decrease(totals, selected):
    assert screen.best_decreasing_trial(1.0, totals) == selected


def test_halvings_reject_a_nonfinite_later_trial_even_after_decrease():
    with pytest.raises(ValueError, match="Nonfinite"):
        screen.best_decreasing_trial(1.0, [0.5, float("nan")])
