"""Focused screen controls; numerical cases run only on allocated compute."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_root = Path(__file__).resolve().parents[1]
_script = _root/"scripts"/"protenix_chemistry_screen.py"
if not _script.exists():
    _script = _root/"protenix_chemistry_screen.py"
_spec = importlib.util.spec_from_file_location("_screen_under_test", _script)
screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen)


def configuration(**changes):
    result = {"version": screen.VERSION, "updates": 20000, "init_seed": 17,
              "initialization": dict(screen.SPARSE), "temperature": 1.,
              "chemistry": {"mode": "constant", "coefficient": 1.},
              "binding": {"mode": "always"}}
    result.update(changes)
    return result


@pytest.mark.parametrize("changed", [
    {"temperature": True}, {"temperature": 0}, {"updates": 0}, {"init_seed": True},
    {"noise": .4}, {"chemistry": {"mode": "constant", "coefficient": True}},
    {"initialization": {"mode": "chain"}}, {"binding": {"mode": "wait_forever"}},
])
def test_metadata_rejects_undeclared_or_ambiguous_settings(changed):
    with pytest.raises(ValueError):
        screen.validate_screen(configuration(**changed))


def test_delayed_baseline_keeps_original50000_horizon():
    config = configuration(chemistry={"mode": "delayed", "initial": .001, "final": 100.,
                          "hold_updates": 5000, "total_updates": 50000})
    policy = screen.Policy(config)
    selected = {}
    for state in range(50001):
        result = policy.source(state, False, 1., 1.)
        if state in (0, 4999, 5000, 19999, 49999, 50000):
            selected[state] = result["repair_weight"]
    assert selected[0] == selected[4999] == .001
    assert selected[5000] > .001
    assert selected[19999] < 1.
    assert selected[49999] == pytest.approx(100.)
    assert selected[50000] == pytest.approx(100.)


def test_adaptive_weight_bounded_and_flat_chemistry_holds_last_value():
    config = configuration(chemistry={"mode": "adaptive", "ratio_target": 1.,
        "minimum": .1, "maximum": 100., "ema": .9, "initial": 1.})
    policy = screen.Policy(config)
    first = policy.source(0, False, 10., 1.)
    assert first["repair_weight"] == pytest.approx(1.9)
    assert policy.source(1, False, 0., 0.)["repair_weight"] == first["repair_weight"]
    assert 1.9 < policy.source(2, False, 1e30, 1.)["repair_weight"] <= 100
    with pytest.raises(ValueError, match="once"):
        policy.source(2, False, 1., 1.)


@pytest.mark.parametrize("valid_at,reason", [(2, "valid_connected_source"), (None, "warmup_cap_without_validity")])
def test_bounded_phase_exit_never_stalls(valid_at, reason):
    policy = screen.Policy(configuration(binding={"mode": "validity_first", "max_chemistry_steps": 5000, "ramp_steps": 2000}))
    exit_state = 5000 if valid_at is None else valid_at
    values = {}
    for state in range(exit_state+2001):
        values[state] = policy.source(state, state == valid_at, 1., 1.)
    assert values[exit_state]["binding_weight"] == 0
    assert values[exit_state+1000]["binding_weight"] == .5
    assert values[exit_state+2000]["binding_weight"] == 1
    assert policy.phase_exit_reason == reason


@pytest.fixture
def logits_type():
    # Parent packaging includes the actual historical helper, which imports
    # the single current GraphLogits class rather than an independent copy.
    return screen.logits_class()


@pytest.mark.parametrize("seed", [17, 23, 29])
def test_sparse_initialization_is_paired_positive_and_hard_connected(seed, logits_type):
    import numpy as np
    config = configuration(init_seed=seed)
    left = screen.initialize(config, None, logits_type)
    right = screen.initialize(config, None, logits_type)
    np.testing.assert_array_equal(left.atoms, right.atoms)
    np.testing.assert_array_equal(left.bonds, right.bonds)
    graph = left.graph()
    assert np.asarray(graph.X).shape == (25, 6)
    assert np.asarray(graph.E_upper).shape == (300, 5)
    assert np.all(np.asarray(graph.X) > 0) and np.all(np.asarray(graph.E_upper) > 0)
    labels = np.asarray(graph.E_upper).argmax(-1)
    assert 24 <= np.count_nonzero(labels) < 60
    adjacency = [set() for _ in range(25)]
    for (i, j), label in zip(((i, j) for i in range(25) for j in range(i+1, 25)), labels):
        if label:
            adjacency[i].add(j); adjacency[j].add(i)
    seen, frontier = {0}, [0]
    while frontier:
        for neighbor in adjacency[frontier.pop()]-seen:
            seen.add(neighbor); frontier.append(neighbor)
    assert len(seen) == 25
    assert len(set(np.asarray(graph.X).argmax(-1))) > 1


def test_dense_control_preserves_original_logits(tmp_path, logits_type):
    import numpy as np
    atoms = np.arange(150, dtype=np.float32).reshape(25, 6)/100
    bonds = np.arange(1500, dtype=np.float32).reshape(300, 5)/100
    path = tmp_path/"original.npz"
    np.savez_compressed(path, atom_logits=atoms, bond_logits=bonds)
    point = screen.initialize(configuration(initialization={"mode": "dense_original"}), path, logits_type)
    np.testing.assert_array_equal(point.atoms, atoms)
    np.testing.assert_array_equal(point.bonds, bonds)


def point3(logits_type, temperature):
    import jax.numpy as jnp
    atoms = jnp.asarray([[3., .2, .1, -.2, -.3, -.5]]*3, dtype=jnp.float32)
    bonds = jnp.asarray([[.2, 3., -.1, -.3, -.5], [3., .2, -.1, -.3, -.5], [.2, 3., -.1, -.3, -.5]], dtype=jnp.float32)
    return logits_type(atoms, bonds, temperature)


def tiny_native(graph, *, key):
    import jax.numpy as jnp
    return jnp.sum(graph.X**2)+.3*jnp.sum(graph.E_upper**2), {"finite_path": jnp.asarray(1.)}


def actual_preparation(point):
    from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    spec = mosaic_spec(point.atoms.shape[0])
    return spec, prepare_failure_guidance(point.decoded(), spec, update=0,
        search=ProbabilityRepairConfig(), progress_search=ProgressSearchConfig())


@pytest.mark.parametrize("temperature", [1., 2.])
def test_native_temperature_pullback_matches_independent_full_autodiff(logits_type, temperature):
    import equinox as eqx
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.state import RelaxedGraph
    point = point3(logits_type, temperature)
    spec, prepared = actual_preparation(point)
    key = jax.random.key(17)
    units = screen.evaluate_units(tiny_native, point, prepared, None, spec, FailureGuidanceConfig(), key)
    # The historical placement helper deliberately reads concrete .sharding;
    # it is not an AD entry point. Differentiate an independent mathematical
    # reference directly, including the temperature in BOTH softmax branches.
    def reference(value):
        graph = RelaxedGraph(jax.nn.softmax(value.atoms/temperature, -1),
                             jax.nn.softmax(value.bonds/temperature, -1))
        return tiny_native(graph, key=key)[0]
    expected = eqx.filter_grad(reference)(point)
    for name in ("atoms", "bonds"):
        np.testing.assert_allclose(getattr(units["binding_gradient"], name), getattr(expected, name), rtol=2e-5, atol=2e-7)


def test_neighbor_private64_restores_native32_and_keeps_attached_bond_gradient(logits_type):
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.environment import prepare_neighbor_environment
    assert not jax.config.x64_enabled
    point = point3(logits_type, 2.)
    spec, prepared = actual_preparation(point)
    environment = prepare_neighbor_environment(point.decoded(), spec, {"version": "common-organic-neighbor-v1", "weight": 1.})
    units = screen.evaluate_units(tiny_native, point, prepared, environment, spec, FailureGuidanceConfig(), jax.random.key(17))
    assert not jax.config.x64_enabled
    assert np.asarray(units["neighbor_gradient"].bonds).dtype == np.float32
    assert np.linalg.norm(np.asarray(units["neighbor_gradient"].bonds)) > 0
    np.testing.assert_array_equal(units["neighbor_gradient"].atoms, np.zeros_like(point.atoms))
    assert np.isfinite(units["neighbor"])


def test_weighted_sum_rejects_overflow_in_combined_gradient(logits_type):
    import jax.numpy as jnp
    from types import SimpleNamespace
    point = point3(logits_type, 1.)
    big = logits_type(jnp.full_like(point.atoms, 1e38), jnp.full_like(point.bonds, 1e38), 1.)
    zero = logits_type(jnp.zeros_like(point.atoms), jnp.zeros_like(point.bonds), 1.)
    units = {"logits": point, "native": SimpleNamespace(value=0.), "repair": 0., "neighbor": 0.,
             "binding_gradient": zero, "chemistry_gradient": big, "neighbor_gradient": zero}
    with pytest.raises(ValueError, match="gradient"):
        screen.weighted(units, {"binding_weight": 1., "repair_weight": 100.})


def test_sampling_retains_all128_draws_without_changing_model_key(tmp_path, logits_type):
    import numpy as np
    point = point3(logits_type, 1.)
    result = screen.sample_validity(point, 17, 0, tmp_path)
    assert result["draws"] == 128
    assert result["strict_valid_connected"]+sum(result["failure_categories"].values()) == 128
    assert result["native_model_key_changed"] is False
    assert result["binding_scores_evaluated_for_samples"] is False
    with np.load(tmp_path/"samples-00000.npz", allow_pickle=False) as samples:
        assert samples["atom_indices"].shape == (128, 3)
        assert samples["bond_indices"].shape == (128, 3)


def test_tiny_loop_keeps_source_teacher_fixed_during_trial_and_reuses_native(tmp_path, logits_type, monkeypatch):
    import jax
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    spec = mosaic_spec(3)
    calls = []
    original = screen.evaluate_units
    def evaluate(*args, **kwargs):
        calls.append((args[2].receipt["source_update"], kwargs.get("native") is not None))
        return original(*args, **kwargs)
    monkeypatch.setattr(screen, "evaluate_units", evaluate)
    def prepare(point, state):
        from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
        from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
        from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
        return prepare_failure_guidance(point.decoded(), spec, update=state,
             search=ProbabilityRepairConfig(), progress_search=ProgressSearchConfig())
    result = screen.run_loop(tiny_native, point3(logits_type, 1.), prepare, spec, FailureGuidanceConfig(),
        jax.random.key(17), configuration(updates=2), tmp_path/"study")
    assert result["applied_updates"] == 2
    assert calls == [(0, False), (0, False), (1, True), (1, False), (2, True)]
    states = [json.loads(s) for s in (tmp_path/"study"/"states.jsonl").read_text().splitlines()]
    updates = [json.loads(s) for s in (tmp_path/"study"/"updates.jsonl").read_text().splitlines()]
    assert [s["step"] for s in states] == [0, 1, 2]
    assert [u["step"] for u in updates] == [0, 1]
    assert updates[0]["preparation_identity"] == states[0]["preparation_identity"]
    assert updates[1]["preparation_identity"] == states[1]["preparation_identity"]
    assert all(u["candidate_count"] == 1 and u["rate"] == .3 for u in updates)
    assert (tmp_path/"study"/"latest-state.npz").is_file()
    assert (tmp_path/"study"/"snapshots"/"step-00002.npz").is_file()


@pytest.mark.parametrize('combined', [False, True])
def test_interrupted_checkpoint_matches_uninterrupted_policy_targets_and_updates(tmp_path, logits_type, combined):
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    config=configuration(updates=22)
    if combined:
        config.update(temperature=2.,neighbor_environment={'version':'common-organic-neighbor-v1','weight':.1},
            chemistry={'mode':'adaptive','initial':1.,'ratio_target':1.,'minimum':1.,'maximum':100.,'ema':.9},
            binding={'mode':'validity_first','max_chemistry_steps':5000,'ramp_steps':2000})
    spec=mosaic_spec(3)
    def prepare(point,state):
        return prepare_failure_guidance(point.decoded(),spec,update=state,
            search=ProbabilityRepairConfig(),progress_search=ProgressSearchConfig())
    args=(tiny_native,point3(logits_type,config['temperature']),prepare,spec,FailureGuidanceConfig(),jax.random.key(17),config)
    screen.run_loop(*args,tmp_path/'full')
    screen.run_loop(*args,tmp_path/'partial',steps=20)
    parent=tmp_path/'partial'
    state=json.loads((parent/'states.jsonl').read_text().splitlines()[-1])
    resume={'offset':20,'state':state,'snapshot':screen.binding(parent/'snapshots/step-00020.npz'),
            'reference':screen.binding(parent/'references/step-00020.json.gz')}
    with np.load(resume['snapshot']['path'],allow_pickle=False) as arrays:
        import jax.numpy as jnp
        point=logits_type(jnp.asarray(arrays['atom_logits']),jnp.asarray(arrays['bond_logits']),config['temperature'])
    report=screen.run_loop(tiny_native,point,prepare,spec,FailureGuidanceConfig(),jax.random.key(17),config,
                           tmp_path/'resumed',resume=resume)
    assert report['applied_updates']==2 and report['global_start']==20
    rows=[json.loads(s) for s in (tmp_path/'resumed/states.jsonl').read_text().splitlines()]
    full=[json.loads(s) for s in (tmp_path/'full/states.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==[0,1,2] and [r['global_step'] for r in rows]==[20,21,22]
    for a,b in zip(rows,full[20:]):
        for key in ('repair_weight','binding_weight','phase_exit_state','preparation_identity','neighbor_preparation_identity','contact_loss'):
            assert a[key]==b[key]
    with np.load(tmp_path/'full/latest-state.npz') as a,np.load(tmp_path/'resumed/latest-state.npz') as b:
        np.testing.assert_array_equal(a['atom_logits'],b['atom_logits'])
        np.testing.assert_array_equal(a['bond_logits'],b['bond_logits'])
