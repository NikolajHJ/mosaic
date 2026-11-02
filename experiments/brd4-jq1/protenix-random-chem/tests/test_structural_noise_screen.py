"""Initializer, binding-only noise and real-loop continuity; arrays on allocated CPU."""
from copy import deepcopy
import json

import pytest

from test_protenix_chemistry_screen import screen, configuration, logits_type, point3, tiny_native


@pytest.mark.parametrize('step,expected', [(0, .4), (10000, .2), (19999, .00002), (20000, 0.), (29999, 0.)])
def test_predeclared_global_noise_schedule(step, expected):
    assert screen.noise_ratio(screen.ANNEALED_NOISE, step) == pytest.approx(expected)


@pytest.mark.parametrize('change', [{'initial_ratio': float('nan')}, {'seed': True},
                                  {'zero_at_update': 0}, {'initial_ratio': 4.}, {'extra': 1}])
def test_bad_noise_metadata_rejected(change):
    with pytest.raises(ValueError):
        screen.validate_noise({**screen.ANNEALED_NOISE, **change})


def test_chain_mixture_probabilities_and_paired_small_perturbation(logits_type):
    import numpy as np
    config = configuration(initialization={**screen.CHAIN_MIX, 'logit_noise_std': 0.})
    plain = screen.initialize(config, None, logits_type)
    graph = plain.graph()
    expected_atoms = np.full((25, 6), .5/6, np.float32); expected_atoms[:, 0] += .5
    expected_bonds = np.full((300, 5), .1, np.float32)
    for k, (i, j) in enumerate((i, j) for i in range(25) for j in range(i+1, 25)):
        expected_bonds[k, 1 if j == i+1 else 0] += .5
    # FP32 log followed by normalized exp is not an exact round trip.
    np.testing.assert_allclose(graph.X, expected_atoms, rtol=8*np.finfo(np.float32).eps)
    np.testing.assert_allclose(graph.E_upper, expected_bonds, rtol=8*np.finfo(np.float32).eps)
    noisy = configuration(initialization=dict(screen.CHAIN_MIX))
    paired = [screen.initialize(noisy, None, logits_type) for _ in range(2)]
    for name in ('atoms', 'bonds'):
        np.testing.assert_array_equal(getattr(paired[0], name), getattr(paired[1], name))
        delta = np.asarray(getattr(paired[0], name))-np.asarray(getattr(plain, name))
        np.testing.assert_allclose(delta.mean(axis=-1), 0., atol=3e-7)
        assert .035 < np.std(delta) < .06
    np.testing.assert_array_equal(paired[0].graph().X.argmax(-1), expected_atoms.argmax(-1))
    np.testing.assert_array_equal(paired[0].graph().E_upper.argmax(-1), expected_bonds.argmax(-1))
    assert np.all(np.asarray(paired[0].graph().X) > 0)


def test_noise_uses_binding_logit_gradient_not_weighted_chemistry(logits_type):
    import jax.numpy as jnp
    import numpy as np
    point = point3(logits_type, 1.)
    native = logits_type(jnp.arange(18, dtype=jnp.float32).reshape(3, 6)/10,
                        jnp.arange(15, dtype=jnp.float32).reshape(3, 5)/20, 1.)
    records = []
    for multiplier in (1., 100.):
        combined = logits_type(native.atoms*multiplier, native.bonds*multiplier, 1.)
        stream = screen.BindingNoise(screen.ANNEALED_NOISE)
        result, record = stream.propose(point, combined, native, .3, 0)
        records.append(record)
        for name in ('atoms', 'bonds'):
            metric = record['blocks'][name]
            assert metric['noise_norm'] == pytest.approx(.4*.3*np.linalg.norm(np.asarray(getattr(native, name), dtype=np.float64)), rel=5e-6)
            displacement = np.asarray(getattr(result, name))-(np.asarray(getattr(point, name))-.3*np.asarray(getattr(combined, name)))
            np.testing.assert_allclose(displacement.mean(axis=-1), 0., atol=5e-7)
    assert records[0]['rng_state_after'] == records[1]['rng_state_after']
    assert records[0]['blocks']['atoms']['requested_noise_norm'] == records[1]['blocks']['atoms']['requested_noise_norm']
    assert records[0]['chemistry_weight_used_for_noise_scale'] is False


def test_zero_binding_blocks_and_noiseless_settling_are_exact(logits_type):
    import jax.numpy as jnp
    import numpy as np
    point = point3(logits_type, 1.)
    gradient = logits_type(jnp.ones_like(point.atoms), jnp.ones_like(point.bonds), 1.)
    zero = logits_type(jnp.zeros_like(point.atoms), jnp.zeros_like(point.bonds), 1.)
    policy = {**screen.ANNEALED_NOISE, 'zero_at_update': 1}
    stream = screen.BindingNoise(policy)
    for step in (0, 1):
        before = stream.checkpoint()
        result, record = stream.propose(point, gradient, zero if step == 0 else gradient, .3, step)
        for name in ('atoms', 'bonds'):
            np.testing.assert_array_equal(getattr(result, name), getattr(point, name)-.3*getattr(gradient, name))
        if step == 1:
            assert before['rng_state'] == stream.checkpoint()['rng_state']
            assert record['ratio'] == 0.


def test_noise_stream_rejects_missing_or_wrong_offset(logits_type):
    with pytest.raises(ValueError, match='checkpoint'):
        screen.BindingNoise(screen.ANNEALED_NOISE, start=20)
    first = screen.BindingNoise(screen.ANNEALED_NOISE)
    with pytest.raises(ValueError, match='offset'):
        screen.BindingNoise(screen.ANNEALED_NOISE, start=1, saved=first.checkpoint())


def test_real_loop_restart_keeps_noise_teacher_and_post_annealing_updates(tmp_path, logits_type):
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    spec = mosaic_spec(3)
    config = configuration(updates=22, chemistry={'mode': 'constant', 'coefficient': 10.},
                           logit_noise={**screen.ANNEALED_NOISE, 'zero_at_update': 20})
    def prepare(point, state):
        return prepare_failure_guidance(point.decoded(), spec, update=state,
            search=ProbabilityRepairConfig(), progress_search=ProgressSearchConfig())
    def run(path, initial, **kwargs):
        return screen.run_loop(tiny_native, initial, prepare, spec, FailureGuidanceConfig(),
                               jax.random.key(17), config, path, **kwargs)
    initial = point3(logits_type, 1.)
    full, partial, child = [tmp_path/name for name in ('full', 'partial', 'child')]
    run(full, initial)
    run(partial, initial, steps=20)
    saved = [json.loads(s) for s in (partial/'states.jsonl').read_text().splitlines()][-1]
    resume = {'offset': 20, 'state': saved,
              'reference': screen.binding(partial/'references/step-00020.json.gz')}
    with np.load(partial/'snapshots/step-00020.npz') as data:
        import jax.numpy as jnp
        checkpoint = logits_type(jnp.asarray(data['atom_logits']), jnp.asarray(data['bond_logits']), 1.)
    run(child, checkpoint, resume=resume)
    with np.load(full/'latest-state.npz') as a, np.load(child/'latest-state.npz') as b:
        for name in ('atom_logits', 'bond_logits', 'X', 'E_upper', 'key_data'):
            np.testing.assert_array_equal(a[name], b[name])
    reference = [json.loads(s) for s in (full/'states.jsonl').read_text().splitlines()]
    continued = [json.loads(s) for s in (child/'states.jsonl').read_text().splitlines()]
    assert reference[-1]['noise_state'] == continued[-1]['noise_state']
    assert [r['preparation_identity'] for r in reference[20:]] == [r['preparation_identity'] for r in continued]
    updates = [json.loads(s) for s in (full/'updates.jsonl').read_text().splitlines()]
    assert all(u['preparation_identity'] == reference[u['global_step']]['preparation_identity'] for u in updates)
    assert [u['noise']['ratio'] for u in updates[-2:]] == [0., 0.]
