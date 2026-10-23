"""Allocated exact placement and trajectory-parameterization regressions."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry.state import GraphLogits, mosaic_spec
from mosaic.small_molecule.chemistry.initialization import initialize_graph
from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
from mosaic.small_molecule.chemistry import trajectory
from mosaic.small_molecule.state import graph_from_hard
from protenix_initialization import ProtenixReplayLogits, VERSION


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get('LSB_JOBID'), 'Numerical fixtures require allocated DTU HPC'


def point():
    hard = graph_from_hard([0] * 3, [1, 0, 1])
    return GraphLogits.from_graph(initialize_graph(hard, {'kind': 'chain_uniform', 'mixing': 0.5}))


def test_replay_keeps_exact_original_logits_and_stock_cpu_softmax():
    old = point()
    new = ProtenixReplayLogits(old.atoms, old.bonds)
    assert VERSION == 'protenix-fp32-cpu-atom-softmax-v1'
    for a, b in [(new.atoms, old.atoms), (new.bonds, old.bonds),
                 (new.graph().X, old.graph().X), (new.graph().E_upper, old.graph().E_upper)]:
        np.testing.assert_array_equal(a, b)
    assert new.graph().X.dtype == new.graph().E_upper.dtype == jnp.float32


def test_atom_transfer_returns_to_original_scorer_device_without_graph_mutation():
    old = point()
    new = ProtenixReplayLogits(old.atoms, old.bonds)
    before = np.asarray(new.atoms).copy()
    assert new.graph().X.sharding == new.atoms.sharding
    assert new.graph().E_upper.sharding == new.bonds.sharding
    np.testing.assert_array_equal(new.atoms, before)


def test_all_trials_and_snapshots_keep_the_declared_mapping(tmp_path, monkeypatch):
    old = point()
    initial = ProtenixReplayLogits(old.atoms, old.bonds)
    original = trajectory.evaluate_combined
    observed = []

    def evaluate(loss, logits, *args, **kwargs):
        assert type(logits) is ProtenixReplayLogits
        observed.append(id(logits))
        return original(loss, logits, *args, **kwargs)

    monkeypatch.setattr(trajectory, 'evaluate_combined', evaluate)
    spec = mosaic_spec(3)
    loss = lambda graph, key: (jnp.sum(graph.X * jnp.arange(6)) + jnp.sum(graph.E_upper * jnp.arange(5)), {})
    config = {'loss': None, 'repair_weight': 0.0, 'element_js_weight': 0.0,
              'native_name': 'contact', 'objective_mode': 'native_only'}
    report = trajectory.run_chemical_updates(
        loss, initial, prepare=lambda z, u: NativeOnlyPreparation(spec.fingerprint(), u),
        spec=spec, reference=jnp.ones(6), key=jax.random.key(17),
        directory=tmp_path / 'study', steps=2, step_size=0.3, max_backtracks=1,
        config=config, acceptance_policy='fixed_step',
    )
    assert report['status'] == 'passed' and report['counts']['applied'] == 2
    assert len(report['states']) == 3 and len(set(observed)) >= 3
    for row in report['updates']:
        assert len(row['attempts']) == 1 and row['accepted_step_size'] == 0.3
    with np.load(tmp_path / 'study' / report['states'][0]['snapshot']) as saved:
        np.testing.assert_array_equal(saved['X'], initial.graph().X)
        np.testing.assert_array_equal(saved['E_upper'], initial.graph().E_upper)


def test_regular_graph_logits_and_boltz_mapping_remain_stock():
    old = point()
    assert type(old) is GraphLogits
    np.testing.assert_array_equal(old.graph().X, jax.nn.softmax(old.atoms, -1))
    np.testing.assert_array_equal(old.graph().E_upper, jax.nn.softmax(old.bonds, -1))
