"""Allocated JAX proofs of unconditional finite fixed updates and honest counts."""

# Shared test import paths; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__, include_tests=True)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec
from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
from mosaic.small_molecule.chemistry.trajectory import evaluate_combined, run_chemical_updates
from mosaic.small_molecule.state import graph_from_hard, mix_with_uniform


def setup():
    hard = graph_from_hard([0]*25, [int(j==i+1) for i in range(25) for j in range(i+1,25)])
    return GraphLogits.from_graph(mix_with_uniform(hard,.5)), mosaic_spec(25)


def run(tmp_path, loss, *, policy='fixed_step', trials=1):
    initial,spec=setup()
    options={'objective_mode':'native_only','repair_weight':0.,'element_js_weight':0.,'native_name':'contact'}
    result=run_chemical_updates(loss,initial,prepare=lambda z,u: NativeOnlyPreparation(spec.fingerprint(),u),
        spec=spec,reference=jnp.ones(6),key=jax.random.key(17),directory=tmp_path/'study',steps=1,
        step_size=.3,max_backtracks=trials,config=options,acceptance_policy=policy)
    return initial,spec,options,result


def test_full_fixed_step_is_applied_even_when_actual_loss_increases(tmp_path):
    def loss(graph,*,key):
        return 1e4*(graph.X[0,0]-.6)**2 + .1*jnp.sum(graph.E_upper[0]*jnp.arange(5)),{}
    initial,spec,options,r=run(tmp_path,loss)
    u=r['updates'][0]
    assert u['fixed_after'] > u['fixed_before'] and u['fixed_increased']
    assert len(u['attempts'])==1 and u['accepted_step_size']==.3 and not u['fallback_used']
    assert r['counts']=={'applied':1,'strict_improving':0,'forced_fallback':0,'no_op':0}
    e=evaluate_combined(loss,initial,NativeOnlyPreparation(spec.fingerprint(),0),spec,jnp.ones(6),key=jax.random.key(17),config=options)
    with np.load(tmp_path/'study'/r['states'][1]['snapshot']) as saved:
        np.testing.assert_allclose(saved['atom_logits'],np.asarray(initial.atoms-.3*e.gradients.atoms),atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(saved['bond_logits'],np.asarray(initial.bonds-.3*e.gradients.bonds),atol=1e-6,rtol=1e-6)


def test_equal_noop_is_logged_without_a_search_fallback(tmp_path, monkeypatch):
    # Unit-test bookkeeping for a constant loss only. This fixture intentionally
    # cannot satisfy a nonzero native derivative gate. Production has no bypass.
    from mosaic.small_molecule.chemistry import trajectory
    def constant_fixture_only(loss, evaluation, *args, **kwargs):
        assert np.all(np.asarray(evaluation.gradients.atoms)==0)
        assert np.all(np.asarray(evaluation.gradients.bonds)==0)
        return []
    monkeypatch.setattr(trajectory,'check_combined_directions',constant_fixture_only)
    _,_,_,r=run(tmp_path,lambda graph,*,key:(jnp.sum(graph.X)*0,{}))
    assert r['counts']=={'applied':1,'strict_improving':0,'forced_fallback':0,'no_op':1}
    assert r['updates'][0]['accepted_step_size']==.3


@pytest.mark.parametrize('policy,trials',[('fixed_step',2),('unknown',1)])
def test_fixed_step_never_silently_enables_backtracking(tmp_path,policy,trials):
    with pytest.raises(ValueError,match='Fixed-step'):
        run(tmp_path,lambda graph,*,key:(jnp.sum(graph.X),{}),policy=policy,trials=trials)


def test_nonfinite_values_remain_a_hard_failure(tmp_path):
    with pytest.raises((ValueError,AssertionError,FloatingPointError)):
        run(tmp_path,lambda graph,*,key:(jnp.sum(graph.X)*jnp.nan,{}))
