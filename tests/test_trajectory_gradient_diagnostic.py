"""Allocated CPU fixtures for version-aware, read-only failed-state replay."""
from copy import deepcopy
import pytest

from latentmol.boltz.failed_gate_attribution import failed_item
from latentmol.boltz.trajectory_gradient_diagnostic import EPSILONS


def fixture():
    secants = {'policy': {'version': 'local_v1'}, 'finite_differences': [{'epsilon': 1e-4}],
               'direction_results': {'random_2': {'passed': False}}, 'direction_selection': {}}
    return {'status': 'failed', 'passed': False, 'native_seed_parity': True,
        'measured': {'loss': 1.}, 'component_gradients': {'dbmol_contact_loss':
            {'status': 'passed', 'passed': False, 'validation_contract': 'branch_local_v2',
             'local_branch_fd': secants}}}


def test_explicit_branch_view_handles_natural_failure_without_rewriting_record():
    gate=fixture(); original=deepcopy(gate)
    key, item=failed_item(gate, policy='branch_local_v2')
    assert key=='dbmol_contact_loss'
    assert item==gate['component_gradients'][key]['local_branch_fd']
    assert gate==original


def test_old_policy_does_not_approve_or_reinterpret_branch_failure():
    with pytest.raises(ValueError): failed_item(fixture())


@pytest.mark.parametrize('change', ('missing_rows','wrong_contract','two_failures','no_parity'))
def test_incomplete_or_ambiguous_historical_failure_is_rejected(change):
    gate=fixture(); item=gate['component_gradients']['dbmol_contact_loss']
    if change=='missing_rows': item['local_branch_fd']['finite_differences']=[]
    if change=='wrong_contract': item['validation_contract']='unknown'
    if change=='two_failures': gate['local_fd_validation']={'passed':False,'status':'failed'}
    if change=='no_parity': gate['native_seed_parity']=False
    with pytest.raises(ValueError): failed_item(gate, policy='branch_local_v2')


def test_diagnostic_grid_declared_and_small_scales_preserved():
    assert EPSILONS==(1e-5,3e-5,1e-4,3e-4,.001,.003)
