"""Allocated scope/rate/readiness fixtures for the explicitly boosted campaign."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import campaign25_contract as contract


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('LR-boost contract fixtures require allocated DTU HPC')


def config(arm):
    return json.loads((ROOT / f'examples/configs/brd4_jq1_25_fixed_lrboost_{arm}_10000.json').read_text())


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return path


def row(rate, before=1.0, after=0.9, *, strict=True, noop=False, weight=0.0):
    return {'step': 0, 'selection_policy': 'fixed_step', 'selected_trial': 0,
            'accepted_step_size': rate, 'fallback_used': False, 'no_op': noop,
            'fixed_before': before, 'fixed_after': after, 'fixed_increased': after > before,
            'repair_weight': weight, 'attempts': [{'trial': 0, 'rate': rate, 'strict': strict,
                'autodiff_surrogate': after, 'independent_primal_surrogate': after,
                'repair_weight': weight, 'components': {'native_loss': after}}]}


def pilot(rate, weight=0.0):
    return {'status': 'passed', 'acceptance_policy': 'fixed_step',
            'updates': [row(rate, weight=weight) for _ in range(10)]}


@pytest.mark.parametrize('arm,rate', [('protenix', 3.0), ('boltz2', 0.05), ('boltz2_chemistry', 0.05)])
def test_new_rate_family_changes_only_explicit_rate_version_and_label(arm, rate):
    new = config(arm)
    old_path = ROOT / f'examples/configs/brd4_jq1_25_fixed_{arm}_10000.json'
    old = json.loads(old_path.read_text())
    assert set(new) == set(old)
    assert {k for k in new if new[k] != old[k]} == {'version', 'initial_step_size', 'label'}
    assert new['version'] == contract.FIXED25_LRBOOST_VERSION
    assert contract.config25(new) is new
    assert contract.update_parameters(new) == {'step_size': rate, 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'}
    assert new['seed'] == 17 and new['n_atoms'] == 25 and new['steps'] == 10000
    assert new['pocket'] == old['pocket'] == contract.POCKET_POLICY
    assert new['initialization'] == {'kind': 'chain_uniform', 'mixing': 0.5}
    assert new['loss'] == old['loss'] and new['search'] == old['search'] and new['progress_search'] == old['progress_search']


@pytest.mark.parametrize('arm', contract.ARMS)
def test_historical_fixed_family_retains_unchanged_rates_and_policy(arm):
    old = json.loads((ROOT / f'examples/configs/brd4_jq1_25_fixed_{arm}_10000.json').read_text())
    rate = 0.3 if arm == 'protenix' else 0.01
    assert old['version'] == contract.FIXED25_VERSION
    assert contract.update_parameters(old) == {'step_size': rate, 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'}
    minimal = {'acceptance_policy': 'fixed_step', 'updates': [{'attempts': [{'trial': 0, 'rate': rate}],
               'selected_trial': 0, 'accepted_step_size': rate, 'fallback_used': False}]}
    assert contract.verify_attempt_rates(minimal, old)['step_size'] == rate


@pytest.mark.parametrize('arm', contract.ARMS)
def test_original_backtracking_configs_keep_original_parameters(arm):
    old = json.loads((ROOT / f'examples/configs/brd4_jq1_25_{arm}_10000.json').read_text())
    assert contract.update_parameters(old) == {'step_size': 0.03, 'max_backtracks': 10}


@pytest.mark.parametrize('damage', ['rate', 'old_pro_rate', 'old_bol_rate', 'backtrack', 'policy', 'seed',
                                    'steps', 'size', 'mixing', 'hold', 'endpoint', 'diffusion', 'loss', 'search',
                                    'progress', 'fixture', 'temperature', 'unknown_option', 'bool_dimension'])
def test_high_rate_scope_rejects_other_scientific_changes(damage):
    value = config('boltz2_chemistry')
    if damage == 'rate': value['initial_step_size'] = 0.025
    elif damage == 'old_pro_rate': value['initial_step_size'] = 0.3
    elif damage == 'old_bol_rate': value['initial_step_size'] = 0.01
    elif damage == 'backtrack': value['max_backtracks'] = 10
    elif damage == 'policy': value['acceptance_policy'] = 'best_primal'
    elif damage == 'seed': value['seed'] = 18
    elif damage == 'steps': value['steps'] = 5000
    elif damage == 'size': value['n_atoms'] = 26
    elif damage == 'mixing': value['initialization']['mixing'] = 0.4
    elif damage == 'hold': value['repair_schedule']['hold_updates'] = 4999
    elif damage == 'endpoint': value['repair_schedule']['final'] = 5.0
    elif damage == 'diffusion': value['pocket']['diffusion_gradient'] = True
    elif damage == 'loss': value['loss']['extra_valid_weight'] = 0.5
    elif damage == 'search': value['search']['max_edits'] = 4
    elif damage == 'progress': value['progress_search']['allow_atom_edits'] = True
    elif damage == 'fixture': value['fixture'] = 'different.json'
    elif damage == 'temperature': value['temperature'] = 2.0
    elif damage == 'unknown_option': value['gradient_normalization'] = True
    elif damage == 'bool_dimension': value['max_backtracks'] = True
    with pytest.raises(ValueError):
        contract.config25(value)


@pytest.mark.parametrize('arm', contract.ARMS)
@pytest.mark.parametrize('before,after,strict,noop', [(1.0, 0.9, True, False), (1.0, 1.1, False, False), (1.0, 1.0, False, True)])
def test_only_candidate_is_authenticated_even_when_uphill_equal_or_noop(arm, before, after, strict, noop):
    value = config(arm)
    result = {'acceptance_policy': 'fixed_step', 'updates': [row(value['initial_step_size'], before, after, strict=strict, noop=noop)]}
    assert contract.verify_attempt_rates(result, value)['step_size'] == value['initial_step_size']


@pytest.mark.parametrize('damage', ['halved', 'backtrack', 'selected', 'policy', 'fallback', 'trial_bool',
                                    'selected_bool', 'infinite', 'candidate', 'coefficient', 'increased'])
def test_declared_high_rate_cannot_be_halved_or_relabelled(damage):
    value = config('protenix')
    update = row(3.0)
    if damage == 'halved': update['attempts'][0]['rate'] = 1.5
    elif damage == 'backtrack': update['attempts'].append(deepcopy(update['attempts'][0]))
    elif damage == 'selected': update['selected_trial'] = 1
    elif damage == 'policy': update['selection_policy'] = 'best_primal'
    elif damage == 'fallback': update['fallback_used'] = True
    elif damage == 'trial_bool': update['attempts'][0]['trial'] = False
    elif damage == 'selected_bool': update['selected_trial'] = False
    elif damage == 'infinite': update['attempts'][0]['autodiff_surrogate'] = float('inf')
    elif damage == 'candidate': update['fixed_after'] = 0.8
    elif damage == 'coefficient': update['attempts'][0]['repair_weight'] = 0.001
    elif damage == 'increased': update['fixed_increased'] = True
    with pytest.raises(ValueError):
        contract.verify_attempt_rates({'acceptance_policy': 'fixed_step', 'updates': [update]}, value)


@pytest.mark.parametrize('fixed_rate,expected,backtracks', [(None, 0.03, 10), (0.3, 0.3, 1), (3.0, 3.0, 1)])
def test_protenix_raw_probe_routes_actual_declared_rate_with_original_gradient_gates(monkeypatch, tmp_path, fixed_rate, expected, backtracks):
    from mosaic.small_molecule import gradient_checks
    calls = []
    monkeypatch.setattr(gradient_checks, 'check_graph_gradients', lambda *a, **kw: calls.append(kw) or {'status': 'passed'})
    spec = importlib.util.spec_from_file_location('_boost_driver_fixture', ROOT / 'examples/brd4_25_atoms.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    result = driver.check_protenix_raw_gradients('loss', 'graph', key='native-key', directory=tmp_path, fixed_rate=fixed_rate)
    assert result['status'] == 'passed'
    assert calls == [{'key': 'native-key', 'output_directory': tmp_path, 'steps': 1, 'initial_step_size': expected,
                      'max_backtracks': backtracks, 'primal_every_step': True, 'artifact_policy': 'compact',
                      'acceptance_policy': 'best_primal'}]


@pytest.fixture(params=contract.ARMS)
def readiness(tmp_path, request):
    arm = request.param
    value = config(arm)
    config_path = put(tmp_path / 'config.json', value)
    root = tmp_path / 'study'
    pocket = put(root / 'pocket_definition.json', {'saved': 'pocket'})
    summary = {'optimization_parameters': contract.update_parameters(value)}
    if arm == 'protenix':
        summary['baseline_initialization_match'] = {'status': 'passed', 'job': '29770179',
            'sha256': '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf'}
        summary['native_gradient_checks'] = {'status': 'passed', 'requested_updates': 1, 'primal_every_step': True,
            'acceptance_policy': 'best_primal', 'updates': [{'selected_trial': 0, 'accepted_step_size': 3.0,
                'attempts': [{'step_size': 3.0}]}]}
    paths = [pocket, put(root / 'summary.json', summary), put(root / 'early_pilot/report.json', pilot(value['initial_step_size']))]
    if arm == 'boltz2_chemistry':
        paths.append(put(root / 'high_pilot/report.json', pilot(value['initial_step_size'], 50.0)))
    assessment = {'status': 'passed', 'source_manifest_sha256': 'a' * 64, 'config_sha256': contract.sha(config_path),
                  'arm': arm, 'n_atoms': 25, 'model_digest': contract.MODEL_DIGESTS['protenix' if arm == 'protenix' else 'boltz2'],
                  'pocket_definition_sha256': contract.sha(pocket), 'pocket_validation_status': 'passed', 'diffusion_gradient': False,
                  'proofs': [{'path': str(p), 'sha256': contract.sha(p)} for p in paths]}
    path = put(root / 'assessment.json', assessment)
    return value, path, config_path, assessment, summary


def test_readiness_authenticates_executed_new_rate_and_both_chemistry_pilots(readiness):
    _, path, config_path, assessment, _ = readiness
    assert contract.verify_readiness(path, contract.sha(path), config_path, 'a' * 64) == assessment


@pytest.mark.parametrize('damage', ['proof', 'optimization', 'early_rate', 'early_count', 'high_rate', 'high_count', 'baseline', 'raw_rate', 'raw_trials'])
def test_readiness_cannot_reuse_old_rate_or_missing_new_rate_pilots(readiness, damage):
    value, path, config_path, assessment, summary = readiness
    root = path.parent
    if damage in ('high_rate', 'high_count') and value['arm'] != 'boltz2_chemistry':
        damage = 'early_rate'
    if damage in ('baseline', 'raw_rate', 'raw_trials') and value['arm'] != 'protenix':
        damage = 'early_count'
    if damage == 'proof': (root / 'pocket_definition.json').write_text('{}')
    elif damage == 'optimization':
        summary['optimization_parameters']['step_size'] = 0.01
        put(root / 'summary.json', summary)
    elif damage in ('early_rate', 'early_count', 'high_rate', 'high_count'):
        target = root / ('high_pilot/report.json' if damage.startswith('high') else 'early_pilot/report.json')
        result = json.loads(target.read_text())
        if damage.endswith('rate'): result['updates'][0]['attempts'][0]['rate'] = 0.01
        else: result['updates'].pop()
        put(target, result)
    elif damage == 'baseline':
        summary['baseline_initialization_match']['status'] = 'failed'; put(root / 'summary.json', summary)
    elif damage == 'raw_rate':
        summary['native_gradient_checks']['updates'][0]['accepted_step_size'] = 0.3; put(root / 'summary.json', summary)
    elif damage == 'raw_trials':
        summary['native_gradient_checks']['updates'][0]['attempts'].append({'step_size': 1.5}); put(root / 'summary.json', summary)
    # Reseal metadata hashes so this exercises semantic gates rather than only a
    # modified-byte rejection. The intentionally changed pocket keeps its old pin.
    for proof in assessment['proofs']:
        if damage != 'proof': proof['sha256'] = contract.sha(proof['path'])
    put(path, assessment)
    with pytest.raises(ValueError):
        contract.verify_readiness(path, contract.sha(path), config_path, 'a' * 64)
