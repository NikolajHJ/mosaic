"""Allocated fixed-rate scope and real submission-vector contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from campaign25_contract import ARMS, config25, update_parameters, verify_attempt_rates
import fixed25_ops as ops

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arm,rate', [('protenix', .3), ('boltz2', .01), ('boltz2_chemistry', .01)])
def test_new_scientific_changes_are_explicit_and_rates_cannot_backtrack(arm, rate):
    new = json.loads(ops.config_path(ROOT, arm).read_text())
    old = json.loads((ROOT / f'examples/configs/brd4_jq1_25_{arm}_10000.json').read_text())
    assert {k for k in old if old[k] != new[k]} == {'version', 'label', 'initial_step_size', 'max_backtracks'}
    assert new['acceptance_policy'] == 'fixed_step'
    assert config25(new) is new
    assert update_parameters(new) == {'step_size': rate, 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'}
    result = {'acceptance_policy': 'fixed_step', 'updates': [{'attempts': [{'trial': 0, 'rate': rate}], 'selected_trial': 0,
              'accepted_step_size': rate, 'fallback_used': False}]}
    assert verify_attempt_rates(result, new)['step_size'] == rate
    result['updates'][0]['attempts'][0]['rate'] = rate / 2
    with pytest.raises(ValueError): verify_attempt_rates(result, new)


@pytest.mark.parametrize('field,value', [('max_backtracks', 10), ('acceptance_policy', 'best_primal'),
                                       ('initial_step_size', .03), ('steps', 9999), ('seed', 18)])
def test_fixed_scope_rejects_silent_policy_or_budget_changes(field, value):
    c = json.loads(ops.config_path(ROOT, 'boltz2').read_text()); c[field] = value
    with pytest.raises(ValueError): config25(c)


@pytest.mark.parametrize('arm', ARMS)
def test_each_long_run_is_independent_and_keeps_resources(tmp_path, monkeypatch, arm):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout='Job <29700001> is submitted' if cmd[0] == 'bsub' else 'PEND', stderr='')
    monkeypatch.setattr(ops.subprocess, 'run', run)
    receipt = ops.submit(tmp_path / 'source', tmp_path, 'a'*64, 'run', arm=arm)
    cmd = calls[0]
    assert receipt['arm'] == arm and receipt['job_id'] == '29700001'
    assert cmd[cmd.index('-q')+1] == 'gpul40s' and cmd[cmd.index('-n')+1] == '4'
    assert cmd[cmd.index('-W')+1] == '23:00' and '-w' not in cmd
    assert cmd[cmd.index('-gpu')+1] == 'num=1:mode=exclusive_process'
    assert 'rusage[mem=8GB]' in cmd


def test_dispatch_waits_for_all_actual_native_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'submit', lambda *a, **k: pytest.fail('No early production'))
    ops.dispatch(tmp_path, tmp_path, 'a'*64)
    for arm in ARMS: ops.write(tmp_path / ('decision-'+arm+'.json'), {'status': 'failed'})
    with pytest.raises(ValueError, match='failed'): ops.dispatch(tmp_path, tmp_path, 'a'*64)


def test_all_passed_arms_require_whole_campaign_storage_before_independent_submission(tmp_path, monkeypatch):
    for arm in ARMS:
        ops.write(tmp_path / ('decision-'+arm+'.json'), {'status': 'passed', 'assessment_sha256': 'a'*64, 'projected_bytes': 100})
    monkeypatch.setattr(ops, 'verify_readiness', lambda *a: {})
    observed = []
    monkeypatch.setattr(ops, 'check_storage', lambda path, projection: observed.append(projection) or {'status': 'passed'})
    monkeypatch.setattr(ops, 'submit', lambda *a, **k: {'arm': k['arm'], 'job_id': '29700001'})
    ops.dispatch(tmp_path, tmp_path, 'a'*64)
    assert observed == [300]
    assert [j['arm'] for j in ops.read(tmp_path / 'long-submissions.json')['jobs']] == list(ARMS)
    ops.dispatch(tmp_path, tmp_path, 'a'*64)
    assert observed == [300]
