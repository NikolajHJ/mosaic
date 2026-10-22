"""Allocated scope/rate/dispatch contracts for the independent Protenix comparison."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from campaign25_contract import config25, update_parameters, verify_attempt_rates
import protenix_lr03_ops as ops


ROOT = _repository_root(__file__)


def config():
    return json.loads((ROOT / ops.CONFIG).read_text())


def test_only_lr_version_and_label_differ_from_original_scientific_config():
    old = json.loads((_layout_path(ROOT / 'examples/configs/brd4_jq1_25_protenix_10000.json')).read_text())
    new = config()
    assert {k for k in old if old[k] != new[k]} == {'version', 'label', 'initial_step_size'}
    assert config25(new) is new
    assert update_parameters(new) == {'step_size': 0.3, 'max_backtracks': 10}
    assert update_parameters(old) == {'step_size': 0.03, 'max_backtracks': 10}


@pytest.mark.parametrize('key,value', [('initial_step_size', .03), ('initial_step_size', .1), ('arm', 'boltz2'),
                                      ('arm', 'boltz2_chemistry'), ('steps', 9999), ('seed', 18),
                                      ('chemistry_enabled', True), ('max_backtracks', 16), ('element_js_weight', .1)])
def test_new_authorization_does_not_enable_other_rates_arms_or_protocols(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError):
        config25(c)


def result():
    return {'updates': [{'attempts': [{'trial': i, 'rate': .3 * .5**i} for i in range(3)],
                         'selected_trial': 2, 'accepted_step_size': .075,
                         'fallback_used': True, 'no_op': True}]}


def test_executed_rates_keep_halving_and_honest_fallback_noop_semantics():
    assert verify_attempt_rates(result(), config())['step_size'] == .3


@pytest.mark.parametrize('mutation', ['nominal', 'selected', 'missing', 'trial_index'])
def test_declared_lr_cannot_hide_a_hardcoded_old_rate(mutation):
    r = result(); u = r['updates'][0]
    if mutation == 'nominal': u['attempts'][0]['rate'] = .03
    if mutation == 'selected': u['accepted_step_size'] = .03
    if mutation == 'missing': u['attempts'] = []
    if mutation == 'trial_index': u['attempts'][0]['trial'] = 1
    with pytest.raises(ValueError): verify_attempt_rates(r, config())


@pytest.mark.parametrize('phase,gpu,wall', [('validate', True, '02:00'), ('run', True, '23:00'), ('after-ready', False, '00:20')])
def test_actual_job_vectors_are_protenix_only_with_no_live_boltz_dependency(tmp_path, monkeypatch, phase, gpu, wall):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='Job <29700001> is submitted' if command[0] == 'bsub' else 'PEND', stderr='')
    monkeypatch.setattr(ops.subprocess, 'run', run)
    args = {'producer': '29700002'} if phase == 'after-ready' else {}
    receipt = ops.submit(tmp_path / 'source', tmp_path, 'a' * 64, phase, **args)
    command = calls[0]
    assert receipt['arm'] == 'protenix' and receipt['job_id'] == '29700001'
    assert command[command.index('-W') + 1] == wall
    assert command[command.index('-q') + 1] == ('gpul40s' if gpu else 'hpc')
    assert command[command.index('-n') + 1] == ('4' if gpu else '1')
    assert ('-gpu' in command) == gpu
    if gpu: assert command[command.index('-gpu') + 1] == 'num=1:mode=exclusive_process'
    assert ('-w' in command) == (phase == 'after-ready')
    assert not any(job in ' '.join(command) for job in ('29770093', '29770124'))


@pytest.mark.parametrize('phase', ['prepare', 'boltz2', 'cpu', 'unknown'])
def test_no_boltz_or_preparation_submission_route(tmp_path, monkeypatch, phase):
    def forbidden(*a, **k): raise AssertionError('No scheduler call')
    monkeypatch.setattr(ops.subprocess, 'run', forbidden)
    with pytest.raises(ValueError): ops.submit(tmp_path, tmp_path, 'a' * 64, phase)


def test_unknown_submission_persists_receipt_and_cannot_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='Connection closed', stderr=''))
    with pytest.raises(RuntimeError, match='Unknown'): ops.submit(tmp_path, tmp_path, 'a' * 64, 'run')
    assert 'job_id' not in json.loads((tmp_path / 'run-protenix/submission.json').read_text())
    with pytest.raises(FileExistsError): ops.submit(tmp_path, tmp_path, 'a' * 64, 'run')
