"""Allocated metadata contracts for honest high-rate saved-run presentation."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import copy
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(_layout_path(_repository_root(__file__) / 'scripts')))
import jq1_lrboost_report_contract as contract


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Higher-rate report fixtures require allocated DTU HPC')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')
    return path


def row(step=0, *, rate=0.05, before=1.0, after=0.9, strict=True, noop=False):
    return {'step': step, 'selection_policy': 'fixed_step', 'selected_trial': 0,
            'accepted_step_size': rate, 'fallback_used': False, 'no_op': noop,
            'fixed_before': before, 'fixed_after': after, 'fixed_increased': after > before,
            'repair_weight': 0.0, 'attempts': [{'trial': 0, 'rate': rate, 'strict': strict,
                'independent_primal_surrogate': after, 'autodiff_surrogate': after,
                'repair_weight': 0.0, 'components': {'native_loss': after}}]}


def lsf(job, *, passed=False, code=1):
    return ('Subject: Job ' + job + ': <native> in cluster <dcc> ' + ('Done' if passed else 'Exited')
            + '\n\n' + ('Successfully completed.\n' if passed else 'Exited with exit code ' + str(code) + '.\n'))


@pytest.mark.parametrize('rate', [3.0, 0.05])
def test_boosted_counts_preserve_uphill_equality_noop_and_nonstrict_decrease(rate):
    rows = [row(0, rate=rate), row(1, rate=rate, after=1.2, strict=False),
            row(2, rate=rate, after=1.0, strict=False, noop=True),
            row(3, rate=rate, after=0.99, strict=False)]
    assert contract.fixed_counts(rows, rate) == {'applied': 4, 'strict_improving': 1,
        'forced_fallback': 0, 'no_op': 1, 'scalar_increased': 1, 'scalar_equal': 1, 'scalar_decreased': 2}


@pytest.mark.parametrize('damage', ['old_rate', 'halved', 'backtrack', 'selected', 'trial', 'policy',
                                    'fallback', 'nonfinite', 'component', 'candidate', 'strict', 'increase', 'noop', 'bool_step'])
def test_fixed_count_policy_and_finite_candidate_are_mandatory(damage):
    value = row(rate=3.0)
    if damage == 'old_rate': value['accepted_step_size'] = 0.3
    elif damage == 'halved': value['attempts'][0]['rate'] = 1.5
    elif damage == 'backtrack': value['attempts'].append(copy.deepcopy(value['attempts'][0]))
    elif damage == 'selected': value['selected_trial'] = 1
    elif damage == 'trial': value['attempts'][0]['trial'] = True
    elif damage == 'policy': value['selection_policy'] = 'best_primal'
    elif damage == 'fallback': value['fallback_used'] = True
    elif damage == 'nonfinite': value['fixed_after'] = float('inf')
    elif damage == 'component': value['attempts'][0]['components']['native_loss'] = float('nan')
    elif damage == 'candidate': value['attempts'][0]['independent_primal_surrogate'] = 0.8
    elif damage == 'strict': value['attempts'][0]['strict'] = 1
    elif damage == 'increase': value['fixed_increased'] = True
    elif damage == 'noop': value['no_op'] = 1
    elif damage == 'bool_step': value['step'] = False
    with pytest.raises(ValueError):
        contract.fixed_counts([value], 3.0)


def test_three_new_jobs_and_independent_output_roots_do_not_reuse_old_protenix():
    assert [pin['job'] for pin in contract.ARMS.values()] == ['29779177', '29779178', '29779179']
    assert [pin['rate'] for pin in contract.ARMS.values()] == [3.0, 0.05, 0.05]
    for arm in contract.ARMS:
        assert contract.source_path(arm) == contract.SCIENCE
        assert contract.ready_path(arm).is_relative_to(contract.ROOT)
        assert 'fixed_lrboost_' + arm in contract.config_path(arm).name
    assert contract.native_parent('protenix').is_relative_to(contract.ROOT)
    assert contract.native_parent('boltz2').is_relative_to(contract.ROOT)
    assert contract.native_parent('boltz2_chemistry').is_relative_to(contract.HOME_ROOT)


def write_prefix(root, states, updates, *, tail=b''):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'states.jsonl').write_bytes(b''.join(json.dumps(v).encode() + b'\n' for v in states) + tail)
    (root / 'updates.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in updates))


def test_prefix_includes_only_one_common_case_and_committed_boundary(tmp_path):
    states = [{'step': i} for i in range(4)]
    updates = [{'step': i} for i in range(3)]
    write_prefix(tmp_path, states, updates, tail=b'{"step":4')
    case = {'states': states[:3], 'updates': updates[:2]}
    captured, proofs = contract.saved_prefix(tmp_path, case)
    assert captured == case and proofs[0]['uncommitted_tail_bytes'] == len(b'{"step":4')
    assert proofs[0]['bytes'] < (tmp_path / 'states.jsonl').stat().st_size
    assert proofs[0]['included_rows'] == 3 and proofs[0]['excluded_committed_rows'] == 1
    assert proofs[0]['bytes'] == len(b''.join(json.dumps(v).encode() + b'\n' for v in states[:3]))


@pytest.mark.parametrize('damage', ['corrupt_complete', 'gap', 'bool_step', 'case_mismatch'])
def test_prefix_never_skips_corruption_or_disagrees_with_original_case(tmp_path, damage):
    states = [{'step': 0}, {'step': 1}]
    updates = [{'step': 0}]
    write_prefix(tmp_path, states, updates)
    case = {'states': copy.deepcopy(states), 'updates': copy.deepcopy(updates)}
    if damage == 'corrupt_complete': (tmp_path / 'states.jsonl').write_bytes(b'{bad}\n')
    elif damage == 'gap': write_prefix(tmp_path, [{'step': 0}, {'step': 2}], updates)
    elif damage == 'bool_step': write_prefix(tmp_path, [{'step': False}], [])
    elif damage == 'case_mismatch': case['states'][1]['different'] = True
    with pytest.raises((ValueError, json.JSONDecodeError)):
        contract.saved_prefix(tmp_path, case)


@pytest.fixture(params=['protenix', 'boltz2'])
def failed(tmp_path, monkeypatch, request):
    arm = request.param
    parent = tmp_path / ('run-' + arm)
    study = parent / 'payload/study'
    old = contract.ARMS[arm]
    config = {'arm': arm, 'label': 'original-label', 'initialization': {'kind': 'chain_uniform', 'mixing': 0.5}}
    case = {'status': 'passed', 'requested_updates': 2,
            'states': [{'step': i, 'repair_weight': 0.0} for i in range(3)],
            'updates': [row(i, rate=old['rate']) for i in range(2)],
            'counts': {'applied': 2, 'strict_improving': 2, 'forced_fallback': 0, 'no_op': 0}}
    summary = {'status': 'failed', 'job_id': old['job'], 'config': config,
               'error': old['failed_terminal']['error'],
               'cases': [{**case, 'label': config['label'], 'initialization': config['initialization']}]}
    summary_path = put(study / 'summary.json', summary)
    case_path = put(study / 'case-0/report.json', case)
    terminal = put(study / 'terminal_gradient_checks/report.json', {'status': 'failed', 'error': summary['error']})
    stdout = parent / ('stdout.' + old['job'] + '.log'); stdout.write_text(lsf(old['job']))
    write_prefix(study / 'case-0', case['states'], case['updates'])
    bound = {'summary': contract.sha(summary_path), 'case': contract.sha(case_path), 'terminal': contract.sha(terminal),
             'stdout': contract.sha(stdout), 'error': summary['error']}
    monkeypatch.setitem(contract.ARMS, arm, {**old, 'output_parent': str(parent), 'failed_terminal': bound})
    return arm, study, summary, case, stdout, bound


def test_failed_overall_remains_distinct_from_passed_optimizer_case(failed):
    arm, _, _, _, _, _ = failed
    outcome = contract.saved_outcome(arm)
    assert outcome['overall_status'] == outcome['terminal_status'] == 'failed'
    assert outcome['readiness_status'] == 'passed' and outcome['all_terminal'] is True
    assert outcome['scientific_passed'] is False and outcome['native_status'] == 'FAILED_PERSISTED'
    assert outcome['budget_status'] == 'partial' and outcome['failed_run_applicability'] == contract.FAILED_APPLICABILITY
    assert 'LSF' in outcome['scheduler_scope'] and 'Live' in outcome['scheduler_scope']


def test_failed_case_applicability_is_explicit_and_never_strict_success(failed):
    _, study, summary, case, _, _ = failed
    with pytest.raises(ValueError, match='overall result'):
        contract.verify_case(study, summary, steps=2)
    assert contract.verify_case(study, summary, steps=2, allow_failed=True) == case


def test_known_failure_cannot_be_relabelled_passed_even_with_resealed_metadata(failed):
    arm, study, summary, _, stdout, bound = failed
    summary['status'] = 'passed'
    put(study / 'summary.json', summary)
    bound['summary'] = contract.sha(study / 'summary.json')
    put(study / 'terminal_gradient_checks/report.json', {'status': 'passed'})
    bound['terminal'] = contract.sha(study / 'terminal_gradient_checks/report.json')
    stdout.write_text(lsf(contract.ARMS[arm]['job'], passed=True))
    bound['stdout'] = contract.sha(stdout)
    with pytest.raises(ValueError):
        contract.saved_outcome(arm)


def test_strict_native_authentication_rejects_known_failed_run_before_any_waiver(failed, monkeypatch):
    arm, _, _, _, _, _ = failed
    monkeypatch.setattr(contract, 'authenticate_inputs', lambda selected: {})
    with pytest.raises(ValueError, match='cannot certify strict'):
        contract.authenticate_study(arm)


@pytest.mark.parametrize('damage', ['extra_complete', 'unfinished_tail', 'wrong_wrapper'])
def test_full_failed_optimizer_case_still_requires_exact_terminal_jsonl_and_wrapper(failed, damage):
    _, study, summary, _, _, _ = failed
    path = study / 'case-0/states.jsonl'
    if damage == 'extra_complete':
        with path.open('a') as stream: stream.write('{"step":3,"repair_weight":0.0}\n')
    elif damage == 'unfinished_tail':
        with path.open('a') as stream: stream.write('{"step":3')
    else:
        summary['cases'][0]['label'] = 'changed'
    with pytest.raises(ValueError):
        contract.verify_case(study, summary, steps=2, allow_failed=True)


@pytest.mark.parametrize('damage', ['summary_hash', 'case_hash', 'terminal_hash', 'lsf_hash', 'wrong_job', 'success_lsf', 'error', 'terminal_pass'])
def test_original_failure_hashes_job_lsf_and_error_are_mandatory(failed, damage):
    arm, study, summary, _, stdout, bound = failed
    if damage == 'summary_hash': (study / 'summary.json').write_text('{}')
    elif damage == 'case_hash': (study / 'case-0/report.json').write_text('{}')
    elif damage == 'terminal_hash': (study / 'terminal_gradient_checks/report.json').write_text('{}')
    elif damage == 'lsf_hash': stdout.write_text('changed')
    elif damage == 'wrong_job':
        stdout.write_text(lsf('29777709')); bound['stdout'] = contract.sha(stdout)
    elif damage == 'success_lsf':
        stdout.write_text(lsf(contract.ARMS[arm]['job'], passed=True)); bound['stdout'] = contract.sha(stdout)
    elif damage == 'error':
        summary['error'] = 'different failure'; put(study / 'summary.json', summary); bound['summary'] = contract.sha(study / 'summary.json')
    elif damage == 'terminal_pass':
        put(study / 'terminal_gradient_checks/report.json', {'status': 'passed'}); bound['terminal'] = contract.sha(study / 'terminal_gradient_checks/report.json')
    with pytest.raises(ValueError):
        contract.failure_evidence(arm)


def test_ageout_returns_failed_persisted_and_never_invents_exit_or_done(failed, monkeypatch):
    arm, _, _, _, _, _ = failed
    calls = []
    monkeypatch.setattr(contract, 'authenticate_study', lambda selected, **kw: calls.append((selected, kw)))
    assert contract.status_or_completion('Job is not found; bhist denied', arm) == 'FAILED_PERSISTED'
    assert calls == [(arm, {'allow_failed': True})]


@pytest.mark.parametrize('status', ['RUN', 'PEND', 'DONE', 'EXIT'])
def test_actual_live_status_retains_exact_current_job_and_queue(status):
    assert contract.status_or_completion('Job <29779179>, Status <' + status + '>, Queue <gpul40s>', 'boltz2_chemistry') == status


@pytest.mark.parametrize('raw', ['Job <29778204>, Status <RUN>, Queue <gpul40s>',
                                'Job <29779179>, Status <DONE>, Queue <hpc>',
                                'Job <29779179> Job <29779179> Status <DONE> Queue <gpul40s>'])
def test_old_or_ambiguous_live_records_cannot_describe_current_chemistry(raw):
    with pytest.raises(ValueError):
        contract.status_or_completion(raw, 'boltz2_chemistry')


def test_missing_current_chemistry_history_does_not_become_a_completion(tmp_path, monkeypatch):
    monkeypatch.setitem(contract.ARMS, 'boltz2_chemistry',
                        {**contract.ARMS['boltz2_chemistry'], 'output_parent': str(tmp_path / 'missing')})
    with pytest.raises(ValueError, match='authentic persisted collector'):
        contract.status_or_completion('Job is not found', 'boltz2_chemistry')


@pytest.fixture(params=['passed', 'failed'])
def generic_terminal(tmp_path, monkeypatch, request):
    """Isolate external input identities, retaining real budget/JSONL/LSF checks."""
    arm, status = 'boltz2_chemistry', request.param
    parent = tmp_path / 'original-chemistry'
    study = parent / 'payload/study'
    monkeypatch.setitem(contract.ARMS, arm, {**contract.ARMS[arm], 'output_parent': str(parent)})
    monkeypatch.setattr(contract, 'authenticate_inputs', lambda selected: {'original_inputs': selected})
    monkeypatch.setattr(contract, '_summary_identity', lambda *args: None)
    monkeypatch.setattr(contract, '_case_identity', lambda *args: {})
    monkeypatch.setattr(contract, 'verify_coefficients', lambda *args: None)
    count = 10000 if status == 'passed' else 1
    case = {'status': status, 'requested_updates': 10000,
            'states': [{'step': i, 'repair_weight': 50.0 if i == 10000 else 0.0} for i in range(count + 1)],
            'updates': [row(i) for i in range(count)],
            'counts': {'applied': count, 'strict_improving': count, 'forced_fallback': 0, 'no_op': 0}}
    config = {'arm': arm, 'label': 'original-chemistry', 'initialization': {'key': 17}}
    summary = {'status': status, 'job_id': contract.ARMS[arm]['job'], 'config': config,
               'cases': [{**case, 'label': config['label'], 'initialization': config['initialization']}]}
    if status == 'passed':
        terminal = {'status': 'passed', 'original_terminal': True}
        summary['terminal_gradient_checks'] = terminal
        put(study / 'terminal_gradient_checks/report.json', terminal)
    else:
        summary['error'] = 'Original partial scientific failure'
    put(study / 'summary.json', summary)
    put(study / 'case-0/report.json', case)
    write_prefix(study / 'case-0', case['states'], case['updates'])
    stdout = parent / ('stdout.' + contract.ARMS[arm]['job'] + '.log')
    stdout.write_text(lsf(contract.ARMS[arm]['job'], passed=status == 'passed'))
    return arm, status, study, summary, case, stdout


def test_generic_terminal_bootstraps_without_a_collector_or_fabricated_scheduler(generic_terminal, tmp_path):
    arm, status, study, _, case, stdout = generic_terminal
    completions = tmp_path / 'analysis'
    # The collector's own assessment is deliberately absent.
    evidence = contract.terminal_evidence(arm, completions)
    expected = 'SUCCESS_PERSISTED' if status == 'passed' else 'FAILED_PERSISTED'
    assert evidence['native_status'] == expected
    assert contract.status_or_completion('Job is not found; bhist denied', arm, completions) == expected
    assert 'Live history unavailable' in evidence['scheduler_scope']
    assert not completions.exists()
    assert evidence['summary'] == contract.hashed(study / 'summary.json')
    assert evidence['stdout'] == contract.hashed(stdout)
    assert all(contract.sha(p['path']) == p['sha256'] for p in evidence['proofs'])
    assert contract.hashed(study / 'case-0/updates.jsonl') in evidence['proofs']
    outcome = contract.saved_outcome(arm, expected)
    assert outcome['all_terminal'] and outcome['scientific_passed'] is (status == 'passed')
    assert outcome['terminal_status'] == ('passed' if status == 'passed' else 'not_reached')
    assert outcome['updates'] == len(case['updates'])


@pytest.mark.parametrize('damage', ['wrong_job_lsf', 'opposite_lsf', 'case_jsonl_mismatch', 'missing_summary', 'input_authentication'])
def test_generic_terminal_fallback_requires_original_authenticated_outcome(generic_terminal, tmp_path, monkeypatch, damage):
    arm, status, study, _, _, stdout = generic_terminal
    if damage == 'wrong_job_lsf': stdout.write_text(lsf('29778204', passed=status == 'passed'))
    elif damage == 'opposite_lsf': stdout.write_text(lsf(contract.ARMS[arm]['job'], passed=status != 'passed'))
    elif damage == 'case_jsonl_mismatch':
        path = study / 'case-0/updates.jsonl'
        path.write_text(path.read_text().replace('"fixed_before": 1.0', '"fixed_before": 2.0', 1))
    elif damage == 'missing_summary': (study / 'summary.json').unlink()
    else:
        def reject(selected): raise ValueError('Original immutable input authentication failed')
        monkeypatch.setattr(contract, 'authenticate_inputs', reject)
    with pytest.raises((ValueError, FileNotFoundError)):
        contract.terminal_evidence(arm, tmp_path / 'no-collector')


@pytest.mark.parametrize('damage', ['short_budget', 'missing_terminal', 'failed_terminal', 'summary_terminal_differs', 'extra_jsonl'])
@pytest.mark.parametrize('generic_terminal', ['passed'], indirect=True)
def test_persisted_success_still_requires_full_budget_and_original_passed_terminal(generic_terminal, damage):
    arm, _, study, summary, case, _ = generic_terminal
    if damage == 'short_budget':
        case['states'].pop(); case['updates'].pop()
        put(study / 'case-0/report.json', case)
        write_prefix(study / 'case-0', case['states'], case['updates'])
    elif damage == 'missing_terminal': (study / 'terminal_gradient_checks/report.json').unlink()
    elif damage == 'failed_terminal': put(study / 'terminal_gradient_checks/report.json', {'status': 'failed'})
    elif damage == 'summary_terminal_differs':
        summary['terminal_gradient_checks'] = {'status': 'passed', 'different_gate': True}
        put(study / 'summary.json', summary)
    else:
        with (study / 'case-0/updates.jsonl').open('a') as stream: stream.write('{"step":10000}\n')
    with pytest.raises((ValueError, FileNotFoundError)):
        contract.terminal_evidence(arm)


def test_running_chemistry_is_not_a_terminal_or_scientific_pass(tmp_path, monkeypatch):
    parent = tmp_path / 'run-chem'
    monkeypatch.setitem(contract.ARMS, 'boltz2_chemistry', {**contract.ARMS['boltz2_chemistry'], 'output_parent': str(parent)})
    put(parent / 'payload/study/case-0/report.json', {'states': [{'step': 0}, {'step': 1}], 'updates': [{'step': 0}]})
    outcome = contract.saved_outcome('boltz2_chemistry', 'RUN')
    assert outcome['native_status'] == 'RUN' and outcome['overall_status'] == 'running'
    assert outcome['terminal_status'] == 'pending' and outcome['all_terminal'] is False
    assert outcome['scientific_passed'] is False and outcome['budget_status'] == 'partial'


def test_queued_arm_has_no_invented_initial_state(tmp_path, monkeypatch):
    monkeypatch.setitem(contract.ARMS, 'boltz2_chemistry', {**contract.ARMS['boltz2_chemistry'], 'output_parent': str(tmp_path / 'queued')})
    outcome = contract.saved_outcome('boltz2_chemistry', 'PEND')
    assert outcome['states'] == outcome['updates'] == 0 and outcome['budget_status'] == 'not_started'
    assert outcome['overall_status'] == 'queued' and not outcome['all_terminal']


def test_partial_chemistry_failure_is_explicit_without_missing_terminal_pass(tmp_path, monkeypatch):
    arm = 'boltz2_chemistry'
    parent = tmp_path / 'run-chem'
    study = parent / 'payload/study'
    monkeypatch.setitem(contract.ARMS, arm, {**contract.ARMS[arm], 'output_parent': str(parent)})
    config = {'arm': arm, 'label': 'chemistry', 'initialization': {'kind': 'chain_uniform', 'mixing': 0.5}}
    case = {'status': 'failed', 'states': [{'step': 0, 'repair_weight': 0.001}], 'updates': [], 'requested_updates': 10000}
    summary = {'job_id': '29779179', 'status': 'failed', 'error': 'Native execution failed', 'config': config}
    put(study / 'summary.json', summary); put(study / 'case-0/report.json', case)
    write_prefix(study / 'case-0', case['states'], case['updates'])
    (parent / 'stdout.29779179.log').write_text(lsf('29779179'))
    outcome = contract.saved_outcome(arm, 'EXIT')
    assert outcome['overall_status'] == 'failed' and outcome['terminal_status'] == 'not_reached'
    assert not outcome['scientific_passed'] and outcome['all_terminal']
    assert contract.verify_case(study, summary, allow_failed=True)['states'] == case['states']
