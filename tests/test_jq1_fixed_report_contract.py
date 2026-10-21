"""Allocated contract fixtures: fixed-step evidence and honest actual outcomes."""

import copy
import json
import math
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import jq1_fixed_report_contract as contract


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Fixed report fixtures require allocated DTU HPC')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')
    return path


def update(step=0, before=1.0, after=0.9, *, strict=True, noop=False, rate=0.01):
    return {'step': step, 'selection_policy': 'fixed_step', 'selected_trial': 0,
            'accepted_step_size': rate, 'fallback_used': False, 'no_op': noop,
            'fixed_before': before, 'fixed_after': after, 'fixed_increased': after > before,
            'repair_weight': 0.0, 'attempts': [{'trial': 0, 'rate': rate, 'strict': strict,
                'independent_primal_surrogate': after, 'autodiff_surrogate': after,
                'repair_weight': 0.0, 'components': {'affinity_loss': after}}]}


def scheduler(job, status='DONE', queue='gpul40s'):
    return 'Job <' + job + '>, Status <' + status + '>, Queue <' + queue + '>\n'


def lsf(job):
    return 'Subject: Job ' + job + ': <fixture> in cluster <dcc> Done\n\nSuccessfully completed.\n'


@pytest.mark.parametrize('rate', [0.3, 0.01])
def test_single_candidate_counts_keep_uphill_equality_noop_and_nonstrict_decrease(rate):
    rows = [update(0, 1.0, 0.9, rate=rate), update(1, 0.9, 1.2, strict=False, rate=rate),
            update(2, 1.2, 1.2, strict=False, noop=True, rate=rate),
            update(3, 1.2, 1.19, strict=False, rate=rate)]
    assert contract.fixed_counts(rows, rate) == {
        'applied': 4, 'strict_improving': 1, 'forced_fallback': 0, 'no_op': 1,
        'scalar_increased': 1, 'scalar_equal': 1, 'scalar_decreased': 2}


def test_empty_arm_has_zero_counts_and_no_invented_update():
    assert all(value == 0 for value in contract.fixed_counts([], 0.01).values())


def test_chemistry_hold_and_endpoint_use_original_one_based_update_schedule():
    ramp = math.exp(math.log(0.001) + (1 / 5000) * (math.log(50.0) - math.log(0.001)))
    states = [{'step': 0, 'repair_weight': 0.001}, {'step': 4999, 'repair_weight': 0.001},
              {'step': 5000, 'repair_weight': ramp}, {'step': 9999, 'repair_weight': 50.0},
              {'step': 10000, 'repair_weight': 50.0}]
    contract.verify_coefficients('boltz2_chemistry', states, states[:-1])
    states[2]['repair_weight'] = 0.001
    with pytest.raises(ValueError, match='chemistry schedule'):
        contract.verify_coefficients('boltz2_chemistry', states, [])


@pytest.mark.parametrize('arm', ['protenix', 'boltz2'])
def test_scorer_only_states_and_updates_cannot_acquire_chemistry(arm):
    with pytest.raises(ValueError):
        contract.verify_coefficients(arm, [{'step': 0, 'repair_weight': 0.001}], [])
    with pytest.raises(ValueError):
        contract.verify_coefficients(arm, [], [{'step': 0, 'repair_weight': 0.001}])


@pytest.mark.parametrize('damage', [
    'halved', 'accepted_halved', 'backtrack', 'selected', 'trial', 'policy', 'fallback',
    'nonfinite_before', 'nonfinite_after', 'nonfinite_autodiff', 'nonfinite_component',
    'candidate_mismatch', 'increased', 'strict_contradiction', 'strict_type', 'noop_type',
    'coefficient', 'missing_component', 'continuity', 'bool_rate'])
def test_wrong_fixed_policy_or_candidate_cannot_enter_a_report(damage):
    row = update()
    if damage == 'halved': row['attempts'][0]['rate'] = 0.005
    elif damage == 'accepted_halved': row['accepted_step_size'] = 0.005
    elif damage == 'backtrack': row['attempts'].append(copy.deepcopy(row['attempts'][0]))
    elif damage == 'selected': row['selected_trial'] = 1
    elif damage == 'trial': row['attempts'][0]['trial'] = 1
    elif damage == 'policy': row['selection_policy'] = 'best_primal'
    elif damage == 'fallback': row['fallback_used'] = True
    elif damage == 'nonfinite_before': row['fixed_before'] = float('nan')
    elif damage == 'nonfinite_after': row['fixed_after'] = float('inf')
    elif damage == 'nonfinite_autodiff': row['attempts'][0]['autodiff_surrogate'] = float('inf')
    elif damage == 'nonfinite_component': row['attempts'][0]['components']['affinity_loss'] = float('nan')
    elif damage == 'candidate_mismatch': row['attempts'][0]['independent_primal_surrogate'] = 0.8
    elif damage == 'increased': row['fixed_increased'] = True
    elif damage == 'strict_contradiction': row['fixed_before'] = 0.8; row['fixed_increased'] = True
    elif damage == 'strict_type': row['attempts'][0]['strict'] = 1
    elif damage == 'noop_type': row['no_op'] = 0
    elif damage == 'coefficient': row['attempts'][0]['repair_weight'] = 0.001
    elif damage == 'missing_component': row['attempts'][0]['components'] = {}
    elif damage == 'continuity': row['step'] = 1
    elif damage == 'bool_rate': row['attempts'][0]['rate'] = True
    with pytest.raises(ValueError):
        contract.fixed_counts([row], 0.01)


def test_readiness_output_and_scientific_source_are_independent_roots():
    assert {p['job'] for p in contract.ARMS.values()} == {'29777709', '29778203', '29778204'}
    for arm in contract.ARMS:
        assert contract.source_path(arm) == contract.SCIENCE
        assert contract.ready_path(arm).is_relative_to(contract.SCIENCE_ROOT)
        assert contract.study_path(arm).is_relative_to(contract.native_parent(arm))
        assert not contract.study_path(arm).is_relative_to(contract.SCIENCE_ROOT)
    assert contract.native_parent('protenix').is_relative_to(contract.ROOT03)
    assert contract.native_parent('boltz2').is_relative_to(contract.ROOT04)
    assert contract.native_parent('boltz2_chemistry').is_relative_to(contract.HOME04)


@pytest.mark.parametrize('status', ['PEND', 'RUN', 'DONE', 'EXIT'])
def test_actual_scheduler_outcome_is_preserved(status):
    assert contract.status_or_completion(scheduler('29778203', status), 'boltz2') == status


@pytest.mark.parametrize('raw', [scheduler('29777710'), scheduler('29778203', queue='hpc'),
                                scheduler('29778203') + scheduler('29778204'),
                                scheduler('29778203', 'UNKWN')])
def test_scheduler_wrong_job_queue_or_ambiguous_record_rejected(raw):
    with pytest.raises(ValueError):
        contract.status_or_completion(raw, 'boltz2')


def test_missing_scheduler_history_without_completion_is_not_done(tmp_path):
    with pytest.raises(ValueError, match='no authenticated completion'):
        contract.status_or_completion('Job <29778203> is not found', 'boltz2', tmp_path)


@pytest.mark.parametrize('status', ['RUN', 'PEND'])
def test_final_scheduler_cannot_certify_a_nonterminal_state(status):
    with pytest.raises(ValueError):
        contract.scheduler_outcome(scheduler('29778203', status), '29778203')


@pytest.fixture
def terminal_case(tmp_path):
    study = tmp_path / 'study'
    rows = [update(0), update(1, 0.9, 1.1, strict=False)]
    case = {'status': 'passed', 'requested_updates': 2, 'updates': rows,
            'states': [{'step': i, 'repair_weight': 0.0} for i in range(3)], 'initial_checks': [{'field': 'atoms'}, {'field': 'bonds'}],
            'acceptance_policy': 'fixed_step', 'nominal_step_size': 0.01, 'parameterization': 'categorical_logits_v1',
            'key_data': [0, 17], 'repair_weight_schedule': None,
            'affinity_objective_identity': contract.objective_identity('boltz2'),
            'counts': {'applied': 2, 'strict_improving': 1, 'forced_fallback': 0, 'no_op': 0}}
    config = {'arm': 'boltz2', 'label': 'sealed-label', 'initialization': {'kind': 'chain_uniform', 'mixing': 0.5}}
    summary = {'status': 'passed', 'config': config, 'cases': [dict(case, label=config['label'], initialization=config['initialization'])]}
    put(study / 'case-0/report.json', case)
    put(study / 'case-0/combined_initial_checks/report.json', {'status': 'passed', 'directions': case['initial_checks']})
    for name in ('updates', 'states'):
        (study / ('case-0/' + name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in case[name]))
    return study, summary, case


def test_exact_summary_wrapper_and_terminal_jsonl_equality(terminal_case):
    study, summary, case = terminal_case
    assert contract.verify_case(study, summary, steps=2) == case


@pytest.mark.parametrize('damage', ['wrapper_label', 'wrapper_initialization', 'case_extra', 'jsonl_changed',
                                    'jsonl_uncommitted', 'jsonl_missing', 'budget', 'old_count_formula', 'coupled'])
def test_terminal_wrapper_budget_jsonl_or_count_drift_rejects(terminal_case, damage):
    study, summary, case = terminal_case
    if damage == 'wrapper_label': summary['cases'][0]['label'] = 'wrong'
    elif damage == 'wrapper_initialization': summary['cases'][0]['initialization'] = {'kind': 'uniform'}
    elif damage == 'case_extra': case['extra'] = True; put(study / 'case-0/report.json', case)
    elif damage == 'jsonl_changed': (study / 'case-0/updates.jsonl').write_text('{}\n')
    elif damage == 'jsonl_uncommitted':
        path = study / 'case-0/states.jsonl'; path.write_text(path.read_text().rstrip('\n'))
    elif damage == 'jsonl_missing': (study / 'case-0/states.jsonl').write_text('')
    elif damage == 'budget': case['requested_updates'] = 3; put(study / 'case-0/report.json', case); summary['cases'][0]['requested_updates'] = 3
    elif damage == 'old_count_formula':
        case['counts']['strict_improving'] = 2; put(study / 'case-0/report.json', case)
    elif damage == 'coupled': put(study / 'case-0/combined_initial_checks/report.json', {'status': 'failed', 'directions': case['initial_checks']})
    with pytest.raises(ValueError):
        contract.verify_case(study, summary, steps=2)


def test_immutable_source_rehash_rejects_changed_member_and_escape(tmp_path):
    source = tmp_path / 'source'
    member = put(source / 'scripts/tool.json', {'original': True})
    manifest = put(source / 'source_manifest.json', {'files': {'scripts/tool.json': contract.sha(member)}})
    pin = contract.sha(manifest)
    assert contract._release(source, pin)['files']
    member.write_text('{}')
    with pytest.raises(ValueError, match='bytes changed'):
        contract._release(source, pin)
    put(manifest, {'files': {'../outside.json': '0' * 64}})
    with pytest.raises(ValueError, match='Unsafe source'):
        contract._release(source, contract.sha(manifest))


@pytest.fixture
def collector(tmp_path, monkeypatch):
    arm = 'boltz2'
    parent = tmp_path / 'native/run-boltz2'
    study = parent / 'payload/study'
    completions = tmp_path / 'report'
    output = completions / 'complete-boltz2/payload'
    paths = [put(study / p, {'saved': p}) for p in ('summary.json', 'case-0/report.json', 'case-0/states.jsonl', 'case-0/updates.jsonl')]
    output.mkdir(parents=True)
    sched = output / 'native-scheduler-0.txt'; sched.write_text(scheduler('29778203'))
    stdout = output / 'native-stdout.txt'; stdout.write_text(lsf('29778203'))
    paths += [sched, stdout]
    proofs = [contract.hashed(p) for p in paths]
    assessment = {'status': 'passed', 'native_job': '29778203', 'arm': arm, 'n_atoms': 25,
                  'updates': 10000, 'states': 10001, 'source_manifest_sha256': contract.SCIENCE_MANIFEST, 'proofs': proofs}
    path = put(output / 'assessment.json', assessment)
    monkeypatch.setitem(contract.ARMS, arm, dict(contract.ARMS[arm], output_parent=str(parent)))
    called = []
    monkeypatch.setattr(contract, 'authenticate_study', lambda selected: called.append(selected))
    return completions, path, assessment, called


def test_agedout_completed_arm_needs_actual_same_job_scheduler_lsf_and_full_study(collector):
    completions, _, _, called = collector
    assert contract.status_or_completion('Job <29778203> is not found', 'boltz2', completions) == 'DONE'
    assert called == ['boltz2']


@pytest.mark.parametrize('damage', ['wrong_job', 'failed', 'wrong_source', 'short_budget', 'proof_tamper',
                                    'missing_scheduler', 'missing_lsf', 'missing_original', 'duplicate_proof', 'bad_lsf'])
def test_completion_cannot_turn_missing_or_failed_evidence_into_done(collector, damage):
    completions, path, assessment, called = collector
    if damage == 'wrong_job': assessment['native_job'] = '29777710'
    elif damage == 'failed': assessment['status'] = 'failed'
    elif damage == 'wrong_source': assessment['source_manifest_sha256'] = '0' * 64
    elif damage == 'short_budget': assessment['updates'] = 9999
    elif damage == 'proof_tamper': Path(assessment['proofs'][0]['path']).write_text('{}')
    elif damage == 'missing_scheduler': assessment['proofs'] = [p for p in assessment['proofs'] if 'native-scheduler' not in p['path']]
    elif damage == 'missing_lsf': assessment['proofs'] = [p for p in assessment['proofs'] if not p['path'].endswith('native-stdout.txt')]
    elif damage == 'missing_original': assessment['proofs'].pop(0)
    elif damage == 'duplicate_proof': assessment['proofs'].append(assessment['proofs'][0])
    elif damage == 'bad_lsf':
        proof = assessment['proofs'][-1]; Path(proof['path']).write_text(lsf('29777710')); proof['sha256'] = contract.sha(proof['path'])
    put(path, assessment)
    with pytest.raises(ValueError):
        contract.status_or_completion('Job <29778203> is not found', 'boltz2', completions)
    assert not called


def test_completion_metadata_never_substitutes_for_full_native_study(collector, monkeypatch):
    completions, _, _, _ = collector
    def failed_native(arm):
        raise ValueError('Terminal native derivative failed')
    monkeypatch.setattr(contract, 'authenticate_study', failed_native)
    with pytest.raises(ValueError, match='Terminal native derivative'):
        contract.status_or_completion('Job not found', 'boltz2', completions)


@pytest.fixture
def observed_pro(tmp_path, monkeypatch):
    root = tmp_path / 'recovery'
    parent = tmp_path / 'source03/run-protenix'
    study = parent / 'payload/study'
    monkeypatch.setattr(contract, 'ROOT04', root)
    monkeypatch.setitem(contract.ARMS, 'protenix', dict(contract.ARMS['protenix'], output_parent=str(parent)))
    paths = [put(study / name, {'original': name}) for name in (
        'summary.json', 'case-0/report.json', 'case-0/updates.jsonl', 'case-0/states.jsonl',
        'native_gradient_checks/report.json', 'terminal_gradient_checks/report.json',
        'case-0/combined_initial_checks/report.json')]
    actual = root / 'pro-observer/native-scheduler-0.txt'
    actual.parent.mkdir(parents=True)
    actual.write_text(scheduler('29777709'))
    stdout = parent / 'stdout.29777709.log'; stdout.write_text(lsf('29777709'))
    proofs = [contract.hashed(path) for path in paths + [actual, stdout]]
    observer_log = root / 'pro-observer/stdout.29778184.log'; observer_log.write_text(lsf('29778184'))
    monkeypatch.setattr(contract, 'PRO_OBSERVER_STDOUT', contract.sha(observer_log))
    record = {'version': 'fixed25-two-arm-storage-recovery-v1', 'status': 'observed_success',
              'native_job': '29777709', 'observer_job': '29778184', 'updates': 10000, 'states': 10001,
              'tool_manifest_sha256': contract.SOURCE04_MANIFEST, 'cpu_certificate_sha256': contract.CPU04,
              'adoption_sha256': contract.ADOPTION04, 'scheduler': proofs[-2], 'stdout': proofs[-1], 'proofs': proofs}
    path = put(root / 'pro-terminal-observation.json', record)
    monkeypatch.setattr(contract, 'PRO_OBSERVATION', contract.sha(path))
    put(root / 'observer-submission.json', {'job_id': '29778184', 'outcome': 'submitted', 'producer_job': '29777709'})
    called = []
    monkeypatch.setattr(contract, 'authenticate_study', lambda arm: called.append(arm))
    return path, record, called


def test_agedout_protenix_uses_original_actual_observation_and_full_study(observed_pro):
    _, _, called = observed_pro
    record = contract.terminal_evidence('protenix')
    assert record['status'] == 'DONE' and record['source'] == 'authenticated_original_pro_observer'
    assert record['scheduler']['path'].endswith('native-scheduler-0.txt')
    assert called == ['protenix']


@pytest.mark.parametrize('damage', ['record_hash', 'native_job', 'observer_job', 'status', 'budget', 'source',
                                    'cpu', 'adoption', 'proof_hash', 'scheduler_exit', 'scheduler_wrong_job',
                                    'missing_original', 'observer_wrong_lsf'])
def test_observer_missing_or_wrong_proofs_do_not_fabricate_done(observed_pro, monkeypatch, damage):
    path, record, called = observed_pro
    if damage == 'record_hash': record['extra'] = 'changed'
    elif damage == 'native_job': record['native_job'] = '29770179'
    elif damage == 'observer_job': record['observer_job'] = '1'
    elif damage == 'status': record['status'] = 'failed'
    elif damage == 'budget': record['states'] = 10000
    elif damage == 'source': record['tool_manifest_sha256'] = '0' * 64
    elif damage == 'cpu': record['cpu_certificate_sha256'] = '0' * 64
    elif damage == 'adoption': record['adoption_sha256'] = '0' * 64
    elif damage == 'proof_hash': Path(record['proofs'][0]['path']).write_text('{}')
    elif damage in ('scheduler_exit', 'scheduler_wrong_job'):
        proof = record['scheduler']
        Path(proof['path']).write_text(scheduler('29777709', 'EXIT') if damage == 'scheduler_exit' else scheduler('29770179'))
        proof['sha256'] = contract.sha(proof['path'])
    elif damage == 'missing_original': record['proofs'].pop(0)
    elif damage == 'observer_wrong_lsf':
        log = contract.ROOT04 / 'pro-observer/stdout.29778184.log'
        log.write_text(lsf('29777709'))
        monkeypatch.setattr(contract, 'PRO_OBSERVER_STDOUT', contract.sha(log))
    put(path, record)
    if damage != 'record_hash':
        monkeypatch.setattr(contract, 'PRO_OBSERVATION', contract.sha(path))
    with pytest.raises(ValueError):
        contract.status_or_completion('Job <29777709> is not found', 'protenix')
    assert not called


@pytest.mark.parametrize('damage', ['bytes', 'reserve', 'files', 'incomplete_home', 'home_scope'])
def test_storage_proof_keeps_full_reserves_and_complete_home_inventory(damage):
    budget = {'status': 'passed', 'projected_bytes': 1, 'required_bytes': 1 + 4 * 1024**3,
              'available_bytes_conservative': 2 + 4 * 1024**3, 'reserve_bytes': 4 * 1024**3,
              'file_reserve': 100000, 'available_files_lower': 100000,
              'filesystem_inode_reporting': 'reported_physical_filesystem_only'}
    gate = {'status': 'passed', 'phase': 'entry', 'projections': contract.PROJECTIONS,
            'filesystems': {'work3': copy.deepcopy(budget), 'zhome': budget},
            'recent_home_inventory': {'complete': True, 'returncode': 0}}
    if damage == 'bytes': budget['available_bytes_conservative'] = 0
    elif damage == 'reserve': budget['reserve_bytes'] -= 1
    elif damage == 'files': budget['available_files_lower'] = 99999
    elif damage == 'incomplete_home': gate['recent_home_inventory']['complete'] = False
    elif damage == 'home_scope': budget['filesystem_inode_reporting'] = 'unreported_all_zero'
    with pytest.raises(ValueError):
        contract._storage_passed(gate, 'entry')


@pytest.mark.parametrize('job', ['29777709', '29778184', '29778203'])
def test_lsf_requires_same_job_success(tmp_path, job):
    path = tmp_path / 'stdout.log'
    path.write_text(lsf(job))
    contract.lsf_success(path, job)
    path.write_text(lsf(job) + 'Exited with exit code 1.\n')
    with pytest.raises(ValueError):
        contract.lsf_success(path, job)
