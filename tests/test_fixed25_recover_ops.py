"""Allocated controller fixtures; never run a molecular model or optimization."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fixed25_recover_ops as ops


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Recovery controller fixtures require allocated DTU HPC')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return path


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root, old = tmp_path / 'new', tmp_path / 'old'
    root.mkdir()
    monkeypatch.setattr(ops, 'ROOT', root)
    monkeypatch.setattr(ops, 'ROOT03', old)
    monkeypatch.setattr(ops, 'TOOL_SOURCE', tmp_path / 'release/source')
    monkeypatch.setattr(ops.storage, 'ROOT', root)
    monkeypatch.setattr(ops.storage, 'HOME_ROOT', tmp_path / 'home-new')
    monkeypatch.setattr(ops.storage, 'ACTIVE_ROOT', old)
    return root, old


@pytest.mark.parametrize('function', ['submission_command', 'submit', 'driver_command', 'run_arm'])
def test_recovery_has_no_protenix_execution_route(function):
    call = getattr(ops, function)
    args = ('protenix',) if function == 'driver_command' else ('protenix', 'a' * 64, 'b' * 64, 'c' * 64)
    with pytest.raises(ValueError, match='Protenix'):
        call(*args)


@pytest.mark.parametrize('arm', ops.NEW_ARMS)
def test_two_long_submissions_have_exact_independent_resources_and_wrapper_api(isolated, arm):
    command = ops.submission_command(arm, 'a' * 64, 'b' * 64, 'c' * 64)
    assert command[:5] == ['bsub', '-q', 'gpul40s', '-n', '4']
    assert command[command.index('-W') + 1] == '23:00'
    assert command[command.index('-gpu') + 1] == 'num=1:mode=exclusive_process'
    assert 'rusage[mem=8GB]' in command and 'span[hosts=1]' in command
    assert '-w' not in command and '-H' not in command
    assert command[-6:] == [str(ops.TOOL_SOURCE), 'a' * 64, str(ops.ROOT), arm, 'b' * 64, 'c' * 64]


@pytest.mark.parametrize('arm', ops.NEW_ARMS)
def test_original_scientific_driver_changes_only_output(isolated, monkeypatch, arm):
    original = [ops.PYTHON, str(ops.SCIENCE / 'examples/brd4_25_atoms.py'), '--mode', 'run', '--config',
                '/original/fixed-config.json', '--output', '/original/output', '--manifest', ops.SCIENCE_MANIFEST,
                '--ready', '/original/ready.json', '--ready-sha256', 'd' * 64, '--prepared', '/original/prepared']
    monkeypatch.setattr(ops, 'legacy_ops', lambda: SimpleNamespace(driver_command=lambda a, python: list(original)))
    actual = ops.driver_command(arm)
    expected = list(original)
    expected[expected.index('--output') + 1] = str(ops.storage.new_output_parent(arm) / 'payload/study')
    assert actual == expected and original[original.index('--output') + 1] == '/original/output'


@pytest.mark.parametrize('status', ['RUN', 'PEND', 'DONE', 'EXIT'])
def test_actual_scheduler_wrapped_values_are_not_invented(status):
    wrapped = status[:1] + '\n ' + status[1:]
    assert ops.scheduler_status(f'Job <{ops.PRO_JOB}>, Status <{wrapped}>, Queue <gpu\nl40s>', ops.PRO_JOB, 'gpul40s') == status


@pytest.mark.parametrize('raw', ['Job <29777709> is not found', 'Job <1>, Status <DONE>, Queue <gpul40s>',
                               'Job <29777709>, Status <DONE>, Queue <hpc>',
                               'Job <29777709>, Status <DONE>, Status <EXIT>, Queue <gpul40s>'])
def test_missing_wrong_or_conflicting_scheduler_identity_blocks(raw):
    with pytest.raises(ValueError):
        ops.scheduler_status(raw, ops.PRO_JOB, 'gpul40s')


def failed_fixture(old, arm, monkeypatch):
    job = ops.FAILURES[arm]['job']
    scheduler = old / 'entry-failures-01' / ('run-' + arm) / 'actual-final-scheduler.txt'
    scheduler.parent.mkdir(parents=True)
    scheduler.write_text(f'Job <{job}>, Status <EXIT>, Queue <gpul40s>')
    parent = old / ('run-' + arm)
    (parent / 'payload').mkdir(parents=True)
    stdout = parent / ('stdout.' + job + '.log')
    stdout.write_text(f'Subject: Job {job}: fixture Exited\nExited with exit code 1.\n')
    quota = put(old / ('entry-quota-' + arm) / 'split-storage-gate.json',
                {'status': 'failed', 'phase': 'entry', 'projections': ops.storage.FULL_PROJECTIONS,
                 'filesystems': {'work3': {'status': 'failed', 'available_bytes_conservative': 1,
                                          'required_bytes': 2, 'reserve_bytes': 4 * 1024**3}}})
    pins = copy.deepcopy(ops.FAILURES)
    pins[arm].update(scheduler=ops.sha(scheduler), stdout=ops.sha(stdout), quota=ops.sha(quota))
    monkeypatch.setattr(ops, 'FAILURES', pins)
    return parent, scheduler, stdout, quota


@pytest.mark.parametrize('arm', ops.NEW_ARMS)
def test_exact_exit1_and_failed_admission_are_authenticated_without_reclassifying(isolated, monkeypatch, arm):
    _, old = isolated
    failed_fixture(old, arm, monkeypatch)
    record = ops.failed_admission(arm)
    assert record['status'] == 'EXIT' and record['scientifically_unstarted']
    assert len(record['proofs']) == 3


@pytest.mark.parametrize('artifact', ['study', 'driver-invocation.json', 'gpu.txt', 'native.log'])
def test_any_old_scientific_execution_blocks_replacement(isolated, monkeypatch, artifact):
    _, old = isolated
    parent, *_ = failed_fixture(old, 'boltz2', monkeypatch)
    (parent / 'payload' / artifact).write_text('present')
    with pytest.raises(ValueError, match='execution exists'):
        ops.failed_admission('boltz2')


def test_changed_failed_evidence_is_not_reused(isolated, monkeypatch):
    _, old = isolated
    _, _, _, quota = failed_fixture(old, 'boltz2', monkeypatch)
    quota.write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        ops.failed_admission('boltz2')


def cpu_fixture(root, mutation=None):
    job = '12345'
    parent = root / 'cpu'
    parent.mkdir(parents=True)
    scheduler = parent / 'scheduler-final-0.txt'
    scheduler.write_text(f'Job <{job}>, Status <DONE>, Queue <hpc>')
    stdout = parent / ('stdout.' + job + '.log')
    stdout.write_text(f'Subject: Job {job}: fixture Done\nSuccessfully completed.\n')
    xml = parent / 'contracts.xml'
    xml.write_text('<testsuites tests="2"><testsuite tests="2"><testcase classname="test_fixed25_recover_ops"/>'
                   '<testcase classname="test_fixed25_recover_storage"/></testsuite></testsuites>')
    if mutation == 'queue':
        scheduler.write_text(f'Job <{job}>, Status <DONE>, Queue <gpul40s>')
    elif mutation == 'exit':
        stdout.write_text(f'Subject: Job {job}: fixture Exited\nExited with exit code 1.\n')
    elif mutation == 'skip':
        xml.write_text('<testsuites><testsuite tests="2"><testcase classname="test_fixed25_recover_ops"><skipped/></testcase>'
                       '<testcase classname="test_fixed25_recover_storage"/></testsuite></testsuites>')
    elif mutation == 'one_module':
        xml.write_text('<testsuites><testsuite tests="2"><testcase classname="test_fixed25_recover_ops"/>'
                       '<testcase classname="test_fixed25_recover_ops"/></testsuite></testsuites>')
    record = {'status': 'passed', 'source_manifest_sha256': 'a' * 64, 'producer_job_id': job,
              'tests': 2, 'suite': ops.SUITE,
              'proofs': [{'path': str(p), 'sha256': ops.sha(p)} for p in (scheduler, stdout, xml)]}
    if mutation == 'suite':
        record['suite'] = 'unrelated-suite'
    cert = put(root / 'cpu-certificate.json', record)
    return cert


def test_new_cpu_requires_both_actual_passing_modules_and_scheduler(isolated):
    root, _ = isolated
    cert = cpu_fixture(root)
    assert ops.verify_tool_cpu('a' * 64, ops.sha(cert))['tests'] == 2


@pytest.mark.parametrize('mutation', ['queue', 'exit', 'skip', 'one_module', 'suite'])
def test_new_cpu_rejects_wrong_or_failed_evidence_even_when_hashes_match(isolated, mutation):
    root, _ = isolated
    cert = cpu_fixture(root, mutation)
    with pytest.raises(ValueError):
        ops.verify_tool_cpu('a' * 64, ops.sha(cert))


def test_new_cpu_hash_mutation_is_not_accepted(isolated):
    root, _ = isolated
    cert = cpu_fixture(root)
    pin = ops.sha(cert)
    cert.write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        ops.verify_tool_cpu('a' * 64, pin)


def dispatch_fixture(monkeypatch):
    events = []
    adoption = {'evidence': {'original': {'arms': {a: {'projected_bytes': n} for a, n in ops.storage.FULL_PROJECTIONS.items()}}}}
    monkeypatch.setattr(ops, 'check_release', lambda *a: None)
    monkeypatch.setattr(ops, 'verify_adoption', lambda *a: adoption)
    monkeypatch.setattr(ops, 'verify_tool_cpu', lambda *a: {})
    monkeypatch.setattr(ops, 'pro_status', lambda: events.append('pro-status') or 'RUN')
    monkeypatch.setattr(ops, 'submit_observer', lambda *a: events.append('observer') or {'job_id': '900', 'outcome': 'submitted'})
    return events


def test_observer_precedes_whole_filesystem_gate_failure_and_no_long_is_submitted(isolated, monkeypatch):
    root, _ = isolated
    events = dispatch_fixture(monkeypatch)
    def fail(projections, receipt, *, phase):
        events.append('quota')
        assert projections == ops.storage.FULL_PROJECTIONS and phase == 'dispatch'
        raise ValueError('headroom')
    monkeypatch.setattr(ops.storage, 'check_campaign_storage', fail)
    monkeypatch.setattr(ops, 'submit', lambda *a: pytest.fail('Scientific long submitted after resource failure'))
    with pytest.raises(ValueError, match='headroom'):
        ops.dispatch('a' * 64, 'b' * 64, 'c' * 64)
    assert events == ['observer', 'pro-status', 'quota']
    assert not (root / 'long-dispatch-started.json').exists()


def test_dispatch_submits_exactly_two_and_preserves_existing_protenix(isolated, monkeypatch):
    root, old = isolated
    preserved = put(old / 'run-protenix/preserve.json', {'job': ops.PRO_JOB})
    before = preserved.read_bytes()
    events = dispatch_fixture(monkeypatch)
    monkeypatch.setattr(ops.storage, 'check_campaign_storage', lambda *a, **k: {'status': 'passed'})
    def submitted(arm, *pins):
        events.append(arm)
        return {'arm': arm, 'job_id': str(len(events)), 'outcome': 'submitted'}
    monkeypatch.setattr(ops, 'submit', submitted)
    result = ops.dispatch('a' * 64, 'b' * 64, 'c' * 64)
    assert [r['arm'] for r in result['long_jobs']] == ['boltz2', 'boltz2_chemistry']
    assert ops.read(root / 'long-submissions.json')['both_boltz_submitted']
    assert preserved.read_bytes() == before
    with pytest.raises(ValueError, match='dispatch exists'):
        ops.dispatch('a' * 64, 'b' * 64, 'c' * 64)
    assert events.count('boltz2') == events.count('boltz2_chemistry') == 1


def test_partial_unknown_dispatch_marker_prevents_all_retries(isolated, monkeypatch):
    root, _ = isolated
    dispatch_fixture(monkeypatch)
    monkeypatch.setattr(ops.storage, 'check_campaign_storage', lambda *a, **k: {'status': 'passed'})
    calls = []
    def unknown(*args):
        calls.append(args[0])
        raise RuntimeError('unknown outcome')
    monkeypatch.setattr(ops, 'submit', unknown)
    with pytest.raises(RuntimeError, match='unknown'):
        ops.dispatch('a' * 64, 'b' * 64, 'c' * 64)
    assert (root / 'long-dispatch-started.json').is_file()
    with pytest.raises(ValueError, match='dispatch exists'):
        ops.dispatch('a' * 64, 'b' * 64, 'c' * 64)
    assert calls == ['boltz2']


def test_bsub_timeout_has_durable_unknown_receipt_and_no_second_bsub(isolated, monkeypatch):
    calls = []
    def timeout(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 60)
    monkeypatch.setattr(ops.subprocess, 'run', timeout)
    with pytest.raises(ValueError, match='Unknown/failed'):
        ops.submit('boltz2', 'a' * 64, 'b' * 64, 'c' * 64)
    parent = ops.storage.new_output_parent('boltz2')
    assert ops.read(parent / 'submission.json')['outcome'] == 'unknown'
    assert (parent / 'submission-started.json').is_file()
    with pytest.raises(FileExistsError):
        ops.submit('boltz2', 'a' * 64, 'b' * 64, 'c' * 64)
    assert len(calls) == 1


def test_observer_is_one_small_cpu_job_with_only_own_ended_dependency(isolated, monkeypatch):
    commands = []
    def reply(command, **kwargs):
        commands.append(command)
        text = 'Job <991> is submitted to queue <hpc>.\n' if command[0] == 'bsub' else 'Job <991>, Status <PEND>, Queue <hpc>'
        return SimpleNamespace(returncode=0, stdout=text, stderr='')
    monkeypatch.setattr(ops.subprocess, 'run', reply)
    first = ops.submit_observer('a' * 64, 'b' * 64, 'c' * 64)
    assert ops.submit_observer('a' * 64, 'b' * 64, 'c' * 64) == first
    assert len([c for c in commands if c[0] == 'bsub']) == 1
    args = first['args']
    assert args[:5] == ['bsub', '-q', 'hpc', '-n', '1'] and 'rusage[mem=2GB]' in args
    assert args[args.index('-w') + 1] == 'ended(29777709)' and args[args.index('-W') + 1] == '00:30'
    assert args[-5:] == [str(ops.TOOL_SOURCE), str(ops.ROOT), 'a' * 64, 'b' * 64, 'c' * 64]


def test_unknown_observer_is_never_duplicated(isolated, monkeypatch):
    monkeypatch.setattr(ops, 'submission_reply', lambda command, record: dict(record, args=command, outcome='unknown'))
    with pytest.raises(ValueError, match='unknown/failed'):
        ops.submit_observer('a' * 64, 'b' * 64, 'c' * 64)
    with pytest.raises(ValueError, match='unknown/different'):
        ops.submit_observer('a' * 64, 'b' * 64, 'c' * 64)


@pytest.mark.parametrize('status', ['RUN', 'EXIT', 'PEND'])
def test_existing_pro_status_is_actual_and_failed_or_unstarted_blocks(isolated, monkeypatch, status):
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0,
        stdout=f'Job <{ops.PRO_JOB}>, Status <{status}>, Queue <gpul40s>', stderr=''))
    if status == 'RUN':
        assert ops.pro_status() == 'RUN'
    else:
        with pytest.raises(ValueError, match='not RUN'):
            ops.pro_status()


def test_missing_pro_live_history_cannot_fabricate_completion(isolated, monkeypatch):
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=255,
        stdout='', stderr='Job <29777709> is not found'))
    with pytest.raises(ValueError, match='no authenticated terminal observation'):
        ops.pro_status()


def test_run_requires_actual_own_lsf_job_before_any_gpu_or_science(isolated, monkeypatch):
    root, _ = isolated
    monkeypatch.setattr(ops, 'check_release', lambda *a: None)
    monkeypatch.setattr(ops, 'verify_adoption', lambda *a: {})
    monkeypatch.setattr(ops, 'verify_tool_cpu', lambda *a: {})
    monkeypatch.setattr(ops, 'pro_status', lambda: 'RUN')
    manifest, cpu, adoption = 'a' * 64, 'b' * 64, 'c' * 64
    put(root / 'long-dispatch-started.json', {'version': ops.VERSION, 'tool_manifest_sha256': manifest,
        'science_manifest_sha256': ops.SCIENCE_MANIFEST, 'cpu_certificate_sha256': cpu,
        'adoption_sha256': adoption, 'placement': ops.storage.PLACEMENT, 'existing_protenix_job': ops.PRO_JOB})
    parent = ops.storage.new_output_parent('boltz2')
    put(parent / 'submission.json', {'outcome': 'submitted', 'job_id': '123', 'arm': 'boltz2',
        'output_parent': str(parent), 'science_manifest_sha256': ops.SCIENCE_MANIFEST,
        'tool_manifest_sha256': manifest, 'args': ops.submission_command('boltz2', manifest, cpu, adoption)})
    monkeypatch.setenv('LSB_JOBID', '999')
    monkeypatch.setattr(ops.storage, 'check_campaign_storage', lambda *a, **k: pytest.fail('Quota reached for wrong job'))
    with pytest.raises(ValueError, match='actual job/submission'):
        ops.run_arm('boltz2', manifest, cpu, adoption)
    assert not (parent / 'payload').exists()


def test_helper_cli_rejects_protenix_even_before_dispatch(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fixed25_recover_ops.py', '--phase', 'run', '--tool-source', str(ops.TOOL_SOURCE),
        '--tool-manifest', 'a' * 64, '--root', str(ops.ROOT), '--arm', 'protenix'])
    with pytest.raises(SystemExit):
        ops.main()


@pytest.mark.parametrize('damage', [None, 'label', 'initialization', 'case', 'jsonl'])
def test_terminal_protenix_accepts_only_declared_summary_wrappers_and_exact_case_jsonl(isolated, monkeypatch, damage):
    _, old = isolated
    science = old.parent / 'science'
    monkeypatch.setattr(ops, 'SCIENCE', science)
    config = {'label': 'fixed-protenix-fixture', 'initialization': {'kind': 'chain_uniform', 'mixing': 0.5}}
    config_file = put(science / 'examples/configs/brd4_jq1_25_fixed_protenix_10000.json', config)
    put(science / 'source_manifest.json', {'files': {'fixture.py': 'a' * 64}})
    pocket = 'b' * 64
    monkeypatch.setattr(ops, 'legacy_ops', lambda: SimpleNamespace(NATIVE_PINS={'protenix': {'config': ops.sha(config_file)}}, POCKET=pocket))
    initial = [{'field': 'atoms'}, {'field': 'bonds'}]
    case = {'status': 'passed', 'requested_updates': 10000, 'initial_checks': initial,
            'counts': {'applied': 10000, 'forced_fallback': 0, 'strict_improving': 10000, 'no_op': 0},
            'updates': [{'step': i, 'selection_policy': 'fixed_step', 'selected_trial': 0,
                         'accepted_step_size': 0.3, 'fallback_used': False, 'no_op': False,
                         'attempts': [{'rate': 0.3, 'strict': True}]} for i in range(10000)],
            'states': [{'step': i, 'repair_weight': 0.0} for i in range(10001)]}
    model = '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2'
    context = 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4'
    wrapped = {'label': config['label'], 'initialization': config['initialization'], **copy.deepcopy(case)}
    if damage == 'label': wrapped['label'] = 'wrong-label'
    if damage == 'initialization': wrapped['initialization'] = {'mixing': 0.1}
    summary = {'status': 'passed', 'job_id': ops.PRO_JOB, 'mode': 'run', 'source_manifest_sha256': ops.SCIENCE_MANIFEST,
               'config_sha256': ops.sha(config_file), 'config': config, 'source': {'files': {'fixture.py': 'a' * 64}},
               'optimization_parameters': {'step_size': 0.3, 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'},
               'pocket_validation_status': 'passed', 'pocket_definition_sha256': pocket, 'diffusion_gradient': False,
               'model_digest': model, 'context_digest': context,
               'frozen_identity_after': {'model_digest': model, 'context_digest': context},
               'native_hard_parity': {'status': 'passed'}, 'native_gradient_checks': {'status': 'passed'},
               'terminal_gradient_checks': {'status': 'passed'}, 'cases': [wrapped]}
    study = old / 'run-protenix/payload/study'
    put(study / 'summary.json', summary)
    if damage == 'case': case['states'][-1]['repair_weight'] = 1.0
    put(study / 'case-0/report.json', case)
    for name in ('updates', 'states'):
        (study / ('case-0/' + name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in case[name]))
    if damage == 'jsonl':
        (study / 'case-0/updates.jsonl').write_text('{"step":0}\n')
    put(study / 'native_gradient_checks/report.json', {'status': 'passed'})
    put(study / 'terminal_gradient_checks/report.json', {'status': 'passed'})
    put(study / 'case-0/combined_initial_checks/report.json', {'status': 'passed', 'directions': initial})
    if damage is None:
        assert len(ops.pro_terminal_proofs()) == 7
    else:
        with pytest.raises(ValueError):
            ops.pro_terminal_proofs()


def test_root_owned_wrappers_match_controller_interfaces_and_authenticate_before_import():
    scripts = Path(__file__).resolve().parents[1] / 'scripts/hpc'
    stage = (scripts / 'fixed25_recover_stage.lsf').read_text()
    observer = (scripts / 'fixed25_recover_observe_pro.lsf').read_text()
    cpu = (scripts / 'fixed25_recover_cpu.lsf').read_text()
    after = (scripts / 'fixed25_recover_after_cpu.lsf').read_text()
    assert 'tool=$1\nmanifest=$2\nroot=$3\narm=$4' in stage
    assert 'tool=$1\nroot=$2\nmanifest=$3\ncpu_sha=$4\nadoption_sha=$5' in observer
    assert '--phase observe-pro' in observer and '--phase run' in stage
    assert '[[ "$arm" == boltz2 || "$arm" == boltz2_chemistry ]]' in stage
    for text in (stage, observer, cpu, after):
        assert ops.SOURCE03_MANIFEST in text and ops.SCIENCE_MANIFEST in text
        assert '$tool/scripts:$original_launcher/scripts:$science/scripts' in text
    assert ops.SUITE in cpu and ops.SUITE in after
    assert 'test_fixed25_recover_ops.py' in cpu and 'test_fixed25_recover_storage.py' in cpu
