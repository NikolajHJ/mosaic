"""Allocated metadata contracts; synthetic proofs never establish native passes."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

try:
    import adoption_contract as contract
    import adoption_ops as ops
except ModuleNotFoundError:
    repository = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/hpc/peer_adoption').is_dir())
    sys.path.insert(0, str(repository / 'scripts/hpc/peer_adoption'))
    import adoption_contract as contract
    import adoption_ops as ops


# Declared rows are independently counted from the allocated raw XML.
METADATA_CASES = 75


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True) + '\n'
    path.write_text(data)
    return path


def unlock(root):
    if root.exists():
        root.chmod(0o755)
        for path in root.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)


def seal(root):
    files = {p.relative_to(root).as_posix(): contract.sha(p) for p in root.rglob('*')
             if p.is_file() and p.name != 'source_manifest.json'}
    write(root / 'source_manifest.json', {'git_revision_reference': contract.REVISION, 'files': files})
    for path in root.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    return contract.sha(root / 'source_manifest.json')


def lsf(parent, job, status='Done'):
    suffix = 'Successfully completed.' if status == 'Done' else 'Exited with exit code 1.'
    return write(parent / ('stdout.' + job + '.log'), 'Subject: Job ' + job + ': ' + status + '\n' + suffix + '\n')


def certificate(path, source, proofs, cases=None, **extra):
    value = {'status': 'passed', 'source_manifest_sha256': source,
             'proofs': [contract.file_record(p) for p in proofs], **extra}
    if cases is not None:
        value['cases'] = cases
    write(path, value)
    return value


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    profiles = copy.deepcopy(contract.PROFILES)
    for number, (key, p) in enumerate(profiles.items()):
        root = tmp_path / key
        p.update(source=root / 'source', root=root / 'output', control=root / 'control', launcher=root / 'launcher')
        p['source'].mkdir(parents=True); p['root'].mkdir(); p['control'].mkdir(); p['launcher'].mkdir()
        for arm in p['arms']:
            name = 'dbmol_pilot' if key == 'dbmol10k' else p['config_prefix'] + arm
            write(p['source'] / 'examples/configs' / (name + '.json'), {'arm': arm, 'synthetic_metadata_only': True})
        p['manifest'] = seal(p['source'])
        if key == 'dbmol10k':
            p['original_launcher'] = root / 'original-launcher'
            write(p['original_launcher'] / 'scripts/placeholder.py', '# synthetic metadata only\n')
            p['original_launcher_manifest'] = seal(p['original_launcher'])
        raw = write(p['root'] / 'cpu-proof.txt', 'allocated synthetic fixture proof\n')
        kwargs = {'launcher_manifest_sha256': p['original_launcher_manifest']} if key == 'dbmol10k' else {}
        certificate(p['root'] / 'cpu-certificate.json', p['manifest'], [raw], p['cpu_cases'], **kwargs)
        p['cpu_sha256'] = contract.sha(p['root'] / 'cpu-certificate.json')
    monkeypatch.setattr(contract, 'PROFILES', profiles)
    monkeypatch.setattr(ops, 'PROFILES', profiles)
    for number, (key, p) in enumerate(profiles.items()):
        jobs, projection = {}, {}
        p['native_ids'] = {}
        for arm_index, arm in enumerate(p['arms']):
            native = contract.native_name(key, arm)
            callback = contract.callback_name(key, arm)
            native_id, callback_id = str(10000 + number * 100 + arm_index * 2), str(10001 + number * 100 + arm_index * 2)
            p['native_ids'][arm] = (native_id, callback_id)
            for label, phase, job, dependency in ((native, 'native', native_id, None),
                    (callback, 'after-native', callback_id, 'ended(' + native_id + ')')):
                parent = p['root'] / label
                write(parent / 'submission.json', {'outcome': 'submitted', 'phase': phase, 'job_id': job,
                    'dependency': dependency, 'args': contract.original_command(key, label, phase, dependency)})
                lsf(parent, job)
                jobs[label] = job
            ready = write(p['root'] / native / 'payload/study/assessment.json', {
                'status': 'passed', 'source_manifest_sha256': p['manifest'], 'arm': 'protenix' if key == 'protenix50k40k' else arm,
                'config_sha256': contract.sha(contract.profile_config(key, arm)),
                'storage_projection': {'maximum_pilot_update_bytes': 1234 + arm_index}})
            certificate(p['root'] / (native + '-certificate.json'), p['manifest'],
                        [ready, p['root'] / native / ('stdout.' + native_id + '.log')])
            projection[arm] = {'unit_bytes': 1234 + arm_index}
        write(p['root'] / 'jobs.json', jobs)
        if key == 'dbmol10k':
            unit = projection['boltz2_dbmol']['unit_bytes']
            projection = {'status': 'passed', 'source_manifest_sha256': p['manifest'], 'unit_bytes': unit,
                          'full_projected_bytes': 2 * unit * 10000 + 4 * contract.GIB}
        write(p['root'] / 'projection.json', projection)
        write(p['control'] / 'jobs.json', {})
        write(p['control'] / 'metadata-cpu-certificate.json', {'status': 'passed', 'version': contract.VERSION})
    try:
        yield profiles
    finally:
        for p in profiles.values():
            unlock(p['source'])
            if 'original_launcher' in p:
                unlock(p['original_launcher'])


def sync_ready(profile, arm, mutate):
    p = contract.PROFILES[profile]
    name = contract.native_name(profile, arm)
    path = p['root'] / name / 'payload/study/assessment.json'
    value = contract.read(path); mutate(value); write(path, value)
    cp = p['root'] / (name + '-certificate.json')
    cert = contract.read(cp)
    cert['proofs'] = [contract.file_record(item['path']) for item in cert['proofs']]
    write(cp, cert)


def test_complete_readonly_seal_is_authenticated_before_import(campaign):
    p = campaign['boltz30k']
    proofs = contract.verify_seal(p['source'], p['manifest'])
    assert len(proofs) == 3 and all(item['sha256'] == contract.sha(item['path']) for item in proofs)


@pytest.mark.parametrize('fault', ('manifest', 'revision', 'member', 'extra', 'root_writable', 'directory_writable', 'symlink'))
def test_source_tampering_or_writable_scope_is_rejected(campaign, fault):
    p = campaign['boltz30k']; root = p['source']; manifest = p['manifest']
    if fault == 'manifest':
        manifest = 'a' * 64
    elif fault == 'root_writable':
        root.chmod(0o755)
    elif fault == 'directory_writable':
        (root / 'examples').chmod(0o755)
    else:
        unlock(root)
        if fault == 'revision':
            value = contract.read(root / 'source_manifest.json'); value['git_revision_reference'] = 'b' * 40
            write(root / 'source_manifest.json', value); manifest = contract.sha(root / 'source_manifest.json')
        elif fault == 'member':
            write(contract.profile_config('boltz30k', 'boltz2'), {'tampered': True})
        elif fault == 'extra':
            write(root / 'unlisted.py', '# changed\n')
        else:
            (root / 'escape').symlink_to(root / 'examples', target_is_directory=True)
        for path in root.rglob('*'):
            if not path.is_symlink(): path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555)
    with pytest.raises((ValueError, KeyError)):
        contract.verify_seal(root, manifest)


def test_certificate_passes_only_with_exact_source_cases_and_all_original_bytes(campaign):
    p = campaign['dbmol10k']
    result = contract.verify_certificate(p['root'] / 'cpu-certificate.json', p['manifest'], p['cpu_cases'])
    assert result['status'] == 'passed' and result['cases'] == p['cpu_cases']


@pytest.mark.parametrize('fault', ('status', 'source', 'cases', 'empty', 'duplicate', 'changed_bytes'))
def test_metadata_certificate_cannot_replace_or_invent_a_scientific_proof(campaign, fault):
    p = campaign['dbmol10k']; path = p['root'] / 'cpu-certificate.json'; value = contract.read(path)
    if fault == 'status': value['status'] = 'failed'
    elif fault == 'source': value['source_manifest_sha256'] = 'b' * 64
    elif fault == 'cases': value['cases'] += 1
    elif fault == 'empty': value['proofs'] = []
    elif fault == 'duplicate': value['proofs'] += value['proofs']
    else: write(Path(value['proofs'][0]['path']), 'changed proof\n')
    write(path, value)
    with pytest.raises(ValueError): contract.verify_certificate(path, p['manifest'], p['cpu_cases'])


def raw_xml(path, count=2, classname='test_peer_adoption', fault=None):
    suite = ET.Element('testsuite', tests=str(count))
    for index in range(count):
        case = ET.SubElement(suite, 'testcase', name='case_' + str(index), classname=classname)
        if index == 0 and fault in ('failure', 'error', 'skipped'): ET.SubElement(case, fault)
    path.write_bytes(ET.tostring(suite))
    write(path.with_suffix('.log'), str(count) + ' passed\n')


def test_raw_xml_requires_actual_cases_module_and_passing_log(tmp_path):
    path = tmp_path / 'metadata.xml'; raw_xml(path)
    assert len(contract.verify_metadata_xml(path, 2, {'test_peer_adoption'})) == 2


@pytest.mark.parametrize('fault', ('count', 'failure', 'error', 'skipped', 'attribution', 'log'))
def test_xml_headers_alone_never_establish_a_metadata_pass(tmp_path, fault):
    path = tmp_path / 'metadata.xml'; raw_xml(path, classname='' if fault == 'attribution' else 'test_peer_adoption', fault=fault)
    if fault == 'log': write(path.with_suffix('.log'), '2 failed\n')
    with pytest.raises(ValueError): contract.verify_metadata_xml(path, 3 if fault == 'count' else 2, {'test_peer_adoption'})


@pytest.mark.parametrize('profile', ('boltz30k', 'dbmol10k', 'protenix50k40k'))
def test_actual_initial_native_ids_and_own_ended_vectors_are_preserved(campaign, profile):
    arm = next(iter(campaign[profile]['arms']))
    actual = contract.original_submissions(profile, arm)
    assert actual['after-native']['dependency'] == 'ended(' + actual['native']['job_id'] + ')'
    assert actual['native']['args'][1:14] == ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R',
        'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00', '-J']


@pytest.mark.parametrize('fault', ('unknown_id', 'mapping', 'source', 'resources', 'dependency', 'outcome'))
def test_changed_original_submission_is_not_adopted(campaign, fault):
    p = campaign['boltz30k']; label = contract.callback_name('boltz30k', 'boltz2'); path = p['root'] / label / 'submission.json'
    value = contract.read(path)
    if fault == 'unknown_id': value['job_id'] = 'unknown'
    elif fault == 'mapping': value['job_id'] = '999'
    elif fault == 'source': value['args'][-5] = '/different/source'
    elif fault == 'resources': value['args'][value['args'].index('-n') + 1] = '1'
    elif fault == 'dependency': value['dependency'] = 'done(1)'
    else: value['outcome'] = 'unknown'
    write(path, value)
    with pytest.raises(ValueError): contract.original_submissions('boltz30k', 'boltz2')


@pytest.mark.parametrize('state', ('known', 'missing', 'failed', 'duplicate', 'no_footer'))
def test_observer_waits_only_for_authentic_existing_own_ended_callbacks(campaign, state, monkeypatch):
    p = campaign['boltz30k']; arm = 'boltz2'; label = contract.callback_name('boltz30k', arm)
    path = p['root'] / label / 'submission.json'; job = contract.read(path)['job_id']
    if state == 'missing': path.unlink()
    elif state == 'failed': lsf(path.parent, job, 'Exited')
    elif state == 'no_footer': write(path.parent / ('stdout.' + job + '.log'), 'Callback still running; no terminal LSF footer.\n')
    if state == 'duplicate':
        with pytest.raises(ValueError): contract.pending_dependencies(('boltz30k', 'boltz30k'))
        return
    result = contract.pending_dependencies(('boltz30k', 'dbmol10k', 'protenix50k40k'))
    assert result['status'] == {'known': 'actual_callbacks_known', 'no_footer': 'actual_callbacks_known',
        'missing': 'waiting_unknown_actual_ids', 'failed': 'failed_peer'}[state]
    assert result['all_callbacks_terminal'] is (state in ('known', 'failed'))
    if state in ('known', 'no_footer'):
        assert len(result['actual_callback_ids']) == 5
    if state == 'no_footer':
        assert result['pending_callback_ids'] == [job]
        assert result['dependency'] == 'ended(' + job + ')'
        assert len(result['terminal_callbacks']) == 4
    else:
        assert result['pending_callback_ids'] == []
        assert result['dependency'] is None
        assert len(result['terminal_callbacks']) == (4 if state == 'missing' else 5)
    if state == 'no_footer':
        parent = p['control'] / 'race-observer'; parent.mkdir(); queued = []
        monkeypatch.setattr(ops, 'metadata_certificate', lambda *args: {'status': 'passed'})
        monkeypatch.setattr(ops, 'maybe_queue_observer', lambda *args: queued.append(args))
        monkeypatch.setattr(contract, 'verify_native_units', lambda *args: pytest.fail('Running callback is not a complete native proof'))
        assert ops.adopt('boltz30k', parent, 'a' * 64, SimpleNamespace()) is False
        assert queued == [('boltz30k', parent, 'a' * 64)]
        assert not (p['control'] / 'ownership.json').exists()


@pytest.mark.parametrize('profile', ('boltz30k', 'dbmol10k', 'protenix50k40k'))
def test_full_peer_future_budget_uses_authenticated_native_maximum(campaign, profile):
    result = contract.verify_native_units(profile)
    assert result['status'] == 'passed'
    for arm, unit in result['units'].items():
        updates, segments = campaign[profile]['arms'][arm]
        assert result['full'][arm] == 2 * unit * updates + segments * contract.GIB


@pytest.mark.parametrize('fault', ('projection', 'ready_failed', 'ready_source', 'ready_config', 'zero_unit', 'lsf', 'cpu'))
def test_full_projection_cannot_outlive_changed_native_or_cpu_evidence(campaign, fault):
    p = campaign['boltz30k']; arm = 'boltz2'
    if fault == 'projection':
        path = p['root'] / 'projection.json'; value = contract.read(path); value[arm]['unit_bytes'] += 1; write(path, value)
    elif fault == 'cpu':
        path = p['root'] / 'cpu-certificate.json'; value = contract.read(path); value['cases'] += 1; write(path, value)
    elif fault == 'lsf':
        name = contract.native_name('boltz30k', arm); cp = p['root'] / (name + '-certificate.json')
        value = contract.read(cp); log = Path(value['proofs'][1]['path']); write(log, 'Subject: Job 999: Done\nSuccessfully completed.\n')
        value['proofs'][1] = contract.file_record(log); write(cp, value)
    else:
        def mutate(value):
            if fault == 'ready_failed': value['status'] = 'failed'
            elif fault == 'ready_source': value['source_manifest_sha256'] = 'a' * 64
            elif fault == 'ready_config': value['config_sha256'] = 'b' * 64
            else: value['storage_projection']['maximum_pilot_update_bytes'] = 0
        sync_ready('boltz30k', arm, mutate)
    with pytest.raises(ValueError): contract.verify_native_units('boltz30k')


@pytest.mark.parametrize('marker', ('none', 'segment', 'completion', 'dispatch', 'jobs', 'boundary'))
def test_adoption_refuses_existing_or_unknown_production_ownership(campaign, marker):
    p = campaign['boltz30k']; root = p['root']
    if marker == 'none': assert contract.absence_of_production('boltz30k'); return
    if marker == 'segment': write(root / 'boltz2-segment-0000/submission-started.json', {'outcome': 'unknown'})
    elif marker == 'completion': write(root / 'completion.json', {})
    elif marker == 'dispatch': write(root / 'dispatch-started.json', {'outcome': 'unknown'})
    elif marker == 'jobs':
        value = contract.read(root / 'jobs.json'); value['boltz2-segment-0000'] = '200'; write(root / 'jobs.json', value)
    else: write(root / 'boltz2-native-0001/submission-started.json', {'outcome': 'unknown'})
    with pytest.raises(ValueError): contract.absence_of_production('boltz30k')


def test_ownership_snapshots_mutable_metadata_but_pins_every_original_proof(campaign):
    p = campaign['boltz30k']; metadata = contract.read(p['control'] / 'metadata-cpu-certificate.json')
    before = contract.proof_inventory('boltz30k')
    result = contract.claim_ownership('boltz30k', metadata)
    assert result['scientific_saved_results_changed'] is False
    assert result['immutable_original_proofs'] == before and contract.verify_inventory(before)
    for name in ('jobs.json', 'projection.json'):
        assert (p['control'] / 'adoption-snapshot' / name).read_bytes() == (p['root'] / name).read_bytes()
    assert contract.read(p['root'] / 'protenix02-adoption-owner.json') == contract.read(p['control'] / 'ownership.json')
    with pytest.raises(FileExistsError): contract.claim_ownership('boltz30k', metadata)


@pytest.mark.parametrize('fault', ('cpu', 'callback', 'unknown'))
def test_ownership_requires_independent_metadata_cpu_and_all_actual_callback_success(campaign, fault):
    p = campaign['boltz30k']; metadata = contract.read(p['control'] / 'metadata-cpu-certificate.json')
    if fault == 'cpu': metadata['status'] = 'failed'
    elif fault == 'callback':
        parent = p['root'] / contract.callback_name('boltz30k', 'boltz2'); job = contract.read(parent / 'submission.json')['job_id']; lsf(parent, job, 'Exited')
    else: write(p['root'] / 'boltz2-segment-0000/submission-started.json', {'outcome': 'unknown'})
    with pytest.raises(ValueError): contract.claim_ownership('boltz30k', metadata)
    assert not (p['control'] / 'ownership.json').exists()


def test_changed_original_inventory_never_becomes_a_valid_new_snapshot(campaign):
    p = campaign['dbmol10k']; records = contract.proof_inventory('dbmol10k')
    path = Path(records[0]['path']); path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError): contract.verify_inventory(records)


@pytest.mark.parametrize('scope', ('declared', 'additional', 'missing_parent'))
def test_unknown_external_capsules_block_capacity_instead_of_assuming_zero(tmp_path, monkeypatch, scope):
    parents = [tmp_path / 'runs', tmp_path / 'releases']
    for path in parents: path.mkdir()
    monkeypatch.setattr(contract, 'NOISY_PARENTS', tuple(parents))
    (parents[0] / '20261023-protenix-noisy-01').mkdir()
    if scope == 'missing_parent':
        parents[1].rmdir()
        with pytest.raises(ValueError): contract.unknown_noisy_capsules()
    else:
        if scope == 'additional': (parents[1] / '20261024-protenix-noisy-02').mkdir()
        value = contract.unknown_noisy_capsules()
        assert value['status'] == ('declared_capsules_only' if scope == 'declared' else 'unknown_capsules_blocked')
        assert value['additional_future_capacity_established'] is False


@pytest.mark.parametrize('profile', ('boltz30k', 'dbmol10k'))
def test_metadata_observer_command_uses_only_actual_ended_ids_and_exact_resources(campaign, profile):
    value = ops.command(profile, 'observe-0001', 'observe', 'a' * 64, 'ended(101)&&ended(202)')
    assert value[1:11] == ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']
    assert value[value.index('-w') + 1] == 'ended(101)&&ended(202)'
    assert str(campaign[profile]['control']) in value and value[-2] == 'observe'
    with pytest.raises(ValueError): ops.command(profile, 'observe-0001', 'observe', 'a' * 64, 'done(101)')


@pytest.mark.parametrize('fault', ('native_zero', 'undeclared_arm', 'index', 'phase'))
def test_metadata_launcher_cannot_replay_initial_native_or_expand_scientific_scope(campaign, fault):
    label, phase = {'native_zero': ('boltz2-native-0000', 'native'), 'undeclared_arm': ('protenix-segment-0000', 'segment'),
                    'index': ('boltz2-segment-0003', 'segment'), 'phase': ('boltz2-segment-0001', 'native')}[fault]
    with pytest.raises(ValueError): ops.parent_path('boltz30k', label, phase)


@pytest.mark.parametrize('state', ('missing', 'failed', 'known', 'unknown_existing'))
def test_observer_submission_waits_without_duplicate_or_invented_dependencies(campaign, monkeypatch, state):
    p = campaign['boltz30k']; parent = p['control'] / 'after-cpu'; parent.mkdir(); calls = []
    result = {'status': {'missing': 'waiting_unknown_actual_ids', 'failed': 'failed_peer'}.get(state, 'actual_callbacks_known'),
              'dependency': None if state in ('missing', 'failed') else 'ended(101)&&ended(202)',
              'all_callbacks_terminal': state == 'failed',
              'actual_callback_ids': ['101', '202'],
              'pending_callback_ids': [] if state in ('missing', 'failed') else ['101', '202']}
    monkeypatch.setattr(contract, 'pending_dependencies', lambda profiles: result)
    monkeypatch.setattr(ops, 'submit', lambda *args: calls.append(args) or {'outcome': 'submitted', 'job_id': '303'})
    if state == 'unknown_existing':
        write(p['control'] / 'jobs.json', {'observe-0001': '303'})
        write(p['control'] / 'observe-0001/submission.json', {'outcome': 'unknown'})
        with pytest.raises(ValueError): ops.maybe_queue_observer('boltz30k', parent, 'a' * 64)
        assert calls == []
    elif state == 'failed':
        with pytest.raises(ValueError): ops.maybe_queue_observer('boltz30k', parent, 'a' * 64)
        assert calls == [] and contract.read(parent / 'peer-observation.json')['status'] == 'failed_peer'
    else:
        value = ops.maybe_queue_observer('boltz30k', parent, 'a' * 64)
        assert len(calls) == (1 if state == 'known' else 0)
        if calls: assert calls[0][-1] == 'ended(101)&&ended(202)'
        else: assert value['status'] == result['status']


@pytest.mark.parametrize('phase', ('native', 'segment'))
def test_gpu_work_does_not_acquire_the_metadata_dispatch_lock(campaign, monkeypatch, phase):
    p = campaign['boltz30k']; label = 'boltz2-' + phase + '-0001'; parent = p['root'] / label
    write(parent / 'submission.json', {'startup_smoke': {'sha256': 'startup'}})
    calls = []
    original = SimpleNamespace(produce=lambda *args: calls.append(args))
    monkeypatch.setenv('LSB_JOBID', '123'); monkeypatch.setattr(ops, 'verify_sources', lambda *args: [])
    monkeypatch.setattr(ops, 'executing', lambda *args: {})
    monkeypatch.setattr(ops, 'load_original', lambda *args: original)
    monkeypatch.setattr(ops, 'metadata_certificate', lambda *args: {})
    monkeypatch.setattr(ops, 'adoption_proofs', lambda *args: [])
    monkeypatch.setattr(ops, 'verify_startup', lambda *args: {'sha256': 'startup'})
    monkeypatch.setattr(ops.fcntl, 'flock', lambda *args: pytest.fail('GPU work retained metadata dispatch lock'))
    monkeypatch.setattr(sys, 'argv', ['adoption_ops.py', '--profile', 'boltz30k', '--launcher', str(p['launcher']),
        '--launcher-manifest', 'a' * 64, '--source', str(p['source']), '--root', str(p['root']), '--manifest', p['manifest'],
        '--control', str(p['control']), '--phase', phase, '--parent', str(parent)])
    ops.main()
    assert len(calls) == 1 and calls[0][1:] == (phase, 'boltz2', 1, p['manifest'])


def test_allocated_metadata_cpu_does_not_require_future_peer_native_success(campaign, monkeypatch):
    p = campaign['boltz30k']; parent = p['control'] / 'cpu'; parent.mkdir(); gates = []
    original = SimpleNamespace(all_budgets=lambda: {'synthetic_validation_scope': True},
        storage_gate=lambda *args: gates.append(args) or {'status': 'passed'})
    monkeypatch.setenv('LSB_JOBID', '123'); monkeypatch.setattr(ops, 'verify_sources', lambda *args: [])
    monkeypatch.setattr(ops, 'executing', lambda *args: {})
    monkeypatch.setattr(ops, 'load_original', lambda *args: original)
    monkeypatch.setattr(contract, 'pending_dependencies', lambda *args: pytest.fail('Metadata CPU waited for future jobs'))
    monkeypatch.setattr(contract, 'verify_native_units', lambda *args: pytest.fail('Metadata CPU replaced a native gate'))
    monkeypatch.setattr(ops.fcntl, 'flock', lambda *args: pytest.fail('CPU entry retained dispatch lock'))
    monkeypatch.setattr(sys, 'argv', ['adoption_ops.py', '--profile', 'boltz30k', '--launcher', str(p['launcher']),
        '--launcher-manifest', 'a' * 64, '--source', str(p['source']), '--root', str(p['root']), '--manifest', p['manifest'],
        '--control', str(p['control']), '--phase', 'cpu', '--parent', str(parent)])
    ops.main()
    assert gates == [(parent, 'metadata-cpu-entry', {'synthetic_validation_scope': True})]


def test_mixed_terminal_originals_and_unfinished_protenix_depend_only_on_actual_pending_callbacks(campaign, monkeypatch):
    expected_pending = []
    for arm in campaign['protenix50k40k']['arms']:
        p = campaign['protenix50k40k']; label = contract.callback_name('protenix50k40k', arm)
        job = contract.read(p['root'] / label / 'submission.json')['job_id']
        write(p['root'] / label / ('stdout.' + job + '.log'), 'Actual callback still running; no terminal footer.\n')
        expected_pending.append(job)
    state = contract.pending_dependencies(('boltz30k', 'dbmol10k', 'protenix50k40k'))
    assert state['status'] == 'actual_callbacks_known' and state['all_callbacks_terminal'] is False
    assert len(state['actual_callback_ids']) == 5 and state['pending_callback_ids'] == expected_pending
    assert len(state['terminal_callbacks']) == 3
    assert {row['profile'] for row in state['terminal_callbacks']} == {'boltz30k', 'dbmol10k'}
    assert all(row['status'] == 'Done' and contract.file_record(row['proof']['path']) == row['proof']
               for row in state['terminal_callbacks'])
    expected = '&&'.join('ended(' + job + ')' for job in expected_pending)
    assert state['dependency'] == expected
    terminal_ids = {row['job_id'] for row in state['terminal_callbacks']}
    assert not terminal_ids.intersection(expected_pending)
    parent = campaign['boltz30k']['control'] / 'after-cpu'; parent.mkdir(); queued = []
    monkeypatch.setattr(ops, 'submit', lambda *args: queued.append(args) or {'outcome': 'submitted', 'job_id': '303'})
    result = ops.maybe_queue_observer('boltz30k', parent, 'a' * 64)
    assert result['job_id'] == '303' and len(queued) == 1 and queued[0][-1] == expected
    assert contract.read(parent / 'peer-observation.json')['terminal_callbacks'] == state['terminal_callbacks']


def test_all_terminal_authenticated_callbacks_require_no_live_dependency_or_observer(campaign, monkeypatch):
    parent = campaign['boltz30k']['control'] / 'after-cpu'; parent.mkdir()
    monkeypatch.setattr(ops, 'submit', lambda *args: pytest.fail('Persisted terminal callbacks need no live scheduler observer'))
    state = ops.maybe_queue_observer('boltz30k', parent, 'a' * 64)
    assert state['status'] == 'actual_callbacks_known' and state['all_callbacks_terminal'] is True
    assert len(state['actual_callback_ids']) == len(state['terminal_callbacks']) == 5
    assert state['pending_callback_ids'] == [] and state['dependency'] is None
    assert all(row['status'] == 'Done' and contract.file_record(row['proof']['path']) == row['proof']
               for row in state['terminal_callbacks'])
    assert contract.read(parent / 'peer-observation.json') == state
    assert contract.read(campaign['boltz30k']['control'] / 'jobs.json') == {}
