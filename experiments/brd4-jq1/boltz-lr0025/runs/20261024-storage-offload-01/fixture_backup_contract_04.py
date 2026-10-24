"""Allocated, small real-file archive/remove failure controls; no science."""
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

import pytest

import offload_protocol as protocol
import verify_stream_backup as verifier
import fresh_stream_backup as client

CONTRACT_CASES = 44


def raw_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def original(tmp_path, monkeypatch):
    assert os.environ.get('LSB_JOBID') and os.environ.get('LSB_QUEUE') == 'hpc', 'Allocated CPU fixture required'
    root = tmp_path / 'mosaic'; inactive = root / 'inactive'; cleanup = root / 'cleanup-control'
    inactive.mkdir(parents=True); cleanup.mkdir()
    current = root / 'current'; current.mkdir()
    protected = current / 'current-input.bin'; protected.write_bytes(b'current frozen input must survive\n')
    paths = [inactive / 'state-00000.npz', inactive / 'states.jsonl']
    paths[0].write_bytes(b'opaque original snapshot bytes; no array decoding\n')
    paths[1].write_bytes(b'{"original":0}\n{"original":1}\n')
    protection = cleanup / 'protection.json'
    protection.write_bytes(raw_json({'version': 'mosaic-storage-offload-protection-v1',
        'exact': [str(protected)], 'subtrees': [], 'negative': []}))
    plan = {'version': protocol.PLAN_VERSION, 'status': 'approved_archive_then_remove', 'operation_id': 'fixture-opaque-backup',
        'cleanup_root': str(cleanup), 'approved_candidate_roots': [str(inactive)],
        'protection_manifest': {'path': str(protection), 'sha256': protocol.sha(protection)},
        'audited_active_job_ids': ['101'], 'candidate_owner_job_ids': ['202'], 'timeout_seconds': 30,
        'files': [{'path': str(path), 'member': 'files/' + format(index, '08d'),
                   'sha256': protocol.sha(path), **protocol.metadata(path.stat())} for index, path in enumerate(paths)]}
    monkeypatch.setattr(protocol, 'CLEANUP_ROOT', cleanup)
    monkeypatch.setattr(protocol, 'MOSAIC_ROOTS', (root,))
    monkeypatch.setattr(verifier, 'REMOTE_ROOTS', (PurePosixPath(str(root)),))
    monkeypatch.setattr(verifier, 'CLEANUP_ROOT', PurePosixPath(str(cleanup)))
    raw = raw_json(plan); validated = protocol.validate_plan(raw, sha(raw))
    archive = tmp_path / 'originals.tar.gz'
    with archive.open('xb') as stream:
        counts = protocol.pack_stream(validated, raw, stream)
    completed = {'version': protocol.STREAM_VERSION, 'status': 'streamed_complete',
        'operation_id': plan['operation_id'], 'plan_sha256': sha(raw), 'plan_bytes': len(raw), **counts}
    verified = verifier.verify_archive(archive, raw, sha(raw), completed)
    assert protocol.verify_delete_authority(plan, raw, completed, verified)
    return {'root': root, 'paths': paths, 'protected': protected, 'plan': plan, 'raw': raw,
            'archive': archive, 'completed': completed, 'verified': verified, 'tmp': tmp_path}


def observation(ids=('101',)):
    return {'jobs': [{'job_id': value, 'status': 'RUN'} for value in ids], 'raw': 'synthetic fixture scheduler metadata only'}


def test_real_regular_files_roundtrip_and_exact_original_manifest_without_decoding(original):
    value = original
    assert value['verified']['status'] == 'verified_all_members'
    assert value['verified']['all_members_verified'] is True and value['verified']['files_count'] == 2
    with gzip.open(value['archive'], 'rb') as stream, tarfile.open(fileobj=stream, mode='r|') as archive:
        first = archive.next()
        assert first.name == verifier.MANIFEST_MEMBER and archive.extractfile(first).read() == value['raw']
    assert all(path.exists() for path in value['paths']) and value['protected'].read_bytes().startswith(b'current frozen')
    assert verifier.permitted_candidate_root(PurePosixPath('/work3/s234818/latentmol/runs/20261023-legacy/output'))
    assert verifier.permitted_candidate_root(PurePosixPath('/work3/s234818/latentmol/runs/source120a004-boltz-defog-staged-24h-attempt004'))
    assert verifier.permitted_candidate_root(PurePosixPath('/work3/s234818/latentmol/runs/source-073-decoder-budget-attempt002-recovery'))
    for name in ('/work3/s234818/latentmol/runs', '/work3/s234818/latentmol/source',
                 '/work3/s234818/latentmol/runs/old', '/work3/s234818/latentmol/runs/20260231-legacy',
                 '/work3/s234818/latentmol/runs/source135'):
        assert not verifier.permitted_candidate_root(PurePosixPath(name))


def malicious_archive(value, fault):
    plan = copy.deepcopy(value['plan']); raw = value['raw']
    entries = [(verifier.MANIFEST_MEMBER, raw, None)]
    entries += [(row['member'], Path(row['path']).read_bytes(), None) for row in plan['files']]
    if fault == 'missing': entries.pop()
    elif fault == 'extra': entries.append(('files/99999999', b'undeclared', None))
    elif fault == 'duplicate': entries.append(entries[1])
    elif fault == 'escape': entries[1] = ('../escaped-original', entries[1][1], None)
    elif fault == 'symlink': entries[1] = (entries[1][0], b'', 'protected-original')
    elif fault == 'changed_bytes': entries[1] = (entries[1][0], b'x' * len(entries[1][1]), None)
    elif fault == 'missing_sha':
        del plan['files'][0]['sha256']; raw = raw_json(plan); entries[0] = (verifier.MANIFEST_MEMBER, raw, None)
    elif fault == 'manifest': entries[0] = (verifier.MANIFEST_MEMBER, raw + b' ', None)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, data, link in entries:
            member = tarfile.TarInfo(name)
            if link is not None: member.type, member.linkname = tarfile.SYMTYPE, link
            else: member.size = len(data)
            archive.addfile(member, None if link is not None else io.BytesIO(data))
    data = gzip.compress(buffer.getvalue(), mtime=0)
    if fault == 'hidden_append':
        hidden = io.BytesIO()
        with tarfile.open(fileobj=hidden, mode='w') as archive:
            member = tarfile.TarInfo('hidden-unapproved-member'); member.size = 4
            archive.addfile(member, io.BytesIO(b'evil'))
        data += gzip.compress(hidden.getvalue(), mtime=0)
    path = value['tmp'] / ('bad-' + fault + '.tar.gz'); path.write_bytes(data)
    receipt = copy.deepcopy(value['completed'])
    receipt.update(plan_sha256=sha(raw), plan_bytes=len(raw), archive_sha256=sha(data), archive_bytes=len(data))
    return path, raw, receipt


@pytest.mark.parametrize('fault', ('missing', 'extra', 'duplicate', 'escape', 'symlink', 'changed_bytes', 'missing_sha', 'manifest', 'hidden_append'))
def test_no_backup_verification_for_incomplete_unsafe_or_unbound_members(original, fault):
    path, raw, receipt = malicious_archive(original, fault)
    with pytest.raises((ValueError, KeyError, tarfile.TarError, EOFError)):
        verifier.verify_archive(path, raw, sha(raw), receipt)
    assert all(path.exists() for path in original['paths']) and original['protected'].exists()


def test_exact_deletion_leaves_verified_laptop_backup_and_current_protected_file(original):
    value = original; journal = value['tmp'] / 'delete-journal.jsonl'
    with journal.open('x') as stream:
        result = protocol.delete_exact(value['plan'], value['raw'], value['completed'], value['verified'], stream,
                                       observe_active=observation)
    assert result['removed_files'] == 2 and result['removed_logical_bytes'] == sum(row['bytes'] for row in value['plan']['files'])
    assert all(not path.exists() for path in value['paths'])
    assert value['protected'].read_bytes() == b'current frozen input must survive\n'
    assert verifier.verify_archive(value['archive'], value['raw'], sha(value['raw']), value['completed'])['all_members_verified']
    protections = json.loads((value['root'] / 'cleanup-control/protection.json').read_bytes())
    assert protocol.is_protected(value['protected'], protections)
    unsafe = copy.deepcopy(value['plan']); path = value['protected']
    unsafe['approved_candidate_roots'] = [str(value['protected'].parent)]
    unsafe['files'][0].update(path=str(path), sha256=protocol.sha(path), **protocol.metadata(path.stat()))
    raw = raw_json(unsafe)
    with pytest.raises(ValueError): protocol.validate_plan(raw, sha(raw))
    assert value['protected'].exists()


@pytest.mark.parametrize('fault', ('status', 'archive_sha', 'files'))
def test_deletion_refuses_missing_or_different_full_laptop_verification(original, fault):
    value = original; verified = copy.deepcopy(value['verified'])
    if fault == 'status': verified['status'] = 'stream_received_unverified'
    elif fault == 'archive_sha': verified['archive_sha256'] = 'a' * 64
    else: verified['files'] = verified['files'][:-1]
    with (value['tmp'] / 'rejected-journal.jsonl').open('x') as stream:
        with pytest.raises(ValueError):
            protocol.delete_exact(value['plan'], value['raw'], value['completed'], verified, stream, observe_active=observation)
    assert all(path.exists() for path in value['paths']) and value['protected'].exists()


def test_new_active_job_blocks_before_any_unlink(original):
    value = original
    with (value['tmp'] / 'active-journal.jsonl').open('x') as stream:
        with pytest.raises(ValueError):
            protocol.delete_exact(value['plan'], value['raw'], value['completed'], value['verified'], stream,
                                  observe_active=lambda: observation(('101', '999')))
    assert all(path.exists() for path in value['paths']) and value['protected'].exists()


def test_changed_file_is_retained_and_never_counted_as_removed(original):
    value = original; changed = value['paths'][1]; changed.write_bytes(b'new saved bytes after backup\n')
    with (value['tmp'] / 'race-journal.jsonl').open('x') as stream:
        result = protocol.delete_exact(value['plan'], value['raw'], value['completed'], value['verified'], stream,
                                       observe_active=observation)
    assert result['retained_changed_or_unreadable'] == 1 and result['removed_files'] == 1
    assert changed.read_bytes() == b'new saved bytes after backup\n' and value['protected'].exists()
    assert not value['paths'][0].exists()


def test_replaced_symlink_cannot_delete_its_protected_target(original):
    value = original; path = value['paths'][0]; path.unlink(); path.symlink_to(value['protected'])
    with (value['tmp'] / 'symlink-journal.jsonl').open('x') as stream:
        result = protocol.delete_exact(value['plan'], value['raw'], value['completed'], value['verified'], stream,
                                       observe_active=observation)
    assert result['retained_changed_or_unreadable'] == 1 and path.is_symlink()
    assert value['protected'].read_bytes() == b'current frozen input must survive\n'


@pytest.mark.parametrize('returncode', (0, 1))
def test_fresh_client_never_claims_member_verification_and_refuses_any_replay(original, monkeypatch, returncode):
    value = original; folder = value['tmp'] / 'laptop'; folder.mkdir(); plan_path = folder / 'approved-plan.json'; plan_path.write_bytes(value['raw'])
    monkeypatch.setattr(client, 'BACKUP_ROOT', folder)
    payload = value['archive'].read_bytes() if returncode == 0 else b'partial unknown gzip bytes'
    calls = []

    class OwnedSSH:
        def __init__(self, args, **kwargs):
            calls.append(args); self.stdin = io.BytesIO(); self.stdout = io.BytesIO(payload)
            self.returncode = returncode
            assert args[0] == 'ssh' and '-T' in args and 'dtu-hpc' in args
        def poll(self): return self.returncode
        def wait(self, timeout): return self.returncode
        def kill(self): pytest.fail('Already terminal fixture SSH child must not be killed')

    monkeypatch.setattr(client.subprocess, 'Popen', OwnedSSH)
    output = folder / 'fresh.tar.gz'
    kwargs = {'remote_python': '/work3/s234818/mosaic/envs/protenix-tiny/bin/python',
              'packer': '/work3/s234818/mosaic/runs/cleanup/stream_backup.py',
              'receipt': str(Path(value['plan']['cleanup_root']) / value['plan']['operation_id'] / 'stream-complete.json')}
    if returncode == 0:
        result = client.stream_once(plan_path, sha(value['raw']), output, **kwargs)
        assert result['status'] == 'stream_received_unverified'
    else:
        with pytest.raises(RuntimeError): client.stream_once(plan_path, sha(value['raw']), output, **kwargs)
        result = json.loads(output.with_name(output.name + '.transfer.json').read_bytes())
        assert result['status'] == 'failed_or_unknown_partial_preserved'
    assert output.read_bytes() == payload and result['all_files_verified'] is False and result['deletion_permitted'] is False
    with pytest.raises(ValueError): client.stream_once(plan_path, sha(value['raw']), output, **kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize('name', (
    '20260927T134617Z_ae-train_dc66226d', '20260927T000000Z-run/output',
    '20260927T235959Z_legacy', '20260928T010203Z-run'))
def test_real_legacy_timestamp_run_names_are_accepted_without_broadening_anchors(name):
    path = '/work3/s234818/latentmol/runs/' + name
    assert protocol.permitted_candidate_root(Path(path))
    assert verifier.permitted_candidate_root(PurePosixPath(path))


@pytest.mark.parametrize('name', (
    '/work3/s234818/latentmol/runs',
    '/work3/s234818/latentmol/runs/20260927T246000Z_legacy',
    '/work3/s234818/latentmol/runs/20260927T126060Z_legacy',
    '/work3/s234818/latentmol/runs/20260231T134617Z_legacy',
    '/work3/s234818/latentmol/runs/20261301T134617Z_legacy',
    '/work3/s234818/latentmol/runs/20260927T13461Z_legacy',
    '/work3/s234818/latentmol/runs/20260927T134617Zevil'))
def test_legacy_timestamp_validation_rejects_malformed_dates_times_and_anchors(name):
    assert not protocol.permitted_candidate_root(Path(name))
    assert not verifier.permitted_candidate_root(PurePosixPath(name))


def test_approved_ancestor_never_makes_cleanup_controls_removable(original, monkeypatch):
    value = original
    unsafe = copy.deepcopy(value['plan']); control = value['root'] / 'cleanup-control/protection.json'
    unsafe['approved_candidate_roots'] = [str(value['root'])]
    unsafe['files'][0].update(path=str(control), sha256=protocol.sha(control), **protocol.metadata(control.stat()))
    raw = raw_json(unsafe)
    with monkeypatch.context() as patch:
        patch.setattr(protocol, 'MOSAIC_ROOTS', (value['tmp'],))
        patch.setattr(verifier, 'REMOTE_ROOTS', (PurePosixPath(str(value['tmp'])),))
        with pytest.raises(ValueError): protocol.validate_plan(raw, sha(raw))
        with pytest.raises(ValueError): verifier.load_plan(raw, sha(raw))
    assert control.exists()
    assert all(path.exists() for path in value['paths'])


def test_allocated_live_scheduler_command_accepts_actual_current_job():
    """Exercise DTU's real CLI; a query error is a failure, never a fallback."""
    own_job = os.environ.get('LSB_JOBID', '')
    assert own_job.isdigit() and os.environ.get('LSB_QUEUE') == 'hpc', 'Allocated CPU fixture required'
    observation = protocol.active_jobs()
    assert observation['query'] == ['bjobs', '-noheader', '-o', 'jobid stat']
    own_rows = [row for row in observation['jobs'] if row['job_id'] == own_job]
    assert own_rows == [{'job_id': own_job, 'status': 'RUN'}]
    assert isinstance(observation['raw'], str) and observation['raw'].strip()
    assert type(observation['utc_unix_ns']) is int and observation['utc_unix_ns'] > 0


@pytest.mark.parametrize('scenario', (
    'descendant', 'root_equal', 'approved_prefix_collision', 'protected_exact',
    'protected_subtree_equal', 'protected_subtree_descendant',
    'protected_prefix_collision', 'negative_pattern'))
def test_compiled_canonical_membership_matches_original_path_and_protection_semantics(scenario):
    anchor = Path('/work3/s234818/latentmol/runs/20261001-inactive')
    roots = [anchor]
    path = anchor / 'payload/state.npz'
    protections = {'exact': [], 'subtrees': [], 'negative': []}
    if scenario == 'root_equal': path = anchor
    elif scenario == 'approved_prefix_collision': path = Path(str(anchor) + '-other/state.npz')
    elif scenario == 'protected_exact': protections['exact'] = [str(path), str(path)]
    elif scenario == 'protected_subtree_equal': protections['subtrees'] = [str(path)]
    elif scenario == 'protected_subtree_descendant': protections['subtrees'] = [str(anchor / 'payload')]
    elif scenario == 'protected_prefix_collision': protections['subtrees'] = [str(anchor / 'pay')]
    elif scenario == 'negative_pattern': protections['negative'] = ['*/payload/*.npz']
    expected = (any(protocol.below(path, root) for root in roots), protocol.is_protected(path, protections))
    assert protocol.compile_plan_membership(roots, protections)(path) == expected
    if scenario in ('root_equal', 'approved_prefix_collision'): assert expected[0] is False
    if scenario in ('protected_exact', 'protected_subtree_equal', 'protected_subtree_descendant', 'negative_pattern'):
        assert expected[1] is True
    if scenario == 'protected_prefix_collision': assert expected[1] is False


@pytest.mark.parametrize('fault', ('noncanonical_root', 'noncanonical_file'))
def test_lookup_never_bypasses_preexisting_canonical_plan_path_rejection(original, fault):
    value = original; plan = copy.deepcopy(value['plan'])
    if fault == 'noncanonical_root': plan['approved_candidate_roots'][0] += '/../inactive'
    else: plan['files'][0]['path'] = plan['files'][0]['path'].replace('/inactive/', '/inactive//')
    raw = raw_json(plan)
    with pytest.raises(ValueError): protocol.validate_plan(raw, sha(raw))
    assert all(path.exists() for path in value['paths']) and value['protected'].exists()


def test_each_protected_literal_hardlink_alias_still_blocks_the_whole_plan(original):
    value = original; first, second = value['paths']
    second.unlink(); os.link(first, second)
    protection_path = value['root'] / 'cleanup-control/protection.json'
    protections = json.loads(protection_path.read_bytes()); protections['exact'].append(str(second))
    protection_path.write_bytes(raw_json(protections))
    plan = copy.deepcopy(value['plan'])
    plan['protection_manifest']['sha256'] = protocol.sha(protection_path)
    for row, path in zip(plan['files'], (first, second)):
        row.update(sha256=protocol.sha(path), **protocol.metadata(path.stat()))
    assert first.stat().st_ino == second.stat().st_ino and first.stat().st_nlink == 2
    compiled = protocol.compile_plan_membership([first.parent], protections)
    assert compiled(first)[1] == protocol.is_protected(first, protections) is False
    assert compiled(second)[1] == protocol.is_protected(second, protections) is True
    raw = raw_json(plan)
    with pytest.raises(ValueError): protocol.validate_plan(raw, sha(raw))
    assert first.exists() and second.exists() and value['protected'].exists()


def test_declared_fast_gzip_level_real_stream_and_independent_all_member_roundtrip(original):
    """Real pack_stream output uses level1 while preserving every original byte."""
    value = original
    header = value['archive'].read_bytes()[:10]
    assert header[:3] == b'\x1f\x8b\x08' and header[8] == 4, 'gzip XFL must declare fastest compression'
    checked = verifier.verify_archive(value['archive'], value['raw'], sha(value['raw']), value['completed'])
    assert checked['status'] == 'verified_all_members' and checked['all_members_verified'] is True
    assert checked['archive_sha256'] == protocol.sha(value['archive'])
    assert checked['files'] == [{key: row[key] for key in ('path', 'member', 'bytes', 'sha256')}
                                for row in value['plan']['files']]
    with gzip.open(value['archive'], 'rb') as stream, tarfile.open(fileobj=stream, mode='r|') as archive:
        members = list(archive)
    assert len(members) == 3 and all(member.isfile() for member in members)
    assert all(path.exists() for path in value['paths']) and value['protected'].exists()
