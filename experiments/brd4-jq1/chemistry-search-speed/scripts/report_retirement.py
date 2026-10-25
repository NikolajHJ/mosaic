"""Exact inactive report-copy retirement; standard-library byte/metadata only.

Nothing runs on import. The explicit --execute entrypoint first revalidates all
three complete laptop archives and their 78 members, then makes one SSH attempt.
The remote program refuses an existing operation root and never removes a
directory, source, original scientific file, changed file or linked alias.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import stat
import subprocess
import sys
import tarfile
import time


VERSION = 'mosaic-derived-report-copy-retirement-v1'
PLAN_SHA = '9b0781ad7f693aea7484d2f8afaa48f74d9b964cac5e321e23f2e37cd1edce66'
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
REMOTE_ROOT = BASE + '/chemistry-report-retirement-01'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
MAX_SECONDS = 240
MAX_AUDIT_BYTES = 32 * 1024**2
SOURCES = (
    ('protenix-extended-release-03/source', 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',
     '20261024-protenix-extended-03/source01/source_manifest.json'),
    ('boltz-lr0025-release-03/source', 'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6',
     '20261024-boltz-lr0025-03/source01/source_manifest.json'),
    ('nonchem-stride20-startup-release-03/source', '704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab',
     '20261025-nonchem-stride20-startup-03/source02/source_manifest.json'),
    ('nonchem-endpoint-release-02/source', '88372f92b6349e6c34c542fb84fb39482c4c340568431d1499ebc166a1e1e7fa',
     '20261025-continuation-worker-fix-02/source01/source_manifest.json'),
    ('protenix-endpoint-fix-release-02/source', '09f759a7befb8a7b8f0e1d203e15acf939126fba493b30e6a2bf9cd2eea2216b',
     '20261025-protenix-endpoint-fix-02/source01/source_manifest.json'),
    ('chemistry-full-model-release-02/source', '05e316a7b1bcceaf286b03231200a2d1f66e3f6a7ce927d4c1e73f08a24f86f9',
     '20261025-chemistry-full-model-02/source01/source_manifest.json'),
    ('chemistry-full-model-release-03/source', '8480a39c05d612d1a80ea1892d816c9429fcd471e97adb31655f8132d9007f31',
     '20261025-chemistry-full-model-03/source01/source_manifest.json'),
    ('nonchem-live-report-release-04/source', '856f788c5959c6fe5239374d3141808ec011907f4e092deb78a65f28ef9b196d',
     '20261025-jq1-live-report-01/source04/source_manifest.json'),
    ('chemistry-production-release-01/source', '54c74c1552e98af2d185d0f07c84f82adffa2246a6b1ec406f17e79590eff458',
     '20261025-chemistry-production-01/source01/source_manifest.json'),
)
REPORTS = (
    ('02', '0001', '29812097', 'observation-07'),
    ('03', '0002', '29812120', 'observation-11'),
    ('04', '0003', '29813792', 'observation-refined04-05'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def write(path, value):
    raw = encoded(value)
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return raw


def validate_plan(raw):
    require(hashlib.sha256(raw).hexdigest() == PLAN_SHA, 'Exact retirement plan bytes changed')
    plan = json.loads(raw)
    require(plan.get('version') == VERSION and plan.get('file_count') == 81
            and plan.get('report_export_count') == 78 and plan.get('archive_duplicate_count') == 3
            and plan.get('maximum_logical_candidate_bytes') == 571264343,
            'Exact reviewed report-copy scope changed')
    expected_roots = {BASE + '/nonchem-live-report-' + generation for generation, *_ in REPORTS}
    rows = plan['files']; paths = [row['remote_path'] for row in rows]
    require(len(rows) == len(set(paths)) == 81, 'Candidate path scope differs or duplicates exist')
    for row in rows:
        path = PurePosixPath(row['remote_path'])
        require(path.is_absolute() and '..' not in path.parts and '\\' not in str(path)
                and str(path) == row['remote_path'] and type(row['bytes']) is int and row['bytes'] > 0
                and re.fullmatch('[a-f0-9]{64}', row['sha256']) is not None,
                'Unsafe candidate path/length/hash')
        if row['kind'] == 'derived_report_export_copy':
            require(str(path.parent.parent.parent) in expected_roots
                    and path.parent.name == 'viewer' and path.parent.parent.name == 'render'
                    and row['archive_member'] == path.name, 'Only exact derived viewer members are allowed')
        else:
            require(row['kind'] == 'duplicate_completed_report_archive'
                    and str(path.parent) in expected_roots and row['archive_member'] is None
                    and re.fullmatch('edition-000[123]\\.tar\\.gz', path.name) is not None,
                    'Only the three declared completed report archives are allowed')
    return plan


def local_preflight(repo, raw):
    """Rehash complete local containers, every member and every extracted copy."""
    plan = validate_plan(raw)
    require(Path(repo).resolve().name == 'mosaic', 'This operation belongs only to Mosaic')
    checked = []
    for backup in plan['backups']:
        archive = Path(backup['archive']['path'])
        require(archive.is_file() and not archive.is_symlink()
                and archive.stat().st_size == backup['archive']['bytes']
                and sha(archive) == backup['archive']['sha256'], 'Complete local backup changed')
        expected = {row['member']: row for row in backup['members']}
        require(len(expected) == 26 and backup['archive_member_prefix'] == '', 'Unexpected backup membership')
        with tarfile.open(archive, 'r:gz') as source:
            members = source.getmembers()
            require(len(members) == 26 and {m.name for m in members} == set(expected), 'Archive member scope differs')
            for member in members:
                part = PurePosixPath(member.name)
                require(member.isfile() and len(part.parts) == 1 and not part.is_absolute()
                        and '..' not in part.parts, 'Unsafe/linked archive member')
                digest = hashlib.sha256(); count = 0
                with source.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024**2), b''):
                        count += len(block); digest.update(block)
                row = expected[member.name]
                require(count == member.size == row['bytes'] and digest.hexdigest() == row['sha256'],
                        'Local archive member differs')
                originals = [r for r in plan['files'] if r['remote_path'] == row['remote_path']]
                require(len(originals) == 1, 'Ambiguous candidate/member mapping')
                copy = Path(originals[0]['local_verified_extracted_copy'])
                require(copy.is_file() and not copy.is_symlink() and copy.stat().st_size == count
                        and sha(copy) == digest.hexdigest(), 'Verified extracted backup copy changed')
            while source.fileobj.read(1024**2):
                pass
        checked.append({'path': str(archive), 'bytes': archive.stat().st_size,
                        'sha256': sha(archive), 'regular_members_verified': 26})
    sources = []
    for relative, pin, local in SOURCES:
        path = Path(repo) / 'results/hpc' / local
        require(sha(path) == pin, 'Local protected source-manifest pin changed: ' + relative)
        manifest = json.loads(path.read_bytes())
        require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Empty protected source manifest')
        sources.append({'path': BASE + '/' + relative, 'manifest_sha256': pin,
                        'files': len(manifest['files'])})
    reports = []
    for generation, edition, job, observation in REPORTS:
        local = Path(repo) / 'results/hpc/20261025-jq1-live-report-01' / observation / 'files/render'
        log = local / ('stdout.' + job + '.log')
        sub = local / 'submission.json'
        require(log.is_file() and sub.is_file(), 'Original terminal producer proofs missing locally')
        reports.append({'root': BASE + '/nonchem-live-report-' + generation, 'job_id': job,
                        'job_name': 'mosaic-live-report' + generation + '-render',
                        'stdout': {'path': BASE + '/nonchem-live-report-' + generation + '/render/' + log.name,
                                   'bytes': log.stat().st_size, 'sha256': sha(log)},
                        'submission': {'path': BASE + '/nonchem-live-report-' + generation + '/render/submission.json',
                                       'bytes': sub.stat().st_size, 'sha256': sha(sub)}})
    return plan, {'status': 'all_complete_laptop_archives_and_78_members_rehashed',
                  'archives': checked, 'protected_sources': sources, 'reports': reports}


def remote_main(request):
    """One bounded remote metadata operation. No model/array imports or jobs."""
    import pwd  # POSIX account evidence is used only inside the remote program.

    started = time.monotonic(); deadline = started + MAX_SECONDS
    root = Path(REMOTE_ROOT)
    require(not os.path.lexists(root), 'Existing/unknown retirement operation prohibits replay')
    require(request['plan_sha256'] == PLAN_SHA and request['version'] == VERSION, 'Wrong retirement request')
    plan_raw = base64.b64decode(request['plan_base64'], validate=True); plan = validate_plan(plan_raw)
    uid = os.getuid(); user = pwd.getpwuid(uid).pw_name
    require(user == 's234818' and root.parent == Path(BASE), 'Wrong authorized account/root')
    candidates = {row['remote_path'] for row in plan['files']}
    expected_sources = {(BASE + '/' + relative, pin) for relative, pin, _ in SOURCES}
    require({(r['path'], r['manifest_sha256']) for r in request['protected_sources']} == expected_sources,
            'Protected nine-source identity differs')
    blocked_tags = ['nonchem-live-report-' + row[0] for row in REPORTS]
    blocked_tags += ['nonchem-live-report-release-' + row[0] for row in REPORTS]
    blocked_tags += ['mosaic-live-report' + row[0] for row in REPORTS]
    counter = 0; removed = []; retained = []
    completion = {'version': VERSION, 'status': 'failed', 'plan_sha256': PLAN_SHA,
                  'original_scientific_results_changed': False, 'directories_removed': 0,
                  'full_production_capacity_resolved': False, 'remote_root': str(root)}

    def budget():
        require(time.monotonic() < deadline, 'Bounded report retirement deadline exhausted')

    def alarm_deadline(_signum, _frame):
        raise TimeoutError('Bounded report retirement deadline exhausted')

    def ordinary_parents(path):
        path = Path(path)
        require(path.is_absolute() and '..' not in path.parts, 'Noncanonical absolute path')
        for parent in [path.parent, *path.parent.parents]:
            info = parent.lstat()
            require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), 'Linked/special candidate parent')
            if parent.is_relative_to(Path('/work3/s234818/mosaic')):
                require(info.st_uid == uid, 'Candidate parent belongs to another account')

    def identity(info):
        return {key: getattr(info, key) for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns',
                'st_ctime_ns', 'st_mode', 'st_nlink', 'st_uid')}

    def fd_digest(descriptor):
        digest = hashlib.sha256(); os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            budget(); block = os.read(descriptor, 1024**2)
            if not block: break
            digest.update(block)
        return digest.hexdigest()

    def exact_binding(row):
        path = Path(row['path']); ordinary_parents(path)
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == uid and not path.is_symlink()
                and info.st_size == row['bytes'] and sha(path) == row['sha256'], 'Protected original proof changed')
        return row

    def command(args):
        budget()
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=max(1, min(30, int(deadline-time.monotonic()))))
        return {'args': args, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}

    def activity():
        nonlocal counter
        queue = command(['bjobs', '-u', user, '-l'])
        require(queue['returncode'] == 0 or (not queue['stdout'].strip()
                and re.fullmatch(r'No (?:unfinished )?job found\.?', queue['stderr'].strip()) is not None),
                'Actual current-user scheduler inventory failed')
        blocks = re.split(r'(?=^\s*Job\s*<\d+>)', queue['stdout'], flags=re.M)
        for block in blocks:
            compact = re.sub(r'\s+', '', block)
            if not compact: continue
            jobs = re.findall(r'Job<(\d+)>', compact)
            states = re.findall(r'Status<([A-Z]+)>', compact)
            require(len(jobs) == len(states) == 1 and 'User<' + user + '>' in compact,
                    'Ambiguous actual scheduler job inventory')
            if states[0] not in ('DONE', 'EXIT'):
                require(not any(token in compact for token in [*candidates, *blocked_tags]),
                        'An active/unknown job refers to a retirement candidate or producer')
        processes = command(['ps', '-u', str(uid), '-o', 'pid=,args='])
        require(processes['returncode'] == 0 and processes['stdout'].strip(), 'Actual process inventory failed')
        for line in processes['stdout'].splitlines():
            match = re.fullmatch(r'\s*(\d+)\s+(.+)', line)
            require(match is not None, 'Malformed current-user process row')
            require(not any(token in match.group(2) for token in [*candidates, *blocked_tags]),
                    'A current process refers to a candidate or old report producer')
        counter += 1
        record = {'index': counter, 'captured_utc': datetime.now(timezone.utc).isoformat(),
                  'queue': queue, 'processes': processes}
        write(root / ('activity-' + str(counter).zfill(4) + '.json'), record)
        return counter

    def verify_sources():
        rows = []
        for row in request['protected_sources']:
            budget(); source = Path(row['path']); ordinary_parents(source / 'source_manifest.json')
            require(source.is_dir() and not source.is_symlink(), 'Protected source root is not ordinary')
            manifest_path = source / 'source_manifest.json'
            require(sha(manifest_path) == row['manifest_sha256'], 'Protected source manifest changed')
            manifest = json.loads(manifest_path.read_bytes()); files = manifest['files']
            require(len(files) == row['files'], 'Protected source member count changed')
            actual = {path.relative_to(source).as_posix() for path in source.rglob('*')
                      if path.is_file() or path.is_symlink()}
            require(actual == set(files) | {'source_manifest.json'}, 'Protected exact source membership changed')
            count = 0
            for name, pin in files.items():
                budget(); part = PurePosixPath(name)
                require(not part.is_absolute() and '..' not in part.parts and '\\' not in name,
                        'Unsafe protected source member')
                path = source.joinpath(*part.parts); ordinary_parents(path); info = path.lstat()
                require(stat.S_ISREG(info.st_mode) and info.st_uid == uid and not info.st_mode & 0o222
                        and str(path) not in candidates and sha(path) == pin, 'Protected sealed source member changed')
                raw = path.read_bytes()
                require(not any(token.encode() in raw for token in candidates),
                        'A protected current source references an inactive report candidate')
                count += info.st_size
            rows.append({**row, 'all_members_verified': True, 'member_bytes': count})
        return rows

    def metadata_module(source, name):
        path = Path(source) / 'scripts/protenix_precision_ops.py'
        spec = importlib.util.spec_from_file_location(name, path)
        require(spec is not None and spec.loader is not None, 'Authenticated metadata module missing')
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
        return module

    def protected_closure(guard):
        manifest = guard.verify(guard.SOURCE, dict(expected_sources)[str(guard.SOURCE)])
        result = guard.external(manifest, guard.read(guard.SOURCE / 'plan.json'))
        require(result['original_results_changed'] is False, 'Original input/proof closure did not authenticate')
        return result

    def producers():
        rows = []
        for row in request['reports']:
            exact_binding(row['stdout']); exact_binding(row['submission'])
            sub = json.loads(Path(row['submission']['path']).read_bytes())
            require(sub['job_id'] == row['job_id'] and sub['phase'] == 'render'
                    and sub['status'] == 'submitted', 'Original renderer submission differs')
            raw = Path(row['stdout']['path']).read_text(); compact = re.sub(r'\s+', '', raw)
            require(compact.count('Subject:Job') == 1 and 'Subject:Job' + row['job_id'] + ':<' + row['job_name']
                    + '>incluster<dcc>Done' in compact and compact.count('Successfullycompleted.') == 1
                    and 'inqueue<hpc>' in compact and 'Exitedwithexitcode' not in compact,
                    'Original same-job renderer did not complete successfully')
            query = command(['bjobs', '-l', row['job_id']]); text = re.sub(r'\s+', '', query['stdout'])
            states = re.findall(r'Status<([A-Z]+)>', text)
            if states:
                require(states == ['DONE'] and re.findall(r'Job<(\d+)>', text) == [row['job_id']],
                        'Original renderer is not actually terminal')
                state = 'actual_DONE_same_job_LSF_success'
            else:
                require(not query['stdout'].strip() and 'not found' in query['stderr'].lower(),
                        'Original renderer terminal scheduler observation is ambiguous')
                state = 'persisted_same_job_LSF_success_live_record_aged_out'
            rows.append({**row, 'terminal_evidence': state, 'actual_query': query})
        return rows

    def journal(stream, value):
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())

    try:
        ordinary_parents(root / 'plan.json') if root.exists() else ordinary_parents(root)
        root.mkdir(mode=0o700)
        signal.signal(signal.SIGALRM, alarm_deadline)
        signal.alarm(MAX_SECONDS)
        with (root / 'plan.json').open('xb') as stream:
            stream.write(plan_raw); stream.flush(); os.fsync(stream.fileno())
        write(root / 'request.json', {key: value for key, value in request.items() if key != 'plan_base64'})
        source_before = verify_sources(); write(root / 'sources-before.json', source_before)
        guard = metadata_module(BASE + '/protenix-endpoint-fix-release-02/source', '_retirement_authenticated_precision')
        write(root / 'protected-inputs-before.json', protected_closure(guard))
        write(root / 'original-producers.json', producers()); activity()
        with (root / 'journal.jsonl').open('x', encoding='utf8', newline='\n') as log:
            for row in plan['files']:
                budget(); path = Path(row['remote_path']); ordinary_parents(path)
                if not os.path.lexists(path):
                    item = {'path': str(path), 'status': 'retained_already_absent'}
                    retained.append(item); journal(log, item); continue
                initial = path.lstat()
                if not stat.S_ISREG(initial.st_mode) or initial.st_uid != uid or initial.st_nlink != 1:
                    item = {'path': str(path), 'status': 'retained_linked_nonregular_or_other_owner',
                            'identity': identity(initial)}
                    retained.append(item); journal(log, item); continue
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    before = os.fstat(descriptor)
                    if (identity(before) != identity(initial) or before.st_size != row['bytes']
                            or fd_digest(descriptor) != row['sha256'] or identity(os.fstat(descriptor)) != identity(before)):
                        item = {'path': str(path), 'status': 'retained_changed_initial_read'}
                        retained.append(item); journal(log, item); continue
                    activity_index = activity(); ordinary_parents(path)
                    if (identity(os.stat(path.parent)) != identity(os.fstat(parent_fd))
                            or identity(os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)) != identity(before)
                            or fd_digest(descriptor) != row['sha256'] or identity(os.fstat(descriptor)) != identity(before)):
                        item = {'path': str(path), 'status': 'retained_changed_final_read'}
                        retained.append(item); journal(log, item); continue
                    intent = {'status': 'intent_exact_single_file_unlink', 'path': str(path),
                              'bytes': row['bytes'], 'sha256': row['sha256'], 'identity': identity(before),
                              'activity_index': activity_index}
                    journal(log, intent)
                    require(identity(os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)) == identity(before),
                            'Candidate identity raced after durable intent; no unlink')
                    os.unlink(path.name, dir_fd=parent_fd); os.fsync(parent_fd)
                    item = {**intent, 'status': 'removed_exact_backed_up_copy'}
                    removed.append(item); journal(log, item)
                finally:
                    os.close(descriptor); os.close(parent_fd)
        write(root / 'protected-inputs-after.json', protected_closure(guard))
        source_after = verify_sources(); write(root / 'sources-after.json', source_after)
        require(source_after == source_before, 'Protected source identities changed across retirement')
        activity()
        completion.update(status='completed_exact_report_copy_retirement', protected_inputs_unchanged=True)
        post = root / 'post-retirement-admission'; post.mkdir()
        try:
            # Existing authenticated metadata guard, no optimizer/model phase.
            gate = guard.storage(post, 'entry')
            write(root / 'post-retirement-full-guard.json', gate)
            completion['post_retirement_validation_storage_status'] = gate['status']
        except BaseException as error:
            completion['post_retirement_validation_storage_status'] = 'failed_preserved'
            completion['post_retirement_guard_error'] = repr(error)
    except BaseException as error:
        completion['error'] = repr(error)
    finally:
        # Preserve the bounded operation's durable audit even after its deadline.
        signal.alarm(0)
        completion.update(removed_files=len(removed), removed_logical_bytes=sum(row['bytes'] for row in removed),
                          retained_files=retained, activity_checks=counter,
                          elapsed_seconds=time.monotonic()-started, completed_utc=datetime.now(timezone.utc).isoformat())
        if root.exists():
            write(root / 'completion.json', completion)
            files = []; total = 0
            for path in sorted(root.rglob('*')):
                require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Audit contains an alias/special file')
                if not path.is_file(): continue
                raw = path.read_bytes(); total += len(raw)
                require(total < MAX_AUDIT_BYTES, 'Compact retirement audit output cap exceeded')
                files.append({'relative': path.relative_to(root).as_posix(), 'bytes': len(raw),
                              'sha256': hashlib.sha256(raw).hexdigest(), 'base64': base64.b64encode(raw).decode()})
            print(json.dumps({'status': completion['status'], 'remote_root': str(root), 'files': files,
                              'completion': completion, 'scientific_or_production_fit_claim': False}), flush=True)
        else:
            print(json.dumps({'status': 'failed_before_operation_creation', 'completion': completion}), flush=True)
    return 0 if completion['status'] == 'completed_exact_report_copy_retirement' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(args.execute, 'Preparation only: explicit --execute is required after review')
    repo = Path(__file__).resolve().parents[4]
    local = repo / 'results/hpc/20261025-chemistry-report-retirement-01'
    raw = (local / 'plan.json').read_bytes(); plan, verification = local_preflight(repo, raw)
    output = local / 'execution-01'; output.mkdir(exist_ok=False)
    write(output / 'local-backup-verification.json', verification)
    request = {'version': VERSION, 'plan_sha256': PLAN_SHA, 'plan_base64': base64.b64encode(raw).decode(),
               'protected_sources': verification['protected_sources'], 'reports': verification['reports'],
               'all_backup_bytes_reverified_locally': True}
    code = Path(__file__).read_bytes(); ast.parse(code, filename=__file__)
    program = ('_RETIREMENT_REQUEST=' + repr(json.dumps(request, allow_nan=False)) + '\n').encode() + code
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
               'dtu-hpc', 'bash -l -c ' + shlex.quote(PYTHON + ' -')]
    write(output / 'started.json', {'status': 'unknown', 'argv': command, 'plan_sha256': PLAN_SHA,
                                   'program_sha256': hashlib.sha256(program).hexdigest(),
                                   'source_sha256': hashlib.sha256(code).hexdigest()})
    (output / 'program.py').write_bytes(program)
    result = {'status': 'unknown', 'remote_root': REMOTE_ROOT}
    try:
        with (output / 'stdout.json').open('xb') as stdout, (output / 'stderr.txt').open('xb') as stderr:
            process = subprocess.run(command, input=program, stdout=stdout, stderr=stderr, timeout=330)
        result['returncode'] = process.returncode
        payload = json.loads((output / 'stdout.json').read_bytes())
        require(payload.get('remote_root') == REMOTE_ROOT and isinstance(payload.get('files'), list),
                'Missing exact raw retirement audit; preserve unknown outcome')
        extracted = output / 'files'; extracted.mkdir()
        seen = set()
        for row in payload['files']:
            relative = PurePosixPath(row['relative'])
            require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in str(relative)
                    and row['relative'] not in seen, 'Unsafe/duplicate raw audit path')
            seen.add(row['relative']); content = base64.b64decode(row['base64'], validate=True)
            require(len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256'],
                    'Raw retirement audit byte/hash mismatch')
            target = extracted.joinpath(*relative.parts); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(content)
        completion = json.loads((extracted / 'completion.json').read_bytes())
        require((extracted / 'plan.json').read_bytes() == raw and payload['completion'] == completion,
                'Delivered retirement plan/completion differs')
        journal_path = extracted / 'journal.jsonl'
        require(journal_path.exists() or completion['removed_files'] == 0,
                'Removal claimed without a durable journal')
        events = ([json.loads(line) for line in journal_path.read_text().splitlines()]
                  if journal_path.exists() else [])
        allowed = {row['remote_path']: row for row in plan['files']}
        intents = {}; removals = []; outcomes = set()
        for event in events:
            require(event['path'] in allowed, 'Journal includes an unapproved path')
            if event['status'] == 'intent_exact_single_file_unlink':
                require(event['path'] not in intents, 'Duplicate unlink intent')
                intents[event['path']] = event
            elif event['status'] == 'removed_exact_backed_up_copy':
                require(event['path'] in intents and event['path'] not in outcomes,
                        'Missing/duplicate exact unlink intent or result')
                prior = intents[event['path']]; expected = allowed[event['path']]
                require(event['bytes'] == expected['bytes'] and event['sha256'] == expected['sha256']
                        and {k:v for k,v in prior.items() if k != 'status'} == {k:v for k,v in event.items() if k != 'status'},
                        'Removal does not reconcile to an exact backed-up intent')
                removals.append(event)
                outcomes.add(event['path'])
            else:
                require(event['status'].startswith('retained_') and event['path'] not in outcomes,
                        'Unknown/duplicate journal outcome')
                outcomes.add(event['path'])
        require(len(removals) == completion['removed_files']
                and sum(row['bytes'] for row in removals) == completion['removed_logical_bytes'],
                'Actual durable journal removal totals differ')
        if completion['status'] == 'completed_exact_report_copy_retirement':
            require(outcomes == set(allowed) and completion.get('protected_inputs_unchanged') is True
                    and (extracted / 'sources-before.json').read_bytes()
                    == (extracted / 'sources-after.json').read_bytes(),
                    'Complete retirement omitted a candidate or changed protected sources')
        result.update(status='raw_audit_bytes_and_exact_journal_verified', completion=completion,
                      raw_audit_files=len(seen), full_production_capacity_resolved=False)
    finally:
        result['completed_utc'] = datetime.now(timezone.utc).isoformat()
        if (output / 'stdout.json').exists(): result['stdout_sha256'] = sha(output / 'stdout.json')
        write(output / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if '_RETIREMENT_REQUEST' in globals():
        raise SystemExit(remote_main(json.loads(_RETIREMENT_REQUEST)))
    main()
