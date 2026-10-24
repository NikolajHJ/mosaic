"""Unexecuted read-only post-offload collector. Root owns invocation and pins.

No file is created, removed, restored or deserialized as a model/array. Output
is one JSON document containing exact control bytes and fresh observations.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time


ROOT = Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
MAX_CONTROL_BYTES = 128 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def canonical(value):
    require(isinstance(value, str) and value.startswith('/') and '\\' not in value
            and not any(c in value for c in '\x00\r\n*?[]'), 'Noncanonical exact path')
    p = PurePosixPath(value)
    require(str(p) == value and '..' not in p.parts, 'Noncanonical exact path')
    return Path(value)


def nofollow_parent(path):
    path = canonical(str(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open('/', flags)
    try:
        for part in path.parts[1:-1]:
            new = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd, path.name
    except BaseException:
        os.close(fd)
        raise


def metadata(row):
    return dict(device=row.st_dev, inode=row.st_ino, bytes=row.st_size,
                mtime_ns=row.st_mtime_ns, mode=stat.S_IMODE(row.st_mode), nlink=row.st_nlink)


def read_exact(path, *, expected_sha=None, maximum=MAX_CONTROL_BYTES, keep_bytes=True):
    parent, name = nofollow_parent(path)
    try:
        entry = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISREG(entry.st_mode) and entry.st_size <= maximum, 'Nonregular/oversized proof')
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            require(metadata(before) == metadata(entry), 'Proof changed before read')
            digest, chunks, count = hashlib.sha256(), [], 0
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                count += len(block)
                require(count <= maximum, 'Proof grew beyond the bounded audit read')
                digest.update(block)
                if keep_bytes:
                    chunks.append(block)
            after = os.fstat(stream.fileno())
        literal = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(count == before.st_size and metadata(before) == metadata(after) == metadata(literal)
                and before.st_ctime_ns == after.st_ctime_ns == literal.st_ctime_ns,
                'Proof changed during read')
        value = digest.hexdigest()
        require(expected_sha is None or value == expected_sha, 'Expected proof SHA changed: ' + str(path))
        record = dict(path=str(path), sha256=value, **metadata(after))
        return (b''.join(chunks) if keep_bytes else None), record
    finally:
        os.close(parent)


def protected(path, value):
    p = PurePosixPath(str(path))
    return str(path) in value['exact'] or any(p == PurePosixPath(x) or p.is_relative_to(x)
                                            for x in value['subtrees'])


def query(argv, *, timeout=60):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return dict(args=argv, returncode=result.returncode, stdout=result.stdout,
                stderr=result.stderr, captured_unix_ns=time.time_ns())


def audit(args, result):
    raw, plan_proof = read_exact(args.plan, expected_sha=args.plan_sha256)
    plan = strict_json(raw)
    require(plan['version'] == 'mosaic-storage-offload-plan-v1'
            and plan['status'] == 'approved_archive_then_remove' and plan['cleanup_root'] == str(ROOT),
            'Wrong approved offload plan')
    operation = plan['operation_id']
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', operation), 'Wrong operation identity')
    directory = ROOT / operation
    require(args.plan == directory / 'approved-plan.json', 'Plan outside exact operation')
    protection_raw, protection_proof = read_exact(canonical(plan['protection_manifest']['path']),
        expected_sha=plan['protection_manifest']['sha256'])
    protection = strict_json(protection_raw)
    require(protection['version'] == 'mosaic-storage-offload-protection-v1', 'Wrong protection version')
    pin_raw, pin_proof = read_exact(args.pin_spec, expected_sha=args.pin_spec_sha256)
    pin_spec = strict_json(pin_raw)
    require(pin_spec['version'] == 'mosaic-storage-offload-post-audit-pins-v1', 'Wrong explicit audit pin spec')
    result.update(plan=plan_proof, protection=protection_proof, pin_spec=pin_proof, operation_id=operation,
                  controls={})
    controls = {}
    for name in ('stream-complete.json', 'local-verification.json', 'deletion-started.json',
                 'deletion-complete.json', 'deletion-journal.jsonl'):
        expected = (args.stream_receipt_sha256 if name == 'stream-complete.json' else
                    args.local_verification_sha256 if name == 'local-verification.json' else None)
        value, proof = read_exact(directory / name, expected_sha=expected)
        result['controls'][name] = dict(proof, base64=base64.b64encode(value).decode('ascii'))
        controls[name] = value if name.endswith('.jsonl') else strict_json(value)
    complete = controls['stream-complete.json']; verified = controls['local-verification.json']
    removed = controls['deletion-complete.json']; started = controls['deletion-started.json']
    require(complete['status'] == 'streamed_complete' and complete['all_files_verified'] is True
            and verified['status'] == 'verified_all_members' and verified['all_members_verified'] is True,
            'No complete stream and independent full local member verification')
    for value in (complete, verified, removed, started):
        require(value['operation_id'] == operation and value['plan_sha256'] == args.plan_sha256,
                'Control receipt plan/operation differs')
    require(started['stream_receipt_sha256'] == args.stream_receipt_sha256
            and started['local_verification_sha256'] == args.local_verification_sha256,
            'Deletion used different backup verification')
    require(complete['archive_sha256'] == verified['archive_sha256']
            and complete['archive_bytes'] == verified['archive_bytes']
            and verified['unexpected_members'] == [] and verified['missing_members'] == []
            and verified['files'] == [dict(path=x['path'], member=x['member'], bytes=x['bytes'], sha256=x['sha256'])
                                     for x in plan['files']]
            and removed['status'] in ('exact_deletion_completed', 'exact_deletion_completed_with_retention'),
            'Backup/deletion completion mismatch')
    journal = controls['deletion-journal.jsonl']
    require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest() == removed['journal_sha256'],
            'Journal missing final newline or authenticated hash')
    planned = {x['path']: x for x in plan['files']}
    require(len(planned) == len(plan['files']), 'Duplicate approved original paths')
    terminal, intents, unlinked, changed, absent = {}, set(), [], [], []
    for line in journal.splitlines():
        row = strict_json(line); event = row['event']
        if event == 'active_jobs_rechecked':
            require({x['job_id'] for x in row['observation']['jobs']} <= set(plan['audited_active_job_ids']),
                    'Journal contains unaudited active jobs')
            continue
        require(row['path'] in planned, 'Journal path outside exact plan')
        if event == 'unlink_intent':
            require(row['path'] not in intents and row['path'] not in terminal,
                    'Duplicate unlink intent or intent after terminal event')
            require(row['sha256'] == planned[row['path']]['sha256'] and row['bytes'] == planned[row['path']]['bytes'],
                    'Unlink intent bytes differ from verified originals')
            intents.add(row['path'])
            continue
        require(event in ('unlinked_verified_copy', 'retained_changed_or_unreadable', 'retained_or_already_absent')
                and row['path'] not in terminal, 'Duplicate/unknown journal terminal event')
        terminal[row['path']] = event
        target = unlinked if event == 'unlinked_verified_copy' else changed if event == 'retained_changed_or_unreadable' else absent
        target.append(row['path'])
        if event == 'unlinked_verified_copy':
            require(row['path'] in intents and row['sha256'] == planned[row['path']]['sha256']
                    and row['bytes'] == planned[row['path']]['bytes'],
                    'Unlink result differs from verified originals')
    require(set(terminal) == set(planned), 'Journal lacks complete exact path outcomes')
    require(len(unlinked) == removed['removed_files']
            and sum(planned[p]['bytes'] for p in unlinked) == removed['removed_logical_bytes']
            and len(changed) == removed['retained_changed_or_unreadable']
            and len(absent) == removed['already_absent_no_removal_claim'], 'Removal accounting differs')
    observations = []
    for path, event in terminal.items():
        try:
            parent, name = nofollow_parent(canonical(path))
            try:
                observed = metadata(os.stat(name, dir_fd=parent, follow_symlinks=False))
            finally:
                os.close(parent)
        except FileNotFoundError:
            observed = None
        require((event == 'unlinked_verified_copy' and observed is None)
                or event != 'unlinked_verified_copy', 'A removed exact path reappeared; inspect before further cleanup')
        require(event != 'retained_changed_or_unreadable' or observed is not None,
                'Retained changed path is no longer present')
        observations.append(dict(path=path, journal_event=event, current_lstat=observed))
    result['original_path_observations'] = observations
    result['removal_accounting'] = dict(removed_files=len(unlinked), removed_logical_bytes=removed['removed_logical_bytes'],
        retained_changed=len(changed), already_absent_not_claimed_removed=len(absent),
        unique_logical_candidates_from_plan=plan.get('unique_logical_candidate_bytes'),
        actual_physical_or_user_quota_bytes_reclaimed_established=False,
        note='Logical removed bytes can count hardlink aliases repeatedly; fresh quota is separate and may be cached or affected by other active writes.')
    # Explicit pins are root-selected from reviewed protection receipts. No
    # recursive result discovery and no huge trajectory summary hashing.
    sources, files = pin_spec['sources'], pin_spec['files']
    require(isinstance(sources, list) and isinstance(files, list) and sources and files,
            'Explicit current source and scientific input pins are required')
    cap = pin_spec['max_total_hash_bytes']
    require(type(cap) is int and 0 < cap <= 16 * 1024**3, 'Unbounded explicit protected pin byte cap')
    result['protected_pin_rehashes'] = []; result['source_scopes'] = []
    total = 0
    for source in sources:
        root = canonical(source['root']); require(protected(root, protection), 'Unprotected source root in audit spec')
        require(isinstance(source['members'], dict) and source['members'], 'Missing exact source member map')
        manifest = canonical(source['manifest_path'])
        require(manifest.is_relative_to(root), 'Source manifest escaped source root')
        _, manifest_proof = read_exact(manifest, expected_sha=source['manifest_sha256'], keep_bytes=False)
        ignored = source.get('ignore_directory_names', [])
        require(isinstance(ignored, list) and set(ignored) <= {'__pycache__', '.pytest_cache'},
                'Only declared nonexecution cache directories may be ignored')
        actual = set()
        def walk_error(error):
            raise error
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            for name in dirs:
                require(stat.S_ISDIR(os.lstat(Path(directory) / name).st_mode), 'Linked/special source directory')
            dirs[:] = [name for name in dirs if name not in ignored]
            for name in names:
                path = Path(directory) / name
                require(stat.S_ISREG(os.lstat(path).st_mode), 'Linked/special source member')
                actual.add(path.relative_to(root).as_posix())
        require(actual == set(source['members']) | {manifest.relative_to(root).as_posix()},
                'Current whole source member scope differs from explicit approved seal')
        for member, expected_sha in source['members'].items():
            rel = PurePosixPath(member)
            require(not rel.is_absolute() and rel.as_posix() == member and '..' not in rel.parts, 'Unsafe source member')
            _, proof = read_exact(root / member, expected_sha=expected_sha,
                                  maximum=min(3 * 1024**3, cap - total), keep_bytes=False)
            total += proof['bytes']; result['protected_pin_rehashes'].append(proof)
        result['source_scopes'].append(dict(root=str(root), members_rehashed=len(source['members']), manifest=manifest_proof))
    for item in files:
        path = canonical(item['path']); require(protected(path, protection), 'Unprotected scientific pin in audit spec')
        require(not ('/payload/study/' in str(path) and path.name in
                    {'summary.json', 'report.json', 'states.jsonl', 'updates.jsonl'}
                    and item['bytes'] > 32 * 1024**2),
                'Do not rehash gigantic saved result summaries for this cleanup audit')
        _, proof = read_exact(path, expected_sha=item['sha256'],
                              maximum=min(3 * 1024**3, cap - total), keep_bytes=False)
        require(proof['bytes'] == item['bytes'], 'Scientific input length differs')
        total += proof['bytes']; result['protected_pin_rehashes'].append(proof)
    require(total <= cap, 'Audit pin scope exceeded its explicit byte cap')
    result['protected_bytes_rehashed'] = total
    result['queue'] = query(['bjobs', '-noheader', '-o', 'jobid stat'])
    require(result['queue']['returncode'] == 0, 'Fresh supported account queue query failed')
    result['work3_quota'] = query(['getquota_work3.sh'])
    require(result['work3_quota']['returncode'] == 0, 'Fresh work3 quota query failed')
    fs = os.statvfs('/work3/s234818')
    result['physical_filesystem_observation'] = dict(f_blocks=fs.f_blocks, f_bfree=fs.f_bfree,
        f_bavail=fs.f_bavail, f_frsize=fs.f_frsize, f_files=fs.f_files, f_ffree=fs.f_ffree, f_favail=fs.f_favail,
        scope='Whole shared filesystem counters; not the user quota or a reservation. Exact all-zero inode counters remain unreported.')
    result.update(status='passed_read_only_post_deletion_protection_audit', completed_unix_ns=time.time_ns(),
                  scientific_results_or_settings_changed=False, files_deleted_by_this_audit=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--plan-sha256', required=True)
    p.add_argument('--stream-receipt-sha256', required=True); p.add_argument('--local-verification-sha256', required=True)
    p.add_argument('--pin-spec', type=Path, required=True); p.add_argument('--pin-spec-sha256', required=True)
    p.add_argument('--timeout-seconds', type=int, default=1800)
    args = p.parse_args(); require(1 <= args.timeout_seconds <= 1800, 'Unbounded read-only audit')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Post-deletion audit timed out')))
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
    result = dict(version='mosaic-storage-offload-post-deletion-audit-v1', status='incomplete',
                  started_unix_ns=time.time_ns(), read_only=True)
    try:
        audit(args, result)
    except BaseException as error:
        result.update(status='failed_or_incomplete_read_only_audit', error=type(error).__name__ + ': ' + str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['status'] == 'passed_read_only_post_deletion_protection_audit' else 1


if __name__ == '__main__':
    raise SystemExit(main())
