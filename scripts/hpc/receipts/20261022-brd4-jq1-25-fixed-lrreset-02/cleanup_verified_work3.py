"""Retire exact Mosaic file copies only after complete local byte backups.

This is an administrative byte/hash operation, not scientific computation.
Unknown, changed, current, linked and non-Mosaic paths are never removed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


BASE = Path('/work3/s234818/mosaic')
CONTROL = BASE / 'runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02/cleanup-recovery-02'
CURRENT_ARCHIVE = BASE / '20261021-brd4-jq1-25-fixed-02.tar.gz'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_path(raw):
    path=Path(raw)
    allowed_roots=[BASE/'runs/20261021-brd4-jq1-25-fixed-storage-03/run-protenix/payload/study',
        BASE/'runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2/payload/study',
        BASE/'runs/20261021-brd4-jq1-25-fixed-report-04/interim-0001/payload']
    allowed_files={BASE/'runs/20261021-brd4-jq1-25-fixed-report-04/interim-0001.tar.gz',
        BASE/'runs/20261021-brd4-jq1-25-fixed-02/native-evidence-01.tar.gz',
        BASE/'20261021-brd4-jq1-25-fixed-02.tar.gz',BASE/'20261021-brd4-jq1-25-fixed-lrboost-01.tar.gz'}
    if not path.is_absolute() or '..' in path.parts or path.resolve()!=path:
        raise ValueError('Unsafe/linked cleanup path')
    if path not in allowed_files and not any(path!=root and path.is_relative_to(root) for root in allowed_roots):
        raise ValueError('Outside the exact inactive older-work3 allowlist')
    return path


def active_jobs(paths, destination):
    result = subprocess.run(['bjobs', '-noheader', '-o', 'jobid stat job_name:100 command:2000'],
                            capture_output=True, text=True, timeout=30)
    if result.returncode and not (not result.stdout.strip()
            and re.fullmatch(r'No unfinished job found\.?\s*', result.stderr.strip(), re.I)):
        raise ValueError('Active-job inventory unavailable; cleanup blocked')
    lines = [line for line in result.stdout.splitlines() if 'mosaic' in line.lower()]
    destination.write_text(json.dumps({'returncode': result.returncode, 'mosaic_jobs': lines,
                                     'stderr': result.stderr}, indent=2) + '\n')
    groups = {str(BASE / 'runs' / p.relative_to(BASE).parts[1]) for p in paths
              if p.is_relative_to(BASE / 'runs')}
    if any(group in line for group in groups for line in lines):
        raise ValueError('An active Mosaic job references a cleanup result group')
    if any(str(path) in line for path in paths if path.parent == BASE for line in lines):
        raise ValueError('An active Mosaic job references a cleanup transport archive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if (args.plan.resolve().parent != CONTROL or CONTROL.resolve() != CONTROL
            or not re.fullmatch(r'attempt-\d{2}', args.attempt)
            or digest(args.plan) != args.plan_sha256):
        raise ValueError('Wrong exact immutable cleanup plan/control identity')
    plan = json.loads(args.plan.read_text())
    if (plan['version'] != 'mosaic-lrreset-verified-work3-backup-cleanup-v1'
            or plan['allowed_base'] != str(BASE)
            or type(plan['maximum_removed_bytes']) is not int
            or not 0 < plan['maximum_removed_bytes'] <= 2 * 1024**3):
        raise ValueError('Wrong declared cleanup scope or byte cap')
    rows = plan['files']
    if not rows or len(rows) != len({row['path'] for row in rows}):
        raise ValueError('Missing/duplicate cleanup candidates')
    paths = [safe_path(row['path']) for row in rows]
    for row in rows:
        if (type(row['bytes']) is not int or row['bytes'] < 0
                or not re.fullmatch(r'[a-f0-9]{64}', row['sha256'])
                or not row['local_backup']):
            raise ValueError('Missing exact local backup byte identity')
    attempt = CONTROL / args.attempt
    attempt.mkdir(exist_ok=False)
    active_jobs(paths, attempt / 'jobs-before-verification.json')
    matches, retained, absent = [], [], []
    for row, path in zip(rows, paths):
        if not path.exists():
            absent.append(row['path'])
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Nonregular file in cleanup candidates')
        if info.st_nlink != 1:
            retained.append({'path': row['path'], 'reason': 'linked_file'})
            continue
        if info.st_size != row['bytes']:
            retained.append({'path': row['path'], 'reason': 'different_size'})
            continue
        value = digest(path)
        after = path.lstat()
        if (value != row['sha256'] or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            retained.append({'path': row['path'], 'reason': 'different_or_changing_bytes'})
            continue
        matches.append(dict(row, device=info.st_dev, inode=info.st_ino, mtime_ns=info.st_mtime_ns))
    preflight = {'plan_sha256': args.plan_sha256, 'matched': matches,
                 'retained': retained, 'absent': absent,
                 'scope': 'Every matched remote file exactly equals its verified local backup member'}
    with (attempt / 'verified-candidates.json').open('x') as stream:
        json.dump(preflight, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    active_jobs(paths, attempt / 'jobs-before-removal.json')
    removed_bytes, removed_files = 0, 0
    with (attempt / 'deletion-journal.jsonl').open('x') as journal:
        for row in matches:
            path = safe_path(row['path'])
            info = path.lstat()
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                    row['device'], row['inode'], row['bytes'], row['mtime_ns']) or info.st_nlink != 1:
                raise ValueError('File changed after byte verification; stop before removal')
            if removed_bytes + row['bytes'] > plan['maximum_removed_bytes']:
                continue
            entry = {'utc': datetime.now(timezone.utc).isoformat(),
                     'path': row['path'], 'bytes': row['bytes'],
                     'sha256': row['sha256'], 'local_backup': row['local_backup'],
                     'archive_member': row.get('archive_member')}
            journal.write(json.dumps(dict(entry, event='intent')) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            path.unlink()
            removed_files += 1
            removed_bytes += row['bytes']
            journal.write(json.dumps(dict(entry, event='removed')) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            if removed_files % 1000 == 0:
                print(json.dumps({'progress_removed_files': removed_files,
                                  'progress_removed_bytes': removed_bytes}), flush=True)
        journal.flush()
        os.fsync(journal.fileno())
    quota = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, timeout=60)
    (attempt / 'quota-after.txt').write_text(quota.stdout + quota.stderr)
    receipt = {'status': 'completed', 'utc': datetime.now(timezone.utc).isoformat(),
               'plan_sha256': args.plan_sha256, 'removed_files': removed_files,
               'removed_logical_bytes': removed_bytes, 'retained_changed_files': len(retained),
               'already_absent_files': len(absent), 'quota_returncode': quota.returncode,
               'deletion_journal_sha256': digest(attempt / 'deletion-journal.jsonl'),
               'verified_candidates_sha256': digest(attempt / 'verified-candidates.json'),
               'scope': 'Only exact locally backed-up previous Mosaic copies removed; all unknown/current data retained'}
    with (attempt / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
