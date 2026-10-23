"""Exact backed-up completed-report copies only; stdlib metadata/byte operations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
CONTROL = BASE / 'restart-lrreset-cleanup-03'
PATTERN = re.compile(r'interim-000[12]/payload/report/(protenix|boltz2|boltz2_chemistry)/(frames-[0-9]+\.(bin|json)|curves\.json)')
ARCHIVES = {'interim-0001.tar.gz', 'interim-0002.tar.gz'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('plan_sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(args.plan.resolve() == CONTROL / 'plan.json' and digest(args.plan) == args.plan_sha256,
            'Wrong declared cleanup plan/identity')
    plan = json.loads(args.plan.read_text())
    require(plan['version'] == 'mosaic-exact-report-backup-retirement-v1'
            and plan['local_all_archive_members_verified'] is True
            and plan['local_verified_backups'] and plan['files'], 'Missing fully verified local backups')
    require(sum(x['bytes'] for x in plan['files']) < 2 * 1024**3, 'Cleanup exceeds declared 2GiB maximum')
    require(len({x['remote_path'] for x in plan['files']}) == len(plan['files']), 'Duplicate target')
    targets = []
    for item in plan['files']:
        p = Path(item['remote_path'])
        require(p.is_relative_to(BASE) and (p.name in ARCHIVES and p.parent == BASE
                or PATTERN.fullmatch(p.relative_to(BASE).as_posix())), 'Target outside explicit completed-report allowlist')
        require(p.resolve() == p.absolute() and not p.is_symlink(), 'Target/parent symlink or alias')
        targets.append((p, item))
    scheduler = subprocess.run(['bjobs', '-w'], text=True, capture_output=True, timeout=30)
    require(scheduler.returncode == 0, 'Cannot establish actual unfinished-job inventory')
    write(CONTROL / ('execute-scheduler.json' if args.execute else 'preflight-scheduler.json'),
          {'stdout': scheduler.stdout, 'stderr': scheduler.stderr, 'rc': scheduler.returncode})
    for name in ('interim-0001', 'interim-0002'):
        require(not any(name in line and re.search(r'\s(RUN|PEND|PSUSP|USUSP|SSUSP)\s', line)
                        for line in scheduler.stdout.splitlines()), 'A report candidate is still unfinished')
    observations = []
    # Verify every exact target before any removal; nonmatching files are retained.
    for p, item in targets:
        if not p.exists():
            observations.append({'path': str(p), 'status': 'absent_retained'}); continue
        require(p.is_file(), 'Nonregular target')
        stat = p.stat()
        same = stat.st_size == item['bytes'] and digest(p) == item['sha256']
        observations.append({'path': str(p), 'status': 'verified' if same else 'changed_retained',
                             'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                             'inode': stat.st_ino, 'device': stat.st_dev})
    label = 'execute' if args.execute else 'preflight'
    write(CONTROL / (label + '-verification.json'), {'plan_sha256': args.plan_sha256, 'files': observations})
    if not args.execute:
        print(json.dumps({'status': 'verified_no_deletion', 'files': len(observations),
                          'bytes': sum(i['bytes'] for i in plan['files'])})); return
    removed = 0; removed_bytes = 0
    with (CONTROL / 'journal.jsonl').open('x', encoding='utf-8') as journal:
        for (p, item), observation in zip(targets, observations):
            event = dict(observation)
            if observation['status'] == 'verified':
                stat = p.stat()
                unchanged = (stat.st_size == item['bytes'] and stat.st_mtime_ns == observation['mtime_ns']
                             and stat.st_ino == observation['inode'] and stat.st_dev == observation['device']
                             and digest(p) == item['sha256'])
                if unchanged:
                    event.update(status='verified_before_unlink', sha256=item['sha256'])
                    journal.write(json.dumps(event) + '\n'); journal.flush(); os.fsync(journal.fileno())
                    p.unlink(); removed += 1; removed_bytes += item['bytes']; event['status'] = 'removed_exact_backed_up_copy'
                else:
                    event['status'] = 'changed_retained'
            journal.write(json.dumps(event) + '\n'); journal.flush(); os.fsync(journal.fileno())
    quota = subprocess.run(['getquota_work3.sh'], text=True, capture_output=True, timeout=60)
    write(CONTROL / 'completion.json', {'status': 'completed', 'utc': datetime.now(timezone.utc).isoformat(),
          'plan_sha256': args.plan_sha256, 'removed_files': removed, 'removed_logical_bytes': removed_bytes,
          'journal_sha256': digest(CONTROL / 'journal.jsonl'), 'quota_rc': quota.returncode,
          'post_quota': quota.stdout, 'quota_stderr': quota.stderr,
          'scope': 'Exact verified duplicate report frames/curves and archives only; all original native studies untouched'})
    print((CONTROL / 'completion.json').read_text())


if __name__ == '__main__':
    main()
