"""Separate exact-file cleanup after root-approved independent local verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

import offload_protocol as protocol


def timeout(signum, frame):
    raise TimeoutError('Bounded deletion time expired; inspect durable journal')


def read_control(path, expected=None):
    path = protocol.canonical_path(str(path))
    protocol.require(protocol.below(path, protocol.CLEANUP_ROOT), 'Control receipt outside exact cleanup root')
    parent, name = protocol.nofollow_parent(path)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, 'rb') as stream:
            protocol.require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'Nonregular control receipt')
            raw = stream.read(protocol.MAX_PLAN_BYTES + 1)
    finally:
        os.close(parent)
    protocol.require(len(raw) <= protocol.MAX_PLAN_BYTES, 'Oversized control receipt')
    if expected is not None:
        protocol.require(hashlib.sha256(raw).hexdigest() == expected, 'Root-approved receipt bytes changed')
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--stream-receipt', type=Path, required=True)
    parser.add_argument('--stream-receipt-sha256', required=True)
    parser.add_argument('--local-verification', type=Path, required=True)
    parser.add_argument('--local-verification-sha256', required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 3600, 1)
    raw = read_control(args.plan, args.plan_sha256)
    plan = protocol.validate_plan(raw, args.plan_sha256)
    directory = protocol.CLEANUP_ROOT / plan['operation_id']
    protocol.require(args.plan == directory / 'approved-plan.json'
                     and args.stream_receipt == directory / 'stream-complete.json'
                     and args.local_verification == directory / 'local-verification.json',
                     'Plan/backup/verification do not belong to the same exact operation')
    completed = protocol.strict_json(read_control(args.stream_receipt, args.stream_receipt_sha256))
    verified = protocol.strict_json(read_control(args.local_verification, args.local_verification_sha256))
    protocol.verify_delete_authority(plan, raw, completed, verified)
    # Exclusive start prevents accidental replay after partial/unknown deletion.
    started = {'version': 'mosaic-storage-offload-delete-v1', 'status': 'started_unknown_until_complete',
               'operation_id': plan['operation_id'], 'plan_sha256': args.plan_sha256,
               'stream_receipt_sha256': args.stream_receipt_sha256,
               'local_verification_sha256': args.local_verification_sha256,
               'started_unix_ns': time.time_ns(), 'mode': 'literal verified file unlink only; no recursive remove'}
    protocol.exclusive(directory / 'deletion-started.json', started)
    signal.setitimer(signal.ITIMER_REAL, plan['timeout_seconds'], 1)
    try:
        with (directory / 'deletion-journal.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
            result = protocol.delete_exact(plan, raw, completed, verified, journal)
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(version=started['version'], operation_id=plan['operation_id'],
                      plan_sha256=args.plan_sha256, completed_unix_ns=time.time_ns(),
                      journal_sha256=protocol.sha(directory / 'deletion-journal.jsonl'))
        protocol.exclusive(directory / 'deletion-complete.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        protocol.exclusive(directory / 'deletion-failed.json',
                           {**started, 'status': 'failed_or_partial_deletion_inspect_journal',
                            'failed_unix_ns': time.time_ns(), 'error': repr(error),
                            'journal_sha256': protocol.sha(directory / 'deletion-journal.jsonl')
                                if (directory / 'deletion-journal.jsonl').exists() else None})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
