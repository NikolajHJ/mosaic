"""Remote stdout-only gzip-tar backup; never deletes or writes a large archive."""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import sys
import time

import offload_protocol as protocol


def timeout(signum, frame):
    raise TimeoutError('Bounded backup time expired; preserve partial/unknown outcome')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-stdin', action='store_true', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    args = parser.parse_args()
    protocol.require(1 <= args.timeout_seconds <= 3600, 'Unbounded input/stream timeout')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds, 1)
    raw = sys.stdin.buffer.read(protocol.MAX_PLAN_BYTES + 1)
    plan = protocol.validate_plan(raw, args.plan_sha256)
    directory = protocol.CLEANUP_ROOT / plan['operation_id']
    protocol.require(args.receipt == directory / 'stream-complete.json', 'Receipt outside exact operation directory')
    protocol.require(protocol.CLEANUP_ROOT.is_dir() and not protocol.CLEANUP_ROOT.is_symlink(),
                     'Parent must prepare the exact cleanup control root')
    # Literal, no-link ancestry; never reuse an operation with unknown outcome.
    parent, name = protocol.nofollow_parent(directory)
    try:
        os.mkdir(name, dir_fd=parent)
    finally:
        os.close(parent)
    with (directory / 'approved-plan.json').open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    started = {'version': protocol.STREAM_VERSION, 'status': 'started_unknown_until_complete',
               'operation_id': plan['operation_id'], 'plan_sha256': args.plan_sha256,
               'plan_bytes': len(raw), 'started_unix_ns': time.time_ns(),
               'output': 'gzip-tar stdout only; no remote large archive', 'deleted_files': 0}
    protocol.exclusive(directory / 'stream-started.json', started)
    complete = False
    try:
        signal.setitimer(signal.ITIMER_REAL, min(args.timeout_seconds, plan['timeout_seconds']), 1)
        before = protocol.check_active(plan, protocol.active_jobs())
        protocol.exclusive(directory / 'active-jobs-before-stream.json', before)
        counts = protocol.pack_stream(plan, raw, sys.stdout.buffer)
        after = protocol.check_active(plan, protocol.active_jobs())
        protocol.exclusive(directory / 'active-jobs-after-stream.json', after)
        result = {**started, **counts, 'status': 'streamed_complete',
                  'completed_unix_ns': time.time_ns(), 'download_or_local_verification_claimed': False,
                  'deleted_files': 0, 'independent_local_verification_required_for_deletion': True}
        protocol.exclusive(args.receipt, result)
        complete = True
    except BaseException as error:
        # Stop further alarms while writing a small failure receipt. No archive
        # completion or deletion permission follows from an interrupted stream.
        signal.setitimer(signal.ITIMER_REAL, 0)
        protocol.exclusive(directory / 'stream-failed.json',
                           {**started, 'status': 'failed_or_partial_stream', 'error': repr(error),
                            'failed_unix_ns': time.time_ns(), 'deleted_files': 0})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    protocol.require(complete, 'No completed remote stream receipt')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
