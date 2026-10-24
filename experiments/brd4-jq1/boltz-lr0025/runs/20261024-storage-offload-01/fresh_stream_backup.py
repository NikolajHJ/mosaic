"""One fresh direct SSH binary stream into an exclusive laptop backup file.

Only the parent/root operator executes this client. Received bytes remain
UNVERIFIED until verify_stream_backup.py accepts the separately fetched remote
completion receipt and every original member. No deletion is implemented.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import threading
import time

from verify_stream_backup import file_record, hash_bytes, load_plan, require

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BACKUP_ROOT = REPO / 'results/hpc/20261024-storage-offload-01'


def exclusive(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def remote_command(plan, plan_sha, remote_python, packer, receipt):
    for name in (remote_python, packer, receipt):
        path = PurePosixPath(name)
        require(path.is_absolute() and path.as_posix() == name and '..' not in path.parts
                and not any(value in name for value in ('\n', '\r', '\\', '\x00')),
                'Noncanonical remote executable/receipt path')
    expected = PurePosixPath(plan['cleanup_root']) / plan['operation_id'] / 'stream-complete.json'
    require(PurePosixPath(receipt) == expected, 'Remote completion receipt escapes exact operation')
    require(remote_python.startswith('/work3/s234818/mosaic/envs/')
            and packer.startswith('/work3/s234818/mosaic/'), 'Undeclared remote interpreter/packer scope')
    argv = [remote_python, packer, '--plan-stdin', '--plan-sha256', plan_sha, '--receipt', receipt]
    shell_command = ' '.join(shlex.quote(value) for value in argv)
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(shell_command)], argv


def stream_once(plan_path, plan_sha, output, *, remote_python, packer, receipt):
    plan_path, output = Path(plan_path), Path(output)
    raw = plan_path.read_bytes(); plan = load_plan(raw, plan_sha)
    require(output.parent.is_dir() and not output.parent.is_symlink()
            and output.parent.resolve().is_relative_to(BACKUP_ROOT.resolve()), 'Backup output must stay under dedicated Mosaic results directory')
    require(not output.is_symlink() and output.name.endswith('.tar.gz'), 'Expected new named gzip archive')
    names = {'archive': output, 'started': output.with_name(output.name + '.started.json'),
             'transfer': output.with_name(output.name + '.transfer.json'),
             'stderr': output.with_name(output.name + '.stderr.bin')}
    require(not any(os.path.lexists(path) for path in names.values()), 'Existing/unknown transfer outcome; no blind replay')
    args, remote_argv = remote_command(plan, plan_sha, remote_python, packer, receipt)
    started = {'version': 'mosaic-storage-offload-fresh-stream-client-v1', 'status': 'unknown',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'operation_id': plan['operation_id'],
        'approved_plan': file_record(plan_path), 'plan_sha256': plan_sha, 'plan_bytes': len(raw),
        'args': args, 'remote_argv': remote_argv, 'expected_remote_completed_receipt': receipt,
        'archive_path': str(output.resolve()), 'client': file_record(Path(__file__)),
        'scope': 'One connection, fresh exact bytes only; no remote big archive, no resume, no deletion or verification claim.'}
    exclusive(names['started'], started)
    received = 0; digest = hashlib.sha256(); child = None; failure = None; returncode = None
    watchdog = None; timed_out = threading.Event()
    begin = time.monotonic()
    try:
        with output.open('xb') as archive, names['stderr'].open('xb') as errors:
            child = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            def expire_owned_client():
                if child.poll() is None:
                    timed_out.set()
                    child.kill()
            watchdog = threading.Timer(plan['timeout_seconds'] + 60, expire_owned_client)
            watchdog.daemon = True; watchdog.start()
            # The packer first reads the exact bounded plan to EOF, then streams
            # its archive. Stderr is a file, so it cannot deadlock this pipeline.
            child.stdin.write(raw); child.stdin.flush(); child.stdin.close()
            child.stdin = None
            for chunk in iter(lambda: child.stdout.read(1024 * 1024), b''):
                require(time.monotonic() - begin <= plan['timeout_seconds'] + 60,
                        'Client elapsed transfer limit reached')
                archive.write(chunk); digest.update(chunk); received += len(chunk)
            archive.flush(); os.fsync(archive.fileno())
            returncode = child.wait(timeout=60)
            require(not timed_out.is_set() and returncode == 0 and received > 0,
                    'Remote stream failed, timed out or produced no archive bytes')
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        if child is not None and child.poll() is None:
            # Only this exact owned SSH child is stopped. No remote job or
            # unrelated transfer is touched; the remote outcome remains unknown.
            child.kill(); returncode = child.wait(timeout=30)
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if child is not None and returncode is None:
            returncode = child.poll()
        result = {'version': started['version'],
            'status': 'stream_received_unverified' if failure is None else 'failed_or_unknown_partial_preserved',
            'operation_id': plan['operation_id'], 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': plan_sha, 'expected_remote_completed_receipt': receipt, 'returncode': returncode,
            'archive_path': str(output.resolve()), 'received_bytes': received, 'received_sha256': digest.hexdigest(),
            'elapsed_seconds': time.monotonic() - begin, 'error': failure,
            'timed_out': timed_out.is_set(),
            'all_files_verified': False, 'deletion_permitted': False,
            'remote_outcome_authenticated': False,
            'next_step': 'Root fetches/authenticates exact completed remote receipt separately; verify every member before any removal.'}
        if names['stderr'].is_file():
            result['stderr'] = file_record(names['stderr'])
        if output.is_file():
            result['actual_local_archive_bytes'] = output.stat().st_size
        exclusive(names['transfer'], result)
    if failure is not None:
        raise RuntimeError('Partial/unknown outcome preserved; inspect exact local and remote receipts before recovery: ' + failure)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-python', required=True); parser.add_argument('--remote-packer', required=True)
    parser.add_argument('--remote-receipt', required=True)
    args = parser.parse_args()
    result = stream_once(args.plan, args.plan_sha256, args.output, remote_python=args.remote_python,
                         packer=args.remote_packer, receipt=args.remote_receipt)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
