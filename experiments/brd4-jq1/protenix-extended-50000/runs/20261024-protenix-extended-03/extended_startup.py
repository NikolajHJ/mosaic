"""Actual Protenix command metadata on CPU; genuine pre-model startup on GPU."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from unittest.mock import patch

VERSION = 'protenix-actual-command-gpu-premodel-subprocess-v2'
MESSAGE = 'Intentional Protenix GPU startup-only boundary; no native model loaded'
REVISION = 'c74c659344123937608077e3e4078f75f09cd743'


class ProtenixPremodelBoundaryReached(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def authenticate_source(source, manifest):
    source = Path(source).resolve()
    require(isinstance(manifest, str) and re.fullmatch('[0-9a-f]{64}', manifest), 'Malformed source identity')
    require(sha(source / 'source_manifest.json') == manifest, 'Source manifest changed')
    value = json.loads((source / 'source_manifest.json').read_bytes())
    require(value['git_revision_reference'] == REVISION, 'Authenticated ancestor revision differs')
    files = value['files']
    require(isinstance(files, dict) and 'pyproject.toml' in files, 'Incomplete source scope')
    observed = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() or p.is_symlink()}
    require(observed == set(files) | {'source_manifest.json'}, 'Sealed execution member scope differs')
    for path in [source, *source.rglob('*')]:
        require(not path.is_symlink() and (path.is_file() or path.is_dir())
                and not path.stat().st_mode & 0o222, 'Execution source must be regular read-only content')
    for name, digest in files.items():
        path = source / name
        require(path.resolve().is_relative_to(source) and sha(path) == digest, 'Source member changed')
    return value


def owned_tmp(directory, queue):
    require(os.environ.get('LSB_JOBID') and os.environ.get('LSB_QUEUE') == queue, 'Actual declared allocation required')
    require(os.environ.get('TMPDIR'), 'Owned node TMPDIR required')
    root = Path(os.environ['TMPDIR']).resolve()
    directory = Path(directory)
    require(not os.path.lexists(directory) and directory.resolve().is_relative_to(root)
            and directory.resolve() != root, 'Disposable startup directory must be new below node TMPDIR')
    return directory.resolve()


def argument(args, option):
    require(args.count(option) == 1 and args.index(option) + 1 < len(args), 'Missing/duplicated driver option: ' + option)
    return args[args.index(option) + 1]


def capture_command(ops, manifest, directory, *, arm, index=0, phase='native'):
    """Execute actual admission/argument factory, intercept only the final driver."""
    require(arm in ('random', 'continuation') and phase in ('native', 'segment'), 'Unknown command scope')
    ops.offset(arm, index)
    directory = Path(directory)
    directory.mkdir()
    parent = directory / 'command'
    parent.mkdir()
    driver = Path(ops.SOURCE) / 'examples/brd4_25_atoms.py'
    calls = []
    original_run = subprocess.run

    def capture(args, *positional, **kwargs):
        if isinstance(args, (list, tuple)) and len(args) > 1 and Path(str(args[1])).resolve() == driver:
            require(not calls and not positional and kwargs.get('cwd') == ops.SOURCE
                    and kwargs.get('check') is True and not kwargs.get('shell', False), 'Unexpected generated driver call')
            calls.append([str(value) for value in args])
            return subprocess.CompletedProcess(args, 0)
        return original_run(args, *positional, **kwargs)

    with patch.object(subprocess, 'run', capture):
        ops.produce(parent, phase, arm, index, manifest)
    require(len(calls) == 1, 'Actual controller did not generate exactly one driver command')
    args = calls[0]
    require(Path(args[0]).resolve() == Path(sys.executable).resolve(), 'Generated production interpreter differs')
    require(argument(args, '--manifest') == manifest
            and argument(args, '--mode') == ('validate' if phase == 'native' else 'run')
            and int(argument(args, '--offset')) == ops.offset(arm, index)
            and argument(args, '--segment-steps') == '10000', 'Actual command phase/global range differs')
    require(('--ready' in args) is (phase == 'segment')
            and ('--resume-snapshot' in args) is (ops.offset(arm, index) > 0), 'Actual ready/predecessor applicability differs')
    output = Path(argument(args, '--output')).resolve()
    require(output.is_relative_to(directory.resolve()) and not output.exists(), 'Actual driver output escapes disposable directory')
    return args


@contextmanager
def execution_context(source, args):
    old_argv, old_path, old_cwd = sys.argv[:], sys.path[:], Path.cwd()
    try:
        sys.argv = list(args[1:])
        sys.path[:0] = [str(source / 'scripts'), str(source / 'src'), str(source / 'tests')]
        os.chdir(source)
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        sys.argv[:] = old_argv


def execute_command(args, source, manifest):
    """Real driver/GPU/FP32/provenance; patch only the native model constructor."""
    require(os.environ.get('LSB_JOBID') and os.environ.get('LSB_QUEUE') == 'gpul40s', 'Genuine GPU allocation required')
    source = Path(source).resolve()
    declared = authenticate_source(source, manifest)
    require(Path(args[1]).resolve() == source / 'examples/brd4_25_atoms.py', 'Driver outside authenticated source')
    output = Path(argument(args, '--output'))
    reached = []

    def boundary(*positional, **kwargs):
        reached.append({'constructor': 'FrozenProtenixTiny.load',
                        'positional': [str(value) for value in positional],
                        'keyword': {key: str(value) for key, value in kwargs.items()}})
        raise ProtenixPremodelBoundaryReached(MESSAGE)

    caught = None
    with execution_context(source, args):
        module = importlib.import_module('mosaic.small_molecule.protenix')
        require(Path(module.__file__).resolve().is_relative_to(source), 'Model adapter imported outside execution source')
        with patch.object(module.FrozenProtenixTiny, 'load', boundary):
            try:
                runpy.run_path(args[1], run_name='__main__')
            except ProtenixPremodelBoundaryReached as error:
                caught = error
    require(caught is not None and len(reached) == 1, 'Actual driver failed before intentional model boundary')
    path = output / 'summary.json'
    summary = json.loads(path.read_bytes())
    require(summary['status'] == 'failed'
            and summary['error'] == 'ProtenixPremodelBoundaryReached: ' + MESSAGE,
            'Intentional disposable driver summary must remain failed')
    require(summary['source_manifest_sha256'] == manifest
            and summary['source']['git_revision_reference'] == REVISION, 'Actual provenance differs')
    execution_files = {name: digest for name, digest in declared['files'].items()
                       if name == 'pyproject.toml' or (Path(name).parts[0] in ('src', 'tests', 'examples', 'scripts', 'experiments')
                           and Path(name).suffix in ('.py', '.json', '.lsf', '.sh', '.toml', '.txt')
                           and '__pycache__' not in Path(name).parts)}
    require(summary['source']['files'] == execution_files
            and summary['source']['execution_content_sha256'] == hashlib.sha256(
                json.dumps(execution_files, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'Actual driver execution content identity differs')
    require(not summary.get('native_gradient_checks') and not summary.get('cases')
            and not summary.get('model_digest') and not (output / 'assessment.json').exists(),
            'Startup must not load model or claim scientific execution')
    return {'summary': summary, 'summary_sha256': sha(path), 'boundary': reached[0]}


def capture_cpu(ops, manifest, directory, *, arm, index=0, phase='native'):
    directory = owned_tmp(directory, 'hpc')
    declared = authenticate_source(ops.SOURCE, manifest)
    args = capture_command(ops, manifest, directory, arm=arm, index=index, phase=phase)
    require(authenticate_source(ops.SOURCE, manifest) == declared, 'Source changed during command capture')
    return {'version': VERSION, 'kind': 'cpu_generated_command_only', 'status': 'command_only_authenticated',
            'job_id': os.environ['LSB_JOBID'], 'queue': 'hpc', 'arm': arm, 'phase': phase, 'index': index,
            'source': str(ops.SOURCE), 'source_manifest_sha256': manifest, 'git_revision_reference': REVISION,
            'args': args, 'inputs': ops.startup_inputs(arm, phase, index, manifest),
            'actual_driver_executed': False, 'gpu_startup_passed': False, 'model_loaded': False,
            'gradient_passed': False, 'scientific_readiness_passed': False, 'optimizer_updates': 0,
            'saved_results_changed': False}


def gpu_startup(ops, manifest, directory, *, arm, index=0, phase='native'):
    directory = owned_tmp(directory, 'gpul40s')
    no_native_backend_imported()
    declared = authenticate_source(ops.SOURCE, manifest)
    args = capture_command(ops, manifest, directory, arm=arm, index=index, phase=phase)
    result = run_startup_child(args, ops.SOURCE, manifest, directory)
    no_native_backend_imported()
    require(authenticate_source(ops.SOURCE, manifest) == declared, 'Source changed during actual GPU startup')
    return {'version': VERSION, 'kind': 'gpu_startup_only', 'status': 'startup_only_passed',
            'job_id': os.environ['LSB_JOBID'], 'queue': 'gpul40s', 'arm': arm, 'phase': phase, 'index': index,
            'source': str(ops.SOURCE), 'source_manifest_sha256': manifest, 'git_revision_reference': REVISION,
            'args': args, 'inputs': ops.startup_inputs(arm, phase, index, manifest), **result,
            'actual_driver_executed': True, 'gpu_backend_guard_executed_unchanged': True,
            'model_loaded': False, 'gradient_passed': False, 'scientific_readiness_passed': False,
            'optimizer_updates': 0, 'saved_results_changed': False}


def no_native_backend_imported():
    """The long-lived controller must never initialize an inference backend."""
    forbidden = ('jax', 'jaxlib', 'torch', 'equinox')
    require(not any(name == root or name.startswith(root + '.')
                    for name in sys.modules for root in forbidden),
            'Native backend already imported in controller; stop before model execution')


def _child_path(path):
    require(os.environ.get('TMPDIR'), 'Owned node TMPDIR required')
    path = Path(path)
    require(not path.is_symlink() and path.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve()),
            'Child metadata must stay below owned node TMPDIR')
    return path


def execute_child_file(source, manifest, argv_path, receipt):
    """Short-lived same-allocation process; only this process imports JAX/model API."""
    require(os.environ.get('LSB_JOBID') and os.environ.get('LSB_QUEUE') == 'gpul40s', 'Actual same-allocation GPU child required')
    argv_path, receipt = _child_path(argv_path), _child_path(receipt)
    require(not os.path.lexists(receipt) and receipt.parent == argv_path.parent, 'Child receipt must be new beside command metadata')
    command = json.loads(argv_path.read_bytes())
    require(set(command) == {'version', 'source', 'source_manifest_sha256', 'args'}
            and command['version'] == VERSION and command['source'] == str(source)
            and command['source_manifest_sha256'] == manifest
            and isinstance(command['args'], list) and all(isinstance(value, str) for value in command['args']),
            'Child command/source identity differs')
    require(Path(argument(command['args'], '--output')).resolve().is_relative_to(argv_path.parent.resolve()),
            'Child driver output escapes disposable command directory')
    result = execute_command(command['args'], source, manifest)
    payload = {'version': VERSION, 'status': 'startup_only_passed', 'job_id': os.environ['LSB_JOBID'],
               'queue': 'gpul40s', 'source': str(source), 'source_manifest_sha256': manifest,
               'command_sha256': sha(argv_path), 'pid': os.getpid(), 'parent_pid': os.getppid(), **result}
    with receipt.open('x', encoding='utf8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())


def run_startup_child(args, source, manifest, directory):
    """Wait for the real GPU child to exit before allowing the production child."""
    no_native_backend_imported()
    directory = Path(directory)
    command_path = directory / 'child-command.json'
    receipt_path = directory / 'child-result.json'
    require(not os.path.lexists(command_path) and not os.path.lexists(receipt_path), 'Existing child outcome; no replay')
    with command_path.open('x', encoding='utf8') as handle:
        json.dump({'version': VERSION, 'source': str(source), 'source_manifest_sha256': manifest, 'args': args}, handle,
                  sort_keys=True, allow_nan=False); handle.write('\n')
    child_args = [args[0], str(Path(__file__).resolve()), '--execute-child', '--source', str(source),
                  '--manifest', manifest, '--argv-json', str(command_path), '--receipt', str(receipt_path)]
    child = subprocess.Popen(child_args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=source)
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        timed_out = True
        child.kill()
        stdout, stderr = child.communicate()
    stdout_path, stderr_path = directory / 'child-stdout.bin', directory / 'child-stderr.bin'
    stdout_path.write_bytes(stdout); stderr_path.write_bytes(stderr)
    exit_record = {'pid': child.pid, 'parent_pid': os.getpid(), 'args': child_args,
                   'returncode': child.returncode, 'timed_out': timed_out, 'child_fully_reaped': child.poll() is not None,
                   'stdout': {'path': str(stdout_path), 'bytes': len(stdout), 'sha256': sha(stdout_path)},
                   'stderr': {'path': str(stderr_path), 'bytes': len(stderr), 'sha256': sha(stderr_path)}}
    with (directory / 'child-exit.json').open('x', encoding='utf8') as handle:
        json.dump(exit_record, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    require(not timed_out and child.returncode == 0 and child.poll() == 0, 'Isolated GPU startup child did not exit successfully')
    require(receipt_path.is_file() and not receipt_path.is_symlink(), 'Successful child has no regular receipt')
    result = json.loads(receipt_path.read_bytes())
    require(result['version'] == VERSION and result['status'] == 'startup_only_passed'
            and result['job_id'] == os.environ.get('LSB_JOBID') and result['queue'] == 'gpul40s'
            and result['source'] == str(source) and result['source_manifest_sha256'] == manifest
            and result['command_sha256'] == sha(command_path)
            and result['pid'] == child.pid and result['parent_pid'] == os.getpid(),
            'Actual same-allocation child receipt/command/PID differs')
    no_native_backend_imported()
    return {'summary': result['summary'], 'summary_sha256': result['summary_sha256'], 'boundary': result['boundary'],
            'execution_isolation': {'mode': 'short_lived_same_allocation_subprocess', **exit_record,
                                    'child_receipt': {'path': str(receipt_path), 'bytes': receipt_path.stat().st_size,
                                                      'sha256': sha(receipt_path)},
                                    'child_receipt_value': result, 'parent_native_backend_imported': False}}


def failure_metadata(directory):
    """Preserve bounded raw child metadata even when admission stops on failure."""
    directory = Path(directory)
    result = []
    total = 0
    if not directory.exists():
        return result
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.is_symlink() or path.suffix not in ('.json', '.bin'):
            continue
        data = path.read_bytes(); total += len(data)
        require(total <= 2 * 1024**2, 'Startup failure metadata exceeded declared bound')
        result.append({'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                       'text': data.decode('utf8', errors='replace')})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-child', action='store_true', required=True)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--argv-json', type=Path, required=True); parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    execute_child_file(args.source, args.manifest, args.argv_json, args.receipt)


if __name__ == '__main__':
    main()
