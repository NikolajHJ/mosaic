"""Reusable allocated-CPU report engine. No model/optimizer entry point exists."""
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import uuid
import xml.etree.ElementTree as ET

import quick_inputs as inputs

GIB = 1024**3


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(inputs.encoded(value))


def verify_source(source, pin):
    source = Path(source)
    inputs.require(inputs.sha(source / 'engine-manifest.json') == pin, 'Report engine manifest changed')
    manifest = inputs.load(source / 'engine-manifest.json')
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
    inputs.require(actual == set(manifest['files']) | {'engine-manifest.json'}
                   and all(not p.is_symlink() for p in source.rglob('*')), 'Unexpected/linked engine member')
    for name, value in manifest['files'].items():
        part = PurePosixPath(name)
        inputs.require(not part.is_absolute() and '..' not in part.parts and '\\' not in name, 'Unsafe engine member')
        path = inputs.regular(source.joinpath(*part.parts))
        inputs.require(inputs.sha(path) == value, 'Report engine file changed: ' + name)
    return manifest


def tree_bytes(root):
    total = 0
    root = Path(root)
    if not root.exists():
        return 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                inputs.require(path.resolve(strict=True).is_relative_to(root.resolve()), 'External owned link')
                total += info.st_size
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
            else:
                inputs.require(stat.S_ISDIR(info.st_mode), 'Owned special file')
    return total


def capacity(profile, parent, phase):
    """Adopt the exact full current guard, then add this tool's own allowance."""
    guard = profile['capacity_guard']
    root = Path(guard['source'])
    source_manifest = root / 'source_manifest.json'
    inputs.require(inputs.sha(inputs.regular(source_manifest)) == guard['manifest_sha256'], 'Capacity source pin changed')
    manifest = inputs.load(source_manifest)
    for name, value in manifest['files'].items():
        part = PurePosixPath(name)
        inputs.require(not part.is_absolute() and '..' not in part.parts and '\\' not in name, 'Unsafe capacity source')
        inputs.require(inputs.sha(inputs.regular(root.joinpath(*part.parts))) == value, 'Capacity source bytes changed')
    path = root / guard['module']
    spec = importlib.util.spec_from_file_location('_quick_report_current_capacity', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    inherited = Path(parent) / ('full-capacity-' + phase)
    inherited.mkdir(exist_ok=False)
    prior = module.storage(inherited, 'entry')
    inputs.require(prior['status'] == 'passed' and prior['reserve_bytes'] == 4*GIB
                   and prior['file_reserve'] == 100000 and prior['forecasts_removed'] is False
                   and prior['original_gate_changed'] is False, 'Full existing futures/reserves did not pass')
    charged = sum(tree_bytes(p) if Path(p).is_dir() else Path(p).stat().st_size
                  for p in [profile['remote_root'], *profile.get('additional_owned_paths', [])] if Path(p).exists())
    allowance = profile['request_cap_bytes'] if phase == 'render' else profile['validation_cap_bytes']
    gate = {'status': 'failed', 'inherited': prior, 'required_bytes': prior['required_bytes'] + allowance,
            'available_bytes': prior['available_bytes'] - charged, 'owned_bytes_charged': charged,
            'independent_allowance_bytes': allowance, 'reserve_bytes': 4*GIB, 'file_reserve': 100000,
            'forecasts_removed': False, 'original_gate_changed': False}
    try:
        inputs.require(gate['available_bytes'] >= gate['required_bytes'], 'Full campaign plus report allowance does not fit')
        gate['status'] = 'passed'
    finally:
        write(Path(parent) / ('capacity-' + phase + '.json'), gate)
    return gate


def bsub_argv(profile, source, pin, phase, parent, request):
    inputs.require(phase in ('validate', 'render'), 'Report tool cannot submit model or optimizer phases')
    inputs.require(re.fullmatch('[a-f0-9]{64}', pin or '') is not None, 'Malformed engine pin')
    parent = Path(parent)
    return ['bsub', '-q', 'hpc', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]',
            '-W', '01:00', '-J', 'mosaic-quick-' + phase + '-' + parent.name[-16:],
            '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log'),
            profile['python'], '-B', '-u', str(Path(source) / 'scripts/quick_runtime.py'), phase,
            '--source', str(source), '--pin', pin, '--request', str(request)]


def submit(profile, source, pin, phase, parent, request):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    claim = parent / 'submission-started.json'
    args = bsub_argv(profile, source, pin, phase, parent, request)
    write(claim, {'status': 'unknown_submission_outcome', 'argv': args, 'engine_pin': pin})
    process = subprocess.run(args, capture_output=True, text=True, timeout=60)
    row = {'argv': args, 'engine_pin': pin, 'returncode': process.returncode,
           'stdout': process.stdout, 'stderr': process.stderr}
    jobs = re.findall(r'Job <(\d+)> is submitted to queue <hpc>\.', process.stdout)
    inputs.require(process.returncode == 0 and len(jobs) == 1, 'Unknown/failed report submission; inspect marker, never blindly replay')
    row.update(status='submitted', job_id=jobs[0])
    write(parent / 'submission.json', row)
    return row


def terminal_evidence(parent):
    parent = Path(parent)
    sub = inputs.load(parent / 'submission.json')
    job = sub['job_id']
    stored = parent / 'terminal-scheduler.json'
    if stored.exists():
        scheduler = inputs.load(stored)
    else:
        process = subprocess.run(['bjobs', '-l', job], capture_output=True, text=True, timeout=30)
        scheduler = {'job_id': job, 'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
        compact = ''.join(process.stdout.split())
        inputs.require(process.returncode == 0 and re.findall(r'Job<(\d+)>', compact) == [job]
                       and re.findall(r'Status<([^>]+)>', compact) == ['DONE'], 'Actual report job has not completed successfully')
        write(stored, scheduler)
    compact = ''.join(scheduler['stdout'].split())
    inputs.require(scheduler['returncode'] == 0 and re.findall(r'Job<(\d+)>', compact) == [job]
                   and re.findall(r'Status<([^>]+)>', compact) == ['DONE'] and 'Queue<hpc>' in compact,
                   'Authenticated actual terminal scheduler proof is missing')
    stdout = parent / ('stdout.' + job + '.log')
    inputs.check_lsf(stdout, job, 'passed')
    return [inputs.binding(stored), inputs.binding(stdout), inputs.binding(parent / 'submission.json')]


def certificate(source, pin):
    root = Path(source).parent
    path = root / 'validation-certificate.json'
    cert = inputs.load(path)
    inputs.require(cert['status'] == 'passed' and cert['engine_pin'] == pin, 'Validated engine certificate missing')
    for row in cert['proofs']:
        inputs.verify_binding(row)
    return cert


def certify(source, pin):
    manifest = verify_source(source, pin)
    root = Path(source).parent
    path = root / 'validation-certificate.json'
    if path.exists():
        return certificate(source, pin)
    phase = root / 'validation'
    proofs = terminal_evidence(phase)
    assessment = inputs.load(phase / 'assessment.json')
    inputs.require(assessment['status'] == 'passed' and assessment['engine_pin'] == pin
                   and set(assessment['modules']) == set(manifest['test_modules']), 'Engine CPU assessment/attribution differs')
    cases = ET.parse(phase / 'tests.xml').getroot().findall('.//testcase')
    inputs.require(len(cases) == assessment['cases'] == assessment['collected_cases']
                   and all(not list(c) for c in cases), 'Raw complete CPU suite failed or skipped cases')
    proofs += [inputs.binding(phase / n) for n in ('assessment.json', 'tests.xml', 'tests.log', 'collection.txt', 'command.json')]
    cert = {'status': 'passed', 'engine_pin': pin, 'cpu_job_id': inputs.load(phase / 'submission.json')['job_id'],
            'cases': len(cases), 'modules': assessment['modules'], 'proofs': proofs}
    write(path, cert)
    return cert


def validate(source, pin, profile, phase):
    manifest = verify_source(source, pin)
    capacity(profile, phase, 'validate')
    tests = [str(Path(source) / p) for p in manifest['test_modules']]
    flags = ['--rootdir', str(source), '--import-mode=importlib', '-o', 'addopts=',
             '-o', 'cache_dir=' + str(phase / 'cache'), '--basetemp', str(Path(os.environ['TMPDIR']) / 'quick-report-tests')]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(source) / 'scripts')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    collect = [profile['python'], '-m', 'pytest', *tests, *flags, '--collect-only', '-q']
    result = subprocess.run(collect, cwd=source, env=env, capture_output=True, text=True, timeout=300)
    (phase / 'collection.txt').write_text(result.stdout + result.stderr)
    inputs.require(result.returncode == 0, 'Full engine test collection failed')
    nodes = [s for s in result.stdout.splitlines() if '::' in s and not s.startswith(' ')]
    inputs.require(nodes and len(set(nodes)) == len(nodes), 'Missing/duplicate collected CPU tests')
    args = [profile['python'], '-m', 'pytest', *tests, *flags, '--junitxml', str(phase / 'tests.xml'), '-q']
    write(phase / 'command.json', {'argv': args, 'collection_argv': collect})
    with (phase / 'tests.log').open('x') as output:
        subprocess.run(args, cwd=source, env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=2400)
    cases = ET.parse(phase / 'tests.xml').getroot().findall('.//testcase')
    inputs.require(len(cases) == len(nodes) and all(not list(c) for c in cases), 'Complete engine CPU tests did not pass')
    inputs.require(tree_bytes(phase) <= profile['validation_cap_bytes'], 'Engine validation output cap exceeded')
    write(phase / 'assessment.json', {'status': 'passed', 'engine_pin': pin, 'cases': len(cases),
                                     'collected_cases': len(nodes), 'modules': manifest['test_modules']})


def assets(arm):
    return [arm['scalars'], *arm['frames']['chunks']]


def registered_seeds(profile):
    """Only actual terminal, certificate-bound earlier requests become caches."""
    index = Path(profile['remote_root']) / 'cache-index.jsonl'
    if not index.exists():
        return []
    rows = [json.loads(line) for line in index.read_text().splitlines() if line]
    seeds = []
    for row in reversed(rows[-128:]):
        parent = Path(row['parent'])
        source = Path(row['source'])
        inputs.require(parent.parent == Path(profile['remote_root']) / 'requests'
                       and source.parent.parent == Path(profile['remote_root']) / 'engines', 'Cache namespace escaped report root')
        verify_source(source, row['engine_pin'])
        cert = certificate(source, row['engine_pin'])
        for proof in row['terminal_proofs']:
            inputs.verify_binding(proof)
        scheduler = inputs.load(parent / 'terminal-scheduler.json')
        compact = ''.join(scheduler['stdout'].split())
        inputs.require(scheduler['returncode'] == 0 and re.findall(r'Job<(\d+)>', compact) == [row['job_id']]
                       and re.findall(r'Status<([^>]+)>', compact) == ['DONE'], 'Cached request was not an actual terminal success')
        inputs.check_lsf(parent / ('stdout.' + row['job_id'] + '.log'), row['job_id'], 'passed')
        root = parent / 'viewer'
        inputs.require(inputs.sha(root / 'report-receipt.json') == row['receipt_sha256'], 'Registered cache receipt changed')
        receipt = inputs.load(root / 'report-receipt.json')
        inputs.require(receipt['status'] == 'passed_saved_only_presentation' and receipt['engine_pin'] == row['engine_pin']
                       and receipt['cpu_job_id'] == cert['cpu_job_id']
                       and inputs.sha(root / 'cpu-certificate.json') == receipt['cpu_certificate_sha256'], 'Registered cache provenance differs')
        for name in ('manifest.json', 'input-plan.json'):
            value = receipt['files'][name]
            inputs.require((root / name).stat().st_size == value['bytes'] and inputs.sha(root / name) == value['sha256'], 'Registered cache plan/manifest changed')
        seeds.append({'root': root, 'receipt': receipt, 'manifest': inputs.load(root / 'manifest.json'),
                      'plan': inputs.load(root / 'input-plan.json'),
                      'renderer_sha256': inputs.sha(source / 'scripts/report_data.py')})
    return seeds


def register_cache(profile, source, pin, parent):
    import fcntl
    parent = Path(parent)
    proofs = terminal_evidence(parent)
    certificate(source, pin)
    delivery = inputs.load(parent / 'delivery.json')
    row = {'parent': str(parent), 'source': str(source), 'engine_pin': pin,
           'job_id': delivery['job_id'], 'receipt_sha256': delivery['receipt_sha256'], 'terminal_proofs': proofs}
    lock = Path(profile['remote_root']) / 'cache-index.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        index = Path(profile['remote_root']) / 'cache-index.jsonl'
        old = [json.loads(line) for line in index.read_text().splitlines() if line] if index.exists() else []
        matching = [value for value in old if value['parent'] == str(parent)]
        if matching:
            inputs.require(matching == [row], 'Existing cache registration differs')
        else:
            with index.open('ab') as output:
                output.write(inputs.encoded(row))
                output.flush()
                os.fsync(output.fileno())


def capture_selection(selection, profile):
    return inputs.capture(selection, profile, registered_seeds(profile))


def cached_arm(arm, renderer_sha, seeds, output):
    key = inputs.science_key(arm, renderer_sha)
    for seed in seeds:
        if seed['renderer_sha256'] != renderer_sha:
            continue
        plans = {a['id']: a for a in seed['plan']['arms']}
        for saved in seed['manifest']['arms']:
            original = plans[saved['id']]
            if inputs.science_key(original, renderer_sha) != key:
                continue
            for row in assets(saved):
                part = PurePosixPath(row['path'])
                inputs.require(len(part.parts) == 1 and not part.is_absolute() and '\\' not in row['path'], 'Unsafe cached report asset path')
                path = seed['root'] / row['path']
                expected = seed['receipt']['files'][row['path']]
                inputs.require(row['bytes'] == expected['bytes'] and row['sha256'] == expected['sha256'], 'Cached asset binding disagrees with validated receipt')
                inputs.require(path.stat().st_size == row['bytes'] and inputs.sha(path) == row['sha256'], 'Cached validated asset bytes changed')
                shutil.copyfile(path, Path(output) / row['path'])
            value = deepcopy(saved)
            for name in ('label', 'status_detail', 'continuation_status', 'primary_metric'):
                if name in arm:
                    value[name] = arm[name]
            return value
    return None


def verify_originals(plan, manifest, output):
    """Cache hits still authenticate every original input and NPZ byte binding."""
    import report_data as data
    for arm in plan['arms']:
        for segment in arm['segments']:
            for name in ('states', 'updates', 'config', 'summary', 'case_report', 'retained_index', 'stdout', 'stderr'):
                if name in segment:
                    data.bound_file(segment[name])
    for arm in manifest['arms']:
        for chunk in arm['frames']['chunks']:
            for frame in data.load(Path(output) / chunk['path'])['frames']:
                data.bound_file(frame['source_snapshot'])


def render_plan(plan, output, renderer_sha, seeds):
    import report_data as data
    output = Path(output)
    output.mkdir(exist_ok=False)
    inputs.require(1 <= len(plan['arms']) <= inputs.MAX_RUNS and len({a['id'] for a in plan['arms']}) == len(plan['arms']), 'Invalid report cohort')
    values, hits, fresh = [], [], []
    for arm in plan['arms']:
        value = cached_arm(arm, renderer_sha, seeds, output)
        if value is None:
            value = data.collect_arm(arm, output)
            fresh.append(arm['id'])
        else:
            hits.append(arm['id'])
        value['primary_metric'] = arm['primary_metric']
        expected = arm['expected_counts']
        inputs.require((value['committed_updates'], value['scalars']['count'], value['frames']['count'], value['latest_global_step']) ==
                       (expected['committed_updates'], expected['scalar_states'], expected['retained_frames'], expected['latest_global_step']),
                       'Captured/rendered saved counts differ')
        inputs.require(value['scientific_status'] == arm['scientific_status'], 'Cache relabeled a scientific outcome')
        values.append(value)
    manifest = {'version': data.VERSION, 'edition': plan['edition'], 'captured_at': plan['captured_at'], 'arms': values,
                'interpretation': 'Saved-only presentation; scientific validation outcomes are unchanged'}
    verify_originals(plan, manifest, output)
    write(output / 'manifest.json', manifest)
    write(output / 'assessment.json', {'status': 'passed_saved_only_render', 'edition': plan['edition'],
          'cached_arms': hits, 'newly_validated_arms': fresh, 'scientific_success_claim': False,
          'original_results_changed': False, 'all_original_input_and_snapshot_hashes_rechecked': True,
          'all_scalar_records_retained': True, 'no_interpolation': True})
    return manifest


def render(source, pin, profile, request, phase):
    verify_source(source, pin)
    cert = certificate(source, pin)
    capacity(profile, phase, 'render')
    started = time.monotonic()
    renderer_sha = inputs.sha(Path(source) / 'scripts/report_data.py')
    inputs.require(renderer_sha == profile['validated_renderer_sha256'], 'Cached scientific renderer identity changed')
    prior_seeds = registered_seeds(profile)
    plan = inputs.capture(request['selection'], profile, prior_seeds)
    inputs.require(inputs.selection_key(plan['arms'], renderer_sha) == request['selection_key'],
                   'Saved inputs changed after request capture; make a fresh report request')
    write(phase / 'captured-plan.json', plan)
    seeds, _ = inputs.seed_catalog(profile)
    seeds = prior_seeds + seeds
    output = phase / 'viewer'
    manifest = render_plan(plan, output, renderer_sha, seeds)
    for src, name in [(Path(source) / 'templates/report.html', 'report.html'),
                      (Path(source).parent / 'validation-certificate.json', 'cpu-certificate.json'),
                      (phase / 'captured-plan.json', 'input-plan.json')]:
        shutil.copyfile(src, output / name)
    receipt = {'status': 'passed_saved_only_presentation', 'engine_pin': pin, 'job_id': os.environ['LSB_JOBID'],
               'cpu_job_id': cert['cpu_job_id'], 'cpu_certificate_sha256': inputs.sha(output / 'cpu-certificate.json'),
               'renderer_sha256': renderer_sha, 'selection_key': inputs.selection_key(plan['arms'], renderer_sha),
               'files': {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': inputs.sha(p)}
                         for p in sorted(output.iterdir()) if p.is_file()},
               'scientific_success_claim': False, 'original_results_changed': False}
    write(output / 'report-receipt.json', receipt)
    known = set(request.get('known_asset_sha256', []))
    archive = phase / 'report-delta.tar.gz'
    omitted = {}
    with tarfile.open(archive, 'x:gz') as bundle:
        for path in sorted(output.iterdir()):
            if path.name not in ('manifest.json', 'report.html', 'input-plan.json', 'assessment.json', 'report-receipt.json', 'cpu-certificate.json') and inputs.sha(path) in known:
                omitted[path.name] = receipt['files'][path.name]
            else:
                bundle.add(path, arcname=path.name, recursive=False)
    inputs.require(tree_bytes(phase) <= profile['request_cap_bytes'], 'Report output exceeded its declared cap')
    write(phase / 'delivery.json', {'status': 'passed_saved_only_presentation', 'engine_pin': pin,
          'job_id': os.environ['LSB_JOBID'], 'edition': plan['edition'], 'archive': inputs.binding(archive),
          'receipt_sha256': inputs.sha(output / 'report-receipt.json'), 'omitted_verified_local_assets': omitted,
          'elapsed_seconds': time.monotonic() - started, 'scientific_success_claim': False,
          'arms': {a['id']: {'updates': a['committed_updates'], 'frames': a['frames']['count']} for a in manifest['arms']}})


def control(action, payload, source, pin):
    """Login-node metadata orchestration; scientific work has no control branch."""
    source = Path(source)
    profile = inputs.load(source / 'profile.json')
    if action == 'setup':
        import fcntl
        parent = source.parent / 'validation'
        parent.mkdir(exist_ok=True)
        with (source.parent / 'setup.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if (source.parent / 'validation-certificate.json').exists():
                return {'state': 'validated', 'certificate': certificate(source, pin)}
            if (parent / 'submission.json').exists():
                return {'state': 'submitted', 'submission': inputs.load(parent / 'submission.json')}
            inputs.require(not (parent / 'submission-started.json').exists(), 'Unknown validation submission: inspect before recovery')
            request = parent / 'request.json'
            expected = {'phase': 'validate', 'engine_pin': pin}
            if request.exists():
                inputs.require(inputs.load(request) == expected, 'Validation request identity changed')
            else:
                write(request, expected)
            admission = parent / ('admission-' + uuid.uuid4().hex)
            admission.mkdir()
            try:
                capacity(profile, admission, 'preflight')
            except Exception as error:
                inputs.require(not (parent / 'submission-started.json').exists(), 'Unknown concurrent submission')
                return {'state': 'blocked_before_submission', 'error': repr(error), 'evidence': str(admission), 'no_jobs_submitted': True}
            return {'state': 'submitted', 'submission': submit(profile, source, pin, 'validate', parent, request)}
    if action == 'certify':
        return certify(source, pin)
    if action == 'capture':
        return capture_selection(payload, profile)
    if action in ('submit', 'recover_submission'):
        inputs.require(re.fullmatch(r'request-[0-9TZ]+-[a-f0-9]{8}', payload['request_id']) is not None, 'Unsafe request identity')
        parent = Path(profile['remote_root']) / 'requests' / payload['request_id']
        request = parent / 'request.json'
        certificate(source, pin)
        if action == 'recover_submission':
            inputs.require(inputs.load(request) == payload, 'Original request identity changed')
            result = inputs.load(parent / 'submission.json')
            inputs.require(result['engine_pin'] == pin and result['argv'] == bsub_argv(profile, source, pin, 'render', parent, request), 'Recovered submission command changed')
            return result
        parent.mkdir(parents=True, exist_ok=False)
        write(request, payload)
        return submit(profile, source, pin, 'render', parent, request)
    inputs.require(action in ('status', 'delivery'), 'Unknown report control action')
    parent = Path(payload['parent'])
    inputs.require(parent.is_relative_to(Path(profile['remote_root'])), 'Report control namespace escaped')
    if action == 'status':
        sub = inputs.load(parent / 'submission.json')
        process = subprocess.run(['bjobs', '-l', sub['job_id']], capture_output=True, text=True, timeout=30)
        states = re.findall(r'Status<([^>]+)>', ''.join(process.stdout.split()))
        result = {'job_id': sub['job_id'], 'returncode': process.returncode, 'state': states[0] if len(states) == 1 else 'UNKNOWN',
                  'stdout': process.stdout, 'stderr': process.stderr}
        if result['state'] == 'EXIT':
            result['job_stderr'] = (parent / ('stderr.' + sub['job_id'] + '.log')).read_text()[-20000:]
        return result
    import base64
    proofs = terminal_evidence(parent)
    register_cache(profile, source, pin, parent)
    cert = certificate(source, pin)
    proofs += cert['proofs'] + [inputs.binding(source.parent / 'validation-certificate.json'),
                              inputs.binding(parent / 'delivery.json'), inputs.binding(parent / 'capacity-render.json')]
    result = {'delivery': inputs.load(parent / 'delivery.json'), 'certificate': cert, 'proof_files': []}
    for row in proofs:
        path = Path(row['path'])
        inputs.require(path.is_relative_to(Path(profile['remote_root'])), 'Evidence path escaped report namespace')
        raw = path.read_bytes()
        inputs.require(len(raw) < 4*1024**2, 'Evidence file exceeds metadata cap')
        result['proof_files'].append({**row, 'base64': base64.b64encode(raw).decode()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['validate', 'render'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(argv)
    verify_source(args.source, args.pin)
    profile = inputs.load(args.source / 'profile.json')
    request = inputs.load(args.request)
    phase = args.request.parent if args.phase == 'render' else args.source.parent / 'validation'
    for _ in range(30):
        if (phase / 'submission.json').exists():
            break
        time.sleep(1)
    sub = inputs.load(phase / 'submission.json')
    inputs.require(sub['job_id'] == os.environ.get('LSB_JOBID') and os.environ.get('LSB_QUEUE') == 'hpc'
                   and os.environ.get('LSB_DJOB_NUMPROC') == '4'
                   and sub['argv'] == bsub_argv(profile, args.source, args.pin, args.phase, phase, args.request),
                   'Actual allocated report command/resources do not match submission')
    if args.phase == 'validate':
        validate(args.source, args.pin, profile, phase)
    else:
        render(args.source, args.pin, profile, request, phase)


if __name__ == '__main__':
    main()
