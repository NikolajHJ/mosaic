"""One-command saved-run report: discover, validate, render, transfer and publish.

Local execution is standard-library metadata/bytes only. Scientific saved-frame
validation/rendering always executes in an authenticated allocated HPC CPU job.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote
from urllib.request import urlopen
import uuid
import xml.etree.ElementTree as ET

import quick_inputs as inputs

OWNER = Path(__file__).resolve().parents[1]
REPO = OWNER.parents[2]
LOCAL_ONLY_PROFILE = {'base_archive', 'default_output', 'artifact_root', 'port', 'ssh_host', 'ledger', 'run_journal'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(inputs.encoded(value))


def record(path, value):
    """Idempotent evidence write; an existing different outcome is never replaced."""
    path = Path(path)
    if path.exists():
        inputs.require(path.read_bytes() == inputs.encoded(value), 'Existing evidence differs: ' + str(path))
    else:
        write(path, value)


def profile_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def validate_profile(profile):
    inputs.require(profile.get('version') == 'mosaic-report-profile-v1', 'Wrong report connection profile')
    inputs.require(re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.@-]*', profile.get('ssh_host', '')) is not None, 'Invalid SSH alias')
    for name in ('python', 'remote_root'):
        path = PurePosixPath(profile[name])
        inputs.require(path.is_absolute() and '..' not in path.parts and '\\' not in str(path), 'Require canonical remote ' + name)
    inputs.require(profile.get('allowed_roots') and all(PurePosixPath(p).is_absolute() and len(PurePosixPath(p).parts) >= 3
                   and '..' not in PurePosixPath(p).parts for p in profile['allowed_roots']), 'Declare specific authorized project roots')
    for name in ('request_cap_bytes', 'validation_cap_bytes'):
        inputs.require(type(profile[name]) is int and 0 < profile[name] <= 1024**3, 'Report caps must be positive and at most1GiB')


def event(profile, work, status, **details):
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'request_id': Path(work).name, 'status': status,
           'evidence': str(work), 'scientific_results_changed': False, **details}
    for path in [Path(work) / 'events.jsonl', *([profile_path(profile['run_journal'])] if profile.get('run_journal') else [])]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab') as stream:
            stream.write(inputs.encoded(row))
    if profile.get('ledger'):
        text = '\n\n### ' + row['utc'] + ' — Reusable saved report: ' + status + '\n\n'
        text += 'Request `' + row['request_id'] + '`; ' + json.dumps(details, sort_keys=True) + '.\n'
        text += 'Evidence `' + str(work) + '`. Report-only metadata/allocated CPU work; scientific settings/results unchanged.\n'
        with profile_path(profile['ledger']).open('a', encoding='utf8', newline='\n') as stream:
            stream.write(text)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, help='Small JSON selection, including ordered continuation segments')
    p.add_argument('--run', action='append', default=[], help='Saved study/job path or a name from the default selection; repeat as needed')
    p.add_argument('--profile', type=Path, default=OWNER / 'configs/dtu-current.json')
    p.add_argument('--output', type=Path, help='Existing stable viewer or a new report directory')
    p.add_argument('--port', type=int)
    p.add_argument('--resume', type=Path, help='Resume this exact recorded local request; never resubmit an unknown outcome')
    p.add_argument('--setup-only', action='store_true', help='Install/validate the engine once without rendering data')
    p.add_argument('--force', action='store_true', help='Create a fresh edition even when the saved inputs are unchanged')
    p.add_argument('--no-serve', action='store_true', help='Publish files without starting/checking localhost HTTP')
    p.add_argument('--preview', action='store_true', help='New-engine UI qualification preview; does not waive data checks')
    p.add_argument('--dry-run', action='store_true', help='Inspect selection/configuration only, without SSH or submission')
    return p


def selection_from_args(args):
    default = inputs.load(OWNER / 'configs/current-runs.json') if (OWNER / 'configs/current-runs.json').exists() else {'runs': []}
    if args.runs:
        selection = inputs.load(args.runs)
    elif args.run:
        named = {a['id']: a for a in default['runs']}
        selection = {'version': inputs.VERSION, 'runs': [named.get(s, s) for s in args.run]}
    else:
        selection = default
    inputs.require(selection.get('version') == inputs.VERSION and isinstance(selection.get('runs'), list)
                   and 1 <= len(selection['runs']) <= inputs.MAX_RUNS, 'Choose between1 and24 runs')
    return selection


def package_engine(profile):
    """The code/fixture/profile hash stays identical across ordinary data updates."""
    base = profile['base_archive']
    path = profile_path(base['path'])
    inputs.require(path.stat().st_size == base['bytes'] and inputs.sha(path) == base['sha256'], 'Validated base source archive is missing or changed')
    members = {}
    with tarfile.open(path) as bundle:
        for member in bundle.getmembers():
            name = PurePosixPath(member.name)
            inputs.require(member.isfile() and not name.is_absolute() and '..' not in name.parts
                           and '\\' not in member.name and member.name not in members, 'Unsafe base source archive')
            members[member.name] = bundle.extractfile(member).read()
    inputs.require(hashlib.sha256(members['source_manifest.json']).hexdigest() == base['manifest_sha256'], 'Base manifest pin changed')
    manifest = json.loads(members['source_manifest.json'])
    inputs.require(set(members) == set(manifest['files']) | {'source_manifest.json'}, 'Base source member set differs')
    for name, value in manifest['files'].items():
        inputs.require(hashlib.sha256(members[name]).hexdigest() == value, 'Base scientific source changed')
    members.pop('source_manifest.json')
    members.pop('plan.json', None)
    # Old observer/orchestration code remains historical; the reusable runtime is
    # the sole submitted entry point. Keep report_ops only for its existing tests.
    members.pop('scripts/observe.py', None)
    for name in ('scripts/quick_inputs.py', 'scripts/quick_runtime.py', 'scripts/report.py',
                 'scripts/publish.py', 'tests/test_quick_report.py', 'tests/test_viewer.py', 'templates/report.html'):
        members[name] = (OWNER / name).read_bytes().replace(b'\r\n', b'\n')
    runtime_profile = {k: v for k, v in profile.items() if k not in LOCAL_ONLY_PROFILE}
    members['profile.json'] = inputs.encoded(runtime_profile)
    members['README.md'] = b'# Reusable saved-report engine\n\nCode and fixtures are validated once per exact engine hash. Run selections are separate requests. No model or optimizer phase exists.\n'
    for name, raw in members.items():
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
    engine = {'version': 'mosaic-reusable-report-engine-v1',
              'files': {n: hashlib.sha256(b).hexdigest() for n, b in sorted(members.items())},
              'test_modules': sorted(n for n in members if n.startswith('tests/test_') and n.endswith('.py')),
              'scientific_renderer_sha256': hashlib.sha256(members['scripts/report_data.py']).hexdigest()}
    inputs.require(engine['scientific_renderer_sha256'] == profile['validated_renderer_sha256'], 'Scientific renderer must retain its validated exact bytes')
    members['engine-manifest.json'] = inputs.encoded(engine)
    pin = hashlib.sha256(members['engine-manifest.json']).hexdigest()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as bundle:
            for name, raw in sorted(members.items()):
                item = tarfile.TarInfo(name)
                item.size, item.mode, item.mtime = len(raw), 0o444, 0
                bundle.addfile(item, io.BytesIO(raw))
    return pin, engine, buffer.getvalue()


class Connection:
    def __init__(self, profile, work):
        self.profile, self.work = profile, Path(work)
        self.counter = len(list(self.work.glob('ssh-*.json')))

    def args(self, command):
        return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
                self.profile['ssh_host'], 'bash -l -c ' + shlex.quote(command)]

    def run(self, code, label, timeout=300):
        ast.parse(code, filename=label + '-remote.py')
        self.counter += 1
        root = self.work / ('ssh-' + str(self.counter).zfill(4) + '-' + label)
        args = self.args(shlex.quote(self.profile['python']) + ' -')
        receipt = {'status': 'unknown', 'label': label, 'program_sha256': hashlib.sha256(code.encode()).hexdigest()}
        write(root.with_suffix('.json'), receipt)
        root.with_suffix('.py').write_text(code, encoding='utf8', newline='\n')
        try:
            with root.with_suffix('.stdout').open('xb') as output, root.with_suffix('.stderr').open('xb') as error:
                result = subprocess.run(args, input=code.encode(), stdout=output, stderr=error, timeout=timeout)
            receipt.update(status='returned', returncode=result.returncode)
            inputs.require(result.returncode == 0, 'HPC action failed; inspect ' + str(root.with_suffix('.stderr')))
            return json.loads(root.with_suffix('.stdout').read_bytes())
        finally:
            receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
            root.with_suffix('.json').write_bytes(inputs.encoded(receipt))

    def action(self, source, pin, action, payload=None, timeout=300):
        # Authenticate all code using only stdlib before importing the runtime.
        code = '''import json,hashlib,pathlib,sys\nsource=pathlib.Path(SOURCE)\npin=PIN\nraw=(source/'engine-manifest.json').read_bytes()\nassert hashlib.sha256(raw).hexdigest()==pin\nm=json.loads(raw)\nfor name,digest in m['files'].items():\n p=pathlib.PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and '\\\\' not in name\n f=source.joinpath(*p.parts);assert f.is_file() and all(not q.is_symlink() for q in (f,*f.parents))\n assert hashlib.sha256(f.read_bytes()).hexdigest()==digest\nassert {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}==set(m['files'])|{'engine-manifest.json'}\nassert all(not p.is_symlink() for p in source.rglob('*'))\nsys.dont_write_bytecode=True\nsys.path.insert(0,str(source/'scripts'))\nimport quick_runtime as runtime,quick_inputs as inputs\nprofile=inputs.load(source/'profile.json')\npayload=PAYLOAD\n'''.replace('SOURCE', repr(str(source))).replace('PIN', repr(pin)).replace('PAYLOAD', repr(payload))
        inputs.require(action in ('setup', 'certify', 'capture', 'submit', 'recover_submission', 'status', 'delivery'), 'Unknown report action')
        return self.run(code + 'result=runtime.control(' + repr(action) + ',payload,source,pin)\nprint(json.dumps(result,allow_nan=False))\n', action, timeout)


def install(connection, profile, pin, archive):
    source = str(PurePosixPath(profile['remote_root']) / 'engines' / pin / 'source')
    payload = base64.b64encode(archive).decode()
    code = '''import base64,hashlib,io,json,os,pathlib,tarfile\nsource=pathlib.Path(SOURCE)\npin=PIN\nif source.exists():\n assert hashlib.sha256((source/'engine-manifest.json').read_bytes()).hexdigest()==pin\nelse:\n source.parent.mkdir(parents=True,exist_ok=False)\n with (source.parent/'install-started.json').open('x') as f:json.dump({'status':'unknown','pin':pin},f)\n source.mkdir()\n with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE)),mode='r:gz') as t:\n  names=set()\n  for m in t.getmembers():\n   p=pathlib.PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and '\\\\' not in m.name and m.name not in names\n   names.add(m.name);target=source.joinpath(*p.parts);target.parent.mkdir(parents=True,exist_ok=True)\n   with target.open('xb') as f:f.write(t.extractfile(m).read())\n   os.chmod(target,0o444)\n assert hashlib.sha256((source/'engine-manifest.json').read_bytes()).hexdigest()==pin\n for p in sorted((p for p in source.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):os.chmod(p,0o555)\n os.chmod(source,0o555)\nprint(json.dumps({'source':str(source),'pin':pin}))\n'''.replace('SOURCE', repr(source)).replace('PIN', repr(pin)).replace('ARCHIVE', repr(payload))
    connection.run(code, 'install')
    return source


def wait(connection, source, pin, parent, *, timeout=7200):
    started = time.monotonic()
    previous = None
    while time.monotonic() - started < timeout:
        value = connection.action(source, pin, 'status', {'parent': str(parent)})
        state = value['state']
        if state != previous:
            print('Report job ' + value['job_id'] + ': ' + state, flush=True)
            event(connection.profile, connection.work, 'observed_' + state, job_id=value['job_id'], scheduler_only=True)
            previous = state
        if state == 'DONE':
            return value
        inputs.require(state in ('PEND', 'RUN'), 'Report job failed/unavailable; inspect recorded evidence, do not resubmit blindly: ' + json.dumps(value))
        time.sleep(15)
    raise TimeoutError('Report wait timed out; resume this recorded request without resubmitting')


def local_assets(viewer):
    """Only previously published, individually verified local assets can be reused."""
    result = {}
    for receipt_path in sorted(Path(viewer).glob('editions/*/report-receipt.json')):
        receipt = inputs.load(receipt_path)
        for name, row in receipt.get('files', {}).items():
            part = PurePosixPath(name)
            inputs.require(not part.is_absolute() and '..' not in part.parts and '\\' not in name, 'Unsafe local receipt member')
            path = receipt_path.parent.joinpath(*part.parts)
            if path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes'] and inputs.sha(path) == row['sha256']:
                result.setdefault(row['sha256'], path)
    return result


def current_plan(viewer):
    latest = Path(viewer) / 'latest.json'
    if not latest.exists():
        return None
    pointer = inputs.load(latest)
    inputs.require(re.fullmatch('edition-[0-9]{4}', pointer['edition']) is not None, 'Unknown latest edition identity')
    root = Path(viewer) / 'editions' / pointer['edition']
    receipt = inputs.load(root / 'report-receipt.json')
    for name, row in receipt['files'].items():
        part = PurePosixPath(name)
        inputs.require(not part.is_absolute() and '..' not in part.parts and '\\' not in name, 'Unsafe current report receipt')
        path = inputs.regular(root.joinpath(*part.parts))
        inputs.require(path.stat().st_size == row['bytes'] and inputs.sha(path) == row['sha256'], 'Published report asset changed: ' + name)
    inputs.require(inputs.sha(root / 'manifest.json') == pointer['manifest_sha256'], 'Latest pointer manifest differs')
    return inputs.load(root / 'input-plan.json')


def next_edition(viewer):
    values = [int(p.name[-4:]) for p in (Path(viewer) / 'editions').glob('edition-[0-9][0-9][0-9][0-9]')]
    return 'edition-' + str(max(values, default=0) + 1).zfill(4)


def download(connection, row, path):
    offset = path.stat().st_size if path.exists() else 0
    inputs.require(offset <= row['bytes'], 'Existing transfer is longer than its authenticated source')
    prefix = inputs.sha(path) if path.exists() else hashlib.sha256(b'').hexdigest()
    if offset == row['bytes'] and prefix == row['sha256']:
        record(path.with_suffix('.complete.json'), {'status': 'passed_exact_bytes', 'binding': row})
        return
    marker = path.with_suffix('.transfer.json')
    record(marker, {'status': 'unknown', 'remote_binding': row})
    code = "import pathlib,hashlib,sys\np=pathlib.Path(sys.argv[1]);assert p.is_file() and not p.is_symlink()\nwith p.open('rb') as f:\n assert p.stat().st_size==int(sys.argv[2]) and hashlib.file_digest(f,'sha256').hexdigest()==sys.argv[3]\n f.seek(0);remaining=int(sys.argv[4]);h=hashlib.sha256()\n while remaining:\n  b=f.read(min(1048576,remaining));assert b;h.update(b);remaining-=len(b)\n assert h.hexdigest()==sys.argv[5]\n for b in iter(lambda:f.read(1048576),b''):sys.stdout.buffer.write(b)\n"
    args = connection.args(' '.join(shlex.quote(s) for s in [connection.profile['python'], '-c', code, row['path'], str(row['bytes']), row['sha256'], str(offset), prefix]))
    error_path = path.with_suffix('.stderr-' + str(offset))
    with path.open('ab' if offset else 'xb') as output, error_path.open('xb') as error:
        result = subprocess.run(args, stdout=output, stderr=error, timeout=300)
    inputs.require(result.returncode == 0 and path.stat().st_size == row['bytes'] and inputs.sha(path) == row['sha256'], 'Report delta transfer failed; preserve its partial and marker')
    record(path.with_suffix('.complete.json'), {'status': 'passed_exact_bytes', 'binding': row})


def verify_delivery(value, source, pin):
    delivery, cert = value['delivery'], value['certificate']
    inputs.require(delivery['status'] == 'passed_saved_only_presentation' and delivery['engine_pin'] == pin
                   and cert['status'] == 'passed' and cert['engine_pin'] == pin, 'Unqualified report delivery')
    raw = {}
    for row in value['proof_files']:
        data = base64.b64decode(row['base64'], validate=True)
        inputs.require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'Evidence byte mismatch')
        raw[row['path']] = data
    for row in cert['proofs']:
        inputs.require(row['path'] in raw and hashlib.sha256(raw[row['path']]).hexdigest() == row['sha256'], 'CPU certificate proof missing')
    xml = [data for path, data in raw.items() if path.endswith('/validation/tests.xml')]
    inputs.require(len(xml) == 1, 'Missing raw CPU XML')
    cases = ET.fromstring(xml[0]).findall('.//testcase')
    inputs.require(len(cases) == cert['cases'] and all(not list(c) for c in cases), 'Allocated engine tests failed/skipped')
    parents = [(str(PurePosixPath(source).parent / 'validation'), cert['cpu_job_id']),
               (str(PurePosixPath(delivery['archive']['path']).parent), delivery['job_id'])]
    for parent, job in parents:
        scheduler = json.loads(raw[parent + '/terminal-scheduler.json'])
        compact = ''.join(scheduler['stdout'].split())
        inputs.require(scheduler['returncode'] == 0 and re.findall(r'Job<(\d+)>', compact) == [job]
                       and re.findall(r'Status<([^>]+)>', compact) == ['DONE'] and 'Queue<hpc>' in compact,
                       'Actual same-job DONE/hpc proof missing')
        lsf = ''.join(raw[parent + '/stdout.' + job + '.log'].decode().split())
        inputs.require('Subject:Job' + job + ':' in lsf and lsf.count('Successfullycompleted.') == 1, 'Same-job report LSF success missing')
    return raw


def publish_delta(archive, delivery, viewer, inventory, engine_pin, cpu_certificate_sha256):
    """Reconstruct verified report files before an atomic immutable publication."""
    destination = Path(viewer) / 'editions' / delivery['edition']
    inputs.require(not destination.exists(), 'Immutable edition already exists')
    staging = destination.with_name('.staging-' + delivery['edition'])
    staging.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as bundle:
        names = set()
        inputs.require(sum(m.size for m in bundle.getmembers()) <= 1024**3, 'Report delta exceeds cap')
        for member in bundle.getmembers():
            part = PurePosixPath(member.name)
            inputs.require(member.isfile() and len(part.parts) == 1 and '..' not in part.parts
                           and '\\' not in member.name and member.name.casefold() not in names, 'Unsafe report delta member')
            names.add(member.name.casefold())
            with (staging / member.name).open('xb') as stream:
                shutil.copyfileobj(bundle.extractfile(member), stream)
    receipt_path = staging / 'report-receipt.json'
    inputs.require(inputs.sha(receipt_path) == delivery['receipt_sha256'], 'Delivered receipt pin differs')
    receipt = inputs.load(receipt_path)
    inputs.require(receipt['engine_pin'] == engine_pin and receipt['job_id'] == delivery['job_id'], 'Report source/job receipt differs')
    inputs.require({'manifest.json', 'report.html', 'input-plan.json', 'assessment.json', 'cpu-certificate.json'} <= set(receipt['files']),
                   'Report is missing required provenance/presentation files')
    omitted = delivery['omitted_verified_local_assets']
    for name, row in omitted.items():
        inputs.require(name in receipt['files'] and receipt['files'][name] == row
                       and len(PurePosixPath(name).parts) == 1 and '\\' not in name, 'Invalid omitted asset identity')
        path = inventory.get(row['sha256'])
        inputs.require(path is not None and path.stat().st_size == row['bytes'] and inputs.sha(path) == row['sha256'], 'Required verified local asset is missing or changed')
        shutil.copyfile(path, staging / name)
    expected = set(receipt['files']) | {'report-receipt.json'}
    inputs.require({p.name for p in staging.iterdir()} == expected, 'Delta reconstruction member set differs')
    for name, row in receipt['files'].items():
        path = staging / name
        inputs.require(path.stat().st_size == row['bytes'] and inputs.sha(path) == row['sha256'], 'Report member bytes differ: ' + name)
    inputs.require(inputs.sha(staging / 'cpu-certificate.json') == receipt['cpu_certificate_sha256'] == cpu_certificate_sha256,
                   'Embedded report CPU certificate differs from authenticated evidence')
    manifest = inputs.load(staging / 'manifest.json')
    inputs.require(manifest['edition'] == delivery['edition'] and 1 <= len(manifest['arms']) <= inputs.MAX_RUNS, 'Wrong published cohort/edition')
    inputs.require({a['id']: {'updates': a['committed_updates'], 'frames': a['frames']['count']} for a in manifest['arms']} == delivery['arms'], 'Rendered/captured arm counts differ')
    staging.rename(destination)
    import publish as legacy_publisher
    shell = Path(viewer) / 'report.html'
    if shell.exists():
        inputs.require(shell.read_text(encoding='utf8') in (legacy_publisher.LEGACY_WRAPPER, legacy_publisher.WRAPPER), 'Unknown existing stable viewer shell')
    else:
        shell.write_text(legacy_publisher.WRAPPER, encoding='utf8', newline='\n')
    pointer = {'version': 'jq1-live-publication-v1', 'edition': delivery['edition'],
               'report_url': 'editions/' + delivery['edition'] + '/report.html',
               'manifest_sha256': inputs.sha(destination / 'manifest.json'), 'engine_pin': engine_pin,
               'published_utc': datetime.now(timezone.utc).isoformat(), 'automatic_optimizer_submission': False}
    write(Path(viewer) / (delivery['edition'] + '-publication.json'), pointer)
    temporary = Path(viewer) / ('latest-' + delivery['edition'] + '.partial')
    write(temporary, pointer)
    temporary.replace(Path(viewer) / 'latest.json')
    return pointer


def ensure_http(viewer, port, work):
    with socket.socket() as sock:
        listening = sock.connect_ex(('127.0.0.1', port)) == 0
    if not listening:
        stdout = (work / 'server.stdout').open('xb')
        stderr = (work / 'server.stderr').open('xb')
        args = [sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1']
        process = subprocess.Popen(args, cwd=viewer, stdout=stdout, stderr=stderr,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        write(work / 'server.json', {'pid': process.pid, 'argv': args, 'cwd': str(viewer), 'hidden_on_windows': True})
        for _ in range(40):
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    break
            time.sleep(.1)
    pointer = inputs.load(Path(viewer) / 'latest.json')
    root = Path(viewer) / 'editions' / pointer['edition']
    files = [Path(viewer) / 'latest.json', Path(viewer) / 'report.html', *[p for p in root.iterdir() if p.is_file()]]
    for path in files:
        url = 'http://127.0.0.1:' + str(port) + '/' + quote(path.relative_to(viewer).as_posix())
        with urlopen(url, timeout=60) as response:
            digest = hashlib.sha256()
            length = 0
            for block in iter(lambda: response.read(1024**2), b''):
                length += len(block)
                digest.update(block)
        inputs.require(length == path.stat().st_size and digest.hexdigest() == inputs.sha(path), 'Localhost serves another directory or changed report bytes')
    return {'status': 'passed_all_new_edition_HTTP_bytes', 'files': len(files),
            'url': 'http://127.0.0.1:' + str(port) + '/report.html'}


def main(argv=None):
    started = time.monotonic()
    args = parser().parse_args(argv)
    profile = inputs.load(args.profile)
    if not args.dry_run:
        validate_profile(profile)
    if args.resume:
        work = args.resume.resolve()
        state = inputs.load(work / 'state.json')
        inputs.require(state['profile_sha256'] == inputs.sha(args.profile), 'Resume profile changed')
        selection, viewer = state['selection'], Path(state['viewer'])
        if (work / 'result.json').exists():
            result = inputs.load(work / 'result.json')
            print(json.dumps(result, indent=2))
            return result
    else:
        selection = selection_from_args(args)
        viewer = profile_path(args.output or profile['default_output']).resolve()
        if args.dry_run:
            print(json.dumps({'status': 'dry_run_no_SSH_or_submission', 'selection': selection, 'output': str(viewer)}, indent=2))
            return
        work = profile_path(profile['artifact_root']) / ('request-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
        work.mkdir(parents=True, exist_ok=False)
        state = {'status': 'prepared', 'selection': selection, 'viewer': str(viewer),
                 'profile_sha256': inputs.sha(args.profile), 'request_id': work.name}
        write(work / 'state.json', state)
        event(profile, work, 'prepared', runs=len(selection['runs']), output=str(viewer))
    viewer.mkdir(parents=True, exist_ok=True)
    print('Report request: ' + str(work), flush=True)
    lock = viewer / '.report-update.lock'
    if lock.exists():
        inputs.require(inputs.load(lock)['request_id'] == state['request_id'], 'Another/unknown report update owns this viewer; inspect its request')
    else:
        write(lock, {'request_id': state['request_id'], 'work': str(work)})
    pin, engine, archive = package_engine(profile)
    connection = Connection(profile, work)
    source = install(connection, profile, pin, archive)
    setup = connection.action(source, pin, 'setup')
    if setup['state'] == 'blocked_before_submission':
        write(work / ('blocked-' + uuid.uuid4().hex + '.json'), setup)
        event(profile, work, 'blocked_before_submission', engine_pin=pin, no_jobs_submitted=True, reason=setup['error'])
        inputs.require(inputs.load(lock)['request_id'] == state['request_id'], 'Viewer lock identity changed')
        lock.unlink()
        raise RuntimeError('Report setup was blocked before any job was submitted: ' + setup['error'] + '; evidence ' + setup['evidence'])
    validation_reused = setup['state'] == 'validated'
    if setup['state'] != 'validated':
        event(profile, work, 'engine_CPU_submitted', engine_pin=pin, job_id=setup['submission']['job_id'])
        wait(connection, source, pin, str(PurePosixPath(source).parent / 'validation'))
    cert = connection.action(source, pin, 'certify')
    record(work / 'engine-validation.json', cert)
    print('Validated report engine ' + pin[:12] + ' (' + str(cert['cases']) + ' CPU cases)', flush=True)
    event(profile, work, 'engine_validated', engine_pin=pin, cpu_job_id=cert['cpu_job_id'], cases=cert['cases'], reused=validation_reused)
    if args.setup_only:
        result = {'status': 'engine_ready', 'engine_pin': pin, 'cpu_job_id': cert['cpu_job_id'], 'work': str(work)}
    else:
        if state.get('submission'):
            sub = state['submission']
            edition = state['edition']
        else:
            if (work / 'submission-attempt.json').exists():
                old_request = inputs.load(work / 'submission-attempt.json')['request']
                selection = old_request['selection']
                edition = selection['edition']
                plan = inputs.load(work / 'captured-plan.json')
            else:
                edition = next_edition(viewer)
                selection = {**selection, 'edition': edition}
                plan = connection.action(source, pin, 'capture', selection)
                record(work / 'captured-plan.json', plan)
            prior = current_plan(viewer)
            same = prior is not None and inputs.selection_key(prior['arms'], profile['validated_renderer_sha256']) == inputs.selection_key(plan['arms'], profile['validated_renderer_sha256'])
            if same and not args.force:
                # Verify current assets before reporting an unchanged delivery.
                local_assets(viewer)
                http = None if args.no_serve else ensure_http(viewer, args.port or profile['port'], work)
                result = {'status': 'unchanged_report_reused_no_render_job', 'edition': prior['edition'], 'viewer': str(viewer), 'HTTP': http,
                          'engine_validation_reused': validation_reused, 'total_seconds': time.monotonic()-started, 'work': str(work)}
                write(work / 'result.json', result)
                event(profile, work, result['status'], edition=prior['edition'], elapsed_seconds=result['total_seconds'])
                inputs.require(inputs.load(lock)['request_id'] == state['request_id'], 'Viewer lock identity changed')
                lock.unlink()
                print(json.dumps(result, indent=2))
                return result
            inventory = local_assets(viewer)
            request = {'request_id': state['request_id'], 'selection': selection, 'known_asset_sha256': sorted(inventory),
                       'selection_key': inputs.selection_key(plan['arms'], profile['validated_renderer_sha256'])}
            # Persist before the non-idempotent action. Resume never calls submit
            # when this marker exists without an authenticated returned receipt.
            if (work / 'submission-attempt.json').exists():
                old = inputs.load(work / 'submission-attempt.json')
                inputs.require(old['engine_pin'] == pin, 'Resume engine changed')
                sub = connection.action(source, pin, 'recover_submission', old['request'])
            else:
                write(work / 'submission-attempt.json', {'engine_pin': pin, 'request': request})
                sub = connection.action(source, pin, 'submit', request)
            state.update(submission=sub, edition=edition, engine_pin=pin)
            (work / 'state.json').write_bytes(inputs.encoded(state))
            event(profile, work, 'render_submitted', engine_pin=pin, job_id=sub['job_id'], edition=edition)
        parent = str(PurePosixPath(profile['remote_root']) / 'requests' / state['request_id'])
        wait(connection, source, pin, parent)
        value = connection.action(source, pin, 'delivery', {'parent': parent})
        verify_delivery(value, source, pin)
        inputs.require(value['certificate']['modules'] == engine['test_modules'], 'Delivered CPU module attribution differs')
        record(work / 'delivery-evidence.json', value)
        delta = work / 'report-delta.tar.gz'
        download(connection, value['delivery']['archive'], delta)
        pointer = publish_delta(delta, value['delivery'], viewer, local_assets(viewer), pin, inputs.sha(work / 'engine-validation.json'))
        http = None if args.no_serve else ensure_http(viewer, args.port or profile['port'], work)
        result = {'status': 'published_verified_saved_report', 'publication': pointer, 'HTTP': http,
                  'runs': value['delivery']['arms'], 'delta_download_bytes': delta.stat().st_size,
                  'reused_local_assets': len(value['delivery']['omitted_verified_local_assets']), 'work': str(work),
                  'original_results_changed': False, 'scientific_success_claim': False,
                  'engine_validation_reused': validation_reused, 'total_seconds': time.monotonic()-started,
                  'render_seconds': value['delivery']['elapsed_seconds'], 'UI_qualification_preview': args.preview}
    write(work / 'result.json', result)
    event(profile, work, result['status'], engine_pin=pin, elapsed_seconds=time.monotonic()-started,
          edition=state.get('edition'), job_id=state.get('submission', {}).get('job_id'))
    inputs.require(inputs.load(lock)['request_id'] == state['request_id'], 'Viewer lock identity changed')
    lock.unlink()
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
