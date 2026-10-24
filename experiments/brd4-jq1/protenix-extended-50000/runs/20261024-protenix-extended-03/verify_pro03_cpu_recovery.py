"""Independent, unexecuted stdlib audit of a future compact Pro03 collection.

No SSH, model/array imports, numerical fixtures, submission or source changes.
Actual job IDs come only from the collected jobs/submission receipts. Original
large parent results and NPZs remain authenticated remote references; this
verifier never claims those bytes were downloaded or recalculates derivatives.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT = BASE + '/protenix-extended-03'
SOURCE = BASE + '/protenix-extended-release-03/source'
OLD = BASE + '/restart-lrreset-04'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
MANIFEST = 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562'
PARENT_MANIFEST = 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
ANCESTOR_MANIFEST = 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
REVISION = 'c74c659344123937608077e3e4078f75f09cd743'
MODEL = '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2'
CONTEXT = 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
REFERENCE = '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
SNAPSHOT = 'edf15f06ba1b6591d62cbc3e04e19ecd93b553a0627da38c370fdeed3c1b780f'
CONTEXT_READY = OLD + '/validate-protenix/payload/study/assessment.json'
CONTEXT_READY_SHA = '9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd'
STARTUP = 'protenix-actual-command-gpu-premodel-subprocess-v2'
FAILED_GATE_SHA = '6d1d0fae516de98615b83ebc005ab2330dd2146c08ef631f1bd5c3d28593a951'
GROUPS = {'hpc-contract': 500, 'legacy': 239, 'native-mapping': 34, 'chemistry': 59, 'startup': 34}
LEGACY_FULLY_DESELECTED_MODULES = {'test_small_molecule_integration', 'test_small_molecule_contact_native'}
MAX_BYTES = 128 * 1024**2
CONTRACT_MODULES = {
    'test_boltz_submission_controller', 'test_boltz_hpc_tmpdir', 'test_campaign25_ops',
    'test_campaign25_release', 'test_campaign25_pocket_ops', 'test_pocket_definition',
    'test_protenix_lr03_ops', 'test_protenix_lr03_storage', 'test_fixed25_ops', 'test_fixed25_storage',
    'test_fixed25_lrboost_contract', 'test_fixed25_lrboost_ops', 'test_fixed25_lrboost_storage',
    'test_fixed25_lrreset_contract', 'test_fixed25_lrreset_ops', 'test_fixed25_lrreset_storage',
    'test_protenix_initialization', 'test_random_chem', 'test_extended_protenix',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def local_bytes(path, maximum=MAX_BYTES):
    """Portable stable local read; retain each stat channel's exact seven fields."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked local evidence: ' + str(path))
    literal_before = path.stat()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(maximum + 1); after = os.fstat(stream.fileno())
    literal_after = path.stat()
    signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns, x.st_mode, x.st_nlink)
    shared = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_mode, x.st_nlink)
    require(not path.is_symlink() and len(raw) == before.st_size <= maximum
            and signature(before) == signature(after) and signature(literal_before) == signature(literal_after)
            and shared(before) == shared(literal_before), 'Local evidence changed or exceeded byte cap')
    return raw


def source_archive(relative, archive_sha, manifest_sha, count):
    path = REPO / relative; raw_archive = local_bytes(path)
    require(sha(raw_archive) == archive_sha, 'Original source archive SHA differs')
    members = {}
    with tarfile.open(fileobj=io.BytesIO(raw_archive)) as archive:
        for item in archive:
            require(item.isfile() and not item.linkname and item.name.startswith('source/'), 'Unsafe source archive type')
            name = item.name.removeprefix('source/'); rel = PurePosixPath(name)
            require(rel.as_posix() == name and not rel.is_absolute() and '..' not in rel.parts
                    and '\\' not in name and ':' not in name and name not in members, 'Unsafe/duplicate source member')
            members[name] = archive.extractfile(item).read()
    manifest_raw = members.pop('source_manifest.json'); manifest = strict_json(manifest_raw)
    require(sha(manifest_raw) == manifest_sha and len(members) == count and set(members) == set(manifest['files'])
            and all(sha(data) == manifest['files'][name] for name, data in members.items()), 'Full source seal/member scope differs')
    return members, manifest, manifest_raw, dict(path=str(path), bytes=len(raw_archive), sha256=sha(raw_archive), members=count)


def verify_sources(result):
    files, manifest, raw, archive = source_archive('results/hpc/20261024-protenix-extended-03/source01/source.tar.gz',
        'bde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef', MANIFEST, 643)
    parent, _, _, old_archive = source_archive('results/hpc/20261023-protenix-extended-01/source02/source.tar.gz',
        '8ba1c3370fee1f09b96af139dbf1403bb3c187c52d447ec4b0b23ebe4a65f63d', PARENT_MANIFEST, 641)
    _, ancestor, _, ancestor_archive = source_archive('results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/source02/source.tar.gz',
        '4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e', ANCESTOR_MANIFEST, 625)
    require(manifest['parent_manifest_sha256'] == PARENT_MANIFEST
            and manifest['git_revision_reference'] == ancestor['git_revision_reference'] == REVISION
            and manifest['git_revision_inheritance'] == dict(version='authenticated-ancestor-git-revision-v1',
                ancestor_manifest_sha256=ANCESTOR_MANIFEST,
                ancestor_archive_sha256=ancestor_archive['sha256'], field='git_revision_reference', value=REVISION),
            'Provenance recovery does not bind the authenticated original625 revision')
    mutable = {'scripts/extended_ops.py', 'scripts/hpc/extended_cpu.lsf', 'docs/protenix_extended.md',
               'scripts/extended_startup.py', 'tests/test_extended_startup.py'}
    changed = {name for name in set(files) | set(parent) if files.get(name) != parent.get(name)}
    require(changed == mutable and all(files[name] == data for name, data in parent.items() if name not in mutable),
            'Undeclared original641 scientific/config/driver/protocol/fixture changes')
    scientific = [name for name in parent if name.startswith('src/')]
    require(len(scientific) == 145 and all(files[name] == parent[name] for name in scientific), 'Scientific library bytes changed')
    for arm in ('random', 'continuation'):
        require(files['examples/configs/extended_' + arm + '.json'] == local_bytes(HERE / (arm + '.json')),
                'Current config bytes differ from immutable scientific source')
    result['sources'] = [archive, old_archive, ancestor_archive]
    result['scientific_src_members_unchanged'] = 145
    result['protected_original641_members_unchanged'] = 638
    result['git_revision_reference'] = REVISION
    return files, manifest, raw


class Evidence:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        require(self.directory.is_relative_to((REPO / 'results/hpc').resolve()), 'Evidence must remain under dedicated results')
        self.stdout = local_bytes(self.directory / 'stdout.bin')
        self.stderr = local_bytes(self.directory / 'stderr.bin')
        outcome = strict_json(local_bytes(self.directory / 'result.json'))
        require(outcome['returncode'] == 0 and outcome['remote_mutations'] is False
                and outcome['stdout_sha256'] == sha(self.stdout) and outcome['stderr_sha256'] == sha(self.stderr),
                'Actual read-only metadata collection did not complete')
        self.value = strict_json(self.stdout)
        self.metadata = strict_json(local_bytes(self.directory / 'metadata.json'))
        self.files = {}; self.pins = {}
        require(set(self.value['files']) == set(self.metadata['files']), 'Extracted/raw compact file scopes differ')
        for remote, row in self.value['files'].items():
            data = base64.b64decode(row['base64'], validate=True)
            require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'Compact raw file hash differs')
            index = self.metadata['files'][remote]
            require({k: index[k] for k in ('bytes', 'sha256')} == {k: row[k] for k in ('bytes', 'sha256')},
                    'Local extraction index differs')
            local = self.directory / index['local_path']
            require(local.resolve().is_relative_to(self.directory) and local_bytes(local) == data, 'Extracted original metadata bytes differ')
            self.files[remote] = data; self.pins[remote] = dict(bytes=len(data), sha256=sha(data), local_bytes_downloaded=True)
        require(self.value['external_files'] == self.metadata['external_files'], 'External reference scope differs')
        for path, row in self.value['external_files'].items():
            require(path not in self.files and row['local_bytes_downloaded'] is False
                    and type(row['bytes']) is int and row['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}', row['sha256']),
                    'External original byte reference is malformed or mislabeled downloaded')
            self.pins[path] = row
        for key in ('captured_utc', 'roots', 'scheduler', 'releases'):
            require(self.value[key] == self.metadata[key], 'Raw/extracted metadata observation differs: ' + key)

    def read(self, path):
        require(str(path) in self.files, 'Required compact original bytes missing: ' + str(path))
        return self.files[str(path)]

    def json(self, path):
        return strict_json(self.read(path))

    def pin(self, path, digest=None, size=None):
        require(str(path) in self.pins, 'Missing actual collected input/proof hash: ' + str(path))
        item = self.pins[str(path)]
        require((digest is None or item['sha256'] == digest) and (size is None or item['bytes'] == size),
                'Actual input/proof identity differs: ' + str(path))
        return item

    def proofs(self, record):
        require(record['status'] == 'passed' and isinstance(record['proofs'], list) and record['proofs'], 'Failed/empty certificate')
        require(len({p['path'] for p in record['proofs']}) == len(record['proofs']), 'Duplicate certificate proof')
        for pin in record['proofs']:
            self.pin(pin['path'], pin['sha256'], pin.get('bytes'))


def command(label, phase, dependency=None):
    if phase == 'cpu':
        resources = ['-q', 'hpc', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '01:00']
    elif phase in ('native', 'segment'):
        resources = ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]',
                     '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00' if phase == 'native' else '23:00']
    else:
        require(phase in ('after-cpu', 'after-native', 'after-segment', 'dispatch'), 'Unknown actual phase')
        resources = ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']
    parent = ROOT + '/' + label
    args = ['bsub', *resources, '-J', 'mosaic-pro50k-' + label, '-oo', parent + '/stdout.%J.log', '-eo', parent + '/stderr.%J.log']
    if dependency:
        args += ['-w', dependency]
    args += ['bash', '-l', SOURCE + '/scripts/hpc/' + ('extended_cpu.lsf' if phase == 'cpu' else 'extended_stage.lsf'), SOURCE, ROOT, MANIFEST]
    if phase != 'cpu':
        args += [phase, parent]
    return args


def own_submission(e, jobs, label, phase, dependency=None):
    value = e.json(ROOT + '/' + label + '/submission.json')
    require(label in jobs and value['job_id'] == jobs[label] and re.fullmatch(r'\d+', value['job_id'])
            and value['outcome'] == 'submitted' and value['returncode'] == 0 and value['phase'] == phase
            and value['dependency'] == dependency and value['args'] == command(label, phase, dependency),
            'Actual own job/source/resource/dependency receipt differs: ' + label)
    started = e.json(ROOT + '/' + label + '/submission-started.json')
    require(started['outcome'] == 'unknown' and started['phase'] == phase and started['args'] == value['args']
            and started['dependency'] == dependency, 'Own exclusive submission-started differs')
    return value


def lsf(e, parent, job, success=True):
    text = e.read(parent + '/stdout.' + job + '.log').decode()
    matches = re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', text, re.M)
    if success:
        require(matches == [(job, 'Done')] and 'Successfully completed.' in text and 'Exited with exit' not in text,
                'Actual same-job LSF success missing: ' + parent)
    else:
        require(matches == [(job, 'Exited')] and 'Exited with exit code 1.' in text, 'Expected preserved historical EXIT1 missing')
    return text


def final_scheduler(e, parent, job, queue, certificate):
    paths = [p['path'] for p in certificate['proofs'] if str(PurePosixPath(p['path']).parent) == parent
             and re.fullmatch(r'scheduler-final-\d+\.txt', PurePosixPath(p['path']).name)]
    require(len(paths) == 1, 'Unique original final scheduler proof missing')
    raw = re.sub(r'\s+', '', e.read(paths[0]).decode())
    require(re.findall(r'Job<([^>]+)>', raw) == [job] and re.findall(r'Status<([^>]+)>', raw) == ['DONE']
            and re.findall(r'Queue<([^>]+)>', raw) == [queue], 'Original final scheduler identity differs')


def verify_gate(value):
    work = value['work3']
    require(value['status'] == work['status'] == 'passed' and value['reserve_bytes'] == work['reserve_bytes'] == 4 * 1024**3
            and value['file_reserve'] == work['file_reserve'] == 100000
            and work['required_bytes'] == value['projected_bytes'] + value['reserve_bytes']
            and work['available_bytes_conservative'] >= work['required_bytes']
            and work['available_files_lower'] >= 100000 and value['noisy_capsule_inventory']['unknown_scopes'] == [],
            'Actual whole validation storage gate failed or reserve/headroom changed')
    require(value['prospective_adoption_metadata_controls']['per_control_bytes'] == 1024**3
            and set(value['full_extended']) == {'random', 'continuation'}
            and all(type(x) is int and x > 0 for x in value['full_extended'].values()), 'Declared full validation scopes changed')
    inode = work['filesystem_inode_observation']
    if tuple(inode[k] for k in ('f_files', 'f_ffree', 'f_favail')) == (0, 0, 0):
        require(work['filesystem_inode_reporting'] == 'unreported_all_zero', 'All-zero supplemental inode counters mislabeled independent proof')


def module_names(cases):
    return {part for row in cases for part in row.get('classname', '').split('.') if part.startswith('test_')}


def verify_raw_cpu(e, files, certificate):
    modules = {
        'hpc-contract': CONTRACT_MODULES,
        'legacy': {PurePosixPath(n).stem for n in files if n.startswith('tests/test_small_molecule_')
                   and n.endswith('.py') and not PurePosixPath(n).name.startswith('test_small_molecule_chemistry')
                   and PurePosixPath(n).stem not in LEGACY_FULLY_DESELECTED_MODULES},
        'chemistry': {PurePosixPath(n).stem for n in files if n.startswith('tests/test_small_molecule_chemistry') and n.endswith('.py')},
        'native-mapping': {'test_boltz_native_mapping', 'test_campaign25_native', 'test_campaign25_contract', 'test_pocket_boltz'},
        'startup': {'test_extended_startup'},
    }
    result = {}; total = 0
    for name, expected in GROUPS.items():
        path = ROOT + '/cpu/payload/' + name
        parsed = ET.fromstring(e.read(path + '.xml')); cases = list(parsed.iter('testcase'))
        require(len(cases) == expected and not any(list(parsed.iter(k)) for k in ('failure', 'error', 'skipped'))
                and len({(c.get('classname'), c.get('name')) for c in cases}) == expected,
                'Raw allocated case count/failure/uniqueness differs: ' + name)
        require(all(int(suite.get(key, '0')) == 0 for suite in parsed.iter('testsuite')
                    for key in ('failures', 'errors', 'skipped')), 'Raw suite failure/error/skip counters differ')
        require(module_names(cases) == modules[name]
                and all(len(set(c.get('classname', '').split('.')) & modules[name]) == 1 for c in cases),
                'Full exact scientific/metadata test-module attribution differs: ' + name)
        log = e.read(path + '.log').decode()
        require(re.search(r'\b' + str(expected) + r'\s+passed\b', log), 'Raw XML/log passing count differs')
        if name == 'legacy':
            require(re.search(r'\b3\s+deselected\b', log), 'Original three deliberate legacy deselections changed')
        if name == 'hpc-contract':
            require(sum('test_extended_protenix' in c.get('classname', '').split('.') for c in cases) == 56,
                    'Original56 exact Pro50k cases missing')
        result[name] = dict(actual_cases=len(cases), modules=sorted(modules[name]),
            raw_suite_headers=[dict(s.attrib) for s in parsed.iter('testsuite')],
            original_fully_deselected_modules=sorted(LEGACY_FULLY_DESELECTED_MODULES) if name == 'legacy' else [])
        total += len(cases)
    require(total == certificate['cases'] == 866 and certificate['groups'] == GROUPS, 'Full original832+startup34 certificate scope differs')
    return result


def startup_inputs(e, arm, parent, config_bytes):
    record = e.json(ROOT + '/startup-command-' + arm + '-native-0000.json'); e.proofs(record)
    result = record['result']; values = result['inputs']
    expected_keys = {'config', 'cpu-certificate', 'reference', 'pocket', 'context-assessment',
                     'context-prepared_context.json', 'context-prepared_context.npz'}
    if arm == 'continuation':
        expected_keys.add('resume-snapshot')
    require(set(values) == expected_keys, 'Exact CPU startup input coverage differs')
    expected_paths = dict(config=SOURCE + '/examples/configs/extended_' + arm + '.json',
        **{'cpu-certificate': ROOT + '/cpu-certificate.json', 'reference': ROOT + '/cpu/payload/composition_reference.json',
           'pocket': ROOT + '/cpu/payload/pocket_definition.json', 'context-assessment': CONTEXT_READY})
    context = e.json(CONTEXT_READY)
    e.pin(CONTEXT_READY, CONTEXT_READY_SHA)
    require(context['model_digest'] == MODEL and context['context_digest'] == CONTEXT
            and set(context['context_files']) == {'prepared_context.json', 'prepared_context.npz'}, 'Original frozen prepared context differs')
    for name, digest in context['context_files'].items():
        key = 'context-' + name; expected_paths[key] = str(PurePosixPath(CONTEXT_READY).parent / 'context' / name)
        require(values[key]['sha256'] == digest, 'Original context member pin differs')
    if arm == 'continuation':
        expected_paths['resume-snapshot'] = parent['snapshot']['path']
        require(values['resume-snapshot']['sha256'] == SNAPSHOT and values['resume-snapshot']['bytes'] == 14226,
                'Original exact completed step10000 snapshot differs')
    require({k: v['path'] for k, v in values.items()} == expected_paths, 'Actual startup input paths changed')
    for row in values.values():
        e.pin(row['path'], row['sha256'], row['bytes'])
    require(values['config']['sha256'] == sha(config_bytes)
            and values['reference']['sha256'] == REFERENCE and values['pocket']['sha256'] == POCKET,
            'Config/reference/pocket startup hashes differ')
    return record, values


def startup_args(arm, values, output):
    args = [PYTHON, SOURCE + '/examples/brd4_25_atoms.py', '--mode', 'validate',
        '--config', values['config']['path'], '--output', output, '--manifest', MANIFEST,
        '--cpu-certificate', values['cpu-certificate']['path'], '--cpu-certificate-sha256', values['cpu-certificate']['sha256'],
        '--reference', values['reference']['path'], '--pocket', values['pocket']['path'], '--pocket-sha256', POCKET,
        '--offset', '0' if arm == 'random' else '10000', '--segment-steps', '10000']
    if arm == 'continuation':
        args += ['--resume-snapshot', values['resume-snapshot']['path'], '--resume-sha256', SNAPSHOT]
    return args


def verify_command_capture(e, jobs, arm, parent, config_bytes):
    record, values = startup_inputs(e, arm, parent, config_bytes); result = record['result']
    require(record['version'] == STARTUP and record['source_manifest_sha256'] == MANIFEST
            and record['arm'] == arm and record['phase'] == 'native' and record['index'] == 0
            and record['owner_parent'] == ROOT + '/after-cpu' and record['owner_job_id'] == jobs['after-cpu']
            and result['version'] == STARTUP and result['kind'] == 'cpu_generated_command_only'
            and result['status'] == 'command_only_authenticated' and result['job_id'] == jobs['after-cpu']
            and result['queue'] == 'hpc' and result['source'] == SOURCE and result['source_manifest_sha256'] == MANIFEST
            and result['git_revision_reference'] == REVISION and result['arm'] == arm and result['phase'] == 'native'
            and result['index'] == 0 and result['optimizer_updates'] == 0 and result['saved_results_changed'] is False,
            'Actual CPU-generated command record/owner scope differs')
    require(all(result[k] is False for k in ('actual_driver_executed', 'gpu_startup_passed', 'model_loaded',
            'gradient_passed', 'scientific_readiness_passed')), 'CPU metadata mislabeled GPU/native execution')
    args = result['args']; require(args.count('--output') == 1, 'Disposable CPU command lacks exact output')
    output = args[args.index('--output') + 1]
    require(PurePosixPath(output).is_absolute() and '/pro-command-' + arm + '-native-0000/' in output
            and args == startup_args(arm, values, output), 'Actual generated scientific command differs')
    expected_proofs = {ROOT + '/after-cpu/submission.json', SOURCE + '/source_manifest.json',
                       SOURCE + '/scripts/extended_startup.py', *[row['path'] for row in values.values()]}
    require({p['path'] for p in record['proofs']} == expected_proofs, 'CPU command proof coverage differs')
    return dict(status='authenticated_command_metadata_only', job_id=jobs['after-cpu'],
                command_sha256=sha(json.dumps(args, separators=(',', ':')).encode()), gpu_startup_passed=False,
                native_gradient_passed=False, optimizer_updates=0)


def gpu_startup_observation(e, jobs, arm, parent, config_bytes, manifest):
    """Authenticate an optional persisted startup receipt, never a native pass."""
    name = arm + '-native-0000'; directory = ROOT + '/' + name
    path = directory + '/startup-gpu.json'; failed_path = directory + '/startup-gpu-failure.json'
    if path not in e.files:
        if failed_path in e.files:
            failure = e.json(failed_path)
            require(failure['status'] == 'failed' and failure['version'] == STARTUP
                    and failure['source_manifest_sha256'] == MANIFEST and failure['owner_job_id'] == jobs[name]
                    and failure['arm'] == arm and failure['phase'] == 'native' and failure['index'] == 0
                    and failure['scientific_readiness_passed'] is False, 'Malformed actual GPU startup failure record')
            return dict(status='original_failed_startup_preserved', proof_sha256=e.pin(failed_path)['sha256'],
                        error=failure['error'], native_gradient_pass_claimed=False)
        return dict(status='no_completed_gpu_startup_receipt', native_gradient_pass_claimed=False)
    record = e.json(path); e.proofs(record); value = record['result']
    _, inputs = startup_inputs(e, arm, parent, config_bytes)
    require(record['version'] == STARTUP and record['source_manifest_sha256'] == MANIFEST
            and record['arm'] == arm and record['phase'] == 'native' and record['index'] == 0
            and value['version'] == STARTUP and value['kind'] == 'gpu_startup_only'
            and value['status'] == 'startup_only_passed' and value['job_id'] == jobs[name] and value['queue'] == 'gpul40s'
            and value['source'] == SOURCE and value['source_manifest_sha256'] == MANIFEST
            and value['git_revision_reference'] == REVISION and value['arm'] == arm and value['phase'] == 'native'
            and value['index'] == 0 and value['inputs'] == inputs and value['actual_driver_executed'] is True
            and value['gpu_backend_guard_executed_unchanged'] is True and value['optimizer_updates'] == 0
            and value['saved_results_changed'] is False
            and all(value[k] is False for k in ('model_loaded', 'gradient_passed', 'scientific_readiness_passed')),
            'Actual GPU startup/source/input/no-model scope differs')
    args = value['args']; require(args.count('--output') == 1, 'GPU startup missing exact disposable output')
    output = args[args.index('--output') + 1]
    require('/pro-gpu-startup-' + arm + '-native-0000/' in output
            and args == startup_args(arm, inputs, output), 'Actual GPU startup command differs')
    expected_proofs = {directory + '/submission.json', ROOT + '/startup-command-' + arm + '-native-0000.json',
        SOURCE + '/source_manifest.json', SOURCE + '/scripts/extended_startup.py', *[p['path'] for p in inputs.values()]}
    require({p['path'] for p in record['proofs']} == expected_proofs, 'GPU startup proof coverage differs')
    summary = value['summary']
    require(summary['status'] == 'failed' and summary['error'] ==
            'ProtenixPremodelBoundaryReached: Intentional Protenix GPU startup-only boundary; no native model loaded'
            and summary['source_manifest_sha256'] == MANIFEST and summary['source']['git_revision_reference'] == REVISION
            and not summary.get('model_digest') and not summary.get('native_gradient_checks') and not summary.get('cases'),
            'Startup sentinel must remain a failed disposable pre-model summary')
    execution_files = {name: digest for name, digest in manifest['files'].items()
        if name == 'pyproject.toml' or (PurePosixPath(name).parts[0] in ('src', 'tests', 'examples', 'scripts', 'experiments')
            and PurePosixPath(name).suffix in ('.py', '.json', '.lsf', '.sh', '.toml', '.txt')
            and '__pycache__' not in PurePosixPath(name).parts)}
    require(summary['source']['files'] == execution_files and summary['source']['execution_content_sha256'] ==
            sha(json.dumps(execution_files, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()),
            'Actual driver startup execution-member identity differs')
    isolation = value['execution_isolation']; child = isolation['child_receipt_value']
    child_raw = (json.dumps(child, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    require(isolation['mode'] == 'short_lived_same_allocation_subprocess' and isolation['returncode'] == 0
            and isolation['timed_out'] is False and isolation['child_fully_reaped'] is True
            and isolation['parent_native_backend_imported'] is False
            and type(isolation['pid']) is int and isolation['pid'] > 0
            and type(isolation['parent_pid']) is int and isolation['parent_pid'] > 0 and isolation['pid'] != isolation['parent_pid']
            and child['pid'] == isolation['pid'] and child['parent_pid'] == isolation['parent_pid']
            and child['version'] == STARTUP and child['status'] == 'startup_only_passed'
            and child['job_id'] == jobs[name] and child['queue'] == 'gpul40s'
            and child['source'] == SOURCE and child['source_manifest_sha256'] == MANIFEST
            and child['summary'] == summary and child['summary_sha256'] == value['summary_sha256']
            and child['boundary'] == value['boundary'] and isolation['child_receipt']['bytes'] == len(child_raw)
            and isolation['child_receipt']['sha256'] == sha(child_raw), 'GPU startup child was not authenticated/exited before production')
    command_path = isolation['args'][8]
    child_command = dict(version=STARTUP, source=SOURCE, source_manifest_sha256=MANIFEST, args=args)
    command_raw = (json.dumps(child_command, sort_keys=True, allow_nan=False) + '\n').encode()
    require(child['command_sha256'] == sha(command_raw) and isolation['args'] == [PYTHON,
        SOURCE + '/scripts/extended_startup.py', '--execute-child', '--source', SOURCE, '--manifest', MANIFEST,
        '--argv-json', command_path, '--receipt', isolation['child_receipt']['path']]
        and PurePosixPath(command_path).parent == PurePosixPath(isolation['child_receipt']['path']).parent,
        'Actual isolated GPU command/source/hash differs')
    return dict(status='authenticated_gpu_startup_only_no_native_pass', job_id=jobs[name],
        proof_sha256=e.pin(path)['sha256'], child_fully_reaped=True, controller_backend_imported=False,
        original_child_value_and_hash_authenticated=True, child_tmp_original_bytes_newly_downloaded=False,
        native_gradient_pass_claimed=False, optimizer_updates=0)


def verify(e, result):
    source_files, manifest, manifest_raw = verify_sources(result)
    require(e.read(SOURCE + '/source_manifest.json') == manifest_raw
            and e.value['releases'][SOURCE] == dict(manifest_sha256=MANIFEST, source_files=643,
                all_members_rehashed=True, readonly=True), 'Actual remote643 source seal observation differs')
    for path, raw in e.files.items():
        if path.startswith(SOURCE + '/') and path != SOURCE + '/source_manifest.json':
            member = path.removeprefix(SOURCE + '/')
            require(member in source_files and raw == source_files[member], 'Collected source member differs from exact archive')
    root = e.value['roots'][ROOT]; jobs = e.json(ROOT + '/jobs.json')
    require(root['jobs_file_present'] is True and jobs == root['jobs'] and {'cpu', 'after-cpu'} <= set(jobs)
            and len(set(jobs.values())) == len(jobs) and all(re.fullmatch(r'\d+', job) for job in jobs.values()),
            'No actual complete unique own job registry')
    result['jobs'] = jobs
    own_submission(e, jobs, 'cpu', 'cpu')
    own_submission(e, jobs, 'after-cpu', 'after-cpu', 'ended(' + jobs['cpu'] + ')')
    certificate = e.json(ROOT + '/cpu-certificate.json'); e.proofs(certificate)
    require(certificate['suite'] == 'protenix-serial50k-v1' and certificate['source_manifest_sha256'] == MANIFEST,
            'CPU866 certificate source/suite differs')
    result['raw_groups'] = verify_raw_cpu(e, source_files, certificate)
    assessment = e.json(ROOT + '/cpu/payload/assessment.json'); e.proofs(assessment)
    require(assessment['status'] == 'passed' and assessment['passed'] == 832
            and assessment['source_manifest_sha256'] == MANIFEST and assessment['reference_sha256'] == REFERENCE
            and {p['path']: p['passed'] for p in assessment['proofs']} == {
                ROOT + '/cpu/payload/' + name + '.xml': count for name, count in GROUPS.items() if name != 'startup'},
            'Original832 CPU assessment must remain separate from additive34 startup rows')
    final_scheduler(e, ROOT + '/cpu', jobs['cpu'], 'hpc', certificate)
    lsf(e, ROOT + '/cpu', jobs['cpu'])
    lsf(e, ROOT + '/after-cpu', jobs['after-cpu'])
    result['actual_cpu_cases'] = 866
    result['cpu_certificate_sha256'] = e.pin(ROOT + '/cpu-certificate.json')['sha256']
    result['cpu_pass_authenticated'] = True
    failed_path = ROOT + '/staging-01/storage-preflight/gate.json'
    e.pin(failed_path, FAILED_GATE_SHA, 6133); failed = e.json(failed_path)
    require(failed['status'] == 'failed' and failed['work3']['required_bytes'] == 41643669252
            and failed['work3']['available_bytes_conservative'] == 27472908456, 'Historical failed preflight relabeled/changed')
    current_gates = []
    for path in (ROOT + '/cpu-recovery-03/storage-preflight/gate.json', ROOT + '/cpu/storage-cpu-entry/gate.json',
                 ROOT + '/after-cpu/storage-native-dispatch/gate.json'):
        value = e.json(path); verify_gate(value)
        require(value['full_long_admission'] is False, 'Validation admission mislabeled full-production pass')
        current_gates.append(dict(path=path, sha256=e.pin(path)['sha256'], phase=value['phase'],
            required_bytes=value['work3']['required_bytes'], available_bytes=value['work3']['available_bytes_conservative'],
            full_long_admission=False))
    result['current_validation_gates'] = current_gates
    result['historical_failed_gate_preserved'] = dict(path=failed_path, sha256=FAILED_GATE_SHA, status='failed')
    recovery = e.json(ROOT + '/cpu-recovery-03/submission-complete.json')
    require(recovery['status'] == 'submitted_not_passed' and recovery['source_manifest_sha256'] == MANIFEST
            and recovery['source_files'] == 643 and recovery['cpu_cases_required'] == 866
            and recovery['cpu']['job_id'] == jobs['cpu'] and recovery['callback']['job_id'] == jobs['after-cpu']
            and recovery['previous_failed_gate']['sha256'] == FAILED_GATE_SHA
            and recovery['original_saved_results_unchanged'] is True, 'Actual unique recovery submission identity differs')
    # The original12 failure bytes were already independently downloaded. The
    # new recovery receipt pins their exact map; do not call that a new download.
    inspection = REPO / 'results/hpc/20261024-boltz-startup-recovery-02/peer-projection-inspection/stdout.bin'
    raw_history = local_bytes(inspection)
    require(sha(raw_history) == '7b28b33288172998d1dd77c4e4f6a986e7eefafb8a3acf8ec0cc5e43a0bac5cc',
            'Prior independently captured original641 failures changed')
    history = strict_json(raw_history); prefix = BASE + '/protenix-extended-01/'
    require(history['source_manifest_sha256'] == PARENT_MANIFEST and history['source_files_rehashed'] == 641
            and history['projection_exists'] is False and history['segment_directories'] == [],
            'Original failed641 history must remain scientifically unstarted')
    old_pins = {p.removeprefix(prefix): dict(bytes=row['bytes'], sha256=row['sha256']) for p, row in history['files'].items()}
    started = e.json(ROOT + '/cpu-recovery-03/started.json')
    require(len(old_pins) == 12 and started['old_failure_proofs'] == old_pins
            and started['source_manifest_sha256'] == MANIFEST and started['source_files'] == 643
            and started['continuation_snapshot_sha256'] == SNAPSHOT, 'Fresh recovery did not bind original641 failure map')
    for path, row in history['files'].items():
        data = row['text'].encode(); require(len(data) == row['bytes'] and sha(data) == row['sha256'], 'Original captured failure bytes differ')
    result['preserved_failure_evidence_scope'] = 'Prior downloaded12 raw original641 failures plus fresh recovery-operator pin map; not a claim of fresh original641/625 byte download.'
    parent_raw = source_files['docs/extended_parent.json']; parent = strict_json(parent_raw)
    require(e.read(SOURCE + '/docs/extended_parent.json') == parent_raw and parent['job'] == '29792670'
            and parent['global_state'] == 10000 and parent['snapshot']['sha256'] == SNAPSHOT
            and parent['snapshot']['bytes'] == 14226, 'Original successful continuation parent identity differs')
    for pin in [parent['snapshot'], *parent['proofs'].values()]:
        e.pin(pin['path'], pin['sha256'], pin['bytes'])
    adoption = e.json(ROOT + '/parent-adoption.json')
    require(adoption['status'] == 'passed' and adoption['job'] == '29792670' and adoption['global_state'] == 10000
            and adoption['snapshot'] == parent['snapshot'], 'Allocated original completed-parent adoption failed')
    retirement_raw = source_files['docs/extended_retirement.json']
    require(e.read(SOURCE + '/docs/extended_retirement.json') == retirement_raw
            and strict_json(retirement_raw)['status'] == 'intentionally_retired_unstarted',
            'Original superseded queued-random retirement was changed')
    require(e.pin(ROOT + '/cpu/payload/composition_reference.json')['sha256'] == REFERENCE
            and e.pin(ROOT + '/cpu/payload/pocket_definition.json')['sha256'] == POCKET, 'Fixed reference or JQ1 pocket changed')
    result['startup_command_metadata'] = {}
    result['initial_native_submissions'] = {}
    for arm in ('random', 'continuation'):
        config_bytes = source_files['examples/configs/extended_' + arm + '.json']
        require(e.read(SOURCE + '/examples/configs/extended_' + arm + '.json') == config_bytes, 'Collected exact config differs')
        result['startup_command_metadata'][arm] = verify_command_capture(e, jobs, arm, parent, config_bytes)
        native = arm + '-native-0000'; callback = arm + '-after-native-0000'
        own_submission(e, jobs, native, 'native')
        own_submission(e, jobs, callback, 'after-native', 'ended(' + jobs[native] + ')')
        observed = e.value['scheduler'].get(jobs[native], {})
        result['initial_native_submissions'][arm] = dict(job_id=jobs[native], callback_job_id=jobs[callback],
            live_scheduler_query=observed, native_readiness_pass_claimed=False,
            gpu_startup_receipt_present=ROOT + '/' + native + '/startup-gpu.json' in e.files,
            native_certificate_present=ROOT + '/' + native + '-certificate.json' in e.files,
            gpu_startup_metadata=gpu_startup_observation(e, jobs, arm, parent, config_bytes, manifest))
    result.update(status='passed_independent_pro03_cpu866_and_command_metadata_verification',
        callback_lsf_success_authenticated=True, cpu_cases=866, source_manifest_sha256=MANIFEST,
        source_and_config_recovery_changed_science=False, compact_original_files_verified=len(e.files),
        authenticated_external_references=len(e.value['external_files']),
        large_original_parent_or_snapshot_bytes_downloaded=False, new_native_gradient_or_production_pass_claimed=False,
        full_extended_production_admission_pass_claimed=False, new_optimization_pass_claimed=False,
        scientific_settings_or_saved_results_changed=False,
        scope='Opaque bytes, XML rows/modules, exact own resources/LSF/source/config/input/gate/CPU-command metadata only. Allocated native and production proofs are separate; no local numerical work.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args(); directory = args.evidence.resolve()
    require(directory.is_relative_to((REPO / 'results/hpc').resolve()), 'Output must remain inside ignored evidence')
    output = directory / 'pro03-cpu-recovery-verification.json'
    require(not output.exists(), 'Existing verifier outcome must be preserved; use a fresh actual collection')
    result = dict(version='mosaic-pro03-independent-cpu-recovery-verification-v1', status='failed_or_not_ready',
        prepared_verifier_sha256=sha(local_bytes(__file__)), verified_utc=datetime.now(timezone.utc).isoformat(),
        cpu_pass_authenticated=False, new_native_gradient_or_production_pass_claimed=False,
        full_extended_production_admission_pass_claimed=False, new_optimization_pass_claimed=False)
    code = 1
    try:
        evidence = Evidence(directory); result['collected_stdout_sha256'] = sha(evidence.stdout)
        verify(evidence, result); code = 0
    except Exception as error:
        result.update(status='failed_or_not_ready', error=type(error).__name__ + ': ' + str(error))
    with output.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'cpu_pass_authenticated', 'new_native_gradient_or_production_pass_claimed',
                     'full_extended_production_admission_pass_claimed', 'new_optimization_pass_claimed')}, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
