"""Independent stdlib-only verification of downloaded allocated evidence."""
import ast
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

repo = Path.cwd()
tag = '20261022-brd4-jq1-lrboost-analysis-01'
base = repo / 'results/hpc' / tag
bundle = base / 'cpu-evidence-01'
manifest_sha = '214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc'
science_sha = '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323'
remote = '/work3/s234818/mosaic/runs/' + tag
source = '/work3/s234818/mosaic/releases/' + tag + '/source'
science_tag = '20261021-brd4-jq1-25-fixed-lrboost-01'
science = '/work3/s234818/mosaic/releases/' + science_tag + '/source'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_members(path):
    result = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            assert member.isfile() and not member.issym() and not member.islnk()
            assert not name.is_absolute() and '..' not in name.parts and str(name) == member.name
            assert '\\' not in member.name and ':' not in member.name and member.name not in result
            raw = archive.extractfile(member).read()
            assert len(raw) == member.size
            result[member.name] = raw
    return result


archive = base / 'cpu-evidence-01.tar.gz'
assert archive.stat().st_size == 24524
assert digest(archive.read_bytes()) == 'b6d05c1eea559db93c650d356e5fffadcac7c68b49fe61dc0084a6fa7125f8f7'
contents = safe_members(archive)
listed = read(bundle / 'evidence_manifest.json')['files']
assert len(contents) == 57 and len(listed) == 56
assert set(contents) == set(listed) | {'evidence_manifest.json'}
for name, raw in contents.items():
    assert (bundle / name).read_bytes() == raw
for name, pin in listed.items():
    assert len(contents[name]) == pin['bytes'] and digest(contents[name]) == pin['sha256']
origins = read(bundle / 'origin-map.json')
assert len(set(origins.values())) == len(origins) and set(origins) <= set(listed)
reverse = {value: key for key, value in origins.items()}
assert digest(contents['source/source_manifest.json']) == manifest_sha
source_map = read(bundle / 'source/source_manifest.json')
assert len(source_map['files']) == 16 and source_map['external_scientific_source_manifest_sha256'] == science_sha
package = safe_members(base / 'source01/source.tar.gz')
assert package['source_manifest.json'] == contents['source/source_manifest.json']
for name, value in source_map['files'].items():
    assert digest(package[name]) == value
pins = next(ast.literal_eval(node.value) for node in ast.parse(package['scripts/jq1_lrboost_report_contract.py'].decode()).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'ARMS' for target in node.targets))

original = repo / 'results/hpc' / science_tag / 'delivery-metadata-02'
verification = read(original / 'verification.json')
assert digest((original / 'verification.json').read_bytes()) == 'd939121ba93e71fccc03413d04ce93e3496624b8e81a9ebb0e54015b211ec267'
for name, pin in verification['files'].items():
    raw = (original / name).read_bytes()
    assert len(raw) == pin['bytes'] and digest(raw) == pin['sha256']
original_source = repo / 'results/hpc' / science_tag / 'source01/source.tar.gz'
assert original_source.stat().st_size == 39582615
assert digest(original_source.read_bytes()) == 'e1a100aebf44698515ba161c1a001e338a902b250bf1a737f888df33f55098d1'
science_members = {str(PurePosixPath(name).relative_to('source')): raw for name, raw in safe_members(original_source).items()}
science_manifest = json.loads(science_members['source_manifest.json'])
assert digest(science_members['source_manifest.json']) == science_sha and len(science_manifest['files']) == 610
assert set(science_members) == set(science_manifest['files']) | {'source_manifest.json'}
for name, value in science_manifest['files'].items():
    assert digest(science_members[name]) == (value['sha256'] if isinstance(value, dict) else value)
for arm, pin in pins.items():
    binding = verification['actual_long_submissions'][arm]
    for left, right in [('job_id', 'job'), ('parent', 'output_parent'), ('rate', 'rate'), ('config_sha256', 'config'),
                        ('submission_sha256', 'submission'), ('driver_invocation_sha256', 'driver')]:
        assert binding[left] == pin[right]
    assert binding['source_manifest_sha256'] == pin['manifest'] == science_sha
    assert digest(science_members['examples/configs/brd4_jq1_25_fixed_lrboost_' + arm + '_10000.json']) == pin['config']
    assert verification['files']['campaign/validate-' + arm + '/payload/study/assessment.json']['sha256'] == pin['ready']


def compact(raw):
    return re.sub(r'\s+', '', raw)


def actual_status(raw, job, queue):
    value = compact(raw)
    assert re.findall(r'Job<([^>]+)>', value) == [job] and re.findall(r'Queue<([^>]+)>', value) == [queue]
    statuses = re.findall(r'Status<([^>]+)>', value)
    assert len(statuses) == 1
    return statuses[0]


def lsf_success(raw, job):
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', raw, re.M)
    assert len(subjects) == 1 and subjects[0][0] == job and subjects[0][1].rstrip().endswith(' Done')
    assert re.search(r'^Successfully completed\.\s*$', raw, re.M) and not re.search(r'Exited\s+(?:with exit|by signal)', raw, re.I)


for name, job in [('cpu', '29780814'), ('after-cpu', '29780815')]:
    assert actual_status((bundle / ('scheduler-' + name + '.txt')).read_text(), job, 'hpc') == 'DONE'
    raw = (bundle / ('analysis/' + name + '/stdout.' + job + '.log')).read_text()
    lsf_success(raw, job)
    assert 'Authenticated 16 immutable source files' in raw and 'Authenticated 610 immutable source files' in raw
cert = read(bundle / 'analysis/cpu-certificate.json')
assert cert['status'] == 'passed' and cert['job_id'] == cert['producer_job_id'] == '29780814'
assert cert['source_manifest_sha256'] == manifest_sha and cert['suite'] == 'jq1-lrboost-saved-report-contract-v1' and cert['tests'] == 278
assert len(cert['proofs']) == 5
for proof in cert['proofs']:
    assert proof['path'] in reverse and digest(contents[reverse[proof['path']]]) == proof['sha256']
xml = ET.fromstring(contents['analysis/cpu/payload/report-tests.xml'])
assert not any(list(xml.iter(name)) for name in ('failure', 'error', 'skipped'))
cases = list(xml.iter('testcase'))
assert len(cases) == 278
counts = Counter(next(part for part in case.attrib['classname'].split('.') if part.startswith('test_')) for case in cases)
expected_counts = {'test_jq1_lrboost_report_contract': 82, 'test_jq1_lrboost_render': 54, 'test_jq1_lrboost_publish': 59,
                   'test_jq1_lrboost_ops': 32, 'test_lrboost_endpoint_diagnostics': 16, 'test_boltz_affinity_report': 26,
                   'test_small_molecule_chemistry_report': 3, 'test_small_molecule_chemistry_ramp_report': 6}
assert counts == cert['tests_by_module'] == expected_counts
for suite in xml.iter('testsuite'):
    assert int(suite.get('tests')) == len(list(suite.iter('testcase')))
    assert all(int(suite.get(field, '0')) == 0 for field in ('errors', 'failures', 'skipped'))
log = contents['analysis/cpu/payload/report-tests.log'].decode()
assert re.search(r'collected 278 items\b', log) and re.search(r'\b278 passed in ', log)
invocation = read(bundle / 'analysis/cpu/payload/test-invocation.json')
argv = invocation['args']
assert invocation['cwd'] == source and argv[:5] == ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-m', 'pytest', '--rootdir', '/work3/s234818/mosaic/releases']
testpaths = [source + '/tests/' + name + '.py' for name in list(expected_counts)[:5]] + [science + '/tests/' + name + '.py' for name in list(expected_counts)[5:]]
assert argv[5:13] == testpaths and 'addopts=' in argv
assert argv[argv.index('--junitxml') + 1] == remote + '/cpu/payload/report-tests.xml'
assert argv[argv.index('--basetemp') + 1].startswith('/tmp/mosaic-boltz.29780814.')

expected = {
    'cpu': ('29780814', 'cpu', 'none', None, 'hpc', '2', '4', '00:45'),
    'after-cpu': ('29780815', 'certify-cpu', 'none', 'ended(29780814)', 'hpc', '1', '2', '00:30'),
    'diagnostic-protenix': ('29780821', 'diagnostic', 'protenix', None, 'gpul40s', '4', '8', '02:00'),
    'diagnostic-capture-protenix': ('29780822', 'diagnostic-capture', 'protenix', 'ended(29780821)', 'hpc', '2', '4', '00:45'),
    'diagnostic-boltz2': ('29780823', 'diagnostic', 'boltz2', None, 'gpul40s', '4', '8', '02:00'),
    'diagnostic-capture-boltz2': ('29780824', 'diagnostic-capture', 'boltz2', 'ended(29780823)', 'hpc', '2', '4', '00:45'),
    'complete-protenix': ('29780825', 'capture', 'protenix', None, 'hpc', '2', '4', '00:45'),
    'complete-boltz2': ('29780826', 'capture', 'boltz2', None, 'hpc', '2', '4', '00:45'),
    'complete-boltz2_chemistry': ('29780827', 'capture', 'boltz2_chemistry', 'ended(29779179)', 'hpc', '2', '4', '00:45'),
    'interim-0001': ('29780829', 'interim', 'none', None, 'hpc', '4', '8', '01:00'),
    'combined-report': ('29780830', 'final', 'none', 'done(29780825) && done(29780826) && done(29780827) && done(29780822) && done(29780824)', 'hpc', '4', '8', '01:00')}
assert {path.parent.name for path in (bundle / 'analysis').glob('*/submission.json')} == set(expected)
submissions = []
for name, (job, phase, arm, dependency, queue, slots, memory, limit) in expected.items():
    parent = remote + '/' + name
    submission = read(bundle / ('analysis/' + name + '/submission.json'))
    started = read(bundle / ('analysis/' + name + '/submission-started.json'))
    command = ['bsub', '-q', queue, '-n', slots, '-R', 'span[hosts=1]', '-R', 'rusage[mem=' + memory + 'GB]']
    if phase == 'diagnostic':
        command += ['-gpu', 'num=1:mode=exclusive_process']
    command += ['-W', limit, '-J', 'mosaic-lrboost-analysis-' + name, '-oo', parent + '/stdout.%J.log', '-eo', parent + '/stderr.%J.log']
    if dependency:
        command += ['-w', dependency]
    command += ['bash', '-l', source + '/scripts/hpc/jq1_lrboost_analysis_stage.lsf', source, remote, manifest_sha, phase, arm, parent]
    assert submission['args'] == started['args'] == command and submission['phase'] == started['phase'] == phase
    assert submission['arm'] == started['arm'] == arm and submission['dependency'] == started['dependency'] == dependency
    assert submission['outcome'] == 'submitted' and submission['returncode'] == 0 and submission['job_id'] == job
    assert submission['stdout'] == 'Job <' + job + '> is submitted to queue <' + queue + '>.\n' and submission['stderr'] == ''
    scheduler = (bundle / ('analysis/' + name + '/scheduler-at-submission.txt')).read_text()
    status = actual_status(scheduler, job, queue)
    assert status in ('PEND', 'RUN', 'DONE', 'EXIT')
    flat = compact(scheduler)
    assert ('Command<' + compact(' '.join(command[command.index('bash'):])) + '>') in flat
    assert 'rusage[mem=' + str(int(memory) * 1024) + '.00]' in flat and 'span[hosts=1]' in flat
    assert ('DependencyCondition<' + compact(dependency) + '>') in flat if dependency else 'DependencyCondition<' not in flat
    if phase == 'diagnostic':
        assert 'RequestedGPU<num=1:mode=exclusive_process>' in flat and '4Task(s)' in flat
    submissions.append({'parent': name, 'job_id': job, 'phase': phase, 'arm': arm, 'dependency': dependency, 'queue': queue,
                        'cpu_slots': int(slots), 'memory_gib_per_slot': int(memory), 'walltime': limit, 'status_at_submission': status})
jobs = read(bundle / 'analysis/jobs.json')
assert jobs['native_jobs'] == {arm: pin['job'] for arm, pin in pins.items()}
assert jobs['cpu'] == '29780814' and jobs['cpu_certificate_job'] == '29780815' and jobs['report_manifest'] == manifest_sha
assert jobs['diagnostics'] == {'protenix': '29780821', 'boltz2': '29780823'}
assert jobs['diagnostic_captures'] == {'protenix': '29780822', 'boltz2': '29780824'}
assert jobs['collectors'] == {'protenix': '29780825', 'boltz2': '29780826', 'boltz2_chemistry': '29780827'}
assert jobs['active_interim'] == '29780829' and jobs['final'] == '29780830'
initial = read(bundle / 'analysis/initial-submissions.json')
assert initial['cpu'] == read(bundle / 'analysis/cpu/submission.json') and initial['after_cpu'] == read(bundle / 'analysis/after-cpu/submission.json')
dispatch = read(bundle / 'analysis/analysis-dispatch-started.json')
assert dispatch['manifest'] == manifest_sha and dispatch['cpu_certificate_sha256'] == digest(contents['analysis/cpu-certificate.json'])
staging = read(bundle / 'analysis/source-staging.json')
assert staging['status'] == 'authenticated' and staging['source_files'] == 16 and staging['manifest_sha256'] == manifest_sha

gates = []
for path in sorted((bundle / 'analysis').glob('**/gate.json')):
    gate = read(path)
    assert gate['status'] == 'passed' and gate['reserve_bytes'] == 4294967296 and gate['file_reserve'] == 100000
    assert gate['full_native_projections'] == {'protenix': 1514341824, 'boltz2': 1820641824}
    outstanding = sum(max(0, gate['full_native_projections'][arm] - gate['current_native_bytes'][arm]) for arm in ('protenix', 'boltz2'))
    assert gate['outstanding_native_bytes'] == outstanding
    allowance = 1073741824 if gate['phase'] in ('cpu', 'preflight') else 2684354560
    assert gate['analysis_allowance_bytes'] == allowance and gate['diagnostic_per_arm_hard_output_bytes'] == 268435456
    assert gate['required_bytes'] == outstanding + allowance + 4294967296 and gate['available_bytes_conservative'] >= gate['required_bytes']
    quota = gate['quota']
    assert quota['available_bytes_lower'] == min(pool['remaining_bytes'] for pool in quota['finite_pools'])
    assert quota['available_files_lower'] == min(pool['remaining_files'] for pool in quota['finite_pools']) >= 100000
    for pool in quota['finite_pools']:
        assert pool['remaining_bytes'] == pool['hard_bytes_lower'] - pool['used_bytes_upper']
    rows = re.findall(r's234818\|306121\|\|\s*([\d.]+) GiB\|\s*([\d.]+) GiB\|\|\s*(\d+)\|\s*(\d+)', gate['raw_quota'])
    assert len(rows) == len(quota['finite_pools']) == 1
    used, hard, used_files, hard_files = rows[0]
    pool = quota['finite_pools'][0]
    assert pool['used_bytes_upper'] == int(Decimal(used).to_integral_value(rounding=ROUND_CEILING)) * 1073741824
    assert pool['hard_bytes_lower'] == int(Decimal(hard).to_integral_value(rounding=ROUND_FLOOR)) * 1073741824
    assert pool['remaining_files'] == int(hard_files) - int(used_files)
    assert gate['available_bytes_conservative'] == quota['available_bytes_lower'] - gate['stale_bytes_subtracted']
    assert gate['filesystem_inode_observation'] == {'f_files': 0, 'f_ffree': 0, 'f_favail': 0}
    assert gate['inode']['filesystem_inode_reporting'] == 'unreported_all_zero'
    assert gate['inode']['available_files_lower'] == quota['available_files_lower']
    assert 'no independent filesystem inode proof' in gate['inode']['inode_limitation']
    gates.append({'path': path.relative_to(bundle).as_posix(), 'sha256': digest(path.read_bytes()), 'phase': gate['phase'],
                  'required_bytes': gate['required_bytes'], 'available_bytes_conservative': gate['available_bytes_conservative'],
                  'file_headroom': quota['available_files_lower'], 'status': 'passed'})
assert len(gates) == 6

receipt = {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(), 'verifier': '/root/fixed_report_contract_impl',
    'scope': 'Independent local standard-library byte/hash/JSON/XML/schema verification of already allocated evidence only; no tests, scientific imports, numerical/model execution, SSH or submission',
    'archive': {'path': str(archive), 'bytes': 24524, 'sha256': digest(archive.read_bytes()), 'safe_regular_members': 57, 'manifest_listed_files': 56},
    'source_manifest_sha256': manifest_sha, 'external_scientific_source_manifest_sha256': science_sha,
    'cpu': {'job': '29780814', 'callback': '29780815', 'scheduler_status': 'DONE', 'queue': 'hpc', 'both_actual_lsf_success': True,
            'tests': 278, 'tests_by_module': dict(counts), 'failures': 0, 'errors': 0, 'skips': 0,
            'certificate_sha256': digest(contents['analysis/cpu-certificate.json']), 'all_five_cpu_proofs_rehashed': True},
    'original_reference_verification': {'verification_sha256': digest((original / 'verification.json').read_bytes()),
        'compact_files_rehashed': len(verification['files']), 'original_source_members_rehashed': 610, 'native_jobs': jobs['native_jobs'],
        'bindings': 'All original job/root/rate/config/submission/driver/readiness/source pins match sealed analysis contract and existing original compact byte proofs',
        'large_original_scope': 'Original large summaries/cases/JSONL/NPZ frames are remote references; this CPU/dispatch verification does not claim their numerical/frame verification or scientific completion'},
    'dispatch_storage_gates': gates, 'actual_submissions': submissions, 'no_native_job_restart_in_submission_set': True,
    'diagnostic_or_report_outcome_scope': 'Actual submissions only. This verification establishes no diagnostic scientific result or rendered/published report.',
    'files': listed, 'verification_script_sha256': digest(Path(__file__).read_bytes())}
output = Path(__file__).with_name('CPU-verification.json')
assert not output.exists()
output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'passed', 'receipt': str(output), 'sha256': digest(output.read_bytes()), 'cases': len(cases),
                  'submissions': len(submissions), 'gates': len(gates), 'original_compact_files': len(verification['files'])}))
