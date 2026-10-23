"""Independent post-seal stdlib evidence audit; never imports numerical code.

Explicit --expect-failure accepts a proven preflight failure with no submitted
job, or authenticated failed CPU/callback records. RUN/PEND is never a failure
or a pass. Full CPU acceptance derives actual testcase rows/logs, not the XML
header's inherited three unittest subtest surplus.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT = BASE + '/boltz-dbmol-pilot-01'
SOURCE = BASE + '/boltz-dbmol-pilot-release-01/source'
MANIFEST = '24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
ARCHIVE = 'ec333fd3e0b33d9e5b82e0cd5e4e12c14ffb4f91782c18cd820d4bfab53de1f8'
ARCHIVE_BYTES = 39676321
SOURCE_FILES = 657
SUITE = 'mosaic-dbmol-pilot-native-components-v1'
CONFIG = 'b3b56f409a59b3b1d00228bb7dcd3f74fc08f17fdf06facfc1469922778d5497'
FAILED_PREFLIGHT = '9822d97a999ba1232692a17e4e9308518eecf09e583988e93f8954be6d5d0c1f'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
REFERENCE = '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
ASSETS = {
    'affinity': ('/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt', 'dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e'),
    'structural': ('/work3/s234818/latentmol/cache/boltz2/boltz2_conf.ckpt', '090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1'),
    'prepared_context': ('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared/context.pt', 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'),
}
OLD_COUNTS = {'hpc-contract': 500, 'legacy': 239, 'native-mapping': 34, 'chemistry': 59}
NEW_MODULES = {'dbmol-native': ('test_dbmol_objective', 'test_dbmol_native_paths'),
               'dbmol-transport': ('test_dbmol_gradient_contract', 'test_dbmol_bridge', 'test_dbmol_pilot_contract')}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def safe(name):
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and '..' not in path.parts
            and '\\' not in name and ':' not in name, 'Unsafe archive member')


def scheduler(raw, job, queue, expected=None):
    compact = re.sub(r'\s+', '', raw)
    require(re.findall(r'Job<([^>]+)>', compact) == [job], 'Wrong/ambiguous scheduler job')
    require(re.findall(r'Queue<([^>]+)>', compact) == [queue], 'Wrong scheduler queue')
    states = re.findall(r'Status<([^>]+)>', compact)
    require(len(states) == 1, 'Missing actual scheduler state')
    if expected is not None:
        require(states == [expected], 'Unexpected original actual scheduler state')
    return states[0]


def resources(phase):
    if phase == 'cpu':
        return ['-q', 'hpc', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '01:00']
    if phase in ('native', 'segment'):
        return ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00' if phase == 'native' else '23:00']
    require(phase in ('after-cpu', 'after-native', 'after-segment', 'dispatch'), 'Unknown submitted phase')
    return ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']


def vector(label, phase, dependency):
    parent = ROOT + '/' + label
    args = ['bsub', *resources(phase), '-J', 'mosaic-dbmol-' + label,
            '-oo', parent + '/stdout.%J.log', '-eo', parent + '/stderr.%J.log']
    if dependency:
        args += ['-w', dependency]
    args += ['bash', '-l', SOURCE + '/scripts/hpc/' + ('dbmol_cpu.lsf' if phase == 'cpu' else 'dbmol_stage.lsf'), SOURCE, ROOT, MANIFEST]
    if phase != 'cpu':
        args += [phase, parent]
    return args


def lsf(raw, job, passed):
    matches = re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', raw, re.M)
    require(matches == [(job, 'Done' if passed else 'Exited')], 'Wrong same-job final LSF subject')
    if passed:
        require('Successfully completed.' in raw and 'Exited with exit' not in raw, 'Missing same-job LSF success')
    else:
        require(re.search(r'Exited with exit code\s+\d+', raw) is not None, 'Missing actual LSF failure')


def source_archive(package, manifest):
    path = REPO / 'results/hpc' / HERE.name / 'source01/source.tar.gz'
    require(path.stat().st_size == package['archive_bytes'] == ARCHIVE_BYTES
            and digest(path.read_bytes()) == package['archive_sha256'] == ARCHIVE,
            'Complete local source archive differs')
    members = {}
    with tarfile.open(path) as archive:
        for member in archive:
            safe(member.name)
            require(member.isfile() and member.name.startswith('source/'), 'Nonregular/foreign source member')
            name = member.name.removeprefix('source/')
            require(name not in members, 'Duplicate source member')
            members[name] = archive.extractfile(member).read()
    raw = members.pop('source_manifest.json')
    require(digest(raw) == MANIFEST and json.loads(raw) == manifest, 'Local/delivered source manifest differs')
    require(len(members) == SOURCE_FILES and set(members) == set(manifest['files']), 'Incomplete657 member source')
    for name, data in members.items():
        require(digest(data) == manifest['files'][name], 'Sealed source member changed: ' + name)
    parent_path = REPO / 'results/hpc/20261023-protenix-extended-01/source02/source.tar.gz'
    require(digest(parent_path.read_bytes()) == '8ba1c3370fee1f09b96af139dbf1403bb3c187c52d447ec4b0b23ebe4a65f63d', 'Parent641 source archive changed')
    old = {}
    with tarfile.open(parent_path) as archive:
        for member in archive:
            safe(member.name)
            require(member.isfile() and member.name.startswith('source/'), 'Unsafe parent641 member')
            name = member.name.removeprefix('source/')
            require(name not in old, 'Duplicate parent641 member')
            old[name] = archive.extractfile(member).read()
    parent_manifest = old.pop('source_manifest.json')
    require(digest(parent_manifest) == manifest['parent_manifest_sha256'] == 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00', 'Wrong authenticated641 parent')
    old_manifest = json.loads(parent_manifest)
    require(len(old) == 641 and set(old) == set(old_manifest['files']), 'Incomplete parent641 source')
    for name, data in old.items():
        require(digest(data) == old_manifest['files'][name], 'Parent641 member changed')
        require(name in members, 'A parent source member was removed')
    changed = {name: digest(data) for name, data in members.items() if old.get(name) != data}
    require(changed == package['changed_members'], 'Actual source delta differs from sealed package')
    inherited_changes = [name for name in old if old[name] != members[name]]
    require(inherited_changes == ['src/mosaic/small_molecule/chemistry/trajectory.py'], 'Unexpected inherited driver/scientific change')
    require(digest(members[inherited_changes[0]]) == '6ab7330726eed2d615f4d518f3a4918c2cdc84fc5c609b52eaa530533d47f64a', 'Declared optional native-name trajectory change differs')
    return members


def declared_tests(data):
    tree = ast.parse(data)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
            names.add(node.name)
        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            names.update(child.name for child in node.body
                         if isinstance(child, ast.FunctionDef) and child.name.startswith('test_'))
    return names


def cpu_groups(folder, source, files):
    groups, headers, coverage = {}, {}, {}
    wrapper = source['scripts/hpc/dbmol_cpu.lsf'].decode()
    inherited_modules = set(re.findall(r'tests/(test_[A-Za-z0-9_]+)\.py', wrapper.split('contracts=(', 1)[1].split(')', 1)[0]))
    for group in (*OLD_COUNTS, *NEW_MODULES):
        name = 'run/cpu/payload/' + group + '.xml'
        require(name in files, 'Required raw allocated XML missing: ' + group)
        parsed = ET.parse(folder / name)
        cases = list(parsed.iter('testcase'))
        require(cases and not any(list(parsed.iter(kind)) for kind in ('failure', 'error', 'skipped')), 'Allocated CPU failure/error/skip exists')
        count = len(cases)
        groups[group] = count
        headers[group] = sum(int(suite.get('tests', '0')) for suite in parsed.iter('testsuite'))
        log = (folder / name).with_suffix('.log').read_text(encoding='utf-8')
        require(re.search(r'\b' + str(count) + r'\s+passed\b', log), 'Raw testcase/log pass count differs')
        if group in OLD_COUNTS:
            require(count == OLD_COUNTS[group], 'Inherited832 actual case scope changed')
        if group == 'hpc-contract':
            found = {part for case in cases for part in case.get('classname', '').split('.')}
            require(inherited_modules <= found and all(case.get('classname') for case in cases), 'Inherited module attribution incomplete')
            require(headers[group] == count + 3, 'Inherited unittest subtest header surplus changed')
        else:
            require(headers[group] == count, 'Unexpected XML header/case discrepancy')
        if group == 'legacy':
            require(re.search(r'\b3\s+deselected\b', log), 'Historical three intentional deselections changed')
        else:
            require(not re.search(r'\b[1-9]\d*\s+deselected\b', log), 'New/inherited nonlegacy suite deselected tests')
        if group in NEW_MODULES:
            coverage[group] = {}
            wanted = set(NEW_MODULES[group])
            observed = set()
            for module in wanted:
                expected = declared_tests(source['tests/' + module + '.py'])
                selected = [case for case in cases if module in case.get('classname', '').split('.')]
                actual = {case.get('name', '').split('[', 1)[0] for case in selected}
                require(expected and selected and expected == actual, 'New source testcase attribution incomplete: ' + module)
                coverage[group][module] = {'functions': len(expected), 'actual_cases': len(selected)}
            for case in cases:
                modules = wanted.intersection(case.get('classname', '').split('.'))
                require(len(modules) == 1, 'Unattributed/ambiguous new testcase')
                observed.update(modules)
            require(observed == wanted, 'A new module was omitted')
    require(sum(OLD_COUNTS.values()) == 832, 'Internal inherited case scope error')
    return groups, headers, coverage


def local_submission(base, folder, files):
    result = read(base / 'submission-result.json')
    raw = (base / 'submission-stdout.bin').read_bytes()
    successful = []
    for line in raw.decode(errors='replace').splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('status') == 'submitted_not_passed':
            successful.append(value)
    if result['returncode'] == 0:
        require(len(successful) == 1, 'Missing unique actual successful stage/submission record')
        value = successful[0]
        require(value['root'] == ROOT and value['source'] == SOURCE and value['manifest'] == MANIFEST, 'Stage submission scope differs')
        return result, {'cpu': value['cpu']['job_id'], 'after-cpu': value['callback']['job_id']}, None
    require(not successful, 'Contradictory successful stage record and exit failure')
    admission_name = 'run/admission-02/outcome.json'
    if admission_name not in files:
        return result, {}, None
    actual = read(folder / admission_name)
    admission = base / 'admission-02'
    require(read(admission / 'result.json')['returncode'] == 0, 'Fresh admission outcome unknown/failed')
    payload = (admission / 'stdout.bin').read_bytes()
    require(payload.count(b'ADMISSION_BEGIN\n') == 1, 'Missing/ambiguous fresh admission sentinel')
    captured = json.loads(payload.split(b'ADMISSION_BEGIN\n', 1)[1])
    require(captured == read(admission / 'outcome.json') == actual,
            'Local original/remote actual admission records differ')
    require(actual['status'] == 'submitted_not_passed' and actual['source_manifest_sha256'] == MANIFEST,
            'Fresh admission source/status differs')
    require(digest((folder / 'run/storage-preflight/gate.json').read_bytes()) == FAILED_PREFLIGHT
            and read(folder / 'run/storage-preflight/gate.json')['status'] == 'failed',
            'Historical failed preflight changed or was relabeled')
    retired = read(folder / 'run/admission-02/upload-retirement.json')
    started = read(folder / 'run/admission-02/started.json')
    require(actual['prior_failed_gate_preserved'] == retired
            and retired['prior_failed_gate_sha256'] == started['prior_failed_gate_sha256'] == FAILED_PREFLIGHT
            and retired['retirement'] == 'exact_redundant_upload_removed_all_studies_and_source_directories_unchanged'
            and retired['immutable_source633_reverified'] is True,
            'Exact owned-upload retirement/old failed gate provenance missing')
    backup = (REPO / retired['local_backup']).resolve()
    require(backup.is_relative_to(REPO.resolve()) and backup.is_file()
            and backup.stat().st_size == retired['bytes']
            and digest(backup.read_bytes()) == retired['sha256'], 'Retired upload complete local backup changed')
    require(retired['retired_upload'] == BASE + '/boltz-lr0025-02.tar.gz'
            and retired['bytes'] == 39618948
            and retired['sha256'] == '40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa',
            'Retirement extends beyond exact third owned redundant upload')
    fresh = read(folder / 'run/admission-02/storage-preflight/gate.json')
    require(actual['fresh_gate'] == fresh and fresh['status'] == 'passed'
            and fresh['full_long_admission'] is False, 'Fresh admission gate missing or mis-scoped')
    for key, phase in (('cpu', 'cpu'), ('callback', 'after-cpu')):
        sub = actual[key]
        require(sub['outcome'] == 'submitted' and sub['returncode'] == 0
                and sub['phase'] == phase and sub['args'] == vector(phase, phase, sub['dependency'])
                and re.findall(r'Job <(\d+)>', sub['stdout']) == [sub['job_id']],
                'Actual fresh admission submission/vector differs')
        require(read(folder / 'run' / phase / 'submission.json') == sub,
                'Admission and own actual submission record differ')
    require(actual['cpu']['job_id'] == '29798549' and actual['callback']['job_id'] == '29798550'
            and actual['cpu']['dependency'] is None
            and actual['callback']['dependency'] == 'ended(29798549)', 'Wrong actual fresh CPU/callback IDs')
    return result, {'cpu': '29798549', 'after-cpu': '29798550'}, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edition', default='01')
    parser.add_argument('--expect-failure', action='store_true')
    parser.add_argument('--admission-only', action='store_true', help='Audit actual fresh admission/submission only, never CPU acceptance')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(not (args.expect_failure and args.admission_only), 'Choose one explicit evidence scope')
    require(args.edition.isdigit() and len(args.edition) == 2, 'Use a two-digit evidence edition')
    base = REPO / 'results/hpc' / HERE.name
    folder = base / ('cpu-evidence-' + args.edition)
    output = args.output or HERE / ('admission-verification.json' if args.admission_only else 'cpu-failure-verification.json' if args.expect_failure else 'cpu-verification.json')
    require(not output.exists(), 'Never overwrite an earlier verification')
    metadata, receipt = read(folder / 'metadata.json'), read(folder / 'delivery-receipt.json')
    require(digest((folder / 'metadata.json').read_bytes()) == receipt['metadata_sha256'], 'Delivered metadata changed')
    archive = Path(receipt['archive'])
    require(archive.resolve() == (base / ('cpu-evidence-' + args.edition + '.tar.gz')).resolve(), 'Unexpected evidence archive path')
    require(archive.stat().st_size == receipt['bytes'] and digest(archive.read_bytes()) == receipt['sha256'], 'Evidence archive bytes differ')
    files, remote = metadata['files'], {}
    require(receipt['listed_files'] == len(files), 'Delivered inventory count differs')
    for name, record in files.items():
        safe(name)
        expected = ROOT + '/' + name[4:] if name.startswith('run/') else SOURCE + '/' + name[7:] if name.startswith('source/') else None
        require(expected is not None and record['path'] == expected and expected not in remote, 'Foreign/duplicate remote proof path')
        data = (folder / name).read_bytes()
        require(len(data) == record['bytes'] and digest(data) == record['sha256'], 'Delivered member changed: ' + name)
        remote[expected] = record
    seen = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            safe(member.name)
            prefix = 'cpu-evidence-' + args.edition + '/'
            require(member.isfile() and member.name.startswith(prefix), 'Unsafe compact evidence member')
            name = member.name.removeprefix(prefix)
            require(name not in seen and name in set(files) | {'metadata.json'}, 'Unexpected/duplicate evidence member')
            seen.add(name)
            require(tar.extractfile(member).read() == (folder / name).read_bytes(), 'Archive/extracted member differs')
    require(seen == set(files) | {'metadata.json'} and len(seen) == receipt['safe_members'], 'Compact evidence incomplete')
    package = read(HERE / 'package.json')
    require(package['manifest_sha256'] == MANIFEST and package['source_files'] == SOURCE_FILES, 'Wrong sealed package')
    manifest_raw = (folder / 'source/source_manifest.json').read_bytes()
    require(digest(manifest_raw) == MANIFEST, 'Delivered657 source manifest differs')
    source = source_archive(package, json.loads(manifest_raw))
    require(metadata['source_integrity'] == {'status': 'passed', 'manifest_sha256': MANIFEST, 'files': SOURCE_FILES, 'readonly': True}, 'Remote source rehash/read-only seal failed')
    for name in ('examples/configs/dbmol_pilot.json', 'scripts/hpc/dbmol_cpu.lsf', 'scripts/hpc/dbmol_stage.lsf'):
        require((folder / 'source' / name).read_bytes() == source[name], 'Delivered config/wrapper differs from archive')
    require(digest(source['examples/configs/dbmol_pilot.json']) == CONFIG, 'Wrong pilot configuration')
    for key, (path, expected) in ASSETS.items():
        ref = metadata['asset_references'][key]
        require(ref['path'] == path and ref['sha256'] == expected and type(ref['bytes']) is int
                and ref['bytes'] > 0 and ref['downloaded'] is False, 'Remote-only native asset identity differs')
    staging = read(folder / 'run/source-staging.json')
    require(staging['status'] == 'passed' and staging['files'] == SOURCE_FILES
            and staging['manifest'] == MANIFEST and staging['archive_sha256'] == ARCHIVE
            and staging['archive_bytes'] == ARCHIVE_BYTES and staging['readonly'] is True
            and staging['original_saved_results_unchanged'] is True, 'Actual staging seal differs')
    stage_result, pinned_jobs, admission = local_submission(base, folder, files)
    jobs = read(folder / 'run/jobs.json') if metadata['jobs_file_present'] else {}
    observed = {}
    for label, job in jobs.items():
        sub = read(folder / 'run' / label / 'submission.json')
        require(sub['outcome'] == 'submitted' and sub['job_id'] == job and sub['returncode'] == 0
                and re.findall(r'Job <(\d+)>', sub['stdout']) == [job]
                and sub['args'] == vector(label, sub['phase'], sub['dependency']), 'Actual job/resource/source vector differs')
        if label in pinned_jobs:
            require(job == pinned_jobs[label], 'Initial job ID differs from original stage output')
        if sub['phase'] == 'after-cpu':
            require(sub['dependency'] == 'ended(' + jobs['cpu'] + ')', 'CPU callback dependency differs')
        elif sub['phase'] in ('after-native', 'after-segment'):
            producer = label.replace('-after-native-', '-native-').replace('-after-segment-', '-segment-')
            require(sub['dependency'] == 'ended(' + jobs[producer] + ')', 'Own-ended callback dependency differs')
        elif sub['phase'] in ('cpu', 'native', 'segment'):
            require(sub['dependency'] is None, 'Unexpected scientific interarm dependency')
        query = metadata['scheduler'][label]
        require(query['job_id'] == job, 'Scheduler inventory differs')
        observed[label] = scheduler(query['raw'], job, 'gpul40s' if sub['phase'] in ('native', 'segment') else 'hpc') if query['returncode'] == 0 else 'unavailable_live_history'
    gates, errors, budgets = {}, {}, {}
    for name in files:
        if name.endswith('/gate.json'):
            gate = read(folder / name)
            require(gate['reserve_bytes'] == 4294967296 and gate['file_reserve'] == 100000, 'Storage reserves weakened')
            gates[name] = gate['status']
            if gate['status'] == 'failed':
                errors[name] = gate.get('error', 'Inspect original failed quota fields; no exception string was stored')
            if 'work3' in gate:
                work = gate['work3']
                budgets[name] = {key: work.get(key) for key in (
                    'required_bytes', 'available_bytes_conservative', 'available_files_lower',
                    'filesystem_inode_reporting')}
                budgets[name]['projected_bytes'] = gate['projected_bytes']
                require(work['required_bytes'] == gate['projected_bytes'] + 4294967296, 'Required bytes/reserve arithmetic differs')
                inode = work['filesystem_inode_observation']
                if tuple(inode[key] for key in ('f_files', 'f_ffree', 'f_favail')) == (0, 0, 0):
                    require(work['filesystem_inode_reporting'] == 'unreported_all_zero', 'All-zero physical inode scope mislabeled')
                if gate['status'] == 'passed':
                    require(work['status'] == 'passed' and work['available_bytes_conservative'] >= work['required_bytes']
                            and work['available_files_lower'] >= 100000, 'Passed capacity proof incoherent')
    base_value = {'files_verified': len(files), 'source_members_verified': SOURCE_FILES,
                  'source_manifest_sha256': MANIFEST, 'source_archive_sha256': ARCHIVE,
                  'evidence_archive_sha256': receipt['sha256'], 'jobs': jobs,
                  'actual_scheduler': observed, 'storage_gates': gates,
                  'storage_errors_verbatim': errors, 'storage_budget_details': budgets,
                  'fresh_admission_sha256': digest((folder / 'run/admission-02/outcome.json').read_bytes()) if admission is not None else None,
                  'original_stage_returncode': stage_result['returncode'],
                  'asset_references': metadata['asset_references'],
                  'native_readiness_passed_by_this_audit': False, 'optimization_started_by_this_audit': False,
                  'scope': 'Independent stdlib byte/XML/schema/source/resource/scheduler/LSF audit only. No local model, scientific array or numerical test execution. Remote-only native assets were not downloaded.'}
    if args.admission_only:
        require(admission is not None and all(jobs.get(key) == value for key, value in pinned_jobs.items()),
                'Fresh actual admission not established')
        require(gates.get('run/storage-preflight/gate.json') == 'failed'
                and gates.get('run/admission-02/storage-preflight/gate.json') == 'passed',
                'Old failure/fresh pass scopes not independently preserved')
        value = {**base_value, 'status': 'passed_admission_evidence_verified',
                 'cpu_submitted': True, 'cpu_passed': False, 'cpu_acceptance_established_by_this_audit': False,
                 'cases_claimed_passed': None, 'preflight_historical_failure_preserved': True,
                 'fresh_admission_is_storage_reservation': False}
    elif args.expect_failure and not jobs:
        require(stage_result['returncode'] != 0 and not metadata['jobs_file_present'] and not pinned_jobs,
                'Preflight failure not established by original process/jobs evidence')
        require(not any(name.endswith('/submission.json') or name.endswith('/submission-started.json') for name in files),
                'Unknown actual submission blocks a no-job claim')
        require(gates.get('run/storage-preflight/gate.json') == 'failed', 'Actual preflight failure missing')
        value = {**base_value, 'status': 'failed_preflight_evidence_verified', 'preflight_passed': False,
                 'cpu_submitted': False, 'cpu_passed': False, 'cases_claimed_passed': None,
                 'original_stage_returncode': stage_result['returncode']}
    elif args.expect_failure:
        require(set(pinned_jobs) == {'cpu', 'after-cpu'} and all(jobs.get(k) == v for k, v in pinned_jobs.items()), 'Initial actual jobs not bound')
        failed = []
        for label in ('cpu', 'after-cpu'):
            raw = (folder / 'run' / label / ('stdout.' + jobs[label] + '.log')).read_text(encoding='utf-8')
            if 'Exited with exit code' in raw:
                lsf(raw, jobs[label], False)
                require(observed[label] in ('EXIT', 'unavailable_live_history'), 'Failed producer still live/pending')
                failed.append(label)
            else:
                lsf(raw, jobs[label], True)
                require(observed[label] in ('DONE', 'unavailable_live_history'), 'Completed producer still live/pending')
        require(failed, 'No actual failed CPU/callback; RUN/PEND is not failed')
        partial = {}
        for name in files:
            if name.startswith('run/cpu/payload/') and name.endswith('.xml'):
                try:
                    parsed = ET.parse(folder / name)
                    partial[name] = {key: len(list(parsed.iter(key))) for key in ('testcase', 'failure', 'error', 'skipped')}
                except ET.ParseError as error:
                    partial[name] = {'incomplete_xml': repr(error)}
        cpu_passed = False
        if 'cpu' not in failed and 'run/cpu-certificate.json' in files:
            groups, headers, coverage = cpu_groups(folder, source, files)
            certificate = read(folder / 'run/cpu-certificate.json')
            require(certificate['status'] == 'passed' and certificate['groups'] == groups
                    and certificate['cases'] == sum(groups.values()) and certificate['suite'] == SUITE
                    and certificate['source_manifest_sha256'] == MANIFEST, 'Callback failed but CPU certificate scope differs')
            for proof in certificate['proofs']:
                require(proof['path'] in remote and remote[proof['path']]['sha256'] == proof['sha256'],
                        'Failed callback CPU certificate proof omitted/changed')
            cpu_passed = True
        value = {**base_value, 'status': 'failed_attempt_evidence_verified', 'failed_producers': failed,
                 'cpu_submitted': True, 'cpu_passed': cpu_passed, 'partial_xml_counts': partial,
                 'cases_claimed_passed': sum(groups.values()) if cpu_passed else None}
    else:
        require((stage_result['returncode'] == 0 or admission is not None)
                and set(pinned_jobs) == {'cpu', 'after-cpu'}, 'Actual initial submissions not established')
        if admission is not None:
            require(gates.get('run/storage-preflight/gate.json') == 'failed'
                    and gates.get('run/admission-02/storage-preflight/gate.json') == 'passed',
                    'Fresh admission is not an authenticated replacement observation')
        require(all(jobs.get(k) == v for k, v in pinned_jobs.items()), 'Initial job IDs changed')
        certificate = read(folder / 'run/cpu-certificate.json')
        require(certificate['status'] == 'passed' and certificate['suite'] == SUITE and certificate['source_manifest_sha256'] == MANIFEST,
                'Wrong full CPU certificate')
        for proof in certificate['proofs']:
            require(proof['path'] in remote and remote[proof['path']]['sha256'] == proof['sha256'], 'CPU certificate proof omitted/changed')
        groups, headers, coverage = cpu_groups(folder, source, files)
        require(certificate['groups'] == groups and certificate['cases'] == sum(groups.values()), 'Actual six-group case counts differ')
        final = [proof for proof in certificate['proofs'] if re.fullmatch(r'scheduler-final-\d+\.txt', PurePosixPath(proof['path']).name)]
        require(len(final) == 1, 'Missing/ambiguous certificate-bound actual CPU final scheduler')
        scheduler((folder / 'run' / final[0]['path'].split(ROOT + '/', 1)[1]).read_text(), jobs['cpu'], 'hpc', 'DONE')
        for label in ('cpu', 'after-cpu'):
            require(observed[label] in ('DONE', 'unavailable_live_history'), 'CPU/callback still RUN/PEND or failed')
            lsf((folder / 'run' / label / ('stdout.' + jobs[label] + '.log')).read_text(encoding='utf-8'), jobs[label], True)
        assessment = read(folder / 'run/cpu/payload/assessment.json')
        require(assessment['status'] == 'passed' and assessment['passed'] == 832
                and assessment['source_manifest_sha256'] == MANIFEST, 'Inherited allocated832-case assessment differs')
        require(digest((folder / 'run/cpu/payload/pocket_definition.json').read_bytes()) == POCKET
                and digest((folder / 'run/cpu/payload/composition_reference.json').read_bytes()) == REFERENCE,
                'Original JQ1 pocket/composition reference changed')
        native = jobs.get('pilot-native-0000')
        callback = jobs.get('pilot-after-native-0000')
        require(native and callback, 'Fresh native/callback submissions not established')
        value = {**base_value, 'status': 'passed', 'cpu_passed': True,
                 'cpu_cases': sum(groups.values()), 'inherited_cases': 832,
                 'new_cases': sum(groups[key] for key in NEW_MODULES), 'groups': groups,
                 'nominal_declared_total': 1063, 'xml_header_counts': headers, 'new_source_test_coverage': coverage,
                 'xml_header_limitation': 'Actual testcase rows/logs are authoritative. The inherited contract header adds3 unittest subtests; they are not extra passed cases.',
                 'cpu_certificate_sha256': digest((folder / 'run/cpu-certificate.json').read_bytes()),
                 'fresh_native_submission': {'job': native, 'callback': callback, 'scheduler_state': observed['pilot-native-0000']}}
    output.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
