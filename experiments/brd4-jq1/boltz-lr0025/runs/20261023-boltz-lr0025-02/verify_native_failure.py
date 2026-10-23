"""Independent stdlib audit of the preserved pre-model native startup failures.

No scientific imports, arrays, numerical tests, SSH, repairs or submissions.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT = BASE + '/boltz-lr0025-02'
SOURCE = BASE + '/boltz-lr0025-release-02/source'
MANIFEST = '3689071a4e7dd135071ff5b420c026543c4d10887c951077dff63d498d9f812c'
CPU = 'b606f924365ece881fc7a348d4c677a4d70858ce92cc259c015a63d2b2dd45b9'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
REFERENCE = '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
JOBS = {'cpu': '29798538', 'after-cpu': '29798539', 'boltz2-native-0000': '29798605', 'boltz2-after-native-0000': '29798606', 'boltz2_chemistry-native-0000': '29798607', 'boltz2_chemistry-after-native-0000': '29798608'}
CONFIGS = {'boltz2': 'ecdf8a4103291f2f3a7bc7197dd394f03a46c3ccc74c72eb42dfc2b76eb13124', 'boltz2_chemistry': '3b3bc33d087f0c03a48c6bf89f03bc707d236ca0f64a95ba5ae2159d1918737c'}


def require(value, message):
    if not value: raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def safe(name):
    p = PurePosixPath(name)
    require(p.parts and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name, 'Unsafe archive member')


def scheduler(raw, job, queue, status):
    raw = re.sub(r'\s+', '', raw)
    require(re.findall(r'Job<([^>]+)>', raw) == [job] and re.findall(r'Queue<([^>]+)>', raw) == [queue] and re.findall(r'Status<([^>]+)>', raw) == [status], 'Persisted scheduler job/queue/status differs')


def lsf(raw, job, passed):
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', raw, re.M) == [(job, 'Done' if passed else 'Exited')], 'Same-job LSF subject differs')
    require(('Successfully completed.' in raw and 'Exited with exit' not in raw) if passed else bool(re.search(r'Exited with exit code\s+1\b', raw)), 'Same-job final LSF outcome missing')


def vector(label, phase, dependency):
    resource = ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00'] if phase == 'native' else ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']
    parent = ROOT + '/' + label
    result = ['bsub', *resource, '-J', 'mosaic-bol30k-' + label, '-oo', parent + '/stdout.%J.log', '-eo', parent + '/stderr.%J.log']
    if dependency: result += ['-w', dependency]
    return result + ['bash', '-l', SOURCE + '/scripts/hpc/boltz_serial_stage.lsf', SOURCE, ROOT, MANIFEST, phase, parent]


def main():
    output = HERE / 'native-failure-verification-01.json'
    require(not output.exists(), 'Never overwrite earlier outcome verification')
    base = REPO / 'results/hpc' / HERE.name
    folder = base / 'native-evidence-01'
    archive = base / 'native-evidence-01.tar.gz'
    metadata, receipt = read(folder / 'metadata.json'), read(folder / 'delivery-receipt.json')
    require(Path(receipt['archive']).resolve() == archive.resolve() and archive.stat().st_size == receipt['bytes'] == 90570 and sha(archive.read_bytes()) == receipt['sha256'] == 'badea0d29e26d40123c51e6efd8376567a5ed54e11f6d537daa8d9040dec6fc5', 'Compact archive changed')
    require(sha((folder / 'metadata.json').read_bytes()) == receipt['metadata_sha256'] == '2ecc7c5e6cb3253168eeb51df5323fb204c8ba8fc127cc86bc08fe5e1ce7d165', 'Compact metadata changed')
    files, remote, seen = metadata['files'], {}, set()
    require(len(files) == receipt['listed_files'] == 64, 'Wrong delivered proof inventory')
    for name, record in files.items():
        safe(name);parts = name.split('/', 1);parent = {'run': ROOT, 'source': SOURCE}.get(parts[0])
        require(parent and len(parts) == 2 and record['path'] == parent + '/' + parts[1] and record['path'] not in remote, 'Foreign/ambiguous proof')
        data = (folder / name).read_bytes()
        require(len(data) == record['bytes'] and sha(data) == record['sha256'], 'Delivered proof changed: ' + name)
        remote[record['path']] = record
    with tarfile.open(archive) as tar:
        for member in tar:
            safe(member.name)
            require(member.isfile() and member.name not in seen and member.name in set(files) | {'metadata.json'}, 'Unsafe/duplicate compact member')
            seen.add(member.name)
            require(tar.extractfile(member).read() == (folder / member.name).read_bytes(), 'Archive/extracted proof differs')
    require(seen == set(files) | {'metadata.json'} and len(seen) == receipt['safe_members'] == 65, 'Compact evidence incomplete')
    package = read(HERE / 'package.json');source_archive = base / 'source01/source.tar.gz'
    require(source_archive.stat().st_size == package['archive_bytes'] == 39618948 and sha(source_archive.read_bytes()) == package['archive_sha256'] == '40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa', 'Complete scientific source archive changed')
    source = {}
    with tarfile.open(source_archive) as tar:
        for member in tar:
            safe(member.name);name = member.name.removeprefix('source/')
            require(member.isfile() and member.name.startswith('source/') and name not in source, 'Unsafe source member')
            source[name] = tar.extractfile(member).read()
    manifest_raw = source.pop('source_manifest.json');manifest = json.loads(manifest_raw)
    require(sha(manifest_raw) == package['manifest_sha256'] == MANIFEST and (folder / 'source/source_manifest.json').read_bytes() == manifest_raw and len(source) == 633 and set(source) == set(manifest['files']), 'Source633 manifest/scope differs')
    for name, data in source.items(): require(sha(data) == manifest['files'][name], 'Source member changed: ' + name)
    require(metadata['source_integrity'] == {'status': 'passed', 'manifest_sha256': MANIFEST, 'files': 633, 'readonly': True}, 'Actual remote full-source seal failed')
    for name in files:
        if name.startswith('source/') and name != 'source/source_manifest.json': require((folder / name).read_bytes() == source[name.removeprefix('source/')], 'Delivered source execution bytes differ')
    require('git_revision_reference' not in manifest, 'Observed source manifest schema does not match reported failure')
    provenance = source['src/mosaic/small_molecule/provenance.py'].decode()
    require('"git_revision_reference"' in provenance and 'revision = json.loads(archived_manifest.read_text' in provenance, 'Unchanged provenance schema consumer differs')
    driver = ast.parse(source['examples/brd4_25_atoms.py'])
    main_function = next(n for n in driver.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    report_assignment = next(n for n in main_function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'report' for t in n.targets))
    require(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'source_identity' for n in ast.walk(report_assignment)), 'Original driver provenance call differs')
    first_main_try = next(n for n in main_function.body if isinstance(n, ast.Try))
    require(report_assignment.lineno < first_main_try.lineno, 'Failure is not before native/model execution try')
    certificate = read(folder / 'run/cpu-certificate.json')
    require(sha((folder / 'run/cpu-certificate.json').read_bytes()) == CPU and certificate['status'] == 'passed' and certificate['cases'] == 799 and certificate['source_manifest_sha256'] == MANIFEST and certificate['suite'] == 'boltz-two-arm-serial30k-v1', 'Original allocated CPU certificate changed')
    for proof in certificate['proofs']: require(proof['path'] in remote and remote[proof['path']]['sha256'] == proof['sha256'], 'CPU proof missing/changed')
    cpu_counts = {}
    for group, expected in [('hpc-contract', 467), ('legacy', 239), ('native-mapping', 34), ('chemistry', 59)]:
        p = folder / ('run/cpu/payload/' + group + '.xml');tree = ET.parse(p);cases = list(tree.iter('testcase'))
        require(len(cases) == expected and not any(list(tree.iter(k)) for k in ('failure', 'error', 'skipped')) and re.search(r'\b' + str(expected) + r'\s+passed\b', p.with_suffix('.log').read_text()), 'Original allocated CPU cases changed')
        cpu_counts[group] = len(cases)
    require(cpu_counts == certificate['groups'], 'Original CPU groups differ')
    require(sha((folder / 'run/cpu/payload/composition_reference.json').read_bytes()) == REFERENCE and sha((folder / 'run/cpu/payload/pocket_definition.json').read_bytes()) == POCKET, 'Original input hashes changed')
    require(read(folder / 'run/jobs.json') == JOBS and metadata['root'] == ROOT and metadata['source'] == SOURCE and metadata['production_prefixes'] == {}, 'Unknown actual production/job scope')
    outcomes = {}
    for arm, config_sha in CONFIGS.items():
        config_path = 'examples/configs/boltz_serial_' + arm + '.json'
        config = json.loads(source[config_path])
        require(sha(source[config_path]) == config_sha and config['initial_step_size'] == .0025 and config['steps'] == 30000 and config['acceptance_policy'] == 'fixed_step' and config['max_backtracks'] == 1, 'Requested scientific settings changed')
        if arm == 'boltz2_chemistry': require(config['repair_schedule'] == {'version': 'delayed-geometric-v1', 'initial': .001, 'final': 100., 'hold_updates': 5000, 'total_updates': 30000}, 'User-selected chemistry ramp changed')
        ids = {'native': JOBS[arm + '-native-0000'], 'after-native': JOBS[arm + '-after-native-0000']}
        actual = {}
        for phase, job in ids.items():
            label = arm + '-' + phase + '-0000';parent = folder / 'run' / label;sub = read(parent / 'submission.json');dependency = None if phase == 'native' else 'ended(' + ids['native'] + ')'
            require(sub['job_id'] == job and sub['phase'] == phase and sub['outcome'] == 'submitted' and sub['returncode'] == 0 and sub['dependency'] == dependency and sub['args'] == vector(label, phase, dependency) and re.findall(r'Job <(\d+)>', sub['stdout']) == [job], 'Actual producer/resource/own-ended vector differs')
            log = (parent / ('stdout.' + job + '.log')).read_text();lsf(log, job, False)
            query = metadata['scheduler'][label]
            require(query['job_id'] == job and query['returncode'] == 0 and re.fullmatch(r'Job\s+<' + re.escape(job) + r'>\s+is\s+not\s+found\s*', query['raw']), 'Current missing live history differs; never fabricate scheduler status')
            error = (parent / ('stderr.' + job + '.log')).read_text()
            require(("KeyError: 'git_revision_reference'" in error and SOURCE + '/src/mosaic/small_molecule/provenance.py' in error and SOURCE + '/examples/brd4_25_atoms.py' in error) if phase == 'native' else 'ValueError: Actual producer failed/not final' in error, 'Actual startup/dependency failure differs')
            actual[phase] = {'job': job, 'persisted_lsf_status': 'Exited', 'exit_code': 1, 'live_scheduler_status': 'unavailable_live_history', 'stdout_sha256': sha((parent / ('stdout.' + job + '.log')).read_bytes()), 'stderr_sha256': sha((parent / ('stderr.' + job + '.log')).read_bytes())}
        parent = folder / 'run' / (arm + '-native-0000')
        final_proofs = []
        for name in sorted(files):
            if name.startswith('run/' + arm + '-native-0000/scheduler-final-'):
                raw = (folder / name).read_text();compact = re.sub(r'\s+', '', raw)
                if re.findall(r'Status<([^>]+)>', compact) == ['EXIT']:
                    scheduler(raw, ids['native'], 'gpul40s', 'EXIT');final_proofs.append(files[name])
        require(final_proofs, 'No authentic persisted native final EXIT scheduler')
        invocation = read(parent / 'payload/driver-invocation.json')
        expected_args = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', SOURCE + '/examples/brd4_25_atoms.py', '--mode', 'validate', '--config', SOURCE + '/' + config_path, '--output', ROOT + '/' + arm + '-native-0000/payload/study', '--manifest', MANIFEST, '--cpu-certificate', ROOT + '/cpu-certificate.json', '--cpu-certificate-sha256', CPU, '--reference', ROOT + '/cpu/payload/composition_reference.json', '--pocket', ROOT + '/cpu/payload/pocket_definition.json', '--pocket-sha256', POCKET, '--prepared', '/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared', '--context-sha256', 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093', '--worker-python', '/work3/s234818/latentmol/env/boltz311/bin/python', '--offset', '0', '--segment-steps', '10000']
        require(invocation == {'args': expected_args, 'original_saved_runs_read_only': True, 'jax': 'cpu_fp32', 'worker': 'cuda_frozen_affinity'}, 'Exact failed native driver argv differs')
        gate_path = parent / 'storage-entry/gate.json';gate = read(gate_path);work = gate['work3']
        require(gate['status'] == work['status'] == 'passed' and gate['reserve_bytes'] == 4294967296 and gate['file_reserve'] == 100000 and work['available_bytes_conservative'] >= work['required_bytes'] == gate['projected_bytes'] + 4294967296 and work['available_files_lower'] >= 100000, 'Original native entry storage proof changed')
        require(not any(name.startswith('run/' + arm + '-native-0000/payload/study/') for name in files) and 'run/' + arm + '-native-0000-certificate.json' not in files and 'run/projection.json' not in files, 'Unexpected native scientific artifacts/projection')
        outcomes[arm] = {'status': 'failed_before_model_at_provenance_metadata', 'error_verbatim': "KeyError: 'git_revision_reference'", 'native': actual['native'], 'callback': actual['after-native'], 'persisted_native_final_scheduler_proofs': final_proofs, 'config_sha256': config_sha, 'driver_invocation_sha256': sha((parent / 'payload/driver-invocation.json').read_bytes()), 'entry_storage_gate_sha256': sha(gate_path.read_bytes()), 'entry_storage': {'status': 'passed', 'required_bytes': work['required_bytes'], 'available_bytes_conservative': work['available_bytes_conservative'], 'scope': 'Native validation allowance only, not full optimization admission/reservation'}, 'native_gradient_checks_started': False, 'native_readiness_passed': False, 'production_submitted': False, 'optimization_started': False, 'committed_updates': 0}
    value = {'version': 'boltz30k-native-startup-failure-byte-audit-v1', 'status': 'failed_native_startup_evidence_verified', 'verified_utc': datetime.now(timezone.utc).isoformat(), 'files_verified': len(files), 'safe_archive_members': len(seen), 'source_members_verified': len(source), 'evidence_archive_sha256': receipt['sha256'], 'source_manifest_sha256': MANIFEST, 'source_archive_sha256': package['archive_sha256'], 'cpu_certificate_sha256': CPU, 'cpu_cases_preserved': cpu_counts, 'cpu_cases_passed': 799, 'native_outcomes': outcomes, 'failure_classification': 'Engineering metadata startup failure before model/worker/native gradients; not a failed gradient numerical comparison', 'scientific_status': 'native_readiness_not_established', 'saved_results_changed': False, 'source_changed': False, 'new_submissions_by_this_audit': False, 'strict_fix_recommendation': 'Preserve original failed proofs and all sealed sources. Prepare a NEW immutable release/manifest with the required git_revision_reference field derived from authenticated revision metadata (never invent a revision). Before any fresh scientific launch, require an allocated startup regression executing the actual staged command and its manifest/provenance path; then rerun every required CPU/native/scientific/storage check at unchanged settings. Do not patch old manifests, relabel old failures, or submit work during this status audit.', 'scope': 'Independent local stdlib archive/member/source/JSON/XML/schema/submission/resource/LSF/persisted-scheduler/byte-gate audit only. No arrays, numerical tests, model execution, SSH, fixes or submissions.'}
    with output.open('x', encoding='utf8') as stream:json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps({'status': value['status'], 'receipt': str(output), 'sha256': sha(output.read_bytes()), 'files_verified': len(files), 'source_members_verified': len(source), 'native_outcomes': {arm: result['status'] for arm, result in outcomes.items()}}, indent=2))


if __name__ == '__main__':
    main()
