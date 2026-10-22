"""Independently verify downloaded report delivery using only stdlib bytes/schema.

Edition-0002 verifier: exact new report allocation, all three full saved budgets,
and truthful optional chemistry terminal evidence. This never decodes FP32 arrays,
reruns simplex/argmax checks or imports scientific or report implementation modules.
Those checks remain allocated-HPC evidence. The edition-0001 verifier is preserved.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

TAG = '20261022-brd4-jq1-lrboost-analysis-01'
MANIFEST = '214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc'
SCIENCE = '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323'
CPU_CERTIFICATE = '42b2b000ecce5d661da9b75ad85f03a41c3b303010ccfa03f87835a25993ca69'
CPU_VERIFICATION = 'ac8d58f18e84cb15823e7e68077c0ac04cfa82e35be129a6c44ddf6da3a080e9'
VERSION = 'jq1-25-lrboost-saved-report-v1'
ROOT = '/work3/s234818/mosaic/runs/' + TAG
SOURCE = '/work3/s234818/mosaic/releases/' + TAG + '/source'
REPORT_JOB = '29781041'
PARENT = 'interim-0002'
EVIDENCE_ARCHIVE = 'interim-delivery-evidence-02.tar.gz'
ENTRY_GATE = '6b7fd2ec8a3311b1a5bb0531786c710a554d10a7d11327231daa47961edd406a'
DISPATCH_GUARD = 'af0f4a4a968dd85d13f3a06e6e2d6d3ea268a83bab6afaa554632463cb7d5d3a'
SUBMISSION_STDOUT = 'fa1d9434fde371a0f818c0d14bed4d16511bdc4996d95288fb74262ccc39ff71'
SUBMISSION_RESULT = '58f00c59a73cc828d8f74dfad323de155ee8f73110c03d3c2e17f506ff8bf33d'
HEX = re.compile(r'[0-9a-f]{64}')
MODULES = {'test_jq1_lrboost_report_contract': 82, 'test_jq1_lrboost_render': 54,
           'test_jq1_lrboost_publish': 59, 'test_jq1_lrboost_ops': 32,
           'test_lrboost_endpoint_diagnostics': 16, 'test_boltz_affinity_report': 26,
           'test_small_molecule_chemistry_report': 3, 'test_small_molecule_chemistry_ramp_report': 6}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant: ' + value)))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name and '\x00' not in name, 'Unsafe member name')
    pure = PurePosixPath(name)
    require(not pure.is_absolute() and str(pure) == name and all(part not in ('', '.', '..') for part in pure.parts), 'Unsafe member path')
    require(all(part.rstrip(' .') == part and part.split('.')[0].upper() not in {'CON', 'PRN', 'AUX', 'NUL', *('COM' + str(i) for i in range(1, 10)), *('LPT' + str(i) for i in range(1, 10))} for part in pure.parts), 'Unsafe platform member path')
    return pure


def local(root, name):
    pure = safe_name(name)
    path = root.joinpath(*pure.parts)
    require(path.resolve().is_relative_to(root.resolve()), 'Member escaped extraction')
    for item in (path, *path.parents):
        if item == root.parent:
            break
        require(not item.is_symlink(), 'Symlink in verified extraction')
    return path


def archive_tree(archive, tree, expected):
    require(archive.is_file() and not archive.is_symlink() and tree.is_dir() and not tree.is_symlink(), 'Missing regular archive/extraction')
    require(type(expected.get('bytes')) is int and archive.stat().st_size == expected['bytes']
            and HEX.fullmatch(expected.get('sha256', '')) and digest(archive) == expected['sha256'], 'Archive receipt bytes/hash differ')
    members = {}
    with tarfile.open(archive, 'r:*') as handle:
        for member in handle.getmembers():
            safe_name(member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and member.name not in members, 'Unsafe or duplicate archive member')
            with handle.extractfile(member) as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            members[member.name] = {'bytes': member.size, 'sha256': checksum}
    manifest_path = local(tree, 'evidence_manifest.json')
    require('evidence_manifest.json' in members and digest(manifest_path) == members['evidence_manifest.json']['sha256'], 'Extracted evidence manifest differs from archive')
    manifest = read(manifest_path)
    mapping = manifest['files']
    require(isinstance(mapping, dict) and mapping and set(members) == set(mapping) | {'evidence_manifest.json'}, 'Archive member map differs')
    actual = set()
    for path in tree.rglob('*'):
        require(not path.is_symlink(), 'Extracted symlink forbidden')
        if path.is_file():
            actual.add(path.relative_to(tree).as_posix())
    require(actual == set(members), 'Unexpected or missing extracted files')
    for name, pin in members.items():
        path = local(tree, name)
        require(path.is_file() and path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256'], 'Extracted member bytes changed: ' + name)
        if name != 'evidence_manifest.json':
            require(mapping[name] == pin, 'Manifest member hash/length differs: ' + name)
    if 'files' in expected:
        require(expected['files'] == len(mapping), 'Declared archive file count differs')
    return manifest, members


def scheduler(path, job, queue='hpc'):
    compact = re.sub(r'\s+', '', path.read_text(encoding='utf-8'))
    require(re.findall(r'Job<([^>]+)>', compact) == [job]
            and re.findall(r'Status<([^>]+)>', compact) == ['DONE']
            and re.findall(r'Queue<([^>]+)>', compact) == [queue], 'Missing actual same-job DONE/queue proof: ' + str(path))


def lsf(path, job, *, failed=False):
    text = path.read_text(encoding='utf-8')
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', text, re.M)
    require(len(subjects) == 1 and subjects[0][0] == job, 'Wrong or ambiguous same-job LSF')
    if failed:
        require(subjects[0][1].rstrip().endswith(' Exited') and re.search(r'^Exited with exit code 1\.\s*$', text, re.M)
                and 'Successfully completed.' not in text, 'Original scientific LSF failure changed')
    else:
        require(subjects[0][1].rstrip().endswith(' Done') and re.search(r'^Successfully completed\.\s*$', text, re.M)
                and not re.search(r'Exited\s+(?:with exit|by signal)', text, re.I), 'Missing successful allocated LSF')



def render_submission(parent, evidence, report_job, parent_name):
    """Authenticate the newly submitted interim without an old CPU submission pin."""
    receipt_dir = Path(__file__).resolve().parent
    stdout = receipt_dir / 'submit-interim-0002-attempt02-stdout.txt'
    result_path = receipt_dir / 'submit-interim-0002-attempt02-result.json'
    require(digest(stdout) == SUBMISSION_STDOUT and digest(result_path) == SUBMISSION_RESULT,
            'Previously saved actual submission command receipts changed')
    result = read(result_path)
    require(result['outcome'] == 'completed' and type(result['returncode']) is int and result['returncode'] == 0
            and result['script'] == 'submit-interim-0002-attempt02.sh'
            and result['script_sha256'] == digest(receipt_dir / result['script'])
            and result['command'] == ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o',
                                      'ConnectionAttempts=1', 'dtu-hpc', 'bash -l -s'],
            'Original successful single-connection actual submission receipt differs')
    records = [json.loads(line) for line in stdout.read_text(encoding='utf-8').splitlines() if line.startswith('{')]
    require(len(records) == 1 and records[0]['status'] == 'actual_submitted'
            and records[0]['guard_sha256'] == DISPATCH_GUARD, 'No unique original successful new report submission')
    submission = read(parent / 'submission.json')
    started = read(parent / 'submission-started.json')
    require(records[0]['submission'] == submission, 'Downloaded new submission differs from original locally saved actual result')
    guard_path = evidence / 'analysis/interim-0002-dispatch-guard.json'
    require(digest(guard_path) == DISPATCH_GUARD, 'Original immutable new dispatch guard bytes changed')
    guard = read(guard_path)
    require(guard['name'] == parent_name and guard['source_manifest_sha256'] == MANIFEST
            and guard['cpu_certificate_sha256'] == CPU_CERTIFICATE and guard['new_chemistry_milestone'] == 10000
            and guard['published_prior_report_receipt_sha256'] == 'e0d07ee7618aacd9dfafa0c374b76530b04c727b2a2b552e2b356468b50e0da7'
            and guard['published_milestones'] == {'protenix': 10000, 'boltz2': 10000, 'boltz2_chemistry': 9500},
            'New interim dispatch guard source/CPU/prior-publication/milestone identity differs')
    for field, count, last in [('states', 10001, 10000), ('updates', 10000, 9999)]:
        prefix = guard['prefixes'][field]
        require(prefix['rows'] == count and prefix['last_step'] == last and type(prefix['bytes']) is int
                and prefix['bytes'] > 0 and HEX.fullmatch(prefix['sha256']), 'Guard did not authenticate complete committed chemistry budget')
    require(records[0]['jobs']['active_interim'] == report_job and records[0]['jobs']['interim_parent'] == parent_name
            and records[0]['jobs']['final'] == '29780830', 'Original actual-submission job map differs')
    remote_parent = ROOT + '/' + parent_name
    expected = ['bsub', '-q', 'hpc', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]', '-W', '01:00',
                '-J', 'mosaic-lrboost-analysis-' + parent_name,
                '-oo', remote_parent + '/stdout.%J.log', '-eo', remote_parent + '/stderr.%J.log',
                'bash', '-l', SOURCE + '/scripts/hpc/jq1_lrboost_analysis_stage.lsf', SOURCE, ROOT,
                MANIFEST, 'interim', 'none', remote_parent]
    require(submission['job_id'] == report_job and submission['phase'] == 'interim' and submission['arm'] == 'none'
            and submission['dependency'] is None and submission['outcome'] == 'submitted'
            and type(submission['returncode']) is int and submission['returncode'] == 0
            and submission['args'] == expected
            and re.findall(r'Job <(\d+)> is submitted to queue <hpc>\.', submission['stdout']) == [report_job],
            'Actual interim submission/resource/source identity differs')
    require(started == {key: submission[key] for key in ('phase', 'arm', 'args', 'dependency', 'utc')} | {'outcome': 'unknown'},
            'Exclusive submission-start marker differs from completed receipt')
    raw = re.sub(r'\s+', '', (parent / 'scheduler-at-submission.txt').read_text(encoding='utf-8'))
    require(re.findall(r'Job<([^>]+)>', raw) == [report_job] and re.findall(r'Queue<([^>]+)>', raw) == ['hpc']
            and re.findall(r'Status<([^>]+)>', raw) in (['PEND'], ['RUN'], ['DONE']),
            'Missing actual initial scheduler identity')
    origin = read(evidence / 'origin-map.json')
    for name in ('submission.json', 'submission-started.json', 'scheduler-at-submission.txt'):
        require(origin.get('analysis/' + parent_name + '/' + name) == remote_parent + '/' + name,
                'Delivery origin map differs from exact interim parent')
    return submission



def report_entry_gate(evidence):
    """Verify the allocated new entry budget, retaining honest inode evidence scope."""
    name = 'analysis/interim-0002/storage-interim/gate.json'
    path = evidence / name
    require(path.stat().st_size == 3462 and digest(path) == ENTRY_GATE,
            'Original allocated interim0002 entry-gate bytes changed')
    gate = read(path)
    require(gate['status'] == 'passed' and gate['phase'] == 'interim'
            and gate['required_bytes'] == 8306479405 and gate['available_bytes_conservative'] == 9534557045
            and gate['reserve_bytes'] == 4294967296 and gate['file_reserve'] == 100000,
            'New allocated report entry byte/reserve gate differs or failed')
    quota, inode = gate['quota'], gate['inode']
    pools = quota['finite_pools']
    require(isinstance(pools, list) and pools
            and all(type(pool.get('remaining_files')) is int and pool['remaining_files'] >= 100000 for pool in pools)
            and min(pool['remaining_files'] for pool in pools) == quota['available_files_lower'] == 19597404
            and inode['available_files_lower'] == 19597404,
            'New entry gate lacks the mandatory all-finite-user-quota file minimum')
    require(gate['filesystem_inode_observation'] == {'f_files': 0, 'f_ffree': 0, 'f_favail': 0}
            and inode['filesystem_inode_reporting'] == 'unreported_all_zero'
            and inode['file_headroom_scope'] == 'Mandatory minimum of all authenticated finite user-quota pools'
            and inode['inode_limitation'] == 'All-zero statvfs tuple supplies no independent filesystem inode proof',
            'Unreported supplemental filesystem inode tuple was relabelled as independent capacity proof')
    require(gate['full_native_projections'] == {'protenix': 1514341824, 'boltz2': 1820641824}
            and gate['analysis_allowance_bytes'] == 2684354560 and gate['diagnostic_per_arm_hard_output_bytes'] == 268435456,
            'Full remaining native/report/diagnostic projections were reduced')
    require(read(evidence / 'origin-map.json').get(name) == ROOT + '/interim-0002/storage-interim/gate.json',
            'New entry-gate original path differs')
    return {'path': ROOT + '/interim-0002/storage-interim/gate.json', 'bytes': 3462, 'sha256': ENTRY_GATE,
            'status': gate['status'], 'phase': gate['phase'], 'required_bytes': gate['required_bytes'],
            'available_bytes_conservative': gate['available_bytes_conservative'], 'reserve_bytes': gate['reserve_bytes'],
            'file_reserve': gate['file_reserve'], 'finite_user_quota_file_minimum': 19597404,
            'filesystem_inode_reporting': inode['filesystem_inode_reporting'],
            'inode_limitation': inode['inode_limitation']}


def native_terminal_lsf(path, job, *, passed):
    """Authenticate own original LSF; never manufacture a terminal scheduler state."""
    if passed:
        lsf(path, job)
        return
    text = path.read_text(encoding='utf-8')
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', text, re.M)
    codes = re.findall(r'^Exited with exit code (\d+)\.\s*$', text, re.M)
    require(len(subjects) == 1 and subjects[0][0] == job and subjects[0][1].rstrip().endswith(' Exited')
            and len(codes) == 1 and int(codes[0]) != 0 and 'Successfully completed.' not in text,
            'Chemistry failed outcome lacks exact same-job nonzero original LSF')


def chemistry_collector(evidence, payload, pin, cpu_expected):
    """Optional independently captured terminal proof; full saved steps alone do not pass science."""
    collector = evidence / 'analysis/complete-boltz2_chemistry'
    if not (collector / 'payload/assessment.json').exists():
        return None
    arm, job = 'boltz2_chemistry', '29780827'
    scheduler(evidence / ('collector-' + arm + '-scheduler.txt'), job)
    lsf(collector / ('stdout.' + job + '.log'), job)
    require(digest(collector / 'submission.json') == cpu_expected['files']['analysis/complete-' + arm + '/submission.json']['sha256'],
            'Chemistry collector original submitted identity differs')
    captured = read(collector / 'payload/assessment.json')
    require(captured['status'] == captured['presentation_status'] == 'passed' and captured['presentation_complete'] is True
            and captured['collector_job'] == job and captured['native_job'] == pin['job'] and captured['arm'] == arm
            and captured['scientific_status'] in ('passed', 'failed')
            and captured['scientific_passed'] is (captured['scientific_status'] == 'passed')
            and captured['states'] == 10001 and captured['updates'] == 10000
            and captured['n_atoms'] == 25 and captured['learning_rate'] == pin['rate']
            and captured['all_probability_frames_verified'] is True and captured['report_source_manifest_sha256'] == MANIFEST
            and captured['cpu_certificate_sha256'] == CPU_CERTIFICATE
            and captured['source_manifest_sha256'] == SCIENCE and captured['config_sha256'] == pin['config']
            and captured['readiness_sha256'] == pin['ready'] and captured['model_digest'] == pin['model']
            and captured['context_digest'] == pin['context']
            and captured['acceptance_policy'] == captured['selection_policy'] == 'fixed_step',
            'Chemistry terminal collector full-frame/source/CPU/outcome scope differs')
    launcher(arm, pin, captured['launcher'])
    proofs = proof_schema(captured['proofs'])
    study = pin['output_parent'] + '/payload/study'
    for suffix in ('summary.json', 'case-0/report.json', 'case-0/states.jsonl', 'case-0/updates.jsonl',
                   'native_gradient_checks/report.json', 'terminal_gradient_checks/report.json',
                   'case-0/combined_initial_checks/report.json', 'initialization.npz', 'pocket_definition.json'):
        require(study + '/' + suffix in proofs, 'Chemistry terminal collector misses original proof: ' + suffix)
    if captured['scientific_passed']:
        original_summary = next(record for record in captured['proofs'] if record['path'] == study + '/summary.json')
        require(original_summary['bytes'] == 311775359
                and original_summary['sha256'] == 'aa96fa193252b2205a89f31fefc352b4303a4878fe04f314146cbe8521c05cd0',
                'Previously observed exact original passed chemistry summary reference differs')
    stdout = collector / 'payload/native-stdout.txt'
    require(proofs.get(ROOT + '/complete-' + arm + '/payload/native-stdout.txt') == digest(stdout),
            'Chemistry collector does not bind original LSF copy')
    native_terminal_lsf(stdout, pin['job'], passed=captured['scientific_passed'])
    outcome_value = captured['outcome']
    outcome(outcome_value | {'latest_update': captured['updates']})
    require(outcome_value['all_terminal'] is True and outcome_value['scientific_passed'] is captured['scientific_passed']
            and outcome_value['overall_status'] == captured['scientific_status']
            and captured['native_status'] in (('DONE', 'SUCCESS_PERSISTED') if captured['scientific_passed'] else ('EXIT', 'FAILED_PERSISTED')),
            'Chemistry terminal scope/own native status differs')
    inventory_path = collector / 'payload/snapshots.json'
    inventory = read(inventory_path)
    require(inventory['status'] == 'passed' and inventory['native_job'] == pin['job'] and inventory['n_atoms'] == 25
            and len(inventory['frames']) == 10001 and [frame['step'] for frame in inventory['frames']] == list(range(10001)),
            'Chemistry terminal collector lacks complete original frame inventory')
    require(all(type(frame.get('bytes')) is int and frame['bytes'] > 0 and HEX.fullmatch(frame.get('sha256', ''))
                and frame['path'].startswith(study + '/case-0/') for frame in inventory['frames']),
            'Malformed chemistry original frame hash references')
    require(proofs.get(ROOT + '/complete-' + arm + '/payload/snapshots.json') == digest(inventory_path),
            'Chemistry collector proof does not bind original saved-frame inventory')
    copied = payload / ('completion-' + arm)
    if copied.exists():
        for name in ('assessment.json', 'snapshots.json', 'native-stdout.txt'):
            require((copied / name).read_bytes() == (collector / 'payload' / name).read_bytes(),
                    'Bundled chemistry collector copy differs')
    return {'assessment': captured, 'frames': inventory['frames'], 'proofs': proofs}


def proof_schema(proofs):
    require(isinstance(proofs, list) and proofs and len({p['path'] for p in proofs}) == len(proofs), 'Missing/duplicate proof paths')
    require(all(isinstance(p.get('path'), str) and p['path'].startswith('/') and HEX.fullmatch(p.get('sha256', '')) for p in proofs), 'Malformed remote proof bindings')
    return {proof['path']: proof['sha256'] for proof in proofs}


def launcher(arm, pin, value):
    require(value['arm'] == arm and value['native_job'] == pin['job'] and value['source_manifest_sha256'] == SCIENCE
            and value['config_sha256'] == pin['config'] and value['readiness_sha256'] == pin['ready']
            and value['cpu_certificate_sha256'] == 'dc76df74f1432f59e4c5f6fb5491a1258752b639df893cf75bb5a6968e481630'
            and value['scientific_execution_started'] is True, 'Scientific input provenance differs')
    for field, name, original_path in [('submission', 'submission', pin['output_parent'] + '/submission.json'),
                                      ('entry', 'entry', pin['science_root'] + '/entry-quota-' + arm + '/lrboost-storage-gate.json'),
                                      ('driver_invocation', 'driver', pin['output_parent'] + '/payload/driver-invocation.json')]:
        require(value[field]['path'] == original_path and value[field]['sha256'] == pin[name], 'Original execution proof binding changed')


def cpu_certificate(payload, evidence, cpu_expected):
    cpu_path = payload / 'cpu-proof/assessment.json'
    require(digest(cpu_path) == CPU_CERTIFICATE and cpu_path.read_bytes() == (evidence / 'analysis/cpu-certificate.json').read_bytes(), 'Report CPU certificate differs')
    cert = read(cpu_path)
    require(cert['status'] == 'passed' and cert['producer_job_id'] == cert['job_id'] == '29780814'
            and cert['source_manifest_sha256'] == MANIFEST and cert['suite'] == 'jq1-lrboost-saved-report-contract-v1'
            and cert['tests'] == 278 and cert['tests_by_module'] == MODULES, 'Wrong report CPU suite/count/source')
    originals = proof_schema(cert['proofs'])
    bundled = read(payload / 'cpu-proof/bundled-proofs.json')
    require(len(bundled) == len(originals) == 5, 'Missing exact bundled CPU proof set')
    paths = []
    for item in bundled:
        require(item['original_path'] in originals and item['sha256'] == originals[item['original_path']], 'CPU original proof binding differs')
        path = local(payload, item['local_path'])
        require(path.is_relative_to(payload / 'cpu-proof') and digest(path) == item['sha256'], 'Bundled CPU proof bytes differ')
        paths.append(path)
    require(len({item['original_path'] for item in bundled}) == 5, 'Duplicate bundled CPU proof')
    scheduler(next(path for path in paths if path.name == 'scheduler-final-0.txt'), '29780814')
    lsf(next(path for path in paths if path.name == 'stdout.29780814.log'), '29780814')
    xml_path = next(path for path in paths if path.suffix == '.xml')
    xml = ET.parse(xml_path)
    rows = list(xml.iter('testcase'))
    require(len(rows) == 278 and not any(list(xml.iter(name)) for name in ('failure', 'error', 'skipped')), 'Allocated CPU XML did not pass278 without omissions')
    counts = Counter(next(part for part in row.attrib['classname'].split('.') if part.startswith('test_')) for row in rows)
    require(counts == MODULES, 'Raw CPU module counts differ')
    log = next(path for path in paths if path.name == 'report-tests.log').read_text()
    require('collected 278 items' in log and '278 passed in ' in log, 'Allocated CPU collection/passes differ')
    require(cpu_expected['cpu']['certificate_sha256'] == digest(cpu_path), 'Independent previous CPU verification binding differs')
    return cert


def outcome(study, *, known_failed=False):
    require(study['readiness_status'] == 'passed' and type(study['scientific_passed']) is bool and type(study['all_terminal']) is bool,
            'Readiness/outcome scope malformed')
    require(study['terminal_status'] in ('passed', 'failed', 'not_reached', 'pending')
            and study['overall_status'] in ('passed', 'failed', 'running', 'queued', 'preparing')
            and study['budget_status'] in ('full', 'partial', 'not_started'), 'Unknown saved outcome')
    if known_failed:
        require(study['states'] == 10001 and study['latest_update'] == 10000 and study['budget_status'] == 'full'
                and study['terminal_status'] == study['overall_status'] == 'failed' and study['scientific_passed'] is False
                and study['all_terminal'] is True and study['native_status'] in ('FAILED_PERSISTED', 'EXIT'), 'Known full-budget scientific failure hidden')
    elif study['scientific_passed']:
        require(study['states'] == 10001 and study['latest_update'] == 10000 and study['terminal_status'] == study['overall_status'] == 'passed'
                and study['all_terminal'] and study['native_status'] in ('DONE', 'SUCCESS_PERSISTED'), 'Scientific pass lacks full native outcome')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('delivery-receipt', 'report-archive', 'report-tree', 'evidence-archive', 'evidence-tree', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--report-job', required=True, help='Exact new interim job; must match this edition verifier')
    parser.add_argument('--parent', required=True, help='Exact new interim parent basename')
    args = parser.parse_args()
    require(args.report_job == REPORT_JOB and args.parent == PARENT, 'Verifier is pinned to actual interim-0002 job29781041')
    receipt_dir = Path(__file__).resolve().parent
    repo = receipt_dir.parents[3]
    require(repo.name == 'mosaic', 'Verifier must remain in Mosaic receipt scope')
    require(not args.output.exists(), 'Verification output is immutable')
    require(not args.output.resolve().is_relative_to(args.report_tree.resolve()) and not args.output.resolve().is_relative_to(args.evidence_tree.resolve()), 'Verification output must not alter verified trees')
    delivery = read(args.delivery_receipt)
    report_expected, evidence_expected = delivery['report_archive'], delivery['delivery_evidence']
    require(report_expected['path'] == ROOT + '/' + PARENT + '.tar.gz'
            and evidence_expected['path'] == ROOT + '/' + EVIDENCE_ARCHIVE, 'Unexpected remote archive identity')
    report_manifest, report_members = archive_tree(args.report_archive, args.report_tree, report_expected)
    evidence_manifest, evidence_members = archive_tree(args.evidence_archive, args.evidence_tree, evidence_expected)
    require(report_manifest.get('job_id') == REPORT_JOB, 'Archive was not produced by exact report allocation')
    evidence = args.evidence_tree
    cpu_record = receipt_dir / 'CPU-verification.json'
    require(digest(cpu_record) == CPU_VERIFICATION, 'Original independent CPU verification changed')
    cpu_expected = read(cpu_record)
    source_manifest = evidence / 'source/source_manifest.json'
    require(digest(source_manifest) == MANIFEST and read(source_manifest)['external_scientific_source_manifest_sha256'] == SCIENCE, 'Analysis/scientific source binding differs')
    package_path = repo / 'results/hpc' / TAG / 'source01/source.tar.gz'
    require(digest(package_path) == '9707bec755c9c116a03a56abceabbe1caeaf019922f4bd68cb2a99141546f3e1', 'Sealed report source archive changed')
    with tarfile.open(package_path) as package:
        source_bytes = package.extractfile('scripts/jq1_lrboost_report_contract.py').read()
    require(hashlib.sha256(source_bytes).hexdigest() == read(source_manifest)['files']['scripts/jq1_lrboost_report_contract.py'], 'Contract archive binding differs')
    pins = next(ast.literal_eval(node.value) for node in ast.parse(source_bytes.decode()).body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'ARMS' for target in node.targets))
    require([pin['job'] for pin in pins.values()] == ['29779177', '29779178', '29779179']
            and [pin['rate'] for pin in pins.values()] == [3.0, 0.05, 0.05], 'Wrong native job/rate set')
    parent = evidence / ('analysis/' + PARENT)
    scheduler(evidence / 'report-final-scheduler.txt', REPORT_JOB)
    lsf(parent / ('stdout.' + REPORT_JOB + '.log'), REPORT_JOB)
    render_submission(parent, evidence, REPORT_JOB, PARENT)
    entry_gate = report_entry_gate(evidence)
    require(read(evidence / ('analysis/' + PARENT + '.tar.receipt.json')) == report_expected, 'Exact sibling archive receipt differs')
    candidates = list(args.report_tree.glob('**/payload/report/report_receipt.json'))
    require(len(candidates) == 1, 'Missing unique report receipt')
    report = candidates[0].parent
    payload = report.parent
    receipt = read(report / 'report_receipt.json')
    assessment = read(payload / 'assessment.json')
    require((payload / 'assessment.json').read_bytes() == (parent / 'payload/assessment.json').read_bytes()
            and (report / 'report_receipt.json').read_bytes() == (parent / 'payload/report/report_receipt.json').read_bytes(), 'Report/compact delivery assessment bytes differ')
    guard = read(evidence / 'analysis/interim-0002-dispatch-guard.json')
    for arm, pin in pins.items():
        launcher(arm, pin, guard['authenticated_inputs'][arm])
    require(assessment['status'] == 'passed' and assessment['job_id'] == REPORT_JOB and assessment['report_source_manifest_sha256'] == MANIFEST
            and assessment['cpu_certificate_sha256'] == CPU_CERTIFICATE and assessment['report_receipt_sha256'] == digest(report / 'report_receipt.json'), 'Allocated report assessment provenance failed')
    require(digest(payload / 'report-source-manifest.json') == MANIFEST, 'Bundled analysis source manifest differs')
    cpu_certificate(payload, evidence, cpu_expected)
    html = (report / 'report.html').read_text(encoding='utf-8')
    require((report / 'report.html').stat().st_size == receipt['html_bytes'] and digest(report / 'report.html') == receipt['html_sha256'], 'Report HTML bytes differ')
    matches = re.findall(r'<script\s+id="manifest"\s+type="application/json">(.*?)</script>', html, re.S)
    require(len(matches) == 1, 'Missing unique embedded JSON manifest')
    data = json.loads(matches[0])
    for value in (receipt, data):
        require(value['version'] == VERSION and value['n_atoms'] == 25 and value['chunk_size'] == 100
                and value['frame_stride_floats'] == 1650 and value['selection_policy'] == 'fixed_step'
                and type(value['complete']) is bool and type(value['presentation_complete']) is bool
                and value['complete'] is value['presentation_complete']
                and value['scientific_passed'] is False and value['campaign_status'] == 'failed'
                and value['strict_scientific_campaign_passed'] is False and value['overall_success'] is False, 'Presentation/science scopes changed')
    require(receipt['complete'] is data['complete'] and receipt['presentation_complete'] is data['presentation_complete'],
            'Embedded and receipt presentation completion differ')
    require(assessment['scientific_passed'] is False and assessment['campaign_status'] == 'failed'
            and assessment['complete'] is receipt['complete'] and assessment['presentation_complete'] is receipt['presentation_complete'], 'Assessment obscures scientific failure')
    require(receipt['input_jobs'] == assessment['input_jobs'] == ['29779177', '29779178', '29779179']
            and receipt['states'] == [10001, 10001, 10001] and receipt['included_updates'] == [10000, 10000, 10000], 'Edition0002 must include each complete saved update budget')
    require(receipt['states'] == assessment['states'] and receipt['included_updates'] == assessment['included_updates'], 'Assessment captured prefix differs')
    studies = data['studies']
    require(len(studies) == len(receipt['arm_bindings']) == 3, 'Missing arm records')
    collector_inventories, collector_original_proofs = {}, {}
    remote_references = []
    for arm, collector_job in [('protenix', '29780825'), ('boltz2', '29780826')]:
        pin = pins[arm]
        collector = evidence / ('analysis/complete-' + arm)
        scheduler(evidence / ('collector-' + arm + '-scheduler.txt'), collector_job)
        lsf(collector / ('stdout.' + collector_job + '.log'), collector_job)
        require(digest(collector / 'submission.json') == cpu_expected['files']['analysis/complete-' + arm + '/submission.json']['sha256'], 'Original collector submission changed')
        captured = read(collector / 'payload/assessment.json')
        require(captured['status'] == captured['presentation_status'] == 'passed' and captured['presentation_complete'] is True
                and captured['collector_job'] == collector_job and captured['native_job'] == pin['job'] and captured['arm'] == arm
                and captured['scientific_status'] == 'failed' and captured['scientific_passed'] is False
                and captured['states'] == 10001 and captured['updates'] == 10000
                and captured['all_probability_frames_verified'] is True and captured['report_source_manifest_sha256'] == MANIFEST
                and captured['cpu_certificate_sha256'] == CPU_CERTIFICATE and captured['learning_rate'] == pin['rate'], 'Collector full-frame/failure scope changed')
        launcher(arm, pin, captured['launcher'])
        original_proofs = proof_schema(captured['proofs'])
        collector_original_proofs[arm] = original_proofs
        study_root = pin['output_parent'] + '/payload/study'
        for suffix, field in [('summary.json', 'summary'), ('case-0/report.json', 'case'), ('terminal_gradient_checks/report.json', 'terminal')]:
            require(original_proofs.get(study_root + '/' + suffix) == pin['failed_terminal'][field], 'Known failed original proof changed')
        require(digest(collector / 'payload/native-stdout.txt') == pin['failed_terminal']['stdout'], 'Original native LSF bytes changed')
        lsf(collector / 'payload/native-stdout.txt', pin['job'], failed=True)
        inventory_path = collector / 'payload/snapshots.json'
        inventory = read(inventory_path)
        require(inventory['status'] == 'passed' and inventory['native_job'] == pin['job'] and inventory['n_atoms'] == 25
                and len(inventory['frames']) == 10001 and [frame['step'] for frame in inventory['frames']] == list(range(10001)), 'Collector lacks complete saved-frame inventory')
        require(all(type(frame.get('bytes')) is int and frame['bytes'] > 0 and HEX.fullmatch(frame.get('sha256', ''))
                    and frame['path'].startswith(study_root + '/case-0/') for frame in inventory['frames']), 'Malformed original frame hash references')
        inventory_remote = ROOT + '/complete-' + arm + '/payload/snapshots.json'
        require(original_proofs.get(inventory_remote) == digest(inventory_path), 'Collector proof does not bind saved-frame inventory')
        collector_inventories[arm] = inventory['frames']
        remote_references.extend(captured['proofs'])
        copied = payload / ('completion-' + arm)
        if copied.exists():
            for name in ('assessment.json', 'snapshots.json', 'native-stdout.txt'):
                require((copied / name).read_bytes() == (collector / 'payload' / name).read_bytes(), 'Bundled collector copy differs')
    chemistry = chemistry_collector(evidence, payload, pins['boltz2_chemistry'], cpu_expected)
    if chemistry is not None:
        collector_inventories['boltz2_chemistry'] = chemistry['frames']
        collector_original_proofs['boltz2_chemistry'] = chemistry['proofs']
        remote_references.extend(chemistry['assessment']['proofs'])
    assets = {'report.html', 'report_receipt.json', 'loss-overlays.svg'}
    verified_arms = []
    for index, ((arm, pin), study, binding) in enumerate(zip(pins.items(), studies, receipt['arm_bindings'])):
        fixed = {'arm': arm, 'job_id': pin['job'], 'learning_rate': pin['rate'], 'selection_policy': 'fixed_step',
                 'source_manifest_sha256': SCIENCE, 'config_sha256': pin['config'], 'readiness_sha256': pin['ready']}
        require(all(study.get(key) == binding.get(key) == value for key, value in fixed.items()), 'Scientific report binding changed')
        require(study['model_digest'] == pin['model'] and study['context_digest'] == pin['context'], 'Original model/context digest differs')
        require(study['states'] == receipt['states'][index] and study['latest_update'] == receipt['included_updates'][index], 'Embedded prefix differs')
        require(type(study['states']) is int and type(study['latest_update']) is int and 1 <= study['states'] <= 10001
                and study['states'] == study['latest_update'] + 1, 'Sparse or absent captured state count')
        outcome(study, known_failed=index < 2)
        if arm == 'boltz2_chemistry':
            if study['all_terminal']:
                require(chemistry is not None, 'Captured chemistry terminal claim needs actual collector DONE/LSF/full inventory evidence')
                collected_outcome = chemistry['assessment']['outcome']
                for field in ('readiness_status', 'terminal_status', 'terminal_error', 'overall_status', 'budget_status', 'scientific_passed', 'all_terminal'):
                    require(study[field] == collected_outcome[field], 'Chemistry report terminal outcome differs from authenticated collector')
            else:
                require(study['native_status'] == 'RUN' and study['overall_status'] == 'running'
                        and study['terminal_status'] == 'pending' and study['terminal_error'] is None
                        and study['scientific_passed'] is False and study['budget_status'] == 'full',
                        'Full chemistry saved budget incorrectly implies terminal scientific validation')
        for field in ('native_status', 'readiness_status', 'terminal_status', 'terminal_error', 'overall_status', 'budget_status', 'scientific_passed', 'all_terminal'):
            require(study[field] == binding[field], 'Outcome binding differs')
        if index < 2:
            require(study['terminal_error'] == pin['failed_terminal']['error'], 'Known terminal error changed')
        require(study['counts']['applied'] == study['latest_update'] and study['counts']['forced_fallback'] == 0, 'Fixed-count metadata differs')
        provenance = study['progress_evidence']
        launcher(arm, pin, provenance['launcher'])
        require(provenance['raw_entry_sha256'] == pin['raw_entry'] and provenance['coupled_entry_sha256'] == pin['coupled_entry'], 'Entry gate binding changed')
        prefixes = provenance['jsonl_prefixes']
        require(len(prefixes) == 2, 'Missing exact state/update committed-prefix references')
        for prefix, field, rows in zip(prefixes, ('states', 'updates'), (study['states'], study['latest_update'])):
            path = pin['output_parent'] + '/payload/study/case-0/' + field + '.jsonl'
            require(prefix['path'] == path and prefix['included_rows'] == rows
                    and type(prefix['bytes']) is int and prefix['bytes'] > 0
                    and HEX.fullmatch(prefix['sha256'])
                    and prefix['available_committed_bytes'] >= prefix['bytes']
                    and prefix['available_committed_rows'] >= rows, 'Malformed captured original JSONL prefix reference')
            if index < 2 or (arm == 'boltz2_chemistry' and study['all_terminal']):
                require(prefix['uncommitted_tail_bytes'] == prefix['excluded_committed_rows'] == 0
                        and collector_original_proofs[arm][path] == prefix['sha256'], 'Full report prefix differs from original collector JSONL binding')
        cursor = 0
        for chunk in study['chunks']:
            require(type(chunk['count']) is int and 1 <= chunk['count'] <= 100 and type(chunk['start']) is int and chunk['start'] == cursor, 'Chunk bounds not contiguous/100-frame bounded')
            start = cursor
            cursor += chunk['count']
            require(cursor <= study['states'], 'Chunk exceeds captured prefix')
            for kind in ('binary', 'metadata'):
                require(len(safe_name(chunk[kind]).parts) == 1, 'Chunk filename is not a basename')
                name = arm + '/' + chunk[kind]
                require(name not in assets, 'Duplicate report asset')
                assets.add(name)
                path = local(report, name)
                require(digest(path) == chunk[kind + '_sha256'], 'Chunk hash differs')
                if kind == 'binary':
                    require(path.stat().st_size == chunk['count'] * 1650 * 4, 'Chunk byte length differs')
                else:
                    rows = read(path)
                    require(isinstance(rows, list) and len(rows) == chunk['count'] and [row['step'] for row in rows] == list(range(start, cursor)), 'Metadata row count/step continuity differs')
                    for row in rows:
                        require(HEX.fullmatch(row['snapshot_sha256']) and not PurePosixPath(row['snapshot']).is_absolute()
                                and '..' not in PurePosixPath(row['snapshot']).parts, 'Malformed original snapshot reference')
                        if arm in collector_inventories:
                            frame = collector_inventories[arm][row['step']]
                            require(row['snapshot_sha256'] == frame['sha256']
                                    and frame['path'] == pin['output_parent'] + '/payload/study/case-0/' + row['snapshot'], 'Report snapshot reference differs from full allocated inventory')
                        previous = row['previous_update']
                        if row['step'] == 0:
                            require(previous is None, 'Initial state invents an applied update')
                        else:
                            require(previous['selection_policy'] == 'fixed_step' and previous['attempt_count'] == 1
                                    and previous['selected_trial'] == 0 and previous['accepted_step_size'] == pin['rate']
                                    and previous['fallback_used'] is False, 'Fixed update display metadata changed')
        require(cursor == study['states'] and len(study['chunks']) == (study['states'] + 99) // 100, 'Saved frames omitted/overchunked')
        curve_name = arm + '/curves.json'
        assets.add(curve_name)
        curves = read(local(report, curve_name))
        require(isinstance(curves, list) and [row['step'] for row in curves] == list(range(study['states'])), 'Scalar metadata prefix is sparse')
        verified_arms.append({**fixed, 'states': study['states'], 'updates': study['latest_update'], 'chunks': len(study['chunks']),
                              **{field: study[field] for field in ('native_status', 'terminal_status', 'overall_status', 'scientific_passed')}})
    require(data['diagnostics'] == receipt['diagnostics'] and len(data['diagnostics']) == 3, 'Diagnostic status display differs')
    for entry, (arm, pin) in zip(data['diagnostics'], pins.items()):
        require(entry['arm'] == arm and entry['original_job'] == pin['job'] and entry['status'] in ('pending', 'available'), 'Diagnostic original-run identity differs')
        if entry['status'] == 'pending':
            require(not entry.get('assets'), 'Pending diagnostic invents assets')
        else:
            require(set(entry['assets']) == {'report', 'capture'}, 'Missing diagnostic asset pair')
            for kind, record in entry['assets'].items():
                name = 'diagnostics/' + arm + '-' + kind + '.json'
                require(record['path'] == name and name not in assets, 'Diagnostic asset scope changed')
                assets.add(name)
                path = local(report, name)
                require(path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], 'Diagnostic served bytes differ')
            raw = read(local(report, entry['assets']['report']['path']))
            require(raw['original_job'] == pin['job'] and raw['scientific_manifest_sha256'] == SCIENCE
                    and raw['can_certify_or_restart_original_study'] is False, 'Diagnostic scope waives original failure')
    require({path.relative_to(report).as_posix() for path in report.rglob('*') if path.is_file()} == assets, 'Unexpected/missing served assets')
    require(receipt['all_terminal'] is data['all_terminal'] is all(study['all_terminal'] for study in studies), 'Terminal campaign scope differs')
    require(not receipt['presentation_complete'] or receipt['all_terminal'], 'Completed presentation requires all original terminal outcomes')
    require(assessment['complete'] is receipt['complete'] and assessment['presentation_complete'] is receipt['presentation_complete'],
            'Allocated assessment presentation completion scope differs')
    record = {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Independent stdlib byte/hash/JSON/XML/schema delivery verification only; no local FP32/simplex/scalar/argmax recomputation, model evaluation, repair, rendering, SSH or submissions',
        'report_job': REPORT_JOB, 'report_scheduler': 'actual DONE/hpc', 'report_lsf_success': True,
        'report_source_manifest_sha256': MANIFEST, 'original_scientific_source_manifest_sha256': SCIENCE,
        'cpu_certificate_sha256': CPU_CERTIFICATE, 'cpu_tests': 278, 'cpu_verification_sha256': CPU_VERIFICATION,
        'report_archive': report_expected, 'delivery_evidence_archive': evidence_expected,
        'report_safe_members': len(report_members), 'delivery_safe_members': len(evidence_members),
        'report_assessment_sha256': digest(payload / 'assessment.json'), 'report_receipt_sha256': digest(report / 'report_receipt.json'),
        'html_sha256': receipt['html_sha256'], 'served_assets': len(assets), 'scientific_passed': False, 'campaign_status': 'failed',
        'presentation_complete': receipt['presentation_complete'], 'all_terminal': receipt['all_terminal'], 'arms': verified_arms,
        'report_parent': PARENT, 'actual_report_submission_sha256': digest(parent / 'submission.json'),
        'allocated_entry_storage_gate': entry_gate,
        'actual_dispatch_guard_sha256': DISPATCH_GUARD, 'original_actual_submission_stdout_sha256': SUBMISSION_STDOUT,
        'original_actual_submission_result_sha256': SUBMISSION_RESULT,
        'chemistry_terminal_capture_verified': chemistry is not None,
        'chemistry_terminal_status_at_render': studies[2]['terminal_status'],
        'chemistry_scientific_passed_at_render': studies[2]['scientific_passed'],
        'collector_allocations': {'protenix': '29780825 actual DONE/hpc/LSFsuccess,10001 full saved-frame references',
                                  'boltz2': '29780826 actual DONE/hpc/LSFsuccess,10001 full saved-frame references'},
        'chemistry_collector_allocation': ('29780827 actual DONE/hpc/LSFsuccess,10001 full saved-frame references' if chemistry is not None else 'No completed chemistry collector bundled; saved10000 budget is not a terminal science claim'),
        'original_big_evidence_scope': 'Original summaries/cases/JSONL/NPZ bytes remain HPC references. Allocated collectors/renderers authenticated and numerically validated frames there; this delivery verifier checks bound local metadata/hash references only.',
        'remote_original_proofs': remote_references,
        'report_files': report_manifest['files'], 'delivery_files': evidence_manifest['files'],
        'verification_script_sha256': digest(Path(__file__))}
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'output': str(args.output), 'sha256': digest(args.output), 'states': receipt['states'], 'assets': len(assets)}))


if __name__ == '__main__':
    main()
