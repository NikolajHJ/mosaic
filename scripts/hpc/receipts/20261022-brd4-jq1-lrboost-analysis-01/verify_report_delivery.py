"""Independently verify downloaded report delivery using only stdlib bytes/schema.

This never decodes FP32 arrays, reruns simplex/argmax checks or imports scientific
or report implementation modules. Those checks remain allocated-HPC evidence.
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
REPORT_JOB = '29780829'
PARENT = 'interim-0001'
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
    args = parser.parse_args()
    receipt_dir = Path(__file__).resolve().parent
    repo = receipt_dir.parents[3]
    require(repo.name == 'mosaic', 'Verifier must remain in Mosaic receipt scope')
    require(not args.output.exists(), 'Verification output is immutable')
    require(not args.output.resolve().is_relative_to(args.report_tree.resolve()) and not args.output.resolve().is_relative_to(args.evidence_tree.resolve()), 'Verification output must not alter verified trees')
    delivery = read(args.delivery_receipt)
    report_expected, evidence_expected = delivery['report_archive'], delivery['delivery_evidence']
    require(report_expected['path'] == ROOT + '/' + PARENT + '.tar.gz'
            and evidence_expected['path'] == ROOT + '/interim-delivery-evidence-01.tar.gz', 'Unexpected remote archive identity')
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
    require(digest(parent / 'submission.json') == cpu_expected['files']['analysis/' + PARENT + '/submission.json']['sha256'], 'Actual render submission changed')
    candidates = list(args.report_tree.glob('**/payload/report/report_receipt.json'))
    require(len(candidates) == 1, 'Missing unique report receipt')
    report = candidates[0].parent
    payload = report.parent
    receipt = read(report / 'report_receipt.json')
    assessment = read(payload / 'assessment.json')
    require((payload / 'assessment.json').read_bytes() == (parent / 'payload/assessment.json').read_bytes()
            and (report / 'report_receipt.json').read_bytes() == (parent / 'payload/report/report_receipt.json').read_bytes(), 'Report/compact delivery assessment bytes differ')
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
                and value['complete'] is False and value['presentation_complete'] is False
                and value['scientific_passed'] is False and value['campaign_status'] == 'failed'
                and value['strict_scientific_campaign_passed'] is False and value['overall_success'] is False, 'Presentation/science scopes changed')
    require(assessment['scientific_passed'] is False and assessment['campaign_status'] == 'failed'
            and assessment['complete'] is receipt['complete'] and assessment['presentation_complete'] is receipt['presentation_complete'], 'Assessment obscures scientific failure')
    require(receipt['input_jobs'] == assessment['input_jobs'] == ['29779177', '29779178', '29779179']
            and receipt['states'][:2] == [10001, 10001] and receipt['included_updates'][:2] == [10000, 10000], 'Wrong saved native scope')
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
            if index < 2:
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
    require(receipt['all_terminal'] is all(study['all_terminal'] for study in studies), 'Terminal campaign scope differs')
    record = {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Independent stdlib byte/hash/JSON/XML/schema delivery verification only; no local FP32/simplex/scalar/argmax recomputation, model evaluation, repair, rendering, SSH or submissions',
        'report_job': REPORT_JOB, 'report_scheduler': 'actual DONE/hpc', 'report_lsf_success': True,
        'report_source_manifest_sha256': MANIFEST, 'original_scientific_source_manifest_sha256': SCIENCE,
        'cpu_certificate_sha256': CPU_CERTIFICATE, 'cpu_tests': 278, 'cpu_verification_sha256': CPU_VERIFICATION,
        'report_archive': report_expected, 'delivery_evidence_archive': evidence_expected,
        'report_safe_members': len(report_members), 'delivery_safe_members': len(evidence_members),
        'report_assessment_sha256': digest(payload / 'assessment.json'), 'report_receipt_sha256': digest(report / 'report_receipt.json'),
        'html_sha256': receipt['html_sha256'], 'served_assets': len(assets), 'scientific_passed': False, 'campaign_status': 'failed',
        'presentation_complete': False, 'arms': verified_arms,
        'collector_allocations': {'protenix': '29780825 actual DONE/hpc/LSFsuccess,10001 full saved-frame references',
                                  'boltz2': '29780826 actual DONE/hpc/LSFsuccess,10001 full saved-frame references'},
        'original_big_evidence_scope': 'Original summaries/cases/JSONL/NPZ bytes remain HPC references. Allocated collectors/renderers authenticated and numerically validated frames there; this delivery verifier checks bound local metadata/hash references only.',
        'remote_original_proofs': remote_references,
        'report_files': report_manifest['files'], 'delivery_files': evidence_manifest['files'],
        'verification_script_sha256': digest(Path(__file__))}
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'passed', 'output': str(args.output), 'sha256': digest(args.output), 'states': receipt['states'], 'assets': len(assets)}))


if __name__ == '__main__':
    main()
