"""Allocated presentation-only focus patch and strict additive publication.

This module never evaluates a model or decodes probability arrays. It adopts an
already authenticated saved-only render, changes one event listener, and retains
the original renderer source/CPU provenance alongside its own patch provenance.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

VERSION = 'jq1-lrboost-viewer-focus-adoption-v1'
SUITE = 'jq1-lrboost-viewer-focus-contract-v1'
ANALYSIS_MANIFEST = '214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc'
ANALYSIS_CPU = '42b2b000ecce5d661da9b75ad85f03a41c3b303010ccfa03f87835a25993ca69'
ANALYSIS_SOURCE = Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-01/source')
FOCUS_ROOT = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/viewer-focus-01')
FOCUS_SOURCE = FOCUS_ROOT.parent / 'viewer-focus-release-01/source'
OLD_COUNTS = {'test_jq1_lrboost_report_contract': 82, 'test_jq1_lrboost_render': 54,
              'test_jq1_lrboost_publish': 59, 'test_jq1_lrboost_ops': 32,
              'test_lrboost_endpoint_diagnostics': 16, 'test_boltz_affinity_report': 26,
              'test_small_molecule_chemistry_report': 3,
              'test_small_molecule_chemistry_ramp_report': 6}
CPU_MODULES = set(OLD_COUNTS) | {'test_viewer_focus', 'test_viewer_focus_ops'}
ANCHOR = b"byId('stepInput').addEventListener('change',()=>chooseStep(byId('stepInput').valueAsNumber));"
FOCUS = b"byId('stepInput').addEventListener('focus',stop);\n"
HEX = re.compile(r'[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON ' + value)))


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def name_path(root, name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name and '\x00' not in name,
            'Unsafe relative proof path')
    pure = PurePosixPath(name)
    require(not pure.is_absolute() and str(pure) == name
            and all(p not in ('', '.', '..') and p.rstrip(' .') == p for p in pure.parts),
            'Unsafe relative proof path')
    require(all(p.split('.')[0].upper() not in {'CON', 'PRN', 'AUX', 'NUL',
                *('COM' + str(i) for i in range(1, 10)), *('LPT' + str(i) for i in range(1, 10))}
                for p in pure.parts), 'Unsafe platform proof path')
    target = Path(root).joinpath(*pure.parts)
    require(target.resolve().is_relative_to(Path(root).resolve()), 'Proof path escaped root')
    for part in (target, *target.parents):
        if part == Path(root).parent:
            break
        require(not part.is_symlink(), 'Proof symlinks forbidden')
    return target


def file_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing regular proof file')
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def tree_records(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Missing regular tree')
    result = {}
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Tree symlinks forbidden')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_record(path)
        else:
            require(path.is_dir(), 'Special tree entry forbidden')
    return result


def actual_success(scheduler, stdout, job):
    require(isinstance(job, str) and job.isdigit(), 'Missing actual job identity')
    compact = re.sub(r'\s+', '', scheduler)
    require(re.findall(r'Job<([^>]+)>', compact) == [job]
            and re.findall(r'Status<([^>]+)>', compact) == ['DONE']
            and re.findall(r'Queue<([^>]+)>', compact) == ['hpc'], 'Actual same-job DONE/hpc required')
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', stdout, re.M)
    require(len(subjects) == 1 and subjects[0][0] == job and subjects[0][1].rstrip().endswith(' Done')
            and re.search(r'^Successfully completed\.\s*$', stdout, re.M)
            and not re.search(r'Exited\s+(?:with exit|by signal)', stdout, re.I),
            'Actual same-job successful LSF required')


def verify_job_proofs(folder, job, *, original_pin=None):
    folder = Path(folder)
    records = tree_records(folder)
    require({'scheduler.txt', 'stdout.log', 'submission.json'} <= set(records), 'Missing actual job proof set')
    actual_success((folder / 'scheduler.txt').read_text(encoding='utf-8'),
                   (folder / 'stdout.log').read_text(encoding='utf-8'), job)
    sub = read(folder / 'submission.json')
    require(sub.get('job_id') == job and sub.get('outcome') == 'submitted'
            and isinstance(sub.get('args'), list) and sub['args'][:1] == ['bsub']
            and '-q' in sub['args'] and sub['args'][sub['args'].index('-q') + 1] == 'hpc',
            'Actual submission identity missing')
    if original_pin is not None:
        require('archive-receipt.json' in records, 'Original archive sibling receipt missing')
        receipt = read(folder / 'archive-receipt.json')
        require(receipt.get('bytes') == original_pin['archive_bytes']
                and receipt.get('sha256') == original_pin['archive_sha256']
                and isinstance(receipt.get('path'), str) and receipt['path'].endswith('.tar.gz'),
                'Original archive receipt differs')
    return records


def verify_patch_submission(folder, job, manifest):
    """Bind the additive presentation producer to its exact declared vector."""
    require(HEX.fullmatch(manifest or ''), 'Malformed patch manifest identity')
    folder = Path(folder)
    records = verify_job_proofs(folder, job)
    require('submission-started.json' in records, 'Exclusive patch submission-started proof missing')
    parent = FOCUS_ROOT / 'patch'
    expected = ['bsub', '-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R',
                'rusage[mem=4GB]', '-W', '00:30', '-J', 'mosaic-viewer-focus-patch',
                '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log'),
                'bash', '-l', str(FOCUS_SOURCE / 'stage.lsf'), str(FOCUS_SOURCE),
                str(FOCUS_ROOT), manifest, 'patch', str(parent)]
    sub, started = read(folder / 'submission.json'), read(folder / 'submission-started.json')
    require(sub.get('phase') == 'patch' and sub.get('args') == expected
            and sub.get('dependency') is None and sub.get('returncode') == 0
            and re.findall(r'Job <([0-9]+)>', sub.get('stdout', '')) == [job]
            and 'error' not in sub, 'Patch producer phase/source/manifest/root/resources differ')
    require(set(started) == {'phase', 'args', 'dependency', 'outcome', 'utc'}
            and started == {'phase': 'patch', 'args': expected, 'dependency': None,
                            'outcome': 'unknown', 'utc': sub.get('utc')},
            'Exclusive patch submission-started identity differs')
    return records


def authenticate_source(root, expected):
    root = Path(root)
    manifest = root / 'source_manifest.json'
    require(HEX.fullmatch(expected or '') and sha(manifest) == expected, 'Source manifest checksum differs')
    records = read(manifest).get('files')
    require(isinstance(records, dict) and records, 'Source manifest has no files')
    actual = tree_records(root)
    require(set(actual) == set(records) | {'source_manifest.json'}, 'Source tree set differs')
    for name, digest in records.items():
        require(HEX.fullmatch(digest or '') and file_record(name_path(root, name))['sha256'] == digest,
                'Source bytes changed: ' + name)
    return records


def load_original_publisher(analysis_source=ANALYSIS_SOURCE):
    """Import only exact authenticated stdlib original publisher/contract bytes."""
    root = Path(analysis_source)
    authenticate_source(root, ANALYSIS_MANIFEST)
    scripts = root / 'scripts'
    def load(label, filename):
        spec = importlib.util.spec_from_file_location(label, scripts / filename)
        require(spec is not None and spec.loader is not None, 'Cannot load authenticated original helper')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    previous = sys.modules.get('jq1_lrboost_report_contract')
    try:
        contract = load('_focus_original_contract', 'jq1_lrboost_report_contract.py')
        sys.modules['jq1_lrboost_report_contract'] = contract
        return load('_focus_original_publisher', 'publish_jq1_lrboost_report.py')
    finally:
        sys.dont_write_bytecode = previous_bytecode
        if previous is None:
            sys.modules.pop('jq1_lrboost_report_contract', None)
        else:
            sys.modules['jq1_lrboost_report_contract'] = previous


def patch_html(content):
    require(isinstance(content, bytes), 'Patch operates on exact HTML bytes')
    require(content.count(ANCHOR) == 1 and FOCUS not in content,
            'Expected unique unpatched exact-update listener')
    require(content.count(b'function stop(){playing=false;playEpoch++;') == 1,
            'Playback epoch stop contract differs')
    return content.replace(ANCHOR, FOCUS + ANCHOR, 1)


def report_path(tree):
    found = list(Path(tree).glob('**/payload/report/report_receipt.json'))
    require(len(found) == 1, 'Require exactly one saved report')
    return found[0].parent


def verify_original(tree, publisher, original_pin, original_job_proof_dir):
    require(set(original_pin) == {'job_id', 'archive_bytes', 'archive_sha256'}
            and isinstance(original_pin['job_id'], str) and original_pin['job_id'].isdigit()
            and type(original_pin['archive_bytes']) is int and original_pin['archive_bytes'] > 0
            and HEX.fullmatch(original_pin['archive_sha256'] or ''), 'Malformed declared original render pin')
    manifest, downloaded = publisher.verify_tree(Path(tree))
    archive = Path(downloaded['archive'])
    require(downloaded['sha256'] == original_pin['archive_sha256']
            and archive.stat().st_size == original_pin['archive_bytes'], 'Original archive pin differs')
    report = report_path(tree)
    receipt, data = publisher.validate_report(report, Path(tree))
    assessment = read(report.parent / 'assessment.json')
    require(assessment['job_id'] == original_pin['job_id']
            and assessment['report_source_manifest_sha256'] == ANALYSIS_MANIFEST
            and assessment['cpu_certificate_sha256'] == ANALYSIS_CPU,
            'Original render job/source/CPU pins differ')
    require(receipt['states'] == [10001] * 3 and receipt['included_updates'] == [10000] * 3
            and receipt['scientific_passed'] is False and receipt['campaign_status'] == 'failed',
            'Require all original saved budgets with scientific failures preserved')
    require([s['scientific_passed'] for s in data['studies']] == [False, False, True]
            and all(s['all_terminal'] for s in data['studies']), 'Original terminal outcomes differ')
    require(not receipt.get('viewer_focus_adoption'), 'Original report is already focus-adopted')
    proof_records = verify_job_proofs(original_job_proof_dir, original_pin['job_id'], original_pin=original_pin)
    return {'manifest': manifest, 'downloaded': downloaded, 'report': report,
            'receipt': receipt, 'data': data, 'assessment': assessment, 'job_proofs': proof_records}


def verify_patch_cpu(folder, manifest, certificate_sha256):
    folder = Path(folder)
    records = tree_records(folder)
    require(sha(folder / 'assessment.json') == certificate_sha256, 'Patch CPU certificate checksum differs')
    cert = read(folder / 'assessment.json')
    require(cert.get('status') == 'passed' and cert.get('suite') == SUITE
            and cert.get('source_manifest_sha256') == manifest, 'Patch CPU source/suite/pass differs')
    proofs, bundles = cert.get('proofs'), read(folder / 'bundled-proofs.json')
    require(isinstance(proofs, list) and proofs and isinstance(bundles, list)
            and len(proofs) == len(bundles), 'Patch CPU proof bundle incomplete')
    originals = {p.get('path'): p.get('sha256') for p in proofs}
    require(len(originals) == len(proofs), 'Duplicate patch CPU proof')
    bound, seen = [], set()
    for row in bundles:
        path = name_path(folder, row.get('local_path'))
        require(row.get('original_path') in originals and row['original_path'] not in seen
                and originals[row['original_path']] == row.get('sha256') == sha(path),
                'Bundled patch CPU proof differs')
        if 'bytes' in row:
            require(path.stat().st_size == row['bytes'], 'Bundled patch CPU length differs')
        seen.add(row['original_path']); bound.append(path)
    schedulers = [p for p in bound if re.fullmatch(r'scheduler-final-\d+\.txt', p.name)]
    job = str(cert.get('producer_job_id', cert.get('job_id', '')))
    logs = [p for p in bound if p.name == 'stdout.' + job + '.log']
    require(len(schedulers) == len(logs) == 1, 'Patch CPU scheduler/LSF proofs missing')
    actual_success(schedulers[0].read_text(encoding='utf-8'), logs[0].read_text(encoding='utf-8'), job)
    xmls = [p for p in bound if p.suffix == '.xml']
    require(len(xmls) == 1, 'Expected one complete patch CPU XML')
    root = ET.parse(xmls[0]).getroot(); cases = list(root.iter('testcase'))
    require(cases and not any(list(root.iter(t)) for t in ('failure', 'error', 'skipped')),
            'Patch CPU fixture failures/errors/skips')
    counts = Counter()
    for case in cases:
        modules = CPU_MODULES.intersection(case.get('classname', '').split('.'))
        require(len(modules) == 1, 'Patch CPU case module attribution differs')
        counts[next(iter(modules))] += 1
    require(set(counts) == CPU_MODULES and all(counts[k] == v for k, v in OLD_COUNTS.items())
            and dict(counts) == cert.get('tests_by_module') and cert.get('tests') == len(cases),
            'Patch CPU must rerun all old278 fixtures and both new modules')
    for suite in list(root.iter('testsuite')) + list(root.iter('testsuites')):
        require(all(int(suite.get(k, '0')) == 0 for k in ('failures', 'errors', 'skipped'))
                and ('tests' not in suite.attrib or int(suite.get('tests')) == len(list(suite.iter('testcase')))),
                'Patch CPU XML header/count mismatch')
    test_logs = [p for p in bound if p.name == 'report-tests.log']
    require(len(test_logs) == 1, 'Missing complete patch CPU raw log')
    raw_log = test_logs[0].read_text(encoding='utf-8')
    require(re.findall(r'collected\s+(\d+)\s+items?', raw_log) == [str(len(cases))]
            and re.search(r'\b' + str(len(cases)) + r'\s+passed\b', raw_log)
            and not re.search(r'\d+\s+(?:deselected|skipped|failed|errors?)\b', raw_log),
            'Patch CPU raw collection/pass log differs')
    return {'certificate': cert, 'records': records, 'counts': dict(counts)}


def _replace_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.focus-tmp')
    put(temporary, value)
    temporary.replace(path)


def _source_provenance(manifest_path, checksum):
    manifest_path = Path(manifest_path)
    require(sha(manifest_path) == checksum, 'Patch source manifest checksum differs')
    files = read(manifest_path).get('files', {})
    own = [(name, digest) for name, digest in files.items() if Path(name).name == Path(__file__).name]
    require(len(own) == 1 and own[0][1] == sha(Path(__file__)), 'Executing patch/publisher bytes are not the declared source')
    return file_record(manifest_path)


def verify_original_parent(parent, archive, publisher, original_pin, original_job_proof_dir):
    """Stream every archived byte and compare stable payload to its owned parent.

    The original collector queried its own scheduler while packaging, so that
    archived snapshot may be RUN. The separate final scheduler/LSF proofs below
    establish DONE; no archived scheduler field is rewritten.
    """
    parent, archive = Path(parent), Path(archive)
    require(set(original_pin) == {'job_id', 'archive_bytes', 'archive_sha256'}
            and isinstance(original_pin['job_id'], str) and original_pin['job_id'].isdigit()
            and type(original_pin['archive_bytes']) is int and original_pin['archive_bytes'] > 0
            and HEX.fullmatch(original_pin['archive_sha256'] or ''), 'Malformed declared original render pin')
    require(parent.is_dir() and not parent.is_symlink() and archive.is_file() and not archive.is_symlink(),
            'Original parent/archive must be regular unaliased inputs')
    require(archive.stat().st_size == original_pin['archive_bytes']
            and sha(archive) == original_pin['archive_sha256'], 'Original archive bytes differ')
    members = {}
    with tarfile.open(archive, 'r:*') as handle:
        for member in handle.getmembers():
            publisher.safe_name(member.name)
            require(member.isfile() and member.name not in members, 'Unsafe/nonregular/duplicate original member')
            with handle.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            members[member.name] = {'bytes': member.size, 'sha256': digest}
        require('evidence_manifest.json' in members, 'Original archive manifest missing')
        manifest_bytes = handle.extractfile('evidence_manifest.json').read()
    manifest = json.loads(manifest_bytes)['files']
    require(set(members) == set(manifest) | {'evidence_manifest.json'}
            and all(members[name] == record for name, record in manifest.items()),
            'Original archive member manifest differs')
    candidates = [name for name in manifest if name.endswith('/payload/report/report_receipt.json')]
    require(len(candidates) == 1, 'Original archive has ambiguous report')
    prefix = candidates[0].removesuffix('/payload/report/report_receipt.json')
    for name, record in manifest.items():
        if name.startswith(prefix + '/payload/'):
            current = name_path(parent, name[len(prefix) + 1:])
            require(file_record(current) == record, 'Original saved payload changed: ' + name)
    report = parent / 'payload/report'
    receipt, data = publisher.validate_report(report, parent)
    assessment = read(parent / 'payload/assessment.json')
    require(assessment['job_id'] == original_pin['job_id']
            and assessment['report_source_manifest_sha256'] == ANALYSIS_MANIFEST
            and assessment['cpu_certificate_sha256'] == ANALYSIS_CPU,
            'Original render job/source/CPU pins differ')
    require(receipt['states'] == [10001] * 3 and receipt['included_updates'] == [10000] * 3
            and receipt['scientific_passed'] is False and receipt['campaign_status'] == 'failed'
            and [s['scientific_passed'] for s in data['studies']] == [False, False, True]
            and all(s['all_terminal'] for s in data['studies']), 'Original saved budgets/outcomes differ')
    require(not receipt.get('viewer_focus_adoption'), 'Original report already focus-adopted')
    proof_records = verify_job_proofs(original_job_proof_dir, original_pin['job_id'], original_pin=original_pin)
    return {'manifest': manifest, 'report': report, 'receipt': receipt, 'data': data,
            'assessment': assessment, 'job_proofs': proof_records,
            'archive_evidence_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest()}


def create_patch(original_parent, output_tree, *, publisher, original_pin, original_archive, original_job_proof_dir,
                 patch_source_manifest_sha256, patch_source_manifest_path, patch_job_id,
                 patch_cpu_certificate_sha256, patch_cpu_proof_dir):
    """Allocated-only byte adoption; caller authenticates own submission/storage."""
    require(os.environ.get('LSB_JOBID') == patch_job_id and patch_job_id.isdigit(),
            'Focus patch requires its actual allocated patch job')
    original_parent, output_tree = Path(original_parent), Path(output_tree)
    require(not output_tree.exists() and not output_tree.resolve().is_relative_to(original_parent.resolve())
            and not original_parent.resolve().is_relative_to(output_tree.resolve()), 'Fresh independent patch tree required')
    baseline = verify_original_parent(original_parent, original_archive, publisher, original_pin, original_job_proof_dir)
    verify_patch_cpu(patch_cpu_proof_dir, patch_source_manifest_sha256, patch_cpu_certificate_sha256)
    _source_provenance(patch_source_manifest_path, patch_source_manifest_sha256)
    return _write_delta(baseline, original_parent, output_tree, original_pin=original_pin,
        original_job_proof_dir=original_job_proof_dir,
        patch_source_manifest_sha256=patch_source_manifest_sha256,
        patch_source_manifest_path=patch_source_manifest_path, patch_job_id=patch_job_id,
        patch_cpu_certificate_sha256=patch_cpu_certificate_sha256,
        patch_cpu_proof_dir=patch_cpu_proof_dir)


def _write_delta(baseline, original_parent, output_tree, *, original_pin, original_job_proof_dir,
                 patch_source_manifest_sha256, patch_source_manifest_path, patch_job_id,
                 patch_cpu_certificate_sha256, patch_cpu_proof_dir):
    """Pure byte writer, isolated so allocated fixtures can use tiny fake assets.

    Production entry always authenticates the complete original and new CPU
    evidence through create_patch before calling this function.
    """
    old_report = baseline['report']; old_payload = old_report.parent
    old_html = (old_report / 'report.html').read_bytes(); new_html = patch_html(old_html)
    output_tree.mkdir(parents=True, exist_ok=False)
    report = payload = output_tree
    for name in ('assessment.json', 'report/report_receipt.json'):
        shutil.copyfile(old_payload / name, payload / ('original-' + Path(name).name))
    shutil.copyfile(patch_source_manifest_path, payload / 'focus-source-manifest.json')
    shutil.copytree(patch_cpu_proof_dir, payload / 'patch-cpu-proof')
    shutil.copytree(original_job_proof_dir, payload / 'original-render-job-proof')
    (report / 'report.html').write_bytes(new_html)
    adoption = {'version': VERSION, 'scope': 'Viewer event handling only; original saved data and scientific outcomes unchanged',
                'original_render_job': original_pin['job_id'], 'original_archive': original_pin,
                'original_report_source_manifest_sha256': ANALYSIS_MANIFEST,
                'original_report_cpu_certificate_sha256': ANALYSIS_CPU,
                'original_report_parent': str(original_parent),
                'original_archive_evidence_manifest_sha256': baseline['archive_evidence_manifest_sha256'],
                'original_report_receipt_sha256': sha(old_report / 'report_receipt.json'),
                'original_assessment_sha256': sha(old_payload / 'assessment.json'),
                'original_html_sha256': hashlib.sha256(old_html).hexdigest(),
                'patched_html_sha256': hashlib.sha256(new_html).hexdigest(),
                'patch_job_id': patch_job_id, 'patch_source_manifest_sha256': patch_source_manifest_sha256,
                'patch_cpu_certificate_sha256': patch_cpu_certificate_sha256,
                'patch': "Add exact-update input focus listener calling existing stop() before typing",
                'scientific_render_reexecuted': False, 'scientific_data_changed': False,
                'scientific_outcomes_changed': False, 'original_scientific_failures_preserved': True,
                'original_job_proofs': baseline['job_proofs'],
                'unchanged_served_assets': {p.relative_to(old_report).as_posix(): file_record(p)
                    for p in sorted(old_report.rglob('*')) if p.is_file()
                    and p.name not in ('report.html', 'report_receipt.json')}}
    receipt = dict(baseline['receipt'])
    receipt.update(html_bytes=len(new_html), html_sha256=adoption['patched_html_sha256'],
                   viewer_focus_adoption={'version': VERSION, 'patch_job_id': patch_job_id,
                       'patch_source_manifest_sha256': patch_source_manifest_sha256,
                       'original_render_job': original_pin['job_id'],
                       'original_html_sha256': adoption['original_html_sha256']})
    put(report / 'report_receipt.json', receipt)
    put(payload / 'focus-adoption.json', adoption)
    assessment = dict(baseline['assessment'])
    assessment.update(job_id=patch_job_id, report_receipt_sha256=sha(report / 'report_receipt.json'),
                      original_render_job=original_pin['job_id'], viewer_focus_adoption_sha256=sha(payload / 'focus-adoption.json'),
                      patch_source_manifest_sha256=patch_source_manifest_sha256,
                      patch_cpu_certificate_sha256=patch_cpu_certificate_sha256,
                      scientific_render_reexecuted=False,
                      scope='Allocated presentation-only event-handler patch; original render source/CPU provenance retained')
    put(payload / 'assessment.json', assessment)
    validate_delta_metadata(old_report, output_tree, adoption)
    return {'report': str(report), 'assessment': str(payload / 'assessment.json'),
            'adoption': str(payload / 'focus-adoption.json')}


def validate_delta_metadata(original_report, delta, adoption):
    original_report, delta = Path(original_report), Path(delta)
    original_records = tree_records(original_report)
    require(adoption.get('version') == VERSION and adoption.get('scientific_render_reexecuted') is False
            and adoption.get('scientific_data_changed') is False and adoption.get('scientific_outcomes_changed') is False
            and adoption.get('original_scientific_failures_preserved') is True, 'Adoption changed scientific scope')
    old_html = (original_report / 'report.html').read_bytes()
    require((delta / 'report.html').read_bytes() == patch_html(old_html), 'HTML delta exceeds one focus listener')
    require(sha(original_report / 'report_receipt.json') == adoption['original_report_receipt_sha256']
            and sha(original_report.parent / 'assessment.json') == adoption['original_assessment_sha256']
            and sha(original_report / 'report.html') == adoption['original_html_sha256']
            and sha(delta / 'report.html') == adoption['patched_html_sha256'], 'Adoption original/patched bytes differ')
    unchanged = {p.relative_to(original_report).as_posix(): file_record(p) for p in original_report.rglob('*')
                 if p.is_file() and p.name not in ('report.html', 'report_receipt.json')}
    require(unchanged == adoption['unchanged_served_assets'], 'Unchanged asset inventory differs')
    require(set(original_records) == set(unchanged) | {'report.html', 'report_receipt.json'}, 'Original served set differs')
    old_receipt, new_receipt = read(original_report / 'report_receipt.json'), read(delta / 'report_receipt.json')
    clean = dict(new_receipt)
    require(clean.pop('viewer_focus_adoption', {}) == {'version': VERSION,
        'patch_job_id': adoption['patch_job_id'], 'patch_source_manifest_sha256': adoption['patch_source_manifest_sha256'],
        'original_render_job': adoption['original_render_job'], 'original_html_sha256': adoption['original_html_sha256']},
        'Receipt lacks exact explicit adoption')
    clean['html_bytes'], clean['html_sha256'] = old_receipt['html_bytes'], old_receipt['html_sha256']
    require(clean == old_receipt, 'Original receipt semantics changed')
    old_assessment, new_assessment = read(original_report.parent / 'assessment.json'), read(delta / 'assessment.json')
    require(new_assessment['job_id'] == adoption['patch_job_id']
            and new_assessment['original_render_job'] == adoption['original_render_job']
            and new_assessment['report_source_manifest_sha256'] == ANALYSIS_MANIFEST
            and new_assessment['cpu_certificate_sha256'] == ANALYSIS_CPU
            and new_assessment['patch_source_manifest_sha256'] == adoption['patch_source_manifest_sha256']
            and new_assessment['patch_cpu_certificate_sha256'] == adoption['patch_cpu_certificate_sha256']
            and new_assessment['scientific_render_reexecuted'] is False
            and adoption['original_archive']['job_id'] == adoption['original_render_job'],
            'Original rendering provenance or actual patch job changed')
    check = dict(new_assessment)
    for field in ('original_render_job', 'viewer_focus_adoption_sha256', 'patch_source_manifest_sha256',
                  'patch_cpu_certificate_sha256', 'scientific_render_reexecuted'):
        require(field in check, 'Missing additive patch assessment field')
        check.pop(field)
    for field in ('job_id', 'report_receipt_sha256', 'scope'):
        check[field] = old_assessment[field]
    require(check == old_assessment, 'Original assessment scientific scope changed')
    require(new_assessment['viewer_focus_adoption_sha256'] == sha(delta / 'focus-adoption.json')
            and new_assessment['report_receipt_sha256'] == sha(delta / 'report_receipt.json'),
            'Patch assessment adoption checksum differs')
    require(new_receipt['html_bytes'] == (delta / 'report.html').stat().st_size
            and new_receipt['html_sha256'] == sha(delta / 'report.html'), 'Patched HTML receipt differs')
    require(sha(delta / 'original-assessment.json') == adoption['original_assessment_sha256']
            and sha(delta / 'original-report_receipt.json') == adoption['original_report_receipt_sha256'],
            'Verbatim original metadata backup changed')
    allowed = {'report.html', 'report_receipt.json', 'assessment.json', 'focus-adoption.json',
               'focus-source-manifest.json', 'original-assessment.json', 'original-report_receipt.json'}
    for name in tree_records(delta):
        require(name in allowed or name.startswith('patch-cpu-proof/')
                or name.startswith('original-render-job-proof/'), 'Unexpected focus delta file')
    return {'unchanged_served_assets': len(unchanged),
            'allowed_changed_files': ['report.html', 'report_receipt.json', 'assessment.json']}


def verify_adoption(original_tree, patched_tree, *, publisher, patch_job_proof_dir,
                    expected_patch_source_manifest_sha256, expected_patch_cpu_certificate_sha256):
    publisher.verify_tree(Path(patched_tree))
    found = list(Path(patched_tree).glob('**/focus-adoption.json'))
    require(len(found) == 1, 'Require exactly one focus delta')
    payload = found[0].parent
    adoption = read(payload / 'focus-adoption.json')
    require(adoption['patch_source_manifest_sha256'] == expected_patch_source_manifest_sha256
            and adoption['patch_cpu_certificate_sha256'] == expected_patch_cpu_certificate_sha256,
            'Additive patch source/CPU pins differ')
    original = verify_original(original_tree, publisher, adoption['original_archive'], payload / 'original-render-job-proof')
    require(original['job_proofs'] == adoption['original_job_proofs'], 'Original job proof inventory changed')
    require(sha(Path(original_tree) / 'evidence_manifest.json') == adoption['original_archive_evidence_manifest_sha256'],
            'Original archive evidence-manifest adoption differs')
    _source_provenance(payload / 'focus-source-manifest.json', expected_patch_source_manifest_sha256)
    verify_patch_cpu(payload / 'patch-cpu-proof', expected_patch_source_manifest_sha256,
                     expected_patch_cpu_certificate_sha256)
    patch_proofs = verify_patch_submission(patch_job_proof_dir, adoption['patch_job_id'],
                                           expected_patch_source_manifest_sha256)
    delta = validate_delta_metadata(report_path(original_tree), payload, adoption)
    return {'status': 'passed', 'version': VERSION, 'adoption': adoption, **delta,
            'patch_job_proofs': patch_proofs}


def publish_patch(original_tree, patched_tree, viewer, edition, *, publisher, patch_job_proof_dir,
                  expected_patch_source_manifest_sha256, expected_patch_cpu_certificate_sha256):
    """Validate additive provenance before the unchanged immutable publisher."""
    result = verify_adoption(original_tree, patched_tree, publisher=publisher,
                             patch_job_proof_dir=patch_job_proof_dir,
                             expected_patch_source_manifest_sha256=expected_patch_source_manifest_sha256,
                             expected_patch_cpu_certificate_sha256=expected_patch_cpu_certificate_sha256)
    original_report = report_path(original_tree)
    delta = next(Path(patched_tree).glob('**/focus-adoption.json')).parent
    viewer = Path(viewer).absolute()
    require(isinstance(edition, str) and publisher.EDITION.fullmatch(edition), 'Safe immutable edition required')
    publisher.validate_viewer(viewer, Path(original_tree))
    require(not viewer.is_relative_to(Path(patched_tree)) and not Path(patched_tree).is_relative_to(viewer),
            'Viewer and delta input must be separate')
    destination = viewer / edition
    require(not destination.exists(), 'Immutable edition already exists')
    # This full local assembly copies already verified bytes only. No HPC copy,
    # scientific rendering, probability decoding or numerical work occurs.
    with tempfile.TemporaryDirectory(prefix='jq1-focus-publication-', dir=viewer.parent) as staging:
        staged_report = Path(staging) / 'payload/report'
        shutil.copytree(original_report, staged_report)
        for name in ('report.html', 'report_receipt.json'):
            shutil.copyfile(delta / name, staged_report / name)
        old_payload = original_report.parent
        for name in ('report-source-manifest.json',):
            shutil.copyfile(old_payload / name, staged_report.parent / name)
        shutil.copytree(old_payload / 'cpu-proof', staged_report.parent / 'cpu-proof')
        shutil.copyfile(delta / 'assessment.json', staged_report.parent / 'assessment.json')
        receipt, data = publisher.validate_report(staged_report, Path(staging))
        assets = tree_records(staged_report)
        for name, entry in result['adoption']['unchanged_served_assets'].items():
            require(assets[name] == entry, 'Local unchanged asset assembly differs')
        viewer.mkdir(parents=True, exist_ok=True)
        marker = viewer / 'viewer-kind.json'
        if not marker.exists():
            publisher.atomic_json(marker, {'version': publisher.VIEWER_KIND, 'report_version': publisher.VERSION})
        destination.mkdir(exist_ok=False)
        for name, record in assets.items():
            target = name_path(destination, name); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name_path(staged_report, name), target)
            require(file_record(target) == record, 'Published asset copy differs')
        original_download = read(Path(original_tree).with_name(Path(original_tree).name + '-download.json'))
        delta_download = read(Path(patched_tree).with_name(Path(patched_tree).name + '-download.json'))
        latest = {'version': publisher.VERSION, 'path': edition + '/report.html', 'complete': receipt['complete'],
                  **{k: receipt[k] for k in ('presentation_complete', 'all_terminal', 'scientific_passed',
                      'strict_scientific_campaign_passed', 'overall_success', 'campaign_status')},
                  'input_jobs': publisher.EXPECTED_JOBS, 'included_updates': receipt['included_updates'],
                  'states': receipt['states'], 'milestone_progress': receipt['milestone_progress'],
                  'html_sha256': receipt['html_sha256'], 'report_receipt_sha256': sha(staged_report / 'report_receipt.json'),
                  'published_utc': datetime.now(timezone.utc).isoformat(),
                  'viewer_focus_adoption': {'version': VERSION, 'original_render_job': result['adoption']['original_render_job'],
                      'patch_job_id': result['adoption']['patch_job_id'],
                      'patch_source_manifest_sha256': expected_patch_source_manifest_sha256,
                      'patch_cpu_certificate_sha256': expected_patch_cpu_certificate_sha256,
                      'scientific_render_reexecuted': False}}
        publisher.atomic_json(destination / 'publication.json', latest | {
            'verified_original_input': str(original_tree), 'verified_delta_input': str(patched_tree),
            'verified_original_archive_sha256': original_download['sha256'],
            'verified_delta_archive_sha256': delta_download['sha256'],
            'verified_files_copied': len(assets)})
        temporary = viewer / 'report.html.tmp'
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(publisher.WRAPPER)
        temporary.replace(viewer / 'report.html')
        publisher.atomic_json(viewer / 'latest.json', latest)
    return {'publication': latest, 'viewer_focus_adoption': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    patch = sub.add_parser('patch')
    patch.add_argument('--input-archive', type=Path, required=True)
    patch.add_argument('--output-parent', type=Path, required=True)
    patch.add_argument('--original-parent', type=Path, required=True)
    patch.add_argument('--original-job', required=True)
    patch.add_argument('--original-archive-bytes', type=int, required=True)
    patch.add_argument('--original-archive-sha256', required=True)
    patch.add_argument('--original-job-proofs', type=Path, required=True)
    patch.add_argument('--patch-manifest', required=True)
    patch.add_argument('--patch-source-manifest', type=Path, required=True)
    patch.add_argument('--patch-cpu-certificate', required=True)
    patch.add_argument('--patch-cpu-proof-dir', type=Path, required=True)
    patch.add_argument('--patch-job', required=True)
    patch.add_argument('--analysis-source', type=Path, default=ANALYSIS_SOURCE)
    publish = sub.add_parser('publish')
    publish.add_argument('--original-tree', type=Path, required=True)
    publish.add_argument('--patched-tree', type=Path, required=True)
    publish.add_argument('--viewer', type=Path, required=True)
    publish.add_argument('--edition', required=True)
    publish.add_argument('--patch-job-proofs', type=Path, required=True)
    publish.add_argument('--patch-manifest', required=True)
    publish.add_argument('--patch-cpu-certificate', required=True)
    publish.add_argument('--analysis-source', type=Path, default=ANALYSIS_SOURCE)
    args = parser.parse_args()
    publisher = load_original_publisher(args.analysis_source)
    if args.command == 'publish':
        result = publish_patch(args.original_tree, args.patched_tree, args.viewer, args.edition,
            publisher=publisher, patch_job_proof_dir=args.patch_job_proofs,
            expected_patch_source_manifest_sha256=args.patch_manifest,
            expected_patch_cpu_certificate_sha256=args.patch_cpu_certificate)
    else:
        require(os.environ.get('LSB_JOBID') == args.patch_job, 'Patch CLI requires its actual allocated job')
        pin = {'job_id': args.original_job, 'archive_bytes': args.original_archive_bytes,
               'archive_sha256': args.original_archive_sha256}
        result = create_patch(args.original_parent, args.output_parent / 'payload', publisher=publisher,
            original_pin=pin, original_archive=args.input_archive, original_job_proof_dir=args.original_job_proofs,
            patch_source_manifest_sha256=args.patch_manifest,
            patch_source_manifest_path=args.patch_source_manifest, patch_job_id=args.patch_job,
            patch_cpu_certificate_sha256=args.patch_cpu_certificate,
            patch_cpu_proof_dir=args.patch_cpu_proof_dir)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
