"""Allocated CPU-only adoption of a saved report's keyboard-focus correction.

No original scientific, diagnostic, collector or report-renderer job is invoked.
The original report archive remains read-only; one new presentation tree is made.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

TAG = '20261022-brd4-jq1-lrboost-viewer-focus-01'
ANALYSIS_TAG = '20261022-brd4-jq1-lrboost-analysis-01'
ANALYSIS = Path('/work3/s234818/mosaic/releases') / ANALYSIS_TAG / 'source'
ANALYSIS_ROOT = Path('/work3/s234818/mosaic/runs') / ANALYSIS_TAG
# Nest every new byte under the original analysis root so its existing, sealed
# diagnostic-entry quota gates continue to charge the complete new adoption.
ROOT = ANALYSIS_ROOT / 'viewer-focus-01'
SOURCE = ANALYSIS_ROOT / 'viewer-focus-release-01/source'
UPLOAD = ANALYSIS_ROOT / (TAG + '.tar.gz')
ANALYSIS_MANIFEST = '214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc'
ANALYSIS_CPU = '42b2b000ecce5d661da9b75ad85f03a41c3b303010ccfa03f87835a25993ca69'
SCIENCE = Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source')
SCIENCE_MANIFEST = '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323'
SCIENCE_ROOT = ANALYSIS_ROOT.parent / '20261021-brd4-jq1-25-fixed-lrboost-01'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
SUITE = 'jq1-lrboost-viewer-focus-contract-v1'
ORIGINAL_PIN = {'job_id': '29781041', 'archive_bytes': 157404869,
                'archive_sha256': 'e92c477b8d173c1c0a9e17813e9cb0f8549cd24c9e97578d43e7591823cbb30a'}
ORIGINAL_PARENT = ANALYSIS_ROOT / 'interim-0002'
ORIGINAL_ARCHIVE = ANALYSIS_ROOT / 'interim-0002.tar.gz'
NEW_TESTS = ('test_viewer_focus.py', 'test_viewer_focus_ops.py')
OLD_COUNTS = {'test_jq1_lrboost_report_contract': 82, 'test_jq1_lrboost_render': 54,
              'test_jq1_lrboost_publish': 59, 'test_jq1_lrboost_ops': 32,
              'test_lrboost_endpoint_diagnostics': 16, 'test_boltz_affinity_report': 26,
              'test_small_molecule_chemistry_report': 3, 'test_small_molecule_chemistry_ramp_report': 6}
FULL_NATIVE = {'protenix': 1514341824, 'boltz2': 1820641824}
RESERVE = 4 * 1024**3
FILE_RESERVE = 100000


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    exclusive(temporary, value); temporary.replace(path)


def authenticate_source(root, manifest):
    """Stdlib authentication before importing any external execution input."""
    root = Path(root)
    require(root.resolve() == root and not root.is_symlink(), 'Source identity is aliased')
    require(sha(root / 'source_manifest.json') == manifest, 'Source manifest changed')
    files = read(root / 'source_manifest.json')['files']
    require(isinstance(files, dict) and files, 'Empty source manifest')
    for name, expected in files.items():
        rel = PurePosixPath(name); q = root.joinpath(*rel.parts)
        require(not rel.is_absolute() and '..' not in rel.parts and q.resolve().is_relative_to(root)
                and q.is_file() and not q.is_symlink() and sha(q) == expected, 'Source bytes escaped/changed: ' + name)
    actual = set()
    for q in root.rglob('*'):
        require(not q.is_symlink(), 'Source symlink is forbidden')
        if q.is_file() and '__pycache__' not in q.parts and q.name != 'source_manifest.json':
            actual.add(q.relative_to(root).as_posix())
    require(actual == set(files), 'Unmanifested source execution input')
    return files


def load_external_ops():
    spec = importlib.util.spec_from_file_location('focus_authenticated_analysis_ops', ANALYSIS / 'scripts/jq1_lrboost_report_ops.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def authenticate_external():
    authenticate_source(ANALYSIS, ANALYSIS_MANIFEST); authenticate_source(SCIENCE, SCIENCE_MANIFEST)
    require(sha(ANALYSIS_ROOT / 'cpu-certificate.json') == ANALYSIS_CPU, 'Original analysis CPU certificate changed')
    old = load_external_ops(); certificate = old.verify_cpu(ANALYSIS_MANIFEST)
    require(certificate['tests'] == 278 and certificate['tests_by_module'] == OLD_COUNTS, 'Original 278-case proof differs')
    return old


def whole_budget(current, quota_available, disk_free, stale):
    require(set(current) == set(FULL_NATIVE) and all(type(n) is int and 0 <= n <= FULL_NATIVE[a]
            for a, n in current.items()), 'Native projection/current bytes differ')
    require(all(type(n) is int and n >= 0 for n in (quota_available, disk_free, stale)), 'Malformed storage accounting')
    outstanding = sum(FULL_NATIVE[a] - current[a] for a in FULL_NATIVE)
    allowance = 2 * 1024**3 + 512 * 1024**2
    return {'current_native_bytes': current, 'full_native_projections': FULL_NATIVE,
            'outstanding_native_bytes': outstanding, 'analysis_allowance_bytes': allowance,
            'reserve_bytes': RESERVE, 'file_reserve': FILE_RESERVE,
            'stale_bytes_subtracted': stale, 'disk_free_bytes': disk_free,
            'available_bytes_conservative': min(quota_available, disk_free) - stale,
            'required_bytes': outstanding + allowance + RESERVE}


def stale_paths():
    old_tags = ('20261021-brd4-jq1-25-fixed-02', '20261021-brd4-jq1-25-fixed-storage-03',
                '20261021-brd4-jq1-25-fixed-storage-04', *('20261021-brd4-jq1-25-fixed-report-' + n for n in ('01', '02', '03', '04', '05')),
                '20261021-brd4-jq1-25-fixed-lrboost-01')
    return [*(ANALYSIS_ROOT.parent / t for t in old_tags), ANALYSIS_ROOT, ANALYSIS.parent,
            Path('/work3/s234818/mosaic') / (ANALYSIS_TAG + '.tar.gz')]


def check_storage(parent, phase):
    """New receipts only; never call old controller check_storage with a new root."""
    require(phase in ('preflight', 'cpu', 'certificate', 'dispatch', 'patch'), 'Undeclared focus phase')
    require(ROOT.resolve() == ROOT and Path(parent).resolve().is_relative_to(ROOT), 'Wrong new storage scope')
    from fixed25_split_storage import tree_bytes, parse_work3, inode_observation, inode_headroom
    folder = Path(parent) / ('storage-' + phase); folder.mkdir(exist_ok=False)
    record = {'status': 'failed', 'phase': phase, 'utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Viewer-only output; unchanged full original analysis/native remaining projections and reserves'}
    try:
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60).stdout
        quota = parse_work3(raw); observed = inode_observation(os.statvfs(ROOT)); inode = inode_headroom(quota, observed)
        paths = stale_paths(); require(len(set(paths)) == len(paths), 'Duplicate stale accounting root')
        require(all(p.is_relative_to(ANALYSIS_ROOT) for p in (ROOT, SOURCE.parent, UPLOAD)), 'New bytes escape original sealed accounting')
        measured = [{'path': str(p), 'bytes': tree_bytes(p)} for p in paths]
        current = {a: tree_bytes(SCIENCE_ROOT / ('run-' + a)) for a in FULL_NATIVE}
        record.update(raw_quota=raw, quota=quota, filesystem_inode_observation=observed, inode=inode,
                      stale_accounting=measured, new_nested_paths_covered_once=[str(p) for p in (ROOT, SOURCE.parent, UPLOAD)],
                      **whole_budget(current, quota['available_bytes_lower'], shutil.disk_usage(ROOT).free,
                                                              sum(p['bytes'] for p in measured)))
        record['status'] = 'passed' if record['available_bytes_conservative'] >= record['required_bytes'] and inode['available_files_lower'] >= FILE_RESERVE else 'failed'
    except Exception as error:
        record['error'] = repr(error); raise
    finally:
        exclusive(folder / 'gate.json', record)
    require(record['status'] == 'passed', 'Full native/analysis remaining budget plus byte/file reserves does not fit')
    return record


def resources(phase):
    require(phase in ('cpu', 'certify-cpu', 'patch'), 'Only viewer CPU phases are declared')
    return ['-q', 'hpc', '-n', '4' if phase == 'cpu' else '1' if phase == 'certify-cpu' else '2',
            '-R', 'span[hosts=1]', '-R', 'rusage[mem=2GB]' if phase == 'certify-cpu' else 'rusage[mem=4GB]',
            '-W', '01:00' if phase == 'cpu' else '00:30']


def command(name, phase, manifest, dependency=None):
    require((name, phase) in (('cpu', 'cpu'), ('after-cpu', 'certify-cpu'), ('patch', 'patch')), 'Undeclared viewer parent/phase')
    parent = ROOT / name
    result = ['bsub', *resources(phase), '-J', 'mosaic-viewer-focus-' + name, '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if phase == 'certify-cpu':
        require(isinstance(dependency, str) and re.fullmatch(r'ended\([0-9]+\)', dependency), 'Certificate requires its own ended CPU dependency')
        result += ['-w', dependency]
    else:
        require(dependency is None, 'Unexpected viewer dependency')
    result += ['bash', '-l', str(SOURCE / 'stage.lsf'), str(SOURCE), str(ROOT), manifest, phase, str(parent)]
    return result


def submit(name, phase, manifest, *, dependency=None):
    args = command(name, phase, manifest, dependency); parent = ROOT / name; parent.mkdir(exist_ok=False)
    record = {'phase': phase, 'args': args, 'dependency': dependency, 'outcome': 'unknown', 'utc': datetime.now(timezone.utc).isoformat()}
    exclusive(parent / 'submission-started.json', record)
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=60)
        record.update(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        jobs = re.findall(r'Job <([0-9]+)>', r.stdout)
        if r.returncode == 0 and len(jobs) == 1:
            record.update(outcome='submitted', job_id=jobs[0])
    except Exception as error:
        record['error'] = repr(error)
    finally:
        exclusive(parent / 'submission.json', record)
    require(record['outcome'] == 'submitted', 'Unknown/failed submission; inspect exact markers, never duplicate')
    r = subprocess.run(['bjobs', '-l', record['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(r.stdout + r.stderr)
    return record


def bootstrap(manifest):
    """Metadata-only initial dispatch, exactly once after all immutable bytes pass."""
    authenticate_source(SOURCE, manifest); authenticate_external()
    ROOT.mkdir(exist_ok=False)
    exclusive(ROOT / 'bootstrap-started.json', {'source_manifest_sha256': manifest, 'utc': datetime.now(timezone.utc).isoformat()})
    parent = ROOT / 'preflight'; parent.mkdir(exist_ok=False); check_storage(parent, 'preflight')
    with (ROOT / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = {'cpu': submit('cpu', 'cpu', manifest)['job_id']}; atomic(ROOT / 'jobs.json', jobs)
        jobs['after_cpu'] = submit('after-cpu', 'certify-cpu', manifest, dependency='ended(' + jobs['cpu'] + ')')['job_id']
        atomic(ROOT / 'jobs.json', jobs); exclusive(ROOT / 'initial-submissions-complete.json', jobs)
    return jobs


def validate_submission(parent, phase, manifest):
    path = Path(parent) / 'submission.json'
    for attempt in range(16):
        if path.exists(): break
        require(attempt < 15, 'Own actual submission receipt is missing'); time.sleep(2)
    sub = read(path)
    if phase == 'certify-cpu':
        require(sub.get('dependency') == 'ended(' + read(ROOT / 'jobs.json')['cpu'] + ')', 'Certificate does not await its own CPU job')
    require(sub.get('outcome') == 'submitted' and sub.get('job_id') == os.environ.get('LSB_JOBID')
            and sub.get('phase') == phase and sub.get('args') == command(Path(parent).name, phase, manifest, sub.get('dependency')),
            'Executing viewer job differs from exact own submission/resources')
    return sub


def test_paths():
    return [*(SOURCE / n for n in NEW_TESTS),
            *(ANALYSIS / 'tests' / (n + '.py') for n in OLD_COUNTS if n.startswith('test_jq1_') or n == 'test_lrboost_endpoint_diagnostics'),
            *(SCIENCE / 'tests' / (n + '.py') for n in OLD_COUNTS if n not in {'test_jq1_lrboost_report_contract', 'test_jq1_lrboost_render', 'test_jq1_lrboost_publish', 'test_jq1_lrboost_ops', 'test_lrboost_endpoint_diagnostics'})]


def collection_counts(raw):
    names = {Path(n).stem for n in NEW_TESTS} | set(OLD_COUNTS); result = dict.fromkeys(names, 0)
    nodeids = []
    for line in raw.splitlines():
        match = re.fullmatch(r'(.+?\.py)::(.+)', line.strip())
        if match:
            module = Path(match[1]).stem; require(module in names, 'Unrelated collected fixture module')
            result[module] += 1; nodeids.append(line.strip())
    require(len(nodeids) == len(set(nodeids)) and all(result.values()), 'Missing or duplicate collected cases')
    require(all(result[n] == count for n, count in OLD_COUNTS.items()), 'Original 278 fixture counts changed')
    totals = re.findall(r'(\d+) tests? collected', raw)
    require(totals == [str(len(nodeids))], 'Collection total differs from exact node IDs')
    return result


def xml_cases(path, expected):
    names = {Path(n).stem for n in NEW_TESTS} | set(OLD_COUNTS)
    require(set(expected) == names and all(type(n) is int and n > 0 for n in expected.values())
            and all(expected[n] == count for n, count in OLD_COUNTS.items()), 'Wrong exact dynamic module expectations')
    root = ET.parse(path).getroot(); cases = list(root.iter('testcase')); counts = dict.fromkeys(names, 0)
    require(cases and not any(list(root.iter(t)) for t in ('failure', 'error', 'skipped')), 'Fixture failed/errored/skipped')
    for case in cases:
        matches = names.intersection(case.get('classname', '').split('.'))
        require(len(matches) == 1, 'Ambiguous/unrelated fixture module'); counts[matches.pop()] += 1
    require(counts == expected, 'Executed raw module counts differ from collection')
    for suite in list(root.iter('testsuite')) + list(root.iter('testsuites')):
        require(all(int(suite.get(k, '0')) == 0 for k in ('failures', 'errors', 'skipped'))
                and ('tests' not in suite.attrib or int(suite.get('tests')) == len(list(suite.iter('testcase')))), 'Raw XML header totals differ')
    return cases


def cpu_tests(parent):
    check_storage(parent, 'cpu'); run = parent / 'payload'; run.mkdir(exist_ok=False)
    common = [PYTHON, '-m', 'pytest', '--rootdir', str(ANALYSIS.parents[1]), *map(str, test_paths()),
              '-o', 'addopts=', '-o', 'cache_dir=' + str(run / 'cache'), '--basetemp', str(Path(os.environ['TMPDIR']) / 'focus-tests')]
    collect_args = [*common, '--collect-only', '-q']
    exclusive(run / 'collect-invocation.json', {'args': collect_args, 'cwd': str(SOURCE)})
    with (run / 'collect-only.log').open('x') as log:
        subprocess.run(collect_args, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
    counts = collection_counts((run / 'collect-only.log').read_text())
    exclusive(run / 'collection.json', {'tests_by_module': counts, 'tests': sum(counts.values())})
    args = [*common, '--junitxml', str(run / 'report-tests.xml')]
    exclusive(run / 'test-invocation.json', {'args': args, 'cwd': str(SOURCE)})
    with (run / 'report-tests.log').open('x') as log:
        subprocess.run(args, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
    xml_cases(run / 'report-tests.xml', counts)


def verify_cpu(manifest):
    c = read(ROOT / 'cpu-certificate.json'); parent = ROOT / 'cpu'
    require(c['status'] == 'passed' and c['suite'] == SUITE and c['source_manifest_sha256'] == manifest
            and c['original_analysis_cpu_certificate_sha256'] == ANALYSIS_CPU, 'Wrong focus CPU certificate')
    for p in c['proofs']:
        q = Path(p['path']); require(q.resolve().is_relative_to(parent) and not q.is_symlink()
                                    and q.stat().st_size == p['bytes'] and sha(q) == p['sha256'], 'CPU proof escaped/changed')
    old = load_external_ops(); job = c['producer_job_id']
    sub = read(parent / 'submission.json')
    require(sub.get('outcome') == 'submitted' and sub.get('job_id') == job and sub.get('phase') == 'cpu'
            and sub.get('args') == command('cpu', 'cpu', manifest), 'Certified CPU differs from exact actual submitted resources')
    scheduler = [p for p in c['proofs'] if Path(p['path']).name.startswith('scheduler-final-')]
    require(len(scheduler) == 1, 'Missing unique actual CPU scheduler proof')
    old.actual_scheduler(Path(scheduler[0]['path']).read_text(), job)
    old.actual_lsf((parent / ('stdout.' + job + '.log')).read_text(), job)
    counts = collection_counts((parent / 'payload/collect-only.log').read_text())
    require(counts == c['tests_by_module'] == read(parent / 'payload/collection.json')['tests_by_module'], 'CPU collection/certificate differs')
    require(len(xml_cases(parent / 'payload/report-tests.xml', counts)) == c['tests'], 'CPU raw case total differs')
    log = (parent / 'payload/report-tests.log').read_text()
    require(re.findall(r'collected\s+(\d+)\s+items?', log) == [str(c['tests'])]
            and re.search(r'\b' + str(c['tests']) + r'\s+passed\b', log)
            and not re.search(r'\d+\s+(?:deselected|skipped|failed|errors?)\b', log), 'Certified CPU raw log is not a full pass')
    return c


def certify_cpu(manifest):
    from campaign25_pocket_ops import final_job
    check_storage(ROOT / 'after-cpu', 'certificate'); jobs = read(ROOT / 'jobs.json'); job = jobs['cpu']
    status, queue, proofs = final_job(job, ROOT / 'cpu'); require(status == 'DONE' and queue == 'hpc', 'Focus CPU final execution failed')
    counts = collection_counts((ROOT / 'cpu/payload/collect-only.log').read_text())
    cases = xml_cases(ROOT / 'cpu/payload/report-tests.xml', counts)
    log = (ROOT / 'cpu/payload/report-tests.log').read_text()
    require(re.findall(r'collected\s+(\d+)\s+items?', log) == [str(len(cases))]
            and re.search(r'\b' + str(len(cases)) + r'\s+passed\b', log)
            and not re.search(r'\d+\s+(?:deselected|skipped|failed|errors?)\b', log), 'Raw CPU execution count/result differs')
    for rel in ('submission.json', 'payload/report-tests.xml', 'payload/report-tests.log', 'payload/test-invocation.json',
                'payload/collect-invocation.json', 'payload/collect-only.log', 'payload/collection.json'):
        q = ROOT / 'cpu' / rel; proofs.append({'path': str(q), 'sha256': sha(q)})
    for p in proofs: p['bytes'] = Path(p['path']).stat().st_size
    c = {'status': 'passed', 'producer_job_id': job, 'job_id': job, 'tests': len(cases), 'tests_by_module': counts,
         'suite': SUITE, 'source_manifest_sha256': manifest, 'original_analysis_cpu_certificate_sha256': ANALYSIS_CPU,
         'original_analysis_manifest_sha256': ANALYSIS_MANIFEST, 'scientific_manifest_sha256': SCIENCE_MANIFEST, 'proofs': proofs}
    exclusive(ROOT / 'cpu/payload/assessment.json', c); exclusive(ROOT / 'cpu-certificate.json', c); verify_cpu(manifest)
    with (ROOT / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exclusive(ROOT / 'patch-dispatch-started.json', {'source_manifest_sha256': manifest, 'cpu_certificate_sha256': sha(ROOT / 'cpu-certificate.json')})
        check_storage(ROOT / 'after-cpu', 'dispatch')
        jobs['patch'] = submit('patch', 'patch', manifest)['job_id']; jobs['cpu_certificate_job'] = os.environ['LSB_JOBID']; atomic(ROOT / 'jobs.json', jobs)
    return jobs


def bundle_cpu(destination, manifest):
    c = verify_cpu(manifest); destination.mkdir(exist_ok=False); shutil.copyfile(ROOT / 'cpu-certificate.json', destination / 'assessment.json')
    refs = []
    for p in c['proofs']:
        q = Path(p['path']); target = destination / 'evidence' / q.relative_to(ROOT / 'cpu')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(q, target)
        refs.append({'original_path': str(q), 'local_path': target.relative_to(destination).as_posix(), 'sha256': p['sha256'], 'bytes': p['bytes']})
    exclusive(destination / 'bundled-proofs.json', refs)
    for name in ('report-tests.xml', 'report-tests.log'):
        shutil.copyfile(ROOT / 'cpu/payload' / name, destination / name)


def original_report_proofs(destination):
    old = load_external_ops(); job = ORIGINAL_PIN['job_id']; destination.mkdir(exist_ok=False)
    r = subprocess.run(['bjobs', '-l', job], capture_output=True, text=True, timeout=30)
    raw = r.stdout + r.stderr; (destination / 'scheduler.txt').write_text(raw)
    require(r.returncode == 0, 'Actual original report final scheduler unavailable')
    old.actual_scheduler(raw, job); stdout = ORIGINAL_PARENT / ('stdout.' + job + '.log'); old.actual_lsf(stdout.read_text(), job)
    shutil.copyfile(stdout, destination / 'stdout.log'); shutil.copyfile(ORIGINAL_PARENT / 'submission.json', destination / 'submission.json')
    sub = read(destination / 'submission.json')
    expected = ['bsub', *old.resources('interim'), '-J', 'mosaic-lrboost-analysis-interim-0002',
                '-oo', str(ORIGINAL_PARENT / 'stdout.%J.log'), '-eo', str(ORIGINAL_PARENT / 'stderr.%J.log'),
                'bash', '-l', str(ANALYSIS / 'scripts/hpc/jq1_lrboost_analysis_stage.lsf'), str(ANALYSIS), str(ANALYSIS_ROOT),
                ANALYSIS_MANIFEST, 'interim', 'none', str(ORIGINAL_PARENT)]
    require(sub['job_id'] == job and sub['outcome'] == 'submitted' and sub['phase'] == 'interim'
            and sub.get('arm') == 'none' and sub.get('dependency') is None and sub.get('args') == expected,
            'Wrong exact original report submission/resources')
    receipt = Path(str(ORIGINAL_PARENT) + '.tar.receipt.json'); entry = read(receipt)
    require(Path(entry['path']) == ORIGINAL_ARCHIVE and entry['bytes'] == ORIGINAL_PIN['archive_bytes']
            and entry['sha256'] == ORIGINAL_PIN['archive_sha256'] and ORIGINAL_ARCHIVE.stat().st_size == entry['bytes']
            and sha(ORIGINAL_ARCHIVE) == entry['sha256'], 'Original report archive bytes/pin changed')
    shutil.copyfile(receipt, destination / 'archive-receipt.json')
    return entry


def produce_patch(parent, manifest):
    verify_cpu(manifest); check_storage(parent, 'patch')
    from viewer_focus import create_patch, load_original_publisher
    temporary = Path(os.environ['TMPDIR']) / 'viewer-focus'; temporary.mkdir(exist_ok=False)
    oldproof = temporary / 'original-job-proof'; original_report_proofs(oldproof)
    patchproof = temporary / 'patch-cpu-proof'; bundle_cpu(patchproof, manifest)
    result = create_patch(ORIGINAL_PARENT, parent / 'payload', publisher=load_original_publisher(ANALYSIS),
                          patch_source_manifest_sha256=manifest, patch_job_id=os.environ['LSB_JOBID'],
                          patch_source_manifest_path=SOURCE / 'source_manifest.json',
                          patch_cpu_certificate_sha256=sha(ROOT / 'cpu-certificate.json'), patch_cpu_proof_dir=patchproof,
                          original_pin=ORIGINAL_PIN, original_archive=ORIGINAL_ARCHIVE, original_job_proof_dir=oldproof)
    exclusive(parent / 'patch-result.json', result)
    subprocess.run([PYTHON, str(SCIENCE / 'scripts/collect_protenix_evidence.py'), '--storage', 'work3', '--run', str(parent),
                    '--output', str(parent) + '.tar.gz'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'cpu', 'certify-cpu', 'patch'), required=True)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', required=True); parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    require(args.source.resolve() == SOURCE and args.root.resolve() == ROOT and args.parent.resolve().is_relative_to(ROOT), 'Wrong immutable focus scope')
    authenticate_source(SOURCE, args.manifest); authenticate_external()
    if args.phase == 'preflight': print(json.dumps(check_storage(args.parent, 'preflight'))); return
    require(bool(os.environ.get('LSB_JOBID')), 'All validation/patch work requires allocated HPC')
    validate_submission(args.parent, args.phase, args.manifest)
    if args.phase == 'cpu': cpu_tests(args.parent)
    elif args.phase == 'certify-cpu': print(json.dumps(certify_cpu(args.manifest)))
    else: produce_patch(args.parent, args.manifest)


if __name__ == '__main__': main()
