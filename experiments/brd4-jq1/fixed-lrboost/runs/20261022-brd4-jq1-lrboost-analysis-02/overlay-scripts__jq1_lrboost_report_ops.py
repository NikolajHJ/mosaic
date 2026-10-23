"""Separate allocated endpoint diagnostics and saved-only report orchestration.

No scientific trajectory is submitted, stopped, resumed or modified. A passed
presentation assessment authenticates evidence, never a failed scientific run.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from jq1_lrboost_report_contract import ARMS, SCIENCE, SCIENCE_MANIFEST, read, require, sha

TAG = '20261022-brd4-jq1-lrboost-analysis-02'
ROOT = Path('/work3/s234818/mosaic/runs') / TAG
SOURCE = Path('/work3/s234818/mosaic/releases') / TAG / 'source'
SCIENCE_ROOT = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01')
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
SUITE = 'jq1-lrboost-saved-report-contract-v1'
TESTS = ('test_jq1_lrboost_report_contract.py', 'test_jq1_lrboost_render.py',
         'test_jq1_lrboost_publish.py', 'test_jq1_lrboost_ops.py', 'test_lrboost_endpoint_diagnostics.py')
LEGACY_TESTS = ('test_boltz_affinity_report.py', 'test_small_molecule_chemistry_report.py',
                'test_small_molecule_chemistry_ramp_report.py')
DIAGNOSTIC_BYTES = 256 * 1024**2
FULL_NATIVE = {'protenix': 1514341824, 'boltz2': 1820641824}


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def actual_scheduler(raw, job, queue='hpc'):
    compact = re.sub(r'\s+', '', raw)
    require(re.findall(r'Job<([^>]+)>', compact) == [job]
            and re.findall(r'Queue<([^>]+)>', compact) == [queue], 'Actual job/queue identity differs')
    require(re.findall(r'Status<([^>]+)>', compact) == ['DONE'], 'Actual diagnostic/report job is not DONE')


def actual_lsf(raw, job):
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', raw, re.M)
    require(len(subjects) == 1 and subjects[0][0] == job and subjects[0][1].rstrip().endswith(' Done')
            and re.search(r'^Successfully completed\.\s*$', raw, re.M)
            and not re.search(r'Exited\s+(?:with exit|by signal)', raw, re.I), 'Actual same-job LSF success missing')


def report_budget(current, quota_available, disk_free, stale, phase):
    require(phase in ('preflight', 'cpu', 'capture', 'interim', 'final', 'diagnostic', 'diagnostic-capture', 'dispatch'), 'Unknown analysis budget phase')
    require(set(current) == set(FULL_NATIVE) and all(type(n) is int and 0 <= n <= FULL_NATIVE[a]
            for a, n in current.items()), 'Native output exceeded pinned full projection')
    require(all(type(n) is int and n >= 0 for n in (quota_available, disk_free, stale)), 'Malformed quota/disk/stale accounting')
    outstanding = sum(FULL_NATIVE[a] - current[a] for a in FULL_NATIVE)
    allowance = 1024**3 if phase in ('preflight', 'cpu') else 2 * 1024**3 + 2 * DIAGNOSTIC_BYTES
    return {'current_native_bytes': current, 'full_native_projections': FULL_NATIVE,
            'outstanding_native_bytes': outstanding, 'analysis_allowance_bytes': allowance,
            'diagnostic_per_arm_hard_output_bytes': DIAGNOSTIC_BYTES,
            'stale_bytes_subtracted': stale, 'available_bytes_conservative': min(quota_available, disk_free) - stale,
            'required_bytes': allowance + outstanding + 4 * 1024**3}


def check_storage(parent, phase):
    """Work3 outputs only; retain both native work3 arms' remaining projections."""
    from fixed25_split_storage import tree_bytes, parse_work3, inode_observation, inode_headroom, RESERVE, FILE_RESERVE
    require(ROOT.resolve() == ROOT and Path(parent).resolve().is_relative_to(ROOT), 'Wrong analysis storage scope')
    folder = Path(parent) / ('storage-' + phase); folder.mkdir(exist_ok=False)
    record = {'status': 'failed', 'phase': phase, 'utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'New work3 diagnostics/report only; original studies unchanged; full remaining native work3 projections retained',
              'reserve_bytes': RESERVE, 'file_reserve': FILE_RESERVE}
    try:
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60).stdout
        quota = parse_work3(raw); observed = inode_observation(os.statvfs(ROOT)); inode = inode_headroom(quota, observed)
        current = {a: tree_bytes(SCIENCE_ROOT / ('run-' + a)) for a in FULL_NATIVE}
        old_tags = ('20261021-brd4-jq1-25-fixed-02', '20261021-brd4-jq1-25-fixed-storage-03',
                    '20261021-brd4-jq1-25-fixed-storage-04', *('20261021-brd4-jq1-25-fixed-report-' + n for n in ('01', '02', '03', '04', '05')),
                    '20261021-brd4-jq1-25-fixed-lrboost-01', '20261022-brd4-jq1-lrboost-analysis-01')
        charged = sum(tree_bytes(ROOT.parent / tag) for tag in old_tags)
        charged += sum(tree_bytes(p) for p in (ROOT, SOURCE.parent, Path('/work3/s234818/mosaic') / (TAG + '.tar.gz')))
        budget = report_budget(current, quota['available_bytes_lower'], shutil.disk_usage(ROOT).free, charged, phase)
        lower_root = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-03')
        lower_marker = lower_root / 'long-dispatch-started.json'
        if lower_marker.exists():
            projected = read(lower_marker)['projections']
            require(set(projected) == set(ARMS) and all(type(n) is int and n >= 1024**3 for n in projected.values()),
                    'Malformed actual lower-rate whole-campaign projections')
            current_lower = {a: tree_bytes(lower_root / ('run-' + a)) for a in ('protenix', 'boltz2')}
            require(all(current_lower[a] <= projected[a] for a in current_lower), 'Lower-rate output exceeded projection')
            outstanding_lower = sum(projected[a] - current_lower[a] for a in current_lower)
            record['concurrent_lower_rate_scope'] = 'Actual authenticated whole-dispatch projection minus current work3 outputs'
        else:
            outstanding_lower = 3 * 1024**3
            record['concurrent_lower_rate_scope'] = 'Full grouped three-arm native validation allowance before any long dispatch'
        budget['required_bytes'] += outstanding_lower
        record['concurrent_lower_rate_outstanding_bytes'] = outstanding_lower
        record.update(raw_quota=raw, quota=quota, inode=inode, filesystem_inode_observation=observed, **budget)
        record['status'] = 'passed' if budget['available_bytes_conservative'] >= budget['required_bytes'] and inode['available_files_lower'] >= FILE_RESERVE else 'failed'
    except Exception as error:
        record['error'] = repr(error); raise
    finally:
        exclusive(folder / 'gate.json', record)
    require(record['status'] == 'passed', 'Full analysis/native remaining byte/inode budget plus reserve does not fit')
    return record


def cpu_tests(parent):
    run = parent / 'payload'; run.mkdir(exist_ok=False); check_storage(parent, 'cpu')
    command = [PYTHON, '-m', 'pytest', '--rootdir', str(SOURCE.parents[1]),
               *[str(SOURCE / 'tests' / n) for n in TESTS], *[str(SCIENCE / 'tests' / n) for n in LEGACY_TESTS],
               '-o', 'addopts=', '-o', 'cache_dir=' + str(run / 'cache'), '--basetemp',
               str(Path(os.environ['TMPDIR']) / 'lrboost-analysis-tests'), '--junitxml', str(run / 'report-tests.xml')]
    exclusive(run / 'test-invocation.json', {'args': command, 'cwd': str(SOURCE)})
    with (run / 'report-tests.log').open('x') as log:
        subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)


def xml_cases(path):
    root = ET.parse(path).getroot(); cases = list(root.iter('testcase'))
    require(cases and not any(list(root.iter(t)) for t in ('failure', 'error', 'skipped')), 'Analysis fixture failures/errors/skips')
    expected = {Path(n).stem for n in TESTS + LEGACY_TESTS}
    modules = {stem for c in cases for stem in c.get('classname', '').split('.') if stem in expected}
    require(modules == expected and all(expected.intersection(c.get('classname', '').split('.')) for c in cases), 'Not every required fixture module ran')
    for suite in list(root.iter('testsuite')) + list(root.iter('testsuites')):
        require(all(int(suite.get(k, '0')) == 0 for k in ('failures', 'errors', 'skipped'))
                and ('tests' not in suite.attrib or int(suite.get('tests')) == len(list(suite.iter('testcase')))), 'Raw XML suite totals differ')
    return cases


def verify_cpu(manifest):
    path = ROOT / 'cpu-certificate.json'; c = read(path)
    require(c['status'] == 'passed' and c['suite'] == SUITE and c['source_manifest_sha256'] == manifest, 'Wrong analysis CPU certificate')
    for proof in c['proofs']:
        q = Path(proof['path']); require(q.resolve().is_relative_to(ROOT / 'cpu') and sha(q) == proof['sha256'], 'CPU evidence changed/escaped')
    schedulers = [p for p in c['proofs'] if re.fullmatch(r'scheduler-final-\d+\.txt', Path(p['path']).name)]
    logs = [p for p in c['proofs'] if Path(p['path']).name == 'stdout.' + c['producer_job_id'] + '.log']
    require(len(schedulers) == len(logs) == 1, 'Missing unique actual CPU/LSF proof')
    actual_scheduler(Path(schedulers[0]['path']).read_text(), c['producer_job_id'])
    actual_lsf(Path(logs[0]['path']).read_text(), c['producer_job_id'])
    require(len(xml_cases(ROOT / 'cpu/payload/report-tests.xml')) == c['tests'], 'CPU case count differs')
    return c


def submission_plan():
    return [('diagnostic-' + a, 'diagnostic', a, None) for a in ('protenix', 'boltz2')] + [
        ('complete-' + a, 'capture', a, 'ended(' + ARMS[a]['job'] + ')' if a == 'boltz2_chemistry' else None) for a in ARMS]


def resources(phase):
    if phase == 'diagnostic':
        return ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00']
    if phase in ('certify-cpu', 'certify-protenix-only'):
        return ['-q', 'hpc', '-n', '1', '-R', 'span[hosts=1]', '-R', 'rusage[mem=2GB]', '-W', '00:30']
    return ['-q', 'hpc', '-n', '4' if phase in ('interim', 'final') else '2', '-R', 'span[hosts=1]',
            '-R', 'rusage[mem=8GB]' if phase in ('interim', 'final') else 'rusage[mem=4GB]', '-W', '01:00' if phase in ('interim', 'final') else '00:45']


def submit(name, phase, manifest, *, arm='none', dependency=None):
    require(re.fullmatch(r'(?:(?:complete|diagnostic|diagnostic-capture)-(?:protenix|boltz2|boltz2_chemistry)|interim-\d{4}|combined-report|cpu|after-cpu)', name), 'Unsafe analysis parent')
    require(phase in ('cpu', 'certify-cpu', 'certify-protenix-only', 'capture', 'interim', 'final', 'diagnostic', 'diagnostic-capture') and
            (phase not in ('diagnostic', 'diagnostic-capture') or arm in ('protenix', 'boltz2')), 'Undeclared analysis phase/arm')
    parent = ROOT / name; parent.mkdir(exist_ok=False)
    command = ['bsub', *resources(phase), '-J', 'mosaic-lrboost-analysis-' + name, '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if dependency: command += ['-w', dependency]
    command += ['bash', '-l', str(SOURCE / 'scripts/hpc/jq1_lrboost_analysis_stage.lsf'), str(SOURCE), str(ROOT), manifest, phase, arm, str(parent)]
    record = {'phase': phase, 'arm': arm, 'args': command, 'dependency': dependency, 'outcome': 'unknown', 'utc': datetime.now(timezone.utc).isoformat()}
    exclusive(parent / 'submission-started.json', record)
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=60); record.update(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        ids = re.findall(r'Job <(\d+)>', r.stdout)
        if r.returncode == 0 and len(ids) == 1: record.update(outcome='submitted', job_id=ids[0])
    except Exception as error: record['error'] = repr(error)
    finally: exclusive(parent / 'submission.json', record)
    require(record['outcome'] == 'submitted', 'Submission failed/unknown; inspect exact markers, never duplicate')
    r = subprocess.run(['bjobs', '-l', record['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(r.stdout + r.stderr)
    return record


def validate_submission(parent, phase, arm, manifest):
    path = Path(parent) / 'submission.json'
    for attempt in range(16):
        if path.exists(): break
        require(attempt < 15, 'Executing job has no completed actual submission receipt')
        time.sleep(2)
    sub = read(path); job = os.environ.get('LSB_JOBID')
    expected = ['bsub', *resources(phase), '-J', 'mosaic-lrboost-analysis-' + Path(parent).name,
                '-oo', str(Path(parent) / 'stdout.%J.log'), '-eo', str(Path(parent) / 'stderr.%J.log')]
    if sub.get('dependency'): expected += ['-w', sub['dependency']]
    expected += ['bash', '-l', str(SOURCE / 'scripts/hpc/jq1_lrboost_analysis_stage.lsf'), str(SOURCE), str(ROOT), manifest, phase, arm, str(parent)]
    require(sub.get('outcome') == 'submitted' and sub.get('job_id') == job and sub.get('phase') == phase
            and sub.get('arm') == arm and sub.get('args') == expected, 'Executing job differs from actual submitted identity/resources/phase')
    return sub


def certify_cpu(manifest, *, only_protenix=False):
    from campaign25_pocket_ops import final_job
    jobs = read(ROOT / 'jobs.json'); job = jobs['cpu']
    status, queue, proofs = final_job(job, ROOT / 'cpu')
    require(status == 'DONE' and queue == 'hpc', 'Analysis CPU final outcome failed')
    cases = xml_cases(ROOT / 'cpu/payload/report-tests.xml')
    raw_log = (ROOT / 'cpu/payload/report-tests.log').read_text(encoding='utf-8')
    collected = re.findall(r'collected\s+(\d+)\s+items?', raw_log)
    require(collected == [str(len(cases))] and re.search(r'\b' + str(len(cases)) + r'\s+passed\b', raw_log)
            and not re.search(r'\d+\s+(?:deselected|skipped|failed|errors?)\b', raw_log), 'CPU log collection/result differs from all raw XML rows')
    for rel in ('payload/report-tests.xml', 'payload/report-tests.log', 'payload/test-invocation.json'):
        q = ROOT / 'cpu' / rel; proofs.append({'path': str(q), 'sha256': sha(q)})
    by_module = {Path(n).stem: sum(Path(n).stem in q.get('classname', '').split('.') for q in cases) for n in TESTS + LEGACY_TESTS}
    c = {'status': 'passed', 'producer_job_id': job, 'job_id': job, 'tests': len(cases), 'tests_by_module': by_module, 'suite': SUITE,
         'source_manifest_sha256': manifest, 'proofs': proofs}
    exclusive(ROOT / 'cpu/payload/assessment.json', c); exclusive(ROOT / 'cpu-certificate.json', c); verify_cpu(manifest)
    if only_protenix:
        with (ROOT / 'dispatch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            exclusive(ROOT / 'protenix-only-dispatch-started.json', {'manifest': manifest, 'original_failed_diagnostic': '29780821'})
            check_storage(ROOT / 'after-cpu', 'dispatch')
            native = submit('diagnostic-protenix', 'diagnostic', manifest, arm='protenix')
            callback = submit('diagnostic-capture-protenix', 'diagnostic-capture', manifest, arm='protenix', dependency='ended(' + native['job_id'] + ')')
            jobs.update(diagnostics={'protenix': native['job_id']}, diagnostic_captures={'protenix': callback['job_id']}, no_original_study_submitted=True)
            atomic(ROOT / 'jobs.json', jobs)
        return jobs
    with (ROOT / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exclusive(ROOT / 'analysis-dispatch-started.json', {'manifest': manifest, 'cpu_certificate_sha256': sha(ROOT / 'cpu-certificate.json')})
        check_storage(ROOT / 'after-cpu', 'dispatch')
        diagnostics, captures, collectors = {}, {}, {}
        for name, phase, arm, dependency in submission_plan():
            job = submit(name, phase, manifest, arm=arm, dependency=dependency)['job_id']
            if phase == 'diagnostic':
                diagnostics[arm] = job
                captures[arm] = submit('diagnostic-capture-' + arm, 'diagnostic-capture', manifest, arm=arm, dependency='ended(' + job + ')')['job_id']
            else: collectors[arm] = job
            jobs.update(diagnostics=diagnostics, diagnostic_captures=captures, collectors=collectors); atomic(ROOT / 'jobs.json', jobs)
        jobs['active_interim'] = submit('interim-0001', 'interim', manifest)['job_id']; jobs['interim_parent'] = 'interim-0001'; atomic(ROOT / 'jobs.json', jobs)
        dependency = ' && '.join('done(' + j + ')' for j in [*collectors.values(), *captures.values()])
        jobs['final'] = submit('combined-report', 'final', manifest, dependency=dependency)['job_id']
        jobs['cpu_certificate_job'] = os.environ['LSB_JOBID']; atomic(ROOT / 'jobs.json', jobs)
    return jobs


def bundle_cpu(run, manifest):
    c = verify_cpu(manifest); dest = run / 'cpu-proof'; dest.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / 'cpu-certificate.json', dest / 'assessment.json'); links = []
    for proof in c['proofs']:
        q = Path(proof['path']); target = dest / 'evidence' / q.relative_to(ROOT / 'cpu'); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(q, target)
        links.append({'original_path': str(q), 'local_path': target.relative_to(run).as_posix(), 'sha256': proof['sha256']})
    for name in ('report-tests.xml', 'report-tests.log'): shutil.copyfile(ROOT / 'cpu/payload' / name, dest / name)
    exclusive(dest / 'bundled-proofs.json', links); shutil.copyfile(SOURCE / 'source_manifest.json', run / 'report-source-manifest.json')


def diagnostic_capture(parent, arm, manifest):
    from campaign25_pocket_ops import final_job
    run = parent / 'payload'; run.mkdir(exist_ok=False)
    original = ROOT / ('diagnostic-' + arm); sub = read(original / 'submission.json'); job = sub['job_id']
    status, queue, proofs = final_job(job, original)
    require(status == 'DONE' and queue == 'gpul40s', 'Separate diagnostic execution failed')
    report = original / 'payload/assessment.json'; raw = read(report)
    require(sub['phase'] == 'diagnostic' and sub['arm'] == arm and sub['outcome'] == 'submitted', 'Wrong diagnostic submission')
    require(raw['version'] == 'jq1-lrboost-endpoint-diagnostics-v2'
            and raw['selected_saved_steps'] == [0, 9900, 10000]
            and raw['persistent_output_allowance_bytes'] == DIAGNOSTIC_BYTES
            and raw['status'] == 'diagnostics_completed_original_failure_preserved' and raw['diagnostic_job'] == job
            and raw['original_job'] == ARMS[arm]['job'] and raw['diagnostic_manifest_sha256'] == manifest
            and raw['scientific_manifest_sha256'] == SCIENCE_MANIFEST and raw['can_certify_or_restart_original_study'] is False
            and raw['frozen_before'] == raw['frozen_after'] == {'model_digest': ARMS[arm]['model'], 'context_digest': ARMS[arm]['context']}, 'Diagnostic identity/completion/frozen scope differs')
    actual_scheduler(next(Path(p['path']).read_text() for p in proofs if Path(p['path']).name.startswith('scheduler-final-')), job, 'gpul40s')
    actual_lsf((original / ('stdout.' + job + '.log')).read_text(), job)
    inventory = []
    study = Path(ARMS[arm]['output_parent']) / 'payload/study'
    for proof in raw['input_snapshots']:
        q = Path(proof['path']); require(q.resolve().is_relative_to(study / 'case-0'), 'Diagnostic original input escaped study')
        require(not q.is_symlink() and q.stat().st_size == proof['bytes'] and sha(q) == proof['sha256'], 'Original diagnostic input changed')
        inventory.append(proof)
    require([p['step'] for p in raw['points']] == [0, 9900, 10000], 'Missing diagnostic point')
    for point in raw['points']:
        require(point['proofs'], 'Diagnostic point has no raw numerical evidence')
        paths = [p['path'] for p in point['proofs']]
        require(len(set(paths)) == len(paths), 'Duplicate diagnostic proof paths')
        for proof in point['proofs']:
            q = Path(proof['path']); require(q.resolve().is_relative_to(original / 'payload') and not q.is_symlink(), 'Diagnostic raw evidence escaped fresh output')
            require(q.stat().st_size == proof['bytes'] and sha(q) == proof['sha256'], 'Diagnostic NPZ bytes changed')
            inventory.append(proof)
    require(sum(q.stat().st_size for q in (original / 'payload').rglob('*') if q.is_file()) <= DIAGNOSTIC_BYTES, 'Diagnostic output exceeded declared hard allowance')
    target = run / 'endpoint-diagnostics.json'; shutil.copyfile(report, target)
    refs = [*proofs, *inventory, {'path': str(report), 'sha256': sha(report)}, {'path': str(original / 'submission.json'), 'sha256': sha(original / 'submission.json')}]
    for name in ('summary.json', 'terminal_gradient_checks/report.json'):
        q = Path(ARMS[arm]['output_parent']) / 'payload/study' / name
        refs.append({'path': str(q), 'sha256': sha(q)})
    exclusive(run / 'assessment.json', {'status': 'passed', 'arm': arm, 'native_job': ARMS[arm]['job'], 'job_id': job,
        'capture_job': os.environ['LSB_JOBID'], 'report_source_manifest_sha256': manifest,
        'cpu_certificate_sha256': sha(ROOT / 'cpu-certificate.json'), 'scientific_passed': False,
        'diagnostic_report': {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target)}, 'proofs': refs})
    archive(original)


def run_diagnostic(command):
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=5520)
        require(code == 0, 'Separate endpoint diagnostic process failed')
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=20)
        raise


def archive(parent):
    subprocess.run([PYTHON, str(SCIENCE / 'scripts/collect_protenix_evidence.py'), '--storage', 'work3', '--run', str(parent), '--output', str(parent) + '.tar.gz'], check=True)


def produce(parent, phase, manifest, arm):
    verify_cpu(manifest); check_storage(parent, phase); run = parent / 'payload'
    if phase == 'diagnostic':
        run_diagnostic([PYTHON, str(SOURCE / 'scripts/lrboost_endpoint_diagnostics.py'), '--arm', arm, '--output', str(run), '--manifest', manifest,
                        '--nearby-step', '9900', '--max-output-bytes', str(DIAGNOSTIC_BYTES), '--max-seconds', '5400'])
        return
    if phase == 'diagnostic-capture': diagnostic_capture(parent, arm, manifest); return
    if phase == 'capture':
        subprocess.run([PYTHON, str(SOURCE / 'scripts/capture_jq1_lrboost_completion.py'), '--arm', arm, '--output', str(run), '--completions', str(ROOT),
                        '--report-manifest', manifest, '--cpu-certificate-sha256', sha(ROOT / 'cpu-certificate.json')], check=True)
        return
    run.mkdir(exist_ok=False)
    if phase == 'final':
        for a, pin in ARMS.items():
            q = read(ROOT / ('complete-' + a) / 'payload/assessment.json')
            require(q['status'] == 'passed' and q['native_job'] == pin['job'] and q['all_probability_frames_verified'] is True
                    and q['report_source_manifest_sha256'] == manifest and q['cpu_certificate_sha256'] == sha(ROOT / 'cpu-certificate.json'), 'Final presentation evidence not verified')
        command = [PYTHON, str(SOURCE / 'scripts/render_jq1_lrboost_report.py'), '--output', str(run / 'report'), '--completions', str(ROOT), '--diagnostics', str(ROOT)]
    else:
        command = [PYTHON, str(SOURCE / 'scripts/render_jq1_lrboost_progress.py'), '--output', str(run / 'report'), '--proofs', str(run / 'native-prefix-proofs'), '--completions', str(ROOT), '--diagnostics', str(ROOT)]
    with (run / 'render.log').open('x') as log: subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = read(run / 'report/report_receipt.json')
    require(receipt['input_jobs'] == [p['job'] for p in ARMS.values()] and receipt['n_atoms'] == 25, 'Report scope changed')
    require(all((n == 0 and step is None) or n == step + 1 for n, step in zip(receipt['states'], receipt['included_updates'])), 'Saved update/frame correspondence differs')
    if phase == 'final': require(receipt['presentation_complete'] and receipt['all_terminal'], 'Final presentation is not terminal')
    require(receipt['scientific_passed'] is False and receipt['campaign_status'] == 'failed', 'Original known failures must remain visible')
    bundle_cpu(run, manifest)
    for a in ARMS:
        q = ROOT / ('complete-' + a) / 'payload/assessment.json'
        if q.exists():
            dest = run / ('completion-' + a); dest.mkdir(); shutil.copyfile(q, dest / 'assessment.json')
            for name in ('snapshots.json', 'native-stdout.txt'):
                if (q.parent / name).exists(): shutil.copyfile(q.parent / name, dest / name)
    exclusive(run / 'assessment.json', {'status': 'passed', 'job_id': os.environ['LSB_JOBID'], 'complete': receipt['complete'],
        'presentation_complete': receipt['presentation_complete'], 'scientific_passed': receipt['scientific_passed'], 'campaign_status': receipt['campaign_status'],
        'report_source_manifest_sha256': manifest, 'native_jobs': receipt['input_jobs'], 'input_jobs': receipt['input_jobs'], 'states': receipt['states'],
        'included_updates': receipt['included_updates'], 'n_atoms': 25, 'report_receipt_sha256': sha(run / 'report/report_receipt.json'),
        'cpu_certificate_sha256': sha(ROOT / 'cpu-certificate.json'), 'scope': 'Saved-only presentation; original scientific failures preserved; diagnostics separate'})
    archive(parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('cpu', 'certify-cpu', 'certify-protenix-only', 'capture', 'interim', 'final', 'preflight', 'diagnostic', 'diagnostic-capture'), required=True)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', required=True); parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--arm', default='none', choices=('none', *ARMS)); args = parser.parse_args()
    require(args.root.resolve() == ROOT and args.source.resolve() == SOURCE and args.parent.resolve().is_relative_to(ROOT), 'Wrong immutable analysis scope')
    if args.phase == 'preflight': print(json.dumps(check_storage(args.parent, 'preflight'))); return
    require(bool(os.environ.get('LSB_JOBID')), 'Diagnostics/validation/rendering require allocated HPC')
    validate_submission(args.parent, args.phase, args.arm, args.manifest)
    if args.phase == 'cpu': cpu_tests(args.parent)
    elif args.phase == 'certify-protenix-only': print(json.dumps(certify_cpu(args.manifest, only_protenix=True)))
    elif args.phase == 'certify-cpu': print(json.dumps(certify_cpu(args.manifest)))
    else: produce(args.parent, args.phase, args.manifest, args.arm)


if __name__ == '__main__': main()
