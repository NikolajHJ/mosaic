"""Independent stdlib byte/XML/schema audit; scientific comparisons ran on HPC."""
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import tarfile
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO / 'results/hpc/20261025-chemistry-speed-02'
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/'
ROOT = BASE + 'chemistry-search-speed-02'
SOURCE = BASE + 'chemistry-search-speed-release-02/source'
PIN = '09c0ab837bede0c5779eac0da06885cfbef6a98744f67af19e7b5a8b33ffa222'
ARCHIVE = 'e079f5a30c4e062b2d8dee04311824425a593d16278c3c378d592dd46d127432'
COUNTS = {'tests.test_speed': 79, 'tests.test_hybrid': 26,
          'tests.test_gpu_matrix': 20, 'tests.test_oracle_profile': 18}
OLD_FILES = REPO / 'results/hpc/20261025-chemistry-speed-01/observation-13/files'
_BOUND_ROOT = None
_BOUND_NAMES = set()


def relative_name(name):
    """Accept only portable Linux receipt names, including on Windows hosts."""
    assert isinstance(name, str) and name and '\\' not in name and ':' not in name
    rel = PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() == name
    return rel


def bound_check(path):
    """Every file read from the delivered tree must belong to its hash listing."""
    path = Path(path).absolute()
    if _BOUND_ROOT is not None and path.is_relative_to(_BOUND_ROOT):
        rel = path.relative_to(_BOUND_ROOT)
        assert rel.as_posix() in _BOUND_NAMES, ('Unlisted delivered artifact', rel)
        assert all(not parent.is_symlink() for parent in (path, *path.parents)
                   if parent.is_relative_to(_BOUND_ROOT))
        assert path.resolve().is_relative_to(_BOUND_ROOT)


def sha(p):
    bound_check(p)
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    bound_check(p)
    return json.loads(Path(p).read_bytes())


def compact(text):
    return ''.join(text.split())


def positive(value):
    assert type(value) in (int, float) and math.isfinite(value) and value > 0
    return value


def finite(value):
    assert type(value) in (int, float) and math.isfinite(value)
    return value


def nonnegative_integer(value):
    assert type(value) is int and value >= 0
    return value


def loss_scalars(value):
    finite(value['value']); finite(value['primal'])
    assert isinstance(value['components'], dict) and value['components']
    assert all(isinstance(k, str) for k in value['components'])
    for number in value['components'].values():
        finite(number)
    assert len(value['gradient_sha256']) == 2
    assert all(isinstance(h, str) and re.fullmatch('[a-f0-9]{64}', h)
               for h in value['gradient_sha256'])


def frozen_reference(row):
    label = row['label']
    assert relative_name(label).name == label
    path = OLD_FILES / 'cpu/payload/benchmark' / (label + '-baseline.json')
    assert path.stat().st_size == row['frozen_reference']['bytes']
    assert sha(path) == row['frozen_reference']['sha256']
    return read(path)


def quota_check(raw):
    """Independently reproduce only the sealed finite work3 metadata policy."""
    pools = []
    units = ('Byte', 'KiB', 'MiB', 'GiB', 'TiB')
    for line in raw.splitlines():
        fields = [x.strip() for x in re.split(r'\|+', line) if x.strip()]
        if not fields or fields[0] != 's234818':
            continue
        assert len(fields) == 6 and fields[1] == '306121'
        if fields[3] == 'unlimited':
            continue
        rounded = []
        for text, rounding in ((fields[2], ROUND_CEILING), (fields[3], ROUND_FLOOR)):
            match = re.fullmatch(r'(\d+(?:\.\d+)?)\s+(Byte|KiB|MiB|GiB|TiB)', text)
            assert match
            gib = Decimal(match[1]) * 1024 ** units.index(match[2]) / 1024**3
            rounded.append(int(gib.to_integral_value(rounding=rounding)) * 1024**3)
        used, hard = rounded
        count, limit = int(fields[4]), int(fields[5])
        assert 0 <= used < hard and 0 <= count < limit
        pools.append({'used_bytes_upper': used, 'hard_bytes_lower': hard,
                      'remaining_bytes': hard - used, 'remaining_files': limit - count})
    assert pools, 'Finite user-quota evidence is mandatory'
    return {'finite_pools': pools,
            'available_bytes_lower': min(p['remaining_bytes'] for p in pools),
            'available_files_lower': min(p['remaining_files'] for p in pools)}


def archive_check(path):
    bound_check(path)
    raw = Path(path).read_bytes()
    assert len(raw) == 35875 and hashlib.sha256(raw).hexdigest() == ARCHIVE
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        members = t.getmembers()
        assert len(members) == len({m.name for m in members}) == 20
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute()
                   and '..' not in PurePosixPath(m.name).parts for m in members)
        manifest_raw = t.extractfile('source_manifest.json').read()
        assert hashlib.sha256(manifest_raw).hexdigest() == PIN
        manifest = json.loads(manifest_raw)
        assert set(manifest['files']) == {m.name for m in members} - {'source_manifest.json'}
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(t.extractfile(name).read()).hexdigest() == digest
        assert len(manifest['files']) == 19 and manifest['cpu_cases'] == 143 and manifest['gpu_cases'] == 19
        assert manifest['scientific_production_changed'] is False
    return manifest


def xml_check(path, expected):
    root = ET.parse(path).getroot()
    cases = root.findall('.//testcase')
    assert len(cases) == len({(c.get('classname'), c.get('name')) for c in cases})
    assert all(not list(c) for c in cases)
    counts = {}
    for case in cases:
        counts[case.get('classname')] = counts.get(case.get('classname'), 0) + 1
    assert counts == expected, (counts, expected)
    for suite in root.iter('testsuite'):
        assert all(int(suite.get(k, '0')) == 0 for k in ('errors', 'failures', 'skipped'))
        assert int(suite.get('tests')) == len(suite.findall('testcase'))
    return counts


def storage_check(g):
    assert g['status'] == 'passed' and g['reserve_bytes'] == 4 * 1024**3
    assert g['file_reserve'] == 100000 and g['additional_allowance_bytes'] == 1024**3
    whole = g['whole_original_gate']; w = whole['work3']
    assert whole['status'] == w['status'] == 'passed'
    assert whole['matrix_stride'] == 20 and whole['loss_stride'] == 1
    assert whole['full_selected_production_admission'] is True
    assert w['filesystem'] == 'work3' and w['reserve_bytes'] == 4 * 1024**3
    assert w['file_reserve'] == 100000
    assert w['quota'] == quota_check(w['raw_quota'])
    for key in ('required_bytes', 'projected_bytes', 'disk_free_bytes', 'stale_bytes_subtracted'):
        nonnegative_integer(w[key])
    assert g['required_bytes'] == w['required_bytes'] + 1024**3
    assert g['available_bytes'] == w['available_bytes_conservative'] >= g['required_bytes']
    assert w['required_bytes'] == w['projected_bytes'] + 4 * 1024**3
    assert w['available_bytes_conservative'] == min(
        w['disk_free_bytes'], w['quota']['available_bytes_lower']) - w['stale_bytes_subtracted']
    assert all(p['remaining_files'] >= 100000 for p in w['quota']['finite_pools'])
    assert w['available_files_lower'] >= 100000
    observed = w['filesystem_inode_observation']
    assert set(observed) == {'f_files', 'f_ffree', 'f_favail'}
    for number in observed.values():
        nonnegative_integer(number)
    total, free, available = (observed[k] for k in ('f_files', 'f_ffree', 'f_favail'))
    if (total, free, available) == (0, 0, 0):
        assert w['filesystem_inode_reporting'] == 'unreported_all_zero'
        assert w['available_files_lower'] == w['quota']['available_files_lower']
        assert 'no independent' in w['inode_limitation']
    else:
        assert total > 0 and 0 <= available <= free <= total
        assert w['filesystem_inode_reporting'] == 'reported_coherent'
        assert w['available_files_lower'] == min(available, w['quota']['available_files_lower'])


def command(phase, dependency=None):
    if phase == 'cpu':
        resource = ['-q', 'hpc', '-n', '16', '-R', 'span[hosts=1]', '-R', 'rusage[mem=2GB]', '-W', '02:00']
    elif phase == 'after-cpu':
        resource = ['-q', 'hpc', '-n', '1', '-R', 'span[hosts=1]', '-R', 'rusage[mem=2GB]', '-W', '00:20']
    else:
        assert phase == 'gpu'
        resource = ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]',
                    '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00']
    argv = ['bsub', *resource, '-J', 'mosaic-chem-speed02-' + phase,
            '-oo', ROOT + '/' + phase + '/stdout.%J.log', '-eo', ROOT + '/' + phase + '/stderr.%J.log']
    if dependency:
        assert phase == 'after-cpu' and re.fullmatch(r'ended\(\d+\)', dependency)
        argv += ['-w', dependency]
    return argv + ['bash', '-l', SOURCE + '/scripts/round2_stage.lsf', SOURCE, PIN, phase]


def submission(files, phase, job):
    s = read(files / phase / 'submission.json')
    assert s['job_id'] == job and s['phase'] == phase and s['status'] == 'submitted'
    assert s['manifest_sha256'] == PIN and s['returncode'] == 0
    assert s['argv'] == command(phase, s['dependency'])
    assert s['stdout'].count(f'Job <{job}> is submitted') == 1
    return s


def same_job_terminal(files, observation, phase, job):
    queue = 'gpul40s' if phase == 'gpu' else 'hpc'
    q = observation['scheduler'][phase]
    assert q['job_id'] == job
    scheduler = compact(q['stdout'])
    # A callback may have retained the actual terminal scheduler before live records expire.
    if f'Job<{job}>' not in scheduler:
        assert phase == 'cpu'
        persisted = read(files / 'after-cpu/cpu-final-scheduler.json')
        assert persisted['returncode'] == 0
        scheduler = compact(persisted['stdout'])
    else:
        assert q['returncode'] == 0
    assert f'Job<{job}>' in scheduler and 'Status<DONE>' in scheduler and f'Queue<{queue}>' in scheduler
    log = files / phase / f'stdout.{job}.log'
    text = compact(log.read_text())
    assert text.count(f'Subject:Job{job}:<mosaic-chem-speed02-{phase}>incluster<dcc>Done') == 1
    assert 'Successfullycompleted.' in text and f'inqueue<{queue}>' in text
    return {'scheduler_sha256': hashlib.sha256(scheduler.encode()).hexdigest(), 'raw_lsf_sha256': sha(log)}


def cpu_rows(result, plan):
    assert result['status'] == 'passed_exact_hybrid_saved_input_comparison'
    assert result['original_runs_changed'] is False
    labels = [r['label'] for r in plan['inputs']]
    modes = ['baseline', 'combined', 'hybrid4', 'hybrid8']
    assert len(result['rows']) == 24 and len(set(labels)) == len(labels) == 6
    summaries = []
    for row in plan['inputs']:
        label = row['label']; group = [r for r in result['rows'] if r['input']['label'] == label]
        assert len(group) == 4 and {r['mode'] for r in group} == set(modes)
        by_mode = {r['mode']: r for r in group}; baseline = by_mode['baseline']
        assert baseline['preparation_identity'] == frozen_reference(row)['identity']
        for mode, r in by_mode.items():
            assert r['input'] == row and r['exact_targets_receipts_loss_gradients'] is True
            for key in ['preparation_identity', 'loss', 'original_checks', 'additional_checks']:
                assert r[key] == baseline[key], (label, mode, key)
            assert len(r['seconds']) == 3
            for t in r['seconds']:
                positive(t)
            assert r['first_encounter_seconds'] == r['seconds'][0]
            assert r['warm_median_seconds'] == statistics.median(r['seconds'][1:])
            assert r['loss']['value'] == r['loss']['primal']
            loss_scalars(r['loss'])
        summaries.append({'input': label, 'first_encounter_seconds': {m: by_mode[m]['seconds'][0] for m in modes},
                          'warm_median_seconds': {m: by_mode[m]['warm_median_seconds'] for m in modes},
                          'warm_speedup': {m: baseline['warm_median_seconds'] / by_mode[m]['warm_median_seconds'] for m in modes}})
    assert [r['mode'] for r in result['oracle_profiles']] == ['baseline', 'combined']
    return summaries


def gpu_rows(result, plan):
    assert result['status'] == 'passed_separate_gpu_matrix_benchmark' and result['saved_runs_changed'] is False
    assert any(d['platform'] == 'gpu' and 'L40S' in d['kind'] for d in result['devices'])
    labels = {r['label'] for r in plan['inputs']}
    assert len(result['feature_checks']) == 6
    assert {r['input'] for r in result['feature_checks']} == labels
    for r in result['feature_checks']:
        assert r['status'] == 'passed_exact_integer_cpu_gpu_features' and r['native_oracle_replaced'] is False
        assert 'valid' not in r['keys']
    checks = result['loss_checks']
    reference = {r['label']: frozen_reference(r)['identity'] for r in plan['inputs']}
    assert len(checks) == len({(r['input'], r['method']) for r in checks}) == 12
    assert {(r['input'], r['method']) for r in checks} == {(k, m) for k in labels for m in ['scan', 'matrix']}
    for r in checks:
        assert r['status'] == 'passed' and r['tolerance'] == 2e-5
        assert r['original_failure_or_readiness_status_changed'] is False
        assert r['preparation_identity'] == reference[r['input']]
        loss_scalars(r['original']); loss_scalars(r['observed'])
        assert r['original']['components'].keys() == r['observed']['components'].keys()
        for key in ('value', 'primal'):
            left, right = r['observed'][key], r['original'][key]
            assert abs(left - right) <= 2e-5 + 2e-5 * abs(right)
        for key, right in r['original']['components'].items():
            assert abs(r['observed']['components'][key] - right) <= 2e-5 + 2e-5 * abs(right)
        assert set(r['max_abs_gradient_error']) == {'atoms', 'bonds'}
        assert all(finite(v) >= 0 for v in r['max_abs_gradient_error'].values())
        assert {v['field'] for v in r['finite_differences']} == {'atoms', 'bonds'}
        for d in r['finite_differences']:
            if d['status'] == 'not_applicable_exact_zero_original_gradient':
                # The actual exact-zero array checks ran on allocated GPU.
                # Hashes cannot independently reveal zero values or zero signs.
                continue
            assert d['status'] == 'passed'
            probes = d['probes']; assert [p['epsilon'] for p in probes] == [.01, .003, .001]
            shape = (25, 6) if d['field'] == 'atoms' else (300, 5)
            assert len(d['coordinate']) == 2
            assert all(type(v) is int and 0 <= v < n for v, n in zip(d['coordinate'], shape))
            for p in probes:
                # JSON scalar-integrity identities only. Both plus/minus loss
                # evaluations and AD/gradient comparisons ran on allocated GPU.
                ad = finite(p['analytic']); epsilon = positive(p['epsilon'])
                assert abs(ad) > 1e-8
                minus, plus = finite(p['primal_minus']), finite(p['primal_plus'])
                calculated_fd = (plus - minus) / (2 * epsilon)
                assert finite(p['finite_difference']) == calculated_fd
                calculated_error = abs(ad - calculated_fd) / max(abs(ad), abs(calculated_fd), 1e-12)
                assert finite(p['relative_error']) == calculated_error
                assert type(p['passed']) is bool
            assert all(p['passed'] == (p['relative_error'] <= .15) for p in probes)
            assert any(probes[k]['passed'] and probes[k + 1]['passed'] for k in range(2))
    assert len(result['timings']) == 12
    ids = []
    for r in result['timings']:
        assert r['batch'] in [1, 32, 256] and len(r['warm_seconds']) == 20
        positive(r['first_compile_and_call_seconds'])
        for t in r['warm_seconds']:
            positive(t)
        assert r['warm_median_seconds'] == statistics.median(r['warm_seconds'])
        if r['component'] == 'integer_matrix_features':
            positive(r['transfer_seconds']); ids.append((r['component'], r['batch']))
        else:
            assert r['component'] == 'local_risk_value_and_gradient' and r['method'] in ['original', 'scan', 'matrix']
            assert r['tolerance'] == 2e-5; ids.append((r['component'], r['batch'], r['method']))
    assert len(set(ids)) == 12
    return result['timings']


def main():
    global _BOUND_ROOT, _BOUND_NAMES
    assert not sys.flags.optimize
    observation = Path(sys.argv[1]).resolve(); data = read(observation / 'raw.json'); files = observation / 'files'
    assert files.is_dir() and not files.is_symlink()
    _BOUND_ROOT, _BOUND_NAMES = files.absolute(), set(data['files'])
    for name, row in data['files'].items():
        relative_name(name)
        p = files / name
        assert not p.is_symlink() and p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    manifest = archive_check(files / 'source.tar.gz')
    plan = read(files / 'plan.json'); initial = read(files / 'initial-submissions.json')
    assert plan['version'] == 'chemistry-hybrid-gpu-matrix-v1'
    assert plan['cpu_counts'] == COUNTS and plan['gpu_counts'] == {'tests.test_gpu_execution': 19}
    assert plan['workers'] == [4, 8] and plan['matrix_batches'] == [1, 32, 256]
    assert plan['value_gradient_tolerance'] == 2e-5 and plan['fd_epsilons'] == [.01, .003, .001]
    assert plan['fd_relative_error'] == .15 and plan['fd_required_adjacent'] == 2
    assert plan['persistent_cap_bytes'] == 1024**3 and plan['original_sources_results_jobs_changed'] is False
    assert plan['monitor_remains_paused'] is True
    assert sha(files / 'plan.json') == initial['plan_sha256']
    assert initial['source_manifest_sha256'] == PIN and initial['inputs'] == plan['inputs']
    assert initial['archive_sha256'] == ARCHIVE and initial['archive_bytes'] == 35875
    assert plan['search'] == {'max_edits': 3, 'checks_per_depth': 32, 'beam_width': 4,
                              'probability_weight': 1.0, 'probability_floor': 1e-12}
    assert plan['progress_search'] == {'version': 'repair-progress-v1', 'max_edits': 6, 'checks_per_depth': 32,
                                       'beam_width': 4, 'probability_weight': 1.0, 'allow_atom_edits': False,
                                       'atom_edit_penalty': 3.0, 'max_atom_edits': 1}
    for row in plan['inputs']:
        assert relative_name(row['label']).name == row['label']
        p = OLD_FILES / 'input-bytes' / (row['label'] + '.npz')
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
        frozen_reference(row)
    storage_check(initial['storage'])
    for phase, job in data['jobs'].items():
        submission(files, phase, job)
    assert data['jobs']['cpu'] == '29809477' and data['jobs']['after-cpu'] == '29809478'
    result = {'status': 'passed_available_byte_xml_schema_evidence_only', 'source_manifest_sha256': PIN,
              'source_files': len(manifest['files']), 'source_archive_sha256': ARCHIVE,
              'observed_utc': data['captured_utc'], 'raw_observation_sha256': sha(observation / 'raw.json'),
              'verifier_source_sha256': sha(__file__), 'verified_files': len(data['files']), 'phases': {},
              'scope': 'Independent metadata audit; all numerical checks ran on allocated HPC',
              'original_runs_changed': False, 'production_adoption': False}
    for phase in ['cpu', 'gpu']:
        path = files / phase / 'payload/assessment.json'
        if not path.exists():
            continue
        a = read(path); assert a['status'] == 'passed'
        job = data['jobs'][phase]; terminal = same_job_terminal(files, data, phase, job)
        assert a['job_id'] == job and a['phase'] == phase and a['source_manifest_sha256'] == PIN
        nonnegative_integer(a['output_bytes'])
        assert a['original_results_changed'] is False and a['output_bytes'] < 1024**3
        assert a['original_source_after'] == {'manifest_sha256': 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562', 'files': 643}
        counts = COUNTS if phase == 'cpu' else {'tests.test_gpu_execution': 19}
        xml = files / phase / 'payload/tests.xml'; assert sha(xml) == a['raw_xml_sha256']
        assert xml_check(xml, counts) == a['cpu_counts' if phase == 'cpu' else 'gpu_counts']
        storage_check(a['storage'])
        bench = files / phase / 'payload/benchmark/assessment.json'; assert sha(bench) == a['benchmark_sha256']
        details = cpu_rows(read(bench), plan) if phase == 'cpu' else gpu_rows(read(bench), plan)
        result['phases'][phase] = {'job_id': job, 'status': 'passed_allocated_checks_and_benchmark',
                                   'raw_cases': sum(counts.values()), 'assessment_sha256': sha(path),
                                   'raw_xml_sha256': sha(xml), 'benchmark_sha256': sha(bench), **terminal,
                                   'details': details}
    gpu_complete = files / 'after-cpu/gpu-submission-complete.json'
    if gpu_complete.exists():
        d = read(gpu_complete); assert d['status'] == 'submitted_not_passed'
        assert d['gpu'] == read(files / 'gpu/submission.json')
        assert d['cpu_terminal']['job_id'] == data['jobs']['cpu']
        assert d['cpu_terminal']['assessment_sha256'] == result['phases']['cpu']['assessment_sha256']
        assert d['cpu_terminal']['lsf_sha256'] == result['phases']['cpu']['raw_lsf_sha256']
        storage_check(d['storage'])
        result['gpu_submission'] = {'job_id': d['gpu']['job_id'], 'status': 'submitted_not_an_execution_pass'}
    assert result['phases'], 'No terminal passing stage available for this audit'
    target = observation / (sys.argv[2] if len(sys.argv) > 2 else 'verification.json')
    assert target.parent == observation and relative_name(target.name).name == target.name
    with target.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
