"""New immutable metadata-only adoption after source02 callback scheduler lag."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = BASE / 'chemistry-search-speed-gpu-adoption-03'
SOURCE = BASE / 'chemistry-search-speed-gpu-adoption-release-03/source'
PARENT = BASE / 'chemistry-search-speed-02'
PARENT_SOURCE = BASE / 'chemistry-search-speed-release-02/source'
PARENT_PIN = '09c0ab837bede0c5779eac0da06885cfbef6a98744f67af19e7b5a8b33ffa222'
CPU_JOB = '29809477'
CALLBACK_JOB = '29809478'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
PLAN_PIN = '672dfbb1380c26212cca7ccb1033cf329b8f426a76d0d60ad1335f2f9885e334'
CPU_ASSESSMENT = '51fccda60b2178d39fb9ae90067d1ea9062c05d0560ec494dae41272ea33ec34'
CPU_XML = 'dcc176b0a4a74aeedb78a61454bacf6d2d2b1cfc6d8024d2e726f23f99c10791'
BENCHMARK = '68b2ef335a15958fc1f2bebaa0411408e2a12eb055637398c7bcde3f26e6b4bf'
CPU_LSF = 'e95e148296b17e523441525eeb507b5a7d37f622b5a1a036c85b5b243c805398'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def load(p):
    return json.loads(Path(p).read_bytes())


def manifest(pin):
    path = SOURCE / 'source_manifest.json'; assert sha(path) == pin
    m = load(path)
    for name, digest in m['files'].items():
        p = SOURCE / name
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert p.is_file() and not p.is_symlink() and sha(p) == digest
    return m


def expected_argv(pin, phase, dependency=None):
    assert phase in ['cpu', 'after-cpu']
    argv = ['bsub', '-q', 'hpc', '-n', '2' if phase == 'cpu' else '1', '-R', 'span[hosts=1]',
            '-R', 'rusage[mem=2GB]', '-W', '00:30' if phase == 'cpu' else '00:20',
            '-J', 'mosaic-chem-speed-gpu-adoption03-' + phase,
            '-oo', str(ROOT / phase / 'stdout.%J.log'), '-eo', str(ROOT / phase / 'stderr.%J.log')]
    if dependency:
        assert phase == 'after-cpu' and dependency.startswith('ended(') and dependency.endswith(')')
        assert dependency[6:-1].isdigit()
        argv += ['-w', dependency]
    return argv + ['bash', '-l', str(SOURCE / 'scripts/gpu_adoption_stage.lsf'), str(SOURCE), pin, phase]


def original_proofs():
    sys.path.insert(0, str(PARENT_SOURCE / 'scripts'))
    import round2_worker as parent
    parent.verify(PARENT_SOURCE, PARENT_PIN); parent.external()
    for proof in load(SOURCE / 'source_manifest.json')['preserved_callback_failure']:
        assert sha(proof['path']) == proof['sha256']
    assert sha(PARENT / 'plan.json') == PLAN_PIN == load(PARENT / 'initial-submissions.json')['plan_sha256']
    plan = load(PARENT / 'plan.json')
    assert plan['cpu_counts'] == parent.CPU_COUNTS and plan['gpu_counts'] == {'tests.test_gpu_execution': 19}
    assert plan['value_gradient_tolerance'] == 2e-5 and plan['fd_epsilons'] == [.01, .003, .001]
    assert plan['fd_relative_error'] == .15 and plan['fd_required_adjacent'] == 2
    for row in plan['inputs']:
        p = Path(row['path']); frozen = Path(row['frozen_reference']['path'])
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
        assert frozen.is_file() and not frozen.is_symlink() and frozen.stat().st_size == row['frozen_reference']['bytes']
        assert sha(frozen) == row['frozen_reference']['sha256']
    a = PARENT / 'cpu/payload/assessment.json'; xml = PARENT / 'cpu/payload/tests.xml'
    b = PARENT / 'cpu/payload/benchmark/assessment.json'; log = PARENT / 'cpu' / ('stdout.' + CPU_JOB + '.log')
    assert sha(a) == CPU_ASSESSMENT and sha(xml) == CPU_XML and sha(b) == BENCHMARK and sha(log) == CPU_LSF
    assessment = load(a)
    assert assessment['status'] == 'passed' and assessment['job_id'] == CPU_JOB
    assert assessment['source_manifest_sha256'] == PARENT_PIN and assessment['raw_xml_sha256'] == CPU_XML
    assert assessment['benchmark_sha256'] == BENCHMARK and assessment['original_results_changed'] is False
    assert parent.xml_cases(xml, parent.CPU_COUNTS) == parent.CPU_COUNTS
    assert load(b)['status'] == 'passed_exact_hybrid_saved_input_comparison'
    for phase, job in [('cpu', CPU_JOB), ('after-cpu', CALLBACK_JOB)]:
        s = load(PARENT / phase / 'submission.json')
        assert s['job_id'] == job and s['manifest_sha256'] == PARENT_PIN and s['status'] == 'submitted'
        assert s['argv'] == parent.argv(phase, PARENT_PIN, s['dependency'])
    # Preserve and authenticate the exact failed callback, not a relabelled pass.
    callback_log = (PARENT / 'after-cpu' / ('stdout.' + CALLBACK_JOB + '.log')).read_text()
    c = ''.join(callback_log.split())
    assert f'Subject:Job{CALLBACK_JOB}:<mosaic-chem-speed02-after-cpu>incluster<dcc>Exited' in c
    assert 'Exitedwithexitcode1.' in c
    failure = load(PARENT / 'after-cpu/cpu-final-scheduler.json')
    assert failure['returncode'] == 0 and f'Job<{CPU_JOB}>' in ''.join(failure['stdout'].split())
    assert 'Status<RUN>' in ''.join(failure['stdout'].split())
    jobs = load(PARENT / 'jobs.json')
    assert jobs == {'cpu': CPU_JOB, 'after-cpu': CALLBACK_JOB}
    assert not os.path.lexists(PARENT / 'gpu'), 'Existing or unknown GPU submission; never duplicate'
    return parent


def main(pin, phase):
    m = manifest(pin)
    sub_path = ROOT / phase / 'submission.json'
    for _ in range(20):
        if sub_path.is_file():
            break
        time.sleep(1)
    sub = load(sub_path)
    assert sub['status'] == 'submitted' and sub['job_id'] == os.environ['LSB_JOBID']
    assert sub['manifest_sha256'] == pin and sub['argv'] == expected_argv(pin, phase, sub['dependency'])
    assert os.environ['LSB_QUEUE'] == 'hpc' and int(os.environ['LSB_DJOB_NUMPROC']) == (2 if phase == 'cpu' else 1)
    payload = ROOT / phase / 'payload'; payload.mkdir(exist_ok=False)
    result = {'status': 'failed', 'job_id': sub['job_id'], 'manifest_sha256': pin,
              'scientific_source_changed': False, 'original_callback_status': 'failed_preserved', 'gpu_submitted': False}
    try:
        parent = original_proofs()
        from gpu_adoption_terminal import wait_terminal, lsf_success

        def terminal(job_id, prefix, job_name='mosaic-chem-speed02-cpu'):
            counter = []

            def query(requested):
                assert requested == job_id
                r = subprocess.run(['bjobs', '-l', job_id], capture_output=True, text=True, timeout=30)
                return {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr,
                        'captured_utc': datetime.now(timezone.utc).isoformat()}

            def record(row):
                counter.append(row)
                write(payload / (prefix + '-query-' + str(len(counter)).zfill(2) + '.json'), row)

            return wait_terminal(query, job_id, record=record, job_name=job_name)

        actual = terminal(CPU_JOB, 'original-cpu')
        log = PARENT / 'cpu' / ('stdout.' + CPU_JOB + '.log')
        lsf_success(log.read_text(), CPU_JOB)
        assert sha(log) == CPU_LSF
        if phase == 'cpu':
            command = [PYTHON, '-m', 'pytest', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
                       str(SOURCE / 'tests/test_gpu_adoption_terminal.py'), '--junitxml=' + str(payload / 'tests.xml')]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(SOURCE / 'scripts'),
                       CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (payload / 'tests.log').open('xb') as f:
                subprocess.run(command, cwd=SOURCE, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=180)
            cases = ET.parse(payload / 'tests.xml').getroot().findall('.//testcase')
            assert len(cases) == m['metadata_cases']
            assert all(not list(c) for c in cases)
            assert len({(c.get('classname'), c.get('name')) for c in cases}) == len(cases)
            assert {c.get('classname') for c in cases} == {'tests.test_gpu_adoption_terminal'}
            result.update(status='passed_metadata_checks_original_CPU_adopted', metadata_cases=len(cases),
                          raw_xml_sha256=sha(payload / 'tests.xml'), cpu_actual_terminal=actual,
                          original_cpu_assessment_sha256=CPU_ASSESSMENT, original_cpu_lsf_sha256=CPU_LSF,
                          original_cpu_xml_sha256=CPU_XML, original_benchmark_sha256=BENCHMARK,
                          plan_sha256=PLAN_PIN, parent_source_manifest_sha256=PARENT_PIN)
            return
        # A new own-ended callback requires actual terminal proof of the new CPU tests too.
        cpu_sub = load(ROOT / 'cpu/submission.json'); cpu_id = cpu_sub['job_id']
        assert sub['dependency'] == 'ended(' + cpu_id + ')'
        own_cpu_terminal = terminal(cpu_id, 'adoption-cpu', 'mosaic-chem-speed-gpu-adoption03-cpu')
        own_cpu_log = ROOT / 'cpu' / ('stdout.' + cpu_id + '.log')
        lsf_success(own_cpu_log.read_text(), cpu_id, 'mosaic-chem-speed-gpu-adoption03-cpu')
        cpu_assessment = ROOT / 'cpu/payload/assessment.json'; ca = load(cpu_assessment)
        assert ca['status'] == 'passed_metadata_checks_original_CPU_adopted' and ca['job_id'] == cpu_id
        assert ca['manifest_sha256'] == pin and ca['metadata_cases'] == m['metadata_cases']
        xml = ROOT / 'cpu/payload/tests.xml'
        assert sha(xml) == ca['raw_xml_sha256']
        assert parent.xml_cases(xml, {'tests.test_gpu_adoption_terminal': m['metadata_cases']})
        # Serialized exclusive dispatch retains all original source02 submission guards.
        with (ROOT / 'dispatch.lock').open('x') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gate = parent.storage(payload, 'gpu-dispatch-adoption03')
            original_proofs()
            current_bytes = sum(p.stat().st_size for root in [PARENT, ROOT, SOURCE]
                                for p in root.rglob('*') if p.is_file())
            assert current_bytes < 1024**3, 'Combined benchmark/adoption output cap'
            gpu = parent.submit('gpu', PARENT_PIN)
            result.update(status='metadata_checks_passed_GPU_submitted_not_passed', metadata_cases=ca['metadata_cases'],
                          raw_xml_sha256=sha(xml), cpu_actual_terminal=actual,
                          original_cpu_assessment_sha256=CPU_ASSESSMENT, original_cpu_lsf_sha256=CPU_LSF,
                          original_cpu_xml_sha256=CPU_XML, original_benchmark_sha256=BENCHMARK,
                          plan_sha256=PLAN_PIN, storage=gate, gpu_submitted=True, gpu=gpu,
                          combined_current_bytes=current_bytes,
                          adoption_cpu_terminal=own_cpu_terminal, adoption_cpu_lsf_sha256=sha(own_cpu_log),
                          adoption_cpu_assessment_sha256=sha(cpu_assessment),
                          parent_source_manifest_sha256=PARENT_PIN)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        write(payload / 'assessment.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
