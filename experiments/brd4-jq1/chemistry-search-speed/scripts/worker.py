"""Allocated CPU gate then benchmark; stdlib until all seals/capacity pass."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = BASE / 'chemistry-search-speed-01'
SOURCE = BASE / 'chemistry-search-speed-release-01/source'
ORIGINAL = BASE / 'protenix-extended-release-03/source'
LAUNCH = BASE / 'nonchem-stride20-startup-release-03/source'
LIMIT = 512 * 1024**2


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(source, pin):
    manifest = source / 'source_manifest.json'
    assert sha(manifest) == pin
    doc = json.loads(manifest.read_text())
    files = doc['files']
    assert len(files) > 0
    for name, expected in files.items():
        p = source / name
        assert p.is_relative_to(source) and not p.is_symlink() and p.is_file()
        # All original release manifests use direct SHA values.
        assert sha(p) == expected, name
    return {'manifest_sha256': pin, 'files': len(files)}


def guard(parent, phase):
    # Reuse the already validated whole-campaign capacity formula unchanged,
    # then add this independent bounded512MiB experiment allowance.
    verify(LAUNCH, '704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab')
    sys.path.insert(0, str(LAUNCH))
    from selected_storage import storage_gate
    gate = storage_gate(parent, phase)
    w = gate['work3']
    available, required = w['available_bytes_conservative'], w['required_bytes']
    result = {'status': 'failed', 'whole_gate': gate, 'additional_allowance_bytes': LIMIT,
              'required_bytes': required + LIMIT, 'available_bytes': available,
              'reserve_bytes': 4 * 1024**3, 'file_reserve': 100000}
    try:
        assert available >= required + LIMIT, 'Separate speed benchmark does not fit full unchanged capacity'
        result['status'] = 'passed'
    finally:
        with (Path(parent) / ('speed-storage-' + phase + '.json')).open('x') as f:
            json.dump(result, f, indent=2)
    return result


def run(manifest):
    assert os.environ.get('LSB_JOBID') and os.environ.get('LSB_QUEUE') == 'hpc'
    assert int(os.environ['LSB_DJOB_NUMPROC']) == 8
    assert not sys.flags.optimize
    payload = ROOT / 'cpu/payload'
    payload.mkdir(exist_ok=False)
    result = {'status': 'failed', 'job_id': os.environ['LSB_JOBID'], 'queue': os.environ['LSB_QUEUE'],
              'source_manifest_sha256': manifest, 'saved_runs_changed': False}
    start = time.monotonic()
    try:
        result['source'] = verify(SOURCE, manifest)
        result['original_source'] = verify(ORIGINAL, 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562')
        result['storage'] = guard(payload, 'entry')
        plan = json.loads((ROOT / 'plan.json').read_text())
        for row in plan['inputs']:
            assert Path(row['path']).stat().st_size == row['bytes'] and sha(row['path']) == row['sha256']
        env = dict(os.environ, PYTHONPATH=str(ORIGINAL / 'src'), JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
        xml = payload / 'tests.xml'
        args = [sys.executable, '-m', 'pytest', '-q', '-o', 'addopts=', str(SOURCE / 'tests/test_speed.py'),
                '--junitxml=' + str(xml)]
        with (payload / 'tests.log').open('xb') as f:
            subprocess.run(args, env=env, cwd=SOURCE, stdout=f, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        cases = ET.parse(xml).getroot().findall('.//testcase')
        assert len(cases) == 79 and all(not list(c) for c in cases), 'Full allocated79 fixtures not clean'
        result['cpu_cases'] = len(cases)
        result['cpu_xml_sha256'] = sha(xml)
        with (payload / 'benchmark.log').open('xb') as f:
            subprocess.run([sys.executable, str(SOURCE / 'scripts/benchmark.py'), str(ROOT / 'plan.json'),
                            str(payload / 'benchmark')], env=env, cwd=SOURCE, stdout=f,
                           stderr=subprocess.STDOUT, check=True, timeout=4500)
        assessment = payload / 'benchmark/assessment.json'
        assert json.loads(assessment.read_text())['status'] == 'passed_exact_on_sampled_saved_inputs'
        result['benchmark_sha256'] = sha(assessment)
        result['original_source_after'] = verify(ORIGINAL, result['original_source']['manifest_sha256'])
        for row in plan['inputs']:
            assert sha(row['path']) == row['sha256']
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        result['output_bytes'] = size
        if size >= LIMIT:
            result['status'] = 'failed'
            result['cap_exceeded'] = True
        (payload / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
        assert size < LIMIT, 'Persistent output cap exceeded'


if __name__ == '__main__':
    run(sys.argv[1])
