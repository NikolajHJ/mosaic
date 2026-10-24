"""Independent original-noise validation/driver routing; no new science code."""
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import types

from selected_contract import (CONTROL, LAUNCHER, NOISE_MANIFEST, NOISE_ROOT,
    NOISE_SOURCE, NOISE_STUDY, PYTHON, exclusive, file_record, read, require, sha,
    verify_noise_source, verify_own, verify_xml)
from selected_storage import storage_gate


def adopt_cpu(manifest,parent):
    """Reuse exact completed original-source fixtures, plus new allocated I/O tests."""
    from selected_ops import metadata_certificate,submit
    import selected_contract as c
    c.previous_admission();metadata_certificate(manifest)
    old=CONTROL.parent/'nonchem-selected-01/noise-cpu'
    ready_path=Path('/work3/s234818/mosaic/runs/20261024-protenix-noisy-04/cpu-ready.json')
    ready=read(ready_path);config=verify_noise_source()['path']
    require(ready['status']=='passed' and ready['manifest_sha256']==NOISE_MANIFEST and ready['config_sha256']==sha(config) and ready['producer_job_id']=='29809138','Original noise CPU binding differs')
    for rec in ready['proofs']:require(sha(rec['path'])==rec['sha256'],'Original noise CPU proof changed')
    c.legacy.lsf_terminal(old,'29809138',expected='Done')
    cases=verify_xml(old/'payload/noise.xml',{'test_noise_protocol','test_driver_contract','test_protenix_initialization'})
    require(cases==ready['passed_cases'],'Original noise case count changed')
    storage_gate(parent,'noise-adoption')
    NOISE_ROOT.mkdir(exist_ok=False)
    ready={**ready,'launcher_manifest_sha256':manifest,'adopted_cpu_ready':file_record(ready_path),
           'proofs':ready['proofs']+[{'path':str(ready_path),'sha256':sha(ready_path)},
                                   {'path':str(metadata_certificate(manifest)),'sha256':sha(metadata_certificate(manifest))}]}
    exclusive(NOISE_ROOT/'cpu-ready.json',ready)
    submit('control','noise-gpu','noise-gpu',manifest)


def execute(phase, parent, manifest):
    from selected_ops import final_job, metadata_certificate, submit
    metadata_certificate(manifest)
    config = verify_noise_source()['path']
    study = NOISE_SOURCE / NOISE_STUDY
    if phase == 'noise-cpu':
        NOISE_ROOT.mkdir(exist_ok=False)
        storage_gate(parent, 'noise-cpu-entry')
        payload = parent / 'payload'
        payload.mkdir()
        tests = sorted(str(p) for p in (NOISE_SOURCE / 'tests').glob('test_small_molecule_*.py')
                       if not p.name.startswith('test_small_molecule_chemistry'))
        tests += [str(NOISE_SOURCE / 'tests/test_protenix_initialization.py'),
                  str(study / 'tests/test_noise_protocol.py'), str(study / 'tests/test_driver_contract.py')]
        # Exact original01 fixture paths, not later unvalidated peer-policy replacements.
        policy = study / 'tests/test_peer_policy.py'
        if policy.is_file():
            tests.append(str(policy))
        args = [PYTHON, '-m', 'pytest', *tests, '--rootdir', str(NOISE_SOURCE), '--import-mode=importlib',
                '-o', 'addopts=', '-m', 'not slow and not small_molecule_model', '-o',
                'cache_dir=' + str(payload / 'cache'), '--basetemp', str(Path(os.environ['TMPDIR']) / 'noise-cpu'),
                '--junitxml', str(payload / 'noise.xml'), '-q']
        exclusive(payload / 'command.json', {'args': args, 'source_manifest_sha256': NOISE_MANIFEST})
        env = os.environ.copy()
        env['PYTHONPATH'] = ':'.join(str(x) for x in (study / 'scripts', NOISE_SOURCE / 'scripts', NOISE_SOURCE / 'src', NOISE_SOURCE / 'tests'))
        with (payload / 'noise.log').open('x') as stream:
            subprocess.run(args, cwd=NOISE_SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=2400)
        count = verify_xml(payload / 'noise.xml', {'test_noise_protocol', 'test_driver_contract', 'test_protenix_initialization'})
        exclusive(parent / 'assessment.json', {'status': 'passed', 'cases': count, 'original_source_manifest_sha256': NOISE_MANIFEST,
                  'config_sha256': sha(config), 'test_paths': tests})
        return
    if phase == 'noise-after-cpu':
        proofs = final_job(CONTROL / 'noise-cpu', 'hpc')
        result = read(CONTROL / 'noise-cpu/assessment.json')
        require(result['status'] == 'passed', 'Noise CPU failed')
        xml = CONTROL / 'noise-cpu/payload/noise.xml'
        count = verify_xml(xml, {'test_noise_protocol', 'test_driver_contract', 'test_protenix_initialization'})
        require(count == result['cases'] and result['config_sha256'] == sha(config), 'Noise exact CPU binding differs')
        proofs += [{'path': str(p), 'sha256': sha(p)} for p in (xml, xml.with_suffix('.log'), CONTROL / 'noise-cpu/assessment.json')]
        exclusive(NOISE_ROOT / 'cpu-ready.json', {'status': 'passed', 'manifest_sha256': NOISE_MANIFEST,
                  'config_sha256': sha(config), 'producer_job_id': read(CONTROL / 'noise-cpu/submission.json')['job_id'],
                  'passed_cases': count, 'junit_sha256': sha(xml), 'launcher_manifest_sha256': manifest, 'proofs': proofs})
        storage_gate(parent, 'noise-gpu-dispatch')
        submit('control', 'noise-gpu', 'noise-gpu', manifest)
        return
    require(phase == 'noise-gpu', 'Unknown noise phase')
    ready = NOISE_ROOT / 'cpu-ready.json'
    value = read(ready)
    require(value['status'] == 'passed' and value['launcher_manifest_sha256'] == manifest, 'Noise CPU certificate scope differs')
    for rec in value['proofs']:
        require(sha(rec['path']) == rec['sha256'], 'Original noise CPU proof changed')
    storage_gate(parent, 'noise-gpu-entry')
    args = [PYTHON, str(LAUNCHER / 'noisy_entry.py'), '--driver', manifest, '--config', config,
            '--config-sha256', sha(config), '--output', str(NOISE_ROOT / 'gpu'), '--source-manifest', NOISE_MANIFEST,
            '--cpu-ready', str(ready), '--cpu-ready-sha256', sha(ready)]
    exclusive(parent / 'driver-invocation.json', {'args': args, 'original_driver': file_record(study / 'scripts/driver.py'),
              'scientific_settings_changed': False, 'storage_only_module_adapter': True})
    remaining = int(6600 - (time.time() - float(os.environ['MOSAIC_NOISY_ALLOCATION_STARTED'])))
    require(remaining > 0, 'Original shared110-minute internal clock exhausted')
    process = subprocess.Popen(args, start_new_session=True)
    deadline = time.monotonic() + remaining
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError('Original noisy internal allocation exhausted')
            current = sum(p.stat().st_size for p in NOISE_ROOT.rglob('*') if p.is_file())
            require(current < 1024**3, 'Declared one-GiB noisy output cap reached')
            time.sleep(2)
        require(process.returncode == 0, 'Original noisy driver failed; preserve output')
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    summary = read(NOISE_ROOT / 'gpu/summary.json')
    require(summary['status'] == 'passed' and summary['applied_updates'] == 1000, 'Original noise scientific acceptance failed')
    exclusive(parent / 'assessment.json', {'status': 'passed', 'applied_updates': 1000,
              'summary': file_record(NOISE_ROOT / 'gpu/summary.json'), 'original_results_unchanged': True})


def driver():
    require(len(sys.argv) > 3 and sys.argv[1] == '--driver', 'Wrong original-driver adapter entry')
    manifest = sys.argv[2]
    verify_own(manifest)
    verify_noise_source()
    require(os.environ.get('LSB_JOBID') and os.environ.get('LSB_QUEUE') == 'gpul40s', 'Actual allocated GPU required')
    # The original driver requests one additional quota observation before its
    # trajectory. Only that metadata call is routed; original numerical code runs verbatim.
    shim = types.ModuleType('stage')
    shim.storage_gate = lambda root, label: storage_gate(root, label)
    sys.modules['stage'] = shim
    original = NOISE_SOURCE / NOISE_STUDY / 'scripts/driver.py'
    sys.argv = [str(original), *sys.argv[3:]]
    sys.path[:0] = [str(original.parent), str(NOISE_SOURCE / 'scripts'), str(NOISE_SOURCE / 'src')]
    from retention import execute_noise
    output=Path(sys.argv[sys.argv.index('--output')+1])
    execute_noise(original,output)


if __name__ == '__main__':
    driver()
