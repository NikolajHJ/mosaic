"""Fail-closed allocated adoption and independent serial chemistry production.

Original scientific functions, derivative gates and boundary certifiers are
loaded from their sealed sources. Only metadata routing, authenticated exact
adapter activation and the already validated stride20 persistence are new.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import production_contract as c

BASE, ROOT, SOURCE = (Path(value) for value in (c.BASE, c.ROOT, c.SOURCE))
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
GUARD_SOURCE = BASE/'protenix-endpoint-fix-release-02/source'
GUARD_PIN = '09f759a7befb8a7b8f0e1d203e15acf939126fba493b30e6a2bf9cd2eea2216b'
RETENTION_SOURCE = BASE/'nonchem-stride20-startup-release-03/source'
RETENTION_PIN = '704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab'
require = c.require


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda value:
                      (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def atomic(path, value):
    temporary = Path(path).with_name(Path(path).name+'.partial')
    write(temporary, value); temporary.replace(path)


def binding(path):
    path = Path(path); info = path.lstat()
    require(path.is_absolute() and '..' not in path.parts and stat.S_ISREG(info.st_mode)
            and not path.is_symlink() and all(not parent.is_symlink() for parent in path.parents),
            'Require an ordinary absolute regular proof: '+str(path))
    return {'path': str(path), 'bytes': info.st_size, 'sha256': sha(path)}


def check_binding(row):
    require(binding(row['path']) == row, 'Exact bound proof bytes/path changed')
    return row


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot load verified metadata module')
    loaded = importlib.util.module_from_spec(spec); sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def verify(source, pin):
    source = Path(source); c.digest(pin)
    require(source.is_dir() and not source.is_symlink() and sha(source/'source_manifest.json') == pin,
            'Immutable source manifest differs')
    manifest = read(source/'source_manifest.json'); files = manifest['files']
    require(type(files) is dict and files, 'Empty immutable source')
    actual = {path.relative_to(source).as_posix() for path in source.rglob('*')
              if path.is_file() or path.is_symlink()}
    require(actual == set(files)|{'source_manifest.json'}, 'Exact source membership differs')
    for path in (source, *source.rglob('*')):
        require(not path.is_symlink() and (path.is_dir() or path.is_file())
                and not path.stat().st_mode & 0o222, 'Source must stay regular/read-only')
    for name, digest in files.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts
                and '\\' not in name and sha(source/name) == digest, 'Immutable source member changed')
    return manifest


def own_plan(pin):
    manifest = verify(SOURCE, pin)
    require(manifest.get('version') == c.VERSION, 'Wrong production seal scope')
    plan=c.validate_plan(read(SOURCE/'plan.json'))
    adapter_parity(plan)
    return manifest,plan


def adapter_parity(plan):
    """Require exact unchanged numerical adapter/plan bytes across the two seals."""
    c.validate_plan(plan)
    observed={}
    for arm,row in plan['pilots'].items():
        manifest=verify(Path(row['source']),row['manifest_sha256'])
        require(all(name in manifest['files'] for name in c.ADAPTER_FILES), 'Missing original numerical adapter member')
        observed[arm]={name:manifest['files'][name] for name in c.ADAPTER_FILES}
    require(observed['protenix']==observed['boltz'],
            'Per-arm pilot sources changed numerical adapters or scientific plans')
    return observed['protenix']


def pilot_metadata(arm,plan):
    c.validate_plan(plan);c.profile(arm)
    row=plan['pilots'][arm];source=Path(row['source']);root=Path(row['root'])
    verify(source,row['manifest_sha256'])
    # Distinct per-arm module namespaces are essential: Pro03 and Bol02 have
    # deliberately different metadata ROOT/SOURCE constants while the numerical
    # adapters and scientific plans above must stay byte-identical.
    loaded=module(source/'scripts/full_model_ops.py','_production_pilot_ops_'+arm)
    require(loaded.ROOT==root and loaded.SOURCE==source,'Wrong per-arm producer metadata identity')
    return loaded


def arm_plan(arm, plan):
    c.profile(arm)
    pilot = pilot_metadata(arm,plan);row=plan['pilots'][arm];source=Path(row['source'])
    manifest = pilot.verify(source,row['manifest_sha256'])
    pilot.external(manifest)
    result = read(source/'plans'/(arm+'.json'))
    require(result['source']['path'] == c.profile(arm)['source']
            and result['source']['manifest_sha256'] == c.profile(arm)['manifest'], 'Scientific source changed')
    return result


def lsf_success(parent, submission):
    path = Path(parent)/('stdout.'+submission['job_id']+'.log')
    raw = path.read_text(); compact = ''.join(raw.split())
    subjects = re.findall(r'Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)', raw)
    subjects = [tuple(''.join(field.split()) for field in row) for row in subjects]
    argv = submission.get('argv', submission.get('args'))
    require(isinstance(argv, list), 'Missing actual submission command')
    job_name = argv[argv.index('-J')+1]; queue = argv[argv.index('-q')+1]
    require(compact.count('Subject:Job') == 1 and subjects == [(submission['job_id'], job_name, 'dcc', 'Done')]
            and re.findall(r'inqueue<([^>]+)>', compact) == [queue]
            and compact.count('Successfullycompleted.') == 1 and 'Exitedwithexit' not in compact,
            'Unique same-job/name/queue successful raw LSF proof required')
    return binding(path)


def actual_terminal(parent, submission, observer, label):
    """Persist bounded raw same-job final scheduler plus genuine LSF evidence."""
    expected=submission['argv']; job=submission['job_id']
    queue=expected[expected.index('-q')+1]; name=expected[expected.index('-J')+1]
    proofs=[]
    for index in range(16):
        result=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
        path=Path(observer)/(label+'-scheduler-final-'+format(index,'03d')+'.json')
        write(path,{'job_id':job,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        proofs.append(binding(path));compact=''.join(result.stdout.split())
        require(result.returncode==0 and re.findall(r'Job<(\d+)>',compact)==[job]
                and re.findall(r'Queue<([^>]+)>',compact)==[queue]
                and re.findall(r'JobName<([^>]+)>',compact)==[name],
                'Actual original terminal scheduler identity unavailable; no fabricated state')
        states=re.findall(r'Status<([^>]+)>',compact)
        require(states in (['DONE'],['RUN'],['PEND']),'Actual producer failed')
        if states==['DONE']:
            try:return [*proofs,lsf_success(parent,submission)]
            except (FileNotFoundError,ValueError):
                if index==15:raise
        require(index<15,'Producer did not reach actual same-job terminal success')
        time.sleep(2)
    raise AssertionError('Unreachable bounded terminal wait')


def phase_parent(phase, arm='control', index=0):
    if phase in ('cpu', 'after-cpu'):
        require(arm == 'control' and index == 0, 'Wrong CPU control owner')
        return ROOT/phase
    return ROOT/arm/c.stage_label(arm, phase, index)


def argv(phase, pin, arm='control', index=0, dependency=None):
    c.digest(pin); parent = phase_parent(phase, arm, index)
    args = ['bsub', *c.resources(phase, None if arm=='control' else arm), '-J',
            'mosaic-chem-prod02-'+arm+'-'+parent.name,
            '-oo', str(parent/'stdout.%J.log'), '-eo', str(parent/'stderr.%J.log')]
    if dependency is not None:
        require(re.fullmatch(r'ended\(\d+\)(?: && ended\(\d+\))*', dependency) is not None,
                'Only explicit exact ended dependencies are allowed')
        args += ['-w', dependency]
    return args+['bash', '-l', str(SOURCE/'scripts/production_stage.lsf'), str(SOURCE), pin,
                 phase, arm, str(index)]


def submit(phase, pin, arm='control', index=0, dependency=None):
    parent = phase_parent(phase, arm, index); parent.parent.mkdir(exist_ok=True)
    parent.mkdir(exist_ok=False)
    command = argv(phase, pin, arm, index, dependency)
    record = {'status': 'unknown', 'phase': phase, 'arm': arm, 'index': index,
              'argv': command, 'source_manifest_sha256': pin, 'dependency': dependency,
              'submitted_utc': datetime.now(timezone.utc).isoformat()}
    write(parent/'submission-started.json', record)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        jobs = re.findall(r'Job <([1-9][0-9]*)> is submitted', result.stdout)
        if result.returncode == 0 and len(jobs) == 1: record.update(status='submitted', job_id=jobs[0])
    except BaseException as error: record['error'] = repr(error)
    finally: write(parent/'submission.json', record)
    require(record['status'] == 'submitted', 'Unknown outcome: preserve markers and never duplicate')
    with (ROOT/'jobs.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = read(ROOT/'jobs.json') if (ROOT/'jobs.json').exists() else {}
        key = arm+'/'+parent.name
        require(key not in jobs, 'An actual job ID cannot be overwritten')
        jobs[key] = record['job_id']; atomic(ROOT/'jobs.json', jobs)
    result = subprocess.run(['bjobs', '-l', record['job_id']], capture_output=True, text=True, timeout=30)
    write(parent/'scheduler-at-submission.json', {'returncode': result.returncode,
                                                 'stdout': result.stdout, 'stderr': result.stderr})
    return record


def own(phase, pin, arm, index):
    parent = phase_parent(phase, arm, index)
    for _ in range(20):
        if (parent/'submission.json').exists(): break
        time.sleep(1)
    sub = read(parent/'submission.json')
    require(sub['status'] == 'submitted' and sub['job_id'] == os.environ.get('LSB_JOBID')
            and sub['source_manifest_sha256'] == pin
            and sub['argv'] == argv(phase, pin, arm, index, sub['dependency']), 'Own exact job/argv/source differs')
    require(os.environ.get('LSB_QUEUE') == ('gpul40s' if phase in ('native', 'segment') else 'hpc'), 'Wrong queue')
    require(os.environ.get('LSB_DJOB_NUMPROC') == ('8' if phase in ('native', 'segment') else '4' if phase=='cpu' else '1'),
            'Wrong CPU allocation')
    return parent, sub


def cpu_certificate(pin, manifest):
    path = ROOT/'cpu-certificate.json'; value = read(path)
    require(value['status'] == 'passed' and value['source_manifest_sha256'] == pin
            and value['plan_sha256'] == sha(SOURCE/'plan.json'), 'Production CPU certificate differs')
    for row in value['proofs']: check_binding(row)
    require(xml_cases(ROOT/'cpu/payload/tests.xml', manifest['cpu_counts']) == value['cases'], 'CPU raw scope differs')
    return path


def xml_cases(path, expected):
    cases = list(ET.parse(path).iter('testcase'))
    require(cases and all(not list(case) for case in cases), 'CPU suite failed/errored/skipped')
    require(len({(case.get('classname'),case.get('name')) for case in cases}) == len(cases), 'Duplicate raw cases')
    require(dict(Counter(case.get('classname') for case in cases)) == expected, 'Exact module/count attribution differs')
    return len(cases)


def pilot_pass(arm, plan):
    """Independently re-evaluate both classifiers, readiness and actual LSF bytes."""
    adapter_parity(plan)
    row=plan['pilots'][arm];source=Path(row['source']);root=Path(row['root'])
    pilot = pilot_metadata(arm,plan); declared = pilot.verify(source,row['manifest_sha256'])
    pilot.external(declared)
    pilot.verify_cpu_certificate(row['manifest_sha256'],declared,row['cpu_certificate_sha256'])
    parent = root/arm; sub = read(parent/'submission.json')
    require(sub['status'] == 'submitted' and sub['job_id'] == row['job_id']
            and sub['argv'] == pilot.argv(arm,row['manifest_sha256'])
            and sub['cpu_certificate_sha256'] == row['cpu_certificate_sha256'], 'Actual per-arm pilot submission differs')
    lsf = lsf_success(parent, sub)
    top = read(parent/'payload/assessment.json'); comparison_path = parent/'payload/comparison/assessment.json'
    require(top['status'] == 'passed' and top['job_id'] == sub['job_id']
            and top['source_manifest_sha256'] == row['manifest_sha256']
            and top['comparison_sha256'] == sha(comparison_path), 'Allocated full-model pilot did not pass')
    comparison = read(comparison_path)
    require(comparison['status'] == 'passed_full_model_comparison'
            and comparison['exact_initial_arrays_and_key'] is True
            and comparison['original_results_changed'] is False, 'Complete matched pilot comparison required')
    contract = module(source/'scripts/full_model_contract.py', '_production_pilot_contract_'+arm)
    native_plan = arm_plan(arm, plan); config = read(native_plan['inputs']['config']['path'])
    proofs = [binding(parent/'submission.json'), lsf, binding(parent/'payload/assessment.json'), binding(comparison_path),
              binding(root/'cpu-certificate.json'),binding(source/'source_manifest.json'),
              binding(source/'plans'/(arm+'.json'))]
    for variant in ('baseline', 'combined'):
        path = parent/'payload/comparison'/variant/'assessment.json'; variant_record = read(path)
        require(contract.assess_result(variant_record, native_plan, config) == variant_record['classification'],
                'Original complete pilot classifier differs')
        embedded = dict(comparison['variants'][variant]); embedded.pop('isolated_child_execution', None)
        require(embedded == variant_record, 'Comparison detached from original child evidence')
        require(variant_record['original_source_before'] == variant_record['original_source_after']
                and variant_record['input_bindings_before'] == variant_record['input_bindings_after'],
                'Original inputs changed during pilot')
        for row in variant_record['input_bindings_after'].values(): check_binding(row)
        check_binding(variant_record['driver_summary_file'])
        require(read(variant_record['driver_summary_file']['path']) == variant_record['driver_summary'], 'Pilot summary embedding differs')
        proofs.append(binding(path))
    ready = parent/'payload/comparison/combined/study/assessment.json'
    result = read(ready)
    require(result['status'] == 'passed' and result['source_manifest_sha256'] == c.profile(arm)['manifest'],
            'Original passed combined readiness absent')
    for row in result['proofs']:
        require(sha(row['path']) == row['sha256'], 'Original native/coupled proof changed')
    unit = result['storage_projection']['maximum_pilot_update_bytes']
    require(type(unit) is int and unit > 0 and result['storage_projection']['projected_bytes'] == 2*unit*10000+c.GIB
            and result['runtime_gate']['status'] == 'passed'
            and result['runtime_gate']['conservative_projected_seconds'] < 82800, 'Pilot runtime/empirical projection failed')
    runtime=module(SOURCE/'scripts/production_context.py','_production_context_adoption').read_context(sys.modules[__name__],arm,plan)
    return {'status': 'passed', 'arm': arm, 'execution_context':runtime, 'unit_bytes': unit, 'readiness': binding(ready),
            'summary': binding(ready.with_name('summary.json')), 'proofs': proofs,
            'original_scientific_failure_reclassified': False}


def _tree_bytes_once(root):
    root = Path(root)
    if not root.exists(): return 0
    require(root.is_dir() and not root.is_symlink(), 'New production tree is linked/special')
    total = 0
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            path = Path(directory)/name
            require(path.is_dir() and not path.is_symlink(), 'Production output directory must be ordinary')
        for name in files:
            path = Path(directory)/name; info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                require(path.resolve(strict=True).is_relative_to(root.resolve()), 'External/unresolved output link')
                total += info.st_size
            else:
                require(stat.S_ISREG(info.st_mode), 'Production output special file')
                total += info.st_size
    return total


def tree_bytes(root):
    """Retry a whole live-tree scan after atomic replacement; never skip a file."""
    for attempt in range(3):
        try:return _tree_bytes_once(root)
        except FileNotFoundError:
            if attempt==2:raise
    raise AssertionError('Unreachable bounded tree scan')


def _control_bytes_once():
    """Count own release and CPU/control trees; never double-count arm data."""
    total=tree_bytes(SOURCE.parent)
    if not ROOT.exists():return total
    require(ROOT.is_dir() and not ROOT.is_symlink(),'Production control root must be ordinary')
    for path in ROOT.iterdir():
        if path.name in c.PROFILES:
            require(path.is_dir() and not path.is_symlink(),'Arm scope must be an ordinary independent directory')
            continue
        if path.is_dir():total+=tree_bytes(path)
        else:
            info=path.lstat()
            require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),'Control proof is linked/special')
            total+=info.st_size
    return total


def control_bytes():
    for attempt in range(3):
        try:return _control_bytes_once()
        except FileNotFoundError:
            if attempt==2:raise
    raise AssertionError('Unreachable bounded control scan')


def enforce_control_cap():
    current=control_bytes()
    require(current<=c.CONTROL_CAP,'New chemistry source/CPU/control64MiB cap exhausted')
    return current


def finish_assessment(path,assessment):
    """Never persist a passed assessment after the own metadata cap failed."""
    current=control_bytes()
    assessment.update(control_cap_bytes=c.CONTROL_CAP,control_bytes_before_assessment=current)
    length=len((json.dumps(assessment,indent=2,allow_nan=False)+'\n').encode('utf8'))
    exceeded=current>c.CONTROL_CAP or current+length>c.CONTROL_CAP
    if exceeded:
        assessment.update(status='failed',control_cap_exceeded=True)
        assessment.setdefault('error','Own chemistry source/CPU/control cap exceeded')
    # A small failed outcome receipt is preserved even if the failure is a cap
    # overrun; no scientific work or successor is admitted by that receipt.
    write(path,assessment)
    require(not exceeded,'New chemistry metadata cap failed at stage completion')
    enforce_control_cap()


def storage(parent, phase, *, adopted=False):
    """Retain the entire current precision/full-campaign gate; add both chem scopes."""
    verify(GUARD_SOURCE, GUARD_PIN)
    guard = module(GUARD_SOURCE/'scripts/protenix_precision_ops.py', '_chemistry_current_guard')
    inherited=Path(parent)/('inherited-storage-'+phase)
    inherited.mkdir(exist_ok=False)
    prior = guard.storage(inherited, 'entry')
    c.require_storage(prior)
    record = {'status': 'failed', 'original_gate_changed': False, 'parent_gate': prior,
              'reserve_bytes': 4*c.GIB, 'file_reserve': 100000,
              'available_bytes': prior['available_bytes'], 'adopted_full_production': adopted}
    try:
        current_control=enforce_control_cap()
        future = c.CONTROL_CAP-current_control
        record['metadata_bytes'] = current_control; record['metadata_full_cap'] = c.CONTROL_CAP
        rows = {}
        for arm in c.PROFILES:
            root = ROOT/arm; current = tree_bytes(root)
            if adopted:
                evidence = adoption(arm)
                unit = evidence['unit_bytes']
                projection_path = root/'empirical-unit.json'
                if projection_path.exists():
                    update = read(projection_path)
                    for proof in update['proofs']: check_binding(proof)
                    require(update['unit_bytes'] >= unit, 'Empirical unit was reduced')
                    unit = update['unit_bytes']
                full = c.projection(unit, arm)
            else:
                require(current == 0, 'Unadopted chemistry output/unknown job exists')
                full = 0
            require(current <= full, 'New chemistry output exceeds full empirical cap')
            rows[arm] = {'full': full, 'current': current, 'remaining': full-current}
            future += full-current
        record.update(chemistry=rows, additional_remaining_bytes=future,
                      required_bytes=prior['required_bytes']+future)
        # BASE is charged in full by the inherited guard; no stale-quota credit
        # or reserve borrowing from another filesystem is introduced here.
        require(record['available_bytes'] >= record['required_bytes'], 'Full chemistry future capacity does not fit')
        record['status'] = 'passed'
    finally: write(Path(parent)/('production-storage-'+phase+'.json'), record)
    c.require_storage(record)
    return record


def adoption(arm):
    path = ROOT/arm/'adoption.json'; value = read(path)
    require(value['status'] == 'passed' and value['arm'] == arm
            and value['production_manifest_sha256'] == sha(SOURCE/'source_manifest.json')
            and value['plan_sha256'] == sha(SOURCE/'plan.json'), 'Different chemistry production owner')
    for row in value['proofs']: check_binding(row)
    check_binding(value['readiness']); check_binding(value['summary'])
    return value


def predecessor(arm, index):
    if index == 0: return None
    root = ROOT/arm; label = c.stage_label(arm, 'segment', index-1)
    certificate = read(root/(label+'-certificate.json'))
    require(certificate['status'] == 'passed' and certificate['source_manifest_sha256'] == c.profile(arm)['manifest']
            and certificate['end'] == index*10000, 'Original complete predecessor boundary required')
    for proof in certificate['proofs']:
        require(sha(proof['path']) == proof['sha256'], 'Predecessor proof changed')
    check_binding(certificate['terminal_snapshot'])
    return certificate['terminal_snapshot']


def readiness(arm, index):
    if index == 0: return adoption(arm)['readiness']
    certificate = read(ROOT/arm/(c.stage_label(arm, 'native', index)+'-certificate.json'))
    require(certificate['status'] == 'passed' and certificate['index'] == index
            and certificate['production_manifest_sha256'] == sha(SOURCE/'source_manifest.json'), 'New exact-boundary readiness missing')
    for proof in certificate['proofs']: check_binding(proof)
    return check_binding(certificate['readiness'])


def driver_args(arm, phase, index, output, plan, *, initial_ready=None):
    spec = c.profile(arm); native = arm_plan(arm, plan); inputs = native['inputs']
    require(phase in ('native', 'segment') and type(index) is int and 0 <= index < spec['segments'], 'Undeclared driver scope')
    args = [PYTHON, native['driver']['path'], '--mode', 'validate' if phase=='native' else 'run',
            '--config', inputs['config']['path'], '--output', str(output), '--manifest', spec['manifest'],
            '--cpu-certificate', inputs['cpu_certificate']['path'], '--cpu-certificate-sha256', inputs['cpu_certificate']['sha256'],
            '--reference', inputs['reference']['path'], '--pocket', inputs['pocket']['path'],
            '--pocket-sha256', inputs['pocket']['sha256'], '--offset', str(index*10000), '--segment-steps', '10000']
    if arm == 'boltz':
        args += ['--prepared', str(Path(inputs['prepared_context']['path']).parent), '--context-sha256',
                 inputs['prepared_context']['sha256'], '--worker-python', native['worker_python']]
    snapshot = predecessor(arm, index)
    if snapshot: args += ['--resume-snapshot', snapshot['path'], '--resume-sha256', snapshot['sha256']]
    if phase == 'segment':
        ready = initial_ready if initial_ready is not None and index == 0 else readiness(arm, index)
        check_binding(ready)
        args += ['--ready', ready['path'], '--ready-sha256', ready['sha256']]
    return args


def driver_environment(arm):
    source = Path(c.profile(arm)['source'])
    return dict(os.environ, PYTHONPATH=os.pathsep.join((str(source/'scripts'), str(source/'src'), str(source/'tests'))),
                JAX_PLATFORMS='cuda,cpu' if arm=='protenix' else 'cpu', JAX_ENABLE_X64='0',
                XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONDONTWRITEBYTECODE='1',
                MOSAIC_SPEED_PLAN=str(BASE/'chemistry-search-speed-02/plan.json'))


def child(arm, phase, index, pin, parent, *, startup=False, terminal=False):
    mode = '--startup-only' if startup else '--certify-terminal' if terminal else '--execute'
    command = [PYTHON, str(SOURCE/'scripts/production_run.py'), '--manifest', pin,
               '--arm', arm, '--phase', phase, '--index', str(index), mode]
    path = parent/('terminal-acceptance' if terminal else 'startup' if startup else 'driver')
    timeout=900 if terminal else 360 if startup else 6700 if phase=='native' else 81000
    plan=c.validate_plan(read(SOURCE/'plan.json'))
    transport=module(SOURCE/'scripts/production_context.py','_production_context_execution')
    actual_context=transport.read_context(sys.modules[__name__],arm,plan)
    require(adoption(arm)['execution_context']==actual_context,'Accepted pilot compiler context changed')
    environment=transport.apply(driver_environment(arm),actual_context)
    write(path.with_name(path.name+'-context.json'),actual_context)
    process=subprocess.Popen(command,cwd=SOURCE,env=environment,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True,start_new_session=True)
    timed_out=False
    started=time.monotonic()
    try:
        unit=adoption(arm)['unit_bytes']
        current_unit=ROOT/arm/'empirical-unit.json'
        if current_unit.exists():unit=max(unit,read(current_unit)['unit_bytes'])
        cap=c.projection(unit,arm)
        while True:
            remaining=timeout-(time.monotonic()-started)
            require(remaining>0,'Allocated child internal walltime exhausted')
            require(tree_bytes(ROOT/arm)<=cap,'Full empirical chemistry output cap exhausted')
            try:
                stdout,stderr=process.communicate(timeout=min(10,remaining));break
            except subprocess.TimeoutExpired:pass
    except BaseException:
        timed_out=True
        if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        try:stdout,stderr=process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate(timeout=10)
        write(path.with_suffix('.json'),{'argv':command,'returncode':process.returncode,
              'stdout':stdout,'stderr':stderr,'timed_out_or_interrupted':True,'child_fully_reaped':process.poll() is not None})
        raise
    write(path.with_suffix('.json'), {'argv': command, 'returncode': process.returncode,
                                     'stdout': stdout, 'stderr': stderr,'timed_out_or_interrupted':timed_out,
                                     'child_fully_reaped':process.poll() is not None})
    require(process.returncode == 0, 'Allocated chemistry '+mode+' child failed; preserve primary log')


def metadata_startups(payload, pin):
    """Persist each actual attempted CPU startup before accepting its outcome.

    The original pilot/readiness acceptance remains in production_run. Failed
    pilots still reject that route; their captured reason must survive the
    controller's return-code assertion, with no implied GPU/native pass.
    """
    payload = Path(payload)
    summary = {'status': 'failed_metadata_only', 'commands': [],
               'gpu_or_gradient_passed': False, 'optimizer_updates': 0}
    try:
        for name in c.PROFILES:
            command = [PYTHON, str(SOURCE/'scripts/production_run.py'), '--manifest', pin,
                       '--arm', name, '--phase', 'segment', '--index', '0', '--metadata-startup']
            attempt = {'arm': name, 'argv': command, 'returncode': None, 'stdout': '', 'stderr': ''}
            try:
                result = subprocess.run(command, cwd=SOURCE, capture_output=True, text=True, timeout=180)
                attempt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            except BaseException as error:
                def text(value):
                    return value.decode('utf8', errors='replace') if isinstance(value, bytes) else value or ''
                attempt.update(error=repr(error), stdout=text(getattr(error, 'stdout', None)),
                               stderr=text(getattr(error, 'stderr', None)),
                               timed_out_or_interrupted=True)
                raise
            finally:
                path = payload/('startup-attempt-'+name+'.json')
                write(path, attempt)
                summary['commands'].append({**attempt, 'captured_attempt': binding(path)})
            require(result.returncode == 0, 'Actual production wrapper/parser metadata startup rejected; '
                    'captured original reason: '+str(path))
        summary['status'] = 'passed_metadata_only'
        return summary
    except BaseException as error:
        summary['error'] = repr(error)
        raise
    finally:
        write(payload/'startup.json', summary)


def bootstrap(pin):
    """Explicit once-only metadata entry; called only after source staging audit."""
    manifest, plan = own_plan(pin)
    require(not ROOT.exists(), 'Existing/unknown production root prohibits bootstrap replay')
    ROOT.mkdir(); write(ROOT/'bootstrap-started.json', {'source_manifest_sha256': pin, 'plan_sha256': sha(SOURCE/'plan.json')})
    storage(ROOT, 'preflight')
    # Actual first production startup arguments contain the accepted readiness
    # digest. Wait for those exact files; never use a synthetic future digest.
    cpu = submit('cpu', pin, dependency='ended('+plan['pilots']['protenix']['job_id']+')')
    callback = submit('after-cpu', pin, dependency=c.callback_dependency(cpu['job_id'],plan['pilots']))
    write(ROOT/'initial-submissions.json', {'cpu': cpu, 'after_cpu': callback, 'scientific_updates': 0})
    enforce_control_cap()
    return cpu, callback


def certify_cpu(pin, manifest):
    parent = ROOT/'cpu'; submission = read(parent/'submission.json')
    plan = c.validate_plan(read(SOURCE/'plan.json'))
    expected_dependency = 'ended('+plan['pilots']['protenix']['job_id']+')'
    require(submission['argv'] == argv('cpu', pin, dependency=expected_dependency)
            and submission['status'] == 'submitted', 'CPU submission changed')
    terminal=actual_terminal(parent,submission,ROOT/'after-cpu','cpu')
    assessment = read(parent/'payload/assessment.json')
    require(assessment['status'] == 'passed' and assessment['source_manifest_sha256'] == pin, 'New allocated CPU did not pass')
    cases = xml_cases(parent/'payload/tests.xml', manifest['cpu_counts'])
    require(cases == assessment['cases'], 'CPU assessment/XML count differs')
    write(ROOT/'cpu-certificate.json', {'status': 'passed', 'source_manifest_sha256': pin,
          'plan_sha256': sha(SOURCE/'plan.json'), 'cases': cases,
          'proofs': [binding(parent/'submission.json'), *terminal, binding(parent/'payload/assessment.json'),
                     binding(parent/'payload/tests.xml'), binding(parent/'payload/startup.json')]})


def queue_pair(arm, phase, index, pin):
    root = ROOT/arm
    with (root/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        producer = phase_parent(phase, arm, index)
        require(not producer.exists() and not phase_parent('after-'+phase, arm, index).exists(),
                'Existing/unknown production pair prohibits replay')
        record = submit(phase, pin, arm, index)
        submit('after-'+phase, pin, arm, index, 'ended('+record['job_id']+')')


def preserve_storage_block(parent,submission,pin,error):
    """Record actual failed admission; never imply a pass or replay authority."""
    parent=Path(parent);candidates=[]
    inherited=parent/'inherited-storage-first-dispatch'
    if inherited.exists():candidates+=list(inherited.rglob('*.json'))
    own_gate=parent/'production-storage-first-dispatch.json'
    if own_gate.exists():candidates.append(own_gate)
    failed=[]
    for path in candidates:
        value=read(path)
        if type(value) is dict and value.get('status')=='failed':failed.append(binding(path))
    if not failed:return False
    require(not any((ROOT/name/c.stage_label(name,'segment',0)).exists() for name in c.PROFILES),
            'A long/unknown outcome already exists; cannot report no submission')
    write(ROOT/'waiting-storage.json',{
        'status':'blocked_storage_production_not_submitted',
        'source_manifest_sha256':pin,'job_id':submission['job_id'],
        'error':repr(error),'failed_storage_receipts':failed,
        'adoptions':[binding(ROOT/name/'adoption.json') for name in c.PROFILES],
        'production_submissions':False,'historical_gates_changed':False,
        'recovery_requires':'New explicit allocated dispatch phase/new immutable validated source; never replay this callback'})
    enforce_control_cap()
    return True


def execute(phase, pin, arm='control', index=0):
    require(os.environ.get('LSB_JOBID') and not sys.flags.optimize, 'Actual allocated unoptimized Python required')
    manifest, plan = own_plan(pin); parent, sub = own(phase, pin, arm, index)
    if phase=='cpu':
        expected='ended('+plan['pilots']['protenix']['job_id']+')'
        require(sub['dependency']==expected,'CPU must wait for the exact fresh Pro producer; preserved Bol LSF is verified')
    elif phase=='after-cpu':
        expected=c.callback_dependency(read(ROOT/'cpu/submission.json')['job_id'],plan['pilots'])
        require(sub['dependency']==expected,'Production callback must wait for its own accepted CPU; both pilot proofs remain required')
    elif phase in ('after-native','after-segment'):
        producer=phase_parent(phase.removeprefix('after-'),arm,index)
        expected='ended('+read(producer/'submission.json')['job_id']+')'
        require(sub['dependency']==expected,'Serial callback dependency must be its own exact producer')
    else:
        require(sub['dependency'] is None,'Independent native/segment must not acquire an inter-arm dependency')
    enforce_control_cap()
    if phase == 'after-cpu':
        with (ROOT/'dispatch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            certify_cpu(pin, manifest)
            for name in c.PROFILES:
                root = ROOT/name; require(not root.exists(), 'Existing chemistry adoption/unknown outcome')
                evidence = pilot_pass(name, plan); root.mkdir()
                evidence.update(production_manifest_sha256=pin, plan_sha256=sha(SOURCE/'plan.json'))
                evidence['proofs'] += [binding(ROOT/'cpu-certificate.json'), binding(SOURCE/'plan.json')]
                write(root/'adoption.json', evidence)
            try:
                storage(parent, 'first-dispatch', adopted=True)
            except BaseException as error:
                # Keep the actual failed capacity receipts and validated
                # adoptions. This is not a passed dispatch and the callback is
                # never replayable: a future recovery needs a new explicit,
                # allocated and validated dispatch source/phase.
                preserve_storage_block(parent,sub,pin,error)
                raise
            for name in c.PROFILES: queue_pair(name, 'segment', 0, pin)
            enforce_control_cap()
        return
    payload = parent/'payload'; payload.mkdir(exist_ok=False)
    assessment = {'status': 'failed', 'phase': phase, 'arm': arm, 'index': index,
                  'job_id': sub['job_id'], 'source_manifest_sha256': pin,
                  'original_results_changed': False}
    try:
        if phase == 'cpu':
            storage(payload, 'cpu-entry')
            command = [PYTHON, '-m', 'pytest', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
                       *[str(SOURCE/'tests'/(name+'.py')) for name in manifest['cpu_modules']],
                       '--junitxml='+str(payload/'tests.xml')]
            result = subprocess.run(command, cwd=SOURCE, capture_output=True, text=True, timeout=2400)
            write(payload/'tests-command.json', {'argv': command, 'returncode': result.returncode,
                                                'stdout': result.stdout, 'stderr': result.stderr})
            require(result.returncode == 0, 'Production contracts failed')
            assessment['cases'] = xml_cases(payload/'tests.xml', manifest['cpu_counts'])
            # Execute the sealed real wrapper and original parser route in fresh
            # metadata-only processes. Neither readiness nor GPU passes follow.
            metadata_startups(payload, pin)
        else:
            cpu_certificate(pin, manifest); adoption(arm)
            assessment['storage'] = storage(payload, phase+'-entry', adopted=True)
            if phase in ('native', 'segment'):
                child(arm, phase, index, pin, parent, startup=True)
                child(arm, phase, index, pin, parent)
            elif phase == 'after-native':
                producer = phase_parent('native', arm, index); producer_sub = read(producer/'submission.json')
                terminal=actual_terminal(producer,producer_sub,parent,'native')
                require(read(producer/'payload/assessment.json')['status'] == 'passed', 'Native producer failed')
                child(arm, 'native', index, pin, parent, terminal=True)
                ready = binding(producer/'payload/study/assessment.json')
                write(ROOT/arm/(c.stage_label(arm,'native',index)+'-certificate.json'), {
                    'status': 'passed', 'index': index, 'production_manifest_sha256': pin,
                    'readiness': ready, 'proofs': [*terminal, binding(producer/'submission.json'),
                                                 binding(parent/'terminal-acceptance.json'), ready]})
                current = read(ready['path'])['storage_projection']['maximum_pilot_update_bytes']
                old = ROOT/arm/'empirical-unit.json'
                prior = read(old) if old.exists() else {'unit_bytes': adoption(arm)['unit_bytes'], 'proofs': []}
                atomic(old, {'unit_bytes': max(current,prior['unit_bytes']), 'proofs': [*prior['proofs'],ready]})
                storage(payload, 'segment-dispatch', adopted=True); queue_pair(arm,'segment',index,pin)
            elif phase == 'after-segment':
                producer = phase_parent('segment',arm,index); producer_sub = read(producer/'submission.json')
                terminal=actual_terminal(producer,producer_sub,parent,'segment')
                write(parent/'terminal-proofs.json',{'proofs':terminal})
                require(read(producer/'payload/assessment.json')['status']=='passed', 'Scientific producer failed')
                child(arm,'segment',index,pin,parent,terminal=True)
                if index+1<c.profile(arm)['segments']:
                    storage(payload,'next-native-dispatch',adopted=True); queue_pair(arm,'native',index+1,pin)
            else: raise ValueError('Unknown phase')
        assessment['status'] = 'passed'
    except BaseException as error:
        assessment['error'] = repr(error); raise
    finally: finish_assessment(payload/'assessment.json',assessment)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('manifest'); parser.add_argument('phase')
    parser.add_argument('arm',nargs='?',default='control'); parser.add_argument('index',nargs='?',type=int,default=0)
    args=parser.parse_args()
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise InterruptedError('Allocated chemistry controller received SIGTERM')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.phase=='bootstrap': bootstrap(args.manifest)
        else: execute(args.phase,args.manifest,args.arm,args.index)
    finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__': main()
