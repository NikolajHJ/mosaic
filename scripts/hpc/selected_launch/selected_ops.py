"""Allocated selected launch routing; original scientific callables stay intact."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import selected_contract as c
from selected_contract import (BASE, CONTROL, COUNTS, LAUNCHER, PROFILES, PYTHON,
    SELECTED, VERSION, arm_label, atomic, decode, exclusive, file_record,
    native_units, profile, read, require, sha, verify_own, verify_source)
from selected_storage import storage_gate

SUITE = 'selected-nonchemistry-stride20-metadata-v2'
CASES = 78


def resources(phase):
    if phase in ('native', 'segment'):
        return ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]',
                '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00' if phase == 'native' else '23:00']
    require(phase in ('cpu', 'after-cpu', 'dispatch', 'after-native', 'after-segment', 'noise-cpu', 'noise-after-cpu', 'noise-gpu'), 'Unknown phase')
    if phase == 'noise-gpu':
        return ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00']
    return ['-q', 'hpc', '-n', '4' if phase == 'noise-cpu' else '2', '-R', 'span[hosts=1]',
            '-R', 'rusage[mem=4GB]', '-W', '01:00' if phase == 'noise-cpu' else '00:30']


def parent_path(name, label, phase):
    if phase in ('cpu', 'after-cpu', 'noise-cpu', 'noise-after-cpu', 'noise-gpu'):
        require(name == 'control' and label == phase, 'Wrong control parent')
        return CONTROL / label
    p = profile(name)
    if phase == 'dispatch':
        require(label == 'dispatch-0004', 'Only fresh selected dispatch0004 is declared')
    else:
        require(decode(name, label)[0] == phase, 'Phase label differs')
    return p['root'] / label


def command(name, label, phase, manifest, dependency=None):
    parent = parent_path(name, label, phase)
    require(re.fullmatch('[0-9a-f]{64}', manifest), 'Missing selected source pin')
    if dependency is not None:
        require(re.fullmatch(r'ended\(\d+\)', dependency), 'Only own-ended dependencies declared')
    args = ['bsub', *resources(phase), '-J', 'mosaic-selected-' + name + '-' + label,
            '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if dependency:
        args += ['-w', dependency]
    return args + ['bash', '-l', str(LAUNCHER / 'stage.lsf'), str(LAUNCHER), manifest,
                   name, phase, str(parent)]


def executing(name, phase, parent, manifest):
    parent = Path(parent)
    require(parent == parent_path(name, parent.name, phase), 'Own allocated parent escapes declaration')
    for index in range(20):
        if (parent / 'submission.json').exists():
            break
        require(index < 19, 'Missing own actual submission')
        time.sleep(1)
    sub = read(parent / 'submission.json')
    require(sub['job_id'] == os.environ.get('LSB_JOBID') and sub['outcome'] == 'submitted'
            and sub['args'] == command(name, parent.name, phase, manifest, sub['dependency']),
            'Actual executing source/resources/owner differ')
    return sub


def submit(name, label, phase, manifest, dependency=None):
    parent = parent_path(name, label, phase)
    parent.mkdir(exist_ok=False)
    args = command(name, label, phase, manifest, dependency)
    value = {'outcome': 'unknown', 'phase': phase, 'profile': name, 'args': args,
             'dependency': dependency, 'launcher_manifest_sha256': manifest,
             'utc': datetime.now(timezone.utc).isoformat()}
    exclusive(parent / 'submission-started.json', value)
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        value.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        ids = re.findall(r'Job <(\d+)>', result.stdout)
        if result.returncode == 0 and len(ids) == 1:
            value.update(outcome='submitted', job_id=ids[0])
    except BaseException as error:
        value['error'] = repr(error)
    finally:
        exclusive(parent / 'submission.json', value)
    require(value['outcome'] == 'submitted', 'Unknown submission; inspect marker, never duplicate')
    path = parent.parent / 'jobs.json'
    jobs = read(path) if path.exists() else {}
    require(label not in jobs, 'Existing actual job ID must not be overwritten')
    jobs[label] = value['job_id']
    atomic(path, jobs)
    q = subprocess.run(['bjobs', '-l', value['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(q.stdout + q.stderr)
    return value


def final_job(parent, queue):
    parent = Path(parent)
    sub = read(parent / 'submission.json')
    job = sub['job_id']
    for index in range(20):
        q = subprocess.run(['bjobs', '-l', job], capture_output=True, text=True, timeout=30)
        path = parent / ('selected-scheduler-final-' + str(index) + '.txt')
        with path.open('x') as stream:
            stream.write(q.stdout + q.stderr)
        compact = re.sub(r'\s+', '', q.stdout + q.stderr)
        require(q.returncode == 0 and re.findall(r'Job<([^>]+)>', compact) == [job]
                and re.findall(r'Queue<([^>]+)>', compact) == [queue], 'Actual final scheduler identity unavailable')
        statuses = re.findall(r'Status<([^>]+)>', compact)
        if statuses == ['DONE']:
            break
        require(statuses in (['RUN'], ['PEND']) and index < 19, 'Actual predecessor failed')
        time.sleep(2)
    lsf = c.legacy.lsf_terminal(parent, job, expected='Done')
    return [{'path': v['path'], 'sha256': v['sha256']} for v in
            (file_record(path), file_record(parent / 'submission.json'), lsf['proof'])]


def metadata_certificate(manifest):
    path = CONTROL / 'cpu-certificate.json'
    value = read(path)
    require(value['status'] == 'passed' and value['suite'] == SUITE and value['cases'] == CASES
            and value['launcher_manifest_sha256'] == manifest, 'Selected metadata CPU scope differs')
    c.legacy.verify_proofs(value)
    return path


def load_original(name, manifest):
    p = profile(name)
    verify_source(name)
    source = p['source']
    sys.path[:0] = [str(source / 'scripts'), str(source / 'src'), str(source / 'tests')]
    module_path = source / 'scripts' / ('extended_ops.py' if name == 'protenix50k40k' else 'boltz_serial_ops.py')
    if name == 'dbmol10k':
        sys.path.insert(0, str(p['original_launcher'] / 'scripts'))
        module_path = p['original_launcher'] / 'scripts/dbmol_storage_ops.py'
    spec = importlib.util.spec_from_file_location('selected_original_' + name, module_path)
    op = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = op
    spec.loader.exec_module(op)
    if name == 'dbmol10k':
        op.configure_launcher(p['original_launcher_manifest'])
    op.storage_gate = storage_gate
    op.all_budgets = lambda: {SELECTED[name]: native_units(name)['full']}
    op.command = lambda label, phase, science, dependency=None: command(name, label, phase, manifest, dependency)
    op.submit = lambda label, phase, science, dependency=None: selected_submit(name, label, phase, manifest, dependency, op)
    op.executing = lambda parent, phase, science: executing(name, phase, parent, manifest)
    op.final_job = final_job
    from retention import adapt_certifier
    adapt_certifier(op,name)
    if name=='protenix50k40k':
        op.command_path=lambda arm,phase,index: p['root']/(arm+'-stride20-v3-command-'+phase+'-'+format(index,'04d')+'.json')
    elif name=='boltz30k':
        op.startup_paths=lambda arm,phase,index:(p['root']/(arm+'-stride20-v3-'+phase+'-'+format(index,'04d')+'.json'),p['root']/(arm+'-stride20-v3-'+phase+'-'+format(index,'04d')+'-certificate.json'))
    else:
        op.startup_path=lambda phase,index:p['root']/('pilot-stride20-v3-'+phase+'-'+format(index,'04d')+'-startup.json')
    if name == 'dbmol10k':
        op.dispatch = lambda parent, science: dispatch(name, parent, manifest, op)
    else:
        op.dispatch = lambda science, parent: dispatch(name, parent, manifest, op)
    return op


def selected_readiness(name, index, op):
    p = profile(name)
    if name == 'dbmol10k':
        return op.readiness(index, p['manifest'])
    return op.native_assessment(SELECTED[name], index, p['manifest'])


def startup(name, phase, index, parent, manifest, op):
    p = profile(name)
    if name == 'protenix50k40k':
        return op.ensure_command_capture(SELECTED[name], phase, index, p['manifest'], parent)
    if name == 'boltz30k':
        return op.ensure_startup(SELECTED[name], phase, index, p['manifest'], parent)
    return op.ensure_startup(phase, index, p['manifest'])


def selected_submit(name, label, phase, manifest, dependency, op):
    if phase in ('native', 'segment'):
        decoded_phase, index = decode(name, label)
        require(decoded_phase == phase, 'Wrong selected submission phase')
        if name == 'protenix50k40k':
            op.verify_command_capture(SELECTED[name], phase, index, profile(name)['manifest'])
        elif name == 'boltz30k':
            op.verify_startup(SELECTED[name], phase, index, profile(name)['manifest'])
        else:
            op.verify_startup_entry(phase, index, profile(name)['manifest'])
    return submit(name, label, phase, manifest, dependency)


def produce_with_retention(name,op,parent,phase,index,manifest):
    """Activate persistence routing only after unchanged startup checks finish."""
    p=profile(name)
    if name=='protenix50k40k':
        op.verify_command_capture(SELECTED[name],phase,index,p['manifest'])
        op.ensure_gpu_startup(SELECTED[name],phase,index,p['manifest'],parent)
    elif name=='boltz30k':op.verify_startup(SELECTED[name],phase,index,p['manifest'])
    else:op.verify_startup_entry(phase,index,p['manifest'])
    from retention import ProcessProxy
    original=op.subprocess
    op.subprocess=ProcessProxy(original,name,manifest)
    try:
        if name=='dbmol10k':return op.produce(parent,phase,index,p['manifest'])
        return op.produce(parent,phase,SELECTED[name],index,p['manifest'])
    finally:op.subprocess=original


def ownership(name, manifest):
    path = profile(name)['root'] / 'nonchem-stride20-v3-owner.json'
    value = read(path)
    require(value['status'] == 'passed' and value['kind']=='selected_owner' and value['profile'] == name
            and value['launcher_manifest_sha256'] == manifest and value['selected_arm'] == SELECTED[name],
            'Exclusive selected ownership differs')
    c.legacy.verify_proofs(value)
    return value


def adopt(name, parent, manifest, op):
    p = profile(name)
    path = p['root'] / 'nonchem-stride20-v3-owner.json'
    if path.exists():
        return ownership(name, manifest)
    require(not (p['root'] / 'protenix02-adoption-owner.json').exists(), 'Another controller owns production')
    require(not c.production_entries(p['root']), 'Original production or unknown marker already exists')
    require(not (p['root'] / 'dispatch-0001').exists(), 'Original dispatch outcome must be resolved')
    c.deferred_absence()
    c.previous_admission()
    units = native_units(name)
    if op is None:
        # Each scientific release gets a fresh module namespace, as in production.
        # The allocated child authenticates the exact original readiness callable.
        code=('import selected_ops as op; name='+repr(name)+'; manifest='+repr(manifest)+'; '
              'original=op.load_original(name,manifest); op.selected_readiness(name,0,original)')
        subprocess.run([PYTHON,'-c',code],check=True,capture_output=True,text=True,timeout=180)
    else:
        selected_readiness(name, 0, op)
    proof_paths = [p['root'] / 'cpu-certificate.json',
                   p['root'] / (c.legacy.native_name(name, SELECTED[name]) + '-certificate.json'),
                   p['root'] / c.legacy.native_name(name, SELECTED[name]) / 'payload/study/assessment.json',
                   metadata_certificate(manifest)]
    value = {'status': 'passed', 'kind':'selected_owner', 'version': VERSION, 'profile': name,
             'selected_arm': SELECTED[name], 'launcher_manifest_sha256': manifest,
             'scientific_source_manifest_sha256': p['manifest'],
             'initial_empirical_unit_bytes': units['unit_bytes'],
             'proofs': [{'path': str(x), 'sha256': sha(x)} for x in proof_paths],
             'science_and_saved_results_changed': False}
    exclusive(path, value)
    return ownership(name, manifest)


def dispatch(name, parent, manifest, op):
    p = profile(name)
    metadata_certificate(manifest)
    adopt(name, parent, manifest, op)
    for index in range(COUNTS[name]):
        native = p['root'] / (c.legacy.native_name(name, SELECTED[name], index) + '-certificate.json')
        if not native.exists():
            continue
        label = arm_label(name, 'segment', index)
        if (p['root'] / label).exists():
            sub = read(p['root'] / label / 'submission.json')
            require(sub['outcome'] == 'submitted' and sub['args'] == command(name, label, 'segment', manifest),
                    'Existing/unknown production cannot be duplicated')
            continue
        selected_readiness(name, index, op)
        storage_gate(parent, 'dispatch-' + str(index))
        startup(name, 'segment', index, parent, manifest, op)
        record = selected_submit(name, label, 'segment', manifest, None, op)
        submit(name, arm_label(name, 'after-segment', index), 'after-segment', manifest,
               'ended(' + record['job_id'] + ')')


def certify_cpu(manifest, parent):
    proofs = final_job(CONTROL / 'cpu', 'hpc')
    xml = CONTROL / 'cpu/payload/selected.xml'
    count = c.verify_xml(xml, {'test_selected_launch','test_retention'})
    require(count == CASES, 'Exact selected metadata case count differs')
    proofs += [{'path': str(path), 'sha256': sha(path)} for path in (xml, xml.with_suffix('.log'))]
    for name in PROFILES:
        native_units(name)
    c.deferred_absence()
    c.previous_admission()
    exclusive(CONTROL / 'cpu-certificate.json', {'status': 'passed', 'suite': SUITE,
              'cases': count, 'launcher_manifest_sha256': manifest, 'proofs': proofs,
              'original_scientific_cpu_replaced': False})
    # Each independent dispatch verifies its own readiness/admission/startup;
    # there is no chemistry or noise dependency on these jobs.
    # Authenticate/create all initial owners before any selected GPU can exist.
    # This keeps the predecessor-absence audit independent of scheduling races.
    for name in PROFILES:
        adopt(name,parent,manifest,None)
    for name in PROFILES:
        submit(name, 'dispatch-0004', 'dispatch', manifest)
    exclusive(Path(parent)/'independent-noise-preserved.json',{'job_id':'29809228','action':'unchanged_no_resubmission','output':str(c.NOISE_ROOT)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    require(os.environ.get('LSB_JOBID'), 'Allocated job required')
    verify_own(args.manifest)
    executing(args.profile, args.phase, args.parent, args.manifest)
    if args.phase.startswith('noise-'):
        from noisy_entry import execute
        execute(args.phase, args.parent, args.manifest)
        return
    if args.phase == 'cpu':
        storage_gate(args.parent, 'cpu-entry')
        return
    if args.phase == 'after-cpu':
        with (CONTROL / 'dispatch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            certify_cpu(args.manifest, args.parent)
        return
    name = args.profile
    p = profile(name)
    op = load_original(name, args.manifest)
    metadata_certificate(args.manifest)
    if args.phase in ('native', 'segment'):
        ownership(name, args.manifest)
        phase, index = decode(name, args.parent.name)
        produce_with_retention(name,op,args.parent,phase,index,args.manifest)
        return
    with (p['root'] / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.phase == 'dispatch':
            dispatch(name, args.parent, args.manifest, op)
            return
        ownership(name, args.manifest)
        phase, index = decode(name, args.parent.name)
        if name == 'dbmol10k':
            if phase == 'after-native':
                op.after_native(args.parent, index, p['manifest'])
            else:
                op.certify_segment(args.parent, index, p['manifest'])
        elif name == 'boltz30k':
            if phase == 'after-native':
                op.after_native(SELECTED[name], index, p['manifest'], args.parent)
            else:
                op.certify_segment(SELECTED[name], index, p['manifest'], args.parent)
        elif phase == 'after-native':
            op.after_native(SELECTED[name], index, p['manifest'])
        else:
            op.certify_segment(SELECTED[name], index, p['manifest'])


if __name__ == '__main__':
    main()
