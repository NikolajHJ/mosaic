"""Two independent Boltz30k chains; metadata orchestration only before drivers."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from boltz_serial_protocol import ARMS, SEGMENT, TARGET, projection, require_boundary

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = BASE / 'boltz-lr0025-02'
SOURCE = BASE / 'boltz-lr0025-release-02/source'
OLD = BASE / 'restart-lrreset-04'
PRO_ROOT = BASE / 'protenix-extended-01'
PRO_SOURCE = BASE / 'protenix-extended-release-01/source'
PRO_MANIFEST = 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
PRO_CPU = 'c0f784d523d358dfd09d7f57fcfe333fbb5c069d93074681ea35562a5f423171'
DBMOL_ROOT = BASE / 'boltz-dbmol-pilot-01'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
WORKER = '/work3/s234818/latentmol/env/boltz311/bin/python'
PREPARED = Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared')
CONTEXT_FILE = 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
MODEL = '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16'
CONTEXT = 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
OLD_FULL = {'protenix': 1493521824, 'boltz2': 1748021824, 'boltz2_chemistry': 5144021824}
SUITE = 'boltz-two-arm-serial30k-v1'
GIB = 1024**3
NOISY_ROOT = Path('/work3/s234818/mosaic/runs/20261023-protenix-noisy-01')
NOISY_RELEASE = Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01')
NOISY_UPLOAD = Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01.tar.gz')
NOISY_MANIFEST = '7bb970f48f6f907b7dc05e74a9bc2645a313addf814f08a0eeca11582f3b84ef'
NOISY_STAGE = 'experiments/brd4-jq1/protenix-noisy-1000/scripts/stage.py'
NOISY_STAGE_SHA = '48e3adc68cdb4aed0268b83e9f09508ac7836e0aec14c0113b4b441f947dc6fc'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    exclusive(temp, value)
    temp.replace(path)


def config(arm):
    require(arm in ARMS, 'Unknown Boltz arm')
    return SOURCE / 'examples/configs' / ('boltz_serial_' + arm + '.json')


def offset(arm, index):
    require(arm in ARMS and type(index) is int and 0 <= index < 3, 'Undeclared segment')
    return SEGMENT * index


def name(arm, phase, index):
    offset(arm, index)
    require(phase in ('native', 'after-native', 'segment', 'after-segment'), 'Unknown segment phase')
    return f'{arm}-{phase}-{index:04d}'


def decode(parent):
    match = re.fullmatch(r'(boltz2|boltz2_chemistry)-(native|after-native|segment|after-segment)-(\d{4})', Path(parent).name)
    require(match is not None, 'Undeclared Boltz parent')
    arm, phase, index = match.group(1), match.group(2), int(match.group(3))
    offset(arm, index)
    return arm, phase, index


def resources(phase):
    if phase == 'cpu':
        return ['-q', 'hpc', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '01:00']
    if phase in ('native', 'segment'):
        return ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process', '-W', '02:00' if phase == 'native' else '23:00']
    require(phase in ('after-cpu', 'after-native', 'after-segment', 'dispatch'), 'Unknown phase')
    return ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']


def command(label, phase, manifest, dependency=None):
    if phase not in ('cpu', 'after-cpu', 'dispatch'):
        require(decode(label)[1] == phase, 'Parent phase mismatch')
    elif phase == 'dispatch':
        require(re.fullmatch(r'dispatch-\d{4}', label) is not None, 'Dispatch needs a new unique parent')
    else:
        require(label == phase, 'CPU parent mismatch')
    parent = ROOT / label
    args = ['bsub', *resources(phase), '-J', 'mosaic-bol30k-' + label, '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if dependency:
        args += ['-w', dependency]
    wrapper = SOURCE / 'scripts/hpc' / ('boltz_serial_cpu.lsf' if phase == 'cpu' else 'boltz_serial_stage.lsf')
    args += ['bash', '-l', str(wrapper), str(SOURCE), str(ROOT), manifest]
    if phase != 'cpu':
        args += [phase, str(parent)]
    return args


def submit(label, phase, manifest, dependency=None):
    parent = ROOT / label
    args = command(label, phase, manifest, dependency)
    parent.mkdir(exist_ok=False)
    record = {'phase': phase, 'args': args, 'dependency': dependency, 'outcome': 'unknown', 'utc': datetime.now(timezone.utc).isoformat()}
    exclusive(parent / 'submission-started.json', record)
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        ids = re.findall(r'Job <(\d+)>', result.stdout)
        if result.returncode == 0 and len(ids) == 1:
            record.update(outcome='submitted', job_id=ids[0])
    except Exception as error:
        record['error'] = repr(error)
    finally:
        exclusive(parent / 'submission.json', record)
    require(record['outcome'] == 'submitted', 'Unknown submission outcome; no blind duplicate')
    jobs = read(ROOT / 'jobs.json')
    jobs[label] = record['job_id']
    atomic(ROOT / 'jobs.json', jobs)
    result = subprocess.run(['bjobs', '-l', record['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(result.stdout + result.stderr)
    return record


def executing(parent, phase, manifest):
    path = parent / 'submission.json'
    for attempt in range(16):
        if path.exists():
            break
        require(attempt < 15, 'Missing own actual submission')
        time.sleep(2)
    record = read(path)
    require(record['outcome'] == 'submitted' and record['job_id'] == os.environ.get('LSB_JOBID') and record['phase'] == phase and record['args'] == command(parent.name, phase, manifest, record['dependency']), 'Own actual job/resource/source binding differs')


def final_job(parent, queue):
    record = read(parent / 'submission.json')
    job = record['job_id']
    for attempt in range(16):
        result = subprocess.run(['bjobs', '-l', job], capture_output=True, text=True, timeout=30)
        raw = result.stdout + result.stderr
        proof = parent / ('scheduler-final-' + str(attempt) + '.txt')
        proof.write_text(raw)
        compact = re.sub(r'\s+', '', raw)
        require(result.returncode == 0 and re.findall(r'Job<([^>]+)>', compact) == [job] and re.findall(r'Queue<([^>]+)>', compact) == [queue], 'Missing authentic producer scheduler')
        status = re.findall(r'Status<([^>]+)>', compact)
        if status == ['DONE']:
            break
        require(status in (['PEND'], ['RUN']) and attempt < 15, 'Actual producer failed/not final')
        time.sleep(2)
    log = parent / ('stdout.' + job + '.log')
    text = log.read_text()
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$', text, re.M) == [job] and 'Successfully completed.' in text and 'Exited with exit' not in text, 'Same-job LSF success missing')
    return [{'path': str(path), 'sha256': sha(path)} for path in [proof, log, parent / 'submission.json']]


def verify_proofs(certificate):
    require(certificate['status'] == 'passed', 'Failed certificate')
    for proof in certificate['proofs']:
        require(sha(proof['path']) == proof['sha256'], 'Authenticated producer bytes changed')


def cpu_certificate(manifest):
    path = ROOT / 'cpu-certificate.json'
    certificate = read(path)
    require(certificate['source_manifest_sha256'] == manifest and certificate['suite'] == SUITE, 'CPU scope differs')
    verify_proofs(certificate)
    return path


def tree(path):
    import fixed25_lrreset_storage as storage
    return storage.tree_bytes(path)


def protenix_budgets(full_required):
    require(sha(PRO_SOURCE / 'source_manifest.json') == PRO_MANIFEST, 'Existing Protenix641 source differs')
    require(sha(PRO_ROOT / 'cpu-certificate.json') == PRO_CPU, 'Existing Protenix641 CPU certificate differs')
    cpu = read(PRO_ROOT / 'cpu-certificate.json')
    verify_proofs(cpu)
    current = {arm: sum(tree(path) for path in PRO_ROOT.glob(arm + '-*')) for arm in ('random', 'continuation')}
    path = PRO_ROOT / 'projection.json'
    if not path.exists() or set(read(path)) != {'random', 'continuation'}:
        require(not full_required, 'Await authenticated full Protenix641 pilot projections')
        require(not any(PRO_ROOT.glob('*-segment-*')), 'Existing Protenix long has no full projection')
        budgets = {'random': GIB, 'continuation': GIB}
        mode = 'native_allowances_only_not_full_long_capacity'
    else:
        values = read(path)
        budgets = {}
        for arm, updates, segments in [('random', 50000, 5), ('continuation', 40000, 4)]:
            unit = values[arm]['unit_bytes']
            require(type(unit) is int and unit > 0, 'Malformed Protenix pilot projection')
            certs = sorted(PRO_ROOT.glob(arm + '-native-*-certificate.json'))
            require(certs, 'Protenix projection lacks native producer certificate')
            observed = []
            for cert_path in certs:
                cert = read(cert_path)
                verify_proofs(cert)
                require(cert['source_manifest_sha256'] == PRO_MANIFEST, 'Protenix native source differs')
                ready = read(cert_path.with_name(cert_path.name.removesuffix('-certificate.json')) / 'payload/study/assessment.json')
                require(ready['status'] == 'passed' and ready['source_manifest_sha256'] == PRO_MANIFEST, 'Existing Protenix readiness failed')
                observed.append(ready['storage_projection']['maximum_pilot_update_bytes'])
            require(unit == max(observed), 'Protenix projection does not match authenticated pilot maxima')
            budgets[arm] = 2 * unit * updates + segments * GIB
        mode = 'full_empirical_projections'
    require(all(type(current[arm]) is int and 0 <= current[arm] <= budgets[arm] for arm in budgets), 'Existing Protenix output exceeds projection')
    return {'mode': mode, 'full': budgets, 'current': current,
            'remaining': sum(budgets[arm] - current[arm] for arm in budgets)}


def dbmol_budget():
    """One validation allowance until the independent pilot publishes full budget.

    The separate DBMol controller must not submit optimization before both this
    pair and Protenix641 have authenticated full projections. Unknown long markers
    never downgrade a full pilot to its validation allowance.
    """
    path = DBMOL_ROOT / 'projection.json'
    current = tree(DBMOL_ROOT)
    if not path.exists():
        require(not any(DBMOL_ROOT.glob('run*')) and not any(DBMOL_ROOT.glob('*segment*')) and not (DBMOL_ROOT / 'long-submissions.json').exists(), 'DBMol long/unknown outcome lacks authenticated full projection')
        budget = GIB
        mode = 'validation_allowance_only_no_optimization_admission'
    else:
        record = read(path)
        require(record['status'] == 'passed', 'DBMol full projection not passed')
        verify_proofs(record)
        source = BASE / 'boltz-dbmol-pilot-release-01/source'
        manifest = sha(source / 'source_manifest.json')
        require(record['source_manifest_sha256'] == manifest, 'DBMol projection source differs')
        cpu_path = DBMOL_ROOT / 'cpu-certificate.json'
        cpu = read(cpu_path)
        verify_proofs(cpu)
        require(cpu['source_manifest_sha256'] == manifest, 'DBMol CPU scope differs')
        config_sha = sha(source / 'examples/configs/dbmol_pilot.json')
        observed = []
        jobs = []
        for cert_path in sorted(DBMOL_ROOT.glob('pilot-native-*-certificate.json')):
            cert = read(cert_path)
            verify_proofs(cert)
            require(cert['source_manifest_sha256'] == manifest, 'DBMol native certificate source differs')
            producer = cert_path.with_name(cert_path.name.removesuffix('-certificate.json'))
            submission = read(producer / 'submission.json')
            require(submission['outcome'] == 'submitted' and submission['phase'] == 'native', 'DBMol native producer submission missing')
            ready_path = producer / 'payload/study/assessment.json'
            ready = read(ready_path)
            require(any(Path(proof['path']) == ready_path and proof['sha256'] == sha(ready_path) for proof in cert['proofs']), 'DBMol native readiness is not certificate-bound')
            require(ready['status'] == 'passed' and ready['source_manifest_sha256'] == manifest and ready['config_sha256'] == config_sha, 'DBMol readiness/source/config mismatch')
            unit = ready['storage_projection']['maximum_pilot_update_bytes']
            require(type(unit) is int and unit > 0, 'Invalid DBMol pilot unit')
            observed.append(unit)
            jobs.append(submission['job_id'])
        require(observed and record['unit_bytes'] == max(observed) and record['producer_job'] in jobs, 'DBMol projection lacks exact authenticated native maximum/producer')
        budget = record['full_projected_bytes']
        require(type(budget) is int and budget == 2 * max(observed) * 10000 + 4 * GIB, 'Malformed DBMol full four-segment projection')
        mode = 'authenticated_full_pilot_projection'
    require(0 <= current <= budget, 'DBMol output exceeded declared capacity')
    return {'mode': mode, 'full': budget, 'current': current, 'remaining': budget - current}


def storage_gate(parent, phase, full=None, *, existing_full=False):
    import fixed25_lrreset_storage as storage
    path = Path(parent) / ('storage-' + phase)
    path.mkdir(exist_ok=False)
    record = {'status': 'failed', 'phase': phase, 'reserve_bytes': 4 * GIB,
              'file_reserve': 100000, 'scope': 'Original3 remaining +report2GiB/diagnostic512MiB +all641 remaining +both Boltz30k remaining +independent DBMol and noisyProtenix prospective; observations not reservations'}
    try:
        old = {arm: tree(OLD / ('run-' + arm)) for arm in OLD_FULL}
        require(all(0 <= old[arm] <= OLD_FULL[arm] for arm in OLD_FULL), 'Original output exceeded projection')
        own = {arm: sum(tree(p) for p in ROOT.glob(arm + '-*')) for arm in ARMS}
        # CPU phases have one TOTAL1GiB allowance, unchanged originalCPU policy.
        # Two native validation arms retain1GiB each; full longs use pilot totals.
        per_arm = GIB // 2 if phase in ('preflight', 'cpu-entry') else GIB
        budgets = {arm: per_arm for arm in ARMS} if full is None else full
        require(set(budgets) == set(ARMS) and all(type(budgets[arm]) is int and budgets[arm] > 0 and 0 <= own[arm] <= budgets[arm] for arm in ARMS), 'New Boltz projection incomplete/exceeded')
        pro = protenix_budgets(existing_full)
        dbmol = dbmol_budget()
        noisy_current = tree(NOISY_ROOT)
        require(0 <= noisy_current <= GIB, 'Independent noisyProtenix output exceeds declared1GiB')
        noisy_bound = 'pending_validation_allowance_only'
        if NOISY_ROOT.exists():
            manifest_path = NOISY_RELEASE / 'source/source_manifest.json'
            bound = manifest_path.is_file() and sha(manifest_path) == NOISY_MANIFEST
            if bound:
                noisy_manifest = read(manifest_path)
                stage_path = NOISY_RELEASE / 'source' / NOISY_STAGE
                bound = (noisy_manifest['files'].get(NOISY_STAGE) == NOISY_STAGE_SHA
                         and stage_path.is_file() and sha(stage_path) == NOISY_STAGE_SHA)
            require(bound or not existing_full, 'Await authenticated noisyProtenix full persistent-output bound')
            if bound:
                noisy_bound = 'authenticated_immutable_controller_one_GiB_output_cap'
        noisy = {'full':GIB,'current':noisy_current,'remaining':GIB-noisy_current,
                 'reporting':noisy_bound,
                 'scope':'Independent bounded1000-update Protenix pilot prospective allowance'}
        projected = sum(OLD_FULL[a] - old[a] for a in OLD_FULL) + sum(budgets[a] - own[a] for a in ARMS) + pro['remaining'] + dbmol['remaining'] + noisy['remaining'] + 2 * GIB + 512 * 1024**2
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60).stdout
        gate = storage.gate_record('work3', raw, projected, storage.stale_work_bytes() + tree(NOISY_ROOT) + tree(NOISY_RELEASE) + tree(NOISY_UPLOAD), shutil.disk_usage(ROOT).free, storage.inode_observation(os.statvfs(ROOT)))
        record.update(status=gate['status'], old_current=old, own_current=own,
                      full_boltz=budgets, existing_protenix=pro, dbmol=dbmol, noisy_protenix=noisy,
                      projected_bytes=projected, work3=gate)
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        exclusive(path / 'gate.json', record)
    require(record['status'] == 'passed', 'Whole prospective byte/file admission failed')
    return record


def all_budgets():
    values = read(ROOT / 'projection.json')
    require(set(values) == set(ARMS), 'Both fresh pilot projections required')
    manifest = sha(SOURCE / 'source_manifest.json')
    budgets = {}
    for arm in ARMS:
        observed = []
        for certificate_path in sorted(ROOT.glob(arm + '-native-*-certificate.json')):
            certificate = read(certificate_path)
            verify_proofs(certificate)
            require(certificate['source_manifest_sha256'] == manifest, 'Pilot certificate source differs')
            ready_path = certificate_path.with_name(certificate_path.name.removesuffix('-certificate.json')) / 'payload/study/assessment.json'
            require(any(Path(proof['path']) == ready_path and proof['sha256'] == sha(ready_path) for proof in certificate['proofs']), 'Pilot projection assessment is not certificate-bound')
            ready = read(ready_path)
            require(ready['status'] == 'passed' and ready['source_manifest_sha256'] == manifest and ready['config_sha256'] == sha(config(arm)) and ready['arm'] == arm, 'Pilot projection scope differs')
            unit = ready['storage_projection']['maximum_pilot_update_bytes']
            require(type(unit) is int and unit > 0 and ready['storage_projection']['projected_bytes'] == 2 * unit * SEGMENT + GIB, 'Native empirical unit/formula differs')
            observed.append(unit)
        require(observed and values[arm]['unit_bytes'] == max(observed), 'Full projection must match all authenticated native pilot maxima')
        budgets[arm] = projection(max(observed))
    return budgets


def input_snapshot(arm, index):
    if index == 0:
        return None
    certificate = read(ROOT / (name(arm, 'segment', index - 1) + '-certificate.json'))
    verify_proofs(certificate)
    require(certificate['end'] == offset(arm, index) and certificate['source_manifest_sha256'] == sha(SOURCE / 'source_manifest.json'), 'Predecessor global/source identity differs')
    snapshot = certificate['terminal_snapshot']
    require(Path(snapshot['path']).stat().st_size == snapshot['bytes'] and sha(snapshot['path']) == snapshot['sha256'], 'Predecessor exact checkpoint changed')
    return snapshot


def expected_boundary(arm, index, execution):
    expected = input_snapshot(arm, index)
    require(execution['start'] == offset(arm, index) and execution['end'] == offset(arm, index) + SEGMENT and execution['input_snapshot_sha256'] == (None if expected is None else expected['sha256']) and execution['input_snapshot'] == (None if expected is None else expected['path']), 'Declared input checkpoint/global boundary differs')


def native_assessment(arm, index, manifest):
    from campaign25_contract import verify_readiness, verify_frozen_response
    path = ROOT / name(arm, 'native', index) / 'payload/study/assessment.json'
    ready = verify_readiness(path, sha(path), config(arm), manifest)
    summary = read(path.with_name('summary.json'))
    execution = require_boundary(summary, ready)
    expected_boundary(arm, index, execution)
    require(ready['model_digest'] == MODEL and ready['context_digest'] == CONTEXT and ready['pocket_definition_sha256'] == POCKET and ready['diffusion_gradient'] is False, 'Native input/model/context/pocket changed')
    require(summary['status'] == 'passed' and summary['native_hard_parity']['status'] == 'passed' and summary['native_gradient_checks']['status'] == 'passed', 'Native/parity/raw gate incomplete')
    verify_frozen_response(summary['frozen_identity_after'], MODEL, CONTEXT)
    require(summary['context_file_sha256'] == CONTEXT_FILE and summary['native_identity']['checkpoint_sha256'] == 'dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e', 'Worker checkpoint/prepared input changed')
    if arm == 'boltz2_chemistry':
        require(summary['chemistry_assay']['status'] == 'passed', 'Chemistry nonzero signal failed')
        for label in ['low', 'high']:
            require(read(path.parent / (label + '_coupled_checks/report.json'))['status'] == 'passed', 'Required coupled chemistry check failed')
    require(ready['runtime_gate']['status'] == 'passed' and ready['runtime_gate']['conservative_projected_seconds'] < 82800, '10k allocation runtime failed')
    unit = ready['storage_projection']['maximum_pilot_update_bytes']
    require(type(unit) is int and unit > 0 and ready['storage_projection']['projected_bytes'] == 2 * unit * SEGMENT + GIB, 'Pilot projection differs')
    return path, ready


def queue_native(arm, index, manifest):
    record = submit(name(arm, 'native', index), 'native', manifest)
    submit(name(arm, 'after-native', index), 'after-native', manifest, 'ended(' + record['job_id'] + ')')


def certify_cpu(manifest):
    proofs = final_job(ROOT / 'cpu', 'hpc')
    counts = {}
    for group, minimum in [('hpc-contract', 396), ('legacy', 238), ('native-mapping', 33), ('chemistry', 58)]:
        xml = ROOT / 'cpu/payload' / (group + '.xml')
        log = xml.with_suffix('.log')
        parsed = ET.parse(xml)
        cases = list(parsed.iter('testcase'))
        require(len(cases) > minimum and not any(list(parsed.iter(kind)) for kind in ['failure', 'error', 'skipped']), 'Inherited/new CPU group incomplete')
        require(re.search(r'\b' + str(len(cases)) + r'\s+passed\b', log.read_text()), 'Actual raw XML/log count differs')
        if group == 'hpc-contract':
            found = {part for case in cases for part in case.get('classname', '').split('.')}
            wrapper = (SOURCE / 'scripts/hpc/boltz_serial_cpu.lsf').read_text()
            declared = wrapper.split('contracts=(', 1)[1].split(')', 1)[0]
            expected_modules = set(re.findall(r'tests/(test_[A-Za-z0-9_]+)\.py', declared))
            require({'test_boltz_serial', 'test_protenix_initialization'} <= expected_modules and expected_modules <= found and all(case.get('classname') for case in cases), 'Missing declared new/inherited CPU module attribution')
        counts[group] = len(cases)
        for item in [xml, log]:
            proofs.append({'path': str(item), 'sha256': sha(item)})
    require(counts['legacy'] == 239 and counts['native-mapping'] == 34 and counts['chemistry'] == 59, 'Inherited CPU scope differs')
    for item in ['composition_reference.json', 'pocket_definition.json', 'assessment.json', 'oracles.json']:
        path = ROOT / 'cpu/payload' / item
        proofs.append({'path': str(path), 'sha256': sha(path)})
    require(sha(ROOT / 'cpu/payload/pocket_definition.json') == POCKET, 'Pocket changed')
    exclusive(ROOT / 'cpu-certificate.json', {'status': 'passed', 'suite': SUITE, 'source_manifest_sha256': manifest, 'cases': sum(counts.values()), 'groups': counts, 'proofs': proofs})
    storage_gate(ROOT / 'after-cpu', 'native-dispatch')
    for arm in ARMS:
        queue_native(arm, 0, manifest)


def peer_dispatch_dependency():
    """Authenticate existing641 actual native/callback submissions before waiting."""
    require(sha(PRO_SOURCE / 'source_manifest.json') == PRO_MANIFEST and sha(PRO_ROOT / 'cpu-certificate.json') == PRO_CPU, 'Existing641 source/CPU differs')
    verify_proofs(read(PRO_ROOT / 'cpu-certificate.json'))
    jobs = read(PRO_ROOT / 'jobs.json')
    callbacks = []
    for arm in ('random', 'continuation'):
        native_label, after_label = arm + '-native-0000', arm + '-after-native-0000'
        native = read(PRO_ROOT / native_label / 'submission.json')
        after = read(PRO_ROOT / after_label / 'submission.json')
        for label, phase, record in [(native_label, 'native', native), (after_label, 'after-native', after)]:
            require(record['outcome'] == 'submitted' and record['job_id'] == jobs[label] and record['phase'] == phase, 'Existing641 actual producer/job mapping differs')
            dependency = None if phase == 'native' else 'ended(' + native['job_id'] + ')'
            resource = resources('native') if phase == 'native' else resources('after-native')
            expected = ['bsub', *resource, '-J', 'mosaic-pro50k-' + label, '-oo', str(PRO_ROOT / label / 'stdout.%J.log'), '-eo', str(PRO_ROOT / label / 'stderr.%J.log')]
            if dependency:
                expected += ['-w', dependency]
            expected += ['bash', '-l', str(PRO_SOURCE / 'scripts/hpc/extended_stage.lsf'), str(PRO_SOURCE), str(PRO_ROOT), PRO_MANIFEST, phase, str(PRO_ROOT / label)]
            require(record['dependency'] == dependency and record['args'] == expected, 'Existing641 source/resources/own-ended dependency differs')
        callbacks.append(after['job_id'])
    return '&&'.join('done(' + job + ')' for job in callbacks)


def queue_dispatch_watch(manifest):
    label = 'dispatch-0001'
    parent = ROOT / label
    dependency = peer_dispatch_dependency()
    if parent.exists():
        require((parent / 'submission.json').exists(), 'Unknown dispatch-watch outcome blocks duplicate')
        record = read(parent / 'submission.json')
        require(record['outcome'] == 'submitted' and record['args'] == command(label, 'dispatch', manifest, dependency), 'Unknown/mismatched dispatch watcher outcome')
        return record
    return submit(label, 'dispatch', manifest, dependency)


def dispatch(manifest, parent):
    values = read(ROOT / 'projection.json') if (ROOT / 'projection.json').exists() else {}
    if set(values) != set(ARMS):
        exclusive(parent / 'waiting.json', {'status': 'waiting_own_native', 'long_submitted': False})
        return
    existing = PRO_ROOT / 'projection.json'
    if not existing.exists() or set(read(existing)) != {'random', 'continuation'}:
        watcher = queue_dispatch_watch(manifest)
        exclusive(parent / 'waiting.json', {'status': 'waiting_existing_protenix_full_projections', 'long_submitted': False, 'watcher_job': watcher['job_id'], 'scope': 'Validation allowances do not authorize long capacity; unique actual CPU dispatch watcher waits done(each original641 native callback); no callback replay'})
        return
    for arm in ARMS:
        for index in range(3):
            certificate_path = ROOT / (name(arm, 'native', index) + '-certificate.json')
            if not certificate_path.exists():
                continue
            label = name(arm, 'segment', index)
            if (ROOT / label).exists():
                require(read(ROOT / label / 'submission.json')['outcome'] == 'submitted', 'Unknown existing segment outcome')
                continue
            verify_proofs(read(certificate_path))
            native_assessment(arm, index, manifest)
            storage_gate(parent, 'dispatch-' + arm + '-' + str(index), all_budgets(), existing_full=True)
            record = submit(label, 'segment', manifest)
            submit(name(arm, 'after-segment', index), 'after-segment', manifest, 'ended(' + record['job_id'] + ')')


def after_native(arm, index, manifest, parent):
    proofs = final_job(ROOT / name(arm, 'native', index), 'gpul40s')
    path, ready = native_assessment(arm, index, manifest)
    proofs.append({'path': str(path), 'sha256': sha(path)})
    exclusive(ROOT / (name(arm, 'native', index) + '-certificate.json'), {'status': 'passed', 'proofs': proofs, 'source_manifest_sha256': manifest, 'start': offset(arm, index)})
    path = ROOT / 'projection.json'
    values = read(path) if path.exists() else {}
    values[arm] = {'unit_bytes': max(values.get(arm, {}).get('unit_bytes', 0), ready['storage_projection']['maximum_pilot_update_bytes'])}
    atomic(path, values)
    dispatch(manifest, parent)


def certify_segment(arm, index, manifest, parent):
    from campaign25_contract import verify_attempt_rates, verify_frozen_response
    producer = ROOT / name(arm, 'segment', index)
    proofs = final_job(producer, 'gpul40s')
    path = producer / 'payload/study'
    summary = read(path / 'summary.json')
    execution = require_boundary(summary)
    expected_boundary(arm, index, execution)
    require(summary['status'] == 'passed' and all(summary[field]['status'] == 'passed' for field in ['terminal_gradient_checks', 'native_gradient_checks', 'native_hard_parity']), 'Segment scientific acceptance failed')
    require(summary['source_manifest_sha256'] == manifest and summary['config_sha256'] == sha(config(arm)), 'Segment source/config differs')
    verify_frozen_response(summary['frozen_identity_after'], MODEL, CONTEXT)
    case = read(path / 'case-0/report.json')
    require({key: value for key, value in summary['cases'][0].items() if key not in ('label', 'initialization')} == case and case['status'] == 'passed' and len(case['updates']) == SEGMENT and len(case['states']) == SEGMENT + 1, 'Case/summary/full budget mismatch')
    verify_attempt_rates(case, read(config(arm)))
    for field in ['states', 'updates']:
        count = 0
        with (path / 'case-0' / (field + '.jsonl')).open() as stream:
            for number, line in enumerate(stream):
                record = json.loads(line)
                require(line.endswith('\n') and number < len(case[field]) and record == case[field][number] and record['step'] == number and record['global_step'] == offset(arm, index) + number, 'Committed global/local JSONL differs')
                count += 1
        require(count == len(case[field]), 'Incomplete committed stream')
        item = path / 'case-0' / (field + '.jsonl')
        proofs.append({'path': str(item), 'sha256': sha(item)})
    for item in [path / 'summary.json', path / 'case-0/report.json', path / 'terminal_gradient_checks/report.json']:
        proofs.append({'path': str(item), 'sha256': sha(item)})
    for state in case['states']:
        item = path / 'case-0' / state['snapshot']
        require(item.is_file() and not item.is_symlink() and item.resolve().is_relative_to((path / 'case-0').resolve()) and sha(item) == state['snapshot_sha256'], 'Saved state snapshot path/bytes differ')
        proofs.append({'path': str(item), 'sha256': sha(item)})
    state = case['states'][-1]
    snapshot = path / 'case-0' / state['snapshot']
    require(sha(snapshot) == state['snapshot_sha256'], 'Terminal snapshot bytes differ')
    exclusive(ROOT / (name(arm, 'segment', index) + '-certificate.json'), {'status': 'passed', 'source_manifest_sha256': manifest, 'start': offset(arm, index), 'end': offset(arm, index) + SEGMENT, 'proofs': proofs, 'terminal_snapshot': {'path': str(snapshot), 'sha256': sha(snapshot), 'bytes': snapshot.stat().st_size}})
    if index < 2:
        storage_gate(parent, 'next-native', all_budgets(), existing_full=True)
        queue_native(arm, index + 1, manifest)
    else:
        chain = [read(ROOT / (name(arm, 'segment', number) + '-certificate.json')) for number in range(3)]
        for certificate in chain:
            verify_proofs(certificate)
        require([value['start'] for value in chain] == [0, 10000, 20000] and chain[-1]['end'] == TARGET, 'Full scientific chain incomplete')
        exclusive(ROOT / (arm + '-completion.json'), {'status': 'passed', 'arm': arm, 'target_global_updates': TARGET, 'new_updates': TARGET, 'segments': chain, 'original_results_unchanged': True})


def produce(parent, phase, arm, index, manifest):
    certificate = cpu_certificate(manifest)
    snapshot = input_snapshot(arm, index)
    ready = None
    if phase == 'segment':
        ready, _ = native_assessment(arm, index, manifest)
    complete = (ROOT / 'projection.json').exists() and set(read(ROOT / 'projection.json')) == set(ARMS)
    storage_gate(parent, 'entry', all_budgets() if complete else None, existing_full=phase == 'segment')
    require(sha(PREPARED / 'context.pt') == CONTEXT_FILE, 'Original prepared context changed')
    output = parent / 'payload/study'
    output.parent.mkdir(exist_ok=False)
    args = [PYTHON, str(SOURCE / 'examples/brd4_25_atoms.py'), '--mode', 'validate' if phase == 'native' else 'run', '--config', str(config(arm)), '--output', str(output), '--manifest', manifest, '--cpu-certificate', str(certificate), '--cpu-certificate-sha256', sha(certificate), '--reference', str(ROOT / 'cpu/payload/composition_reference.json'), '--pocket', str(ROOT / 'cpu/payload/pocket_definition.json'), '--pocket-sha256', POCKET, '--prepared', str(PREPARED), '--context-sha256', CONTEXT_FILE, '--worker-python', WORKER, '--offset', str(offset(arm, index)), '--segment-steps', str(SEGMENT)]
    if snapshot:
        args += ['--resume-snapshot', snapshot['path'], '--resume-sha256', snapshot['sha256']]
    if ready:
        args += ['--ready', str(ready), '--ready-sha256', sha(ready)]
    exclusive(parent / 'payload/driver-invocation.json', {'args': args, 'original_saved_runs_read_only': True, 'jax': 'cpu_fp32', 'worker': 'cuda_frozen_affinity'})
    subprocess.run(args, cwd=SOURCE, check=True, timeout=6600 if phase == 'native' else 81000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    require(args.root.resolve() == ROOT and args.source.resolve() == SOURCE and os.environ.get('LSB_JOBID'), 'Wrong/unallocated execution')
    parent = ROOT / 'cpu' if args.phase == 'cpu-entry' else args.parent
    require(parent is not None and parent.resolve().is_relative_to(ROOT), 'Parent escaped')
    executing(parent, 'cpu' if args.phase == 'cpu-entry' else args.phase, args.manifest)
    if args.phase == 'cpu-entry':
        storage_gate(parent, 'cpu-entry')
        return
    if args.phase in ('after-cpu', 'dispatch'):
        with (ROOT / 'dispatch.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if args.phase == 'after-cpu':
                certify_cpu(args.manifest)
            else:
                cpu_certificate(args.manifest)
                dispatch(args.manifest, parent)
        return
    arm, phase, index = decode(parent)
    if phase in ('native', 'segment'):
        produce(parent, phase, arm, index, args.manifest)
    else:
        cpu_certificate(args.manifest)
        with (ROOT / 'dispatch.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if phase == 'after-native':
                after_native(arm, index, args.manifest, parent)
            elif phase == 'after-segment':
                certify_segment(arm, index, args.manifest, parent)
            else:
                raise ValueError('Unknown phase')


if __name__ == '__main__':
    main()
