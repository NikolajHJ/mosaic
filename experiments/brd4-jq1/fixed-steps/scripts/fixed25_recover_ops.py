"""Two-arm storage recovery; the existing Protenix trajectory is never changed.

Only authenticated, scientifically unstarted Boltz admissions may be replaced.
Original scientific source02/readiness and source03 launcher evidence stay intact.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import fixed25_recover_storage as storage


ROOT = storage.ROOT
TOOL_SOURCE = Path('/work3/s234818/mosaic/releases') / storage.TAG / 'source'
SOURCE03 = Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-storage-03/source')
ROOT03 = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-03')
SCIENCE = Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source')
SCIENCE_MANIFEST = '1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64'
SOURCE03_MANIFEST = '9ee7bf1d6bf412acdc37fc83415476dbefdf45db89533571d4a646ee16f33772'
SOURCE03_CPU = '4b2b9f59d22719771528c9439dfbd338fec3a02922df08190b52bb22b0875bd4'
SOURCE03_ADOPTION = '7592facd2832058fa81055c72aaa64f06213bb369c68cb39a0d693b1b5496a60'
PRO_JOB = '29777709'
PRO_SUBMISSION = '088d768ef618b66c6e38916c76cab4d1da44a690cfba34976dfa27513f3e634d'
PRO_ENTRY = 'b0f7009c68a0e49d203f34fa82f9636f4a85e7cac1e7a60c84ee747e586ae75d'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
NEW_ARMS = ('boltz2', 'boltz2_chemistry')
SUITE = 'fixed25-two-arm-recovery-contract-v1'
VERSION = 'fixed25-two-arm-storage-recovery-v1'
FAILURES = {
    'boltz2': {'job': '29777710', 'scheduler': '75b32202c77921b8daa25b146a3476cf1c63bb615c0bd77e316b63eeaee91813',
               'stdout': '70c37fa20439f79a7e59d1438a50c8c95cf2b41c0fece85d3c46015cba14e40b',
               'quota': 'd58e107ab49748c7f7bc5cd3aff50156d84b61df927c5e7e06175e262b9aeafb'},
    'boltz2_chemistry': {'job': '29777711', 'scheduler': '4ecb81967b07e16c6f6dea6c958cffa602d7838c8ec46e1b68399b44f942c18d',
                         'stdout': 'f07eb0a06b400caa496aa7f11f18ffd7e1d5cae4facad986e4f60cac42543958',
                         'quota': '1b6f07a120dc804f192c70de5fed2556e29e483292e5e271337202f0e4230fc4'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exclusive(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_release(source, manifest):
    require(isinstance(manifest, str) and re.fullmatch('[a-f0-9]{64}', manifest), 'Missing manifest identity')
    subprocess.run([sys.executable, str(_layout_path(SCIENCE / 'scripts/verify_stage0_release.py')), str(source), manifest],
                   check=True, timeout=180)


def hashed(path, expected=None):
    path = Path(path)
    require(path.resolve() == path.absolute() and path.is_file() and not path.is_symlink(), 'Unsafe evidence path')
    value = sha(path)
    require(expected is None or value == expected, 'Persisted evidence changed: ' + str(path))
    return {'path': str(path), 'sha256': value}


def legacy_ops():
    """Load the authenticated source03 implementation, never an unsealed overlay."""
    check_release(SOURCE03, SOURCE03_MANIFEST)
    for name in ('fixed25_split_storage', 'fixed25_split_ops'):
        module = sys.modules.get(name)
        if module is not None:
            require(Path(module.__file__).resolve() == _layout_path(SOURCE03 / 'scripts' / (name + '.py')),
                    'Original launcher module was shadowed')
    previous = list(sys.path)
    try:
        sys.path[:0] = [str(_layout_path(SOURCE03 / 'scripts')), str(_layout_path(SCIENCE / 'scripts'))]
        spec = importlib.util.spec_from_file_location('_fixed25_authenticated_source03_ops', _layout_path(SOURCE03 / 'scripts/fixed25_split_ops.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def scheduler_status(raw, job, queue):
    compact = re.sub(r'\s+', '', raw)
    jobs = re.findall(r'Job<([^>]+)>', compact)
    statuses = re.findall(r'Status<([^>]+)>', compact)
    queues = re.findall(r'Queue<([^>]+)>', compact)
    require(jobs == [job] and len(statuses) == 1 and queues == [queue], 'Actual scheduler identity/queue missing')
    require(statuses[0] in ('RUN', 'PEND', 'DONE', 'EXIT'), 'Unknown actual scheduler status')
    return statuses[0]


def lsf_outcome(path, job, success):
    text = Path(path).read_text()
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', text, re.M)
    require(len(subjects) == 1 and subjects[0][0] == job, 'LSF subject job differs')
    if success:
        require(subjects[0][1].rstrip().endswith(' Done') and re.search(r'^Successfully completed\.\s*$', text, re.M)
                and not re.search(r'Exited\s+(?:with exit|by signal)', text, re.I), 'Missing same-job LSF success')
    else:
        require(subjects[0][1].rstrip().endswith(' Exited') and re.search(r'^Exited with exit code 1\.\s*$', text, re.M)
                and 'Successfully completed.' not in text, 'Failed admission is not actual LSF exit1')


def failed_admission(arm):
    require(arm in NEW_ARMS, 'Only the two scientifically unstarted Boltz arms may be recovered')
    pin = FAILURES[arm]
    parent = ROOT03 / ('run-' + arm)
    scheduler = ROOT03 / 'entry-failures-01' / ('run-' + arm) / 'actual-final-scheduler.txt'
    stdout = parent / ('stdout.' + pin['job'] + '.log')
    quota = ROOT03 / ('entry-quota-' + arm) / 'split-storage-gate.json'
    proofs = [hashed(scheduler, pin['scheduler']), hashed(stdout, pin['stdout']), hashed(quota, pin['quota'])]
    require(scheduler_status(scheduler.read_text(), pin['job'], 'gpul40s') == 'EXIT', 'Old Boltz admission did not EXIT')
    lsf_outcome(stdout, pin['job'], False)
    gate = read(quota)
    require(gate['status'] == 'failed' and gate['phase'] == 'entry' and gate['projections'] == storage.FULL_PROJECTIONS,
            'Historical failed entry storage gate changed')
    budget = gate['filesystems']['work3']
    require(budget['status'] == 'failed' and budget['available_bytes_conservative'] < budget['required_bytes']
            and budget['reserve_bytes'] == 4 * 1024**3, 'Old entry failure was not preserved byte headroom rejection')
    for name in ('study', 'driver-invocation.json', 'gpu.txt', 'native.log'):
        require(not (parent / 'payload' / name).exists(), 'Old Boltz scientific execution exists; no fresh replacement authorized')
    return {'arm': arm, 'job': pin['job'], 'status': 'EXIT', 'scientifically_unstarted': True, 'proofs': proofs}


def existing_pro_identity(old):
    parent = ROOT03 / 'run-protenix'
    submission = parent / 'submission.json'
    invocation = parent / 'payload/driver-invocation.json'
    quota = ROOT03 / 'entry-quota-protenix/split-storage-gate.json'
    proofs = [hashed(submission, PRO_SUBMISSION), hashed(quota, PRO_ENTRY), hashed(invocation)]
    receipt, actual = read(submission), read(invocation)
    require(receipt['job_id'] == PRO_JOB and receipt['arm'] == 'protenix' and receipt['outcome'] == 'submitted'
            and receipt['tool_manifest_sha256'] == SOURCE03_MANIFEST and receipt['science_manifest_sha256'] == SCIENCE_MANIFEST,
            'Existing Protenix submission identity changed')
    require(actual == {'args': old.driver_command('protenix', python=PYTHON), 'cwd': str(SCIENCE),
                       'tool_manifest_sha256': SOURCE03_MANIFEST, 'science_manifest_sha256': SCIENCE_MANIFEST},
            'Existing Protenix scientific invocation differs')
    require(read(quota)['status'] == 'passed', 'Existing Protenix did not pass its actual entry gate')
    return {'job': PRO_JOB, 'parent': str(parent), 'proofs': proofs,
            'policy': 'Existing trajectory only; recovery cannot submit, cancel, resume or restart it'}


def original_evidence():
    old = legacy_ops()
    old.verify_tool_cpu(SOURCE03_MANIFEST, SOURCE03_CPU)
    adoption = old.verify_adoption(SOURCE03_ADOPTION)
    return {'source03_manifest_sha256': SOURCE03_MANIFEST,
            'source03_cpu_certificate': hashed(ROOT03 / 'cpu-certificate.json', SOURCE03_CPU),
            'source03_adoption': hashed(ROOT03 / 'adoption.json', SOURCE03_ADOPTION),
            'original': adoption['original'],
            'failed_admissions': {arm: failed_admission(arm) for arm in NEW_ARMS},
            'existing_protenix': existing_pro_identity(old)}


def capture_adoption():
    record = {'version': VERSION, 'status': 'passed', 'evidence': original_evidence(),
              'placement': storage.PLACEMENT, 'outputs': {arm: str(storage.new_output_parent(arm)) for arm in NEW_ARMS},
              'scope': 'Evidence-only two-arm admission recovery; scientific settings/readiness and existing Protenix unchanged'}
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'adoption.json'
    if path.exists():
        require(read(path) == record, 'Existing immutable recovery adoption changed')
    else:
        exclusive(path, record)
    return record


def verify_adoption(expected):
    path = ROOT / 'adoption.json'
    hashed(path, expected)
    record = read(path)
    require(record.get('version') == VERSION and record.get('status') == 'passed'
            and record.get('evidence') == original_evidence() and record.get('placement') == storage.PLACEMENT
            and record.get('outputs') == {a: str(storage.new_output_parent(a)) for a in NEW_ARMS},
            'Recovery adoption no longer authenticates identical original evidence')
    return record


def verify_tool_cpu(manifest, expected):
    path = ROOT / 'cpu-certificate.json'
    hashed(path, expected)
    cert = read(path)
    job = cert.get('producer_job_id')
    require(cert.get('status') == 'passed' and cert.get('suite') == SUITE and cert.get('source_manifest_sha256') == manifest
            and isinstance(job, str) and job.isdigit() and type(cert.get('tests')) is int and cert['tests'] > 0,
            'Recovery CPU certificate status/source/suite/count missing')
    proofs = cert['proofs']
    require(len(proofs) == len({p['path'] for p in proofs}), 'Duplicated recovery CPU proofs')
    for proof in proofs:
        p = Path(proof['path'])
        require(p.resolve().is_relative_to(ROOT / 'cpu'), 'Recovery CPU proof escaped its root')
        hashed(p, proof['sha256'])
    schedulers = [Path(p['path']) for p in proofs if re.fullmatch(r'scheduler-final-\d+\.txt', Path(p['path']).name)]
    logs = [Path(p['path']) for p in proofs if Path(p['path']).name == 'stdout.' + job + '.log']
    xmls = [Path(p['path']) for p in proofs if Path(p['path']).suffix == '.xml']
    require(len(schedulers) == len(logs) == len(xmls) == 1, 'Recovery CPU needs actual final/LSF/raw XML')
    require(scheduler_status(schedulers[0].read_text(), job, 'hpc') == 'DONE', 'Recovery CPU has not actually completed')
    lsf_outcome(logs[0], job, True)
    result = ET.parse(xmls[0]).getroot()
    cases = list(result.iter('testcase'))
    require(len(cases) == cert['tests'] and {c.get('classname', '').split('.')[-1] for c in cases}
            == {'test_fixed25_recover_storage', 'test_fixed25_recover_ops'}
            and not any(list(result.iter(tag)) for tag in ('failure', 'error', 'skipped')),
            'Both complete recovery CPU modules must pass without failures/errors/skips')
    for suite in list(result.iter('testsuite')) + list(result.iter('testsuites')):
        require(all(int(suite.get(k, '0')) == 0 for k in ('failures', 'errors', 'skipped'))
                and ('tests' not in suite.attrib or int(suite.get('tests')) == len(list(suite.iter('testcase')))),
                'Recovery CPU raw suite counts differ')
    return cert


def canonical_digest(value):
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value):
        digest.update(part.encode())
    return digest.hexdigest()


def pro_terminal_proofs():
    """Bounded-memory saved metadata authentication; final report still checks frames."""
    study = ROOT03 / 'run-protenix/payload/study'
    summary = read(study / 'summary.json')
    old = legacy_ops()
    pin = old.NATIVE_PINS['protenix']
    require(summary['status'] == 'passed' and summary['job_id'] == PRO_JOB and summary['mode'] == 'run'
            and summary['source_manifest_sha256'] == SCIENCE_MANIFEST and summary['config_sha256'] == pin['config']
            and summary['optimization_parameters'] == {'step_size': 0.3, 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'}
            and summary['pocket_validation_status'] == 'passed' and summary['pocket_definition_sha256'] == old.POCKET
            and summary['diffusion_gradient'] is False,
            'Existing Protenix terminal source/config/pocket/policy failed')
    source_files = read(SCIENCE / 'source_manifest.json')['files']
    require(summary['source']['files'] and all(source_files.get(k) == v for k, v in summary['source']['files'].items()),
            'Existing Protenix executed source file map differs')
    require(summary['frozen_identity_after'] == {'model_digest': summary['model_digest'], 'context_digest': summary['context_digest']}
            and summary['model_digest'] == '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2'
            and summary['context_digest'] == 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4',
            'Existing Protenix terminal frozen identity changed')
    require(summary['native_hard_parity']['status'] == 'passed'
            and all(summary[k]['status'] == 'passed' for k in ('native_gradient_checks', 'terminal_gradient_checks'))
            and len(summary['cases']) == 1, 'Existing Protenix terminal native gates failed')
    config_path = _layout_path(SCIENCE / 'examples/configs/brd4_jq1_25_fixed_protenix_10000.json')
    config = read(config_path)
    require(sha(config_path) == pin['config'] and summary['config'] == config, 'Existing Protenix embedded configuration differs')
    wrapped = summary['cases'][0]
    require(wrapped.get('label') == config['label'] and wrapped.get('initialization') == config['initialization'],
            'Existing Protenix case label/initialization wrapper differs')
    summary_case_digest = canonical_digest({k: v for k, v in wrapped.items() if k not in ('label', 'initialization')})
    del wrapped
    del summary
    case = read(study / 'case-0/report.json')
    require('label' not in case and 'initialization' not in case and canonical_digest(case) == summary_case_digest
            and case['status'] == 'passed' and case['requested_updates'] == 10000
            and len(case['updates']) == 10000 and len(case['states']) == 10001, 'Existing Protenix terminal budget/case mismatch')
    for field in ('updates', 'states'):
        for i, row in enumerate(case[field]):
            require(row['step'] == i, 'Existing Protenix terminal continuity failed')
        with (study / ('case-0/' + field + '.jsonl')).open() as stream:
            seen = 0
            for seen, line in enumerate(stream, 1):
                require(seen <= len(case[field]) and line.endswith('\n') and json.loads(line) == case[field][seen - 1],
                        'Existing Protenix JSONL differs from terminal case')
            require(seen == len(case[field]), 'Existing Protenix JSONL budget differs')
    require(case['counts']['applied'] == 10000 and case['counts']['forced_fallback'] == 0,
            'Existing Protenix fixed update counts failed')
    require(case['counts']['strict_improving'] == sum(bool(u['attempts'][u['selected_trial']]['strict']) for u in case['updates'])
            and case['counts']['no_op'] == sum(bool(u['no_op']) for u in case['updates'])
            and all(s['repair_weight'] == 0 for s in case['states']), 'Existing Protenix terminal bookkeeping differs')
    for update in case['updates']:
        require(update['selection_policy'] == 'fixed_step' and len(update['attempts']) == 1 and update['attempts'][0]['rate'] == 0.3
                and update['accepted_step_size'] == 0.3 and update['selected_trial'] == 0 and update['fallback_used'] is False,
                'Existing Protenix actual fixed rates changed')
    initial_checks = case['initial_checks']
    del case
    names = ['summary.json', 'case-0/report.json', 'case-0/updates.jsonl', 'case-0/states.jsonl',
             'native_gradient_checks/report.json', 'terminal_gradient_checks/report.json', 'case-0/combined_initial_checks/report.json']
    coupled = read(study / names[-1])
    require(coupled['status'] == 'passed' and coupled['directions'] == initial_checks,
            'Existing Protenix coupled terminal binding failed')
    return [hashed(study / name) for name in names]


def pro_status():
    """An EXIT blocks recovery; absent live history requires authentic persisted DONE."""
    result = subprocess.run(['bjobs', '-l', PRO_JOB], capture_output=True, text=True, timeout=30)
    raw = result.stdout + result.stderr
    ROOT.mkdir(parents=True, exist_ok=True)
    observed = ROOT / ('pro-status-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.txt')
    with observed.open('x') as stream:
        stream.write(raw)
    if result.returncode == 0:
        status = scheduler_status(raw, PRO_JOB, 'gpul40s')
        require(status in ('RUN', 'DONE'), 'Existing Protenix is not RUN or successful DONE; recovery blocked')
        if status == 'DONE':
            lsf_outcome(ROOT03 / ('run-protenix/stdout.' + PRO_JOB + '.log'), PRO_JOB, True)
            pro_terminal_proofs()
        return status
    record_path = ROOT / 'pro-terminal-observation.json'
    require(record_path.exists(), 'Live Protenix outcome unavailable and no authenticated terminal observation')
    record = read(record_path)
    require(record['status'] == 'observed_success' and record['native_job'] == PRO_JOB,
            'Existing Protenix terminal observation failed')
    require(record.get('version') == VERSION and record.get('tool_manifest_sha256') == sha(TOOL_SOURCE / 'source_manifest.json')
            and record.get('cpu_certificate_sha256') == sha(ROOT / 'cpu-certificate.json')
            and record.get('adoption_sha256') == sha(ROOT / 'adoption.json'), 'Protenix observer proof provenance differs')
    require(record['scheduler'] in record['proofs'] and record['stdout'] in record['proofs']
            and Path(record['scheduler']['path']).is_relative_to(ROOT / 'pro-observer')
            and record['stdout']['path'] == str(ROOT03 / ('run-protenix/stdout.' + PRO_JOB + '.log')),
            'Persisted Protenix observation scheduler/LSF proof omitted or escaped')
    for proof in record['proofs']:
        hashed(proof['path'], proof['sha256'])
    require(scheduler_status(Path(record['scheduler']['path']).read_text(), PRO_JOB, 'gpul40s') == 'DONE',
            'Persisted Protenix observation lacks actual DONE')
    lsf_outcome(record['stdout']['path'], PRO_JOB, True)
    pro_terminal_proofs()
    return 'DONE'


def observer_command(tool_manifest, cpu_sha, adoption_sha):
    parent = ROOT / 'pro-observer'
    return ['bsub', '-q', 'hpc', '-n', '1', '-R', 'span[hosts=1]', '-R', 'rusage[mem=2GB]', '-W', '00:30',
            '-J', 'mosaic-jq1-fixed-observe-pro', '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log'),
            '-w', 'ended(' + PRO_JOB + ')', 'bash', '-l', str(_layout_path(TOOL_SOURCE / 'scripts/hpc/fixed25_recover_observe_pro.lsf')),
            str(TOOL_SOURCE), str(ROOT), tool_manifest, cpu_sha, adoption_sha]


def submit_observer(tool_manifest, cpu_sha, adoption_sha):
    parent, receipt_path = ROOT / 'pro-observer', ROOT / 'observer-submission.json'
    command = observer_command(tool_manifest, cpu_sha, adoption_sha)
    if receipt_path.exists():
        record = read(receipt_path)
        require(record.get('outcome') == 'submitted' and record.get('producer_job') == PRO_JOB
                and record.get('tool_manifest_sha256') == tool_manifest and record.get('cpu_certificate_sha256') == cpu_sha
                and record.get('adoption_sha256') == adoption_sha and record.get('args') == command
                and read(parent / 'submission.json') == record, 'Previous observer submission unknown/different')
        return record
    marker = ROOT / 'pro-observer-submission-started.json'
    require(not marker.exists() and not parent.exists(), 'Previous observer submission exists/unknown; no duplicate')
    parent.mkdir(exist_ok=False)
    exclusive(marker, {'args': command, 'producer_job': PRO_JOB})
    record = submission_reply(command, {'phase': 'observe-pro', 'producer_job': PRO_JOB,
        'tool_manifest_sha256': tool_manifest, 'cpu_certificate_sha256': cpu_sha, 'adoption_sha256': adoption_sha})
    exclusive(parent / 'submission.json', record)
    exclusive(receipt_path, record)
    require(record['outcome'] == 'submitted', 'Observer submission unknown/failed; do not duplicate')
    actual = subprocess.run(['bjobs', '-l', record['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(actual.stdout + actual.stderr)
    return record


def submission_reply(command, record):
    record = dict(record, utc=datetime.now(timezone.utc).isoformat(), args=command, outcome='unknown')
    try:
        reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
        record.update(stdout=reply.stdout, stderr=reply.stderr, returncode=reply.returncode)
        jobs = re.findall(r'Job <(\d+)>', reply.stdout)
        if reply.returncode == 0 and len(jobs) == 1:
            record.update(outcome='submitted', job_id=jobs[0])
    except Exception as error:
        record['error'] = repr(error)
    return record


def submission_command(arm, manifest, cpu_sha, adoption_sha):
    require(arm in NEW_ARMS, 'Existing Protenix can never be submitted by recovery')
    parent = storage.new_output_parent(arm)
    return ['bsub', '-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]', '-W', '23:00',
            '-J', 'mosaic-jq1-fixed-recover-' + arm, '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log'),
            '-gpu', 'num=1:mode=exclusive_process', 'bash', '-l', str(_layout_path(TOOL_SOURCE / 'scripts/hpc/fixed25_recover_stage.lsf')),
            str(TOOL_SOURCE), manifest, str(ROOT), arm, cpu_sha, adoption_sha]


def submit(arm, manifest, cpu_sha, adoption_sha):
    command = submission_command(arm, manifest, cpu_sha, adoption_sha)
    parent = storage.new_output_parent(arm)
    require(parent.resolve() == parent.absolute(), 'New output parent alias is unsupported')
    parent.mkdir(parents=True, exist_ok=False)
    exclusive(parent / 'submission-started.json', {'args': command, 'arm': arm})
    record = submission_reply(command, {'arm': arm, 'phase': 'run', 'output_parent': str(parent),
        'tool_manifest_sha256': manifest, 'science_manifest_sha256': SCIENCE_MANIFEST})
    exclusive(parent / 'submission.json', record)
    require(record['outcome'] == 'submitted', 'Unknown/failed recovery submission; no blind duplicate')
    actual = subprocess.run(['bjobs', '-l', record['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(actual.stdout + actual.stderr)
    return record


def dispatch(manifest, cpu_sha, adoption_sha):
    check_release(TOOL_SOURCE, manifest)
    adoption = verify_adoption(adoption_sha)
    verify_tool_cpu(manifest, cpu_sha)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(not (ROOT / 'long-dispatch-started.json').exists() and not (ROOT / 'long-submissions.json').exists(),
                'Recovery dispatch exists/unknown; no duplicates')
        require(not any(storage.new_output_parent(a).exists() for a in NEW_ARMS), 'Recovery output exists/unknown')
        observer = submit_observer(manifest, cpu_sha, adoption_sha)
        status = pro_status()
        projections = {a: p['projected_bytes'] for a, p in adoption['evidence']['original']['arms'].items()}
        quota_path = ROOT / ('quota-dispatch-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        gate = storage.check_campaign_storage(projections, quota_path, phase='dispatch')
        exclusive(ROOT / 'long-dispatch-started.json', {'version': VERSION, 'tool_manifest_sha256': manifest,
            'science_manifest_sha256': SCIENCE_MANIFEST, 'cpu_certificate_sha256': cpu_sha, 'adoption_sha256': adoption_sha,
            'placement': storage.PLACEMENT, 'existing_protenix_job': PRO_JOB, 'existing_protenix_status': status,
            'observer_submission': observer, 'storage_gate': gate})
        jobs = []
        for arm in NEW_ARMS:
            jobs.append(submit(arm, manifest, cpu_sha, adoption_sha))
            temporary = ROOT / 'long-submissions.json.tmp'
            temporary.write_text(json.dumps({'jobs': jobs, 'both_boltz_submitted': len(jobs) == 2,
                'existing_protenix_job': PRO_JOB, 'observer_submission': observer,
                'tool_manifest_sha256': manifest, 'science_manifest_sha256': SCIENCE_MANIFEST}, indent=2) + '\n')
            temporary.replace(ROOT / 'long-submissions.json')
        return {'long_jobs': jobs, 'observer': observer, 'existing_protenix_job': PRO_JOB}


def driver_command(arm):
    require(arm in NEW_ARMS, 'Recovery cannot run or resume Protenix')
    command = legacy_ops().driver_command(arm, python=PYTHON)
    require(command.count('--output') == 1, 'Original driver output parameter ambiguous')
    command[command.index('--output') + 1] = str(storage.new_output_parent(arm) / 'payload/study')
    return command


def run_arm(arm, manifest, cpu_sha, adoption_sha):
    require(arm in NEW_ARMS, 'Recovery cannot execute Protenix')
    check_release(TOOL_SOURCE, manifest)
    adoption = verify_adoption(adoption_sha)
    verify_tool_cpu(manifest, cpu_sha)
    pro_status()
    marker = read(ROOT / 'long-dispatch-started.json')
    require(all(marker.get(k) == v for k, v in {'version': VERSION, 'tool_manifest_sha256': manifest,
        'science_manifest_sha256': SCIENCE_MANIFEST, 'cpu_certificate_sha256': cpu_sha,
        'adoption_sha256': adoption_sha, 'placement': storage.PLACEMENT, 'existing_protenix_job': PRO_JOB}.items()),
        'GPU entry differs from authenticated recovery dispatch')
    parent = storage.new_output_parent(arm)
    receipt = parent / 'submission.json'
    for _ in range(15):
        if receipt.exists():
            break
        time.sleep(1)
    record = read(receipt)
    require(record.get('outcome') == 'submitted' and record.get('job_id') == os.environ.get('LSB_JOBID')
            and record.get('arm') == arm and record.get('output_parent') == str(parent)
            and record.get('science_manifest_sha256') == SCIENCE_MANIFEST and record.get('tool_manifest_sha256') == manifest
            and record.get('args') == submission_command(arm, manifest, cpu_sha, adoption_sha),
            'GPU entry actual job/submission args/source/output mismatch')
    payload = parent / 'payload'
    payload.mkdir(exist_ok=False)
    projections = {a: p['projected_bytes'] for a, p in adoption['evidence']['original']['arms'].items()}
    storage.check_campaign_storage(projections, ROOT / ('entry-quota-' + arm), phase='entry')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                         capture_output=True, text=True, check=True, timeout=30)
    (payload / 'gpu.txt').write_text(gpu.stdout)
    command = driver_command(arm)
    exclusive(payload / 'driver-invocation.json', {'args': command, 'cwd': str(SCIENCE),
        'tool_manifest_sha256': manifest, 'science_manifest_sha256': SCIENCE_MANIFEST})
    with (payload / 'native.log').open('x') as log:
        subprocess.run(command, cwd=SCIENCE, stdout=log, stderr=subprocess.STDOUT, check=True)


def observe_pro(manifest, cpu_sha, adoption_sha):
    check_release(TOOL_SOURCE, manifest)
    verify_adoption(adoption_sha)
    verify_tool_cpu(manifest, cpu_sha)
    parent = ROOT / 'pro-observer'
    record = {'version': VERSION, 'status': 'failed', 'native_job': PRO_JOB, 'observer_job': os.environ.get('LSB_JOBID'),
              'tool_manifest_sha256': manifest, 'cpu_certificate_sha256': cpu_sha, 'adoption_sha256': adoption_sha,
              'scope': 'Actual terminal observation and saved metadata only; final scientific frame report remains separate'}
    try:
        receipt = read(ROOT / 'observer-submission.json')
        require(receipt['outcome'] == 'submitted' and receipt['job_id'] == os.environ.get('LSB_JOBID')
                and receipt['producer_job'] == PRO_JOB and receipt['tool_manifest_sha256'] == manifest
                and receipt['cpu_certificate_sha256'] == cpu_sha and receipt['adoption_sha256'] == adoption_sha,
                'Protenix observer differs from its actual submission')
        for attempt in range(7):
            result = subprocess.run(['bjobs', '-l', PRO_JOB], capture_output=True, text=True, timeout=30)
            raw = result.stdout + result.stderr
            path = parent / ('native-scheduler-' + str(attempt) + '.txt')
            with path.open('x') as stream:
                stream.write(raw)
            require(result.returncode == 0, 'Actual Protenix terminal scheduler unavailable')
            status = scheduler_status(raw, PRO_JOB, 'gpul40s')
            require(status != 'EXIT', 'Existing Protenix failed; recovery blocked')
            if status == 'DONE':
                try:
                    lsf_outcome(ROOT03 / ('run-protenix/stdout.' + PRO_JOB + '.log'), PRO_JOB, True)
                    break
                except (FileNotFoundError, ValueError):
                    require(attempt < 6, 'Actual Protenix DONE lacks a completed same-job LSF success record')
            require(attempt < 6, 'Existing Protenix has not reached an actual final status')
            time.sleep(10)
        stdout = ROOT03 / ('run-protenix/stdout.' + PRO_JOB + '.log')
        lsf_outcome(stdout, PRO_JOB, True)
        proofs = pro_terminal_proofs()
        record.update(status='observed_success', scheduler=hashed(path), stdout=hashed(stdout),
                      updates=10000, states=10001, proofs=proofs + [hashed(path), hashed(stdout)])
    except Exception as error:
        record['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        exclusive(ROOT / 'pro-terminal-observation.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('capture', 'dispatch', 'run', 'observe-pro'))
    parser.add_argument('--tool-source', type=Path, required=True)
    parser.add_argument('--tool-manifest', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=NEW_ARMS)
    parser.add_argument('--cpu-certificate-sha256')
    parser.add_argument('--adoption-sha256')
    args = parser.parse_args()
    require(args.root.resolve() == ROOT and args.tool_source.resolve() == TOOL_SOURCE,
            'Wrong declared immutable recovery source/control scope')
    require(bool(os.environ.get('LSB_JOBID')), 'Recovery operations require allocated DTU HPC')
    check_release(TOOL_SOURCE, args.tool_manifest)
    if args.phase == 'capture':
        print(json.dumps(capture_adoption()))
        return
    require(args.cpu_certificate_sha256 and args.adoption_sha256, 'Actual new CPU/adoption pins required')
    if args.phase == 'dispatch':
        print(json.dumps(dispatch(args.tool_manifest, args.cpu_certificate_sha256, args.adoption_sha256)))
    elif args.phase == 'run':
        run_arm(args.arm, args.tool_manifest, args.cpu_certificate_sha256, args.adoption_sha256)
    else:
        print(json.dumps(observe_pro(args.tool_manifest, args.cpu_certificate_sha256, args.adoption_sha256)))


if __name__ == '__main__':
    main()
