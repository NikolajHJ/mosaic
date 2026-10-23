"""Saved-only high-rate JQ1 report contract, with explicit failed-run scope.

Presentation verification never changes a scientific outcome. Terminal gradient
failures remain failed; passed saved optimizer cases are descriptive trajectories.
All source/input authentication is read-only. Importing this module is stdlib only.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time
import xml.etree.ElementTree as ET

VERSION = 'jq1-25-lrreset-saved-report-v1'
FAILED_APPLICABILITY = 'jq1-lrreset-failed-terminal-presentation-v1'
TAG = '20261022-brd4-jq1-25-fixed-lrreset-04'
HOME = Path('/zhome/95/e/206121/mosaic-runs')
RELEASES = Path('/work3/s234818/mosaic/releases')
SCIENCE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-04/source')
ROOT = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04')
HOME_ROOT = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04')
SCIENCE_MANIFEST = 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
CPU_SHA = '6a87093f3da0d1ca0694c7152de84fdeed308251f709c685b526d978b1455e23'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
REFERENCE = '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
PREPARED = HOME / '20261021-brd4-jq1-25-01/prepare/payload/prepared'
BOLTZ_CONTEXT = 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
DISPATCH_SHA = '5ed789a3ffc83821332dcb615ec7d94dc88969fe49ef1b761ec09cc5393d5441'
JOBS_SHA = '2aa2f705d039a18ccb9adf8e2a7721e74685959aaab264b2df4cd8d348b47c59'
ARMS = {'protenix': {'metric': 'contact', 'color': '#246bb2', 'label': 'Protenix · JQ1 pocket contact', 'job': '29792670', 'rate': 0.3, 'manifest': 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057', 'config': '5b1739f0c9c82e97c05d1fee4fe2b26b8efdacb96d2f4152cd873d96b4464959', 'ready': '9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd', 'native_job': '29792413', 'model': '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2', 'context': 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4', 'source': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-04/source', 'science_root': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04', 'output_parent': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04/run-protenix', 'submission': '5dc53846d086b2b6522d1d4aeb42c3461e4875f890b547b8362904793f33e58a', 'decision': 'd15fa46fefefb66aeaae6147293d280ad3b24aa8b9a48e37846feae009793537', 'projected_bytes': 1493521824, 'driver': '2768739f1896c0ca89872323eb5bf2dd75cb1dda444b53701e250fc44c7e47e5', 'entry': 'e6f331c0a5502a60c8abcbd91e36bd7bea45dd4b7ad42f46b1891bad66a7289f', 'raw_entry': '04e641b69132069a2cfd42b7de25d30c78aa6a5a374fd30bb6282809e12c1064', 'coupled_entry': '3ef8a644ce2963837f6c723860ca585dc925727a9edda20a24b65c30f4d5c593'}, 'boltz2': {'metric': 'affinity', 'color': '#9a4dba', 'label': 'Boltz-2 · fixed JQ1-pocket pose', 'job': '29792671', 'rate': 0.005, 'manifest': 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057', 'config': '40f8038dac8485bdfcde4591d48b1162edaf6294e04406efee97a68813670d7f', 'ready': '6d9fe45458296f8e0320e4cf771f363a900ed0b002bb1fddb0d30a63171bf67a', 'native_job': '29792415', 'model': '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16', 'context': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4', 'source': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-04/source', 'science_root': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04', 'output_parent': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04/run-boltz2', 'submission': 'ddc416a8ed853307b639e9011fefadbb484d2721dfd3e3f9d054045c1eee8df0', 'decision': '603ea09a5ef241688f2db220c25d2840a04cf1dadda42c73520dc4c53277fd5b', 'projected_bytes': 1748021824, 'driver': '00f0d951c87b311d76aebd7ca87391b2db17ad8176fe98257ac0b88be66708c8', 'entry': '4fdfa14d791a1966fdea7da9d1c6939fbe952387d5ee683a732ec6c721eaf3a6', 'raw_entry': 'cfa919e63cac1f0a4bbdb3cd09cfce86b92a1da77b5074ea664265721de46e3f', 'coupled_entry': '98c1020b11dace99570d297701704d9a5a39f8a77da24f2ef6d325f59e939962', 'failed_terminal': {'applicability': 'jq1-lrreset-failed-terminal-presentation-v1', 'summary': '06a68008a0dfb8b6a66668447bcda4f01c7b54427d9b68f0d84e5926db278cc1', 'case': '1c2ee9ef8f62c1c0034757f4e8d000e62b90d5c912d12fa1be182ec9526f6e95', 'terminal': '1a95448ea1d3fe373f7a3f8ad64017ef7ffd620f52efc3086cf2262ecf4868a4', 'stdout': 'b40c2741e3d500f74aafa8819a597d260b7372f5372eb2642aeb6c450341f5c0', 'error': 'AssertionError: Native affinity E_upper finite differences failed'}}, 'boltz2_chemistry': {'metric': 'affinity', 'color': '#008979', 'label': 'Boltz-2 + delayed chemistry', 'job': '29792672', 'rate': 0.005, 'manifest': 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057', 'config': '02b65a4face438c4e37d15b92f7833673513cce2dbfb6feceff8aef7a2c63063', 'ready': '785332602a1e72a3e5d59add5c0e7e338d7f42ceceb3b24ad48f14be30305457', 'native_job': '29792417', 'model': '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16', 'context': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4', 'source': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-04/source', 'science_root': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04', 'output_parent': '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04/run-boltz2_chemistry', 'submission': 'da519e68877d3ab007f4e83a78038068f8df198cc69d4b155b3503a306b2f969', 'decision': '043981f8b755c996cfea45eb6fe94754498d46fc6590a4805495f758f33c7995', 'projected_bytes': 5144021824, 'driver': None, 'entry': None, 'raw_entry': None, 'coupled_entry': None}}

def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def hashed(path, expected=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(), 'Unsafe evidence path: ' + str(path))
    digest = sha(path)
    require(expected is None or digest == expected, 'Evidence bytes changed: ' + str(path))
    return {'path': str(path), 'sha256': digest}


def canonical_digest(value):
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value):
        digest.update(part.encode())
    return digest.hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fixed_counts(updates, rate):
    """Count saved outcomes without converting fixed steps into strict descent."""
    require(_finite(rate) and rate in (0.3, 0.005), 'Unexpected declared fixed learning rate')
    counts = dict(applied=len(updates), strict_improving=0, forced_fallback=0, no_op=0,
                  scalar_increased=0, scalar_equal=0, scalar_decreased=0)
    for index, update in enumerate(updates):
        require(type(update.get('step')) is int and update.get('step') == index and update.get('selection_policy') == 'fixed_step', 'Fixed policy/continuity differs')
        attempts = update.get('attempts')
        require(isinstance(attempts, list) and len(attempts) == 1, 'Fixed step must have exactly one candidate')
        attempt = attempts[0]
        require(type(attempt.get('trial')) is int and attempt['trial'] == 0
                and type(update.get('selected_trial')) is int and update['selected_trial'] == 0
                and _finite(attempt.get('rate')) and attempt['rate'] == rate
                and _finite(update.get('accepted_step_size')) and update['accepted_step_size'] == rate
                and update.get('fallback_used') is False, 'Selected fixed rate/trial/fallback differs')
        before, after = update.get('fixed_before'), update.get('fixed_after')
        require(all(_finite(v) for v in (before, after, attempt.get('autodiff_surrogate'), attempt.get('independent_primal_surrogate'))),
                'Nonfinite or missing saved candidate scalar')
        require(attempt['independent_primal_surrogate'] == after, 'Selected candidate differs from applied fixed scalar')
        require(type(attempt.get('strict')) is bool and type(update.get('no_op')) is bool
                and type(update.get('fixed_increased')) is bool and update['fixed_increased'] is (after > before),
                'Fixed strict/no-op/increase bookkeeping differs')
        require(not attempt['strict'] or after < before - 1e-7, 'Saved strict flag contradicts fixed primal decrease')
        components = attempt.get('components')
        require(isinstance(components, dict) and components and all(_finite(v) for v in components.values()),
                'Nonfinite or missing saved candidate components')
        require(_finite(attempt.get('repair_weight')) and _finite(update.get('repair_weight'))
                and attempt['repair_weight'] == update['repair_weight'], 'Trial coefficient changed within fixed preparation')
        counts['strict_improving'] += int(attempt['strict'])
        counts['no_op'] += int(update['no_op'])
        counts['scalar_increased'] += int(after > before)
        counts['scalar_equal'] += int(after == before)
        counts['scalar_decreased'] += int(after < before)
    return counts


def verify_coefficients(arm, states, updates):
    """Authenticate the producer's saved state/update-indexed coefficients."""
    def expected(state):
        if arm != 'boltz2_chemistry':
            return 0.0
        applied = min(state + 1, 10000)
        if applied <= 5000:
            return 0.001
        if applied == 10000:
            return 100.0
        fraction = (applied - 5000) / 5000
        return math.exp(math.log(0.001) + fraction * (math.log(100.0) - math.log(0.001)))
    for state in states:
        require(state.get('repair_weight') == expected(state['step']), 'Saved state chemistry schedule differs')
    for row in updates:
        require(row.get('repair_weight') == expected(row['step']), 'Saved applied-update chemistry schedule differs')


def objective_identity(arm):
    pin = ARMS[arm]
    return {'model_digest': pin['model'], 'context_digest': pin['context'], 'native_name': pin['metric'],
            'coefficient': 1.0, 'n_atoms': 25, 'pocket_definition_sha256': POCKET, 'diffusion_gradient': False,
            'scope': 'JQ1 pocket-restricted contact, geometry-off' if arm == 'protenix'
                     else 'JQ1-constrained frozen native pose, zero recycles, no MW correction'}


def _release(source, expected):
    manifest_path = Path(source) / 'source_manifest.json'
    hashed(manifest_path, expected)
    manifest = read(manifest_path)
    require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Empty immutable source file map')
    for name, pin in manifest['files'].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == name, 'Unsafe source manifest member')
        digest = pin['sha256'] if isinstance(pin, dict) else pin
        proof = hashed(Path(source) / name, digest)
        if isinstance(pin, dict) and 'bytes' in pin:
            require(Path(proof['path']).stat().st_size == pin['bytes'], 'Immutable source length changed')
    return manifest


def scheduler_status(raw, job, queue='gpul40s'):
    compact = re.sub(r'\s+', '', raw)
    jobs, queues = re.findall(r'Job<([^>]+)>', compact), re.findall(r'Queue<([^>]+)>', compact)
    statuses = re.findall(r'Status<([^>]+)>', compact)
    require(jobs == [job] and queues == [queue] and len(statuses) == 1, 'Scheduler identity/queue missing or ambiguous')
    require(statuses[0] in ('RUN', 'PEND', 'DONE', 'EXIT'), 'Actual scheduler outcome unavailable')
    return statuses[0]


def scheduler_outcome(raw, job):
    status = scheduler_status(raw, job)
    require(status in ('DONE', 'EXIT'), 'Actual final scheduler outcome unavailable')
    return status


def lsf_success(path, job):
    raw = Path(path).read_text(encoding='utf-8')
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', raw, re.M)
    require(len(subjects) == 1 and subjects[0][0] == job and subjects[0][1].rstrip().endswith(' Done')
            and re.search(r'^Successfully completed\.\s*$', raw, re.M)
            and not re.search(r'Exited\s+(?:with exit|by signal)', raw, re.I), 'Missing same-job actual LSF success')


def source_path(arm):
    return Path(ARMS[arm]['source'])


def config_path(arm):
    return source_path(arm) / ('examples/configs/brd4_jq1_25_fixed_lrreset_' + arm + '_10000.json')


def ready_path(arm):
    return Path(ARMS[arm]['science_root']) / ('validate-' + arm) / 'payload/study/assessment.json'


def native_parent(arm):
    return Path(ARMS[arm]['output_parent'])


def study_path(arm):
    return native_parent(arm) / 'payload/study'


def _science_ops():
    _release(SCIENCE, SCIENCE_MANIFEST)
    previous = list(sys.path)
    try:
        sys.path[:0] = [str(SCIENCE / 'scripts')]
        spec = importlib.util.spec_from_file_location('_lrreset_original_science', SCIENCE / 'scripts/fixed25_lrreset_ops.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def _cpu(ops):
    cert = ops.verify_cpu(SCIENCE_MANIFEST, CPU_SHA)
    require(cert['producer_job_id'] == '29792389' and cert['tests'] == 728, 'Wrong original CPU job/count')
    required = {'hpc-contract.xml': 396, 'legacy.xml': 239, 'native-mapping.xml': 34, 'chemistry.xml': 59}
    proofs = {p['path']: p['sha256'] for p in cert['proofs']}
    for name, count in required.items():
        path = ROOT / 'cpu/payload' / name
        require(str(path) in proofs, 'Original CPU omitted raw XML')
        tree = ET.parse(path)
        require(len(list(tree.iter('testcase'))) == count and not any(list(tree.iter(t)) for t in ('failure','error','skipped')), 'Original raw CPU cases failed/changed')
    scheduler=ROOT/'cpu/scheduler-final-0.txt';stdout=ROOT/'cpu/stdout.29792389.log'
    require(str(scheduler) in proofs and str(stdout) in proofs and scheduler_status(scheduler.read_text(),'29792389','hpc')=='DONE','Original CPU final proof missing')
    lsf_success(stdout,'29792389')
    return cert


def _storage(gate, phase):
    projections={a:p['projected_bytes'] for a,p in ARMS.items()}
    require(gate.get('version')=='jq1-lrreset-all-work3-launch-adoption-v1' and gate.get('status')=='passed'
            and gate.get('phase')==phase and gate.get('projections')==projections
            and gate.get('placement')==dict.fromkeys(ARMS,'work3') and gate.get('home_output_writes') is False,
            'Original whole all-work3 storage scope/gate changed')
    require(set(gate['filesystems'])=={'work3'},'Wrong assigned output filesystem')
    budget=gate['filesystems']['work3']
    outstanding=sum(projections[a]-gate['current_outputs'][a] for a in ARMS)
    require(all(type(gate['current_outputs'][a]) is int and 0<=gate['current_outputs'][a]<=projections[a] for a in ARMS)
            and gate['outstanding_native_bytes']==outstanding
            and gate['concurrent_analysis_allowance_bytes']==2684354560
            and budget['status']=='passed' and budget['projected_bytes']==outstanding+2684354560
            and budget['reserve_bytes']==4294967296 and budget['file_reserve']==100000
            and budget['required_bytes']==outstanding+2684354560+4294967296
            and budget['available_bytes_conservative']>=budget['required_bytes']
            and budget['available_files_lower']>=100000,'Original combined budget/reserves differ')
    require(budget['filesystem_inode_reporting'] in ('unreported_all_zero','reported_physical_filesystem_only'), 'Wrong physical inode reporting')


def authenticate_inputs(arm):
    require(arm in ARMS,'Unknown report arm')
    pin=ARMS[arm];ops=_science_ops();_cpu(ops)
    launcher=_launcher();cpu=launcher.verify_cpu(LAUNCH_MANIFEST)
    require(cpu['sha256']==LAUNCH_CPU_SHA,'Original launcher CPU pin differs')
    marker_path,jobs_path=ROOT/'long-dispatch-started.json',ROOT/'long-submissions.json'
    hashed(marker_path,DISPATCH_SHA);hashed(jobs_path,JOBS_SHA)
    marker,jobs=read(marker_path),read(jobs_path)
    inherited=launcher.inherited()
    require(inherited=={k:marker[k] for k in inherited},'Original scientific/native/proof adoption changed')
    require(marker['source_manifest_sha256']==SCIENCE_MANIFEST and marker['launch_manifest_sha256']==LAUNCH_MANIFEST
            and marker['launcher_cpu']['sha256']==LAUNCH_CPU_SHA and jobs['all_three_submitted'] is True
            and {j['job_id'] for j in jobs['jobs']}=={p['job'] for p in ARMS.values()},'Wrong original dispatch/cohort')
    hashed(marker['adoption']['path'],marker['adoption']['sha256']);_storage(marker['storage_gate'],'dispatch')
    for selected,bound in ARMS.items():
        decision_path=ROOT/('decision-'+selected+'.json');hashed(decision_path,bound['decision'])
        decision=read(decision_path);ready=ops.authenticate_ready(selected,SCIENCE_MANIFEST,decision)
        require(decision['native_job']==bound['native_job'] and decision['assessment_sha256']==bound['ready']
                and ready['model_digest']==bound['model'] and ready['context_digest']==bound['context']
                and ready['pocket_definition_sha256']==POCKET and ready['diffusion_gradient'] is False
                and decision['projected_bytes']==bound['projected_bytes'],'Original passed readiness pins differ')
    parent=native_parent(arm);path=parent/'submission.json';hashed(path,pin['submission']);sub=read(path)
    require(sub in jobs['jobs'] and sub['outcome']=='submitted' and sub['job_id']==pin['job']
            and sub['source_manifest_sha256']==SCIENCE_MANIFEST and sub['launch_manifest_sha256']==LAUNCH_MANIFEST
            and sub['launch_cpu_certificate_sha256']==LAUNCH_CPU_SHA and sub['cpu_certificate_sha256']==CPU_SHA
            and sub['config_sha256']==pin['config'] and sub['learning_rate']==pin['rate']
            and sub['output_parent']==str(parent)
            and sub['args']==launcher.long_command(arm,LAUNCH_MANIFEST,LAUNCH_CPU_SHA),'Original actual submission differs')
    hashed(config_path(arm),pin['config'])
    invocation=parent/'payload/driver-invocation.json';gate=parent/'entry-quota/lrreset-storage-gate.json'
    started=invocation.exists() or study_path(arm).exists()
    if started:
        require(pin.get('entry') and pin.get('driver'),'New arm entry requires a fresh immutable report pin')
        hashed(gate,pin['entry']);_storage(read(gate),'entry');hashed(invocation,pin['driver'])
        expected={'args':launcher.driver_command(arm),'source_manifest_sha256':SCIENCE_MANIFEST,
                  'launch_manifest_sha256':LAUNCH_MANIFEST,'launch_cpu_sha256':LAUNCH_CPU_SHA,'scientific_cpu_sha256':CPU_SHA}
        require(read(invocation)==expected,'Original scientific driver/config/readiness/output changed')
    require(sha(ROOT/'cpu/payload/composition_reference.json')==REFERENCE
            and sha(ROOT/'cpu/payload/pocket_definition.json')==POCKET,'Original pocket/reference changed')
    if arm!='protenix':require(sha(PREPARED/'context.pt')==BOLTZ_CONTEXT,'Original detached context bytes changed')
    return {'version':VERSION,'arm':arm,'native_job':pin['job'],'source_manifest_sha256':SCIENCE_MANIFEST,
            'cpu_certificate_sha256':CPU_SHA,'launcher_manifest_sha256':LAUNCH_MANIFEST,
            'launcher_cpu_certificate_sha256':LAUNCH_CPU_SHA,'config_sha256':pin['config'],'readiness_sha256':pin['ready'],
            'submission':hashed(path),'entry':hashed(gate) if started else None,
            'driver_invocation':hashed(invocation) if started else None,'scientific_execution_started':started}


authenticate_launcher = authenticate_inputs


def lsf_failure(path, job):
    text = Path(path).read_text(encoding='utf-8')
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', text, re.M)
    match = re.search(r'^Exited with exit code (\d+)\.\s*$', text, re.M)
    require(len(subjects) == 1 and subjects[0][0] == job and subjects[0][1].rstrip().endswith(' Exited')
            and match is not None and int(match[1]) != 0 and 'Successfully completed.' not in text,
            'Missing original same-job LSF nonzero outcome')
    return int(match[1])


def _read_live(path):
    for attempt in range(5):
        try:
            return read(path)
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(0.2)


def _jsonl_prefix(path, *, with_bytes=False):
    raw = Path(path).read_bytes()
    boundary = raw.rfind(b'\n') + 1
    committed = raw[:boundary]
    rows = [json.loads(line) for line in committed.splitlines()]
    require(all(type(row.get('step')) is int and row['step'] == index for index, row in enumerate(rows)),
            'Noncontiguous or corrupt complete saved JSONL rows')
    proof = {'path': str(path), 'bytes': boundary, 'sha256': hashlib.sha256(committed).hexdigest(),
             'uncommitted_tail_bytes': len(raw) - boundary}
    return (rows, proof, committed) if with_bytes else (rows, proof)


def saved_prefix(root, case=None):
    """Common committed prefix, additionally bounded by the captured case JSON."""
    root = Path(root)
    states, sp, state_bytes = _jsonl_prefix(root / 'states.jsonl', with_bytes=True)
    updates, up, update_bytes = _jsonl_prefix(root / 'updates.jsonl', with_bytes=True)
    latest = min(len(states) - 1, len(updates))
    if case is not None:
        latest = min(latest, len(case['states']) - 1, len(case['updates']))
    require(0 <= latest <= 10000, 'No common saved state or unexpected update budget')
    captured = {'states': states[:latest + 1], 'updates': updates[:latest]}
    if case is not None:
        require(captured == {'states': case['states'][:latest + 1], 'updates': case['updates'][:latest]},
                'Saved case and committed JSONL differ within included prefix')
    for proof, raw, included, available in ((sp, state_bytes, latest + 1, len(states)),
                                            (up, update_bytes, latest, len(updates))):
        used = b''.join(raw.splitlines(keepends=True)[:included])
        proof.update(available_committed_bytes=proof['bytes'], available_committed_rows=available,
                     bytes=len(used), sha256=hashlib.sha256(used).hexdigest(), included_rows=included,
                     excluded_committed_rows=available - included)
    return captured, [sp, up]


def _case_identity(arm, case, captured, study):
    pin = ARMS[arm]
    config = read(config_path(arm))
    require(case['status'] in ('running', 'passed', 'failed') and case['requested_updates'] == 10000
            and case['parameterization'] == 'categorical_logits_v1' and case['acceptance_policy'] == 'fixed_step'
            and case['nominal_step_size'] == pin['rate'] and case['key_data'] == [0, 17]
            and case.get('repair_weight_schedule') == config.get('repair_schedule')
            and case[pin['metric'] + '_objective_identity'] == objective_identity(arm), 'Saved case policy/key/objective differs')
    require(pin.get('raw_entry') and pin.get('coupled_entry'), 'A newly started arm needs a fresh immutable entry pin')
    raw_path, coupled_path = study / 'native_gradient_checks/report.json', study / 'case-0/combined_initial_checks/report.json'
    hashed(raw_path, pin['raw_entry'])
    hashed(coupled_path, pin['coupled_entry'])
    raw, coupled = read(raw_path), read(coupled_path)
    require(raw['status'] == 'passed' and coupled['status'] == 'passed'
            and coupled['directions'] == case['initial_checks']
            and coupled[pin['metric'] + '_objective_identity'] == objective_identity(arm)
            and sha(study / 'pocket_definition.json') == POCKET, 'Original entry/coupled/pocket gate differs/failed')
    fixed_counts(captured['updates'], pin['rate'])
    verify_coefficients(arm, captured['states'], captured['updates'])
    return {'raw_entry_sha256': pin['raw_entry'], 'coupled_entry_sha256': pin['coupled_entry'],
            'objective_identity': objective_identity(arm), 'pocket_definition_sha256': POCKET}


def _summary_identity(arm, summary, study):
    pin = ARMS[arm]
    require(summary['job_id'] == pin['job'] and summary['mode'] == 'run'
            and summary['source_manifest_sha256'] == SCIENCE_MANIFEST and summary['config_sha256'] == pin['config']
            and summary['config'] == read(config_path(arm))
            and summary['optimization_parameters'] == {'step_size': pin['rate'], 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'},
            'Original terminal source/config/policy/job differs')
    manifest = read(SCIENCE / 'source_manifest.json')
    expected = {name: value for name, value in manifest['files'].items()
                if name == 'pyproject.toml' or (PurePosixPath(name).parts[0] in ('src', 'tests', 'examples', 'scripts')
                    and PurePosixPath(name).suffix in ('.py', '.json', '.lsf', '.sh', '.toml', '.txt')
                    and '__pycache__' not in PurePosixPath(name).parts)}
    require(summary['source']['files'] == expected and summary['source']['execution_content_sha256'] == canonical_digest(expected)
            and summary['source']['git_revision_reference'] == manifest['git_revision_reference'], 'Complete executed source file map differs')
    require(summary['pocket_validation_status'] == 'passed' and summary['pocket_definition_sha256'] == POCKET
            and summary['diffusion_gradient'] is False and summary['native_hard_parity']['status'] == 'passed'
            and summary['native_hard_parity']['n_atoms'] == 25 and summary['native_hard_parity']['atol'] == 2e-5
            and summary['initialization_sha256'] == sha(study / 'initialization.npz'), 'Initial native/pocket/input identity failed')
    for field in ('model', 'context'):
        require(summary[field + '_digest'] == summary['frozen_identity_after'][field + '_digest'] == pin[field], 'Frozen identity differs')
    if arm != 'protenix':
        require(summary['frozen_identity_after']['status'] == 'passed' and summary['context_file_sha256'] == BOLTZ_CONTEXT,
                'Original detached Boltz context/frozen response failed')
    else:
        require(summary.get('initialization_execution', {}).get('version') == 'protenix-fp32-cpu-atom-softmax-v1', 'Exact declared FP32 placement missing')
        initial = summary.get('baseline_initialization_match', {})
        require(initial.get('status') == 'passed' and initial.get('job') == '29770179'
                and initial.get('sha256') == '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf',
                'Protenix original initialization arrays/key identity differs')
    require(summary['native_gradient_checks'] == read(study / 'native_gradient_checks/report.json')
            and summary['native_gradient_checks']['status'] == 'passed', 'Original initial native derivative gate failed/differs')


def verify_case(study, summary, *, steps=10000, allow_failed=False):
    """No partial/failed case can become a full scientific completion."""
    study = Path(study)
    arm = summary['config']['arm']
    case = read(study / 'case-0/report.json')
    require(summary['status'] == 'passed' or (allow_failed and summary['status'] == 'failed'), 'Native overall result did not pass')
    captured, _ = saved_prefix(study / 'case-0', case)
    full = summary['status'] == 'passed' or case['status'] == 'passed' or bool(ARMS[arm].get('failed_terminal'))
    if full:
        require(case['status'] == 'passed' and case['requested_updates'] == steps
                and len(case['updates']) == steps and len(case['states']) == steps + 1
                and captured == {'states': case['states'], 'updates': case['updates']}, 'Incomplete terminal optimizer budget/JSONL')
        for field in ('states', 'updates'):
            rows, proof = _jsonl_prefix(study / ('case-0/' + field + '.jsonl'))
            require(rows == case[field] and proof['uncommitted_tail_bytes'] == 0,
                    'Full terminal JSONL differs from original case or has unfinished bytes')
    else:
        require(allow_failed and case['status'] == 'failed', 'Partial failure applicability requires actual failed optimizer case')
    wrapped = summary.get('cases', [])
    if full or wrapped:
        require(len(wrapped) == 1 and wrapped[0].get('label') == summary['config']['label']
                and wrapped[0].get('initialization') == summary['config']['initialization']
                and canonical_digest({k: v for k, v in wrapped[0].items() if k not in ('label', 'initialization')}) == canonical_digest(case),
                'Original summary wrapper and optimizer case differ')
    counts = fixed_counts(captured['updates'], ARMS[arm]['rate'])
    if full:
        require(all(case['counts'].get(k) == counts[k] for k in ('applied', 'strict_improving', 'forced_fallback', 'no_op')),
                'Original optimizer bookkeeping differs')
    verify_coefficients(arm, captured['states'], captured['updates'])
    return case if full else {**case, **captured}


def failure_evidence(arm):
    """Actual persisted LSF/scientific failure, never a fabricated scheduler EXIT."""
    pin, study = ARMS[arm], study_path(arm)
    bound = pin.get('failed_terminal')
    summary_path, stdout = study / 'summary.json', native_parent(arm) / ('stdout.' + pin['job'] + '.log')
    if bound:
        hashed(summary_path, bound['summary'])
        hashed(study / 'case-0/report.json', bound['case'])
        hashed(study / 'terminal_gradient_checks/report.json', bound['terminal'])
        hashed(stdout, bound['stdout'])
    summary = read(summary_path)
    require(summary.get('job_id') == pin['job'] and summary.get('status') == 'failed'
            and isinstance(summary.get('error'), str) and summary['error'], 'No actual original same-job scientific failure')
    code = lsf_failure(stdout, pin['job'])
    if bound:
        require(code == 1 and summary['error'] == bound['error'], 'Known original terminal error/outcome changed')
        terminal = read(study / 'terminal_gradient_checks/report.json')
        require(terminal.get('status') == 'failed', 'Known failed terminal gate changed')
    return {'native_status': 'FAILED_PERSISTED', 'scheduler_scope': 'Live history unavailable; original same-job LSF and saved scientific failure only',
            'applicability': FAILED_APPLICABILITY, 'summary': hashed(summary_path), 'stdout': hashed(stdout),
            'exit_code': code, 'error': summary['error']}


def authenticate_study(arm, *, allow_failed=False):
    authenticate_inputs(arm)
    study = study_path(arm)
    if ARMS[arm].get('failed_terminal'):
        failure_evidence(arm)
        require(allow_failed, 'Known failed native run cannot certify strict scientific success')
    summary = read(study / 'summary.json')
    _summary_identity(arm, summary, study)
    if summary['status'] == 'passed':
        lsf_success(native_parent(arm) / ('stdout.' + ARMS[arm]['job'] + '.log'), ARMS[arm]['job'])
        terminal = read(study / 'terminal_gradient_checks/report.json')
        require(terminal['status'] == 'passed' and summary.get('terminal_gradient_checks') == terminal,
                'Mandatory terminal native derivative gate failed/absent')
    else:
        require(allow_failed and summary['status'] == 'failed', 'Overall native failure cannot certify scientific success')
        failure_evidence(arm)
    case = verify_case(study, summary, allow_failed=allow_failed)
    _case_identity(arm, case, {'states': case['states'], 'updates': case['updates']}, study)
    if summary['status'] == 'passed' and arm == 'boltz2_chemistry':
        require(case['states'][-1]['repair_weight'] == 100.0, 'Chemistry endpoint did not reach 100')
    return study, summary, case


def saved_outcome(arm, native_status=None):
    """Separate optimizer budget, initial readiness and scientific final outcome."""
    study, pin = study_path(arm), ARMS[arm]
    summary_path, case_path = study / 'summary.json', study / 'case-0/report.json'
    case = _read_live(case_path) if case_path.exists() else None
    updates = 0 if case is None else min(len(case.get('updates', [])), max(0, len(case.get('states', [])) - 1))
    terminal_status, terminal_error, scope = 'pending', None, 'Actual live status supplied by caller' if native_status else 'No live scheduler status supplied'
    passed, all_terminal = False, False
    if summary_path.exists():
        if pin.get('failed_terminal'):
            failure_evidence(arm)
        summary = read(summary_path)
        require(summary.get('job_id') == pin['job'] and summary.get('status') in ('passed', 'failed'), 'Original terminal summary job/status differs')
        all_terminal = True
        overall = summary['status']
        if overall == 'failed':
            failure = failure_evidence(arm)
            native_status = native_status if native_status == 'EXIT' else failure['native_status']
            scope = failure['scheduler_scope'] if native_status == 'FAILED_PERSISTED' else scope
            terminal_error = summary['error']
            terminal = study / 'terminal_gradient_checks/report.json'
            terminal_status = read(terminal)['status'] if terminal.exists() else 'not_reached'
            require(terminal_status in ('passed', 'failed'), 'Unexpected terminal derivative status') if terminal.exists() else None
        else:
            terminal = read(study / 'terminal_gradient_checks/report.json')
            require(terminal['status'] == 'passed' and summary.get('terminal_gradient_checks') == terminal, 'Overall pass lacks passed terminal gate')
            lsf_success(native_parent(arm) / ('stdout.' + pin['job'] + '.log'), pin['job'])
            terminal_status, passed = 'passed', True
            native_status = native_status or 'SUCCESS_PERSISTED'
            scope = 'Original passed native summary/terminal gate and same-job LSF; live scheduler status distinct'
    else:
        overall = 'failed' if native_status == 'EXIT' else ('queued' if native_status == 'PEND' else 'running' if case else 'preparing')
        all_terminal = native_status == 'EXIT'
        terminal_status = 'not_reached' if all_terminal else 'pending'
    return {'native_status': native_status or 'UNKNOWN', 'readiness_status': 'passed', 'terminal_status': terminal_status,
            'terminal_error': terminal_error, 'overall_status': overall, 'scientific_passed': passed,
            'all_terminal': all_terminal, 'budget_status': 'full' if updates == 10000 else 'partial' if updates else 'not_started',
            'updates': updates, 'states': updates + 1 if case else 0, 'scheduler_scope': scope,
            'failed_run_applicability': FAILED_APPLICABILITY if overall == 'failed' else None}


def terminal_evidence(arm, completions=None):
    if ARMS[arm].get('failed_terminal'):
        authenticate_study(arm, allow_failed=True)
        return failure_evidence(arm)
    study = study_path(arm)
    if (study / 'summary.json').exists():
        # Bootstrap a collector from original terminal bytes, never from its own
        # not-yet-created assessment or an invented historical scheduler record.
        study, summary, _ = authenticate_study(arm, allow_failed=True)
        if summary['status'] == 'failed':
            evidence = failure_evidence(arm)
        else:
            stdout = native_parent(arm) / ('stdout.' + ARMS[arm]['job'] + '.log')
            lsf_success(stdout, ARMS[arm]['job'])
            evidence = {'native_status': 'SUCCESS_PERSISTED',
                        'scheduler_scope': 'Live history unavailable; authenticated original full native summary/terminal gate and same-job LSF success only',
                        'summary': hashed(study / 'summary.json'), 'stdout': hashed(stdout)}
        names = ('summary.json', 'case-0/report.json', 'case-0/states.jsonl', 'case-0/updates.jsonl',
                 'terminal_gradient_checks/report.json')
        evidence['proofs'] = [hashed(study / name) for name in names if (study / name).exists()]
        evidence['proofs'].append(evidence['stdout'])
        return evidence
    require(completions is not None, 'Aged-out current native needs an authentic persisted collector')
    path = Path(completions) / ('complete-' + arm) / 'payload/assessment.json'
    record = read(path)
    pin = ARMS[arm]
    require(record.get('native_job') == pin['job'] and record.get('arm') == arm
            and record.get('source_manifest_sha256') == SCIENCE_MANIFEST
            and record.get('presentation_status', record.get('status')) == 'passed', 'Missing authentic current-job collector proof')
    proofs = record.get('proofs', [])
    require(proofs and len({p['path'] for p in proofs}) == len(proofs), 'Missing/duplicate original collector proofs')
    for proof in proofs:
        hashed(proof['path'], proof['sha256'])
    schedulers = [Path(p['path']) for p in proofs if Path(p['path']).name.startswith(('native-scheduler', 'scheduler-final'))]
    statuses = [scheduler_status(p.read_text(), pin['job']) for p in schedulers if re.search(r'Status\s*<', p.read_text())]
    require(any(s in ('DONE', 'EXIT') for s in statuses), 'Collector lacks actual native terminal scheduler proof')
    status = 'EXIT' if 'EXIT' in statuses else 'DONE'
    authenticate_study(arm, allow_failed=status == 'EXIT')
    return {'native_status': status, 'assessment': hashed(path), 'proofs': proofs}


def status_or_completion(raw, arm, completions=None):
    if re.search(r'Status<[^>]+>', re.sub(r'\s+', '', raw)):
        return scheduler_status(raw, ARMS[arm]['job'])
    return terminal_evidence(arm, completions)['native_status']


def authenticate_progress(arm, status):
    require(status in ('RUN', 'PEND', 'DONE', 'EXIT', 'FAILED_PERSISTED', 'SUCCESS_PERSISTED'), 'Unknown report native outcome')
    launcher = authenticate_inputs(arm)
    study = study_path(arm)
    if (study / 'summary.json').exists():
        _, _, case = authenticate_study(arm, allow_failed=True)
        captured = {'states': case['states'], 'updates': case['updates']}
        _, prefixes = saved_prefix(study / 'case-0', case)
    elif (study / 'case-0/report.json').exists() and (study / 'case-0/states.jsonl').exists():
        case = _read_live(study / 'case-0/report.json')
        captured, prefixes = saved_prefix(study / 'case-0', case)
        _case_identity(arm, case, captured, study)
    else:
        require(status not in ('DONE', 'FAILED_PERSISTED', 'SUCCESS_PERSISTED'), 'Terminal native has no saved trajectory')
        return None, {**saved_outcome(arm, status), 'launcher': launcher, 'native_status': status,
                      'state': 'No authenticated saved molecule; queued, preparing or failed before first state'}
    proof = {**saved_outcome(arm, status), **_case_identity(arm, case, captured, study),
             'launcher': launcher, 'jsonl_prefixes': prefixes, 'saved_optimizer_status': case['status'],
             'updates': len(captured['updates']), 'states': len(captured['states'])}
    return captured, proof



LAUNCH_SOURCE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-allwork3-launch-02/source')
LAUNCH_MANIFEST = '40c5055150c06e0aeb9989d33f869aecb6642c3823f6bdc9330e66f41a3bcbbe'
LAUNCH_CPU_SHA = 'f71291c8a542871773f18b4cbeeaa9641683b04b554e40324e1c79073b4d72d4'

def _launcher():
    _release(LAUNCH_SOURCE, LAUNCH_MANIFEST)
    previous=list(sys.path)
    try:
        sys.path[:0]=[str(SCIENCE/'scripts'),str(SCIENCE/'src')]
        spec=importlib.util.spec_from_file_location('_lrreset_original_launch',LAUNCH_SOURCE/'all_work3_launch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    finally:sys.path[:]=previous
