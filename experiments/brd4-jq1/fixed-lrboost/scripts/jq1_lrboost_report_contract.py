"""Saved-only high-rate JQ1 report contract, with explicit failed-run scope.

Presentation verification never changes a scientific outcome. Terminal gradient
failures remain failed; passed saved optimizer cases are descriptive trajectories.
All source/input authentication is read-only. Importing this module is stdlib only.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time
import xml.etree.ElementTree as ET

VERSION = 'jq1-25-lrboost-saved-report-v1'
FAILED_APPLICABILITY = 'jq1-lrboost-failed-terminal-presentation-v1'
TAG = '20261021-brd4-jq1-25-fixed-lrboost-01'
HOME = Path('/zhome/95/e/206121/mosaic-runs')
RELEASES = Path('/work3/s234818/mosaic/releases')
SCIENCE = RELEASES / TAG / 'source'
ROOT = Path('/work3/s234818/mosaic/runs') / TAG
HOME_ROOT = HOME / TAG
SCIENCE_MANIFEST = '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323'
CPU_SHA = 'dc76df74f1432f59e4c5f6fb5491a1258752b639df893cf75bb5a6968e481630'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
REFERENCE = '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
PREPARED = HOME / '20261021-brd4-jq1-25-01/prepare/payload/prepared'
BOLTZ_CONTEXT = 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
DISPATCH_SHA = 'b4c6ecbe7b6ec6af07e543ee4215ad6872cda31339c41d45fb9364fed02f1cb6'
JOBS_SHA = '5d221d4b353a78d1062c3e875a739756c49884325b231b69c33dbbf3905ab062'
ARMS = {'protenix': {'job': '29779177',
              'rate': 3.0,
              'metric': 'contact',
              'color': '#246bb2',
              'label': 'Protenix · JQ1 pocket contact',
              'manifest': '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323',
              'config': 'f1c551e5f9cc6f5671acf8b7fef7fe3dbb3197312b096ef55772cb005195fb93',
              'ready': '4f69927bf12970bd8f573bb3b5eb439beda217f2454e508bf1b5f7ad50dc9185',
              'native_job': '29778424',
              'model': '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2',
              'context': 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4',
              'source': '/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source',
              'science_root': '/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01',
              'output_parent': '/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01/run-protenix',
              'submission': 'c893645a494b72cd3930146eca9263f68c653c492f3349f7408540a22907ab96',
              'driver': 'e0beb7158844b08c1a5fed69ee00ab55e7d98a7ad3b017d4a2bea9082d59b570',
              'decision': 'b510f30591d4f23771346d44f0bab774ee3916e41e544221a8164eee70991463',
              'entry': 'd7a32584442f6dafd550e36ab6aca5e61a88aed4bead79b90025001b259704f3',
              'raw_entry': '27fe5fdbcc2ce8338b164560e1e16a6f8685168de20f4b4e2960ba1dbe442bee',
              'coupled_entry': 'bf5cd8b69e58923e71f462d2366b8337e2c8994b1a54641e466bf5ab4f403867',
              'projected_bytes': 1514341824,
              'failed_terminal': {'applicability': 'jq1-lrboost-failed-terminal-presentation-v1',
                                  'summary': '02bcc7dede3c8bd69e741c230e1eea598946534b15ff7d6cfa7999dbebed2d08',
                                  'case': '873e5a90443de510a130f09203cefdc01bfc210b757ae42e26070b998b457a73',
                                  'terminal': '27273461eddae2362766f95d8e903297e6999020d059cb886c13c4869cc13d64',
                                  'error': 'ValueError: Predeclared epsilon leaves the simplex interior',
                                  'stdout': 'cfc173b8d26a55a76874e3aca89d4a28344e1c1aa27d3c83e0ecc0f85c3cf64b'}},
 'boltz2': {'job': '29779178',
            'rate': 0.05,
            'metric': 'affinity',
            'color': '#9a4dba',
            'label': 'Boltz-2 · fixed JQ1-pocket pose',
            'manifest': '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323',
            'config': 'efb4a41cd3077f03632ca0b4085c0cbfe98c0800b53c0509a857ae28ae4111fd',
            'ready': '1bda32ebcb188e9749a7d21563ad44e8091bb5eb0c3f3ef8dd505197cb61c11c',
            'native_job': '29778426',
            'model': '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16',
            'context': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4',
            'source': '/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source',
            'science_root': '/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01',
            'output_parent': '/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01/run-boltz2',
            'submission': 'b692114e9569d0c75ebb3fd69ef24fb339e353b1e6e029a20f48976800549d18',
            'driver': '25cc0f70633288c849c393a96be73eea30be48c75cbc5948b9a916dfedfa3220',
            'decision': 'e0fe8d0b7abb09be03312bb5e2182aa7e60329af04c2715d530a254abefb6153',
            'entry': '53f0b023bc4a2810016720cab22f121ca9fd963951e0aa384ca0e094f68226b6',
            'raw_entry': 'cfa919e63cac1f0a4bbdb3cd09cfce86b92a1da77b5074ea664265721de46e3f',
            'coupled_entry': '98c1020b11dace99570d297701704d9a5a39f8a77da24f2ef6d325f59e939962',
            'projected_bytes': 1820641824,
            'failed_terminal': {'applicability': 'jq1-lrboost-failed-terminal-presentation-v1',
                                'summary': '9e48647de26a21ba619e0ad368e8b3848d2ef5ad4ebbda35ff68dfdf3a6c9e7e',
                                'case': 'a1e59c6a6f662942ef9ddd137bea5a088150fbee2f033d4d72e84fa5ab9e29e2',
                                'terminal': '9f84726f8c412aa001cd735de14d3d5ed008eaf6b6075cabe61585a51b110685',
                                'error': 'AssertionError: Native affinity E_upper finite differences failed',
                                'stdout': '6cc0a72b3f5593425b72a53f6decb5c6c7148781103bc5fafb8be58631d4fee9'}},
 'boltz2_chemistry': {'job': '29779179',
                      'rate': 0.05,
                      'metric': 'affinity',
                      'color': '#008979',
                      'label': 'Boltz-2 + delayed chemistry',
                      'manifest': '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323',
                      'config': '6a6f15614989e44b6cc1149ee4ca5e35717d5c21fdc7d2b5748832d912e9b3f0',
                      'ready': 'a963facfd8fd57af3ee12b8b071ef5d2d7bcdcea18b5a09e2c3373e166b87385',
                      'native_job': '29778428',
                      'model': '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16',
                      'context': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4',
                      'source': '/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source',
                      'science_root': '/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01',
                      'output_parent': '/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-fixed-lrboost-01/run-boltz2_chemistry',
                      'submission': '327afa4d85948a4120eecb6552f4be5666c51918dee47e30b07d8adcca465319',
                      'driver': '6d0f982f78d0d113aad1bed50e9470e2ad53850bf0235362ef43c526b3a057a4',
                      'decision': '98151f5df98b5c9efbc9868486cfa47c64659d25a5f955be34fbbfad52cd74bd',
                      'entry': 'fa8c1015984ba1df73db3b2a15641c2736c4f56c2daa39eb5aa2925f14f58a5c',
                      'raw_entry': 'cfa919e63cac1f0a4bbdb3cd09cfce86b92a1da77b5074ea664265721de46e3f',
                      'coupled_entry': 'a4d1d08f5edcedc095c58829b1e8cafa934194da8b9578bff3fe189062bba76f',
                      'projected_bytes': 5198501824}}

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
    require(_finite(rate) and rate in (3.0, 0.05), 'Unexpected declared fixed learning rate')
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
            return 50.0
        fraction = (applied - 5000) / 5000
        return math.exp(math.log(0.001) + fraction * (math.log(50.0) - math.log(0.001)))
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
    return _layout_path(source_path(arm) / ('examples/configs/brd4_jq1_25_fixed_lrboost_' + arm + '_10000.json'))


def ready_path(arm):
    return Path(ARMS[arm]['science_root']) / ('validate-' + arm) / 'payload/study/assessment.json'


def native_parent(arm):
    return Path(ARMS[arm]['output_parent'])


def study_path(arm):
    return native_parent(arm) / 'payload/study'


def _science_ops():
    """Authenticate all original 610 members before executing any helper import."""
    _release(SCIENCE, SCIENCE_MANIFEST)
    for name in ('campaign25_contract', 'campaign25_pocket_ops', 'fixed25_lrboost_storage',
                 'fixed25_split_storage', 'fixed25_recover_storage'):
        module = sys.modules.get(name)
        require(module is None or Path(module.__file__).resolve() == _layout_path(SCIENCE / 'scripts' / (name + '.py')),
                'Scientific contract/helper module was shadowed: ' + name)
    previous = list(sys.path)
    try:
        sys.path[:0] = [str(_layout_path(SCIENCE / 'scripts'))]
        spec = importlib.util.spec_from_file_location('_jq1_lrboost_authenticated_science_ops', _layout_path(SCIENCE / 'scripts/fixed25_lrboost_ops.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def _cpu(ops):
    cert = ops.verify_cpu(SCIENCE_MANIFEST, CPU_SHA)
    require(cert['producer_job_id'] == '29778388' and cert['tests'] == 598, 'Wrong original CPU job/count')
    required = {'hpc-contract.xml': 266, 'legacy.xml': 239, 'native-mapping.xml': 34, 'chemistry.xml': 59}
    proofs = {p['path']: p['sha256'] for p in cert['proofs']}
    for name, count in required.items():
        path = ROOT / 'cpu/payload' / name
        require(str(path) in proofs, 'Original CPU certificate omitted raw XML')
        tree = ET.parse(path)
        require(len(list(tree.iter('testcase'))) == count
                and not any(list(tree.iter(tag)) for tag in ('failure', 'error', 'skipped')),
                'Original CPU raw actual testcase rows failed/changed')
    scheduler = ROOT / 'cpu/scheduler-final-0.txt'
    stdout = ROOT / 'cpu/stdout.29778388.log'
    require(str(scheduler) in proofs and str(stdout) in proofs
            and scheduler_status(scheduler.read_text(), '29778388', 'hpc') == 'DONE',
            'Original CPU lacks actual terminal scheduler/LSF proofs')
    lsf_success(stdout, '29778388')
    return cert


def _storage(gate, phase):
    projections = {a: pin['projected_bytes'] for a, pin in ARMS.items()}
    require(gate.get('version') == 'fixed25-three-arm-lrboost-storage-v1'
            and gate.get('status') == 'passed' and gate.get('phase') == phase
            and gate.get('projections') == projections, 'Original whole-campaign storage gate failed/changed')
    expected_pools = {'work3': ('protenix', 'boltz2'), 'zhome': ('boltz2_chemistry',)}
    require(set(gate.get('filesystems', {})) == set(expected_pools), 'Missing per-filesystem storage scope')
    for pool, arms in expected_pools.items():
        budget = gate['filesystems'][pool]
        outstanding = budget.get('outstanding_projected_bytes')
        require(budget.get('status') == 'passed' and set(budget.get('arms', [])) == set(arms)
                and budget.get('full_projected_bytes') == sum(projections[a] for a in arms)
                and type(outstanding) is int and 0 <= outstanding <= budget['full_projected_bytes']
                and budget.get('projected_bytes') == outstanding
                and budget.get('reserve_bytes') == 4 * 1024**3 and budget.get('file_reserve') == 100000
                and budget.get('required_bytes') == outstanding + 4 * 1024**3
                and budget.get('available_bytes_conservative', -1) >= budget['required_bytes']
                and budget.get('available_files_lower', -1) >= 100000, 'Original storage reserve/headroom differs')
    inventory = gate.get('recent_home_inventory', {})
    require(inventory.get('complete') is True and inventory.get('returncode') == 0
            and gate['filesystems']['zhome'].get('filesystem_inode_reporting') == 'reported_physical_filesystem_only',
            'Original home storage scope/inventory was not complete')


def authenticate_inputs(arm):
    """Exact original science, 598 CPU cases, all new readiness, and own entry."""
    pin = ARMS[arm]
    ops = _science_ops()
    _cpu(ops)
    for selected, bound in ARMS.items():
        decision_path = ROOT / ('decision-' + selected + '.json')
        hashed(decision_path, bound['decision'])
        decision = read(decision_path)
        require(decision['native_job'] == bound['native_job'] and decision['assessment_sha256'] == bound['ready'],
                'Actual new native readiness job/hash differs')
        ready = ops.authenticate_ready(selected, SCIENCE_MANIFEST, decision)
        require(ready['model_digest'] == bound['model'] and ready['context_digest'] == bound['context']
                and ready['pocket_definition_sha256'] == POCKET and ready['diffusion_gradient'] is False
                and decision['projected_bytes'] == bound['projected_bytes'], 'Original native frozen/pocket/projection identity differs')
    ops.verify_stopped_predecessors()
    marker_path, jobs_path = ROOT / 'long-dispatch-started.json', ROOT / 'long-submissions.json'
    hashed(marker_path, DISPATCH_SHA)
    hashed(jobs_path, JOBS_SHA)
    marker, jobs = read(marker_path), read(jobs_path)
    require(marker['source_manifest_sha256'] == SCIENCE_MANIFEST and marker['cpu_certificate_sha256'] == CPU_SHA
            and marker['authorization'] == ops.AUTHORIZATION and jobs['all_three_submitted'] is True
            and {j['job_id'] for j in jobs['jobs']} == {p['job'] for p in ARMS.values()}, 'Original dispatch/job set changed')
    _storage(marker['storage_gate'], 'dispatch')
    parent = native_parent(arm)
    submitted_path = parent / 'submission.json'
    hashed(submitted_path, pin['submission'])
    submitted = read(submitted_path)
    require(submitted in jobs['jobs'] and submitted['outcome'] == 'submitted' and submitted['job_id'] == pin['job']
            and submitted['phase'] == 'run' and submitted['arm'] == arm and submitted['output_parent'] == str(parent)
            and submitted['source_manifest_sha256'] == pin['manifest'] and submitted['cpu_certificate_sha256'] == CPU_SHA
            and submitted['learning_rate'] == pin['rate'] and submitted['config_sha256'] == pin['config']
            and submitted['args'] == ops.submission_command(arm, 'run', SCIENCE_MANIFEST, CPU_SHA), 'Actual scientific submission changed')
    hashed(config_path(arm), pin['config'])
    gate_path = ROOT / ('entry-quota-' + arm) / 'lrboost-storage-gate.json'
    invocation_path = parent / 'payload/driver-invocation.json'
    started = study_path(arm).exists() or invocation_path.exists()
    if started:
        hashed(gate_path, pin['entry'])
        _storage(read(gate_path), 'entry')
        hashed(invocation_path, pin['driver'])
        expected = {'args': ops.driver_command(arm, 'run', SCIENCE_MANIFEST, CPU_SHA), 'cwd': str(SCIENCE),
                    'source_manifest_sha256': SCIENCE_MANIFEST, 'cpu_certificate_sha256': CPU_SHA, 'job_id': pin['job']}
        require(read(invocation_path) == expected, 'Original actual driver invocation/config/output differs')
    require(sha(ROOT / 'cpu/payload/composition_reference.json') == REFERENCE
            and sha(ROOT / 'cpu/payload/pocket_definition.json') == POCKET, 'Original pocket/reference bytes changed')
    if arm != 'protenix':
        require(sha(PREPARED / 'context.pt') == BOLTZ_CONTEXT, 'Original fixed detached Boltz pose bytes changed')
    return {'version': VERSION, 'arm': arm, 'native_job': pin['job'], 'source_manifest_sha256': SCIENCE_MANIFEST,
            'cpu_certificate_sha256': CPU_SHA, 'config_sha256': pin['config'], 'readiness_sha256': pin['ready'],
            'submission': hashed(submitted_path), 'entry': hashed(gate_path) if started else None,
            'driver_invocation': hashed(invocation_path) if started else None, 'scientific_execution_started': started}


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
        require(case['states'][-1]['repair_weight'] == 50.0, 'Chemistry endpoint did not reach 50')
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


