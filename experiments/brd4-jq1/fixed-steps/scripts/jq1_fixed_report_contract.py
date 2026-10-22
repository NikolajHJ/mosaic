"""Exact saved-artifact contract for the three fixed-step JQ1 trajectories.

Importing this module does no scientific work. Authentication is read-only and
uses the original, authenticated launcher implementations on allocated DTU CPU.
Historical report contracts and scientific execution records are never changed.
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


VERSION = 'jq1-25-fixed-saved-report-v1'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
HOME = Path('/zhome/95/e/206121/mosaic-runs')
RELEASES = Path('/work3/s234818/mosaic/releases')
RUNS = Path('/work3/s234818/mosaic/runs')
SCIENCE_TAG = '20261021-brd4-jq1-25-fixed-02'
SCIENCE = RELEASES / SCIENCE_TAG / 'source'
SCIENCE_ROOT = RUNS / SCIENCE_TAG
SCIENCE_MANIFEST = '1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64'
SOURCE03 = RELEASES / '20261021-brd4-jq1-25-fixed-storage-03/source'
ROOT03 = RUNS / '20261021-brd4-jq1-25-fixed-storage-03'
SOURCE04 = RELEASES / '20261021-brd4-jq1-25-fixed-storage-04/source'
ROOT04 = RUNS / '20261021-brd4-jq1-25-fixed-storage-04'
HOME04 = HOME / '20261021-brd4-jq1-25-fixed-storage-04'
SOURCE03_MANIFEST = '9ee7bf1d6bf412acdc37fc83415476dbefdf45db89533571d4a646ee16f33772'
SOURCE04_MANIFEST = '75a76dacf8de25f4ebd2d3b3fe114b1af6092064a01edf141650295300670ab3'
CPU03 = '4b2b9f59d22719771528c9439dfbd338fec3a02922df08190b52bb22b0875bd4'
ADOPTION03 = '7592facd2832058fa81055c72aaa64f06213bb369c68cb39a0d693b1b5496a60'
CPU04 = 'b9f7e4a35289b5b6e7ab240b71fe119ed54ca91ef322736152195f83071408b1'
ADOPTION04 = '58b6b9fc2c55d3f8ebd144f474ab0928145597f9e34ffaa6fa33a68dc08d1cfd'
PRO_OBSERVATION = 'e28a11dd257e68791e8a126c751a504376befdfcdedc1df6d17a78ed47a6b5b7'
PRO_OBSERVER_STDOUT = '1cf1fd7bdb2e7b1554a8553c9f7f137b116b1ec562a0bfbf11a6000487d26309'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
PREPARED = HOME / '20261021-brd4-jq1-25-01/prepare/payload/prepared'
BOLTZ_CONTEXT = 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
REFERENCE = '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
PROJECTIONS = {'protenix': 1493301824, 'boltz2': 1770121824, 'boltz2_chemistry': 5164341824}

ARMS = {
    'protenix': {
        'job': '29777709', 'rate': 0.3, 'metric': 'contact', 'color': '#246bb2',
        'label': 'Protenix · JQ1 pocket contact', 'manifest': SCIENCE_MANIFEST,
        'config': '69257fd9700d8ca0f42e13ae95ec4c3a15ca623064f657acf4fb8c85b65f1d00',
        'ready': '3e1a61215ecb5dd5b7999a05030c3de96b1e811a5f2c1b606af53649e2ca3953',
        'model': '8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2',
        'context': 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4',
        'source': str(SCIENCE), 'science_root': str(SCIENCE_ROOT),
        'output_parent': str(ROOT03 / 'run-protenix'),
        'launcher_source': str(SOURCE03), 'launcher_root': str(ROOT03),
        'launcher_manifest': SOURCE03_MANIFEST, 'launcher_cpu': CPU03, 'adoption': ADOPTION03,
        'submission': '088d768ef618b66c6e38916c76cab4d1da44a690cfba34976dfa27513f3e634d',
        'entry': 'b0f7009c68a0e49d203f34fa82f9636f4a85e7cac1e7a60c84ee747e586ae75d'},
    'boltz2': {
        'job': '29778203', 'rate': 0.01, 'metric': 'affinity', 'color': '#9a4dba',
        'label': 'Boltz-2 · fixed JQ1-pocket pose', 'manifest': SCIENCE_MANIFEST,
        'config': 'b6217159b0180b8c2f4b3e57764b2101b79e20e1e8cd021b357ae6563e5b161b',
        'ready': '903a7facf258e29c868160d325287d6c332b339d7a68bf367182b20fca6ff0f0',
        'model': '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16',
        'context': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4',
        'source': str(SCIENCE), 'science_root': str(SCIENCE_ROOT),
        'output_parent': str(ROOT04 / 'run-boltz2'),
        'launcher_source': str(SOURCE04), 'launcher_root': str(ROOT04),
        'launcher_manifest': SOURCE04_MANIFEST, 'launcher_cpu': CPU04, 'adoption': ADOPTION04,
        'submission': 'e1a75f7b923a5be08e93d70cd0a1198b25d7d150217fc727c4c66da74e9da5b2'},
    'boltz2_chemistry': {
        'job': '29778204', 'rate': 0.01, 'metric': 'affinity', 'color': '#008979',
        'label': 'Boltz-2 + delayed chemistry', 'manifest': SCIENCE_MANIFEST,
        'config': 'dd6347a0756320bc79dea5fbbf7da9aeca8e8bf93dc38f30cd141fd240059049',
        'ready': '5602e2c80dd37c6951356eba04762ceb9799b24ccd6126db7fd8c175ec662e88',
        'model': '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16',
        'context': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4',
        'source': str(SCIENCE), 'science_root': str(SCIENCE_ROOT),
        'output_parent': str(HOME04 / 'run-boltz2_chemistry'),
        'launcher_source': str(SOURCE04), 'launcher_root': str(ROOT04),
        'launcher_manifest': SOURCE04_MANIFEST, 'launcher_cpu': CPU04, 'adoption': ADOPTION04,
        'submission': 'b0b4bedc82c6c4db8162569b04160be067477bcf959788279ae53c1d5bb61676'},
}


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


def source_path(arm):
    return Path(ARMS[arm]['source'])


def config_path(arm):
    return _layout_path(source_path(arm) / ('examples/configs/brd4_jq1_25_fixed_' + arm + '_10000.json'))


def ready_path(arm):
    return Path(ARMS[arm]['science_root']) / ('validate-' + arm) / 'payload/study/assessment.json'


def native_parent(arm):
    return Path(ARMS[arm]['output_parent'])


def study_path(arm):
    return native_parent(arm) / 'payload/study'


def canonical_digest(value):
    digest = hashlib.sha256()
    for part in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value):
        digest.update(part.encode())
    return digest.hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fixed_counts(updates, rate):
    """Count saved outcomes without converting fixed steps into strict descent."""
    require(_finite(rate) and rate in (0.3, 0.01), 'Unexpected declared fixed learning rate')
    counts = dict(applied=len(updates), strict_improving=0, forced_fallback=0, no_op=0,
                  scalar_increased=0, scalar_equal=0, scalar_decreased=0)
    for index, update in enumerate(updates):
        require(update.get('step') == index and update.get('selection_policy') == 'fixed_step', 'Fixed policy/continuity differs')
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


def _recovery_ops():
    """Only authenticated original launchers may supply their evidence validators."""
    _release(SCIENCE, SCIENCE_MANIFEST)
    _release(SOURCE03, SOURCE03_MANIFEST)
    _release(SOURCE04, SOURCE04_MANIFEST)
    paths = {
        'fixed25_split_storage': _layout_path(SOURCE03 / 'scripts/fixed25_split_storage.py'),
        'fixed25_split_ops': _layout_path(SOURCE03 / 'scripts/fixed25_split_ops.py'),
        'fixed25_recover_storage': _layout_path(SOURCE04 / 'scripts/fixed25_recover_storage.py'),
        'fixed25_ops': _layout_path(SCIENCE / 'scripts/fixed25_ops.py'),
        'campaign25_contract': _layout_path(SCIENCE / 'scripts/campaign25_contract.py'),
        'campaign25_pocket_ops': _layout_path(SCIENCE / 'scripts/campaign25_pocket_ops.py'),
    }
    for name, path in paths.items():
        loaded = sys.modules.get(name)
        require(loaded is None or Path(loaded.__file__).resolve() == path, 'Authenticated launcher dependency was shadowed: ' + name)
    previous = list(sys.path)
    try:
        sys.path[:0] = [str(_layout_path(SOURCE04 / 'scripts')), str(_layout_path(SOURCE03 / 'scripts')), str(_layout_path(SCIENCE / 'scripts'))]
        spec = importlib.util.spec_from_file_location('_jq1_fixed_authenticated_recovery', _layout_path(SOURCE04 / 'scripts/fixed25_recover_ops.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def _storage_passed(gate, phase):
    require(gate.get('status') == 'passed' and gate.get('phase') == phase
            and gate.get('projections') == PROJECTIONS, 'Launcher whole-campaign storage gate failed/changed')
    filesystems = gate.get('filesystems', {})
    require(bool(filesystems), 'Missing per-filesystem storage evidence')
    for budget in filesystems.values():
        require(budget.get('status') == 'passed' and budget.get('reserve_bytes') == 4 * 1024**3
                and budget.get('file_reserve') == 100000
                and budget.get('required_bytes') == budget.get('projected_bytes', -1) + 4 * 1024**3
                and budget.get('available_bytes_conservative', -1) >= budget['required_bytes']
                and budget.get('available_files_lower', -1) >= 100000, 'Storage byte/file reserve or headroom differs')
    if 'zhome' in filesystems:
        inventory = gate.get('recent_home_inventory', {})
        require(inventory.get('complete') is True and inventory.get('returncode') == 0,
                'Home storage proof used incomplete inventory')
        require(filesystems['zhome'].get('filesystem_inode_reporting') == 'reported_physical_filesystem_only',
                'Home file-headroom scope changed')


def authenticate_launcher(arm):
    """Authenticate source02 science and independent source03/04 launch provenance.

    Empty queued/preparing arms need no GPU-entry pass. Any actual scientific
    output requires the original exact invocation and passed whole entry gate.
    """
    pin = ARMS[arm]
    recover = _recovery_ops()
    cert = recover.verify_tool_cpu(SOURCE04_MANIFEST, CPU04)
    require(cert['producer_job_id'] == '29778180' and cert['tests'] == 85, 'Wrong allocated recovery CPU identity/count')
    adoption = recover.verify_adoption(ADOPTION04)
    old = recover.legacy_ops()
    cert03 = old.verify_tool_cpu(SOURCE03_MANIFEST, CPU03)
    require(cert03['producer_job_id'] == '29777704' and cert03['tests'] == 65, 'Wrong original launcher CPU identity/count')
    original = adoption['evidence']['original']['arms'][arm]
    require(original['config']['sha256'] == pin['config'] and original['assessment']['sha256'] == pin['ready']
            and original['model_digest'] == pin['model'] and original['context_digest'] == pin['context'],
            'Adopted readiness/config/frozen identity differs')
    parent = native_parent(arm)
    submission_path = parent / 'submission.json'
    submission = read(submission_path)
    proof = hashed(submission_path, pin['submission'])
    command = old.driver_command(arm, python=PYTHON) if arm == 'protenix' else recover.driver_command(arm)
    require(command[command.index('--output') + 1] == str(study_path(arm)), 'Declared driver output root differs')
    if arm == 'protenix':
        expected_args = ['bsub', '-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]',
                         '-W', '23:00', '-J', 'mosaic-jq1-fixed-split-protenix', '-oo', str(parent / 'stdout.%J.log'),
                         '-eo', str(parent / 'stderr.%J.log'), '-gpu', 'num=1:mode=exclusive_process', 'bash', '-l',
                         str(_layout_path(SOURCE03 / 'scripts/hpc/fixed25_split_stage.lsf')), str(SOURCE03), SOURCE03_MANIFEST,
                         str(ROOT03), arm, CPU03, ADOPTION03]
        marker_path = ROOT03 / 'long-dispatch-started.json'
        hashed(marker_path, '7036bf03654d548d337bf0412a1fdf57252debd790c4bce519b9957e60c84b1e')
        gate_path = ROOT03 / 'entry-quota-protenix/split-storage-gate.json'
    else:
        expected_args = recover.submission_command(arm, SOURCE04_MANIFEST, CPU04, ADOPTION04)
        marker_path = ROOT04 / 'long-dispatch-started.json'
        hashed(marker_path, '4951799e66bd3d601e2ec2e75c5060c17521502398443c9adc391b37c7a6baae')
        gate_path = ROOT04 / ('entry-quota-' + arm) / 'recovery-storage-gate.json'
        all_jobs = read(ROOT04 / 'long-submissions.json')
        require(all_jobs.get('both_boltz_submitted') is True and all_jobs.get('existing_protenix_job') == ARMS['protenix']['job']
                and len(all_jobs.get('jobs', [])) == 2 and submission in all_jobs['jobs']
                and {j['job_id'] for j in all_jobs['jobs']} == {ARMS[a]['job'] for a in ('boltz2', 'boltz2_chemistry')},
                'Actual recovery submissions/job set differs')
    require(submission.get('job_id') == pin['job'] and submission.get('outcome') == 'submitted'
            and submission.get('arm') == arm and submission.get('phase') == 'run'
            and submission.get('output_parent') == str(parent) and submission.get('args') == expected_args
            and submission.get('tool_manifest_sha256') == pin['launcher_manifest']
            and submission.get('science_manifest_sha256') == SCIENCE_MANIFEST, 'Actual scientific submission identity/roots/arguments differ')
    marker = read(marker_path)
    require(all(marker.get(k) == v for k, v in {
        'tool_manifest_sha256': pin['launcher_manifest'], 'science_manifest_sha256': SCIENCE_MANIFEST,
        'cpu_certificate_sha256': pin['launcher_cpu'], 'adoption_sha256': pin['adoption']}.items()), 'Dispatch adoption/source/CPU differs')
    _storage_passed(marker['storage_gate'], 'dispatch')
    entry = None
    if gate_path.exists():
        if arm == 'protenix':
            hashed(gate_path, pin['entry'])
        entry = read(gate_path)
    invocation_path = parent / 'payload/driver-invocation.json'
    started = invocation_path.exists() or study_path(arm).exists()
    invocation_proof = None
    if started:
        require(entry is not None, 'Scientific execution lacks entry quota evidence')
        _storage_passed(entry, 'entry')
        expected = {'args': command, 'cwd': str(SCIENCE), 'tool_manifest_sha256': pin['launcher_manifest'],
                    'science_manifest_sha256': SCIENCE_MANIFEST}
        require(read(invocation_path) == expected, 'Actual scientific invocation/source/output differs')
        invocation_proof = hashed(invocation_path)
    return {'version': VERSION, 'arm': arm, 'native_job': pin['job'], 'scientific_source': str(SCIENCE),
            'source_manifest_sha256': SCIENCE_MANIFEST, 'config_sha256': pin['config'], 'readiness_sha256': pin['ready'],
            'launcher_source': pin['launcher_source'], 'launcher_manifest_sha256': pin['launcher_manifest'],
            'launcher_cpu_certificate_sha256': pin['launcher_cpu'], 'adoption_sha256': pin['adoption'],
            'submission': proof, 'output_parent': str(parent), 'entry_status': None if entry is None else entry['status'],
            'entry': None if entry is None else hashed(gate_path), 'driver_invocation': invocation_proof,
            'scientific_execution_started': started}


def authenticate_inputs(arm):
    """All immutable input/readiness pins, including for an empty queued arm."""
    launcher = authenticate_launcher(arm)
    from campaign25_contract import verify_readiness
    pin = ARMS[arm]
    require(sha(config_path(arm)) == pin['config'], 'Fixed configuration changed')
    ready = verify_readiness(ready_path(arm), pin['ready'], config_path(arm), pin['manifest'])
    require(ready['context_digest'] == pin['context'] and ready['model_digest'] == pin['model']
            and ready['pocket_definition_sha256'] == POCKET and ready['diffusion_gradient'] is False
            and ready['runtime_gate']['status'] == 'passed', 'Readiness runtime/frozen/pocket/no-diffusion differs')
    require(sha(SCIENCE_ROOT / 'cpu/payload/composition_reference.json') == REFERENCE
            and sha(SCIENCE_ROOT / 'cpu/payload/pocket_definition.json') == POCKET, 'Original reference/pocket bytes changed')
    if arm != 'protenix':
        require(sha(PREPARED / 'context.pt') == BOLTZ_CONTEXT, 'Fixed detached Boltz context bytes changed')
    return launcher


def verify_case(study, summary, *, steps=10000):
    """Exact terminal wrapper/case/JSONL equality and fixed-step bookkeeping."""
    study = Path(study)
    config = summary['config']
    arm = config['arm']
    require(arm in ARMS, 'Unknown terminal arm')
    case = read(study / 'case-0/report.json')
    require(summary.get('status') == case.get('status') == 'passed', 'Native study/case failed or partial')
    require(len(summary.get('cases', [])) == 1, 'Wrong terminal case count')
    wrapped = summary['cases'][0]
    require(wrapped.get('label') == config['label'] and wrapped.get('initialization') == config['initialization'],
            'Summary case label/initialization wrapper differs')
    require('label' not in case and 'initialization' not in case
            and canonical_digest({k: v for k, v in wrapped.items() if k not in ('label', 'initialization')}) == canonical_digest(case),
            'Summary/case records differ')
    require(case.get('requested_updates') == steps and len(case['updates']) == steps and len(case['states']) == steps + 1,
            'Incomplete update/state budget')
    require(case.get('acceptance_policy') == 'fixed_step' and case.get('nominal_step_size') == ARMS[arm]['rate']
            and case.get('parameterization') == 'categorical_logits_v1' and case.get('key_data') == [0, 17]
            and case.get('repair_weight_schedule') == config.get('repair_schedule')
            and case.get(ARMS[arm]['metric'] + '_objective_identity') == objective_identity(arm),
            'Terminal case policy/key/objective/schedule identity differs')
    require([r['step'] for r in case['updates']] == list(range(steps))
            and [r['step'] for r in case['states']] == list(range(steps + 1)), 'Noncontiguous terminal saved records')
    for name in ('updates', 'states'):
        seen = 0
        with (study / ('case-0/' + name + '.jsonl')).open(encoding='utf-8') as stream:
            for seen, line in enumerate(stream, 1):
                require(seen <= len(case[name]) and line.endswith('\n') and json.loads(line) == case[name][seen - 1],
                        'Saved JSONL differs from terminal case')
        require(seen == len(case[name]), 'Terminal JSONL budget differs')
    coupled = read(study / 'case-0/combined_initial_checks/report.json')
    require(coupled['status'] == 'passed' and coupled['directions'] == case['initial_checks'], 'Initial coupled derivative check differs/failed')
    counts = fixed_counts(case['updates'], ARMS[arm]['rate'])
    verify_coefficients(arm, case['states'], case['updates'])
    require(all(case['counts'].get(k) == counts[k] for k in ('applied', 'strict_improving', 'forced_fallback', 'no_op')),
            'Saved fixed update counts differ')
    return case


def authenticate_study(arm):
    """Full original terminal science gates; frame checks remain renderer work."""
    authenticate_inputs(arm)
    pin, study = ARMS[arm], study_path(arm)
    summary = read(study / 'summary.json')
    require(summary['job_id'] == pin['job'] and summary['mode'] == 'run', 'Wrong scientific job/mode')
    require(summary['source_manifest_sha256'] == pin['manifest'] and summary['config_sha256'] == pin['config']
            and summary['config'] == read(config_path(arm)), 'Scientific source/configuration changed')
    config = summary['config']
    require(config['n_atoms'] == 25 and config['steps'] == 10000 and config['arm'] == arm
            and config['initial_step_size'] == pin['rate'] and config['max_backtracks'] == 1
            and config['acceptance_policy'] == 'fixed_step'
            and summary['optimization_parameters'] == {'step_size': pin['rate'], 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'},
            'Wrong atom/update/arm/fixed-policy scope')
    manifest = read(source_path(arm) / 'source_manifest.json')
    expected_files = {name: value for name, value in manifest['files'].items()
                      if name == 'pyproject.toml' or (PurePosixPath(name).parts[0] in ('src', 'tests', 'examples', 'scripts')
                          and PurePosixPath(name).suffix in ('.py', '.json', '.lsf', '.sh', '.toml', '.txt')
                          and '__pycache__' not in PurePosixPath(name).parts)}
    require(summary['source'].get('files') == expected_files
            and summary['source'].get('git_revision_reference') == manifest['git_revision_reference']
            and summary['source'].get('execution_content_sha256') == canonical_digest(expected_files), 'Complete executed source file map differs')
    require(summary.get('pocket_definition_sha256') == POCKET and summary.get('diffusion_gradient') is False
            and sha(study / 'pocket_definition.json') == POCKET, 'Pocket/no-diffusion identity changed')
    require(summary['pocket_validation_status'] == 'passed' and summary['native_hard_parity']['status'] == 'passed',
            'Native hard parity/pocket gate failed')
    require(summary['native_hard_parity'].get('n_atoms') == 25 and summary['native_hard_parity'].get('atol') == 2e-5
            and summary.get('initialization_sha256') == sha(study / 'initialization.npz'),
            'Native hard parity scope/initialization bytes differ')
    if arm == 'protenix':
        match = summary.get('baseline_initialization_match', {})
        require(match.get('status') == 'passed' and match.get('job') == '29770179'
                and match.get('sha256') == '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf',
                'Protenix original initial arrays/key identity differs')
    for field in ('model', 'context'):
        require(summary[field + '_digest'] == summary['frozen_identity_after'][field + '_digest'] == pin[field], 'Frozen before/after identity differs')
    if arm != 'protenix':
        require(summary['frozen_identity_after'].get('status') == 'passed' and summary['context_file_sha256'] == BOLTZ_CONTEXT,
                'Fixed Boltz context/frozen worker response differs')
    for field in ('native_gradient_checks', 'terminal_gradient_checks'):
        require(summary[field]['status'] == 'passed' and read(study / field / 'report.json') == summary[field],
                'Initial/terminal native derivative gate failed/differs')
    if arm == 'boltz2_chemistry':
        ready = ready_path(arm)
        require(read(ready.with_name('summary.json'))['chemistry_assay']['status'] == 'passed', 'Native chemistry assay failed')
        for label in ('low', 'high'):
            require(read(ready.parent / (label + '_coupled_checks/report.json'))['status'] == 'passed', 'Native coupled chemistry readiness failed')
    case = verify_case(study, summary)
    if arm == 'boltz2_chemistry':
        require(case['states'][-1]['repair_weight'] == 50.0, 'Chemistry schedule did not reach 50')
    else:
        require(all(s['repair_weight'] == 0 for s in case['states']), 'Scorer-only arm acquired chemistry guidance')
    return study, summary, case


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


def terminal_evidence(arm, completions=None):
    """Resolve actual terminal proof bytes; never synthesize an aged-out DONE."""
    pin = ARMS[arm]
    assessment_path = None if completions is None else Path(completions) / ('complete-' + arm) / 'payload/assessment.json'
    if assessment_path is not None and assessment_path.exists():
        record = read(assessment_path)
        require(record.get('status') == 'passed' and record.get('native_job') == pin['job'] and record.get('arm') == arm
                and record.get('n_atoms') == 25 and record.get('updates') == 10000 and record.get('states') == 10001
                and record.get('source_manifest_sha256') == pin['manifest'], 'Missing exact passed native completion proof')
        proofs = record.get('proofs', [])
        require(proofs and len({p['path'] for p in proofs}) == len(proofs), 'Missing/duplicated completion proofs')
        for proof in proofs:
            hashed(proof['path'], proof['sha256'])
        schedulers = [Path(p['path']) for p in proofs if Path(p['path']).name.startswith(('scheduler-final', 'native-scheduler'))]
        stdout = [Path(p['path']) for p in proofs if Path(p['path']).name == 'native-stdout.txt']
        require(len(stdout) == 1 and any(scheduler_status(p.read_text(), pin['job']) == 'DONE' for p in schedulers),
                'Completion omitted authentic same-job final scheduler/LSF')
        lsf_success(stdout[0], pin['job'])
        required = [study_path(arm) / p for p in ('summary.json', 'case-0/report.json', 'case-0/states.jsonl', 'case-0/updates.jsonl')]
        require(all(str(path) in {p['path'] for p in proofs} for path in required), 'Completion omits original terminal JSON/JSONL proofs')
        authenticate_study(arm)
        return {'status': 'DONE', 'source': 'completion_collector', 'assessment': hashed(assessment_path),
                'proofs': proofs, 'scheduler': next(p for p in proofs if Path(p['path']) in schedulers), 'stdout': next(p for p in proofs if Path(p['path']) == stdout[0])}
    require(arm == 'protenix', 'Live scheduler history unavailable; no authenticated completion collector')
    path = ROOT04 / 'pro-terminal-observation.json'
    hashed(path, PRO_OBSERVATION)
    record = read(path)
    require(record.get('version') == 'fixed25-two-arm-storage-recovery-v1' and record.get('status') == 'observed_success'
            and record.get('native_job') == pin['job'] and record.get('observer_job') == '29778184'
            and record.get('tool_manifest_sha256') == SOURCE04_MANIFEST and record.get('cpu_certificate_sha256') == CPU04
            and record.get('adoption_sha256') == ADOPTION04 and record.get('updates') == 10000 and record.get('states') == 10001,
            'Protenix persisted terminal observer identity/status differs')
    proofs = record['proofs']
    require(len(proofs) == 9 and len({p['path'] for p in proofs}) == 9
            and record['scheduler'] in proofs and record['stdout'] in proofs
            and record['scheduler']['path'] == str(ROOT04 / 'pro-observer/native-scheduler-0.txt')
            and record['stdout']['path'] == str(native_parent(arm) / ('stdout.' + pin['job'] + '.log')),
            'Persisted observer scheduler/LSF omitted or escaped')
    for proof in proofs:
        hashed(proof['path'], proof['sha256'])
    require(scheduler_outcome(Path(record['scheduler']['path']).read_text(), pin['job']) == 'DONE', 'Observer lacks actual native DONE')
    lsf_success(record['stdout']['path'], pin['job'])
    observer_stdout = ROOT04 / 'pro-observer/stdout.29778184.log'
    hashed(observer_stdout, PRO_OBSERVER_STDOUT)
    lsf_success(observer_stdout, '29778184')
    observer_receipt = read(ROOT04 / 'observer-submission.json')
    require(observer_receipt.get('job_id') == '29778184' and observer_receipt.get('outcome') == 'submitted'
            and observer_receipt.get('producer_job') == pin['job'], 'Observer actual submission differs')
    authenticate_study(arm)
    return {'status': 'DONE', 'source': 'authenticated_original_pro_observer', 'assessment': hashed(path),
            'proofs': proofs + [hashed(observer_stdout)], 'scheduler': record['scheduler'], 'stdout': record['stdout']}


def status_or_completion(raw, arm, completions=None):
    if re.search(r'Status<[^>]+>', re.sub(r'\s+', '', raw)):
        return scheduler_status(raw, ARMS[arm]['job'])
    return terminal_evidence(arm, completions)['status']
