"""Prospective unchanged-setting chemistry production metadata contracts.

This module has no numerical imports and confers no readiness or launch pass.
The complete comparison, original readiness, startup and storage proofs remain
separate required allocated evidence.
"""
from pathlib import PurePosixPath
import re

VERSION = 'chemistry-accelerated-production-v4'
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT = BASE + '/chemistry-production-04'
SOURCE = BASE + '/chemistry-production-release-04/source'
PILOTS = {
    'protenix': {'root':BASE+'/chemistry-full-model-05',
                 'source':BASE+'/chemistry-full-model-release-05/source'},
    'boltz': {'root':BASE+'/chemistry-full-model-02',
              'source':BASE+'/chemistry-full-model-release-02/source'},
}
ADAPTER_FILES = tuple('scripts/'+name+'.py' for name in (
    'exact_reuse','full_model_contract','full_model_pilot','gpu_local_bridge',
    'gpu_local_worker','gpu_matrix','gpu_process_admission','hybrid_search',
    'parallel_search','profile_oracle','round2_data')) + ('plans/protenix.json','plans/boltz.json')
PROFILES = {
    'protenix': {'arm': 'random', 'profile': 'protenix-random50k', 'segments': 5,
                 'target': 50000, 'rate': .3, 'native_arm': 'protenix',
                 'source': BASE + '/protenix-extended-release-03/source',
                 'manifest': 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562'},
    'boltz': {'arm': 'boltz2_chemistry', 'profile': 'boltz-chem30k', 'segments': 3,
              'target': 30000, 'rate': .0025, 'native_arm': 'boltz2_chemistry',
              'source': BASE + '/boltz-lr0025-release-03/source',
              'manifest': 'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'},
}
GIB = 1024**3
# Only this new source and CPU/control records. Each arm's complete tree,
# including its metadata, is already charged to that arm's full projection.
CONTROL_CAP = 64 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    require(type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None,
            'Missing exact immutable SHA256')
    return value


def validate_plan(plan):
    require(type(plan) is dict and set(plan) == {'version','pilots'}, 'Production plan fields differ')
    require(plan['version'] == VERSION, 'Undeclared chemistry production version')
    require(type(plan['pilots']) is dict and set(plan['pilots']) == set(PROFILES),
            'Both exact full-model pilot IDs required')
    jobs=[]
    for arm in PROFILES:
        row=plan['pilots'][arm]
        require(type(row) is dict and set(row)=={'root','source','manifest_sha256','cpu_certificate_sha256','job_id'},
                'Per-arm producer fields differ')
        require(row['root']==PILOTS[arm]['root'] and row['source']==PILOTS[arm]['source'],
                'Cross-arm, failed or undeclared pilot producer source/root')
        digest(row['manifest_sha256']);digest(row['cpu_certificate_sha256'])
        jobs.append(row['job_id'])
    require(all(type(job) is str and re.fullmatch('[1-9][0-9]*', job) is not None for job in jobs)
            and len(set(jobs)) == 2, 'Pilot IDs must be actual distinct submissions')
    return plan


def profile(arm):
    require(arm in PROFILES, 'Only the two approved chemistry arms may execute')
    return PROFILES[arm]


def stage_label(arm, phase, index):
    spec = profile(arm)
    require(phase in ('native', 'after-native', 'segment', 'after-segment')
            and type(index) is int and 0 <= index < spec['segments'], 'Invalid production boundary')
    require(index > 0 or phase not in ('native', 'after-native'),
            'Initial native readiness comes only from the authenticated new combined pilot')
    return spec['arm'] + '-' + phase + '-' + format(index, '04d')


def projection(unit, arm):
    spec = profile(arm)
    require(type(unit) is int and unit > 0, 'An actual pilot empirical byte maximum is mandatory')
    # Identical to the validated retention.full_projection, without importing it
    # before its immutable source has been authenticated.
    return 2 * unit * spec['segments'] * (10000 // 20 + 3) + spec['segments'] * GIB


def resources(phase, arm=None):
    if phase in ('native', 'segment'):
        profile(arm)
        mode = ('num=1:mode=exclusive_process' if arm == 'protenix'
                else 'num=1:mode=shared:j_exclusive=yes:mps=no')
        return ['-q', 'gpul40s', '-n', '8', '-R', 'span[hosts=1]', '-R',
                'rusage[mem=4GB]', '-gpu', mode, '-W', '02:00' if phase == 'native' else '23:00']
    require(phase in ('cpu', 'after-cpu', 'after-native', 'after-segment'), 'Unknown allocated phase')
    return ['-q', 'hpc', '-n', '4' if phase == 'cpu' else '1', '-R',
            'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '01:00' if phase == 'cpu' else '00:30']


def callback_dependency(cpu_job, pilots):
    require(re.fullmatch('[1-9][0-9]*', cpu_job or '') is not None, 'Actual own CPU ID required')
    validate_plan({'version':VERSION,'pilots':pilots})
    # Both actual pilots are independently authenticated inside the CPU gate.
    # Aged-out Boltz LSF evidence is not an LSF missing-job dependency.
    return 'ended(' + cpu_job + ')'


def cpu_modules(counts):
    require(type(counts) is dict and counts, 'Nonempty exact CPU class counts required')
    modules = set()
    for name, count in counts.items():
        require(type(name) is str and re.fullmatch(r'tests\.test_[a-z0-9_]+\.[A-Za-z][A-Za-z0-9_]*', name)
                and type(count) is int and count > 0, 'Malformed CPU class/count')
        modules.add(name.split('.')[1])
    return sorted(modules)


def require_storage(record):
    require(record.get('status') == 'passed' and record.get('reserve_bytes') == 4*GIB
            and record.get('file_reserve') == 100000 and record.get('original_gate_changed') is False,
            'Current complete inherited storage guard and reserves must pass')
    for name in ('required_bytes', 'available_bytes'):
        require(type(record.get(name)) is int and record[name] >= 0, 'Malformed finite storage result')
    require(record['available_bytes'] >= record['required_bytes'], 'Whole chemistry production does not fit')


def canonical(path):
    require(type(path) is str and '\\' not in path and PurePosixPath(path).is_absolute()
            and '..' not in PurePosixPath(path).parts, 'Require canonical absolute input/output path')
    return path
