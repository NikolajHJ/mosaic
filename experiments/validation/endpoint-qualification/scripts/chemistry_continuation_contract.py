"""Closed metadata contracts for unchanged-setting chemistry continuations."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
SCOPE = BASE/'nonchem-precision-continuation-26'
ROOT = SCOPE/'chemistry-continuation-04'
OUTPUTS = BASE/'chemistry-continuation-04'
SOURCE = SCOPE/'chemistry-continuation-release-04/source'
CURRENT = BASE/'nonchem-precision-release-26/source'
CURRENT_PIN = '8324c1d312adeaa4c67c475c01ef1217df6f3ad4044b997d1b73e168325412d3'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
VERSION = 'chemistry-reference-continuation-v1'
GIB = 1024**3
ARMS = {'protenix': {'rate': .3, 'start': 10000, 'target': 50000, 'remaining': 40000},
        'boltz': {'rate': .0025, 'start': 10000, 'target': 30000, 'remaining': 20000}}


def require(value, message):
    if not value: raise ValueError(message)


def raw(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),
            'Ordinary absolute proof required')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def bound(row):
    require(type(row) is dict and set(row)=={'path','bytes','sha256'} and type(row['bytes']) is int
            and row['bytes']>=0 and re.fullmatch('[a-f0-9]{64}',row['sha256'] or ''), 'Exact proof binding required')
    require(binding(row['path'])==row, 'Proof/input changed: '+row['path']);return Path(row['path'])


def write(path, value):
    with Path(path).open('xb') as stream:stream.write(raw(value))


def validate_plan(plan):
    require(plan.get('version')==VERSION and plan.get('root')==str(ROOT) and plan.get('source_root')==str(SOURCE),
            'Chemistry continuation owner/version differs')
    require(plan.get('original_results_changed') is False and plan.get('optimization_settings_changed') is False
            and plan.get('original_failures_preserved') is True, 'Original results/settings/failures must be preserved')
    require(set(plan.get('arms',{}))==set(ARMS), 'Exactly both chemistry arms required')
    for name, expected in ARMS.items():
        arm=plan['arms'][name]
        require(all(arm.get(k)==v for k,v in expected.items()),'Original boundary/rate/target changed: '+name)
        require(arm.get('id')==name and arm.get('chemistry_enabled') is True
                and arm.get('element_js_weight')==0 and arm.get('confidence_margin_enabled') is False,
                'Chemistry/composition/new-margin policy changed')
        require(arm.get('matrix_stride')==20 and arm.get('loss_stride')==1 and arm.get('max_backtracks')==1
                and arm.get('acceptance_policy')=='fixed_step', 'Retention or fixed-step policy changed')
        require(arm.get('repair_schedule')=={'version':'delayed-geometric-v1','initial':.001,'final':100.,
                'hold_updates':5000,'total_updates':expected['target']}, 'Original global chemistry ramp changed')
    require(plan.get('raw_policy')=='prospective-endpoint-admission-v3'
            and plan.get('logit_policy')=='prospective-fixed-reference-logit-admission-v1',
            'Previously authorized prospective check policy differs')
    require(plan.get('component_only_can_submit') is False and plan.get('both_complete_pilots_required') is True
            and plan.get('actual_coefficient_check_required') is True, 'Combined/pilot prerequisites removed')
    return plan


def configured_arm(plan, name):
    validate_plan(plan);require(name in ARMS,'Unknown chemistry arm');return deepcopy(plan['arms'][name])


def frozen_identity(arm,summary,expected,original_verify):
    """Use the original Boltz response contract, retaining its transport fields.

    A successful worker response contains id/status/elapsed_seconds as well as
    the two immutable digests. Exact comparison with a two-field dictionary is
    therefore invalid for Boltz. No field or scientific response is modified.
    Protenix's local identity remains the original exact two-field dictionary.
    """
    require(arm in ARMS and set(expected)=={'model_digest','context_digest'},'Unknown frozen identity scope')
    require(summary['model_digest']==expected['model_digest'] and summary['context_digest']==expected['context_digest'],
            'Original frozen model/context digest differs')
    after=summary['frozen_identity_after']
    if arm=='boltz':original_verify(after,expected['model_digest'],expected['context_digest'])
    else:require(after==expected,'Original Protenix local frozen identity differs')
    return True


def label(phase, arm='all'):
    require(phase in ('cpu','after-cpu','validate','after-validate','segment','after-segment'), 'Unknown chemistry phase')
    if phase in ('cpu','after-cpu'):
        require(arm=='all','CPU scope differs');return phase
    require(arm in ARMS,'Unknown chemistry arm');return arm+'-'+phase


def allocation(phase):
    require(phase in ('cpu','after-cpu','validate','after-validate','segment','after-segment'),'Unknown allocation phase')
    gpu=phase in ('validate','segment')
    # Whole40000/20000 child JSON/JSONL certification needs16GiB host RAM;
    # this CPU callback never occupies a GPU or changes scientific operations.
    slots=8 if gpu else (4 if phase in ('cpu','after-segment') else 1)
    return {'queue':'gpul40s' if gpu else 'hpc','slots':slots,
            'memory_per_slot':'4GB' if slots in (4,8) else '2GB'}


def output_projection(unit,updates,actual_bytes=0):
    """Exact qualified stride20 forecast, never a reduced quota/reserve gate."""
    require(type(unit) is int and unit>0 and type(updates) is int and updates in (20000,40000)
            and type(actual_bytes) is int and actual_bytes>=0,'Exact full chemistry horizon/byte observation required')
    full=2*unit*((updates+19)//20+3)+GIB
    require(actual_bytes<=full,'Actual scientific output exceeds the admitted complete cap')
    return {'full_bytes':full,'remaining_growth_bytes':full-actual_bytes,
            'matrix_bundles':(updates+19)//20+3,'text_allowance_bytes':GIB,
            'loss_stride':1,'matrix_stride':20,'safety_factor':2,'updates':updates}


def runtime(summary, arm):
    """Full remaining budget, upward-rounded measured allocation, no fallback."""
    require(arm in ARMS,'Unknown runtime arm')
    gate=summary.get('runtime_gate',{});pilots=gate.get('mean_update_seconds')
    require(gate.get('status')=='passed' and type(pilots) is list and len(pilots)==2
            and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in pilots),
            'BOTH complete pilot timings required')
    elapsed=summary.get('elapsed_seconds')
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0, 'Complete validation runtime missing')
    steps=ARMS[arm]['remaining'];per_update=1.5*max(pilots)
    fixed=2*elapsed+1800;projected=steps*per_update+fixed
    walltime=int(math.ceil((projected+600)/900)*900)
    require(walltime<=82800,'Full remaining horizon does not fit; new largest-fitting plan required, no silent shortening')
    return {'status':'passed','requested_updates':steps,'complete_pilot_mean_seconds':pilots,
            'per_update_safety_factor':1.5,'complete_validation_seconds':elapsed,
            'fixed_seconds':fixed,'projected_seconds':projected,'walltime_seconds':walltime,
            'walltime':f'{walltime//3600:02d}:{walltime%3600//60:02d}'}
