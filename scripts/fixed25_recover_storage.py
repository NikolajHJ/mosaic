"""Declared storage-only recovery for the two unstarted fixed-step Boltz arms.

Protenix stays in its existing storage03 work3 run. New Boltz-2 output uses
work3, and new Boltz-2+chemistry output uses a dedicated home Mosaic directory.
All three original full projections remain pinned; each filesystem must fit
its assigned remaining output, its own 4 GiB reserve, and 100,000-file headroom.
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

# Root-owned wrappers resolve this unchanged helper from authenticated source03.
from fixed25_split_storage import (FILE_RESERVE, RESERVE, VERSION as ORIGINAL_STORAGE_VERSION,
                                   inode_headroom, inode_observation, parse_work3, tree_bytes)

if ORIGINAL_STORAGE_VERSION != 'fixed25-work3-storage-adoption-v2':
    raise ValueError('Recovery requires the authenticated original storage03 v2 helpers')

TAG = '20261021-brd4-jq1-25-fixed-storage-04'
VERSION = 'fixed25-two-arm-recovery-storage-v1'
ROOT = Path('/work3/s234818/mosaic/runs') / TAG
HOME_SCOPE = Path('/zhome/95/e/206121/mosaic-runs')
HOME_ROOT = HOME_SCOPE / TAG
SCIENCE_ROOT = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-02')
ACTIVE_ROOT = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-03')
WORK_RELEASE = Path('/work3/s234818/mosaic/releases') / TAG
WORK_UPLOAD = Path('/work3/s234818/mosaic') / (TAG + '.tar.gz')
NEW_ARMS = ('boltz2', 'boltz2_chemistry')
PLACEMENT = {'protenix': 'work3', 'boltz2': 'work3', 'boltz2_chemistry': 'zhome'}
FULL_PROJECTIONS = {'protenix': 1493301824, 'boltz2': 1770121824, 'boltz2_chemistry': 5164341824}
EXISTING_PROTENIX_JOB = '29777709'
MAX_HOME_CACHE_SECONDS = 4 * 60 * 60


def output_parent(arm):
    if arm == 'protenix':
        return ACTIVE_ROOT / 'run-protenix'
    if arm == 'boltz2':
        return ROOT / 'run-boltz2'
    if arm == 'boltz2_chemistry':
        return HOME_ROOT / 'run-boltz2_chemistry'
    raise ValueError('Unknown declared recovery arm')


def new_output_parent(arm):
    if arm not in NEW_ARMS:
        raise ValueError('Existing Protenix must never be submitted or restarted by recovery')
    return output_parent(arm)


def canonical(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('Storage root or parent alias changes its declared identity')
    return path


def validate_projection(projections):
    if set(projections) != set(FULL_PROJECTIONS) or any(type(n) is not int for n in projections.values()):
        raise ValueError('All three authenticated integer full projections are required')
    if projections != FULL_PROJECTIONS:
        raise ValueError('Original full output projections cannot be reduced or changed')
    if PLACEMENT != {'protenix': 'work3', 'boltz2': 'work3', 'boltz2_chemistry': 'zhome'}:
        raise ValueError('Recovery placement is an immutable declaration')
    return dict(projections)


def budget_inputs(projections, current_outputs):
    projections = validate_projection(projections)
    if set(current_outputs) != set(projections) or any(type(n) is not int or n < 0 for n in current_outputs.values()):
        raise ValueError('Exact nonnegative current output sizes are required for all three arms')
    if any(current_outputs[a] > projections[a] for a in projections):
        raise ValueError('An existing or new arm exceeded its authenticated full projection')
    return {pool: {'arms': [a for a in PLACEMENT if PLACEMENT[a] == pool],
                   'full_projected_bytes': sum(projections[a] for a in PLACEMENT if PLACEMENT[a] == pool),
                   'outstanding_projected_bytes': sum(max(projections[a] - current_outputs[a], 0)
                                                      for a in PLACEMENT if PLACEMENT[a] == pool)}
            for pool in ('work3', 'zhome')}


def stale_work_bytes():
    # The active Protenix and failed Boltz entry records remain intact in root03.
    return sum(tree_bytes(p) for p in (SCIENCE_ROOT, ACTIVE_ROOT, ROOT, WORK_RELEASE, WORK_UPLOAD))


def recent_home_bytes():
    """Complete read-only no-follow GNU find; count F/link metadata within 4h.

    This command never traverses symlinks, never reads outside Mosaic home
    paths, and must finish successfully. Partial output, timeout, unknown
    sizes, or unreadable paths cannot certify a storage pass.
    """
    scope = canonical(HOME_SCOPE)
    if not scope.is_dir():
        raise ValueError('Existing Mosaic home parent is required; no parent is created by a gate')
    command = ['find', '-P', str(scope), '-mmin', '-240', '(', '-type', 'f', '-o', '-type', 'l', ')',
               '-printf', '%s\n']
    reply = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if reply.returncode != 0:
        raise ValueError('Complete recent Mosaic home byte inventory failed; partial results rejected')
    if reply.stdout and not reply.stdout.endswith('\n'):
        raise ValueError('Recent home inventory has an incomplete size record')
    rows = reply.stdout.splitlines()
    if any(not re.fullmatch(r'\d+', row) for row in rows):
        raise ValueError('Recent home inventory has an unknown/negative size')
    return {'command': command, 'returncode': reply.returncode, 'bytes': sum(int(row) for row in rows),
            'files_or_links': len(rows), 'stderr': reply.stderr,
            'stdout_sha256': hashlib.sha256(reply.stdout.encode('utf-8')).hexdigest(),
            'policy': 'GNU find -P; full regular-file/symlink lstat sizes modified within 240 minutes; no symlink traversal',
            'timeout_seconds': 600, 'complete': True}


def parse_home(raw, *, now=None):
    matches = re.findall(r'using\s+(\d+(?:\.\d+)?)G\s+of\s+(\d+(?:\.\d+)?)G\s+quota', raw, re.I)
    stamps = re.findall(r'Last Update:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', raw)
    if len(matches) != 1 or len(stamps) != 1:
        raise ValueError('One finite home quota and one cache timestamp are required')
    used, hard = (Decimal(x) for x in matches[0])
    if not used.is_finite() or not hard.is_finite() or not 0 <= used < hard:
        raise ValueError('Invalid or exhausted finite home byte quota')
    snapshot = datetime.strptime(stamps[0], '%Y-%m-%d %H:%M:%S').replace(tzinfo=ZoneInfo('Europe/Copenhagen'))
    age = (time.time() if now is None else now) - snapshot.timestamp()
    if not 0 <= age <= MAX_HOME_CACHE_SECONDS:
        raise ValueError('Home quota cache timestamp is future or older than four hours')
    upper = int(used.to_integral_value(rounding=ROUND_CEILING)) * 1024**3
    lower = int(hard.to_integral_value(rounding=ROUND_FLOOR)) * 1024**3
    return {'used_bytes_upper': upper, 'hard_bytes_lower': lower, 'available_bytes_lower': lower - upper,
            'reported_snapshot': snapshot.isoformat(), 'cache_age_seconds': age,
            'maximum_cache_age_seconds': MAX_HOME_CACHE_SECONDS, 'reported_user_inode_quota': False}


def home_inode_headroom(observed):
    keys = {'f_files', 'f_ffree', 'f_favail'}
    if set(observed) != keys or any(type(n) is not int or n < 0 for n in observed.values()):
        raise ValueError('Unknown home filesystem inode tuple')
    total, free, available = (observed[k] for k in ('f_files', 'f_ffree', 'f_favail'))
    if total <= 0 or not 0 <= available <= free <= total:
        raise ValueError('Home requires coherent positive reported physical filesystem inode totals')
    return {'available_files_lower': available, 'filesystem_inode_reporting': 'reported_physical_filesystem_only',
            'file_headroom_scope': 'Physical home filesystem f_favail; no per-user inode quota is reported by DTU home quota',
            'inode_limitation': 'This is physical filesystem headroom, not an invented user inode quota or a reservation'}


def gate_record(pool, raw, projected, stale_bytes, disk_free, observed, *, now=None):
    if pool not in ('work3', 'zhome'):
        raise ValueError('Unknown recovery filesystem')
    if any(type(n) is not int or n < 0 for n in (projected, stale_bytes, disk_free)):
        raise ValueError('Invalid filesystem byte accounting input')
    quota = parse_work3(raw) if pool == 'work3' else parse_home(raw, now=now)
    inode = inode_headroom(quota, observed) if pool == 'work3' else home_inode_headroom(observed)
    available = min(quota['available_bytes_lower'], disk_free) - stale_bytes
    required = projected + RESERVE
    return {'status': 'passed' if available >= required and inode['available_files_lower'] >= FILE_RESERVE else 'failed',
            'filesystem': pool, 'raw_quota': raw, 'quota': quota, 'projected_bytes': projected,
            'stale_bytes_subtracted': stale_bytes, 'disk_free_bytes': disk_free,
            'available_bytes_conservative': available, 'required_bytes': required,
            'reserve_bytes': RESERVE, 'file_reserve': FILE_RESERVE,
            'filesystem_inode_observation': observed, **inode,
            'limitation': 'Observations are conservative budget checks, not storage reservations'}


def _record_start(phase):
    return {'version': VERSION, 'utc': datetime.now(timezone.utc).isoformat(), 'phase': phase, 'status': 'failed',
            'placement': PLACEMENT, 'existing_protenix_job': EXISTING_PROTENIX_JOB,
            'existing_protenix_parent': str(output_parent('protenix')),
            'filesystems': {}, 'raw_quotas': {}, 'filesystem_inode_observations': {},
            'inode_policies': {
                'work3': 'Original storage03 v2: mandatory all-finite user-quota minimum; coherent supplemental FS totals when reported',
                'zhome': 'Positive coherent physical FS inode totals required; no user inode quota is reported and work3 counts cannot be borrowed'},
            'inode_limitation': 'Home headroom is physical-filesystem evidence only; no quota or free-space observation is a reservation',
            'scope': 'Only two new Boltz outputs; original Protenix continues unchanged; no old failure is reclassified'}


def _receipt_dir(receipt_dir):
    if canonical(ROOT) != ROOT:
        raise ValueError('Wrong control root identity')
    path = canonical(receipt_dir)
    if not path.is_relative_to(ROOT):
        raise ValueError('Recovery receipt path escapes dedicated work3 control root')
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save(path, record):
    with (path / 'recovery-storage-gate.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')


def check_cpu_storage(root, receipt_dir):
    """CPU entry/preflight needs work3 only: 1 GiB allowance plus 4 GiB reserve."""
    if Path(root).resolve() != ROOT:
        raise ValueError('Wrong recovery CPU control root')
    path = _receipt_dir(receipt_dir)
    record = _record_start('cpu')
    try:
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60).stdout
        observed = inode_observation(os.statvfs(ROOT))
        record['raw_quotas']['work3'] = raw
        record['filesystem_inode_observations']['work3'] = observed
        gate = gate_record('work3', raw, 1024**3, stale_work_bytes(), shutil.disk_usage(ROOT).free, observed)
        record['filesystems']['work3'] = gate
        record['status'] = gate['status']
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        _save(path, record)
    if record['status'] != 'passed':
        raise ValueError('Recovery CPU work3 byte/inode allowance does not fit')
    return record


def check_campaign_storage(projections, receipt_dir, *, phase):
    """Gate both filesystems and include the running Protenix exposure on work3."""
    projections = validate_projection(projections)
    if phase not in ('dispatch', 'entry'):
        raise ValueError('Unknown recovery storage phase')
    canonical(ROOT); canonical(HOME_ROOT); canonical(ACTIVE_ROOT); canonical(SCIENCE_ROOT)
    if not canonical(HOME_SCOPE).is_dir():
        raise ValueError('Existing Mosaic home parent must be authenticated before recovery')
    path = _receipt_dir(receipt_dir)
    record = _record_start(phase)
    record['projections'] = projections
    try:
        # Complete the bounded read-only inventory before the fresh quota query.
        recent = recent_home_bytes()
        if (recent.get('complete') is not True or recent.get('returncode') != 0 or
                type(recent.get('bytes')) is not int or recent['bytes'] < 0):
            raise ValueError('Incomplete/invalid recent home inventory cannot certify recovery')
        record['recent_home_inventory'] = recent
        own_home = tree_bytes(HOME_ROOT)
        record['complete_home_root_bytes_subtracted_again'] = own_home
        current = {a: tree_bytes(output_parent(a)) for a in PLACEMENT}
        record['current_outputs'] = current
        budgets = budget_inputs(projections, current)
        for pool, command, target, stale in (
            ('work3', 'getquota_work3.sh', ROOT, stale_work_bytes()),
            ('zhome', 'getquota_zhome.sh', HOME_ROOT if HOME_ROOT.exists() else HOME_SCOPE, recent['bytes'] + own_home),
        ):
            raw = subprocess.run([command], capture_output=True, text=True, check=True, timeout=60).stdout
            observed = inode_observation(os.statvfs(target))
            record['raw_quotas'][pool] = raw
            record['filesystem_inode_observations'][pool] = observed
            gate = gate_record(pool, raw, budgets[pool]['outstanding_projected_bytes'], stale,
                               shutil.disk_usage(target).free, observed)
            gate.update(budgets[pool], target=str(target), stale_quota_scope=(
                'Full science02/root03/new04/release04/upload04 bytes'
                if pool == 'work3' else 'Complete last-four-hour Mosaic-only writes plus complete new home root charged again'))
            record['filesystems'][pool] = gate
        record['status'] = 'passed' if all(g['status'] == 'passed' for g in record['filesystems'].values()) else 'failed'
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        _save(path, record)
    if record['status'] != 'passed':
        raise ValueError('Recovery grouped per-filesystem byte/inode projection plus reserve does not fit')
    return record
