"""Fresh three-arm high-rate campaign storage, independent of old run placement.

Only the pure quota/inode/inventory algorithms are reused.  New readiness
projections and new output roots determine both filesystem budgets; each keeps
its own 4 GiB and 100,000-file headroom.  Historical studies are preserved.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from fixed25_split_storage import FILE_RESERVE, RESERVE, inode_observation, tree_bytes
from fixed25_recover_storage import canonical, gate_record, recent_home_bytes


TAG = '20261022-brd4-jq1-25-fixed-lrreset-03'
VERSION = 'fixed25-three-arm-lrreset-storage-v1'
ANALYSIS_ROOT = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = ANALYSIS_ROOT / 'restart-lrreset-03'
HOME_SCOPE = Path('/zhome/95/e/206121/mosaic-runs')
HOME_ROOT = HOME_SCOPE / TAG
RELEASE = ANALYSIS_ROOT / 'restart-lrreset-release-03'
UPLOAD = ANALYSIS_ROOT / (TAG + '.tar.gz')
CONCURRENT_ANALYSIS_ALLOWANCE = 2 * 1024**3 + 512 * 1024**2
ARMS = ('protenix', 'boltz2', 'boltz2_chemistry')
PLACEMENT = {'protenix': 'work3', 'boltz2': 'work3', 'boltz2_chemistry': 'zhome'}
HISTORICAL_WORK_ROOTS = tuple(Path('/work3/s234818/mosaic/runs') / tag for tag in (
    '20261021-brd4-jq1-25-fixed-02', '20261021-brd4-jq1-25-fixed-storage-03',
    '20261021-brd4-jq1-25-fixed-storage-04', '20261021-brd4-jq1-25-fixed-report-01',
    '20261021-brd4-jq1-25-fixed-report-02', '20261021-brd4-jq1-25-fixed-report-03',
    '20261021-brd4-jq1-25-fixed-report-04', '20261021-brd4-jq1-25-fixed-report-05',
    '20261021-brd4-jq1-25-fixed-lrboost-01', '20261022-brd4-jq1-lrboost-analysis-01'))


def output_parent(arm):
    if arm not in ARMS:
        raise ValueError('Unknown fresh high-rate arm')
    return (HOME_ROOT if PLACEMENT[arm] == 'zhome' else ROOT) / ('run-' + arm)


def budget_inputs(projections, current):
    if (set(projections) != set(ARMS) or set(current) != set(ARMS)
            or any(type(v) is not int or v < 1024**3 for v in projections.values())
            or any(type(v) is not int or v < 0 for v in current.values())):
        raise ValueError('All three new authenticated full projections/current output sizes are required')
    if any(current[a] > projections[a] for a in ARMS):
        raise ValueError('Fresh scientific output exceeded its full new readiness projection')
    if PLACEMENT != {'protenix': 'work3', 'boltz2': 'work3', 'boltz2_chemistry': 'zhome'}:
        raise ValueError('Declared fresh output placement differs')
    return {pool: {
        'arms': [a for a in ARMS if PLACEMENT[a] == pool],
        'full_projected_bytes': sum(projections[a] for a in ARMS if PLACEMENT[a] == pool),
        'outstanding_projected_bytes': sum(projections[a] - current[a] for a in ARMS if PLACEMENT[a] == pool),
    } for pool in ('work3', 'zhome')}


CONCURRENT_DIAGNOSTIC_PATHS = (
    Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-02'),
    Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-02'),
    Path('/work3/s234818/mosaic/20261022-brd4-jq1-lrboost-analysis-02.tar.gz'),
)


def stale_work_bytes():
    # The containing analysis root charges new run/release/upload bytes exactly once.
    roots = (*HISTORICAL_WORK_ROOTS, ROOT, RELEASE, UPLOAD, *CONCURRENT_DIAGNOSTIC_PATHS)
    return sum(tree_bytes(p) for p in roots if not any(p != q and p.is_relative_to(q) for q in roots))


def receipt_dir(path):
    path = canonical(path)
    if canonical(ROOT) != ROOT or not path.is_relative_to(ROOT):
        raise ValueError('New campaign quota receipts must remain within its work3 root')
    path.mkdir(parents=True, exist_ok=False)
    return path


def start_record(phase):
    return {'version': VERSION, 'utc': datetime.now(timezone.utc).isoformat(),
            'phase': phase, 'status': 'failed', 'placement': PLACEMENT,
            'output_parents': {a: str(output_parent(a)) for a in ARMS},
            'filesystems': {}, 'raw_quotas': {}, 'filesystem_inode_observations': {},
            'reserve_bytes_per_filesystem': RESERVE, 'file_reserve_per_filesystem': FILE_RESERVE,
            'scope': 'Three fresh lower-rate trajectories; immutable finished studies retained',
            'concurrent_analysis_allowance_bytes': CONCURRENT_ANALYSIS_ALLOWANCE,
            'limitation': 'Fresh conservative observations are not reservations; home inode evidence is physical filesystem scope only'}


def save(path, record):
    with (path / 'lrreset-storage-gate.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_work_storage(receipt, *, phase):
    """CPU gets 1 GiB; three concurrent readiness outputs share a grouped 3 GiB allowance."""
    if phase not in ('preflight', 'cpu', 'validate'):
        raise ValueError('Unknown work3-only validation phase')
    path = receipt_dir(receipt)
    record = start_record(phase)
    try:
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True,
                             check=True, timeout=60).stdout
        observed = inode_observation(os.statvfs(ROOT))
        record['raw_quotas']['work3'] = raw
        record['filesystem_inode_observations']['work3'] = observed
        projected = 1024**3
        if phase == 'validate':
            current = {a: tree_bytes(ROOT / ('validate-' + a)) for a in ARMS}
            if any(n > 1024**3 for n in current.values()):
                raise ValueError('Disposable native output exceeded its 1 GiB validation allowance')
            projected = sum(1024**3 - current[a] for a in ARMS)
            record['current_validation_outputs'] = current
            record['full_validation_allowance_bytes'] = 3 * 1024**3
        gate = gate_record('work3', raw, projected + CONCURRENT_ANALYSIS_ALLOWANCE, stale_work_bytes(), shutil.disk_usage(ROOT).free, observed)
        record['filesystems']['work3'] = gate
        record['status'] = gate['status']
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        save(path, record)
    if record['status'] != 'passed':
        raise ValueError('Fresh validation work3 byte/inode allowance plus reserve does not fit')
    return record


def check_campaign_storage(projections, receipt, *, phase):
    """Fresh whole-campaign grouping before dispatch and every scientific entry."""
    if phase not in ('dispatch', 'entry'):
        raise ValueError('Unknown fresh full campaign storage phase')
    budget_inputs(projections, {a: 0 for a in ARMS})
    for path in (ROOT, HOME_ROOT, HOME_SCOPE):
        canonical(path)
    if not HOME_SCOPE.is_dir():
        raise ValueError('Existing dedicated Mosaic home parent is required')
    path = receipt_dir(receipt)
    record = start_record(phase)
    record['projections'] = dict(projections)
    try:
        # The reused inventory function is a complete no-follow Mosaic-only
        # four-hour scan.  Partial/timeout/unreadable rows cannot certify space.
        recent = recent_home_bytes()
        if (recent.get('complete') is not True or recent.get('returncode') != 0
                or type(recent.get('bytes')) is not int or recent['bytes'] < 0):
            raise ValueError('A completed recent home inventory is required before fresh quota')
        record['recent_home_inventory'] = recent
        current = {a: tree_bytes(output_parent(a)) for a in ARMS}
        record['current_outputs'] = current
        budgets = budget_inputs(projections, current)
        own_home = tree_bytes(HOME_ROOT)
        record['complete_home_root_bytes_subtracted_again'] = own_home
        for pool, command, target, stale in (
            ('work3', 'getquota_work3.sh', ROOT, stale_work_bytes()),
            ('zhome', 'getquota_zhome.sh', HOME_ROOT if HOME_ROOT.exists() else HOME_SCOPE,
             recent['bytes'] + own_home),
        ):
            raw = subprocess.run([command], capture_output=True, text=True, check=True, timeout=60).stdout
            observed = inode_observation(os.statvfs(target))
            record['raw_quotas'][pool] = raw
            record['filesystem_inode_observations'][pool] = observed
            gate = gate_record(pool, raw, budgets[pool]['outstanding_projected_bytes'] + (CONCURRENT_ANALYSIS_ALLOWANCE if pool == 'work3' else 0), stale,
                               shutil.disk_usage(target).free, observed)
            gate.update(budgets[pool], target=str(target))
            record['filesystems'][pool] = gate
        record['status'] = 'passed' if all(g['status'] == 'passed' for g in record['filesystems'].values()) else 'failed'
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        save(path, record)
    if record['status'] != 'passed':
        raise ValueError('Fresh three-arm per-filesystem projection plus own reserves/headroom does not fit')
    return record
