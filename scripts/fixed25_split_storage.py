"""All-work3 storage adoption after authenticated Mosaic archival cleanup.

This new launcher retains the original source02 scientific proof and failed
storage result. All THREE fresh arms share one whole-campaign projection and
one unchanged 4 GiB reserve plus 100,000 free inode headroom. Current fixed02,
new controller/output, and new tool-release bytes are charged again against
potentially lagging quota. No home path is created, queried, or scanned.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

from fixed25_storage import FILE_RESERVE, RESERVE, parse_work3

TAG = '20261021-brd4-jq1-25-fixed-storage-03'
ROOT = Path('/work3/s234818/mosaic/runs') / TAG
SCIENCE_ROOT = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-02')
WORK_RELEASE = Path('/work3/s234818/mosaic/releases') / TAG
WORK_UPLOAD = Path('/work3/s234818/mosaic') / (TAG + '.tar.gz')
PLACEMENT = {'protenix': 'work3', 'boltz2': 'work3', 'boltz2_chemistry': 'work3'}
VERSION = 'fixed25-work3-storage-adoption-v2'
INODE_POLICY = 'finite-user-quota-required; supplemental-statvfs-v2'


def output_parent(arm):
    if arm not in PLACEMENT or PLACEMENT[arm] != 'work3':
        raise ValueError('Unknown all-work3 storage-adoption arm')
    return ROOT / ('run-' + arm)


def tree_bytes(root):
    """Count regular files and internal-link metadata without traversing links.

    Pytest creates ``current`` aliases to sibling fixture directories. Their
    target bytes are counted through real paths only; each link itself adds
    its lstat size. Root/parent aliases and links outside this exact tree are
    rejected, as are dangling links and special file types.
    """
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError('Storage root/parent alias is outside its declared identity')
    if not root.exists():
        return 0
    mode = root.lstat().st_mode
    if stat.S_ISREG(mode):
        return root.lstat().st_size
    if not stat.S_ISDIR(mode):
        raise ValueError('Unknown/special storage root type')
    total = 0
    def walk_error(error):
        raise error
    for folder, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in dirs + files:
            path = Path(folder) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                try:
                    target = path.resolve(strict=True)
                except (OSError, RuntimeError) as error:
                    raise ValueError('Unresolved storage evidence link') from error
                if not target.is_relative_to(root):
                    raise ValueError('Storage evidence link escapes its exact tree')
                total += info.st_size
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError('Unknown/special storage evidence file type')
    return total


def stale_work_bytes():
    return sum(tree_bytes(path) for path in (SCIENCE_ROOT, ROOT, WORK_RELEASE, WORK_UPLOAD))


def check_cpu_storage(root, receipt_dir):
    """Separate 1 GiB CPU allowance plus the unchanged 4 GiB reserve."""
    if Path(root).resolve() != ROOT:
        raise ValueError('Wrong storage-adoption CPU scope')
    receipt_dir = Path(receipt_dir).resolve()
    if not receipt_dir.is_relative_to(ROOT):
        raise ValueError('Wrong storage-adoption CPU receipt scope')
    receipt_dir.mkdir(parents=True, exist_ok=False)
    record = {'version': VERSION, 'phase': 'cpu', 'status': 'failed',
              'utc': datetime.now(timezone.utc).isoformat(), 'inode_policy': INODE_POLICY,
              'filesystem_inode_observation': None, 'filesystem_inode_reporting': 'not_observed',
              'file_headroom_scope': 'Mandatory authenticated finite user-quota minimum remains required',
              'inode_limitation': 'No supplemental filesystem inode proof until a coherent positive tuple is validated'}
    try:
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True,
                             check=True, timeout=60).stdout
        record['raw_quota'] = raw
        record['filesystem_inode_observation'] = inode_observation(os.statvfs(ROOT))
        record['filesystem_inode_reporting'] = 'not_validated'
        record.update(gate_record('work3', raw, 1024**3, stale_work_bytes(),
                                  shutil.disk_usage(ROOT).free, record['filesystem_inode_observation']))
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        with (receipt_dir / 'split-storage-gate.json').open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
    if record['status'] != 'passed':
        raise ValueError('Storage-adoption CPU allowance plus reserve exceeds fresh work3 budget')
    return record


def validate_projection(projections):
    if set(projections) != set(PLACEMENT) or any(type(n) is not int or n <= 0 for n in projections.values()):
        raise ValueError('Every declared arm requires a positive integer authenticated projection')
    if set(PLACEMENT.values()) != {'work3'}:
        raise ValueError('This declared storage adoption supports work3 only')
    return dict(projections)


def budget_inputs(projections, current_outputs):
    projections = validate_projection(projections)
    if set(current_outputs) != set(PLACEMENT) or any(type(n) is not int or n < 0 for n in current_outputs.values()):
        raise ValueError('Missing/invalid current per-arm output bytes')
    if any(current_outputs[a] > projections[a] for a in PLACEMENT):
        raise ValueError('An arm already exceeded its authenticated full-output projection')
    return {'work3': {'arms': list(PLACEMENT),
                      'full_projected_bytes': sum(projections.values()),
                      'outstanding_projected_bytes': sum(projections[a] - current_outputs[a] for a in PLACEMENT)}}


def inode_observation(result):
    return {key: getattr(result, key) for key in ('f_files', 'f_ffree', 'f_favail')}


def inode_headroom(quota, observed):
    """Preserve finite user-quota headroom; distinguish unreported FS totals.

    A filesystem reporting the exact all-zero tuple supplies no supplemental
    inode evidence. It never replaces the original mandatory authenticated
    finite user-quota pools and their 100,000-file headroom requirement.
    """
    keys = {'f_files', 'f_ffree', 'f_favail'}
    if set(observed) != keys or any(type(n) is not int or n < 0 for n in observed.values()):
        raise ValueError('Invalid supplemental filesystem inode tuple')
    files, free, available = (observed[k] for k in ('f_files', 'f_ffree', 'f_favail'))
    if (files, free, available) == (0, 0, 0):
        return {'available_files_lower': quota['available_files_lower'],
                'filesystem_inode_reporting': 'unreported_all_zero',
                'file_headroom_scope': 'Mandatory minimum of all authenticated finite user-quota pools',
                'inode_limitation': 'All-zero statvfs tuple supplies no independent filesystem inode proof'}
    if files <= 0 or not 0 <= available <= free <= files:
        raise ValueError('Inconsistent supplemental filesystem inode tuple')
    return {'available_files_lower': min(available, quota['available_files_lower']),
            'filesystem_inode_reporting': 'reported_coherent',
            'file_headroom_scope': 'Minimum of all finite user-quota pools and reported statvfs f_favail',
            'inode_limitation': 'Quota and supplemental filesystem counts are observations, not reservations'}


def gate_record(pool, raw, projected, current_bytes, disk_free, filesystem_inodes):
    """Apply the original finite work3 rounding and reserve without alteration."""
    if pool != 'work3':
        raise ValueError('No non-work3 filesystem in this declared storage adoption')
    if any(type(n) is not int or n < 0 for n in (projected, current_bytes, disk_free)):
        raise ValueError('Invalid measured storage budget inputs')
    quota = parse_work3(raw)
    inode = inode_headroom(quota, filesystem_inodes)
    available_files = inode['available_files_lower']
    available = min(quota['available_bytes_lower'], disk_free) - current_bytes
    required = projected + RESERVE
    return {'status': 'passed' if available >= required and available_files >= FILE_RESERVE else 'failed',
            'filesystem': pool, 'quota': quota, 'raw_quota': raw,
            'projected_bytes': projected, 'reserve_bytes': RESERVE, 'required_bytes': required,
            'available_bytes_conservative': available, 'mosaic_bytes_subtracted': current_bytes,
            'disk_free_bytes': disk_free, 'available_files_lower': available_files,
            'file_reserve': FILE_RESERVE, 'inode_policy': INODE_POLICY,
            'filesystem_inode_observation': filesystem_inodes, **inode,
            'limitation': 'Fresh rounded quota and filesystem headroom are not reservations'}


def check_campaign_storage(projections, receipt_dir, *, phase):
    """Recheck the full outstanding projection for ALL THREE arms together."""
    projections = validate_projection(projections)
    receipt_dir = Path(receipt_dir).resolve()
    if not receipt_dir.is_relative_to(ROOT) or phase not in ('dispatch', 'entry'):
        raise ValueError('Wrong immutable storage-adoption control/phase scope')
    if ROOT.resolve() != ROOT:
        raise ValueError('Storage-adoption root cannot resolve through symbolic links')
    receipt_dir.mkdir(parents=True, exist_ok=False)
    ROOT.mkdir(parents=True, exist_ok=True)
    record = {'version': VERSION, 'utc': datetime.now(timezone.utc).isoformat(), 'phase': phase,
              'status': 'failed', 'placement': PLACEMENT, 'projections': projections,
              'current_outputs': {}, 'filesystems': {}, 'inode_policy': INODE_POLICY,
              'filesystem_inode_observation': None, 'filesystem_inode_reporting': 'not_observed',
              'file_headroom_scope': 'Mandatory authenticated finite user-quota minimum remains required',
              'inode_limitation': 'No supplemental filesystem inode proof until a coherent positive tuple is validated',
              'scope': 'NEW all-work3 outputs only; original failed quota and scientific source unchanged'}
    try:
        current = {a: tree_bytes(output_parent(a)) for a in PLACEMENT}
        record['current_outputs'] = current
        budgets = budget_inputs(projections, current)
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True,
                             check=True, timeout=60).stdout
        record['raw_quota'] = raw
        record['filesystem_inode_observation'] = inode_observation(os.statvfs(ROOT))
        record['filesystem_inode_reporting'] = 'not_validated'
        gate = gate_record('work3', raw, budgets['work3']['outstanding_projected_bytes'], stale_work_bytes(),
                           shutil.disk_usage(ROOT).free, record['filesystem_inode_observation'])
        gate.update(budgets['work3'], target=str(ROOT),
                    stale_quota_policy='Full original fixed02 + new control/output + new release + new upload archive bytes')
        record['filesystems']['work3'] = gate
        for name in ('filesystem_inode_reporting', 'file_headroom_scope', 'inode_limitation'):
            record[name] = gate[name]
        record['status'] = gate['status']
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        with (receipt_dir / 'split-storage-gate.json').open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
    if record['status'] != 'passed':
        raise ValueError('All-three whole-campaign byte/inode budget exceeds fresh work3 headroom')
    return record
