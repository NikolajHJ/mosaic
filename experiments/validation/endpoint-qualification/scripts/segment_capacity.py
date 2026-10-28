"""Metadata-only WORK3 rolling admission; no scientific evaluation or mutation.

This is a new resource policy, not a reinterpretation of any historical gate.
An authenticated caller supplies a complete no-follow inventory and the actual
scheduler/proof bindings.  This module validates that closed ledger and assesses
one next bounded segment plus every already admitted Mosaic writer.  Deferred
segments retain their original update targets and MUST re-admit before launch.

For a mutable scope, current bytes are subtracted from fresh quota availability
and only ``cap-current`` is reserved: their sum is the complete cap, once.  A
retained immutable tree can avoid a stale charge only with authenticated evidence
that those exact bytes were included in quota.  Terminal proof retires unspent
future allowance, never the files or their quota usage.  This does not reserve
capacity against unrelated account jobs; every entry and periodic checkpoint
must obtain another fresh observation.  All original scientific guards remain
independent and mandatory.
"""
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import hashlib
from pathlib import PurePosixPath
import re

GIB = 1024**3
RESERVE_BYTES = 4 * GIB
FILE_RESERVE = 100000
ACTIVE_FILE_RESERVE = 130000
VERSION = 'mosaic-rolling-segment-capacity-v1'
ROOT = '/work3/s234818/mosaic'
FUNDED = frozenset(('active', 'admitted', 'requested'))
STATES = FUNDED | frozenset(('retained', 'terminal', 'deferred'))
PHASES = frozenset(('preflight', 'dispatch', 'entry', 'periodic'))
ROW = re.compile(r'^\s*([^|]+)\|\s*(\d+)\s*\|\|\s*([^|]+)\|\s*([^|]+)\|\|\s*([^|]+)\|\s*([^|]+)\s*$')
SIZE = re.compile(r'^(\d+(?:\.\d+)?)\s+(Byte|KiB|MiB|GiB|TiB)$')
UNITS = {'Byte': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': GIB, 'TiB': 1024**4}


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ' must be a finite integer')
    return value


def fields(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), label + ' fields differ')


def digest(value, label):
    require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value), label + ' SHA256 missing')
    return value


def path(value, label):
    require(type(value) is str and (value == ROOT or value.startswith(ROOT + '/')), label + ' must be a dedicated Mosaic WORK3 path')
    p = PurePosixPath(value)
    require(str(p) == value and '..' not in p.parts and '.' not in p.parts, label + ' path is not canonical')
    return p


def proof(value, label):
    fields(value, ('path', 'bytes', 'sha256'), label)
    path(value['path'], label)
    integer(value['bytes'], label + ' bytes', 1)
    digest(value['sha256'], label)


def sparse_segment_cap(unit_bytes, updates, text_allowance_bytes=GIB):
    """Existing 2x pilot bound, stride20, three endpoint frames, one text cap.

    This forecasts just ONE bounded segment.  It does not change saving cadence,
    update count, objective or any numerical/scientific validation.
    """
    integer(unit_bytes, 'Pilot maximum matrix bytes', 1)
    integer(updates, 'Segment updates', 1)
    integer(text_allowance_bytes, 'Whole segment text/control allowance', 1)
    return 2 * unit_bytes * ((updates + 19) // 20 + 3) + text_allowance_bytes


def _size(token, label):
    match = SIZE.fullmatch(token.strip())
    require(match is not None, label + ' size/unit malformed')
    try:
        result = Decimal(match.group(1)) * UNITS[match.group(2)]
    except InvalidOperation as exc:
        raise ValueError(label + ' size malformed') from exc
    require(result.is_finite() and result >= 0, label + ' size must be finite')
    return result


def parse_work3_quota(raw, account):
    """All finite account pools, retaining used-up/hard-down whole-GiB rounding."""
    fields(account, ('name', 'id'), 'Quota account')
    require(account == {'name': 's234818', 'id': 306121}, 'Quota account differs')
    require(type(raw) is str and raw.endswith('\n') and raw.strip(), 'Complete newline-terminated raw quota required')
    pools, byte_limits, file_limits = [], [], []
    for line in raw.splitlines():
        if re.match(r'^\s*s234818\s*\|', line) is None:
            continue
        match = ROW.fullmatch(line)
        require(match is not None, 'Malformed account quota row')
        name, uid, used, hard, used_files, hard_files = (v.strip() for v in match.groups())
        require(name == account['name'] and int(uid) == account['id'], 'Quota owner identity differs')
        used_size = _size(used, 'Used quota')
        require(re.fullmatch(r'\d+', used_files), 'Used file quota malformed')
        row = {'used_reported': used, 'hard_reported': hard, 'used_files': int(used_files),
               'hard_files': None, 'used_bytes_upper': None, 'hard_bytes_lower': None}
        if hard != 'unlimited':
            hard_size = _size(hard, 'Hard quota')
            upper = int((used_size / GIB).to_integral_value(rounding=ROUND_CEILING)) * GIB
            lower = int((hard_size / GIB).to_integral_value(rounding=ROUND_FLOOR)) * GIB
            require(lower > 0, 'Finite quota hard bound must be positive')
            row.update(used_bytes_upper=upper, hard_bytes_lower=lower, remaining_bytes=lower - upper)
            byte_limits.append(lower - upper)
        if hard_files != 'unlimited':
            require(re.fullmatch(r'\d+', hard_files), 'Hard file quota malformed')
            row['hard_files'] = int(hard_files)
            require(row['hard_files'] > 0, 'Finite file quota must be positive')
            row['remaining_files'] = row['hard_files'] - row['used_files']
            file_limits.append(row['remaining_files'])
        pools.append(row)
    require(pools and byte_limits and file_limits, 'All finite authenticated byte/file quota evidence required')
    return {'pools': pools, 'available_bytes_lower': min(byte_limits),
            'available_files_lower': min(file_limits),
            'rounding': 'each finite used quota rounded up and hard quota down to whole GiB',
            'raw_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest()}


def _inodes(value, quota_files):
    fields(value, ('f_files', 'f_ffree', 'f_favail'), 'Filesystem inode observation')
    total, free, available = (integer(value[k], k) for k in ('f_files', 'f_ffree', 'f_favail'))
    if (total, free, available) == (0, 0, 0):
        return quota_files, 'unreported_all_zero', 'No independent filesystem inode evidence; finite account quota remains mandatory'
    require(total > 0 and 0 <= available <= free <= total, 'Mixed-zero or incoherent filesystem inode observation')
    return min(quota_files, available), 'reported_positive', 'Minimum finite account quota and coherent physical filesystem availability'


def validate_ledger(ledger):
    """Validate caller-authenticated metadata; never infer liveness from time."""
    fields(ledger, ('version', 'filesystem', 'phase', 'account', 'source_manifest_sha256',
                   'observation', 'scopes', 'goals', 'shared_account'), 'Admission ledger')
    require(ledger['version'] == VERSION and ledger['filesystem'] == 'work3', 'Admission policy/filesystem differs')
    require(ledger['phase'] in PHASES, 'Admission phase differs')
    digest(ledger['source_manifest_sha256'], 'Sealed source manifest')
    observation = ledger['observation']
    fields(observation, ('complete', 'nofollow', 'untracked_mosaic_writers', 'inventory_proof', 'scheduler_proof'), 'Observation')
    require(observation['complete'] is True and observation['nofollow'] is True,
            'Completed no-follow scope inventory required')
    require(observation['untracked_mosaic_writers'] == [], 'Unknown active Mosaic writes block admission')
    proof(observation['inventory_proof'], 'Complete inventory proof')
    proof(observation['scheduler_proof'], 'Actual scheduler observation proof')
    shared = ledger['shared_account']
    fields(shared, ('non_mosaic_job_ids', 'their_future_bytes_reserved', 'whole_account_reservation',
                    'entry_and_periodic_recheck_required'), 'Shared account scope')
    require(type(shared['non_mosaic_job_ids']) is list and
            len(shared['non_mosaic_job_ids']) == len(set(shared['non_mosaic_job_ids'])) and
            all(type(v) is str and re.fullmatch(r'[1-9]\d*', v) for v in shared['non_mosaic_job_ids']),
            'Explicit observed non-Mosaic job identities required')
    require(shared['their_future_bytes_reserved'] is False and shared['whole_account_reservation'] is False and
            shared['entry_and_periodic_recheck_required'] is True,
            'Mosaic admission cannot claim unrelated-account future reservation')
    require(type(ledger['scopes']) is list and ledger['scopes'], 'Explicit scope ledger required')
    scope_ids, selections, rows = set(), [], []
    active_defog = False
    for scope in ledger['scopes']:
        fields(scope, ('id', 'root', 'excluded_roots', 'state', 'role', 'actual_bytes', 'actual_files', 'cap_bytes',
                       'cap_files', 'inventory_proof', 'liveness_proof', 'quota_inclusion_proof'), 'Scope')
        sid, state, role = scope['id'], scope['state'], scope['role']
        require(type(sid) is str and re.fullmatch(r'[a-z0-9][a-z0-9_-]*', sid) and sid not in scope_ids,
                'Duplicate or malformed scope identity')
        scope_ids.add(sid)
        root = path(scope['root'], 'Scope root')
        require(root != PurePosixPath(ROOT) or (sid == 'retained-mosaic' and state == 'retained' and role == 'history'),
                'Whole Mosaic root is permitted only as retained-mosaic history')
        require(type(scope['excluded_roots']) is list, 'Explicit scope inventory exclusions required')
        excluded = [path(v, 'Excluded inventory root') for v in scope['excluded_roots']]
        require(len(excluded) == len(set(excluded)) and all(root in p.parents for p in excluded),
                'Inventory exclusions must be unique strict descendants')
        require(all(a not in b.parents for a in excluded for b in excluded if a != b),
                'Nested inventory exclusions are ambiguous')
        selections.append((root, excluded))
        require(state in STATES and role in ('segment', 'validation', 'control', 'release', 'cache', 'history', 'defog'),
                'Scope state/role malformed')
        actual, files = integer(scope['actual_bytes'], 'Actual entire-tree bytes'), integer(scope['actual_files'], 'Actual files')
        proof(scope['inventory_proof'], 'Scope no-follow inventory')
        proof(scope['liveness_proof'], 'Actual scope state proof')
        included = scope['quota_inclusion_proof'] is not None
        if included:
            proof(scope['quota_inclusion_proof'], 'Exact unchanged-tree quota inclusion proof')
        if state in FUNDED:
            cap, cap_files = integer(scope['cap_bytes'], 'Full bounded scope cap', 1), integer(scope['cap_files'], 'Full bounded scope file cap', 1)
            require(actual <= cap and files <= cap_files, 'Mutable scope exceeds its full bounded cap')
            require(not included, 'Mutable bytes cannot be credited against an unknown quota snapshot')
            stale, stale_files = actual, files
            future, future_files = cap - actual, cap_files - files
            active_defog |= role == 'defog'
        else:
            require(scope['cap_bytes'] is None and scope['cap_files'] is None,
                    'Terminal/retained/deferred scope cannot retain an unspent execution cap')
            require(state != 'deferred' or (actual == 0 and files == 0 and not included),
                    'Deferred execution must be absent and require later admission')
            stale, stale_files = (0, 0) if included else (actual, files)
            future = future_files = 0
        rows.append({'id': sid, 'root': str(root), 'excluded_roots': [str(p) for p in excluded],
                     'state': state, 'role': role,
                     'actual_bytes': actual, 'actual_files': files,
                     'entire_mutable_or_unproven_tree_stale_charge_bytes': stale,
                     'entire_mutable_or_unproven_tree_stale_charge_files': stale_files,
                     'remaining_admitted_cap_bytes': future, 'remaining_admitted_cap_files': future_files,
                     'quota_inclusion_proved': included})
    declared_roots = [r for r, _ in selections]
    require(len(declared_roots) == len(set(declared_roots)), 'Duplicate inventory roots would double count')
    for outer, excluded in selections:
        require(all(p in declared_roots for p in excluded), 'Every excluded subtree needs its own ledger scope')
        for inner in declared_roots:
            if outer in inner.parents:
                require(any(p == inner or p in inner.parents for p in excluded),
                        'Overlapping/nested accounting roots would double count')
    goals = ledger['goals']
    require(type(goals) is list and goals, 'Original total update targets must remain explicit')
    goal_ids, next_scopes = set(), set()
    states = {s['id']: s['state'] for s in ledger['scopes']}
    for goal in goals:
        fields(goal, ('id', 'saved_updates', 'target_updates', 'next_updates', 'next_scope_id',
                      'later_updates_deferred', 'readmission_before_later_execution'), 'Goal')
        require(type(goal['id']) is str and goal['id'] and goal['id'] not in goal_ids, 'Goal identity differs')
        goal_ids.add(goal['id'])
        saved = integer(goal['saved_updates'], 'Saved contiguous updates')
        target = integer(goal['target_updates'], 'Original target updates', 1)
        next_updates = integer(goal['next_updates'], 'Next bounded segment updates')
        later = integer(goal['later_updates_deferred'], 'Deferred later updates')
        require(saved + next_updates + later == target and goal['readmission_before_later_execution'] is True,
                'Original target or mandatory later readmission changed')
        sid = goal['next_scope_id']
        if next_updates:
            require(sid in states and states[sid] in FUNDED and sid not in next_scopes,
                    'Each next segment needs one uniquely funded scope')
            next_scopes.add(sid)
        else:
            require(sid is None, 'No next execution can be silently admitted')
    return rows, ACTIVE_FILE_RESERVE if active_defog else FILE_RESERVE


def assess(ledger, raw_quota, filesystem_free_bytes, filesystem_inodes):
    """Return a reviewable passed/failed receipt; malformed/incomplete input raises.

    Caller must authenticate every supplied proof, obtain the quota *after* the
    complete inventory, atomically publish this observation, and fail its launch
    if ``status != passed``.  Capacity failure returns a receipt to preserve.
    """
    rows, file_reserve = validate_ledger(ledger)
    quota = parse_work3_quota(raw_quota, ledger['account'])
    disk = integer(filesystem_free_bytes, 'Physical filesystem available bytes')
    available_files, reporting, inode_scope = _inodes(filesystem_inodes, quota['available_files_lower'])
    stale = sum(r['entire_mutable_or_unproven_tree_stale_charge_bytes'] for r in rows)
    stale_files = sum(r['entire_mutable_or_unproven_tree_stale_charge_files'] for r in rows)
    future = sum(r['remaining_admitted_cap_bytes'] for r in rows)
    future_files = sum(r['remaining_admitted_cap_files'] for r in rows)
    available = min(quota['available_bytes_lower'], disk) - stale
    required, required_files = RESERVE_BYTES + future, file_reserve + future_files
    remaining_files = available_files - stale_files
    passed = available >= required and remaining_files >= required_files
    return {'version': VERSION, 'status': 'passed' if passed else 'failed', 'phase': ledger['phase'],
            'source_manifest_sha256': ledger['source_manifest_sha256'], 'ledger': ledger,
            'quota': quota, 'filesystem_free_bytes': disk, 'filesystem_inode_observation': filesystem_inodes,
            'filesystem_inode_reporting': reporting, 'file_headroom_scope': inode_scope,
            'scope_accounting': rows, 'entire_tree_stale_charge_bytes': stale,
            'entire_tree_stale_charge_files': stale_files, 'future_admitted_bytes': future,
            'future_admitted_files': future_files, 'available_bytes': available,
            'required_bytes': required, 'byte_margin': available - required,
            'available_files': remaining_files, 'required_files': required_files,
            'file_margin': remaining_files - required_files, 'reserve_bytes': RESERVE_BYTES,
            'file_reserve': file_reserve, 'historical_gate_results_changed': False,
            'scientific_settings_or_guards_changed': False, 'saved_results_changed': False,
            'lifetime_targets_preserved': True, 'later_segments_admitted': False,
            'scope': 'All explicitly inventoried Mosaic active/admitted/requested bounded scopes; deferred targets must re-admit',
            'limitation': 'Fresh scoped admission observation, not an account reservation; unrelated jobs can consume capacity; entry and periodic rechecks mandatory'}
