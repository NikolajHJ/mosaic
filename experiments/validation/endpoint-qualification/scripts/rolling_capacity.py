"""Live metadata adapter for explicit rolling WORK3 admission.

Only the caller's NEW immutable release may use this policy.  It never imports
the historical nested capacity chain and never mutates an old gate, study, job,
source or saved scientific state.  The sealed caller authenticates its source
and plan before invoking ``storage(op, parent, phase)``.  Dispatch calls must
hold its existing dispatch.lock.  This adapter writes only its own exclusive
capacity receipts and, after a passed dispatch gate, one immutable scope claim.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

SPEC = importlib.util.spec_from_file_location('_rolling_segment_capacity', Path(__file__).with_name('segment_capacity.py'))
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
VERSION = 'mosaic-rolling-work3-adapter-v1'
CONTROL_CAP = 2 * c.GIB
CONTROL_FILES = 30000
RELEASE_CAP = 64 * 1024**2
RELEASE_FILES = 3000
SEGMENT_FILES_EXTRA = 3000
FILE_CAPACITY_POLICY = 'three-stride20-streams-two-validated-boundary-cohorts-v1'
VALIDATION_INVENTORY_POLICY = 'portable-relative-path-type-length-content-sha256-v1'
LIVE_STATES = frozenset(('PEND', 'RUN', 'SSUSP', 'USUSP', 'PSUSP', 'WAIT'))
TERMINAL_STATES = frozenset(('DONE', 'EXIT'))


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def binding(path):
    p = Path(path)
    info = p.lstat()
    c.require(p.is_absolute() and stat.S_ISREG(info.st_mode) and not p.is_symlink()
              and all(not parent.is_symlink() for parent in p.parents), 'Metadata proof linked/special')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return {'path': str(p), 'bytes': info.st_size, 'sha256': h.hexdigest()}


def check_binding(row):
    c.proof(row, 'Exact metadata binding')
    c.require(binding(row['path']) == row, 'Metadata proof changed: ' + row['path'])
    return row


def allocation_jobs(row):
    """Exact jobs sharing ONE explicitly bounded output/control tree.

    Some immutable launchers derive their output root from the sealed source,
    rather than spelling it in argv.  Their native validators may share one
    whole control cap.  Listing exact additional jobs authenticates each job
    without duplicating that cap or inventing another output tree.
    """
    result = [{'job_id': row['job_id'], 'command': row['command']}]
    if 'additional_jobs' in row:
        c.require(type(row['additional_jobs']) is list, 'Additional exact allocation jobs must be a list')
        for job in row['additional_jobs']:
            c.fields(job, ('job_id', 'command'), 'Additional allocation job')
            result.append(job)
    return result


def policy(plan):
    value = plan.get('rolling_storage_policy')
    c.fields(value, ('version', 'control_cap_bytes', 'control_cap_files', 'release_cap_bytes',
                     'release_cap_files', 'known_mosaic_jobs', 'deferred_goals'), 'Rolling storage policy')
    c.require(value['version'] == VERSION and value['control_cap_bytes'] == CONTROL_CAP
              and value['control_cap_files'] == CONTROL_FILES and value['release_cap_bytes'] == RELEASE_CAP
              and value['release_cap_files'] == RELEASE_FILES, 'Declared bounded adapter allowances differ')
    c.require(type(value['known_mosaic_jobs']) is list, 'Explicit independent Mosaic allocations required')
    ids, roots = set(), set()
    for row in value['known_mosaic_jobs']:
        c.require(type(row) is dict, 'Independent allocation must be a closed object')
        expected = ('job_id', 'command', 'root', 'cap_bytes', 'cap_files', 'role',
                    'release_root', 'release_cap_bytes', 'release_cap_files', 'source_manifest')
        c.fields(row, expected + (('additional_jobs',) if 'additional_jobs' in row else ()), 'Independent allocation')
        c.proof(row['source_manifest'], 'Independent source manifest')
        c.path(row['root'], 'Independent output root')
        for job in allocation_jobs(row):
            c.require(type(job['job_id']) is str and re.fullmatch(r'[1-9]\d*', job['job_id'])
                      and job['job_id'] not in ids, 'Duplicate/invalid independent job')
            ids.add(job['job_id'])
            c.require(type(job['command']) is str and job['command']
                      and (row['root'] in job['command']
                           or str(Path(row['source_manifest']['path']).parent) in job['command']),
                      'Exact allocation command must identify its output root or immutable source')
        c.require(row['root'] not in roots, 'Independent output root repeated')
        roots.add(row['root'])
        c.integer(row['cap_bytes'], 'Independent entire output cap', 1)
        c.integer(row['cap_files'], 'Independent entire file cap', 1)
        c.require(row['role'] in ('defog', 'cache', 'validation'), 'Independent allocation role differs')
        if row['release_root'] is None:
            c.require(row['release_cap_bytes'] is None and row['release_cap_files'] is None,
                      'Absent independent release cannot have an allowance')
        else:
            c.path(row['release_root'], 'Independent release root')
            c.require(row['release_cap_bytes'] == RELEASE_CAP and row['release_cap_files'] == RELEASE_FILES,
                      'Independent sealed release allowance differs')
        if row['role'] == 'defog':
            c.require(row['job_id'] == '29839884' and row['cap_bytes'] == 4 * c.GIB
                      and row['cap_files'] >= 30000 and row['release_root'] is not None
                      and len(allocation_jobs(row)) == 1
                      and row['source_manifest']['sha256'] == 'a15c8bb2a8ac7e9e12a65c54079e14c994e53b119b9a59e1a9788b5044d8686f',
                      'Original independent DeFoG allocation/cap/source pin differs')
    c.require(type(value['deferred_goals']) is list, 'Explicit other authorized deferred targets required')
    return value


def parse_scheduler(raw):
    value = json.loads(raw)
    c.require(type(value) is dict and type(value.get('RECORDS')) is list,
              'Complete actual account scheduler JSON required')
    rows, ids = [], set()
    for row in value['RECORDS']:
        c.require(type(row) is dict and all(type(row.get(k)) is str and row[k]
                  for k in ('JOBID', 'STAT', 'QUEUE', 'COMMAND')), 'Scheduler row malformed')
        c.require(re.fullmatch(r'[1-9]\d*', row['JOBID']) and row['JOBID'] not in ids
                  and row['STAT'] in LIVE_STATES | TERMINAL_STATES, 'Unknown/duplicate account allocation state')
        ids.add(row['JOBID'])
        rows.append({k: row[k] for k in ('JOBID', 'STAT', 'QUEUE', 'COMMAND')})
    if 'JOBS' in value:
        c.require(str(value['JOBS']) == str(len(rows)), 'Scheduler complete row count differs')
    return rows


def classify_jobs(rows, known_jobs, own_jobs, source):
    """Unknown Mosaic queued/running writes and unbounded external RUN fail."""
    known = {job['job_id']: job for row in known_jobs for job in allocation_jobs(row)}
    observed, external, unknown = {}, [], []
    for row in rows:
        if row['STAT'] in TERMINAL_STATES:
            continue
        jid, command = row['JOBID'], row['COMMAND']
        if jid in known:
            c.require(command == known[jid]['command'], 'Independent actual job command changed')
            observed[jid] = row
        elif str(source) in command and jid in own_jobs:
            observed[jid] = row
        elif c.ROOT in command:
            unknown.append(jid)
        else:
            external.append(jid)
            c.require(row['STAT'] in ('PEND', 'PSUSP', 'WAIT'),
                      'Unrelated active account writer has no authenticated bounded declaration: ' + jid)
    c.require(not unknown, 'Unknown nonterminal Mosaic allocation(s): ' + ','.join(unknown))
    return observed, external


def _scope(sid, root, state, role, liveness, *, cap=None, files=None):
    return {'id': sid, 'root': str(root), 'excluded_roots': [], 'state': state, 'role': role,
            'actual_bytes': 0, 'actual_files': 0, 'cap_bytes': cap, 'cap_files': files,
            'inventory_proof': None, 'liveness_proof': liveness, 'quota_inclusion_proof': None}


def segment_file_capacity(updates, validation_files):
    """Three retained streams plus fixed/empirical boundary evidence headroom.

    Retention saves distinct snapshot, trial and preparation-reference files.
    The completed new native validation is inventoried in full, including its
    raw FD/precision/pilot evidence. Twice that observed cohort plus100 metadata
    files is retained in addition to at least3000 fixed files. This is a
    conservative empirical admission forecast, not a proof of arbitrary future
    filesystem growth; the complete scope cap remains mandatory throughout.
    """
    c.integer(updates, 'Whole segment updates', 1)
    c.integer(validation_files, 'Completed actual validation files', 1)
    fixed = max(SEGMENT_FILES_EXTRA, 2 * validation_files + 100)
    return {'policy': FILE_CAPACITY_POLICY, 'retained_streams': 3,
            'retained_update_bundles_upper': (updates + 19) // 20,
            'completed_validation_files': validation_files,
            'fixed_boundary_and_metadata_files': fixed,
            'cap_files': 3 * ((updates + 19) // 20) + fixed}


def _local_file_signature(info, *, windows=None):
    """One-read race check; Windows stat/fstat have different ctime meanings.

    Keep the POSIX change-time check on HPC. Windows still compares device,
    inode, type, length and modification time; semantic identity uses SHA256.
    """
    windows = os.name == 'nt' if windows is None else windows
    c.require(type(windows) is bool, 'Read-signature platform flag must be literal')
    signature = (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)
    return signature if windows else (*signature, info.st_ctime_ns)


def _stable_file_row(path, relative, deadline):
    c.require(all(not p.is_symlink() for p in path.parents), 'Completed validation file has linked parent')
    before = path.lstat()
    c.require(stat.S_ISREG(before.st_mode), 'Completed validation file linked/special')
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        c.require(stat.S_ISREG(opened.st_mode)
                  and _local_file_signature(before) == _local_file_signature(opened),
                  'Completed validation file changed before open')
        digest, length = hashlib.sha256(), 0
        for block in iter(lambda: stream.read(1024**2), b''):
            c.require(time.monotonic() < deadline, 'Complete validation content inventory exceeded bounded deadline')
            digest.update(block)
            length += len(block)
        after = os.fstat(stream.fileno())
    literal = path.lstat()
    c.require(length == before.st_size
              and _local_file_signature(before) == _local_file_signature(after) == _local_file_signature(literal),
              'Completed validation file changed while hashing')
    return {'path': relative, 'type': 'regular_file', 'bytes': length, 'sha256': digest.hexdigest()}


def stable_validation_inventory(root, identity, *, timeout=600):
    """Portable immutable basis, independent of node-local physical metadata.

    No device, inode, allocated-block, timestamp or atime field is persisted.
    Those fields are appropriate to a fresh physical-capacity observation, not
    equality of saved scientific artifacts across allocated HPC nodes.
    """
    root = Path(root)
    c.require(root.is_dir() and not root.is_symlink()
              and all(not p.is_symlink() for p in root.parents),
              'Completed validation study tree missing/linked')
    deadline, rows, pending, directories = time.monotonic() + timeout, [], [root], []
    while pending:
        c.require(time.monotonic() < deadline, 'Complete validation content inventory exceeded bounded deadline')
        directory = pending.pop()
        before = directory.lstat()
        c.require(stat.S_ISDIR(before.st_mode) and all(not p.is_symlink() for p in directory.parents),
                  'Completed validation directory changed to linked/special')
        directories.append((directory, _local_file_signature(before)))
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    rows.append({'path': relative, 'type': 'directory'})
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    rows.append(_stable_file_row(path, relative, deadline))
                else:
                    raise ValueError('Completed validation inventory linked/special: ' + str(path))
    for directory, original in directories:
        c.require(_local_file_signature(directory.lstat()) == original,
                  'Completed validation directory membership changed during inventory')
    rows.sort(key=lambda row: row['path'])
    files = [row for row in rows if row['type'] == 'regular_file']
    c.require(files, 'Completed full validation inventory is empty/incomplete')
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return {'id': identity, 'root': str(root), 'identity_policy': VALIDATION_INVENTORY_POLICY,
            'complete': True, 'nofollow': True, 'actual_files': len(files),
            'actual_bytes': sum(row['bytes'] for row in files), 'entries': rows,
            'semantic_rows_sha256': hashlib.sha256(encoded).hexdigest(),
            'identity_scope': 'Sorted relative path/type/logical length/content SHA256; no node-local physical metadata'}


def validation_inventory(op, arm, offset):
    """Actual passed validation tree, including every raw proof and reference."""
    root = op.parent_path('validate', arm, offset) / 'payload/study'
    return stable_validation_inventory(root, arm + '-validation-basis-' + str(offset))


def segment_certificate(op, plan, pin, arm, offset):
    path = op.ROOT / (op.label('segment', arm, offset) + '-certificate.json')
    if not path.exists():
        return None
    value = op.read(path)
    steps = op.ARMS[arm]['segment_steps']
    c.require(value.get('status') == 'passed' and value.get('arm') == arm and value.get('offset') == offset
              and value.get('global_end') == offset + steps and value.get('source_manifest_sha256') == pin
              and value.get('plan_sha256') == binding(op.SOURCE / 'plan.json')['sha256'],
              'Completed segment certificate/source/boundary differs')
    c.require(type(value.get('proofs')) is list and value['proofs'], 'Completed segment proof chain absent')
    for row in value['proofs']:
        op.check_generated_proof(row)
    return binding(path)


def candidate_from_parent(op, parent, phase):
    if phase != 'dispatch':
        return None
    for arm in op.DISPATCH_ARMS:
        offsets = op.ARMS[arm]['offsets']
        for offset in offsets:
            if parent == op.parent_path('after-validate', arm, offset):
                return arm, offset
    return None


def build_scopes(op, plan, pin, phase, parent, scheduler_proof, observed):
    """Build disjoint metadata selections and immutable next-segment claims."""
    p = policy(plan)
    scopes = [_scope('retained-mosaic', c.ROOT, 'retained', 'history', scheduler_proof),
              _scope('own-control', op.ROOT, 'admitted', 'control', binding(op.SOURCE / 'plan.json'),
                     cap=CONTROL_CAP, files=CONTROL_FILES),
              _scope('own-release', op.SOURCE.parent, 'admitted', 'release', binding(op.SOURCE / 'source_manifest.json'),
                     cap=RELEASE_CAP, files=RELEASE_FILES)]
    retained, control = scopes[0], scopes[1]
    retained['excluded_roots'] += [str(op.ROOT), str(op.SOURCE.parent)]
    for row in p['known_mosaic_jobs']:
        check_binding(row['source_manifest'])
        state = 'active' if any(job['job_id'] in observed and observed[job['job_id']]['STAT'] == 'RUN'
                                for job in allocation_jobs(row)) else 'admitted'
        sid = 'independent-' + row['job_id']
        scopes.append(_scope(sid, row['root'], state, row['role'], scheduler_proof,
                             cap=row['cap_bytes'], files=row['cap_files']))
        retained['excluded_roots'].append(row['root'])
        if row['release_root'] is not None:
            scopes.append(_scope(sid + '-release', row['release_root'], 'admitted', 'release', row['source_manifest'],
                                 cap=row['release_cap_bytes'], files=row['release_cap_files']))
            retained['excluded_roots'].append(row['release_root'])
    candidate = candidate_from_parent(op, Path(parent), phase)
    new_claim, goals = None, []
    for arm in op.DISPATCH_ARMS:
        declared = op.ARMS[arm]
        done, live = [], []
        for offset in declared['offsets']:
            sid = arm + '-segment-' + str(offset)
            root = op.parent_path('segment', arm, offset) / 'payload/study'
            control['excluded_roots'].append(str(root))
            certificate = segment_certificate(op, plan, pin, arm, offset)
            claim_path = op.ROOT / 'storage-claims' / (sid + '.json')
            claim = read(claim_path) if claim_path.exists() else None
            if certificate is not None:
                submission = op.read(op.parent_path('segment', arm, offset) / 'submission.json')
                c.require(submission.get('job_id') not in observed,
                          'A nonterminal actual segment cannot retire its future allowance')
                done.append(offset)
                scopes.append(_scope(sid, root, 'terminal', 'segment', certificate))
                continue
            expected = {'version': VERSION, 'source_manifest_sha256': pin, 'plan_sha256': binding(op.SOURCE / 'plan.json')['sha256'],
                        'arm': arm, 'offset': offset, 'updates': declared['segment_steps'], 'root': str(root)}
            if claim is not None:
                c.require(all(claim.get(k) == v for k, v in expected.items()), 'Existing immutable segment claim changed')
                check_binding(claim['readiness_certificate'])
                check_binding(claim['admission_receipt'])
                admitted = read(claim['admission_receipt']['path'])
                c.require(claim.get('admission_policy') == c.VERSION
                          and claim.get('original_total_goal_preserved') is True
                          and admitted.get('status') == 'passed'
                          and admitted.get('source_manifest_sha256') == pin
                          and admitted.get('adapter_version') == VERSION,
                          'Immutable segment claim lacks its actual passed scoped admission')
                unit = c.integer(claim.get('unit_bytes'), 'Authenticated admitted pilot maximum', 1)
                op.verify_ready(plan, pin, arm, offset)
                ready_result = op.verify_result(plan, pin, 'validate', arm, offset)
                c.require(unit >= ready_result['projections']['maximum_pilot_update_bytes'],
                          'Segment claim is smaller than its actual authenticated pilot')
                cap = c.sparse_segment_cap(unit, declared['segment_steps'])
                check_binding(claim['validation_inventory_evidence'])
                basis = read(claim['validation_inventory_evidence']['path'])
                c.require(basis == claim.get('validation_inventory') == validation_inventory(op, arm, offset),
                          'Authenticated completed validation file-cap basis changed')
                file_capacity = segment_file_capacity(declared['segment_steps'], basis['actual_files'])
                c.require(claim.get('file_capacity') == file_capacity
                          and claim.get('cap_bytes') == cap and claim.get('cap_files') == file_capacity['cap_files'],
                          'Existing entire segment cap changed')
                selected_basis = admitted.get('proposed_segment_claim_basis', {})
                c.require(all(selected_basis.get(k) == claim[k] for k in (
                    'file_capacity', 'validation_inventory', 'validation_inventory_evidence', 'readiness_certificate')),
                    'Segment file capacity differs from its immutable passed admission basis')
                budget = [s for s in admitted['ledger']['scopes'] if s['id'] == sid]
                c.require(len(budget) == 1 and budget[0]['root'] == str(root)
                          and budget[0]['cap_bytes'] == cap and budget[0]['cap_files'] == claim['cap_files']
                          and budget[0]['state'] == 'requested', 'Claim differs from uniquely admitted bounded output scope')
                liveness, state = binding(claim_path), 'admitted'
            elif candidate == (arm, offset):
                # This is NEW allocated readiness, not a historical pilot guess.
                op.verify_ready(plan, pin, arm, offset)
                ready_path = op.ROOT / (op.label('validate', arm, offset) + '-certificate.json')
                projection = read(op.ROOT / 'projections.json')[arm]
                c.require(projection.get('remaining_offsets') == declared['offsets'][declared['offsets'].index(offset):],
                          'Actual new pilot remaining offsets differ')
                unit = c.integer(projection.get('unit_bytes'), 'Actual new pilot maximum', 1)
                cap = c.sparse_segment_cap(unit, declared['segment_steps'])
                basis = validation_inventory(op, arm, offset)
                file_capacity = segment_file_capacity(declared['segment_steps'], basis['actual_files'])
                liveness, state = binding(ready_path), 'requested'
                new_claim = dict(expected, unit_bytes=unit, cap_bytes=cap,
                                 cap_files=file_capacity['cap_files'], file_capacity=file_capacity,
                                 validation_inventory=basis, readiness_certificate=liveness)
            else:
                scopes.append(_scope(sid, root, 'deferred', 'segment', binding(op.SOURCE / 'plan.json')))
                continue
            live.append(offset)
            scopes.append(_scope(sid, root, state, 'segment', liveness, cap=cap,
                                 files=file_capacity['cap_files']))
        c.require(done == declared['offsets'][:len(done)] and len(live) <= 1,
                  'Completed prefix or independently admitted per-arm segment count differs')
        if live:
            c.require(live == declared['offsets'][len(done):len(done) + 1], 'Out-of-order segment claim')
        saved = declared['offsets'][0] + len(done) * declared['segment_steps']
        next_updates = declared['segment_steps'] if live else 0
        goals.append({'id': arm, 'saved_updates': saved, 'target_updates': declared['global_target'],
                      'next_updates': next_updates,
                      'next_scope_id': arm + '-segment-' + str(live[0]) if live else None,
                      'later_updates_deferred': declared['global_target'] - saved - next_updates,
                      'readmission_before_later_execution': True})
    goals += p['deferred_goals']
    return scopes, goals, new_claim


def inventory_selections(scopes, *, timeout=600):
    """Full no-follow metadata inventory, every selected path exactly once.

    Hard-linked aliases are charged conservatively by path, never traversed.
    Retained symlinks count their metadata once and are never followed.  All
    mutable/output/control/release symlinks and every special type fail closed.
    Missing declared fresh roots are zero; a linked/missing ancestor fails.
    """
    deadline, result = time.monotonic() + timeout, []
    for scope in scopes:
        root = Path(scope['root'])
        excluded = {Path(p) for p in scope['excluded_roots']}
        digest = hashlib.sha256()
        total, files, links, logical = 0, 0, 0, 0
        c.require(all(not p.is_symlink() for p in root.parents), 'Inventory root has a linked parent')
        c.require(not root.is_symlink(), 'Inventory root linked')
        if root.exists():
            c.require(root.is_dir(), 'Inventory root is not a directory')
            pending = [root]
            while pending:
                c.require(time.monotonic() < deadline, 'Complete metadata inventory exceeded bounded deadline')
                directory = pending.pop()
                with os.scandir(directory) as entries:
                    for entry in sorted(entries, key=lambda item: item.name):
                        p = Path(entry.path)
                        if p in excluded:
                            continue
                        info = entry.stat(follow_symlinks=False)
                        kind = stat.S_IFMT(info.st_mode)
                        record = [str(p), kind, info.st_size, getattr(info, 'st_blocks', 0),
                                  info.st_dev, info.st_ino, info.st_mtime_ns]
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(p)
                        elif stat.S_ISREG(info.st_mode):
                            files += 1
                            logical += info.st_size
                            total += max(info.st_size, getattr(info, 'st_blocks', 0) * 512)
                        elif stat.S_ISLNK(info.st_mode) and scope['state'] == 'retained':
                            record.append(os.readlink(p))
                            links += 1
                            files += 1
                            logical += info.st_size
                            total += max(info.st_size, getattr(info, 'st_blocks', 0) * 512)
                        else:
                            raise ValueError('Unknown linked/special active inventory path: ' + str(p))
                        digest.update(json.dumps(record, separators=(',', ':'), ensure_ascii=True).encode() + b'\n')
        result.append({'id': scope['id'], 'root': str(root), 'excluded_roots': sorted(str(p) for p in excluded),
                       'actual_bytes': total, 'actual_files': files, 'logical_bytes': logical,
                       'retained_symlinks_not_traversed': links, 'complete': True, 'nofollow': True,
                       'metadata_rows_sha256': digest.hexdigest(),
                       'byte_scope': 'Per selected path maximum(logical bytes, allocated blocks); hard-link aliases conservatively charged separately'})
    return result


def _scope_contains(scope, path):
    root = Path(scope['root'])
    return (path == root or root in path.parents) and not any(
        path == Path(excluded) or Path(excluded) in path.parents for excluded in scope['excluded_roots'])


def owned_atomic_temporary(op, scopes, value):
    """Known live writer rename patterns only; immutable/missing inputs fail.

    This permission allows a COMPLETE new inventory attempt, never omission of
    the vanished entry. It does not permit retrying missing stable artifacts,
    unknown suffixes, arbitrary retained files, releases or invalid symlinks.
    """
    if type(value) is not str:
        return False
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or str(path) != value:
        return False
    owners = [scope for scope in scopes if _scope_contains(scope, path)]
    if len(owners) != 1:
        return False
    owner = owners[0]
    if owner['state'] not in c.FUNDED or owner['role'] in ('release', 'history'):
        return False
    if any(parent.is_symlink() for parent in path.parents):
        return False
    atomic_io = path.name.endswith('.json.tmp') or path.name.endswith('.npz.tmp')
    if owner['role'] == 'segment':
        case = Path(owner['root']) / 'case-0'
        return atomic_io or path == case / 'best-observed.partial'
    if owner['id'] == 'own-control':
        if path in (op.ROOT / 'jobs.json.partial', op.ROOT / 'projections.json.partial'):
            return True
        return atomic_io and any((op.parent_path('validate', arm, offset) / 'payload/study') in path.parents
                                 for arm in op.DISPATCH_ARMS for offset in op.ARMS[arm]['offsets'])
    if owner['role'] == 'defog':
        return owner['id'] == 'independent-29839884' and path.name.endswith('.json.tmp') \
            and (Path(owner['root']) / 'pilot') in path.parents
    if owner['role'] == 'validation':
        relative = path.relative_to(Path(owner['root'])).parts
        return atomic_io and any(relative[index:index + 2] == ('payload', 'study')
                                 for index in range(len(relative) - 1))
    return False


def complete_capacity_inventory(op, scopes, attempt, *, timeout=600, maximum_attempts=3):
    """Bounded full restarts for proven live atomic rename races only."""
    c.require(type(maximum_attempts) is int and 1 <= maximum_attempts <= 3,
              'Capacity inventory retry count exceeds declared bound')
    deadline = time.monotonic() + timeout
    for number in range(1, maximum_attempts + 1):
        remaining = deadline - time.monotonic()
        c.require(remaining > 0, 'Complete capacity inventory retry deadline exhausted')
        try:
            return inventory_selections(scopes, timeout=remaining)
        except FileNotFoundError as error:
            allowed = owned_atomic_temporary(op, scopes, error.filename)
            record = {'status': 'failed_incomplete_inventory_not_admission', 'attempt': number,
                      'error': repr(error), 'path': error.filename,
                      'recognized_owned_atomic_temporary': allowed,
                      'whole_scope_restart_required': True, 'partial_totals_used': False,
                      'files_skipped': False, 'retry_will_occur': allowed and number < maximum_attempts}
            op.write(Path(attempt) / ('incomplete-capacity-inventory-' + str(number).zfill(2) + '.json'), record)
            if not allowed or number == maximum_attempts:
                raise
    raise AssertionError('Unreachable capacity inventory retry state')


def _command(argv, timeout=60):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    c.require(result.returncode == 0, 'Actual metadata command failed: ' + repr(argv) + ': ' + result.stderr)
    return {'argv': argv, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def storage(op, parent, phase):
    """NEW sealed caller adapter; metadata-only, never a scheduler submitter."""
    c.require(phase in ('preflight', 'entry', 'dispatch', 'periodic'), 'Rolling admission phase differs')
    parent = Path(parent)
    attempt = parent / ('rolling-capacity-' + phase + '-' + str(time.time_ns()))
    attempt.mkdir(exist_ok=False)
    try:
        return _storage(op, parent, phase, attempt)
    except BaseException as error:
        op.write(attempt / 'failure.json', {'status': 'failed_not_an_admission', 'error': repr(error),
                                          'partial_inventory_used': False, 'files_skipped': False,
                                          'historical_gates_or_scientific_guards_changed': False})
        raise


def _storage(op, parent, phase, attempt):
    plan = op.read(op.SOURCE / 'plan.json')
    p = policy(plan)
    pin = binding(op.SOURCE / 'source_manifest.json')['sha256']
    # Write actual complete account observation before assigning any scopes.
    command = _command(['bjobs', '-u', 's234818', '-json', '-o', 'jobid stat queue job_name command'])
    op.write(attempt / 'scheduler.json', command)
    scheduler_proof = binding(attempt / 'scheduler.json')
    rows = parse_scheduler(command['stdout'])
    own = op.read(op.ROOT / 'jobs.json') if (op.ROOT / 'jobs.json').exists() else {}
    c.require(type(own) is dict and len(set(own.values())) == len(own), 'Actual own job registry malformed')
    for name, jid in own.items():
        sub = op.read(op.ROOT / name / 'submission.json')
        c.require(sub.get('status') == 'submitted' and sub.get('job_id') == jid,
                  'Own job registry/submission identity differs')
    observed, external = classify_jobs(rows, p['known_mosaic_jobs'], set(own.values()), op.SOURCE)
    scopes, goals, claim = build_scopes(op, plan, pin, phase, parent, scheduler_proof, observed)
    if claim is not None:
        basis_path = attempt / ('completed-validation-inventory-' + claim['arm'] + '-' + str(claim['offset']) + '.json')
        op.write(basis_path, claim['validation_inventory'])
        claim['validation_inventory_evidence'] = binding(basis_path)
    # Uncertain quota visibility is covered by charging ALL retained current
    # Mosaic bytes once as well; no invented cache age or LastUpdate is used.
    inventories = complete_capacity_inventory(op, scopes, attempt)
    op.write(attempt / 'inventory.json', {'status': 'complete', 'complete': True, 'nofollow': True,
                                        'scope_rows': inventories, 'created_utc': datetime.now(timezone.utc).isoformat()})
    inventory_proof = binding(attempt / 'inventory.json')
    for scope, inventory in zip(scopes, inventories):
        scope.update(actual_bytes=inventory['actual_bytes'], actual_files=inventory['actual_files'],
                     inventory_proof=inventory_proof)
    quota = _command(['getquota_work3.sh'])
    op.write(attempt / 'quota.json', quota)
    fs = os.statvfs('/work3/s234818')
    inode = {'f_files': fs.f_files, 'f_ffree': fs.f_ffree, 'f_favail': fs.f_favail}
    ledger = {'version': c.VERSION, 'filesystem': 'work3', 'phase': phase,
              'account': {'name': 's234818', 'id': 306121}, 'source_manifest_sha256': pin,
              'observation': {'complete': True, 'nofollow': True, 'untracked_mosaic_writers': [],
                              'inventory_proof': inventory_proof, 'scheduler_proof': scheduler_proof},
              'scopes': scopes, 'goals': goals,
              'shared_account': {'non_mosaic_job_ids': external, 'their_future_bytes_reserved': False,
                                 'whole_account_reservation': False, 'entry_and_periodic_recheck_required': True}}
    result = c.assess(ledger, quota['stdout'], fs.f_bavail * fs.f_frsize, inode)
    result.update(adapter_version=VERSION, raw_quota_command=binding(attempt / 'quota.json'),
                  all_retained_mosaic_bytes_charged_once=True, historical_lifetime_allowance_chain_used=False,
                  inventory_proof=inventory_proof, scheduler_proof=scheduler_proof)
    if claim is not None:
        result['proposed_segment_claim_basis'] = dict(claim)
    op.write(attempt / 'assessment.json', result)
    c.require(result['status'] == 'passed', 'Scoped bounded continuation capacity failed; exact receipt preserved')
    if claim is not None:
        claim.update(admission_receipt=binding(attempt / 'assessment.json'),
                     admission_policy=c.VERSION, original_total_goal_preserved=True)
        directory = op.ROOT / 'storage-claims'
        directory.mkdir(exist_ok=True)
        op.write(directory / (claim['arm'] + '-segment-' + str(claim['offset']) + '.json'), claim)
    # Immutable attempt receipt is authoritative; compatibility file is exclusive
    # per original caller phase and never rewrites historical release receipts.
    if phase != 'periodic':
        op.write(parent / ('continuation-storage-' + phase + '.json'), result)
    return result
