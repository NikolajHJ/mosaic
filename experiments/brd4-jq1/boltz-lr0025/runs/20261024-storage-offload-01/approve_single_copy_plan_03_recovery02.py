"""Distinct recovery02 metadata approval; previous Windows metadata failure preserved.

No SSH, backup, deletion, scientific admission or setting change. Earlier plans,
partial archives and original files remain untouched. Fresh original remote
activity/protection/hash/race guards still apply to every subsequent action.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RESULTS = REPO / 'results/hpc/20261024-storage-offload-01'
ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'
SOURCE = '8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
OLD_PLAN_SHA = '5fc74e3eeb3eedb2df5dfec541bb62b31e2b03dc01b8d0dd20eb3ad277dff6ea'
RETIREMENT_SHA = '3324ff5a10fe9e461ab1946c4072859a8a09d4ffce72eca9d09cebcf2a9a2da2'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing/linked metadata: ' + str(path))
    literal_before = path.stat()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(64 * 1024**2 + 1); after = os.fstat(stream.fileno())
    literal_after = path.stat()
    signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns, x.st_mode, x.st_nlink)
    shared = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_mode, x.st_nlink)
    # Windows reports creation-time precision differently through fstat/stat.
    # Both observation channels must remain exact and independently unchanged;
    # all shared identity fields still cross-match, including inode/size/mtime.
    require(not path.is_symlink() and len(raw) == before.st_size <= 64 * 1024**2
            and signature(before) == signature(after)
            and signature(literal_before) == signature(literal_after)
            and shared(before) == shared(literal_before), 'Metadata changed/oversized')
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field'); result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    return raw, value


def canonical(name):
    require(isinstance(name, str) and name.startswith('/') and not any(c in name for c in '\\\n\r\x00*?[]'), 'Unsafe absolute path')
    path = PurePosixPath(name)
    require(str(path) == name and '..' not in path.parts, 'Noncanonical path')
    return path


def protection_check(value):
    exact = frozenset(value['exact']); subtrees = frozenset(canonical(x) for x in value['subtrees'])
    negative = tuple(value['negative'])
    def check(name):
        p = canonical(name)
        return (name in exact or p in subtrees or any(x in subtrees for x in p.parents)
                or any(fnmatch.fnmatchcase(name, pattern) for pattern in negative))
    return check


def retired_proof(path):
    raw, record = load(path)
    require(Path(path).resolve() == HERE / 'stream-retirement-02.json'
            and record['version'] == 'mosaic-intentional-backup-retirement-proof-v1'
            and record['path'] == ROOT + '/stream-retirement-complete-02.json'
            and record['bytes'] == 1132 and record['sha256'] == RETIREMENT_SHA
            and record['status'] == 'intentionally_stopped_own_incomplete_backup'
            and record['old_exact_producer_running'] is False and record['all_remote_originals_retained'] is True,
            'No authenticated incomplete02 retirement')
    require(set(record['files']) == {ROOT + '/stream-retirement-intent-02.json', record['path']}, 'Retirement scope differs')
    for name, pin in record['files'].items():
        data, _ = load(pin['local_path'])
        require(name == pin['path'] and len(data) == pin['bytes'] and sha(data) == pin['sha256'], 'Retirement raw proof differs')
    completed_raw, completed = load(record['local_path'])
    expected = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', ROOT + '/operator-source-04/stream_backup.py',
                '--plan-stdin', '--plan-sha256', OLD_PLAN_SHA, '--receipt', ROOT + '/legacy-offload-02/stream-complete.json']
    require(sha(completed_raw) == RETIREMENT_SHA and completed['pid'] == 177888 and completed['uid'] == 306121
            and completed['hostname'] == 'hpclogin1' and completed['signal'] == 'SIGTERM'
            and completed['producer_no_longer_running'] is True and completed['remote_originals_retained'] is True
            and completed['backup_verified'] is False and completed['original_files_deleted'] == 0
            and completed['scientific_jobs_or_settings_changed'] is False and completed['expected_argv'] == expected
            and completed['operation_entries'] == ['active-jobs-before-stream.json', 'approved-plan.json', 'stream-started.json'],
            'Wrong owned producer/operation retirement')
    return {'canonical_receipt': str(Path(path).resolve()), 'canonical_receipt_sha256': sha(raw),
            'path': record['path'], 'bytes': len(completed_raw), 'sha256': sha(completed_raw), 'status': completed['status']}


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def save(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retirement-receipt', type=Path, required=True)
    parser.add_argument('--cpu-evidence', type=Path, required=True)
    args = parser.parse_args()
    failure_raw, failure = load(HERE / 'plan03-preparation-attempt01-failure.json')
    require(failure['status'] == 'failed_before_plan03_outputs' and failure['plan03_outputs_absent'] is True
            and failure['source_sha256'] == 'f7fa6b07aed73162e455bd3dff5068575e892c27cf2117b9e89ec848a50ace00'
            and sha((HERE / 'approve_single_copy_plan_03_attempt01.py').read_bytes()) == failure['source_sha256'], 'Prior failed preparer proof differs')
    retired = retired_proof(args.retirement_receipt)
    # Reuse only the reviewed actual CPU/source/XML verifier, never its emitter.
    binder = HERE / 'bind_offload_operation_02.py'
    require(sha(binder.read_bytes()) == '368142296f3c95e15e5b6e853e51272794fa3851a1bcb8899b2c97a37d845444', 'CPU verifier changed')
    spec = importlib.util.spec_from_file_location('known_cpu04_contract', binder)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    job, cpu_raw, _ = module.prove_actual_cpu(args.cpu_evidence)
    require(job == '29806439', 'Actual CPU44 identity differs')
    old_raw, old = load(RESULTS / 'backup-02/approved-plan.json')
    approval_raw, old_approval = load(HERE / 'approved-selection-02.json')
    require(sha(old_raw) == OLD_PLAN_SHA and old_approval['plan_sha256'] == OLD_PLAN_SHA
            and old_approval['status'] == 'approved_for_backup_then_conditional_identical_copy_removal'
            and old_approval['operation_id'] == old['operation_id'] == 'legacy-offload-02'
            and old['operator_source_manifest_sha256'] == SOURCE and old['allocated_operator_cases'] == 44
            and old['allocated_operator_job_id'] == job, 'Original approved02 identity differs')
    inventory_raw, inventory = load(RESULTS / 'exact-inventory-02/stdout.bin')
    _, outcome = load(RESULTS / 'exact-inventory-02/result.json')
    config_raw, config = load(HERE / 'exact-inventory-input-02.json')
    require(outcome['returncode'] == 0 and outcome['script_sha256'] == sha((HERE / 'scan_exact_inventory_02.py').read_bytes())
            and inventory['complete'] is True and inventory['read_only'] is True and inventory['numerical_execution'] is False
            and inventory['input_sha256'] == sha(config_raw) and sha(inventory_raw) == old['inventory_sha256'], 'Complete inventory differs')
    rows = inventory['files']; by_path = {r['path']: r for r in rows}
    require(len(rows) == len(by_path) == 25737 and sum(r['bytes'] for r in rows) == inventory['logical_bytes'] == 32041124599,
            'Original file inventory differs')
    require([{k: v for k, v in row.items() if k != 'member'} for row in old['files']] == rows, 'Old plan no longer equals exact inventory')
    covered, selected, linked = set(), set(), set()
    singles, link_groups = [], []
    for group in inventory['inode_groups']:
        aliases = group['aliases']
        require(type(group['nlink']) is int and group['nlink'] >= 1 and len(aliases) == len(set(aliases)) == group['nlink']
                and not covered.intersection(aliases), 'Incomplete/duplicate inode group')
        covered.update(aliases); records = [by_path[name] for name in aliases]; first = records[0]
        for row in records:
            canonical(row['path'])
            require(row['nlink'] == group['nlink'] and row['device'] == group['device'] and row['inode'] == group['inode']
                    and row['bytes'] == group['unique_bytes'] and row['blocks512'] == group['allocated_blocks512']
                    and all(row[k] == first[k] for k in ('sha256', 'bytes', 'mtime_ns', 'mode', 'ctime_ns')), 'Group/file identity differs')
        if group['nlink'] == 1:
            require(len(records) == 1 and first['nlink'] == 1, 'Selected file is linked')
            selected.add(aliases[0]); singles.append(group)
        else:
            linked.update(aliases); link_groups.append(group)
    require(covered == set(by_path) and selected.isdisjoint(linked) and selected | linked == set(by_path), 'Incomplete inode partition')
    chosen = [row for row in rows if row['path'] in selected]
    require(len(chosen) == len(singles) == 25413 and len(link_groups) == 123 and len(linked) == 324
            and sum(r['bytes'] for r in chosen) == 15725430139 and sum(g['unique_bytes'] for g in link_groups) == 6166336979,
            'Exact single-link subset differs')
    old_roots = frozenset(canonical(x) for x in old['approved_candidate_roots']); roots = set()
    for row in chosen:
        matches = set(canonical(row['path']).parents) & old_roots
        require(len(matches) == 1, 'Selected path not in exactly one old root'); roots.update(matches)
    old_protection_raw, protection = load(HERE / 'flat-protection-approved-01.json')
    require(sha(old_protection_raw) == old['protection_manifest']['sha256']
            and old['protection_manifest']['path'] == ROOT + '/protection-01.json', 'Original protection differs')
    protection = copy.deepcopy(protection)
    protection['exact'] = sorted(set(protection['exact']) | linked)
    protection['subtrees'] = sorted(set(protection['subtrees']) | {str(x) for x in old_roots - roots})
    blocked = protection_check(protection)
    require(not any(blocked(r['path']) for r in chosen) and all(blocked(name) for name in linked), 'Protection overlaps selection/omits linked files')
    protection_raw = encode(protection)
    plan = copy.deepcopy(old)
    plan.update(operation_id='legacy-offload-03', approved_utc=datetime.now(timezone.utc).isoformat(),
        approved_candidate_roots=sorted(str(x) for x in roots), protection_manifest={'path': ROOT + '/protection-03.json', 'sha256': sha(protection_raw)},
        files=[{**row, 'member': 'files/' + format(index, '08d')} for index, row in enumerate(chosen)],
        selection_policy='complete_single_link_inode_groups_only', selection_unchanged_from_original_approved_plan=False,
        previous_approved_plan_sha256=OLD_PLAN_SHA, retired_previous_operation=retired,
        allocated_operator_verification_sha256=sha(cpu_raw), allocated_operator_job_id=job, allocated_operator_cases=44,
        logical_backup_bytes=15725430139, unique_logical_candidate_bytes=15725430139,
        excluded_linked_inode_groups=123, excluded_linked_alias_files=324, excluded_linked_unique_logical_bytes=6166336979,
        excluded_alias_logical_backup_bytes=16315694460, actual_quota_savings_established=False, current_saved_results_or_settings_changed=False,
        root_review='Only complete old nlink1 groups; all linked aliases and old roots without selected files additionally protected. Scientific budgets/gates and all partial archives preserved.')
    require(plan['audited_active_job_ids'] == config['audited_active_job_ids'] and plan['candidate_owner_job_ids'] == [], 'Active/owner scope differs')
    require(shutil.disk_usage(RESULTS).free >= plan['logical_backup_bytes'] + 4 * 1024**3, 'Full laptop backup plus4GiB reserve does not fit')
    raw = encode(plan); folder = RESULTS / 'backup-03'; marker = HERE / 'single-copy-plan-03-started.json'
    approval_path = HERE / 'approved-selection-03.json'; protection_path = HERE / 'flat-protection-approved-03.json'
    require(not any(p.exists() or p.is_symlink() for p in (folder, marker, approval_path, protection_path)), 'Existing/unknown03 outcome; inspect, never replay')
    save(marker, encode({'status': 'unknown_until03_files_written', 'retirement': retired, 'source_manifest_sha256': SOURCE,
                         'actual_cpu_job_id': job, 'actual_cpu_cases': 44, 'backup_or_deletion_executed': False}))
    folder.mkdir(exist_ok=False)
    save(folder / 'approved-plan.json', raw); save(protection_path, protection_raw)
    approval = {'status': 'approved_for_backup_then_conditional_identical_copy_removal', 'operation_id': plan['operation_id'],
        'plan_path': str(folder / 'approved-plan.json'), 'plan_bytes': len(raw), 'plan_sha256': sha(raw), 'files': len(chosen), 'candidate_roots': len(roots),
        'logical_bytes': plan['logical_backup_bytes'], 'unique_logical_bytes': plan['unique_logical_candidate_bytes'],
        'unique_allocated_blocks512': sum(g['allocated_blocks512'] for g in singles), 'protection_sha256': sha(protection_raw),
        'operator_source_manifest_sha256': SOURCE, 'allocated_operator_job_id': job, 'allocated_operator_cases': 44,
        'allocated_operator_verification_sha256': sha(cpu_raw), 'selection_policy': plan['selection_policy'],
        'previous_approved_plan_sha256': OLD_PLAN_SHA, 'original_inventory_sha256': sha(inventory_raw), 'retired_previous_operation': retired,
        'excluded_linked_inode_groups': 123, 'excluded_linked_alias_files': 324, 'excluded_linked_unique_logical_bytes': 6166336979,
        'old_roots_additionally_protected': len(old_roots - roots), 'retained_inventory_groups': len(inventory['retained']),
        'backup_or_deletion_completed': False, 'actual_quota_savings_established': False, 'whole_scientific_budget_or_gates_changed': False,
        'required_before_removal': 'Completed actual stream and all laptop members verified; original fresh activity/protection/content/inode/race gates.'}
    save(approval_path, encode(approval))
    require(sha((RESULTS / 'backup-02/approved-plan.json').read_bytes()) == OLD_PLAN_SHA
            and sha((HERE / 'approved-selection-02.json').read_bytes()) == sha(approval_raw), 'Historical02 changed')
    print(json.dumps(approval))


if __name__ == '__main__':
    main()
