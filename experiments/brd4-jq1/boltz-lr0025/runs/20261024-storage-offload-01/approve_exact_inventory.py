"""Root-selected exact archive/remove plan after positive dependency reviews."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RESULTS = REPO / 'results/hpc/20261024-storage-offload-01'
REMOTE_ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def protected(name, value):
    p = PurePosixPath(name)
    return name in value['exact'] or any(p == PurePosixPath(r) or p.is_relative_to(PurePosixPath(r)) for r in value['subtrees'])


def main():
    source = HERE / 'verify_stream_backup_02.py'
    assert digest(source) == '9401d8220cf82d38991179f3c1ae2da33a902058ab03020803e751266d95c46a'
    spec = importlib.util.spec_from_file_location('validated_offload_verifier', source)
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    record = json.loads((RESULTS / 'exact-inventory-02/result.json').read_bytes())
    assert record['returncode'] == 0 and record['script_sha256'] == digest(HERE / 'scan_exact_inventory_02.py')
    inventory_raw = (RESULTS / 'exact-inventory-02/stdout.bin').read_bytes()
    inventory = json.loads(inventory_raw)
    assert inventory['complete'] is True and inventory['read_only'] is True
    assert inventory['input_sha256'] == digest(HERE / 'exact-inventory-input-02.json')
    config = json.loads((HERE / 'exact-inventory-input-02.json').read_bytes())
    rules = {row['path']: row for row in config['roots']}
    rows = inventory['files']
    assert len(rows) == 25737 and len({row['path'] for row in rows}) == len(rows)
    assert inventory['logical_bytes'] == sum(row['bytes'] for row in rows) == 32041124599
    assert inventory['unique_logical_bytes'] == 21891767118
    roots = set()
    for row in rows:
        p = PurePosixPath(row['path'])
        assert p.is_relative_to('/work3/s234818/latentmol/runs')
        root = str(PurePosixPath(*p.parts[:6]))
        assert root in rules and not protected(row['path'], config['protection'])
        if rules[root]['mode'] == 'binary_only':
            assert p.suffix.lower() in {'.pt', '.pth', '.ckpt', '.npz', '.npy'}
        assert len(row['sha256']) == 64 and row['bytes'] >= 0
        roots.add(root)
    group_paths = set()
    for group in inventory['inode_groups']:
        assert group['nlink'] == len(group['aliases'])
        assert not group_paths.intersection(group['aliases'])
        group_paths.update(group['aliases'])
    assert group_paths == {row['path'] for row in rows}
    for observation in (inventory['queue_before'], inventory['queue_after']):
        assert observation['query'] == ['bjobs', '-noheader', '-o', 'jobid stat']
        assert {row['job_id'] for row in observation['jobs']} <= set(config['audited_active_job_ids'])
    protection = config['protection']
    scan = json.loads((RESULTS / 'shared-closure-scan-02/stdout.bin').read_bytes())
    protection['subtrees'] = sorted(set(protection['subtrees']) | (set(scan['top_run_inventory']) - roots))
    for item in inventory['retained']:
        protection['exact'].extend(item.get('paths', []))
        if 'path' in item: protection['exact'].append(item['path'])
        protection['exact'].extend(row['path'] for row in item.get('aliases', []))
    protection['exact'] = sorted(set(protection['exact']))
    assert not any(protected(row['path'], protection) for row in rows)
    protection_raw = (json.dumps(protection, indent=2, sort_keys=True) + '\n').encode()
    cpu = json.loads((HERE / 'operator-verification-03.json').read_bytes())
    assert cpu['status'] == 'passed' and cpu['job_id'] == '29806228' and cpu['cases'] == 32
    assert cpu['source_manifest_sha256'] == '9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
    plan = {'version': 'mosaic-storage-offload-plan-v1', 'status': 'approved_archive_then_remove',
        'operation_id': 'legacy-offload-01', 'cleanup_root': REMOTE_ROOT,
        'approved_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User authorized inactive artifacts only after complete laptop download and verification, preserving every current experiment input/result/proof.',
        'approved_candidate_roots': sorted(roots),
        'protection_manifest': {'path': REMOTE_ROOT + '/protection-01.json', 'sha256': hashlib.sha256(protection_raw).hexdigest()},
        'audited_active_job_ids': config['audited_active_job_ids'], 'candidate_owner_job_ids': [],
        'timeout_seconds': 3600,
        'files': [{**row, 'member': 'files/' + format(index, '08d')} for index, row in enumerate(rows)],
        'root_review': 'Current actual VAE/embedding plans, 577+582 source seals, 176-module import graph, fourteen raw controller members/MRO and metadata proof closure; all current scientific dependencies kept. Only reviewed inactive run files selected. Other roots, root metadata and incomplete inode groups retained.',
        'inventory_sha256': hashlib.sha256(inventory_raw).hexdigest(),
        'operator_source_manifest_sha256': cpu['source_manifest_sha256'],
        'allocated_operator_verification_sha256': digest(HERE / 'operator-verification-03.json'),
        'allocated_operator_job_id': '29806228', 'allocated_operator_cases': 32,
        'logical_backup_bytes': inventory['logical_bytes'],
        'unique_logical_candidate_bytes': inventory['unique_logical_bytes'],
        'actual_quota_savings_established': False, 'current_saved_results_or_settings_changed': False}
    raw = (json.dumps(plan, indent=2, sort_keys=True) + '\n').encode()
    plan_sha = hashlib.sha256(raw).hexdigest()
    verifier.load_plan(raw, plan_sha)
    assert shutil.disk_usage(RESULTS).free >= plan['logical_backup_bytes'] + 4 * 1024**3
    folder = RESULTS / 'backup-01'; folder.mkdir(exist_ok=False)
    for path, data in ((folder / 'approved-plan.json', raw), (HERE / 'flat-protection-approved-01.json', protection_raw)):
        with path.open('xb') as stream: stream.write(data)
    approval = {'status': 'approved_for_backup_then_conditional_identical_copy_removal',
        'operation_id': plan['operation_id'], 'plan_bytes': len(raw), 'plan_sha256': plan_sha,
        'plan_path': str(folder / 'approved-plan.json'), 'files': len(rows), 'candidate_roots': len(roots),
        'logical_bytes': inventory['logical_bytes'], 'unique_logical_bytes': inventory['unique_logical_bytes'],
        'unique_allocated_blocks512': inventory['unique_allocated_blocks512'],
        'protection_sha256': hashlib.sha256(protection_raw).hexdigest(),
        'backup_or_deletion_completed': False, 'actual_quota_savings_established': False,
        'retained_inventory_groups': len(inventory['retained']),
        'required_before_removal': 'Complete actual remote stream receipt + exact laptop compressed archive hash and every original member verified; fresh active-set/protection/content/inode/race checks.'}
    with (HERE / 'approved-selection-01.json').open('x') as stream:
        json.dump(approval, stream, indent=2); stream.write('\n')
    print(json.dumps(approval))


if __name__ == '__main__':
    main()
