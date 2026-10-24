"""Root-only, one-shot collection of small authenticated offload receipts.

This caller never streams original payloads, verifies a laptop backup, deletes
files, submits jobs or replays a backup. Preparation is standard-library only.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RESULTS = REPO / 'results/hpc/20261024-storage-offload-01'
ROOT = '/work3/s234818/mosaic/runs/20261024-storage-offload-01'
OPERATION = 'legacy-offload-01'
MANIFEST = '9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
MAX_BYTES = 64 * 1024 ** 2
PHASE_FILES = {
    'stream': ('stream-started.json', 'stream-complete.json',
               'active-jobs-before-stream.json', 'active-jobs-after-stream.json'),
    'deletion': ('deletion-started.json', 'deletion-complete.json', 'deletion-journal.jsonl'),
}

REMOTE = r'''
import base64,hashlib,json,stat,sys
from datetime import datetime,timezone
from pathlib import Path
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
source=root/'operator-source-03';operation=root/'legacy-offload-01'
phase,expected_plan=sys.argv[1:]
names={'stream':('stream-started.json','stream-complete.json','active-jobs-before-stream.json','active-jobs-after-stream.json'),
       'deletion':('deletion-started.json','deletion-complete.json','deletion-journal.jsonl')}[phase]
maximum=64*1024**2
def raw(path):
 path=Path(path)
 assert path.is_file() and not path.is_symlink() and path.resolve()==path,'Noncanonical/linked receipt or source'
 with path.open('rb') as stream:data=stream.read(maximum+1)
 assert len(data)<=maximum,'Metadata file exceeds64MiB'
 return data
def sha(data):return hashlib.sha256(data).hexdigest()
assert root.is_dir() and root.resolve()==root and source.is_dir() and source.resolve()==source
manifest_path=source/'source-manifest.json';manifest_raw=raw(manifest_path)
assert sha(manifest_raw)=='9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
manifest=json.loads(manifest_raw);members=manifest['files']
assert len(members)==11 and set(p.name for p in source.iterdir())==set(members)|{'source-manifest.json'}
assert not source.stat().st_mode&0o222
source_proofs=[]
for name in ['source-manifest.json',*sorted(members)]:
 assert Path(name).name==name and name not in ('.','..')
 path=source/name;data=raw(path)
 assert stat.S_ISREG(path.stat().st_mode) and not path.stat().st_mode&0o222
 assert sha(data)==(sha(manifest_raw) if name=='source-manifest.json' else members[name])
 source_proofs.append({'path':str(path),'bytes':len(data),'sha256':sha(data),'local_bytes_downloaded':False})
assert operation.is_dir() and operation.resolve()==operation
plan_path=operation/'approved-plan.json';plan_raw=raw(plan_path)
assert sha(plan_raw)==expected_plan,'Remote approved plan differs from laptop plan'
plan=json.loads(plan_raw)
assert plan['operation_id']=='legacy-offload-01' and plan['cleanup_root']==str(root)
assert plan['version']=='mosaic-storage-offload-plan-v1' and plan['status']=='approved_archive_then_remove'
assert plan['operator_source_manifest_sha256']==sha(manifest_raw)
assert plan['protection_manifest']['path']==str(root/'protection-01.json')
protection_raw=raw(root/'protection-01.json')
assert sha(protection_raw)==plan['protection_manifest']['sha256']
files={}
for name in names:
 path=operation/name;data=raw(path)
 files[str(path)]={'bytes':len(data),'sha256':sha(data),'base64':base64.b64encode(data).decode('ascii')}
print(json.dumps({'version':'mosaic-offload-receipt-collection-v1','phase':phase,
 'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'operation_id':plan['operation_id'],
 'plan':{'path':str(plan_path),'bytes':len(plan_raw),'sha256':sha(plan_raw),'local_bytes_downloaded':False},
 'protection':{'path':str(root/'protection-01.json'),'bytes':len(protection_raw),'sha256':sha(protection_raw),'local_bytes_downloaded':False},
 'source_manifest_sha256':sha(manifest_raw),'source_proofs':source_proofs,'files':files,
 'original_payload_downloaded':False,'backup_replayed':False,'files_deleted':0}))
'''


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def load_plan():
    approval = strict_json((HERE / 'approved-selection-01.json').read_bytes())
    raw = (RESULTS / 'backup-01/approved-plan.json').read_bytes()
    require(0 < len(raw) <= MAX_BYTES and sha(raw) == approval['plan_sha256']
            and len(raw) == approval['plan_bytes'], 'Exact locally approved plan differs')
    plan = strict_json(raw)
    require(approval['status'] == 'approved_for_backup_then_conditional_identical_copy_removal'
            and approval['operation_id'] == plan['operation_id'] == OPERATION
            and plan['cleanup_root'] == ROOT and plan['status'] == 'approved_archive_then_remove',
            'Wrong exact approved operation')
    protection = (HERE / 'flat-protection-approved-01.json').read_bytes()
    require(plan['protection_manifest'] == {'path': ROOT + '/protection-01.json',
            'sha256': sha(protection)} and sha(protection) == approval['protection_sha256'],
            'Approved protection identity differs')
    audit_raw = (HERE / 'operator-verification-03.json').read_bytes()
    audit = strict_json(audit_raw)
    require(audit['status'] == 'passed' and audit['job_id'] == '29806228' and audit['cases'] == 32
            and audit['source_manifest_sha256'] == plan['operator_source_manifest_sha256'] == MANIFEST
            and sha(audit_raw) == plan['allocated_operator_verification_sha256'],
            'Allocated operator source/CPU proof differs')
    return plan, raw


def validate_receipts(phase, records, plan, plan_raw):
    directory = ROOT + '/' + OPERATION + '/'
    require(set(records) == {directory + name for name in PHASE_FILES[phase]}, 'Receipt inventory differs')
    decoded = {}
    for name, record in records.items():
        raw = base64.b64decode(record['base64'], validate=True)
        require(len(raw) <= MAX_BYTES and len(raw) == record['bytes'] and sha(raw) == record['sha256'],
                'Collected raw receipt length/SHA differs')
        decoded[name.removeprefix(directory)] = raw
    started = strict_json(decoded[phase + '-started.json'])
    completed = strict_json(decoded[phase + '-complete.json'])
    for value in (started, completed):
        require(value['operation_id'] == OPERATION and value['plan_sha256'] == sha(plan_raw),
                'Receipt operation/approved plan differs')
    if phase == 'stream':
        require(started['version'] == completed['version'] == 'mosaic-storage-offload-stream-v1'
                and started['status'] == 'started_unknown_until_complete'
                and completed['status'] == 'streamed_complete' and completed['all_files_verified'] is True
                and completed['plan_bytes'] == len(plan_raw)
                and completed['manifest_member'] == 'offload-manifest.json'
                and completed['files_count'] == len(plan['files'])
                and completed['member_count'] == len(plan['files']) + 1
                and completed['logical_bytes'] == sum(row['bytes'] for row in plan['files'])
                and type(completed['archive_bytes']) is int and completed['archive_bytes'] > 0
                and re.fullmatch(r'[0-9a-f]{64}', completed['archive_sha256']), 'Stream did not complete exactly')
        for name in ('active-jobs-before-stream.json', 'active-jobs-after-stream.json'):
            value = strict_json(decoded[name]); ids = [row['job_id'] for row in value['jobs']]
            require(value['query'] == ['bjobs', '-noheader', '-o', 'jobid stat']
                    and len(ids) == len(set(ids)) and set(ids) <= set(plan['audited_active_job_ids'])
                    and not set(ids) & set(plan['candidate_owner_job_ids']), 'Unauthenticated active-set observation')
    else:
        require(started['version'] == completed['version'] == 'mosaic-storage-offload-delete-v1'
                and started['status'] == 'started_unknown_until_complete'
                and completed['status'] in ('exact_deletion_completed', 'exact_deletion_completed_with_retention')
                and completed['journal_sha256'] == sha(decoded['deletion-journal.jsonl'])
                and completed['recursive_delete_used'] is False
                and completed['scientific_settings_or_saved_retained_results_changed'] is False,
                'Deletion metadata did not complete with its exact journal')
        lines = decoded['deletion-journal.jsonl'].splitlines(keepends=True)
        require(lines and all(line.endswith(b'\n') for line in lines), 'Incomplete deletion journal')
        events = [strict_json(line) for line in lines]
        unlinked = [row for row in events if row['event'] == 'unlinked_verified_copy']
        originals = {row['path']: row for row in plan['files']}
        require(len({row['path'] for row in unlinked}) == len(unlinked)
                and all(row['path'] in originals and row['bytes'] == originals[row['path']]['bytes']
                        and row['sha256'] == originals[row['path']]['sha256'] for row in unlinked),
                'Journal removal is not an exact unique planned copy')
        require(len(unlinked) == completed['removed_files']
                and sum(row['bytes'] for row in unlinked) == completed['removed_logical_bytes']
                and completed['removed_files'] + completed['retained_changed_or_unreadable']
                    + completed['already_absent_no_removal_claim'] == len(plan['files']),
                'Deletion completion/journal totals differ')
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=tuple(PHASE_FILES), required=True)
    args = parser.parse_args(); plan, plan_raw = load_plan()
    folder = RESULTS / 'backup-01' / (args.phase + '-receipts-01')
    folder.mkdir(exist_ok=False)
    (folder / 'remote-caller.py').write_text(REMOTE, encoding='utf8', newline='\n')
    remote = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE, args.phase, sha(plan_raw)]
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
               'dtu-hpc', 'bash -l -c ' + shlex.quote(' '.join(shlex.quote(value) for value in remote))]
    (folder / 'started.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
        'outcome': 'unknown', 'phase': args.phase, 'args': command, 'plan_sha256': sha(plan_raw),
        'collector_sha256': sha(Path(__file__).read_bytes()), 'remote_code_sha256': sha(REMOTE.encode())}) + '\n')
    try:
        with (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=180)
    except BaseException as error:
        (folder / 'failed-unknown.json').write_text(json.dumps({'outcome': 'unknown_preserved', 'error': repr(error)}) + '\n')
        raise
    (folder / 'result.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
    require(result.returncode == 0, 'Collection failed; preserved outcome must be inspected, never blindly replayed')
    value = strict_json((folder / 'stdout.bin').read_bytes())
    require(value['read_only'] is True and value['phase'] == args.phase and value['operation_id'] == OPERATION
            and value['source_manifest_sha256'] == MANIFEST and len(value['source_proofs']) == 12
            and value['plan']['path'] == ROOT + '/' + OPERATION + '/approved-plan.json'
            and value['plan']['sha256'] == sha(plan_raw) and value['plan']['bytes'] == len(plan_raw)
            and value['protection']['sha256'] == plan['protection_manifest']['sha256']
            and value['original_payload_downloaded'] is False and value['files_deleted'] == 0,
            'Collected source/operation/scope metadata differs')
    manifest_raw = (HERE / 'operator-source-manifest-03.json').read_bytes()
    require(sha(manifest_raw) == MANIFEST, 'Local validated source manifest differs')
    expected = {ROOT + '/operator-source-03/' + name: digest
                for name, digest in strict_json(manifest_raw)['files'].items()}
    expected[ROOT + '/operator-source-03/source-manifest.json'] = MANIFEST
    require({row['path']: row['sha256'] for row in value['source_proofs']} == expected
            and all(row['local_bytes_downloaded'] is False and type(row['bytes']) is int
                    and row['bytes'] >= 0 for row in value['source_proofs']), 'Source-member rehash inventory differs')
    decoded = validate_receipts(args.phase, value['files'], plan, plan_raw)
    index = {key: value[key] for key in ('version', 'phase', 'utc', 'operation_id', 'plan',
                                      'protection', 'source_manifest_sha256', 'source_proofs')}
    index.update(status='authenticated_receipt_metadata_collected', files={},
                 is_laptop_backup_verification=False, authorizes_deletion=False, scientific_passed=False)
    for name, raw in decoded.items():
        with (folder / name).open('xb') as stream: stream.write(raw)
        index['files'][ROOT + '/' + OPERATION + '/' + name] = {
            'local_path': name, 'bytes': len(raw), 'sha256': sha(raw)}
    with (folder / 'receipt-index.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(index, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'status': index['status'], 'phase': args.phase, 'files': len(decoded), 'folder': str(folder)}))


if __name__ == '__main__':
    main()
