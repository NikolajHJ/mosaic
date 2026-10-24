"""Approve the unchanged exact file selection for a distinct validated backup.

Only metadata/container tooling changes. Original plan, partial stream and
retirement evidence are preserved; every original byte is rehashed while the
new stream is packed and again before any future literal unlink.
"""
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
RESULTS=REPO/'results/hpc/20261024-storage-offload-01'
MANIFEST='8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    audit_path=HERE/'operator-verification-04.json'
    audit=json.loads(audit_path.read_bytes())
    assert audit['status']=='passed' and audit['cases']==44 and audit['source_manifest_sha256']==MANIFEST
    assert audit['failures']==audit['errors']==audit['skips']==0
    old_approval=json.loads((HERE/'approved-selection-01.json').read_bytes())
    old_raw=(RESULTS/'backup-01/approved-plan.json').read_bytes()
    assert hashlib.sha256(old_raw).hexdigest()==old_approval['plan_sha256']=='6573d5d471df56b229e5d1f9adb28b74ddf8959b7d6c419332ca36b45f11d07b'
    old=json.loads(old_raw);assert old['operation_id']=='legacy-offload-01'
    retirement=json.loads((HERE/'stream-retirement-01.json').read_bytes())
    assert retirement['status']=='intentionally_stopped_own_incomplete_backup'
    assert retirement['old_exact_producer_running'] is False and retirement['all_remote_originals_retained'] is True
    assert retirement['sha256']=='41d238499d97bdb0e415c684a5ccf95bb2c73b7264395c310816e144e83988ab'
    transfer=json.loads((RESULTS/'backup-01/legacy-offload-01.tar.gz.transfer.json').read_bytes())
    assert transfer['status']=='failed_or_unknown_partial_preserved' and transfer['timed_out'] is False
    assert transfer['all_files_verified'] is False and transfer['deletion_permitted'] is False
    assert transfer['received_bytes']==1169453216 and transfer['received_sha256']=='ae5c9639cf018247be0b007b28c77ab0e1fedc30c982a9189236119e70b336e1'
    protection_raw=(HERE/'flat-protection-approved-01.json').read_bytes()
    assert hashlib.sha256(protection_raw).hexdigest()==old['protection_manifest']['sha256']=='35c78f3571b26304e453eb9da1e04c1fe0bdd05d0a6e3756e81e3f825d121511'
    protection=json.loads(protection_raw)
    exact=set(protection['exact']);subtrees={PurePosixPath(p) for p in protection['subtrees']}
    roots={PurePosixPath(p) for p in old['approved_candidate_roots']}
    for row in old['files']:
        path=PurePosixPath(row['path']);ancestors=set(path.parents)
        assert roots.intersection(ancestors) and row['path'] not in exact
        assert not subtrees.intersection(ancestors|{path})
        assert not any(fnmatch.fnmatchcase(row['path'],p) for p in protection['negative'])
    assert len(old['files'])==25737 and sum(r['bytes'] for r in old['files'])==32041124599
    plan={**old,'operation_id':'legacy-offload-02','approved_utc':datetime.now(timezone.utc).isoformat(),
        'operator_source_manifest_sha256':MANIFEST,'allocated_operator_verification_sha256':digest(audit_path),
        'allocated_operator_job_id':audit['job_id'],'allocated_operator_cases':44,
        'audited_active_job_ids':sorted(set(old['audited_active_job_ids'])|{audit['job_id']}),
        'previous_attempt':{'approved_plan_sha256':hashlib.sha256(old_raw).hexdigest(),
            'retirement_receipt':{k:retirement[k] for k in ('path','bytes','sha256')},
            'partial_archive_bytes':transfer['received_bytes'],'partial_archive_sha256':transfer['received_sha256'],
            'status':'intentionally_retired_incomplete_backup_no_original_deletion'},
        'container_and_metadata_change':'New independently allocated44-case source04: gzip level1 and equivalent compiled canonical path membership; every byte/hash/alias/protection/activity/race/deletion guard retained.',
        'selection_unchanged_from_original_approved_plan':True}
    raw=(json.dumps(plan,indent=2,sort_keys=True)+'\n').encode();sha=hashlib.sha256(raw).hexdigest()
    verifier=HERE/'verify_stream_backup_02.py'
    assert digest(verifier)=='9401d8220cf82d38991179f3c1ae2da33a902058ab03020803e751266d95c46a'
    spec=importlib.util.spec_from_file_location('validated_stream_verifier',verifier)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.load_plan(raw,sha)
    assert shutil.disk_usage(RESULTS).free>=plan['logical_backup_bytes']+4*1024**3
    folder=RESULTS/'backup-02';folder.mkdir(exist_ok=False)
    with (folder/'approved-plan.json').open('xb') as out:out.write(raw)
    approval={'status':'approved_for_backup_then_conditional_identical_copy_removal','operation_id':plan['operation_id'],
        'plan_path':str((folder/'approved-plan.json').resolve()),'plan_bytes':len(raw),'plan_sha256':sha,
        'files':len(plan['files']),'candidate_roots':len(roots),'logical_bytes':plan['logical_backup_bytes'],
        'unique_logical_bytes':plan['unique_logical_candidate_bytes'],'protection_sha256':plan['protection_manifest']['sha256'],
        'operator_source_manifest_sha256':MANIFEST,'allocated_operator_job_id':audit['job_id'],'allocated_operator_cases':44,
        'selection_unchanged':True,'previous_approved_plan_sha256':hashlib.sha256(old_raw).hexdigest(),
        'backup_or_deletion_completed':False,'actual_quota_savings_established':False,
        'remote_protection_policy':'Reuse only the exact existing immutable protection-01.json by its original SHA; no overwrite.'}
    with (HERE/'approved-selection-02.json').open('x') as out:json.dump(approval,out,indent=2);out.write('\n')
    print(json.dumps(approval))


if __name__=='__main__':main()
