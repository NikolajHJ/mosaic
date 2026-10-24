"""Record actual backup02 preparation without changing original receipts."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'campaign.json';raw=path.read_bytes();old=json.loads(raw)
    with (HERE/'campaign-checkpoint-01.json').open('xb') as out:out.write(raw)
    approval=json.loads((HERE/'approved-selection-02.json').read_bytes())
    audit=json.loads((HERE/'operator-verification-04.json').read_bytes())
    assert audit['status']=='passed' and audit['cases']==44 and audit['job_id']=='29806439'
    value={**old,'status':'fresh_backup02_client_started_no_deletion_verified',
        'updated_utc':datetime.now(timezone.utc).isoformat(),'operation_id':'legacy-offload-02',
        'operator_source_manifest_sha256':audit['source_manifest_sha256'],
        'allocated_operator':{'job_id':audit['job_id'],'actual_status':'DONE','same_job_lsf_success':True,
            'actual_raw_cases_passed':44,'verification':'operator-verification-04.json'},
        'previous_checkpoint':{'path':'campaign-checkpoint-01.json','sha256':hashlib.sha256(raw).hexdigest()},
        'previous_backup_attempt':old['backup'],
        'approved_plan':{'path':'results/hpc/20261024-storage-offload-01/backup-02/approved-plan.json',
            'bytes':approval['plan_bytes'],'sha256':approval['plan_sha256'],'exact_files':approval['files'],
            'exact_candidate_roots':approval['candidate_roots'],'logical_backup_bytes':approval['logical_bytes'],
            'unique_logical_candidate_bytes':approval['unique_logical_bytes'],
            'retained_incomplete_or_protected_groups':7,'selection_unchanged':True},
        'backup':{'local_path':'results/hpc/20261024-storage-offload-01/backup-02/legacy-offload-02.tar.gz',
            'remote_completion_receipt':'/work3/s234818/mosaic/runs/20261024-storage-offload-01/legacy-offload-02/stream-complete.json',
            'status':'fresh_client_started_completion_not_yet_verified','gzip_compresslevel':1,
            'no_large_remote_archive':True,'all_members_locally_verified':False},
        'metadata_changes':'Indexed equivalent canonical path protections and faster gzip1; original file bytes and all authentication/alias/finite-quota/current-job/race/hash/unlink guards unchanged.',
        'local_transport_failure_preserved':'operator-submission-04 WinError206 before child spawn; separate framed-stdin caller stages identical20605B reviewed code and exact31403B archive, succeeded.',
        'removed_files_verified':0,'actual_quota_savings_established':False}
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'status':value['status'],'operation_id':value['operation_id'],'cpu44':audit['job_id'],'removed_files_verified':0}))


if __name__=='__main__':main()
