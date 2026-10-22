"""Make an exact removal plan only from a fully verified local byte backup."""
from pathlib import Path
import ast,hashlib,json
h=Path(__file__).resolve().parent;repo=h.parents[3]
result=json.loads((h/'older-work3-backup-02-result.json').read_text())
audit=json.loads((h/'older-work3-backup-02-stream-audit.json').read_text())
assert audit['status']=='passed' and result['outcome']=='archive_bytes_verified' and result['returncode']==0
archive=Path(result['local_archive']);root=Path(audit['manifest_directory'])
assert archive.stat().st_size==audit['archive_bytes'] and audit['archive_sha256']==result['receipt']['sha256']
manifest=json.loads((root/'evidence_manifest.json').read_text())['files']
origins=json.loads((root/'origin-map.json').read_text())
rows=[]
for name,origin in origins.items():
    entry=manifest[name]
    rows.append({'path':origin['path'],'bytes':entry['bytes'],'sha256':entry['sha256'],
                 'local_backup':str(archive),'archive_member':name})
assert len(rows)==len(origins) and 0<sum(r['bytes'] for r in rows)<=2*1024**3
# No direct proof from the three higher-rate runs or diagnostics may be removed.
delivery=repo/'results/hpc/20261022-brd4-jq1-lrboost-analysis-01/interim-delivery-evidence-02/analysis'
protected=set()
for arm in ('protenix','boltz2','boltz2_chemistry'):
    a=json.loads((delivery/('complete-'+arm)/'payload/assessment.json').read_text())
    protected.update(p['path'] for p in a['proofs'])
assert not protected.intersection(r['path'] for r in rows)
plan={'version':'mosaic-lrreset-verified-work3-backup-cleanup-v1','allowed_base':'/work3/s234818/mosaic',
    'maximum_removed_bytes':2*1024**3,'authorization':'Existing user request to download previous results and clean storage; limited to inactive older work3 copies after complete local verification',
    'protected':'All three current higher-rate saved trajectories, diagnostic/report inputs, baseline initialization, all home files, releases/models/caches/environments/other projects',
    'backup_archive_sha256':result['receipt']['sha256'],'files':rows}
plan_path=repo/'results/hpc'/h.name/'cleanup-plan-02.json'
with plan_path.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
old=(h.parent/'20261021-mosaic-storage-cleanup-01/cleanup_verified_files.py').read_text()
start=old.index('def safe_path(raw):');end=old.index('\n\ndef active_jobs',start)
old=old[:start]+'''def safe_path(raw):
    path=Path(raw)
    allowed_roots=[BASE/'runs/20261021-brd4-jq1-25-fixed-storage-03/run-protenix/payload/study',
        BASE/'runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2/payload/study',
        BASE/'runs/20261021-brd4-jq1-25-fixed-report-04/interim-0001/payload']
    allowed_files={BASE/'runs/20261021-brd4-jq1-25-fixed-report-04/interim-0001.tar.gz',
        BASE/'runs/20261021-brd4-jq1-25-fixed-02/native-evidence-01.tar.gz',
        BASE/'20261021-brd4-jq1-25-fixed-02.tar.gz',BASE/'20261021-brd4-jq1-25-fixed-lrboost-01.tar.gz'}
    if not path.is_absolute() or '..' in path.parts or path.resolve()!=path:
        raise ValueError('Unsafe/linked cleanup path')
    if path not in allowed_files and not any(path!=root and path.is_relative_to(root) for root in allowed_roots):
        raise ValueError('Outside the exact inactive older-work3 allowlist')
    return path
'''+old[end:]
old=old.replace("CONTROL = BASE / 'runs/20261021-mosaic-storage-cleanup-01'", "CONTROL = BASE / 'runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02/cleanup-recovery-02'")
old=old.replace("'mosaic-verified-backup-cleanup-v1'", "'mosaic-lrreset-verified-work3-backup-cleanup-v1'").replace('<= 6 * 1024**3','<= 2 * 1024**3')
ast.parse(old)
with (h/'cleanup_verified_work3.py').open('xb') as f:f.write(old.encode())
receipt={'status':'ready','files':len(rows),'logical_bytes':sum(r['bytes'] for r in rows),'plan_local_path':str(plan_path.resolve()),'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(old.encode()).hexdigest(),'all_current_collector_proofs_excluded':True,'all_home_files_excluded':True,'deletion_performed':False}
with (h/'cleanup-plan-02.receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
