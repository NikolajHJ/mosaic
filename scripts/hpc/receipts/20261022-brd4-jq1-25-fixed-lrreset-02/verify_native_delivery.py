"""Independent byte/schema/scheduler/resource/cleanup audit, without numerics."""
from pathlib import Path
import hashlib,json,re
H=Path(__file__).resolve().parent;REPO=H.parents[3]
D=REPO/'results/hpc'/H.name/'native-delivery-01'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
manifest=read(D/'evidence_manifest.json')['files'];assert len(manifest)==41
for n,pin in manifest.items():
 p=D/n;assert p.resolve().is_relative_to(D.resolve()) and p.is_file() and not p.is_symlink()
 assert p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256']
origins=read(D/'origin-map.json');lookup={remote:D/name for name,remote in origins.items()}
adoption=read(D/'campaign/native-recovery-01/adoption.json')
assert sha(D/'campaign/native-recovery-01/adoption.json')=='434a42093cb466c32eb7fe144eb952f462c72e75014c0a5347fd95b1370ce8d4'
assert adoption['status']=='passed' and adoption['scientific_or_resource_policy_changes'] is False and adoption['original_after_cpu_callback_repeated'] is False
assert adoption['cpu_job']=='29781363' and adoption['tests']==723
for key in ('original_failed_gate','cleanup_receipt','fresh_storage_gate'):
 assert sha(lookup[adoption[key]['path']])==adoption[key]['sha256']
cert=D/'campaign/cpu-certificate.json'
assert sha(cert)==adoption['cpu_certificate_sha256']=='6e02392176f71238b1af36564c796615758ba82af7715ebe487bd48030a5a5d5'
assert read(H/'cpu-delivery-verification.json')['status']=='passed' and read(cert)['tests']==723
source=D/'source/source_manifest.json';source_sha=sha(source)
assert source_sha==adoption['source_manifest_sha256']=='832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3'
gate=read(D/'campaign/native-validation-recovery-01-quota/lrreset-storage-gate.json');g=gate['filesystems']['work3']
assert gate['status']==g['status']=='passed' and g['required_bytes']==10200547328 and g['available_bytes_conservative']==10286294816
assert gate['full_validation_allowance_bytes']==3*1024**3 and gate['concurrent_analysis_allowance_bytes']==2*1024**3+512*1024**2
assert g['reserve_bytes']==4*1024**3 and g['file_reserve']==100000 and g['available_files_lower']==19619040
assert g['filesystem_inode_reporting']=='unreported_all_zero'
root='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02'
src='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-02/source'
jobs=read(D/'campaign/native-submissions.json');assert jobs['all_three_submitted'] is True and jobs['long_runs_submitted'] is False
expected={'protenix':('29783703','29783705',.3),'boltz2':('29783707','29783708',.005),'boltz2_chemistry':('29783710','29783711',.005)}
states={}
compact=re.sub(r'\s+','',(D/'scheduler-current.txt').read_text())
for segment in re.split(r'(?=Job<\d+>)',compact):
 job=re.match(r'Job<(\d+)>',segment);status=re.search(r'Status<([^>]+)>',segment)
 if job and status:states[job.group(1)]=status.group(1)
for row in jobs['jobs']:
 arm=row['arm'];native,callback,rate=expected[arm]
 for role,job,phase in [('native',native,'validate'),('callback',callback,'after-ready')]:
  r=row[role];a=r['args'];parent=root+'/'+phase+'-'+arm
  assert r==read(D/'campaign'/(phase+'-'+arm)/'submission.json')
  assert r['outcome']=='submitted' and r['job_id']==job and r['returncode']==0
  assert r['source_manifest_sha256']==source_sha and r['cpu_certificate_sha256']==sha(cert)
  assert r['learning_rate']==rate and r['config_sha256']==read(source)['files'][f'examples/configs/brd4_jq1_25_fixed_lrreset_{arm}_10000.json']
  assert a[:5]==['bsub','-q','gpul40s' if role=='native' else 'hpc','-n','4' if role=='native' else '1']
  assert a[-8:]==[src,root,source_sha,phase,arm,parent,'0' if role=='native' else native,sha(cert)]
  assert a[a.index('-W')+1]==('02:00' if role=='native' else '00:30')
  if role=='native':assert '-w' not in a and a[a.index('-gpu')+1]=='num=1:mode=exclusive_process' and 'rusage[mem=8GB]' in a
  else:assert '-gpu' not in a and a[a.index('-w')+1]=='ended('+native+')' and 'rusage[mem=2GB]' in a
  assert states[job] in ('PEND','RUN')
cleanup=read(D/'cleanup/attempt-02/receipt.json');plan=read(D/'cleanup/cleanup-plan.json')
assert cleanup['status']=='completed' and cleanup['removed_files']==34847 and cleanup['removed_logical_bytes']==1540133784
assert cleanup['retained_changed_files']==cleanup['already_absent_files']==0
assert cleanup['plan_sha256']==sha(D/'cleanup/cleanup-plan.json')=='03bbc4fa67a286b8f51b6c02f3af8b898e0ed4acd80255e9f88527f82787c8bb'
assert cleanup['deletion_journal_sha256']==sha(D/'cleanup/attempt-02/deletion-journal.jsonl')
assert cleanup['verified_candidates_sha256']==sha(D/'cleanup/attempt-02/verified-candidates.json')
assert plan['maximum_removed_bytes']==2*1024**3
archive_audit=read(H/'older-work3-backup-02-stream-audit.json');assert archive_audit['status']=='passed'
assert read(D/'cleanup/backup-receipt.json')['sha256']==archive_audit['archive_sha256']=='1f6b91446dba1866ff7c6e732a1ce2457e60c59661943fdbd2eb6cf870a305f4'
backup=read(Path(archive_audit['manifest_directory'])/'evidence_manifest.json')['files']
mapping={row['path']:row for row in plan['files']};assert len(mapping)==34847
for row in mapping.values():assert backup[row['archive_member']]=={'bytes':row['bytes'],'sha256':row['sha256']}
seen=set();pending=None
with (D/'cleanup/attempt-02/deletion-journal.jsonl').open() as f:
 for line in f:
  assert line.endswith('\n');row=json.loads(line);original=mapping[row['path']]
  assert all(row[k]==original[k] for k in ('bytes','sha256','local_backup','archive_member'))
  if row['event']=='intent':assert pending is None and row['path'] not in seen;pending=row['path']
  else:assert row['event']=='removed' and pending==row['path'];seen.add(row['path']);pending=None
assert pending is None and len(seen)==34847
state=read(D/'capture-state.json');assert not state['new_scientific_long_submission_file_exists'] and not state['new_long_dispatch_marker_exists'] and not any(state['production_output_parents_exist'].values())
record={'status':'passed','scope':'Independent local standard-library byte/schema/actual scheduler/resource/config/CPU723/source622 and complete cleanup-journal audit; no scientific or numerical calculation',
 'files':41,'captured_utc':state['utc'],'native_jobs':{a:expected[a][0] for a in expected},'callbacks':{a:expected[a][1] for a in expected},'actual_states':states,
 'source_manifest_sha256':source_sha,'cpu_certificate_sha256':sha(cert),'cpu_tests':723,'adoption_sha256':sha(D/'campaign/native-recovery-01/adoption.json'),
 'storage_required_bytes':g['required_bytes'],'storage_available_bytes':g['available_bytes_conservative'],
 'cleanup_verified_removed_files':34847,'cleanup_verified_removed_logical_bytes':1540133784,'backup_archive_sha256':archive_audit['archive_sha256'],
 'new_native_readiness_pass_established':False,'new_10000_update_job_ids_established':False,'current_original_higherrate_studies_and_home_untouched':True}
with (H/'native-delivery-verification.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','verification_sha256':sha(H/'native-delivery-verification.json'),'new_native_jobs':record['native_jobs'],'states':states}))
