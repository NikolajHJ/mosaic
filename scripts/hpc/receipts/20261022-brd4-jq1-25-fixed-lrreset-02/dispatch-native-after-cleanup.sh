set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,re,subprocess,sys
source=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-02/source')
root=source.parent.parent/'restart-lrreset-02'
manifest='832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3'
cpu='6e02392176f71238b1af36564c796615758ba82af7715ebe487bd48030a5a5d5'
def require(v,message):
 if not v:raise ValueError(message)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
require(source.resolve()==source and root.resolve()==root,'Unsafe source/root')
subprocess.run(['/work3/s234818/mosaic/envs/protenix-tiny/bin/python',str(source/'scripts/verify_stage0_release.py'),str(source),manifest],check=True)
sys.path.insert(0,str(source/'scripts'))
import fixed25_lrreset_ops as m
require(m.SOURCE==source and m.ROOT==root,'Wrong CPU723-certified operator')
m.validate_configs();m.verify_cpu(manifest,cpu);m.verify_finished_predecessors()
failed=root/'native-validation-dispatch-quota/lrreset-storage-gate.json'
require(sha(failed)=='2c47b1e90fe4f7a4c2d259f95e370dea38dbef9c392c4becd7a186ebf31d8bc8'
        and m.read(failed)['status']=='failed','Original failed storage admission changed')
raw=(root/'cpu/scheduler-final-0.txt').read_text();compact=re.sub(r'\s+','',raw)
require(re.findall(r'Job<([^>]+)>',compact)==['29781363'] and re.findall(r'Status<([^>]+)>',compact)==['DONE'] and re.findall(r'Queue<([^>]+)>',compact)==['hpc'],'Actual original CPU DONE proof missing')
lsf=(root/'cpu/stdout.29781363.log').read_text()
require('Subject: Job 29781363:' in lsf and 'Successfully completed.' in lsf,'Original CPU723 LSF missing')
cleanup=root/'cleanup-recovery-02/attempt-02/receipt.json'
require(cleanup.is_file(),'Verified-backup cleanup has not completed')
clean=m.read(cleanup)
require(clean['status']=='completed' and clean['removed_logical_bytes']>0,'Cleanup receipt is not terminal')
with (root/'dispatch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for p in [root/'native-dispatch-started.json',root/'native-submissions.json',root/'long-dispatch-started.json',root/'long-submissions.json',root/'native-recovery-01']:
  require(not p.exists() and not p.is_symlink(),'Actual/unknown dispatch outcome exists; no duplicate')
 for arm in m.ARMS:
  for phase in ('validate','after-ready','run'):require(not m.native_parent(arm,phase).exists(),'Existing/unknown native/callback/scientific parent; no duplicate')
 recovery=root/'native-recovery-01';recovery.mkdir()
 m.exclusive(recovery/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'scope':'Metadata-only resource-admission retry through original CPU723-tested APIs; no original callback rerun/scientific change','source_manifest_sha256':manifest,'cpu_certificate_sha256':cpu})
 # The failed gate remains immutable. This is a new whole-capacity observation
 # with exactly the original3GiB validation+2.5GiB analysis+4GiB/file reserves.
 gate=m.storage.check_work_storage(root/'native-validation-recovery-01-quota',phase='validate')
 adoption={'version':'jq1-lrreset-native-resource-recovery-v1','status':'passed','utc':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':manifest,'cpu_certificate_sha256':cpu,'cpu_job':'29781363','tests':723,'original_failed_gate':m.hashed(failed),'cleanup_receipt':m.hashed(cleanup),'fresh_storage_gate':m.hashed(root/'native-validation-recovery-01-quota/lrreset-storage-gate.json'),'authorization':m.AUTHORIZATION,'scientific_or_resource_policy_changes':False,'original_after_cpu_callback_repeated':False}
 m.exclusive(recovery/'adoption.json',adoption)
 m.exclusive(root/'native-dispatch-started.json',{'version':m.VERSION,'source_manifest_sha256':manifest,'cpu_certificate_sha256':cpu,'authorization':m.AUTHORIZATION,'resource_recovery_adoption':m.hashed(recovery/'adoption.json')})
 jobs=[]
 for arm in m.ARMS:
  native=m.submit(arm,'validate',manifest,cpu)
  callback=m.submit(arm,'after-ready',manifest,cpu,producer=native['job_id'])
  jobs.append({'arm':arm,'native':native,'callback':callback})
  m.atomic(root/'native-submissions.json',{'version':m.VERSION,'jobs':jobs,'all_three_submitted':len(jobs)==3,'long_runs_submitted':False})
 print(json.dumps({'status':'all_three_fresh_native_checks_actually_submitted','jobs':[(j['arm'],j['native']['job_id'],j['callback']['job_id']) for j in jobs],'adoption_sha256':sha(recovery/'adoption.json'),'fresh_storage_gate':gate},indent=2))
PY
