"""One explicit admission after source05's preserved pre-submission storage failure.

This invokes only the exact existing source05 storage/submit APIs. No upload,
source modification, numerical evaluation, gate override or optimizer branch.
"""
import ast,hashlib,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[4]
OUT=REPO/'results/hpc/20261027-protenix-endpoint-fix-05/readmission01'
PROGRAM=r'''
import fcntl,hashlib,json,os,stat,sys
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
source=base/'protenix-endpoint-fix-release-05/source';root=base/'protenix-endpoint-fix-05'
pin='65b4bd350386a8550c3b588eb7de2b0f6084b32ad790a907851c632dba0f023e'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(source/'source_manifest.json')==pin
m=json.loads((source/'source_manifest.json').read_bytes());assert len(m['files'])==16 and sum(m['cpu_counts'].values())==286
assert set(m['cpu_counts'])=={'tests.test_current_capacity','tests.test_protenix_precision_admission','tests.test_protenix_precision_native','tests.test_protenix_precision_ops','tests.test_protenix_precision_reference'}
for name,digest in m['files'].items():
 p=source/name;assert p.is_file() and all(not q.is_symlink() for q in (p,*p.parents)) and sha(p)==digest
assert {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}==set(m['files'])|{'source_manifest.json'}
assert set(p.name for p in root.iterdir())=={'preflight'},'Existing/unknown admission or jobs must be inspected, never replayed'
failed=root/'preflight/current-chemistry-capacity-preflight/production-storage-entry.json'
assert sha(failed)=='450240abaa7f29cf5a41df06daa55e3de02ed6276d3e3991b9c54058d5df6a16'
failure=json.loads(failed.read_bytes());assert failure['status']=='failed'
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.path.insert(0,str(source/'scripts'))
import protenix_precision_ops as op
assert op.ROOT==root and op.SOURCE==source
manifest=op.verify(source,pin);plan=op.validate_plan(op.read(source/'plan.json'));op.external(manifest,plan)
operation=root/'admission-recovery-01';operation.mkdir(exist_ok=False)
op.write(operation/'started.json',{'source_manifest_sha256':pin,'preserved_failed_gate_sha256':sha(failed),'no_jobs_before':True,'original_results_changed':False})
with (operation/'dispatch.lock').open('x') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not (root/'jobs.json').exists() and all(not (root/p).exists() for p in ('cpu','after-cpu','gpu'))
 gate=op.storage(operation,'preflight')
 assert gate['status']=='passed' and gate['reserve_bytes']==4*1024**3 and gate['file_reserve']==100000
 cpu=op.submit('cpu',pin);callback=op.submit('after-cpu',pin,dependency='ended('+cpu['job_id']+')')
 assert sha(failed)=='450240abaa7f29cf5a41df06daa55e3de02ed6276d3e3991b9c54058d5df6a16'
 record={'status':'exact_existing_source05_cpu_admitted_not_passed','source_manifest_sha256':pin,'cpu':cpu,'after_cpu':callback,'whole_storage':gate,'original_results_changed':False,'optimization_settings_changed':False,'optimizer_updates':0}
 op.write(operation/'completion.json',record);print(json.dumps(record))
'''
def main():
    ast.parse(PROGRAM);OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'program.py').write_text(PROGRAM,encoding='utf8',newline='\n')
    receipt={'status':'unknown_inspect_remote_operation_before_recovery','started_utc':datetime.now(timezone.utc).isoformat(),'program_sha256':hashlib.sha256(PROGRAM.encode()).hexdigest()}
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    try:
        with (OUT/'stdout.json').open('xb') as output,(OUT/'stderr.txt').open('xb') as error:r=subprocess.run(argv,input=PROGRAM.encode(),stdout=output,stderr=error,timeout=400)
        receipt['returncode']=r.returncode
        if r.returncode:raise RuntimeError((OUT/'stderr.txt').read_text(errors='replace')[-3000:])
        data=json.loads((OUT/'stdout.json').read_bytes());receipt.update(status=data['status'],cpu=data['cpu']['job_id'],after_cpu=data['after_cpu']['job_id'])
        print(json.dumps(receipt,indent=2))
    finally:
        receipt['ended_utc']=datetime.now(timezone.utc).isoformat();receipt['stdout_sha256']=hashlib.sha256((OUT/'stdout.json').read_bytes()).hexdigest()
        (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
if __name__=='__main__':main()
