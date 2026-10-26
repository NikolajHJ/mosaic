"""One-shot authorized stage plus conditionally gated production CPU/callback.

Authoring/importing this helper performs no action. Execution authenticates a
fresh local package and remote paths, stages a read-only source, and invokes
the sealed bootstrap once. All numerical work remains allocated HPC only.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261026-chemistry-production-04'
REMOTE=r'''
import ast,base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
if sys.flags.optimize:raise RuntimeError('Optimized Python cannot bypass source/receipt checks')
def require(value,message):
 if not value:raise ValueError(message)
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'chemistry-production-04';release=base/'chemistry-production-release-04';source=release/'source'
require(not any(os.path.lexists(path) for path in (root,release)),
        'Existing/unknown production or release outcome; inspect, never replay')
archive=base64.b64decode(ARCHIVE,validate=True)
require(len(archive)==LENGTH and hashlib.sha256(archive).hexdigest()==SHA,'Transferred archive identity differs')
with tarfile.open(fileobj=io.BytesIO(archive)) as target:
 rows=target.getmembers()
 require(len(rows)==MEMBERS and len({row.name for row in rows})==len(rows),'Archive membership differs')
 require(all(row.isfile() and not PurePosixPath(row.name).is_absolute()
             and '..' not in PurePosixPath(row.name).parts and '\\' not in row.name for row in rows),
         'Archive traversal/link/special member rejected')
 require(sum(row.size for row in rows)<=64*1024**2,'Source exceeds complete own control cap')
 data={row.name:target.extractfile(row).read() for row in rows}
require(hashlib.sha256(data['source_manifest.json']).hexdigest()==MANIFEST,'Source manifest changed')
manifest=json.loads(data['source_manifest.json'])
require(set(data)==set(manifest['files'])|{'source_manifest.json'},'Exact source member scope differs')
for name,pin in manifest['files'].items():
 require(hashlib.sha256(data[name]).hexdigest()==pin,'Source member bytes changed')
 if name.endswith('.py'):ast.parse(data[name],filename=name)
release.mkdir();source.mkdir()
with (release/'source.tar.gz').open('xb') as stream:stream.write(archive)
with (release/'staging-started.json').open('x',encoding='utf8') as stream:
 json.dump({'source_manifest_sha256':MANIFEST,'archive_sha256':SHA,'archive_bytes':LENGTH},stream,indent=2);stream.write('\n')
for name,raw in data.items():
 path=source/name;path.parent.mkdir(exist_ok=True,parents=True)
 with path.open('xb') as stream:stream.write(raw)
 path.chmod(0o444)
for path in sorted((p for p in source.rglob('*') if p.is_dir()),reverse=True):path.chmod(0o555)
source.chmod(0o555)
(release/'source.tar.gz').chmod(0o444);(release/'staging-started.json').chmod(0o444);release.chmod(0o555)
sys.path.insert(0,str(source/'scripts'))
import production_ops as ops
actual_manifest,plan=ops.own_plan(MANIFEST)
require(actual_manifest['cpu_counts']==manifest['cpu_counts'],'Sealed CPU attribution differs')
# These are actual submitted producer/CPU proof checks. A RUN/queued pilot is
# not claimed passed; the new CPU dependency waits its actual ended event.
for arm,row in plan['pilots'].items():
 metadata=ops.pilot_metadata(arm,plan)
 old=metadata.verify(Path(row['source']),row['manifest_sha256']);metadata.external(old)
 metadata.verify_cpu_certificate(row['manifest_sha256'],old,row['cpu_certificate_sha256'])
 parent=Path(row['root'])/arm;sub=ops.read(parent/'submission.json')
 require(sub['status']=='submitted' and sub['job_id']==row['job_id']
         and sub['argv']==metadata.argv(arm,row['manifest_sha256'])
         and sub['cpu_certificate_sha256']==row['cpu_certificate_sha256'],
         'Declared actual producer/source/CPU/argv differs')
 require(ops.read(Path(row['root'])/'jobs.json')[arm]==row['job_id'],'Actual producer map differs')
subprocess.run(['bash','-n',str(source/'scripts/production_stage.lsf')],check=True,capture_output=True)
cpu,after=ops.bootstrap(MANIFEST)
receipt={'status':'CPU_after_both_actual_pilots_callback_submitted_not_passed',
         'cpu':cpu,'after_cpu':after,'source_manifest_sha256':MANIFEST,'plan_sha256':ops.sha(source/'plan.json'),
         'archive_sha256':SHA,'archive_bytes':LENGTH,'source_files':MEMBERS-1,
         'storage':ops.read(root/'production-storage-preflight.json'),
         'production_jobs_submitted':False,'scientific_updates':0,
         'allocated_CPU_pass':False,'new_readiness_passed':False,'monitor_resumed':False}
ops.write(root/'staging-complete.json',receipt)
ops.enforce_control_cap()
print(json.dumps(receipt))
'''


def require(value,message):
    if not value:raise ValueError(message)


def main():
    require(REPO.name=='mosaic','Wrong authorized repository')
    package=LOCAL/'source03';preparation=json.loads((package/'preparation.json').read_bytes())
    raw=(package/'source.tar.gz').read_bytes()
    require(len(raw)==preparation['archive_bytes'] and hashlib.sha256(raw).hexdigest()==preparation['archive_sha256'],
            'Fresh local archive identity differs')
    out=LOCAL/'launch-01';out.mkdir(exist_ok=False)
    values={'ARCHIVE':base64.b64encode(raw).decode(),'LENGTH':len(raw),'SHA':preparation['archive_sha256'],
            'MANIFEST':preparation['manifest_sha256'],'MEMBERS':preparation['files']+1}
    code='\n'.join(name+'='+repr(value) for name,value in values.items())+'\n'+REMOTE
    (out/'remote-program.py').write_text(code,encoding='utf8',newline='\n')
    (out/'started.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
        'program_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'unknown_outcome_policy':'Inspect exact remote paths/receipts after any interruption; never replay'},indent=2)+'\n')
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
             'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'status':'unknown_inspect_remote_before_recovery'}
    try:
        with (out/'stdout.json').open('xb') as output,(out/'stderr.txt').open('xb') as error:
            result=subprocess.run(command,input=code.encode(),stdout=output,stderr=error,timeout=300)
        receipt={'returncode':result.returncode,'completed_utc':datetime.now(timezone.utc).isoformat(),
                 'stdout_sha256':hashlib.sha256((out/'stdout.json').read_bytes()).hexdigest()}
    except BaseException as error:
        receipt.update(error=repr(error));raise
    finally:(out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    if result.returncode:
        print((out/'stderr.txt').read_text(errors='replace')[-6000:]);raise SystemExit(result.returncode)
    delivery=json.loads((out/'stdout.json').read_bytes())
    print(json.dumps({'status':delivery['status'],'cpu':delivery['cpu']['job_id'],
        'after_cpu':delivery['after_cpu']['job_id'],'CPU_dependency':delivery['cpu']['dependency'],
        'callback_dependency':delivery['after_cpu']['dependency'],
        'required_bytes':delivery['storage']['required_bytes'],'available_bytes':delivery['storage']['available_bytes']},indent=2))


if __name__=='__main__':main()
