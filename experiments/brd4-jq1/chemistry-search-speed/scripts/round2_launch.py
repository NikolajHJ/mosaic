"""One-shot stdlib stage/CPU submission; GPU only via verified callback."""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-chemistry-speed-02'
REMOTE=r'''
import ast,base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'chemistry-search-speed-02';release=base/'chemistry-search-speed-release-02';source=release/'source'
assert not any(os.path.lexists(p) for p in [root,release]),'Existing/unknown benchmark outcome; inspect, never replay'
archive=base64.b64decode(ARCHIVE)
assert len(archive)==LENGTH and hashlib.sha256(archive).hexdigest()==SHA
with tarfile.open(fileobj=io.BytesIO(archive)) as t:
 rows=t.getmembers();assert len(rows)==MEMBERS and len({r.name for r in rows})==len(rows)
 assert all(r.isfile() and not PurePosixPath(r.name).is_absolute() and '..' not in PurePosixPath(r.name).parts for r in rows)
 data={r.name:t.extractfile(r).read() for r in rows}
assert hashlib.sha256(data['source_manifest.json']).hexdigest()==MANIFEST
m=json.loads(data['source_manifest.json']);assert set(data)==set(m['files'])|{'source_manifest.json'}
for name,pin in m['files'].items():
 assert hashlib.sha256(data[name]).hexdigest()==pin
 if name.endswith('.py'):ast.parse(data[name],filename=name)
root.mkdir()
with (root/'source.tar.gz').open('xb') as f:f.write(archive)
source.mkdir(parents=True)
for name,raw in data.items():
 p=source/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('xb') as f:f.write(raw)
 p.chmod(0o444)
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555);release.chmod(0o555)
sys.path.insert(0,str(source/'scripts'))
import round2_worker as w
w.verify(source,MANIFEST);w.external()
subprocess.run(['bash','-n',str(source/'scripts/round2_stage.lsf')],check=True,capture_output=True)
old=json.loads((w.OLD_ROOT/'plan.json').read_text());inputs=[]
for row in old['inputs']:
 row=dict(row);p=Path(row['path']);assert w.sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
 f=w.OLD_ROOT/'cpu/payload/benchmark'/(row['label']+'-baseline.json')
 assert f.is_file() and not f.is_symlink()
 row['frozen_reference']={'path':str(f),'bytes':f.stat().st_size,'sha256':w.sha(f),'scope':'CPU prepared source01 reference; frozen for GPU kernel comparison'}
 inputs.append(row)
plan={'version':'chemistry-hybrid-gpu-matrix-v1','inputs':inputs,'search':old['search'],'progress_search':old['progress_search'],
      'cpu_counts':w.CPU_COUNTS,'gpu_counts':{'tests.test_gpu_execution':19},'workers':[4,8],
      'matrix_batches':[1,32,256],'value_gradient_tolerance':2e-5,'fd_epsilons':[.01,.003,.001],
      'fd_relative_error':.15,'fd_required_adjacent':2,'persistent_cap_bytes':w.LIMIT,
      'original_sources_results_jobs_changed':False,'monitor_remains_paused':True}
w.write(root/'plan.json',plan)
pre=root/'preflight';pre.mkdir();gate=w.storage(pre,'preflight')
cpu=w.submit('cpu',MANIFEST);after=w.submit('after-cpu',MANIFEST,'ended('+cpu['job_id']+')')
receipt={'status':'CPU_and_ownended_callback_submitted_not_passed','cpu':cpu,'after_cpu':after,
         'source_manifest_sha256':MANIFEST,'archive_sha256':SHA,'archive_bytes':LENGTH,
         'plan_sha256':w.sha(root/'plan.json'),'inputs':inputs,'storage':gate,'GPU_submitted':False}
w.write(root/'initial-submissions.json',receipt)
print(json.dumps(receipt))
'''


def main():
    out=LOCAL/'launch-01';out.mkdir(exist_ok=False)
    package=LOCAL/'source01';p=json.loads((package/'preparation.json').read_text());raw=(package/'source.tar.gz').read_bytes()
    assert len(raw)==p['archive_bytes'] and hashlib.sha256(raw).hexdigest()==p['archive_sha256']
    bindings={'ARCHIVE':base64.b64encode(raw).decode(),'LENGTH':len(raw),'SHA':p['archive_sha256'],
              'MANIFEST':p['manifest_sha256'],'MEMBERS':p['files']+1}
    code='\n'.join(k+'='+repr(v) for k,v in bindings.items())+'\n'+REMOTE
    (out/'remote-program.py').write_text(code,encoding='utf-8',newline='\n')
    (out/'started.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
                                               'program_sha256':hashlib.sha256(code.encode()).hexdigest()},indent=2)+'\n')
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (out/'stdout.json').open('xb') as f,(out/'stderr.txt').open('xb') as e:
        r=subprocess.run(args,input=code.encode(),stdout=f,stderr=e,timeout=240)
    record={'returncode':r.returncode,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'stdout_sha256':hashlib.sha256((out/'stdout.json').read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    if r.returncode:print((out/'stderr.txt').read_text(errors='replace')[-5000:])
    else:
        d=json.loads((out/'stdout.json').read_bytes())
        print(json.dumps({'cpu':d['cpu']['job_id'],'after_cpu':d['after_cpu']['job_id'],
                          'status':d['status'],'required_bytes':d['storage']['required_bytes'],
                          'available_bytes':d['storage']['available_bytes']},indent=2))


if __name__=='__main__':main()
