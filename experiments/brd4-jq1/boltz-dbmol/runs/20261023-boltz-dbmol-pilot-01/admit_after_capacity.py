"""One fresh metadata-only admission after verified owned-upload retirement.

The failed gate remains immutable. No previous CPU/submission exists. This caller
does not modify or replace the authenticated scientific source, run settings,
native checks, quotas or reserve. It uses a unique exclusive admission record.
"""
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
r=b/'boltz-dbmol-pilot-01';s=b/'boltz-dbmol-pilot-release-01/source'
manifest='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
failed=r/'storage-preflight/gate.json'
assert hashlib.sha256(failed.read_bytes()).hexdigest()=='9822d97a999ba1232692a17e4e9308518eecf09e583988e93f8954be6d5d0c1f'
assert json.loads(failed.read_bytes())['status']=='failed'
assert not (r/'jobs.json').exists() and not (r/'cpu').exists() and not (r/'after-cpu').exists()
assert not list(r.glob('**/submission*.json')),'Unknown submission outcome blocks admission'
raw=(s/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest
for name,digest in json.loads(raw)['files'].items():
 p=s/name
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
# Retire only a third redundant upload created in THIS request. The caller has
# independently authenticated its complete local backup and immutable633 source.
upload=b/'boltz-lr0025-02.tar.gz';source=b/'boltz-lr0025-release-02/source'
assert upload.is_file() and not upload.is_symlink() and upload.stat().st_size==39618948
with upload.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa'
raw=(source/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='3689071a4e7dd135071ff5b420c026543c4d10887c951077dff63d498d9f812c'
for name,digest in json.loads(raw)['files'].items():
 p=source/name
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
admission=r/'admission-02';admission.mkdir(exist_ok=False)
receipt={'utc':datetime.now(timezone.utc).isoformat(),'prior_failed_gate_sha256':hashlib.sha256(failed.read_bytes()).hexdigest(),
 'scope':'Fresh byte/file observation only; old failed gate preserved; no prior CPU or scientific execution',
 'retired_upload':str(upload),'bytes':39618948,'sha256':'40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa',
 'local_backup':'results/hpc/20261023-boltz-lr0025-02/source01/source.tar.gz','immutable_source633_reverified':True}
with (admission/'started.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
upload.unlink()
receipt['retirement']='exact_redundant_upload_removed_all_studies_and_source_directories_unchanged'
with (admission/'upload-retirement.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
sys.dont_write_bytecode=True;sys.path[:0]=[str(s/'scripts'),str(s/'src')]
import dbmol_ops as ops
gate=ops.storage_gate(admission,'preflight')
ops.exclusive(r/'jobs.json',{})
cpu=ops.submit('cpu','cpu',manifest)
callback=ops.submit('after-cpu','after-cpu',manifest,'ended('+cpu['job_id']+')')
result={'status':'submitted_not_passed','source_manifest_sha256':manifest,'prior_failed_gate_preserved':receipt,
 'cpu':cpu,'callback':callback,'fresh_gate':gate}
ops.exclusive(admission/'outcome.json',result)
print('ADMISSION_BEGIN');print(json.dumps(result))
'''

def main():
    data=(REPO/'results/hpc/20261023-boltz-lr0025-02/source01/source.tar.gz').read_bytes()
    assert len(data)==39618948 and hashlib.sha256(data).hexdigest()=='40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa'
    folder=REPO/'results/hpc'/HERE.name/'admission-02';folder.mkdir(exist_ok=False)
    cmd=' '.join(shlex.quote(v) for v in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE])
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(cmd)]
    (folder/'started.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'args':args},indent=2)+'\n')
    try:q=subprocess.run(args,capture_output=True,timeout=180)
    except subprocess.TimeoutExpired as e:(folder/'unknown.json').write_text(json.dumps({'error':repr(e),'outcome':'unknown_no_blind_duplicate'})+'\n');raise
    (folder/'stdout.bin').write_bytes(q.stdout);(folder/'stderr.bin').write_bytes(q.stderr);(folder/'result.json').write_text(json.dumps({'returncode':q.returncode})+'\n')
    print(q.stdout.decode()[-14000:]);print(q.stderr.decode()[-1500:]);assert q.returncode==0,'Preserved failed/unknown admission; inspect before any recovery'
    result=json.loads(q.stdout.split(b'ADMISSION_BEGIN\n')[1]);(folder/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
