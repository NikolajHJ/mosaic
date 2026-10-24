"""Prepare one fresh read-only adoption metadata collection; root owns execution.

No arrays/models are imported or deserialized. Large/expired references remain
explicitly scoped; collection alone certifies neither native science nor a start.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[3]
REMOTE = r'''
import base64,hashlib,json,os,re,stat,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
profiles={
 'boltz30k':{'root':BASE/'boltz-lr0025-03','control':BASE/'boltz-pro02-adoption-04','launcher':BASE/'boltz-pro02-adoption-release-04/source','source':BASE/'boltz-lr0025-release-03/source','manifest':'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'},
 'dbmol10k':{'root':BASE/'boltz-dbmol-pilot-02','control':BASE/'dbmol-pro02-adoption-06','launcher':BASE/'dbmol-pro02-adoption-release-06/source','source':BASE/'boltz-dbmol-pilot-release-02/source','manifest':'600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed'}}
profile,launcher_sha=sys.argv[1:]
assert profile in profiles and re.fullmatch('[0-9a-f]{64}',launcher_sha) and not sys.flags.optimize
p=profiles[profile];begin=time.monotonic()
result={'version':'mosaic-peer-adoption-read-only-evidence-v1','profile':profile,'launcher_manifest_sha256':launcher_sha,'captured_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'numerical_execution':False,'files':{},'external_files':{},'missing_references':[],'opaque_traversal_boundaries':[],'sources':{},'roots':{},'scheduler':{}}
PARENT_FRAME_INVENTORY=str(BASE/'lrreset-analysis-03/complete-protenix/payload/snapshots.json')
PARENT_FRAME_SHA='ec8fa62462e5e9db4bb83f431ed43121e30b619f3bc7b43e53b7295224d5d919'
downloaded=hashed=0;queue=[];seen=set();MAX_FILES=6000
def require(ok,message):
 if not ok:raise ValueError(message)
def signature(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_nlink)
def allowed(path):
 return any(path.is_relative_to(root) for root in (Path('/work3/s234818/mosaic'),Path('/zhome/95/e/206121/mosaic-runs'),Path('/work3/s234818/latentmol/cache')))
def add(path,expected=None,metadata_only=False):
 global downloaded,hashed
 path=Path(path);key=str(path)
 require(path.is_absolute() and '..' not in path.parts and allowed(path),'Proof outside reviewed scopes: '+key)
 if key in result['files'] or key in result['external_files']:
  row=result['files'].get(key,result['external_files'].get(key))
  if row['sha256'] is None and expected is not None and row['remote_bytes_rehashed'] is False:row['sha256']=expected
  require(expected is None or row['sha256']==expected,'Conflicting proof pin');return
 require(len(result['files'])+len(result['external_files'])<MAX_FILES and time.monotonic()-begin<210,'Bounded evidence collection exceeded')
 require(path.resolve()==path and not path.is_symlink(),'Linked proof: '+key)
 if not path.exists():
  result['missing_references'].append({'path':key,'expected_sha256':expected,'claim':'Missing reference is not a pass; disposable node-TMP references may have expired.'});return
 before=path.stat();require(stat.S_ISREG(before.st_mode),'Nonregular proof')
 size=before.st_size
 if size>128*1024**2 or hashed+size>2*1024**3:
  require(expected is None or re.fullmatch('[0-9a-f]{64}',expected),'Malformed large original pin')
  result['external_files'][key]={'path':key,'bytes':size,'sha256':expected,'local_bytes_downloaded':False,'remote_bytes_rehashed':False,'scope':'Prior proof reference and current length only; no new content digest claim. Null SHA means unpinned live output, not authenticated content.'};return
 keep=not metadata_only and size<=4*1024**2 and path.suffix not in ('.npz','.pt','.ckpt','.pkl','.gz','.eqx')
 with path.open('rb') as f:
  opened=os.fstat(f.fileno());require(signature(before)==signature(opened),'Proof changed before read')
  digest=hashlib.sha256();parts=[];count=0
  for block in iter(lambda:f.read(1024**2),b''):
   count+=len(block);require(count<=size and time.monotonic()-begin<210,'Proof grew or collection expired');digest.update(block)
   if keep:parts.append(block)
  after=os.fstat(f.fileno())
 require(count==size and signature(before)==signature(after)==signature(path.stat()),'Proof changed during read')
 hashed+=count;value=digest.hexdigest();require(expected is None or value==expected,'Original proof digest changed: '+key)
 row={'path':key,'bytes':size,'sha256':value,'remote_bytes_rehashed':True,'local_bytes_downloaded':keep}
 if keep:
  data=b''.join(parts);downloaded+=size;require(downloaded<=32*1024**2,'Compact byte limit exceeded')
  row['base64']=base64.b64encode(data).decode();result['files'][key]=row
  if path.suffix=='.json':queue.append(key)
 else:result['external_files'][key]=row
def seal(root,expected,own=False):
 root=Path(root);require(root.is_dir() and root.resolve()==root and not root.stat().st_mode&0o222,'Source root not sealed')
 mp=root/'source_manifest.json';add(mp,expected)
 manifest=json.loads(base64.b64decode(result['files'][str(mp)]['base64']));files=manifest['files'];observed=set();proofs=[]
 for directory,dirs,names in os.walk(root,followlinks=False):
  require(not Path(directory).stat().st_mode&0o222,'Writable source directory')
  for name in dirs:require(not (Path(directory)/name).is_symlink(),'Source directory link')
  for name in names:
   path=Path(directory)/name;require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode) and not path.stat().st_mode&0o222,'Source member not sealed');observed.add(path.relative_to(root).as_posix())
 require(observed==set(files)|{'source_manifest.json'},'Source member scope changed')
 for rel,digest in files.items():
  require(not Path(rel).is_absolute() and '..' not in Path(rel).parts,'Unsafe member')
  path=root/rel;add(path,digest,metadata_only=not own and not rel.startswith('examples/configs/'));row=result['files'].get(str(path),result['external_files'].get(str(path)));proofs.append({k:row[k] for k in ('path','bytes','sha256','remote_bytes_rehashed')})
 require(not own or len(files)==7,'Own launcher must contain seven execution members')
 result['sources'][str(root)]={'manifest_sha256':expected,'members':proofs,'file_count':len(files),'all_members_rehashed':all(x['remote_bytes_rehashed'] for x in proofs),'readonly':True}
seal(p['launcher'],launcher_sha,True)
for source,expected in [(BASE/'boltz-lr0025-release-03/source','c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'),(BASE/'boltz-dbmol-pilot-release-02/source','600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed'),(BASE/'boltz-dbmol-storage-release-05/source','865303de06fc0a0bdf9297857c5bbc78c6ccd9ee05177e372a51869adc599bc3'),(BASE/'protenix-extended-release-03/source','da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562')]:seal(source,expected)
roots=[profiles[x][k] for x in profiles for k in ('root','control')]+[BASE/'protenix-extended-03']
ids=[]
for root in roots:
 if not root.exists():result['roots'][str(root)]={'exists':False,'jobs':{}};continue
 require(root.is_dir() and not root.is_symlink(),'Study/control root linked')
 jobs=json.loads((root/'jobs.json').read_bytes()) if (root/'jobs.json').is_file() else {}
 result['roots'][str(root)]={'exists':True,'jobs':jobs,'jobs_file_present':(root/'jobs.json').is_file(),'saved_prefixes':{}}
 for path in root.glob('*.json'):add(path)
 for directory in root.iterdir():
  if not directory.is_dir() or directory.is_symlink():continue
  for path in directory.glob('*.json'):add(path)
  for path in directory.glob('storage-*/*.json'):add(path)
  sub=directory/'submission.json'
  if sub.is_file():
   value=json.loads(sub.read_bytes());job=value.get('job_id')
   if job:ids.append(job)
   for pattern in ('*.txt','stdout.*.log','stderr.*.log','payload/*.xml','payload/*.log','payload/*.json'):
    for path in directory.glob(pattern):add(path)
   study=directory/'payload/study'
   if study.is_dir():
    for rel in ('assessment.json','summary.json','case-0/report.json','driver-invocation.json'):
     path=study/rel
     if path.is_file():add(path)
    counts={}
    for field in ('states','updates'):
     path=study/'case-0'/(field+'.jsonl')
     if not path.is_file():continue
     count=0;last=None;tail=False
     with path.open('rb') as stream:
      for line in stream:
       require(time.monotonic()-begin<210,'Prefix read expired')
       if not line.endswith(b'\n'):tail=True;break
       row=json.loads(line);require(row['step']==count,'Noncontiguous committed saved prefix');last={'step':row['step'],'global_step':row.get('global_step')};count+=1
     counts[field]={'committed_records':count,'last':last,'uncommitted_tail_excluded':tail}
    if counts:result['roots'][str(root)]['saved_prefixes'][directory.name]=counts
def walk(value):
 if isinstance(value,dict):
  if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',value['sha256']):
   path=Path(value['path'])
   if path.is_absolute() and allowed(path):add(path,value['sha256'])
  for x in value.values():walk(x)
 elif isinstance(value,list):
  for x in value:walk(x)
while queue:
 key=queue.pop()
 if key in seen:continue
 seen.add(key)
 data=base64.b64decode(result['files'][key]['base64']);value=json.loads(data)
 if key==PARENT_FRAME_INVENTORY:
  require(len(data)==2549313 and hashlib.sha256(data).hexdigest()==PARENT_FRAME_SHA
   and value['status']=='passed' and value['native_job']=='29792670' and value['n_atoms']==25
   and isinstance(value['frames'],list) and len(value['frames'])==10001,'Reviewed parent frame-inventory boundary changed')
  result['opaque_traversal_boundaries'].append({'path':key,'bytes':len(data),'sha256':PARENT_FRAME_SHA,
   'inventory_bytes_freshly_rehashed_and_downloaded':True,'omitted_child_frame_references':10001,
   'child_probability_frames_freshly_rehashed_by_this_collection':False,
   'scope':'Only this exact prior completed-parent inventory is opaque for metadata-CPU collection. No original/current CPU or native certificate proof is omitted; no fresh all-parent-frame audit is claimed.'})
  continue
 walk(value)
for job in dict.fromkeys(ids):
 require(re.fullmatch('[0-9]+',job) and len(result['scheduler'])<64 and time.monotonic()-begin<330,'Malformed/excess actual job IDs or expired scheduler phase')
 q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=15)
 result['scheduler'][job]={'args':q.args,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
result.update(downloaded_metadata_bytes=downloaded,remote_bytes_rehashed=hashed,scientific_or_production_pass_claimed=False,collector_elapsed_seconds=time.monotonic()-begin)
print(json.dumps(result,sort_keys=True,allow_nan=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('boltz30k', 'dbmol10k'), required=True)
    parser.add_argument('--launcher-manifest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import re
    assert re.fullmatch('[0-9a-f]{64}', args.launcher_manifest)
    out = args.output.resolve()
    assert out.is_relative_to(REPO / 'results/hpc')
    out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(REMOTE.encode()).hexdigest()
    bootstrap = "import hashlib,sys; raw=sys.stdin.buffer.read(131073); assert len(raw)<=131072 and hashlib.sha256(raw).hexdigest()==sys.argv[1]; sys.argv=['-c',*sys.argv[2:]]; exec(compile(raw,'<read-only-adoption-collector>','exec'),{'__name__':'__main__'})"
    remote = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', bootstrap,
              digest, args.profile, args.launcher_manifest]
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(' '.join(shlex.quote(x) for x in remote))]
    (out / 'started.json').write_text(json.dumps(dict(status='unknown', utc=datetime.now(timezone.utc).isoformat(),
        args=argv, collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        remote_sha256=digest, profile=args.profile, launcher_manifest_sha256=args.launcher_manifest,
        remote_mutations=False), indent=2) + '\n')
    try:
        with (out / 'stdout.bin').open('xb') as stdout, (out / 'stderr.bin').open('xb') as stderr:
            result = subprocess.run(argv, input=REMOTE.encode(), stdout=stdout, stderr=stderr, timeout=600)
    except BaseException as error:
        (out / 'failed-unknown.json').write_text(json.dumps(dict(error=repr(error), remote_mutations=False)) + '\n')
        raise
    stdout, stderr = (out / 'stdout.bin').read_bytes(), (out / 'stderr.bin').read_bytes()
    (out / 'result.json').write_text(json.dumps(dict(returncode=result.returncode, remote_mutations=False,
        stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_sha256=hashlib.sha256(stderr).hexdigest()), indent=2) + '\n')
    assert result.returncode == 0, stderr.decode(errors='replace')[-2000:]
    metadata = json.loads(stdout)
    for index, (remote_path, value) in enumerate(metadata['files'].items()):
        data = __import__('base64').b64decode(value.pop('base64'), validate=True)
        assert len(data) == value['bytes'] and hashlib.sha256(data).hexdigest() == value['sha256']
        relative = 'files/' + str(index).zfill(4) + '/' + PurePosixPath(remote_path).name
        path = out / relative; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
        value['local_path'] = relative
    with (out / 'metadata.json').open('x', encoding='utf8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status='collected_metadata_only', profile=args.profile,
        files=len(metadata['files']), external_references=len(metadata['external_files']),
        scientific_or_production_pass_claimed=False)))


if __name__ == '__main__': main()
