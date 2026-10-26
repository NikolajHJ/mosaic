"""Read-only ordinary duplicate-container catalogue; never deletes or submits."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
PROGRAM=r'''
import hashlib,json,os,stat
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
roots=[base,Path('/work3/s234818/mosaic/releases')]
result={'status':'read_only_catalogue_not_retirement_authority','captured_utc':datetime.now(timezone.utc).isoformat(),'archives':[],'capacity_failure':{}}
for root in roots:
 assert root.is_dir() and not root.is_symlink()
 for parent,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[name for name in dirs if name not in ('source','study','payload','cache','tmp','node-tmp','models','envs') and not (Path(parent)/name).is_symlink()]
  for name in files:
   if not name.endswith(('.tar.gz','.tgz')):continue
   p=Path(parent)/name;s=p.lstat()
   if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or p.is_symlink():continue
   result['archives'].append({'path':str(p),'bytes':s.st_size,'identity':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink]})
failed=base/'protenix-endpoint-fix-04/cpu/current-chemistry-capacity-entry'
if failed.is_dir():
 for p in failed.rglob('*.json'):
  assert not p.is_symlink() and p.is_file() and p.stat().st_size<2*1024**2
  raw=p.read_bytes();result['capacity_failure'][str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
print(json.dumps(result))
'''

def main():
    assert REPO.name=='mosaic';ast.parse(PROGRAM)
    out=REPO/'results/hpc/20261027-repair-storage-01/inventory01';out.mkdir(parents=True,exist_ok=False)
    (out/'program.py').write_text(PROGRAM,encoding='utf8',newline='\n')
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
             'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'status':'unknown_read_only_outcome','program_sha256':hashlib.sha256(PROGRAM.encode()).hexdigest()}
    try:
        with (out/'raw.json').open('xb') as output,(out/'stderr.txt').open('xb') as error:
            process=subprocess.run(command,input=PROGRAM.encode(),stdout=output,stderr=error,timeout=120)
        receipt['returncode']=process.returncode
        if process.returncode:raise RuntimeError('Read-only connection failed; no retry')
        value=json.loads((out/'raw.json').read_bytes())
        for row in value['capacity_failure'].values():
            data=row['text'].encode();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        receipt.update(status='read_only_archive_catalogue_and_failure_bytes_verified',archives=len(value['archives']))
    finally:
        receipt['ended_utc']=datetime.now(timezone.utc).isoformat()
        receipt['raw_sha256']=hashlib.sha256((out/'raw.json').read_bytes()).hexdigest()
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
