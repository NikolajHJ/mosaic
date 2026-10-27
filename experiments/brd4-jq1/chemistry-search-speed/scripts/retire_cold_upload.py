"""Offload one inspected inactive source upload, preserving all other bytes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

REPO=Path(__file__).resolve().parents[4]
OUT=REPO/'results/hpc/20261027-repair-storage-03'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def ssh(program,output):
    (output.parent/(output.name+'.program.py')).write_text(program,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    record={'argv':argv,'started_utc':datetime.now(timezone.utc).isoformat(),
            'program_sha256':hashlib.sha256(program.encode()).hexdigest(),'status':'unknown'}
    try:
        with output.open('xb') as stdout,(output.parent/(output.name+'.stderr')).open('xb') as stderr:
            run=subprocess.run(argv,input=program.encode(),stdout=stdout,stderr=stderr,timeout=180)
        record.update(status='returned',returncode=run.returncode)
    finally:
        record.update(ended_utc=datetime.now(timezone.utc).isoformat(),output_sha256=sha(output))
        (output.parent/(output.name+'.receipt.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
    assert run.returncode==0,(output.parent/(output.name+'.stderr')).read_text()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('backup','retire'));a=p.parse_args()
    inventory=OUT/'cold-source-index/raw.json';index=json.loads(inventory.read_bytes());archive=index['archive']
    assert archive['path']=='/work3/s234818/mosaic/releases/20261019-brd4-chemistry-02/source.tar.gz'
    assert not index['runtime_manifest_references'] and len(index['members'])==351
    backup=OUT/'backup-source.tar.gz';verification=OUT/'backup-verification.json'
    if a.phase=='backup':
        program='ROW='+repr(archive)+'\n'+r'''
from pathlib import Path
import hashlib,sys
p=Path(ROW['path']);s=p.lstat()
assert [s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size]==ROW['identity'] and not p.is_symlink()
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==ROW['sha256']
with p.open('rb') as f:
 while chunk:=f.read(1024*1024):sys.stdout.buffer.write(chunk)
sys.stdout.buffer.flush()
'''
        ssh(program,backup)
        assert backup.stat().st_size==archive['bytes'] and sha(backup)==archive['sha256']
        members={}
        with tarfile.open(backup) as t:
            for row in t:
                assert not Path(row.name).is_absolute() and '..' not in Path(row.name).parts and '\\' not in row.name
                if row.isdir():continue
                assert row.isfile() and row.name not in members
                with t.extractfile(row) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                members[row.name]={'bytes':row.size,'sha256':digest}
        assert members==index['members']
        result={'status':'laptop_archive_and_every_member_verified','archive':archive,
                'backup':str(backup),'members':len(members),'index_sha256':sha(inventory),'numerical_work':False}
        with verification.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result,indent=2));return
    v=json.loads(verification.read_bytes())
    assert v['status']=='laptop_archive_and_every_member_verified' and v['archive']==archive
    assert backup.stat().st_size==archive['bytes'] and sha(backup)==archive['sha256']
    program='ROW='+repr(archive)+'\nBACKUP='+repr({'path':str(backup),'sha256':sha(backup),'verification_sha256':sha(verification)})+'\n'+r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,stat
base=Path('/work3/s234818/mosaic');p=Path(ROW['path'])
assert p==base/'releases/20261019-brd4-chemistry-02/source.tar.gz'
assert all(not q.is_symlink() for q in (p,*p.parents))
s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
assert [s.st_dev,s.st_ino,s.st_mtime_ns,s.st_size]==ROW['identity']
def sha(x):
 with x.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(p)==ROW['sha256'] and BACKUP['sha256']==ROW['sha256']
operation=base/'runs/20261022-brd4-jq1-lrboost-analysis-01/storage-upload-retirement-01'
operation.mkdir(exist_ok=False)
others={str(q):{'bytes':q.stat().st_size,'sha256':sha(q)} for q in p.parent.iterdir() if q!=p and q.is_file()}
def write(name,value):
 with (operation/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
write('started.json',{'archive':ROW,'backup':BACKUP,'other_files':others,'started_utc':datetime.now(timezone.utc).isoformat()})
assert all(sha(Path(q))==r['sha256'] for q,r in others.items())
p.unlink()
assert not p.exists() and all(sha(Path(q))==r['sha256'] for q,r in others.items())
result={'status':'one_exact_backed_up_inactive_upload_removed','removed_files':1,'logical_bytes':ROW['bytes'],
 'archive':ROW,'backup':BACKUP,'preserved_other_files':others,'completed_utc':datetime.now(timezone.utc).isoformat(),
 'scientific_results_or_settings_changed':False,'quota_reservation':False}
write('completion.json',result);print(json.dumps(result))
'''
    output=OUT/'retirement.json';ssh(program,output);print(output.read_text())

if __name__=='__main__':main()
