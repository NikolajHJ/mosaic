"""Back up then retire an exact inactive tail of the previously approved plan.

Only non-bound, non-twentieth old LR-reset trial NPZ copies may be selected.
No numerical contents are interpreted; all values remain exact in laptop tar.
"""
import argparse,ast,base64,gzip,hashlib,io,json
from pathlib import Path,PurePosixPath
import shlex,shutil,subprocess,tarfile

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261027-repair-storage-01'
PREFIX='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04/'
ALLOWED=('run-protenix','run-boltz2','run-boltz2_chemistry')
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8',newline='\n') as stream:json.dump(v,stream,indent=2);stream.write('\n')

def prepare():
    original=LOCAL/'remainder-candidates.json';r=json.loads(original.read_bytes());selected=[]
    protected=set(r['protected_paths'])
    for row in r['files']:
        p=PurePosixPath(row['path'])
        if not str(p).startswith(PREFIX):continue
        rel=p.relative_to(PREFIX)
        if rel.parts[0] not in ALLOWED or rel.parts[1:4]!=('payload','study','case-0'):continue
        if rel.parts[4]!='trials' or p.suffix!='.npz' or not p.stem.startswith('step-'):continue
        step=int(p.stem.removeprefix('step-'))
        if step%20==0 or str(p) in protected:continue
        selected.append(row)
    assert selected and sum(r['bytes'] for r in selected)>80*1024**2
    write(LOCAL/'remainder-plan01.json',{'version':'exact_inactive_old_trial_tail-v1','files':selected,
        'protected_paths':sorted(protected),'origin_plan_sha256':sha(original),
        'current_studies_or_settings_changed':False,'preserve_twentieth':True})
    print(json.dumps({'files':len(selected),'bytes':sum(r['bytes'] for r in selected),'status':'new_backup_required_before_any_unlink'}))

COMMON=r'''
import base64,gzip,hashlib,io,json,os,re,stat,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01');root=base/'repair-remainder-retirement-01'
raw=base64.b64decode(PLAN,validate=True);assert hashlib.sha256(raw).hexdigest()==PIN
plan=json.loads(raw);assert plan['version']=='exact_inactive_old_trial_tail-v1'
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as stream:json.dump(v,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def current():
 source=base/'chemistry-production-release-04/source';pin='01e2de2a3413bd903590dd331c5cedf6b756e856a83df2b73ad65e02b9f9fceb'
 assert sha(source/'source_manifest.json')==pin
 m=json.loads((source/'source_manifest.json').read_bytes());assert len(m['files'])==17
 for name,digest in m['files'].items():assert sha(source/name)==digest
 sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'scripts'));import production_ops as op
 _,declared=op.own_plan(pin);protected=set(plan['protected_paths'])
 def bindings(v):
  if isinstance(v,dict):
   if {'path','bytes','sha256'}<=set(v):op.check_binding({k:v[k] for k in ('path','bytes','sha256')});protected.add(v['path'])
   for item in v.values():bindings(item)
  elif isinstance(v,list):
   for item in v:bindings(item)
 bindings(m);bindings(declared)
 for arm in ('protenix','boltz'):bindings(op.adoption(arm))
 assert not any(row['path'] in protected for row in plan['files'])
 return {'status':'original2176_and_current_bound_inputs_protected','protected_paths':len(protected),'source_manifest_sha256':pin}
def check(row):
 p=Path(row['path']);prefix=base/'restart-lrreset-04';rel=p.relative_to(prefix)
 assert rel.parts[0] in ('run-protenix','run-boltz2','run-boltz2_chemistry') and rel.parts[1:5]==('payload','study','case-0','trials')
 assert re.fullmatch(r'step-[0-9]+\.npz',p.name) and int(p.stem.removeprefix('step-'))%20!=0 and str(p) not in plan['protected_paths']
 for parent in (p,*p.parents):assert not parent.is_symlink()
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink]==row['identity']
 assert sha(p)==row['sha256'];return p
'''

def remote(phase):
    raw=(LOCAL/'remainder-plan01.json').read_bytes();plan=json.loads(raw)
    assert shutil.disk_usage(REPO).free>sum(r['bytes'] for r in plan['files'])+2*1024**3
    prefix='PLAN='+repr(base64.b64encode(raw).decode())+'\nPIN='+repr(hashlib.sha256(raw).hexdigest())+'\n'
    if phase=='backup':
        suffix=r'''
assert not os.path.lexists(root),'Existing/unknown operation, inspect instead of replay'
proof=current();root.mkdir();(root/'plan.json').write_bytes(raw);write(root/'started.json',{'plan_sha256':PIN,'current':proof})
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.bytes=0
 def write(self,data):sys.stdout.buffer.write(data);self.h.update(data);self.bytes+=len(data);return len(data)
 def flush(self):sys.stdout.buffer.flush()
writer=Writer();start=time.monotonic()
with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as zipped:
 with tarfile.open(fileobj=zipped,mode='w|') as archive:
  info=tarfile.TarInfo('plan.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
  for i,row in enumerate(plan['files']):
   assert time.monotonic()-start<1000,'Bounded backup deadline'
   p=check(row);info=tarfile.TarInfo(str(p).lstrip('/'));info.size=row['bytes'];info.mode=0o444
   with p.open('rb') as stream:archive.addfile(info,stream)
writer.flush();after=current()
receipt={'status':'complete_exact_old_trial_backup_originals_unchanged','plan_sha256':PIN,'archive_bytes':writer.bytes,'archive_sha256':writer.h.hexdigest(),'files':len(plan['files']),'after':after}
write(root/'backup-complete.json',receipt);sys.stderr.write(json.dumps(receipt)+'\n')
'''
    elif phase=='retire':
        backup=json.loads((LOCAL/'remainder-backup01/laptop-verification.json').read_bytes())
        assert sha(LOCAL/'remainder-backup01/data.tar.gz')==backup['archive_sha256']
        prefix+='BACKUP='+repr(backup)+'\n'
        suffix=r'''
assert root.is_dir() and sha(root/'plan.json')==PIN and not os.path.lexists(root/'retirement-started.json')
proof=current();prior=json.loads((root/'backup-complete.json').read_bytes())
assert prior['status']=='complete_exact_old_trial_backup_originals_unchanged' and prior['archive_sha256']==BACKUP['archive_sha256'] and prior['archive_bytes']==BACKUP['archive_bytes'] and BACKUP['status']=='all_laptop_archive_members_verified' and BACKUP['plan_sha256']==PIN
write(root/'retirement-started.json',{'current':proof,'laptop':BACKUP});removed=0;size=0;start=time.monotonic()
with (root/'journal.jsonl').open('x',encoding='utf8') as journal:
 for row in plan['files']:
  assert time.monotonic()-start<1000,'Bounded retirement deadline; inspect journal before recovery'
  p=check(row);journal.write(json.dumps({'event':'intent',**row})+'\n');journal.flush();os.fsync(journal.fileno())
  p.unlink();removed+=1;size+=row['bytes']
  journal.write(json.dumps({'event':'removed','path':str(p)})+'\n');journal.flush();os.fsync(journal.fileno())
after=current();receipt={'status':'exact_verified_inactive_trial_tail_retired','removed_files':removed,'removed_bytes':size,'journal_sha256':sha(root/'journal.jsonl'),'plan_sha256':PIN,'after':after,'current_results_changed':False}
write(root/'completion.json',receipt);print(json.dumps(receipt))
'''
    else:
        suffix=r'''
assert root.is_dir() and sha(root/'plan.json')==PIN
done=json.loads((root/'completion.json').read_bytes());assert done['status']=='exact_verified_inactive_trial_tail_retired' and done['plan_sha256']==PIN and done['journal_sha256']==sha(root/'journal.jsonl')
assert all(not os.path.lexists(row['path']) for row in plan['files'])
files=[]
for name in ('plan.json','started.json','backup-complete.json','retirement-started.json','journal.jsonl','completion.json'):
 p=root/name;data=p.read_bytes();files.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()})
print(json.dumps({'status':'complete_exact_retirement_audit','all_selected_absent':True,'current':current(),'files':files}))
'''
    program=prefix+COMMON+suffix;ast.parse(program)
    out=LOCAL/('remainder-'+phase+'01');out.mkdir(exist_ok=False);(out/'program.py').write_text(program,encoding='utf8',newline='\n')
    cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    target=out/('data.tar.gz' if phase=='backup' else 'stdout.json');receipt={'status':'unknown_inspect_exact_remote_receipts_before_recovery'}
    try:
        with target.open('xb') as output,(out/'stderr.txt').open('xb') as error:
            process=subprocess.run(cmd,input=program.encode(),stdout=output,stderr=error,timeout=1100)
        receipt['returncode']=process.returncode
        if process.returncode:raise RuntimeError((out/'stderr.txt').read_text(errors='replace')[-2000:])
        if phase=='backup':
            candidates=[json.loads(line) for line in (out/'stderr.txt').read_text().splitlines() if line.startswith('{')]
            matches=[r for r in candidates if r.get('status')=='complete_exact_old_trial_backup_originals_unchanged'];assert len(matches)==1
            proof=matches[0];assert proof['archive_bytes']==target.stat().st_size and proof['archive_sha256']==sha(target)
            expected={r['path'].lstrip('/'):r for r in plan['files']};seen=set()
            with tarfile.open(target,'r:gz') as tar:
                for member in tar:
                    assert member.isfile() and member.name not in seen;seen.add(member.name)
                    with tar.extractfile(member) as stream:
                        if member.name=='plan.json':assert stream.read()==raw
                        else:assert member.size==expected[member.name]['bytes'] and hashlib.file_digest(stream,'sha256').hexdigest()==expected[member.name]['sha256']
                assert seen==set(expected)|{'plan.json'}
            write(out/'laptop-verification.json',{'status':'all_laptop_archive_members_verified',**{k:proof[k] for k in ('archive_bytes','archive_sha256','plan_sha256','files')}})
            receipt['status']='complete_backup_every_member_verified_zero_unlinks'
        elif phase=='retire':receipt.update(json.loads(target.read_bytes()))
        else:
            result=json.loads(target.read_bytes());assert result['status']=='complete_exact_retirement_audit' and result['all_selected_absent']
            files={}
            for row in result['files']:
                data=base64.b64decode(row['base64'],validate=True);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'] and row['name'] in ('plan.json','started.json','backup-complete.json','retirement-started.json','journal.jsonl','completion.json')
                assert row['name'] not in files;files[row['name']]=data;(out/row['name']).write_bytes(data)
            assert files['plan.json']==raw
            journal=[json.loads(line) for line in files['journal.jsonl'].splitlines()]
            assert len(journal)==2*len(plan['files'])
            for i,row in enumerate(plan['files']):
                assert journal[2*i]=={'event':'intent',**row} and journal[2*i+1]=={'event':'removed','path':row['path']}
            receipt.update(status='independently_reconciled_every_unlink_to_exact_verified_laptop_member',removed_files=len(plan['files']),removed_bytes=sum(row['bytes'] for row in plan['files']))
        print(json.dumps(receipt,indent=2))
    finally:write(out/'receipt.json',receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','backup','retire','audit'));a=p.parse_args()
    if a.phase=='prepare':prepare()
    else:remote(a.phase)
