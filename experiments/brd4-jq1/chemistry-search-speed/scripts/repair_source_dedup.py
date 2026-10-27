"""Physical duplicate consolidation; every sealed byte and pathname stays exact.

All operations are metadata/byte-only. Backup is fully verified on the laptop
before any inode replacement. No optimizer/test/model or scientific array import.
"""
import argparse,ast,base64,hashlib,json,shlex,subprocess,tarfile
from pathlib import Path,PurePosixPath

REPO=Path(__file__).resolve().parents[4]
OUT=REPO/'results/hpc/20261027-repair-storage-02'
COMMON=r'''
import base64,gzip,hashlib,io,json,os,re,stat,sys,time,tarfile
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'physical-source-dedup-01'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ident(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and not s.st_mode&0o222
 assert all(not q.is_symlink() for q in (p,*p.parents))
 return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink,stat.S_IMODE(s.st_mode)]
def verify_source(row):
 source=Path(row['path']);assert sha(source/'source_manifest.json')==row['manifest_sha256']
 m=json.loads((source/'source_manifest.json').read_bytes())
 for rel,digest in m['files'].items():
  part=Path(rel);assert not part.is_absolute() and '..' not in part.parts
  p=source/rel;assert all(not q.is_symlink() for q in (p,*p.parents))
  assert p.is_file() and sha(p)==digest
 return len(m['files'])
def protect():
 source=base/'chemistry-production-release-04/source';pin='01e2de2a3413bd903590dd331c5cedf6b756e856a83df2b73ad65e02b9f9fceb'
 assert sha(source/'source_manifest.json')==pin
 m=json.loads((source/'source_manifest.json').read_bytes());assert len(m['files'])==17
 for name,digest in m['files'].items():assert sha(source/name)==digest
 sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'scripts'));import production_ops as op
 _,declared=op.own_plan(pin);bindings={}
 def check(v):
  if isinstance(v,dict):
   if {'path','bytes','sha256'}<=set(v):
    row={k:v[k] for k in ('path','bytes','sha256')};op.check_binding(row);bindings[row['path']]=row
   for child in v.values():check(child)
  elif isinstance(v,list):
   for child in v:check(child)
 check(m);check(declared)
 for arm in ('protenix','boltz'):check(op.adoption(arm))
 return {'status':'all_current_bound_inputs_rehashed','source_manifest_sha256':pin,'bindings':list(bindings.values())}
'''

def program(phase):
    if phase=='inventory':
        tail=r'''
assert not os.path.lexists(root),'Existing unknown operation; inspect instead'
start=time.monotonic();sources=[];byhash={};skipped=[]
candidates=list(Path('/work3/s234818/mosaic/releases').glob('*/source'))+list(base.glob('*release*/source'))
for source in sorted(set(candidates)):
 assert time.monotonic()-start<120,'Bounded source inventory incomplete'
 p=source/'source_manifest.json'
 if not p.is_file() or p.is_symlink():continue
 m=json.loads(p.read_bytes());files=m.get('files')
 if not isinstance(files,dict) or not files or not all(isinstance(v,str) and re.fullmatch('[a-f0-9]{64}',v) for v in files.values()):continue
 row={'path':str(source),'manifest_sha256':sha(p),'files':len(files)}
 verify_source(row);sources.append(row)
 for rel,digest in files.items():
  p=source/rel;s=p.lstat()
  if s.st_size<65536 or s.st_nlink!=1 or s.st_uid!=os.getuid() or s.st_mode&0o222:continue
  record={'path':str(p),'sha256':digest,'bytes':s.st_size,'identity':ident(p),'source':str(source)}
  byhash.setdefault((digest,s.st_size,stat.S_IMODE(s.st_mode),s.st_dev),[]).append(record)
groups=[{'sha256':key[0],'bytes':key[1],'canonical':values[0],'duplicates':values[1:]} for key,values in sorted(byhash.items()) if len(values)>1]
plan={'version':'sealed-byte-preserving-physical-dedup-v1','sources':sources,'groups':groups,'protected':protect(),
 'reclaimable_physical_bytes':sum(v['bytes']*len(v['duplicates']) for v in groups),'scientific_bytes_changed':False,
 'original_paths_preserved':True,'captured_utc':datetime.now(timezone.utc).isoformat()}
assert groups;root.mkdir();write(root/'plan.json',plan);print(json.dumps({'plan':plan,'plan_sha256':sha(root/'plan.json')}))
'''
    else:
        packet=json.loads((OUT/'source-inventory01/stdout.json').read_bytes())
        tail='PLAN_PIN='+repr(packet['plan_sha256'])+'\n'+r'''
assert root.is_dir() and sha(root/'plan.json')==PLAN_PIN
plan=json.loads((root/'plan.json').read_bytes());assert plan['version']=='sealed-byte-preserving-physical-dedup-v1'
for row in plan['sources']:verify_source(row)
before=protect()
'''
        if phase=='backup':
            tail+=r'''
assert not os.path.lexists(root/'backup-started.json')
write(root/'backup-started.json',{'plan_sha256':PLAN_PIN,'utc':datetime.now(timezone.utc).isoformat()})
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.bytes=0
 def write(self,b):sys.stdout.buffer.write(b);self.h.update(b);self.bytes+=len(b);return len(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with gzip.GzipFile(fileobj=w,mode='wb',compresslevel=1,mtime=0) as z:
 with tarfile.open(fileobj=z,mode='w|') as tar:
  raw=(root/'plan.json').read_bytes();info=tarfile.TarInfo('plan.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
  for group in plan['groups']:
   for row in [group['canonical'],*group['duplicates']]:assert ident(Path(row['path']))==row['identity'] and sha(row['path'])==row['sha256']
   p=Path(group['canonical']['path']);info=tarfile.TarInfo('blobs/'+group['sha256']);info.size=group['bytes'];info.mode=0o444
   with p.open('rb') as f:tar.addfile(info,f)
w.flush();after=protect();assert after==before
receipt={'status':'complete_content_addressed_backup_originals_unchanged','plan_sha256':PLAN_PIN,'archive_bytes':w.bytes,'archive_sha256':w.h.hexdigest(),'groups':len(plan['groups'])}
write(root/'backup-complete.json',receipt);sys.stderr.write(json.dumps(receipt)+'\n')
'''
        elif phase=='consolidate':
            local=json.loads((OUT/'source-backup01/laptop-verification.json').read_bytes())
            tail+='LAPTOP='+repr(local)+'\n'+r'''
assert not os.path.lexists(root/'replacement-started.json')
prior=json.loads((root/'backup-complete.json').read_bytes())
assert LAPTOP['status']=='every_unique_blob_and_every_path_mapping_verified' and all(LAPTOP[k]==prior[k] for k in ('plan_sha256','archive_bytes','archive_sha256'))
# Authenticate EVERY planned initial identity/hash before replacing even one path.
for group in plan['groups']:
 for row in [group['canonical'],*group['duplicates']]:assert ident(Path(row['path']))==row['identity'] and sha(row['path'])==row['sha256']
write(root/'replacement-started.json',{'laptop':LAPTOP,'before':before});replaced=0;size=0
with (root/'journal.jsonl').open('x',encoding='utf8') as journal:
 def event(v):journal.write(json.dumps(v)+'\n');journal.flush();os.fsync(journal.fileno())
 for group in plan['groups']:
  canonical=Path(group['canonical']['path']);canonical_initial=ident(canonical)
  for row in group['duplicates']:
   p=Path(row['path']);assert ident(p)==row['identity'] and sha(p)==group['sha256']
   assert ident(canonical)[:4]==canonical_initial[:4] and sha(canonical)==group['sha256']
   temporary=root/('link-'+str(replaced));assert not os.path.lexists(temporary)
   directory_mode=stat.S_IMODE(p.parent.stat().st_mode)
   assert p.parent.stat().st_uid==os.getuid() and not p.parent.is_symlink()
   event({'event':'intent','file':row,'canonical':str(canonical),'parent_mode':directory_mode})
   os.link(canonical,temporary,follow_symlinks=False)
   try:
    p.parent.chmod(directory_mode|stat.S_IWUSR)
    assert ident(p)==row['identity'] and sha(p)==row['sha256']
    os.replace(temporary,p)
   finally:p.parent.chmod(directory_mode)
   assert sha(p)==row['sha256'] and p.stat().st_ino==canonical.stat().st_ino and stat.S_IMODE(p.parent.stat().st_mode)==directory_mode
   replaced+=1;size+=row['bytes'];event({'event':'replaced_by_identical_readonly_hardlink','path':str(p),'identity':ident(p)})
for row in plan['sources']:verify_source(row)
after=protect();assert after==before
receipt={'status':'all_original_paths_and_manifest_bytes_preserved_physical_duplicates_consolidated','replaced_paths':replaced,'logical_bytes_unchanged':True,'redundant_original_allocated_bytes':size,'plan_sha256':PLAN_PIN,'journal_sha256':sha(root/'journal.jsonl'),'sources_rehashed':len(plan['sources']),'after':after}
write(root/'completion.json',receipt);print(json.dumps(receipt))
'''
        elif phase=='audit':
            tail+=r'''
done=json.loads((root/'completion.json').read_bytes());assert done['plan_sha256']==PLAN_PIN and sha(root/'journal.jsonl')==done['journal_sha256']
for group in plan['groups']:
 canonical=Path(group['canonical']['path'])
 for row in group['duplicates']:
  p=Path(row['path']);assert sha(p)==row['sha256'] and p.stat().st_ino==canonical.stat().st_ino and p.stat().st_dev==canonical.stat().st_dev
files=[]
for name in ('plan.json','backup-started.json','backup-complete.json','replacement-started.json','journal.jsonl','completion.json'):
 raw=(root/name).read_bytes();files.append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()})
print(json.dumps({'status':'independent_all_source_paths_hashes_and_shared_inodes_verified','files':files,'protected':protect()}))
'''
    return COMMON+tail

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('inventory','backup','consolidate','audit'));args=p.parse_args()
    if args.phase=='consolidate':
        prior=json.loads((OUT/'source-backup01/laptop-verification.json').read_bytes())
        with (OUT/'source-backup01/data.tar.gz').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==prior['archive_sha256']
    code=program(args.phase);ast.parse(code);out=OUT/('source-'+args.phase+'01');out.mkdir(parents=True,exist_ok=False)
    (out/'program.py').write_text(code,encoding='utf8',newline='\n')
    cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    target=out/('data.tar.gz' if args.phase=='backup' else 'stdout.json');receipt={'status':'unknown_inspect_before_recovery'}
    try:
        with target.open('xb') as output,(out/'stderr.txt').open('xb') as error:r=subprocess.run(cmd,input=code.encode(),stdout=output,stderr=error,timeout=360)
        receipt['returncode']=r.returncode
        if r.returncode:raise RuntimeError((out/'stderr.txt').read_text(errors='replace')[-3000:])
        if args.phase=='backup':
            records=[json.loads(v) for v in (out/'stderr.txt').read_text().splitlines() if v.startswith('{')]
            proof=[v for v in records if v.get('status')=='complete_content_addressed_backup_originals_unchanged'];assert len(proof)==1;proof=proof[0]
            with target.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==proof['archive_sha256']
            assert target.stat().st_size==proof['archive_bytes']
            expected=json.loads((OUT/'source-inventory01/stdout.json').read_bytes());plan=expected['plan'];seen=set()
            with tarfile.open(target,'r:gz') as tar:
                for m in tar:
                    assert m.isfile() and m.name not in seen;seen.add(m.name)
                    with tar.extractfile(m) as f:
                        digest=hashlib.file_digest(f,'sha256').hexdigest()
                    if m.name=='plan.json':assert digest==expected['plan_sha256']
                    else:
                        match=[g for g in plan['groups'] if m.name=='blobs/'+g['sha256']];assert len(match)==1 and m.size==match[0]['bytes'] and digest==match[0]['sha256']
            assert seen=={'plan.json'}|{'blobs/'+g['sha256'] for g in plan['groups']}
            proof.update(status='every_unique_blob_and_every_path_mapping_verified',paths=sum(len(g['duplicates'])+1 for g in plan['groups']))
            (out/'laptop-verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8',newline='\n');receipt.update(proof)
        else:
            data=json.loads(target.read_bytes());receipt['status']=data.get('status','read_only_plan_prepared')
            if args.phase=='inventory':receipt.update(groups=len(data['plan']['groups']),bytes=data['plan']['reclaimable_physical_bytes'],sources=len(data['plan']['sources']))
            elif args.phase=='consolidate':receipt.update(replaced_paths=data['replaced_paths'],redundant_original_allocated_bytes=data['redundant_original_allocated_bytes'])
            else:
                for row in data['files']:
                    raw=base64.b64decode(row['base64'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'];(out/row['name']).write_bytes(raw)
    finally:
        receipt['program_sha256']=hashlib.sha256(code.encode()).hexdigest()
        with target.open('rb') as f:receipt['output_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
