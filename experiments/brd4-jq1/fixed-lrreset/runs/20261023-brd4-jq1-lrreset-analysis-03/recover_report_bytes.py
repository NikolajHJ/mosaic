"""Authenticated byte-range recovery after a verified stopped transfer only.

Retain the old partial and timeout receipt, authenticate its exact remote prefix,
download four disjoint byte ranges into fresh files, verify every range and the
whole original archive SHA before atomic local replacement. No scientific work.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,shlex,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    name=sys.argv[1];folder=REPO/'results/hpc'/HERE.name
    old=folder/(name+'.tar.gz');receipt=json.loads((folder/('delivery-'+name)/'archive-receipt.json').read_bytes())
    previous=json.loads((folder/(name+'-stream-result.json')).read_bytes())
    assert previous['outcome']=='timeout_stopped_partial_preserved' and 0<old.stat().st_size<receipt['bytes']
    work=folder/(name+'-byte-recovery-01');work.mkdir(exist_ok=False)
    prefix={'bytes':old.stat().st_size,'sha256':sha(old)};backup=work/'original-failed-prefix.bin';shutil.copyfile(old,backup);assert sha(backup)==prefix['sha256']
    code='''import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);total=int(sys.argv[2]);full=sys.argv[3];n=int(sys.argv[4]);prefix=sys.argv[5]
assert p.is_file() and not p.is_symlink() and p.stat().st_size==total
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==full
with p.open('rb') as f:assert hashlib.sha256(f.read(n)).hexdigest()==prefix
parts=[];size=(total-n+3)//4
with p.open('rb') as f:
 for i in range(4):
  start=n+i*size;length=min(size,total-start)
  if length<=0:continue
  f.seek(start);data=f.read(length);assert len(data)==length
  parts.append({'index':i,'start':start,'bytes':length,'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'path':str(p),'prefix':{'bytes':n,'sha256':prefix},'parts':parts,'archive_sha256':full}))
'''
    cmd=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',code,receipt['path'],str(receipt['bytes']),receipt['sha256'],str(prefix['bytes']),prefix['sha256']])
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',cmd]
    q=subprocess.run(args,capture_output=True,timeout=120);(work/'planner-stderr.bin').write_bytes(q.stderr);assert q.returncode==0,q.stderr.decode()[-1000:]
    plan=json.loads(q.stdout);(work/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    stream='''import hashlib,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);start=int(sys.argv[2]);length=int(sys.argv[3]);checksum=sys.argv[4]
assert p.is_file() and not p.is_symlink()
with p.open('rb') as f:
 f.seek(start);data=f.read(length);assert len(data)==length and hashlib.sha256(data).hexdigest()==checksum
 sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
'''
    def transfer(part):
        label='part-'+str(part['index']);out=work/(label+'.bin')
        cmd=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',stream,receipt['path'],str(part['start']),str(part['bytes']),part['sha256']])
        args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',cmd]
        record={'part':part,'args':args,'outcome':'unknown','utc':datetime.now(timezone.utc).isoformat()};(work/(label+'-started.json')).write_text(json.dumps(record,indent=2)+'\n')
        try:
            with out.open('xb') as f:
                p=subprocess.Popen(args,stdout=f,stderr=subprocess.PIPE);record['pid']=p.pid
                try:_,err=p.communicate(timeout=1800)
                except subprocess.TimeoutExpired:p.kill();_,err=p.communicate(timeout=20);record['outcome']='timeout_stopped_partial_preserved';raise
                finally:(work/(label+'-stderr.bin')).write_bytes(err)
            assert p.returncode==0 and out.stat().st_size==part['bytes'] and sha(out)==part['sha256'];record['outcome']='complete_range_verified'
        finally:(work/(label+'-result.json')).write_text(json.dumps(record,indent=2)+'\n')
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(transfer,plan['parts']))
    assembled=work/'reassembled.tar.gz'
    with assembled.open('xb') as f:
        for p in [backup,*[work/('part-'+str(p['index'])+'.bin') for p in plan['parts']]]:
            with p.open('rb') as src:shutil.copyfileobj(src,f,1048576)
    assert assembled.stat().st_size==receipt['bytes'] and sha(assembled)==receipt['sha256'] and old.stat().st_size==prefix['bytes'] and sha(old)==prefix['sha256']
    assembled.replace(old)
    subprocess.run([sys.executable,str(REPO/'scripts/evidence/verify_hpc_evidence.py'),str(old),receipt['sha256']],check=True)
    record={'status':'passed','prefix':prefix,'parts':results,'archive':receipt,'original_prefix_preserved':str(backup),'scope':'Exact original archive byte recovery/safe every-member verification only; no scientific mutation'}
    (work/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':'passed','archive_bytes':receipt['bytes'],'archive_sha256':receipt['sha256']}))
if __name__=='__main__':main()
