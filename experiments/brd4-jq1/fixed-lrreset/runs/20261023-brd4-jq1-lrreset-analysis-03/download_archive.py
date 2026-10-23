"""Fresh direct SSH stream; refuse any prior output/unknown marker. Stdlib byte only."""
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def main():
    name=sys.argv[1];folder=REPO/'results/hpc'/HERE.name
    receipt=json.loads((folder/('delivery-'+name)/'archive-receipt.json').read_bytes())
    assert receipt['path']=='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/lrreset-analysis-03/'+name+'.tar.gz'
    out=folder/(name+'.tar.gz');marker=folder/(name+'-stream-started.json');result=folder/(name+'-stream-result.json')
    assert not any(p.exists() for p in [out,marker,result]),'Inspect prior outcomes; never blindly resume/repeat'
    code="""import hashlib,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);assert p.is_file() and not p.is_symlink()
with p.open('rb') as f:
 assert os.fstat(f.fileno()).st_size==int(sys.argv[2]) and hashlib.file_digest(f,'sha256').hexdigest()==sys.argv[3]
 f.seek(0)
 for d in iter(lambda:f.read(1048576),b''):sys.stdout.buffer.write(d)
 sys.stdout.buffer.flush()
"""
    cmd=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',code,receipt['path'],str(receipt['bytes']),receipt['sha256']])
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',cmd]
    record={'utc':datetime.now(timezone.utc).isoformat(),'receipt':receipt,'args':args,'outcome':'unknown'};marker.write_text(json.dumps(record,indent=2)+'\n')
    try:
        with out.open('xb') as f:
            p=subprocess.Popen(args,stdout=f,stderr=subprocess.PIPE);record['pid']=p.pid
            try:_,err=p.communicate(timeout=180)
            except subprocess.TimeoutExpired:p.kill();_,err=p.communicate(timeout=20);record['outcome']='timeout_stopped_partial_preserved';raise
            finally:(folder/(name+'-stream-stderr.bin')).write_bytes(err)
        assert p.returncode==0 and out.stat().st_size==receipt['bytes']
        with out.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
        record['outcome']='complete_bytes_verified'
        subprocess.run([sys.executable,str(REPO/'scripts/evidence/verify_hpc_evidence.py'),str(out),receipt['sha256']],check=True)
        record['outcome']='safe_extraction_every_member_verified'
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat();record['bytes']=out.stat().st_size if out.exists() else 0;result.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
if __name__=='__main__':main()
