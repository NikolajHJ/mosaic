"""Finish two explicitly stopped byte ranges using captured SSH stdout.

Preserves all earlier transfer receipts/ranges and never restarts a scientific
job. Only exact range SHA plus the original full archive SHA permits assembly.
"""
import hashlib,json,shlex,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    name=sys.argv[1];folder=REPO/'results/hpc'/HERE.name;work=folder/(name+'-byte-recovery-01');retry=folder/(name+'-byte-recovery-03');retry.mkdir(exist_ok=False)
    assert (work/'intentional-stalled-transfer-stop.json').exists()
    plan=json.loads((work/'plan.json').read_bytes());receipt=json.loads((folder/('delivery-'+name)/'archive-receipt.json').read_bytes())
    old=folder/(name+'.tar.gz');prefix=work/'original-failed-prefix.bin'
    assert old.stat().st_size==plan['prefix']['bytes'] and sha(old)==sha(prefix)==plan['prefix']['sha256']
    sources=[prefix]
    code='''import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1]);assert p.is_file() and not p.is_symlink() and p.stat().st_size==90878067
with p.open('rb') as f:
 f.seek(int(sys.argv[2]));data=f.read(int(sys.argv[3]));assert len(data)==int(sys.argv[3]) and hashlib.sha256(data).hexdigest()==sys.argv[4]
 n=int(sys.argv[5]);assert hashlib.sha256(data[:n]).hexdigest()==sys.argv[6]
 sys.stdout.buffer.write(data[n:]);sys.stdout.buffer.flush()
'''
    for part in plan['parts']:
        i=part['index'];prior=work/('part-'+str(i)+'.bin')
        if i<2:
            assert prior.stat().st_size==part['bytes'] and sha(prior)==part['sha256'];sources.append(prior);continue
        assert 0<=prior.stat().st_size<part['bytes'] and (work/('part-'+str(i)+'-result.json')).exists()
        prior_size=prior.stat().st_size;prior_sha=sha(prior)
        out=retry/('part-'+str(i)+'.bin');marker=retry/('part-'+str(i)+'-started.json')
        command=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',code,receipt['path'],str(part['start']),str(part['bytes']),part['sha256'],str(prior_size),prior_sha])
        args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',command]
        marker.write_text(json.dumps({'args':args,'part':part,'prior_prefix':{'bytes':prior_size,'sha256':prior_sha},'outcome':'unknown'},indent=2)+'\n')
        try:q=subprocess.run(args,capture_output=True,timeout=180)
        except subprocess.TimeoutExpired as e:
            out.write_bytes(e.output or b'');(retry/('part-'+str(i)+'-timeout.json')).write_text(json.dumps({'outcome':'timeout_stopped_partial_preserved','bytes':out.stat().st_size})+'\n');raise
        with out.open('xb') as f:
            with prior.open('rb') as p:shutil.copyfileobj(p,f)
            f.write(q.stdout)
        (retry/('part-'+str(i)+'-stderr.bin')).write_bytes(q.stderr)
        record={'returncode':q.returncode,'bytes':out.stat().st_size,'sha256':sha(out),'expected':part};(retry/('part-'+str(i)+'-result.json')).write_text(json.dumps(record,indent=2)+'\n')
        assert q.returncode==0 and record['bytes']==part['bytes'] and record['sha256']==part['sha256'];sources.append(out)
    assembled=retry/'reassembled.tar.gz'
    with assembled.open('xb') as f:
        for p in sources:
            with p.open('rb') as src:shutil.copyfileobj(src,f,1048576)
    assert assembled.stat().st_size==receipt['bytes'] and sha(assembled)==receipt['sha256'] and sha(old)==plan['prefix']['sha256']
    assembled.replace(old)
    subprocess.run([sys.executable,str(REPO/'scripts/evidence/verify_hpc_evidence.py'),str(old),receipt['sha256']],check=True)
    value={'status':'passed','original_archive':receipt,'prefix_preserved':str(prefix),'all_ranges_verified':True,'safe_extraction_every_file_verified':True,'scope':'Byte recovery only; original scientific runs/results unchanged'}
    (retry/'verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
