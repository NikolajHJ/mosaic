"""Independent stdlib audit of allocated776 CPU cases, inputs, quota and actual jobs."""
import base64,hashlib,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import base64,hashlib,json,subprocess
from pathlib import Path
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01');r=b/'protenix-random-chem-01';s=b/'protenix-random-chem-release-01/source';files={}
def add(p):
 n=('run/'+p.relative_to(r).as_posix()) if p.is_relative_to(r) else ('source/'+p.relative_to(s).as_posix())
 if n in files:return
 d=p.read_bytes();assert len(d)<20000000;files[n]={'path':str(p),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'base64':base64.b64encode(d).decode()}
for name in ['jobs.json','source-staging.json','cpu-certificate.json']:add(r/name)
for p in r.rglob('submission.json'):
 add(p)
 for q in p.parent.glob('scheduler-*.txt'):add(q)
 if p.parent.name in ['cpu','after-cpu']:
  j=json.loads(p.read_bytes())['job_id']
  for label in ['stdout','stderr']:add(p.parent/(label+'.'+j+'.log'))
for p in r.rglob('storage-*/gate.json'):add(p)
for proof in json.loads((r/'cpu-certificate.json').read_bytes())['proofs']:
 p=Path(proof['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['sha256'];add(p)
for p in [s/'source_manifest.json',s/'examples/configs/protenix_random_delayed_chem_10000.json']:add(p)
jobs=json.loads((r/'jobs.json').read_bytes());ids=list(jobs.values())+['29796551','29792672']
q=subprocess.run(['bjobs','-l',*ids],capture_output=True,text=True,timeout=30)
print(json.dumps({'files':files,'actual_scheduler':q.stdout+q.stderr,'scheduler_returncode':q.returncode}))
'''
def main():
    out=REPO/'results/hpc'/HERE.name/'cpu-evidence-01';out.mkdir(exist_ok=False)
    q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"],input=REMOTE.encode(),capture_output=True,timeout=60)
    (out/'ssh-stderr.bin').write_bytes(q.stderr);assert q.returncode==0,q.stderr.decode()[-1000:]
    j=json.loads(q.stdout);files=j.pop('files')
    for n,p in files.items():
        data=base64.b64decode(p.pop('base64'));assert len(data)==p['bytes'] and hashlib.sha256(data).hexdigest()==p['sha256'];target=out/n
        assert target.resolve().is_relative_to(out.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (out/'metadata.json').write_text(json.dumps({**j,'files':files},indent=2)+'\n')
    read=lambda n:json.loads((out/n).read_bytes())
    package=json.loads((HERE/'package.json').read_bytes());manifest=package['manifest_sha256'];cpu=read('run/cpu-certificate.json');jobs=read('run/jobs.json')
    assert cpu['status']=='passed' and cpu['suite']=='protenix-random-delayed-chem-v1' and cpu['cases']==776 and cpu['source_manifest_sha256']==manifest
    assert hashlib.sha256((out/'source/source_manifest.json').read_bytes()).hexdigest()==manifest
    assert (out/'source/examples/configs/protenix_random_delayed_chem_10000.json').read_bytes()==(HERE/'config.json').read_bytes()
    for proof in cpu['proofs']:
        n='run/'+proof['path'].split('/protenix-random-chem-01/',1)[1];assert files[n]['sha256']==proof['sha256']
    expected={'hpc-contract':444,'legacy':239,'native-mapping':34,'chemistry':59}
    for group,n in expected.items():
        x=ET.parse(out/('run/cpu/payload/'+group+'.xml'));cases=list(x.iter('testcase'));assert len(cases)==n and not any(list(x.iter(k)) for k in ['failure','error','skipped'])
        text=(out/('run/cpu/payload/'+group+'.log')).read_text(encoding='utf-8');assert re.search(r'\b'+str(n)+r'\s+passed\b',text)
        if group=='hpc-contract':assert sum('test_random_chem' in c.get('classname','').split('.') for c in cases)==48
    def actual(raw,job,queue):
        compact=re.sub(r'\s+','',raw);assert re.findall(r'Job<([^>]+)>',compact)==[job] and re.findall(r'Status<([^>]+)>',compact)==['DONE'] and re.findall(r'Queue<([^>]+)>',compact)==[queue]
    schedule=[p for p in cpu['proofs'] if re.fullmatch('scheduler-final-\\d+\\.txt',Path(p['path']).name)];assert len(schedule)==1
    actual((out/('run/'+schedule[0]['path'].split('/protenix-random-chem-01/',1)[1])).read_text(),jobs['cpu'],'hpc')
    for parent,phase in [('cpu','cpu'),('after-cpu','after-cpu'),('native','validate'),('after-ready','after-ready')]:
        sub=read('run/'+parent+'/submission.json');a=sub['args'];assert sub['outcome']=='submitted' and sub['phase']==phase and a[0]=='bsub'
        assert a[a.index('-q')+1]==('gpul40s' if phase=='validate' else 'hpc')
        assert str(manifest) in a and ('/protenix-random-chem-release-01/source' in ' '.join(a))
        if phase=='validate':assert a[a.index('-gpu')+1]=='num=1:mode=exclusive_process' and a[a.index('-W')+1]=='02:00'
        if parent in ['cpu','after-cpu']:
            log=(out/('run/'+parent+'/stdout.'+sub['job_id']+'.log')).read_text(encoding='utf-8');assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',log,re.M)==[sub['job_id']] and 'Successfully completed.' in log
    for n in files:
        if n.endswith('/gate.json'):
            gate=read(n);assert gate['status']=='passed' and gate['reserve_bytes']==4294967296 and gate['file_reserve']==100000 and gate['work3']['available_bytes_conservative']>=gate['work3']['required_bytes']
    assert 'run' not in jobs
    value={'status':'passed','files_verified':len(files),'cpu_cases':776,'groups':expected,'manifest':manifest,'cpu_certificate_sha256':hashlib.sha256((out/'run/cpu-certificate.json').read_bytes()).hexdigest(),'jobs':jobs,'new_long_submitted':False,'new_native_passed':False,'scope':'Independent stdlib byte/XML/JSON/source/config/scheduler/LSF/fulladmission/actualsubmission audit only; no local model calculations'}
    (out/'verification.json').write_text(json.dumps(value,indent=2)+'\n');(HERE/'cpu-verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
