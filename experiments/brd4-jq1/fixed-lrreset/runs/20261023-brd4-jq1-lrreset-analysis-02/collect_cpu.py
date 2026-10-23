"""Independent byte/XML/scheduler/submission audit of this allocated analysis CPU."""
import base64,hashlib,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import base64,hashlib,json,subprocess
from pathlib import Path
r=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/lrreset-analysis-02')
s=r.with_name('lrreset-analysis-release-02')/'source';files={}
def add(p):
 n=('analysis/'+p.relative_to(r).as_posix()) if p.is_relative_to(r) else ('source/'+p.relative_to(s).as_posix())
 if n not in files:
  d=p.read_bytes();assert len(d)<10000000;files[n]={'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'base64':base64.b64encode(d).decode()}
for name in ['source-staging.json','jobs.json','cpu-certificate.json','analysis-dispatch-started.json','after-cpu/storage-dispatch/gate.json']:add(r/name)
for p in r.rglob('submission.json'):
 if p.parent.name in ['cache','payload']:continue
 add(p)
 for q in p.parent.glob('scheduler-*.txt'):add(q)
 sub=json.loads(p.read_bytes())
 if p.parent.name in ['cpu','after-cpu']:
  add(p.parent/('stdout.'+sub['job_id']+'.log'));add(p.parent/('stderr.'+sub['job_id']+'.log'))
for p in r.rglob('storage-*/gate.json'):add(p)
cpu=json.loads((r/'cpu-certificate.json').read_bytes())
for proof in cpu['proofs']:
 p=Path(proof['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['sha256'];add(p)
for p in [s/'source_manifest.json',s/'analysis-input-pins.json']:add(p)
jobs=json.loads((r/'jobs.json').read_bytes());ids=[jobs['cpu'],jobs['after_cpu'],*jobs['diagnostics'].values(),*jobs['diagnostic_captures'].values(),*jobs['collectors'].values(),jobs['active_interim'],jobs['diagnostic_interim'],jobs['final'],'29792672']
q=subprocess.run(['bjobs','-l',*ids],capture_output=True,text=True,timeout=30)
print(json.dumps({'files':files,'actual_scheduler':q.stdout+q.stderr,'scheduler_returncode':q.returncode}))
'''

def main():
    out=REPO/'results/hpc'/HERE.name/'cpu-evidence-01';out.mkdir(exist_ok=False)
    q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"],input=REMOTE.encode(),capture_output=True,timeout=60)
    (out/'ssh-stderr.bin').write_bytes(q.stderr);assert q.returncode==0,q.stderr.decode()[-2000:]
    j=json.loads(q.stdout);files=j.pop('files')
    for n,p in files.items():
        d=base64.b64decode(p.pop('base64'));assert len(d)==p['bytes'] and hashlib.sha256(d).hexdigest()==p['sha256']
        f=out/n;assert f.resolve().is_relative_to(out.resolve());f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(d)
    (out/'metadata.json').write_text(json.dumps({**j,'files':files},indent=2)+'\n')
    read=lambda n:json.loads((out/n).read_bytes())
    cpu=read('analysis/cpu-certificate.json');manifest=json.loads((HERE/'package.json').read_bytes())['manifest_sha256']
    assert cpu['status']=='passed' and cpu['tests']==283 and cpu['suite']=='jq1-lrreset-saved-report-contract-v1' and cpu['source_manifest_sha256']==manifest
    assert hashlib.sha256((out/'source/source_manifest.json').read_bytes()).hexdigest()==manifest
    x=ET.parse(out/'analysis/cpu/payload/report-tests.xml');cases=list(x.iter('testcase'))
    assert len(cases)==283 and not any(list(x.iter(k)) for k in ['failure','error','skipped'])
    modules={k:sum(k in c.get('classname','').split('.') for c in cases) for k in cpu['tests_by_module']}
    assert modules==cpu['tests_by_module'] and len(modules)==8 and sum(modules.values())==283
    text=(out/'analysis/cpu/payload/report-tests.log').read_text();assert 'collected 283 items' in text and '283 passed' in text
    for proof in cpu['proofs']:
        key='analysis/'+proof['path'].split('/lrreset-analysis-02/',1)[1];assert files[key]['sha256']==proof['sha256']
    def done(job,raw,queue):
        c=re.sub(r'\s+','',raw);assert re.findall(r'Job<([^>]+)>',c)==[job] and re.findall(r'Status<([^>]+)>',c)==['DONE'] and re.findall(r'Queue<([^>]+)>',c)==[queue]
    schedules=[out/n for n in files if re.fullmatch(r'analysis/cpu/scheduler-final-\d+\.txt',n)]
    assert len(schedules)==1;done(cpu['job_id'],schedules[0].read_text(),'hpc')
    jobs=read('analysis/jobs.json')
    for parent in ['cpu','after-cpu']:
        sub=read('analysis/'+parent+'/submission.json');job=sub['job_id'];raw=(out/('analysis/'+parent+'/stdout.'+job+'.log')).read_text()
        assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',raw,re.M)==[job] and 'Successfully completed.' in raw
        assert 'Exited with exit' not in raw
    assert set(jobs['diagnostics'])==set(jobs['diagnostic_captures'])=={'boltz2'} and set(jobs['collectors'])=={'protenix','boltz2','boltz2_chemistry'}
    for n in files:
        if not n.endswith('/submission.json'):continue
        sub=read(n);assert sub['outcome']=='submitted' and sub['args'][0]=='bsub'
        phase=sub['phase'];args=sub['args'];queue=args[args.index('-q')+1]
        assert queue==('gpul40s' if phase=='diagnostic' else 'hpc')
        assert phase not in ['run','train'] and (phase!='diagnostic' or sub['arm']=='boltz2')
    for n in files:
        if not n.endswith('/gate.json'):continue
        gate=read(n);assert gate['status']=='passed' and gate['reserve_bytes']==4294967296 and gate['file_reserve']==100000
        assert gate['work3']['available_bytes_conservative']>=gate['work3']['required_bytes']
    value={'status':'passed','source_manifest':manifest,'cpu_job':cpu['job_id'],'cases':283,'tests_by_module':modules,'files_verified':len(files),'jobs':jobs,'cpu_certificate_sha256':hashlib.sha256((out/'analysis/cpu-certificate.json').read_bytes()).hexdigest(),'scope':'Stdlib byte/XML/JSON/scheduler proof audit only; no diagnostic finding or report completion inferred'}
    (out/'verification.json').write_text(json.dumps(value,indent=2)+'\n');(HERE/'cpu-verification.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
if __name__=='__main__':main()
