"""Fresh compact byte/XML proof delivery and independent stdlib acceptance audit."""
import base64,hashlib,json,re,subprocess,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import base64,hashlib,json,subprocess
from pathlib import Path
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01');r=b/'protenix-extended-01';s=b/'protenix-extended-release-01/source';files={}
def add(p):
 n=('run/'+p.relative_to(r).as_posix()) if p.is_relative_to(r) else ('source/'+p.relative_to(s).as_posix())
 if n in files:return
 d=p.read_bytes();assert len(d)<20000000;files[n]={'path':str(p),'bytes':len(d),'sha256':hashlib.sha256(d).hexdigest(),'base64':base64.b64encode(d).decode()}
for n in ['jobs.json','source-staging.json','cpu-certificate.json','parent-adoption.json']:add(r/n)
for p in r.rglob('submission.json'):
 add(p)
 for q in p.parent.glob('scheduler-*.txt'):add(q)
 if p.parent.name in ['cpu','after-cpu']:
  job=json.loads(p.read_bytes())['job_id']
  for label in ['stdout','stderr']:add(p.parent/(label+'.'+job+'.log'))
for p in r.rglob('storage-*/gate.json'):add(p)
for proof in json.loads((r/'cpu-certificate.json').read_bytes())['proofs']:
 p=Path(proof['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==proof['sha256'];add(p)
for n in ['source_manifest.json','examples/configs/extended_random.json','examples/configs/extended_continuation.json']:add(s/n)
jobs=json.loads((r/'jobs.json').read_bytes());q=subprocess.run(['bjobs','-l',*jobs.values()],capture_output=True,text=True,timeout=30)
print('EVIDENCE_BEGIN');print(json.dumps({'files':files,'actual_scheduler':q.stdout+q.stderr,'scheduler_returncode':q.returncode}))
'''
def main():
    out=REPO/'results/hpc'/HERE.name/'cpu-evidence-01';out.mkdir(exist_ok=False)
    q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"],input=REMOTE.encode(),capture_output=True,timeout=60)
    (out/'stdout.bin').write_bytes(q.stdout);(out/'stderr.bin').write_bytes(q.stderr);assert q.returncode==0,q.stderr.decode()[-1500:]
    data=json.loads(q.stdout.split(b'EVIDENCE_BEGIN\n')[1]);files=data.pop('files')
    for n,p in files.items():
        blob=base64.b64decode(p.pop('base64'));assert len(blob)==p['bytes'] and hashlib.sha256(blob).hexdigest()==p['sha256'];target=out/n
        assert target.resolve().is_relative_to(out.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
    (out/'metadata.json').write_bytes((json.dumps({**data,'files':files},indent=2)+'\n').encode())
    read=lambda n:json.loads((out/n).read_bytes())
    package=json.loads((HERE/'package.json').read_bytes());manifest=package['manifest_sha256'];cpu=read('run/cpu-certificate.json');jobs=read('run/jobs.json')
    assert cpu['status']=='passed' and cpu['suite']=='protenix-serial50k-v1' and cpu['source_manifest_sha256']==manifest
    assert hashlib.sha256((out/'source/source_manifest.json').read_bytes()).hexdigest()==manifest
    for arm in ['random','continuation']:assert (out/('source/examples/configs/extended_'+arm+'.json')).read_bytes()==(HERE/(arm+'.json')).read_bytes()
    proof_values={p['path']:p['sha256'] for p in files.values()}
    for proof in cpu['proofs']:assert proof_values[proof['path']]==proof['sha256']
    groups={};new_count=0
    for group in ['hpc-contract','legacy','native-mapping','chemistry']:
        tree=ET.parse(out/('run/cpu/payload/'+group+'.xml'));cases=list(tree.iter('testcase'));assert not any(list(tree.iter(k)) for k in ['failure','error','skipped']);groups[group]=len(cases)
        log=(out/('run/cpu/payload/'+group+'.log')).read_text(encoding='utf-8');assert re.search(r'\b'+str(len(cases))+r'\s+passed\b',log)
        if group=='hpc-contract':
            new_count=sum('test_extended_protenix' in c.get('classname','').split('.') for c in cases);assert new_count==56 and groups[group]==500
    assert groups=={'hpc-contract':500,'legacy':239,'native-mapping':34,'chemistry':59} and cpu['cases']==sum(groups.values())==832
    final=[p for p in cpu['proofs'] if re.fullmatch(r'scheduler-final-\d+\.txt',Path(p['path']).name)];assert len(final)==1
    n='run/'+final[0]['path'].split('/protenix-extended-01/',1)[1];raw=re.sub(r'\s+','',(out/n).read_text())
    assert re.findall(r'Job<([^>]+)>',raw)==[jobs['cpu']] and re.findall(r'Status<([^>]+)>',raw)==['DONE'] and re.findall(r'Queue<([^>]+)>',raw)==['hpc']
    for parent in ['cpu','after-cpu']:
        sub=read('run/'+parent+'/submission.json');log=(out/('run/'+parent+'/stdout.'+sub['job_id']+'.log')).read_text(encoding='utf-8')
        assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',log,re.M)==[jobs[parent]] and 'Successfully completed.' in log and 'Exited with exit' not in log
    assert read('run/parent-adoption.json')['status']=='passed'
    for n in files:
        if n.endswith('/gate.json'):
            g=read(n);assert g['status']=='passed' and g['reserve_bytes']==4294967296 and g['file_reserve']==100000 and g['work3']['available_bytes_conservative']>=g['work3']['required_bytes']
    native={}
    for arm in ['random','continuation']:
        for phase in ['native','after-native']:
            parent=arm+'-'+phase+'-0000';sub=read('run/'+parent+'/submission.json');args=sub['args'];assert sub['outcome']=='submitted' and sub['job_id']==jobs[parent] and sub['phase']==phase and manifest in args
            if phase=='native':assert args[args.index('-gpu')+1]=='num=1:mode=exclusive_process' and args[args.index('-q')+1]=='gpul40s' and args[args.index('-W')+1]=='02:00'
            else:assert sub['dependency']=='ended('+jobs[arm+'-native-0000']+')'
        native[arm]=jobs[arm+'-native-0000']
    assert not any('-segment-' in n for n in jobs)
    # Authenticate the complete source archive and every source member locally as bytes only.
    arc=REPO/'results/hpc'/HERE.name/'source02/source.tar.gz';assert arc.stat().st_size==package['archive_bytes'] and hashlib.sha256(arc.read_bytes()).hexdigest()==package['archive_sha256']
    with tarfile.open(arc) as tar:
        expected=read('source/source_manifest.json')['files'];seen=set()
        for m in tar:
            assert m.isfile() and m.name.startswith('source/') and '..' not in Path(m.name).parts
            n=m.name.removeprefix('source/');assert n not in seen;seen.add(n);blob=tar.extractfile(m).read()
            if n=='source_manifest.json':assert hashlib.sha256(blob).hexdigest()==manifest
            else:assert hashlib.sha256(blob).hexdigest()==expected[n]
        assert seen==set(expected)|{'source_manifest.json'} and len(expected)==641
    value={'status':'passed','cpu_cases':832,'new_cases':new_count,'groups':groups,'files_verified':len(files),'source_manifest':manifest,'source_members_verified':641,'cpu_certificate_sha256':hashlib.sha256((out/'run/cpu-certificate.json').read_bytes()).hexdigest(),'jobs':jobs,'new_native_passed':False,'new_optimization_submitted':False,'scope':'Stdlib byte/XML/JSON/source/config/submission/scheduler/LSF/adoption/storage audit only; no local numerical work'}
    (HERE/'cpu-verification.json').write_bytes((json.dumps(value,indent=2)+'\n').encode());print(json.dumps(value))
if __name__=='__main__':main()
