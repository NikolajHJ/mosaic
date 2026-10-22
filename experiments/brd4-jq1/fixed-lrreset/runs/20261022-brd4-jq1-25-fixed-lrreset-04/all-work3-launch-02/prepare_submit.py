"""One-shot stdlib packaging/staging of launcher-only adoption; never replay outcomes."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

OWNER=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[6]
TAG='20261022-brd4-jq1-25-fixed-lrreset-04'
OUTPUT=REPO/'results/hpc'/TAG/'all-work3-launch-02'

REMOTE=r'''
import ast,hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
r=b/'restart-lrreset-04'; control=r/'all-work3-launch-02'
release=b/'restart-lrreset-allwork3-launch-02';source=release/'source'
upload=b/'restart-lrreset-allwork3-launch-02.tar.gz'
expected_sha,size,manifest=sys.argv[1],int(sys.argv[2]),sys.argv[3]
assert not any(p.exists() for p in [control,release,upload,r/'long-dispatch-started.json',r/'long-submissions.json',*[r/('run-'+a) for a in ('protenix','boltz2','boltz2_chemistry')]])
assert not Path('/zhome/95/e/206121/mosaic-runs/20261022-brd4-jq1-25-fixed-lrreset-04').exists()
oldjob='29792418'; oldparent=r/'after-ready-boltz2_chemistry'
final=subprocess.run(['bjobs','-l',oldjob],capture_output=True,text=True,check=True,timeout=30).stdout
c=re.sub(r'\s+','',final)
assert re.findall(r'Job<([^>]+)>',c)==[oldjob] and re.findall(r'Status<([^>]+)>',c)==['EXIT'] and re.findall(r'Queue<([^>]+)>',c)==['hpc']
lsf=oldparent/('stdout.'+oldjob+'.log');err=oldparent/('stderr.'+oldjob+'.log')
assert 'Subject: Job '+oldjob+':' in lsf.read_text() and 'Exited with exit code 1.' in lsf.read_text()
gates=list(r.glob('quota-dispatch-*/lrreset-storage-gate.json'));assert len(gates)==1
failed=json.loads(gates[0].read_bytes());assert failed['status']=='failed' and 'TimeoutExpired' in failed['error']
with upload.open('xb') as f:
 for data in iter(lambda:sys.stdin.buffer.read(1048576),b''):f.write(data)
assert upload.stat().st_size==size and hashlib.sha256(upload.read_bytes()).hexdigest()==expected_sha
release.mkdir();source.mkdir();seen=set()
with tarfile.open(upload) as tar:
 for m in tar:
  p=Path(m.name);assert m.isfile() and len(p.parts)==2 and p.parts[0]=='source' and p.name not in seen
  seen.add(p.name)
  with (source/p.name).open('xb') as f:f.write(tar.extractfile(m).read())
raw=(source/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest
m=json.loads(raw);assert len(m['files'])==4 and seen==set(m['files'])|{'source_manifest.json'}
for name,h in m['files'].items():
 data=(source/name).read_bytes();assert hashlib.sha256(data).hexdigest()==h
 if name.endswith('.py'):ast.parse(data)
subprocess.run(['bash','-n',str(source/'all_work3_launch.lsf')],check=True)
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
science=b/'restart-lrreset-release-04/source'
subprocess.run([python,str(science/'scripts/verify_stage0_release.py'),str(science),'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'],check=True)
for p in source.iterdir():p.chmod(0o444)
source.chmod(0o555)
control.mkdir()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def proof(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
prior=r/'all-work3-launch-01'
priorproofs=[]
for name,job,status in [('cpu','29792659','DONE'),('after-cpu','29792660','EXIT')]:
 q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,check=True,timeout=30).stdout
 compact=re.sub(r'\s+','',q)
 assert re.findall(r'Job<([^>]+)>',compact)==[job] and re.findall(r'Status<([^>]+)>',compact)==[status] and re.findall(r'Queue<([^>]+)>',compact)==['hpc']
 saved=control/(name+'-predecessor-scheduler.txt')
 with saved.open('x') as f:f.write(q)
 log=prior/name/('stdout.'+job+'.log')
 assert 'Subject: Job '+job+':' in log.read_text()
 assert ('Successfully completed.' if status=='DONE' else 'Exited with exit code 1.') in log.read_text()
 priorproofs.extend([proof(saved),proof(log),proof(prior/name/('stderr.'+job+'.log')),proof(prior/name/'submission.json')])
oldmanifest=b/'restart-lrreset-allwork3-launch-01/source/source_manifest.json'
assert hashlib.sha256(oldmanifest.read_bytes()).hexdigest()=='99fcd95e150a671adb30a1cb910079474452319ddd67a36b042dc6a8e94940fa'
import xml.etree.ElementTree as ET
xml=prior/'cpu/payload/tests.xml'; tree=ET.parse(xml)
assert len(list(tree.iter('testcase')))==20 and not any(list(tree.iter(t)) for t in ['failure','error','skipped'])
assert not (prior/'cpu-certificate.json').exists() and not (prior/'adoption.json').exists()
priorproofs.extend([proof(oldmanifest),proof(xml),proof(prior/'scheduler-diagnostic/assessment.json'),proof(prior/'scheduler-diagnostic/bjobs-raw.txt')])
write(control/'previous-launcher-failure.json',{'cpu':'29792659','cpu_status':'DONE','cpu_cases':20,'callback':'29792660','callback_status':'EXIT','no_prior_long_submission':True,'proofs':priorproofs})
saved=control/'original-callback-scheduler.txt'
with saved.open('x') as f:f.write(final)
pin={'job':oldjob,'queue':'hpc','status':'EXIT','gate':proof(gates[0]),'scheduler':proof(saved),'proofs':[proof(p) for p in (gates[0],saved,lsf,err)]}
write(control/'inherited-failure.json',pin)
write(control/'source-staging.json',{'status':'passed','manifest':manifest,'archive_sha256':expected_sha,'archive_bytes':size,'files':4,'readonly':True})
sys.path.insert(0,str(source));import all_work3_launch as launch
native=launch.inherited()
gate=launch.science.storage.check_work_storage(control/'preflight-quota',phase='preflight')
jobs={}
for name in ('cpu','after-cpu'):
 parent=control/name;parent.mkdir()
 cmd=launch.cpu_command(name,manifest,jobs['cpu']['job_id'] if name=='after-cpu' else '0')
 write(parent/'submission-started.json',{'outcome':'unknown','args':cmd})
 record={'outcome':'unknown','args':cmd,'launcher_manifest':manifest}
 try:
  q=subprocess.run(cmd,text=True,capture_output=True,timeout=60);record.update(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
  ids=re.findall(r'Job <(\d+)>',q.stdout)
  if q.returncode==0 and len(ids)==1:record.update(outcome='submitted',job_id=ids[0])
 except Exception as error:
  record['error']=repr(error);raise
 finally:write(parent/'submission.json',record)
 assert record['outcome']=='submitted','Unknown outcome prohibits replay'
 jobs[name]=record
 q=subprocess.run(['bjobs','-l',record['job_id']],capture_output=True,text=True,timeout=30)
 with (parent/'scheduler-at-submission.txt').open('x') as f:f.write(q.stdout+q.stderr)
write(control/'initial-submissions.json',jobs)
print(json.dumps({'status':'submitted_not_passed','manifest':manifest,'control':str(control),'jobs':jobs,'original_native_all_passed':True,'preflight_gate':gate}))
'''


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    files={}
    for name in ['README.md','all_work3_launch.py','fixture_all_work3_launch.py','all_work3_launch.lsf']:
        data=(OWNER/name).read_bytes().replace(b'\r\n',b'\n')
        if name.endswith('.py'):ast.parse(data,filename=name)
        files[name]=data
    metadata={'version':'jq1-lrreset-all-work3-launch-adoption-v1','files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
    raw=(json.dumps(metadata,indent=2,sort_keys=True)+'\n').encode();manifest=hashlib.sha256(raw).hexdigest()
    archive=OUTPUT/'source.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for name,data in {**files,'source_manifest.json':raw}.items():
            info=tarfile.TarInfo('source/'+name);info.size=len(data);info.mode=0o444;t.addfile(info,io.BytesIO(data))
    package={'files':4,'manifest_sha256':manifest,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size}
    (OUTPUT/'source_manifest.json').write_bytes(raw)
    (OUTPUT/'package.json').write_text(json.dumps(package,indent=2)+'\n')
    (OWNER/'package.json').write_text(json.dumps(package,indent=2)+'\n')
    command=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE,package['archive_sha256'],str(package['archive_bytes']),manifest])
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(command)]
    (OUTPUT/'submission-started.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'args':args},indent=2)+'\n')
    try:
        result=subprocess.run(args,input=archive.read_bytes(),capture_output=True,timeout=180)
    except subprocess.TimeoutExpired as error:
        (OUTPUT/'submission-timeout.json').write_text(json.dumps({'error':repr(error),'outcome':'unknown'})+'\n')
        raise
    (OUTPUT/'submission-stdout.txt').write_bytes(result.stdout);(OUTPUT/'submission-stderr.txt').write_bytes(result.stderr)
    (OUTPUT/'submission-result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    print(result.stdout.decode()[-12000:]);print(result.stderr.decode()[-1200:])
    assert result.returncode==0,'Failed/unknown staging preserved; inspect, do not replay'


if __name__=='__main__':
    main()
