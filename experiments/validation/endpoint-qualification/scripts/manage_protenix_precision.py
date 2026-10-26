"""Stdlib packaging, one-shot staging and compact observation for a diagnostic.

All tests, model evaluations and precision calculations run in allocated HPC.
This operator can never launch optimizer segments or alter old results.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[4]
OWNER = REPO / 'experiments/validation/endpoint-qualification'
TAG = '20261026-protenix-endpoint-fix-03'
RUN = OWNER / 'runs' / TAG
LOCAL = REPO / 'results/hpc' / TAG
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
SOURCE = BASE + '/protenix-endpoint-fix-release-03/source'
ROOT = BASE + '/protenix-endpoint-fix-03'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'


def raw(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def command(program, name, timeout=180):
    out = LOCAL / name
    out.mkdir(exist_ok=False)
    (out / 'program.py').write_text(program, encoding='utf8', newline='\n')
    argv = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','ConnectionAttempts=1','dtu-hpc','bash -l -c ' + shlex.quote(PYTHON + ' -')]
    receipt = {'started_utc':datetime.now(timezone.utc).isoformat(), 'argv':argv,
               'program_sha256':sha(program.encode()), 'status':'unknown'}
    try:
        with (out/'stdout.json').open('xb') as stdout, (out/'stderr.txt').open('xb') as stderr:
            result = subprocess.run(argv, input=program.encode(), stdout=stdout, stderr=stderr, timeout=timeout)
        receipt.update(status='returned', returncode=result.returncode)
    finally:
        receipt['ended_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['stdout_sha256'] = sha((out/'stdout.json').read_bytes())
        (out/'receipt.json').write_bytes(raw(receipt))
    assert result.returncode == 0, (out/'stderr.txt').read_text()
    return json.loads((out/'stdout.json').read_bytes())


def prepare(counts_path):
    counts = json.loads(Path(counts_path).read_bytes())
    assert set(counts) == {'tests.test_protenix_precision_reference', 'tests.test_protenix_precision_admission',
                           'tests.test_protenix_precision_native', 'tests.test_protenix_precision_ops','tests.test_current_capacity'}
    assert all(type(v) is int and v > 0 for v in counts.values())
    plan = json.loads((RUN/'plan.json').read_bytes())
    forensic = json.loads((OWNER/'runs/20261025-protenix-endpoint-forensics-01/source-manifest.json').read_bytes())
    parents = list(forensic['parent_sources']) + [
        {'path':BASE+'/protenix-endpoint-forensics-release-01/source',
         'manifest_sha256':'d127d6abe87307a5d305edb6d04e3a32e3545c508e58b82ce45290e0f3fc03d8','files':13}]
    parents.append({'path':BASE+'/chemistry-production-release-03/source','files':14,'manifest_sha256':'c73796b041d8432ef1634e8c56920006170a094d3bdc74867c08f3b5415126d7'})
    proofs = list(forensic['preserved_proofs']) + [plan['original_forensics_plan'], plan['prior_forensic_report']]
    predecessor=plan.get('engineering_predecessor')
    if predecessor:
        parents += [predecessor['source']]
        proofs += predecessor['proofs']
    proofs = list({v['path']:v for v in proofs}.values())
    names = ['README.md','scripts/README.md','tests/README.md',
             'scripts/protenix_precision_admission.py','scripts/protenix_precision_reference.py',
             'scripts/protenix_precision_native.py','scripts/protenix_precision_ops.py',
             'scripts/protenix_precision_stage.lsf','scripts/current_capacity.py','tests/test_current_capacity.py','tests/test_protenix_precision_admission.py',
             'tests/test_protenix_precision_reference.py','tests/test_protenix_precision_native.py',
             'tests/test_protenix_precision_ops.py']
    members = {}
    for name in names:
        data = (OWNER/name).read_bytes().replace(b'\r\n',b'\n')
        if name.endswith('.py'): ast.parse(data, filename=name)
        members[name] = data
    members['plan.json'] = (RUN/'plan.json').read_bytes()
    members['protocol.md'] = (RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    manifest = {'version':'protenix-endpoint-precision-source-v2',
                'files':{n:sha(v) for n,v in sorted(members.items())},
                'parent_sources':parents,'preserved_proofs':proofs,'cpu_counts':counts,
                'original_results_changed':False,'optimization_settings_changed':False,
                'can_certify_or_restart_original_study':False,'optimizer_updates':0}
    members['source_manifest.json'] = raw(manifest)
    LOCAL.mkdir(parents=True, exist_ok=True)
    out = LOCAL/'source01'; out.mkdir(exist_ok=False)
    for name,data in members.items():
        target = out/name; target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for name,data in sorted(members.items()):
            info=tarfile.TarInfo(name);info.mode=0o444;info.size=len(data);info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    data=buf.getvalue();(out/'source.tar.gz').write_bytes(data)
    result={'status':'prepared_not_validated','files':len(manifest['files']),
            'manifest_sha256':sha(members['source_manifest.json']),
            'archive_bytes':len(data),'archive_sha256':sha(data),
            'plan_sha256':sha(members['plan.json']),'cpu_counts':counts}
    (out/'preparation.json').write_bytes(raw(result))
    (RUN/'source-manifest.json').write_bytes(members['source_manifest.json'])
    (RUN/'preparation.json').write_bytes(raw(result))
    print(json.dumps(result,indent=2))


def stage():
    src=LOCAL/'source01'; prep=json.loads((src/'preparation.json').read_bytes())
    archive=(src/'source.tar.gz').read_bytes()
    assert len(archive)==prep['archive_bytes'] and sha(archive)==prep['archive_sha256']
    program='PREP='+repr(prep)+'\nARCHIVE='+repr(base64.b64encode(archive).decode())+'\n'+r"""
import base64,hashlib,io,json,os,sys,tarfile
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
source=base/'protenix-endpoint-fix-release-03/source';root=base/'protenix-endpoint-fix-03'
assert not root.exists() and not source.parent.exists(),'Existing/unknown outcome; inspect, never replay'
data=base64.b64decode(ARCHIVE,validate=True)
assert len(data)==PREP['archive_bytes'] and hashlib.sha256(data).hexdigest()==PREP['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive.getmembers():
  name=PurePosixPath(member.name)
  assert member.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in member.name
  assert member.name not in members
  members[member.name]=archive.extractfile(member).read();assert len(members[member.name])==member.size
m=json.loads(members['source_manifest.json'])
assert hashlib.sha256(members['source_manifest.json']).hexdigest()==PREP['manifest_sha256']
assert set(members)==set(m['files'])|{'source_manifest.json'} and len(m['files'])==PREP['files']
for name,digest in m['files'].items():assert hashlib.sha256(members[name]).hexdigest()==digest
# Authenticate the exact already allocated metadata verifier before importing it.
p=base/'nonchem-endpoint-release-02/source'
r=(p/'source_manifest.json').read_bytes();assert hashlib.sha256(r).hexdigest()=='88372f92b6349e6c34c542fb84fb39482c4c340568431d1499ebc166a1e1e7fa'
f=json.loads(r)['files'];assert len(f)==14
for name,digest in f.items():
 with (p/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(p/'scripts'));import continuation_ops as parent
for row in m['parent_sources']:assert len(parent.verify(row['path'],row['manifest_sha256'])['files'])==row['files']
for row in m['preserved_proofs']:parent.check_binding(row)
root.mkdir();source.mkdir(parents=True)
for name,content in members.items():
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as stream:stream.write(content)
 target.chmod(0o444)
with (source.parent/'source.tar.gz').open('xb') as stream:stream.write(data)
(source.parent/'source.tar.gz').chmod(0o444)
for directory in sorted((p for p in source.rglob('*') if p.is_dir()),reverse=True):directory.chmod(0o555)
source.chmod(0o555)
syntax=__import__('subprocess').run(['bash','-n',str(source/'scripts/protenix_precision_stage.lsf')],capture_output=True,text=True)
assert syntax.returncode==0,syntax.stderr
sys.path.insert(0,str(source/'scripts'));import protenix_precision_ops as ops
pin=PREP['manifest_sha256'];ops.verify(source,pin)
preflight=root/'preflight';preflight.mkdir();gate=ops.storage(preflight,'preflight')
cpu=ops.submit('cpu',pin);callback=ops.submit('after-cpu',pin,dependency='ended('+cpu['job_id']+')')
receipt={'status':'diagnostic_cpu_submitted_not_passed','preparation':PREP,'cpu':cpu,'after_cpu':callback,'storage':gate,'original_results_changed':False,'optimizer_updates':0}
ops.write(root/'bootstrap-complete.json',receipt)
print(json.dumps(receipt))
"""
    result=command(program,'stage-01',360)
    (RUN/'submission.json').write_bytes(raw(result))
    print(json.dumps(result,indent=2))


def observe(name):
    assert name and all(c.isalnum() or c=='-' for c in name)
    program='ROOT='+repr(ROOT)+'\n'+r"""
from datetime import datetime,timezone
import json,hashlib,subprocess
from pathlib import Path
root=Path(ROOT)
def file(path):
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest,'text':path.read_text()}
records=[];jobs=[]
for phase in ('cpu','after-cpu','gpu'):
 p=root/phase
 if (p/'submission.json').exists():jobs.append(json.loads((p/'submission.json').read_bytes())['job_id'])
 for pattern in ('submission*.json','stdout.*.log','stderr.*.log','*storage*.json','terminal-query*.json'):
  for path in sorted(p.glob(pattern)):
   if path.is_file() and path.stat().st_size<2000000:records.append(file(path))
for relative in ('bootstrap-complete.json','cpu-certificate.json',
 'after-cpu/dispatch-storage/precision-storage-dispatch.json','cpu/payload/tests.xml',
 'cpu/payload/command.json','cpu/payload/tests.log','cpu/payload/assessment.json',
 'cpu/payload/native-startup-invocation.json',
 'cpu/payload/native-startup/endpoint-startup.json','gpu/payload/assessment.json',
 'gpu/post-execution-preservation.json','gpu/native-invocation.json',
 'gpu/payload/protected-before.json','gpu/payload/protected-after.json',
 'gpu/payload/production/assessment.json','gpu/payload/reference/assessment.json',
 'gpu/payload/reference/admission.json','gpu/payload/reference/native-dtype-audit.json',
 'gpu/payload/reference/oracle-points.json'):
 p=root/relative
 if p.exists():records.append(file(p))
q=subprocess.run(['bjobs','-l',*jobs],capture_output=True,text=True,timeout=40) if jobs else None
print(json.dumps({'captured_utc':datetime.now(timezone.utc).isoformat(),'files':records,'scheduler':None if q is None else {'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}}))
"""
    result=command(program,name,120)
    statuses=[]
    for f in result['files']:
        if f['path'].endswith(('assessment.json','cpu-certificate.json')):
            a=json.loads(f['text']);statuses.append({'path':f['path'],'status':a.get('status'),'cases':a.get('cases')})
    scheduler=result['scheduler']
    jobs=[] if scheduler is None else re.findall(r'Job <(\d+)>.*?Status\s*<([^>]+)>',scheduler['stdout'],re.S)
    print(json.dumps({'captured_utc':result['captured_utc'],'statuses':statuses,'jobs':jobs,'scheduler_returncode':None if scheduler is None else scheduler['returncode']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','stage','observe'));p.add_argument('--counts');p.add_argument('--name')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.counts)
    elif a.action=='stage':stage()
    else:observe(a.name)

if __name__=='__main__':main()
