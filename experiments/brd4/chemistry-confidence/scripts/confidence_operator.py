"""Stdlib-only packaging and node-local allocated CPU evidence transfer.

Account-bound DTU execution profile; reusable scientific code has no account
paths. No persistent remote shared-storage output or saved-run mutation.
"""

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import xml.etree.ElementTree as ET

OWNER = Path(__file__).resolve().parents[1]
REPO = OWNER.parents[2]
TAG = "20261029-chemistry-confidence-05"
RUN = OWNER/"runs"/TAG
LOCAL = REPO/"results/hpc"/TAG
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"


def raw(value): return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
def sha(data): return hashlib.sha256(data).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()


def prepare():
    destination = LOCAL/"source01"; destination.mkdir(parents=True, exist_ok=False)
    paths = [p for p in (REPO/"src/mosaic").rglob("*") if p.is_file() and p.suffix in (".py", ".json")]
    paths += [REPO/"tests/small_molecule/test_chemistry_confidence.py"]
    paths += [p for p in OWNER.rglob("*") if p.is_file() and p.suffix in ('.py', '.md') and "runs" not in p.relative_to(OWNER).parts]
    members = {p.relative_to(REPO).as_posix(): p.read_bytes() for p in sorted(paths)}
    for name, data in members.items():
        if name.endswith(".py"): ast.parse(data, filename=name)
    control = subprocess.check_output(["git", "show", "HEAD:src/mosaic/small_molecule/chemistry/trajectory.py"], cwd=REPO)
    members["controls/trajectory.py"] = control
    unchanged = {}
    for name in ("loss.py", "preparation.py"):
        path = "src/mosaic/small_molecule/chemistry/"+name
        original = subprocess.check_output(["git", "show", "HEAD:"+path], cwd=REPO)
        # Windows checkout newline conversion is not a scientific source edit.
        assert original.replace(b"\r\n", b"\n") == members[path].replace(b"\r\n", b"\n")
        unchanged[path] = {"archive_sha256": sha(members[path]), "git_sha256": sha(original)}
    manifest = {"version": "chemistry-confidence-source-v1", "prepared_utc": now(),
        "files": {name: {"bytes": len(data), "sha256": sha(data)} for name, data in members.items()},
        "control_trajectory_sha256": sha(control), "unchanged_repair": unchanged,
        "expected_tests": 35, "expected_paired_rows": 24, "source_scope": "library plus bounded CPU comparison",
        "saved_runs_written": False, "native_models_evaluated": False}
    members["source-manifest.json"] = raw(manifest)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in sorted(members.items()):
            info = tarfile.TarInfo(name); info.size=len(data); info.mode=0o444; info.mtime=0
            archive.addfile(info, io.BytesIO(data))
    data = buffer.getvalue(); (destination/"source.tar.gz").write_bytes(data)
    (RUN/"source-manifest.json").write_bytes(members["source-manifest.json"])
    receipt = {"status": "prepared_not_numerically_validated", "utc": now(),
        "archive_bytes": len(data), "archive_sha256": sha(data),
        "manifest_sha256": sha(members["source-manifest.json"]), "files": len(manifest["files"]),
        "local_checks": "stdlib AST/byte checks only", "saved_results_changed": False}
    (RUN/"preparation.json").write_bytes(raw(receipt)); print(json.dumps(receipt, indent=2))


NODE = r'''
import base64,hashlib,io,json,os,pathlib,shutil,socket,subprocess,sys,tarfile,tempfile,time,traceback,xml.etree.ElementTree as ET
assert os.environ.get('LSB_JOBID','').isdigit() and os.environ.get('LSB_QUEUE')=='hpcint'
assert os.environ.get('LSB_DJOB_NUMPROC')=='4'
assert socket.gethostname().startswith('n-'),'Actual compute host required'
scratch=pathlib.Path('/tmp').resolve()
assert not str(scratch).startswith(('/work3/','/zhome/','/home/')),'Node-local scratch required'
assert shutil.disk_usage(scratch).free>=1024**3
root=pathlib.Path(tempfile.mkdtemp(prefix='chem-confidence-',dir=scratch));source=root/'source';source.mkdir();output=root/'evidence';output.mkdir()
os.chdir(root)
data=base64.b64decode(ARCHIVE,validate=True)
assert len(data)==PREP['archive_bytes'] and hashlib.sha256(data).hexdigest()==PREP['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 seen=set()
 for member in archive.getmembers():
  p=pathlib.PurePosixPath(member.name)
  assert member.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in member.name and member.name not in seen
  seen.add(member.name);dest=source/member.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.extractfile(member).read());dest.chmod(0o444)
manifest_raw=(source/'source-manifest.json').read_bytes();assert hashlib.sha256(manifest_raw).hexdigest()==PREP['manifest_sha256'];manifest=json.loads(manifest_raw)
assert seen==set(manifest['files'])|{'source-manifest.json'}
for name,pin in manifest['files'].items():
 b=(source/name).read_bytes();assert len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['sha256']
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555)
(output/'source-manifest.json').write_bytes(manifest_raw)
runtime={'job_id':os.environ['LSB_JOBID'],'queue':os.environ['LSB_QUEUE'],'slots':os.environ['LSB_DJOB_NUMPROC'],'host':socket.gethostname(),'source_manifest_sha256':PREP['manifest_sha256'],'node_local_root':str(root),'persistent_shared_writes':False}
(output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','JAX_PLATFORMS':'cpu','JAX_ENABLE_X64':'1','PYTHONPATH':str(source/'src'),'XDG_CACHE_HOME':str(root/'cache'),'JAX_COMPILATION_CACHE_DIR':str(root/'cache/jax'),'MOSAIC_CONTROL_TRAJECTORY':str(source/'controls/trajectory.py'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','JAX_NUM_THREADS':'1'}
temporary=root/'tmp';temporary.mkdir()
environment.update(TMPDIR=str(temporary),TEMP=str(temporary),TMP=str(temporary),MPLCONFIGDIR=str(root/'cache/matplotlib'))
probe=[sys.executable,'-B','-c','import tempfile; print(tempfile.gettempdir())']
observed=subprocess.run(probe,cwd=source,env=environment,capture_output=True,text=True,timeout=20)
assert observed.returncode==0 and observed.stdout.strip()==str(temporary),'Child temp directory must be node-local'
runtime['child_temp_directory']=str(temporary)
(output/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
(output/'tempdir-proof.json').write_text(json.dumps({'argv':probe,'returncode':observed.returncode,'stdout':observed.stdout,'stderr':observed.stderr,'expected':str(temporary)},indent=2)+'\n')
assessment={'status':'running','job_id':os.environ['LSB_JOBID'],'source_manifest_sha256':PREP['manifest_sha256'],'saved_results_changed':False,'native_binding_validated':False}
commands=[];begin=time.monotonic()
try:
 tests=[sys.executable,'-B','-m','pytest',str(source/'tests/small_molecule/test_chemistry_confidence.py'),'-o','addopts=','-o','cache_dir='+str(root/'pytest-cache'),'--basetemp',str(root/'pytest-tmp'),'--junitxml='+str(output/'tests.xml'),'-q']
 commands.append({'argv':tests,'scope':'allocated scientific fixtures'})
 with (output/'tests.log').open('wb') as stream:
  process=subprocess.run(tests,cwd=source,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
 assert process.returncode==0,'Allocated fixtures failed'
 cases=ET.parse(output/'tests.xml').findall('.//testcase')
 assert len(cases)==manifest['expected_tests'] and not any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped'))
 assessment['tests_passed']=len(cases)
 compare=[sys.executable,'-B',str(source/'experiments/brd4/chemistry-confidence/scripts/compare.py'),'--output',str(output/'comparison')]
 commands.append({'argv':compare,'scope':'chemistry-only comparison; no native binding model'})
 with (output/'comparison.log').open('wb') as stream:
  process=subprocess.run(compare,cwd=source,env={**environment,'JAX_ENABLE_X64':'0'},stdout=stream,stderr=subprocess.STDOUT,timeout=450)
 assert process.returncode==0,'Allocated controlled comparison failed'
 comparison=json.loads((output/'comparison/comparison.json').read_bytes())
 assert comparison['status']=='completed_measurements_not_native_validation' and len(comparison['rows'])==manifest['expected_paired_rows']
 assert all(len(row['updates'])==128 and len(row['checkpoints'])==3 for row in comparison['rows'])
 assessment.update(status='passed',paired_rows=len(comparison['rows']),scope='CPU implementation/gradient validation and chemistry-only paired measurements; not native binding efficacy')
except BaseException as error:
 assessment.update(status='failed',error=repr(error),traceback=traceback.format_exc())
finally:
 assessment['elapsed_seconds']=time.monotonic()-begin
 (output/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
 (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 files={p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in output.rglob('*') if p.is_file()}
 assert sum(pin['bytes'] for pin in files.values())<256*1024**2
 (output/'evidence-manifest.json').write_text(json.dumps({'files':files,'assessment':assessment},indent=2)+'\n')
 buffer=io.BytesIO()
 with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
  for p in sorted(output.rglob('*')):
   if p.is_file():archive.add(p,arcname=p.relative_to(output).as_posix(),recursive=False)
 result=buffer.getvalue()
 print('CONFIDENCE_EVIDENCE_META '+json.dumps({'bytes':len(result),'sha256':hashlib.sha256(result).hexdigest(),'assessment':assessment}),flush=True)
 print('CONFIDENCE_EVIDENCE_BEGIN',flush=True);print(base64.b64encode(result).decode(),flush=True);print('CONFIDENCE_EVIDENCE_END',flush=True)
 # This is exclusively the newly created node-local temporary directory.
 assert root.parent==scratch and root.name.startswith('chem-confidence-')
 for p in source.rglob('*'):
  if p.is_dir():p.chmod(0o755)
 source.chmod(0o755);shutil.rmtree(root)
sys.exit(0 if assessment['status']=='passed' else 1)
'''


def run():
    assert not (RUN/'execution-started.json').exists(),'Prior/unknown execution marker exists; inspect receipts before recovery'
    prep=json.loads((RUN/"preparation.json").read_bytes())
    data=(LOCAL/"source01/source.tar.gz").read_bytes()
    assert len(data)==prep['archive_bytes'] and sha(data)==prep['archive_sha256']
    destination=LOCAL/"allocated01";destination.mkdir(exist_ok=False)
    node="PREP="+repr(prep)+"\nARCHIVE="+repr(base64.b64encode(data).decode())+"\n"+NODE
    # The job reads its sealed interpreter/program from stdin. No command or
    # output path points at a shared Mosaic study/release. This is NOT a way to
    # exempt shared writes: the bootstrap actually enforces node-local paths.
    remote=r'''
import base64,json,os,re,subprocess,sys,time
payload=base64.b64decode(PAYLOAD,validate=True)
argv=['bsub','-I','-q','hpcint','-n','4','-R','span[hosts=1]','-R','rusage[mem=2GB]','-J','chem-confidence-node-local-cpu','timeout','--signal=TERM','--kill-after=15','840s','bash','-l','-c','read -r executor; exec "$executor" -B -u -']
process=subprocess.run(argv,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800)
sys.stdout.buffer.write(process.stdout);sys.stdout.flush();sys.stderr.buffer.write(process.stderr);sys.stderr.flush()
ids=set(re.findall(rb'Job <([0-9]+)>',process.stdout+process.stderr))
proofs=[]
for jid in ids:
 for attempt in range(15):
  observed=subprocess.run(['bjobs','-json','-o','jobid stat queue exit_code',jid.decode()],capture_output=True,text=True,timeout=30)
  if observed.returncode==0 and all(r['STAT'] in ('DONE','EXIT') for r in json.loads(observed.stdout)['RECORDS']):break
  time.sleep(1)
 for command in (['bjobs','-l',jid.decode()],['bjobs','-json','-o','jobid stat queue exit_code',jid.decode()]):
  q=subprocess.run(command,capture_output=True,text=True,timeout=30)
  proofs.append({'argv':command,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
print('CONFIDENCE_SCHEDULER '+json.dumps({'actual_job_ids':[v.decode() for v in sorted(ids)],'bsub_argv':argv,'bsub_returncode':process.returncode,'proofs':proofs}),flush=True)
sys.exit(process.returncode)
'''
    payload=(PYTHON+"\n"+node).encode()
    program="PAYLOAD="+repr(base64.b64encode(payload).decode())+"\n"+remote
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1',
          'dtu-hpc', 'bash -l -c '+shlex.quote(PYTHON+' -B -u -')]
    (destination/"node-program.py").write_text(node,encoding='utf8',newline='\n')
    (destination/"transport-program.py").write_text(program,encoding='utf8',newline='\n')
    receipt={'status':'unknown_do_not_repeat','started_utc':now(),'source_manifest_sha256':prep['manifest_sha256'],
             'archive_sha256':prep['archive_sha256'],'transport_program_sha256':sha(program.encode())}
    (RUN/"execution-started.json").write_bytes(raw(receipt))
    print('Starting allocated CPU validation; outcome/actual job ID will be recorded.',flush=True)
    try:
        with (destination/'stdout.txt').open('xb') as stdout, (destination/'stderr.txt').open('xb') as stderr:
            process=subprocess.run(argv,input=program.encode(),stdout=stdout,stderr=stderr,timeout=1900)
        receipt.update(status='returned',returncode=process.returncode,ended_utc=now())
    finally:
        receipt.update(stdout_sha256=sha((destination/'stdout.txt').read_bytes()),stderr_sha256=sha((destination/'stderr.txt').read_bytes()))
        (destination/'transport-receipt.json').write_bytes(raw(receipt))
    stdout=(destination/'stdout.txt').read_text()
    meta_lines=[v for v in stdout.splitlines() if v.startswith('CONFIDENCE_EVIDENCE_META ')]
    scheduler_lines=[v for v in stdout.splitlines() if v.startswith('CONFIDENCE_SCHEDULER ')]
    if len(meta_lines)!=1 or len(scheduler_lines)!=1:
        (RUN/'early-failure.json').write_bytes(raw({**receipt,'status':'missing_evidence_or_early_startup_failure',
            'scheduler':None if len(scheduler_lines)!=1 else json.loads(scheduler_lines[0].split(' ',1)[1]),
            'numerical_pass_established':False}))
        raise AssertionError('Missing/unknown allocated outcome; inspect receipts, do not repeat')
    meta=json.loads(meta_lines[0].split(' ',1)[1]);scheduler=json.loads(scheduler_lines[0].split(' ',1)[1])
    (RUN/'scheduler.json').write_bytes(raw(scheduler))
    encoded=stdout.split('CONFIDENCE_EVIDENCE_BEGIN\n')[1].split('\nCONFIDENCE_EVIDENCE_END')[0]
    archive_data=base64.b64decode(encoded.strip(),validate=True)
    assert len(archive_data)==meta['bytes'] and sha(archive_data)==meta['sha256']
    (destination/'evidence.tar.gz').write_bytes(archive_data)
    extracted=destination/'verified';extracted.mkdir()
    members={}
    with tarfile.open(fileobj=io.BytesIO(archive_data),mode='r:gz') as archive:
        for member in archive.getmembers():
            path=PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts and '\\' not in member.name and member.name not in members
            members[member.name]=archive.extractfile(member).read()
    manifest=json.loads(members['evidence-manifest.json'])
    assert set(members)==set(manifest['files'])|{'evidence-manifest.json'}
    for name,pin in manifest['files'].items():
        assert len(members[name])==pin['bytes'] and sha(members[name])==pin['sha256']
    for name,value in members.items():
        target=extracted/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
    assessment=json.loads(members['assessment.json']);runtime=json.loads(members['runtime.json'])
    assert scheduler['actual_job_ids']==[runtime['job_id']]==[assessment['job_id']]
    assert runtime['queue']=='hpcint' and runtime['slots']=='4' and runtime['persistent_shared_writes'] is False
    assert assessment['source_manifest_sha256']==prep['manifest_sha256'] and sha(members['source-manifest.json'])==prep['manifest_sha256']
    verified={'status':'evidence_verified_original_outcome_preserved','utc':now(), 'assessment':assessment,
              'job_id':runtime['job_id'],'archive_bytes':len(archive_data),'archive_sha256':sha(archive_data),
              'verified_members':len(members),'saved_results_changed':False,'native_binding_validated':False}
    if 'tests.xml' in members:
        cases=ET.fromstring(members['tests.xml']).findall('.//testcase')
        verified.update(actual_cases=len(cases),failures=sum(c.find('failure') is not None for c in cases),
                        errors=sum(c.find('error') is not None for c in cases),skips=sum(c.find('skipped') is not None for c in cases))
    if process.returncode==0:
        assert assessment['status']=='passed' and verified['actual_cases']==35 and not any(verified[k] for k in ('failures','errors','skips'))
        # Actual same-job scheduler DONE, not a generated status.
        raw_scheduler=' '.join(p['stdout'] for p in scheduler['proofs'])
        assert re.search(r'\bDONE\b',raw_scheduler) and scheduler['bsub_returncode']==0
        verified['status']='passed_allocated_cpu_and_byte_verified_not_native_binding_validation'
    (RUN/'verification.json').write_bytes(raw(verified));print(json.dumps(verified,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','run'))
    phase=parser.parse_args().phase
    prepare() if phase=='prepare' else run()
