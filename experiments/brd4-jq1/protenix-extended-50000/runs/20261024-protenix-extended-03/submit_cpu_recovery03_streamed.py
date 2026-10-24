"""Distinct framed-stdin CPU admission recovery of staged Pro03 after storage offload.

No upload, extraction, source mutation, cleanup or numerical work is provided.
The original failed gate and all unknown outcomes stay immutable. The parent
operator alone may execute this file after reviewing cleanup and current state.
Execution makes one SSH attempt, reruns the exact original full preflight, and
only then uses the sealed controller's original CPU866/own-ended callback.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TAG = "20261024-protenix-extended-03"
MANIFEST = "da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562"
ARCHIVE = "bde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef"
ARCHIVE_BYTES = 39644558
OLD_MANIFEST = "a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00"
REVISION = "c74c659344123937608077e3e4078f75f09cd743"
INSPECTION_SHA = "7b28b33288172998d1dd77c4e4f6a986e7eefafb8a3acf8ec0cc5e43a0bac5cc"
FAILED_GATE_SHA = "6d1d0fae516de98615b83ebc005ab2330dd2146c08ef631f1bd5c3d28593a951"
OLD_PINS = {
    "continuation-after-native-0000/stderr.29798162.log": {
        "bytes": 1761,
        "sha256": "bd53db318c0b92fcac7c95b4ee18f7095283ea4d6355c2bcbd0ccd4753ceec7d"
    },
    "continuation-after-native-0000/stdout.29798162.log": {
        "bytes": 2344,
        "sha256": "da7983522401fefd7cb0d6ccf63c8785a105fe56d7d4fe19772586e14caab707"
    },
    "continuation-after-native-0000/submission.json": {
        "bytes": 1382,
        "sha256": "318770b52d775cb93f25192b5f9f33f618d2f28c09bff104c3332176b50c8daf"
    },
    "continuation-native-0000/stderr.29798161.log": {
        "bytes": 3710,
        "sha256": "747533af4205a17eb3e4073720a4f5cd1c2bdcd46255de7adb72a0ecd3d9ddd4"
    },
    "continuation-native-0000/stdout.29798161.log": {
        "bytes": 2342,
        "sha256": "2bacdc93fa3398c3238f156cebeac853c39d647f94502d8e7cb810ccb8bd5919"
    },
    "continuation-native-0000/submission.json": {
        "bytes": 1356,
        "sha256": "287f339427b96c7b269938e7b09638a12855522d9a5c8d390056f162ac443df7"
    },
    "random-after-native-0000/stderr.29798160.log": {
        "bytes": 1761,
        "sha256": "bd53db318c0b92fcac7c95b4ee18f7095283ea4d6355c2bcbd0ccd4753ceec7d"
    },
    "random-after-native-0000/stdout.29798160.log": {
        "bytes": 2320,
        "sha256": "69d249db60f9760a774225af40c5df92ce66e6d8e874adc74c7e678d9daed5be"
    },
    "random-after-native-0000/submission.json": {
        "bytes": 1358,
        "sha256": "5bb9375eff799d02c9cf55ab55ffdfe9f9813a59abfdc54e73058bd23c90a254"
    },
    "random-native-0000/stderr.29798159.log": {
        "bytes": 3440,
        "sha256": "3b6a507c49322336efd96f9df70f673c69eeb870c4c81d11f7e11f2a75b024ca"
    },
    "random-native-0000/stdout.29798159.log": {
        "bytes": 2318,
        "sha256": "629e9eed183a7ea0413b89d16540d36cf37c053a9c6644c8d52f0249030078da"
    },
    "random-native-0000/submission.json": {
        "bytes": 1332,
        "sha256": "b1b2d0a191ac1f40a85b76a48709880c0eec897c1b2ef5fab502849bb27eda01"
    }
}

REMOTE = r'''
import ast,hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'protenix-extended-03';release=base/'protenix-extended-release-03';source=release/'source'
upload=base/'protenix-extended-03.tar.gz';marker=base/'protenix-extended-03-staging-started.json'
oldroot=base/'protenix-extended-01';oldsource=base/'protenix-extended-release-01/source'
ancestor=base/'restart-lrreset-release-04/source'
manifest,pins=sys.argv[1],json.loads(sys.argv[2])
expected='da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562'
old_sha='a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
ancestor_sha='c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
revision='c74c659344123937608077e3e4078f75f09cd743'
snapshot_sha='edf15f06ba1b6591d62cbc3e04e19ecd93b553a0627da38c370fdeed3c1b780f'
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
jobs_expected={'cpu':'29798115','after-cpu':'29798116','random-native-0000':'29798159','random-after-native-0000':'29798160','continuation-native-0000':'29798161','continuation-after-native-0000':'29798162'}
control=root/'cpu-recovery-03'
failed_gate=root/'staging-01/storage-preflight/gate.json'
failed_started=root/'staging-01/started.json'
failed_gate_sha='6d1d0fae516de98615b83ebc005ab2330dd2146c08ef631f1bd5c3d28593a951'
failed_started_sha='56af07542fca09fbfbcc7f78ccbab0a44c174fd5bf3956371b2bbc6ff0350ab6'
GIB=1024**3

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def exclusive(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())

def verify_seal(path,checksum,count):
 assert path.is_dir() and not path.is_symlink() and sha(path/'source_manifest.json')==checksum,'Exact historical/new source manifest differs'
 value=read(path/'source_manifest.json');files=value['files']
 assert len(files)==count and {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() or p.is_symlink()}==set(files)|{'source_manifest.json'}
 for p in [path,*path.rglob('*')]:assert not p.is_symlink() and (p.is_file() or p.is_dir()) and not p.stat().st_mode&0o222,'Source not read-only regular content'
 for name,digest in files.items():
  p=path/name;assert p.resolve().is_relative_to(path.resolve()) and sha(p)==digest,'Source member changed'
 return value

def original_vector(label,phase,dependency):
 parent=oldroot/label
 if phase=='native':resources=['-q','gpul40s','-n','4','-R','span[hosts=1]','-R','rusage[mem=8GB]','-gpu','num=1:mode=exclusive_process','-W','02:00']
 else:resources=['-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30']
 args=['bsub',*resources,'-J','mosaic-pro50k-'+label,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
 if dependency:args+=['-w',dependency]
 return args+['bash','-l',str(oldsource/'scripts/hpc/extended_stage.lsf'),str(oldsource),str(oldroot),old_sha,phase,str(parent)]

def authenticate_history():
 assert read(oldroot/'jobs.json')==jobs_expected,'Unexpected old actual/unknown job outcomes'
 assert not os.path.lexists(oldroot/'projection.json') and not list(oldroot.glob('*-segment-*')),'Old scientific work unexpectedly exists'
 assert len(pins)==12
 for name,pin in pins.items():
  pure=PurePosixPath(name);p=oldroot/name
  assert not pure.is_absolute() and '..' not in pure.parts and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(oldroot.resolve())
  assert p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'],'Preserved predecessor proof changed'
 for label,job in jobs_expected.items():
  if label in ('cpu','after-cpu'):continue
  parent=oldroot/label;phase='after-native' if '-after-native-' in label else 'native'
  sub=read(parent/'submission.json');dep=None if phase=='native' else 'ended('+jobs_expected[label.replace('-after-native-','-native-')]+')'
  assert sub['job_id']==job and sub['outcome']=='submitted' and sub['phase']==phase and sub['dependency']==dep and sub['args']==original_vector(label,phase,dep),'Original source/resource/dependency submission differs'
  log=(parent/('stdout.'+job+'.log')).read_text()
  assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',log,re.M)==[(job,'Exited')] and re.search(r'Exited with exit code\s+1\b',log),'Original actual pre-model failure missing'
  error=(parent/('stderr.'+job+'.log')).read_text()
  if phase=='native':
   assert "KeyError: 'git_revision_reference'" in error and 'source_identity' in error,'Original cause changed'
   assert not list((parent/'payload/study').rglob('*')),'Unexpected saved scientific output in original failed validator'
  else:assert 'ValueError: Actual predecessor failed' in error,'Original own-producer failure changed'
 cpu=oldroot/'cpu-certificate.json'
 assert sha(cpu)=='c0f784d523d358dfd09d7f57fcfe333fbb5c069d93074681ea35562a5f423171'
 c=read(cpu);assert c['status']=='passed' and c['source_manifest_sha256']==old_sha and c['cases']==832,'Original CPU success must remain successful'
 for proof in c['proofs']:assert sha(proof['path'])==proof['sha256'],'Original passed CPU proof changed'

def preserved_failure():
 for path,size,digest in ((failed_gate,6133,failed_gate_sha),(failed_started,2319,failed_started_sha)):
  assert path.is_file() and not path.is_symlink() and path.stat().st_size==size and sha(path)==digest,'Original failed staging bytes changed'
 value=read(failed_gate)
 assert value['status']=='failed' and value['phase']=='preflight' and value['reserve_bytes']==4*GIB and value['file_reserve']==100000
 assert value['projected_bytes']==37348701956 and value['work3']['required_bytes']==41643669252 and value['work3']['available_bytes_conservative']==27472908456
 assert value['full_long_admission'] is False and value['full_extended']=={'random':GIB,'continuation':GIB}
 started=read(failed_started)
 assert started=={'outcome':'unknown','source_manifest_sha256':manifest,'old_source_manifest_sha256':old_sha,'old_failure_proofs':pins,'continuation_snapshot_sha256':snapshot_sha},'Original staged attempt identity changed'
 return {'path':str(failed_gate),'bytes':6133,'sha256':failed_gate_sha,'status':'failed_preserved'}

def no_new_jobs(*,control_exists=False):
 assert root.is_dir() and not root.is_symlink() and root.resolve()==root,'Missing/linked actual staged root'
 expected_children={'staging-01'}|({'cpu-recovery-03'} if control_exists else set())
 assert {p.name for p in root.iterdir()}==expected_children,'Prior CPU, callback, certificate, projection, native, segment, control, or unknown outcome exists; inspect before any new submission'
 assert {p.name for p in (root/'staging-01').iterdir()}=={'started.json','storage-preflight'}
 assert {p.name for p in (root/'staging-01/storage-preflight').iterdir()}=={'gate.json'}
 for name in ('jobs.json','jobs.json.tmp','cpu','after-cpu','cpu-certificate.json','projection.json','source-staging.json','dispatch.lock'):
  assert not os.path.lexists(root/name),'Actual/unknown CPU or successor state already exists'
 assert not list(root.glob('*-native-*')) and not list(root.glob('*-segment-*')),'Scientific output already exists'

assert not sys.flags.optimize and manifest==expected and len(pins)==12,'Wrong exact recovery request'
assert not any(os.path.lexists(p) for p in (base/'protenix-extended-02',base/'protenix-extended-release-02',base/'protenix-extended-02.tar.gz',base/'protenix-extended-02-staging-started.json')),'Local-only02 unexpectedly staged'
no_new_jobs()
old_manifest=verify_seal(oldsource,old_sha,641)
ancestor_manifest=verify_seal(ancestor,ancestor_sha,625)
new_manifest=verify_seal(source,manifest,643)
assert ancestor_manifest['git_revision_reference']==revision
assert new_manifest['parent_manifest_sha256']==old_sha and new_manifest['git_revision_reference']==revision and new_manifest['git_revision_inheritance']['ancestor_manifest_sha256']==ancestor_sha
mutable={'scripts/extended_ops.py','scripts/hpc/extended_cpu.lsf','docs/protenix_extended.md','scripts/extended_startup.py','tests/test_extended_startup.py'}
for name,digest in old_manifest['files'].items():
 if name not in mutable:assert new_manifest['files'][name]==digest,'Protected scientific original641 member changed'
assert len([n for n in old_manifest['files'] if n.startswith('src/')])==145
assert upload.is_file() and not upload.is_symlink() and upload.stat().st_size==39644558 and sha(upload)=='bde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef','Existing exact staged upload changed'
assert marker.is_file() and not marker.is_symlink()
assert read(marker)=={'outcome':'unknown','source_manifest_sha256':manifest,'archive_sha256':'bde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef','archive_bytes':39644558,'previous_source_manifest_sha256':old_sha,'scope':'One-shot metadata staging; CPU/native/science not passed'},'Existing staging marker differs'
parent=read(source/'docs/extended_parent.json');snapshot=parent['snapshot'];snapshot_path=Path(snapshot['path'])
assert parent['job']=='29792670' and parent['global_state']==10000 and snapshot['sha256']==snapshot_sha and snapshot['bytes']==14226
assert snapshot_path==base/'restart-lrreset-04/run-protenix/payload/study/case-0/snapshots/step-10000.npz' and snapshot_path.stat().st_size==14226 and sha(snapshot_path)==snapshot_sha,'Original parent snapshot changed'
authenticate_history();failure=preserved_failure()
for wrapper in ('extended_cpu.lsf','extended_stage.lsf'):subprocess.run(['bash','-n',str(source/'scripts/hpc'/wrapper)],check=True)
sys.dont_write_bytecode=True;sys.path[:0]=[str(source/'scripts'),str(source/'src')]
from campaign25_contract import config25
import extended_ops as ops
assert ops.ROOT==root and ops.SOURCE==source and ops.PYTHON==python and ops.STARTUP_CASES==34
assert ops.START=={'random':0,'continuation':10000} and ops.COUNTS=={'random':5,'continuation':4}
assert ops.REVISION==revision and ops.STARTUP_VERSION=='protenix-actual-command-gpu-premodel-subprocess-v2'
assert ops.sha(ops.CONTEXT_READY)==ops.CONTEXT_READY_SHA=='9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd'
for arm in ops.ARMS:config25(read(ops.config(arm)))
expected_cpu=['bsub','-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00','-J','mosaic-pro50k-cpu','-oo',str(root/'cpu/stdout.%J.log'),'-eo',str(root/'cpu/stderr.%J.log'),'bash','-l',str(source/'scripts/hpc/extended_cpu.lsf'),str(source),str(root),manifest]
assert ops.command('cpu','cpu',manifest)==expected_cpu,'Immutable actual CPU resource/source command differs'
no_new_jobs();control.mkdir(exist_ok=False)
exclusive(control/'started.json',{'outcome':'unknown','operator':'pro03-cpu-recovery-03','source_manifest_sha256':manifest,'source_files':643,'previous_failed_gate':failure,'old_failure_proofs':pins,'continuation_snapshot_sha256':snapshot_sha,'scope':'Reuse existing readonly source; no bootstrap/upload; fresh unchanged admission then only original CPU and own-ended callback'})
try:
 gate=ops.storage_gate(control,'preflight')
 assert gate['status']=='passed' and gate['reserve_bytes']==4*GIB and gate['file_reserve']==100000 and gate['full_long_admission'] is False
 assert gate['full_extended']=={'random':GIB,'continuation':GIB} and gate['missing_full_projections']==[]
 assert gate['work3']['available_bytes_conservative']>=gate['work3']['required_bytes'] and gate['work3']['available_files_lower']>=100000
 preserved_failure();authenticate_history();no_new_jobs(control_exists=True)
 # This is the original bootstrap's first jobs creation, performed only after capacity passes.
 exclusive(root/'jobs.json',{})
 cpu=ops.submit('cpu','cpu',manifest)
 assert cpu['outcome']=='submitted' and re.fullmatch(r'\d+',cpu['job_id']) and cpu['args']==expected_cpu and cpu['dependency'] is None
 dependency='ended('+cpu['job_id']+')'
 expected_callback=['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30','-J','mosaic-pro50k-after-cpu','-oo',str(root/'after-cpu/stdout.%J.log'),'-eo',str(root/'after-cpu/stderr.%J.log'),'-w',dependency,'bash','-l',str(source/'scripts/hpc/extended_stage.lsf'),str(source),str(root),manifest,'after-cpu',str(root/'after-cpu')]
 assert ops.command('after-cpu','after-cpu',manifest,dependency)==expected_callback
 callback=ops.submit('after-cpu','after-cpu',manifest,dependency)
 assert callback['outcome']=='submitted' and re.fullmatch(r'\d+',callback['job_id']) and callback['args']==expected_callback and callback['dependency']==dependency
 assert read(root/'jobs.json')=={'cpu':cpu['job_id'],'after-cpu':callback['job_id']}
 preserved_failure()
 result={'status':'submitted_not_passed','source':str(source),'root':str(root),'control':str(control),'source_manifest_sha256':manifest,'source_files':643,'cpu_cases_required':866,'cpu':cpu,'callback':callback,'preflight':{'path':str(control/'storage-preflight/gate.json'),'sha256':sha(control/'storage-preflight/gate.json'),'gate':gate},'previous_failed_gate':failure,'original_saved_results_unchanged':True,'cpu_passed':False,'gpu_startup_passed':False,'native_passed':False,'optimization_started':False}
 exclusive(control/'submission-complete.json',result);print(json.dumps(result))
except BaseException as error:
 exclusive(control/'operator-failure.json',{'status':'failed_or_unknown_preserved','error':repr(error),'source_manifest_sha256':manifest,'jobs_file_present':(root/'jobs.json').is_file(),'jobs':read(root/'jobs.json') if (root/'jobs.json').is_file() else None,'scope':'No retry: inspect actual submissions and exclusive markers; submitted jobs are never blindly duplicated'})
 raise
finally:preserved_failure()
'''

def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def local_archive(path, package):
    if (package["manifest_sha256"] != MANIFEST or package["archive_sha256"] != ARCHIVE
            or package["archive_bytes"] != ARCHIVE_BYTES or package["source_files"] != 643
            or package["parent_manifest_sha256"] != OLD_MANIFEST or package["git_revision_reference"] != REVISION):
        raise ValueError("Wrong immutable Pro03 package")
    if path.stat().st_size != ARCHIVE_BYTES or sha(path) != ARCHIVE:
        raise ValueError("Local complete archive changed")
    seen = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = member.name.removeprefix("source/"); pure = PurePosixPath(name)
            if (not member.isfile() or not member.name.startswith("source/") or name in seen
                    or pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name):
                raise ValueError("Unsafe/duplicate package member")
            seen[name] = archive.extractfile(member).read()
    raw = seen.pop("source_manifest.json"); manifest = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != MANIFEST or len(seen) != 643 or set(seen) != set(manifest["files"]):
        raise ValueError("Incomplete exact package source")
    for name, data in seen.items():
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
            raise ValueError("Local source member changed: " + name)
    if len(OLD_PINS) != 12:
        raise ValueError("Exact predecessor inventory not prepared")
    inspection = REPO / "results/hpc/20261024-boltz-startup-recovery-02/peer-projection-inspection"
    if sha(inspection / "stdout.bin") != INSPECTION_SHA:
        raise ValueError("Authenticated predecessor inventory bytes changed")
    outcome = json.loads((inspection / "result.json").read_bytes())
    if (outcome.get("returncode") != 0 or outcome.get("read_only") is not True
            or outcome.get("stdout_sha256") != INSPECTION_SHA
            or outcome.get("stderr_sha256") != sha(inspection / "stderr.bin")):
        raise ValueError("Read-only predecessor inspection did not complete authentically")
    captured = json.loads((inspection / "stdout.bin").read_bytes())
    prefix = "/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/protenix-extended-01/"
    pins = {path.removeprefix(prefix): {"bytes": item["bytes"], "sha256": item["sha256"]}
            for path, item in captured["files"].items()}
    if (pins != OLD_PINS or captured["source_manifest_sha256"] != OLD_MANIFEST
            or captured["source_files_rehashed"] != 641 or captured["projection_exists"] is not False
            or captured["segment_directories"] != []):
        raise ValueError("Exact preserved predecessor inventory scope differs")
    for item in captured["files"].values():
        data = item["text"].encode()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Captured raw predecessor text/hash differs")


def authenticate_staged_failure():
    """Check downloaded exact pre-CPU failure, never reinterpret it as passed."""
    previous=HERE/'submit_recovery02.py'
    if sha(previous)!='307ee96a42e1fa0c446ce2e7aaf44072eebc96ec37aad38561d1e0b56e5790d2':
        raise ValueError('Preserved staging operator changed')
    folder=REPO/'results/hpc'/TAG/'evidence-02'
    metadata=json.loads((folder/'metadata.json').read_bytes())
    prefix='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/'
    expected={
        prefix+'protenix-extended-03/staging-01/started.json':('files/0000/started.json',2319,'56af07542fca09fbfbcc7f78ccbab0a44c174fd5bf3956371b2bbc6ff0350ab6'),
        prefix+'protenix-extended-03/staging-01/storage-preflight/gate.json':('files/0001/gate.json',6133,FAILED_GATE_SHA),
        prefix+'protenix-extended-release-03/source/source_manifest.json':('files/0002/source_manifest.json',78898,MANIFEST)}
    if set(metadata['files'])!=set(expected) or metadata['captured_utc']!='2026-10-24T09:25:58.104007+00:00':
        raise ValueError('Wrong exact failed staging evidence')
    for remote,(relative,size,digest) in expected.items():
        record=metadata['files'][remote];path=folder/relative
        if record!={'local_path':relative,'bytes':size,'sha256':digest} or not path.is_file() or path.is_symlink() or path.stat().st_size!=size or sha(path)!=digest:
            raise ValueError('Original staged failure proof differs')
    root=prefix+'protenix-extended-03';source=prefix+'protenix-extended-release-03/source'
    if (metadata['roots'][root]['jobs']!={} or metadata['roots'][root]['jobs_file_present'] is not False
            or metadata['releases'][source]!={'all_members_rehashed':True,'manifest_sha256':MANIFEST,'readonly':True,'source_files':643}):
        raise ValueError('Not an authenticated source643/no-CPU outcome')
    gate=json.loads((folder/'files/0001/gate.json').read_bytes())
    if (gate['status']!='failed' or gate['phase']!='preflight' or gate['reserve_bytes']!=4294967296
            or gate['file_reserve']!=100000 or gate['work3']['required_bytes']!=41643669252
            or gate['work3']['available_bytes_conservative']!=27472908456 or gate['full_long_admission'] is not False):
        raise ValueError('Preserved failed capacity outcome differs')
    attempt=REPO/'results/hpc'/TAG/'attempt02'
    if json.loads((attempt/'submission-result.json').read_bytes())!={'returncode':1,'outcome':'inspect_actual_receipts'} or (attempt/'submission-stdout.bin').read_bytes()!=b'':
        raise ValueError('Wrong actual failed pre-CPU staging result')
    if not (attempt/'submission-stderr.bin').read_text().rstrip().endswith('ValueError: Whole extended admission failed'):
        raise ValueError('Historical resource-admission failure changed')



ORIGINAL_OPERATOR_SHA = '4715ca7cb9a4646f3c03a19b3cb81cb6cd3c9f4d4a1fd3da99694ec0971b31bc'
REMOTE_SHA = '97b891efb301fbe4b50a34c34bc63e09bcc7341327f334c7d6f578316c6cb696'
ORIGINAL_ARGV_SHA = '5030766f5568675292a3e352326e889fe9cace77657a8f4fcfef698bef1fe56a'
FRAME_MAGIC = b'MOSAIC-PRO03-CPU-RECOVERY-STREAM-V1\n'
BOOTSTRAP = "import hashlib,json,sys\nMAGIC=b'MOSAIC-PRO03-CPU-RECOVERY-STREAM-V1\\n'\nraw=sys.stdin.buffer.read(65537)\nif sys.flags.optimize or len(raw)>65536 or not raw.startswith(MAGIC):raise ValueError('Invalid bounded recovery transport frame')\noffset=len(MAGIC)\nif len(raw)<offset+4:raise ValueError('Truncated recovery transport header')\nlength=int.from_bytes(raw[offset:offset+4],'big');offset+=4\nif not 0<length<=2048 or len(raw)<offset+length:raise ValueError('Invalid recovery transport header size')\ndef pairs(rows):\n result={}\n for key,value in rows:\n  if key in result:raise ValueError('Duplicate transport field')\n  result[key]=value\n return result\nheader=json.loads(raw[offset:offset+length],object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)));offset+=length\nexpected={'version':'pro03-exact-cpu-recovery-framed-stdin-v1','operator_sha256':'4715ca7cb9a4646f3c03a19b3cb81cb6cd3c9f4d4a1fd3da99694ec0971b31bc','remote_bytes':13820,'remote_sha256':'97b891efb301fbe4b50a34c34bc63e09bcc7341327f334c7d6f578316c6cb696','argv_bytes':1808,'argv_sha256':'5030766f5568675292a3e352326e889fe9cace77657a8f4fcfef698bef1fe56a'}\nif header!=expected:raise ValueError('Unapproved recovery code or original argv binding')\ncode=raw[offset:offset+expected['remote_bytes']];offset+=expected['remote_bytes']\nargv_raw=raw[offset:offset+expected['argv_bytes']];offset+=expected['argv_bytes']\nif offset!=len(raw) or hashlib.sha256(code).hexdigest()!=expected['remote_sha256'] or hashlib.sha256(argv_raw).hexdigest()!=expected['argv_sha256']:raise ValueError('Truncated, appended or changed recovery frame')\nargv=json.loads(argv_raw,object_pairs_hook=pairs)\nif not isinstance(argv,list) or len(argv)!=2 or not all(isinstance(x,str) for x in argv):raise ValueError('Malformed original recovery argv')\nsys.argv=['-c',*argv]\nexec(compile(code,'<exact-original-pro03-cpu-recovery>','exec'),{'__name__':'__main__','__builtins__':__builtins__})\n"


def main():
    # This distinct caller prevents the unexecuted original's Windows command
    # length risk. It executes the EXACT original remote program/argv by stdin.
    original_operator=HERE/'submit_cpu_recovery03.py'
    if sha(original_operator)!=ORIGINAL_OPERATOR_SHA:
        raise ValueError('Preserved unexecuted original recovery operator changed')
    authenticate_staged_failure()
    original=REPO/'results/hpc'/TAG
    package=json.loads((HERE/'package.json').read_bytes())
    local_archive(original/'source01/source.tar.gz',package)
    code=REMOTE.encode('utf8')
    argv=[MANIFEST,json.dumps(OLD_PINS,separators=(',',':'),sort_keys=True)]
    argv_raw=json.dumps(argv,separators=(',',':'),ensure_ascii=True).encode('utf8')
    if (len(code)!=13820 or hashlib.sha256(code).hexdigest()!=REMOTE_SHA
            or hashlib.sha256(argv_raw).hexdigest()!=ORIGINAL_ARGV_SHA):
        raise ValueError('Original remote code/argv changed before framed transport')
    header={'version':'pro03-exact-cpu-recovery-framed-stdin-v1',
        'operator_sha256':ORIGINAL_OPERATOR_SHA,'remote_bytes':len(code),'remote_sha256':REMOTE_SHA,
        'argv_bytes':len(argv_raw),'argv_sha256':ORIGINAL_ARGV_SHA}
    header_raw=json.dumps(header,separators=(',',':'),sort_keys=True).encode('utf8')
    frame=FRAME_MAGIC+len(header_raw).to_bytes(4,'big')+header_raw+code+argv_raw
    if len(frame)>65536 or len(header_raw)>2048:
        raise ValueError('Exact framed request exceeds declared transport bounds')
    remote_args=['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',BOOTSTRAP]
    command=' '.join(shlex.quote(value) for value in remote_args)
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(command)]
    command_chars=len(subprocess.list2cmdline(args))
    if command_chars>=16000:
        raise ValueError('Short transport command unexpectedly approaches Windows limit')
    folder=original/'cpu-recovery-03-streamed-01'
    folder.mkdir(exist_ok=False)
    with (folder/'submission-started.json').open('x',encoding='utf8',newline='\n') as handle:
        json.dump({'utc':datetime.now(timezone.utc).isoformat(),'outcome':'unknown','args':args,
            'source_manifest_sha256':MANIFEST,'source_files':643,'operator_sha256':sha(Path(__file__)),
            'original_unexecuted_operator_sha256':ORIGINAL_OPERATOR_SHA,'remote_code_sha256':REMOTE_SHA,
            'original_argv_sha256':ORIGINAL_ARGV_SHA,'framed_request_bytes':len(frame),
            'framed_request_sha256':hashlib.sha256(frame).hexdigest(),'bootstrap_sha256':hashlib.sha256(BOOTSTRAP.encode()).hexdigest(),
            'windows_command_chars':command_chars,'previous_failed_gate_sha256':FAILED_GATE_SHA,
            'scope':'Transport-only distinct caller. Exact original remote guard/admission/submission program and argv; one new admission attempt, existing readonly source, no upload/cleanup. Original oversized caller never executed; CPU/native not passed.'},handle,indent=2,sort_keys=True);handle.write('\n');handle.flush()
    try:
        result=subprocess.run(args,input=frame,capture_output=True,timeout=600)
    except subprocess.TimeoutExpired as error:
        with (folder/'submission-timeout.json').open('x',encoding='utf8') as handle:
            json.dump({'outcome':'unknown','error':repr(error),'scope':'Inspect exact remote cpu-recovery-03 and actual CPU/submission markers; never replay either caller'},handle);handle.write('\n')
        if error.stdout:(folder/'submission-partial-stdout.bin').write_bytes(error.stdout)
        if error.stderr:(folder/'submission-partial-stderr.bin').write_bytes(error.stderr)
        raise
    except BaseException as error:
        with (folder/'submission-failed-unknown.json').open('x',encoding='utf8') as handle:
            json.dump({'outcome':'unknown_inspect_exact_remote_markers','error':repr(error)},handle);handle.write('\n')
        raise
    (folder/'submission-stdout.bin').write_bytes(result.stdout)
    (folder/'submission-stderr.bin').write_bytes(result.stderr)
    with (folder/'submission-result.json').open('x',encoding='utf8',newline='\n') as handle:
        json.dump({'returncode':result.returncode,'outcome':'inspect_actual_receipts',
            'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest()},handle,indent=2);handle.write('\n')
    print(result.stdout.decode(errors='replace')[-16000:]);print(result.stderr.decode(errors='replace')[-2500:])
    if result.returncode:raise RuntimeError('Preserved failed/unknown recovery; inspect actual receipts without retry')


if __name__=='__main__':main()
