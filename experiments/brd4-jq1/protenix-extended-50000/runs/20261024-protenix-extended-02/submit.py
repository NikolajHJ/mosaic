"""One-shot Pro02 metadata staging and CPU submission; operator outside seal.

Preparation and source inspection use only the standard library. Execution is
owned by the parent operator; no numerical work runs on the login node. Every
unknown/failed attempt remains recorded and must be inspected before recovery.
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
TAG = "20261024-protenix-extended-02"
MANIFEST = "f1c15dc99052a8c4764e4ea0d0e9d8ca068e1c496e12c8c76c55251233b1220e"
ARCHIVE = "e04c57d588ef3bc531902f8ccc8fbb2d634ceea78dc8cbe6640cfbb3607d6b36"
ARCHIVE_BYTES = 39638922
OLD_MANIFEST = "a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00"
REVISION = "c74c659344123937608077e3e4078f75f09cd743"
SNAPSHOT = "edf15f06ba1b6591d62cbc3e04e19ecd93b553a0627da38c370fdeed3c1b780f"
# Exact twelve already downloaded predecessor submission/LSF/stderr receipts.
# Filled from the authenticated read-only 24 October metadata inventory.
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
root=base/'protenix-extended-02';release=base/'protenix-extended-release-02';source=release/'source'
upload=base/'protenix-extended-02.tar.gz';marker=base/'protenix-extended-02-staging-started.json'
oldroot=base/'protenix-extended-01';oldsource=base/'protenix-extended-release-01/source'
ancestor=base/'restart-lrreset-release-04/source'
archive_sha,size,manifest,count,pins=(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]),json.loads(sys.argv[5]))
expected='f1c15dc99052a8c4764e4ea0d0e9d8ca068e1c496e12c8c76c55251233b1220e'
old_sha='a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
ancestor_sha='c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
revision='c74c659344123937608077e3e4078f75f09cd743'
snapshot_sha='edf15f06ba1b6591d62cbc3e04e19ecd93b553a0627da38c370fdeed3c1b780f'
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
jobs_expected={'cpu':'29798115','after-cpu':'29798116','random-native-0000':'29798159','random-after-native-0000':'29798160','continuation-native-0000':'29798161','continuation-after-native-0000':'29798162'}

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
  else:assert 'Producer failed' in error,'Original own-producer failure changed'
 cpu=oldroot/'cpu-certificate.json'
 assert sha(cpu)=='c0f784d523d358dfd09d7f57fcfe333fbb5c069d93074681ea35562a5f423171'
 c=read(cpu);assert c['status']=='passed' and c['source_manifest_sha256']==old_sha and c['cases']==832,'Original CPU success must remain successful'
 for proof in c['proofs']:assert sha(proof['path'])==proof['sha256'],'Original passed CPU proof changed'

assert manifest==expected and count==643 and size==39638922 and archive_sha=='e04c57d588ef3bc531902f8ccc8fbb2d634ceea78dc8cbe6640cfbb3607d6b36'
assert not any(os.path.lexists(p) for p in (root,release,upload,marker)),'Inspect prior staging/unknown outcome; never replay'
old_manifest=verify_seal(oldsource,old_sha,641)
ancestor_manifest=verify_seal(ancestor,ancestor_sha,625)
assert ancestor_manifest['git_revision_reference']==revision,'Revision is not authenticated original625 metadata'
authenticate_history()
exclusive(marker,{'outcome':'unknown','source_manifest_sha256':manifest,'archive_sha256':archive_sha,'archive_bytes':size,'previous_source_manifest_sha256':old_sha,'scope':'One-shot metadata staging; CPU/native/science not passed'})
with upload.open('xb') as f:
 for chunk in iter(lambda:sys.stdin.buffer.read(1048576),b''):f.write(chunk)
assert upload.stat().st_size==size and sha(upload)==archive_sha,'New archive transfer differs'
release.mkdir();source.mkdir();seen=set()
with tarfile.open(upload) as tar:
 for member in tar:
  assert member.isfile() and member.name.startswith('source/'),'Nonregular/foreign archive member'
  name=member.name.removeprefix('source/');pure=PurePosixPath(name);path=source/name
  assert name not in seen and not pure.is_absolute() and '..' not in pure.parts and '\\' not in name and ':' not in name and path.resolve().is_relative_to(source.resolve()),'Unsafe/duplicate archive member'
  seen.add(name);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(tar.extractfile(member).read())
assert sha(source/'source_manifest.json')==manifest
new_manifest=read(source/'source_manifest.json');assert len(new_manifest['files'])==count and seen==set(new_manifest['files'])|{'source_manifest.json'}
assert new_manifest['parent_manifest_sha256']==old_sha and new_manifest['git_revision_reference']==revision and new_manifest['git_revision_inheritance']['ancestor_manifest_sha256']==ancestor_sha
mutable={'scripts/extended_ops.py','scripts/hpc/extended_cpu.lsf','docs/protenix_extended.md','scripts/extended_startup.py','tests/test_extended_startup.py'}
for name,digest in new_manifest['files'].items():
 path=source/name;assert sha(path)==digest
 if name.endswith('.py'):ast.parse(path.read_bytes(),filename=name)
for name,digest in old_manifest['files'].items():
 if name not in mutable:assert new_manifest['files'][name]==digest,'Protected original scientific/driver/config/protocol/check/parent member changed'
assert len([name for name in old_manifest['files'] if name.startswith('src/')])==145
for wrapper in ('extended_cpu.lsf','extended_stage.lsf'):subprocess.run(['bash','-n',str(source/'scripts/hpc'/wrapper)],check=True)
parent=read(source/'docs/extended_parent.json');snapshot=parent['snapshot'];snapshot_path=Path(snapshot['path'])
assert parent['job']=='29792670' and parent['global_state']==10000 and snapshot['sha256']==snapshot_sha and snapshot['bytes']==14226
assert snapshot_path==base/'restart-lrreset-04/run-protenix/payload/study/case-0/snapshots/step-10000.npz' and snapshot_path.stat().st_size==14226 and sha(snapshot_path)==snapshot_sha,'Exact original continuation snapshot changed'
for path in source.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
source.chmod(0o555);verify_seal(source,manifest,count)
sys.dont_write_bytecode=True;sys.path[:0]=[str(source/'scripts'),str(source/'src')]
from campaign25_contract import config25
import extended_ops as ops
assert ops.ROOT==root and ops.SOURCE==source and ops.STARTUP_CASES==28 and ops.START=={'random':0,'continuation':10000} and ops.COUNTS=={'random':5,'continuation':4}
for arm in ops.ARMS:config25(read(ops.config(arm)))
authenticate_history()
root.mkdir();control=root/'staging-01';control.mkdir()
exclusive(control/'started.json',{'outcome':'unknown','source_manifest_sha256':manifest,'old_source_manifest_sha256':old_sha,'old_failure_proofs':pins,'continuation_snapshot_sha256':snapshot_sha})
gate=ops.storage_gate(control,'preflight')
assert gate['status']=='passed' and gate['reserve_bytes']==4*1024**3 and gate['file_reserve']==100000 and gate['full_long_admission'] is False
assert gate['work3']['available_bytes_conservative']>=gate['work3']['required_bytes'] and gate['work3']['available_files_lower']>=100000
exclusive(root/'source-staging.json',{'status':'passed','files':count,'manifest':manifest,'archive_sha256':archive_sha,'archive_bytes':size,'readonly':True,'original_scientific_results_unchanged':True,'original_pre_model_failures_preserved':True,'old_failure_proofs':pins,'continuation_snapshot_sha256':snapshot_sha})
exclusive(root/'jobs.json',{})
cpu=ops.submit('cpu','cpu',manifest)
callback=ops.submit('after-cpu','after-cpu',manifest,'ended('+cpu['job_id']+')')
result={'status':'submitted_not_passed','source':str(source),'root':str(root),'manifest':manifest,'cpu':cpu,'callback':callback,'preflight':gate,'native_passed':False,'optimization_started':False}
exclusive(control/'submission-complete.json',result)
print(json.dumps(result))
'''


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_archive(path, package):
    if (package["manifest_sha256"] != MANIFEST or package["archive_sha256"] != ARCHIVE
            or package["archive_bytes"] != ARCHIVE_BYTES or package["source_files"] != 643
            or package["parent_manifest_sha256"] != OLD_MANIFEST or package["git_revision_reference"] != REVISION):
        raise ValueError("Wrong immutable Pro02 package")
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


def main():
    folder = REPO / "results/hpc" / TAG
    package = json.loads((HERE / "package.json").read_bytes())
    archive = folder / "source01/source.tar.gz"
    local_archive(archive, package)
    marker = folder / "submission-started.json"
    if any(path.exists() for path in (marker, folder / "submission-result.json", folder / "submission-timeout.json")):
        raise FileExistsError("Inspect the preserved exact outcome; never replay this operator")
    remote_args = ["/work3/s234818/mosaic/envs/protenix-tiny/bin/python", "-c", REMOTE, ARCHIVE,
                   str(ARCHIVE_BYTES), MANIFEST, "643", json.dumps(OLD_PINS, separators=(",", ":"), sort_keys=True)]
    command = " ".join(shlex.quote(value) for value in remote_args)
    args = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1",
            "dtu-hpc", "bash -l -c " + shlex.quote(command)]
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"utc": datetime.now(timezone.utc).isoformat(), "outcome": "unknown", "args": args,
                   "source_manifest_sha256": MANIFEST, "archive_sha256": ARCHIVE, "archive_bytes": ARCHIVE_BYTES,
                   "operator_sha256": sha(Path(__file__)), "scope": "One authorized connection; no blind retry"}, handle, indent=2)
        handle.write("\n")
    try:
        result = subprocess.run(args, input=archive.read_bytes(), capture_output=True, timeout=600)
    except subprocess.TimeoutExpired as error:
        (folder / "submission-timeout.json").write_text(json.dumps({"outcome": "unknown", "error": repr(error)}) + "\n")
        if error.stdout: (folder / "submission-partial-stdout.bin").write_bytes(error.stdout)
        if error.stderr: (folder / "submission-partial-stderr.bin").write_bytes(error.stderr)
        raise
    (folder / "submission-stdout.bin").write_bytes(result.stdout)
    (folder / "submission-stderr.bin").write_bytes(result.stderr)
    with (folder / "submission-result.json").open("x", encoding="utf-8") as handle:
        json.dump({"returncode": result.returncode, "outcome": "inspect_actual_receipts"}, handle); handle.write("\n")
    print(result.stdout.decode(errors="replace")[-14000:])
    print(result.stderr.decode(errors="replace")[-2500:])
    if result.returncode:
        raise RuntimeError("Preserved failed/unknown outcome; inspect exact remote paths before recovery")


if __name__ == "__main__":
    main()
