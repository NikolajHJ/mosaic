"""Standard-library transport for the independent DiffSBDD study.

This local entry point never imports numerical packages. Remote probes inspect
metadata/bytes only; allocated operators own all tests, fitting and inference.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import io
import inspect
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1", "dtu-hpc"]
REMOTE_PYTHON = "/work3/s234818/latentmol/env/boltz311/bin/python"
CONFIG = ROOT / "configs/diffsbdd_vae_alternation_v1.json"
STAGES = ("cpu", "native", "scorer", "fit", "audit", "smoke", "pilot", "followup")
CAMPAIGN = re.compile(r"diffsbdd-vae-alternation-v1-attempt[0-9]{3,}\Z")
STAGE_CAPS = {
    "cpu": {"queue": "hpc", "slots": 4, "wall_minutes": 45, "internal_seconds": 2400},
    "native": {"queue": "gpua100", "slots": 4, "wall_minutes": 60, "internal_seconds": 3300},
    "scorer": {"queue": "gpua100", "slots": 4, "wall_minutes": 120, "internal_seconds": 6600},
    "fit": {"queue": "gpua100", "slots": 4, "wall_minutes": 240, "internal_seconds": 13800},
    "audit": {"queue": "gpua100", "slots": 4, "wall_minutes": 120, "internal_seconds": 6600},
    "smoke": {"queue": "gpua100", "slots": 4, "wall_minutes": 240, "internal_seconds": 13800},
    "pilot": {"queue": "gpua100", "slots": 4, "wall_minutes": 1440, "internal_seconds": 82800},
    "followup": {"queue": "gpua100", "slots": 4, "wall_minutes": 1440, "internal_seconds": 82800},
}


def configuration(path=CONFIG):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    name = value.get("campaign", "")
    if (value.get("version") != "diffsbdd-vae-alternation-protocol-v1" or not CAMPAIGN.fullmatch(name)
            or value.get("run_root") != "/work3/s234818/latentmol/runs/" + name
            or value.get("release_root") != "/work3/s234818/latentmol/releases/" + name + "/source"):
        raise ValueError("Independent experiment namespace/config mismatch")
    storage = value.get("storage", {})
    if (storage.get("own_initial_cap_bytes") != 2 * 1024**3
            or storage.get("additional_unreserved_headroom_bytes") != 10 * 1024**3
            or storage.get("never_reduce_existing_forecasts_or_reserves") is not True):
        raise ValueError("Storage caps/headroom/reserves differ from the sealed v1 protocol")
    if value.get("stages") != STAGE_CAPS or value.get("runtime_forecast_factor") != 1.5:
        raise ValueError("Allocation caps or runtime forecast factor differ from sealed v1")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_archive_member(name):
    """Only literal POSIX source/ members, never source-evil or special paths."""
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name):
        raise ValueError("Unsafe archive path: " + repr(name))
    parts = name.split("/")
    if (parts[0] != "source" or any(part in ("", ".", "..") for part in parts)
            or PurePosixPath(name).is_absolute()):
        raise ValueError("Archive member escapes the exact source/ namespace: " + name)
    return "/".join(parts[1:])


def inspect_source_archive(archive_path, expected, capacity_bytes):
    """Bound/hash the entire source envelope before any extraction or upload."""
    archive_path = Path(archive_path)
    if (not isinstance(capacity_bytes, int) or capacity_bytes <= 0
            or archive_path.stat().st_size > capacity_bytes):
        raise ValueError("Source archive exceeds the sealed own-artifact cap")
    if sha(archive_path) != expected["archive_sha256"]:
        raise ValueError("Source archive SHA256 mismatch")
    files, total_bytes, manifest_raw, directories = {}, 0, None, []
    seen_names = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_archive_member(member.name)
            if member.name in seen_names:
                raise ValueError("Duplicate source archive member")
            seen_names.add(member.name)
            if len(seen_names) > 4 * (expected["source_members"] + 1) + 128:
                raise ValueError("Source archive exceeds its bounded metadata member cap")
            if not (member.isdir() or member.isreg()) or member.size < 0:
                raise ValueError("Source archive contains a link/device/FIFO/special member")
            if member.isdir():
                if member.size:
                    raise ValueError("Directory archive member has a nonzero size")
                directories.append(relative)
                continue
            if not relative:
                raise ValueError("The source root must be a directory")
            total_bytes += member.size
            if total_bytes > capacity_bytes or len(files) >= expected["source_members"] + 1:
                raise ValueError("Unpacked source archive exceeds its sealed byte/member cap")
            content = archive.extractfile(member)
            if content is None:
                raise ValueError("Missing regular archive member payload")
            digest, actual_size, parts = hashlib.sha256(), 0, []
            for block in iter(lambda: content.read(4 * 1024 * 1024), b""):
                actual_size += len(block)
                digest.update(block)
                if relative == "source-manifest.json":
                    parts.append(block)
            if actual_size != member.size:
                raise ValueError("Truncated source archive member")
            files[relative] = digest.hexdigest()
            if relative == "source-manifest.json":
                manifest_raw = b"".join(parts)
    if manifest_raw is None or files.get("source-manifest.json") != expected["source_manifest_sha256"]:
        raise ValueError("Source archive manifest SHA256 mismatch")
    manifest = json.loads(manifest_raw)
    members = manifest.get("files")
    if manifest.get("version") != "diffsbdd-source-manifest-v1" or not isinstance(members, dict):
        raise ValueError("Unsupported source archive manifest")
    for relative in members:
        if not relative or safe_archive_member("source/" + relative) != relative:
            raise ValueError("Unsafe source manifest path")
    expected_files = {**members, "source-manifest.json": expected["source_manifest_sha256"]}
    if files != expected_files or len(members) != expected["source_members"]:
        raise ValueError("Source archive inventory/hashes differ from its sealed manifest")
    if any(directory and not any(name.startswith(directory + "/") for name in files)
           for directory in directories):
        raise ValueError("Source archive contains an unmanifested empty directory")
    if files.get("configs/protocol.json") != expected["config_sha256"]:
        raise ValueError("Source archive config hash differs from its seal")
    if "source_bytes" in expected and total_bytes != expected["source_bytes"]:
        raise ValueError("Source unpacked bytes differ from their local seal")
    if total_bytes + archive_path.stat().st_size > capacity_bytes:
        raise ValueError("Source plus retained archive exceed the own-artifact cap")
    return {"regular_files": len(files), "unpacked_bytes": total_bytes,
            "archive_bytes": archive_path.stat().st_size}


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def remote_script(script, output, *, timeout=90):
    command = SSH + ["bash -lc " + shlex.quote(REMOTE_PYTHON + " -B -")]
    before = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        run = subprocess.run(command, input=script, text=True, capture_output=True, timeout=timeout)
        result = dict(command=command, started_utc=before, returncode=run.returncode,
                      stdout=run.stdout, stderr=run.stderr)
    except subprocess.TimeoutExpired as error:
        result = dict(command=command, started_utc=before, returncode=None,
                      status="transport_timeout_unknown_remote_outcome",
                      stdout=str(error.stdout or ""), stderr=str(error.stderr or ""))
        save_new(output, result)
        raise RuntimeError("Transport timed out; preserved receipt: " + str(output)) from error
    save_new(output, result)
    if run.returncode:
        raise RuntimeError("Remote operation failed; preserved receipt: " + str(output))
    return json.loads(run.stdout)


def probe(output):
    script = '''import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path('/work3/s234818/latentmol')
assets={
 'vae39':root/'runs/source138-vae-cli-startup-fix-attempt001/segments/003/20261026T040252Z_chemistry-train_40cdfc66/epoch_0039.pt',
 'dataset_manifest':root/'data/source095-moses-triangle-v2/manifest.json',
 'boltz_affinity':root/'cache/boltz2/boltz2_aff.ckpt',
 'boltz_confidence':root/'cache/boltz2/boltz2_conf.ckpt'}
result={'observation_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'read-only metadata and bytes; no model imports or job mutation','assets':{},'commands':{}}
for key,path in assets.items():
 row={'path':str(path),'exists':path.is_file()}
 if row['exists']:
  row['size_bytes']=path.stat().st_size
  if key in ('vae39','dataset_manifest'):
   h=hashlib.sha256()
   with path.open('rb') as f:
    for block in iter(lambda:f.read(4194304),b''):h.update(block)
   row['sha256']=h.hexdigest()
 result['assets'][key]=row
for key,argv in [('work_quota',['getquota_work3.sh']),('home_quota',['getquota_zhome.sh']),
 ('own_jobs',['bjobs','-a','-noheader','-o','jobid stat queue job_name','-u','s234818'])]:
 p=subprocess.run(argv,text=True,capture_output=True,timeout=35)
 result['commands'][key]={'argv':argv,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
result['env_candidates']={str(p):p.exists() for p in [root/'env/boltz311/bin/python',root/'env/ae/bin/python',root/'env/diffsbdd/bin/python',Path('/work3/s234818/mosaic/envs/protenix-tiny/bin/python')]}
print(json.dumps(result))
'''
    return remote_script(script, output, timeout=150)


def runtime_probe(output):
    script = '''import importlib.metadata as m,json,subprocess,sysconfig
from pathlib import Path
names=['torch','pytorch-lightning','lightning','torch-scatter','rdkit','openbabel-wheel','openbabel','scipy','numpy','boltz','pytest']
versions={}
for name in names:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:versions[name]=None
p=subprocess.run(['bjobs','-l','29829679'],capture_output=True,text=True,timeout=25)
result={'versions':versions,'site_packages':sysconfig.get_paths()['purelib'],'active_cpu_job':p.stdout}
root=Path('/work3/s234818/mosaic/submissions')
result['recent_controls']=[str(p) for p in sorted(root.glob('*'),key=lambda p:p.stat().st_mtime,reverse=True)[:8]] if root.exists() else []
print(json.dumps(result))
'''
    return remote_script(script, output, timeout=90)


def reservations(output):
    script = '''import json,subprocess,shutil
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
result={'files':{},'commands':{}}
for folder in [base/'nonchem-endpoint-release-08/source',base/'nonchem-endpoint-continuation-08']:
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.py') and p.stat().st_size<800000:
   raw=p.read_text(errors='replace')
   if any(s in raw.lower() for s in ('reserved','headroom','storage_admission','storage-admission')):
    result['files'][str(p)]=raw
for key,argv in [('python',['bash','-lc','command -v python3.10; command -v uv; command -v apptainer; command -v conda']),('quota',['getquota_work3.sh'])]:
 p=subprocess.run(argv,text=True,capture_output=True,timeout=35)
 result['commands'][key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(result))
'''
    return remote_script(script, output, timeout=150)


def admission(output, config_path=CONFIG):
    """Read-only conservative admission; never reclaim another campaign's space."""
    cfg = configuration(config_path)
    policy = cfg["storage"]
    if (policy["own_initial_cap_bytes"] != 2 * 1024**3
            or policy["additional_unreserved_headroom_bytes"] != 10 * 1024**3
            or policy["never_reduce_existing_forecasts_or_reserves"] is not True):
        raise ValueError("Storage policy may not waive the planned cap/reserves")
    script = '''import datetime,hashlib,json,re,subprocess
from pathlib import Path
gib=1024**3
p=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-endpoint-continuation-08/after-cpu/continuation-storage-entry.json')
raw=p.read_bytes(); prior=json.loads(raw)
if prior.get('status')!='passed':raise RuntimeError('Existing whole-campaign capacity is not passed')
quota=subprocess.run(['getquota_work3.sh'],text=True,capture_output=True,timeout=40)
if quota.returncode:raise RuntimeError(quota.stderr)
pools=[]
for line in quota.stdout.splitlines():
 match=re.search(r's234818\\|306121\\|\\|\\s*([0-9.]+) GiB\\|\\s*([0-9.]+) GiB',line)
 if match:
  used,hard=map(float,match.groups())
  # Rounded report is conservatively widened by one displayed unit per bound.
  pools.append({'used_gib_display':used,'hard_gib_display':hard,'available_bytes_lower':int((hard-used-0.02)*gib)})
if not pools:raise RuntimeError('No finite quota pool authenticated')
fresh=min(x['available_bytes_lower'] for x in pools)
available=min(fresh,int(prior['available_bytes']))
other=int(prior['required_bytes']); own=2*gib; additional=10*gib
required=other+own+additional
result={'version':'diffsbdd-storage-admission-v1','observation_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'read-only metadata and byte accounting; no staging, environment mutation or jobs',
 'prior_capacity_path':str(p),'prior_capacity_sha256':hashlib.sha256(raw).hexdigest(),
 'prior_capacity':prior,'quota_command':['getquota_work3.sh'],'quota_stdout':quota.stdout,
 'finite_pools':pools,'fresh_available_bytes_lower':fresh,'available_bytes_conservative':available,
 'other_campaign_forecasts_and_reserves_bytes':other,'own_initial_cap_bytes':own,
 'additional_unreserved_headroom_bytes':additional,'required_bytes':required,
 'shortfall_bytes':max(0,required-available),'passed':available>=required,
 'other_reserves_changed':False,'jobs_submitted':False,'runtime_modified':False}
print(json.dumps(result))
'''
    script = script.replace("p=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-endpoint-continuation-08/after-cpu/continuation-storage-entry.json')",
                            "p=Path(" + repr(policy["other_campaign_capacity_path"]) + ")")
    script = script.replace("other=int(prior['required_bytes']); own=2*gib; additional=10*gib",
        "other=int(prior['required_bytes']); own=" + str(policy["own_initial_cap_bytes"])
        + "; additional=" + str(policy["additional_unreserved_headroom_bytes"]))
    return remote_script(script, output, timeout=120)


def prepare(config_path, assets_path, output):
    """Seal exact local source bytes; no model imports or HPC mutation."""
    cfg = configuration(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    assets_path = Path(assets_path)
    assets = json.loads((assets_path / "assets.json").read_text())
    if (assets["checkpoint_sha256"] != cfg["upstream"]["checkpoint_sha256"]
            or assets["source_manifest_sha256"] != cfg["upstream"]["source_manifest_sha256"]):
        raise ValueError("Local assets differ from the sealed protocol")
    paths = list((ROOT / "src").rglob("*"))
    paths += list((ROOT / "tests").glob("test_diffsbdd_*.py"))
    paths += [ROOT / "tests/test_decoder_features.py", ROOT / "tests/test_refined_decoder.py",
              ROOT / "pyproject.toml", ROOT / "scripts/diffsbdd_assets.py",
              ROOT / "scripts/diffsbdd_campaign.py", ROOT / "scripts/diffsbdd_static_review.py",
              ROOT / "scripts/hpc/diffsbdd_operator.py"]
    paths += list((ROOT / "experiments/diffsbdd-vae-alternation/hpc").glob("*"))
    for path in paths:
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("No symlinks in the source release")
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    target = source / "configs/protocol.json"
    target.parent.mkdir()
    shutil.copyfile(config_path, target)
    # Keep the repository's named protocol too: allocated controller fixtures
    # exercise the same canonical CONFIG path as the local CLI.
    shutil.copyfile(config_path, target.parent / "diffsbdd_vae_alternation_v1.json")
    shutil.copytree(ROOT / "experiments/diffsbdd-vae-alternation/inputs", source / "inputs")
    shutil.copytree(assets_path, source / "assets")
    files = {path.relative_to(source).as_posix():sha(path) for path in source.rglob("*") if path.is_file()}
    save_new(source / "source-manifest.json", {"version":"diffsbdd-source-manifest-v1",
        "branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "files":files})
    archive = output / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="source")
    summary = dict(version="diffsbdd-local-seal-v1",campaign=cfg["campaign"],
        config_sha256=sha(target),source_manifest_sha256=sha(source / "source-manifest.json"),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        source_bytes=sum(path.stat().st_size for path in source.rglob("*") if path.is_file()),
        source_members=len(files),configuration=str(Path(config_path).resolve()),
        numerical_tests_run=False,models_loaded=False,jobs_submitted=False)
    inspect_source_archive(archive, summary, cfg["storage"]["own_initial_cap_bytes"])
    save_new(output / "seal.json", summary)
    return summary


def stage(config_path, seal_path, output):
    cfg = configuration(config_path)
    seal_path = Path(seal_path)
    seal = json.loads((seal_path / "seal.json").read_text())
    if (seal["campaign"] != cfg["campaign"] or seal["config_sha256"] != sha(config_path)
            or seal["archive_sha256"] != sha(seal_path / "source.tar.gz")):
        raise ValueError("Sealed bytes/config changed")
    inspect_source_archive(seal_path / "source.tar.gz", seal,
                           cfg["storage"]["own_initial_cap_bytes"])
    admitted = admission(output.with_name(output.stem + "-admission.json"), config_path)
    if admitted["passed"] is not True:
        result = dict(version="diffsbdd-stage-transport-v1",status="blocked_storage_before_any_remote_write",
            passed=False,admission=admitted,seal=seal,jobs_submitted=False,remote_writes=False)
        save_new(output, result)
        return result
    remote_parent = cfg["release_root"].removesuffix("/source")
    # Only after admission may this independent remote namespace be created.
    remote_script("from pathlib import Path\nimport json\np=Path(" + repr(remote_parent)
        + ")\np.mkdir(parents=True,exist_ok=False)\nprint(json.dumps({'created':str(p)}))\n",
        output.with_name(output.stem + "-mkdir.json"))
    argv = ["scp","-o","BatchMode=yes","-o","ConnectTimeout=15",str(seal_path / "source.tar.gz"),
            "dtu-hpc:" + remote_parent + "/source.tar.gz"]
    transfer = subprocess.run(argv,text=True,capture_output=True,timeout=180)
    save_new(output.with_name(output.stem + "-scp.json"),dict(command=argv,returncode=transfer.returncode,
             stdout=transfer.stdout,stderr=transfer.stderr))
    transfer.check_returncode()
    script = ("import hashlib,json,tarfile\nfrom pathlib import Path,PurePosixPath\n"
        + inspect.getsource(sha) + "\n" + inspect.getsource(safe_archive_member) + "\n"
        + inspect.getsource(inspect_source_archive) + "\n"
        + "parent=Path(" + repr(remote_parent) + ")\nexpected=" + repr(seal) + "\n"
        + "capacity=" + str(cfg["storage"]["own_initial_cap_bytes"]) + "\n") + '''archive=parent/'source.tar.gz'
if parent.is_symlink() or parent.resolve().parent!=Path('/work3/s234818/latentmol/releases').resolve():raise RuntimeError('Independent release namespace escaped')
if (parent/'source').exists():raise RuntimeError('Source already staged; preserve attempt')
archive_receipt=inspect_source_archive(archive,expected,capacity)
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  if not (parent/member.name).resolve().is_relative_to(parent.resolve()):raise RuntimeError('Extraction target escapes release parent')
 tar.extractall(parent)
source=parent/'source'; manifest=source/'source-manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected['source_manifest_sha256']:raise RuntimeError('Manifest SHA mismatch')
files=json.loads(manifest.read_text())['files']
for name,value in files.items():
 path=source/name
 if not path.resolve().is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=value:raise RuntimeError('Staged member mismatch:'+name)
actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
if actual!=set(files)|{'source-manifest.json'}:raise RuntimeError('Staged inventory differs')
if hashlib.sha256((source/'configs/protocol.json').read_bytes()).hexdigest()!=expected['config_sha256']:raise RuntimeError('Staged config mismatch')
result={'version':'diffsbdd-authenticated-stage-v1','passed':True,'seal':expected,'source':str(source),'archive_validation':archive_receipt}
with (parent/'staged.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''
    return remote_script(script, output, timeout=120)


def status(config_path, output):
    cfg = configuration(config_path)
    script = "import json,subprocess\nfrom pathlib import Path\nroot=Path(" + repr(cfg["run_root"]) + ")\n" + '''result={'scope':'read-only own experiment status','files':{}}
if root.exists():
 for p in root.rglob('*.json'):
  if p.name in ('receipt.json','stage.json') or p.name.startswith('submission_'):
   if p.stat().st_size<200000:result['files'][str(p)]=json.loads(p.read_text())
p=subprocess.run(['bjobs','-a','-noheader','-o','jobid stat queue job_name','-u','s234818'],text=True,capture_output=True,timeout=30)
result['own_jobs_output']=p.stdout
print(json.dumps(result))
'''
    return remote_script(script, output)


def submit(config_path, seal_path, stage_name, output):
    cfg = configuration(config_path)
    if stage_name not in STAGES:
        raise ValueError("Unknown stage")
    seal = json.loads((Path(seal_path) / "seal.json").read_text())
    if seal["config_sha256"] != sha(config_path):
        raise ValueError("Submission config differs from the sealed bytes")
    admitted = admission(output.with_name(output.stem + "-admission.json"), config_path)
    if admitted["passed"] is not True:
        result = dict(version="diffsbdd-submission-v1",status="blocked_storage_before_submission",
                      passed=False,stage=stage_name,jobs_submitted=False,admission=admitted)
        save_new(output,result)
        return result
    script = "import importlib.util,json,subprocess,sys,re\nfrom pathlib import Path\n" + "cfg=" + repr(cfg) + "\nseal=" + repr(seal) + "\nstage=" + repr(stage_name) + "\n" + '''source=Path(cfg['release_root']); root=Path(cfg['run_root'])
spec=importlib.util.spec_from_file_location('operator',source/'scripts/hpc/diffsbdd_operator.py')
op=importlib.util.module_from_spec(spec); spec.loader.exec_module(op)
identity=op.authenticate_source(source,source/'source-manifest.json',seal['source_manifest_sha256'],source/'configs/protocol.json',seal['config_sha256'])
identity.update(run_root=str(root),python=cfg['runtime'])
if stage!='cpu':op.predecessors(root,stage,identity)
if stage in ('audit','smoke','pilot','followup'):
 previous=op.STAGES[op.STAGES.index(stage)-1]
 record=json.loads((root/previous/'stage.json').read_text())
 forecast=record.get('successor_forecast',{})
 if stage=='followup' and not forecast:
  bootstrap=cfg.get('followup',{}).get('bootstrap_forecast',{})
  required_bootstrap={'mode':'bounded_first_case_then_measured_forecast','initial_case_seconds':3300,'factor':1.5,'retain_all_prespecified_memberships':True}
  if bootstrap!=required_bootstrap:raise RuntimeError('No sealed bounded-first-case followup forecast policy')
 else:
  if forecast.get('stage')!=stage or forecast.get('factor')!=1.5 or forecast.get('passed') is not True:
   raise RuntimeError('No authenticated measured successor forecast; stop/repartition before execution')
  if not 0<forecast['seconds']<=cfg['stages'][stage]['internal_seconds']:raise RuntimeError('Forecast exceeds internal cap')
root.mkdir(parents=True,exist_ok=True)
intent=root/('submission_'+stage+'_intent.json'); receipt=root/('submission_'+stage+'.json')
if intent.exists() or receipt.exists() or (root/stage).exists():raise RuntimeError('Stage already attempted; no ambiguous retry')
budget=cfg['stages'][stage]
cmd=['bsub','-q',budget['queue'],'-n',str(budget['slots']),'-W',str(budget['wall_minutes']),'-J','diffsbdd-'+stage,
 '-R','span[hosts=1]','-R','rusage[mem=12000]','-o',str(root/(stage+'.lsf.log'))]
if stage!='cpu':cmd+=['-gpu','num=1:mode=exclusive_process:gmem=80G']
operator=[cfg['runtime'],str(source/'scripts/hpc/diffsbdd_operator.py'),'run','--stage',stage,
 '--source-root',str(source),'--source-manifest',str(source/'source-manifest.json'),
 '--source-manifest-sha256',seal['source_manifest_sha256'],'--config',str(source/'configs/protocol.json'),
 '--config-sha256',seal['config_sha256'],'--run-root',str(root),'--python',cfg['runtime'],
 '--internal-seconds',str(budget['internal_seconds'])]
if stage=='cpu':
 cmd+=['bash',str(source/'experiments/diffsbdd-vae-alternation/hpc/bootstrap.sh'),str(source),*operator]
else:cmd+=operator
with intent.open('x') as f:json.dump({'command':cmd,'identity':identity},f,indent=2)
process=subprocess.run(cmd,text=True,capture_output=True,timeout=45)
result={'version':'diffsbdd-submission-v1','stage':stage,'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr,'command':cmd,'passed':False}
match=re.search(r'Job <([0-9]+)>',process.stdout)
if process.returncode==0 and match:result.update(job_id=int(match.group(1)),status='submitted_not_passed',jobs_submitted=True)
else:result.update(status='submission_failed_or_unknown_do_not_retry',jobs_submitted=False)
with receipt.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
'''
    return remote_script(script, output, timeout=90)


def collect(config_path, output):
    cfg = configuration(config_path)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    # Only this campaign's own source and run receipts are ever retrieved.
    argv = ["scp","-r","-o","BatchMode=yes","-o","ConnectTimeout=15",
            "dtu-hpc:" + cfg["run_root"],str(output / "run")]
    run = subprocess.run(argv,text=True,capture_output=True,timeout=300)
    result = dict(version="diffsbdd-collection-v1",command=argv,returncode=run.returncode,
                  stdout=run.stdout,stderr=run.stderr,remote_mutations=False)
    save_new(output / "collection.json",result)
    run.check_returncode()
    return result


def report(directory, output):
    """Saved-only summary; absence of a pass never means a completed stage."""
    directory = Path(directory)
    records = {}
    for name in STAGES:
        path = directory / name / "stage.json"
        records[name] = json.loads(path.read_text()) if path.is_file() else {"status":"not_reached","passed":False}
    result = dict(version="diffsbdd-saved-only-report-v1",stages=records,
                  numerical_work=False,all_stages_passed=all(row.get("passed") is True for row in records.values()))
    save_new(output,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "runtime-probe", "reservations", "admission",
        "prepare", "stage", "submit", "status", "collect", "report"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--assets", type=Path, default=ROOT / "outputs/diffsbdd-assets/attempt001")
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/hpc-submissions/diffsbdd-vae-alternation" / ("probe-" + str(time.time_ns()) + ".json")
    if args.action in ("prepare","collect") and args.output is None:
        parser.error("--output must explicitly name a new local directory")
    if args.action in ("stage","submit") and args.seal is None:
        parser.error("--seal is required")
    if args.action == "submit" and args.stage is None:
        parser.error("--stage is required")
    if args.action == "report" and args.input is None:
        parser.error("--input is required")
    if args.action in ("probe","runtime-probe","reservations"):
        result = {"probe":probe,"runtime-probe":runtime_probe,"reservations":reservations}[args.action](output)
    elif args.action == "admission":
        result = admission(output,args.config)
    elif args.action == "prepare":
        result = prepare(args.config,args.assets,output)
    elif args.action == "stage":
        result = stage(args.config,args.seal,output)
    elif args.action == "submit":
        result = submit(args.config,args.seal,args.stage,output)
    elif args.action == "status":
        result = status(args.config,output)
    elif args.action == "collect":
        result = collect(args.config,output)
    else:
        result = report(args.input,output)
    if args.action not in ("prepare","collect"):
        save_new(output.with_name(output.stem + "-parsed.json"), result)
    compact = {key:value for key,value in result.items() if key not in {"files","prior_capacity","quota_stdout"}}
    print(json.dumps({"saved":str(output), **compact}, indent=2))
    if result.get("passed") is False and args.action in ("admission","stage","submit"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
