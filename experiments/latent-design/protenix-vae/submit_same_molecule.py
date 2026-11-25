"""Seal and launch the disjoint two-parent and ten-parent Protenix shards."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import uuid


EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
REL = EXP.relative_to(ROOT).as_posix()
TAG = "protenix-same-molecule-597604-20261122a"
BASE = "/work3/s234818/latentmol"
CHECKPOINT = BASE + "/runs/vae-e48-r7-motifs-20261121a/r7/checkpoint-000597604.pt"
CHECKPOINT_SHA = "abfec8c32ab7d6e00240ee7d68492c7e271f4aa95f14be874674e1fe0cd742f4"
TORCH = BASE + "/env/boltz311/bin/python"
STUDY = "same-molecule-597604.json"
FRESH = "same-molecule-597604-fresh-dtu.json"
PLAN = "paired-gpu-plan.json"
SHARDS = {"first": (0, 2, 15), "remaining": (2, 12, 120)}
FILES = ("source.tar.gz", "hpc_launch.py", "vae_segment_chain.py", "job.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def locations(shard):
    name = "protenix-temperature-600116-20261123a" if shard == "temperature" else TAG + "-" + shard
    return (ROOT / "outputs/hpc-submissions" / name,
            "/zhome/95/e/206121/dbmol-control/" + name, BASE + "/runs/" + name)


def seal(shard, panel):
    local, control, output = locations(shard)
    if shard == "temperature":
        study_file, fresh_file = "temperature-600116.json", "temperature-600116-fresh-dtu.json"
        checkpoint = BASE + "/runs/vae-e48-r7-rings-20261122a-s02/segment-02/checkpoint-000600116.pt"
        study = read(EXP / study_file)
        minutes = 35
        name = "dbmol-600116-temp-1123a"
        metadata = dict(temperatures=study["normalized_search_temperatures"])
    else:
        study_file, fresh_file, checkpoint = STUDY, FRESH, CHECKPOINT
        study = read(EXP / study_file)
        start, end, minutes = SHARDS[shard]
        if (study["checkpoint_sha256"] != CHECKPOINT_SHA
                or study["same_molecule_panel"]["parents"] != SHARDS["remaining"][1]
                or study["same_molecule_panel"]["sha256"] != digest(panel)):
            raise ValueError("Input panel/checkpoint does not match the committed study recipe")
        study["parent_slice"] = [start, end]
        name = f"dbmol-597604-same-{shard}-1122a"
        metadata = dict(slice=[start, end], panel_sha256=digest(panel))
    previous = read(EXP / "protenix-repeat-run-20261103-before-v100.json")["plan"]
    gpu = {k: previous[k] for k in ("cuda_ownership_helper", "cuda_ownership_sha256", "mosaic_python")}
    gpu_bytes = json_bytes(gpu)

    # Snapshot current Python sources, including the restored seven-round decoder,
    # alongside the exact study inputs. Both shards use the same panel bytes.
    package = ROOT / "src/latentmol"
    sources = list(package.rglob("*.py"))
    sources.extend(p for p in (package / "reporting/assets").iterdir() if p.is_file())
    records = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sources}
    for rel in ("scripts/hpc/protenix_latent_run.sh", *(f"{REL}/{n}" for n in
            ("run_temperature.sh", "native-dtu.json", "one-atom-dtu.json", "pilot.json", fresh_file))):
        data = (ROOT / rel).read_bytes()
        records[rel] = data.replace(b"\r\n", b"\n") if rel.endswith(".sh") else data
    records[f"{REL}/{study_file}"] = json_bytes(study)
    if "same_molecule_panel" in study:
        records[f"{REL}/{study['same_molecule_panel']['path']}"] = panel.read_bytes()
    records[f"{REL}/{PLAN}"] = gpu_bytes
    local.mkdir(exist_ok=False)
    archive = local / FILES[0]
    with archive.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for path, data in sorted(records.items()):
            member = tarfile.TarInfo(path)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    for filename in ("hpc_launch.py", "vae_segment_chain.py"):
        shutil.copyfile(ROOT / "scripts/hpc" / filename, local / filename)
    shell = ["bash", f"{REL}/run_temperature.sh", TORCH, gpu["mosaic_python"], checkpoint,
             f"{REL}/{study_file}", f"{REL}/native-dtu.json", f"{REL}/{fresh_file}", output,
             f"{REL}/{PLAN}", hashlib.sha256(gpu_bytes).hexdigest()]
    seconds = minutes * 60 - 35
    job_args = ["-q", "gpuv100", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=4096]",
                "-W", f"{minutes//60:02d}:{minutes%60:02d}",
                "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=16G",
                "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "/usr/bin/python3", control+"/hpc_launch.py", "--archive", control+"/source.tar.gz",
                "--archive-sha256", digest(archive), "--release", control+"/source",
                "--receipt", control+"/launch.json", "--startup-seconds", str(min(seconds, 1200)),
                "--run-seconds", str(seconds), "--", *shell]
    job = dict(name=name, args=job_args, output=output, checkpoint_sha256=study["checkpoint_sha256"], **metadata)
    (local / "job.json").write_bytes(json_bytes(job))
    print(json.dumps(dict(local=str(local), args=job_args, **metadata)))


def dispatch(action, shard, connector):
    local, control, output = locations(shard)
    hashes = {name: digest(local / name) for name in FILES}
    prelude = f"""import os,json,hashlib,sys
from pathlib import Path
c=Path({control!r})
expected={hashes!r}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
"""
    def remote(body):
        cmd = [str(connector), "exec", "--timeout", "95", "bash -lc "+shlex.quote("/usr/bin/python3 -I -")]
        result = subprocess.run(cmd, input=prelude+body, capture_output=True, text=True, timeout=105)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return json.loads(result.stdout)

    if action == "upload":
        missing = remote(f"""
if not c.exists() and Path({output!r}).exists():raise FileExistsError({output!r})
c.mkdir(exist_ok=True)
for n,h in expected.items():
 if (c/n).exists() and digest(c/n)!=h:raise ValueError('Release changed: '+n)
print(json.dumps([n for n in expected if not (c/n).exists()]))
""")
        for name in missing:
            temporary = name + ".stage-" + uuid.uuid4().hex
            subprocess.run([str(connector), "upload", "--timeout", "75", str(local / name),
                            control + "/" + temporary], check=True, timeout=85)
            remote(f"""
n={name!r}; staged=c/{temporary!r}
if digest(staged)!=expected[n]:raise ValueError('Staged bytes changed')
try:os.link(staged,c/n)
except FileExistsError:
 if digest(c/n)!=expected[n]:raise
staged.unlink()
print('true')
""")
    body = """
for n,h in expected.items():
 if digest(c/n)!=h:raise ValueError('Remote release changed: '+n)
"""
    if action == "submit":
        body += """
sys.path.insert(0,str(c))
from vae_segment_chain import submit, read
j=read(c/'job.json'); receipt=c/(j['name']+'.receipt.json')
if not receipt.exists():
 if Path(j['output']).exists():raise FileExistsError(j['output'])
 submit(c,j['name'],j['args'])
r=read(receipt)
if r['argv'][-len(j['args']):]!=j['args']:raise ValueError('Unexpected existing submission')
print(json.dumps(r))
"""
    else:
        body += "print(json.dumps({'uploaded':str(c),'sha256':expected}))"
    result = remote(body)
    (local / (action + ".json")).write_bytes(json_bytes(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("seal", "upload", "submit"))
    ap.add_argument("shard", choices=(*SHARDS, "temperature"))
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--connector", type=Path, default=Path.home()/"DTU-HPC/helper-v1/dtu-hpc.exe")
    args = ap.parse_args()
    if args.action == "seal":
        if args.shard != "temperature" and args.panel is None:
            ap.error("seal requires --panel")
        seal(args.shard, args.panel)
    else:
        dispatch(args.action,args.shard,args.connector)
