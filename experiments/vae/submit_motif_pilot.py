"""Package and submit the short, independent functional-group comparison to one DTU V100."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
TAG = "vae-e48-weak-motifs-20261121b"
NAME = "dbmol-" + TAG
CONTROL = "/zhome/95/e/206121/dbmol-control/" + TAG
OUTPUT = "/work3/s234818/latentmol/runs/" + TAG
PARENT_CONTROL = "/zhome/95/e/206121/dbmol-control/vae-e48-paired-leg1-20261120c"
PARENT_RECEIPT = PARENT_CONTROL + "/parent.json"
BASE = "experiments/vae/epoch48-paired-leg1.json"
PILOT = "experiments/vae/epoch48-motif-pilot.json"
LOCAL = ROOT / "outputs/hpc-submissions" / TAG


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal():
    LOCAL.mkdir(parents=True, exist_ok=False)
    paths = sorted((ROOT / "src/latentmol").rglob("*.py"))
    paths += [ROOT / p for p in (BASE, PILOT, "experiments/vae/motif_pilot.py",
              "experiments/vae/run_checkpoint_comparison.sh", "experiments/vae/training_data_eda/chemistry_profile.py")]
    # All numerics and scripts are sealed locally; the legacy production jobs never read this archive.
    with tarfile.open(LOCAL / "source.tar.gz", "x:gz") as archive:
        for path in paths:
            raw = path.read_bytes().replace(b"\r\n", b"\n")
            entry = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
            entry.size = len(raw)
            archive.addfile(entry, io.BytesIO(raw))
    for helper in ("hpc_launch.py", "vae_segment_chain.py"):
        (LOCAL / helper).write_bytes((ROOT / "scripts/hpc" / helper).read_bytes())


def dispatch(connector):
    files = ("source.tar.gz", "hpc_launch.py", "vae_segment_chain.py")
    hashes = {f: sha(LOCAL / f) for f in files}
    prefix = f"""from pathlib import Path
import json, hashlib, subprocess, sys
c=Path({CONTROL!r})
name={NAME!r}
hashes={hashes!r}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
"""
    def remote(body):
        remote_cmd = "bash -lc " + shlex.quote("python3 -I -")
        result = subprocess.run([str(connector), "exec", "--timeout", "90", remote_cmd], input=prefix+body,
                                text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(result.stdout)

    missing = remote("""
c.mkdir(exist_ok=True)
for f,h in hashes.items():
    if (c/f).exists() and digest(c/f)!=h: raise ValueError('Sealed file collision: '+f)
print(json.dumps([f for f in hashes if not (c/f).exists()]))
""")
    for f in missing:
        subprocess.run([str(connector), "upload", "--timeout", "120", str(LOCAL/f), CONTROL+"/"+f], check=True)
    args = ["-q", "gpuv100", "-n", "8", "-R", "select[gpu32gb] span[hosts=1] rusage[mem=6144]",
            "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G", "-W", "06:00",
            "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "/usr/bin/python3", CONTROL+"/hpc_launch.py", "--archive", CONTROL+"/source.tar.gz",
            "--archive-sha256", hashes["source.tar.gz"], "--release", CONTROL+"/source",
            "--receipt", CONTROL+"/launch.json", "--startup-seconds", "2400", "--run-seconds", "21000", "--",
            "/usr/bin/env", "LATENTMOL_CPU_THREADS=4", "LATENTMOL_EVALUATION_MODULE=experiments.vae.motif_pilot",
            "bash", "experiments/vae/run_checkpoint_comparison.sh", "--config", BASE, "--pilot", PILOT,
            "--parent-receipt", PARENT_RECEIPT, "--output", OUTPUT]
    receipt = remote(f"""
assert all(digest(c/f)==h for f,h in hashes.items())
sys.path.insert(0,str(c))
from vae_segment_chain import submit
receipt=c/(name+'.receipt.json')
if receipt.exists():
    seen=json.loads(receipt.read_bytes())
    if seen['argv'][-len({args!r}):]!={args!r}: raise ValueError('Prior submission differs')
    print(json.dumps(seen))
else:
    if Path({OUTPUT!r}).exists(): raise FileExistsError('Pilot output already present')
    # The four-round job can start only after the parent receipt has been published,
    # including when a parent callback schedules a second allocation. Never alter those jobs.
    control_receipt=Path({PARENT_CONTROL!r})/'dbmol-e48-pair-leg1-20261120c-r4.receipt.json'
    worker_id=int(json.loads(control_receipt.read_bytes())['id'])
    dependency=[]
    if not Path({PARENT_RECEIPT!r}).is_file():
        state=subprocess.check_output(['bjobs','-a','-noheader','-o','stat',str(worker_id)],text=True).strip()
        if state not in ('PEND','PSUSP','RUN','DONE'): raise ValueError('Parent-publication worker: '+state)
        if state in ('RUN','DONE') and not Path({PARENT_RECEIPT!r}).is_file():
            raise FileNotFoundError('Parent worker started without the shared receipt')
        if state in ('PEND','PSUSP'): dependency=['-w','started(%d)'%worker_id]
    submit(c,name,[*dependency,*{args!r}])
    print(receipt.read_text())
""")
    (LOCAL/"submitted.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seal", "submit"])
    parser.add_argument("--connector", type=Path, default=Path.home()/"DTU-HPC/helper-v1/dtu-hpc.exe")
    cli = parser.parse_args()
    if cli.action == "seal":
        seal()
    else:
        dispatch(cli.connector)
