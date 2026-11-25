"""Seal and submit the two first-leg workers via the workstation's unattended DTU connector."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
TAG = "vae-e48-paired-leg1-20261120c"
CONTROL = "/zhome/95/e/206121/dbmol-control/" + TAG
PARENT = "/zhome/95/e/206121/dbmol-control/vae-pubchem60m-e47-dropout-20261119a/plan.json"
OUTPUT = "/work3/s234818/latentmol/runs/" + TAG
HERE = ROOT / "outputs/hpc-submissions" / TAG
CFG = "experiments/vae/epoch48-paired-leg1.json"
HELPERS = ("hpc_launch.py", "vae_segment_chain.py", "vae_paired_first_leg.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal():
    HERE.mkdir(parents=True, exist_ok=False)
    paths = sorted((ROOT / "src/latentmol").rglob("*.py"))
    paths.extend((ROOT / CFG, ROOT / "experiments/vae/run_checkpoint_comparison.sh"))
    archive = HERE / "source.tar.gz"
    with tarfile.open(archive, "x:gz") as tar:
        for path in paths:
            raw = path.read_bytes().replace(b"\r\n", b"\n")
            member = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
    for helper in HELPERS:
        shutil.copyfile(ROOT / "scripts/hpc" / helper, HERE / helper)
    cfg = json.loads((ROOT / CFG).read_bytes())
    prefix = "dbmol-e48-pair-leg1-20261120c"
    arms = []
    for rounds, queue, memory, reservation in [(4, "gpuv100", 32, 30), (7, "gpuv100", 32, 30)]:
        name = prefix + "-r" + str(rounds)
        args = ["-q", queue, "-n", "8", "-R", f"select[gpu{memory}gb] span[hosts=1] rusage[mem=6144]",
                "-gpu", f"num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem={reservation}G", "-W", "24:00",
                "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "/usr/bin/python3", CONTROL + "/hpc_launch.py", "--archive", CONTROL + "/source.tar.gz",
                "--archive-sha256", sha(archive), "--release", CONTROL + "/source-r" + str(rounds),
                "--receipt", CONTROL + "/" + name + ".launch.json", "--startup-seconds", "2400",
                "--run-seconds", "85000", "--", "/usr/bin/env", "LATENTMOL_CPU_THREADS=4",
                "LATENTMOL_EVALUATION_MODULE=latentmol.training.projected_continuation", "bash",
                "experiments/vae/run_checkpoint_comparison.sh", "--config", CFG, "--extra-rounds", str(rounds - 4),
                "--parent-receipt", CONTROL + "/parent.json", "--output", OUTPUT + "/r" + str(rounds)]
        arms.append(dict(name=name, args=args))
    predecessor = ROOT / "outputs/hpc-submissions/vae-pubchem60m-e47-dropout-20261119a/plan.json"
    plan = dict(prefix=prefix, parent_plan=PARENT, parent_plan_sha256=sha(predecessor),
                parent_config_sha256=cfg["parent_config_sha256"], parent_updates=cfg["parent_updates"], arms=arms)
    (HERE / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def dispatch(connector):
    files = ("source.tar.gz", *HELPERS, "plan.json")
    hashes = {key: sha(HERE / key) for key in files}
    plan = json.loads((HERE / "plan.json").read_bytes())
    prelude = f"""import hashlib, json, subprocess
from pathlib import Path
c=Path({CONTROL!r})
hashes={hashes!r}
prefix={plan['prefix']!r}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def records():
    args=['bjobs','-a','-u','s234818','-json','-o','jobid stat queue job_name:95']
    return [j for j in json.loads(subprocess.check_output(args,text=True))['RECORDS'] if j.get('JOB_NAME','').startswith(prefix)]
"""
    def remote(body):
        command = [str(connector), "exec", "--timeout", "90", "bash -lc " + shlex.quote("python3 -I -")]
        result = subprocess.run(command, input=prelude+body, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return json.loads(result.stdout)
    missing = remote(f"""
if not c.exists():
    assert not records() and not Path({OUTPUT!r}).exists()
    c.mkdir()
assert digest(Path({PARENT!r}))=={plan['parent_plan_sha256']!r}
for f, h in hashes.items():
    if (c/f).exists(): assert digest(c/f)==h
print(json.dumps([f for f in hashes if not (c/f).exists()]))
""")
    for filename in missing:
        subprocess.run([str(connector), "upload", "--timeout", "90", str(HERE / filename),
                        CONTROL + "/" + filename], check=True)
    result = remote("""
assert all(digest(c/f)==h for f,h in hashes.items())
for j in records():
    r=c/(j['JOB_NAME']+'.receipt.json')
    assert r.exists() and int(json.loads(r.read_bytes())['id'])==int(j['JOBID']), 'Reconcile scheduler receipt first'
result=subprocess.run(['/usr/bin/python3',str(c/'vae_paired_first_leg.py'),'submit',str(c/'plan.json')],
                      capture_output=True,text=True)
if result.returncode: raise RuntimeError(result.stderr + result.stdout)
print(json.dumps(dict(submission=result.stdout.strip(), scheduler=records())))
""")
    (HERE / "scheduler.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "submit"))
    parser.add_argument("--connector", type=Path, default=Path.home() / "DTU-HPC/helper-v1/dtu-hpc.exe")
    args = parser.parse_args()
    if args.action == "seal":
        seal()
    else:
        dispatch(args.connector)
