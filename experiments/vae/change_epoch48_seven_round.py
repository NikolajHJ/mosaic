"""Change the held seven-round worker's command while retaining its LSF ID and parent gate."""

import argparse
from copy import copy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/hpc-submissions"
ORIGINAL = "vae-e48-paired-v100-20261120d"
PILOT = "vae-e48-weak-motifs-20261121b"
TAG = "vae-e48-r7-motifs-20261121a"
LOCAL = RUNS / TAG
REMOTE = "/zhome/95/e/206121/dbmol-control/" + TAG
WORKER = "30131363"
FOUR = "30131204"
GATE = "30131364"
BASE_CONFIG = "experiments/vae/epoch48-paired-leg1.json"
CONFIG = "experiments/vae/epoch48-seven-round-motifs.json"
TRAINER = "src/latentmol/training/projected_continuation.py"
LOSS = "src/latentmol/guidance/functional_groups.py"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def seal():
    old_dir, pilot_dir = RUNS / ORIGINAL, RUNS / PILOT
    arms = json.loads((old_dir / "plan.json").read_bytes())["arms"]
    old = arms[1]["args"]
    pilot = json.loads((pilot_dir / "submitted.json").read_bytes())["argv"]
    for directory, args in ((old_dir, old), (pilot_dir, pilot)):
        if digest(directory / "source.tar.gz") != args[args.index("--archive-sha256") + 1]:
            raise ValueError(f"Sealed source changed in {directory}")

    config = (ROOT / CONFIG).read_bytes().replace(b"\r\n", b"\n")
    with tarfile.open(old_dir / "source.tar.gz") as original, tarfile.open(pilot_dir / "source.tar.gz") as tested:
        base = json.loads(original.extractfile(BASE_CONFIG).read())
        approved = json.loads(tested.extractfile("experiments/vae/epoch48-motif-pilot.json").read())
        base["functional_groups"] = dict(approved["functional_groups"], aromatic_floor=.08, amide_floor=.12)
        if json.loads(config) != base or CONFIG in original.getnames() or LOSS in original.getnames():
            raise ValueError("Expected only the requested motif settings on the seven-round recipe")
        LOCAL.mkdir(exist_ok=False)
        archive = LOCAL / "source.tar.gz"
        with tarfile.open(archive, "x:gz") as new:
            for member in original:
                raw = (tested if member.name == TRAINER else original).extractfile(member.name).read()
                entry = copy(member)
                entry.size = len(raw)
                new.addfile(entry, BytesIO(raw))
            for name, raw in ((CONFIG, config), (LOSS, tested.extractfile(LOSS).read())):
                entry = tarfile.TarInfo(name)
                entry.size = len(raw)
                new.addfile(entry, BytesIO(raw))
    new = old[old.index("/usr/bin/python3"):]
    old_command = new.copy()
    for flag, value in {"--archive": REMOTE + "/source.tar.gz", "--archive-sha256": digest(archive),
                        "--config": CONFIG, "--receipt": REMOTE + "/launch.json", "--release": REMOTE + "/source",
                        "--output": "/work3/s234818/latentmol/runs/" + TAG + "/r7"}.items():
        new[new.index(flag) + 1] = value
    command4 = arms[0]["args"]
    plan = dict(worker=WORKER, four=FOUR, gate=GATE, original_directory=REMOTE.rsplit("/",1)[0]+"/"+ORIGINAL,
                worker_name=arms[1]["name"], four_name=arms[0]["name"], gate_name="dbmol-e48-pair-v100-20261120d-parent-01",
                four_command=command4[command4.index("/usr/bin/python3"):], original_command=old_command,
                command=new, archive_sha256=digest(archive), original_plan_sha256=digest(old_dir / "plan.json"))
    (LOCAL / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


REMOTE_ACTION = """
import hashlib, json, shlex, subprocess, datetime, time
from pathlib import Path

def check(test, message):
    if not test: raise RuntimeError(message)
def read(path): return json.loads(path.read_bytes())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot():
    cmd = ['bjobs','-a','-json','-o','jobid stat queue job_name:95 nreq_slot runtimelimit run_time dependency combined_resreq command',
           p['worker'],p['four'],p['gate']]
    rows = json.loads(subprocess.check_output(cmd, text=True))['RECORDS']
    return {r['JOBID']:r for r in rows}
def record(path, value):
    if path.exists():
        check(read(path) == value, 'Different existing command-change receipt: ' + str(path))
    else:
        with path.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2)

def state():
    rows = snapshot()
    w, four, gate = [rows[p[key]] for key in ('worker','four','gate')]
    for row, key in ((w,'worker_name'),(four,'four_name'),(gate,'gate_name')):
        check(row['JOB_NAME'] == p[key], 'Job ID or name changed')
    check(shlex.split(four['COMMAND']) == p['four_command'], 'Four-round command changed')
    check((w['QUEUE'],w['NREQ_SLOT'],w['RUNTIMELIMIT']) == ('gpuv100','8','24:00:00') and not w['DEPENDENCY']
          and 'gpu32gb' in w['COMBINED_RESREQ'] and 'gmem=30720' in w['COMBINED_RESREQ'], 'Unexpected resources')
    check(gate['DEPENDENCY'].replace(' ','')=='done(30114867)&&ended(30131206)&&ended(30131205)', 'Unexpected parent gate')
    return rows, w, four, gate

old = Path(p['original_directory'])
check(digest(old/'plan.json') == p['original_plan_sha256'], 'Previous plan changed')
rc = read(old/(p['worker_name']+'.receipt.json'))
check(str(rc['id'])==p['worker'] and rc['argv'][rc['argv'].index('/usr/bin/python3'):] == p['original_command'],
      'Previous worker receipt differs')
rows, w, four, gate = state()
seen = shlex.split(w['COMMAND'])
check(seen in (p['original_command'],p['command']), 'Unrecognized worker command')
if action == 'stage':
    dest.mkdir(exist_ok=True)
    for name, h in hashes.items():
        check(not (dest/name).exists() or digest(dest/name)==h, 'Sealed file collision: '+name)
    print(json.dumps([name for name in hashes if not (dest/name).exists()]))
else:
    check(all(digest(dest/k)==v for k,v in hashes.items()),'Uploaded package digest mismatch')
    check(read(dest/'plan.json')==p, 'Remote plan differs')
    if seen == p['original_command']:
        check(action=='apply', 'New seven-round command not yet scheduled')
        check(w['STAT']=='PSUSP' and w['RUN_TIME']=='00:00:00' and gate['STAT']=='PEND' and four['STAT']=='PSUSP',
              'Worker or gate has already started; do not cut over')
        for cmd in (seen, p['command']):
            for flag in ('--receipt','--release','--output','--parent-receipt'):
                check(not Path(cmd[cmd.index(flag)+1]).exists(), 'A worker or its parent has already published '+flag)
        record(dest/'command-change.intent.json', dict(id=p['worker'], original=seen, replacement=p['command']))
        change=subprocess.run(['bmod','-Z',' '.join(shlex.quote(a) for a in p['command']),p['worker']],
                              text=True, capture_output=True)
        check(change.returncode==0, 'Scheduler command change failed: '+change.stderr+change.stdout)
        for _ in range(12):
            rows, w, four, gate = state()
            if shlex.split(w['COMMAND']) == p['command']: break
            time.sleep(1)
        check((w['STAT'],w['RUN_TIME'],four['STAT'],gate['STAT'])==('PSUSP','00:00:00','PSUSP','PEND'),
              'Post-change scheduler state differs; review the parent gate and held workers')
    check(shlex.split(w['COMMAND'])==p['command'], 'Scheduler has not confirmed the replacement command')
    proof = dict(id=p['worker'], four=p['four'], gate=p['gate'], archive_sha256=p['archive_sha256'], command=p['command'])
    record(dest/'command-change.receipt.json', proof)
    print(json.dumps(dict(checked=datetime.datetime.now().astimezone().isoformat(), receipt=proof, scheduler=rows)))
"""


def dispatch(connector, action):
    plan = json.loads((LOCAL / "plan.json").read_bytes())
    hashes = {k: digest(LOCAL / k) for k in ("source.tar.gz", "plan.json")}
    if hashes["source.tar.gz"] != plan["archive_sha256"]:
        raise ValueError("Source differs from the local plan")

    def remote(verb):
        variables = f"from pathlib import Path\np={plan!r}\nhashes={hashes!r}\ndest=Path({REMOTE!r})\naction={verb!r}\n"
        command = "bash -lc " + shlex.quote("python3 -I -")
        proc = subprocess.run([str(connector), "exec", "--timeout", "60", command], input=variables+REMOTE_ACTION,
                              text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        return json.loads(proc.stdout)

    if action == "apply":
        for name in remote("stage"):
            subprocess.run([str(connector), "upload", "--timeout", "90", str(LOCAL/name), REMOTE+"/"+name], check=True)
    result = remote(action)
    (LOCAL / "verified.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: [v['STAT'], v['JOB_NAME'], v['RUN_TIME']] for k,v in result['scheduler'].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal","apply","verify"))
    parser.add_argument("--connector", type=Path, default=Path.home()/"DTU-HPC/helper-v1/dtu-hpc.exe")
    args = parser.parse_args()
    if args.action == "seal":
        seal()
    else:
        dispatch(args.connector, args.action)
