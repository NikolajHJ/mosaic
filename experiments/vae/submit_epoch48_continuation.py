"""Seal the seven-round epoch-48 continuation and dispatch it via the DTU connector."""

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


ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = ROOT / "outputs/hpc-submissions"
MOTIF_TAG = "vae-e48-r7-motif-followon-20261122b"
MOTIF_PLAN_SHA = "2943b6178a88e777c9dc0eb86c83a4d1f103f702ef9c42661483dae9ad459cf5"
CONTROL_ROOT = "/zhome/95/e/206121/dbmol-control/"
OUTPUT_ROOT = "/work3/s234818/latentmol/runs/"
BASE = SUBMISSIONS / "vae-e48-r7-motifs-20261121a"
BASE_HASH = "34082d26a2056e9393f27786bca2ca6aca3221ac3c7897a71b62b4f7298f2dc8"
TRAINER = "src/latentmol/training/projected_continuation.py"
MOTIFS = "src/latentmol/guidance/functional_groups.py"
MONITOR = "src/latentmol/training/occupancy_monitor.py"
COMPLETION = "src/latentmol/guidance/motif_completion.py"
PATTERNS = "src/latentmol/guidance/motif_patterns.py"
MOTIF_CONFIG = "experiments/vae/epoch48-seven-round-motif-followon.json"
RING_CONFIG = "experiments/vae/epoch48-seven-round-ring-sizes.json"
RING_TAG = "vae-e48-r7-rings-20261122a-s02"
RING_PLAN_SHA = "6fa6312fcf1fed476c13a1bb8f23d96896243f478a10992f2f7c598c1cbc2c27"
ROUNDTRIP_TAG = "vae-e48-r7-roundtrip-20261123a-s03"
ROUNDTRIP_CONFIG = "experiments/vae/epoch48-seven-round-generated-roundtrip.json"
EARLY_TAG = "vae-e48-r7-roundtrip-early-20261123a-s03"
ROUNDTRIP_PLAN_SHA = "b5c9aff1d5c478334f91e2069d6dd52681181fa8ef156d96290a05a277e75f86"
CHAIN = "epoch48_segment_chain.py"
FILES = ("source.tar.gz", "hpc_launch.py", "vae_segment_chain.py", CHAIN, "plan.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locations(handoff):
    if handoff == "early":
        tag = EARLY_TAG
    elif handoff == "roundtrip":
        tag = ROUNDTRIP_TAG
    else:
        tag = MOTIF_TAG if handoff is None else f"vae-e48-r7-rings-20261122a-s{handoff:02d}"
    return SUBMISSIONS / tag, CONTROL_ROOT + tag, OUTPUT_ROOT + tag


def seal(handoff):
    """Derive the new archive from the deployed release, preserving its other members verbatim."""
    if handoff == "early":
        return seal_early_roundtrip()
    if handoff == "roundtrip":
        return seal_roundtrip()
    original = BASE / "source.tar.gz"
    if sha(original) != BASE_HASH:
        raise ValueError("Unexpected first-leg release")
    first = json.loads((BASE / "reservation-verify.json").read_bytes())
    command = json.loads((BASE / "plan.json").read_bytes())["command"]
    local, control, output = locations(handoff)
    config = RING_CONFIG if handoff else MOTIF_CONFIG
    additions = [COMPLETION, PATTERNS, config]
    replacements = [TRAINER, MOTIFS, MONITOR]
    policy = json.loads((ROOT / config).read_bytes())["functional_groups"]
    if handoff:
        if sha(SUBMISSIONS / MOTIF_TAG / "plan.json") != MOTIF_PLAN_SHA:
            raise ValueError("Unexpected previous continuation plan")
        additions.extend((MOTIF_CONFIG, "src/latentmol/training/ring_activation.py",
                          "src/latentmol/guidance/ring_size.py", "src/latentmol/guidance/ring_size_reference.py"))
    if policy.get("neighbor_weight"):
        additions.append("src/latentmol/guidance/motif_neighbors.py")
    if policy.get("release_ring_teacher"):
        additions.append("src/latentmol/guidance/ring_teacher.py")
    if policy.get("activation_gate"):
        replacements.append("src/latentmol/training/combined_continuation.py")
        additions.append("src/latentmol/training/motif_activation.py")
    changes = {name: (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
               for name in (*replacements, *additions)}
    if b"--resume-receipt" not in changes[TRAINER]:
        raise ValueError("Missing resume entry point")
    with tarfile.open(original) as source:
        old_config = json.loads(source.extractfile(command[command.index("--config")+1]).read())
    candidate = json.loads(changes[config])
    if old_config.pop("functional_groups") == candidate.pop("functional_groups") or old_config != candidate:
        raise ValueError("Continuation requires a change confined to the motif policy")
    archive = build_archive(local, original, changes, replacements, additions)
    for helper in ("hpc_launch.py", "vae_segment_chain.py"):
        shutil.copyfile(SUBMISSIONS / "vae-e48-paired-leg1-20261120c" / helper, local / helper)
    shutil.copyfile(Path(__file__).with_name(CHAIN), local / CHAIN)
    plan = dict(archive=f"{control}/source.tar.gz", archive_sha256=sha(archive),
                output_root=output, first_directory=first["run"], parent_receipt=first["parent_receipt"],
                first_job_id=first["worker"], first_launch_receipt=command[command.index("--receipt")+1],
                config=config, config_sha256=hashlib.sha256(changes[config]).hexdigest())
    if handoff:
        plan["handoff"] = dict(control=CONTROL_ROOT + MOTIF_TAG, plan_sha256=MOTIF_PLAN_SHA,
                               segment=handoff, worker_id=30211418, callback_id=30211420)
        shutil.copyfile(Path(__file__).with_name("epoch48_cutover.py"), local / "epoch48_cutover.py")
    (local / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def build_archive(local, original, changes, replacements, additions):
    with tarfile.open(original) as source:
        names = source.getnames()
        if any(names.count(name) != 1 for name in replacements) or any(name in names for name in additions):
            raise ValueError("Unexpected source archive layout")
        local.mkdir(exist_ok=False)
        archive = local / "source.tar.gz"
        with archive.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|") as target:
                for member in source:
                    data = changes[member.name] if member.name in changes else source.extractfile(member).read()
                    member.size = len(data)
                    target.addfile(member, io.BytesIO(data))
                for name in additions:
                    member = tarfile.TarInfo(name)
                    member.size = len(changes[name])
                    target.addfile(member, io.BytesIO(changes[name]))
    return archive


def seal_roundtrip():
    source = SUBMISSIONS / RING_TAG
    old_path = source / "plan.json"
    if sha(old_path) != RING_PLAN_SHA:
        raise ValueError("Unexpected rings segment-2 plan")
    plan = json.loads(old_path.read_bytes())
    original = source / "source.tar.gz"
    if sha(original) != plan["archive_sha256"]:
        raise ValueError("Unexpected rings segment-2 release")
    replacements = (TRAINER, MONITOR, "src/latentmol/training/motif_activation.py")
    additions = (ROUNDTRIP_CONFIG, "src/latentmol/training/generated_roundtrip.py")
    changes = {n: (ROOT / n).read_bytes().replace(b"\r\n", b"\n") for n in (*replacements, *additions)}
    with tarfile.open(original) as before:
        source_config = before.extractfile(RING_CONFIG).read()
    candidate = json.loads(changes[ROUNDTRIP_CONFIG])
    if (hashlib.sha256(source_config).hexdigest() != plan["config_sha256"]
            or candidate.pop("generated_roundtrip") != dict(draws=16, max_parents=8, every_updates=4, weight=0.1)
            or candidate != json.loads(source_config)):
        raise ValueError("Only the generated round-trip objective may change")
    local, control, output = locations("roundtrip")
    archive = build_archive(local, original, changes, replacements, additions)
    for name in ("hpc_launch.py", "vae_segment_chain.py"):
        shutil.copyfile(source / name, local / name)
    for name in (CHAIN, "epoch48_cutover.py"):
        shutil.copyfile(Path(__file__).with_name(name), local / name)
    plan.update(archive=f"{control}/source.tar.gz", archive_sha256=sha(archive), output_root=output,
                config=ROUNDTRIP_CONFIG, config_sha256=hashlib.sha256(changes[ROUNDTRIP_CONFIG]).hexdigest(),
                job_prefix="e48-r7-roundtrip", handoff=dict(control=CONTROL_ROOT + RING_TAG,
                    plan_sha256=RING_PLAN_SHA, segment=3, worker_id=30249335, callback_id=30249336))
    (local / "plan.json").write_bytes((json.dumps(plan, indent=2) + "\n").encode())


def seal_early_roundtrip():
    source = SUBMISSIONS / ROUNDTRIP_TAG
    if sha(source / "plan.json") != ROUNDTRIP_PLAN_SHA:
        raise ValueError("Unexpected queued round-trip plan")
    plan = json.loads((source / "plan.json").read_bytes())
    if sha(source / "source.tar.gz") != plan["archive_sha256"]:
        raise ValueError("Unexpected queued science archive")
    local, _, output = locations("early")
    local.mkdir(exist_ok=False)
    for name in ("hpc_launch.py", "vae_segment_chain.py", "epoch48_cutover.py"):
        shutil.copyfile(source / name, local / name)
    for name in (CHAIN, "epoch48_early_cutover.py"):
        shutil.copyfile(Path(__file__).with_name(name), local / name)
    # Reuse the exact same archive at its original immutable location.
    plan.update(output_root=output, job_prefix="e48-r7-rtearly", early_stop=dict(
        superseded_control=CONTROL_ROOT + ROUNDTRIP_TAG, superseded_plan_sha256=ROUNDTRIP_PLAN_SHA,
        superseded_worker_id=30263213, superseded_callback_id=30263214))
    (local / "plan.json").write_bytes((json.dumps(plan, indent=2) + "\n").encode())


def dispatch(connector, handoff, *, upload_only=False, action="submit"):
    local, control, output = locations(handoff)
    files = (*FILES, "epoch48_cutover.py") if handoff else FILES
    if handoff == "early":
        files = tuple(name for name in (*files, "epoch48_early_cutover.py") if name != "source.tar.gz")
    hashes = {name: sha(local / name) for name in files}
    prelude = f"""import hashlib, json, os, subprocess
from pathlib import Path
c=Path({control!r})
files={hashes!r}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
"""
    def remote(code):
        result = subprocess.run([str(connector), "exec", "--timeout", "120",
            "bash -lc " + shlex.quote("python3 -I -")], input=prelude+code,
            capture_output=True, text=True, timeout=135)
        if result.returncode:
            raise RuntimeError(result.stderr + result.stdout)
        return json.loads(result.stdout)

    missing = remote(f"""
if not c.exists():
    if Path({output!r}).exists(): raise ValueError('Run directory already exists')
    c.mkdir()
for name, hashed in files.items():
    if (c/name).exists() and digest(c/name) != hashed: raise ValueError('Control file differs: '+name)
print(json.dumps([name for name in files if not (c/name).exists()]))
""")
    for name in missing:
        temporary = name + ".stage-" + uuid.uuid4().hex
        subprocess.run([str(connector), "upload", "--timeout", "120", str(local / name),
                        f"{control}/{temporary}"], check=True, timeout=135)
        remote(f"""
staged=c/{temporary!r}
if digest(staged) != files[{name!r}]: raise ValueError('Upload hash mismatch')
final=c/{name!r}
try: os.link(staged, final)
except FileExistsError:
    if digest(final) != files[{name!r}]: raise
staged.unlink()
print('true')
""")
    remote("""
for name, hashed in files.items():
    if digest(c/name) != hashed: raise ValueError('Release hash mismatch: '+name)
print('true')
""")
    if upload_only:
        print(json.dumps(dict(verified_control=control, sha256=hashes)))
        return
    if handoff:
        controller = "epoch48_early_cutover.py" if handoff == "early" else "epoch48_cutover.py"
        flags = ([action] if handoff == "early" else
                 {"submit": [], "stage": ["--stage"], "rollback": ["--rollback-staged"]}[action])
        result = remote(f"""
proc=subprocess.run(['/usr/bin/python3',str(c/{controller!r}),str(c/'plan.json'),*{flags!r}],capture_output=True,text=True)
if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
print(proc.stdout)
""")
        receipt = "scheduler.json" if action == "submit" else f"scheduler-{action}.json"
        (local / receipt).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    result = remote(f"""
query=['bjobs','-u','s234818','-json','-o','jobid stat queue job_name:100 dependency:140']
def jobs():
    return [r for r in json.loads(subprocess.check_output(query,text=True))['RECORDS']
            if 'e48-r7-complete' in r.get('JOB_NAME','') or r.get('JOBID')=='30151761']
for row in jobs():
    if row['JOBID'] == '30151761': continue
    receipt=c/(row['JOB_NAME']+'.receipt.json')
    if not receipt.is_file() or str(json.loads(receipt.read_bytes())['id']) != row['JOBID']:
        raise ValueError('A live job requires reconciliation: '+row['JOB_NAME'])
result=subprocess.run(['/usr/bin/python3', str(c/{CHAIN!r}), 'submit', str(c/'plan.json')],
                      capture_output=True, text=True)
if result.returncode: raise RuntimeError(result.stdout+result.stderr)
print(json.dumps(dict(submission=json.loads(result.stdout),scheduler=jobs())))
""")
    (local / "scheduler.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "upload", "stage", "submit", "rollback", "retire", "stop", "release"))
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--ring-handoff", type=int, choices=(2, 3),
                        help="Use the matched ring-size release at this epoch-48 segment boundary.")
    variant.add_argument("--generated-roundtrip", action="store_true",
                         help="Continue the selected rings arm with the generated round-trip objective in segment 3.")
    variant.add_argument("--early-roundtrip", action="store_true",
                         help="Stop the active rings worker and resume its durable save in a fresh round-trip allocation.")
    parser.add_argument("--connector", type=Path, default=Path.home()/"DTU-HPC/helper-v1/dtu-hpc.exe")
    args = parser.parse_args()
    handoff = "early" if args.early_roundtrip else "roundtrip" if args.generated_roundtrip else args.ring_handoff
    if handoff == "early" and args.action in ("submit", "rollback"):
        parser.error("The early handoff uses stage, retire, stop and release")
    if args.action in ("retire", "stop", "release") and handoff != "early":
        parser.error("retire, stop and release require --early-roundtrip")
    if args.action in ("stage", "rollback") and not handoff:
        parser.error("stage and rollback require a handoff")
    if args.action == "seal":
        seal(handoff)
    else:
        dispatch(args.connector, handoff, upload_only=args.action == "upload", action=args.action)
