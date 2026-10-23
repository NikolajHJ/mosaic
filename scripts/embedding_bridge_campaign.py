"""Local standard-library sealing/transfers and read-only remote collection.

No numerical libraries or models are imported here. The immutable archive
contains actual working bytes, including uncommitted code. Numerical stages
are dispatched only by the allocated-HPC operator in that archive.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
WORK = "/work3/s234818/latentmol"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ConnectionAttempts=1", "dtu-hpc"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def call(command, receipt, *, payload=None, timeout=180):
    run = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=timeout)
    value = dict(command=command, returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
    save(receipt, value)
    if run.returncode:
        raise RuntimeError(f"Operation failed; preserved receipt: {receipt}")
    return value


def seal(folder, attempt):
    folder.mkdir(parents=True, exist_ok=False)
    label = "source137-epoch36-boltz-embedding-bridge-" + attempt
    files = []
    for directory in ("src", "tests", "scripts"):
        files += [p for p in (ROOT / directory).rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.suffix in (".py", ".sh", ".html", ".css", ".js")]
    files += [ROOT / "pyproject.toml", ROOT / "configs/epoch36_boltz_embedding_bridge.json"]
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    members = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}
    manifest_path = folder / "embedding-release-manifest.json"
    save(manifest_path, {"version": 1, "members": members})
    archive = folder / (label + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        for name in members:
            stream.add(ROOT / name, arcname=name, recursive=False)
        stream.add(manifest_path, arcname=manifest_path.name)
    plan = dict(label=label, archive=str(archive.resolve()), archive_sha256=sha(archive),
        manifest_sha256=sha(manifest_path), plan_sha256=sha(ROOT / "configs/epoch36_boltz_embedding_bridge.json"),
        release=WORK + "/releases/" + label.replace("source137-", "source-137-", 1),
        campaign=WORK + "/runs/" + label, members=len(members),
        status="sealed_not_submitted")
    save(folder / "seal.json", plan)
    print(json.dumps(plan, indent=2))


def stage(folder, identity):
    archive = Path(identity["archive"])
    if sha(archive) != identity["archive_sha256"]:
        raise ValueError("Local sealed archive changed")
    remote_archive = identity["release"] + ".tar.gz"
    # A preflight prevents clobbering any earlier release or upload.
    preflight = "from pathlib import Path\np=Path(" + repr(remote_archive) + "); r=Path(" + repr(identity["release"]) + ")\nassert not p.exists() and not r.exists(), 'Release/upload already exists'\nprint('new_release_paths_available')\n"
    call(SSH + ["bash -lc " + shlex.quote(WORK + "/env/boltz311/bin/python -")],
         folder / "stage-preflight.json", payload=preflight)
    call(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", str(archive), "dtu-hpc:" + remote_archive],
         folder / "upload.json", timeout=300)
    script = f'''import hashlib,json,tarfile
from pathlib import Path
archive=Path({remote_archive!r}); release=Path({identity['release']!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={identity['archive_sha256']!r}
assert not release.exists()
with tarfile.open(archive) as stream:
    for member in stream.getmembers():
        path=(release/member.name).resolve()
        assert path.is_relative_to(release) and member.isfile(), member.name
    release.mkdir()
    stream.extractall(release, filter='data')
manifest=release/'embedding-release-manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=={identity['manifest_sha256']!r}
members=json.loads(manifest.read_text())['members']
for name,sha in members.items():
    assert hashlib.sha256((release/name).read_bytes()).hexdigest()==sha, name
print(json.dumps({{'status':'staged_authenticated','members':len(members),'release':str(release)}}))
'''
    reply = call(SSH + ["bash -lc " + shlex.quote(WORK + "/env/boltz311/bin/python -")],
                 folder / "stage.json", payload=script)
    print(reply["stdout"])


def remote(folder, identity, action, stage_name):
    argv = [WORK + "/env/boltz311/bin/python", identity["release"] + "/scripts/hpc/embedding_operator.py",
            action, "--stage", stage_name, "--release", identity["release"],
            "--archive-sha256", identity["archive_sha256"], "--plan-sha256", identity["plan_sha256"],
            "--campaign", identity["campaign"]]
    receipt = folder / (action + "-" + stage_name + "-" + str(time.time_ns()) + ".json")
    reply = call(SSH + ["bash -lc " + shlex.quote(shlex.join(argv))], receipt)
    parsed = json.loads(reply["stdout"])
    if action == "collect":
        result_folder = ROOT / "outputs/hpc-results" / identity["label"]
        result_folder.mkdir(parents=True, exist_ok=True)
        output = result_folder / ("status-" + str(time.time_ns()) + ".json")
        save(output, parsed)
        print(json.dumps({"saved": str(output), "scheduler": {key: value for key, value in parsed.items() if key.endswith("_scheduler")}}, indent=2))
    else:
        print(json.dumps(parsed, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "stage", "submit", "collect"))
    parser.add_argument("--attempt", default="attempt001")
    parser.add_argument("--stage", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"attempt[0-9]{3}", args.attempt):
        raise ValueError("Attempt must be a versioned attemptNNN label")
    folder = ROOT / "outputs/hpc-submissions" / ("source137-epoch36-boltz-embedding-bridge-" + args.attempt)
    if args.action == "seal":
        seal(folder, args.attempt)
    else:
        identity = json.loads((folder / "seal.json").read_text())
        if args.action == "stage":
            stage(folder, identity)
        else:
            remote(folder, identity, args.action, args.stage)


if __name__ == "__main__":
    main()
