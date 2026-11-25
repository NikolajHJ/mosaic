"""Arm the BRD4 allocation follow-up after staging the candidate GPU commands."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from tempfile import NamedTemporaryFile
import uuid
import zipfile


POLICIES = ("fixed150", "fixed400", "extend")
PANELS = ("a", "b", "c")
ARMS = {f"{policy}-{panel}" for policy in POLICIES for panel in PANELS}
CONTROL = "/zhome/95/e/206121/dbmol-control"
PYTHON = "/work3/s234818/latentmol/env/boltz311/bin/python"
CONNECTOR = r"C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe"


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    """Publish complete immutable JSON; concurrent identical publications are harmless."""
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read(path) != value:
                raise ValueError(f"Conflicting recorded data at {path}")
    finally:
        temporary.unlink()


def option(argv, flag):
    return argv[argv.index(flag) + 1]


def accepted(argv, job_id, **details):
    return dict(argv=argv, job_id=job_id, queue=option(argv, "-q"), returncode=0, **details)


def reconcile(argv):
    """Recover only an exact scheduler match after a lost bsub reply."""
    name, queue = option(argv, "-J"), option(argv, "-q")
    result = subprocess.run(["bjobs", "-a", "-u", "s234818", "-J", name, "-noheader",
                             "-o", "jobid job_name:200 queue command:4096"],
                            capture_output=True, text=True, timeout=45)
    matches = []
    for line in result.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and fields[0].isdigit() and fields[1:3] == [name, queue]:
            command = shlex.split(fields[3])
            if command and argv[-len(command):] == command:
                matches.append(int(fields[0]))
    if result.returncode or len(matches) != 1:
        raise RuntimeError(f"Unresolved submit intent for {name}; refused to repeat bsub")
    return accepted(argv, matches[0], reconciled=True)


def submit_once(root, argv, output=None):
    import fcntl
    receipt = root / "submission.json"
    intent = root / "submit-intent.json"
    with (root / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            row = read(receipt)
            if row["argv"] != argv or row["returncode"] or not row["job_id"]:
                raise RuntimeError(f"Unconfirmed or conflicting submission at {receipt}")
            return row
        if intent.exists():
            if read(intent) != argv:
                raise ValueError(f"Command differs from existing intent at {intent}")
            row = reconcile(argv)
        else:
            publish(intent, argv)
            result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            match = re.search(r"Job <(\d+)> is submitted to queue <([^>]+)>", result.stdout)
            if result.returncode or not match or match[2] != option(argv, "-q"):
                raise RuntimeError(f"Unconfirmed submission for {option(argv, '-J')}: {result.stdout} {result.stderr}")
            row = accepted(argv, int(match[1]), stdout=result.stdout, stderr=result.stderr)
        if output is not None:
            row["output"] = output
        publish(receipt, row)
        return row


def validated_results(manifest):
    rows = []
    for arm, expected in sorted(manifest["pilots"].items()):
        root = Path(manifest["pilot_control"]) / arm
        submission, launch = read(root / "submission.json"), read(root / "launch.json")
        result_path = Path(expected["output"]) / "result.json"
        result = read(result_path)
        if (submission["returncode"] or submission["job_id"] != expected["job_id"]
                or submission["output"] != expected["output"]
                or option(submission["argv"], "--archive-sha256") != expected["source"]
                or launch.get("status") != "worker_exited" or launch.get("returncode") != 0
                or launch.get("ready") is not True or str(launch.get("job_id")) != str(expected["job_id"])
                or launch.get("source_archive_sha256") != expected["source"]
                or result.get("status") != "completed" or str(result.get("job_id")) != str(expected["job_id"])
                or result.get("source_sha256") != expected["source"]):
            raise ValueError(f"Unverified pilot or launcher outcome for {arm}")
        rows.append(result)
    return rows


def dispatch(manifest):
    from boltz_policy_selection import select_policies
    control = Path(manifest["control"])
    decision = select_policies(validated_results(manifest))
    publish(control / "gate-decision.json", decision)
    if decision["status"] == "inconclusive" and not decision["selected_policies"]:
        return dict(status="inconclusive", jobs={})
    policies = decision["selected_policies"]
    if decision["status"] != "selected" or len(set(policies)) != 2 or len(policies) != 2 or set(policies) - set(POLICIES):
        raise ValueError("Selection did not produce two distinct eligible policies")
    jobs = {}
    for policy in policies:
        for panel in PANELS:
            arm = f"{policy}-{panel}"
            argv = manifest["commands"][arm]
            row = submit_once(control / arm, argv, output=option(argv, "--output"))
            jobs[arm] = row["job_id"]
    if len(set(jobs.values())) != 6:
        raise ValueError("Expected six distinct scheduler receipts")
    return dict(status="submitted", policies=policies, jobs=jobs)


def gate_main(action, package, expected_sha):
    if sha(package) != expected_sha:
        raise ValueError("Gate archive changed")
    with zipfile.ZipFile(package) as bundle:
        manifest = json.loads(bundle.read("gate-plan.json"))
    control = Path(manifest["control"])
    gate_dir = control / "policy-gate"
    if action == "dispatch":
        if not os.environ.get("LSB_JOBID", "").isdigit():
            raise RuntimeError("Gate dispatch requires an allocated LSF job")
        outcome = dispatch(manifest)
        publish(control / "gate-outcome.json", outcome)
        return outcome
    if action != "register":
        raise ValueError(f"Unknown gate action {action}")
    gate_dir.mkdir(exist_ok=True)
    dependencies = " && ".join(f"ended({p['job_id']})" for p in manifest["pilots"].values())
    argv = ["bsub", "-J", f"dbmol-{control.name}-policy-gate", "-q", "hpc", "-n", "1",
            "-R", "rusage[mem=1024]", "-W", "00:15", "-w", dependencies,
            "-oo", str(gate_dir / "gate.%J.out"), "-eo", str(gate_dir / "gate.%J.err"),
            PYTHON, "-I", str(package), "dispatch", expected_sha]
    return submit_once(gate_dir, argv)


def package_gate(name, pilot):
    root = Path("outputs") / name
    plan = read(root / "gpu-plan.json")
    if not plan.get("selection_required") or read(root / "storage.json")["status"] != "passed":
        raise ValueError("Stage the gated candidate campaign before arming")
    commands = read(root / "commands.json")
    if set(commands) != ARMS:
        raise ValueError("Expected three policies and three panels in staged commands")
    pilots = {}
    for arm in sorted(ARMS):
        receipt = read(Path("outputs") / pilot / f"{arm}-submission.json")
        if receipt["returncode"] or type(receipt.get("job_id")) != int:
            raise ValueError(f"Unconfirmed pilot submission for {arm}")
        argv = commands[arm]
        if (argv[0] != "bsub" or option(argv, "--arm") != arm or option(argv, "--output") != plan["arms"][arm]
                or option(argv, "--plan-sha256") != sha(root / "gpu-plan.json")
                or plan["planned_fresh_attempts"][arm] != 144):
            raise ValueError(f"Mismatched staged command for {arm}")
        pilots[arm] = dict(job_id=receipt["job_id"], output=receipt["output"],
                           source=option(receipt["argv"], "--archive-sha256"))
    manifest = dict(control=f"{CONTROL}/{name}", pilot_control=f"{CONTROL}/{pilot}", pilots=pilots, commands=commands)
    if len({p["job_id"] for p in pilots.values()}) != 9:
        raise ValueError("Expected nine distinct pilot IDs")
    # Extract the selector from the already sealed comparison source so later
    # edits cannot alter eligibility or policy ranking.
    import tarfile
    with tarfile.open(root / "source.tar.gz") as source:
        selector = source.extractfile("src/latentmol/design/boltz_policy_selection.py").read()
    package = root / "policy-gate.pyz"
    import io
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as bundle:
        for path, content in [("__main__.py", Path(__file__).read_bytes()), ("boltz_policy_selection.py", selector),
                              ("gate-plan.json", json.dumps(manifest, sort_keys=True).encode())]:
            bundle.writestr(zipfile.ZipInfo(path), content)
    if not package.exists():
        with package.open("xb") as stream:
            stream.write(data.getvalue())
    elif package.read_bytes() != data.getvalue():
        raise ValueError("Gate package differs from existing sealed package")
    return package


def arm(name, pilot):
    if any(not re.fullmatch(r"boltz-restart-[a-z0-9-]+", s) for s in (name, pilot)):
        raise ValueError("Use BRD4 restart campaign identifiers")
    package = package_gate(name, pilot)
    digest = sha(package)
    destination = f"{CONTROL}/{name}/{package.name}"
    # Upload to a private temporary file, then publish without replacing any armed callback.
    temporary = destination + "." + uuid.uuid4().hex
    subprocess.run([CONNECTOR, "upload", "--timeout", "90", str(package), temporary], check=True, timeout=100)
    install = """import hashlib, os, pathlib, sys
tmp, path, expected = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
try:
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != expected: raise ValueError('Upload changed')
    try: os.link(tmp, path)
    except FileExistsError:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected: raise ValueError('Different gate already armed')
finally: tmp.unlink()
"""
    cmd = (shlex.join(["/usr/bin/python3", "-c", install, temporary, destination, digest])
           + " && " + shlex.join([PYTHON, "-I", destination, "register", digest]))
    result = subprocess.run([CONNECTOR, "exec", "--timeout", "120", "bash -lc " + shlex.quote(cmd)],
                            capture_output=True, text=True, timeout=130)
    if result.returncode:
        raise RuntimeError(f"Could not confirm policy gate: {result.stderr} {result.stdout}")
    row = json.loads(result.stdout)
    publish(package.parent / "gate-submission.json", row)
    print(f"Policy gate {row['job_id']} submitted to {row['queue']}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("dispatch", "register"):
        print(json.dumps(gate_main(sys.argv[1], Path(sys.argv[0]).resolve(), sys.argv[2])))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--name", required=True)
        parser.add_argument("--pilot", required=True)
        args = parser.parse_args()
        arm(args.name, args.pilot)
