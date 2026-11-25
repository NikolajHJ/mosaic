"""Replay fixed150-a on the original SXM2 device and rank the nine pilots at matched time."""
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
from uuid import uuid4
import zipfile


CAMPAIGN = "boltz-restart-brd4-timing-sxm2-20261119b"
ORIGINAL = "boltz-restart-brd4-methods-20261117a"
CONTROL = Path("/zhome/95/e/206121/dbmol-control")
OUTPUT = Path("/work3/s234818/latentmol/runs") / (CAMPAIGN + "-fixed150-a")
ARM = "fixed150-a"


def read(path):
    return json.loads(path.read_bytes())


def pilot(group, names):
    from arm_boltz_generations import option
    root = CONTROL / group
    rows = {}
    for name in names:
        receipt = read(root / name / "submission.json")
        rows[name] = dict(job_id=receipt["job_id"], output=receipt["output"],
                          source=option(receipt["argv"], "--archive-sha256"))
    return dict(pilot_control=str(root), pilots=rows)


def run_report():
    from arm_boltz_generations import publish, validated_results
    from boltz_policy_selection import select_policies
    names = [f"{policy}-{seed}" for seed in "abc" for policy in ("fixed150", "fixed400", "extend")]
    others = validated_results(pilot(ORIGINAL, [name for name in names if name != ARM]))
    original, = validated_results(pilot(ORIGINAL, [ARM]))
    repeat, = validated_results(pilot(CAMPAIGN, [ARM]))
    if original["start_hashes"] != repeat["start_hashes"] or original["case"] != repeat["case"]:
        raise ValueError("Timing rerun starts or scientific case differ from the completed fixed150-a")
    decision = select_policies([*others, repeat], fixed150_sxm2_job=int(repeat["job_id"]))
    decision.update(replaced_job_id=original["job_id"], timing_repeat_job_id=repeat["job_id"],
                    original_gpu=original["model_device"], repeat_gpu=repeat["model_device"])
    publish(CONTROL / CAMPAIGN / "timing-decision.json", decision)
    return decision


def sxm2_hosts():
    listing = subprocess.check_output(["bhosts", "-gpu", "-w"], text=True)
    models = {}
    host = None
    for line in listing.splitlines():
        match = re.match(r"^(?:(\S+)\s+|\s+)(\d+)\s+(\S+)", line)
        if match:
            host = match[1] or host
            models.setdefault(host, set()).add(match[3])
    queue = subprocess.check_output(["bqueues", "-l", "gpuv100"], text=True)
    host_groups = re.search(r"^HOSTS:\s*(.+)$", queue, flags=re.MULTILINE).group(1).split()
    allowed = set()
    for group in host_groups:
        if group.endswith("/"):
            output = subprocess.check_output(["bmgroup", "-r", "-w", group[:-1]], text=True)
            allowed.update(token.lstrip("+") for line in output.splitlines()[1:] for token in line.split()[1:])
        else:
            allowed.add(group.lstrip("+"))
    hosts = sorted(name for name, model in models.items() if name in allowed and model == {"TeslaV100_SXM2_32GB"})
    if not hosts:
        raise ValueError("No hosts advertising exclusively V100 SXM2 32GB devices")
    return hosts


def submit(package, digest):
    from arm_boltz_generations import option, publish, submit_once, validated_results
    old = read(CONTROL / ORIGINAL / ARM / "submission.json")
    argv = list(old["argv"])
    root = CONTROL / CAMPAIGN
    arm = root / ARM
    arm.mkdir(exist_ok=True)
    if not (arm / "submission.json").exists() and OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with tarfile.open(option(argv, "--archive")) as archive:
        raw = archive.extractfile(option(argv, "--plan")).read()
    if hashlib.sha256(raw).hexdigest() != option(argv, "--plan-sha256"):
        raise ValueError("Original plan differs from the submitted timing control")
    plan = json.loads(raw)
    plan["arms"][ARM] = str(OUTPUT)
    plan_path = root / "timing-plan.json"
    publish(plan_path, plan)
    for key, value in {"-J":CAMPAIGN + "-" + ARM, "-oo":str(arm / "run.%J.out"),
                       "-eo":str(arm / "run.%J.err"), "--receipt":str(arm / "launch.json"),
                       "--output":str(OUTPUT), "--plan":str(plan_path),
                       "--plan-sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest()}.items():
        argv[argv.index(key) + 1] = value
    hosts_path = root / "sxm2-hosts.json"
    if not hosts_path.exists():
        publish(hosts_path, sxm2_hosts())
    argv[1:1] = ["-m", " ".join(read(hosts_path))]
    repeat = submit_once(arm, argv, str(OUTPUT))
    report = root / "report"
    report.mkdir(exist_ok=True)
    command_file = report / "command.json"
    if not command_file.exists():
        wait = [repeat["job_id"]]
        for seed in "abc":
            for policy in ("fixed150", "fixed400", "extend"):
                name = f"{policy}-{seed}"
                if name == ARM:
                    continue
                manifest = pilot(ORIGINAL, [name])
                launch = CONTROL / ORIGINAL / name / "launch.json"
                if launch.exists() and read(launch).get("status") == "worker_exited":
                    validated_results(manifest)
                else:
                    wait.append(manifest["pilots"][name]["job_id"])
        publish(command_file, ["bsub", "-J", CAMPAIGN + "-analysis", "-q", "hpc", "-n", "1",
            "-R", "rusage[mem=1024]", "-W", "00:10", "-w", " && ".join(f"ended({job})" for job in wait),
            "-oo", str(report / "run.%J.out"), "-eo", str(report / "run.%J.err"),
            "/work3/s234818/latentmol/env/boltz311/bin/python", "-I", str(package), "report", digest])
    return dict(timing_repeat=repeat, report=submit_once(report, read(command_file)))


def stage():
    from arm_boltz_generations import CONNECTOR
    here = Path(__file__).resolve()
    local = Path("outputs") / CAMPAIGN
    local.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, source in [("__main__.py", here), ("arm_boltz_generations.py", here.with_name("arm_boltz_generations.py")),
                             ("boltz_policy_selection.py", Path("src/latentmol/design/boltz_policy_selection.py"))]:
            archive.writestr(zipfile.ZipInfo(name), source.read_bytes())
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    package = local / "timing.pyz"
    if not package.exists():
        with package.open("xb") as stream:
            stream.write(data)
    if package.read_bytes() != data:
        raise ValueError("A different timing package already exists")
    folder = (CONTROL / CAMPAIGN).as_posix()
    absolute = folder + "/timing.pyz"
    temp = absolute + "." + uuid4().hex
    def remote(cmd):
        return subprocess.check_output([CONNECTOR, "exec", "--timeout", "90", "bash -lc " + shlex.quote(cmd)], text=True)
    remote(shlex.join(["mkdir", "-p", folder]))
    subprocess.run([CONNECTOR, "upload", "--timeout", "90", str(package.resolve()), temp], check=True)
    install = """import hashlib, os, pathlib, sys
t,p,h=sys.argv[1:]
if hashlib.sha256(pathlib.Path(t).read_bytes()).hexdigest()!=h: raise ValueError('Transfer checksum changed')
try: os.link(t,p)
except FileExistsError:
    if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h: raise ValueError('Existing package differs')
finally: os.unlink(t)"""
    remote(shlex.join(["/usr/bin/python3", "-c", install, temp, absolute, digest]))
    receipts = json.loads(remote(shlex.join(["/work3/s234818/latentmol/env/boltz311/bin/python", "-I",
                                            absolute, "submit", digest])))
    receipt_file = local / "submissions.json"
    if not receipt_file.exists():
        with receipt_file.open("x") as stream:
            json.dump(receipts, stream, indent=2)
    if read(receipt_file) != receipts:
        raise ValueError("Conflicting timing receipts")
    print({name: row["job_id"] for name, row in receipts.items()})


if __name__ == "__main__":
    if sys.argv[1:] == ["stage"]:
        stage()
    elif len(sys.argv) == 3 and sys.argv[1] in {"submit", "report"}:
        package = Path(sys.argv[0]).resolve(strict=True)
        if hashlib.sha256(package.read_bytes()).hexdigest() != sys.argv[2]:
            raise ValueError("Timing package changed")
        print(json.dumps(submit(package, sys.argv[2]) if sys.argv[1] == "submit" else run_report()))
    else:
        raise SystemExit("Usage: python repeat_boltz_timing.py stage")
