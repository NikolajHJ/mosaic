"""Stage or submit one config-driven DTU restart campaign from the repository root."""

import argparse
import json
from pathlib import Path
import re
import runpy
import tarfile


EXPERIMENT = Path("experiments/latent-design/protenix-vae")
FRESH = EXPERIMENT / "restarts-fresh-dtu.json"
NATIVE = EXPERIMENT / "native-dtu.json"
ASSETS = EXPERIMENT / "one-atom-dtu.json"
WRAPPER = Path("scripts/hpc/protenix_latent_run.sh")
LAUNCHER = Path("scripts/hpc/hpc_launch.py")
STORAGE = Path(r"C:\Users\nikol\OneDrive\Dokumenter\GitHub\mosaic\scripts\hpc\storage_accounting.py")
TRANSPORT = runpy.run_path(str(Path(__file__).with_name("protenix-restart-followup-20261107a") / "submit.py"))
remote, upload, sha = (TRANSPORT[key] for key in ("remote", "upload", "sha"))
BASE = TRANSPORT["BASE"]
OUTPUT_MIB = 768
OUTPUT_FILES = 6000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def stage(args, local, control, release):
    config = EXPERIMENT / args.config
    cfg = read(config)
    outputs = {}
    budgets = {}
    for arm, override in cfg["arms"].items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", arm):
            raise ValueError(f"Invalid arm name: {arm}")
        campaign = cfg["campaign"] | cfg.get("campaign_arms", {}).get(arm, {})
        design = cfg["design"] | override
        branches = 1 + (campaign["cycles"] - 1) * campaign["elite_count"] * campaign["branches_per_elite"]
        budgets[arm] = branches * design["updates"]
        outputs[arm] = f"{BASE}/runs/{args.name}-{arm}"
    if not budgets or any(not 0 < updates <= args.max_updates for updates in budgets.values()):
        raise ValueError(f"Arm exceeds the declared update budget: {budgets}")

    prior = read(Path("outputs/protenix-restarts-20261107a/gpu-plan.json"))
    plan = {key: prior[key] for key in ("cuda_ownership_helper", "cuda_ownership_sha256", "mosaic_python")}
    plan.update(arms=outputs, queue=args.queue, planned_updates=budgets,
                output_cap_mib=OUTPUT_MIB, config_sha256=sha(config))
    local.mkdir()
    save(local / "gpu-plan.json", plan)
    if b"\r\n" in WRAPPER.read_bytes():
        raise ValueError("Unexpected CRLF in launch wrapper")
    package = local / "source.tar.gz"
    remote_archive = release + ".tar.gz"
    plan_name = EXPERIMENT / "restart-campaign-gpu-plan.json"
    with tarfile.open(package, "x:gz") as bundle:
        for path in [*sorted(Path("src/latentmol").rglob("*.py")), config, FRESH, NATIVE, ASSETS, WRAPPER]:
            bundle.add(path, arcname=path.as_posix())
        bundle.add(local / "gpu-plan.json", arcname=plan_name.as_posix())

    assets, native, fresh = map(read, (ASSETS, NATIVE, FRESH))
    expected = {
        assets["checkpoint"]: assets["checkpoint_sha256"],
        assets["native"]["weights_path"]: assets["native"]["weights_sha256"],
        native["context_ready"]: native["context_ready_sha256"],
        native["pocket_path"]: native["pocket_sha256"],
        fresh["pocket"]["path"]: fresh["pocket"]["sha256"],
        plan["cuda_ownership_helper"]: plan["cuda_ownership_sha256"],
    }
    torch_python, fresh_deps, fresh_data = (TRANSPORT[key] for key in (
        "TORCH_PYTHON", "FRESH_DEPS", "FRESH_DATA"))
    required = [native["mosaic_source"], plan["mosaic_python"], assets["native"]["source_root"],
                torch_python, fresh_deps, fresh_data, fresh["ccd_root"]]
    probe = f"{BASE}/runs/{args.name}-capacity-probe"
    targets = [remote_archive, release, control, probe, *outputs.values()]
    save(local / "preflight.json", remote(f"""
import hashlib, json, subprocess
from pathlib import Path
for target in map(Path, {targets!r}):
    if target.exists():
        raise FileExistsError(target)
for path, expected in {expected!r}.items():
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('Changed input: ' + path)
for path in {required!r}:
    if not Path(path).exists():
        raise FileNotFoundError(path)
scheduler = subprocess.run(['bqueues', {args.queue!r}], capture_output=True, text=True, check=True, timeout=15)
if 'Open:Active' not in scheduler.stdout:
    raise ValueError('Queue is not active: ' + scheduler.stdout)
control = Path({control!r})
control.mkdir()
for arm in {list(outputs)!r}:
    (control / arm).mkdir()
print(json.dumps(dict(inputs={expected!r}, outputs={outputs!r}, scheduler=scheduler.stdout)))
"""))
    uploads = [(package, remote_archive), (LAUNCHER, f"{control}/hpc_launch.py"),
               (STORAGE, f"{control}/storage_accounting.py")]
    for source, target in uploads:
        upload(source, target)
    save(local / "storage.json", remote(f"""
import hashlib, subprocess, tarfile
from pathlib import Path
for path, expected in {[(target, sha(source)) for source, target in uploads]!r}:
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError('Upload checksum differs: ' + path)
with tarfile.open({remote_archive!r}) as bundle:
    subprocess.run(['bash', '-n'], input=bundle.extractfile({WRAPPER.as_posix()!r}).read(), check=True)
probe = Path({probe!r})
probe.mkdir()
try:
    result = subprocess.run(['/usr/bin/python3', {uploads[-1][1]!r}, '--output-root', str(probe),
        '--cap-mib', {str(OUTPUT_MIB * len(outputs))!r}, '--file-cap', {str(OUTPUT_FILES * len(outputs))!r}],
        check=True, text=True, capture_output=True)
    print(result.stdout)
finally:
    probe.rmdir()
"""))
    archive_sha = sha(package)
    plan_sha = sha(local / "gpu-plan.json")
    commands = {}
    for arm, output in outputs.items():
        receipts = f"{control}/{arm}"
        commands[arm] = [
            "bsub", "-J", f"dbmol-{args.name}-{arm}", "-q", args.queue, "-n", "4",
            "-R", "span[hosts=1]", "-R", "rusage[mem=4096]", "-gpu", TRANSPORT["GPU_RESOURCE"][args.queue],
            "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "-W", f"{args.hours:02d}:00", "-oo", f"{receipts}/run.%J.out", "-eo", f"{receipts}/run.%J.err",
            "/usr/bin/python3", f"{control}/hpc_launch.py", "--archive", remote_archive,
            "--archive-sha256", archive_sha, "--release", release, "--receipt", f"{receipts}/launch.json",
            "--startup-seconds", "900", "--run-seconds", str(args.hours * 3600 - 900),
            "--", "bash", WRAPPER.as_posix(), torch_python, "-m", "latentmol.design.protenix_restarts",
            "--config", config.as_posix(), "--arm", arm, "--output", output,
            "--checkpoint", assets["checkpoint"], "--checkpoint-sha256", assets["checkpoint_sha256"],
            "--mosaic-python", plan["mosaic_python"], "--native-config", NATIVE.as_posix(),
            "--fresh-config", FRESH.as_posix(), "--fresh-deps", fresh_deps, "--fresh-data", fresh_data,
            "--plan", plan_name.as_posix(), "--plan-sha256", plan_sha,
        ]
    save(local / "commands.json", commands)
    print(f"Staged {len(commands)} arms from source {archive_sha}")


def submit(local, control):
    if read(local / "storage.json")["status"] != "passed":
        raise ValueError("Missing successful capacity check")
    for arm, argv in read(local / "commands.json").items():
        output = argv[argv.index("--output") + 1]
        queue = argv[argv.index("-q") + 1]
        receipt = remote(f"""
import json, re, subprocess
from pathlib import Path
root = Path({f"{control}/{arm}"!r})
argv = {argv!r}
receipt_path = root / 'submission.json'
if receipt_path.exists():
    receipt = json.loads(receipt_path.read_text())
    if receipt['argv'] != argv:
        raise ValueError('Existing submission command differs')
else:
    with (root / 'submit-intent.json').open('x') as stream:
        json.dump(argv, stream)
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    job = re.search(r'Job <([0-9]+)>', result.stdout)
    receipt = dict(job_id=int(job[1]) if job else None, returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, argv=argv, queue={queue!r}, output={output!r})
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
""")
        path = local / f"{arm}-submission.json"
        if not path.exists():
            save(path, receipt)
        elif read(path) != receipt:
            raise ValueError(f"Local submission differs for {arm}")
        if receipt["returncode"] or receipt["job_id"] is None:
            raise RuntimeError(f"Scheduler did not confirm {arm}: {receipt['stderr']}")
        print(arm, receipt["stdout"].strip(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "submit"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--config")
    parser.add_argument("--queue", choices=tuple(TRANSPORT["GPU_RESOURCE"]), default="gpul40s")
    parser.add_argument("--hours", type=int, default=12)
    parser.add_argument("--max-updates", type=int, required=False)
    args = parser.parse_args()
    if not WRAPPER.is_file() or not re.fullmatch(r"protenix-restart-[a-z0-9-]+", args.name):
        parser.error("Run from the DBMol root with a unique Protenix restart campaign name")
    local = Path("outputs") / args.name
    control = f"/zhome/95/e/206121/dbmol-control/{args.name}"
    if args.phase == "stage":
        if (not args.config or not re.fullmatch(r"restart-[a-z0-9-]+\.json", args.config)
                or not args.max_updates or not 1 <= args.hours <= 24):
            parser.error("Staging requires a restart config, update limit, and 1–24 hour walltime")
        stage(args, local, control, f"{BASE}/releases/{args.name}")
    else:
        submit(local, control)


if __name__ == "__main__":
    main()
