"""Stage and submit the eight bounded Protenix restart follow-up comparisons."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONNECTOR = r"C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe"
BASE = "/work3/s234818/latentmol"
NAME = HERE.name
CONTROL = f"/zhome/95/e/206121/dbmol-control/{NAME}"
RELEASE = f"{BASE}/releases/{NAME}"
ARCHIVE = RELEASE + ".tar.gz"
EXPERIMENT = Path("experiments/latent-design/protenix-vae")
CONFIG = EXPERIMENT / "restart-followup.json"
FRESH = EXPERIMENT / "restart-followup-fresh-dtu.json"
NATIVE = EXPERIMENT / "native-dtu.json"
ASSETS = EXPERIMENT / "one-atom-dtu.json"
WRAPPER = Path("scripts/hpc/protenix_latent_run.sh")
LAUNCHER = Path("scripts/hpc/hpc_launch.py")
STORAGE = ROOT.parent / "mosaic/scripts/hpc/storage_accounting.py"
PLAN = EXPERIMENT / "restart-followup-gpu-plan.json"
MODULE = "latentmol.design.protenix_restart_study"
TORCH_PYTHON = f"{BASE}/env/boltz311/bin/python"
FRESH_ROOT = f"{BASE}/protenix_occupancy"
FRESH_DEPS = f"{FRESH_ROOT}/environment-attempt003/deps"
FRESH_DATA = f"{FRESH_ROOT}/data"
OUTPUT_MIB = 1280
OUTPUT_FILES = 12000
WALL = "06:00"
RUN_SECONDS = "21300"
STARTUP_SECONDS = "900"

# Keep both members of each within-task comparison on one GPU. Run one
# independent parent cohort on L40S and spread replication across two queues.
QUEUES = {
    "mutation-a": "gpul40s", "cap-a": "gpul40s",
    "duration-a": "gpul40s", "lineage-a": "gpul40s",
    "mutation-b": "gpua40", "cap-b": "gpuv100",
    "duration-b": "gpua40", "lineage-b": "gpuv100",
}
GPU_RESOURCE = {
    "gpul40s": "num=1:mode=exclusive_process:j_exclusive=yes:mps=no",
    "gpua40": "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=44G",
    "gpuv100": "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(filename, value):
    with (HERE / filename).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def remote(script, timeout=70):
    result = subprocess.run(
        [CONNECTOR, "exec", "--timeout", str(timeout), "bash -lc " + shlex.quote("/usr/bin/python3 -I -")],
        input=script, text=True, capture_output=True, timeout=timeout + 10)
    if result.returncode:
        raise RuntimeError(f"DTU exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def upload(source, target):
    subprocess.run([CONNECTOR, "upload", "--timeout", "90", str(source), target], timeout=100, check=True)


def submission_argv(task, output, plan, source_sha, plan_sha, assets):
    queue = QUEUES[task]
    receipt = f"{CONTROL}/{task}"
    return [
        "bsub", "-J", f"dbmol-protenix-followup-{task}-1107a", "-q", queue,
        "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=4096]",
        "-gpu", GPU_RESOURCE[queue],
        "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "-W", WALL, "-oo", f"{receipt}/run.%J.out", "-eo", f"{receipt}/run.%J.err",
        "/usr/bin/python3", f"{CONTROL}/hpc_launch.py", "--archive", ARCHIVE,
        "--archive-sha256", source_sha, "--release", RELEASE,
        "--receipt", f"{receipt}/launch.json", "--startup-seconds", STARTUP_SECONDS,
        "--run-seconds", RUN_SECONDS, "--", "bash", WRAPPER.as_posix(), TORCH_PYTHON,
        "-m", MODULE, "--config", CONFIG.as_posix(), "--arm", task,
        "--output", output, "--checkpoint", assets["checkpoint"],
        "--checkpoint-sha256", assets["checkpoint_sha256"], "--mosaic-python", plan["mosaic_python"],
        "--native-config", NATIVE.as_posix(), "--fresh-config", FRESH.as_posix(),
        "--fresh-deps", FRESH_DEPS, "--fresh-data", FRESH_DATA,
        "--plan", PLAN.as_posix(), "--plan-sha256", plan_sha,
    ]


def planned_updates(config):
    budgets = {}
    for task, assignment in config["tasks"].items():
        comparison = config["comparisons"][assignment["comparison"]]
        parents = comparison["parents"]
        if assignment["panel_offset"] + parents > config["panel_size"]:
            raise ValueError(f"Parent range exceeds declared panel size for {task}")
        if assignment["comparison"] == "lineage":
            # Elite and fresh-prior policies each spend the same update budget.
            updates = 2 * comparison["cycles"] * parents * comparison["branches_per_parent"] * comparison["updates"]
        else:
            updates = parents * sum(sum(arm["updates"]) for arm in comparison["variants"].values())
        if not 0 < updates <= 10000:
            raise ValueError(f"Update budget for {task} is {updates}, expected at most 10000")
        budgets[task] = updates
    return budgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands locally")
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        raise ValueError("Run the follow-up submitter from the DBMol repository root")
    config = read(CONFIG)
    if set(config["tasks"]) != set(QUEUES) or config["design"]["learning_rate"] != 150:
        raise ValueError("Expected eight declared comparisons at LR 150")
    parent_roots = list(config["parent_cohorts"])
    if len(set(parent_roots)) != len(parent_roots) or len(parent_roots) != 3 or any(
        not PurePosixPath(parent).is_absolute() or ".." in PurePosixPath(parent).parts
        or not PurePosixPath(parent).is_relative_to(PurePosixPath(BASE) / "runs")
        for parent in parent_roots
    ):
        raise ValueError("Expected three distinct existing work3 parent cohorts")
    parent_summaries = [f"{root}/fresh/summary.json" for root in parent_roots]
    fresh_config = read(FRESH)
    if FRESH.parent / fresh_config["assets_config"] != ASSETS:
        raise ValueError("Fresh scoring assets config differs from the packaged config")
    native = read(NATIVE)
    assets = read(ASSETS)
    previous = read(ROOT / "outputs/protenix-restarts-20261107a/gpu-plan.json")
    outputs = {task: f"{BASE}/runs/{NAME}-{task}" for task in config["tasks"]}
    plan = {key: previous[key] for key in
            ("cuda_ownership_helper", "cuda_ownership_sha256", "mosaic_python")}
    plan.update(tasks=outputs, queues=QUEUES, walltime=WALL, output_cap_mib=OUTPUT_MIB,
                planned_updates=planned_updates(config),
                parent_cohorts=parent_roots, config_sha256=sha(CONFIG), fresh_config_sha256=sha(FRESH))
    files = [CONFIG, FRESH, NATIVE, ASSETS, WRAPPER]
    for path in [Path("src") / (MODULE.replace(".", "/") + ".py"), LAUNCHER, STORAGE]:
        compile(path.read_bytes(), str(path), "exec")
    if b"\r\n" in WRAPPER.read_bytes():
        raise ValueError("Unexpected CRLF in GPU launch wrapper")
    if args.dry_run:
        print(json.dumps(dict(plan=plan, commands={
            task: submission_argv(task, output, plan, "<source-sha256>", "<plan-sha256>", assets)
            for task, output in outputs.items()
        }), indent=2))
        return

    save("gpu-plan.json", plan)
    package = HERE / "source.tar.gz"
    with tarfile.open(package, "x:gz") as bundle:
        for path in [*sorted(Path("src/latentmol").rglob("*.py")), *files]:
            bundle.add(path, arcname=path.as_posix())
        bundle.add(HERE / "gpu-plan.json", arcname=PLAN.as_posix())
    inputs = {assets["checkpoint"]: assets["checkpoint_sha256"],
              assets["native"]["weights_path"]: assets["native"]["weights_sha256"],
              native["context_ready"]: native["context_ready_sha256"],
              native["pocket_path"]: native["pocket_sha256"],
              fresh_config["pocket"]["path"]: fresh_config["pocket"]["sha256"],
              plan["cuda_ownership_helper"]: plan["cuda_ownership_sha256"]}
    required = [native["mosaic_source"], plan["mosaic_python"], assets["native"]["source_root"],
                TORCH_PYTHON, FRESH_DEPS, FRESH_DATA, fresh_config["ccd_root"], *parent_roots]
    probe = f"{BASE}/runs/{NAME}-capacity-probe"
    save("preflight.json", remote(f"""
import hashlib, json, subprocess
from pathlib import Path
for path in map(Path, {list(outputs.values()) + [ARCHIVE, RELEASE, CONTROL, probe]!r}):
    if path.exists():
        raise FileExistsError(path)
for name, expected in {inputs!r}.items():
    digest = hashlib.sha256()
    with Path(name).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('Changed input: ' + name)
for name in {required!r}:
    if not Path(name).exists():
        raise FileNotFoundError(name)
parent_hashes = {{}}
for name in {parent_summaries!r}:
    raw = Path(name).read_bytes()
    summary = json.loads(raw)
    if summary['status'] != 'completed' or summary['geometry_seed'] != {fresh_config["geometry_seed"]!r}:
        raise ValueError('Parent cohort fresh scoring contract differs: ' + name)
    parent_hashes[name] = hashlib.sha256(raw).hexdigest()
queues = {sorted(set(QUEUES.values()))!r}
scheduler = subprocess.run(['bqueues', *queues], check=True, capture_output=True, text=True, timeout=15).stdout
active = {{fields[0] for line in scheduler.splitlines() if len(fields := line.split()) > 2
          and fields[2] == 'Open:Active'}}
if active != set(queues):
    raise ValueError('Declared GPU queues are no longer active: ' + scheduler)
Path({CONTROL!r}).mkdir()
for task in {list(outputs)!r}:
    (Path({CONTROL!r}) / task).mkdir()
print(json.dumps(dict(inputs={inputs!r}, parent_summaries=parent_hashes, outputs={outputs!r}, scheduler=scheduler)))
"""))
    uploads = [(package, ARCHIVE), (LAUNCHER, f"{CONTROL}/hpc_launch.py"),
               (STORAGE, f"{CONTROL}/storage_accounting.py")]
    for source, destination in uploads:
        upload(source, destination)
    save("storage.json", remote(f"""
import hashlib, subprocess, tarfile
from pathlib import Path
for name, expected in {[(destination, sha(source)) for source, destination in uploads]!r}:
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise ValueError('Upload checksum differs: ' + name)
with tarfile.open({ARCHIVE!r}) as bundle:
    subprocess.run(['bash', '-n'], input=bundle.extractfile({WRAPPER.as_posix()!r}).read(), check=True)
probe = Path({probe!r})
probe.mkdir()
try:
    result = subprocess.run(['/usr/bin/python3', {uploads[-1][1]!r}, '--output-root', str(probe),
        '--cap-mib', {str(OUTPUT_MIB)!r}, '--file-cap', {str(OUTPUT_FILES)!r}],
        check=True, text=True, capture_output=True)
    print(result.stdout)
finally:
    probe.rmdir()
"""))
    source_sha = sha(package)
    plan_sha = sha(HERE / "gpu-plan.json")
    jobs = []
    for task, output in outputs.items():
        argv = submission_argv(task, output, plan, source_sha, plan_sha, assets)
        receipt = remote(f"""
import json, re, subprocess
from pathlib import Path
root = Path({f"{CONTROL}/{task}"!r})
argv = {argv!r}
with (root / 'submit-intent.json').open('x') as stream:
    json.dump(argv, stream)
result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
job = re.search(r'Job <([0-9]+)>', result.stdout)
receipt = dict(job_id=int(job[1]) if job else None, returncode=result.returncode,
               stdout=result.stdout, stderr=result.stderr, argv=argv, output={output!r}, queue={QUEUES[task]!r})
with (root / 'submission.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
""")
        save(f"{task}-submission.json", receipt)
        print(task, receipt["stdout"], receipt["stderr"], flush=True)
        if receipt["returncode"] or receipt["job_id"] is None:
            raise RuntimeError(f"Scheduler did not confirm {task}")
        jobs.append(str(receipt["job_id"]))
    save("scheduler-initial.json", remote(f"""
import json, subprocess
result = subprocess.run(['bjobs', '-w', *{jobs!r}], text=True, capture_output=True, timeout=20)
pending = subprocess.run(['bjobs', '-p', *{jobs!r}], text=True, capture_output=True, timeout=20)
print(json.dumps(dict(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode,
                      pending_stdout=pending.stdout, pending_stderr=pending.stderr)))
"""))


if __name__ == "__main__":
    main()
