"""Stage or submit one config-driven DTU restart campaign from the repository root."""

import argparse
import hashlib
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
GPU_RESOURCE = TRANSPORT["GPU_RESOURCE"] | {
    "gpua100": "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=36G",
}
OUTPUT_MIB = 768
OUTPUT_FILES = 6000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def matched_source(receipt_path, config):
    """Verify a learning-rate-only change against the original sealed inputs."""
    receipt = read(receipt_path)
    argv = receipt["argv"]
    if receipt["returncode"] or argv[0] != "bsub":
        raise ValueError("Matched source requires a successful submission receipt")
    path = receipt_path.with_name("source.tar.gz")
    digest = sha(path)
    if digest != argv[argv.index("--archive-sha256") + 1]:
        raise ValueError("Matched source archive checksum differs from the receipt")
    old_config = argv[argv.index("--config") + 1]
    old_plan = argv[argv.index("--plan") + 1]
    with tarfile.open(path) as source:
        original = json.load(source.extractfile(old_config))
        learning_rate = config["design"]["learning_rate"]
        if (original["design"]["learning_rate"] == learning_rate
                or any("learning_rate" in arm for arm in config["arms"].values())):
            raise ValueError("Expected one changed learning rate for all matched arms")
        original["design"]["learning_rate"] = learning_rate
        if original != config:
            raise ValueError("Matched configuration differs beyond the learning rate")
        raw_plan = source.extractfile(old_plan).read()
        if hashlib.sha256(raw_plan).hexdigest() != argv[argv.index("--plan-sha256") + 1]:
            raise ValueError("Matched GPU plan checksum differs from the receipt")
        for asset in (FRESH, NATIVE, ASSETS, WRAPPER):
            if source.extractfile(asset.as_posix()).read() != asset.read_bytes():
                raise ValueError(f"Matched launch input changed: {asset}")
    return path, digest, {old_config, old_plan}, json.loads(raw_plan)


def stage(args, local, control, release):
    config = EXPERIMENT / args.config
    cfg = read(config)
    study = cfg.get("study")
    benchmark = study == "batch_throughput"
    restart_search = study == "restart_search"
    uses_tasks = study in ("initialization_noise", "lineage_feedback", "batch_throughput")
    uses_cohorts = uses_tasks or restart_search
    fresh_path = EXPERIMENT / "restart-followup-fresh-dtu.json" if uses_cohorts else FRESH
    if restart_search:
        module = "latentmol.design.protenix_search"
    elif benchmark:
        module = "latentmol.design.protenix_restart_benchmark"
    else:
        module = "latentmol.design.protenix_restart_study" if uses_tasks else "latentmol.design.protenix_restarts"
    startup_seconds = {"initialization_noise": 1800, "lineage_feedback": 2700,
                       "batch_throughput": 2700, "restart_search": 2700}.get(study, 900)
    a100_80gb = benchmark or args.a100_80gb
    if args.a100_80gb and not restart_search:
        raise ValueError("Explicit 80 GB GPU selection requires a restart_search config")
    host_requirements = ["-R", "select[gpu80gb]"] if a100_80gb else []
    worker_env = "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
    outputs = {}
    budgets = {}
    queues = {}
    cpu_workers = {}
    configured = cfg["tasks" if uses_tasks else "arms"]
    selected = args.arm or list(configured)
    if len(set(selected)) != len(selected) or set(selected) - configured.keys():
        raise ValueError(f"Unknown or duplicate requested arms: {selected}")
    for arm in selected:
        override = configured[arm]
        if not re.fullmatch(r"[a-z][a-z0-9-]*", arm):
            raise ValueError(f"Invalid arm name: {arm}")
        cpu_workers[arm] = 4 if benchmark else 1
        if benchmark:
            cases = cfg["cases"]
            batches = {case["batch_size"] for case in cases}
            measured = (cfg["design"]["updates"] * cfg["parents"] * cfg["children_per_parent"]
                        * len(cases) * cfg["repeats"])
            budgets[arm] = measured + cfg["warmup_steps"] * sum(batches)
        elif restart_search:
            options = cfg["search"] | override.get("search", {})
            design = cfg["design"] | override.get("design", {})
            counts = [options[key] for key in ("waves", "parents", "children", "cpu_workers")]
            if any(type(count) is not int or count < 1 for count in [*counts, design["updates"]]):
                raise ValueError(f"Restart search requires positive integer budget counts for {arm}")
            branches_per_wave = options["parents"] * options["children"]
            if options["initialization"] == "archived":
                branches = options["waves"] * branches_per_wave
            elif options["initialization"] == "bootstrap":
                branches = 1 + (options["waves"] - 1) * branches_per_wave
            else:
                raise ValueError(f"Unknown restart initialization for {arm}")
            budgets[arm] = branches * design["updates"]
            cpu_workers[arm] = options["cpu_workers"]
        elif uses_tasks:
            comparison = cfg["comparisons"][override["comparison"]]
            if study == "lineage_feedback":
                budgets[arm] = (3 * comparison["parents"] * comparison["cycles"]
                                * comparison["branches_per_parent"] * comparison["updates"])
            else:
                budgets[arm] = comparison["parents"] * sum(
                    sum(variant["updates"]) for variant in comparison["variants"].values())
        else:
            campaign = cfg["campaign"] | cfg.get("campaign_arms", {}).get(arm, {})
            design = cfg["design"] | override
            branches = 1 + (campaign["cycles"] - 1) * campaign["elite_count"] * campaign["branches_per_elite"]
            budgets[arm] = branches * design["updates"]
        queues[arm] = override.get("queue", args.queue) if uses_cohorts else args.queue
        outputs[arm] = f"{BASE}/runs/{args.name}-{arm}"
    if not budgets or any(not 0 < updates <= args.max_updates for updates in budgets.values()):
        raise ValueError(f"Arm exceeds the declared update budget: {budgets}")
    if set(queues.values()) - GPU_RESOURCE.keys():
        raise ValueError(f"Unsupported GPU queues: {queues}")
    if benchmark and queues != {"bench": "gpua100"}:
        raise ValueError("The throughput benchmark requires one bench task on gpua100")
    if a100_80gb and set(queues.values()) != {"gpua100"}:
        raise ValueError("The 80 GB host requirement needs gpua100 for every selected arm")
    storage = cfg.get("storage", {}) if restart_search else {}
    output_mib = storage.get("output_cap_mib", OUTPUT_MIB)
    output_files = storage.get("output_file_cap", OUTPUT_FILES)
    if any(type(limit) is not int or limit < 1 for limit in (output_mib, output_files)):
        raise ValueError("Output capacity limits must be positive integers per arm")

    source_path = None
    if args.source_receipt:
        if uses_cohorts:
            raise ValueError("Matched archive staging is for learning-rate repeats")
        source_path, source_digest, replaced, prior = matched_source(args.source_receipt, cfg)
    else:
        prior = read(Path("outputs/protenix-restarts-20261107a/gpu-plan.json"))
    plan = {key: prior[key] for key in ("cuda_ownership_helper", "cuda_ownership_sha256", "mosaic_python")}
    plan.update(arms=outputs, queue=args.queue, planned_updates=budgets,
                output_cap_mib=output_mib, config_sha256=sha(config))
    if uses_cohorts:
        plan["queues"] = queues
        del plan["queue"]
    if restart_search:
        plan["a100_80gb"] = a100_80gb
        plan["output_file_cap"] = output_files
    if source_path:
        plan["matched_source_sha256"] = source_digest
    local.mkdir()
    save(local / "gpu-plan.json", plan)
    if b"\r\n" in WRAPPER.read_bytes():
        raise ValueError("Unexpected CRLF in launch wrapper")
    package = local / "source.tar.gz"
    remote_archive = release + ".tar.gz"
    plan_name = EXPERIMENT / "restart-campaign-gpu-plan.json"
    with tarfile.open(package, "x:gz") as bundle:
        if source_path:
            with tarfile.open(source_path) as source:
                for member in source:
                    if member.name in replaced:
                        continue
                    if not member.isfile():
                        raise ValueError(f"Unexpected matched archive member: {member.name}")
                    with source.extractfile(member) as stream:
                        bundle.addfile(member, stream)
        else:
            for path in [*sorted(Path("src/latentmol").rglob("*.py")), fresh_path, NATIVE, ASSETS, WRAPPER]:
                bundle.add(path, arcname=path.as_posix())
        bundle.add(config, arcname=config.as_posix())
        bundle.add(local / "gpu-plan.json", arcname=plan_name.as_posix())

    assets, native, fresh = map(read, (ASSETS, NATIVE, fresh_path))
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
    cohorts = cfg.get("parent_cohorts", []) if uses_cohorts else []
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
for root in map(Path, {cohorts!r}):
    if not (root / 'panel.json').is_file():
        raise FileNotFoundError(root / 'panel.json')
    fresh = json.loads((root / 'fresh' / 'summary.json').read_text())
    if fresh['status'] != 'completed' or fresh['geometry_seed'] != {fresh["geometry_seed"]!r}:
        raise ValueError('Source cohort fresh score protocol differs: ' + str(root))
queues = {sorted(set(queues.values()))!r}
scheduler = subprocess.run(['bqueues', *queues], capture_output=True, text=True, check=True, timeout=15)
active = {{fields[0] for line in scheduler.stdout.splitlines() if len(fields := line.split()) > 2
           and fields[2] == 'Open:Active'}}
if active != set(queues):
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
        '--cap-mib', {str(output_mib * len(outputs))!r}, '--file-cap', {str(output_files * len(outputs))!r}],
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
        queue = queues[arm]
        parallel_cpu = cpu_workers[arm] > 1
        cpu_slots = cpu_workers[arm] + 4 if parallel_cpu else 4
        # Override the wrapper's thread defaults after it resolves GPU ownership.
        thread_env = (["/usr/bin/env", "DBMOL_CPU_THREADS=1", "OMP_NUM_THREADS=1",
                       "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1"] if parallel_cpu else [])
        commands[arm] = [
            "bsub", "-J", f"dbmol-{args.name}-{arm}", "-q", queue, "-n", str(cpu_slots),
            "-R", "span[hosts=1]", "-R", "rusage[mem=4096]", *host_requirements,
            "-gpu", GPU_RESOURCE[queue],
            "-env", worker_env,
            "-W", f"{args.hours:02d}:00", "-oo", f"{receipts}/run.%J.out", "-eo", f"{receipts}/run.%J.err",
            "/usr/bin/python3", f"{control}/hpc_launch.py", "--archive", remote_archive,
            "--archive-sha256", archive_sha, "--release", release, "--receipt", f"{receipts}/launch.json",
            "--startup-seconds", str(startup_seconds), "--run-seconds", str(args.hours * 3600 - 900),
            "--", "bash", WRAPPER.as_posix(), *thread_env, torch_python, "-m", module,
            "--config", config.as_posix(), "--arm", arm, "--output", output,
            "--checkpoint", assets["checkpoint"], "--checkpoint-sha256", assets["checkpoint_sha256"],
            "--mosaic-python", plan["mosaic_python"], "--native-config", NATIVE.as_posix(),
            "--fresh-config", fresh_path.as_posix(), "--fresh-deps", fresh_deps, "--fresh-data", fresh_data,
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
    parser.add_argument("--arm", action="append", help="Stage only the named arm; repeat to select several")
    parser.add_argument("--source-receipt", type=Path, help="Reuse sealed source from a matched submission")
    parser.add_argument("--queue", choices=tuple(GPU_RESOURCE), default="gpul40s")
    parser.add_argument("--a100-80gb", action="store_true", help="Require an 80 GB A100 for a restart_search config")
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
        if args.arm or args.a100_80gb:
            parser.error("Arm and 80 GB GPU selection belong to staging; submit uses sealed commands")
        submit(local, control)


if __name__ == "__main__":
    main()
