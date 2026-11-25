"""Stage or submit a native probability-space Protenix restart search on DTU LSF."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
import runpy
import tarfile


EXP = Path("experiments/latent-design/direct-jump")
OPS = EXP.parent / "protenix-vae/restart-operations/submit_campaign.py"
NATIVE_ROOT = "/work3/s234818/mosaic/runs/stage3-protenix-defog-lr60-3seeds-20261112a"
NATIVE_SOURCE = NATIVE_ROOT + "/projection-kkt-retry-20261117a/source"
CUDA_HELPER = "/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-release-26/source/scripts/cuda_ownership.py"
CUDA_HASH = "578d2b6c9453ad14ce6778d306aa6c1d926acb442dcf8ae9f9a75bccc6b2e3cb"
ROUTER = Path("scripts/hpc/protenix_latent_run.sh")


def stage(args, ops, local, control):
    remote, sha, save = (ops[k] for k in ("remote", "sha", "save"))
    cfg_path = args.config.resolve().relative_to(Path.cwd().resolve())
    cfg = ops["read"](cfg_path)
    seeds = cfg["seeds"]
    weights = cfg["repair_weights"]
    if not weights or len(set(weights)) != len(weights) or not all(math.isfinite(w) and w >= 0 for w in weights):
        raise ValueError("Repair weights must be distinct, finite and nonnegative")
    groups = [[w] for w in weights] if cfg.get("split_weight_jobs") else [weights]
    arm_updates = cfg["founder_updates"] + cfg["child_updates"] * cfg["parents"] * cfg["children_per_parent"] * cfg["child_generations"]
    updates = arm_updates * len(groups[0])
    if (args.updates_per_job is not None and updates != args.updates_per_job) or not all((EXP / k).is_file() for k in (
        "repair_search.py", "repair_report.py", "repair_signal.py", "repair_frontier.py", "protenix_native.py"
    )):
        raise ValueError("Worker is missing or the config differs from the declared update budget")
    if len(args.queues) != len(seeds) or set(args.queues) - {"gpuv100", "gpul40s", "gpua40", "gpua100", "gpua10"}:
        raise ValueError("Specify one DTU GPU queue per seed")
    plan = dict(config_sha256=sha(cfg_path), repair_weights=cfg["repair_weights"],
                report_output=f"{ops['BASE']}/runs/{args.name}-report", cuda_ownership_helper=CUDA_HELPER,
                cuda_ownership_sha256=CUDA_HASH, tasks={})
    for version in cfg["search_versions"]:
        if not re.fullmatch("[a-z][a-z0-9_-]*", version):
            raise ValueError("Invalid search version name")
        for seed, queue in zip(seeds, args.queues, strict=True):
            if type(seed) is not int or seed < 0:
                raise ValueError("Invalid search seed")
            for group in groups:
                task = f"{version}-{seed}"
                if cfg.get("split_weight_jobs"):
                    task += f"-w{group[0]:g}"
                plan["tasks"][task] = dict(seed=seed, policy=version, queue=queue, updates=updates,
                                           repair_weights=group, output=f"{ops['BASE']}/runs/{args.name}-{task}")
    if not plan["tasks"] or len(plan["tasks"]) != len(seeds) * len(cfg["search_versions"]) * len(groups):
        raise ValueError("Empty campaign or duplicate task")
    local.mkdir()
    report_local = local / "report"
    report_local.mkdir()
    plan_path = EXP / "gpu-plan.json"
    save(local / "gpu-plan.json", plan)
    archive = local / "source.tar.gz"
    with tarfile.open(archive, "x:gz") as bundle:
        for path in [*sorted(Path("src/latentmol").rglob("*.py")), *sorted(EXP.glob("*.py")),
                     *sorted(EXP.glob("*.sh")), cfg_path, ROUTER]:
            if path.suffix == ".sh" and b"\r" in path.read_bytes():
                raise ValueError(f"Shell script has CRLF: {path}")
            bundle.add(path, arcname=path.as_posix())
        bundle.add(local / "gpu-plan.json", arcname=plan_path.as_posix())
    tasks = plan["tasks"]
    vina = cfg["vina_parent_panel"]
    assets = {CUDA_HELPER: CUDA_HASH, NATIVE_ROOT + "/assets.json":
              "1015877ada015ac0b37dcd972e1b44e358eb5404633952bf1e9f98ff13194c19",
              **{vina[k]["path"]: vina[k]["sha256"] for k in ("box", "receptor")}}
    required = [NATIVE_SOURCE + "/experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_math.py",
                NATIVE_SOURCE + "/src/mosaic/small_molecule/Stage2/operators.py",
                "/work3/s234818/mosaic/envs/protenix-tiny/bin/python",
                "/work3/s234818/latentmol/env/boltz311/bin/python", vina["python"],
                str(PurePosixPath(vina["python"]).with_name("vina"))]
    preflight = remote(f"""
from pathlib import Path
import hashlib,json,shutil,subprocess
control=Path({control!r})
tasks={tasks!r}
for p in [control, Path({plan['report_output']!r}), *(Path(t['output']) for t in tasks.values())]:
    if p.exists(): raise FileExistsError(p)
for p in {required!r}:
    if not Path(p).is_file(): raise FileNotFoundError(p)
for p,s in {assets!r}.items():
    if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=s: raise ValueError('Changed input: '+p)
q=subprocess.run(['bqueues',*{sorted(set(args.queues))!r}],text=True,capture_output=True,check=True).stdout
active={{row[0] for line in q.splitlines() if len(row:=line.split())>2 and row[2]=='Open:Active'}}
if active!=set({sorted(set(args.queues))!r}):
    raise ValueError(q)
free=shutil.disk_usage({ops["BASE"]!r}+'/runs').free
control.mkdir()
for k in [*tasks, "report"]: (control/k).mkdir()
print(json.dumps(dict(status='passed',available_output_bytes=free,queues=q,inputs={assets!r})))
""")
    save(local / "storage.json", preflight)
    save(report_local / "storage.json", preflight)
    save(report_local / "gpu-plan.json", plan)
    files = [(archive, control + "/source.tar.gz"), (ops["LAUNCHER"], control + "/hpc_launch.py")]
    for source, dest in files:
        ops["upload"](source, dest)
    remote(f"""
import hashlib,json
from pathlib import Path
for dest,digest in {[(dest,sha(source)) for source,dest in files]!r}:
    if hashlib.sha256(Path(dest).read_bytes()).hexdigest()!=digest: raise ValueError('Changed upload: '+dest)
print(json.dumps(True))
""")
    commands = {}
    for name, task in tasks.items():
        folder = control + "/" + name
        commands[name] = [
            "bsub", "-J", f"{args.name}-{name}", "-q", task["queue"], "-n", "4",
            "-R", "span[hosts=1]", "-R", "rusage[mem=4096]",
            "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=16G",
            "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "-W", f"{args.hours:02d}:00", "-oo", folder + "/run.%J.out", "-eo", folder + "/run.%J.err",
            "/usr/bin/python3", control + "/hpc_launch.py", "--archive", control + "/source.tar.gz",
            "--archive-sha256", sha(archive), "--release", control + "/source", "--receipt", folder + "/launch.json",
            "--startup-seconds", "2700", "--run-seconds", str(args.hours * 3600 - 120),
            "--", "bash", ROUTER.as_posix(), "bash", (EXP / "repair_search_env.sh").as_posix(),
            "--config", cfg_path.as_posix(), "--seed", str(task["seed"]), "--policy", task["policy"], "--task", name,
            "--output", task["output"], "--plan", plan_path.as_posix(), "--plan-sha256", sha(local / "gpu-plan.json"),
        ]
    save(local / "commands.json", commands)
    print(f"Staged {len(tasks)} GPU jobs, {updates} updates per job; source {sha(archive)}")


def submit_report(ops, local, control):
    read, save = ops["read"], ops["save"]
    plan = read(local / "gpu-plan.json")
    gpu_commands = read(local / "commands.json")
    ids = []
    for task, command in gpu_commands.items():
        receipt = read(local / f"{task}-submission.json")
        if receipt["argv"] != command or receipt["returncode"] or type(receipt["job_id"]) is not int:
            raise ValueError(f"Missing confirmed GPU submission for {task}")
        ids.append(receipt["job_id"])
    if not ids or set(gpu_commands) != set(plan["tasks"]):
        raise ValueError("Expected all staged GPU submissions")
    dependency = " && ".join(f"ended({job})" for job in ids)
    folder = control + "/report"
    digest = ops["sha"](local / "source.tar.gz")
    for command in gpu_commands.values():
        if command[command.index("--archive-sha256")+1] != digest:
            raise ValueError("Local source differs from the GPU submissions")
    command = [
        "bsub", "-J", local.name + "-report", "-q", "hpc", "-n", "1", "-R", "rusage[mem=2048]",
        "-W", "00:20", "-w", dependency, "-oo", folder + "/run.%J.out", "-eo", folder + "/run.%J.err",
        "/usr/bin/python3", control + "/hpc_launch.py", "--archive", control + "/source.tar.gz",
        "--archive-sha256", digest, "--release", control + "/source", "--receipt", folder + "/launch.json",
        "--startup-seconds", "1050", "--run-seconds", "1100", "--", "/usr/bin/python3",
        (EXP / "repair_report.py").as_posix(), "--plan", (EXP / "gpu-plan.json").as_posix(),
        "--output", plan["report_output"],
    ]
    file = local / "report/commands.json"
    if file.exists():
        if read(file) != {"report": command}:
            raise ValueError("Preserve previous report submission")
    else:
        save(file, {"report": command})
    ops["submit"](file.parent, control)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "submit", "submit-report"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--queues", nargs="+", default=["gpuv100", "gpua40", "gpul40s"])
    parser.add_argument("--config", type=Path, default=EXP / "repair-search-strong.json")
    parser.add_argument("--hours", type=int, default=16)
    parser.add_argument("--updates-per-job", type=int, help="Optionally check the per-job budget computed from the configuration")
    args = parser.parse_args()
    if (not re.fullmatch("protenix-repair-[a-z0-9-]+", args.name) or not 1 <= args.hours <= 24 or
            (args.updates_per_job is not None and args.updates_per_job < 1)):
        parser.error("Use a unique protenix-repair name, a positive budget and 1–24 hours")
    ops = runpy.run_path(str(OPS))
    local = Path("outputs") / args.name
    control = "/zhome/95/e/206121/dbmol-control/" + args.name
    if args.phase == "stage":
        stage(args, ops, local, control)
    elif args.phase == "submit":
        ops["submit"](local, control)
        submit_report(ops, local, control)
    else:
        submit_report(ops, local, control)


if __name__ == "__main__":
    main()
