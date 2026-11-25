"""Stage paired combined-gradient BRD4 searches and run bounded calibration on DTU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shlex
import subprocess
import tarfile

from submit_repair_search import CUDA_HASH, CUDA_HELPER, EXP, NATIVE_ROOT, NATIVE_SOURCE, OPS, ROUTER


CONFIG = EXP / "combined-search.json"
PLAN = EXP / "gpu-plan.json"
WRAPPER = EXP / "combined_search_env.sh"
NESSO_ASSETS = "/work3/s234818/latentmol/assets/nesso1-brd4-20261120a"
CONTROL = "/zhome/95/e/206121/dbmol-control"


def baseline_comparisons(folder, cfg, tasks, ops):
    """Pair each confidence job with the already-submitted fixed job for its seed."""
    reports = {}
    old_plan = ops["read"](folder / "gpu-plan.json")
    with tarfile.open(folder / "source.tar.gz") as source:
        for name, task in tasks.items():
            if task["policy"] != "confidence" or not task["updates"]:
                continue
            old_name = f"fixed-{task['seed']}"
            prior = old_plan["tasks"][old_name]
            receipt = ops["read"](folder / f"{old_name}-submission.json")
            command = receipt["argv"]
            if (receipt["returncode"] or type(receipt["job_id"]) is not int or prior["seed"] != task["seed"]
                    or prior["policy"] != "fixed" or command[command.index("--output")+1] != prior["output"]):
                raise ValueError(f"Invalid paired fixed submission for {name}")
            old_config_bytes = source.extractfile(command[command.index("--config")+1]).read()
            if hashlib.sha256(old_config_bytes).hexdigest() != old_plan["config_sha256"]:
                raise ValueError("Fixed comparison config digest changed")
            old = json.loads(old_config_bytes)
            common = ("founder_updates", "child_updates", "parents", "children_per_parent", "repair_weights",
                      "learning_rate", "max_step_norm", "occupied_atoms", "update_noise", "restart_mass",
                      "restart_mass_after_merge", "probability_floor", "connectivity_operator", "vina_parent_panel")
            if (any(old[k] != cfg[k] for k in common) or old["child_generations"] > cfg["child_generations"]
                    or old["search_versions"]["fixed"]["vina_weight"] != cfg["search_versions"]["confidence"]["vina_weight"]):
                raise ValueError("Confidence run differs from the fixed baseline on shared search settings")
            reports[f"compare-{task['seed']}"] = dict(task=name, seed=task["seed"], fixed=prior["output"],
                fixed_job=receipt["job_id"], matched_updates=prior["updates"], confidence=task["output"],
                fixed_launch_receipt=command[command.index("--receipt")+1], output=task["output"] + "-comparison.json")
    if not reports:
        raise ValueError("The supplied fixed comparison needs at least one confidence job")
    return reports


def stage(args, ops, local, control):
    config_path = args.config.resolve().relative_to(Path.cwd().resolve())
    cfg = ops["read"](config_path)
    updates = cfg["founder_updates"] + (cfg["parents"] * cfg["children_per_parent"]
                                        * cfg["child_generations"] * cfg["child_updates"])
    if len(args.queues) != len(cfg["seeds"]) or updates < 1:
        raise ValueError("Require one GPU queue per paired seed and a positive update budget")
    policies = args.policies or list(cfg["search_versions"])
    if not policies or len(set(policies)) != len(policies) or not set(policies) <= set(cfg["search_versions"]):
        raise ValueError("Select distinct policies listed in the config's search_versions")
    calibration_policy = next((name for name in ("confidence", "fixed") if name in policies), policies[0])
    tasks = {}
    for seed, queue in zip(cfg["seeds"], args.queues, strict=True):
        for policy in policies:
            name = f"{policy}-{seed}"
            tasks[name] = dict(seed=seed, policy=policy, queue=queue, updates=updates,
                               output=f"{ops['BASE']}/runs/{args.name}-{name}")
        smoke = f"calibrate-{seed}"
        tasks[smoke] = dict(seed=seed, policy=calibration_policy, queue="gpua100i", updates=0,
                            output=f"{ops['BASE']}/runs/{args.name}-{smoke}")
    plan = dict(config_sha256=ops["sha"](config_path), cuda_ownership_helper=CUDA_HELPER,
                cuda_ownership_sha256=CUDA_HASH, tasks=tasks)
    reports = baseline_comparisons(Path("outputs") / args.fixed_campaign, cfg, tasks, ops) if args.fixed_campaign else {}
    if reports:
        plan["comparisons"] = reports
    required = [NATIVE_SOURCE + "/experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_math.py",
                NATIVE_ROOT + "/assets.json", "/work3/s234818/mosaic/envs/protenix-tiny/bin/python",
                ops["BASE"] + "/env/boltz311/bin/python", NESSO_ASSETS + "/env/bin/python",
                NESSO_ASSETS + "/checkpoint/model.safetensors", NESSO_ASSETS + "/esm/model.safetensors",
                NESSO_ASSETS + "/ccd.pkl", cfg["vina_parent_panel"]["python"],
                *(cfg["vina_parent_panel"][key]["path"] for key in ("box","receptor"))]
    required_code = [EXP / f for f in ("combined_search.py", "combined_native.py", "combined_nesso_worker.py")]
    if reports:
        required_code.append(EXP / "compare_confidence.py")
    for f in required_code:
        if not f.is_file():
            raise FileNotFoundError(f)
    local.mkdir(exist_ok=False)
    ops["save"](local / "gpu-plan.json", plan)
    archive = local / "source.tar.gz"
    files = [*sorted(Path("src/latentmol").rglob("*.py")), *sorted(EXP.glob("*.py")),
             EXP / "protenix_env.sh", WRAPPER, ROUTER, config_path]
    with tarfile.open(archive, "x:gz") as bundle:
        for f in files:
            if f.suffix == ".sh" and b"\r" in f.read_bytes():
                raise ValueError(f"Linux shell has carriage returns: {f}")
            bundle.add(f, arcname=f.as_posix())
        bundle.add(local / "gpu-plan.json", arcname=PLAN.as_posix())
    checks = ops["remote"](f"""
import json, shutil, subprocess
from pathlib import Path
control=Path({control!r})
tasks={tasks!r}
reports={reports!r}
for p in [control, *(Path(t["output"]) for t in [*tasks.values(), *reports.values()])]:
    if p.exists(): raise FileExistsError(p)
for p in {required!r}:
    if not Path(p).is_file(): raise FileNotFoundError(p)
queues=sorted({{t["queue"] for t in tasks.values()}})
rows=subprocess.check_output(["bqueues", *queues], text=True)
if {{x[0] for line in rows.splitlines() if len(x:=line.split())>2 and x[2]=="Open:Active"}} != set(queues):
    raise RuntimeError(rows)
control.mkdir()
for task in [*tasks, *reports]: (control/task).mkdir()
print(json.dumps(dict(status="passed", queues=rows, free_bytes=shutil.disk_usage({ops["BASE"]!r}+"/runs").free)))
""")
    ops["save"](local / "storage.json", checks)
    for src, dst in [(archive, "source.tar.gz"), (ops["LAUNCHER"], "hpc_launch.py")]:
        ops["upload"](src, control + "/" + dst)
    digest = ops["sha"](archive)
    commands, interactive = {}, {}
    for key, task in tasks.items():
        folder = f"{control}/{key}"
        smoke = task["queue"] == "gpua100i"
        command = ["/usr/bin/python3", f"{control}/hpc_launch.py", "--archive", f"{control}/source.tar.gz",
                   "--archive-sha256", digest, "--release", f"{control}/source",
                   "--receipt", f"{folder}/launch.json", "--startup-seconds", "850" if smoke else "2700",
                   "--run-seconds", "890" if smoke else str(args.hours * 3600 - 90),
                   "--", "bash", WRAPPER.as_posix(), "--config", config_path.as_posix(),
                   "--plan", PLAN.as_posix(), "--plan-sha256", ops["sha"](local / "gpu-plan.json"),
                   "--task", key, "--seed", str(task["seed"]), "--policy", task["policy"], "--output", task["output"]]
        if smoke:
            interactive[key] = command + ["--calibration-only"]
            continue
        a100 = task["queue"] == "gpua100"
        commands[key] = ["bsub", "-J", f"{args.name}-{key}", "-q", task["queue"], "-n", "4",
                         "-R", "span[hosts=1]", "-R", "rusage[mem=8192]",
                         *(["-R", "select[gpu80gb]"] if a100 else []),
                         "-gpu", f"num=1:mode=exclusive_process:j_exclusive=yes:mps=yes:gmem={'64G' if a100 else '44G'}",
                         "-env", "all,~CUDA_VISIBLE_DEVICES", "-W", f"{args.hours:02d}:00",
                         "-oo", f"{folder}/run.%J.out", "-eo", f"{folder}/run.%J.err", *command]
    ops["save"](local / "commands.json", commands)
    ops["save"](local / "interactive-commands.json", interactive)
    print(f"Staged {len(commands)} jobs at {updates:,} updates each; source {digest}")


def submit_comparisons(local, control, ops):
    plan = ops["read"](local / "gpu-plan.json")
    if not plan.get("comparisons"):
        return
    folder = local / "reports"
    folder.mkdir(exist_ok=True)
    commands = {}
    for key, info in plan["comparisons"].items():
        receipt = ops["read"](local / f"{info['task']}-submission.json")
        if receipt["returncode"] or type(receipt["job_id"]) is not int:
            raise ValueError(f"Confidence job has no successful scheduler receipt: {key}")
        gpu_cmd = receipt["argv"]
        launcher = gpu_cmd[gpu_cmd.index("--archive")-2:gpu_cmd.index("--receipt")]
        job_dir = f"{control}/{key}"
        commands[key] = [
            "bsub", "-J", f"{local.name}-{key}", "-q", "hpc", "-n", "1", "-R", "rusage[mem=2048]",
            "-W", "00:15", "-w", f"ended({info['fixed_job']}) && ended({receipt['job_id']})",
            "-oo", job_dir + "/run.%J.out", "-eo", job_dir + "/run.%J.err", *launcher,
            "--receipt", job_dir + "/launch.json", "--startup-seconds", "800", "--run-seconds", "850", "--",
            "/usr/bin/python3", (EXP / "compare_confidence.py").as_posix(),
            "--fixed", info["fixed"], "--confidence", info["confidence"], "--output", info["output"],
            "--fixed-launch-receipt", info["fixed_launch_receipt"],
            "--confidence-launch-receipt", gpu_cmd[gpu_cmd.index("--receipt")+1]]
    for name, value in (("commands.json", commands), ("storage.json", ops["read"](local / "storage.json")),
                        ("gpu-plan.json", plan)):
        path = folder / name
        if path.exists() and ops["read"](path) != value:
            raise ValueError(f"Existing paired report differs: {name}")
        if not path.exists():
            ops["save"](path, value)
    ops["submit"](folder, control)


def calibrate(seed, ops, local, control):
    key = f"calibrate-{seed}"
    argv = ops["read"](local / "interactive-commands.json")[key]
    folder = control + "/" + key
    # a100sh allocates an interactive GPU. Model imports and execution happen
    # inside that job, while the login process only drives the terminal.
    wrapped = " ".join(map(shlex.quote, argv)) + " > " + shlex.quote(folder + "/worker.log") + " 2>&1"
    result = ops["remote"](f"""
import json, os, pexpect, pathlib, time
folder = pathlib.Path({folder!r})
with (folder/"terminal.log").open("x") as log:
    env=dict(os.environ,TERM="xterm")
    session=pexpect.spawn("/bin/bash",["-lc","exec a100sh"],encoding="utf-8",codec_errors="replace",env=env,timeout=120)
    session.logfile_read=log
    try:
        session.expect(r"Job <([0-9]+)> is submitted")
        job=int(session.match.group(1))
        (folder/"interactive-job.json").write_text(json.dumps(dict(job_id=job,queue="gpua100i")))
        session.expect("<<Starting on",timeout=150)
        time.sleep(1)
        session.sendline({wrapped!r}+"; code=$?; echo COMBO_CALIBRATION_RC=$code; exit $code")
        session.expect(r"COMBO_CALIBRATION_RC=([0-9]+)",timeout=950)
        rc=int(session.match.group(1))
        session.expect(pexpect.EOF,timeout=30)
    finally:
        session.close(force=True)
print(json.dumps(dict(job_id=job,returncode=rc,queue="gpua100i",log=str(folder/"worker.log"))))
""", timeout=1180)
    ops["save"](local / f"{key}-interactive.json", result)
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError(f"Native calibration failed; see {result['log']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("stage", "calibrate", "submit"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config", type=Path, default=CONFIG, help="Config to snapshot during staging")
    ap.add_argument("--policies", nargs="+", help="Stage only these search_versions, e.g. confidence")
    ap.add_argument("--fixed-campaign", help="Submit one paired CPU readout per confidence seed using this baseline")
    ap.add_argument("--hours", type=int, default=22)
    ap.add_argument("--queues", choices=("gpua100", "gpul40s", "gpua40"), nargs="+", default=["gpul40s","gpua100"])
    args = ap.parse_args()
    if (not re.fullmatch(r"combined-brd4-[a-z0-9-]+", args.name) or not 1 <= args.hours <= 24
            or args.fixed_campaign and not re.fullmatch(r"combined-brd4-[a-z0-9-]+", args.fixed_campaign)):
        ap.error("Supply a unique combined-brd4 name and 1–24 hours")
    local = Path("outputs") / args.name
    control = CONTROL + "/" + args.name
    ops = runpy.run_path(str(OPS))
    if args.phase == "stage":
        stage(args, ops, local, control)
    elif args.phase == "calibrate":
        if args.seed is None:
            ap.error("calibrate requires --seed")
        calibrate(args.seed, ops, local, control)
    else:
        ops["submit"](local, control)
        submit_comparisons(local, control, ops)


if __name__ == "__main__":
    main()
