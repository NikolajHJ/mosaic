"""Advance successful VAE segments and start the next recipe at epoch completion."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def submit(control, name, args):
    # Claim the unique job name before bsub. An interrupted submission needs
    # scheduler reconciliation before any retry with the same name.
    command = ["bsub", "-J", name, "-oo", str(control / f"{name}.%J.out"),
               "-eo", str(control / f"{name}.%J.err"), *args]
    write_once(control / f"{name}.intent.json", {"argv": command})
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    match = re.search(r"Job <(\d+)>", result.stdout)
    if not match:
        raise RuntimeError(f"Unknown LSF submission outcome for {name}")
    job = int(match[1])
    write_once(control / f"{name}.receipt.json", dict(id=job, argv=command, stdout=result.stdout))
    return job


def callback(plan_path, phase, segment, job, *, other_jobs=()):
    control = plan_path.parent
    prefix = read(plan_path).get("job_prefix", "dbmol-vae60m")
    name = f"{prefix}-after-{phase}-{segment:02d}"
    dependency = " && ".join(f"done({item})" for item in (job, *other_jobs))
    return submit(control, name, ["-q", "hpc", "-n", "1", "-R", "rusage[mem=1024]",
        "-W", "00:15", "-w", dependency, "/usr/bin/python3", str(Path(__file__).resolve()),
        "advance", str(plan_path), phase, str(segment)])


def output(plan, phase, segment):
    if phase == "combined" and segment == 0:
        return Path(plan["initial_output"])
    return Path(plan["run_root"]) / f"{phase}-{segment:02d}"


def checkpoint(summary, directory, recipe):
    points = summary["real_points"]
    parent_epoch = recipe.get("parent_epoch", 45)
    cursor = summary["latest"]
    path = Path(cursor["checkpoint"])
    if (summary["dataset_sha256"] != recipe["dataset_sha256"]
            or summary["training_molecules"] != recipe["training_molecules"]
            or not 0 < points <= recipe["training_molecules"]
            or summary["segment_updates"] < 1
            or not re.fullmatch(r"checkpoint-\d+\.pt", path.name)
            or path.parent != directory or not path.is_file()
            or not re.fullmatch(r"[a-f0-9]{64}", cursor["sha256"])
            or any(cursor[key] != summary[key] for key in ("real_points", "epoch", "row", "updates"))):
        raise ValueError("Invalid VAE segment summary or resume checkpoint")
    if summary["status"] == "completed":
        if (summary["stop_reason"] != "epoch_target_reached" or summary["epoch"] != parent_epoch + 1
                or summary["row"] != 0 or points != recipe["training_molecules"]
                or cursor["completed_epoch"] != parent_epoch + 1):
            raise ValueError("Completed segment does not cover the whole epoch")
    elif (summary["status"] != "paused" or summary["stop_reason"] != "allocation_budget_reached"
          or summary["epoch"] != parent_epoch or summary["row"] != points or points >= recipe["training_molecules"]):
        raise ValueError("Segment has no authorized continuation")
    return ["--resume", str(path), "--resume-sha256", cursor["sha256"]]


def launch(plan_path, plan, phase, segment, flags, *, callback_jobs=()):
    control = plan_path.parent
    directory = output(plan, phase, segment)
    if directory.exists():
        raise FileExistsError(directory)
    recipe = plan["combined" if phase == "combined" else "sparse"]
    name = f"{plan.get('job_prefix', 'dbmol-vae60m')}-{phase}-{segment:02d}"
    is_benchmark = phase in ("benchmark", "timing")
    command = [*recipe["worker"], "--receipt", str(control / f"{name}.launch.json"),
               "--startup-seconds", "2400", "--run-seconds", "4680" if is_benchmark else "84000",
               "--", *recipe["training"], "--output", str(directory), *flags]
    args = [*recipe["resources"], "-W", "02:00" if is_benchmark else "24:00", *command]
    job = submit(control, name, args)
    result = dict(job=job, output=str(directory))
    if phase != "timing":
        result["callback"] = callback(plan_path, phase, segment, job, other_jobs=callback_jobs)
    return result


def queue_benchmarks(plan_path, plan):
    directory = Path(plan["completed_combined"])
    summary = read(directory / "summary.json")
    _, path, _, sha = checkpoint(summary, directory, plan["combined"])
    if (summary["status"] != "completed" or summary["epoch"] != plan["sparse"]["parent_epoch"]
            or dict(path=path, sha256=sha) != plan["parent_checkpoint"]):
        raise ValueError("Benchmarks require the pinned completed combined epoch")
    timing = launch(plan_path, plan, "timing", 0, ["--benchmark", "--timing-study", plan["timing_study"]])
    benchmark = launch(plan_path, plan, "benchmark", 0, ["--benchmark"], callback_jobs=(timing["job"],))
    return dict(timing=timing, benchmark=benchmark)


def verify_handoff(plan):
    """A queued replacement may run only if the retired allocation never launched."""
    source = plan["superseded_segment"]
    if Path(source["launch_receipt"]).exists() or Path(source["output"]).exists():
        raise RuntimeError("Superseded segment launched or created outputs; reconcile its progress before resuming")
    if read(source["latest_path"]) != source["latest"]:
        raise RuntimeError("Saved predecessor cursor changed after the handoff was prepared")
    return {"verified_checkpoint": source["latest"]["checkpoint"]}


def advance(plan_path, plan, phase, segment):
    directory = output(plan, phase, segment)
    if phase == "benchmark":
        report_path = directory / "benchmark.json"
        report = read(report_path)
        recipe = plan["sparse"]
        if (report["status"] != "passed" or report["dataset_sha256"] != recipe["dataset_sha256"]
                or report["training_molecules"] != recipe["training_molecules"]):
            raise ValueError("Sparse VAE benchmark did not pass on the target dataset")
        if "timing_study" in plan:
            timing = read(output(plan, "timing", 0) / "timing.json")
            if (timing["status"] != "completed" or timing["dataset_sha256"] != recipe["dataset_sha256"]
                    or timing["training_molecules"] != recipe["training_molecules"]
                    or timing["config_sha256"] != recipe["config_sha256"]
                    or report["config_sha256"] != recipe["config_sha256"]):
                raise ValueError("Both A100 benchmarks must complete on the pinned training recipe")
        return launch(plan_path, plan, "sparse", 1,
                      ["--benchmark-report", str(report_path), "--wall-seconds", "82800"])

    recipe = plan[phase]
    summary = read(directory / "summary.json")
    flags = checkpoint(summary, directory, recipe)
    if phase == "combined" and summary["status"] == "completed":
        return launch(plan_path, plan, "benchmark", 0, ["--benchmark"])
    if summary["status"] == "completed":
        return {"completed": str(directory / "summary.json")}
    if segment >= recipe["max_segments"]:
        raise RuntimeError("VAE segment limit reached")
    previous = output(plan, phase, segment - 1) / "summary.json" if segment > 0 else None
    if phase == "combined" and previous is not None and summary["real_points"] <= read(previous)["real_points"]:
        raise ValueError("Combined VAE cursor did not advance")
    if phase == "sparse" and segment > 1 and summary["real_points"] <= read(previous)["real_points"]:
        raise ValueError("Sparse VAE cursor did not advance")
    if "resume_shape" in recipe:
        shape = recipe["resume_shape"]
        if any(type(shape[k]) is not int or shape[k] < 1 or summary.get(k) != shape[k]
               for k in ("batch_size", "microbatch_size")):
            raise ValueError("Segment changed the pinned resume batch shape")
        sizing = ["--batch-size", str(shape["batch_size"]), "--microbatch-size", str(shape["microbatch_size"])]
    else:
        report = recipe["benchmark_report"] if phase == "combined" else str(
            output(plan, "benchmark", 0) / "benchmark.json"
        )
        sizing = ["--benchmark-report", report]
    return launch(plan_path, plan, phase, segment + 1,
                  [*sizing, "--wall-seconds", "82800", *flags])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("queue", "queue-benchmarks", "advance", "verify-handoff"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("phase", choices=("combined", "benchmark", "sparse"), nargs="?", default="combined")
    parser.add_argument("segment", type=int, nargs="?", default=0)
    parser.add_argument("--exec-command", nargs=argparse.REMAINDER,
                        help="Start the supplied worker after verifying the retired segment")
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    plan = read(plan_path)
    if args.mode == "verify-handoff":
        result = verify_handoff(plan)
        if args.exec_command:
            os.execvp(args.exec_command[0], args.exec_command)
    elif args.exec_command is not None:
        parser.error("--exec-command requires verify-handoff")
    elif args.mode == "queue":
        result = {"callback": callback(plan_path, "combined", 0, plan["initial_job"])}
    elif args.mode == "queue-benchmarks":
        result = queue_benchmarks(plan_path, plan)
    else:
        result = advance(plan_path, plan, args.phase, args.segment)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
