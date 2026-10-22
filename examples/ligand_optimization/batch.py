#!/usr/bin/env python3
"""Submit repeated ligand optimization runs to DTU's LSF scheduler."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples/ligand_optimization/run.py"
TERMINAL = {"FINISHED", "FAILED"}


def rows_from(markdown):
    rows = []
    for line in markdown.splitlines():
        if line.startswith("| ") and not line.startswith("| Run ") and not line.startswith("|---"):
            cells = [cell.strip() for cell in line.strip("| ").split("|")]
            if len(cells) == 4:
                rows.append(cells)
    return rows


def update_status(status_file, run_id, state, job_id="-", note="-"):
    lock_file = status_file.with_suffix(".lock")
    with lock_file.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = status_file.read_text(encoding="utf-8").splitlines()
        found = False
        updated = []
        for line in lines:
            if line.startswith(f"| {run_id} |"):
                current = line.strip("| ").split("|")[1].strip()
                if state == "SUBMITTED" and current in {"RUNNING", *TERMINAL}:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    return
                line = f"| {run_id} | {state} | {job_id} | {note.replace('|', '/')} |"
                found = True
            updated.append(line)
        if not found:
            raise ValueError(f"Run {run_id} is missing from {status_file}")
        rows = rows_from("\n".join(updated))
        done = sum(row[1] in TERMINAL for row in rows)
        succeeded = sum(row[1] == "FINISHED" for row in rows)
        failed = sum(row[1] == "FAILED" for row in rows)
        total = len(rows)
        if done == total:
            heading = f"# DONE — all {total} runs reached a final state ({succeeded} successful, {failed} failed)"
        else:
            heading = f"# IN PROGRESS — {done}/{total} runs finished ({succeeded} successful, {failed} failed)"
        updated[0] = heading
        handle, temporary = tempfile.mkstemp(dir=status_file.parent, prefix=".batch-status-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(updated) + "\n")
            os.replace(temporary, status_file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        fcntl.flock(lock, fcntl.LOCK_UN)


def run_worker(args):
    status_file = Path(args.status).resolve()
    output = Path(args.output).resolve()
    update_status(status_file, args.run_id, "RUNNING", args.job_id, "job started")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    command = [
        sys.executable, str(RUNNER), "--config", str(Path(args.config).resolve()),
        "--output", str(output),
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if result.returncode == 0:
            update_status(status_file, args.run_id, "FINISHED", args.job_id, "results saved")
        else:
            update_status(status_file, args.run_id, "FAILED", args.job_id, f"runner exited {result.returncode}")
        return result.returncode
    except Exception as error:
        update_status(status_file, args.run_id, "FAILED", args.job_id, f"{type(error).__name__}: {error}")
        raise


def submit(args):
    if not Path(".venv/bin/activate").is_file():
        raise SystemExit("Run this from the Mosaic repository on HPC; .venv/bin/activate was not found.")
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output folder must be empty or new: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    status_file = output_root / "batch_status.md"
    if status_file.exists():
        raise SystemExit(f"Status file already exists: {status_file}")

    configs = [Path(config).resolve() for config in args.configs]
    for config in configs:
        if not config.is_file():
            raise SystemExit(f"Config file not found: {config}")
    tasks = []
    index = 0
    for config in configs:
        for repeat in range(1, args.repeat + 1):
            index += 1
            run_id = f"{index:03d}-{config.stem}-run-{repeat:03d}"
            run_config = output_root / "configs" / f"{run_id}.json"
            contents = json.loads(config.read_text(encoding="utf-8"))
            base = config.parent
            def absolute_path(owner, key):
                value = owner.get(key)
                if value:
                    path = Path(value)
                    owner[key] = str(path if path.is_absolute() else (base / path).resolve())
            absolute_path(contents, "initialization")
            absolute_path(contents["target"], "structure")
            for key in ("cache_dir", "prepared_context", "checkpoint", "worker_python"):
                absolute_path(contents["model"], key)
            contents["seed"] = int(contents.get("seed", 17)) + repeat - 1
            run_config.parent.mkdir(parents=True, exist_ok=True)
            run_config.write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")
            tasks.append((run_id, run_config, output_root / run_id))

    table = [
        f"# IN PROGRESS — 0/{len(tasks)} runs finished (0 successful, 0 failed)", "",
        f"Batch of {len(tasks)} runs; created by `batch.py`.", "",
        "| Run | Status | HPC job | Note |", "|---|---|---:|---|",
    ]
    table.extend(f"| {run_id} | PREPARING | - | - |" for run_id, _, _ in tasks)
    status_file.write_text("\n".join(table) + "\n", encoding="utf-8")

    python = ".venv/bin/python"
    for run_id, config, output in tasks:
        (output_root / f"{run_id}.stdout.%J.log").parent.mkdir(parents=True, exist_ok=True)
        worker_command = " ".join(shlex.quote(item) for item in [
            python, str(Path(__file__).resolve()), "--worker",
            "--status", str(status_file), "--run-id", run_id,
            "--config", str(config), "--output", str(output), "--job-id", "${LSB_JOBID}",
        ])
        # Keep LSB_JOBID expandable by the job shell, rather than quoting it as a literal.
        worker_command = worker_command.replace("'${LSB_JOBID}'", '"${LSB_JOBID}"')
        login_command = f"cd {shlex.quote(str(ROOT))} && source .venv/bin/activate && {worker_command}"
        stdout = output_root / f"{run_id}.stdout.%J.log"
        stderr = output_root / f"{run_id}.stderr.%J.log"
        command = [
            "bsub", "-q", args.queue, "-n", str(args.cores),
            "-R", "span[hosts=1]", "-R", f"rusage[mem={args.memory}]",
            "-gpu", "num=1:mode=exclusive_process", "-W", args.walltime,
            "-J", run_id[:120], "-oo", str(stdout), "-eo", str(stderr),
            "bash", "-lc", login_command,
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        match = re.search(r"Job <(\d+)> is submitted", result.stdout)
        if result.returncode == 0 and match:
            update_status(status_file, run_id, "SUBMITTED", match.group(1), "waiting in HPC queue")
            print(f"Submitted {run_id}: job {match.group(1)}")
        else:
            detail = (result.stderr or result.stdout or f"bsub exited {result.returncode}").strip()
            update_status(status_file, run_id, "FAILED", "-", detail[:160])
            print(f"Could not submit {run_id}: {detail}", file=sys.stderr)
    print(f"Status tracker: {status_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--status")
    parser.add_argument("--run-id")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--job-id")
    parser.add_argument("--repeat", type=int, default=1, help="number of runs per config (default: 1)")
    parser.add_argument("--output-root", default="results/ligand_batch")
    parser.add_argument("--queue", default="gpua10")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--memory", default="8GB", help="memory request, e.g. 8GB")
    parser.add_argument("--walltime", default="02:00", help="LSF time limit, e.g. 02:00")
    parser.add_argument("configs", nargs="*")
    args = parser.parse_args()
    if args.worker:
        return run_worker(args)
    if not args.configs:
        parser.error("provide one or more config files")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    submit(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
