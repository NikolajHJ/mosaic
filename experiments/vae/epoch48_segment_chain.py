"""Continue the seven-round motif arm in 24-hour allocations through its epoch boundary.

Run `submit plan.json` on the DTU login node. Each success callback inspects the
finished worker and queues at most one successor and its callback. All scratch
reads for this check have a timeout. A failed worker never advances the chain.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

# On DTU the submission bundle puts both existing helpers next to this file.
if not Path(__file__).with_name("vae_segment_chain.py").is_file():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/hpc"))
from hpc_launch import bounded_file
from vae_segment_chain import read, submit, write_once


CONFIG = "experiments/vae/epoch48-seven-round-motifs.json"
CONFIG_SHA = "014b79d0626ffa0cd76134a5e4a3ccdf74d4397d69bade2a298ffd653a363bba"
FIRST_SOURCE_SHA = "34082d26a2056e9393f27786bca2ca6aca3221ac3c7897a71b62b4f7298f2dc8"
DATA_SHA = "64247d04ec1f94d00320bb1e5a0d9c5540419eb37830050d3100775bf4385e18"
PARENT_UPDATES = 575498
TRAIN_SIZE = 60_000_000
BATCH = 240
PREFIX = "e48-r7-complete"
PYTHON = "/usr/bin/python3"


def prefix(plan):
    if not plan or "handoff" not in plan:
        return PREFIX
    value = plan.get("job_prefix", "e48-r7-rings")
    if not re.fullmatch(r"e48-r7-[a-z0-9]+", value) or value == PREFIX:
        raise ValueError("Invalid epoch-48 job prefix")
    return value


def name(segment, *, after=False, plan=None):
    label = prefix(plan)
    return f"{label}-{'after-' if after else ''}s{segment:02d}"


def previous_plan(plan):
    handoff = plan["handoff"]
    source = Path(handoff["control"]).resolve(strict=True)
    raw = (source / "plan.json").read_bytes()
    old = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != handoff["plan_sha256"]
            or ("handoff" in old and (old["handoff"]["segment"] >= handoff["segment"]
                                     or prefix(old) == prefix(plan)))
            or handoff["segment"] not in (2, 3) or plan["output_root"] == old["output_root"]
            or any(plan[k] != old[k] for k in ("first_job_id", "first_directory", "first_launch_receipt", "parent_receipt"))
            or handoff["worker_id"] != job_id(source, old, 2)
            or handoff["callback_id"] != int(read(source / (name(2, after=True, plan=old) + ".receipt.json"))["id"])):
        raise ValueError("Conflicting previous epoch-48 plan, parent, or scheduler receipts")
    return source, old


def output(plan, segment):
    if segment == 1:
        return Path(plan["first_directory"])
    if "handoff" in plan and segment < plan["handoff"]["segment"]:
        return output(previous_plan(plan)[1], segment)
    return Path(plan["output_root"]) / f"segment-{segment:02d}"


def job_id(control, plan, segment):
    if segment == 1:
        return int(plan["first_job_id"])
    if "handoff" in plan and segment < plan["handoff"]["segment"]:
        directory, old = previous_plan(plan)
        return job_id(directory, old, segment)
    return int(read(control / (name(segment, plan=plan) + ".receipt.json"))["id"])


def inspect_fork(plan, segment):
    directory = output(plan, segment)
    parent = read(plan["parent_receipt"])
    fork = bounded_file(directory / "fork.json")
    config_sha = CONFIG_SHA if segment == 1 else plan["config_sha256"]
    identity = hashlib.sha256(f"{config_sha}:{parent['sha256']}:3".encode()).hexdigest()
    if (parent["epoch"] != 47 or parent["row"] != 0 or parent["updates"] != PARENT_UPDATES
            or fork["parent"] != parent or fork["config_sha256"] != identity or fork["extra_rounds"] != 3):
        raise ValueError("Unexpected E47 parent or seven-round training recipe")
    start_updates = PARENT_UPDATES
    if segment == 1:
        if fork["target_updates"] != 25_000:
            raise ValueError("Unexpected first-leg update budget")
    else:
        start = bounded_file(output(plan, segment - 1) / "latest.json")
        if (fork["resume"] != start or fork["target_updates"] is not None
                or fork["target_epoch"] != 48 or fork["wall_seconds"] != 81_000):
            raise ValueError("Worker did not resume its assigned checkpoint and budget")
        start_updates = start["updates"]
    return fork, start_updates


def inspect(control, plan, segment):
    """Authenticate a *terminal* segment, including exact E48 row/update accounting."""
    if "handoff" in plan and segment < plan["handoff"]["segment"]:
        source, old = previous_plan(plan)
        return inspect(source, old, segment)
    directory = output(plan, segment)
    launch_path = (Path(plan["first_launch_receipt"]) if segment == 1
                   else control / (name(segment, plan=plan) + ".launch.json"))
    launch = read(launch_path)
    expected_sha = FIRST_SOURCE_SHA if segment == 1 else plan["archive_sha256"]
    if (launch.get("status") != "worker_exited" or launch.get("ready") is not True
            or launch.get("returncode") != 0 or launch["source_archive_sha256"] != expected_sha
            or int(launch["job_id"]) != job_id(control, plan, segment)):
        raise ValueError(f"Segment {segment} lacks a successful launch receipt for its assigned job")
    _, start_updates = inspect_fork(plan, segment)

    summary = bounded_file(directory / "summary.json")
    latest = bounded_file(directory / "latest.json")
    points = latest["real_points"]
    updates = latest["updates"]
    path = directory / f"checkpoint-{updates:09d}.pt"
    if (summary["latest"] != latest or summary["training_molecules"] != TRAIN_SIZE
            or summary["dataset_sha256"] != DATA_SHA or summary["batch_size"] != BATCH
            or summary["microbatch_size"] != 48 or not start_updates < updates
            or summary["segment_updates"] != updates - start_updates
            or not 0 < points <= TRAIN_SIZE or points != (updates - PARENT_UPDATES) * BATCH
            or Path(latest["checkpoint"]) != path or not re.fullmatch(r"[0-9a-f]{64}", latest["sha256"])
            or any(latest[k] != summary[k] for k in ("updates", "epoch", "row", "real_points"))
            or latest["completed_epoch"] != latest["epoch"]):
        raise ValueError("Invalid epoch-48 checkpoint, dataset, or cursor")
    if segment == 1 and updates - start_updates > 25_000:
        raise ValueError("First worker exceeded its sealed update budget")
    if points == TRAIN_SIZE:
        valid = (summary["status"] == "completed" and summary["stop_reason"] == "epoch_target_reached"
                 and latest["epoch"] == 48 and latest["row"] == 0)
    else:
        permitted = (summary["stop_reason"] == "allocation_budget_reached"
                     or (segment == 1 and summary["stop_reason"] == "update_budget_reached"
                         and summary["segment_updates"] == 25_000))
        valid = summary["status"] == "paused" and latest["epoch"] == 47 and latest["row"] == points and permitted
    if not valid or bounded_file(path, "hash", timeout=180) != latest["sha256"]:
        raise ValueError("Unexpected terminal state or checkpoint checksum")
    return summary


def saved_checkpoint(plan):
    """Authenticate the running or stopped ring worker's published post-selection save."""
    if "early_stop" not in plan or plan["handoff"]["segment"] != 3:
        raise ValueError("Early handoff requires the dedicated segment-3 replacement")
    old_control, old = previous_plan(plan)
    worker = plan["handoff"]["worker_id"]
    label = name(2, plan=old)
    launch = read(old_control / (label + ".launch.json"))
    ready = read(old_control / (label + ".launch.ready.json"))
    if (read(old_control / (label + ".claim.json")) != {"job": worker}
            or int(launch["job_id"]) != worker or launch["source_archive_sha256"] != old["archive_sha256"]
            or launch["status"] not in ("worker_ready", "worker_exited", "launch_failed")
            or ready.get("status") != "models_and_inputs_ready" or int(ready["job_id"]) != worker
            or not ready.get("nonce") or ready["nonce"] != launch["nonce"]):
        raise ValueError("Previous worker claim, source or authenticated readiness differs")
    directory = output(old, 2)
    fork, start = inspect_fork(old, 2)
    decision = bounded_file(directory / "activation" / "decision.json")
    latest = bounded_file(directory / "latest.json")
    updates = latest["updates"]
    points = latest["real_points"]
    path = directory / f"checkpoint-{updates:09d}.pt"
    if (decision["protocol"] != fork["config_sha256"] or decision["predecessor"] != fork["resume"]
            or decision["winner"] not in ("control", "candidate")
            or not start < decision["arms"][decision["winner"]]["latest"]["updates"] <= updates
            or latest["epoch"] != 47 or latest["completed_epoch"] != 47
            or not 0 < points < TRAIN_SIZE or latest["row"] != points
            or points != (updates - PARENT_UPDATES) * BATCH or Path(latest["checkpoint"]) != path
            or not re.fullmatch(r"[0-9a-f]{64}", latest["sha256"])
            or bounded_file(path, "hash", timeout=180) != latest["sha256"]):
        raise ValueError("Saved checkpoint differs from the selected arm, epoch cursor or checksum")
    return latest


def stopped_checkpoint(control, plan, terminal_row):
    """Bind the stopped writer's exact LSF identity to its last verifiable save."""
    source, old = previous_plan(plan)
    worker = plan["handoff"]["worker_id"]
    label = name(2, plan=old)
    argv = read(source / (label + ".receipt.json"))["argv"]
    if (int(terminal_row["JOBID"]) != worker or terminal_row["STAT"] != "EXIT"
            or terminal_row["USER"] != "s234818" or terminal_row["JOB_NAME"] != label
            or shlex.split(terminal_row["COMMAND"]) != argv[argv.index(PYTHON):]):
        raise ValueError("Missing terminal LSF identity for the stopped predecessor")
    raw = (control / "plan.json").read_bytes()
    if json.loads(raw) != plan:
        raise ValueError("Early handoff plan differs from disk")
    return dict(plan_sha256=hashlib.sha256(raw).hexdigest(), worker_id=worker,
                terminal=terminal_row, latest=saved_checkpoint(plan))


def validate_saved_payload(checkpoint, latest, fork, decision_path, config_path):
    """Read a downloaded save on CPU before stopping its writer. The login helper never imports Torch."""
    import numpy as np
    import torch
    from latentmol.models import GraphAutoencoder, ModelConfig
    from latentmol.representation import GraphSpec
    from latentmol.training.motif_activation import _digest, policies

    checkpoint = Path(checkpoint)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != latest["sha256"]:
        raise ValueError("Downloaded checkpoint SHA-256 differs from its published receipt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = Path(config_path).read_bytes().replace(b"\r\n", b"\n")
    config_sha = hashlib.sha256(config).hexdigest()
    identity = hashlib.sha256(f"{config_sha}:{fork['parent']['sha256']}:3".encode()).hexdigest()
    cursor = state["continuation"]
    updates = latest["updates"]
    if (state["config_sha256"] != identity or fork["config_sha256"] != identity
            or state["dataset_sha256"] != DATA_SHA or state["format_version"] != 1
            or state.get("diagnostic_only") or state["combined_version"] != "combined-vae-continuation-v1"
            or state["batch_size"] != BATCH or state["microbatch_size"] != 48
            or state["model_config"]["extra_bond_rounds"] != 3 or state["model_config"]["refinement_rounds"] != 4
            or state["epoch"] != 47 or cursor["epoch"] != 47 or cursor["epoch_complete"]
            or latest["completed_epoch"] != 47 or cursor["real_points"] != (updates - PARENT_UPDATES) * BATCH
            or not 0 < cursor["row"] == cursor["real_points"] < TRAIN_SIZE
            or any(cursor[k] != latest[k] for k in ("epoch", "row", "real_points", "updates"))):
        raise ValueError("Downloaded checkpoint differs from the production recipe or shuffle cursor")
    decision_raw = Path(decision_path).read_bytes()
    decision = json.loads(decision_raw)
    stamp = state["experiment_state"]["motif_activation"]
    winner = decision["winner"]
    _, arms = policies(json.loads(config))
    if (decision["protocol"] != identity or decision["predecessor"] != fork["resume"]
            or winner not in arms or stamp["protocol"] != identity or stamp["winner"] != winner
            or stamp["decision_sha256"] != hashlib.sha256(decision_raw).hexdigest()
            or Path(stamp["decision_file"]) != Path(latest["checkpoint"]).parent / "activation" / "decision.json"
            or stamp["policy_sha256"] != _digest(arms[winner])
            or stamp["adopted_at_update"] != decision["arms"][winner]["latest"]["updates"]
            or not fork["resume"]["updates"] < stamp["adopted_at_update"] <= updates):
        raise ValueError("Downloaded checkpoint differs from the recorded activation decision")
    with torch.random.fork_rng(devices=[]):
        model = GraphAutoencoder(GraphSpec.from_dict(state["graph_spec"]), ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(state["optimizer_state"])
    steps = set()
    for parameter in model.parameters():
        saved = optimizer.state.get(parameter, {})
        if not all(k in saved for k in ("step", "exp_avg", "exp_avg_sq")):
            raise ValueError("Missing saved AdamW moments or step counter")
        if (any(saved[k].shape != parameter.shape for k in ("exp_avg", "exp_avg_sq"))
                or not all(np.isfinite(t.detach().numpy()).all() for t in
                           (parameter, saved["exp_avg"], saved["exp_avg_sq"]))
                or not 0 < saved["step"].item() <= updates):
            raise ValueError("Incompatible or nonfinite saved model or AdamW state")
        steps.add(saved["step"].item())
    rng = cursor["rng"]
    if (not {"python", "numpy", "torch", "cuda", "loaders"}.issubset(rng) or not rng["cuda"]
            or "fresh_prior" not in rng["loaders"] or not all(
                v.dtype == torch.uint8 and v.numel() > 0 for v in (rng["torch"], *rng["cuda"], *rng["loaders"].values()))):
        raise ValueError("Missing CPU, CUDA or fresh-prior RNG state")
    return dict(sha256=digest, updates=updates, row=cursor["row"], winner=winner,
                decision_sha256=stamp["decision_sha256"], optimizer_parameters=len(optimizer.state),
                optimizer_steps=sorted(steps), loader_rngs=sorted(rng["loaders"]), cuda_rngs=len(rng["cuda"]))


def submit_once(control, label, args):
    receipt = control / (label + ".receipt.json")
    if receipt.exists():
        existing = read(receipt)
        if existing["argv"][-len(args):] != args:
            raise RuntimeError(f"Existing {label} submission has different arguments")
        return int(existing["id"])
    # The shared helper creates an exclusive intent before calling bsub. An
    # ambiguous submission stops for reconciliation, including concurrent submits.
    return submit(control, label, args)


def queue(plan_path, plan, segment, plan_sha):
    control = plan_path.parent
    script = str(Path(__file__).resolve())
    context = [str(plan_path), "--segment", str(segment), "--plan-sha256", plan_sha]
    predecessor = job_id(control, plan, segment - 1)
    worker = name(segment, plan=plan)
    switch = plan.get("handoff", {}).get("segment")
    # Cutover verifies the completed first leg before reserving its replacement.
    # This held job must also work after LSF ages the first leg out of history.
    dependency = [] if segment == switch == 2 else ["-w", f"done({predecessor})"]
    if segment == switch == 3 and "early_stop" in plan:
        dependency = ["-w", f"ended({predecessor})"]
    if not (control / (worker + ".receipt.json")).exists() and output(plan, segment).exists():
        raise FileExistsError(output(plan, segment))
    gpu = submit_once(control, worker, [
        "-q", "gpuv100", "-n", "8", "-R", "select[gpu32gb] span[hosts=1] rusage[mem=6144]",
        "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G", "-W", "24:00",
        "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        *(["-H"] if segment == switch else []), *dependency, PYTHON, script, "worker", *context])
    callback = submit_once(control, name(segment, after=True, plan=plan), [
        "-q", "hpc", "-n", "1", "-R", "rusage[mem=1024]", "-W", "00:15",
        "-w", f"ended({gpu})", PYTHON, script, "advance", *context])
    return {"segment": segment, "predecessor": predecessor, "worker": gpu, "callback": callback}


def run_worker(plan_path, plan, segment):
    control = plan_path.parent
    current = int(os.environ["LSB_JOBID"])
    # bsub can start a worker before the submitting process finishes writing its
    # receipt. Wait only for publication, never claim the segment for a foreign ID.
    deadline = time.monotonic() + 15
    while True:
        try:
            assigned = job_id(control, plan, segment)
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Missing or incomplete submission receipt for segment {segment}")
            time.sleep(.2)
    if current != assigned:
        raise ValueError(f"Allocated job {current} differs from assigned worker {assigned} for segment {segment}")
    previous = segment - 1
    worker = name(segment, plan=plan)
    switch = plan.get("handoff", {}).get("segment")
    if switch == segment:
        authorized = read(control / "cutover-authorized.json")
        if authorized != {"plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(), "worker": current}:
            raise RuntimeError("Scheduler replacement has not been authorized for this worker")
    if segment == switch == 3 and "early_stop" in plan:
        stopped = read(control / "stopped.json")
        if stopped_checkpoint(control, plan, stopped["terminal"]) != stopped:
            raise ValueError("Stopped predecessor checkpoint or LSF receipt changed")
    elif inspect(control, plan, previous)["status"] != "paused":
        raise RuntimeError("Predecessor already finished epoch 48")
    directory = output(plan, segment)
    if directory.exists():
        raise FileExistsError(directory)
    write_once(control / (worker + ".claim.json"), {"job": current})
    command = [PYTHON, str(control / "hpc_launch.py"), "--archive", plan["archive"],
               "--archive-sha256", plan["archive_sha256"], "--release", str(control / f"source-s{segment:02d}"),
               "--receipt", str(control / (worker + ".launch.json")), "--startup-seconds", "2400",
               "--run-seconds", "85000", "--", "/usr/bin/env", "LATENTMOL_CPU_THREADS=4",
               "LATENTMOL_EVALUATION_MODULE=latentmol.training.projected_continuation",
               "bash", "experiments/vae/run_checkpoint_comparison.sh", "--config", plan["config"],
               "--extra-rounds", "3", "--parent-receipt", plan["parent_receipt"],
               "--resume-receipt", str(output(plan, previous) / "latest.json"),
               "--wall-seconds", "81000", "--output", str(directory)]
    # A recipe transition authenticates the exact config of the incoming checkpoint.
    if segment == 3 and segment == switch:
        command.extend(["--previous-config", previous_plan(plan)[1]["config"]])
    elif segment == 2 and plan["config_sha256"] != CONFIG_SHA:
        command.extend(["--previous-config", CONFIG])
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "worker", "advance"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--segment", type=int, default=2)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    raw = plan_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    plan = json.loads(raw)
    config = Path(plan["config"])
    if (config.parent.as_posix() != "experiments/vae" or config.suffix != ".json" or args.segment < 2
            or (args.action != "submit" and args.plan_sha256 != sha)
            or not re.fullmatch(r"[0-9a-f]{64}", plan["config_sha256"])
            or not re.fullmatch(r"[0-9a-f]{64}", plan["archive_sha256"])):
        raise ValueError("Invalid release plan or segment identity")
    if args.action == "submit":
        if args.segment != plan.get("handoff", {}).get("segment", 2):
            parser.error("submit must identify the first segment in this plan")
        result = queue(plan_path, plan, args.segment, sha)
    elif args.action == "worker":
        run_worker(plan_path, plan, args.segment)
        return
    else:
        summary = inspect(plan_path.parent, plan, args.segment)
        if summary["status"] == "completed":
            completion = plan_path.parent / "epoch48-completed.json"
            result = {"segment": args.segment, "latest": summary["latest"]}
            if not completion.exists():
                write_once(completion, result)
            if read(completion) != result:
                raise ValueError("Conflicting epoch 48 completion receipt")
        else:
            result = queue(plan_path, plan, args.segment + 1, sha)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
