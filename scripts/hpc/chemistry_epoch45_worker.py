"""Serial CUDA children for unchanged chemistry training and observations.

The LSF allocation uses exclusive-process CUDA. This module's coordinator is
standard-library/metadata-only: each CUDA owner exits before the next phase.
Tensor cleanup cannot replace that process boundary. The original 23-hour
shared deadline, 30-minute observation reserve, scientific gates and trainer
command remain authoritative.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


PROCESS_POLICY = "epoch45-serial-cuda-children-v1"


def _allocation():
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise ValueError("Epoch45 workflow requires an allocated job")


def _metadata(args):
    """Authenticate scheduler/source metadata without importing tensor code."""
    from chemistry_epoch45_campaign import Epoch45Campaign
    from latentmol.chemistry_epoch45_extension import validate_extension
    _allocation()
    campaign = Epoch45Campaign(args.plan, args.plan_sha256)
    plan = dict(campaign.plan, _plan_sha256=args.plan_sha256)
    validate_extension(plan)
    return campaign, plan


def _numerical_plan(args):
    """Keep the original readiness checks inside every scientific CUDA child."""
    from latentmol import chemistry_continuation as c
    from latentmol.chemistry_epoch45_extension import validate_extension
    from latentmol.experiments import file_sha256
    plan = c.read_plan(args.plan, args.plan_sha256)
    plan["_plan_sha256"] = args.plan_sha256
    validate_extension(plan)
    root = Path(plan["campaign"])
    plan["_benchmark_gate_sha256"] = file_sha256(root / "benchmark_gate.json")
    plan["_failure_readiness_sha256"] = file_sha256(root / "failure_readiness.json")
    c.require_failure_readiness(plan)
    return plan


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining < 1:
        raise RuntimeError("Epoch45 shared allocation deadline exhausted")
    return int(remaining)


def _require_cuda_phase(args, plan):
    import torch
    _allocation()
    if not torch.cuda.is_available():
        raise ValueError("Epoch45 scientific child requires allocated CUDA")
    if (args.deadline is None or not math.isfinite(args.deadline)
            or args.deadline - time.monotonic() > plan["protocol"]["segment_wall_seconds"]):
        raise ValueError("CUDA child lost the shared allocation deadline")
    _remaining(args.deadline)


def _phase_command(args, phase, deadline, **values):
    command = [sys.executable, str(Path(__file__).resolve()), "--plan", str(args.plan),
        "--plan-sha256", args.plan_sha256, "--segment", str(args.segment),
        "--phase", phase, "--deadline", repr(deadline)]
    for key, value in values.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    return command


def _run_child(command, root, segment, label, deadline):
    """A bounded, logged process owner; no CUDA query occurs in this parent."""
    from latentmol.cuda_process import run_cuda_child, require_cuda_child
    result = run_cuda_child(command, root / "phase-logs" / f"segment-{segment:03d}-{label}.log",
        timeout_seconds=_remaining(deadline))
    require_cuda_child(result)
    return result


def _ownership_probe(args):
    """Tiny native access proof; executed only as its own allocated CUDA child."""
    import torch
    from latentmol.experiments import write_json
    _, plan = _metadata(args)
    _require_cuda_phase(args, plan)
    root = Path(plan["campaign"]) / "cuda-ownership" / f"segment-{args.segment:03d}"
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"probe-{args.probe_index:03d}.json"
    if output.exists():
        raise FileExistsError("Preserve prior CUDA ownership evidence")
    value = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    if value.item() != 1.:
        raise RuntimeError("CUDA ownership probe failed")
    write_json(output, dict(status="passed", process_policy=PROCESS_POLICY,
        plan_sha256=args.plan_sha256, job_id=os.environ["LSB_JOBID"],
        child_pid=os.getpid(), coordinator_pid=os.getppid(),
        probe_index=args.probe_index, cuda_initialized=torch.cuda.is_initialized(),
        device=str(value.device), device_name=torch.cuda.get_device_name(value.device)))


def _check_ownership(args, root, deadline):
    """Two actual CUDA processes prove sequential access after pytest exits."""
    from chemistry_campaign import read
    from latentmol.cuda_process import require_cuda_free_coordinator
    from latentmol.experiments import file_sha256, write_json
    coordinator = require_cuda_free_coordinator()
    children = []
    for index in (1, 2):
        result = _run_child(_phase_command(args, "ownership-probe", deadline, probe_index=index),
            root, args.segment, f"ownership-probe-{index:03d}", deadline)
        path = root / "cuda-ownership" / f"segment-{args.segment:03d}" / f"probe-{index:03d}.json"
        proof = read(path)
        if (proof.get("status") != "passed" or proof.get("process_policy") != PROCESS_POLICY
                or proof.get("plan_sha256") != args.plan_sha256
                or proof.get("job_id") != os.environ["LSB_JOBID"]
                or proof.get("child_pid") != result["child_pid"]
                or proof.get("coordinator_pid") != os.getpid()
                or proof.get("probe_index") != index or proof.get("cuda_initialized") is not True):
            raise ValueError("Sequential CUDA child ownership proof differs")
        children.append(dict(path=str(path), sha256=file_sha256(path), **proof))
    after = require_cuda_free_coordinator()
    if len({row["child_pid"] for row in children}) != 2 or coordinator != after:
        raise ValueError("CUDA ownership processes were not separate from their coordinator")
    output = root / f"cuda-ownership-segment-{args.segment:03d}.json"
    if output.exists():
        raise FileExistsError("Preserve prior ownership handoff receipt")
    write_json(output, dict(status="passed_sequential_cuda_processes", process_policy=PROCESS_POLICY,
        plan_sha256=args.plan_sha256, coordinator=coordinator, children=children))


def _observe_baseline(args):
    """Original exact40 setup/handshake and real observer, isolated in one owner."""
    import torch
    from latentmol import chemistry_continuation as c
    from latentmol.chemistry_epoch45_extension import CHECKPOINT, CHECKPOINT_SHA
    from latentmol.chemistry_epoch45_generation import observe_checkpoint
    from latentmol.experiments import write_json
    from latentmol.train import capture_rng_state, state_digest
    plan = _numerical_plan(args)
    _require_cuda_phase(args, plan)
    root = Path(plan["campaign"])
    model, reference, optimizer, config, data, cohorts, generators, teacher, epoch, updates, algorithm, runtime = c.setup(
        plan, resume=(Path(CHECKPOINT), CHECKPOINT_SHA))
    try:
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        checks = dict(epoch40=epoch == 40, updates=updates == 263840,
            model_exact=state_digest(model.state_dict()) == state_digest(payload["model_state"]),
            optimizer_exact=c.nested_equal(optimizer.state_dict(), payload["optimizer_state"]),
            rng_exact=c.nested_equal(capture_rng_state(generators, True), payload["continuation"]["rng"]),
            frozen36=model._valid_reference_anchor.assert_unchanged()["reference_epoch"] == 36)
        if not all(checks.values()):
            write_json(root / "epoch40-handoff.json", dict(status="failed", checks=checks))
            raise RuntimeError("Exact epoch40 handoff checks failed")
        observation = observe_checkpoint(plan, model, generators, root / "generation/epoch0040",
            checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA, epoch=epoch, updates=updates)
        write_json(root / "epoch40-handoff.json", dict(status="passed", checks=checks,
            plan_sha256=args.plan_sha256, checkpoint_sha256=CHECKPOINT_SHA, observation=observation))
    finally:
        teacher.close()
    # Context release is process exit, not del/empty_cache. No child is spawned
    # from this phase; the still-CUDA-free coordinator launches training next.


def _observe_saved(args):
    import torch
    from latentmol.chemistry_epoch45_generation import observe_checkpoint
    from latentmol.evaluate import load_model
    from latentmol.experiments import write_json
    plan = _numerical_plan(args)
    _require_cuda_phase(args, plan)
    model, payload = load_model(Path(args.checkpoint), "cuda")
    if (payload["epoch"] != args.epoch
            or payload["chemistry_phase"].get("epoch45_extension") != plan["epoch45_extension"]):
        raise ValueError("Observed checkpoint lost extension identity")
    root = Path(plan["campaign"])
    observation = observe_checkpoint(plan, model, {}, root / f"generation/epoch{args.epoch:04d}",
        checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
        epoch=args.epoch, updates=args.updates)
    receipt = root / "observation-receipts" / f"epoch{args.epoch:04d}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise FileExistsError("Preserve prior saved-epoch observation receipt")
    write_json(receipt, observation)


def run(args):
    """Metadata-only orchestrator: tests -> probe -> baseline -> trainer -> saved."""
    from chemistry_campaign import read
    from latentmol.chemistry_runtime import epoch_fit_seconds
    from latentmol.cuda_process import require_cuda_free_coordinator
    from latentmol.experiments import file_sha256, write_json
    require_cuda_free_coordinator()
    campaign, plan = _metadata(args)
    root = Path(plan["campaign"])
    started = time.monotonic()
    deadline = started + plan["protocol"]["segment_wall_seconds"]
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command[:4] != ["python", "-m", "latentmol.chemistry_continuation", "train"]:
        raise ValueError("Worker received an unexpected trainer")
    if args.segment == 1:
        campaign.cuda_check()  # CUDA pytest child must exit before either probe.
        require_cuda_free_coordinator()
    _check_ownership(args, root, deadline)
    if args.segment == 1:
        _run_child(_phase_command(args, "baseline", deadline), root, args.segment, "baseline", deadline)
    handoff = read(root / "epoch40-handoff.json")
    if handoff.get("status") != "passed" or handoff.get("plan_sha256") != args.plan_sha256:
        raise ValueError("Missing exact epoch40 handoff proof")
    remaining = deadline - time.monotonic()
    conservative = read(root / "stability_gate.json")["conservative_epoch_seconds"]
    if remaining < 1800 + epoch_fit_seconds(plan, conservative):
        raise RuntimeError("Full epoch plus observation reserve does not fit allocation")
    _run_child(command, root, args.segment, "trainer", deadline - 1800)
    completion_path = root / f"segment-{args.segment:03d}-completion.json"
    completion = read(completion_path)
    records = [json.loads(line) for line in (Path(completion["run_dir"]) / "metrics.jsonl").read_text().splitlines() if line.strip()]
    observations, epochs = [], []
    for row in (item for item in records if "checkpoint_sha256" in item):
        _remaining(deadline)
        _run_child(_phase_command(args, "observe", deadline, checkpoint=row["checkpoint"],
            checkpoint_sha256=row["checkpoint_sha256"], epoch=row["epoch"], updates=row["updates"]),
            root, args.segment, f"observe-epoch{row['epoch']:04d}", deadline)
        receipt = read(root / "observation-receipts" / f"epoch{row['epoch']:04d}.json")
        # Child success cannot substitute for the persisted observation bytes.
        if file_sha256(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError("Saved child observation receipt changed")
        observations.append(receipt)
        epochs.append(row["epoch"])
    if not epochs or epochs[-1] != completion["completed_epoch"]:
        raise ValueError("Missing saved-epoch observations")
    require_cuda_free_coordinator()
    write_json(root / f"generation-segment-{args.segment:03d}.json", dict(status="passed_observation_integrity",
        plan_sha256=args.plan_sha256, completion_sha256=file_sha256(completion_path),
        observed_epochs=epochs, observations=observations, elapsed_seconds=time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--phase", choices=("coordinate", "baseline", "observe", "ownership-probe"), default="coordinate")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--probe-index", type=int, choices=(1,2))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.phase == "coordinate":
        if args.deadline is not None:
            parser.error("Coordinator creates its own shared deadline")
        run(args)
    elif args.phase == "baseline":
        if args.segment != 1:
            parser.error("Initial observation belongs only to segment1")
        _observe_baseline(args)
    elif args.phase == "observe":
        if any(value is None for value in (args.checkpoint,args.checkpoint_sha256,args.epoch,args.updates)):
            parser.error("Saved observation requires exact checkpoint/epoch/update identity")
        _observe_saved(args)
    else:
        if args.probe_index is None:
            parser.error("Ownership probe requires an explicit index")
        _ownership_probe(args)


if __name__ == "__main__":
    main()
