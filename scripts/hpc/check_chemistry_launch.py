"""Allocated, fail-closed dispatch check for the exact generated trainer argv.

Run the real module entrypoint and plan reader, authenticate its actual readiness
and stability evidence, then stop at entry to train_phase BEFORE model setup or
updates. This is a startup/receipt check, not a CUDA/native/training result. The
unaltered command is subsequently queued behind all existing GPU launch checks.
There is no test/skip flag in the production trainer.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


class DispatchReached(Exception):
    """Private harness stop; never raised by production training."""


def check_launch(command):
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise ValueError("Actual launch dispatch checks require an allocated job")
    if command[:4] != ["python", "-m", "latentmol.chemistry_continuation", "train"]:
        raise ValueError("Require the exact generated chemistry training command")
    result = None
    old_argv, old_trace = sys.argv, sys.gettrace()

    def boundary(frame, event, arg):
        nonlocal result
        if (event == "call" and frame.f_code.co_name == "train_phase"
                and frame.f_globals.get("__name__") == "__main__"
                and frame.f_code.co_filename.endswith("chemistry_continuation.py")):
            # No fixture replacement of read_plan or readiness: these are the
            # current release's real source/lineage/proof/health verifiers.
            plan = frame.f_locals["plan"]
            from latentmol.chemistry_valid_reference_training_policy import validate_valid_reference_continuation
            from latentmol.chemistry_short_continuation import continuation_end_epoch
            marker = validate_valid_reference_continuation(plan)
            if marker is None:
                raise ValueError("This allocated check is scoped to the epoch36->40 continuation")
            receipt = frame.f_globals["require_failure_readiness"](plan)
            if not receipt or receipt.get("status") != "passed":
                raise ValueError("The actual CLI did not authenticate passed readiness")
            result = dict(status="passed_dispatch_not_training", command=command,
                job_id=os.environ["LSB_JOBID"], utc=datetime.now(timezone.utc).isoformat(),
                plan_sha256=plan["_plan_sha256"], benchmark_gate_sha256=plan["_benchmark_gate_sha256"],
                failure_readiness_sha256=plan["_failure_readiness_sha256"],
                start_epoch=plan["starting_checkpoint"]["epoch"], end_epoch=continuation_end_epoch(plan),
                segment=frame.f_locals["segment"], resume=None if frame.f_locals["resume"] is None
                    else [str(x) for x in frame.f_locals["resume"]],
                model_setup_started=False, updates=0)
            raise DispatchReached()
        # Call events suffice; avoid tracing every import/verification line.
        return None

    try:
        sys.argv = [command[2], *command[3:]]
        sys.settrace(boundary)
        try:
            runpy.run_module(command[2], run_name="__main__")
        except DispatchReached:
            pass
        if result is None:
            raise RuntimeError("Actual module entrypoint did not reach authenticated training dispatch")
        return result
    finally:
        sys.settrace(old_trace)
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.output.exists():
        raise FileExistsError("Preserve the original launch-command check")
    result = check_launch(command)
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps(dict(status=result["status"], output=str(args.output),
                          sha256=hashlib.sha256(data).hexdigest())))


if __name__ == "__main__":
    main()
