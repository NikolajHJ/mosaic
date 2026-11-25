"""One CUDA-owning optimization child, ending before native CLI children start."""

import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback


def run(intent_path, intent_sha256):
    from latentmol.experiments import file_sha256
    intent_path = Path(intent_path)
    if file_sha256(intent_path) != intent_sha256:
        raise ValueError("Optimization child intent changed")
    intent = json.loads(intent_path.read_text())
    if (intent.get("version") != "embedding-gpu-child-v1"
            or os.getppid() != intent.get("parent_pid")
            or os.environ.get("LSB_JOBID") != intent.get("job_id")
            or os.environ.get("LATENTMOL_SOURCE_SHA256") != intent.get("source_sha256")
            or not intent.get("job_id") or not intent.get("source_sha256")):
        raise ValueError("Optimization child differs from its sealed allocated parent")
    output = Path(intent["output"])
    if not output.is_dir() or intent_path.resolve().parent != output:
        raise ValueError("Optimization child output differs from its exact handoff root")
    deadline = intent["deadline"]
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or deadline <= time.monotonic()):
        raise ValueError("Optimization child lacks a live shared deadline")
    plan_path = Path(intent["plan"]["path"])
    if file_sha256(plan_path) != intent["plan"]["sha256"]:
        raise ValueError("Optimization child plan changed")
    config = json.loads(plan_path.read_text())
    config["_plan_sha256"] = intent["plan"]["sha256"]
    config["_plan_path"] = str(plan_path)
    from .embedding_comparison_experiment import VERSION, gpu_optimization_stage, write
    if config.get("version") != VERSION:
        raise ValueError("Optimization child has an unknown comparison protocol")
    import torch
    torch.set_num_threads(min(4, int(os.environ.get("LSB_DJOB_NUMPROC", "4"))))
    try:
        result = gpu_optimization_stage(config, output, deadline)
        write(output / "optimization-phase.json", {**result, "intent_sha256": intent_sha256,
            "child_pid": os.getpid(), "context_release": "Process exits before native follow-up starts."})
    except BaseException as error:
        write(output / "optimization-phase.json", dict(status="failed_not_passed",
            intent_sha256=intent_sha256, error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--intent-sha256", required=True)
    args = parser.parse_args()
    run(args.intent, args.intent_sha256)


if __name__ == "__main__":
    main()
