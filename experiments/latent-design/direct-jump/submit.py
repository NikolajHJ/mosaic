"""Submit one training/evaluation job per scorer and cohort, then their paired report."""

import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--job-prefix", default="dj-261119a")
    parser.add_argument("--recovery-queues", action="store_true",
                        help="Use V100 for both Protenix cohorts and A100/L40S for Boltz2")
    args = parser.parse_args()
    root = args.root.resolve()
    exp = root / "source/experiments/latent-design/direct-jump"
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=False)
    choices = (("protenix", 0, "gpuv100", 30), ("protenix", 1, "gpua10", 21),
               ("boltz2", 0, "gpua100", 38), ("boltz2", 1, "gpua100", 38))
    if args.recovery_queues:
        choices = (("protenix", 0, "gpuv100", 30), ("protenix", 1, "gpuv100", 30),
                   ("boltz2", 0, "gpua100", 38), ("boltz2", 1, "gpul40s", 38))

    def submit(label, resources, command):
        argv = ["bsub", "-J", args.job_prefix + "-" + label, "-oo", str(jobs / (label + "-%J.out")),
                "-eo", str(jobs / (label + "-%J.err"))] + resources + command
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        match = re.search(r"Job <(\d+)>", result.stdout)
        job = int(match.group(1)) if match else None
        receipt = dict(job_id=job, argv=argv, stdout=result.stdout, stderr=result.stderr, rc=result.returncode)
        with (jobs / (label + ".json")).open("x") as stream:
            json.dump(receipt, stream, indent=2)
        print(label, job, result.stdout.strip(), result.stderr.strip(), flush=True)
        if result.returncode or job is None:
            raise RuntimeError("Submission lacks an acceptance receipt")
        return job

    ids = []
    for model, cohort, queue, memory in choices:
        label = f"{model}-{cohort}"
        resources = ["-q", queue, "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=4096]",
                     "-gpu", f"num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem={memory}G",
                     "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                     "-W", "12:00" if model == "boltz2" else "04:00"]
        ids.append(submit(label, resources, ["bash", str(exp / "run.sh"), str(root), model, str(cohort)]))
    submit("report", ["-q", "hpc", "-n", "1", "-R", "rusage[mem=1024]", "-W", "00:15",
                      "-w", " && ".join(f"done({job})" for job in ids)],
           ["/work3/s234818/latentmol/env/boltz311/bin/python", str(exp / "report.py"),
            "--root", str(root), "--output", str(root / "comparison.json")])


if __name__ == "__main__":
    main()
