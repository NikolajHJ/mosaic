"""Authenticate native readiness and conservatively budget compact continuation.

Run on allocated DTU CPU after the pilot; this authorizes no failed result.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Pilot assessment requires allocated DTU HPC")
    summary = args.pilot / "summary.json"
    pilot = json.loads(summary.read_text())
    config = json.loads(args.config.read_text())
    if config["checks"].get("artifact_policy") != "compact":
        raise ValueError("This storage projection supports compact evidence only")
    if (
        pilot["status"] != "passed"
        or len(pilot["cases"]) != 3
        or any(c["status"] != "passed" or len(c["states"]) != 4 for c in pilot["cases"])
    ):
        raise ValueError("The complete three-case native pilot must pass")
    for field in ("model_digest", "context_digest"):
        if pilot[field + "_before"] != pilot[field + "_after"]:
            raise ValueError("Frozen pilot identity changed")
    example = (
        Path(__file__).resolve().parents[1] / "examples/protenix_relaxed_gradients.py"
    )
    spec = importlib.util.spec_from_file_location("pilot_policy", example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._validate_prepared_config(
        config, pilot["config"], allow_update_count_change=True
    )
    pilot_bytes = sum(p.stat().st_size for p in args.pilot.rglob("*") if p.is_file())
    run_bytes = pilot_bytes
    step_seconds = []
    for index in range(3):
        directory = args.pilot / f"case-{index}"
        trials = list(directory.glob("step-*-trial-*.npz")) + list(
            directory.glob("primal-step-*-trial-*.npz")
        )
        snapshots = list((directory / "snapshots").glob("*.npz"))
        # Bound every requested update by ten trials of each evaluation path,
        # one state, and generous per-update JSON accounting.
        largest_trial = max(p.stat().st_size for p in trials)
        largest_snapshot = max(p.stat().st_size for p in snapshots)
        run_bytes += config["checks"]["steps"] * (
            20 * largest_trial + largest_snapshot + 8192
        )
        step_seconds.append(
            (
                (directory / "snapshots/step-0003.npz").stat().st_mtime
                - (directory / "snapshots/step-0000.npz").stat().st_mtime
            )
            / 3
        )
    report_bytes = 128 * 1024**2
    archive_bytes = int(1.01 * (run_bytes + pilot_bytes + report_bytes)) + 8 * 1024**2
    required_gib = 4 + (run_bytes + report_bytes + archive_bytes) / 1024**3
    args.output.mkdir(parents=True, exist_ok=False)
    quota_result = subprocess.run(
        [
            "/work3/s234818/mosaic/envs/protenix-tiny/bin/python",
            str(Path(__file__).parent / "hpc/stage0_quota_preflight.py"),
            "--run-dir",
            str(args.output),
            "--storage-pool",
            "6",
            "--minimum-free-gib",
            str(required_gib),
        ],
        check=False,
    )
    receipt = {
        "status": "ready" if quota_result.returncode == 0 else "stopped_quota_gate",
        "job_id": os.environ["LSB_JOBID"],
        "pilot_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "pilot_bytes": pilot_bytes,
        "projected_new_run_bytes_upper_bound": run_bytes,
        "projected_report_bytes": report_bytes,
        "projected_archive_bytes_upper_bound": archive_bytes,
        "required_free_gib": required_gib,
        "observed_mean_step_seconds_by_case": step_seconds,
        "projected_3000_update_seconds": sum(step_seconds) * config["checks"]["steps"],
        "scope": "metadata-based conservative storage/runtime projection; not completed 1000-update evidence",
    }
    (args.output / "assessment.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    if quota_result.returncode:
        raise SystemExit(quota_result.returncode)


if __name__ == "__main__":
    main()
