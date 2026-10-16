"""Bind one uniform control to the passed BRD4 shape budget on allocated CPU.

This is a code/resource readiness gate. The new uniform graph must still
pass all native derivative/null/parity checks at the start of its GPU job.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess


def _authenticated(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Reference identity differs: " + str(path))
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-assessment", type=Path, required=True)
    parser.add_argument("--reference-assessment-sha256", required=True)
    parser.add_argument("--reference-pilot", type=Path, required=True)
    parser.add_argument("--reference-pilot-sha256", required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference-manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outstanding-study-count", type=int, default=1)
    parser.add_argument("--extra-artifact-reserve-gib", type=float, default=0)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Uniform resource assessment requires allocated DTU HPC")
    if args.outstanding_study_count < 0:
        raise ValueError("Outstanding study count cannot be negative")
    if (
        not math.isfinite(args.extra_artifact_reserve_gib)
        or args.extra_artifact_reserve_gib < 0
    ):
        raise ValueError("Extra artifact reserve must be finite and nonnegative")
    root = Path(__file__).resolve().parents[1]
    reference = _authenticated(
        args.reference_assessment, args.reference_assessment_sha256
    )
    pilot = _authenticated(args.reference_pilot, args.reference_pilot_sha256)
    old_manifest = _authenticated(
        args.reference_source / "source_manifest.json", args.reference_manifest_sha256
    )
    current_manifest = json.loads((root / "source_manifest.json").read_text())
    if {k: v for k, v in old_manifest["files"].items() if k.startswith("src/")} != {
        k: v for k, v in current_manifest["files"].items() if k.startswith("src/")
    }:
        raise ValueError(
            "The native graph/model library changed from the measured pilot"
        )
    if (
        reference["status"] != "ready"
        or reference["storage"] != "zhome"
        or reference["pilot_summary_sha256"] != args.reference_pilot_sha256
        or pilot["status"] != "passed"
        or len(pilot["cases"]) != 3
        or any(c["status"] != "passed" or len(c["states"]) != 4 for c in pilot["cases"])
    ):
        raise ValueError("Requires the passed three-case BRD4 pilot and resource gate")
    for field in ("model_digest", "context_digest"):
        if pilot[field + "_before"] != pilot[field + "_after"]:
            raise ValueError("Reference frozen identity changed")
    reference_config_path = args.reference_source / "examples/configs/brd4_5000.json"
    previous = _authenticated(reference_config_path, reference["config_sha256"])
    config = json.loads(args.config.read_text())
    spec = importlib.util.spec_from_file_location(
        "uniform_policy", root / "examples/protenix_relaxed_gradients.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._validate_prepared_config(config, previous, allow_initialization_change=True)
    if len(module._case_specs(config)) != 1 or config["initialization"]["kind"] not in (
        "uniform",
        "noisy_uniform",
        "canonical_diffuse",
    ):
        raise ValueError("One explicit initialization control is required")
    fixture = root / config["fixture"]
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != pilot["fixture_sha256"]:
        raise ValueError("Fixture/graph shape differs from the measured pilot")
    # Retain the full previous THREE-case run/report/archive bound for this ONE
    # control, plus the same bound for EACH still-outstanding study. This
    # deliberately overbudgets singleton controls using the full three-case run.
    reserved_studies = 1 + args.outstanding_study_count
    required = (
        4
        + reserved_studies * (reference["required_free_gib"] - 4)
        + args.extra_artifact_reserve_gib
    )
    args.output.mkdir(parents=True, exist_ok=False)
    quota = subprocess.run(
        [
            "/work3/s234818/mosaic/envs/protenix-tiny/bin/python",
            str(root / "scripts/hpc/zhome_quota_preflight.py"),
            "--run-dir",
            str(args.output),
            "--minimum-free-gib",
            str(required),
        ],
        check=False,
    )
    receipt = {
        "status": "ready" if quota.returncode == 0 else "stopped_quota_gate",
        "job_id": os.environ["LSB_JOBID"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "pilot_summary_sha256": args.reference_pilot_sha256,
        "reference_assessment_sha256": args.reference_assessment_sha256,
        "reference_source_manifest_sha256": args.reference_manifest_sha256,
        "required_free_gib": required,
        "new_control_cases": 1,
        "reserved_study_count": reserved_studies,
        "extra_artifact_reserve_gib": args.extra_artifact_reserve_gib,
        "initialization": config["initialization"],
        "scope": "unchanged native library/shape/settings; initialization-only control; full prior three-case artifact bound reserved for each outstanding study and new control; native derivative/null/parity/descent gates pending at GPU entry",
    }
    (args.output / "assessment.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    if quota.returncode:
        raise SystemExit(quota.returncode)


if __name__ == "__main__":
    main()
