"""Sealed allocated CPU validation; never changes a training/optimization job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Validation requires an allocated LSF CPU job")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = root / "validation-manifest.json"
    if sha(manifest) != args.manifest_sha256:
        raise ValueError("Validation manifest identity differs")
    records = json.loads(manifest.read_text())
    for relative, expected in records["files"].items():
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Sealed source differs: " + relative)
    if sha(root / "src/latentmol/guidance/environment_policy.py") != sha(
            root / "mosaic/src/mosaic/small_molecule/chemistry/environment_policy.py"):
        raise ValueError("Portable policy copies diverged")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    # DTU's module sets a Python3.11 PYTHONHOME. Mosaic uses its own3.12
    # interpreter: each child must discover its own standard library/venv.
    environment.pop("PYTHONHOME", None)
    torch_python = "/work3/s234818/latentmol/env/boltz311/bin/python"
    jax_python = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
    stages = []
    started = time.monotonic()
    deadline = started + 3300

    def run(name, command, cwd, extra=None, junit=None):
        env = {**environment, **(extra or {})}
        t0 = time.monotonic()
        remaining = int(deadline - t0)
        if remaining < 30:
            raise RuntimeError("Insufficient internal CPU validation budget")
        with (args.output / (name + ".log")).open("x") as stream:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=min(2100, remaining))
        record = dict(name=name, command=command, returncode=result.returncode,
                      elapsed_seconds=time.monotonic()-t0)
        if junit and junit.is_file():
            suites = list(ET.parse(junit).getroot().iter("testsuite"))
            record["junit"] = {key: sum(int(s.get(key, "0")) for s in suites)
                                for key in ("tests", "failures", "errors", "skipped")}
        stages.append(record)
        write(args.output / (name + "-result.json"), record)
        if (result.returncode or (junit and (not junit.is_file() or
                any(record["junit"][k] for k in ("failures", "errors", "skipped"))))):
            raise RuntimeError("Allocated validation stage failed: " + name)

    status, error = "failed", None
    try:
        junit = args.output / "torch-tests.xml"
        run("torch-tests", [torch_python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "tests/guidance/test_chemical_environment.py", "tests/guidance/test_chemical_environment_reference.py",
            "tests/test_atom_preserving_chemistry.py", "tests/test_repair_failure_guidance.py",
            "--junitxml", str(junit)], root,
            dict(PYTHONPATH=str(root / "src") + ":" + str(root / "tests")), junit)
        run("torch-probe", [torch_python, "scripts/hpc/neighbor_environment_probe.py", "torch",
            "--output", str(args.output), "--data", records["dataset"],
            "--checkpoint", records["checkpoint"]], root,
            dict(PYTHONPATH=str(root / "src") + ":" + str(root / "tests")))
        junit = args.output / "jax-tests.xml"
        run("jax-tests", [jax_python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "tests/small_molecule/test_chemistry_environment.py", "--junitxml", str(junit)],
            root / "mosaic", dict(PYTHONPATH=str(root / "mosaic/src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), junit)
        run("jax-probe", [jax_python, str(root / "scripts/hpc/neighbor_environment_probe.py"),
            "jax", "--output", str(args.output)], root / "mosaic",
            dict(PYTHONPATH=str(root / "mosaic/src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        torch_rows = json.loads((args.output / "torch-parity.json").read_text())
        jax_rows = json.loads((args.output / "jax-parity.json").read_text())
        def flat(values):
            for value in values:
                if isinstance(value, list):
                    yield from flat(value)
                else:
                    yield value
        comparisons = []
        for left, right in zip(torch_rows, jax_rows):
            lg, rg = list(flat(left["gradient"])), list(flat(right["gradient"]))
            if len(lg) != len(rg):
                raise ValueError("Gradient shape differs")
            record = dict(loss_error=abs(left["loss"]-right["loss"]),
                          max_gradient_error=max(abs(a-b) for a, b in zip(lg, rg)),
                          assessments_equal=left["assessments"] == right["assessments"])
            comparisons.append(record)
            if record["loss_error"] > 1e-9 or record["max_gradient_error"] > 1e-9 or not record["assessments_equal"]:
                raise ValueError("Cross-framework policy/scalar/VJP parity failed")
        if len(torch_rows) != 6 or len(jax_rows) != 6:
            raise ValueError("Missing cross-framework cases")
        write(args.output / "cross-framework-parity.json", dict(status="passed", cases=comparisons,
              tolerance="absolute1e-9 scalar/raw symmetric-bond VJP FP64"))
        status = "passed_validation_not_training_adoption"
    except BaseException as exception:
        error = type(exception).__name__ + ": " + str(exception)
        raise
    finally:
        write(args.output / "result.json", dict(status=status, error=error,
            job_id=os.environ["LSB_JOBID"], stages=stages, elapsed_seconds=time.monotonic()-started,
            manifest_sha256=args.manifest_sha256,
            scope="CPU engineering/native chemistry/held-out corpus/frozen VAE VJP/parity; no training or binding experiment",
            scientific_defaults_unchanged=True))


if __name__ == "__main__":
    main()
