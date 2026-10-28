"""Portable metadata preflight; never a native/model or HPC admission gate.

Run with a Python that has pytest, or pass --python. Only the explicit portable
runtime/storage modules and pure file-count cases are selected. Integration
fixtures, all scientific libraries and real GPU/LSF operations are excluded.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

OWNER = Path(__file__).resolve().parents[1]
REPO = OWNER.parents[2]
SELECTION = (
    "tests/test_runtime_reservations.py",
    "tests/test_segment_capacity.py",
    "tests/test_rolling_capacity.py::test_file_capacity_funds_all_three_actual_retained_file_streams",
    "tests/test_rolling_capacity.py::test_large_verified_validation_cohort_expands_fixed_boundary_allowance",
    "tests/test_rolling_capacity.py::test_file_capacity_requires_finite_complete_new_validation_count",
    "tests/test_rolling_capacity.py::test_completed_validation_identity_survives_different_host_device_inode_and_blocks",
    "tests/test_rolling_capacity.py::test_completed_validation_identity_rejects_same_length_content_or_path_change",
    "tests/test_rolling_capacity.py::test_windows_handle_ctime_difference_does_not_weaken_posix_race_checks",
)
EXPECTED_CASES = 98
FORBIDDEN = {"numpy", "jax", "torch", "equinox", "mosaic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("LSB_JOBID") or os.environ.get("LSB_QUEUE"):
        raise ValueError("This local-only preflight never fabricates or adopts an HPC allocation")
    output = args.output.resolve()
    if not output.is_relative_to(REPO / "results") or output.exists() or any(p.is_symlink() for p in output.parents):
        raise ValueError("Use a fresh unlinked output directory within this repository's results/")
    tests = sorted({row.split("::")[0] for row in SELECTION})
    dependencies = tests + ["scripts/precision_continuation_driver.py", "scripts/segment_capacity.py",
                            "scripts/rolling_capacity.py", "scripts/local_metadata_preflight.py"]
    def proofs():
        rows = []
        for name in dependencies:
            raw = (OWNER / name).read_bytes()
            rows.append({"path": str(OWNER / name), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        return rows
    before = proofs()
    for name in tests:
        for node in ast.walk(ast.parse((OWNER / name).read_bytes(), filename=name)):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            if names & FORBIDDEN:
                raise ValueError("Scientific imports do not belong in the portable preflight")
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "tmp"
    assert not temporary.exists()  # pytest may remove basetemp; only our new owned path is permitted.
    command = [args.python, "-I", "-B", "-m", "pytest", "-p", "no:cacheprovider", "--noconftest", "--import-mode=importlib",
               "--rootdir", str(OWNER), "-o", "addopts=", "-o", "pythonpath=",
               *[str(OWNER / row.split("::")[0]) + ("::" + row.split("::", 1)[1] if "::" in row else "")
                 for row in SELECTION], "-q", "--tb=short", "--basetemp", str(temporary),
               "--junitxml=" + str(output / "tests.xml")]
    environment = dict(os.environ)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    started = time.monotonic()
    with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
        result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, timeout=120)
    cases = list(ET.parse(output / "tests.xml").iter("testcase")) if (output / "tests.xml").exists() else []
    after = proofs()
    unchanged = before == after
    passed = unchanged and result.returncode == 0 and len(cases) == EXPECTED_CASES and all(
        not list(case.iter(tag)) for case in cases for tag in ("failure", "error", "skipped"))
    assessment = {"status": "passed" if passed else "failed", "cases": len(cases),
                  "expected_cases": EXPECTED_CASES, "elapsed_seconds": time.monotonic() - started,
                  "argv": command, "returncode": result.returncode, "source_proofs": before,
                  "source_proofs_after": after, "source_unchanged_during_tests": unchanged,
                  "scope": "portable resource/runtime metadata only", "scientific_evaluation": False,
                  "allocation_fabricated": False, "HPC_full_suite_still_required": True,
                  "native_or_gradient_readiness_established": False,
                  "external_temp_permissions_or_files_changed": False}
    (output / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf8", newline="\n")
    print((output / "stdout.log").read_text())
    print(json.dumps({key: assessment[key] for key in ("status", "cases", "elapsed_seconds", "scientific_evaluation")}, indent=2))
    if not passed:
        raise SystemExit("Portable metadata preflight failed; raw evidence retained")


if __name__ == "__main__":
    main()
