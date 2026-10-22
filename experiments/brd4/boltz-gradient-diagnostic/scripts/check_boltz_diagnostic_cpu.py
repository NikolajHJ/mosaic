"""Authenticate persisted CPU certificates without depending on live LSF retention.

This stdlib byte/schema verifier cannot certify a molecular run or change a
historical outcome. Producer assessment, source bindings, raw XML and final
scheduler/LSF captures must all match the caller's authenticated external pins.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    require(path.is_file(), "Required producer proof missing: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(
    root,
    *,
    assessment_sha256,
    job_id,
    tools_manifest_sha256,
    scientific_manifest_sha256,
    tests,
    scheduler_sha256,
    stdout_sha256,
):
    """Return authenticated persisted evidence; never query or infer scheduler state."""
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), "CPU root is not a directory")
    require(
        isinstance(job_id, str) and job_id.isdecimal(), "Invalid expected producer job"
    )
    require(type(tests) is int and tests > 0, "Invalid expected CPU test count")
    for pin in (
        assessment_sha256,
        tools_manifest_sha256,
        scientific_manifest_sha256,
        scheduler_sha256,
        stdout_sha256,
    ):
        require(
            isinstance(pin, str) and re.fullmatch("[0-9a-f]{64}", pin),
            "Invalid external SHA-256 pin",
        )
    assessment_path = root / "assessment.json"
    require(sha(assessment_path) == assessment_sha256, "CPU assessment hash differs")
    record = json.loads(assessment_path.read_text(encoding="utf-8"))
    require(
        record["status"] == "passed"
        and record["job_id"] == job_id
        and type(record["tests"]) is int
        and record["tests"] == tests,
        "CPU assessment status/job/test count differs",
    )
    require(
        all(
            type(record[k]) is int and record[k] == 0
            for k in ("failures", "errors", "skips")
        ),
        "CPU assessment contains failures/errors/skips",
    )
    require(
        record["tools_manifest_sha256"] == tools_manifest_sha256
        and record["scientific_manifest_sha256"] == scientific_manifest_sha256,
        "CPU producer source bindings differ",
    )
    require(
        isinstance(record["proofs"], list) and record["proofs"],
        "CPU proof list is empty",
    )
    proofs, seen = [], set()
    for proof in record["proofs"]:
        supplied = Path(proof["path"])
        path = (supplied if supplied.is_absolute() else root / supplied).resolve(
            strict=True
        )
        require(
            path.is_relative_to(root) and path not in seen,
            "Unsafe or duplicate CPU proof path",
        )
        seen.add(path)
        require(sha(path) == proof["sha256"], "CPU declared proof hash differs")
        proofs.append(
            {"path": str(path), "sha256": proof["sha256"], "bytes": path.stat().st_size}
        )
    xml_path = (root / "tests.xml").resolve(strict=True)
    require(xml_path in seen, "Raw tests.xml is not an authenticated declared proof")
    xml = ET.parse(xml_path).getroot()
    cases = list(xml.iter("testcase"))
    require(len(cases) == tests, "Raw XML testcase count differs")
    require(
        not any(
            list(case.iter(tag))
            for case in cases
            for tag in ("failure", "error", "skipped")
        ),
        "Raw XML contains failed/error/skipped tests",
    )
    suites = list(xml.iter("testsuite"))
    require(bool(suites), "Raw XML has no test suite")
    containers = suites + list(xml.iter("testsuites"))
    require(
        all(
            int(suite.get(k, "0")) == 0
            for suite in containers
            for k in ("failures", "errors", "skipped")
        ),
        "Raw XML suite records failed/error/skipped tests",
    )
    require(
        all(
            "tests" not in suite.attrib
            or int(suite.attrib["tests"]) == len(list(suite.iter("testcase")))
            for suite in containers
        ),
        "Raw XML declared count differs from testcase records",
    )
    final = root.parent / "scheduler-final.txt"
    stdout = root.parent / ("stdout." + job_id + ".log")
    require(
        sha(final) == scheduler_sha256 and sha(stdout) == stdout_sha256,
        "Persisted scheduler/LSF hash differs",
    )
    compact = " ".join(final.read_text(encoding="utf-8").split())
    header = re.match(r"^Job\s*<([^>]+)>,", compact)
    statuses = [
        "".join(value.split()) for value in re.findall(r"\bStatus\s*<([^>]+)>", compact)
    ]
    queues = [
        "".join(value.split())
        for value in re.findall(r"\bQue\s*ue\s*<([^>]+)>", compact)
    ]
    require(
        header is not None
        and "".join(header[1].split()) == job_id
        and statuses == ["DONE"]
        and queues == ["hpc"]
        and "Done successfully." in compact,
        "Persisted scheduler did not certify this CPU job DONE/hpc",
    )
    log = stdout.read_text(encoding="utf-8")
    subjects = re.findall(r"^Subject:\s*Job\s+(\d+):([^\r\n]*)", log, re.MULTILINE)
    require(
        len(subjects) == 1
        and subjects[0][0] == job_id
        and subjects[0][1].rstrip().endswith(" Done")
        and re.search(r"^Successfully completed\.\s*$", log, re.MULTILINE)
        and not re.search(
            r"Exited\s+(?:with exit (?:code|status)|by signal)", log, re.IGNORECASE
        ),
        "Persisted LSF output did not certify successful completion",
    )
    return {
        "version": "mosaic-boltz-diagnostic-cpu-proof-v1",
        "status": "passed",
        "scope": "persisted producer byte/XML/scheduler proof only; no native readiness approval",
        "root": str(root),
        "job_id": job_id,
        "tests": tests,
        "failures": 0,
        "errors": 0,
        "skips": 0,
        "assessment_sha256": assessment_sha256,
        "tools_manifest_sha256": tools_manifest_sha256,
        "scientific_manifest_sha256": scientific_manifest_sha256,
        "scheduler": {
            "path": str(final),
            "sha256": scheduler_sha256,
            "bytes": final.stat().st_size,
        },
        "stdout": {
            "path": str(stdout),
            "sha256": stdout_sha256,
            "bytes": stdout.stat().st_size,
        },
        "proofs": proofs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    for name in (
        "assessment-sha256",
        "job",
        "tools-manifest",
        "scientific-manifest",
        "scheduler-sha256",
        "stdout-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tests", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.root,
        assessment_sha256=args.assessment_sha256,
        job_id=args.job,
        tools_manifest_sha256=args.tools_manifest,
        scientific_manifest_sha256=args.scientific_manifest,
        tests=args.tests,
        scheduler_sha256=args.scheduler_sha256,
        stdout_sha256=args.stdout_sha256,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
