"""Metadata-only orchestration for one immutable Protenix endpoint investigation.

The only dispatched native work is a bounded forensic evaluation.  This module
contains no optimizer or continuation branch and never certifies a saved study.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


VERSION = "protenix-endpoint-forensics-source-v1"
PLAN_VERSION = "protenix-endpoint-forensics-native-v1"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
SOURCE = BASE / "protenix-endpoint-forensics-release-01/source"
ROOT = BASE / "protenix-endpoint-forensics-01"
PARENT_SOURCE = BASE / "nonchem-endpoint-release-02/source"
PARENT_PIN = "88372f92b6349e6c34c542fb84fb39482c4c340568431d1499ebc166a1e1e7fa"
PARENT_FILES = 14
ORIGINAL_PLAN = {"path": str(BASE / "nonchem-endpoint-release-01/source/plan.json"),
                 "sha256": "cf7292ff571e202d1dadd12686a341e45d45865816c95d204dd752f8da89de39"}
FAILED_ADMISSION = {"path": str(BASE / "nonchem-endpoint-continuation-01/protenix-validate-20000/payload/study/native_gradient_checks-prospective/admission.json"),
                    "bytes": 2531142, "sha256": "3b4e6c14aa21aed083594ab040a7faf4a50f93b4adce5a4fa679108bc531aaf1"}
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
PHASES = ("cpu", "after-cpu", "gpu")
MODULES = {"tests.test_protenix_endpoint_forensics", "tests.test_protenix_forensics_native", "tests.test_protenix_forensics_ops"}
GIB = 1024**3
OUTPUT_CAP = 256 * 1024**2
METADATA_CAP = 16 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    write(temporary, value)
    temporary.replace(path)


def binding(path):
    path = Path(path)
    info = path.lstat()
    require(path.is_absolute() and not path.is_symlink() and stat.S_ISREG(info.st_mode)
            and ".." not in path.parts and all(not parent.is_symlink() for parent in path.parents),
            "Proof is not an absolute regular file with ordinary parents")
    return {"path": str(path), "bytes": info.st_size, "sha256": sha(path)}


def check_binding(row):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}
            and isinstance(row["path"], str) and type(row["bytes"]) is int and row["bytes"] > 0
            and re.fullmatch("[a-f0-9]{64}", row["sha256"] or ""), "Malformed exact file binding")
    require(binding(row["path"]) == row, "Proof bytes changed: " + row["path"])
    return row


def verify(source, pin):
    source = Path(source)
    require(source.is_dir() and not source.is_symlink() and all(not path.is_symlink() for path in source.parents),
            "Source root or parent missing/linked")
    require(re.fullmatch("[a-f0-9]{64}", pin or "") and sha(source / "source_manifest.json") == pin,
            "Source manifest pin changed")
    manifest = read(source / "source_manifest.json")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Source file inventory missing")
    for name, digest in files.items():
        require(isinstance(name, str) and re.fullmatch("[a-f0-9]{64}", digest or ""), "Malformed source member")
        part = PurePosixPath(name)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name
                and name not in ("", ".", "source_manifest.json"), "Unsafe source member")
        path = source.joinpath(*part.parts)
        require(path.is_file() and not path.is_symlink() and sha(path) == digest
                and all(not parent.is_symlink() for parent in path.parents), "Source member changed/linked: " + name)
    actual = set()
    for path in source.rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "Source contains linked/special entry")
        if path.is_file():
            actual.add(path.relative_to(source).as_posix())
    require(actual == set(files) | {"source_manifest.json"}, "Source scope changed")
    return manifest


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Authenticated metadata module missing")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def parent_ops():
    manifest = verify(PARENT_SOURCE, PARENT_PIN)
    require(len(manifest["files"]) == PARENT_FILES, "Parent source scope differs")
    return module(PARENT_SOURCE / "scripts/continuation_ops.py", "_protenix_forensics_authenticated_parent")


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("version") == PLAN_VERSION
            and plan.get("source_root") == str(SOURCE) and plan.get("control_root") == str(ROOT),
            "Forensic plan identity/root differs")
    require(plan.get("original_results_changed") is False and plan.get("optimization_settings_changed") is False
            and plan.get("can_certify_or_restart_original_study") is False and plan.get("monitor_remains_paused") is True,
            "Diagnostic-only preservation scope differs")
    for name, value in (("max_bytes", OUTPUT_CAP), ("cap_bytes", OUTPUT_CAP), ("metadata_cap_bytes", METADATA_CAP),
                        ("internal_seconds", 5400), ("maximum_unique_graphs", 109)):
        require(type(plan.get(name)) is int and plan[name] == value, "Forensic bound differs: " + name)
    require(plan.get("output") == str(ROOT / "gpu/payload"), "Forensic output scope differs")
    original = plan.get("original_plan")
    require(isinstance(original, dict) and original.get("path") == ORIGINAL_PLAN["path"]
            and original.get("sha256") == ORIGINAL_PLAN["sha256"], "Original source01 plan binding differs")
    check_binding(original)
    prior = read(original["path"])
    candidates = [row for row in prior.get("arms", []) if row.get("id") == "protenix"]
    require(len(candidates) == 1 and plan.get("arm") == candidates[0], "Exact original Protenix arm/settings differ")
    require(plan.get("failed_admission") == FAILED_ADMISSION, "Original failed admission binding differs")
    check_binding(plan["failed_admission"])
    failed = read(plan["failed_admission"]["path"])
    require(failed.get("status") == "failed" and failed.get("can_admit_new_child") is False
            and failed.get("original_run_failed_preserved") is True and failed.get("can_certify_original_study") is False,
            "Original failed admission was relabeled")
    return plan


def external(manifest, plan):
    require(manifest.get("version") == VERSION, "Wrong forensic source schema")
    parents = manifest.get("parent_sources")
    expected = {"path": str(PARENT_SOURCE), "manifest_sha256": PARENT_PIN, "files": PARENT_FILES}
    require(isinstance(parents, list) and expected in parents, "Exact worker-fix02 parent source missing")
    require(len({row.get("path") for row in parents}) == len(parents), "Duplicate parent source scope")
    for row in parents:
        require(set(row) == {"path", "manifest_sha256", "files"} and type(row["files"]) is int and row["files"] > 0,
                "Malformed parent source binding")
        require(len(verify(row["path"], row["manifest_sha256"])["files"]) == row["files"], "Parent source member count changed")
    proofs = manifest.get("preserved_proofs")
    require(isinstance(proofs, list) and proofs and FAILED_ADMISSION in proofs, "Preserved failed-admission proofs missing")
    for row in proofs:
        check_binding(row)
    validate_plan(plan)
    parent = parent_ops()
    parent_manifest = parent.verify(PARENT_SOURCE, PARENT_PIN)
    adoption = parent.external(parent_manifest, parent.read(PARENT_SOURCE / "plan.json"))
    return {"status": "authenticated_read_only_originals", "parent_adoption": adoption,
            "original_results_changed": False, "can_certify_or_restart_original_study": False}


def resources(phase):
    require(phase in PHASES, "No optimizer/continuation phase exists")
    if phase == "gpu":
        return ["-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]",
                "-gpu", "num=1:mode=exclusive_process", "-W", "02:00"]
    return ["-q", "hpc", "-n", "4" if phase == "cpu" else "1", "-R", "span[hosts=1]",
            "-R", "rusage[mem=4GB]" if phase == "cpu" else "rusage[mem=2GB]",
            "-W", "01:00" if phase == "cpu" else "00:20"]


def argv(phase, pin, dependency=None):
    require(re.fullmatch("[a-f0-9]{64}", pin or ""), "Malformed source pin")
    if phase == "after-cpu":
        require(isinstance(dependency, str) and re.fullmatch(r"ended\([1-9]\d*\)", dependency), "Callback needs exact own-ended dependency")
    else:
        require(dependency is None, "Diagnostic/CPU job must be independent")
    parent = ROOT / phase
    command = ["bsub", *resources(phase), "-J", "mosaic-pro-forensics-" + phase,
               "-oo", str(parent / "stdout.%J.log"), "-eo", str(parent / "stderr.%J.log")]
    if dependency is not None:
        command += ["-w", dependency]
    return command + ["bash", "-l", str(SOURCE / "scripts/protenix_forensics_stage.lsf"), str(SOURCE), pin, phase]


def submit(phase, pin, dependency=None):
    command = argv(phase, pin, dependency)
    parent = ROOT / phase
    parent.mkdir(exist_ok=False)
    record = {"status": "unknown", "phase": phase, "argv": command, "dependency": dependency,
              "source_manifest_sha256": pin, "submitted_utc": datetime.now(timezone.utc).isoformat()}
    write(parent / "submission-started.json", record)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        matches = re.findall(r"Job <([1-9]\d*)> is submitted", result.stdout)
        if result.returncode == 0 and len(matches) == 1:
            record.update(status="submitted", job_id=matches[0])
    except BaseException as error:
        record["error"] = repr(error)
    finally:
        write(parent / "submission.json", record)
    require(record["status"] == "submitted", "Unknown submission outcome: inspect exact markers, never duplicate")
    import fcntl
    with (ROOT / "jobs.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = read(ROOT / "jobs.json") if (ROOT / "jobs.json").exists() else {}
        require(phase not in jobs, "Actual job identity cannot be replaced")
        jobs[phase] = record["job_id"]
        atomic(ROOT / "jobs.json", jobs)
    return record


def own(phase, pin):
    require(phase in PHASES, "Unknown forensic phase")
    path = ROOT / phase / "submission.json"
    for number in range(21):
        if path.exists():
            break
        require(number < 20, "Own actual submission not committed")
        time.sleep(1)
    sub = read(path)
    require(sub.get("status") == "submitted" and sub.get("job_id") == os.environ.get("LSB_JOBID")
            and sub.get("phase") == phase and sub.get("source_manifest_sha256") == pin
            and sub.get("argv") == argv(phase, pin, sub.get("dependency")), "Own startup/job/source/resource vector differs")
    require(os.environ.get("LSB_QUEUE") == ("gpul40s" if phase == "gpu" else "hpc")
            and os.environ.get("LSB_DJOB_NUMPROC") == ("1" if phase == "after-cpu" else "4"), "Actual queue/slots differ")
    if phase == "after-cpu":
        producer = read(ROOT / "cpu/submission.json")
        require(producer.get("status") == "submitted" and sub["dependency"] == "ended(" + producer["job_id"] + ")",
                "Callback is not bound to its own CPU producer")
    return sub


def lsf_success(text, job, name, queue):
    require(isinstance(text, str) and re.fullmatch(r"[1-9]\d*", job or ""), "Malformed raw LSF/job")
    raw = "".join(text.split())
    subjects = re.findall(r"Subject:Job([1-9]\d*):<([^>]+)>incluster<([^>]+)>Done", raw)
    require(raw.count("Subject:Job") == 1 and subjects == [(job, name, "dcc")]
            and re.findall(r"inqueue<([^>]+)>", raw) == [queue]
            and raw.count("Successfullycompleted.") == 1 and "Exitedwithexit" not in raw,
            "Actual unique same-job/name/queue LSF success missing")
    return True


def terminal_cpu():
    parent = ROOT / "cpu"
    sub = read(parent / "submission.json")
    job = sub["job_id"]
    require(sub.get("status") == "submitted" and sub.get("argv") == argv("cpu", sub["source_manifest_sha256"]),
            "CPU producer submission differs")
    tools = parent_ops()
    proofs = [binding(parent / "submission.json")]
    for number in range(13):
        run = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, timeout=30)
        row = {"job_id": job, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
        path = parent / ("terminal-query-" + str(number).zfill(3) + ".json")
        write(path, row)
        proofs.append(binding(path))
        if tools.scheduler_status(row, job, "hpc") == "DONE":
            break
        require(number < 12, "Bounded real CPU terminal wait expired")
        time.sleep(5)
    path = parent / ("stdout." + job + ".log")
    for number in range(21):
        if path.exists() and "Successfullycompleted." in "".join(path.read_text().split()):
            break
        require(number < 20, "Actual CPU terminal LSF unavailable")
        time.sleep(1)
    lsf_success(path.read_text(), job, "mosaic-pro-forensics-cpu", "hpc")
    proofs.append(binding(path))
    return proofs


def xml_cases(path, expected):
    require(isinstance(expected, dict) and set(expected) == MODULES
            and all(type(value) is int and value > 0 for value in expected.values()), "CPU module counts missing/differ")
    cases = list(ET.parse(path).iter("testcase"))
    require(cases and all(not list(row) and row.get("classname") and row.get("name") for row in cases),
            "Raw CPU case failed/errored/skipped/malformed")
    require(len({(row.get("classname"), row.get("name")) for row in cases}) == len(cases), "Duplicate raw CPU case")
    actual = dict(Counter(row.get("classname") for row in cases))
    require(actual == expected, "Exact raw CPU module counts differ")
    return sum(actual.values())


def owned_bytes(root, excluded=()):
    root = Path(root)
    total = 0
    if not root.exists():
        return total
    require(root.is_dir() and not root.is_symlink(), "Owned root linked/special")
    for directory, directories, files in os.walk(root, followlinks=False):
        directory = Path(directory)
        for name in list(directories):
            path = directory / name
            require(path.is_dir() and not path.is_symlink(), "Owned directory linked/special")
            if path in excluded:
                directories.remove(name)
        for name in files:
            path = directory / name
            require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), "Owned file linked/special")
            total += path.stat().st_size
    return total


def storage(parent, phase="entry"):
    require(phase in ("preflight", "entry", "dispatch"), "Unknown storage observation phase")
    prior = parent_ops().storage(parent, "forensics")
    record = {"status": "failed", "parent_gate": prior, "original_gate_changed": False,
              "output_cap_bytes": OUTPUT_CAP, "metadata_cap_bytes": METADATA_CAP,
              "additional_allowance_bytes": OUTPUT_CAP + METADATA_CAP,
              "required_bytes": prior["required_bytes"] + OUTPUT_CAP + METADATA_CAP,
              "available_bytes": prior["available_bytes"], "reserve_bytes": 4 * GIB, "file_reserve": 100000}
    try:
        require(prior.get("status") == "passed" and prior.get("original_gate_changed") is False
                and prior.get("reserve_bytes") == 4 * GIB and prior.get("file_reserve") == 100000,
                "Unchanged full parent storage/reserves did not pass")
        record["output_bytes"] = owned_bytes(ROOT / "gpu/payload")
        record["metadata_bytes"] = owned_bytes(ROOT, (ROOT / "gpu/payload",)) + owned_bytes(SOURCE.parent)
        require(record["output_bytes"] <= OUTPUT_CAP and record["metadata_bytes"] <= METADATA_CAP,
                "Own bounded diagnostic/control allowance exhausted")
        require(type(record["available_bytes"]) is int and type(record["required_bytes"]) is int
                and record["available_bytes"] >= record["required_bytes"], "Full campaign plus new diagnostic allowance does not fit")
        record["status"] = "passed"
    finally:
        write(Path(parent) / ("forensics-storage-" + phase + ".json"), record)
    return record


def certificate(pin, manifest):
    value = read(ROOT / "cpu-certificate.json")
    require(value.get("status") == "passed" and value.get("source_manifest_sha256") == pin
            and value.get("plan_sha256") == sha(SOURCE / "plan.json")
            and value.get("original_scientific_cpu_replaced") is False,
            "Own allocated CPU certificate identity differs")
    proofs = value.get("proofs")
    require(isinstance(proofs, list) and proofs, "CPU certificate has no original raw proofs")
    for proof in proofs:
        check_binding(proof)
    require(xml_cases(ROOT / "cpu/payload/tests.xml", manifest["cpu_counts"]) == value["cases"], "Raw CPU cases changed")
    sub = read(ROOT / "cpu/submission.json")
    lsf_success((ROOT / "cpu" / ("stdout." + sub["job_id"] + ".log")).read_text(), sub["job_id"], "mosaic-pro-forensics-cpu", "hpc")
    return value


def native_argv(plan):
    return [PYTHON, str(SOURCE / "scripts/protenix_forensics_native.py"), "--plan", str(SOURCE / "plan.json"),
            "--plan-sha256", sha(SOURCE / "plan.json"), "--output", plan["output"]]


def execute(pin, phase):
    require(phase in PHASES and os.environ.get("LSB_JOBID"), "Real allocated diagnostic phase required")
    deadline = time.monotonic() + (3300 if phase == "cpu" else 1080 if phase == "after-cpu" else 6600)
    manifest = verify(SOURCE, pin)
    plan = validate_plan(read(SOURCE / "plan.json"))
    require(manifest["files"].get("plan.json") == sha(SOURCE / "plan.json"), "Own plan not source-bound")
    own(phase, pin)
    adoption = external(manifest, plan)
    parent = ROOT / phase
    storage(parent)
    if phase == "cpu":
        payload = parent / "payload"
        payload.mkdir(exist_ok=False)
        temporary = Path(os.environ["TMPDIR"])
        require(temporary.is_absolute() and temporary.is_dir() and not temporary.is_symlink(), "Owned allocated TMPDIR unavailable")
        command = [PYTHON, "-m", "pytest", str(SOURCE / "tests"), "--rootdir", str(SOURCE), "--import-mode=importlib",
                   "-o", "addopts=", "-o", "cache_dir=" + str(payload / "cache"),
                   "--basetemp", str(temporary / "protenix-forensics-fixtures"), "-q", "--junitxml=" + str(payload / "tests.xml")]
        write(payload / "command.json", {"argv": command})
        with (payload / "tests.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=max(1, deadline - time.monotonic()))
        require(result.returncode == 0, "Allocated forensic CPU fixtures failed")
        cases = xml_cases(payload / "tests.xml", manifest["cpu_counts"])
        write(payload / "assessment.json", {"status": "passed", "cases": cases, "source_manifest_sha256": pin,
              "plan_sha256": sha(SOURCE / "plan.json"), "adoption": adoption,
              "native_result_established": False, "can_certify_or_restart_original_study": False})
        return
    if phase == "after-cpu":
        import fcntl
        with (ROOT / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            require(not (ROOT / "cpu-certificate.json").exists() and not (ROOT / "gpu").exists(),
                    "Existing CPU/GPU/unknown diagnostic outcome cannot be repeated")
            proofs = terminal_cpu()
            cases = xml_cases(ROOT / "cpu/payload/tests.xml", manifest["cpu_counts"])
            assessment = read(ROOT / "cpu/payload/assessment.json")
            require(assessment.get("status") == "passed" and assessment.get("cases") == cases
                    and assessment.get("source_manifest_sha256") == pin
                    and assessment.get("plan_sha256") == sha(SOURCE / "plan.json")
                    and assessment.get("can_certify_or_restart_original_study") is False,
                    "Own CPU assessment identity differs")
            proofs += [binding(ROOT / "cpu/payload" / name) for name in ("tests.xml", "tests.log", "command.json", "assessment.json")]
            write(ROOT / "cpu-certificate.json", {"status": "passed", "cases": cases, "source_manifest_sha256": pin,
                  "plan_sha256": sha(SOURCE / "plan.json"), "proofs": proofs, "original_scientific_cpu_replaced": False})
            fresh = parent / "dispatch-storage"
            fresh.mkdir(exist_ok=False)
            storage(fresh, "dispatch")
            require(time.monotonic() < deadline, "Bounded callback metadata budget exhausted")
            submit("gpu", pin)
        return
    certificate(pin, manifest)
    command = native_argv(plan)
    write(parent / "native-invocation.json", {"argv": command, "optimization_updates": 0,
          "original_results_changed": False, "can_certify_or_restart_original_study": False})
    record = {"execution_status": "unknown", "preservation_status": "failed", "source_manifest_sha256": pin,
              "original_results_changed": None, "argv": command}
    try:
        result = subprocess.run(command, timeout=max(1, deadline - time.monotonic()))
        record.update(execution_status="returned", returncode=result.returncode)
        require(result.returncode == 0, "Separate native forensic evaluation failed")
    except BaseException as error:
        record["execution_error"] = repr(error)
        raise
    finally:
        try:
            verify(SOURCE, pin)
            external(manifest, plan)
            require(owned_bytes(ROOT / "gpu/payload") <= OUTPUT_CAP, "Native diagnostic output cap exceeded")
            record.update(preservation_status="passed", original_results_changed=False)
        except BaseException as error:
            record["preservation_error"] = repr(error)
            raise
        finally:
            write(parent / "post-execution-preservation.json", record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    args = parser.parse_args()
    execute(args.pin, args.phase)


if __name__ == "__main__":
    main()
