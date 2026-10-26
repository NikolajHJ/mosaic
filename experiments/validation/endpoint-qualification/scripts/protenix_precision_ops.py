"""Allocated metadata-only control for prospective Protenix precision validation.

The only native phase is a bounded fixed-point precision gate.  There is no
optimizer, continuation or original-run recertification branch in this capsule.
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

VERSION = "protenix-endpoint-precision-source-v2"
PLAN_VERSION = "protenix-endpoint-precision-plan-v2"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
SOURCE = BASE / "protenix-endpoint-fix-release-05/source"
ROOT = BASE / "protenix-endpoint-fix-05"
PARENT_SOURCE = BASE / "nonchem-endpoint-release-02/source"
PARENT_PIN = "88372f92b6349e6c34c542fb84fb39482c4c340568431d1499ebc166a1e1e7fa"
PARENT_FILES = 14
FORENSIC_SOURCE = BASE / "protenix-endpoint-forensics-release-01/source"
FORENSIC_PIN = "d127d6abe87307a5d305edb6d04e3a32e3545c508e58b82ce45290e0f3fc03d8"
FORENSIC_FILES = 13
FORENSIC_ROOT = BASE / "protenix-endpoint-forensics-01"
FORENSIC_PLAN = {"path": str(FORENSIC_SOURCE / "plan.json"),
                 "sha256": "14d0cf5b03af73ee5d87ece55b8c48616897521c50bbb40e8f76b6a9bb8f5861"}
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
PHASES = ("cpu", "after-cpu", "gpu")
GIB = 1024**3
OUTPUT_CAP = 256 * 1024**2
METADATA_CAP = 16 * 1024**2
GPU_STATUS = "prospective_precision_gate_passed_original_failures_preserved"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def atomic(path, value):
    temporary = Path(path).with_name(Path(path).name + ".partial")
    write(temporary, value)
    temporary.replace(path)


def binding(path):
    path = Path(path)
    info = path.lstat()
    require(path.is_absolute() and stat.S_ISREG(info.st_mode) and not path.is_symlink()
            and ".." not in path.parts and all(not p.is_symlink() for p in path.parents),
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
    require(source.is_dir() and not source.is_symlink() and all(not p.is_symlink() for p in source.parents),
            "Source root or parent missing/linked")
    require(re.fullmatch("[a-f0-9]{64}", pin or "") and sha(source / "source_manifest.json") == pin,
            "Source manifest pin changed")
    manifest = read(source / "source_manifest.json")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Source inventory missing")
    for name, digest in files.items():
        require(isinstance(name, str) and re.fullmatch("[a-f0-9]{64}", digest or ""), "Malformed member")
        part = PurePosixPath(name)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name
                and name not in ("", ".", "source_manifest.json"), "Unsafe source member")
        path = source.joinpath(*part.parts)
        require(path.is_file() and not path.is_symlink() and sha(path) == digest
                and all(not p.is_symlink() for p in path.parents), "Source member changed/linked")
    actual = set()
    for path in source.rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "Source linked/special entry")
        if path.is_file(): actual.add(path.relative_to(source).as_posix())
    require(actual == set(files) | {"source_manifest.json"}, "Source scope changed")
    return manifest


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Authenticated metadata module missing")
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def parent_ops():
    require(len(verify(PARENT_SOURCE, PARENT_PIN)["files"]) == PARENT_FILES, "Parent source scope differs")
    return module(PARENT_SOURCE / "scripts/continuation_ops.py", "_precision_authenticated_parent")


def forensic_ops():
    require(len(verify(FORENSIC_SOURCE, FORENSIC_PIN)["files"]) == FORENSIC_FILES, "Forensic source scope differs")
    return module(FORENSIC_SOURCE / "scripts/protenix_forensics_ops.py", "_precision_authenticated_forensics")


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("version") == PLAN_VERSION
            and plan.get("source_root") == str(SOURCE) and plan.get("control_root") == str(ROOT),
            "Precision plan identity/root differs")
    require(plan.get("original_results_changed") is False and plan.get("optimization_settings_changed") is False
            and plan.get("can_certify_or_restart_original_study") is False and plan.get("monitor_remains_paused") is True,
            "Validation-only preservation scope differs")
    for key, value in (("max_bytes", OUTPUT_CAP), ("cap_bytes", OUTPUT_CAP),
                       ("metadata_cap_bytes", METADATA_CAP), ("internal_seconds", 5400)):
        require(type(plan.get(key)) is int and plan[key] == value, "Precision bound differs: " + key)
    require(plan.get("output") == str(ROOT / "gpu/payload"), "Native output scope differs")
    prior = plan.get("original_forensics_plan")
    require(isinstance(prior, dict) and prior.get("path") == FORENSIC_PLAN["path"]
            and prior.get("sha256") == FORENSIC_PLAN["sha256"], "Original forensic plan binding differs")
    check_binding(prior)
    original = read(prior["path"])
    require(plan.get("arm") == original.get("arm") and isinstance(plan.get("arm"), dict)
            and plan["arm"].get("id") == "protenix", "Exact original Protenix arm/settings differ")
    return plan


def external(manifest, plan):
    require(manifest.get("version") == VERSION, "Wrong precision source schema")
    parents = manifest.get("parent_sources")
    expected = {"path": str(FORENSIC_SOURCE), "manifest_sha256": FORENSIC_PIN, "files": FORENSIC_FILES}
    require(isinstance(parents, list) and expected in parents
            and len({r.get("path") for r in parents}) == len(parents), "Exact unique forensic source missing")
    for row in parents:
        require(set(row) == {"path", "manifest_sha256", "files"} and type(row["files"]) is int and row["files"] > 0,
                "Malformed parent source binding")
        require(len(verify(row["path"], row["manifest_sha256"])["files"]) == row["files"], "Parent source count differs")
    proofs = manifest.get("preserved_proofs")
    require(isinstance(proofs, list) and proofs and plan["original_forensics_plan"] in proofs,
            "Protected original forensic plan missing")
    for row in proofs: check_binding(row)
    validate_plan(plan)
    prior = forensic_ops()
    adoption = prior.external(prior.verify(FORENSIC_SOURCE, FORENSIC_PIN), prior.read(FORENSIC_SOURCE / "plan.json"))
    return {"status": "authenticated_read_only_originals", "prior_adoption": adoption,
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
        require(dependency is None, "Native/CPU job must be independent")
    parent = ROOT / phase
    command = ["bsub", *resources(phase), "-J", "mosaic-pro-precision05-" + phase,
               "-oo", str(parent / "stdout.%J.log"), "-eo", str(parent / "stderr.%J.log")]
    if dependency is not None: command += ["-w", dependency]
    return command + ["bash", "-l", str(SOURCE / "scripts/protenix_precision_stage.lsf"), str(SOURCE), pin, phase]


def submit(phase, pin, dependency=None):
    command = argv(phase, pin, dependency)
    parent = ROOT / phase; parent.mkdir(exist_ok=False)
    record = {"status": "unknown", "phase": phase, "argv": command, "dependency": dependency,
              "source_manifest_sha256": pin, "submitted_utc": datetime.now(timezone.utc).isoformat()}
    write(parent / "submission-started.json", record)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        matches = re.findall(r"Job <([1-9]\d*)> is submitted", result.stdout)
        if result.returncode == 0 and len(matches) == 1: record.update(status="submitted", job_id=matches[0])
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
        jobs[phase] = record["job_id"]; atomic(ROOT / "jobs.json", jobs)
    return record


def own(phase, pin):
    require(phase in PHASES, "Unknown precision phase")
    path = ROOT / phase / "submission.json"
    for number in range(21):
        if path.exists(): break
        require(number < 20, "Own actual submission not committed"); time.sleep(1)
    sub = read(path)
    require(sub.get("status") == "submitted" and sub.get("job_id") == os.environ.get("LSB_JOBID")
            and sub.get("phase") == phase and sub.get("source_manifest_sha256") == pin
            and sub.get("argv") == argv(phase, pin, sub.get("dependency")), "Own startup/job/source/resources differ")
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
    parent = ROOT / "cpu"; sub = read(parent / "submission.json"); job = sub["job_id"]
    require(sub.get("status") == "submitted" and sub.get("argv") == argv("cpu", sub["source_manifest_sha256"]), "CPU producer differs")
    tools = parent_ops(); proofs = [binding(parent / "submission.json")]
    for number in range(13):
        run = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, timeout=30)
        row = {"job_id": job, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
        path = parent / ("terminal-query-" + str(number).zfill(3) + ".json")
        write(path, row); proofs.append(binding(path))
        if tools.scheduler_status(row, job, "hpc") == "DONE": break
        require(number < 12, "Bounded real CPU terminal wait expired"); time.sleep(5)
    path = parent / ("stdout." + job + ".log")
    for number in range(21):
        if path.exists() and "Successfullycompleted." in "".join(path.read_text().split()): break
        require(number < 20, "Actual CPU terminal LSF unavailable"); time.sleep(1)
    lsf_success(path.read_text(), job, "mosaic-pro-precision05-cpu", "hpc")
    return proofs + [binding(path)]


def xml_cases(path, expected):
    require(isinstance(expected, dict) and expected and all(isinstance(k, str) and k.startswith("tests.test_")
            and type(v) is int and v > 0 for k, v in expected.items()), "CPU dynamic module counts missing/differ")
    cases = list(ET.parse(path).iter("testcase"))
    require(cases and all(not list(row) and row.get("classname") and row.get("name") for row in cases),
            "Raw CPU case failed/errored/skipped/malformed")
    require(len({(r.get("classname"), r.get("name")) for r in cases}) == len(cases), "Duplicate raw CPU case")
    require(dict(Counter(r.get("classname") for r in cases)) == expected, "Exact raw CPU module counts differ")
    return len(cases)


def owned_bytes(root, excluded=()):
    root = Path(root); total = 0
    if not root.exists(): return total
    require(root.is_dir() and not root.is_symlink(), "Owned root linked/special")
    for directory, directories, files in os.walk(root, followlinks=False):
        directory = Path(directory)
        for name in list(directories):
            path = directory / name
            require(path.is_dir() and not path.is_symlink(), "Owned directory linked/special")
            if path in excluded: directories.remove(name)
        for name in files:
            path = directory / name
            require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), "Owned file linked/special")
            total += path.stat().st_size
    return total


def storage(parent, phase="entry"):
    require(phase in ("preflight", "entry", "dispatch"), "Unknown storage phase")
    prior = parent_ops().storage(parent, "precision")
    # The pinned parent already charges containing BASE in full for stale quota.
    # Charge these exact current trees once more conservatively; never discount
    # other experiments, previous forensic bytes or any prospective allowance.
    current = {str(p): owned_bytes(p) for p in (FORENSIC_ROOT, FORENSIC_SOURCE.parent, ROOT, SOURCE.parent)}
    record = {"status": "failed", "parent_gate": prior, "original_gate_changed": False,
              "output_cap_bytes": OUTPUT_CAP, "metadata_cap_bytes": METADATA_CAP,
              "additional_allowance_bytes": OUTPUT_CAP + METADATA_CAP,
              "required_bytes": prior["required_bytes"] + OUTPUT_CAP + METADATA_CAP,
              "available_bytes": prior["available_bytes"] - sum(current.values()),
              "extra_conservative_current_tree_charge": current,
              "reserve_bytes": 4 * GIB, "file_reserve": 100000}
    try:
        require(prior.get("status") == "passed" and prior.get("original_gate_changed") is False
                and prior.get("reserve_bytes") == 4 * GIB and prior.get("file_reserve") == 100000,
                "Unchanged full parent storage/reserves did not pass")
        record["output_bytes"] = owned_bytes(ROOT / "gpu/payload")
        record["metadata_bytes"] = owned_bytes(ROOT, (ROOT / "gpu/payload",)) + owned_bytes(SOURCE.parent)
        require(record["output_bytes"] <= OUTPUT_CAP and record["metadata_bytes"] <= METADATA_CAP, "Own output/control cap exhausted")
        require(type(record["available_bytes"]) is int and type(record["required_bytes"]) is int
                and record["available_bytes"] >= record["required_bytes"], "Full campaign plus new precision allowance does not fit")
        current_gate = current_capacity_gate(parent, phase)
        record['current_chemistry_and_new_scopes'] = current_gate
        record['required_bytes'] = max(record['required_bytes'], current_gate['required_bytes'])
        record['available_bytes'] = min(record['available_bytes'], current_gate['available_bytes'])
        require(record['available_bytes'] >= record['required_bytes'], 'Full current chemistry/precision group budgets do not fit')
        record["status"] = "passed"
    finally:
        write(Path(parent) / ("precision-storage-" + phase + ".json"), record)
    return record


def current_capacity_gate(parent, phase):
    from current_capacity import gate
    return gate(parent, phase)


def native_argv(plan, *, startup=False):
    output = ROOT / "cpu/payload/native-startup" if startup else Path(plan["output"])
    command = [PYTHON, str(SOURCE / "scripts/protenix_precision_native.py"), "--plan", str(SOURCE / "plan.json"),
               "--plan-sha256", sha(SOURCE / "plan.json"), "--output", str(output)]
    return command + (["--startup-only"] if startup else [])


def startup_proof(plan, payload, deadline):
    command = native_argv(plan, startup=True)
    remaining = deadline - time.monotonic()
    require(remaining > 0, "CPU startup shared budget exhausted")
    run = subprocess.run(command, capture_output=True, text=True, timeout=min(180, remaining))
    invocation = {"argv": command, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    write(Path(payload) / "native-startup-invocation.json", invocation)
    require(run.returncode == 0, "Actual native startup parser/provenance command failed")
    receipt = Path(payload) / "native-startup/endpoint-startup.json"
    value = read(receipt)
    require(value.get("status") == "passed_metadata_only" and value.get("numerical_setup_entered") is False
            and type(value.get("updates")) is int and value["updates"] == 0
            and value.get("plan_sha256") == sha(SOURCE / "plan.json")
            and value.get("job_id") == os.environ.get("LSB_JOBID"), "Actual metadata-only native startup proof differs")
    return {"receipt": binding(receipt), "invocation": binding(Path(payload) / "native-startup-invocation.json")}


def verify_startup(proof):
    require(isinstance(proof, dict) and set(proof) == {"receipt", "invocation"}, "Startup proof missing")
    for row in proof.values(): check_binding(row)
    receipt, invocation = (read(proof[key]["path"]) for key in ("receipt", "invocation"))
    require(invocation.get("argv") == native_argv(read(SOURCE / "plan.json"), startup=True)
            and invocation.get("returncode") == 0 and receipt.get("status") == "passed_metadata_only"
            and receipt.get("numerical_setup_entered") is False and type(receipt.get("updates")) is int and receipt["updates"] == 0
            and receipt.get("plan_sha256") == sha(SOURCE / "plan.json")
            and receipt.get("job_id") == read(ROOT / "cpu/submission.json")["job_id"], "Authenticated startup result differs")
    return list(proof.values())


def certificate(pin, manifest):
    value = read(ROOT / "cpu-certificate.json")
    require(value.get("status") == "passed" and value.get("source_manifest_sha256") == pin
            and value.get("plan_sha256") == sha(SOURCE / "plan.json")
            and value.get("original_scientific_cpu_replaced") is False, "Allocated CPU certificate differs")
    require(isinstance(value.get("proofs"), list) and value["proofs"], "CPU raw proofs missing")
    for proof in value["proofs"]: check_binding(proof)
    require(xml_cases(ROOT / "cpu/payload/tests.xml", manifest["cpu_counts"]) == value["cases"], "Raw CPU cases changed")
    verify_startup(read(ROOT / "cpu/payload/assessment.json")["actual_native_startup"])
    sub = read(ROOT / "cpu/submission.json")
    lsf_success((ROOT / "cpu" / ("stdout." + sub["job_id"] + ".log")).read_text(), sub["job_id"], "mosaic-pro-precision05-cpu", "hpc")
    return value


def execute(pin, phase):
    require(phase in PHASES and os.environ.get("LSB_JOBID"), "Real allocated precision phase required")
    deadline = time.monotonic() + (3300 if phase == "cpu" else 1080 if phase == "after-cpu" else 6600)
    manifest = verify(SOURCE, pin); plan = validate_plan(read(SOURCE / "plan.json"))
    require(manifest["files"].get("plan.json") == sha(SOURCE / "plan.json"), "Plan not source-bound")
    own(phase, pin); adoption = external(manifest, plan); parent = ROOT / phase; storage(parent)
    if phase == "cpu":
        payload = parent / "payload"; payload.mkdir(exist_ok=False)
        startup = startup_proof(plan, payload, deadline)
        temporary = Path(os.environ["TMPDIR"])
        require(temporary.is_absolute() and temporary.is_dir() and not temporary.is_symlink(), "Owned allocated TMPDIR unavailable")
        command = [PYTHON, "-m", "pytest", str(SOURCE / "tests"), "--rootdir", str(SOURCE), "--import-mode=importlib",
                   "-o", "addopts=", "-o", "cache_dir=" + str(payload / "cache"),
                   "--basetemp", str(temporary / "protenix-precision-fixtures"), "-q", "--junitxml=" + str(payload / "tests.xml")]
        write(payload / "command.json", {"argv": command})
        with (payload / "tests.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=max(1, deadline - time.monotonic()))
        require(result.returncode == 0, "Allocated precision CPU fixtures failed")
        cases = xml_cases(payload / "tests.xml", manifest["cpu_counts"])
        write(payload / "assessment.json", {"status": "passed", "cases": cases, "source_manifest_sha256": pin,
              "plan_sha256": sha(SOURCE / "plan.json"), "adoption": adoption, "actual_native_startup": startup,
              "native_result_established": False, "can_certify_or_restart_original_study": False})
        return
    if phase == "after-cpu":
        import fcntl
        with (ROOT / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            require(not (ROOT / "cpu-certificate.json").exists() and not (ROOT / "gpu").exists(), "Existing/unknown CPU/GPU outcome cannot be replayed")
            proofs = terminal_cpu(); cases = xml_cases(ROOT / "cpu/payload/tests.xml", manifest["cpu_counts"])
            assessment = read(ROOT / "cpu/payload/assessment.json")
            require(assessment.get("status") == "passed" and assessment.get("cases") == cases
                    and assessment.get("source_manifest_sha256") == pin and assessment.get("plan_sha256") == sha(SOURCE / "plan.json")
                    and assessment.get("can_certify_or_restart_original_study") is False, "Own CPU assessment differs")
            proofs += verify_startup(assessment["actual_native_startup"])
            proofs += [binding(ROOT / "cpu/payload" / n) for n in ("tests.xml", "tests.log", "command.json", "assessment.json")]
            write(ROOT / "cpu-certificate.json", {"status": "passed", "cases": cases, "source_manifest_sha256": pin,
                  "plan_sha256": sha(SOURCE / "plan.json"), "proofs": proofs, "original_scientific_cpu_replaced": False})
            fresh = parent / "dispatch-storage"; fresh.mkdir(exist_ok=False); storage(fresh, "dispatch")
            require(time.monotonic() < deadline, "Bounded CPU callback budget exhausted"); submit("gpu", pin)
        return
    certificate(pin, manifest); command = native_argv(plan)
    write(parent / "native-invocation.json", {"argv": command, "optimization_updates": 0, "original_results_changed": False,
          "can_certify_or_restart_original_study": False})
    record = {"execution_status": "unknown", "preservation_status": "failed", "source_manifest_sha256": pin,
              "original_results_changed": None, "argv": command}
    try:
        result = subprocess.run(command, timeout=max(1, deadline - time.monotonic()))
        record.update(execution_status="returned", returncode=result.returncode)
        require(result.returncode == 0, "Separate prospective precision native validation failed")
        assessment = read(parent / "payload/assessment.json")
        require(assessment.get("status") == GPU_STATUS and type(assessment.get("updates")) is int and assessment["updates"] == 0
                and assessment.get("original_results_changed") is False and assessment.get("optimization_settings_changed") is False
                and assessment.get("can_certify_or_restart_original_study") is False, "Native precision gate/preservation scope differs")
    except BaseException as error:
        record["execution_error"] = repr(error); raise
    finally:
        try:
            verify(SOURCE, pin); external(manifest, plan)
            require(owned_bytes(ROOT / "gpu/payload") <= OUTPUT_CAP, "Native output cap exceeded")
            record.update(preservation_status="passed", original_results_changed=False)
        except BaseException as error:
            record["preservation_error"] = repr(error); raise
        finally:
            write(parent / "post-execution-preservation.json", record)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--pin", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True); args = parser.parse_args()
    execute(args.pin, args.phase)


if __name__ == "__main__":
    main()
