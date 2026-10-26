"""Authenticated orchestration for new endpoint-aware child continuations.

This module performs metadata checks only.  Native admission, frozen-model
checks, pilots and optimizer execution belong to allocated continuation_driver
children.  An original failed study is adopted as a read-only starting artifact;
it never becomes a passed original certificate.
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


VERSION = "endpoint-aware-continuation-source-v1"
PLAN_VERSION = "endpoint-aware-continuation-plan-v1"
RESULT_VERSION = "endpoint-aware-continuation-result-v1"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
ROOT = BASE / "nonchem-endpoint-continuation-06"
SOURCE = BASE / "nonchem-endpoint-release-06/source"
ADMISSION = BASE / "nonchem-stride20-startup-release-03/source"
ADMISSION_SHA = "704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab"
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
GIB = 1024**3
CONTROL_CAP = 512 * 1024**2
ADDITIONAL = GIB + 2 * GIB + 768 * 1024**2 + CONTROL_CAP
ARMS = {
    "protenix": {"offsets": [20000, 30000, 40000], "segment_steps": 10000,
                 "global_target": 50000, "rate": 0.3, "old_profile": "protenix50k40k"},
    "boltz": {"offsets": [10000, 20000], "segment_steps": 10000,
              "global_target": 30000, "rate": 0.0025, "old_profile": "boltz30k"},
    "dbmol": {"offsets": [5000, 7500], "segment_steps": 2500,
              "global_target": 10000, "rate": 0.0025, "old_profile": "dbmol10k"},
}
ORIGINALS = {
    "protenix": {"release": "protenix-extended-release-03/source", "files": 643,
                 "manifest": "da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562",
                 "config": "examples/configs/extended_continuation.json", "config_bytes": 1733,
                 "config_sha256": "f91c90770fe907c00e081f3ed28b46071ac35cc3f869edae504cf8cc16f6b445", "job_id": "29809291"},
    "boltz": {"release": "boltz-lr0025-release-03/source", "files": 635,
              "manifest": "c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6",
              "config": "examples/configs/boltz_serial_boltz2.json", "config_bytes": 1698,
              "config_sha256": "ecdf8a4103291f2f3a7bc7197dd394f03a46c3ccc74c72eb42dfc2b76eb13124", "job_id": "29809297"},
    "dbmol": {"release": "boltz-dbmol-pilot-release-02/source", "files": 657,
              "manifest": "600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed",
              "config": "examples/configs/dbmol_pilot.json", "config_bytes": 1816,
              "config_sha256": "b3b56f409a59b3b1d00228bb7dcd3f74fc08f17fdf06facfc1469922778d5497", "job_id": "29810971"},
}
DISPATCH_ARMS = ("boltz", "dbmol")
BLOCKED_ARMS = {"protenix": {"status": "blocked_failed_endpoint_admission", "source01_admission": {
    "path": str(BASE / "nonchem-endpoint-continuation-01/protenix-validate-20000/payload/study/native_gradient_checks-prospective/admission.json"),
    "bytes": 2531142,
    "sha256": "3b4e6c14aa21aed083594ab040a7faf4a50f93b4adce5a4fa679108bc531aaf1"}}}
PHASES = ("cpu", "after-cpu", "validate", "after-validate", "segment", "after-segment")
CHECKS = ("entry_admission", "hard_parity", "null_controls", "native_components",
          "pilot_early", "pilot_high", "runtime", "terminal_admission", "case_jsonl", "fp32_boundary")


def require(condition, message):
    if not condition:
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
    """Exclusive durable receipt; existing outcomes can never be replaced."""
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
    require(path.is_absolute() and not path.is_symlink() and stat.S_ISREG(info.st_mode),
            "Proof is not a regular absolute file")
    require(".." not in path.parts and all(not parent.is_symlink() for parent in path.parents), "Proof has traversal or linked parent")
    return {"path": str(path), "bytes": info.st_size, "sha256": sha(path)}


def check_binding(row):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, "Malformed exact file binding")
    require(isinstance(row["path"], str) and type(row["bytes"]) is int and row["bytes"] > 0
            and re.fullmatch("[a-f0-9]{64}", row["sha256"] or ""), "Malformed proof identity")
    require(binding(row["path"]) == row, "Proof bytes changed: " + row["path"])
    return row


def verify(source, pin):
    source = Path(source)
    require(source.is_dir() and not source.is_symlink(), "Source root missing or linked")
    require(re.fullmatch("[a-f0-9]{64}", pin or ""), "Malformed source pin")
    require(sha(source / "source_manifest.json") == pin, "Source manifest changed")
    manifest = read(source / "source_manifest.json")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Source has no authenticated files")
    for name, digest in files.items():
        require(isinstance(name, str), "Source member is not a string")
        part = PurePosixPath(name)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name
                and name not in ("", ".", "source_manifest.json")
                and re.fullmatch("[a-f0-9]{64}", digest or ""), "Unsafe source member")
        path = source.joinpath(*part.parts)
        require(path.is_file() and not path.is_symlink() and sha(path) == digest, "Source member changed: " + name)
        require(all(not parent.is_symlink() for parent in path.parents if parent != source.parent), "Linked source member parent")
    actual = set()
    for path in source.rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "Source contains an alias/special file")
        if path.is_file():
            actual.add(path.relative_to(source).as_posix())
    require(actual == set(files) | {"source_manifest.json"}, "Source member scope changed")
    return manifest


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("version") == PLAN_VERSION, "Wrong continuation plan schema")
    require(plan.get("admission_policy") == "prospective-endpoint-admission-v2"
            and plan.get("user_authorized_prospective_checks_before_children") is True
            and plan.get("original_failures_preserved") is True
            and plan.get("optimization_settings_changed") is False
            and plan.get("monitor_remains_paused") is True
            and type(plan.get("matrix_stride")) is int and plan["matrix_stride"] == 20
            and type(plan.get("loss_stride")) is int and plan["loss_stride"] == 1,
            "Prospective admission/preservation/retention scope changed")
    require(plan.get("root") == plan.get("control_root") == str(ROOT)
            and plan.get("source_root") == str(SOURCE), "Own scientific/control/source scope changed")
    require(plan.get("dispatch_arms") == list(DISPATCH_ARMS) and plan.get("blocked_arms") == BLOCKED_ARMS,
            "Only Boltz and DBMol may dispatch; original Protenix admission remains blocked")
    arms = plan.get("arms")
    require(isinstance(arms, list) and len(arms) == len(ARMS)
            and {row.get("id") for row in arms} == set(ARMS), "Require exact three distinct continuation arms")
    for arm in arms:
        declared = ARMS[arm["id"]]
        old = ORIGINALS[arm["id"]]
        require(arm.get("offsets") == declared["offsets"] and arm.get("segment_steps") == declared["segment_steps"]
                and arm.get("global_target") == declared["global_target"] and arm.get("rate") == declared["rate"], "Continuation range/rate changed")
        require(isinstance(arm.get("source"), dict) and isinstance(arm.get("config"), dict)
                and isinstance(arm.get("failed_parent"), dict) and isinstance(arm.get("inputs"), dict), "Missing original arm provenance")
        original_source = str(BASE / old["release"])
        require(arm["source"] == {"path": original_source, "manifest_sha256": old["manifest"], "files": old["files"]}
                and arm["config"] == {"path": str(Path(original_source) / old["config"]), "bytes": old["config_bytes"], "sha256": old["config_sha256"]},
                "Original source/config pin changed")
        require(arm["failed_parent"].get("job_id") == old["job_id"], "Wrong failed predecessor job")
        require(isinstance(arm["inputs"].get("original_driver_argv"), list)
                and arm["inputs"]["original_driver_argv"] and all(isinstance(v, str) for v in arm["inputs"]["original_driver_argv"]),
                "Original startup argv missing")
    return plan


def arm_plan(plan, arm):
    validate_plan(plan)
    require(arm in DISPATCH_ARMS, "Blocked or undeclared continuation arm")
    return next(row for row in plan["arms"] if row["id"] == arm)


def label(phase, arm="all", offset=0):
    require(phase in PHASES, "Undeclared continuation phase")
    if phase in ("cpu", "after-cpu"):
        require(arm == "all" and offset == 0, "CPU phase cannot select a scientific arm")
        return phase
    require(arm in DISPATCH_ARMS and type(offset) is int and offset in ARMS[arm]["offsets"], "Undeclared arm or boundary")
    return arm + "-" + phase + "-" + str(offset).zfill(5)


def parent_path(phase, arm="all", offset=0):
    return ROOT / label(phase, arm, offset)


def resources(phase):
    require(phase in PHASES, "Undeclared resources")
    if phase in ("validate", "segment"):
        return ["-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]",
                "-gpu", "num=1:mode=exclusive_process", "-W", "02:00" if phase == "validate" else "23:00"]
    return ["-q", "hpc", "-n", "4" if phase == "cpu" else "1", "-R", "span[hosts=1]",
            "-R", "rusage[mem=4GB]" if phase == "cpu" else "rusage[mem=2GB]",
            "-W", "01:00" if phase == "cpu" else "00:20"]


def argv(phase, pin, arm="all", offset=0, dependency=None):
    name = label(phase, arm, offset)
    require(re.fullmatch("[a-f0-9]{64}", pin or ""), "Malformed submission source pin")
    callback = phase.startswith("after-")
    if callback:
        require(isinstance(dependency, str) and re.fullmatch(r"ended\([1-9]\d*\)", dependency), "Callback requires exact own-ended dependency")
    else:
        require(dependency is None, "Scientific/CPU phases have no interarm dependencies")
    parent = ROOT / name
    command = ["bsub", *resources(phase), "-J", "mosaic-endpoint06-" + name,
               "-oo", str(parent / "stdout.%J.log"), "-eo", str(parent / "stderr.%J.log")]
    if dependency is not None:
        command += ["-w", dependency]
    return command + ["bash", "-l", str(SOURCE / "scripts/continuation_stage.lsf"), str(SOURCE), pin, phase, arm, str(offset)]


def submit(phase, pin, arm="all", offset=0, dependency=None):
    parent = parent_path(phase, arm, offset)
    command = argv(phase, pin, arm, offset, dependency)
    parent.mkdir(exist_ok=False)
    record = {"status": "unknown", "phase": phase, "arm": arm, "offset": offset, "argv": command,
              "dependency": dependency, "source_manifest_sha256": pin,
              "submitted_utc": datetime.now(timezone.utc).isoformat()}
    write(parent / "submission-started.json", record)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        jobs = re.findall(r"Job <([1-9]\d*)> is submitted", result.stdout)
        if result.returncode == 0 and len(jobs) == 1:
            record.update(status="submitted", job_id=jobs[0])
    except BaseException as error:
        record["error"] = repr(error)
    finally:
        write(parent / "submission.json", record)
    require(record["status"] == "submitted", "Unknown submission outcome: inspect markers; never duplicate")
    import fcntl
    with (ROOT / "jobs.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / "jobs.json"
        jobs = read(path) if path.exists() else {}
        require(parent.name not in jobs, "Existing job identity cannot be replaced")
        jobs[parent.name] = record["job_id"]
        atomic(path, jobs)
    return record


def scheduler_status(row, job, queue):
    require(isinstance(row, dict) and type(row.get("returncode")) is int
            and row["returncode"] == 0 and row.get("job_id") == job, "Actual scheduler query unavailable")
    raw = "".join(row.get("stdout", "").split())
    require(re.findall(r"Job<([^>]+)>", raw) == [job]
            and re.findall(r"Queue<([^>]+)>", raw) == [queue], "Scheduler identity/queue differs")
    states = re.findall(r"Status<([^>]+)>", raw)
    require(len(states) == 1 and states[0] in ("DONE", "RUN", "PEND"), "Actual predecessor failed or unknown")
    return states[0]


def lsf_success(text, job, name, queue):
    require(isinstance(text, str) and re.fullmatch(r"[1-9]\d*", job or ""), "Malformed raw LSF/job identity")
    compact = "".join(text.split())
    subjects = re.findall(r"Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)", text)
    subjects = [tuple("".join(value.split()) for value in row) for row in subjects]
    require(compact.count("Subject:Job") == 1 and subjects == [(job, name, "dcc", "Done")], "Same-job/name DONE LSF missing")
    require(re.findall(r"inqueue<([^>]+)>", compact) == [queue]
            and compact.count("Successfullycompleted.") == 1 and "Exitedwithexit" not in compact,
            "Same-job LSF queue/success missing")
    return True


def terminal(parent, queue):
    parent = Path(parent)
    sub = read(parent / "submission.json")
    job = sub["job_id"]
    proofs = [binding(parent / "submission.json")]
    for index in range(13):
        run = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, timeout=30)
        row = {"job_id": job, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
        path = parent / ("terminal-query-" + str(index).zfill(3) + ".json")
        write(path, row)
        proofs.append(binding(path))
        if scheduler_status(row, job, queue) == "DONE":
            break
        require(index < 12, "Bounded actual terminal observation expired")
        time.sleep(5)
    path = parent / ("stdout." + job + ".log")
    for index in range(21):
        if path.exists() and "Successfully completed." in path.read_text():
            break
        require(index < 20, "Terminal LSF output unavailable")
        time.sleep(1)
    lsf_success(path.read_text(), job, "mosaic-endpoint06-" + parent.name, queue)
    proofs.append(binding(path))
    return proofs


def xml_cases(path, expected):
    require(isinstance(expected, dict) and expected and all(type(n) is int and n > 0 for n in expected.values()), "Invalid expected CPU module counts")
    cases = list(ET.parse(path).iter("testcase"))
    require(cases and all(not list(case) and case.get("classname") and case.get("name") for case in cases), "CPU fixture missing/failed/errored/skipped")
    require(len({(row.get("classname"), row.get("name")) for row in cases}) == len(cases), "Duplicate raw CPU fixture")
    actual = dict(Counter(row.get("classname") for row in cases))
    require(actual == expected, "Raw CPU module attribution/count differs")
    return sum(actual.values())


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Missing authenticated metadata module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def blocked_admission(manifest, plan):
    """Keep the rejected Protenix endpoint immutable and outside dispatch."""
    validate_plan(plan)
    require(manifest.get("blocked_arms") == BLOCKED_ARMS, "Blocked original admission is not pinned by new source")
    blocked = BLOCKED_ARMS["protenix"]["source01_admission"]
    require(blocked in manifest["preserved_proofs"], "Blocked admission missing from preserved exact proofs")
    check_binding(blocked)
    blocked_observation = read(blocked["path"])
    require(blocked_observation.get("version") == "prospective-endpoint-admission-v1"
            and blocked_observation.get("status") == "failed"
            and blocked_observation.get("can_admit_new_child") is False
            and blocked_observation.get("original_run_failed_preserved") is True
            and blocked_observation.get("can_certify_original_study") is False
            and blocked_observation.get("frozen_unchanged") is True
            and blocked_observation.get("field_outcomes", {}).get("E_upper", {}).get("passed") is False,
            "Original Protenix failed endpoint admission was changed or relabeled")
    return blocked


def external(manifest, plan):
    require(manifest.get("version") == VERSION, "Wrong new source schema")
    for row in manifest["parent_sources"]:
        observed = verify(row["path"], row["manifest_sha256"])
        require(len(observed["files"]) == row["files"], "Original source file count differs")
    for row in manifest["preserved_proofs"]:
        check_binding(row)
    validate_plan(plan)
    blocked_admission(manifest, plan)
    adoption = {}
    for arm in plan["arms"]:
        observed = verify(arm["source"]["path"], arm["source"]["manifest_sha256"])
        require(len(observed["files"]) == arm["source"]["files"], "Original scientific source scope differs")
        check_binding(arm["config"])
        failed = arm["failed_parent"]
        for name in ("endpoint", "summary", "stdout", "stderr", "states", "updates", "submission", "terminal"):
            check_binding(failed[name])
        require(read(failed["summary"]["path"])["status"] == "failed", "Failed original summary was relabeled")
        sub = read(failed["submission"]["path"])
        require(sub.get("job_id") == failed["job_id"] and sub.get("outcome", sub.get("status")) == "submitted", "Failed original submission owner differs")
        raw = "".join(Path(failed["stdout"]["path"]).read_text().split())
        require(raw.count("Subject:Job") == 1 and "Subject:Job" + failed["job_id"] + ":" in raw
                and "Exitedwithexitcode1." in raw and "inqueue<gpul40s>" in raw, "Original same-job scientific failure missing")
        for row in arm["inputs"].values():
            if isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}:
                check_binding(row)
        adoption[arm["id"]] = {"status": "failed_original_artifact_adopted", "original_scientific_status": "failed",
                                "original_gate_replaced": False, "endpoint": failed["endpoint"]}
    return {"status": "read_only_failed_artifact_adoption", "arms": adoption, "original_results_changed": False}


def sparse_projection(unit, steps, segments):
    require(all(type(x) is int and x > 0 for x in (unit, steps, segments)) and steps % segments == 0,
            "Invalid full sparse output projection")
    length = steps // segments
    return 2 * unit * segments * ((length + 19) // 20 + 3) + segments * GIB


def projection_excess(original, projections):
    require(isinstance(projections, dict) and set(projections) <= set(ARMS), "Unknown new projection arm")
    details = {}
    total = 0
    for arm, value in projections.items():
        require(isinstance(value, dict) and type(value.get("unit_bytes")) is int and value["unit_bytes"] > 0,
                "New pilot empirical maximum missing")
        offsets = value.get("remaining_offsets", ARMS[arm]["offsets"])
        require(isinstance(offsets, list) and offsets and offsets == [x for x in ARMS[arm]["offsets"] if x >= offsets[0]], "Remaining offsets not a declared suffix")
        full = sparse_projection(value["unit_bytes"], len(offsets) * ARMS[arm]["segment_steps"], len(offsets))
        old = original["selected"][ARMS[arm]["old_profile"]]["remaining"]
        require(type(old) is int and old >= 0, "Original unfunded projection missing")
        excess = max(0, full - old)
        details[arm] = {"unit_bytes": value["unit_bytes"], "remaining_offsets": offsets,
                        "new_full_remaining_projection": full, "original_unfunded_projection": old,
                        "positive_excess_bytes": excess}
        total += excess
    return total, details


def metadata_bytes():
    """Bound the new control/validation/source allowance, not projected studies.

    Production studies have their own empirical full-output projections.  The
    rest of the new capsule, including every validation output and raw log, must
    fit the separately funded 512MiB control allowance.
    """
    excluded = {parent_path("segment", arm, offset) / "payload/study"
                for arm in DISPATCH_ARMS for offset in ARMS[arm]["offsets"]}
    total = 0
    for root in (ROOT, SOURCE.parent):
        if not root.exists():
            continue
        require(root.is_dir() and not root.is_symlink(), "Owned metadata root missing or linked")
        for directory, directories, files in os.walk(root, followlinks=False):
            directory = Path(directory)
            for name in list(directories):
                path = directory / name
                if path in excluded:
                    require(path.is_dir() and not path.is_symlink(), "Projected study root is linked")
                    directories.remove(name)
                    continue
                info = path.lstat()
                require(stat.S_ISDIR(info.st_mode) and not path.is_symlink(), "Owned metadata directory is linked/special")
            for name in files:
                path = directory / name
                info = path.lstat()
                require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "Owned metadata file is linked/special")
                total += info.st_size
    return total


def storage(parent, phase):
    verify(ADMISSION, ADMISSION_SHA)
    if str(ADMISSION) not in sys.path:
        sys.path.insert(0, str(ADMISSION))
    loaded = sys.modules.get("selected_storage")
    require(loaded is None or Path(loaded.__file__).resolve() == (ADMISSION / "selected_storage.py").resolve(), "Unverified storage helper already imported")
    from selected_storage import storage_gate
    old = storage_gate(parent, phase)
    projections = read(ROOT / "projections.json") if (ROOT / "projections.json").exists() else {}
    excess, details = projection_excess(old, projections)
    work3 = old["work3"]
    record = {"status": "failed", "whole_original_gate": old, "projection_excess": details,
              "positive_projection_excess_bytes": excess, "additional_allowance_bytes": ADDITIONAL,
              "required_bytes": work3["required_bytes"] + ADDITIONAL + excess,
              "available_bytes": work3["available_bytes_conservative"], "reserve_bytes": 4 * GIB,
              "file_reserve": 100000, "original_gate_changed": False,
              "scope": "Whole original forecasts plus report1GiB/chemistry2GiB/diagnostics768MiB/newcontrol512MiB and positive empirical excess"}
    try:
        require(old["status"] == "passed" and work3["status"] == "passed", "Original whole campaign storage failed")
        record["new_metadata_bytes"] = metadata_bytes()
        require(record["new_metadata_bytes"] <= CONTROL_CAP, "New control/validation/source allowance exhausted")
        require(record["available_bytes"] >= record["required_bytes"], "Whole continuation campaign storage does not fit")
        current = current_capacity_gate(parent, phase, excess)
        record['current_chemistry_and_new_scopes'] = current
        record['required_bytes'] = max(record['required_bytes'], current['required_bytes'])
        record['available_bytes'] = min(record['available_bytes'], current['available_bytes'])
        require(record['available_bytes'] >= record['required_bytes'], 'Current chemistry futures plus complete continuation budgets do not fit')
        record["status"] = "passed"
    finally:
        write(Path(parent) / ("continuation-storage-" + phase + ".json"), record)
    return record


def current_capacity_gate(parent, phase, excess=0):
    from current_capacity import gate
    return gate(parent, phase, excess)


def certificate(pin, manifest):
    path = ROOT / "cpu-certificate.json"
    value = read(path)
    require(value.get("status") == "passed" and value.get("source_manifest_sha256") == pin
            and value.get("plan_sha256") == sha(SOURCE / "plan.json"), "New allocated CPU certificate missing")
    for row in value["proofs"]:
        check_binding(row)
    require(xml_cases(ROOT / "cpu/payload/tests.xml", manifest["cpu_counts"]) == value["cases"], "CPU raw cases changed")
    return value


def own(phase, pin, arm, offset):
    parent = parent_path(phase, arm, offset)
    path = parent / "submission.json"
    for index in range(21):
        if path.exists():
            break
        require(index < 20, "Own actual submission not committed")
        time.sleep(1)
    sub = read(path)
    require(sub.get("status") == "submitted" and sub.get("job_id") == os.environ.get("LSB_JOBID")
            and sub.get("source_manifest_sha256") == pin
            and sub.get("argv") == argv(phase, pin, arm, offset, sub.get("dependency")), "Actual own startup/source/resources differ")
    gpu = phase in ("validate", "segment")
    require(os.environ.get("LSB_QUEUE") == ("gpul40s" if gpu else "hpc")
            and os.environ.get("LSB_DJOB_NUMPROC") == ("4" if gpu or phase == "cpu" else "1"), "Actual allocated queue/slots differ")
    if phase.startswith("after-"):
        before = "cpu" if phase == "after-cpu" else phase.removeprefix("after-")
        predecessor = read(parent_path(before, arm, offset) / "submission.json")
        require(sub["dependency"] == "ended(" + predecessor["job_id"] + ")", "Callback does not depend on its own producer")
    return sub


def input_snapshot(plan, arm, offset):
    row = arm_plan(plan, arm)
    require(offset in row["offsets"], "Undeclared input boundary")
    index = row["offsets"].index(offset)
    if index == 0:
        return check_binding(row["failed_parent"]["endpoint"])
    previous = row["offsets"][index - 1]
    path = ROOT / (label("segment", arm, previous) + "-certificate.json")
    cert = read(path)
    require(cert.get("status") == "passed" and cert.get("arm") == arm and cert.get("global_end") == offset,
            "Own previous child segment not certified")
    require(cert.get("source_manifest_sha256") == sha(SOURCE / "source_manifest.json")
            and cert.get("plan_sha256") == sha(SOURCE / "plan.json"), "Previous child certificate source/plan differs")
    for proof in cert["proofs"]:
        check_binding(proof)
    return check_binding(cert["terminal_snapshot"])


def driver_argv(plan, pin, phase, arm, offset):
    parent = parent_path(phase, arm, offset)
    snapshot = input_snapshot(plan, arm, offset)
    command = [PYTHON, str(SOURCE / "scripts/continuation_driver.py"), "--plan", str(SOURCE / "plan.json"),
               "--plan-sha256", sha(SOURCE / "plan.json"), "--manifest", pin,
               "--arm", arm, "--phase", phase, "--offset", str(offset),
               "--output", str(parent / "payload/study"), "--snapshot", snapshot["path"],
               "--snapshot-sha256", snapshot["sha256"]]
    if phase == "segment":
        ready = ROOT / label("validate", arm, offset) / "payload/study/assessment.json"
        verify_ready(plan, pin, arm, offset)
        command += ["--ready", str(ready), "--ready-sha256", sha(ready)]
    return command


def cpu_startups(plan, pin, payload, deadline):
    """Exercise real generated parser/provenance commands before native setup.

    A new readiness artifact cannot exist before native validation.  The CPU
    segment sentinel therefore stops before original readiness dispatch, with an
    explicit deferred flag.  A second callback startup must exercise the actual
    newly passed readiness before the production segment can be submitted.
    """
    module = _module(SOURCE / "scripts/continuation_driver.py", "_authenticated_endpoint_startup_driver")
    reports = []
    for arm in DISPATCH_ARMS:
        offset = ARMS[arm]["offsets"][0]
        row = arm_plan(plan, arm)
        snapshot = input_snapshot(plan, arm, offset)
        for phase in ("validate", "segment"):
            directory = Path(payload) / ("startup-" + arm + "-" + phase)
            directory.mkdir(exist_ok=False)
            output = directory / "study"
            ready = check_binding(row["inputs"]["ready"]) if phase == "segment" else None
            command = [PYTHON, str(SOURCE / "scripts/continuation_driver.py"), "--plan", str(SOURCE / "plan.json"),
                       "--plan-sha256", sha(SOURCE / "plan.json"), "--manifest", pin,
                       "--arm", arm, "--phase", phase, "--offset", str(offset), "--output", str(output),
                       "--snapshot", snapshot["path"], "--snapshot-sha256", snapshot["sha256"]]
            if ready is not None:
                command += ["--ready", ready["path"], "--ready-sha256", ready["sha256"]]
            command += ["--startup-only"]
            if phase == "segment":
                command += ["--startup-defer-ready"]
            expected = module.build_original_argv(plan, arm, phase, offset, output, snapshot, ready)
            remaining = deadline - time.monotonic()
            require(remaining > 0, "CPU shared startup/fixture budget exhausted")
            run = subprocess.run(command, capture_output=True, text=True, timeout=min(180, remaining))
            invocation = {"argv": command, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
                          "scope": "metadata parser/provenance only; CPU segment stops before readiness; original ready never authorizes new production"}
            write(directory / "invocation.json", invocation)
            require(run.returncode == 0, "Actual generated startup command failed before native dispatch")
            path = directory / "endpoint-startup.json"
            receipt = read(path)
            require(receipt.get("status") == "passed_metadata_only" and receipt.get("numerical_setup_entered") is False
                    and receipt.get("job_id") == os.environ.get("LSB_JOBID")
                    and type(receipt.get("updates")) is int and receipt["updates"] == 0
                    and receipt.get("readiness_deferred") is (phase == "segment")
                    and receipt.get("original_argv") == expected, "Actual startup metadata sentinel differs")
            reports.append({"arm": arm, "phase": phase, "receipt": binding(path), "invocation": binding(directory / "invocation.json")})
    workers = cpu_worker_startups(plan, pin, payload, deadline)
    write(Path(payload) / "actual-startup-checks.json", {"status": "passed_metadata_only", "checks": reports,
                                                         "worker_checks": workers,
                                                         "blocked_arms": BLOCKED_ARMS,
                                                         "native_readiness_established": False,
                                                         "fresh_prospective_readiness_deferred_until_native_validation": True})
    return reports


def cpu_worker_startups(plan, pin, payload, deadline):
    """Run real old worker import/help paths without loading models or updating."""
    reports = []
    for arm in DISPATCH_ARMS:
        offset = ARMS[arm]["offsets"][0]
        row = arm_plan(plan, arm)
        snapshot = input_snapshot(plan, arm, offset)
        directory = Path(payload) / ("worker-startup-" + arm)
        directory.mkdir(exist_ok=False)
        output = directory / "study"
        command = [PYTHON, str(SOURCE / "scripts/continuation_driver.py"), "--plan", str(SOURCE / "plan.json"),
                   "--plan-sha256", sha(SOURCE / "plan.json"), "--manifest", pin,
                   "--arm", arm, "--phase", "validate", "--offset", str(offset), "--output", str(output),
                   "--snapshot", snapshot["path"], "--snapshot-sha256", snapshot["sha256"], "--worker-startup-only"]
        remaining = deadline - time.monotonic()
        require(remaining > 0, "CPU shared worker startup/fixture budget exhausted")
        run = subprocess.run(command, capture_output=True, text=True, timeout=min(180, remaining))
        write(directory / "invocation.json", {"argv": command, "returncode": run.returncode,
              "stdout": run.stdout, "stderr": run.stderr,
              "scope": "Actual old worker interpreter/import/help only; models and updates forbidden"})
        require(run.returncode == 0, "Actual worker import/startup command failed before native dispatch")
        receipt_path = directory / "worker-startup.json"
        receipt = read(receipt_path)
        verify_worker_startup(receipt, row, arm)
        reports.append({"arm": arm, "receipt": binding(receipt_path), "invocation": binding(directory / "invocation.json")})
    return reports


def verify_worker_startup(receipt, row, arm):
    original = row["inputs"]["original_driver_argv"]
    worker_python = original[original.index("--worker-python") + 1] if "--worker-python" in original else "/work3/s234818/latentmol/env/boltz311/bin/python"
    require(receipt.get("status") == "passed_metadata_only" and receipt.get("arm") == arm
            and receipt.get("worker_python") == worker_python
            and receipt.get("mosaic_origin") == str(Path(row["source"]["path"]) / "src/mosaic/__init__.py")
            and receipt.get("numerical_setup_entered") is False
            and type(receipt.get("updates")) is int and receipt["updates"] == 0
            and receipt.get("original_scientific_source_unchanged") is True,
            "Actual worker startup/import origin differs")
    return receipt


def startup_proofs(startups, plan):
    """Authenticate four real CLI routes and both real worker imports."""
    require(startups.get("status") == "passed_metadata_only" and startups.get("native_readiness_established") is False
            and startups.get("fresh_prospective_readiness_deferred_until_native_validation") is True
            and startups.get("blocked_arms") == BLOCKED_ARMS
            and len(startups.get("checks", [])) == 4
            and {(row["arm"], row["phase"]) for row in startups.get("checks", [])}
                == {(arm, phase) for arm in DISPATCH_ARMS for phase in ("validate", "segment")}
            and len(startups.get("worker_checks", [])) == 2
            and {row["arm"] for row in startups.get("worker_checks", [])} == set(DISPATCH_ARMS),
            "Actual four CLI and two worker startup commands not established")
    proofs = []
    for row in startups["checks"] + startups["worker_checks"]:
        proofs += [check_binding(row["receipt"]), check_binding(row["invocation"])]
    for row in startups["worker_checks"]:
        verify_worker_startup(read(row["receipt"]["path"]), arm_plan(plan, row["arm"]), row["arm"])
    return proofs


def final_startup(plan, pin, arm, offset, parent, deadline):
    """Test the actual new readiness route before any production submission."""
    verify_ready(plan, pin, arm, offset)
    ready = binding(ROOT / label("validate", arm, offset) / "payload/study/assessment.json")
    snapshot = input_snapshot(plan, arm, offset)
    output = Path(parent) / "startup-final/payload/study"
    output.parent.mkdir(parents=True, exist_ok=False)
    module = _module(SOURCE / "scripts/continuation_driver.py", "_authenticated_endpoint_final_startup_driver")
    expected = module.build_original_argv(plan, arm, "segment", offset, output, snapshot, ready)
    command = [PYTHON, str(SOURCE / "scripts/continuation_driver.py"), "--plan", str(SOURCE / "plan.json"),
               "--plan-sha256", sha(SOURCE / "plan.json"), "--manifest", pin, "--arm", arm,
               "--phase", "segment", "--offset", str(offset), "--output", str(output),
               "--snapshot", snapshot["path"], "--snapshot-sha256", snapshot["sha256"],
               "--ready", ready["path"], "--ready-sha256", ready["sha256"], "--startup-only"]
    remaining = deadline - time.monotonic()
    require(remaining > 0, "Actual readiness startup budget exhausted")
    run = subprocess.run(command, capture_output=True, text=True, timeout=min(180, remaining))
    invocation = output.parent / "invocation.json"
    write(invocation, {"argv": command, "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
                       "actual_new_readiness": ready, "scope": "Actual prospective readiness route, stopped before model/updates"})
    require(run.returncode == 0, "Actual newly qualified readiness startup failed")
    receipt_path = output.parent / "endpoint-startup.json"
    receipt = read(receipt_path)
    require(receipt.get("status") == "passed_metadata_only" and receipt.get("numerical_setup_entered") is False
            and receipt.get("job_id") == read(Path(parent) / "submission.json")["job_id"]
            and type(receipt.get("updates")) is int and receipt["updates"] == 0
            and receipt.get("readiness_deferred") is False and receipt.get("original_argv") == expected,
            "Actual new readiness dispatch was not exercised")
    value = {"status": "passed_metadata_only", "source_manifest_sha256": pin, "plan_sha256": sha(SOURCE / "plan.json"),
             "arm": arm, "offset": offset, "readiness": ready, "input_snapshot": snapshot,
             "readiness_deferred": False, "numerical_setup_entered": False, "updates": 0,
             "proofs": [binding(receipt_path), binding(invocation), binding(parent / "submission.json"),
                        binding(ROOT / (label("validate", arm, offset) + "-certificate.json"))]}
    write(ROOT / (label("validate", arm, offset) + "-segment-startup.json"), value)
    return value


def verify_segment_startup(plan, pin, arm, offset):
    value = read(ROOT / (label("validate", arm, offset) + "-segment-startup.json"))
    ready = binding(ROOT / label("validate", arm, offset) / "payload/study/assessment.json")
    require(value.get("status") == "passed_metadata_only" and value.get("source_manifest_sha256") == pin
            and value.get("plan_sha256") == sha(SOURCE / "plan.json") and value.get("arm") == arm and value.get("offset") == offset
            and value.get("readiness_deferred") is False and value.get("numerical_setup_entered") is False
            and type(value.get("updates")) is int and value["updates"] == 0
            and value.get("readiness") == ready and value.get("input_snapshot") == input_snapshot(plan, arm, offset),
            "Own exact new-readiness startup certificate missing")
    for row in value["proofs"]:
        check_binding(row)
    return value


def verify_result(plan, pin, phase, arm, offset):
    require(phase in ("validate", "segment"), "Only scientific producers have results")
    row = arm_plan(plan, arm)
    parent = parent_path(phase, arm, offset)
    path = parent / "payload/assessment.json"
    result = read(path)
    require(result.get("version") == RESULT_VERSION and result.get("status") == "passed"
            and result.get("arm") == arm and result.get("phase") == phase and result.get("offset") == offset,
            "Own scientific producer assessment failed or changed")
    producer = read(parent / "submission.json")
    require(producer.get("status") == "submitted" and result.get("job_id") == producer.get("job_id"), "Scientific assessment belongs to another actual job")
    preserved = read(parent / "post-execution-preservation.json")
    require(preserved.get("preservation_status") == "passed" and preserved.get("source_manifest_sha256") == pin
            and preserved.get("original_results_changed") is False, "Original source/result after-check missing or failed")
    require(result.get("source_manifest_sha256") == pin and result.get("plan_sha256") == sha(SOURCE / "plan.json")
            and result.get("config_sha256") == row["config"]["sha256"], "Scientific result source/plan/config differs")
    require(result.get("original_results_changed") is False and result.get("frozen_unchanged") is True,
            "Original results or frozen identities changed")
    require(result.get("original_failed_parent_preserved") is True
            and result.get("endpoint_policy") == "prospective-endpoint-admission-v2"
            and result.get("scientific_source_manifest_sha256") == row["source"]["manifest_sha256"],
            "Prospective policy/original failure/source scope differs")
    expected_applicability = {"pilot_high": "not_applicable_chemistry_disabled",
                              "terminal_admission": "not_applicable_validation_phase" if phase == "validate" else "required",
                              "case_jsonl": "not_applicable_validation_phase" if phase == "validate" else "required",
                              "fp32_boundary": "required",
                              "null_controls": "typed_bonds_required_nonnull" if arm == "boltz" else "original_null_controls"}
    require(result.get("applicability") == expected_applicability, "A phase-inapplicable check lacks its explicit scope")
    require(isinstance(result.get("checks"), dict) and set(result["checks"]) == set(CHECKS)
            and all(result["checks"][key] is True for key in CHECKS), "Required prospective/native/pilot/runtime/boundary gate missing")
    require(isinstance(result.get("proofs"), list) and result["proofs"], "Scientific result has no file-bound raw proofs")
    for proof in result["proofs"]:
        check_binding(proof)
    require(result.get("input_snapshot") == input_snapshot(plan, arm, offset), "Scientific input endpoint differs")
    projection = result.get("projections")
    require(isinstance(projection, dict) and type(projection.get("maximum_pilot_update_bytes")) is int
            and projection["maximum_pilot_update_bytes"] > 0, "Actual complete pilot projection missing")
    require(result.get("global_end") == offset + (row["segment_steps"] if phase == "segment" else 0), "Reported global boundary differs")
    if phase == "segment":
        check_binding(result["terminal_snapshot"])
    return result


def verify_ready(plan, pin, arm, offset):
    path = ROOT / (label("validate", arm, offset) + "-certificate.json")
    value = read(path)
    require(value.get("status") == "passed" and value.get("source_manifest_sha256") == pin
            and value.get("plan_sha256") == sha(SOURCE / "plan.json") and value.get("arm") == arm and value.get("offset") == offset,
            "Own prospective native readiness certificate missing")
    for proof in value["proofs"]:
        check_binding(proof)
    verify_result(plan, pin, "validate", arm, offset)
    return value


def update_projection(arm, unit, remaining_offsets):
    require(type(unit) is int and unit > 0, "Invalid empirical pilot bytes")
    path = ROOT / "projections.json"
    values = read(path) if path.exists() else {}
    previous = values.get(arm, {}).get("unit_bytes", 0)
    values[arm] = {"unit_bytes": max(previous, unit), "remaining_offsets": remaining_offsets}
    atomic(path, values)


def _pair(phase, pin, arm, offset):
    producer = submit(phase, pin, arm, offset)
    callback = submit("after-" + phase, pin, arm, offset, "ended(" + producer["job_id"] + ")")
    return {"producer": producer["job_id"], "callback": callback["job_id"]}


def certify_producer(plan, pin, phase, arm, offset):
    parent = parent_path(phase, arm, offset)
    proofs = terminal(parent, "gpul40s")
    result = verify_result(plan, pin, phase, arm, offset)
    proofs += [binding(parent / "payload/assessment.json"), binding(parent / "post-execution-preservation.json"), *result["proofs"]]
    value = {"status": "passed", "arm": arm, "offset": offset, "global_end": result["global_end"],
             "source_manifest_sha256": pin, "plan_sha256": sha(SOURCE / "plan.json"),
             "original_scientific_failure_preserved": True, "original_gate_replaced": False,
             "proofs": proofs}
    if phase == "segment":
        value["terminal_snapshot"] = result["terminal_snapshot"]
    write(ROOT / (parent.name + "-certificate.json"), value)
    return result


def run_child(command, timeout, parent, manifest, plan, pin):
    """Retain child errors and rehash originals even after failure or timeout."""
    record = {"execution_status": "unknown", "preservation_status": "failed", "source_manifest_sha256": pin,
              "original_results_changed": None, "argv": command}
    try:
        result = subprocess.run(command, timeout=timeout)
        record.update(execution_status="returned", returncode=result.returncode)
        return result
    except BaseException as error:
        record["execution_error"] = repr(error)
        raise
    finally:
        try:
            verify(SOURCE, pin)
            external(manifest, plan)
            record.update(preservation_status="passed", original_results_changed=False)
        except BaseException as error:
            record["preservation_error"] = repr(error)
            raise
        finally:
            write(Path(parent) / "post-execution-preservation.json", record)


def execute(pin, phase, arm="all", offset=0):
    phase_deadline = time.monotonic() + (3300 if phase == "cpu" else 6900 if phase == "validate" else 82200 if phase == "segment" else 1080)
    manifest = verify(SOURCE, pin)
    plan = validate_plan(read(SOURCE / "plan.json"))
    require(manifest.get("version") == VERSION and manifest["files"].get("plan.json") == sha(SOURCE / "plan.json"), "Plan not bound by own manifest")
    own(phase, pin, arm, offset)
    parent = parent_path(phase, arm, offset)
    adoption = external(manifest, plan)
    storage(parent, "entry")
    if phase == "cpu":
        deadline = phase_deadline
        payload = parent / "payload"
        payload.mkdir(exist_ok=False)
        startups = cpu_startups(plan, pin, payload, deadline)
        temporary = Path(os.environ["TMPDIR"])
        require(temporary.is_absolute() and temporary.is_dir() and not temporary.is_symlink(), "Owned allocated fixture TMPDIR missing")
        command = [PYTHON, "-m", "pytest", str(SOURCE / "tests"), "--rootdir", str(SOURCE), "--import-mode=importlib", "-o", "addopts=",
                   "-o", "cache_dir=" + str(payload / "cache"), "--basetemp", str(temporary / "endpoint-fixtures"),
                   "-q", "--junitxml=" + str(payload / "tests.xml")]
        write(payload / "command.json", {"argv": command})
        remaining = deadline - time.monotonic()
        require(remaining > 0, "CPU startup/fixture shared budget exhausted")
        with (payload / "tests.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=remaining)
        require(result.returncode == 0, "Allocated CPU fixtures failed")
        cases = xml_cases(payload / "tests.xml", manifest["cpu_counts"])
        write(payload / "assessment.json", {"status": "passed", "cases": cases, "source_manifest_sha256": pin,
                                              "plan_sha256": sha(SOURCE / "plan.json"), "adoption": adoption,
                                              "actual_startup_checks": startups,
                                              "actual_worker_startup_checks": read(payload / "actual-startup-checks.json")["worker_checks"]})
        return
    if phase in ("validate", "segment"):
        certificate(pin, manifest)
        if phase == "segment":
            verify_ready(plan, pin, arm, offset)
            verify_segment_startup(plan, pin, arm, offset)
        command = driver_argv(plan, pin, phase, arm, offset)
        write(parent / "driver-invocation.json", {"argv": command, "original_results_changed": False})
        remaining = phase_deadline - time.monotonic()
        require(remaining > 0, "Allocated metadata/driver shared runtime exhausted")
        result = run_child(command, remaining, parent, manifest, plan, pin)
        require(result.returncode == 0, "Prospective continuation driver failed")
        verify_result(plan, pin, phase, arm, offset)
        return
    import fcntl
    with (ROOT / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if phase == "after-cpu":
            proofs = terminal(ROOT / "cpu", "hpc")
            cases = xml_cases(ROOT / "cpu/payload/tests.xml", manifest["cpu_counts"])
            assessment = read(ROOT / "cpu/payload/assessment.json")
            require(assessment.get("status") == "passed" and assessment.get("cases") == cases
                    and assessment.get("source_manifest_sha256") == pin, "Allocated CPU assessment differs")
            startups = read(ROOT / "cpu/payload/actual-startup-checks.json")
            proofs += startup_proofs(startups, plan)
            require(assessment.get("actual_startup_checks") == startups["checks"]
                    and assessment.get("actual_worker_startup_checks") == startups["worker_checks"], "Startup check summary differs")
            proofs += [binding(ROOT / "cpu/payload" / name) for name in ("tests.xml", "tests.log", "assessment.json", "command.json", "actual-startup-checks.json")]
            write(ROOT / "cpu-certificate.json", {"status": "passed", "cases": cases, "source_manifest_sha256": pin,
                                                    "plan_sha256": sha(SOURCE / "plan.json"), "proofs": proofs,
                                                    "original_scientific_cpu_replaced": False})
            for item in DISPATCH_ARMS:
                require(not parent_path("validate", item, ARMS[item]["offsets"][0]).exists(), "Existing/unknown validation must not be duplicated")
            require(time.monotonic() < phase_deadline, "CPU callback metadata runtime exhausted")
            for item in DISPATCH_ARMS:
                _pair("validate", pin, item, ARMS[item]["offsets"][0])
            return
        certificate(pin, manifest)
        producer = phase.removeprefix("after-")
        result = certify_producer(plan, pin, producer, arm, offset)
        offsets = ARMS[arm]["offsets"]
        index = offsets.index(offset)
        remaining = offsets[index:] if producer == "validate" else offsets[index + 1:]
        if remaining:
            update_projection(arm, result["projections"]["maximum_pilot_update_bytes"], remaining)
        else:
            values = read(ROOT / "projections.json")
            values.pop(arm, None)
            atomic(ROOT / "projections.json", values)
        storage(parent, "dispatch")
        require(time.monotonic() < phase_deadline, "Own continuation callback runtime exhausted")
        if producer == "validate":
            final_startup(plan, pin, arm, offset, parent, phase_deadline)
            _pair("segment", pin, arm, offset)
        elif remaining:
            _pair("validate", pin, arm, remaining[0])
        else:
            write(ROOT / (arm + "-completion.json"), {"status": "passed_new_child_chain", "global_end": ARMS[arm]["global_target"],
                                                       "original_scientific_failure_preserved": True,
                                                       "terminal_certificate": binding(ROOT / (label("segment", arm, offset) + "-certificate.json"))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--arm", choices=("all", *DISPATCH_ARMS), default="all")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    require(os.environ.get("LSB_JOBID"), "A real allocated job is required")
    execute(args.pin, args.phase, args.arm, args.offset)


if __name__ == "__main__":
    main()
