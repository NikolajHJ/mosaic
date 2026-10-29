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
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


VERSION = "precision-reference-continuation-source-v1"
PLAN_VERSION = "precision-reference-continuation-plan-v1"
RESULT_VERSION = "precision-reference-continuation-result-v1"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
ROOT = BASE / "nonchem-precision-continuation-22"
SOURCE = BASE / "nonchem-precision-release-22/source"
ADMISSION = BASE / "nonchem-stride20-startup-release-03/source"
ADMISSION_SHA = "704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab"
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
GIB = 1024**3
CONTROL_CAP = 512 * 1024**2
ADDITIONAL = GIB + 2 * GIB + 768 * 1024**2 + CONTROL_CAP
ARMS = {
    "protenix": {"offsets": [20000], "segment_steps": 30000,
                 "global_target": 50000, "rate": 0.3, "old_profile": "protenix50k40k"},
    "boltz": {"offsets": [10000, 20000], "segment_steps": 10000,
              "global_target": 30000, "rate": 0.0025, "old_profile": "boltz30k"},
    "dbmol": {"offsets": [7500], "segment_steps": 2500,
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
              "config_sha256": "b3b56f409a59b3b1d00228bb7dcd3f74fc08f17fdf06facfc1469922778d5497", "job_id": "29830999"},
}
DISPATCH_ARMS = ("protenix", "dbmol")
AUTOMATIC_ARMS = ("protenix", "dbmol")
EXISTING_ROOT = BASE / "nonchem-endpoint-continuation-08"
PRESERVED_FAILED_ADMISSIONS = {"protenix": {"status": "blocked_failed_endpoint_admission", "source01_admission": {
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


def check_preserved_binding(row):
    if isinstance(row, dict) and row.get("bytes") == 0:
        if row.get("path") == str(BASE / "dbmol-endpoint-recovery-13/gpu/payload/native.log"):
            require(set(row) == {"path", "bytes", "sha256"} and type(row["bytes"]) is int
                    and row["sha256"] == hashlib.sha256(b"").hexdigest() and binding(row["path"]) == row,
                    "Exact qualified native empty log changed; never a scientific pass proof")
            return row
        source = BASE / "dbmol-endpoint-recovery-release-13/source"
        verify(source, "0f82d44bbb65ea957a7e5faa9e9837aba7fc1f2256f7344a96606b11ef3d664d")
        module = _module(source / "scripts/dbmol_recovery_ops.py", "_qualified_historical_empty_lock")
        return module.check_preserved_binding(sys.modules[__name__], row)
    return check_binding(row)


def check_generated_proof(row):
    if isinstance(row, dict) and row.get("bytes") == 0:
        path = Path(row.get("path", ""))
        probe_roots = [parent_path(phase, arm, offset) / "cuda-ownership"
                       for phase in ("validate", "segment") for arm in DISPATCH_ARMS for offset in ARMS[arm]["offsets"]]
        if path.parent in probe_roots and path.name in ("production_fp32.stderr.log", "reference_fp64.stderr.log"):
            require(binding(path) == row and row["sha256"] == hashlib.sha256(b"").hexdigest(),
                    "Owned CUDA smoke empty stderr identity differs")
            return row
        studies = [parent_path(phase, arm, offset) / "payload/study"
                   for phase in ("validate", "segment") for arm in DISPATCH_ARMS for offset in ARMS[arm]["offsets"]]
        study = next((root for root in studies if path.is_relative_to(root)), None)
        require(study is not None and path.name == "stderr.log" and path.parent.name == "reference-session",
                "Only an actual new reference stderr stream may be empty")
        adapter = _module(SOURCE / "scripts/precision_continuation_driver.py", "_precision_empty_stream_verifier")
        return adapter.check_child_proof(row, study)
    return check_binding(row)


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
    require(plan.get("admission_policy") == "precision-reference-continuation-policy-v1"
            and plan.get('logit_policy')=='prospective-fixed-reference-logit-admission-v1'
            and plan.get("user_authorized_prospective_checks_before_children") is True
            and plan.get("original_failures_preserved") is True
            and plan.get("optimization_settings_changed") is False
            and plan.get("monitor_remains_paused") is True
            and type(plan.get("matrix_stride")) is int and plan["matrix_stride"] == 20
            and type(plan.get("loss_stride")) is int and plan["loss_stride"] == 1,
            "Prospective admission/preservation/retention scope changed")
    require(plan.get("root") == plan.get("control_root") == str(ROOT)
            and plan.get("source_root") == str(SOURCE), "Own scientific/control/source scope changed")
    require(plan.get("dispatch_arms") == list(DISPATCH_ARMS) and plan.get("preserved_failed_admissions") == PRESERVED_FAILED_ADMISSIONS,
            "Only newly qualified Protenix and saved7500 DBMol may dispatch")
    require(plan.get("automatic_dispatch_arms") == list(AUTOMATIC_ARMS), "Only the two newly qualified arms may dispatch automatically")
    require(plan.get("precision_adapter_version") == "precision-reference-continuation-adapter-v1"
            and isinstance(plan.get("reference_configuration"), dict)
            and set(plan["reference_configuration"].get("arms", {})) == set(DISPATCH_ARMS),
            "Closed separate-reference integration scope missing")
    arms = plan.get("arms")
    require(isinstance(arms, list) and len(arms) == len(DISPATCH_ARMS)
            and {row.get("id") for row in arms} == set(DISPATCH_ARMS), "Require exact three distinct continuation arms")
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


def segment_budget(arm, offset):
    """Use the actual complete qualification; never guess a shorter walltime."""
    require(arm in DISPATCH_ARMS and offset in ARMS[arm]["offsets"], "Undeclared runtime boundary")
    adapter = _module(SOURCE / "scripts/precision_continuation_driver.py", "_precision_measured_runtime")
    summary = parent_path("validate", arm, offset) / "payload/study/summary.json"
    return adapter.runtime_budget(summary, arm)


def resources(phase, budget=None):
    require(phase in PHASES, "Undeclared resources")
    if phase in ("validate", "segment"):
        return ["-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]",
                "-gpu", "num=1:mode=exclusive_process:mps=yes", "-W", "02:00" if phase == "validate" else budget["walltime"]]
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
    budget = segment_budget(arm, offset) if phase == "segment" else None
    from cuda_ownership import SUBMISSION_ENV
    command = ["bsub", "-env", SUBMISSION_ENV, *resources(phase, budget), "-J", "mosaic-precision22-" + name,
               "-oo", str(parent / "stdout.%J.log"), "-eo", str(parent / "stderr.%J.log")]
    if dependency is not None:
        command += ["-w", dependency]
    return command + ["bash", "-l", str(SOURCE / "scripts/precision_continuation_stage.lsf"), str(SOURCE), pin, phase, arm, str(offset)]


def submit(phase, pin, arm="all", offset=0, dependency=None):
    require(phase in ("cpu", "after-cpu") or arm in AUTOMATIC_ARMS, "Only exact new declared two-arm phases may submit")
    parent = parent_path(phase, arm, offset)
    command = argv(phase, pin, arm, offset, dependency)
    parent.mkdir(exist_ok=False)
    record = {"status": "unknown", "phase": phase, "arm": arm, "offset": offset, "argv": command,
              "dependency": dependency, "source_manifest_sha256": pin,
              "submitted_utc": datetime.now(timezone.utc).isoformat()}
    if phase == "segment":
        record["runtime_budget"] = segment_budget(arm, offset)
    write(parent / "submission-started.json", record)
    try:
        from cuda_ownership import submission_environment
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, env=submission_environment(os.environ))
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
    lsf_success(path.read_text(), job, "mosaic-precision22-" + parent.name, queue)
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


def cpu_xml_proofs(manifest):
    """Bind both interpreter runs and prove the merged record set is exact."""
    payload = ROOT / "cpu/payload"
    expected = manifest["cpu_counts"]
    groups = {"torch": {"tests.test_dbmol_reference": expected["tests.test_dbmol_reference"]},
              "jax": {k: v for k, v in expected.items() if k != "tests.test_dbmol_reference"}}
    raw_cases = []
    proofs = []
    for name, counts in groups.items():
        path = payload / ("tests-" + name + ".xml")
        xml_cases(path, counts)
        raw_cases.extend(ET.tostring(row) for row in ET.parse(path).iter("testcase"))
        proofs += [binding(payload / ("tests-" + name + suffix)) for suffix in (".xml", ".log", "-command.json")]
    merged = payload / "tests.xml"
    cases = xml_cases(merged, expected)
    require(Counter(raw_cases) == Counter(ET.tostring(row) for row in ET.parse(merged).iter("testcase")),
            "Merged CPU evidence differs from the untouched two raw interpreter records")
    return cases, proofs


def cpu_fixtures(manifest, payload, deadline):
    expected = manifest["cpu_counts"]
    require("tests.test_dbmol_reference" in expected, "Native Torch reference fixtures missing")
    temporary = Path(os.environ["TMPDIR"])
    require(temporary.is_absolute() and temporary.is_dir() and not temporary.is_symlink(), "Owned allocated fixture TMPDIR missing")
    groups = [("torch", "/work3/s234818/latentmol/env/boltz311/bin/python", ["tests.test_dbmol_reference"]),
              ("jax", PYTHON, [k for k in expected if k != "tests.test_dbmol_reference"])]
    merged = ET.Element("testsuites")
    for name, python, modules in groups:
        command = [python, "-m", "pytest", *[str(SOURCE / (m.replace(".", "/") + ".py")) for m in modules],
                   "--rootdir", str(SOURCE), "--import-mode=importlib", "-o", "addopts=", "-o",
                   "cache_dir=" + str(payload / ("cache-" + name)), "--basetemp", str(temporary / ("precision-fixtures-" + name)),
                   "-q", "--junitxml=" + str(payload / ("tests-" + name + ".xml"))]
        write(payload / ("tests-" + name + "-command.json"), {"argv": command})
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Allocated CPU shared startup/fixture budget exhausted")
        with (payload / ("tests-" + name + ".log")).open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=remaining)
        require(result.returncode == 0, "Allocated " + name + " CPU fixtures failed")
        root = ET.parse(payload / ("tests-" + name + ".xml")).getroot()
        if root.tag == "testsuite": merged.append(root)
        else:
            require(root.tag == "testsuites", "Malformed raw JUnit root")
            merged.extend(list(root))
    with (payload / "tests.xml").open("xb") as stream:
        ET.ElementTree(merged).write(stream, encoding="utf-8", xml_declaration=True)
    return cpu_xml_proofs(manifest)[0]


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Missing authenticated metadata module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preserved_failed_admission(manifest, plan):
    """Keep every rejected original Protenix endpoint immutable; only new references admit children."""
    validate_plan(plan)
    require(manifest.get("preserved_failed_admissions") == PRESERVED_FAILED_ADMISSIONS, "Blocked original admission is not pinned by new source")
    blocked = PRESERVED_FAILED_ADMISSIONS["protenix"]["source01_admission"]
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


QUALIFIED_NATIVE = {
    "protenix": {"root": str(BASE / "protenix-endpoint-fix-09"),
                 "source": {"path": str(BASE / "protenix-endpoint-fix-release-09/source"), "files": 20,
                            "manifest_sha256": "f2043af74b8fa9ff485a44b80fcf96454bf8585868859389828c37f0073a3664"},
                 "job_id": "29838596", "name": "mosaic-pro-precision09-gpu",
                 "wrapper": "protenix_precision_stage.lsf"},
    "dbmol": {"root": str(BASE / "dbmol-endpoint-recovery-13"),
              "source": {"path": str(BASE / "dbmol-endpoint-recovery-release-13/source"), "files": 14,
                         "manifest_sha256": "0f82d44bbb65ea957a7e5faa9e9837aba7fc1f2256f7344a96606b11ef3d664d"},
              "job_id": "29838712", "name": "mosaic-dbmol-reference13-gpu",
              "wrapper": "dbmol_recovery_stage.lsf"},
}


def qualified_admissions(plan):
    """Adopt actual original native component passes, never certify a child.

    Only exact Pro09 and DB13 completed qualifications are admissible. Their
    sources, actual submission/LSF, CPU certificate, raw native reports, frozen
    state and unchanged protected inputs remain independently authenticated.
    Complete live original-driver validation and every child check still follow.
    """
    declared = plan.get("qualified_admissions")
    require(isinstance(declared, dict) and set(declared) == set(DISPATCH_ARMS),
            "Both exact native qualifications are required before admission")
    adopted = {}
    for arm in DISPATCH_ARMS:
        row, expected = declared[arm], QUALIFIED_NATIVE[arm]
        require(isinstance(row, dict) and set(row) == {"source", "job_id", "assessment", "submission", "stdout", "proofs"}
                and row["source"] == expected["source"] and row["job_id"] == expected["job_id"],
                "Wrong native qualification source/job: " + arm)
        source, root = Path(expected["source"]["path"]), Path(expected["root"])
        manifest = verify(source, expected["source"]["manifest_sha256"])
        require(len(manifest["files"]) == expected["source"]["files"], "Qualified native source inventory differs")
        require(row["assessment"]["path"] == str(root / "gpu/payload/assessment.json")
                and row["submission"]["path"] == str(root / "gpu/submission.json")
                and row["stdout"]["path"] == str(root / "gpu" / ("stdout." + row["job_id"] + ".log")),
                "Qualification artifact belongs to another producer")
        proofs = [check_preserved_binding(item) for item in row["proofs"]]
        paths = {item["path"] for item in proofs}
        require(len(paths) == len(proofs) and proofs, "Missing/duplicate qualified native proof inventory")
        for field in ("assessment", "submission", "stdout"):
            check_binding(row[field])
            require(row[field]["path"] in paths, "Native qualification owner proof is not in preserved inventory")
        def evidence(relative):
            path = root / relative
            require(str(path) in paths, "Qualified native raw proof missing: " + relative)
            return read(path)
        submitted = read(row["submission"]["path"])
        command = ["bsub", "-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]",
                   "-gpu", "num=1:mode=exclusive_process", "-W", "02:00", "-J", expected["name"],
                   "-oo", str(root / "gpu/stdout.%J.log"), "-eo", str(root / "gpu/stderr.%J.log"),
                   "bash", "-l", str(source / "scripts" / expected["wrapper"]), str(source),
                   expected["source"]["manifest_sha256"], "gpu"]
        require(submitted.get("status") == "submitted" and submitted.get("job_id") == expected["job_id"]
                and submitted.get("phase") == "gpu" and submitted.get("dependency") is None
                and submitted.get("source_manifest_sha256") == expected["source"]["manifest_sha256"]
                and submitted.get("argv") == command and submitted.get("returncode") == 0,
                "Actual native qualification submission/resources differ")
        lsf_success(Path(row["stdout"]["path"]).read_text(), row["job_id"], expected["name"], "gpul40s")
        cpu = read(root / "cpu-certificate.json")
        require(cpu.get("status") == "passed" and cpu.get("source_manifest_sha256") == expected["source"]["manifest_sha256"]
                and cpu.get("cases") == sum(manifest["cpu_counts"].values()), "Qualified native CPU certificate differs")
        require(isinstance(cpu.get("proofs"), list) and cpu["proofs"], "Qualified CPU raw proof set missing")
        for item in cpu["proofs"]: check_binding(item)
        assessment = read(row["assessment"]["path"])
        if arm == "protenix":
            require(assessment.get("status") == "prospective_precision_gate_passed_original_failures_preserved"
                    and assessment.get("job_id") == row["job_id"] and assessment.get("updates") == 0
                    and assessment.get("original_results_changed") is False and assessment.get("optimization_settings_changed") is False
                    and assessment.get("can_certify_or_restart_original_study") is False
                    and assessment.get("production_optimizer_executed") is False and assessment.get("frozen_unchanged") is True
                    and assessment.get("preservation_errors") == [], "Protenix native precision/preservation did not pass")
            for key in ("plan", "reference_assessment"): check_binding(assessment[key])
            for worker in ("fp32", "fp64"):
                item = assessment["worker_proofs"][worker]
                check_binding(item["assessment"]); check_binding(item["preservation"])
                preserved = read(item["preservation"]["path"])
                require(preserved.get("status") == "passed" and preserved.get("updates") == 0
                        and preserved.get("protected_inputs_unchanged") is True and preserved.get("frozen_unchanged") is True
                        and preserved.get("preservation_errors") == [], "Qualified Protenix worker preservation incomplete")
            require(evidence("gpu/payload/protected-before.json") == evidence("gpu/payload/protected-after.json"),
                    "Qualified Protenix original protected inputs changed")
            audit = evidence("gpu/payload/reference/native-dtype-audit.json")
            require(audit.get("status") == "passed" and audit.get("floating_operations_float64") is True
                    and audit.get("narrow_float_avals") == [], "Qualified Protenix full dtype audit failed")
            admission = evidence("gpu/payload/reference/admission.json")
            expected_policy = "protenix-separate-precision-admission-v1"
            negative_prefix, negative_suffix = "gpu/payload/reference/", "-control.json"
        else:
            require(assessment.get("status") == "passed_affinity_reference_only"
                    and assessment.get("optimizer_jobs_submitted") is False, "DBMol native component qualification incomplete")
            check_binding(assessment["native_assessment"])
            actual = read(assessment["native_assessment"]["path"])
            require(actual.get("status") == "passed_affinity_reference_only" and actual.get("error") is None
                    and actual.get("optimizer_updates") == 0 and actual.get("original_results_changed") is False
                    and actual.get("can_submit_continuation") is False
                    and actual.get("remaining_admission") == "all_components_raw_null_parity_pilot_runtime_startup_full_storage",
                    "DBMol affinity-only evidence cannot establish full continuation admission")
            preserved = evidence("gpu/payload/native/preservation.json")
            require(preserved.get("status") == "passed" and preserved.get("before")
                    and preserved.get("before") == preserved.get("after"), "Qualified DBMol original inputs changed")
            audit = evidence("gpu/payload/native/reference-operation-audit.json")
            require(audit.get("status") == "closed"
                    and audit.get("audit", {}).get("status") == "passed_full_operation_FP64_audit"
                    and audit["audit"].get("narrow_floating_operations") == 0
                    and audit["audit"].get("production_execution_changed") is False,
                    "Qualified DBMol complete FP64 operation audit failed")
            require(evidence("gpu/payload/native/production-frozen-after.json").get("frozen") is True,
                    "Qualified original DBMol model/context did not remain frozen")
            admission = evidence("gpu/payload/native/native-admission.json")
            expected_policy = "dbmol-separate-precision-finer-admission-v1"
            negative_prefix, negative_suffix = "gpu/payload/native/", "-admission.json"
        require(admission.get("version") == expected_policy and admission.get("status") == "passed"
                and admission.get("can_admit_new_child") is True and admission.get("can_certify_original_study") is False
                and admission.get("original_failed_outcome_preserved") is True and admission.get("frozen_unchanged") is True
                and admission.get("baseline_fp32_unchanged") is True and admission.get("optimization_updates") == 0
                and admission.get("failures") == [] and set(admission.get("field_outcomes", {})) == {"X", "E_upper"}
                and all(item.get("passed") is True for item in admission["field_outcomes"].values())
                and admission.get("directions") and all(item.get("passed") is True for item in admission["directions"]),
                "Mandatory qualified reference FD/parity/locality/cohort/frozen proof incomplete")
        for name in ("zero", "reversed", "scaled"):
            negative = evidence(negative_prefix + name + negative_suffix)
            require(negative.get("status") == "failed" and negative.get("can_admit_new_child") is False
                    and negative.get("can_certify_original_study") is False,
                    "Qualified reference did not reject incorrect production gradients")
        adopted[arm] = {"status": "passed_native_component_qualification_only", "job_id": row["job_id"],
                        "source": row["source"], "assessment": row["assessment"],
                        "native_policy": expected_policy, "full_continuation_admission_established": False}
    return {"status": "passed_readonly_native_qualification_adoption", "arms": adopted,
            "full_continuation_admission_established": False, "original_results_changed": False}


def external(manifest, plan):
    require(manifest.get("version") == VERSION, "Wrong new source schema")
    for row in manifest["parent_sources"]:
        observed = verify(row["path"], row["manifest_sha256"])
        require(len(observed["files"]) == row["files"], "Original source file count differs")
    for row in manifest["preserved_proofs"]:
        check_preserved_binding(row)
    validate_plan(plan)
    predecessor = plan.get("launch_predecessor")
    require(isinstance(predecessor, dict) and predecessor.get("source") == {
        "path": str(BASE / "nonchem-precision-release-15/source"), "files": 35,
        "manifest_sha256": "5448503965365f4db726d558df4ac91a41a7f711366b6ae47ae936993fd0c8bb"}
        and predecessor.get("failed_jobs") == {"protenix": "29839086", "dbmol": "29839088"}
        and predecessor.get("failures_preserved") is True and predecessor.get("original_results_changed") is False,
        "Source15 failed engineering evidence must remain explicit")
    for proof in predecessor["proofs"]:
        check_binding(proof)
    failed_root = BASE / "nonchem-precision-continuation-15"
    for arm, job in predecessor["failed_jobs"].items():
        offset = 20000 if arm == "protenix" else 7500
        parent = failed_root / (arm + "-validate-" + str(offset).zfill(5))
        submitted = read(parent / "submission.json")
        require(submitted.get("job_id") == job and submitted.get("source_manifest_sha256") == predecessor["source"]["manifest_sha256"],
                "Failed source15 job identity differs")
        raw = "".join((parent / ("stdout." + job + ".log")).read_text().split())
        require(raw.count("Subject:Job" + job + ":") == 1 and "Exitedwithexitcode1." in raw
                and "inqueue<gpul40s>" in raw, "Original same-job source15 failed LSF missing")
    allocation = plan.get("superseded_allocation_plan")
    require(isinstance(allocation, dict) and allocation.get("source") == {
        "path": str(BASE / "nonchem-precision-release-16/source"), "files": 41,
        "manifest_sha256": "cd75d01b41520f7af916ff4fa824f7059d7eb4713d9c611e24d00e0cd12444f2"}
        and allocation.get("CPU_pass_preserved") is True and allocation.get("original_results_changed") is False,
        "Superseded small-allocation source/evidence missing")
    retired = read(check_binding(allocation["retirement_assessment"])["path"])
    require(retired.get("status") == "superseded_unstarted_native_jobs_and_dispatchers_intentionally_retired"
            and retired.get("source") == allocation["source"] and retired.get("optimizer_children") == []
            and retired.get("original_results_changed") is False
            and retired.get("validation_study_presence") == {"protenix-validate-20000": False, "dbmol-validate-07500": False},
            "Old small-chunk execution not safely retired before new admission")
    for proof in retired["proofs"]:
        check_binding(proof)
    expected_retired = {"cpu": "29841063", "after-cpu": "29841064", "protenix-validate-20000": "29841081",
                        "protenix-after-validate-20000": "29841082", "dbmol-validate-07500": "29841083",
                        "dbmol-after-validate-07500": "29841084"}
    require(retired.get("jobs") == expected_retired and len(retired.get("actual_scheduler", [])) == 6,
            "Superseded job inventory changed")
    for observed in retired["actual_scheduler"]:
        name, job = observed["name"], observed["job_id"]
        raw = "".join(observed["stdout"].split())
        expected_state = "DONE" if name in ("cpu", "after-cpu") else "EXIT"
        require(expected_retired.get(name) == job and observed["returncode"] == 0
                and re.findall(r"Job<([^>]+)>", raw) == [job]
                and re.findall(r"Status<([^>]+)>", raw) == [expected_state]
                and observed["state"] == expected_state, "Actual original retirement scheduler proof differs")
    previous = plan.get("engineering_fixture_predecessor")
    require(isinstance(previous, dict) and previous.get("source") == {
        "path": str(BASE / "nonchem-precision-release-17/source"), "files": 41,
        "manifest_sha256": "4d29d85856e696d7fc096dc5f12bd54ed51d64fc2032946be2b51a8d5b4c480c"},
        "Source17 fixture failure provenance missing")
    failure = read(check_binding(previous["assessment"])["path"])
    require(failure.get("status") == "preserved_CPU_fixture_message_failure_no_native_or_optimizer_submitted"
            and failure.get("source") == previous["source"]
            and failure.get("jobs") == {"cpu": "29841248", "after-cpu": "29841249"}
            and failure.get("raw_counts") == {"torch": {"cases": 17, "passed": 17, "failures": 0},
                                               "jax": {"cases": 790, "passed": 789, "failures": 1}}
            and failure.get("native_jobs_submitted") is False and failure.get("optimizer_jobs_submitted") is False,
            "Original source17 fixture failure was changed or relabeled")
    for proof in failure["proofs"]:
        check_binding(proof)
    shared = plan.get("shared_vae")
    require(type(shared) is dict and shared.get("version") == "mosaic-authenticated-vae-shared-quota-v1"
            and shared.get("growth_bytes") == 10 * GIB and shared.get("growth_files") == 30000,
            "Authenticated concurrent VAE accounting missing")
    for key in ("plan", "archive", "manifest", "quota_before"):
        check_binding(shared[key])
    resource = plan.get("resource_failure_predecessor")
    require(type(resource) is dict and resource.get("source") == {
        "path": str(BASE / "nonchem-precision-release-18/source"), "files": 42,
        "manifest_sha256": "15fb307c704a0f460f882c701df6eec0655d678554c56f3abb581cbacf6a4b0e"},
        "Source18 resource failure evidence missing")
    preserved18 = read(check_binding(resource["assessment"])["path"])
    require(preserved18.get("status") == "preserved_CPU817_pass_native_entry_resource_failures_no_models_or_optimizers"
            and preserved18.get("CPU_cases") == 817 and preserved18.get("native_models_invoked") is False
            and preserved18.get("optimizer_jobs_submitted") is False,
            "Source18 resource failures were relabeled")
    for proof in preserved18["proofs"]:
        check_binding(proof)
    preflight = plan.get("preflight_file_predecessor")
    require(type(preflight) is dict and preflight.get("source") == {
        "path": str(BASE / "nonchem-precision-release-19/source"), "files": 44,
        "manifest_sha256": "a428f807f566b0a4e765178d8699e501691d8d6b97c7cc08227c0768ecff3dda"},
        "Source19 preflight failure missing")
    preserved19 = read(check_binding(preflight["assessment"])["path"])
    require(preserved19.get("status") == "preserved_preflight_file_cap_failure_before_any_CPU_or_native_submission"
            and preserved19.get("jobs_submitted") is False, "Source19 failed preflight was relabeled")
    for proof in preserved19["proofs"]:
        check_binding(proof)
    conflict = plan.get("report_conflict_predecessor")
    require(type(conflict) is dict and conflict.get("source") == {
        "path": str(BASE / "nonchem-precision-release-20/source"), "files": 44,
        "manifest_sha256": "0f5411bb45c408a0591ff8c985460398ee7196b27759b2eaf66c909712eab9d6"},
        "Source20 reader-conflict evidence missing")
    preserved20 = read(check_binding(conflict["assessment"])["path"])
    require(preserved20.get("status") == "preserved_CPU861_pass_callback_reader_classification_failure_before_any_native_or_optimizer"
            and preserved20.get("CPU_cases") == 861 and preserved20.get("native_or_optimizer_submitted") is False,
            "Source20 CPU pass or failed callback was changed")
    for proof in preserved20["proofs"]:
        check_binding(proof)
    preserved_failed_admission(manifest, plan)
    qualified = qualified_admissions(plan)
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
    return {"status": "read_only_failed_artifact_adoption", "arms": adoption,
            "qualified_native_components": qualified, "original_results_changed": False}


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


def projection_excess_from_gate(prior):
    """Fund only positive fresh-pilot excess over the smallest inherited budget."""
    candidates = []
    def visit(value):
        if isinstance(value, dict):
            if value.get("version") == "selected-nonchem-full-storage-v1":
                selected = value.get("selected")
                require(isinstance(selected, dict), "Inherited selected forecasts missing")
                candidates.append(selected)
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(prior)
    require(candidates, "No authenticated inherited selected full forecasts")
    selected = {}
    for name in DISPATCH_ARMS:
        profile = ARMS[name]["old_profile"]
        values = [row[profile]["remaining"] for row in candidates if profile in row]
        require(values and all(type(n) is int and n >= 0 for n in values), "Invalid inherited full forecast")
        selected[profile] = {"remaining": min(values)}
    path = ROOT / "projections.json"
    projections = read(path) if path.exists() else {}
    require(set(projections) <= set(DISPATCH_ARMS), "Undeclared new empirical forecast")
    return projection_excess({"selected": selected}, projections)


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



def owned_bytes(root):
    root = Path(root)
    if not root.exists(): return 0
    require(root.is_dir() and not root.is_symlink(), "Owned root linked/special")
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            p = Path(directory) / name
            require(p.is_dir() and not p.is_symlink(), "Owned directory linked/special")
        for name in files:
            p = Path(directory) / name; info = p.lstat()
            require(stat.S_ISREG(info.st_mode), "Owned file linked/special")
            total += info.st_size
    return total


def legacy_storage(parent, phase):
    """Complete inherited futures plus every new boundary-reference proof budget."""
    require(phase in ("entry", "dispatch", "preflight"), "Unknown full precision storage phase")
    parent = Path(parent)
    allowed = {ROOT / (label("segment", arm, offset) + "/payload/study/case-0/references/latest-reference.json.tmp")
               for arm in DISPATCH_ARMS for offset in ARMS[arm]["offsets"]}
    allowed.add(BASE / "nonchem-endpoint-continuation-09/boltz-segment-10000/payload/study/case-0/references/latest-reference.json.tmp")
    for number in range(1, 4):
        attempt = parent / ("whole-precision-" + phase + "-" + str(number)); attempt.mkdir(exist_ok=False)
        try:
            capacity = _module(SOURCE / "scripts/recovery_capacity.py", "_whole_precision_capacity")
            prior = capacity.gate(attempt, phase)
            require(prior.get("status") == "passed" and prior.get("original_gate_changed") is False
                    and prior.get("reserve_bytes") == 4 * GIB and prior.get("file_reserve") == 100000,
                    "Every original/current future and reserve must pass unchanged")
            # Three Pro validators+six Pro entry/terminal references, one DB
            # validator+two DB entry/terminal references; no old scope removed.
            refs = 12 * 64 * 1024**2
            own = {str(path): owned_bytes(path) for path in (ROOT, SOURCE.parent,
                BASE / "nonchem-precision-continuation-14", BASE / "nonchem-precision-release-14")}
            excess, details = projection_excess_from_gate(prior)
            result = {"status": "failed", "inherited_gate": prior,
                      "additional_complete_reference_and_control_bytes": refs + CONTROL_CAP,
                      "additional_positive_pilot_projection_bytes": excess,
                      "new_pilot_projection_details": details,
                      "required_bytes": prior["required_bytes"] + refs + CONTROL_CAP + excess,
                      "available_bytes": prior["available_bytes"] - sum(own.values()),
                      "entire_new_tree_stale_charge": own, "reserve_bytes": 4 * GIB,
                      "file_reserve": 100000, "forecasts_removed": False,
                      "original_gate_changed": False, "reference_sessions": 12,
                      "reference_session_cap": 64 * 1024**2}
            try:
                require(metadata_bytes() <= CONTROL_CAP, "Full new control cap exceeded")
                require(result["available_bytes"] >= result["required_bytes"],
                        "Complete campaign plus every new reference proof does not fit")
                result["status"] = "passed"
            except BaseException as error:
                result["error"] = repr(error)
                raise
            finally:
                write(attempt / "full-precision-storage.json", result)
            write(parent / ("continuation-storage-" + phase + ".json"), result)
            return result
        except FileNotFoundError as error:
            if error.filename is None or Path(error.filename) not in allowed: raise
            write(attempt / "incomplete-owned-atomic-inventory.json", {
                "status": "failed_incomplete_inventory_not_admission", "path": error.filename,
                "partial_totals_used": False, "files_skipped": False, "attempt": number})
            if number == 3: raise


def storage(parent, phase):
    """New user-authorized rolling admission; historical guard remains intact."""
    capacity = _module(SOURCE / "scripts/rolling_capacity.py", "_rolling_precision_capacity")
    receipt = capacity.storage(sys.modules[__name__], parent, phase)
    require(receipt.get("status") == "passed", "Rolling segment capacity failed")
    return receipt


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
    require(cpu_xml_proofs(manifest)[0] == value["cases"], "CPU raw cases changed")
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
    command = [PYTHON, str(SOURCE / "scripts/precision_continuation_driver.py"), "--plan", str(SOURCE / "plan.json"),
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
    module = _module(SOURCE / "scripts/precision_continuation_driver.py", "_authenticated_endpoint_startup_driver")
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
            command = [PYTHON, str(SOURCE / "scripts/precision_continuation_driver.py"), "--plan", str(SOURCE / "plan.json"),
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
    references = cpu_reference_startups(plan, pin, payload, deadline)
    write(Path(payload) / "actual-startup-checks.json", {"status": "passed_metadata_only", "checks": reports,
                                                         "worker_checks": workers,
                                                         "reference_checks": references,
                                                         "preserved_failed_admissions": PRESERVED_FAILED_ADMISSIONS,
                                                         "native_readiness_established": False,
                                                         "fresh_prospective_readiness_deferred_until_native_validation": True})
    return reports


def reference_startup_command(plan, arm, payload):
    hooks = _module(SOURCE / "scripts/precision_continuation_hooks.py", "_precision_reference_startup_hooks")
    config = plan["reference_configuration"]["arms"][arm]
    command = hooks.expanded_command(config, sha(SOURCE / "plan.json"))
    output = Path(payload) / (arm + "-reference-startup.json")
    require(str(SOURCE / ("scripts/protenix_reference_worker.py" if arm == "protenix" else "scripts/dbmol_reference.py"))
            in command and "--output" not in command and "--startup-only" not in command,
            "Actual new reference startup source/command differs")
    return command + ["--startup-only", "--startup-output", str(output)], output


def verify_reference_startup(receipt, arm, pin, plan_sha):
    require(arm in DISPATCH_ARMS and receipt.get("status") == "passed_metadata_only"
            and receipt.get("numerical_setup_entered") is False
            and type(receipt.get("optimizer_updates")) is int and receipt["optimizer_updates"] == 0
            and isinstance(receipt.get("job_id"), str) and receipt["job_id"].isdigit(),
            "Reference actual startup entered numerical evaluation or has no owner")
    if arm == "protenix":
        require(receipt.get("own_source_manifest_sha256") == pin and receipt.get("plan_sha256") == plan_sha,
                "Actual Pro reference startup plan/source pin differs")
    return receipt


def cpu_reference_startups(plan, pin, payload, deadline):
    reports = []
    for arm in DISPATCH_ARMS:
        command, output = reference_startup_command(plan, arm, payload)
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Allocated CPU reference startup budget exhausted")
        environment = dict(os.environ)
        environment.update(plan["reference_configuration"]["arms"][arm]["environment"])
        # Metadata startup imports neither numerical runtime nor models. Native
        # GPU placement is deferred, while precision flags remain declared.
        environment.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        run = subprocess.run(command, capture_output=True, text=True, timeout=min(180, remaining), env=environment)
        invocation = Path(payload) / (arm + "-reference-startup-invocation.json")
        write(invocation, {"argv": command, "returncode": run.returncode, "stdout": run.stdout,
                           "stderr": run.stderr, "numerical_evaluations": 0})
        require(run.returncode == 0, "Actual reference metadata startup failed")
        receipt = verify_reference_startup(read(output), arm, pin, sha(SOURCE / "plan.json"))
        require(receipt["job_id"] == os.environ.get("LSB_JOBID"), "Reference startup job differs")
        reports.append({"arm": arm, "receipt": binding(output), "invocation": binding(invocation)})
    return reports


def reference_startup_proofs(startups, plan, pin):
    reports = startups.get("reference_checks")
    require(isinstance(reports, list) and len(reports) == 2
            and {r.get("arm") for r in reports} == set(DISPATCH_ARMS), "Both actual new reference startups required")
    proofs = []
    for row in reports:
        receipt, invocation = check_binding(row["receipt"]), check_binding(row["invocation"])
        actual = verify_reference_startup(read(receipt["path"]), row["arm"], pin, sha(SOURCE / "plan.json"))
        require(actual["job_id"] == read(ROOT / "cpu/submission.json")["job_id"], "Reference startup belongs to another CPU job")
        expected, output = reference_startup_command(plan, row["arm"], ROOT / "cpu/payload")
        observed = read(invocation["path"])
        require(receipt["path"] == str(output) and observed.get("argv") == expected
                and observed.get("returncode") == 0 and observed.get("numerical_evaluations") == 0,
                "Actual reference startup invocation/output differs")
        proofs += [receipt, invocation]
    return proofs


def cpu_worker_startups(plan, pin, payload, deadline):
    """Run real old worker import/help paths without loading models or updating."""
    reports = []
    for arm in ("dbmol",):
        offset = ARMS[arm]["offsets"][0]
        row = arm_plan(plan, arm)
        snapshot = input_snapshot(plan, arm, offset)
        directory = Path(payload) / ("worker-startup-" + arm)
        directory.mkdir(exist_ok=False)
        output = directory / "study"
        command = [PYTHON, str(SOURCE / "scripts/precision_continuation_driver.py"), "--plan", str(SOURCE / "plan.json"),
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
            and startups.get("preserved_failed_admissions") == PRESERVED_FAILED_ADMISSIONS
            and len(startups.get("checks", [])) == 4
            and {(row["arm"], row["phase"]) for row in startups.get("checks", [])}
                == {(arm, phase) for arm in DISPATCH_ARMS for phase in ("validate", "segment")}
            and len(startups.get("worker_checks", [])) == 1
            and {row["arm"] for row in startups.get("worker_checks", [])} == {"dbmol"},
            "Actual four CLI and DB worker startup commands not established")
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
    module = _module(SOURCE / "scripts/precision_continuation_driver.py", "_authenticated_endpoint_final_startup_driver")
    expected = module.build_original_argv(plan, arm, "segment", offset, output, snapshot, ready)
    command = [PYTHON, str(SOURCE / "scripts/precision_continuation_driver.py"), "--plan", str(SOURCE / "plan.json"),
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
            and result.get("endpoint_policy") == "precision-reference-continuation-policy-v1"
            and result.get("scientific_source_manifest_sha256") == row["source"]["manifest_sha256"],
            "Prospective policy/original failure/source scope differs")
    logit_records=result.get('logit_records')
    require(isinstance(logit_records,list) and (arm!='boltz' or bool(logit_records)),
            'Mandatory Boltz pilot logit qualification records missing')
    for item in logit_records:
        require(item.get('status')=='passed' and item.get('version')=='prospective-fixed-reference-logit-admission-v1',
                'New pilot logit measurement is not qualified')
        proof=check_binding(item['report']);check_binding(item['original_coarse_report'])
        require(Path(proof['path']).is_relative_to(parent/'payload/study'),'Logit proof outside own scientific producer')
        observed=read(proof['path'])
        require(observed.get('version')==item['version'] and observed.get('status')=='passed'
                and observed.get('can_admit_new_child') is True and observed.get('frozen_unchanged') is True
                and observed.get('can_certify_original_study') is False
                and observed.get('original_failure_preserved') is True
                and observed.get('selection_before_perturbed_forwards') is True
                and isinstance(observed.get('directions'),list)
                and {r.get('field') for r in observed['directions']}=={'atoms','bonds'}
                and all(r.get('pair_passed') is True and r.get('finer_resolved_disagreements')==[] for r in observed['directions']),
                'Complete independently measured pilot logit proof missing')
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
        check_generated_proof(proof)
    require(result.get("input_snapshot") == input_snapshot(plan, arm, offset), "Scientific input endpoint differs")
    adapter = _module(SOURCE / "scripts/precision_continuation_driver.py", "_precision_result_verifier")
    adapter.verify_precision_result(result, plan, pin, phase, arm, offset, input_snapshot(plan, arm, offset))
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
        check_generated_proof(proof)
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
    proofs += cuda_probe_proofs(parent, pin, plan, arm)
    proofs += [binding(parent / "payload/assessment.json"), binding(parent / "post-execution-preservation.json"), *result["proofs"]]
    value = {"status": "passed", "arm": arm, "offset": offset, "global_end": result["global_end"],
             "source_manifest_sha256": pin, "plan_sha256": sha(SOURCE / "plan.json"),
             "original_scientific_failure_preserved": True, "original_gate_replaced": False,
             "proofs": proofs}
    if phase == "segment":
        value["terminal_snapshot"] = result["terminal_snapshot"]
    write(ROOT / (parent.name + "-certificate.json"), value)
    return result


def cuda_probe_proofs(parent, pin, plan, arm):
    """A CPU startup pass never substitutes for real concurrent CUDA clients."""
    parent = Path(parent)
    source = _module(SOURCE / "scripts/cuda_ownership.py", "_precision_cuda_probe_verify")
    value = read(parent / "cuda-ownership/assessment.json")
    submitted = read(parent / "submission.json")
    identities = {"source_manifest": binding(SOURCE / "source_manifest.json"),
                  "plan": binding(SOURCE / "plan.json"),
                  "context_identity": plan["qualified_admissions"][arm]["assessment"]}
    require(value.get("version") == source.VERSION and value.get("status") == "passed_two_client_cuda_ownership"
            and value.get("job_id") == submitted["job_id"] and value.get("error") is None
            and value.get("input_bindings") == identities == value.get("input_bindings_after")
            and value.get("submission") == binding(parent / "submission.json")
            and value.get("optimizer_updates") == 0 and value.get("native_or_gradient_gate_replaced") is False
            and value.get("scientific_results_changed") is False, "Actual two-client CUDA smoke did not pass")
    require(len(value.get("cleanup", [])) == 2 and all(row.get("reaped") is True
            and row.get("forced_termination") is False and row.get("error") is None for row in value["cleanup"]),
            "CUDA smoke process cleanup incomplete")
    allocation = value["allocation"]
    routing = source.resolve_visibility(value.get("allocation_environment_before_resolution", {}))
    require(value.get("visibility_resolution") == routing
            and routing["selected_cuda_visible_devices"] == allocation.get("cuda_visible_devices"),
            "Actual scheduler CUDA visibility translation is not authenticated")
    require(allocation.get("job_id") == submitted["job_id"] and allocation.get("queue") == "gpul40s"
            and allocation.get("cpu_slots") == 4 and re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", allocation.get("cuda_visible_devices", "")),
            "Actual CUDA probe allocation differs")
    source._submission(parent / "submission.json", allocation, identities)
    invocation = read(parent / "cuda-ownership/invocation.json")
    nonce = invocation["nonce"]
    require(invocation.get("allocation") == allocation and invocation.get("input_bindings") == identities
            and invocation.get("gpu_request") == source.GPU_REQUEST, "CUDA probe invocation differs")
    require(set(value.get("records", {})) == set(source.ROLES), "Both actual CUDA probe roles required")
    for role in source.ROLES:
        rows = value["records"][role]
        require(len(rows) == 3, "Both CUDA repeat measurements required")
        for sequence, item in enumerate(rows):
            source._validate_message(item, allocation, role, nonce, sequence, rows[0] if sequence else None)
    require(value["records"][source.ROLES[0]][0]["pid"] != value["records"][source.ROLES[1]][0]["pid"],
            "Distinct concurrently measured CUDA children required")
    return [binding(p) for p in sorted((parent / "cuda-ownership").iterdir()) if p.is_file()]


def run_child(command, timeout, parent, manifest, plan, pin):
    """Supervise only the owned child group; recheck live capacity periodically."""
    record = {"execution_status": "unknown", "preservation_status": "failed", "source_manifest_sha256": pin,
              "original_results_changed": None, "argv": command}
    child = None
    try:
        require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
                "Finite positive native/segment child deadline required")
        deadline = time.monotonic() + timeout
        child = subprocess.Popen(command, start_new_session=True)
        record["owned_process_group"] = child.pid
        next_gate = time.monotonic() + 300
        while child.poll() is None:
            require(time.monotonic() < deadline, "Allocated native/segment child deadline exhausted")
            if time.monotonic() >= next_gate:
                gate = storage(parent, "periodic")
                require(gate.get("status") == "passed", "Periodic bounded output admission failed")
                require(time.monotonic() < deadline, "Periodic metadata exhausted child deadline")
                next_gate = time.monotonic() + 300
            time.sleep(min(5, max(.01, deadline - time.monotonic())))
        require(time.monotonic() < deadline, "Native/segment child completed outside its internal deadline")
        record.update(execution_status="returned", returncode=child.returncode)
        return subprocess.CompletedProcess(command, child.returncode)
    except BaseException as error:
        record["execution_error"] = repr(error)
        raise
    finally:
        try:
            if child is not None and child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try: os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    child.wait(timeout=15)
                record["owned_process_group_reaped"] = True
            verify(SOURCE, pin)
            external(manifest, plan)
            record.update(preservation_status="passed", original_results_changed=False)
        except BaseException as error:
            record["preservation_error"] = repr(error)
            raise
        finally:
            write(Path(parent) / "post-execution-preservation.json", record)


def execute(pin, phase, arm="all", offset=0):
    started = time.monotonic()
    manifest = verify(SOURCE, pin)
    plan = validate_plan(read(SOURCE / "plan.json"))
    require(manifest.get("version") == VERSION and manifest["files"].get("plan.json") == sha(SOURCE / "plan.json"), "Plan not bound by own manifest")
    submitted = own(phase, pin, arm, offset)
    budget = segment_budget(arm, offset) if phase == "segment" else None
    if budget is not None:
        require(submitted.get("runtime_budget") == budget, "Actual measured runtime receipt differs")
    phase_deadline = started + (3300 if phase == "cpu" else 6900 if phase == "validate" else budget["limit_seconds"] if phase == "segment" else 1080)
    parent = parent_path(phase, arm, offset)
    adoption = external(manifest, plan)
    storage(parent, "entry")
    if phase == "cpu":
        deadline = phase_deadline
        payload = parent / "payload"
        payload.mkdir(exist_ok=False)
        startups = cpu_startups(plan, pin, payload, deadline)
        cases = cpu_fixtures(manifest, payload, deadline)
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
        probe = _module(SOURCE / "scripts/cuda_ownership.py", "_precision_cuda_ownership")
        probe.run_probe(parent / "cuda-ownership", submission=parent / "submission.json",
                        expected_job_id=os.environ["LSB_JOBID"],
                        input_bindings={"source_manifest": binding(SOURCE / "source_manifest.json"),
                                        "plan": binding(SOURCE / "plan.json"),
                                        "context_identity": plan["qualified_admissions"][arm]["assessment"]},
                        deadline=phase_deadline)
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
            cases, fixture_proofs = cpu_xml_proofs(manifest)
            assessment = read(ROOT / "cpu/payload/assessment.json")
            require(assessment.get("status") == "passed" and assessment.get("cases") == cases
                    and assessment.get("source_manifest_sha256") == pin, "Allocated CPU assessment differs")
            startups = read(ROOT / "cpu/payload/actual-startup-checks.json")
            proofs += startup_proofs(startups, plan)
            proofs += reference_startup_proofs(startups, plan, pin)
            proofs += fixture_proofs
            require(assessment.get("actual_startup_checks") == startups["checks"]
                    and assessment.get("actual_worker_startup_checks") == startups["worker_checks"], "Startup check summary differs")
            proofs += [binding(ROOT / "cpu/payload" / name) for name in ("tests.xml", "assessment.json", "actual-startup-checks.json")]
            write(ROOT / "cpu-certificate.json", {"status": "passed", "cases": cases, "source_manifest_sha256": pin,
                                                    "plan_sha256": sha(SOURCE / "plan.json"), "proofs": proofs,
                                                    "original_scientific_cpu_replaced": False})
            for item in AUTOMATIC_ARMS:
                require(not parent_path("validate", item, ARMS[item]["offsets"][0]).exists(), "Existing/unknown validation must not be duplicated")
            require(time.monotonic() < phase_deadline, "CPU callback metadata runtime exhausted")
            for item in AUTOMATIC_ARMS:
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
