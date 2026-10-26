"""Stdlib controller for separate full-model validation pilots, never production.

Every phase authenticates its immutable source, sealed plans and original
inputs before importing adapters. Scientific execution occurs only in allocated
children. Existing studies/jobs/settings are read-only inputs. Unknown submission
markers prohibit replay; scheduler allocation or an ended dependency is not a
scientific or CPU success certificate.
"""

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


VERSION = "chemistry-full-model-source-v5"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
ROOT = BASE / "chemistry-full-model-05"
SOURCE = BASE / "chemistry-full-model-release-05/source"
ADMISSION = BASE / "nonchem-stride20-startup-release-03/source"
ADMISSION_SHA = "704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab"
SCIENCE = BASE / "protenix-extended-release-03/source"
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
LIMIT = 2 * 1024**3
PRIOR_OWNED_ROOTS = tuple(BASE / name for name in (
    "chemistry-full-model-02", "chemistry-full-model-release-02",
    "chemistry-full-model-03", "chemistry-full-model-release-03",
    "chemistry-replay-01", "chemistry-replay-release-01",
    "chemistry-replay-02", "chemistry-replay-release-02",
    "chemistry-full-model-04", "chemistry-full-model-release-04"))
CURRENT_ADMISSION = BASE / "protenix-endpoint-fix-release-02/source"
CURRENT_ADMISSION_SHA = "09f759a7befb8a7b8f0e1d203e15acf939126fba493b30e6a2bf9cd2eea2216b"
PHASES = ("cpu", "after-cpu", "protenix", "boltz")
CPU_MODULES = (
    "test_speed", "test_hybrid", "test_gpu_matrix", "test_oracle_profile",
    "test_full_model_pilot", "test_gpu_local_bridge", "test_full_model_ops",
    "test_full_model_replay", "test_gpu_process_admission", "test_full_model_retry",
    "test_replay_context_qualification", "test_full_model_context",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    """Exclusive receipt write; an existing result is never overwritten."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _canonical(path):
    require(isinstance(path, str) and "\\" not in path, "Require canonical POSIX input path")
    parsed = PurePosixPath(path)
    require(parsed.is_absolute() and ".." not in parsed.parts, "Require absolute input path without traversal")
    return Path(path)


def _regular(path):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "Require regular non-symlink input: " + str(path))
    return info


def _binding(binding, *, length_required=True):
    require(isinstance(binding, dict), "Missing explicit file binding")
    path = _canonical(binding.get("path"))
    require(re.fullmatch("[a-f0-9]{64}", binding.get("sha256", "")) is not None, "Malformed file SHA")
    info = _regular(path)
    if length_required or "bytes" in binding:
        require(type(binding.get("bytes")) is int and binding["bytes"] > 0
                and info.st_size == binding["bytes"], "Original input length changed: " + str(path))
    require(sha(path) == binding["sha256"], "Original input SHA changed: " + str(path))
    return {"path": str(path), "bytes": info.st_size, "sha256": binding["sha256"]}


def verify(source, pin):
    """Authenticate every manifest member without importing scientific code."""
    source = Path(source)
    require(not source.is_symlink() and source.is_dir(), "Require real immutable source root")
    _regular(source / "source_manifest.json")
    require(re.fullmatch("[a-f0-9]{64}", pin or "") is not None, "Malformed manifest pin")
    require(sha(source / "source_manifest.json") == pin, "Source manifest changed")
    manifest = json.loads((source / "source_manifest.json").read_text())
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Missing exact source members")
    for name, digest in files.items():
        relative = PurePosixPath(name)
        require(isinstance(name, str) and not relative.is_absolute() and ".." not in relative.parts
                and "\\" not in name and re.fullmatch("[a-f0-9]{64}", digest or "") is not None,
                "Unsafe source member")
        path = source.joinpath(*relative.parts)
        _regular(path)
        require(all(not parent.is_symlink() for parent in path.parents if parent != source.parent),
                "Source member parent is a symlink")
        require(sha(path) == digest, "Source member changed: " + name)
    return manifest


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load authenticated metadata module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plans(manifest):
    contract = _module(SOURCE / "scripts/full_model_contract.py", "_authenticated_full_model_contract")
    plans = {}
    for arm in ("protenix", "boltz"):
        relative = "plans/" + arm + ".json"
        require(relative in manifest["files"], "Missing sealed per-arm plan")
        path = SOURCE / relative
        require(sha(path) == manifest["files"][relative], "Sealed plan changed")
        plan = json.loads(path.read_text())
        contract.validate_plan(plan)
        require(plan["profile"] == {"protenix": "protenix-random50k", "boltz": "boltz-chem30k"}[arm],
                "Per-arm plan profile changed")
        _binding(plan["driver"])
        for binding in plan["inputs"].values():
            _binding(binding)
        config = json.loads(Path(plan["inputs"]["config"]["path"]).read_text())
        contract.validate_config(config, plan["profile"])
        plans[arm] = plan
    return plans


def _lsf_success(text, job, name, queue="hpc"):
    require(type(job) is str and re.fullmatch(r"[1-9]\d*", job) is not None, "Invalid exact job ID")
    require(isinstance(text, str), "Raw LSF proof must be text")
    compact = "".join(text.split())
    subjects = re.findall(
        r"Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)",
        text,
    )
    subjects = [tuple("".join(field.split()) for field in row) for row in subjects]
    require(compact.count("Subject:Job") == 1 and subjects == [(job, name, "dcc", "Done")],
            "Require unique same-job/name DONE LSF subject")
    require(re.findall(r"inqueue<([^>]+)>", compact) == [queue]
            and compact.count("Successfullycompleted.") == 1, "Require original same-job LSF queue/success")
    return True


def xml_cases(path, expected):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    require(cases and all(not list(case) for case in cases), "Raw JUnit has missing cases/failure/error/skip children")
    require(len({(case.get("classname"), case.get("name")) for case in cases}) == len(cases),
            "Duplicate raw JUnit identities")
    actual = dict(Counter(case.get("classname") for case in cases))
    require(actual == expected and all(type(count) is int and count > 0 for count in expected.values()),
            "Raw JUnit module attribution/count mismatch: " + repr((actual, expected)))
    return actual


def previous_failures(manifest, bindings):
    expected={"protenix":"29811883","boltz":"29811884"}
    require(manifest.get("failed_predecessors")==expected,"Failed predecessor scope changed")
    for arm,job in expected.items():
        parent=BASE/"chemistry-full-model-01"/arm
        assessment=parent/"payload/assessment.json";log=parent/("stdout."+job+".log")
        require(str(assessment) in bindings and str(log) in bindings,"Missing exact terminal predecessor bindings")
        row=json.loads(assessment.read_text());raw=log.read_text();compact="".join(raw.split())
        require(row.get("status")=="failed" and row.get("job_id")==job
                and row.get("source_manifest_sha256")=="347e753d481ec3d584f3b0b05cc2801ee745036ee17638d7c3672fc2ca2cba16"
                and row.get("production_authorized") is False,"Predecessor was not the original failed validation")
        require(compact.count("Subject:Job")==1
                and re.findall(r"Subject:Job(\d+):<([^>]+)>incluster<([^>]+)>Exited",compact)
                   ==[(job,"mosaic-chem-full01-"+arm,"dcc")]
                and re.findall(r"inqueue<([^>]+)>",compact)==["gpul40s"]
                and re.findall(r"Exitedwithexitcode(\d+)\.",compact)==["1"],
                "Retained previous pilot has no authentic terminal failure")
    return expected


def context_api():
    from types import SimpleNamespace
    return SimpleNamespace(ROOT=ROOT, SOURCE=SOURCE, require=require, verify=verify, sha=sha,
                           _module=_module, _binding=_binding, _file_binding=_file_binding,
                           _lsf_success=_lsf_success)


def context_tools():
    return _module(SOURCE / "scripts/full_model_context.py", "_full_model_execution_context")


def external(manifest):
    """Rehash unchanged sources, original readiness/results and every input."""
    require(manifest.get("version") == VERSION, "Wrong full-model source schema")
    parents = manifest.get("parent_sources")
    require(isinstance(parents, list) and parents, "Missing original source bindings")
    seen = set()
    for row in parents:
        path = _canonical(row["path"])
        require(str(path) not in seen, "Duplicate parent source binding")
        seen.add(str(path))
        old = verify(path, row["manifest_sha256"])
        require(type(row["files"]) is int and len(old["files"]) == row["files"], "Parent source count changed")
    preserved = manifest.get("preserved_proofs")
    require(isinstance(preserved, list) and preserved, "Missing preserved result bindings")
    by_path = {}
    recursively_verified = 0
    for binding in preserved:
        actual = _binding(binding)
        require(actual["path"] not in by_path, "Duplicate preserved proof")
        by_path[actual["path"]] = actual
        if actual["path"].endswith("restart-lrreset-04/validate-protenix/payload/study/assessment.json"):
            assessment = json.loads(Path(actual["path"]).read_text())
            require(assessment.get("status") == "passed" and isinstance(assessment.get("proofs"), list),
                    "Original saved Protenix context/readiness assessment differs")
            for item in assessment["proofs"]:
                _binding(item, length_required=False)
                recursively_verified += 1
    old_root = BASE / "chemistry-search-speed-02"
    for phase, job, queue, name, cases, benchmark_status in (
        ("cpu", "29809477", "hpc", "mosaic-chem-speed02-cpu", 143,
         "passed_exact_hybrid_saved_input_comparison"),
        ("gpu", "29809554", "gpul40s", "mosaic-chem-speed02-gpu", 19,
         "passed_separate_gpu_matrix_benchmark"),
    ):
        parent = old_root / phase
        paths = [parent / "payload/assessment.json", parent / "payload/tests.xml",
                 parent / "payload/benchmark/assessment.json", parent / ("stdout." + job + ".log")]
        require(all(str(path) in by_path for path in paths), "Missing immutable previous benchmark proof pins")
        assessment = json.loads(paths[0].read_text())
        counts = assessment.get("cpu_counts" if phase == "cpu" else "gpu_counts")
        require(assessment.get("status") == "passed" and assessment.get("job_id") == job
                and isinstance(counts, dict) and sum(counts.values()) == cases,
                "Previous allocated CPU/GPU benchmark assessment changed")
        xml_cases(paths[1], counts)
        require(sha(paths[1]) == assessment.get("raw_xml_sha256"), "Previous raw XML binding changed")
        require(sha(paths[2]) == assessment.get("benchmark_sha256")
                and json.loads(paths[2].read_text()).get("status") == benchmark_status,
                "Previous saved-input benchmark differs")
        _lsf_success(paths[3].read_text(), job, name, queue)
    failed=previous_failures(manifest,by_path)
    plans = _plans(manifest)
    context_tools().producer(context_api(), manifest)
    return {"replay_context_producer": manifest["replay_context_producer"], "parent_sources": len(parents), "preserved_proofs": len(preserved),
            "recursive_original_proofs": recursively_verified,
            "plans": {arm: sha(SOURCE / "plans" / (arm + ".json")) for arm in plans},
            "failed_predecessors_preserved":failed,
            "original_results_changed": False}


def own_bytes():
    """Bound the entire replacement group; retry whole scans on atomic turnover.

    A vanished temporary file is not a zero-byte file. Rewalk every root up to
    three times, preserving the hard cap and rejecting continued instability.
    No file is ignored, removed or moved by this metadata accounting helper.
    """
    roots = [ROOT, SOURCE.parent, *PRIOR_OWNED_ROOTS]
    roots = [p for p in roots if not any(p != q and p.is_relative_to(q) for q in roots)]
    for attempt in range(3):
        total=0
        try:
            for root in roots:
                if not root.exists():continue
                require(not root.is_symlink() and root.is_dir(),"Owned output root is not a real directory")
                for directory,directories,files in os.walk(root,followlinks=False):
                    for name in directories+files:
                        path=Path(directory)/name;info=path.lstat()
                        if stat.S_ISLNK(info.st_mode):
                            resolved=path.resolve(strict=True)
                            require(any(resolved.is_relative_to(item.resolve()) for item in roots if item.exists()),"Owned output contains external alias")
                            total+=info.st_size
                        elif stat.S_ISREG(info.st_mode):total+=info.st_size
                        else:require(stat.S_ISDIR(info.st_mode),"Owned output contains a special file")
            return total
        except FileNotFoundError:
            if attempt==2:raise
            time.sleep(.01)


def storage(parent, phase):
    """Retain the current whole-campaign guard and its reserved chemistry cap.

    The pinned continuation guard already funds2GiB of chemistry-validation
    future growth, in addition to all native forecasts and newer precision
    validation. This sole replacement validation uses that existing allowance;
    no production chemistry growth is admitted here. Prior failed pilot jobs
    are terminal and their entire retained bytes remain in the containing BASE
    stale-quota charge. Nothing is deleted or discounted to obtain admission.
    """
    prior_manifest=verify(CURRENT_ADMISSION,CURRENT_ADMISSION_SHA)
    require(len(prior_manifest["files"])==14,"Current whole-campaign guard changed")
    prior_module=_module(CURRENT_ADMISSION / "scripts/protenix_precision_ops.py", "_full_model_current_admission")
    kind="preflight" if phase=="preflight" else "dispatch" if phase=="two-gpu-dispatch" else "entry"
    prior=prior_module.storage(parent,kind)
    retained=prior["parent_gate"]
    record={"status":"failed","whole_current_gate":prior,
        "additional_allowance_bytes":0,"reserved_chemistry_validation_bytes":LIMIT,
        "required_bytes":prior["required_bytes"],"available_bytes":prior["available_bytes"],
        "reserve_bytes":4*1024**3,"file_reserve":100000,"original_gate_changed":False,
        "scope":"Current whole native/report/chemistry/diagnostic/control/precision guard; uses its existing2GiB validation allowance; no chemistry production admission"}
    try:
        require(prior["status"]=="passed" and retained["status"]=="passed"
                and prior["reserve_bytes"]==4*1024**3 and prior["file_reserve"]==100000,
                "Current whole-campaign guard/reserves did not pass")
        require(retained["additional_allowance_bytes"]==1024**3+LIMIT+768*1024**2+512*1024**2,
                "Current original chemistry-validation allowance is absent")
        record["new_owned_bytes"]=own_bytes()
        require(record["new_owned_bytes"]<LIMIT,"Replacement validation source/output cap exceeded")
        require(record["available_bytes"]>=record["required_bytes"],"Current whole-campaign storage does not fit")
        record["status"]="passed"
    finally:
        write(Path(parent)/("full-model-storage-"+phase+".json"),record)
    return record


def resources(phase):
    require(phase in PHASES, "Undeclared full-model phase")
    common = ["-R", "span[hosts=1]"]
    if phase == "cpu":
        return ["-q", "hpc", "-n", "16", *common, "-R", "rusage[mem=2GB]", "-W", "02:00"]
    if phase == "after-cpu":
        return ["-q", "hpc", "-n", "1", *common, "-R", "rusage[mem=2GB]", "-W", "00:20"]
    return ["-q", "gpul40s", "-n", "8", *common, "-R", "rusage[mem=4GB]",
            "-gpu", ("num=1:mode=shared:j_exclusive=yes:mps=no" if phase=="boltz" else "num=1:mode=exclusive_process"), "-W", "02:00"]


def argv(phase, pin, dependency=None):
    require(re.fullmatch("[a-f0-9]{64}", pin or "") is not None, "Invalid actual submission source pin")
    parent = ROOT / phase
    command = ["bsub", *resources(phase), "-J", "mosaic-chem-full05-" + phase,
               "-oo", str(parent / "stdout.%J.log"), "-eo", str(parent / "stderr.%J.log")]
    if dependency is not None:
        require((phase == "after-cpu" and re.fullmatch(r"ended\([1-9]\d*\)", dependency) is not None)
                or (phase == "cpu" and dependency == "ended(29818513)"),
                "Only own-ended CPU callback or exact replay-ended CPU may have a dependency")
        command += ["-w", dependency]
    else:
        require(phase != "after-cpu", "CPU callback requires its exact own-ended dependency")
    return command + ["bash", "-l", str(SOURCE / "scripts/full_model_stage.lsf"), str(SOURCE), pin, phase]


def submit(phase, pin, dependency=None):
    """Create an exclusive unknown marker before bsub; never retry an outcome."""
    parent = ROOT / phase
    parent.mkdir(exist_ok=False)
    command = argv(phase, pin, dependency)
    record = {"status": "unknown", "phase": phase, "argv": command, "dependency": dependency,
              "manifest_sha256": pin, "submitted_utc": datetime.now(timezone.utc).isoformat()}
    if phase in ("protenix", "boltz"):
        certificate = ROOT / "cpu-certificate.json"
        proof = json.loads(certificate.read_text())
        require(proof.get("status") == "passed" and proof.get("source_manifest_sha256") == pin,
                "New allocated CPU certificate missing before GPU submission")
        record["cpu_certificate_sha256"] = sha(certificate)
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
    require(record["status"] == "submitted", "Unknown submission outcome: preserve marker and never duplicate")
    import fcntl
    with (ROOT / "jobs.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / "jobs.json"
        jobs = json.loads(path.read_text()) if path.exists() else {}
        require(phase not in jobs, "Existing actual phase job cannot be replaced")
        jobs[phase] = record["job_id"]
        temporary = ROOT / ("jobs-" + phase + ".partial")
        write(temporary, jobs)
        temporary.replace(path)
    result = subprocess.run(["bjobs", "-l", record["job_id"]], capture_output=True, text=True, timeout=30)
    write(parent / "scheduler-at-submission.json",
          {"job_id": record["job_id"], "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    return record


def own(phase, pin):
    path = ROOT / phase / "submission.json"
    for _ in range(20):
        if path.exists():
            break
        time.sleep(1)
    record = json.loads(path.read_text())
    require(record.get("status") == "submitted" and record.get("job_id") == os.environ.get("LSB_JOBID"),
            "Phase does not belong to actual current submission")
    require(record.get("argv") == argv(phase, pin, record.get("dependency"))
            and record.get("manifest_sha256") == pin, "Actual phase argv/source differs")
    require(os.environ.get("LSB_QUEUE") == ("gpul40s" if phase in ("protenix", "boltz") else "hpc"),
            "Actual allocated queue differs")
    require(os.environ.get("LSB_DJOB_NUMPROC") == {"cpu": "16", "after-cpu": "1", "protenix": "8", "boltz": "8"}[phase],
            "Actual allocated CPU slots differ")
    if phase == "after-cpu":
        predecessor = json.loads((ROOT / "cpu/submission.json").read_text())["job_id"]
        require(record["dependency"] == "ended(" + predecessor + ")", "Callback dependency is not its own actual CPU job")
    if phase in ("protenix", "boltz"):
        require(record.get("cpu_certificate_sha256") == sha(ROOT / "cpu-certificate.json"),
                "GPU submission's exact new CPU certificate changed")
    return record


def _terminal_status(raw, job, name):
    require(isinstance(raw, dict) and type(raw.get("returncode")) is int and raw["returncode"] == 0
            and isinstance(raw.get("stdout"), str) and isinstance(raw.get("stderr"), str),
            "Actual terminal query failed; no state inferred")
    text = "".join(raw["stdout"].split())
    require(re.findall(r"Job<(\d+)>", text) == [job]
            and re.findall(r"JobName<([^>]+)>", text) == [name]
            and re.findall(r"Queue<([^>]+)>", text) == ["hpc"], "Require unique actual same CPU job/name/queue")
    states = re.findall(r"Status<([^>]+)>", text)
    require(len(states) == 1 and states[0] in ("DONE", "RUN", "PEND"), "Actual CPU state is neither success nor bounded wait")
    return states[0]


def wait_cpu_terminal(query, job, *, record, pause=lambda: time.sleep(5), maximum=13):
    require(type(maximum) is int and 1 <= maximum <= 13, "Invalid bounded terminal query count")
    for index in range(maximum):
        raw = query(job)
        record(index, raw)
        if _terminal_status(raw, job, "mosaic-chem-full05-cpu") == "DONE":
            return raw
        if index + 1 < maximum:
            pause()
    raise TimeoutError("Bounded actual scheduler queries did not establish CPU DONE")


def cpu_terminal(pin, manifest):
    parent = ROOT / "after-cpu"
    submission = json.loads((ROOT / "cpu/submission.json").read_text())
    require(submission["status"] == "submitted" and submission["manifest_sha256"] == pin
            and submission["argv"] == argv("cpu", pin), "Original CPU submission differs")
    job = submission["job_id"]

    def query(job_id):
        result = subprocess.run(["bjobs", "-l", job_id], capture_output=True, text=True, timeout=30)
        return {"job_id": job_id, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    scheduler = wait_cpu_terminal(query, job, record=lambda index, raw:
                                  write(parent / ("cpu-terminal-query-" + str(index).zfill(3) + ".json"), raw))
    path = ROOT / "cpu" / ("stdout." + job + ".log")
    for _ in range(20):
        if path.exists() and "Successfully completed." in path.read_text():
            break
        time.sleep(1)
    _lsf_success(path.read_text(), job, "mosaic-chem-full05-cpu")
    assessment_path = ROOT / "cpu/payload/assessment.json"
    assessment = json.loads(assessment_path.read_text())
    require(assessment.get("status") == "passed" and assessment.get("job_id") == job
            and assessment.get("source_manifest_sha256") == pin
            and assessment.get("cpu_counts") == manifest["cpu_counts"], "CPU assessment is not an exact allocated pass")
    xml = ROOT / "cpu/payload/tests.xml"
    xml_cases(xml, manifest["cpu_counts"])
    require(sha(xml) == assessment["raw_xml_sha256"], "Original CPU XML changed")
    return {"job_id": job, "assessment_sha256": sha(assessment_path), "raw_xml_sha256": sha(xml),
            "lsf_sha256": sha(path), "scheduler": scheduler,
            "cpu_cases": sum(manifest["cpu_counts"].values())}


def _file_binding(path):
    info = _regular(path)
    return {"path": str(path), "bytes": info.st_size, "sha256": sha(path)}


def certify_cpu(pin, manifest, terminal):
    """Freeze actual allocated terminal proofs before either independent GPU."""
    job = terminal["job_id"]
    queries = sorted((ROOT / "after-cpu").glob("cpu-terminal-query-*.json"))
    require(queries, "CPU certificate needs original recorded scheduler queries")
    result = {
        "version": "chemistry-full-model-cpu-certificate-v1", "status": "passed",
        "source_manifest_sha256": pin, "job_id": job, "cpu_counts": manifest["cpu_counts"],
        "cpu_submission": _file_binding(ROOT / "cpu/submission.json"),
        "cpu_assessment": _file_binding(ROOT / "cpu/payload/assessment.json"),
        "cpu_xml": _file_binding(ROOT / "cpu/payload/tests.xml"),
        "cpu_lsf": _file_binding(ROOT / "cpu" / ("stdout." + job + ".log")),
        "scheduler_queries": [_file_binding(path) for path in queries],
        "plans": {arm: manifest["files"]["plans/" + arm + ".json"] for arm in ("protenix", "boltz")},
        "scope": "Actual same-job CPU DONE/LSF/raw XML; not a native-model or GPU pass",
    }
    write(ROOT / "cpu-certificate.json", result)
    return verify_cpu_certificate(pin, manifest, sha(ROOT / "cpu-certificate.json"))


def verify_cpu_certificate(pin, manifest, expected_sha):
    """Reauthenticate persisted actual terminal proofs, including aged-out jobs.

    This never constructs a DONE record or substitutes scheduler absence for
    failure/success. Every original raw query and LSF byte is certificate-bound;
    the actual GPU submission separately pins the entire certificate digest.
    """
    path = ROOT / "cpu-certificate.json"
    require(sha(path) == expected_sha, "New allocated CPU certificate SHA differs")
    certificate = json.loads(path.read_text())
    require(certificate.get("version") == "chemistry-full-model-cpu-certificate-v1"
            and certificate.get("status") == "passed" and certificate.get("source_manifest_sha256") == pin
            and certificate.get("cpu_counts") == manifest["cpu_counts"], "New CPU certificate identity differs")
    job = certificate.get("job_id")
    require(type(job) is str and re.fullmatch(r"[1-9]\d*", job) is not None, "New CPU certificate lacks exact job")
    expected_paths = {
        "cpu_submission": ROOT / "cpu/submission.json", "cpu_assessment": ROOT / "cpu/payload/assessment.json",
        "cpu_xml": ROOT / "cpu/payload/tests.xml", "cpu_lsf": ROOT / "cpu" / ("stdout." + job + ".log"),
    }
    for name, expected_path in expected_paths.items():
        require(certificate[name]["path"] == str(expected_path), "CPU certificate redirected an original proof")
        _binding(certificate[name])
    require(certificate.get("plans") == {
        arm: manifest["files"]["plans/" + arm + ".json"] for arm in ("protenix", "boltz")
    }, "CPU certificate's original plans changed")
    submission = json.loads(expected_paths["cpu_submission"].read_text())
    require(submission.get("status") == "submitted" and submission.get("job_id") == job
            and submission.get("argv") == argv("cpu", pin) and submission.get("manifest_sha256") == pin,
            "CPU certificate's actual submission differs")
    assessment = json.loads(expected_paths["cpu_assessment"].read_text())
    require(assessment.get("status") == "passed" and assessment.get("job_id") == job
            and assessment.get("source_manifest_sha256") == pin
            and assessment.get("cpu_counts") == manifest["cpu_counts"], "New allocated CPU assessment differs")
    xml_cases(expected_paths["cpu_xml"], manifest["cpu_counts"])
    require(assessment.get("raw_xml_sha256") == sha(expected_paths["cpu_xml"]), "New CPU assessment XML binding differs")
    _lsf_success(expected_paths["cpu_lsf"].read_text(), job, "mosaic-chem-full05-cpu")
    queries = certificate.get("scheduler_queries")
    require(isinstance(queries, list) and 1 <= len(queries) <= 13, "Missing bounded original scheduler evidence")
    observed = []
    for index, binding in enumerate(queries):
        expected_path = ROOT / "after-cpu" / ("cpu-terminal-query-" + str(index).zfill(3) + ".json")
        require(binding["path"] == str(expected_path), "Original CPU query identity/order changed")
        _binding(binding)
        observed.append(_terminal_status(json.loads(expected_path.read_text()), job, "mosaic-chem-full05-cpu"))
    require(observed[-1] == "DONE" and all(value in ("RUN", "PEND") for value in observed[:-1]),
            "Persisted actual CPU scheduler queries never established terminal success")
    return {"status": "passed", "sha256": expected_sha, "job_id": job,
            "cpu_cases": sum(manifest["cpu_counts"].values()), "actual_terminal_states": observed}


def _run_child(command, log, environment, *, timeout):
    """Bound one owned process group by walltime and joint persistent storage."""
    started = time.monotonic()
    with Path(log).open("xb") as stream:
        process = subprocess.Popen(command, cwd=SOURCE, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                require(own_bytes() < LIMIT, "Combined new source/output cap exceeded during child")
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Bounded allocated child walltime exceeded")
                time.sleep(2)
            require(process.returncode == 0, "Allocated child exited " + str(process.returncode))
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
    return {"argv": command, "returncode": process.returncode, "elapsed_seconds": time.monotonic() - started}


@contextmanager
def termination_guard():
    """Route CLI SIGTERM through owned-child teardown, then restore the handler.

    Scientific children have their own process group. Raising here lets
    `_run_child` terminate that exact group and allow the full pilot its bounded
    thirty-second descendant cleanup before forced termination.
    """
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError("Allocated full-model controller received SIGTERM")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run(pin, phase):
    require(not sys.flags.optimize, "Optimized Python cannot bypass controller assertions")
    manifest = verify(SOURCE, pin)
    require(phase in PHASES and manifest.get("version") == VERSION, "Undeclared phase/source")
    counts = manifest.get("cpu_counts")
    require(isinstance(counts, dict) and set(counts) == {"tests." + name for name in CPU_MODULES}
            and all(type(value) is int and value > 0 for value in counts.values()), "Exact dynamic CPU module counts missing")
    submission = own(phase, pin)
    before = external(manifest)
    parent = ROOT / phase
    if phase == "after-cpu":
        import fcntl
        with (ROOT / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            require(manifest.get("active_gpu_arms")==["protenix"],"Retry05 cannot duplicate Boltz02 validation")
            require(not any((ROOT / arm).exists() for arm in ("protenix", "boltz")),
                    "Existing/unknown GPU outcome prohibits dispatch replay")
            cpu = cpu_terminal(pin, manifest)
            certificate = certify_cpu(pin, manifest, cpu)
            qualified, qualified_binding = context_tools().adopt_cpu(context_api(), manifest)
            admission = storage(parent, "two-gpu-dispatch")
            actual = {arm: submit(arm, pin) for arm in ("protenix",)}
            after = external(manifest)
            require(after == before, "Original external bindings changed during dispatch")
            write(parent / "gpu-submissions-complete.json", {
                "status": "submitted_not_passed", "cpu_terminal": cpu, "storage": admission,
                "new_cpu_certificate": certificate, "qualified_context": qualified_binding,
                "submissions": actual, "source_manifest_sha256": pin, "external_after": after,
            })
        return actual
    payload = parent / "payload"
    payload.mkdir(exist_ok=False)
    result = {"status": "failed", "phase": phase, "job_id": submission["job_id"],
              "source_manifest_sha256": pin, "external_before": before,
              "original_results_changed": False, "production_authorized": False}
    started = time.monotonic()
    try:
        if phase in ("protenix", "boltz"):
            result["new_cpu_certificate"] = verify_cpu_certificate(
                pin, manifest, submission["cpu_certificate_sha256"]
            )
        result["storage"] = storage(payload, phase + "-entry")
        plans = _plans(manifest)
        # The authenticated original launcher helper owns short node-local
        # TMPDIR and its guarded EXIT cleanup. Work3 output paths exceed Unix
        # socket limits and must not hold multiprocessing listener sockets.
        temporary = Path(os.environ.get("TMPDIR", ""))
        require(re.fullmatch(r"/tmp/mosaic-boltz\." + submission["job_id"] + r"\.[A-Za-z0-9]{6}",
                             str(temporary)) is not None
                and temporary.is_dir() and not temporary.is_symlink()
                and temporary.stat().st_uid == os.getuid(),
                "Require exact newly owned short node TMPDIR from authenticated original helper")
        result["owned_node_tmp"] = str(temporary)
        environment = dict(
            os.environ, PYTHONPATH=str(SCIENCE / "src"), PYTHONDONTWRITEBYTECODE="1",
            JAX_ENABLE_X64="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
            CUDA_CACHE_PATH=str(payload / "cuda-cache"), CUDA_CACHE_MAXSIZE=str(64 * 1024**2),
            XDG_CACHE_HOME=str(payload / "cache"), JAX_ENABLE_COMPILATION_CACHE="false",
            TMPDIR=str(temporary), MOSAIC_SPEED_PLAN=str(BASE / "chemistry-search-speed-02/plan.json"),
        )
        if phase == "cpu":
            environment.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
            command = [PYTHON, "-m", "pytest", "-q", "-o", "addopts=", "-o",
                       "cache_dir=" + str(payload / "pytest-cache"),
                       *[str(SOURCE / "tests" / (name + ".py")) for name in CPU_MODULES],
                       "--junitxml=" + str(payload / "tests.xml")]
            result["command"] = _run_child(command, payload / "tests.log", environment, timeout=2400)
            result["cpu_counts"] = xml_cases(payload / "tests.xml", counts)
            require(re.search(r"\b(?:[1-9]\d*)\s+deselected\b", (payload / "tests.log").read_text()) is None,
                    "Allocated full CPU suite unexpectedly deselected cases")
            result["cpu_cases"] = sum(counts.values())
            result["raw_xml_sha256"] = sha(payload / "tests.xml")
            qualified = context_tools().qualify_predecessor(context_api(), manifest)
            write(payload / "qualified-context.json", qualified)
            result["qualified_context"] = _file_binding(payload / "qualified-context.json")
        else:
            environment["JAX_PLATFORMS"] = "cuda,cpu"
            qualified, qualified_binding = context_tools().adopt_cpu(context_api(), manifest)
            require(phase == "protenix", "Context retry04 may not rerun Boltz")
            environment = context_tools().apply_environment(environment, qualified)
            result["qualified_context"] = qualified_binding
            result["execution_context_overrides"] = qualified["qualification"]["selected_environment"]
            command = [PYTHON, str(SOURCE / "scripts/full_model_pilot.py"), "--plan",
                       str(SOURCE / "plans" / (phase + ".json")), "--output", str(payload / "comparison")]
            result["command"] = _run_child(command, payload / "pilot.log", environment, timeout=6700)
            assessment = payload / "comparison/assessment.json"
            require(json.loads(assessment.read_text()).get("status") == "passed_full_model_comparison",
                    "Full-model comparison did not pass its original scientific/pilot gates")
            result["comparison_sha256"] = sha(assessment)
            result["plan_sha256"] = sha(SOURCE / "plans" / (phase + ".json"))
        result["external_after"] = external(manifest)
        require(result["external_after"] == before, "Original inputs/proofs changed across allocated child")
        verify(SOURCE, pin)
        result["status"] = "passed"
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["joint_owned_bytes"] = own_bytes()
        if result["joint_owned_bytes"] >= LIMIT:
            result.update(status="failed", cap_exceeded=True)
        write(payload / "assessment.json", result)
        require(result["joint_owned_bytes"] < LIMIT, "Joint persistent output cap exceeded")
    return result


if __name__ == "__main__":
    require(len(sys.argv) == 3, "Usage: full_model_ops.py MANIFEST PHASE")
    with termination_guard():
        run(sys.argv[1], sys.argv[2])
