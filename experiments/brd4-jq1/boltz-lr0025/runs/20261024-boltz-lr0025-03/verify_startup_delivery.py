"""Independent stdlib delivery/CPU/startup/submission audit for both new runs.

This operator is outside the sealed releases. It never imports a controller,
driver, array library or model, and never connects to HPC. Startup success is
classified separately from native scientific readiness and optimization.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = "/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01"
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
WORKER = "/work3/s234818/latentmol/env/boltz311/bin/python"
PREPARED = "/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared"
CONTEXT = "a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093"
POCKET = "9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43"
REFERENCE = "37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb"
REVISION = "c74c659344123937608077e3e4078f75f09cd743"
STARTUP = "mosaic-actual-generated-command-premodel-v1"
SENTINEL = "PremodelBoundaryReached: Intentional startup-only boundary; no native model loaded"
HELPER = "51236d7bb18e25e52f2407ecc2efc196aead972ba9694003c78339461934be77"
FIXTURE = "176b280261d77b60f812ca2c0e7c0828ea92672a506c1eaba364cdab1ed0d4a1"

CAMPAIGNS = {
    "baseline": {
        "tag": "20261024-boltz-lr0025-03", "root": BASE + "/boltz-lr0025-03",
        "source": BASE + "/boltz-lr0025-release-03/source",
        "manifest": "c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6",
        "archive_sha": "d77d298b7cd2df41b4fb97fb22bf241a623c0d9825edab89226c54fa90b4b9ba",
        "archive_bytes": 39628961, "source_files": 635, "cpu": "29805913", "callback": "29805914",
        "cpu_parent": "cpu", "callback_parent": "after-cpu", "suite": "boltz-two-arm-serial30k-v1",
        "segment": 10000, "arms": ["boltz2", "boltz2_chemistry"],
        "groups": {"hpc-contract": 467, "legacy": 239, "native-mapping": 34, "chemistry": 59, "startup": 28},
    },
    "dbmol": {
        "tag": "20261024-boltz-dbmol-pilot-02", "root": BASE + "/boltz-dbmol-pilot-02",
        "source": BASE + "/boltz-dbmol-pilot-release-02/source",
        "manifest": "600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed",
        "archive_sha": "592a4b3d7d95909111013dacc95d4a1301b6e875fb124267ad08af7ca402381e",
        "archive_bytes": 39676528, "source_files": 657, "cpu": "29805915", "callback": "29805916",
        "cpu_parent": "cpu-05", "callback_parent": "after-cpu-05", "suite": "mosaic-dbmol-pilot-native-components-v1",
        "segment": 2500, "arms": ["boltz2_dbmol"],
        "groups": {"hpc-contract": 500, "legacy": 239, "native-mapping": 34, "chemistry": 59,
                   "dbmol-native": 83, "dbmol-transport": 148, "launcher-adoption": 72, "startup": 28},
        "launcher": BASE + "/boltz-dbmol-storage-release-05/source",
        "launcher_tag": "20261024-boltz-dbmol-storage-05",
        "launcher_manifest": "865303de06fc0a0bdf9297857c5bbc78c6ccd9ee05177e372a51869adc599bc3",
        "launcher_archive_sha": "d3690757d45422550aca1317491903a446a13ccedb3209621886933229b24586",
        "launcher_archive_bytes": 46951,
    },
}
EXTERNAL = {
    PREPARED + "/context.pt": (None, CONTEXT),
    BASE + "/restart-lrreset-04/run-boltz2/payload/study/summary.json":
        (301971481, "06a68008a0dfb8b6a66668447bcda4f01c7b54427d9b68f0d84e5926db278cc1"),
    BASE + "/restart-lrreset-04/run-protenix/payload/study/summary.json":
        (298580732, "b94d4bae58b7c82b92f4cf0b6b0e9804734afff0f54ea0b162b65d825ec2661d"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def safe_relative(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name,
            "Unsafe relative evidence/archive member: " + name)


def load_seal(tag, archive_sha, archive_bytes, manifest_sha, count):
    archive = REPO / "results/hpc" / tag / "source01/source.tar.gz"
    require(archive.stat().st_size == archive_bytes and digest(archive.read_bytes()) == archive_sha, "Pinned full source archive changed: " + tag)
    files = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            safe_relative(member.name)
            require(member.isfile() and member.name.startswith("source/"), "Nonregular/foreign sealed member")
            name = member.name.removeprefix("source/")
            require(name not in files, "Duplicate sealed member")
            files[name] = tar.extractfile(member).read()
    raw = files.pop("source_manifest.json")
    require(digest(raw) == manifest_sha, "Pinned manifest member changed")
    manifest = json.loads(raw)
    require(len(files) == count and set(files) == set(manifest["files"]), "Incomplete source member scope")
    for name, data in files.items():
        require(digest(data) == manifest["files"][name], "Source member hash differs: " + name)
    return manifest, {**files, "source_manifest.json": raw}


class Evidence:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.metadata = read(self.folder / "metadata.json")
        self.files, self.external, self.archive_files = {}, {}, {}
        locals_seen = set()
        for remote, record in self.metadata["files"].items():
            # Current collector format: remote keys and numbered local paths.
            local = record["local_path"]
            safe_relative(local)
            require(remote.startswith("/") and local not in locals_seen, "Duplicate/invalid delivered file")
            locals_seen.add(local)
            path = self.folder / local
            require(not path.is_symlink() and path.resolve().is_relative_to(self.folder), "Delivered path escapes extraction root")
            data = path.read_bytes()
            require(len(data) == record["bytes"] and digest(data) == record["sha256"], "Delivered bytes differ: " + remote)
            self.files[remote] = (record, data)
        for remote, record in self.metadata.get("external_files", {}).items():
            require(remote in EXTERNAL and record["local_bytes_downloaded"] is False, "Undeclared metadata-only external file")
            size, checksum = EXTERNAL[remote]
            require(type(record["bytes"]) is int and record["bytes"] > 0 and (size is None or record["bytes"] == size)
                    and record["sha256"] == checksum, "External known input identity differs")
            self.external[remote] = record

    def add_archive(self, root, files):
        for name, data in files.items():
            self.archive_files[root + "/" + name] = data
            if root + "/" + name in self.files:
                require(self.files[root + "/" + name][1] == data, "Remote-delivered source differs from pinned archive")

    def has(self, remote):
        return remote in self.files

    def bytes(self, remote):
        require(remote in self.files or remote in self.archive_files, "Required small proof not downloaded: " + remote)
        return self.files[remote][1] if remote in self.files else self.archive_files[remote]

    def json(self, remote):
        return json.loads(self.bytes(remote))

    def text(self, remote):
        return self.bytes(remote).decode("utf-8")

    def checksum(self, remote):
        if remote in self.external:
            return self.external[remote]["sha256"]
        return digest(self.bytes(remote))

    def item(self, remote):
        if remote in self.external:
            return {"path": remote, "bytes": self.external[remote]["bytes"], "sha256": self.checksum(remote)}
        return {"path": remote, "bytes": len(self.bytes(remote)), "sha256": self.checksum(remote)}

    def proofs(self, proofs):
        seen = set()
        for proof in proofs:
            path = proof["path"]
            require(path not in seen and self.checksum(path) == proof["sha256"], "Omitted/duplicate/mismatched certificate proof: " + path)
            seen.add(path)
        return seen


def scheduler(raw, job, queue, status=None):
    compact = re.sub(r"\s+", "", raw)
    require(re.findall(r"Job<([^>]+)>", compact) == [job] and re.findall(r"Queue<([^>]+)>", compact) == [queue], "Wrong/ambiguous actual scheduler job/queue")
    states = re.findall(r"Status<([^>]+)>", compact)
    require(len(states) == 1 and (status is None or states == [status]), "Wrong actual scheduler state")
    return states[0]


def lsf(raw, job, status="Done"):
    require(re.findall(r"^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$", raw, re.M) == [(job, status)], "Wrong/ambiguous same-job LSF outcome")
    require(("Successfully completed." in raw and "Exited with exit" not in raw) if status == "Done"
            else re.search(r"Exited with exit code\s+\d+", raw), "Missing actual LSF terminal footer")


def resources(phase):
    if phase == "cpu":
        return ["-q", "hpc", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=4GB]", "-W", "01:00"]
    if phase in ("native", "segment"):
        return ["-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]", "-gpu", "num=1:mode=exclusive_process", "-W", "02:00" if phase == "native" else "23:00"]
    require(phase in ("after-cpu", "after-native", "after-segment", "dispatch"), "Unknown submitted phase")
    return ["-q", "hpc", "-n", "2", "-R", "span[hosts=1]", "-R", "rusage[mem=4GB]", "-W", "00:30"]


def vector(c, label, phase, dependency):
    parent = c["root"] + "/" + label
    prefix = "mosaic-dbmol-" if "launcher" in c else "mosaic-bol30k-"
    args = ["bsub", *resources(phase), "-J", prefix + label, "-oo", parent + "/stdout.%J.log", "-eo", parent + "/stderr.%J.log"]
    if dependency:
        args += ["-w", dependency]
    if "launcher" in c:
        wrapper = c["launcher"] + "/scripts/hpc/" + ("dbmol_storage_cpu.lsf" if phase == "cpu" else "dbmol_storage_stage.lsf")
        args += ["bash", "-l", wrapper, c["launcher"], c["launcher_manifest"], c["source"], c["root"], c["manifest"]]
    else:
        wrapper = c["source"] + "/scripts/hpc/" + ("boltz_serial_cpu.lsf" if phase == "cpu" else "boltz_serial_stage.lsf")
        args += ["bash", "-l", wrapper, c["source"], c["root"], c["manifest"]]
    if phase != "cpu":
        args += [phase, parent]
    return args


def submissions(e, c):
    jobs = e.json(c["root"] + "/jobs.json")
    require(jobs.get(c["cpu_parent"]) == c["cpu"] and jobs.get(c["callback_parent"]) == c["callback"], "Actual CPU/callback IDs differ")
    states, records = {}, {}
    for label, job in jobs.items():
        path = c["root"] + "/" + label + "/submission.json"
        sub = e.json(path)
        phase, dependency = sub["phase"], sub["dependency"]
        require(sub["outcome"] == "submitted" and sub["job_id"] == job and sub["returncode"] == 0
                and re.findall(r"Job <(\d+)>", sub["stdout"]) == [job]
                and sub["args"] == vector(c, label, phase, dependency), "Actual submission/source/resource vector differs: " + label)
        if phase == "after-cpu":
            require(dependency == "ended(" + c["cpu"] + ")", "CPU callback own-ended dependency differs")
        elif phase in ("after-native", "after-segment"):
            producer = label.replace("-after-native-", "-native-").replace("-after-segment-", "-segment-")
            require(dependency == "ended(" + jobs[producer] + ")", "Producer callback dependency differs")
        elif phase in ("cpu", "native", "segment"):
            require(dependency is None, "Unexpected interarm or native dependency")
        raw = e.metadata.get("scheduler", {}).get(job)
        if raw and raw["returncode"] == 0:
            states[label] = scheduler(raw["stdout"] + raw["stderr"], job, "gpul40s" if phase in ("native", "segment") else "hpc")
        else:
            states[label] = "live_history_unavailable"
        records[label] = sub
    return jobs, states, records


def cpu(e, c, source_files, allow_incomplete):
    path = c["root"] + "/cpu-certificate.json"
    if not e.has(path):
        require(allow_incomplete, "Full CPU certificate not established")
        partial = {}
        for group in c["groups"]:
            xml = c["root"] + "/" + c["cpu_parent"] + "/payload/" + group + ".xml"
            if e.has(xml):
                tree = ET.fromstring(e.bytes(xml))
                partial[group] = {kind: len(list(tree.iter(kind))) for kind in ("testcase", "failure", "error", "skipped")}
        return {"status": "not_yet_certified", "cpu_passed": False, "partial_xml_only": partial}
    certificate = e.json(path)
    require(certificate["status"] == "passed" and certificate["suite"] == c["suite"]
            and certificate["source_manifest_sha256"] == c["manifest"] and certificate["groups"] == c["groups"]
            and certificate["cases"] == sum(c["groups"].values()), "Full CPU certificate scope differs")
    if "launcher" in c:
        require(certificate["launcher_manifest_sha256"] == c["launcher_manifest"] and certificate["launcher_source"] == c["launcher"], "CPU launcher binding differs")
    e.proofs(certificate["proofs"])
    wrapper = source_files["scripts/hpc/boltz_serial_cpu.lsf"].decode() if "launcher" not in c else e.text(c["launcher"] + "/scripts/hpc/dbmol_storage_cpu.lsf")
    declared = set(re.findall(r"tests/(test_[A-Za-z0-9_]+)\.py", wrapper.split("contracts=(", 1)[1].split(")", 1)[0]))
    headers = {}
    for group, expected in c["groups"].items():
        xml = c["root"] + "/" + c["cpu_parent"] + "/payload/" + group + ".xml"
        tree = ET.fromstring(e.bytes(xml)); rows = list(tree.iter("testcase"))
        require(len(rows) == expected and not any(list(tree.iter(kind)) for kind in ("failure", "error", "skipped")), "Raw allocated CPU group failed/incomplete: " + group)
        for suite in tree.iter("testsuite"):
            require(all(int(suite.get(k, "0")) == 0 for k in ("failures", "errors", "skipped")), "Raw JUnit header reports an unrepresented failure/error/skip")
        log = e.text(xml.removesuffix(".xml") + ".log")
        require(re.search(r"\b" + str(expected) + r"\s+passed\b", log), "Actual XML rows/log count differ")
        found = {part for row in rows for part in row.get("classname", "").split(".")}
        required = declared if group == "hpc-contract" else {"test_premodel_startup"} if group == "startup" else {
            "dbmol-native": {"test_dbmol_objective", "test_dbmol_native_paths"},
            "dbmol-transport": {"test_dbmol_gradient_contract", "test_dbmol_bridge", "test_dbmol_pilot_contract"},
            "launcher-adoption": {"fixture_storage_adoption"},
        }.get(group, set())
        require(required <= found and (not required or all(row.get("classname") for row in rows)), "Missing required exact CPU module attribution: " + group)
        if group == "legacy":
            require(re.search(r"\b3\s+deselected\b", log), "Historical deliberate deselections changed")
        headers[group] = sum(int(suite.get("tests", "0")) for suite in tree.iter("testsuite"))
    finals = [proof for proof in certificate["proofs"] if re.fullmatch(r"scheduler-final-\d+\.txt", PurePosixPath(proof["path"]).name)
              and str(PurePosixPath(proof["path"]).parent) == c["root"] + "/" + c["cpu_parent"]]
    require(len(finals) == 1, "CPU final actual scheduler proof missing or ambiguous")
    scheduler(e.text(finals[0]["path"]), c["cpu"], "hpc", "DONE")
    lsf(e.text(c["root"] + "/" + c["cpu_parent"] + "/stdout." + c["cpu"] + ".log"), c["cpu"])
    payload = c["root"] + "/" + c["cpu_parent"] + "/payload/"
    require(e.checksum(payload + "composition_reference.json") == REFERENCE and e.checksum(payload + "pocket_definition.json") == POCKET, "CPU reference/pocket changed")
    assessment = e.json(payload + "assessment.json")
    require(assessment["status"] == "passed" and assessment["source_manifest_sha256"] == c["manifest"], "CPU assessment source/status differs")
    # The inherited assessor records only its own first four groups, while the
    # callback certificate covers the new objective/launcher/startup groups too.
    inherited = sum(c["groups"][name] for name in ("hpc-contract", "legacy", "native-mapping", "chemistry"))
    require(assessment["passed"] == inherited, "Inherited assessor actual raw case scope differs")
    return {"status": "full_cpu_evidence_verified", "cpu_passed": True, "cases": certificate["cases"],
            "groups": c["groups"], "raw_xml_header_counts": headers, "certificate_sha256": e.checksum(path),
            "count_policy": "Actual raw testcase rows and logs; inherited unittest subtest header differences are recorded separately"}


def argument(args, flag):
    require(args.count(flag) == 1 and args.index(flag) + 1 < len(args), "Missing/duplicate startup driver option: " + flag)
    return args[args.index(flag) + 1]


def startup(e, c, arm, phase, index, submissions_map, manifest_files):
    if "launcher" in c:
        path = c["root"] + "/startup-" + phase + "-" + format(index, "04d") + ".json"
        if not e.has(path):
            return None
        outer = e.json(path); result = outer["result"]
        require(outer["status"] == "startup_only_passed" and outer["source_manifest_sha256"] == c["manifest"]
                and outer["launcher_manifest_sha256"] == c["launcher_manifest"] and outer["phase"] == phase and outer["index"] == index, "DBMol startup outer identity differs")
        owner_path = outer["owner"]["path"]; owner_id = outer["owner"]["job_id"]
        require(e.checksum(owner_path) == outer["owner"]["sha256"] and outer["cpu_sha256"] == e.checksum(c["root"] + "/cpu-certificate.json"), "Startup owner/CPU proof changed")
    else:
        stem = c["root"] + "/startup-" + arm + "-" + phase + "-" + format(index, "04d")
        path = stem + ".json"
        if not e.has(path):
            return None
        result = e.json(path); certificate = e.json(stem + "-certificate.json")
        require(certificate["status"] == "passed" and certificate["version"] == STARTUP and certificate["source_manifest_sha256"] == c["manifest"]
                and certificate["arm"] == arm and certificate["phase"] == phase and certificate["index"] == index
                and certificate["model_loaded"] is False and certificate["scientific_readiness_passed"] is False, "Baseline startup certificate identity differs")
        owner_path = certificate["owner_parent"] + "/submission.json"; owner_id = certificate["owner_job_id"]
        proof_paths = e.proofs(certificate["proofs"])
        require(result["inputs"] == certificate["inputs"], "Startup certificate input map differs")
        expected_proofs = {path, owner_path, c["source"] + "/source_manifest.json", c["source"] + "/scripts/premodel_startup.py"}
        expected_proofs |= {item["path"] for item in result["inputs"].values()}
        require(proof_paths == expected_proofs, "Startup certificate proof set incomplete or changed")
    require(result["version"] == STARTUP and result["status"] == "startup_only_passed" and result["source_manifest_sha256"] == c["manifest"]
            and result["source"] == c["source"] and result["git_revision_reference"] == REVISION and result["job_id"] == owner_id
            and result["arm"] == arm and result["phase"] == phase and result["index"] == index, "Actual startup result identity differs")
    require(all(result[key] is False for key in ("model_loaded", "gradient_passed", "scientific_readiness_passed", "saved_results_changed"))
            and result["optimizer_updates"] == 0, "Startup result mislabels scientific readiness/optimization")
    owner = e.json(owner_path); label = PurePosixPath(owner_path).parent.name
    require(owner["job_id"] == owner_id and owner["outcome"] == "submitted" and owner["phase"] in ("after-cpu", "after-native", "after-segment", "dispatch")
            and owner["args"] == vector(c, label, owner["phase"], owner["dependency"]) and submissions_map[label] == owner, "Startup owner actual callback identity/resources differ")
    inputs = result["inputs"]
    expected_paths = {"config": c["source"] + ("/examples/configs/dbmol_pilot.json" if "launcher" in c else "/examples/configs/boltz_serial_" + arm + ".json"),
                      "cpu-certificate": c["root"] + "/cpu-certificate.json",
                      "reference": c["root"] + "/" + c["cpu_parent"] + "/payload/composition_reference.json",
                      "pocket": c["root"] + "/" + c["cpu_parent"] + "/payload/pocket_definition.json", "prepared_context": PREPARED + "/context.pt"}
    producer = "pilot-native-" + format(index, "04d") if "launcher" in c else arm + "-native-" + format(index, "04d")
    if phase == "segment":
        expected_paths["ready"] = c["root"] + "/" + producer + "/payload/study/assessment.json"
    snapshot = None
    if index:
        previous = "pilot-segment-" + format(index - 1, "04d") if "launcher" in c else arm + "-segment-" + format(index - 1, "04d")
        previous_certificate = e.json(c["root"] + "/" + previous + "-certificate.json")
        e.proofs(previous_certificate["proofs"])
        snapshot = previous_certificate["terminal_snapshot"]
        expected_paths["resume-snapshot"] = snapshot["path"]
    require(set(inputs) == set(expected_paths), "Startup input inventory applicability differs")
    for key, expected in expected_paths.items():
        require(inputs[key] == e.item(expected), "Startup input path/bytes/hash differs: " + key)
    if "launcher" in c:
        require(outer["input_snapshot"] == snapshot, "DBMol exact startup predecessor differs")
        require(outer["ready"] == ({"path": expected_paths["ready"], "sha256": inputs["ready"]["sha256"]} if phase == "segment" else None), "DBMol exact startup readiness differs")
    args = result["args"]; output = argument(args, "--output")
    out = PurePosixPath(output)
    require(out.is_absolute() and ".." not in out.parts and out.name == "study" and out.parent.name == "payload"
            and out.parent.parent.name == "command" and not output.startswith(BASE + "/"), "Startup output is not disposable node command output")
    driver = c["source"] + ("/scripts/dbmol_driver.py" if "launcher" in c else "/examples/brd4_25_atoms.py")
    expected_args = [PYTHON, driver, "--mode", "validate" if phase == "native" else "run", "--config", inputs["config"]["path"], "--output", output,
                     "--manifest", c["manifest"], "--cpu-certificate", inputs["cpu-certificate"]["path"], "--cpu-certificate-sha256", inputs["cpu-certificate"]["sha256"],
                     "--reference", inputs["reference"]["path"], "--pocket", inputs["pocket"]["path"], "--pocket-sha256", POCKET, "--prepared", PREPARED,
                     "--context-sha256", CONTEXT, "--worker-python", WORKER, "--offset", str(index * c["segment"]), "--segment-steps", str(c["segment"])]
    if index:
        expected_args += ["--resume-snapshot", inputs["resume-snapshot"]["path"], "--resume-sha256", inputs["resume-snapshot"]["sha256"]]
    if phase == "segment":
        expected_args += ["--ready", inputs["ready"]["path"], "--ready-sha256", inputs["ready"]["sha256"]]
    require(args == expected_args, "Actual generated startup driver command differs")
    summary = result["summary"]
    require(summary["status"] == "failed" and summary["error"] == SENTINEL and summary["source_manifest_sha256"] == c["manifest"]
            and not summary.get("native_gradient_checks") and not summary.get("cases"), "Startup sentinel was mislabeled or not reached")
    require(result["summary_sha256"] == digest((json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()), "Embedded actual driver summary bytes differ")
    execution = {name: checksum for name, checksum in manifest_files.items() if name == "pyproject.toml" or
                 (PurePosixPath(name).parts[0] in ("src", "tests", "examples", "scripts", "experiments")
                  and PurePosixPath(name).suffix in (".py", ".json", ".lsf", ".sh", ".toml", ".txt") and "__pycache__" not in PurePosixPath(name).parts)}
    require(summary["source"]["git_revision_reference"] == REVISION and summary["source"]["files"] == execution
            and summary["source"]["execution_content_sha256"] == digest(json.dumps(execution, sort_keys=True, separators=(",", ":")).encode()), "Real driver provenance/file map differs")
    environment = summary["environment"]
    require(environment["jax_enable_x64"] is ("launcher" in c) and environment["devices"] and all(device["platform"] == "cpu" for device in environment["devices"]), "Actual startup JAX precision/backend differs")
    boundary = result["boundary"]
    worker_args = [WORKER, "-u", c["source"] + "/scripts/dbmol_worker.py"] if "launcher" in c else [WORKER, "-u", "-m", "mosaic.small_molecule.boltz2.worker"]
    worker_args += ["--context", PREPARED + "/context.pt", "--context-sha256", CONTEXT, "--checkpoint", "/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt"]
    if "launcher" in c:
        worker_args += ["--structure-checkpoint", "/work3/s234818/latentmol/cache/boltz2/boltz2_conf.ckpt"]
    require(boundary["command"] == worker_args and boundary["stderr"] == output + "/worker.log", "Actual constructor sentinel boundary differs")
    return {"path": path, "sha256": e.checksum(path), "arm": arm, "phase": phase, "index": index, "owner_cpu_job_id": owner_id,
            "status": "startup_only_passed", "native_gradient_passed": False, "optimization_started": False}


def verify(kind, folder, allow_incomplete=False, require_dispatch=False):
    c = CAMPAIGNS[kind]; e = Evidence(folder)
    manifest, source_files = load_seal(c["tag"], c["archive_sha"], c["archive_bytes"], c["manifest"], c["source_files"])
    require(manifest["git_revision_reference"] == REVISION, "Authenticated release revision missing or changed")
    e.add_archive(c["source"], source_files)
    require(e.has(c["source"] + "/source_manifest.json"), "Actual remote science manifest was not collected")
    if "launcher" in c:
        launcher_manifest, launcher_files = load_seal(c["launcher_tag"], c["launcher_archive_sha"], c["launcher_archive_bytes"], c["launcher_manifest"], 10)
        require(launcher_manifest["scientific_source_manifest_sha256"] == c["manifest"], "Launcher scientific adoption differs")
        e.add_archive(c["launcher"], launcher_files)
        require(e.has(c["launcher"] + "/source_manifest.json"), "Actual remote launcher manifest not collected")
        helper_files = launcher_files
    else:
        helper_files = source_files
    require(digest(helper_files["scripts/premodel_startup.py"]) == HELPER and digest(helper_files["tests/test_premodel_startup.py"]) == FIXTURE, "Exact shared startup controls changed")
    staging = e.json(c["root"] + "/source-staging.json")
    require(staging["status"] == "passed" and staging["readonly"] is True, "Actual remote staging/read-only seal absent")
    if "launcher" in c:
        require(staging["science_manifest_sha256"] == c["manifest"] and staging["launcher_manifest_sha256"] == c["launcher_manifest"], "Actual dual-source staging differs")
    else:
        require(staging["manifest"] == c["manifest"] and staging["files"] == 635 and staging["archive_sha256"] == c["archive_sha"] and staging["archive_bytes"] == c["archive_bytes"], "Actual full science staging differs")
    jobs, states, records = submissions(e, c)
    cpu_result = cpu(e, c, source_files, allow_incomplete)
    callback_log = c["root"] + "/" + c["callback_parent"] + "/stdout." + c["callback"] + ".log"
    callback_done = False
    if e.has(callback_log):
        text = e.text(callback_log)
        if re.search(r"^Subject:\s*Job\s+" + c["callback"] + r":.*\bDone\s*$", text, re.M):
            lsf(text, c["callback"]); callback_done = True
    startup_results = []
    for arm in c["arms"]:
        for phase in ("native", "segment"):
            for index in range(4 if "launcher" in c else 3):
                result = startup(e, c, arm, phase, index, records, manifest["files"])
                if result:
                    require(cpu_result["cpu_passed"], "Startup cannot certify before actual CPU success")
                    startup_results.append(result)
    native_submitted = {}
    for label, sub in records.items():
        if sub["phase"] not in ("native", "segment"):
            continue
        match = re.fullmatch(r"pilot-(native|segment)-(\d{4})", label) if "launcher" in c else re.fullmatch(r"(boltz2(?:_chemistry)?)-(native|segment)-(\d{4})", label)
        require(match is not None, "Undeclared actual scientific parent")
        arm = "boltz2_dbmol" if "launcher" in c else match[1]
        phase = match[1] if "launcher" in c else match[2]
        index = int(match[2] if "launcher" in c else match[3])
        found = [r for r in startup_results if (r["arm"], r["phase"], r["index"]) == (arm, phase, index)]
        require(len(found) == 1, "Actual scientific submission lacks its exact startup proof")
        if "launcher" in c:
            require(sub["startup_smoke"] == {"path": found[0]["path"], "sha256": found[0]["sha256"]}, "Actual DBMol submission startup pin differs")
        native_submitted[label] = {"job_id": sub["job_id"], "scheduler_state": states[label], "phase": phase,
                                   "scientific_readiness": "not assessed by this metadata audit"}
    if require_dispatch:
        require(cpu_result["cpu_passed"] and callback_done, "Completed CPU dispatch callback not authenticated")
        for arm in c["arms"]:
            label = "pilot-native-0000" if "launcher" in c else arm + "-native-0000"
            require(label in native_submitted, "Required initial native validator not actually submitted")
    gates = {}
    for path in e.files:
        if path.startswith(c["root"] + "/") and path.endswith("/gate.json"):
            gate = e.json(path)
            require(gate["reserve_bytes"] == 4294967296 and gate["file_reserve"] == 100000, "Storage reserves changed")
            gates[path] = gate["status"]
            if gate["status"] == "passed":
                work = gate.get("work3", gate)
                require(work["available_bytes_conservative"] >= work["required_bytes"] and work["available_files_lower"] >= 100000, "Passed byte/file gate incoherent")
    return {"status": "cpu_startup_delivery_verified" if cpu_result["cpu_passed"] else "incomplete_cpu_delivery_verified",
            "kind": kind, "source_manifest_sha256": c["manifest"], "source_members_verified": c["source_files"],
            "cpu": cpu_result, "callback_done_lsf_authenticated": callback_done, "jobs": jobs, "scheduler_snapshot": states,
            "startup_only_results": startup_results, "actual_native_or_segment_submissions": native_submitted,
            "storage_gates": gates, "downloaded_small_files_verified": len(e.files),
            "metadata_sha256": digest((Path(folder) / "metadata.json").read_bytes()),
            "external_files_metadata_only": e.external,
            "scope": "Independent stdlib byte/JSON/XML/LSF/source/resource/startup verification only. Original big summaries/context are remote-hash references, not downloaded bytes. Startup success is not native gradient readiness or optimization. No local numerical/model execution."}


def catalog(kind):
    c = CAMPAIGNS[kind]
    root = c["root"]
    paths = [root + "/" + name for name in ("jobs.json", "source-staging.json", "cpu-certificate.json")]
    for label, job in ((c["cpu_parent"], c["cpu"]), (c["callback_parent"], c["callback"])):
        paths += [root + "/" + label + "/submission.json", root + "/" + label + "/stdout." + job + ".log"]
    for group in c["groups"]:
        paths += [root + "/" + c["cpu_parent"] + "/payload/" + group + suffix for suffix in (".xml", ".log")]
    paths += [root + "/" + c["cpu_parent"] + "/payload/" + name for name in ("assessment.json", "oracles.json", "composition_reference.json", "pocket_definition.json")]
    paths += [c["source"] + "/source_manifest.json"]
    if "launcher" in c:
        paths += [c["launcher"] + "/source_manifest.json", root + "/startup-native-0000.json"]
    else:
        paths += [root + "/startup-" + arm + "-native-0000" + suffix for arm in c["arms"] for suffix in (".json", "-certificate.json")]
    return {"required_initial_paths_when_present": paths, "dynamic_small_proofs": "Collect every CPU/startup certificate proofs entry, startup owner submission, source/launcher executing members, actual jobs submissions and raw scheduler/LSF; preserve exact remote path.",
            "external_hash_inventory_only": EXTERNAL, "metadata_schema": {"files": "remote_path -> {local_path,bytes,sha256}", "external_files": "remote_path -> {bytes,sha256,local_bytes_downloaded:false}", "scheduler": "actual_job_id -> {returncode,stdout,stderr}"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(CAMPAIGNS), required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-dispatch", action="store_true")
    args = parser.parse_args()
    if args.catalog:
        print(json.dumps(catalog(args.kind), indent=2)); return
    require(args.bundle is not None and args.output is not None and not args.output.exists(), "Provide fresh bundle/output paths; preserve earlier audits")
    result = verify(args.kind, args.bundle, args.allow_incomplete, args.require_dispatch)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
