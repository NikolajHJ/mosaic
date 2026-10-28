"""Independent stdlib byte/XML/LSF audit of precision recovery CPU delivery.

This verifier imports no experiment runtime, array package or numerical model.
It authenticates collected metadata and actual allocated CPU/callback evidence;
an authenticated native submission never becomes a native or optimizer pass.
"""
import argparse
import ast
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET


GIB = 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def loads(data):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON field: " + key)
            result[key] = value
        return result
    return json.loads(data, parse_constant=invalid, object_pairs_hook=unique)


def remote_path(value):
    require(isinstance(value, str) and "\\" not in value, "Malformed remote path")
    path = PurePosixPath(value)
    require(path.is_absolute() and ".." not in path.parts and str(path) == value,
            "Remote path must be canonical and absolute")
    return path


def local_source(source):
    source = Path(source)
    require(source.is_dir() and not source.is_symlink(), "Missing regular local source")
    manifest_bytes = (source / "source_manifest.json").read_bytes()
    manifest = loads(manifest_bytes)
    members = manifest.get("files")
    require(isinstance(members, dict) and members, "Source inventory absent")
    for name, digest in members.items():
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and ".." not in rel.parts and "\\" not in name
                and str(rel) == name and re.fullmatch(r"[0-9a-f]{64}", digest or ""),
                "Unsafe or malformed source member")
        path = source.joinpath(*rel.parts)
        require(path.is_file() and not path.is_symlink()
                and sha(path.read_bytes()) == digest, "Local source member changed: " + name)
    # Packaging sidecars source.tar.gz/preparation.json are outside the inventory.
    allowed = set(members) | {"source_manifest.json", "source.tar.gz", "preparation.json"}
    actual = set()
    for path in source.rglob("*"):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()),
                "Linked or special local source entry")
        if path.is_file():
            actual.add(path.relative_to(source).as_posix())
    require(actual <= allowed and set(members) <= actual, "Unexpected/missing local source member")
    require(manifest.get("original_results_changed") is False
            and manifest.get("optimization_settings_changed") is False
            and manifest.get("optimizer_updates") == 0, "Source preservation scope differs")
    return manifest, sha(manifest_bytes), loads((source / "plan.json").read_bytes())


class Packet:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.raw = loads(self.data)
        self.files = {}
        rows = self.raw.get("files")
        require(isinstance(rows, list) and rows, "Observation not ready: no collected files")
        for row in rows:
            name = str(remote_path(row.get("path")))
            require(("text" in row) != ("base64" in row), "Ambiguous observation encoding")
            value = row["text"].encode("utf8") if "text" in row else base64.b64decode(row["base64"], validate=True)
            require(name not in self.files and type(row.get("bytes")) is int
                    and len(value) == row["bytes"] and sha(value) == row.get("sha256"),
                    "Duplicate or changed collected bytes: " + name)
            self.files[name] = (row, value)

    def bytes(self, path):
        name = str(path)
        require(name in self.files, "Observation not ready: missing " + name)
        return self.files[name][1]

    def json(self, path):
        return loads(self.bytes(path))

    def proof(self, row):
        require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"},
                "Malformed certificate/startup file binding")
        value = self.bytes(remote_path(row["path"]))
        require(type(row["bytes"]) is int and len(value) == row["bytes"]
                and sha(value) == row["sha256"], "Certificate proof bytes differ")
        return value


def string_constant(source, member, name):
    tree = ast.parse((Path(source) / member).read_bytes())
    values = [node.value.value for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)
              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(values) == 1, "Authenticated source constant absent/ambiguous: " + name)
    return values[0]


def resources(phase):
    if phase == "gpu":
        return ["-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]",
                "-gpu", "num=1:mode=exclusive_process", "-W", "02:00"]
    return ["-q", "hpc", "-n", "4" if phase == "cpu" else "1", "-R", "span[hosts=1]",
            "-R", "rusage[mem=4GB]" if phase == "cpu" else "rusage[mem=2GB]",
            "-W", "01:00" if phase == "cpu" else "00:20"]


def submission(row, kind, phase, root, source, pin, cpu_job=None):
    job = row.get("job_id")
    require(isinstance(job, str) and re.fullmatch(r"[1-9][0-9]*", job), "Missing actual job ID")
    dependency = "ended(" + cpu_job + ")" if phase == "after-cpu" else None
    number = root.name.rsplit("-", 1)[-1]
    prefix = "mosaic-pro-precision" if kind == "protenix" else "mosaic-dbmol-reference"
    name = prefix + number + "-" + phase
    wrapper = "protenix_precision_stage.lsf" if kind == "protenix" else "dbmol_recovery_stage.lsf"
    parent = root / phase
    argv = ["bsub", *resources(phase), "-J", name, "-oo", str(parent / "stdout.%J.log"),
            "-eo", str(parent / "stderr.%J.log")]
    if dependency:
        argv += ["-w", dependency]
    argv += ["bash", "-l", str(source / "scripts" / wrapper), str(source), pin, phase]
    require(row.get("status") == "submitted" and row.get("returncode") == 0
            and row.get("phase") == phase and row.get("source_manifest_sha256") == pin
            and row.get("dependency") == dependency and row.get("argv") == argv,
            "Actual submission/source/resources/dependency differ: " + phase)
    require(re.findall(r"Job <([1-9][0-9]*)> is submitted to queue <([^>]+)>", row.get("stdout", ""))
            == [(job, "gpul40s" if phase == "gpu" else "hpc")], "Submission output does not establish unique job")
    return job, name


def lsf_success(data, job, name, queue):
    compact = "".join(data.decode("utf8").split())
    require(compact.count("Subject:Job") == 1 and re.findall(
        r"Subject:Job([1-9][0-9]*):<([^>]+)>incluster<([^>]+)>(Done|Exited)", compact)
        == [(job, name, "dcc", "Done")]
        and re.findall(r"inqueue<([^>]+)>", compact) == [queue]
        and compact.count("Successfullycompleted.") == 1 and "Exitedwithexit" not in compact,
        "Observation not ready: actual unique same-job LSF success missing for " + job)


def scheduler_state(row, job, name, queue):
    if not isinstance(row, dict):
        return None
    compact = "".join(row.get("stdout", "").split())
    blocks = re.findall(r"Job<([1-9][0-9]*)>,(.*?)(?=Job<[1-9][0-9]*>,|$)", compact, re.S)
    own = [block for ident, block in blocks if ident == job]
    require(len(own) <= 1, "Duplicate actual scheduler record")
    if not own:
        return None
    require(row.get("returncode") == 0 and re.findall(r"JobName<([^>]+)>", own[0]) == [name]
            and re.findall(r"Queue<([^>]+)>", own[0]) == [queue], "Scheduler owner/queue differs")
    states = re.findall(r"Status<([^>]+)>", own[0])
    require(len(states) == 1, "Missing/duplicate scheduler state")
    return states[0]


def terminal(packet, root, phase, job, name):
    lsf_path = root / phase / ("stdout." + job + ".log")
    lsf_success(packet.bytes(lsf_path), job, name, "hpc")
    observed = scheduler_state(packet.raw.get("scheduler"), job, name, "hpc")
    require(observed in (None, "DONE"), "Observation not ready: actual job " + job + " is " + str(observed))
    persisted = []
    for path in packet.files:
        if (path.startswith(str(root / "cpu") + "/terminal-query-")
                or path == str(root / "after-cpu/payload/cpu-final-scheduler.json")):
            value = packet.json(path)
            if scheduler_state(value, job, name, "hpc") == "DONE":
                persisted.append(path)
    # An aged-out live record is explicitly represented via original LSF, never
    # fabricated as a newly read scheduler DONE. CPU additionally has its own
    # certificate-bound actual final query; callbacks have their unique LSF.
    require(phase != "cpu" or observed == "DONE" or persisted,
            "Observation not ready: CPU actual final scheduler proof absent")
    return {"job_id": job, "live_scheduler_status": observed,
            "terminal_evidence": "actual_scheduler_DONE_and_LSF" if observed == "DONE" else "same_job_LSF_DONE_persisted",
            "persisted_scheduler_proofs": persisted, "lsf_sha256": sha(packet.bytes(lsf_path))}


def raw_cases(packet, root, kind, expected):
    require(isinstance(expected, dict) and expected and all(isinstance(k, str) and k.startswith("tests.test_")
            and type(v) is int and v > 0 for k, v in expected.items()), "Exact module counts absent")
    names = ["tests.xml"] if kind == "protenix" else ["tests-torch.xml", "tests-policy.xml"]
    cases = []
    for name in names:
        tree = ET.fromstring(packet.bytes(root / "cpu/payload" / name))
        require(not any(node.tag in ("failure", "error", "skipped") for node in tree.iter()),
                "Allocated CPU XML contains failures/errors/skips")
        for suite in tree.iter("testsuite"):
            require(all(int(suite.get(key, "0")) == 0 for key in ("failures", "errors", "skipped")),
                    "Allocated CPU suite header reports failure/error/skip")
        cases.extend(tree.iter("testcase"))
    require(cases and all(case.get("classname") and case.get("name") and not list(case) for case in cases),
            "Malformed or unsuccessful raw CPU case")
    require(len({(c.get("classname"), c.get("name")) for c in cases}) == len(cases), "Duplicate raw CPU cases")
    counts = dict(Counter(c.get("classname") for c in cases))
    require(counts == expected, "Raw CPU exact per-module attribution/counts differ")
    return len(cases), counts


def fixture_commands(packet, root, source, kind, python, torch=None, *, local=None, expected_modules=None):
    payload = root / "cpu/payload"
    if kind == "protenix":
        actual = packet.json(payload / "command.json").get("argv")
        require(isinstance(actual, list) and actual.count("--basetemp") == 1,
                "Actual allocated fixture TMPDIR argument missing")
        index = actual.index("--basetemp")
        require(index + 1 < len(actual), "Fixture TMPDIR value missing")
        temporary = remote_path(actual[index + 1])
        require(temporary.name == "protenix-precision-fixtures", "Owned fixture directory differs")
        expected = [python, "-m", "pytest", str(source / "tests"), "--rootdir", str(source),
                    "--import-mode=importlib", "-o", "addopts=", "-o", "cache_dir=" + str(payload / "cache"),
                    "--basetemp", str(temporary), "-q", "--junitxml=" + str(payload / "tests.xml")]
        require(actual == expected, "Actual Protenix allocated fixture command differs")
        return [actual]
    # Read the literal generated fixture vectors from the authenticated release,
    # rather than carrying a stale list when an allocated metadata module is added.
    tree = ast.parse((Path(local) / "scripts/dbmol_recovery_ops.py").read_bytes())
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                  and isinstance(node.target, ast.Tuple)
                  and [n.id if isinstance(n, ast.Name) else None for n in node.target.elts]
                  == ["python", "names", "label"]]
    require(len(candidates) == 1 and isinstance(candidates[0].iter, ast.Tuple),
            "Unique literal allocated fixture vector missing")
    specifications = []
    for item in candidates[0].iter.elts:
        require(isinstance(item, ast.Tuple) and len(item.elts) == 3
                and isinstance(item.elts[0], ast.Name), "Malformed literal fixture group")
        interpreter = {"TORCH": torch, "PYTHON": python}.get(item.elts[0].id)
        names, label = (ast.literal_eval(n) for n in item.elts[1:])
        require(isinstance(interpreter, str) and isinstance(names, list) and names
                and all(isinstance(n, str) and re.fullmatch(r"test_[a-z0-9_]+\.py", n) for n in names)
                and label in ("torch", "policy"), "Invalid declared fixture group")
        specifications.append((interpreter, names, label))
    require([entry[2] for entry in specifications] == ["torch", "policy"]
            and specifications[0][1] == ["test_dbmol_reference.py"],
            "Both exact declared fixture interpreters required")
    named = ["tests." + name[:-3] for _, names, _ in specifications for name in names]
    require(len(named) == len(set(named)) and set(named) == set(expected_modules),
            "Declared fixture vectors differ from exact raw XML attribution")
    commands = []
    for interpreter, names, label in specifications:
        actual = packet.json(payload / ("command-" + label + ".json")).get("argv")
        expected = [interpreter, "-m", "pytest", *[str(source / "tests" / name) for name in names],
                    "--rootdir", str(source), "--import-mode=importlib", "-q", "-o", "addopts=",
                    "-p", "no:cacheprovider", "--junitxml=" + str(payload / ("tests-" + label + ".xml"))]
        require(actual == expected, "Actual DBMol allocated fixture command differs: " + label)
        commands.append(actual)
    return commands


def storage(value):
    require(value.get("status") == "passed" and value.get("original_gate_changed") is False
            and value.get("reserve_bytes") == 4 * GIB and value.get("file_reserve") == 100000
            and type(value.get("required_bytes")) is int and type(value.get("available_bytes")) is int
            and value["required_bytes"] > 4 * GIB and value["available_bytes"] >= value["required_bytes"],
            "Fresh complete storage/reserve gate did not pass")
    require(value.get("forecasts_removed", False) is False, "Full-future projections were removed")
    inherited = value.get("inherited_gate", value.get("parent_gate"))
    require(isinstance(inherited, dict), "Whole inherited future gate missing")
    require(inherited.get("status") == "passed" and inherited.get("reserve_bytes") == 4 * GIB
            and inherited.get("file_reserve") == 100000 and inherited.get("original_gate_changed") is False,
            "Inherited whole futures/reserves not passed")
    current = value.get("current_chemistry_and_new_scopes")
    if current is not None:
        storage(current)
    return {"required_bytes": value["required_bytes"], "available_bytes": value["available_bytes"],
            "reserve_bytes": value["reserve_bytes"], "file_reserve": value["file_reserve"],
            "scope": "Authenticated unchanged full-future guard; capacity observation, not reservation"}


def verify(kind, observation, source, initial_submission):
    require(kind in ("protenix", "dbmol"), "Unknown recovery kind")
    packet = Packet(observation)
    manifest, pin, plan = local_source(source)
    plan_sha = sha((Path(source) / "plan.json").read_bytes())
    started = loads(Path(initial_submission).read_bytes())
    prep = started.get("preparation")
    require(isinstance(prep, dict) and prep.get("manifest_sha256") == pin
            and prep.get("plan_sha256") == plan_sha and prep.get("files") == len(manifest["files"])
            and prep.get("cpu_counts") == manifest["cpu_counts"], "Initial submission/source/plan differs")
    archive = Path(source) / "source.tar.gz"
    if archive.exists():
        archive_bytes = archive.read_bytes()
        require(not archive.is_symlink() and len(archive_bytes) == prep.get("archive_bytes")
                and sha(archive_bytes) == prep.get("archive_sha256"), "Packaged source archive bytes differ")
    if kind == "protenix":
        require(started.get("original_results_changed") is False and started.get("optimizer_updates") == 0,
                "Initial preservation/zero-update scope differs")
    else:
        # DBMol bootstrap has no duplicate preservation flags; its authenticated
        # manifest and closed plan carry them. Do not invent absent fields.
        require(started.get("status") == "CPU_submitted_native_unvalidated"
                and plan.get("optimizer_updates") == 0 and plan.get("original_results_changed") is False,
                "DBMol initial CPU-only preservation scope differs")
    root = remote_path(plan["control_root"] if kind == "protenix" else plan["root"])
    original_cpu = started["cpu"]
    command = original_cpu.get("argv", [])
    require(len(command) >= 6 and command[-1] == "cpu" and command[-2] == pin, "CPU wrapper/pin missing")
    remote_source = remote_path(command[-3])
    require(kind != "protenix" or str(remote_source) == plan["source_root"], "Plan/source root differs")
    own_cpu = packet.json(root / "cpu/submission.json")
    own_callback = packet.json(root / "after-cpu/submission.json")
    require(own_cpu == original_cpu and own_callback == started["after_cpu"], "Actual initial job identity changed")
    cpu, cpu_name = submission(own_cpu, kind, "cpu", root, remote_source, pin)
    callback, callback_name = submission(own_callback, kind, "after-cpu", root, remote_source, pin, cpu)
    cpu_terminal = terminal(packet, root, "cpu", cpu, cpu_name)
    callback_terminal = terminal(packet, root, "after-cpu", callback, callback_name)
    count, counts = raw_cases(packet, root, kind, manifest["cpu_counts"])
    assessment = packet.json(root / "cpu/payload/assessment.json")
    require(assessment.get("status") == "passed" and assessment.get("cases") == count
            and assessment.get("source_manifest_sha256") == pin, "Allocated CPU assessment differs")
    certificate_path = root / "cpu-certificate.json"
    certificate = packet.json(certificate_path)
    require(certificate.get("status") == "passed" and certificate.get("cases") == count
            and certificate.get("source_manifest_sha256") == pin, "CPU certificate/source/counts differ")
    require(isinstance(certificate.get("proofs"), list) and certificate["proofs"], "CPU certificate proofs absent")
    proof_names = []
    for proof in certificate["proofs"]:
        packet.proof(proof)
        require(remote_path(proof["path"]).is_relative_to(root / "cpu"), "CPU proof outside own CPU producer")
        proof_names.append(proof["path"])
    require(len(proof_names) == len(set(proof_names)), "Duplicate certificate proof binding")
    for relative in (["submission.json", "stdout." + cpu + ".log", "payload/tests.xml", "payload/assessment.json"]
                     if kind == "protenix" else ["submission.json", "stdout." + cpu + ".log", "payload/tests-torch.xml",
                                                "payload/tests-policy.xml", "payload/assessment.json"]):
        require(str(root / "cpu" / relative) in proof_names, "Required original CPU proof not certificate-bound")
    module = "scripts/protenix_precision_ops.py" if kind == "protenix" else "scripts/dbmol_recovery_ops.py"
    python = string_constant(source, module, "PYTHON")
    commands = fixture_commands(packet, root, remote_source, kind, python,
                                string_constant(source, module, "TORCH") if kind == "dbmol" else None,
                                local=source, expected_modules=manifest["cpu_counts"])
    for name in (["command.json"] if kind == "protenix" else ["command-torch.json", "command-policy.json"]):
        require(str(root / "cpu/payload" / name) in proof_names, "Fixture command not CPU-certificate-bound")
    if kind == "protenix":
        require(certificate.get("plan_sha256") == plan_sha and certificate.get("original_scientific_cpu_replaced") is False
                and assessment.get("plan_sha256") == plan_sha, "Protenix original CPU/plan scope differs")
        proof = assessment.get("actual_native_startup")
        require(isinstance(proof, dict) and set(proof) == {"receipt", "invocation"}, "Actual startup proofs absent")
        for binding in proof.values():
            packet.proof(binding)
            require(binding["path"] in proof_names, "Startup proof not CPU-certificate-bound")
        invocation = packet.json(root / "cpu/payload/native-startup-invocation.json")
        endpoint = packet.json(root / "cpu/payload/native-startup/endpoint-startup.json")
        expected = [python, str(remote_source / "scripts/protenix_precision_native.py"), "--plan", str(remote_source / "plan.json"),
                    "--plan-sha256", plan_sha, "--output", str(root / "cpu/payload/native-startup"), "--startup-only"]
        require(invocation.get("argv") == expected and invocation.get("returncode") == 0,
                "Actual Protenix native startup command failed/differs")
        require(endpoint == {"status": "passed_metadata_only", "numerical_setup_entered": False, "updates": 0,
                             "plan_sha256": plan_sha, "job_id": cpu}, "Protenix startup result differs")
        gate_path = root / "after-cpu/dispatch-storage/precision-storage-dispatch.json"
        startup_commands = [expected]
    else:
        require(assessment.get("optimizer_updates") == 0, "DBMol CPU optimizer scope differs")
        command = packet.json(root / "cpu/payload/startup-command.json").get("argv")
        expected = [python, str(remote_source / "scripts/dbmol_endpoint_precision.py"), "--plan", str(remote_source / "plan.json"),
                    "--plan-sha256", plan_sha, "--output", str(root / "cpu/payload/startup"), "--startup-only"]
        require(command == expected, "Actual DBMol coordinator startup command differs")
        endpoint = packet.json(root / "cpu/payload/startup/assessment.json")
        require(endpoint == {"status": "passed_metadata_only", "optimizer_updates": 0,
                             "numerical_setup_entered": False, "plan_sha256": plan_sha}, "DBMol coordinator startup result differs")
        reference_command = packet.json(root / "cpu/payload/reference-startup-command.json").get("argv")
        expected_reference = plan["reference_argv"] + ["--startup-only", "--startup-output", str(root / "cpu/payload/reference-startup.json")]
        require(reference_command == expected_reference and reference_command[0] == string_constant(source, module, "TORCH")
                and reference_command[2] == str(remote_source / "scripts/dbmol_reference.py"), "Reference actual startup command differs")
        reference = packet.json(root / "cpu/payload/reference-startup.json")
        require(reference == {"status": "passed_metadata_only", "numerical_setup_entered": False,
                              "optimizer_updates": 0, "job_id": cpu}, "DBMol reference startup result differs")
        for relative in ("startup-command.json", "startup/assessment.json", "reference-startup-command.json", "reference-startup.json"):
            require(str(root / "cpu/payload" / relative) in proof_names, "DBMol startup proof not certificate-bound")
        gate_path = root / "after-cpu/recovery-capacity-dispatch.json"
        startup_commands = [expected, expected_reference]
    gate = storage(packet.json(gate_path))
    native = None
    native_path = str(root / "gpu/submission.json")
    if native_path in packet.files:
        own_native = packet.json(native_path)
        job, name = submission(own_native, kind, "gpu", root, remote_source, pin)
        state = scheduler_state(packet.raw.get("scheduler"), job, name, "gpul40s")
        native = {"job_id": job, "live_scheduler_status": state, "actual_resources_verified": True,
                  "native_pass_established": False, "optimizer_jobs_submitted": False}
    return {"status": "passed_independent_CPU_certificate_startup_storage_delivery_audit",
            "kind": kind, "observed_utc": packet.raw.get("captured_utc", packet.raw.get("observed_utc")),
            "source_manifest_sha256": pin, "source_members_verified": len(manifest["files"]), "plan_sha256": plan_sha,
            "files_byte_verified": len(packet.files), "cases": count, "module_counts": counts,
            "cpu": cpu_terminal, "callback": callback_terminal,
            "cpu_certificate_sha256": sha(packet.bytes(certificate_path)), "certificate_proofs_verified": len(proof_names),
            "actual_fixture_commands": commands, "actual_startup_commands": startup_commands,
            "storage": gate, "storage_gate": str(gate_path),
            "native_submission": native, "native_pass_established": False, "optimizer_jobs_submitted": False,
            "original_results_changed": False, "local_scientific_calculations": False,
            "observation_sha256": sha(packet.data), "initial_submission_sha256": sha(Path(initial_submission).read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("protenix", "dbmol"), required=True)
    for option in ("observation", "source", "submission", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists() and Path(args.output).parent.is_dir(),
            "A fresh output file in an existing directory is required")
    result = verify(args.kind, args.observation, args.source, args.submission)
    with Path(args.output).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
