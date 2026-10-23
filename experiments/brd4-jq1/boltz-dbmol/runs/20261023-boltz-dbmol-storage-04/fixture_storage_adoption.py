"""Allocated metadata-only tests of completion-aware storage adoption.

No model, array, derivative or optimizer is imported or evaluated here. Small
synthetic terminal files exercise byte/source/job/status authentication. A
failed original scientific summary remains failed when its future byte budget
is zero because its entire requested update budget was already consumed.
"""

import copy
import importlib.util
import json
import shlex
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


def _load():
    try:
        import dbmol_storage_ops
        return dbmol_storage_ops
    except ModuleNotFoundError:
        path = Path(__file__).with_name("dbmol_storage_ops.py")
        spec = importlib.util.spec_from_file_location("dbmol_storage_ops", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


ops = _load()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def paths(monkeypatch, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name, path in (("BASE", base), ("ROOT", base / "pilot"),
                       ("SOURCE", base / "scientific" / "source"),
                       ("LAUNCHER", base / "adoption" / "source"),
                       ("OLD", base / "original"),
                       ("ORIGINAL_SOURCE", base / "original-source"),
                       ("CPU03_SOURCE", base / "launcher03-source")):
        monkeypatch.setattr(ops, name, path)
    ops.ROOT.mkdir()
    monkeypatch.setattr(ops, "LAUNCHER_MANIFEST", None)
    return ops


@pytest.fixture
def terminals(paths, monkeypatch):
    manifest = paths.ORIGINAL_SOURCE / "source_manifest.json"
    write(manifest, {"files": {}})
    monkeypatch.setattr(paths, "ORIGINAL_SOURCE_MANIFEST", paths.sha(manifest))
    pins = {}
    for arm, job, status, lsf_status in (("protenix", "100", "passed", "Done"),
                                        ("boltz2", "101", "failed", "Exited")):
        root = paths.OLD / ("run-" + arm)
        summary = root / "payload" / "study" / "summary.json"
        # Update/state counts are metadata only. In production the fixed pins
        # bind the actual original summaries with all 10,000/10,001 rows.
        value = {"status": status, "job_id": job,
                 "cases": [{"status": "passed", "updates": [0] * 10000,
                            "states": [0] * 10001}],
                 "source_manifest_sha256": paths.ORIGINAL_SOURCE_MANIFEST}
        if arm == "boltz2":
            value["error"] = "AssertionError: Native affinity E_upper finite differences failed"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log = root / ("stdout." + job + ".log")
        log.write_text("Subject: Job " + job + ": original " + lsf_status + "\n" +
                       ("Successfully completed.\n" if lsf_status == "Done" else
                        "Exited with exit code 1.\n"), encoding="utf-8")
        submission = root / "submission.json"
        write(submission, {"outcome": "submitted", "job_id": job, "phase": "run",
                           "source_manifest_sha256": paths.ORIGINAL_SOURCE_MANIFEST,
                           "args": ["bsub", str(paths.ORIGINAL_SOURCE),
                                    paths.ORIGINAL_SOURCE_MANIFEST], "dependency": None})
        pins[arm] = {"job_id": job, "status": status, "lsf_status": lsf_status,
                     "summary_bytes": summary.stat().st_size, "summary_sha256": paths.sha(summary),
                     "stdout_bytes": log.stat().st_size, "stdout_sha256": paths.sha(log),
                     "submission_bytes": submission.stat().st_size,
                     "submission_sha256": paths.sha(submission)}
    monkeypatch.setattr(paths, "TERMINAL_PINS", pins)
    return paths, pins


def test_exact_completed_original_protenix_and_failed_boltz_are_terminal_evidence(terminals):
    module, _ = terminals
    pro = module.terminal_original_evidence("protenix")
    bol = module.terminal_original_evidence("boltz2")
    assert pro is not None and bol is not None
    # Accounting is not scientific certification: the original failed bytes
    # and failed status remain unmodified.
    summary = module.read(module.OLD / "run-boltz2" / "payload" / "study" / "summary.json")
    assert summary["status"] == "failed"
    assert len(summary["cases"][0]["updates"]) == 10000
    raw = (module.OLD / "run-boltz2" / "payload" / "study" / "summary.json").read_text()
    assert raw.index('"cases"') < raw.index('"status"')
    assert raw.rindex('"status"') > 65536
    assert summary["cases"][0]["status"] == "passed"  # root status must be read independently


def test_future_zero_requires_terminal_proofs_but_active_chemistry_retains_its_full_remaining_projection(terminals):
    module, _ = terminals
    current = {"protenix": 1234, "boltz2": 2345, "boltz2_chemistry": 3456}
    remaining, evidence = module.original_remaining(current)
    assert remaining["protenix"] == remaining["boltz2"] == 0
    assert remaining["boltz2_chemistry"] == module.OLD_FULL["boltz2_chemistry"] - 3456
    assert {"protenix", "boltz2"}.issubset(evidence)
    assert current == {"protenix": 1234, "boltz2": 2345, "boltz2_chemistry": 3456}
    assert module.read(module.OLD / "run-boltz2" / "payload" / "study" / "summary.json")["status"] == "failed"


@pytest.mark.parametrize("fault", ("summary_missing", "stdout_missing", "submission_missing",
                                   "summary_bytes", "summary_hash", "stdout_hash", "submission_hash",
                                   "partial_updates", "partial_states",
                                   "authenticated_partial_updates", "authenticated_partial_states", "case_failed"))
def test_missing_partial_or_changed_terminal_evidence_cannot_release_future_budget(terminals, fault):
    module, pins = terminals
    root = module.OLD / "run-boltz2"
    summary, log, submission = (root / "payload" / "study" / "summary.json",
                                 root / "stdout.101.log", root / "submission.json")
    if fault.endswith("_missing"):
        {"summary_missing": summary, "stdout_missing": log, "submission_missing": submission}[fault].unlink()
    elif fault == "summary_bytes": summary.write_bytes(summary.read_bytes() + b" ")
    elif fault == "summary_hash": pins["boltz2"]["summary_sha256"] = "a" * 64
    elif fault == "stdout_hash": log.write_text(log.read_text() + "changed\n")
    elif fault == "submission_hash": submission.write_text(submission.read_text() + " ")
    else:
        value = module.read(summary)
        if fault == "case_failed": value["cases"][0]["status"] = "failed"
        else: value["cases"][0]["updates" if fault.endswith("updates") else "states"].pop()
        write(summary, value)
        if fault.startswith("authenticated_") or fault == "case_failed":
            pins["boltz2"]["summary_bytes"] = summary.stat().st_size
            pins["boltz2"]["summary_sha256"] = module.sha(summary)
    with pytest.raises((ValueError, AssertionError, OSError)):
        module.original_remaining({"protenix": 1, "boltz2": 1, "boltz2_chemistry": 1})


@pytest.mark.parametrize("fault", ("summary_status", "lsf_job", "lsf_status", "submission_job", "source_manifest"))
def test_correct_hashes_do_not_replace_terminal_status_same_job_and_source_authentication(terminals, fault):
    module, pins = terminals
    root, pin = module.OLD / "run-boltz2", pins["boltz2"]
    if fault == "source_manifest":
        path = module.ORIGINAL_SOURCE / "source_manifest.json"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with pytest.raises((ValueError, AssertionError)):
            module.terminal_original_evidence("boltz2")
        path.write_bytes(original)
        # The genuine original submission uses its explicit science field;
        # argv names the historical launcher and must not be invented anew.
        for prefix, artifact in (("summary", root / "payload" / "study" / "summary.json"),
                                 ("submission", root / "submission.json")):
            saved_bytes = artifact.read_bytes()
            saved_pin = (pin[prefix + "_bytes"], pin[prefix + "_sha256"])
            value = module.read(artifact)
            value["source_manifest_sha256"] = "a" * 64
            write(artifact, value)
            pin[prefix + "_bytes"], pin[prefix + "_sha256"] = artifact.stat().st_size, module.sha(artifact)
            with pytest.raises((ValueError, AssertionError)):
                module.terminal_original_evidence("boltz2")
            artifact.write_bytes(saved_bytes)
            pin[prefix + "_bytes"], pin[prefix + "_sha256"] = saved_pin
        return
    if fault == "summary_status":
        path = root / "payload" / "study" / "summary.json"
        value = module.read(path); value["status"] = "passed"; write(path, value)
        prefix = "summary"
    elif fault.startswith("lsf_"):
        path = root / "stdout.101.log"
        path.write_text("Subject: Job " + ("999" if fault == "lsf_job" else "101") +
                        ": original " + ("Done" if fault == "lsf_status" else "Exited") +
                        "\nExited with exit code 1.\n")
        prefix = "stdout"
    else:
        path = root / "submission.json"
        value = module.read(path)
        value["job_id"] = "999"
        write(path, value); prefix = "submission"
    pin[prefix + "_bytes"], pin[prefix + "_sha256"] = path.stat().st_size, module.sha(path)
    with pytest.raises((ValueError, AssertionError)):
        module.terminal_original_evidence("boltz2")


@pytest.mark.parametrize("arm", ("boltz2_chemistry", "unknown", "protenix-random"))
def test_no_other_arm_gets_original_terminal_future_zero_by_label(terminals, arm):
    module, _ = terminals
    with pytest.raises((ValueError, KeyError)):
        module.terminal_original_evidence(arm)


@pytest.mark.parametrize("fault", ("missing", "extra", "negative", "boolean", "float", "active_oversize"))
def test_completion_accounting_rejects_malformed_current_inventory_and_active_projection_overrun(terminals, fault):
    module, _ = terminals
    current = {"protenix": 1, "boltz2": 2, "boltz2_chemistry": 3}
    if fault == "missing": del current["protenix"]
    elif fault == "extra": current["new-arm"] = 0
    elif fault == "negative": current["boltz2"] = -1
    elif fault == "boolean": current["protenix"] = True
    elif fault == "float": current["protenix"] = 1.0
    else: current["boltz2_chemistry"] = module.OLD_FULL["boltz2_chemistry"] + 1
    with pytest.raises(ValueError): module.original_remaining(current)


def readonly_tree(root, files):
    root.mkdir(parents=True)
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write(root / "source_manifest.json", {"files": {name: ops.sha(root / name) for name in files}})
    digest = ops.sha(root / "source_manifest.json")
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    return digest


def writable_tree(root):
    if not root.exists(): return
    root.chmod(0o755)
    for path in root.rglob("*"):
        if not path.is_symlink(): path.chmod(0o755 if path.is_dir() else 0o644)


@pytest.fixture
def sources(paths, monkeypatch):
    science = readonly_tree(paths.SOURCE, {"science.py": b"# immutable scientific fixture\n"})
    launcher = readonly_tree(paths.LAUNCHER, {"launch.py": b"# immutable launcher fixture\n"})
    monkeypatch.setattr(paths, "SCIENCE_MANIFEST", science)
    try:
        yield paths, science, launcher
    finally:
        writable_tree(paths.SOURCE)
        writable_tree(paths.LAUNCHER)


def test_dual_source_verification_hashes_every_scientific_and_launcher_member(sources):
    module, science, launcher = sources
    proofs = module.verify_launcher(science, launcher)
    assert module.LAUNCHER_MANIFEST == launcher
    assert {Path(proof["path"]).name for proof in proofs} == {"launch.py", "source_manifest.json"}
    assert all(module.sha(proof["path"]) == proof["sha256"] for proof in proofs)


@pytest.mark.parametrize("fault", ("science_manifest", "launcher_manifest", "science_member", "launcher_member", "writable_source", "unmanifested_member", "symlink"))
def test_changed_unsealed_or_extra_execution_input_cannot_enter_launcher_adoption(sources, fault):
    module, science, launcher = sources
    root = module.SOURCE if fault.startswith("science") or fault == "writable_source" else module.LAUNCHER
    if fault == "writable_source": root.chmod(0o755)
    elif fault == "unmanifested_member":
        root.chmod(0o755); (root / "undeclared.py").write_text("# undeclared\n")
        (root / "undeclared.py").chmod(0o444); root.chmod(0o555)
    elif fault == "symlink":
        root.chmod(0o755); (root / "alias.py").symlink_to(module.SOURCE / "science.py")
        root.chmod(0o555)
    else:
        name = "source_manifest.json" if fault.endswith("manifest") else ("science.py" if root == module.SOURCE else "launch.py")
        path = root / name; path.chmod(0o644); path.write_bytes(path.read_bytes() + b"changed")
        path.chmod(0o444)
    with pytest.raises(ValueError): module.verify_launcher(science, launcher)


@pytest.mark.parametrize("fault", ("missing", "malformed", "wrong_science", "changed_launcher"))
def test_dual_source_identities_cannot_be_omitted_or_replaced(paths, fault):
    if fault == "missing":
        with pytest.raises(ValueError): paths.command("cpu-04", "cpu", paths.SCIENCE_MANIFEST)
    elif fault == "malformed":
        with pytest.raises(ValueError): paths.configure_launcher("not-a-sha")
    elif fault == "wrong_science":
        with pytest.raises(ValueError): paths.verify_launcher("a" * 64, "b" * 64)
    else:
        paths.configure_launcher("a" * 64)
        with pytest.raises(ValueError): paths.configure_launcher("b" * 64)


@pytest.mark.parametrize("name,phase,slots,memory,wall", (
    ("cpu-04", "cpu", "4", "rusage[mem=4GB]", "01:00"),
    ("after-cpu-04", "after-cpu", "2", "rusage[mem=4GB]", "00:30"),
    ("pilot-native-0000", "native", "4", "rusage[mem=8GB]", "02:00"),
    ("pilot-segment-0000", "segment", "4", "rusage[mem=8GB]", "23:00")))
def test_new_cpu_parents_and_all_later_jobs_authenticate_both_source_wrappers(paths, name, phase, slots, memory, wall):
    paths.configure_launcher("b" * 64)
    args = paths.command(name, phase, paths.SCIENCE_MANIFEST)
    assert args[args.index("-n") + 1] == slots and memory in args
    assert args[args.index("-W") + 1] == wall
    assert str(paths.LAUNCHER) in args and str(paths.SOURCE) in args
    assert "b" * 64 in args and paths.SCIENCE_MANIFEST in args
    expected_wrapper = paths.LAUNCHER / "scripts" / "hpc" / ("dbmol_storage_cpu.lsf" if phase == "cpu" else "dbmol_storage_stage.lsf")
    assert str(expected_wrapper) in args
    assert str(paths.SOURCE / "scripts" / "hpc" / "dbmol_cpu.lsf") not in args
    assert str(paths.ROOT / "cpu") not in args and str(paths.ROOT / "after-cpu") not in args


@pytest.mark.parametrize("name,phase", (("cpu", "cpu"), ("after-cpu", "after-cpu"),
                                       ("cpu-02", "cpu"), ("after-cpu-02", "after-cpu")))
def test_failed_original_cpu_and_callback_parents_can_never_be_reused(paths, name, phase):
    paths.configure_launcher("b" * 64)
    with pytest.raises(ValueError): paths.command(name, phase, paths.SCIENCE_MANIFEST)


def test_completed_bytes_remain_stale_charged_and_active_and_other_allowances_are_unchanged(terminals, monkeypatch):
    module, _ = terminals
    module.configure_launcher("b" * 64)
    current = {"protenix": 100, "boltz2": 200, "boltz2_chemistry": 300}
    baseline_stale = sum(current.values()) + 12345
    captured = {}
    def gate(pool, raw, projected, stale, disk, inodes):
        captured.update(pool=pool, projected=projected, stale=stale)
        return {"status": "passed", "stale_bytes": stale,
                "reserve_bytes": 4 * 1024**3, "file_reserve": 100000}
    storage = SimpleNamespace(gate_record=gate, stale_work_bytes=lambda: baseline_stale,
                              inode_observation=lambda raw: {"mode": "fixture"})
    monkeypatch.setitem(sys.modules, "fixed25_lrreset_storage", storage)
    monkeypatch.setattr(module, "tree", lambda path: current.get(Path(path).name.removeprefix("run-"), 0))
    monkeypatch.setattr(module, "external_budgets", lambda full: ({"peer": {"remaining": 777}}, []))
    monkeypatch.setattr(module, "own_budget", lambda: 1024**3)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="authenticated quota fixture"))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3))
    parent = module.ROOT / module.CPU_PARENT; parent.mkdir()
    report = module.storage_gate(parent, "entry-fixture")
    expected = module.OLD_FULL["boltz2_chemistry"] - 300 + 777 + 1024**3 + 2 * 1024**3 + 512 * 1024**2 + 1024**3
    assert captured == {"pool": "work3", "projected": expected, "stale": baseline_stale}
    assert report["original_current"] == current
    assert report["original_future"] == {"protenix": 0, "boltz2": 0,
                                            "boltz2_chemistry": module.OLD_FULL["boltz2_chemistry"] - 300}
    assert report["reserve_bytes"] == 4 * 1024**3 and report["file_reserve"] == 100000
    assert report["terminal_original_evidence"]["boltz2"]["scientific_status"] == "failed"


def test_native_driver_consumes_authenticated_new_cpu04_reference_assets_and_unchanged_scientific_driver(paths, monkeypatch):
    paths.configure_launcher("b" * 64)
    cpu = paths.ROOT / "cpu-certificate.json"
    write(cpu, {"status": "passed", "source_manifest_sha256": paths.SCIENCE_MANIFEST,
                "launcher_manifest_sha256": paths.LAUNCHER_MANIFEST})
    payload = paths.ROOT / paths.CPU_PARENT / "payload"
    write(payload / "composition_reference.json", {"fixture": "composition"})
    write(payload / "pocket_definition.json", {"fixture": "pocket"})
    old = paths.ROOT / "cpu" / "storage-cpu-entry" / "gate.json"
    write(old, {"status": "failed", "reason": "historical resource rejection"})
    old_hash = paths.sha(old)
    parent = paths.ROOT / "pilot-native-0000"; parent.mkdir()
    monkeypatch.setattr(paths, "cpu_certificate", lambda manifest: cpu)
    monkeypatch.setattr(paths, "input_snapshot", lambda index: None)
    monkeypatch.setattr(paths, "storage_gate", lambda *args: {"status": "passed"})
    calls = []
    monkeypatch.setattr(paths.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    paths.produce(parent, "native", 0, paths.SCIENCE_MANIFEST)
    assert len(calls) == 1
    args = calls[0][0]
    assert args[1] == str(paths.SOURCE / "scripts" / "dbmol_driver.py")
    assert args[args.index("--reference") + 1] == str(payload / "composition_reference.json")
    assert args[args.index("--pocket") + 1] == str(payload / "pocket_definition.json")
    assert args[args.index("--cpu-certificate") + 1] == str(cpu)
    assert paths.sha(old) == old_hash and paths.read(old)["status"] == "failed"
    invocation = paths.read(parent / "payload" / "driver-invocation.json")
    assert invocation["science_manifest_sha256"] == paths.SCIENCE_MANIFEST
    assert invocation["launcher_manifest_sha256"] == paths.LAUNCHER_MANIFEST


def raw_cpu_groups(module):
    inventory = {
        "hpc-contract": ["test_inherited_contract"] * 500,
        "legacy": ["test_inherited_legacy"] * 239,
        "native-mapping": ["test_inherited_mapping"] * 34,
        "chemistry": ["test_inherited_chemistry"] * 59,
        "dbmol-native": ["tests.test_dbmol_objective", "tests.test_dbmol_native_paths"] + ["tests.test_dbmol_objective"] * 81,
        "dbmol-transport": ["tests.test_dbmol_pilot_contract", "tests.test_dbmol_gradient_contract", "tests.test_dbmol_bridge"] + ["tests.test_dbmol_pilot_contract"] * 145,
        "launcher-adoption": ["launchers.fixture_storage_adoption"] * 72,
    }
    payload = module.ROOT / module.CPU_PARENT / "payload"
    payload.mkdir(parents=True)
    for group, classes in inventory.items():
        suite = ET.Element("testsuite", tests=str(len(classes)), errors="0", failures="0", skipped="0")
        for index, name in enumerate(classes):
            ET.SubElement(suite, "testcase", classname=name, name="synthetic_metadata_" + str(index))
        ET.ElementTree(suite).write(payload / (group + ".xml"), encoding="utf-8", xml_declaration=True)
        (payload / (group + ".log")).write_text(str(len(classes)) + " passed\n")
    for name in ("composition_reference.json", "pocket_definition.json", "assessment.json", "oracles.json"):
        write(payload / name, {"metadata_fixture": name})
    return payload, {group: len(classes) for group, classes in inventory.items()}


def configure_cpu_fixture(module, monkeypatch):
    module.configure_launcher("b" * 64)
    proof = module.ROOT / module.CPU_PARENT / "actual-scheduler-fixture.txt"
    proof.parent.mkdir(parents=True)
    proof.write_text("Synthetic metadata proof only; no real scheduler claim\n")
    record = {"path": str(proof), "sha256": module.sha(proof)}
    monkeypatch.setattr(module, "final_job", lambda parent, queue: [record])
    monkeypatch.setattr(module, "verify_launcher", lambda science, launcher: [record])
    monkeypatch.setattr(module, "verify_failed_old_cpu", lambda: [record])
    actions = []
    monkeypatch.setattr(module, "storage_gate", lambda *args: actions.append(("storage", args)))
    monkeypatch.setattr(module, "queue_native", lambda *args: actions.append(("native", args)))
    return actions


def test_new_cpu_certificate_requires_raw_all_six_scientific_groups_plus_separate_adoption_group(paths, monkeypatch):
    actions = configure_cpu_fixture(paths, monkeypatch)
    _, expected = raw_cpu_groups(paths)
    paths.certify_cpu(paths.SCIENCE_MANIFEST)
    certificate = paths.read(paths.ROOT / "cpu-certificate.json")
    assert certificate["status"] == "passed"
    assert certificate["source_manifest_sha256"] == paths.SCIENCE_MANIFEST
    assert certificate["launcher_manifest_sha256"] == paths.LAUNCHER_MANIFEST
    assert certificate["launcher_source"] == str(paths.LAUNCHER)
    assert certificate["groups"] == expected and certificate["cases"] == sum(expected.values())
    proof_paths = {Path(proof["path"]).name for proof in certificate["proofs"]}
    assert {group + ".xml" for group in expected}.issubset(proof_paths)
    assert {group + ".log" for group in expected}.issubset(proof_paths)
    assert all(paths.sha(proof["path"]) == proof["sha256"] for proof in certificate["proofs"])
    assert actions == [("storage", (paths.ROOT / paths.AFTER_CPU_PARENT, "native-dispatch")),
                       ("native", (0, paths.SCIENCE_MANIFEST))]
    # Every transport file must share the declared pytest root, including the
    # unchanged bridge copy. A passing out-of-root file lacks its JUnit module
    # attribution and must never satisfy the strict certificate below.
    wrapper = Path(ops.__file__).parent / "hpc" / "dbmol_storage_cpu.lsf"
    transport_line = next(line for line in wrapper.read_text().splitlines()
                          if ' -m pytest ' in line and 'dbmol-transport.xml' in line)
    arguments = shlex.split(transport_line)
    expected_paths = {"$launcher/tests/" + name for name in
                      ("test_dbmol_gradient_contract.py", "test_dbmol_bridge.py", "test_dbmol_pilot_contract.py")}
    actual_paths = {argument for argument in arguments if argument.endswith(".py")}
    assert actual_paths == expected_paths
    assert arguments[arguments.index("--rootdir") + 1] == "$launcher"


@pytest.mark.parametrize("fault", ("missing_group", "case_count", "failure", "error", "skip", "log_count", "native_attribution", "transport_attribution", "adoption_attribution"))
def test_missing_failed_or_unattributed_raw_cpu_groups_cannot_certify_or_dispatch(paths, monkeypatch, fault):
    actions = configure_cpu_fixture(paths, monkeypatch)
    payload, _ = raw_cpu_groups(paths)
    if fault == "missing_group": (payload / "launcher-adoption.xml").unlink()
    elif fault == "log_count": (payload / "chemistry.log").write_text("58 passed\n")
    else:
        group = {"case_count": "dbmol-transport", "native_attribution": "dbmol-native", "transport_attribution": "dbmol-transport",
                 "adoption_attribution": "launcher-adoption"}.get(fault, "hpc-contract")
        path = payload / (group + ".xml")
        tree = ET.parse(path); suite = tree.getroot()
        if fault == "case_count":
            suite.remove(list(suite)[-1])
            # Matching rows/log/module attribution still cannot substitute for
            # every exact fixture declared by the immutable source.
            (payload / (group + ".log")).write_text("147 passed\n")
        elif fault in ("failure", "error", "skip"):
            ET.SubElement(list(suite)[0], "skipped" if fault == "skip" else fault)
        elif fault == "transport_attribution":
            # Exact CPU03 regression: only the unchanged bridge rows lose
            # attribution, while all 148 rows and both other modules pass.
            for case in suite:
                if "test_dbmol_bridge" in case.get("classname", "").split("."):
                    case.set("classname", "")
        else:
            for case in suite: case.set("classname", "wrong_module")
        tree.write(path, encoding="utf-8", xml_declaration=True)
    with pytest.raises((ValueError, OSError)):
        paths.certify_cpu(paths.SCIENCE_MANIFEST)
    assert actions == [] and not (paths.ROOT / "cpu-certificate.json").exists()


@pytest.mark.parametrize("fault", ("science", "launcher", "launcher_path", "proof"))
def test_later_phase_rejects_cpu_certificate_with_wrong_dual_source_or_changed_proof(paths, monkeypatch, fault):
    paths.configure_launcher("b" * 64)
    proof = paths.ROOT / "cpu-proof.json"; write(proof, {"status": "passed"})
    certificate = {"status": "passed", "suite": paths.SUITE,
                   "source_manifest_sha256": paths.SCIENCE_MANIFEST,
                   "launcher_manifest_sha256": paths.LAUNCHER_MANIFEST,
                   "launcher_source": str(paths.LAUNCHER),
                   "proofs": [{"path": str(proof), "sha256": paths.sha(proof)}]}
    if fault == "science": certificate["source_manifest_sha256"] = "a" * 64
    elif fault == "launcher": certificate["launcher_manifest_sha256"] = "a" * 64
    elif fault == "launcher_path": certificate["launcher_source"] = "/wrong/launcher"
    else: proof.write_text("changed")
    write(paths.ROOT / "cpu-certificate.json", certificate)
    monkeypatch.setattr(paths, "verify_launcher", lambda science, launcher: [])
    with pytest.raises(ValueError): paths.cpu_certificate(paths.SCIENCE_MANIFEST)


@pytest.fixture
def failed_cpu(paths, monkeypatch):
    pins = {}
    for name in paths.FAILED_CPU_PINS:
        path = paths.ROOT / name
        parent = Path(name).parts[0]
        job = "29798549" if parent == "cpu" else "29798550"
        if path.name == "submission.json":
            write(path, {"job_id": job, "outcome": "submitted"})
        elif path.name.endswith(".log"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Subject: Job " + job + ": old resource rejection Exited\nExited with exit code 1.\n")
        else:
            write(path, {"status": "failed", "reason": "entry byte admission"})
        pins[name] = paths.sha(path)
    monkeypatch.setattr(paths, "FAILED_CPU_PINS", pins)
    payload = paths.ROOT / "cpu-02" / "payload"
    payload.mkdir(parents=True)
    for parent, job in (("cpu-02", "29798699"), ("after-cpu-02", "29798700")):
        write(paths.ROOT / parent / "submission.json", {"job_id": job, "outcome": "submitted"})
        (paths.ROOT / parent / ("stdout." + job + ".log")).write_text(
            "Subject: Job " + job + ": previous fixture rejection Exited\nExited with exit code 1.\n")
    write(paths.ROOT / "cpu-02" / "storage-cpu-entry" / "gate.json", {"status": "passed"})
    for group, count in (("hpc-contract", 500), ("legacy", 239), ("native-mapping", 34),
                         ("chemistry", 59), ("dbmol-native", 83), ("dbmol-transport", 148)):
        suite = ET.Element("testsuite")
        for index in range(count):
            failed_name = paths.FAILED_CPU02_NAMES[index] if group == "dbmol-transport" and index < 7 else None
            case = ET.SubElement(suite, "testcase", classname="tests.fixture_historical",
                                 name=failed_name or "historical_metadata_" + str(index))
            if failed_name: ET.SubElement(case, "failure", message="Preserved historical fixture defect")
        ET.ElementTree(suite).write(payload / (group + ".xml"), encoding="utf-8", xml_declaration=True)
        (payload / (group + ".log")).write_text("7 failed, 141 passed\n" if group == "dbmol-transport" else str(count) + " passed\n")
    monkeypatch.setattr(paths, "FAILED_CPU02_PINS", {name: paths.sha(paths.ROOT / name)
                                                    for name in paths.FAILED_CPU02_PINS})
    # CPU03 passed every allocated case; its callback failed only attribution.
    # Preserve these separate actual outcomes rather than relabeling CPU03.
    source_manifest = paths.CPU03_SOURCE / "source_manifest.json"
    write(source_manifest, {"files": {"bridge.py": "a" * 64}})
    monkeypatch.setattr(paths, "CPU03_MANIFEST", paths.sha(source_manifest))
    history_payload = paths.ROOT / "cpu-03" / "payload"
    history_payload.mkdir(parents=True)
    for parent, job, status in (("cpu-03", "29798769", "Done"),
                                ("after-cpu-03", "29798770", "Exited")):
        write(paths.ROOT / parent / "submission.json", {"job_id": job, "outcome": "submitted"})
        (paths.ROOT / parent / ("stdout." + job + ".log")).write_text(
            "Subject: Job " + job + ": historical " + status + "\n" +
            ("Successfully completed.\n" if status == "Done" else "Exited with exit code 1.\n"))
    (paths.ROOT / "after-cpu-03" / "stderr.29798770.log").write_text(
        "ValueError: Missing transport/scope fixture attribution\n")
    write(paths.ROOT / "cpu-03" / "storage-cpu-entry" / "gate.json", {"status": "passed"})
    for group, count in (("hpc-contract", 500), ("legacy", 239), ("native-mapping", 34),
                         ("chemistry", 59), ("dbmol-native", 83), ("dbmol-transport", 148),
                         ("launcher-adoption", 72)):
        suite = ET.Element("testsuite")
        for index in range(count):
            # Nine passing bridge rows were outside CPU03's pytest root.
            classname = "" if group == "dbmol-transport" and index < 9 else "tests.historical_fixture"
            ET.SubElement(suite, "testcase", classname=classname, name="passed_history_" + str(index))
        ET.ElementTree(suite).write(history_payload / (group + ".xml"), encoding="utf-8", xml_declaration=True)
        (history_payload / (group + ".log")).write_text(str(count) + " passed\n")
    monkeypatch.setattr(paths, "CPU03_HISTORY_PINS", {name: paths.sha(paths.ROOT / name)
                                                     for name in paths.CPU03_HISTORY_PINS})
    return paths, pins


def test_original_failed_cpu_callback_and_gate_remain_failed_immutable_evidence(failed_cpu):
    module, pins = failed_cpu
    proofs = module.verify_failed_old_cpu()
    assert {proof["path"] for proof in proofs} == ({str(module.ROOT / name) for name in
                                                  set(pins) | set(module.FAILED_CPU02_PINS) | set(module.CPU03_HISTORY_PINS)} |
                                                 {str(module.CPU03_SOURCE / "source_manifest.json")})
    assert all(module.sha(proof["path"]) == proof["sha256"] for proof in proofs)
    assert module.read(module.ROOT / "cpu" / "storage-cpu-entry" / "gate.json")["status"] == "failed"
    assert module.CPU_PARENT == "cpu-04" and module.AFTER_CPU_PARENT == "after-cpu-04"
    # Source03 also preserves the seven genuine CPU02 fixture failures. Check
    # its independent reader within this existing metadata case, without
    # promoting or rewriting those failed raw proofs.
    transport = module.ROOT / "cpu-02" / "payload" / "dbmol-transport.xml"
    original_bytes = transport.read_bytes()
    pin_name = "cpu-02/payload/dbmol-transport.xml"
    original_pin = module.FAILED_CPU02_PINS[pin_name]
    tree = ET.parse(transport)
    failing = next(case for case in tree.iter("testcase") if case.find("failure") is not None)
    failing.remove(failing.find("failure"))
    tree.write(transport, encoding="utf-8", xml_declaration=True)
    module.FAILED_CPU02_PINS[pin_name] = module.sha(transport)
    with pytest.raises(ValueError): module.verify_failed_cpu02()
    transport.write_bytes(original_bytes)
    module.FAILED_CPU02_PINS[pin_name] = original_pin
    later = transport.with_name("launcher-adoption.xml")
    later.write_text("<testsuite/>\n")
    with pytest.raises(ValueError): module.verify_failed_cpu02()
    later.unlink()
    assert module.verify_failed_cpu02()
    # Correct hashes cannot convert historical CPU success to failure or its
    # failed callback to success. Rehashed outcome changes remain rejected.
    for name, changed in (
        ("cpu-03/stdout.29798769.log", "Subject: Job 29798769: old Exited\nExited with exit code 1.\n"),
        ("after-cpu-03/stdout.29798770.log", "Subject: Job 29798770: old Done\nSuccessfully completed.\n"),
        ("after-cpu-03/stderr.29798770.log", "ValueError: different failure\n"),
    ):
        path = module.ROOT / name
        raw, digest = path.read_bytes(), module.CPU03_HISTORY_PINS[name]
        path.write_text(changed)
        module.CPU03_HISTORY_PINS[name] = module.sha(path)
        with pytest.raises(ValueError): module.verify_cpu03_history()
        path.write_bytes(raw); module.CPU03_HISTORY_PINS[name] = digest
    history_xml = module.ROOT / "cpu-03/payload/dbmol-transport.xml"
    raw = history_xml.read_bytes()
    pin_name = "cpu-03/payload/dbmol-transport.xml"
    digest = module.CPU03_HISTORY_PINS[pin_name]
    tree = ET.parse(history_xml)
    ET.SubElement(next(tree.iter("testcase")), "failure", message="invented failure")
    tree.write(history_xml, encoding="utf-8", xml_declaration=True)
    module.CPU03_HISTORY_PINS[pin_name] = module.sha(history_xml)
    with pytest.raises(ValueError): module.verify_cpu03_history()
    history_xml.write_bytes(raw); module.CPU03_HISTORY_PINS[pin_name] = digest
    assert module.verify_cpu03_history()


@pytest.mark.parametrize("fault", ("changed_bytes", "missing_proof", "relabelled_gate", "wrong_job", "wrong_lsf", "wrong_exit", "old_pytest_xml"))
def test_historical_failed_cpu_cannot_be_overwritten_relabelled_or_used_as_a_pass(failed_cpu, fault):
    module, pins = failed_cpu
    if fault == "old_pytest_xml":
        path = module.ROOT / "cpu" / "payload" / "made_up.xml"
        path.parent.mkdir(parents=True)
        path.write_text("<testsuite/>\n")
    elif fault in ("changed_bytes", "missing_proof", "relabelled_gate"):
        name = "cpu/storage-cpu-entry/gate.json"
        path = module.ROOT / name
        if fault == "changed_bytes": path.write_text(path.read_text() + " ")
        elif fault == "missing_proof": path.unlink()
        else:
            write(path, {"status": "passed"}); pins[name] = module.sha(path)
    elif fault == "wrong_job":
        name = "after-cpu/submission.json"; path = module.ROOT / name
        write(path, {"job_id": "999", "outcome": "submitted"}); pins[name] = module.sha(path)
    else:
        name = "cpu/stdout.29798549.log"; path = module.ROOT / name
        if fault == "wrong_lsf": path.write_text("Subject: Job 29798549: old Done\nSuccessfully completed.\n")
        else: path.write_text("Subject: Job 29798549: old Exited\nExited with exit code 2.\n")
        pins[name] = module.sha(path)
    with pytest.raises((ValueError, OSError)):
        module.verify_failed_old_cpu()
