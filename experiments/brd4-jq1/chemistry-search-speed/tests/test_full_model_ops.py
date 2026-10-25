"""Allocated stdlib fixtures for exact full-model orchestration/admission scope.

Scheduler/LSF rows below are synthetic parser fixtures, never scientific proof.
No fixture submits a real job, imports model code or changes a saved study.
"""

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import full_model_ops as ops


PIN = "a" * 64


def raw(state="DONE", *, job="10", name="mosaic-chem-full03-cpu", queue="hpc", code=0):
    return {"returncode": code, "stdout": f"Job <{job}>, Job Name <{name}>, Status <{state}>, Queue <{queue}>\n",
            "stderr": ""}


def lsf(*, job="10", name="mosaic-chem-full03-cpu", queue="hpc", status="Done"):
    return (f"Subject: Job {job}: <{name}> in cluster <dcc> {status}\n"
            f"Job was executed on host <node> in queue <{queue}>.\nSuccessfully completed.\n")


@pytest.fixture
def roots(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    source = tmp_path / "release/source"
    source.mkdir(parents=True)
    monkeypatch.setattr(ops, "ROOT", root)
    monkeypatch.setattr(ops, "SOURCE", source)
    monkeypatch.setattr(ops, "PRIOR_OWNED_ROOTS", ())
    return root, source


@pytest.mark.parametrize("phase,slots,queue,memory", (
    ("cpu", "16", "hpc", "rusage[mem=2GB]"),
    ("after-cpu", "1", "hpc", "rusage[mem=2GB]"),
    ("protenix", "8", "gpul40s", "rusage[mem=4GB]"),
    ("boltz", "8", "gpul40s", "rusage[mem=4GB]"),
))
def test_exact_resource_vectors(phase, slots, queue, memory):
    value = ops.resources(phase)
    assert value[value.index("-q") + 1] == queue
    assert value[value.index("-n") + 1] == slots
    assert memory in value
    assert value[value.index("-W") + 1] == ("00:20" if phase == "after-cpu" else "02:00")
    assert ("num=1:mode=exclusive_process" in value) == (phase == "protenix")
    assert ("num=1:mode=shared:j_exclusive=yes:mps=no" in value) == (phase == "boltz")


@pytest.mark.parametrize("phase", ("cpu", "protenix", "boltz"))
def test_actual_argv_has_no_cross_arm_dependency(roots, phase):
    _, source = roots
    command = ops.argv(phase, PIN)
    assert command[-6:] == ["bash", "-l", str(source / "scripts/full_model_stage.lsf"), str(source), PIN, phase]
    assert "-w" not in command
    assert "mosaic-chem-full03-" + phase in command
    assert "production" not in " ".join(command)


def test_callback_requires_exact_ended_dependency():
    command = ops.argv("after-cpu", PIN, "ended(10)")
    assert command[command.index("-w") + 1] == "ended(10)"


@pytest.mark.parametrize("phase,dependency", (
    ("after-cpu", None), ("after-cpu", "done(10)"),
    ("boltz", "ended(10)"), ("protenix", "ended(0)"),
))
def test_undeclared_dependencies_fail(phase, dependency):
    with pytest.raises(ValueError):
        ops.argv(phase, PIN, dependency)


def test_receipts_are_exclusive(tmp_path):
    path = tmp_path / "receipt.json"
    ops.write(path, {"status": "preserved"})
    with pytest.raises(FileExistsError):
        ops.write(path, {"status": "replaced"})
    assert json.loads(path.read_text())["status"] == "preserved"


@pytest.mark.parametrize("changed", ("none", "member", "manifest", "symlink"))
def test_source_authentication_rejects_changed_bytes_and_aliases(tmp_path, changed):
    source = tmp_path / "source"
    source.mkdir()
    member = source / "member.txt"
    member.write_text("exact")
    manifest = {"files": {"member.txt": sha256(member.read_bytes()).hexdigest()}}
    path = source / "source_manifest.json"
    ops.write(path, manifest)
    pin = sha256(path.read_bytes()).hexdigest()
    if changed == "member":
        member.write_text("changed")
    elif changed == "manifest":
        path.write_text("{}")
    elif changed == "symlink":
        member.unlink()
        target = tmp_path / "elsewhere.txt"
        target.write_text("exact")
        member.symlink_to(target)
    if changed == "none":
        assert ops.verify(source, pin) == manifest
    else:
        with pytest.raises(ValueError):
            ops.verify(source, pin)


@pytest.mark.parametrize("kind", ("passed", "failure", "error", "skipped", "wrongmodule", "duplicate"))
def test_raw_xml_has_exact_attribution_and_no_hidden_failure(tmp_path, kind):
    child = "" if kind in ("passed", "wrongmodule", "duplicate") else f"<{kind}/>"
    module = "tests.wrong" if kind == "wrongmodule" else "tests.fixture"
    case = f'<testcase classname="{module}" name="a">{child}</testcase>'
    path = tmp_path / "tests.xml"
    path.write_text("<testsuites><testsuite>" + case * (2 if kind == "duplicate" else 1) + "</testsuite></testsuites>")
    if kind == "passed":
        assert ops.xml_cases(path, {"tests.fixture": 1}) == {"tests.fixture": 1}
    else:
        with pytest.raises(ValueError):
            ops.xml_cases(path, {"tests.fixture": 1})


@pytest.mark.parametrize("state", ("DONE", "RUN", "PEND"))
def test_actual_terminal_parser_returns_observed_state_only(state):
    assert ops._terminal_status(raw(state), "10", "mosaic-chem-full03-cpu") == state


@pytest.mark.parametrize("change", ("job", "queue", "name", "exit", "query"))
def test_terminal_parser_rejects_other_job_queue_name_failure(change):
    options = {"job": "11"} if change == "job" else {"queue": "gpul40s"} if change == "queue" else {"name": "other"} if change == "name" else {"code": 255} if change == "query" else {}
    value = raw("EXIT" if change == "exit" else "DONE", **options)
    with pytest.raises(ValueError):
        ops._terminal_status(value, "10", "mosaic-chem-full03-cpu")


def test_bounded_terminal_wait_records_raw_running_before_actual_done():
    values = iter([raw("RUN"), raw("PEND"), raw("DONE")])
    records, pauses = [], []
    final = ops.wait_cpu_terminal(lambda job: next(values), "10", record=lambda index, value: records.append((index, value)), pause=lambda: pauses.append(True))
    assert final == raw("DONE")
    assert [value[1]["stdout"] for value in records] == [raw("RUN")["stdout"], raw("PEND")["stdout"], raw("DONE")["stdout"]]
    assert len(pauses) == 2


def test_bounded_running_timeout_never_becomes_success():
    records = []
    with pytest.raises(TimeoutError):
        ops.wait_cpu_terminal(lambda job: raw("RUN"), "10", maximum=2,
                              record=lambda index, value: records.append(value), pause=lambda: None)
    assert records == [raw("RUN"), raw("RUN")]


@pytest.mark.parametrize("change", ("none", "duplicate", "name", "queue", "failure"))
def test_unique_same_job_lsf_success(change):
    text = lsf()
    if change == "duplicate":
        text += lsf()
    elif change == "name":
        text = lsf(name="other")
    elif change == "queue":
        text = lsf(queue="gpul40s")
    elif change == "failure":
        text = lsf(status="Exited")
    if change == "none":
        assert ops._lsf_success(text, "10", "mosaic-chem-full03-cpu") is True
    else:
        with pytest.raises(ValueError):
            ops._lsf_success(text, "10", "mosaic-chem-full03-cpu")


def test_actual_submission_records_unique_id_and_prevents_replay(roots, monkeypatch):
    root, _ = roots
    ops.write(root / "cpu-certificate.json", {"status": "passed", "source_manifest_sha256": PIN})
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="Job <123> is submitted to queue <gpul40s>.\n" if command[0] == "bsub" else raw()["stdout"], stderr="")
    monkeypatch.setattr(ops.subprocess, "run", execute)
    result = ops.submit("protenix", PIN)
    assert result["job_id"] == "123" and result["status"] == "submitted"
    assert json.loads((root / "jobs.json").read_text()) == {"protenix": "123"}
    assert (root / "protenix/submission-started.json").exists()
    with pytest.raises(FileExistsError):
        ops.submit("protenix", PIN)
    assert len([command for command in calls if command[0] == "bsub"]) == 1


def test_unknown_submission_outcome_keeps_exclusive_marker(roots, monkeypatch):
    root, _ = roots
    ops.write(root / "cpu-certificate.json", {"status": "passed", "source_manifest_sha256": PIN})
    monkeypatch.setattr(ops.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="transport truncated", stderr=""))
    with pytest.raises(ValueError, match="Unknown submission"):
        ops.submit("boltz", PIN)
    assert json.loads((root / "boltz/submission.json").read_text())["status"] == "unknown"
    assert not (root / "jobs.json").exists()
    with pytest.raises(FileExistsError):
        ops.submit("boltz", PIN)


def test_owned_cap_counts_internal_alias_metadata_without_traversal(roots):
    root, source = roots
    member = root / "owned.bin"
    member.write_bytes(b"1234")
    (source / "owned.txt").write_bytes(b"56")
    alias = root / "alias"
    alias.symlink_to(member)
    assert ops.own_bytes() == 6 + alias.lstat().st_size


def test_owned_cap_rejects_alias_to_other_project(roots, tmp_path):
    root, _ = roots
    other = tmp_path / "other-project"
    other.write_text("protected")
    (root / "alias").symlink_to(other)
    with pytest.raises(ValueError, match="external alias"):
        ops.own_bytes()


@pytest.mark.parametrize("races",[1,3])
def test_owned_atomic_rename_retries_entire_inventory_and_never_omits_bytes(roots,monkeypatch,races):
    root,_=roots;member=root/"prepared_context.json.tmp";member.write_bytes(b"1234")
    original=Path.lstat;calls=[]
    def turnover(path,*args,**kwargs):
        if path==member and len(calls)<races:
            calls.append(True);raise FileNotFoundError("Fixture atomic rename during entire scan")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"lstat",turnover)
    if races==1:assert ops.own_bytes()==4
    else:
        with pytest.raises(FileNotFoundError):ops.own_bytes()
    assert len(calls)==races


@pytest.mark.parametrize("phase,slots,queue", (
    ("cpu", "16", "hpc"), ("after-cpu", "1", "hpc"),
    ("protenix", "8", "gpul40s"), ("boltz", "8", "gpul40s"),
))
def test_own_phase_checks_actual_command_and_resource_identity(roots, monkeypatch, phase, slots, queue):
    root, _ = roots
    parent = root / phase
    parent.mkdir()
    dependency = "ended(9)" if phase == "after-cpu" else None
    record = {"status": "submitted", "job_id": "10", "manifest_sha256": PIN,
              "dependency": dependency, "argv": ops.argv(phase, PIN, dependency)}
    if phase in ("protenix", "boltz"):
        ops.write(root / "cpu-certificate.json", {"status": "passed", "source_manifest_sha256": PIN})
        record["cpu_certificate_sha256"] = ops.sha(root / "cpu-certificate.json")
    ops.write(parent / "submission.json", record)
    if phase == "after-cpu":
        (root / "cpu").mkdir()
        ops.write(root / "cpu/submission.json", {"job_id": "9"})
    monkeypatch.setenv("LSB_JOBID", "10")
    monkeypatch.setenv("LSB_QUEUE", queue)
    monkeypatch.setenv("LSB_DJOB_NUMPROC", slots)
    assert ops.own(phase, PIN) == record
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "99")
    with pytest.raises(ValueError, match="CPU slots"):
        ops.own(phase, PIN)


@pytest.mark.parametrize("change", ("none", "xml", "query", "plan"))
def test_persisted_new_cpu_certificate_rehashes_actual_proofs_without_live_query(roots, change):
    root, _ = roots
    (root / "cpu/payload").mkdir(parents=True)
    (root / "after-cpu").mkdir()
    ops.write(root / "cpu/submission.json", {
        "status": "submitted", "job_id": "10", "manifest_sha256": PIN,
        "argv": ops.argv("cpu", PIN),
    })
    xml = root / "cpu/payload/tests.xml"
    xml.write_text('<testsuites><testsuite><testcase classname="tests.fixture" name="a"/></testsuite></testsuites>')
    manifest = {"cpu_counts": {"tests.fixture": 1}, "files": {
        "plans/protenix.json": "b" * 64, "plans/boltz.json": "c" * 64,
    }}
    ops.write(root / "cpu/payload/assessment.json", {
        "status": "passed", "job_id": "10", "source_manifest_sha256": PIN,
        "cpu_counts": manifest["cpu_counts"], "raw_xml_sha256": ops.sha(xml),
    })
    (root / "cpu/stdout.10.log").write_text(lsf())
    query = root / "after-cpu/cpu-terminal-query-000.json"
    ops.write(query, raw("DONE"))
    proof = ops.certify_cpu(PIN, manifest, {"job_id": "10"})
    expected_sha = proof["sha256"]
    if change == "xml":
        xml.write_text("changed original raw proof")
    elif change == "query":
        query.write_text(json.dumps(raw("RUN")))
    elif change == "plan":
        manifest["files"]["plans/boltz.json"] = "d" * 64
    if change == "none":
        assert ops.verify_cpu_certificate(PIN, manifest, expected_sha) == proof
        assert proof["actual_terminal_states"] == ["DONE"]
    else:
        with pytest.raises(ValueError):
            ops.verify_cpu_certificate(PIN, manifest, expected_sha)


def test_cli_sigterm_becomes_teardown_exception_and_restores_previous_handler():
    previous = ops.signal.getsignal(ops.signal.SIGTERM)
    with pytest.raises(InterruptedError, match="received SIGTERM"):
        with ops.termination_guard():
            assert ops.signal.getsignal(ops.signal.SIGTERM) is not previous
            ops.signal.raise_signal(ops.signal.SIGTERM)
    assert ops.signal.getsignal(ops.signal.SIGTERM) is previous
