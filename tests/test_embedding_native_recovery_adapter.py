"""Allocated metadata-gate tests: tampered CLI proof cannot authorize prediction."""
import json

import pytest

from latentmol.boltz import embedding_native_recovery_experiment as adapter


def saved_gate(tmp_path):
    cpu = tmp_path / "cpu"
    (cpu / "validation").mkdir(parents=True)
    xml = cpu / "cpu-tests.xml"
    xml.write_text('<testsuites><testsuite><testcase name="engineering"/></testsuite></testsuites>')
    proof = cpu / "validation/result.json"
    proof.write_text(json.dumps(dict(status="passed_authentication_not_prediction",
        source_sha256="a"*64, plan_sha256="b"*64)))
    record = dict(status="passed", source_sha256="a"*64, plan_sha256="b"*64,
        junit_sha256=adapter.digest(xml), validation_sha256=adapter.digest(proof))
    (cpu / "result.json").write_text(json.dumps(record))
    return cpu, record


def test_exact_completed_cli_and_fixture_gate_is_eligible(tmp_path):
    cpu, record = saved_gate(tmp_path)
    assert adapter.require_cpu_gate(cpu, "a"*64, "b"*64) == record


@pytest.mark.parametrize("change", ["missing", "bytes", "status", "source", "plan", "junit-skip"])
def test_changed_cli_proof_or_fixture_gate_cannot_release_gpu(tmp_path, change):
    cpu, record = saved_gate(tmp_path)
    proof = cpu / "validation/result.json"
    if change == "missing":
        proof.unlink()
    elif change == "bytes":
        with proof.open("a") as stream:stream.write(" ")
    elif change == "junit-skip":
        xml = cpu / "cpu-tests.xml"
        xml.write_text('<testsuites><testsuite><testcase name="engineering"><skipped/></testcase></testsuite></testsuites>')
        record["junit_sha256"] = adapter.digest(xml)
    else:
        observed = json.loads(proof.read_text())
        observed[{"status":"status", "source":"source_sha256", "plan":"plan_sha256"}[change]] = "changed"
        proof.write_text(json.dumps(observed))
        # An internally consistent edited receipt must still fail its declared
        # source/plan/status contract, not just an incidental stale checksum.
        record["validation_sha256"] = adapter.digest(proof)
    (cpu / "result.json").write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.require_cpu_gate(cpu, "a"*64, "b"*64)
