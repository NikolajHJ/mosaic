"""Allocated metadata fixtures for runtime-only adoption and serial callbacks."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
import chemistry_campaign
import chemistry_runtime_continuation as controller
from latentmol.experiments import file_sha256, write_json


def test_all_existing_and_new_contract_modules_remain_in_the_cpu_command(tmp_path):
    command = controller.cpu_command("python", tmp_path)
    assert command[:7] == ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml"]
    for module in (*controller.FIXTURES, *controller.OPTIMIZED_FIXTURES,
                   *controller.EXTRA_FIXTURES, *controller.SPEED_FIXTURES,
                   *controller.AMENDMENT_FIXTURES):
        assert module in command


def test_serial_callbacks_keep_the_runtime_adoption_controller(tmp_path):
    campaign = object.__new__(controller.RuntimeContinuation)
    campaign.work, campaign.release, campaign.path, campaign.sha = tmp_path, tmp_path / "release", tmp_path / "plan", "a" * 64
    command = campaign.controller("advance", after_stage="segment-001")
    assert "scripts/hpc/chemistry_runtime_continuation.py" in command
    assert command[-2:] == ["--after-stage", "segment-001"]
    assert "PYTHONPATH=" + str(campaign.release / "src") in command


def test_after_cpu_cannot_submit_before_its_checks_pass():
    campaign = object.__new__(controller.RuntimeContinuation)
    calls = []
    campaign.require_cpu = lambda: calls.append("checked")
    campaign.advance = lambda stage: calls.append(stage)
    campaign.after_cpu()
    assert calls == ["checked", "benchmark"]


def test_failed_cpu_check_does_not_call_advance():
    campaign = object.__new__(controller.RuntimeContinuation)
    calls = []
    def failed():
        raise ValueError("fixture failed CPU validation")
    campaign.require_cpu, campaign.advance = failed, lambda stage: calls.append(stage)
    with pytest.raises(ValueError, match="failed CPU"):
        campaign.after_cpu()
    assert calls == []


def _cpu_fixture(tmp_path, monkeypatch):
    from latentmol import chemistry_runtime_adoption
    campaign = object.__new__(controller.RuntimeContinuation)
    campaign.work, campaign.root, campaign.sha = tmp_path, tmp_path / "run", "a" * 64
    campaign.root.mkdir()
    campaign.plan = {"runtime_adoption": {"version": "fixture"}}
    campaign.identity = {"plan_sha256": campaign.sha}
    campaign.complete = lambda stage: None
    campaign.receipt_job = lambda stage: "123"
    xml = campaign.root / "runtime-amendment-pytest.xml"
    xml.write_text('<testsuites><testsuite tests="10" failures="0" errors="0" skipped="2" /></testsuites>')
    gate = dict(status="passed", returncode=0, job_id="123", identity=campaign.identity,
        command=controller.cpu_command(str(tmp_path / "env/boltz311/bin/python"), campaign.root),
        junit_sha256=file_sha256(xml))
    write_json(campaign.root / "cpu_gate.json", gate)
    write_json(campaign.root / "failure_readiness.json", {"fixture": True})
    calls = []
    monkeypatch.setattr(chemistry_runtime_adoption, "authenticate_adoption", lambda *args: calls.append(args))
    return campaign, gate, xml, calls


def test_cpu_success_always_reauthenticates_the_additive_adoption_receipt(tmp_path, monkeypatch):
    campaign, _, _, calls = _cpu_fixture(tmp_path, monkeypatch)
    campaign.require_cpu()
    assert len(calls) == 1 and calls[0][0]["_plan_sha256"] == campaign.sha


@pytest.mark.parametrize("key", ["status", "returncode", "job_id", "identity", "command", "junit_sha256"])
def test_cpu_receipt_mutation_blocks_training(tmp_path, monkeypatch, key):
    campaign, gate, _, calls = _cpu_fixture(tmp_path, monkeypatch)
    gate[key] = "changed"
    write_json(campaign.root / "cpu_gate.json", gate)
    with pytest.raises(ValueError):
        campaign.require_cpu()
    assert calls == []


@pytest.mark.parametrize("attribute,value", [("failures", "1"), ("errors", "1"), ("tests", "2")])
def test_failed_or_entirely_skipped_cpu_fixtures_block_training(tmp_path, monkeypatch, attribute, value):
    campaign, gate, xml, calls = _cpu_fixture(tmp_path, monkeypatch)
    attributes = dict(tests="10", failures="0", errors="0", skipped="2")
    attributes[attribute] = value
    xml.write_text('<testsuites><testsuite ' + ' '.join(k + '="' + v + '"' for k, v in attributes.items()) + ' /></testsuites>')
    gate["junit_sha256"] = file_sha256(xml)
    write_json(campaign.root / "cpu_gate.json", gate)
    with pytest.raises(ValueError):
        campaign.require_cpu()
    assert calls == []
