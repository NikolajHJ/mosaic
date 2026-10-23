"""Immutable byte/metadata and no-duplicate operator contracts, run on HPC.

No real scheduler, SSH, quota, training or external model operation occurs.
Filesystem fixtures live entirely below pytest's temporary directory.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest


def operator_module():
    location = Path(__file__).resolve().parents[1] / "scripts/hpc/embedding_operator.py"
    spec = importlib.util.spec_from_file_location("embedding_operator_under_test", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sealed_fixture(operator, release):
    release.mkdir(parents=True)
    plan = release / "configs/epoch36_boltz_embedding_bridge.json"
    plan.parent.mkdir()
    plan.write_text('{"version":"test-sealed-plan"}\n')
    member = release / "src/example.py"
    member.parent.mkdir()
    member.write_text('IMMUTABLE = "original"\n')
    manifest = release / "embedding-release-manifest.json"
    manifest.write_text(json.dumps({"version": 1, "members": {
        "configs/epoch36_boltz_embedding_bridge.json": operator.digest(plan),
        "src/example.py": operator.digest(member)}}))
    archive = Path(str(release) + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        for path in (plan, member, manifest):
            stream.add(path, arcname=path.relative_to(release).as_posix(), recursive=False)
    return SimpleNamespace(release=release, plan=plan, member=member, manifest=manifest,
                           archive=archive, archive_sha=operator.digest(archive),
                           plan_sha=operator.digest(plan))


def test_authentication_binds_extracted_manifest_to_original_archive(tmp_path):
    operator = operator_module()
    sealed = sealed_fixture(operator, tmp_path / "release")
    operator.authenticate(sealed.release, sealed.archive_sha, sealed.plan_sha)
    # Updating a changed member's hash in the extracted manifest used to make
    # the pair self-consistent while leaving the pinned archive hash unchanged.
    sealed.member.write_text('IMMUTABLE = "silently changed"\n')
    rewritten = json.loads(sealed.manifest.read_text())
    rewritten["members"]["src/example.py"] = operator.digest(sealed.member)
    sealed.manifest.write_text(json.dumps(rewritten))
    assert operator.digest(sealed.archive) == sealed.archive_sha
    with pytest.raises(ValueError, match="manifest differs from authenticated archive"):
        operator.authenticate(sealed.release, sealed.archive_sha, sealed.plan_sha)


def test_individual_member_tamper_cannot_pass_unchanged_manifest(tmp_path):
    operator = operator_module()
    sealed = sealed_fixture(operator, tmp_path / "release")
    sealed.member.write_text('IMMUTABLE = "changed without receipt"\n')
    with pytest.raises(ValueError, match="Release member differs"):
        operator.authenticate(sealed.release, sealed.archive_sha, sealed.plan_sha)


@pytest.mark.parametrize("binding", ["archive", "plan"])
def test_wrong_expected_source_or_plan_binding_stops_authentication(tmp_path, binding):
    operator = operator_module()
    sealed = sealed_fixture(operator, tmp_path / "release")
    with pytest.raises(ValueError, match="archive hash" if binding == "archive" else "plan hash"):
        operator.authenticate(sealed.release, "0" * 64 if binding == "archive" else sealed.archive_sha,
                              "0" * 64 if binding == "plan" else sealed.plan_sha)


@pytest.mark.parametrize("existing", ["cpu-intent.json", "cpu-receipt.json"])
def test_prior_submission_intent_or_receipt_blocks_all_scheduler_operations(tmp_path, monkeypatch, existing):
    operator = operator_module()
    work = tmp_path / "isolated-work-root"
    sealed = sealed_fixture(operator, work / "releases/test-release")
    campaign = work / "runs/source137-epoch36-boltz-embedding-bridge-attempt999"
    control = work / "submissions" / campaign.name
    control.mkdir(parents=True)
    original = '{"status":"prior ambiguous or confirmed submission"}\n'
    (control / existing).write_text(original)
    original_path = operator.Path

    def isolated_path(value):
        if str(value) == "/work3/s234818/latentmol":
            return work
        return original_path(value)

    calls = []

    def forbidden_scheduler(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Duplicate attempt must stop before quota or scheduler calls")

    monkeypatch.setattr(operator, "Path", isolated_path)
    monkeypatch.setattr(operator.subprocess, "run", forbidden_scheduler)
    monkeypatch.setattr(operator.sys, "argv", ["embedding_operator", "submit", "--stage", "cpu",
        "--release", str(sealed.release), "--archive-sha256", sealed.archive_sha,
        "--plan-sha256", sealed.plan_sha, "--campaign", str(campaign)])
    with pytest.raises(FileExistsError, match="already attempted"):
        operator.main()
    assert not calls
    assert (control / existing).read_text() == original


def test_exclusive_receipt_write_never_overwrites_earlier_evidence(tmp_path):
    operator = operator_module()
    receipt = tmp_path / "intent.json"
    operator.write_new(receipt, {"identity": "first attempt"})
    original = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        operator.write_new(receipt, {"identity": "replacement"})
    assert receipt.read_bytes() == original


@pytest.mark.parametrize("returncode,stdout", [
    (1, ""),
    (0, "quota output cannot be parsed"),
    (0, "s234818|306121||   95.01 GiB|  100.00 GiB|| 1| 20000000\n"),
])
def test_quota_failure_or_insufficient_headroom_stops_submission(monkeypatch, returncode, stdout):
    operator = operator_module()
    monkeypatch.setattr(operator.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=returncode, stdout=stdout, stderr="quota detail"))
    with pytest.raises(RuntimeError, match="quota"):
        operator.quota()


def test_quota_records_fresh_output_and_minimum_available_headroom(monkeypatch):
    operator = operator_module()
    stdout = ("s234818|306121||      0 Byte|   unlimited|| 0|unlimited\n"
              "s234818|306121||   86.50 GiB|  100.00 GiB|| 289250| 20000000\n")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(operator.subprocess, "run", run)
    receipt = operator.quota()
    assert calls[0][0] == ["getquota_work3.sh"]
    assert calls[0][1]["timeout"] == 60
    assert receipt["minimum_free_gib"] == 13.5 and receipt["stdout"] == stdout
