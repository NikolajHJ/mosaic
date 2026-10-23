"""Metadata-only speed-suite provenance/promotion fixtures; allocated HPC only.

All allocation IDs, scheduler replies, native gates and source plans below are
synthetic temporary evidence. No live scheduler, model, checkpoint or numerical
benchmark is invoked; passing these tests does not establish a selected run.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from latentmol.chemistry_speed_selection import REQUIRED_GATES, REQUIRED_PARITY_CHECKS, choose_candidate


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/hpc"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("_chemistry_speed_campaign_fixture", SCRIPTS / "chemistry_speed_campaign.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _suite(tmp_path, monkeypatch):
    """Bypass external release discovery, preserving actual metadata readers."""
    suite = module.SpeedSuite.__new__(module.SpeedSuite)
    suite.work = tmp_path / "work"
    suite.release = suite.work / "releases" / "source116"
    suite.root = suite.work / "runs" / "suite"
    suite.path = suite.work / "submissions" / "suite.json"
    suite.receipts = suite.root / "receipts"
    suite.release.mkdir(parents=True)
    suite.root.mkdir(parents=True)
    suite.sha = "a" * 64
    suite.identity = {"plan_sha256": suite.sha, "archive_sha256": "b" * 64,
                      "manifest_sha256": "c" * 64, "parent_sha256": "d" * 64}
    suite.plan = {"work_root": str(suite.work), "release": str(suite.release), "campaign": str(suite.root),
                  "archive_sha256": "b" * 64, "manifest_sha256": "c" * 64,
                  "parent_checkpoint_sha256": "d" * 64, "config_sha256": "e" * 64,
                  "dataset_sha256": "f" * 64, "split_sha256": "1" * 64,
                  "parent_epoch": 35, "parent_updates": 230860, "training_molecules": 1688539,
                  "selection_threshold_seconds": 36000, "suite_gpu_wall_seconds": 19800,
                  "execution": {"backend": "optimized", "workers": 28, "torch_threads": 4,
                                "gpu_cpus": 32, "overlap": True}}
    suite.variants = []
    for name, ratio, workers, microbatch, overlap in (("r3_12_128_sync", 3, 12, 128, False),
            ("r3_28_256_async", 3, 28, 256, True), ("r2_12_128_async", 2, 12, 128, True)):
        plan_path = suite.work / "submissions" / (name + ".json")
        plan = {**deepcopy(suite.plan), "campaign": str(suite.work / "runs" / name),
                "speed_variant_id": name, "speed_suite_root": str(suite.root),
                "protocol": {"prior_ratio": ratio, "benchmark_updates": 2, "microbatch_size": microbatch},
                "execution": {"backend": "optimized", "workers": workers, "torch_threads": 4,
                              "gpu_cpus": 32, "overlap": overlap}}
        _json(plan_path, plan)
        suite.variants.append({"id": name, "prior_ratio": ratio, "workers": workers,
            "microbatch_size": microbatch, "overlap": overlap, "benchmark_updates": 2,
            "plan": str(plan_path), "plan_sha256": module.file_sha256(plan_path)})
    suite.plan["variants"] = suite.variants
    _json(suite.path, suite.plan)
    monkeypatch.setattr(suite, "receipt_job", lambda stage: {"cpu": "11", "benchmark": "22"}[stage])
    monkeypatch.setattr(suite, "complete", lambda stage: suite.receipt_job(stage))
    return suite


def _cpu(suite):
    profile = {"status": "passed", "job_id": "11",
               "identity": {**suite.identity, "plan_sha256": suite.variants[0]["plan_sha256"]}}
    path = suite.root / "repair_profile.json"
    _json(path, profile)
    gate = {"status": "passed", "returncode": 0, "profile_returncode": 0,
            "identity": deepcopy(suite.identity), "job_id": "11",
            "candidate_plan_sha256": {row["id"]: row["plan_sha256"] for row in suite.variants},
            "profile_sha256": module.file_sha256(path), "profile_plan_sha256": suite.variants[0]["plan_sha256"]}
    _json(suite.root / "cpu_gate.json", gate)
    return gate, profile


def _native(suite, variant, *, update_seconds=4., status="passed"):
    plan = module.read(variant["plan"])
    root = Path(plan["campaign"])
    run = root / "benchmark" / "native"
    gate = {"status": status, "plan_sha256": variant["plan_sha256"],
            "parent_sha256": suite.plan["parent_checkpoint_sha256"], "run_dir": str(run),
            "execution": deepcopy(plan["execution"]), "gates": {key: True for key in REQUIRED_GATES},
            "native_execution_parity": {"status": "passed", "passed": True,
                "restoration": {"passed": True}, "checks": {key: True for key in REQUIRED_PARITY_CHECKS}},
            "updates": [{"update_seconds": update_seconds, "real_points": 256,
                "fresh_prior_points": 256 * variant["prior_ratio"]} for _ in range(variant["benchmark_updates"])],
            "projected_epoch_seconds": update_seconds * suite.plan["training_molecules"] / 256 + 600.}
    record = {"status": "completed", "environment": {"job_id": "22"},
              "source_archive_sha256": suite.plan["archive_sha256"],
              "source": {"sha256": suite.plan["manifest_sha256"]},
              "config": {"_plan_sha256": variant["plan_sha256"]}}
    _json(root / "benchmark_gate.json", gate)
    _json(run / "gate.json", gate)
    _json(run / "run.json", record)
    return root, gate, record


def _selection(suite, rows, **changes):
    selection = choose_candidate(rows, preferred_epoch_seconds=suite.plan["selection_threshold_seconds"],
                                 fallback_evaluated=any(row["prior_ratio"] == 2 for row in rows))
    cuda = {"identity": deepcopy(suite.identity), "job_id": "22", "status": "passed", "returncode": 0,
            "command": ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1", "python", "-m", "pytest",
                        "-q", "tests/test_chemistry_resume_cuda.py"]}
    _json(suite.root / "cuda-fixtures.json", cuda)
    result = {"identity": deepcopy(suite.identity), "observations": rows, "selection": selection,
              "cuda_fixture_gate_sha256": module.file_sha256(suite.root / "cuda-fixtures.json"),
              "benchmark_job_id": "22", "training_started": False}
    result.update(changes)
    _json(suite.root / "selection.json", result)
    _json(suite.root / "selection.sha256.json", {"sha256": module.file_sha256(suite.root / "selection.json")})
    return result


def test_shared_cpu_requires_all_candidate_plans_and_exact_source_parent_profile(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    _cpu(suite)
    suite.require_cpu()
    # These are actual shared allocation receipts, not invented per-variant jobs.
    assert all(not (Path(module.read(row["plan"])["campaign"]) / "cpu_gate.json").exists()
               for row in suite.variants)


def test_suite_cpu_reserves_sixteen_slots_gpu_thirtytwo_and_callback_two(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    cpu = suite.arguments("cpu", limit="2:00")
    gpu = suite.arguments("benchmark", gpu=True, limit="6:00")
    callback = suite.arguments("after-benchmark")
    assert cpu[cpu.index("-n") + 1] == "16"
    assert gpu[gpu.index("-n") + 1] == "32"
    assert callback[callback.index("-n") + 1] == "2"
    assert "rusage[mem=4GB]" in cpu  # Sixteen per-slot requests reserve 64GB.
    assert cpu[cpu.index("-q") + 1] == callback[callback.index("-q") + 1] == "hpc"
    assert gpu[gpu.index("-q") + 1] == "gpua100"
    assert "-gpu" not in cpu and "-gpu" in gpu
    assert suite.plan["execution"]["gpu_cpus"] == 32
    assert all(module.read(row["plan"])["execution"]["gpu_cpus"] == 32 for row in suite.variants)


def test_cpu_fixtures_accept_sixteen_slots_and_still_parse_every_candidate_before_profile(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    monkeypatch.setenv("LSB_JOBID", "11")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "16")
    parsed, commands = [], []
    monkeypatch.setitem(sys.modules, "latentmol.chemistry_continuation", SimpleNamespace(
        read_plan=lambda path, sha: parsed.append((path, sha))))
    def run(command, **kwargs):
        commands.append(command)
        if "scripts/benchmark_atom_preserving_search.py" in command:
            _json(suite.root / "repair_profile.json", {"status": "passed", "job_id": "11",
                "identity": {**suite.identity, "plan_sha256": suite.variants[0]["plan_sha256"]}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", run)
    suite.cpu()
    assert parsed == [(row["plan"], row["plan_sha256"]) for row in suite.variants]
    assert len(commands) == 2
    assert all(value in commands[0] for value in [*module.FIXTURES, *module.OPTIMIZED_FIXTURES, *module.EXTRA_FIXTURES])
    assert commands[1][commands[1].index("--plan") + 1] == suite.variants[0]["plan"]
    gate = module.read(suite.root / "cpu_gate.json")
    assert gate["status"] == "passed" and gate["job_id"] == "11"
    assert gate["candidate_plan_sha256"] == {row["id"]: row["plan_sha256"] for row in suite.variants}
    suite.require_cpu()


@pytest.mark.parametrize("slots", ["", "8", "15", "17", "32", "16.0", "016"])
def test_cpu_fixtures_reject_wrong_resource_before_parsing_or_starting_subprocess(tmp_path, monkeypatch, slots):
    suite = _suite(tmp_path, monkeypatch)
    monkeypatch.setenv("LSB_JOBID", "11")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", slots)
    calls = []
    monkeypatch.setitem(sys.modules, "latentmol.chemistry_continuation", SimpleNamespace(
        read_plan=lambda *args: calls.append(args)))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="16-slot"):
        suite.cpu()
    assert calls == [] and not (suite.root / "cpu_gate.json").exists()


@pytest.mark.parametrize("field", ["release", "archive_sha256", "manifest_sha256", "parent_checkpoint_sha256",
    "config_sha256", "dataset_sha256", "split_sha256", "parent_epoch", "parent_updates", "speed_variant_id", "speed_suite_root"])
def test_constructor_rejects_candidate_escaping_common_source_parent_or_data(tmp_path, monkeypatch, field):
    original = _suite(tmp_path, monkeypatch)
    path = Path(original.variants[0]["plan"])
    plan = module.read(path)
    plan[field] = 36 if field == "parent_epoch" else 230861 if field == "parent_updates" else "wrong-bound-identity"
    _json(path, plan)
    def initialize(self, unused_path, unused_sha):
        for name in ("work", "release", "root", "path", "receipts", "sha", "identity", "plan"):
            setattr(self, name, deepcopy(getattr(original, name)))
    monkeypatch.setattr(module.Campaign, "__init__", initialize)
    monkeypatch.setattr(module, "validate_plan", lambda path, sha: module.read(path))
    with pytest.raises(ValueError, match="Candidate escaped"):
        module.SpeedSuite(original.path, original.sha)


@pytest.mark.parametrize("change", ["failed", "returncode", "profile_returncode", "job", "source", "parent",
    "missing_plan", "stale_plan", "profile_job", "profile_source", "profile_parent", "profile_plan", "profile_tamper"])
def test_shared_cpu_rejects_failed_stale_or_incomplete_allocation_evidence(tmp_path, monkeypatch, change):
    suite = _suite(tmp_path, monkeypatch)
    gate, profile = _cpu(suite)
    if change == "failed": gate["status"] = "failed"
    elif change == "returncode": gate["returncode"] = 1
    elif change == "profile_returncode": gate["profile_returncode"] = 1
    elif change == "job": gate["job_id"] = "12"
    elif change == "source": gate["identity"]["archive_sha256"] = "0" * 64
    elif change == "parent": gate["identity"]["parent_sha256"] = "0" * 64
    elif change == "missing_plan": gate["candidate_plan_sha256"].pop(suite.variants[1]["id"])
    elif change == "stale_plan": gate["candidate_plan_sha256"][suite.variants[0]["id"]] = "0" * 64
    elif change == "profile_job": profile["job_id"] = "12"
    elif change == "profile_source": profile["identity"]["manifest_sha256"] = "0" * 64
    elif change == "profile_parent": profile["identity"]["parent_sha256"] = "0" * 64
    elif change == "profile_plan": profile["identity"]["plan_sha256"] = suite.sha
    elif change == "profile_tamper": profile["status"] = "failed"
    if change.startswith("profile_") and change not in ("profile_returncode",):
        _json(suite.root / "repair_profile.json", profile)
        if change != "profile_tamper":
            gate["profile_sha256"] = module.file_sha256(suite.root / "repair_profile.json")
    _json(suite.root / "cpu_gate.json", gate)
    with pytest.raises(ValueError):
        suite.require_cpu()


def test_observe_gate_binds_real_shared_job_and_all_eighteen_native_gates(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    variant = suite.variants[0]
    _native(suite, variant)
    row = suite.observe_gate(variant, 0)
    assert row["eligible"] is True and set(row["gates"]) == REQUIRED_GATES
    assert row["native_gate_plan_sha256"] == variant["plan_sha256"]
    rejected = suite.observe_gate(variant, 1)
    assert rejected["eligible"] is False and "successfully" in rejected["rejection"]


@pytest.mark.parametrize("change", ["not_completed", "other_job", "archive", "manifest", "config_plan",
    "gate_plan", "parent", "execution", "gate_copy", "escaped_run", "failed_gate"])
def test_native_gate_cannot_be_borrowed_from_another_plan_source_or_allocation(tmp_path, monkeypatch, change):
    suite = _suite(tmp_path, monkeypatch)
    variant = suite.variants[0]
    root, gate, record = _native(suite, variant)
    run = Path(gate["run_dir"])
    if change == "not_completed": record["status"] = "failed"
    elif change == "other_job": record["environment"]["job_id"] = "23"
    elif change == "archive": record["source_archive_sha256"] = "0" * 64
    elif change == "manifest": record["source"]["sha256"] = "0" * 64
    elif change == "config_plan": record["config"]["_plan_sha256"] = suite.variants[1]["plan_sha256"]
    elif change == "gate_plan": gate["plan_sha256"] = suite.sha
    elif change == "parent": gate["parent_sha256"] = "0" * 64
    elif change == "execution": gate["execution"]["workers"] = 28
    elif change == "failed_gate": gate["status"] = "failed"
    elif change == "escaped_run":
        run = suite.root / "other-native-run"
        gate["run_dir"] = str(run)
    _json(run / "run.json", record)
    _json(root / "benchmark_gate.json", gate)
    _json(run / "gate.json", {**gate, "tampered": True} if change == "gate_copy" else gate)
    with pytest.raises(ValueError, match="Native benchmark"):
        suite.observe_gate(variant, 0)


def test_selected_recomputes_winner_and_reauthenticates_current_native_evidence(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    _cpu(suite)
    rows = []
    for variant in suite.variants[:2]:
        _native(suite, variant)
        rows.append(suite.observe_gate(variant, 0))
    _selection(suite, rows)
    variant, digest = suite.selected()
    assert variant == suite.variants[0] and digest == module.file_sha256(suite.root / "selection.json")
    root, gate, record = _native(suite, variant, update_seconds=3.)
    with pytest.raises(ValueError, match="changed after ranking"):
        suite.selected()


def test_selection_reauthenticates_passing_nonwinner_instead_of_only_selected_gate(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    _cpu(suite)
    rows = []
    for variant in suite.variants[:2]:
        _native(suite, variant)
        rows.append(suite.observe_gate(variant, 0))
    _selection(suite, rows)
    _native(suite, suite.variants[1], update_seconds=5.)
    with pytest.raises(ValueError, match="changed after ranking"):
        suite.selected()


@pytest.mark.parametrize("change", ["failed", "returncode", "job", "source", "parent", "bytes"])
def test_selection_requires_same_allocated_unskipped_cuda_fixture_receipt(tmp_path, monkeypatch, change):
    suite = _suite(tmp_path, monkeypatch)
    _cpu(suite)
    rows = []
    for variant in suite.variants[:2]:
        _native(suite, variant)
        rows.append(suite.observe_gate(variant, 0))
    saved = _selection(suite, rows)
    cuda = module.read(suite.root / "cuda-fixtures.json")
    if change == "failed": cuda["status"] = "failed"
    elif change == "returncode": cuda["returncode"] = 1
    elif change == "job": cuda["job_id"] = "23"
    elif change == "source": cuda["identity"]["archive_sha256"] = "0" * 64
    elif change == "parent": cuda["identity"]["parent_sha256"] = "0" * 64
    else: cuda["changed_bytes"] = True
    _json(suite.root / "cuda-fixtures.json", cuda)
    if change != "bytes":
        saved["cuda_fixture_gate_sha256"] = module.file_sha256(suite.root / "cuda-fixtures.json")
        _json(suite.root / "selection.json", saved)
        _json(suite.root / "selection.sha256.json", {"sha256": module.file_sha256(suite.root / "selection.json")})
    with pytest.raises(ValueError):
        suite.selected()


@pytest.mark.parametrize("change", ["identity", "job", "winner", "missing_candidate", "duplicate_candidate",
    "wrong_ratio", "partial_fallback", "tampered_pointer", "failed_scheduler"])
def test_selection_rejects_incomplete_tampered_or_unfinished_suite(tmp_path, monkeypatch, change):
    suite = _suite(tmp_path, monkeypatch)
    _cpu(suite)
    rows = []
    for variant in suite.variants:
        _native(suite, variant, update_seconds=8. if change == "partial_fallback" and variant["prior_ratio"] == 3 else 4.)
        rows.append(suite.observe_gate(variant, 0))
    if change != "partial_fallback":
        rows = rows[:2]
    if change == "missing_candidate": rows.pop(1)
    elif change == "duplicate_candidate": rows.append(deepcopy(rows[0]))
    elif change == "wrong_ratio": rows[1]["prior_ratio"] = 2
    elif change == "partial_fallback":
        # A fallback observation cannot stand in for the missing ratio-3 family.
        rows.pop(1)
    if change == "duplicate_candidate":
        # A corrupted stored selection may contain duplicate rows even though
        # the genuine producer cannot create it through choose_candidate.
        saved = _selection(suite, rows[:2])
        saved["observations"] = rows
        _json(suite.root / "selection.json", saved)
        _json(suite.root / "selection.sha256.json", {"sha256": module.file_sha256(suite.root / "selection.json")})
    else:
        saved = _selection(suite, rows)
    if change == "identity": saved["identity"]["parent_sha256"] = "0" * 64
    elif change == "job": saved["benchmark_job_id"] = "23"
    elif change == "winner": saved["selection"]["winner"] = suite.variants[1]["id"]
    if change in ("identity", "job", "winner", "tampered_pointer"):
        _json(suite.root / "selection.json", saved if change != "tampered_pointer" else {**saved, "tamper": True})
        if change != "tampered_pointer":
            _json(suite.root / "selection.sha256.json", {"sha256": module.file_sha256(suite.root / "selection.json")})
    elif change == "failed_scheduler":
        def complete(stage):
            if stage == "benchmark":
                raise ValueError("Shared native allocation EXIT, despite saved passed artifacts")
            return "11"
        monkeypatch.setattr(suite, "complete", complete)
    with pytest.raises(ValueError):
        suite.selected()


def test_candidate_delegates_shared_allocation_evidence_and_preserves_future_callback_binding(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    candidate = module.SpeedCandidate.__new__(module.SpeedCandidate)
    candidate.suite, candidate.variant = suite, suite.variants[1]
    candidate.root = Path(module.read(candidate.variant["plan"])["campaign"])
    candidate.sha = candidate.variant["plan_sha256"]
    jobs, completed = [], []
    monkeypatch.setattr(suite, "receipt_job", lambda stage: jobs.append(stage) or "shared-" + stage)
    monkeypatch.setattr(suite, "complete", lambda stage: completed.append(stage) or "done-" + stage)
    monkeypatch.setattr(module.Campaign, "receipt_job", lambda self, stage: "own-" + stage)
    monkeypatch.setattr(module.Campaign, "complete", lambda self, stage: "own-done-" + stage)
    assert candidate.receipt_job("benchmark") == "shared-benchmark"
    assert candidate.receipt_job("cpu") == "shared-cpu"
    assert candidate.complete("benchmark") == "done-benchmark"
    assert candidate.complete("cpu") == "done-cpu"
    assert candidate.receipt_job("segment-001") == "own-segment-001"
    assert candidate.complete("segment-001") == "own-done-segment-001"
    assert jobs == completed == ["benchmark", "cpu"]
    command = candidate.controller("advance", after_stage="segment-001")
    assert "scripts/hpc/chemistry_speed_campaign.py" in command
    assert command[command.index("--suite") + 1] == str(suite.path)
    assert command[command.index("--suite-sha256") + 1] == suite.sha
    assert command[command.index("--variant") + 1] == candidate.variant["id"]
    assert command[command.index("--after-stage") + 1] == "segment-001"


def test_only_selected_candidate_can_reuse_shared_cpu_provenance(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    candidate = module.SpeedCandidate.__new__(module.SpeedCandidate)
    candidate.suite, candidate.variant = suite, suite.variants[1]
    candidate.root = Path(module.read(candidate.variant["plan"])["campaign"])
    candidate.sha = candidate.variant["plan_sha256"]
    monkeypatch.setattr(suite, "selected", lambda: (suite.variants[0], "f" * 64))
    with pytest.raises(ValueError, match="Unselected"):
        candidate.require_cpu()
    assert not (candidate.root / "authenticated_selection.json").exists()
    monkeypatch.setattr(suite, "selected", lambda: (candidate.variant, "f" * 64))
    # __new__ bypasses Campaign.__init__, which creates this directory in every
    # real controller. Restore that fixture precondition before the write path.
    candidate.root.mkdir(parents=True, exist_ok=True)
    candidate.require_cpu()
    assert module.read(candidate.root / "authenticated_selection.json") == {
        "suite_sha256": suite.sha, "selection_sha256": "f" * 64,
        "variant_id": candidate.variant["id"], "plan_sha256": candidate.sha}


def test_suite_controller_retains_real_common_receipts_and_has_no_candidate_job_fabrication(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    queued = []
    monkeypatch.setattr(module, "quota_receipt", lambda minimum: {"minimum_gib": minimum})
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="EXIT", stderr="", returncode=0))
    monkeypatch.setattr(suite, "queue", lambda stage, command, **kwargs:
                        queued.append((stage, command, kwargs)) or str(100 + len(queued)))
    result = suite.queue_initial()
    assert [row[0] for row in queued] == ["cpu", "benchmark", "after-benchmark"]
    assert result["cpu"] == "101" and result["benchmark"] == "102" and result["promotion_callback"] == "103"
    assert queued[1][2]["dependency"] == "done(101)"
    assert queued[2][2]["dependency"] == "ended(102)"
    assert all("scripts/hpc/chemistry_speed_campaign.py" in row[1] for row in queued)
    assert result["status"] == "submitted_not_passed_no_training_yet"


@pytest.mark.parametrize("mode", ["preferred", "fallback", "failed_first", "all_failed", "cuda_failed"])
def test_native_suite_runs_sequential_children_and_only_declared_fallback_family(tmp_path, monkeypatch, mode):
    suite = _suite(tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "require_cpu", lambda: None)
    monkeypatch.setenv("LSB_JOBID", "22")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "32")
    for variant in suite.variants:
        _native(suite, variant, update_seconds=8. if mode in ("fallback", "all_failed") and variant["prior_ratio"] == 3 else 4.)
    spawned, fixtures, in_progress = [], [], []
    def cuda_run(command, **kwargs):
        fixtures.append(command)
        return SimpleNamespace(returncode=1 if mode == "cuda_failed" else 0)
    class Child:
        def __init__(self, command, **kwargs):
            assert not in_progress  # A second candidate cannot overlap this one.
            self.number = len(spawned)
            self.command, self.pid = command, 1234 + self.number
            spawned.append((command, kwargs))
            in_progress.append(self)
        def wait(self, timeout):
            assert 0 < timeout <= 6300
            assert in_progress.pop() is self
            return 1 if mode == "all_failed" or mode == "failed_first" and self.number == 0 else 0
    monkeypatch.setattr(module.subprocess, "run", cuda_run)
    monkeypatch.setattr(module.subprocess, "Popen", Child)
    if mode in ("all_failed", "cuda_failed"):
        with pytest.raises(RuntimeError, match="No passing|CUDA fixtures failed"):
            suite.benchmark()
    else:
        suite.benchmark()
    assert len(fixtures) == 1 and "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1" in fixtures[0]
    cuda = module.read(suite.root / "cuda-fixtures.json")
    assert cuda["job_id"] == "22" and cuda["identity"] == suite.identity
    if mode == "cuda_failed":
        assert spawned == [] and not (suite.root / "selection.json").exists()
        return
    expected = suite.variants if mode in ("fallback", "all_failed") else suite.variants[:2]
    assert [row[0][row[0].index("--plan") + 1] for row in spawned] == [value["plan"] for value in expected]
    assert all(row[1]["start_new_session"] is True for row in spawned)
    result = module.read(suite.root / "selection.json")
    assert result["training_started"] is False and result["benchmark_job_id"] == "22"
    assert [row["id"] for row in result["observations"]] == [value["id"] for value in expected]
    assert result["cuda_fixture_gate_sha256"] == module.file_sha256(suite.root / "cuda-fixtures.json")
    if mode == "all_failed":
        assert result["selection"]["status"] == "stopped"
    else:
        winner = suite.variants[2] if mode == "fallback" else suite.variants[1] if mode == "failed_first" else suite.variants[0]
        assert result["selection"]["winner"] == winner["id"]
        if mode == "failed_first":
            assert result["observations"][0]["eligible"] is False


def test_promote_dispatches_once_to_selected_candidate_without_fictional_benchmark_submission(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    selected, calls = suite.variants[1], []
    monkeypatch.setattr(suite, "selected", lambda: (selected, "f" * 64))
    monkeypatch.setattr(module, "SpeedSuite", lambda path, sha: suite)
    class Candidate:
        def __init__(self, actual_suite, actual_variant):
            assert actual_suite is suite and actual_variant == selected
        def advance(self, stage):
            calls.append(stage)
    monkeypatch.setattr(module, "SpeedCandidate", Candidate)
    monkeypatch.setattr(sys, "argv", ["chemistry_speed_campaign.py", "--suite", str(suite.path),
                                    "--suite-sha256", suite.sha, "--mode", "promote"])
    module.main()
    assert calls == ["benchmark"]


def _small_policy_suite(tmp_path, monkeypatch):
    """Version the same synthetic evidence with twelve workers/sixteen slots.

    Only declared execution/resources change; candidate scientific gates and
    the shared allocation readers remain the production implementations.
    """
    suite = _suite(tmp_path, monkeypatch)
    suite.plan["allocation_policy"] = module.SMALL_ALLOCATION_POLICY
    suite.plan["protocol"] = {"segment_wall_seconds": 23 * 3600}
    suite.plan["execution"].update(workers=12, gpu_cpus=16)
    for variant in suite.variants:
        variant["workers"] = 12
        plan = module.read(variant["plan"])
        plan["allocation_policy"] = module.SMALL_ALLOCATION_POLICY
        plan["protocol"]["segment_wall_seconds"] = 23 * 3600
        plan["execution"].update(workers=12, gpu_cpus=16)
        _json(Path(variant["plan"]), plan)
        variant["plan_sha256"] = module.file_sha256(Path(variant["plan"]))
    _json(suite.path, suite.plan)
    return suite


def test_versioned_resource_reader_preserves_legacy_32_slot_policy(tmp_path, monkeypatch):
    suite = _suite(tmp_path, monkeypatch)
    before = deepcopy(suite.plan)
    assert module.suite_gpu_slots(suite.plan) == 32
    # Historical 72-hour requests remain readable; the new user cap applies
    # only to a newly sealed, explicitly versioned operational protocol.
    module.require_allocation_limit(suite.plan, "72:00")
    assert suite.plan == before and "allocation_policy" not in suite.plan


def test_small_resource_reader_accepts_twelve_workers_sixteen_slots_23h(tmp_path, monkeypatch):
    suite = _small_policy_suite(tmp_path, monkeypatch)
    before = deepcopy(suite.plan)
    assert module.suite_gpu_slots(suite.plan) == 16
    assert all(module.suite_gpu_slots(module.read(row["plan"])) == 16
               for row in suite.variants)
    assert suite.plan == before


@pytest.mark.parametrize("field,value", [
    ("workers", 20), ("workers", 28), ("workers", "12"), ("workers", None),
    ("gpu_cpus", 32), ("gpu_cpus", 12), ("gpu_cpus", "16"), ("gpu_cpus", True),
])
def test_small_resource_reader_rejects_mismatched_execution(tmp_path, monkeypatch, field, value):
    suite = _small_policy_suite(tmp_path, monkeypatch)
    suite.plan["execution"][field] = value
    with pytest.raises(ValueError, match="twelve workers|CPU allocation"):
        module.suite_gpu_slots(suite.plan)


@pytest.mark.parametrize("value", [0, -1, 23 * 3600 + 1, 24 * 3600, "82800", True, None])
def test_small_resource_reader_rejects_invalid_or_over_23h_internal_cap(tmp_path, monkeypatch, value):
    suite = _small_policy_suite(tmp_path, monkeypatch)
    suite.plan["protocol"]["segment_wall_seconds"] = value
    with pytest.raises(ValueError, match="23h"):
        module.suite_gpu_slots(suite.plan)


@pytest.mark.parametrize("value", ["max24h", "", False, 1])
def test_resource_reader_rejects_unknown_versioned_policy(tmp_path, monkeypatch, value):
    suite = _small_policy_suite(tmp_path, monkeypatch)
    suite.plan["allocation_policy"] = value
    with pytest.raises(ValueError, match="Unsupported"):
        module.suite_gpu_slots(suite.plan)


def test_small_constructor_accepts_bound_candidates_and_rejects_policy_escape(tmp_path, monkeypatch):
    original = _small_policy_suite(tmp_path, monkeypatch)

    def initialize(self, unused_path, unused_sha):
        for name in ("work", "release", "root", "path", "receipts", "sha", "identity", "plan"):
            setattr(self, name, deepcopy(getattr(original, name)))

    monkeypatch.setattr(module.Campaign, "__init__", initialize)
    monkeypatch.setattr(module, "validate_plan", lambda path, sha: module.read(path))
    accepted = module.SpeedSuite(original.path, original.sha)
    assert len(accepted.variants) == len(original.variants)
    path = Path(original.variants[0]["plan"])
    escaped = module.read(path)
    escaped.pop("allocation_policy")
    _json(path, escaped)
    with pytest.raises(ValueError, match="Candidate escaped"):
        module.SpeedSuite(original.path, original.sha)


@pytest.mark.parametrize("owner", ["suite", "candidate"])
@pytest.mark.parametrize("limit", ["0:01", "6:00", "24:00"])
def test_small_suite_and_candidate_submission_accept_caps_up_to_24h(tmp_path, monkeypatch, owner, limit):
    suite = _small_policy_suite(tmp_path, monkeypatch)
    candidate = module.SpeedCandidate.__new__(module.SpeedCandidate)
    candidate.suite, candidate.plan, candidate.work = suite, module.read(suite.variants[0]["plan"]), suite.work
    controller = suite if owner == "suite" else candidate
    command = controller.arguments("segment-001", gpu=True, limit=limit)
    assert command[command.index("-n") + 1] == "16"
    assert command[command.index("-W") + 1] == limit
    assert "rusage[mem=4GB]" in command and "span[hosts=1]" in command


@pytest.mark.parametrize("owner", ["suite", "candidate"])
@pytest.mark.parametrize("limit", ["24:01", "72:00", "invalid", "1", "1:60", "0:00", "24:00:00", None])
def test_small_suite_and_candidate_submission_reject_over_24h_or_invalid_caps(tmp_path, monkeypatch, owner, limit):
    suite = _small_policy_suite(tmp_path, monkeypatch)
    candidate = module.SpeedCandidate.__new__(module.SpeedCandidate)
    candidate.suite = suite
    controller = suite if owner == "suite" else candidate
    # The cap must fail before touching a scheduler or entering parent
    # Campaign.arguments (the deliberately incomplete candidate has no plan).
    with pytest.raises(ValueError, match="24-hour cap"):
        controller.arguments("segment-001", gpu=True, limit=limit)


def test_small_native_suite_accepts_actual_16_slot_environment_with_mocked_children(tmp_path, monkeypatch):
    suite = _small_policy_suite(tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "require_cpu", lambda: None)
    monkeypatch.setenv("LSB_JOBID", "22")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "16")
    for variant in suite.variants:
        _native(suite, variant)
    fixtures, spawned, active = [], [], []

    def cuda_run(command, **kwargs):
        fixtures.append(command)
        return SimpleNamespace(returncode=0)

    class Child:
        def __init__(self, command, **kwargs):
            assert active == []
            self.pid = 9000 + len(spawned)
            spawned.append((command, kwargs))
            active.append(self)

        def wait(self, timeout):
            assert 0 < timeout <= 6300 and active.pop() is self
            return 0

    monkeypatch.setattr(module.subprocess, "run", cuda_run)
    monkeypatch.setattr(module.subprocess, "Popen", Child)
    suite.benchmark()
    assert len(fixtures) == 1 and "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1" in fixtures[0]
    assert len(spawned) == 2  # A passing preferred family does not run fallback.
    assert all(row[1]["start_new_session"] is True for row in spawned)
    result = module.read(suite.root / "selection.json")
    assert result["training_started"] is False and result["benchmark_job_id"] == "22"
    assert result["selection"]["status"] == "selected_not_training_started"
    assert result["selection"]["prior_ratio"] == 3


def test_small_native_suite_rejects_legacy_32_slot_environment_before_subprocess(tmp_path, monkeypatch):
    suite = _small_policy_suite(tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "require_cpu", lambda: None)
    monkeypatch.setenv("LSB_JOBID", "22")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "32")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="16-slot"):
        suite.benchmark()
    assert calls == [] and not (suite.root / "cuda-fixtures.json").exists()
    assert not (suite.root / "selection.json").exists()
