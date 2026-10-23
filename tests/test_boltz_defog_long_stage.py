"""Allocated metadata fixtures for serial allocation boundaries, not model tests.

Native arm validation is reused rather than reimplemented. Fixtures substitute
its expensive reader only when testing immutable orchestration/receipt logic;
the existing long-comparison suite separately covers every native guard.
"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def stage(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_boltz_defog_long_stage")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def stage_fixture(stage, tmp_path, monkeypatch, *, stalled=False):
    """Real source/config/hash receipts with a stub native scientific reader."""
    from latentmol.config_identity import config_sha256, CONFIG_HASH_VERSION
    plan_path = tmp_path / "plan.json"
    plan = {"output_root": str(tmp_path / "campaign"), "repair_search": {"version": "unchanged-test-search"},
            "cases": [{"id": case, "config": {"path": str(tmp_path / (case + ".json")), "sha256": "a" * 64}}
                      for case in stage.comparison.CASE_IDS]}
    write(plan_path, plan)
    plan_ref = stage._ref(plan_path)
    identity = {"source_manifest_sha256": "b" * 64, "source_archive_sha256": "c" * 64}
    inputs = {"assets": [{"path": "/frozen/model", "sha256": "d" * 64}], "native_sources": []}
    cfgs = {case: object() for case in stage.comparison.CASE_IDS}
    calls = []
    monkeypatch.setattr(stage, "_scientific_inputs", lambda configs: deepcopy(inputs))

    def native(case, row, cfg, source, parent):
        calls.append(case["id"])
        if row.get("native_gate") is not True or source != identity or cfg is not cfgs[case["id"]]:
            raise ValueError("Fixture native scientific guard failed")
        if row["status"] not in ("passed", "stalled") or row["outcome"]["healthy_terminal_passed"] is not True:
            raise ValueError("Fixture native terminal failed")
        return Path(row["child_run"])
    monkeypatch.setattr(stage, "_revalidate_arm", native)

    def build(name, previous=None):
        root = stage.stage_root(plan, name)
        parent = root / "work" / ("experiment-" + name)
        parent.mkdir(parents=True)
        before = [] if previous is None else deepcopy(previous["rows"])
        if name in stage._CASE:
            case_id = stage._CASE[name]
            child = Path(plan["output_root"]) / case_id / "optimization" / "native-child"
            write(child / "run.json", {"run_id": "native-" + case_id})
            write(child / "native-evidence.json", {"all_required_native_checks": True})
            before.append({"id": case_id, "child_run": str(child), "native_gate": True,
                "status": "stalled" if stalled else "passed", "initial_z_sha256": "e" * 64,
                "outcome": {"healthy_terminal_passed": True, "full_500_passed": not stalled}})
        else:
            for row in before:
                case_id = row["id"]
                record_path = Path(row["child_run"]) / "run.json"
                followups = {"status": "completed", "case_id": case_id, "child_run": row["child_run"],
                    "plan_sha256": plan_ref["sha256"], "child_run_sha256": stage._ref(record_path)["sha256"],
                    "prediction_attempts": 0,
                    "memberships": [{"step": step, "role": role, "seed": seed,
                                     "status": "invalid_raw" if role == "raw" else "no_witness"}
                                    for step in stage.comparison.CHECKPOINTS for role in ("raw", "repaired") for seed in (17, 23)]}
                row.update(followup_completed=True, followup_execution={"returncode": 0, "timed_out": False,
                    "failure": None, "interrupted": False}, followups=followups)
                for label in ("repair-audit", "fresh-pose"):
                    folder = Path(plan["output_root"]) / case_id / label / "subrun"
                    config = {"parent_run_id": "native-" + case_id, "case_id": case_id,
                              "parent_run_sha256": stage._ref(record_path)["sha256"], "plan_sha256": plan_ref["sha256"]}
                    if label == "repair-audit":
                        config = {"parents": [{"id": case_id, "pilot": row["child_run"],
                            "run_sha256": stage._ref(record_path)["sha256"], "steps": list(stage.comparison.CHECKPOINTS)}],
                            "search": deepcopy(plan["repair_search"]), "output_root": str(folder.parent)}
                    write(folder / "run.json", {"status": "completed", "config": config,
                        "config_hash_version": CONFIG_HASH_VERSION, "config_sha256": config_sha256(config),
                        "source": {"sha256": identity["source_manifest_sha256"]},
                        "source_archive_sha256": identity["source_archive_sha256"]})
                    write(folder / "summary.json", {"status": "completed", "case_id": case_id})
                    if label == "repair-audit":
                        followups["repair_audit"] = {"run": str(folder), "summary": stage._ref(folder / "summary.json")}
                    else:
                        followups["fresh_pose_run"] = str(folder)
        previous_ref = None if previous is None else previous["ref"]
        config = {"version": stage.VERSION, "stage": name, "plan": plan_ref, "previous": previous_ref,
                  "budget": stage._budget(name), **identity}
        job = str(100 + stage.STAGES.index(name))
        write(root / "attempt.json", {"stage": name, "job_id": job, "plan": plan_ref, "previous": previous_ref})
        write(parent / "run.json", {"status": "completed", "config": config,
            "config_hash_version": CONFIG_HASH_VERSION, "config_sha256": config_sha256(config),
            "source": {"sha256": identity["source_manifest_sha256"]},
            "source_archive_sha256": identity["source_archive_sha256"], "environment": {"job_id": job}})
        result = stage._state_report(name, before, evaluation_complete=name == "evaluation")
        result["interactive_reports"] = []
        write(parent / "stage-report.json", result)
        receipt = {"version": stage.VERSION, "stage": name, "status": "passed", "plan": plan_ref,
            "source": identity, "budget": stage._budget(name), "previous": previous_ref,
            "experiment": stage._ref(parent / "run.json"), "report": stage._ref(parent / "stage-report.json"),
            "scientific_inputs": deepcopy(inputs), "artifacts": stage._seal(stage._stage_outputs(name, parent, before, plan, identity),
                extra_files=[root / "attempt.json"]),
            "job_id": job, "elapsed_seconds": 10.}
        receipt_path = root / "receipt.json"
        write(receipt_path, receipt)
        return {"name": name, "ref": stage._ref(receipt_path), "path": receipt_path,
                "receipt": receipt, "rows": before, "parent": parent, "result": result}

    def verify(value):
        return stage._receipt_context(value["name"], value["ref"], plan_ref, plan, cfgs, identity)

    return SimpleNamespace(plan=plan, plan_ref=plan_ref, identity=identity, inputs=inputs, cfgs=cfgs,
                           build=build, verify=verify, calls=calls)


def test_all_allocations_at_most_24h_preserve_full_original_reservations(stage):
    assert [stage._budget(name)["allocation_seconds"] for name in stage.STAGES] == [64800, 86400, 21600]
    assert sum(stage._budget(name)["allocation_seconds"] for name in stage.STAGES) == 172800
    assert sum(stage._budget(name)["internal_wall_seconds"] for name in stage.STAGES) == 165600
    assert all(stage._budget(name)["internal_wall_seconds"] < stage._budget(name)["allocation_seconds"] <= 86400
               for name in stage.STAGES)
    assert stage.comparison.BUDGETS["optimization_timeout_seconds"] == {"boltz_only": 54000, "boltz_defog": 82800}
    assert stage.comparison.BUDGETS["followup_timeout_seconds"] == 7200
    assert not any("resume" in key for budget in stage.STAGE_BUDGETS.values() for key in budget)


@pytest.mark.parametrize("stalled", [False, True])
def test_authenticated_chain_rechecks_native_arms_and_keeps_partial_claims_truthful(stage, tmp_path, monkeypatch, stalled):
    f = stage_fixture(stage, tmp_path, monkeypatch, stalled=stalled)
    control = f.build("control")
    c = f.verify(control)
    assert [row["id"] for row in c["rows"]] == ["boltz_only"]
    assert not control["result"]["engineering_passed"] and not control["result"]["evaluation_complete"]
    guided = f.build("guided", control)
    g = f.verify(guided)
    assert g["job_ids"] == {"100", "101"}
    evaluation = f.build("evaluation", guided)
    e = f.verify(evaluation)
    assert e["job_ids"] == {"100", "101", "102"}
    assert evaluation["result"]["healthy_terminal_passed"] and evaluation["result"]["evaluation_complete"]
    assert evaluation["result"]["full_500_passed"] is (not stalled)
    assert evaluation["result"]["status"] == ("stalled_observed" if stalled else "completed")
    assert f.calls.count("boltz_only") >= 3 and f.calls.count("boltz_defog") >= 2
    assert all(row["initial_z_sha256"] == "e" * 64 for row in e["rows"])
    # Native evaluation files outside the stage parent are part of the seal.
    assert any("/fresh-pose/" in item["path"].replace("\\", "/") for item in evaluation["receipt"]["artifacts"])
    assert any("/repair-audit/" in item["path"].replace("\\", "/") for item in evaluation["receipt"]["artifacts"])


@pytest.mark.parametrize("defect", ["failed", "wrong_budget", "wrong_source", "wrong_plan", "missing_previous",
    "previous_wrong_stage", "reused_job", "failed_experiment", "changed_config", "rewritten_control", "different_initial_z",
    "failed_native_gate", "changed_frozen_inputs", "changed_artifact", "missing_artifact", "extra_artifact", "symlink"])
def test_success_bit_cannot_override_failed_or_changed_predecessor_evidence(stage, tmp_path, monkeypatch, defect):
    f = stage_fixture(stage, tmp_path, monkeypatch)
    control = f.build("control")
    value = f.build("guided", control)
    receipt = value["receipt"]
    if defect == "failed": receipt["status"] = "failed"
    elif defect == "wrong_budget": receipt["budget"]["allocation_seconds"] += 1
    elif defect == "wrong_source": receipt["source"] = {**receipt["source"], "source_archive_sha256": "f" * 64}
    elif defect == "wrong_plan": receipt["plan"] = {**receipt["plan"], "sha256": "f" * 64}
    elif defect == "missing_previous": receipt["previous"] = None
    elif defect == "previous_wrong_stage": receipt["previous"] = value["ref"]
    elif defect == "reused_job": receipt["job_id"] = "100"
    elif defect in ("failed_experiment", "changed_config"):
        path = value["parent"] / "run.json"; record = stage._read(path)
        if defect == "failed_experiment": record["status"] = "failed"
        else: record["config"]["budget"]["internal_wall_seconds"] += 1
        write(path, record); receipt["experiment"] = stage._ref(path)
    elif defect in ("rewritten_control", "different_initial_z", "failed_native_gate"):
        if defect == "rewritten_control": value["rows"][0]["invented_result"] = True
        elif defect == "different_initial_z": value["rows"][1]["initial_z_sha256"] = "f" * 64
        else: value["rows"][1]["native_gate"] = False
        result = stage._state_report("guided", value["rows"])
        write(value["parent"] / "stage-report.json", result)
        receipt["report"] = stage._ref(value["parent"] / "stage-report.json")
    elif defect == "changed_frozen_inputs": f.inputs["assets"][0]["sha256"] = "f" * 64
    else:
        child = Path(value["rows"][1]["child_run"])
        if defect == "changed_artifact": write(child / "native-evidence.json", {"all_required_native_checks": False})
        elif defect == "missing_artifact": (child / "native-evidence.json").unlink()
        elif defect == "extra_artifact": write(child / "unsealed.json", {"replacement": True})
        else:
            # Allocated DTU runs support symlinks; no filesystem escape may be sealed.
            (child / "foreign-evidence.json").symlink_to(value["parent"] / "run.json")
    write(value["path"], receipt); value["ref"] = stage._ref(value["path"])
    with pytest.raises((ValueError, FileNotFoundError, TypeError)):
        f.verify(value)


def test_evaluation_rejects_native_output_tampering_even_when_memberships_still_complete(stage, tmp_path, monkeypatch):
    f = stage_fixture(stage, tmp_path, monkeypatch)
    control = f.build("control"); guided = f.build("guided", control); value = f.build("evaluation", guided)
    f.verify(value)
    path = Path(value["rows"][0]["followups"]["fresh_pose_run"]) / "summary.json"
    write(path, {"status": "completed", "invented_score": -1000})
    with pytest.raises(ValueError, match="artifact membership or bytes"):
        f.verify(value)


@pytest.mark.parametrize("defect", ["parent", "parent_sha", "checkpoints", "search", "root", "fresh_plan"])
def test_completed_evaluation_namespace_does_not_approve_unrelated_native_subrun(stage, tmp_path, monkeypatch, defect):
    from latentmol.config_identity import config_sha256
    f = stage_fixture(stage, tmp_path, monkeypatch)
    control = f.build("control"); guided = f.build("guided", control); value = f.build("evaluation", guided)
    row = value["rows"][0]
    folder = Path(row["followups"]["fresh_pose_run"] if defect == "fresh_plan" else row["followups"]["repair_audit"]["run"])
    path = folder / "run.json"; record = stage._read(path); config = record["config"]
    if defect == "parent": config["parents"][0]["pilot"] = "/unrelated/child"
    elif defect == "parent_sha": config["parents"][0]["run_sha256"] = "f" * 64
    elif defect == "checkpoints": config["parents"][0]["steps"] = [0, 500]
    elif defect == "search": config["search"] = {"different": "teacher"}
    elif defect == "root": config["output_root"] = "/unrelated/audit"
    else: config["plan_sha256"] = "f" * 64
    # A self-consistent config checksum cannot approve unrelated science.
    record["config_sha256"] = config_sha256(config); write(path, record)
    with pytest.raises(ValueError, match="different optimized arm|parent/search/membership"):
        stage._stage_outputs("evaluation", value["parent"], value["rows"], f.plan, f.identity)


def test_full_native_guard_failure_prevents_any_next_optimization_and_attempt_claim(stage, tmp_path, monkeypatch):
    f = stage_fixture(stage, tmp_path, monkeypatch)
    control = f.build("control")
    control["rows"][0]["native_gate"] = False
    write(control["parent"] / "stage-report.json", stage._state_report("control", control["rows"]))
    control["receipt"]["report"] = stage._ref(control["parent"] / "stage-report.json")
    # Keep the byte seal self-consistent so only the failed native scientific
    # proof blocks advancement, rather than an unrelated stale-file checksum.
    control["receipt"]["artifacts"] = stage._seal(
        stage._stage_outputs("control", control["parent"], control["rows"], f.plan, f.identity),
        extra_files=[stage.stage_root(f.plan, "control") / "attempt.json"])
    write(control["path"], control["receipt"]); control["ref"] = stage._ref(control["path"])
    monkeypatch.setenv("LSB_JOBID", "101")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", f.identity["source_archive_sha256"])
    monkeypatch.setattr(stage.comparison, "load_plan", lambda *a: (f.plan, f.cfgs, f.identity))
    touched = []
    monkeypatch.setattr(stage, "_optimize_arm", lambda *a, **k: touched.append(True))
    with pytest.raises(ValueError, match="scientific guard failed"):
        stage.run_stage("guided", f.plan_ref["path"], f.plan_ref["sha256"], previous_ref=control["ref"])
    assert not touched and not stage.stage_root(f.plan, "guided").exists()


@pytest.mark.parametrize("failure", ["returncode", "timed_out", "failure", "interrupted", "membership", "plan", "parent"])
def test_checkpoint_failure_stops_before_second_arm_evaluation(stage, tmp_path, monkeypatch, failure):
    import latentmol.boltz.workflow_process as process
    calls = []
    plan_path = tmp_path / "plan.json"; write(plan_path, {})
    rows = []
    for case in stage.comparison.CASE_IDS:
        child = tmp_path / case; write(child / "run.json", {"case": case})
        rows.append({"id": case, "child_run": str(child)})
    def execute(command, log, *, timeout_seconds):
        calls.append((command, timeout_seconds))
        output = Path(command[command.index("--output") + 1])
        row = rows[0]
        result = {"status": "completed", "case_id": row["id"], "child_run": row["child_run"],
            "plan_sha256": "a" * 64, "child_run_sha256": stage._ref(Path(row["child_run"]) / "run.json")["sha256"],
            "prediction_attempts": 0, "memberships": [{"step": step, "role": role, "seed": seed, "status": "invalid_raw"}
                for step in stage.comparison.CHECKPOINTS for role in ("raw", "repaired") for seed in (17, 23)]}
        execution = {"returncode": 0, "timed_out": False, "failure": None, "interrupted": False}
        if failure in execution: execution[failure] = {"returncode": 1, "timed_out": True, "failure": {"error": "native"}, "interrupted": True}[failure]
        elif failure == "membership": result["memberships"].pop()
        elif failure == "plan": result["plan_sha256"] = "f" * 64
        else: result["child_run_sha256"] = "f" * 64
        write(output, result)
        return execution
    monkeypatch.setattr(process, "run_logged", execute)
    monkeypatch.setattr(stage, "_require_remaining", lambda *a: None)
    with pytest.raises(RuntimeError, match="no next evaluation or replacement"):
        stage._evaluate(rows, {"followup_timeout_seconds": 7200}, plan_path, "a" * 64, tmp_path / "evaluation", 100000.)
    assert len(calls) == 1 and calls[0][1] == 7200
    command = calls[0][0]
    assert command[command.index("--plan") + 1] == str(plan_path)
    assert command[command.index("--sha256") + 1] == "a" * 64
    assert rows[0]["followup_completed"] is False and "followups" not in rows[1]


def test_insufficient_remaining_stage_wall_fails_before_launch(stage, monkeypatch):
    monkeypatch.setattr(stage.time, "monotonic", lambda: 1000.)
    stage._require_remaining(2000., 1000., "exact reservation")
    with pytest.raises(TimeoutError): stage._require_remaining(1999., 1000., "one second short")


def test_only_artifact_reads_are_needed_to_revalidate_native_arm(stage):
    # No tensor-loading/start re-optimization call belongs in callback validation.
    import ast
    tree = ast.parse(Path(stage.__file__).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_revalidate_arm")
    names = [node.id for node in ast.walk(function) if isinstance(node, ast.Name)]
    assert "verify_start" not in names and "_optimize" not in names
    assert "validate_prior_run" in names and "read_arm_terminal" in names
    assert any(isinstance(node, ast.Attribute) and node.attr == "validate_case" for node in ast.walk(function))
