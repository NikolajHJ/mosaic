"""Allocated contract tests for conditional restart and disposable evidence."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from latentmol import chemistry_restart_policy as policy
from latentmol.experiments import file_sha256, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
import chemistry_guarded_restart as controller


def _plan(tmp_path):
    return dict(campaign=str(tmp_path), _plan_sha256="a"*64, archive_sha256="b"*64,
        manifest_sha256="c"*64, starting_checkpoint=dict(epoch=36, updates=237456, sha256="d"*64),
        guarded_restart=dict(version=policy.VERSION, validation_updates=2500, validation_wall_seconds=82800),
        protocol=dict(failure_guidance=dict(loss=dict(version="repair-failure-guidance-v2")),
            early_health=dict(version="chemistry-early-health-v1", first_check_updates=32, interval_updates=100,
                              max_examples_per_failure=4, max_strict_validity_drop=.02),
            segment_wall_seconds=82800))


def test_historical_plans_do_not_require_the_new_gate(tmp_path):
    assert policy.validate_guarded_restart(dict(protocol={})) is None
    assert policy.validate_guarded_restart(dict(protocol=dict(failure_guidance=None))) is None


def test_corrected_objective_cannot_omit_its_longer_validation(tmp_path):
    value = _plan(tmp_path)
    del value["guarded_restart"]
    with pytest.raises(ValueError, match="requires explicit"):
        policy.validate_guarded_restart(value)


@pytest.mark.parametrize("key,value", [("version", "unversioned"), ("validation_updates", 8),
    ("validation_updates", True), ("validation_wall_seconds", 172800)])
def test_unknown_or_too_short_validation_cannot_authorize_a_restart(tmp_path, key, value):
    plan = _plan(tmp_path)
    plan["guarded_restart"][key] = value
    with pytest.raises(ValueError):
        policy.validate_guarded_restart(plan)


def test_panel_schedule_covers_first_check_intervals_and_terminal_state():
    assert policy.panel_schedule(dict(first_check_updates=32, interval_updates=100), 2500) == [32, *range(100,2501,100)]
    assert policy.panel_schedule(dict(first_check_updates=32, interval_updates=100), 2450)[-2:] == [2400,2450]


def _evidence(tmp_path):
    plan = _plan(tmp_path)
    root = tmp_path
    run = root / "stability" / "fixture"
    run.mkdir(parents=True)
    write_json(root / "early-health-baseline.json", {"fixture": "classified authenticated epoch36"})
    baseline_sha = file_sha256(root / "early-health-baseline.json")
    write_json(root / "benchmark_gate.json", {"fixture": "independently authenticated native readiness"})
    native = dict(status="passed", native_panel=dict(completed_epoch=36), gates=dict(fixture=True),
                  early_health_baseline_sha256=baseline_sha)
    panels = []
    for relative in policy.panel_schedule(plan["protocol"]["early_health"],2500):
        path = run / f"panel_{relative}.json"
        write_json(path, dict(status="passed", checks=dict(retention=True), plan_sha256=plan["_plan_sha256"],
                             baseline_sha256=baseline_sha, updates=237456+relative,
                             archive_sha256=plan["archive_sha256"],manifest_sha256=plan["manifest_sha256"],
                             starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],job_id="123"))
        panels.append(dict(relative_update=relative, path=str(path), sha256=file_sha256(path)))
    gate = dict(version=policy.GATE_VERSION, status="passed", plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"],manifest_sha256=plan["manifest_sha256"],
        starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"], starting_epoch=36,
        starting_update=237456, last_update=239956, completed_updates=2500,
        validation_weights_discarded=True,epoch_complete=False, gates={key:True for key in policy.GATES},
        native_readiness_science_sha256=policy.readiness_science_sha256(native),
        native_benchmark_sha256=file_sha256(root/"benchmark_gate.json"),
        early_health_baseline_sha256=baseline_sha, observed_panel_schedule=[r["relative_update"] for r in panels],
        panels=panels,run_dir=str(run),job_id="123",conservative_epoch_seconds=1000.)
    write_json(run/"run.json",dict(status="completed",environment=dict(job_id="123"),
        source_archive_sha256=plan["archive_sha256"],source=dict(sha256=plan["manifest_sha256"])))
    import json
    updates_path=run/"updates.jsonl"
    updates_path.write_text("\n".join(json.dumps(dict(relative_update=i,update=237456+i,real_points=256,
        fresh_prior_points=512,total=1.,reconstruction=.01,repair_real=.1,repair_prior=.2,update_seconds=1.))
        for i in range(1,2501)))
    gate.update(updates_path=str(updates_path),updates_sha256=file_sha256(updates_path))
    _bind(root,run,gate,native)
    return plan,gate,native,run


def _bind(root,run,gate,native):
    write_json(run/"gate.json",gate)
    write_json(root/"stability_gate.json",gate)
    native["stability_gate_sha256"]=file_sha256(root/"stability_gate.json")
    write_json(root/"failure_readiness.json",native)


def test_completed_disposable_trajectory_and_all_panels_authenticate(tmp_path):
    plan,gate,_,_ = _evidence(tmp_path)
    assert policy.authenticate_stability(plan)==gate


@pytest.mark.parametrize("key,value", [("completed_updates",2499),("last_update",239955),
    ("epoch_complete",True),("validation_weights_discarded",False),("starting_epoch",37),
    ("starting_checkpoint_sha256","e"*64),("native_readiness_science_sha256","e"*64)])
def test_rebinding_a_partial_or_wrong_parent_result_cannot_make_it_eligible(tmp_path,key,value):
    plan,gate,native,run=_evidence(tmp_path)
    gate[key]=value
    _bind(tmp_path,run,gate,native)
    with pytest.raises(ValueError):
        policy.authenticate_stability(plan)


def test_a_failed_check_cannot_be_hidden_by_passed_outer_status(tmp_path):
    plan,gate,native,run=_evidence(tmp_path)
    gate["gates"]["complete_epoch_fits"]=False
    _bind(tmp_path,run,gate,native)
    with pytest.raises(ValueError):
        policy.authenticate_stability(plan)


def test_wrong_panel_state_is_rejected_even_after_rebinding_file_hash(tmp_path):
    plan,gate,native,run=_evidence(tmp_path)
    panel=gate["panels"][0]
    write_json(Path(panel["path"]),dict(status="passed",checks=dict(retention=True),
        plan_sha256=plan["_plan_sha256"],baseline_sha256=gate["early_health_baseline_sha256"],updates=237456+31,
        archive_sha256=plan["archive_sha256"],manifest_sha256=plan["manifest_sha256"],
        starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],job_id="123"))
    panel["sha256"]=file_sha256(Path(panel["path"]))
    _bind(tmp_path,run,gate,native)
    with pytest.raises(ValueError,match="fixed panel"):
        policy.authenticate_stability(plan)


def test_missing_classified_baseline_or_unfinished_run_blocks_restart(tmp_path):
    plan,gate,native,run=_evidence(tmp_path)
    write_json(run/"run.json",dict(status="running",environment=dict(job_id="123")))
    with pytest.raises(ValueError,match="completed allocated"):
        policy.authenticate_stability(plan)


def test_every_contract_module_is_in_allocated_command_and_callbacks_keep_the_new_controller(tmp_path):
    command=controller.cpu_command("python",tmp_path)
    assert all(name in command for name in controller.GUARDED_FIXTURES)
    assert len(command)==len(set(command))
    campaign=object.__new__(controller.GuardedRestart)
    campaign.work=tmp_path;campaign.release=tmp_path/"source";campaign.path=tmp_path/"plan";campaign.sha="a"*64
    argv=campaign.controller("advance",after_stage="segment-001")
    assert "scripts/hpc/chemistry_guarded_restart.py" in argv
    assert argv[-2:]==["--after-stage","segment-001"]


def test_failed_cpu_cannot_submit_native_validation(tmp_path):
    campaign=object.__new__(controller.GuardedRestart)
    campaign.root=tmp_path;campaign.sha="a"*64;campaign.identity={"fixture":True}
    calls=[]
    def fail():
        raise ValueError("CPU failed")
    campaign.require_cpu=fail
    campaign.queue=lambda *args,**kwargs:calls.append(args)
    with pytest.raises(ValueError,match="CPU failed"):
        campaign.after_cpu()
    assert not calls


def _startup_evidence(tmp_path):
    """Actual sized epoch:6596 updates and67 conservative observation panels."""
    from latentmol.chemistry_worker_startup import NATIVE_MODULES, VERSION, project_warmed_epoch
    plan, gate, native, run = _evidence(tmp_path)
    plan.update(worker_startup=dict(version=VERSION), dataset_sha256="e" * 64,
        training_molecules=1_688_576,
        execution=dict(backend="optimized", workers=12, torch_threads=4, gpu_cpus=16))
    plan["protocol"].update(real_batch_size=256)
    checks = {name: True for name in ("model", "reference", "optimizer", "gradients",
              "reference_gradients", "model_modes", "reference_modes", "rng")}
    receipt = dict(version=VERSION, status="passed", passed=True, job_id="123",
        plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"], dataset_sha256=plan["dataset_sha256"],
        starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"], model_sha256="f" * 64,
        model_forwards=0, optimizer_steps=0, random_samples=0, target_searches=0, target_cache_entries=0,
        modules=list(NATIVE_MODULES), warmup_real_points=256, initialized_workers=12,
        worker_initialization_barrier_passed=True, observed_task_pids=[101, 104],
        worker_start_and_classification_seconds=10., elapsed_seconds=15., real_row_ids_sha256="1" * 64,
        hard_graphs=dict(sha256="2" * 64, named_entries=3, nonmissing_entries=3, missing_names=[]),
        classification=dict(attempts=256), restoration=dict(passed=True, checks=checks))
    gate.update(starting_model_sha256="f" * 64, fixed_panel_seconds=[2.] * len(gate["panels"]),
                mean_update_seconds=1., slowest_update_seconds=1.)
    projection = project_warmed_epoch([1.] * 2500, plan["training_molecules"], 256,
                                     startup_seconds=15., panel_seconds=2. * 67)
    gate.update(warmed_runtime_projection=projection, projected_mean_epoch_seconds=projection["projected_mean_epoch_seconds"],
                conservative_epoch_seconds=projection["conservative_epoch_seconds"])
    native["conservative_epoch_seconds"] = 1000.
    _bind_startup(tmp_path, plan, gate, native, run, receipt)
    return plan, gate, native, run, receipt


def _bind_startup(root, plan, gate, native, run, receipt):
    path = run / "worker-startup.json"
    write_json(path, receipt)
    gate["worker_startup"] = dict(path=str(path), sha256=file_sha256(path),
        elapsed_seconds=receipt["elapsed_seconds"], status=receipt["status"])
    gate["native_readiness_science_sha256"] = policy.readiness_science_sha256(native)
    _bind(root, run, gate, native)


def test_opt_in_startup_authenticates_all_walls_and_original_fit_guard(tmp_path):
    plan, gate, _, _, _ = _startup_evidence(tmp_path)
    assert gate["warmed_runtime_projection"]["epoch_updates"] == 6596
    assert gate["warmed_runtime_projection"]["fixed_panel_seconds"] == 134.
    assert gate["conservative_epoch_seconds"] == 6596. + 134. + 15. + 600.
    assert policy.authenticate_stability(plan) == gate


@pytest.mark.parametrize("key,value", [
    ("version", "unversioned"), ("status", "running"), ("passed", False),
    ("job_id", "124"), ("plan_sha256", "9" * 64), ("archive_sha256", "9" * 64),
    ("manifest_sha256", "9" * 64), ("dataset_sha256", "9" * 64),
    ("starting_checkpoint_sha256", "9" * 64), ("model_sha256", "9" * 64),
    ("warmup_real_points", 255), ("warmup_real_points", 256.), ("initialized_workers", 11),
    ("worker_initialization_barrier_passed", False), ("observed_task_pids", []),
    ("observed_task_pids", [101, 101]), ("observed_task_pids", [True]),
    ("modules", ["latentmol.atom_preserving_fast"]), ("model_forwards", 1),
    ("optimizer_steps", 1), ("random_samples", 1), ("target_searches", 1),
    ("target_cache_entries", 1), ("target_cache_entries", False),
    ("real_row_ids_sha256", "not_a_sha"), ("worker_start_and_classification_seconds", 16.),
])
def test_rehashed_startup_cannot_hide_changed_identity_work_or_missing_barrier(tmp_path, key, value):
    plan, gate, native, run, receipt = _startup_evidence(tmp_path)
    receipt[key] = value
    _bind_startup(tmp_path, plan, gate, native, run, receipt)
    with pytest.raises(ValueError):
        policy.authenticate_stability(plan)


@pytest.mark.parametrize("change", ["failed_restoration", "missing_gradients", "partial_rows", "missing_head"])
def test_startup_requires_complete_gradient_mode_rng_restore_and_all_real_rows(tmp_path, change):
    plan, gate, native, run, receipt = _startup_evidence(tmp_path)
    if change == "failed_restoration":
        receipt["restoration"]["checks"]["rng"] = False
    elif change == "missing_gradients":
        del receipt["restoration"]["checks"]["gradients"]
    elif change == "partial_rows":
        receipt["classification"]["attempts"] = 255
    else:
        receipt["hard_graphs"]["nonmissing_entries"] = 2
    _bind_startup(tmp_path, plan, gate, native, run, receipt)
    with pytest.raises(ValueError):
        policy.authenticate_stability(plan)


def test_startup_cannot_be_rebound_to_another_directory_or_filename(tmp_path):
    plan, gate, native, run, _ = _startup_evidence(tmp_path)
    outside = tmp_path / "worker-startup.json"
    outside.write_bytes((run / "worker-startup.json").read_bytes())
    gate["worker_startup"]["path"] = str(outside)
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError, match="escaped"):
        policy.authenticate_stability(plan)


@pytest.mark.parametrize("key", ["worker_startup", "warmed_runtime_projection"])
def test_explicit_startup_cannot_omit_its_binding_or_projection(tmp_path, key):
    plan, gate, native, run, _ = _startup_evidence(tmp_path)
    del gate[key]
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError):
        policy.authenticate_stability(plan)


def test_startup_binding_cannot_omit_or_change_elapsed_time(tmp_path):
    plan, gate, native, run, _ = _startup_evidence(tmp_path)
    gate["worker_startup"]["elapsed_seconds"] = 14.
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError, match="zero-update"):
        policy.authenticate_stability(plan)


@pytest.mark.parametrize("field", ["one_time_startup_seconds", "fixed_panel_seconds", "base_overhead_seconds",
    "slowest_update_seconds", "epoch_updates", "projected_mean_epoch_seconds", "conservative_epoch_seconds"])
def test_warm_projection_cannot_trim_any_cost_even_after_rebinding_hashes(tmp_path, field):
    plan, gate, native, run, _ = _startup_evidence(tmp_path)
    gate["warmed_runtime_projection"][field] -= 1
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError, match="projection"):
        policy.authenticate_stability(plan)


def test_new_slow_recurring_update_cannot_be_ignored_as_worker_startup(tmp_path):
    import json
    plan, gate, native, run, _ = _startup_evidence(tmp_path)
    path = Path(gate["updates_path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1234]["update_seconds"] = 2.
    path.write_text("\n".join(json.dumps(row) for row in rows))
    gate["updates_sha256"] = file_sha256(path)
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError, match="projection"):
        policy.authenticate_stability(plan)


def test_new_slow_fixed_panel_cannot_be_ignored(tmp_path):
    plan, gate, native, run, _ = _startup_evidence(tmp_path)
    gate["fixed_panel_seconds"][-1] = 10.
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError, match="projection"):
        policy.authenticate_stability(plan)


def test_warmed_runtime_keeps_native_conservative_maximum(tmp_path):
    plan, gate, native, run, _ = _startup_evidence(tmp_path)
    native["conservative_epoch_seconds"] = 10_000.
    gate["conservative_epoch_seconds"] = 10_000.
    gate["native_readiness_science_sha256"] = policy.readiness_science_sha256(native)
    _bind(tmp_path, run, gate, native)
    assert policy.authenticate_stability(plan) == gate
    gate["conservative_epoch_seconds"] -= 1
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError, match="conservative"):
        policy.authenticate_stability(plan)
