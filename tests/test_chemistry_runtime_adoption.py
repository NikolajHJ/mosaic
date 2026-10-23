"""Allocated metadata-only adoption fixtures; synthetic records prove no science."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentmol import chemistry_runtime_adoption as adoption
from latentmol.chemistry_speed_selection import REQUIRED_GATES, REQUIRED_PARITY_CHECKS


def fixture(tmp_path, monkeypatch):
    work = tmp_path / "work"
    suite_root = work / "runs/source126"
    old_root = suite_root / "variants/bond6"
    new_root = work / "runs/new-adoption"
    old_release, new_release = (work / "releases" / name for name in ("old", "new"))
    artifacts = {}
    def put(path, value, *, raw=False, bind=True):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        data = value.encode() if raw else adoption._encode(value)
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        if bind: artifacts[str(path)] = sha
        return sha
    def release(path, runtime):
        files = {}
        for name, text in {"src/latentmol/model.py": "unchanged molecular bytes", "src/latentmol/chemistry_runtime.py": runtime}.items():
            files[name] = put(path / name, text, raw=True, bind=False)
        sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        put(path / "release_manifest.json", dict(root=str(path), files=files, sha256=sha), bind=False)
        archive = put(str(path) + ".tar.gz", runtime + " archive", raw=True, bind=False)
        return archive, sha
    old_archive, old_manifest = release(old_release, "old runtime")
    new_archive, new_manifest = release(new_release, "new runtime")
    suite_path = work / "submissions/suite.json"
    suite_sha = put(suite_path, dict(campaign=str(suite_root), release=str(old_release),
        archive_sha256=old_archive, manifest_sha256=old_manifest), bind=False)
    ancestor = work / "runs/source121"
    ancestor_plan = work / "submissions/source121.json"
    ancestor_sha = put(ancestor_plan, dict(campaign=str(ancestor)), bind=False)
    completion = ancestor / "segment-001-completion.json"
    completion_sha = put(completion, dict(status="passed"), bind=False)
    checkpoint = ancestor / "epoch36.pt"
    checkpoint_sha = put(checkpoint, "fixture bytes, not a model", raw=True, bind=False)
    binding = dict(epoch=36, updates=237456, path=str(checkpoint), sha256=checkpoint_sha,
        source_plan=str(ancestor_plan), source_plan_sha256=ancestor_sha,
        completion=str(completion), completion_sha256=completion_sha)
    protocol = dict(benchmark_updates=8, real_batch_size=256, prior_ratio=2, segment_wall_seconds=82800,
        failure_guidance=dict(progress_search=dict(allow_atom_edits=False)))
    source = dict(work_root=str(work), release=str(old_release), campaign=str(old_root),
        archive_sha256=old_archive, manifest_sha256=old_manifest, failure_variant_id="bond6",
        failure_suite_path=str(suite_path), failure_suite_sha256=suite_sha, protocol=protocol,
        starting_checkpoint=binding, training_molecules=1688539, parent_checkpoint_sha256="a" * 64,
        dataset_sha256="b" * 64, split_sha256="c" * 64)
    source_path = suite_root / "plans/bond6.json"
    source_sha = put(source_path, source)
    for name, value in (("SOURCE_PLAN_SHA256", source_sha), ("SOURCE_ARCHIVE_SHA256", old_archive),
                         ("SOURCE_MANIFEST_SHA256", old_manifest), ("SOURCE_SUITE_SHA256", suite_sha)):
        monkeypatch.setattr(adoption, name, value)
    binding_sha = put(suite_root / "starting-checkpoint.json", binding)
    bound_sha = put(suite_root / "bound-plans.json", dict(suite_sha256=suite_sha, binding_sha256=binding_sha,
        variants=[dict(id=name, path=str(source_path) if name == "bond6" else str(suite_root / (name + ".json")),
                       sha256=source_sha if name == "bond6" else "0" * 64) for name in ("legacy3", "fallback", "bond6")]))
    identity = dict(plan_sha256=suite_sha, archive_sha256=old_archive,
                    manifest_sha256=old_manifest, parent_sha256=source["parent_checkpoint_sha256"])
    states = {}
    for stage, job in (("cpu", "101"), ("native", "102"), ("after-native", "103")):
        command = ["bsub", "-oo", str(work / "logs" / (stage + ".%J.out")), suite_sha]
        intent = dict(command=command, identity=identity)
        put(suite_root / "receipts" / (stage + "-intent.json"), intent)
        put(suite_root / "receipts" / (stage + "-receipt.json"), dict(intent=intent, job_id=job))
        put(work / "logs" / (stage + "." + job + ".out"),
            f"Subject: Job {job}: <fixture> in cluster <dcc> Done\n\nSuccessfully completed.\n", raw=True)
        if stage != "after-native":
            put(suite_root / "allocations" / stage / "run.json", dict(status="completed", returncode=0,
                environment=dict(job_id=job), source_archive_sha256=old_archive, source=dict(sha256=old_manifest)))
    def query(command, **kwargs):
        status = states.get(command[-1], "DONE")
        return SimpleNamespace(returncode=0, stdout="" if status == "missing" else status,
                               stderr=f"Job <{command[-1]}> is not found" if status == "missing" else "")
    monkeypatch.setattr(adoption.subprocess, "run", query)
    profile_sha = put(suite_root / "repair_profile.json", dict(status="passed", identity=identity, job_id="101"))
    extra_profile_sha = put(suite_root / "failure_preparation_profile.json", dict(status="passed", identity=identity,
        job_id="101", checkpoint_sha256=checkpoint_sha, completed_epoch=36, scientific_receipts_exact=True, points=256,
        rows=[dict(start=start, count=128, exact_preparation_parity=True) for start in (0, 128)]))
    put(suite_root / "cpu_gate.json", dict(status="passed", returncode=0, job_id="101", identity=identity,
        profile_sha256=profile_sha, execution_profile_sha256=extra_profile_sha))
    for name, count, skipped, job in (("cpu", 1126, 65, "101"), ("deterministic-events", 104, 0, "102"),
                                      ("loss-validation", 22, 0, "102")):
        xml_sha = put(suite_root / (name + "-pytest.xml"),
            f'<testsuites><testsuite tests="{count}" skipped="{skipped}" errors="0" failures="0" /></testsuites>', raw=True)
        if name != "cpu":
            gate = dict(returncode=0, job_id=job, suite_sha256=suite_sha)
            if name == "loss-validation":
                gate.update(junit_sha256=xml_sha, identity=identity,
                    reference_module_sha256="34c1a67cb339233312a1317db39a926b55f378c0eb8ed42c38bde86307fdd184",
                    command=["tests/test_failure_loss_validation_reuse.py"],
                    environment=dict(LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES="1"))
            put(suite_root / (name + "-gate.json"), gate)
    put(suite_root / "cuda-fixtures.json", dict(returncode=0, job_id="102", suite_sha256=suite_sha))
    population = dict(expected_size=22., element_js=.001, charge_js=.001, sulfur_frequency=.01,
        nonempty_fraction=1., raw_argmax_validity=149 / 256)
    reconstruction = dict(exact_reconstruction=1., validity=1.)
    panel = dict(status="observed", completed_epoch=36, plan_sha256=source_sha, job_id="102",
        archive_sha256=old_archive, manifest_sha256=old_manifest, panel_count=1024, panel_seed=123904017,
        source_snapshot_chain_sha256="f" * 64, starting_population=population,
        starting_reconstruction=reconstruction, starting_retention_population=population,
        counts=dict(attempts=1024, already_valid=643, original_accepted=359, original_censored=22,
            original_failed=22, unsupported=0, extra_valid=12, partial=10,
            failed_with_nonzero_gradient=22, failed_without_nonzero_gradient=0),
        failed_row_gradient_norms=[.1] * 22, additional_decoder_gradient_norms=[.1], complete_target_atom_edits=0,
        native_directional_check=dict(passed=True, dtype="float64", analytic=.01, finite_difference=.01,
                                      rtol=1e-4, atol=1e-8))
    composition_sha = put(old_root / "training_composition.json", dict(elements_distribution=[1.],
        charges_distribution=[1.], receipt=dict(split="train", dataset_sha256=source["dataset_sha256"],
            split_sha256=source["split_sha256"], molecules=source["training_molecules"])))
    gates = dict.fromkeys(REQUIRED_GATES, True); gates["complete_epoch_fits"] = False
    run = old_root / "benchmark/run"
    updates = [dict(update_seconds=9., real_points=256, fresh_prior_points=512,
                    failure_only_decoder_gradient_norm=.1) for _ in range(8)]
    native = dict(status="failed", plan_sha256=source_sha, parent_sha256=source["parent_checkpoint_sha256"],
        gates=gates, elapsed_seconds=11., error=adoption.RUNTIME_ERROR, run_dir=str(run),
        native_execution_parity=dict(status="passed", passed=True, restoration=dict(passed=True),
                                    checks=dict.fromkeys(REQUIRED_PARITY_CHECKS, True)),
        native_resume_probe=dict(status="passed", checks=dict(model=True, optimizer=True, rng=True)),
        parent_checkpoint_authentication=dict(passed=True, parent_checkpoint_sha256=source["parent_checkpoint_sha256"],
            objective_change_handoff=dict(starting_checkpoint=binding)),
        updates=updates, projected_epoch_seconds=source["training_molecules"] * 9 / 256 + 600.,
        updated_population=dict(population, raw_argmax_validity=172 / 256), updated_reconstruction=reconstruction,
        composition_sha256=composition_sha, benchmark_weights_discarded=True)
    put(old_root / "benchmark_gate.json", native)
    run_gate = dict(native, elapsed_seconds=10.); run_gate.pop("error")
    put(run / "gate.json", run_gate)
    put(run / "run.json", dict(status="failed", error=adoption.RUNTIME_ERROR.removeprefix("RuntimeError: "),
        gate_status="failed", gates=gates, environment=dict(job_id="102"), source_archive_sha256=old_archive,
        source=dict(sha256=old_manifest), config=dict(_plan_sha256=source_sha)))
    put(old_root / "failure_readiness.json", dict(status="failed", error=adoption.RUNTIME_ERROR,
        plan_sha256=source_sha, job_id="102", archive_sha256=old_archive, manifest_sha256=old_manifest, native_panel=panel))
    selection = dict(suite_sha256=suite_sha, job_id="102", winner=None, reason="no_passing_new_objective",
        bound_plans_sha256=bound_sha, rows=[dict(id="legacy3", eligible=True, gates=dict.fromkeys(REQUIRED_GATES, True),
            readiness=dict(native_panel=dict(source_snapshot_chain_sha256=panel["source_snapshot_chain_sha256"])))])
    put(suite_root / "selection.json", selection)
    put(suite_root / "promotion.json", dict(status="stopped_no_passing_new_objective", selection=selection))
    plan = dict(deepcopy(source), release=str(new_release), campaign=str(new_root), archive_sha256=new_archive,
        manifest_sha256=new_manifest, runtime_policy_amendment=dict(version="chemistry-epoch-fit-amendment-v1",
            factor=1.25, reserve_seconds=1800), runtime_adoption=dict(version=adoption.VERSION,
            source_plan=str(source_path), source_plan_sha256=source_sha, artifacts=artifacts))
    plan["_plan_sha256"] = hashlib.sha256(adoption._encode(plan)).hexdigest()
    return SimpleNamespace(plan=plan, source=source, old_root=old_root, suite_root=suite_root,
        new_root=new_root, old_release=old_release, new_release=new_release, artifacts=artifacts,
        put=put, states=states, native=native, panel=panel, ancestor=ancestor)


def test_full_bound_runtime_only_adoption_is_additive_and_reauthenticates(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    before = {p: Path(p).read_bytes() for p in case.artifacts}
    receipt = adoption.materialize_adoption(case.plan)
    sha = adoption._sha(case.new_root / "failure_readiness.json")
    assert adoption.authenticate_adoption(case.plan, sha) == receipt
    assert receipt["evidence_kind"] == adoption.KIND
    assert receipt["source_evidence"]["original_status"] == "failed"
    gate = adoption._read(case.new_root / "benchmark_gate.json")
    assert gate["status"] == "passed" and gate["evidence_kind"] == adoption.KIND
    assert gate["projected_epoch_seconds"] == receipt["conservative_epoch_seconds"]
    assert all(receipt["gates"].values()) and receipt["amended_required_seconds"] < 82800
    assert all(Path(p).read_bytes() == b for p, b in before.items())
    with pytest.raises(ValueError, match="uncertain adoption"):
        adoption.materialize_adoption(case.plan)


@pytest.mark.parametrize("gate", sorted(REQUIRED_GATES - {"complete_epoch_fits"}))
def test_no_nonruntime_native_failure_is_waived(tmp_path, monkeypatch, gate):
    case = fixture(tmp_path, monkeypatch)
    case.native["gates"][gate] = False
    case.put(case.old_root / "benchmark_gate.json", case.native)
    with pytest.raises(ValueError, match="runtime-only"):
        adoption.materialize_adoption(case.plan)
    assert not case.new_root.exists()


@pytest.mark.parametrize("defect", ["protocol", "ratio", "checkpoint", "molecular_bytes", "missing_proof",
    "changed_bytes", "running_native", "pending_callback", "unknown_cpu", "successor", "ancestor_successor"])
def test_changed_science_unbound_proof_or_active_predecessor_blocks_materialization(tmp_path, monkeypatch, defect):
    case = fixture(tmp_path, monkeypatch)
    if defect == "protocol": case.plan["protocol"]["loss_changed"] = True
    elif defect == "ratio": case.plan["protocol"]["prior_ratio"] = 3
    elif defect == "checkpoint": case.plan["starting_checkpoint"]["epoch"] = 37
    elif defect == "molecular_bytes": (case.new_release / "src/latentmol/model.py").write_text("changed molecular code")
    elif defect == "missing_proof": case.artifacts.pop(str(case.suite_root / "loss-validation-gate.json"))
    elif defect == "changed_bytes": (case.old_root / "failure_readiness.json").write_text("{}")
    elif defect == "running_native": case.states["102"] = "RUN"
    elif defect == "pending_callback": case.states["103"] = "PEND"
    elif defect == "unknown_cpu": case.states["101"] = "UNKNOWN"
    else:
        root = case.old_root if defect == "successor" else case.ancestor
        case.put(root / "receipts/segment-002-intent.json", {}, bind=False)
    with pytest.raises((ValueError, OSError)):
        adoption.materialize_adoption(case.plan)
    assert not case.new_root.exists()


@pytest.mark.parametrize("defect", ["validity", "fd", "gradient_count", "atom_edits", "profile_epoch", "cuda_skips"])
def test_reconstructed_supplementary_proofs_are_all_hard_requirements(tmp_path, monkeypatch, defect):
    case = fixture(tmp_path, monkeypatch)
    if defect == "validity":
        case.native["updated_population"]["raw_argmax_validity"] = 0
        case.put(case.old_root / "benchmark_gate.json", case.native)
    elif defect in ("fd", "gradient_count", "atom_edits"):
        if defect == "fd": case.panel["native_directional_check"]["passed"] = False
        elif defect == "gradient_count": case.panel["counts"]["failed_with_nonzero_gradient"] = 0
        else: case.panel["complete_target_atom_edits"] = 1
        p = case.old_root / "failure_readiness.json"; value = adoption._read(p); value["native_panel"] = case.panel
        case.put(p, value)
    elif defect == "profile_epoch":
        p = case.suite_root / "failure_preparation_profile.json"; value = adoption._read(p); value["completed_epoch"] = 35
        sha = case.put(p, value); gate = adoption._read(case.suite_root / "cpu_gate.json"); gate["execution_profile_sha256"] = sha
        case.put(case.suite_root / "cpu_gate.json", gate)
    else:
        case.put(case.suite_root / "loss-validation-pytest.xml",
            '<testsuites><testsuite tests="22" skipped="6" errors="0" failures="0" /></testsuites>', raw=True)
    with pytest.raises(ValueError):
        adoption.materialize_adoption(case.plan)


@pytest.mark.parametrize("name", ["failure_readiness.json", "benchmark_gate.json", "training_composition.json"])
def test_new_receipts_and_copied_composition_cannot_change_after_materialization(tmp_path, monkeypatch, name):
    case = fixture(tmp_path, monkeypatch)
    adoption.materialize_adoption(case.plan)
    sha = adoption._sha(case.new_root / "failure_readiness.json")
    (case.new_root / name).write_text("{}")
    with pytest.raises(ValueError):
        adoption.authenticate_adoption(case.plan, sha)


def test_exact_aged_out_jobs_still_need_the_persisted_done_logs(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    case.states.update({job: "missing" for job in ("101", "102", "103")})
    adoption.materialize_adoption(case.plan)
    case.put(Path(case.plan["work_root"]) / "logs/native.102.out", "Subject: Job 999: Wrong\n", raw=True)
    with pytest.raises(ValueError, match="LSF DONE"):
        adoption.authenticate_adoption(case.plan, adoption._sha(case.new_root / "failure_readiness.json"))


def test_unknown_or_partial_materialization_is_never_replayed(tmp_path, monkeypatch):
    case = fixture(tmp_path, monkeypatch)
    case.new_root.mkdir(parents=True)
    (case.new_root / "runtime-adoption-intent.json").write_text("{}")
    with pytest.raises(ValueError, match="uncertain adoption"):
        adoption.materialize_adoption(case.plan)
