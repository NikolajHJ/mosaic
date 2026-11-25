"""Allocated CPU fixtures for exact saved-parent native recovery selection."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from latentmol.boltz import embedding_native_recovery as subject
from latentmol.boltz.embedding_comparison_experiment import native_members
from latentmol.boltz.embedding_comparison_rescore import freeze_selection, load_target, NATIVE_SETTINGS


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return subject._sha(path)


def parent_fixture(tmp_path):
    root = tmp_path / "parent/gpu"
    root.mkdir(parents=True)
    msa = tmp_path / "query.a3m"; msa.write_text(">query\nAAA\n")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": "AAA", "msa": str(msa)}},
        {"ligand": {"id": "B", "smiles": "CCC"}}], "properties": [{"affinity": {"binder": "B"}}]}))
    asset = lambda path: {"path": str(path), "sha256": subject._sha(path)}
    affinity = tmp_path / "aff.ckpt"; affinity.write_bytes(b"not loaded; hash-only fixture")
    confidence = tmp_path / "conf.ckpt"; confidence.write_bytes(b"not loaded; other native weight")
    config = dict(fresh_native={"pose_seeds": [17, 23], "native_settings": NATIVE_SETTINGS,
        "native_timeout_seconds": 1800, "audit_timeout_seconds": 600,
        "target_sequence_sha256": hashlib.sha256(b"AAA").hexdigest(),
        "msa_sha256": subject._sha(msa)}, native_template=asset(template),
        affinity=asset(affinity), dbmol={"confidence_checkpoint": asset(confidence)}, boltz_cache=str(tmp_path))
    config_path = tmp_path / "original-plan.json"; put(config_path, config)
    hashes = {}; summaries = []; gates = []; members = []
    for index, smiles in enumerate(("CCO", "CCC", "CCCC"), 1):
        case_id = f"case-{index:03d}"
        runs = []
        for arm in ("relaxed", "learned"):
            checkpoints = []
            for update in (0, 49):
                checkpoint = root / case_id / "optimization" / arm / f"checkpoint-{update:04d}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"{case_id}-{arm}-{update}".encode())
                checkpoints.append(dict(update=update, tensor_path=str(checkpoint),
                    tensor_sha256=subject._sha(checkpoint), latent_sha256="b" * 64,
                    classifications={"conditioned": dict(smiles=smiles, strict_teacher_valid=True, reason=None)}))
            runs.append(dict(path=arm, status="completed", applied=50, checkpoints=checkpoints,
                             best_observed={"update": 49}))
        comparison = dict(status="completed_comparison_not_binding_validated", runs=runs)
        relative = case_id + "/optimization/comparison.json"
        hashes[relative] = put(root / relative, comparison)
        summaries.append(dict(case_id=case_id, full_record=str(root / relative)))
        members.extend(native_members(case_id, comparison))
        gates.append(dict(case_id=case_id, native_oracle="passed_native_oracle_only",
            start_gates={arm: "passed_pointwise_start_only" for arm in ("relaxed", "learned")}))
        values = {"native-oracle.json": {"status": "passed_native_oracle_only"},
            "relaxed-start-gate.json": {"passed": True, "status": "passed_pointwise_start_only"},
            "learned-start-gate.json": {"passed": True, "status": "passed_pointwise_start_only"},
            "production-binding.json": {arm: {"passed": True} for arm in ("relaxed", "learned")}}
        for name, value in values.items():hashes[case_id + "/" + name] = put(root / case_id / name, value)
    selection = freeze_selection(members)
    target = load_target(template, config["fresh_native"]["target_sequence_sha256"], subject._sha(msa),
                         template_sha256=subject._sha(template))
    result = dict(status="completed_matched_pilot", frozen_vae_unchanged=True,
        plan_sha256=subject._sha(config_path), cases=summaries, start_gates=gates,
        native_followup={"successful_native_requests": 0})
    for relative, value in {"result.json": result, "case-summary.json": {"cases": summaries},
        "start-gates.json": {"cases": gates}, "fresh-native/selection.json": selection,
        "fresh-native/target.json": target}.items():hashes[relative] = put(root / relative, value)
    plan = dict(version=subject.VERSION, internal_seconds=18000,
        parent=dict(root=str(root), plan=asset(config_path), artifacts=hashes))
    return SimpleNamespace(plan=plan, root=root, config=config, selection=selection, result=result)


def test_recovery_reconstructs_the_exact_six_requests_and_selected_checkpoints(tmp_path):
    fixture = parent_fixture(tmp_path)
    config, summaries, selection, target, checkpoints = subject.authenticate_parent(fixture.plan)
    assert selection == fixture.selection and len(selection["requests"]) == 6
    assert len(selection["members"]) == 15 and len(checkpoints) == 9
    assert len(summaries) == 3 and config == fixture.config


@pytest.mark.parametrize("change", ["artifact", "gate", "selection", "checkpoint", "budget", "path"])
def test_parent_tampering_or_budget_change_never_releases_native_prediction(tmp_path, change):
    fixture = parent_fixture(tmp_path); plan = deepcopy(fixture.plan)
    if change == "artifact":
        (fixture.root / "case-summary.json").write_text("{}")
    elif change == "gate":
        name = "case-001/learned-start-gate.json"
        plan["parent"]["artifacts"][name] = put(fixture.root / name, {"passed": False})
    elif change == "selection":
        value = deepcopy(fixture.selection); value["pose_seeds"] = [1, 2]
        name = "fresh-native/selection.json"; plan["parent"]["artifacts"][name] = put(fixture.root / name, value)
    elif change == "checkpoint":
        Path(fixture.selection["members"][0]["checkpoint_path"]).write_bytes(b"replaced")
    elif change == "budget":
        plan["internal_seconds"] = 1000
    else:
        plan["parent"]["artifacts"]["../outside.json"] = "a" * 64
    with pytest.raises(ValueError):subject.authenticate_parent(plan)


def test_validation_mode_never_launches_a_cuda_child(tmp_path, monkeypatch):
    fixture = parent_fixture(tmp_path)
    path = tmp_path / "recovery-plan.json"; put(path, fixture.plan)
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "a" * 64)
    monkeypatch.setattr(subject, "cuda_child_smoke", lambda *a: pytest.fail("CPU validation never initializes CUDA"))
    result = subject.run(path, subject._sha(path), tmp_path / "validation", mode="validate")
    assert result["status"] == "passed_authentication_not_prediction"
    assert not (tmp_path / "validation/fresh-native").exists()
    with pytest.raises(ValueError, match="fresh directory"):
        subject.run(path, subject._sha(path), tmp_path / "validation", mode="validate")


def test_recovery_refuses_to_write_inside_completed_parent(tmp_path, monkeypatch):
    fixture = parent_fixture(tmp_path)
    path = tmp_path / "recovery-plan.json"; put(path, fixture.plan)
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "a" * 64)
    with pytest.raises(ValueError, match="outside"):
        subject.run(path, subject._sha(path), fixture.root / "new", mode="validate")
