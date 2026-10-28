"""Native coordinator association fixtures; numerical tests are HPC-only."""

import hashlib
import json
import time
from pathlib import Path

import pytest
import torch

from latentmol.boltz import embedding_long_experiment as long
from latentmol.representation import GraphSpec


def completed(request):
    return dict(id=request["id"], smiles=request["smiles"], seed=request["seed"],
        status="completed", retention_eligible=True,
        affinity=dict(affinity_probability_binary=.6, affinity_pred_value=-.2))


def settings():
    return dict(fresh_native=dict(pose_seeds=[17, 23], native_settings={"pinned": "settings"}))


def test_native_success_is_associated_with_exact_request():
    request = long.request_for(settings(), "CC", 17)
    result = completed(request)
    assert long.validate_native_result(result, request) is result


@pytest.mark.parametrize("field,value,error", [
    ("id", "wrong", ValueError), ("smiles", "CCC", ValueError), ("seed", 23, ValueError),
    ("status", "native_failed", RuntimeError), ("retention_eligible", False, RuntimeError),
    ("affinity", {}, ValueError),
    ("affinity", {"affinity_probability_binary": 1.1, "affinity_pred_value": -.2}, ValueError),
    ("affinity", {"affinity_probability_binary": .5, "affinity_pred_value": float("nan")}, ValueError)])
def test_reused_native_result_cannot_bypass_association_success_or_finiteness(field, value, error):
    request = long.request_for(settings(), "CC", 17)
    result = completed(request)
    result[field] = value
    with pytest.raises(error):
        long.validate_native_result(result, request)


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    return long.asset(path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def test_smoke_reads_authenticated_tensor_starts_not_unbound_json(tmp_path, monkeypatch):
    from latentmol import cuda_process
    from latentmol.boltz import embedding_native_recovery
    starts = dict(cases=[dict(record=dict(case_id=name, identity="C" * (i + 1)))
                        for i, name in enumerate(long.CASES)])
    parent = save_tensor(tmp_path / "parent.pt", starts)
    config = settings() | dict(precision_correction=dict(parent_starts=parent))
    long.write(tmp_path / "cpu/starts.json", dict(cases=[dict(identity="WRONG")]))
    calls = []
    monkeypatch.setattr(cuda_process, "require_cuda_free_coordinator", lambda: calls.append("cuda_free"))
    monkeypatch.setattr(embedding_native_recovery, "cuda_child_smoke", lambda *args: calls.append("probe"))
    monkeypatch.setattr(long, "native_target", lambda *args: {"fixed_target": True})

    def execute(config, target, request, output, deadline):
        calls.append(request)
        long.write(output / "result.json", completed(request))
        return completed(request)

    monkeypatch.setattr(long, "execute_native", execute)
    out = tmp_path / "pilot"
    result = long.native_smoke(config, tmp_path, out, time.monotonic() + 1000)
    assert result["smiles"] == "C" and result["seed"] == 17
    gate = json.loads((out / "native-smoke-gate.json").read_text())
    assert gate["starts_asset"] == parent and gate["counted_once"] is True
    assert calls[0] == calls[-1] == "cuda_free"
    assert len([v for v in calls if isinstance(v, dict)]) == 1


def followup_fixture(tmp_path, monkeypatch):
    from latentmol import cuda_process
    monkeypatch.setattr(cuda_process, "require_cuda_free_coordinator", lambda: None)
    source, plan = "a" * 64, "b" * 64
    identity = dict(source_sha256=source, plan_sha256=plan)
    model = dict(checkpoint_sha256="c" * 64, calibration_sha256="d" * 64)
    spec = GraphSpec(max_nodes=3)
    starts = dict(graph_spec=spec.to_dict(), cases=[dict(record=dict(case_id=name, identity="C" * (i + 1)))
                                                   for i, name in enumerate(long.CASES)])
    starts_asset = save_tensor(tmp_path / "cpu/starts.pt", starts)
    config = settings() | dict(precision_correction=dict(parent_starts=starts_asset))
    target = {"fixed_target": True}
    monkeypatch.setattr(long, "native_target", lambda *args: target)
    long.write(tmp_path / "cpu/result.json", dict(**identity, status="passed", artifacts=dict(starts=starts_asset)))
    long.write(tmp_path / "fit/passed-model.json", model)
    long.write(tmp_path / "fit/result.json", dict(**identity, status="passed", model_identity=model,
                                                artifacts=dict(model=long.asset(tmp_path / "fit/passed-model.json"))))
    final = dict(identity=identity, cases={})
    for i, name in enumerate(long.CASES, 1):
        root = tmp_path / "trajectories" / name
        z = torch.tensor([[float(i), 0]])
        atoms = torch.zeros(1, 3, dtype=torch.long)
        atoms[0, :i] = 1
        charges = torch.full((1, 3), spec.formal_charges.index(0), dtype=torch.long)
        bonds = torch.zeros(1, 3, 3, dtype=torch.long)
        for j in range(i - 1):
            bonds[0, j, j + 1] = bonds[0, j + 1, j] = 1
        tensors = dict(z=z, conditioned=dict(atoms=atoms, charges=charges, bonds=bonds))
        binding = dict(**identity, bridge_checkpoint_sha256=model["checkpoint_sha256"],
                       calibration_sha256=model["calibration_sha256"])
        signature = dict(binding=binding, case_id=name)
        signature_sha = digest_json(signature)
        long.write(root / "manifest.json", dict(signature=signature, sha256=signature_sha))
        observations = []
        for update in (0, 10000):
            tensor_asset = save_tensor(root / "observations" / f"update-{update:06d}.pt", tensors)
            record_path = root / "observations" / f"update-{update:06d}.json"
            long.write(record_path, dict(update=update, tensors=tensor_asset, signature_sha256=signature_sha))
            observations.append(dict(update=update, record=long.asset(record_path), tensors=tensor_asset))
        state = dict(update=10000, case_id=name, signature_sha256=signature_sha, z0=z, z=z,
            best_valid=dict(tensors=observations[0]["tensors"]),
            baseline_observation=observations[0], last_observation=observations[1])
        saved = save_tensor(root / "states/state-010000.pt", state)
        final["cases"][name] = dict(completed_updates=10000, state=saved["path"], state_sha256=saved["sha256"])
    request = long.request_for(config, "C", 17)
    long.write(tmp_path / "pilot/native-smoke/result.json", completed(request))
    smoke = dict(status="passed", counted_once=True, request=request,
        result=long.asset(tmp_path / "pilot/native-smoke/result.json"),
        target_sha256=hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest(),
        starts_asset=starts_asset)
    long.write(tmp_path / "pilot/native-smoke-gate.json", smoke)
    long.write(tmp_path / "pilot/result.json", dict(**identity, status="passed",
        artifacts=dict(native_smoke=long.asset(tmp_path / "pilot/native-smoke-gate.json"))))
    executed = []

    def execute(config, target, request, output, deadline):
        executed.append(request)
        long.write(output / "result.json", completed(request))
        return completed(request)

    monkeypatch.setattr(long, "execute_native", execute)
    return config, final, executed


def test_followup_verifies_pinned_observations_and_counts_smoke_exactly_once(tmp_path, monkeypatch):
    config, final, executed = followup_fixture(tmp_path, monkeypatch)
    out = tmp_path / "followup"
    result = long.followup(config, tmp_path, out, time.monotonic() + 1000, final)
    summary = json.loads((out / "summary.json").read_text())
    assert result["status"] == "passed" and result["all_requests_accounted"] is True
    assert len(summary["members"]) == 9 and len(summary["requests"]) == len(summary["results"]) == 6
    assert len(executed) == 5
    smoke_id = long.request_for(config, "C", 17)["id"]
    assert sum(row["id"] == smoke_id for row in summary["results"]) == 1


def test_followup_rejects_changed_terminal_observation_without_replaying_models(tmp_path, monkeypatch):
    config, final, executed = followup_fixture(tmp_path, monkeypatch)
    path = tmp_path / "trajectories/case-001/observations/update-010000.json"
    value = json.loads(path.read_text())
    value["update"] = 9999
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        long.followup(config, tmp_path, tmp_path / "followup", time.monotonic() + 1000, final)
    assert not executed


def test_fully_rebound_smoke_receipts_still_cannot_reuse_another_seed(tmp_path, monkeypatch):
    config, final, executed = followup_fixture(tmp_path, monkeypatch)
    result_path = tmp_path / "pilot/native-smoke/result.json"
    result = json.loads(result_path.read_text())
    result["seed"] = 23
    result_path.write_text(json.dumps(result), encoding="utf-8")
    gate_path = tmp_path / "pilot/native-smoke-gate.json"
    gate = json.loads(gate_path.read_text())
    gate["result"] = long.asset(result_path)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    pilot_path = tmp_path / "pilot/result.json"
    pilot = json.loads(pilot_path.read_text())
    pilot["artifacts"]["native_smoke"] = long.asset(gate_path)
    pilot_path.write_text(json.dumps(pilot), encoding="utf-8")
    with pytest.raises(ValueError, match="association differs"):
        long.followup(config, tmp_path, tmp_path / "followup", time.monotonic() + 1000, final)
    assert not executed
