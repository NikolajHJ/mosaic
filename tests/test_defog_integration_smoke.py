"""HPC receipt checks: fallback and partial completion cannot pass the smoke."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from latentmol.repair_guidance.integration_smoke import validate_config, validate_outcome


def fixture():
    result = {"status": "completed", "accepted_updates": 6,
        "repair_evidence": {"accepted_native_targets": 8, "accepted_targets": 2, "refreshes": 2,
            "repair_validation_passed": True, "guided_accepted_updates": 6, "nonzero_guided_updates": 6,
            "target_only_fd_passes": 3}}
    trajectory = {"completed_requested_budget": True, "requested_steps": 6, "fixed_counts": [22],
        "trace": [{"step": step, "accepted": step < 6, "used_target": True, "gradient_measured": step < 6,
            "gradient_diagnostics": {"binding_norm": 1., "weighted_repair_norm": .1}}
            for step in range(7)]}
    gates = {f"step-{step:04d}": {"passed": True, "status": "passed", "native_seed_parity": True,
        "local_fd_validation": {"passed": True},
        "repair_only_validation": {"passed": True, "status": "validated_nonzero"},
        "component_gradients": {key: {"passed": True} for key in
            ("dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss")}}
        for step in (0, 5, 6)}
    return result, trajectory, gates


def test_complete_both_branch_native_evidence_passes():
    report = validate_outcome(*fixture())
    assert report["passed"] and all(report["checks"].values())


@pytest.mark.parametrize("mutation", ["fallback", "partial", "missing_endpoint", "failed_guard",
                                     "no_binding_gradient", "nonfinite_repair", "changed_count", "missing_target",
                                     "repair_fd_missing", "native_parity_missing"])
def test_incomplete_or_false_guidance_cannot_pass(mutation):
    result, trajectory, gates = deepcopy(fixture())
    if mutation == "fallback":
        result["repair_evidence"]["nonzero_guided_updates"] = 0
    elif mutation == "partial":
        result["accepted_updates"] = 5
    elif mutation == "missing_endpoint":
        trajectory["trace"].pop()
    elif mutation == "failed_guard":
        gates["step-0005"]["passed"] = False
    elif mutation == "no_binding_gradient":
        trajectory["trace"][0]["gradient_diagnostics"]["binding_norm"] = 0.
    elif mutation == "nonfinite_repair":
        trajectory["trace"][0]["gradient_diagnostics"]["weighted_repair_norm"] = float("nan")
    elif mutation == "changed_count":
        trajectory["fixed_counts"] = [22, 23]
    elif mutation == "missing_target":
        result["repair_evidence"]["accepted_targets"] = 1
    elif mutation == "repair_fd_missing":
        gates["step-0005"]["repair_only_validation"] = {"status": "zero_at_current_point_uninformative"}
    elif mutation == "native_parity_missing":
        gates["step-0000"]["native_seed_parity"] = False
    assert not validate_outcome(result, trajectory, gates)["passed"]


@pytest.mark.parametrize("changed", [None, "decoder", "fd_rtol", "initial_latent", "charge_policy", "steps"])
def test_configuration_only_allows_declared_precision_and_cadence(changed):
    original = {"name": "parent", "output_root": "/old", "gradient_audit_interval": 1,
                "dbmol": {}, "decoder": {"sha256": "a" * 64}, "fd_rtol": .05,
                "initial_latent": {"sha256": "b" * 64}}
    base = SimpleNamespace(record=lambda: deepcopy(original), optimizer=SimpleNamespace(steps=6),
        gradient_audit_interval=1, arms=("fixed_initial_mask",), fd_policy="local_v1",
        step_control=object(), dbmol=object(), start=SimpleNamespace(kind="standard_normal", seed=0),
        initial_latent=original["initial_latent"], repair_guidance=SimpleNamespace(refresh_interval=5,
            attempts_per_refresh=4, weight_schedule=object(), provider={}))
    new = deepcopy(original)
    new.update(name="smoke", output_root="/new", gradient_audit_interval=6)
    new["dbmol"]["objective_arithmetic"] = "contact_and_assembly_float64_v1"
    if changed:
        new[changed] = "undeclared"
    candidate = SimpleNamespace(record=lambda: deepcopy(new), name="smoke", output_root="/new")
    if changed:
        with pytest.raises(ValueError, match="undeclared"):
            validate_config(base, candidate)
    else:
        validate_config(base, candidate)


def test_submitter_requires_cpu_gate_and_managed_a100_mps():
    import importlib.util
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts/hpc"))
    spec = importlib.util.spec_from_file_location("smoke_submitter", root / "scripts/hpc/submit_defog_integration_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.workflow(Path("/work3/u/release"), Path("/work3/u"), Path("/work3/u/plan.json"),
                           "a" * 64, "b" * 64, "c" * 64)
    cpu, native = rows
    assert cpu["stage"] == "cpu" and native["dependency"] == "done:cpu"
    assert "2:00" in native["arguments"] and "select[gpu80gb]" in native["arguments"]
    assert "num=1:mode=exclusive_process:mps=yes:j_exclusive=yes" in native["arguments"]
    assert "latentmol.repair_guidance.integration_smoke" in native["job_command"]
