"""Allocated synthetic controls of the original mandatory raw FD contract."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def load_checker():
    try:
        import dbmol_native_validation
        return dbmol_native_validation
    except ModuleNotFoundError:
        pass
    path = Path(__file__).with_name("native_validation.py")
    spec = importlib.util.spec_from_file_location("dbmol_native_validation_fixture", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = load_checker()


def arrays():
    return np.full((25, 6), 1 / 6, np.float32), np.full((300, 5), .2, np.float32)


def gradients():
    return (np.tile(np.arange(6, dtype=np.float32), (25, 1)),
            np.tile(np.arange(5, dtype=np.float32), (300, 1)))


def linear_primal(x, e):
    # FP64 arithmetic at the scalar head, with the original FP32 graph inputs.
    return float((x.astype(np.float64) * np.arange(6)).sum()
                 + (e.astype(np.float64) * np.arange(5)).sum())


def test_raw_checker_accepts_independent_linear_primal_and_records_every_endpoint():
    x, e = arrays()
    gx, ge = gradients()
    calls = []
    def independent(ax, ae):
        calls.append((ax.copy(), ae.copy()))
        return linear_primal(ax, ae)
    result = checker.check_raw_finite_differences(independent, x, e, gx, ge,
                                                 objective="synthetic_only")
    assert result["status"] == "passed" and len(calls) == 12
    assert result["epsilons"] == [.01, .003, .001]
    assert result["relative_tolerance"] == .15
    assert result["required_adjacent_passes"] == 2
    assert result["typed_bond_contrast_nonzero"] is True
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert len(row["probes"]) == 3
        assert all(p["agrees"] and len(p["endpoint_graph_sha256"]) == 2 for p in row["probes"])


@pytest.mark.parametrize("field", ("X", "E_upper"))
@pytest.mark.parametrize("fault", ("zero", "nonfinite", "sign", "scale"))
def test_raw_checker_rejects_missing_wrong_or_nonfinite_derivative_paths(field, fault):
    x, e = arrays()
    gx, ge = gradients()
    target = gx if field == "X" else ge
    if fault == "zero": target[:] = 0
    elif fault == "nonfinite": target[0, 0] = np.nan
    elif fault == "sign": target *= -1
    else: target *= 2
    with pytest.raises(checker.DerivativeCheckError) as error:
        checker.check_raw_finite_differences(linear_primal, x, e, gx, ge, objective="negative_control")
    assert error.value.report["status"] == "failed"
    assert error.value.report["rows"][-1]["field"] == field


@pytest.mark.parametrize("passing_scales", ((.001,), (.01, .001), ()))
def test_one_scale_or_nonadjacent_scales_never_certify_original_gate(passing_scales):
    x, e = arrays()
    gx, ge = gradients()
    baseline, base_atom = linear_primal(x, e), float(x[0, 5])
    def curved(ax, ae):
        epsilon = abs(float(ax[0, 5]) - base_atom)
        if epsilon == 0:
            return linear_primal(ax, ae)
        nearest = min(checker.EPSILONS, key=lambda value: abs(value - epsilon))
        return baseline + (linear_primal(ax, ae) - baseline) * (1 if nearest in passing_scales else -1)
    with pytest.raises(checker.DerivativeCheckError) as error:
        checker.check_raw_finite_differences(curved, x, e, gx, ge, objective="curved_fixture_only")
    probes = error.value.report["rows"][0]["probes"]
    assert [p["epsilon"] for p in probes if p["agrees"]] == list(passing_scales)


def test_infeasible_original_epsilon_fails_without_replacement_direction_or_projection():
    x, e = arrays()
    x[0] = np.array([.0001, .19998, .19998, .19998, .19998, .19998], np.float32)
    gx, ge = gradients()
    calls = []
    with pytest.raises(checker.DerivativeCheckError) as error:
        checker.check_raw_finite_differences(lambda ax, ae: calls.append(1) or linear_primal(ax, ae),
                                             x, e, gx, ge, objective="boundary_fixture_only")
    assert "simplex interior" in str(error.value)
    assert calls == []
    assert error.value.report["rows"][0]["infeasible_epsilon"] == .01
    assert error.value.report["rows"][0]["probes"] == []


def test_present_bond_type_contrast_is_separate_from_bond_existence_signal():
    x, e = arrays()
    gx, ge = gradients()
    ge[:, 1:] = 1
    def existence_only(ax, ae):
        return float((ax.astype(np.float64) * np.arange(6)).sum()
                     + ae[:, 1:].astype(np.float64).sum())
    with pytest.raises(checker.DerivativeCheckError, match="present-bond-type"):
        checker.check_raw_finite_differences(existence_only, x, e, gx, ge,
                                             objective="typed_null_fixture_only")


@pytest.mark.parametrize("fault", ("shape", "dtype", "simplex", "nonfinite"))
def test_checker_rejects_bad_graph_input_before_any_model_request(fault):
    x, e = arrays()
    gx, ge = gradients()
    if fault == "shape": x = x[:-1]
    elif fault == "dtype": x = x.astype(np.float64)
    elif fault == "simplex": x[0, 0] = -.1
    else: e[0, 0] = np.nan
    calls = []
    with pytest.raises(ValueError):
        checker.check_raw_finite_differences(lambda *args: calls.append(1), x, e, gx, ge,
                                             objective="bad_graph_fixture_only")
    assert calls == []


def test_pretrained_checker_refuses_execution_outside_allocation(monkeypatch, tmp_path):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(RuntimeError, match="allocated DTU HPC"):
        checker.validate_native_components(None, None, None, tmp_path / "never_created",
                                            hard_graph=None, make_loss=None)
    assert not (tmp_path / "never_created").exists()


@pytest.mark.parametrize("fault", ("missing", "changed_digest", "not_frozen", "malformed"))
def test_native_frozen_identity_requires_all_models_full_context_and_assets(fault):
    raw = {name: "a" * 64 for name in (
        "model_digest", "affinity_model_digest", "structural_model_digest",
        "context_digest", "full_context_digest", "context_file_sha256",
        "checkpoint_sha256", "structural_checkpoint_sha256")}
    raw["frozen"] = True
    if fault == "missing": del raw["structural_model_digest"]
    elif fault == "changed_digest": raw["full_context_digest"] = None
    elif fault == "not_frozen": raw["frozen"] = False
    else: raw["context_digest"] = "bad"
    with pytest.raises(AssertionError):
        checker._identity(raw)
