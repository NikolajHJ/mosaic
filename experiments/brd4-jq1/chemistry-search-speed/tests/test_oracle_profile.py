"""Allocated-only exact-passthrough fixtures for detached-oracle telemetry."""

from pathlib import Path
import json
import sys

import jax.numpy as jnp
import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from profile_oracle import profile_oracle
from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec, prepare_failure_guidance
from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig


@pytest.mark.parametrize("name", ("DetectChemistryProblems", "SanitizeMol"))
def test_arguments_keyword_objects_and_return_identity_pass_through(monkeypatch, name):
    calls, result = [], object()
    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    sentinel, option = object(), object()
    with monkeypatch.context() as patch:
        patch.setattr(Chem, name, native)
        with profile_oracle() as report:
            assert getattr(Chem, name)(sentinel, option, catchErrors=True) is result
        assert getattr(Chem, name) is native
    assert len(calls) == 1
    assert calls[0][0][0] is sentinel and calls[0][0][1] is option
    assert calls[0][1] == {"catchErrors": True}
    row = report["functions"][name]
    assert row["calls"] == 1 and row["errors"] == 0 and row["seconds"] >= 0
    assert report["arguments_and_results_changed"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("name", ("DetectChemistryProblems", "SanitizeMol"))
@pytest.mark.parametrize("kind", (ValueError, TypeError, RuntimeError))
def test_exact_exception_object_propagates_and_original_binding_restored(monkeypatch, name, kind):
    error = kind("unchanged native error")
    def native(*args, **kwargs):
        raise error
    with monkeypatch.context() as patch:
        patch.setattr(Chem, name, native)
        with pytest.raises(kind) as observed:
            with profile_oracle() as report:
                getattr(Chem, name)(object())
        assert observed.value is error
        assert getattr(Chem, name) is native
    row = report["functions"][name]
    assert row["calls"] == row["errors"] == 1
    assert row["seconds"] >= 0


@pytest.mark.parametrize("args, kwargs, mode", (
    ((object(),), {}, "unspecified"),
    ((object(), 1, True), {}, "true"),
    ((object(), 1, False), {}, "false"),
    ((object(),), {"catchErrors": True}, "true"),
    ((object(),), {"catchErrors": False}, "false"),
    ((object(),), {"catchErrors": 1}, "other_type"),
))
def test_catch_errors_flag_telemetry_never_changes_native_arguments(monkeypatch, args, kwargs, mode):
    calls = []
    def native(*given, **options):
        calls.append((given, options))
        return 0
    with monkeypatch.context() as patch:
        patch.setattr(Chem, "SanitizeMol", native)
        with profile_oracle() as report:
            assert Chem.SanitizeMol(*args, **kwargs) == 0
    assert calls == [(args, kwargs)]
    counts = report["functions"]["SanitizeMol"]["catch_errors_counts"]
    assert counts[mode] == 1 and sum(counts.values()) == 1


def test_nested_profile_refused_and_outer_wrapper_preserved():
    originals = (Chem.DetectChemistryProblems, Chem.SanitizeMol)
    with profile_oracle():
        wrapped = (Chem.DetectChemistryProblems, Chem.SanitizeMol)
        with pytest.raises(RuntimeError, match="nested"):
            with profile_oracle():
                pytest.fail("Nested profiler unexpectedly accepted")
        assert (Chem.DetectChemistryProblems, Chem.SanitizeMol) == wrapped
    assert (Chem.DetectChemistryProblems, Chem.SanitizeMol) == originals


def test_unrelated_body_error_restores_both_native_functions():
    originals = (Chem.DetectChemistryProblems, Chem.SanitizeMol)
    with pytest.raises(LookupError, match="body"):
        with profile_oracle():
            raise LookupError("body")
    assert (Chem.DetectChemistryProblems, Chem.SanitizeMol) == originals


@pytest.mark.parametrize("invalid", (False, True))
def test_real_small_preparation_identity_and_every_array_unaffected(invalid):
    n = 3
    spec = mosaic_spec(n)
    atoms = jnp.full((n, 6), -1.0, dtype=jnp.float32).at[:, 4 if invalid else 0].set(2.0)
    bonds = jnp.full((3, 5), -1.0, dtype=jnp.float32)
    bonds = bonds.at[:, 3 if invalid else 1].set(2.0)
    logits = GraphLogits(atoms, bonds)
    config = ProbabilityRepairConfig(max_edits=1, checks_per_depth=2, beam_width=1)
    original = prepare_failure_guidance(logits.decoded(), spec, search=config)
    with profile_oracle() as report:
        observed = prepare_failure_guidance(logits.decoded(), spec, search=config)
    original.verify(spec)
    observed.verify(spec)
    assert observed.receipt == original.receipt and observed.identity == original.identity
    for name in original.arrays:
        left, right = np.asarray(original.arrays[name]), np.asarray(observed.arrays[name])
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    assert report["functions"]["DetectChemistryProblems"]["calls"] > 0
    assert report["functions"]["SanitizeMol"]["calls"] > 0
    json.dumps(report, allow_nan=False)
