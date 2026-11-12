"""Compatibility/locality instrumentation, using a transparent fake teacher."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("stage3_teacher_probe", Path(__file__).parents[1] / "scripts/stage3_teacher_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def graph(n=4):
    rng = np.random.default_rng(17)
    return SimpleNamespace(X=rng.dirichlet(np.ones(6), size=n).astype(np.float32),
                           E_upper=rng.dirichlet(np.ones(5), size=n * (n - 1) // 2).astype(np.float32))


class Teacher:
    def __init__(self):
        self.queries = 0
        self.identity = {"feature_policy": "fake_identity", "checkpoint": "no_native_model"}
        self.inputs = []

    def predict(self, current, *, time):
        self.queries += 1
        self.inputs.append((np.array(current.X), np.array(current.E_upper), time))
        return SimpleNamespace(X=np.array(current.X), E_upper=np.array(current.E_upper),
                               metadata={"excluded_element_mass_mean": 0.0})

    def verify_frozen(self):
        return {"queries": self.queries, "forwards": self.queries, "state_digest": "unchanged"}


def test_default_32_query_budget_repeatability_bounds_and_metadata():
    teacher, current = Teacher(), graph()
    x, e = current.X.copy(), current.E_upper.copy()
    result = probe.probe_graph(teacher, current)
    assert result["status"] == "complete"
    assert result["design"]["planned_queries"] == 32
    assert teacher.queries == result["successful_predictions"] == result["attempted_predict_calls"] == 32
    assert result["teacher_reported_query_delta"] == 32
    assert result["frozen"]["status"] == "passed"
    repeats = [row for row in result["queries"] if row["kind"] == "repeat"]
    assert len(repeats) == 4 and all(row["prediction_exactly_repeated"] for row in repeats)
    for row in result["queries"]:
        assert row["input_tv_bound"]["satisfied"]
        assert row["prediction_displacement"]["max_tv"] == 0
        assert row["vocabulary_metadata"]["excluded_element_mass_mean"] == 0
        if row["output_delta"] is not None:
            assert row["output_delta"]["max_tv"] == pytest.approx(row["input_delta"]["max_tv"])
    np.testing.assert_array_equal(current.X, x)
    np.testing.assert_array_equal(current.E_upper, e)
    json.dumps(result, allow_nan=False)


def test_direction_shared_across_times_and_scales_and_deterministic():
    first, second = Teacher(), Teacher()
    current = graph()
    probe.probe_graph(first, current, times=(.2, .8), noise_scales=(.001, .01), perturbation_seeds=(1,))
    probe.probe_graph(second, current, times=(.2, .8), noise_scales=(.001, .01), perturbation_seeds=(1,))
    for a, b in zip(first.inputs, second.inputs):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(first.inputs[2][0], first.inputs[6][0])
    np.testing.assert_allclose((first.inputs[2][0] - current.X) / .001,
                               (first.inputs[3][0] - current.X) / .01, atol=2e-5)


def test_individual_prediction_failure_recorded_without_resend():
    class OneFailure(Teacher):
        def predict(self, current, *, time):
            if len(self.inputs) == 2:
                self.inputs.append((None, None, time))
                raise ValueError("unsupported fractional native auxiliary feature")
            return super().predict(current, time=time)
    teacher = OneFailure()
    result = probe.probe_graph(teacher, graph(), times=(.2,), noise_scales=(.001,), perturbation_seeds=(1, 2))
    assert result["status"] == "partial"
    assert result["failed_predictions"] == 1 and result["successful_predictions"] == 3
    assert result["attempted_predict_calls"] == 4
    assert result["queries"][2]["status"] == "failed" and result["queries"][3]["status"] == "passed"
    assert "unsupported fractional" in result["queries"][2]["error"]


def test_closed_worker_skips_remaining_queries_without_blind_retry():
    class Closes(Teacher):
        def predict(self, current, *, time):
            self._closed = True
            raise RuntimeError("native feature failure closed worker")
    result = probe.probe_graph(Closes(), graph(), times=(.2,), noise_scales=(.001,), perturbation_seeds=(1, 2))
    assert result["failed_predictions"] == 1 and result["skipped_queries"] == 3
    assert result["attempted_predict_calls"] == 1
    assert result["frozen"]["status"] == "unavailable"
    assert not result["frozen"]["forward_count_reconciled"]


def test_frozen_or_forward_count_failure_is_fatal_with_collected_observations():
    class BadFrozen(Teacher):
        def verify_frozen(self):
            raise RuntimeError("Frozen model state or query/forward count changed")
    with pytest.raises(probe.ProbeIntegrityError) as error:
        probe.probe_graph(BadFrozen(), graph(), times=(.2,), noise_scales=(), perturbation_seeds=())
    assert len(error.value.result["queries"]) == 2
    assert error.value.result["frozen"]["status"] == "failed"
    assert error.value.result["status"] == "failed"


def test_mutated_input_is_fatal_and_original_caller_graph_unchanged():
    class Mutates(Teacher):
        def predict(self, current, *, time):
            current.X[0] = np.eye(6)[0]
            return super().predict(current, time=time)
    current = graph()
    before = current.X.copy()
    with pytest.raises(probe.ProbeIntegrityError, match="mutated"):
        probe.probe_graph(Mutates(), current, times=(.2,), noise_scales=(), perturbation_seeds=())
    np.testing.assert_array_equal(current.X, before)


def test_nonfinite_prediction_and_explicit_forward_assertion_are_fatal():
    class Nonfinite(Teacher):
        def predict(self, current, *, time):
            reply = super().predict(current, time=time)
            reply.X[0, 0] = float("nan")
            return reply
    with pytest.raises(probe.ProbeIntegrityError, match="invalid prediction") as error:
        probe.probe_graph(Nonfinite(), graph(), times=(.2,), noise_scales=(), perturbation_seeds=())
    assert error.value.result["queries"][0]["status"] == "integrity_failed"
    class BadCount(Teacher):
        def predict(self, current, *, time):
            raise RuntimeError("Teacher graph, time, forward-count or target mismatch")
    with pytest.raises(probe.ProbeIntegrityError, match="integrity failure"):
        probe.probe_graph(BadCount(), graph(), times=(.2,), noise_scales=(), perturbation_seeds=())


def test_inconsistent_returned_forward_count_is_not_claimed_passed():
    class BadCount(Teacher):
        def verify_frozen(self):
            return {"queries": self.queries, "forwards": self.queries + 1}
    with pytest.raises(probe.ProbeIntegrityError) as error:
        probe.probe_graph(BadCount(), graph(), times=(.2,), noise_scales=(), perturbation_seeds=())
    assert error.value.result["frozen"]["status"] == "failed"


def test_empty_bond_head_is_supported():
    result = probe.probe_graph(Teacher(), graph(1), times=(.2,), noise_scales=(.001,), perturbation_seeds=(1,))
    assert result["status"] == "complete"
    assert all(row["input_delta"]["bond_mean_tv"] == 0 for row in result["queries"])


@pytest.mark.parametrize("settings", [{"times": (0., 2.)}, {"noise_scales": (float("nan"),)},
                                     {"perturbation_seeds": (True,)}, {"times": (.2, .2)}])
def test_invalid_probe_design_rejected(settings):
    with pytest.raises(ValueError):
        probe.probe_graph(Teacher(), graph(), **settings)


def test_cli_allocation_gate_precedes_model_import_or_output_writes(tmp_path, monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    output = tmp_path / "fresh"
    with pytest.raises(RuntimeError, match="allocated"):
        probe.main(["--assets", str(tmp_path / "missing.json"), "--output", str(output)])
    assert not output.exists()
