"""Allocated CPU controls; toy observations cannot certify a native endpoint."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path

if not os.environ.get("LSB_JOBID"):
    raise RuntimeError("Numerical forensics fixtures require allocated DTU HPC")

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


forensics = load("protenix_forensics_fixture", "protenix_endpoint_forensics.py")
admission = load("protenix_forensics_admission_fixture", "admission.py")


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get("LSB_JOBID"), "Numerical forensics fixtures require allocated DTU HPC"


def graph(boundary=False):
    x = np.full((25, 6), np.float32(1 / 6))
    e = np.full((300, 5), np.float32(.2))
    if boundary:
        x[10] = np.asarray([.9999, .0001, 0, 0, 0, 0], np.float32)
    return {"X": x, "E_upper": e}


def failed_admission(boundary=False):
    plan = admission.plan(graph(boundary), {"X": [{"row": 10, "source": 1, "target": 2}], "E_upper": []})
    result = {"version": admission.VERSION, "status": "failed", "can_admit_new_child": False,
              "original_run_failed_preserved": True, "can_certify_original_study": False,
              "plan": plan, "failures": ["Preserved synthetic original contradiction"]}
    result["receipt_sha256"] = admission._seal(result)
    return result


def frozen():
    return {"model": "exact-model", "context": "exact-context", "key": 17}


def quadratic(payload, *, scalar_offset=0.0, reduction64=False):
    # Small nonlinearity with an analytic exact endpoint-gradient trapezoid.
    scale = .1
    value = scale * sum(np.sum(v.astype(np.float64)**2, dtype=np.float64) for v in payload.values())
    gradients = {name: (2 * scale * v).astype(np.float32) for name, v in payload.items()}
    contacts = np.tile(np.asarray([-.3, -.4, -.5], np.float32), (25, 1))
    result = {"primal": float(value + scalar_offset), "ad_value": float(value), "gradients": gradients,
              "primal_contacts": contacts.copy(), "ad_contacts": contacts.copy()}
    if reduction64:
        result.update(primal_contacts64=contacts.astype(np.float64), ad_contacts64=contacts.astype(np.float64))
    return result


def run(tmp_path, evaluator=quadratic, record=None, identity=frozen, **kwargs):
    return forensics.run_forensics(failed_admission() if record is None else record,
                                  evaluator, identity, tmp_path / "new", **kwargs)


def observations(report):
    return [probe["analysis"] for direction in report["directions"] for probe in direction["probes"]
            if probe["analysis"]["evaluated"]]


def reseal(record):
    record["plan"]["plan_sha256"] = admission._seal({k: v for k, v in record["plan"].items() if k != "plan_sha256"})
    record["receipt_sha256"] = admission._seal({k: v for k, v in record.items() if k != "receipt_sha256"})
    return record


def test_fixed_nine_direction_six_scale_cohort_and_every_byte_identity_preserved(tmp_path):
    original = failed_admission()
    before = deepcopy(original)
    report = run(tmp_path, record=original)
    assert original == before and report["original_payload_unchanged"]
    assert report["original_plan"] == before["plan"]
    assert len(report["directions"]) == 9
    assert all(tuple(p["epsilon"] for p in d["probes"]) == forensics.SCALES for d in report["directions"])
    assert all(p["original_fixed_probe"] == before["plan"]["directions"][i]["probes"][j]
               for i, d in enumerate(report["directions"]) for j, p in enumerate(d["probes"]))
    assert report["frozen_before"] == report["frozen_after"] == frozen()
    assert report["frozen_unchanged"] and report["status"] == "diagnostics_completed_original_failure_preserved"
    assert not report["can_admit_new_child"] and not report["can_certify_or_restart_original_study"]
    assert report["original_failed_admission_status"] == "failed"


def test_quadratic_endpoint_gradient_trapezoid_measures_curvature_without_acceptance_rule(tmp_path):
    report = run(tmp_path, record=failed_admission(boundary=True))
    rows = observations(report)
    assert any(abs(row["independent_primal_minus_baseline_AD_prediction"]) > 1e-9 for row in rows)
    assert all(abs(row["independent_primal_minus_endpoint_gradient_trapezoid"]) < 1e-8 for row in rows)
    assert all("agrees" not in row and "passed" not in row for row in rows)


def test_dedup_evaluates_each_exact_graph_once_after_three_baseline_repeats(tmp_path):
    calls = []
    def counted(payload):
        calls.append(forensics._graph_id(payload))
        return quadratic(payload)
    report = run(tmp_path, counted, record=failed_admission(boundary=True))
    assert len(calls) == report["unique_graph_count"] + 2
    assert calls.count(report["baseline_graph_id"]) == 3
    assert all(calls.count(identity) == 1 for identity in set(calls) if identity != report["baseline_graph_id"])
    assert len(report["points"]) <= 109
    assert any(p["left"] == report["baseline_graph_id"] for d in report["directions"] for p in d["probes"])
    assert any(not p["analysis"]["evaluated"] for d in report["directions"] for p in d["probes"])


def test_every_raw_npz_keeps_scalar_contacts_gradients_and_exact_requested_input(tmp_path):
    report = run(tmp_path)
    for point in report["points"]:
        for evaluation in point["evaluations"]:
            proof = evaluation["proof"]
            path = Path(proof["path"])
            assert path.stat().st_size == proof["bytes"]
            assert forensics.hashlib.sha256(path.read_bytes()).hexdigest() == proof["sha256"]
            with np.load(path, allow_pickle=False) as saved:
                assert {"X", "E_upper", "primal", "ad_value", "gradient_X", "gradient_E_upper", "primal_contacts", "ad_contacts"} == set(saved.files)
                for name in ("X", "E_upper"):
                    assert forensics._array_record(saved[name]) == point["input"][name]


def test_deterministic_primal_arithmetic_disagreement_remains_visible_not_accepted(tmp_path):
    report = run(tmp_path, lambda payload: quadratic(payload, scalar_offset=float(payload["X"][10, 1]) * 1e-5))
    assert report["baseline_repeats"]["primal"][0] == report["baseline_repeats"]["primal"][1]
    assert any(abs(row["independent_primal_difference"] - row["AD_returned_scalar_difference"]) > 1e-12
               for row in observations(report))
    assert report["status"] == "diagnostics_completed_original_failure_preserved"
    assert not report["can_admit_new_child"]


def test_hard_max_winner_switch_and_fixed_baseline_branch_reduction_are_explicit(tmp_path):
    baseline_value = float(graph()["X"][10, 1])
    def switching(payload):
        result = quadratic(payload)
        shift = np.float32(float(payload["X"][10, 1]) - baseline_value)
        contacts = np.tile(np.asarray([.5 + shift, .5 - shift, .1], np.float32), (25, 1))
        result.update(primal_contacts=contacts, ad_contacts=contacts.copy())
        return result
    report = run(tmp_path, switching)
    assert any(row["primal_contacts_reduction_control"]["left_changed_winner_rows"]
               or row["primal_contacts_reduction_control"]["right_changed_winner_rows"] for row in observations(report))
    assert any(abs(row["primal_contacts_reduction_control"]["best_difference"] -
                   row["primal_contacts_reduction_control"]["fixed_baseline_winner_difference"]) > 0
               for row in observations(report))


def test_optional_float64_reduction_controls_stay_separate_from_native_scalars(tmp_path):
    report = run(tmp_path, lambda payload: quadratic(payload, reduction64=True))
    assert all("primal_contacts64_reduction_control" in row for row in observations(report))
    assert all(e["contacts"]["primal_contacts64"]["array"]["dtype"] == "float64"
               for p in report["points"] for e in p["evaluations"])
    assert not report["original_optimization_settings_changed"]


@pytest.mark.parametrize("damage", ["status", "receipt", "point_bytes", "scale", "direction_count"])
def test_changed_original_status_plan_or_point_is_rejected_before_evaluator(tmp_path, damage):
    record = failed_admission()
    if damage == "status": record["status"] = "passed"
    if damage == "receipt": record["receipt_sha256"] = "0" * 64
    if damage == "point_bytes":
        record["plan"]["directions"][0]["probes"][0]["left"]["sha256"] = "0" * 64
        reseal(record)
    if damage == "scale":
        record["plan"]["directions"][0]["probes"][0]["epsilon"] = .02
        reseal(record)
    if damage == "direction_count":
        record["plan"]["directions"].pop()
        reseal(record)
    with pytest.raises(ValueError):
        run(tmp_path, evaluator=lambda _: pytest.fail("Evaluator must not run"), record=record)


@pytest.mark.parametrize("damage", ["mutated_payload", "nonfinite", "wrong_gradient_dtype", "wrong_contact_dtype"])
def test_bad_evaluator_preserves_original_and_frozen_after_failure(tmp_path, damage):
    record = failed_admission()
    before = deepcopy(record)
    def broken(payload):
        result = quadratic(payload)
        if damage == "mutated_payload": payload["X"][0, 0] = 0
        if damage == "nonfinite": result["primal"] = float("nan")
        if damage == "wrong_gradient_dtype": result["gradients"]["X"] = result["gradients"]["X"].astype(np.float64)
        if damage == "wrong_contact_dtype": result["ad_contacts"] = result["ad_contacts"].astype(np.float64)
        return result
    with pytest.raises(ValueError):
        run(tmp_path, broken, record=record)
    assert record == before
    report = forensics.json.loads((tmp_path / "new/report.json").read_text())
    assert report["status"] == "diagnostics_failed_original_failure_preserved"
    assert report["frozen_unchanged"] and report["original_payload_unchanged"]


def test_frozen_identity_change_is_a_diagnostic_failure_and_cannot_authorize_continuation(tmp_path):
    calls = []
    def changed():
        calls.append(True)
        return {"model": "old" if len(calls) == 1 else "changed", "context": "same"}
    with pytest.raises(ValueError, match="Frozen identity"):
        run(tmp_path, identity=changed)
    report = forensics.json.loads((tmp_path / "new/report.json").read_text())
    assert not report["frozen_unchanged"] and not report["can_admit_new_child"]
    assert not report["can_certify_or_restart_original_study"]


def test_completed_baseline_evaluation_survives_later_evaluator_failure(tmp_path):
    calls = []
    def interrupted(payload):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("Preserved synthetic native interruption")
        return quadratic(payload)
    with pytest.raises(RuntimeError, match="synthetic native interruption"):
        run(tmp_path, interrupted)
    report = forensics.json.loads((tmp_path / "new/report.json").read_text())
    assert len(report["points"]) == 1 and len(report["points"][0]["evaluations"]) == 1
    proof = report["points"][0]["evaluations"][0]["proof"]
    assert Path(proof["path"]).is_file()
    assert forensics.hashlib.sha256(Path(proof["path"]).read_bytes()).hexdigest() == proof["sha256"]
    assert report["frozen_unchanged"] and report["original_payload_unchanged"]


@pytest.mark.parametrize("kwargs", [{"maximum_output_bytes": 1}, {"maximum_output_bytes": 256 * 1024**2 + 1},
                                     {"maximum_unique_graphs": 1}, {"maximum_unique_graphs": 110}])
def test_limits_are_hard_and_cannot_be_enlarged_or_bypassed(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run(tmp_path, evaluator=lambda _: pytest.fail("Evaluator must not run"), **kwargs)


def test_existing_output_refused_without_overwriting_saved_files(tmp_path):
    (tmp_path / "new").mkdir()
    saved = tmp_path / "new/saved.bin"
    saved.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert saved.read_bytes() == b"preserve"


def test_numerical_entrypoint_rejects_local_execution_before_import_or_decode(monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(RuntimeError, match="allocated DTU HPC"):
        forensics.run_forensics({}, None, None, "never-created")
