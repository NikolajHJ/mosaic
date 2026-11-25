"""Allocated CPU checks for isolated V100 qualification and authenticated evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v100_qualification", ROOT / "scripts/hpc/embedding_v100_qualification.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/embedding_v100_qualification.json").read_text())


def test_configuration_validation_does_not_open_scientific_payloads(config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Schema validation must not load remote scientific inputs")
    monkeypatch.setattr(probe, "verify", forbidden)
    assert probe.validate_config(config) == config


@pytest.mark.parametrize("field,value", [
    ("gpu_profile", "v10016"), ("fixed_learning_rate", 400.),
    ("step_candidates", 16), ("line_search", True),
    ("repeats", 0), ("memory_headroom_gib", 0),
    ("internal_seconds", 86400),
])
def test_qualification_cannot_change_resource_safety_or_add_step_search(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        probe.validate_config(config)


@pytest.mark.parametrize("field,value", [
    ("release", "/work3/s234818/latentmol/releases/source-144-embedding-runtime-probe-attempt002"),
    ("source_sha256", "0" * 64),
    ("plan_sha256", "not-a-sha"),
])
def test_scientific_parent_is_the_original_sealed_campaign(config, field, value):
    config["parent"][field] = value
    with pytest.raises(ValueError):
        probe.validate_config(config)


def test_inputs_cannot_point_at_a_different_run(config):
    config["parent"]["artifacts"]["case001_state"]["path"] = (
        "/work3/s234818/latentmol/runs/source140-epoch40-to45-continuation-attempt002/latest.pt")
    with pytest.raises(ValueError):
        probe.validate_config(config)


def test_all_three_prescribed_resume_states_are_required(config):
    config["parent"]["artifacts"].pop("case003_state")
    with pytest.raises(ValueError):
        probe.validate_config(config)


@pytest.mark.parametrize("stage", ["cpu", "gpu"])
def test_output_remains_exclusively_in_the_qualification_campaign(stage):
    release = Path("/work3/s234818/latentmol/releases/source-146-embedding-v100-qualification-attempt001")
    own = Path("/work3/s234818/latentmol/runs/source146-embedding-v100-qualification-attempt001")
    probe.validate_output(release, own / stage, stage)
    live = Path("/work3/s234818/latentmol/runs/source142-epoch36-guarded-embedding-long-attempt001")
    with pytest.raises(ValueError):
        probe.validate_output(release, live / stage, stage)
    with pytest.raises(ValueError):
        probe.validate_output(release, own / "nested" / stage, stage)


def test_asset_hash_rejects_changed_input_without_overwriting_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    probe.write(path, {"status": "passed"})
    asset = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert probe.verify(asset) == path
    with pytest.raises(FileExistsError):
        probe.write(path, {"status": "failed"})
    path.write_text("corrupt")
    with pytest.raises(ValueError):
        probe.verify(asset)


@pytest.mark.parametrize("values", [[], [-1.], [float("nan")], [float("inf")]])
def test_invalid_timing_samples_cannot_be_a_successful_benchmark(values):
    with pytest.raises(ValueError):
        probe.quantiles(values)


def test_timing_report_retains_raw_samples_and_uses_nearest_rank_p90():
    samples = [4., 1., 3., 2., 5.]
    report = probe.quantiles(samples)
    assert report["n"] == 5
    assert report["mean"] == report["median"] == 3.
    assert report["p90"] == 5.
    assert report["samples"] == samples


@pytest.fixture
def cpu_proof(tmp_path):
    junit = tmp_path / "cpu-tests.xml"
    junit.write_text('<testsuites><testsuite tests="3" errors="0" failures="0" skipped="0"/></testsuites>')
    value = dict(version=probe.VERSION, stage="cpu", status="passed", job_id="29800001",
        source_sha256="1" * 64, plan_sha256="2" * 64,
        junit=dict(tests=3, failures=0, errors=0, skipped=0),
        junit_asset=dict(path=str(junit), sha256=hashlib.sha256(junit.read_bytes()).hexdigest()))
    return tmp_path / "result.json", value


def write_cpu_proof(path, value):
    probe.write(path, value)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gpu_accepts_only_the_exact_cpu_result_and_junit_bytes(cpu_proof):
    path, value = cpu_proof
    digest = write_cpu_proof(path, value)
    probe.require_passed_cpu(value, path, "1" * 64, "2" * 64, digest)
    with pytest.raises(ValueError):
        probe.require_passed_cpu(value, path, "1" * 64, "2" * 64, "0" * 64)
    changed = copy.deepcopy(value)
    changed["job_id"] = "29800002"
    with pytest.raises(ValueError):
        probe.require_passed_cpu(changed, path, "1" * 64, "2" * 64, digest)
    Path(value["junit_asset"]["path"]).write_text("changed after CPU submission")
    with pytest.raises(ValueError):
        probe.require_passed_cpu(value, path, "1" * 64, "2" * 64, digest)


@pytest.mark.parametrize("corruption", [
    "failed_status", "wrong_source", "wrong_plan", "empty_job", "wrong_stage",
    "zero_tests", "skipped", "failures", "errors",
])
def test_forged_or_incomplete_cpu_readiness_cannot_authorize_gpu(cpu_proof, corruption):
    path, value = cpu_proof
    if corruption == "failed_status":
        value["status"] = "failed"
    elif corruption == "wrong_source":
        value["source_sha256"] = "3" * 64
    elif corruption == "wrong_plan":
        value["plan_sha256"] = "3" * 64
    elif corruption == "empty_job":
        value["job_id"] = ""
    elif corruption == "wrong_stage":
        value["stage"] = "gpu"
    elif corruption == "zero_tests":
        value["junit"]["tests"] = 0
    else:
        value["junit"][corruption] = 1
    digest = write_cpu_proof(path, value)
    with pytest.raises(ValueError):
        probe.require_passed_cpu(value, path, "1" * 64, "2" * 64, digest)


@pytest.mark.parametrize("corruption", [None, "score", "gradient", "context", "policy", "path", "failed_gate"])
def test_resume_binding_requires_fresh_same_device_full_score_and_gradient(corruption):
    context, policy, gradient = "c" * 64, "d" * 64, "b" * 64
    evaluation = dict(scores={"dbmol_total_loss": 1., "affinity_pred_value": .25},
        context_sha256=context, comparison_graph_policy_sha256=policy)
    gate = dict(passed=True, scoring_path="learned", baseline_scores=dict(evaluation["scores"]),
        production_gradient_sha256=gradient, context_sha256=context,
        comparison_graph_policy_sha256=policy)
    if corruption == "score":
        evaluation["scores"]["affinity_pred_value"] = .5
    elif corruption == "gradient":
        gradient = "e" * 64
    elif corruption == "context":
        evaluation["context_sha256"] = "e" * 64
    elif corruption == "policy":
        gate["comparison_graph_policy_sha256"] = "e" * 64
    elif corruption == "path":
        gate["scoring_path"] = "relaxed"
    elif corruption == "failed_gate":
        gate["passed"] = False
    if corruption is None:
        probe.require_exact_binding(evaluation, gate, gradient, context, policy)
    else:
        with pytest.raises(ValueError):
            probe.require_exact_binding(evaluation, gate, gradient, context, policy)
