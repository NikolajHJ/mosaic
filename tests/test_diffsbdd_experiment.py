"""Allocated entrypoint/source/deadline and durable failure contracts.

No real model, scheduler, environment mutation or subprocess executes in these
fixtures. Run the authored suite only in an allocated HPC CPU stage.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from latentmol.diffsbdd import experiment


def base_config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/diffsbdd_vae_alternation_v1.json").read_text())


def bound_fixture(tmp_path, monkeypatch):
    source = tmp_path / "release/source"
    source.mkdir(parents=True)
    config = source / "configs/protocol.json"
    config.parent.mkdir()
    cfg = base_config()
    cfg.update(release_root=str(source), run_root=str(tmp_path / "independent-run"))
    config.write_text(json.dumps(cfg), encoding="utf-8")
    manifest = source / "source-manifest.json"
    manifest.write_text(json.dumps(dict(version="diffsbdd-source-manifest-v1",
        files={"configs/protocol.json": experiment.sha(config)})), encoding="utf-8")
    monkeypatch.setenv("LSB_JOBID", "12345678")
    monkeypatch.setenv("DIFFSBDD_CONFIG_SHA256", experiment.sha(config))
    monkeypatch.setenv("DIFFSBDD_SOURCE_MANIFEST_SHA256", experiment.sha(manifest))
    monkeypatch.setenv("DIFFSBDD_DEADLINE_UNIX", str(time.time() + 120))
    return config, cfg


@pytest.mark.parametrize("section,key,value", (
    ("vae", "epoch", 40), ("vae", "sha256", "0" * 64),
    ("fitting", "min_atom_agreement", 0.90), ("fitting", "steps", 100),
    ("fitting", "calibrate_runtime", False),
    ("dataset", "requested_sources", 100), ("storage", "own_initial_cap_bytes", 3 * 1024**3)))
def test_predeclared_checkpoint_budget_and_thresholds_cannot_be_relaxed(section, key, value):
    cfg = base_config()
    cfg[section][key] = value
    with pytest.raises(ValueError, match="changed"):
        experiment.validate_protocol(cfg)


def test_operator_binding_deadline_and_output_namespace(tmp_path, monkeypatch):
    config, original = bound_fixture(tmp_path, monkeypatch)
    cfg, deadline = experiment.configuration(config, "cpu", Path(original["run_root"]) / "cpu")
    assert cfg["_config_sha256"] == experiment.sha(config)
    assert 0 < deadline - time.monotonic() <= 120
    with pytest.raises(ValueError, match="Release/output paths"):
        experiment.configuration(config, "cpu", tmp_path / "other-campaign/cpu")
    monkeypatch.setenv("DIFFSBDD_CONFIG_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="bindings"):
        experiment.configuration(config, "cpu", Path(original["run_root"]) / "cpu")


def test_unallocated_or_expired_stage_does_not_create_output(tmp_path, monkeypatch):
    config, cfg = bound_fixture(tmp_path, monkeypatch)
    output = Path(cfg["run_root"]) / "cpu"
    monkeypatch.delenv("LSB_JOBID")
    with pytest.raises(RuntimeError, match="allocated"):
        experiment.run(config, "cpu", output)
    monkeypatch.setenv("LSB_JOBID", "12345678")
    monkeypatch.setenv("DIFFSBDD_DEADLINE_UNIX", "0")
    with pytest.raises(TimeoutError, match="deadline"):
        experiment.run(config, "cpu", output)
    assert not output.exists()


def test_stage_pass_is_bound_to_every_output_file_and_exact_boolean_checks(tmp_path):
    cfg = dict(_config_sha256="a" * 64, _source_manifest_sha256="b" * 64)
    output = tmp_path / "cpu"
    output.mkdir()
    (output / "evidence.json").write_text('{"observed":true}\n', encoding="utf-8")
    result = experiment.finalize_output(output, "cpu", cfg, dict(checks={"observed": True}), time.monotonic())
    assert result["passed"] is True
    manifest = json.loads((output / "output-manifest.json").read_text())
    assert manifest["files"] == {"evidence.json": experiment.sha(output / "evidence.json")}
    assert result["output_manifest"]["sha256"] == experiment.sha(output / "output-manifest.json")
    other = tmp_path / "bad"
    other.mkdir()
    with pytest.raises(ValueError, match="actual boolean"):
        experiment.finalize_output(other, "cpu", cfg, dict(checks={"observed": 1}), time.monotonic())


def test_failure_is_durable_and_cannot_be_overwritten(tmp_path, monkeypatch):
    config, cfg = bound_fixture(tmp_path, monkeypatch)
    output = Path(cfg["run_root"]) / "followup"

    def failed_stage(*args):
        raise RuntimeError("preserved native gate failure")

    monkeypatch.setattr(experiment, "followup_stage", failed_stage)
    result = experiment.run(config, "followup", output)
    assert result["passed"] is False and result["error"] == "preserved native gate failure"
    assert "preserved native gate failure" in (output / "failure.txt").read_text()
    original = (output / "stage.json").read_bytes()
    with pytest.raises(FileExistsError):
        experiment.run(config, "followup", output)
    assert (output / "stage.json").read_bytes() == original


def test_native_original_child_exits_before_combined_cuda_loader(tmp_path, monkeypatch):
    output = tmp_path / "native"
    output.mkdir()
    completed = []
    cfg = dict(reference_runtime="/new/original-runtime/python", _config_path=tmp_path / "protocol.json")

    def child(command, **kwargs):
        assert not completed
        assert command[-2:] == ["--device", "cuda"]
        path = Path(command[command.index("--output") + 1])
        path.write_bytes(b"fixture original native reference")
        path.with_suffix(".json").write_text(json.dumps(dict(passed=True, sha256=experiment.sha(path))))
        completed.append("original_child_exited")
        return SimpleNamespace(returncode=0)

    def combined(*args):
        assert completed == ["original_child_exited"]
        raise RuntimeError("stop before actual CUDA/model work")

    monkeypatch.setattr(experiment.subprocess, "run", child)
    monkeypatch.setattr(experiment, "barrier", lambda *args: 0)
    monkeypatch.setattr(experiment, "load_native", combined)
    with pytest.raises(RuntimeError, match="stop before actual CUDA"):
        experiment.native_stage(cfg, output, time.monotonic() + 30)


def test_own_artifact_limit_and_deadline_are_hard_barriers(tmp_path):
    (tmp_path / "large.bin").write_bytes(b"12345")
    cfg = dict(run_root=str(tmp_path), storage=dict(own_initial_cap_bytes=4))
    with pytest.raises(RuntimeError, match="artifact"):
        experiment.barrier(cfg, time.monotonic() + 30)
    with pytest.raises(TimeoutError, match="deadline"):
        experiment.barrier(cfg, time.monotonic() - 1)
