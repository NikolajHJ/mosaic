"""Compatibility paths share real module identity after grouping implementations."""
import importlib
import subprocess
import sys

import pytest


MODULES = {
    "losses": "training.losses", "diagnostics": "training.diagnostics",
    "train_and_evaluate": "training.workflow", "data": "datasets.data", "data_audit": "datasets.audit",
    "evaluate": "evaluation.evaluate", "quality": "evaluation.quality", "quality_panel": "evaluation.panel",
    "chemical_environment": "guidance.environment", "chemical_environment_policy": "guidance.environment_policy",
    "neighbor_training": "guidance.neighbor", "chemistry_regularization": "guidance.regularization",
    "connectivity_regularization": "guidance.connectivity", "optimization": "design.step_control",
    "latent_optimization": "design.latent", "latent_priors": "design.priors",
}


@pytest.mark.parametrize("old,new", MODULES.items())
def test_old_import_is_the_same_module(old, new):
    assert importlib.import_module("latentmol." + old) is importlib.import_module("latentmol." + new)


def test_monkeypatch_does_not_split_legacy_and_grouped_state(monkeypatch):
    old = importlib.import_module("latentmol.losses")
    new = importlib.import_module("latentmol.training.losses")
    marker = object()
    monkeypatch.setattr(old, "latent_kl", marker)
    assert new.latent_kl is marker


@pytest.mark.parametrize("module", ["latentmol.train_and_evaluate", "latentmol.training.workflow",
    "latentmol.data", "latentmol.datasets.data", "latentmol.evaluate", "latentmol.evaluation.evaluate"])
def test_existing_and_grouped_help_do_not_start_a_run(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_cache_provenance_still_names_the_actual_dataset_implementation():
    from pathlib import Path
    from latentmol.large_data import _preparation_code_hashes
    from latentmol.datasets import data
    from latentmol.experiments import file_sha256
    assert _preparation_code_hashes()["data.py"] == file_sha256(Path(data.__file__))


def test_recovery_protects_grouped_scientific_source_bytes(tmp_path):
    from latentmol.decoder_budget_recovery import verify_shared_source
    from latentmol.experiments import file_sha256
    name = "src/latentmol/training/losses.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("original source")
    parent = {"source": {"files": {name: file_sha256(path)}}}
    verify_shared_source(parent, tmp_path)
    path.write_text("changed source")
    with pytest.raises(ValueError, match="scientific source"):
        verify_shared_source(parent, tmp_path)
