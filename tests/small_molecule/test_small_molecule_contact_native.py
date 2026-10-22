"""Real frozen Tiny gradients and measured descent; allocated DTU GPU only."""

# Shared test import paths; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__, include_tests=True)

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.slow
@pytest.mark.small_molecule_model
def test_real_dbmol_contact_reaches_relaxed_atoms_and_bonds(tmp_path):
    if not os.environ.get("LSB_JOBID"):
        pytest.fail("Native gradient study requires an allocated DTU job")
    cache = os.environ.get("MOSAIC_STAGE0_CACHE")
    if not cache:
        pytest.fail("Set MOSAIC_STAGE0_CACHE to authenticated resident Tiny assets")
    root = _repository_root(__file__)
    output = (
        Path(os.environ.get("MOSAIC_STAGE0_RESULTS", str(tmp_path)))
        / "contact-gradients"
    )
    command = [
        sys.executable,
        str(_layout_path(root / "examples/protenix_relaxed_gradients.py")),
        "--cache-dir",
        cache,
        "--output",
        str(output),
    ]
    if os.environ.get("MOSAIC_GRADIENT_PREPARED_STUDY"):
        command.extend(
            [
                "--prepared-study",
                os.environ["MOSAIC_GRADIENT_PREPARED_STUDY"],
                "--prepared-summary-sha256",
                os.environ["MOSAIC_GRADIENT_PREPARED_SUMMARY_SHA256"],
            ]
        )
    subprocess.run(command, cwd=root, check=True)
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "passed"
    assert report["numerical_profile"]["jax_default_matmul_precision"] == "highest"
    assert len(report["cases"]) == 2
    assert report["model_digest_before"] == report["model_digest_after"]
    assert report["context_digest_before"] == report["context_digest_after"]
    for case in report["cases"]:
        assert case["final_loss"] < case["initial_loss"]
        assert len(case["updates"]) == 3
        assert {d["field"] for d in case["directions"]} == {"X", "E_upper"}
        assert all(d["autodiff"] > 0 for d in case["directions"])
