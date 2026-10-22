"""Interim display scope must never conceal a missing or changed native input."""

# Shared test import paths; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__, include_tests=True)

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


def test_interim_scope_retains_missing_case_and_rejects_identity_or_count_changes(
    tmp_path,
):
    root = _repository_root(__file__)
    spec = importlib.util.spec_from_file_location(
        "trajectory_scope", _layout_path(root / "scripts/render_protenix_trajectory.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scope = json.loads(
        (_layout_path(root / "examples/configs/brd4_interim_report_scope.json")).read_text()
    )
    receipts = [{"job_id": job} for job in scope["included_native_job_ids"]]
    study = {"cases": [{} for _ in receipts]}
    for field in ("model_digest", "context_digest"):
        for suffix in ("_before", "_after"):
            study[field + suffix] = scope["expected_" + field]
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope))
    loaded = module._load_report_scope(path, study, receipts)
    assert loaded["pending_studies"][0]["job_id"] == "29745503"
    assert loaded["planned_case_count"] == 6 and len(loaded["manifest_sha256"]) == 64
    for field, value in (
        ("included_native_job_ids", scope["included_native_job_ids"][:-1]),
        ("pending_studies", []),
        ("planned_case_count", 5),
        ("expected_model_digest", "changed"),
        ("expected_context_digest", "changed"),
    ):
        bad = deepcopy(scope)
        bad[field] = value
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            module._load_report_scope(path, study, receipts)
    assert module._load_report_scope(None, study, receipts) is None
