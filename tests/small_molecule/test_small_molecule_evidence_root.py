"""The evidence collector expects a run root for its pytest-alias boundary."""

# Shared test import paths; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__, include_tests=True)

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest


def collector():
    script = (
        _layout_path(_repository_root(__file__) / "scripts/collect_protenix_evidence.py")
    )
    spec = importlib.util.spec_from_file_location("collect_evidence", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "temporary_name",
    ("pytest-unit-tmp", "pytest-unit-tmp-legacy", "pytest-unit-tmp-new"),
)
def test_packaging_run_root_records_internal_pytest_alias_without_following_it(
    tmp_path, monkeypatch, temporary_name
):
    module = collector()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LSB_JOBID", "123")
    root = tmp_path / "mosaic-runs"
    run = root / "report-run"
    tests = run / "payload" / temporary_name
    target = tests / "test_case0"
    target.mkdir(parents=True)
    (target / "artifact.json").write_text('{"saved":true}')
    (tests / "test_casecurrent").symlink_to(target, target_is_directory=True)
    output = root / "report.tar.gz"

    def command(args, **kwargs):
        if "--run-dir" in args:
            receipt = Path(args[args.index("--run-dir") + 1])
            receipt.mkdir()
            (receipt / "quota.json").write_text('{"status":"ready"}')
        return SimpleNamespace(stdout="DONE", stderr="")

    monkeypatch.setattr(module.subprocess, "run", command)
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector", "--storage", "zhome", "--run", str(run), "--output", str(output)],
    )
    module.main()
    with tarfile.open(output) as archive:
        manifest = json.load(
            io.TextIOWrapper(archive.extractfile("evidence_manifest.json"))
        )
        assert (
            f"report-run/payload/{temporary_name}/test_casecurrent"
            in manifest["omitted_pytest_current_aliases"]
        )
        assert not any("test_casecurrent" in name for name in archive.getnames())
        assert (
            f"report-run/payload/{temporary_name}/test_case0/artifact.json"
            in manifest["files"]
        )


@pytest.mark.parametrize(
    "temporary_name,alias_name,target_kind",
    [
        ("pytest-unit-tmp", "test_casecurrent", "outside"),
        ("pytest-unit-tmp-legacy", "test_casecurrent", "other_basetemp"),
        ("pytest-unit-tmp-mystery", "test_casecurrent", "internal"),
        ("pytest-unit-tmp-new", "test_casealias", "internal"),
        ("elsewhere", "test_casecurrent", "internal"),
        ("pytest-unit-tmp-new", "test_casecurrent", "file"),
        ("pytest-unit-tmp-new", "test_casecurrent", "parent"),
    ],
)
def test_packaging_rejects_non_pytest_or_escaping_symlinks(
    tmp_path, monkeypatch, temporary_name, alias_name, target_kind
):
    module = collector()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LSB_JOBID", "123")
    root = tmp_path / "mosaic-runs"
    run = root / "report-run"
    tests = run / "payload" / temporary_name
    tests.mkdir(parents=True)
    target = tests / "test_case0"
    if target_kind == "outside":
        target = tmp_path / "outside"
    elif target_kind == "other_basetemp":
        target = run / "payload/pytest-unit-tmp-new/test_case0"
    elif target_kind == "parent":
        target = tests
    if target_kind == "file":
        target.write_text("saved")
    else:
        target.mkdir(parents=True, exist_ok=True)
    alias = tests / alias_name
    alias.symlink_to(target, target_is_directory=target.is_dir())
    output = root / "rejected.tar.gz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector", "--storage", "zhome", "--run", str(run), "--output", str(output)],
    )
    try:
        with pytest.raises(ValueError, match="Non-pytest symlink"):
            module.main()
    finally:
        # Keep deliberately invalid test links out of the enclosing run's
        # later evidence bundle; their rejection is retained in the XML proof.
        alias.unlink()
    assert not output.exists()
