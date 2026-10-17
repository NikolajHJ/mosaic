"""The evidence collector expects a run root for its pytest-alias boundary."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace


def test_packaging_run_root_records_internal_pytest_alias_without_following_it(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/collect_protenix_evidence.py"
    )
    spec = importlib.util.spec_from_file_location("collect_evidence", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LSB_JOBID", "123")
    root = tmp_path / "mosaic-runs"
    run = root / "report-run"
    tests = run / "payload/pytest-unit-tmp"
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
            "report-run/payload/pytest-unit-tmp/test_casecurrent"
            in manifest["omitted_pytest_current_aliases"]
        )
        assert not any("test_casecurrent" in name for name in archive.getnames())
        assert (
            "report-run/payload/pytest-unit-tmp/test_case0/artifact.json"
            in manifest["files"]
        )
