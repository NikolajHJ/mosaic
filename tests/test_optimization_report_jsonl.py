"""Saved-only journal input: long runs need no per-update metadata files."""
import json

import pytest

from latentmol.reporting.optimization import build_report, render_report


def _run(tmp_path):
    root = tmp_path / "saved"
    arm = root / "latent"
    arm.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps(dict(status="completed", config={})))
    rows = [dict(step=i, score=dict(loss=2-i, binding_loss=1-i,
            repair_guidance=dict(normalized_nll=.1, applied_weight=1))) for i in (0, 1)]
    (arm / "metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    return root, arm


def test_journal_only_run_renders_saved_losses(tmp_path):
    root, _ = _run(tmp_path)
    data = build_report([root])
    frames = data["runs"][0]["trajectories"][0]["frames"]
    assert [f["metrics"]["loss"] for f in frames] == [2, 1]
    assert render_report([root], tmp_path/"report").is_file()


def test_journal_and_legacy_duplicates_are_rejected(tmp_path):
    root, arm = _run(tmp_path)
    (arm/"metrics-0000.json").write_text(json.dumps(dict(step=0, loss=1)))
    with pytest.raises(ValueError, match="unique"):
        build_report([root])
