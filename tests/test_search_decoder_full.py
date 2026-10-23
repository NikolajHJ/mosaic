"""Allocated HPC checks for the expensive full-comparison boundaries.

These cover scientific matching, native-gate failure and duplicate safety;
they do not establish better random generation. Numerical execution is HPC only.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from latentmol.experiments import file_sha256
from latentmol.search_decoder_protocol import read_full_protocol, segment_protocol
from latentmol import search_decoder_full as full


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    path = ROOT / "configs/search_decoder_full.json"
    return read_full_protocol(path, file_sha256(path))


@pytest.mark.parametrize("change", ["subset", "epochs", "arms", "reserve", "batch", "identity"])
def test_full_protocol_rejects_incompatible_comparison(tmp_path, change):
    p = protocol()
    if change == "subset": p["train_limit"] = 16384
    elif change == "epochs": p["fine_tune"]["epochs"] = 3
    elif change == "arms": p["arms"][0]["repair_weight"] = 1
    elif change == "reserve": p["controller_budget_seconds"] = 3600
    elif change == "batch": p["training_prior_batch"] = 257
    elif change == "identity": p["dataset_sha256"] = "unknown"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError): read_full_protocol(path, file_sha256(path))


def test_full_protocol_and_original_pilot_remain_distinct():
    from latentmol.search_decoder_study import read_protocol as read_pilot
    p = protocol()
    assert segment_protocol(p, 0)["seed"] == 17 and segment_protocol(p, 1)["seed"] == 23
    path = ROOT / "configs/search_decoder_pilot.json"
    pilot = read_pilot(path, file_sha256(path))
    assert len(pilot["arms"]) == 4 and pilot["train_limit"] == 16384
    full_path = ROOT / "configs/search_decoder_full.json"
    with pytest.raises(ValueError): read_pilot(full_path, file_sha256(full_path))


def test_full_controller_requires_native_gate_and_preserves_failure(tmp_path, monkeypatch):
    observed = {}
    def failed_engine(p, root, source, **kwargs):
        observed.update(kwargs)
        assert p["seed"] == 23 and len(p["arms"]) == 2
        root.mkdir()
        raise RuntimeError("native fixed-target gradient inconclusive")
    monkeypatch.setattr(full, "run_study", failed_engine)
    with pytest.raises(RuntimeError, match="inconclusive"):
        full.run_full_segment(protocol(), tmp_path / "study", tmp_path, 1)
    assert observed["require_full_data"] and observed["require_native_gradient"]
    assert observed["evaluation_reserve_seconds"] == 3600
    failure = json.loads((tmp_path / "study/segment-1/controller_failure.json").read_text())
    assert failure["complete"] is False
    summary = json.loads((tmp_path / "study/summary.json").read_text())
    assert summary["complete"] is False


def test_duplicate_segment_never_changes_existing_outputs(tmp_path):
    root = tmp_path / "study"
    (root / "segment-1").mkdir(parents=True)
    (root / "summary.json").write_text('"original summary"')
    with pytest.raises(FileExistsError): full.run_full_segment(protocol(), root, tmp_path, 1)
    assert (root / "summary.json").read_text() == '"original summary"'
    assert not (root / "segment-1/controller_failure.json").exists()


def test_aggregate_keeps_seed_results_separate_and_missing_is_incomplete(tmp_path):
    saved_segment(tmp_path, 0)
    assert full.aggregate_segments(tmp_path)["complete"] is False
    saved_segment(tmp_path, 1)
    result = full.aggregate_segments(tmp_path)
    assert result["complete"] is True and len(result["segments"]) == 2
    assert "do not pool" in result["scope"]
    (tmp_path / "segment-1/summary.json").write_text(json.dumps({"complete": True, "cases": []}))
    assert full.aggregate_segments(tmp_path)["complete"] is False


def saved_segment(root, index):
    """Metadata fixtures are opaque bytes, not pretend trained model weights."""
    p = protocol()
    directory = root / f"segment-{index}"
    directory.mkdir()
    (directory / "protocol.json").write_text(json.dumps(segment_protocol(p, index)))
    probe = {"gradient_verified": True, "direction_status": {
        "gradient_unit": "verified_local_plateau", "seeded_random_unit": "verified_local_plateau"}}
    (directory / "native_engineering_gate.json").write_text(json.dumps({"status": "passed",
        "native_search_coverage": {"accepted_targets": 1}, "native_gradient_probe": probe}))
    cases = []
    for arm in p["arms"]:
        run = directory / "continuations" / arm["name"]
        run.mkdir(parents=True)
        checkpoint = run / "epoch-010.pt"
        checkpoint.write_bytes(b"opaque metadata fixture")
        guidance = {"name": arm["name"], "repair_weight": arm["repair_weight"], "panel_sha256": "a" * 64}
        config = {"seed": p["seeds"][index], "train_count": 215000, "initializer_sha256": p["initializer_sha256"],
                  "dataset_sha256": p["dataset_sha256"], "split_sha256": p["split_sha256"],
                  "config": p["fine_tune"], "extra_guidance": guidance}
        (run / "run.json").write_text(json.dumps({"status": "completed", "completed_epochs": 10,
            "config": config, "source_archive_sha256": "f" * 64}))
        cases.append({"arm": arm, "status": "completed", "fit": {"run_dir": str(run), "checkpoint": str(checkpoint),
            "sha256": file_sha256(checkpoint), "encoder_sha256": "e" * 64, "extra_guidance": guidance,
            "terminal_rng": {"loader": "shared within seed"}}, "prior": {"panel_sha256": "b" * 64}})
    (directory / "summary.json").write_text(json.dumps({"complete": True, "cases": cases}))
    return directory


@pytest.mark.parametrize("change", ["seed", "checkpoint", "protocol", "gate", "encoder"])
def test_aggregate_rejects_mixed_or_tampered_completed_segments(tmp_path, change):
    saved_segment(tmp_path, 0)
    directory = saved_segment(tmp_path, 1)
    if change in ("seed", "protocol"):
        path = directory / "protocol.json"
        d = json.loads(path.read_text())
        if change == "seed": d["seed"] = 17
        else: d["repair"]["beam_width"] = 8
        path.write_text(json.dumps(d))
    elif change == "checkpoint": (directory / "continuations/search/epoch-010.pt").write_bytes(b"changed")
    elif change == "gate":
        path = directory / "native_engineering_gate.json"
        d = json.loads(path.read_text());d["native_gradient_probe"]["gradient_verified"] = False
        path.write_text(json.dumps(d))
    else:
        path = directory / "summary.json"
        d = json.loads(path.read_text());d["cases"][1]["fit"]["encoder_sha256"] = "d" * 64
        path.write_text(json.dumps(d))
    assert full.aggregate_segments(tmp_path, protocol())["complete"] is False


def test_native_full_gate_rejects_zero_coverage_and_partial_direction_agreement():
    from latentmol.search_decoder_study import informative_native_gradient
    probe = {"gradient_verified": True, "direction_status": {
        "gradient_unit": "verified_local_plateau", "seeded_random_unit": "verified_local_plateau"}}
    assert informative_native_gradient({"accepted_targets": 1}, probe)
    assert not informative_native_gradient({"accepted_targets": 0}, probe)
    probe["direction_status"]["seeded_random_unit"] = "inconclusive_no_measurable_agreement_plateau"
    assert not informative_native_gradient({"accepted_targets": 1}, probe)


def test_scheduler_plan_is_serial_and_second_retains_cpu_dependency():
    path = ROOT / "scripts/hpc/submit_search_decoder_full.py"
    spec = importlib.util.spec_from_file_location("full_submit", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(path.parent))
    try: spec.loader.exec_module(module)
    finally: sys.path.pop(0)
    release, work = Path("/work3/u/releases/source075"), Path("/work3/u")
    rows = module.workflow(release, work, protocol(), release / "config.json", "a"*64, "b"*64, "c"*64, "123")
    assert len(rows) == 2
    first = rows[0]["arguments"]
    assert first[first.index("-w")+1] == "done(123) && ended(29625193)"
    second = module.following_segment(rows[1], "123", "456")
    args = second["arguments"]
    assert args[args.index("-w")+1] == "done(123) && ended(456)"
    assert args.count("-w") == 1 and second["dependency"] is None
    assert args[args.index("-W")+1] == "23:00"
    with pytest.raises(ValueError): module.following_segment(rows[1], "unknown", "456")
