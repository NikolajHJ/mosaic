"""Both early branches can restore the pinned source without a terminal summary."""

from argparse import Namespace
import json

import pytest

from experiments.vae import early_encoder_arms as arms, encoder_upgrade_pilot as pilot
from experiments.vae.test_encoder_upgrade_pilot import settings


def test_early_fork_selects_single_scientific_change_and_identical_source(tmp_path, monkeypatch):
    source, recipe, state = settings()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    receipt = {"updates": 611000, "sha256": "a" * 64,
               "checkpoint": str(source_dir / "checkpoint-000611000.pt")}
    parent = tmp_path / "parent.json"
    parent.write_text('{"checkpoint":"base.pt","sha256":"base"}', encoding="utf8")
    args = Namespace(config=pilot.PILOT_RECIPE.with_name(recipe["source_recipe"]),
                     parent_receipt=parent, resume_receipt=source_dir / "latest.json",
                     production_control=tmp_path / "control", output=tmp_path / "candidate",
                     wall_seconds=arms.WALL_SECONDS["candidate"], updates=None)

    class Corpus:
        def split(self, name):
            return range(60_000_000) if name == "train" else range(256)
    monkeypatch.setenv("LSB_JOBID", "1")
    monkeypatch.setattr(pilot.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pilot, "_check_cuda_requirement", lambda _: None)
    monkeypatch.setattr(pilot.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(pilot, "BalancedSparseCorpus", lambda *a, **k: Corpus())
    monkeypatch.setattr(pilot, "verify_canonical_training_cache", lambda *_: None)
    monkeypatch.setattr(pilot, "_load", lambda *_: (None, dict(config_sha256=source["parent_config_sha256"],
                        dataset_sha256=source["data_sha256"], continuation={"updates": source["parent_updates"]})))
    monkeypatch.setattr(pilot, "_arm_identity", lambda *_: "roundtrip")
    monkeypatch.setattr(pilot, "_read_resume", lambda *_: (receipt, state))
    calls = []
    monkeypatch.setattr(pilot, "_train", lambda *a, **k: calls.append((a,k)))

    with pytest.raises(ValueError, match="pinned early fork"):
        pilot.run(args, recipe, early_arm="candidate", source_sha256="c"*64)
    with pytest.raises(FileNotFoundError):
        pilot.run(args, recipe)
    assert not calls and not args.output.exists()
    for arm in ("candidate", "control"):
        args.wall_seconds = arms.WALL_SECONDS[arm]
        args.output = tmp_path / arm
        pilot.run(args, recipe, early_arm=arm, source_sha256=receipt["sha256"])

    (candidate, ck), (control, ok) = calls
    assert candidate[7:9] == control[7:9] == (state, recipe["updates"])
    assert "encoder_upgrade" in ck["experiment_state"] and ok["experiment_state"] == state["experiment_state"]
    assert ok["student_factory"].func is pilot._student and ck["student_factory"].func is pilot.candidate_student
    assert control[0]["generated_roundtrip"] == source["generated_roundtrip"]
    assert candidate[0]["generated_roundtrip"] == dict(control[0]["generated_roundtrip"],
        excess_confidence=recipe["excess_confidence"])
    for key in control[0].keys() - {"generated_roundtrip"}:
        assert candidate[0][key] == control[0][key]
    forks = [json.loads((tmp_path / arm / "fork.json").read_bytes()) for arm in ("candidate", "control")]
    assert forks[0]["resume"] == forks[1]["resume"] == receipt
    assert forks[0]["source_config_sha256"] == forks[1]["config_sha256"] == "roundtrip"
    assert [f["wall_seconds"] for f in forks] == [81000, 28200]
