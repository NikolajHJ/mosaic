"""Regression for the candidate handoff and equivalence to the running control's recipe."""

from argparse import Namespace
from copy import deepcopy
import json

import pytest

from experiments.vae import encoder_upgrade_pilot as pilot


def settings():
    candidate = json.loads(pilot.PILOT_RECIPE.read_bytes())
    source = json.loads(pilot.PILOT_RECIPE.with_name(candidate["source_recipe"]).read_bytes())
    _, arms = pilot.policies(source)
    state = dict(config_sha256="roundtrip", continuation={"updates": 611000}, experiment_state={
        "roundtrip_handoff": {"source_protocol": "rings", "target_protocol": "roundtrip"},
        "motif_activation": {"protocol": "rings", "winner": "control", "adopted_at_update": 598000,
                             "policy_sha256": pilot._digest(arms["control"])}})
    return source, candidate, state


def test_selected_loss_and_evaluation_match_production():
    source, recipe, state = settings()
    old = deepcopy(source)
    config = pilot.effective_config(source, state, "roundtrip", recipe)
    assert config["functional_groups"] == source["functional_groups"]["activation_gate"]["control"]
    assert "ring_size" not in config["functional_groups"]
    assert config["quality_panel"] == dict(source["quality_panel"], generated_roundtrip_points=128,
                                           monitor_motifs=True, monitor_ring_sizes=True)
    extra = config["generated_roundtrip"].pop("excess_confidence")
    assert extra == {"weight": 0.1, "tolerance": 1.0}
    assert config["generated_roundtrip"] == source["generated_roundtrip"]
    assert source == old
    state["experiment_state"]["motif_activation"]["policy_sha256"] = "wrong"
    with pytest.raises(ValueError, match="selected motif policy"):
        pilot.effective_config(source, state, "roundtrip", recipe)


def test_dispatches_only_candidate_from_terminal_receipt(tmp_path, monkeypatch):
    source, recipe, state = settings()
    common = tmp_path / "production"
    current = common / "segment-03"
    control = common / "segment-04"
    current.mkdir(parents=True)
    receipt = {"checkpoint": str(current / "checkpoint.pt"), "updates": 611000, "sha256": "test"}
    (current / "summary.json").write_text(json.dumps(dict(status="paused", stop_reason="allocation_budget_reached",
                                                    segment_updates=5000, latest=receipt)), encoding="utf8")
    parent = tmp_path / "parent.json"
    parent.write_text('{"checkpoint":"base.pt","sha256":"base"}', encoding="utf8")
    args = Namespace(config=pilot.PILOT_RECIPE.with_name(recipe["source_recipe"]),
                     parent_receipt=parent, resume_receipt=current / "latest.json",
                     output=tmp_path / "candidate", production_control=control, updates=128, wall_seconds=999)
    class FakeCorpus:
        def split(self, name):
            return range(60_000_000) if name == "train" else range(256)
    monkeypatch.setenv("LSB_JOBID", "123")
    monkeypatch.setenv("DBMOL_LAUNCH_READY", str(tmp_path / "ready.json"))
    monkeypatch.setenv("DBMOL_LAUNCH_NONCE", "abc")
    monkeypatch.setattr(pilot.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pilot, "_check_cuda_requirement", lambda _: None)
    monkeypatch.setattr(pilot.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(pilot, "BalancedSparseCorpus", lambda *a, **k: FakeCorpus())
    monkeypatch.setattr(pilot, "verify_canonical_training_cache", lambda *_: None)
    monkeypatch.setattr(pilot, "_load", lambda *args: (None, dict(config_sha256=source["parent_config_sha256"],
                    dataset_sha256=source["data_sha256"], continuation={"updates": source["parent_updates"]})))
    monkeypatch.setattr(pilot, "_arm_identity", lambda *_: "roundtrip")
    monkeypatch.setattr(pilot, "_read_resume", lambda *_: (receipt, state))
    calls = []
    monkeypatch.setattr(pilot, "_train", lambda *a, **k: calls.append((a,k)))
    pilot.run(args, recipe)
    assert len(calls) == 1 and not control.exists()
    positional, keywords = calls[0]
    assert positional[4] == args.output and positional[6:9] == (240, state, 128)
    assert positional[10] == 48 and keywords["update_fn"] is pilot.projected_update
    assert keywords["experiment_state"]["motif_activation"] == state["experiment_state"]["motif_activation"]
    fork = json.loads((args.output / "fork.json").read_bytes())
    assert fork["resume"] == receipt and fork["production_control"] == str(control.resolve())
    assert json.loads((tmp_path / "ready.json").read_bytes())["nonce"] == "abc"
    with pytest.raises(FileExistsError):
        pilot.run(args, recipe)
