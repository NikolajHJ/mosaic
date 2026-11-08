import json

import pytest

from latentmol.design.protenix_restart_path_metrics import PathTrace, read_path, summarize_paths


def test_path_reader_keeps_invalid_steps_and_skips_sampled_candidates(tmp_path):
    path = tmp_path / "path.jsonl"
    rows = [
        dict(step=0, smiles="CCO", raw_valid=True),
        dict(step=0, smiles="CCC", kind="sample", valid=True),
        dict(step=1, smiles="CC.O", kind="argmax", valid=True),
        dict(step=2, smiles=None, raw_valid=False),
        dict(step=4, smiles="CCN", kind="argmax", valid=True),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert read_path(path) == {0: "CCO", 1: None, 2: None, 4: "CCN"}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(step=4, smiles=None, raw_valid=False)) + "\n")
    with pytest.raises(ValueError, match="duplicate trajectory step"):
        read_path(path)


def test_path_metrics_separate_retention_replay_and_valid_endpoint_agreement():
    source = {12: "CCO", 13: "CCO", 14: None, 15: "CCN", 16: "CCCl", 17: "CCC"}
    a = PathTrace("a", "kick", "A", "CCO",
                  {0: "CCO", 1: "CCO", 2: None, 3: "CCN", 4: "CCC"}, source, 12)
    b = PathTrace("b", "kick", "A", "CCO", {0: "CCO", 1: "CCO", 2: None, 3: "CCN", 4: None})
    c = PathTrace("c", "kick", "B", "CCC", {0: "CCC", 1: "CNC", 2: None, 3: "CNC", 4: "CCC"})
    ref = PathTrace("control", "untouched", "A", "CCO", a.observations)
    summary = summarize_paths([a, b, c, ref], late_steps=2)
    arm = summary["arms"]["kick"]
    first = summary["branches"][0]
    assert first["parent_revisit"]["fraction"] == pytest.approx(1 / 3)
    assert first["source_revisit"]["fraction"] == 1
    assert first["source_nonparent_revisit"]["valid"] == 2
    assert first["source_aligned_continuation"] == dict(
        observed=4, jointly_valid=3, identical=2, fraction=2 / 3,
        first_different_step=4, first_rejoin_step=None)
    siblings = arm["same_parent"]
    assert siblings["pairs"] == 1
    assert siblings["trajectory_jaccard"]["mean"] == pytest.approx(2 / 3)
    assert siblings["aligned"]["observed"] == 4
    assert siblings["aligned"]["jointly_valid"] == 2
    assert siblings["aligned"]["identical"] == 2
    assert siblings["late_aligned"]["jointly_valid"] == 1
    assert siblings["late_nearest_tanimoto"]["count"] == 1
    assert siblings["endpoint"]["fraction"] is None
    assert arm["different_parent"]["pairs"] == 2
    assert arm["different_parent"]["endpoint"]["fraction"] == 1
    assert arm["different_parent"]["endpoint"]["tanimoto"]["mean"] == 1
    assert arm["reference"]["pairs"] == 2
    assert not any({row["left"], row["right"]} == {"c", "control"} for row in summary["pairs"])


def test_censored_endpoint_is_not_replaced_with_last_valid_and_empty_paths_are_unknown():
    a = PathTrace("a", "kick", "A", "CCO", {0: "CCO", 1: "CCN"}, expected_updates=3)
    b = PathTrace("b", "kick", "A", "CCO", {0: "CCO", 1: None, 2: None, 3: None})
    summary = summarize_paths([a, b])
    arm = summary["arms"]["kick"]
    first = summary["branches"][0]
    assert first["last_valid_smiles"] == "CCN"
    assert first["endpoint_smiles"] is None
    assert not first["endpoint_observed"]
    assert arm["endpoint_validity"] == dict(valid=0, observed=1, fraction=0)
    assert arm["same_parent"]["trajectory_jaccard"] == dict(mean=None, count=0)
    assert arm["same_parent"]["aligned"]["fraction"] is None


def test_aligned_divergence_and_rejoin_skip_jointly_invalid_steps():
    left = PathTrace("left", "kick", "A", "CCO", {0: "CCO", 1: "CCN", 2: None, 3: "CCC"},
                     hard_observations={1: b"different1", 2: b"same-invalid", 3: b"same"})
    right = PathTrace("right", "kick", "A", "CCO", {0: "CCO", 1: "CCCl", 2: None, 3: "CCC"},
                      hard_observations={1: b"different2", 2: b"same-invalid", 3: b"same"})
    result = summarize_paths([left, right])
    pair = result["pairs"][0]
    assert pair["early_aligned"]["first_different_step"] == 1
    assert pair["early_aligned"]["first_rejoin_step"] == 3
    assert pair["early_aligned"]["jointly_valid"] == 2
    assert pair["aligned_hard_labels"]["identical"] == 2
    assert result["arms"]["kick"]["same_parent"]["aligned"]["rejoin_fraction"] == 1
