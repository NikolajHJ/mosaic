"""Convergence decisions and real tiny staged-training checks, run on HPC only."""

import json
from pathlib import Path

import pytest

from latentmol.data import prepare_dataset
from latentmol.diagnostics import continuation_decision
from latentmol.evaluate import binomial_interval
from latentmol.models import ModelConfig
from latentmol.representation import GraphSpec
from latentmol.train import TrainConfig


def _rows(loss_before=1., loss_after=1., valid_after=.1, exact_after=.02):
    result = []
    for epoch in range(1, 21):
        later = epoch > 10
        result.append({"epoch": epoch, "val": {"reconstruction": loss_after if later else loss_before},
                       "train_probe": {"reconstruction": .4 if later else .5},
                       "val_probe": {"reconstruction": loss_after if later else loss_before,
                                     "validity": valid_after if later else .1,
                                     "exact_reconstruction": exact_after if later else .02}})
    return result


@pytest.mark.parametrize("options,extends", [
    ({}, False), ({"loss_after": .98}, True), ({"valid_after": .12}, True),
    ({"exact_after": .03}, True), ({"loss_after": 1.1}, False),
])
def test_continuation_uses_validation_progress_not_falling_training_loss(options, extends):
    decision = continuation_decision(_rows(**options))
    assert decision["extend"] is extends
    if not extends:
        assert decision["overfitting_warning"]
    assert decision["before_epochs"] == [1, 10]
    assert decision["after_epochs"] == [11, 20]


def test_convergence_rejects_gaps_and_nonfinite_measurements():
    history = _rows()
    history[-1]["epoch"] = 21
    with pytest.raises(ValueError, match="contiguous"):
        continuation_decision(history)
    history = _rows()
    history[-1]["val_probe"]["validity"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        continuation_decision(history)


def test_wilson_interval_retains_uncertainty_at_zero_or_all_valid():
    zero, all_valid = binomial_interval(0, 100), binomial_interval(100, 100)
    assert zero[0] < 1e-12 and .03 < zero[1] < .04
    assert .96 < all_valid[0] < .97 and all_valid[1] == pytest.approx(1.)
    assert binomial_interval(0, 0) is None


def test_real_tiny_workflow_seals_selection_before_testing_and_bounds_generation(tmp_path, monkeypatch):
    from latentmol import full_training as workflow
    source = tmp_path / "molecules.smi"
    source.write_text("\n".join([
        "C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC",
        "CO", "CCO", "CCCO", "CCCCO", "CN", "CCN", "CNC", "CCNC", "CF", "CCF", "CCl", "CCCl"]))
    cache, runs = tmp_path / "data", tmp_path / "runs"
    prepare_dataset(source, cache, spec=GraphSpec(max_nodes=8), train_count=12, val_count=4)
    config = TrainConfig(mode="vae", epochs=4, batch_size=4, validation_limit=4, chemistry_limit=4,
                         train_probe_limit=4, full_chemistry_every=2, checkpoint_every=2,
                         checkpoint_epochs=(2,), lr_milestones=((3, .0003),), device="cpu",
                         model=ModelConfig(hidden=8, latent=4, layers=1))
    original = workflow.evaluate_checkpoint
    test_calls = []
    def checked_evaluation(*args, **kwargs):
        if kwargs["split"] == "test":
            active = list(runs.glob("*-full-workflow_*"))
            assert len(active) == 1
            selection = json.loads((active[0] / "selection_before_test.json").read_text())
            assert selection["test_was_used_for_selection"] is False
            assert str(args[0]) in [row["path"] for row in selection["checkpoints"]]
            test_calls.append(str(args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(workflow, "evaluate_checkpoint", checked_evaluation)
    result = workflow.run_full_training(config, cache, runs, initial_epochs=2, extension_epochs=2,
                                         window=1, final_limit=100, neighborhood_limit=2, prior_limit=7)
    summary = json.loads((result / "summary.json").read_text())
    assert summary["completed_epochs"] == 4
    assert summary["stages"][0]["decision"] == "mandatory_lower_learning_rate_check"
    assert len(test_calls) == len(summary["checkpoints"])
    assert (result / "learning_curves.png").stat().st_size > 1000
    assert (result / "learning_curves.pdf").stat().st_size > 1000
    for row in summary["evaluations"]:
        evaluated = json.loads((Path(row["run"]) / "summary.json").read_text())
        assert evaluated["reconstruction"]["molecules"] == 4
        assert evaluated["reconstruction"]["chemistry_samples"] == 4
        assert evaluated["neighborhood"][0]["samples"] == 2
        assert evaluated["prior"]["samples"] == 7
