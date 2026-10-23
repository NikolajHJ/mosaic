"""Fixed-panel/paired-denominator checks; numerical execution belongs on HPC."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from latentmol.decoder_quality_evaluation import evaluate_random_panel, paired_compare
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec


class _DecoderFixture(nn.Module):
    """Finite logits for a valid ethane or an invalid oxygen triple bond."""
    def __init__(self, always_valid=False):
        super().__init__()
        self.spec = GraphSpec(max_nodes=2)
        self.anchor = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.register_buffer("variant", torch.tensor(int(always_valid)))

    def decode(self, z):
        batch = len(z)
        atoms = torch.full((batch, 2, 10), -90., dtype=z.dtype, device=z.device)
        charges = torch.full((batch, 2, 5), -90., dtype=z.dtype, device=z.device)
        bonds = torch.full((batch, 2, 2, 5), -90., dtype=z.dtype, device=z.device)
        valid = (z[:, 0] >= 0) | bool(self.variant)
        atoms[:, 1, self.spec.atomic_numbers.index(6)] = 0
        for row in range(batch):
            element = 6 if bool(valid[row]) else 8
            atoms[row, 0, self.spec.atomic_numbers.index(element)] = 0
            label = 1 if bool(valid[row]) else 3
            bonds[row, 0, 1, label] = bonds[row, 1, 0, label] = 0
        charges[..., self.spec.formal_charges.index(0)] = 0
        bonds[:, 0, 0, 0] = bonds[:, 1, 1, 0] = 0
        return DecoderOutput(atoms + self.anchor * 0, charges + self.anchor * 0, bonds + self.anchor * 0)


def _panel():
    return torch.tensor([[-1.], [-2.], [-3.], [1.]], dtype=torch.float64)


def test_fixed_draw_quota_invalid_duplicates_training_novelty_and_restored_model(tmp_path):
    model = _DecoderFixture()
    model.train()
    before = model.anchor.detach().clone()
    folder = tmp_path / "mixed"
    result = evaluate_random_panel(model, _panel(), {"CC"}, folder, repair_limit=0)
    assert model.training and torch.equal(model.anchor, before)
    assert result["argmax"]["attempted"] == 4 and result["argmax"]["valid_count"] == 1
    assert result["sample"]["attempted"] == 16 and result["sample"]["valid_count"] == 4
    assert result["all_attempts"]["attempted"] == 20
    assert result["sample"]["unique_valid_count"] == 1
    assert result["sample"]["duplicate_valid_count"] == 3
    assert result["sample"]["novel_valid_vs_training_count"] == 0
    assert result["argmax"]["histograms"]["failure_reason"] == {"sanitization": 3}
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    assert [row["latent_id"] for row in rows] == [latent for latent in range(4) for _ in range(5)]
    assert all(row["inspection"]["detail"] for row in rows if not row["valid"])
    assert result["repair"]["attempted"] == 0 and not result["advantage_established"]
    assert "expected_local_bond_order_excess" in result["relaxed_diagnostic_means"]
    assert "expected_element_counts" in result["per_latent"][0]["relaxed_diagnostics"]
    with np.load(folder / "labels.npz") as labels:
        assert labels["argmax_atoms"].shape == (4, 2)
        assert labels["sample_atoms"].shape == (4, 4, 2)
        assert labels["sample_bonds"].shape == (4, 4, 2, 2)
    with pytest.raises(FileExistsError):
        evaluate_random_panel(model, _panel(), set(), folder, repair_limit=0)


def test_latent_order_rng_and_panel_protocol_are_decode_batch_invariant(tmp_path):
    model = _DecoderFixture()
    first_dir, second_dir = tmp_path / "one", tmp_path / "three"
    first = evaluate_random_panel(model, _panel(), set(), first_dir, batch_size=1, repair_limit=0)
    second = evaluate_random_panel(model, _panel(), set(), second_dir, batch_size=3, repair_limit=0)
    assert first["panel_sha256"] == second["panel_sha256"]
    assert (first_dir / "rows.jsonl").read_bytes() == (second_dir / "rows.jsonl").read_bytes()
    assert first["per_latent"] == second["per_latent"]
    compared = paired_compare(first_dir, second_dir, bootstrap_draws=40)
    assert compared["comparisons"]["sample"]["mean_difference_candidate_minus_reference"] == 0
    assert compared["comparisons"]["argmax"]["ci95"] == [0, 0]


def test_repair_budget_is_predeclared_argmax_prefix_and_probabilities_are_conditional(tmp_path, monkeypatch):
    calls = []
    def repair(raw, probabilities, spec, config):
        calls.append((tuple(value.clone() for value in raw), tuple(value.clone() for value in probabilities)))
        assert tuple(probabilities[2].shape) == (2, 2, 5)
        torch.testing.assert_close(probabilities[2].sum(-1), torch.ones(2, 2, dtype=torch.float64))
        return {"status": "censored", "checks": 1, "checks_by_depth": [{"depth": 1, "checks": 1, "capped": True}],
                "best_observed_weighted_cost": None, "distance_is_exact": False}
    monkeypatch.setattr("latentmol.decoder_quality_evaluation.search_probability_repair", repair)
    output = evaluate_random_panel(_DecoderFixture(), _panel(), set(), tmp_path / "repair", repair_limit=2)
    assert len(calls) == 2
    assert output["repair"]["latent_ids"] == [0, 1]
    assert output["repair"]["status_counts"] == {"censored": 2}
    assert output["repair"]["capped_depths"] == 2
    receipts = [json.loads(line) for line in (tmp_path / "repair/repairs.jsonl").read_text().splitlines()]
    assert [receipt["latent_id"] for receipt in receipts] == [0, 1]
    assert all(receipt["candidate_kind"] == "argmax" for receipt in receipts)


def test_actual_bounded_probability_repair_retains_search_witness_scope(tmp_path):
    cfg = ProbabilityRepairConfig(max_edits=1, checks_per_depth=16, beam_width=2)
    result = evaluate_random_panel(_DecoderFixture(), _panel()[:1], set(), tmp_path / "native-repair",
                                   repair_limit=1, repair_config=cfg)
    receipt = json.loads((tmp_path / "native-repair/repairs.jsonl").read_text().strip())
    assert receipt["status"] in ("witness_found", "censored")
    assert result["repair"]["attempted"] == 1
    assert receipt["checks"] <= 16
    assert not result["repair"]["distance_is_globally_proven"]
    assert result["argmax"]["valid_count"] == 0  # A witness never replaces raw output.


def test_paired_bootstrap_clusters_draws_and_never_bypasses_reconstruction_gate(tmp_path):
    reference_dir, candidate_dir = tmp_path / "reference", tmp_path / "candidate"
    evaluate_random_panel(_DecoderFixture(), _panel(), set(), reference_dir, repair_limit=0)
    evaluate_random_panel(_DecoderFixture(always_valid=True), _panel(), set(), candidate_dir, repair_limit=0)
    compared = paired_compare(reference_dir, candidate_dir, bootstrap_seed=47, bootstrap_draws=1000)
    sample = compared["comparisons"]["sample"]
    assert sample["latent_clusters"] == 4 and sample["reference_attempted"] == sample["candidate_attempted"] == 16
    assert sample["mean_difference_candidate_minus_reference"] == .75
    assert compared["comparisons"]["argmax"]["mean_difference_candidate_minus_reference"] == .75
    assert not compared["decision"]["advantage_established"]
    assert "controller gate" in compared["decision"]["reconstruction_retention"]
    assert (candidate_dir / f"paired_comparison-{compared['reference_model_state_sha256'][:12]}.json").is_file()


def test_protocol_mismatch_and_mutated_artifacts_fail_closed(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    evaluate_random_panel(_DecoderFixture(), _panel(), set(), first_dir, repair_limit=0)
    evaluate_random_panel(_DecoderFixture(), _panel(), set(), second_dir, repair_limit=0, sample_seed=99)
    with pytest.raises(ValueError, match="protocol identities"):
        paired_compare(first_dir, second_dir, bootstrap_draws=10)
    third_dir = tmp_path / "third"
    evaluate_random_panel(_DecoderFixture(), _panel(), set(), third_dir, repair_limit=0)
    with (third_dir / "rows.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        paired_compare(first_dir, third_dir, bootstrap_draws=10)


def test_distinct_baselines_keep_both_comparison_receipts(tmp_path):
    first, second, candidate = tmp_path / "original", tmp_path / "control", tmp_path / "candidate"
    evaluate_random_panel(_DecoderFixture(), _panel(), set(), first, repair_limit=0)
    evaluate_random_panel(_DecoderFixture(always_valid=True), _panel(), set(), second, repair_limit=0)
    evaluate_random_panel(_DecoderFixture(always_valid=True), _panel(), set(), candidate, repair_limit=0)
    versus_original = paired_compare(first, candidate, bootstrap_draws=10)
    original_bytes = (candidate / f"paired_comparison-{versus_original['reference_model_state_sha256'][:12]}.json").read_bytes()
    versus_control = paired_compare(second, candidate, bootstrap_draws=10)
    assert versus_original["receipt_path"] != versus_control["receipt_path"]
    assert (candidate / f"paired_comparison-{versus_original['reference_model_state_sha256'][:12]}.json").read_bytes() == original_bytes


def test_failure_receipt_is_retained_and_all_module_training_flags_restored(tmp_path):
    class Failing(_DecoderFixture):
        def decode(self, z):
            raise RuntimeError("fixture decoder failure")
    model = Failing()
    model.train()
    with pytest.raises(RuntimeError, match="fixture decoder failure"):
        evaluate_random_panel(model, _panel(), set(), tmp_path / "failed", repair_limit=0)
    assert model.training
    receipt = json.loads((tmp_path / "failed/panel.json").read_text())
    assert receipt["status"] == "failed" and receipt["retained_hard_attempts"] == 0


@pytest.mark.parametrize("kwargs", [{"samples_per_latent": 0}, {"sample_seed": True}, {"repair_limit": -1}, {"batch_size": 0}])
def test_strict_panel_budgets(kwargs, tmp_path):
    with pytest.raises(ValueError):
        evaluate_random_panel(_DecoderFixture(), _panel(), set(), tmp_path / "invalid", **kwargs)
