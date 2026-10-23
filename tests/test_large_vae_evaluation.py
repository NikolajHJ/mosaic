"""Allocated regressions for serialization failures in generation diagnostics."""

import gzip
import json
from types import SimpleNamespace

import pytest
import torch
from rdkit.Chem import rdFingerprintGenerator

from latentmol.chemistry import DecodeResult
import latentmol.large_vae_evaluation as evaluation
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec


def record(smiles, *, valid=True):
    return {"valid": valid, "smiles": smiles, "failure_reason": None,
            "atoms": 2, "elements": {"6": 2}, "formal_charges": {"0": 2},
            "bond_types": {"SINGLE": 1}, "rings": None}


def test_summary_retains_legacy_unparsable_attempt_and_separates_sanitization():
    rows = [record("CC"), record("c1cc1"), record(None, valid=False)]
    rows[1]["target_smiles"] = "c1cc1"
    report = evaluation.summarize(rows)
    assert report["attempts"] == 3 and report["valid_count"] == 1
    assert report["validity"] == pytest.approx(1 / 3)
    assert report["smiles_roundtrip_failures"] == 1
    assert report["graph_sanitization_valid_count"] == 2
    assert report["exact_reconstruction"] == 0
    assert report["valid_ring_count_mean"] == 0
    assert report["diversity_cohort_size"] == 1
    assert rows[1]["valid"] and rows[1]["rings"] is None  # Historical evidence unchanged.


def test_graph_record_preserves_failed_smiles_and_raw_graph(monkeypatch):
    monkeypatch.setattr(evaluation, "inspect_graph", lambda *args: DecodeResult(smiles="c1cc1"))
    spec = GraphSpec(max_nodes=3)
    hard = SimpleNamespace(atoms=torch.tensor([[1, 1, 1]]), charges=torch.tensor([[2, 2, 2]]),
                           bonds=torch.zeros((1, 3, 3), dtype=torch.long))
    row = evaluation.graph_record(hard, 0, spec)
    assert row["graph_sanitization_valid"] is True
    assert row["valid"] is False and row["smiles_roundtrip_valid"] is False
    assert row["failure_reason"] == "smiles_roundtrip"
    assert row["smiles_roundtrip_failure_detail"] == "rdkit_parse_failed"
    assert row["smiles"] == "c1cc1" and row["rings"] is None
    assert row["graph"]["atoms"] == [1, 1, 1]


def test_anchor_fingerprint_requires_parse_success():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    assert evaluation.checked_fingerprint(generator, record("c1cc1")) is None
    assert evaluation.checked_fingerprint(generator, record("CC")) is not None
    assert evaluation.checked_fingerprint(generator, record("CC", valid=False)) is None


def test_code_evaluation_retains_all_failed_draws_and_unparsable_anchors(tmp_path, monkeypatch):
    # Exercise the real batched evaluator, sampler and decoder with the exact
    # serialization-success/parse-failure contract that crashed the prior panel.
    monkeypatch.setattr(evaluation, "inspect_graph", lambda *args: DecodeResult(smiles="c1cc1"))
    spec = GraphSpec(max_nodes=3)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1))
    codes = torch.zeros((2, 4))
    anchors = evaluation.argmax_records(model, codes, 1)
    # Also accept legacy rows whose old validity flag was too permissive.
    anchors[0]["valid"] = True
    protocol = {"evaluation_batch_size": 1, "categorical_samples_per_latent": 2,
                "evaluation_seed": 17}
    report = evaluation.evaluate_codes(model, codes, tmp_path / "prior.jsonl", protocol,
                                       anchors={"baseline_decode": anchors})
    assert report["argmax"]["attempts"] == 2
    assert report["categorical"]["attempts"] == 4
    for method in ("argmax", "categorical"):
        assert report[method]["valid_count"] == 0
        assert report[method]["smiles_roundtrip_failures"] == report[method]["attempts"]
        assert report[method]["anchor_comparisons"]["similarity_to_baseline_decode"]["eligible"] == 0
        assert report[method]["anchor_comparisons"]["edits_to_baseline_decode"]["eligible"] == report[method]["attempts"]
    with gzip.open(tmp_path / "prior.jsonl.gz", "rt") as handle:
        raw = [json.loads(line) for line in handle]
    assert len(raw) == 6 and all("graph" in row for row in raw)
    assert all(not row["valid"] and row["graph_sanitization_valid"] for row in raw)
