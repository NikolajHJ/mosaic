"""Small CPU fixtures for matching, consistent encoding and frozen evaluation."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
import torch

from latentmol.chemistry import encode_smiles
from latentmol.datasets.data import prepare_dataset
from latentmol.evaluation.checkpoint_comparison import graph_record
from latentmol.evaluation.interpolation import (
    _encode, _one_hot, _validate, interpolation_codes, probability_distance, run_interpolation,
)
from latentmol.evaluation.molecular_pairs import _assignment, match_atoms, permute_graph, select_pairs
from latentmol.experiments import file_sha256
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec, HardGraphBatch


def _graph(smiles, spec):
    encoded = encode_smiles(smiles, spec)
    return HardGraphBatch(*(torch.tensor(encoded[key], dtype=torch.long)[None]
                            for key in ("atom_types", "charges", "bond_types")))


def test_matching_preserves_identity_and_reencodes_the_reordered_endpoint():
    spec = GraphSpec(max_nodes=12)
    a = encode_smiles("Cc1ccccc1O", spec)["smiles"]
    b = encode_smiles("Oc1ccccc1F", spec)["smiles"]
    mapping, reason = match_atoms(Chem.MolFromSmiles(a), Chem.MolFromSmiles(b), minimum_fraction=.8)
    assert reason is None and mapping["common_core_fraction"] >= .8
    permutation = mapping["b_original_atom_for_aligned_slot"]
    assert sorted(permutation) == list(range(8)) and not mapping["identity_permutation"]
    canonical = _graph(b, spec)
    matched = permute_graph(canonical, permutation)
    assert graph_record(matched, 0, spec)["smiles"] == b
    assert not torch.equal(canonical.atoms, matched.atoms)
    torch.manual_seed(7)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1)).eval()
    original, reordered = _encode(model, [canonical, matched], 2)
    assert not torch.allclose(original, reordered)


def test_assignment_is_optimal_for_small_residual_and_rejects_unbounded_work():
    assert _assignment([[3, 0], [0, 3]]) == (1, 0)
    assert _assignment([]) == ()
    with pytest.raises(ValueError, match="<=6"):
        _assignment([[0]*7 for _ in range(7)])


def test_interpolation_fractions_and_probability_targets_are_exact():
    a, b = torch.zeros(1, 4), torch.ones(1, 4)
    result = interpolation_codes(a, b, [0., .25, .5, .75, 1.])
    torch.testing.assert_close(result[:, 0], torch.tensor([0., .25, .5, .75, 1.]))
    spec = GraphSpec(max_nodes=6)
    ga, gb = _graph("CCO", spec), _graph("CCN", spec)
    pa, pb = _one_hot(ga, spec), _one_hot(gb, spec)
    mixed = tuple(.75*x+.25*y for x, y in zip(pa, pb))
    assert probability_distance(mixed, mixed, ga, gb)["equal_term_mean"] == pytest.approx(0.)
    assert probability_distance(pa, pb, ga, gb)["atoms"] > 0


def test_atom_map_checks_and_declared_fractions():
    spec = GraphSpec(max_nodes=6)
    with pytest.raises(ValueError, match="bijective"):
        permute_graph(_graph("CCO", spec), [0, 0, 2])
    mapping, reason = match_atoms(Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CCC"))
    assert mapping is None and reason == "different_atom_count"
    with pytest.raises(ValueError, match="25%"):
        _validate(dict(version="matched-molecular-interpolation-v1", split="val", fractions=[.2]))


def test_complete_frozen_interpolation_and_saved_report(tmp_path):
    spec = GraphSpec(max_nodes=8)
    source = tmp_path/"molecules.smi"
    source.write_text("\n".join(["CCO", "CCN", "CCS", "CCC", "CCF", "CCCl", "COC", "CNC"]))
    data = tmp_path/"data"
    prepare_dataset(source, data, spec=spec, train_count=2, val_count=6, seed=31)
    manifest = json.loads((data/"manifest.json").read_text())
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1))
    checkpoint = tmp_path/"epoch45.pt"
    torch.save(dict(format_version=1, epoch=45, graph_spec=spec.to_dict(),
        model_config=asdict(model.config), model_state=model.state_dict(),
        dataset_sha256=manifest["dataset_sha256"]), checkpoint)
    config = dict(version="matched-molecular-interpolation-v1", epoch=45,
        checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
        data=str(data), data_manifest_sha256=file_sha256(data/"manifest.json"),
        dataset_sha256=manifest["dataset_sha256"], split="val", pool_count=6,
        pair_count=2, pair_seed=13, min_atoms=2, max_atoms=8, min_similarity=.01,
        min_core_fraction=.5, neighbors_per_molecule=8, max_candidate_pairs=20,
        max_pairs_per_scaffold=10, mcs_timeout_seconds=1, match_limit=16,
        alignment_limit=32, max_atom_changes=2, max_charge_changes=2, max_bond_changes=4,
        fractions=[.25, .5, .75], categorical_seed=14, categorical_draws=2,
        probability_snapshot_pairs=1, batch_size=4)
    output = tmp_path/"results"
    result = run_interpolation(config, output, device="cpu")
    assert result["status"] == "completed" and result["model_unchanged"]
    assert not result["training_started"] and result["final_test_split_untouched"]
    assert file_sha256(checkpoint) == config["checkpoint_sha256"]
    pairs = json.loads((output/"pairs.json").read_text())["pairs"]
    assert len(pairs) == 2
    assert len({p[k] for p in pairs for k in ("a_index", "b_index")}) == 4
    for arm in ("canonical", "matched"):
        with gzip.open(output/arm/"raw-graphs.jsonl.gz", "rt") as stream:
            raw = [json.loads(line) for line in stream]
        assert len(raw) == 2*5*3 and all("graph" in r for r in raw)
        assert {r["fraction"] for r in raw} == {0., .25, .5, .75, 1.}
        points = result["arms"][arm]["points"]
        assert len(points) == 5
        assert points[2]["views"]["all_pairs"]["argmax"]["attempts"] == 2
        assert "validity_wilson95" not in points[2]["views"]["all_pairs"]["categorical"]
    assert (output/"report.html").is_file() and not (output/"report-failure.json").exists()
    assert "Raw labels and chemistry assessment" in (output/"report.html").read_text()
    with pytest.raises(FileExistsError):
        run_interpolation(config, output, device="cpu")


def test_no_pairs_is_a_recorded_finding_without_relaxing_thresholds():
    config = dict(pool_count=2, pair_seed=1, min_atoms=1, max_atoms=8,
        min_similarity=.99, neighbors_per_molecule=2, max_candidate_pairs=4,
        pair_count=1, max_pairs_per_scaffold=2)
    pairs, receipt = select_pairs(["CC", "CO"], config)
    assert not pairs and receipt["warning"] and receipt["selected_pairs"] == 0
