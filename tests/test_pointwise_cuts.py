"""Independent bridge regressions; execute only on allocated DTU HPC.

These fixtures exercise the real adapter without importing Boltz or executing a
protein model. The atom map is intentionally permuted and noncontiguous.
"""
from copy import deepcopy

import pytest
import torch

from latentmol.boltz import pointwise_cuts as cuts
from latentmol.boltz.pointwise_contract import verify_bridge
from latentmol.multimodel.study import tensor_digest
from test_boltz_adapter import prepared_fixture


def graph_fixture():
    prepared = prepared_fixture()
    graph = prepared.seed.to_relaxed(prepared.spec)
    graph.atoms[0, 0].zero_()
    graph.atoms[0, 0, 1:3] = torch.tensor([.6, .4])
    graph.charges[0, 0].zero_()
    graph.charges[0, 0, 1:3] = torch.tensor([.3, .7])
    mixture = torch.tensor([.25, 0., 0., .75, 0.])
    graph.bonds[0, 0, 2] = graph.bonds[0, 2, 0] = mixture
    return prepared, graph


def audit(prepared, graph):
    return cuts.validate_bridge(graph, prepared, context_sha256="c" * 64,
                                parity_rtol=1e-4, parity_atol=1e-5)


def test_affine_bridge_has_independent_value_both_ad_and_fixed_fd_receipts():
    prepared, graph = graph_fixture()
    identity = tensor_digest({key: getattr(graph, key) for key in cuts.FIELDS})
    report = audit(prepared, graph)
    assert report["passed"]
    assert verify_bridge(report, graph_sha256=identity, context_sha256="c" * 64)
    assert report["scorer_calls"] == 0
    assert report["affine_extension_not_molecule_samples"] is True
    assert report["auxiliary_unchanged"] and report["context_unchanged"]
    assert [row["name"] for row in report["directions"]] == ["affine_1", "affine_2"]
    for direction in report["directions"]:
        assert set(direction["jvp"]["rows"]) == set(cuts.CHANNELS)
        assert set(direction["vjp"]["rows"]) == set(cuts.FIELDS)
        assert [row["epsilon"] for row in direction["finite_differences"]] == [1e-4, .001]
    assert tensor_digest({key: getattr(graph, key) for key in cuts.FIELDS}) == identity


def test_declared_tangents_preserve_probability_sums_occupancy_and_symmetry():
    prepared, graph = graph_fixture()
    directions = cuts.tangent_plan(graph, prepared)
    for direction in directions.values():
        for field in cuts.FIELDS:
            value = getattr(direction, field)
            torch.testing.assert_close(value.sum(-1), torch.zeros_like(value[..., 0]), rtol=0, atol=0)
        assert direction.atoms[..., 0].count_nonzero() == 0
        assert direction.atoms[:, ~prepared.active].count_nonzero() == 0
        assert direction.charges[:, ~prepared.active].count_nonzero() == 0
        assert direction.bonds[:, ~prepared.active].count_nonzero() == 0
        assert direction.bonds.diagonal(dim1=1, dim2=2).count_nonzero() == 0
        assert torch.equal(direction.bonds, direction.bonds.transpose(1, 2))
        torch.testing.assert_close(sum(getattr(direction, key).square().sum() for key in cuts.FIELDS),
                                   torch.tensor(1., dtype=torch.float64), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("channel", ["ref_charge", "token_bonds", "bond_types"])
def test_independent_oracle_rejects_corrupted_mapping(monkeypatch, channel):
    prepared, graph = graph_fixture()
    original = cuts.adapt_graph
    def corrupted(value, context):
        features, typed = original(value, context)
        features = dict(features)
        if channel == "bond_types":
            typed = typed.roll(1, -1)
        else:
            features[channel] = features[channel] * 2
        return features, typed
    monkeypatch.setattr(cuts, "adapt_graph", corrupted)
    assert not audit(prepared, graph)["passed"]


def test_detached_mapping_tangent_fails_closed(monkeypatch):
    prepared, graph = graph_fixture()
    original = cuts.adapt_graph
    def detached(value, context):
        features, typed = original(value, context)
        return features, typed.detach()
    monkeypatch.setattr(cuts, "adapt_graph", detached)
    with pytest.raises(ValueError, match="tangent missing"):
        audit(prepared, graph)


def test_auxiliary_replacement_and_tampered_receipts_rejected(monkeypatch):
    prepared, graph = graph_fixture()
    report = audit(prepared, graph)
    changed = deepcopy(report)
    changed["directions"][0]["finite_differences"][0]["rows"]["ref_element"]["analytic_sha256"] = "0" * 64
    assert not verify_bridge(changed, graph_sha256=report["graph_sha256"], context_sha256="c" * 64)
    original = cuts.adapt_graph
    def replaced(value, context):
        features, typed = original(value, context)
        return {**features, "ref_pos": features["ref_pos"].clone()}, typed
    monkeypatch.setattr(cuts, "adapt_graph", replaced)
    assert not audit(prepared, graph)["auxiliary_unchanged"]
