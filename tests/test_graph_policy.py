"""Shared policy retains old Boltz import and original probability semantics."""

import torch
import pytest

from latentmol.boltz.adapter import fixed_occupancy_graph as legacy_fixed_graph
from latentmol.graph_policy import fixed_occupancy_graph, graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec


def test_legacy_import_is_same_function_and_shared_policy_chooses_explicitly():
    assert legacy_fixed_graph is fixed_occupancy_graph
    spec = GraphSpec(max_nodes=3)
    decoded = DecoderOutput(torch.randn(1, 3, 10), torch.randn(1, 3, 5), torch.randn(1, 3, 3, 5))
    active = torch.tensor([True, False, True])
    soft = graph_from_decoder(decoded, spec, temperature=0.7)
    expected_soft = decoded.probabilities(0.7)
    fixed = graph_from_decoder(decoded, spec, temperature=0.7, fixed_active=active)
    expected_fixed = legacy_fixed_graph(decoded, active, spec, 0.7)
    for name in ("atoms", "charges", "bonds"):
        torch.testing.assert_close(getattr(soft, name), getattr(expected_soft, name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(fixed, name), getattr(expected_fixed, name), rtol=0, atol=0)
    assert not torch.allclose(soft.occupancy, active.float()[None])
    torch.testing.assert_close(fixed.occupancy, active.float()[None], rtol=0, atol=0)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("field,shape", [
    ("atoms", (1, 2, 10)), ("atoms", (1, 3, 9)),
    ("charges", (2, 3, 5)), ("bonds", (1, 3, 3, 4)),
])
def test_graph_policy_rejects_wrong_schema_or_batch_in_both_paths(fixed, field, shape):
    spec = GraphSpec(max_nodes=3)
    decoded = DecoderOutput(torch.zeros(1, 3, 10), torch.zeros(1, 3, 5), torch.zeros(1, 3, 3, 5))
    setattr(decoded, field, torch.zeros(shape))
    with pytest.raises(ValueError, match="GraphSpec"):
        graph_from_decoder(decoded, spec, fixed_active=torch.ones(3, dtype=torch.bool) if fixed else None)


@pytest.mark.parametrize("mask", [
    torch.ones(3), torch.ones(3, dtype=torch.long),
    torch.ones(1, 3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool),
])
def test_fixed_policy_requires_exact_nonempty_boolean_mask(mask):
    spec = GraphSpec(max_nodes=3)
    decoded = DecoderOutput(torch.zeros(1, 3, 10), torch.zeros(1, 3, 5), torch.zeros(1, 3, 3, 5))
    with pytest.raises(ValueError, match="nonempty boolean mask"):
        graph_from_decoder(decoded, spec, fixed_active=mask)


def test_fixed_policy_rejects_unsupported_batch_before_legacy_tensor_operations():
    spec = GraphSpec(max_nodes=3)
    decoded = DecoderOutput(torch.zeros(2, 3, 10), torch.zeros(2, 3, 5), torch.zeros(2, 3, 3, 5))
    assert graph_from_decoder(decoded, spec).atoms.shape == (2, 3, 10)
    with pytest.raises(ValueError, match="batch size one"):
        graph_from_decoder(decoded, spec, fixed_active=torch.ones(3, dtype=torch.bool))
