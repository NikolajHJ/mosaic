"""Guard the DeFoG codec against silent charge, occupancy, or bond changes."""

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.defog.codec import GUACAMOL_ATOMIC_NUMBERS, to_guacamol_hard
from latentmol.representation import GraphSpec, HardGraphBatch


def _graph(smiles: str):
    spec = GraphSpec()
    row = encode_smiles(smiles, spec)
    return spec, HardGraphBatch(
        atoms=torch.as_tensor(row["atom_types"], dtype=torch.long)[None],
        charges=torch.as_tensor(row["charges"], dtype=torch.long)[None],
        bonds=torch.as_tensor(row["bond_types"], dtype=torch.long)[None],
    )


def test_neutral_aromatic_ring_maps_without_padding_or_bond_relabeling():
    spec, graph = _graph("c1ccccc1")
    mapped = to_guacamol_hard(graph, spec)
    assert mapped.x.shape == (1, 6, 12)
    assert mapped.e.shape == (1, 6, 6, 5)
    assert mapped.occupied_slots == tuple(range(6))
    assert mapped.node_mask.all()
    assert torch.equal(mapped.x.argmax(-1), torch.zeros((1, 6), dtype=torch.long))
    assert int(mapped.e[0, :, :, 4].sum()) == 12
    assert torch.equal(mapped.e[0], mapped.e[0].transpose(0, 1))
    assert not mapped.e[0, torch.arange(6), torch.arange(6)].any()
    assert GUACAMOL_ATOMIC_NUMBERS[0] == 6


def test_charged_candidate_is_rejected_instead_of_neutralized():
    spec, graph = _graph("CC(=O)[O-]")
    with pytest.raises(ValueError, match="Formal charges"):
        to_guacamol_hard(graph, spec)


def test_invalid_padding_bond_and_float_labels_are_rejected():
    spec, graph = _graph("CO")
    graph.bonds[0, 0, 10] = graph.bonds[0, 10, 0] = 1
    with pytest.raises(ValueError, match="NO_ATOM padding"):
        to_guacamol_hard(graph, spec)
    graph.bonds[0, 0, 10] = graph.bonds[0, 10, 0] = 0
    graph.atoms = graph.atoms.float()
    with pytest.raises(ValueError, match="must be integers"):
        to_guacamol_hard(graph, spec)


def test_noncontiguous_occupied_slots_are_preserved_in_mapping():
    spec, graph = _graph("CO")
    # Padding is explicit, so a gap is legal. Its position is recorded.
    graph.atoms[0, 3] = graph.atoms[0, 1]
    graph.charges[0, 3] = graph.charges[0, 1]
    graph.bonds[0, 0, 3] = graph.bonds[0, 3, 0] = graph.bonds[0, 0, 1]
    graph.atoms[0, 1] = 0
    graph.bonds[0, 0, 1] = graph.bonds[0, 1, 0] = 0
    mapped = to_guacamol_hard(graph, spec)
    assert mapped.occupied_slots == (0, 3)
    assert mapped.x.shape[1] == 2
