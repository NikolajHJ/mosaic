import torch
from torch import nn

from latentmol.evaluation.decoder_topology import graph_topology, reconstruction_topology
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def _hard(atoms, edge_lists):
    atoms = torch.tensor(atoms)
    edges = torch.zeros(*atoms.shape, atoms.shape[1], dtype=torch.long)
    for row, pairs in enumerate(edge_lists):
        for i, j in pairs:
            edges[row, i, j] = edges[row, j, i] = 1
    return HardGraphBatch(atoms, torch.zeros_like(atoms), edges)


def test_cycle_rank_ignores_slot_order_and_bond_multiplicity():
    graph = _hard([[1]*4, [1, 1, 0, 1]], [[(0, 1), (1, 2), (2, 0), (2, 3), (0, 3)], [(0, 1)]])
    graph.bonds[0, 0, 1] = graph.bonds[0, 1, 0] = 2
    assert graph_topology(graph).tolist() == [[4, 1, 2], [3, 2, 0]]
    p = [3, 1, 2, 0]
    shuffled = HardGraphBatch(graph.atoms[:, p], graph.charges[:, p], graph.bonds[:, p][:, :, p])
    torch.testing.assert_close(graph_topology(graph), graph_topology(shuffled))


def test_empty_and_fragmented_predictions_do_not_get_tree_credit():
    truth = _hard([[1, 1, 1]] * 3, [[(0, 1), (1, 2)], [(0, 1), (1, 2)], [(0, 1), (1, 2), (0, 2)]])
    pred = _hard([[0, 0, 0], [1, 0, 1], [1, 1, 1]], [[], [], [(0, 1), (1, 2), (2, 0)]])

    class Model(nn.Module):
        spec = GraphSpec(max_nodes=3)

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(1.))

        def encoder(self, graph, from_canonical_cache):
            return graph.atoms, None

        def decode(self, z):
            return DecoderOutput(*(nn.functional.one_hot(v, n).float() for v, n in
                                   ((pred.atoms, len(self.spec.atomic_numbers)),
                                    (pred.charges, len(self.spec.formal_charges)),
                                    (pred.bonds, len(self.spec.bond_names)))))

    report = reconstruction_topology(Model(), [truth])
    assert report["acyclic_targets"]["empty_predictions"] == 1
    assert report["acyclic_targets"]["connected_and_correct_cycle_rank"] == 0
    assert report["cyclic_targets"]["connected_and_correct_cycle_rank"] == 1
    assert report["all"]["targets"] == 3
