"""Numerical diagnostic-contract tests; run on DTU HPC only."""

import torch

from latentmol.architecture_study import graph_patterns
from latentmol.representation import GraphSpec, HardGraphBatch


def test_pattern_counts_keep_invalid_graphs_and_charge_bond_relationships():
    spec = GraphSpec(max_nodes=3)
    atoms = torch.tensor([[1, 3, 0], [3, 1, 1]])
    neutral, positive = spec.formal_charges.index(0), spec.formal_charges.index(1)
    charges = torch.tensor([[neutral, neutral, neutral], [positive, neutral, neutral]])
    bonds = torch.zeros(2, 3, 3, dtype=torch.long)
    bonds[0, 0, 1] = bonds[0, 1, 0] = 1
    bonds[1, 0, 1:] = 2
    bonds[1, 1:, 0] = 2
    counts = graph_patterns(HardGraphBatch(atoms, charges, bonds), spec)
    # The second graph has an unusual oxygen pattern; the raw diagnostic must
    # retain it rather than silently repairing or filtering it.
    assert counts["size"] == {"2": 1, "3": 1}
    assert counts["element_charge"]["8|1"] == 1
    assert counts["element_bond_signature"]["8|1|0,2,0,0"] == 1
    assert counts["bond_endpoint_chemistry"]["(6, 0)|(8, 1)|DOUBLE"] == 2
    assert counts["bond_endpoint_chemistry"]["(6, 0)|(6, 0)|NONE"] == 1
