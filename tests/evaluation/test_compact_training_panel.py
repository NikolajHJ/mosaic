"""The on-training panel compares the same graphs and prior latents at every observation."""

from types import SimpleNamespace

import pytest
from rdkit import rdBase
import torch
from torch.nn.functional import one_hot

from latentmol.chemistry import encode_canonical_smiles
from latentmol.evaluation.compact_training_panel import CompactTrainingPanel
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.training.losses import LossConfig


class ValidationRows(list):
    split = "val"
    dataset_sha256 = "test-val"
    spec = GraphSpec(max_nodes=5)
    manifest = dict(provenance={"canonicalization": "RDKit canonical SMILES; reparsed for stable atom ordering"},
                    rdkit_version=rdBase.rdkitVersion)


class ReversedReconstruction(torch.nn.Module):
    """Reproduce chemistry under reversed slots, but decode the prior to empty graphs."""

    config = SimpleNamespace(latent=2)
    spec = ValidationRows.spec

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)

    def _output(self, graph):
        return DecoderOutput(*(50 * one_hot(x, n).float() for x, n in zip(
            (graph.atoms, graph.charges, graph.bonds),
            (len(self.spec.atomic_numbers), len(self.spec.formal_charges), len(self.spec.bond_names)))))

    def forward(self, graph, *, sample_latent, from_canonical_cache):
        assert not self.training and not sample_latent and from_canonical_cache
        mu = torch.zeros(len(graph.atoms), self.config.latent)
        decoded = self._output(HardGraphBatch(graph.atoms.flip([1]), graph.charges.flip([1]),
                                              graph.bonds.flip([1, 2])))
        return SimpleNamespace(decoded=decoded, mu=mu, logvar=torch.zeros_like(mu))

    def decode(self, codes):
        n, slots = len(codes), self.spec.max_nodes
        atoms = torch.zeros(n, slots, dtype=torch.long)
        return self._output(HardGraphBatch(atoms, atoms, torch.zeros(n, slots, slots, dtype=torch.long)))


def test_training_panel_reuses_permutation_aware_reconstruction_and_preserves_training_rng():
    rows = ValidationRows()
    for smiles in ("CC", "CO", "CCO", "CNC", "CCN", "CCCl"):
        raw = encode_canonical_smiles(smiles, rows.spec)
        rows.append({key: torch.tensor(raw[source]).long() for key, source in
                    (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})
    config = SimpleNamespace(mode="vae", loss=LossConfig(atom_order="graph"),
                             model=SimpleNamespace(latent=2, encoder_pooling="canonical"))
    model = ReversedReconstruction()
    model.projection.eval()
    rng = torch.random.get_rng_state()
    panel = CompactTrainingPanel(rows, config, seed=121, validation_count=4, prior_count=5, batch_size=2)
    assert panel.metadata["canonical_rows_checked"] == 4
    assert len(set(panel.metadata["validation_indices"])) == 4
    assert sum(panel.metadata["validation_target_size_histogram"].values()) == 4
    result = panel.observe(model)
    assert result["reconstruction"]["exact_labeled_graph_count"] == 4
    assert result["reconstruction"]["valid_chemistry_count"] == 4
    assert result["reconstruction"]["exact_reconstruction"] == 1
    for summary in (result["prior"], result["prior"]["categorical"]):
        assert summary["prior_empty"] == summary["prior_points"] == 5
        assert summary["prior_empty_fraction"] == 1
        assert summary["prior_at_capacity_fraction"] == summary["prior_valid_fraction"] == 0
    assert result == panel.observe(model)
    assert model.training and not model.projection.training and torch.equal(rng, torch.random.get_rng_state())
    rows.split = "train"
    with pytest.raises(ValueError, match="validation split"):
        CompactTrainingPanel(rows, config, seed=121, validation_count=4)
