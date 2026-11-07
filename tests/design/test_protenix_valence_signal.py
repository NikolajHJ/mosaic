from dataclasses import asdict, replace
import hashlib
import json

import torch

from latentmol.design.latent_protenix import LatentDesignConfig, evaluate_point
from latentmol.representation import DecoderOutput, GraphSpec


class DenseDecoder:
    spec = GraphSpec(max_nodes=4)

    def decode(self, point):
        atoms = torch.zeros(1, 4, len(self.spec.atomic_numbers))
        atoms[..., 1] = 15
        charges = torch.zeros(1, 4, len(self.spec.formal_charges))
        charges[..., 2] = 15
        bonds = torch.zeros(1, 4, 4, len(self.spec.bond_names))
        bonds[..., 3] = point[0, 0]
        bonds[:, :, 1, 3] = point[0, 1]
        bonds[:, 1, :, 3] = point[0, 1]
        return DecoderOutput(atoms, charges, bonds)


def test_valence_excess_supplies_gradient_without_a_repair_witness():
    def scorer(graph):
        return graph.bonds[..., 3].sum() * 0, {}

    active = torch.tensor([True, False, True, True])
    point = torch.tensor([[2., 3.]])
    control = LatentDesignConfig()
    treatment = replace(control, valence_excess_weight=1.)
    settings = dict(target_bundle=(None, {"status": "censored"}))
    baseline = evaluate_point(DenseDecoder(), scorer, point, active, control, 0, **settings)
    result = evaluate_point(DenseDecoder(), scorer, point, active, treatment, 0, **settings)
    assert baseline["gradient"].count_nonzero() == 0
    assert "valence_excess" not in baseline
    assert result["chemistry"] == 0 and result["valence_excess"] > 0
    assert result["gradient"][0, 0] > 0
    assert result["gradient"][0, 1] == 0
    torch.testing.assert_close(result["gradient"], result["valence_excess_gradient"])


def test_zero_valence_weight_preserves_legacy_checkpoint_identity():
    config = LatentDesignConfig()
    legacy = asdict(config)
    del legacy["valence_excess_weight"]
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    assert config.identity() == expected
    assert replace(config, valence_excess_weight=1.).identity() != expected
