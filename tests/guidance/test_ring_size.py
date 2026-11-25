"""Fixed-size conditional ring credit and its real pooled score-gradient path."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.guidance import functional_groups
from latentmol.guidance.ring_size import ring_size_score, validate_ring_policy
from latentmol.guidance.ring_size_reference import REFERENCE_ID, TRAINING_VIEW_SHA256
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob


def test_condition_on_matched_sizes_and_preserve_acyclic_and_invalid_mass():
    # At size 8 the cyclic fraction has one ring of each kind and a normal six-ring.
    # At size 5 there's only a single cyclic draw: measurable, no comparison partner.
    rows = np.array([[8, 1, 1, 0], [8, 1, 0, 1], [8, 1, 0, 0], [8, 0, 0, 0],
                     [5, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    ref = np.zeros((9, 3), dtype=int)
    ref[5], ref[8] = [20, 2, 0], [20, 1, 1]
    w, loss, stats = ring_size_score(rows, ref)
    assert stats["ring_reference_supported_valid"] == 5 and stats["ring_reference_unsupported_valid"] == 1
    assert stats["ring_any_hits"] == 4 and stats["ring_matched_cyclic_rows"] == 3
    assert stats["ring3_observed_rate"] == pytest.approx(2/5)
    assert stats["ring3_reference_rate"] == pytest.approx((.10 + 4*.05)/5)
    assert stats["ring8plus_reference_rate"] == pytest.approx(4*.05/5)
    assert loss == pytest.approx((.34**2 + .16**2)/4)
    assert w[0] > w[2] and w[1] > w[2] and w[2] < 0
    assert w[:3].sum() == pytest.approx(0, abs=1e-14)
    np.testing.assert_array_equal(w[3:], 0)
    # There is no penalty when both reference proportions are met or exceeded by the real data.
    ref[5, 1:] = ref[8, 1:] = 20
    w, loss, _ = ring_size_score(rows, ref)
    assert not w.any() and loss == 0
    w, loss, stats = ring_size_score(rows[-2:], ref)
    assert not w.any() and loss == 0 and stats["ring3_observed_rate"] is None


def test_ring_policy_requires_right_reference_and_finite_nonnegative_weight():
    policy = {"reference": REFERENCE_ID, "weight": .1}
    validate_ring_policy(policy, TRAINING_VIEW_SHA256)
    with pytest.raises(ValueError, match="different training view"):
        validate_ring_policy(policy, "different")
    for bad in (dict(policy, weight=-1.), dict(policy, weight=float("nan")), dict(policy, reference="wrong")):
        with pytest.raises(ValueError):
            validate_ring_policy(bad)


def test_rdkit_ring_labels_share_sanitization_and_feed_pooled_bond_gradients(monkeypatch):
    spec = GraphSpec(max_nodes=12)
    # The first four molecules have eight heavy atoms. The last is made disconnected.
    smiles = ("C1CCCCCCC1", "C1CC1CCCCC", "CCC1CCCCC1", "CCCCCCCC", "C", "C1CCCCCCC1")
    encoded = [encode_smiles(s, spec) for s in smiles]
    hard = HardGraphBatch(*(torch.tensor(np.stack([g[k] for g in encoded]), dtype=torch.long)
                            for k in ("atom_types", "charges", "bond_types")))
    hard.atoms[-1, -1] = spec.atomic_numbers.index(6)
    heads = [torch.nn.Parameter(F.one_hot(t, len(v)).double() * 30) for t, v in
             zip((hard.atoms, hard.charges, hard.bonds), (spec.atomic_numbers, spec.formal_charges, spec.bond_names))]
    class Model:
        def __init__(self):
            self.spec = spec
            self.decode_calls = 0

        def decode(self, ix):
            self.decode_calls += 1
            return DecoderOutput(*(head[ix] for head in heads))

    original = functional_groups.inspect_graph
    calls = []
    def count_inspections(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(functional_groups, "inspect_graph", count_inspections)
    model = Model()
    policy = dict(score_draws=len(smiles), aromatic_floor=0, amide_floor=0,
                  ring_size={"reference": REFERENCE_ID, "weight": .1})
    ix = torch.arange(len(smiles))
    pool = functional_groups.prepare_pooled_motif_score(model, ix, policy, 2, torch.Generator().manual_seed(827))
    assert model.decode_calls == 3 and len(calls) == 6
    assert (pool.stats["valid"], pool.stats["ring3_hits"], pool.stats["ring8plus_hits"],
            pool.stats["ring_any_hits"], pool.stats["ring_reference_supported_valid"]) == (5, 1, 1, 3, 4)
    assert pool.stats["penalty"] == pytest.approx(.1 * pool.stats["ring_size_penalty"])
    assert pool.stats["penalty"] > 0
    for i, graph in pool.samples.items():
        torch.testing.assert_close(graph.bonds, hard.bonds[i:i+2])
    # Compare chunked surrogate to direct all-six score and check live edge gradients.
    expected = hard_graph_log_prob(model.decode(ix), hard, spec) @ torch.tensor(pool.weights)
    observed = sum(pool.surrogate(i, model.decode(ix[i:i+2]), spec) for i in pool.samples)
    logged, stats = functional_groups.functional_group_loss(
        model.decode(ix[:2]), spec, torch.Generator(), aromatic_floor=0, amide_floor=0, pooled_score=pool)
    assert (logged + observed).item() == pytest.approx(.1 * stats["ring_size_penalty"])
    g1 = torch.autograd.grad(expected, heads)
    g2 = torch.autograd.grad(observed, heads)
    for a,b in zip(g1,g2):
        torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-15)
        assert torch.isfinite(b).all()
    assert g2[2][:3].abs().sum() > 0 and g2[2][3:].abs().sum() == 0
