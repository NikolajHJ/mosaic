"""Check complete motif labels, the finite-batch objective and its score gradient."""

import itertools

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.guidance.functional_groups import (
    _batch_shortfall, _motif_labels, functional_group_loss, prepare_pooled_motif_score,
)
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob, sample_hard_graph


SPEC = GraphSpec(max_nodes=12)


def _hard(smiles):
    graphs = [encode_smiles(s, SPEC) for s in smiles]
    return HardGraphBatch(*(torch.tensor(np.stack([g[k] for g in graphs]), dtype=torch.long)
                            for k in ("atom_types", "charges", "bond_types")))


def _logits(graphs):
    return DecoderOutput(*((F.one_hot(t, len(v)).float() * 16).requires_grad_()
                         for t, v in zip((graphs.atoms, graphs.charges, graphs.bonds),
                                         (SPEC.atomic_numbers, SPEC.formal_charges, SPEC.bond_names))))


def test_complete_rings_and_linked_amides_on_valid_molecules():
    graphs = _hard(["c1ccccc1", "C1CCCCC1", "C1=CC=CC=C1", "CC(=O)N", "NCCC=O", "CC(=O)NC"])
    # A ring of alternating single/double bonds is aromatic after perception.
    for i in range(6):
        j = (i + 1) % 6
        graphs.bonds[2, i, j] = graphs.bonds[2, j, i] = 1 + (i % 2 == 0)
    # An otherwise complete amide in a disconnected graph yields no credit.
    graphs.atoms[-1, -1] = SPEC.atomic_numbers.index(6)
    labels, valid = _motif_labels(graphs, SPEC)
    assert valid == 5
    assert labels.tolist() == [[1, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0]]


def test_expectation_of_score_gradient_matches_enumerated_batch_objective():
    # Different Bernoulli probabilities model three distinct decoder rows.
    logits = torch.tensor([-0.9, 0.2, 1.1], dtype=torch.double, requires_grad=True)
    floor = (0.45, 0.0)
    exact = 0
    estimated = 0
    for bits in itertools.product((0, 1), repeat=3):
        labels = np.column_stack((bits, (0, 0, 0)))
        weights, stats = _batch_shortfall(labels, floor)
        t = logits.new_tensor(bits)
        log_prob = -F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        probability = log_prob.sum().exp()
        exact = exact + probability * stats["penalty"]
        estimated = estimated + probability.detach() * (log_prob * logits.new_tensor(weights)).sum()
    exact_grad = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    score_grad = torch.autograd.grad(estimated, logits)[0]
    torch.testing.assert_close(exact_grad, score_grad, atol=1e-13, rtol=1e-12)
    assert (score_grad < 0).all()


def test_no_hits_or_abundant_groups_have_zero_gradient_and_distinct_diagnostics():
    for smiles, expected_penalty, active in [
        (["CC", "CC"], 0.005, 2), (["NC(=O)c1ccccc1"] * 2, 0, 0)
    ]:
        d = _logits(_hard(smiles))
        loss, stats = functional_group_loss(d, SPEC, torch.Generator().manual_seed(45),
                                           aromatic_floor=0.1, amide_floor=0.1)
        assert stats["valid"] == 2
        assert stats["aromatic_active_rows"] == stats["amide_active_rows"] == active
        assert not stats["aromatic_informative"] and not stats["amide_informative"]
        assert loss.item() == pytest.approx(expected_penalty)
        for grad in torch.autograd.grad(loss, (d.atoms, d.charges, d.bonds)):
            assert not grad.count_nonzero()


def test_informative_graphs_have_finite_live_derivatives_and_independent_rng():
    d = _logits(_hard(["CCC", "CC(=O)N", "c1ccccc1", "CCCC"]))
    state = torch.get_rng_state()
    loss, stats = functional_group_loss(d, SPEC, torch.Generator().manual_seed(79),
                                       aromatic_floor=0.4, amide_floor=0.4)
    assert torch.equal(torch.get_rng_state(), state)
    assert stats["aromatic_hits"] == stats["amide_hits"] == 1
    assert stats["aromatic_informative"] and stats["amide_informative"]
    assert loss.item() == pytest.approx(0.5 * (0.4 - 0.25) ** 2)
    for grad in torch.autograd.grad(loss, (d.atoms, d.charges, d.bonds)):
        assert torch.isfinite(grad).all() and grad.count_nonzero()


def test_invalid_floor_and_single_draw_fail():
    d = _logits(_hard(["C"]))
    for floor in (float("nan"), 1.1, 0.05):
        with pytest.raises(ValueError):
            functional_group_loss(d, SPEC, torch.Generator(), aromatic_floor=floor, amide_floor=0.08)


def test_pooled_score_matches_single_batch_gradient_and_replays_dropout():
    class Fixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = SPEC
            fixture = _logits(_hard(['CC', 'NC=O', 'c1ccccc1', 'CC', 'CC', 'CC', 'CC(=O)N', 'C']))
            self.tables = nn.ParameterList([nn.Parameter(x.detach().double() * .85)
                                            for x in (fixture.atoms, fixture.charges, fixture.bonds)])
            self.calls = []

        def decode(self, indices):
            heads = [table[indices] + nn.functional.dropout(torch.ones_like(table[indices]), .35, training=True)
                     for table in self.tables]
            self.calls.append([x.detach().clone() for x in heads])
            return DecoderOutput(*heads)

    model = Fixture()
    prior = torch.arange(8)
    policy = dict(score_draws=7, aromatic_floor=.87056, amide_floor=.47988)
    rng = torch.get_rng_state()
    sampler = torch.Generator().manual_seed(113)
    pooled = prepare_pooled_motif_score(model, prior, policy, 3, sampler)
    assert torch.equal(torch.get_rng_state(), rng)
    assert (pooled.stats['draws'], pooled.stats['aromatic_hits'], pooled.stats['amide_hits']) == (7,1,2)
    assert [len(s.atoms) for s in pooled.samples.values()] == [3, 3, 1]
    # Each physical gradient forward reuses its own exact dropout realization,
    # including the third batch where only the first of two rows is supervised.
    live = [model.decode(chunk) for chunk in prior.split(3)]
    for old,new in zip(model.calls[:3],model.calls[3:]):
        for a,b in zip(old,new):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    rng_after_live = torch.get_rng_state()
    separate_rng = torch.Generator().manual_seed(113)
    samples = []
    for i,d in enumerate(live):
        n = min(3, 7-i*3)
        sample = sample_hard_graph(DecoderOutput(d.atoms[:n], d.charges[:n], d.bonds[:n]),
                                   SPEC, generator=separate_rng)
        samples.append(sample)
        torch.testing.assert_close(sample.bonds, pooled.samples[3*i].bonds)
    joined = DecoderOutput(*(torch.cat([getattr(d,k) for d in live])[:7] for k in ('atoms','charges','bonds')))
    hard = HardGraphBatch(*(torch.cat([getattr(d,k) for d in samples]) for k in ('atoms','charges','bonds')))
    labels, _ = _motif_labels(hard, SPEC)
    weights, expected_stats = _batch_shortfall(labels, (.87056,.47988))
    np.testing.assert_array_equal(weights, pooled.weights)
    reference = hard_graph_log_prob(joined, hard, SPEC) @ joined.atoms.new_tensor(weights)
    direct = torch.autograd.grad(reference, model.tables,retain_graph=True)
    value, stats = functional_group_loss(live[0], SPEC, sampler, aromatic_floor=.87056, amide_floor=.47988,
                                         pooled_score=pooled)
    loss = value + sum(pooled.surrogate(3*i, d, SPEC) for i,d in enumerate(live))
    assert loss.item() == expected_stats['penalty'] and stats['local_draws'] == 3
    for expected, actual in zip(direct,torch.autograd.grad(loss,model.tables)):
        torch.testing.assert_close(expected,actual,atol=1e-13,rtol=1e-12)
        assert torch.isfinite(actual).all() and actual.count_nonzero()
    assert torch.equal(torch.get_rng_state(),rng_after_live)
