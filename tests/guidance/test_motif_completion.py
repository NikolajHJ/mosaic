"""Behavior of the conditional motif fallback, from decoder logits to a valid local edit."""

import numpy as np
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.guidance.functional_groups import _motif_labels, functional_group_loss
from latentmol.guidance.motif_completion import _completion_targets, completion_loss
from latentmol.representation import DecoderOutput, GraphSpec


SPEC = GraphSpec(max_nodes=12)


def logits(smiles):
    graphs = [encode_smiles(s, SPEC) for s in smiles]
    arrays = [np.stack([graph[field] for graph in graphs])
              for field in ("atom_types", "charges", "bond_types")]
    return DecoderOutput(*(F.one_hot(torch.as_tensor(array).long(), len(vocab))
                           .float().mul(18).requires_grad_()
                           for array, vocab in zip(arrays, (SPEC.atomic_numbers, SPEC.formal_charges, SPEC.bond_names))))


def test_zero_hit_batch_has_conditional_gradient_only_on_selected_undirected_edges():
    decoded = logits(["C1=CCCCC1", "NC(O)C", "CC"])

    def objective(weight):
        return functional_group_loss(decoded, SPEC, torch.Generator().manual_seed(37),
                                     aromatic_floor=.08, amide_floor=.12, completion_weight=weight)

    old_loss, old = objective(0)
    loss, new = objective(.1)
    assert old["valid"] == new["valid"] == 3
    assert old["aromatic_hits"] == old["amide_hits"] == new["aromatic_hits"] == new["amide_hits"] == 0
    assert new["completion_aromatic_rows"] == new["completion_amide_rows"] == 1
    fields = decoded.atoms, decoded.charges, decoded.bonds
    assert all(g.count_nonzero() == 0 for g in torch.autograd.grad(old_loss, fields))
    atom, charge, bonds = torch.autograd.grad(loss, fields)
    assert torch.isfinite(loss) and torch.isfinite(bonds).all()
    assert not atom.count_nonzero() and not charge.count_nonzero() and not bonds[..., 0].count_nonzero()
    torch.testing.assert_close(bonds, bonds.transpose(1, 2), atol=0, rtol=0)
    selected = bonds.abs().sum(-1) > 0
    assert selected.sum((1, 2)).tolist() == [12, 2, 0]
    hard = decoded.harden(SPEC)
    assert torch.all(hard.bonds[selected] != 0)
    steps = decoded.bonds.detach() - bonds
    old_type = decoded.bonds[..., 1:].detach().softmax(-1)
    new_type = steps[..., 1:].softmax(-1)
    # Every target type becomes more probable after a gradient step. Applying
    # those labels to the untouched skeleton should also make the claimed raw motif.
    for row, name in enumerate(("AROMATIC", "DOUBLE")):
        index = SPEC.bond_names.index(name)
        chosen = selected[row]
        assert torch.all(bonds[row, :, :, index][chosen] < 0)
        assert torch.all(new_type[row, :, :, index - 1][chosen] > old_type[row, :, :, index - 1][chosen])
        hard.bonds[row][chosen] = index
    labels, valid = _motif_labels(hard, SPEC)
    assert valid == 3 and labels.tolist() == [[True, False], [False, True], [False, False]]


def test_finished_infeasible_and_small_rings_receive_no_completion():
    # The gem-dimethyl ring has a tetra-substituted carbon. The tertiary
    # carbinolamine cannot become a carbonyl without losing another bond.
    examples = ["NC(=O)c1ccccc1", "C1CC1", "C1CCC1", "C1CCCC1", "CC1(C)CCCCC1",
                "CC(C)(O)N", "COC(N)C", "OCCN"]
    decoded = logits(examples)
    loss, observed = completion_loss(decoded, SPEC, need_ring=True, need_amide=True)
    assert observed == {"completion_aromatic_rows": 0, "completion_amide_rows": 0}
    assert loss.item() == 0 and not torch.autograd.grad(loss, decoded.bonds)[0].count_nonzero()


def test_each_completion_switch_only_guides_the_requesting_motif():
    # The six-member *unsubstituted* saturated carbon ring is an eligible target.
    decoded = logits(["C1CCCCC1", "NC(O)C"])
    for ring, amide in [(False, False), (True, False), (False, True), (True, True)]:
        loss, stats = completion_loss(decoded, SPEC, need_ring=ring, need_amide=amide)
        assert (stats["completion_aromatic_rows"], stats["completion_amide_rows"]) == (int(ring), int(amide))
        grad = torch.autograd.grad(loss, decoded.bonds)[0]
        assert (grad.abs().sum((1, 2, 3)) > 0).tolist() == [ring, amide]
def test_fixed_panel_keeps_failed_draws_in_motif_yields_and_counts_each_ring_size_once():
    from latentmol.training.occupancy_monitor import _motif_panel

    result = _motif_panel(["c1ccccc1-c1ccccc1", "CC(=O)N", "C1CCCCCCC1", None, "invalid-smiles"])
    assert result == dict(attempts=5, valid=3, aromatic=1, amide=1, smiles_roundtrip=1,
                          any_ring_8_or_larger=1, molecules_by_ring_size={6: 1, 8: 1})


def test_kekule_target_has_valid_intermediates_and_the_same_aromatic_endpoint():
    decoded = logits(["C1CCCCC1", "N1CCCCC1"])
    graphs = decoded.harden(SPEC)
    for atoms, charges, bonds in zip(graphs.atoms.numpy(), graphs.charges.numpy(), graphs.bonds.numpy()):
        endpoints = []
        for encoding in ("aromatic", "kekule"):
            ring, _ = _completion_targets(atoms, charges, bonds, SPEC, True, False, encoding)
            assert len(ring) == 6
            changed = bonds.copy()
            for i, j, label in ring:
                if encoding == "kekule":
                    assert SPEC.bond_names[label] in ("SINGLE", "DOUBLE")
                changed[i, j] = changed[j, i] = label
                if encoding == "kekule":
                    assert inspect_graph(atoms, charges, changed, SPEC).smiles
            endpoints.append(inspect_graph(atoms, charges, changed, SPEC).smiles)
        assert endpoints[0] and endpoints[0] == endpoints[1]
    loss, stats = completion_loss(decoded, SPEC, need_ring=True, need_amide=False, ring_encoding="kekule")
    gradient, = torch.autograd.grad(loss, (decoded.bonds,))
    assert stats["completion_aromatic_rows"] == 2
    assert torch.isfinite(gradient).all() and gradient[..., SPEC.bond_names.index("DOUBLE")].min() < 0


def test_sampled_smiles_roundtrip_failure_stays_in_denominator(monkeypatch):
    from rdkit import Chem

    hard = logits(["CC", "CCC"]).harden(SPEC)
    parse = Chem.MolFromSmiles
    monkeypatch.setattr(Chem, "MolFromSmiles", lambda value: None if value == "CC" else parse(value))
    failures = dict(smiles_roundtrip=0)
    labels, valid = _motif_labels(hard, SPEC, failures=failures)
    assert labels.shape == (2, 2) and not labels.any() and valid == 1
    assert failures["smiles_roundtrip"] == 1


def test_batch_pattern_loss_uses_the_observed_motif_deficit():
    decoded = logits(["C1CCCCC1", "C1CCCCC1", "CC(O)N", "CCOC"])
    loss, stats = functional_group_loss(decoded, SPEC, torch.Generator().manual_seed(81),
        aromatic_floor=.20, amide_floor=.20, completion_weight=1., completion_mode="pattern",
        amide_relative_weight=.10)
    assert stats["aromatic_hits"] == stats["amide_hits"] == 0
    assert stats["pattern_aromatic_candidates"] == 2 and stats["pattern_aromatic_selected"] == 1
    assert stats["pattern_amide_selected"] == 1 and stats["pattern_valid_rows"] == 4
    assert stats["completion_pattern_loss"] > 0
    grad, = torch.autograd.grad(loss, (decoded.bonds,))
    assert torch.isfinite(grad).all() and grad.count_nonzero()
