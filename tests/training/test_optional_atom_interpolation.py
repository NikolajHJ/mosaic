import math

import pytest
import torch
from rdkit import Chem

from latentmol.chemistry import encode_smiles
from latentmol.evaluation.molecular_pairs import match_optional_atom, permute_graph, relabel_categories
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.training.losses import LossConfig
from latentmol.training.optional_atom_interpolation import optional_atom_interpolation_loss, optional_atom_targets


SPEC = GraphSpec(max_nodes=10)
LOSS = LossConfig()


def aligned_pair(small, large):
    encoded = [encode_smiles(smiles, SPEC) for smiles in (small, large)]
    molecules = [Chem.MolFromSmiles(row["smiles"]) for row in encoded]
    order = match_optional_atom(*molecules)
    assert order is not None
    a, b = [HardGraphBatch(*(torch.as_tensor(row[k]).long()[None]
                             for k in ("atom_types", "charges", "bond_types"))) for row in encoded]
    aligned = permute_graph(b, list(order))
    frame = torch.tensor([[*order, *range(len(order), SPEC.max_nodes)]])
    return a, aligned, frame


@pytest.mark.parametrize(("small", "large"), [
    ("c1ccccc1", "Fc1ccccc1"),
    ("CC", "CCC"),
    ("CC", "CC[O-]"),
])
def test_exact_terminal_addition_matches_in_permuted_input(small, large):
    a, b = Chem.MolFromSmiles(small), Chem.MolFromSmiles(large)
    b = Chem.RenumberAtoms(b, list(reversed(range(b.GetNumAtoms()))))
    order = match_optional_atom(a, b)
    assert order is not None
    assert b.GetAtomWithIdx(order[-1]).GetDegree() == 1
    assert sorted(order) == list(range(b.GetNumAtoms()))


@pytest.mark.parametrize(("small", "large"), [
    ("CCC", "C1CCC1"),  # Addition in a ring, not terminal.
    ("C=C", "CCC"),  # Core bond changed.
    ("CCO", "CCCC"),  # Core element changed.
    ("CC[NH3+]", "CCCN"),  # Core formal charge changed.
    ("CC", "CCCC"),  # Two extra atoms.
    ("CC", "CO"),  # Same size, handled by the existing matcher.
])
def test_nonlocal_or_multiple_changes_are_excluded(small, large):
    assert match_optional_atom(Chem.MolFromSmiles(small), Chem.MolFromSmiles(large)) is None


def _logits(target):
    return target.clamp_min(1e-12).log()


def test_charge_and_bond_marginals_track_the_one_atom_occupancy():
    a, b, _ = aligned_pair("CC", "CC[O-]")
    t = torch.tensor([0., .25, .5, .75, 1.])
    atoms, charges, bonds = optional_atom_targets(a, b, t.tolist(), SPEC, LOSS)
    optional = 2
    neighbor = int((b.bonds[0, optional] != 0).nonzero().item())
    torch.testing.assert_close(1 - atoms[:, optional, 0], t)
    torch.testing.assert_close(charges[:, optional].sum(-1), t)
    torch.testing.assert_close(bonds[:, neighbor, optional].sum(-1), t)
    torch.testing.assert_close(bonds[:, neighbor, optional, 1], t)

    # Give the decoder the conditional bond and charge distributions from the
    # weighted targets. Its normal relaxed graph path applies the gate once.
    bond_logits = _logits(bonds + bonds.transpose(1, 2))
    decoded = DecoderOutput(_logits(atoms), _logits(charges), bond_logits)
    graph = decoded.probabilities()
    torch.testing.assert_close(graph.expected_size, 2 + t, atol=1e-6, rtol=0)
    torch.testing.assert_close(graph.bonds[:, neighbor, optional, 1], t, atol=1e-6, rtol=0)
    torch.testing.assert_close(graph.expected_charge(SPEC)[:, optional], -t, atol=1e-6, rtol=0)


def test_optional_occupancy_and_incident_bond_receive_useful_gradients():
    a, b, _ = aligned_pair("c1ccccc1", "Fc1ccccc1")
    atoms = torch.zeros((3, SPEC.max_nodes, len(SPEC.atomic_numbers)))
    atoms[..., 0] = math.log(len(SPEC.atomic_numbers) - 1)
    charges = torch.zeros((3, SPEC.max_nodes, len(SPEC.formal_charges)))
    bonds = torch.zeros((3, SPEC.max_nodes, SPEC.max_nodes, len(SPEC.bond_names)))
    decoded = DecoderOutput(atoms.requires_grad_(), charges.requires_grad_(), bonds.requires_grad_())
    loss = optional_atom_interpolation_loss(decoded, a, b, [.25, .5, .75], SPEC, LOSS)
    loss.backward()
    for value in (atoms, charges, bonds):
        assert torch.isfinite(value.grad).all()
    slot = 6
    neighbor = int((b.bonds[0, slot] != 0).nonzero().item())
    assert atoms.grad[0, slot, 0] < 0 < atoms.grad[2, slot, 0]
    assert bonds.grad[0, neighbor, slot, 1] < 0 < bonds.grad[0, neighbor, slot, 0]
    assert charges.grad[0, slot, SPEC.formal_charges.index(0)] < 0


def test_loss_accepts_the_larger_canonical_frame():
    a, b, frame = aligned_pair("c1ccccc1", "Fc1ccccc1")
    fractions = [.25, .5, .75]
    atoms, charges, bonds = optional_atom_targets(a, b, fractions, SPEC, LOSS)
    aligned = (_logits(atoms), _logits(charges), _logits(bonds + bonds.transpose(1, 2)))
    canonical = DecoderOutput(*relabel_categories(*aligned, frame.argsort(dim=-1).expand(3, -1)))
    reference = optional_atom_interpolation_loss(DecoderOutput(*aligned), a, b, fractions, SPEC, LOSS)
    adapted = optional_atom_interpolation_loss(canonical, a, b, fractions, SPEC, LOSS, orders=frame)
    torch.testing.assert_close(adapted, reference)
    assert optional_atom_interpolation_loss(canonical, a, b, fractions, SPEC, LOSS) > adapted


def test_target_builder_rejects_a_changed_shared_core():
    a, b, _ = aligned_pair("CC", "CC[O-]")
    b.bonds[:, 0, 1] = b.bonds[:, 1, 0] = 2
    with pytest.raises(ValueError, match="identical core"):
        optional_atom_targets(a, b, [.5], SPEC, LOSS)
