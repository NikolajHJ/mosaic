"""Small active-set boundary fixtures; these do not load native Boltz weights."""

import pytest
import torch
from torch.nn import functional as F

from latentmol.boltz.active_set import audit_start, bind_slots, select_slots
from latentmol.boltz.contracts import BOLTZ_COMMIT, PreparedComplex
from latentmol.boltz.provenance import context_digest
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def carbon_pair():
    """Two native ligand rows on four graph slots, with complete hard parity."""
    spec = GraphSpec(max_nodes=4)
    seed = HardGraphBatch(torch.tensor([[1, 1, 0, 0]]),
                          torch.tensor([[2, 2, 2, 2]]),
                          torch.tensor([[[0, 1, 0, 0], [1, 0, 0, 0],
                                         [0, 0, 0, 0], [0, 0, 0, 0]]]))
    bonds = torch.tensor([[[0, 2], [2, 0]]])
    pool = torch.eye(2)[None]
    features = {"ref_element": F.one_hot(torch.tensor([[6, 6]]), 128),
                "ref_charge": torch.zeros(1, 2),
                "token_pad_mask": torch.ones(1, 2),
                "atom_pad_mask": torch.ones(1, 2),
                "affinity_token_mask": torch.ones(1, 2),
                "mol_type": torch.full((1, 2), 3),
                "type_bonds": bonds, "token_bonds": (bonds > 0).float()[..., None],
                "atom_to_token": pool, "token_to_rep_atom": pool,
                "ref_pos": torch.zeros(1, 2, 3)}
    context = PreparedComplex(features, torch.zeros(1, 1, 2, 3),
                              torch.tensor([0, 1, -1, -1]),
                              torch.tensor([0, 1, -1, -1]), seed, spec,
                              {"boltz_commit": BOLTZ_COMMIT, "seed_smiles": "CC"})
    context.validate()
    return context


def logits(spec):
    atoms = torch.full((1, spec.max_nodes, len(spec.atomic_numbers)), -20.0)
    atoms[..., 1] = 0.0
    atoms[0, :, 0] = torch.tensor([-2.0, 2.0, -2.0, 2.0])
    return DecoderOutput(atoms.requires_grad_(),
                         torch.zeros(1, spec.max_nodes, len(spec.formal_charges),
                                     requires_grad=True),
                         torch.randn(1, spec.max_nodes, spec.max_nodes,
                                     len(spec.bond_names), requires_grad=True))


def test_selector_and_bound_context_preserve_native_rows_and_conditional_gradients():
    template = carbon_pair()
    decoded = logits(template.spec)
    chosen = select_slots(decoded, temperature=1.0, threshold=0.5)
    assert chosen.record()["active_slots"] == [0, 2]
    bound = bind_slots(template, chosen.active)
    assert bound.active.tolist() == [True, False, True, False]
    assert context_digest(bound) != context_digest(template)
    assert torch.equal(bound.features["ref_element"], template.features["ref_element"])
    graph = graph_from_decoder(decoded, template.spec, fixed_active=bound.active)
    assert torch.equal(graph.occupancy[0], bound.active.float())
    assert graph.bonds[0, 0, 2, 1:].sum() > 0
    # The old NO_ATOM logits do not directly feed the frozen Boltz input.
    scalar = (graph.atoms * torch.arange(len(template.spec.atomic_numbers))).sum()
    gradient, = torch.autograd.grad(scalar, decoded.atoms)
    assert torch.equal(gradient[..., 0], torch.zeros_like(gradient[..., 0]))


def test_selector_tie_is_absent_and_empty_is_only_allowed_for_diagnostics():
    spec = GraphSpec(max_nodes=1)
    atom_logits = torch.full((1, 1, len(spec.atomic_numbers)), -1e9)
    atom_logits[0, 0, 0] = 0
    atom_logits[0, 0, 1] = 0
    decoded = DecoderOutput(atom_logits, torch.zeros(1, 1, len(spec.formal_charges)),
                            torch.zeros(1, 1, 1, len(spec.bond_names)))
    with pytest.raises(ValueError, match="empty ligand"):
        select_slots(decoded, temperature=1.0, threshold=0.5)
    diagnostic = select_slots(decoded, temperature=1.0, threshold=0.5,
                              require_nonempty=False)
    assert diagnostic.count == 0


def _exact_seed_logits(seed, spec):
    """Concentrated categorical fixture that reconstructs each seed field exactly."""
    values = []
    for key, categories in (("atoms", len(spec.atomic_numbers)),
                            ("charges", len(spec.formal_charges)),
                            ("bonds", len(spec.bond_names))):
        values.append(F.one_hot(getattr(seed, key), categories).float() * 40 - 20)
    return DecoderOutput(*values)


def test_start_audit_accepts_exact_seed_and_records_both_identities():
    context = carbon_pair()
    selection, report = audit_start(_exact_seed_logits(context.seed, context.spec),
                                    context.seed, context.spec,
                                    temperature=1.0, threshold=0.5)
    assert report["passed"] and report["selected_mask_matches_seed"]
    assert report["ordinary_hard_fields_equal"] == {
        "atoms": True, "charges": True, "bonds": True}
    assert report["seed_sha256"] == report["ordinary_hard_sha256"]
    assert selection.record()["active_slots"] == [0, 1]


@pytest.mark.parametrize("changed_field", ["atoms", "charges", "bonds"])
def test_start_audit_rejects_changed_identity_even_with_matching_count(changed_field):
    context = carbon_pair()
    decoded = _exact_seed_logits(context.seed, context.spec)
    if changed_field == "bonds":
        for left, right in ((0, 1), (1, 0)):
            decoded.bonds[0, left, right] = -20
            decoded.bonds[0, left, right, 2] = 20
    else:
        values = getattr(decoded, changed_field)
        values[0, 0] = -20
        values[0, 0, 2 if changed_field == "atoms" else 3] = 20
    selection, report = audit_start(decoded, context.seed, context.spec,
                                    temperature=1.0, threshold=0.5)
    assert selection.count == 2
    assert report["selected_mask_matches_seed"]
    assert not report["passed"]
    assert not report["ordinary_hard_fields_equal"][changed_field]
    assert report["seed_sha256"] != report["ordinary_hard_sha256"]


def test_start_audit_rejects_threshold_mask_even_when_argmax_seed_is_exact():
    context = carbon_pair()
    decoded = _exact_seed_logits(context.seed, context.spec)
    # Padding is the most likely individual class, but the real elements jointly
    # carry more than half the mass. The threshold selects this additional slot.
    decoded.atoms[0, 2] = -20
    decoded.atoms[0, 2, :3] = torch.tensor([0.4, 0.3, 0.3]).log()
    selection, report = audit_start(decoded, context.seed, context.spec,
                                    temperature=1.0, threshold=0.5)
    assert selection.count == 3
    assert all(report["ordinary_hard_fields_equal"].values())
    assert not report["selected_mask_matches_seed"] and not report["passed"]


def test_start_audit_retains_empty_selection_as_a_failed_report():
    context = carbon_pair()
    decoded = _exact_seed_logits(context.seed, context.spec)
    decoded.atoms.fill_(-20)
    decoded.atoms[..., 0] = 20
    selection, report = audit_start(decoded, context.seed, context.spec,
                                    temperature=1.0, threshold=0.5)
    assert selection.count == 0
    assert report["selection"]["active_slots"] == []
    assert not report["passed"] and not report["selected_mask_matches_seed"]


def test_random_start_accepts_invalid_chemistry_without_repair_or_count_clamp():
    from latentmol.boltz.active_set import audit_random_start
    context = carbon_pair()
    decoded = _exact_seed_logits(context.seed, context.spec)
    decoded.bonds.fill_(-20)
    decoded.bonds[..., 0] = 20  # Two disconnected carbons: still a usable soft start.
    selected, report = audit_random_start(decoded, context.spec, temperature=1.0,
                                          threshold=0.5, available_counts=[2])
    assert selected.count == 2 and report["passed"]
    assert report["initial_hard_graph"]["smiles"] is None
    assert report["initial_hard_graph"]["failure"] == "disconnected"
    unsupported, failed = audit_random_start(decoded, context.spec, temperature=1.0,
                                              threshold=0.5, available_counts=[1, 3])
    assert unsupported.count == 2 and not failed["passed"]
    decoded.atoms.fill_(-20)
    decoded.atoms[..., 0] = 20
    empty, failed = audit_random_start(decoded, context.spec, temperature=1.0,
                                       threshold=0.5, available_counts=[1, 2, 3])
    assert empty.count == 0 and not failed["passed"]
