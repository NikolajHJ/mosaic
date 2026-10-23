"""Allocated CPU/CUDA contracts for the explicit weighted chemistry surrogate.

These fixtures verify arithmetic and provenance, not generation efficacy. The
mandatory CUDA marker makes a missing GPU a failure in the native allocation.
"""
import os

import pytest
import torch
from torch.nn import functional as F

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.chemistry_classification import VERSION as CLASSIFICATION_VERSION
from latentmol.chemistry_target_weighting import (VERSION, validate_chemistry_target_weighting,
    fixed_hard_target_ce, fixed_hard_target_ce_terms)
from latentmol.chemistry_valid_self_guidance import prepare_valid_self_guidance, valid_prior_self_nll
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob_terms


POLICY = dict(version=VERSION, none_bond_weight=.1)
SELF_POLICY = dict(version="valid-prior-self-nll-v1", weight=.1)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES") == "1":
            pytest.fail("Mandatory weighted-chemistry CUDA fixture has no visible device")
        pytest.skip("CUDA fixture runs in its allocated native stage")
    return request.param


def _fields(value):
    return value.atoms, value.charges, value.bonds


def _select(value, rows):
    return type(value)(*(part[list(rows)] for part in _fields(value)))


def _clone(decoded, device=None, dtype=None):
    return DecoderOutput(*(part.detach().to(device=device or part.device, dtype=dtype or part.dtype)
                           .clone().requires_grad_() for part in _fields(decoded)))


def _case(device="cpu", dtype=torch.float64, nodes=4):
    spec = GraphSpec(max_nodes=nodes)
    a = torch.zeros(3, nodes, dtype=torch.long, device=device)
    q = torch.full_like(a, spec.formal_charges.index(0))
    b = torch.zeros(3, nodes, nodes, dtype=torch.long, device=device)
    a[0, :3] = torch.tensor([1, 2, 3], device=device)
    q[0, 1] = spec.formal_charges.index(1)
    b[0, 0, 1] = b[0, 1, 0] = 1
    b[0, 1, 2] = b[0, 2, 1] = 2
    # A one-edge molecule with noncontiguous occupancy tests padding masks.
    a[1, 1] = 1; a[1, nodes - 1] = 3
    q[1, 1] = spec.formal_charges.index(-1)
    b[1, 1, nodes - 1] = b[1, nodes - 1, 1] = 1
    shapes = [(3, nodes, len(spec.atomic_numbers)), (3, nodes, len(spec.formal_charges)),
              (3, nodes, nodes, len(spec.bond_names))]
    parts = []
    for index, shape in enumerate(shapes):
        number = 1
        for extent in shape:
            number *= extent
        values = ((torch.arange(number, dtype=dtype, device=device) + index * 3)
                  .remainder(19 + index * 2).reshape(shape) / (7 + index) - .8)
        parts.append(values.requires_grad_())
    return spec, DecoderOutput(*parts), HardGraphBatch(a, q, b)


def _manual(decoded, target, spec):
    """Independent row/slot/pair loop; no shared loss primitive or vector masks."""
    rows = []
    for row in range(len(target.atoms)):
        atom = decoded.atoms[row].log_softmax(-1)
        charge = decoded.charges[row].log_softmax(-1)
        # Deliberately form symmetric pairs individually, rather than using the
        # DecoderOutput property or the implementation's pair-index gather.
        total = decoded.atoms[row].reshape(-1)[0] * 0
        denominator = float(spec.max_nodes)
        for i in range(spec.max_nodes):
            total = total - atom[i, target.atoms[row, i]]
            if int(target.atoms[row, i]) != 0:
                total = total - charge[i, target.charges[row, i]]
                denominator += 1.
                for j in range(i + 1, spec.max_nodes):
                    if int(target.atoms[row, j]) == 0:
                        continue
                    label = int(target.bonds[row, i, j])
                    weight = .1 if label == 0 else 1.
                    pair = ((decoded.bonds[row, i, j] + decoded.bonds[row, j, i]) / 2).log_softmax(-1)
                    total = total - weight * pair[label]
                    denominator += weight
        rows.append(total / denominator)
    return torch.stack(rows)


def test_policy_is_closed_metadata_only_and_absent_preserves_historical_mode():
    assert validate_chemistry_target_weighting(None) is None
    value = validate_chemistry_target_weighting(POLICY)
    value["none_bond_weight"] = .2
    assert POLICY["none_bond_weight"] == .1


@pytest.mark.parametrize("value", [False, {}, dict(version="unknown", none_bond_weight=.1),
    dict(version=VERSION, none_bond_weight=True), dict(version=VERSION, none_bond_weight=0),
    dict(version=VERSION, none_bond_weight=1), dict(version=VERSION, none_bond_weight=.2),
    dict(version=VERSION, none_bond_weight="0.1"), dict(version=VERSION, none_bond_weight=float("nan")),
    dict(version=VERSION, none_bond_weight=float("inf")), dict(POLICY, extra=True)])
def test_unknown_or_implicit_weighting_policies_are_rejected(value):
    with pytest.raises(ValueError):
        validate_chemistry_target_weighting(value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonuniform_loss_and_all_head_gradients_match_independent_loop(device, dtype):
    spec, decoded, target = _case(device, dtype)
    expected = _manual(decoded, target, spec)
    actual = fixed_hard_target_ce(decoded, target, spec, POLICY)
    tolerance = dict(rtol=2e-6, atol=2e-8) if dtype == torch.float32 else dict(rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(actual, expected, **tolerance)
    left = torch.autograd.grad(actual.mean(), _fields(decoded), retain_graph=True)
    right = torch.autograd.grad(expected.mean(), _fields(decoded))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, **tolerance)
    # All padding atom labels remain supervised; padded charge/edge logits do
    # not contribute. An undirected edge sends equal gradients to both logits.
    assert torch.count_nonzero(left[0][0, 3]) > 0
    assert torch.count_nonzero(left[1][0, 3]) == 0
    assert torch.count_nonzero(left[2][0, 0, 3]) == 0
    assert torch.count_nonzero(left[2].diagonal(dim1=1, dim2=2)) == 0
    torch.testing.assert_close(left[2], left[2].transpose(1, 2), rtol=0, atol=0)


def test_native50_slot_denominator_and_one_edge_empty_cases(device):
    spec, decoded, target = _case(device, torch.float64, nodes=50)
    result = fixed_hard_target_ce_terms(decoded, target, spec, POLICY)
    torch.testing.assert_close(result["occupied_atoms"], decoded.atoms.new_tensor([3, 2, 0]), rtol=0, atol=0)
    torch.testing.assert_close(result["present_pairs"], decoded.atoms.new_tensor([2, 1, 0]), rtol=0, atol=0)
    torch.testing.assert_close(result["none_pairs"], decoded.atoms.new_tensor([1, 0, 0]), rtol=0, atol=0)
    torch.testing.assert_close(result["denominator"], decoded.atoms.new_tensor([55.1, 53., 50.]), rtol=0, atol=0)
    torch.testing.assert_close(sum(result[k] for k in ("atoms", "charges", "present_bonds", "none_bonds")),
                               result["total"], rtol=0, atol=0)
    assert result["none_bonds"][1:].tolist() == [0., 0.]
    assert result["charges"][2].item() == result["present_bonds"][2].item() == 0.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("component,field,coordinate", [("total", "bonds", (0, 0, 1, 1)),
    ("total", "atoms", (0, 3, 0)), ("total", "charges", (0, 1, 3)),
    ("present_bonds", "bonds", (0, 0, 1, 1)), ("none_bonds", "bonds", (0, 0, 2, 0))])
def test_total_and_present_none_components_match_central_difference(device, dtype, component, field, coordinate):
    spec, decoded, target = _case(device, dtype)
    value = fixed_hard_target_ce_terms(decoded, target, spec, POLICY)[component].mean()
    gradient = torch.autograd.grad(value, getattr(decoded, field))[0][coordinate]
    epsilon = 2e-3 if dtype == torch.float32 else 1e-5
    values = []
    for sign in (1., -1.):
        changed = _clone(decoded)
        with torch.no_grad():
            getattr(changed, field)[coordinate] += sign * epsilon
        values.append(float(fixed_hard_target_ce_terms(changed, target, spec, POLICY)[component].mean()))
    numeric = (values[0] - values[1]) / (2 * epsilon)
    assert numeric == pytest.approx(float(gradient), rel=5e-3 if dtype == torch.float32 else 1e-7,
                                   abs=3e-6 if dtype == torch.float32 else 1e-10)


@pytest.mark.parametrize("defect", ["asymmetric", "diagonal", "padding_bond", "padding_charge", "float_labels"])
def test_graph_support_rejections_are_not_hidden_by_weighting(device, defect):
    spec, decoded, target = _case(device)
    if defect == "asymmetric":
        target.bonds[0, 0, 1] = 3
    elif defect == "diagonal":
        target.bonds[0, 1, 1] = 1
    elif defect == "padding_bond":
        target.bonds[2, 0, 1] = target.bonds[2, 1, 0] = 1
    elif defect == "padding_charge":
        target.charges[2, 0] = 0
    else:
        target.atoms = target.atoms.float()
    with pytest.raises(ValueError):
        fixed_hard_target_ce(decoded, target, spec, POLICY)


@pytest.mark.parametrize("defect", ["policy_none", "nonfinite", "wrong_capacity", "mismatched_dtype"])
def test_primitive_preserves_explicit_policy_and_shared_logit_contract(device, defect):
    spec, decoded, target = _case(device)
    policy = POLICY
    if defect == "policy_none":
        policy = None
    elif defect == "nonfinite":
        with torch.no_grad():
            decoded.atoms[0, 0, 0] = float("nan")
    elif defect == "wrong_capacity":
        decoded.atoms = decoded.atoms[:, :-1]
    else:
        decoded.charges = decoded.charges.float()
    with pytest.raises(ValueError):
        fixed_hard_target_ce(decoded, target, spec, policy)


def test_microbatch_means_and_reordered_rows_preserve_pergraph_values_and_gradients(device):
    spec, decoded, target = _case(device)
    complete = fixed_hard_target_ce(decoded, target, spec, POLICY)
    pieces = [fixed_hard_target_ce(_select(decoded, indices), _select(target, indices), spec, POLICY).mean()
              * len(indices) / 3 for indices in ((0, 1), (2,))]
    torch.testing.assert_close(sum(pieces), complete.mean(), rtol=1e-13, atol=1e-14)
    left = torch.autograd.grad(complete.mean(), _fields(decoded), retain_graph=True)
    right = torch.autograd.grad(sum(pieces), _fields(decoded))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-14)
    order = [2, 0, 1]
    torch.testing.assert_close(fixed_hard_target_ce(_select(decoded, order), _select(target, order), spec, POLICY),
                               complete[order], rtol=0, atol=0)


def _native_targets():
    """Real teacher witnesses: CCC, invalid O#CC and an empty failed attempt."""
    spec, decoded, graph = _case()
    graph.atoms.zero_(); graph.charges.fill_(spec.formal_charges.index(0)); graph.bonds.zero_()
    graph.atoms[:2, :3] = 1
    graph.atoms[1, 0] = spec.atomic_numbers.index(8)
    for row in (0, 1):
        graph.bonds[row, 0, 1] = graph.bonds[row, 1, 0] = 1 if row == 0 else 3
        graph.bonds[row, 1, 2] = graph.bonds[row, 2, 1] = 1
    raw = [value.detach() + 6 * F.one_hot(labels, value.shape[-1])
           for value, labels in zip(_fields(decoded), _fields(graph))]
    decoded = DecoderOutput(*(value.requires_grad_() for value in raw))
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=32),
                                    classification_version=CLASSIFICATION_VERSION) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=[70, 71, 72], update=5)
    assert base.targets[0] is None and base.targets[1] is not None and base.targets[2] is None
    prepared = prepare_valid_self_guidance(decoded, base, spec)
    assert prepared.eligible_rows == (0,)
    return spec, decoded, base, prepared


def test_repair_and_self_use_same_primitive_and_all_attempt_denominator(device):
    spec, original, base, prepared = _native_targets()
    decoded = _clone(original, device)
    repair = repair_loss(decoded, base, spec, target_weighting=POLICY)
    self_loss = valid_prior_self_nll(decoded, prepared, spec, SELF_POLICY, target_weighting=POLICY)
    accepted = base.targets[1].graph.to(device)
    expected_repair = fixed_hard_target_ce(_select(decoded, [1]), accepted, spec, POLICY).sum() / 3
    expected_self = fixed_hard_target_ce(_select(decoded, [0]), _select(prepared.graph.to(device), [0]), spec, POLICY).sum() / 3
    torch.testing.assert_close(repair, expected_repair, rtol=0, atol=0)
    torch.testing.assert_close(self_loss["loss"], expected_self, rtol=0, atol=0)
    torch.testing.assert_close(self_loss["weighted_loss"], .1 * expected_self, rtol=0, atol=0)
    assert self_loss["eligible_rows"] == (0,) and self_loss["receipt"]["accepted_repair_witnesses_created"] == 0
    assert self_loss["receipt"]["target_weighting"] == POLICY
    assert "not exact graph NLL" in self_loss["receipt"]["objective"]
    actual = torch.autograd.grad(repair + self_loss["loss"], _fields(decoded), retain_graph=True)
    expected = torch.autograd.grad(expected_repair + expected_self, _fields(decoded))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.count_nonzero(a[2]) == 0
    complete_repair = repair_loss(decoded, base, spec, target_weighting=POLICY)
    complete_self = valid_prior_self_nll(decoded, prepared, spec, SELF_POLICY, target_weighting=POLICY)["loss"]
    pieces = []
    for indices in ((0, 1), (2,)):
        one = _select(decoded, indices)
        one_repair = repair_loss(one, base.subset(indices), spec, target_weighting=POLICY)
        one_self = valid_prior_self_nll(one, prepared.subset(indices), spec, SELF_POLICY, target_weighting=POLICY)["loss"]
        pieces.append((one_repair + one_self) * len(indices) / 3)
    torch.testing.assert_close(sum(pieces), complete_repair + complete_self, rtol=1e-13, atol=1e-14)


def test_explicit_none_retains_bitwise_historical_losses_gradients_and_receipt(device):
    spec, original, base, prepared = _native_targets()
    results = []
    for kwargs in ({}, {"target_weighting": None}):
        decoded = _clone(original, device)
        repair = repair_loss(decoded, base, spec, **kwargs)
        self_result = valid_prior_self_nll(decoded, prepared, spec, SELF_POLICY, **kwargs)
        gradients = torch.autograd.grad(repair + self_result["weighted_loss"], _fields(decoded))
        results.append((repair.detach(), self_result["loss"].detach(), self_result["receipt"], gradients))
    assert results[0][2] == results[1][2] == prepared.receipt
    for left, right in zip(results[0][:2], results[1][:2]):
        assert torch.equal(left, right)
    for left, right in zip(results[0][3], results[1][3]):
        assert torch.equal(left, right)
    # Historical oracle remains the exact unweighted graph log probability.
    decoded = _clone(original, device)
    hard = base.targets[1].graph.to(device)
    terms = hard_graph_log_prob_terms(_select(decoded, [1]), hard, spec)
    old = -(terms["atoms"] + terms["charges"] + terms["bonds"]).sum() / base.targets[1].decisions / 3
    torch.testing.assert_close(repair_loss(decoded, base, spec), old, rtol=0, atol=0)


def test_invalid_policy_or_tampered_provenance_never_becomes_zero_guidance(device):
    spec, original, base, prepared = _native_targets()
    decoded = _clone(original, device)
    with pytest.raises(ValueError):
        repair_loss(_select(decoded, [2]), base.subset([2]), spec, target_weighting={})
    with pytest.raises(ValueError):
        valid_prior_self_nll(_select(decoded, [2]), prepared.subset([2]), spec, SELF_POLICY, target_weighting={})
    base.receipt["elapsed_seconds"] += 1.
    with pytest.raises(ValueError, match="identity"):
        valid_prior_self_nll(decoded, prepared, spec, SELF_POLICY, target_weighting=POLICY)
