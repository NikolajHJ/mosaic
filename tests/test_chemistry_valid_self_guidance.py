"""Allocated CPU/CUDA engineering fixtures; no generation-efficacy claim."""
from copy import deepcopy
from dataclasses import replace
import os

import pytest
import torch
from torch.nn import functional as F

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.chemistry_classification import VERSION as CLASSIFICATION_VERSION
from latentmol.chemistry_valid_self_guidance import (VERSION, prepare_valid_self_guidance,
    valid_prior_self_nll, validate_valid_prior_self_guidance)
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec
from test_atom_preserving_chemistry import _decoded


POLICY = {"version": VERSION, "weight": .01}


def _prepared(kinds=("valid", "invalid", "empty")):
    spec = GraphSpec(max_nodes=3)
    decoded = _decoded(spec, kinds)
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=32),
                                    classification_version=CLASSIFICATION_VERSION) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=list(range(40, 40 + len(kinds))), update=7)
    return spec, decoded, base, prepare_valid_self_guidance(decoded, base, spec)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES") == "1":
            pytest.fail("Mandatory allocated self-NLL CUDA fixture has no visible device")
        pytest.skip("CUDA-only allocated engineering fixture")
    return request.param


def _live(decoded, device):
    return DecoderOutput(*(value.detach().to(device).clone().requires_grad_()
                           for value in (decoded.atoms, decoded.charges, decoded.bonds)))


def test_policy_delegates_the_shared_metadata_schema_and_does_not_mutate():
    assert validate_valid_prior_self_guidance(None) is None
    parsed = validate_valid_prior_self_guidance(POLICY)
    parsed["weight"] = 0
    assert POLICY["weight"] == .01
    for value in (True, -.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            validate_valid_prior_self_guidance(dict(version=VERSION, weight=value))


def test_only_native_strict_already_valid_rows_are_self_targets_not_repairs():
    spec, _, base, prepared = _prepared()
    assert prepared.candidate_ids == (40, 41, 42)
    assert prepared.eligible_rows == (0,)
    assert base.targets[0] is None and base.targets[1] is not None
    assert prepared.receipt["scheduled_prior_attempts"] == 3
    assert prepared.receipt["strict_valid_self_targets"] == 1
    assert prepared.receipt["original_accepted_repair_targets"] == 1
    assert prepared.receipt["accepted_repair_witnesses_created"] == 0
    assert all(not value.requires_grad and value.device.type == "cpu"
               for value in (prepared.graph.atoms, prepared.graph.charges, prepared.graph.bonds))
    public = prepared.graph
    public.atoms.zero_()
    assert int(prepared.graph.atoms[0, 0]) == spec.atomic_numbers.index(6)


def test_live_scalar_matches_independent_masked_full_graph_nll_and_all_attempt_mean(device):
    spec, decoded, _, prepared = _prepared()
    live = _live(decoded, device)
    actual = valid_prior_self_nll(live, prepared, spec, POLICY)
    target = prepared.graph.to(device)
    active = target.atoms[0] != 0
    atom = F.log_softmax(live.atoms[0], -1).gather(-1, target.atoms[0, :, None]).sum()
    charge = F.log_softmax(live.charges[0], -1).gather(-1, target.charges[0, :, None]).squeeze(-1)[active].sum()
    bonds = F.log_softmax(live.symmetric_bond_logits[0], -1)
    left, right = torch.triu_indices(spec.max_nodes, spec.max_nodes, offset=1, device=device)
    present = active[left] & active[right]
    bond = bonds[left[present], right[present], target.bonds[0, left[present], right[present]]].sum()
    k = int(active.sum())
    direct = -(atom + charge + bond) / (spec.max_nodes + k + k * (k - 1) // 2) / 3
    torch.testing.assert_close(actual["loss"], direct, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(actual["weighted_loss"], .01 * direct, rtol=0, atol=0)
    assert actual["eligible_rows"] == (0,) and actual["valid_mask"].tolist() == [True, False, False]
    actual_grad = torch.autograd.grad(actual["loss"], (live.atoms, live.charges, live.bonds), retain_graph=True)
    direct_grad = torch.autograd.grad(direct, (live.atoms, live.charges, live.bonds))
    for left_grad, right_grad in zip(actual_grad, direct_grad):
        torch.testing.assert_close(left_grad, right_grad, rtol=1e-13, atol=1e-14)
        assert torch.count_nonzero(left_grad[1:]) == 0
    assert torch.count_nonzero(actual_grad[0][0, 2]) > 0  # The NO_ATOM decision is supervised.
    assert torch.count_nonzero(actual_grad[1][0, 2]) == 0  # Absent charge is not.
    assert torch.count_nonzero(actual_grad[2][0, 0, 0]) == 0
    assert torch.count_nonzero(actual_grad[2][0, 0, 2]) == 0
    torch.testing.assert_close(actual_grad[2][0, 0, 1], actual_grad[2][0, 1, 0], rtol=0, atol=0)


def test_confidence_gradient_is_nonzero_for_already_valid_and_matches_finite_difference(device):
    spec, decoded, _, prepared = _prepared(("valid",))
    live = _live(decoded, device)
    value = valid_prior_self_nll(live, prepared, spec, POLICY)["weighted_loss"]
    gradient = torch.autograd.grad(value, live.bonds)[0]
    single = spec.bond_names.index("SINGLE")
    assert float(gradient[0, 0, 1, single]) < 0
    epsilon = 1e-5
    values = []
    for sign in (1., -1.):
        changed = _live(decoded, device)
        with torch.no_grad():
            changed.bonds[0, 0, 1, single] += sign * epsilon
        values.append(float(valid_prior_self_nll(changed, prepared, spec, POLICY)["weighted_loss"]))
    numeric = (values[0] - values[1]) / (2 * epsilon)
    assert numeric == pytest.approx(float(gradient[0, 0, 1, single]), rel=1e-7, abs=1e-11)


def test_no_eligible_rows_and_zero_weight_produce_explicit_zero_gradients(device):
    spec, decoded, _, prepared = _prepared(("invalid", "empty"))
    live = _live(decoded, device)
    result = valid_prior_self_nll(live, prepared, spec, POLICY)
    assert result["loss"].item() == 0 and result["eligible_rows"] == ()
    gradients = torch.autograd.grad(result["loss"], (live.atoms, live.charges, live.bonds))
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
    spec, decoded, _, prepared = _prepared(("valid",))
    live = _live(decoded, device)
    result = valid_prior_self_nll(live, prepared, spec, dict(version=VERSION, weight=0.))
    gradients = torch.autograd.grad(result["weighted_loss"], (live.atoms, live.charges, live.bonds))
    assert result["loss"].item() > 0 and result["weighted_loss"].item() == 0
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)


def test_subset_preserves_full_receipt_integrity_and_all_attempt_weighting(device):
    spec, decoded, base, prepared = _prepared()
    live = _live(decoded, device)
    complete = valid_prior_self_nll(live, prepared, spec, POLICY)["loss"]
    pieces = []
    for indices in ((0,), (1, 2)):
        one = DecoderOutput(*(value[list(indices)] for value in (live.atoms, live.charges, live.bonds)))
        pieces.append(valid_prior_self_nll(one, prepared.subset(indices), spec, POLICY)["loss"] * len(indices) / 3)
    torch.testing.assert_close(sum(pieces), complete, rtol=1e-13, atol=1e-14)
    left = torch.autograd.grad(complete, (live.atoms, live.charges, live.bonds), retain_graph=True)
    right = torch.autograd.grad(sum(pieces), (live.atoms, live.charges, live.bonds))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-14)
    assert prepared.subset([2, 0]).candidate_ids == (42, 40)
    assert prepared.subset([2, 0]).eligible_rows == (1,)
    with pytest.raises(ValueError, match="complete"):
        prepare_valid_self_guidance(DecoderOutput(*(value[:1] for value in (decoded.atoms, decoded.charges, decoded.bonds))),
                                    base.subset([0]), spec)


@pytest.mark.parametrize("part", ["receipt", "graph", "point", "mask"])
def test_post_preparation_tampering_is_rejected(part):
    spec, decoded, base, prepared = _prepared()
    if part == "receipt":
        base.receipt["elapsed_seconds"] += 1.
    elif part == "graph":
        prepared._graph.atoms[0, 0] = 0
    elif part == "point":
        prepared = replace(prepared, _view=(2, 1, 0))
    else:
        prepared = replace(prepared, _eligible=(0, 1))
    with pytest.raises(ValueError, match="identity"):
        valid_prior_self_nll(decoded, prepared, spec, POLICY)


def test_semantic_receipt_ignores_timing_and_cache_but_private_identity_binds_them():
    spec, decoded, base, prepared = _prepared()
    changed = deepcopy(base)
    changed.receipt["elapsed_seconds"] += 1.
    changed.receipt["cache_hits"] += 1
    changed.receipt["attempts"][0]["cache_hit"] = True
    other = prepare_valid_self_guidance(decoded, changed, spec)
    assert other.identity != prepared.identity
    assert other.receipt == prepared.receipt


def test_missing_classification_or_changed_probability_snapshot_cannot_be_self_labels():
    spec, decoded, base, _ = _prepared()
    incomplete = deepcopy(base)
    del incomplete.receipt["attempts"][0]["chemical_classification"]
    with pytest.raises(ValueError, match="provenance"):
        prepare_valid_self_guidance(decoded, incomplete, spec)
    changed = _live(decoded, "cpu")
    with torch.no_grad():
        changed.atoms[0, 0, 1] += .1
    with pytest.raises(ValueError, match="provenance"):
        prepare_valid_self_guidance(changed, base, spec)


def test_live_logits_can_move_while_target_and_validity_selection_stay_detached():
    spec, decoded, _, prepared = _prepared(("valid",))
    live = _live(decoded, "cpu")
    with torch.no_grad():
        live.bonds[0, 0, 1, spec.bond_names.index("AROMATIC")] += 30.
    assert live.harden(spec).bonds[0, 0, 1].item() != prepared.graph.bonds[0, 0, 1].item()
    result = valid_prior_self_nll(live, prepared, spec, POLICY)
    assert result["eligible_rows"] == (0,)
    assert prepared.graph.bonds[0, 0, 1].item() == spec.bond_names.index("SINGLE")
    assert all(value.grad is None for value in (prepared._graph.atoms, prepared._graph.charges, prepared._graph.bonds))


@pytest.mark.parametrize("field,value", [("schema_roundtrip_valid", False), ("strict_teacher_valid", False),
    ("failure_kind", "representation"), ("representation_smiles", None)])
def test_an_allegedly_already_valid_row_requires_all_native_chemical_and_schema_checks(field, value):
    spec, decoded, base, _ = _prepared(("valid",))
    base.receipt["attempts"][0]["chemical_classification"][field] = value
    with pytest.raises(ValueError):
        prepare_valid_self_guidance(decoded, base, spec)


def test_cpu_source_and_live_schema_are_explicit(device):
    spec, decoded, base, prepared = _prepared(("valid",))
    if device == "cuda":
        with pytest.raises(ValueError, match="CPU"):
            prepare_valid_self_guidance(_live(decoded, device), base, spec)
    changed = GraphSpec(max_nodes=spec.max_nodes, atomic_numbers=(0, 6, 8, 7, 9, 15, 16, 17, 35, 53))
    with pytest.raises(ValueError, match="schema"):
        valid_prior_self_nll(_live(decoded, device), prepared, changed, POLICY)
