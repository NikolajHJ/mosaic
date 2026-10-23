"""Meaningful chemistry/gradient fixtures; execute numerical tests only on HPC."""
import copy
from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from latentmol.atom_preserving_chemistry import (
    AtomPreservingRepairTeacher, RepairTargetBatch, composition_statistics,
    hard_composition_statistics, population_js, population_js_cotangent,
    repair_loss, repair_policy, search_atom_preserving_repair,
)
from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.probability_repair import ProbabilityRepairConfig, measure_probability_edits
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def _decoded(spec, kinds=("invalid",)):
    batch, n = len(kinds), spec.max_nodes
    atom = torch.full((batch, n, len(spec.atomic_numbers)), -12.0, dtype=torch.float64)
    charge = torch.full((batch, n, len(spec.formal_charges)), -12.0, dtype=torch.float64)
    bond = torch.full((batch, n, n, len(spec.bond_names)), -12.0, dtype=torch.float64)
    atom[..., 0] = 0
    charge[..., spec.formal_charges.index(0)] = 0
    bond[..., 0] = 0
    for row, kind in enumerate(kinds):
        if kind == "empty":
            continue
        atom[row, :2, 0] = -12
        atom[row, 0, spec.atomic_numbers.index(8 if kind == "invalid" else 6)] = 0
        atom[row, 1, spec.atomic_numbers.index(6)] = 0
        if kind == "disconnected":
            continue
        # Give the valid oxygen DOUBLE witness greater support than SINGLE.
        bond[row, 0, 1, 0] = bond[row, 1, 0, 0] = -12
        bond[row, 0, 1, spec.bond_names.index("SINGLE")] = -4
        bond[row, 1, 0, spec.bond_names.index("SINGLE")] = -4
        bond[row, 0, 1, spec.bond_names.index("DOUBLE")] = -0.1
        bond[row, 1, 0, spec.bond_names.index("DOUBLE")] = -0.1
        label = spec.bond_names.index("TRIPLE" if kind == "invalid" else "SINGLE")
        bond[row, 0, 1, label] = bond[row, 1, 0, label] = 0
    return DecoderOutput(*(value.requires_grad_() for value in (atom, charge, bond)))


def _probabilities(decoded):
    return (decoded.atoms.softmax(-1), decoded.charges.softmax(-1),
            decoded.symmetric_bond_logits.softmax(-1))


def _teacher(spec, **kwargs):
    return AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=32,
                                                                    beam_width=2), **kwargs)


def test_bond_only_search_preserves_atom_charge_slots_and_prefers_probability_support():
    spec, output = GraphSpec(max_nodes=3), None
    output = _decoded(spec)
    hard = output.harden(spec)
    raw = tuple(value[0] for value in (hard.atoms, hard.charges, hard.bonds))
    report = search_atom_preserving_repair(raw, tuple(value[0] for value in _probabilities(output)), spec,
                                          ProbabilityRepairConfig(max_edits=1, checks_per_depth=32))
    assert report["status"] == "witness_found"
    witness = report["weighted_witness"]
    assert witness["labels"]["atoms"] == raw[0].tolist()
    assert witness["labels"]["charges"] == raw[1].tolist()
    assert witness["edit_counts"] == {"atom": 0, "charge": 0, "bond": 1}
    assert witness["labels"]["bonds"][0][1] == spec.bond_names.index("DOUBLE")
    assert report["policy"]["allowed_edits"] == ["bond"]
    p = _probabilities(output)[2][0, 0, 1]
    assert witness["weighted_cost"] == pytest.approx(1 + math.log(float(p[3] / p[2])))
    assert not report["distance_is_exact"]


def test_valid_expanded_valence_sulfur_is_supported_but_carbon_cannot_be_repaired_by_substitution():
    spec = GraphSpec(max_nodes=7)
    sulfur = encode_smiles("FS(F)(F)(F)(F)F", spec)
    valid = tuple(np.array(sulfur[name], dtype=np.int64) for name in ("atom_types", "charges", "bond_types"))
    assert inspect_graph(*valid, spec).smiles is not None
    raw = tuple(value.copy() for value in valid)
    sulfur_slot = int(np.flatnonzero(raw[0] == spec.atomic_numbers.index(16))[0])
    raw[0][sulfur_slot] = spec.atomic_numbers.index(6)
    p = (np.eye(len(spec.atomic_numbers))[raw[0]], np.eye(len(spec.formal_charges))[raw[1]],
         np.eye(len(spec.bond_names))[raw[2]])
    config = ProbabilityRepairConfig(max_edits=1, checks_per_depth=128)
    assert search_atom_preserving_repair(valid, p, spec, config)["status"] == "already_valid"
    report = search_atom_preserving_repair(raw, p, spec, config)
    assert report["status"] == "censored" and report["weighted_witness"] is None
    # The historical general search could replace this C with S in one edit;
    # the new bond-only policy must never take that chemically valid shortcut.
    assert raw[0][sulfur_slot] == spec.atomic_numbers.index(6)


def test_disconnected_graph_is_repaired_only_to_a_connected_molecule():
    spec, output = GraphSpec(max_nodes=3), None
    output = _decoded(spec, ("disconnected",))
    batch = _teacher(spec).build_targets(output, candidate_ids=[5], update=0)
    assert batch.receipt["accepted_targets"] == 1
    target = batch.targets[0]
    assert target is not None and inspect_graph(*(v[0] for v in (target.graph.atoms, target.graph.charges,
                                                               target.graph.bonds)), spec).smiles is not None
    assert target.graph.bonds[0, 0, 1] != 0


def test_sanitized_but_unrepresentable_raw_graph_is_not_silently_skipped_valid(monkeypatch):
    import latentmol.atom_preserving_chemistry as module
    def reject_roundtrip(*args):
        raise ValueError("synthetic schema-roundtrip rejection")
    monkeypatch.setattr(module, "encode_smiles", reject_roundtrip)
    spec = GraphSpec(max_nodes=3)
    batch = _teacher(spec).build_targets(_decoded(spec, ("valid",)), candidate_ids=[2], update=0)
    attempt = batch.receipt["attempts"][0]
    assert batch.targets == (None,) and attempt["status"] != "skipped_already_valid"
    assert attempt["search"]["raw_rdkit_valid"]
    assert not attempt["search"]["raw_valid"]
    assert "roundtrip rejection" in attempt["search"]["raw_roundtrip_error"]


def test_roundtrip_rejected_intermediate_can_still_lead_to_a_two_edit_witness(monkeypatch):
    import latentmol.atom_preserving_chemistry as module
    spec = GraphSpec(max_nodes=3)
    encoded = encode_smiles("CCC", spec)
    raw = tuple(np.array(encoded[name], dtype=np.int64) for name in ("atom_types", "charges", "bond_types"))
    raw[2][raw[2] == spec.bond_names.index("SINGLE")] = spec.bond_names.index("TRIPLE")
    probabilities = (np.eye(len(spec.atomic_numbers))[raw[0]], np.eye(len(spec.formal_charges))[raw[1]],
                     np.eye(len(spec.bond_names))[raw[2]])
    native_encode = module.encode_smiles
    def require_propane(smiles, graph_spec):
        if smiles != "CCC":
            raise ValueError("fixture rejects sanitized intermediate representation")
        return native_encode(smiles, graph_spec)
    monkeypatch.setattr(module, "encode_smiles", require_propane)
    result = search_atom_preserving_repair(raw, probabilities, spec,
                                          ProbabilityRepairConfig(max_edits=2, checks_per_depth=128, beam_width=32))
    assert result["status"] == "witness_found"
    assert result["roundtrip_rejected_candidates"] > 0
    assert result["weighted_witness"]["edit_count"] == 2
    assert result["weighted_witness"]["smiles"] == "CCC"


def test_teacher_detaches_targets_has_fresh_ids_and_skips_valid_empty_with_explicit_counts():
    spec = GraphSpec(max_nodes=3)
    output = _decoded(spec, ("invalid", "valid", "empty"))
    teacher = _teacher(spec)
    batch = teacher.build_targets(output, candidate_ids=[11, 12, 13], update=7)
    assert batch.receipt["attempted_candidates"] == 3
    assert batch.receipt["accepted_targets"] == 1
    assert batch.receipt["status_counts"] == {"accepted": 1, "skipped_already_valid": 1, "unsupported": 1}
    assert batch.targets[1:] == (None, None)
    target = batch.targets[0]
    assert target.candidate_id == 11 and target.source_update == 7
    assert all(v.device.type == "cpu" and not v.requires_grad and v.dtype == torch.long for v in
               (target.graph.atoms, target.graph.charges, target.graph.bonds))
    assert batch.receipt["search_checks"] > 0 and batch.receipt["elapsed_seconds"] >= 0
    assert batch.receipt["policy"] == repair_policy(teacher.config)
    assert output.atoms.grad is None  # Search never differentiates decoder logits.
    copied = target.graph
    copied.atoms[0, 0] = 0
    target.verify(spec)


def test_exact_cache_scope_and_no_stale_target_for_a_new_probability_state():
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    first = teacher.build_targets(_decoded(spec), candidate_ids=[1], update=1)
    repeated = teacher.build_targets(_decoded(spec), candidate_ids=[2], update=2)
    assert repeated.receipt["cache_hits"] == 1
    assert repeated.targets[0].candidate_id == 2
    assert repeated.targets[0].source_update == 2
    assert first.targets[0].source_probability_sha256 == repeated.targets[0].source_probability_sha256
    changed = _decoded(spec)
    with torch.no_grad():
        changed.bonds[0, 0, 1, 2] -= 0.25
        changed.bonds[0, 1, 0, 2] -= 0.25
    new = teacher.build_targets(changed, candidate_ids=[3], update=3)
    assert new.receipt["cache_hits"] == 0
    assert new.targets[0].source_probability_sha256 != first.targets[0].source_probability_sha256
    empty = teacher.build_targets(_decoded(spec, ("empty",)), candidate_ids=[4], update=4)
    assert empty.targets == (None,) and empty.receipt["accepted_targets"] == 0


@pytest.mark.parametrize("mutation", ["atom", "charge", "occupancy", "boolean", "invalid", "cost", "provenance"])
def test_independent_target_postcheck_rejects_malicious_search_witness(monkeypatch, mutation):
    import latentmol.atom_preserving_chemistry as module
    native = module.search_atom_preserving_repair
    def poisoned(*args):
        result = copy.deepcopy(native(*args))
        witness = result["weighted_witness"]
        spec = args[2]
        if mutation == "atom":
            witness["labels"]["atoms"][0] = spec.atomic_numbers.index(16)
        elif mutation == "charge":
            witness["labels"]["charges"][0] = spec.formal_charges.index(1)
        elif mutation == "occupancy":
            witness["labels"]["atoms"][2] = spec.atomic_numbers.index(6)
        elif mutation == "boolean":
            witness["labels"]["atoms"][0] = True
        elif mutation == "invalid":
            witness["labels"]["bonds"][0][1] = witness["labels"]["bonds"][1][0] = 3
        elif mutation == "cost":
            witness["weighted_cost"] += 1
        else:
            result["probability_sha256"] = "0" * 64
        return result
    monkeypatch.setattr(module, "search_atom_preserving_repair", poisoned)
    spec, teacher = GraphSpec(max_nodes=3), None
    teacher = _teacher(spec)
    batch = teacher.build_targets(_decoded(spec), candidate_ids=[1], update=0)
    assert batch.targets == (None,)
    assert batch.receipt["status_counts"] == {"rejected_or_error": 1}
    assert "error" in batch.receipt["attempts"][0]
    assert batch.receipt["search_checks"] > 0  # Rejection does not erase work.


def test_repair_nll_matches_full_formula_with_unordered_bonds_and_missing_zero_rows():
    spec = GraphSpec(max_nodes=3)
    teacher = _teacher(spec)
    batch = teacher.build_targets(_decoded(spec, ("invalid", "valid")), candidate_ids=[1, 2], update=0)
    output = _decoded(spec, ("invalid", "valid"))
    observed = repair_loss(output, batch, spec)
    target = batch.targets[0].graph
    active = target.atoms[0] != 0
    pairs = (active[:, None] & active[None, :]).triu(1)
    exact = (F.cross_entropy(output.atoms[0], target.atoms[0], reduction="sum")
             + F.cross_entropy(output.charges[0, active], target.charges[0, active], reduction="sum")
             + F.cross_entropy(output.symmetric_bond_logits[0, pairs], target.bonds[0, pairs], reduction="sum"))
    exact /= batch.targets[0].decisions * 2  # ALL rows, not just accepted targets.
    torch.testing.assert_close(observed, exact)
    gradients = torch.autograd.grad(observed, (output.atoms, output.charges, output.bonds), retain_graph=True)
    expected = torch.autograd.grad(exact, (output.atoms, output.charges, output.bonds))
    for left, right in zip(gradients, expected):
        torch.testing.assert_close(left, right)
        assert torch.count_nonzero(left[1]) == 0
    assert gradients[0][0, 2, 0] < 0  # Padding atom is an unconditional decision.
    assert torch.count_nonzero(gradients[1][0, 2]) == 0
    assert torch.count_nonzero(gradients[2][0, 2]) == 0
    torch.testing.assert_close(gradients[2], gradients[2].transpose(1, 2))


def test_no_witness_keeps_all_head_gradients_exactly_zero():
    spec = GraphSpec(max_nodes=3)
    output = _decoded(spec, ("valid", "empty"))
    batch = _teacher(spec).build_targets(output, candidate_ids=[8, 9], update=1)
    loss = repair_loss(output, batch, spec)
    assert loss == 0
    gradients = torch.autograd.grad(loss, (output.atoms, output.charges, output.bonds))
    assert all(torch.count_nonzero(value) == 0 for value in gradients)


def test_extreme_target_logit_has_no_probability_floor_and_retains_gradient():
    spec = GraphSpec(max_nodes=3)
    batch = _teacher(spec).build_targets(_decoded(spec), candidate_ids=[1], update=0)
    output = _decoded(spec)
    label = int(batch.targets[0].graph.bonds[0, 0, 1])
    with torch.no_grad():
        output.bonds[0, 0, 1, label] = output.bonds[0, 1, 0, label] = -1000
    loss = repair_loss(output, batch, spec)
    assert torch.isfinite(loss) and loss > 100
    gradient, = torch.autograd.grad(loss, (output.bonds,))
    assert gradient[0, 0, 1, label] < 0 and torch.isfinite(gradient).all()


def test_private_label_mutation_and_point_misalignment_fail_closed():
    spec = GraphSpec(max_nodes=3)
    batch = _teacher(spec).build_targets(_decoded(spec), candidate_ids=[1], update=0)
    mismatch = replace(batch, candidate_ids=(9,))
    with pytest.raises(ValueError, match="alignment"):
        repair_loss(_decoded(spec), mismatch, spec)
    batch.targets[0]._graph.bonds[0, 0, 1] = 0
    with pytest.raises(ValueError, match="identity changed"):
        repair_loss(_decoded(spec), batch, spec)


def test_private_extra_batch_row_and_duplicate_subset_fail_closed():
    spec = GraphSpec(max_nodes=3)
    batch = _teacher(spec).build_targets(_decoded(spec), candidate_ids=[1], update=0)
    with pytest.raises(ValueError, match="indices"):
        batch.subset([0, 0])
    # The identity of row zero is unchanged; exact shape checking must still
    # reject an extra tensor row that otherwise changes NLL broadcasting.
    # Label storage can originate in NumPy and need not be resizable. Replace
    # this mutable inner field so the fixture reaches our shape validator rather
    # than failing prematurely in PyTorch's storage resize implementation.
    original = batch.targets[0]._graph.atoms
    batch.targets[0]._graph.atoms = torch.cat((original, original), dim=0)
    with pytest.raises(ValueError, match="shape changed"):
        repair_loss(_decoded(spec), batch, spec)


def test_fixed_target_nll_gradient_matches_finite_difference():
    spec = GraphSpec(max_nodes=3)
    batch = _teacher(spec).build_targets(_decoded(spec), candidate_ids=[5], update=0)
    output = _decoded(spec)
    values = (output.atoms, output.charges, output.bonds)
    generator = torch.Generator().manual_seed(177)
    direction = tuple(torch.randn(value.shape, generator=generator, dtype=value.dtype) for value in values)
    gradients = torch.autograd.grad(repair_loss(output, batch, spec), values)
    analytic = sum((g * d).sum() for g, d in zip(gradients, direction))
    epsilon = 1e-5
    positive = DecoderOutput(*(v.detach() + epsilon * d for v, d in zip(values, direction)))
    negative = DecoderOutput(*(v.detach() - epsilon * d for v, d in zip(values, direction)))
    observed = (repair_loss(positive, batch, spec) - repair_loss(negative, batch, spec)) / (2 * epsilon)
    torch.testing.assert_close(analytic, observed, rtol=1e-6, atol=1e-8)


def test_expected_composition_counts_all_outputs_excludes_no_atom_and_gates_charge():
    spec = GraphSpec(max_nodes=3)
    output = _decoded(spec, ("invalid", "empty"))
    result = composition_statistics(output, spec)
    p = output.atoms.softmax(-1)
    occupancy = p[..., 1:].sum(-1)
    torch.testing.assert_close(result.elements, p[..., 1:].sum((0, 1)))
    torch.testing.assert_close(result.charges, (output.charges.softmax(-1) * occupancy[..., None]).sum((0, 1)))
    torch.testing.assert_close(result.occupied_mass, result.elements.sum())
    torch.testing.assert_close(result.occupied_mass, result.charges.sum())
    assert result.elements.numel() == len(spec.atomic_numbers) - 1
    # Reference counts are empirical occupied labels, never charges in padding.
    hard = output.harden(spec)
    hard.charges[1] = spec.formal_charges.index(2)
    reference = hard_composition_statistics(hard, spec)
    assert reference.occupied_mass == 2
    assert reference.elements[spec.atomic_numbers.index(8) - 1] == 1
    assert reference.charges[spec.formal_charges.index(2)] == 0


def test_js_equal_composition_zero_sulfur_inflation_penalty_and_zero_reference_categories():
    empirical = torch.tensor([90., 5., 5., 0.], dtype=torch.float64)
    assert population_js(empirical, empirical) == 0
    sulfur = torch.tensor([5., 5., 90., 0.], dtype=torch.float64, requires_grad=True)
    loss = population_js(sulfur, empirical)
    assert 0 < loss < math.log(2)
    gradient, = torch.autograd.grad(loss, (sulfur,))
    assert torch.isfinite(gradient).all() and gradient[2] > 0 and gradient[0] < 0
    # A rare/previously absent category receives finite gradients, not infinity.
    assert torch.isfinite(gradient[3])


def test_zero_occupied_population_has_explicit_uniform_fallback_and_finite_zero_cotangent():
    counts = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    value, cotangent = population_js_cotangent(counts, [1., 0., 0., 0.])
    assert torch.isfinite(value) and 0 < value < math.log(2)
    assert torch.isfinite(cotangent).all() and torch.count_nonzero(cotangent) == 0


def test_population_js_streamed_cotangent_matches_true_full_batch_gradient_and_fd():
    spec = GraphSpec(max_nodes=3)
    output = _decoded(spec, ("invalid", "empty", "valid"))
    reference = torch.tensor([.6, .1, .15, .04, .01, .04, .04, .01, .01], dtype=torch.float64)
    counts = composition_statistics(output, spec).elements
    loss = population_js(counts, reference)
    full_gradient, = torch.autograd.grad(loss, (output.atoms,))
    micro_outputs = [DecoderOutput(*(v.detach()[rows].clone().requires_grad_() for v in
                                    (output.atoms, output.charges, output.bonds)))
                     for rows in (slice(0, 1), slice(1, 3))]
    detached_total = sum(composition_statistics(micro, spec).elements.detach() for micro in micro_outputs)
    value, cotangent = population_js_cotangent(detached_total, reference)
    torch.testing.assert_close(value, loss.detach())
    gradients = []
    for micro in micro_outputs:
        linear = (composition_statistics(micro, spec).elements * cotangent).sum()
        gradients.append(torch.autograd.grad(linear, (micro.atoms,))[0])
    torch.testing.assert_close(torch.cat(gradients), full_gradient, rtol=1e-10, atol=1e-12)
    generator = torch.Generator().manual_seed(211)
    direction = torch.randn(output.atoms.shape, generator=generator, dtype=torch.float64)
    epsilon = 1e-5
    plus = DecoderOutput(output.atoms.detach() + epsilon * direction, output.charges.detach(), output.bonds.detach())
    minus = DecoderOutput(output.atoms.detach() - epsilon * direction, output.charges.detach(), output.bonds.detach())
    fd = (population_js(composition_statistics(plus, spec).elements, reference)
          - population_js(composition_statistics(minus, spec).elements, reference)) / (2 * epsilon)
    torch.testing.assert_close((full_gradient * direction).sum(), fd, rtol=1e-5, atol=1e-9)


def test_charge_composition_streamed_vjp_includes_both_occupancy_and_charge_heads():
    spec = GraphSpec(max_nodes=3)
    output = _decoded(spec, ("invalid", "empty", "valid"))
    # Different charge laws and soft occupancy make the atom-to-charge-count
    # path material; testing only uniform charges would hide that dependency.
    with torch.no_grad():
        output.atoms[1, :, 0] = -0.5
        output.atoms[1, :, 1] = -1.0
        output.charges[1, :, spec.formal_charges.index(1)] = 1.5
    reference = torch.tensor([0., .02, .93, .05, 0.], dtype=torch.float64)
    counts = composition_statistics(output, spec).charges
    loss = population_js(counts, reference)
    full = torch.autograd.grad(loss, (output.atoms, output.charges))
    value, cotangent = population_js_cotangent(counts.detach(), reference)
    torch.testing.assert_close(value, loss.detach())
    pieces = [[], []]
    for rows in (slice(0, 2), slice(2, 3)):
        micro = DecoderOutput(*(v.detach()[rows].clone().requires_grad_() for v in
                                (output.atoms, output.charges, output.bonds)))
        linear = (composition_statistics(micro, spec).charges * cotangent).sum()
        gradients = torch.autograd.grad(linear, (micro.atoms, micro.charges))
        for destination, gradient in zip(pieces, gradients):
            destination.append(gradient)
    for observed, expected in zip(pieces, full):
        torch.testing.assert_close(torch.cat(observed), expected, rtol=1e-10, atol=1e-12)
        assert torch.count_nonzero(expected) > 0


@pytest.mark.parametrize("counts,reference", [([-1., 2.], [1., 1.]),
                                              ([float("nan"), 2.], [1., 1.]),
                                              ([1., 2.], [0., 0.]),
                                              ([1., 2.], [1., -1.]),
                                              ([1., 2.], [1., 1., 1.])])
def test_js_invalid_population_or_reference_rejected(counts, reference):
    with pytest.raises(ValueError):
        population_js(torch.tensor(counts, dtype=torch.float64), reference)


def test_duplicate_candidate_ids_and_reference_gradient_are_rejected():
    spec = GraphSpec(max_nodes=3)
    with pytest.raises(ValueError, match="unique"):
        _teacher(spec).build_targets(_decoded(spec, ("invalid", "invalid")), candidate_ids=[1, 1], update=0)
    with pytest.raises(ValueError, match="constant"):
        population_js(torch.ones(3), torch.ones(3, requires_grad=True))
