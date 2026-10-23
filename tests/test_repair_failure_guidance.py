"""Allocated-HPC fixtures for the opt-in research objective, not efficacy tests."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.chemistry import encode_smiles
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.repair_failure_guidance import (FailureGuidanceConfig, failure_aware_repair_loss,
                                              prepare_failure_guidance)
from latentmol.repair_progress import ProgressSearchConfig, search_progress_repair, strictly_improves
from latentmol.representation import DecoderOutput, GraphSpec


def output_for(atoms, edges, *, confidence=3., spec=None):
    spec = GraphSpec(max_nodes=len(atoms)) if spec is None else spec
    a = torch.full((1, spec.max_nodes, len(spec.atomic_numbers)), -confidence, dtype=torch.float64)
    q = torch.full((1, spec.max_nodes, len(spec.formal_charges)), -confidence, dtype=torch.float64)
    b = torch.full((1, spec.max_nodes, spec.max_nodes, len(spec.bond_names)), -confidence, dtype=torch.float64)
    a[..., 0] = confidence
    for i, element in enumerate(atoms):
        a[0, i, 0] = -confidence
        a[0, i, spec.atomic_numbers.index(element)] = confidence
    q[..., spec.formal_charges.index(0)] = confidence
    b[..., 0] = confidence
    for i, j, name in edges:
        b[0, i, j, 0] = b[0, j, i, 0] = -confidence
        b[0, i, j, spec.bond_names.index(name)] = b[0, j, i, spec.bond_names.index(name)] = confidence
    return spec, DecoderOutput(*(v.requires_grad_() for v in (a, q, b)))


def failed_batch(output, spec):
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1)) as teacher:
        batch = teacher.build_targets(output, candidate_ids=list(range(len(output.atoms))), update=0)
    assert all(t is None for t in batch.targets)
    return batch


def test_failed_validity_search_no_longer_silently_removes_local_bond_gradient():
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    base = failed_batch(output, spec)  # The first proposed edit is NONE: disconnected.
    assert float(repair_loss(output, base, spec)) == 0
    prepared = prepare_failure_guidance(output, base, spec)
    result = failure_aware_repair_loss(output, prepared, spec)
    assert result["local_bond_risk"] > 0 and result["total"] > 0
    result["total"].backward()
    assert output.bonds.grad[0, 0, 1, spec.bond_names.index("TRIPLE")] > 0
    assert torch.isfinite(output.bonds.grad).all()
    assert torch.count_nonzero(output.atoms.grad) == 0
    assert torch.count_nonzero(output.charges.grad) == 0


def test_very_confident_invalid_graph_keeps_finite_nonzero_bond_gradient():
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")], confidence=1000.)
    prepared = prepare_failure_guidance(output, failed_batch(output, spec), spec)
    result = failure_aware_repair_loss(output, prepared, spec)
    result["total"].backward()
    assert torch.isfinite(result["total"]) and torch.isfinite(output.bonds.grad).all()
    assert output.bonds.grad.norm() > 0


def test_disconnected_failed_graph_receives_a_cross_component_bond_gradient():
    spec, output = output_for([9, 9, 9, 9], [(0, 1, "TRIPLE"), (2, 3, "TRIPLE")])
    prepared = prepare_failure_guidance(output, failed_batch(output, spec), spec)
    result = failure_aware_repair_loss(output, prepared, spec)
    assert result["component_cut_risk"] > 0
    result["component_cut_risk"].backward()
    assert output.bonds.grad[0, 0, 2, spec.bond_names.index("SINGLE")] < 0


def test_empty_output_gets_stable_nonempty_gradient_without_choosing_sulfur():
    spec, output = output_for([], [], spec=GraphSpec(max_nodes=3), confidence=1000.)
    prepared = prepare_failure_guidance(output, failed_batch(output, spec), spec)
    result = failure_aware_repair_loss(output, prepared, spec)
    assert result["nonempty_risk"] > 0 and torch.isfinite(result["total"])
    result["total"].backward()
    assert torch.isfinite(output.atoms.grad).all() and output.atoms.grad.norm() > 0
    assert torch.allclose(output.atoms.grad[..., 1:], output.atoms.grad[..., 1:2].expand_as(output.atoms.grad[..., 1:]))


def test_partial_target_only_supervises_the_changed_bond_and_stays_invalid():
    spec, output = output_for([9, 9, 9, 9], [(0, 1, "TRIPLE"), (2, 3, "TRIPLE")])
    base = failed_batch(output, spec)
    prepared = prepare_failure_guidance(output, base, spec,
        progress_search=ProgressSearchConfig(max_edits=1, checks_per_depth=128))
    witness = prepared.rows[0].witness
    assert witness is not None and witness.kind == "partial"
    assert strictly_improves(dict(witness.before), dict(witness.after))
    assert witness.original[0] == witness.labels[0]
    result = failure_aware_repair_loss(output, prepared, spec)
    result["partial_edited_nll"].backward()
    edited = [(i, j) for i in range(4) for j in range(i + 1, 4)
              if witness.original[2][i][j] != witness.labels[2][i][j]]
    for i in range(4):
        for j in range(i + 1, 4):
            assert bool(output.bonds.grad[0, i, j].abs().sum()) is ((i, j) in edited)


def test_wider_escalation_can_find_a_valid_graph_the_small_check_budget_missed():
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    prepared = prepare_failure_guidance(output, failed_batch(output, spec), spec,
        progress_search=ProgressSearchConfig(max_edits=2, checks_per_depth=32))
    assert prepared.rows[0].witness.kind == "valid"
    result = failure_aware_repair_loss(output, prepared, spec)
    assert result["extra_valid_nll"] > 0


def test_depth_four_repair_can_fix_four_distinct_overbonded_fluorines():
    # Four separate C-F triples each require a changed label while identities
    # stay fixed. This covers depth, unlike the one-edit F-F widening fixture.
    spec, output = output_for([6, 6, 6, 6, 9, 9, 9, 9],
        [(0, 1, "SINGLE"), (1, 2, "SINGLE"), (2, 3, "SINGLE")]
        + [(i, i + 4, "TRIPLE") for i in range(4)])
    hard = output.harden(spec)
    raw = tuple(v[0].numpy() for v in (hard.atoms, hard.charges, hard.bonds))
    p = tuple(v[0].detach().softmax(-1).numpy() for v in
              (output.atoms, output.charges, output.symmetric_bond_logits))
    shallow = search_progress_repair(raw, p, spec,
        ProgressSearchConfig(max_edits=3, checks_per_depth=4096, beam_width=32))
    assert shallow["status"] != "valid"
    deeper = search_progress_repair(raw, p, spec,
        ProgressSearchConfig(max_edits=4, checks_per_depth=4096, beam_width=32))
    witness = deeper["witness"]
    assert witness is not None and witness.kind == "valid"
    assert sum(witness.labels[2][i][j] != witness.original[2][i][j]
               for i in range(8) for j in range(i + 1, 8)) == 4


def test_all_attempt_normalization_includes_already_valid_rows():
    spec, failed = output_for([9, 9], [(0, 1, "TRIPLE")])
    _, valid = output_for([6, 6], [(0, 1, "SINGLE")], spec=spec)
    single = prepare_failure_guidance(failed, failed_batch(failed, spec), spec)
    mixed = DecoderOutput(*(torch.cat([a.detach(), b.detach()]).requires_grad_()
                            for a, b in zip((failed.atoms, failed.charges, failed.bonds),
                                            (valid.atoms, valid.charges, valid.bonds))))
    targets = failed_batch(mixed, spec)
    together = prepare_failure_guidance(mixed, targets, spec)
    one = failure_aware_repair_loss(failed, single, spec)
    two = failure_aware_repair_loss(mixed, together, spec)
    assert torch.allclose(two["local_bond_risk"], one["local_bond_risk"] / 2)


def test_existing_accepted_witness_nll_is_preserved_exactly():
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=32)) as teacher:
        base = teacher.build_targets(output, candidate_ids=[0], update=0)
    assert base.targets[0] is not None
    prepared = prepare_failure_guidance(output, base, spec)
    result = failure_aware_repair_loss(output, prepared, spec)
    assert prepared.rows == (None,)
    assert torch.equal(result["total"], repair_loss(output, base, spec))
    other_update = replace(base.targets[0], source_update=1)
    with pytest.raises(ValueError, match="authenticated source attempt"):
        prepare_failure_guidance(output, replace(base, targets=(other_update,)), spec)


def test_existing_valid_outputs_still_have_no_chemistry_self_target():
    spec, output = output_for([6, 6], [(0, 1, "SINGLE")])
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig()) as teacher:
        base = teacher.build_targets(output, candidate_ids=[0], update=0)
    assert base.receipt["status_counts"] == {"skipped_already_valid": 1}
    prepared = prepare_failure_guidance(output, base, spec)
    assert prepared.rows == (None,)
    assert float(failure_aware_repair_loss(output, prepared, spec)["total"]) == 0


def test_atom_substitution_is_explicit_and_can_take_a_sulfur_shortcut():
    spec = GraphSpec(max_nodes=7)
    encoded = encode_smiles("FS(F)(F)(F)(F)F", spec)
    raw = tuple(np.array(encoded[key], dtype=np.int64) for key in ("atom_types", "charges", "bond_types"))
    central = int(np.flatnonzero(raw[0] == spec.atomic_numbers.index(16))[0])
    raw[0][central] = spec.atomic_numbers.index(6)
    probabilities = (np.eye(len(spec.atomic_numbers))[raw[0]], np.eye(len(spec.formal_charges))[raw[1]],
                     np.eye(len(spec.bond_names))[raw[2]])
    fixed = search_progress_repair(raw, probabilities, spec, ProgressSearchConfig(max_edits=1, checks_per_depth=512))
    assert fixed["status"] != "valid"
    changed = search_progress_repair(raw, probabilities, spec,
        ProgressSearchConfig(max_edits=1, checks_per_depth=512, allow_atom_edits=True))
    assert changed["status"] == "valid"
    assert changed["witness"].labels[0] != changed["witness"].original[0]
    assert changed["policy"]["allow_atom_edits"]


def test_valid_atom_edited_witness_replaces_conflicting_original_identity_risk():
    spec = GraphSpec(max_nodes=7)
    encoded = encode_smiles("FS(F)(F)(F)(F)F", spec)
    elements = [spec.atomic_numbers[label] for label in encoded["atom_types"]]
    elements[elements.index(16)] = 6
    edges = [(i, j, spec.bond_names[label]) for i, row in enumerate(encoded["bond_types"])
             for j, label in enumerate(row) if j > i and label]
    _, output = output_for(elements, edges, spec=spec)
    prepared = prepare_failure_guidance(output, failed_batch(output, spec), spec,
        progress_search=ProgressSearchConfig(max_edits=1, checks_per_depth=512, allow_atom_edits=True))
    assert prepared.rows[0].witness.kind == "valid"
    result = failure_aware_repair_loss(output, prepared, spec)
    assert result["extra_valid_nll"] > 0 and result["local_bond_risk"] == 0
    assert result["component_cut_risk"] == 0 and result["nonempty_risk"] == 0


def test_unknown_charge_environment_is_reported_as_uncovered_not_given_a_cap():
    # Unsupported by OUR finite-cap table does not mean RDKit rejects it.
    # Three isolated atoms need at least two new bonds for connectivity, so
    # a ONE-edit search cannot repair them even if native valence is unbounded.
    spec, output = output_for([9, 9, 9], [])
    with torch.no_grad():
        output.charges.fill_(-3.)
        output.charges[..., spec.formal_charges.index(-2)] = 3.
    prepared = prepare_failure_guidance(output, failed_batch(output, spec), spec)
    assert prepared.receipt["rows"][0]["unsupported_local_atoms"] == 3
    assert all(cap is None for cap in prepared.rows[0].caps)
    result = failure_aware_repair_loss(output, prepared, spec)
    assert float(result["local_bond_risk"]) == 0


def test_error_or_stale_source_cannot_be_reinterpreted_as_a_legitimate_failure():
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    base = failed_batch(output, spec)
    base.receipt["attempts"][0]["status"] = "rejected_or_error"
    with pytest.raises(ValueError, match="errors"):
        prepare_failure_guidance(output, base, spec)
    base.receipt["attempts"][0]["status"] = "censored"
    with torch.no_grad():
        output.bonds[0, 0, 1, 0] += 0.1
    with pytest.raises(ValueError, match="exact original"):
        prepare_failure_guidance(output, base, spec)


def test_prepared_reference_does_not_change_with_live_decoder_logits():
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    prepared = prepare_failure_guidance(output, failed_batch(output, spec), spec)
    old = prepared.identity
    with torch.no_grad():
        output.bonds[0, 0, 1, spec.bond_names.index("DOUBLE")] += 1.
    assert failure_aware_repair_loss(output, prepared, spec)["total"].isfinite()
    assert prepared.identity == old
    corrupted = replace(prepared, identity="changed")
    with pytest.raises(ValueError, match="identities"):
        failure_aware_repair_loss(output, corrupted, spec)


def test_base_and_coverage_receipts_are_bound_to_the_prepared_identity():
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    base = failed_batch(output, spec)
    prepared = prepare_failure_guidance(output, base, spec)
    altered = replace(prepared, base=replace(base, candidate_ids=(99,)))
    with pytest.raises(ValueError, match="identities"):
        failure_aware_repair_loss(output, altered, spec)
    prepared.receipt["failed_rows"] = 0
    with pytest.raises(ValueError, match="identities"):
        failure_aware_repair_loss(output, prepared, spec)


def test_partial_progress_must_not_trade_one_native_failure_for_another():
    assert strictly_improves({"valence": 3, "extra_components": 1}, {"valence": 2, "extra_components": 1})
    assert not strictly_improves({"valence": 3}, {"valence": 2, "radical": 1})
    assert not strictly_improves({"valence": 3}, {"valence": 2, "extra_components": 1})
    assert not strictly_improves({"valence": 3}, {"valence": 3})


@pytest.mark.parametrize("kwargs", [{"version": "unknown"}, {"max_edits": 0}, {"allow_atom_edits": 1},
                                   {"atom_edit_penalty": 0}, {"checks_per_depth": 100000}])
def test_explicit_bounded_progress_settings(kwargs):
    with pytest.raises(ValueError):
        ProgressSearchConfig(**kwargs)


def test_failure_weight_configuration_does_not_accept_nan():
    with pytest.raises(ValueError):
        FailureGuidanceConfig(local_bond_weight=float("nan"))


@pytest.mark.parametrize("component", ["valid_full_nll", "extra_valid_nll", "partial_edited_nll",
                                      "local_bond_risk", "component_cut_risk", "nonempty_risk"])
def test_each_enabled_component_matches_a_frozen_target_directional_derivative(component):
    if component == "nonempty_risk":
        spec, output = output_for([], [], spec=GraphSpec(max_nodes=3))
    elif component in ("partial_edited_nll", "component_cut_risk"):
        spec, output = output_for([9, 9, 9, 9], [(0, 1, "TRIPLE"), (2, 3, "TRIPLE")])
    else:
        spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    if component == "valid_full_nll":
        with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=32)) as teacher:
            base = teacher.build_targets(output, candidate_ids=[0], update=0)
    else:
        base = failed_batch(output, spec)
    extra = (ProgressSearchConfig(max_edits=1, checks_per_depth=128)
             if component in ("extra_valid_nll", "partial_edited_nll") else None)
    prepared = prepare_failure_guidance(output, base, spec, progress_search=extra)
    value = failure_aware_repair_loss(output, prepared, spec)[component]
    assert float(value) > 0
    fields = (output.atoms, output.charges, output.bonds)
    generator = torch.Generator().manual_seed(123017)
    directions = [torch.randn(v.shape, generator=generator, dtype=v.dtype) for v in fields]
    magnitude = torch.stack([v.square().sum() for v in directions]).sum().sqrt()
    directions = [v / magnitude for v in directions]
    gradients = torch.autograd.grad(value, fields, allow_unused=True)
    analytic = sum((g * v).sum() for g, v in zip(gradients, directions) if g is not None)
    epsilon = 1e-3
    def at(sign):
        shifted = DecoderOutput(*(v.detach() + sign * epsilon * direction
                                  for v, direction in zip(fields, directions)))
        return failure_aware_repair_loss(shifted, prepared, spec)[component]
    numeric = (at(1) - at(-1)) / (2 * epsilon)
    assert torch.isfinite(analytic) and abs(float(analytic)) > 1e-9
    assert torch.allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


def test_local_bond_risk_and_gradient_do_not_depend_on_extra_padding_columns():
    compact_spec, compact = output_for([9, 9], [(0, 1, "TRIPLE")])
    padded_spec, padded = output_for([9, 9], [(0, 1, "TRIPLE")], spec=GraphSpec(max_nodes=50))
    values = []
    for spec, output in ((compact_spec, compact), (padded_spec, padded)):
        prepared = prepare_failure_guidance(output, failed_batch(output, spec), spec)
        risk = failure_aware_repair_loss(output, prepared, spec)["local_bond_risk"]
        gradient = torch.autograd.grad(risk, output.bonds)[0]
        values.append((risk, gradient[0, :2, :2]))
    assert torch.equal(values[0][0], values[1][0])
    assert torch.equal(values[0][1], values[1][1])
