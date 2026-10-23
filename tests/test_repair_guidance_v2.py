"""Allocated-HPC numerical fixtures for the explicit normalization correction."""
from dataclasses import asdict
import os

import pytest
import torch
from torch.nn import functional as F

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.chemistry_runtime import validate_failure_guidance
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.repair_failure_guidance import (FailureGuidanceConfig, LOSS_V1, LOSS_V2,
                                              failure_aware_repair_loss, prepare_failure_guidance)
from latentmol.repair_progress import ProgressSearchConfig
from latentmol.representation import DecoderOutput
from test_repair_failure_guidance import failed_batch, output_for


def _partial():
    spec, output = output_for([9, 9, 9, 9], [(0, 1, "TRIPLE"), (2, 3, "TRIPLE")])
    return spec, output, failed_batch(output, spec), ProgressSearchConfig(max_edits=1, checks_per_depth=128)


def test_loss_versions_are_explicit_and_v1_remains_the_default():
    assert FailureGuidanceConfig().version == LOSS_V1
    for version in (LOSS_V1, LOSS_V2):
        config = FailureGuidanceConfig(version=version)
        policy = {"version": "chemistry-failure-guidance-v1", "loss": asdict(config), "progress_search": None}
        assert validate_failure_guidance(policy)["loss"]["version"] == version
    with pytest.raises(ValueError):
        FailureGuidanceConfig(version="repair-failure-guidance-v3")


def test_v1_partial_value_and_gradient_are_the_historical_edited_label_mean():
    spec, output, base, search = _partial()
    prepared = prepare_failure_guidance(output, base, spec, progress_search=search)
    assert "loss_version" not in prepared.receipt
    witness = prepared.rows[0].witness
    assert witness.kind == "partial"
    logp = F.log_softmax(output.symmetric_bond_logits[0], -1)
    edited = [(i, j) for i in range(spec.max_nodes) for j in range(i + 1, spec.max_nodes)
              if witness.original[2][i][j] != witness.labels[2][i][j]]
    direct = torch.stack([-logp[i, j, witness.labels[2][i][j]] for i, j in edited]).mean()
    actual = failure_aware_repair_loss(output, prepared, spec)["partial_edited_nll"]
    assert torch.equal(actual, direct)
    left = torch.autograd.grad(actual, output.bonds, retain_graph=True)[0]
    right = torch.autograd.grad(direct, output.bonds)[0]
    assert torch.equal(left, right)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_v2_partial_sum_uses_full_graph_decision_denominator_and_same_edited_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES") == "1":
            pytest.fail("Mandatory allocated CUDA normalization/VJP fixture has no visible CUDA device")
        pytest.skip("Allocated CUDA fixture required; CPU pass is not GPU validation")
    spec, output, base, search = _partial()
    output = DecoderOutput(*(value.detach().to(device).clone().requires_grad_()
                             for value in (output.atoms, output.charges, output.bonds)))
    old = prepare_failure_guidance(output, base, spec, progress_search=search)
    new = prepare_failure_guidance(output, base, spec, progress_search=search, loss_version=LOSS_V2)
    witness = new.rows[0].witness
    assert old.rows[0].witness == witness and witness.kind == "partial"
    size = sum(bool(a) for a in witness.labels[0])
    decisions = spec.max_nodes + size + size * (size - 1) // 2
    changed = sum(witness.labels[2][i][j] != witness.original[2][i][j]
                  for i in range(spec.max_nodes) for j in range(i + 1, spec.max_nodes))
    old_terms = failure_aware_repair_loss(output, old, spec)
    new_terms = failure_aware_repair_loss(output, new, spec, FailureGuidanceConfig(version=LOSS_V2))
    torch.testing.assert_close(new_terms["partial_edited_nll"], old_terms["partial_edited_nll"] * changed / decisions,
                               rtol=1e-13, atol=1e-14)
    for key in ("valid_full_nll", "extra_valid_nll", "local_bond_risk", "component_cut_risk", "nonempty_risk"):
        assert torch.equal(new_terms[key], old_terms[key]), key
    old_gradient = torch.autograd.grad(old_terms["partial_edited_nll"],
        (output.atoms, output.charges, output.bonds), retain_graph=True)
    new_gradient = torch.autograd.grad(new_terms["partial_edited_nll"],
        (output.atoms, output.charges, output.bonds))
    for old_value, new_value in zip(old_gradient, new_gradient):
        torch.testing.assert_close(new_value, old_value * changed / decisions, rtol=1e-13, atol=1e-14)
    assert torch.count_nonzero(new_gradient[0]) == torch.count_nonzero(new_gradient[1]) == 0


def test_v2_all_attempt_mean_keeps_an_already_valid_second_row_at_zero():
    spec, output, base, search = _partial()
    _, valid = output_for([6, 6, 6, 6], [(0, 1, "SINGLE"), (1, 2, "SINGLE"), (2, 3, "SINGLE")])
    batch = DecoderOutput(*(torch.cat((a.detach(), b.detach()), 0).requires_grad_()
        for a, b in zip((output.atoms, output.charges, output.bonds), (valid.atoms, valid.charges, valid.bonds))))
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1)) as teacher:
        targets = teacher.build_targets(batch, candidate_ids=[0, 1], update=0)
    single = prepare_failure_guidance(output, base, spec, progress_search=search, loss_version=LOSS_V2)
    together = prepare_failure_guidance(batch, targets, spec, progress_search=search, loss_version=LOSS_V2)
    config = FailureGuidanceConfig(version=LOSS_V2)
    one = failure_aware_repair_loss(output, single, spec, config)["partial_edited_nll"]
    two = failure_aware_repair_loss(batch, together, spec, config)["partial_edited_nll"]
    assert torch.equal(two, one / 2)
    gradients = torch.autograd.grad(two, (batch.atoms, batch.charges, batch.bonds))
    assert all(torch.count_nonzero(value[1]) == 0 for value in gradients)


def test_declared_preparation_and_live_loss_versions_must_match():
    spec, output, base, search = _partial()
    for prepared_version, live_version in ((LOSS_V1, LOSS_V2), (LOSS_V2, LOSS_V1)):
        prepared = prepare_failure_guidance(output, base, spec, progress_search=search, loss_version=prepared_version)
        with pytest.raises(ValueError, match="normalization"):
            failure_aware_repair_loss(output, prepared, spec, FailureGuidanceConfig(version=live_version))


def test_v2_complete_repair_nll_is_unchanged():
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=32)) as teacher:
        base = teacher.build_targets(output, candidate_ids=[0], update=0)
    assert base.targets[0] is not None
    old = prepare_failure_guidance(output, base, spec)
    new = prepare_failure_guidance(output, base, spec, loss_version=LOSS_V2)
    one = failure_aware_repair_loss(output, old, spec)
    two = failure_aware_repair_loss(output, new, spec, FailureGuidanceConfig(version=LOSS_V2))
    assert torch.equal(one["total"], two["total"])
    assert torch.equal(one["valid_full_nll"], two["valid_full_nll"])
