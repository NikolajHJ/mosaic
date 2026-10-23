"""Allocated-HPC exact source/provenance parity for preparation invariant reuse."""
import os

import pytest
import torch

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol import repair_failure_guidance as guidance
from latentmol.repair_execution_reference import load_preparation_reference
from latentmol.repair_progress import (ProgressSearchConfig, search_progress_repair,
                                      search_progress_repair_reference)
from latentmol.representation import DecoderOutput, GraphSpec
from test_repair_failure_guidance import output_for


ORACLE_SHA = "34c1a67cb339233312a1317db39a926b55f378c0eb8ed42c38bde86307fdd184"


@pytest.fixture(scope="module")
def oracle():
    path = os.environ.get("LATENTMOL_FAILURE_REFERENCE_MODULE")
    if not path:
        pytest.fail("Mandatory preparation parity requires the pinned historical source module")
    return load_preparation_reference(path, ORACLE_SHA)


def _fixture(kind):
    spec = GraphSpec(max_nodes=50)
    if kind == "empty":
        return output_for([], [], spec=spec)
    if kind == "disconnected":
        return output_for([9] * 4, [(0, 1, "TRIPLE"), (2, 3, "TRIPLE")], spec=spec)
    if kind == "invalid":
        return output_for([9, 9], [(0, 1, "TRIPLE")], spec=spec)
    if kind == "valid":
        return output_for([6, 6], [(0, 1, "SINGLE")], spec=spec)
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["empty", "disconnected", "invalid", "valid"])
@pytest.mark.parametrize("depth", [None, 2, 6])
def test_historical_preparation_identity_and_all_loss_components_are_exact(oracle, kind, depth):
    spec, decoded = _fixture(kind)
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1)) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=[17], update=3)
    policy = None if depth is None else ProgressSearchConfig(max_edits=depth)
    old = oracle.prepare_failure_guidance(decoded, base, spec, progress_search=policy,
        progress_map=None if policy is None else lambda tasks: [search_progress_repair_reference(*task) for task in tasks])
    phases = {}
    new = guidance.prepare_failure_guidance(decoded, base, spec, progress_search=policy,
        progress_map=None if policy is None else lambda tasks: [search_progress_repair(*task) for task in tasks],
        phase_seconds=phases)
    assert old.identity == new.identity
    assert old.receipt == new.receipt
    old_rows = tuple(None if row is None else guidance.FrozenFailureRow(**{
        key: getattr(row, key) for key in guidance.FrozenFailureRow.__dataclass_fields__}) for row in old.rows)
    aligned = guidance.PreparedFailureGuidance(base, old_rows, old.schema_sha256, old.identity, old.receipt)
    expected = guidance.failure_aware_repair_loss(decoded, aligned, spec)
    actual = guidance.failure_aware_repair_loss(decoded, new, spec)
    fields = (decoded.atoms, decoded.charges, decoded.bonds)
    for name in ("total", "valid_full_nll", "extra_valid_nll", "partial_edited_nll", "local_bond_risk", "component_cut_risk", "nonempty_risk"):
        assert torch.equal(expected[name], actual[name])
    left = torch.autograd.grad(expected["total"], fields, allow_unused=True)
    right = torch.autograd.grad(actual["total"], fields, allow_unused=True)
    for before, after in zip(left, right):
        assert (before is None) == (after is None)
        if before is not None:
            assert torch.equal(before, after)
    assert set(phases) == {"snapshot", "source_validation", "mapped_progress", "assembly_and_identity"}
    assert all(value >= 0 for value in phases.values())


def test_mapped_source_hashing_occurs_once_per_row_without_losing_validation(monkeypatch):
    from latentmol import probability_repair
    spec, decoded = _fixture("invalid")
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1)) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=[17], update=3)
    original = probability_repair.probability_identity
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(probability_repair, "probability_identity", counted)
    guidance.prepare_failure_guidance(decoded, base, spec, progress_search=ProgressSearchConfig(max_edits=1),
        progress_map=lambda tasks: [search_progress_repair(*task) for task in tasks])
    assert len(calls) == 1


def test_all_source_rows_are_checked_before_progress_dispatch():
    spec, invalid = _fixture("invalid")
    _, valid = _fixture("valid")
    decoded = DecoderOutput(*(torch.cat((getattr(invalid, name), getattr(valid, name)))
                              for name in ("atoms", "charges", "bonds")))
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1)) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=[17, 18], update=3)
    base.receipt["attempts"][1]["probability_sha256"] = "changed"
    dispatched = []
    with pytest.raises(ValueError, match="exact original"):
        guidance.prepare_failure_guidance(decoded, base, spec, progress_search=ProgressSearchConfig(),
            progress_map=lambda tasks: dispatched.append(tasks))
    assert dispatched == []


def test_timing_cannot_be_reused_or_inserted_into_scientific_identity():
    spec, decoded = _fixture("valid")
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig()) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=[17], update=3)
    with pytest.raises(ValueError, match="fresh empty"):
        guidance.prepare_failure_guidance(decoded, base, spec, phase_seconds={"old": 1})


def test_mapper_cannot_reclassify_a_failed_attempt_after_its_prevalidation():
    spec, decoded = _fixture("invalid")
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1)) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=[17], update=3)
    def changed(tasks):
        result = [search_progress_repair(*task) for task in tasks]
        base.receipt["attempts"][0]["status"] = "skipped_already_valid"
        return result
    with pytest.raises(ValueError, match="changed its authenticated"):
        guidance.prepare_failure_guidance(decoded, base, spec, progress_search=ProgressSearchConfig(), progress_map=changed)


def test_mapper_cannot_mutate_accepted_target_tensor_after_source_prevalidation():
    spec, decoded = _fixture("invalid")
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=32)) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=[17], update=3)
    assert base.targets[0] is not None
    def changed(tasks):
        base.targets[0]._graph.bonds[0, 0, 1] = 0
        return []
    with pytest.raises(ValueError, match="label identity changed"):
        guidance.prepare_failure_guidance(decoded, base, spec, progress_search=ProgressSearchConfig(), progress_map=changed)
