"""Allocated-HPC chemistry-stage diagnostics; classification never repairs data."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from latentmol import chemistry_classification as module
from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.chemistry import DecodeResult, MoleculeRejected, encode_smiles
from latentmol.failure_guidance_training import FailureTeacherBatch, build_repair_teacher
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.repair_failure_guidance import FailureGuidanceConfig, LOSS_V2
from latentmol.representation import GraphSpec
from test_atom_preserving_chemistry import _decoded


def graph(smiles="CCO"):
    spec = GraphSpec(max_nodes=8)
    encoded = encode_smiles(smiles, spec)
    return spec, tuple(encoded[key].copy() for key in ("atom_types", "charges", "bond_types"))


def test_supported_raw_chemistry_passes_all_stages_without_input_mutation():
    spec, raw = graph()
    before = deepcopy(raw)
    result = module.classify_repair_graph(*raw, spec)
    assert result["failure_kind"] == "none" and result["reason"] is None
    assert all(result[key] is True for key in module.STAGES)
    assert result["smiles"] == result["representation_smiles"] == "CCO"
    assert all(np.array_equal(a, b) for a, b in zip(before, raw))


def test_sanitized_disconnected_graph_is_not_conflated_with_roundtrip_failure():
    spec, raw = graph("CC")
    raw[2].fill(0)
    result = module.classify_repair_graph(*raw, spec)
    assert result["graph_sanitization_valid"] is True
    assert result["raw_chemical_valid"] is False
    assert result["schema_roundtrip_valid"] is result["smiles_parse_valid"] is None
    assert result["failure_kind"] == "chemical" and result["reason"] == "disconnected"


def test_impossible_valence_is_a_native_chemical_failure():
    spec = GraphSpec(max_nodes=3)
    output = _decoded(spec, ("invalid",))
    hard = output.harden(spec)
    result = module.classify_repair_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
    assert result["graph_sanitization_valid"] is False
    assert result["raw_chemical_valid"] is False
    assert result["failure_kind"] == "chemical" and result["reason"] == "sanitization"
    assert result["strict_teacher_valid"] is False


def test_label_input_failure_remains_not_assessed_chemistry():
    spec, raw = graph()
    result = module.classify_repair_graph(raw[0].astype(float), raw[1], raw[2], spec)
    assert result["failure_kind"] == "input"
    assert result["graph_sanitization_valid"] is result["raw_chemical_valid"] is None
    assert result["strict_teacher_valid"] is False


def test_expected_schema_rejection_is_representation_failure_after_valid_native_graph(monkeypatch):
    spec, raw = graph()
    def rejected(*args): raise MoleculeRejected("representation_roundtrip", "fixture lost an unrepresented feature")
    monkeypatch.setattr(module, "encode_smiles", rejected)
    result = module.classify_repair_graph(*raw, spec)
    assert result["graph_sanitization_valid"] and result["raw_chemical_valid"] and result["smiles_parse_valid"]
    assert result["schema_roundtrip_valid"] is False and result["strict_teacher_valid"] is False
    assert result["failure_kind"] == "representation" and result["reason"] == "representation_roundtrip"


def test_serialized_smiles_parse_and_canonical_identity_failures_are_explicit(monkeypatch):
    spec, raw = graph()
    native_parse = module.Chem.MolFromSmiles
    monkeypatch.setattr(module.Chem, "MolFromSmiles", lambda text: None)
    result = module.classify_repair_graph(*raw, spec)
    assert result["raw_chemical_valid"] is True and result["smiles_parse_valid"] is False
    assert result["failure_kind"] == "representation" and result["reason"] == "serialized_smiles_parse"
    monkeypatch.setattr(module.Chem, "MolFromSmiles", native_parse)
    monkeypatch.setattr(module, "encode_smiles", lambda *args: {"smiles": "C"})
    result = module.classify_repair_graph(*raw, spec)
    assert result["raw_chemical_valid"] is True and result["schema_roundtrip_valid"] is False
    assert result["reason"] == "canonical_identity_mismatch"


def test_unexpected_runtime_exception_is_not_hidden_as_bad_chemistry(monkeypatch):
    spec, raw = graph()
    def broken(*args): raise RuntimeError("unexpected native implementation failure")
    monkeypatch.setattr(module, "encode_smiles", broken)
    with pytest.raises(RuntimeError, match="unexpected native"):
        module.classify_repair_graph(*raw, spec)


def test_summaries_keep_all_attempt_denominators_and_bound_separate_failure_examples():
    spec, raw = graph()
    good = module.classify_repair_graph(*raw, spec)
    empty = tuple(value.copy() for value in raw); empty[0].fill(0)
    bad = module.classify_repair_graph(*empty, spec)
    records = [{"classification": bad, "candidate_id": i, "labels": {"atoms": empty[0].tolist()}}
               for i in range(6)] + [{"classification": good, "candidate_id": 6}]
    together = module.classification_summary(records, max_examples_per_failure=2)
    merged = module.merge_classification_summaries([
        module.classification_summary(records[:3], max_examples_per_failure=2),
        module.classification_summary(records[3:], max_examples_per_failure=2)], max_examples_per_failure=2)
    assert together == merged
    assert together["attempts"] == 7 and together["counts"]["strict_teacher_valid"] == 1
    assert together["counts"]["chemical_failures"] == 6
    assert len(together["failure_examples"]["chemical"]) == 2
    assert together["assessment_counts"]["schema_roundtrip_valid"] == {"passed": 1, "failed": 0, "not_evaluated": 6}
    records[0]["labels"]["atoms"][0] = 999
    assert together["failure_examples"]["chemical"][0]["labels"]["atoms"][0] == 0


def test_optional_teacher_diagnostics_preserve_search_targets_and_loss():
    spec = GraphSpec(max_nodes=4)
    output = _decoded(spec, ("invalid", "valid", "disconnected", "empty"))
    config = ProbabilityRepairConfig(max_edits=1, checks_per_depth=32)
    with AtomPreservingRepairTeacher(spec, config) as legacy:
        original = legacy.build_targets(output, candidate_ids=[0, 1, 2, 3], update=0)
    with AtomPreservingRepairTeacher(spec, config, classification_version=module.VERSION,
                                     classification_examples=1) as annotated:
        current = annotated.build_targets(output, candidate_ids=[0, 1, 2, 3], update=0)
    assert "chemical_classification" not in original.receipt
    assert current.receipt["chemical_classification"]["attempts"] == 4
    assert current.receipt["chemical_classification"]["counts"]["strict_teacher_valid"] == 1
    assert current.receipt["chemical_classification"]["counts"]["chemical_failures"] == 3
    assert len(current.receipt["chemical_classification"]["failure_examples"]["chemical"]) == 1
    assert current.receipt["status_counts"] == original.receipt["status_counts"]
    assert torch.equal(repair_loss(output, current, spec), repair_loss(output, original, spec))
    assert [None if t is None else t.record() for t in current.targets] == [None if t is None else t.record() for t in original.targets]


def test_new_loss_teacher_enables_classification_and_preparation_normalization_explicitly():
    spec = GraphSpec(max_nodes=4)
    protocol = {"search": asdict(ProbabilityRepairConfig(max_edits=1, checks_per_depth=1)),
        "failure_guidance": {"version": "chemistry-failure-guidance-v1",
            "loss": asdict(FailureGuidanceConfig(version=LOSS_V2)), "progress_search": None}}
    with build_repair_teacher(spec, protocol) as teacher:
        result = teacher.build_targets(_decoded(spec), candidate_ids=[0], update=0)
    assert isinstance(result, FailureTeacherBatch)
    assert result.base.receipt["chemical_classification"]["attempts"] == 1
    assert result.prepared.receipt["loss_version"] == LOSS_V2


def test_merged_metadata_rejects_inconsistent_stage_and_failure_denominators():
    spec, raw = graph()
    classified = module.classify_repair_graph(*raw, spec)
    initial = module.classification_summary([{"classification": classified}])
    missing = deepcopy(initial); missing["counts"]["strict_teacher_valid"] = 0
    with pytest.raises(ValueError, match="counts"):
        module.merge_classification_summaries([missing])
    extra = deepcopy(initial); extra["failure_counts"]["chemical"] = {"sanitization": 1}
    with pytest.raises(ValueError, match="denominator"):
        module.merge_classification_summaries([extra])
