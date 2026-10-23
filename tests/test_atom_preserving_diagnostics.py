"""Reference-parity chemistry fixtures; execute only on allocated DTU HPC.

Every fast-path assertion compares search decisions, rank burden and proposal
slot identities against the unchanged full diagnostic. No speed or chemical
training improvement is inferred merely from these engineering fixtures.
"""
import numpy as np
import pytest
from rdkit import Chem

from latentmol import atom_preserving_diagnostics as fast
from latentmol.chemistry import encode_smiles
from latentmol.quality import diagnose_graph, graph_identity
from latentmol.representation import GraphSpec


def _encoded(smiles, spec):
    result = encode_smiles(smiles, spec)
    return tuple(np.asarray(result[name], dtype=np.int64).copy()
                 for name in ("atom_types", "charges", "bond_types"))


def _raw(spec, elements, pairs=(), charges=None, slots=None):
    slots = list(range(len(elements))) if slots is None else list(slots)
    atoms = np.zeros(spec.max_nodes, dtype=np.int64)
    formal = np.full(spec.max_nodes, spec.formal_charges.index(0), dtype=np.int64)
    bonds = np.zeros((spec.max_nodes, spec.max_nodes), dtype=np.int64)
    for row, (slot, element) in enumerate(zip(slots, elements)):
        atoms[slot] = spec.atomic_numbers.index(element)
        formal[slot] = spec.formal_charges.index(0 if charges is None else charges[row])
    for left, right, name in pairs:
        bonds[slots[left], slots[right]] = bonds[slots[right], slots[left]] = spec.bond_names.index(name)
    return atoms, formal, bonds


def _assert_search_parity(raw, spec, *, include_graph_id=True, diagnostic=None):
    expected = diagnose_graph(*raw, spec)
    observed = (diagnostic or fast.BondOnlyDiagnostic(raw, spec)).diagnose(raw[2], include_graph_id=include_graph_id)
    for field in fast.SEARCH_FIELDS:
        assert (field in observed) == (field in expected), field
        assert observed.get(field) == expected.get(field), field
    if include_graph_id and "graph_id" in expected:
        assert observed["graph_id"] == expected["graph_id"]
    else:
        assert "graph_id" not in observed
    return observed


@pytest.mark.parametrize("smiles", ["C", "CCO", "C[N+](C)(C)C", "[O-]C=O",
                                   "c1ccccc1", "c1ccncc1", "CS(=O)(=O)C", "FS(F)(F)(F)(F)F"])
def test_supported_valid_charge_aromatic_and_expanded_sulfur_reference_parity(smiles):
    spec = GraphSpec(max_nodes=10)
    report = _assert_search_parity(_encoded(smiles, spec), spec)
    assert report["valid"] and report["problem_burden"] == 0


@pytest.mark.parametrize("case", ["oxygen_triple", "carbon_overvalence", "disconnected",
                                  "aromatic_nonring", "odd_aromatic_ring", "empty"])
def test_invalid_and_disconnected_rank_and_problem_slots_match(case):
    spec = GraphSpec(max_nodes=10)
    if case == "oxygen_triple":
        raw = _raw(spec, [8, 6], [(0, 1, "TRIPLE")], slots=[3, 8])
    elif case == "carbon_overvalence":
        raw = _raw(spec, [6] * 6, [(0, right, "SINGLE") for right in range(1, 6)],
                   slots=[8, 0, 2, 4, 6, 9])
    elif case == "disconnected":
        raw = _raw(spec, [6, 8, 6], [(0, 1, "SINGLE")], slots=[1, 5, 9])
    elif case == "aromatic_nonring":
        raw = _raw(spec, [6, 6], [(0, 1, "AROMATIC")], slots=[2, 7])
    elif case == "odd_aromatic_ring":
        raw = _raw(spec, [6] * 5, [(left, (left + 1) % 5, "AROMATIC") for left in range(5)],
                   slots=[0, 2, 4, 7, 9])
    else:
        raw = _raw(spec, [])
    report = _assert_search_parity(raw, spec)
    assert not report["valid"]
    if case == "oxygen_triple":
        assert any(3 in problem["slots"] for problem in report["problems"])
    if case == "disconnected":
        assert report["component_count"] == 2 and report["problem_burden"] == 0


def test_custom_element_and_charge_follow_reference_without_inferred_radical_assumptions():
    # Radical perception is RDKit's decision. In the pinned 2024.09.6 runtime
    # this graph is valid; an electron-count guess must not change its verdict.
    spec = GraphSpec(max_nodes=4, atomic_numbers=(0, 2, 6))
    raw = _raw(spec, [2], charges=[1], slots=[3])
    _assert_search_parity(raw, spec)


def test_synthetic_post_sanitize_radical_is_rejected_with_original_noncontiguous_slot(monkeypatch):
    # Controlled branch fixture: mark the sanitized atom as a radical in BOTH
    # authoritative and optimized paths. This tests rejection and slot burden
    # without guessing which implicit-H graph a particular RDKit version labels
    # radical. It is not evidence that the input methane graph is a radical.
    spec = GraphSpec(max_nodes=4)
    raw = _raw(spec, [6], slots=[3])
    native_sanitize = Chem.SanitizeMol
    def mark_sanitized_radical(mol, *args, **kwargs):
        flag = native_sanitize(mol, *args, **kwargs)
        if flag is None or int(flag) == 0:
            mol.GetAtomWithIdx(0).SetNumRadicalElectrons(1)
        return flag
    monkeypatch.setattr(Chem, "SanitizeMol", mark_sanitized_radical)
    report = _assert_search_parity(raw, spec)
    assert not report["valid"]
    assert {"type": "radical", "slots": [3]} in report["problems"]


def test_fixed_template_preserves_slot_order_charges_and_caller_mutation_isolation():
    spec = GraphSpec(max_nodes=10)
    raw = _raw(spec, [8, 6, 7], [(0, 1, "SINGLE"), (1, 2, "TRIPLE")],
               charges=[-1, 0, 0], slots=[8, 2, 5])
    original = tuple(value.copy() for value in raw)
    diagnostic = fast.BondOnlyDiagnostic(raw, spec)
    raw[0][:] = 0
    raw[1][:] = 0
    first = _assert_search_parity(original, spec, diagnostic=diagnostic)
    changed = tuple(value.copy() for value in original)
    changed[2][2, 5] = changed[2][5, 2] = spec.bond_names.index("DOUBLE")
    second = _assert_search_parity(changed, spec, diagnostic=diagnostic)
    assert first["graph_id"] != second["graph_id"]


@pytest.mark.parametrize("defect", ["asymmetric", "self_bond", "padding_bond", "padding_charge",
                                   "category", "float", "shape"])
def test_unsupported_candidates_use_exact_authoritative_fallback(defect):
    spec = GraphSpec(max_nodes=5)
    raw = _encoded("CO", spec)
    diagnostic = fast.BondOnlyDiagnostic(raw, spec)
    if defect == "asymmetric": raw[2][0, 1] = 2
    elif defect == "self_bond": raw[2][0, 0] = 1
    elif defect == "padding_bond": raw[2][0, 4] = raw[2][4, 0] = 1
    elif defect == "padding_charge":
        raw[1][4] = 0
        diagnostic = fast.BondOnlyDiagnostic(raw, spec)
    elif defect == "category": raw[2][0, 1] = raw[2][1, 0] = -1
    elif defect == "float": raw = (raw[0], raw[1], raw[2].astype(float))
    else: raw = (raw[0], raw[1], raw[2][:-1, :-1])
    _assert_search_parity(raw, spec, diagnostic=diagnostic)


def test_unknown_element_or_bond_schema_falls_back_without_inventing_rank():
    spec = GraphSpec(max_nodes=1, atomic_numbers=(0, 200))
    raw = (np.array([1]), np.array([2]), np.zeros((1, 1), dtype=np.int64))
    _assert_search_parity(raw, spec)
    spec = GraphSpec(max_nodes=2, bond_names=("NONE", "QUADRUPLE"))
    raw = _raw(spec, [6, 6], [(0, 1, "QUADRUPLE")])
    _assert_search_parity(raw, spec)


def test_fast_candidate_has_one_detect_one_sanitize_and_no_unused_hash_or_valence_loop(monkeypatch):
    spec = GraphSpec(max_nodes=8)
    raw = _raw(spec, [8, 6], [(0, 1, "TRIPLE")], slots=[2, 7])
    expected = diagnose_graph(*raw, spec)
    calls = {"detect": 0, "sanitize": 0}
    detect, sanitize = Chem.DetectChemistryProblems, Chem.SanitizeMol
    def observed_detect(*args, **kwargs):
        calls["detect"] += 1
        return detect(*args, **kwargs)
    def observed_sanitize(*args, **kwargs):
        calls["sanitize"] += 1
        return sanitize(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Fast candidate attempted a full diagnostic or eager graph hash")
    monkeypatch.setattr(Chem, "DetectChemistryProblems", observed_detect)
    monkeypatch.setattr(Chem, "SanitizeMol", observed_sanitize)
    monkeypatch.setattr(fast, "reference_diagnose", forbidden)
    monkeypatch.setattr(fast, "graph_identity", forbidden)
    report = fast.BondOnlyDiagnostic(raw, spec).diagnose(raw[2])
    assert {field: report.get(field) for field in fast.SEARCH_FIELDS} == {
        field: expected.get(field) for field in fast.SEARCH_FIELDS}
    assert calls == {"detect": 1, "sanitize": 1}
    assert "graph_id" not in report and "valence_checks" not in report


def test_graph_identity_is_lazy_and_exact_when_explicitly_requested():
    spec = GraphSpec(max_nodes=4)
    raw = _encoded("CO", spec)
    diagnostic = fast.BondOnlyDiagnostic(raw, spec)
    assert "graph_id" not in diagnostic.diagnose(raw[2])
    assert diagnostic.diagnose(raw[2], include_graph_id=True)["graph_id"] == graph_identity(*raw, spec)
