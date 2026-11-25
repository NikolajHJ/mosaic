import sys

import numpy as np
import pytest

import repair_signal as repair
from latentmol.guidance import repair_proposals
from latentmol.representation import GraphSpec


def chain(*, gap=False):
    atom = np.full((25, 6), .01, dtype=np.float32)
    atom[:, 0] = .95
    bond = np.full((300, 5), .01, dtype=np.float32)
    labels = (repair.RIGHT == repair.LEFT + 1)
    if gap:
        labels &= (repair.LEFT != 11)
    bond[np.arange(300), labels.astype(int)] = .96
    return atom, bond


def test_direct_column_mapping_and_independent_bonds():
    a, b = chain()
    # An S/F tie breaks in direct category order; shared schema puts F before S.
    a[0] = [0, 0, 0, .5, .5, 0]
    spec = GraphSpec(max_nodes=25)
    raw, (pa, pq, pb), _ = repair._bridge(a, b, spec)
    assert raw[0][0] == spec.atomic_numbers.index(16)
    assert (pa[:, 0] == 0).all()
    assert np.allclose(pa[:, [spec.atomic_numbers.index(z) for z in repair.ATOMIC_NUMBERS]], a)
    assert (pq[:, spec.formal_charges.index(0)] == 1).all()
    assert (pb[repair.LEFT, repair.RIGHT] == b).all()
    assert (pb[repair.RIGHT, repair.LEFT] == b).all()
    assert (pb[np.arange(25), np.arange(25), 0] == 1).all()


def test_fixed_repair_probability_derivative_and_zero_cases(monkeypatch):
    atom, bond = chain(gap=True)
    loss, da, db, info = repair.evaluate_repair(atom, bond)
    assert info["status"] == "verified_witness" and info["smiles"] is None
    assert info["component_count"] == 2 and info["repair_edits"] == 1
    assert info["repair_smiles"] and info["normalization_decisions"] == 350
    atom_label, bond_label = da.argmin(1), db.argmin(1)
    assert np.count_nonzero(da) == 25 and np.count_nonzero(db) == 300
    expected = -(np.log(atom[np.arange(25), atom_label]).sum(dtype=float)
                 + np.log(bond[np.arange(300), bond_label]).sum(dtype=float)) / 350
    assert loss == pytest.approx(expected, rel=2e-7)
    # Hold the witness fixed; perturb two coordinates on its added bond's simplex.
    index, = np.flatnonzero(bond_label != bond.argmax(1))
    target = bond_label[index]
    def score(delta):
        p = bond.astype(np.float64)
        p[index, target] += delta
        p[index, 0] -= delta
        return -np.log(p[np.arange(300), bond_label]).sum() / 350
    epsilon = 1e-5
    assert (score(epsilon) - score(-epsilon)) / (2*epsilon) == pytest.approx(
        float(db[index, target] - db[index, 0]), rel=1e-5)
    monkeypatch.setattr(repair_proposals, "search_with_proposals", lambda *args: {
        "weighted_witness": None, "status": "censored", "checks": 2})
    for graph, status in ((chain(), "already_valid"), ((atom, bond), "no_witness")):
        loss, da, db, info = repair.evaluate_repair(*graph)
        assert info["status"] == status and loss == 0
        assert not da.any() and not db.any()
        assert info["search_checks"] == (0 if status == "already_valid" else 2)


def test_reject_changed_atoms_even_if_witness_valid(monkeypatch):
    spec = GraphSpec(max_nodes=25)
    raw, _, _ = repair._bridge(*chain(), spec)
    raw[0][0] = spec.atomic_numbers.index(7)
    witness = dict(labels=dict(zip(("atoms", "charges", "bonds"), raw)), smiles="N" + "C"*24, edit_count=1)
    monkeypatch.setattr(repair_proposals, "search_with_proposals", lambda *args: {
        "weighted_witness": witness, "status": "witness_found", "checks": 1})
    with pytest.raises(ValueError, match="changed fixed atoms"):
        repair.evaluate_repair(*chain(gap=True))


def test_smiles_that_fail_reparse_are_not_eligible(monkeypatch):
    import study

    monkeypatch.setattr(study, "hard_smiles", lambda g: "C(")
    monkeypatch.setattr(repair_proposals, "search_with_proposals", lambda *args: {
        "weighted_witness": None, "status": "censored", "checks": 1})
    _, _, _, info = repair.evaluate_repair(*chain())
    assert not info["valid"] and info["smiles"] is None
    assert info["failure_reason"] == "smiles_reparse" and info["status"] == "no_witness"
    assert repair.canonical_valid_smiles("C" * 23 + ".CC") is None


def test_resident_worker_reuses_process_and_exits():
    with repair.RepairOracle(python=sys.executable) as oracle:
        process = oracle.process
        for graph in (chain(), chain(gap=True)):
            local = repair.evaluate_repair(*graph)
            remote = oracle.evaluate(*graph)
            assert local[-1] == remote[-1] and local[0] == remote[0]
            for x, y in zip(local[1:3], remote[1:3]):
                np.testing.assert_array_equal(x, y)
            assert oracle.process is process
    assert process.returncode == 0
