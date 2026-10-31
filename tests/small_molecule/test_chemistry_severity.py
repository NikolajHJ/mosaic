"""Allocated CPU checks of opt-in severity-aware detached target preparation."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry import (
    GraphLogits, failure_aware_repair_loss, mosaic_spec, prepare_failure_guidance,
)
from mosaic.small_molecule.chemistry._reference.probability_repair import (
    ProbabilityRepairConfig,
)
from mosaic.small_molecule.chemistry._reference.repair_progress import (
    ProgressSearchConfig, search_progress_repair,
)
from mosaic.small_molecule.chemistry._reference.representation import GraphSpec
from mosaic.small_molecule.chemistry import severity
from mosaic.small_molecule.chemistry.severity import (
    SeverityProgressConfig, search_severity_progress,
)


def source(elements, edges):
    spec = mosaic_spec(len(elements))
    atoms = np.asarray([spec.atomic_numbers.index(a) for a in elements], dtype=np.int64)
    charges = np.zeros(len(elements), dtype=np.int64)
    bonds = np.zeros((len(elements), len(elements)), dtype=np.int64)
    for i, j, name in edges:
        bonds[i, j] = bonds[j, i] = spec.bond_names.index(name)
    probabilities = []
    for labels, size in zip((atoms, charges, bonds),
                            (len(spec.atomic_numbers), 1, len(spec.bond_names))):
        p = np.full(labels.shape + (size,), 0.1 / size)
        np.put_along_axis(p, labels[..., None], 0.9 + 0.1 / size, axis=-1)
        if size == 1:
            p[...] = 1.0
        probabilities.append(p)
    return spec, (atoms, charges, bonds), tuple(probabilities)


def clique():
    return source([6] * 7, [(i, j, "SINGLE") for i in range(7) for j in range(i + 1, 7)])


def decoded_clique():
    atoms = jnp.zeros((7, 6), jnp.float32).at[:, 0].set(2.0)
    bonds = jnp.zeros((21, 5), jnp.float32).at[:, 1].set(2.0)
    return GraphLogits(atoms, bonds)


@pytest.mark.parametrize("fields", [
    {"version": "unknown"}, {"max_edits": 0}, {"max_edits": True},
    {"max_edits": 301}, {"max_checks": 0}, {"max_checks": 4097},
    {"candidates_per_step": 0}, {"candidates_per_step": 1.5},
])
def test_invalid_or_unbounded_policy_rejected(fields):
    with pytest.raises(ValueError):
        SeverityProgressConfig(**fields)


def test_severity_detects_progress_that_error_count_search_cannot():
    spec, raw, p = clique()
    old = search_progress_repair(raw, p, spec, ProgressSearchConfig(
        max_edits=1, checks_per_depth=16, beam_width=1,
    ))
    assert old["status"] == "no_progress"
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig(max_edits=1))
    assert result["status"] == "partial"
    witness = result["witness"]
    before, after = dict(witness.before), dict(witness.after)
    assert before["AtomValenceException"] == after["AtomValenceException"]
    assert after["severity_total_half_units"] < before["severity_total_half_units"]
    assert result["policy"]["partial_is_valid"] is False
    witness.verify(spec)


def test_all_retained_stages_preserve_identities_connectivity_and_inputs():
    spec, raw, p = clique()
    originals = [v.copy() for v in (*raw, *p)]
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig(max_edits=3))
    assert result["status"] == "partial"
    witness = result["witness"]
    assert witness.original[0] == witness.labels[0]
    assert witness.original[1] == witness.labels[1]
    before, after = dict(witness.before), dict(witness.after)
    assert after["extra_components"] <= before["extra_components"] == 0
    assert all(after.get(k, 0) <= value for k, value in before.items())
    for wanted, actual in zip(originals, (*raw, *p)):
        np.testing.assert_array_equal(wanted, actual)


def test_complete_witness_requires_actual_native_and_schema_validity():
    spec, raw, p = source([6, 6, 9], [
        (0, 1, "SINGLE"), (0, 2, "SINGLE"), (1, 2, "SINGLE"),
    ])
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig())
    assert result["status"] == "valid"
    assert result["witness"].kind == "valid"
    result["witness"].verify(spec)


def test_bond_deletion_cannot_split_an_infeasible_fixed_identity_chain():
    spec, raw, p = source([9, 9, 9], [(0, 1, "SINGLE"), (1, 2, "SINGLE")])
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig())
    assert result["status"] == "no_progress" and result["witness"] is None
    assert result["connectivity_rejections"] > 0


def test_new_native_error_category_is_rejected(monkeypatch):
    spec, raw, p = clique()
    original = severity._observe
    calls = []

    def observe(*args):
        diagnostic, vector = original(*args)
        calls.append(True)
        if len(calls) > 1:
            vector = {**vector, "new_native_error": 1}
        return diagnostic, vector

    monkeypatch.setattr(severity, "_observe", observe)
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig(max_checks=3))
    assert result["status"] == "no_progress"
    assert result["native_guard_rejections"] == 2
    assert result["checks"] == len(calls) == 3


def test_roundtrip_failure_is_never_reported_as_valid(monkeypatch):
    spec, raw, p = source([6, 6, 9], [
        (0, 1, "SINGLE"), (0, 2, "SINGLE"), (1, 2, "SINGLE"),
    ])
    monkeypatch.setattr(severity, "encode_smiles", lambda *a: {"smiles": "wrong"})
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig())
    assert result["valid_witnesses"] == 0
    assert result["status"] != "valid"


def test_actual_native_assessment_budget_includes_baseline():
    spec, raw, p = clique()
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig(
        max_edits=6, max_checks=2,
    ))
    assert result["checks"] == 2 and result["budget_exhausted"]
    assert result["status"] == "partial"


def test_valid_source_has_no_self_target():
    spec, raw, p = source([6, 6, 6], [(0, 1, "SINGLE"), (1, 2, "SINGLE")])
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig())
    assert result["status"] == "already_valid" and result["witness"] is None


def test_unknown_frozen_rule_is_unassessed_not_given_a_zero_cap():
    spec = GraphSpec(max_nodes=1, atomic_numbers=(0, 54), formal_charges=(0,))
    raw = (np.asarray([1]), np.asarray([0]), np.asarray([[0]]))
    p = (np.asarray([[0.1, 0.9]]), np.asarray([[1.0]]),
         np.full((1, 1, 5), 0.2))
    result = search_severity_progress(raw, p, spec, SeverityProgressConfig())
    assert result["status"] == "unsupported" and result["checks"] == 0


def test_opt_in_preparation_routes_partial_target_without_an_atom_gradient():
    state, spec = decoded_clique(), mosaic_spec(7)
    prep = prepare_failure_guidance(
        state.decoded(), spec, search=ProbabilityRepairConfig(max_edits=1, checks_per_depth=8),
        severity_progress={"max_edits": 1},
    )
    assert int(prep.arrays["route"][0]) == 3
    assert prep.receipt["severity_progress"]["version"] == "severity-progress-v1"
    value, gradient = eqx.filter_value_and_grad(
        lambda trial: failure_aware_repair_loss(trial.decoded(), prep, spec)[0]
    )(state)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient.bonds)).all()
    assert np.count_nonzero(np.asarray(gradient.atoms)) == 0
    assert np.linalg.norm(np.asarray(gradient.bonds)) > 0
    changed = np.argwhere(np.asarray(prep.arrays["changed"])[0])[0]
    pair = [(i, j) for i in range(7) for j in range(i + 1, 7)].index(tuple(changed))
    direction = jnp.zeros_like(state.bonds).at[pair, 0].set(1).at[pair, 1].set(-1)
    analytic = float(jnp.sum(gradient.bonds * direction))
    points = [eqx.tree_at(lambda v: v.bonds, state, state.bonds + sign * 0.001 * direction)
              for sign in (-1, 1)]
    measured = [float(failure_aware_repair_loss(v.decoded(), prep, spec)[0]) for v in points]
    assert abs(analytic) > 0.1
    np.testing.assert_allclose((measured[1] - measured[0]) / 0.002,
                               analytic, rtol=0.01, atol=0.001)
    identity = prep.identity
    changed_trial = eqx.tree_at(lambda v: v.bonds, state, -state.bonds)
    failure_aware_repair_loss(changed_trial.decoded(), prep, spec)
    assert prep.identity == identity
    prep.verify(spec)


def test_target_tampering_is_rejected():
    state, spec = decoded_clique(), mosaic_spec(7)
    prep = prepare_failure_guidance(
        state.decoded(), spec, search=ProbabilityRepairConfig(max_edits=1, checks_per_depth=8),
        severity_progress={"max_edits": 1},
    )
    broken = replace(prep, arrays={**prep.arrays, "target_atoms": prep.arrays["target_atoms"].at[0, 0].set(4)})
    with pytest.raises(ValueError, match="identity"):
        broken.verify(spec)


def test_absent_option_and_explicit_none_have_identical_historical_receipts():
    state, spec = decoded_clique(), mosaic_spec(7)
    options = {"search": ProbabilityRepairConfig(max_edits=1, checks_per_depth=8)}
    old = prepare_failure_guidance(state.decoded(), spec, **options)
    none = prepare_failure_guidance(state.decoded(), spec, severity_progress=None, **options)
    assert old.identity == none.identity and old.receipt == none.receipt
    assert "severity_progress" not in old.receipt
    for key in old.arrays:
        np.testing.assert_array_equal(old.arrays[key], none.arrays[key])
