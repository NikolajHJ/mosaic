"""Allocated CPU fixtures for the JAX port, not chemical efficacy claims.

The required readiness invocation sets MOSAIC_REQUIRE_CHEMISTRY_ORACLES=1;
missing original-source oracles then fail rather than skip the parity gate.
"""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry import (
    DecoderLogits,
    GraphLogits,
    failure_aware_repair_loss,
    mosaic_spec,
    prepare_failure_guidance,
)
from mosaic.small_molecule.chemistry._reference.probability_repair import (
    ProbabilityRepairConfig,
)
from mosaic.small_molecule.chemistry._reference.repair_progress import (
    ProgressSearchConfig,
    search_progress_repair,
)
from mosaic.small_molecule.chemistry._reference.representation import GraphSpec
from mosaic.small_molecule.chemistry.loss import (
    FailureGuidanceConfig,
    composition_statistics,
    population_js,
)
from mosaic.small_molecule.chemistry.state import ELEMENTS
from mosaic.small_molecule.chemistry.trajectory import (
    evaluate_combined,
    run_chemical_updates,
)
from mosaic.small_molecule.state import graph_from_hard, mix_with_uniform


@pytest.fixture(scope="module")
def oracles():
    path = os.environ.get("MOSAIC_CHEMISTRY_ORACLES")
    if not path:
        if os.environ.get("MOSAIC_REQUIRE_CHEMISTRY_ORACLES") == "1":
            pytest.fail("Required authenticated original-source oracles are missing")
        pytest.skip("Original-source numerical fixtures require allocated oracle stage")
    value = json.loads(Path(path).read_text())
    assert value["version"] == "dbmol-chemistry-oracles-v1"
    assert len(value["cases"]) == 12
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/mosaic/small_molecule/chemistry/_reference/source_manifest.json"
        ).read_text()
    )
    expected = {
        name: entry["source_sha256"] for name, entry in manifest["files"].items()
    }
    expected.update(manifest["oracle_source_sha256"])
    assert value["source_sha256"] == expected
    assert jax.config.jax_enable_x64, "Exact reference comparison requires CPU float64"
    return value["cases"]


def prepared_case(case):
    spec = GraphSpec.from_dict(case["spec"])
    decoded = DecoderLogits(
        *(jnp.asarray(v, dtype=jnp.float64) for v in case["logits"])
    )
    prepared = prepare_failure_guidance(
        decoded,
        spec,
        search=ProbabilityRepairConfig(**case["search"]),
        progress_search=None
        if case["progress_search"] is None
        else ProgressSearchConfig(**case["progress_search"]),
    )
    return spec, decoded, prepared


@pytest.mark.parametrize("index", range(12))
def test_original_torch_values_and_every_component_vjp(oracles, index):
    case = oracles[index]
    spec, decoded, prepared = prepared_case(case)
    total, terms = failure_aware_repair_loss(decoded, prepared, spec)
    terms = {"total": total, **terms}
    for key, expected in case["values"].items():
        np.testing.assert_allclose(
            terms[key],
            expected,
            rtol=2e-8,
            atol=1e-9,
            err_msg=case["label"] + ":" + key,
        )

        def objective(output):
            value, values = failure_aware_repair_loss(output, prepared, spec)
            return value if key == "total" else values[key]

        gradient = eqx.filter_grad(objective)(decoded)
        for actual, wanted in zip(
            (gradient.atoms, gradient.charges, gradient.bonds), case["gradients"][key]
        ):
            assert np.isfinite(np.asarray(actual)).all()
            np.testing.assert_allclose(
                actual, wanted, rtol=2e-7, atol=1e-9, err_msg=case["label"] + ":" + key
            )
    # Full valid targets replace local/cut/nonempty fallback on their rows.
    if case["label"] == "extra_valid_nll":
        assert float(terms["extra_valid_nll"]) > 0
        assert all(
            float(terms[k]) == 0
            for k in ("local_bond_risk", "component_cut_risk", "nonempty_risk")
        )


@pytest.mark.parametrize("index", range(12))
def test_empirical_statistics_and_aggregate_js_match_source(oracles, index):
    case = oracles[index]
    _, decoded, _ = prepared_case(case)
    stats = composition_statistics(decoded)
    expected = case["composition"]
    for key in ("elements", "charges", "occupied_mass"):
        np.testing.assert_allclose(stats[key], expected[key], rtol=1e-10, atol=1e-12)
    law = jnp.asarray(expected["law"])
    value, gradient = jax.value_and_grad(lambda counts: population_js(counts, law))(
        stats["elements"]
    )
    np.testing.assert_allclose(value, expected["js"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(gradient, expected["js_gradient"], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize(
    "component",
    [
        "valid_full_nll",
        "extra_valid_nll",
        "partial_edited_nll",
        "local_bond_risk",
        "component_cut_risk",
        "nonempty_risk",
    ],
)
def test_frozen_preparation_component_directional_derivatives(oracles, component):
    case = next(v for v in oracles if v["label"] == component)
    spec, decoded, prepared = prepared_case(case)

    def objective(value):
        return failure_aware_repair_loss(value, prepared, spec)[1][component]

    gradients = eqx.filter_grad(objective)(decoded)
    norm = jnp.sqrt(
        sum(
            jnp.sum(v * v)
            for v in (gradients.atoms, gradients.charges, gradients.bonds)
        )
    )
    assert float(norm) > 1e-10
    direction = jax.tree.map(lambda v: v / norm, gradients)
    values = []
    for sign in (-1, 1):
        perturbed = jax.tree.map(lambda a, d: a + sign * 1e-3 * d, decoded, direction)
        values.append(float(objective(perturbed)))
    np.testing.assert_allclose(
        (values[1] - values[0]) / 0.002, float(norm), rtol=1e-4, atol=1e-8
    )
    prepared.verify(spec)


def test_chemistry_is_not_the_old_expected_valence_proxy_and_has_no_direct_atom_cap_gradient(
    oracles,
):
    case = next(v for v in oracles if v["label"] == "local_bond_risk")
    spec, decoded, prepared = prepared_case(case)
    gradient = eqx.filter_grad(
        lambda v: failure_aware_repair_loss(v, prepared, spec)[1]["local_bond_risk"]
    )(decoded)
    assert np.count_nonzero(gradient.atoms) == np.count_nonzero(gradient.charges) == 0
    assert np.linalg.norm(gradient.bonds) > 0
    # Frozen F cap is one even if live logits are nudged toward sulfur.
    changed = eqx.tree_at(
        lambda v: v.atoms,
        decoded,
        decoded.atoms.at[..., spec.atomic_numbers.index(16)].add(10.0),
    )
    np.testing.assert_allclose(
        failure_aware_repair_loss(changed, prepared, spec)[1]["local_bond_risk"],
        failure_aware_repair_loss(decoded, prepared, spec)[1]["local_bond_risk"],
        rtol=0,
        atol=0,
    )


def test_target_and_coverage_mutations_fail_closed(oracles):
    spec, decoded, prepared = prepared_case(oracles[0])
    altered = {
        **prepared.arrays,
        "target_bonds": prepared.arrays["target_bonds"].at[0, 0, 1].set(0),
    }
    with pytest.raises(ValueError, match="identity"):
        failure_aware_repair_loss(decoded, replace(prepared, arrays=altered), spec)
    prepared.receipt["rows"][0]["base"]["status"] = "rejected_or_error"
    with pytest.raises(ValueError, match="identity"):
        failure_aware_repair_loss(decoded, prepared, spec)


def test_reference_symbol_bytes_and_full_valence_table_are_pinned():
    root = (
        Path(__file__).resolve().parents[1]
        / "src/mosaic/small_molecule/chemistry/_reference"
    )
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, entry in manifest["files"].items():
        assert (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            == entry["extracted_sha256"]
        )
    from mosaic.small_molecule.chemistry._reference.chemistry_regularization import (
        ALLOWED_VALENCES,
    )

    assert ALLOWED_VALENCES[9][-2] is None
    assert ALLOWED_VALENCES[16][0] == (2, 4, 6)
    assert (
        "chemistry_regularization_losses"
        not in (root / "chemistry_regularization.py").read_text()
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_fixed_mosaic_schema_uses_actual_order_and_preserves_positive_initial_probabilities(
    dtype,
):
    assert ELEMENTS == ("C", "N", "O", "S", "F", "Cl")
    spec = mosaic_spec(6)
    assert spec.atomic_numbers == (0, 6, 7, 8, 16, 9, 17)
    assert spec.formal_charges == (0,)
    hard = graph_from_hard(
        [0] * 6,
        [1 if j == i + 1 else 0 for i in range(6) for j in range(i + 1, 6)],
        dtype=dtype,
    )
    initial = mix_with_uniform(hard, 0.5)
    logits = GraphLogits.from_graph(initial)
    # log -> softmax is a floating-point roundtrip, not a bitwise identity.
    tolerance = 2 * np.finfo(np.asarray(initial.X).dtype).eps
    np.testing.assert_allclose(logits.graph().X, initial.X, rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        logits.graph().E_upper, initial.E_upper, rtol=0, atol=tolerance
    )
    decoded = logits.decoded()
    assert float(jnp.max(jax.nn.softmax(decoded.atoms, -1)[..., 0])) == 0
    with pytest.raises(ValueError, match="positive"):
        GraphLogits.from_graph(hard)


def test_real_four_edit_fixture_not_just_wider_one_edit_search():
    spec = GraphSpec(max_nodes=8)
    atoms = np.asarray([spec.atomic_numbers.index(z) for z in [6] * 4 + [9] * 4])
    charges = np.full(8, spec.formal_charges.index(0))
    bonds = np.zeros((8, 8), dtype=np.int64)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        bonds[i, j] = bonds[j, i] = 1
    for i in range(4):
        bonds[i, i + 4] = bonds[i + 4, i] = 3
    raw = (atoms, charges, bonds)
    p = tuple(
        (np.eye(k)[v] + 0.01) / (1 + k * 0.01)
        for v, k in zip(raw, (len(spec.atomic_numbers), len(spec.formal_charges), 5))
    )
    shallow = search_progress_repair(
        raw,
        p,
        spec,
        ProgressSearchConfig(max_edits=3, checks_per_depth=4096, beam_width=32),
    )
    deep = search_progress_repair(
        raw,
        p,
        spec,
        ProgressSearchConfig(max_edits=4, checks_per_depth=4096, beam_width=32),
    )
    assert shallow["status"] != "valid"
    assert deep["status"] == "valid"
    assert (
        np.count_nonzero(np.triu(np.asarray(deep["witness"].labels[2]) != bonds, 1))
        == 4
    )


def test_no_search_or_native_chemistry_in_differentiable_forward(oracles, monkeypatch):
    import mosaic.small_molecule.chemistry.preparation as module

    spec, decoded, prepared = prepared_case(oracles[0])

    def forbidden(*args, **kwargs):
        raise AssertionError("Detached preparation entered AD")

    monkeypatch.setattr(module, "search_atom_preserving_repair", forbidden)
    monkeypatch.setattr(module, "inspect_graph", forbidden)
    gradient = eqx.filter_grad(
        lambda v: failure_aware_repair_loss(v, prepared, spec)[0]
    )(decoded)
    assert np.isfinite(gradient.bonds).all()


def test_noop_fallback_consumes_budget_and_records_changing_reference_scope(
    tmp_path, monkeypatch
):
    import mosaic.small_molecule.chemistry.trajectory as module

    # A line-selection fixture, not a waiver of production/native derivatives.
    monkeypatch.setattr(
        module,
        "check_combined_directions",
        lambda *a, **k: [{"fixture": "constant objective"}],
    )
    graph = mix_with_uniform(graph_from_hard([0, 0], [1]), 0.5)
    initial, spec = GraphLogits.from_graph(graph), mosaic_spec(2)
    seen = []

    def prepare(logits, update):
        seen.append(update)
        return prepare_failure_guidance(logits.decoded(), spec, update=update)

    reference = np.asarray(initial.graph().X).sum(0)
    result = run_chemical_updates(
        lambda g, key: (jnp.asarray(0.0, g.X.dtype), {}),
        initial,
        prepare=prepare,
        spec=spec,
        reference=reference,
        key=jax.random.key(17),
        directory=tmp_path / "run",
        steps=3,
        step_size=0.03,
        max_backtracks=10,
        config={
            "loss": FailureGuidanceConfig(),
            "repair_weight": 0.5,
            "element_js_weight": 0.0,
        },
    )
    assert seen == [0, 1, 2, 3]
    assert result["status"] == "passed"
    assert result["counts"] == {
        "applied": 3,
        "strict_improving": 0,
        "forced_fallback": 3,
        "no_op": 3,
    }
    assert all(
        len(v["attempts"]) == 10 and v["selected_trial"] == 0 for v in result["updates"]
    )
    assert len(result["states"]) == 4
    assert all(v["preparation_identity"] for v in result["states"])


def test_native_pullback_and_chemistry_sum_match_end_to_end_autodiff():
    graph = mix_with_uniform(graph_from_hard([4, 4], [3]), 0.5)
    state, spec = GraphLogits.from_graph(graph), mosaic_spec(2)
    prepared = prepare_failure_guidance(state.decoded(), spec)
    reference = jnp.asarray([4.0, 1.0, 1.0, 0.2, 0.1, 0.1], dtype=state.atoms.dtype)
    options = {
        "loss": FailureGuidanceConfig(),
        "repair_weight": 0.5,
        "element_js_weight": 0.1,
    }
    key = jax.random.key(17)

    def native(g, *, key):
        value = (
            jnp.sum(g.X * jnp.arange(6)) * 0.01 + jnp.sum(g.E_upper * g.E_upper) * 0.2
        )
        return value, {"native_fixture": value}

    actual = evaluate_combined(
        native, state, prepared, spec, reference, key=key, config=options
    )

    def direct(logits):
        contact, _ = native(logits.graph(), key=key)
        chemistry, _ = failure_aware_repair_loss(logits.decoded(), prepared, spec)
        js = population_js(
            composition_statistics(logits.decoded())["elements"], reference
        )
        return contact + 0.5 * chemistry + 0.1 * js

    expected = eqx.filter_grad(direct)(state)
    np.testing.assert_allclose(
        actual.gradients.atoms, expected.atoms, rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        actual.gradients.bonds, expected.bonds, rtol=1e-6, atol=1e-8
    )
    reused = evaluate_combined(
        native,
        state,
        prepared,
        spec,
        reference,
        key=key,
        config=options,
        native=actual.native,
    )
    assert reused.primal == actual.primal
    with pytest.raises(ValueError, match="reuse"):
        evaluate_combined(
            native,
            state,
            prepared,
            spec,
            reference,
            key=jax.random.key(18),
            config=options,
            native=actual.native,
        )
    changed = eqx.tree_at(lambda v: v.atoms, state, state.atoms.at[0, 0].add(0.1))
    with pytest.raises(ValueError, match="reuse"):
        evaluate_combined(
            native,
            changed,
            prepared,
            spec,
            reference,
            key=key,
            config=options,
            native=actual.native,
        )


def test_nonfinite_native_outputs_remain_a_failure_not_a_fallback():
    graph = mix_with_uniform(graph_from_hard([0, 0], [1]), 0.5)
    state, spec = GraphLogits.from_graph(graph), mosaic_spec(2)
    prepared = prepare_failure_guidance(state.decoded(), spec)
    with pytest.raises(FloatingPointError):
        evaluate_combined(
            lambda g, key: (jnp.sum(g.X), {"native_hidden": jnp.asarray(float("nan"))}),
            state,
            prepared,
            spec,
            jnp.ones(6),
            key=jax.random.key(17),
            config={
                "loss": FailureGuidanceConfig(),
                "repair_weight": 0.5,
                "element_js_weight": 0.1,
            },
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_source_math_precision_preserved_inside_float64_readiness(oracles, dtype):
    case = next(v for v in oracles if v["label"] == "local_bond_risk")
    spec = GraphSpec.from_dict(case["spec"])
    decoded = DecoderLogits(*(jnp.asarray(v, dtype=dtype) for v in case["logits"]))
    prepared = prepare_failure_guidance(
        decoded, spec, search=ProbabilityRepairConfig(**case["search"])
    )
    value, _ = failure_aware_repair_loss(decoded, prepared, spec)
    assert value.dtype == jnp.float32
    gradient = eqx.filter_grad(
        lambda v: failure_aware_repair_loss(v, prepared, spec)[0]
    )(decoded)
    assert all(
        np.isfinite(np.asarray(v)).all()
        for v in (gradient.atoms, gradient.charges, gradient.bonds)
    )
