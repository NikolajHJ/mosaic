"""Allocated-HPC-only fixtures for additive error propagation and strict gates.

Synthetic tensor/number examples establish engineering behavior, not native
Boltz readiness or retrospective approval of a failed optimization trajectory.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz.additive_fd import (
    ANALYTIC_KIND, DIRECTIONS, EPSILONS, PARITY_VERSION, assess_row, certify,
    exact_recomposition_proof, grouped_native_expression, primitive_weights,
)


def config():
    return SimpleNamespace(affinity_regression_weight=1., affinity_logit_weight=1.,
        contact_weight=2.5, confidence_weight=2.5,
        confidence_weights={"iptm": 1., "complex_iplddt": .3, "complex_ipde": .3,
                            "complex_pde": .1, "confidence_score": .3, "ptm": .1})


def proof(value=0., derivative=1.):
    left = {"scalar": torch.tensor(value, dtype=torch.float64),
            "reverse": torch.tensor([derivative], dtype=torch.float32),
            "jvp": torch.tensor(derivative, dtype=torch.float64)}
    return exact_recomposition_proof(left, {key: tensor.clone() for key, tensor in left.items()})


def row(epsilon=1e-4, direction="gradient", *, observed_a=1001., observed_b=999.):
    # Strongly canceling native tangents 1000 - 999 = 1. Both primitive
    # secants remain within their original tolerance; the net FD can differ.
    fields = lambda derivative, secant: {"analytic": derivative,
        "f_plus": epsilon * secant, "f_minus": -epsilon * secant,
        "loss_dtype_plus": "torch.float64", "loss_dtype_minus": "torch.float64"}
    return {"epsilon": epsilon, "direction": direction, "analytic_kind": ANALYTIC_KIND,
        "direction_sha256": str(DIRECTIONS.index(direction) + 1) * 64,
        "primitives": {"a": fields(1000., observed_a), "b": fields(999., observed_b)},
        "aggregate": {"analytic": 1., "reverse_projection": 1., "baseline_loss": 0.,
            "f_plus": epsilon * observed_a - epsilon * observed_b,
            "f_minus": -epsilon * observed_a + epsilon * observed_b,
            "loss_dtype_plus": "torch.float64", "loss_dtype_minus": "torch.float64"},
        "recomposition": proof()}


def test_declared_dbmol_weights_and_scaling_preserve_native_signs():
    cfg = config()
    assert primitive_weights(cfg, "dbmol_affinity_loss") == {
        "affinity_pred_value": 1., "affinity_logits_binary": -1.}
    assert primitive_weights(cfg, "dbmol_contact_loss") == {
        "dbmol_contact_pocket_loss": .5, "dbmol_contact_anchor_loss": .5}
    objective = SimpleNamespace(output="dbmol_total_loss", direction="maximize", scale=2., offset=3.)
    weights = primitive_weights(cfg, "combined", objective=objective, repair_weight=.1)
    assert len(weights) == 11
    assert weights["affinity_pred_value"] == -.5 and weights["affinity_logits_binary"] == .5
    assert weights["dbmol_contact_anchor_loss"] == -.625
    assert weights["iptm"] == 1.25 and weights["complex_ipde"] == -.375
    assert weights["repair_nll"] == .1  # Added after binding sign/scale.
    with pytest.raises(ValueError, match="positive fixed"):
        primitive_weights(cfg, "combined", repair_weight=0.)
    with pytest.raises(ValueError, match="Unknown"):
        primitive_weights(cfg, "some_new_loss")
    cfg.confidence_weights["unexpected"] = .2
    with pytest.raises(ValueError, match="six native"):
        primitive_weights(cfg, "combined")


def test_grouped_expression_preserves_native_operation_order_and_both_ad_modes():
    from latentmol.boltz.dbmol_objective import reduce_objective
    from latentmol.multimodel.config import Objective
    from latentmol.multimodel.engine import scalar_loss
    cfg = config()
    cfg.objective_arithmetic = "contact_head_and_assembly_float64_v1"
    objective = Objective(output="dbmol_total_loss", direction="minimize", offset=.4, scale=2.)
    weights = primitive_weights(cfg, "combined", objective=objective, repair_weight=.02575)
    point = torch.tensor(.2, requires_grad=True)
    def expressions(value):
        values = {key: ((index + 1) * value).sin() for index, key in enumerate(weights)}
        values["dbmol_contact_pocket_loss"] = values["dbmol_contact_pocket_loss"].double()
        values["dbmol_contact_anchor_loss"] = values["dbmol_contact_anchor_loss"].double()
        contact = (values["dbmol_contact_pocket_loss"] + values["dbmol_contact_anchor_loss"]) / 2
        outputs = reduce_objective({**values, "dbmol_contact_loss": contact}, cfg)
        native = scalar_loss(outputs, objective)[0] + .02575 * values["repair_nll"]
        grouped = grouped_native_expression(values, cfg, "combined", objective=objective, repair_weight=.02575)
        return native, grouped
    native, grouped = expressions(point)
    original_gradient, = torch.autograd.grad(native, point, retain_graph=True)
    grouped_gradient, = torch.autograd.grad(grouped, point)
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        left, right = expressions(torch.autograd.forward_ad.make_dual(point.detach(), torch.ones_like(point)))
        native_jvp = torch.autograd.forward_ad.unpack_dual(left).tangent
        grouped_jvp = torch.autograd.forward_ad.unpack_dual(right).tangent
    observed = exact_recomposition_proof(
        {"scalar": native, "reverse": original_gradient, "jvp": native_jvp},
        {"scalar": grouped, "reverse": grouped_gradient, "jvp": grouped_jvp})
    assert observed["passed"] and observed["version"] == PARITY_VERSION
    assert all(item["exact_equal"] for item in observed["rows"].values())


def test_cancellation_is_measured_and_every_primitive_still_has_to_pass():
    result = assess_row({"a": 1., "b": -1.}, row(), rtol=.05, atol=1e-6)
    assert result["passed"]
    assert result["aggregate"]["original_net_passed"] is False
    assert result["aggregate"]["cancellation_condition"] == 1999.
    assert all(item["passed"] for item in result["primitives"].values())
    assert result["aggregate"]["absolute_error"] <= (
        result["aggregate"]["weighted_measured_error_bound"] + result["aggregate"]["assembly_fd_bound"])
    # Net agreement does not rescue two equally wrong primitives.
    bad = row(observed_a=1200., observed_b=1199.)
    result = assess_row({"a": 1., "b": -1.}, bad, rtol=.05, atol=1e-6)
    assert result["aggregate"]["original_net_passed"]
    assert result["passed"] is False


@pytest.mark.parametrize("mutation", ["missing", "unknown", "nan", "zero_weight", "wrong_kind"])
def test_missing_unknown_nonfinite_and_undeclared_values_never_pass(mutation):
    observed, weights = row(), {"a": 1., "b": -1.}
    if mutation == "missing": observed["primitives"].pop("b")
    if mutation == "unknown": observed["primitives"]["extra"] = observed["primitives"]["a"]
    if mutation == "nan": observed["primitives"]["a"]["analytic"] = float("nan")
    if mutation == "zero_weight": weights["a"] = 0.
    if mutation == "wrong_kind": observed["analytic_kind"] = "summed_fp32_vjp"
    with pytest.raises(ValueError): assess_row(weights, observed, rtol=.05, atol=1e-6)


@pytest.mark.parametrize("mutation", ["no_proof", "wrong_gradient", "wrong_jvp", "wrong_scalar", "zero_signal"])
def test_strict_independent_recomposition_and_jvp_are_not_optional(mutation):
    observed = row()
    if mutation == "no_proof": observed["recomposition"] = {"passed": True}
    if mutation == "wrong_gradient": observed["recomposition"]["rows"]["reverse"]["exact_equal"] = False
    if mutation == "wrong_jvp": observed["aggregate"]["reverse_projection"] = 2.
    if mutation == "wrong_scalar": observed["aggregate"]["f_plus"] += 1.
    if mutation == "zero_signal": observed["primitives"]["a"].update(analytic=0., f_plus=0., f_minus=0.)
    assert not assess_row({"a": 1., "b": -1.}, observed, rtol=.05, atol=1e-6)["passed"]


def test_all_primitives_need_the_same_neighboring_scales_not_independent_favorable_ones():
    directions = {name: str(index + 1) * 64 for index, name in enumerate(DIRECTIONS)}
    rows = [row(epsilon, name) for name in directions for epsilon in EPSILONS]
    result = certify({"a": 1., "b": -1.}, rows, directions=directions, rtol=.05, atol=1e-6)
    assert result["passed"] and all(item["passed"] for item in result["direction_results"].values())
    # Each primitive has favorable scales in isolation, but never together.
    for observed in rows:
        if observed["epsilon"] in EPSILONS[:2]:
            changed = "a"
        else:
            changed = "b"
        observed["primitives"][changed].update(f_plus=0., f_minus=0.)
        for side in ("f_plus", "f_minus"):
            observed["aggregate"][side] = observed["primitives"]["a"][side] - observed["primitives"]["b"][side]
    result = certify({"a": 1., "b": -1.}, rows, directions=directions, rtol=.05, atol=1e-6)
    assert not result["passed"]
    assert not any(item["passed"] for item in result["rows"])


def test_duplicate_missing_or_replaced_probe_is_rejected():
    directions = {name: str(index + 1) * 64 for index, name in enumerate(DIRECTIONS)}
    rows = [row(epsilon, name) for name in directions for epsilon in EPSILONS]
    for broken in (rows[:-1], rows + [deepcopy(rows[0])]):
        with pytest.raises(ValueError):
            certify({"a": 1., "b": -1.}, broken, directions=directions, rtol=.05, atol=1e-6)
    rows[0]["direction_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        certify({"a": 1., "b": -1.}, rows, directions=directions, rtol=.05, atol=1e-6)


@pytest.mark.parametrize("mutation", ["baseline", "primitive_tangent", "aggregate_tangent", "reverse_projection", "proof"])
def test_one_direction_cannot_mix_baselines_tangents_or_proofs_across_scales(mutation):
    directions = {name: str(index + 1) * 64 for index, name in enumerate(DIRECTIONS)}
    rows = [row(epsilon, name) for name in directions for epsilon in EPSILONS]
    changed = rows[1]
    if mutation == "baseline": changed["aggregate"]["baseline_loss"] = .01
    if mutation == "primitive_tangent": changed["primitives"]["a"]["analytic"] = 1000.01
    if mutation == "aggregate_tangent": changed["aggregate"]["analytic"] = 1.01
    if mutation == "reverse_projection": changed["aggregate"]["reverse_projection"] = 1.01
    if mutation == "proof": changed["recomposition"] = proof(derivative=1.01)
    with pytest.raises(ValueError, match="changed across epsilon"):
        certify({"a": 1., "b": -1.}, rows, directions=directions, rtol=.05, atol=1e-6)


@pytest.mark.parametrize("mutation", ["baseline", "reverse_witness"])
def test_all_directions_must_share_the_same_frozen_baseline_and_reverse_witness(mutation):
    directions = {name: str(index + 1) * 64 for index, name in enumerate(DIRECTIONS)}
    rows = [row(epsilon, name) for name in directions for epsilon in EPSILONS]
    for changed in rows:
        if changed["direction"] == "random_1":
            if mutation == "baseline":
                changed["aggregate"]["baseline_loss"] = .01
                changed["recomposition"] = proof(value=.01)
            else:
                changed["recomposition"] = proof(derivative=1.01)
    with pytest.raises(ValueError, match="across declared directions"):
        certify({"a": 1., "b": -1.}, rows, directions=directions, rtol=.05, atol=1e-6)


@pytest.mark.parametrize("part,field,value", [
    ("scalar", "dtype", "torch.float32"), ("jvp", "dtype", "torch.float32"),
    ("reverse", "dtype", "torch.float64"), ("scalar", "shape", [2]),
    ("jvp", "shape", [1, 2]), ("reverse", "shape", []), ("reverse", "shape", [0]),
    ("reverse", "shape", [-1]), ("reverse", "shape", [True]),
    ("reverse", "shape", [1.]), ("scalar", "shape", "[]"),
    ("scalar", "value", True), ("jvp", "value", float("nan")),
    ("scalar", "sha256", "f" * 64),
])
def test_certificate_rejects_wrong_dtype_empty_or_malformed_shape_and_stale_scalar_hash(part, field, value):
    observed = row()
    for side in ("native", "recomposed"):
        observed["recomposition"]["rows"][part][side][field] = value
    assert not assess_row({"a": 1., "b": -1.}, observed, rtol=.05, atol=1e-6)["passed"]


def test_certificate_rejects_a_changed_scalar_value_even_if_both_proof_sides_agree():
    observed = row()
    observed["aggregate"]["baseline_loss"] = .1
    for side in ("native", "recomposed"):
        observed["recomposition"]["rows"]["scalar"][side]["value"] = .1
    # The original byte witness still represents zero, not the new .1 claim.
    assert not assess_row({"a": 1., "b": -1.}, observed, rtol=.05, atol=1e-6)["passed"]
