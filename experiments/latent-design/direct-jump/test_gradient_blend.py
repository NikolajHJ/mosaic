import json

import numpy as np
import pytest

from gradient_blend import blend_gradients, blend_three, calibrate, calibrate_three


def gradient(atom, bond):
    return (np.array([atom], dtype=np.float32), np.array([bond], dtype=np.float32))


def test_orthogonal_gradients_calibrate_magnitude_and_balance_before_repair():
    p = gradient([1, -1], [0, 0])
    n = gradient([0, 0], [3, -3])
    panel = calibrate([(p, n), (tuple(x * 2 for x in p), tuple(x * 2 for x in n))])
    assert panel["protenix_weight"] == pytest.approx(1 / np.sqrt(2))
    assert panel["nesso_weight"] == pytest.approx(1 / np.sqrt(18))
    assert panel["valid_nonzero_pairs"] == 2
    for mode in ("fixed", "adaptive"):
        rows, info = blend_gradients(p, n, panel, mode)
        assert info["combined_to_protenix"] == pytest.approx(1.)
        assert info["nesso_to_protenix_contribution"] == pytest.approx(1.)
        assert info["centered_combined_to_protenix"] == pytest.approx(1.)
        assert info["cosine"] == pytest.approx(0.)
        assert all(x.dtype == np.float32 for x in rows)
    stronger_n = tuple(x * 9 for x in n)
    _, fixed = blend_gradients(p, stronger_n, panel, "fixed")
    _, adaptive = blend_gradients(p, stronger_n, panel, "adaptive")
    assert fixed["nesso_to_protenix_contribution"] == pytest.approx(9.)
    assert adaptive["nesso_to_protenix_contribution"] == pytest.approx(1.)
    assert adaptive["combined_to_protenix"] == pytest.approx(1.)


def test_fixed_coefficients_differentiate_weighted_native_losses():
    def scores(rows):
        a, b = rows
        return np.sum(a*a) - np.sum(b), np.sum(b*b) + 3*np.sum(a)

    rows = gradient([.3, .7], [.2, .8])
    a, b = rows
    p, n = (2*a, -np.ones_like(b)), (3*np.ones_like(a), 2*b)
    panel = calibrate([(p, n)])
    combined, metrics = blend_gradients(p, n, panel, "fixed")
    delta = gradient([.02, -.02], [-.01, .01])
    minus = tuple(x.astype(np.float64) - d for x, d in zip(rows, delta))
    plus = tuple(x.astype(np.float64) + d for x, d in zip(rows, delta))
    difference = (np.asarray(scores(plus)) - scores(minus)) / 2
    predicted = sum(np.vdot(g, d) for g, d in zip(combined, delta))
    assert difference @ [panel["protenix_weight"], panel["nesso_weight"]] == pytest.approx(predicted)
    assert metrics["centered_cosine"] == pytest.approx(0.)
    assert metrics["centered_nesso_to_protenix_contribution"] != pytest.approx(1.)


def test_control_zero_and_opposing_gradients_yield_finite_reports():
    p = gradient([1, -1], [1, -1])
    negative = tuple(-x for x in p)
    zero = tuple(x * 0 for x in p)
    panel = calibrate([(p, negative)])
    assert panel["calibration_status"] == "near_cancellation_limited"
    g, info = blend_gradients(p, negative, panel, "adaptive")
    assert info["status"] == "near_cancellation_limited"
    assert all((x == 0).all() for x in g)
    g, info = blend_gradients(p, None, panel, "protenix")
    assert all(np.array_equal(x, y) for x, y in zip(g, p))
    assert info["nesso_norm"] is None
    assert info["combined_to_protenix"] == pytest.approx(1.)
    flat = calibrate([(zero, zero)])
    assert flat["calibration_status"] == "no_nonzero_pairs_unscaled"
    for pair in ((zero, p), (p, zero), (zero, zero)):
        g, info = blend_gradients(*pair, flat, "adaptive")
        assert all(np.isfinite(x).all() for x in g)
        json.dumps(info, allow_nan=False)
    json.dumps(flat, allow_nan=False)


def test_incompatible_or_nonfinite_inputs_fail():
    p = gradient([1, -1], [1, -1])
    with pytest.raises(ValueError, match="shapes"):
        calibrate([(p, gradient([1, 2, 3], [2, 3]))])
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        calibrate([(p, gradient([np.nan, 0], [1, 0]))])


def test_three_head_fixed_weights_target_shares_and_sum_norm_on_seven_probes():
    p = gradient([1., -1.], [0., 0.])
    n = gradient([0., 0.], [3., -3.])
    c = gradient([2., 2.], [0., 0.])
    triples = [tuple(tuple(x * scale for x in head) for head in (p, n, c))
               for scale in (1., 2., 3., 4., 5., 6., 7.)]
    panel = calibrate_three(triples, (.35, .35, .30))
    out, info = blend_three(p, n, c, panel)
    assert panel["valid_nonzero_triples"] == 7
    assert panel["calibration_status"] == "calibrated"
    assert all(v.dtype == np.float32 for v in out)
    assert [info[k + "_share"] for k in ("protenix", "nesso", "confidence")] == pytest.approx([.35, .35, .30])
    assert info["combined_to_protenix"] == pytest.approx(1)
    assert info["centered_confidence_share"] == 0
    assert info["centered_protenix_confidence_cosine"] is None
    assert panel["ranges"]["combined_to_protenix"]["median"] == pytest.approx(1)
    expected_weights = np.array([.35, .35 / 3, .30 / 2]) / np.linalg.norm([.35, .35, .30])
    assert [info[k + "_weight"] for k in ("protenix", "nesso", "confidence")] == pytest.approx(expected_weights)
    json.dumps(panel, allow_nan=False)


def test_three_head_signed_loss_uses_gradient_of_negative_log_binder_probability():
    a, b = gradient([.2, .8], [.35, .65])
    def values(a, b):
        logit = 2*a[0, 1] + b[0, 0]
        return np.array([np.sum(a*a), np.sum(b*b), np.logaddexp(0., -logit)])

    multiplier = -1 / (1 + np.exp(2*a[0, 1] + b[0, 0]))
    p, n = (2*a, np.zeros_like(b)), (np.zeros_like(a), 2*b)
    c = gradient([0., 2*multiplier], [multiplier, 0.])
    fit = calibrate_three([(p, n, c)], shares=(.4, .4, .2))
    out, metrics = blend_three(p, n, c, fit)
    da, db = gradient([.3, -.3], [.25, -.25])
    eps = 1.e-4
    a, b = a.astype(np.float64), b.astype(np.float64)
    diff = (values(a + eps*da, b + eps*db) - values(a - eps*da, b - eps*db)) / (2*eps)
    assert sum([np.vdot(out[0], da), np.vdot(out[1], db)]) == pytest.approx(
        diff @ [fit[k + "_weight"] for k in ("protenix", "nesso", "confidence")], abs=1e-7)
    assert metrics["confidence_share"] == pytest.approx(.2)


def test_fixed_weights_target_mean_contributions_when_probes_differ():
    p = gradient([0., 1.], [0., 1.])
    panel = [(p, tuple(x * a for x in p), tuple(x * c for x in p))
             for a, c in ((.1, 4.), (.3, .3), (5., 3.), (1., 1.), (3., .5), (.3, 2.), (1., 5.))]
    fit = calibrate_three(panel)
    for key, target in (("protenix_share", .35), ("nesso_share", .35), ("confidence_share", .30)):
        assert np.mean([sample[key] for sample in fit["samples"]]) == pytest.approx(target, abs=1.e-7)
    assert fit["ranges"]["combined_to_protenix"]["median"] == pytest.approx(1.)


def test_three_head_handles_cancellation_flat_and_incompatible_inputs():
    p = gradient([0., 1.], [0., 1.])
    negative = tuple(-x for x in p)
    zero = tuple(0*x for x in p)
    fit = calibrate_three([(p, negative, negative)], shares=(.5, .25, .25))
    assert fit["near_cancellation_limited"]
    assert fit["ranges"]["combined_to_protenix"]["median"] == 0
    flat = calibrate_three([(p, p, zero)])
    assert flat["calibration_status"] == "unmeasured_head_ratio"
    assert flat["valid_nonzero_triples"] == 0
    assert flat["samples"][0]["confidence_share"] == 0
    json.dumps(calibrate_three([(zero, zero, zero)]), allow_nan=False)
    with pytest.raises(ValueError, match="shares"):
        calibrate_three([(p, p, p)], shares=(.35, .30, .30))
    with pytest.raises(ValueError, match="shapes"):
        calibrate_three([(p, p, gradient([1, 2, 3], [2, 3]))])
