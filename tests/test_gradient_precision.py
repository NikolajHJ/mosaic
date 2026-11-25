"""Low-signal certificate regressions; execution belongs on allocated HPC."""
import pytest

from latentmol.boltz.gradient_precision import relative_ad_consistency, scalar_replay_consistency


@pytest.mark.parametrize("forward,reverse,norm", [(0., 0., 0.), (1e-17, 1e-17, 1e-16)])
def test_zero_or_consistent_small_derivative_is_valid(forward, reverse, norm):
    assert relative_ad_consistency(forward, reverse, norm)["passed"]


@pytest.mark.parametrize("forward,reverse,norm", [
    (1e-7, 0., 0.), (1e-14, -1e-14, 1e-13), (1e-17, 1e-17, 0.), (1e-8, -1e-8, 1e-7),
])
def test_wrong_small_sign_zero_or_impossible_norm_is_not_absolute_tolerance_noise(forward, reverse, norm):
    assert abs(forward - reverse) <= 1e-6
    assert not relative_ad_consistency(forward, reverse, norm)["passed"]


@pytest.mark.parametrize("values", [(float("nan"), 0., 0.), (0., float("inf"), 0.), (0., 0., -1.)])
def test_invalid_gradient_evidence_fails(values):
    with pytest.raises(ValueError):
        relative_ad_consistency(*values)


def test_scalar_replay_accepts_rounding_but_rejects_material_or_sign_changes():
    assert scalar_replay_consistency(1. + 2 * 2.**-23, 1., "torch.float32")["passed"]
    assert not scalar_replay_consistency(1. + 2.**-10, 1., "torch.float32")["passed"]
    assert not scalar_replay_consistency(-1., 1., "torch.float32")["passed"]
    assert scalar_replay_consistency(2.**-149, 0., "torch.float32")["passed"]
    assert not scalar_replay_consistency(2.**-30, 0., "torch.float32")["passed"]
    with pytest.raises(ValueError):
        scalar_replay_consistency(float("nan"), 1., "torch.float32")
    with pytest.raises(ValueError):
        scalar_replay_consistency(1. + 2.**-23, 1., "torch.bfloat16")


@pytest.mark.parametrize("dtype", ["torch.float16", "torch.bfloat16"])
def test_uncalibrated_coarse_scalar_dtype_cannot_admit_percent_level_drift(dtype):
    # Both values are exactly represented in these dtypes. Supporting them with
    # the FP32/64 epsilon multiplier would silently expand the scientific bound.
    with pytest.raises(ValueError):
        scalar_replay_consistency(1.125, 1., dtype)
