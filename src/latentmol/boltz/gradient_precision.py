"""Pure metadata rules for prospective scale-aware derivative certificates.

An absolute tolerance alone cannot distinguish a small derivative from zero or
an incorrect sign. New checks require relative AD agreement as additional
evidence for every direction; historical gate tolerances remain unchanged.
"""
import math
import struct


def relative_ad_consistency(forward, reverse_projection, gradient_norm):
    """Return auditable relative agreement and a unit-direction norm bound."""
    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in (forward, reverse_projection, gradient_norm)) or gradient_norm < 0):
        raise ValueError("Relative AD requires finite values and a nonnegative gradient norm")
    tolerance = 1e-3 * max(abs(forward), abs(reverse_projection))
    # Directions normalize in their production dtype (at least FP32). A small
    # normalization allowance is appropriate; an exactly zero gradient still
    # has exactly zero directional projection.
    projection_bound = gradient_norm * (1 + 8 * 2.0**-23)
    error = abs(forward - reverse_projection)
    return dict(rtol=1e-3, atol=0., absolute_error=error, tolerance=tolerance,
                reverse_projection_bound=projection_bound,
                passed=error <= tolerance and abs(reverse_projection) <= projection_bound)


def scalar_replay_consistency(observed, expected, dtype, expected_dtype=None):
    """Numerical scalar replay, distinct from source/state byte identity."""
    expected_dtype = dtype if expected_dtype is None else expected_dtype
    # Half-precision scalar replay needs a separately calibrated policy: 32
    # epsilons there would permit material percent-level drift. The current
    # production contracts and regression evidence cover FP32/FP64 only.
    formats = {"torch.float32": (2.**-23, 2.**-149), "torch.float64": (2.**-52, 2.**-1074)}
    if (not isinstance(dtype, str) or not isinstance(expected_dtype, str)
            or dtype not in formats or expected_dtype not in formats
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in (observed, expected))):
        raise ValueError("Scalar replay requires finite values and declared floating dtypes")
    def represented(value, kind):
        code = {"torch.float32": "f", "torch.float64": "d"}[kind]
        return struct.unpack("=" + code, struct.pack("=" + code, value))[0] == value
    try:
        supported = represented(observed, dtype) and represented(expected, expected_dtype)
    except (OverflowError, struct.error):
        supported = False
    if not supported:
        raise ValueError("Scalar replay values are not represented in their declared dtypes")
    epsilon = max(formats[dtype][0], formats[expected_dtype][0])
    subnormal = max(formats[dtype][1], formats[expected_dtype][1])
    tolerance = 32 * epsilon * max(abs(observed), abs(expected)) + 32 * subnormal
    # Match the common replay policy: opposite nonzero signs fail. A zero may
    # differ by a few subnormal units; the element-local bound still rejects
    # erasing any resolved scalar. This is separate from derivative agreement.
    same_sign = (observed == 0 or expected == 0 or (observed > 0) == (expected > 0))
    error = abs(observed - expected)
    if not math.isfinite(error) or not math.isfinite(tolerance):
        raise FloatingPointError("Nonfinite scalar replay difference or error budget")
    return dict(policy="dtype-relative-reproducibility-v1", observed=observed, expected=expected,
                dtype=dtype, expected_dtype=expected_dtype, epsilon_multiplier=32,
                subnormal_units=32, absolute_error=error, tolerance=tolerance,
                sign_consistent=same_sign, passed=same_sign and error <= tolerance)
