"""Calibrate Nesso affinity and Protenix contact derivatives in probability space."""

import math

import numpy as np


# Opposing directions can have equal contributions yet no usable sum. Limit the
# common rescaling in that case. The diagnostics retain the actual shortfall.
MIN_MIXED_NORM_RATIO = 0.25


def _rows(gradient):
    if len(gradient) != 2:
        raise ValueError("Expected atom and unordered bond gradient matrices")
    rows = tuple(np.asarray(g, dtype=np.float64) for g in gradient)
    if any(g.ndim != 2 or not g.size for g in rows):
        raise ValueError("Expected nonempty gradient matrices")
    if any(not np.isfinite(g).all() for g in rows):
        raise FloatingPointError("Nonfinite molecular gradient")
    return rows


def _norm(rows):
    return math.sqrt(sum(float(np.vdot(g, g)) for g in rows))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _centered(rows):
    # Adding a constant to every category of one probability row gives a
    # component orthogonal to its unit-sum simplex. The solver clips BEFORE
    # projection, so retain raw derivatives for calibration and the update.
    return tuple(g - g.mean(axis=1, keepdims=True) for g in rows)


def _cosine(first, second, first_norm, second_norm):
    if not first_norm or not second_norm:
        return None
    dot = sum(float(np.vdot(a, b)) for a, b in zip(first, second))
    return float(np.clip(dot / (first_norm * second_norm), -1., 1.))


def _pair(protenix, nesso):
    protenix, nesso = _rows(protenix), _rows(nesso)
    if any(p.shape != n.shape for p, n in zip(protenix, nesso)):
        raise ValueError("Model gradients refer to different graph shapes")
    return protenix, nesso


def _mix(protenix, nesso, p_weight, n_weight):
    return tuple(p_weight * p + n_weight * n for p, n in zip(protenix, nesso))


def _diagnostics(protenix, nesso, combined, p_weight, n_weight):
    p_norm = _norm(protenix)
    p_centered = _centered(protenix)
    pc_norm = _norm(p_centered)
    combined_norm = _norm(combined)
    metrics = dict(protenix_weight=p_weight, nesso_weight=n_weight,
                   protenix_norm=p_norm, nesso_norm=None, cosine=None,
                   combined_norm=combined_norm,
                   protenix_contribution_norm=p_weight * p_norm, nesso_contribution_norm=None,
                   nesso_to_protenix_contribution=None,
                   combined_to_protenix=_ratio(combined_norm, p_norm),
                   centered_cosine=None, centered_nesso_to_protenix_contribution=None,
                   centered_combined_to_protenix=_ratio(_norm(_centered(combined)), pc_norm),
                   protenix_flat=(p_norm == 0.), nesso_flat=None)
    if nesso is not None:
        n_norm = _norm(nesso)
        n_centered = _centered(nesso)
        nc_norm = _norm(n_centered)
        metrics.update(nesso_norm=n_norm, nesso_flat=(n_norm == 0.),
                       cosine=_cosine(protenix, nesso, p_norm, n_norm),
                       nesso_contribution_norm=n_weight * n_norm,
                       nesso_to_protenix_contribution=_ratio(n_weight * n_norm, p_weight * p_norm),
                       centered_cosine=_cosine(p_centered, n_centered, pc_norm, nc_norm),
                       centered_nesso_to_protenix_contribution=_ratio(n_weight * nc_norm, p_weight * pc_norm))
    return metrics


def _range(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return None
    return dict(min=min(values), median=float(np.median(values)), max=max(values))


def calibrate(pairs):
    """Fit fixed coefficients on unweighted, same-graph gradient pairs.

    First match component sizes by the median of ||P||/||N||. Then scale the
    sum so its median length relative to ||P|| is one. This second operation
    accounts for local alignment between the directions. All norms pool both
    matrices and match the solver's global L2 preclip convention.
    """
    panel = [_pair(protenix, nesso) for protenix, nesso in pairs]
    if not panel:
        raise ValueError("Calibration requires at least one pair of native gradients")
    ratios, usable = [], []
    for protenix, nesso in panel:
        p_norm, n_norm = _norm(protenix), _norm(nesso)
        if p_norm and n_norm:
            ratios.append(p_norm / n_norm)
            usable.append((protenix, nesso, p_norm))
    p_weight = n_weight = 1.
    status = "no_nonzero_pairs_unscaled"
    if usable:
        n_to_p = float(np.median(ratios))
        mixed_norm = float(np.median([_norm(_mix(p, n, 1., n_to_p)) / p_norm
                                      for p, n, p_norm in usable]))
        p_weight = 1. / max(mixed_norm, MIN_MIXED_NORM_RATIO)
        n_weight = p_weight * n_to_p
        status = "calibrated" if mixed_norm >= MIN_MIXED_NORM_RATIO else "near_cancellation_limited"
    sample_rows = [_diagnostics(p, n, _mix(p, n, p_weight, n_weight), p_weight, n_weight)
                   for p, n in panel]
    range_fields = ("cosine", "combined_to_protenix", "nesso_to_protenix_contribution",
                    "centered_cosine", "centered_combined_to_protenix",
                    "centered_nesso_to_protenix_contribution")
    return dict(protenix_weight=p_weight, nesso_weight=n_weight, calibration_status=status,
                valid_nonzero_pairs=len(usable), min_mixed_norm_ratio=MIN_MIXED_NORM_RATIO,
                norm_basis="raw L2 over atom and unordered bond probabilities, before repair and clipping",
                samples=sample_rows, ranges={k: _range(sample_rows, k) for k in range_fields})


def blend_gradients(protenix, nesso, calibration, mode):
    """Return FP32 binding derivatives and measurements BEFORE adding repair.

    Fixed coefficients differentiate a weighted scalar objective. Adaptive
    coefficients define a local direction and are held constant for that step.
    The unweighted Protenix control can omit the Nesso gradient entirely.
    """
    if mode == "protenix":
        protenix = _rows(protenix)
        nesso, p_weight, n_weight, status = None, 1., 0., "control"
        combined = protenix
    elif mode in ("fixed", "adaptive"):
        if nesso is None:
            raise ValueError("Blended updates require both native gradients")
        protenix, nesso = _pair(protenix, nesso)
        p_weight, n_weight = float(calibration["protenix_weight"]), float(calibration["nesso_weight"])
        if not all(math.isfinite(w) and w > 0 for w in (p_weight, n_weight)):
            raise ValueError("Expected positive finite fixed calibration weights")
        status = calibration.get("calibration_status", "calibrated")
        if mode == "adaptive":
            p_norm, n_norm = _norm(protenix), _norm(nesso)
            if p_norm and n_norm:
                ratio = p_norm / n_norm
                mixed_norm = _norm(_mix(protenix, nesso, 1., ratio)) / p_norm
                p_weight = 1. / max(mixed_norm, MIN_MIXED_NORM_RATIO)
                n_weight = p_weight * ratio
                status = "balanced" if mixed_norm >= MIN_MIXED_NORM_RATIO else "near_cancellation_limited"
            elif p_norm:
                p_weight, status = 1., "flat_nesso"
            else:
                # No nonzero local contact reference. A measured fixed Nesso
                # weight retains its native signal without dividing by zero.
                status = "flat_protenix_fixed_coefficients"
        combined = _mix(protenix, nesso, p_weight, n_weight)
    else:
        raise ValueError(f"Unknown model gradient blend mode: {mode}")
    output = tuple(g.astype(np.float32) for g in combined)
    if not all(np.isfinite(g).all() for g in output):
        raise FloatingPointError("Nonfinite combined gradient in FP32")
    observed = _diagnostics(protenix, nesso, tuple(g.astype(np.float64) for g in output), p_weight, n_weight)
    return output, dict(mode=mode, status=status, **observed)


_THREE_HEADS = ("protenix", "nesso", "confidence")


def _triple(protenix, nesso, confidence):
    p, n = _pair(protenix, nesso)
    c = _rows(confidence)
    if any(x.shape != y.shape for x, y in zip(p, c)):
        raise ValueError("Confidence gradient refers to different graph shapes")
    return p, n, c


def _mix_three(triple, weights):
    wp, wn, wc = weights
    return tuple(wp*p + wn*n + wc*c for p, n, c in zip(*triple))


def _three_geometry(triple, weights, combined):
    norms = [_norm(g) for g in triple]
    contributions = [w*n for w, n in zip(weights, norms)]
    total = sum(contributions)
    length = _norm(combined)
    fields = dict(combined_norm=length, combined_to_protenix=_ratio(length, norms[0]),
                  summed_norm_fraction_retained=_ratio(length, total))
    for name, norm, contribution in zip(_THREE_HEADS, norms, contributions):
        fields[name + "_norm"] = norm
        fields[name + "_contribution_norm"] = contribution
        fields[name + "_share"] = _ratio(contribution, total)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        fields[f"{_THREE_HEADS[i]}_{_THREE_HEADS[j]}_cosine"] = _cosine(
            triple[i], triple[j], norms[i], norms[j])
    return fields


def blend_three(protenix, nesso, confidence, calibration):
    """Fixed gradient of wp*contact + wn*affinity + wc*(-log binder probability)."""
    triple = _triple(protenix, nesso, confidence)
    weights = [float(calibration[k + "_weight"]) for k in _THREE_HEADS]
    if not all(math.isfinite(w) and w > 0 for w in weights):
        raise ValueError("Expected three positive finite calibration weights")
    combined = tuple(g.astype(np.float32) for g in _mix_three(triple, weights))
    if not all(np.isfinite(g).all() for g in combined):
        raise FloatingPointError("Nonfinite three-head gradient in FP32")
    actual = tuple(g.astype(np.float64) for g in combined)
    metrics = _three_geometry(triple, weights, actual)
    tangent = _three_geometry([_centered(g) for g in triple], weights, _centered(actual))
    metrics.update({"centered_" + k: v for k, v in tangent.items()})
    metrics.update({k + "_weight": w for k, w in zip(_THREE_HEADS, weights)})
    return combined, dict(mode="fixed_three", status=calibration["calibration_status"], **metrics)


def calibrate_three(triples, shares=(.35, .35, .30)):
    """Target ratios of weighted gradient magnitudes and match the Protenix step scale.

    Initialize with median native norm ratios, then balance the mean per-probe
    shares over probes with three nonzero gradients. A common multiplier sets the median combined
    gradient norm relative to native contact to one, subject to the existing
    cancellation guard. Individual probe shares can depart from the targets.
    """
    shares = tuple(float(x) for x in shares)
    if (len(shares) != 3 or not all(math.isfinite(x) and x > 0 for x in shares)
            or not math.isclose(sum(shares), 1., abs_tol=1e-6)):
        raise ValueError("Expected three positive gradient shares summing to one")
    panel = [_triple(*triple) for triple in triples]
    if not panel:
        raise ValueError("Calibration requires at least one native gradient triple")
    norms = [[_norm(g) for g in triple] for triple in panel]
    relative = [1.]
    counts = []
    for index in (1, 2):
        ratios = [ns[0] / ns[index] for ns in norms if ns[0] and ns[index]]
        counts.append(len(ratios))
        relative.append((shares[index] / shares[0]) * (float(np.median(ratios)) if ratios else 1.))
    nonflat = np.asarray([ns for ns in norms if all(ns)], dtype=np.float64)
    if len(nonflat):
        # Row normalization gives every calibration molecule equal influence.
        relative = np.asarray(relative)
        for _ in range(250):
            parts = nonflat * relative
            observed_shares = (parts / parts.sum(axis=1, keepdims=True)).mean(axis=0)
            if np.max(np.abs(observed_shares - shares)) < 1.e-9:
                break
            relative *= np.asarray(shares) / observed_shares
            relative /= relative[0]
    lengths = [_norm(_mix_three(t, relative)) / ns[0] for t, ns in zip(panel, norms) if ns[0]]
    typical = float(np.median(lengths)) if lengths else None
    scale = 1. / max(typical, MIN_MIXED_NORM_RATIO) if lengths else 1.
    weights = {k + "_weight": v * scale for k, v in zip(_THREE_HEADS, relative)}
    cancellation = typical is not None and typical < MIN_MIXED_NORM_RATIO
    status = "calibrated"
    if cancellation:
        status = "near_cancellation_limited"
    elif not lengths or min(counts) == 0:
        status = "unmeasured_head_ratio"
    calibration = dict(weights, calibration_status=status, near_cancellation_limited=cancellation,
                       requested_shares=dict(zip(_THREE_HEADS, shares)),
                       valid_nonzero_triples=sum(all(ns) for ns in norms),
                       paired_nonzero_probes=dict(nesso=counts[0], confidence=counts[1]),
                       min_mixed_norm_ratio=MIN_MIXED_NORM_RATIO,
                       norm_basis="raw L2 over atom and unordered bond probabilities, before repair and clipping")
    samples = [blend_three(*t, calibration)[1] for t in panel]
    keys = ("combined_to_protenix", "protenix_share", "nesso_share", "confidence_share",
            "summed_norm_fraction_retained", "protenix_nesso_cosine",
            "protenix_confidence_cosine", "nesso_confidence_cosine")
    ranges = {prefix + k: _range(samples, prefix + k) for k in keys for prefix in ("", "centered_")}
    return dict(calibration, samples=samples, ranges=ranges)
