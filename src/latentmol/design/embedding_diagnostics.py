"""Observations of chemical prototype spans and categorical sensitivity.

These helpers characterize a learned embedding; they impose no admission or
quality threshold. Prototype span means the affine span, not the convex hull
or proof that an embedding represents a molecule.
"""
from __future__ import annotations

import math

import torch


def _finite_float(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not bool(value.isfinite().all()):
        raise ValueError(name + " must be a finite floating tensor")


def decompose_prototype_span(values, prototypes, gradient=None):
    """Return differentiable affine-span/residual components and plain metrics.

    ``values`` has shape ``[..., width]`` and ``prototypes`` is ``[labels,
    width]``. The SVD is evaluated in FP64 on detached prototype differences,
    with a rank tolerance based on the original prototype dtype's resolution.
    Prediction components retain their gradient to ``values``. Native gradient
    decomposition, when supplied, uses the corresponding *linear* tangent span.
    """
    _finite_float(values, "Values")
    _finite_float(prototypes, "Prototypes")
    if (values.ndim < 1 or prototypes.ndim != 2 or not prototypes.shape[0]
            or not prototypes.shape[1] or values.shape[-1] != prototypes.shape[-1]):
        raise ValueError("Prototype/value shapes are incompatible")
    original_epsilon = torch.finfo(prototypes.dtype).eps
    reference = prototypes.detach().to(device=values.device, dtype=torch.float64)
    center64 = reference[0]
    matrix = reference - center64
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.numel() else 0.
    tolerance = max(matrix.shape) * original_epsilon * largest
    rank = int((singular_values > tolerance).sum()) if largest else 0
    basis = vh[:rank].transpose(0, 1).to(values.dtype)
    center = center64.to(values.dtype)

    def project(v):
        return (v @ basis) @ basis.transpose(0, 1) if rank else v * 0.

    centered = values - center
    span = center + project(centered)
    residual = values - span
    residual_norm = float(residual.detach().double().norm())
    centered_norm = float(centered.detach().double().norm())
    value_norm = float(values.detach().double().norm())
    metrics = dict(width=values.shape[-1], labels=prototypes.shape[0], rank=rank,
        rank_tolerance=tolerance, prototype_dtype=str(prototypes.dtype),
        value_norm=value_norm, centered_value_norm=centered_norm,
        residual_norm=residual_norm,
        residual_value_ratio=residual_norm / value_norm if value_norm else 0.,
        residual_centered_ratio=residual_norm / centered_norm if centered_norm else 0.,
        scope="affine prototype span only; convex-hull membership unmeasured")
    result = dict(span_component=span, residual_component=residual, metrics=metrics)
    if gradient is not None:
        _finite_float(gradient, "Gradient")
        if gradient.shape != values.shape or gradient.device != values.device or gradient.dtype != values.dtype:
            raise ValueError("Native gradient must match the values' shape, device and dtype")
        in_span = project(gradient)
        orthogonal = gradient - in_span
        norm = float(gradient.detach().double().norm())
        orthogonal_norm = float(orthogonal.detach().double().norm())
        metrics.update(gradient_norm=norm, gradient_orthogonal_norm=orthogonal_norm,
                       gradient_orthogonal_ratio=orthogonal_norm / norm if norm else 0.)
        result.update(gradient_in_span=in_span, gradient_orthogonal=orthogonal)
    return result


def softmax_sensitivity(logits, temperature=1.):
    """Summarize row entropy/saturation and exact softmax Jacobian Frobenius norm.

    The Jacobian is with respect to the *unscaled* logits. Its norm includes
    the ``1 / temperature`` factor. This describes local sensitivity, not a
    derivative-accuracy certificate or an optimization-quality score.
    """
    _finite_float(logits, "Logits")
    if (logits.ndim < 1 or logits.shape[-1] < 2 or logits.numel() == 0
            or isinstance(temperature, bool) or not isinstance(temperature, (float, int))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Require nonempty categorical rows and a positive finite temperature")
    with torch.no_grad():
        probability = (logits.detach().double() / temperature).softmax(-1)
        p2 = probability.square().sum(-1)
        p3 = probability.pow(3).sum(-1)
        frobenius = (p2 - 2. * p3 + p2.square()).clamp_min(0.).sqrt() / temperature
        entropy = -(probability * probability.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1)
        pmax = probability.max(-1).values
        categories = logits.shape[-1]
        return dict(rows=pmax.numel(), categories=categories, temperature=float(temperature),
            entropy_mean=float(entropy.mean()), normalized_entropy_mean=float(entropy.mean()) / math.log(categories),
            maximum_probability_mean=float(pmax.mean()), saturated_fraction_099=float((pmax >= .99).double().mean()),
            jacobian_frobenius_mean=float(frobenius.mean()), jacobian_frobenius_min=float(frobenius.min()),
            jacobian_frobenius_max=float(frobenius.max()),
            scope="local categorical sensitivity with respect to unscaled logits")
