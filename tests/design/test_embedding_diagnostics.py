import torch

from latentmol.design.embedding_diagnostics import decompose_prototype_span, softmax_sensitivity


def test_affine_span_and_native_gradient_components_preserve_the_vjp():
    prototypes = torch.tensor([[1., 1., 0.], [3., 3., 0.], [5., 5., 0.]])
    values = torch.tensor([[4., 2., 5.]], requires_grad=True)
    native_gradient = torch.tensor([[0., 0., 2.]])
    result = decompose_prototype_span(values, prototypes, native_gradient)
    assert result['metrics']['rank'] == 1
    assert torch.allclose(result['span_component'], torch.tensor([[3., 3., 0.]]), atol=1e-6)
    assert torch.allclose(result['residual_component'], torch.tensor([[1., -1., 5.]]), atol=1e-6)
    assert result['metrics']['gradient_orthogonal_ratio'] == 1.
    in_span, = torch.autograd.grad(result['span_component'].sum(), values, retain_graph=True)
    residual, = torch.autograd.grad(result['residual_component'].sum(), values)
    assert torch.allclose(in_span, torch.tensor([[1., 1., 0.]]), atol=1e-6)
    assert torch.allclose(in_span + residual, torch.ones_like(values), atol=1e-6)
    # Identical prototypes define a constant affine point, with zero tangent VJP.
    flat = decompose_prototype_span(values, torch.ones(2, 3))
    assert flat['metrics']['rank'] == 0
    assert torch.equal(flat['span_component'], torch.ones_like(values))


def test_softmax_sensitivity_matches_explicit_jacobians_and_detects_saturation():
    temperature = .7
    logits = torch.tensor([[0., 0., 0.], [30., -30., -30.]], dtype=torch.float64)
    result = softmax_sensitivity(logits, temperature)
    probability = (logits / temperature).softmax(-1)
    jacobians = (torch.diag_embed(probability) - probability[..., :, None] * probability[..., None, :]) / temperature
    norms = jacobians.square().sum((-1, -2)).sqrt()
    assert abs(result['jacobian_frobenius_mean'] - float(norms.mean())) < 1e-12
    assert result['saturated_fraction_099'] == .5
    assert result['jacobian_frobenius_min'] < 1e-12
    assert result['normalized_entropy_mean'] <= .500001
