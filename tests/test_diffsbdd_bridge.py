"""Synthetic chemical-boundary contracts; numerical execution is allocated HPC only."""

import pytest
import torch
from torch import nn

from latentmol.diffsbdd.bridge import (ATOMIC_NUMBER, PILOT_ELEMENTS, ConstrainedAtomBridge,
    NativeAtomBoundary, exact_native_probabilities, hard_element_agreement)
from latentmol.representation import DecoderOutput, GraphSpec


NATIVE = ("C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F")


def boundary(vocabulary=NATIVE, *, bias=0.3, factor=4.0):
    encoder = nn.Linear(len(vocabulary), 8).eval().requires_grad_(False)
    return NativeAtomBoundary(encoder, vocabulary, factor, bias).eval()


def decoded(spec, logits):
    batch, nodes = logits.shape[:2]
    return DecoderOutput(logits, logits.new_zeros(batch, nodes, len(spec.formal_charges)),
                         logits.new_zeros(batch, nodes, nodes, len(spec.bond_names)))


def test_vocabulary_conversion_uses_symbols_and_records_conditioning_mass():
    spec = GraphSpec(max_nodes=3, atomic_numbers=(0, 17, 6, 7, 8, 9, 15, 16, 35, 53))
    logits = torch.arange(30.0).reshape(1, 3, 10) * 0.1
    value = decoded(spec, logits)
    active = torch.tensor([[True, False, True]])
    full = exact_native_probabilities(value, spec, active, NATIVE)
    expected = value.atoms[..., 1:].softmax(-1)
    assert full.mode == "native_full_exact"
    for index, number in enumerate(spec.atomic_numbers[1:]):
        symbol = next(x for x, n in ATOMIC_NUMBER.items() if n == number)
        torch.testing.assert_close(full.probabilities[..., NATIVE.index(symbol)], expected[..., index] * active)
    torch.testing.assert_close(full.probabilities.sum(-1), active.float())
    assert not bool(full.probabilities[..., NATIVE.index("B")].any())
    six = exact_native_probabilities(value, spec, active, NATIVE, allowed_elements=PILOT_ELEMENTS)
    assert six.mode == "explicit_conditioned_exact"
    assert bool((six.excluded_mass[active] > 0).all())
    assert six.excluded_mass[0, 1] == 0
    assert not bool(six.probabilities[..., NATIVE.index("P")].any())
    torch.testing.assert_close(six.probabilities.sum(-1), active.float())
    with pytest.raises(ValueError, match="every nonpadding"):
        exact_native_probabilities(value, spec, active, PILOT_ELEMENTS)


def test_native_normalization_and_corruption_cover_clean_zero_channels():
    native = boundary()
    active = torch.ones((2, 3), dtype=torch.bool)
    q = torch.softmax(torch.randn(2, 3, 6), -1)
    p = native.scatter(q)
    noise = torch.randn_like(p)
    alpha, sigma = torch.tensor([[0.8], [0.7]]), torch.tensor([[0.2], [0.3]])
    got = native.features(p, active, alpha=alpha, sigma=sigma, noise=noise)
    expected_input = alpha[..., None] * ((p - 0.3) / 4.0) + sigma[..., None] * noise
    torch.testing.assert_close(got, native.atom_encoder(expected_input), rtol=0, atol=0)
    no_b_noise = noise.clone()
    no_b_noise[..., NATIVE.index("B")] = 0
    altered = native.features(p, active, alpha=alpha, sigma=sigma, noise=no_b_noise)
    assert not torch.equal(got, altered)
    assert torch.equal(p[..., NATIVE.index("B")], torch.zeros_like(p[..., 0]))


def test_latent_gradient_crosses_frozen_native_encoder_without_weight_gradients():
    native = boundary()
    head = ConstrainedAtomBridge(hidden_width=4, width=8)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    active = torch.tensor([[True, False, True]])
    p = head.native_probabilities(hidden, native, active)
    output = native.features(p, active, alpha=0.9, sigma=0.1, noise=torch.randn_like(p))
    output.square().sum().backward()
    assert hidden.grad is not None and bool(hidden.grad[active].abs().sum() > 0)
    assert torch.equal(hidden.grad[~active], torch.zeros_like(hidden.grad[~active]))
    assert all(p.grad is None for p in native.parameters())
    assert any(p.grad is not None for p in head.parameters())


def test_population_agreement_cannot_admit_a_wrong_individual_handoff():
    spec = GraphSpec(max_nodes=2)
    logits = torch.full((2, 2, 10), -20.0)
    logits[..., spec.atomic_numbers.index(6)] = 20
    value = decoded(spec, logits)
    head_logits = torch.full((2, 2, 6), -20.0)
    head_logits[..., 0] = 20
    head_logits[1, 0, 0], head_logits[1, 0, 1] = -20, 20
    active = torch.ones((2, 2), dtype=torch.bool)
    assert hard_element_agreement(value, head_logits, spec, active).tolist() == [True, False]


def test_boundary_fails_closed_on_training_mode_invalid_noise_and_normalization():
    with pytest.raises(ValueError, match="already be frozen"):
        NativeAtomBoundary(nn.Linear(10, 8), NATIVE, 4.0, 0.0)
    encoder = nn.Linear(10, 8).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="positive finite"):
        NativeAtomBoundary(encoder, NATIVE, 0.0, 0.0)
    native = boundary()
    p = torch.softmax(torch.randn(1, 2, 10), -1)
    active = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="every native"):
        native.features(p, active, alpha=1, sigma=0, noise=p[..., :6])
    native.atom_encoder.train()
    with pytest.raises(ValueError, match="remain frozen"):
        native.features(p, active, alpha=1, sigma=0, noise=torch.zeros_like(p))
