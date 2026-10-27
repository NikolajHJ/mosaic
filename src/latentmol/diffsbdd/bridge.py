"""Constrained VAE chemical heads and the frozen native DiffSBDD input boundary.

No hidden state is inverted. Every learned embedding is obtained by applying the
unchanged native atom encoder to a normalized, natively corrupted distribution
over named atom types. This module is numerical and must run on allocated HPC.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


BRIDGE_VERSION = "diffsbdd-constrained-chemical-head-v1"
PILOT_ELEMENTS = ("C", "N", "O", "S", "F", "Cl")
ATOMIC_NUMBER = {"B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15,
                 "S": 16, "Cl": 17, "Br": 35, "I": 53}


def _elements(values):
    result = tuple(values)
    if not result or len(set(result)) != len(result) or any(x not in ATOMIC_NUMBER for x in result):
        raise ValueError("Atom vocabularies must be nonempty, unique supported element symbols")
    return result


def validate_active(active, shape):
    if (active.dtype != torch.bool or tuple(active.shape) != tuple(shape)
            or active.ndim != 2 or not bool(active.any(-1).all())):
        raise ValueError("Require a nonempty boolean occupied-slot mask for every example")


@dataclass(frozen=True)
class ExactConversion:
    """A named exact codec, with its removed conditional probability mass."""

    probabilities: torch.Tensor
    excluded_mass: torch.Tensor
    mode: str
    allowed_elements: tuple[str, ...]


def exact_native_probabilities(decoded, spec, active, native_elements, *, allowed_elements=None):
    """Convert ordinary conditional VAE probabilities by symbol, never by index.

    ``allowed_elements=None`` retains every nonpadding VAE channel and is the
    native-full diagnostic. Explicit six-element conditioning is the A5-exact
    control. It renormalizes only after returning the removed probability mass
    and an unambiguous mode, so the restriction cannot be mistaken for native
    full-vocabulary conversion. Coordinates and occupancy are not predictions.
    """
    native = _elements(native_elements)
    validate_active(active, decoded.atoms.shape[:2])
    inverse = {number: symbol for symbol, number in ATOMIC_NUMBER.items()}
    source = tuple(inverse.get(number) for number in spec.atomic_numbers[1:])
    if any(symbol is None or symbol not in native for symbol in source):
        raise ValueError("Native vocabulary does not represent every nonpadding VAE atom channel")
    allowed = source if allowed_elements is None else _elements(allowed_elements)
    if any(symbol not in source or symbol not in native for symbol in allowed):
        raise ValueError("Exact conditioning contains an unavailable source/native element")
    full = decoded.atoms[..., 1:].softmax(-1)
    if not bool(full.isfinite().all()):
        raise FloatingPointError("Nonfinite ordinary VAE element probabilities")
    indices = [source.index(symbol) for symbol in allowed]
    # Conditioning logits directly avoids catastrophic loss of tiny retained
    # probability mass. The removed mass still comes from the full softmax.
    retained = decoded.atoms[..., [index + 1 for index in indices]].softmax(-1)
    excluded = (1.0 - full[..., indices].sum(-1)).clamp(0, 1).masked_fill(~active, 0)
    result = full.new_zeros((*full.shape[:2], len(native)))
    for index, symbol in enumerate(allowed):
        result[..., native.index(symbol)] = retained[..., index]
    result = result * active[..., None]
    return ExactConversion(result, excluded,
        "native_full_exact" if allowed_elements is None else "explicit_conditioned_exact", tuple(allowed))


class NativeAtomBoundary(nn.Module):
    """Explicit loaded normalization plus unchanged, frozen native atom encoder.

    ``noise`` has shape [batch, slots, native_channels], including clean-zero
    unsupported channels. Supplying noise rather than sampling here preserves
    the sampler's independent RNG stream and target/prediction common noise.
    """

    def __init__(self, atom_encoder, native_elements, normalize_factor, normalize_bias):
        super().__init__()
        self.native_elements = _elements(native_elements)
        scale, bias = torch.as_tensor(normalize_factor), torch.as_tensor(normalize_bias)
        if (scale.numel() not in (1, len(self.native_elements))
                or bias.numel() not in (1, len(self.native_elements))
                or not bool(scale.isfinite().all()) or not bool((scale > 0).all())
                or not bool(bias.isfinite().all())):
            raise ValueError("Require explicit positive finite loaded normalization and finite bias")
        self.register_buffer("normalize_factor", scale.detach().float().reshape(-1))
        self.register_buffer("normalize_bias", bias.detach().float().reshape(-1))
        self.atom_encoder = atom_encoder
        if atom_encoder.training or any(p.requires_grad for p in atom_encoder.parameters()):
            raise ValueError("Native atom encoder must already be frozen in eval mode")

    def train(self, mode=True):
        super().train(mode)
        self.atom_encoder.eval()
        return self

    def metadata(self):
        return dict(version=BRIDGE_VERSION, native_elements=list(self.native_elements),
                    normalize_factor=self.normalize_factor.detach().cpu().tolist(),
                    normalize_bias=self.normalize_bias.detach().cpu().tolist(),
                    normalization="(clean_probabilities-loaded_bias)/loaded_factor")

    def scatter(self, probabilities, elements=PILOT_ELEMENTS):
        elements = _elements(elements)
        if probabilities.shape[-1] != len(elements) or any(x not in self.native_elements for x in elements):
            raise ValueError("Head/native vocabulary mismatch")
        result = probabilities.new_zeros((*probabilities.shape[:-1], len(self.native_elements)))
        for index, symbol in enumerate(elements):
            result[..., self.native_elements.index(symbol)] = probabilities[..., index]
        return result

    def normalize(self, native_probabilities):
        if native_probabilities.shape[-1] != len(self.native_elements):
            raise ValueError("Native atom feature width does not match loaded vocabulary")
        return (native_probabilities - self.normalize_bias.to(native_probabilities)) / self.normalize_factor.to(native_probabilities)

    def features(self, native_probabilities, active, *, alpha, sigma, noise):
        validate_active(active, native_probabilities.shape[:2])
        if noise.shape != native_probabilities.shape or not bool(noise.isfinite().all()):
            raise ValueError("Native Gaussian noise must cover every native atom channel")
        if (self.atom_encoder.training or any(p.requires_grad for p in self.atom_encoder.parameters())):
            raise ValueError("Native atom encoder must remain frozen and in eval mode")
        clean = self.normalize(native_probabilities)
        alpha = torch.as_tensor(alpha, device=clean.device, dtype=clean.dtype)
        sigma = torch.as_tensor(sigma, device=clean.device, dtype=clean.dtype)
        while alpha.ndim < clean.ndim:
            alpha = alpha.unsqueeze(-1)
        while sigma.ndim < clean.ndim:
            sigma = sigma.unsqueeze(-1)
        if not bool(alpha.isfinite().all() & sigma.isfinite().all()) or bool((sigma < 0).any()):
            raise ValueError("Native noise schedule returned nonfinite/negative scale")
        value = self.atom_encoder(alpha * clean + sigma * noise)
        if value.shape[:2] != active.shape:
            raise ValueError("Native atom encoder changed the atom axes")
        return value * active[..., None]


class ConstrainedAtomBridge(nn.Module):
    """The plan's LayerNorm(96), 96->128->128->6 chemical head."""

    def __init__(self, hidden_width=96, width=128, elements=PILOT_ELEMENTS):
        super().__init__()
        if type(hidden_width) is not int or hidden_width < 1 or type(width) is not int or width < 1:
            raise ValueError("Bridge widths must be positive integers")
        self.hidden_width, self.width = hidden_width, width
        self.elements = _elements(elements)
        self.head = nn.Sequential(nn.LayerNorm(hidden_width), nn.Linear(hidden_width, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, len(self.elements)))

    def configuration(self):
        return dict(version=BRIDGE_VERSION, hidden_width=self.hidden_width, width=self.width,
                    elements=list(self.elements))

    def forward(self, atom_hidden):
        if atom_hidden.ndim != 3 or atom_hidden.shape[-1] != self.hidden_width:
            raise ValueError("Expected the final VAE per-atom hidden features")
        return self.head(atom_hidden)

    def native_probabilities(self, atom_hidden, boundary, active):
        validate_active(active, atom_hidden.shape[:2])
        return boundary.scatter(self(atom_hidden).softmax(-1), self.elements) * active[..., None]


def hard_element_agreement(decoded, bridge_logits, spec, active, elements=PILOT_ELEMENTS):
    """Individual handoffs require ordinary full-vocabulary hard identity."""
    validate_active(active, decoded.atoms.shape[:2])
    numbers = bridge_logits.new_tensor([ATOMIC_NUMBER[x] for x in _elements(elements)], dtype=torch.long)
    source_numbers = bridge_logits.new_tensor(spec.atomic_numbers, dtype=torch.long)
    ordinary = source_numbers[decoded.atoms[..., 1:].argmax(-1) + 1]
    predicted = numbers[bridge_logits.argmax(-1)]
    return ((ordinary == predicted) | ~active).all(-1)
