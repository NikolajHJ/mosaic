"""Explicit molecular-latent starting policies, independent of any scorer.

An unfiltered standard-normal draw tests the VAE prior itself. It is never an
encoded molecule plus noise, a normalized vector, or a validity-conditioned draw.
The CPU generator makes its identity independent of GPU assignment and of other
random work performed while loading or scoring a molecule.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentStart:
    kind: str = "encoded_seed"
    seed: int | None = None

    def __post_init__(self):
        if self.kind not in ("encoded_seed", "standard_normal"):
            raise ValueError("Unknown latent initialization policy")
        if self.kind == "encoded_seed" and self.seed is not None:
            raise ValueError("Encoded-seed initialization does not use a random seed")
        if self.kind == "standard_normal" and (
            type(self.seed) is not int or not 0 <= self.seed < 2**63
        ):
            raise ValueError("Standard-normal initialization requires an explicit integer seed")


def sample_prior(width: int, seed: int, *, device="cpu") -> torch.Tensor:
    """One float32 N(0,I) vector; no data, rejection sampling or graph conversion."""
    LatentStart("standard_normal", seed)
    if type(width) is not int or width < 1:
        raise ValueError("Latent width must be a positive integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((1, width), generator=generator, dtype=torch.float32).to(device)
