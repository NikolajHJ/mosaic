"""AE and VAE share every architectural parameter and differ only in latent use."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph
from .decoder import GraphDecoder
from .encoder import GraphEncoder
from .refined_decoder import RefinedGraphDecoder


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 128
    latent: int = 128
    layers: int = 3
    decoder_variant: str = "legacy"
    global_hidden: int | None = None
    pair_hidden: int = 32
    refinement_rounds: int = 2
    attention_heads: int = 4
    trunk_hidden: int | None = None
    encoder_residual_layers: int = 0
    triangle_hidden: int = 24
    encoder_pooling: str = "ordered"
    encoder_aggregation: str = "mean"
    encoder_degree_features: bool = False
    encoder_sum_scale: float = 3.

    def __post_init__(self):
        if self.hidden < 1 or self.latent < 1 or self.layers < 1:
            raise ValueError("Model widths and encoder layer count must be positive")
        if self.encoder_pooling not in ("ordered", "invariant", "canonical"):
            raise ValueError("Encoder pooling must be ordered, invariant or canonical")
        if self.encoder_aggregation not in ("mean", "sum"):
            raise ValueError("Encoder aggregation must be mean or sum")
        if type(self.encoder_degree_features) is not bool:
            raise ValueError("Encoder degree_features must be boolean")
        if not math.isfinite(self.encoder_sum_scale) or self.encoder_sum_scale <= 0:
            raise ValueError("Encoder sum scale must be finite and positive")
        if self.decoder_variant not in ("legacy", "residual", "pair_refine",
                                        "deep_pair_refine", "global_bond",
                                        "wide_global_bond", "stable_deep_pair_refine", "triangle_refine"):
            raise ValueError("Unknown decoder variant")
        if self.global_hidden is not None and self.global_hidden < 1:
            raise ValueError("Global hidden width must be positive")
        if self.pair_hidden < 1 or self.refinement_rounds < 0 or self.attention_heads < 1:
            raise ValueError("Pair width and attention heads must be positive; rounds cannot be negative")
        if self.trunk_hidden is not None and self.trunk_hidden < 1:
            raise ValueError("Shared trunk width must be positive")
        if self.encoder_residual_layers < 0 or self.triangle_hidden < 1:
            raise ValueError("Encoder residual count must be nonnegative; triangle width positive")
        if self.decoder_variant == "legacy" and self.trunk_hidden is not None:
            raise ValueError("Legacy decoder does not use trunk_hidden")
        if self.decoder_variant != "legacy":
            if self.global_hidden is None or self.global_hidden < self.latent:
                raise ValueError("Refined decoders require a global width at least as large as the latent")
            if self.hidden % self.attention_heads:
                raise ValueError("Atom width must be divisible by the attention head count")
            if self.decoder_variant == "residual" and self.refinement_rounds != 0:
                raise ValueError("The residual control must declare zero pair-refinement rounds")
            if self.decoder_variant in ("pair_refine", "deep_pair_refine", "global_bond",
                                        "wide_global_bond", "stable_deep_pair_refine", "triangle_refine") and self.refinement_rounds < 1:
                raise ValueError("Pair refinement needs at least one round")
            if self.decoder_variant in ("deep_pair_refine", "global_bond",
                                        "wide_global_bond", "stable_deep_pair_refine"):
                if self.trunk_hidden is None or self.trunk_hidden < self.global_hidden:
                    raise ValueError("Deep decoders require a trunk width covering global width")
            elif self.trunk_hidden is not None:
                raise ValueError("Only deep decoders use trunk_hidden")


@dataclass
class AEOutput:
    decoded: DecoderOutput
    mu: Tensor
    logvar: Tensor
    z: Tensor


class GraphAutoencoder(nn.Module):
    """The encoder initializes molecules; only the decoder is needed to optimize z."""

    def __init__(self, spec: GraphSpec, config: ModelConfig):
        super().__init__()
        self.spec, self.config = spec, config
        self.encoder = GraphEncoder(spec, config.hidden, config.latent, config.layers,
                                    pool_hidden=config.global_hidden,
                                    residual_layers=config.encoder_residual_layers,
                                    pooling=config.encoder_pooling,
                                    aggregation=config.encoder_aggregation,
                                    degree_features=config.encoder_degree_features,
                                    sum_scale=config.encoder_sum_scale)
        if config.decoder_variant == "legacy":
            self.decoder = GraphDecoder(spec, config.hidden, config.latent)
        else:
            self.decoder = RefinedGraphDecoder(spec, config.hidden, config.latent,
                                               config.global_hidden, config.pair_hidden,
                                               config.refinement_rounds, config.attention_heads,
                                               trunk_hidden=config.trunk_hidden,
                                               stable_refinement=config.decoder_variant in (
                                                   "stable_deep_pair_refine", "triangle_refine"),
                                               direct_slots=config.decoder_variant == "triangle_refine",
                                               deep_refinement=config.decoder_variant == "triangle_refine",
                                               triangle_hidden=config.triangle_hidden if
                                               config.decoder_variant == "triangle_refine" else None,
                                               global_bond_context=config.decoder_variant in (
                                                   "global_bond", "wide_global_bond"),
                                               wide_trunk=config.decoder_variant == "wide_global_bond")

    def encode(self, graph: RelaxedGraph):
        return self.encoder(graph)

    def decode(self, z) -> DecoderOutput:
        return self.decoder(z)

    def decode_with_features(self, z):
        """Expose refined decoder states without changing its molecular heads."""
        if not isinstance(self.decoder, RefinedGraphDecoder):
            raise ValueError("Hidden-state bridges require a refined decoder")
        return self.decoder.decode_with_features(z)

    def forward(self, graph: RelaxedGraph | HardGraphBatch, sample_latent: bool = False, *,
                from_canonical_cache: bool = False) -> AEOutput:
        """Use hard labels directly only after the trainer verifies the cache."""
        if from_canonical_cache:
            mu, logvar = self.encoder(graph, from_canonical_cache=True)
        else:
            mu, logvar = self.encode(graph)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample_latent else mu
        return AEOutput(self.decode(z), mu, logvar, z)
