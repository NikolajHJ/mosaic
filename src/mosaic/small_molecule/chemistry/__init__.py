"""Detached atom-preserving repair guidance with live JAX categorical losses.

RDKit prepares references only. Scorers consume the same unrepaired graph.
The generic loss supports source-schema padding/charges; the current Mosaic
adapter explicitly fixes occupancy and neutral charge.
"""

from .loss import FailureGuidanceConfig, failure_aware_repair_loss
from .preparation import prepare_failure_guidance
from .state import DecoderLogits, GraphLogits, mosaic_spec

__all__ = [
    "FailureGuidanceConfig",
    "failure_aware_repair_loss",
    "prepare_failure_guidance",
    "DecoderLogits",
    "GraphLogits",
    "mosaic_spec",
]
