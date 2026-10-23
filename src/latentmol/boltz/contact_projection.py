"""Explicit FP64 evaluation of the pinned, frozen Boltz distogram projection.

The structural trunk remains native FP32. Widen before symmetrization and the
affine map, rather than widening logits that have already been rounded. This
only supplies the contact branch: confidence still receives native logits.
No checkpoint parameter, module forward method or chemical identity changes.
"""

import torch
from torch import nn
from torch.nn import functional as F


def contact_distogram_float64(module, pair: torch.Tensor) -> torch.Tensor:
    """Evaluate the supported native affine map with FP64 intermediate arithmetic.

    Native DistogramModule computes Linear(z + z.transpose(1, 2)) and reshapes
    to [batch, token, token, one_distogram, 64_bins]. Reject other head layouts
    rather than guessing an equivalent map. Casting the frozen weights is a
    differentiable input operation and never mutates their registered state.
    This cannot recover rounding inside the upstream trunk or smooth its kinks.
    """
    linear = getattr(module, "distogram", None)
    if (not isinstance(linear, nn.Linear)
            or getattr(module, "num_distograms", None) != 1
            or getattr(module, "num_bins", None) != 64
            or linear.out_features != 64
            or pair.ndim != 4 or pair.shape[1] != pair.shape[2]
            or pair.shape[-1] != linear.in_features
            or pair.dtype != torch.float32
            or module.training
            or any(p.requires_grad for p in module.parameters())):
        raise ValueError("Contact projection requires the frozen eval FP32 single native 64-bin head")
    if linear.weight.dtype != torch.float32 or (linear.bias is not None and linear.bias.dtype != torch.float32):
        raise ValueError("Contact projection must retain native FP32 checkpoint parameters")
    widened = pair.double()
    logits = F.linear(widened + widened.transpose(1, 2), linear.weight.double(),
                      None if linear.bias is None else linear.bias.double())
    return logits.reshape(*pair.shape[:3], 1, 64)
