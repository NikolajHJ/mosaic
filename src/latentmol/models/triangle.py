"""Gated, symmetric two-edge paths for the canonical relaxed-graph decoder.

This is a compact triangle-inspired update, not an AF3 coordinate module.
Only pair tensors [B,N,N,C] are retained: no [B,N,N,N,C] allocation is needed.
"""

import torch
from torch import nn


class TrianglePairUpdate(nn.Module):
    """Pass information from (i,k) and (k,j) into (i,j), excluding i=j,k=i,j.

    Intermediate occupancy is counted once, then the contraction is normalized
    by eligible occupancy. Endpoint occupancy gates the output once. Masking
    happens after biased projections so absent self-paths cannot leak biases.
    A zero output projection starts this branch at the residual identity.
    """

    def __init__(self, pair_width: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(pair_width)
        self.left = nn.Linear(pair_width, bottleneck)
        self.right = nn.Linear(pair_width, bottleneck)
        self.left_gate = nn.Linear(pair_width, bottleneck)
        self.right_gate = nn.Linear(pair_width, bottleneck)
        self.product_norm = nn.LayerNorm(bottleneck)
        self.output = nn.Linear(bottleneck, pair_width)
        self.output_gate = nn.Linear(pair_width, pair_width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, pair, occupancy):
        nodes = pair.shape[1]
        off_diagonal = ~torch.eye(nodes, dtype=torch.bool, device=pair.device)
        x = self.norm(pair)
        left = (self.left(x) * self.left_gate(x).sigmoid()).masked_fill(
            ~off_diagonal[None, :, :, None], 0.0)
        right = (self.right(x) * self.right_gate(x).sigmoid()).masked_fill(
            ~off_diagonal[None, :, :, None], 0.0)
        # k is the shared/intermediate atom. Soft occupancy is not squared.
        paths = torch.einsum("bikc,bkjc,bk->bijc", left, right, occupancy)
        eligible = (occupancy.sum(-1)[:, None, None]
                    - occupancy[:, :, None] - occupancy[:, None, :]).clamp_min(0.0)
        paths = paths / eligible.clamp_min(1.0)[..., None]
        delta = self.output(self.product_norm(paths)) * self.output_gate(x).sigmoid()
        delta = (delta + delta.transpose(1, 2)) * 0.5
        endpoint_mask = occupancy[:, :, None] * occupancy[:, None, :]
        mask = off_diagonal[None] & (eligible > 0)
        return delta * endpoint_mask[..., None] * mask[..., None]
