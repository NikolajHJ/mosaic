"""Residual attention over possible bonds incident to the same atom."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class IncidentBondAttention(nn.Module):
    """Update each undirected pair from its two endpoint neighborhoods.

    A query for (i,j) through i can see (i,k) for every k other than i or j.
    It can therefore learn from a weak prospective bond. Key occupancy is a
    soft prior on the attention scores. Both endpoints use the same projections.
    """

    def __init__(self, width: int, heads: int, head_dim: int):
        super().__init__()
        if min(width, heads, head_dim) < 1:
            raise ValueError("Bond attention dimensions must be positive")
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * inner)
        self.output = nn.Linear(inner, width)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                         nn.Linear(2 * width, width))
        for projection in (self.output, self.feedforward[-1]):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def _endpoint(self, rows, occupancy, anchors, slots, diagonal):
        # Each row fixes one endpoint and contains all candidate incident pairs.
        count, nodes, _ = rows.shape
        qkv = self.qkv(self.attention_norm(rows)).reshape(
            count, nodes, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        eligible_keys = (slots[None] != anchors[:, None]) & (occupancy > 0)
        eligible = eligible_keys[:, None, :] & ~diagonal
        log_occupancy = occupancy.clamp_min(torch.finfo(occupancy.dtype).tiny).log().to(query.dtype)
        bias = log_occupancy[:, None, :].masked_fill(~eligible, torch.finfo(query.dtype).min)
        # Attention works on N keys per query without an [N,N,N,H,D] value tensor.
        context = F.scaled_dot_product_attention(query, key, value, attn_mask=bias[:, None])
        context = context.transpose(1, 2).reshape(count, nodes, -1)
        return self.output(context) * eligible.any(dim=-1, keepdim=True)

    def forward(self, pair, occupancy):
        batch, nodes, neighbors, width = pair.shape
        if nodes != neighbors or occupancy.shape != (batch, nodes):
            raise ValueError("Bond attention requires square pair states and one occupancy per slot")
        slots = torch.arange(nodes, device=pair.device)
        diagonal = torch.eye(nodes, dtype=torch.bool, device=pair.device)
        rows = pair.reshape(batch * nodes, nodes, width)
        # Limit activation size on 32 GB training GPUs and on large evaluation
        # panels. Recompute each chunk in backward instead of saving all Q/K/V.
        chunk_size = 256
        recompute = pair.is_cuda and torch.is_grad_enabled() and len(rows) > chunk_size
        messages = []
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]
            ids = torch.arange(start, start + len(chunk), device=pair.device)
            args = (chunk, occupancy[ids // nodes], ids % nodes, slots, diagonal)
            if recompute:
                message = checkpoint(self._endpoint, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                message = self._endpoint(*args)
            messages.append(message)
        forward = torch.cat(messages).reshape_as(pair)
        gates = (occupancy[:, :, None] * occupancy[:, None, :]).masked_fill(diagonal, 0.)[..., None]
        pair = pair + gates * ((forward + forward.transpose(1, 2)) * 0.5)
        return pair + gates * self.feedforward(self.feedforward_norm(pair))
