"""Small reusable latent updater with an explicit normalized coordinate system."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def optimize_latents(start: Tensor, anchor: Tensor, objective: Callable[[Tensor], Tensor],
                     *, steps: int, learning_rate: float, max_move: float) -> tuple[Tensor, dict]:
    """Minimize a per-example objective while bounding movement from shared starts.

    ``start`` and ``anchor`` use the same standardized latent coordinates. The
    objective receives those coordinates and returns one finite scalar per row.
    No model, chemistry, scorer, logging, or study policy is hidden here.
    """
    if start.shape != anchor.shape or start.ndim != 2:
        raise ValueError("Start and anchor must be matching [batch, latent] tensors")
    if steps < 1 or learning_rate <= 0 or max_move <= 0:
        raise ValueError("Invalid latent update budget")
    position = torch.nn.Parameter(start.detach().clone())
    optimizer = torch.optim.Adam([position], lr=learning_rate)
    gradient_norms = []
    first_loss = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = objective(position)
        if losses.shape != (len(start),) or not torch.isfinite(losses).all():
            raise ValueError("Objective must return a finite loss per latent")
        if step == 0:
            first_loss = float(losses.detach().mean())
        losses.mean().backward()
        if position.grad is None or not torch.isfinite(position.grad).all():
            raise ValueError("Latent objective has no finite gradient")
        gradient_norms.append(float(position.grad.detach().norm(dim=-1).mean()))
        optimizer.step()
        with torch.no_grad():
            movement = position - start
            scale = (max_move / movement.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
            position.copy_(start + movement * scale)
    with torch.no_grad():
        final_loss = float(objective(position).mean())
        displacement = float((position - start).norm(dim=-1).mean())
        anchor_distance = float((position - anchor).norm(dim=-1).mean())
    return position.detach(), {"first_loss": first_loss, "final_loss": final_loss,
                               "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
                               "mean_displacement_from_start": displacement,
                               "mean_distance_from_anchor": anchor_distance}
