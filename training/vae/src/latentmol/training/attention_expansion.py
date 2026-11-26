"""Widen existing atom heads and graft two bond-attention blocks onto a trained VAE."""

from dataclasses import replace

import torch

from latentmol.models import GraphAutoencoder


def _head_geometry(name, width):
    """Return the changed axis and packing of a known parameter."""
    module, section, _, field = name.split(".", 3)
    if (module, section) == ("encoder", "graph_attention"):
        if field in ("qkv.weight", "qkv.bias"):
            return 0, 3, width, False
        if field == "attention_output.weight":
            return 1, 1, width, True
        if field in ("distance_bias.weight", "bond_bias.weight"):
            return 1, 1, 1, False
    if (module, section) == ("decoder", "rounds"):
        if field in ("query.weight", "query.bias", "key.weight", "key.bias",
                     "value.weight", "value.bias", "edge_value.weight", "edge_value.bias"):
            return 0, 1, width, False
        if field in ("edge_bias.weight", "edge_bias.bias"):
            return 0, 1, 1, False
        if field == "node_update.0.weight":
            return 1, 1, width, True
    raise ValueError(f"Unrecognized changed attention tensor: {name}")


def _replicate(value, name, head_ids, old_heads, head_width, moment):
    """Repack Q/K/V and head biases. Share outgoing columns over equivalent copies."""
    axis, groups, channels, outgoing = _head_geometry(name, head_width)
    moved = value.movedim(axis, -1)
    if moved.size(-1) != groups * old_heads * channels:
        raise ValueError(f"Unexpected head packing for {name}: {tuple(value.shape)}")
    source = moved.unflatten(-1, (groups, old_heads, channels))
    head_ids = head_ids.to(value.device)
    copies = torch.bincount(head_ids, minlength=old_heads).index_select(0, head_ids)
    result = source.index_select(-2, head_ids)
    if outgoing and moment == 0:
        # A balanced, nonuniform division breaks clone symmetry while preserving
        # the total contribution of each trained head. Opposite signs alternate
        # across output features so the new gradient directions can differ.
        ordinal = torch.arange(len(head_ids), device=value.device) // old_heads
        centered = (2 * ordinal - (copies - 1)).to(value.dtype)
        sign = (1 - 2 * (torch.arange(moved.shape[0], device=value.device) % 2)).to(value.dtype)
        share = (1 + .125 * sign[:, None] * centered[None, :]) / copies
        result = result * share[:, None, :, None]
    elif not outgoing and moment:
        # For clone-internal gradients, the even division is the parent-gradient
        # estimate. The small symmetry-breaking split has no historical moments.
        result = result / copies.to(value.dtype).pow(moment)[:, None]
    # Outgoing-column gradients receive the same input as the old column at the
    # fork, so their first and second moment estimates are simply repeated.
    return result.flatten(-3).movedim(-1, axis).contiguous()


def expanded_attention_config(old_config):
    """Give the graft and subsequent checkpoint restores one architecture."""
    if (old_config.attention_heads != 4 or old_config.bond_attention_layers != 0
            or old_config.encoder_graph_attention_layers != 3 or old_config.refinement_rounds != 4
            or old_config.decoder_variant != "triangle_refine" or old_config.extra_bond_rounds != 3):
        raise ValueError("Expected the three-block encoder and seven-round decoder with four atom heads")
    head_width = old_config.attention_head_width or old_config.hidden // old_config.attention_heads
    return replace(old_config, attention_heads=10, attention_head_width=head_width,
                   bond_attention_layers=2, bond_attention_heads=10, bond_attention_head_dim=24)


@torch.no_grad()
def expand_attention(model, optimizer, *, seed):
    """Expand four equal-width heads to ten and add two identity residual blocks.

    The trained weights, unchanged Parameter objects, optimizer steps and RNG
    streams survive the graft. AdamW moment estimates for widened parameters
    are mapped by head; new bond blocks receive AdamW state on their first
    update. Later training trajectories can differ immediately.
    """
    old_config = model.config
    config = expanded_attention_config(old_config)
    head_width = config.attention_head_width
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 1:
        raise ValueError("Attention expansion expects the production single AdamW group")
    group = optimizer.param_groups[0]
    original = dict(model.named_parameters())
    if [id(p) for p in group["params"]] != [id(p) for p in original.values()]:
        raise ValueError("AdamW parameter order differs from the source model")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        expanded = GraphAutoencoder(model.spec, config)
    reference = next(model.parameters())
    expanded.to(device=reference.device, dtype=reference.dtype)
    params = dict(expanded.named_parameters())
    new_names = params.keys() - original.keys()
    if (not original.keys() <= params.keys() or not new_names
            or any(not name.startswith("decoder.bond_attention.") for name in new_names)):
        raise ValueError("Unexpected parameter names after attention expansion")
    head_ids = torch.arange(config.attention_heads) % old_config.attention_heads
    states = {}
    for name, old in original.items():
        dest = params[name]
        if dest.shape == old.shape:
            module, _, attr = name.rpartition(".")
            setattr(expanded.get_submodule(module), attr, old)
            continue
        weight = _replicate(old, name, head_ids, old_config.attention_heads, head_width, 0)
        if weight.shape != dest.shape:
            raise ValueError(f"Widened shape does not agree with destination for {name}")
        dest.copy_(weight)
        dest.requires_grad_(old.requires_grad)
        moments = {}
        for key, val in optimizer.state.get(old, {}).items():
            if key == "step":
                moments[key] = val
                continue
            powers = {"exp_avg": 1, "exp_avg_sq": 2, "max_exp_avg_sq": 2}
            if key not in powers or val.shape != old.shape:
                raise ValueError(f"Unsupported AdamW state {key!r} for {name}")
            moments[key] = _replicate(val, name, head_ids, old_config.attention_heads, head_width, powers[key])
        states[old] = (dest, moments)
    for name, buffer in model.named_buffers():
        expanded.get_buffer(name).copy_(buffer)
    expanded.train(model.training)
    for old, (dest, moments) in states.items():
        optimizer.state.pop(old, None)
        if moments:
            optimizer.state[dest] = moments
    model.encoder, model.decoder, model.config = expanded.encoder, expanded.decoder, config
    group["params"] = list(model.parameters())
    return sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in original.values())
