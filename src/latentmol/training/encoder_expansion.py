"""Insert the epoch-48 encoder extension without resetting trained AdamW moments."""

from dataclasses import replace

import torch

from latentmol.models import GraphAutoencoder


def expand_encoder(model, optimizer, *, seed):
    """Graft the new branches once. Existing parameter objects and RNG streams survive."""
    if model.config.encoder_degree_features or model.config.encoder_graph_attention_layers:
        raise ValueError("Encoder already contains a requested extension")
    if len(optimizer.param_groups) != 1:
        raise ValueError("Encoder graft expects the production single AdamW group")
    group = optimizer.param_groups[0]
    if [id(p) for p in group["params"]] != [id(p) for p in model.parameters()]:
        raise ValueError("Unexpected optimizer parameter order before encoder graft")

    config = replace(model.config, encoder_degree_features=True, encoder_graph_attention_layers=3)
    old = model.encoder
    ref = next(old.parameters())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        encoder = GraphAutoencoder(model.spec, config).encoder.to(device=ref.device, dtype=ref.dtype)

    # AdamW keys moments by Parameter object in memory and by position on disk.
    # Transplant original objects, then use the expanded model's canonical order.
    inherited = dict(old.named_parameters())
    expanded = dict(encoder.named_parameters())
    new_names = expanded.keys() - inherited.keys()
    if (not inherited.keys() <= expanded.keys() or "degree_projection.weight" not in new_names
            or not any(name.startswith("graph_attention.") for name in new_names)
            or any(name != "degree_projection.weight" and not name.startswith("graph_attention.")
                   for name in new_names)):
        raise ValueError("Encoder graft found unexpected trainable tensors")
    for name, param in inherited.items():
        if expanded[name].shape != param.shape:
            raise ValueError(f"Incompatible inherited encoder shape for {name}")
        path, _, field = name.rpartition(".")
        setattr(encoder.get_submodule(path), field, param)
    with torch.no_grad():
        encoder.degree_projection.weight.zero_()
        for name, value in old.named_buffers():
            encoder.get_buffer(name).copy_(value)
    encoder.train(old.training)
    model.encoder = encoder
    model.config = config
    group["params"] = list(model.parameters())
    return sum(expanded[name].numel() for name in new_names)
