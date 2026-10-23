"""Identity-cloned contact frontiers and early native ReLU witnesses.

This instrumentation does not score chemistry or change production models.
It is dynamically importable before authenticated old-source bootstrap, so
Torch is deliberately imported only inside tensor operations. Numerical
fixtures and actual-model use belong on allocated DTU HPC.

Cuts are complete only for ONE structural input-embedder call and ZERO
recycling: early cuts retain both chemical bond bypasses, whereas every
trunk cut contains sibling sequence and pair states. A held ReLU mask is an
explicit diagnostic counterfactual; it never certifies the historical gate.
"""
from __future__ import annotations

from functools import partial
import math


FIELDS = ("atoms", "charges", "bonds")
ADAPTER_FIELDS = ("ref_element", "ref_charge", "token_bonds", "bond_types")


def selected_indices(count):
    """Return fixed quarter/middle/three-quarter/end native block indices."""
    if type(count) is not int or count < 1:
        raise ValueError("Require a nonempty structural trunk")
    return sorted({max(0, math.ceil(count * fraction) - 1)
                   for fraction in (.25, .5, .75, 1.)})


def cut_layout(count):
    """Declare every frontier, including chemical paths around atom encoding."""
    bypass = ["adapter.token_bonds", "adapter.bond_types"]
    cuts = {"graph": ["graph." + field for field in FIELDS],
            "adapter": ["adapter." + field for field in ADAPTER_FIELDS],
            "atom_encoder": ["atom_encoder." + field for field in ("q", "c", "p")] + bypass,
            "atom_attention": ["atom_attention.a"] + bypass,
            "input_embedder": ["embedder.trunk"] + bypass}
    for index in selected_indices(count):
        cuts[f"trunk_block_{index:02d}"] = [f"trunk.{index}.s", f"trunk.{index}.z"]
    cuts["distogram"] = ["distogram"]
    return cuts


def tensor_clones(value, *, detached=False):
    """Clone tensors recursively without moving them or retaining aliases.

    Only floating tensors may be frontiers; this general helper also copies
    integer/bool witnesses. Opaque native callbacks are returned unchanged.
    """
    import torch
    if isinstance(value, torch.Tensor):
        return (value.detach() if detached else value).clone()
    if isinstance(value, tuple):
        return tuple(tensor_clones(item, detached=detached) for item in value)
    if isinstance(value, list):
        return [tensor_clones(item, detached=detached) for item in value]
    if isinstance(value, dict):
        return {key: tensor_clones(item, detached=detached) for key, item in value.items()}
    return value


def relu_crossings(baseline, endpoint):
    """Summarize exact branch changes; counts alone do not establish causality."""
    import torch
    if set(baseline) != set(endpoint):
        raise ValueError("ReLU witness membership changed")
    rows = {}
    for name, original in baseline.items():
        current = endpoint[name]
        if (original.shape != current.shape or original.dtype != current.dtype
                or original.device != current.device):
            raise ValueError("ReLU witness representation changed: " + name)
        if not bool(original.isfinite().all()) or not bool(current.isfinite().all()):
            raise ValueError("Nonfinite ReLU witness: " + name)
        before, after = original > 0, current > 0
        rows[name] = {"elements": original.numel(),
            "crossed": int((before != after).sum()),
            "inactive_to_active": int((~before & after).sum()),
            "active_to_inactive": int((before & ~after).sum()),
            "baseline_zero": int((original == 0).sum()),
            "endpoint_zero": int((current == 0).sum()),
            "baseline_min_abs": float(original.abs().min()),
            "endpoint_min_abs": float(current.abs().min()),
            "baseline_active": int(before.sum()), "endpoint_active": int(after.sum())}
    return rows


class ContactCapture:
    """Collect one frozen eval structural forward with removable hooks.

    Call ``require_complete`` AFTER the forward and BEFORE backward. That
    freezes telemetry: non-reentrant checkpoint recomputation can then revisit
    internal activations without replacing first-pass witnesses. Held masks
    remain active during recomputation so forward and backward use one declared
    counterfactual function. Context exit removes hooks even after exceptions.
    """
    def __init__(self, structure, *, telemetry=True, held_masks=None):
        import torch
        if structure.training or any(parameter.requires_grad for parameter in structure.parameters()):
            raise ValueError("Require a frozen structural eval model")
        self.structure, self.telemetry = structure, telemetry
        self.values, self.relus, self.handles, self.visits = {}, {}, [], {}
        self.trunk_indices = selected_indices(len(structure.pairformer_module.layers))
        self.cuts = cut_layout(len(structure.pairformer_module.layers))
        self.topology = {"trunk_depth": len(structure.pairformer_module.layers),
                         "trunk_block_indices": self.trunk_indices,
                         "recycling_steps_supported": 0,
                         "embedding_calls_supported": 1}
        self.relu_modules = {name: module for name, module in structure.input_embedder.named_modules()
                             if isinstance(module, torch.nn.ReLU)}
        if not self.relu_modules:
            raise ValueError("No native input-embedder ReLUs found")
        if any(module.inplace for module in self.relu_modules.values()):
            raise ValueError("In-place native ReLUs are unsupported")
        self.held_masks = None if held_masks is None else dict(held_masks)
        if self.held_masks is not None:
            if set(self.held_masks) != set(self.relu_modules):
                raise ValueError("Held ReLU mask membership changed")
            for name, mask in self.held_masks.items():
                if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.requires_grad:
                    raise ValueError("Held ReLU masks must be detached bool tensors: " + name)
            self.held_masks = tensor_clones(self.held_masks, detached=True)
        self.embedding_active = False
        self.forward_complete = False
        self.entered = False

    def save(self, name, value):
        """Return an independent identity clone that the actual path consumes."""
        import torch
        if name in self.values:
            raise ValueError("Repeated contact boundary: " + name)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError("Only floating contact frontiers are captured")
        if self.forward_complete:
            raise ValueError("Contact boundary capture continued after forward completion")
        self.values[name] = value.clone()
        return self.values[name]

    def _visit(self, name):
        self.visits[name] = self.visits.get(name, 0) + 1
        if self.visits[name] != 1:
            raise ValueError("Repeated native contact forward: " + name)

    def _embedding_pre(self, _module, args, kwargs):
        if kwargs.get("affinity", False) or len(args) > 1:
            raise ValueError("Contact capture requires one non-affinity input embedder")
        self._visit("input_embedder")
        self.embedding_active = True

    def _embedding_post(self, _module, _args, _kwargs, output):
        if not self.embedding_active:
            raise ValueError("Unexpected input-embedder completion")
        self.embedding_active = False
        return self.save("embedder.trunk", output)

    def _atom_encoder(self, _module, _args, output):
        if not self.embedding_active or not isinstance(output, tuple) or len(output) != 4:
            raise ValueError("Unexpected native atom-encoder call/output")
        self._visit("atom_encoder")
        # Native q and c initially alias. Separate sibling clones are essential
        # to sum their legitimate independent downstream cotangents exactly once.
        return (*[self.save("atom_encoder." + name, value)
                  for name, value in zip(("q", "c", "p"), output[:3])], output[3])

    def _atom_attention(self, _module, _args, output):
        if not self.embedding_active or not isinstance(output, tuple) or len(output) != 4:
            raise ValueError("Unexpected native atom-attention call/output")
        self._visit("atom_attention")
        # InputEmbedder consumes only a. The unused q/c/callback stay untouched.
        return (self.save("atom_attention.a", output[0]), *output[1:])

    def _trunk(self, index, _module, _args, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("Unexpected structural pairformer output")
        self._visit(f"trunk.{index}")
        return tuple(self.save(f"trunk.{index}.{name}", value)
                     for name, value in zip(("s", "z"), output))

    def _distogram(self, _module, _args, output):
        self._visit("distogram")
        return self.save("distogram", output)

    def _relu_pre(self, name, _module, args):
        if len(args) != 1:
            raise ValueError("Unexpected native ReLU input: " + name)
        if self.forward_complete:
            return  # Checkpoint recomputation is not a second observation.
        if not self.embedding_active:
            raise ValueError("Native ReLU outside structural embedding: " + name)
        self._visit("relu." + name)
        if self.telemetry:
            self.relus[name] = args[0].detach().clone()

    def _relu_held(self, name, _module, args, _output):
        import torch
        value, mask = args[0], self.held_masks[name]
        if mask.shape != value.shape or mask.device != value.device:
            raise ValueError("Held ReLU mask shape/device changed: " + name)
        # where rather than multiplication also preserves native positive zeros
        # in inactive entries for byte-level baseline parity.
        return torch.where(mask, value, torch.zeros_like(value))

    def __enter__(self):
        if self.entered:
            raise ValueError("Contact capture cannot be re-entered")
        self.entered = True
        try:
            # Append each handle immediately: partial registration failures must
            # not leave an earlier hook attached to the live frozen model.
            self.handles.append(self.structure.input_embedder.register_forward_pre_hook(
                self._embedding_pre, with_kwargs=True))
            self.handles.append(self.structure.input_embedder.register_forward_hook(
                self._embedding_post, with_kwargs=True))
            self.handles.append(self.structure.input_embedder.atom_encoder.register_forward_hook(self._atom_encoder))
            self.handles.append(self.structure.input_embedder.atom_attention_encoder.register_forward_hook(self._atom_attention))
            self.handles.append(self.structure.distogram_module.register_forward_hook(self._distogram))
            for index in self.trunk_indices:
                self.handles.append(self.structure.pairformer_module.layers[index].register_forward_hook(
                    partial(self._trunk, index)))
            if self.telemetry or self.held_masks is not None:
                for name, module in self.relu_modules.items():
                    self.handles.append(module.register_forward_pre_hook(partial(self._relu_pre, name)))
                    if self.held_masks is not None:
                        self.handles.append(module.register_forward_hook(partial(self._relu_held, name)))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_args):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def require_complete(self):
        """Authenticate first-forward membership, then seal its telemetry."""
        expected = {name for names in self.cuts.values() for name in names}
        if set(self.values) != expected:
            raise ValueError("Incomplete contact frontier membership")
        if self.embedding_active:
            raise ValueError("Input embedder did not complete")
        required_visits = {"input_embedder", "atom_encoder", "atom_attention", "distogram",
                           *[f"trunk.{index}" for index in self.trunk_indices]}
        if self.telemetry or self.held_masks is not None:
            required_visits |= {"relu." + name for name in self.relu_modules}
        if set(self.visits) != required_visits or any(value != 1 for value in self.visits.values()):
            raise ValueError("Unexpected native contact forward membership")
        if self.telemetry and set(self.relus) != set(self.relu_modules):
            raise ValueError("Incomplete native ReLU telemetry")
        self.forward_complete = True

    require_complete_membership = require_complete

    def relu_masks(self):
        """Return detached baseline masks; exact zeros use native zero derivative."""
        if not self.forward_complete or set(self.relus) != set(self.relu_modules):
            raise ValueError("Complete ReLU telemetry is required for baseline masks")
        return {name: (value > 0).detach().clone() for name, value in self.relus.items()}
