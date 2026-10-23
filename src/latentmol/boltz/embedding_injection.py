"""Scoped native early-embedding injection for the frozen epoch-36 bridge.

Only mapped ligand chemical contributions change. Native position/name features,
receptor/cross rows, masks, geometry, inference modules and objective mathematics
remain in their existing paths. The context must encompass both forward and
backward: non-reentrant native activation checkpoints can recompute projections.

This boundary validates tensor alignment, not scientific checkpoint provenance or
learned fidelity. The caller authenticates frozen assets and independently audits
native parity, identity and derivatives on allocated DTU HPC.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F


INJECTION_VERSION = "native-chemical-embedding-injection-v1"
_ACTIVE_INJECTION = ContextVar("latentmol_native_embedding_injection", default=None)


@dataclass
class InjectionReceipt:
    """Counts actual consumed channels, including activation recomputation."""

    expected: tuple[str, ...]
    visited: dict[str, int] = field(default_factory=dict)

    def record(self) -> dict:
        return {"version": INJECTION_VERSION, "expected_channels": list(self.expected),
                "visited_channels": dict(self.visited),
                "all_channels_consumed": set(self.visited) == set(self.expected),
                "fixed_occupancy": True, "native_diagonal_retained": True,
                "ligand_chemical_contributions_only": True}

    def consume(self, key: str) -> None:
        if key not in self.expected:
            raise ValueError("Unexpected native embedding channel: " + key)
        self.visited[key] = self.visited.get(key, 0) + 1

    def require_complete(self) -> None:
        missing = set(self.expected) - set(self.visited)
        if missing:
            raise ValueError("Native embedding channels were not consumed: " + ", ".join(sorted(missing)))


def native_embedding_models(scorer) -> dict[str, nn.Module]:
    """Resolve the existing single-affinity or two-checkpoint DBMol scorer."""
    if hasattr(scorer, "structure") and hasattr(getattr(scorer, "affinity", None), "model"):
        models = {"affinity": scorer.affinity.model, "structure": scorer.structure}
    elif hasattr(scorer, "model"):
        models = {"affinity": scorer.model}
    else:
        raise ValueError("Embedding injection requires a native affinity or DBMol scorer")
    if len({id(model) for model in models.values()}) != len(models):
        raise ValueError("Checkpoint-specific embedding models must be distinct")
    return models


def _sites(models):
    """Fail closed on native layout changes; never guess trained projections."""
    atoms, connectivity, typed, widths = {}, {}, {}, {}

    def bonds(name, owner, prefix):
        linear = owner.token_bonds
        if (not isinstance(linear, nn.Linear) or linear.in_features != 1
                or linear.bias is not None):
            raise ValueError("Unsupported native connectivity projection: " + name)
        key = f"{name}.{prefix}_connectivity"
        if id(linear) in connectivity:
            raise ValueError("Native connectivity modules must not alias")
        connectivity[id(linear)] = (linear, key)
        widths[key] = linear.out_features
        if owner.bond_type_feature:
            embedding = owner.token_bonds_type
            if (not isinstance(embedding, nn.Embedding) or embedding.num_embeddings != 7
                    or embedding.embedding_dim != linear.out_features):
                raise ValueError("Unsupported native typed-bond projection: " + name)
            key = f"{name}.{prefix}_type"
            if id(owner) in typed:
                raise ValueError("Native typed-bond owners must not alias")
            typed[id(owner)] = (owner, key)
            widths[key] = embedding.embedding_dim

    for name, model in models.items():
        if (not isinstance(model, nn.Module) or model.training
                or any(parameter.requires_grad for parameter in model.parameters())):
            raise ValueError("Native embedding models must remain frozen and in eval mode")
        encoder = model.input_embedder.atom_encoder
        linear = encoder.embed_atom_features
        expected = 132 + (0 if encoder.use_no_atom_char else 256)
        if (getattr(encoder, "use_atom_backbone_feat", False)
                or getattr(encoder, "use_residue_feats_atoms", False)
                or not isinstance(linear, nn.Linear) or linear.in_features != expected):
            raise ValueError("Native atom projection disagrees with the pinned feature layout")
        if id(linear) in atoms:
            raise ValueError("Checkpoint-specific atom projections must not alias")
        keys = (name + ".atom_element", name + ".atom_charge")
        atoms[id(linear)] = (linear, keys)
        widths.update({key: linear.out_features for key in keys})
        bonds(name, model, "trunk")
        # FrozenBoltzAffinity intentionally executes no confidence module, even
        # when its checkpoint advertises one. Only the DBMol structure role
        # reaches that head; discovery must follow execution, not saved flags.
        if name == "structure" and getattr(model, "confidence_prediction", False):
            confidence = model.confidence_module
            if confidence.add_z_input_to_z:
                bonds(name, confidence, "confidence")
    return atoms, connectivity, typed, widths


class _Injection:
    def __init__(self, scorer, prepared, predictions):
        if not isinstance(predictions, Mapping):
            raise ValueError("Native embedding predictions must be a named tensor mapping")
        prepared.validate()
        self.atoms, self.connectivity, self.typed, widths = _sites(native_embedding_models(scorer))
        if set(predictions) != set(widths):
            raise ValueError("Native embedding prediction keys disagree with reachable projections")
        self.predictions = dict(predictions)
        self.active = prepared.active.detach().clone()
        self.slots = self.active.nonzero().flatten()
        self.atom_rows = prepared.slot_to_atom[self.active].detach().clone()
        token_rows = prepared.slot_to_token[self.active].detach().clone()
        self.native_atoms = prepared.features["ref_element"].shape[1]
        self.native_tokens = prepared.features["token_pad_mask"].shape[1]
        if (bool(prepared.features["token_bonds"][0, token_rows, token_rows].count_nonzero())
                or bool(prepared.features["type_bonds"][0, token_rows, token_rows].count_nonzero())):
            raise ValueError("Native ligand diagonal must have zero connectivity and NONE type")
        nodes = prepared.spec.max_nodes
        pair_mask = self.active[:, None] & self.active[None, :]
        # Preserve native NONE on the diagonal regardless of approximate heads.
        self.off_diagonal = ~torch.eye(len(self.slots), device=self.slots.device, dtype=torch.bool)
        self.pair_rows = (token_rows[:, None] * self.native_tokens + token_rows[None, :])[self.off_diagonal]
        for key, width in widths.items():
            value = predictions[key]
            atom = key.endswith((".atom_element", ".atom_charge"))
            expected = (1, nodes, width) if atom else (1, nodes, nodes, width)
            if (not isinstance(value, Tensor) or value.shape != expected
                    or not value.is_floating_point() or value.device != self.active.device
                    or not bool(value.isfinite().all())):
                raise ValueError("Malformed native embedding prediction: " + key)
            mask = self.active if atom else pair_mask
            if bool(value.masked_select(~mask[None, ..., None]).count_nonzero()):
                raise ValueError("Absent native embedding prediction rows must be zero: " + key)
            if not atom and not torch.equal(value, value.transpose(1, 2)):
                raise ValueError("Native pair embedding predictions must be symmetric: " + key)
        self.receipt = InjectionReceipt(tuple(sorted(widths)))

    def atom_hook(self, module, arguments, kwargs, output):
        if _ACTIVE_INJECTION.get() is not self:
            raise ValueError("Native atom hook ran outside its active injection context")
        value = arguments[0] if len(arguments) == 1 else kwargs.get("input")
        if (not isinstance(value, Tensor) or not isinstance(output, Tensor)
                or value.shape != (1, self.native_atoms, module.in_features)
                or output.shape != (1, self.native_atoms, module.out_features)):
            raise ValueError("Native atom projection invocation shape changed")
        _, keys = self.atoms[id(module)]
        current = value.index_select(1, self.atom_rows)
        # Remove precisely the chemical columns from the *actual* native input.
        # Bias, reference coordinates and atom-name columns remain untouched.
        element = F.linear(current[..., 4:132], module.weight[:, 4:132])
        charge = F.linear(current[..., 3:4], module.weight[:, 3:4])
        replacement = (output.index_select(1, self.atom_rows) - element - charge
                       + self.predictions[keys[0]].index_select(1, self.slots).to(output)
                       + self.predictions[keys[1]].index_select(1, self.slots).to(output))
        for key in keys:
            self.receipt.consume(key)
        return output.index_copy(1, self.atom_rows, replacement)

    def pair_replace(self, output, key):
        width = self.predictions[key].shape[-1]
        if output.shape != (1, self.native_tokens, self.native_tokens, width):
            raise ValueError("Native pair projection invocation shape changed: " + key)
        predicted = self.predictions[key].index_select(1, self.slots).index_select(2, self.slots)
        predicted = predicted[:, self.off_diagonal].to(output)
        result = output.reshape(1, self.native_tokens * self.native_tokens, width).index_copy(
            1, self.pair_rows, predicted).reshape_as(output)
        self.receipt.consume(key)
        return result

    def connectivity_hook(self, module, arguments, kwargs, output):
        if _ACTIVE_INJECTION.get() is not self:
            raise ValueError("Native connectivity hook ran outside its active injection context")
        value = arguments[0] if len(arguments) == 1 else kwargs.get("input")
        if not isinstance(value, Tensor) or value.shape != (1, self.native_tokens, self.native_tokens, 1):
            raise ValueError("Native connectivity input invocation shape changed")
        diagonal = value[0].diagonal(dim1=0, dim2=1)
        if bool(diagonal.count_nonzero()):
            raise ValueError("Native connectivity diagonal must remain zero")
        _, key = self.connectivity[id(module)]
        return self.pair_replace(output, key)


@contextmanager
def inject_native_embeddings(scorer, prepared, predictions):
    """Temporarily inject all reachable native chemical channels, then restore.

    Keep this context open through autograd/backward recomputation. Successful
    exit requires every declared projection to have actually executed. Failures
    retain their original exception; every hook and ContextVar is still restored.
    Nested contexts are rejected, including contexts on another scorer.
    """
    if _ACTIVE_INJECTION.get() is not None:
        raise ValueError("Nested native embedding injection contexts are unsupported")
    state = _Injection(scorer, prepared, predictions)
    handles, succeeded = [], False
    token = _ACTIVE_INJECTION.set(state)
    try:
        for module, _ in state.atoms.values():
            handles.append(module.register_forward_hook(state.atom_hook, with_kwargs=True))
        for module, _ in state.connectivity.values():
            handles.append(module.register_forward_hook(state.connectivity_hook, with_kwargs=True))
        yield state.receipt
        succeeded = True
    finally:
        for handle in reversed(handles):
            handle.remove()
        _ACTIVE_INJECTION.reset(token)
        if succeeded:
            state.receipt.require_complete()


def bond_type_embedding(module, bond_probabilities):
    """Native expected typed-bond matmul, with scoped ligand-only replacement.

    ``module`` is the owning Boltz trunk or confidence module, rather than its
    nn.Embedding, because existing attached paths intentionally use raw weights.
    Outside a bridge context this is the unchanged historical expression.
    """
    output = bond_probabilities @ module.token_bonds_type.weight
    state = _ACTIVE_INJECTION.get()
    if state is None:
        return output
    entry = state.typed.get(id(module))
    if entry is None or entry[0] is not module:
        raise ValueError("Typed-bond owner is outside the active native embedding schema")
    diagonal = bond_probabilities[0].diagonal(dim1=0, dim2=1)
    if (not torch.equal(diagonal[0], torch.ones_like(diagonal[0]))
            or bool(diagonal[1:].count_nonzero())):
        raise ValueError("Native typed-bond diagonal must remain NONE")
    return state.pair_replace(output, entry[1])


def any_injected_requires_grad() -> bool:
    """Report bridge attachment even when native feature placeholders are fixed."""
    state = _ACTIVE_INJECTION.get()
    return state is not None and any(value.requires_grad for value in state.predictions.values())
