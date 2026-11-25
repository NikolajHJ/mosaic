"""Choose one molecular probability representation before external branches split.

Policies apply to raw decoder logits, once. Every scorer then receives the same
RelaxedGraph. A model-specific adapter may remap its vocabulary, but must not
independently change temperature, occupancy, formal charge, or hard identity.
"""

import torch
from torch.nn import functional as F

from .representation import DecoderOutput, GraphSpec, RelaxedGraph


def graph_from_mosaic(x: torch.Tensor, e: torch.Tensor, spec: GraphSpec, *, active_slots=None) -> RelaxedGraph:
    """Embed fixed-size neutral Mosaic probabilities into an explicitly ordered DBMol GraphSpec.

    x has C/N/O/S/F/Cl rows. e has NONE/SINGLE/DOUBLE/TRIPLE/AROMATIC rows in
    lexicographic upper-triangle order. active_slots maps each input atom into
    its parent slot, in order. Both input gradients survive the remapping.
    """
    elements = (6, 7, 8, 16, 9, 17)
    bond_types = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[2] != len(elements):
        raise ValueError("Mosaic atoms must have shape [batch, atoms, 6]")
    batch, n, _ = x.shape
    if batch < 1 or not 1 <= n <= spec.max_nodes:
        raise ValueError("The input graph needs a nonempty batch and must fit the parent slots")
    if not isinstance(e, torch.Tensor) or e.shape != (batch, n * (n - 1) // 2, len(bond_types)):
        raise ValueError("Mosaic bonds must have one five-channel row per upper-triangle pair")
    if not x.is_floating_point() or e.dtype != x.dtype or e.device != x.device:
        raise ValueError("Atom and bond probabilities must share a floating dtype and device")
    for rows in (x, e):
        if (not rows.isfinite().all() or (rows < 0).any()
                or not torch.allclose(rows.sum(-1), torch.ones_like(rows[..., 0]), rtol=0, atol=2e-5)):
            raise ValueError("Expected finite, nonnegative, normalized molecular probabilities")
    if not set(elements).issubset(spec.atomic_numbers) or not set(bond_types).issubset(spec.bond_names):
        raise ValueError("Parent graph vocabulary does not contain all Mosaic input categories")
    slots = torch.arange(n, device=x.device) if active_slots is None else torch.as_tensor(active_slots, device=x.device)
    if (slots.shape != (n,) or slots.dtype not in (torch.int32, torch.int64)
            or slots.min() < 0 or slots.max() >= spec.max_nodes or slots.unique().numel() != n):
        raise ValueError("active_slots must map every input atom to a distinct existing parent slot")
    slots = slots.long()
    atom_map = torch.tensor([spec.atomic_numbers.index(v) for v in elements], device=x.device)
    bond_map = torch.tensor([spec.bond_names.index(v) for v in bond_types], device=x.device)
    atoms = x.new_zeros(batch, spec.max_nodes, len(spec.atomic_numbers))
    atoms[..., 0] = 1
    atoms[:, slots, 0] = 0
    atoms[:, slots[:, None], atom_map[None, :]] = x
    charges = x.new_zeros(batch, spec.max_nodes, len(spec.formal_charges))
    charges[..., spec.formal_charges.index(0)] = 1
    bonds = e.new_zeros(batch, spec.max_nodes, spec.max_nodes, len(spec.bond_names))
    bonds[..., 0] = 1
    i, j = torch.triu_indices(n, n, 1, device=x.device)
    left, right = slots[i, None], slots[j, None]
    bonds[:, left, right, bond_map[None, :]] = e
    bonds[:, right, left, bond_map[None, :]] = e
    return RelaxedGraph(atoms, charges, bonds)


def fixed_occupancy_graph(decoded: DecoderOutput, active: torch.Tensor,
                          spec: GraphSpec, temperature: float = 1.0) -> RelaxedGraph:
    """Condition raw decoder logits on a fixed set of existing heavy-atom slots.

    Conditioning precedes occupancy gating. Removing NO_ATOM from the ordinary
    ``probabilities()`` result would leave bonds spuriously attenuated by its old
    soft occupancy. This function deliberately gives NO_ATOM logits no gradient.
    This implementation is moved unchanged from the original Boltz adapter.
    """
    if temperature <= 0 or active.shape != (spec.max_nodes,) or not active.any():
        raise ValueError("Require positive temperature and a nonempty fixed slot mask")
    mask = active.to(device=decoded.atoms.device, dtype=decoded.atoms.dtype)[None]
    real = (decoded.atoms[..., 1:] / temperature).softmax(-1)
    atoms = torch.cat(((1 - mask)[..., None], real * mask[..., None]), -1)
    q = (decoded.charges / temperature).softmax(-1)
    neutral = F.one_hot(torch.tensor(spec.formal_charges.index(0), device=q.device), q.shape[-1]).to(q)
    charges = q * mask[..., None] + neutral * (1 - mask[..., None])
    pair = mask[:, :, None] * mask[:, None, :]
    pair = pair * (1 - torch.eye(spec.max_nodes, device=pair.device)[None])
    conditional = (decoded.symmetric_bond_logits / temperature).softmax(-1)
    none = torch.zeros_like(conditional)
    none[..., 0] = 1
    bonds = conditional * pair[..., None] + none * (1 - pair[..., None])
    return RelaxedGraph(atoms, charges, bonds)


def graph_from_decoder(decoded: DecoderOutput, spec: GraphSpec, *,
                       temperature: float = 1.0,
                       fixed_active: torch.Tensor | None = None) -> RelaxedGraph:
    """Select the shared soft-occupancy or fixed-occupancy graph explicitly.

    This does not give a fixed-occupancy external model variable-size support.
    The legacy fixed policy supports one example; the soft policy supports a
    batch. Eligibility remains the responsibility of each external branch.
    """
    if not isinstance(decoded.atoms, torch.Tensor) or decoded.atoms.ndim != 3 or decoded.atoms.shape[0] < 1:
        raise ValueError("Decoder atom logits require a nonempty [batch, nodes, categories] tensor")
    batch = decoded.atoms.shape[0]
    shapes = ((batch, spec.max_nodes, len(spec.atomic_numbers)),
              (batch, spec.max_nodes, len(spec.formal_charges)),
              (batch, spec.max_nodes, spec.max_nodes, len(spec.bond_names)))
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
           for value, shape in zip((decoded.atoms, decoded.charges, decoded.bonds), shapes)):
        raise ValueError("Decoder logits do not match GraphSpec dimensions and common batch size")
    if fixed_active is None:
        return decoded.probabilities(temperature)
    if (not isinstance(fixed_active, torch.Tensor) or fixed_active.dtype != torch.bool
            or fixed_active.shape != (spec.max_nodes,) or not fixed_active.any()):
        raise ValueError("fixed_active must be a nonempty boolean mask of GraphSpec.max_nodes slots")
    if batch != 1:
        raise ValueError("The fixed-occupancy graph policy currently requires batch size one")
    return fixed_occupancy_graph(decoded, fixed_active, spec, temperature)
