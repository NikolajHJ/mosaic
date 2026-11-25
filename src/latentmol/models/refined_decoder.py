"""Residual slot decoder with optional iterative atom-pair refinement.

All feedback uses soft categorical probabilities. The returned logits obey the
same DecoderOutput contract as the historical one-shot decoder, so hard graph
conversion remains outside the differentiable path. Refinement coordinates
logits but does not turn independent final categorical draws into a joint law.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from latentmol.representation import DecoderOutput, GraphSpec
from .triangle import TrianglePairUpdate


@dataclass(frozen=True)
class DecoderFeatures:
    """Final decoder states for separately trained, checkpoint-bound input bridges.

    This is an additional view of one decoding operation, not an alternative
    molecular identity. Keeping tensors attached permits latent derivatives
    after both the VAE and a supervised bridge have been frozen.
    """

    atom_hidden: torch.Tensor
    pair_hidden: torch.Tensor
    node_context: torch.Tensor
    charge_input: torch.Tensor
    bond_input: torch.Tensor


def symmetric_pair_features(h: torch.Tensor) -> torch.Tensor:
    """Features invariant to exchanging the two endpoints of an atom pair."""
    left, right = h[:, :, None, :], h[:, None, :, :]
    return torch.cat((left + right, (left - right).abs(), left * right), dim=-1)


class ResidualGlobalBlock(nn.Module):
    """Deepen the shared molecule trunk without widening its slot projection."""

    def __init__(self, width: int, expansion: int, *, stabilize: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        # No affine state: the stabilized variant can strictly load historical
        # weights for a diagnostic, while the old forward remains unchanged.
        self.postnorm = nn.LayerNorm(width, elementwise_affine=False) if stabilize else nn.Identity()
        self.update = nn.Sequential(
            nn.Linear(width, expansion), nn.SiLU(),
            nn.Linear(expansion, width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.postnorm(value + self.update(self.norm(value)))


class WideResidualSlotProjection(nn.Module):
    """Keep the expanded trunk wide while giving every output a short z path."""

    def __init__(self, latent: int, global_hidden: int, trunk_hidden: int,
                 output_features: int):
        super().__init__()
        self.initial = nn.Linear(latent, global_hidden)
        self.expand = nn.Linear(global_hidden, trunk_hidden)
        self.latent_skip = nn.Linear(latent, trunk_hidden, bias=False)
        self.output = nn.Linear(trunk_hidden, output_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.silu(self.initial(z))
        hidden = torch.nn.functional.silu(self.expand(hidden)) + self.latent_skip(z)
        return self.output(hidden)


class BondMessageAggregation(nn.Module):
    """Pass one learned neighbor value per real bond type, using conditional bond logits."""

    def __init__(self, width: int, bond_types: int):
        super().__init__()
        if bond_types < 2:
            raise ValueError("Bond messaging needs a NO_BOND class and at least one real bond type")
        self.real_types = bond_types - 1
        self.values = nn.Linear(width, self.real_types * width, bias=False)

    def forward(self, node_context, conditional_bonds, occupancy):
        batch, nodes, width = node_context.shape
        # The bond head is shared with the final output. NO_BOND carries zero
        # message. A finite weak bond remains available throughout refinement.
        weights = conditional_bonds.softmax(-1)[..., 1:]
        pair_occupancy = occupancy[:, :, None] * occupancy[:, None, :]
        diagonal = torch.eye(nodes, device=node_context.device, dtype=torch.bool)
        weights = weights * pair_occupancy.masked_fill(diagonal, 0.)[..., None]
        values = self.values(node_context).reshape(batch, nodes, self.real_types, width)
        messages = torch.einsum("bijt,bjth->bih", weights, values)
        # Smooth degree scaling controls the initial dense proposal. Keeping the
        # 1 also preserves edge-strength information for a lone weak neighbor.
        degree = weights.sum(dim=(-1, -2))
        return messages * (1. + degree).rsqrt()[..., None]


class PairRefinementLayer(nn.Module):
    """Update undirected pairs, then aggregate attention or predicted typed bonds.

    The attention variant normalizes over neighbors and scores chemical bonds at
    the decoder output. The bond-message variant reuses that final bond classifier
    inside every round so messages use the same single/double/etc. vocabulary.
    An optional triangle branch refines the pair state before either aggregation.
    """

    def __init__(self, slot_hidden: int, pair_hidden: int, heads: int,
                 *, stabilize: bool = False, deep_mlp: bool = False,
                 triangle_hidden: int | None = None, atom_refinement_dropout: float = 0.,
                 message_bond_types: int | None = None):
        super().__init__()
        self.heads = heads
        self.head_width = slot_hidden // heads
        self.edge_norm = nn.LayerNorm(pair_hidden)
        self.edge_postnorm = (nn.LayerNorm(pair_hidden, elementwise_affine=False)
                              if stabilize else nn.Identity())
        self.edge_update = nn.Sequential(
            nn.Linear(pair_hidden + 3 * slot_hidden, slot_hidden), nn.SiLU(),
            *([nn.Linear(slot_hidden, slot_hidden), nn.SiLU()] if deep_mlp else []),
            nn.Linear(slot_hidden, pair_hidden))
        self.triangle = (TrianglePairUpdate(pair_hidden, triangle_hidden)
                         if triangle_hidden is not None else None)
        self.node_norm = nn.LayerNorm(slot_hidden)
        self.node_postnorm = (nn.LayerNorm(slot_hidden, elementwise_affine=False)
                              if stabilize else nn.Identity())
        self.bond_messages = (None if message_bond_types is None else
                              BondMessageAggregation(slot_hidden, message_bond_types))
        if self.bond_messages is None:
            self.query = nn.Linear(slot_hidden, slot_hidden)
            self.key = nn.Linear(slot_hidden, slot_hidden)
            self.value = nn.Linear(slot_hidden, slot_hidden)
            self.edge_bias = nn.Linear(pair_hidden, heads)
            self.edge_value = nn.Linear(pair_hidden, slot_hidden)
        self.node_update = nn.Sequential(
            nn.Linear(slot_hidden, slot_hidden), nn.SiLU(),
            *([nn.Linear(slot_hidden, slot_hidden), nn.SiLU()] if deep_mlp else []),
            nn.Linear(slot_hidden, slot_hidden))
        self.atom_update_dropout = nn.Dropout(atom_refinement_dropout)

    def forward(self, h: torch.Tensor, edge: torch.Tensor,
                node_context: torch.Tensor, occupancy: torch.Tensor, *, bond_head=None):
        nodes = h.shape[1]
        pair_features = symmetric_pair_features(node_context)
        delta = self.edge_update(torch.cat((self.edge_norm(edge), pair_features), dim=-1))
        if self.triangle is not None:
            # Both updates live inside one residual/norm. Zero-initialized
            # triangle output therefore starts as an identity extra branch.
            delta = delta + self.triangle(edge, occupancy)
        edge = self.edge_postnorm(edge + delta)
        # A pair has one state regardless of endpoint order, including after
        # each residual update. Diagonal states cannot represent self-bonds.
        edge = (edge + edge.transpose(1, 2)) * 0.5
        diagonal = torch.eye(nodes, device=h.device, dtype=torch.bool)[None, :, :, None]
        edge = edge.masked_fill(diagonal, 0.0)

        if self.bond_messages is None:
            messages = self._attention_messages(self.node_norm(h), edge, occupancy)
        else:
            conditional_bonds = bond_head(torch.cat((edge, pair_features), dim=-1))
            messages = self.bond_messages(self.node_norm(node_context), conditional_bonds, occupancy)
        h = self.node_postnorm(h + self.atom_update_dropout(self.node_update(messages)))
        return h, edge

    def _attention_messages(self, x, edge, occupancy):
        batch, nodes, width = x.shape
        q = self.query(x).reshape(batch, nodes, self.heads, self.head_width)
        k = self.key(x).reshape(batch, nodes, self.heads, self.head_width)
        v = self.value(x).reshape(batch, nodes, self.heads, self.head_width)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.head_width)
        scores = scores + self.edge_bias(edge).permute(0, 3, 1, 2)
        # A soft occupancy bias suppresses absent keys without removing the
        # gradient that allows a currently absent slot to become occupied.
        scores = scores + occupancy.clamp_min(1e-6).log()[:, None, None, :]
        weights = scores.softmax(dim=-1)
        pair_values = self.edge_value(edge).reshape(
            batch, nodes, nodes, self.heads, self.head_width).permute(0, 3, 1, 2, 4)
        atom_values = v.permute(0, 2, 1, 3)[:, :, None, :, :]
        messages = (weights[..., None] * (atom_values + pair_values)).sum(dim=3)
        return messages.permute(0, 2, 1, 3).reshape(batch, nodes, width)


class RefinedGraphDecoder(nn.Module):
    """Decode z to atom, charge and bond logits with residual soft feedback.

    With zero pair-refinement rounds this is the width-and-residual control.
    Positive rounds evolve hidden pair states and feed them back to atom slots.
    Parameters are shared over slots and pairs within each round.
    """

    def __init__(self, spec: GraphSpec, slot_hidden: int, latent: int,
                 global_hidden: int, pair_hidden: int, rounds: int, heads: int,
                 *, trunk_hidden: int | None = None, global_bond_context: bool = False,
                 wide_trunk: bool = False, stable_refinement: bool = False,
                 direct_slots: bool = False, deep_refinement: bool = False,
                 triangle_hidden: int | None = None, atom_refinement_dropout: float = 0.,
                 bond_messages: bool = False, extra_bond_rounds: int = 0):
        super().__init__()
        if min(slot_hidden, latent, global_hidden, pair_hidden, heads) < 1 or rounds < 0:
            raise ValueError("Refined decoder widths/heads must be positive and rounds nonnegative")
        if global_hidden < latent or (not bond_messages and slot_hidden % heads):
            raise ValueError("Global width must cover latent and heads must divide slot width")
        if trunk_hidden is not None and trunk_hidden < global_hidden:
            raise ValueError("Shared trunk expansion must cover global width")
        if global_bond_context and trunk_hidden is None:
            raise ValueError("Global bond context requires the deep shared trunk")
        if wide_trunk and not global_bond_context:
            raise ValueError("Wide trunk requires a global bond context")
        if stable_refinement and not direct_slots and (trunk_hidden is None or wide_trunk):
            raise ValueError("Stabilized refinement requires the narrow deep trunk")
        if direct_slots and (trunk_hidden is not None or global_bond_context or wide_trunk):
            raise ValueError("Direct slots do not use a molecule trunk or separate context slot")
        if bond_messages and not direct_slots:
            raise ValueError("Bond-message decoder requires direct slots")
        self.spec = spec
        self.global_bond_context = global_bond_context
        self.wide_trunk = wide_trunk
        self.stable_refinement = stable_refinement
        self.bond_messages = bond_messages
        output_slots = spec.max_nodes + int(global_bond_context)
        if direct_slots:
            # Every slot has a direct affine path to z; all additional nonlinear
            # processing happens in the local residual/refinement blocks.
            self.nodes = nn.Linear(latent, output_slots * slot_hidden)
        elif wide_trunk:
            self.nodes = WideResidualSlotProjection(
                latent, global_hidden, trunk_hidden, output_slots * slot_hidden)
        else:
            shared = [nn.Linear(latent, global_hidden), nn.SiLU()]
            if trunk_hidden is not None:
                shared.append(ResidualGlobalBlock(global_hidden, trunk_hidden,
                                                  stabilize=stable_refinement))
            shared.append(nn.Linear(global_hidden, output_slots * slot_hidden))
            self.nodes = nn.Sequential(*shared)
        self.bond_context_norm = nn.LayerNorm(slot_hidden) if global_bond_context else None
        self.bond_context_direct = (nn.Linear(latent, slot_hidden, bias=False)
                                    if wide_trunk else None)
        self.bond_context_update = (nn.Sequential(
            nn.Linear(slot_hidden, slot_hidden), nn.SiLU(),
            nn.Linear(slot_hidden, slot_hidden)) if wide_trunk else None)
        self.node_norm = nn.LayerNorm(slot_hidden)
        self.slot_postnorm = (nn.LayerNorm(slot_hidden, elementwise_affine=False)
                              if stable_refinement else nn.Identity())
        self.context_norm = (nn.LayerNorm(slot_hidden, elementwise_affine=False)
                             if stable_refinement else nn.Identity())
        self.node_residual = nn.Sequential(
            nn.Linear(slot_hidden, slot_hidden), nn.SiLU(),
            nn.Linear(slot_hidden, slot_hidden))
        self.atom_head = nn.Linear(slot_hidden, len(spec.atomic_numbers))
        self.atom_feedback = nn.Linear(len(spec.atomic_numbers), slot_hidden, bias=False)
        self.charge_head = nn.Linear(2 * slot_hidden, len(spec.formal_charges))
        self.charge_feedback = nn.Linear(len(spec.formal_charges), slot_hidden, bias=False)
        self.pair_init = nn.Sequential(
            nn.Linear(3 * slot_hidden, pair_hidden), nn.SiLU(),
            nn.Linear(pair_hidden, pair_hidden))
        self.rounds = nn.ModuleList(
            PairRefinementLayer(slot_hidden, pair_hidden, heads,
                                stabilize=stable_refinement, deep_mlp=deep_refinement,
                                atom_refinement_dropout=atom_refinement_dropout,
                                message_bond_types=len(spec.bond_names) if bond_messages else None,
                                triangle_hidden=triangle_hidden) for _ in range(rounds))
        self.bond_head = nn.Sequential(
            nn.LayerNorm(pair_hidden + 3 * slot_hidden
                         + (slot_hidden if global_bond_context else 0)),
            nn.Linear(pair_hidden + 3 * slot_hidden
                      + (slot_hidden if global_bond_context else 0), slot_hidden), nn.SiLU(),
            nn.Linear(slot_hidden, len(spec.bond_names)))
        self.extra_rounds = nn.ModuleList()
        for _ in range(extra_bond_rounds):
            block = PairRefinementLayer(slot_hidden, pair_hidden, heads, stabilize=True, deep_mlp=True,
                                        triangle_hidden=triangle_hidden, message_bond_types=len(spec.bond_names))
            # Each new block starts as an identity on the trained decoder.
            block.atom_gate = nn.Parameter(torch.zeros(()))
            block.pair_gate = nn.Parameter(torch.zeros(()))
            self.extra_rounds.append(block)

    def _node_predictions(self, h: torch.Tensor):
        atoms = self.atom_head(h)
        atom_context = self.atom_feedback(atoms.softmax(dim=-1))
        charges = self.charge_head(torch.cat((h, atom_context), dim=-1))
        charge_context = self.charge_feedback(charges.softmax(dim=-1))
        return atoms, charges, self.context_norm(h + atom_context + charge_context)

    def forward(self, z: torch.Tensor, *, return_features: bool = False):
        slots = self.nodes(z).reshape(
            -1, self.spec.max_nodes + int(self.global_bond_context),
            self.atom_head.in_features)
        bond_context = None
        if self.global_bond_context:
            if self.wide_trunk:
                # This explicit z skip and the identity around the two 96-wide
                # context layers give bond gradients a short path to z.
                base_context = slots[:, -1] + self.bond_context_direct(z)
                bond_context = base_context + self.bond_context_update(
                    self.bond_context_norm(base_context))
            else:
                bond_context = self.bond_context_norm(slots[:, -1])
        h = slots[:, :self.spec.max_nodes]
        h = self.slot_postnorm(h + self.node_residual(self.node_norm(h)))
        atoms, charges, node_context = self._node_predictions(h)
        edge = self.pair_init(symmetric_pair_features(node_context))
        diagonal = torch.eye(self.spec.max_nodes, device=z.device, dtype=torch.bool)[None, :, :, None]
        edge = edge.masked_fill(diagonal, 0.0)
        for layer in self.rounds:
            h, edge = layer(h, edge, node_context, 1.0 - atoms.softmax(dim=-1)[..., 0],
                            bond_head=self.bond_head if self.bond_messages else None)
            atoms, charges, node_context = self._node_predictions(h)
        for layer in self.extra_rounds:
            next_h, next_edge = layer(h, edge, node_context, 1.0 - atoms.softmax(-1)[..., 0],
                                      bond_head=self.bond_head)
            h = h + layer.atom_gate * (next_h - h)
            edge = edge + layer.pair_gate * (next_edge - edge)
            atoms, charges, node_context = self._node_predictions(h)
        bond_inputs = [edge, symmetric_pair_features(node_context)]
        if bond_context is not None:
            # One molecule-wide vector reaches every pair directly from z.
            # Broadcasting it equally to (i,j) and (j,i) preserves symmetry.
            bond_inputs.append(bond_context[:, None, None, :].expand(
                -1, self.spec.max_nodes, self.spec.max_nodes, -1))
        bond_input = torch.cat(bond_inputs, dim=-1)
        bonds = self.bond_head(bond_input)
        decoded = DecoderOutput(atoms, charges, bonds)
        if return_features:
            atom_context = self.atom_feedback(atoms.softmax(dim=-1))
            return decoded, DecoderFeatures(
                atom_hidden=h, pair_hidden=edge, node_context=node_context,
                charge_input=torch.cat((h, atom_context), dim=-1),
                bond_input=bond_input)
        return decoded

    def decode_with_features(self, z: torch.Tensor):
        """Return ordinary logits and their final hidden states in one pass."""
        return self.forward(z, return_features=True)
