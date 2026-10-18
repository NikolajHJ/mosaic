"""Differentiable, rule-based local defect proxies for decoder continuation.

No RDKit call, discrete graph construction, or sampled repair lies in this
forward path. The principal term is the expectation of local *bond-order
excess*, not a distance to the nearest valid molecule or a count of bond-label
edits. A capped categorical convolution preserves its exact overflow first
moment; applying a hinge to the expected valence would lose risky tails.

The finite element/charge rules below follow the RDKit Book's allowed-valence
table (as of 2024.09.1), its isoelectronic charge rule, and its P/S exceptions:
https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences
The effective-atomic-number and cross-period charge checks were also reviewed
in the official Atom.cpp on 2026-10-07:
https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Atom.cpp
This explicit table is versioned here; installing another RDKit does not
silently change training. Held-out RDKit audits remain evaluation work.

Undercoordination costs zero because the heavy-atom graph omits hydrogen and
implicit hydrogen may fill a permitted valence. AROMATIC contributes only one
bond-order unit, an optimistic lower bound: aromatic valence, Kekulization,
rings, radicals, stereochemistry and globally consistent repairs are outside
scope. Frozen-reference occupancy/aromatic anchors belong in the caller and
should be applied identically to continuation controls. Decreasing this loss
alone cannot establish chemical validity or model improvement.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from latentmol.representation import DecoderOutput, GraphSpec


RULE_VERSION = "rdkit_book_2024_09_1_explicit_charge_v1"
RULE_SOURCES = (
    "https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences",
    "https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Atom.cpp",
)

# Inner keys are actual formal charges, never vocabulary indices. None means
# unsupported, not an unlimited or guessed cap. In particular, halogen -2
# would move the isoelectronic element to the next periodic-table row.
ALLOWED_VALENCES: dict[int, dict[int, tuple[int, ...] | None]] = {
    6: {-2: (2,), -1: (3,), 0: (4,), 1: (3,), 2: (2,)},
    7: {-2: (1,), -1: (2,), 0: (3,), 1: (4,), 2: (3,)},
    8: {-2: (0,), -1: (1,), 0: (2,), 1: (3,), 2: (4,)},
    9: {-2: None, -1: (0,), 0: (1,), 1: (2,), 2: (3,)},
    15: {-2: (1, 3, 5), -1: (2, 4, 6), 0: (3, 5), 1: (4,), 2: (3,)},
    16: {-2: (0, 2, 4), -1: (1, 3, 5), 0: (2, 4, 6), 1: (3, 5), 2: (4,)},
    17: {-2: None, -1: (0,), 0: (1,), 1: (2, 4, 6), 2: (3, 5)},
    35: {-2: None, -1: (0,), 0: (1,), 1: (2, 4, 6), 2: (3, 5)},
    53: {-2: None, -1: (0, 2, 4, 6), 0: (1, 3, 5), 1: (2, 4, 6), 2: (3, 5)},
}


@dataclass(frozen=True)
class ChemistryRegularizationConfig:
    """Explicit nonnegative component weights; zero disables only that term.

    Valence excess is divided by three, the largest single bond-order change;
    a one-unit violation remains visible and severe excess can exceed one.
    Other normalized components are approximately on a unit scale. Size anchoring
    uses a supplied detached reference, with squared error divided by
    max(reference_size,1)^2; it can exceed one. An all-empty graph incurs a unit
    empty-graph term instead of receiving a perfect local-defect score. These
    terms complement reconstruction and reference anchors, not replace them.
    """

    version: str = "probability_valence_v1"
    valence_weight: float = 1.0
    charge_support_weight: float = 1.0
    empty_weight: float = 1.0
    isolated_weight: float = 1.0
    ghost_weight: float = 0.0
    size_weight: float = 0.0

    def __post_init__(self):
        if self.version != "probability_valence_v1":
            raise ValueError("Unsupported chemistry regularization version")
        for name, value in asdict(self).items():
            if name == "version":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")

    @classmethod
    def from_mapping(cls, raw: Mapping) -> "ChemistryRegularizationConfig":
        """Reject unknown fields instead of silently ignoring misspelled weights."""
        if not isinstance(raw, Mapping):
            raise ValueError("Chemistry regularization config must be a mapping")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown chemistry regularization fields: {sorted(unknown)}")
        return cls(**dict(raw))

    def record(self) -> dict:
        return asdict(self)


def rule_record(spec: GraphSpec) -> dict:
    """Return all represented rules and declared normalization/feature loss."""
    _validate_spec(spec)
    return {
        "rule_version": RULE_VERSION,
        "sources": list(RULE_SOURCES),
        "reviewed_date": "2026-10-07",
        "element_charge_allowed_valences": {
            str(element): {str(charge): list(ALLOWED_VALENCES[element][charge])
                           if ALLOWED_VALENCES[element][charge] is not None else None
                           for charge in spec.formal_charges}
            for element in spec.atomic_numbers[1:]
        },
        "valence_distance": "E[(incident_bond_order_sum - max_allowed_valence)_+]",
        "implicit_hydrogen": "Undercoordination is unpenalized; no H/radical inference",
        "aromatic_order_policy": "optimistic_order_one_lower_bound; exact aromatic chemistry unsupported",
        "unsupported_charge_policy": "Separate one-category support penalty; no invented bond-order cap",
        "occupancy_policy": "Condition on focal atom presence; gate neighbors once; multiply focal atom mass once",
        "local_normalization": "Per graph sum / max(E[occupied_atoms],1); each endpoint contributes separately",
        "valence_scale": 3,
        "isolated_scope": "Exact expected isolated atoms in multi-atom draws; disconnected larger components can be missed",
        "ghost_scope": "Raw conditional non-NONE pressure on absent endpoint pairs; shared graph still gates occupancy once",
        "validity_guarantee": False,
        "globally_minimal_edit_distance": False,
    }


def _validate_spec(spec: GraphSpec) -> None:
    if len(spec.atomic_numbers) < 2 or any(element not in ALLOWED_VALENCES for element in spec.atomic_numbers[1:]):
        raise ValueError("Chemistry rules support only the explicit heavy-element vocabulary")
    if any(charge not in (-2, -1, 0, 1, 2) for charge in spec.formal_charges):
        raise ValueError("Chemistry rules support only formal charges -2 through 2")
    if set(spec.bond_names) - {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
        raise ValueError("Chemistry rules do not support additional bond categories")


def _logit_probabilities(output: DecoderOutput, spec: GraphSpec, temperature: float) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite positive number")
    n = spec.max_nodes
    values = (output.atoms, output.charges, output.bonds)
    batch = output.atoms.shape[0] if output.atoms.ndim == 3 else 0
    expected = ((batch, n, len(spec.atomic_numbers)), (batch, n, len(spec.formal_charges)),
                (batch, n, n, len(spec.bond_names)))
    if batch < 1 or any(tuple(value.shape) != shape for value, shape in zip(values, expected)):
        raise ValueError("Decoder output shapes do not match the graph specification")
    if any(not value.is_floating_point() for value in values):
        raise ValueError("Decoder logits must be floating point")
    if any(value.device != values[0].device or value.dtype != values[0].dtype for value in values):
        raise ValueError("Decoder logits must share device and dtype")
    # Negative infinity is allowed for exact hard-probability audit fixtures.
    # Reject an all-negative-infinity row before softmax can produce NaNs.
    if any(bool(torch.isnan(value).any() | torch.isposinf(value).any()) or
           not bool(torch.isfinite(value).any(-1).all()) for value in values):
        raise ValueError("Logit rows require finite support and no NaN/positive infinity")
    dtype = torch.float32 if values[0].dtype in (torch.float16, torch.bfloat16) else values[0].dtype
    atoms = (output.atoms.to(dtype) / temperature).softmax(-1)
    charges = (output.charges.to(dtype) / temperature).softmax(-1)
    # This is the same undirected conditional distribution as DecoderOutput.
    # Its occupancy-gated RelaxedGraph view is intentionally not used here.
    symmetric = output.symmetric_bond_logits.to(dtype)
    if not bool(torch.isfinite(symmetric).any(-1).all()):
        raise ValueError("Symmetric bond rows require finite categorical support")
    bonds = (symmetric / temperature).softmax(-1)
    return atoms, charges, bonds


def capped_bond_order_distribution(edge_orders: Tensor, cap: int) -> tuple[Tensor, Tensor, Tensor]:
    """Exact categorical sum law up to cap, plus overflow mass and first moment.

    ``edge_orders[..., neighbor, order]`` uses order categories 0,1,2,3.
    ``low[..., v]`` is P(V=v), v<=cap. The other outputs are P(V>cap) and
    E[V 1(V>cap)], not a clamped overflow approximation. Multiplication and
    fixed shifts avoid CUDA scatter/reduction kernels with nondeterministic
    accumulation. Internal callers provide normalized probability rows.
    """
    if type(cap) is not int or cap < 0:
        raise ValueError("DP cap must be a nonnegative integer")
    if edge_orders.ndim < 2 or edge_orders.shape[-1] != 4 or not edge_orders.is_floating_point():
        raise ValueError("Expected categorical order probabilities [..., neighbor, 4]")
    # A fresh constant state keeps the autograd graph free of in-place writes.
    prefix = edge_orders.shape[:-2]
    low = edge_orders.new_zeros((*prefix, cap + 1))
    low = low + F.one_hot(torch.tensor(0, device=edge_orders.device), cap + 1).to(edge_orders.dtype)
    overflow_mass = edge_orders.new_zeros(prefix)
    overflow_moment = edge_orders.new_zeros(prefix)
    states = torch.arange(cap + 1, device=edge_orders.device, dtype=edge_orders.dtype)
    for neighbor in range(edge_orders.shape[-2]):
        probabilities = edge_orders[..., neighbor, :]
        additions = []
        newly_overflow_mass = overflow_mass * 0
        newly_overflow_moment = overflow_moment * 0
        for order in range(4):
            probability = probabilities[..., order]
            if order <= cap:
                additions.append(F.pad(low[..., :cap + 1 - order], (order, 0)) * probability[..., None])
            else:
                additions.append(low * 0)
            cut = max(0, cap + 1 - order)
            tail = low[..., cut:]
            newly_overflow_mass = newly_overflow_mass + tail.sum(-1) * probability
            newly_overflow_moment = newly_overflow_moment + (tail * (states[cut:] + order)).sum(-1) * probability
        expected_increment = (probabilities * probabilities.new_tensor((0, 1, 2, 3))).sum(-1)
        total_probability = probabilities.sum(-1)
        overflow_moment = (overflow_moment * total_probability + overflow_mass * expected_increment
                           + newly_overflow_moment)
        overflow_mass = overflow_mass * total_probability + newly_overflow_mass
        low = sum(additions)
    return low, overflow_mass, overflow_moment


def chemistry_regularization_losses(output: DecoderOutput, spec: GraphSpec,
                                    config: ChemistryRegularizationConfig | None = None, *,
                                    temperature: float = 1.0, reference_size: Tensor | None = None) -> dict:
    """Return scalar losses, per-graph values, diagnostics and the rule receipt.

    This evaluates the decoder's factorized categorical law, not merely its
    mean graph. Incident bonds are conditional on the focal slot being real,
    so only neighbor occupancy appears inside the convolution. Element and
    charge probabilities then weight each permitted rule jointly. Do not feed
    already occupancy-gated bonds back into this routine.

    If size_weight is positive, reference_size must be a finite detached [B]
    tensor (e.g. the old frozen decoder's expected size at the same prior z).
    No value is guessed from a different molecule or branch. A reference size
    anchors count only; the caller separately retains reconstruction and any
    shared frozen-reference occupancy/aromatic probability anchors.
    """
    config = ChemistryRegularizationConfig() if config is None else config
    if not isinstance(config, ChemistryRegularizationConfig):
        raise ValueError("Expected a ChemistryRegularizationConfig")
    _validate_spec(spec)
    atoms, charges, bonds = _logit_probabilities(output, spec, temperature)
    batch, n = atoms.shape[:2]
    occupancy = atoms[..., 1:].sum(-1)  # Avoid cancellation for tiny real-atom mass.
    expected_size = occupancy.sum(-1)
    denominator = expected_size.clamp_min(1)
    diagonal = torch.eye(n, device=atoms.device, dtype=torch.bool)
    neighbor_occupancy = occupancy[:, None, :].expand(batch, n, n).masked_fill(diagonal, 0)
    order = {name: index for index, name in enumerate(spec.bond_names)}
    none_probability = bonds[..., order["NONE"]]
    real_edge_probability = bonds[..., 1:].sum(-1)
    zero = torch.zeros_like(none_probability)
    aromatic = bonds[..., order["AROMATIC"]] if "AROMATIC" in order else zero
    single = bonds[..., order["SINGLE"]] if "SINGLE" in order else zero
    double = bonds[..., order["DOUBLE"]] if "DOUBLE" in order else zero
    triple = bonds[..., order["TRIPLE"]] if "TRIPLE" in order else zero
    edge_orders = torch.stack(((1 - neighbor_occupancy) + neighbor_occupancy * none_probability,
                               neighbor_occupancy * (single + aromatic),
                               neighbor_occupancy * double,
                               neighbor_occupancy * triple), dim=-1)
    max_cap = max(max(values) for element in spec.atomic_numbers[1:]
                  for charge in spec.formal_charges
                  if (values := ALLOWED_VALENCES[element][charge]) is not None)
    low, overflow_mass, overflow_moment = capped_bond_order_distribution(edge_orders, max_cap)
    states = torch.arange(max_cap + 1, device=atoms.device, dtype=atoms.dtype)
    excess_by_cap = {
        cap: (low * (states - cap).clamp_min(0)).sum(-1) + overflow_moment - cap * overflow_mass
        for cap in range(max_cap + 1)
    }
    local_excess = torch.zeros_like(occupancy)
    unsupported = torch.zeros_like(occupancy)
    for atom_index, element in enumerate(spec.atomic_numbers[1:], start=1):
        for charge_index, charge in enumerate(spec.formal_charges):
            joint = atoms[..., atom_index] * charges[..., charge_index]
            allowed = ALLOWED_VALENCES[element][charge]
            if allowed is None:
                unsupported = unsupported + joint
            else:
                local_excess = local_excess + joint * excess_by_cap[max(allowed)]
    # Under independent occupied-slot draws these are exact probabilities;
    # no divisions by tiny spectral gaps or soft connectivity determinants.
    absent = atoms[..., 0]
    empty = absent.prod(-1)
    other_absent = absent[:, None, :].expand(batch, n, n).masked_fill(diagonal, 1).prod(-1)
    isolated_given_present = (1 - neighbor_occupancy * real_edge_probability).prod(-1)
    multi_isolated = occupancy * (isolated_given_present - other_absent).clamp_min(0)
    pair_present = occupancy[:, :, None] * occupancy[:, None, :]
    upper = torch.ones(n, n, device=atoms.device, dtype=torch.bool).triu(1)
    pair_count = max(1, n * (n - 1) // 2)
    ghost = (real_edge_probability * (1 - pair_present))[:, upper].sum(-1) / pair_count
    aromatic_affected = occupancy * (1 - (1 - neighbor_occupancy * aromatic).prod(-1))
    # Multiplication by focal occupancy appears exactly once above, in joint.
    # Per-node sums count an undirected bond's effect at both endpoint valences.
    valence_scale = 3
    size_anchor = expected_size * 0
    if reference_size is not None:
        if not isinstance(reference_size, Tensor) or tuple(reference_size.shape) != (batch,):
            raise ValueError("Reference size must be a detached tensor of shape [batch]")
        if reference_size.requires_grad or not bool(torch.isfinite(reference_size).all()) or bool((reference_size < 0).any() | (reference_size > n).any()):
            raise ValueError("Reference size must be finite, detached and within slot capacity")
        reference = reference_size.to(device=atoms.device, dtype=atoms.dtype)
        size_anchor = ((expected_size - reference) / reference.clamp_min(1)).square()
    elif config.size_weight > 0:
        raise ValueError("Positive size_weight requires an explicit detached reference_size")
    per_graph = {
        "valence_excess": local_excess.sum(-1) / denominator / valence_scale,
        "unsupported_charge": unsupported.sum(-1) / denominator,
        "empty_graph": empty,
        "isolated_multi": multi_isolated.sum(-1) / denominator,
        "ghost_bonds": ghost,
        "reference_size": size_anchor,
    }
    weights = {"valence_excess": config.valence_weight, "unsupported_charge": config.charge_support_weight,
               "empty_graph": config.empty_weight, "isolated_multi": config.isolated_weight,
               "ghost_bonds": config.ghost_weight, "reference_size": config.size_weight}
    per_graph["total"] = sum(value * weights[name] for name, value in per_graph.items())
    return {
        **{name: value.mean() for name, value in per_graph.items()},
        "per_graph": per_graph,
        "diagnostics": {
            "expected_size": expected_size,
            "expected_local_bond_order_excess": local_excess.sum(-1),
            "unsupported_element_charge_mass": unsupported.sum(-1),
            "aromatic_affected_atom_mass": aromatic_affected.sum(-1),
            "aromatic_affected_atom_fraction": aromatic_affected.sum(-1) / denominator,
            "conditional_aromatic_pair_probability": aromatic[:, upper].mean(-1) if n > 1 else expected_size * 0,
            "expected_aromatic_edges": (aromatic * pair_present)[:, upper].sum(-1),
            "expected_element_counts": atoms[..., 1:].sum(-2),
            "expected_charge_counts": (charges * occupancy[..., None]).sum(-2),
            "dp_low_mass": low.sum(-1),
            "dp_overflow_mass": overflow_mass,
            "dp_overflow_first_moment": overflow_moment,
        },
        "scope": {**rule_record(spec), "config": config.record(), "temperature": temperature},
    }
