"""Monte Carlo connectivity risk with a declared first-order score gradient.

Every hard graph is drawn from the existing decoder law, retained without
validity filtering, and counted outside autograd. The cost is the unrestricted
number of edges required to join its components, normalized by max(K-1,1).
This is a graph-theoretic lower bound, not a chemically feasible repair count.
Empty/singleton costs are zero; the caller's exact empty/size guards cover them.

Connectivity depends only on node and edge presence. Its score function uses
Bernoulli occupancy and occupied-pair bond-presence probabilities, marginalizing
element identity, formal charge and specific non-NONE bond category. This
Rao--Blackwellization removes gradients from irrelevant sampled labels. It does
not backpropagate through component counting or categorical sampling.

Independent draws and a leave-one-out baseline produce an unbiased first-order
gradient of expected connectivity cost. A small draw budget has substantial
variance. The displayed value is sampled risk; its custom surrogate derivative
is not a pathwise derivative, finite differences of sampled realizations do not
validate it, and its autograd higher derivatives have no claimed interpretation.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor

from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob_terms, sample_hard_graph


@dataclass(frozen=True)
class ConnectivityRegularizationConfig:
    version: str = "components_mc_v1"
    draws: int = 4
    weight: float = 1.0

    def __post_init__(self):
        if self.version != "components_mc_v1":
            raise ValueError("Unsupported connectivity regularization version")
        if type(self.draws) is not int or self.draws < 2:
            raise ValueError("Leave-one-out connectivity requires at least two draws")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)) or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Connectivity weight must be a finite nonnegative number")

    @classmethod
    def from_mapping(cls, raw: Mapping) -> "ConnectivityRegularizationConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("Connectivity config must be a mapping")
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown connectivity regularization fields: {sorted(unknown)}")
        return cls(**dict(raw))

    def record(self) -> dict:
        return asdict(self)


@torch.no_grad()
def component_costs(graph: HardGraphBatch) -> dict[str, Tensor]:
    """Count components exactly by min-label propagation on the graph device.

    The loop runs N iterations, enough for any N-node graph, without copying to
    RDKit, reindexing slots, or using floating-point eigenvalue thresholds.
    All chemically invalid but structurally supported graphs are counted too.
    """
    if graph.atoms.ndim != 2 or graph.atoms.shape[0] < 1 or graph.atoms.shape[1] < 1:
        raise ValueError("Expected a nonempty [batch,slot] hard atom tensor")
    batch, n = graph.atoms.shape
    if graph.atoms.dtype != torch.long or graph.bonds.dtype != torch.long or tuple(graph.bonds.shape) != (batch, n, n):
        raise ValueError("Component counting requires long labels and matching bond shape")
    if graph.bonds.device != graph.atoms.device:
        raise ValueError("Hard atom/bond labels must share their device")
    if bool((graph.atoms < 0).any() | (graph.bonds < 0).any()):
        raise ValueError("Hard graph labels cannot be negative")
    active = graph.atoms != 0
    edge = graph.bonds != 0
    pair = active[:, :, None] & active[:, None, :]
    if not bool((edge == edge.transpose(1, 2)).all()) or bool(edge.diagonal(dim1=1, dim2=2).any()) or bool((edge & ~pair).any()):
        raise ValueError("Hard graph requires symmetric edges, no self-bonds and no padding edges")
    indices = torch.arange(n, device=graph.atoms.device)[None].expand(batch, n)
    labels = torch.where(active, indices, n)
    diagonal = torch.eye(n, device=graph.atoms.device, dtype=torch.bool)[None]
    adjacency = edge | (diagonal & pair)
    for _ in range(n):
        neighbors = labels[:, None, :].expand(batch, n, n)
        labels = torch.where(adjacency, neighbors, n).amin(-1)
    components = ((labels == indices) & active).sum(-1)
    count = active.sum(-1)
    additions = (components - 1).clamp_min(0)
    normalized = additions.to(torch.float64) / (count - 1).clamp_min(1)
    return {"active_count": count, "components": components,
            "minimum_unrestricted_added_edges": additions, "normalized_cost": normalized}


def presence_log_probability(output: DecoderOutput, graph: HardGraphBatch, spec: GraphSpec, *,
                             temperature: float = 1.0, fixed_active: Tensor | None = None) -> Tensor:
    """Log probability of node/edge presence, marginalizing other categories.

    Existing public sampling validation establishes structural support. Only
    upper-triangle occupied pairs contribute to the marginal presence law.
    Fixed occupancy has probability one, so it produces no occupancy gradient.
    The decoder's symmetric conditional logits are used before pair masking.
    Stable logsumexp/logsoftmax retain tiny probabilities without epsilon
    clipping or a changed draw law.
    """
    checked = hard_graph_log_prob_terms(output, graph, spec, temperature=temperature, fixed_active=fixed_active)
    if not bool(checked["supported"].all()):
        raise ValueError("Hard draw lies outside the declared graph sampling support")
    atom_log = (output.atoms / temperature).log_softmax(-1)
    bond_log = (output.symmetric_bond_logits / temperature).log_softmax(-1)
    active = graph.atoms != 0
    if fixed_active is None:
        occupied_log = torch.logsumexp(atom_log[..., 1:], -1)
        atom_term = torch.where(active, occupied_log, atom_log[..., 0]).sum(-1)
    else:
        atom_term = output.atoms.sum(dim=(-1, -2)) * 0
    left, right = torch.triu_indices(spec.max_nodes, spec.max_nodes, offset=1, device=output.atoms.device)
    present_pairs = active[:, left] & active[:, right]
    edge_present = graph.bonds[:, left, right] != 0
    selected_bond_log = bond_log[:, left, right]
    occupied_edge_log = torch.logsumexp(selected_bond_log[..., 1:], -1)
    edge_term = torch.where(edge_present, occupied_edge_log, selected_bond_log[..., 0])
    edge_term = torch.where(present_pairs, edge_term, torch.zeros_like(edge_term)).sum(-1)
    # Explicit zero attachment makes the marginalized charge gradient zero,
    # rather than relying on a caller interpreting an unused gradient as zero.
    charge_zero = output.charges.sum(dim=(-1, -2)) * 0
    return atom_term + edge_term + charge_zero


def leave_one_out_connectivity_surrogate(costs: Tensor, log_probabilities: Tensor) -> dict[str, Tensor]:
    """Expose the estimator algebra for exact-enumeration HPC fixtures.

    Both inputs have [draw,batch] axes. A draw's baseline contains only other
    draws at the same fixed decoder/context, so it is independent of that draw.
    Costs and baselines are stopped; only the marginal log probability carries
    gradients. Callers must actually use independent draws, not copied labels.
    """
    if costs.ndim != 2 or tuple(costs.shape) != tuple(log_probabilities.shape) or costs.shape[0] < 2:
        raise ValueError("Expected matching [draw,batch] tensors with at least two draws")
    if costs.device != log_probabilities.device or not costs.is_floating_point() or not log_probabilities.is_floating_point():
        raise ValueError("Costs and log probabilities must be floating tensors on the same device")
    if not bool(torch.isfinite(costs).all() & torch.isfinite(log_probabilities).all()):
        raise ValueError("Connectivity costs and marginal log probabilities must be finite")
    stopped = costs.detach().to(log_probabilities.dtype)
    baseline = (stopped.sum(0, keepdim=True) - stopped) / (costs.shape[0] - 1)
    advantage = stopped - baseline
    surrogate = (advantage * log_probabilities).mean(0)
    risk = stopped.mean(0)
    displayed = risk + (surrogate - surrogate.detach())
    return {"risk": risk, "surrogate": displayed, "baselines": baseline, "advantages": advantage}


def connectivity_regularization_losses(output: DecoderOutput, spec: GraphSpec,
                                      config: ConnectivityRegularizationConfig | None = None, *,
                                      generator: torch.Generator, temperature: float = 1.0,
                                      fixed_active: Tensor | None = None) -> dict:
    """Sample the fixed quota and return displayed risk plus score gradients.

    The supplied generator must match the decoder device. No outcome changes
    the number of attempts, no graph is repaired/rejected, and generator state
    is owned by the caller for replayable training checkpoints. Set an outer
    comparison-arm coefficient to zero while retaining config.weight=1 if the
    control must consume identical draws and diagnostics. Config weight zero
    explicitly disables all sampling; that is a different budget policy.
    """
    config = ConnectivityRegularizationConfig() if config is None else config
    if not isinstance(config, ConnectivityRegularizationConfig):
        raise ValueError("Expected a ConnectivityRegularizationConfig")
    if not isinstance(generator, torch.Generator):
        raise ValueError("Connectivity sampling requires an explicit torch.Generator")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite positive number")
    if config.weight == 0:
        zero = output.atoms.sum(dim=(-1, -2)) * 0 + output.bonds.sum(dim=(-1, -2, -3)) * 0 + output.charges.sum(dim=(-1, -2)) * 0
        return {"total": zero.mean(), "risk": zero.mean(), "per_graph": {"risk": zero, "total": zero},
                "diagnostics": {"attempted_draws": 0},
                "scope": {"config": config.record(), "status": "disabled_weight_zero", "first_order_score_function_only": True}}
    costs, log_probabilities, draws, details = [], [], [], []
    for _ in range(config.draws):
        graph = sample_hard_graph(output, spec, generator=generator, temperature=temperature, fixed_active=fixed_active)
        detail = component_costs(graph)
        costs.append(detail["normalized_cost"])
        log_probabilities.append(presence_log_probability(output, graph, spec, temperature=temperature, fixed_active=fixed_active))
        draws.append(graph)
        details.append(detail)
    estimated = leave_one_out_connectivity_surrogate(torch.stack(costs), torch.stack(log_probabilities))
    total = config.weight * estimated["surrogate"]
    return {"total": total.mean(), "risk": estimated["risk"].mean(),
            "per_graph": {"risk": estimated["risk"], "total": total},
            "diagnostics": {
                "attempted_draws": config.draws,
                "costs": torch.stack(costs), "baselines": estimated["baselines"],
                "advantages": estimated["advantages"],
                "active_counts": torch.stack([detail["active_count"] for detail in details]),
                "components": torch.stack([detail["components"] for detail in details]),
                "minimum_unrestricted_added_edges": torch.stack([detail["minimum_unrestricted_added_edges"] for detail in details]),
                "sampled_atoms": torch.stack([graph.atoms for graph in draws]),
                "sampled_bonds": torch.stack([graph.bonds for graph in draws]),
            },
            "scope": {
                "config": config.record(), "temperature": temperature,
                "occupancy": "fixed" if fixed_active is not None else "sampled_NO_ATOM_categorical",
                "sampling_law": "existing sample_hard_graph conditional unordered occupied-pair law; no filtering",
                "cost": "max(components-1,0)/max(occupied_atoms-1,1)",
                "gradient": "First-order Rao-Blackwellized score function of occupancy and bond presence",
                "baseline": "Independent leave-one-out other draws at fixed decoder/context",
                "marginalized_categories": ["real atom identity", "formal charge", "specific non-NONE bond type"],
                "higher_derivatives_supported": False, "pathwise_derivative": False,
                "chemical_repair_distance": False, "validity_guarantee": False,
                "empty_and_singleton_cost": 0,
                "empty_guard_requirement": "Caller must retain exact empty/size or occupancy-reference guards",
            }}
