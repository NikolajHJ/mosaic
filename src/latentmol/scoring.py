"""Small contracts for frozen external models evaluating one shared molecule.

Adapters translate inputs; they do not choose a different molecule or fit a
replacement reward predictor. Objective signs, scales and weights belong to the
study that calls these branches. This module contains no model registry, loading,
training, sampling, or chemistry conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from torch import Tensor, nn

from .representation import GraphSpec, HardGraphBatch, RelaxedGraph


@dataclass(frozen=True)
class BranchCapabilities:
    """Declared input and derivative domain, not evidence that validation passed.

    ``none`` permits a discrete external oracle; a future score-function estimator
    differentiates the sampling distribution, never that oracle. ``partial`` and
    ``straight_through`` must not be described as full native pathwise gradients.
    """

    input_kind: Literal["relaxed_graph", "hard_graph"]
    occupancy: Literal["fixed", "variable"]
    gradient_mode: Literal["pathwise", "partial", "straight_through", "none"]
    differentiable_channels: tuple[str, ...]
    supported_atomic_numbers: tuple[int, ...]
    supported_formal_charges: tuple[int, ...]
    supported_bond_names: tuple[str, ...]
    context_policy: str
    relaxation_policy: str

    def __post_init__(self):
        if self.input_kind not in ("relaxed_graph", "hard_graph"):
            raise ValueError("Unknown external branch input kind")
        if self.occupancy not in ("fixed", "variable"):
            raise ValueError("Unknown external branch occupancy policy")
        if self.gradient_mode not in ("pathwise", "partial", "straight_through", "none"):
            raise ValueError("Unknown external branch gradient mode")
        if self.gradient_mode == "none" and self.differentiable_channels:
            raise ValueError("A nondifferentiable oracle cannot declare input-gradient channels")
        if not self.context_policy or not self.relaxation_policy:
            raise ValueError("Context and relaxation policies must be explicit")


@dataclass(frozen=True)
class BranchManifest:
    """Versioned provenance for one model/input bridge/fixed-context combination.

    A validation report is separate evidence, tied to the tested inputs and
    objective. Its hash is not a universal guarantee of useful latent gradients.
    Optional adapter weights refer only to an input-alignment bridge, never a
    locally trained substitute scorer. Hashes describe externally verified assets;
    this record does not itself load or verify model checkpoint bytes.
    """

    name: str
    graph_spec_sha256: str
    model_source_version: str
    model_weights_sha256: str
    adapter_version: str
    adapter_source_sha256: str
    capabilities: BranchCapabilities
    context_sha256: str | None = None
    adapter_weights_sha256: str | None = None
    validation_report_sha256: str | None = None
    contract_version: int = 1
    settings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.contract_version != 1:
            raise ValueError("Unsupported scoring contract version")
        if not self.name or not self.model_source_version or not self.adapter_version:
            raise ValueError("Branch, model-source and adapter versions are required")
        for field in ("graph_spec_sha256", "model_weights_sha256", "adapter_source_sha256",
                      "context_sha256", "adapter_weights_sha256", "validation_report_sha256"):
            value = getattr(self, field)
            if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError(f"{field} must be a lowercase SHA256 digest")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InputSupport:
    """Eligibility of this graph under the branch's declared representation.

    ``exact`` means the supported molecular information is retained within the
    declared context. It does not imply soft-input native parity, a fresh pose,
    validity, or measured binding. Lossy inputs need an explicit caller opt-in;
    unsupported inputs must never be silently repaired or assigned a zero loss.
    """

    status: Literal["exact", "lossy", "unsupported"]
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in ("exact", "lossy", "unsupported"):
            raise ValueError("Unknown input-support status")
        if self.status != "exact" and not self.reasons:
            raise ValueError("Lossy or unsupported inputs require an explanation")


class GraphScoreBranch(Protocol):
    """A relaxed graph enters; native named tensor outputs leave, still attached.

    Implementations currently must also be nn.Module instances, registering all
    scorer/learned-adapter submodules so composition can verify frozen eval state.
    """

    spec: GraphSpec
    manifest: BranchManifest

    def assess(self, graph: RelaxedGraph) -> InputSupport: ...

    def __call__(self, graph: RelaxedGraph) -> Mapping[str, Tensor]: ...


class HardGraphScoreBranch(Protocol):
    """Future discrete-oracle boundary; no score-function algorithm is implemented.

    Callers must explicitly sample/harden outside this branch. This separate
    protocol prevents a relaxed-graph adapter from hiding an argmax, SMILES
    conversion, or a switch to a different molecule in its gradient path.
    """

    spec: GraphSpec
    manifest: BranchManifest

    def assess(self, graph: HardGraphBatch) -> InputSupport: ...

    def __call__(self, graph: HardGraphBatch) -> Mapping[str, Tensor]: ...


def require_frozen_eval(module: nn.Module) -> None:
    """Reject accidental scorer/adapter training without disabling input autograd."""
    if any(part.training for part in module.modules()):
        raise ValueError("External scoring modules must remain in eval mode")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError("External scoring parameters must be frozen")


def check_branches(graph: RelaxedGraph, branches: Mapping[str, GraphScoreBranch],
                   *, spec: GraphSpec, allow_lossy: bool | Mapping[str, bool] = False) -> dict[str, InputSupport]:
    """Preflight every branch before scoring; loss consent may be per alias.

    The legacy boolean remains supported. New multi-model studies always pass
    an explicit name-keyed mapping, so one adapter's loss opt-in cannot authorize
    another adapter's unrelated information loss.
    """
    if not branches:
        raise ValueError("At least one external branch is required")
    if not isinstance(allow_lossy, (bool, Mapping)):
        raise ValueError("allow_lossy must be boolean or a per-branch mapping")
    if isinstance(allow_lossy, Mapping):
        if set(allow_lossy) - set(branches) or any(type(v) is not bool for v in allow_lossy.values()):
            raise ValueError("Invalid per-branch loss consent")
    expected = spec.fingerprint()
    supports = {}
    for name, branch in branches.items():
        if branch.spec.fingerprint() != expected or branch.manifest.graph_spec_sha256 != expected:
            raise ValueError(f"{name}: canonical graph schema differs")
        if branch.manifest.capabilities.input_kind != "relaxed_graph":
            raise ValueError(f"{name}: hard-input oracle requires explicit sampling outside this evaluator")
        if not isinstance(branch, nn.Module):
            raise ValueError(f"{name}: graph branches must be nn.Module instances with registered scorers")
        require_frozen_eval(branch)
        support = branch.assess(graph)
        consent = allow_lossy.get(name, False) if isinstance(allow_lossy, Mapping) else allow_lossy
        if support.status == "unsupported" or (support.status == "lossy" and not consent):
            raise ValueError(f"{name}: {support.status} input: {'; '.join(support.reasons)}")
        supports[name] = support
    return supports


def evaluate_branches(graph: RelaxedGraph, branches: Mapping[str, GraphScoreBranch],
                      *, spec: GraphSpec, allow_lossy: bool | Mapping[str, bool] = False) -> dict[str, dict[str, Tensor]]:
    """Give every branch the identical graph object and preserve named outputs.

    All schema/eligibility checks happen before any scorer runs. There is no
    automatic normalization, weighting, detachment or sampling. The caller may
    form one scalar from the returned tensors and differentiate it once into z.
    Branches must treat input tensors as immutable; adapters allocate their own
    native views. Hard-only oracles require the separate explicit sampling path.
    """
    check_branches(graph, branches, spec=spec, allow_lossy=allow_lossy)
    results = {}
    for name, branch in branches.items():
        outputs = dict(branch(graph))
        if not outputs or any(not isinstance(key, str) or not key or not isinstance(value, Tensor)
                              for key, value in outputs.items()):
            raise ValueError(f"{name}: return nonempty named tensor outputs")
        if any(not value.isfinite().all() for value in outputs.values()):
            raise ValueError(f"{name}: nonfinite external-model output")
        results[name] = outputs
    return results
