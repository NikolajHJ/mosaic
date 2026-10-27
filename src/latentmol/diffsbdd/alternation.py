"""Bounded latent descent and explicit native-state alternating sampling.

Model/context preparation, derivative qualification and feature mapping are
injected through narrow callbacks. The loop refuses an admission without all
mandatory checks; it never invents a scorer or replaces failed native evidence.
The diffusion process and discrete handoffs are detached from latent autograd.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from .native import CleanEstimate, FinalSample, GaussianStream, NativeState, NativeStepper
from .transport import (HandoffRejected, MolecularGraph, PoseCandidate,
                        TransportPolicy)


REQUIRED_ADMISSION_CHECKS = (
    "native_pose_parity", "component_derivatives", "total_derivative",
    "independent_primal_ad", "wrong_gradient_control", "frozen_weights",
    "pocket_frame", "ordinary_roundtrip_identity", "adapter_hard_identity",
)
REQUIRED_CONTROL_ADMISSION_CHECKS = (
    "native_builder_identity", "ordinary_roundtrip_identity", "adapter_hard_identity",
    "pocket_frame", "frozen_weights",
)


@dataclass(frozen=True)
class ObjectiveEvaluation:
    loss: Tensor
    atom_probabilities: Tensor
    bond_probabilities: Tensor
    components: Mapping[str, Tensor] = field(default_factory=dict)

    def check(self) -> None:
        if self.loss.numel() != 1 or not torch.isfinite(self.loss).all():
            raise RuntimeError("Nonfinite/non-scalar fixed-context objective")
        for name, probabilities in (("atoms", self.atom_probabilities),
                                    ("bonds", self.bond_probabilities)):
            if probabilities.ndim < 2 or not torch.isfinite(probabilities).all():
                raise RuntimeError(f"Invalid {name} probabilities")
            if (probabilities < -1e-6).any() or (probabilities > 1 + 1e-6).any():
                raise RuntimeError(f"{name} lies outside the probability simplex")
            if not torch.allclose(probabilities.sum(-1),
                                  torch.ones_like(probabilities[..., 0]),
                                  atol=2e-5, rtol=2e-5):
                raise RuntimeError(f"{name} rows do not sum to one")
        for name, value in self.components.items():
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise RuntimeError(f"Invalid scalar objective component: {name}")


@dataclass(frozen=True)
class DescentResult:
    latent: Tensor
    receipt: dict


def _frozen_versions(modules: Sequence[torch.nn.Module]) -> tuple:
    versions = []
    for module in modules:
        if module.training or any(p.requires_grad or p.grad is not None
                                  for p in module.parameters()):
            raise RuntimeError("Latent optimization requires frozen eval modules")
        versions.append(tuple((name, id(tensor), tensor._version)
                              for name, tensor in list(module.named_parameters()) +
                              list(module.named_buffers())))
    return tuple(versions)


def _maximum_tv(before: Tensor, after: Tensor) -> float:
    if before.shape != after.shape:
        raise RuntimeError("A latent step changed the fixed soft graph schema")
    return float((0.5 * (before.detach() - after.detach()).abs().sum(-1)).max())


def bounded_latent_descent(
    baseline: Tensor,
    objective: Callable[[Tensor], ObjectiveEvaluation],
    max_updates: int,
    *,
    frozen_modules: Sequence[torch.nn.Module] = (),
    describe_proposal: Callable[[Tensor], dict] | None = None,
) -> DescentResult:
    """Sealed v1 descent: step .25, radius 1, TV .02, nominal + 7 halvings.

    Each block starts from ``baseline`` and has no optimizer moments. Component
    derivatives and every attempted latent are saved. Probability/gradient
    failures are hard errors. Finite but unsuccessful proposals are retained
    as rejected trials without updating the latent.
    """
    if max_updates < 0 or max_updates > 5:
        raise ValueError("Each prespecified inner block permits zero to five updates")
    if not torch.isfinite(baseline).all() or baseline.numel() == 0:
        raise ValueError("Invalid baseline latent")
    versions = _frozen_versions(frozen_modules)
    start = baseline.detach().clone()
    current = start.clone()
    receipt = {"policy": "diffsbdd-latent-descent-v1", "max_updates": max_updates,
               "initial_step": 0.25, "trust_radius": 1.0, "max_row_tv": 0.02,
               "maximum_halvings": 7, "forward_calls": 0, "backward_calls": 0,
               "accepted_updates": 0, "updates": []}
    for index in range(max_updates):
        leaf = current.detach().requires_grad_(True)
        with torch.enable_grad():
            evaluation = objective(leaf)
            receipt["forward_calls"] += 1
            evaluation.check()
            if not evaluation.loss.requires_grad:
                raise RuntimeError("Fixed-context objective has no latent gradient path")
            component_gradients = {}
            for name, component in evaluation.components.items():
                if component.requires_grad:
                    derivative = torch.autograd.grad(
                        component, leaf, retain_graph=True, allow_unused=True)[0]
                    receipt["backward_calls"] += 1
                    derivative = torch.zeros_like(leaf) if derivative is None else derivative
                else:
                    derivative = torch.zeros_like(leaf)
                if not torch.isfinite(derivative).all():
                    raise RuntimeError(f"Nonfinite latent component derivative: {name}")
                component_gradients[name] = derivative.detach().cpu().tolist()
            gradient = torch.autograd.grad(evaluation.loss, leaf)[0]
            receipt["backward_calls"] += 1
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Nonfinite full fixed-context latent gradient")
        norm = float(gradient.norm())
        update = {"index": index + 1, "loss_before": float(evaluation.loss.detach()),
                  "components_before": {k: float(v.detach()) for k, v in
                                        evaluation.components.items()},
                  "gradient": gradient.detach().cpu().tolist(),
                  "gradient_norm": norm, "component_gradients": component_gradients,
                  "trials": [], "accepted": False}
        receipt["updates"].append(update)
        if norm == 0.0:
            update["reason"] = "zero_gradient"
            break
        direction = gradient.detach() / max(norm, 1.0)
        for halving in range(8):
            rate = 0.25 / (2 ** halving)
            proposal = current - rate * direction
            delta = proposal - start
            distance = float(delta.norm())
            if distance > 1.0:
                proposal = start + delta / distance
            with torch.no_grad():
                proposed = objective(proposal)
                receipt["forward_calls"] += 1
                if any(not torch.isfinite(value).all() for value in (
                        proposed.loss, proposed.atom_probabilities,
                        proposed.bond_probabilities, *proposed.components.values())):
                    update["trials"].append({
                        "halvings": halving, "nominal_rate": rate,
                        "latent": proposal.detach().cpu().tolist(),
                        "accepted": False, "reason": "nonfinite_proposed_evaluation"})
                    continue
                proposed.check()
            atom_tv = _maximum_tv(evaluation.atom_probabilities, proposed.atom_probabilities)
            bond_tv = _maximum_tv(evaluation.bond_probabilities, proposed.bond_probabilities)
            nonincrease = bool(proposed.loss.detach() <= evaluation.loss.detach())
            trial = {"halvings": halving, "nominal_rate": rate,
                     "actual_displacement": float((proposal - current).norm()),
                     "radius_from_baseline": float((proposal - start).norm()),
                     "max_atom_row_tv": atom_tv, "max_bond_row_tv": bond_tv,
                     "loss": float(proposed.loss.detach()),
                     "components": {k: float(v.detach()) for k, v in
                                    proposed.components.items()},
                     "same_context_nonincrease": nonincrease,
                     "latent": proposal.detach().cpu().tolist(),
                     "accepted": nonincrease and max(atom_tv, bond_tv) <= 0.02}
            if trial["actual_displacement"] == 0.0:
                trial.update(accepted=False, reason="zero_displacement_at_trust_boundary")
            if describe_proposal is not None:
                trial["hard_decode"] = describe_proposal(proposal.detach())
            update["trials"].append(trial)
            if trial["accepted"]:
                current = proposal.detach()
                update["accepted"] = True
                receipt["accepted_updates"] += 1
                break
        if not update["accepted"]:
            update["reason"] = "no_finite_nonincreasing_TV_bounded_trial"
            break
        if _frozen_versions(frozen_modules) != versions:
            raise RuntimeError("A frozen model/buffer changed during latent descent")
    if _frozen_versions(frozen_modules) != versions:
        raise RuntimeError("Frozen model identity changed during latent descent")
    receipt["changed_latent"] = not torch.equal(current, start)
    receipt["terminal_latent"] = current.cpu().tolist()
    return DescentResult(current, receipt)


@dataclass(frozen=True)
class Admission:
    """Evidence-backed detached VAE roundtrip and fixed scorer context."""

    latent: Tensor
    baseline: PoseCandidate
    roundtrip_graph: MolecularGraph
    native_features: Tensor
    checks: Mapping[str, bool]
    context: object = None
    receipt: dict = field(default_factory=dict)

    def require(self, atom_count: int, *, optimization: bool = True) -> None:
        self.baseline.graph.require_pilot(atom_count)
        if self.roundtrip_graph != self.baseline.graph:
            raise HandoffRejected("ordinary_encode_decode_graph_identity")
        required = REQUIRED_ADMISSION_CHECKS if optimization else REQUIRED_CONTROL_ADMISSION_CHECKS
        missing = [key for key in required
                   if self.checks.get(key) is not True]
        if missing:
            raise RuntimeError(f"Mandatory admission checks not passed: {missing}")
        if not torch.isfinite(self.latent).all():
            raise RuntimeError("Admission latent is nonfinite")


@dataclass(frozen=True)
class LatentProposal:
    latent: Tensor
    graph: MolecularGraph
    native_features: Tensor
    mapping_elements: tuple[str, ...]
    receipt: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LoopCallbacks:
    admit: Callable[[NativeState, CleanEstimate], Admission]
    propose: Callable[[Admission, int], LatentProposal]
    native_builder: Callable[[PoseCandidate], tuple[MolecularGraph, dict]]
    rebuilt_objective: Callable[[PoseCandidate], Mapping[str, float]]
    admit_control: Callable[[NativeState, CleanEstimate], Admission] | None = None


@dataclass(frozen=True)
class AlternationResult:
    final: FinalSample
    opportunities: tuple[dict, ...]
    gaussian_receipts: tuple[dict, ...]


@dataclass(frozen=True)
class SequentialResult:
    final: FinalSample
    blocks: tuple[dict, ...]


class AlternatingSampler:
    """One native transition then a bounded optional detached guidance block."""

    def __init__(self, stepper: NativeStepper, callbacks: LoopCallbacks,
                 native_atom_decoder: Sequence[str], transport: TransportPolicy,
                 qualification: Mapping[str, bool]):
        needed = ("native_replay", "codec", "exact_boundary", "native_pose_context",
                  "derivatives", "frozen_models", "T1_transport")
        if transport.stage == "T2":
            needed += ("T2_transport",)
        failed = [key for key in needed if qualification.get(key) is not True]
        if failed:
            raise RuntimeError(f"Alternation prerequisites have not passed: {failed}")
        if len(native_atom_decoder) != stepper.core.atom_nf:
            raise ValueError("Native vocabulary differs from loaded checkpoint")
        self.stepper = stepper
        self.callbacks = callbacks
        self.atom_decoder = tuple(native_atom_decoder)
        self.transport = transport

    def _check_features(self, graph: MolecularGraph, features: Tensor):
        if features.shape != (graph.size, len(self.atom_decoder)):
            raise RuntimeError("Mapping feature shape differs from native atom order")
        if not torch.isfinite(features).all() or (features < -1e-6).any():
            raise RuntimeError("Mapping returned invalid clean atom probabilities")
        if not torch.allclose(features.sum(-1), torch.ones_like(features[:, 0]),
                              atol=2e-5, rtol=2e-5):
            raise RuntimeError("Mapping clean feature rows do not sum to one")
        decoded = tuple(self.atom_decoder[i] for i in features.argmax(-1).tolist())
        if decoded != graph.elements:
            raise HandoffRejected("ordinary_adapter_hard_elements_disagree")

    def guidance(self, state: NativeState, max_updates: int) -> tuple[NativeState, dict]:
        """Rejected or K=0 blocks return the original native object exactly."""
        if state.batch_size != 1:
            raise ValueError("Version-one handoffs process one seed/pocket at a time")
        if not 0 <= max_updates <= 5:
            raise ValueError("Guidance block budget must be between zero and five")
        record = {"level": state.level, "time": state.level / self.stepper.timesteps,
                  "max_updates": max_updates, "applied": False}
        # Caller callbacks receive detached copies so a rejected attempt cannot
        # accidentally mutate the native diffusion tensors or its pocket frame.
        detached = state.clone()
        clean = self.stepper.clean_estimate(detached)
        record["raw_clean_positions"] = clean.positions.cpu().tolist()
        record["raw_clean_features"] = clean.features.cpu().tolist()
        try:
            if max_updates == 0:
                if self.callbacks.admit_control is None:
                    raise RuntimeError("H0 requires a structural-only control admission callback")
                admission = self.callbacks.admit_control(detached, clean)
            else:
                admission = self.callbacks.admit(detached, clean)
            admission.require(self.transport.atom_count, optimization=max_updates > 0)
            self._check_features(admission.baseline.graph, admission.native_features)
            record["admission"] = admission.receipt
            record["baseline_graph"] = asdict(admission.baseline.graph)
            record["baseline_positions"] = admission.baseline.positions.cpu().tolist()
            if max_updates == 0:
                # Eligibility and exact no-op transport are still audited. No
                # proposal/scorer-gradient call, centering or RNG draw is made.
                lifted = self.transport.lift(admission.baseline,
                                             admission.baseline.graph,
                                             self.callbacks.native_builder)
                record.update(reason="H0_noop", transport=lifted.receipt,
                              effective_transfer=False)
                return state, record
            proposal = self.callbacks.propose(admission, max_updates)
            record["proposal"] = proposal.receipt
            record["proposed_graph"] = asdict(proposal.graph)
            record["proposed_latent"] = proposal.latent.detach().cpu().tolist()
            if proposal.mapping_elements != proposal.graph.elements:
                raise HandoffRejected("ordinary_learned_hard_elements_disagree")
            self._check_features(proposal.graph, proposal.native_features)
            if torch.equal(proposal.latent.detach(), admission.latent.detach()):
                if proposal.graph != admission.baseline.graph:
                    raise RuntimeError("Identical latent decoded different ordinary hard graphs")
                if not torch.equal(proposal.native_features, admission.native_features):
                    raise RuntimeError("Identical latent changed deterministic mapping features")
                record.update(reason="unchanged_latent", effective_transfer=False)
                return state, record
            lifted = self.transport.lift(admission.baseline, proposal.graph,
                                         self.callbacks.native_builder)
            record["transport"] = lifted.receipt
            record["hard_graph_changed"] = proposal.graph != admission.baseline.graph
            record["hard_elements_changed"] = proposal.graph.elements != admission.baseline.graph.elements
            record["hard_bonds_changed"] = proposal.graph.bonds != admission.baseline.graph.bonds
            record["pose_changed"] = not torch.equal(lifted.candidate.positions,
                                                     admission.baseline.positions)
            baseline_input = self.stepper.clean_input(
                admission.baseline.positions, admission.native_features, state)
            proposed_input = self.stepper.clean_input(
                lifted.candidate.positions, proposal.native_features, state)
            effective = not torch.equal(baseline_input, proposed_input)
            record["effective_transfer"] = effective
            record["clean_state_delta_norm"] = float((proposed_input - baseline_input).norm())
            if not effective:
                record["reason"] = "null_transport"
                return state, record
            # Independent ordinary native evaluation uses hard chemistry and the
            # supplied/lifted pose. It is distinct from the fixed-context proxy.
            before = dict(self.callbacks.rebuilt_objective(admission.baseline))
            after = dict(self.callbacks.rebuilt_objective(lifted.candidate))
            if "total" not in before or "total" not in after:
                raise RuntimeError("Rebuilt native scorer did not return total objective")
            if not all(math.isfinite(float(v)) for v in list(before.values()) + list(after.values())):
                raise RuntimeError("Nonfinite rebuilt ordinary-native score")
            record["rebuilt_baseline"] = before
            record["rebuilt_proposed"] = after
            if float(after["total"]) > float(before["total"]):
                raise HandoffRejected("rebuilt_native_objective_worsened")
            corrected = self.stepper.correct(state, baseline_input, proposed_input)
            t = torch.full((state.batch_size, 1), state.level,
                           device=state.ligand.device) / self.stepper.timesteps
            alpha = self.stepper.core.alpha(self.stepper.core.gamma(t), state.ligand)
            record["residual_norm_before"] = float((
                state.ligand - alpha[state.ligand_mask] * baseline_input).norm())
            record["native_state_displacement"] = float((corrected.ligand - state.ligand).norm())
            record.update(applied=True, reason="accepted")
            return corrected, record
        except HandoffRejected as exc:
            record.update(reason=str(exc), rejected=True, effective_transfer=False)
            if exc.receipt:
                record["rejection_receipt"] = exc.receipt
            return state, record

    def run(self, pocket: dict, num_nodes_lig: Tensor, stream: GaussianStream,
            opportunity_levels: Sequence[int], max_updates: int,
            on_opportunity: Callable[[dict], None] | None = None,
            before_transition: Callable[[NativeState], None] | None = None) -> AlternationResult:
        levels = tuple(int(level) for level in opportunity_levels)
        if len(levels) != len(set(levels)):
            raise ValueError("Duplicate opportunity level")
        if any(level < 1 or level >= self.stepper.timesteps for level in levels):
            raise ValueError("Interleaved opportunities must arrive at 1..T-1")
        state = self.stepper.initialize(pocket, num_nodes_lig, stream)
        receipts = []
        while state.level:
            if before_transition is not None:
                before_transition(state)
            state = self.stepper.reverse_step(state, stream)
            if state.level in levels:
                count_before = len(stream.receipts)
                state, receipt = self.guidance(state, max_updates)
                if len(stream.receipts) != count_before:
                    raise RuntimeError("Guidance consumed a native Gaussian draw")
                receipts.append(receipt)
                if on_opportunity is not None:
                    on_opportunity(receipt)
        final = self.stepper.finalize(state, stream)
        return AlternationResult(final, tuple(receipts), tuple(stream.receipts))

    def run_sequential(self, final: FinalSample,
                       admit_terminal: Callable[[FinalSample], Admission], *,
                       blocks: int = 20, max_updates: int = 5,
                       on_block: Callable[[dict], None] | None = None,
                       before_block: Callable[[int], None] | None = None) -> SequentialResult:
        """The S arm starts AFTER the actual native terminal p(x,h|z0) draw.

        It cannot call the diffusion kernel, reset a timestep or consume sampling
        noise. Each block rebuilds the terminal graph/context and applies exactly
        the same detached transport and rebuilt native nonincrease contract.
        Invalid terminal samples remain in all twenty attempted-block receipts.
        """
        if blocks != 20 or max_updates != 5:
            raise ValueError("The sealed S arm requires 20 blocks of up to 5 updates")
        current, records = final, []
        for index in range(blocks):
            if before_block is not None:
                before_block(index)
            record = {"block": index + 1, "arm": "S", "max_updates": max_updates,
                      "after_native_terminal_draw": True, "applied": False}
            records.append(record)
            try:
                # Copy callback-visible tensors; a rejected block restores the
                # exact previous output, including unsupported native atoms.
                visible = FinalSample(*(value.detach().clone() for value in (
                    current.positions, current.one_hot, current.pocket_positions,
                    current.pocket_features, current.ligand_mask, current.pocket_mask,
                    current.internal_ligand, current.internal_pocket)))
                admission = admit_terminal(visible)
                admission.require(self.transport.atom_count)
                self._check_features(admission.baseline.graph, admission.native_features)
                if not torch.equal(admission.baseline.positions, current.positions):
                    raise RuntimeError("S admission replaced the actual terminal pose")
                record["admission"] = admission.receipt
                record["baseline_graph"] = asdict(admission.baseline.graph)
                proposal = self.callbacks.propose(admission, max_updates)
                record["proposal"] = proposal.receipt
                record["proposed_graph"] = asdict(proposal.graph)
                if proposal.mapping_elements != proposal.graph.elements:
                    raise HandoffRejected("ordinary_learned_hard_elements_disagree")
                self._check_features(proposal.graph, proposal.native_features)
                if torch.equal(proposal.latent.detach(), admission.latent.detach()):
                    if proposal.graph != admission.baseline.graph:
                        raise RuntimeError("Identical latent decoded different graphs in S")
                    if not torch.equal(proposal.native_features, admission.native_features):
                        raise RuntimeError("Identical latent changed deterministic S mapping")
                    record["reason"] = "unchanged_latent"
                    continue
                lifted = self.transport.lift(admission.baseline, proposal.graph,
                                             self.callbacks.native_builder)
                record["transport"] = lifted.receipt
                record["hard_graph_changed"] = proposal.graph != admission.baseline.graph
                record["hard_elements_changed"] = proposal.graph.elements != admission.baseline.graph.elements
                record["hard_bonds_changed"] = proposal.graph.bonds != admission.baseline.graph.bonds
                record["pose_changed"] = not torch.equal(lifted.candidate.positions,
                                                         admission.baseline.positions)
                # S ends at a hard molecule, unlike an intermediate soft state.
                hard_features = torch.nn.functional.one_hot(
                    proposal.native_features.argmax(-1), len(self.atom_decoder)).to(current.one_hot)
                effective = not (torch.equal(lifted.candidate.positions, current.positions)
                                 and torch.equal(hard_features, current.one_hot)
                                 and proposal.graph == admission.baseline.graph)
                record["effective_hard_transfer"] = effective
                if not effective:
                    record["reason"] = "null_terminal_transport"
                    continue
                before = dict(self.callbacks.rebuilt_objective(admission.baseline))
                after = dict(self.callbacks.rebuilt_objective(lifted.candidate))
                if "total" not in before or "total" not in after or not all(
                        math.isfinite(float(value)) for value in
                        list(before.values()) + list(after.values())):
                    raise RuntimeError("Invalid ordinary-native S endpoint score")
                record["rebuilt_baseline"], record["rebuilt_proposed"] = before, after
                if float(after["total"]) > float(before["total"]):
                    raise HandoffRejected("rebuilt_native_objective_worsened")
                offset_raw = current.positions - current.internal_ligand[:, :3]
                internal_xyz = lifted.candidate.positions - offset_raw
                current = FinalSample(
                    lifted.candidate.positions, hard_features,
                    current.pocket_positions, current.pocket_features,
                    current.ligand_mask, current.pocket_mask,
                    torch.cat((internal_xyz, hard_features), dim=1), current.internal_pocket)
                record.update(applied=True, reason="accepted")
            except HandoffRejected as exc:
                record.update(reason=str(exc), rejected=True,
                              effective_hard_transfer=False)
                if exc.receipt:
                    record["rejection_receipt"] = exc.receipt
            finally:
                if on_block is not None:
                    on_block(record)
        return SequentialResult(current, tuple(records))


def require_h0_exact(baseline: AlternationResult, handoff: AlternationResult) -> dict:
    """The H0 arm is an equality control, never an approximate scientific gate."""
    checks = {
        "ligand": torch.equal(baseline.final.internal_ligand, handoff.final.internal_ligand),
        "pocket": torch.equal(baseline.final.internal_pocket, handoff.final.internal_pocket),
        "ligand_mask": torch.equal(baseline.final.ligand_mask, handoff.final.ligand_mask),
        "pocket_mask": torch.equal(baseline.final.pocket_mask, handoff.final.pocket_mask),
        "global_positions": torch.equal(baseline.final.positions, handoff.final.positions),
        "gaussian_stream": baseline.gaussian_receipts == handoff.gaussian_receipts,
        "no_applied_handoffs": not any(r["applied"] for r in handoff.opportunities),
    }
    if not all(checks.values()):
        raise RuntimeError(f"H0 changed native states/RNG: {checks}")
    return {"passed": True, "checks": checks}
