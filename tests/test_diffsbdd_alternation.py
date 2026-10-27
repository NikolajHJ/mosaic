"""Allocated loop/transport engineering tests; real-model gates are separate."""
from dataclasses import replace

import pytest
import torch

from latentmol.diffsbdd.alternation import (
    Admission, AlternatingSampler, LatentProposal, LoopCallbacks,
    ObjectiveEvaluation, REQUIRED_ADMISSION_CHECKS, REQUIRED_CONTROL_ADMISSION_CHECKS,
    bounded_latent_descent,
    require_h0_exact,
)
from latentmol.diffsbdd.native import GaussianStream, NativeStepper
from latentmol.diffsbdd.transport import (
    HandoffRejected, MolecularGraph, NativeBuilder, PoseCandidate, TransportPolicy,
)
from test_diffsbdd_native import EngineeringCore


def graph(elements=("C", "C", "C"), bonds=((0, 1, 0), (1, 0, 1), (0, 1, 0))):
    return MolecularGraph(elements, (0, 0, 0), bonds, (0, 1, 2))


def soft_evaluation(z, *, tv_sensitive=False):
    loss = (z - 1).square().sum()
    if tv_sensitive:
        p = 0.5 + z.reshape(-1)[0] * 0.25
        atoms = torch.stack((p, 1 - p))[None, None]
    else:
        atoms = z.new_tensor([[[1., 0.]]])
    bonds = z.new_tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]])
    return ObjectiveEvaluation(loss, atoms, bonds, {"quadratic": loss})


def test_bounded_descent_saves_proposals_derivatives_and_tv_backtracking():
    start = torch.tensor([0.])
    result = bounded_latent_descent(start, lambda z: soft_evaluation(z, tv_sensitive=True), 1)
    update = result.receipt["updates"][0]
    assert result.receipt["accepted_updates"] == 1
    assert [trial["accepted"] for trial in update["trials"]] == [False, False, True]
    assert update["trials"][-1]["halvings"] == 2
    assert update["trials"][-1]["max_atom_row_tv"] <= 0.02
    assert update["component_gradients"]["quadratic"] == [-2.0]
    assert result.receipt["backward_calls"] == 2
    assert result.receipt["forward_calls"] == 4
    assert torch.equal(start, torch.tensor([0.]))


def test_descent_trust_region_and_nonincrease():
    start = torch.tensor([-5.])
    result = bounded_latent_descent(start, soft_evaluation, 5)
    assert (result.latent - start).norm() <= 1.0
    for update in result.receipt["updates"]:
        for trial in update["trials"]:
            if trial["accepted"]:
                assert trial["same_context_nonincrease"]
                assert trial["radius_from_baseline"] <= 1.000001


def test_zero_update_block_does_not_call_objective():
    def must_not_run(z):
        raise AssertionError("objective called for K=0")

    result = bounded_latent_descent(torch.tensor([1.]), must_not_run, 0)
    assert result.receipt["forward_calls"] == result.receipt["backward_calls"] == 0


def test_descent_rejects_nonfinite_trials_and_tries_halved_step():
    def objective(z):
        result = soft_evaluation(z)
        if not z.requires_grad and float(z) > 0.1:
            return replace(result, loss=z.new_tensor(float("nan")))
        return result

    result = bounded_latent_descent(torch.tensor([0.]), objective, 1)
    trials = result.receipt["updates"][0]["trials"]
    assert [t.get("reason") for t in trials[:2]] == [
        "nonfinite_proposed_evaluation", "nonfinite_proposed_evaluation"]
    assert trials[2]["accepted"]


def test_descent_does_not_accept_detached_or_wrong_simplex_objective():
    def detached(z):
        return replace(soft_evaluation(z), loss=z.detach().square().sum())

    with pytest.raises(RuntimeError, match="no latent gradient"):
        bounded_latent_descent(torch.tensor([0.]), detached, 1)
    with pytest.raises(ValueError, match="five"):
        bounded_latent_descent(torch.tensor([0.]), soft_evaluation, 6)


def test_frozen_model_mutation_is_a_hard_error():
    model = torch.nn.Linear(1, 1).eval().requires_grad_(False)

    def objective(z):
        with torch.no_grad():
            model.weight.add_(0.1)
        return soft_evaluation(z)

    with pytest.raises(RuntimeError, match="frozen|Frozen"):
        bounded_latent_descent(torch.tensor([0.]), objective, 1, frozen_modules=(model,))


def test_T1_rejects_bond_only_change_and_preserves_exact_noop_pose():
    base = PoseCandidate(graph(), torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [3., 0., 0.]]))
    policy = TransportPolicy("T1", atom_count=3)

    def must_not_run(_):
        raise AssertionError("unchanged graph called builder")

    no_op = policy.lift(base, base.graph, must_not_run)
    assert no_op.candidate is base
    assert no_op.candidate.positions is base.positions
    changed_bonds = graph(bonds=((0, 2, 0), (2, 0, 1), (0, 1, 0)))
    with pytest.raises(HandoffRejected, match="T1_bond_change"):
        policy.lift(base, changed_bonds, must_not_run)


def test_T1_requires_native_complete_graph_recovery_and_atom_map():
    base = PoseCandidate(graph(), torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [3., 0., 0.]]))
    changed = graph(("C", "N", "C"))
    policy = TransportPolicy("T1", atom_count=3)
    with pytest.raises(HandoffRejected, match="not_recovered"):
        policy.lift(base, changed, lambda c: (base.graph, {}))
    lifted = policy.lift(base, changed, lambda c: (c.graph, {"tested": True}))
    assert lifted.candidate.positions is base.positions
    assert lifted.candidate.graph == changed
    with pytest.raises(HandoffRejected, match="atom_map"):
        policy.lift(base, replace(changed, atom_ids=(1, 0, 2)), lambda c: (c.graph, {}))


def test_graph_admission_keeps_unsupported_charged_disconnected_in_denominator():
    for candidate, reason in (
        (graph(("C", "B", "C")), "unsupported_element"),
        (replace(graph(), charges=(0, 1, -1)), "charged_graph"),
        (graph(bonds=((0, 1, 0), (1, 0, 0), (0, 0, 0))), "disconnected_graph"),
    ):
        with pytest.raises(HandoffRejected, match=reason):
            candidate.require_pilot(3)


def test_T2_unchanged_graph_bypasses_forcefield(monkeypatch):
    policy = TransportPolicy("T2", atom_count=3)
    base = PoseCandidate(graph(), torch.zeros(3, 3))

    def forbidden(*args):
        raise AssertionError("T2 no-op invoked UFF")

    monkeypatch.setattr(TransportPolicy, "_restrained_uff", forbidden)
    result = policy.lift(base, base.graph, forbidden)
    assert result.candidate is base
    assert result.receipt["uff_bypassed"]


def test_T2_bond_handoff_calls_lift_then_native_builder(monkeypatch):
    policy = TransportPolicy("T2", atom_count=3)
    base = PoseCandidate(graph(), torch.zeros(3, 3))
    changed = graph(bonds=((0, 2, 0), (2, 0, 1), (0, 1, 0)))
    calls = []

    def lift(self, proposed, positions):
        calls.append("uff")
        return positions + 0.1, {"max_displacement": 0.1733}

    def builder(candidate):
        calls.append("builder")
        assert torch.equal(candidate.positions, base.positions + 0.1)
        return candidate.graph, {"complete_ordered_graph_matches": True}

    monkeypatch.setattr(TransportPolicy, "_restrained_uff", lift)
    result = policy.lift(base, changed, builder)
    assert calls == ["uff", "builder"]
    assert result.candidate.graph == changed


def test_native_builder_audits_its_documented_three_decimal_quantization():
    base = PoseCandidate(graph(), torch.tensor([
        [0.00049, 0.00017, 0.], [1.50049, 0.00017, 0.], [3.00049, 0.00017, 0.]]))

    def engineering_builder(xyz, types, dataset_info, **kwargs):
        quantized = xyz.new_tensor([[float(f"{value:.3f}") for value in row]
                                    for row in xyz.tolist()])
        return base.graph.to_rdkit(quantized)

    builder = NativeBuilder(engineering_builder, {"atom_decoder": ["C", "N"]})
    built, receipt = builder(base)
    assert built == base.graph
    assert receipt["maximum_native_coordinate_quantization"] > 2e-5
    assert receipt["policy"]["native_xyz_decimal_places"] == 3
    assert torch.equal(base.positions, torch.tensor([
        [0.00049, 0.00017, 0.], [1.50049, 0.00017, 0.], [3.00049, 0.00017, 0.]]))


def test_native_builder_rejects_atom_coordinate_permutation_after_serialization():
    base = PoseCandidate(graph(), torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [3., 0., 0.]]))

    def permuting_builder(xyz, types, dataset_info, **kwargs):
        return base.graph.to_rdkit(xyz.flip(0))

    builder = NativeBuilder(permuting_builder, {"atom_decoder": ["C", "N"]})
    with pytest.raises(HandoffRejected, match="atom_order_or_pose_change"):
        builder(base)


def _pocket():
    return {"x": torch.tensor([[10., 2., -1.], [11., 3., 0.], [12., 2., 1.]]),
            "one_hot": torch.tensor([[1., 0.], [0., 1.], [1., 0.]]),
            "mask": torch.zeros(3, dtype=torch.long), "size": torch.tensor([3])}


def _sampler(*, reject=False, worsening=False, missing_check=False):
    stepper = NativeStepper(EngineeringCore())
    state = {"proposal_calls": 0, "score_calls": 0}

    def admit(native_state, clean):
        if reject:
            # Attempted callback mutation must not leak into the real sampler.
            native_state.ligand.fill_(999)
            raise HandoffRejected("engineering_ineligible")
        checks = dict.fromkeys(REQUIRED_ADMISSION_CHECKS, True)
        checks["native_builder_identity"] = True
        if missing_check:
            checks.pop("total_derivative")
        return Admission(torch.tensor([0.]), PoseCandidate(graph(), clean.positions),
                         graph(), torch.tensor([[1., 0.]] * 3), checks)

    def propose(admission, k):
        state["proposal_calls"] += 1
        changed = graph(("C", "N", "C"))
        return LatentProposal(admission.latent + 0.1, changed,
                              torch.tensor([[1., 0.], [0., 1.], [1., 0.]]),
                              changed.elements, {"max_updates": k})

    def score(candidate):
        state["score_calls"] += 1
        return {"total": float(worsening) if "N" in candidate.graph.elements else 0.0}

    callbacks = LoopCallbacks(admit, propose, lambda c: (c.graph, {}), score, admit)
    qualification = dict.fromkeys(("native_replay", "codec", "exact_boundary",
                                  "native_pose_context", "derivatives", "frozen_models",
                                  "T1_transport"), True)
    return AlternatingSampler(stepper, callbacks, ("C", "N"),
                              TransportPolicy("T1", atom_count=3), qualification), state


@pytest.mark.parametrize("reject", [False, True])
def test_H0_noop_or_rejected_handoff_matches_D_tensors_and_draws_exactly(reject):
    sampler, counters = _sampler(reject=reject)
    baseline = sampler.run(_pocket(), torch.tensor([3]), GaussianStream(17), (), 0)
    handoff = sampler.run(_pocket(), torch.tensor([3]), GaussianStream(17), (3, 2, 1), 0)
    assert require_h0_exact(baseline, handoff)["passed"]
    assert len(handoff.opportunities) == 3
    assert counters["proposal_calls"] == counters["score_calls"] == 0


def test_nonzero_handoff_reaches_native_features_and_uses_same_indexed_noise():
    sampler, counters = _sampler()
    state = sampler.stepper.initialize(_pocket(), torch.tensor([3]), GaussianStream(17))
    corrected, receipt = sampler.guidance(state, 5)
    assert corrected is not state
    assert receipt["applied"] and receipt["hard_elements_changed"]
    assert not receipt["hard_bonds_changed"]
    assert counters["proposal_calls"] == 1 and counters["score_calls"] == 2
    original_stream, changed_stream = GaussianStream(17), GaussianStream(17)
    sampler.stepper.reverse_step(state, original_stream)
    sampler.stepper.reverse_step(corrected, changed_stream)
    assert original_stream.receipts == changed_stream.receipts


def test_worsening_rebuilt_native_score_restores_original_state():
    sampler, _ = _sampler(worsening=True)
    state = sampler.stepper.initialize(_pocket(), torch.tensor([3]), GaussianStream(17))
    corrected, receipt = sampler.guidance(state, 5)
    assert corrected is state
    assert receipt["reason"] == "rebuilt_native_objective_worsened"
    assert not receipt["applied"]


def test_missing_required_admission_check_is_not_a_skippable_opportunity():
    sampler, _ = _sampler(missing_check=True)
    state = sampler.stepper.initialize(_pocket(), torch.tensor([3]), GaussianStream(17))
    with pytest.raises(RuntimeError, match="total_derivative"):
        sampler.guidance(state, 1)


def test_H0_uses_structural_checks_without_fabricated_derivative_checks():
    sampler, _ = _sampler()
    state = sampler.stepper.initialize(_pocket(), torch.tensor([3]), GaussianStream(17))
    original_control = sampler.callbacks.admit_control

    def control(native_state, clean):
        admission = original_control(native_state, clean)
        return replace(admission, checks={key: True for key in REQUIRED_CONTROL_ADMISSION_CHECKS})

    sampler.callbacks = replace(sampler.callbacks, admit_control=control)
    result, receipt = sampler.guidance(state, 0)
    assert result is state
    assert receipt["reason"] == "H0_noop"
    with pytest.raises(RuntimeError, match="Mandatory admission checks"):
        control(state, sampler.stepper.clean_estimate(state)).require(3, optimization=True)


def test_prerequisite_failure_blocks_loop_construction():
    sampler, _ = _sampler()
    with pytest.raises(RuntimeError, match="prerequisites"):
        AlternatingSampler(sampler.stepper, sampler.callbacks, ("C", "N"),
                           sampler.transport, {"native_replay": True})


def test_S_runs_twenty_blocks_after_terminal_without_diffusion_draws():
    sampler, counters = _sampler(reject=True)
    stream = GaussianStream(17)
    final = sampler.stepper.replay(_pocket(), torch.tensor([3]), stream)
    before_draws = list(stream.receipts)
    before_core_calls = list(sampler.stepper.core.calls)

    def terminal_admission(sample):
        assert torch.equal(sample.positions, final.positions)
        raise HandoffRejected("native_terminal_ineligible")

    result = sampler.run_sequential(final, terminal_admission)
    assert result.final is final
    assert len(result.blocks) == 20
    assert all(r["after_native_terminal_draw"] for r in result.blocks)
    assert stream.receipts == before_draws
    assert sampler.stepper.core.calls == before_core_calls
    assert counters["proposal_calls"] == 0
