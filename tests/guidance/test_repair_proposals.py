"""Fixed-atom discovery, failure honesty, legacy isolation and useful NLL paths."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.guidance.repair_proposals import (ProposalRepairTeacher, RepairProposalConfig,
                                               search_with_proposals)
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec


def raw_graph(spec, count, *, complete=False, fragments=False):
    atoms = np.zeros(spec.max_nodes, dtype=np.int64)
    atoms[:count] = spec.atomic_numbers.index(6)
    charges = np.full(spec.max_nodes, spec.formal_charges.index(0), dtype=np.int64)
    bonds = np.zeros((spec.max_nodes, spec.max_nodes), dtype=np.int64)
    for i in range(count):
        for j in range(i+1, count):
            if complete or (fragments and i//2 == j//2):
                bonds[i, j] = bonds[j, i] = spec.bond_names.index("SINGLE")
    return atoms, charges, bonds


def decoded(raw, spec):
    arrays = [np.full((*v.shape, classes), -8., dtype=np.float64)
              for v, classes in zip(raw, (len(spec.atomic_numbers), len(spec.formal_charges), len(spec.bond_names)))]
    for labels, logits in zip(raw, arrays):
        np.put_along_axis(logits, labels[..., None], 0., axis=-1)
    return DecoderOutput(*(torch.tensor(v[None], requires_grad=True) for v in arrays))


def probabilities(output):
    return tuple(x.detach()[0].numpy() for x in (output.atoms.softmax(-1), output.charges.softmax(-1),
                                                output.symmetric_bond_logits.softmax(-1)))


def assert_witness(report, raw, spec):
    assert report["status"] == "witness_found"
    labels = report["weighted_witness"]["labels"]
    assert labels["atoms"] == raw[0].tolist()
    assert labels["charges"] == raw[1].tolist()
    assert inspect_graph(*(np.array(labels[k]) for k in ("atoms", "charges", "bonds")), spec).smiles


def test_component_bridges_join_more_than_one_pair_of_fragments():
    spec = GraphSpec(max_nodes=10); raw = raw_graph(spec, 8, fragments=True)
    out = decoded(raw, spec)
    report = search_with_proposals(raw, probabilities(out), spec, ProbabilityRepairConfig(),
                                  RepairProposalConfig(sample_draws=0, prune_dense=False))
    assert_witness(report, raw, spec)
    assert report["weighted_witness"]["edit_count"] == 3
    assert report["accepted_source"] == "macro" and report["checks"] == 1


def test_dense_repair_can_exceed_beam_edit_radius_without_atom_shortcuts():
    spec = GraphSpec(max_nodes=10); raw = raw_graph(spec, 8, complete=True)
    report = search_with_proposals(raw, probabilities(decoded(raw, spec)), spec,
                                  ProbabilityRepairConfig(max_edits=1), RepairProposalConfig(sample_draws=0))
    assert_witness(report, raw, spec)
    assert report["weighted_witness"]["edit_count"] >= 12
    assert report["weighted_witness"]["edit_counts"]["atom"] == 0
    assert report["accepted_source"] == "macro"


def test_target_builder_keeps_native_normalization_and_attached_gradient():
    spec = GraphSpec(max_nodes=10); raw = raw_graph(spec, 8, complete=True)
    out = decoded(raw, spec)
    with ProposalRepairTeacher(spec, ProbabilityRepairConfig(), proposals=RepairProposalConfig(sample_draws=0)) as teacher:
        targets = teacher.build_targets(out, candidate_ids=[5], update=0)
        assert targets.receipt["status_counts"] == {"accepted": 1}
        loss = repair_loss(out, targets, spec)
        loss.backward()
        assert torch.isfinite(loss) and loss > 0
        assert out.bonds.grad is not None and torch.isfinite(out.bonds.grad).all()
        assert float(out.bonds.grad.norm()) > 0
        target = targets.targets[0]
        assert target.decisions == spec.max_nodes + 8 + 8*7//2
        for tensor in target.graph.__dict__.values():
            assert tensor.device.type == "cpu" and not tensor.requires_grad
        for logits in (out.atoms, out.charges):
            assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("smiles", ["c1ccccc1", "CCS(=O)(=O)C", "C1CCCCC1", "O=S(=O)=O"])
def test_already_valid_chemistry_is_not_rewritten(smiles):
    spec = GraphSpec(max_nodes=10); encoded = encode_smiles(smiles, spec)
    raw = tuple(np.array(encoded[k]) for k in ("atom_types", "charges", "bond_types"))
    report = search_with_proposals(raw, probabilities(decoded(raw, spec)), spec,
                                  ProbabilityRepairConfig(), RepairProposalConfig())
    assert report["status"] == "already_valid" and report["checks"] == 0


def test_impossible_fixed_atom_connectivity_is_censored_not_claimed_repaired():
    spec = GraphSpec(max_nodes=5); raw = raw_graph(spec, 3)
    raw[0][:3] = spec.atomic_numbers.index(9)
    report = search_with_proposals(raw, probabilities(decoded(raw, spec)), spec,
        ProbabilityRepairConfig(max_edits=1), RepairProposalConfig(sample_draws=0, candidate_checks=3))
    assert report["status"] == "censored" and report["weighted_witness"] is None
    assert report["macro_status"] == "insufficient_total_valence_capacity"
    assert report["checks"] <= 3


def test_sampling_is_symmetric_fixed_slot_and_does_not_consume_global_rng():
    spec = GraphSpec(max_nodes=5); raw = raw_graph(spec, 4)
    out = decoded(raw, spec)
    with torch.no_grad():
        out.bonds[..., 1] = -1.
    p = probabilities(out)
    config = RepairProposalConfig(sample_draws=8, connect_components=False, prune_dense=False)
    state = torch.get_rng_state().clone()
    report = search_with_proposals(raw, p, spec, ProbabilityRepairConfig(), config)
    assert torch.equal(state, torch.get_rng_state())
    assert report == search_with_proposals(raw, p, spec, ProbabilityRepairConfig(), config)
    assert_witness(report, raw, spec)
    assert report["accepted_source"] == "bond_sample"


def test_disabled_proposals_keep_legacy_target_and_loss():
    spec = GraphSpec(max_nodes=5); raw = raw_graph(spec, 2); out = decoded(raw, spec)
    search = ProbabilityRepairConfig(max_edits=1, checks_per_depth=8)
    policy = RepairProposalConfig(sample_draws=0, connect_components=False, prune_dense=False, candidate_checks=8)
    with AtomPreservingRepairTeacher(spec, search, backend="optimized") as old, \
         ProposalRepairTeacher(spec, search, proposals=policy) as new:
        a = old.build_targets(out, candidate_ids=[0], update=0)
        b = new.build_targets(out, candidate_ids=[0], update=0)
        assert a.targets[0].record() == b.targets[0].record()
        assert torch.equal(repair_loss(out, a, spec), repair_loss(out, b, spec))
        assert a.receipt["policy"] != b.receipt["policy"]


def test_serial_async_and_worker_execution_preserve_same_targets():
    spec = GraphSpec(max_nodes=6); raw = raw_graph(spec, 6, complete=True); out = decoded(raw, spec)
    policy = RepairProposalConfig(sample_draws=1)
    with ProposalRepairTeacher(spec, ProbabilityRepairConfig(), proposals=policy) as serial, \
         ProposalRepairTeacher(spec, ProbabilityRepairConfig(), proposals=policy, workers=1) as parallel:
        first = serial.build_targets(out, candidate_ids=[4], update=2)
        second = parallel.build_targets_async(out, candidate_ids=[4], update=2).result()
        assert first.targets[0].record() == second.targets[0].record()
        assert first.receipt["policy"] == second.receipt["policy"]


def test_input_and_budget_errors_are_not_hidden_as_failed_repair():
    spec = GraphSpec(max_nodes=4); raw = raw_graph(spec, 3)
    p = probabilities(decoded(raw, spec)); p[2][0, 1, 0] = np.nan
    with pytest.raises(ValueError):
        search_with_proposals(raw, p, spec, ProbabilityRepairConfig(), RepairProposalConfig())
    with pytest.raises(ValueError):
        replace(RepairProposalConfig(), candidate_checks=0)


def test_proposal_and_fallback_share_check_and_edit_caps_and_do_not_mutate_inputs():
    spec = GraphSpec(max_nodes=10); raw = raw_graph(spec, 8, complete=True)
    p = probabilities(decoded(raw, spec)); originals = [x.copy() for x in (*raw, *p)]
    report = search_with_proposals(raw, p, spec, ProbabilityRepairConfig(max_edits=3),
        RepairProposalConfig(candidate_checks=2, max_bond_changes=1))
    assert report["checks"] <= 2
    assert report["status"] == "censored"  # One edit cannot fix complete C8.
    assert all(np.array_equal(a, b) for a, b in zip((*raw, *p), originals))


def test_existing_factory_requires_explicit_proposal_opt_in():
    from latentmol.failure_guidance_training import build_repair_teacher
    spec = GraphSpec(max_nodes=5)
    protocol = {"search": {"max_edits": 1, "checks_per_depth": 4, "beam_width": 1}}
    with build_repair_teacher(spec, protocol) as old, \
         build_repair_teacher(spec, protocol, proposal_config={"sample_draws": 0}) as new:
        assert type(old) is AtomPreservingRepairTeacher
        assert type(new) is ProposalRepairTeacher
    assert "proposal_config" not in protocol
