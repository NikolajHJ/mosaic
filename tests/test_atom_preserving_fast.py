"""Reference-equivalent search/teacher/loss fixtures; numerical execution on HPC."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss, search_atom_preserving_repair
from latentmol.atom_preserving_fast import search_atom_preserving_repair_fast
from latentmol.chemistry import encode_smiles
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec
from test_atom_preserving_chemistry import _decoded, _probabilities


def semantic(receipt):
    result = {k: v for k, v in receipt.items() if k not in ("elapsed_seconds", "cache_hits", "search_checks")}
    result["attempts"] = [{k: v for k, v in row.items() if k != "cache_hit"} for row in result["attempts"]]
    return result


@pytest.mark.parametrize("smiles", ["CCC", "CCO", "C=O", "C#N", "c1ccccc1", "c1ccncc1", "CS(=O)(=O)C",
                                    "FS(F)(F)(F)(F)F", "C[NH2+]C", "CC(=O)[O-]"])
@pytest.mark.parametrize("mode", ["native", "disconnected", "triple", "aromatic"])
def test_ordered_search_complete_receipt_matches_reference(smiles, mode):
    spec = GraphSpec(max_nodes=9)
    encoded = encode_smiles(smiles, spec)
    raw = tuple(np.array(encoded[k], dtype=np.int64) for k in ("atom_types", "charges", "bond_types"))
    slots = np.flatnonzero(raw[0]); i, j = slots[:2]
    if mode != "native":
        raw[2][i, j] = raw[2][j, i] = {"disconnected": 0, "triple": 3, "aromatic": 4}[mode]
    # Equal support creates real cost ties; exact zero support exercises flooring.
    p = (np.eye(len(spec.atomic_numbers))[raw[0]], np.eye(len(spec.formal_charges))[raw[1]],
         np.full((*raw[2].shape, len(spec.bond_names)), 1 / len(spec.bond_names)))
    policy = ProbabilityRepairConfig(max_edits=3, checks_per_depth=24, beam_width=4)
    assert search_atom_preserving_repair_fast(raw, p, spec, policy) == search_atom_preserving_repair(raw, p, spec, policy)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("budget", [(1, 1, 1), (2, 12, 3), (3, 32, 4)])
def test_probability_rankings_zero_support_and_censored_search(seed, budget):
    spec = GraphSpec(max_nodes=5)
    output = _decoded(spec, ("invalid",))
    hard = output.harden(spec)
    raw = tuple(v[0].numpy().copy() for v in (hard.atoms, hard.charges, hard.bonds))
    rng = np.random.default_rng(seed)
    bond = rng.random((5, 5, len(spec.bond_names)))
    bond = bond + bond.transpose(1, 0, 2)
    if seed == 0:
        bond[..., 0] = 0
    bond /= bond.sum(-1, keepdims=True)
    p = (np.eye(len(spec.atomic_numbers))[raw[0]], np.eye(len(spec.formal_charges))[raw[1]], bond)
    depth, checks, beam = budget
    policy = ProbabilityRepairConfig(max_edits=depth, checks_per_depth=checks, beam_width=beam)
    assert search_atom_preserving_repair_fast(raw, p, spec, policy) == search_atom_preserving_repair(raw, p, spec, policy)


def test_fast_search_roundtrip_rejected_intermediate_remains_searchable(monkeypatch):
    import latentmol.atom_preserving_chemistry as reference
    import latentmol.atom_preserving_fast as fast
    spec = GraphSpec(max_nodes=3)
    encoded = encode_smiles("CCC", spec)
    raw = tuple(np.array(encoded[k], dtype=np.int64) for k in ("atom_types", "charges", "bond_types"))
    raw[2][raw[2] == 1] = 3
    p = (np.eye(len(spec.atomic_numbers))[raw[0]], np.eye(len(spec.formal_charges))[raw[1]],
         np.eye(len(spec.bond_names))[raw[2]])
    original = encode_smiles
    def require_propane(smiles, graph_spec):
        if smiles != "CCC":
            raise ValueError("fixture rejects intermediate")
        return original(smiles, graph_spec)
    monkeypatch.setattr(reference, "encode_smiles", require_propane)
    monkeypatch.setattr(fast, "encode_smiles", require_propane)
    config = ProbabilityRepairConfig(max_edits=2, checks_per_depth=128, beam_width=32)
    result = fast.search_atom_preserving_repair_fast(raw, p, spec, config)
    assert result == reference.search_atom_preserving_repair(raw, p, spec, config)
    assert result["roundtrip_rejected_candidates"] > 0 and result["weighted_witness"]["edit_count"] == 2


@pytest.mark.parametrize("workers", [0, 2])
def test_teacher_lru_target_receipt_loss_and_gradient_exact(workers):
    spec = GraphSpec(max_nodes=3)
    config = ProbabilityRepairConfig(max_edits=2, checks_per_depth=16, beam_width=3)
    with AtomPreservingRepairTeacher(spec, config, cache_capacity=2) as ref, \
         AtomPreservingRepairTeacher(spec, config, cache_capacity=2, backend="optimized", workers=workers) as fast:
        # Repeated states, within/across batches and after eviction, exercise
        # exact sequential cache semantics independently of worker completion.
        for update, kinds in enumerate([("invalid", "valid", "invalid", "empty", "invalid"),
                                        ("invalid", "valid", "disconnected", "invalid")]):
            left, right = _decoded(spec, kinds), _decoded(spec, kinds)
            ids = list(range(update * 100, update * 100 + len(kinds)))
            first = ref.build_targets(left, candidate_ids=ids, update=update)
            second = fast.build_targets(right, candidate_ids=ids, update=update)
            assert semantic(first.receipt) == semantic(second.receipt)
            assert first.receipt["cache_hits"] == second.receipt["cache_hits"]
            assert first.receipt["search_checks"] == second.receipt["search_checks"]
            losses = [repair_loss(output, batch, spec) for output, batch in ((left, first), (right, second))]
            assert torch.equal(*losses)
            grads = [torch.autograd.grad(loss, (output.atoms, output.charges, output.bonds))
                     for loss, output in zip(losses, (left, right))]
            assert all(torch.equal(a, b) for a, b in zip(*grads))
        ref.clear_cache(); fast.clear_cache()
        assert not ref._cache and not fast._cache


def test_fast_teacher_does_not_advance_parent_rng_or_modify_logits():
    spec = GraphSpec(max_nodes=3)
    output = _decoded(spec, ("invalid", "empty"))
    tensors = [v.detach().clone() for v in (output.atoms, output.charges, output.bonds)]
    state = (random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone())
    with AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(), backend="optimized", workers=2) as teacher:
        teacher.build_targets(output, candidate_ids=[1, 2], update=1)
    assert random.getstate() == state[0]
    new = np.random.get_state()
    assert new[0] == state[1][0] and np.array_equal(new[1], state[1][1]) and new[2:] == state[1][2:]
    assert torch.equal(torch.get_rng_state(), state[2])
    assert all(torch.equal(a, b) for a, b in zip(tensors, (output.atoms, output.charges, output.bonds)))
