"""Allocated CPU/CUDA contracts for frozen valid-reference targets, not efficacy.

Numerical tests must run on DTU HPC. In particular, these fixtures distinguish
an informative anchor after a live-invalid transition from the historical
current-output eligibility dropout; they do not prove generation improves.
"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from latentmol import atom_preserving_chemistry as repair
from latentmol.chemistry_classification import classify_repair_graph
from latentmol.chemistry_valid_reference_guidance import (
    VERSION, prepare_valid_reference_guidance, valid_prior_reference_nll,
    validate_valid_prior_reference_guidance,
)
from latentmol.chemistry_valid_self_guidance import prepare_valid_self_guidance, valid_prior_self_nll
from latentmol.representation import DecoderOutput, GraphSpec
from test_atom_preserving_chemistry import _decoded
from test_chemistry_valid_self_guidance import device


POLICY = dict(version=VERSION, weight=.1)
REFERENCE = dict(epoch=36, checkpoint_sha256="a"*64, model_sha256="b"*64)
LATENT_SHA = "c"*64


def _fields(decoded):
    return decoded.atoms, decoded.charges, decoded.bonds


def _live(decoded, device="cpu", dtype=None):
    return DecoderOutput(*(value.detach().to(device=device, dtype=dtype or value.dtype).clone().requires_grad_()
                           for value in _fields(decoded)))


def _prepare(decoded, spec, **kwargs):
    values = dict(candidate_ids=list(range(40, 40+len(decoded.atoms))), update=7,
                  reference_identity=deepcopy(REFERENCE), latent_sha256=LATENT_SHA)
    values.update(kwargs)
    return prepare_valid_reference_guidance(decoded, spec, **values)


def _old_self(decoded, spec):
    from latentmol.chemistry_classification import VERSION as CLASSIFICATION_VERSION
    from latentmol.probability_repair import ProbabilityRepairConfig
    with repair.AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=32),
                                           classification_version=CLASSIFICATION_VERSION) as teacher:
        base = teacher.build_targets(decoded, candidate_ids=list(range(40, 40+len(decoded.atoms))), update=7)
    return prepare_valid_self_guidance(decoded, base, spec)


def _chain(dtype=torch.float64):
    """A nondegenerate actual C4 chain with three present and three NONE pairs."""
    spec = GraphSpec(max_nodes=4)
    atom = torch.full((1,4,len(spec.atomic_numbers)), -3., dtype=dtype)
    atom[..., spec.atomic_numbers.index(6)] = 2.
    charge = torch.full((1,4,len(spec.formal_charges)), -3., dtype=dtype)
    charge[..., spec.formal_charges.index(0)] = 2.
    bond = torch.full((1,4,4,len(spec.bond_names)), -3., dtype=dtype)
    bond[...,0] = 2.
    for left in range(3):
        bond[0,left,left+1,0] = bond[0,left+1,left,0] = -3.
        bond[0,left,left+1,spec.bond_names.index("SINGLE")] = 3.
        bond[0,left+1,left,spec.bond_names.index("SINGLE")] = 3.
    # Asymmetric but small deterministic variation exercises symmetric logits
    # without introducing another sampled latent or changing the hard chain.
    values = [value + torch.arange(value.numel(), dtype=dtype).reshape_as(value)*.0001
              for value in (atom,charge,bond)]
    return spec, DecoderOutput(*(value.requires_grad_() for value in values))


def test_closed_reference_policy_is_metadata_safe_and_defensively_copied():
    assert validate_valid_prior_reference_guidance(None) is None
    result = validate_valid_prior_reference_guidance(POLICY)
    result["weight"] = 9.
    assert POLICY["weight"] == .1


@pytest.mark.parametrize("value", [True, {}, {"version":VERSION},
    {"version":"unknown","weight":.1}, {"version":VERSION,"weight":1},
    {"version":VERSION,"weight":True}, {"version":VERSION,"weight":.01},
    {"version":VERSION,"weight":float("nan")}, {"version":VERSION,"weight":float("inf")},
    {"version":VERSION,"weight":.1,"extra":0}])
def test_policy_rejects_aliases_unknown_versions_and_changed_coefficients(value):
    with pytest.raises(ValueError): validate_valid_prior_reference_guidance(value)


def test_reference_preparation_classifies_only_and_never_creates_a_repair_receipt(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    decoded = _decoded(spec, ("valid","invalid","empty"))
    def forbidden(*args, **kwargs):
        raise AssertionError("A frozen reference classifier called repair search")
    monkeypatch.setattr(repair, "search_atom_preserving_repair", forbidden)
    monkeypatch.setattr(repair.AtomPreservingRepairTeacher, "build_targets", forbidden)
    before = torch.get_rng_state().clone()
    prepared = _prepare(decoded,spec)
    assert torch.equal(before,torch.get_rng_state())
    assert prepared.candidate_ids == (40,41,42) and prepared.eligible_rows == (0,)
    receipt = prepared.receipt
    assert receipt["scheduled_prior_attempts"] == 3
    assert receipt["strict_valid_reference_targets"] == 1 and receipt["excluded_prior_attempts"] == 2
    assert receipt["accepted_repair_witnesses_created"] == receipt["reference_searches"] == 0
    assert "original_accepted_repair_targets" not in receipt
    assert receipt["reference_identity"] == REFERENCE and receipt["latent_sha256"] == LATENT_SHA
    assert receipt["reference_classification"]["counts"]["strict_teacher_valid"] == 1


@pytest.mark.parametrize("dtype", [torch.float32,torch.float64])
def test_original36_same_snapshot_matches_historical_full_nll_scalar_and_all_head_vjp_exactly(device,dtype):
    spec = GraphSpec(max_nodes=3)
    reference = _live(_decoded(spec,("valid","invalid","empty")),dtype=dtype)
    old = _old_self(reference,spec)
    prepared = _prepare(reference,spec)
    live = _live(reference,device)
    actual = valid_prior_reference_nll(live,prepared,spec,POLICY)
    expected = valid_prior_self_nll(live,old,spec,dict(version="valid-prior-self-nll-v1",weight=.1))
    assert prepared.eligible_rows == old.eligible_rows
    assert torch.equal(actual["loss"],expected["loss"])
    assert torch.equal(actual["weighted_loss"],expected["weighted_loss"])
    left = torch.autograd.grad(actual["weighted_loss"],_fields(live),retain_graph=True)
    right = torch.autograd.grad(expected["weighted_loss"],_fields(live))
    assert all(torch.equal(a,b) for a,b in zip(left,right))
    assert all(value.grad is None for value in _fields(reference))


def test_live_invalid_c4_keeps_frozen_valid_target_and_correct_gradient_when_current_self_drops_out(device):
    spec, reference = _chain()
    prepared = _prepare(reference,spec)
    live_cpu = _live(reference)
    triple, none = spec.bond_names.index("TRIPLE"), 0
    with torch.no_grad():
        # Carbon2 already has two single bonds, so this new triple bond makes
        # the live graph invalid. The frozen target keeps NONE on this pair.
        live_cpu.bonds[0,0,2,triple] = live_cpu.bonds[0,2,0,triple] = 14.
    hard = live_cpu.harden(spec)
    verdict = classify_repair_graph(*(value[0].detach().numpy() for value in _fields(hard)),spec)
    assert verdict["strict_teacher_valid"] is False and verdict["reason"] == "sanitization"
    current = _old_self(live_cpu,spec)
    assert current.eligible_rows == () and prepared.eligible_rows == (0,)
    live = _live(live_cpu,device)
    actual = valid_prior_reference_nll(live,prepared,spec,POLICY)
    old = valid_prior_self_nll(live,current,spec,dict(version="valid-prior-self-nll-v1",weight=.1))
    assert actual["weighted_loss"].item() > 0 and old["weighted_loss"].item() == 0
    gradients = torch.autograd.grad(actual["weighted_loss"],_fields(live),retain_graph=True)
    excluded = torch.autograd.grad(old["weighted_loss"],_fields(live))
    assert all(torch.count_nonzero(value) > 0 for value in gradients)
    assert all(torch.count_nonzero(value) == 0 for value in excluded)
    assert gradients[2][0,0,2,none].item() < 0 < gradients[2][0,0,2,triple].item()
    torch.testing.assert_close(gradients[2],gradients[2].transpose(1,2),rtol=0,atol=0)
    assert prepared.graph.bonds[0,0,2].item() == none
    assert actual["reference_identity"] == REFERENCE


def test_manual_all_attempt_mean_masks_and_ordered_microbatch_vjp(device):
    spec = GraphSpec(max_nodes=3)
    reference = _decoded(spec,("valid","invalid","empty"))
    prepared = _prepare(reference,spec)
    live = _live(reference,device)
    result = valid_prior_reference_nll(live,prepared,spec,POLICY)
    target = prepared.graph.to(device)
    active = target.atoms[0] != 0
    atom = F.log_softmax(live.atoms[0],-1).gather(-1,target.atoms[0,:,None]).sum()
    charge = F.log_softmax(live.charges[0],-1).gather(-1,target.charges[0,:,None]).squeeze(-1)[active].sum()
    left,right = torch.triu_indices(spec.max_nodes,spec.max_nodes,1,device=device)
    occupied_pair = active[left]&active[right]
    bonds = F.log_softmax(live.symmetric_bond_logits[0],-1)
    bond = bonds[left[occupied_pair],right[occupied_pair],target.bonds[0,left[occupied_pair],right[occupied_pair]]].sum()
    k = int(active.sum())
    direct = -(atom+charge+bond)/(spec.max_nodes+k+k*(k-1)//2)/3
    torch.testing.assert_close(result["loss"],direct,rtol=1e-13,atol=1e-14)
    pieces = []
    for indices in ((0,),(1,2)):
        chunk = DecoderOutput(*(value[list(indices)] for value in _fields(live)))
        pieces.append(valid_prior_reference_nll(chunk,prepared.subset(indices),spec,POLICY)["loss"]*len(indices)/3)
    left_grad = torch.autograd.grad(result["loss"],_fields(live),retain_graph=True)
    right_grad = torch.autograd.grad(sum(pieces),_fields(live))
    torch.testing.assert_close(sum(pieces),result["loss"],rtol=1e-13,atol=1e-14)
    for a,b in zip(left_grad,right_grad):
        torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-14)
        assert torch.count_nonzero(a[1:]) == 0
    assert torch.count_nonzero(left_grad[0][0,2]) > 0  # Every atom slot, including NO_ATOM.
    assert torch.count_nonzero(left_grad[1][0,2]) == 0
    assert torch.count_nonzero(left_grad[2][0,0,2]) == 0
    assert torch.count_nonzero(left_grad[2].diagonal(dim1=1,dim2=2)) == 0
    selected = prepared.subset([2,0])
    assert selected.candidate_ids == (42,40) and selected.eligible_rows == (1,)


@pytest.mark.parametrize("dtype", [torch.float32,torch.float64])
@pytest.mark.parametrize("field,coordinate", [("atoms",(0,0,1)),("charges",(0,1,2)),("bonds",(0,0,2,0))])
def test_fixed_reference_nll_has_finite_nonzero_central_difference_for_each_head(device,dtype,field,coordinate):
    spec, reference = _chain(dtype)
    prepared = _prepare(reference,spec)
    live = _live(reference,device)
    value = valid_prior_reference_nll(live,prepared,spec,POLICY)["weighted_loss"]
    derivative = torch.autograd.grad(value,getattr(live,field))[0][coordinate]
    assert torch.isfinite(derivative) and derivative.item() != 0
    epsilon = 2e-3 if dtype == torch.float32 else 1e-5
    values = []
    for sign in (1.,-1.):
        changed = _live(reference,device)
        with torch.no_grad(): getattr(changed,field)[coordinate] += sign*epsilon
        values.append(float(valid_prior_reference_nll(changed,prepared,spec,POLICY)["weighted_loss"]))
    numeric = (values[0]-values[1])/(2*epsilon)
    assert numeric == pytest.approx(float(derivative),rel=5e-3 if dtype == torch.float32 else 2e-6,
                                   abs=3e-6 if dtype == torch.float32 else 1e-10)


def test_reference_invalid_rows_keep_all_heads_connected_to_zero_even_for_valid_live_outputs(device):
    spec = GraphSpec(max_nodes=3)
    prepared = _prepare(_decoded(spec,("invalid","empty")),spec)
    live = _live(_decoded(spec,("valid","valid")),device)
    result = valid_prior_reference_nll(live,prepared,spec,POLICY)
    assert result["loss"].item() == 0 and result["eligible_rows"] == ()
    assert result["valid_mask"].tolist() == [False,False]
    gradients = torch.autograd.grad(result["weighted_loss"],_fields(live))
    assert all(torch.count_nonzero(value) == 0 for value in gradients)


def test_all_public_readers_and_original_inputs_are_defensive_copies():
    spec, decoded = _chain()
    identity = deepcopy(REFERENCE)
    prepared = _prepare(decoded,spec,reference_identity=identity)
    expected = deepcopy(prepared.receipt)
    prepared.graph.atoms.zero_()
    prepared.reference_identity["checkpoint_sha256"] = "f"*64
    prepared.classification_records[0]["classification"]["smiles"] = "changed"
    prepared.receipt["reference_classification"]["counts"]["strict_teacher_valid"] = 0
    identity["model_sha256"] = "e"*64
    with torch.no_grad(): decoded.atoms.zero_()
    assert prepared.receipt == expected
    assert prepared.graph.atoms[0,0].item() == spec.atomic_numbers.index(6)


@pytest.mark.parametrize("part", ["graph","receipt","reference","probability","latent","eligible","view"])
def test_private_state_or_detached_provenance_tampering_is_rejected(part):
    spec, decoded = _chain()
    prepared = _prepare(decoded,spec)
    if part == "graph": prepared._graph.atoms[0,0] = 0
    elif part == "receipt": prepared._source_receipt["source_update"] += 1
    elif part == "reference": prepared._source_receipt["reference_identity"]["model_sha256"] = "f"*64
    elif part == "probability": prepared._source_receipt["records"][0]["probability_sha256"] = "f"*64
    elif part == "latent": prepared._source_receipt["latent_sha256"] = "f"*64
    elif part == "eligible": prepared = replace(prepared,_eligible=())
    else: prepared = replace(prepared,_view=(0,0))
    with pytest.raises(ValueError,match="identity"):
        valid_prior_reference_nll(decoded,prepared,spec,POLICY)


@pytest.mark.parametrize("change", [lambda x:x.update(epoch=37),lambda x:x.update(epoch=True),
    lambda x:x.update(checkpoint_sha256="a"*63),lambda x:x.update(model_sha256="B"*64),
    lambda x:x.update(extra=0)])
def test_reference_requires_exact_original36_checkpoint_and_model_metadata(change):
    spec, decoded = _chain(); identity = deepcopy(REFERENCE); change(identity)
    with pytest.raises(ValueError): _prepare(decoded,spec,reference_identity=identity)


@pytest.mark.parametrize("kwargs", [dict(candidate_ids=[]),dict(candidate_ids=[True]),dict(update=True),
    dict(update=-1),dict(latent_sha256=None),dict(latent_sha256="not-a-hash")])
def test_point_alignment_update_and_latent_identity_are_explicit(kwargs):
    spec, decoded = _chain()
    with pytest.raises(ValueError): _prepare(decoded,spec,**kwargs)


def test_native_batch_classifier_contract_preserves_order_and_does_not_expose_mutable_source():
    spec = GraphSpec(max_nodes=3); decoded = _decoded(spec,("valid","invalid","empty"))
    calls, returned = [], []
    def classifier(graphs,actual_spec):
        calls.append(len(graphs))
        assert actual_spec is spec
        assert all(not value.flags.writeable for graph in graphs for value in graph)
        returned.extend(classify_repair_graph(*graph,spec) for graph in graphs)
        return returned
    actual = _prepare(decoded,spec,classifier=classifier)
    expected = _prepare(decoded,spec)
    assert calls == [3] and actual.receipt == expected.receipt and actual.identity == expected.identity
    returned[0]["strict_teacher_valid"] = False
    assert actual.eligible_rows == (0,) and actual.receipt == expected.receipt


@pytest.mark.parametrize("defect", ["missing","count","version","strict_alias","missing_stage","contradictory_valid"])
def test_incomplete_or_non_native_classifier_verdicts_are_rejected(defect):
    spec, decoded = _chain()
    def classifier(graphs,actual_spec):
        if defect == "missing": return None
        if defect == "count": return []
        result = [classify_repair_graph(*graph,actual_spec) for graph in graphs]
        if defect == "version": result[0]["version"] = "other"
        elif defect == "strict_alias": result[0]["strict_teacher_valid"] = 1
        elif defect == "missing_stage": del result[0]["raw_chemical_valid"]
        else: result[0]["schema_roundtrip_valid"] = False
        return result
    with pytest.raises(ValueError): _prepare(decoded,spec,classifier=classifier)


def test_reference_probability_identity_changes_without_changing_frozen_labels():
    spec, decoded = _chain()
    original = _prepare(decoded,spec)
    changed = _live(decoded)
    with torch.no_grad(): changed.atoms[0,0,spec.atomic_numbers.index(6)] += .1
    other = _prepare(changed,spec)
    assert original.receipt["raw_graph_sha256"] == other.receipt["raw_graph_sha256"]
    assert original.receipt["source_probability_sha256"] != other.receipt["source_probability_sha256"]
    assert original.identity != other.identity


def test_a_native_callback_cannot_change_its_detached_input_before_claiming_validity():
    spec, decoded = _chain()
    def classifier(graphs,actual_spec):
        # Read-only copies protect ordinary adapters. Also detect an adapter
        # deliberately re-enabling writes instead of classifying its input.
        graphs[0][0].setflags(write=True)
        graphs[0][0][0] = 0
        return [classify_repair_graph(*graph,actual_spec) for graph in graphs]
    with pytest.raises(ValueError,match="changed"):
        _prepare(decoded,spec,classifier=classifier)


def test_cpu_reference_source_live_shape_and_subset_contracts(device):
    spec, decoded = _chain(); prepared = _prepare(decoded,spec)
    if device == "cuda":
        with pytest.raises(ValueError,match="CPU"): _prepare(_live(decoded,device),spec)
    for indices in ([],[0,0],[True],[1]):
        with pytest.raises(ValueError): prepared.subset(indices)
    live = _live(decoded,device)
    with pytest.raises(ValueError): valid_prior_reference_nll(live,prepared,spec,None)
    wrong = DecoderOutput(*(value.repeat(2,*([1]*(value.ndim-1))) for value in _fields(live)))
    with pytest.raises(ValueError,match="misaligned"):
        valid_prior_reference_nll(wrong,prepared,spec,POLICY)
