"""Allocated-HPC parity/tamper checks for same-call repair validation reuse.

The historical NLL below is the pre-extraction execution oracle. No validation
bundle may survive a forward call; public loss calls always inspect current
content. All numerical execution belongs on allocated DTU HPC.
"""
from dataclasses import replace

import pytest
import torch

from latentmol import atom_preserving_chemistry as chemistry
from latentmol import repair_failure_guidance as guidance
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.sampling import hard_graph_log_prob_terms
from test_atom_preserving_chemistry import _decoded, _teacher


def _legacy_repair_loss(decoded, targets, spec):
    """Historical validation and exact tensor arithmetic before extraction."""
    batch = chemistry._validated_decoded(decoded, spec)
    if not isinstance(targets, chemistry.RepairTargetBatch) or len(targets.targets) != batch or len(targets.candidate_ids) != batch:
        raise ValueError("Repair targets and decoder batch disagree")
    if len(set(targets.candidate_ids)) != batch or any(type(value) is not int or value < 0 for value in targets.candidate_ids):
        raise ValueError("Repair loss requires unique nonnegative point IDs")
    total = sum(value.reshape(-1)[0] * 0 for value in (decoded.atoms, decoded.charges, decoded.bonds))
    accepted_rows, accepted_targets = [], []
    for row, target in enumerate(targets.targets):
        if target is None:
            continue
        if target.candidate_id != targets.candidate_ids[row]:
            raise ValueError("Repair target and point ID alignment changed")
        target.verify(spec)
        accepted_rows.append(row)
        accepted_targets.append(target)
    if accepted_rows:
        one = DecoderOutput(*(value[accepted_rows] for value in (decoded.atoms, decoded.charges, decoded.bonds)))
        if one.atoms.dtype in (torch.float16, torch.bfloat16):
            one = DecoderOutput(*(value.float() for value in (one.atoms, one.charges, one.bonds)))
        graph = HardGraphBatch(*(torch.cat([getattr(target._graph, name) for target in accepted_targets], dim=0)
                                 for name in ("atoms", "charges", "bonds"))).to(one.atoms.device)
        terms = hard_graph_log_prob_terms(one, graph, spec)
        if not bool(terms["supported"].all()):
            raise ValueError("Frozen target lies outside the unconditional graph law")
        decisions = one.atoms.new_tensor([target.decisions for target in accepted_targets])
        nll = -(terms["atoms"] + terms["charges"] + terms["bonds"]) / decisions
        if not bool(torch.isfinite(nll).all()):
            raise ValueError("Repair NLL is nonfinite")
        total = total + nll.sum()
    return total / batch


def _fixture(kinds=("invalid", "valid", "invalid", "empty"), *, dtype=torch.float64):
    spec = GraphSpec(max_nodes=3)
    decoded = _decoded(spec, kinds)
    with _teacher(spec) as teacher:
        targets = teacher.build_targets(decoded, candidate_ids=list(range(71, 71 + len(kinds))), update=37)
    live = DecoderOutput(*(value.detach().to(dtype).clone().requires_grad_()
                           for value in (decoded.atoms, decoded.charges, decoded.bonds)))
    return spec, live, targets


def _fresh(decoded):
    return DecoderOutput(*(value.detach().clone().requires_grad_()
                           for value in (decoded.atoms, decoded.charges, decoded.bonds)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kinds", [("invalid",), ("invalid", "valid", "invalid", "empty"),
                                  ("valid", "empty")])
def test_public_and_same_call_private_nll_match_historical_values_and_head_gradients(dtype, kinds):
    spec, decoded, targets = _fixture(kinds, dtype=dtype)
    observed = []
    for mode in ("historical", "public", "same_call"):
        output = _fresh(decoded)
        if mode == "historical":
            loss = _legacy_repair_loss(output, targets, spec)
        elif mode == "public":
            loss = chemistry.repair_loss(output, targets, spec)
        else:
            validated = chemistry._validate_repair_loss_inputs(output, targets, spec)
            loss = chemistry._repair_loss_from_validated(output, validated, spec)
        gradients = torch.autograd.grad(loss, (output.atoms, output.charges, output.bonds))
        observed.append((loss.detach(), gradients))
    for loss, gradients in observed[1:]:
        assert torch.equal(loss, observed[0][0])
        assert all(torch.equal(actual, expected) for actual, expected in zip(gradients, observed[0][1]))
    if all(target is None for target in targets.targets):
        assert observed[0][0] == 0
        assert all(torch.count_nonzero(value) == 0 for value in observed[0][1])


def test_extraction_preserves_exact_shared_parameter_vjp():
    spec, base, targets = _fixture()
    observed = []
    for search in (_legacy_repair_loss, chemistry.repair_loss):
        parameters = torch.linspace(-.1, .1, 9, dtype=torch.float64, requires_grad=True)
        heads = []
        for index, value in enumerate((base.atoms, base.charges, base.bonds)):
            weight = (torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape).remainder(7) + 1) / 10
            heads.append(value.detach() + weight * parameters[index]
                + weight.square() * parameters[index + 3].sin() + .1 * weight * parameters[index + 6].square())
        output = DecoderOutput(*heads)
        loss = search(output, targets, spec)
        observed.append((loss.detach(), torch.autograd.grad(loss, parameters)[0]))
    assert torch.equal(observed[0][0], observed[1][0])
    assert torch.equal(observed[0][1], observed[1][1])
    assert observed[0][1].norm() > 0


def test_validated_records_preserve_all_rows_order_and_have_no_label_aliases():
    spec, decoded, targets = _fixture()
    validated = chemistry._validate_repair_loss_inputs(decoded, targets, spec)
    assert validated.batch == 4 and validated.accepted_rows == (0, 2)
    assert validated.accepted_targets == (targets.targets[0], targets.targets[2])
    assert validated.target_records == tuple(None if target is None else target.record() for target in targets.targets)
    validated.target_records[0]["labels"]["atoms"][0] = 0
    # A metadata consumer cannot mutate private target labels through a record.
    assert targets.targets[0].record()["labels"]["atoms"][0] != 0
    current = chemistry._validate_repair_loss_inputs(decoded, targets, spec)
    assert current.target_records[0] == targets.targets[0].record()


def test_public_loss_revalidates_every_target_once_on_every_call(monkeypatch):
    spec, decoded, targets = _fixture()
    original_verify = chemistry.FrozenAtomPreservingTarget.verify
    original_decoded = chemistry._validated_decoded
    target_calls, decoded_calls = [], []
    def verify(target, graph_spec):
        target_calls.append(target.candidate_id)
        return original_verify(target, graph_spec)
    def shape(output, graph_spec):
        decoded_calls.append(1)
        return original_decoded(output, graph_spec)
    monkeypatch.setattr(chemistry.FrozenAtomPreservingTarget, "verify", verify)
    monkeypatch.setattr(chemistry, "_validated_decoded", shape)
    chemistry.repair_loss(decoded, targets, spec)
    chemistry.repair_loss(decoded, targets, spec)
    assert target_calls == [71, 73, 71, 73] and len(decoded_calls) == 2


def test_enclosing_failure_loss_reuses_target_verification_within_that_call_only(monkeypatch):
    spec, decoded, targets = _fixture()
    prepared = guidance.prepare_failure_guidance(decoded, targets, spec)
    original = chemistry.FrozenAtomPreservingTarget.verify
    calls = []
    def verify(target, graph_spec):
        calls.append(target.candidate_id)
        return original(target, graph_spec)
    monkeypatch.setattr(chemistry.FrozenAtomPreservingTarget, "verify", verify)
    guidance.failure_aware_repair_loss(decoded, prepared, spec)
    assert calls == [71, 73]
    guidance.failure_aware_repair_loss(decoded, prepared, spec)
    assert calls == [71, 73, 71, 73]


@pytest.mark.parametrize("storage", ["_graph", "_raw"])
@pytest.mark.parametrize("field", ["atoms", "charges", "bonds"])
def test_public_loss_rejects_private_label_mutation_after_an_earlier_success(storage, field):
    spec, decoded, targets = _fixture(("invalid",))
    chemistry.repair_loss(decoded, targets, spec)
    value = getattr(getattr(targets.targets[0], storage), field)
    value.reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="identity changed"):
        chemistry.repair_loss(decoded, targets, spec)


@pytest.mark.parametrize("defect", ["duplicate_ids", "negative_id", "misalignment", "extra_row", "wrong_label_dtype"])
def test_public_batch_and_target_validation_is_not_bypassed(defect):
    spec, decoded, targets = _fixture()
    if defect == "duplicate_ids":
        targets = replace(targets, candidate_ids=(71, 71, 73, 74))
    elif defect == "negative_id":
        targets = replace(targets, candidate_ids=(-1, 72, 73, 74))
    elif defect == "misalignment":
        targets = replace(targets, candidate_ids=(99, 72, 73, 74))
    elif defect == "extra_row":
        target = targets.targets[0]
        target._graph.atoms = torch.cat((target._graph.atoms, target._graph.atoms))
    else:
        targets.targets[0]._graph.charges = targets.targets[0]._graph.charges.float()
    with pytest.raises(ValueError):
        chemistry.repair_loss(decoded, targets, spec)


@pytest.mark.parametrize("defect", ["nonfinite", "mismatched_dtype", "shape"])
def test_decoder_validation_still_rejects_each_invalid_call(defect):
    spec, decoded, targets = _fixture()
    if defect == "nonfinite":
        with torch.no_grad():
            decoded.atoms.reshape(-1)[0] = float("nan")
    elif defect == "mismatched_dtype":
        decoded.charges = decoded.charges.float()
    else:
        decoded.bonds = decoded.bonds[:, :-1]
    with pytest.raises(ValueError):
        chemistry.repair_loss(decoded, targets, spec)


@pytest.mark.parametrize("defect", ["source_update", "source_probability", "receipt"])
def test_enclosing_identity_rejects_current_metadata_tampering_after_an_earlier_success(defect):
    spec, decoded, targets = _fixture(("invalid",))
    prepared = guidance.prepare_failure_guidance(decoded, targets, spec)
    guidance.failure_aware_repair_loss(decoded, prepared, spec)
    if defect == "receipt":
        targets.receipt["attempts"][0]["target"]["decisions"] += 1
    else:
        target = targets.targets[0]
        change = ({"source_update": target.source_update + 1} if defect == "source_update" else
                  {"source_probability_sha256": "f" * 64})
        altered = replace(targets, targets=(replace(target, **change),))
        prepared = replace(prepared, base=altered)
    with pytest.raises(ValueError, match="identit"):
        guidance.failure_aware_repair_loss(decoded, prepared, spec)
