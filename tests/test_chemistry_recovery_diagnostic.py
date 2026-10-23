"""Allocated-only contracts for a disposable, nonpromotable mechanism probe."""
from copy import deepcopy
import os

import pytest
import torch
from torch import nn

from latentmol import chemistry_recovery_diagnostic as diagnostic
from latentmol.chemistry_execution_validation import _restore, _snapshot
from latentmol.chemistry_continuation import nested_equal
from latentmol.train import learning_rate_at_epoch


def _plan():
    return dict(starting_checkpoint=dict(epoch=36), recovery_diagnostic=dict(
        version=diagnostic.VERSION, candidate_scales=[1., .5, .25, .125], panel_count=1024,
        maximum_updates=1, prefix_updates=8, prefix_streams=["benchmark", "production"]))


def test_scope_is_closed_and_does_not_mutate_plan():
    plan = _plan()
    result = diagnostic.validate_policy(plan)
    result["candidate_scales"][0] = 9.
    assert plan["recovery_diagnostic"]["candidate_scales"][0] == 1.


@pytest.mark.parametrize("key,value", [("maximum_updates", 8), ("maximum_updates", True),
    ("prefix_updates", 9), ("panel_count", 256), ("candidate_scales", [1, .5, .25, .125]),
    ("prefix_streams", ["production", "benchmark"]), ("version", "unversioned")])
def test_undeclared_diagnostic_budget_or_stream_is_rejected(key, value):
    plan = _plan()
    plan["recovery_diagnostic"][key] = value
    with pytest.raises(ValueError):
        diagnostic.validate_policy(plan)


def test_diagnostic_requires_completed_epoch36():
    plan = _plan()
    plan["starting_checkpoint"]["epoch"] = 37
    with pytest.raises(ValueError, match="epoch36"):
        diagnostic.validate_policy(plan)


class SmallModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False, device=device, dtype=torch.float64)
        self.decoder = nn.Linear(2, 1, bias=False, device=device, dtype=torch.float64)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.tensor([[.5, -.2], [.1, .7]], device=device))
            self.decoder.weight.copy_(torch.tensor([[.4, -.3]], device=device))


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES") == "1":
            pytest.fail("Mandatory native recovery CUDA fixture has no allocated CUDA device")
        pytest.skip("CUDA-only allocated fixture")
    return request.param


def _backward_branch(model, branch, rows, weight):
    z = model.encoder(rows) if branch == "real" else rows
    value = model.decoder(z)
    # The production function also takes autograd.grad probes before backward.
    # A parameter .grad hook would count them twice; post-accumulate must not.
    torch.autograd.grad(value.square().sum(), tuple(model.decoder.parameters()), retain_graph=True)
    (value.square().sum() * weight).backward()


def test_branch_vjps_exclude_autograd_grad_probes_and_sum_to_full_update(device):
    model = SmallModel(device)
    real = torch.tensor([[.2, .7], [.9, -.3]], device=device, dtype=torch.float64)
    prior = torch.tensor([[1., -.4], [.8, .5], [-.2, .3], [.7, -.8]], device=device, dtype=torch.float64)
    model.zero_grad(set_to_none=True)
    with diagnostic.BranchGradientCapture(model, 2, 4, 1) as capture:
        # No-grad teacher prepasses are not part of the streamed VJP schedule.
        with torch.no_grad():
            model.decoder(real)
        for row in real:
            _backward_branch(model, "real", row.unsqueeze(0), .25)
        for row in prior:
            _backward_branch(model, "prior", row.unsqueeze(0), .75)
    full = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    assert capture.verify(full)["passed"]
    assert "encoder.weight" in capture.values["real"]
    assert "encoder.weight" not in capture.values["prior"]
    independent = SmallModel(device)
    independent.zero_grad(set_to_none=True)
    for row in real:
        _backward_branch(independent, "real", row.unsqueeze(0), .25)
    assert all(torch.allclose(capture.values["real"][name], parameter.grad, rtol=1e-12, atol=1e-12)
               for name, parameter in independent.named_parameters())
    assert not model.decoder._forward_hooks
    assert all(not parameter._post_accumulate_grad_hooks for parameter in model.parameters())


def test_gradient_dot_displacement_can_detect_inherited_adam_momentum_going_uphill(device):
    parameter = nn.Parameter(torch.tensor([1.], device=device, dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=.01, weight_decay=0.)
    # Old negative gradients build momentum. A new positive gradient need not
    # reverse AdamW's displacement immediately, despite a finite correct VJP.
    for _ in range(4):
        parameter.grad = torch.tensor([-1.], device=device, dtype=torch.float64)
        optimizer.step()
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([.01], device=device, dtype=torch.float64)
    gradient = parameter.grad.clone()
    optimizer.step()
    values = diagnostic.gradient_displacement({"p": gradient}, {"p": parameter.detach() - before})
    assert values["gradient_dot_displacement"] > 0
    assert values["cosine"] > 0


def test_grouped_dots_expose_decoder_uphill_even_if_full_dot_is_downhill():
    gradients = {"encoder.x": torch.tensor([10.]), "decoder.x": torch.tensor([1.])}
    delta = {"encoder.x": torch.tensor([-1.]), "decoder.x": torch.tensor([1.])}
    assert diagnostic.gradient_displacement(gradients, delta)["gradient_dot_displacement"] == -9.
    grouped = diagnostic.grouped_gradient_displacement(gradients, delta)
    assert grouped["encoder"]["gradient_dot_displacement"] == -10.
    assert grouped["decoder"]["gradient_dot_displacement"] == 1.


def test_scaled_proposal_does_not_step_adam_or_modify_snapshot(device):
    model = SmallModel(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    initial = deepcopy(model.state_dict())
    retained = deepcopy(initial)
    before_optimizer = deepcopy(optimizer.state_dict())
    displacement = {name: torch.full_like(parameter, .02) for name, parameter in model.named_parameters()}
    proposal = {name: value + displacement[name] for name, value in initial.items()}
    for scale in diagnostic.SCALES:
        diagnostic.apply_displacement(model, initial, displacement, scale, proposal_state=proposal)
        assert all(torch.allclose(parameter, initial[name] + scale * displacement[name], rtol=0., atol=1e-15)
                   for name, parameter in model.named_parameters())
        assert nested_equal(optimizer.state_dict(), before_optimizer)
        assert nested_equal(initial, retained)


def test_full_scale_keeps_exact_native_weights_under_fp32_cancellation():
    model = nn.Linear(1, 1, bias=False)
    initial = {"weight": torch.tensor([[1.]], dtype=torch.float32)}
    proposal = {"weight": torch.tensor([[1e-8]], dtype=torch.float32)}
    rounded_delta = {"weight": proposal["weight"] - initial["weight"]}
    assert not torch.equal(initial["weight"] + rounded_delta["weight"], proposal["weight"])
    diagnostic.apply_displacement(model, initial, rounded_delta, 1., proposal_state=proposal)
    assert torch.equal(model.weight, proposal["weight"])
    with pytest.raises(ValueError, match="exact saved"):
        diagnostic.apply_displacement(model, initial, rounded_delta, 1.)


def test_full_snapshot_restores_gradients_modes_and_every_rng_after_adam_proposal(device):
    model, reference = SmallModel(device), SmallModel(device)
    reference.eval().requires_grad_(False)
    model.train()
    model.decoder.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .01)
    optimizer.step()
    generators = {"train": torch.Generator().manual_seed(9), "fresh_prior": torch.Generator().manual_seed(11)}
    before = _snapshot(model, reference, optimizer, generators)
    model.eval()
    reference.train()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, -.03)
    optimizer.step()
    torch.randn(4, generator=generators["train"])
    torch.randn(4, generator=generators["fresh_prior"])
    torch.randn(4, device=device)
    _restore(before, model, reference, optimizer, generators)
    assert nested_equal(before, _snapshot(model, reference, optimizer, generators))
    # Restore uses deep copies: later native steps cannot corrupt this receipt.
    retained = deepcopy(before)
    optimizer.step()
    assert nested_equal(before, retained)


def test_drift_excludes_padding_charge_diagonal_and_reversed_bond():
    atoms = torch.tensor([[[0., 1.], [1., 0.], [0., 1.]]])
    charges = torch.tensor([[[1., 0.], [1., 0.], [1., 0.]]])
    bonds = torch.zeros(1, 3, 3, 2)
    bonds[..., 0] = 1.
    before = dict(atoms=atoms, charges=charges, bonds=bonds)
    after = {key: value.clone() for key, value in before.items()}
    after["charges"][0, 1] = torch.tensor([0., 1.])  # Inactive starting slot.
    after["bonds"][0, 0, 0] = torch.tensor([0., 1.])  # Diagonal.
    after["bonds"][0, 2, 0] = torch.tensor([0., 1.])  # Reversed duplicate.
    result = diagnostic.output_drift(before, after)
    assert result["charges"]["argmax_label_flips"] == 0
    assert result["bonds"]["argmax_label_flips"] == 0
    assert result["bonds"]["categorical_vectors"] == 1
    after["bonds"][0, 0, 2] = torch.tensor([0., 1.])
    result = diagnostic.output_drift(before, after)
    assert result["bonds"]["argmax_label_flips"] == 1
    assert result["bonds"]["mean_total_variation"] == 1.


def test_parent_schedule_is_constant_between_epochs36_and37():
    # Matches the authenticated configuration; the benchmark's missing explicit
    # epoch assignment cannot explain source128's drop at this phase boundary.
    from types import SimpleNamespace
    config = SimpleNamespace(learning_rate=.001, lr_milestones=((61, .0003), (81, .0001)))
    assert learning_rate_at_epoch(config, 36) == learning_rate_at_epoch(config, 37) == .001


def test_fixed_coordinate_transitions_keep_raw_chemistry_separate_and_examples_bounded():
    from types import SimpleNamespace
    old, new, labels, changed = [], [], [], []
    for index in range(6):
        old.append(dict(candidate_id=index, classification=dict(smiles_parse_valid=True,
            failure_kind="none", reason=None)))
        new.append(dict(candidate_id=index, classification=dict(smiles_parse_valid=False,
            failure_kind="chemical", reason="sanitization")))
        label = dict(atoms=torch.tensor([1, 1]), charges=torch.tensor([0, 0]),
                     bonds=torch.tensor([[0, 1], [1, 0]]))
        altered = {key: value.clone() for key, value in label.items()}
        altered["bonds"][:] = 0
        labels.append(label)
        changed.append(altered)
    # A schema roundtrip rejection still parses and remains raw-valid.
    new[-1]["classification"].update(smiles_parse_valid=True, failure_kind="representation",
                                     reason="canonical_identity_mismatch")
    result = diagnostic.chemical_transitions(SimpleNamespace(records=old, labels=labels),
        SimpleNamespace(records=new, labels=changed))
    assert result["attempts"] == 6
    assert result["counts"] == dict(valid_to_invalid=5, valid_to_valid=1)
    assert result["candidate_ids"]["valid_to_invalid"] == list(range(5))
    assert result["endpoint_reasons"]["valid_to_invalid"] == {"chemical:sanitization": 5}
    assert result["endpoint_reasons"]["valid_to_valid"] == {"representation:canonical_identity_mismatch": 1}
    examples = result["bounded_changed_examples"]["valid_to_invalid"]
    assert len(examples) == 4
    assert examples[0]["categorical_label_changes"]["bonds"] == [dict(index=[0, 1], before=1, after=0)]


def test_coordinate_reordering_or_missing_labels_cannot_create_transition_evidence():
    from types import SimpleNamespace
    left = SimpleNamespace(records=[dict(candidate_id=0)], labels=[{}])
    right = SimpleNamespace(records=[dict(candidate_id=1)], labels=[{}])
    with pytest.raises(ValueError, match="identities"):
        diagnostic.chemical_transitions(left, right)
    right.labels = []
    with pytest.raises(ValueError, match="correspondence"):
        diagnostic.chemical_transitions(left, right)
