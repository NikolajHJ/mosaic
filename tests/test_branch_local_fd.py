"""HPC-only fixtures for branch-aware derivative validation and fail-closed AD."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentmol.boltz.branch_local_fd import AffinityReadoutBranches, validate_derivative


def readout():
    head = nn.Module()
    for field in ('affinity_out_mlp', 'to_affinity_pred_value', 'to_affinity_pred_score'):
        layers = [nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU()]
        if field != 'affinity_out_mlp': layers += [nn.Linear(2, 2)]
        setattr(head, field, nn.Sequential(*layers))
    return head.eval().requires_grad_(False)


def fixture():
    heads = (readout(), readout())
    model = SimpleNamespace(affinity_ensemble=True,
        affinity_module1=SimpleNamespace(affinity_heads=heads[0]),
        affinity_module2=SimpleNamespace(affinity_heads=heads[1]))
    return SimpleNamespace(affinity=SimpleNamespace(model=model)), heads


def evaluate(heads, value):
    return sum(head.to_affinity_pred_value(head.affinity_out_mlp(value)).sum()
               + head.to_affinity_pred_score(head.affinity_out_mlp(value)).sum() for head in heads)


def once(heads, value):
    # Native code invokes the shared block once before the two output branches.
    total = 0
    for head in heads:
        hidden = head.affinity_out_mlp(value)
        total = total + head.to_affinity_pred_value(hidden).sum() + head.to_affinity_pred_score(hidden).sum()
    return total


def test_readout_branch_has_exact_baseline_scalar_and_gradient_and_hooks_restore():
    scorer, heads = fixture()
    branches = AffinityReadoutBranches(scorer)
    value = torch.tensor([[.2, -.3]], requires_grad=True)
    original = once(heads, value)
    gradient, = torch.autograd.grad(original, value)
    with branches.capture():
        captured = once(heads, value)
    with branches.hold():
        held = once(heads, value)
        held_grad, = torch.autograd.grad(held, value)
    assert torch.equal(original, captured) and torch.equal(original, held)
    assert torch.equal(gradient, held_grad)
    assert len(branches.masks) == 12
    with pytest.raises(RuntimeError, match='fixture exception'):
        with branches.hold():
            raise RuntimeError('fixture exception')
    assert all(not layer._forward_hooks for layer in branches.layers.values())


def test_capture_rejects_incomplete_or_reused_head_calls_and_restores_hooks():
    scorer, heads = fixture()
    branches = AffinityReadoutBranches(scorer)
    with pytest.raises(ValueError, match='Incomplete'):
        with branches.capture(): pass
    assert all(not layer._forward_hooks for layer in branches.layers.values())
    branches = AffinityReadoutBranches(scorer)
    with pytest.raises(ValueError, match='one native FP32'):
        with branches.capture(): evaluate(heads, torch.ones(1, 2))
    assert all(not layer._forward_hooks for layer in branches.layers.values())


@pytest.mark.parametrize('wrong_jvp', [False, True])
def test_crossing_secant_is_retained_but_only_verified_local_branch_can_pass(wrong_jvp):
    point = torch.zeros(2, dtype=torch.float64)
    gradient = torch.tensor([2., 0.], dtype=torch.float64)
    baseline = 1e-7
    def natural(candidate):
        return {'loss': float(torch.relu(candidate[0] + baseline) + candidate[0]), 'loss_dtype': 'torch.float64'}
    def held(candidate):
        return {'loss': float(2 * candidate[0] + baseline), 'loss_dtype': 'torch.float64'}
    def forward(direction):
        # Deliberately wrong forward tangent must not be hidden by perfect FD.
        tangent = float((gradient * direction).sum()) * (1.5 if wrong_jvp else 1.)
        return {'fixture': {'primal': baseline, 'tangent': tangent}}
    report = {}
    validate_derivative(point, gradient, natural, held, forward, baseline=baseline,
        key='fixture', seed=17, rtol=.05, atol=1e-7, minimum_passes=2,
        report=report, persist=lambda: None)
    assert report['natural_secants']['passed'] is False
    assert report['local_branch_fd']['passed'] is True
    assert report['passed'] is (not wrong_jvp)
    assert len(report['independent_forward_ad']['rows']) == 4
    assert report['policy']['native_model_during_optimization'] == 'unmodified ReLU branches'


def test_changed_forward_primal_fails_even_with_matching_derivatives():
    point, gradient = torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    score = lambda p: {'loss': float(p.sum()), 'loss_dtype': 'torch.float64'}
    report = {}
    validate_derivative(point, gradient, score, score,
        lambda d: {'fixture': {'primal': 1., 'tangent': float(d.sum())}},
        baseline=0., key='fixture', seed=17, rtol=.05, atol=1e-7,
        minimum_passes=2, report=report, persist=lambda: None)
    assert report['passed'] is False
    assert report['local_branch_fd']['passed'] is True
    assert all(not row['primal_exact'] for row in report['independent_forward_ad']['rows'])


def test_reader_rejects_stale_forward_ad_aggregate():
    from latentmol.repair_guidance.integration_smoke import verified_forward_ad
    item = {'baseline_loss': 0., 'independent_forward_ad': {'passed': True, 'rtol': 1e-3, 'atol': 1e-6,
        'rows': [{'direction': label, 'primal': 0., 'tangent': 1., 'reverse_projection': 1.,
                  'absolute_error': 0., 'tolerance': 1e-6 + 1e-3, 'passed': True, 'primal_exact': True}
                 for label in ('gradient', 'random_1', 'random_2', 'random_3')]}}
    assert verified_forward_ad(item)
    item['independent_forward_ad']['rows'][2]['passed'] = False
    assert not verified_forward_ad(item)
