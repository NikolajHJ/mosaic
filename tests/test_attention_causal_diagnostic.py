"""Engineering fixtures for explicitly counterfactual diagnostic arithmetic."""

import pytest
import torch

from latentmol.boltz.attention_causal_diagnostic import controlled_attention, whole_mask_rows


def _softmax(x, dim):
    return x.softmax(dim)


def _call(q, k, v, mask, bias, mode):
    return controlled_attention(q, k, v, [mask, bias], mode=mode, softmax=_softmax)[0]


def test_stop_preserves_native_forward_and_value_derivative():
    q = torch.tensor([[[[.2, .4]]]], requires_grad=True)
    k = torch.tensor([[[[.5, -.2], [.1, .7]]]], requires_grad=True)
    v = torch.tensor([[[[1.], [3.]]]], requires_grad=True)
    bias = torch.tensor([[[[.3, -.4]]]], requires_grad=True)
    mask = torch.full((1, 1, 1, 2), -1e9)
    native = _call(q, k, v, mask, bias, "native")
    stop = _call(q, k, v, mask, bias, "stop_masked_logits")
    assert torch.equal(native, stop)
    qg, kg, vg, bg = torch.autograd.grad(stop.sum(), (q, k, v, bias))
    assert qg.count_nonzero() == kg.count_nonzero() == bg.count_nonzero() == 0
    assert torch.equal(vg, torch.full_like(v, .5))


@pytest.mark.parametrize("leading", [(2, 3), (2, 2, 3)])
def test_broadcast_rows_and_one_valid_key_preserve_other_gradients(leading):
    q = torch.ones((*leading, 2, 2), requires_grad=True)
    k = torch.ones((*leading, 3, 2), requires_grad=True)
    v = torch.arange(3.).expand(*leading, 3).unsqueeze(-1).clone().requires_grad_()
    mask = torch.tensor([0., -1e9, -1e9]).reshape(*([1] * len(leading)), 1, 3)
    bias = torch.zeros_like(mask, requires_grad=True)
    assert not whole_mask_rows(mask, q, k).any()
    native = _call(q, k, v, mask, bias, "native")
    stopped = _call(q, k, v, mask, bias, "stop_masked_logits")
    assert torch.equal(native, stopped)
    ng = torch.autograd.grad(native.sum(), (q, k, v, bias), retain_graph=True)
    sg = torch.autograd.grad(stopped.sum(), (q, k, v, bias))
    assert all(torch.equal(a, b) for a, b in zip(ng, sg))


def test_invalid_mask_and_trainable_mask_are_rejected():
    q, k = torch.ones(1, 1, 1, 2), torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="exact 0/-1e9"):
        whole_mask_rows(torch.full((1, 1, 1, 2), -1e6), q, k)
    with pytest.raises(ValueError, match="fixed floating"):
        whole_mask_rows(torch.zeros(1, 1, 1, 2, requires_grad=True), q, k)


def test_float64_control_is_explicitly_changed_forward_for_all_masked_rows():
    q = torch.tensor([[[[.3, .5]]]])
    k = torch.tensor([[[[.7, .2], [-.4, .1]]]])
    v = torch.tensor([[[[1.], [3.]]]])
    mask = torch.full((1, 1, 1, 2), -1e9)
    bias = torch.tensor([[[[.2, -.3]]]])
    native = _call(q, k, v, mask, bias, "native")
    wide = _call(q, k, v, mask, bias, "masked_logits_float64")
    expected = ((q.double() @ k.double().transpose(-1, -2) + mask.double() + bias.double()).softmax(-1).float() @ v)
    assert torch.equal(wide, expected)
    assert not torch.equal(native, wide)


def test_mixed_query_rows_keep_valid_row_qk_and_bias_gradients():
    q = torch.tensor([[[[.2, .4], [.5, -.2]]]], requires_grad=True)
    k = torch.tensor([[[[.5, -.2], [.1, .7], [-.4, .3]]]], requires_grad=True)
    v = torch.tensor([[[[1.], [3.], [2.]]]], requires_grad=True)
    mask = torch.tensor([[[[-1e9, -1e9, -1e9], [0., 0., -1e9]]]])
    bias = torch.tensor([[[[.1, -.2, .3], [.3, .4, -.1]]]], requires_grad=True)
    native = _call(q, k, v, mask, bias, "native")
    stopped = _call(q, k, v, mask, bias, "stop_masked_logits")
    assert torch.equal(native, stopped)
    ng = torch.autograd.grad(native[..., 1, :].sum(), (q, k, v, bias), retain_graph=True)
    sg = torch.autograd.grad(stopped[..., 1, :].sum(), (q, k, v, bias))
    assert all(torch.equal(a, b) for a, b in zip(ng, sg))
    assert ng[0][..., 1, :].count_nonzero() > 0
