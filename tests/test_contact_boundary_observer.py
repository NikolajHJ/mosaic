"""Allocated CPU fixtures for complete contact cuts and ReLU counterfactuals.

These small synthetic fixtures verify instrumentation, not pretrained Boltz
derivative accuracy. They must run on DTU HPC and have not been run locally.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


SPEC = spec_from_file_location("_contact_boundary_fixture",
    Path(__file__).parents[1] / "src/latentmol/boltz/contact_boundary_observer.py")
observer = module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


class AtomEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_c, self.relu_p = nn.ReLU(), nn.ReLU()
        self.callback = lambda value: value

    def forward(self, features):
        c = self.relu_c(features["ref_element"] + .3 * features["ref_charge"])
        q = c  # Deliberate alias, as in the native initial atom encoder.
        p = self.relu_p(.4 * c + .1)
        return q, c, p, self.callback


class AtomAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, q, c, p, callback):
        return self.relu(q + .7 * c + .3 * p), q, c, callback


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_encoder, self.atom_attention_encoder = AtomEncoder(), AtomAttention()

    def forward(self, features, affinity=False):
        q, c, p, callback = self.atom_encoder(features)
        a, _, _, _ = self.atom_attention_encoder(q, c, p, callback)
        return a + .05


class Block(nn.Module):
    def forward(self, s, z):
        return s + .1 * z.sin(), z + .2 * s.tanh() + .1 * z.sin()


class Structure(nn.Module):
    def __init__(self, depth=4, *, rematerialize=False):
        super().__init__()
        self.input_embedder = Embedder()
        self.pairformer_module = nn.Module()
        self.pairformer_module.layers = nn.ModuleList(Block() for _ in range(depth))
        self.distogram_module = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.distogram_module.weight.copy_(torch.tensor([
                [.3, -.2, .1], [.2, .4, -.1], [-.1, .2, .3], [.4, .1, .2]]))
        if rematerialize:
            # Same non-reentrant policy as production: call the captured bound
            # forward inside checkpoint, so OUTER module hooks run only once.
            encoder = self.input_embedder.atom_encoder
            original = encoder.forward

            def encoder_forward(*args, **kwargs):
                if not torch.is_grad_enabled():
                    return original(*args, **kwargs)
                return checkpoint(original, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)
            encoder.forward = encoder_forward
            for block in self.pairformer_module.layers:
                original_block = block.forward

                def block_forward(*args, original_block=original_block, **kwargs):
                    if not torch.is_grad_enabled():
                        return original_block(*args, **kwargs)
                    return checkpoint(original_block, *args, use_reentrant=False,
                                      preserve_rng_state=True, **kwargs)
                block.forward = block_forward
        self.eval().requires_grad_(False)

    def forward(self, features, bond_types):
        inputs = self.input_embedder(features)
        s = inputs
        # Both bond routes bypass atom/attention/embedder frontiers.
        z = .4 * inputs + .5 * bond_types + .2 * features["token_bonds"]
        for block in self.pairformer_module.layers:
            s, z = block(s, z)
        return self.distogram_module(z)


def forward(model, point, capture=None):
    def save(name, value):
        return capture.save(name, value) if capture is not None else value
    atoms = save("graph.atoms", point * .3 + torch.tensor([.7, -.4, .2]))
    charges = save("graph.charges", point * .2 + .1)
    bonds = save("graph.bonds", point * .4 + .2)
    features = {"ref_element": save("adapter.ref_element", atoms * .8),
                "ref_charge": save("adapter.ref_charge", charges * .7),
                "token_bonds": save("adapter.token_bonds", bonds * .9)}
    bond_types = save("adapter.bond_types", bonds * .6)
    result = model(features, bond_types)
    return (result.square() * torch.tensor([.2, .3, .4, .1])).sum()


def cotangents(loss, values):
    gradients = torch.autograd.grad(loss, tuple(values), retain_graph=True, allow_unused=True)
    return tuple(torch.zeros_like(value) if gradient is None else gradient
                 for value, gradient in zip(values, gradients))


@pytest.mark.parametrize("rematerialize", [False, True])
def test_sibling_frontiers_alias_and_bypasses_preserve_vjp(rematerialize):
    model = Structure(rematerialize=rematerialize)
    point = torch.tensor([.3, -.2, .1], requires_grad=True)
    ordinary = forward(model, point)
    ordinary_grad, = torch.autograd.grad(ordinary, point)
    for cut in observer.cut_layout(4):
        point = point.detach().clone().requires_grad_()
        with observer.ContactCapture(model) as capture:
            measured = forward(model, point, capture)
            capture.require_complete()
            assert torch.equal(measured, ordinary)
            direct, = torch.autograd.grad(measured, point, retain_graph=True)
            frontier = tuple(capture.values[name] for name in capture.cuts[cut])
            weights = cotangents(measured, frontier)
            chain, = torch.autograd.grad(frontier, point, grad_outputs=weights, retain_graph=True)
            torch.testing.assert_close(direct, ordinary_grad, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(chain, direct, rtol=1e-6, atol=1e-7)
            assert capture.values["atom_encoder.q"] is not capture.values["atom_encoder.c"]
            assert torch.equal(capture.values["atom_encoder.q"], capture.values["atom_encoder.c"])
            assert set(capture.relus) == set(capture.relu_modules)
        assert all(not module._forward_hooks and not module._forward_pre_hooks
                   for module in model.modules())


def test_missing_bond_bypass_cannot_reproduce_complete_chain():
    model = Structure()
    point = torch.tensor([.3, -.2, .1], requires_grad=True)
    with observer.ContactCapture(model) as capture:
        loss = forward(model, point, capture)
        capture.require_complete()
        direct, = torch.autograd.grad(loss, point, retain_graph=True)
        incomplete = (capture.values["embedder.trunk"],)
        missing, = torch.autograd.grad(incomplete, point,
                                      grad_outputs=cotangents(loss, incomplete), retain_graph=True)
        assert not torch.allclose(missing, direct, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("rematerialize", [False, True])
def test_held_relu_baseline_value_and_gradient_parity_and_crossings(rematerialize):
    model = Structure(rematerialize=rematerialize)
    point = torch.tensor([.3, -.2, .1], requires_grad=True)
    with observer.ContactCapture(model) as baseline:
        value = forward(model, point, baseline)
        baseline.require_complete()
        gradient, = torch.autograd.grad(value, point)
        masks = baseline.relu_masks()
    point = point.detach().clone().requires_grad_()
    with observer.ContactCapture(model, held_masks=masks) as held:
        held_value = forward(model, point, held)
        held.require_complete()
        held_gradient, = torch.autograd.grad(held_value, point)
    assert torch.equal(held_value, value)
    torch.testing.assert_close(held_gradient, gradient, rtol=1e-6, atol=1e-7)
    crossing = observer.relu_crossings(baseline.relus, held.relus)
    assert all(row["crossed"] == 0 for row in crossing.values())
    # Changed signs remain visible even when the diagnostic holds old branches.
    endpoint = {name: value.neg() for name, value in baseline.relus.items()}
    crossing = observer.relu_crossings(baseline.relus, endpoint)
    assert any(row["crossed"] > 0 for row in crossing.values())


def test_cleanup_after_forward_error_and_membership_rejection():
    model = Structure()
    with pytest.raises(RuntimeError, match="deliberate"):
        with observer.ContactCapture(model):
            raise RuntimeError("deliberate forward failure")
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())
    with pytest.raises(ValueError, match="membership"):
        with observer.ContactCapture(model) as capture:
            capture.require_complete()
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())


def test_duplicate_embedding_and_affinity_calls_rejected():
    model = Structure()
    feats = {"ref_element": torch.ones(3), "ref_charge": torch.ones(3)}
    with observer.ContactCapture(model):
        model.input_embedder(feats)
        with pytest.raises(ValueError, match="Repeated"):
            model.input_embedder(feats)
    with observer.ContactCapture(model):
        with pytest.raises(ValueError, match="non-affinity"):
            model.input_embedder(feats, affinity=True)


def test_mask_membership_dtype_shape_and_frozen_requirements():
    model = Structure()
    with pytest.raises(ValueError, match="membership"):
        observer.ContactCapture(model, held_masks={})
    names = list(observer.ContactCapture(model).relu_modules)
    with pytest.raises(ValueError, match="bool"):
        observer.ContactCapture(model, held_masks={name: torch.ones(3) for name in names})
    with observer.ContactCapture(model, held_masks={name: torch.ones(1, dtype=torch.bool) for name in names}):
        with pytest.raises(ValueError, match="shape/device"):
            forward(model, torch.zeros(3))
    model.train()
    with pytest.raises(ValueError, match="frozen"):
        observer.ContactCapture(model)
    model.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        observer.ContactCapture(model)


def test_layout_and_recursive_clone_contracts():
    assert observer.selected_indices(64) == [15, 31, 47, 63]
    assert observer.selected_indices(1) == [0]
    for invalid in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            observer.selected_indices(invalid)
    x = torch.tensor([.2], requires_grad=True)
    callback = SimpleNamespace()
    cloned = observer.tensor_clones({"tuple": (x, callback), "list": [x]}, detached=True)
    assert cloned["tuple"][0] is not x and not cloned["tuple"][0].requires_grad
    assert cloned["tuple"][1] is callback
    assert cloned["list"][0] is not cloned["tuple"][0]


def test_native_tuple_callbacks_and_held_zero_derivatives():
    model = Structure()
    capture = observer.ContactCapture(model)
    capture.embedding_active = True
    q = torch.ones(3, requires_grad=True)
    callback = model.input_embedder.atom_encoder.callback
    encoded = capture._atom_encoder(None, (), (q, q, q * .3, callback))
    assert encoded[3] is callback
    attended = capture._atom_attention(None, (), (q, encoded[0], encoded[1], callback))
    assert attended[1:] == (encoded[0], encoded[1], callback)
    with pytest.raises(ValueError, match="call/output"):
        capture._atom_attention(None, (), (q, q, callback))
    value = torch.tensor([-2., 0., 3.], requires_grad=True)
    name = next(iter(capture.relu_modules))
    capture.held_masks = {name: value.detach() > 0}
    result = capture._relu_held(name, None, (value,), None)
    assert torch.equal(result, torch.relu(value))
    assert not bool(torch.signbit(result[:2]).any())
    gradient, = torch.autograd.grad(result.sum(), value)
    assert torch.equal(gradient, torch.tensor([0., 0., 1.]))
