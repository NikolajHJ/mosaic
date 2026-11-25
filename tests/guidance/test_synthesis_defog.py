"""Model-free checks of the declared continuous DeFoG prior and lifecycle."""
import hashlib
import io
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from latentmol.guidance import synthesis_defog as module
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec


class ToyCore(torch.nn.Module):
    """Explicitly a deterministic fixture, not a pretrained DeFoG model."""
    def __init__(self):
        super().__init__()
        self.register_buffer("atom_kernel", torch.arange(144).reshape(12, 12).float() / 144)
        self.register_buffer("bond_kernel", torch.arange(25).reshape(5, 5).float() / 25)

    def forward(self, x, e, y, mask):
        atom = x[..., :12] @ self.atom_kernel + 0.02 * x[..., 12:32].sum(-1, keepdim=True)
        atom = atom + y.sum(-1)[:, None, None] * 0.01
        bond = e[..., :5] @ self.bond_kernel + 0.02 * e[..., 5:].sum(-1, keepdim=True)
        return SimpleNamespace(X=atom, E=bond)


def native_graph(nodes=6, *, generator=None):
    x_logits = torch.randn((1, nodes, 12), generator=generator, requires_grad=True)
    e_logits = torch.randn((1, nodes, nodes, 5), generator=generator, requires_grad=True)
    x = x_logits.softmax(-1)
    sym = (e_logits + e_logits.transpose(1, 2)) * 0.5
    e = sym.softmax(-1) * (1 - torch.eye(nodes)[None, :, :, None])
    return (x_logits, e_logits), x, e


def metadata(time=0.95):
    return dict(max_nodes=88, atom_weights=torch.arange(1, 13).float(), max_weight=1000., model_time=time)


def test_complete_feature_widths_and_native_rrwp_powers():
    _, x, e = native_graph(generator=torch.Generator().manual_seed(43))
    node, edge, global_ = module.continuous_features(x, e, **metadata())
    assert node.shape == (1, 6, 34)
    assert edge.shape == (1, 6, 6, 25)
    assert global_.shape == (1, 7)
    a = e[..., 1:].sum(-1)
    transition = a / a.sum(-1, keepdim=True)
    for k in (0, 1, 5, 19):
        expected = torch.linalg.matrix_power(transition, k)
        torch.testing.assert_close(edge[..., 5 + k], expected)
        torch.testing.assert_close(node[..., 12 + k], expected.diagonal(dim1=-2, dim2=-1))


@pytest.mark.parametrize("size,cycle_slot", [(3, 0), (4, 1), (5, 2), (6, 3)])
def test_hard_isolated_cycle_features_preserve_the_native_polynomial_limit(size, cycle_slot):
    adjacency = torch.zeros(1, size, size)
    for i in range(size):
        adjacency[:, i, (i + 1) % size] = 1
        adjacency[:, (i + 1) % size, i] = 1
    expected = torch.zeros(1, 4)
    expected[:, cycle_slot] = 0.1
    torch.testing.assert_close(module._cycles(adjacency), expected, atol=1e-6, rtol=0)


def test_legacy_category_auxiliaries_are_expected_indices_not_new_valence_semantics():
    x = torch.zeros(1, 3, 12)
    x[..., 1] = 1  # native nitrogen index1
    e = torch.zeros(1, 3, 3, 5)
    e[..., 0] = 1
    e[:, 0, 1] = e[:, 1, 0] = torch.tensor([0., 0., 0., 0., 1.])
    e *= 1 - torch.eye(3)[None, :, :, None]
    node, _, _ = module.continuous_features(x, e, **metadata())
    # Original code's aromatic argmax returns category4, not bond order1.5.
    torch.testing.assert_close(node[..., -1], torch.tensor([[4., 4., 0.]]))
    torch.testing.assert_close(node[..., -2], torch.tensor([[-3., -3., 1.]]))


def test_deterministic_attached_atom_and_bond_paths_without_noise_or_weight_writes():
    core = ToyCore().eval().requires_grad_(False)
    leaves, x, e = native_graph(generator=torch.Generator().manual_seed(46))
    rng = torch.random.get_rng_state().clone()
    before = {key: value.clone() for key, value in core.state_dict().items()}
    first, aux = module.consistency_loss(core, x, e, **metadata())
    second, _ = module.consistency_loss(core, x, e, **metadata())
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)
    gradients = torch.autograd.grad(first.sum(), leaves)
    assert first.shape == (1,)
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    for key, value in core.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert aux["atom_predictions"].shape[-1] == 12
    assert all(parameter.grad is None for parameter in core.parameters())


def test_zero_degree_and_fractional_cycle_features_have_finite_backward():
    core = ToyCore().eval().requires_grad_(False)
    x = torch.full((1, 4, 12), 1 / 12, requires_grad=True)
    probability = torch.tensor(1e-10, requires_grad=True)
    e = torch.zeros(1, 4, 4, 5)
    off = 1 - torch.eye(4)[None]
    e[..., 0] = (1 - probability) * off
    e[..., 1] = probability * off
    value, _ = module.consistency_loss(core, x, e, **metadata())
    gradients = torch.autograd.grad(value.sum(), (x, probability))
    assert all(bool(g.isfinite().all()) for g in gradients)
    assert bool((module._cycles(e[..., 1:].sum(-1)) >= 0).all())


def test_incompatible_or_nonfinite_feature_inputs_fail():
    _, x, e = native_graph()
    with pytest.raises(ValueError):
        module.continuous_features(x[..., :11], e, **metadata())
    x = x.clone()
    x[..., 0] = torch.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        module.continuous_features(x, e, **metadata())


def test_lazy_independent_worker_lifecycle_and_explicit_null_charge_direction(tmp_path, monkeypatch):
    weights = tmp_path / "fixture.ckpt"
    weights.write_bytes(b"toy transport fixture, not model weights")
    monkeypatch.setattr(module, "GUACAMOL_SHA256", hashlib.sha256(weights.read_bytes()).hexdigest())
    calls = []

    class ResidentFixture:
        def __init__(self, spec, active, **kwargs):
            self.identity = {"model": "toy", "frozen_state_sha256": "fixture"}
            self.shutdown_status = None
            calls.append(("start", kwargs["command"], kwargs["stderr_path"]))

        def score_at(self, graph, active):
            atom_weights = torch.arange(graph.atoms.shape[-1]).to(graph.atoms)
            bond_weights = torch.arange(graph.bonds.shape[-1]).to(graph.bonds)
            value = (graph.atoms * atom_weights).sum() + (graph.bonds * bond_weights).sum()
            value = value + graph.charges.sum() * 0
            return value, {}

        def close(self):
            calls.append(("close",))
            self.shutdown_status = "graceful"

    monkeypatch.setattr(module, "MosaicProtenixScorer", ResidentFixture)
    spec = GraphSpec(max_nodes=4)
    model = module.FrozenDeFoG(spec, weights, python=sys.executable, source_path=tmp_path)
    other = module.FrozenDeFoG(spec, weights, python=sys.executable, source_path=tmp_path)
    assert calls == [] and model.stderr_path != other.stderr_path
    generator = torch.Generator().manual_seed(128)
    leaves = [torch.randn(shape, generator=generator, requires_grad=True) for shape in
              ((1, 4, 10), (1, 4, 5), (1, 4, 4, 5))]
    active = torch.tensor([True, True, True, False])
    graph = graph_from_decoder(DecoderOutput(*leaves), spec, fixed_active=active)
    result = model(graph, active)
    gradients = torch.autograd.grad(result["difficulty"].sum(), leaves)
    assert result["difficulty"].shape == (1,)
    assert gradients[0].norm() > 0 and gradients[2].norm() > 0
    assert torch.count_nonzero(gradients[1]) == 0
    assert calls[0][1][-1] == "--worker"
    assert model.identity["model"] == "toy"
    model.close()
    model.close()
    other.close()
    assert calls[-1] == ("close",) and model.shutdown_status == "graceful"
    assert not Path(model.stderr_path).parent.exists()
    with pytest.raises(RuntimeError, match="closed"):
        model(graph, active)


def test_worker_transport_negotiates_protocol_on_init_only(monkeypatch):
    """Exercise the real JSON parser with mocked native asset/model loading."""
    from latentmol.guidance.mosaic_protenix import PROTOCOL

    class NativeFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ToyCore()
            self.cfg = SimpleNamespace(model=SimpleNamespace(extra_features="rrwp", rrwp_steps=20))
            self.extra_features = SimpleNamespace(max_n_nodes=88)
            self.domain_features = SimpleNamespace(weight=SimpleNamespace(
                atom_weight_list=torch.arange(1, 13).float(), max_weight=1000.))

    native = NativeFixture().eval().requires_grad_(False)
    before = {name: value.clone() for name, value in native.state_dict().items()}
    loads = []
    loader = ModuleType("latentmol.defog.interface_smoke")
    loader._verify_assets = lambda source, checkpoint: loads.append((source, checkpoint))
    loader._load_native_model = lambda source, checkpoint, device: native
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setenv("LSB_JOBID", "model-free-unit-test")
    # Do not change the test process's thread settings when invoking the worker.
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    arrays = dict(atoms=[[.7, .3], [.2, .8], [.5, .5]], charges=[[1.], [1.], [1.]],
        bonds=[[[1., 0.], [.8, .2], [.6, .4]],
               [[.8, .2], [1., 0.], [.3, .7]],
               [[.6, .4], [.3, .7], [1., 0.]]])
    requests = [dict(op="init", protocol=PROTOCOL, request_id=0, n_atoms=3,
        atomic_numbers=[6, 7], formal_charges=[0], bond_names=["NONE", "SINGLE"],
        config=dict(policy=module.POLICY, source="fixture-source", checkpoint="fixture.ckpt", model_time=.95)),
        # MosaicProtenixScorer sends no protocol field after initialization.
        dict(op="evaluate", request_id=1, arrays=arrays),
        dict(op="evaluate", protocol="wrong-protocol", request_id=2, arrays=arrays),
        dict(op="close", request_id=3)]
    incoming = io.StringIO("".join(json.dumps(request) + "\n" for request in requests))
    outgoing = io.StringIO()
    monkeypatch.setattr(sys, "stdin", incoming)
    monkeypatch.setattr(sys, "stdout", outgoing)

    module._worker()

    responses = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert len(loads) == 1
    assert [response["request_id"] for response in responses] == [0, 1, 2, 3]
    assert all(response["protocol"] == PROTOCOL for response in responses)
    assert [response["status"] for response in responses] == ["ok", "ok", "error", "ok"]
    result = responses[1]
    assert torch.isfinite(torch.tensor(result["loss"]))
    for name, values in arrays.items():
        gradient = torch.tensor(result["gradients"][name])
        assert gradient.shape == torch.tensor(values).shape and bool(gradient.isfinite().all())
        assert gradient.norm() > 0 if name != "charges" else torch.count_nonzero(gradient) == 0
    assert "protocol" in responses[2]["error"].lower()
    assert responses[3]["frozen_unchanged"] is True
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
