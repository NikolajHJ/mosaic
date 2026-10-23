"""Shared-molecule external-branch composition, executed only on HPC."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from latentmol.representation import GraphSpec, RelaxedGraph
from latentmol.scoring import (BranchCapabilities, BranchManifest, InputSupport,
                               evaluate_branches)


class MockBranch(nn.Module):
    """A boundary fixture, not a trained molecular score or substitute objective."""

    def __init__(self, spec, coefficient, output_name, support=None):
        super().__init__()
        self.spec = spec
        self.coefficient = nn.Parameter(torch.tensor(float(coefficient)), requires_grad=False)
        self.output_name, self.seen = output_name, []
        self.support = support or InputSupport("exact")
        self.manifest = BranchManifest(
            name=output_name, graph_spec_sha256=spec.fingerprint(),
            model_source_version="mock-v1", model_weights_sha256="a" * 64,
            adapter_version="mock-v1", adapter_source_sha256="b" * 64,
            capabilities=BranchCapabilities(
                input_kind="relaxed_graph", occupancy="variable", gradient_mode="pathwise",
                differentiable_channels=("atoms",), supported_atomic_numbers=(6,),
                supported_formal_charges=(0,), supported_bond_names=("NONE",),
                context_policy="No native context in mock", relaxation_policy="Mock tensor arithmetic"))
        self.eval()

    def assess(self, graph):
        return self.support

    def forward(self, graph):
        self.seen.append(graph)
        return {self.output_name: self.coefficient * graph.atoms[..., 1].sum(-1)}


def fixture():
    spec = GraphSpec(max_nodes=1, atomic_numbers=(0, 6), formal_charges=(0,), bond_names=("NONE",))
    z = torch.tensor([[0.4]], requires_grad=True)
    graph = RelaxedGraph(torch.stack((1 - z, z), -1), torch.ones(1, 1, 1), torch.ones(1, 1, 1, 1))
    return spec, z, graph


def test_two_frozen_branches_share_one_graph_and_one_latent_backward():
    spec, z, graph = fixture()
    first, second = MockBranch(spec, 2, "affinity"), MockBranch(spec, 3, "liability")
    outputs = evaluate_branches(graph, {"target": first, "off_target": second}, spec=spec)
    assert first.seen[0] is second.seen[0] is graph
    assert set(outputs) == {"target", "off_target"}
    assert set(outputs["target"]) == {"affinity"}
    assert set(outputs["off_target"]) == {"liability"}
    # Weights and signs are caller-owned. Both graph paths contribute to z.
    loss = -0.5 * outputs["target"]["affinity"].sum() + 2 * outputs["off_target"]["liability"].sum()
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([[5.0]]))
    assert first.coefficient.grad is second.coefficient.grad is None
    assert not first.coefficient.requires_grad and not second.coefficient.requires_grad


@pytest.mark.parametrize("kind", ["schema", "unsupported", "lossy", "hard_only", "training", "trainable"])
def test_invalid_composition_rejected_before_any_oracle_runs(kind):
    spec, _, graph = fixture()
    first, second = MockBranch(spec, 2, "a"), MockBranch(spec, 3, "b")
    if kind == "schema":
        second.spec = replace(spec, atomic_numbers=(0, 7))
    elif kind in ("unsupported", "lossy"):
        second.support = InputSupport(kind, ("Fixture eligibility restriction",))
    elif kind == "hard_only":
        second.manifest = replace(second.manifest, capabilities=replace(
            second.manifest.capabilities, input_kind="hard_graph", gradient_mode="none",
            differentiable_channels=()))
    elif kind == "training":
        second.train()
    else:
        second.coefficient.requires_grad_(True)
    with pytest.raises(ValueError):
        evaluate_branches(graph, {"a": first, "b": second}, spec=spec)
    assert not first.seen and not second.seen


def test_lossy_eligibility_requires_explicit_opt_in_and_remains_declared():
    spec, _, graph = fixture()
    branch = MockBranch(spec, 2, "value", InputSupport("lossy", ("Fixture drops information",)))
    outputs = evaluate_branches(graph, {"fixture": branch}, spec=spec, allow_lossy=True)
    assert branch.assess(graph).status == "lossy"
    assert outputs["fixture"]["value"].requires_grad


def test_plain_callable_cannot_hide_a_trainable_model_from_frozen_guard():
    spec, _, graph = fixture()

    class PlainWrapper:
        def __init__(self):
            self.inner = MockBranch(spec, 2, "value")
            self.inner.coefficient.requires_grad_(True)
            self.spec, self.manifest = self.inner.spec, self.inner.manifest

        def assess(self, value):
            return self.inner.assess(value)

        def __call__(self, value):
            return self.inner(value)

    wrapper = PlainWrapper()
    with pytest.raises(ValueError, match="nn.Module"):
        evaluate_branches(graph, {"hidden": wrapper}, spec=spec)
    assert not wrapper.inner.seen
