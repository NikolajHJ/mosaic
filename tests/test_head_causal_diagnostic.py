"""HPC-only synthetic engineering fixtures; no weights/network/native inference."""
from types import SimpleNamespace
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from latentmol.boltz.head_causal_diagnostic import (HEADS, capture_heads, chain_completeness, cut_vjp,
    fixed_readout, native_bin_centers, selected_cases, cached_controls)


class Affinity(nn.Module):
    def forward(self, x):
        return {"affinity_pred_value": x[0].square().mean(), "affinity_logits_binary": x[1].sin().mean()}


class Distogram(nn.Module):
    def forward(self, x):
        return x[2].sin()


class Confidence(nn.Module):
    def forward(self, x):
        return {"pae_logits": x[3].square(), "pde_logits": x[4].cos(), "plddt_logits": x[5].exp()}


class Toy(nn.Module):
    """Six-channel non-ancestor cut, including distogram-dependent PDE weight."""
    def __init__(self):
        super().__init__()
        self.affinity = Affinity()
        self.structure = nn.Module()
        self.structure.distogram_module = Distogram()
        self.structure.confidence_module = nn.Module()
        self.structure.confidence_module.confidence_heads = Confidence()

    def forward(self, x):
        affinity = self.affinity(x)
        distogram = self.structure.distogram_module(x)
        confidence = self.structure.confidence_module.confidence_heads(x)
        contact = distogram.square().sum()
        binding = (affinity["affinity_pred_value"] - affinity["affinity_logits_binary"] + contact
            + .7 * (distogram * confidence["pde_logits"]).sum()
            + confidence["pae_logits"].sigmoid().sum() + confidence["plddt_logits"].logsumexp(0))
        return contact, binding


def _point():
    return torch.linspace(.1, 1.8, 18).reshape(6, 3).requires_grad_()


def test_complete_cut_preserves_contact_and_pde_distogram_paths():
    model, point = Toy(), _point()
    with capture_heads(model) as heads:
        contact, binding = model(point)
        direct, = torch.autograd.grad(binding, point, retain_graph=True)
        chained, cotangents, unused = cut_vjp(binding, heads, point)
        assert not unused and set(cotangents) == set(HEADS)
        assert chain_completeness(direct, chained)["engineering_chain_complete"]
        expected = 2 * heads["distogram"] + .7 * heads["pde_logits"]
        torch.testing.assert_close(cotangents["distogram"], expected)
        contact_chain, contact_cot, inactive = cut_vjp(contact, heads, point)
        assert set(inactive) == set(HEADS) - {"distogram"}
        assert not torch.equal(cotangents["distogram"], contact_cot["distogram"])
        incomplete = dict(cotangents)
        incomplete["distogram"] = contact_cot["distogram"]  # Deliberately omit PDE weight path.
        wrong, = torch.autograd.grad(tuple(heads[k] for k in HEADS), point,
            grad_outputs=tuple(incomplete[k] for k in HEADS), retain_graph=True)
        assert not chain_completeness(direct, wrong)["engineering_chain_complete"]


def test_frozen_widened_readout_is_zero_at_base_and_has_correct_chain():
    model, point = Toy(), _point()
    with capture_heads(model) as heads:
        _, binding = model(point)
        chained, cotangents, _ = cut_vjp(binding, heads, point)
        base = {key: tensor.detach().clone() for key, tensor in heads.items()}
        scalar, terms = fixed_readout(heads, base, cotangents)
        assert scalar.dtype == torch.float64 and float(scalar) == 0.
        assert all(value == 0. for value in terms.values())
        measured, = torch.autograd.grad(scalar, point)
        assert chain_completeness(chained, measured)["engineering_chain_complete"]
    assert not model.affinity._forward_hooks


def test_head_replay_is_rejected_and_hooks_removed_even_on_error():
    model = Toy()
    with pytest.raises(ValueError, match="unexpectedly recomputed"):
        with capture_heads(model):
            model(_point())
            model(_point())
    assert not model.affinity._forward_hooks
    assert not model.structure.distogram_module._forward_hooks
    assert not model.structure.confidence_module.confidence_heads._forward_hooks


def test_partial_capture_is_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        with capture_heads(Toy()):
            pass


def test_cached_contact_controls_keep_native_bin_partition_and_declare_new_dtype():
    from latentmol.boltz.dbmol_objective import CONFIDENCE_WEIGHTS, contact_loss, reduce_objective
    from latentmol.multimodel.config import Objective
    logits = torch.linspace(-1., 1., 256).reshape(1, 2, 2, 64).requires_grad_()
    pocket, ligand = torch.tensor([True, False]), torch.tensor([False, True])
    centers = native_bin_centers(logits)
    dbmol = SimpleNamespace(distance_cutoff=8., epsilon=1e-8, affinity_regression_weight=1.,
        affinity_logit_weight=1., confidence_weights=CONFIDENCE_WEIGHTS, contact_weight=2.5, confidence_weight=.2)
    primitives = {key: torch.tensor(.3, requires_grad=True) for key in CONFIDENCE_WEIGHTS}
    primitives.update(affinity_pred_value=torch.tensor(.4), affinity_logits_binary=torch.tensor(.7))
    primitives.update(contact_loss(logits, pocket, ligand, bin_centers=centers))
    outputs = reduce_objective(primitives, dbmol)
    config = SimpleNamespace(dbmol=dbmol, objective=Objective(output="dbmol_total_loss", direction="minimize"))
    controls = cached_controls(outputs, {"distogram": logits}, config, pocket, ligand, centers, torch.tensor(.2), .01)
    assert all(value.dtype == torch.float64 for value in controls.values())
    assert torch.equal(centers < 8., centers.double() < 8.)
    gradient, = torch.autograd.grad(controls["contact_float64"], logits)
    assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_two_prespecified_cases_are_selected_without_replacing_failed_directions():
    plan = {"cases": [{"id": name, "step": step} for name, step in [("prior-0", 20), ("prior-3", 50), ("prior-5", 20)]]}
    assert [case["id"] for case in selected_cases(plan)] == ["prior-0", "prior-5"]
    plan["cases"][2]["step"] = 19
    with pytest.raises(ValueError, match="prespecified"):
        selected_cases(plan)


def test_empty_natural_selection_is_only_a_warning_for_fixed_mask_endpoint(monkeypatch):
    import latentmol.boltz.head_causal_diagnostic as diagnostic
    import latentmol.boltz.active_set as active_set
    @contextmanager
    def captured(scorer):
        yield {}
    graph = SimpleNamespace(atoms=torch.ones(1), charges=torch.ones(1), bonds=torch.ones(1))
    terms = {key: torch.tensor(.1) for key in ("contact", "binding", "repair_nll", "combined")}
    monkeypatch.setattr(diagnostic, "capture_heads", captured)
    monkeypatch.setattr(diagnostic, "_forward", lambda *args: (None, graph, {}, terms, {"target_sha256": "fixed"}))
    monkeypatch.setattr(diagnostic, "cached_controls", lambda *args: {})
    def empty(decoded, *, temperature, threshold, require_nonempty):
        assert require_nonempty is False
        return SimpleNamespace(active=torch.tensor([False]))
    monkeypatch.setattr(active_set, "select_slots", empty)
    branch = SimpleNamespace(scorer=SimpleNamespace(last_contact_selection={}), prepared=SimpleNamespace(active=torch.tensor([True])))
    row = diagnostic.endpoint_probe(None, torch.zeros(1), branch, SimpleNamespace(temperature=1., threshold=.5),
                                    None, None, .01, {}, {}, ())
    assert row["natural_selector_crossed"] is True
