"""Allocated CPU fixtures for structural-frontier/cached native reductions.

Small synthetic graphs certify observer engineering only. Native-head fixtures
use the installed frozen implementation, without large checkpoints or models.
All execution belongs on DTU HPC; these fixtures were not run locally.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentmol.boltz.dbmol_model import strict_confidence_scores
from latentmol.boltz.dbmol_objective import (CONTACT64_ARITHMETIC, NATIVE_ARITHMETIC,
    contact_loss, native_bin_centers, pocket_token_mask, reduce_profile)
from test_dbmol_objective import profile


SPEC = importlib.util.spec_from_file_location("_structural_fixture",
    Path(__file__).parents[1] / "src/latentmol/boltz/structural_loss_diagnostic.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def geometry():
    # Two protein residues with four atoms each, then a three-atom ligand.
    # The chain axes and ligand frames satisfy the production native guards.
    xyz = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [3., 0., 0.], [4., 0., 0.], [3., 1., 0.], [3., 0., 1.],
                         [1., 2., 0.], [2., 2., 0.], [1., 3., 0.]]])
    mapping = torch.zeros(1, 11, 5)
    mapping[0, :4, 0] = mapping[0, 4:8, 1] = 1
    for atom, token in zip(range(8, 11), range(2, 5)):
        mapping[0, atom, token] = 1
    rep = torch.zeros(1, 5, 11)
    for token, atom in enumerate((0, 4, 8, 9, 10)):
        rep[0, token, atom] = 1
    feats = {"token_pad_mask": torch.ones(1, 5), "atom_pad_mask": torch.ones(1, 11),
        "atom_to_token": mapping, "token_to_rep_atom": rep,
        "mol_type": torch.tensor([[0, 0, 3, 3, 3]]), "asym_id": torch.tensor([[0, 0, 1, 1, 1]]),
        "frames_idx": torch.tensor([[[[0, 1, 2], [4, 5, 6], [8, 9, 10], [8, 9, 10], [8, 9, 10]]]]),
        "residue_index": torch.tensor([[0, 1, 0, 1, 2]]), "affinity_token_mask": torch.tensor([[0, 0, 1, 1, 1]])}
    representative = torch.bmm(rep, xyz)
    return {"feats": feats, "x_pred": xyz, "d": torch.cdist(representative, representative)}


class Producer(nn.Module):
    """Native small heads fed by a shared synthetic source, with no affinity."""
    def __init__(self, *, separate, token):
        super().__init__()
        native = pytest.importorskip("boltz.model.modules.confidencev2")
        torch.manual_seed(71)
        self.distogram_module = nn.Linear(3, 64)
        self.confidence_module = nn.Module()
        self.confidence_module.confidence_heads = native.ConfidenceHeads(4, 3,
            token_level_confidence=token, use_separate_heads=separate)
        self.eval().requires_grad_(False)

    def forward(self, point, geo):
        s = torch.linspace(-1., 1., 20).reshape(1, 5, 4) + point
        z = torch.linspace(-.7, .7, 75).reshape(1, 5, 5, 3) + point * .3
        distogram = self.distogram_module(z)
        heads = self.confidence_module.confidence_heads
        raw = heads(s=s, z=z, x_pred=geo["x_pred"], d=geo["d"], feats=geo["feats"],
            pred_distogram_logits=distogram, multiplicity=1)
        scores = strict_confidence_scores(raw, geo["feats"], geo["x_pred"])
        return distogram, scores, raw


@pytest.mark.parametrize("separate,token", [(False, True), (True, True), (False, False), (True, False)])
def test_native_cached_outputs_complete_vjp_and_distogram_bypass(tmp_path, separate, token):
    model, geo = Producer(separate=separate, token=token), geometry()
    config = profile(tmp_path, pocket_residues=[{"asym_id": 0, "residue_index": 0}],
                     objective_arithmetic=CONTACT64_ARITHMETIC)
    affinity = {"affinity_pred_value": torch.tensor(.3), "affinity_logits_binary": torch.tensor(.2)}
    point = torch.tensor([.2], requires_grad=True)
    native_distogram, native_scores, _ = model(point, geo)
    native = reduce_profile(affinity, native_scores, native_distogram, geo["feats"], config)
    native_derivative, = torch.autograd.grad(native["dbmol_confidence_loss"], point)
    original_weights = {key: value.clone() for key, value in model.state_dict().items()}
    projection_ids = {name: id(getattr(model.confidence_module.confidence_heads, name)) for name in
                      observer.projection_names(model.confidence_module.confidence_heads)}
    with observer.StructuralCapture(model) as capture:
        distogram, scores, raw = model(point, geo)
        capture.require_complete()
        observed = reduce_profile(affinity, scores, distogram, geo["feats"], config)
        assert torch.equal(observed["dbmol_confidence_loss"], native["dbmol_confidence_loss"])
        direct, = torch.autograd.grad(observed["dbmol_confidence_loss"], point, retain_graph=True)
        cotangents = observer.tensor_gradients(observed["dbmol_confidence_loss"], capture.values)
        chain, = torch.autograd.grad(tuple(capture.values.values()), point,
            grad_outputs=tuple(cotangents.values()), retain_graph=True)
        torch.testing.assert_close(direct, native_derivative, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(chain, direct, rtol=1e-6, atol=1e-7)
        assert cotangents["distogram"].abs().sum() > 0  # PDE depends on predicted contact probabilities.
        assert torch.count_nonzero(cotangents["to_resolved_logits"]) == 0
        geometry_before = deepcopy(capture.geometry)
        raw_ids_before = {key: id(value) for key, value in capture.raw.items()}
        cached = observer.cached_raw(model.confidence_module.confidence_heads, capture.values, capture.geometry)
        assert raw_ids_before == {key: id(value) for key, value in capture.raw.items()}
        assert capture.geometry.keys() == geometry_before.keys()  # Copied prehooks must not erase bin constants.
        for key in geometry_before:
            if key == "feats":
                assert capture.geometry[key].keys() == geometry_before[key].keys()
                assert all(torch.equal(value, geometry_before[key][name]) for name, value in capture.geometry[key].items())
            else:
                assert torch.equal(capture.geometry[key], geometry_before[key])
        for key in ("complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde", "pae_logits", "plddt_logits", "pde_logits"):
            assert torch.equal(cached[key], raw[key])
        terms, selection = observer.cached_terms(model.confidence_module.confidence_heads,
            capture.values, capture.geometry, config, affinity)
        for key in observer.TERM_KEYS:
            assert torch.equal(terms[key], observed[key])
        cache_cotangents = observer.tensor_gradients(terms["dbmol_confidence_loss"], capture.values)
        for key in cotangents:
            torch.testing.assert_close(cache_cotangents[key], cotangents[key], rtol=1e-6, atol=1e-7)
        held, _ = observer.cached_terms(model.confidence_module.confidence_heads,
            capture.values, capture.geometry, config, affinity, held=selection)
        held_cotangents = observer.tensor_gradients(held["dbmol_confidence_loss"], capture.values)
        assert torch.equal(held["dbmol_confidence_loss"], terms["dbmol_confidence_loss"])
        for key in cotangents:
            torch.testing.assert_close(held_cotangents[key], cotangents[key], rtol=1e-6, atol=1e-7)
        widened, _ = observer.cached_terms(model.confidence_module.confidence_heads,
            {key: value.double() for key, value in capture.values.items()}, capture.geometry, config, affinity)
        assert all(value.dtype == torch.float64 for value in widened.values())
    assert projection_ids == {name: id(getattr(model.confidence_module.confidence_heads, name)) for name in projection_ids}
    assert all(torch.equal(value, original_weights[key]) for key, value in model.state_dict().items())
    # Forward hooks are removed from the original producer, not just its clone.
    assert not model.distogram_module._forward_hooks
    assert all(not getattr(model.confidence_module.confidence_heads, name)._forward_hooks for name in projection_ids)


def test_projection_vocabulary_is_complete_and_fails_closed():
    head = nn.Module()
    head.use_separate_heads = False
    head.to_plddt_logits = nn.Linear(1, 2)
    with pytest.raises(ValueError, match="membership"):
        observer.projection_names(head)
    head.to_resolved_logits, head.to_pde_logits, head.to_pae_logits = (nn.Linear(1, 2) for _ in range(3))
    assert len(observer.projection_names(head)) == 4
    head.unexpected = nn.Linear(1, 2)
    with pytest.raises(ValueError, match="membership"):
        observer.projection_names(head)


@pytest.mark.parametrize("mode", [NATIVE_ARITHMETIC, CONTACT64_ARITHMETIC])
def test_contact_pair_witness_and_held_branch_preserve_production_vjp(tmp_path, mode):
    cfg = profile(tmp_path, objective_arithmetic=mode)
    logits = torch.linspace(-2., 2., 1024).reshape(1, 4, 4, 64).requires_grad_()
    pocket, ligand = torch.tensor([True, True, False, False]), torch.tensor([False, False, True, True])
    original = contact_loss(logits.double() if mode == CONTACT64_ARITHMETIC else logits, pocket, ligand,
                            bin_centers=native_bin_centers(logits).double() if mode == CONTACT64_ARITHMETIC else None)
    pairs = observer.contact_pairs(logits, pocket, ligand, cfg)
    selection = observer.selection_state(pairs, {"ptm": torch.tensor([[.2, .3]]), "iptm": torch.tensor([[.1, .4]])},
        {"iptm": torch.tensor([.4])})
    held = observer.held_contact(pairs, selection)
    for key in held:
        assert torch.equal(held[key], original[key])
        direct, = torch.autograd.grad(original[key], logits, retain_graph=True)
        counterfactual, = torch.autograd.grad(held[key], logits, retain_graph=True)
        torch.testing.assert_close(direct, counterfactual, rtol=1e-6, atol=1e-8)


def test_global_contact_minimum_ties_use_distributed_native_subgradient():
    pairs = torch.tensor([[1., 1.], [2., 1.]], dtype=torch.float64, requires_grad=True)
    selection = observer.selection_state(pairs, {"ptm": torch.tensor([[.2, .3]]), "iptm": torch.tensor([[0., 0.]])},
        {"iptm": torch.zeros(1)})
    assert selection["contact_anchor"]["tie_indices"] == [0, 1, 3]
    held = observer.held_contact(pairs, selection)
    direct = (pairs.min(dim=1).values.mean() + pairs.min()) / 2
    gd, = torch.autograd.grad(direct, pairs, retain_graph=True)
    gh, = torch.autograd.grad(held["dbmol_contact_loss"], pairs)
    assert torch.equal(held["dbmol_contact_loss"], direct)
    torch.testing.assert_close(gh, gd, rtol=1e-12, atol=1e-12)
    # A baseline winner change can occur at both endpoints with identical
    # plus/minus winners. Comparing only endpoints would miss that switch.
    changed = deepcopy(selection)
    changed["contact_anchor"].update(index=3, tie_indices=[3], exact_ties=1)
    changed["tm_rows"]["ptm"]["index"] = 0
    changed["confidence_tm_uses_iptm"] = True
    switches = observer.selection_changes(changed, selection)
    assert switches["contact_anchor"] and switches["tm_rows"]["ptm"] and switches["confidence_tm_fallback"]


def test_native_bin_constants_survive_float64_confidence_variant(tmp_path):
    cfg = profile(tmp_path, objective_arithmetic=CONTACT64_ARITHMETIC)
    native = torch.zeros(1, 2, 2, 64)
    centers = native_bin_centers(native)
    cfg = replace(cfg, distance_cutoff=float(centers[19]))
    pocket, ligand = torch.tensor([True, False]), torch.tensor([False, True])
    actual = observer.contact_pairs(native.double(), pocket, ligand, cfg, bin_centers=centers.double())
    expected = contact_loss(native.double(), pocket, ligand, cutoff=cfg.distance_cutoff, bin_centers=centers.double())
    assert torch.equal(actual.min(), expected["dbmol_contact_loss"])
    assert not bool((centers.double() < cfg.distance_cutoff)[19])


def test_structural_provenance_does_not_require_affinity_members(tmp_path, monkeypatch):
    import inspect
    source = tmp_path / "site-packages/boltz/model/modules/confidencev2.py"
    source.parent.mkdir(parents=True)
    source.write_text("synthetic native source")
    class StructuralOnly(nn.Module):
        pass
    StructuralOnly.__module__ = "boltz.model.modules.confidencev2"
    model = StructuralOnly()
    assert not hasattr(model, "affinity_module1")
    monkeypatch.setattr(inspect, "getfile", lambda _type: str(source))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    files = {str(source.resolve()).replace("\\", "/"): digest}
    for index in range(49):
        file = source.with_name(f"fixture_{index}.py")
        file.write_text("synthetic exported native source " + str(index))
        files[str(file.resolve()).replace("\\", "/")] = hashlib.sha256(file.read_bytes()).hexdigest()
    fake = SimpleNamespace(helper=lambda: SimpleNamespace(_sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    monkeypatch.setattr(observer, "tools", lambda: fake)
    result = observer.structural_provenance({"parent_artifacts_sha256": files}, model)
    assert result["all_installed_model_sources_authenticated"] and not result["affinity_members_assumed"]
    with pytest.raises(ValueError, match="not pinned"):
        observer.structural_provenance({"parent_artifacts_sha256": dict(list(files.items())[:49])}, model)
    files[str(source.resolve()).replace("\\", "/")] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        observer.structural_provenance({"parent_artifacts_sha256": files}, model)


def test_fixed_budget_accounts_all_native_and_cached_calls():
    value = observer.budget((2, 4))
    assert value["maximum_total_full_forwards"] == 116
    assert value["explicit_full_forwards_per_case"] == [37, 73]
    assert value["cached_reductions_per_case"] == [148, 292]
    assert value["controller_headroom_seconds"] == 300
    assert value["baseline_upstream_reverse_traversals_per_case"] == 22
    assert value["maximum_physical_trunk_forwards_excluding_checkpoint_recomputation"] == 232
    assert value["optimizer_updates"] == value["new_defog_calls"] == 0
    with pytest.raises(ValueError, match="two/four"):
        observer.budget((4, 4))


def test_over_budget_cache_fails_before_any_model_import(monkeypatch):
    monkeypatch.setattr(observer, "_MAX_CACHE_CALLS", 2)
    monkeypatch.setattr(observer, "_CACHE_CALLS", 2)
    with pytest.raises(ValueError, match="budget exceeded"):
        observer.cached_terms(None, None, None, None, None)


def test_over_budget_native_forward_fails_before_scorer(monkeypatch):
    monkeypatch.setattr(observer, "_MAX_FULL_CALLS", 1)
    monkeypatch.setattr(observer, "_FULL_CALLS", 1)
    with pytest.raises(ValueError, match="budget exceeded"):
        observer.full_forward(None, None, None, None, None, None, "fixture")


def test_residual_eligibility_uses_independent_measured_chain_gradient():
    spec = importlib.util.spec_from_file_location("_structural_boundary_fixture",
        Path(__file__).parents[1] / "src/latentmol/boltz/operation_boundary_diagnostic.py")
    boundary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boundary)
    point, direction = torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    direct, chain = torch.tensor([4.], dtype=torch.float64), torch.tensor([4.00001], dtype=torch.float64)
    epsilon, numerical = .001, 4.0001
    native, frontier, attribution = observer.derivative_rows(point, direction, direct, chain, epsilon,
        epsilon * numerical, -epsilon * numerical, epsilon * numerical, -epsilon * numerical, 0.,
        dtype="torch.float64", rtol=0., atol=0., observer=boundary)
    assert native["analytic"] == 4. and frontier["analytic"] == 4.00001
    assert attribution["maximum_projected_chain_discrepancy"] > 0
    assert not attribution["failure_localization_eligible"]  # Chain error exceeds 1% of residual.
