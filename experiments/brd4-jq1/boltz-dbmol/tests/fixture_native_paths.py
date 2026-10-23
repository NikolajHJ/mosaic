"""Allocated path fixtures; trained checkpoint evidence is a separate gate."""

import copy

import pytest
import torch
from torch import nn

from mosaic.small_molecule.boltz2.dbmol import (
    DBMolScorer, POCKET_SHA256, attached_confidence, fixed_pocket_masks,
    load_structure_model,
    validate_frame_context, validate_graph,
)
from mosaic.small_molecule.boltz2.native import ATOMIC_NUMBERS, TYPED_BOND_IDS, inject


def mapped_context(n=25):
    tokens = n + 2
    indices = torch.arange(2, tokens)
    return {
        "atom_indices": indices,
        "token_indices": indices,
        "features": {
            "ref_element": torch.nn.functional.one_hot(torch.zeros(1, tokens, dtype=torch.long), 128).float(),
            "ref_charge": torch.zeros(1, tokens),
            "token_bonds": torch.zeros(1, tokens, tokens, 1),
            "type_bonds": torch.zeros(1, tokens, tokens, dtype=torch.long),
            "token_pad_mask": torch.ones(1, tokens),
        },
    }


def test_shared_probability_mapping_has_manual_atom_and_unordered_typed_bond_values():
    context = mapped_context()
    x = torch.arange(1, 151, dtype=torch.float32).reshape(25, 6)
    x = (x / x.sum(-1, keepdim=True)).requires_grad_()
    e = torch.arange(1, 1501, dtype=torch.float32).reshape(300, 5)
    e = (e / e.sum(-1, keepdim=True)).requires_grad_()
    before = copy.deepcopy(context)
    f, bond_types = inject(x, e, context)
    for slot in (0, 1, 24):
        for category, atomic_number in enumerate(ATOMIC_NUMBERS):
            assert f["ref_element"][0, 2 + slot, atomic_number] == x[slot, category]
    for row, (i, j) in enumerate(zip(*torch.triu_indices(25, 25, 1))):
        i, j = int(i) + 2, int(j) + 2
        torch.testing.assert_close(f["token_bonds"][0, i, j, 0], 1 - e[row, 0], atol=0, rtol=0)
        torch.testing.assert_close(f["token_bonds"][0, j, i, 0], 1 - e[row, 0], atol=0, rtol=0)
        for category, native_id in enumerate(TYPED_BOND_IDS):
            assert bond_types[0, i, j, native_id] == e[row, category]
            assert bond_types[0, j, i, native_id] == e[row, category]
    assert torch.equal(bond_types[0].diagonal(0, 0, 1).T[:, 0], torch.ones(27))
    assert torch.equal(bond_types[0].diagonal(0, 0, 1).T[:, 1:], torch.zeros(27, 6))
    # Mapped changes must not leak to receptor/cross/charge/auxiliary state.
    torch.testing.assert_close(f["ref_element"][:, :2], before["features"]["ref_element"][:, :2], atol=0, rtol=0)
    torch.testing.assert_close(f["token_bonds"][:, :2], before["features"]["token_bonds"][:, :2], atol=0, rtol=0)
    torch.testing.assert_close(f["ref_charge"], before["features"]["ref_charge"], atol=0, rtol=0)
    for name, original in before["features"].items():
        torch.testing.assert_close(context["features"][name], original, atol=0, rtol=0)


def test_mapping_vjp_adds_both_directions_and_keeps_present_typed_contrasts():
    context = mapped_context()
    x = torch.full((25, 6), 1 / 6, requires_grad=True)
    e = torch.full((300, 5), .2, requires_grad=True)
    f, bonds = inject(x, e, context)
    atom_weights = torch.arange(128, dtype=torch.float32)
    typed_weights = torch.arange(7, dtype=torch.float32)
    value = (f["ref_element"] * atom_weights).sum()
    value = value + 3 * f["token_bonds"].sum() + (bonds * typed_weights).sum()
    gx, ge = torch.autograd.grad(value, (x, e))
    torch.testing.assert_close(gx, torch.tensor(ATOMIC_NUMBERS).float().expand(25, -1), atol=0, rtol=0)
    # Every undirected edge is inserted in both matrix orientations. Connectivity
    # contributes -6 only to NONE, while type contributes twice each native ID.
    expected = torch.tensor([2 * n for n in TYPED_BOND_IDS], dtype=torch.float32)
    expected[0] -= 6
    torch.testing.assert_close(ge, expected.expand(300, -1), atol=0, rtol=0)
    assert (ge[:, 1:].max(-1).values - ge[:, 1:].min(-1).values).max() > 0


class Pair(nn.Module):
    def forward(self, s, z, **_):
        return s, z


class Readout(nn.Module):
    def forward(self, *, z, **_):
        return {"fixture": z.sum()}


class ConfidenceFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_inputs_norm, self.s_norm, self.z_norm = nn.Identity(), nn.Identity(), nn.Identity()
        self.no_update_s = False
        self.add_s_input_to_s = False
        self.add_z_input_to_z = True
        self.bond_type_feature = True
        self.add_s_to_z_prod = False
        self.token_bonds_type = nn.Embedding(7, 1)
        with torch.no_grad(): self.token_bonds_type.weight[:, 0] = torch.arange(7.)
        self.s_to_z, self.s_to_z_transpose = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        self.dist_bin_pairwise_embed = nn.Embedding(3, 1)
        self.boundaries = torch.tensor([2., 8.])
        self.pairformer_stack, self.confidence_heads = Pair(), Readout()
        self.eval().requires_grad_(False)

    def rel_pos(self, _): return torch.zeros(1, 2, 2, 1)
    def token_bonds(self, values): return values * 0
    def contact_conditioning(self, _): return torch.zeros(1, 2, 2, 1)


def test_attached_confidence_uses_live_soft_typed_bonds_in_additional_confidence_path():
    module = ConfidenceFixture()
    features = {"token_bonds": torch.zeros(1, 2, 2, 1),
                "type_bonds": torch.zeros(1, 2, 2, dtype=torch.long),
                "token_pad_mask": torch.ones(1, 2),
                "token_to_rep_atom": torch.eye(2)[None]}
    bonds = torch.nn.functional.one_hot(features["type_bonds"], 7).float()
    bonds[0, 0, 1] = torch.tensor([0., 0., .4, .6, 0., 0., 0.])
    bonds.requires_grad_()
    result = attached_confidence(module, torch.ones(1, 2, 1), torch.ones(1, 2, 1),
                                 torch.zeros(1, 2, 2, 1), features,
                                 torch.zeros(1, 1, 2, 3),
                                 torch.zeros(1, 2, 2, 64), bonds)
    gradient, = torch.autograd.grad(result["fixture"], bonds)
    torch.testing.assert_close(gradient[0, 0, 1], torch.arange(7.), atol=0, rtol=0)
    assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert features["type_bonds"].sum() == 0


def frame_fixture():
    return {
        "token_pad_mask": torch.ones(1, 3), "atom_pad_mask": torch.ones(1, 3),
        "asym_id": torch.tensor([[0, 1, 1]]), "mol_type": torch.tensor([[0, 3, 3]]),
        "frames_idx": torch.tensor([[[[0, 0, 0], [1, 1, 1], [2, 2, 2]]]]),
        "atom_to_token": torch.eye(3)[None],
    }


def test_frame_contract_supports_native_single_ensemble_and_legacy_without_mutation():
    f = frame_fixture()
    original = f["frames_idx"].clone()
    validate_frame_context(f, torch.zeros(1, 1, 3, 3))
    validate_frame_context({**f, "frames_idx": f["frames_idx"][:, 0]}, torch.zeros(1, 3, 3))
    torch.testing.assert_close(f["frames_idx"], original, atol=0, rtol=0)


@pytest.mark.parametrize("fault", ("ensemble", "dtype", "bounds", "coordinate_shape", "nonfinite", "token_padding", "atom_padding", "chain_order", "chain_types"))
def test_frame_contract_rejects_native_helpers_silent_failure_layouts(fault):
    f, coords = frame_fixture(), torch.zeros(1, 1, 3, 3)
    if fault == "ensemble": f["frames_idx"] = f["frames_idx"].repeat(1, 2, 1, 1)
    elif fault == "dtype": f["frames_idx"] = f["frames_idx"].float()
    elif fault == "bounds": f["frames_idx"][0, 0, 2, 0] = 3
    elif fault == "coordinate_shape": coords = torch.zeros(1, 2, 3, 3)
    elif fault == "nonfinite": coords[0, 0, 0, 0] = float("nan")
    elif fault == "token_padding": f["token_pad_mask"][0, 1] = 0
    elif fault == "atom_padding": f["atom_pad_mask"][0, 1] = 0
    elif fault == "chain_order": f["asym_id"] = torch.tensor([[1, 0, 1]])
    else: f["mol_type"] = torch.tensor([[0, 0, 3]])
    with pytest.raises(ValueError):
        validate_frame_context(f, coords)


def test_frame_contract_rejects_more_nonpolymer_atoms_than_tokens():
    f = {"token_pad_mask": torch.ones(1, 4), "atom_pad_mask": torch.ones(1, 5),
         "asym_id": torch.tensor([[0, 1, 1, 1]]), "mol_type": torch.tensor([[0, 3, 3, 3]]),
         "frames_idx": torch.zeros(1, 4, 3, dtype=torch.long),
         "atom_to_token": torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.],
                                          [0., 1., 0., 0.], [0., 0., 1., 0.],
                                          [0., 0., 0., 1.]]])}
    with pytest.raises(ValueError):
        validate_frame_context(f, torch.zeros(1, 1, 5, 3))


def test_fixed_pocket_reuses_authenticated_original_ids_and_excludes_all_padding():
    context = mapped_context()
    context["features"].update(
        mol_type=torch.tensor([[0, 0] + [3] * 25]),
        affinity_token_mask=torch.tensor([[0, 0] + [1] * 25]),
    )
    context["pocket"] = {"pocket_definition_sha256": POCKET_SHA256,
                         "diffusion_gradient": False,
                         "pocket_pose_audit": {"status": "passed",
                                               "native_pocket_token_indices": [0, 1]}}
    p, l = fixed_pocket_masks(context)
    assert p.tolist() == [True, True] + [False] * 25
    assert l.tolist() == [False, False] + [True] * 25
    context["features"]["token_pad_mask"][0, 0] = 0
    with pytest.raises(ValueError):
        fixed_pocket_masks(context)


@pytest.mark.parametrize("fault", ("hash", "diffusion", "audit", "empty", "duplicate", "negative", "noninteger", "bounds", "ligand", "count"))
def test_fixed_pocket_refuses_regeneration_ambiguity_or_another_biological_context(fault):
    context = mapped_context()
    context["features"].update(
        mol_type=torch.tensor([[0, 0] + [3] * 25]),
        affinity_token_mask=torch.tensor([[0, 0] + [1] * 25]),
    )
    context["pocket"] = {"pocket_definition_sha256": POCKET_SHA256,
                         "diffusion_gradient": False,
                         "pocket_pose_audit": {"status": "passed",
                                               "native_pocket_token_indices": [0, 1]}}
    pocket, audit = context["pocket"], context["pocket"]["pocket_pose_audit"]
    if fault == "hash": pocket["pocket_definition_sha256"] = "a" * 64
    elif fault == "diffusion": pocket["diffusion_gradient"] = True
    elif fault == "audit": audit["status"] = "pending"
    elif fault == "empty": audit["native_pocket_token_indices"] = []
    elif fault == "duplicate": audit["native_pocket_token_indices"] = [0, 0]
    elif fault == "negative": audit["native_pocket_token_indices"] = [-1]
    elif fault == "noninteger": audit["native_pocket_token_indices"] = [0.0]
    elif fault == "bounds": audit["native_pocket_token_indices"] = [27]
    elif fault == "ligand": audit["native_pocket_token_indices"] = [2]
    else: context["features"]["affinity_token_mask"][0, 2] = 0
    with pytest.raises(ValueError):
        fixed_pocket_masks(context)


@pytest.mark.parametrize("fault", ("double", "atom_count", "bond_count", "nonfinite", "negative", "unnormalized"))
def test_direct_graph_boundary_never_hardens_projects_or_clips_a_bad_input(fault):
    x, e = torch.full((25, 6), 1 / 6), torch.full((300, 5), .2)
    context = {"coordinates": torch.zeros(1, 27, 3)}
    if fault == "double": x = x.double()
    elif fault == "atom_count": x = x[:-1]
    elif fault == "bond_count": e = e[:-1]
    elif fault == "nonfinite": x[0, 0] = float("nan")
    elif fault == "negative": x[0, 0] = -.1
    else: e[0, 0] = .3
    with pytest.raises((ValueError, AssertionError)):
        validate_graph(x, e, context)


def test_structure_checkpoint_hash_is_checked_before_any_native_import_or_load(tmp_path):
    checkpoint = tmp_path / "untrusted.ckpt"
    checkpoint.write_bytes(b"untrusted checkpoint fixture")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        load_structure_model(checkpoint)


def scorer_context():
    context = mapped_context()
    tokens = 27
    context["features"].update(
        atom_pad_mask=torch.ones(1, tokens), atom_to_token=torch.eye(tokens)[None],
        frames_idx=torch.arange(tokens).reshape(1, 1, tokens, 1).expand(-1, -1, -1, 3).clone(),
        asym_id=torch.tensor([[0, 0] + [1] * 25]),
        mol_type=torch.tensor([[0, 0] + [3] * 25]),
        affinity_token_mask=torch.tensor([[0, 0] + [1] * 25]),
    )
    context["coordinates"] = torch.zeros(1, tokens, 3)
    context["pocket"] = {"pocket_definition_sha256": POCKET_SHA256,
                         "diffusion_gradient": False,
                         "pocket_pose_audit": {"status": "passed",
                                               "native_pocket_token_indices": [0, 1]}}
    return context


def test_constructor_keeps_both_model_families_frozen_and_eval(monkeypatch):
    # Synthetic context arithmetic only. Actual native.validate_context plus
    # installation/asset hashes remain mandatory in the real worker.
    monkeypatch.setattr("mosaic.small_molecule.boltz2.native.validate_context", lambda value: None)
    a, s = nn.Linear(1, 1), nn.Linear(1, 1)
    model = DBMolScorer(a, s, scorer_context())
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("field", ("ref_element", "coordinates"))
def test_context_input_tensors_cannot_silently_become_trainable(field, monkeypatch):
    monkeypatch.setattr("mosaic.small_molecule.boltz2.native.validate_context", lambda value: None)
    context = scorer_context()
    target = context["coordinates"] if field == "coordinates" else context["features"][field]
    target.requires_grad_()
    with pytest.raises(ValueError):
        DBMolScorer(nn.Linear(1, 1), nn.Linear(1, 1), context)
