"""HPC engineering fixtures for native raw logits and confidence soft-bond paths.

Small arithmetic modules are fixtures, not pretrained scorers or trained priors.
Real checkpoint parity/derivative gates remain separate required HPC work.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentmol.boltz.dbmol_model import attached_confidence, load_confidence_model, validate_frame_context
from latentmol.boltz.model import native_heads


def test_confidence_loader_authenticates_string_path_before_import_or_load(tmp_path):
    path = tmp_path / "confidence.ckpt"
    path.write_bytes(b"a malformed untrusted checkpoint")
    with pytest.raises(ValueError, match="Confidence checkpoint SHA256 mismatch"):
        load_confidence_model({"path": str(path), "sha256": "a" * 64})


class Head(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, *, z, **kw):
        return {"affinity_pred_value": z.sum().reshape(1),
                "affinity_logits_binary": (z.sum() * 0 + self.bias).reshape(1)}


def test_raw_ensemble_logits_are_not_reconstructed_from_averaged_probability():
    m = SimpleNamespace(input_embedder=lambda f, **kw: f["inputs"], affinity_ensemble=True,
                        affinity_module1=Head(-2.), affinity_module2=Head(4.))
    f = {"token_pad_mask": torch.ones(1, 2), "mol_type": torch.tensor([[0, 3]]),
         "affinity_token_mask": torch.tensor([[0, 1]]), "inputs": torch.ones(1, 2, 1)}
    z = torch.ones(1, 2, 2, 1)
    old = native_heads(m, f, z, torch.zeros(1, 1, 2, 3))
    new = native_heads(m, f, z, torch.zeros(1, 1, 2, 3), include_logits=True)
    assert set(old) == {"affinity_pred_value", "affinity_probability_binary"}
    assert new["affinity_logits_binary"] == 1
    assert not torch.allclose(new["affinity_logits_binary"].sigmoid(), new["affinity_probability_binary"])
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)


class Pair(nn.Module):
    def forward(self, s, z, **kw):
        return s, z


class Heads(nn.Module):
    def forward(self, *, z, **kw):
        return {"fixture": z.sum()}


class ConfidenceFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_inputs_norm = nn.Identity(); self.s_norm = nn.Identity(); self.z_norm = nn.Identity()
        self.no_update_s = False; self.add_s_input_to_s = False
        self.add_z_input_to_z = True; self.bond_type_feature = True; self.add_s_to_z_prod = False
        self.token_bonds_type = nn.Embedding(7, 1)
        with torch.no_grad(): self.token_bonds_type.weight[:, 0] = torch.arange(7.)
        self.s_to_z = nn.Linear(1, 1, bias=False); self.s_to_z_transpose = nn.Linear(1, 1, bias=False)
        self.dist_bin_pairwise_embed = nn.Embedding(3, 1)
        self.boundaries = torch.tensor([2., 8.])
        self.pairformer_stack = Pair(); self.confidence_heads = Heads()
        self.eval().requires_grad_(False)

    def rel_pos(self, f): return torch.zeros(1, 2, 2, 1)
    def token_bonds(self, x): return x * 0
    def contact_conditioning(self, f): return torch.zeros(1, 2, 2, 1)


def test_confidence_uses_soft_typed_bonds_even_when_seed_integer_bonds_are_fixed():
    module = ConfidenceFixture()
    f = {"token_bonds": torch.zeros(1, 2, 2, 1), "type_bonds": torch.zeros(1, 2, 2, dtype=torch.long),
         "token_pad_mask": torch.ones(1, 2), "token_to_rep_atom": torch.eye(2)[None]}
    types = torch.zeros(1, 2, 2, 7)
    types[..., 0] = 1
    types[0, 0, 1] = torch.tensor([0., 0., .4, .6, 0., 0., 0.])
    types.requires_grad_()
    out = attached_confidence(module, torch.ones(1, 2, 1), torch.ones(1, 2, 1),
                              torch.zeros(1, 2, 2, 1), f, torch.zeros(1, 1, 2, 3),
                              torch.zeros(1, 2, 2, 64), types)
    derivative, = torch.autograd.grad(out["fixture"], types)
    torch.testing.assert_close(derivative[0, 0, 1], torch.arange(7.))
    assert all(p.grad is None and not p.requires_grad for p in module.parameters())
    assert f["type_bonds"].sum() == 0


def test_integer_and_onehot_confidence_embedding_agree_exactly():
    module = ConfidenceFixture()
    labels = torch.tensor([[[0, 2], [2, 0]]])
    probabilities = torch.nn.functional.one_hot(labels, 7).float()
    torch.testing.assert_close(probabilities @ module.token_bonds_type.weight,
                               module.token_bonds_type(labels), rtol=0, atol=0)


def test_native_frame_assignment_layout_is_checked_before_silent_helper_fallback():
    f = {"token_pad_mask": torch.ones(1, 4), "atom_pad_mask": torch.ones(1, 5),
         "asym_id": torch.tensor([[0, 1, 1, 1]]), "mol_type": torch.tensor([[0, 3, 3, 3]]),
         "frames_idx": torch.zeros(1, 4, 3, dtype=torch.long),
         "atom_to_token": torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.],
                                          [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]])}
    validate_frame_context(f, torch.zeros(1, 1, 5, 3))
    # Reassign one protein atom to the ligand: native slice assignment would
    # now try to put four atom frames into only three nonpolymer token rows.
    f["atom_to_token"][0, 1] = torch.tensor([0., 1., 0., 0.])
    with pytest.raises(ValueError, match="one atom per nonpolymer token"):
        validate_frame_context(f, torch.zeros(1, 1, 5, 3))


def _frame_fixture():
    """Pinned featurizer's batch/ensemble/token layout, without model weights."""
    return {"token_pad_mask": torch.ones(1, 3), "atom_pad_mask": torch.ones(1, 3),
            "asym_id": torch.tensor([[0, 1, 1]]), "mol_type": torch.tensor([[0, 3, 3]]),
            "frames_idx": torch.tensor([[[[0, 0, 0], [1, 1, 1], [2, 2, 2]]]]),
            "atom_to_token": torch.eye(3)[None]}


def test_native_single_ensemble_frames_are_validated_without_mutating_features():
    features = _frame_fixture()
    original = features["frames_idx"]
    saved = original.clone()
    validate_frame_context(features, torch.zeros(1, 1, 3, 3))
    assert features["frames_idx"] is original
    torch.testing.assert_close(original, saved, rtol=0, atol=0)
    # Native compute_frame_pred reshapes either accepted layout to this same
    # batch/multiplicity/token tensor. Do not squeeze the scorer's shared input.
    legacy = {**features, "frames_idx": original[:, 0]}
    validate_frame_context(legacy, torch.zeros(1, 3, 3))
    torch.testing.assert_close(original.reshape(1, 1, -1, 3),
                               legacy["frames_idx"].reshape(1, 1, -1, 3), rtol=0, atol=0)


@pytest.mark.parametrize("fault, message", [
    ("ensemble", "frame shape"), ("dtype", "frame dtype"),
    ("index", "index out of bounds"), ("coordinates", "coordinate shape"),
    ("nonfinite", "Nonfinite.*coordinates")])
def test_native_frame_checks_keep_rejecting_unsupported_or_corrupt_inputs(fault, message):
    features = _frame_fixture()
    coords = torch.zeros(1, 1, 3, 3)
    if fault == "ensemble":
        features["frames_idx"] = features["frames_idx"].repeat(1, 2, 1, 1)
    elif fault == "dtype":
        features["frames_idx"] = features["frames_idx"].float()
    elif fault == "index":
        features["frames_idx"][0, 0, 2, 0] = 3
    elif fault == "coordinates":
        coords = torch.zeros(1, 2, 3, 3)
    elif fault == "nonfinite":
        coords[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match=message):
        validate_frame_context(features, coords)
