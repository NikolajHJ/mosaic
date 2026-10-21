"""Allocated Torch pose-placement and no-diffusion gradient checks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosaic.small_molecule.pocket import derive_definition, audit_boltz_pose
from mosaic.small_molecule.boltz2 import native
from test_campaign25_native import context25


def fixture():
    root = Path(__file__).resolve().parents[1]
    return derive_definition(root / "tests/data/brd4_3mxf/3MXF.pdb", root / "tests/data/brd4_3mxf/fixture_25.json")


def test_actual_pose_gate_fails_far_ligand_and_missing_pocket_mapping():
    p = fixture()
    names = {v: k for k, v in __import__('mosaic.small_molecule.pocket', fromlist=['THREE_TO_ONE']).THREE_TO_ONE.items()}
    rows = [{"atom_num": 1, "mol_type": 0, "asym_id": 0, "res_idx": i,
             "res_name": names[aa]} for i, aa in enumerate(p["protein_sequence"])]
    rows += [{"atom_num": 1, "mol_type": 3, "asym_id": 1, "res_idx": 0, "res_name": "LIG"}] * 25
    t = SimpleNamespace(tokens=rows)
    batch = {"coords": torch.zeros(1, 1, 152, 3), "atom_pad_mask": torch.ones(1, 152)}
    assert audit_boltz_pose(t, batch, p, list(range(127, 152)))["status"] == "passed"
    batch["coords"][:, :, 127:] = 20
    assert audit_boltz_pose(t, batch, p, list(range(127, 152)))["status"] == "failed"
    rows[98]["res_idx"] = 97
    with pytest.raises(ValueError, match="mapping"):
        audit_boltz_pose(t, batch, p, list(range(127, 152)))


def test_affinity_reaches_graph_inputs_but_not_prepared_coordinates_or_diffusion(monkeypatch):
    c = context25()
    c["coordinates"] = c["coordinates"].requires_grad_(True)
    c["features"]["mol_type"] = torch.zeros(1, 28, dtype=torch.long)
    c["features"]["mol_type"][0, c["token_indices"]] = 3
    x = torch.full((25, 6), 1 / 6, requires_grad=True)
    e = torch.full((300, 5), .2, requires_grad=True)
    def trunk(model, f, types):
        return (types * torch.arange(7)).sum().reshape(1, 1, 1, 1)
    monkeypatch.setattr(native, "attached_trunk", trunk)
    def head(**kwargs):
        assert not kwargs["x_pred"].requires_grad and kwargs["x_pred"].grad_fn is None
        value = kwargs["z"].sum() + (kwargs["feats"]["ref_element"] * torch.arange(128)).sum()
        return {"affinity_pred_value": value, "affinity_logits_binary": value * 0}
    model = SimpleNamespace(input_embedder=lambda f, **k: f["ref_element"], affinity_ensemble=False, affinity_module=head)
    values = native.score(model, c, x, e)
    gx, ge, gc = torch.autograd.grad(values[0], (x, e, c["coordinates"]), allow_unused=True)
    assert gx.abs().sum() > 0 and ge.abs().sum() > 0 and gc is None


def test_failed_pocket_context_cannot_be_loaded_for_scoring():
    c = context25()
    c["pocket"] = {"pocket_definition_sha256": "a" * 64, "pocket_pose_audit": {"status": "failed"}, "diffusion_gradient": False}
    with pytest.raises(ValueError, match="pocket"):
        native.validate_context(c)
