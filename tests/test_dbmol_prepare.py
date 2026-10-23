"""HPC-only pocket geometry fixture, not a biologically validated site."""

from types import SimpleNamespace

import torch
from torch.nn.functional import one_hot

from latentmol.boltz import dbmol_prepare


def test_initial_pocket_excludes_hydrogen_and_cutoff_boundary(monkeypatch):
    validation_calls = []
    f = {"mol_type": torch.tensor([[0, 0, 3]]), "token_pad_mask": torch.ones(1, 3),
         "ref_element": one_hot(torch.tensor([[6, 6, 1, 6]]), 128),
         "atom_pad_mask": torch.ones(1, 4), "asym_id": torch.tensor([[0, 0, 1]]),
         "residue_index": torch.tensor([[11, 12, 0]]),
         "atom_to_token": torch.tensor([[[1., 0., 0.], [0., 1., 0.],
                                          [0., 1., 0.], [0., 0., 1.]]])}
    prepared = SimpleNamespace(features=f, validate=lambda: validation_calls.append(True),
        active=torch.tensor([True]), slot_to_atom=torch.tensor([3]),
        coordinates=torch.tensor([[[[7.9, 0., 0.], [8., 0., 0.],
                                    [1., 0., 0.], [0., 0., 0.]]]]))
    monkeypatch.setattr(dbmol_prepare, "context_digest", lambda p: "a" * 64)
    before = prepared.coordinates.clone()
    pocket = dbmol_prepare.initial_pose_pocket(prepared)
    assert validation_calls == [True]
    assert pocket["pocket_residues"] == [{"asym_id": 0, "residue_index": 11}]
    assert pocket["protein_heavy_atoms_in_pocket"] == 1
    torch.testing.assert_close(before, prepared.coordinates, rtol=0, atol=0)
