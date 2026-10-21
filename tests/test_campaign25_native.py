"""Allocated Torch-only 25-slot mapping and correspondence regression checks."""

import pytest
import torch

from mosaic.small_molecule.boltz2.native import inject, validate_context


def context25():
    n = 25
    ai = torch.roll(torch.arange(n) + 3, 5)
    ti = torch.roll(torch.arange(n) + 3, 9)
    pool = torch.zeros(1, n + 3, n + 3)
    pool[0, ai, ti] = 1
    pool[0, torch.arange(3), torch.arange(3)] = 1
    types = torch.zeros(1, n + 3, n + 3, dtype=torch.long)
    types[0, ti[:-1], ti[1:]] = types[0, ti[1:], ti[:-1]] = 2
    types[0, 0, 1] = types[0, 1, 0] = 1
    element = torch.nn.functional.one_hot(torch.full((1, n + 3), 7), 128)
    element[:, ai] = torch.nn.functional.one_hot(torch.full((1, n), 6), 128)
    affinity = torch.zeros(1, n + 3, dtype=torch.bool)
    affinity[:, ti] = True
    x = torch.nn.functional.one_hot(torch.zeros(n, dtype=torch.long), 6).float()
    i, j = torch.triu_indices(n, n, 1)
    return {
        "version": "mosaic-boltz-fixed-pose-v1", "n_atoms": n,
        "atom_indices": ai, "token_indices": ti,
        "coordinates": torch.zeros(1, 1, n + 3, 3), "seed_X": x,
        "seed_E": torch.nn.functional.one_hot((j == i + 1).long(), 5).float(),
        "features": {"ref_element": element, "ref_charge": torch.zeros(1, n + 3),
                     "atom_to_token": pool, "token_to_rep_atom": pool.transpose(1, 2),
                     "token_pad_mask": torch.ones(1, n + 3), "atom_pad_mask": torch.ones(1, n + 3),
                     "affinity_token_mask": affinity, "type_bonds": types,
                     "token_bonds": (types != 0).float()[..., None]},
    }


def test_25_slot_native_seed_exact_parity_and_protein_fields_unchanged():
    c = context25()
    validate_context(c)
    f, types = inject(c["seed_X"], c["seed_E"], c)
    torch.testing.assert_close(f["ref_element"], c["features"]["ref_element"].float(), atol=0, rtol=0)
    torch.testing.assert_close(f["token_bonds"], c["features"]["token_bonds"], atol=0, rtol=0)
    assert types[0, 0, 1, 1] == 1
    assert f["atom_to_token"] is c["features"]["atom_to_token"]


def test_all_300_unordered_pairs_have_native_typed_bond_derivatives():
    c = context25()
    x = torch.full((25, 6), 1 / 6, requires_grad=True)
    e = torch.full((300, 5), .2, requires_grad=True)
    f, types = inject(x, e, c)
    value = (types * torch.arange(7)).sum() + (f["ref_element"] * torch.arange(128)).sum()
    gx, ge = torch.autograd.grad(value, (x, e))
    torch.testing.assert_close(gx, torch.tensor([6., 7., 8., 16., 9., 17.]).expand(25, -1))
    torch.testing.assert_close(ge, (2 * torch.tensor([0., 2., 3., 4., 5.])).expand(300, -1))


@pytest.mark.parametrize("error", ["duplicate", "wrong_count", "charge", "lost_ligand"])
def test_25_slot_correspondence_rejects_incomplete_or_changed_context(error):
    c = context25()
    if error == "duplicate":
        c["atom_indices"][0] = c["atom_indices"][1]
    elif error == "wrong_count":
        c["n_atoms"] = 6
    elif error == "charge":
        c["features"]["ref_charge"][0, c["atom_indices"][0]] = 1
    else:
        c["features"]["affinity_token_mask"][0, c["token_indices"][0]] = False
    with pytest.raises(ValueError):
        validate_context(c)


def test_six_slot_arrays_cannot_be_injected_into_25_slot_context():
    with pytest.raises(ValueError, match="shapes"):
        inject(torch.ones(6, 6) / 6, torch.ones(15, 5) / 5, context25())
