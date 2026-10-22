"""Allocated CPU tests of native Torch mapping; no JAX/decoder dependencies."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import pytest
import torch

from mosaic.small_molecule.boltz2.native import (
    ATOMIC_NUMBERS,
    TYPED_BOND_IDS,
    inject,
    validate_context,
)


def context_fixture():
    ai, ti = torch.tensor([8, 2, 9, 4, 6, 10]), torch.tensor([7, 1, 8, 3, 5, 6])
    n, a = 9, 11
    elements = torch.nn.functional.one_hot(torch.tensor([[7] * a]), 128)
    elements[:, ai] = torch.nn.functional.one_hot(torch.full((1, 6), 6), 128)
    pool = torch.zeros(1, a, n)
    pool[0, ai, ti] = 1
    pool[0, [0, 1, 3], [0, 2, 4]] = 1
    types = torch.zeros(1, n, n, dtype=torch.long)
    types[0, ti[:-1], ti[1:]] = types[0, ti[1:], ti[:-1]] = 2
    types[0, 0, 2] = types[0, 2, 0] = 1
    affinity = torch.zeros(1, n, dtype=torch.bool)
    affinity[:, ti] = True
    atom_pad = (pool.sum(-1) > 0).float()
    f = {
        "ref_element": elements,
        "ref_charge": torch.zeros(1, a),
        "token_bonds": (types != 0).float()[..., None],
        "type_bonds": types,
        "token_pad_mask": torch.ones(1, n),
        "atom_pad_mask": atom_pad,
        "affinity_token_mask": affinity,
        "atom_to_token": pool,
        "token_to_rep_atom": pool.transpose(1, 2),
        "mol_type": torch.where(affinity, 3, 0),
    }
    x = torch.nn.functional.one_hot(torch.zeros(6, dtype=torch.long), 6).float()
    i, j = torch.triu_indices(6, 6, 1)
    edges = torch.nn.functional.one_hot((j == i + 1).long(), 5).float()
    return {
        "version": "mosaic-boltz-fixed-pose-v1",
        "features": f,
        "coordinates": torch.zeros(1, 1, a, 3),
        "atom_indices": ai,
        "token_indices": ti,
        "seed_X": x,
        "seed_E": edges,
    }


def test_exact_native_seed_mapping_and_auxiliary_identity():
    c = context_fixture()
    validate_context(c)
    f, types = inject(c["seed_X"], c["seed_E"], c)
    for key in ("ref_element", "ref_charge", "token_bonds"):
        torch.testing.assert_close(
            f[key], c["features"][key].to(f[key]), atol=0, rtol=0
        )
    assert types[0, 0, 2, 1] == 1  # native OTHER remains on protein pair.
    assert f["atom_to_token"] is c["features"]["atom_to_token"]
    assert ATOMIC_NUMBERS == (6, 7, 8, 16, 9, 17)
    assert TYPED_BOND_IDS == (0, 2, 3, 4, 5)


def test_probability_expectations_native_none_and_mirrored_derivatives():
    c = context_fixture()
    x = torch.full((6, 6), 1 / 6, requires_grad=True)
    e = torch.full((15, 5), 0.2, requires_grad=True)
    f, types = inject(x, e, c)
    ti = c["token_indices"]
    torch.testing.assert_close(f["token_bonds"][0, ti[0], ti[1]], torch.tensor([0.8]))
    torch.testing.assert_close(
        types[0, ti[0], ti[1]], torch.tensor([0.2, 0, 0.2, 0.2, 0.2, 0.2, 0])
    )
    weights = torch.arange(7).float()
    value = (
        (types * weights).sum()
        + (f["ref_element"] * torch.arange(128)).sum()
        + 2 * f["token_bonds"].sum()
    )
    gx, ge = torch.autograd.grad(value, (x, e))
    torch.testing.assert_close(gx, torch.tensor(ATOMIC_NUMBERS).float().expand(6, -1))
    # One edge appears twice, with a separate -2 connectivity derivative per orientation.
    expected = 2 * torch.tensor([-2.0, 2.0, 3.0, 4.0, 5.0])
    torch.testing.assert_close(ge, expected.expand(15, -1))


def test_invalid_slot_mapping_is_rejected():
    c = context_fixture()
    c["token_indices"][1] = c["token_indices"][0]
    with pytest.raises(ValueError, match="injective"):
        validate_context(c)


def test_charged_context_rejected_by_current_neutral_contract():
    c = context_fixture()
    c["features"]["ref_charge"][0, c["atom_indices"][0]] = 1
    with pytest.raises(ValueError, match="neutral"):
        validate_context(c)
