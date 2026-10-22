"""Allocated JAX pocket objective and native-token correspondence checks."""

# Shared test import paths; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__, include_tests=True)

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.contracts import MolecularTrunkOutput
from mosaic.small_molecule.losses import NativeContactLoss
from mosaic.small_molecule.pocket import derive_definition, map_protenix_pocket
from mosaic.small_molecule.state import graph_from_hard


def test_outside_pocket_contacts_cannot_improve_the_objective_or_gradient():
    graph = graph_from_hard([0, 0], [1])
    logits = jnp.zeros((5, 5, 3)).at[3, 0, 0].set(1).at[4, 0, 0].set(2)
    def loss(z):
        output = MolecularTrunkOutput(z, jnp.array([4., 10., jnp.inf]), (3, 4), (0, 1, 2))
        return NativeContactLoss(8., (0,))(graph, output, key=jax.random.key(17))[0]
    changed = logits.at[:, 1:3, 0].set(100)
    np.testing.assert_array_equal(loss(logits), loss(changed))
    gradient = jax.grad(loss)(logits)
    np.testing.assert_array_equal(gradient[:, 1:3], 0)
    assert float(jnp.linalg.norm(gradient[:, 0])) > 0


@pytest.mark.parametrize("pocket", [(), (0, 0), (-1,), (True,)])
def test_bad_pocket_sets_rejected(pocket):
    with pytest.raises(ValueError):
        NativeContactLoss(8., pocket)


def test_native_ligand_id_cannot_be_used_as_pocket():
    output = MolecularTrunkOutput(jnp.zeros((3, 3, 3)), jnp.array([4., 10., jnp.inf]), (1, 2), (0,))
    with pytest.raises(ValueError, match="protein map"):
        NativeContactLoss(8., (1,))(graph_from_hard([0, 0], [1]), output, key=jax.random.key(17))


def test_permuted_native_token_map_uses_residue_metadata_and_checks_sequence():
    root = _repository_root(__file__)
    pocket = derive_definition(root / "tests/data/brd4_3mxf/3MXF.pdb", root / "tests/data/brd4_3mxf/fixture_25.json")
    # Residue1 lives at token2, while the native protein map is permuted.
    ctx = SimpleNamespace(protein_token_indices=tuple(np.roll(np.arange(2, 129), 17).tolist()),
        base_features={"residue_index": np.r_[0, 0, np.arange(1, 128)], "asym_id": np.ones(129)},
        identities={"protein_sequence": pocket["protein_sequence"]})
    mapped = map_protenix_pocket(ctx, pocket)
    assert mapped == tuple(r["sequence_position"] + 1 for r in pocket["residues"])
    ctx.identities["protein_sequence"] = "A" * 127
    with pytest.raises(ValueError, match="sequence"):
        map_protenix_pocket(ctx, pocket)
