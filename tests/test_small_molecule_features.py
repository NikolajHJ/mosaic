"""Synthetic feature-boundary tests; real native execution lives separately.

Authored offline. These are numerical tests and must run on allocated DTU HPC.
"""

import copy
from pathlib import Path
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.contracts import PreparedContext, validate_context
from mosaic.small_molecule.features import NATIVE_ELEMENT_INDICES, inject_graph
from mosaic.small_molecule.native_hook import (
    GeometryMaskedAtomInput,
    SOURCE_SHA256,
    verify_native_source,
)
from mosaic.small_molecule.state import RelaxedGraph, graph_from_hard, mix_with_uniform


def synthetic_context():
    """Deliberately permuted ligand atom AND token indices, with protein atoms."""
    graph = graph_from_hard([0, 1, 2], [1, 2, 0])
    ligand_atoms, ligand_tokens, protein_tokens = (4, 0, 6), (3, 1, 4), (0, 2)
    atom_mapping = np.array([1, 0, 2, 0, 3, 2, 4], dtype=np.int32)
    element = np.zeros((7, 128), dtype=np.int32)
    element[:, 5] = 1
    element[0] = np.eye(128, dtype=np.int32)[6]
    element[6] = np.eye(128, dtype=np.int32)[7]
    bonds = np.zeros((5, 5), dtype=np.float32)
    bonds[0, 2] = bonds[2, 0] = 0.7  # detects accidentally rewriting protein blocks
    bonds[3, 1] = bonds[1, 3] = 1
    bonds[3, 4] = bonds[4, 3] = 1
    base = {
        "ref_element": jnp.asarray(element),  # integer native hard one-hot regression
        "ref_pos": jnp.asarray(np.arange(21, dtype=np.float32).reshape(7, 3)),
        "ref_mask": jnp.ones(7, dtype=jnp.int32),
        "ref_charge": jnp.zeros(7, dtype=jnp.int32),
        "ref_atom_name_chars": jnp.zeros((7, 4, 64)),
        "ref_space_uid": jnp.asarray([2, 0, 1, 0, 2, 1, 2]),
        "atom_to_token_idx": jnp.asarray(atom_mapping),
        "atom_to_tokatom_idx": jnp.zeros(7, dtype=jnp.int32),
        "is_protein": jnp.asarray([0, 1, 1, 1, 0, 1, 0]),
        "is_ligand": jnp.asarray([1, 0, 0, 0, 1, 0, 1]),
        "restype": jax.nn.one_hot(jnp.array([0, 20, 1, 20, 20]), 32),
        "profile": jax.nn.one_hot(jnp.array([0, 20, 1, 20, 20]), 32),
        "deletion_mean": jnp.zeros(5),
        "token_bonds": jnp.asarray(bonds),
        "token_index": jnp.arange(5),
        "residue_index": jnp.arange(5),
        "entity_id": jnp.asarray([0, 2, 0, 2, 2]),
        "asym_id": jnp.asarray([0, 2, 0, 2, 2]),
        "sym_id": jnp.zeros(5, dtype=jnp.int32),
        "has_frame": jnp.asarray([1, 0, 1, 0, 0]),
        "frame_atom_index": jnp.asarray(
            [[1, 3, 1], [-1, -1, -1], [2, 5, 2], [-1, -1, -1], [-1, -1, -1]]
        ),
        "atom_pad_mask": jnp.ones(7, dtype=bool),
        "token_pad_mask": jnp.ones(5, dtype=bool),
    }
    geometry = jnp.ones(7).at[jnp.asarray(ligand_atoms)].set(0)
    context = PreparedContext(
        base_features=base,
        ligand_atom_indices=ligand_atoms,
        ligand_token_indices=ligand_tokens,
        protein_token_indices=protein_tokens,
        reference_geometry_mask=geometry,
        atom_names=("Q000", "Q001", "Q002"),
        native_features=copy.deepcopy(base),
        identities={"fixture": "synthetic-interleaved-v1"},
    )
    validate_context(context)
    return context, graph


def test_integer_native_element_channel_does_not_truncate_relaxed_values_or_gradients():
    context, hard = synthetic_context()
    graph = mix_with_uniform(hard, 0.3)
    features = inject_graph(context, graph)
    assert np.issubdtype(features["ref_element"].dtype, np.floating)
    observed = features["ref_element"][jnp.asarray(context.ligand_atom_indices)][
        :, jnp.asarray(NATIVE_ELEMENT_INDICES)
    ]
    np.testing.assert_allclose(observed, graph.X, rtol=0, atol=0)
    weights = jnp.arange(128, dtype=jnp.float32)

    def scalar(state):
        return jnp.sum(inject_graph(context, state)["ref_element"] * weights)

    gradient = jax.grad(scalar)(graph)
    np.testing.assert_array_equal(
        gradient.X, np.broadcast_to(NATIVE_ELEMENT_INDICES, graph.X.shape)
    )
    # C -> N in an interior row must retain a derivative of 6-5 = 1.
    np.testing.assert_allclose(gradient.X[0, 1] - gradient.X[0, 0], 1, rtol=0, atol=0)


def test_injection_preserves_protein_and_static_metadata_without_mutating_context():
    context, graph = synthetic_context()
    before = {
        name: np.array(value, copy=True)
        for name, value in context.base_features.items()
    }
    features = inject_graph(context, mix_with_uniform(graph, 0.2))
    protein_atoms = np.flatnonzero(before["is_protein"])
    np.testing.assert_array_equal(
        features["ref_element"][protein_atoms], before["ref_element"][protein_atoms]
    )
    np.testing.assert_array_equal(
        features["ref_pos"][protein_atoms], before["ref_pos"][protein_atoms]
    )
    proteins = np.asarray(context.protein_token_indices, dtype=np.int32)
    np.testing.assert_array_equal(
        features["token_bonds"][np.ix_(proteins, proteins)],
        before["token_bonds"][np.ix_(proteins, proteins)],
    )
    for name, value in before.items():
        np.testing.assert_array_equal(context.base_features[name], value)
        if name not in ("ref_element", "token_bonds", "ref_pos"):
            np.testing.assert_array_equal(features[name], value)
    assert np.all(
        np.asarray(features["ref_mask"])[list(context.ligand_atom_indices)] == 1
    )
    assert np.all(
        np.asarray(features["atom_pad_mask"])[list(context.ligand_atom_indices)]
    )


def test_shared_pair_gradient_sums_both_native_orientations():
    context, graph = synthetic_context()
    a, b = context.ligand_token_indices[:2]

    def scalar(state):
        bonds = inject_graph(context, state)["token_bonds"]
        return 3 * bonds[a, b] + 5 * bonds[b, a]

    gradient = jax.grad(scalar)(mix_with_uniform(graph, 0.2))
    np.testing.assert_allclose(gradient.E_upper[0, 0], -8, rtol=0, atol=0)
    np.testing.assert_array_equal(gradient.E_upper[0, 1:], 0)


def test_bond_order_exchanges_are_exact_null_directions():
    context, hard = synthetic_context()
    graph = mix_with_uniform(hard, 0.4)
    shifted = RelaxedGraph(
        graph.X, graph.E_upper.at[0, 1].add(-0.01).at[0, 4].add(0.01)
    )
    np.testing.assert_array_equal(
        inject_graph(context, graph)["token_bonds"],
        inject_graph(context, shifted)["token_bonds"],
    )


def test_conformer_substitution_and_a_b_a_have_no_feature_cache():
    context, hard = synthetic_context()
    graph = mix_with_uniform(hard, 0.2)
    moved = eqx.tree_at(
        lambda c: c.base_features["ref_pos"],
        context,
        context.base_features["ref_pos"]
        .at[jnp.asarray(context.ligand_atom_indices)]
        .set(900),
    )
    a0 = inject_graph(context, graph)
    b = inject_graph(
        context,
        RelaxedGraph(graph.X.at[0, 0].add(-0.02).at[0, 1].add(0.02), graph.E_upper),
    )
    a1 = inject_graph(context, graph)
    substituted = inject_graph(moved, graph)
    assert not np.array_equal(a0["ref_element"], b["ref_element"])
    for name in a0:
        np.testing.assert_array_equal(a0[name], a1[name])
        np.testing.assert_array_equal(a0[name], substituted[name])


class _Projection(eqx.Module):
    scale: float

    def __call__(self, values):
        return jnp.sum(values, axis=-1, keepdims=True) * self.scale


class _Zero(eqx.Module):
    def __call__(self, values):
        return jnp.zeros_like(values)


class _Attention(eqx.Module):
    def __call__(self, q, c, p, atom_mask=None):
        return q + p.sum(axis=-2)[0]


class _Encoder(eqx.Module):
    has_coords: bool = False
    input_feature_keys_ordered: tuple = ("ref_element",)
    input_feature: dict = eqx.field(default_factory=lambda: {"ref_element": 128})
    n_queries: int = 7
    n_keys: int = 7
    linear_no_bias_ref_pos: _Projection = eqx.field(
        default_factory=lambda: _Projection(1)
    )
    linear_no_bias_ref_charge: _Projection = eqx.field(
        default_factory=lambda: _Projection(1)
    )
    linear_no_bias_f: _Projection = eqx.field(default_factory=lambda: _Projection(2))
    linear_no_bias_d: _Projection = eqx.field(default_factory=lambda: _Projection(3))
    linear_no_bias_invd: _Projection = eqx.field(
        default_factory=lambda: _Projection(10)
    )
    linear_no_bias_v: _Projection = eqx.field(default_factory=lambda: _Projection(20))
    linear_no_bias_cl: _Projection = eqx.field(default_factory=lambda: _Projection(4))
    linear_no_bias_cm: _Projection = eqx.field(default_factory=lambda: _Projection(5))
    linear_no_bias_q: _Projection = eqx.field(default_factory=lambda: _Projection(1))
    small_mlp: _Zero = eqx.field(default_factory=_Zero)
    atom_transformer: _Attention = eqx.field(default_factory=_Attention)


def test_geometry_hook_masks_zero_distance_constants_before_identity_attention(
    monkeypatch,
):
    """A geometry=0 mask must suppress invd(0)=1 AND validity projections."""

    def pack(q, k, **kwargs):
        if isinstance(q, list):
            return (
                [v[None] for v in q],
                [v[None] for v in k],
                {"mask_trunked": jnp.ones((1, 7, 7))},
            )
        return q[None], k[None], {}

    upstream = types.ModuleType("protenix.protenij")
    upstream.rearrange_qk_to_dense_trunk = pack
    upstream.average_over_atoms = lambda x_atom, **kwargs: x_atom
    atom_padding = types.ModuleType("protenix.atom_padding")
    atom_padding.mask_atom_features = lambda features, names: features
    atom_padding.mask_atom_values = lambda values, mask, **kwargs: values
    monkeypatch.setitem(__import__("sys").modules, "protenix.protenij", upstream)
    monkeypatch.setitem(
        __import__("sys").modules, "protenix.atom_padding", atom_padding
    )
    context, graph = synthetic_context()
    features = inject_graph(context, graph)
    hook = GeometryMaskedAtomInput(_Encoder())
    _, _, single, pair = hook(features)
    ligand = np.asarray(context.ligand_atom_indices, dtype=np.int32)
    # Ligand identity survives with ref_mask=1, although all geometry is off.
    np.testing.assert_allclose(single[ligand], 2, rtol=0, atol=0)
    # Ligand-ligand pair conditioning contains cl(2)+cm(2)=18, WITHOUT the
    # otherwise nonzero invd(0)*10 + valid(1)*20 geometric constants.
    np.testing.assert_allclose(pair[0][np.ix_(ligand, ligand)], 18, rtol=0, atol=0)
    protein = np.flatnonzero(np.asarray(features["is_protein"]))
    # Native protein self-reference contribution remains 10+20=30.
    for atom in protein:
        np.testing.assert_allclose(
            pair[0, atom, atom, 0] - 9 * single[atom, 0], 30, rtol=0, atol=0
        )
    changed = features | {
        "ref_pos": features["ref_pos"].at[jnp.asarray(ligand)].set(999)
    }
    for actual, expected in zip(hook(changed), hook(features)):
        np.testing.assert_array_equal(actual, expected)


def test_native_source_guard_rejects_unaudited_source_before_import(tmp_path):
    for relative in SOURCE_SHA256:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unaudited", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unaudited Protenij source"):
        verify_native_source(Path(tmp_path))
