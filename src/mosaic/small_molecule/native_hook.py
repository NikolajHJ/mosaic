"""Versioned geometry ablation at the frozen native atom-input boundary.

The upstream encoder combines reference geometry with identity before nonlinear
attention. This small inference-input specialization preserves its trained
layers and calls its native packing, attention and atom aggregation helpers.
Only reference-position and reference-pair contributions are gated; occupancy,
element/name conditioning and protein reference geometry remain native.
"""

import hashlib
import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

UPSTREAM_REVISION = "97cf8172f61829681582f06de7052485bc94eae4"
HOOK_VERSION = "mosaic-protenij-reference-geometry-v1"
GEOMETRY_FEATURE = "_mosaic_reference_geometry_mask"

# Guard actual executed source, rather than the upstream package's generic
# 0.1.0 metadata. No runtime patch is applied to the dependency checkout.
SOURCE_SHA256 = {
    "protenij.py": "37f337839034afd2a814c1ed6b76867b67918f73dd56cdd365e67b8fed882a6d",
    "backend.py": "fb0bf601239eef711081d4e254f06208bcb36002d4e57a50b5a54135948fd12c",
    "atom_padding.py": "7332a868c47d2f1db75b4f9d6bb693958cd719474058698a91e6086daf5332a2",
    "token_padding.py": "cdf48c62dba90fb52cb73cae7f9c29ccc6a918691e3ce68d1f8ba9c6e9a52233",
    "data/featurizer.py": "63423d72f38777b449614d1d709b6fa87486639ee0c2bc103069311cce58e288",
    "data/json_parser.py": "3c4503540322049ad50231996a4af8673df2bf8331664b7a32cac0df8186c93c",
    "data/json_to_feature.py": "08180ba03606e0d8e86a064f472342fa272cbda55edb6b9a98c0382c54f5ca94",
    "data/constants.py": "df284ef5e91ff4de1ce70006940e259e64ff1d39044d67fa8d9439fd3db5e9ff",
    "data/tokenizer.py": "ad70ef6e5a0c976132ee1e96af71b8ba0d27b92ab9d9284a1c68326ca8d1a265",
    "data/ccd.py": "94f6c72b35a792156026bbde1ccb22c8452c9be3adaf506194d823a8381379ed",
    "data/parser.py": "46dcffac721936a2e52eb973430f6500448e6113675768954ae92758e1a05c0d",
    "data/msa_featurizer.py": "96d72b1e03b3ba923d5cd7e0980676faa451d3590f9b7fb37027a3bd06c7371a",
    "configs/configs_base.py": "734ea1a757662bb2ba66dd0fffb7d5a0e02e8bdc927a561ccb58e8f634d7761b",
    "configs/configs_data.py": "0a85e9d690363d0f979eec036febfc3118d13f440b66c642a9c4dc85ef117296",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_native_source(package_dir: Path | None = None) -> dict[str, str]:
    """Fail closed when the audited consumer code has changed (host only)."""
    if package_dir is None:
        spec = importlib.util.find_spec("protenix")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("Install Mosaic's pinned Protenij dependency first")
        package_dir = Path(next(iter(spec.submodule_search_locations)))
    actual = {}
    for relative, expected in SOURCE_SHA256.items():
        path = package_dir / relative
        if not path.is_file():
            raise RuntimeError(f"Missing audited native source: {path}")
        actual[relative] = file_sha256(path)
        if actual[relative] != expected:
            raise RuntimeError(
                f"Unaudited Protenij source {relative}: {actual[relative]}; "
                f"Stage 0 requires revision {UPSTREAM_REVISION}"
            )
    return actual


class GeometryMaskedAtomInput(eqx.Module):
    """Native input atom encoder with a separate reference-geometry mask.

    This wraps only ``input_embedder.atom_attention_encoder``. It deliberately
    refuses noisy-coordinate/diffusion calls: Stage 0 runs the trunk alone and
    has no validated ligand frame/confidence contract.
    """

    native: eqx.Module

    def __call__(self, input_feature_dict, r_l=None, s=None, z=None):
        if self.native.has_coords or r_l is not None or s is not None or z is not None:
            raise ValueError("The Stage 0 geometry hook supports native input embedding only")
        if GEOMETRY_FEATURE not in input_feature_dict:
            # Protein-only use of a hooked model remains exactly the native API.
            return self.native(input_feature_dict=input_feature_dict)

        from protenix.atom_padding import mask_atom_features, mask_atom_values
        from protenix.protenij import average_over_atoms, rearrange_qk_to_dense_trunk

        encoder = self.native
        atom_mask = input_feature_dict.get("atom_pad_mask")
        features = mask_atom_features(
            input_feature_dict,
            (*encoder.input_feature_keys_ordered, "ref_pos", "ref_charge",
             "ref_mask", "ref_space_uid", "atom_to_token_idx"),
        )
        geometry = mask_atom_values(
            features[GEOMETRY_FEATURE], atom_mask, axis=0
        )
        atom_count = features["ref_pos"].shape[-2]
        batch_shape = features["ref_pos"].shape[:-2]

        # Gate the projected position contribution, not all atom conditioning.
        # ``ref_mask`` stays native: zeroing it would also erase the element.
        position = encoder.linear_no_bias_ref_pos(features["ref_pos"])
        c_l = position * geometry[..., None]
        c_l += encoder.linear_no_bias_ref_charge(
            jnp.arcsinh(features["ref_charge"]).reshape(*batch_shape, atom_count, 1)
        )
        c_l += encoder.linear_no_bias_f(jnp.concatenate([
            features[name].reshape(*batch_shape, atom_count, encoder.input_feature[name])
            for name in encoder.input_feature_keys_ordered
        ], axis=-1))
        c_l *= features["ref_mask"].reshape(*batch_shape, atom_count, 1)

        q, k, padding = rearrange_qk_to_dense_trunk(
            q=[features["ref_pos"], features["ref_space_uid"], geometry],
            k=[features["ref_pos"], features["ref_space_uid"], geometry],
            dim_q=[-2, -1, -1], dim_k=[-2, -1, -1],
            n_queries=encoder.n_queries, n_keys=encoder.n_keys,
            compute_mask=True,
        )
        displacement = q[0][..., None, :] - k[0][..., None, :, :]
        same_space = (q[1][..., None].astype(jnp.int32)
                      == k[1][..., None, :].astype(jnp.int32))[..., None]
        geometry_pair = (q[2][..., None] * k[2][..., None, :])[..., None]

        # Inverse-distance and validity embeddings can be nonzero even at zero
        # coordinates. Gate all three geometry terms before identity conditioning
        # and its nonlinear MLP, for EVERY pair involving a ligand atom.
        p_lm = encoder.linear_no_bias_d(displacement) * same_space
        p_lm *= padding["mask_trunked"][..., None]
        p_lm += encoder.linear_no_bias_invd(
            1 / (1 + jnp.square(displacement).sum(axis=-1, keepdims=True))
        ) * same_space
        p_lm += encoder.linear_no_bias_v(same_space)
        p_lm *= geometry_pair

        c_q, c_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l, k=c_l, dim_q=-2, dim_k=-2,
            n_queries=encoder.n_queries, n_keys=encoder.n_keys,
            compute_mask=False,
        )
        p_lm += encoder.linear_no_bias_cl(jax.nn.relu(c_q[..., None, :]))
        p_lm += encoder.linear_no_bias_cm(jax.nn.relu(c_k[..., None, :, :]))
        p_lm += encoder.small_mlp(p_lm)
        q_l = encoder.atom_transformer(c_l, c_l, p_lm, atom_mask=atom_mask)
        a = average_over_atoms(
            x_atom=jax.nn.relu(encoder.linear_no_bias_q(q_l)),
            atom_to_token_idx=features["atom_to_token_idx"],
            n_res=features["residue_index"].shape[-1], atom_mask=atom_mask,
        )
        return a, q_l, c_l, p_lm


def with_geometry_hook(model):
    """Share all frozen weights with a clone; never mutate the native model."""
    encoder = model.input_embedder.atom_attention_encoder
    if isinstance(encoder, GeometryMaskedAtomInput):
        return model
    if encoder.has_coords:
        raise ValueError("Unexpected native input encoder: noisy coordinates required")
    return eqx.tree_at(
        lambda m: m.input_embedder.atom_attention_encoder,
        model, GeometryMaskedAtomInput(encoder),
    )
