"""Native assets for this study; no model is imported by the plan command."""

import hashlib
import json
from pathlib import Path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(tree):
    """Same numerical-leaf digest as the archived BRD4 study."""
    import equinox as eqx
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if eqx.is_array(leaf):
            array = np.asarray(leaf)
            digest.update(str(path).encode())
            digest.update(str((array.dtype, array.shape)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def load_prepared_context(directory, expected):
    """Read the archived safe JSON/NPZ context without regenerating features."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.core.contracts import (
        PreparedContext, contract_manifest, validate_context,
    )

    directory = Path(directory)
    for name in ("prepared_context.json", "prepared_context.npz"):
        if expected.get(name) != file_sha(directory / name):
            raise ValueError(f"Prepared context identity mismatch: {name}")
    metadata = json.loads((directory / "prepared_context.json").read_text())
    if metadata["contract"] != contract_manifest():
        raise ValueError("Archived context uses a different graph contract")

    def restore(array):
        value = jnp.asarray(array)
        if np.dtype(value.dtype) != array.dtype:
            raise ValueError("Context restoration would change array precision")
        return value

    with np.load(directory / "prepared_context.npz", allow_pickle=False) as arrays:
        def decode(node):
            if "array" in node:
                return restore(arrays[node["array"]])
            if "rng_array" in node:
                return jax.random.wrap_key_data(
                    restore(arrays[node["rng_array"]]), impl=node["impl"],
                )
            if "dict" in node:
                return {k: decode(v) for k, v in node["dict"].items()}
            if "tuple" in node:
                return tuple(map(decode, node["tuple"]))
            if "list" in node:
                return list(map(decode, node["list"]))
            if set(node) != {"scalar"}:
                raise ValueError("Unknown context codec node")
            return node["scalar"]

        context = PreparedContext(
            base_features=decode(metadata["base_features"]),
            native_features=decode(metadata["native_features"]),
            reference_geometry_mask=decode(metadata["reference_geometry_mask"]),
            ligand_atom_indices=tuple(metadata["ligand_atom_indices"]),
            ligand_token_indices=tuple(metadata["ligand_token_indices"]),
            protein_token_indices=tuple(metadata["protein_token_indices"]),
            atom_names=tuple(metadata["atom_names"]),
            geometry_mode=metadata["geometry_mode"], identities=metadata["identities"],
        )
    validate_context(context)
    return context


def open_native(assets, root):
    """Load the original frozen BRD4 pocket scorer, once per GPU bundle."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.core.losses import NativeContactLoss
    from mosaic.small_molecule.core.pocket import derive_definition, map_protenix_pocket
    from mosaic.small_molecule.core.protenix import FrozenProtenixTiny

    if not any(d.platform == "gpu" for d in jax.devices()) or jax.config.jax_enable_x64:
        raise RuntimeError("Native campaign requires an allocated FP32 JAX GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    context = load_prepared_context(assets["context_dir"], assets["context_files"])
    if context.n_atoms != 25:
        raise ValueError("This study requires exactly 25 occupied graph slots")
    model = FrozenProtenixTiny.load(cache_dir=assets["model_cache"])
    actual = (array_digest(model), array_digest(context))
    expected = (assets["model_digest"], assets["context_digest"])
    if actual != expected:
        raise ValueError("Native model/context differs from the archived BRD4 study")
    pocket = derive_definition(
        root / "tests/data/brd4_3mxf/3MXF.pdb",
        root / "tests/data/brd4_3mxf/fixture_25.json",
    )
    tokens = map_protenix_pocket(context, pocket)
    native = model.build_loss(context=context, loss=NativeContactLoss(8.0, tokens))

    @eqx.filter_jit
    def compact(graph, *, key):
        value, aux = native(graph, key=key)
        aux = dict(aux)
        finite = jnp.asarray(True)
        for name in ("distogram_logits", "trunk_s", "trunk_z"):
            finite &= jnp.all(jnp.isfinite(aux.pop(name)))
        return jnp.where(finite, value, jnp.nan), aux

    def finish():
        if (array_digest(model), array_digest(context)) != expected:
            raise RuntimeError("Frozen native model/context changed during bundle")
        return {"model_digest": expected[0], "context_digest": expected[1], "passed": True}

    return compact, {
        "model": model.identities, "model_digest": actual[0],
        "context_digest": actual[1], "pocket_token_indices": list(tokens),
        "objective": "BRD4 pocket mean ligand best contact, cutoff8A; higher is better",
        "score_is_binding_affinity": False,
    }, finish


def mock_native():
    """Differentiable CPU fixture. Its scores are never native evidence."""
    import jax.numpy as jnp

    def loss(graph, *, key):
        # Both atom and bond paths are informative, including at simplex corners.
        score = jnp.mean(graph.X[:, 0] + .3 * graph.X[:, 1])
        score += .1 * jnp.mean(graph.E_upper[:, 1] - graph.E_upper[:, 0])
        return -score, {"contact_score": score}

    return loss, {"model": "mock_cpu_fixture", "native": False}, lambda: {"mock": True}
