"""Resident, frozen Tiny input/trunk/distogram adapter for fixed relaxed graphs.

The protein-facing Mosaic model remains unchanged. This interface never builds
``StructureModelOutput`` and never samples coordinates or calls confidence.
"""

from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from mosaic.common import LossTerm

from .contracts import MolecularTrunkOutput, PreparedContext
from .features import (
    ASSET_REVISION, ASSET_SHA256, MODEL_ALIAS, bootstrap_native_assets,
    inject_graph, native_cache_dir,
)
from .native_hook import HOOK_VERSION, UPSTREAM_REVISION, with_geometry_hook
from .state import RelaxedGraph


def native_distogram_bins():
    """64 native class upper edges, not invented equally spaced class centers.

    Native DistogramLoss labels a distance by the number of 63 boundaries below
    it. Selecting upper edges strictly below 8 A therefore includes only whole
    classes below the cutoff; the straddling class is conservatively excluded.
    The final +inf edge is metadata, not a generated model quantity.
    """
    return jnp.concatenate((jnp.linspace(2.3125, 21.6875, 63), jnp.array([jnp.inf])))


def _check_recycling(recycling_steps):
    if isinstance(recycling_steps, bool) or not isinstance(recycling_steps, Integral) or recycling_steps != 1:
        raise ValueError("Stage 0 verifies exactly one final native trunk iteration; recycling_steps must be 1")


def _native_trunk(model, features, key):
    # Reuse Mosaic's existing manual recycling semantics. It detaches previous
    # recycle state and differentiates the final trunk iteration. With one cycle
    # no previous graph-dependent state or stale input embedding is reused.
    from mosaic.losses.protenix import get_trunk_state
    return get_trunk_state(
        model=model, features=features, initial_recycling_state=None,
        recycling_steps=1, key=key,
    )


class FrozenProtenixTiny(eqx.Module):
    """Load once, sharing frozen weights between native and geometry-off views.

    Differentiate only a ``RelaxedGraph`` argument, holding this model and the
    prepared context fixed. No optimizer or parameter mutation is provided.
    The class is an Equinox pytree, matching Mosaic's resident model convention.
    """

    native_model: Any
    model: Any
    identities: dict[str, str] = eqx.field(static=True)

    @classmethod
    def load(cls, *, cache_dir=None):
        """Authenticate staged source/data/weights; absent assets fail offline.

        All five assets must exist with the pinned hashes BEFORE native loading,
        so its implicit data/model download branches are never reachable. This
        is intentionally a local-cache loader; staging is a separate HPC action.
        """
        bootstrap_native_assets(cache_dir, include_model=True)
        import protenix.backend as backend
        native_cache = native_cache_dir(cache_dir)
        backend._CACHE_DIR = str(native_cache)
        model = backend.load_model(str(native_cache / MODEL_ALIAS))
        # Native MSA sampling uses a training flag. Evaluation keeps its supported
        # test cutoff while input derivatives remain enabled. There is no torch
        # no_grad or stop_gradient on the mutable element/bond channels.
        model = eqx.tree_at(lambda m: m.msa_module.training, model, False)
        return cls(
            native_model=model, model=with_geometry_hook(model),
            identities={
                "model_alias": MODEL_ALIAS, "upstream_revision": UPSTREAM_REVISION,
                "asset_revision": ASSET_REVISION, "hook_version": HOOK_VERSION,
                "model_eqx_sha256": ASSET_SHA256[MODEL_ALIAS + ".eqx"],
                "model_skeleton_sha256": ASSET_SHA256[MODEL_ALIAS + ".skeleton.pkl"],
                "evaluation_mode": "native MSA training=False; frozen pretrained weights",
            },
        )

    @eqx.filter_jit
    def forward_trunk(self, *, context: PreparedContext, graph: RelaxedGraph, key, recycling_steps=1):
        _check_recycling(recycling_steps)
        features = inject_graph(context, graph)
        _, trunk = _native_trunk(self.model, features, key)
        logits = self.model.distogram_head(trunk.z)
        return MolecularTrunkOutput(
            distogram_logits=logits, distogram_bins=native_distogram_bins(),
            ligand_token_indices=context.ligand_token_indices,
            protein_token_indices=context.protein_token_indices, trunk_state=trunk,
        )

    @eqx.filter_jit
    def forward_native_trunk(self, *, context: PreparedContext, key, recycling_steps=1):
        """Ordinary hard/conformer reference, explicitly outside the graph path."""
        _check_recycling(recycling_steps)
        if context.native_features is None:
            raise ValueError("No original native hard-input baseline was retained")
        _, trunk = _native_trunk(self.native_model, context.native_features, key)
        return MolecularTrunkOutput(
            distogram_logits=self.native_model.distogram_head(trunk.z),
            distogram_bins=native_distogram_bins(),
            ligand_token_indices=context.ligand_token_indices,
            protein_token_indices=context.protein_token_indices, trunk_state=trunk,
        )

    def build_loss(self, *, context, loss, recycling_steps=1):
        _check_recycling(recycling_steps)
        return GraphProtenixLoss(self, context, loss)

    def confidence_output(self, *, context, graph, key, sampling_steps=2):
        """Explicit Stage 1 boundary: geometry-off ligand frames are invalid.

        Stage 0 supplies no claimed ligand PAE, atom37, backbone or sampled
        structure output. A future native confidence gate must establish its
        own truthful frames and gradient path before exposing those fields.
        """
        raise NotImplementedError(
            "The geometry-off graph has no valid ligand reference frames. "
            "Native ligand confidence/PAE is unsupported pending the Stage 1 gate; "
            "use the specified native target-contact fallback."
        )


class GraphProtenixLoss(LossTerm):
    """Mosaic-style scalar+aux callable with a graph rather than sequence leaf."""

    model: FrozenProtenixTiny
    context: PreparedContext
    loss: Any

    def __call__(self, graph: RelaxedGraph, *, key):
        output = self.model.forward_trunk(context=self.context, graph=graph, key=key)
        value, aux = self.loss(graph, output, key=key)
        # The strict evaluator checks generated native output before transforming
        # any derivative. Do not include the intentionally infinite bin edge in
        # this generated-quantity finiteness contract.
        metrics = dict(aux) if isinstance(aux, dict) else {"objective": aux}
        return value, metrics | {
            "distogram_logits": output.distogram_logits,
            "trunk_s": output.trunk_state.s,
            "trunk_z": output.trunk_state.z,
        }


def ligand_pae_gate_prerequisites(context: PreparedContext) -> dict[str, Any]:
    """Detached, truthful Stage 1 handoff; this is not a completed PAE test."""
    import numpy as np
    frames = np.asarray(context.base_features["has_frame"])[list(context.ligand_token_indices)]
    valid_count = int(np.count_nonzero(frames))
    return {
        "supported": False,
        "valid_ligand_frame_count": valid_count,
        "reason": "Geometry-off ligand frames are invalid; native confidence path has not been validated",
        "fallback": "NativeContactLoss with complete target context",
    }
