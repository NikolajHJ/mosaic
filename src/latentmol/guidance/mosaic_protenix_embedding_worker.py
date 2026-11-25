"""Resident native JAX VJP server for projected chemical input embeddings.

This replaces only the ligand chemical linear contributions. The native atom
attention, pair initialization, geometry ablation, trunk and contact objective
remain identical to the existing probability server. No diffusion is called.
"""
from __future__ import annotations

import contextlib
import json
import math
import sys
import traceback

from latentmol.guidance.mosaic_protenix_worker import NativeServer

PROTOCOL = "mosaic-protenix-chemical-embedding-vjp-v1"
HEADS = ("atom_element", "atom_charge", "pair_presence")


def projected_chemical_bank(model, atomic_numbers, formal_charges):
    """Derive prototypes by evaluating the frozen native linear maps."""
    import jax.numpy as jnp
    import numpy as np
    encoder = model.input_embedder.atom_attention_encoder.native
    keys = encoder.input_feature_keys_ordered
    widths = encoder.input_feature
    if "ref_element" not in keys or widths["ref_element"] != 128:
        raise ValueError("Unsupported native element input layout")
    offset = sum(widths[k] for k in keys[:keys.index("ref_element")])
    basis = jnp.zeros((len(atomic_numbers), sum(widths[k] for k in keys)), dtype=jnp.float32)
    basis = basis.at[jnp.arange(len(atomic_numbers)), offset + jnp.asarray(atomic_numbers) - 1].set(1.)
    element = encoder.linear_no_bias_f(basis)
    charge = encoder.linear_no_bias_ref_charge(jnp.arcsinh(jnp.asarray(formal_charges, dtype=jnp.float32))[:, None])
    pair = model.linear_no_bias_token_bond(jnp.asarray([[0.], [1.]], dtype=jnp.float32))
    return {key: dict(kind=key, width=int(value.shape[-1]), prototypes=np.asarray(value).tolist(),
                     labels=list(labels)) for key, value, labels in zip(HEADS, (element, charge, pair),
                     (atomic_numbers, formal_charges, (0, 1)))}


def injected_model_and_features(model, context, element, charge, pair):
    """Add learned arrays at the two native linear consumers, before attention.

    Zeroing ligand raw chemicals avoids retaining the old graph contribution.
    The additive wrapper adds before ref_mask and the original nonlinear atom
    attention. Native atom-name contributions and all protein rows are retained.
    """
    import equinox as eqx
    import jax.numpy as jnp

    class AdditiveLinear(eqx.Module):
        native: object
        contribution: object

        def __call__(self, value):
            return self.native(value) + self.contribution

    ai = jnp.asarray(context.ligand_atom_indices, dtype=jnp.int32)
    ti = jnp.asarray(context.ligand_token_indices, dtype=jnp.int32)
    f = context.base_features
    from mosaic.small_molecule.native_hook import GEOMETRY_FEATURE
    atom_contribution = jnp.zeros((f["ref_element"].shape[0], element.shape[-1]), dtype=element.dtype).at[ai].set(element + charge)
    pair = (pair + jnp.swapaxes(pair, 0, 1)) * .5
    pair = pair.at[jnp.diag_indices(context.n_atoms)].set(0.)
    pair_contribution = jnp.zeros((*f["token_bonds"].shape, pair.shape[-1]), dtype=pair.dtype)
    pair_contribution = pair_contribution.at[ti[:, None], ti[None, :]].set(pair)
    features = f | dict(ref_element=f["ref_element"].at[ai].set(0.),
        ref_charge=f["ref_charge"].at[ai].set(0.), ref_pos=f["ref_pos"].at[ai].set(0.),
        token_bonds=f["token_bonds"].at[ti[:, None], ti[None, :]].set(0.),
        **{GEOMETRY_FEATURE: context.reference_geometry_mask})
    atom_linear = model.input_embedder.atom_attention_encoder.native.linear_no_bias_f
    cloned = eqx.tree_at(lambda m: (m.input_embedder.atom_attention_encoder.native.linear_no_bias_f,
                                     m.linear_no_bias_token_bond), model,
        (AdditiveLinear(atom_linear, atom_contribution), AdditiveLinear(model.linear_no_bias_token_bond, pair_contribution)))
    return cloned, features


class EmbeddingServer(NativeServer):
    def __init__(self, request):
        super().__init__(request)
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from mosaic.small_molecule.protenix import _native_trunk, native_distogram_bins
        from mosaic.small_molecule.contracts import MolecularTrunkOutput
        from mosaic.small_molecule.losses import NativeContactLoss
        from mosaic.small_molecule.pocket import map_protenix_pocket
        from latentmol.guidance.mosaic_protenix_worker import _read_authenticated
        config = request["config"]
        pocket, _ = _read_authenticated(config["pocket_path"], config.get("pocket_sha256"))
        loss = NativeContactLoss(8., map_protenix_pocket(self.context, pocket))
        self.projection_bank = projected_chemical_bank(self.model.model, self.atomic_numbers, self.formal_charges)

        @eqx.filter_jit
        def evaluate(model, context, key, element, charge, pair):
            def objective(e, c, b):
                injected, features = injected_model_and_features(model.model, context, e, c, b)
                _, trunk = _native_trunk(injected, features, key)
                logits = injected.distogram_head(trunk.z)
                output = MolecularTrunkOutput(distogram_logits=logits, distogram_bins=native_distogram_bins(),
                    ligand_token_indices=context.ligand_token_indices, protein_token_indices=context.protein_token_indices,
                    trunk_state=trunk)
                value, aux = loss(None, output, key=key)
                finite = jnp.isfinite(logits).all() & jnp.isfinite(trunk.s).all() & jnp.isfinite(trunk.z).all()
                return value, (aux, finite)
            return jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True)(element, charge, pair)
        self._embedding_evaluate = evaluate
        self.identity = dict(self.identity, protocol=PROTOCOL,
            feature_map="learned_native_element_charge_and_pair_presence_contributions",
            chemical_boundary="native atom linear_f plus charge before ref_mask/attention; native token-bond linear before trunk",
            pair_policy="symmetric_mean_zero_diagonal", learned_output_constraints="finite_real_arrays")

    def evaluate(self, request):
        import numpy as np
        import jax.numpy as jnp
        n = self.context.n_atoms
        values = [np.asarray(request["arrays"][key], dtype=np.float32) for key in HEADS]
        for key, value in zip(HEADS, values):
            width = self.projection_bank[key]["width"]
            shape = (n, n, width) if key == "pair_presence" else (n, width)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Incompatible/nonfinite chemical embedding: " + key)
        (value, (aux, finite)), grads = self._embedding_evaluate(self.model, self.context, self.key, *(jnp.asarray(v) for v in values))
        value = float(value)
        gradients = [np.asarray(v) for v in grads]
        if not bool(finite) or not math.isfinite(value) or any(not np.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Nonfinite native trunk/objective/embedding derivative")
        return dict(loss=value, gradients={key: g.tolist() for key, g in zip(HEADS, gradients)},
                    aux={k: np.asarray(v).item() for k, v in aux.items()},
                    gradient_norms={key: float(np.linalg.norm(g.astype(np.float64))) for key, g in zip(HEADS, gradients)})

    def evaluate_graph(self, request):
        """Report native raw-graph versus exact projected-prototype score parity."""
        import numpy as np
        native = super().evaluate(request)
        a, c, b = (np.asarray(request["arrays"][key], dtype=np.float32) for key in ("atoms", "charges", "bonds"))
        bank = self.projection_bank
        projections = {"atom_element": a @ np.asarray(bank["atom_element"]["prototypes"], dtype=np.float32),
            "atom_charge": c @ np.asarray(bank["atom_charge"]["prototypes"], dtype=np.float32),
            "pair_presence": (1 - b[..., 0])[..., None] * np.asarray(bank["pair_presence"]["prototypes"], dtype=np.float32)[1]}
        projected = self.evaluate(dict(arrays=projections))
        return dict(native_loss=native["loss"], projected_loss=projected["loss"],
                    absolute_difference=abs(native["loss"] - projected["loss"]),
                    scope="hard-label exact projection smoke; mixtures of nonlinear charge transforms differ",
                    aux=projected["aux"])


def main():
    wire = sys.stdout
    server = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("protocol") != PROTOCOL:
                raise ValueError("Unknown embedding worker protocol")
            with contextlib.redirect_stdout(sys.stderr):
                op = request.get("op")
                if op == "init":
                    if server is not None:
                        raise ValueError("Worker already initialized")
                    server = EmbeddingServer(request)
                    result = dict(identity=server.identity, n_atoms=server.context.n_atoms,
                                  devices=server.devices, projection_bank=server.projection_bank)
                elif op == "evaluate":
                    result = server.evaluate(request)
                elif op == "evaluate_graph":
                    result = server.evaluate_graph(request)
                elif op == "evaluate_probabilities":
                    result = NativeServer.evaluate(server, request)
                elif op == "close":
                    result = server.close()
                else:
                    raise ValueError("Unknown embedding worker operation")
            wire.write(json.dumps(dict(result, status="ok", protocol=PROTOCOL,
                                       request_id=request["request_id"]), allow_nan=False) + "\n")
            wire.flush()
            if op == "close":
                return
        except BaseException as e:
            traceback.print_exc(file=sys.stderr)
            wire.write(json.dumps(dict(status="error", protocol=PROTOCOL,
                request_id=request.get("request_id"), error=type(e).__name__ + ": " + str(e))) + "\n")
            wire.flush()
            return


if __name__ == "__main__":
    main()
