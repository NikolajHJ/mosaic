"""Resident JAX Protenix contact VJP server; no Torch import or diffusion.

Run in Mosaic's native environment. Native model/context features are injected
with the full declared VAE element vocabulary and expected formal charge. This
extends the public six-neutral graph wrapper by an explicit feature map, while
reusing the geometry-off native trunk and exact Mosaic NativeContactLoss.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback


PROTOCOL = "mosaic-protenix-probability-vjp-v1"


def _read_authenticated(path, expected=None):
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("Changed native context/pocket input: " + str(path))
    return json.loads(data), actual


def inject_full_schema(context, atoms, charges, bonds, atomic_numbers, formal_charges):
    """Pure JAX feature copy; bond order is deliberately collapsed to presence.

    The matrix input includes both orientations. Symmetrization's pullback gives
    half of each native orientation to each input; a shared symmetric decoder
    then sums the two contributions. Diagonal token bonds are forced absent.
    """
    import jax.numpy as jnp
    from mosaic.small_molecule.native_hook import GEOMETRY_FEATURE

    ai = jnp.asarray(context.ligand_atom_indices, dtype=jnp.int32)
    ti = jnp.asarray(context.ligand_token_indices, dtype=jnp.int32)
    ligand = jnp.zeros((context.n_atoms,128),dtype=atoms.dtype).at[:,jnp.asarray(atomic_numbers)-1].set(atoms)
    elements = jnp.asarray(context.base_features["ref_element"],dtype=atoms.dtype).at[ai].set(ligand)
    expected_charge = charges @ jnp.asarray(formal_charges,dtype=charges.dtype)
    native_charges = jnp.asarray(context.base_features["ref_charge"],dtype=charges.dtype).at[ai].set(expected_charge)
    presence = 1 - (bonds[...,0] + bonds[...,0].T)*.5
    presence = presence.at[jnp.diag_indices(context.n_atoms)].set(0)
    native_bonds = jnp.asarray(context.base_features["token_bonds"],dtype=bonds.dtype).at[ti[:,None],ti[None,:]].set(presence)
    positions = context.base_features["ref_pos"].at[ai].set(0)
    return context.base_features | dict(ref_element=elements,ref_charge=native_charges,
        token_bonds=native_bonds,ref_pos=positions,**{GEOMETRY_FEATURE:context.reference_geometry_mask})


class NativeServer:
    def __init__(self, request):
        config = request["config"]
        source = str(Path(config["mosaic_source"]).resolve())
        sys.path.insert(0,source)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import equinox as eqx
        from mosaic.small_molecule.protenix import FrozenProtenixTiny, _native_trunk, native_distogram_bins
        from mosaic.small_molecule.io import load_context
        from mosaic.small_molecule.contracts import MolecularTrunkOutput
        from mosaic.small_molecule.losses import NativeContactLoss
        from mosaic.small_molecule.pocket import map_protenix_pocket
        from mosaic.small_molecule.gradient_checks import array_digest

        if jax.config.x64_enabled:
            raise ValueError("Native Protenix worker requires FP32 JAX")
        jax.config.update("jax_default_matmul_precision","highest")
        self.devices = [dict(platform=d.platform,kind=d.device_kind,id=d.id) for d in jax.devices()]
        if config.get("require_gpu",True) and sum(d["platform"]=="gpu" for d in self.devices) != 1:
            raise ValueError("Native Protenix requires exactly one visible allocated GPU")
        self.atomic_numbers = tuple(request["atomic_numbers"])
        self.formal_charges = tuple(request["formal_charges"])
        if (not self.atomic_numbers or len(set(self.atomic_numbers))!=len(self.atomic_numbers)
                or any(type(n) is not int or not 1<=n<=128 for n in self.atomic_numbers)
                or len(set(self.formal_charges))!=len(self.formal_charges)
                or not self.formal_charges or any(type(q) is not int for q in self.formal_charges)):
            raise ValueError("Unsupported element/charge vocabulary")
        if request["bond_names"] != ["NONE","SINGLE","DOUBLE","TRIPLE","AROMATIC"]:
            raise ValueError("Unsupported bond vocabulary")
        ready, ready_sha = _read_authenticated(config["context_ready"],config.get("context_ready_sha256"))
        pocket, pocket_sha = _read_authenticated(config["pocket_path"],config.get("pocket_sha256"))
        self.context = load_context(Path(config["context_ready"]).parent/"context",ready["context_files"])
        if self.context.geometry_mode!="off" or self.context.n_atoms!=request["n_atoms"]:
            raise ValueError("Prepared native ligand slot count/geometry differs")
        self.model = FrozenProtenixTiny.load(cache_dir=Path(config["cache_dir"]))
        self.before = {"model_digest":array_digest(self.model),"context_digest":array_digest(self.context)}
        for field in self.before:
            expected=config.get("expected_"+field)
            if expected is not None and self.before[field]!=expected:
                raise ValueError("Frozen native identity differs: " + field)
        loss = NativeContactLoss(8.,map_protenix_pocket(self.context,pocket))
        self.key = jax.random.key(int(config.get("model_key",17)))
        context,model,key = self.context,self.model,self.key
        elements,charges = self.atomic_numbers,self.formal_charges

        @eqx.filter_jit
        def evaluate(model,context,key,a,c,b):
            def objective(aa,cc,bb):
                features = inject_full_schema(context,aa,cc,bb,elements,charges)
                _,trunk = _native_trunk(model.model,features,key)
                logits = model.model.distogram_head(trunk.z)
                output = MolecularTrunkOutput(distogram_logits=logits,distogram_bins=native_distogram_bins(),
                    ligand_token_indices=context.ligand_token_indices,protein_token_indices=context.protein_token_indices,
                    trunk_state=trunk)
                value,aux = loss(None,output,key=key)  # NativeContactLoss never reads the graph argument.
                finite = jnp.isfinite(logits).all() & jnp.isfinite(trunk.s).all() & jnp.isfinite(trunk.z).all()
                return value,(aux,finite)
            return jax.value_and_grad(objective,argnums=(0,1,2),has_aux=True)(a,c,b)
        self._evaluate = evaluate
        self._digest = array_digest
        self.identity = dict(protocol=PROTOCOL,**self.before,model=self.model.identities,
            source=source,context_ready_sha256=ready_sha,pocket_sha256=pocket_sha,
            n_atoms=context.n_atoms,atomic_numbers=list(elements),formal_charges=list(charges),
            model_key=int(config.get("model_key",17)),geometry_mode="off",diffusion=False,
            objective="negative_mean_ligand_max_pocket_probability_distance_below8A",
            feature_map="native128_elements_expected_formal_charge_bond_presence",
            limitations=["bond order/aromaticity collapse to presence","fixed occupied slots, names and protein context",
                         "contact surrogate is not affinity","new full-vocabulary feature map requires native pilot evidence"])

    def evaluate(self, request):
        import numpy as np
        import jax.numpy as jnp
        arrays=request["arrays"];n=self.context.n_atoms
        values=[np.asarray(arrays[k],dtype=np.float32) for k in ["atoms","charges","bonds"]]
        shapes=[(n,len(self.atomic_numbers)),(n,len(self.formal_charges)),(n,n,5)]
        for v,shape in zip(values,shapes):
            if v.shape!=shape or not np.isfinite(v).all() or (v<0).any() or not np.allclose(v.sum(-1),1,atol=2e-5,rtol=2e-5):
                raise ValueError("Incompatible/nonfinite native graph probabilities")
        if not np.allclose(values[2],values[2].transpose(1,0,2),atol=1e-6,rtol=0):
            raise ValueError("Native bond probabilities must be symmetric")
        diagonal=values[2][np.arange(n),np.arange(n)]
        if not np.allclose(diagonal[:,0],1,atol=1e-6,rtol=0) or not np.allclose(diagonal[:,1:],0,atol=1e-6,rtol=0):
            raise ValueError("Native diagonal bonds must be NONE")
        (value,(aux,finite)),gradients=self._evaluate(self.model,self.context,self.key,*(jnp.asarray(v) for v in values))
        result=float(value)
        gradients=[np.asarray(v) for v in gradients]
        if not bool(finite) or not math.isfinite(result) or any(not np.isfinite(v).all() for v in gradients):
            raise FloatingPointError("Nonfinite native trunk, objective or probability derivative")
        return dict(loss=result,gradients={k:v.tolist() for k,v in zip(["atoms","charges","bonds"],gradients)},
            aux={k:np.asarray(v).item() for k,v in aux.items()},
            gradient_norms={k:float(np.linalg.norm(v.astype(np.float64))) for k,v in zip(["atoms","charges","bonds"],gradients)})

    def close(self):
        after={"model_digest":self._digest(self.model),"context_digest":self._digest(self.context)}
        if after!=self.before:
            raise RuntimeError("Frozen Protenix/context changed during scoring")
        return dict(frozen_unchanged=True,identity=after)


def main():
    # Keep a dedicated wire stdout even if native imports/model loading print.
    wire=sys.stdout
    server=None
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            if request.get("protocol",PROTOCOL)!=PROTOCOL:
                raise ValueError("Unknown native bridge protocol")
            with contextlib.redirect_stdout(sys.stderr):
                operation=request.get("op")
                if operation=="init":
                    if server is not None:raise ValueError("Native worker already initialized")
                    server=NativeServer(request);result=dict(identity=server.identity,n_atoms=server.context.n_atoms,devices=server.devices)
                elif operation=="evaluate":
                    if server is None:raise ValueError("Initialize native worker first")
                    result=server.evaluate(request)
                elif operation=="close":
                    if server is None:raise ValueError("Initialize native worker first")
                    result=server.close()
                else:raise ValueError("Unknown native worker operation")
            wire.write(json.dumps(dict(result,status="ok",protocol=PROTOCOL,request_id=request["request_id"]),allow_nan=False)+"\n");wire.flush()
            if operation=="close":return
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            wire.write(json.dumps(dict(status="error",protocol=PROTOCOL,request_id=request.get("request_id"),
                error=type(exc).__name__+": "+str(exc)))+"\n");wire.flush()
            return


if __name__=="__main__":
    main()
