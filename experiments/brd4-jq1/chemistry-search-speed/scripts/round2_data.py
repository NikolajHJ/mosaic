"""Allocated-only saved graph and authenticated frozen-reference loading."""
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cases(plan):
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry.state import GraphLogits, mosaic_spec
    from mosaic.small_molecule.chemistry.preparation import PreparedFailureGuidance

    result=[]
    for row in plan['inputs']:
        p=Path(row['path']);f=Path(row['frozen_reference']['path'])
        assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        assert f.is_file() and not f.is_symlink() and sha(f)==row['frozen_reference']['sha256']
        with np.load(p,allow_pickle=False) as a:
            logits=GraphLogits(jnp.asarray(a['atom_logits']),jnp.asarray(a['bond_logits']))
        assert logits.atoms.shape==(25,6) and logits.bonds.shape==(300,5)
        assert logits.atoms.dtype==logits.bonds.dtype==jnp.float32
        v=json.loads(f.read_text());spec=mosaic_spec(25)
        boolean={'active','changed','cuts','cut_active'}
        arrays={k:jnp.asarray(a,dtype=jnp.bool_ if k in boolean else jnp.int32) for k,a in v['arrays'].items()}
        prepared=PreparedFailureGuidance(arrays,v['receipt'],v['identity'],0.0,())
        prepared.verify(spec)
        result.append((row,logits,prepared,spec))
    return result


def evaluation(logits,prepared,spec):
    import equinox as eqx
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry.loss import failure_aware_repair_loss
    f=lambda z:failure_aware_repair_loss(z.decoded(),prepared,spec)[0]
    value,grad=eqx.filter_value_and_grad(f)(logits)
    primal,terms=failure_aware_repair_loss(logits.decoded(),prepared,spec)
    jax.block_until_ready((value,grad,primal,terms))
    arrays=[np.asarray(grad.atoms),np.asarray(grad.bonds)]
    assert np.isfinite(float(value)) and all(np.isfinite(a).all() for a in arrays)
    return {'value':float(value),'primal':float(primal),'components':{k:float(v) for k,v in terms.items()},
            'gradient_sha256':[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]},grad
