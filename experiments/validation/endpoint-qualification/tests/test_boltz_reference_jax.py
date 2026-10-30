"""Allocated-only CPU64 chemistry/softmax differentiation controls."""
import os

def modules():
    assert os.environ.get('LSB_JOBID','').isdigit() and os.environ.get('LSB_QUEUE')=='hpc'
    import jax,jax.numpy as jnp
    return jax,jnp

def test_reference_precision_context_restores_production_configuration():
    jax,jnp=modules();original=jax.config.jax_enable_x64
    with jax.experimental.enable_x64():assert jnp.asarray([.1],dtype=jnp.float64).dtype==jnp.float64
    assert jax.config.jax_enable_x64==original

def test_probability_to_logit_chain_matches_independent_AD():
    jax,jnp=modules()
    with jax.experimental.enable_x64():
        z=jnp.asarray([.1,.2,-.3],dtype=jnp.float64);weights=jnp.asarray([1.,-.3,.8],dtype=jnp.float64)
        p=jax.nn.softmax(z);g=jax.grad(lambda x:jnp.sum(jax.nn.softmax(x)*weights))(z)
        assert bool(jnp.allclose(g,p*(weights-jnp.sum(p*weights)),rtol=1e-12,atol=1e-14))

def test_fixed_target_gradient_flows_without_target_refresh():
    jax,jnp=modules()
    with jax.experimental.enable_x64():
        target=jax.lax.stop_gradient(jnp.asarray([0.,1.,0.],dtype=jnp.float64));z=jnp.asarray([.2,-.4,.1],dtype=jnp.float64)
        loss=lambda x:-jnp.sum(target*jax.nn.log_softmax(x));gradient=jax.grad(loss)(z)
        assert bool(jnp.allclose(gradient,jax.nn.softmax(z)-target,rtol=1e-12,atol=1e-14))
        assert float(jnp.linalg.norm(gradient))>0 and target.tolist()==[0.,1.,0.]
