"""Allocated FP64 scalar/FP32 first-order transport fixtures; no real model."""

import importlib.util
import json
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

from mosaic.small_molecule.state import RelaxedGraph


def _bridge():
    try:
        import dbmol_bridge
        return dbmol_bridge
    except ModuleNotFoundError:
        path = Path(__file__).resolve().parents[1] / "scripts" / "bridge.py"
        spec = importlib.util.spec_from_file_location("dbmol_bridge", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


bridge = _bridge()


TOY = r'''import json,sys
metrics_order=__METRICS__
print(json.dumps({'status':'ready','protocol':'mosaic-boltz-dbmol-v1',
 'optimized_output':'dbmol_total_loss','scalar_dtype':'float64',
 'graph_dtype':'float32','gradient_dtype':'float32','metric_order':metrics_order,
 'n_atoms':25,'recycling_steps':0,'diffusion_gradient':False}),flush=True)
for line in sys.stdin:
 r=json.loads(line);x,e=r['X'],r['E'];objective=r.get('objective','dbmol_total_loss')
 base=sum(v*k for row in x for k,v in enumerate(row))+sum(v*k for row in e for k,v in enumerate(row))
 metrics={key:(i+1)*base for i,key in enumerate(metrics_order)}
 metrics['dbmol_total_loss']+=100000000.+1./3.
 metrics['affinity_probability_binary']=.75
 i=metrics_order.index(objective)+1
 out={'values':[metrics[objective],metrics['affinity_probability_binary'],metrics['affinity_logits_binary']],
      'metrics':metrics,'objective':objective,'graph_sha256':r['graph_sha256']}
 if r['gradient']:
  out.update(gradient_X=[[float(i*k) for k in range(6)]]*25,
             gradient_E=[[float(i*k) for k in range(5)]]*300)
 print(json.dumps({'id':r['id'],'status':'passed',**out}),flush=True)
'''


@pytest.fixture
def worker(tmp_path):
    path = tmp_path / "toy_worker.py"
    path.write_text(TOY.replace("__METRICS__", repr(list(bridge.METRICS))))
    value = bridge.DBMolWorker([sys.executable, "-u", str(path)],
                               stderr=tmp_path / "worker.log", timeout=30)
    try:
        yield value
    finally:
        value.close()


def graph():
    import jax.numpy as jnp
    return RelaxedGraph(jnp.full((25, 6), 1 / 6, jnp.float32),
                        jnp.full((300, 5), .2, jnp.float32))


def test_total_transport_retains_fp64_scalar_raw_fp32_vjp_and_stopped_logging(worker):
    assert jax.config.jax_enable_x64
    g, key = graph(), jax.random.key(17)
    loss = bridge.dbmol_loss(worker)
    (value, aux), gradient = jax.value_and_grad(loss, has_aux=True)(g, key=key)
    primal, _ = loss(g, key=key)
    independent, gx, ge = worker.evaluate(g.X, g.E_upper, gradient=True)
    assert np.asarray(value).dtype == np.float64
    assert float(value) == float(primal) == independent[0]
    assert float(value) != float(np.float32(value))
    assert np.asarray(gradient.X).dtype == np.float32
    assert np.asarray(gradient.E_upper).dtype == np.float32
    np.testing.assert_array_equal(gradient.X, gx)
    np.testing.assert_array_equal(gradient.E_upper, ge)
    assert set(aux) == set(bridge.METRICS)
    assert float(aux["affinity_probability_binary"]) == .75
    stopped = jax.grad(lambda v: loss(v, key=key)[1]["affinity_pred_value"])(g)
    np.testing.assert_array_equal(stopped.X, np.zeros((25, 6), np.float32))
    np.testing.assert_array_equal(stopped.E_upper, np.zeros((300, 5), np.float32))
    scaled = jax.grad(lambda v: 2.5 * loss(v, key=key)[0])(g)
    np.testing.assert_array_equal(scaled.X, 2.5 * gx)
    np.testing.assert_array_equal(scaled.E_upper, 2.5 * ge)


@pytest.mark.parametrize("objective", ("dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss"))
def test_each_component_selects_own_primitive_scalar_and_raw_vjp(worker, objective):
    g, key = graph(), jax.random.key(17)
    loss = bridge.dbmol_loss(worker, objective=objective)
    (value, aux), gradient = jax.value_and_grad(loss, has_aux=True)(g, key=key)
    expected, gx, ge = worker.evaluate(g.X, g.E_upper, gradient=True, objective=objective)
    assert float(value) == expected[0]
    np.testing.assert_array_equal(gradient.X, gx)
    np.testing.assert_array_equal(gradient.E_upper, ge)
    assert float(aux[objective]) == float(value)


@pytest.mark.parametrize("fault", ("float64_graph", "atom_shape", "bond_shape", "nonfinite"))
def test_transport_refuses_undeclared_graph_types_before_request(worker, fault):
    x, e = np.asarray(graph().X).copy(), np.asarray(graph().E_upper).copy()
    if fault == "float64_graph": x = x.astype(np.float64)
    elif fault == "atom_shape": x = x[:-1]
    elif fault == "bond_shape": e = e[:-1]
    else: x[0, 0] = np.nan
    with pytest.raises(ValueError):
        worker.evaluate(x, e, gradient=True)


def test_fp64_scalar_policy_cannot_be_silently_disabled(worker):
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(ValueError, match="x64"):
            bridge.dbmol_loss(worker)
    finally:
        jax.config.update("jax_enable_x64", previous)
