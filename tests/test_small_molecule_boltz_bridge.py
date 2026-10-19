"""Allocated CPU transport/VJP/trajectory tests; the worker here is an explicit fixture."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.boltz2.bridge import AffinityWorker, affinity_loss
from mosaic.small_molecule.chemistry import (
    GraphLogits,
    FailureGuidanceConfig,
    mosaic_spec,
    prepare_failure_guidance,
)
from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
from mosaic.small_molecule.chemistry.trajectory import run_chemical_updates
from mosaic.small_molecule.state import graph_from_hard, mix_with_uniform

TOY = """import json,sys
print(json.dumps({"status":"ready","protocol":"mosaic-boltz-affinity-v1"}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['action']=='evaluate':
  x,e=r['X'],r['E']
  value=sum(v*k for row in x for k,v in enumerate(row))+sum(v*k for row in e for k,v in enumerate(row))
  out={'values':[value,.5,0.], 'graph_sha256':r['graph_sha256']}
  if r['gradient']:out.update(gradient_X=[list(range(6))]*6,gradient_E=[list(range(5))]*15)
 else:out={}
 print(json.dumps({'status':'passed','id':r['id'],**out}),flush=True)
"""


@pytest.fixture
def worker(tmp_path):
    script = tmp_path / "toy.py"
    script.write_text(TOY)
    client = AffinityWorker(
        [sys.executable, "-u", str(script)], stderr=tmp_path / "worker.log", timeout=20
    )
    try:
        yield client
    finally:
        client.close()


def initial():
    return mix_with_uniform(
        graph_from_hard(
            [0] * 6, [int(j == i + 1) for i in range(6) for j in range(i + 1, 6)]
        ),
        0.5,
    )


def test_native_vjp_primal_jit_and_logging_metrics(worker):
    loss = affinity_loss(worker)
    graph = initial()
    (value, aux), grad = jax.value_and_grad(loss, has_aux=True)(
        graph, key=jax.random.key(17)
    )
    primal, _ = jax.jit(loss)(graph, key=jax.random.key(17))
    assert float(value) == float(primal)
    np.testing.assert_array_equal(grad.X, np.tile(np.arange(6), (6, 1)))
    np.testing.assert_array_equal(grad.E_upper, np.tile(np.arange(5), (15, 1)))
    assert float(aux["affinity_probability_binary"]) == 0.5
    # The VJP must respect the caller's scalar cotangent, not assume unit weight.
    scaled = jax.grad(lambda g: 3 * loss(g, key=jax.random.key(17))[0])(graph)
    np.testing.assert_array_equal(scaled.X, 3 * grad.X)


def test_bridge_rejects_schema_and_double_inputs(worker):
    with pytest.raises(ValueError, match="FP32"):
        worker.evaluate(
            np.zeros((6, 6), dtype=np.float64),
            np.zeros((15, 5), dtype=np.float32),
            gradient=True,
        )


def test_affinity_trajectory_uses_shared_chemistry_schedule_without_contact_labels(
    worker, tmp_path
):
    state = GraphLogits.from_graph(initial())
    spec = mosaic_spec(6)

    def prepare(logits, update):
        return prepare_failure_guidance(logits.decoded(), spec, update=update)

    report = run_chemical_updates(
        affinity_loss(worker),
        state,
        prepare=prepare,
        spec=spec,
        reference=jnp.ones(6) / 6,
        key=jax.random.key(17),
        directory=tmp_path / "trajectory",
        steps=2,
        step_size=0.03,
        max_backtracks=2,
        config={
            "loss": FailureGuidanceConfig(),
            "repair_weight": 0.001,
            "element_js_weight": 0.1,
            "native_name": "affinity",
        },
        schedule=RepairWeightSchedule(0.001, 50.0, 1, 2),
    )
    assert report["status"] == "passed" and len(report["states"]) == 3
    assert "best_observed_affinity" in report and "best_observed_contact" not in report
    for saved in report["states"]:
        assert "affinity_loss" in saved and "contact_loss" not in saved
        assert "affinity_probability_binary" in saved["components"]
    assert report["states"][-1]["repair_weight"] == 50.0
    with np.load(tmp_path / "trajectory" / report["states"][-1]["snapshot"]) as saved:
        assert "affinity_gradient_X" in saved and "contact_gradient_X" not in saved
