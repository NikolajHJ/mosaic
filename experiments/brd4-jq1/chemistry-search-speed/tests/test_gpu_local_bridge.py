"""Allocated CPU transport/VJP fixtures using explicitly fake arithmetic.

No fixture here claims a GPU device, native model result or offload speedup.
The actual L40S worker, matrix parity and original FD ladder remain mandatory
in the separate full Boltz pilot.
"""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gpu_local_bridge as bridge
from gpu_local_worker import VERSION, encode, decode


SHAPE = (1, 25, 25, 5)


def arrays():
    values = jnp.linspace(-.2, .2, 25 * 25 * 5, dtype=jnp.float32).reshape(SHAPE)
    return values, jnp.ones((1, 25), dtype=bool), jnp.ones((1, 25), dtype=jnp.int32)


class FakeClient:
    """Fresh analytic CPU fixture. Its readiness explicitly does NOT say GPU."""
    instances = []

    def __init__(self, directory=None):
        self.calls, self.closed = [], False
        self.stats = {'scope': 'fake_CPU_arithmetic_fixture_only'}
        self.ready = {'device': {'platform': 'fake_cpu_fixture'}, 'GPU_proof': False}
        self.instances.append(self)

    def evaluate(self, op, values, active, caps, indices):
        values, active, caps = np.asarray(values), np.asarray(active), np.asarray(caps)
        self.calls.append({'op': op, 'values': values.copy(), 'active': active.copy(), 'caps': caps.copy(),
                           'indices': tuple(indices)})
        scale = np.float32(np.sum(active, dtype=np.int32) + np.sum(caps, dtype=np.int32))
        value = np.asarray([scale * np.sum(values * values, dtype=np.float32)], dtype=np.float32)
        return value if op == 'primal' else (value, np.asarray(2 * scale * values, dtype=np.float32))

    def close(self):
        self.closed = True


def test_custom_vjp_uses_correct_upstream_cotangent_and_chain_rule_on_cpu_fake():
    client = FakeClient(); local = bridge.remote_local(client)
    values, active, caps = arrays()
    scalar = lambda x: 3.25 * jnp.sum(local(2 * x, active, caps, (0, 1, 2, 3, 4)))
    value, gradient = jax.value_and_grad(scalar)(values)
    jax.block_until_ready((value, gradient))
    scale = 50.0
    np.testing.assert_allclose(gradient, 3.25 * 8 * scale * values, atol=2e-5, rtol=2e-6)
    assert client.calls and all(c['op'] == 'value_grad' for c in client.calls)
    assert client.ready['device']['platform'] == 'fake_cpu_fixture'


def test_compiled_callbacks_never_reuse_stale_values_masks_or_caps():
    client = FakeClient(); local = bridge.remote_local(client)
    values, active, caps = arrays()
    kernel = jax.jit(jax.value_and_grad(lambda v, a, c: jnp.sum(local(v, a, c, (0, 1, 2, 3, 4)))))
    first = kernel(values, active, caps); jax.block_until_ready(first)
    second = kernel(values + .1, active, caps); jax.block_until_ready(second)
    third = kernel(values, active.at[0, 0].set(False), caps.at[0, 0].set(2)); jax.block_until_ready(third)
    # Same combined scale in third, but raw boolean/int inputs must still arrive.
    assert len(client.calls) == 3
    assert not np.array_equal(client.calls[0]['values'], client.calls[1]['values'])
    assert not np.array_equal(client.calls[0]['active'], client.calls[2]['active'])
    assert not np.array_equal(client.calls[0]['caps'], client.calls[2]['caps'])
    assert float(first[0]) != float(second[0])


def test_independent_primal_and_ad_requests_are_fresh_not_a_shared_cached_result():
    client = FakeClient(); local = bridge.remote_local(client)
    values, active, caps = arrays()
    primal = local(values, active, caps, (0, 1, 2, 3, 4)); jax.block_until_ready(primal)
    value, gradient = jax.value_and_grad(lambda v: jnp.sum(local(v, active, caps, (0, 1, 2, 3, 4))))(values)
    jax.block_until_ready((value, gradient))
    assert [c['op'] for c in client.calls] == ['primal', 'value_grad']
    np.testing.assert_allclose(primal[0], value, atol=0, rtol=0)


@pytest.mark.parametrize('fail', (False, True))
def test_install_restores_original_math_and_closes_fake_client(monkeypatch, fail):
    from mosaic.small_molecule.chemistry import loss
    old = loss._local, loss._compiled_terms
    monkeypatch.setattr(bridge, 'LocalGPUClient', FakeClient)
    try:
        with bridge.install_offloaded_matrix_loss('/unused/fake-fixture-path') as receipt:
            assert receipt['ready']['device']['platform'] == 'fake_cpu_fixture'
            assert receipt['parent_optimizer_backend'] == 'cpu'
            assert receipt['native_model_worker_unchanged'] is True
            assert loss._compiled_terms is not old[1]
            if fail:
                raise LookupError('fake fixture body')
    except LookupError:
        assert fail
    assert (loss._local, loss._compiled_terms) == old
    assert FakeClient.instances[-1].closed is True


def test_nested_offload_context_refused_without_replacing_outer_client(monkeypatch):
    monkeypatch.setattr(bridge, 'LocalGPUClient', FakeClient)
    before = len(FakeClient.instances)
    with bridge.install_offloaded_matrix_loss('/unused/fake-fixture-path'):
        with pytest.raises(RuntimeError, match='nested'):
            with bridge.install_offloaded_matrix_loss('/unused/fake-fixture-path'):
                pytest.fail('Nested bridge unexpectedly accepted')
    assert len(FakeClient.instances) == before + 1


@pytest.mark.parametrize('dtype, shape', ((np.float32, SHAPE), (np.bool_, (1, 25)), (np.int32, (1, 25))))
def test_wire_exact_dtype_shape_and_byte_roundtrip(dtype, shape):
    value = np.ones(shape, dtype=dtype)
    wire = encode(value)
    original = deepcopy(wire)
    result = decode(wire, dtype, shape)
    assert result.dtype == value.dtype and result.shape == value.shape
    assert result.tobytes() == value.tobytes() and wire == original
    result.reshape(-1)[0] = 0
    assert decode(wire, dtype, shape).tobytes() == value.tobytes()


@pytest.mark.parametrize('fault', ('dtype', 'shape', 'length', 'base64'))
def test_wire_wrong_dtype_shape_length_or_encoding_fails_closed(fault):
    wire = encode(np.ones(SHAPE, dtype=np.float32))
    if fault == 'dtype': wire['dtype'] = np.dtype(np.float64).str
    elif fault == 'shape': wire['shape'] = [1, 25, 25, 4]
    elif fault == 'length': wire['base64'] = 'AA=='
    else: wire['base64'] = 'invalid!'
    with pytest.raises((AssertionError, ValueError)):
        decode(wire, np.float32, SHAPE)


def transport_client(response):
    """Bypass constructor solely to test wire validation without a subprocess."""
    client = bridge.LocalGPUClient.__new__(bridge.LocalGPUClient)
    client.sequence, client.lock, client.closed = 0, threading.RLock(), False
    client.stats = {'calls': 0, 'primal_calls': 0, 'value_grad_calls': 0,
                    'request_seconds': 0.0, 'worker_compute_seconds': 0.0}
    client.process = SimpleNamespace(poll=lambda: None, stdin=io.StringIO())
    client._read = lambda: response
    return client


def response():
    return {'version': VERSION, 'sequence': 1, 'status': 'passed', 'device_platform': 'gpu',
            'elapsed_seconds': .01, 'value': encode(np.asarray([1.0], dtype=np.float32)),
            'gradient': encode(np.ones(SHAPE, dtype=np.float32))}


def test_client_wire_has_monotonic_sequence_exact_shapes_and_fresh_operation():
    client = transport_client(response()); values, active, caps = arrays()
    value, gradient = client.evaluate('value_grad', values, active, caps, (0, 1, 2, 3, 4))
    packet = json.loads(client.process.stdin.getvalue())
    assert packet['sequence'] == 1 and packet['op'] == 'value_grad' and packet['version'] == VERSION
    assert decode(packet['log_bonds'], np.float32, SHAPE).tobytes() == np.asarray(values).tobytes()
    assert decode(packet['active'], np.bool_, (1, 25)).tobytes() == np.asarray(active).tobytes()
    assert decode(packet['caps'], np.int32, (1, 25)).tobytes() == np.asarray(caps).tobytes()
    assert value.shape == (1,) and gradient.shape == SHAPE
    assert client.stats['calls'] == client.stats['value_grad_calls'] == 1


@pytest.mark.parametrize('fault', ('sequence', 'version', 'status', 'device', 'nonfinite'))
def test_client_rejects_stale_failed_cpu_or_nonfinite_response(fault):
    row = response()
    if fault == 'sequence': row['sequence'] = 2
    elif fault == 'version': row['version'] = 'other'
    elif fault == 'status': row['status'] = 'failed'
    elif fault == 'device': row['device_platform'] = 'cpu'
    else: row['value'] = encode(np.asarray([np.nan], dtype=np.float32))
    client = transport_client(row); values, active, caps = arrays()
    with pytest.raises(AssertionError):
        client.evaluate('primal', values, active, caps, (0, 1, 2, 3, 4))


@pytest.mark.parametrize('fault', ('timeout', 'eof', 'bad_json'))
def test_read_timeout_eof_and_malformed_json_are_real_errors(monkeypatch, fault):
    client = bridge.LocalGPUClient.__new__(bridge.LocalGPUClient)
    client.timeout = .01
    client.process = SimpleNamespace(stdout=io.StringIO('{' if fault == 'bad_json' else ''))
    monkeypatch.setattr(bridge.select, 'select', lambda *args: ([], [], []) if fault == 'timeout'
                        else ([client.process.stdout], [], []))
    expected = TimeoutError if fault == 'timeout' else RuntimeError if fault == 'eof' else json.JSONDecodeError
    with pytest.raises(expected):
        client._read()


def test_popen_failure_closes_opened_stderr_and_preserves_startup_error(monkeypatch, tmp_path):
    handles = []
    original_open = Path.open
    def observed_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name == 'stderr.log': handles.append(handle)
        return handle
    error = OSError('fake process startup failed')
    def failed_popen(*args, **kwargs):
        raise error
    monkeypatch.setattr(Path, 'open', observed_open)
    monkeypatch.setattr(bridge.subprocess, 'Popen', failed_popen)
    with pytest.raises(OSError) as caught:
        bridge.LocalGPUClient(tmp_path / 'failed-worker')
    assert caught.value is error
    assert len(handles) == 1 and handles[0].closed
