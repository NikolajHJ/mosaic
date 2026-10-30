from gpu_process_admission import legacy_adapter_profile
"""Explicit CPU-JAX to GPU local-risk VJP bridge for the frozen Boltz worker.

The Boltz driver retains its declared CPU optimizer and original CUDA PyTorch
oracle. Only the existing differentiable chemistry local-risk computation is
offloaded. Primal and value/gradient calls are fresh evaluations; no scientific
result, repair target or probability value is cached by this transport.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from gpu_local_worker import VERSION, encode, decode
from gpu_process_admission import capture_gpu_admission, coexistence_canary

_INSTALLED = False


class LocalGPUClient:
    def __init__(self, directory, *, timeout=90):
        directory = Path(directory); directory.mkdir(exist_ok=False)
        self.directory, self.timeout = directory, timeout
        self.sequence = 0; self.lock = threading.RLock(); self.closed = False
        self.stats = {'version': VERSION, 'calls': 0, 'primal_calls': 0, 'value_grad_calls': 0,
                      'request_seconds': 0.0, 'worker_compute_seconds': 0.0,
                      'fresh_evaluation_every_request': True, 'parent_optimizer_backend': 'cpu'}
        self.stderr = (directory / 'stderr.log').open('xb')
        env = dict(os.environ, JAX_PLATFORMS='cuda,cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   JAX_ENABLE_X64='0', JAX_ENABLE_COMPILATION_CACHE='false', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   CUDA_CACHE_PATH=str(directory / 'cuda-cache'), CUDA_CACHE_MAXSIZE=str(64 * 1024**2),
                   XDG_CACHE_HOME=str(directory / 'cache'))
        self.process = None
        try:
            self.allocation = capture_gpu_admission(directory)
            self.process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('gpu_local_worker.py'))],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                            text=True, bufsize=1, env=env)
            self.ready = self._read()
            assert self.ready['version'] == VERSION and self.ready['event'] == 'ready'
            assert self.ready['device']['platform'] == 'gpu' and legacy_adapter_profile()['family'] in self.ready['device']['kind']
            assert self.ready['model_or_search_loaded'] is False
            self.coexistence = coexistence_canary(self, directory)
        except BaseException:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=15)
            self.stderr.close()
            raise

    def _read(self):
        available, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not available:
            raise TimeoutError('GPU chemistry worker did not return within the declared bound')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('GPU chemistry worker ended before its response')
        return json.loads(line)

    def evaluate(self, op, log_bonds, active, caps, indices):
        import numpy as np
        with self.lock:
            assert not self.closed and op in ('primal', 'value_grad')
            assert self.process.poll() is None
            self.sequence += 1
            message = {'version': VERSION, 'sequence': self.sequence, 'op': op,
                       'log_bonds': encode(np.asarray(log_bonds, dtype=np.float32)),
                       'active': encode(np.asarray(active, dtype=np.bool_)),
                       'caps': encode(np.asarray(caps, dtype=np.int32)), 'order_indices': list(indices)}
            started = time.perf_counter()
            self.process.stdin.write(json.dumps(message) + '\n'); self.process.stdin.flush()
            row = self._read()
            assert row['version'] == VERSION and row['sequence'] == self.sequence and row['status'] == 'passed'
            assert row['device_platform'] == 'gpu'
            value = decode(row['value'], np.float32, (1,))
            assert np.isfinite(value).all()
            self.stats['calls'] += 1; self.stats[op + '_calls'] += 1
            self.stats['request_seconds'] += time.perf_counter() - started
            self.stats['worker_compute_seconds'] += row['elapsed_seconds']
            if op == 'primal':
                return value
            gradient = decode(row['gradient'], np.float32, (1, 25, 25, 5))
            assert np.isfinite(gradient).all()
            return value, gradient

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                if self.process.poll() is None:
                    self.sequence += 1
                    self.process.stdin.write(json.dumps({'version': VERSION, 'sequence': self.sequence,
                                                         'op': 'shutdown'}) + '\n')
                    self.process.stdin.flush(); row = self._read()
                    assert row['sequence'] == self.sequence and row['status'] == 'closed'
                assert self.process.wait(timeout=15) == 0
            finally:
                if self.process.poll() is None:
                    self.process.kill(); self.process.wait(timeout=15)
                self.stderr.close()
                (self.directory / 'receipt.json').write_text(json.dumps({'ready': self.ready,
                    'stats': self.stats, 'allocation': self.allocation, 'coexistence': self.coexistence,
                    'closed': True, 'returncode': self.process.returncode}, indent=2) + '\n')


def remote_local(client):
    import jax
    import jax.numpy as jnp
    functions = {}

    def local(log_bonds, active, caps, order_indices):
        assert log_bonds.shape == (1, 25, 25, 5) and active.shape == caps.shape == (1, 25)
        indices = tuple(order_indices)
        if indices not in functions:
            @jax.custom_vjp
            def evaluate(v, a, c):
                return jax.pure_callback(lambda v, a, c: client.evaluate('primal', v, a, c, indices),
                    jax.ShapeDtypeStruct((1,), jnp.float32), v, a, c)

            def forward(v, a, c):
                value, gradient = jax.pure_callback(
                    lambda v, a, c: client.evaluate('value_grad', v, a, c, indices),
                    (jax.ShapeDtypeStruct((1,), jnp.float32), jax.ShapeDtypeStruct(v.shape, jnp.float32)), v, a, c)
                return value, gradient

            def backward(gradient, cotangent):
                return gradient * cotangent[:, None, None, None], None, None

            evaluate.defvjp(forward, backward)
            functions[indices] = evaluate
        return functions[indices](log_bonds, active, caps)

    return local


@contextmanager
def install_offloaded_matrix_loss(directory):
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError('GPU chemistry bridge contexts cannot be nested')
    import jax
    from mosaic.small_molecule.chemistry import loss
    assert jax.default_backend() == 'cpu' and not jax.config.jax_enable_x64
    original_local, original_terms = loss._local, loss._compiled_terms
    client = LocalGPUClient(directory)
    _INSTALLED = True
    try:
        loss._local = remote_local(client)

        def terms(decoded, arrays, weights, order_indices):
            return loss._terms(decoded, arrays, weights, order_indices)

        loss._compiled_terms = jax.jit(terms, static_argnames=('order_indices',))
        yield {'version': VERSION, 'stats': client.stats, 'ready': client.ready,
               'GPU_process_admission': getattr(client, 'allocation', None),
               'GPU_process_coexistence': getattr(client, 'coexistence', None),
               'native_model_worker_unchanged': True, 'parent_optimizer_backend': 'cpu',
               'native_oracle_replaced': False, 'probability_or_target_cache': False}
    finally:
        loss._local, loss._compiled_terms = original_local, original_terms
        _INSTALLED = False
        client.close()
