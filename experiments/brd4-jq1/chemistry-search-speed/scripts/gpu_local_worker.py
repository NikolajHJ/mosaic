from gpu_process_admission import legacy_adapter_profile
"""Allocated, single-molecule GPU local-risk arithmetic worker; no model/search."""
import base64
import hashlib
import json
import os
import sys
import time

VERSION = 'chemistry-local-matrix-GPU-RPC-v1'


def encode(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    return {'shape': list(array.shape), 'dtype': array.dtype.str,
            'base64': base64.b64encode(array.tobytes()).decode()}


def decode(row, dtype, shape):
    import numpy as np
    assert row['dtype'] == np.dtype(dtype).str and row['shape'] == list(shape)
    raw = base64.b64decode(row['base64'], validate=True)
    assert len(raw) == np.dtype(dtype).itemsize * int(np.prod(shape))
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def main():
    assert not sys.flags.optimize and os.environ.get('LSB_JOBID')
    assert os.environ.get('LSB_QUEUE') == legacy_adapter_profile()['queue']
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpu_matrix import local_valence_matrix, require_backend
    require_backend('gpu')
    assert len(jax.devices('gpu')) == 1 and legacy_adapter_profile()['family'] in jax.devices('gpu')[0].device_kind
    assert not jax.config.jax_enable_x64
    print(json.dumps({'version': VERSION, 'event': 'ready', 'pid': os.getpid(),
                      'device': {'platform': 'gpu', 'kind': jax.devices('gpu')[0].device_kind},
                      'model_or_search_loaded': False}), flush=True)
    kernels = {}
    for line in sys.stdin:
        try:
            request = json.loads(line); assert request['version'] == VERSION
            sequence = request['sequence']; op = request['op']
            if op == 'shutdown':
                print(json.dumps({'version': VERSION, 'sequence': sequence, 'status': 'closed'}), flush=True)
                return
            assert op in ('primal', 'value_grad')
            indices = tuple(request['order_indices'])
            assert len(indices) == 5 and set(indices) == set(range(5))
            v = decode(request['log_bonds'], np.float32, (1, 25, 25, 5))
            active = decode(request['active'], np.bool_, (1, 25))
            caps = decode(request['caps'], np.int32, (1, 25))
            assert not np.isnan(v).any() and not np.isposinf(v).any()
            if indices not in kernels:
                def make_kernels(fixed_indices):
                    def values(v, active, caps):
                        return local_valence_matrix(v, active, caps, fixed_indices)

                    def scalar(v, active, caps):
                        result = values(v, active, caps)
                        return jnp.sum(result), result

                    return jax.jit(values), jax.jit(jax.value_and_grad(scalar, has_aux=True))

                kernels[indices] = make_kernels(indices)
            primal, value_grad = kernels[indices]
            started = time.perf_counter()
            if op == 'primal':
                value = primal(v, active, caps); jax.block_until_ready(value)
                response = {'value': encode(np.asarray(value))}
            else:
                (_, value), grad = value_grad(v, active, caps)
                jax.block_until_ready((value, grad))
                assert np.isfinite(np.asarray(grad)).all()
                response = {'value': encode(np.asarray(value)), 'gradient': encode(np.asarray(grad))}
            assert np.isfinite(np.asarray(value)).all()
            print(json.dumps({'version': VERSION, 'sequence': sequence, 'status': 'passed',
                              'device_platform': 'gpu', 'elapsed_seconds': time.perf_counter() - started,
                              **response}), flush=True)
        except BaseException as error:
            print(json.dumps({'version': VERSION, 'status': 'failed', 'error': repr(error)}), flush=True)
            raise


if __name__ == '__main__':
    main()
