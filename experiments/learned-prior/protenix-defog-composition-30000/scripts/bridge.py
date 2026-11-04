"""Serial authenticated stopped-teacher subprocess, with bounded responses."""
import json
import os
from pathlib import Path
import selectors
import subprocess

from contract import canonical_sha, teacher_time, query_identity


class Teacher:
    def __init__(self, command, log, c):
        self.c, self.sequence = c, 0
        self.log = Path(log).open('w', encoding='utf-8')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        env.pop('PYTHONHOME', None)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, env=env, text=True, bufsize=1)
        try:
            self.identity = self.read()
            if (self.identity.get('status') != 'ready' or self.identity.get('protocol') != c['version']
                or self.identity.get('policy') != c['teacher']['policy']
                or self.identity.get('checkpoint_sha256') != c['teacher']['checkpoint_sha256']
                or self.identity.get('source_commit') != c['teacher']['source_commit']
                or self.identity.get('time_ramp') != c['teacher']['time_ramp']
                or self.identity.get('device') != 'cpu'):
                raise ValueError('Teacher asset/policy handshake mismatch')
        except BaseException:
            self.close()
            raise

    def read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(300):
                raise TimeoutError('Teacher response timeout')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Teacher exited: ' + str(self.process.poll()))
        return json.loads(line)

    def request(self, action, **values):
        self.sequence += 1
        self.process.stdin.write(json.dumps({'id': self.sequence, 'action': action, **values}, allow_nan=False) + '\n')
        self.process.stdin.flush()
        result = self.read()
        if result.get('id') != self.sequence or result.get('status') != 'passed':
            raise ValueError('Teacher request failed: ' + json.dumps(result))
        return result

    def query(self, graph, *, state_index):
        import numpy as np
        active_time = teacher_time(self.c, state_index)
        x, e = np.asarray(graph.X), np.asarray(graph.E_upper)
        if x.dtype != np.float32 or e.dtype != np.float32:
            raise ValueError('Transport requires FP32 graph')
        payload = {'X': x.tolist(), 'E': e.tolist()}
        graph_sha = canonical_sha(payload)
        query_sha = query_identity(graph_sha, state_index, active_time)
        result = self.request('teacher', **payload, graph_sha256=graph_sha,
                              state_index=state_index, teacher_time=active_time, query_sha256=query_sha)
        if (result['graph_sha256'] != graph_sha or result['target_sha256'] != canonical_sha(result['target'])
            or result['policy'] != self.c['teacher']['policy'] or result['forward_calls'] != 1
            or result['state_index'] != state_index or result['teacher_time'] != active_time
            or result['query_sha256'] != query_sha):
            raise ValueError('Teacher graph/target identity mismatch')
        tx, te = (np.asarray(result['target'][name], dtype=np.float32) for name in ('X', 'E'))
        if tx.shape != x.shape or te.shape != e.shape:
            raise ValueError('Teacher changed slot/count/category shape')
        for value in (tx, te):
            if (not np.isfinite(value).all() or np.any(value < 0)
                or not np.allclose(value.sum(-1), 1, atol=1e-6, rtol=0)):
                raise ValueError('Teacher target off simplex')
        return (tx, te), result

    def frozen(self):
        result = self.request('frozen')
        if result['state_digest'] != self.identity['state_digest'] or result['queries'] != result['forwards']:
            raise ValueError('Teacher frozen state/query count failed')
        return result

    def close(self):
        if getattr(self, 'process', None) and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
        self.log.close()
