"""One frozen CPU denoiser query per request, with explicitly adapted features."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

from contract import ELEMENT_NATIVE_INDICES, canonical_sha, validate, teacher_time, query_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('LSB_JOBID') or sys.flags.optimize:
        raise RuntimeError('Allocated job with native assertions required')
    c = validate(json.loads(a.config.read_text()))
    # Reuse exactly the already tested checkpoint loader/shape contract, sealed
    # as a separate scientific dependency; no source or weight patching.
    root = Path(__file__).resolve().parents[4]
    path = root / 'experiments/learned-prior/defog-soft-teacher/probe.py'
    spec = importlib.util.spec_from_file_location('validated_defog_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import torch
    from torch.nn import functional as F
    from omegaconf import OmegaConf
    module.torch, module.F, module.OmegaConf = torch, F, OmegaConf
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(c['seed'])
    with contextlib.redirect_stdout(sys.stderr):
        model, audit = module.load_model(a.source, a.checkpoint, c['teacher'])
    before = module.state_digest(model)
    calls = [0]
    hook = model.model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    def emit(value):
        print(json.dumps(value, allow_nan=False), flush=True)
    emit({'status': 'ready', 'protocol': c['version'], 'state_digest': before,
          'checkpoint_sha256': c['teacher']['checkpoint_sha256'], 'source_commit': c['teacher']['source_commit'],
          'policy': c['teacher']['policy'], 'time_ramp': c['teacher']['time_ramp'],
          'audit': audit, 'device': 'cpu'})
    requests = 0
    try:
        for line in sys.stdin:
            request = json.loads(line)
            reply = {'id': request.get('id')}
            try:
                if request['action'] == 'frozen':
                    after = module.state_digest(model)
                    if before != after:
                        raise ValueError('DeFoG state changed')
                    emit({**reply, 'status': 'passed', 'state_digest': after, 'queries': requests, 'forwards': calls[0]})
                    continue
                if request['action'] != 'teacher' or requests >= c['teacher']['max_queries']:
                    raise ValueError('Unknown action or query budget exhausted')
                if canonical_sha({'X': request['X'], 'E': request['E']}) != request['graph_sha256']:
                    raise ValueError('Graph transport digest mismatch')
                active_time = teacher_time(c, request['state_index'])
                if (request['teacher_time'] != active_time or
                    request['query_sha256'] != query_identity(request['graph_sha256'], request['state_index'], active_time)):
                    raise ValueError('Teacher time/index transport mismatch')
                requests += 1
                x = torch.tensor(request['X'], dtype=torch.float32)
                edges = torch.tensor(request['E'], dtype=torch.float32)
                n = c['n_atoms']
                if x.shape != (n, 6) or edges.shape != (n * (n - 1) // 2, 5):
                    raise ValueError('Wrong fixed-occupancy graph')
                nx = torch.zeros((1, n, 12))
                nx[0, :, list(ELEMENT_NATIVE_INDICES)] = x
                ne = torch.zeros((1, n, n, 5))
                upper = torch.triu_indices(n, n, 1)
                ne[0, upper[0], upper[1]] = edges
                ne[0, upper[1], upper[0]] = edges
                module.validate_graph(nx, ne)
                hx, he = module.hard_view(nx, ne)
                data = module.noisy(nx, ne, active_time)
                prior_calls = calls[0]
                with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                    extra = model.compute_extra_data(module.noisy(hx, he, active_time))
                    pred = model.forward(data, extra, data['node_mask'])
                    full_x, full_e = pred.X.softmax(-1), pred.E.softmax(-1)
                    qx = full_x[0, :, list(ELEMENT_NATIVE_INDICES)]
                    supported = qx.sum(-1, keepdim=True)
                    if not torch.isfinite(supported).all() or bool((supported < 1e-8).any()):
                        raise ValueError('Insufficient supported teacher element mass')
                    qx = qx / supported
                    qe = full_e[0, upper[0], upper[1]]
                    if not all(torch.isfinite(v).all() for v in (qx, qe)) or not torch.allclose(full_e, full_e.transpose(1, 2), atol=1e-7, rtol=0):
                        raise ValueError('Nonfinite/asymmetric teacher output')
                    if calls[0] - prior_calls != 1 or qx.requires_grad or qe.requires_grad:
                        raise ValueError('Expected one detached teacher forward')
                target = {'X': qx.tolist(), 'E': qe.tolist()}
                emit({**reply, 'status': 'passed', 'graph_sha256': request['graph_sha256'],
                      'state_index': request['state_index'], 'teacher_time': active_time,
                      'query_sha256': request['query_sha256'],
                      'target_sha256': canonical_sha(target), 'target': target, 'forward_calls': 1,
                      'excluded_atom_mass_mean': float((1 - supported).mean()),
                      'excluded_atom_mass_max': float((1 - supported).max()),
                      'full_native_atom_predictions': full_x[0].tolist(), 'policy': c['teacher']['policy']})
            except Exception as exc:
                emit({**reply, 'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
    finally:
        hook.remove()


if __name__ == '__main__':
    main()
