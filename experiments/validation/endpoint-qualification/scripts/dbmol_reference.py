"""Separate full-FP64 affinity reference; original DBMol production stays exact.

Imports are lazy. CPU controls and native evaluations require real allocations.
The reference promotes an independently loaded frozen copy and intercepts only
explicit floating dtype requests. It rejects any remaining narrow operation.
No optimizer, structural-model substitution, diffusion or saved-state writer.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def precision_mode():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    class FullDouble(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.operations = Counter(); self.promoted_requests = Counter()
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            dtype = kwargs.get('dtype')
            if dtype in (torch.float16, torch.bfloat16, torch.float32):
                self.promoted_requests[str(function) + ':' + str(dtype)] += 1
                kwargs['dtype'] = torch.float64
            def tensors(value):
                if isinstance(value, torch.Tensor): yield value
                elif isinstance(value, (tuple, list)):
                    for part in value: yield from tensors(part)
                elif isinstance(value, dict):
                    for part in value.values(): yield from tensors(part)
            for value in tensors((args, kwargs)):
                require(not value.is_floating_point() or value.dtype == torch.float64,
                        'Narrow floating reference input: ' + str(function) + ':' + str(value.dtype))
            result = function(*args, **kwargs)
            for value in tensors(result):
                require(not value.is_floating_point() or value.dtype == torch.float64,
                        'Narrow floating reference output: ' + str(function) + ':' + str(value.dtype))
            self.operations[str(function)] += 1
            return result
        def assessment(self):
            require(sum(self.operations.values()) > 0, 'No actual reference operations audited')
            return {'status': 'passed_full_operation_FP64_audit', 'operations': dict(self.operations),
                    'explicit_dtype_requests_promoted': dict(self.promoted_requests),
                    'narrow_floating_operations': 0, 'production_execution_changed': False}
    return FullDouble()


def promote_model(model, digest):
    """Promote a reference copy, proving every original state entry casts back."""
    import torch
    originals = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before = digest(originals)
    model.double()
    rows = []
    after = model.state_dict()
    require(set(after) == set(originals), 'Reference model inventory changed')
    for name, original in originals.items():
        current = after[name]
        require(current.dtype == (torch.float64 if original.is_floating_point() else original.dtype),
                'Reference model promotion missed a state entry')
        require(torch.equal(current.to(original.dtype), original), 'Model promotion is not lossless')
        rows.append({'name': name, 'original_dtype': str(original.dtype),
                     'reference_dtype': str(current.dtype), 'castback_exact': True})
    require(digest({k: after[k].to(v.dtype) for k, v in originals.items()}) == before,
            'Original model castback digest differs')
    return {'original_digest': before, 'reference_digest': digest(after), 'entries': rows,
            'all_castbacks_exact': True, 'original_checkpoint_file_changed': False}


def promote_context(context):
    import torch
    def promote(value):
        if isinstance(value, torch.Tensor):
            result = value.double() if value.is_floating_point() else value.clone()
            require(torch.equal(result.to(value.dtype), value), 'Context promotion is not lossless')
            return result
        if isinstance(value, dict): return {k: promote(v) for k, v in value.items()}
        if isinstance(value, list): return [promote(v) for v in value]
        if isinstance(value, tuple): return tuple(promote(v) for v in value)
        return value
    return promote(context)


def original_metadata():
    """Authenticate the policy engine before its single scalar-transport adapter."""
    import importlib.util
    root = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/protenix-endpoint-fix-release-07/source')
    raw = (root / 'source_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == '63db0a002c8aca2a282d50f9e7227d57b69ac6820b8ae0e06287ec95770dd8a3',
            'Precision policy source identity differs')
    m = json.loads(raw); path = root / 'scripts/protenix_precision_admission.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == m['files']['scripts/protenix_precision_admission.py'],
            'Precision policy engine changed')
    spec = importlib.util.spec_from_file_location('_dbmol_separate_precision_policy', path)
    p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    scalar = p._scalar
    def observed_scalar(value, dtype):
        # DBMol already assembles its FP32 native heads in FP64. Preserve its
        # actual scalar bytes; the engine still requires FP32 production raw
        # gradients and every reference scalar/gradient to be FP64.
        return scalar(value, 'float64' if dtype == 'float32' else dtype)
    p._scalar = observed_scalar
    return p


def worker(args):
    require(str(os.environ.get('LSB_JOBID', '')).isdigit() and os.environ.get('LSB_QUEUE') == 'gpul40s',
            'Native reference requires its real allocated GPU job')
    import torch
    import numpy as np
    from mosaic.small_molecule.boltz2 import native
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and 'L40S' in torch.cuda.get_device_name(), 'One exclusive L40S required')
    require(native.file_digest(args.context) == args.context_sha256, 'Original context file changed')
    context = torch.load(args.context, map_location='cuda', weights_only=True)
    native.validate_context(context)
    with contextlib.redirect_stdout(sys.stderr):
        model = native.load_model(args.checkpoint).to('cuda').eval().requires_grad_(False)
    require(native.tensor_digest(model.state_dict()) == args.model_digest, 'Original affinity model differs')
    promoted = promote_model(model, native.tensor_digest); context = promote_context(context)
    context_values = context['features'] | {'coordinates': context['coordinates']}
    frozen = {'model': native.tensor_digest(model.state_dict()), 'context': native.tensor_digest(context_values)}
    torch.set_default_dtype(torch.float64)
    mode = precision_mode()
    print(json.dumps({'status': 'ready_full_double_reference', 'promotion': promoted,
                      'frozen': frozen, 'optimizer_updates': 0}), flush=True)
    with mode:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('action') == 'close':
                require(native.tensor_digest(model.state_dict()) == frozen['model']
                        and native.tensor_digest(context_values) == frozen['context'], 'Reference frozen values changed')
                print(json.dumps({'status': 'closed', 'audit': mode.assessment(), 'frozen': frozen}), flush=True)
                return
            values = {f: np.asarray(request[f], dtype=np.float64) for f in ('X', 'E_upper')}
            require(all(v.shape == shape and np.isfinite(v).all() for v, shape in
                        zip(values.values(), ((25, 6), (300, 5)))), 'Malformed reference graph')
            observed = hashlib.sha256(b''.join(values[f].tobytes() for f in ('X', 'E_upper'))).hexdigest()
            require(observed == request['graph_sha256'], 'Reference transport changed FP64 bytes')
            gradient = request['gradient']; require(type(gradient) is bool, 'Explicit gradient flag required')
            point = {f: torch.tensor(values[f], dtype=torch.float64, device='cuda', requires_grad=gradient) for f in values}
            with torch.set_grad_enabled(gradient):
                output = native.score(model, context, point['X'], point['E_upper'])
                loss = output[0] - output[2]
                require(loss.dtype == torch.float64 and bool(torch.isfinite(loss)), 'Nonfinite or narrow affinity reference')
                result = {'id': request['id'], 'value': float(loss.detach().cpu()), 'graph_sha256': observed}
                if gradient:
                    gradients = torch.autograd.grad(loss, tuple(point.values()))
                    require(all(v.dtype == torch.float64 and bool(v.isfinite().all()) for v in gradients),
                            'Invalid reference gradients')
                    result['gradients'] = {f: v.detach().cpu().tolist() for f, v in zip(point, gradients)}
            print(json.dumps(result, allow_nan=False), flush=True)
    raise ValueError('Reference closed without an explicit frozen/audit handshake')


def startup_output_path():
    return Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/dbmol-endpoint-recovery-13/cpu/payload/reference-startup.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', required=True); p.add_argument('--context-sha256', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--model-digest', required=True)
    p.add_argument('--startup-only', action='store_true'); p.add_argument('--startup-output')
    args = p.parse_args()
    if args.startup_only:
        require(str(os.environ.get('LSB_JOBID', '')).isdigit(), 'Actual allocated startup required')
        expected = startup_output_path()
        require(args.startup_output == str(expected), 'Closed startup output scope differs')
        with Path(args.context).open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
        require(digest == args.context_sha256 and Path(args.checkpoint).is_file(), 'Native startup inputs differ')
        with expected.open('x') as f:
            json.dump({'status': 'passed_metadata_only', 'numerical_setup_entered': False,
                       'optimizer_updates': 0, 'job_id': os.environ['LSB_JOBID']}, f)
        return
    require(args.startup_output is None, 'Unexpected startup output during native reference')
    worker(args)


if __name__ == '__main__': main()
