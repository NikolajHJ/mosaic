"""Allocated, exact-state diagnosis of source105's new trajectory failures.

This observes the SAME frozen functions, point, occupied slots, native context
and stopped repair target. All additive loss primitives are measured on the
same failed directions and endpoints. It neither approves a failed guard nor
changes the scorer, objective, optimizer, target or model parameters.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys

EPSILONS = (1e-5, 3e-5, 1e-4, 3e-4, .001, .003)
CASE_SECONDS = 3600
_HELPER = None


def helper():
    """Resolve new observer bytes before the authenticated old-source bootstrap."""
    global _HELPER
    if _HELPER is None:
        path = Path(__file__).with_name('failed_gate_attribution.py')
        spec = importlib.util.spec_from_file_location('_trajectory_saved_state_helper', path)
        _HELPER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_HELPER)
        _HELPER.CASE_TIMEOUT_SECONDS = CASE_SECONDS
        _HELPER.CLAIM_LIMIT = __doc__
    return _HELPER


class ReluObserver:
    """Observe actual preactivations; do not alter any module output or gradient."""

    def __init__(self, decoder, scorer):
        from torch import nn
        self.layers = {root + '.' + name: module
            for root, model in (('decoder', decoder), ('scorer', scorer))
            for name, module in model.named_modules() if isinstance(module, nn.ReLU)}
        self.baseline = {}

    @contextmanager
    def observe(self, *, baseline=False):
        import torch
        counts, rows, handles = {}, {}, []
        def hook(name):
            def forward(module, args, output):
                index = counts.get(name, 0)
                counts[name] = index + 1
                key = name + ':' + str(index)
                value = args[0].detach()
                if baseline:
                    self.baseline[key] = value.clone()
                old = self.baseline[key]
                if old.shape != value.shape:
                    raise ValueError('Observed ReLU invocation topology changed')
                crossed = (old > 0) != (value > 0)
                rows[key] = {'units': value.numel(), 'crossings': int(crossed.sum()),
                    'baseline_minimum_absolute': float(old.abs().min()),
                    'minimum_absolute': float(value.abs().min()),
                    'maximum_delta': float((value - old).abs().max())}
            return forward
        try:
            handles = [module.register_forward_hook(hook(name)) for name, module in self.layers.items()]
            yield rows
            if set(rows) != set(self.baseline):
                raise ValueError('Incomplete actual ReLU invocation coverage')
        finally:
            for handle in handles:
                handle.remove()


def scalars(decoder, point, branch, config, counters, target, weight):
    """Keep live primitive scalars and dual tensors; use exact production grouping."""
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.engine import scalar_loss
    from latentmol.repair_guidance.loss import fixed_target_nll
    decoded = decoder.decode(point)
    graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                               fixed_active=branch.prepared.active)
    outputs = counters.call('boltz', 'trajectory_mechanism', 'forward', lambda: branch(graph), point.device)
    total, _ = scalar_loss(outputs, config.objective)
    result = dict(outputs)
    if target is not None:
        repair, _ = fixed_target_nll(decoded, graph, target, decoder.spec,
            temperature=config.temperature, normalization=config.repair_guidance.normalization)
        result['repair_nll'] = repair
        total = total + weight * repair
    result['combined'] = total
    return graph, result


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item,
               *, persist, report):
    """Replay declared original directions; record components, kinks and local AD."""
    import torch
    from torch.autograd import forward_ad
    from latentmol.boltz.active_set import score_at
    from latentmol.boltz.branch_local_fd import AffinityReadoutBranches
    from latentmol.multimodel.study import tensor_digest
    h = helper()
    components = {}
    measured, gradient = score_at(decoder, point, branch, config, counters, 'trajectory_baseline',
        gradients=True, component_gradients=components, repair_target=target, repair_weight=weight)
    gate = h._read(h.authenticated_artifact(_CASE, Path(_CASE['parent_run']) / _CASE['arm'] / 'gate-0015.json'))
    if measured['loss'] != gate['measured']['loss'] or measured['components'] != gate['measured']['components']:
        raise ValueError('Saved failing baseline did not replay exactly')
    relevant = gradient if item_name == 'combined' else components[item_name]
    directions = h.reconstruct_directions(point, relevant, item)
    if any(meta['hash_matches_original'] is not True for name, (_, meta) in directions.items() if name != 'gradient'):
        raise ValueError('A saved failed random direction changed')
    branches = AffinityReadoutBranches(branch.scorer)
    observer = ReluObserver(decoder, branch.scorer)
    with torch.no_grad(), branches.capture(), observer.observe(baseline=True) as activations:
        graph, base = scalars(decoder, point, branch, config, counters, target, weight)
    base = {key: float(value) for key, value in base.items()}
    if base['combined'] != measured['loss']:
        raise ValueError('Instrumented scalar differs from the production baseline')
    report.update(status='probing', optimizer_updates=0, new_defog_calls=0,
        diagnostic_completed=False, original_gate_not_reclassified=True,
        baseline=base, baseline_relu=activations,
        baseline_graph_sha256=tensor_digest({key:getattr(graph,key) for key in ('atoms','charges','bonds')}),
        direction_replay={key:value[1] for key,value in directions.items()},
        epsilons=list(EPSILONS), directions={}, endpoint_rows=[])
    persist()
    for label, (direction, replay) in directions.items():
        with torch.no_grad(), forward_ad.dual_level():
            dual = forward_ad.make_dual(point, direction)
            _, outputs = scalars(decoder, dual, branch, config, counters, target, weight)
            jvp = {}
            for key, output in outputs.items():
                primal, tangent = forward_ad.unpack_dual(output)
                if tangent is None or not bool(tangent.isfinite().all()) or float(primal) != base[key]:
                    raise ValueError('Native forward AD primal/tangent missing or changed: ' + key)
                jvp[key] = float(tangent)
        reverse = float((relevant.double()*direction.double()).sum())
        report['directions'][label] = {'jvp':jvp, 'reverse_projection':reverse,
            'independent_ad_error':abs(jvp[item_name] - reverse), 'replay':replay}
        persist()
        for epsilon in EPSILONS:
            # Both namespaces remain separate. Held readout probes reproduce
            # the failed v2 guard; natural probes diagnose actual branch changes.
            for mode in ('natural', 'held_readout'):
                observations = {}
                for sign in (-1,1):
                    candidate = point + sign * epsilon * direction
                    with torch.no_grad(), observer.observe() as relus:
                        if mode == 'held_readout':
                            with branches.hold():
                                g, values = scalars(decoder, candidate, branch, config, counters, target, weight)
                        else:
                            g, values = scalars(decoder, candidate, branch, config, counters, target, weight)
                    observations[str(sign)] = {'values':{key:float(value) for key,value in values.items()},
                        'relu_crossings':{key:value for key,value in relus.items() if value['crossings']},
                        'graph_sha256':tensor_digest({key:getattr(g,key) for key in ('atoms','charges','bonds')}),
                        'contact_selection':dict(branch.scorer.last_contact_selection)}
                plus, minus = observations['1']['values'], observations['-1']['values']
                fd = {key:(plus[key]-minus[key])/(2*epsilon) for key in base}
                report['endpoint_rows'].append({'direction':label, 'epsilon':epsilon, 'mode':mode,
                    'plus':observations['1'], 'minus':observations['-1'], 'secants':fd,
                    'residuals':{key:fd[key]-jvp[key] for key in base}})
                persist()
    report.update(status='completed', diagnostic_completed=True,
        original_gate_not_reclassified=True, old_gate_certified_by_this_diagnostic=False)
    persist()


_CASE = None


def run_study(path, expected, plan):
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    h = helper()
    with ExperimentRun(Path(plan['output_root']), plan['name'],
        {'plan':plan, 'plan_sha256':expected, 'epsilons':EPSILONS,
         'case_seconds':CASE_SECONDS, 'optimizer_updates':0, 'new_defog_calls':0}, Path.cwd()) as run:
        cases=[]
        for case in plan['cases']:
            result=run_logged([sys.executable,str(Path(__file__).resolve()),'--plan',str(path),
                '--sha256',expected,'--child',case['id'],'--child-output',str(run.path/case['id'])],
                run.path/(case['id']+'.log'),timeout_seconds=CASE_SECONDS)
            complete=False
            if result['returncode']==0 and not result['timed_out'] and len(result['run_paths'])==1:
                out=Path(result['run_paths'][0]); evidence=h._read(out/'attribution.json')
                complete=(h._read(out/'run.json')['status']=='completed'
                    and all(evidence.get(key) is True for key in ('diagnostic_completed','frozen_state_unchanged','parent_unchanged')))
            cases.append({'case_id':case['id'],'diagnostic_completed':complete,'execution':result})
            write_json(run.path/'cases.json',cases)
        run.annotate(diagnostic_cases_completed=sum(row['diagnostic_completed'] for row in cases))
        if not all(row['diagnostic_completed'] for row in cases):
            raise RuntimeError('Trajectory mechanism replay incomplete; retain actual child outcomes')
        return run.path


def main():
    global _CASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--child')
    parser.add_argument('--child-output',type=Path)
    args=parser.parse_args(); h=helper()
    if not os.environ.get('LSB_JOBID') or not h._hash(os.environ.get('LATENTMOL_SOURCE_SHA256')):
        raise ValueError('Require an immutable allocated DTU job')
    plan=h.load_plan(args.plan,args.sha256)
    if args.child:
        matches=[case for case in plan['cases'] if case['id']==args.child]
        if len(matches)!=1 or args.child_output is None:
            raise ValueError('Unknown child or missing unique output')
        _CASE=matches[0]
        print(h.run_case(plan,_CASE,args.child_output,expected_policy='branch_local_v2',probe=probe_case),flush=True)
    else:
        print(run_study(args.plan,args.sha256,plan),flush=True)


if __name__=='__main__':
    main()
