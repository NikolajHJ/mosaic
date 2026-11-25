"""Replay both repair algorithms at identical saved native-trajectory latents.

Allocated CPU diagnostic only: no native scorer, optimizer update or old-run write.
Sampling is stratified by recorded search status; coverage is not a population rate.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from latentmol.design.exchange_protenix import read_rows
from latentmol.design.latent_protenix import LatentDesignConfig, _json, _model_identity, structural_target
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.exchange import ExchangeConfig, StructureReference, exchange_guidance, preservation_loss
from latentmol.repair_guidance.loss import fixed_target_nll


def sampled_steps(metrics, per_status):
    selected=[]
    for status in ('already_valid','witness_found','censored'):
        ids=[r['step'] for r in metrics[:-1] if r['score']['repair_guidance']['search']['status']==status]
        if ids:
            selected.extend(ids[i] for i in np.unique(np.linspace(0,len(ids)-1,min(len(ids),per_status)).astype(int)))
    return sorted(selected)


def cosine(a,b):
    scale=float(a.norm()*b.norm())
    return float((a*b).sum())/scale if scale else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cohort','checkpoint','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--per-status',type=int,default=12)
    args=parser.parse_args()
    if not os.environ.get('LSB_JOBID','').isdigit():raise RuntimeError('Use allocated CPU resources')
    if args.per_status<1 or args.per_status>32:raise ValueError('Bound the audit sample')
    metadata=json.loads((args.cohort/'run.json').read_text());config=metadata['config']
    if file_sha256(args.checkpoint)!=config['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    torch.set_num_threads(4)
    model,_=load_model(args.checkpoint,'cpu');model.eval().requires_grad_(False)
    before=_model_identity(model)
    state=torch.load(args.cohort/'start.pt',map_location='cpu',weights_only=True)
    active=state['active'];reference=StructureReference.create(tuple(state['exchange_reference'][k] for k in ('atoms','charges','bonds')),model.spec)
    design=LatentDesignConfig(**config['design'])
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[];started=time.monotonic()
    for arm in config['arms']:
        records=args.cohort/arm/'arms'/'latent';metrics=read_rows(records/'metrics.jsonl')
        selected=sampled_steps(metrics,args.per_status);saved={}
        for path in sorted((records/'raw-blocks').glob('states-*.npz')):
            with np.load(path) as block:
                for index,step in enumerate(block['steps']):
                    if int(step) in selected:
                        bonds=np.zeros((model.spec.max_nodes,model.spec.max_nodes),dtype=np.int64)
                        bonds[block['upper_left'],block['upper_right']]=block['bonds_upper'][index]
                        bonds+=bonds.T
                        saved[int(step)]=(block['latent'][index].copy(),(block['atoms'][index].copy(),block['charges'][index].copy(),bonds))
        if set(saved)!=set(selected):raise ValueError('Missing selected saved states')
        for step in selected:
            latent,old_labels=saved[step]
            point=torch.from_numpy(latent)[None].requires_grad_(True)
            decoded=model.decode(point);graph=graph_from_decoder(decoded,model.spec,temperature=1.,fixed_active=active)
            raw=tuple(getattr(graph,k)[0].argmax(-1).detach().numpy() for k in ('atoms','charges','bonds'))
            same=all(np.array_equal(x,y) for x,y in zip(raw,old_labels))
            def gradient(loss):
                g,=torch.autograd.grad(loss,point,retain_graph=True)
                if not bool(loss.isfinite() and g.isfinite().all()):raise ValueError('Nonfinite audit path')
                return g
            target,current=structural_target(graph,model.spec,active,design,step)
            current_loss=sum(getattr(decoded,k).sum()*0 for k in ('atoms','charges','bonds'))
            if target is not None:
                current_loss,_=fixed_target_nll(decoded,graph,target,model.spec,temperature=1.,normalization='mean_per_decision')
                if not torch.equal(target.graph.atoms,graph.atoms.argmax(-1).cpu()):raise ValueError('Current repair exchanged atoms')
            current_gradient=gradient(current_loss)
            compared={}
            for name,settings in config['arms'].items():
                if settings is None:continue
                cfg=ExchangeConfig(**settings)
                loss,detail,mixed_target,report=exchange_guidance(decoded,graph,model.spec,reference,cfg,temperature=1.,step=step)
                anchor,terms=preservation_loss(graph,reference,cfg)
                anchor_gradient=gradient(anchor);repair_gradient=gradient(loss-anchor)
                witness=report['weighted_witness']
                compared[name]=dict(status=report['status'],checks=report['checks'],
                    edits=None if witness is None else witness['edit_counts'],
                    total_loss=float(loss.detach()),repair_loss=float((loss-anchor).detach()),**terms,
                    repair_gradient_norm=float(repair_gradient.norm()),anchor_gradient_norm=float(anchor_gradient.norm()),
                    repair_anchor_cosine=cosine(repair_gradient,anchor_gradient),
                    current_repair_anchor_cosine=cosine(current_gradient,anchor_gradient))
            row=dict(source_arm=arm,step=step,recorded_status=metrics[step]['score']['repair_guidance']['search']['status'],
                saved_hard_labels_match=same,current=current,current_loss=float(current_loss.detach()),
                current_gradient_norm=float(current_gradient.norm()),mixed=compared)
            rows.append(row)
            with (args.output/'cases.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(arm=arm,selected=len(selected),elapsed_seconds=time.monotonic()-started)),flush=True)
    if _model_identity(model)!=before or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen decoder changed')
    summaries={}
    for arm in config['arms']:
        panel=[r for r in rows if r['source_arm']==arm and r['saved_hard_labels_match']]
        strata={}
        for status in ('already_valid','witness_found','censored'):
            group=[r for r in panel if r['recorded_status']==status]
            if not group:continue
            strata[status]=dict(cases=len(group),current_witness=sum(r['current']['status']=='witness_found' for r in group),
                current_censored=sum(r['current']['status']=='censored' for r in group),
                current_more_than_two_edits=sum((r['current']['edit_count'] or 0)>2 for r in group),
                mixed={name:dict(witness=sum(r['mixed'][name]['status']=='witness_found' for r in group),
                    censored=sum(r['mixed'][name]['status']=='censored' for r in group)) for name in config['arms'] if config['arms'][name] is not None})
        summaries[arm]=strata
    _json(args.output/'result.json',dict(status='completed',cohort=str(args.cohort),cases=len(rows),
        source_config=config,checkpoint_sha256=config['checkpoint_sha256'],frozen_decoder_unchanged=True,
        saved_label_mismatches=sum(not r['saved_hard_labels_match'] for r in rows),
        elapsed_seconds=time.monotonic()-started,per_status=args.per_status,summary=summaries,
        scope='Stratified same-latent decoder/repair replay; no native binding or optimization calls'))


if __name__=='__main__':main()
