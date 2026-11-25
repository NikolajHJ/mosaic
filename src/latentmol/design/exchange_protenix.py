"""Matched native Protenix optimization with opt-in exchange guidance.

Reuse run_design for every update and its frozen-state checks. New output only;
no continuation or changes to an existing campaign. Hard readouts retain the
same native context and are not fresh geometry/affinity qualification.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import LatentDesignConfig, _json, choose_active, run_design
from latentmol.design.temperature_study import hard_at
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.exchange import StructureReference, structure_distance
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer


def arm_design(config, arm, seed):
    """Keep legacy mixed experiments intact; opt in separately to pure penalties."""
    mode=config.get('guidance_mode','mixed_exchange')
    if mode not in ('mixed_exchange','current_search_preservation'):
        raise ValueError('Unknown comparison guidance mode')
    key='preservation_guidance' if mode=='current_search_preservation' else 'exchange_guidance'
    return replace(LatentDesignConfig(**config['design']),noise_seed=seed,**{key:config['arms'][arm]})


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def choose_start(model, config, seed):
    if seed not in config['seeds']:
        raise ValueError('Seed is outside this comparison')
    design=LatentDesignConfig(**config['design'])
    for candidate_seed in range(seed,seed+config['screen_limit']):
        with torch.no_grad():
            point=torch.randn((1,model.config.latent),generator=torch.Generator().manual_seed(candidate_seed))
            decoded=model.decode(point)
            active=choose_active(decoded,model.spec,design.occupied_atoms)
            graph=graph_from_decoder(decoded,model.spec,temperature=design.temperature_initial,fixed_active=active)
            raw=tuple(getattr(graph,k)[0].argmax(-1).cpu().numpy() for k in ('atoms','charges','bonds'))
        inspected=inspect_graph(*raw,model.spec)
        if inspected.smiles is None:continue
        try:reference=StructureReference.create(raw,model.spec)
        except ValueError:continue
        state=dict(latent=point,active=active,step=0,kind='screened_valid_prior',
            provenance=dict(screen_seed=seed,selected_seed=candidate_seed,smiles=inspected.smiles),
            exchange_reference={k:getattr(reference.target.graph,k)[0].clone() for k in ('atoms','charges','bonds')})
        return state,reference
    raise ValueError('No valid canonical reference in bounded screening budget')


class ObservedNativeScorer:
    """Observe probabilities without changing the native differentiable path."""
    def __init__(self,native,folder,reference):
        self.native=native;self.folder=Path(folder);self.reference=reference
        self.identity=native.identity;self.rows=[];self.closed=False
        self.label='Protenix native contact + chemistry comparison'

    def __call__(self,graph):
        value,aux=self.native(graph)
        raw=tuple(getattr(graph,k)[0].detach().argmax(-1).cpu().numpy() for k in ('atoms','charges','bonds'))
        sulfur=self.native.spec.atomic_numbers.index(16)
        self.rows.append(dict(step=len(self.rows),hard_sulfur=int(np.count_nonzero(raw[0]==sulfur)),
            expected_sulfur=float(graph.atoms[0,:,sulfur].detach().sum()),
            atom_exchanges=int(np.count_nonzero(raw[0]!=self.reference.labels()[0])),
            substructure_changes=structure_distance(raw,self.reference),
            hard_elements={str(z):int(np.count_nonzero(raw[0]==i)) for i,z in enumerate(self.native.spec.atomic_numbers) if i}))
        return value,aux

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            _json(self.folder/'chemistry-observations.json',self.rows)
            records=self.folder/'arms'/'latent'
            if not (records/'candidates.jsonl').exists():return
            candidates=read_rows(records/'candidates.jsonl');metrics=read_rows(records/'metrics.jsonl')
            if not candidates:return
            by_step={r['step']:r for r in candidates};scores={r['step']:r['score']['binding_loss'] for r in metrics}
            valid=[r for r in candidates if r['valid']]
            requested=[('initial',candidates[0]),('endpoint',candidates[-1])]
            if valid:requested.append(('best_relaxed_valid',min(valid,key=lambda r:scores[r['step']])))
            readouts=[];cache={}
            for role,row in requested:
                step=row['step']
                result=dict(role=role,step=step,valid=row['valid'],smiles=row['smiles'],loss=None,
                    scope='hard graph in the same geometry-off native context; no fresh conformer')
                if row['valid']:
                    # Cache by exact step; canonical SMILES can hide slot permutations.
                    if step not in cache:
                        hard=hard_at(records/'raw-blocks',step)
                        with torch.no_grad():value,aux=self.native(hard.to_relaxed(self.native.spec))
                        cache[step]=dict(loss=float(value),aux=aux)
                    result.update(cache[step])
                readouts.append(result)
            _json(self.folder/'hard-readouts.json',readouts)
        finally:
            self.native.close()  # Native weight/context audit before success.


def cohort_summary(output, config):
    results={}
    for arm in config['arms']:
        folder=output/arm
        complete=json.loads((folder/'result.json').read_text())
        records=folder/'arms'/'latent'
        candidates=read_rows(records/'candidates.jsonl');metrics=read_rows(records/'metrics.jsonl')
        updates=read_rows(records/'updates.jsonl')
        observations=json.loads((folder/'chemistry-observations.json').read_text())
        if not (complete['completed_step']==config['design']['updates'] and len(candidates)==len(metrics)==len(observations)==config['design']['updates']+1):
            raise ValueError('Incomplete comparison arm')
        valid=[r for r in candidates if r['valid']]
        losses=[r['score']['binding_loss'] for r in metrics]
        sulfur=[r['hard_sulfur'] for r in observations]
        results[arm]=dict(initial_relaxed_loss=losses[0],final_relaxed_loss=losses[-1],best_relaxed_loss=min(losses),
            final_valid=candidates[-1]['valid'],final_smiles=candidates[-1]['smiles'],
            valid_sources=sum(r['valid'] for r in candidates[:-1]),source_observations=len(updates),
            unique_valid_trajectory=len({r['smiles'] for r in valid}),
            initial_sulfur=sulfur[0],final_sulfur=sulfur[-1],max_sulfur=max(sulfur),
            initial_expected_sulfur=observations[0]['expected_sulfur'],final_expected_sulfur=observations[-1]['expected_sulfur'],
            max_expected_sulfur=max(r['expected_sulfur'] for r in observations),
            final_atom_exchanges=observations[-1]['atom_exchanges'],final_substructure_changes=observations[-1]['substructure_changes'],
            bounded_updates=sum(r['bound_factor']<1 for r in updates),
            mean_bound_factor=float(np.mean([r['bound_factor'] for r in updates])),
            native_updates=len(updates),hard_readouts=json.loads((folder/'hard-readouts.json').read_text()),
            technical_checks=complete['technical_checks'],trajectory_statistics=complete['trajectory_statistics'])
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','mosaic-config','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mosaic-python',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--plan');parser.add_argument('--plan-sha256')
    args=parser.parse_args(argv)
    if not os.environ.get('LSB_JOBID','').isdigit():raise RuntimeError('Use an allocated GPU job')
    config=json.loads(args.config.read_text());native_config=json.loads(args.mosaic_config.read_text())
    if file_sha256(args.checkpoint)!=config['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    torch.set_num_threads(4)
    model,_=load_model(args.checkpoint,'cpu');model.eval().requires_grad_(False)
    state,reference=choose_start(model,config,args.seed)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.save(state,args.output/'start.pt')
    _json(args.output/'run.json',dict(config=config,seed=args.seed,start=state['provenance'],
        reference_sha256=reference.target.identity,motifs=reference.motifs,job_id=os.environ['LSB_JOBID']))
    for arm,exchange in config['arms'].items():
        folder=args.output/arm
        design=arm_design(config,arm,args.seed)
        def factory(spec,active):
            native=MosaicProtenixScorer(spec,active,command=[args.mosaic_python,'-m','latentmol.guidance.mosaic_protenix_worker'],
                config=native_config,stderr_path=folder/'native.stderr.log',timeout=900,startup_timeout=900)
            return ObservedNativeScorer(native,folder,reference)
        run_design(model,factory,design,folder,checkpoint_sha256=config['checkpoint_sha256'],
            start_state=state,native_metadata=dict(objective='native Protenix contact, not affinity',seed=args.seed))
    results=cohort_summary(args.output,config)
    _json(args.output/'comparison.json',dict(status='completed',config=config,seed=args.seed,
        start=state['provenance'],arms=results))


if __name__=='__main__':main()
